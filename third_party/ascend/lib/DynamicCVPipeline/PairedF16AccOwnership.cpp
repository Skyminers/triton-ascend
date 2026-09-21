// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
// Licensed under the MIT license.

#include "ascend/include/DynamicCVPipeline/PairedF16AccOwnership.h"
#include "ascend/include/DynamicCVPipeline/PairedF16PVAccumulate.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::triton;

namespace {
struct OwnershipPlan {
  func::FuncOp function;
  arith::ExtFOp oldMaxExt;
  hivm::CustomOp softmax;
  arith::MulFOp accMul;
  arith::AddFOp accAdd;
  arith::DivFOp normalize;
};

struct SelectedHalf {
  scf::IfOp selector;
  memref::AllocOp thenAlloc;
  memref::AllocOp elseAlloc;
};

struct FixpipePair {
  scf::IfOp selector;
  hivm::FixpipeOp thenFixpipe;
  hivm::FixpipeOp elseFixpipe;
};

struct ScoreBufferPlan {
  tensor::InsertSliceOp lowInsert;
  tensor::InsertSliceOp highInsert;
  SelectedHalf lowReceiver;
  SelectedHalf highReceiver;
  memref::AllocOp lowProducerThen;
  memref::AllocOp lowProducerElse;
  memref::AllocOp highProducerThen;
  memref::AllocOp highProducerElse;
  FixpipePair lowFixpipe;
  FixpipePair highFixpipe;
  annotation::MarkOp lowReceiverThenMark;
  annotation::MarkOp lowReceiverElseMark;
  annotation::MarkOp highReceiverThenMark;
  annotation::MarkOp highReceiverElseMark;
  annotation::MarkOp lowProducerThenMark;
  annotation::MarkOp lowProducerElseMark;
  annotation::MarkOp highProducerThenMark;
  annotation::MarkOp highProducerElseMark;
};

static annotation::MarkOp getTightlyCoupledMark(memref::AllocOp alloc) {
  annotation::MarkOp result;
  for (Operation *user : alloc->getUsers()) {
    auto mark = dyn_cast<annotation::MarkOp>(user);
    if (!mark || !mark->hasAttr("hivm.tightly_coupled_buffer"))
      continue;
    if (result)
      return nullptr;
    result = mark;
  }
  return result;
}

static std::optional<int64_t> getTightlyCoupledId(memref::AllocOp alloc) {
  auto mark = getTightlyCoupledMark(alloc);
  if (!mark)
    return std::nullopt;
  auto attr = mark->getAttrOfType<hivm::HIVMTightlyCoupledBufferAttr>(
      "hivm.tightly_coupled_buffer");
  if (!attr || !attr.getId())
    return std::nullopt;
  return *attr.getId();
}

static FailureOr<SelectedHalf> matchSelectedHalf(Value value) {
  auto result = dyn_cast<OpResult>(value);
  auto selector = result ? dyn_cast<scf::IfOp>(result.getOwner()) : nullptr;
  if (!selector || selector.getNumResults() != 1 ||
      result.getResultNumber() != 0 || selector.getElseRegion().empty())
    return failure();

  auto getAlloc = [](Block *block) -> memref::AllocOp {
    auto yield = dyn_cast<scf::YieldOp>(block->getTerminator());
    if (!yield || yield.getNumOperands() != 1)
      return nullptr;
    auto toTensor =
        yield.getOperand(0).getDefiningOp<bufferization::ToTensorOp>();
    if (!toTensor || !toTensor.getRestrict() || !toTensor.getWritable() ||
        !toTensor->hasOneUse())
      return nullptr;
    auto cast = toTensor.getBuffer().getDefiningOp<memref::MemorySpaceCastOp>();
    if (!cast || !cast->hasOneUse())
      return nullptr;
    auto alloc = cast.getSource().getDefiningOp<memref::AllocOp>();
    if (!alloc || !getTightlyCoupledMark(alloc))
      return nullptr;
    for (Operation &op : *block)
      if (!isa<memref::MemorySpaceCastOp, bufferization::ToTensorOp,
               scf::YieldOp>(op))
        return nullptr;
    return alloc;
  };

  memref::AllocOp thenAlloc = getAlloc(selector.thenBlock());
  memref::AllocOp elseAlloc = getAlloc(selector.elseBlock());
  if (!thenAlloc || !elseAlloc || thenAlloc == elseAlloc ||
      thenAlloc.getType() != elseAlloc.getType())
    return failure();
  return SelectedHalf{selector, thenAlloc, elseAlloc};
}

static hivm::FixpipeOp getOnlyFixpipe(Block *block) {
  hivm::FixpipeOp result;
  for (Operation &op : *block) {
    if (isa<scf::YieldOp>(op))
      continue;
    auto fixpipe = dyn_cast<hivm::FixpipeOp>(op);
    if (!fixpipe || result)
      return nullptr;
    result = fixpipe;
  }
  return result;
}

static bool equivalentFixpipes(hivm::FixpipeOp thenFixpipe,
                               hivm::FixpipeOp elseFixpipe, Value thenDst,
                               Value elseDst) {
  if (thenFixpipe->getAttrs() != elseFixpipe->getAttrs() ||
      thenFixpipe->getNumOperands() != elseFixpipe->getNumOperands() ||
      thenFixpipe->getResultTypes() != elseFixpipe->getResultTypes() ||
      thenFixpipe.getDst() != thenDst || elseFixpipe.getDst() != elseDst)
    return false;
  for (auto [thenOperand, elseOperand] :
       llvm::zip_equal(thenFixpipe->getOperands(),
                       elseFixpipe->getOperands())) {
    if (thenOperand == thenDst && elseOperand == elseDst)
      continue;
    if (thenOperand != elseOperand)
      return false;
  }
  return true;
}

static FailureOr<FixpipePair>
findFixpipePair(ModuleOp module, memref::AllocOp thenAlloc,
                memref::AllocOp elseAlloc) {
  FixpipePair result;
  bool multiple = false;
  module.walk([&](scf::IfOp selector) {
    if (selector.getNumResults() != 0 || selector.getElseRegion().empty())
      return;
    auto thenFixpipe = getOnlyFixpipe(selector.thenBlock());
    auto elseFixpipe = getOnlyFixpipe(selector.elseBlock());
    if (!thenFixpipe || !elseFixpipe ||
        !equivalentFixpipes(thenFixpipe, elseFixpipe, thenAlloc, elseAlloc))
      return;
    if (result.selector) {
      multiple = true;
      return;
    }
    result = FixpipePair{selector, thenFixpipe, elseFixpipe};
  });
  if (!result.selector || multiple)
    return failure();
  return result;
}

static bool coversDenseConcat(tensor::InsertSliceOp low,
                              tensor::InsertSliceOp high) {
  auto fullType = dyn_cast<RankedTensorType>(high.getResult().getType());
  auto halfType = dyn_cast<RankedTensorType>(low.getSource().getType());
  auto hasUnitStrides = [](ArrayRef<int64_t> strides) {
    return llvm::all_of(strides, [](int64_t stride) { return stride == 1; });
  };
  if (!fullType || !halfType || !fullType.hasStaticShape() ||
      !halfType.hasStaticShape() || fullType.getRank() != halfType.getRank() ||
      fullType.getElementType() != halfType.getElementType() ||
      !fullType.getElementType().isF16() ||
      !hasUnitStrides(low.getStaticStrides()) ||
      !hasUnitStrides(high.getStaticStrides()) ||
      low.getStaticSizes() != halfType.getShape() ||
      high.getStaticSizes() != halfType.getShape())
    return false;
  auto lowOffsets = low.getStaticOffsets();
  auto highOffsets = high.getStaticOffsets();
  for (int64_t dim = 0; dim < fullType.getRank(); ++dim) {
    if (dim == 0) {
      if (lowOffsets[dim] != 0 ||
          highOffsets[dim] != halfType.getDimSize(dim) ||
          fullType.getDimSize(dim) != 2 * halfType.getDimSize(dim))
        return false;
      continue;
    }
    if (lowOffsets[dim] != 0 || highOffsets[dim] != 0 ||
        fullType.getDimSize(dim) != halfType.getDimSize(dim))
      return false;
  }
  return true;
}

static bool hasOnlyReceiverUsers(memref::AllocOp alloc,
                                 scf::IfOp selector) {
  annotation::MarkOp mark = getTightlyCoupledMark(alloc);
  unsigned allowed = 0;
  for (Operation *user : alloc->getUsers()) {
    if (user == mark.getOperation()) {
      ++allowed;
      continue;
    }
    auto cast = dyn_cast<memref::MemorySpaceCastOp>(user);
    if (!cast || !selector->isAncestor(cast))
      return false;
    ++allowed;
  }
  return allowed == 2;
}

static bool hasOnlyProducerUsers(memref::AllocOp alloc,
                                 scf::IfOp selector) {
  annotation::MarkOp mark = getTightlyCoupledMark(alloc);
  unsigned allowed = 0;
  for (Operation *user : alloc->getUsers()) {
    if (user == mark.getOperation()) {
      ++allowed;
      continue;
    }
    auto fixpipe = dyn_cast<hivm::FixpipeOp>(user);
    if (!fixpipe || !selector->isAncestor(fixpipe))
      return false;
    ++allowed;
  }
  return allowed == 2;
}

static FailureOr<ScoreBufferPlan> matchScoreBufferPlan(ModuleOp module,
                                                        hivm::CustomOp custom) {
  if (custom.getName() != "__builtin_online_softmax_nz" ||
      custom->getNumOperands() == 0)
    return failure();
  auto highInsert =
      custom->getOperand(0).getDefiningOp<tensor::InsertSliceOp>();
  auto lowInsert = highInsert
                       ? highInsert.getDest().getDefiningOp<tensor::InsertSliceOp>()
                       : nullptr;
  if (!highInsert || !lowInsert || !highInsert->hasOneUse() ||
      !lowInsert->hasOneUse() || !coversDenseConcat(lowInsert, highInsert))
    return failure();

  auto lowReceiver = matchSelectedHalf(lowInsert.getSource());
  auto highReceiver = matchSelectedHalf(highInsert.getSource());
  if (failed(lowReceiver) || failed(highReceiver) ||
      lowReceiver->selector.getCondition() !=
          highReceiver->selector.getCondition())
    return failure();

  SmallVector<memref::AllocOp> receiverAllocs{
      lowReceiver->thenAlloc, lowReceiver->elseAlloc,
      highReceiver->thenAlloc, highReceiver->elseAlloc};
  llvm::DenseSet<Operation *> receiverSet;
  for (memref::AllocOp alloc : receiverAllocs)
    if (!receiverSet.insert(alloc).second)
      return failure();

  DenseMap<int64_t, SmallVector<memref::AllocOp>> allocsById;
  module.walk([&](memref::AllocOp alloc) {
    if (auto id = getTightlyCoupledId(alloc))
      allocsById[*id].push_back(alloc);
  });
  auto findProducer = [&](memref::AllocOp receiver)
      -> FailureOr<memref::AllocOp> {
    auto id = getTightlyCoupledId(receiver);
    if (!id)
      return failure();
    auto found = allocsById.find(*id);
    if (found == allocsById.end() || found->second.size() != 2)
      return failure();
    for (memref::AllocOp alloc : found->second)
      if (alloc != receiver)
        return alloc;
    return failure();
  };

  auto lowProducerThen = findProducer(lowReceiver->thenAlloc);
  auto lowProducerElse = findProducer(lowReceiver->elseAlloc);
  auto highProducerThen = findProducer(highReceiver->thenAlloc);
  auto highProducerElse = findProducer(highReceiver->elseAlloc);
  if (failed(lowProducerThen) || failed(lowProducerElse) ||
      failed(highProducerThen) || failed(highProducerElse))
    return failure();

  auto lowFixpipe =
      findFixpipePair(module, *lowProducerThen, *lowProducerElse);
  auto highFixpipe =
      findFixpipePair(module, *highProducerThen, *highProducerElse);
  if (failed(lowFixpipe) || failed(highFixpipe) ||
      lowFixpipe->selector.getCondition() !=
          highFixpipe->selector.getCondition())
    return failure();

  if (!hasOnlyReceiverUsers(lowReceiver->thenAlloc, lowReceiver->selector) ||
      !hasOnlyReceiverUsers(lowReceiver->elseAlloc, lowReceiver->selector) ||
      !hasOnlyReceiverUsers(highReceiver->thenAlloc, highReceiver->selector) ||
      !hasOnlyReceiverUsers(highReceiver->elseAlloc, highReceiver->selector) ||
      !hasOnlyProducerUsers(*lowProducerThen, lowFixpipe->selector) ||
      !hasOnlyProducerUsers(*lowProducerElse, lowFixpipe->selector) ||
      !hasOnlyProducerUsers(*highProducerThen, highFixpipe->selector) ||
      !hasOnlyProducerUsers(*highProducerElse, highFixpipe->selector))
    return failure();

  auto sameHalfType = [&](memref::AllocOp alloc) {
    return alloc.getType() == lowReceiver->thenAlloc.getType();
  };
  for (memref::AllocOp alloc : receiverAllocs)
    if (!sameHalfType(alloc))
      return failure();
  for (memref::AllocOp alloc : {*lowProducerThen, *lowProducerElse,
                                *highProducerThen, *highProducerElse})
    if (!sameHalfType(alloc))
      return failure();

  return ScoreBufferPlan{
      lowInsert,
      highInsert,
      *lowReceiver,
      *highReceiver,
      *lowProducerThen,
      *lowProducerElse,
      *highProducerThen,
      *highProducerElse,
      *lowFixpipe,
      *highFixpipe,
      getTightlyCoupledMark(lowReceiver->thenAlloc),
      getTightlyCoupledMark(lowReceiver->elseAlloc),
      getTightlyCoupledMark(highReceiver->thenAlloc),
      getTightlyCoupledMark(highReceiver->elseAlloc),
      getTightlyCoupledMark(*lowProducerThen),
      getTightlyCoupledMark(*lowProducerElse),
      getTightlyCoupledMark(*highProducerThen),
      getTightlyCoupledMark(*highProducerElse)};
}

static bool tensorIs(Value value, ArrayRef<int64_t> shape, Type elementType) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  return type && type.getShape() == shape &&
         type.getElementType() == elementType && !type.getEncoding();
}

static bool consumesPFull(linalg::MatmulOp matmul, Value p) {
  auto convert = matmul.getDpsInputs()[0].getDefiningOp<hivm::ConvertLayoutOp>();
  return convert && convert.getSource() == p &&
         tensorIs(convert.getResult(), {128, 512},
                  Float8E4M3FNType::get(matmul.getContext()));
}

static bool hasLoopCarriedState(func::FuncOp func) {
  bool hasState = false;
  func.walk([&](hivm::CustomOp op) {
    if (op.getName() != "__builtin_online_softmax_nz")
      return;
    auto loop = dyn_cast<scf::ForOp>(op->getParentOp());
    hasState |= loop &&
                (loop.getNumRegionIterArgs() != 0 || loop.getNumResults() != 0);
  });
  return hasState;
}

static FailureOr<OwnershipPlan> matchOwnership(func::FuncOp func) {
  hivm::CustomOp softmax;
  bool multiple = false;
  func.walk([&](hivm::CustomOp op) {
    if (op.getName() != "__builtin_online_softmax_nz")
      return;
    if (softmax)
      multiple = true;
    softmax = op;
  });
  if (!softmax || multiple)
    return failure();

  auto *ctx = softmax.getContext();
  if (softmax->getNumOperands() != 6 || softmax->getNumResults() != 3 ||
      !tensorIs(softmax->getOperand(0), {32, 8, 16, 16},
                Float16Type::get(ctx)) ||
      !tensorIs(softmax->getOperand(1), {128}, Float16Type::get(ctx)) ||
      !tensorIs(softmax->getOperand(2), {256}, IntegerType::get(ctx, 8)) ||
      !tensorIs(softmax->getResult(0), {16, 8, 16, 32},
                Float8E4M3FNType::get(ctx)) ||
      !tensorIs(softmax->getResult(1), {128}, Float16Type::get(ctx)) ||
      !tensorIs(softmax->getResult(2), {128}, Float32Type::get(ctx)))
    return failure();
  for (unsigned i = 0; i < 3; ++i)
    if (softmax->getOperand(i + 3).getType() !=
        softmax->getResult(i).getType())
      return failure();

  auto loop = dyn_cast<scf::ForOp>(softmax->getParentOp());
  if (!loop || loop.getNumRegionIterArgs() != 3 || loop.getNumResults() != 3)
    return failure();
  auto args = loop.getRegionIterArgs();
  if (!tensorIs(args[0], {128}, Float16Type::get(ctx)) ||
      !tensorIs(args[1], {128}, Float32Type::get(ctx)) ||
      !tensorIs(args[2], {128, 128}, Float32Type::get(ctx)) ||
      !args[2].hasOneUse())
    return failure();

  auto yield = dyn_cast<scf::YieldOp>(loop.getBody()->getTerminator());
  if (!yield || yield.getNumOperands() != 3)
    return failure();

  auto lAdd = yield.getOperand(1).getDefiningOp<arith::AddFOp>();
  if (!lAdd || lAdd->getBlock() != loop.getBody() ||
      lAdd.getRhs() != softmax->getResult(2) ||
      !lAdd.getResult().hasOneUse())
    return failure();
  auto lMul = lAdd.getLhs().getDefiningOp<arith::MulFOp>();
  if (!lMul || lMul->getBlock() != loop.getBody() ||
      lMul.getLhs() != args[1] || !lMul.getResult().hasOneUse())
    return failure();
  auto alpha = lMul.getRhs().getDefiningOp<math::ExpOp>();
  if (!alpha || alpha->getBlock() != loop.getBody())
    return failure();
  auto delta = alpha.getOperand().getDefiningOp<arith::SubFOp>();
  if (!delta || delta->getBlock() != loop.getBody() ||
      !delta.getResult().hasOneUse())
    return failure();
  auto oldMaxExt = delta.getLhs().getDefiningOp<arith::ExtFOp>();
  auto nextMaxExt = delta.getRhs().getDefiningOp<arith::ExtFOp>();
  if (!oldMaxExt || !nextMaxExt ||
      oldMaxExt->getBlock() != loop.getBody() ||
      nextMaxExt->getBlock() != loop.getBody() ||
      oldMaxExt.getIn() != args[0] ||
      nextMaxExt.getIn() != softmax->getResult(1) ||
      !oldMaxExt.getResult().hasOneUse() ||
      !nextMaxExt.getResult().hasOneUse())
    return failure();

  auto accAdd = yield.getOperand(2).getDefiningOp<arith::AddFOp>();
  if (!accAdd || accAdd->getBlock() != loop.getBody() ||
      !tensorIs(accAdd.getResult(), {128, 128}, Float32Type::get(ctx)) ||
      !accAdd.getResult().hasOneUse())
    return failure();
  auto accMul = accAdd.getLhs().getDefiningOp<arith::MulFOp>();
  if (!accMul || accMul->getBlock() != loop.getBody() ||
      accMul.getLhs() != args[2] ||
      !tensorIs(accMul.getResult(), {128, 128}, Float32Type::get(ctx)) ||
      !accMul.getResult().hasOneUse())
    return failure();
  auto alphaBroadcast = accMul.getRhs().getDefiningOp<linalg::BroadcastOp>();
  if (!alphaBroadcast || alphaBroadcast->getBlock() != loop.getBody() ||
      alphaBroadcast.getDpsInputs().size() != 1 ||
      alphaBroadcast.getDpsInputs()[0] != alpha.getResult() ||
      !alphaBroadcast->getResult(0).hasOneUse())
    return failure();
  for (Operation *user : alpha.getResult().getUsers())
    if (user != lMul.getOperation() && user != alphaBroadcast.getOperation())
      return failure();

  // The other addend must be the scaled result of the full-width K512 PV.
  auto pvScale = accAdd.getRhs().getDefiningOp<arith::MulFOp>();
  if (!pvScale || pvScale->getBlock() != loop.getBody() ||
      !tensorIs(pvScale.getResult(), {128, 128}, Float32Type::get(ctx)) ||
      !pvScale.getResult().hasOneUse())
    return failure();
  auto pv1 = pvScale.getLhs().getDefiningOp<linalg::MatmulOp>();
  if (!pv1)
    return failure();
  if (pv1->getBlock() != loop.getBody() ||
      !tensorIs(pv1.getResult(0), {128, 128}, Float32Type::get(ctx)))
    return failure();
  if (!consumesPFull(pv1, softmax->getResult(0)))
    return failure();

  arith::DivFOp normalize;
  for (Operation *user : loop.getResult(2).getUsers()) {
    auto div = dyn_cast<arith::DivFOp>(user);
    if (!div || div.getLhs() != loop.getResult(2) || normalize)
      return failure();
    normalize = div;
  }
  if (!normalize || !normalize.getResult().hasOneUse() ||
      !isa<bufferization::MaterializeInDestinationOp>(
          *normalize.getResult().getUsers().begin()))
    return failure();

  auto hasDestination = [](Operation *op, int64_t expected) {
    Attribute attr = op->getAttr(kDestinationOperand);
    if (!attr)
      return true;
    auto index = dyn_cast<IntegerAttr>(attr);
    return index && index.getInt() == expected;
  };
  if (!hasDestination(accMul, 0) || !hasDestination(accAdd, 0) ||
      !hasDestination(normalize, 0))
    return failure();
  return OwnershipPlan{func, oldMaxExt, softmax, accMul, accAdd, normalize};
}

class PairedF16AccOwnershipPass final
    : public PassWrapper<PairedF16AccOwnershipPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PairedF16AccOwnershipPass)
  StringRef getArgument() const final { return "paired-f16-acc-ownership"; }
  StringRef getDescription() const final {
    return "Prove paired FP16 accumulator destinations before scheduling";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, bufferization::BufferizationDialect,
                    func::FuncDialect, hivm::HIVMDialect,
                    linalg::LinalgDialect, scf::SCFDialect>();
  }
  void runOnOperation() override {
    SmallVector<OwnershipPlan> plans;
    for (auto func : getOperation().getOps<func::FuncOp>()) {
      if (!func->hasAttr(kPairedF16PVAccumulate))
        continue;
      if (!func->getAttrOfType<UnitAttr>(kPairedF16PVAccumulate)) {
        func.emitError("paired FP16 accumulator ownership requires a unit PV opt-in");
        return signalPassFailure();
      }
      auto plan = matchOwnership(func);
      if (failed(plan)) {
        // A single KV512 tile is fully unrolled and has no loop-carried state.
        // It still needs PV task formation, but there is no recurrence ownership
        // to prove or annotate.
        if (!hasLoopCarriedState(func))
          continue;
        func.emitError("paired FP16 accumulator ownership requires an exclusive "
                       "loop-carried acc recurrence and terminal normalize");
        return signalPassFailure();
      }
      plans.push_back(*plan);
    }
    Builder builder(&getContext());
    auto operandZero = builder.getI64IntegerAttr(0);
    for (const OwnershipPlan &plan : plans) {
      // Keep the old running maximum live only before the softmax update.  The
      // cast is independent of the custom op, and moving it makes the true SSA
      // order explicit to the double-buffer scheduler instead of looking like
      // an update-before-use recurrence across vector blocks.
      if (plan.softmax->isBeforeInBlock(plan.oldMaxExt))
        plan.oldMaxExt->moveBefore(plan.softmax);
      plan.function->setAttr(kLoopCarriedReadBeforeUpdate, operandZero);
      // Reuse the old accumulator in place. DynamicCV adds one mutually
      // exclusive else path that forwards the old value unchanged; the
      // conversion validates that exact structure before honoring this opt-in.
      plan.accMul->setAttr(kDestinationOperand, operandZero);
      plan.accMul->setAttr(kDestinationOppositeIfForward,
                           builder.getUnitAttr());
      plan.accAdd->setAttr(kDestinationOperand, operandZero);
      plan.normalize->setAttr(kDestinationOperand, operandZero);
    }
    for (auto func : getOperation().getOps<func::FuncOp>())
      func->removeAttr(kPairedF16PVAccumulate);
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createPairedF16AccOwnershipPass() {
  return std::make_unique<PairedF16AccOwnershipPass>();
}

LogicalResult mlir::triton::coalescePairedF16ScoreBuffers(ModuleOp module) {
  SmallVector<hivm::CustomOp> candidates;
  module.walk([&](hivm::CustomOp custom) {
    if (custom.getName() != "__builtin_online_softmax_nz" ||
        custom->getNumOperands() == 0)
      return;
    if (custom->getOperand(0).getDefiningOp<tensor::InsertSliceOp>())
      candidates.push_back(custom);
  });
  if (candidates.empty())
    return success();
  if (candidates.size() != 1)
    return failure();

  auto matched = matchScoreBufferPlan(module, candidates.front());
  if (failed(matched))
    return failure();
  ScoreBufferPlan &plan = *matched;
  IRRewriter rewriter(module.getContext());
  auto fullTensorType = cast<RankedTensorType>(plan.highInsert.getType());
  auto halfType = plan.lowReceiver.thenAlloc.getType();
  auto fullType = MemRefType::get(fullTensorType.getShape(),
                                  fullTensorType.getElementType(),
                                  MemRefLayoutAttrInterface{},
                                  halfType.getMemorySpace());

  auto createFullPair = [&](memref::AllocOp thenHalf,
                            memref::AllocOp elseHalf,
                            annotation::MarkOp thenMark,
                            annotation::MarkOp elseMark) {
    Operation *first = thenHalf->isBeforeInBlock(elseHalf)
                           ? thenHalf.getOperation()
                           : elseHalf.getOperation();
    rewriter.setInsertionPoint(first);
    auto thenFull = rewriter.create<memref::AllocOp>(first->getLoc(), fullType);
    thenFull->setAttrs(thenHalf->getAttrs());
    auto thenFullMark = rewriter.create<annotation::MarkOp>(
        thenMark.getLoc(), thenFull.getResult());
    thenFullMark->setAttrs(thenMark->getAttrs());
    auto elseFull = rewriter.create<memref::AllocOp>(first->getLoc(), fullType);
    elseFull->setAttrs(elseHalf->getAttrs());
    auto elseFullMark = rewriter.create<annotation::MarkOp>(
        elseMark.getLoc(), elseFull.getResult());
    elseFullMark->setAttrs(elseMark->getAttrs());
    return std::pair<memref::AllocOp, memref::AllocOp>{thenFull, elseFull};
  };

  auto receiverFull = createFullPair(
      plan.lowReceiver.thenAlloc, plan.lowReceiver.elseAlloc,
      plan.lowReceiverThenMark, plan.lowReceiverElseMark);
  auto producerFull = createFullPair(
      plan.lowProducerThen, plan.lowProducerElse, plan.lowProducerThenMark,
      plan.lowProducerElseMark);

  auto createSelectedRoot = [&](Location loc, Value condition,
                                memref::AllocOp thenFull,
                                memref::AllocOp elseFull) {
    auto selector =
        rewriter.create<scf::IfOp>(loc, TypeRange{fullType}, condition, true);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(selector.thenBlock());
      rewriter.create<scf::YieldOp>(loc, thenFull.getResult());
    }
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(selector.elseBlock());
      rewriter.create<scf::YieldOp>(loc, elseFull.getResult());
    }
    return selector;
  };

  auto makeSubview = [&](Location loc, Value root,
                         tensor::InsertSliceOp geometry) {
    SmallVector<OpFoldResult> offsets;
    SmallVector<OpFoldResult> sizes;
    SmallVector<OpFoldResult> strides;
    for (int64_t value : geometry.getStaticOffsets())
      offsets.push_back(rewriter.getIndexAttr(value));
    for (int64_t value : geometry.getStaticSizes())
      sizes.push_back(rewriter.getIndexAttr(value));
    for (int64_t value : geometry.getStaticStrides())
      strides.push_back(rewriter.getIndexAttr(value));
    auto viewType = cast<MemRefType>(memref::SubViewOp::inferResultType(
        fullType, offsets, sizes, strides));
    return rewriter.create<memref::SubViewOp>(loc, viewType, root, offsets,
                                              sizes, strides);
  };

  // Producer: use the same selected full root for both FIXPIPE writes. The two
  // existing transfer/synchronization groups remain intact; only their storage
  // is coalesced into disjoint subviews.
  rewriter.setInsertionPoint(plan.lowFixpipe.selector);
  auto producerRoot = createSelectedRoot(
      plan.lowFixpipe.selector.getLoc(),
      plan.lowFixpipe.selector.getCondition(), producerFull.first,
      producerFull.second);
  auto lowView = makeSubview(plan.lowFixpipe.selector.getLoc(),
                             producerRoot.getResult(0), plan.lowInsert);
  IRMapping lowMapping;
  lowMapping.map(plan.lowFixpipe.thenFixpipe.getDst(), lowView.getResult());
  rewriter.clone(*plan.lowFixpipe.thenFixpipe, lowMapping);
  rewriter.eraseOp(plan.lowFixpipe.selector);

  rewriter.setInsertionPoint(plan.highFixpipe.selector);
  auto highView = makeSubview(plan.highFixpipe.selector.getLoc(),
                              producerRoot.getResult(0), plan.highInsert);
  IRMapping highMapping;
  highMapping.map(plan.highFixpipe.thenFixpipe.getDst(), highView.getResult());
  rewriter.clone(*plan.highFixpipe.thenFixpipe, highMapping);
  rewriter.eraseOp(plan.highFixpipe.selector);

  // Consumer: select the complete ping/pong score root and feed it directly to
  // softmax. This removes both tensor inserts and therefore the later 64 KiB
  // concatenation allocation and its two copies.
  rewriter.setInsertionPoint(plan.lowInsert);
  auto receiverRoot = createSelectedRoot(
      plan.lowInsert.getLoc(), plan.lowReceiver.selector.getCondition(),
      receiverFull.first, receiverFull.second);
  auto genericFullType = MemRefType::get(fullTensorType.getShape(),
                                         fullTensorType.getElementType());
  auto receiverCast = rewriter.create<memref::MemorySpaceCastOp>(
      plan.lowInsert.getLoc(), genericFullType, receiverRoot.getResult(0));
  auto score = rewriter.create<bufferization::ToTensorOp>(
      plan.lowInsert.getLoc(), fullTensorType, receiverCast.getResult(), true,
      true);
  plan.highInsert.getResult().replaceAllUsesWith(score.getResult());
  rewriter.eraseOp(plan.highInsert);
  rewriter.eraseOp(plan.lowInsert);
  rewriter.eraseOp(plan.highReceiver.selector);
  rewriter.eraseOp(plan.lowReceiver.selector);

  SmallVector<annotation::MarkOp> oldMarks{
      plan.lowReceiverThenMark,  plan.lowReceiverElseMark,
      plan.highReceiverThenMark, plan.highReceiverElseMark,
      plan.lowProducerThenMark,  plan.lowProducerElseMark,
      plan.highProducerThenMark, plan.highProducerElseMark};
  for (annotation::MarkOp mark : oldMarks)
    rewriter.eraseOp(mark);
  SmallVector<memref::AllocOp> oldAllocs{
      plan.lowReceiver.thenAlloc,  plan.lowReceiver.elseAlloc,
      plan.highReceiver.thenAlloc, plan.highReceiver.elseAlloc,
      plan.lowProducerThen,         plan.lowProducerElse,
      plan.highProducerThen,        plan.highProducerElse};
  for (memref::AllocOp alloc : oldAllocs) {
    if (!alloc->use_empty())
      return failure();
    rewriter.eraseOp(alloc);
  }
  return success();
}
