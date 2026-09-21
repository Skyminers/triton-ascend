// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
// Licensed under the MIT license.

#include "ascend/include/DynamicCVPipeline/PairedF16PVAccumulate.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::triton;

namespace {
bool hasTensorType(Value value, ArrayRef<int64_t> shape, Type element) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  return type && type.getShape() == shape && type.getElementType() == element &&
         !type.getEncoding();
}

bool isPositiveZeroInit(Value value) {
  auto fill = value.getDefiningOp<linalg::FillOp>();
  if (!fill || !fill.getDpsInits()[0].getDefiningOp<tensor::EmptyOp>())
    return false;
  FloatAttr number;
  return matchPattern(fill.getDpsInputs()[0], m_Constant(&number)) &&
         number.getType().isF32() && number.getValue().isZero() &&
         !number.getValue().isNegative();
}

bool isCanonicalPV(linalg::MatmulOp op) {
  auto *ctx = op.getContext();
  auto fp8 = Float8E4M3FNType::get(ctx);
  auto f32 = Float32Type::get(ctx);
  if (op.getNumResults() != 1 || op.getDpsInputs().size() != 2 ||
      op.getDpsInits().size() != 1 ||
      !hasTensorType(op.getDpsInits()[0], {128, 128}, f32) ||
      !hasTensorType(op.getResult(0), {128, 128}, f32))
    return false;
  auto rhs = dyn_cast<RankedTensorType>(op.getDpsInputs()[1].getType());
  if (!rhs || !hasTensorType(op.getDpsInputs()[0], {128, 512}, fp8) ||
      rhs.getShape() != ArrayRef<int64_t>({512, 128}) ||
      rhs.getElementType() != fp8 || rhs.getEncoding())
    return false;
  AffineExpr m, n, k;
  bindDims(ctx, m, n, k);
  SmallVector<AffineMap> expected{
      AffineMap::get(3, 0, {m, k}, ctx),
      AffineMap::get(3, 0, {k, n}, ctx),
      AffineMap::get(3, 0, {m, n}, ctx)};
  return op.getIndexingMapsArray() == expected;
}

LogicalResult validateFullPV(hivm::CustomOp custom) {
  auto *ctx = custom.getContext();
  auto f16 = Float16Type::get(ctx);
  auto f32 = Float32Type::get(ctx);
  auto fp8 = Float8E4M3FNType::get(ctx);
  // Tensor V2 ABI. This is not an ownership shortcut for a bufferized custom.
  if (custom->getNumOperands() != 6 || custom->getNumResults() != 3 ||
      !isa<scf::ForOp>(custom->getParentOp()) ||
      !hasTensorType(custom->getOperand(0), {32, 8, 16, 16}, f16) ||
      !hasTensorType(custom->getOperand(1), {128}, f16) ||
      !hasTensorType(custom->getOperand(2), {256}, IntegerType::get(ctx, 8)) ||
      !hasTensorType(custom.getResult(0), {16, 8, 16, 32}, fp8) ||
      !hasTensorType(custom.getResult(1), {128}, f16) ||
      !hasTensorType(custom.getResult(2), {128}, f32))
    return failure();
  for (unsigned i = 0; i < 3; ++i)
    if (custom->getOperand(i + 3).getType() != custom.getResult(i).getType())
      return failure();

  // Consume the complete logical P[128, 512] in one Cube matmul. The
  // Fractal-to-ND conversion is a layout view, not a split or copy.
  if (!custom.getResult(0).hasOneUse())
    return failure();
  auto convert = dyn_cast<hivm::ConvertLayoutOp>(
      *custom.getResult(0).getUsers().begin());
  if (!convert || convert->getBlock() != custom->getBlock() ||
      convert.getSource() != custom.getResult(0) ||
      !convert.getResult().hasOneUse() ||
      convert.getSrcLayout().getDataLayout() != hivm::DataLayout::Fractal ||
      convert.getDstLayout().getDataLayout() != hivm::DataLayout::ND ||
      convert.getSrcLayout().getTransposeValue().value_or(false) ||
      convert.getDstLayout().getTransposeValue().value_or(false) ||
      !convert.getSrcLayout().getFractalSizes() ||
      convert.getSrcLayout().getFractalSizes().asArrayRef() !=
          ArrayRef<int64_t>({16, 32}) ||
      !hasTensorType(convert.getResult(), {128, 512}, fp8))
    return failure();
  auto matmul = dyn_cast<linalg::MatmulOp>(
      *convert.getResult().getUsers().begin());
  if (!matmul || matmul->getBlock() != custom->getBlock() ||
      !isCanonicalPV(matmul) ||
      matmul.getDpsInputs()[0] != convert.getResult() ||
      !isPositiveZeroInit(matmul.getDpsInits()[0]) ||
      !matmul.getResult(0).hasOneUse())
    return failure();
  return success();
}

class PairedF16PVAccumulatePass final
    : public PassWrapper<PairedF16PVAccumulatePass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PairedF16PVAccumulatePass)
  StringRef getArgument() const final { return "paired-f16-pv-accumulate"; }
  StringRef getDescription() const final {
    return "Validate the FP16 N512 full-width PV task";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect, linalg::LinalgDialect, scf::SCFDialect,
                    tensor::TensorDialect, hivm::HIVMDialect>();
  }
  void runOnOperation() override {
    ModuleOp module = getOperation();
    constexpr StringLiteral requestAttr =
        "ascend.request_paired_f16_pv_accumulate";
    SmallVector<func::FuncOp> functions(module.getOps<func::FuncOp>());
    if (module->hasAttr(requestAttr)) {
      if (!module->getAttrOfType<UnitAttr>(requestAttr) ||
          functions.size() != 1) {
        module.emitError(
            "paired FP16 PV task request requires a unit marker and one function");
        return signalPassFailure();
      }
      functions.front()->setAttr(kPairedF16PVAccumulate,
                                 UnitAttr::get(&getContext()));
      module->removeAttr(requestAttr);
    }

    // Full-width PV is already present in the source IR; this pass only
    // validates the selected task shape for later ownership scheduling.
    for (auto func : getOperation().getOps<func::FuncOp>()) {
      Attribute request = func->getAttr(kPairedF16PVAccumulate);
      if (!request)
        continue;
      if (!isa<UnitAttr>(request)) {
        func.emitError("ascend.paired_f16_pv_accumulate must be a unit attribute");
        return signalPassFailure();
      }
      bool found = false;
      auto walk = func.walk([&](hivm::CustomOp custom) -> WalkResult {
        if (custom.getName() != "__builtin_online_softmax_nz")
          return WalkResult::advance();
        found = true;
        if (failed(validateFullPV(custom))) {
          custom.emitError("paired FP16 PV requires an exclusive canonical "
                           "full-width P input and zero-init K512 matmul");
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
      if (walk.wasInterrupted())
        return signalPassFailure();
      if (!found) {
        func.emitError("paired FP16 PV accumulation requires an N512 FP16 custom");
        return signalPassFailure();
      }
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createPairedF16PVAccumulatePass() {
  return std::make_unique<PairedF16PVAccumulatePass>();
}
