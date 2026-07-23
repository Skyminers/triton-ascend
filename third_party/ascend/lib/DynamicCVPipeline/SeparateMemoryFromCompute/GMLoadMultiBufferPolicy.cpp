/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

// GMLoadMultiBufferPolicyPass
// ===========================
// Chooses a per-load GM-load multi-buffer depth by weighing each load against a
// UB / L1 byte budget and the surrounding compute-block pipeline, and records
// the decision as an `hivm.multi_buffer` attribute on the load's backing buffer
// mark. The downstream AddMultiBufferToGMLoad pass reads that attribute and does
// the actual loop rewriting.
//
// Model (see also the design doc GMLoadMultiBufferPolicy.md and the design
// memory `gmload-multibuffer-budget-plan`):
//   * Two independent budget pools, keyed by the buffer's hivm address space:
//       - UB pool  = 248KB * ub_budget%   -> vector GM loads
//       - L1 pool  = 512KB (always full)  -> cube   GM loads
//     UB may exceed 100% because PlanMemory reuses it downstream, so the
//     constraint is a plain sum  Sum(depth*size) <= budget  (no liveness). L1
//     has no such downstream reuse, so its budget is fixed at the physical size.
//   * inter-core transfer buffers are a *fixed* pre-charge on each pool (capped
//     at double buffering); only GM loads carry a variable depth.
//   * Depth is binary {single, double}: throughput varies by compute mode, so no
//     reliable block *time* can be computed. Each load instead gets a purely
//     topological urgency = size / same-core masking work of its consumer, and a
//     greedy knapsack grants double from the most urgent load down until the
//     pool budget is exhausted.
//   * A user-supplied `hivm.multi_buffer` (via compile_hint / al.multibuffer) is
//     honored verbatim as a pure output override; it is not fed into urgency or
//     pre-charged against the budget, so changing one hint never perturbs the
//     depth chosen for any other load.

#include "ascend/include/DynamicCVPipeline/SeparateMemoryFromCompute/GMLoadMultiBufferPolicyPass.h"
#include "ascend/include/DynamicCVPipeline/SeparateMemoryFromCompute/AddMultiBufferToGMLoadTypes.h"
#include "ascend/include/DynamicCVPipeline/Common/BufferCountManager.h"
#include "ascend/include/DynamicCVPipeline/Common/MultiBufferOverride.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <queue>

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::CVPipeline;

static constexpr const char *DEBUG_TYPE = "gm-load-multi-buffer-policy";
#define LDBG(...) LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << __VA_ARGS__ << "\n")

namespace {

// --- Physical sizes and policy knobs ---
constexpr int64_t kUBBaseBytes = 248 * 1024;
constexpr int64_t kL1BaseBytes = 512 * 1024;
constexpr int kDefaultBudgetPct = 100;

// Depth is a binary {single, double} choice. Whether double is worth it is a
// relative-urgency question, not an absolute one: hardware throughput varies by
// compute mode, so no reliable block *time* can be computed. We therefore rank
// loads by a purely topological, throughput-free urgency and let the budget grant
// double from the most urgent down.
constexpr int kDoubleBuffer = 2;

enum class Pool { UB, L1, None };

struct LoadEdge {
    memref::AllocOp alloc;
    Pool pool = Pool::None;
    int64_t size = 0;             // bytes of one buffer
    double urgency = 0.0;         // size / same-core masking work (higher = more urgent)
    int depth = 1;                // current assignment (knapsack output)
    bool pinned = false;          // user fixed the depth via hivm.multi_buffer; output override only
};

// Number of statically-known elements in a shaped type, or 0 if dynamic/unknown.
int64_t staticNumElements(Type type)
{
    auto shaped = dyn_cast<ShapedType>(type);
    if (!shaped || !shaped.hasStaticShape())
        return 0;
    int64_t n = 1;
    for (int64_t d : shaped.getShape())
        n *= d;
    return n;
}

// Bytes of one instance of a memref's backing buffer, or 0 if the shape is
// dynamic (hivm::GetBufferBitSize returns ShapedType::kDynamic, a large
// negative sentinel, in that case).
int64_t memrefBytes(memref::AllocOp alloc)
{
    auto bits = hivm::GetBufferBitSize(alloc.getResult());
    if (!bits || *bits <= 0)
        return 0;
    return *bits / 8;
}

// Read the hivm address space carried by a memref, if any.
Pool poolOfMemref(MemRefType type)
{
    auto space = dyn_cast_or_null<hivm::AddressSpaceAttr>(type.getMemorySpace());
    if (!space)
        return Pool::None;
    switch (space.getAddressSpace()) {
    case hivm::AddressSpace::UB:
        return Pool::UB;
    case hivm::AddressSpace::L1:
        return Pool::L1;
    default:
        return Pool::None;
    }
}

struct ConsumerInfo {
    int blockId = -1;
    CoreType core = CoreType::UNDETERMINED;
};

// The first block-carrying compute op reachable from the load: its block id
// anchors the masking analysis, and its core type is the fallback pool
// classification when the backing alloc has no address space.
ConsumerInfo firstConsumer(Operation *markedOp)
{
    llvm::DenseSet<Operation *> seen;
    std::queue<Operation *> work;
    for (Value res : markedOp->getResults())
        for (Operation *user : res.getUsers())
            work.push(user);

    while (!work.empty()) {
        Operation *op = work.front();
        work.pop();
        if (!seen.insert(op).second)
            continue;
        if (auto bid = getOpBlockId(op)) {
            CoreType core = getOpCoreType(op);
            if (core == CoreType::CUBE_ONLY || core == CoreType::VECTOR_ONLY)
                return {*bid, core};
        }
        for (Value res : op->getResults())
            for (Operation *user : res.getUsers())
                work.push(user);
    }
    return {};
}

// Per-iteration compute-block graph, purely topological: each block's static
// output element count, its core type, and its same-iteration predecessor blocks
// (SSA cross-block deps; loop-carried iter_arg edges have no defining op and are
// naturally excluded). No time/throughput is computed anywhere.
struct BlockGraph {
    llvm::DenseMap<int, double> elems;                  // block -> total output element count
    llvm::DenseMap<int, CoreType> core;
    llvm::DenseMap<int, llvm::SmallVector<int>> preds;  // block -> its predecessor blocks
};

BlockGraph buildBlockGraph(ModuleOp module)
{
    BlockGraph g;
    module.walk([&](Operation *op) {
        auto bid = getOpBlockId(op);
        if (!bid.has_value())
            return;
        g.core[*bid] = getOpCoreType(op);
        for (Value res : op->getResults())
            g.elems[*bid] += static_cast<double>(staticNumElements(res.getType()));
        for (Value operand : op->getOperands()) {
            Operation *def = operand.getDefiningOp();
            if (!def)
                continue;
            auto srcBid = getOpBlockId(def);
            if (srcBid.has_value() && *srcBid != *bid)
                g.preds[*bid].push_back(*srcBid);
        }
    });
    return g;
}

// Same-core masking work available to a load feeding block C: the total element
// count of C's same-core transitive ancestors. In steady state the consumer's
// core runs all of them before C in the same iteration, so their compute can
// hide the load's transfer. Element counts are throughput-free and, within one
// core, directly comparable (the core's unknown throughput cancels in any ratio).
// `cache` memoizes by consumer block, since multiple loads (e.g. the two GM
// operands of a matmul) commonly feed the same block and would otherwise
// recompute an identical ancestor traversal.
double sameCoreMaskWork(int consumerBlock, const BlockGraph &g, llvm::DenseMap<int, double> &cache)
{
    if (auto it = cache.find(consumerBlock); it != cache.end())
        return it->second;

    auto coreIt = g.core.find(consumerBlock);
    if (coreIt == g.core.end())
        return cache[consumerBlock] = 0.0;
    CoreType core = coreIt->second;

    double sum = 0.0;
    llvm::DenseSet<int> visited;
    SmallVector<int> work;
    auto predsIt = g.preds.find(consumerBlock);
    if (predsIt != g.preds.end())
        work.assign(predsIt->second.begin(), predsIt->second.end());

    while (!work.empty()) {
        int b = work.pop_back_val();
        if (!visited.insert(b).second)
            continue;
        auto bc = g.core.find(b);
        if (bc != g.core.end() && bc->second == core)
            sum += g.elems.lookup(b);
        auto pit = g.preds.find(b);
        if (pit != g.preds.end())
            for (int p : pit->second)
                work.push_back(p);
    }
    return cache[consumerBlock] = sum;
}

int readBudgetPct(ModuleOp module, llvm::StringRef key)
{
    if (auto attr = module->getAttrOfType<IntegerAttr>(key))
        return std::max<int>(1, static_cast<int>(attr.getInt()));
    return kDefaultBudgetPct;
}

// Record the chosen depth as a static attr on the load's backing mark (creating
// a mark if the alloc has none), matching the compile_hint / cost-model contract.
void tagMultiBuffer(memref::AllocOp alloc, int depth, OpBuilder &builder)
{
    auto depthAttr = builder.getI32IntegerAttr(depth);
    if (annotation::MarkOp mark = gmload::findAllocMark(alloc)) {
        mark->setAttr(kMultiBuffer, depthAttr);
        return;
    }
    builder.setInsertionPointAfter(alloc);
    auto mark = builder.create<annotation::MarkOp>(alloc.getLoc(), alloc.getResult());
    mark->setAttr(kMultiBuffer, depthAttr);
}

// Budget allocation by urgency: every load starts at single (1). Grant double
// from the most urgent down (urgency = size / same-core masking work), spending
// the pool budget until no more candidate fits. A more urgent load that doesn't
// fit is skipped so a smaller, less urgent one can still be served.
void assignKnapsack(SmallVectorImpl<LoadEdge *> &loads, int64_t budget, int64_t used)
{
    for (LoadEdge *e : loads)
        used += e->size; // single baseline
    llvm::stable_sort(loads, [](const LoadEdge *a, const LoadEdge *b) { return a->urgency > b->urgency; });
    for (LoadEdge *e : loads) {
        if (used + e->size <= budget) {
            e->depth = kDoubleBuffer;
            used += e->size;
        }
    }
}

// Per-pool budget/usage/worklist, indexed by `Pool::UB`/`Pool::L1` (`Pool::None`
// never indexes into this — it is resolved to UB or L1 before use).
struct PoolState {
    int64_t budget = 0;
    int64_t used = 0;
    SmallVector<LoadEdge *> loads;
};

} // namespace

void GMLoadMultiBufferPolicyPass::getDependentDialects(DialectRegistry &registry) const
{
    registry.insert<annotation::AnnotationDialect, memref::MemRefDialect>();
}

void GMLoadMultiBufferPolicyPass::runOnOperation()
{
    ModuleOp module = getOperation();
    OpBuilder builder(module.getContext());

    std::array<PoolState, 2> pools;
    int ubPct = readBudgetPct(module, kUbBudget);
    pools[static_cast<size_t>(Pool::UB)].budget = kUBBaseBytes * ubPct / 100;
    // L1 has no downstream PlanMemory reuse, so the plain-sum estimate is exact:
    // there is nothing to tune, just fill the physical L1.
    pools[static_cast<size_t>(Pool::L1)].budget = kL1BaseBytes;
    LDBG("budgets: UB=" << pools[static_cast<size_t>(Pool::UB)].budget << " (" << ubPct << "%)  L1="
                         << pools[static_cast<size_t>(Pool::L1)].budget << " (full)");

    // inter-core actual buffer count (capped at 2 by design).
    // TODO(flag-downgrade): mirror AddMultiBufferOuterScope's flag-budget
    // downgrade so a group forced back to single is pre-charged as 1, not 2.
    int interN = std::clamp(BufferCountManager(module).getBufferCountByType(BufferCountManager::DepType::InterCore),
                            1, 2);

    SmallVector<gmload::MarkedLoad> marked = gmload::collectMarkedOps(module);
    BlockGraph graph = buildBlockGraph(module);
    llvm::DenseMap<int, double> maskWorkCache;

    // Fixed pre-charge: inter-core transfer buffers, split by their address space.
    module.walk([&](memref::AllocOp alloc) {
        if (!alloc->hasAttr(kTransferId))
            return;
        Pool pool = poolOfMemref(alloc.getType());
        if (pool == Pool::None)
            return;
        pools[static_cast<size_t>(pool)].used += static_cast<int64_t>(interN) * memrefBytes(alloc);
    });
    LDBG("inter-core pre-charge (n=" << interN << "): UB=" << pools[static_cast<size_t>(Pool::UB)].used << " L1="
                                     << pools[static_cast<size_t>(Pool::L1)].used);

    // Enumerate and classify every GM load: pool and a purely topological urgency
    // (size / same-core masking work). The budget knapsack then grants double
    // from the most urgent down.
    SmallVector<LoadEdge> edges;
    for (const gmload::MarkedLoad &ml : marked) {
        memref::AllocOp alloc = ml.allocOp;
        if (!alloc)
            continue;
        int64_t size = memrefBytes(alloc);
        if (size <= 0) // dynamic shape: cannot budget precisely, leave to global default.
            continue;

        ConsumerInfo consumer = firstConsumer(ml.markedOp);

        Pool pool = poolOfMemref(alloc.getType());
        if (pool == Pool::None) {
            // Fallback when the alloc lacks an address space: classify by the
            // consuming core (cube->L1, everything else->UB).
            pool = (consumer.core == CoreType::CUBE_ONLY) ? Pool::L1 : Pool::UB;
        }

        // Purely topological urgency: how exposed the load is. urgency = size /
        // same-core masking work of the consumer. Small masking work (consumer
        // near a source, little same-core compute in front) or a large transfer
        // -> more urgent. Within a pool the consumer core is fixed, so this
        // ratio's order is throughput-free (the unknown per-core / DMA throughputs
        // are common factors that cancel).
        double maskWork = consumer.blockId >= 0 ? sameCoreMaskWork(consumer.blockId, graph, maskWorkCache) : 0.0;
        LoadEdge e;
        e.alloc = alloc;
        e.pool = pool;
        e.size = size;
        e.pinned = gmload::getMultiBufferOverride(alloc).has_value();
        e.urgency = static_cast<double>(size) / std::max(maskWork, 1e-6);
        edges.push_back(e);
        LDBG("load: pool=" << (pool == Pool::UB ? "UB" : "L1") << " size=" << size << " maskWork=" << maskWork
                           << " urgency=" << e.urgency
                           << (e.pinned ? " [user-pinned; algorithm unaffected]" : ""));
    }

    // Bucket loads into their pool's worklist (pointers so knapsack writes back).
    for (LoadEdge &e : edges)
        pools[static_cast<size_t>(e.pool)].loads.push_back(&e);

    for (PoolState &pool : pools)
        assignKnapsack(pool.loads, pool.budget, pool.used);

    // Emit marks. User-pinned loads keep their own mark untouched (the pin is
    // the final word); the rest get the algorithm's depth. depth==1 is the
    // default single-buffer behavior and needs no mark.
    for (LoadEdge &e : edges) {
        if (e.pinned)
            continue;
        if (e.depth <= 1)
            continue;
        tagMultiBuffer(e.alloc, e.depth, builder);
        LDBG("assigned depth=" << e.depth << " to load size=" << e.size);
    }
}

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createGMLoadMultiBufferPolicyPass()
{
    return std::make_unique<GMLoadMultiBufferPolicyPass>();
}

void registerGMLoadMultiBufferPolicyPass()
{
    registerPass([]() -> std::unique_ptr<mlir::Pass> { return createGMLoadMultiBufferPolicyPass(); });
}

} // namespace triton
} // namespace mlir
