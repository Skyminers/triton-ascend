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

#include "ascend/include/DynamicCVPipeline/SplitDataflow/FlagIdReuse.h"

#include <cstddef>
#include <optional>

#include "DynamicCVPipeline/Common/MemoryEffectsTracker.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"

static constexpr const char *DEBUG_TYPE = "flag-id-reuse";
#define LOG_DEBUG(...) LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << __VA_ARGS__)

using namespace mlir;
using namespace mlir::triton;
using namespace hivm;

void FlagIdReuseManager::insertRelationBetweenSetAndWait(Operation *before, Operation *after)
{
    if (!before || !after) {
        LOG_DEBUG("Warning: inserting a relation with null operation, ignored.");
        return;
    }
    
    LOG_DEBUG("Add an edge from \n" << *before << " to \n" << *after << "\n");
    relations[before].push_back(after);
    return;
}

DenseMap<int, int> FlagIdReuseManager::reuseInterCoreTransferFlagIds(const llvm::SmallVector<Operation *> &syncOps)
{
    DenseMap<int, int> remapResult;
    if (syncOps.empty()) {
        return remapResult;
    }

    preworkForAnalyze(syncOps);

    // Step 1: Check reuse flagIds and merge them into groups.
    for (auto &[lhs, lhsOps]: flagIdToOps) {
        for (auto &[rhs, rhsOps]: flagIdToOps) {
            // Check whether can add this remap result : lhs -> rhs
            if (lhsOps.empty() || rhsOps.empty() || // FlagId which is merged will be empty
                disjointSet.findRoot(lhs) == disjointSet.findRoot(rhs)) {
                continue;
            }
            if (checkReusable(lhs, rhs)) {
                merge(lhs, rhs);
            }
        }
    }

    // Step 2: Remap every non-root flagId to its group representative (root).
    // The root keeps its original flagId, so the whole group converges onto one
    // existing flagId and never collides with unmerged ids.
    DenseMap<int, int> compactedId;
    int newFlagId = 1;
    for (auto &[flagId, _]: flagIdToOps) {
        auto root = disjointSet.findRoot(flagId);
        if (!compactedId.contains(root)) {
            compactedId[root] = newFlagId++;
        }
        remapResult[flagId] = compactedId[root];
    }
    return remapResult;
}

void FlagIdReuseManager::preworkForAnalyze(const llvm::SmallVector<Operation *> &syncOps)
{
    flagIdToOps.clear();
    for (auto op: syncOps) {
        auto flagId = getFlagId(op);
        if (flagId == -1) {
            continue;
        }
        flagIdToOps[flagId].push_back(op);
    }
    disjointSet.fa.clear();
}

// Reusable condition:
// 1. Operations from [flagId = tobe] all have path to [flagId = asIs]
// 2. Operations from [flagId = asIs] have no path to [flagId = toBe]
// Which:
// 1. Represent asIs is process all after toBe
// 2. This is not a circle process. (Current scenes does not support circle, is this still necessary?)
bool FlagIdReuseManager::checkReusable(int asIs, int toBe)
{
    LOG_DEBUG("Cheecking reusable for rewritting " << asIs << " to " << toBe << "\n");
    llvm::SmallSet<Operation *, CVPipeline::INIT_SIZE> visited;
    for (auto fromOp: flagIdToOps[toBe]) {
        for (auto destOp: flagIdToOps[asIs]) {
            visited.clear();
            if (!hasPath(visited, fromOp, destOp)) {
                LOG_DEBUG("No path from\n" << *fromOp << "\nto\n" << *destOp << ", failed to reuse.\n");
                return false;
            }
        }
    }
    LOG_DEBUG("Reuse successfully.");
    return true;
}

bool FlagIdReuseManager::hasPath(llvm::SmallSet<Operation *, CVPipeline::INIT_SIZE> &visited, Operation *from, Operation *to)
{
    if (from == to) {
        return true;
    }
    if (visited.contains(from)) {
        return false;
    }
    visited.insert(from);
    for (auto nxt: relations[from]) {
        if (hasPath(visited, nxt, to)) {
            return true;
        }
    }
    return false;
}

int FlagIdReuseManager::getFlagId(Operation *op)
{
    if (auto setOp = llvm::dyn_cast<SyncBlockSetOp>(op)) {
        if (auto staticFlagId = setOp.getStaticFlagId()) {
            return staticFlagId->getInt();
        }
        LOG_DEBUG("Warning: SyncBlockSetOp has no static flagId (dynamic?), skipped: \n" << *op << "\n");
        return -1;
    }
    if (auto waitOp = llvm::dyn_cast<SyncBlockWaitOp>(op)) {
        if (auto staticFlagId = waitOp.getStaticFlagId()) {
            return staticFlagId->getInt();
        }
        LOG_DEBUG("Warning: SyncBlockWaitOp has no static flagId (dynamic?), skipped: \n" << *op << "\n");
        return -1;
    }
    LOG_DEBUG("Warning: failed to get flagId from op: \n" << *op << "\n");
    return -1;
}

// This function will merge rhs to lhs. Including the flagIdToOps and the disjointSet.
void FlagIdReuseManager::merge(int lhs, int rhs)
{
    flagIdToOps[lhs].append(flagIdToOps[rhs].begin(), flagIdToOps[rhs].end());
    flagIdToOps[rhs].clear();
    disjointSet.merge(lhs, rhs);
}

int FlagIdReuseManager::DisjointSet::findRoot(int flagId)
{
    auto it = fa.find(flagId);
    if (it == fa.end()) {
        fa[flagId] = flagId;
        return flagId;
    }
    
    if (it->second == flagId) {
        return flagId;
    }
    it->second = findRoot(it->second);
    return it->second;
}

void FlagIdReuseManager::DisjointSet::merge(int lhs, int rhs)
{
    auto lhsFa = findRoot(lhs);
    auto rhsFa = findRoot(rhs);
    fa[rhsFa] = lhsFa;
}
