// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
// Licensed under the MIT license.
#ifndef TRITON_ASCEND_PAIRED_F16_ACC_OWNERSHIP_H
#define TRITON_ASCEND_PAIRED_F16_ACC_OWNERSHIP_H

#include "llvm/ADT/StringRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include <memory>

namespace mlir::triton {
// Explicit destination contract consumed by ArithToHFusion. The integer value
// is the tensor operand whose storage is reused for the operation result.
inline constexpr llvm::StringLiteral kDestinationOperand =
    "ascend.destination_operand";
// Allows the destination one additional use only when DynamicCV forwards it
// unchanged from the branch opposite the annotated update.
inline constexpr llvm::StringLiteral kDestinationOppositeIfForward =
    "ascend.destination_opposite_if_forward";
// Function-level scheduling contract attached only after the paired
// online-softmax recurrence has been proved. The integer is the loop-carried
// argument whose old value may be extended after its softmax update.
inline constexpr llvm::StringLiteral kLoopCarriedReadBeforeUpdate =
    "ascend.loop_carried_read_before_update";
std::unique_ptr<OperationPass<ModuleOp>> createPairedF16AccOwnershipPass();
// After DynamicCV has materialized the two FIXPIPE transfers and their
// ping-pong buffers, fold the two half-score buffers of each slot into one
// dense score buffer. The softmax then consumes that selected buffer directly
// instead of allocating and copying through a third concatenation buffer.
LogicalResult coalescePairedF16ScoreBuffers(ModuleOp module);
} // namespace mlir::triton

#endif
