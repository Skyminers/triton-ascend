// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
// Licensed under the MIT license.
#ifndef TRITON_ASCEND_PAIRED_F16_PV_ACCUMULATE_H
#define TRITON_ASCEND_PAIRED_F16_PV_ACCUMULATE_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"
#include <memory>

namespace mlir::triton {
// Opt in to validation of the existing full-width K512 PV task.
inline constexpr llvm::StringLiteral kPairedF16PVAccumulate =
    "ascend.paired_f16_pv_accumulate";
std::unique_ptr<OperationPass<ModuleOp>> createPairedF16PVAccumulatePass();
} // namespace mlir::triton

#endif
