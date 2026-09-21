// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
// Licensed under the MIT license.
#ifndef TRITON_ASCEND_PAIRED_F16_PV_ACCUMULATE_H
#define TRITON_ASCEND_PAIRED_F16_PV_ACCUMULATE_H

#include "llvm/ADT/StringRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir::triton {
// Explicit numerical opt-in on func.func; never inferred from a kernel name.
// This permits the restricted PV reduction-order change, not unsafe aliasing
// or a claim that delay-two scheduling has been implemented.
inline constexpr llvm::StringLiteral kPairedF16PVAccumulate =
    "ascend.paired_f16_pv_accumulate";
std::unique_ptr<OperationPass<ModuleOp>> createPairedF16PVAccumulatePass();
} // namespace mlir::triton

#endif
