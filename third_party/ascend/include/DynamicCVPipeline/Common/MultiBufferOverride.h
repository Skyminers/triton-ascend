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

#ifndef TRITON_ADAPTER_COMMON_MULTI_BUFFER_OVERRIDE_H
#define TRITON_ADAPTER_COMMON_MULTI_BUFFER_OVERRIDE_H

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include <optional>

namespace mlir {
namespace annotation {
class MarkOp;
} // namespace annotation
} // namespace mlir

// Shared by GMLoadMultiBufferPolicyPass (writer), AddMultiBufferToGMLoad
// (reader), and the SeparateMemoryFromCompute umbrella pass (existence
// check) — kept in one place rather than three so all three agree on what
// counts as a per-load override. Defined in Common/MultiBufferOverride.cpp.
namespace gmload {

// Find the annotation.mark that annotates `alloc`, if one already exists.
mlir::annotation::MarkOp findAllocMark(mlir::memref::AllocOp alloc);

// Per-load multi-buffer depth set by GMLoadMultiBufferPolicyPass or the user
// (compile_hint / al.multibuffer), read from an `hivm.multi_buffer` attr on
// the backing alloc's annotation.mark. Returns std::nullopt when the load has
// no override.
std::optional<int> getMultiBufferOverride(mlir::memref::AllocOp alloc);

} // namespace gmload

#endif // TRITON_ADAPTER_COMMON_MULTI_BUFFER_OVERRIDE_H
