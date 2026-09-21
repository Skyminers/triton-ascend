# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Licensed under the MIT License. See LICENSE for details.
"""Online softmax with native NZ score/probability layouts."""
import triton
import triton.language as tl
from triton.language.core import builtin
from .custom_op import custom_semantic


@builtin
def _online_softmax_nz(scores, m, indices, _semantic=None):
    rows = scores.type.shape[1] * 16
    outputs = [tl.full((scores.type.shape[0] // 2, rows // 16, 16, 32), 0,
                       tl.float8e4nv, _semantic=_semantic)]
    outputs += [tl.full((rows,), 0, scores.dtype, _semantic=_semantic),
                tl.full((rows,), 0, tl.float32, _semantic=_semantic)]
    return custom_semantic("__builtin_online_softmax_nz", scores, m, indices,
                           out=outputs, _semantic=_semantic)


@triton.jit
def online_softmax_nz(scores, m):
    """Compute an online-softmax tile on Ascend950 (A5).

    ``scores`` is FP32 NZ16 [N1, M/16, 16, 16]; N1=16 selects N=256,
    N1=32 selects restricted N=512. ``m`` is the FP32 [M] running maximum.
    N=512 also accepts FP16 scores with FP16 m and m_new; block_sum stays
    FP32. Half state is explicit: arbitrary FP32 m is never rounded silently.
    Callers cast old/new m to FP32 before computing alpha and updating l.
    M is 32, 64 or 128. Scores must already include scaling and masking.

    Returns (P, m_new, block_sum): P is FP8 E4M3FN NZ32
    [N1/2, M/16, 16, 32]; m_new matches the score dtype and block_sum is FP32 [M].
    m_new = max(m, rowmax(scores)); P = cast_fp8(exp(scores - m_new));
    block_sum = rowsum(exp(scores - m_new)). The sum is reduced before FP8
    quantization. The caller owns running l and alpha updates. All outputs are
    fully written. Rows are independent; the reduction spans all N columns.
    Nonfinite rows follow device exp/max/cast semantics (no empty-row zeroing).

    N=256 retains its existing layout and semantics. Restricted N=512 requires
    a single score buffer slot and a same-root strided P view supplied by
    dedicated bufferization, plus a matching dedicated kernel template; this
    frontend does not implement in-place reuse. For R=M rows in each call,
    score memref strides are [16*R, 256, 16, 1] (FP32 elements), and P strides
    are [128*R, 1024, 64, 1] (FP8 elements), not dense NZ32. The N=512 two-AIV
    path requires M=128 before splitting and M=64 per AIV afterwards.
    For FP16 N512, score strides are the same in half elements; P strides
    are [64*R, 512, 32, 1] in a shared R*512*2-byte root. FusedExpSub uses
    half inputs and FP32 register results; both 256-column halves share
    one maximum and one FP32 sum. No full FP32 score or separate P is needed.
    """
    tl.static_assert(len(scores.shape) == 4, "scores must be rank-4 NZ16")
    tl.static_assert((scores.shape[0] == 16 or scores.shape[0] == 32) and
                     scores.shape[2] == 16 and scores.shape[3] == 16)
    tl.static_assert(scores.shape[1] == 2 or scores.shape[1] == 4 or scores.shape[1] == 8)
    i = tl.arange(0, 256) % 128
    if scores.dtype == tl.float16:
        i = tl.arange(0, 256)
        indices = ((i // 32) * 32 + (i % 16) * 2 + (i % 32) // 16).to(tl.uint8)
    else:
        indices = (((i // 32) * 16 + i % 16) * 4 + (i % 32) // 16).to(tl.uint8)
    return _online_softmax_nz(scores, m, indices)
