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
    outputs = [tl.full((8, rows // 16, 16, 32), 0, tl.float8e4nv, _semantic=_semantic)]
    outputs += [tl.full((rows,), 0, tl.float32, _semantic=_semantic) for _ in range(2)]
    return custom_semantic("__builtin_online_softmax_nz", scores, m, indices,
                           out=outputs, _semantic=_semantic)


@triton.jit
def online_softmax_nz(scores, m):
    """Compute an online-softmax tile on Ascend950 (A5).

    ``scores`` is FP32 NZ16 [16, M/16, 16, 16], representing M x 256.
    ``m`` is the FP32 [M] running maximum. M is 32, 64 or 128 (32
    supports a half tile). Scores must already include scaling and masking.

    Returns (P, m_new, block_sum): P is FP8 E4M3FN NZ32
    [8, M/16, 16, 32]; the other results are FP32 [M].
    m_new = max(m, rowmax(scores)); P = cast_fp8(exp(scores - m_new));
    block_sum = rowsum(exp(scores - m_new)). The sum is reduced before FP8
    quantization. The caller owns running l and alpha updates. All outputs are
    fully written. Rows are independent; the reduction spans all 256 columns.
    Nonfinite rows follow device exp/max/cast semantics (no empty-row zeroing).
    """
    tl.static_assert(len(scores.shape) == 4, "scores must be rank-4 NZ16")
    tl.static_assert(scores.shape[0] == 16 and scores.shape[2] == 16 and scores.shape[3] == 16)
    tl.static_assert(scores.shape[1] == 2 or scores.shape[1] == 4 or scores.shape[1] == 8)
    i = tl.arange(0, 256) % 128
    indices = (((i // 32) * 16 + i % 16) * 4 + (i % 32) // 16).to(tl.uint8)
    return _online_softmax_nz(scores, m, indices)
