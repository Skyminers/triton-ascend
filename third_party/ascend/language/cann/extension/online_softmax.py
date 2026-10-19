# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Licensed under the MIT License. See LICENSE for details.
"""Online softmax with native NZ score/probability layouts."""
import triton
import triton.language as tl
from triton.language.core import builtin
from .custom_op import custom_semantic


@builtin
def _online_softmax_nz(scores, m, indices, causal_limits=None, mask_required=None, _semantic=None):
    rows = scores.type.shape[1] * 16
    outputs = [tl.full((scores.type.shape[0] // 2, rows // 16, 16, 32), 0,
                       tl.float8e4nv, _semantic=_semantic)]
    outputs += [tl.full((rows,), 0, scores.dtype, _semantic=_semantic),
                tl.full((rows,), 0, tl.float32, _semantic=_semantic)]
    return custom_semantic("__builtin_online_softmax_nz", scores, m, indices, causal_limits, mask_required,
                           out=outputs, _semantic=_semantic)


@triton.jit
def online_softmax_nz(scores, m, causal_limits=None, mask_required=None):
    """Return (FP8 NZ32 P, FP16 m_new, FP32 block_sum) for scaled FP16 NZ16 [512/16,M/16,16,16].
    M=32/64/128; optional int32[M] causal_limits are inclusive local column limits.
    mask_required=False requires all limits>=511; callers widen m/m_new before alpha and own l updates."""
    tl.static_assert(len(scores.shape) == 4, "scores must be rank-4 NZ16")
    tl.static_assert(scores.dtype == tl.float16, "scores must be FP16")
    tl.static_assert(scores.shape[0] == 32 and
                     scores.shape[2] == 16 and scores.shape[3] == 16)
    tl.static_assert(scores.shape[1] == 2 or scores.shape[1] == 4 or scores.shape[1] == 8)
    i = tl.arange(0, 256)
    indices = ((i // 32) * 32 + (i % 16) * 2 + (i % 32) // 16).to(tl.uint8)
    return _online_softmax_nz(scores, m, indices, causal_limits, mask_required)
