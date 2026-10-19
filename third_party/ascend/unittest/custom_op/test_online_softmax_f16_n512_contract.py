"""Host contract tests; run remotely, no installed API or device required."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


@pytest.fixture
def registration():
    source = Path(__file__).parents[2] / "language/cann/extension/builtin_custom_ops.py"
    node = next(n for n in ast.parse(source.read_text()).body
                if isinstance(n, ast.ClassDef) and n.name == "_online_softmax_nz")
    node.decorator_list = []
    env = dict(tl=NS(float16="f16", float32="f32", float8e4nv="f8", uint8="u8", int32="i32", int1="i1"),
               CORE=NS(VECTOR=0), PIPE=NS(PIPE_V=0), MODE=NS(SIMD=0))
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), env)
    return env["_online_softmax_nz"]


def tensor(dtype, shape):
    return NS(dtype=dtype, type=NS(shape=shape), shape=shape)


@pytest.mark.parametrize("rows", [32, 64, 128])
def test_f16_n512_registration(registration, rows):
    registration(tensor("f16", [32, rows // 16, 16, 16]), tensor("f16", [rows]),
                 tensor("u8", [256]), out=[tensor("f8", [16, rows // 16, 16, 32]),
                 tensor("f16", [rows]), tensor("f32", [rows])])


@pytest.mark.parametrize("dtype,n1", [("f16", 16), ("f32", 16), ("f32", 32)])
def test_reject_unsupported_score_abi(registration, dtype, n1):
    with pytest.raises(AssertionError):
        registration(tensor(dtype, [n1, 4, 16, 16]), tensor(dtype, [64]),
                     tensor("u8", [256]), out=[tensor("f8", [n1 // 2, 4, 16, 32]),
                     tensor(dtype, [64]), tensor("f32", [64])])


@pytest.mark.parametrize("n1,old,new,total", [(16, "f16", "f16", "f32"),
    (32, "f32", "f16", "f32"), (32, "f16", "f32", "f32"),
    (32, "f16", "f16", "f16")])
def test_f16_n512_reject_mixed_state(registration, n1, old, new, total):
    with pytest.raises(AssertionError):
        registration(tensor("f16", [n1, 4, 16, 16]), tensor(old, [64]),
                     tensor("u8", [256]), out=[tensor("f8", [n1 // 2, 4, 16, 32]),
                     tensor(new, [64]), tensor(total, [64])])


@pytest.mark.parametrize("rows", [32, 64, 128])
def test_f16_n512_causal_limits_registration(registration, rows):
    registration(tensor("f16", [32, rows // 16, 16, 16]), tensor("f16", [rows]),
                 tensor("u8", [256]), causal_limits=tensor("i32", [rows]),
                 out=[tensor("f8", [16, rows // 16, 16, 32]),
                      tensor("f16", [rows]), tensor("f32", [rows])])


@pytest.mark.parametrize("limit_dtype,limit_shape", [
    ("f32", [64]), ("i32", [32]), ("i32", [64, 1]),
])
def test_causal_limits_reject_incompatible_abi(registration, limit_dtype, limit_shape):
    with pytest.raises(AssertionError):
        registration(tensor("f16", [32, 4, 16, 16]), tensor("f16", [64]),
                     tensor("u8", [256]), causal_limits=tensor(limit_dtype, limit_shape),
                     out=[tensor("f8", [16, 4, 16, 32]),
                          tensor("f16", [64]), tensor("f32", [64])])


@pytest.mark.parametrize("rows", [32, 64, 128])
def test_f16_n512_cce_offsets_and_pack(rows):
    # Simulate physical byte ownership, including all 8-row store groups.
    written = set()
    for r in range(0, rows, 8):
        for k in range(16):
            reads = {2 * ((2 * k + j) * 16 * rows + (r // 16) * 256
                         + (r % 16) * 16 + t) + b
                     for j in range(2) for t in range(128) for b in range(2)}
            assert not reads & written
            store = 64 * rows * k + 512 * (r // 16) + 32 * (r % 16)
            output = set(range(store, store + 256))
            assert output <= reads
            assert not output & written
            assert max(output) < rows * 512 * 2
            written |= output
    assert len(written) == rows * 512
    # CCE ZERO/TWO carry even/odd columns of first NZ16 block;
    # ONE/THREE carry even/odd columns of the second NZ16 block.
    packed = [None] * 256
    for row in range(8):
        for col in range(16):
            lane = row * 8 + col // 2
            packed[4 * lane + 2 * (col % 2)] = (row, col)
            packed[4 * lane + 2 * (col % 2) + 1] = (row, col + 16)
    lut = [(i // 32) * 32 + (i % 16) * 2 + (i % 32) // 16 for i in range(256)]
    assert [packed[i] for i in lut] == [(r, c) for r in range(8) for c in range(32)]


@pytest.mark.parametrize("flag", [False, True, tensor("i1", [])])
def test_f16_n512_conditional_causal_registration(registration, flag):
    registration(tensor("f16", [32, 4, 16, 16]), tensor("f16", [64]),
                 tensor("u8", [256]), causal_limits=tensor("i32", [64]),
                 mask_required=flag,
                 out=[tensor("f8", [16, 4, 16, 32]),
                      tensor("f16", [64]), tensor("f32", [64])])


@pytest.mark.parametrize("limits,flag", [(None, False), (tensor("i32", [64]), tensor("i32", [])),
                                         (tensor("i32", [64]), tensor("i1", [1]))])
def test_conditional_causal_reject_incompatible_flag(registration, limits, flag):
    with pytest.raises(AssertionError):
        registration(tensor("f16", [32, 4, 16, 16]), tensor("f16", [64]),
                     tensor("u8", [256]), causal_limits=limits, mask_required=flag,
                     out=[tensor("f8", [16, 4, 16, 32]),
                          tensor("f16", [64]), tensor("f32", [64])])
