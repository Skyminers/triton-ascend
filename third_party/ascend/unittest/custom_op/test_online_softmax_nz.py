"""A5 builtin: NZ P/max/block-sum with caller-owned online state."""
import pytest
import torch
import torch_npu
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al


@triton.jit
def step(S, M, L, P, MN, BS, LN, A, BM: tl.constexpr, BN: tl.constexpr):
    i = tl.arange(0, BM * BN)
    rows = tl.arange(0, BM)
    scores = tl.load(S + i).reshape(BN // 16, BM // 16, 16, 16)
    m = tl.load(M + rows)
    l = tl.load(L + rows)
    p, mn, block_sum = al.online_softmax_nz(scores, m)
    alpha = tl.exp(m - mn)
    ln = l * alpha + block_sum
    tl.store(P + i, p.reshape(BM * BN))
    tl.store(MN + rows, mn)
    tl.store(BS + rows, block_sum)
    tl.store(LN + rows, ln)
    tl.store(A + rows, alpha)


@triton.jit
def nd_reference(S, M, L, P, MN, BS, LN, A, BN: tl.constexpr):
    rows = tl.program_id(0) * 32 + tl.arange(0, 32)
    cols = tl.arange(0, BN)
    scores = tl.load(S + rows[:, None] * BN + cols[None, :])
    m, l = tl.load(M + rows), tl.load(L + rows)
    mn = tl.maximum(m, tl.max(scores, 1, propagate_nan=True), propagate_nan=tl.PropagateNan.ALL)
    prob = tl.exp(scores - mn[:, None])
    block_sum = tl.sum(prob, 1)
    alpha = tl.exp(m - mn)
    ln = l * alpha + block_sum
    tl.store(P + rows[:, None] * BN + cols[None, :], prob.to(tl.float8e4nv))
    tl.store(MN + rows, mn)
    tl.store(BS + rows, block_sum)
    tl.store(LN + rows, ln)
    tl.store(A + rows, alpha)


@pytest.mark.parametrize("bn", [256, 512])
@pytest.mark.parametrize("bm", [32, 64, 128])
def test_online_softmax_nz(bm, bn):
    torch.manual_seed(17)
    m = torch.full((bm,), -float("inf"), device="npu")
    l = torch.zeros_like(m)
    mr, lr = m.clone(), l.clone()
    for iteration, offset in enumerate([0., 20., -20., 80.]):
        scores = torch.randn(bm, bn) * 3 + offset
        if iteration % 2 == 0:
            scores[:, bn // 2:] = -float("inf")
        nz = scores.reshape(bm // 16, 16, bn // 16, 16).permute(2, 0, 1, 3).contiguous().npu()
        p = torch.empty((bn // 32, bm // 16, 16, 32), dtype=torch.float8_e4m3fn, device="npu")
        mn, block_sum, ln, alpha = [torch.empty_like(m) for _ in range(4)]
        # N512/M128 needs two-AIV row splitting. Keep this case strict: the
        # pure-vector pipeline currently does not apply the requested split.
        compiled = step[(1,)](nz, m, l, p, mn, block_sum, ln, alpha, bm, bn,
                              enable_auto_bind_sub_block=(bn == 512 and bm == 128))
        torch.npu.synchronize()
        ir = compiled.asm["ttadapter"]
        assert "__builtin_online_softmax_nz" in ir and "bitcode =" not in ir
        expected_p = torch.empty((bm, bn), dtype=torch.float8_e4m3fn, device="npu")
        new_m, expected_sum, new_l, a = [torch.empty_like(m) for _ in range(4)]
        nd_reference[(bm // 32,)](scores.npu(), mr, lr, expected_p,
                                  new_m, expected_sum, new_l, a, bn)
        decoded = p.cpu().float().permute(1, 2, 0, 3).reshape(bm, bn)
        torch.testing.assert_close(decoded, expected_p.cpu().float(), rtol=0, atol=0)
        for actual, expected in [(mn, new_m), (block_sum, expected_sum),
                                 (ln, new_l), (alpha, a)]:
            torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=1e-5, atol=1e-5)
        m, l, mr, lr = mn, ln, new_m, new_l


def test_reject_unsupported_columns():
    bm, bn = 32, 128
    scores = torch.empty((8, 2, 16, 16), device="npu")
    state = torch.zeros((bm,), device="npu")
    p = torch.empty((4, 2, 16, 32), dtype=torch.float8_e4m3fn, device="npu")
    outputs = [torch.empty_like(state) for _ in range(4)]
    with pytest.raises(triton.CompilationError):
        step[(1,)](scores, state, state, p, *outputs, bm, bn)


@pytest.mark.parametrize("bn", [256, 512])
@pytest.mark.parametrize("bm", [32, 64, 128])
@pytest.mark.parametrize("value", [-float("inf"), float("inf"), float("nan")])
def test_nonfinite_rows(value, bm, bn):
    scores = torch.ones((bm, bn))
    scores[0, :] = value
    scores[1, 0] = value
    nz = scores.reshape(bm // 16, 16, bn // 16, 16).permute(2, 0, 1, 3).contiguous().npu()
    m = torch.full((bm,), -float("inf"), device="npu")
    l = torch.zeros_like(m)
    p = torch.empty((bn // 32, bm // 16, 16, 32), dtype=torch.float8_e4m3fn, device="npu")
    actual = [torch.empty_like(m) for _ in range(4)]
    expected_p = torch.empty((bm, bn), dtype=torch.float8_e4m3fn, device="npu")
    expected = [torch.empty_like(m) for _ in range(4)]
    step[(1,)](nz, m, l, p, *actual, bm, bn,
                enable_auto_bind_sub_block=(bn == 512 and bm == 128))
    nd_reference[(bm // 32,)](scores.npu(), m, l, expected_p, *expected, bn)
    decoded = p.cpu().float().permute(1, 2, 0, 3).reshape(bm, bn)
    torch.testing.assert_close(decoded, expected_p.cpu().float(), rtol=0, atol=0, equal_nan=True)
    for x, y in zip(actual, expected):
        torch.testing.assert_close(x.cpu(), y.cpu(), rtol=1e-5, atol=1e-5, equal_nan=True)
