# test_bwd_expanded_many_configs.py
# Run with:  pytest -q test_bwd_expanded_many_configs.py

import math
import random
import pytest
import torch

# === IMPORT YOUR KERNELS HERE ===
# Adjust these to your actual module paths.
# from mykernels import prepare_wy_repr_bwd, prepare_wy_repr_bwd_expanded

torch.manual_seed(0)
random.seed(0)

# ---------------- helpers to build matching A / A_expanded from one ground-truth causal matrix ----------------

def _strictly_lower_mask(T, device, dtype):
    i = torch.arange(T, device=device)
    return (i[:, None] > i[None, :]).to(dtype)

def _make_A_full(T, device, dtype):
    A = torch.randn(T, T, device=device, dtype=dtype)
    return A * _strictly_lower_mask(T, device, dtype)

def _make_A_BT_from_full(A_full, BT):
    """Per-tile A of shape (T, BT) so tile i uses columns [i*BT : i*BT+BT)."""
    T = A_full.shape[0]
    device, dtype = A_full.device, A_full.dtype
    A_BT = torch.zeros(T, BT, device=device, dtype=dtype)
    ntiles = (T + BT - 1) // BT
    for it in range(ntiles):
        j0 = it * BT
        j1 = min(j0 + BT, T)
        rows = slice(j0, min(j0 + BT, T))
        A_BT[rows, :j1 - j0] = A_full[rows, j0:j1]
    return A_BT

def _make_A_expanded_from_full(A_full, N):
    """Expanded-window A_expanded of shape (T, N) with windows [t0 : t0+N)."""
    T = A_full.shape[0]
    device, dtype = A_full.device, A_full.dtype
    A_exp = torch.zeros(T, N, device=device, dtype=dtype)
    nchunks = (T + N - 1) // N
    for it_e in range(nchunks):
        t0 = it_e * N
        j0, j1 = t0, min(t0 + N, T)
        i1 = min(t0 + N, T)
        for i in range(t0, i1):
            A_exp[i, :j1 - j0] = A_full[i, j0:j1]
    return A_exp

def _make_block_diag_from_lens(lens, device, dtype):
    """Block-diagonal strictly-lower matrix for a packed ragged batch (sum(lens) x sum(lens))."""
    T_total = sum(lens)
    A = torch.zeros(T_total, T_total, device=device, dtype=dtype)
    off = 0
    for L in lens:
        A_block = _make_A_full(L, device, dtype)
        A[off:off+L, off:off+L] = A_block
        off += L
    return A

# ---------------- parameter sets ----------------

FIXED_CASES = [
    # (B, T, H, K, V, N, BT)
    dict(B=1, T=64,  H=1, K=32, V=32,  N=48,  BT=16),   # small, clean multiples
    dict(B=2, T=96,  H=2, K=64, V=80,  N=112, BT=32),   # tail in N (N > T sometimes), stress boundaries
    dict(B=1, T=33,  H=3, K=48, V=40,  N=25,  BT=16),   # odd lengths
]

VARLEN_CASES = [
    dict(lens=[17, 31],      H=2, K=32, V=24, N=33, BT=16),
    dict(lens=[57, 83, 41],  H=2, K=48, V=32, N=71, BT=32),
]

RTOL = 1e-3
ATOL = 2e-3

cuda_available = torch.cuda.is_available()

# ---------------- tests: fixed-length ----------------

@pytest.mark.skipif(not cuda_available, reason="CUDA is required for Triton kernels")
@pytest.mark.parametrize("cfg", FIXED_CASES)
def test_bwd_expanded_equivalence_fixed(cfg):
    device = "cuda"
    dtype = torch.float16

    B = cfg["B"]; T = cfg["T"]; H = cfg["H"]; K = cfg["K"]; V = cfg["V"]; N = cfg["N"]; BT = cfg["BT"]

    # inputs
    k    = torch.randn(B, T, H, K, device=device, dtype=dtype)
    v    = torch.randn(B, T, H, V, device=device, dtype=dtype)
    g    = torch.randn(B, T, H,     device=device, dtype=dtype)
    beta = torch.randn(B, T, H,     device=device, dtype=dtype)

    # upstream grads
    dw = torch.randn(B, T, H, K, device=device, dtype=dtype)
    du = torch.randn(B, T, H, V, device=device, dtype=dtype)

    # A_full -> A (T,BT) and A_expanded (T,N)
    A_full     = _make_A_full(T, device, dtype)
    A_BT       = _make_A_BT_from_full(A_full, BT)        # (T, BT)
    A_expanded = _make_A_expanded_from_full(A_full, N)   # (T, N)

    A     = A_BT.expand(B, T, H, BT).contiguous()
    A_exp = A_expanded.expand(B, T, H, N).contiguous()

    # run baseline
    dk_ref, dv_ref, dbeta_ref, dg_ref = prepare_wy_repr_bwd(
        k=k, v=v, g=g, beta=beta, A=A, dw=dw, du=du, cu_seqlens=None, chunk_size=BT
    )
    # run expanded
    dk_tst, dv_tst, dbeta_tst, dg_tst = prepare_wy_repr_bwd_expanded(
        k=k, v=v, g=g, beta=beta, A_expanded=A_exp, dw=dw, du=du, cu_seqlens=None, chunk_size=BT
    )

    torch.testing.assert_close(dk_ref, dk_tst, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(dv_ref, dv_tst, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(dbeta_ref, dbeta_tst, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(dg_ref, dg_tst, rtol=RTOL, atol=ATOL)

# ---------------- tests: variable-length packed ----------------

@pytest.mark.skipif(not cuda_available, reason="CUDA is required for Triton kernels")
@pytest.mark.parametrize("cfg", VARLEN_CASES)
def test_bwd_expanded_equivalence_varlen(cfg):
    device = "cuda"
    dtype = torch.float16

    lens = cfg["lens"]; H = cfg["H"]; K = cfg["K"]; V = cfg["V"]; N = cfg["N"]; BT = cfg["BT"]
    B = len(lens)
    T_total = sum(lens)

    k    = torch.randn(B, T_total, H, K, device=device, dtype=dtype)
    v    = torch.randn(B, T_total, H, V, device=device, dtype=dtype)
    g    = torch.randn(B, T_total, H,     device=device, dtype=dtype)
    beta = torch.randn(B, T_total, H,     device=device, dtype=dtype)

    dw = torch.randn(B, T_total, H, K, device=device, dtype=dtype)
    du = torch.randn(B, T_total, H, V, device=device, dtype=dtype)

    # Block-diagonal A_full per sequence, then map to packed axis
    A_full_block = _make_block_diag_from_lens(lens, device, dtype)
    A_BT_total   = _make_A_BT_from_full(A_full_block, BT)       # (T_total, BT)
    A_exp_total  = _make_A_expanded_from_full(A_full_block, N)  # (T_total, N)

    A     = A_BT_total.expand(B, T_total, H, BT).contiguous()
    A_exp = A_exp_total.expand(B, T_total, H, N).contiguous()

    # cu_seqlens prefix sums
    cu_seqlens = torch.tensor([0] + [sum(lens[:i+1]) for i in range(B)], device=device, dtype=torch.int32)

    # run baseline
    dk_ref, dv_ref, dbeta_ref, dg_ref = prepare_wy_repr_bwd(
        k=k, v=v, g=g, beta=beta, A=A, dw=dw, du=du, cu_seqlens=cu_seqlens, chunk_size=BT
    )
    # run expanded
    dk_tst, dv_tst, dbeta_tst, dg_tst = prepare_wy_repr_bwd_expanded(
        k=k, v=v, g=g, beta=beta, A_expanded=A_exp, dw=dw, du=du, cu_seqlens=cu_seqlens, chunk_size=BT
    )

    torch.testing.assert_close(dk_ref, dk_tst, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(dv_ref, dv_tst, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(dbeta_ref, dbeta_tst, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(dg_ref, dg_tst, rtol=RTOL, atol=ATOL)

# ---------------- optional sanity: coverage (nonzero/finite) on tails ----------------

@pytest.mark.skipif(not cuda_available, reason="CUDA is required for Triton kernels")
def test_bwd_expanded_tail_coverage_sanity():
    """Quick check that tail tiles (when N and BT don't evenly divide T) still produce finite grads."""
    device = "cuda"
    dtype = torch.float16

    B, T, H, K, V = 1, 77, 2, 40, 24
    BT, N = 16, 50   # both create tails

    k    = torch.randn(B, T, H, K, device=device, dtype=dtype)
    v    = torch.randn(B, T, H, V, device=device, dtype=dtype)
    g    = torch.randn(B, T, H,     device=device, dtype=dtype)
    beta = torch.randn(B, T, H,     device=device, dtype=dtype)
    dw   = torch.randn(B, T, H, K,  device=device, dtype=dtype)
    du   = torch.randn(B, T, H, V,  device=device, dtype=dtype)

    A_full     = _make_A_full(T, device, dtype)
    A_BT       = _make_A_BT_from_full(A_full, BT)
    A_expanded = _make_A_expanded_from_full(A_full, N)

    A     = A_BT.expand(B, T, H, BT).contiguous()
    A_exp = A_expanded.expand(B, T, H, N).contiguous()

    # run both and ensure finite & close
    dk_ref, dv_ref, dbeta_ref, dg_ref = prepare_wy_repr_bwd(
        k=k, v=v, g=g, beta=beta, A=A, dw=dw, du=du, cu_seqlens=None, chunk_size=BT
    )
    dk_tst, dv_tst, dbeta_tst, dg_tst = prepare_wy_repr_bwd_expanded(
        k=k, v=v, g=g, beta=beta, A_expanded=A_exp, dw=dw, du=du, cu_seqlens=None, chunk_size=BT
    )

    for t in (dk_tst, dv_tst, dbeta_tst, dg_tst):
        assert torch.isfinite(t).all()

    torch.testing.assert_close(dk_ref, dk_tst, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(dv_ref, dv_tst, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(dbeta_ref, dbeta_tst, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(dg_ref, dg_tst, rtol=RTOL, atol=ATOL)
