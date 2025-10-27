# test_equivalence_many_configs.py
# Run with:  pytest -q test_equivalence_many_configs.py

import math
import os
import random
import pytest
import torch

# ==== IMPORT YOUR KERNELS HERE ====
# Adjust the module path as needed.
# from mykernels import (
#     recompute_w_u_expanded,   # (k, v, beta, A_expanded, *, g=None, gk=None, cu_seqlens=None, BK=64, BV=64, BT=64)
#     recompute_w_u_fwd,        # (k, v, beta, A, g=None, gk=None, cu_seqlens=None)
# )
#
# For illustration in this template, we assume they're available in the namespace.

torch.manual_seed(0)
random.seed(0)

# --------- helpers to build matching A and A_expanded from one ground-truth causal matrix ---------

def _strictly_lower_mask(T, device, dtype):
    i = torch.arange(T, device=device)
    m = (i[:, None] > i[None, :])
    return m.to(dtype)

def _make_A_full(T, device, dtype):
    A = torch.randn(T, T, device=device, dtype=dtype)
    return A * _strictly_lower_mask(T, device, dtype)

def _make_A_BT_from_full(A_full, BT):
    """Materialize per-tile A of shape (T, BT) such that tile i uses columns [i*BT : i*BT+BT)."""
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
    """Materialize expanded-window A_expanded of shape (T, N) with windows [t0 : t0+N)."""
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
    """Build a block-diagonal strictly-lower matrix for a packed ragged batch (sum(lens) x sum(lens))."""
    T_total = sum(lens)
    A = torch.zeros(T_total, T_total, device=device, dtype=dtype)
    off = 0
    for L in lens:
        A_block = _make_A_full(L, device, dtype)
        A[off:off+L, off:off+L] = A_block
        off += L
    return A

# -------------------- parameter sets --------------------

# Fixed-length shapes: (B, T, H, K, V, N, BT)
FIXED_CASES = [
    # small & exact multiples
    dict(B=1, T=64,  H=1, K=32, V=32,  N=48,  BT=16),
    # larger, N not a multiple of BT (tail tiles)
    dict(B=2, T=96,  H=2, K=64, V=80,  N=112, BT=32),
    # odd sizes, exercise many boundaries
    dict(B=1, T=33,  H=3, K=48, V=40,  N=25,  BT=16),
]

# Varlen: each item defines per-batch sequence lengths
VARLEN_CASES = [
    # small ragged
    dict(lens=[17, 31], H=2, K=32, V=24, N=33, BT=16),
    # larger ragged
    dict(lens=[57, 83, 41], H=2, K=48, V=32, N=71, BT=32),
]

# whether to include gating terms
GATE_MODES = [(False, False), (True, False), (True, True)]

# tolerances tuned for fp16 accumulation-in-fp32 flows
RTOL = 1e-3
ATOL = 2e-3

# Default per-head dims for blocking (must be compatible with your kernels’ defaults)
BK_DEFAULT = 64
BV_DEFAULT = 64

cuda_available = torch.cuda.is_available()


@pytest.mark.skipif(not cuda_available, reason="CUDA is required for Triton kernels")
@pytest.mark.parametrize("cfg", FIXED_CASES)
@pytest.mark.parametrize("use_g,use_gk", GATE_MODES)
def test_equivalence_fixed_many(cfg, use_g, use_gk):
    device = "cuda"
    dtype = torch.float16

    B = cfg["B"]; T = cfg["T"]; H = cfg["H"]; K = cfg["K"]; V = cfg["V"]; N = cfg["N"]; BT = cfg["BT"]

    # inputs
    k    = torch.randn(B, T, H, K, device=device, dtype=dtype)
    v    = torch.randn(B, T, H, V, device=device, dtype=dtype)
    beta = torch.randn(B, T, H,     device=device, dtype=dtype)
    g    = torch.randn(B, T, H,     device=device, dtype=dtype) if use_g  else None
    gk   = torch.randn(B, T, H, K,  device=device, dtype=dtype) if use_gk else None

    # shared ground-truth causal matrix
    A_full     = _make_A_full(T, device, dtype)
    A_BT       = _make_A_BT_from_full(A_full, BT)       # (T, BT)
    A_expanded = _make_A_expanded_from_full(A_full, N)  # (T, N)

    # broadcast to (B, T, H, last)
    A     = A_BT.expand(B, T, H, BT).contiguous()
    A_exp = A_expanded.expand(B, T, H, N).contiguous()

    # run both forwards
    w_tile, u_tile = recompute_w_u_fwd(
        k=k, v=v, beta=beta, A=A, g=g, gk=gk, cu_seqlens=None
    )
    w_exp, u_exp = recompute_w_u_expanded(
        k=k, v=v, beta=beta, A_expanded=A_exp, g=g, gk=gk,
        cu_seqlens=None, BK=BK_DEFAULT, BV=BV_DEFAULT, BT=BT
    )

    torch.testing.assert_close(w_tile, w_exp, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(u_tile, u_exp, rtol=RTOL, atol=ATOL)


@pytest.mark.skipif(not cuda_available, reason="CUDA is required for Triton kernels")
@pytest.mark.parametrize("cfg", VARLEN_CASES)
@pytest.mark.parametrize("use_g,use_gk", [(False, False), (True, True)])  # two representative gate modes
def test_equivalence_varlen_many(cfg, use_g, use_gk):
    device = "cuda"
    dtype = torch.float16

    lens = cfg["lens"]; H = cfg["H"]; K = cfg["K"]; V = cfg["V"]; N = cfg["N"]; BT = cfg["BT"]
    B = len(lens)
    T_total = sum(lens)

    # inputs for packed ragged batch (B, T_total, H, *)
    k    = torch.randn(B, T_total, H, K, device=device, dtype=dtype)
    v    = torch.randn(B, T_total, H, V, device=device, dtype=dtype)
    beta = torch.randn(B, T_total, H,     device=device, dtype=dtype)
    g    = torch.randn(B, T_total, H,     device=device, dtype=dtype) if use_g  else None
    gk   = torch.randn(B, T_total, H, K,  device=device, dtype=dtype) if use_gk else None

    # block-diagonal A_full aligned with lens, then map to A and A_expanded on the packed axis
    A_full_block = _make_block_diag_from_lens(lens, device, dtype)
    A_BT_total   = _make_A_BT_from_full(A_full_block, BT)      # (T_total, BT)
    A_exp_total  = _make_A_expanded_from_full(A_full_block, N) # (T_total, N)

    A     = A_BT_total.expand(B, T_total, H, BT).contiguous()
    A_exp = A_exp_total.expand(B, T_total, H, N).contiguous()

    # cu_seqlens: [0, L0, L0+L1, ...]
    cu_seqlens = torch.tensor([0] + [sum(lens[:i+1]) for i in range(B)], device=device, dtype=torch.int32)

    # run both forwards
    w_tile, u_tile = recompute_w_u_fwd(
        k=k, v=v, beta=beta, A=A, g=g, gk=gk, cu_seqlens=cu_seqlens
    )
    w_exp, u_exp = recompute_w_u_expanded(
        k=k, v=v, beta=beta, A_expanded=A_exp, g=g, gk=gk,
        cu_seqlens=cu_seqlens, BK=BK_DEFAULT, BV=BV_DEFAULT, BT=BT
    )

    torch.testing.assert_close(w_tile, w_exp, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(u_tile, u_exp, rtol=RTOL, atol=ATOL)
