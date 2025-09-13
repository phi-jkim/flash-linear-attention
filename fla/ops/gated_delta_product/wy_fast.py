# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.op import exp
from fla.utils import check_shared_mem

# Expanded-chunk recompute W/U using 64-row micro-tiles.
# N must be a multiple of 64. A_expanded is (B, T, H, N) with columns aligned
# to the expanded window [t0, t0+N), i.e., col index 0..N-1 == absolute j = t0+col.
# For rows outside [t0, t0+N), or j > i, A should be zero (lower-tri masked).

@triton.heuristics({
    'USE_G':  lambda args: args['g']  is not None,
    'USE_GK': lambda args: args['gk'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[triton.Config({}, num_warps=nw, num_stages=ns)
             for nw in (2, 4, 8) for ns in (2, 3, 4)],
    key=['H', 'K', 'V', 'N', 'BK', 'BV', 'IS_VARLEN'],
)
@triton.jit(do_not_specialize=['T'])
def recompute_w_u_fwd_kernel_expanded(
    k,                 # [B, T, H, K]
    v,                 # [B, T, H, V]
    beta,              # [B, T, H]
    w,                 # [B, T, H, K] (out)
    u,                 # [B, T, H, V] (out)
    A,                 # [B, T, H, N] (expanded)
    g,                 # [B, T, H] or None
    gk,                # [B, T, H, K] or None
    cu_seqlens,        # [B+1] or None
    chunk_indices,     # [num_expanded_chunks, 2] (n, it_exp) or None
    T,                 # int total length if fixed; ignored for varlen
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    N: tl.constexpr,   # expanded chunk size (N % 64 == 0)
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    BT = 64
    M = N // BT  # number of micro-tiles within the expanded window
    pid_exp = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh % H

    # Resolve per-sequence bounds and expanded-chunk start t0 (absolute index)
    if IS_VARLEN:
        i_n  = tl.load(chunk_indices + pid_exp * 2 + 0).to(tl.int32)
        it_e = tl.load(chunk_indices + pid_exp * 2 + 1).to(tl.int32)
        bos  = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos  = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        Tseq = eos - bos
        t0   = it_e * N
    else:
        i_n  = i_b
        bos  = i_b * T
        eos  = bos + T
        Tseq = T
        it_e = pid_exp
        t0   = it_e * N

    # Base pointers (assume contiguous: last dim is stride-1)
    base_beta = beta + bos * H + i_h            # (Tseq,), stride row = H
    base_k    = k    + (bos * H + i_h) * K      # (Tseq, K), strides (H*K, 1)
    base_v    = v    + (bos * H + i_h) * V      # (Tseq, V), strides (H*V, 1)
    base_w    = w    + (bos * H + i_h) * K
    base_u    = u    + (bos * H + i_h) * V
    base_A    = A    + (bos * H + i_h) * N      # (Tseq, N), strides (H*N, 1)
    if USE_G:
        base_g  = g  + bos * H + i_h            # (Tseq,), stride row = H
    if USE_GK:
        base_gk = gk + (bos * H + i_h) * K      # (Tseq, K)

    # Loop over row micro-tiles r, and accumulate from all causal column tiles c <= r
    for r in range(M):
        row_off = t0 + r * BT  # absolute i-range start

        # ---------------- U path (V dim) ----------------
        for iv in range(tl.cdiv(V, BV)):
            acc_u = tl.zeros((BT, BV), dtype=tl.float32)

            for c in range(r + 1):
                col_off = t0 + c * BT  # absolute j-range start

                # A[r,c] block: rows at row_off..row_off+BT-1, cols at (c*BT)..(c*BT+BT-1) within the expanded window
                p_A_rc = tl.make_block_ptr(base_A, (Tseq, N), (H * N, 1),
                                           (row_off, c * BT), (BT, BT), (1, 0))
                b_A_rc = tl.load(p_A_rc, boundary_check=(0, 1))

                # Source rows (j-tile): v and beta (and gates for W path)
                p_v_c = tl.make_block_ptr(base_v, (Tseq, V), (H * V, 1),
                                          (col_off, iv * BV), (BT, BV), (1, 0))
                b_v_c = tl.load(p_v_c, boundary_check=(0, 1))

                # IMPORTANT: β must be taken from SOURCE rows j (col_off), not destination rows i
                p_beta_c = tl.make_block_ptr(base_beta, (Tseq,), (H,), (col_off,), (BT,), (0,))
                b_beta_c = tl.load(p_beta_c, boundary_check=(0,))

                b_vb = (b_v_c * b_beta_c[:, None]).to(b_v_c.dtype)
                acc_u += tl.dot(b_A_rc, b_vb, allow_tf32=False).to(tl.float32)

            p_u_r = tl.make_block_ptr(base_u, (Tseq, V), (H * V, 1),
                                      (row_off, iv * BV), (BT, BV), (1, 0))
            tl.store(p_u_r, acc_u.to(p_u_r.dtype.element_ty), boundary_check=(0, 1))

        # ---------------- W path (K dim) ----------------
        for ik in range(tl.cdiv(K, BK)):
            acc_w = tl.zeros((BT, BK), dtype=tl.float32)

            for c in range(r + 1):
                col_off = t0 + c * BT  # absolute j-range start

                p_A_rc = tl.make_block_ptr(base_A, (Tseq, N), (H * N, 1),
                                           (row_off, c * BT), (BT, BT), (1, 0))
                b_A_rc = tl.load(p_A_rc, boundary_check=(0, 1))

                p_k_c = tl.make_block_ptr(base_k, (Tseq, K), (H * K, 1),
                                          (col_off, ik * BK), (BT, BK), (1, 0))
                b_k_c = tl.load(p_k_c, boundary_check=(0, 1))

                # β, g, gk — all from SOURCE rows j (col_off)
                p_beta_c = tl.make_block_ptr(base_beta, (Tseq,), (H,), (col_off,), (BT,), (0,))
                b_kb = b_k_c * tl.load(p_beta_c, boundary_check=(0,))[:, None]

                if USE_G:
                    p_g_c = tl.make_block_ptr(base_g, (Tseq,), (H,), (col_off,), (BT,), (0,))
                    b_kb *= tl.exp(tl.load(p_g_c, boundary_check=(0,)))[:, None]
                if USE_GK:
                    p_gk_c = tl.make_block_ptr(base_gk, (Tseq, K), (H * K, 1),
                                               (col_off, ik * BK), (BT, BK), (1, 0))
                    b_kb *= tl.exp(tl.load(p_gk_c, boundary_check=(0, 1)))

                acc_w += tl.dot(b_A_rc, b_kb.to(b_k_c.dtype), allow_tf32=False).to(tl.float32)

            p_w_r = tl.make_block_ptr(base_w, (Tseq, K), (H * K, 1),
                                      (row_off, ik * BK), (BT, BK), (1, 0))
            tl.store(p_w_r, acc_w.to(p_w_r.dtype.element_ty), boundary_check=(0, 1))


def recompute_w_u_expanded(
    k: torch.Tensor,            # (B, T, H, K)
    v: torch.Tensor,            # (B, T, H, V)
    beta: torch.Tensor,         # (B, T, H)
    A_expanded: torch.Tensor,   # (B, T, H, N): per-row, cols 0..N-1 align to absolute j=t0..t0+N-1
    *,
    N: int,                     # multiple of 64
    g: Optional[torch.Tensor] = None,     # (B, T, H)
    gk: Optional[torch.Tensor] = None,    # (B, T, H, K)
    cu_seqlens: Optional[torch.LongTensor] = None,
    BK: int = 64,
    BV: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert N % 64 == 0, "N must be a multiple of 64"
    B, T, H, K = k.shape
    V = v.shape[-1]
    assert A_expanded.shape == (B, T, H, N), "A_expanded must be (B, T, H, N)"

    if cu_seqlens is None:
        NT_expanded = triton.cdiv(T, N)
        chunk_indices = None
    else:
        # Build expanded-window starts aligned on N
        chunk_indices = prepare_chunk_indices(cu_seqlens, N).to(torch.int32)
        NT_expanded = chunk_indices.size(0)

    w = torch.empty_like(k)
    u = torch.empty_like(v)

    recompute_w_u_fwd_kernel_expanded[(NT_expanded, B * H)](
        k=k, v=v, beta=beta, w=w, u=u, A=A_expanded,
        g=g, gk=gk, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
        T=T, H=H, K=K, V=V, N=N, BK=BK, BV=BV,
    )
    return w, u


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in (2, 4)
        for num_stages in (2, 3, 4)
    ],
    # Key on expanded N (math window), BK/BV, etc.
    key=['H', 'K', 'V', 'N', 'BK', 'BV', 'IS_VARLEN'],
)
@triton.jit(do_not_specialize=['T'])
def prepare_wy_repr_bwd_kernel_expanded(
    # inputs
    k, v, g, beta, A_exp,   # k:[B,T,H,K], v:[B,T,H,V], g:[B,T,H], beta:[B,T,H], A_exp:[B,T,H,N]
    # upstream grads
    dw, du,                  # dw:[B,T,H,K], du:[B,T,H,V]
    # outputs
    dk, dv, dbeta, dg,       # same shapes as inputs (dk like k, etc.)
    # varlen plumbing
    cu_seqlens, chunk_indices,
    # meta
    T,                       # total seq len for fixed-length
    H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    N: tl.constexpr,         # expanded window size (multiple of 64)
    BK: tl.constexpr, BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    # ---- micro-tiling ----
    BT = 64
    M = N // BT  # number of 64-row micro-tiles within expanded window

    pid_exp = tl.program_id(0)     # expanded-chunk id
    i_bh    = tl.program_id(1)     # batch*head id
    i_b = i_bh // H
    i_h = i_bh % H

    # ---- resolve sequence bounds and expanded-chunk start ----
    if IS_VARLEN:
        i_n  = tl.load(chunk_indices + pid_exp * 2 + 0).to(tl.int32)
        it_e = tl.load(chunk_indices + pid_exp * 2 + 1).to(tl.int32)
        bos  = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos  = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        Tseq = eos - bos
        t0   = it_e * N
    else:
        i_n  = i_b
        bos  = i_b * T
        eos  = bos + T
        Tseq = T
        it_e = pid_exp
        t0   = it_e * N

    # ---- base pointers (contiguous last dim) ----
    base_beta = beta + bos * H + i_h              # (Tseq,), stride row = H
    base_g    = g    + bos * H + i_h              # (Tseq,), stride row = H
    base_k    = k    + (bos * H + i_h) * K        # (Tseq, K), strides (H*K, 1)
    base_v    = v    + (bos * H + i_h) * V        # (Tseq, V), strides (H*V, 1)
    base_dw   = dw   + (bos * H + i_h) * K
    base_du   = du   + (bos * H + i_h) * V
    base_dk   = dk   + (bos * H + i_h) * K
    base_dv   = dv   + (bos * H + i_h) * V
    base_db   = dbeta + bos * H + i_h
    base_dg   = dg    + bos * H + i_h

    # expanded A layout: (Tseq, N), strides (H*N, 1); cols 0..N-1 == j=t0..t0+N-1
    base_A    = A_exp + (bos * H + i_h) * N

    # ---- loop over row micro-tiles r ----
    for r in range(M):
        row_off = t0 + r * BT

        # Load per-row-tile beta/g (for "post" steps)
        p_beta_r = tl.make_block_ptr(base_beta, (Tseq,), (H,), (row_off,), (BT,), (0,))
        p_g_r    = tl.make_block_ptr(base_g,    (Tseq,), (H,), (row_off,), (BT,), (0,))
        b_beta_r = tl.load(p_beta_r, boundary_check=(0,))
        b_g_r    = tl.load(p_g_r,    boundary_check=(0,))

        # Intra-row A block A_rr for local transforms / dv path
        p_A_rr = tl.make_block_ptr(base_A, (Tseq, N), (H * N, 1),
                                   (row_off, r * BT), (BT, BT), (1, 0))
        b_A_rr = tl.load(p_A_rr, boundary_check=(0, 1))

        # Upstream grads at row tile (dw_r, du_r), we’ll reuse across c
        # We iterate features in tiles and accumulate \partial A_r (BT×BT)
        b_dA_r = tl.zeros((BT, BT), dtype=tl.float32)

        # ---- accumulate dA_r from all causal column tiles c ≤ r ----
        # K contribution: Σ_c dW_r · (k_c * β_c * exp(g_c))^T
        for ik in range(tl.cdiv(K, BK)):
            p_dw_r = tl.make_block_ptr(base_dw, (Tseq, K), (H * K, 1),
                                       (row_off, ik * BK), (BT, BK), (1, 0))
            b_dw_r = tl.load(p_dw_r, boundary_check=(0, 1))

            for c in range(r + 1):
                col_off = t0 + c * BT

                # A_{r,c} (masking later is based on absolute i/j)
                # Note: for dA construction we don’t need A values here; we’ll
                # use A_rr in the local transforms just like your original code.
                # (The bare dA term depends on upstream *and* source-side k/v/beta/g).
                p_beta_c = tl.make_block_ptr(base_beta, (Tseq,), (H,), (col_off,), (BT,), (0,))
                p_g_c    = tl.make_block_ptr(base_g,    (Tseq,), (H,), (col_off,), (BT,), (0,))
                b_beta_c = tl.load(p_beta_c, boundary_check=(0,))
                b_g_c    = tl.load(p_g_c,    boundary_check=(0,))

                p_k_c = tl.make_block_ptr(base_k, (Tseq, K), (H * K, 1),
                                          (col_off, ik * BK), (BT, BK), (1, 0))
                b_k_c = tl.load(p_k_c, boundary_check=(0, 1))

                # source-side gating (β_c * exp(g_c))
                b_k_c_gated = (b_k_c * b_beta_c[:, None] * tl.exp(b_g_c)[:, None]).to(b_k_c.dtype)

                # accumulate into dA_r from w-path
                b_dA_r += tl.dot(b_dw_r, tl.trans(b_k_c_gated)).to(tl.float32)

        # V contribution: Σ_c dU_r · (v_c * β_c)^T
        for iv in range(tl.cdiv(V, BV)):
            p_du_r = tl.make_block_ptr(base_du, (Tseq, V), (H * V, 1),
                                       (row_off, iv * BV), (BT, BV), (1, 0))
            b_du_r = tl.load(p_du_r, boundary_check=(0, 1))

            for c in range(r + 1):
                col_off = t0 + c * BT

                p_beta_c = tl.make_block_ptr(base_beta, (Tseq,), (H,), (col_off,), (BT,), (0,))
                b_beta_c = tl.load(p_beta_c, boundary_check=(0,))

                p_v_c = tl.make_block_ptr(base_v, (Tseq, V), (H * V, 1),
                                          (col_off, iv * BV), (BT, BV), (1, 0))
                b_v_c = tl.load(p_v_c, boundary_check=(0, 1))

                b_v_c_gated = (b_v_c * b_beta_c[:, None]).to(b_v_c.dtype)

                # accumulate into dA_r from u-path
                b_dA_r += tl.dot(b_du_r, tl.trans(b_v_c_gated)).to(tl.float32)

        # ---- make strictly-lower mask for (row_off, col_off=r*BT) local tile indices ----
        # We mask with absolute indices; A_rr is BT×BT block at (r,r)
        o_i = row_off + tl.arange(0, BT)
        o_j = t0 + r * BT + tl.arange(0, BT)
        m_i = o_i < Tseq
        m_j = o_j < Tseq
        m_lower_rr = (o_i[:, None] > o_j[None, :]) & (m_i[:, None] & m_j[None, :])

        # Keep only strictly-lower entries as in your original code
        b_dA_r = tl.where(m_lower_rr, b_dA_r, 0)

        # ---- local transforms with A_rr (same as your original: dA = dA*A + A*dA) ----
        tmp = tl.dot(b_dA_r.to(b_A_rr.dtype), b_A_rr)
        b_dA_r = tl.dot(b_A_rr, tmp.to(b_A_rr.dtype))

        # ---- apply -exp(g_i - g_j) scaling (row-tile g on both i and j, as in original) ----
        # Note: original did exp(b_g[:,None] - b_g[None,:]) from the *same* tile’s g.
        # We mirror that here for the local (r,r) block post-transform.
        scale = tl.exp(b_g_r[:, None] - b_g_r[None, :])
        b_dA_r = tl.where(m_lower_rr, -b_dA_r * scale, 0)

        # ---- SECOND PASS over K for dk/dbeta using b_dA_r (same as original) ----
        # Rebuild b_A_k (local statistic) and apply b_dA_r contributions to k/beta.
        b_A_k = tl.zeros((BT, BT), dtype=tl.float32)

        for ik in range(tl.cdiv(K, BK)):
            p_k_r  = tl.make_block_ptr(base_k,  (Tseq, K), (H * K, 1),
                                       (row_off, ik * BK), (BT, BK), (1, 0))
            p_dk_r = tl.make_block_ptr(base_dk, (Tseq, K), (H * K, 1),
                                       (row_off, ik * BK), (BT, BK), (1, 0))
            b_k_r  = tl.load(p_k_r,  boundary_check=(0, 1))
            b_dk_r = tl.load(p_dk_r, boundary_check=(0, 1))

            b_k_beta_r = (b_k_r * b_beta_r[:, None]).to(b_k_r.dtype)

            # accumulate local b_A_k (like original)
            b_A_k += tl.dot(b_k_beta_r, tl.trans(b_k_r)).to(tl.float32)

            # apply dA term to k/beta (like original)
            b_dk_beta_r = tl.dot(b_dA_r, b_k_r)
            # dbeta add
            # sum over feature dim (axis=1 on BT×BK against b_k_r)
            dbeta_add = tl.sum(b_dk_beta_r * b_k_r, 1)
            # update dk_r
            b_dk_r += tl.dot(tl.trans(b_dA_r), b_k_beta_r)
            b_dk_r += b_dk_beta_r * b_beta_r[:, None]

            tl.store(p_dk_r, b_dk_r.to(p_dk_r.dtype.element_ty), boundary_check=(0, 1))

            # accumulate into scalar dbeta buffer (we store after dv step to avoid extra loads)
            # We keep it in registers here by reusing b_beta_r after dv loop.
            # To persist across feature tiles, we’ll accumulate to a local vector:
            if ik == 0:
                b_dbeta_r = dbeta_add.to(tl.float32)
            else:
                b_dbeta_r += dbeta_add.to(tl.float32)

        # ---- DV path for the *row tile r only* (as in original) ----
        for iv in range(tl.cdiv(V, BV)):
            p_v_r  = tl.make_block_ptr(base_v,  (Tseq, V), (H * V, 1),
                                       (row_off, iv * BV), (BT, BV), (1, 0))
            p_du_r = tl.make_block_ptr(base_du, (Tseq, V), (H * V, 1),
                                       (row_off, iv * BV), (BT, BV), (1, 0))
            p_dv_r = tl.make_block_ptr(base_dv, (Tseq, V), (H * V, 1),
                                       (row_off, iv * BV), (BT, BV), (1, 0))

            b_v_r  = tl.load(p_v_r,  boundary_check=(0, 1))
            b_du_r = tl.load(p_du_r, boundary_check=(0, 1))
            b_dv_r = tl.load(p_dv_r, boundary_check=(0, 1))

            b_v_beta_r  = (b_v_r * b_beta_r[:, None]).to(b_v_r.dtype)
            b_dv_beta_r = tl.dot(b_A_rr, b_du_r)
            b_dv_r      = b_dv_r + b_dv_beta_r * b_beta_r[:, None]

            # dbeta accum from dv branch:
            b_dbeta_r += tl.sum(b_dv_beta_r * b_v_r, 1)

            tl.store(p_dv_r, b_dv_r.to(p_dv_r.dtype.element_ty), boundary_check=(0, 1))

        # ---- dg (same as original using local b_A_k) ----
        b_dA_A = b_dA_r * b_A_k
        b_dg_r = tl.sum(b_dA_A, axis=1) - tl.sum(b_dA_A, axis=0)

        # ---- write dbeta/dg for this row tile ----
        p_dbeta_r = tl.make_block_ptr(base_db, (Tseq,), (H,), (row_off,), (BT,), (0,))
        p_dg_r    = tl.make_block_ptr(base_dg, (Tseq,), (H,), (row_off,), (BT,), (0,))
        tl.store(p_dbeta_r, b_dbeta_r.to(p_dbeta_r.dtype.element_ty), boundary_check=(0,))
        tl.store(p_dg_r,    b_dg_r.to(p_dg_r.dtype.element_ty),       boundary_check=(0,))


def prepare_wy_repr_bwd_expanded(
    k: torch.Tensor,          # (B, T, H, K)
    v: torch.Tensor,          # (B, T, H, V)
    g: torch.Tensor,          # (B, T, H)
    beta: torch.Tensor,       # (B, T, H)
    A_expanded: torch.Tensor, # (B, T, H, N) ; N must be multiple of 64
    dw: torch.Tensor,         # (B, T, H, K)
    du: torch.Tensor,         # (B, T, H, V)
    cu_seqlens: Optional[torch.LongTensor],
    *,
    N: int,                   # Expanded chunk length (multiple of 64)
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Backward over an expanded causal window of length N (N % 64 == 0),
    computed internally with 64-row micro-tiles.
    """
    assert N % 64 == 0, "N must be a multiple of 64"
    B, T, H, K = k.shape
    V = v.shape[-1]
    assert A_expanded.shape == (B, T, H, N), "A_expanded must be (B, T, H, N)"

    # Tiling for feature dims (keep it conservative and Triton-friendly)
    # Use min/max bounds like your existing wrapper logic but avoid external deps.
    def _next_pow2(x: int) -> int:
        return 1 if x <= 1 else 1 << (x - 1).bit_length()
    CONST_TILING = 64
    BK = min(max(_next_pow2(K), 16), CONST_TILING)
    BV = min(max(_next_pow2(V), 16), CONST_TILING)

    # Chunking grid over expanded windows
    if cu_seqlens is None:
        NT_expanded = triton.cdiv(T, N)
        chunk_indices = None
    else:
        chunk_indices = prepare_chunk_indices(cu_seqlens, N).to(torch.int32)
        NT_expanded = chunk_indices.size(0)

    # Allocate grads
    dk     = torch.zeros_like(k)
    dv     = torch.zeros_like(v)
    dbeta_ = torch.zeros_like(beta)
    dg_    = torch.zeros_like(g)

    prepare_wy_repr_bwd_kernel_expanded[(NT_expanded, B * H)](
        k=k, v=v, g=g, beta=beta, A_exp=A_expanded,
        dw=dw, du=du,
        dk=dk, dv=dv, dbeta=dbeta_, dg=dg_,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
        T=T, H=H, K=K, V=V, N=N, BK=BK, BV=BV,
    )
    return dk, dv, dbeta_, dg_
