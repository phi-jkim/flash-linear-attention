# -*- coding: utf-8 -*-
# Expanded-window kernels (size N) computed in user-chosen BT×BT tiles.
# Mathematically identical to gated-DeltaNet over the expanded window:
# - Forward enforces strict causality inside the diagonal BT×BT tile.
# - Backward supports dgk when gk is provided (USE_GK) and re-masks after A*dA*A.
# Copyright (c) 2023-2025

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

# Keep parity with your codebase; only prepare_chunk_indices is required here.
from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.op import exp as op_exp  # not used directly; we use tl.exp, but kept for parity
from fla.utils import check_shared_mem      # optional in BK/BV selection for bwd

# ---------------------------------------------------------------------------
# Autotune choices (define explicitly so they exist in this file)
# ---------------------------------------------------------------------------
NUM_WARPS_FWD   = (2, 4, 8)
NUM_STAGES_FWD  = (2, 3, 4)

NUM_WARPS_BWD   = (2, 4)
NUM_STAGES_BWD  = (2, 3, 4)

@triton.heuristics({
    'USE_G':      lambda args: args['g']  is not None,
    'USE_GK':     lambda args: args['gk'] is not None,
    'IS_VARLEN':  lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    # We do NOT tune BT here; caller provides BT. We tune warps/stages.
    configs=[triton.Config({}, num_warps=nw, num_stages=ns)
             for nw in NUM_WARPS_FWD for ns in NUM_STAGES_FWD],
    key=['H', 'K', 'V', 'N', 'BT', 'BK', 'BV', 'IS_VARLEN'],
)
@triton.jit(do_not_specialize=['T'])
def recompute_w_u_fwd_kernel_expanded(
    # inputs
    k, v, beta, A_exp,        # k:[B,T,H,K], v:[B,T,H,V], beta:[B,T,H], A_exp:[B,T,H,N]
    g, gk,                    # g:[B,T,H] or None, gk:[B,T,H,K] or None
    # outputs
    w, u,                     # w:[B,T,H,K], u:[B,T,H,V]
    # varlen plumbing
    cu_seqlens, chunk_indices,
    # meta
    T,                        # total seq len for fixed-length batches
    H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    N: tl.constexpr,          # expanded window width (columns in A_exp)
    BT: tl.constexpr,         # micro-tile height
    BK: tl.constexpr, BV: tl.constexpr,
    USE_G: tl.constexpr, USE_GK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    # Program ids:
    #  - pid_exp steps along expanded-window tiles (each covers N columns)
    #  - i_bh enumerates batch*head
    pid_exp = tl.program_id(0)
    i_bh    = tl.program_id(1)
    i_b     = i_bh // H
    i_h     = i_bh % H

    # Resolve sequence window and expanded-chunk start t0 (absolute index)
    if IS_VARLEN:
        i_n  = tl.load(chunk_indices + pid_exp * 2 + 0).to(tl.int32)
        it_e = tl.load(chunk_indices + pid_exp * 2 + 1).to(tl.int32)
        bos  = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos  = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        Tseq = eos - bos
        t0   = it_e * N
    else:
        bos  = i_b * T
        eos  = bos + T
        Tseq = T
        it_e = pid_exp
        t0   = it_e * N

    # Base pointers (sequence-local addressing after bos)
    base_beta = beta  + bos * H + i_h                  # (Tseq,)
    base_k    = k     + (bos * H + i_h) * K            # (Tseq,K)
    base_v    = v     + (bos * H + i_h) * V            # (Tseq,V)
    base_w    = w     + (bos * H + i_h) * K            # (Tseq,K)
    base_u    = u     + (bos * H + i_h) * V            # (Tseq,V)
    base_A    = A_exp + (bos * H + i_h) * N            # (Tseq,N) with strides (H*N, 1)
    if USE_G:
        base_g  = g  + bos * H + i_h                   # (Tseq,)
    if USE_GK:
        base_gk = gk + (bos * H + i_h) * K             # (Tseq,K)

    # Number of BT-row tiles inside the N-wide window
    MR = tl.cdiv(N, BT)

    # Iterate over BT-row tiles inside the expanded window
    for r in range(MR):
        row_off_abs = t0 + r * BT
        if row_off_abs >= bos + Tseq:
            continue
        row_lo = row_off_abs - bos  # sequence-local row offset

        # ---- row-side scalars (destination i) for this BT-row tile ----
        p_beta_r = tl.make_block_ptr(base_beta, (Tseq,), (H,), (row_lo,), (BT,), (0,))
        b_beta_r = tl.load(p_beta_r, boundary_check=(0,))                      # (BT,)

        if USE_G:
            p_g_r = tl.make_block_ptr(base_g, (Tseq,), (H,), (row_lo,), (BT,), (0,))
            b_g_r = tl.exp(tl.load(p_g_r, boundary_check=(0,)))                # (BT,)
        else:
            # Dummy value (won't be used if USE_G == 0)
            b_g_r = tl.zeros((BT,), dtype=tl.float32)

        # ------------------- U path (accumulate over V) -------------------
        for iv in range(tl.cdiv(V, BV)):
            acc_u = tl.zeros((BT, BV), dtype=tl.float32)

            # accumulate from causal column tiles c <= r
            for c in range(r + 1):
                col_off_abs = t0 + c * BT
                if col_off_abs >= bos + Tseq:
                    continue
                col_lo = col_off_abs - bos

                # A[r,c] block (BT×BT) within the expanded window
                p_A_rc = tl.make_block_ptr(base_A, (Tseq, N), (H * N, 1),
                                           (row_lo, c * BT), (BT, BT), (1, 0))
                b_A_rc = tl.load(p_A_rc, boundary_check=(0, 1))

                # Strict-lower mask on diagonal tile (keep j < i)
                if c == r:
                    i_abs = (t0 + r * BT) + tl.arange(0, BT)
                    j_abs = (t0 + c * BT) + tl.arange(0, BT)
                    m_i = i_abs < (bos + Tseq)
                    m_j = j_abs < (bos + Tseq)
                    m_lower = (i_abs[:, None] > j_abs[None, :]) & (m_i[:, None] & m_j[None, :])
                    b_A_rc = tl.where(m_lower, b_A_rc, 0)

                # Source V[j] from the column tile
                p_v_c = tl.make_block_ptr(base_v, (Tseq, V), (H * V, 1),
                                          (col_lo, iv * BV), (BT, BV), (1, 0))
                b_v_c = tl.load(p_v_c, boundary_check=(0, 1))

                # Apply row-side beta (and optional g) to the ROWS of A before dot
                if USE_G:
                    # row_scale = beta[i] * exp(g[i])  -> shape (BT,1)
                    row_scale = (b_beta_r * b_g_r)[:, None].to(b_A_rc.dtype)
                else:
                    row_scale = b_beta_r[:, None].to(b_A_rc.dtype)

                b_A_scaled = (row_scale * b_A_rc).to(b_A_rc.dtype)
                acc_u += tl.dot(b_A_scaled, b_v_c, allow_tf32=False).to(tl.float32)

            # Store the accumulated U block
            p_u_r = tl.make_block_ptr(base_u, (Tseq, V), (H * V, 1),
                                      (row_lo, iv * BV), (BT, BV), (1, 0))
            tl.store(p_u_r, acc_u.to(p_u_r.dtype.element_ty), boundary_check=(0, 1))

        # ------------------- W path (accumulate over K) -------------------
        for ik in range(tl.cdiv(K, BK)):
            acc_w = tl.zeros((BT, BK), dtype=tl.float32)

            # Preload row-side gk (if used) for this (row,BK) tile
            if USE_GK:
                p_gk_r = tl.make_block_ptr(base_gk, (Tseq, K), (H * K, 1),
                                           (row_lo, ik * BK), (BT, BK), (1, 0))
                b_gk_r = tl.exp(tl.load(p_gk_r, boundary_check=(0, 1))).to(tl.float32)  # (BT,BK)
            else:
                b_gk_r = None

            # Row-side scalar (beta * exp(g)) shaped to (BT,1)
            if USE_G:
                row_scale = (b_beta_r * b_g_r)[:, None].to(tl.float32)
            else:
                row_scale = b_beta_r[:, None].to(tl.float32)

            for c in range(r + 1):
                col_off_abs = t0 + c * BT
                if col_off_abs >= bos + Tseq:
                    continue
                col_lo = col_off_abs - bos

                # A[r,c] block with strict-lower on diagonal
                p_A_rc = tl.make_block_ptr(base_A, (Tseq, N), (H * N, 1),
                                           (row_lo, c * BT), (BT, BT), (1, 0))
                b_A_rc = tl.load(p_A_rc, boundary_check=(0, 1))
                if c == r:
                    i_abs = (t0 + r * BT) + tl.arange(0, BT)
                    j_abs = (t0 + c * BT) + tl.arange(0, BT)
                    m_i = i_abs < (bos + Tseq)
                    m_j = j_abs < (bos + Tseq)
                    m_lower = (i_abs[:, None] > j_abs[None, :]) & (m_i[:, None] & m_j[None, :])
                    b_A_rc = tl.where(m_lower, b_A_rc, 0)

                # Source K[j] from the column tile
                p_k_c = tl.make_block_ptr(base_k, (Tseq, K), (H * K, 1),
                                          (col_lo, ik * BK), (BT, BK), (1, 0))
                b_k_c = tl.load(p_k_c, boundary_check=(0, 1))

                # Apply all row-side modifiers to A before the dot
                b_A_scaled = (row_scale * b_A_rc).to(b_A_rc.dtype)         # beta, g (if any)
                if USE_GK:
                    b_A_scaled = (b_A_scaled * b_gk_r).to(b_A_scaled.dtype) # gk per-(i,BK)

                acc_w += tl.dot(b_A_scaled, b_k_c.to(b_k_c.dtype), allow_tf32=False).to(tl.float32)

            # Store the accumulated W block
            p_w_r = tl.make_block_ptr(base_w, (Tseq, K), (H * K, 1),
                                      (row_lo, ik * BK), (BT, BK), (1, 0))
            tl.store(p_w_r, acc_w.to(p_w_r.dtype.element_ty), boundary_check=(0, 1))


def recompute_w_u_expanded(
    k: torch.Tensor,            # (B, T, H, K)
    v: torch.Tensor,            # (B, T, H, V)
    beta: torch.Tensor,         # (B, T, H)
    A_expanded: torch.Tensor,   # (B, T, H, N) causal window columns assembled externally
    *,
    g: Optional[torch.Tensor] = None,     # (B, T, H)
    gk: Optional[torch.Tensor] = None,    # (B, T, H, K)
    cu_seqlens: Optional[torch.LongTensor] = None,
    BK: int = 64,
    BV: int = 64,
    BT: int = 64,               # REQUIRED: micro-tile height (must match bwd)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute W and U over expanded causal windows of width N (from A_expanded[..., N])
    by tiling that window into BT×BT blocks and accumulating over all causal column tiles.

    This matches the "original" row-side semantics:
      - Row i scalars beta[i], exp(g[i]) and (optionally) exp(gk[i,:]) are applied to rows.
      - Strict-lower mask inside diagonal tile; off-diagonal tiles are unmasked.
    """
    assert k.ndim == 4 and v.ndim == 4 and beta.ndim == 3 and A_expanded.ndim == 4
    B, T, H, K = k.shape
    V = v.shape[-1]
    N = A_expanded.shape[-1]
    assert A_expanded.shape == (B, T, H, N), "A_expanded must be (B, T, H, N)"
    assert BT % 16 == 0 and BT > 0, "BT must be a positive multiple of 16"

    if cu_seqlens is None:
        NT_expanded = (T + N - 1) // N
        chunk_indices = None
    else:
        # Each expanded chunk corresponds to an N-wide band in sequence-local indexing
        chunk_indices = prepare_chunk_indices(cu_seqlens, N).to(torch.int32)
        NT_expanded = chunk_indices.size(0)

    w = torch.empty_like(k)
    u = torch.empty_like(v)

    grid = (NT_expanded, B * H)
    recompute_w_u_fwd_kernel_expanded[grid](
        k=k, v=v, beta=beta, A_exp=A_expanded,
        g=g, gk=gk,
        w=w, u=u,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
        T=T, H=H, K=K, V=V, N=N, BT=BT, BK=BK, BV=BV,
    )
    return w, u



@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[triton.Config({}, num_warps=nw, num_stages=ns)
             for nw in NUM_WARPS_BWD for ns in NUM_STAGES_BWD],
    key=['H', 'K', 'V', 'BT', 'BK', 'BV', 'IS_VARLEN'],
)
@triton.jit(do_not_specialize=['T'])
def prepare_wy_repr_bwd_kernel_expanded(
    # inputs
    k, v, beta, g, A_exp,        # A_exp: [B, T, H, N]
    # upstream grads
    dw, du,
    # outputs
    dk, dv, dbeta, dg,
    # varlen
    cu_seqlens, chunk_indices,
    # meta
    T,
    H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    N: tl.constexpr,              # expanded-window width
    BT: tl.constexpr,             # micro-tile
    BK: tl.constexpr, BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    pid0, i_bh = tl.program_id(0), tl.program_id(1)  # along time, batch*head
    i_b, i_h = i_bh // H, i_bh % H

    # Map pid0 -> (expanded window idx, row tile r in that window)
    MR = tl.cdiv(N, BT)               # number of BT-row tiles in the window (ceiling)  (docs: tl.cdiv)
    it_e = pid0 // MR                 # expanded window index
    r    = pid0 %  MR                 # row tile inside this window

    # Resolve sequence bounds and window origin
    if IS_VARLEN:
        i_n  = tl.load(chunk_indices + it_e * 2 + 0).to(tl.int32)
        it_e = tl.load(chunk_indices + it_e * 2 + 1).to(tl.int32)
        bos  = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos  = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        Tseq = eos - bos
        t0   = it_e * N
    else:
        bos  = i_b * T
        eos  = bos + T
        Tseq = T
        t0   = it_e * N

    # Base pointers (sequence-local after bos)
    base_beta = beta + (bos * H + i_h)
    base_g    = g    + (bos * H + i_h)
    base_k    = k    + (bos * H + i_h) * K
    base_v    = v    + (bos * H + i_h) * V
    base_dw   = dw   + (bos * H + i_h) * K
    base_du   = du   + (bos * H + i_h) * V
    base_dk   = dk   + (bos * H + i_h) * K
    base_dv   = dv   + (bos * H + i_h) * V
    base_db   = dbeta + (bos * H + i_h)
    base_dg   = dg    + (bos * H + i_h)
    base_Aexp = A_exp + (bos * H + i_h) * N  # parent (Tseq, N) with strides (H*N, 1)

    # Row offset for this diagonal BT×BT tile
    row_abs = t0 + r * BT
    if row_abs >= bos + Tseq:
        return  # nothing to do (tile starts beyond sequence end)
    row_lo = row_abs - bos

    # Load per-row scalars
    p_beta = tl.make_block_ptr(base_beta, (Tseq,), (H,), (row_lo,), (BT,), (0,))
    p_g    = tl.make_block_ptr(base_g,    (Tseq,), (H,), (row_lo,), (BT,), (0,))
    b_beta = tl.load(p_beta, boundary_check=(0,))
    b_g    = tl.load(p_g,    boundary_check=(0,))
    b_g_exp = tl.exp(b_g)

    # DIFFERENCE vs baseline: load b_A from expanded window at the diagonal (r,r) tile
    # Parent is (Tseq, N), row-major on last dim; pick BT×BT starting at (row_lo, r*BT)
    p_A = tl.make_block_ptr(base_Aexp, (Tseq, N), (H * N, 1),
                            (row_lo, r * BT), (BT, BT), (1, 0))  # docs: make_block_ptr
    b_A = tl.load(p_A, boundary_check=(0, 1))                    # docs: load with boundary_check

    # Accumulators identical to baseline
    b_dbeta = tl.zeros([BT], dtype=tl.float32)
    b_dA    = tl.zeros([BT, BT], dtype=tl.float32)
    b_dg    = tl.zeros([BT], dtype=tl.float32)

    # ---- K path (first pass) ----
    for i_k in range(tl.cdiv(K, BK)):
        p_k  = tl.make_block_ptr(base_k,  (Tseq, K), (H*K, 1), (row_lo, i_k * BK), (BT, BK), (1, 0))
        p_dk = tl.make_block_ptr(base_dk, (Tseq, K), (H*K, 1), (row_lo, i_k * BK), (BT, BK), (1, 0))
        p_dw = tl.make_block_ptr(base_dw, (Tseq, K), (H*K, 1), (row_lo, i_k * BK), (BT, BK), (1, 0))

        b_k  = tl.load(p_k,  boundary_check=(0, 1))
        b_dw = tl.load(p_dw, boundary_check=(0, 1))

        b_k_beta_g = (b_k * b_beta[:, None] * b_g_exp[:, None]).to(b_k.dtype)
        b_dA      += tl.dot(b_dw, tl.trans(b_k_beta_g))
        b_dk_beta_g = tl.dot(b_A, b_dw)
        b_dk        = b_dk_beta_g * b_beta[:, None] * b_g_exp[:, None]

        b_dbeta += tl.sum(b_dk_beta_g * b_k * b_g_exp[:, None], 1)
        b_dg    += tl.sum(b_dk_beta_g * b_k * b_g_exp[:, None] * b_beta[:, None], 1)

        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))  # docs: store with boundary_check

    # ---- V path ----
    for i_v in range(tl.cdiv(V, BV)):
        p_v  = tl.make_block_ptr(base_v,  (Tseq, V), (H*V, 1), (row_lo, i_v * BV), (BT, BV), (1, 0))
        p_dv = tl.make_block_ptr(base_dv, (Tseq, V), (H*V, 1), (row_lo, i_v * BV), (BT, BV), (1, 0))
        p_du = tl.make_block_ptr(base_du, (Tseq, V), (H*V, 1), (row_lo, i_v * BV), (BT, BV), (1, 0))

        b_v  = tl.load(p_v,  boundary_check=(0, 1))
        b_du = tl.load(p_du, boundary_check=(0, 1))

        b_v_beta = (b_v * b_beta[:, None]).to(b_v.dtype)
        b_dA += tl.dot(b_du, tl.trans(b_v_beta))

        b_dv_beta = tl.dot(b_A, b_du)
        b_dv_out  = b_dv_beta * b_beta[:, None]
        tl.store(p_dv, b_dv_out.to(p_dv.dtype.element_ty), boundary_check=(0, 1))

        b_dbeta += tl.sum(b_dv_beta * b_v, 1)

    # ---- strict-lower mask (same as baseline) ----
    o_t = row_abs + tl.arange(0, BT)
    m_t = o_t < (bos + Tseq)
    m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
    b_dA = tl.where(m_A, b_dA, 0)

    # ---- local transform: dA = A * dA * A ----
    b_dA = tl.dot(b_dA.to(b_A.dtype), b_A)
    b_dA = tl.dot(b_A, b_dA.to(b_A.dtype))

    # ---- scale & re-mask ----
    b_dA = tl.where(m_A, -b_dA * tl.exp(b_g[:, None] - b_g[None, :]), 0)
    b_dA = b_dA.to(k.dtype.element_ty)

    # ---- Second K pass for dk/dbeta accumulation, plus A_k ----
    b_A_k = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_k  = tl.make_block_ptr(base_k,  (Tseq, K), (H*K, 1), (row_lo, i_k * BK), (BT, BK), (1, 0))
        p_dk = tl.make_block_ptr(base_dk, (Tseq, K), (H*K, 1), (row_lo, i_k * BK), (BT, BK), (1, 0))

        b_k  = tl.load(p_k,  boundary_check=(0, 1))
        b_dk = tl.load(p_dk, boundary_check=(0, 1))

        b_k_beta = (b_k * b_beta[:, None]).to(b_k.dtype)
        b_A_k += tl.dot(b_k_beta, tl.trans(b_k))

        b_dk_beta = tl.dot(b_dA, b_k)
        b_dbeta  += tl.sum(b_dk_beta * b_k, 1)

        b_dk += tl.dot(tl.trans(b_dA), b_k_beta)
        b_dk += b_dk_beta * b_beta[:, None]

        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))

    # ---- dg from local A terms ----
    b_dA_A = b_dA * b_A_k
    b_dg  += tl.sum(b_dA_A, axis=1) - tl.sum(b_dA_A, axis=0)

    # ---- write dbeta/dg ----
    p_db = tl.make_block_ptr(base_db, (Tseq,), (H,), (row_lo,), (BT,), (0,))
    p_dg = tl.make_block_ptr(base_dg, (Tseq,), (H,), (row_lo,), (BT,), (0,))
    tl.store(p_db, b_dbeta.to(p_db.dtype.element_ty), boundary_check=(0,))
    tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty),   boundary_check=(0,))


# ---- Python wrapper: identical signature to baseline, but takes A_expanded ----
def prepare_wy_repr_bwd_expanded(
    k: torch.Tensor,            # (B, T, H, K)
    v: torch.Tensor,            # (B, T, H, V)
    g: torch.Tensor,            # (B, T, H)
    beta: torch.Tensor,         # (B, T, H)
    A_expanded: torch.Tensor,   # (B, T, H, N)
    dw: torch.Tensor,           # (B, T, H, K)
    du: torch.Tensor,           # (B, T, H, V)
    cu_seqlens: Optional[torch.LongTensor] = None,
    *,
    chunk_size: int = 64,       # BT
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Expanded-window backward that matches baseline prepare_wy_repr_bwd math/flow
    but iterates tiles inside each expanded window. Returns (dk, dv, dbeta, dg).
    """
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = chunk_size
    N  = A_expanded.shape[-1]
    assert A_expanded.shape == (B, T, H, N)
    assert BT % 16 == 0 and BT > 0, "BT must be a positive multiple of 16"

    # Feature tiling like baseline
    CONST = 64 if check_shared_mem() else 32
    BK = min(max(_next_pow2(K), 16), CONST)
    BV = min(max(_next_pow2(V), 16), CONST)

    # Expanded windows & per-window row tiles
    if cu_seqlens is None:
        NT_expanded = (T + N - 1) // N
        chunk_indices = None
    else:
        chunk_indices = prepare_chunk_indices(cu_seqlens, N).to(torch.int32)
        NT_expanded = int(chunk_indices.size(0))

    MR = (N + BT - 1) // BT  # same as tl.cdiv(N, BT), on host

    # Launch grid: one kernel per (expanded window, row tile in window) and per (B*H)
    grid = (NT_expanded * MR, B * H)

    dk    = torch.empty_like(k)
    dv    = torch.empty_like(v)
    dbeta = torch.empty_like(beta)
    dg    = torch.empty_like(g)

    prepare_wy_repr_bwd_kernel_expanded[grid](
        k=k, v=v, beta=beta, g=g, A_exp=A_expanded,
        dw=dw, du=du,
        dk=dk, dv=dv, dbeta=dbeta, dg=dg,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
        T=T, H=H, K=K, V=V, N=N, BT=BT, BK=BK, BV=BV,
    )
    return dk, dv, dbeta, dg


fwd_recompute_w_u_expanded   = recompute_w_u_expanded
bwd_prepare_wy_repr_expanded = prepare_wy_repr_bwd_expanded
