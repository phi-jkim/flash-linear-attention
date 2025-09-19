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

def _next_pow2(x: int) -> int:
    return 1 if x <= 1 else 1 << (x - 1).bit_length()

# ---------------------------------------------------------------------------
# Forward: recompute W, U over expanded windows of width N (A_expanded[..., N])
#          using internal BT×BT tiles. BT is provided by the user.
#          Causality is enforced within the diagonal tile (strictly j < i).
# ---------------------------------------------------------------------------

@triton.heuristics({
    'USE_G':      lambda args: args['g']  is not None,
    'USE_GK':     lambda args: args['gk'] is not None,
    'IS_VARLEN':  lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    # Note: we do NOT tune BT; user must pass it explicitly.
    configs=[triton.Config({}, num_warps=nw, num_stages=ns)
             for nw in NUM_WARPS_FWD for ns in NUM_STAGES_FWD],
    key=['H', 'K', 'V', 'N', 'BT', 'BK', 'BV', 'IS_VARLEN'],  # include BT/N for cache-correctness
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
    N: tl.constexpr,          # expanded window width (columns of A_exp)
    BT: tl.constexpr,         # micro-tile height (user-provided)
    BK: tl.constexpr, BV: tl.constexpr,
    USE_G: tl.constexpr, USE_GK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    pid_exp = tl.program_id(0)   # expanded-chunk id along the sequence
    i_bh    = tl.program_id(1)   # batch*head id
    i_b     = i_bh // H
    i_h     = i_bh % H

    # Resolve bounds and expanded-chunk start (absolute indices)
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

    # Base pointers rebased at bos (sequence-local addressing after this)
    base_beta = beta  + bos * H + i_h
    base_k    = k     + (bos * H + i_h) * K
    base_v    = v     + (bos * H + i_h) * V
    base_w    = w     + (bos * H + i_h) * K
    base_u    = u     + (bos * H + i_h) * V
    base_A    = A_exp + (bos * H + i_h) * N   # (Tseq, N) with strides (H*N, 1)
    if USE_G:
        base_g  = g  + bos * H + i_h
    if USE_GK:
        base_gk = gk + (bos * H + i_h) * K

    # Number of BT-row tiles within the expanded window
    MR = tl.cdiv(N, BT)

    for r in range(MR):
        row_off_abs = t0 + r * BT
        if row_off_abs >= bos + Tseq:
            continue
        row_lo = row_off_abs - bos

        # -------- U path (over V) --------
        for iv in range(tl.cdiv(V, BV)):
            acc_u = tl.zeros((BT, BV), dtype=tl.float32)

            # accumulate from causal column tiles c <= r
            for c in range(r + 1):
                col_off_abs = t0 + c * BT
                if col_off_abs >= bos + Tseq:
                    continue
                col_lo = col_off_abs - bos

                # A[r,c] block within expanded window
                p_A_rc = tl.make_block_ptr(base_A, (Tseq, N), (H * N, 1),
                                           (row_lo, c * BT), (BT, BT), (1, 0))
                b_A_rc = tl.load(p_A_rc, boundary_check=(0, 1))

                # If this is the diagonal tile, keep strictly-lower (j < i) only
                if c == r:
                    i_abs = (t0 + r * BT) + tl.arange(0, BT)
                    j_abs = (t0 + c * BT) + tl.arange(0, BT)
                    m_i = i_abs < (bos + Tseq)
                    m_j = j_abs < (bos + Tseq)
                    m_lower = (i_abs[:, None] > j_abs[None, :]) & (m_i[:, None] & m_j[None, :])
                    b_A_rc = tl.where(m_lower, b_A_rc, 0)

                # Source rows j: v[j], beta[j]
                p_v_c = tl.make_block_ptr(base_v, (Tseq, V), (H * V, 1),
                                          (col_lo, iv * BV), (BT, BV), (1, 0))
                b_v_c = tl.load(p_v_c, boundary_check=(0, 1))

                p_beta_c = tl.make_block_ptr(base_beta, (Tseq,), (H,), (col_lo,), (BT,), (0,))
                b_beta_c = tl.load(p_beta_c, boundary_check=(0,))

                b_vb = (b_v_c * b_beta_c[:, None]).to(b_v_c.dtype)
                acc_u += tl.dot(b_A_rc, b_vb, allow_tf32=False).to(tl.float32)

            p_u_r = tl.make_block_ptr(base_u, (Tseq, V), (H * V, 1),
                                      (row_lo, iv * BV), (BT, BV), (1, 0))
            tl.store(p_u_r, acc_u.to(p_u_r.dtype.element_ty), boundary_check=(0, 1))

        # -------- W path (over K) --------
        for ik in range(tl.cdiv(K, BK)):
            acc_w = tl.zeros((BT, BK), dtype=tl.float32)

            for c in range(r + 1):
                col_off_abs = t0 + c * BT
                if col_off_abs >= bos + Tseq:
                    continue
                col_lo = col_off_abs - bos

                p_A_rc = tl.make_block_ptr(base_A, (Tseq, N), (H * N, 1),
                                           (row_lo, c * BT), (BT, BT), (1, 0))
                b_A_rc = tl.load(p_A_rc, boundary_check=(0, 1))

                # Strict lower-tri mask on diagonal tile
                if c == r:
                    i_abs = (t0 + r * BT) + tl.arange(0, BT)
                    j_abs = (t0 + c * BT) + tl.arange(0, BT)
                    m_i = i_abs < (bos + Tseq)
                    m_j = j_abs < (bos + Tseq)
                    m_lower = (i_abs[:, None] > j_abs[None, :]) & (m_i[:, None] & m_j[None, :])
                    b_A_rc = tl.where(m_lower, b_A_rc, 0)

                p_k_c = tl.make_block_ptr(base_k, (Tseq, K), (H * K, 1),
                                          (col_lo, ik * BK), (BT, BK), (1, 0))
                b_k_c = tl.load(p_k_c, boundary_check=(0, 1))

                p_beta_c = tl.make_block_ptr(base_beta, (Tseq,), (H,), (col_lo,), (BT,), (0,))
                b_kb = b_k_c * tl.load(p_beta_c, boundary_check=(0,))[:, None]

                if USE_G:
                    p_g_c = tl.make_block_ptr(base_g, (Tseq,), (H,), (col_lo,), (BT,), (0,))
                    b_kb *= tl.exp(tl.load(p_g_c, boundary_check=(0,)))[:, None]
                if USE_GK:
                    p_gk_c = tl.make_block_ptr(base_gk, (Tseq, K), (H * K, 1),
                                               (col_lo, ik * BK), (BT, BK), (1, 0))
                    b_kb *= tl.exp(tl.load(p_gk_c, boundary_check=(0, 1)))

                acc_w += tl.dot(b_A_rc, b_kb.to(b_k_c.dtype), allow_tf32=False).to(tl.float32)

            p_w_r = tl.make_block_ptr(base_w, (Tseq, K), (H * K, 1),
                                      (row_lo, ik * BK), (BT, BK), (1, 0))
            tl.store(p_w_r, acc_w.to(p_w_r.dtype.element_ty), boundary_check=(0, 1))


def recompute_w_u_expanded(
    k: torch.Tensor,            # (B, T, H, K)
    v: torch.Tensor,            # (B, T, H, V)
    beta: torch.Tensor,         # (B, T, H)
    A_expanded: torch.Tensor,   # (B, T, H, N)
    *,
    g: Optional[torch.Tensor] = None,     # (B, T, H)
    gk: Optional[torch.Tensor] = None,    # (B, T, H, K)
    cu_seqlens: Optional[torch.LongTensor] = None,
    BK: int = 64,
    BV: int = 64,
    BT: int = 64,               # REQUIRED: user-specified micro-tile height
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute W and U over expanded causal windows of width N (from A_expanded[..., N])
    by tiling that window into BT×BT blocks. The same BT should be used in backward.
    """
    B, T, H, K = k.shape
    V = v.shape[-1]
    N = A_expanded.shape[-1]
    assert A_expanded.shape == (B, T, H, N), "A_expanded must be (B, T, H, N)"
    assert BT % 16 == 0 and BT > 0, "BT must be a positive multiple of 16"

    if cu_seqlens is None:
        NT_expanded = (T + N - 1) // N
        chunk_indices = None
    else:
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

# ---------------------------------------------------------------------------
# Backward: WY-repr backward over expanded windows (width N) in BT×BT tiles
# - Computes dk, dv, dbeta, dg.
# - If gk is provided, also computes dgk (matching the forward exp(gk) gating).
# ---------------------------------------------------------------------------

@triton.heuristics({
    'USE_G':  lambda args: args['g']  is not None,
    'USE_GK': lambda args: args['gk'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[triton.Config({}, num_warps=nw, num_stages=ns)
             for nw in NUM_WARPS_BWD for ns in NUM_STAGES_BWD],
    key=['H', 'K', 'V', 'N', 'BT', 'BK', 'BV', 'IS_VARLEN', 'USE_GK'],
)
@triton.jit(do_not_specialize=['T'])
def prepare_wy_repr_bwd_kernel_expanded(
    # inputs
    k, v, g, beta, A_exp,       # k:[B,T,H,K], v:[B,T,H,V], g:[B,T,H], beta:[B,T,H], A_exp:[B,T,H,N]
    gk,                         # gk:[B,T,H,K] or None
    # upstream grads
    dw, du,                      # dw:[B,T,H,K], du:[B,T,H,V]
    # outputs
    dk, dv, dbeta, dg, dgk,      # dk/dv/dbeta/dg/dgk (dgk written iff USE_GK)
    # varlen plumbing
    cu_seqlens, chunk_indices,
    # meta
    T,
    H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    N: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    USE_G: tl.constexpr, USE_GK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    pid_exp = tl.program_id(0)
    i_bh    = tl.program_id(1)
    i_b     = i_bh // H
    i_h     = i_bh % H

    # Bounds / expanded-chunk start
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

    # Base pointers (sequence-local after bos)
    base_beta = beta + bos * H + i_h
    base_k    = k    + (bos * H + i_h) * K
    base_v    = v    + (bos * H + i_h) * V
    base_dw   = dw   + (bos * H + i_h) * K
    base_du   = du   + (bos * H + i_h) * V
    base_dk   = dk   + (bos * H + i_h) * K
    base_dv   = dv   + (bos * H + i_h) * V
    base_db   = dbeta + bos * H + i_h
    base_dg   = dg    + bos * H + i_h
    base_A    = A_exp + (bos * H + i_h) * N  # (Tseq, N), strides (H*N, 1)
    if USE_G:
        base_g = g + bos * H + i_h
    if USE_GK:
        base_gk  = gk  + (bos * H + i_h) * K
        base_dgk = dgk + (bos * H + i_h) * K

    MR = tl.cdiv(N, BT)

    for r in range(MR):
        row_off_abs = t0 + r * BT
        if row_off_abs >= bos + Tseq:
            continue
        row_lo = row_off_abs - bos

        # Load row-tile scalars
        p_beta_r = tl.make_block_ptr(base_beta, (Tseq,), (H,), (row_lo,), (BT,), (0,))
        b_beta_r = tl.load(p_beta_r, boundary_check=(0,))
        if USE_G:
            p_g_r    = tl.make_block_ptr(base_g, (Tseq,), (H,), (row_lo,), (BT,), (0,))
            b_g_r    = tl.load(p_g_r, boundary_check=(0,))
        else:
            b_g_r    = tl.zeros((BT,), dtype=tl.float32)

        # Local A_rr block
        p_A_rr = tl.make_block_ptr(base_A, (Tseq, N), (H * N, 1),
                                   (row_lo, r * BT), (BT, BT), (1, 0))
        b_A_rr = tl.load(p_A_rr, boundary_check=(0, 1))

        # Accumulate dA_r from causal columns c ≤ r
        b_dA_r = tl.zeros((BT, BT), dtype=tl.float32)

        # --- K contribution (build dA_r; also accumulate dgk if USE_GK) ---
        for ik in range(tl.cdiv(K, BK)):
            p_dw_r = tl.make_block_ptr(base_dw, (Tseq, K), (H * K, 1),
                                       (row_lo, ik * BK), (BT, BK), (1, 0))
            b_dw_r = tl.load(p_dw_r, boundary_check=(0, 1))

            for c in range(r + 1):
                col_off_abs = t0 + c * BT
                if col_off_abs >= bos + Tseq:
                    continue
                col_lo = col_off_abs - bos

                # source scalars at j-tile
                p_beta_c = tl.make_block_ptr(base_beta, (Tseq,), (H,), (col_lo,), (BT,), (0,))
                b_beta_c = tl.load(p_beta_c, boundary_check=(0,))
                if USE_G:
                    p_g_c    = tl.make_block_ptr(base_g, (Tseq,), (H,), (col_lo,), (BT,), (0,))
                    b_g_c    = tl.load(p_g_c, boundary_check=(0,))
                    b_eg_c   = tl.exp(b_g_c)
                else:
                    b_eg_c   = tl.ones((BT,), dtype=tl.float32)

                p_k_c = tl.make_block_ptr(base_k, (Tseq, K), (H * K, 1),
                                          (col_lo, ik * BK), (BT, BK), (1, 0))
                b_k_c = tl.load(p_k_c, boundary_check=(0, 1))

                # A[r,c] block (needed for dgk path)
                p_A_rc = tl.make_block_ptr(base_A, (Tseq, N), (H * N, 1),
                                           (row_lo, c * BT), (BT, BT), (1, 0))
                b_A_rc = tl.load(p_A_rc, boundary_check=(0, 1))
                # diagonal causality (strictly lower) for local algebra
                if c == r:
                    i_abs = (t0 + r * BT) + tl.arange(0, BT)
                    j_abs = (t0 + c * BT) + tl.arange(0, BT)
                    m_i = i_abs < (bos + Tseq)
                    m_j = j_abs < (bos + Tseq)
                    m_lower = (i_abs[:, None] > j_abs[None, :]) & (m_i[:, None] & m_j[None, :])
                    b_A_rc = tl.where(m_lower, b_A_rc, 0)

                # gating without gk (used by both dA and dgk terms)
                b_k_c_gate = (b_k_c * b_beta_c[:, None] * b_eg_c[:, None]).to(b_k_c.dtype)

                # add gk if used
                if USE_GK:
                    p_gk_c = tl.make_block_ptr(base_gk, (Tseq, K), (H * K, 1),
                                               (col_lo, ik * BK), (BT, BK), (1, 0))
                    b_gk_c  = tl.load(p_gk_c, boundary_check=(0, 1))
                    b_egk_c = tl.exp(b_gk_c)
                    b_k_c_gated = (b_k_c_gate * b_egk_c).to(b_k_c.dtype)
                else:
                    b_k_c_gated = b_k_c_gate

                # accumulate into dA_r from w-path
                b_dA_r += tl.dot(b_dw_r, tl.trans(b_k_c_gated)).to(tl.float32)

                # If USE_GK, accumulate dgk:
                if USE_GK:
                    # s_c = A_rc^T @ dW_r    (BT, BK)
                    s_c = tl.dot(tl.trans(b_A_rc), b_dw_r)
                    # contribution to dgk at j-tile: s_c ⊙ (k_c * β_c * exp(g_c) * exp(gk_c))
                    b_dgk_add = (s_c * b_k_c_gate * b_egk_c).to(s_c.dtype)
                    # write-back to dgk at the source tile
                    p_dgk_c = tl.make_block_ptr(base_dgk, (Tseq, K), (H * K, 1),
                                                (col_lo, ik * BK), (BT, BK), (1, 0))
                    # accumulate into existing tile
                    b_dgk_old = tl.load(p_dgk_c, boundary_check=(0, 1))
                    tl.store(p_dgk_c, (b_dgk_old + b_dgk_add).to(p_dgk_c.dtype.element_ty),
                             boundary_check=(0, 1))

        # --- V contribution (build dA_r from U path) ---
        for iv in range(tl.cdiv(V, BV)):
            p_du_r = tl.make_block_ptr(base_du, (Tseq, V), (H * V, 1),
                                       (row_lo, iv * BV), (BT, BV), (1, 0))
            b_du_r = tl.load(p_du_r, boundary_check=(0, 1))

            for c in range(r + 1):
                col_off_abs = t0 + c * BT
                if col_off_abs >= bos + Tseq:
                    continue
                col_lo = col_off_abs - bos

                p_beta_c = tl.make_block_ptr(base_beta, (Tseq,), (H,), (col_lo,), (BT,), (0,))
                b_beta_c = tl.load(p_beta_c, boundary_check=(0,))

                p_v_c = tl.make_block_ptr(base_v, (Tseq, V), (H * V, 1),
                                          (col_lo, iv * BV), (BT, BV), (1, 0))
                b_v_c = tl.load(p_v_c, boundary_check=(0, 1))

                # A[r,c] block
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

                b_v_c_gated = (b_v_c * b_beta_c[:, None]).to(b_v_c.dtype)
                b_dA_r += tl.dot(b_du_r, tl.trans(b_v_c_gated)).to(tl.float32)

        # ---- Local transforms with A_rr: dA := A * dA * A ----
        tmp    = tl.dot(b_dA_r.to(b_A_rr.dtype), b_A_rr)
        b_dA_r = tl.dot(b_A_rr, tmp.to(b_A_rr.dtype))

        # ---- Recompute STRICT lower-tri mask AFTER transform; then apply scaling ----
        i_abs = row_off_abs + tl.arange(0, BT)
        j_abs = (t0 + r * BT) + tl.arange(0, BT)
        m_i   = i_abs < (bos + Tseq)
        m_j   = j_abs < (bos + Tseq)
        m_lower = (i_abs[:, None] > j_abs[None, :]) & (m_i[:, None] & m_j[None, :])

        if USE_G:
            scale_rr = tl.exp(b_g_r[:, None] - b_g_r[None, :])
        else:
            scale_rr = tl.ones((BT, BT), dtype=tl.float32)

        b_dA_r = tl.where(m_lower, -b_dA_r * scale_rr, 0)

        # ---- SECOND PASS over K for dk/dbeta; also local A_k = (k*β) @ k^T ----
        b_A_k = tl.zeros((BT, BT), dtype=tl.float32)

        for ik in range(tl.cdiv(K, BK)):
            p_k_r  = tl.make_block_ptr(base_k,  (Tseq, K), (H * K, 1),
                                       (row_lo, ik * BK), (BT, BK), (1, 0))
            p_dk_r = tl.make_block_ptr(base_dk, (Tseq, K), (H * K, 1),
                                       (row_lo, ik * BK), (BT, BK), (1, 0))
            b_k_r  = tl.load(p_k_r,  boundary_check=(0, 1))
            b_dk_r = tl.load(p_dk_r, boundary_check=(0, 1))

            b_k_beta_r = (b_k_r * b_beta_r[:, None]).to(b_k_r.dtype)
            b_A_k += tl.dot(b_k_beta_r, tl.trans(b_k_r)).to(tl.float32)

            b_dk_beta_r = tl.dot(b_dA_r, b_k_r)
            dbeta_add   = tl.sum(b_dk_beta_r * b_k_r, 1)

            b_dk_r += tl.dot(tl.trans(b_dA_r), b_k_beta_r)
            b_dk_r += b_dk_beta_r * b_beta_r[:, None]
            tl.store(p_dk_r, b_dk_r.to(p_dk_r.dtype.element_ty), boundary_check=(0, 1))

            if ik == 0:
                b_dbeta_r = dbeta_add.to(tl.float32)
            else:
                b_dbeta_r += dbeta_add.to(tl.float32)

        # ---- DV path for this row tile ----
        for iv in range(tl.cdiv(V, BV)):
            p_v_r  = tl.make_block_ptr(base_v,  (Tseq, V), (H * V, 1),
                                       (row_lo, iv * BV), (BT, BV), (1, 0))
            p_du_r = tl.make_block_ptr(base_du, (Tseq, V), (H * V, 1),
                                       (row_lo, iv * BV), (BT, BV), (1, 0))
            p_dv_r = tl.make_block_ptr(base_dv, (Tseq, V), (H * V, 1),
                                       (row_lo, iv * BV), (BT, BV), (1, 0))

            b_v_r  = tl.load(p_v_r,  boundary_check=(0, 1))
            b_du_r = tl.load(p_du_r, boundary_check=(0, 1))
            b_dv_r = tl.load(p_dv_r, boundary_check=(0, 1))

            b_v_beta_r  = (b_v_r * b_beta_r[:, None]).to(b_v_r.dtype)
            b_dv_beta_r = tl.dot(b_A_rr, b_du_r)
            b_dv_r      = b_dv_r + b_dv_beta_r * b_beta_r[:, None]

            b_dbeta_r += tl.sum(b_dv_beta_r * b_v_r, 1)
            tl.store(p_dv_r, b_dv_r.to(p_dv_r.dtype.element_ty), boundary_check=(0, 1))

        # ---- dg from local A terms ----
        b_dA_A = b_dA_r * b_A_k
        b_dg_r = tl.sum(b_dA_A, axis=1) - tl.sum(b_dA_A, axis=0)

        # ---- write dbeta/dg ----
        p_dbeta_r = tl.make_block_ptr(base_db, (Tseq,), (H,), (row_lo,), (BT,), (0,))
        p_dg_r    = tl.make_block_ptr(base_dg,  (Tseq,), (H,), (row_lo,), (BT,), (0,))
        tl.store(p_dbeta_r, b_dbeta_r.to(p_dbeta_r.dtype.element_ty), boundary_check=(0,))
        tl.store(p_dg_r,    b_dg_r.to(p_dg_r.dtype.element_ty),       boundary_check=(0,))


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
    BT: int = 64,               # REQUIRED: user-specified micro-tile height
    BK_hint: Optional[int] = 64,
    BV_hint: Optional[int] = 64,
    gk: Optional[torch.Tensor] = None,    # (B, T, H, K) or None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Backward over the same expanded windows (width N) using BT×BT tiles.
    Returns (dk, dv, dbeta, dg, dgk). If gk is None, dgk is returned as None.
    """
    B, T, H, K = k.shape
    V = v.shape[-1]
    N = A_expanded.shape[-1]
    assert A_expanded.shape == (B, T, H, N)
    assert BT % 16 == 0 and BT > 0, "BT must be a positive multiple of 16"

    # Feature tiling heuristics (shared-mem aware)
    CONST_TILING = 64 if check_shared_mem() else 32
    BK = min(max(_next_pow2(K), 16), CONST_TILING) if BK_hint is None else min(BK_hint, CONST_TILING)
    BV = min(max(_next_pow2(V), 16), CONST_TILING) if BV_hint is None else min(BV_hint, CONST_TILING)

    if cu_seqlens is None:
        NT_expanded = (T + N - 1) // N
        chunk_indices = None
    else:
        chunk_indices = prepare_chunk_indices(cu_seqlens, N).to(torch.int32)
        NT_expanded = chunk_indices.size(0)

    dk     = torch.zeros_like(k)
    dv     = torch.zeros_like(v)
    dbeta_ = torch.zeros_like(beta)
    dg_    = torch.zeros_like(g)
    if gk is not None:
        dgk_ = torch.zeros_like(gk)
    else:
        # allocate a dummy 1-element tensor to satisfy kernel signature when not used
        dgk_ = torch.empty(1, device=k.device, dtype=k.dtype)

    grid = (NT_expanded, B * H)
    prepare_wy_repr_bwd_kernel_expanded[grid](
        k=k, v=v, g=g, beta=beta, A_exp=A_expanded,
        gk=gk,
        dw=dw, du=du,
        dk=dk, dv=dv, dbeta=dbeta_, dg=dg_, dgk=dgk_,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
        T=T, H=H, K=K, V=V, N=N, BT=BT, BK=BK, BV=BV,
    )
    if gk is None:
        return dk, dv, dbeta_, dg_, None
    return dk, dv, dbeta_, dg_, dgk_


fwd_recompute_w_u_expanded   = recompute_w_u_expanded
bwd_prepare_wy_repr_expanded = prepare_wy_repr_bwd_expanded
