# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_indices
from fla.utils import check_shared_mem, is_nvidia_hopper


BKV_LIST = [64, 128] if check_shared_mem() else [32, 64]
NUM_WARPS = [2, 4] if is_nvidia_hopper() else [2, 4, 8]


# =========================
# Forward: output (o)
# =========================
@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({'BK': BK, 'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in BKV_LIST
        for BV in BKV_LIST
        for num_warps in NUM_WARPS
        for num_stages in [2, 3, 4]
    ],
    key=['H', 'K', 'V', 'BT'],
)
@triton.jit(do_not_specialize=['T'])
def chunk_fwd_kernel_o(
    q,
    k,
    v,
    h,
    g,
    o,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    num_householder: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_tg = i_t
        i_n  = tl.load(chunk_indices + i_t * 2 + 0).to(tl.int32)
        i_t  = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos  = tl.load(cu_seqlens + i_n + 0).to(tl.int32)
        eos  = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T    = eos - bos
        NT   = tl.cdiv(T, BT)
    else:
        NT   = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos  = i_b * T
        eos  = i_b * T + T

    # base offsets
    q += (bos * H + i_h) * K
    k += (bos * num_householder * H + i_h) * K
    v += (bos * num_householder * H + i_h) * V
    o += (bos * H + i_h) * V
    h += (i_tg * H + i_h).to(tl.int64) * K * V

    b_o = tl.zeros([BT, BV], dtype=tl.float32)

    # O += Q @ H
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_h = tl.make_block_ptr(h, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))  # [BT,BK]
        b_h = tl.load(p_h, boundary_check=(0, 1))  # [BK,BV]
        b_o += tl.dot(b_q, b_h)                    # [BT,BV]

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    if USE_G:
        g += bos * H + i_h
        p_g = tl.make_block_ptr(g, (T,), (H,), (i_t * BT,), (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,))            # [BT]
        m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
        b_m = tl.where(m_A, tl.exp(b_g[:, None] - b_g[None, :]), 0.0)
        b_o = b_o * tl.exp(b_g)[:, None]
    else:
        b_m = ((o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)).to(tl.float32)

    # Householder steps: A = (Q @ K^T) * mask/gate; O += A @ V
    for i_dp in range(num_householder):
        b_A = tl.zeros([BT, BT], dtype=tl.float32)
        for i_k in range(tl.cdiv(K, BK)):
            p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            p_k = tl.make_block_ptr(k + i_dp * H * K, (K, T), (1, num_householder * H * K),
                                     (i_k * BK, i_t * BT), (BK, BT), (0, 1))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_A += tl.dot(b_q, b_k)
        b_A = b_A * b_m

        p_v = tl.make_block_ptr(v + i_dp * H * V, (T, V), (H * V * num_householder, 1),
                                 (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_o += tl.dot(b_A.to(b_v.dtype), b_v)

    b_o = b_o * scale
    p_o = tl.make_block_ptr(o, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


def chunk_gated_delta_product_fwd_o(
    q: torch.Tensor,                # [B, T_true, H, K] (TRUE)
    k: torch.Tensor,                # [B, T_exp,  H, K] (EXPANDED)
    v: torch.Tensor,                # [B, T_exp,  H, V] (EXPANDED)
    h: torch.Tensor,                # [B, NT_true, H, K, V] slabs indexed per true-time chunk
    g: Optional[torch.Tensor] = None,   # [B, T_true, H] (log-gates for TRUE), or None
    scale: Optional[float] = 1.0,
    cu_seqlens: Optional[torch.LongTensor] = None,  # TRUE lengths if varlen
    chunk_size: int = 64,
    num_householder: int = 1,
) -> torch.Tensor:
    """
    Forward: compute O on the TRUE timeline with gated/causal structure and Householder steps.

    Assumes k/v are expanded by num_householder along time; g is on TRUE time.
    h is provided per TRUE-time chunk (NT_true = ceil(T_true/BT)).
    """
    assert q.shape[1] * num_householder == k.shape[1], \
        "q.shape[1] * num_householder must equal k.shape[1] (expanded length)."
    scale = 1.0 if scale is None else float(scale)

    B, T_true, H, K = q.shape
    V = v.shape[-1]
    BT = min(chunk_size, max(16, triton.next_power_of_2(T_true)))
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T_true, BT) if cu_seqlens is None else len(chunk_indices)

    o = v.new_empty(B, T_true, H, V).fill_(-float('inf'))

    def grid(meta):
        return (triton.cdiv(V, meta['BV']), NT, B * H)

    chunk_fwd_kernel_o[grid](
        q, k, v, h, g, o, cu_seqlens, chunk_indices, scale,
        T=T_true, num_householder=num_householder, H=H, K=K, V=V, BT=BT,
    )
    return o


# =========================================================
# Backward: dv on EXPANDED timeline (local dv kernel)
# =========================================================
@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_G_GAMMA': lambda args: args['g_gamma'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in NUM_WARPS
        for num_stages in [2, 3, 4]
    ],
    key=['H', 'K', 'V', 'BT', 'BK', 'BV', 'USE_G'],
)
@triton.jit(do_not_specialize=['T'])
def chunk_gated_delta_product_bwd_kernel_dv_local(
    q,                # TRUE timeline
    k,                # EXPANDED
    g,                # EXPANDED (or None)
    g_gamma,          # unused
    do,               # TRUE timeline
    dv,               # EXPANDED (OUT)
    cu_seqlens,
    chunk_indices,
    scale,
    T,                         # expanded length
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,          # rows (expanded)
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_G_GAMMA: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    num_householder: tl.constexpr,
    expanded_chunk_size: tl.constexpr,   # == BT * next_power_of_2(num_householder)
):
    # program ids
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h  = i_bh // H, i_bh % H

    M = num_householder
    EXP_CHUNK = expanded_chunk_size
    BTC = (EXP_CHUNK + M - 1) // M  # ceil(EXP_CHUNK / M)

    # sequence bounds (expanded)
    if IS_VARLEN:
        i_n  = tl.load(chunk_indices + i_t * 2 + 0).to(tl.int32)
        t0   = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)  # tile id at BT granularity (expanded)
        bos  = tl.load(cu_seqlens + i_n + 0).to(tl.int32)
        eos  = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        Tloc = eos - bos
        bos_true = bos // M
        T_true   = Tloc // M
        row_start = t0 * BT
    else:
        bos  = i_b * T
        eos  = i_b * T + T
        Tloc = T
        bos_true = bos // M
        T_true   = Tloc // M
        row_start = i_t * BT

    # base pointers (batch/head offsets)
    q  += (bos_true * H + i_h) * K
    do += (bos_true * H + i_h) * V
    k  += (bos      * H + i_h) * K
    dv += (bos      * H + i_h) * V
    if USE_G:
        g  += (bos      * H + i_h)

    # coords
    o_row = row_start + tl.arange(0, BT)                    # [BT] expanded rows
    m_row = o_row < Tloc

    chunk_lo = (row_start // EXP_CHUNK) * EXP_CHUNK
    j0_true  = chunk_lo // M
    o_col_exp = j0_true * M + (M - 1) + tl.arange(0, BTC) * M
    m_col_exp = (o_col_exp < Tloc) & (o_col_exp < (chunk_lo + EXP_CHUNK))

    # A = K_rows(expanded) @ Q_cols(TRUE)  => [BT, BTC]
    b_A = tl.zeros([BT, BTC], dtype=tl.float32)
    for i_kblk in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(k, (Tloc, K), (H * K, 1),
                                (row_start, i_kblk * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))

        p_q = tl.make_block_ptr(q, (K, T_true), (1, H * K),
                                (i_kblk * BK, j0_true), (BK, BTC), (0, 1))
        b_q = tl.load(p_q, boundary_check=(0, 1))

        b_A += tl.dot(b_k, b_q)

    # causal mask in expanded coords (row <= col)
    m_tri = (o_row[:, None] <= o_col_exp[None, :]) & (m_row[:, None] & m_col_exp[None, :])

    # gating
    if USE_G:
        p_g_rows = tl.make_block_ptr(g, (Tloc,), (H,), (row_start,), (BT,), (0,))
        b_g_rows = tl.load(p_g_rows, boundary_check=(0,))                  # [BT]
        b_g_cols = tl.load(g + o_col_exp * H, mask=m_col_exp, other=0.0)   # [BTC]
        b_A = tl.where(m_tri, b_A * tl.exp(b_g_cols[None, :] - b_g_rows[:, None]) * scale, 0.0)
        b_A = b_A.to(do.dtype.element_ty)
    else:
        b_A = tl.where(m_tri, b_A * scale, 0.0).to(do.dtype.element_ty)

    # dv(expanded rows) = A [BT×BTC] @ do(TRUE) [BTC×BV]
    for i_v in range(tl.cdiv(V, BV)):
        p_do = tl.make_block_ptr(do, (T_true, V), (H * V, 1),
                                 (j0_true, i_v * BV), (BTC, BV), (1, 0))
        b_do = tl.load(p_do, boundary_check=(0, 1))  # [BTC,BV]

        p_dv = tl.make_block_ptr(dv, (Tloc, V), (H * V, 1),
                                 (row_start, i_v * BV), (BT, BV), (1, 0))
        b_dv = tl.dot(b_A.to(b_do.dtype), b_do)      # [BT,BV]
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))


def chunk_gated_delta_product_bwd_dv_local(
    q: torch.Tensor,              # [B, T_true, H, K]  (TRUE timeline)
    k: torch.Tensor,              # [B, T_exp,  H, K]  (EXPANDED timeline)
    do: torch.Tensor,             # [B, T_true, H, V]  (TRUE timeline)
    g: Optional[torch.Tensor] = None,          # [B, T_exp, H] (expanded), or None
    g_gamma: Optional[torch.Tensor] = None,    # [H], optional gamma gating (unused)
    scale: float = 1.0,
    cu_seqlens: Optional[torch.LongTensor] = None,   # expanded seqlens if varlen
    chunk_size: int = 64,
    num_householder: int = 1,
) -> torch.Tensor:
    """
    Compute dv on the EXPANDED timeline without materializing zero rows/cols.
    Varlen: pass expanded cu_seqlens (lengths on the expanded axis).
    """
    scale = 1.0 if scale is None else float(scale)
    B, T_exp, H, K = k.shape
    V = do.shape[-1]
    M = num_householder
    GRP = triton.next_power_of_2(M)
    BT = chunk_size
    expanded_chunk_size = BT * GRP

    # tiling for K/V
    if check_shared_mem('hopper', k.device.index):
        CONST_TILING = 128
    elif check_shared_mem():
        CONST_TILING = 64
    else:
        CONST_TILING = 32
    BK = min(max(triton.next_power_of_2(K), 16), CONST_TILING)
    BV = min(max(triton.next_power_of_2(V), 16), CONST_TILING)

    # expanded-time tiles at BT granularity
    if cu_seqlens is None:
        NT = triton.cdiv(T_exp, BT)
        chunk_indices = None
    else:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)  # expanded seqlens
        NT = len(chunk_indices)

    dv = torch.empty((B, T_exp, H, V), dtype=do.dtype, device=do.device)

    chunk_gated_delta_product_bwd_kernel_dv_local[(NT, B * H)](
        q=q, k=k, g=g, g_gamma=g_gamma, do=do, dv=dv,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, scale=scale,
        T=T_exp, H=H, K=K, V=V, BT=BT, BK=BK, BV=BV,
        num_householder=M, expanded_chunk_size=expanded_chunk_size,
    )
    return dv


# =========================================================
# Backward: dQ/dK/(optional dW) with grouping over expanded chunks
# =========================================================
@triton.heuristics({
    'USE_G':        lambda args: args['g'] is not None,
    'USE_G_GAMMA':  lambda args: args['g_gamma'] is not None,
    'USE_DW':       lambda args: args['dw'] is not None,
    'IS_VARLEN':    lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in NUM_WARPS
        for num_stages in [2, 3, 4]
    ],
    key=['H', 'K', 'V', 'BT', 'BK', 'BV', 'USE_G', 'USE_G_GAMMA', 'USE_DW'],
)
@triton.jit(do_not_specialize=['T'])
def chunk_bwd_kernel_dqkwg(
    q,              # [B, T_true, H, K] (TRUE timeline)
    k,              # [B, T_exp,  H, K] (EXPANDED timeline)
    v,              # [B, T_exp,  H, V] (EXPANDED)
    h,              # [B, NTG,    H, V, K] (only if USE_DW)
    g,              # [B, T_exp,  H] (EXPANDED, log-space), optional
    g_gamma,        # unused
    do,             # [B, T_true, H, V] (TRUE)
    dh,             # unused (signature parity)
    dq,             # [B, T_true, H, K] (TRUE)  OUT
    dk,             # [B, T_exp,  H, K] (EXPANDED) OUT
    w,              # [B, T_exp,  H, K] (EXPANDED), optional (for dw)
    dv,             # [B, T_exp,  H, V] (EXPANDED), optional (for dw)
    dw,             # [B, T_exp,  H, K] (EXPANDED) OUT if USE_DW
    cu_seqlens,     # [B+1] expanded lengths or None
    chunk_indices,  # not used here (API parity)
    scale,          # float
    B: tl.constexpr,
    T,              # expanded length per sequence if fixed-length
    num_householder: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,      # base tile along EXPANDED axis; group size uses GRP
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_G_GAMMA: tl.constexpr,
    USE_DW: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    NTG: tl.constexpr,     # number of expanded-chunk groups per sequence
):
    # program ids
    i_kblk, i_grp, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    # geometry
    M   = num_householder
    GRP = triton.next_power_of_2(M)
    EXP_CHUNK: tl.constexpr = BT * GRP
    BTC: tl.constexpr = (EXP_CHUNK + M - 1) // M  # ceil(EXP_CHUNK / M)

    # sequence bounds (expanded)
    if IS_VARLEN:
        bos_e = tl.load(cu_seqlens + i_b).to(tl.int32)
        eos_e = tl.load(cu_seqlens + i_b + 1).to(tl.int32)
        Tloc_e = eos_e - bos_e
        NT_grp = tl.cdiv(Tloc_e, EXP_CHUNK)
        chunk_lo = i_grp * EXP_CHUNK
        chunk_hi = tl.minimum(chunk_lo + EXP_CHUNK, Tloc_e)
    else:
        bos_e   = i_b * T
        Tloc_e  = T
        NT_grp  = tl.cdiv(Tloc_e, EXP_CHUNK)
        chunk_lo = i_grp * EXP_CHUNK
        chunk_hi = tl.minimum(chunk_lo + EXP_CHUNK, Tloc_e)

    # TRUE length
    Tloc_t = Tloc_e // M

    # base offsets per (batch, head)
    # TRUE tensors
    q  += (((bos_e // M) * H + i_h) * K)
    do += (((bos_e // M) * H + i_h) * V)
    dq += (((bos_e // M) * H + i_h) * K)
    # EXPANDED tensors
    k  += ((bos_e * H + i_h) * K)
    v  += ((bos_e * H + i_h) * V)
    dk += ((bos_e * H + i_h) * K)
    if USE_DW:
        w  += ((bos_e * H + i_h) * K)
        dv += ((bos_e * H + i_h) * V)
        dw += ((bos_e * H + i_h) * K)
        # h slab per (batch, group, head)
        h  += (((i_b * NTG + i_grp) * H + i_h) * (V * K))
    if USE_G:
        g  += ((bos_e * H + i_h))

    # strides
    stride_true_k = H * K
    stride_true_v = H * V
    stride_exp_k  = H * K
    stride_exp_v  = H * V

    # TRUE columns covered by this expanded-chunk group
    j0_true   = chunk_lo // M
    j_rel     = tl.arange(0, BTC)
    o_col_exp = chunk_lo + (M - 1) + j_rel * M
    m_cols_e  = o_col_exp < chunk_hi
    m_cols_t  = (j0_true + j_rel) < Tloc_t
    m_cols    = m_cols_e & m_cols_t
    has_cols  = tl.any(m_cols)

    # Load Q slab only if we own TRUE columns
    if has_cols:
        p_q = tl.make_block_ptr(q, (K, Tloc_t), (1, stride_true_k),
                                (i_kblk * BK, j0_true), (BK, BTC), (0, 1))
        b_q = tl.load(p_q, boundary_check=(0, 1))            # [BK, BTC]
        b_q = tl.where(m_cols[None, :], b_q, 0.0)
    else:
        b_q = tl.zeros([BK, BTC], dtype=tl.float32)

    b_dq_cols = tl.zeros([BK, BTC], dtype=tl.float32)

    # iterate expanded row tiles in this group
    for i_rt in range(GRP):
        tile_row_base = chunk_lo + i_rt * BT
        o_rows = tile_row_base + tl.arange(0, BT)
        m_rows = o_rows < chunk_hi
        if not tl.any(m_rows):
            continue

        # K rows (BT × BK)
        p_k = tl.make_block_ptr(k, (Tloc_e, K), (stride_exp_k, 1),
                                (tile_row_base, i_kblk * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))   # [BT, BK]

        # dA^T accumulator (BT × BTC)
        b_dA_T = tl.zeros([BT, BTC], dtype=tl.float32)

        # accumulate over V tiles
        if USE_DW:
            b_dw_tile = tl.zeros([BT, BK], dtype=tl.float32)

        for i_vblk in range(tl.cdiv(V, BV)):
            # V rows (BT × BV)
            p_v = tl.make_block_ptr(v, (Tloc_e, V), (stride_exp_v, 1),
                                    (tile_row_base, i_vblk * BV), (BT, BV), (1, 0))
            b_v = tl.load(p_v, boundary_check=(0, 1))      # [BT, BV]

            if has_cols:
                p_do = tl.make_block_ptr(do, (Tloc_t, V), (stride_true_v, 1),
                                         (j0_true, i_vblk * BV), (BTC, BV), (1, 0))
                b_do = tl.load(p_do, boundary_check=(0, 1))    # [BTC, BV]
                b_do = tl.where(m_cols[:, None], b_do, 0.0)
                b_dA_T += tl.dot(b_v, tl.trans(b_do))          # [BT, BTC]

            if USE_DW:
                p_dv = tl.make_block_ptr(dv, (Tloc_e, V), (stride_exp_v, 1),
                                         (tile_row_base, i_vblk * BV), (BT, BV), (1, 0))
                b_dv = tl.load(p_dv, boundary_check=(0, 1))               # [BT, BV]
                p_h  = tl.make_block_ptr(h, (V, K), (K, 1),
                                         (i_vblk * BV, i_kblk * BK), (BV, BK), (0, 1))
                b_h  = tl.load(p_h, boundary_check=(0, 1))                # [BV, BK]
                b_dw_tile += tl.dot(b_dv.to(b_h.dtype), b_h.to(b_h.dtype)) # [BT, BK]

        if has_cols:
            # gating & scaling
            if USE_G:
                p_gr = tl.make_block_ptr(g, (Tloc_e,), (H,), (tile_row_base,), (BT,), (0,))
                b_gr = tl.load(p_gr, boundary_check=(0,))                  # [BT]
                b_gc = tl.load(g + o_col_exp * H, mask=m_cols, other=0.0)  # [BTC]
                gate = tl.exp(b_gc[None, :] - b_gr[:, None])               # [BT,BTC]
                b_dA_T = b_dA_T * gate * scale
            else:
                b_dA_T = b_dA_T * scale

            # causal mask (expanded): row >= col
            m_tri = (o_rows[:, None] >= o_col_exp[None, :])
            m_rc  = (m_rows[:, None] & m_cols[None, :])
            b_dA_T = tl.where(m_tri & m_rc, b_dA_T, 0.0)

            # dQ contribution:  K^T @ dA_T
            b_dq_cols += tl.dot(tl.trans(b_k), b_dA_T.to(b_k.dtype))

            # dK tile = dA_T @ Q^T
            b_dk_tile = tl.dot(b_dA_T.to(b_q.dtype), tl.trans(b_q))  # [BT, BK]
        else:
            b_dk_tile = tl.zeros([BT, BK], dtype=tl.float32)

        # store dk (masked by active rows)
        p_dk = tl.make_block_ptr(dk, (Tloc_e, K), (stride_exp_k, 1),
                                 (tile_row_base, i_kblk * BK), (BT, BK), (1, 0))
        tl.store(p_dk, b_dk_tile.to(p_dk.dtype.element_ty),
                 mask=m_rows[:, None], boundary_check=(0, 1))

        if USE_DW:
            p_dw = tl.make_block_ptr(dw, (Tloc_e, K), (stride_exp_k, 1),
                                     (tile_row_base, i_kblk * BK), (BT, BK), (1, 0))
            tl.store(p_dw, (-b_dw_tile).to(p_dw.dtype.element_ty),
                     mask=m_rows[:, None], boundary_check=(0, 1))

    # final masked store for dQ over TRUE columns owned by this group
    if has_cols:
        p_dq_cols = tl.make_block_ptr(
            dq, (Tloc_t, K), (stride_true_k, 1),
            (j0_true, i_kblk * BK), (BTC, BK), (1, 0)
        )
        dq_tile = tl.trans(b_dq_cols).to(p_dq_cols.dtype.element_ty)  # [BTC,BK]
        tl.store(p_dq_cols, dq_tile, mask=m_cols[:, None], boundary_check=(0, 1))


def chunk_bwd_dqkwg(
    q: torch.Tensor,           # [B, T_true, H, K]  (TRUE)
    k: torch.Tensor,           # [B, T_exp,  H, K]  (EXPANDED)
    v: torch.Tensor,           # [B, T_exp,  H, V]
    do: torch.Tensor,          # [B, T_true, H, V]
    h: torch.Tensor = None,    # used only if computing dw; expected [B, NTG, H, V, K]
    dh: torch.Tensor = None,   # unused (signature parity)
    g: torch.Tensor = None,    # [B, T_exp, H] or None
    g_gamma: torch.Tensor = None,
    dv: torch.Tensor = None,   # [B, T_exp, H, V] if computing dw
    w: torch.Tensor = None,    # [B, T_exp, H, K] if computing dw
    cu_seqlens: torch.LongTensor = None,  # EXPANDED lengths or None
    chunk_size: int = 64,
    scale: float = 1.0,
    num_householder: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Backward for dQ/dK and (optionally) dW. Operates by grouping EXPANDED chunks.
    - If you want dW, pass both `w` and `dv` (and provide `h` slabs).
    - Varlen: pass EXPANDED cu_seqlens (lengths on the expanded axis).
    """
    scale = 1.0 if scale is None else float(scale)

    B, T_true, H, K = q.shape
    V = do.shape[-1]
    T_exp = k.shape[1]
    M = num_householder
    assert T_exp == T_true * M, f"Expanded length mismatch: T_exp={T_exp}, T_true*M={T_true*M}"

    # tiling
    BT = chunk_size
    GRP = triton.next_power_of_2(M)
    EXP_CHUNK = BT * GRP
    CONST_TILING = 64 if check_shared_mem() else 32
    BK = min(max(triton.next_power_of_2(K), 16), CONST_TILING)
    BV = min(max(triton.next_power_of_2(V), 16), CONST_TILING)
    NK = triton.cdiv(K, BK)

    # expanded grouping
    if cu_seqlens is None:
        chunk_indices = None
        NTG = triton.cdiv(T_exp, EXP_CHUNK)
    else:
        # cu_seqlens is expected to be EXPANDED lengths
        chunk_indices = prepare_chunk_indices(cu_seqlens, EXP_CHUNK)
        NTG = len(chunk_indices)

    grid = (NK, NTG, B * H)

    dq_out = torch.empty_like(q)
    dk_out = torch.empty_like(k)

    # determine whether to compute dw
    compute_dw = (w is not None) and (dv is not None) and (h is not None)
    dw_out: Optional[torch.Tensor] = torch.empty_like(w) if compute_dw else None

    chunk_bwd_kernel_dqkwg[grid](
        q=q, k=k, v=v,
        h=h,
        g=g, g_gamma=g_gamma, do=do, dh=dh,
        dq=dq_out, dk=dk_out,
        w=(w if compute_dw else None),
        dv=(dv if compute_dw else None),
        dw=dw_out,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
        scale=scale, B=B, T=T_exp, num_householder=M, H=H, K=K, V=V,
        BT=BT, BK=BK, BV=BV,
        NTG=NTG,
    )
    return dq_out, dk_out, dw_out
