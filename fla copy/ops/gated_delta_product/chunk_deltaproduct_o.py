# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_indices
from fla.utils import check_shared_mem, is_nvidia_hopper


BKV_LIST = [64, 128] if check_shared_mem() else [32, 64]
NUM_WARPS = [2, 4] if is_nvidia_hopper else [2, 4, 8]


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
        # causal mask (lower-triangular in true time)
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
            p_k = tl.make_block_ptr(
                k + i_dp * H * K,
                (K, T),
                (1, num_householder * H * K),
                (i_k * BK, i_t * BT),
                (BK, BT),
                (0, 1),
            )
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_A += tl.dot(b_q, b_k)
        b_A = b_A * b_m

        p_v = tl.make_block_ptr(
            v + i_dp * H * V,
            (T, V),
            (H * V * num_householder, 1),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
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
    BT = chunk_size
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T_true, BT) if cu_seqlens is None else len(chunk_indices)

    o = v.new_empty(B, T_true, H, V)

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
    BTC: tl.constexpr,                   # == ceil(expanded_chunk_size / num_householder)
):
    # program ids
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h  = i_bh // H, i_bh % H

    M = num_householder
    EXP_CHUNK = expanded_chunk_size

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
    o_col_exp = (j0_true * M + (M - 1) + tl.arange(0, BTC) * M).to(tl.int32)
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

    # gating (keep float32; cast right before dot with b_do)
    if USE_G:
        p_g_rows = tl.make_block_ptr(g, (Tloc,), (H,), (row_start,), (BT,), (0,))
        b_g_rows = tl.load(p_g_rows, boundary_check=(0,))                  # [BT]
        b_g_cols = tl.load(g + o_col_exp * H, mask=m_col_exp, other=0.0)   # [BTC]
        b_A = tl.where(m_tri, b_A * tl.exp(b_g_cols[None, :] - b_g_rows[:, None]) * scale, 0.0)
    else:
        b_A = tl.where(m_tri, b_A * scale, 0.0)

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
    BTC = (expanded_chunk_size + M - 1) // M  # ceil(expanded_chunk_size / M)

    # tiling for K/V
    if is_nvidia_hopper:
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
        num_householder=M, expanded_chunk_size=expanded_chunk_size, BTC=BTC,
    )
    return dv

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
        # causal mask (lower-triangular in true time)
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
            p_k = tl.make_block_ptr(
                k + i_dp * H * K,
                (K, T),
                (1, num_householder * H * K),
                (i_k * BK, i_t * BT),
                (BK, BT),
                (0, 1),
            )
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_A += tl.dot(b_q, b_k)
        b_A = b_A * b_m

        p_v = tl.make_block_ptr(
            v + i_dp * H * V,
            (T, V),
            (H * V * num_householder, 1),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
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
    BT = chunk_size
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T_true, BT) if cu_seqlens is None else len(chunk_indices)

    o = v.new_empty(B, T_true, H, V)

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
    BTC: tl.constexpr,                   # == ceil(expanded_chunk_size / num_householder)
):
    # program ids
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h  = i_bh // H, i_bh % H

    M = num_householder
    EXP_CHUNK = expanded_chunk_size

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
    o_col_exp = (j0_true * M + (M - 1) + tl.arange(0, BTC) * M).to(tl.int32)
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

    # gating (keep float32; cast right before dot with b_do)
    if USE_G:
        p_g_rows = tl.make_block_ptr(g, (Tloc,), (H,), (row_start,), (BT,), (0,))
        b_g_rows = tl.load(p_g_rows, boundary_check=(0,))                  # [BT]
        b_g_cols = tl.load(g + o_col_exp * H, mask=m_col_exp, other=0.0)   # [BTC]
        b_A = tl.where(m_tri, b_A * tl.exp(b_g_cols[None, :] - b_g_rows[:, None]) * scale, 0.0)
    else:
        b_A = tl.where(m_tri, b_A * scale, 0.0)

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
    BTC = (expanded_chunk_size + M - 1) // M  # ceil(expanded_chunk_size / M)

    # tiling for K/V
    if is_nvidia_hopper:
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
        num_householder=M, expanded_chunk_size=expanded_chunk_size, BTC=BTC,
    )
    return dv



# @triton.heuristics({
#     'USE_G': lambda args: args['g'] is not None,
#     'USE_G_GAMMA': lambda args: args['g_gamma'] is not None,
#     'USE_DW': lambda args: args['dw'] is not None,
#     'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
# })
# @triton.autotune(
#     configs=[
#         triton.Config({}, num_warps=num_warps, num_stages=num_stages)
#         for num_warps in NUM_WARPS
#         for num_stages in [2, 3, 4]
#     ],
#     key=['H', 'K', 'V', 'BT', 'BK', 'BV', 'USE_G', 'USE_G_GAMMA', 'USE_DW'],
# )
# @triton.jit(do_not_specialize=['T'])
# def chunk_bwd_kernel_dqkwg(
#     q,
#     k,
#     v,
#     h,
#     g,
#     g_gamma,
#     do,
#     dh,
#     dq,
#     dk,
#     dg,
#     w,
#     dv,
#     dw,
#     cu_seqlens,
#     chunk_indices,
#     scale,
#     B: tl.constexpr,
#     T,
#     num_householder: tl.constexpr,
#     H: tl.constexpr,
#     K: tl.constexpr,
#     V: tl.constexpr,
#     BT: tl.constexpr,
#     BK: tl.constexpr,
#     BV: tl.constexpr,
#     USE_G: tl.constexpr,
#     USE_G_GAMMA: tl.constexpr,
#     USE_DW: tl.constexpr,
#     IS_VARLEN: tl.constexpr,
# ):
#     # TODO use g_interleaved 
#     # i_t corresponds to index of BT * num_householder chunk 
#     i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
#     i_b, i_h = i_bh // H, i_bh % H
#     if IS_VARLEN:
#         i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
#         bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
#         all = T
#         T = eos - bos
#         i_tg = i_t
#         NT = tl.cdiv(T, BT)
#         # Calculate true sequence dimensions
#         bos_q_o = bos // num_householder
#         eos_q_o = eos // num_householder
#         T_true = T // num_householder
#         boh = tl.cdiv(bos // num_householder, BT)
#     else:
#         NT = tl.cdiv(T, BT)
#         i_tg = i_b * tl.cdiv(T // num_householder, BT) + i_t
#         bos, eos = i_b * T, i_b * T + T
#         all = B * T
#         # Calculate true sequence dimensions
#         bos_q_o = bos // num_householder
#         eos_q_o = eos // num_householder
#         T_true = T // num_householder
#         # boh = tl.cdiv(bos // num_householder, BT)
#         boh = i_b * tl.cdiv(T // num_householder, BT) 

#     # TODO FIX I_T 

#     # i_t corresponds to index of BT * num_householder chunk 
#     # 
#     i_tkw = i_t * num_householder
#     # index for which BT chunk current thread block corresponds in expanded sequence
#     # i_BT = i_t % num_householder

#     # offset calculation
#     v += (bos * H + i_h) * V 
#     do += (bos_q_o * H + i_h) * V  # do uses true sequence indexing
#     # h0, use NT to get state for each num householder * BT 
#     h += (boh * H + i_h).to(tl.int64) * K*V 
#     dh += (boh * H + i_h).to(tl.int64) * K*V 
#     # h += (bos * H + i_h).to(tl.int64) * K*V 
#     # dh += (bos * H + i_h).to(tl.int64) * K*V 
#     q += (bos_q_o * H + i_h) * K  # q uses true sequence indexing
#     k += (bos * H + i_h) * K  # k uses expanded sequence indexing
#     dq += (bos_q_o * H + i_h) * K  # dq uses true sequence indexing
#     dk += (bos * H + i_h) * K  # dk uses expanded sequence indexing

#     # for delta rule only
#     if USE_DW:
#         w += (bos * H + i_h) * K
#         dw += (bos * H + i_h) * K
#         dv += (bos * H + i_h) * V

#     if USE_G:
#         dg += i_k * all * H
#         b_dg_last = tl.zeros([1,], dtype=tl.float32) if USE_G else None
#     # if USE_G_GAMMA:
#     #     b_gamma = tl.load(g_gamma + i_h)
#     #     b_g = b_gamma * (tl.arange(0, BT) + 1)
#     #     b_g_last = b_gamma * min(BT, T - i_t * BT)
#     b_dq = tl.zeros([BT, BK], dtype=tl.float32)
#     b_dk = tl.zeros([BT, BK], dtype=tl.float32)
#     b_dw = tl.zeros([BT, BK], dtype=tl.float32) if USE_DW else None

#     if USE_G: 
#         b_dg = tl.zeros([BT], dtype=tl.float32)
#         b_dg_expanded = tl.zeros([BT*num_householder], dtype=tl.float32)

#     # Process num_householder Householder transformations
#     # Each iteration processes BT tokens from expanded sequence
#     # Load b_v blocks and compute attention scores for gradient computation
#     # TODO check out of bounds 
#     for i_nh in range(num_householder):
#         # zero out b_dk and b_dw
#         b_dk = tl.zeros([BT, BK], dtype=tl.float32)
#         b_dw = tl.zeros([BT, BK], dtype=tl.float32) if USE_DW else None
#         b_ds = tl.zeros([BT, BT], dtype=tl.float32)

#         for i_v in range(tl.cdiv(V, BV)):
#             # Load values for this Householder step - offset by i_nh 
#             p_v = tl.make_block_ptr(v, (T, V), (H * V, 1), ((i_t * num_householder + i_nh) * BT, i_v * BV), (BT, BV), (1, 0))
#             # h indexing: h is already offset to current chunk, just index by i_nh
#             p_h = tl.make_block_ptr(h + (i_t) * K * V, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1)) 

#             p_do = tl.make_block_ptr(do, (T_true, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))

#             p_dh = tl.make_block_ptr(dh + (i_t) * K * V, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1)) 
#             # [BT, BV]
#             b_v = tl.load(p_v, boundary_check=(0, 1))
#             b_do = tl.load(p_do, boundary_check=(0, 1))
#             # [BV, BK]
#             b_h = tl.load(p_h, boundary_check=(0, 1))
#             b_dh = tl.load(p_dh, boundary_check=(0, 1))
#             if USE_G:
#                 if i_nh == 0:
#                     b_dg_last += (tl.sum(b_h * b_dh))

#             # add only once throughout range num_householder 
#             if i_nh == 0: 
#                 b_dq += tl.dot(b_do, b_h.to(b_do.dtype))

#             # [BT, BV] @ [BV, BT] -> [BT, BT]
#             b_ds += tl.dot(b_do, tl.trans(b_v))
#             # Compute gradient w.r.t. k: dk += v @ dh^T
#             # [BT, BV] @ [BV, BK] -> [BT, BK]
#             b_dk += tl.dot(b_v, b_dh.to(b_v.dtype)) 
#             if USE_DW:
#                 p_dv = tl.make_block_ptr(dv, (T, V), (H*V, 1), ((i_t * num_householder + i_nh) * BT, i_v * BV), (BT, BV), (1, 0))
#                 b_dv = tl.load(p_dv, boundary_check=(0, 1))
#                 # Compute gradient w.r.t. w: dw += dv @ h^T
#                 b_dw += tl.dot(b_dv.to(b_v.dtype), b_h.to(b_v.dtype))

#         # Store gradient for this Householder step's dw
#         if USE_DW:
#             p_dw = tl.make_block_ptr(dw, (T, K), (H * K, 1), ((i_t * num_householder + i_nh) * BT, i_k * BK), (BT, BK), (1, 0))
#             tl.store(p_dw, -b_dw.to(p_dw.dtype.element_ty), boundary_check=(0, 1))

#         tl.debug_barrier()
#         p_q = tl.make_block_ptr(q, (T_true, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
#         p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), ((i_t * num_householder + i_nh) * BT, i_k * BK), (BT, BK), (1, 0))
#         b_q = tl.load(p_q, boundary_check=(0, 1))
#         b_k = tl.load(p_k, boundary_check=(0, 1))

#         p_dk = tl.make_block_ptr(dk, (T, K), (H * K, 1), ((i_t * num_householder + i_nh) * BT, i_k * BK), (BT, BK), (1, 0))

#         # Create expanded mask like in dv_local
#         o_t = i_t * BT * num_householder + tl.arange(0, BT * num_householder)
#         m_t = o_t < T
#         # m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
#         # m_A_reduced = m_A[num_householder-1::num_householder, i_nh*BT:(i_nh+1)*BT]

#         o_t_rows = i_t * BT * num_householder + (num_householder - 1) + tl.arange(0, BT) * num_householder
#         o_t_cols= i_t * BT * num_householder + i_nh * BT + tl.arange(0, BT) # i_T * BT + tl.arange(0, BT)
        
#         m_t_rows = o_t_rows < T
#         m_t_cols = o_t_cols < T
#         m_A_reduced = (o_t_rows[:, None] >= o_t_cols[None, :]) & (m_t_rows[:, None] & m_t_cols[None, :])

#         # Reduce mask to BT x BT for current Householder step
#         if USE_G:
#             g += bos * H + i_h
#             dg += bos * H + i_h
#             p_g = tl.make_block_ptr(g, (T_true,), (H,), (i_t * BT,), (BT,), (0,))
#             b_g = tl.load(p_g, boundary_check=(0,))
#             if i_nh == 0: 
#                 b_g_last = tl.load(g + (min(i_t * BT + BT, T_true) - 1) * H)
#                 b_dg_last *= exp(b_g_last)

#             # Apply gating to gradients
#             if i_nh == 0: 
#                 b_dq = b_dq * exp(b_g)[:, None] * scale
#                 b_dg += tl.sum(b_dq * b_q, axis=1)

#             # Create expanded g by repeating each element num_householder times
#             # and then extract the slice for current Householder step
#             # indices = tl.arange(0, BT) // num_householder
#             # b_g_expanded = b_g[indices] 
    
#             # b_dk = b_dk * tl.where(m_t, exp(-b_g_expanded + b_g_last), 0)[:, None]
#             # b_dg_expanded[i_nh*BT:(i_nh+1)*BT] -= tl.sum(b_k * b_dk, axis=1)
#             # b_dg_expanded[i_nh*BT + tl.arange(0, BT)] -= tl.sum(b_k * b_dk, axis=1)
#             # b_dg_last += tl.sum(b_dk * b_k)

#             # # Apply causal mask and gating to attention scores like in dv_local
#             # # Create gating mask for reduced BT x BT block
#             # indices = tl.arange(0, BT) // num_householder
#             # b_g_cols = b_g[indices]
#             # g_mask = (b_g[None, :] - b_g_cols[:, None])
#             # b_ds = tl.where(m_A_reduced, b_ds * exp(g_mask), 0) * scale
#             # b_ds2 = b_ds * tl.dot(b_q, tl.trans(b_k))
#             # b_dg_expanded[i_nh*BT:(i_nh+1)*BT] += tl.sum(b_ds2, axis=1)
#             # b_dg_expanded[i_nh*BT:(i_nh+1)*BT] -= tl.sum(b_ds2, axis=0)

#             # and then extract the slice for current Householder step
#             indices = (BT * i_nh + tl.arange(0, BT)) // num_householder
#             b_g_expanded = b_g[indices] 
    
#             # Create mask for current Householder step (BT elements)
#             o_t_nh = i_t * BT * num_householder + i_nh * BT + tl.arange(0, BT)
#             m_t_nh = o_t_nh < T
            
#             b_dk = b_dk * tl.where(m_t_nh, exp(-b_g_expanded + b_g_last), 0)[:, None]
            
#             # Accumulate gradients for this Householder step
#             b_dg_nh = tl.zeros([BT], dtype=tl.float32)
#             b_dg_nh -= tl.sum(b_k * b_dk, axis=1)
            
#             # Store in the appropriate slice of b_dg_expanded
#             for idx in range(BT):
#                 b_dg_expanded[i_nh*BT + idx] += b_dg_nh[idx]
            
#             b_dg_last += tl.sum(b_dk * b_k)

#             # Apply causal mask and gating to attention scores like in dv_local
#             # Create gating mask for reduced BT x BT block
#             # indices = tl.arange(0, BT) // num_householder
#             # TODO fix this part 
#             b_g_cols = b_g[indices]
#             g_mask = (b_g[None, :] - b_g_cols[:, None])
#             b_ds = tl.where(m_A_reduced, b_ds * exp(g_mask), 0) * scale
#             b_ds2 = b_ds * tl.dot(b_q, tl.trans(b_k))

#             b_dg += tl.sum(b_ds2, axis=1)
            
#             # Compute gradient contributions for this Householder step
#             b_dg_contrib = - tl.sum(b_ds2, axis=0)
            
#             # Store in the appropriate slice of b_dg_expanded
#             for idx in range(BT):
#                 b_dg_expanded[i_nh*BT + idx] += b_dg_contrib[idx]

#             # Gate gradients are computed from q, k, and attention scores

#             b_ds = b_ds.to(b_k.dtype)
#             # [BT, BK]
#             b_dq += tl.dot(b_ds, b_k)
#             b_dk += tl.dot(tl.trans(b_ds), b_q)
#             # p_dg = tl.make_block_ptr(dg, (T_true,), (H,), (i_t * BT,), (BT,), (0,))
#             # (SY 09/21) revcumsum in a separate kernel due to strange triton compiler issue
#             tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))

#         # elif USE_G_GAMMA:
#         #     b_dq = b_dq * exp(b_g)[:, None] * scale
#         #     b_dk = b_dk * tl.where(m_t, exp(-b_g + b_g_last), 0)[:, None]
#         #     b_ds = tl.where(m_A, b_ds * exp(b_g[:, None] - b_g[None, :]), 0) * scale
#         #     b_ds = b_ds.to(b_k.dtype)
#         #     # [BT, BK]
#         #     b_dq += tl.dot(b_ds, b_k)
#         #     b_dk += tl.dot(tl.trans(b_ds), b_q)
#         #     tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
#         #     tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))

#         else:
#             # No gating case - apply causal mask only
#             if i_nh == 0: 
#                 b_dq *= scale # scale b_do, b_h 

#             b_ds = tl.where(m_A_reduced, b_ds, 0)
#             b_ds = b_ds.to(b_k.dtype)
#             b_dq += tl.dot(b_ds, b_k) * scale 
#             b_dk += tl.dot(tl.trans(b_ds), b_q) * scale
#             # b_dq *= scale
#             tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
        
        
#     # Store final gradients
#     p_dq = tl.make_block_ptr(dq, (T_true, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
#     tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
#     if USE_G:
#         # Handle boundary condition for gate gradients
#         # Collapse b_dg_expanded to b_dg by summing gradients across num_householder repetitions
#         # If g was [1,2,3] but computed as [1,1,1,2,2,2,3,3,3], sum back to [1,2,3] size
#         for i in range(BT):
#             for i_nh in range(num_householder):
#                 b_dg[i] += b_dg_expanded[i * num_householder + i_nh] 
#         o_t_true = i_t * BT + tl.arange(0, BT)
#         b_dg = tl.where(o_t_true < min(i_t * BT + BT, T_true) - 1, b_dg, b_dg + b_dg_last)
#         p_dg = tl.make_block_ptr(dg, (T_true,), (H,), (i_t * BT,), (BT,), (0,))
#         tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0,))


# def chunk_bwd_dqkwg(
#     q: torch.Tensor,
#     k: torch.Tensor,
#     v: torch.Tensor,
#     do: torch.Tensor,
#     h: torch.Tensor,
#     dh: torch.Tensor,
#     g: Optional[torch.Tensor] = None,
#     g_gamma: Optional[torch.Tensor] = None,
#     dv: Optional[torch.Tensor] = None,
#     w: Optional[torch.Tensor] = None,
#     cu_seqlens: Optional[torch.LongTensor] = None,
#     chunk_size: int = 64,
#     scale: float = 1.0,
#     num_householder: int = 1,
# ):
#     # q and do are original (non-expanded), k and v are expanded
#     B, T_true, H, K = q.shape
#     V = do.shape[-1]
#     T = k.shape[1]  # Expanded T from k
#     assert T == T_true * num_householder, f"k.shape[1] ({T}) must equal q.shape[1] * num_householder ({T_true * num_householder})"
#     BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
#     chunk_indices = prepare_chunk_indices(cu_seqlens // num_householder, BT) if cu_seqlens is not None else None
#     NT = triton.cdiv(T // num_householder, BT) if cu_seqlens is None else len(chunk_indices)

#     CONST_TILING = 64 if check_shared_mem() else 32
#     BK = min(max(triton.next_power_of_2(K), 16), CONST_TILING)
#     BV = min(max(triton.next_power_of_2(V), 16), CONST_TILING)
#     NK = triton.cdiv(K, BK)
#     dq = torch.empty_like(q)
#     dk = torch.empty_like(k)
#     dg = torch.empty(NK, *g.shape, dtype=torch.float32, device=g.device) if g is not None else None
#     dw = torch.empty_like(w) if w is not None else None

#     grid = (NK, NT, B * H)

#     # Reduce NT to account for num_householder expansion
#     chunk_bwd_kernel_dqkwg[grid](
#         q=q,
#         k=k,
#         v=v,
#         h=h,
#         g=g,
#         g_gamma=g_gamma,
#         do=do,
#         dh=dh,
#         dv=dv,
#         w=w,
#         dw=dw,
#         dq=dq,
#         dk=dk,
#         dg=dg,
#         cu_seqlens=cu_seqlens,
#         chunk_indices=chunk_indices,
#         scale=scale,
#         B=B,
#         T=T,
#         num_householder=num_householder,
#         H=H,
#         K=K,
#         V=V,
#         BT=BT,
#         BK=BK,
#         BV=BV,
#     )

#     if dg is not None:
#         dg = dg.sum(0)
#     return dq, dk, dw, dg

@triton.heuristics({
    'USE_G':        lambda args: args['g'] is not None,
    'USE_G_GAMMA':  lambda args: args['g_gamma'] is not None,    # kept for API parity (unused here)
    'USE_DW':       lambda args: args['dw'] is not None,
    'IS_VARLEN':    lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=nw, num_stages=ns)
        for nw in [2, 4, 8]  # or NUM_WARPS
        for ns in [2, 3, 4]
    ],
    key=['H', 'K', 'V', 'BT', 'BK', 'BV', 'USE_G', 'USE_G_GAMMA', 'USE_DW'],
)
@triton.jit(do_not_specialize=['T_exp'])
def chunk_bwd_kernel_dqkwg_true_group(
    # Inputs / outputs
    q,              # [B, T_true, H, K]  (TRUE)
    k,              # [B, T_exp,  H, K]  (EXPANDED)
    v,              # [B, T_exp,  H, V]  (EXPANDED)
    h_slab,         # [B, NT_grp, H, V, K]  (group slab; V-major inside slab)
    g,              # [B, T_exp,  H] (EXPANDED, log-space), optional
    g_gamma,        # unused (parity)
    do,             # [B, T_true, H, V] (TRUE)
    dh_slab,        # [B, NT_grp, H, V, K] (group slab; V-major)
    dq,             # [B, T_true, H, K] (TRUE) OUT/acc
    dk,             # [B, T_exp,  H, K] (EXPANDED) OUT/acc
    dg,             # [NK, B, T_exp, H] (EXPANDED) OUT (K-stripes)
    w,              # [B, T_exp,  H, K] (EXPANDED), optional (for dw)
    dv,             # [B, T_exp,  H, V] (EXPANDED), optional (for dw)
    dw,             # [B, T_exp,  H, K] (EXPANDED) OUT if USE_DW
    cu_seqlens,     # expanded seqlens if varlen
    chunk_indices,  # pairs (seq_id, group_idx) over EXP_CHUNK
    chunk_offsets,  # prefix-sum of groups per sequence
    scale,          # float

    # Problem sizes / tiling
    B: tl.constexpr,
    T_exp,                         # expanded length if fixed-len
    M: tl.constexpr,               # num_householder (any)
    GRP: tl.constexpr,             # == next_power_of_2(M)
    EXP_CHUNK: tl.constexpr,       # == BT * GRP
    BTC: tl.constexpr,             # == BT * ceil(GRP / M)
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,              # base row tile on expanded axis
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_G_GAMMA: tl.constexpr,
    USE_DW: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    # ---- IDs ----
    i_k, i_grp, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b = i_bh // H
    i_h = i_bh % H

    # ---- bounds / offsets per sequence ----
    if IS_VARLEN:
        seq_id    = tl.load(chunk_indices + 2 * i_grp + 0).to(tl.int32)
        grp_inseq = tl.load(chunk_indices + 2 * i_grp + 1).to(tl.int32)
        bos       = tl.load(cu_seqlens + seq_id + 0).to(tl.int32)
        eos       = tl.load(cu_seqlens + seq_id + 1).to(tl.int32)
        Tloc      = eos - bos                       # expanded local length
        NT_grp    = tl.cdiv(Tloc, EXP_CHUNK)
        bos_true  = bos // M
        T_true    = Tloc // M
        chunk_lo  = grp_inseq * EXP_CHUNK
        chunk_hi  = tl.minimum(chunk_lo + EXP_CHUNK, Tloc)
        # TRUE cols that actually map into this expanded group
        BTC_eff   = (chunk_hi - chunk_lo + M - 1) // M
        grp_base  = tl.load(chunk_offsets + seq_id).to(tl.int32)
    else:
        Tloc      = T_exp
        NT_grp    = tl.cdiv(Tloc, EXP_CHUNK)
        bos       = i_b * Tloc
        eos       = bos + Tloc
        bos_true  = bos // M
        T_true    = Tloc // M
        grp_inseq = i_grp
        chunk_lo  = grp_inseq * EXP_CHUNK
        chunk_hi  = tl.minimum(chunk_lo + EXP_CHUNK, Tloc)
        BTC_eff   = (chunk_hi - chunk_lo + M - 1) // M  # <= BTC
        grp_base  = i_b * NT_grp

    # index of TRUE columns for this group start
    j0_true = chunk_lo // M

    # ---- base pointers (apply batch/head) ----
    # TRUE tensors
    q  += (bos_true * H + i_h) * K
    do += (bos_true * H + i_h) * V
    dq += (bos_true * H + i_h) * K

    # EXPANDED tensors
    k  += (bos * H + i_h) * K
    v  += (bos * H + i_h) * V
    dk += (bos * H + i_h) * K
    if USE_G:
        g += (bos * H + i_h)
    if USE_DW:
        w  += (bos * H + i_h) * K
        dv += (bos * H + i_h) * V
        dw += (bos * H + i_h) * K

    # slabs per TRUE-group (V-major inside slab)
    h_slab  += ((grp_base + grp_inseq) * H + i_h).to(tl.int64) * K * V
    dh_slab += ((grp_base + grp_inseq) * H + i_h).to(tl.int64) * K * V

    if USE_G: 
        # dg NK stripe base: [NK, B, T_exp, H] → stride between stripes = B*T_exp*H
        dg += i_k * (B * T_exp) * H
        dg += bos * H + i_h  # move into (batch, head) + expanded offset

    # TRUE columns mapped to expanded last-of-householder indices
    # o_cols_exp[j] = chunk_lo + (M-1) + j*M
    j_rel      = tl.arange(0, BTC)
    o_cols_exp = (j0_true * M + (M - 1)) + j_rel * M                         # [BTC]
    # valid TRUE rows to touch (in-range and within sequence tail)
    m_cols = (o_cols_exp >= chunk_lo) & (o_cols_exp < chunk_hi) & ((j0_true + j_rel) < T_true)

    # --------------------------
    # 1) direct dq (once / group)
    # --------------------------
    b_dq_grp = tl.zeros([BTC, BK], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        # do TRUE block [BTC,BV]
        p_do = tl.make_block_ptr(do, (T_true, V), (H * V, 1),
                                 (j0_true, i_v * BV), (BTC, BV), (1, 0))
        b_do = tl.load(p_do, boundary_check=(0, 1))  # [BTC,BV]

        # h slab (V-major): read as (K,V) → [BK,BV]
        p_h  = tl.make_block_ptr(h_slab, (K, V), (V, 1),
                                 (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_h  = tl.load(p_h, boundary_check=(0, 1))   # [BK,BV]

        # [BTC,BV] @ [BV,BK] = [BTC,BK]
        b_dq_grp += tl.dot(b_do, tl.trans(b_h).to(b_do.dtype))

    # gate on TRUE columns for direct dq
    if USE_G:
        b_g_cols = tl.load(g + o_cols_exp * H, mask=m_cols, other=0.0)  # [BTC]
        # expand to [BTC,1] then scale rows
        b_dq_grp = b_dq_grp * tl.exp(b_g_cols[:, None]) * scale
    else:
        b_dq_grp = b_dq_grp * scale

    # tail mask rows (stay in regs; final masked store below)
    b_dq_grp = b_dq_grp * m_cols[:, None].to(b_dq_grp.dtype)
    # rows that are not part of index are zeroed 

    # direct column-side dG (TRUE side), once per group
    b_dg_cols_grp = tl.zeros([BTC], dtype=tl.float32)
    p_q_dir = tl.make_block_ptr(q, (K, T_true), (1, H * K),
                                (i_k * BK, j0_true), (BK, BTC), (0, 1))
    b_q_dir = tl.load(p_q_dir, boundary_check=(0, 1))                 # [BK,BTC]
    b_dg_cols_grp += tl.sum((tl.trans(b_q_dir) * b_dq_grp), axis=1)   # [BTC]

    # For the "last" row scalar (group-level)
    b_dg_last_acc = tl.zeros([1], dtype=tl.float32)

    # --------------------------------------------------
    # 2) per expanded sub-tile: dk, attention path, dG rows
    # --------------------------------------------------
    for i_nh in range(GRP):
        row_start = chunk_lo + i_nh * BT
        BT_eff = tl.minimum(BT, tl.maximum(0, Tloc - row_start))
        if BT_eff <= 0:
            continue

        b_dk      = tl.zeros([BT, BK], dtype=tl.float32)
        b_ds      = tl.zeros([BT, BTC], dtype=tl.float32)
        b_dg_rows = tl.zeros([BT], dtype=tl.float32)
        if USE_DW:
            b_dw = tl.zeros([BT, BK], dtype=tl.float32)

        # V loop: dk direct + build ds
        for i_v in range(tl.cdiv(V, BV)):
            # expanded rows (v)
            p_v = tl.make_block_ptr(v, (Tloc, V), (H * V, 1),
                                    (row_start, i_v * BV), (BT, BV), (1, 0))
            b_v = tl.load(p_v, boundary_check=(0, 1))                  # [BT,BV]

            # TRUE block (do)
            p_do = tl.make_block_ptr(do, (T_true, V), (H * V, 1),
                                     (j0_true, i_v * BV), (BTC, BV), (1, 0))
            b_do = tl.load(p_do, boundary_check=(0, 1))                # [BTC,BV]

            # dh slab
            p_dh = tl.make_block_ptr(dh_slab, (K, V), (V, 1),
                                     (i_k * BK, i_v * BV), (BK, BV), (1, 0))
            b_dh = tl.load(p_dh, boundary_check=(0, 1))                # [BK,BV]

            # dk direct: [BT,BV] @ [BV,BK]
            b_dk += tl.dot(b_v, tl.trans(b_dh).to(b_v.dtype))
            # ds build: [BT,BV] @ [BV,BTC]
            b_ds += tl.dot(b_v, tl.trans(b_do))

            if USE_DW:
                p_dv = tl.make_block_ptr(dv, (Tloc, V), (H * V, 1),
                                         (row_start, i_v * BV), (BT, BV), (1, 0))
                b_dv = tl.load(p_dv, boundary_check=(0, 1))
                p_h = tl.make_block_ptr(h_slab, (K, V), (V, 1),
                                        (i_k * BK, i_v * BV), (BK, BV), (1, 0))
                b_h = tl.load(p_h, boundary_check=(0, 1))
                b_dw += tl.dot(b_dv.to(b_v.dtype), tl.trans(b_h).to(b_v.dtype))

        if USE_DW:
            p_dw = tl.make_block_ptr(dw, (Tloc, K), (H * K, 1),
                                     (row_start, i_k * BK), (BT, BK), (1, 0))
            tl.store(p_dw, (-b_dw).to(p_dw.dtype.element_ty), boundary_check=(0, 1))

        # coords/masks for this sub-tile
        o_rows = row_start + tl.arange(0, BT)   # [BT] expanded rows
        m_rows = o_rows < Tloc

        # load q TRUE block and k rows for attention projection
        p_q = tl.make_block_ptr(q, (K, T_true), (1, H * K),
                                (i_k * BK, j0_true), (BK, BTC), (0, 1))
        b_q = tl.load(p_q, boundary_check=(0, 1))                       # [BK,BTC]
        p_k = tl.make_block_ptr(k, (Tloc, K), (H * K, 1),
                                (row_start, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))                       # [BT,BK]

        # causal mask on (expanded rows, TRUE cols)
        m_tri = (o_rows[:, None] <= o_cols_exp[None, :]) & (m_rows[:, None] & m_cols[None, :])

        # gating on ds and dk rows
        if USE_G:
            p_g_rows = tl.make_block_ptr(g, (Tloc,), (H,), (row_start,), (BT,), (0,))
            b_g_rows = tl.load(p_g_rows, boundary_check=(0,))                 # [BT]
            b_g_cols = tl.load(g + o_cols_exp * H, mask=m_cols, other=0.0)    # [BTC]
            last_idx = chunk_hi - 1
            b_g_last = tl.load(g + last_idx * H)

            # BT x BTC 
            # ds gate + mask
            b_ds = tl.where(m_tri, b_ds * tl.exp(b_g_cols[None, :] - b_g_rows[:, None]) * scale, 0.0)

            # dk rows gate
            b_dk = b_dk * tl.where(m_rows, tl.exp(-b_g_cols + b_g_last), 0.0)[:, None]
        else:
            b_ds = tl.where(m_tri, b_ds * scale, 0.0)

        # project for gate grads: s = (q^T @ k^T)^T
        s = tl.dot(tl.trans(b_q), tl.trans(b_k))    # [BTC,BT]
        b_ds2 = b_ds * tl.trans(s)                  # [BT,BTC]

        if USE_G:
            # attention-path gate grads
            b_dg_rows = b_dg_rows - tl.sum(b_ds2, axis=1)        # rows negative
            b_dg_cols_grp += tl.sum(b_ds2, axis=0)               # cols positive

            # row grads from dk direct
            b_dg_rows = b_dg_rows - tl.sum(b_k * b_dk, axis=1)

            # accumulate last-index part from dk rows (added later at last_idx)
            b_dg_last_acc += tl.sum(b_dk * b_k)

        # attention contributions to dq/dk
        b_dq_grp += tl.dot(tl.trans(b_ds).to(b_k.dtype), b_k)   # [BTC,BK]
        b_dk     += tl.dot(b_ds.to(b_q.dtype), tl.trans(b_q))   # [BT,BK]

        # store dk rows
        p_dk = tl.make_block_ptr(dk, (Tloc, K), (H * K, 1),
                                 (row_start, i_k * BK), (BT, BK), (1, 0))
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))

        if USE_G:
            # write row dG for this sub-tile (expanded)
            p_dg_rows = tl.make_block_ptr(dg, (Tloc,), (H,), (row_start,), (BT,), (0,))
            tl.store(p_dg_rows, b_dg_rows.to(p_dg_rows.dtype.element_ty), boundary_check=(0,))
    # end for i_nh

    # group “last” scalar (once) = (sum(h*dh) + sum_rows_k(b_dk*b_k)) * exp(g_last)
    if USE_G:
        b_dg_last_total = tl.zeros([1], dtype=tl.float32)
        for i_v in range(tl.cdiv(V, BV)):
            p_h  = tl.make_block_ptr(h_slab,  (K, V), (V, 1),
                                     (i_k * BK, i_v * BV), (BK, BV), (1, 0))
            p_dh = tl.make_block_ptr(dh_slab, (K, V), (V, 1),
                                     (i_k * BK, i_v * BV), (BK, BV), (1, 0))
            b_h  = tl.load(p_h,  boundary_check=(0, 1))
            b_dh = tl.load(p_dh, boundary_check=(0, 1))
            b_dg_last_total += tl.sum(b_h * b_dh)
        b_dg_last_total += b_dg_last_acc
        last_idx = chunk_hi - 1
        b_g_last = tl.load(g + last_idx * H)
        b_dg_last_total *= tl.exp(b_g_last)
        tl.atomic_add(dg + last_idx * H, b_dg_last_total[0])

        # scatter column dG (TRUE → expanded last-of-householder) via masked atomics
        # (add 0.0 outside group to avoid dynamic if)
        for j in range(BTC):
            addval = tl.where(m_cols[j], b_dg_cols_grp[j], 0.0)
            tl.atomic_add(dg + o_cols_exp[j] * H, addval)

    # --------------------------------------------
    # SAFE dq store: masked block store (no races)
    # --------------------------------------------
    # expand mask to [BTC,BK] without Python branching
    mask_rows = m_cols[:, None]
    mask_k = tl.arange(0, BK)[None, :] == tl.arange(0, BK)[None, :]
    mask_store = mask_rows & mask_k
    p_dq_blk = tl.make_block_ptr(dq, (T_true, K), (H * K, 1),
                                 (j0_true, i_k * BK), (BTC, BK), (1, 0))
    tl.store(p_dq_blk, b_dq_grp.to(p_dq_blk.dtype.element_ty),
             boundary_check=(0, 1), mask=mask_store)


def chunk_bwd_dqkwg(
    q: torch.Tensor,           # [B, T_true, H, K]  (TRUE)
    k: torch.Tensor,           # [B, T_exp,  H, K]  (EXPANDED)
    v: torch.Tensor,           # [B, T_exp,  H, V]  (EXPANDED)
    do: torch.Tensor,          # [B, T_true, H, V]  (TRUE)
    h: torch.Tensor,           # [B, NT_grp, H, V, K]  group slabs (V-major inside slab)
    dh: torch.Tensor,          # [B, NT_grp, H, V, K]
    g: torch.Tensor | None = None,           # [B, T_exp, H] or None (log-space)
    g_gamma: torch.Tensor | None = None,     # unused
    dv: torch.Tensor | None = None,          # [B, T_exp, H, V] if dw
    w: torch.Tensor | None = None,           # [B, T_exp, H, K] if dw
    cu_seqlens: torch.LongTensor | None = None,  # expanded lengths or None
    chunk_size: int = 64,
    scale: float = 1.0,
    num_householder: int = 1,                # M
):
    """
    TRUE-group launcher (grid over NT_grp). Computes dq, dk, (optional) dw, dg.
    Handles any M by grouping with GRP = next_power_of_2(M). BTC is chosen as the
    *correct* upper bound: BT * ceil(GRP/M).
    """

    B, T_exp, H, K = k.shape
    V = do.shape[-1]
    M = int(num_householder)

    # Derive TRUE length and verify expanded length multiple-of-M when using interleaving layout.
    T_true = q.shape[1]
    # (Allow mismatch in tail/padded cases; kernel masks ensure safety.)

    # Tiling
    BT = int(chunk_size)
    if 'is_nvidia_hopper' in globals() and is_nvidia_hopper():
        CONST_TILING = 128
    elif 'check_shared_mem' in globals() and check_shared_mem():
        CONST_TILING = 64
    else:
        CONST_TILING = 32
    BK = min(max(triton.next_power_of_2(K), 16), CONST_TILING)
    BV = min(max(triton.next_power_of_2(V), 16), CONST_TILING)
    NK = triton.cdiv(K, BK)

    # Expanded-group geometry
    GRP = triton.next_power_of_2(M)            # rows per TRUE chunk multiplier
    EXP_CHUNK = BT * GRP                        # expanded rows per group
    # RATIO = (GRP + M - 1) // M                  # ceil(GRP / M) ∈ {1,2}
    # BTC = BT * RATIO                            # TRUE cols per group (upper bound)

    BTC = tl.cdiv(EXP_CHUNK, M)

    # Build expanded-time group indices at EXP_CHUNK granularity
    if cu_seqlens is None:
        NT_grp = triton.cdiv(T_exp, EXP_CHUNK)
        chunk_indices_dev = None
        chunk_offsets_dev = None
    else:
        # Both helpers index on expanded lengths
        chunk_indices_dev = prepare_chunk_indices(cu_seqlens, EXP_CHUNK)   # (N_groups, 2)
        chunk_offsets_dev = prepare_chunk_offsets(cu_seqlens, EXP_CHUNK)   # (N_seqs,)
        NT_grp = len(chunk_indices_dev)

    # Outputs
    dq_out = torch.empty_like(q)
    dk_out = torch.empty_like(k)
    dg_striped = (torch.empty((NK, *g.shape), dtype=torch.float32, device=g.device)
                  if g is not None else None)
    dw_out = torch.empty_like(w) if (w is not None) else None

    grid = (NK, NT_grp, B * H)
    chunk_bwd_kernel_dqkwg_true_group[grid](
        q=q,
        k=k,
        v=v,
        h_slab=h,
        g=g,
        g_gamma=g_gamma,
        do=do,
        dh_slab=dh,
        dq=dq_out,
        dk=dk_out,
        dg=dg_striped,
        w=w,
        dv=dv,
        dw=dw_out,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices_dev,
        chunk_offsets=chunk_offsets_dev,
        scale=float(scale),
        #
        B=B,
        T_exp=T_exp,
        M=M,
        GRP=GRP,
        EXP_CHUNK=EXP_CHUNK,
        BTC=BTC,
        H=H, K=K, V=V,
        BT=BT, BK=BK, BV=BV,
    )

    dg_out = dg_striped.sum(0) if dg_striped is not None else None
    return dq_out, dk_out, dw_out, dg_out
