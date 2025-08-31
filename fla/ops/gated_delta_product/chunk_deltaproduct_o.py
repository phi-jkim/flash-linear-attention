# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional

import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.op import exp
from fla.utils import check_shared_mem, is_nvidia_hopper

BKV_LIST = [64, 128] if check_shared_mem() else [32, 64]
NUM_WARPS = [2, 4] if is_nvidia_hopper else [2, 4, 8]


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
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    # offset calculation
    q += (bos * H + i_h) * K
    k += (bos * num_householder * H + i_h) * K
    v += (bos * num_householder * H + i_h) * V
    o += (bos * H + i_h) * V
    h += (i_tg * H + i_h).to(tl.int64) * K*V

    b_o = tl.zeros([BT, BV], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_h = tl.make_block_ptr(h, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        # [BT, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        # [BK, BV]
        b_h = tl.load(p_h, boundary_check=(0, 1))
        # [BT, BK] @ [BK, BV] -> [BT, BV]
        b_o += tl.dot(b_q, b_h)

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    if USE_G:
        g += bos * H + i_h
        p_g = tl.make_block_ptr(g, (T,), (H,), (i_t * BT,), (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,))
        m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
        b_m = tl.where(m_A, exp(b_g[:, None] - b_g[None, :]), 0)
        b_o = b_o * exp(b_g)[:, None]
    else:
        b_m = ((o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)).to(tl.float32)

    for i_dp in range(num_householder):
        b_A = tl.zeros([BT, BT], dtype=tl.float32)
        for i_k in range(tl.cdiv(K, BK)):
            p_q = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            p_k = tl.make_block_ptr(k+i_dp*H*K, (K, T), (1, num_householder*H*K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
            # [BT, BK]
            b_q = tl.load(p_q, boundary_check=(0, 1))
            # [BK, BT]
            b_k = tl.load(p_k, boundary_check=(0, 1))
            # [BT, BK] @ [BK, BT] -> [BT, BT]
            b_A += tl.dot(b_q, b_k)
        b_A = b_A * b_m
        p_v = tl.make_block_ptr(v+i_dp*H*V, (T, V), (H*V*num_householder, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_o += tl.dot(b_A.to(b_v.dtype), b_v)
    b_o = b_o * scale
    p_o = tl.make_block_ptr(o, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


def chunk_gated_delta_product_fwd_o(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    g: Optional[torch.Tensor] = None,  # cumsum of log decay
    scale: Optional[float] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_size: int = 64,
    num_householder: int = 1,
) -> torch.Tensor:
    assert q.shape[1] * num_householder == k.shape[1], "q.shape[1] * num_householder must be equal to k.shape[1]"
    B, T, H, K, V = *q.shape, v.shape[-1]
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    o = v.new_empty(B, T, H, V).fill_(-float('inf'))
    def grid(meta): return (triton.cdiv(V, meta['BV']), NT, B * H)
    chunk_fwd_kernel_o[grid](
        q,
        k,
        v,
        h,
        g,
        o,
        cu_seqlens,
        chunk_indices,
        scale,
        T=T,
        num_householder=num_householder,
        H=H,
        K=K,
        V=V,
        BT=BT,
    )
    return o

def chunk_gated_delta_product_bwd_dv_local(
    q: torch.Tensor,
    k: torch.Tensor,
    do: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    g_gamma: Optional[torch.Tensor] = None,
    scale: float = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_size: int = 64,
    num_householder: int = 1,
) -> torch.Tensor:
    B, T, H, K, V = *k.shape, do.shape[-1]
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    # H100 can have larger block size
    if check_shared_mem('hopper', k.device.index):
        CONST_TILING = 128
    elif check_shared_mem:
        CONST_TILING = 64
    else:
        CONST_TILING = 32
    BK = min(max(triton.next_power_of_2(K), 16), CONST_TILING)
    BV = min(max(triton.next_power_of_2(V), 16), CONST_TILING)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    dv = torch.zeros((B, T, H, V), dtype=torch.float32, device=k.device)
    grid = (NT, B * H)
    chunk_gated_delta_product_bwd_kernel_dv_local[grid](
        q=q,
        k=k,
        g=g,
        g_gamma=g_gamma,
        do=do,
        dv=dv,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        num_householder=num_householder,
    )
    return dv


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
    q,
    k,
    g,
    g_gamma,
    do,
    dv,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_G_GAMMA: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    num_householder: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos 

        bos_q_o = bos // num_householder
        eos_q_o = eos // num_householder
        
        T_true = T // num_householder
    else:
        bos, eos = i_b * T, i_b * T + T
        T = T
        T_true = T // num_householder

        bos_q_o = bos // num_householder
        eos_q_o = eos // num_householder

    # i_t corresponds to BT actual tokens (0, 1, 2, 3, 4, 5, 6) num householder = 3 
    # i_tq corresponds to BT * num householder actual tokens 
    i_tq_o = i_t // num_householder # 0, 1, 2 should correspond to 0 and 3, 4, 5 correspond to 1 

    # index for which BT chunk current thread block corresponds in a block of BT * num householder chunks 
    i_BT = i_t % num_householder

    # offset calculation
    # q += (bos * H + i_h) * K
    q += (bos_q_o * H + i_h) * K
    k += (bos * H + i_h) * K
    # do += (bos * H + i_h) * V
    do += (bos_q_o * H + i_h) * V
    dv += (bos * H + i_h) * V

    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        # p_q = tl.make_block_ptr(q, (K, T_true), (1, H*K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_q = tl.make_block_ptr(q, (K, T_true), (1, H*K), (i_k * BK, i_tq_o * BT), (BK, BT), (0, 1))

        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_A += tl.dot(b_k, b_q)

    if USE_G:
        g += bos_q_o * H + i_h
        p_g = tl.make_block_ptr(g, (T_true,), (H,), (i_tq_o * BT,), (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,))

    # if USE_G_GAMMA:
    #     b_gamma = tl.load(g_gamma + i_h)
    #     b_g = b_gamma * (tl.arange(0, BT) + 1)

    # transpose of mask 
    # o_t = i_tq_o * BT * num_householder + tl.arange(0, BT * num_householder)
    # m_t = o_t < T
    # m_A = (o_t[:, None] <= o_t[None, :]) & (m_t[:, None] & m_t)
    # m_A_reduced = m_A[i_BT*BT:(i_BT+1)*BT, num_householder-1::num_householder]
    
    o_t_rows = i_tq_o * BT * num_householder + i_BT * BT + tl.arange(0, BT)
    o_t_cols = i_tq_o * BT * num_householder + (num_householder - 1) + tl.arange(0, BT) * num_householder
    
    m_t_rows = o_t_rows < T
    m_t_cols = o_t_cols < T
    m_A_reduced = (o_t_rows[:, None] <= o_t_cols[None, :]) & (m_t_rows[:, None] & m_t_cols[None, :])
    
    if USE_G:
        # BT x BT 
        g_mask = (b_g[:, None] - b_g[None, :]) 
        # g_mask = g_mask.repeat_interleave(num_householder, dim=1)
        # g_mask = tl.trans(g_mask[:, i_BT*BT:(i_BT+1)*BT]) 
        ones3d = tl.zeros([BT, BT, num_householder], dtype=g_mask.dtype) + 1
        g_mask_expanded = tl.reshape(g_mask[:, :, None] * ones3d, [BT, BT * num_householder])

        start = i_BT * BT
        cols  = start + tl.arange(0, BT)        # shape [BT]
        g_mask_expanded = tl.trans(g_mask_expanded[:, cols])   # shape [BT, BT]

        b_A = tl.where(m_A_reduced, b_A * exp(g_mask_expanded) * scale, 0).to(do.dtype.element_ty)
    else:
        b_A = tl.where(m_A_reduced, b_A * scale, 0).to(do.dtype.element_ty)

    for i_v in range(tl.cdiv(V, BV)):
        # p_do = tl.make_block_ptr(do, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_do = tl.make_block_ptr(do, (T_true, V), (H*V, 1), (i_tq_o * BT, i_v * BV), (BT, BV), (1, 0))
        p_dv = tl.make_block_ptr(dv, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_dv = tl.dot(b_A.to(b_do.dtype), b_do)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))


# @triton.heuristics({
#     'USE_G': lambda args: args['g'] is not None,
#     'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
# })
# @triton.autotune(
#     configs=[
#         triton.Config({'BK': BK, 'BV': BV}, num_warps=num_warps, num_stages=num_stages)
#         for BK in BKV_LIST
#         for BV in BKV_LIST
#         for num_warps in NUM_WARPS
#         for num_stages in [2, 3, 4]
#     ],
#     key=['H', 'K', 'V', 'BT'],
# )
# @triton.jit(do_not_specialize=['T'])
# def chunk_gated_delta_product_bwd_kernel_o(
#     q,
#     k,
#     v,
#     h,
#     g,
#     do,
#     dq,
#     dk,
#     dv,
#     dh,
#     cu_seqlens,
#     chunk_indices,
#     scale,
#     T,
#     num_householder: tl.constexpr,
#     H: tl.constexpr,
#     K: tl.constexpr,
#     V: tl.constexpr,
#     BT: tl.constexpr,
#     BK: tl.constexpr,
#     BV: tl.constexpr,
#     USE_G: tl.constexpr,
#     IS_VARLEN: tl.constexpr,
# ):
#     # same parameters as forward pass
#     i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
#     i_b, i_h = i_bh // H, i_bh % H

#     if IS_VARLEN:
#         i_tg = i_t
#         i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
#         bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
#         T = eos - bos
#         NT = tl.cdiv(T, BT)
#     else:
#         NT = tl.cdiv(T, BT)
#         i_tg = i_b * NT + i_t
#         bos, eos = i_b * T, i_b * T + T

#     # offset calculation
#     q += (bos * H + i_h) * K
#     k += (bos * H + i_h) * K
#     v += (bos * H + i_h) * V
#     do += (bos * H + i_h) * V
#     dq += (bos * H + i_h) * K
#     dk += (bos * H + i_h) * K
#     dv += (bos * H + i_h) * V
#     h += (i_tg * H + i_h).to(tl.int64) * K*V
#     dh += (i_tg * H + i_h).to(tl.int64) * K*V

#     b_dq = tl.zeros([BT, BK], dtype=tl.float32)
#     b_dk = tl.zeros([BT, BK], dtype=tl.float32)
#     b_dv = tl.zeros([BT, BV], dtype=tl.float32)
#     b_ds = tl.zeros([BT, BT], dtype=tl.float32)

#     # Compute gradients from hidden state
#     for i_k in range(tl.cdiv(K, BK)):
#         p_q = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
#         p_h = tl.make_block_ptr(h, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
#         p_dh = tl.make_block_ptr(dh, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
#         p_do = tl.make_block_ptr(do, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))

#         # [BT, BK]
#         b_q = tl.load(p_q, boundary_check=(0, 1))
#         # [BK, BV]
#         b_h = tl.load(p_h, boundary_check=(0, 1))
#         b_dh = tl.load(p_dh, boundary_check=(0, 1))
#         # [BT, BV]
#         b_do = tl.load(p_do, boundary_check=(0, 1))

#         # Compute gradients w.r.t. q: dq += do @ h^T
#         b_dq += tl.dot(b_do, tl.trans(b_h))

#         # Compute gradients w.r.t. h: dh += q^T @ do
#         tl.store(p_dh, (b_dh + tl.dot(tl.trans(b_q), b_do)).to(p_dh.dtype.element_ty), boundary_check=(0, 1))

#     # Process multiple Householder transformations
#     for i_dp in range(num_householder):
#         b_A = tl.zeros([BT, BT], dtype=tl.float32)

#         # Compute attention matrix A = Q @ K^T for this Householder step
#         for i_k in range(tl.cdiv(K, BK)):
#             p_q = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
#             p_k = tl.make_block_ptr(k+i_dp*H*K, (K, T), (1, num_householder*H*K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))

#             b_q = tl.load(p_q, boundary_check=(0, 1))
#             b_k = tl.load(p_k, boundary_check=(0, 1))
#             b_A += tl.dot(b_q, b_k)

#         # Apply causal mask and gating
#         o_t = i_t * BT + tl.arange(0, BT)
#         m_t = o_t < T
#         if USE_G:
#             g += bos * H + i_h
#             p_g = tl.make_block_ptr(g, (T,), (H,), (i_t * BT,), (BT,), (0,))
#             b_g = tl.load(p_g, boundary_check=(0,))
#             m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
#             b_A = tl.where(m_A, b_A * exp(b_g[:, None] - b_g[None, :]), 0)
#         else:
#             m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
#             b_A = tl.where(m_A, b_A, 0)

#         # Load values for this Householder step
#         p_v = tl.make_block_ptr(v+i_dp*H*V, (T, V), (H*V*num_householder, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
#         p_do = tl.make_block_ptr(do, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
#         b_v = tl.load(p_v, boundary_check=(0, 1))
#         b_do = tl.load(p_do, boundary_check=(0, 1))

#         # Gradient w.r.t. values: dv += A^T @ do
#         b_dv += tl.dot(tl.trans(b_A.to(b_v.dtype)), b_do)

#         # Gradient w.r.t. attention scores: ds = do @ v^T
#         b_ds += tl.dot(b_do, tl.trans(b_v))

#     # Apply scale and gating to score gradients
#     b_ds = b_ds * scale
#     if USE_G:
#         b_ds = tl.where(m_A, b_ds * exp(b_g[:, None] - b_g[None, :]), 0)
#     else:
#         b_ds = tl.where(m_A, b_ds, 0)

#     # Compute final gradients for each Householder step
#     for i_dp in range(num_householder):
#         for i_k in range(tl.cdiv(K, BK)):
#             p_q = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
#             p_k = tl.make_block_ptr(k+i_dp*H*K, (K, T), (1, num_householder*H*K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
#             p_dq = tl.make_block_ptr(dq, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
#             p_dk = tl.make_block_ptr(dk+i_dp*H*K, (K, T), (1, num_householder*H*K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))

#             b_q = tl.load(p_q, boundary_check=(0, 1))
#             b_k = tl.load(p_k, boundary_check=(0, 1))

#             # dq += ds @ k^T
#             b_dq += tl.dot(b_ds, tl.trans(b_k))
#             # dk += q^T @ ds
#             b_dk = tl.dot(tl.trans(b_q), b_ds)

#             tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
#             tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))

#     # Store value gradients
#     for i_dp in range(num_householder):
#         p_dv = tl.make_block_ptr(dv+i_dp*H*V, (T, V), (H*V*num_householder, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
#         tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))


# def chunk_gated_delta_product_bwd_o(
#     q: torch.Tensor,
#     k: torch.Tensor,
#     v: torch.Tensor,
#     h: torch.Tensor,
#     g: Optional[torch.Tensor] = None,
#     do: torch.Tensor = None,
#     scale: Optional[float] = None,
#     cu_seqlens: Optional[torch.LongTensor] = None,
#     chunk_size: int = 64,
#     num_householder: int = 1,
# ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
#     assert q.shape[1] * num_householder == k.shape[1], "q.shape[1] * num_householder must be equal to k.shape[1]"
#     B, T, H, K, V = *q.shape, v.shape[-1]
#     BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
#     chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
#     NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

#     dq = torch.zeros_like(q)
#     dk = torch.zeros_like(k)
#     dv = torch.zeros_like(v)
#     dh = torch.zeros_like(h)

#     def grid(meta): return (triton.cdiv(V, meta['BV']), NT, B * H)
#     chunk_gated_delta_product_bwd_kernel_o[grid](
#         q,
#         k,
#         v,
#         h,
#         g,
#         do,
#         dq,
#         dk,
#         dv,
#         dh,
#         cu_seqlens,
#         chunk_indices,
#         scale,
#         T=T,
#         num_householder=num_householder,
#         H=H,
#         K=K,
#         V=V,
#         BT=BT,
#     )
#     return dq, dk, dv, dh

@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_G_GAMMA': lambda args: args['g_gamma'] is not None,
    'USE_DW': lambda args: args['dw'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
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
    q,
    k,
    v,
    h,
    g,
    g_gamma,
    do,
    dh,
    dq,
    dk,
    dg,
    w,
    dv,
    dw,
    cu_seqlens,
    chunk_indices,
    scale,
    B: tl.constexpr,
    T,
    num_householder: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_G_GAMMA: tl.constexpr,
    USE_DW: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    # TODO use g_interleaved 
    # i_t corresponds to index of BT * num_householder chunk 
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        all = T
        T = eos - bos
        NT = tl.cdiv(T, BT)
        # Calculate true sequence dimensions
        bos_q_o = bos // num_householder
        eos_q_o = eos // num_householder
        T_true = T // num_householder
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * tl.cdiv(T // num_householder, BT) + i_t
        bos, eos = i_b * T, i_b * T + T
        all = B * T
        # Calculate true sequence dimensions
        bos_q_o = bos // num_householder
        eos_q_o = eos // num_householder
        T_true = T // num_householder

    # i_t corresponds to index of BT * num_householder chunk 
    # 
    i_tkw = i_t * num_householder
    i_tg_expanded = i_tg * num_householder # TODO check 
    # index for which BT chunk current thread block corresponds in expanded sequence
    # i_BT = i_t % num_householder

    # offset calculation
    v += (bos * H + i_h) * V
    do += (bos_q_o * H + i_h) * V  # do uses true sequence indexing
    # h0, use NT to get state for each num householder * BT 
    h += (i_tg_expanded * H + i_h).to(tl.int64) * K*V
    dh += (i_tg * H + i_h).to(tl.int64) * K*V 
    q += (bos_q_o * H + i_h) * K  # q uses true sequence indexing
    k += (bos * H + i_h) * K  # k uses expanded sequence indexing
    dq += (bos_q_o * H + i_h) * K  # dq uses true sequence indexing
    dk += (bos * H + i_h) * K  # dk uses expanded sequence indexing

    # for delta rule only
    if USE_DW:
        w += (bos * H + i_h) * K
        dw += (bos * H + i_h) * K
        dv += (bos * H + i_h) * V

    if USE_G:
        dg += i_k * all * H
        b_dg_last = tl.zeros([1,], dtype=tl.float32) if USE_G else None
    # if USE_G_GAMMA:
    #     b_gamma = tl.load(g_gamma + i_h)
    #     b_g = b_gamma * (tl.arange(0, BT) + 1)
    #     b_g_last = b_gamma * min(BT, T - i_t * BT)
    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_ds = tl.zeros([BT, BT], dtype=tl.float32)
    b_dw = tl.zeros([BT, BK], dtype=tl.float32) if USE_DW else None

    if USE_G: 
        b_dg = tl.zeros([BT], dtype=tl.float32)
        b_dg_expanded = tl.zeros([BT*num_householder], dtype=tl.float32)

    # Process num_householder Householder transformations
    # Each iteration processes BT tokens from expanded sequence
    # Load b_v blocks and compute attention scores for gradient computation
    # TODO check out of bounds 
    for i_nh in range(num_householder):
        # zero out b_dk and b_dw
        b_dk = tl.zeros([BT, BK], dtype=tl.float32)
        b_dw = tl.zeros([BT, BK], dtype=tl.float32) if USE_DW else None

        for i_v in range(tl.cdiv(V, BV)):
            # Load values for this Householder step - offset by i_nh 
            # TODO fix H indexing (+ H * V?)
            p_v = tl.make_block_ptr(v, (T, V), (H * V, 1), ((i_tkw + i_nh) * BT, i_v * BV), (BT, BV), (1, 0))
            # TODO check indexing 
            p_h = tl.make_block_ptr(h, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))

            p_do = tl.make_block_ptr(do, (T_true, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_dh = tl.make_block_ptr(dh, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
            # [BT, BV]
            b_v = tl.load(p_v, boundary_check=(0, 1))
            b_do = tl.load(p_do, boundary_check=(0, 1))
            # [BV, BK]
            b_h = tl.load(p_h, boundary_check=(0, 1))
            b_dh = tl.load(p_dh, boundary_check=(0, 1))
            if USE_G:
                if i_nh == 0:
                    b_dg_last += (tl.sum(b_h * b_dh))

            if i_nh == 0: 
                b_dq += tl.dot(b_do, b_h.to(b_do.dtype))

        # Compute attention scores for this Householder step
        # [BT, BV] @ [BV, BT] -> [BT, BT]

            # BT, BT chunk 
            b_ds += tl.dot(b_do, tl.trans(b_v))
            # Compute gradient w.r.t. k: dk += v @ dh^T
            # [BT, BV] @ [BV, BK] -> [BT, BK]
            b_dk += tl.dot(b_v, b_dh.to(b_v.dtype)) 
            if USE_DW:
                p_dv = tl.make_block_ptr(dv, (T, V), (H*V, 1), ((i_tkw + i_nh) * BT, i_v * BV), (BT, BV), (1, 0))
                b_dv = tl.load(p_dv, boundary_check=(0, 1))
                # Compute gradient w.r.t. w: dw += dv @ h^T
                b_dw += tl.dot(b_dv.to(b_v.dtype), b_h.to(b_v.dtype))

    
        # Store gradient for this Householder step's dw
        if USE_DW:
            p_dw = tl.make_block_ptr(dw + H * K, (T, K), (H * K, 1), ((i_tkw + i_nh) * BT, i_k * BK), (BT, BK), (1, 0))
            tl.store(p_dw, -b_dw.to(p_dw.dtype.element_ty), boundary_check=(0, 1))

        tl.debug_barrier()
        p_q = tl.make_block_ptr(q, (T_true, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), ((i_tkw + i_nh) * BT, i_k * BK), (BT, BK), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))

        p_dq = tl.make_block_ptr(dq, (T_true, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dk = tl.make_block_ptr(dk + H * K, (T, K), (H * K, 1), ((i_tkw + i_nh) * BT, i_k * BK), (BT, BK), (1, 0))

        # Create expanded mask like in dv_local
        o_t = i_t * BT * num_householder + tl.arange(0, BT * num_householder)
        m_t = o_t < T
        m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
        # Reduce mask to BT x BT for current Householder step
        m_A_reduced = m_A[num_householder-1::num_householder, i_nh*BT:(i_nh+1)*BT]
        if USE_G:
            g += bos * H + i_h
            dg += bos * H + i_h
            p_g = tl.make_block_ptr(g, (T_true,), (H,), (i_t * BT,), (BT,), (0,))
            b_g = tl.load(p_g, boundary_check=(0,))
            b_g_last = tl.load(g + (min(i_t * BT + BT, T_true) - 1) * H)
            b_dg_last *= exp(b_g_last)

            # Apply gating to gradients
            if i_nh == 0: 
                b_dq = b_dq * exp(b_g)[:, None] * scale
                b_dg += tl.sum(b_dq * b_q, axis=1)

            b_g_expanded = b_g.repeat_interleave(num_householder, dim=1)
            b_g_expanded = b_g_expanded[i_nh*BT:(i_nh+1)*BT] 
    
            b_dk = b_dk * tl.where(m_t, exp(-b_g_expanded + b_g_last), 0)[:, None]
            b_dg_expanded[i_nh*BT:(i_nh+1)*BT] -= tl.sum(b_k * b_dk, axis=1)
            b_dg_last += tl.sum(b_dk * b_k)

            # Apply causal mask and gating to attention scores like in dv_local
            g_mask = (b_g[None, :] - b_g[:, None]) 
            g_mask = g_mask.repeat_interleave(num_householder, dim=1)
            g_mask = g_mask[:, i_nh*BT:(i_nh+1)*BT]
            b_ds = tl.where(m_A_reduced, b_ds * exp(g_mask), 0) * scale
            b_ds2 = b_ds * tl.dot(b_q, tl.trans(b_k))
            b_dg_expanded[i_nh*BT:(i_nh+1)*BT] += tl.sum(b_ds2, axis=1)
            b_dg_expanded[i_nh*BT:(i_nh+1)*BT] -= tl.sum(b_ds2, axis=0)

            # Gate gradients are computed from q, k, and attention scores

            b_ds = b_ds.to(b_k.dtype)
            # [BT, BK]
            b_dq += tl.dot(b_ds, b_k)
            b_dk += tl.dot(tl.trans(b_ds), b_q)
            # p_dg = tl.make_block_ptr(dg, (T_true,), (H,), (i_t * BT,), (BT,), (0,))
            # (SY 09/21) revcumsum in a separate kernel due to strange triton compiler issue
            tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))

        # elif USE_G_GAMMA:
        #     b_dq = b_dq * exp(b_g)[:, None] * scale
        #     b_dk = b_dk * tl.where(m_t, exp(-b_g + b_g_last), 0)[:, None]
        #     b_ds = tl.where(m_A, b_ds * exp(b_g[:, None] - b_g[None, :]), 0) * scale
        #     b_ds = b_ds.to(b_k.dtype)
        #     # [BT, BK]
        #     b_dq += tl.dot(b_ds, b_k)
        #     b_dk += tl.dot(tl.trans(b_ds), b_q)
        #     tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
        #     tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))

        else:
            # No gating case - apply causal mask only
            if i_nh == 0: 
                b_dq *= scale # scale b_do, b_h 

            b_ds = tl.where(m_A_reduced, b_ds, 0)
            b_ds = b_ds.to(b_k.dtype)
            b_dq += tl.dot(b_ds, b_k) * scale 
            b_dk += tl.dot(tl.trans(b_ds), b_q) * scale
            # b_dq *= scale
            tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
        
        
    # Store final gradients
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    if USE_G:
        # Handle boundary condition for gate gradients
        # Collapse b_dg_expanded to b_dg by summing gradients across num_householder repetitions
        # If g was [1,2,3] but computed as [1,1,1,2,2,2,3,3,3], sum back to [1,2,3] size
        for i in range(BT):
            for i_nh in range(num_householder):
                b_dg[i] += b_dg_expanded[i * num_householder + i_nh] 
        o_t_true = i_t * BT + tl.arange(0, BT)
        b_dg = tl.where(o_t_true < min(i_t * BT + BT, T_true) - 1, b_dg, b_dg + b_dg_last)
        p_dg = tl.make_block_ptr(dg, (T_true,), (H,), (i_t * BT,), (BT,), (0,))
        tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0,))

def chunk_bwd_dqkwg(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    h: torch.Tensor,
    dh: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    g_gamma: Optional[torch.Tensor] = None,
    dv: Optional[torch.Tensor] = None,
    w: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_size: int = 64,
    scale: float = 1.0,
    num_householder: int = 1,
):

    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    chunk_indices = prepare_chunk_indices(cu_seqlens // num_householder, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T // num_householder, BT) if cu_seqlens is None else len(chunk_indices)

    CONST_TILING = 64 if check_shared_mem() else 32
    BK = min(max(triton.next_power_of_2(K), 16), CONST_TILING)
    BV = min(max(triton.next_power_of_2(V), 16), CONST_TILING)
    NK = triton.cdiv(K, BK)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dg = torch.empty(NK, *g.shape, dtype=torch.float32, device=g.device) if g is not None else None
    dw = torch.empty_like(w) if w is not None else None

    grid = (NK, NT, B * H)

    # Reduce NT to account for num_householder expansion
    chunk_bwd_kernel_dqkwg[grid](
        q=q,
        k=k,
        v=v,
        h=h,
        g=g,
        g_gamma=g_gamma,
        do=do,
        dh=dh,
        dv=dv,
        w=w,
        dw=dw,
        dq=dq,
        dk=dk,
        dg=dg,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        B=B,
        T=T,
        num_householder=num_householder,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
    )

    if dg is not None:
        dg = dg.sum(0)
    return dq, dk, dw, dg