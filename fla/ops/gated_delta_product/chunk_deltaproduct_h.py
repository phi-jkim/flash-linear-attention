# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_indices, prepare_chunk_offsets
from fla.utils import is_nvidia_hopper

# Hopper runs well with fewer warps; older archs try wider
NUM_WARPS = [2, 4] if is_nvidia_hopper else [2, 4, 8, 16]


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'SAVE_NEW_VALUE': lambda args: args['v_new'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BV': BV}, num_warps=nw, num_stages=ns)
        for nw in [2, 4]
        for ns in [2, 3, 4]
        for BV in [32, 64]
    ],
    key=['H', 'K', 'V', 'BT', 'USE_G'],
)
@triton.jit(do_not_specialize=['T'])
def chunk_gated_delta_product_fwd_kernel_h_blockdim64_expanded(
    k,
    v,
    w,
    v_new,
    g,
    h,
    h0,
    ht,
    cu_seqlens,
    chunk_offsets,
    T,
    num_householder: tl.constexpr,  # number of delta products (M)
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    GRP: tl.constexpr,              # == next_power_of_2(num_householder)
    USE_G: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    SAVE_NEW_VALUE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H

    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos = i_n * T
        eos = i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * tl.cdiv(T, GRP * BT)

    # Hidden-state tiles [64, BV] × up to K<=256
    b_h1 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 64:
        b_h2 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 128:
        b_h3 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 192:
        b_h4 = tl.zeros([64, BV], dtype=tl.float32)

    # Base offsets
    h += (boh * H + i_h) * K * V
    v += (bos * H + i_h) * V
    k += (bos * H + i_h) * K
    w += (bos * H + i_h) * K
    if SAVE_NEW_VALUE:
        v_new += (bos * H + i_h) * V
    stride_v = H * V
    stride_h = H * K * V
    stride_k = H * K
    if USE_INITIAL_STATE:
        h0 = h0 + i_nh * K * V
    if STORE_FINAL_STATE:
        ht = ht + i_nh * K * V

    # Load initial state
    if USE_INITIAL_STATE:
        p_h0 = tl.make_block_ptr(h0, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        b_h1 += tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32)
        if K > 64:
            p_h0_64 = tl.make_block_ptr(h0, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            b_h2 += tl.load(p_h0_64, boundary_check=(0, 1)).to(tl.float32)
        if K > 128:
            p_h0_128 = tl.make_block_ptr(h0, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            b_h3 += tl.load(p_h0_128, boundary_check=(0, 1)).to(tl.float32)
        if K > 192:
            p_h0_192 = tl.make_block_ptr(h0, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            b_h4 += tl.load(p_h0_192, boundary_check=(0, 1)).to(tl.float32)

    # Snapshots used within a (GRP * BT) expanded group (SSA value snapshot)
    b_h1_cpy = b_h1
    if K > 64:
        b_h2_cpy = b_h2
    if K > 128:
        b_h3_cpy = b_h3
    if K > 192:
        b_h4_cpy = b_h4

    # Main scan across expanded tiles
    for i_t in range(NT):
        i_t_true = i_t // GRP

        if (i_t % GRP) == 0:
            # refresh group snapshot & store slab into h for this true chunk
            b_h1_cpy = b_h1
            if K > 64:
                b_h2_cpy = b_h2
            if K > 128:
                b_h3_cpy = b_h3
            if K > 192:
                b_h4_cpy = b_h4

            p_h = tl.make_block_ptr(h + i_t_true * stride_h, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
            tl.store(p_h, b_h1.to(p_h.dtype.element_ty), boundary_check=(0, 1))
            if K > 64:
                p_h_64 = tl.make_block_ptr(h + i_t_true * stride_h, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
                tl.store(p_h_64, b_h2.to(p_h_64.dtype.element_ty), boundary_check=(0, 1))
            if K > 128:
                p_h_128 = tl.make_block_ptr(h + i_t_true * stride_h, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
                tl.store(p_h_128, b_h3.to(p_h_128.dtype.element_ty), boundary_check=(0, 1))
            if K > 192:
                p_h_192 = tl.make_block_ptr(h + i_t_true * stride_h, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
                tl.store(p_h_192, b_h4.to(p_h_192.dtype.element_ty), boundary_check=(0, 1))

        # v_new(tile) = v(tile) - w(tile) @ h_snapshot
        p_v = tl.make_block_ptr(v, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v_new = tl.zeros([BT, BV], dtype=tl.float32)

        p_w = tl.make_block_ptr(w, (T, K), (stride_k, 1), (i_t * BT, 0), (BT, 64), (1, 0))
        b_w = tl.load(p_w, boundary_check=(0, 1))
        b_v_new += tl.dot(b_w, b_h1_cpy.to(b_w.dtype))
        if K > 64:
            p_w_64 = tl.make_block_ptr(w, (T, K), (stride_k, 1), (i_t * BT, 64), (BT, 64), (1, 0))
            b_w = tl.load(p_w_64, boundary_check=(0, 1))
            b_v_new += tl.dot(b_w, b_h2_cpy.to(b_w.dtype))
        if K > 128:
            p_w_128 = tl.make_block_ptr(w, (T, K), (stride_k, 1), (i_t * BT, 128), (BT, 64), (1, 0))
            b_w = tl.load(p_w_128, boundary_check=(0, 1))
            b_v_new += tl.dot(b_w, b_h3_cpy.to(b_w.dtype))
        if K > 192:
            p_w_192 = tl.make_block_ptr(w, (T, K), (stride_k, 1), (i_t * BT, 192), (BT, 64), (1, 0))
            b_w = tl.load(p_w_192, boundary_check=(0, 1))
            b_v_new += tl.dot(b_w, b_h4_cpy.to(b_w.dtype))

        b_v_new = -b_v_new + tl.load(p_v, boundary_check=(0, 1))

        if SAVE_NEW_VALUE:
            p_vn = tl.make_block_ptr(v_new, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            tl.store(p_vn, b_v_new.to(p_vn.dtype.element_ty), boundary_check=(0, 1))

        # gating
        if USE_G:
            m_t = (i_t * BT + tl.arange(0, BT)) < T
            last_idx = tl.minimum((i_t_true + 1) * BT * GRP, T) - 1
            b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
            p_g = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
            b_g = tl.load(p_g, boundary_check=(0,))
            b_v_new = b_v_new * tl.where(m_t, tl.exp(b_g_last - b_g), 0)[:, None]
            s = tl.exp(b_g_last)
            if (i_t % GRP) == 0:
                b_h1 = b_h1 * s
                if K > 64:
                    b_h2 = b_h2 * s
                if K > 128:
                    b_h3 = b_h3 * s
                if K > 192:
                    b_h4 = b_h4 * s

        # h <- h + K^T(tile) @ v_new(tile)
        p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (0, i_t * BT), (64, BT), (0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_vn_cast = b_v_new.to(b_k.dtype)
        b_h1 += tl.dot(b_k, b_vn_cast)
        if K > 64:
            p_k_64 = tl.make_block_ptr(k, (K, T), (1, stride_k), (64, i_t * BT), (64, BT), (0, 1))
            b_k = tl.load(p_k_64, boundary_check=(0, 1))
            b_h2 += tl.dot(b_k, b_vn_cast)
        if K > 128:
            p_k_128 = tl.make_block_ptr(k, (K, T), (1, stride_k), (128, i_t * BT), (64, BT), (0, 1))
            b_k = tl.load(p_k_128, boundary_check=(0, 1))
            b_h3 += tl.dot(b_k, b_vn_cast)
        if K > 192:
            p_k_192 = tl.make_block_ptr(k, (K, T), (1, stride_k), (192, i_t * BT), (64, BT), (0, 1))
            b_k = tl.load(p_k_192, boundary_check=(0, 1))
            b_h4 += tl.dot(b_k, b_vn_cast)

    # Final state (per-head)
    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_ht_64 = tl.make_block_ptr(ht, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            tl.store(p_ht_64, b_h2.to(p_ht_64.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_ht_128 = tl.make_block_ptr(ht, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            tl.store(p_ht_128, b_h3.to(p_ht_128.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_ht_192 = tl.make_block_ptr(ht, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            tl.store(p_ht_192, b_h4.to(p_ht_192.dtype.element_ty), boundary_check=(0, 1))


def chunk_gated_delta_product_fwd_h_expanded(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    save_new_value: bool = True,
    cu_seqlens: Optional[torch.LongTensor] = None,
    num_householder: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Expanded-time forward (T is expanded length = T_true * num_householder).
    Produces per-(true chunk) slabs h and optionally v_new & final_state.
    """
    B, T, H, K, V = *k.shape, u.shape[-1]
    assert T % num_householder == 0, "T must be divisible by num_householder"
    BT = chunk_size
    GRP = triton.next_power_of_2(num_householder)

    if cu_seqlens is None:
        N = B
        NT = triton.cdiv(T, BT * GRP)
        chunk_indices = None
        chunk_offsets = None
    else:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT * GRP)
        N = len(cu_seqlens) - 1
        NT = len(chunk_indices)
        chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT * GRP)

    assert K <= 256, "current kernel does not support head dimension larger than 256."
    h = k.new_empty(B, NT, H, K, V)
    final_state = k.new_empty(N, H, K, V, dtype=torch.float32) if output_final_state else None
    v_new = torch.empty_like(u) if save_new_value else None

    def grid(meta):
        return (triton.cdiv(V, meta['BV']), N * H)

    chunk_gated_delta_product_fwd_kernel_h_blockdim64_expanded[grid](
        k=k, v=u, w=w, v_new=v_new, g=g, h=h,
        h0=initial_state, ht=final_state,
        cu_seqlens=cu_seqlens, chunk_offsets=chunk_offsets,
        num_householder=num_householder, T=T, H=H, K=K, V=V, BT=BT, GRP=GRP
    )
    return h, v_new, final_state


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'SAVE_NEW_VALUE': lambda args: args['v_new'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BV': BV}, num_warps=nw, num_stages=ns)
        for nw in [2, 4]
        for ns in [2, 3, 4]
        for BV in [32, 64]
    ],
    key=['H', 'K', 'V', 'BT', 'USE_G'],
)
@triton.jit(do_not_specialize=['T'])
def chunk_gated_delta_product_fwd_kernel_h_blockdim64(
    k,
    v,
    w,
    v_new,
    g,
    h,
    h0,
    ht,
    cu_seqlens,
    chunk_offsets,
    T,
    num_householder: tl.constexpr,  # M
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    SAVE_NEW_VALUE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H

    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos = i_n * T
        eos = i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * tl.cdiv((T // num_householder), BT)

    b_h1 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 64:
        b_h2 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 128:
        b_h3 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 192:
        b_h4 = tl.zeros([64, BV], dtype=tl.float32)

    # Base offsets
    h += (boh * H + i_h) * K * V
    v += (bos * H + i_h) * V
    k += (bos * H + i_h) * K
    w += (bos * H + i_h) * K
    if SAVE_NEW_VALUE:
        v_new += (bos * H + i_h) * V
    stride_v = H * V
    stride_h = H * K * V
    stride_k = H * K
    if USE_INITIAL_STATE:
        h0 = h0 + i_nh * K * V
    if STORE_FINAL_STATE:
        ht = ht + i_nh * K * V

    # Load initial state
    if USE_INITIAL_STATE:
        p_h0 = tl.make_block_ptr(h0, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        b_h1 += tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32)
        if K > 64:
            p_h0_64 = tl.make_block_ptr(h0, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            b_h2 += tl.load(p_h0_64, boundary_check=(0, 1)).to(tl.float32)
        if K > 128:
            p_h0_128 = tl.make_block_ptr(h0, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            b_h3 += tl.load(p_h0_128, boundary_check=(0, 1)).to(tl.float32)
        if K > 192:
            p_h0_192 = tl.make_block_ptr(h0, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            b_h4 += tl.load(p_h0_192, boundary_check=(0, 1)).to(tl.float32)

    for i_t in range(NT):
        # Store slab at the beginning of each true chunk (every M expanded tiles)
        if (i_t % num_householder) == 0:
            i_t_true = i_t // num_householder
            p_h = tl.make_block_ptr(h + i_t_true * stride_h, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
            tl.store(p_h, b_h1.to(p_h.dtype.element_ty), boundary_check=(0, 1))
            if K > 64:
                p_h_64 = tl.make_block_ptr(h + i_t_true * stride_h, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
                tl.store(p_h_64, b_h2.to(p_h_64.dtype.element_ty), boundary_check=(0, 1))
            if K > 128:
                p_h_128 = tl.make_block_ptr(h + i_t_true * stride_h, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
                tl.store(p_h_128, b_h3.to(p_h_128.dtype.element_ty), boundary_check=(0, 1))
            if K > 192:
                p_h_192 = tl.make_block_ptr(h + i_t_true * stride_h, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
                tl.store(p_h_192, b_h4.to(p_h_192.dtype.element_ty), boundary_check=(0, 1))

        # v_new(tile)
        p_v = tl.make_block_ptr(v, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v_new = tl.zeros([BT, BV], dtype=tl.float32)

        p_w = tl.make_block_ptr(w, (T, K), (stride_k, 1), (i_t * BT, 0), (BT, 64), (1, 0))
        b_w = tl.load(p_w, boundary_check=(0, 1))
        b_v_new += tl.dot(b_w, b_h1.to(b_w.dtype))
        if K > 64:
            p_w_64 = tl.make_block_ptr(w, (T, K), (stride_k, 1), (i_t * BT, 64), (BT, 64), (1, 0))
            b_w = tl.load(p_w_64, boundary_check=(0, 1))
            b_v_new += tl.dot(b_w, b_h2.to(b_w.dtype))
        if K > 128:
            p_w_128 = tl.make_block_ptr(w, (T, K), (stride_k, 1), (i_t * BT, 128), (BT, 64), (1, 0))
            b_w = tl.load(p_w_128, boundary_check=(0, 1))
            b_v_new += tl.dot(b_w, b_h3.to(b_w.dtype))
        if K > 192:
            p_w_192 = tl.make_block_ptr(w, (T, K), (stride_k, 1), (i_t * BT, 192), (BT, 64), (1, 0))
            b_w = tl.load(p_w_192, boundary_check=(0, 1))
            b_v_new += tl.dot(b_w, b_h4.to(b_w.dtype))

        b_v_new = -b_v_new + tl.load(p_v, boundary_check=(0, 1))

        if SAVE_NEW_VALUE:
            p_vn = tl.make_block_ptr(v_new, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            tl.store(p_vn, b_v_new.to(p_vn.dtype.element_ty), boundary_check=(0, 1))

        # gating
        if USE_G:
            m_t = (i_t * BT + tl.arange(0, BT)) < T
            last_idx = tl.minimum((i_t + 1) * BT, T) - 1
            b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
            p_g = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
            b_g = tl.load(p_g, boundary_check=(0,))
            b_v_new = b_v_new * tl.where(m_t, tl.exp(b_g_last - b_g), 0)[:, None]
            s = tl.exp(b_g_last)
            b_h1 = b_h1 * s
            if K > 64:
                b_h2 = b_h2 * s
            if K > 128:
                b_h3 = b_h3 * s
            if K > 192:
                b_h4 = b_h4 * s

        # h update
        p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (0, i_t * BT), (64, BT), (0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_vn_cast = b_v_new.to(b_k.dtype)
        b_h1 += tl.dot(b_k, b_vn_cast)
        if K > 64:
            p_k_64 = tl.make_block_ptr(k, (K, T), (1, stride_k), (64, i_t * BT), (64, BT), (0, 1))
            b_k = tl.load(p_k_64, boundary_check=(0, 1))
            b_h2 += tl.dot(b_k, b_vn_cast)
        if K > 128:
            p_k_128 = tl.make_block_ptr(k, (K, T), (1, stride_k), (128, i_t * BT), (64, BT), (0, 1))
            b_k = tl.load(p_k_128, boundary_check=(0, 1))
            b_h3 += tl.dot(b_k, b_vn_cast)
        if K > 192:
            p_k_192 = tl.make_block_ptr(k, (K, T), (1, stride_k), (192, i_t * BT), (64, BT), (0, 1))
            b_k = tl.load(p_k_192, boundary_check=(0, 1))
            b_h4 += tl.dot(b_k, b_vn_cast)

    # Final state
    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_ht_64 = tl.make_block_ptr(ht, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            tl.store(p_ht_64, b_h2.to(p_ht_64.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_ht_128 = tl.make_block_ptr(ht, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            tl.store(p_ht_128, b_h3.to(p_ht_128.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_ht_192 = tl.make_block_ptr(ht, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            tl.store(p_ht_192, b_h4.to(p_ht_192.dtype.element_ty), boundary_check=(0, 1))


def chunk_gated_delta_product_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    save_new_value: bool = True,
    cu_seqlens: Optional[torch.LongTensor] = None,
    num_householder: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    True-time forward (T_true = T_exp / num_householder).
    """
    B, T_exp, H, K, V = *k.shape, u.shape[-1]
    assert T_exp % num_householder == 0, "T must be divisible by num_householder"
    T_true = T_exp // num_householder
    BT = chunk_size

    if cu_seqlens is None:
        N = B
        NT = triton.cdiv(T_true, BT)
        chunk_indices = None
        chunk_offsets = None
    else:
        # Convert expanded cu_seqlens -> true timeline for indexing
        cu_true = cu_seqlens // num_householder
        chunk_indices = prepare_chunk_indices(cu_true, BT)
        N = len(cu_seqlens) - 1
        NT = len(chunk_indices)
        chunk_offsets = prepare_chunk_offsets(cu_true, BT)

    assert K <= 256, "current kernel does not support head dimension larger than 256."
    h = k.new_empty(B, NT, H, K, V)
    final_state = k.new_empty(N, H, K, V, dtype=torch.float32) if output_final_state else None
    v_new = torch.empty_like(u) if save_new_value else None

    def grid(meta):
        return (triton.cdiv(V, meta['BV']), N * H)

    chunk_gated_delta_product_fwd_kernel_h_blockdim64[grid](
        k=k, v=u, w=w, v_new=v_new, g=g, h=h,
        h0=initial_state, ht=final_state,
        cu_seqlens=cu_seqlens, chunk_offsets=chunk_offsets,
        num_householder=num_householder, T=T_true, H=H, K=K, V=V, BT=BT
    )
    return h, v_new, final_state


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_INITIAL_STATE': lambda args: args['dh0'] is not None,
    'USE_FINAL_STATE_GRADIENT': lambda args: args['dht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BV': BV}, num_warps=nw, num_stages=ns)
        for nw in [2, 4]
        for ns in [4, 3, 2]
        for BV in [64, 32]
    ],
    key=['H', 'K', 'V', 'BT', 'BV', 'USE_G'],
)
@triton.jit(do_not_specialize=['T_true'])
def chunk_gated_delta_product_bwd_kernel_dhu_blockdim64(
    q,               # [B, T_true, H, K]  (TRUE)
    k,               # [B, T_exp,  H, K]  (EXPANDED)
    w,               # [B, T_exp,  H, K]  (EXPANDED)
    g,               # [B, T_exp,  H]     (EXPANDED, log-space)
    dht,             # [N, H, K, V] or None (seed)
    dh0,             # [N, H, K, V] or None
    do,              # [B, T_true, H, V]  (TRUE)
    dh,              # [B or total, NT_grp, H, K, V] (expanded-group states)
    dv,              # [B, T_exp,  H, V]  (in/out)
    dv2,             # [B, T_exp,  H, V]  (out)
    cu_seqlens,
    chunk_offsets,
    scale,
    T_exp,           # expanded length (rows)
    T_true,          # true length (cols)
    num_householder: tl.constexpr,
    expanded_chunk_size: tl.constexpr,  # EXP_CHUNK = BT * next_power_of_2(M)
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,                   # row tile (expanded)
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H

    M   = num_householder
    EXP_CHUNK = expanded_chunk_size
    GRP = EXP_CHUNK // BT
    BTC = (expanded_chunk_size + M - 1) // M  # ceil(EXP_CHUNK / M)

    # expanded bounds
    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        Tloc = eos - bos
        NT   = tl.cdiv(Tloc, BT)
        NT_grp = tl.cdiv(Tloc, EXP_CHUNK)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos = i_n * T_exp
        eos = i_n * T_exp + T_exp
        Tloc = T_exp
        NT   = tl.cdiv(Tloc, BT)
        NT_grp = tl.cdiv(Tloc, EXP_CHUNK)
        boh = i_n * NT_grp

    # base offsets (per head)
    q   += ((bos // M) * H + i_h) * K
    do  += ((bos // M) * H + i_h) * V
    k   += (bos * H + i_h) * K
    w   += (bos * H + i_h) * K
    if USE_G:
        g   += (bos * H + i_h)
    dv  += (bos * H + i_h) * V
    dv2 += (bos * H + i_h) * V
    dh  += (boh * H + i_h) * K * V
    if USE_INITIAL_STATE:
        dh0 += i_nh * K * V
    if USE_FINAL_STATE_GRADIENT:
        dht += i_nh * K * V

    stride_v = H * V
    stride_h = H * K * V
    stride_k = H * K

    # accumulators across one expanded group
    b_dh1 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 64:  b_dh2 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 128: b_dh3 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 192: b_dh4 = tl.zeros([64, BV], dtype=tl.float32)

    # PRE snapshots (constant within a group, refreshed at group entry)
    # (SSA assignment = snapshot)
    b_dh1_pre = b_dh1
    if K > 64:  b_dh2_pre = b_dh2
    if K > 128: b_dh3_pre = b_dh3
    if K > 192: b_dh4_pre = b_dh4

    # seed from final-state gradient, before the first store
    if USE_FINAL_STATE_GRADIENT:
        p_dht = tl.make_block_ptr(dht, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        b_dh1 += tl.load(p_dht, boundary_check=(0, 1))
        if K > 64:
            p_dht_64 = tl.make_block_ptr(dht, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            b_dh2 += tl.load(p_dht_64, boundary_check=(0, 1))
        if K > 128:
            p_dht_128 = tl.make_block_ptr(dht, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            b_dh3 += tl.load(p_dht_128, boundary_check=(0, 1))
        if K > 192:
            p_dht_192 = tl.make_block_ptr(dht, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            b_dh4 += tl.load(p_dht_192, boundary_check=(0, 1))

    # reverse tiles
    for i_t in range(NT - 1, -1, -1):
        # 1) TOP-OF-LOOP STORE at expanded-chunk boundary
        if (i_t + 1 == NT) or (((i_t + 1) % GRP) == 0):
            i_grp = (NT_grp - 1) if (i_t + 1 == NT) else (((i_t + 1) // GRP) - 1)
            p_dh = tl.make_block_ptr(dh + i_grp * stride_h, (K, V), (V, 1),
                                     (0, i_v * BV), (64, BV), (1, 0))
            tl.store(p_dh, b_dh1.to(p_dh.dtype.element_ty), boundary_check=(0, 1))
            if K > 64:
                p_dh_64 = tl.make_block_ptr(dh + i_grp * stride_h, (K, V), (V, 1),
                                            (64, i_v * BV), (64, BV), (1, 0))
                tl.store(p_dh_64, b_dh2.to(p_dh_64.dtype.element_ty), boundary_check=(0, 1))
            if K > 128:
                p_dh_128 = tl.make_block_ptr(dh + i_grp * stride_h, (K, V), (V, 1),
                                             (128, i_v * BV), (64, BV), (1, 0))
                tl.store(p_dh_128, b_dh3.to(p_dh_128.dtype.element_ty), boundary_check=(0, 1))
            if K > 192:
                p_dh_192 = tl.make_block_ptr(dh + i_grp * stride_h, (K, V), (V, 1),
                                             (192, i_v * BV), (64, BV), (1, 0))
                tl.store(p_dh_192, b_dh4.to(p_dh_192.dtype.element_ty), boundary_check=(0, 1))

        # group geometry for this tile
        grp_lo_tiles = (i_t // GRP) * GRP
        grp_hi_tiles = tl.minimum(grp_lo_tiles + GRP, NT)
        tile_row_base = i_t * BT
        chunk_lo = grp_lo_tiles * BT
        chunk_hi = tl.minimum(grp_hi_tiles * BT, Tloc)
        last_idx = chunk_hi - 1

        # 2) GROUP ENTRY (first tile from the right)
        is_group_entry = i_t == (grp_hi_tiles - 1) or i_t == (NT - 1)
        if USE_G:
            bg_last = tl.load(g + last_idx * H)
            if is_group_entry:
                # capture PRE snapshot
                b_dh1_pre = b_dh1
                if K > 64:  b_dh2_pre = b_dh2
                if K > 128: b_dh3_pre = b_dh3
                if K > 192: b_dh4_pre = b_dh4
                s = tl.exp(bg_last)
                b_dh1 *= s
                if K > 64:  b_dh2 *= s
                if K > 128: b_dh3 *= s
                if K > 192: b_dh4 *= s
            p_g_rows = tl.make_block_ptr(g, (Tloc,), (H,), (tile_row_base,), (BT,), (0,))
            b_g_rows = tl.load(p_g_rows, boundary_check=(0,))
        else:
            if is_group_entry:
                b_dh1_pre = b_dh1
                if K > 64:  b_dh2_pre = b_dh2
                if K > 128: b_dh3_pre = b_dh3
                if K > 192: b_dh4_pre = b_dh4

        # 3) dv_new(tile) uses PRE within this group
        b_dv = tl.zeros([BT, BV], dtype=tl.float32)
        p_k = tl.make_block_ptr(k, (Tloc, K), (stride_k, 1), (tile_row_base, 0), (BT, 64), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_dv += tl.dot(b_k, b_dh1_pre.to(b_k.dtype))
        if K > 64:
            p_k_64 = tl.make_block_ptr(k, (Tloc, K), (stride_k, 1), (tile_row_base, 64), (BT, 64), (1, 0))
            b_k = tl.load(p_k_64, boundary_check=(0, 1))
            b_dv += tl.dot(b_k, b_dh2_pre.to(b_k.dtype))
        if K > 128:
            p_k_128 = tl.make_block_ptr(k, (Tloc, K), (stride_k, 1), (tile_row_base, 128), (BT, 64), (1, 0))
            b_k = tl.load(p_k_128, boundary_check=(0, 1))
            b_dv += tl.dot(b_k, b_dh3_pre.to(b_k.dtype))
        if K > 192:
            p_k_192 = tl.make_block_ptr(k, (Tloc, K), (stride_k, 1), (tile_row_base, 192), (BT, 64), (1, 0))
            b_k = tl.load(p_k_192, boundary_check=(0, 1))
            b_dv += tl.dot(b_k, b_dh4_pre.to(b_k.dtype))

        if USE_G:
            m_rows = (tile_row_base + tl.arange(0, BT)) < Tloc
            b_dv *= tl.where(m_rows, tl.exp(bg_last - b_g_rows), 0)[:, None]

        p_dv  = tl.make_block_ptr(dv,  (Tloc, V), (stride_v, 1), (tile_row_base, i_v * BV), (BT, BV), (1, 0))
        p_dv2 = tl.make_block_ptr(dv2, (Tloc, V), (stride_v, 1), (tile_row_base, i_v * BV), (BT, BV), (1, 0))
        b_dv += tl.load(p_dv, boundary_check=(0, 1))
        tl.store(p_dv2, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))

        # 4) dh update inside the group: restrict TRUE columns to this expanded group
        j0_true = chunk_lo // M
        o_col_exp = chunk_lo + (M - 1) + tl.arange(0, BTC) * M
        m_cols = (o_col_exp < chunk_hi) & ((j0_true + tl.arange(0, BTC)) < T_true)

        if is_group_entry:
            p_q = tl.make_block_ptr(q,  (K, T_true), (1, H * K), (0,       j0_true), (64, BTC), (0, 1))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            p_do_blk = tl.make_block_ptr(do, (T_true, V), (H * V, 1), (j0_true, i_v * BV), (BTC, BV), (1, 0))
            b_do_blk = tl.load(p_do_blk, boundary_check=(0, 1))

            b_q      = b_q * m_cols[None, :].to(b_q.dtype)
            b_do_blk = b_do_blk * m_cols[:, None].to(b_do_blk.dtype)

            if USE_G:
                b_g_cols = tl.load(g + o_col_exp * H, mask=m_cols, other=0.0)
                b_q = (b_q * tl.exp(b_g_cols)[None, :]).to(b_q.dtype)

            b_dh1 += tl.dot((b_q * scale).to(b_q.dtype), b_do_blk.to(b_q.dtype))

        p_w = tl.make_block_ptr(w, (K, Tloc), (1, stride_k), (0, tile_row_base), (64, BT), (0, 1))
        b_w = tl.load(p_w, boundary_check=(0, 1))
        b_dh1 -= tl.dot(b_w, b_dv.to(b_w.dtype))

        if K > 64:
            if is_group_entry:
                p_q = tl.make_block_ptr(q, (K, T_true), (1, H * K), (64, j0_true), (64, BTC), (0, 1))
                b_q = tl.load(p_q, boundary_check=(0, 1))
                b_q = b_q * m_cols[None, :].to(b_q.dtype)
                if USE_G:
                    b_q = (b_q * tl.exp(b_g_cols)[None, :]).to(b_q.dtype)
                p_do_blk = tl.make_block_ptr(do, (T_true, V), (H * V, 1), (j0_true, i_v * BV), (BTC, BV), (1, 0))
                b_do_blk = tl.load(p_do_blk, boundary_check=(0, 1))
                b_dh2 += tl.dot((b_q * scale).to(b_q.dtype), b_do_blk.to(b_q.dtype))

            p_w = tl.make_block_ptr(w, (K, Tloc), (1, stride_k), (64, tile_row_base), (64, BT), (0, 1))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_dh2 -= tl.dot(b_w, b_dv.to(b_w.dtype))

        if K > 128:
            if is_group_entry:
                p_q = tl.make_block_ptr(q, (K, T_true), (1, H * K), (128, j0_true), (64, BTC), (0, 1))
                b_q = tl.load(p_q, boundary_check=(0, 1))
                b_q = b_q * m_cols[None, :].to(b_q.dtype)
                if USE_G:
                    b_q = (b_q * tl.exp(b_g_cols)[None, :]).to(b_q.dtype)
                p_do_blk = tl.make_block_ptr(do, (T_true, V), (H * V, 1), (j0_true, i_v * BV), (BTC, BV), (1, 0))
                b_do_blk = tl.load(p_do_blk, boundary_check=(0, 1))
                b_dh3 += tl.dot((b_q * scale).to(b_q.dtype), b_do_blk.to(b_q.dtype))

            p_w = tl.make_block_ptr(w, (K, Tloc), (1, stride_k), (128, tile_row_base), (64, BT), (0, 1))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_dh3 -= tl.dot(b_w, b_dv.to(b_w.dtype))

        if K > 192:
            if is_group_entry:
                p_q = tl.make_block_ptr(q, (K, T_true), (1, H * K), (192, j0_true), (64, BTC), (0, 1))
                b_q = tl.load(p_q, boundary_check=(0, 1))
                b_q = b_q * m_cols[None, :].to(b_q.dtype)
                if USE_G:
                    b_q = (b_q * tl.exp(b_g_cols)[None, :]).to(b_q.dtype)
                p_do_blk = tl.make_block_ptr(do, (T_true, V), (H * V, 1), (j0_true, i_v * BV), (BTC, BV), (1, 0))
                b_do_blk = tl.load(p_do_blk, boundary_check=(0, 1))
                b_dh4 += tl.dot((b_q * scale).to(b_q.dtype), b_do_blk.to(b_q.dtype))

            p_w = tl.make_block_ptr(w, (K, Tloc), (1, stride_k), (192, tile_row_base), (64, BT), (0, 1))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_dh4 -= tl.dot(b_w, b_dv.to(b_w.dtype))

    # dh0 write
    if USE_INITIAL_STATE:
        p_dh0 = tl.make_block_ptr(dh0, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        tl.store(p_dh0, b_dh1.to(p_dh0.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_dh0_64 = tl.make_block_ptr(dh0, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            tl.store(p_dh0_64, b_dh2.to(p_dh0_64.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_dh0_128 = tl.make_block_ptr(dh0, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            tl.store(p_dh0_128, b_dh3.to(p_dh0_128.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_dh0_192 = tl.make_block_ptr(dh0, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            tl.store(p_dh0_192, b_dh4.to(p_dh0_192.dtype.element_ty), boundary_check=(0, 1))


def chunk_gated_delta_product_bwd_dhu(
    q: torch.Tensor,          # [B, T_true, H, K]
    k: torch.Tensor,          # [B, T_exp,  H, K]
    w: torch.Tensor,          # [B, T_exp,  H, K]
    g: torch.Tensor,          # [B, T_exp,  H]
    h0: torch.Tensor,         # [N, H, K, V]
    dht: Optional[torch.Tensor],
    do: torch.Tensor,         # [B, T_true, H, V]
    dv: torch.Tensor,         # [B, T_exp,  H, V]
    scale: float,
    cu_seqlens: Optional[torch.LongTensor] = None,  # expanded lengths
    chunk_size: int = 64,
    num_householder: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Backward pass computing:
      - dh (per expanded-chunk group hidden slabs),
      - dh0 (gradient wrt initial state),
      - dv2 (accumulated dv with local contributions).
    """
    B, T_exp, H, K, V = *k.shape, do.shape[-1]
    assert T_exp % num_householder == 0, "Expanded T must be divisible by num_householder"
    T_true = T_exp // num_householder

    BT = chunk_size
    GRP = triton.next_power_of_2(num_householder)  # expanded-chunk grouping
    EXP_CHUNK = BT * GRP

    assert K <= 256, "current kernel does not support head dimension being larger than 256."

    if cu_seqlens is None:
        N = B
        NT_grp = triton.cdiv(T_exp, EXP_CHUNK)
        # Keep a consistent (B, NT_grp, ...) layout for fixed-length
        dh = q.new_empty(B, NT_grp, H, K, V)
        chunk_offsets = None
    else:
        # Varlen: allocate per total number of expanded groups across the batch
        chunk_indices = prepare_chunk_indices(cu_seqlens, EXP_CHUNK)
        N = len(cu_seqlens) - 1
        NT_total = len(chunk_indices)
        dh = q.new_empty(B, NT_total, H, K, V)  # oversize but simple; kernel uses chunk_offsets
        chunk_offsets = prepare_chunk_offsets(cu_seqlens, EXP_CHUNK)

    dh0_out = torch.empty_like(h0, dtype=torch.float32) if h0 is not None else None
    dv2 = torch.empty_like(dv)

    def grid(meta):
        return (triton.cdiv(V, meta['BV']), N * H)

    chunk_gated_delta_product_bwd_kernel_dhu_blockdim64[grid](
        q=q,
        k=k,
        w=w,
        g=g,
        dht=dht,
        dh0=dh0_out,
        do=do,
        dh=dh,
        dv=dv,
        dv2=dv2,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        scale=float(scale),
        T_exp=T_exp,
        T_true=T_true,
        num_householder=num_householder,
        expanded_chunk_size=EXP_CHUNK,
        H=H, K=K, V=V, BT=BT,
    )
    return dh, dh0_out, dv2
