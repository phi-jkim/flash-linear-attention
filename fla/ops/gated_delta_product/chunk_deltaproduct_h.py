# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_indices, prepare_chunk_offsets
from fla.ops.utils.op import exp
from fla.utils import is_nvidia_hopper, use_cuda_graph

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
        triton.Config({'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        for num_stages in [2, 3, 4]
        for BV in [32, 64]
    ],
    key=['H', 'K', 'V', 'BT', 'USE_G'],
    use_cuda_graph=use_cuda_graph,
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
    num_householder: tl.constexpr,  # number of delta products
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
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * tl.cdiv(T // num_householder, BT)

    # [BK, BV]
    b_h1 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 64:
        b_h2 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 128:
        b_h3 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 192:
        b_h4 = tl.zeros([64, BV], dtype=tl.float32)

    # calculate offset
    h += (boh * H + i_h) * K*V
    v += (bos * H + i_h) * V
    k += (bos * H + i_h) * K
    w += (bos * H + i_h) * K
    if SAVE_NEW_VALUE:
        v_new += (bos * H + i_h) * V
    stride_v = H*V
    stride_h = H*K*V
    stride_k = H*K
    if USE_INITIAL_STATE:
        h0 = h0 + i_nh * K*V
    if STORE_FINAL_STATE:
        ht = ht + i_nh * K*V

    # load initial state
    if USE_INITIAL_STATE:
        p_h0_1 = tl.make_block_ptr(h0, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
        if K > 64:
            p_h0_2 = tl.make_block_ptr(h0, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            b_h2 += tl.load(p_h0_2, boundary_check=(0, 1)).to(tl.float32)
        if K > 128:
            p_h0_3 = tl.make_block_ptr(h0, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            b_h3 += tl.load(p_h0_3, boundary_check=(0, 1)).to(tl.float32)
        if K > 192:
            p_h0_4 = tl.make_block_ptr(h0, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            b_h4 += tl.load(p_h0_4, boundary_check=(0, 1)).to(tl.float32)

    #     # make copies of b_dh 
    b_h1_cpy = b_dh1.copy()
    if K > 64:
        b_h2_cpy = b_dh2.copy()
    if K > 128:
        b_h3_cpy = b_dh3.copy()
    if K > 192:
        b_h4_cpy = b_dh4.copy()

    # main recurrence
    for i_t in range(NT):
        i_t_true = i_t // num_householder
        if i_t % num_householder == 0:
            b_h1_cpy = b_h1.copy()
            if K > 64:
                b_h2_cpy = b_h2.copy()
            if K > 128:
                b_h3_cpy = b_h3.copy()
            if K > 192:
                b_h4_cpy = b_h4.copy()

            # i_t_true = i_t // num_householder
            p_h1 = tl.make_block_ptr(h + i_t_true * stride_h, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
            tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), boundary_check=(0, 1))
            if K > 64:
                p_h2 = tl.make_block_ptr(h + i_t_true * stride_h, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
                tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), boundary_check=(0, 1))
            if K > 128:
                p_h3 = tl.make_block_ptr(h + i_t_true * stride_h, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
                tl.store(p_h3, b_h3.to(p_h3.dtype.element_ty), boundary_check=(0, 1))
            if K > 192:
                p_h4 = tl.make_block_ptr(h + i_t_true * stride_h, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
                tl.store(p_h4, b_h4.to(p_h4.dtype.element_ty), boundary_check=(0, 1))

        p_v = tl.make_block_ptr(v, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_v_new = tl.make_block_ptr(v_new, (T, V), (stride_v, 1), (i_t * BT, i_v * BV),
                                    (BT, BV), (1, 0)) if SAVE_NEW_VALUE else None
        b_v_new = tl.zeros([BT, BV], dtype=tl.float32)
        p_w = tl.make_block_ptr(w, (T, K), (stride_k, 1), (i_t * BT, 0), (BT, 64), (1, 0))
        b_w = tl.load(p_w, boundary_check=(0, 1))
        b_v_new += tl.dot(b_w, b_h1_cpy.to(b_w.dtype))
        if K > 64:
            p_w = tl.make_block_ptr(w, (T, K), (stride_k, 1), (i_t * BT, 64), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v_new += tl.dot(b_w, b_h2_cpy.to(b_w.dtype))
        if K > 128:
            p_w = tl.make_block_ptr(w, (T, K), (stride_k, 1), (i_t * BT, 128), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v_new += tl.dot(b_w, b_h3_cpy.to(b_w.dtype))
        if K > 192:
            p_w = tl.make_block_ptr(w, (T, K), (stride_k, 1), (i_t * BT, 192), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v_new += tl.dot(b_w, b_h4_cpy.to(b_w.dtype))
        b_v_new = -b_v_new + tl.load(p_v, boundary_check=(0, 1))

        if SAVE_NEW_VALUE:
            p_v_new = tl.make_block_ptr(v_new, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            tl.store(p_v_new, b_v_new.to(p_v_new.dtype.element_ty), boundary_check=(0, 1))

        if USE_G:
            m_t = (i_t * BT + tl.arange(0, BT)) < T
            # last_idx = min((i_t + 1) * BT, T) - 1
            last_idx = min((i_t_true + 1) * BT * num_householder, T) - 1
            b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
            p_g = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
            b_g = tl.load(p_g, boundary_check=(0,))
            b_v_new = b_v_new * tl.where(m_t, exp(b_g_last - b_g), 0)[:, None]
            b_g_last = exp(b_g_last)
            if i_t % num_householder == 0:
                b_h1 = b_h1 * b_g_last
                if K > 64:
                    b_h2 = b_h2 * b_g_last
                if K > 128:
                    b_h3 = b_h3 * b_g_last
                if K > 192:
                    b_h4 = b_h4 * b_g_last
        b_v_new = b_v_new.to(k.dtype.element_ty)
        p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (0, i_t * BT), (64, BT), (0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_h1 += tl.dot(b_k, b_v_new)
        if K > 64:
            p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (64, i_t * BT), (64, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h2 += tl.dot(b_k, b_v_new)
        if K > 128:
            p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (128, i_t * BT), (64, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h3 += tl.dot(b_k, b_v_new)
        if K > 192:
            p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (192, i_t * BT), (64, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h4 += tl.dot(b_k, b_v_new)
    # epilogue
    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            tl.store(p_ht, b_h2.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            tl.store(p_ht, b_h3.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            tl.store(p_ht, b_h4.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


# @triton.heuristics({
#     'USE_G': lambda args: args['g'] is not None,
#     'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
#     'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
#     'SAVE_NEW_VALUE': lambda args: args['v_new'] is not None,
#     'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
# })
# @triton.autotune(
#     configs=[
#         triton.Config({'BV': BV}, num_warps=num_warps, num_stages=num_stages)
#         for num_warps in [2, 4]
#         for num_stages in [2, 3, 4]
#         for BV in [32, 64]
#     ],
#     key=['H', 'K', 'V', 'BT', 'USE_G'],
#     use_cuda_graph=use_cuda_graph,
# )
# @triton.jit(do_not_specialize=['T'])
# def chunk_gated_delta_product_fwd_kernel_h_blockdim64(
#     k,
#     v,
#     w,
#     v_new,
#     g,
#     h,
#     h0,
#     ht,
#     cu_seqlens,
#     chunk_offsets,
#     T,
#     num_householder: tl.constexpr,  # number of delta products
#     H: tl.constexpr,
#     K: tl.constexpr,
#     V: tl.constexpr,
#     BT: tl.constexpr,
#     BV: tl.constexpr,
#     USE_G: tl.constexpr,
#     USE_INITIAL_STATE: tl.constexpr,
#     STORE_FINAL_STATE: tl.constexpr,
#     SAVE_NEW_VALUE: tl.constexpr,
#     IS_VARLEN: tl.constexpr,
# ):
#     i_v, i_nh = tl.program_id(0), tl.program_id(1)
#     i_n, i_h = i_nh // H, i_nh % H
#     if IS_VARLEN:
#         bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
#         T = eos - bos
#         NT = tl.cdiv(T, BT)
#         boh = tl.load(chunk_offsets + i_n).to(tl.int32)
#     else:
#         bos, eos = i_n * T, i_n * T + T
#         NT = tl.cdiv(T, BT)
#         boh = i_n * tl.cdiv(T // num_householder, BT)

#     # [BK, BV]
#     b_h1 = tl.zeros([64, BV], dtype=tl.float32)
#     if K > 64:
#         b_h2 = tl.zeros([64, BV], dtype=tl.float32)
#     if K > 128:
#         b_h3 = tl.zeros([64, BV], dtype=tl.float32)
#     if K > 192:
#         b_h4 = tl.zeros([64, BV], dtype=tl.float32)

#     # calculate offset
#     h += (boh * H + i_h) * K*V
#     v += (bos * H + i_h) * V
#     k += (bos * H + i_h) * K
#     w += (bos * H + i_h) * K
#     if SAVE_NEW_VALUE:
#         v_new += (bos * H + i_h) * V
#     stride_v = H*V
#     stride_h = H*K*V
#     stride_k = H*K
#     if USE_INITIAL_STATE:
#         h0 = h0 + i_nh * K*V
#     if STORE_FINAL_STATE:
#         ht = ht + i_nh * K*V

#     # load initial state
#     if USE_INITIAL_STATE:
#         p_h0_1 = tl.make_block_ptr(h0, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
#         b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
#         if K > 64:
#             p_h0_2 = tl.make_block_ptr(h0, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
#             b_h2 += tl.load(p_h0_2, boundary_check=(0, 1)).to(tl.float32)
#         if K > 128:
#             p_h0_3 = tl.make_block_ptr(h0, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
#             b_h3 += tl.load(p_h0_3, boundary_check=(0, 1)).to(tl.float32)
#         if K > 192:
#             p_h0_4 = tl.make_block_ptr(h0, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
#             b_h4 += tl.load(p_h0_4, boundary_check=(0, 1)).to(tl.float32)

#     # main recurrence
#     for i_t in range(NT):
#         if i_t % num_householder == 0:
#             i_t_true = i_t // num_householder
#             p_h1 = tl.make_block_ptr(h + i_t_true * stride_h, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
#             tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), boundary_check=(0, 1))
#             if K > 64:
#                 p_h2 = tl.make_block_ptr(h + i_t_true * stride_h, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
#                 tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), boundary_check=(0, 1))
#             if K > 128:
#                 p_h3 = tl.make_block_ptr(h + i_t_true * stride_h, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
#                 tl.store(p_h3, b_h3.to(p_h3.dtype.element_ty), boundary_check=(0, 1))
#             if K > 192:
#                 p_h4 = tl.make_block_ptr(h + i_t_true * stride_h, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
#                 tl.store(p_h4, b_h4.to(p_h4.dtype.element_ty), boundary_check=(0, 1))

#         p_v = tl.make_block_ptr(v, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
#         p_v_new = tl.make_block_ptr(v_new, (T, V), (stride_v, 1), (i_t * BT, i_v * BV),
#                                     (BT, BV), (1, 0)) if SAVE_NEW_VALUE else None
#         b_v_new = tl.zeros([BT, BV], dtype=tl.float32)
#         p_w = tl.make_block_ptr(w, (T, K), (stride_k, 1), (i_t * BT, 0), (BT, 64), (1, 0))
#         b_w = tl.load(p_w, boundary_check=(0, 1))
#         b_v_new += tl.dot(b_w, b_h1.to(b_w.dtype))
#         if K > 64:
#             p_w = tl.make_block_ptr(w, (T, K), (stride_k, 1), (i_t * BT, 64), (BT, 64), (1, 0))
#             b_w = tl.load(p_w, boundary_check=(0, 1))
#             b_v_new += tl.dot(b_w, b_h2.to(b_w.dtype))
#         if K > 128:
#             p_w = tl.make_block_ptr(w, (T, K), (stride_k, 1), (i_t * BT, 128), (BT, 64), (1, 0))
#             b_w = tl.load(p_w, boundary_check=(0, 1))
#             b_v_new += tl.dot(b_w, b_h3.to(b_w.dtype))
#         if K > 192:
#             p_w = tl.make_block_ptr(w, (T, K), (stride_k, 1), (i_t * BT, 192), (BT, 64), (1, 0))
#             b_w = tl.load(p_w, boundary_check=(0, 1))
#             b_v_new += tl.dot(b_w, b_h4.to(b_w.dtype))
#         b_v_new = -b_v_new + tl.load(p_v, boundary_check=(0, 1))

#         if SAVE_NEW_VALUE:
#             p_v_new = tl.make_block_ptr(v_new, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
#             tl.store(p_v_new, b_v_new.to(p_v_new.dtype.element_ty), boundary_check=(0, 1))

#         if USE_G:
#             m_t = (i_t * BT + tl.arange(0, BT)) < T
#             last_idx = min((i_t + 1) * BT, T) - 1
#             b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
#             p_g = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
#             b_g = tl.load(p_g, boundary_check=(0,))
#             b_v_new = b_v_new * tl.where(m_t, exp(b_g_last - b_g), 0)[:, None]
#             b_g_last = exp(b_g_last)
#             b_h1 = b_h1 * b_g_last
#             if K > 64:
#                 b_h2 = b_h2 * b_g_last
#             if K > 128:
#                 b_h3 = b_h3 * b_g_last
#             if K > 192:
#                 b_h4 = b_h4 * b_g_last
#         b_v_new = b_v_new.to(k.dtype.element_ty)
#         p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (0, i_t * BT), (64, BT), (0, 1))
#         b_k = tl.load(p_k, boundary_check=(0, 1))
#         b_h1 += tl.dot(b_k, b_v_new)
#         if K > 64:
#             p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (64, i_t * BT), (64, BT), (0, 1))
#             b_k = tl.load(p_k, boundary_check=(0, 1))
#             b_h2 += tl.dot(b_k, b_v_new)
#         if K > 128:
#             p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (128, i_t * BT), (64, BT), (0, 1))
#             b_k = tl.load(p_k, boundary_check=(0, 1))
#             b_h3 += tl.dot(b_k, b_v_new)
#         if K > 192:
#             p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (192, i_t * BT), (64, BT), (0, 1))
#             b_k = tl.load(p_k, boundary_check=(0, 1))
#             b_h4 += tl.dot(b_k, b_v_new)
#     # epilogue
#     if STORE_FINAL_STATE:
#         p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
#         tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
#         if K > 64:
#             p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
#             tl.store(p_ht, b_h2.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
#         if K > 128:
#             p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
#             tl.store(p_ht, b_h3.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
#         if K > 192:
#             p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
#             tl.store(p_ht, b_h4.to(p_ht.dtype.element_ty), boundary_check=(0, 1))



# @triton.heuristics({
#     'USE_G': lambda args: args['g'] is not None,
#     'USE_INITIAL_STATE': lambda args: args['dh0'] is not None,
#     'USE_FINAL_STATE_GRADIENT': lambda args: args['dht'] is not None,
#     'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
# })
# @triton.autotune(
#     configs=[
#         triton.Config({'BV': BV}, num_warps=num_warps, num_stages=num_stages)
#         for num_warps in [2, 4]
#         for num_stages in [4, 3, 2]
#         for BV in [64, 32]
#     ],
#     key=['H', 'K', 'V', 'BT', 'BV', 'USE_G'],
#     use_cuda_graph=use_cuda_graph,
# )
# @triton.jit(do_not_specialize=['T'])
# def chunk_gated_delta_product_bwd_kernel_dhu_blockdim64(
#     q,
#     k,
#     w,
#     g,
#     dht,
#     dh0,
#     do,
#     dh,
#     dv,
#     dv2,
#     cu_seqlens,
#     chunk_offsets,
#     scale,
#     T,
#     num_householder: tl.constexpr,
#     H: tl.constexpr,
#     K: tl.constexpr,
#     V: tl.constexpr,
#     BT: tl.constexpr,
#     BV: tl.constexpr,
#     USE_G: tl.constexpr,
#     USE_INITIAL_STATE: tl.constexpr,
#     USE_FINAL_STATE_GRADIENT: tl.constexpr,
#     IS_VARLEN: tl.constexpr
# ):
#     i_v, i_nh = tl.program_id(0), tl.program_id(1)
#     i_n, i_h = i_nh // H, i_nh % H
#     # i_n corresponds to batch and i_h corresponds to head

#     if IS_VARLEN:
#         bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
#         T = eos - bos # expanded tokens 
#         NT = tl.cdiv(T, BT)
#         NT_TRUE = tl.cdiv(T // num_householder, BT) 
#         # boh corresponds to 
#         boh = tl.load(chunk_offsets + i_n).to(tl.int32) 
#         # TODO: for each i_n in chunk_offsets, store index for which BT * num householder chunk 
#         # for each batch 
#         bos_q_o = i_n * T // num_householder
#     else:
#         bos, eos = i_n * T, i_n * T + T
#         NT = tl.cdiv(T, BT)
#         # boh = i_n * tl.cdiv(T, BT)
#         # Jinha: update boh to match the chunk_gated_delta_product_fwd_kernel_h_blockdim64 implementation
#         NT_TRUE = tl.cdiv(T // num_householder, BT) 
#         boh = i_n * tl.cdiv(T // num_householder, BT) 
#         # each batch has (T / (num householder * BT)) chunks 
#         bos_q_o = i_n * T // num_householder

#     # [BK, BV]
#     b_dh1 = tl.zeros([64, BV], dtype=tl.float32)
#     if K > 64:
#         b_dh2 = tl.zeros([64, BV], dtype=tl.float32)
#     if K > 128:
#         b_dh3 = tl.zeros([64, BV], dtype=tl.float32)
#     if K > 192:
#         b_dh4 = tl.zeros([64, BV], dtype=tl.float32) 
        
#     # q is (B, T_true, H, K)
#     # o is (B, T_true, H, V)
#     # do is (B, T_true, H, V) 
#     # h is (B, T_actual, H, K, V) 
#     # dh is (B, \ceil(T_true / BT), H, K, V)

#     # calculate offset
#     dh += (boh * H + i_h) * K*V
#     dv += (bos * H + i_h) * V
#     dv2 += (bos * H + i_h) * V
#     q += (bos_q_o * H + i_h) * K
#     k += (bos * H + i_h) * K
#     w += (bos * H + i_h) * K
#     do += (bos_q_o * H + i_h) * V
#     stride_v = H*V
#     stride_h = H*K*V
#     stride_k = H*K
#     if USE_INITIAL_STATE:
#         dh0 += i_nh * K*V
#     if USE_FINAL_STATE_GRADIENT:
#         dht += i_nh * K*V

#     # dh0, dh1, dh2 ..., dht, (dht should always exist at the end)

#     if USE_FINAL_STATE_GRADIENT:
#         p_dht1 = tl.make_block_ptr(dht, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
#         b_dh1 += tl.load(p_dht1, boundary_check=(0, 1))
#         if K > 64:
#             p_dht2 = tl.make_block_ptr(dht, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
#             b_dh2 += tl.load(p_dht2, boundary_check=(0, 1))
#         if K > 128:
#             p_dht3 = tl.make_block_ptr(dht, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
#             b_dh3 += tl.load(p_dht3, boundary_check=(0, 1))
#         if K > 192:
#             p_dht4 = tl.make_block_ptr(dht, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
#             b_dh4 += tl.load(p_dht4, boundary_check=(0, 1))

#     # ceil (NT // num_householder)
#     p_dh1 = tl.make_block_ptr(dh + (NT_TRUE-1)*stride_h, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
#     tl.store(p_dh1, b_dh1.to(p_dh1.dtype.element_ty), boundary_check=(0, 1))
#     if K > 64:
#         p_dh2 = tl.make_block_ptr(dh + (NT_TRUE-1)*stride_h, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
#         tl.store(p_dh2, b_dh2.to(p_dh2.dtype.element_ty), boundary_check=(0, 1))
#     if K > 128:
#         p_dh3 = tl.make_block_ptr(dh + (NT_TRUE-1)*stride_h, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
#         tl.store(p_dh3, b_dh3.to(p_dh3.dtype.element_ty), boundary_check=(0, 1))
#     if K > 192:
#         p_dh4 = tl.make_block_ptr(dh + (NT_TRUE-1)*stride_h, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
#         tl.store(p_dh4, b_dh4.to(p_dh4.dtype.element_ty), boundary_check=(0, 1))

#     # make copies of b_dh 
#     b_dh1_cpy = b_dh1.copy()
#     if K > 64:
#         b_dh2_cpy = b_dh2.copy()
#     if K > 128:
#         b_dh3_cpy = b_dh3.copy()
#     if K > 192:
#         b_dh4_cpy = b_dh4.copy()

#     NEW_BLOCK_BT_EXPANDED = True # flag to check if we are in a new BT * num_householder chunk

#     for i_t in range(NT - 1, -1, -1): 
#         # # TODO store every BT * num_householder chunks 
#         # if i_t % num_householder == 0: 
#         #     p_dh1 = tl.make_block_ptr(dh + i_t*stride_h, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
#         #     tl.store(p_dh1, b_dh1.to(p_dh1.dtype.element_ty), boundary_check=(0, 1))
#         #     if K > 64:
#         #         p_dh2 = tl.make_block_ptr(dh + i_t*stride_h, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
#         #         tl.store(p_dh2, b_dh2.to(p_dh2.dtype.element_ty), boundary_check=(0, 1))
#         #     if K > 128:
#         #         p_dh3 = tl.make_block_ptr(dh + i_t*stride_h, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
#         #         tl.store(p_dh3, b_dh3.to(p_dh3.dtype.element_ty), boundary_check=(0, 1))
#         #     if K > 192:
#         #         p_dh4 = tl.make_block_ptr(dh + i_t*stride_h, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
#         #         tl.store(p_dh4, b_dh4.to(p_dh4.dtype.element_ty), boundary_check=(0, 1))

#         i_th = i_t // num_householder
#         i_th_rem = i_t % num_householder

#         # TODO use g which is t_true and index by i_t // (BT * num householder)
#         # if USE_G:
#         #     last_idx = min((i_t + 1) * BT, T) - 1
#         #     bg_last = tl.load(g + (bos + last_idx) * H + i_h)
#         #     bg_last_exp = exp(bg_last)
#         #     p_g = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
#         #     b_g = tl.load(p_g, boundary_check=(0,))
#         #     b_g_exp = exp(b_g)
#         # else:
#         #     bg_last = None
#         #     last_idx = None
#         #     b_g = None
#         #     b_g_exp = None

#         if USE_G:
#             T_true = T // num_householder
#             last_idx = min((i_th) * BT + BT, T_true) - 1
#             bg_last = tl.load(g + (bos_q_o + last_idx) * H + i_h)
#             bg_last_exp = exp(bg_last)
#             p_g = tl.make_block_ptr(g + bos_q_o * H + i_h, (T_true,), (H,), (i_th * BT,), (BT,), (0,))
#             b_g = tl.load(p_g, boundary_check=(0,))
#             b_g_exp = exp(b_g)
#             # Create interleaved pattern: repeat each element of b_g num_householder times
#             indices = tl.arange(0, BT*num_householder) // num_householder
#             b_g_expanded = b_g[indices]
#             # Extract the slice for current remainder
#             start_idx = i_th_rem * BT
#             end_idx = (i_th_rem + 1) * BT
#             slice_indices = tl.arange(0, BT) + start_idx
#             b_g_interleaved = b_g_expanded[slice_indices]
#         else:
#             bg_last = None
#             last_idx = None
#             b_g = None
#             b_g_exp = None

#         p_dv = tl.make_block_ptr(dv, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
#         # p_do = tl.make_block_ptr(do, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
#         p_do = tl.make_block_ptr(do, (T_true, V), (stride_v, 1), (i_th * BT, i_v * BV), (BT, BV), (1, 0))
#         p_dv2 = tl.make_block_ptr(dv2, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))

#         b_do = tl.load(p_do, boundary_check=(0, 1))
#         b_dv = tl.zeros([BT, BV], dtype=tl.float32)
#         # block multiplication of b_k and b_dh to get b_dv 

#         # Update dv based on hidden state gradients
#         p_k = tl.make_block_ptr(k, (T, K), (stride_k, 1), (i_t * BT, 0), (BT, 64), (1, 0))
#         b_k = tl.load(p_k, boundary_check=(0, 1))
#         # b_dv += tl.dot(b_k, b_dh1.to(b_k.dtype))
#         b_dv += tl.dot(b_k, b_dh1_cpy.to(b_k.dtype))

#         if K > 64:
#             p_k = tl.make_block_ptr(k, (T, K), (stride_k, 1), (i_t * BT, 64), (BT, 64), (1, 0))
#             b_k = tl.load(p_k, boundary_check=(0, 1))
#             # b_dv += tl.dot(b_k, b_dh2.to(b_k.dtype))
#             b_dv += tl.dot(b_k, b_dh2_cpy.to(b_k.dtype))

#         if K > 128:
#             p_k = tl.make_block_ptr(k, (T, K), (stride_k, 1), (i_t * BT, 128), (BT, 64), (1, 0))
#             b_k = tl.load(p_k, boundary_check=(0, 1))
#             # b_dv += tl.dot(b_k, b_dh3.to(b_k.dtype))
#             b_dv += tl.dot(b_k, b_dh3_cpy.to(b_k.dtype))

#         if K > 192:
#             p_k = tl.make_block_ptr(k, (T, K), (stride_k, 1), (i_t * BT, 192), (BT, 64), (1, 0))
#             b_k = tl.load(p_k, boundary_check=(0, 1))
#             # b_dv += tl.dot(b_k, b_dh4.to(b_k.dtype))
#             b_dv += tl.dot(b_k, b_dh4_cpy.to(b_k.dtype))

#         if USE_G:
#             m_t = (i_t * BT + tl.arange(0, BT)) < T 
#             # TODO use b_g 
#             # b_dv *= tl.where(m_t, exp(bg_last - b_g), 0)[:, None] 
#             b_dv *= tl.where(m_t, exp(bg_last - b_g_interleaved), 0)[:, None] 
#         b_dv += tl.load(p_dv, boundary_check=(0, 1))

#         tl.store(p_dv2, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))

#         # Update hidden state gradients
#         p_w = tl.make_block_ptr(w, (K, T), (1, stride_k), (0, i_t * BT), (64, BT), (0, 1))
#         # p_q = tl.make_block_ptr(q, (K, T), (1, stride_k), (0, i_t * BT), (64, BT), (0, 1))
#         b_w = tl.load(p_w, boundary_check=(0, 1))
#         if NEW_BLOCK_BT_EXPANDED:
#             p_q = tl.make_block_ptr(q, (K, T_true), (1, stride_k), (0, i_th * BT), (64, BT), (0, 1))
#             b_q = tl.load(p_q, boundary_check=(0, 1))
#             if USE_G:
#                 b_dh1 *= bg_last_exp
#                 b_q = b_q * b_g_exp[None, :]
#             b_q = (b_q * scale).to(b_q.dtype)

#             # TODO increment b_q @ b_do only once for each BT * num_householder chunk (check by dividing)
#             # TODO block sum of b_w, b_dv 
#             b_dh1 += tl.dot(b_q, b_do.to(b_q.dtype)) 

#         b_dh1 -= tl.dot(b_w, b_dv.to(b_w.dtype))

#         if K > 64:
#             p_w = tl.make_block_ptr(w, (K, T), (1, stride_k), (64, i_t * BT), (64, BT), (0, 1))
#             b_w = tl.load(p_w, boundary_check=(0, 1))
#             if NEW_BLOCK_BT_EXPANDED:
#                 p_q = tl.make_block_ptr(q, (K, T_true), (1, stride_k), (64, i_th * BT), (64, BT), (0, 1))
#                 b_q = tl.load(p_q, boundary_check=(0, 1))
#                 if USE_G:
#                     b_dh2 *= bg_last_exp
#                     b_q = b_q * b_g_exp[None, :]
#                 b_q = (b_q * scale).to(b_q.dtype)
#                 b_dh2 += tl.dot(b_q, b_do.to(b_q.dtype)) 
#             b_dh2 -= tl.dot(b_w, b_dv.to(b_w.dtype))

#         if K > 128:
#             p_w = tl.make_block_ptr(w, (K, T), (1, stride_k), (128, i_t * BT), (64, BT), (0, 1))
#             b_w = tl.load(p_w, boundary_check=(0, 1))
#             if NEW_BLOCK_BT_EXPANDED:
#                 p_q = tl.make_block_ptr(q, (K, T_true), (1, stride_k), (128, i_th * BT), (64, BT), (0, 1))
#                 b_q = tl.load(p_q, boundary_check=(0, 1))
#                 if USE_G:
#                     b_dh3 *= bg_last_exp
#                     b_q = b_q * b_g_exp[None, :]
#                 b_q = (b_q * scale).to(b_q.dtype)
#                 b_dh3 += tl.dot(b_q, b_do.to(b_q.dtype)) 
#             b_dh3 -= tl.dot(b_w, b_dv.to(b_w.dtype))

#         if K > 192:
#             p_w = tl.make_block_ptr(w, (K, T), (1, stride_k), (192, i_t * BT), (64, BT), (0, 1))
#             b_w = tl.load(p_w, boundary_check=(0, 1))
#             if NEW_BLOCK_BT_EXPANDED:
#                 p_q = tl.make_block_ptr(q, (K, T_true), (1, stride_k), (192, i_th * BT), (64, BT), (0, 1))
#                 b_q = tl.load(p_q, boundary_check=(0, 1))
#                 if USE_G:
#                     b_dh4 *= bg_last_exp
#                     b_q = b_q * b_g_exp[None, :]
#                 b_q = (b_q * scale).to(b_q.dtype)
#                 b_dh4 += tl.dot(b_q, b_do.to(b_q.dtype)) 
#             b_dh4 -= tl.dot(b_w, b_dv.to(b_w.dtype))

#         NEW_BLOCK_BT_EXPANDED = False 

#         # TODO store every BT * num_householder chunks 
#         # don't store initial state 

#         if i_t % num_householder == 0 and i_t > 0: 
#             NEW_BLOCK_BT_EXPANDED = True 
#             p_dh1 = tl.make_block_ptr(dh + i_th*stride_h, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
#             tl.store(p_dh1, b_dh1.to(p_dh1.dtype.element_ty), boundary_check=(0, 1))
#             if K > 64:
#                 p_dh2 = tl.make_block_ptr(dh + i_th*stride_h, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
#                 tl.store(p_dh2, b_dh2.to(p_dh2.dtype.element_ty), boundary_check=(0, 1))
#             if K > 128:
#                 p_dh3 = tl.make_block_ptr(dh + i_th*stride_h, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
#                 tl.store(p_dh3, b_dh3.to(p_dh3.dtype.element_ty), boundary_check=(0, 1))
#             if K > 192:
#                 p_dh4 = tl.make_block_ptr(dh + i_th*stride_h, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
#                 tl.store(p_dh4, b_dh4.to(p_dh4.dtype.element_ty), boundary_check=(0, 1))

#             # make copies of b_dh 
#             b_dh1_cpy = b_dh1.copy()
#             if K > 64:
#                 b_dh2_cpy = b_dh2.copy()
#             if K > 128:
#                 b_dh3_cpy = b_dh3.copy()
#             if K > 192:
#                 b_dh4_cpy = b_dh4.copy()


#     if USE_INITIAL_STATE:
#         p_dh0 = tl.make_block_ptr(dh0, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
#         tl.store(p_dh0, b_dh1.to(p_dh0.dtype.element_ty), boundary_check=(0, 1))
#         if K > 64:
#             p_dh1 = tl.make_block_ptr(dh0, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
#             tl.store(p_dh1, b_dh2.to(p_dh1.dtype.element_ty), boundary_check=(0, 1))
#         if K > 128:
#             p_dh2 = tl.make_block_ptr(dh0, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
#             tl.store(p_dh2, b_dh3.to(p_dh2.dtype.element_ty), boundary_check=(0, 1))
#         if K > 192:
#             p_dh3 = tl.make_block_ptr(dh0, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
#             tl.store(p_dh3, b_dh4.to(p_dh3.dtype.element_ty), boundary_check=(0, 1))


def chunk_gated_delta_product_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    chunk_size: int = 64,  # SY: remove this argument and force chunk size 64?
    save_new_value: bool = True,
    cu_seqlens: Optional[torch.LongTensor] = None,
    num_householder: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, u.shape[-1]
    assert T % num_householder == 0, "T must be divisible by num_householder"
    T_true = T // num_householder
    BT = chunk_size
    chunk_indices = prepare_chunk_indices(cu_seqlens // num_householder, chunk_size) if cu_seqlens is not None else None
    # N: the actual number of sequences in the batch with either equal or variable lengths
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T_true, BT), None
    else:
        N, NT, chunk_offsets = len(cu_seqlens) - \
            1, len(chunk_indices), prepare_chunk_offsets(cu_seqlens // num_householder, BT)
    assert K <= 256, "current kernel does not support head dimension larger than 256."
    h = k.new_empty(B, NT, H, K, V)
    final_state = k.new_empty(N, H, K, V, dtype=torch.float32) if output_final_state else None
    v_new = torch.empty_like(u) if save_new_value else None

    def grid(meta): return (triton.cdiv(V, meta['BV']), N*H)
    chunk_gated_delta_product_fwd_kernel_h_blockdim64[grid](
        k=k,
        v=u,
        w=w,
        v_new=v_new,
        g=g,
        h=h,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        num_householder=num_householder,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT
    )
    return h, v_new, final_state


# def chunk_gated_delta_product_bwd_dhu(
#     q: torch.Tensor,
#     k: torch.Tensor,
#     v: torch.Tensor,  # v_new from forward pass
#     g: torch.Tensor,
#     h0: torch.Tensor,
#     dht: Optional[torch.Tensor],
#     do: torch.Tensor,
#     dv: torch.Tensor,
#     scale: float,
#     cu_seqlens: Optional[torch.LongTensor] = None,
#     chunk_size: int = 64,
#     num_householder: int = 1,
# ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#     # q and do are original (non-expanded), k is expanded
#     B, T_true, H, K = q.shape
#     V = do.shape[-1]
#     T = k.shape[1]  # Expanded T from k
#     assert T == T_true * num_householder, f"k.shape[1] ({T}) must equal q.shape[1] * num_householder ({T_true * num_householder})"
#     assert T % num_householder == 0, "T must be divisible by num_householder"

#     # N: the actual number of sequences in the batch with either equal or variable lengths
#     BT = 64
#     assert K <= 256, "current kernel does not support head dimension being larger than 256."

#     # Jinha: each index in cu_seqlens is expected to be divisible by num householder 
#     chunk_indices = prepare_chunk_indices(cu_seqlens // num_householder, BT) if cu_seqlens is not None else None
#     if cu_seqlens is None:
#         N, NT, chunk_offsets = B, triton.cdiv(T // num_householder, BT), None
#     else:
#         N, NT, chunk_offsets = (
#             len(cu_seqlens) - 1, len(chunk_indices),
#             prepare_chunk_offsets(cu_seqlens // num_householder, BT)
#         )

#     # each NT block corresponds to num_householder * BT 
#     dh = q.new_empty(B, NT, H, K, V)
#     dh0 = torch.empty_like(h0, dtype=torch.float32) if h0 is not None else None
#     dv2 = torch.empty_like(dv)

#     def grid(meta): return (triton.cdiv(V, meta['BV']), N*H)
#     # Extract w from v (v_new contains the values after householder transformation)
#     # In the forward pass, v_new = u - w @ h, so we need to reconstruct w
#     # For now, pass v as w parameter since kernel expects w
#     chunk_gated_delta_product_bwd_kernel_dhu_blockdim64[grid](
#         q=q,
#         k=k,
#         w=v,  # Pass v_new as w for compatibility
#         g=g,
#         dht=dht,
#         dh0=dh0,
#         do=do,
#         dh=dh,
#         dv=dv,
#         dv2=dv2,
#         cu_seqlens=cu_seqlens,
#         chunk_offsets=chunk_offsets,
#         scale=scale,
#         T=T,
#         num_householder=num_householder,
#         H=H,
#         K=K,
#         V=V,
#         BT=BT,
#     )
#     # could call chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64 instead
#     # after adjusting number of tokens
#     return dh, dh0, dv2


def chunk_gated_delta_product_bwd_dhu(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    g: torch.Tensor,
    h0: torch.Tensor,
    dht: Optional[torch.Tensor],
    do: torch.Tensor,
    dv: torch.Tensor,
    scale: float,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_size: int = 64,  # SY: remove this argument and force chunk size 64?
    num_householder: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *q.shape, do.shape[-1]
    # N: the actual number of sequences in the batch with either equal or variable lengths
    chunk_indices = prepare_chunk_indices(cu_seqlens // num_householder, chunk_size) if cu_seqlens is not None else None
    BT = chunk_size
    assert K <= 256, "current kernel does not support head dimension being larger than 256."

    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T // num_householder, BT), None
    else:
        N, NT, chunk_offsets = len(cu_seqlens) - 1, len(chunk_indices), prepare_chunk_offsets(cu_seqlens // num_householder, BT)

    dh = q.new_empty(B, NT, H, K, V)
    dh0 = torch.empty_like(h0, dtype=torch.float32) if h0 is not None else None
    dv2 = torch.empty_like(dv)

    def grid(meta): return (triton.cdiv(V, meta['BV']), N*H)
    chunk_gated_delta_product_bwd_kernel_dhu_blockdim64[grid](
        q=q,
        k=k,
        w=w,
        g=g,
        dht=dht,
        dh0=dh0,
        do=do,
        dh=dh,
        dv=dv,
        dv2=dv2,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        scale=scale,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        num_householder=num_householder,
    )
    return dh, dh0, dv2


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_INITIAL_STATE': lambda args: args['dh0'] is not None,
    'USE_FINAL_STATE_GRADIENT': lambda args: args['dht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        for num_stages in [4, 3, 2]
        for BV in [64, 32]
    ],
    key=['H', 'K', 'V', 'BT', 'BV', 'USE_G'],
    use_cuda_graph=use_cuda_graph,
)
@triton.jit(do_not_specialize=['T'])
def chunk_gated_delta_product_bwd_kernel_dhu_blockdim64(
    q,
    k,
    w,
    g,
    dht,
    dh0,
    do,
    dh,
    dv,
    dv2,
    cu_seqlens,
    chunk_offsets,
    scale,
    T,
    num_householder: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr,
    IS_VARLEN: tl.constexpr
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        NT_TRUE = tl.cdiv(T // num_householder, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        # Jinha: i_n is which batch, bos is which token 
        # boh is which state 
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        NT_TRUE = tl.cdiv(T // num_householder, BT)
        boh = i_n * tl.cdiv(T // num_householder, BT)


    # [BK, BV]
    b_dh1 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 64:
        b_dh2 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 128:
        b_dh3 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 192:
        b_dh4 = tl.zeros([64, BV], dtype=tl.float32)

    # calculate offset
    dh += (boh * H + i_h) * K*V
    dv += (bos * H + i_h) * V
    dv2 += (bos * H + i_h) * V
    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    w += (bos * H + i_h) * K
    do += (bos * H + i_h) * V
    stride_v = H*V
    stride_h = H*K*V
    stride_k = H*K
    if USE_INITIAL_STATE:
        dh0 += i_nh * K*V
    if USE_FINAL_STATE_GRADIENT:
        dht += i_nh * K*V

    if USE_FINAL_STATE_GRADIENT:
        p_dht1 = tl.make_block_ptr(dht, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        b_dh1 += tl.load(p_dht1, boundary_check=(0, 1))
        if K > 64:
            p_dht2 = tl.make_block_ptr(dht, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            b_dh2 += tl.load(p_dht2, boundary_check=(0, 1))
        if K > 128:
            p_dht3 = tl.make_block_ptr(dht, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            b_dh3 += tl.load(p_dht3, boundary_check=(0, 1))
        if K > 192:
            p_dht4 = tl.make_block_ptr(dht, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            b_dh4 += tl.load(p_dht4, boundary_check=(0, 1))

    for i_t in range(NT - 1, -1, -1):
        if i_t + 1 == NT or (i_t + 1) % num_householder == 0:  
            i_t_true = NT_TRUE - 1 if i_t + 1 == NT else ((i_t + 1) // num_householder) - 1 
            # NT - 1 
            p_dh1 = tl.make_block_ptr(dh + i_t_true*stride_h, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
            tl.store(p_dh1, b_dh1.to(p_dh1.dtype.element_ty), boundary_check=(0, 1))
            if K > 64:
                p_dh2 = tl.make_block_ptr(dh + i_t_true*stride_h, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
                tl.store(p_dh2, b_dh2.to(p_dh2.dtype.element_ty), boundary_check=(0, 1))
            if K > 128:
                p_dh3 = tl.make_block_ptr(dh + i_t_true*stride_h, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
                tl.store(p_dh3, b_dh3.to(p_dh3.dtype.element_ty), boundary_check=(0, 1))
            if K > 192:
                p_dh4 = tl.make_block_ptr(dh + i_t_true*stride_h, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
                tl.store(p_dh4, b_dh4.to(p_dh4.dtype.element_ty), boundary_check=(0, 1))

        if USE_G:
            last_idx = min((i_t + 1) * BT, T) - 1
            bg_last = tl.load(g + (bos + last_idx) * H + i_h)
            bg_last_exp = exp(bg_last)
            p_g = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
            b_g = tl.load(p_g, boundary_check=(0,))
            b_g_exp = exp(b_g)
        else:
            bg_last = None
            last_idx = None
            b_g = None
            b_g_exp = None

        p_dv = tl.make_block_ptr(dv, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_do = tl.make_block_ptr(do, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dv2 = tl.make_block_ptr(dv2, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))

        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_dv = tl.zeros([BT, BV], dtype=tl.float32)

        # Update dv
        p_k = tl.make_block_ptr(k, (T, K), (stride_k, 1), (i_t * BT, 0), (BT, 64), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_dv += tl.dot(b_k, b_dh1.to(b_k.dtype))

        if K > 64:
            p_k = tl.make_block_ptr(k, (T, K), (stride_k, 1), (i_t * BT, 64), (BT, 64), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_dv += tl.dot(b_k, b_dh2.to(b_k.dtype))

        if K > 128:
            p_k = tl.make_block_ptr(k, (T, K), (stride_k, 1), (i_t * BT, 128), (BT, 64), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_dv += tl.dot(b_k, b_dh3.to(b_k.dtype))

        if K > 192:
            p_k = tl.make_block_ptr(k, (T, K), (stride_k, 1), (i_t * BT, 192), (BT, 64), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_dv += tl.dot(b_k, b_dh4.to(b_k.dtype))

        if USE_G:
            m_t = (i_t * BT + tl.arange(0, BT)) < T
            b_dv *= tl.where(m_t, exp(bg_last - b_g), 0)[:, None]
        b_dv += tl.load(p_dv, boundary_check=(0, 1))

        tl.store(p_dv2, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
        # Update dh
        p_w = tl.make_block_ptr(w, (K, T), (1, stride_k), (0, i_t * BT), (64, BT), (0, 1))
        p_q = tl.make_block_ptr(q, (K, T), (1, stride_k), (0, i_t * BT), (64, BT), (0, 1))
        b_w = tl.load(p_w, boundary_check=(0, 1))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        if USE_G:
            b_dh1 *= bg_last_exp
            b_q = b_q * b_g_exp[None, :]
        b_q = (b_q * scale).to(b_q.dtype)
        b_dh1 += tl.dot(b_q, b_do.to(b_q.dtype))-tl.dot(b_w, b_dv.to(b_w.dtype))
        if K > 64:
            p_q = tl.make_block_ptr(q, (K, T), (1, stride_k), (64, i_t * BT), (64, BT), (0, 1))
            p_w = tl.make_block_ptr(w, (K, T), (1, stride_k), (64, i_t * BT), (64, BT), (0, 1))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            if USE_G:
                b_dh2 *= bg_last_exp
                b_q = b_q * b_g_exp[None, :]
            b_q = (b_q * scale).to(b_q.dtype)
            b_dh2 += tl.dot(b_q, b_do.to(b_q.dtype))-tl.dot(b_w, b_dv.to(b_w.dtype))
        if K > 128:
            p_q = tl.make_block_ptr(q, (K, T), (1, stride_k), (128, i_t * BT), (64, BT), (0, 1))
            p_w = tl.make_block_ptr(w, (K, T), (1, stride_k), (128, i_t * BT), (64, BT), (0, 1))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            if USE_G:
                b_dh3 *= bg_last_exp
                b_q = b_q * b_g_exp[None, :]
            b_q = (b_q * scale).to(b_q.dtype)
            b_dh3 += tl.dot(b_q, b_do.to(b_q.dtype))-tl.dot(b_w, b_dv.to(b_w.dtype))
        if K > 192:
            p_q = tl.make_block_ptr(q, (K, T), (1, stride_k), (192, i_t * BT), (64, BT), (0, 1))
            p_w = tl.make_block_ptr(w, (K, T), (1, stride_k), (192, i_t * BT), (64, BT), (0, 1))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            if USE_G:
                b_dh4 *= bg_last_exp
                b_q = b_q * b_g_exp[None, :]
            b_q = (b_q * scale).to(b_q.dtype)
            b_dh4 += tl.dot(b_q, b_do.to(b_q.dtype))-tl.dot(b_w, b_dv.to(b_w.dtype))

    if USE_INITIAL_STATE:
        p_dh0 = tl.make_block_ptr(dh0, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        tl.store(p_dh0, b_dh1.to(p_dh0.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_dh1 = tl.make_block_ptr(dh0, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            tl.store(p_dh1, b_dh2.to(p_dh1.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_dh2 = tl.make_block_ptr(dh0, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            tl.store(p_dh2, b_dh3.to(p_dh2.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_dh3 = tl.make_block_ptr(dh0, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            tl.store(p_dh3, b_dh4.to(p_dh3.dtype.element_ty), boundary_check=(0, 1))
