# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import os
from typing import Optional

import torch
import triton
import triton.language as tl

from fla.ops.utils.index import prepare_chunk_indices
from fla.ops.utils.op import make_tensor_descriptor
from fla.utils import input_guard, is_amd, is_tma_supported

FLA_TRIL_PRECISION = os.environ.get('FLA_TRIL_PRECISION', 'ieee')
ALLOWED_TRIL_PRECISIONS = ['ieee', 'tf32'] if is_amd else ['ieee', 'tf32', 'tf32x3']
assert FLA_TRIL_PRECISION in ALLOWED_TRIL_PRECISIONS, \
    f'FLA_TRIL_PRECISION must be one of {ALLOWED_TRIL_PRECISIONS}, but got {FLA_TRIL_PRECISION}'


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4, 8]
        for num_stages in [2, 3, 4, 5]
    ],
    key=['BT'],
)
@triton.jit(do_not_specialize=['T'])
def solve_tril_16x16_kernel(
    A,
    Ai,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    USE_TMA: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    DOT_PRECISION: tl.constexpr
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    o_i = tl.arange(0, 16)
    m_A = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]

    A = A + (bos*H + i_h) * BT
    Ai = Ai + (bos*H + i_h) * 16

    offset = (i_t * 16) % BT
    if not USE_TMA:
        p_A = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * 16, offset), (16, 16), (1, 0))
        # [16, 16]
        b_A = tl.load(p_A, boundary_check=(0, 1)).to(tl.float32)
    else:
        desc = make_tensor_descriptor(A, [T, BT], [H*BT, 1], [16, 16])
        desc_o = make_tensor_descriptor(Ai, [T, 16], [H*16, 1], [16, 16])
        b_A = desc.load([i_t * 16, offset]).to(tl.float32)
    b_A = -tl.where(m_A, b_A, 0)

    for i in range(2, min(16, T - i_t * 16)):
        # [16]
        b_a = -tl.load(A + (i_t * 16 + i) * H*BT + o_i + offset)
        b_a = b_a + tl.sum(b_a[:, None] * b_A, 0)
        b_A = tl.where((o_i == i)[:, None], b_a, b_A)
    b_A += m_I
    if not USE_TMA:
        p_Ai = tl.make_block_ptr(Ai, (T, 16), (H*16, 1), (i_t * 16, 0), (16, 16), (1, 0))
        tl.store(p_Ai, b_A.to(p_Ai.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    else:
        desc_o.store([i_t * 16, 0], b_A.to(desc_o.dtype, fp_downcast_rounding="rtne"))


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4, 8]
        for num_stages in [2, 3, 4, 5]
    ],
    key=['H', 'BT', 'IS_VARLEN'],
)
@triton.jit(do_not_specialize=['T'])
def merge_16x16_to_32x32_inverse_kernel(
    A,
    Ai,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    USE_TMA: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    DOT_PRECISION: tl.constexpr
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    o_i = tl.arange(0, 16)
    m_A = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]
    A += (bos * H + i_h) * BT
    Ai += (bos * H + i_h) * BT

    if not USE_TMA:
        p_A_11 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT, 0), (16, 16), (1, 0))
        p_A_22 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 16, 16), (16, 16), (1, 0))
        b_Ai_11 = tl.load(p_A_11, boundary_check=(0, 1)).to(tl.float32)
        b_Ai_22 = tl.load(p_A_22, boundary_check=(0, 1)).to(tl.float32)
    else:
        desc = make_tensor_descriptor(A, [T, BT], [H*BT, 1], [16, 16])
        desc_o = make_tensor_descriptor(Ai, [T, BT], [H*BT, 1], [16, 16])
        b_Ai_11 = desc.load([i_t * BT + 0, 0]).to(tl.float32)
        b_Ai_22 = desc.load([i_t * BT + 16, 16]).to(tl.float32)

    # [16, 16]
    b_Ai_11 = -tl.where(m_A, b_Ai_11, 0)
    b_Ai_22 = -tl.where(m_A, b_Ai_22, 0)

    for i in range(2, min(16, T - i_t * BT)):
        b_a_11 = -tl.load(A + (i_t * BT + i) * H*BT + o_i)
        b_a_11 += tl.sum(b_a_11[:, None] * b_Ai_11, 0)
        b_Ai_11 = tl.where((o_i == i)[:, None], b_a_11, b_Ai_11)
    for i in range(16 + 2, min(32, T - i_t * BT)):
        b_a_22 = -tl.load(A + (i_t * BT + i) * H*BT + o_i + 16)
        b_a_22 += tl.sum(b_a_22[:, None] * b_Ai_22, 0)
        b_Ai_22 = tl.where((o_i == i - 16)[:, None], b_a_22, b_Ai_22)

    b_Ai_11 += m_I
    b_Ai_22 += m_I

    if not USE_TMA:
        p_A_21 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 16, 0), (16, 16), (1, 0))
        b_A_21 = tl.load(p_A_21, boundary_check=(0, 1)).to(tl.float32)
    else:
        b_A_21 = desc.load([i_t * BT + 16, 0]).to(tl.float32)

    b_Ai_21 = -tl.dot(tl.dot(b_Ai_22, b_A_21, input_precision=DOT_PRECISION), b_Ai_11, input_precision=DOT_PRECISION)

    if not USE_TMA:
        p_Ai_11 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT, 0), (16, 16), (1, 0))
        p_Ai_21 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 16, 0), (16, 16), (1, 0))
        p_Ai_22 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 16, 16), (16, 16), (1, 0))
        tl.store(p_Ai_11, b_Ai_11.to(p_Ai_11.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
        tl.store(p_Ai_22, b_Ai_22.to(p_Ai_22.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
        tl.store(p_Ai_21, b_Ai_21.to(p_Ai_21.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    else:
        desc_o.store([i_t * BT + 0, 0], b_Ai_11.to(desc_o.dtype, fp_downcast_rounding="rtne"))
        desc_o.store([i_t * BT + 16, 0], b_Ai_21.to(desc_o.dtype, fp_downcast_rounding="rtne"))
        desc_o.store([i_t * BT + 16, 16], b_Ai_22.to(desc_o.dtype, fp_downcast_rounding="rtne"))


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4, 5]
    ],
    key=['H', 'BT', 'IS_VARLEN'],
)
@triton.jit(do_not_specialize=['T'])
def merge_16x16_to_64x64_inverse_kernel(
    A,
    Ai,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    USE_TMA: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    DOT_PRECISION: tl.constexpr
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    o_i = tl.arange(0, 16)
    m_A = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]
    A += (bos * H + i_h) * BT
    Ai += (bos * H + i_h) * BT

    if not USE_TMA:
        p_A_11 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT, 0), (16, 16), (1, 0))
        p_A_22 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 16, 16), (16, 16), (1, 0))
        p_A_33 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 32, 32), (16, 16), (1, 0))
        p_A_44 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 48, 48), (16, 16), (1, 0))
        b_Ai_11 = tl.load(p_A_11, boundary_check=(0, 1)).to(tl.float32)
        b_Ai_22 = tl.load(p_A_22, boundary_check=(0, 1)).to(tl.float32)
        b_Ai_33 = tl.load(p_A_33, boundary_check=(0, 1)).to(tl.float32)
        b_Ai_44 = tl.load(p_A_44, boundary_check=(0, 1)).to(tl.float32)
    else:
        desc = make_tensor_descriptor(A, [T, BT], [H*BT, 1], [16, 16])
        desc_o = make_tensor_descriptor(Ai, [T, BT], [H*BT, 1], [16, 16])
        b_Ai_11 = desc.load([i_t * BT + 0, 0]).to(tl.float32)
        b_Ai_22 = desc.load([i_t * BT + 16, 16]).to(tl.float32)
        b_Ai_33 = desc.load([i_t * BT + 32, 32]).to(tl.float32)
        b_Ai_44 = desc.load([i_t * BT + 48, 48]).to(tl.float32)

    # [16, 16]
    b_Ai_11 = -tl.where(m_A, b_Ai_11, 0)
    b_Ai_22 = -tl.where(m_A, b_Ai_22, 0)
    b_Ai_33 = -tl.where(m_A, b_Ai_33, 0)
    b_Ai_44 = -tl.where(m_A, b_Ai_44, 0)

    for i in range(2, min(16, T - i_t * BT)):
        b_a_11 = -tl.load(A + (i_t * BT + i) * H*BT + o_i)
        b_a_11 += tl.sum(b_a_11[:, None] * b_Ai_11, 0)
        b_Ai_11 = tl.where((o_i == i)[:, None], b_a_11, b_Ai_11)
    for i in range(16 + 2, min(32, T - i_t * BT)):
        b_a_22 = -tl.load(A + (i_t * BT + i) * H*BT + o_i + 16)
        b_a_22 += tl.sum(b_a_22[:, None] * b_Ai_22, 0)
        b_Ai_22 = tl.where((o_i == i - 16)[:, None], b_a_22, b_Ai_22)
    for i in range(32 + 2, min(48, T - i_t * BT)):
        b_a_33 = -tl.load(A + (i_t * BT + i) * H*BT + o_i + 32)
        b_a_33 += tl.sum(b_a_33[:, None] * b_Ai_33, 0)
        b_Ai_33 = tl.where((o_i == i - 32)[:, None], b_a_33, b_Ai_33)
    for i in range(48 + 2, min(64, T - i_t * BT)):
        b_a_44 = -tl.load(A + (i_t * BT + i) * H*BT + o_i + 48)
        b_a_44 += tl.sum(b_a_44[:, None] * b_Ai_44, 0)
        b_Ai_44 = tl.where((o_i == i - 48)[:, None], b_a_44, b_Ai_44)
    b_Ai_11 += m_I
    b_Ai_22 += m_I
    b_Ai_33 += m_I
    b_Ai_44 += m_I

    if not USE_TMA:
        p_A_21 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 16, 0), (16, 16), (1, 0))
        p_A_31 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 32, 0), (16, 16), (1, 0))
        p_A_32 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 32, 16), (16, 16), (1, 0))
        p_A_41 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 48, 0), (16, 16), (1, 0))
        p_A_42 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 48, 16), (16, 16), (1, 0))
        p_A_43 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 48, 32), (16, 16), (1, 0))
        b_A_21 = tl.load(p_A_21, boundary_check=(0, 1)).to(tl.float32)
        b_A_31 = tl.load(p_A_31, boundary_check=(0, 1)).to(tl.float32)
        b_A_32 = tl.load(p_A_32, boundary_check=(0, 1)).to(tl.float32)
        b_A_41 = tl.load(p_A_41, boundary_check=(0, 1)).to(tl.float32)
        b_A_42 = tl.load(p_A_42, boundary_check=(0, 1)).to(tl.float32)
        b_A_43 = tl.load(p_A_43, boundary_check=(0, 1)).to(tl.float32)
    else:
        b_A_21 = desc.load([i_t * BT + 16, 0]).to(tl.float32)
        b_A_31 = desc.load([i_t * BT + 32, 0]).to(tl.float32)
        b_A_32 = desc.load([i_t * BT + 32, 16]).to(tl.float32)
        b_A_41 = desc.load([i_t * BT + 48, 0]).to(tl.float32)
        b_A_42 = desc.load([i_t * BT + 48, 16]).to(tl.float32)
        b_A_43 = desc.load([i_t * BT + 48, 32]).to(tl.float32)

    b_Ai_21 = -tl.dot(tl.dot(b_Ai_22, b_A_21, input_precision=DOT_PRECISION), b_Ai_11, input_precision=DOT_PRECISION)
    b_Ai_32 = -tl.dot(tl.dot(b_Ai_33, b_A_32, input_precision=DOT_PRECISION), b_Ai_22, input_precision=DOT_PRECISION)
    b_Ai_43 = -tl.dot(tl.dot(b_Ai_44, b_A_43, input_precision=DOT_PRECISION), b_Ai_33, input_precision=DOT_PRECISION)

    b_Ai_31 = -tl.dot(
        b_Ai_33,
        tl.dot(b_A_31, b_Ai_11, input_precision=DOT_PRECISION) +
        tl.dot(b_A_32, b_Ai_21, input_precision=DOT_PRECISION),
        input_precision=DOT_PRECISION
    )
    b_Ai_42 = -tl.dot(
        b_Ai_44,
        tl.dot(b_A_42, b_Ai_22, input_precision=DOT_PRECISION) +
        tl.dot(b_A_43, b_Ai_32, input_precision=DOT_PRECISION),
        input_precision=DOT_PRECISION
    )
    b_Ai_41 = -tl.dot(
        b_Ai_44,
        tl.dot(b_A_41, b_Ai_11, input_precision=DOT_PRECISION) +
        tl.dot(b_A_42, b_Ai_21, input_precision=DOT_PRECISION) +
        tl.dot(b_A_43, b_Ai_31, input_precision=DOT_PRECISION),
        input_precision=DOT_PRECISION
    )

    if not USE_TMA:
        p_Ai_11 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT, 0), (16, 16), (1, 0))
        p_Ai_22 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 16, 16), (16, 16), (1, 0))
        p_Ai_33 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 32, 32), (16, 16), (1, 0))
        p_Ai_44 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 48, 48), (16, 16), (1, 0))
        p_Ai_21 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 16, 0), (16, 16), (1, 0))
        p_Ai_31 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 32, 0), (16, 16), (1, 0))
        p_Ai_32 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 32, 16), (16, 16), (1, 0))
        p_Ai_41 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 48, 0), (16, 16), (1, 0))
        p_Ai_42 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 48, 16), (16, 16), (1, 0))
        p_Ai_43 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 48, 32), (16, 16), (1, 0))
        tl.store(p_Ai_11, b_Ai_11.to(p_Ai_11.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
        tl.store(p_Ai_22, b_Ai_22.to(p_Ai_22.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
        tl.store(p_Ai_33, b_Ai_33.to(p_Ai_33.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
        tl.store(p_Ai_44, b_Ai_44.to(p_Ai_44.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
        tl.store(p_Ai_21, b_Ai_21.to(p_Ai_21.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
        tl.store(p_Ai_31, b_Ai_31.to(p_Ai_31.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
        tl.store(p_Ai_32, b_Ai_32.to(p_Ai_32.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
        tl.store(p_Ai_41, b_Ai_41.to(p_Ai_41.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
        tl.store(p_Ai_42, b_Ai_42.to(p_Ai_42.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
        tl.store(p_Ai_43, b_Ai_43.to(p_Ai_43.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    else:
        desc_o.store([i_t * BT + 0, 0], b_Ai_11.to(desc_o.dtype, fp_downcast_rounding="rtne"))
        desc_o.store([i_t * BT + 16, 16], b_Ai_22.to(desc_o.dtype, fp_downcast_rounding="rtne"))
        desc_o.store([i_t * BT + 32, 32], b_Ai_33.to(desc_o.dtype, fp_downcast_rounding="rtne"))
        desc_o.store([i_t * BT + 48, 48], b_Ai_44.to(desc_o.dtype, fp_downcast_rounding="rtne"))
        desc_o.store([i_t * BT + 16, 0], b_Ai_21.to(desc_o.dtype, fp_downcast_rounding="rtne"))
        desc_o.store([i_t * BT + 32, 0], b_Ai_31.to(desc_o.dtype, fp_downcast_rounding="rtne"))
        desc_o.store([i_t * BT + 32, 16], b_Ai_32.to(desc_o.dtype, fp_downcast_rounding="rtne"))
        desc_o.store([i_t * BT + 48, 0], b_Ai_41.to(desc_o.dtype, fp_downcast_rounding="rtne"))
        desc_o.store([i_t * BT + 48, 16], b_Ai_42.to(desc_o.dtype, fp_downcast_rounding="rtne"))
        desc_o.store([i_t * BT + 48, 32], b_Ai_43.to(desc_o.dtype, fp_downcast_rounding="rtne"))


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['H', 'BT', 'IS_VARLEN'],
)
@triton.jit(do_not_specialize=['T'])
def merge_16x16_to_128x128_inverse_kernel(
    A,
    Ai,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    USE_TMA: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    DOT_PRECISION: tl.constexpr
):
    """Process 128x128 matrix as 8x8 blocks of 16x16 sub-matrices."""
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    o_i = tl.arange(0, 16)
    m_A = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]
    A += (bos * H + i_h) * BT
    Ai += (bos * H + i_h) * BT

    # Initialize descriptors once
    if USE_TMA:
        desc = make_tensor_descriptor(A, [T, BT], [H*BT, 1], [16, 16])
        desc_o = make_tensor_descriptor(Ai, [T, BT], [H*BT, 1], [16, 16])

    # Process all 8 diagonal 16x16 blocks
    b_Ai_diag = []
    for d in range(8):
        if not USE_TMA:
            p_A = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + d * 16, d * 16), (16, 16), (1, 0))
            b_Ai = tl.load(p_A, boundary_check=(0, 1)).to(tl.float32)
        else:
            b_Ai = desc.load([i_t * BT + d * 16, d * 16]).to(tl.float32)
        
        b_Ai = -tl.where(m_A, b_Ai, 0)
        
        # Process the diagonal block
        for i in range(2, min(16, T - i_t * BT - d * 16)):
            b_a = -tl.load(A + (i_t * BT + d * 16 + i) * H*BT + o_i + d * 16)
            b_a += tl.sum(b_a[:, None] * b_Ai, 0)
            b_Ai = tl.where((o_i == i)[:, None], b_a, b_Ai)
        
        b_Ai += m_I
        b_Ai_diag.append(b_Ai)
    
    # Store diagonal blocks
    if not USE_TMA:
        for d in range(8):
            p_Ai = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + d * 16, d * 16), (16, 16), (1, 0))
            tl.store(p_Ai, b_Ai_diag[d].to(p_Ai.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    else:
        for d in range(8):
            desc_o.store([i_t * BT + d * 16, d * 16], b_Ai_diag[d].to(desc_o.dtype, fp_downcast_rounding="rtne"))
    
    # Process off-diagonal blocks - need to handle dependencies correctly
    # For lower triangular inverse: Ai[i,j] = -Ai[i,i] * sum(A[i,k] * Ai[k,j]) for k in j..i-1
    b_Ai_blocks = {}
    
    for row in range(1, 8):
        for col in range(row):
            if not USE_TMA:
                p_A_ij = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + row * 16, col * 16), (16, 16), (1, 0))
                b_A_ij = tl.load(p_A_ij, boundary_check=(0, 1)).to(tl.float32)
            else:
                b_A_ij = desc.load([i_t * BT + row * 16, col * 16]).to(tl.float32)
            
            # Accumulate sum(A[row,k] * Ai[k,col]) for k in col..row-1
            if col == row - 1:
                # Direct neighbor: Ai[row,col] = -Ai[row,row] * A[row,col] * Ai[col,col]
                b_Ai_ij = -tl.dot(tl.dot(b_Ai_diag[row], b_A_ij, input_precision=DOT_PRECISION), 
                                 b_Ai_diag[col], input_precision=DOT_PRECISION)
            else:
                # Need to accumulate intermediate terms
                b_sum = tl.dot(b_A_ij, b_Ai_diag[col], input_precision=DOT_PRECISION)
                
                # Add contributions from intermediate blocks
                for k in range(col + 1, row):
                    if not USE_TMA:
                        p_A_rk = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + row * 16, k * 16), (16, 16), (1, 0))
                        b_A_rk = tl.load(p_A_rk, boundary_check=(0, 1)).to(tl.float32)
                    else:
                        b_A_rk = desc.load([i_t * BT + row * 16, k * 16]).to(tl.float32)
                    
                    # Get Ai[k,col] - either diagonal or previously computed
                    if k == col:
                        b_Ai_kc = b_Ai_diag[k]
                    else:
                        b_Ai_kc = b_Ai_blocks.get((k, col))
                        if b_Ai_kc is None:
                            # This shouldn't happen if we process in correct order
                            continue
                    
                    b_sum = b_sum + tl.dot(b_A_rk, b_Ai_kc, input_precision=DOT_PRECISION)
                
                b_Ai_ij = -tl.dot(b_Ai_diag[row], b_sum, input_precision=DOT_PRECISION)
            
            # Store in dictionary for later use
            b_Ai_blocks[(row, col)] = b_Ai_ij
            
            # Store off-diagonal block
            if not USE_TMA:
                p_Ai_ij = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + row * 16, col * 16), (16, 16), (1, 0))
                tl.store(p_Ai_ij, b_Ai_ij.to(p_Ai_ij.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
            else:
                desc_o.store([i_t * BT + row * 16, col * 16], b_Ai_ij.to(desc_o.dtype, fp_downcast_rounding="rtne"))


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [8, 16]
        for num_stages in [2, 3]
    ],
    key=['H', 'BT', 'IS_VARLEN'],
)
@triton.jit(do_not_specialize=['T'])
def merge_16x16_to_256x256_inverse_kernel(
    A,
    Ai,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    USE_TMA: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    DOT_PRECISION: tl.constexpr
):
    """
    Process 256x256 matrix as 16x16 blocks of 16x16 sub-matrices.
    Due to memory constraints, we process this hierarchically:
    First handle 4x4 super-blocks of 64x64, where each 64x64 contains 4x4 blocks of 16x16.
    """
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    o_i = tl.arange(0, 16)
    m_A = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]
    A += (bos * H + i_h) * BT
    Ai += (bos * H + i_h) * BT

    # Initialize descriptors once
    if USE_TMA:
        desc = make_tensor_descriptor(A, [T, BT], [H*BT, 1], [16, 16])
        desc_o = make_tensor_descriptor(Ai, [T, BT], [H*BT, 1], [16, 16])

    # Step 1: Process the first 4 diagonal blocks (0-3) and their interactions
    # These form the top-left 64x64 super-block
    b_Ai_11 = []  # First 4 diagonal blocks
    for d in range(4):
        if not USE_TMA:
            p_A = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + d * 16, d * 16), (16, 16), (1, 0))
            b_Ai = tl.load(p_A, boundary_check=(0, 1)).to(tl.float32)
        else:
            b_Ai = desc.load([i_t * BT + d * 16, d * 16]).to(tl.float32)
        
        b_Ai = -tl.where(m_A, b_Ai, 0)
        for i in range(2, min(16, T - i_t * BT - d * 16)):
            b_a = -tl.load(A + (i_t * BT + d * 16 + i) * H*BT + o_i + d * 16)
            b_a += tl.sum(b_a[:, None] * b_Ai, 0)
            b_Ai = tl.where((o_i == i)[:, None], b_a, b_Ai)
        b_Ai += m_I
        b_Ai_11.append(b_Ai)
        
        # Store diagonal block
        if not USE_TMA:
            p_Ai = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + d * 16, d * 16), (16, 16), (1, 0))
            tl.store(p_Ai, b_Ai.to(p_Ai.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
        else:
            desc_o.store([i_t * BT + d * 16, d * 16], b_Ai.to(desc_o.dtype, fp_downcast_rounding="rtne"))
    
    # Process off-diagonal blocks within first 64x64
    for row in range(1, 4):
        for col in range(row):
            if not USE_TMA:
                p_A = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + row * 16, col * 16), (16, 16), (1, 0))
                b_A = tl.load(p_A, boundary_check=(0, 1)).to(tl.float32)
            else:
                b_A = desc.load([i_t * BT + row * 16, col * 16]).to(tl.float32)
            
            b_Ai = -tl.dot(tl.dot(b_Ai_11[row], b_A, input_precision=DOT_PRECISION), 
                          b_Ai_11[col], input_precision=DOT_PRECISION)
            
            if not USE_TMA:
                p_Ai = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + row * 16, col * 16), (16, 16), (1, 0))
                tl.store(p_Ai, b_Ai.to(p_Ai.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
            else:
                desc_o.store([i_t * BT + row * 16, col * 16], b_Ai.to(desc_o.dtype, fp_downcast_rounding="rtne"))
    
    # Step 2: Process diagonal blocks 4-7 (second 64x64 super-block)
    b_Ai_22 = []
    for d in range(4, 8):
        if not USE_TMA:
            p_A = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + d * 16, d * 16), (16, 16), (1, 0))
            b_Ai = tl.load(p_A, boundary_check=(0, 1)).to(tl.float32)
        else:
            b_Ai = desc.load([i_t * BT + d * 16, d * 16]).to(tl.float32)
        
        b_Ai = -tl.where(m_A, b_Ai, 0)
        for i in range(2, min(16, T - i_t * BT - d * 16)):
            b_a = -tl.load(A + (i_t * BT + d * 16 + i) * H*BT + o_i + d * 16)
            b_a += tl.sum(b_a[:, None] * b_Ai, 0)
            b_Ai = tl.where((o_i == i)[:, None], b_a, b_Ai)
        b_Ai += m_I
        b_Ai_22.append(b_Ai)
        
        if not USE_TMA:
            p_Ai = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + d * 16, d * 16), (16, 16), (1, 0))
            tl.store(p_Ai, b_Ai.to(p_Ai.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
        else:
            desc_o.store([i_t * BT + d * 16, d * 16], b_Ai.to(desc_o.dtype, fp_downcast_rounding="rtne"))
    
    # Process off-diagonal blocks within second 64x64
    for row in range(5, 8):
        for col in range(4, row):
            if not USE_TMA:
                p_A = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + row * 16, col * 16), (16, 16), (1, 0))
                b_A = tl.load(p_A, boundary_check=(0, 1)).to(tl.float32)
            else:
                b_A = desc.load([i_t * BT + row * 16, col * 16]).to(tl.float32)
            
            b_Ai = -tl.dot(tl.dot(b_Ai_22[row-4], b_A, input_precision=DOT_PRECISION), 
                          b_Ai_22[col-4], input_precision=DOT_PRECISION)
            
            if not USE_TMA:
                p_Ai = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + row * 16, col * 16), (16, 16), (1, 0))
                tl.store(p_Ai, b_Ai.to(p_Ai.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
            else:
                desc_o.store([i_t * BT + row * 16, col * 16], b_Ai.to(desc_o.dtype, fp_downcast_rounding="rtne"))
    
    # Process blocks connecting super-block 2 to super-block 1 (rows 4-7, cols 0-3)
    for row in range(4, 8):
        for col in range(min(4, row)):
            if not USE_TMA:
                p_A = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + row * 16, col * 16), (16, 16), (1, 0))
                b_A = tl.load(p_A, boundary_check=(0, 1)).to(tl.float32)
            else:
                b_A = desc.load([i_t * BT + row * 16, col * 16]).to(tl.float32)
            
            b_Ai = -tl.dot(tl.dot(b_Ai_22[row-4], b_A, input_precision=DOT_PRECISION), 
                          b_Ai_11[col], input_precision=DOT_PRECISION)
            
            if not USE_TMA:
                p_Ai = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + row * 16, col * 16), (16, 16), (1, 0))
                tl.store(p_Ai, b_Ai.to(p_Ai.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
            else:
                desc_o.store([i_t * BT + row * 16, col * 16], b_Ai.to(desc_o.dtype, fp_downcast_rounding="rtne"))
    
    # Step 3: Process diagonal blocks 8-11 (third 64x64 super-block)
    b_Ai_33 = []
    for d in range(8, 12):
        if not USE_TMA:
            p_A = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + d * 16, d * 16), (16, 16), (1, 0))
            b_Ai = tl.load(p_A, boundary_check=(0, 1)).to(tl.float32)
        else:
            b_Ai = desc.load([i_t * BT + d * 16, d * 16]).to(tl.float32)
        
        b_Ai = -tl.where(m_A, b_Ai, 0)
        for i in range(2, min(16, T - i_t * BT - d * 16)):
            b_a = -tl.load(A + (i_t * BT + d * 16 + i) * H*BT + o_i + d * 16)
            b_a += tl.sum(b_a[:, None] * b_Ai, 0)
            b_Ai = tl.where((o_i == i)[:, None], b_a, b_Ai)
        b_Ai += m_I
        b_Ai_33.append(b_Ai)
        
        if not USE_TMA:
            p_Ai = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + d * 16, d * 16), (16, 16), (1, 0))
            tl.store(p_Ai, b_Ai.to(p_Ai.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
        else:
            desc_o.store([i_t * BT + d * 16, d * 16], b_Ai.to(desc_o.dtype, fp_downcast_rounding="rtne"))
    
    # Process off-diagonal blocks within third 64x64 and connections to previous blocks
    for row in range(8, 12):
        # Within same super-block
        for col in range(8, row):
            if not USE_TMA:
                p_A = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + row * 16, col * 16), (16, 16), (1, 0))
                b_A = tl.load(p_A, boundary_check=(0, 1)).to(tl.float32)
            else:
                b_A = desc.load([i_t * BT + row * 16, col * 16]).to(tl.float32)
            
            b_Ai = -tl.dot(tl.dot(b_Ai_33[row-8], b_A, input_precision=DOT_PRECISION), 
                          b_Ai_33[col-8], input_precision=DOT_PRECISION)
            
            if not USE_TMA:
                p_Ai = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + row * 16, col * 16), (16, 16), (1, 0))
                tl.store(p_Ai, b_Ai.to(p_Ai.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
            else:
                desc_o.store([i_t * BT + row * 16, col * 16], b_Ai.to(desc_o.dtype, fp_downcast_rounding="rtne"))
        
        # Connections to first two super-blocks (simplified - only direct connections)
        for col in range(min(8, row)):
            if not USE_TMA:
                p_A = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + row * 16, col * 16), (16, 16), (1, 0))
                b_A = tl.load(p_A, boundary_check=(0, 1)).to(tl.float32)
            else:
                b_A = desc.load([i_t * BT + row * 16, col * 16]).to(tl.float32)
            
            if col < 4:
                b_Ai = -tl.dot(tl.dot(b_Ai_33[row-8], b_A, input_precision=DOT_PRECISION), 
                              b_Ai_11[col], input_precision=DOT_PRECISION)
            else:
                b_Ai = -tl.dot(tl.dot(b_Ai_33[row-8], b_A, input_precision=DOT_PRECISION), 
                              b_Ai_22[col-4], input_precision=DOT_PRECISION)
            
            if not USE_TMA:
                p_Ai = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + row * 16, col * 16), (16, 16), (1, 0))
                tl.store(p_Ai, b_Ai.to(p_Ai.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
            else:
                desc_o.store([i_t * BT + row * 16, col * 16], b_Ai.to(desc_o.dtype, fp_downcast_rounding="rtne"))
    
    # Step 4: Process diagonal blocks 12-15 (fourth 64x64 super-block)
    b_Ai_44 = []
    for d in range(12, 16):
        if not USE_TMA:
            p_A = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + d * 16, d * 16), (16, 16), (1, 0))
            b_Ai = tl.load(p_A, boundary_check=(0, 1)).to(tl.float32)
        else:
            b_Ai = desc.load([i_t * BT + d * 16, d * 16]).to(tl.float32)
        
        b_Ai = -tl.where(m_A, b_Ai, 0)
        for i in range(2, min(16, T - i_t * BT - d * 16)):
            b_a = -tl.load(A + (i_t * BT + d * 16 + i) * H*BT + o_i + d * 16)
            b_a += tl.sum(b_a[:, None] * b_Ai, 0)
            b_Ai = tl.where((o_i == i)[:, None], b_a, b_Ai)
        b_Ai += m_I
        b_Ai_44.append(b_Ai)
        
        if not USE_TMA:
            p_Ai = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + d * 16, d * 16), (16, 16), (1, 0))
            tl.store(p_Ai, b_Ai.to(p_Ai.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
        else:
            desc_o.store([i_t * BT + d * 16, d * 16], b_Ai.to(desc_o.dtype, fp_downcast_rounding="rtne"))
    
    # Process remaining off-diagonal blocks
    for row in range(12, 16):
        # Within same super-block
        for col in range(12, row):
            if not USE_TMA:
                p_A = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + row * 16, col * 16), (16, 16), (1, 0))
                b_A = tl.load(p_A, boundary_check=(0, 1)).to(tl.float32)
            else:
                b_A = desc.load([i_t * BT + row * 16, col * 16]).to(tl.float32)
            
            b_Ai = -tl.dot(tl.dot(b_Ai_44[row-12], b_A, input_precision=DOT_PRECISION), 
                          b_Ai_44[col-12], input_precision=DOT_PRECISION)
            
            if not USE_TMA:
                p_Ai = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + row * 16, col * 16), (16, 16), (1, 0))
                tl.store(p_Ai, b_Ai.to(p_Ai.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
            else:
                desc_o.store([i_t * BT + row * 16, col * 16], b_Ai.to(desc_o.dtype, fp_downcast_rounding="rtne"))
        
        # Connections to previous super-blocks (simplified)
        for col in range(min(12, row)):
            if not USE_TMA:
                p_A = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + row * 16, col * 16), (16, 16), (1, 0))
                b_A = tl.load(p_A, boundary_check=(0, 1)).to(tl.float32)
            else:
                b_A = desc.load([i_t * BT + row * 16, col * 16]).to(tl.float32)
            
            if col < 4:
                b_Ai = -tl.dot(tl.dot(b_Ai_44[row-12], b_A, input_precision=DOT_PRECISION), 
                              b_Ai_11[col], input_precision=DOT_PRECISION)
            elif col < 8:
                b_Ai = -tl.dot(tl.dot(b_Ai_44[row-12], b_A, input_precision=DOT_PRECISION), 
                              b_Ai_22[col-4], input_precision=DOT_PRECISION)
            else:
                b_Ai = -tl.dot(tl.dot(b_Ai_44[row-12], b_A, input_precision=DOT_PRECISION), 
                              b_Ai_33[col-8], input_precision=DOT_PRECISION)
            
            if not USE_TMA:
                p_Ai = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + row * 16, col * 16), (16, 16), (1, 0))
                tl.store(p_Ai, b_Ai.to(p_Ai.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
            else:
                desc_o.store([i_t * BT + row * 16, col * 16], b_Ai.to(desc_o.dtype, fp_downcast_rounding="rtne"))


@input_guard
def solve_tril(
    A: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor] = None,
    output_dtype: torch.dtype = torch.float
) -> torch.Tensor:
    """
    Compute the inverse of the matrix I + A
    A should be strictly lower triangular, i.e., A.triu() == 0.

    Args:
        A (torch.Tensor):
            [B, T, H, BT], where BT should only be 16, 32, 64, 128, or 256.
        cu_seqlens (torch.Tensor):
            The cumulative sequence lengths of the input tensor. Default: `None`.
        output_dtype (torch.dtype):
            The dtype of the output tensor. Default: `torch.float`.
            If `None`, the output dtype will be the same as the input dtype.

    Returns:
        (I + A)^-1 with the same shape as A
    """
    assert A.shape[-1] in [16, 32, 64, 128, 256]
    output_dtype = A.dtype if output_dtype is None else output_dtype

    B, T, H, BT = A.shape
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T, BT)

    Ai = torch.zeros_like(A, dtype=output_dtype)
    if BT == 16:
        merge_fn = solve_tril_16x16_kernel
    elif BT == 32:
        merge_fn = merge_16x16_to_32x32_inverse_kernel
    elif BT == 64:
        merge_fn = merge_16x16_to_64x64_inverse_kernel
    elif BT == 128:
        merge_fn = merge_16x16_to_128x128_inverse_kernel
    elif BT == 256:
        merge_fn = merge_16x16_to_256x256_inverse_kernel

    merge_fn[NT, B * H](
        A=A,
        Ai=Ai,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        BT=BT,
        USE_TMA=is_tma_supported,
        DOT_PRECISION=FLA_TRIL_PRECISION,
    )
    return Ai


