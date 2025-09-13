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


# @triton.heuristics({
#     'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
# })
# @triton.autotune(
#     configs=[
#         triton.Config({}, num_warps=num_warps, num_stages=num_stages)
#         for num_warps in [4, 8]
#         for num_stages in [2, 3, 4]
#     ],
#     key=['H', 'BT', 'IS_VARLEN'],
# )
# @triton.jit(do_not_specialize=['T'])
# def merge_16x16_to_128x128_inverse_kernel(
#     A,
#     Ai,
#     cu_seqlens,
#     chunk_indices,
#     T,
#     H: tl.constexpr,
#     BT: tl.constexpr,
#     USE_TMA: tl.constexpr,
#     IS_VARLEN: tl.constexpr,
#     DOT_PRECISION: tl.constexpr
# ):
#     i_t, i_bh = tl.program_id(0), tl.program_id(1)
#     i_b, i_h = i_bh // H, i_bh % H
#     if IS_VARLEN:
#         i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
#         bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
#         T = eos - bos
#     else:
#         bos, eos = i_b * T, i_b * T + T

#     o_i = tl.arange(0, 16)
#     m_A = o_i[:, None] > o_i[None, :]
#     m_I = o_i[:, None] == o_i[None, :]
#     A += (bos * H + i_h) * BT
#     Ai += (bos * H + i_h) * BT

#     if not USE_TMA:
#         # Load and process all 8 diagonal blocks
#         p_A_11 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT, 0), (16, 16), (1, 0))
#         p_A_22 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 16, 16), (16, 16), (1, 0))
#         p_A_33 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 32, 32), (16, 16), (1, 0))
#         p_A_44 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 48, 48), (16, 16), (1, 0))
#         p_A_55 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 64, 64), (16, 16), (1, 0))
#         p_A_66 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 80, 80), (16, 16), (1, 0))
#         p_A_77 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 96, 96), (16, 16), (1, 0))
#         p_A_88 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 112, 112), (16, 16), (1, 0))
        
#         b_Ai_11 = tl.load(p_A_11, boundary_check=(0, 1)).to(tl.float32)
#         b_Ai_22 = tl.load(p_A_22, boundary_check=(0, 1)).to(tl.float32)
#         b_Ai_33 = tl.load(p_A_33, boundary_check=(0, 1)).to(tl.float32)
#         b_Ai_44 = tl.load(p_A_44, boundary_check=(0, 1)).to(tl.float32)
#         b_Ai_55 = tl.load(p_A_55, boundary_check=(0, 1)).to(tl.float32)
#         b_Ai_66 = tl.load(p_A_66, boundary_check=(0, 1)).to(tl.float32)
#         b_Ai_77 = tl.load(p_A_77, boundary_check=(0, 1)).to(tl.float32)
#         b_Ai_88 = tl.load(p_A_88, boundary_check=(0, 1)).to(tl.float32)
#     else:
#         desc = make_tensor_descriptor(A, [T, BT], [H*BT, 1], [16, 16])
#         desc_o = make_tensor_descriptor(Ai, [T, BT], [H*BT, 1], [16, 16])
#         b_Ai_11 = desc.load([i_t * BT + 0, 0]).to(tl.float32)
#         b_Ai_22 = desc.load([i_t * BT + 16, 16]).to(tl.float32)
#         b_Ai_33 = desc.load([i_t * BT + 32, 32]).to(tl.float32)
#         b_Ai_44 = desc.load([i_t * BT + 48, 48]).to(tl.float32)
#         b_Ai_55 = desc.load([i_t * BT + 64, 64]).to(tl.float32)
#         b_Ai_66 = desc.load([i_t * BT + 80, 80]).to(tl.float32)
#         b_Ai_77 = desc.load([i_t * BT + 96, 96]).to(tl.float32)
#         b_Ai_88 = desc.load([i_t * BT + 112, 112]).to(tl.float32)

#     # Process diagonal blocks
#     b_Ai_11 = -tl.where(m_A, b_Ai_11, 0)
#     b_Ai_22 = -tl.where(m_A, b_Ai_22, 0)
#     b_Ai_33 = -tl.where(m_A, b_Ai_33, 0)
#     b_Ai_44 = -tl.where(m_A, b_Ai_44, 0)
#     b_Ai_55 = -tl.where(m_A, b_Ai_55, 0)
#     b_Ai_66 = -tl.where(m_A, b_Ai_66, 0)
#     b_Ai_77 = -tl.where(m_A, b_Ai_77, 0)
#     b_Ai_88 = -tl.where(m_A, b_Ai_88, 0)

#     for i in range(2, min(16, T - i_t * BT)):
#         b_a_11 = -tl.load(A + (i_t * BT + i) * H*BT + o_i)
#         b_a_11 += tl.sum(b_a_11[:, None] * b_Ai_11, 0)
#         b_Ai_11 = tl.where((o_i == i)[:, None], b_a_11, b_Ai_11)
#     for i in range(16 + 2, min(32, T - i_t * BT)):
#         b_a_22 = -tl.load(A + (i_t * BT + i) * H*BT + o_i + 16)
#         b_a_22 += tl.sum(b_a_22[:, None] * b_Ai_22, 0)
#         b_Ai_22 = tl.where((o_i == i - 16)[:, None], b_a_22, b_Ai_22)
#     for i in range(32 + 2, min(48, T - i_t * BT)):
#         b_a_33 = -tl.load(A + (i_t * BT + i) * H*BT + o_i + 32)
#         b_a_33 += tl.sum(b_a_33[:, None] * b_Ai_33, 0)
#         b_Ai_33 = tl.where((o_i == i - 32)[:, None], b_a_33, b_Ai_33)
#     for i in range(48 + 2, min(64, T - i_t * BT)):
#         b_a_44 = -tl.load(A + (i_t * BT + i) * H*BT + o_i + 48)
#         b_a_44 += tl.sum(b_a_44[:, None] * b_Ai_44, 0)
#         b_Ai_44 = tl.where((o_i == i - 48)[:, None], b_a_44, b_Ai_44)
#     for i in range(64 + 2, min(80, T - i_t * BT)):
#         b_a_55 = -tl.load(A + (i_t * BT + i) * H*BT + o_i + 64)
#         b_a_55 += tl.sum(b_a_55[:, None] * b_Ai_55, 0)
#         b_Ai_55 = tl.where((o_i == i - 64)[:, None], b_a_55, b_Ai_55)
#     for i in range(80 + 2, min(96, T - i_t * BT)):
#         b_a_66 = -tl.load(A + (i_t * BT + i) * H*BT + o_i + 80)
#         b_a_66 += tl.sum(b_a_66[:, None] * b_Ai_66, 0)
#         b_Ai_66 = tl.where((o_i == i - 80)[:, None], b_a_66, b_Ai_66)
#     for i in range(96 + 2, min(112, T - i_t * BT)):
#         b_a_77 = -tl.load(A + (i_t * BT + i) * H*BT + o_i + 96)
#         b_a_77 += tl.sum(b_a_77[:, None] * b_Ai_77, 0)
#         b_Ai_77 = tl.where((o_i == i - 96)[:, None], b_a_77, b_Ai_77)
#     for i in range(112 + 2, min(128, T - i_t * BT)):
#         b_a_88 = -tl.load(A + (i_t * BT + i) * H*BT + o_i + 112)
#         b_a_88 += tl.sum(b_a_88[:, None] * b_Ai_88, 0)
#         b_Ai_88 = tl.where((o_i == i - 112)[:, None], b_a_88, b_Ai_88)

#     b_Ai_11 += m_I
#     b_Ai_22 += m_I
#     b_Ai_33 += m_I
#     b_Ai_44 += m_I
#     b_Ai_55 += m_I
#     b_Ai_66 += m_I
#     b_Ai_77 += m_I
#     b_Ai_88 += m_I

#     # Load off-diagonal blocks
#     if not USE_TMA:
#         # Row 2
#         p_A_21 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 16, 0), (16, 16), (1, 0))
#         b_A_21 = tl.load(p_A_21, boundary_check=(0, 1)).to(tl.float32)
#         # Row 3
#         p_A_31 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 32, 0), (16, 16), (1, 0))
#         p_A_32 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 32, 16), (16, 16), (1, 0))
#         b_A_31 = tl.load(p_A_31, boundary_check=(0, 1)).to(tl.float32)
#         b_A_32 = tl.load(p_A_32, boundary_check=(0, 1)).to(tl.float32)
#         # Row 4
#         p_A_41 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 48, 0), (16, 16), (1, 0))
#         p_A_42 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 48, 16), (16, 16), (1, 0))
#         p_A_43 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 48, 32), (16, 16), (1, 0))
#         b_A_41 = tl.load(p_A_41, boundary_check=(0, 1)).to(tl.float32)
#         b_A_42 = tl.load(p_A_42, boundary_check=(0, 1)).to(tl.float32)
#         b_A_43 = tl.load(p_A_43, boundary_check=(0, 1)).to(tl.float32)
#         # Row 5
#         p_A_51 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 64, 0), (16, 16), (1, 0))
#         p_A_52 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 64, 16), (16, 16), (1, 0))
#         p_A_53 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 64, 32), (16, 16), (1, 0))
#         p_A_54 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 64, 48), (16, 16), (1, 0))
#         b_A_51 = tl.load(p_A_51, boundary_check=(0, 1)).to(tl.float32)
#         b_A_52 = tl.load(p_A_52, boundary_check=(0, 1)).to(tl.float32)
#         b_A_53 = tl.load(p_A_53, boundary_check=(0, 1)).to(tl.float32)
#         b_A_54 = tl.load(p_A_54, boundary_check=(0, 1)).to(tl.float32)
#         # Row 6
#         p_A_61 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 80, 0), (16, 16), (1, 0))
#         p_A_62 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 80, 16), (16, 16), (1, 0))
#         p_A_63 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 80, 32), (16, 16), (1, 0))
#         p_A_64 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 80, 48), (16, 16), (1, 0))
#         p_A_65 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 80, 64), (16, 16), (1, 0))
#         b_A_61 = tl.load(p_A_61, boundary_check=(0, 1)).to(tl.float32)
#         b_A_62 = tl.load(p_A_62, boundary_check=(0, 1)).to(tl.float32)
#         b_A_63 = tl.load(p_A_63, boundary_check=(0, 1)).to(tl.float32)
#         b_A_64 = tl.load(p_A_64, boundary_check=(0, 1)).to(tl.float32)
#         b_A_65 = tl.load(p_A_65, boundary_check=(0, 1)).to(tl.float32)
#         # Row 7
#         p_A_71 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 96, 0), (16, 16), (1, 0))
#         p_A_72 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 96, 16), (16, 16), (1, 0))
#         p_A_73 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 96, 32), (16, 16), (1, 0))
#         p_A_74 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 96, 48), (16, 16), (1, 0))
#         p_A_75 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 96, 64), (16, 16), (1, 0))
#         p_A_76 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 96, 80), (16, 16), (1, 0))
#         b_A_71 = tl.load(p_A_71, boundary_check=(0, 1)).to(tl.float32)
#         b_A_72 = tl.load(p_A_72, boundary_check=(0, 1)).to(tl.float32)
#         b_A_73 = tl.load(p_A_73, boundary_check=(0, 1)).to(tl.float32)
#         b_A_74 = tl.load(p_A_74, boundary_check=(0, 1)).to(tl.float32)
#         b_A_75 = tl.load(p_A_75, boundary_check=(0, 1)).to(tl.float32)
#         b_A_76 = tl.load(p_A_76, boundary_check=(0, 1)).to(tl.float32)
#         # Row 8
#         p_A_81 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 112, 0), (16, 16), (1, 0))
#         p_A_82 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 112, 16), (16, 16), (1, 0))
#         p_A_83 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 112, 32), (16, 16), (1, 0))
#         p_A_84 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 112, 48), (16, 16), (1, 0))
#         p_A_85 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 112, 64), (16, 16), (1, 0))
#         p_A_86 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 112, 80), (16, 16), (1, 0))
#         p_A_87 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 112, 96), (16, 16), (1, 0))
#         b_A_81 = tl.load(p_A_81, boundary_check=(0, 1)).to(tl.float32)
#         b_A_82 = tl.load(p_A_82, boundary_check=(0, 1)).to(tl.float32)
#         b_A_83 = tl.load(p_A_83, boundary_check=(0, 1)).to(tl.float32)
#         b_A_84 = tl.load(p_A_84, boundary_check=(0, 1)).to(tl.float32)
#         b_A_85 = tl.load(p_A_85, boundary_check=(0, 1)).to(tl.float32)
#         b_A_86 = tl.load(p_A_86, boundary_check=(0, 1)).to(tl.float32)
#         b_A_87 = tl.load(p_A_87, boundary_check=(0, 1)).to(tl.float32)
#     else:
#         # Row 2
#         b_A_21 = desc.load([i_t * BT + 16, 0]).to(tl.float32)
#         # Row 3
#         b_A_31 = desc.load([i_t * BT + 32, 0]).to(tl.float32)
#         b_A_32 = desc.load([i_t * BT + 32, 16]).to(tl.float32)
#         # Row 4
#         b_A_41 = desc.load([i_t * BT + 48, 0]).to(tl.float32)
#         b_A_42 = desc.load([i_t * BT + 48, 16]).to(tl.float32)
#         b_A_43 = desc.load([i_t * BT + 48, 32]).to(tl.float32)
#         # Row 5
#         b_A_51 = desc.load([i_t * BT + 64, 0]).to(tl.float32)
#         b_A_52 = desc.load([i_t * BT + 64, 16]).to(tl.float32)
#         b_A_53 = desc.load([i_t * BT + 64, 32]).to(tl.float32)
#         b_A_54 = desc.load([i_t * BT + 64, 48]).to(tl.float32)
#         # Row 6
#         b_A_61 = desc.load([i_t * BT + 80, 0]).to(tl.float32)
#         b_A_62 = desc.load([i_t * BT + 80, 16]).to(tl.float32)
#         b_A_63 = desc.load([i_t * BT + 80, 32]).to(tl.float32)
#         b_A_64 = desc.load([i_t * BT + 80, 48]).to(tl.float32)
#         b_A_65 = desc.load([i_t * BT + 80, 64]).to(tl.float32)
#         # Row 7
#         b_A_71 = desc.load([i_t * BT + 96, 0]).to(tl.float32)
#         b_A_72 = desc.load([i_t * BT + 96, 16]).to(tl.float32)
#         b_A_73 = desc.load([i_t * BT + 96, 32]).to(tl.float32)
#         b_A_74 = desc.load([i_t * BT + 96, 48]).to(tl.float32)
#         b_A_75 = desc.load([i_t * BT + 96, 64]).to(tl.float32)
#         b_A_76 = desc.load([i_t * BT + 96, 80]).to(tl.float32)
#         # Row 8
#         b_A_81 = desc.load([i_t * BT + 112, 0]).to(tl.float32)
#         b_A_82 = desc.load([i_t * BT + 112, 16]).to(tl.float32)
#         b_A_83 = desc.load([i_t * BT + 112, 32]).to(tl.float32)
#         b_A_84 = desc.load([i_t * BT + 112, 48]).to(tl.float32)
#         b_A_85 = desc.load([i_t * BT + 112, 64]).to(tl.float32)
#         b_A_86 = desc.load([i_t * BT + 112, 80]).to(tl.float32)
#         b_A_87 = desc.load([i_t * BT + 112, 96]).to(tl.float32)

#     # Compute off-diagonal blocks using block triangular inverse formula
#     # Row 2
#     b_Ai_21 = -tl.dot(tl.dot(b_Ai_22, b_A_21, input_precision=DOT_PRECISION), b_Ai_11, input_precision=DOT_PRECISION)
#     # Row 3
#     b_Ai_32 = -tl.dot(tl.dot(b_Ai_33, b_A_32, input_precision=DOT_PRECISION), b_Ai_22, input_precision=DOT_PRECISION)
#     b_Ai_31 = -tl.dot(b_Ai_33, tl.dot(b_A_31, b_Ai_11, input_precision=DOT_PRECISION) + tl.dot(b_A_32, b_Ai_21, input_precision=DOT_PRECISION), input_precision=DOT_PRECISION)
#     # Row 4
#     b_Ai_43 = -tl.dot(tl.dot(b_Ai_44, b_A_43, input_precision=DOT_PRECISION), b_Ai_33, input_precision=DOT_PRECISION)
#     b_Ai_42 = -tl.dot(b_Ai_44, tl.dot(b_A_42, b_Ai_22, input_precision=DOT_PRECISION) + tl.dot(b_A_43, b_Ai_32, input_precision=DOT_PRECISION), input_precision=DOT_PRECISION)
#     b_Ai_41 = -tl.dot(b_Ai_44, tl.dot(b_A_41, b_Ai_11, input_precision=DOT_PRECISION) + tl.dot(b_A_42, b_Ai_21, input_precision=DOT_PRECISION) + tl.dot(b_A_43, b_Ai_31, input_precision=DOT_PRECISION), input_precision=DOT_PRECISION)
#     # Row 5
#     b_Ai_54 = -tl.dot(tl.dot(b_Ai_55, b_A_54, input_precision=DOT_PRECISION), b_Ai_44, input_precision=DOT_PRECISION)
#     b_Ai_53 = -tl.dot(b_Ai_55, tl.dot(b_A_53, b_Ai_33, input_precision=DOT_PRECISION) + tl.dot(b_A_54, b_Ai_43, input_precision=DOT_PRECISION), input_precision=DOT_PRECISION)
#     b_Ai_52 = -tl.dot(b_Ai_55, tl.dot(b_A_52, b_Ai_22, input_precision=DOT_PRECISION) + tl.dot(b_A_53, b_Ai_32, input_precision=DOT_PRECISION) + tl.dot(b_A_54, b_Ai_42, input_precision=DOT_PRECISION), input_precision=DOT_PRECISION)
#     b_Ai_51 = -tl.dot(b_Ai_55, tl.dot(b_A_51, b_Ai_11, input_precision=DOT_PRECISION) + tl.dot(b_A_52, b_Ai_21, input_precision=DOT_PRECISION) + tl.dot(b_A_53, b_Ai_31, input_precision=DOT_PRECISION) + tl.dot(b_A_54, b_Ai_41, input_precision=DOT_PRECISION), input_precision=DOT_PRECISION)
#     # Row 6
#     b_Ai_65 = -tl.dot(tl.dot(b_Ai_66, b_A_65, input_precision=DOT_PRECISION), b_Ai_55, input_precision=DOT_PRECISION)
#     b_Ai_64 = -tl.dot(b_Ai_66, tl.dot(b_A_64, b_Ai_44, input_precision=DOT_PRECISION) + tl.dot(b_A_65, b_Ai_54, input_precision=DOT_PRECISION), input_precision=DOT_PRECISION)
#     b_Ai_63 = -tl.dot(b_Ai_66, tl.dot(b_A_63, b_Ai_33, input_precision=DOT_PRECISION) + tl.dot(b_A_64, b_Ai_43, input_precision=DOT_PRECISION) + tl.dot(b_A_65, b_Ai_53, input_precision=DOT_PRECISION), input_precision=DOT_PRECISION)
#     b_Ai_62 = -tl.dot(b_Ai_66, tl.dot(b_A_62, b_Ai_22, input_precision=DOT_PRECISION) + tl.dot(b_A_63, b_Ai_32, input_precision=DOT_PRECISION) + tl.dot(b_A_64, b_Ai_42, input_precision=DOT_PRECISION) + tl.dot(b_A_65, b_Ai_52, input_precision=DOT_PRECISION), input_precision=DOT_PRECISION)
#     b_Ai_61 = -tl.dot(b_Ai_66, tl.dot(b_A_61, b_Ai_11, input_precision=DOT_PRECISION) + tl.dot(b_A_62, b_Ai_21, input_precision=DOT_PRECISION) + tl.dot(b_A_63, b_Ai_31, input_precision=DOT_PRECISION) + tl.dot(b_A_64, b_Ai_41, input_precision=DOT_PRECISION) + tl.dot(b_A_65, b_Ai_51, input_precision=DOT_PRECISION), input_precision=DOT_PRECISION)
#     # Row 7
#     b_Ai_76 = -tl.dot(tl.dot(b_Ai_77, b_A_76, input_precision=DOT_PRECISION), b_Ai_66, input_precision=DOT_PRECISION)
#     b_Ai_75 = -tl.dot(b_Ai_77, tl.dot(b_A_75, b_Ai_55, input_precision=DOT_PRECISION) + tl.dot(b_A_76, b_Ai_65, input_precision=DOT_PRECISION), input_precision=DOT_PRECISION)
#     b_Ai_74 = -tl.dot(b_Ai_77, tl.dot(b_A_74, b_Ai_44, input_precision=DOT_PRECISION) + tl.dot(b_A_75, b_Ai_54, input_precision=DOT_PRECISION) + tl.dot(b_A_76, b_Ai_64, input_precision=DOT_PRECISION), input_precision=DOT_PRECISION)
#     b_Ai_73 = -tl.dot(b_Ai_77, tl.dot(b_A_73, b_Ai_33, input_precision=DOT_PRECISION) + tl.dot(b_A_74, b_Ai_43, input_precision=DOT_PRECISION) + tl.dot(b_A_75, b_Ai_53, input_precision=DOT_PRECISION) + tl.dot(b_A_76, b_Ai_63, input_precision=DOT_PRECISION), input_precision=DOT_PRECISION)
#     b_Ai_72 = -tl.dot(b_Ai_77, tl.dot(b_A_72, b_Ai_22, input_precision=DOT_PRECISION) + tl.dot(b_A_73, b_Ai_32, input_precision=DOT_PRECISION) + tl.dot(b_A_74, b_Ai_42, input_precision=DOT_PRECISION) + tl.dot(b_A_75, b_Ai_52, input_precision=DOT_PRECISION) + tl.dot(b_A_76, b_Ai_62, input_precision=DOT_PRECISION), input_precision=DOT_PRECISION)
#     b_Ai_71 = -tl.dot(b_Ai_77, tl.dot(b_A_71, b_Ai_11, input_precision=DOT_PRECISION) + tl.dot(b_A_72, b_Ai_21, input_precision=DOT_PRECISION) + tl.dot(b_A_73, b_Ai_31, input_precision=DOT_PRECISION) + tl.dot(b_A_74, b_Ai_41, input_precision=DOT_PRECISION) + tl.dot(b_A_75, b_Ai_51, input_precision=DOT_PRECISION) + tl.dot(b_A_76, b_Ai_61, input_precision=DOT_PRECISION), input_precision=DOT_PRECISION)
#     # Row 8
#     b_Ai_87 = -tl.dot(tl.dot(b_Ai_88, b_A_87, input_precision=DOT_PRECISION), b_Ai_77, input_precision=DOT_PRECISION)
#     b_Ai_86 = -tl.dot(b_Ai_88, tl.dot(b_A_86, b_Ai_66, input_precision=DOT_PRECISION) + tl.dot(b_A_87, b_Ai_76, input_precision=DOT_PRECISION), input_precision=DOT_PRECISION)
#     b_Ai_85 = -tl.dot(b_Ai_88, tl.dot(b_A_85, b_Ai_55, input_precision=DOT_PRECISION) + tl.dot(b_A_86, b_Ai_65, input_precision=DOT_PRECISION) + tl.dot(b_A_87, b_Ai_75, input_precision=DOT_PRECISION), input_precision=DOT_PRECISION)
#     b_Ai_84 = -tl.dot(b_Ai_88, tl.dot(b_A_84, b_Ai_44, input_precision=DOT_PRECISION) + tl.dot(b_A_85, b_Ai_54, input_precision=DOT_PRECISION) + tl.dot(b_A_86, b_Ai_64, input_precision=DOT_PRECISION) + tl.dot(b_A_87, b_Ai_74, input_precision=DOT_PRECISION), input_precision=DOT_PRECISION)
#     b_Ai_83 = -tl.dot(b_Ai_88, tl.dot(b_A_83, b_Ai_33, input_precision=DOT_PRECISION) + tl.dot(b_A_84, b_Ai_43, input_precision=DOT_PRECISION) + tl.dot(b_A_85, b_Ai_53, input_precision=DOT_PRECISION) + tl.dot(b_A_86, b_Ai_63, input_precision=DOT_PRECISION) + tl.dot(b_A_87, b_Ai_73, input_precision=DOT_PRECISION), input_precision=DOT_PRECISION)
#     b_Ai_82 = -tl.dot(b_Ai_88, tl.dot(b_A_82, b_Ai_22, input_precision=DOT_PRECISION) + tl.dot(b_A_83, b_Ai_32, input_precision=DOT_PRECISION) + tl.dot(b_A_84, b_Ai_42, input_precision=DOT_PRECISION) + tl.dot(b_A_85, b_Ai_52, input_precision=DOT_PRECISION) + tl.dot(b_A_86, b_Ai_62, input_precision=DOT_PRECISION) + tl.dot(b_A_87, b_Ai_72, input_precision=DOT_PRECISION), input_precision=DOT_PRECISION)
#     b_Ai_81 = -tl.dot(b_Ai_88, tl.dot(b_A_81, b_Ai_11, input_precision=DOT_PRECISION) + tl.dot(b_A_82, b_Ai_21, input_precision=DOT_PRECISION) + tl.dot(b_A_83, b_Ai_31, input_precision=DOT_PRECISION) + tl.dot(b_A_84, b_Ai_41, input_precision=DOT_PRECISION) + tl.dot(b_A_85, b_Ai_51, input_precision=DOT_PRECISION) + tl.dot(b_A_86, b_Ai_61, input_precision=DOT_PRECISION) + tl.dot(b_A_87, b_Ai_71, input_precision=DOT_PRECISION), input_precision=DOT_PRECISION)

#     # Store all blocks
#     if not USE_TMA:
#         # Diagonal blocks
#         p_Ai_11 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT, 0), (16, 16), (1, 0))
#         p_Ai_22 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 16, 16), (16, 16), (1, 0))
#         p_Ai_33 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 32, 32), (16, 16), (1, 0))
#         p_Ai_44 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 48, 48), (16, 16), (1, 0))
#         p_Ai_55 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 64, 64), (16, 16), (1, 0))
#         p_Ai_66 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 80, 80), (16, 16), (1, 0))
#         p_Ai_77 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 96, 96), (16, 16), (1, 0))
#         p_Ai_88 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 112, 112), (16, 16), (1, 0))
#         tl.store(p_Ai_11, b_Ai_11.to(p_Ai_11.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
#         tl.store(p_Ai_22, b_Ai_22.to(p_Ai_22.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
#         tl.store(p_Ai_33, b_Ai_33.to(p_Ai_33.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
#         tl.store(p_Ai_44, b_Ai_44.to(p_Ai_44.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
#         tl.store(p_Ai_55, b_Ai_55.to(p_Ai_55.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
#         tl.store(p_Ai_66, b_Ai_66.to(p_Ai_66.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
#         tl.store(p_Ai_77, b_Ai_77.to(p_Ai_77.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
#         tl.store(p_Ai_88, b_Ai_88.to(p_Ai_88.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
#         # Off-diagonal blocks
#         p_Ai_21 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 16, 0), (16, 16), (1, 0))
#         p_Ai_31 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 32, 0), (16, 16), (1, 0))
#         p_Ai_32 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 32, 16), (16, 16), (1, 0))
#         p_Ai_41 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 48, 0), (16, 16), (1, 0))
#         p_Ai_42 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 48, 16), (16, 16), (1, 0))
#         p_Ai_43 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 48, 32), (16, 16), (1, 0))
#         p_Ai_51 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 64, 0), (16, 16), (1, 0))
#         p_Ai_52 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 64, 16), (16, 16), (1, 0))
#         p_Ai_53 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 64, 32), (16, 16), (1, 0))
#         p_Ai_54 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 64, 48), (16, 16), (1, 0))
#         p_Ai_61 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 80, 0), (16, 16), (1, 0))
#         p_Ai_62 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 80, 16), (16, 16), (1, 0))
#         p_Ai_63 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 80, 32), (16, 16), (1, 0))
#         p_Ai_64 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 80, 48), (16, 16), (1, 0))
#         p_Ai_65 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 80, 64), (16, 16), (1, 0))
#         p_Ai_71 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 96, 0), (16, 16), (1, 0))
#         p_Ai_72 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 96, 16), (16, 16), (1, 0))
#         p_Ai_73 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 96, 32), (16, 16), (1, 0))
#         p_Ai_74 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 96, 48), (16, 16), (1, 0))
#         p_Ai_75 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 96, 64), (16, 16), (1, 0))
#         p_Ai_76 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 96, 80), (16, 16), (1, 0))
#         p_Ai_81 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 112, 0), (16, 16), (1, 0))
#         p_Ai_82 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 112, 16), (16, 16), (1, 0))
#         p_Ai_83 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 112, 32), (16, 16), (1, 0))
#         p_Ai_84 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 112, 48), (16, 16), (1, 0))
#         p_Ai_85 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 112, 64), (16, 16), (1, 0))
#         p_Ai_86 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 112, 80), (16, 16), (1, 0))
#         p_Ai_87 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 112, 96), (16, 16), (1, 0))
#         tl.store(p_Ai_21, b_Ai_21.to(p_Ai_21.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
#         tl.store(p_Ai_31, b_Ai_31.to(p_Ai_31.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
#         tl.store(p_Ai_32, b_Ai_32.to(p_Ai_32.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
#         tl.store(p_Ai_41, b_Ai_41.to(p_Ai_41.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
#         tl.store(p_Ai_42, b_Ai_42.to(p_Ai_42.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
#         tl.store(p_Ai_43, b_Ai_43.to(p_Ai_43.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
#         tl.store(p_Ai_51, b_Ai_51.to(p_Ai_51.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
#         tl.store(p_Ai_52, b_Ai_52.to(p_Ai_52.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
#         tl.store(p_Ai_53, b_Ai_53.to(p_Ai_53.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
#         tl.store(p_Ai_54, b_Ai_54.to(p_Ai_54.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
#         tl.store(p_Ai_61, b_Ai_61.to(p_Ai_61.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
#         tl.store(p_Ai_62, b_Ai_62.to(p_Ai_62.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
#         tl.store(p_Ai_63, b_Ai_63.to(p_Ai_63.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
#         tl.store(p_Ai_64, b_Ai_64.to(p_Ai_64.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
#         tl.store(p_Ai_65, b_Ai_65.to(p_Ai_65.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
#         tl.store(p_Ai_71, b_Ai_71.to(p_Ai_71.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
#         tl.store(p_Ai_72, b_Ai_72.to(p_Ai_72.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
#         tl.store(p_Ai_73, b_Ai_73.to(p_Ai_73.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
#         tl.store(p_Ai_74, b_Ai_74.to(p_Ai_74.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
#         tl.store(p_Ai_75, b_Ai_75.to(p_Ai_75.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
#         tl.store(p_Ai_76, b_Ai_76.to(p_Ai_76.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
#         tl.store(p_Ai_81, b_Ai_81.to(p_Ai_81.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
#         tl.store(p_Ai_82, b_Ai_82.to(p_Ai_82.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
#         tl.store(p_Ai_83, b_Ai_83.to(p_Ai_83.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
#         tl.store(p_Ai_84, b_Ai_84.to(p_Ai_84.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
#         tl.store(p_Ai_85, b_Ai_85.to(p_Ai_85.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
#         tl.store(p_Ai_86, b_Ai_86.to(p_Ai_86.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
#         tl.store(p_Ai_87, b_Ai_87.to(p_Ai_87.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
#     else:
#         # Diagonal blocks
#         desc_o.store([i_t * BT + 0, 0], b_Ai_11.to(desc_o.dtype, fp_downcast_rounding="rtne"))
#         desc_o.store([i_t * BT + 16, 16], b_Ai_22.to(desc_o.dtype, fp_downcast_rounding="rtne"))
#         desc_o.store([i_t * BT + 32, 32], b_Ai_33.to(desc_o.dtype, fp_downcast_rounding="rtne"))
#         desc_o.store([i_t * BT + 48, 48], b_Ai_44.to(desc_o.dtype, fp_downcast_rounding="rtne"))
#         desc_o.store([i_t * BT + 64, 64], b_Ai_55.to(desc_o.dtype, fp_downcast_rounding="rtne"))
#         desc_o.store([i_t * BT + 80, 80], b_Ai_66.to(desc_o.dtype, fp_downcast_rounding="rtne"))
#         desc_o.store([i_t * BT + 96, 96], b_Ai_77.to(desc_o.dtype, fp_downcast_rounding="rtne"))
#         desc_o.store([i_t * BT + 112, 112], b_Ai_88.to(desc_o.dtype, fp_downcast_rounding="rtne"))
#         # Off-diagonal blocks
#         desc_o.store([i_t * BT + 16, 0], b_Ai_21.to(desc_o.dtype, fp_downcast_rounding="rtne"))
#         desc_o.store([i_t * BT + 32, 0], b_Ai_31.to(desc_o.dtype, fp_downcast_rounding="rtne"))
#         desc_o.store([i_t * BT + 32, 16], b_Ai_32.to(desc_o.dtype, fp_downcast_rounding="rtne"))
#         desc_o.store([i_t * BT + 48, 0], b_Ai_41.to(desc_o.dtype, fp_downcast_rounding="rtne"))
#         desc_o.store([i_t * BT + 48, 16], b_Ai_42.to(desc_o.dtype, fp_downcast_rounding="rtne"))
#         desc_o.store([i_t * BT + 48, 32], b_Ai_43.to(desc_o.dtype, fp_downcast_rounding="rtne"))
#         desc_o.store([i_t * BT + 64, 0], b_Ai_51.to(desc_o.dtype, fp_downcast_rounding="rtne"))
#         desc_o.store([i_t * BT + 64, 16], b_Ai_52.to(desc_o.dtype, fp_downcast_rounding="rtne"))
#         desc_o.store([i_t * BT + 64, 32], b_Ai_53.to(desc_o.dtype, fp_downcast_rounding="rtne"))
#         desc_o.store([i_t * BT + 64, 48], b_Ai_54.to(desc_o.dtype, fp_downcast_rounding="rtne"))
#         desc_o.store([i_t * BT + 80, 0], b_Ai_61.to(desc_o.dtype, fp_downcast_rounding="rtne"))
#         desc_o.store([i_t * BT + 80, 16], b_Ai_62.to(desc_o.dtype, fp_downcast_rounding="rtne"))
#         desc_o.store([i_t * BT + 80, 32], b_Ai_63.to(desc_o.dtype, fp_downcast_rounding="rtne"))
#         desc_o.store([i_t * BT + 80, 48], b_Ai_64.to(desc_o.dtype, fp_downcast_rounding="rtne"))
#         desc_o.store([i_t * BT + 80, 64], b_Ai_65.to(desc_o.dtype, fp_downcast_rounding="rtne"))
#         desc_o.store([i_t * BT + 96, 0], b_Ai_71.to(desc_o.dtype, fp_downcast_rounding="rtne"))
#         desc_o.store([i_t * BT + 96, 16], b_Ai_72.to(desc_o.dtype, fp_downcast_rounding="rtne"))
#         desc_o.store([i_t * BT + 96, 32], b_Ai_73.to(desc_o.dtype, fp_downcast_rounding="rtne"))
#         desc_o.store([i_t * BT + 96, 48], b_Ai_74.to(desc_o.dtype, fp_downcast_rounding="rtne"))
#         desc_o.store([i_t * BT + 96, 64], b_Ai_75.to(desc_o.dtype, fp_downcast_rounding="rtne"))
#         desc_o.store([i_t * BT + 96, 80], b_Ai_76.to(desc_o.dtype, fp_downcast_rounding="rtne"))
#         desc_o.store([i_t * BT + 112, 0], b_Ai_81.to(desc_o.dtype, fp_downcast_rounding="rtne"))
#         desc_o.store([i_t * BT + 112, 16], b_Ai_82.to(desc_o.dtype, fp_downcast_rounding="rtne"))
#         desc_o.store([i_t * BT + 112, 32], b_Ai_83.to(desc_o.dtype, fp_downcast_rounding="rtne"))
#         desc_o.store([i_t * BT + 112, 48], b_Ai_84.to(desc_o.dtype, fp_downcast_rounding="rtne"))
#         desc_o.store([i_t * BT + 112, 64], b_Ai_85.to(desc_o.dtype, fp_downcast_rounding="rtne"))
#         desc_o.store([i_t * BT + 112, 80], b_Ai_86.to(desc_o.dtype, fp_downcast_rounding="rtne"))
#         desc_o.store([i_t * BT + 112, 96], b_Ai_87.to(desc_o.dtype, fp_downcast_rounding="rtne"))

@triton.heuristics({
    "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=nw, num_stages=ns)
        for nw in (2, 4, 8)
        for ns in (2, 3, 4, 5)
    ],
    key=["H", "BT", "IS_VARLEN", "NB", "BLK"],  # include NB & BLK in key
)
@triton.jit(do_not_specialize=["T"])
def merge_16x16_to_nxn_inverse_kernel(
    A,
    Ai,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    NB: tl.constexpr,            # number of sub-blocks per side
    BLK: tl.constexpr,           # must be 16
    USE_TMA: tl.constexpr,       # unused (manual path)
    IS_VARLEN: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    # program ids
    i_t_pid, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H

    # resolve BOS/EOS, T
    if IS_VARLEN:
        i_n  = tl.load(chunk_indices + i_t_pid * 2).to(tl.int32)
        i_t  = tl.load(chunk_indices + i_t_pid * 2 + 1).to(tl.int32)
        bos  = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos  = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T    = eos - bos
    else:
        i_t  = i_t_pid
        bos  = i_b * T
        eos  = bos + T

    # base plane advance
    A  += (bos * H + i_h) * BT
    Ai += (bos * H + i_h) * BT

    # compile-time shapes derived from BLK
    o16 = tl.arange(0, BLK)              # requires BLK constexpr
    m_strict_lower = o16[:, None] > o16[None, :]
    m_eye          = o16[:, None] == o16[None, :]

    #  invert diagonal BLK×BLK blocks =====
    row_base = i_t * BT
    for b in range(NB):
        r0 = row_base + b * BLK
        c0 = b * BLK

        p_Lbb = tl.make_block_ptr(A,  (T, BT), (H*BT, 1), (r0, c0), (BLK, BLK), (1, 0))
        Lbb   = tl.load(p_Lbb, boundary_check=(0, 1)).to(tl.float32)

        Inv_bb = -tl.where(m_strict_lower, Lbb, 0.0)
        for i_local in range(2, BLK):
            row_idx = r0 + i_local
            col_vec = c0 + o16
            row_vals = tl.load(
                A + row_idx * (H * BT) + col_vec,
                mask=(row_idx < T) & (col_vec < BT),
                other=0.0,
            ).to(tl.float32)
            # Use elementwise-mul + reduction (avoids MMA min-size constraint)
            b_row = -row_vals
            b_row = b_row + tl.sum(b_row[:, None] * Inv_bb, axis=0)
            Inv_bb = tl.where((o16 == i_local)[:, None], b_row, Inv_bb)

        Inv_bb += m_eye
        p_Invbb = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (r0, c0), (BLK, BLK), (1, 0))
        tl.store(
            p_Invbb,
            Inv_bb.to(p_Invbb.dtype.element_ty, fp_downcast_rounding="rtne"),
            boundary_check=(0, 1)
        )

    # off-diagonals =====
    for i_blk in range(1, NB):
        r0_i = row_base + i_blk * BLK
        p_Ai_ii = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (r0_i, i_blk * BLK), (BLK, BLK), (1, 0))
        Ai_ii   = tl.load(p_Ai_ii, boundary_check=(0, 1)).to(tl.float32)

        for j_blk in range(0, i_blk):
            c0_j = j_blk * BLK
            S = tl.zeros((BLK, BLK), dtype=tl.float32)      # requires BLK constexpr
            for k_blk in range(j_blk, i_blk):
                p_L_ik = tl.make_block_ptr(A,  (T, BT), (H*BT, 1), (r0_i, k_blk * BLK), (BLK, BLK), (1, 0))
                L_ik   = tl.load(p_L_ik, boundary_check=(0, 1)).to(tl.float32)
                r0_k   = row_base + k_blk * BLK
                p_Ai_kj = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (r0_k, c0_j), (BLK, BLK), (1, 0))
                Ai_kj   = tl.load(p_Ai_kj, boundary_check=(0, 1)).to(tl.float32)
                S += tl.dot(L_ik, Ai_kj, input_precision=DOT_PRECISION)

            Ai_ij = -tl.dot(Ai_ii, S, input_precision=DOT_PRECISION)
            p_Ai_ij = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (r0_i, c0_j), (BLK, BLK), (1, 0))
            tl.store(
                p_Ai_ij,
                Ai_ij.to(p_Ai_ij.dtype.element_ty, fp_downcast_rounding="rtne"),
                boundary_check=(0, 1)
            )

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
    assert A.shape[-1] in [16, 32, 64, 128, 256, 512]
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
    elif BT >= 128: 
        merge_fn = merge_16x16_to_nxn_inverse_kernel

    if BT < 128:
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
    else:    
        NB = BT // 16
        merge_fn[NT, B * H](
            A=A,
            Ai=Ai,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            T=T,
            H=H,
            BT=BT,
            NB=NB,
            BLK=16,
            USE_TMA=is_tma_supported,
            DOT_PRECISION=FLA_TRIL_PRECISION,
        )
    return Ai




