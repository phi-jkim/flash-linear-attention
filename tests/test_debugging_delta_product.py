# -*- coding: utf-8 -*-

## Test expanded chunk size of 64

# expanded chunk size (chunk_gated_delta_product_bwd) of 64 with 16 and num householder 4

# same operations of chunk_scaled_dot_kkt_fwd (expanded chunk size of 64)

# solve_tril (expanded chunk size of 64)

# recompute_w_u_expanded (expanded chunk size of 64)

# chunk_gated_delta_product_fwd_h_expanded (chunk size of 16)

# chunk_gated_delta_product_bwd_dv_local (chunk size of 16)

# chunk_gated_delta_product_bwd_dhu (chunk size of 16)

# chunk_gated_delta_product_bwd_dqkwg (chunk size of 16)


# recompute_w_u_expanded

# chunk gated delta rule bwd with chunk size 64

# recompute_w_u_fwd (64) -> w, u (Test 1 compare with chunk_gated_delta_product_bwd)

# chunk_gated_delta_rule_fwd_h (64) -> h, v_new  (Test 2 compare with chunk_gated_delta_rule_fwd_h)

# chunk_gated_delta_product_bwd_dv_local (chunk size of 64) -> dv_new (Test 3 compare with chunk_gated_delta_product_bwd_dv_local)

# chunk_gated_delta_product_bwd_dhu (chunk size of 64) (Test 4 compare with chunk_gated_delta_product_bwd_dhu)

# chunk_gated_delta_product_bwd_dqkwg (chunk size of 64) -> dq, dk, dw, dg (Test 5 compare with chunk_gated_delta_product_bwd_dqkwg)

# prepare_wy_repr_bwd_expanded (chunk size of 64) -> dk_hidden_state_gradient, dv, dbeta, dg2 (Test 6 compare with prepare_wy_repr_bwd_expanded)


# repeat test for expanded chunk size of 32 with 16 and num householder 2

# repeat test for expanded chunk size of 32 with 32 and num householder 1


# add token length of more than 200 ~ 300 (various lengths like other tests, make sure that the tensor dimensions are correct) 
# -*- coding: utf-8 -*-
"""Tests for expanded chunk sizes in gated delta product backward helpers."""

from __future__ import annotations

import pytest
import torch
from einops import rearrange

from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu
from fla.ops.common.chunk_o import (
    chunk_bwd_dqkwg as delta_rule_chunk_bwd_dqkwg,
    chunk_bwd_dv_local as delta_rule_chunk_bwd_dv_local,
)
from fla.ops.common.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from fla.ops.gated_delta_product.chunk import chunk_gated_delta_product_fwd
from fla.ops.gated_delta_product.chunk_deltaproduct_h import (
    chunk_gated_delta_product_bwd_dhu,
    chunk_gated_delta_product_fwd_h,
)
from fla.ops.gated_delta_product.chunk_deltaproduct_o import (
    chunk_bwd_dqkwg as delta_product_chunk_bwd_dqkwg,
    chunk_bwd_dv_local as delta_product_chunk_bwd_dv_local,
)
from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_fwd_h
from fla.ops.gated_delta_rule.wy_fast import prepare_wy_repr_bwd, recompute_w_u_fwd
from fla.ops.utils import chunk_local_cumsum, solve_tril
from fla.utils import assert_close


@pytest.mark.parametrize(
    "base_chunk_size, num_householder",
    [
        (16, 4),
        (16, 2),
        (32, 1),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Expanded chunk tests require CUDA support")
def test_gated_delta_product_expanded_chunk(base_chunk_size: int, num_householder: int) -> None:
    torch.manual_seed(0)

    B = 1
    H = 2
    D = 8
    T_base = base_chunk_size
    T_total = T_base * num_householder
    scale = 0.75

    dtype = torch.float16
    torch_device = torch.device("cuda")

    q = torch.randn(B, T_base, H, D, dtype=dtype, device=torch_device)
    k = torch.randn(B, T_total, H, D, dtype=dtype, device=torch_device)
    k = torch.nn.functional.normalize(k.to(torch.float32), dim=-1).to(dtype)
    v = torch.randn(B, T_total, H, D, dtype=dtype, device=torch_device)
    beta = torch.rand(B, T_total, H, dtype=dtype, device=torch_device).sigmoid()
    g_base = torch.randn(B, T_base, H, dtype=dtype, device=torch_device)
    initial_state = torch.zeros(B, H, D, D, dtype=torch.float32, device=torch_device)
    do = torch.randn(B, T_base, H, D, dtype=dtype, device=torch_device)
    dht = torch.randn(B, H, D, D, dtype=torch.float32, device=torch_device)

    _, g_interleaved_cumsum, _, A_forward, _ = chunk_gated_delta_product_fwd(
        q=q,
        k=k,
        v=v,
        g=g_base,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=True,
        num_householder=num_householder,
    )

    expanded_chunk_size = base_chunk_size * (1 << (num_householder - 1).bit_length())

    q_expanded = q.new_zeros(B, T_base, num_householder, H, D)
    q_expanded[:, :, -1] = q
    q_expanded = rearrange(q_expanded, "b t n h d -> b (t n) h d").contiguous()

    do_expanded = do.new_zeros(B, T_base, num_householder, H, D)
    do_expanded[:, :, -1] = do
    do_expanded = rearrange(do_expanded, "b t n h d -> b (t n) h d").contiguous()

    A = chunk_scaled_dot_kkt_fwd(
        k=k,
        beta=beta,
        g_cumsum=g_interleaved_cumsum,
        cu_seqlens=None,
        output_dtype=torch.float32,
    )
    A = solve_tril(A=A, cu_seqlens=None, output_dtype=k.dtype)
    assert_close("A", A_forward, A, atol=1e-3)

    w_gdp, u_gdp = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        g_cumsum=g_interleaved_cumsum,
        cu_seqlens=None,
    )

    h_gdp, v_new_gdp, _ = chunk_gated_delta_product_fwd_h(
        k=k,
        w=w_gdp,
        u=u_gdp,
        g=g_interleaved_cumsum,
        initial_state=initial_state,
        output_final_state=True,
        chunk_size=base_chunk_size,
        cu_seqlens=None,
        num_householder=num_householder,
    )

    dv_local_gdp = delta_product_chunk_bwd_dv_local(
        q=q_expanded,
        k=k,
        g=g_interleaved_cumsum,
        do=do_expanded,
        scale=scale,
        cu_seqlens=None,
        chunk_size=base_chunk_size,
    )

    dh_gdp, dh0_gdp, dv_after_dhu_gdp = chunk_gated_delta_product_bwd_dhu(
        q=q_expanded,
        k=k,
        w=w_gdp,
        g=g_interleaved_cumsum,
        h0=initial_state,
        dht=dht,
        do=do_expanded,
        dv=dv_local_gdp,
        scale=scale,
        cu_seqlens=None,
        chunk_size=base_chunk_size,
    )

    dq_gdp, dk_gdp, dw_gdp, dg_partial_gdp = delta_product_chunk_bwd_dqkwg(
        q=q_expanded,
        k=k,
        v=v_new_gdp,
        g=g_interleaved_cumsum,
        do=do_expanded,
        h=h_gdp,
        dh=dh_gdp,
        dv=dv_after_dhu_gdp,
        w=w_gdp,
        cu_seqlens=None,
        chunk_size=base_chunk_size,
        scale=scale,
    )

    dk_hidden_gdp, dv_prepare_gdp, db_gdp, dg_prepare_gdp = prepare_wy_repr_bwd(
        k=k,
        v=v,
        beta=beta,
        g=g_interleaved_cumsum,
        A=A,
        dw=dw_gdp,
        du=dv_after_dhu_gdp,
        cu_seqlens=None,
    )

    dk_total_gdp = dk_gdp + dk_hidden_gdp
    dg_total_gdp = dg_partial_gdp + dg_prepare_gdp
    dg_total_gdp = chunk_local_cumsum(
        g=dg_total_gdp,
        chunk_size=64,
        reverse=True,
        cu_seqlens=None,
    )

    w_rule, u_rule = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        g_cumsum=g_interleaved_cumsum,
        cu_seqlens=None,
    )

    h_rule, v_new_rule, _ = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w_rule,
        u=u_rule,
        g=g_interleaved_cumsum,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=None,
        chunk_size=expanded_chunk_size,
    )

    dv_local_rule = delta_rule_chunk_bwd_dv_local(
        q=q_expanded,
        k=k,
        g=g_interleaved_cumsum,
        do=do_expanded,
        scale=scale,
        cu_seqlens=None,
        chunk_size=expanded_chunk_size,
    )

    dh_rule, dh0_rule, dv_after_dhu_rule = chunk_gated_delta_rule_bwd_dhu(
        q=q_expanded,
        k=k,
        w=w_rule,
        g=g_interleaved_cumsum,
        h0=initial_state,
        dht=dht,
        do=do_expanded,
        dv=dv_local_rule,
        scale=scale,
        cu_seqlens=None,
        chunk_size=expanded_chunk_size,
    )

    dq_rule, dk_rule, dw_rule, dg_partial_rule = delta_rule_chunk_bwd_dqkwg(
        q=q_expanded,
        k=k,
        v=v_new_rule,
        g=g_interleaved_cumsum,
        do=do_expanded,
        h=h_rule,
        dh=dh_rule,
        dv=dv_after_dhu_rule,
        w=w_rule,
        cu_seqlens=None,
        chunk_size=expanded_chunk_size,
        scale=scale,
    )

    dk_hidden_rule, dv_prepare_rule, db_rule, dg_prepare_rule = prepare_wy_repr_bwd(
        k=k,
        v=v,
        beta=beta,
        g=g_interleaved_cumsum,
        A=A,
        dw=dw_rule,
        du=dv_after_dhu_rule,
        cu_seqlens=None,
    )

    dk_total_rule = dk_rule + dk_hidden_rule
    dg_total_rule = dg_partial_rule + dg_prepare_rule
    dg_total_rule = chunk_local_cumsum(
        g=dg_total_rule,
        chunk_size=64,
        reverse=True,
        cu_seqlens=None,
    )

    assert_close("w", w_gdp, w_rule, atol=1e-3)
    assert_close("u", u_gdp, u_rule, atol=1e-3)
    assert_close("h", h_gdp, h_rule, atol=2e-3)
    assert_close("v_new", v_new_gdp, v_new_rule, atol=2e-3)
    assert_close("dv_local", dv_local_gdp, dv_local_rule, atol=2e-3)
    assert_close("dh", dh_gdp, dh_rule, atol=2e-3)
    assert_close("dh0", dh0_gdp, dh0_rule, atol=2e-3)
    assert_close("dv_after_dhu", dv_after_dhu_gdp, dv_after_dhu_rule, atol=2e-3)
    assert_close("dq", dq_gdp, dq_rule, atol=2e-3)
    assert_close("dk_total", dk_total_gdp, dk_total_rule, atol=2e-3)
    assert_close("dw", dw_gdp, dw_rule, atol=2e-3)
    assert_close("dg_total", dg_total_gdp, dg_total_rule, atol=2e-3)
    assert_close("dv_prepare", dv_prepare_gdp, dv_prepare_rule, atol=2e-3)
    assert_close("db", db_gdp, db_rule, atol=2e-3)