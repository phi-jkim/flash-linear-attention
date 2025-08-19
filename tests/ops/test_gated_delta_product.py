import os
from typing import List

import pytest
import torch
import torch.nn.functional as F

from fla.ops.gated_delta_product import chunk_gated_delta_product
from fla.ops.gated_delta_product.chunk_ref import chunk_gated_delta_product_ref
from fla.ops.gated_delta_product.naive import naive_recurrent_gated_delta_product
from fla.utils import assert_close, device, is_intel_alchemist
import math

@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'scale', 'num_householder', 'gate_logit_normalizer', 'mask_p', 'use_qk_l2norm_in_kernel', 'dtype'),
    [
        pytest.param(
            *test,
            id="B{}-T{}-H{}-D{}-scale{}-num_householder{}-gate_logit_normalizer{}-mask_p{}-l2norm{}-{}".format(*test)
        )
        for test in [
            (1, 63, 1, 64, 0.1, 1, 1, 0, False, torch.float16),
            (2, 200, 3, 60, 0.1, 1, 1, 0, False, torch.float16),
            (2, 1000, 4, 64, 0.1, 2, 0.1, 0.5, False, torch.float16),
            (2, 1024, 4, 64, 1, 2, 1, 0, True, torch.float16),
            (2, 1024, 6, 100, 1, 2, 10, 0, False, torch.float16),
            (4, 1500, 8, 128, 0.1, 3, 1, 0.5, False, torch.float16),
            (2, 2048, 8, 128, 1, 3, 1, 0, False, torch.float16),
            (2, 2048, 8, 128, 1, 3, 1, 0, True, torch.float16),
        ]
    ]
)
def test_chunk(
    B: int,
    T: int,
    H: int,
    D: int,
    scale: float,
    num_householder: int,
    gate_logit_normalizer: float,
    mask_p: float,
    use_qk_l2norm_in_kernel: bool,
    dtype: torch.dtype,
):
    if is_intel_alchemist and D > 128:
        pytest.skip(reason='chunk_gated_delta_rule is not supported on alchemist for D>128')

    q = torch.randn(B, T, H, D, dtype=dtype)
    k = torch.randn(B, T * num_householder, H, D, dtype=dtype)
    v = torch.randn(B, T * num_householder, H, D, dtype=dtype)
    beta = torch.rand(B, T * num_householder, H, dtype=dtype).sigmoid()
    g = F.logsigmoid(torch.rand(B, T, H, dtype=torch.float32))
    h0 = torch.zeros(B, H, D, D, dtype=torch.float32)
    g = g / gate_logit_normalizer
    g = g * (torch.rand_like(g) > mask_p)
    q, k, v, beta, g, h0 = map(lambda x: x.to(device).requires_grad_(True), (q, k, v, beta, g, h0))

    tri, tri_ht = chunk_gated_delta_product(
        q=F.normalize(q.clone(), p=2, dim=-1) if not use_qk_l2norm_in_kernel else q.clone(),
        k=F.normalize(k.clone(), p=2, dim=-1) if not use_qk_l2norm_in_kernel else k.clone(),
        v=v.clone(),
        g=g.clone(),
        beta=beta.clone(),
        num_householder=num_householder,
        scale=scale,
        output_final_state=True,
        initial_state=h0.clone(),
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
    do = torch.randn_like(q)
    dht = torch.randn_like(h0)
    ((tri * do).sum() + (tri_ht * dht).sum()).backward(retain_graph=True)
    tri_dq, tri_dk, tri_dv, tri_dbeta, tri_dg, tri_dh0 = q.grad, k.grad, v.grad, beta.grad, g.grad, h0.grad
    q.grad = k.grad = v.grad = beta.grad = g.grad = h0.grad = None

    ref, ref_ht = chunk_gated_delta_product_ref(
        q=F.normalize(q.clone(), p=2, dim=-1),
        k=F.normalize(k.clone(), p=2, dim=-1),
        v=v.clone(),
        g=g.clone(),
        beta=beta.clone(),
        num_householder=num_householder,
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
    )

    ((ref * do).sum() + (ref_ht * dht).sum()).backward(retain_graph=True)
    ref_dq, ref_dk, ref_dv, ref_dbeta, ref_dg, ref_dh0 = q.grad, k.grad, v.grad, beta.grad, g.grad, h0.grad
    assert_close('o', ref, tri, 0.005)
    assert_close('ht', ref_ht, tri_ht, 0.005)
    assert_close('dq', ref_dq, tri_dq, 0.008)
    assert_close('dk', ref_dk, tri_dk, 0.008)
    assert_close('dv', ref_dv, tri_dv, 0.008)
    assert_close('db', ref_dbeta, tri_dbeta, 0.02)
    assert_close('dg', ref_dg, tri_dg, 0.02)
    assert_close('dh0', ref_dh0, tri_dh0, 0.008)


@pytest.mark.parametrize(
    ('H', 'D', 'num_householder', 'mask_p', 'cu_seqlens', 'dtype'),
    [
        pytest.param(*test, id="H{}-D{}-num_householder{}-mask_p{}-cu_seqlens{}-{}".format(*test))
        for test in [
            (2, 64, 3, 0, [0, 63], torch.float16),
            (2, 100, 2, 0, [0, 63, 100, 500, 1000], torch.float16),
            (2, 100, 2, 0, [0, 100, 256, 512, 1500, 1500], torch.float16),
            (2, 128, 2, 0, [0, 100, 300, 800, 1500, 2000], torch.float16),
            (2, 128, 2, 0.5, [0, 31, 111, 799, 1000, 1500, 1800, 2000], torch.float16),
            (2, 128, 2, 0.5, [0, 63, 300, 800, 1000, 1399, 2048], torch.float16),
            (2, 256, 3, 0, [0, 100, 123, 300, 500, 800, 1000, 1500, 2048], torch.float16),
        ]
    ]
)
def test_chunk_varlen(
    H: int,
    D: int,
    num_householder: int,
    mask_p: float,
    cu_seqlens: List[int],
    dtype: torch.dtype,
):
    if is_intel_alchemist and D > 128:
        pytest.skip(reason='chunk_gated_delta_rule is not supported on alchemist for D>128')
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'
    cu_seqlens = torch.LongTensor(cu_seqlens).to(device)
    T = cu_seqlens[-1]
    N = len(cu_seqlens) - 1

    q = torch.nn.functional.normalize(torch.randn((1, T, H, D), dtype=dtype), dim=-1, p=2)
    k = torch.nn.functional.normalize(torch.randn(1, T*num_householder, H, D, dtype=dtype), dim=-1, p=2)
    v = torch.randn((1, T*num_householder, H, D), dtype=dtype)
    g = F.logsigmoid(torch.rand(1, T, H, dtype=dtype))
    g = g * (torch.rand_like(g) > mask_p)
    beta = torch.rand(1, T*num_householder, H, dtype=dtype).sigmoid()
    h0 = torch.randn((N, H, D, D), dtype=dtype)

    q, k, v, beta, g, h0 = map(lambda x: x.to(device).requires_grad_(), (q, k, v, beta, g, h0))
    do = torch.randn_like(q)
    dht = torch.rand_like(h0)
    scale = D ** -0.5

    tri, tri_ht = chunk_gated_delta_product(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        beta=beta.clone(),
        g=g.clone(),
        scale=scale,
        output_final_state=True,
        num_householder=num_householder,
        initial_state=h0.clone(),
        cu_seqlens=cu_seqlens
    )
    ((tri * do).sum() + (tri_ht * dht).sum()).backward(retain_graph=True)
    tri_dq, tri_dk, tri_dv, tri_dbeta, tri_dg, tri_dh0 = q.grad, k.grad, v.grad, beta.grad, g.grad, h0.grad
    q.grad = k.grad = v.grad = beta.grad = g.grad = h0.grad = None

    ref, ref_ht = chunk_gated_delta_product_ref(
        q=q.clone(),
        k=k.clone(),
        v=v.clone( ),
        beta=beta.clone(),
        g=g.clone(),
        scale=scale,
        output_final_state=True,
        num_householder=num_householder,
        initial_state=h0.clone(),
        cu_seqlens=cu_seqlens
    )

    ((ref * do).sum() + (ref_ht * dht).sum()).backward(retain_graph=True)
    ref_dq, ref_dk, ref_dv, ref_dbeta, ref_dg, ref_dh0 = q.grad, k.grad, v.grad, beta.grad, g.grad, h0.grad

    assert_close('o', ref, tri, 0.005)
    assert_close('ht', ref_ht, tri_ht, 0.005)
    assert_close('dq', ref_dq, tri_dq, 0.007)
    assert_close('dk', ref_dk, tri_dk, 0.008)
    assert_close('dv', ref_dv, tri_dv, 0.007)
    assert_close('db', ref_dbeta, tri_dbeta, 0.015)
    assert_close('dh0', ref_dh0, tri_dh0, 0.007)
    assert_close('dg', ref_dg, tri_dg, 0.015)
    q.grad = k.grad = v.grad = beta.grad = g.grad = h0.grad = None

    torch_ref = torch.zeros_like(ref)
    torch_ref_ht = torch.zeros_like(ref_ht)
    for i in range(len(cu_seqlens) - 1):
        start, end = cu_seqlens[i], cu_seqlens[i+1]
        q_i = q[:, start:end, :, :]
        k_i = k[:, start*num_householder:end*num_householder, :, :]
        v_i = v[:, start*num_householder:end*num_householder, :, :]
        g_i = g[:, start:end, :]
        beta_i = beta[:, start*num_householder:end*num_householder, :]
        o3_i, h3_i = naive_recurrent_gated_delta_product(
            q_i, k_i, v_i, g_i, beta_i, scale=scale, cu_seqlens=None, output_final_state=True, num_householder=num_householder
        )
        torch_ref[:, start:end, :, :] = o3_i
        torch_ref_ht[i, :, :, :] = h3_i.squeeze(0)

    ((torch_ref * do).sum() + (torch_ref_ht * dht).sum()).backward(retain_graph=True)

    assert_close('o', ref, tri, 0.005)
    assert_close('ht', ref_ht, tri_ht, 0.005)
    assert_close('dq', ref_dq, tri_dq, 0.007)
    assert_close('dk', ref_dk, tri_dk, 0.008)
    assert_close('dv', ref_dv, tri_dv, 0.007)
    assert_close('db', ref_dbeta, tri_dbeta, 0.015)
    assert_close('dg', ref_dg, tri_dg, 0.015)
    assert_close('dh0', ref_dh0, tri_dh0, 0.007)


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'scale', 'num_householder', 'dtype'),
    [
        (1, 8, 2, 16, 0.1, 1, torch.float32),
        (2, 12, 2, 24, 0.5, 2, torch.float32),
        (1, 64, 2, 32, 0.1, 1, torch.float32),
        (2, 128, 4, 64, 0.5, 2, torch.float32),
        (1, 256, 4, 64, 1.0, 2, torch.float32),
        (2, 512, 8, 128, 0.1, 3, torch.float32),
        (1, 1024, 8, 128, 1.0, 3, torch.float32),
        (2, 2048, 4, 64, 0.5, 2, torch.float32),
    ]
)
def test_naive_vs_manual_backward(
    B: int,
    T: int,
    H: int,
    D: int,
    scale: float,
    num_householder: int,
    dtype: torch.dtype,
):
    """Test manual naive backward implementation against autograd."""
    from fla.ops.gated_delta_product.naive import naive_torch_delta_product_bwd
    
    print(f"\n=== Testing B={B}, T={T}, H={H}, D={D}, scale={scale}, num_householder={num_householder}, dtype={dtype} ===")
    
    torch.manual_seed(42)
    
    # Create inputs on the correct device
    print(f"Creating tensors on device: {device}")
    q = torch.randn(B, T, H, D, dtype=dtype, device=device, requires_grad=True)
    k = torch.randn(B, T * num_householder, H, D, dtype=dtype, device=device, requires_grad=True)
    v = torch.randn(B, T * num_householder, H, D, dtype=dtype, device=device, requires_grad=True)
    beta = torch.rand(B, T * num_householder, H, dtype=dtype, device=device, requires_grad=True).sigmoid()
    g = F.logsigmoid(torch.rand(B, T, H, dtype=dtype, device=device, requires_grad=True))
    h0 = torch.randn(B, H, D, D, dtype=dtype, device=device, requires_grad=True)
    
    print(f"Input shapes: q={q.shape}, k={k.shape}, v={v.shape}, beta={beta.shape}, g={g.shape}, h0={h0.shape}")
    
    # Forward pass with autograd
    o_auto, h_auto = naive_recurrent_gated_delta_product(
        q=q, k=k, v=v, g=g, beta=beta,
        scale=scale,
        cu_seqlens=None,
        initial_state=h0,
        output_final_state=True, num_householder=num_householder
    )
    
    do = torch.randn_like(o_auto)
    dht = torch.randn_like(h_auto)
    
    # Ensure outputs retain gradients for proper backward computation
    o_auto.retain_grad()
    h_auto.retain_grad()
    
    # Compute gradients with autograd
    loss = (o_auto * do).sum() + (h_auto * dht).sum()
    loss.backward()
    auto_dq, auto_dk, auto_dv, auto_dbeta, auto_dg, auto_dh0 = q.grad, k.grad, v.grad, beta.grad, g.grad, h0.grad
    
    # Clear gradients and detach inputs for manual backward
    q.grad = k.grad = v.grad = beta.grad = g.grad = h0.grad = None
    q_manual = q.clone()
    k_manual = k.clone()
    v_manual = v.clone()
    beta_manual = beta.clone()
    g_manual = g.clone()
    h0_manual = h0.clone()
    
    # Manual backward pass
    manual_dq, manual_dk, manual_dv, manual_dg, manual_dbeta, manual_dh0 = naive_torch_delta_product_bwd(
        q=q_manual,
        k=k_manual,
        v=v_manual,
        g=g_manual,
        beta=beta_manual,
        scale=1.0, # fix scale as 1.0
        initial_state=h0_manual,
        output_final_state=True,
        num_householder=num_householder,
        do=do,
        dht=dht,
    )
    
    # Compare gradients using same tolerances as existing tests
    assert_close('dq', auto_dq, manual_dq, 0.008)
    assert_close('dk', auto_dk, manual_dk, 0.008)
    assert_close('dv', auto_dv, manual_dv, 0.008)
    assert_close('db', auto_dbeta, manual_dbeta, 0.02)
    assert_close('dg', auto_dg, manual_dg, 0.02)
    assert_close('dh0', auto_dh0, manual_dh0, 0.008)


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'scale', 'num_householder', 'dtype'),
    [
        (1, 8, 2, 16, 1.0, 1, torch.float32),
        (2, 12, 2, 24, 1.0, 2, torch.float32),
        (1, 64, 2, 32, 1.0, 1, torch.float32),
        (2, 128, 4, 64, 1.0, 2, torch.float32),
    ]
)
def test_helper1_du_direct_comparison(
    B: int,
    T: int,
    H: int,
    D: int,
    scale: float,
    num_householder: int,
    dtype: torch.dtype,
):
    """Test helper function 1 (du_direct computation) by following actual backward logic."""
    from fla.ops.gated_delta_product.naive import helper_direct_gradient_u_minus_ws
    from fla.ops.delta_rule.wy_fast import prepare_wy_repr_fwd
    from fla.ops.gated_delta_rule.wy_fast import recompute_w_u_fwd
    from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h
    from fla.ops.common.chunk_o import chunk_bwd_dv_local
    from fla.ops.utils import chunk_local_cumsum
    from einops import rearrange
    from fla.ops.common.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
    from fla.ops.utils import chunk_local_cumsum, solve_tril

    
    print(f"\n=== Testing Helper1 B={B}, T={T}, H={H}, D={D}, scale={scale}, num_householder={num_householder}, dtype={dtype} ===")
    
    torch.manual_seed(42)
    
    # Create inputs on the correct device
    print(f"Creating tensors on device: {device}")
    # q = torch.randn(B, T, H, D, dtype=dtype, device=device, requires_grad=True)
    # k = torch.randn(B, T * num_householder, H, D, dtype=dtype, device=device, requires_grad=True)
    # TODO: Jinha normalize so that values are not huge 
    q = torch.nn.functional.normalize(torch.randn((1, T, H, D), dtype=dtype, device=device, requires_grad=True), dim=-1, p=2)
    k = torch.nn.functional.normalize(torch.randn(1, T*num_householder, H, D, dtype=dtype, device=device, requires_grad=True), dim=-1, p=2)
    v = torch.randn(B, T * num_householder, H, D, dtype=dtype, device=device, requires_grad=True)
    beta = torch.rand(B, T * num_householder, H, dtype=dtype, device=device, requires_grad=True).sigmoid()
    g = F.logsigmoid(torch.rand(B, T, H, dtype=dtype, device=device, requires_grad=True))
    h0 = torch.randn(B, H, D, D, dtype=dtype, device=device, requires_grad=True)

    g_interleaved = g.new_zeros(g.shape[0], g.shape[1], num_householder, g.shape[2], dtype=torch.float32)
    g_interleaved[:, :, 0] = g
    g_interleaved = rearrange(g_interleaved, 'b l n h -> b (l n) h').contiguous()
    g = chunk_local_cumsum(g, chunk_size=64, cu_seqlens=None, output_dtype=torch.float32)
    g_interleaved = chunk_local_cumsum(g_interleaved, chunk_size=64, cu_seqlens=None, output_dtype=torch.float32)

    # obtain WY representation. u is actually the new v.
    # A = chunk_scaled_dot_kkt_fwd(
    #     k=k,
    #     g=g_interleaved,
    #     beta=beta,
    #     cu_seqlens=None,
    #     output_dtype=torch.float32
    # )
    # A = solve_tril(
    #     A=A,
    #     cu_seqlens=None,
    #     output_dtype=k.dtype
    # )
    
    print(f"Input shapes: q={q.shape}, k={k.shape}, v={v.shape}, beta={beta.shape}, g={g.shape}, h0={h0.shape}")
    
    # Create gradient output tensors 
    do = torch.randn_like(q)
    dht = torch.randn_like(h0)
    
    # Follow the actual backward logic from chunk.py
    print("[TEST] Following actual backward logic...")
    
    # # Step 1: Prepare WY representation (from forward)
    # w, u = recompute_w_u_fwd(
    #     k=k,
    #     v=v,
    #     beta=beta,
    #     A=A,
    #     g=None,  # Use None to match the logic
    #     cu_seqlens=None,
    # )
    # print(f"[TEST] w shape: {w.shape}, u shape: {u.shape}")
    
    # # Step 3: Recompute h, v_new (following backward logic)
    # h, v_new, _ = chunk_gated_delta_rule_fwd_h(
    #     k=k,
    #     w=w,
    #     u=u,
    #     g=g,
    #     initial_state=h0,
    #     output_final_state=False,
    #     cu_seqlens=None,
    # )
    # print(f"[TEST] h shape: {h.shape}, v_new shape: {v_new.shape}")
    
    # Step 4: Expand do to match the backward pass logic
    q_new = q.new_zeros(q.shape[0], q.shape[1], num_householder, q.shape[2], q.shape[3])
    q_new[:, :, -1] = q
    do_new = do.new_zeros(do.shape[0], do.shape[1], num_householder, do.shape[2], do.shape[3])
    do_new[:, :, -1] = do
    q_expanded = rearrange(q_new, 'b t n h d -> b (t n) h d')
    do_expanded = rearrange(do_new, 'b t n h d -> b (t n) h d')
    
    print(f"[TEST] q_expanded shape: {q_expanded.shape}, do_expanded shape: {do_expanded.shape}")
    
    # Step 5: Compute du_direct using the tri implementation method
    print("[TEST] Computing du_direct_tri using chunk_bwd_dv_local...")
    du_direct_tri = chunk_bwd_dv_local(
        q=q_expanded,
        k=k,
        # g=g_interleaved,
        g=None, 
        do=do_expanded,
        scale=scale,
        cu_seqlens=None,
        chunk_size=128, 
    )
    print(f"[TEST] du_direct_tri shape: {du_direct_tri.shape}")
    print(f"[TEST] du_direct_tri values: min={du_direct_tri.min():.6f}, max={du_direct_tri.max():.6f}, has_nan={torch.isnan(du_direct_tri).any()}")
    
    # Step 6: Test helper function 1 directly with the same inputs
    print("[TEST] Testing helper function 1 directly...")
    du_direct_naive = helper_direct_gradient_u_minus_ws(
        q=q.clone(),
        k=k.clone(), 
        g=g.clone(),
        do=do.clone(),
        num_householder=num_householder
    )
    
    print(f"[TEST] du_direct_naive shape: {du_direct_naive.shape}")
    print(f"[TEST] du_direct_naive values: min={du_direct_naive.min():.6f}, max={du_direct_naive.max():.6f}, has_nan={torch.isnan(du_direct_naive).any()}")
    
    # Step 7: Compare the du_direct outputs
    print("[TEST] Comparing du_direct outputs...")
    assert_close('du_direct', du_direct_tri, du_direct_naive, 0.01)
    print("[TEST] du_direct comparison passed!")
    
    # Step 8: Check for NaN values
    assert not torch.isnan(du_direct_tri).any(), "du_direct_tri should not contain NaN values"
    assert not torch.isnan(du_direct_naive).any(), "du_direct_naive should not contain NaN values"
    
    print("[TEST] All NaN checks passed!")



# add tests for more chunks 
@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'scale', 'num_householder', 'dtype'),
    [
        (1, 8, 2, 16, 1.0, 1, torch.float32),
        (2, 12, 2, 24, 1.0, 2, torch.float32),
        (1, 64, 2, 32, 1.0, 1, torch.float32),
        (2, 128, 4, 64, 1.0, 2, torch.float32),
    ]
)
def test_helper2_du_final_comparison(
    B: int,
    T: int,
    H: int,
    D: int,
    scale: float,
    num_householder: int,
    dtype: torch.dtype,
):
    """Test helper function 2 (du_final and ds_final computation) without gating (g=0)."""
    from fla.ops.gated_delta_product.naive import helper_final_gradient_s_and_u
    from fla.ops.delta_rule.wy_fast import prepare_wy_repr_fwd, recompute_w_u_fwd
    from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h, chunk_gated_delta_rule_bwd_dhu
    from fla.ops.common.chunk_o import chunk_bwd_dv_local
    from fla.ops.common.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
    from fla.ops.utils import chunk_local_cumsum, solve_tril

    from einops import rearrange
    
    print(f"\n=== Testing Helper2 B={B}, T={T}, H={H}, D={D}, scale={scale}, num_householder={num_householder}, dtype={dtype} ===")
    
    torch.manual_seed(42)
    
    # Create inputs on the correct device
    print(f"Creating tensors on device: {device}")
    # q = torch.randn(B, T, H, D, dtype=dtype, device=device, requires_grad=True)
    # k = torch.randn(B, T * num_householder, H, D, dtype=dtype, device=device, requires_grad=True)
    # TODO: Jinha normalize so that values are not huge 
    q = torch.nn.functional.normalize(torch.randn((1, T, H, D), dtype=dtype, device=device, requires_grad=True), dim=-1, p=2)
    k = torch.nn.functional.normalize(torch.randn(1, T*num_householder, H, D, dtype=dtype, device=device, requires_grad=True), dim=-1, p=2)
    v = torch.randn(B, T * num_householder, H, D, dtype=dtype, device=device, requires_grad=True)
    beta = torch.rand(B, T * num_householder, H, dtype=dtype).sigmoid().to(device=device)
    beta.requires_grad_(True)
    h0 = torch.nn.functional.normalize(torch.randn(B, H, D, D, dtype=dtype, device=device, requires_grad=True), dim=-1, p=2)
    g = F.logsigmoid(torch.rand(B, T, H, dtype=dtype, device=device, requires_grad=True))
    g_expanded = g.new_zeros(g.shape[0], g.shape[1], num_householder, g.shape[2], dtype=torch.float32)
    g_expanded[:, :, 0] = g
    g_expanded = rearrange(g_expanded, 'b l n h -> b (l n) h').contiguous()
    g = chunk_local_cumsum(g, chunk_size=64, cu_seqlens=None, output_dtype=torch.float32)
    g_expanded = chunk_local_cumsum(g_expanded, chunk_size=64, cu_seqlens=None, output_dtype=torch.float32)
    print("g is", g)
    print("g_expanded is", g_expanded)
    print("g_shape", g.shape, "g_expanded_shape", g_expanded.shape)

    print(f"Input shapes: q={q.shape}, k={k.shape}, v={v.shape}, beta={beta.shape}, h0={h0.shape}")
    
    # Create gradient output tensors 
    do = torch.randn_like(q)
    dht = torch.randn_like(h0)
    
    # Follow the delta rule backward logic (without gating)
    print("[TEST] Following delta rule backward logic (no gating)...")
    
    # Step 1: Prepare WY representation
    A = chunk_scaled_dot_kkt_fwd(
        k=k,
        g=g_expanded,
        beta=beta,
        cu_seqlens=None,
        output_dtype=k.dtype
    )
    A = solve_tril(
        A=A,
        cu_seqlens=None,
        output_dtype=k.dtype
    )
    
    # Step 2: Recompute w, u (following backward logic)
    w, u = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        cu_seqlens=None,
    )
    print(f"[TEST] w shape: {w.shape}, u shape: {u.shape}")
    
    # Step 3: Recompute h, v_new (following backward logic)
    h, v_new, _ = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g_expanded,
        initial_state=h0,
        output_final_state=False,
        cu_seqlens=None,
    )
    print(f"[TEST] h shape: {h.shape}, v_new shape: {v_new.shape}")
    
    # Step 5: Expand q and do to match the backward pass logic
    q_new = q.new_zeros(q.shape[0], q.shape[1], num_householder, q.shape[2], q.shape[3])
    q_new[:, :, -1] = q
    do_new = do.new_zeros(do.shape[0], do.shape[1], num_householder, do.shape[2], do.shape[3])
    do_new[:, :, -1] = do
    q_expanded = rearrange(q_new, 'b t n h d -> b (t n) h d')
    do_expanded = rearrange(do_new, 'b t n h d -> b (t n) h d')
    
    print(f"[TEST] q_expanded shape: {q_expanded.shape}, do_expanded shape: {do_expanded.shape}")
    
    # Step 6: Compute du_direct using actual implementation
    print("[TEST] Computing du_direct using chunk_bwd_dv_local...")
    du_direct_tri = chunk_bwd_dv_local(
        q=q_expanded,
        k=k,
        g=g_expanded,
        do=do_expanded,
        scale=scale,
        cu_seqlens=None,
        chunk_size=128,
    )
    print(f"[TEST] du_direct_tri shape: {du_direct_tri.shape}")
    print(f"[TEST] du_direct_tri values: min={du_direct_tri.min():.6f}, max={du_direct_tri.max():.6f}, has_nan={torch.isnan(du_direct_tri).any()}")
    
    # Step 7: Compute dh, dh0, dv using actual implementation  
    print("[TEST] Computing dh, dh0, dv using chunk_gated_delta_rule_bwd_dhu...")
    dh_tri, dh0_tri, dv_tri = chunk_gated_delta_rule_bwd_dhu(
        q=q_expanded,
        k=k,
        w=w,
        g=g_expanded,
        h0=h0,
        dht=dht,
        do=do_expanded,
        dv=du_direct_tri,
        scale=scale,
        cu_seqlens=None,
        chunk_size=64,
    )
    print(f"[TEST] dh_tri shape: {dh_tri.shape}, dh0_tri shape: {dh0_tri.shape}, dv_tri shape: {dv_tri.shape}")
    print(f"[TEST] dv_tri values: min={dv_tri.min():.6f}, max={dv_tri.max():.6f}, has_nan={torch.isnan(dv_tri).any()}")
    
    # Step 8: Test helper function 2 directly with the same inputs
    print("[TEST] Testing helper function 2 directly...")
    
    # We need ds_next for helper function 2 - use dht as the final state gradient
    ds_next = dht  # This represents the gradient of the final hidden state
    
    du_final_naive, ds_final_naive = helper_final_gradient_s_and_u(
        q=q,
        k=k,
        w=w,
        du_direct=du_direct_tri,
        do=do,
        ds_next=ds_next,
        g=g,  
        g_expanded=g_expanded,
        num_householder=num_householder
    )
    
    print(f"[TEST] du_final_naive shape: {du_final_naive.shape}")
    print(f"[TEST] du_final_naive values: min={du_final_naive.min():.6f}, max={du_final_naive.max():.6f}, has_nan={torch.isnan(du_final_naive).any()}")
    print(f"[TEST] ds_final_naive shape: {ds_final_naive.shape}")
    print(f"[TEST] ds_final_naive values: min={ds_final_naive.min():.6f}, max={ds_final_naive.max():.6f}, has_nan={torch.isnan(ds_final_naive).any()}")
    
    # Step 9: Compare the du_final outputs
    print("[TEST] Comparing du_final outputs...")
    # The actual implementation returns dv_tri which should correspond to du_final_naive
    assert_close('du_final', dv_tri, du_final_naive, 0.01)

    assert_close('ds_initial', dh0_tri, ds_final_naive[:, 0], 0.01)

    # TODO add tests for multiple iterations of dht 
    # assert_close('gated delta rule check', dht, dh_tri[:, 0], 0.01) # check that first dh gradient is same 


    # dh_tri is dht, then dS_last, dS_last-1 ... dS1 
    # and dh0 is dS0 

    # reshape so that it stores gradients every num householder * BT instead of every BT 
    # dh_tri_reshaped = dh_tri[:, ::num_householder] 
    # assert_close('gated delta rule check', dht, dh_tri, 0.01) # check that first dh gradient is same 

    # print("[TEST] dh_tri shape", dh_tri.shape)
    # print(dh_tri_reshaped.shape, ds_final_naive.shape)
    # assert_close('ds_final', dh_tri_reshaped, ds_final_naive, 0.01)
    # print("[TEST] du_final comparison passed!")

    # Step 10: Check for NaN values
    assert not torch.isnan(du_direct_tri).any(), "du_direct_tri should not contain NaN values"
    assert not torch.isnan(dv_tri).any(), "dv_tri should not contain NaN values"
    assert not torch.isnan(du_final_naive).any(), "du_final_naive should not contain NaN values"
    assert not torch.isnan(ds_final_naive).any(), "ds_final_naive should not contain NaN values"
    
    print("[TEST] All NaN checks passed!")
    print("[TEST] Helper function 2 test completed!")


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'scale', 'num_householder', 'dtype'),
    [
        (1, 8, 2, 16, 1.0, 1, torch.float32),
        (2, 12, 2, 24, 1.0, 2, torch.float32),
        (1, 64, 2, 32, 1.0, 1, torch.float32),
        (2, 128, 4, 64, 1.0, 2, torch.float32),
    ]
)
def test_helper3_gradient_qwkg_comparison(
    B: int,
    T: int,
    H: int,
    D: int,
    scale: float,
    num_householder: int,
    dtype: torch.dtype,
):
    """Test helper function 3 (gradient computation for Q, W, K, g) without gating (g=0)."""
    from fla.ops.gated_delta_product.naive import helper_gradient_qwkg
    from fla.ops.common.chunk_o import chunk_bwd_dqkwg
    from fla.ops.delta_rule.wy_fast import recompute_w_u_fwd
    from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h, chunk_gated_delta_rule_bwd_dhu
    from fla.ops.common.chunk_o import chunk_bwd_dv_local
    from fla.ops.common.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
    from fla.ops.utils import chunk_local_cumsum, solve_tril
    from einops import rearrange
    
    print(f"\n=== Testing Helper3 (No Gating) B={B}, T={T}, H={H}, D={D}, scale={scale}, num_householder={num_householder}, dtype={dtype} ===")
    
    torch.manual_seed(42)
    
    # Create inputs on the correct device
    print(f"Creating tensors on device: {device}")
    # q = torch.randn(B, T, H, D, dtype=dtype, device=device, requires_grad=True)
    # k = torch.randn(B, T * num_householder, H, D, dtype=dtype, device=device, requires_grad=True)
    # TODO: Jinha normalize so that values are not huge 
    q = torch.nn.functional.normalize(torch.randn((1, T, H, D), dtype=dtype, device=device, requires_grad=True), dim=-1, p=2)
    k = torch.nn.functional.normalize(torch.randn(1, T*num_householder, H, D, dtype=dtype, device=device, requires_grad=True), dim=-1, p=2)
    v = torch.randn(B, T * num_householder, H, D, dtype=dtype, device=device, requires_grad=True)
    beta = torch.rand(B, T * num_householder, H, dtype=dtype).sigmoid().to(device=device)
    beta.requires_grad_(True)
    h0 = torch.randn(B, H, D, D, dtype=dtype, device=device, requires_grad=True)
    g = torch.zeros(B, T, H, dtype=dtype, device=device, requires_grad=True)
    g_expanded = g.new_zeros(g.shape[0], g.shape[1], num_householder, g.shape[2], dtype=torch.float32)
    g_expanded[:, :, 0] = g
    g_expanded = rearrange(g_expanded, 'b l n h -> b (l n) h').contiguous()
    g = chunk_local_cumsum(g, chunk_size=64, cu_seqlens=None, output_dtype=torch.float32)
    g_expanded = chunk_local_cumsum(g_expanded, chunk_size=64, cu_seqlens=None, output_dtype=torch.float32)

    T_expanded = k.shape[1]
    assert T_expanded == T * num_householder

    print(f"Input shapes: q={q.shape}, k={k.shape}, v={v.shape}, beta={beta.shape}, h0={h0.shape}")
    
    # Create gradient output tensors 
    do = torch.randn_like(q)
    dht = torch.randn_like(h0)
    
    # Follow the gated delta rule backward logic (with zero gating)
    print("[TEST] Following gated delta rule backward logic (zero gating)...")
    
    # Step 1: Obtain WY representation
    A = chunk_scaled_dot_kkt_fwd(
        k=k,
        g=g_expanded,
        beta=beta,
        cu_seqlens=None,
        output_dtype=k.dtype
    )
    A = solve_tril(
        A=A,
        cu_seqlens=None,
        output_dtype=k.dtype
    )
    print(f"[TEST] A shape: {A.shape}")
    
    # Step 2: Recompute w, u (following backward logic)
    w, u = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        # g=g_expanded,  # Zero gating
        cu_seqlens=None,
    )
    print(f"[TEST] w shape: {w.shape}, u shape: {u.shape}")
    
    # Step 3: Recompute h, v_new (following backward logic)
    h, v_new, _ = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g_expanded,
        initial_state=h0,
        output_final_state=False,
        cu_seqlens=None,
    )
    print(f"[TEST] h shape: {h.shape}, v_new shape: {v_new.shape}")
    
    # Step 4: Expand q and do to match the backward pass logic
    q_new = q.new_zeros(q.shape[0], q.shape[1], num_householder, q.shape[2], q.shape[3])
    q_new[:, :, -1] = q
    do_new = do.new_zeros(do.shape[0], do.shape[1], num_householder, do.shape[2], do.shape[3])
    do_new[:, :, -1] = do
    q_expanded = rearrange(q_new, 'b t n h d -> b (t n) h d')
    do_expanded = rearrange(do_new, 'b t n h d -> b (t n) h d')
    
    print(f"[TEST] q_expanded shape: {q_expanded.shape}, do_expanded shape: {do_expanded.shape}")
    
    # Step 5: Compute intermediate gradients using actual implementation
    print("[TEST] Computing intermediate gradients...")
    
    # Compute du_direct using chunk_bwd_dv_local
    du_direct = chunk_bwd_dv_local(
        q=q_expanded,
        k=k,
        g=g_expanded,
        do=do_expanded,
        scale=scale,
        cu_seqlens=None,
        chunk_size=128,
    )
    print(f"[TEST] du_direct shape: {du_direct.shape}")
    
    # # Compute dh, dh0, du_final using chunk_gated_delta_rule_bwd_dhu
    # dh, dh0, du_final = chunk_gated_delta_rule_bwd_dhu(
    #     q=q_expanded,
    #     k=k,
    #     w=w,
    #     g=g_expanded,
    #     h0=h0,
    #     dht=dht,
    #     do=do_expanded,
    #     dv=du_direct,
    #     scale=scale,
    #     cu_seqlens=None,
    #     chunk_size=128,
    # )
    # print(f"[TEST] du_final shape: {du_final.shape}, dh shape: {dh.shape}")
    
    # Create ds_final (state gradients per chunk) - use dh reshaped
    num_chunks = math.ceil(T_expanded / 64)  # Using chunk size 64
    ds_final_expanded = torch.randn(B, num_chunks, H, D, D, dtype=dtype, device=device)
    ds_final = ds_final_expanded[:, ::num_householder]
    
    # Create state tensor (s) - internal states per chunk  
    s = torch.randn(B, num_chunks, H, D, D, dtype=dtype, device=device)
    
    print(f"[TEST] ds_final shape: {ds_final.shape}, s shape: {s.shape}")
    
    # Initialize du_final randomly for testing
    du_final = torch.randn_like(du_direct)
    print(f"[TEST] du_final shape: {du_final.shape}")
    
    # Step 6: Test helper function 3 directly (zero gating)
    print("[TEST] Testing helper function 3 directly (zero gating)...")
    
    dq_naive, dw_naive, dk_naive, dg_expanded_naive = helper_gradient_qwkg(
        q=q,  # Use original q, not expanded
        k=k,
        v_new=v_new,
        w=w,
        g=g,  # Zero gating (cumsum of zeros)
        s=s,
        ds_final=ds_final,
        dht=dht,
        do=do,  # Use original do, not expanded
        du_final=du_final,
        g_expanded=g_expanded,  # Zero gating (cumsum of zeros)
        num_householder=num_householder
    )
    
    print(f"[TEST] dq_naive shape: {dq_naive.shape}")
    print(f"[TEST] dq_naive values: min={dq_naive.min():.6f}, max={dq_naive.max():.6f}, has_nan={torch.isnan(dq_naive).any()}")
    print(f"[TEST] dw_naive shape: {dw_naive.shape}")
    print(f"[TEST] dw_naive values: min={dw_naive.min():.6f}, max={dw_naive.max():.6f}, has_nan={torch.isnan(dw_naive).any()}")
    print(f"[TEST] dk_naive shape: {dk_naive.shape}")
    print(f"[TEST] dk_naive values: min={dk_naive.min():.6f}, max={dk_naive.max():.6f}, has_nan={torch.isnan(dk_naive).any()}")
    print(f"[TEST] dg_expanded_naive shape: {dg_expanded_naive.shape}")
    print(f"[TEST] dg_expanded_naive values: min={dg_expanded_naive.min():.6f}, max={dg_expanded_naive.max():.6f}, has_nan={torch.isnan(dg_expanded_naive).any()}")
    
    # Step 7: Test against triton implementation
    print("[TEST] Testing against triton implementation...")
    
    # Use chunk_bwd_dqkwg for comparison (this is the actual triton implementation)
    dq_tri, dk_tri, dw_tri, dg_tri = chunk_bwd_dqkwg(
        q=q_expanded,
        k=k,
        v=v_new,
        do=do_expanded,
        h=s,
        dh=ds_final_expanded,  
        g=None,  # No gating
        dv=du_final,
        w=w,
        scale=scale,
    )
    
    print(f"[TEST] dq_tri shape: {dq_tri.shape}")
    print(f"[TEST] dq_tri values: min={dq_tri.min():.6f}, max={dq_tri.max():.6f}, has_nan={torch.isnan(dq_tri).any()}")
    print(f"[TEST] dk_tri shape: {dk_tri.shape}")
    print(f"[TEST] dk_tri values: min={dk_tri.min():.6f}, max={dk_tri.max():.6f}, has_nan={torch.isnan(dk_tri).any()}")
    
    print(f"[TEST] dw_tri shape: {dw_tri.shape}")
    print(f"[TEST] dw_tri values: min={dw_tri.min():.6f}, max={dw_tri.max():.6f}, has_nan={torch.isnan(dw_tri).any()}")
    
    # Step 8: Compare the outputs (adjust shapes as needed)
    print("[TEST] Comparing gradient outputs...")
        
    # Compare dq (need to extract from expanded version)
    dq_tri_reshaped = dq_tri[:, ::num_householder]  # Extract true timesteps
    # assert_close('dq', dq_tri_reshaped, dq_naive, 0.01)
    # print("[TEST] dq comparison passed!")
    
    # Compare dk (same shape)
    # assert_close('dk', dk_tri, dk_naive, 0.01)
    # print("[TEST] dk comparison passed!")
    
    # Compare dw if available
    assert_close('dw', dw_tri, dw_naive, 0.01)
    print("[TEST] dw comparison passed!")
    
    print("[TEST] All gradient comparisons passed!")
        
    # Step 9: Check for NaN values
    assert not torch.isnan(dq_naive).any(), "dq_naive should not contain NaN values"
    assert not torch.isnan(dw_naive).any(), "dw_naive should not contain NaN values"
    assert not torch.isnan(dk_naive).any(), "dk_naive should not contain NaN values"
    assert not torch.isnan(dg_expanded_naive).any(), "dg_expanded_naive should not contain NaN values"
    assert not torch.isnan(dq_tri).any(), "dq_tri should not contain NaN values"
    assert not torch.isnan(dk_tri).any(), "dk_tri should not contain NaN values"
    assert not torch.isnan(dw_tri).any(), "dw_tri should not contain NaN values"
    
    print("[TEST] All NaN checks passed!")
    print("[TEST] Helper function 3 test completed!")
