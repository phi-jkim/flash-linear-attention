# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional

from fla.ops.abc import chunk
from fla.ops.gated_delta_product.wy_fast import prepare_wy_repr_bwd_expanded, recompute_w_u_expanded
import torch
import triton
from einops import rearrange
from fla.modules.l2norm import l2norm_bwd, l2norm_fwd
from fla.ops.common.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from fla.ops.delta_rule.wy_fast import recompute_w_u_fwd as dn_recompute_w_u_fwd
from fla.ops.gated_delta_product.chunk_deltaproduct_h import chunk_gated_delta_product_fwd_h
from fla.ops.gated_delta_product.chunk_deltaproduct_o import chunk_gated_delta_product_fwd_o
from fla.ops.gated_delta_rule.wy_fast import recompute_w_u_fwd as gdn_recompute_w_u_fwd
from fla.ops.utils import chunk_local_cumsum, solve_tril
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard


def chunk_gated_delta_product_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    cu_seqlens: Optional[torch.LongTensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    num_householder: int = 1,
):
    cu_seqlens_dp = cu_seqlens * num_householder if cu_seqlens is not None else None
    # next power of 2 of num_householder * BT 
    chunk_size = 64
    expanded_chunk_size = chunk_size * triton.next_power_of_2(num_householder)
    if g is not None:
        g_interleaved = g.new_zeros(g.shape[0], g.shape[1], num_householder, g.shape[2], dtype=torch.float32)
        g_interleaved[:, :, 0] = g
        g_interleaved = rearrange(g_interleaved, 'b l n h -> b (l n) h').contiguous()
        g = chunk_local_cumsum(g, chunk_size=chunk_size, cu_seqlens=cu_seqlens, output_dtype=torch.float32)
        g_interleaved = chunk_local_cumsum(g_interleaved, chunk_size=chunk_size, cu_seqlens=cu_seqlens_dp, output_dtype=torch.float32)
        g_interleaved_N = chunk_local_cumsum(g_interleaved, chunk_size=expanded_chunk_size, cu_seqlens=cu_seqlens_dp, output_dtype=torch.float32)
    else:
        g_interleaved_N = None
        g_interleaved = None
        g = None

    # obtain WY representation. u is actually the new v.
    A = chunk_scaled_dot_kkt_fwd(
        k=k,
        g=g_interleaved,
        beta=beta,
        cu_seqlens=cu_seqlens_dp,
        output_dtype=torch.float32
    )
    A = solve_tril(
        A=A,
        cu_seqlens=cu_seqlens_dp,
        output_dtype=k.dtype
    )
    if g is not None:
        w, u = gdn_recompute_w_u_fwd(
            k=k,
            v=v,
            beta=beta,
            A=A,
            g=g_interleaved,
            cu_seqlens=cu_seqlens_dp,
        )
    else:
        w, u = dn_recompute_w_u_fwd(
            k=k,
            v=v,
            beta=beta,
            A=A,
            cu_seqlens=cu_seqlens_dp,
        )
    h, v_new, final_state = chunk_gated_delta_product_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g_interleaved,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens_dp,
        num_householder=num_householder,
    )
    o = chunk_gated_delta_product_fwd_o(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
        num_householder=num_householder,
    )
    return g, g_interleaved, g_interleaved_N, o, A, final_state


def chunk_gated_delta_product_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    g_interleaved: torch.Tensor,
    g_interleaved_N: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    scale: float,
    cu_seqlens: Optional[torch.LongTensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    num_householder: int = 1,
):
    chunk_size = 64
    expanded_chunk_size = chunk_size * triton.next_power_of_2(num_householder) # 64 * 8 when num householder is 5 

    q_new = q.new_zeros(q.shape[0], q.shape[1], num_householder, q.shape[2], q.shape[3])
    q_new[:, :, -1] = q
    do_new = do.new_zeros(do.shape[0], do.shape[1], num_householder, do.shape[2], do.shape[3])
    do_new[:, :, -1] = do
    q_org, q = q, rearrange(q_new, 'b t n h d -> b (t n) h d')
    do_org, do = do, rearrange(do_new, 'b t n h d -> b (t n) h d')

    from fla.ops.gated_delta_product.chunk_deltaproduct_h import (
        chunk_gated_delta_product_bwd_dhu,
        chunk_gated_delta_product_fwd_h
    )

    cu_seqlens_dp = cu_seqlens * num_householder if cu_seqlens is not None else None

    # recompute w, u from WY representation to compute gradient
    from fla.ops.gated_delta_rule.wy_fast import recompute_w_u_fwd 

    A = chunk_scaled_dot_kkt_fwd(
        k=k,
        g=g_interleaved,
        beta=beta,
        cu_seqlens=cu_seqlens_dp,
        output_dtype=torch.float32, 
        # chunk_size=64*num_householder,
        # chunk_size=64,
        chunk_size=expanded_chunk_size,
    )

    A = solve_tril(
        A=A,
        cu_seqlens=cu_seqlens_dp,
        output_dtype=k.dtype
    )

    from fla.ops.gated_delta_product.wy_fast import recompute_w_u_expanded

    w, u = recompute_w_u_expanded(
        k=k, v=v, beta=beta, A_expanded=A, g=g_interleaved_N, cu_seqlens=cu_seqlens_dp, N=expanded_chunk_size
    )

    # w, u = recompute_w_u_fwd(
    #     k=k, v=v, beta=beta, A=A, g=g_interleaved, cu_seqlens=cu_seqlens_dp, 
    # )


    # from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h

    # # TODO replace with delta product fwd_h 
    # h, v_new, _ = chunk_gated_delta_rule_fwd_h(
    #     k=k,
    #     w=w,
    #     u=u,
    #     g=g_interleaved_N,
    #     initial_state=initial_state,
    #     output_final_state=False,
    #     cu_seqlens=cu_seqlens_dp,
    #     # chunk_size=64*num_householder,
    #     chunk_size=64,
    # )

    from fla.ops.gated_delta_product.chunk_deltaproduct_h import chunk_gated_delta_product_fwd_h_expanded
    
    # Recompute h and v_new using delta product forward
    h, v_new, _ = chunk_gated_delta_product_fwd_h_expanded(
        k=k,
        w=w,
        u=u,
        g=g_interleaved_N,
        initial_state=initial_state,
        output_final_state=False,
        cu_seqlens=cu_seqlens_dp,
        chunk_size = chunk_size,
        num_householder=num_householder,
    )

    # from fla.ops.common.chunk_o import chunk_bwd_dv_local

    # Compute local gradient w.r.t v_new
    # dv_new = chunk_bwd_dv_local(
    #     q=q,
    #     k=k,
    #     g=g_interleaved,
    #     do=do,
    #     scale=scale,
    #     cu_seqlens=cu_seqlens_dp,
    #     # chunk_size=64*num_householder,
    #     chunk_size=64,
    # )

    from fla.ops.gated_delta_product.chunk_deltaproduct_o import chunk_gated_delta_product_bwd_dv_local

    dv_new = chunk_gated_delta_product_bwd_dv_local(
        q=q_org,
        k=k,
        g=g_interleaved_N,
        do=do_org,
        scale=scale,
        cu_seqlens=cu_seqlens_dp,
        # chunk_size=64*num_householder,
        chunk_size=chunk_size,
        num_householder=num_householder
    )


    # from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu

    # dq, dk, dw, dg = chunk_gated_delta_rule_bwd_dqkwg(
    #     q=q,
    #     k=k,
    #     w=w,
    #     g=g_interleaved,  # chunk_gated_delta_product_fwd_h uses g_interleaved
    #     h0=initial_state,  # H_0
    #     dht=dht,  # gradient w.r.t to last hidden state
    #     do=do,  # gradient of the output
    #     dv=dv_new,  # gradient w.r.t. v_new
    #     scale=scale,
    #     cu_seqlens=cu_seqlens_dp,  # use cu_seqlens_dp which is expanded
    #     # chunk_size=64*num_householder,
    #     num_householder=num_householder,
    #     chunk_size=64,
    # )

    from fla.ops.gated_delta_product.chunk_deltaproduct_h import chunk_gated_delta_product_bwd_dhu

    # Use optimized delta product backward for hidden states
    # Note: This uses v_new from the forward pass for efficiency
    dh, dh0, du = chunk_gated_delta_product_bwd_dhu(
        q=q_org,                   # Use original q (not expanded)
        k=k,                       # k is already expanded (B, T*num_householder, H, K)
        w=w,                       # pass W (not v_new)
        g=g_interleaved_N,         # Use original g (not interleaved)
        h0=initial_state,
        dht=dht,
        do=do_org,                 # Use original do (not expanded)
        dv=dv_new,                 # Gradient w.r.t. v_new
        scale=scale,
        cu_seqlens=cu_seqlens_dp,  # Expanded sequence lengths
        num_householder=num_householder,
        chunk_size=chunk_size,
    )

    # Use optimized delta product backward for output gradients
    # from fla.ops.gated_delta_product.chunk_deltaproduct_o import chunk_bwd_dqkwg

    # # This version works directly with original tensors and handles num_householder internally
    # dq, dk_direct_gradient, dw, dg_local = chunk_bwd_dqkwg(
    #     q=q_org,                      # Original q (not expanded)
    #     k=k,                          # Expanded k (B, T*num_householder, H, K)
    #     v=v,                          # Original v (expanded)
    #     do=do_org,                    # Original do (not expanded)
    #     h=h,                          # Hidden states from forward
    #     dh=dh,                        # Hidden state gradients from dhu
    #     g=g,                          # Original g (not interleaved)
    #     g_gamma=None,                 # Not used in this version
    #     dv=du,                      # Not needed for delta product version
    #     w=w,                       # W handled internally
    #     cu_seqlens=cu_seqlens_dp,     # Expanded sequence lengths
    #     scale=scale,
    #     num_householder=num_householder,
    # )

    # from fla.ops.common.chunk_o import chunk_bwd_dqkwg

    # # # TODO implement delta product version of chunk_bwd_dqkwg
    # # # dq and dw is final
    # # # By multivariate chain rule, for L = f(O) and O = g(k, v_new(k))
    # # # dL/dK = dL/dO * dO/dK + dL/dO * dO/dv_new * dv_new/dK
    # # # dk_direct_gradient is the direct gradient computed considering v_new as a fixed quantity (by product rule)
    # # # dk_direct_gradient should be fully parallelizable
    # # # might be sequential as dq depends on H (might need forward pass)
    # dq, dk_direct_gradient, dw, dg_local = chunk_bwd_dqkwg(
    #     q=q,
    #     k=k,
    #     v=v_new,  # v_new = U[i] - W[i]H[i]^T
    #     w=w,
    #     g=g_interleaved,  # should this be g or g_interleaved? since we don't find dk for the hidden states, is it g
    #     h=h,
    #     dv=du,  # can be thought as gradient wrt v_new
    #     do=do,
    #     dh=dh,
    #     scale=scale,
    #     cu_seqlens=cu_seqlens_dp,  # cu_seqlens * num_householder
    #     # chunk_size=64*num_householder,
    #     chunk_size=64,
    # )

    from fla.ops.gated_delta_product.chunk_deltaproduct_o import chunk_bwd_dqkwg

    dq, dk_direct_gradient, dw, dg_local = chunk_bwd_dqkwg(
        q=q_org,
        k=k,
        v=v_new,  # v_new = U[i] - W[i]H[i]^T
        w=w,
        g=g_interleaved,  # should this be g or g_interleaved? since we don't find dk for the hidden states, is it g
        h=h,
        dv=du,  # can be thought as gradient wrt v_new
        do=do_org,
        dh=dh,
        scale=scale,
        cu_seqlens=cu_seqlens_dp,  # cu_seqlens * num_householder
        # chunk_size=64*num_householder,
        chunk_size=chunk_size,
    )


    # compute gradients w.r.t. WY representation (dk, dv, dbeta, dg)
    # This involves computing gradients through the Householder transformations
    # from fla.ops.gated_delta_rule.wy_fast import prepare_wy_repr_bwd

    # # compute gradient descent wrt W and U
    # # g_interleaved is used for computing W and U in the forward pass
    # # dv is the final gradient wrt to v (only place v appears)
    # # this should be fully parallelized
    # # TODO implement delta product version of prepare_wy_repr_bwd
    # dk_hidden_state_gradient, dv, dbeta, dg2 = prepare_wy_repr_bwd(
    #     k=k,
    #     v=v,
    #     beta=beta,
    #     g=g_interleaved,
    #     A=A,
    #     dw=dw,  # Use key gradients from output as weights gradients
    #     du=du,  # Use value gradients from hidden tate backward
    #     cu_seqlens=cu_seqlens_dp,
    #     # chunk_size=64*num_householder,
    #     chunk_size=64,
    # )

    from fla.ops.gated_delta_product.wy_fast import prepare_wy_repr_bwd_expanded
    dk_hidden_state_gradient, dv, dbeta, dg2 = prepare_wy_repr_bwd_expanded(
        k=k,
        v=v,
        beta=beta,
        g=g_interleaved_N,
        A_expanded=A,
        dw=dw,  # Use key gradients from output as weights gradients
        du=du,  # Use value gradients from hidden tate backward
        cu_seqlens=cu_seqlens_dp,
        N = expanded_chunk_size,
        # chunk_size=64*num_householder,
        # chunk_size=chunk_size,
    )

    # Accumulate gradients
    # For delta product: dk = dk_direct_gradient + dk_hidden_state_gradient
    # dk_direct_gradient comes from chunk_bwd_dqkwg (direct path through output)
    # dk_hidden_state_gradient comes from prepare_wy_repr_bwd (through WY representation)
    dk = dk_direct_gradient.add_(dk_hidden_state_gradient)  # dL/dK = dL/dO * dO/dK + dL/dO * dO/dv_new * dv_new/dK
    
    # For delta product, dv comes from prepare_wy_repr_bwd
    # (Delta rule would combine dv from chunk_bwd_dqkwg and prepare_wy_repr_bwd)
    dv_final = dv
    
    # Gate gradients: combine local gradients from output and WY representation
    dg_final = dg2
    
    # process gating gradients with local cumsum (reverse)
    if g_interleaved is not None:
        dg_final.add_(dg_local)  # dL/dg = dL/dO * dO/dg + dL/dO * dO/dv_new * dv_new/dg = dg_local + dg2
        assert dg_final.dtype == torch.float32, "dg_final should be fp32"
        from fla.ops.utils import chunk_local_cumsum
        # dg_final = chunk_local_cumsum(dg_final, chunk_size=64, reverse=True, cu_seqlens=cu_seqlens_dp)
        dg_final = chunk_local_cumsum(dg_final, chunk_size=expanded_chunk_size, reverse=True, cu_seqlens=cu_seqlens_dp)

        # Convert interleaved gating gradients back to original format
        dg_final = rearrange(dg_final, 'b (l n) h -> b l n h', n=num_householder)[:, :, 0].contiguous()
    else:
        dg_final = None

    return dq, dk, dv_final, dg_final, dbeta, dh0


class ChunkGatedDeltaProductFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        num_householder: int,
        initial_state: torch.Tensor,
        output_final_state: bool,
        use_qk_l2norm_in_kernel: bool = False,
        cu_seqlens: Optional[torch.LongTensor] = None,
    ):
        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)
        else:
            q_rstd, k_rstd = None, None

        g, g_interleaved, g_interleaved_N, o, A, final_state = chunk_gated_delta_product_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            num_householder=num_householder,
        )
        ctx.save_for_backward(q, q_rstd, k, k_rstd, v, g, g_interleaved, g_interleaved_N, beta, A, initial_state, cu_seqlens)
        ctx.scale = scale
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.num_householder = num_householder
        return o.to(q.dtype), final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(
        ctx,
        do: torch.Tensor,
        dht: torch.Tensor
    ):
        q, q_rstd, k, k_rstd, v, g, g_interleaved, g_interleaved_N, beta, A, initial_state, cu_seqlens = ctx.saved_tensors

        # recompute forward intermediate values
        # Call optimized delta product backward pass
        dq, dk, dv, dg, db, dh0 = chunk_gated_delta_product_bwd(
            q=q,
            k=k,
            v=v,
            g=g,  # use g
            g_interleaved=g_interleaved,  # use computed g_interleaved
            g_interleaved_N=g_interleaved_N,
            beta=beta,
            A=A,
            do=do,
            dht=dht,
            scale=ctx.scale,
            cu_seqlens=cu_seqlens,
            initial_state=initial_state,
            num_householder=ctx.num_householder,
        )

        # dq = rearrange(dq, 'b (l n) h d -> b l n h d', n=ctx.num_householder)[:, :, -1].contiguous()
        
        # if use_qk_l2norm_in_kernel, do l2norm_bwd (calculate gradient for l2norm)
        if ctx.use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q, q_rstd, dq)
            dk = l2norm_bwd(k, k_rstd, dk)

        return dq.to(q), dk.to(k), dv.to(v), dg, db.to(beta), None, None, dh0, None, None, None


@torch.compiler.disable
def chunk_gated_delta_product(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    num_householder: int,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
):
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, T, H, V]`.
        g (torch.Tensor):
            (forget) gating tensor (in log space!) of shape `[B, T, H]`.
        beta (torch.Tensor):
            betas of shape `[B, T, H]`.
        num_householder (int):
            Number of householder transformations to apply. Default: `1`.
        scale (Optional[float]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.
        use_qk_l2norm_in_kernel (Optional[bool]):
            Whether to use qk l2norm within the kernel for saving GPU memory.
            Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, H, K, V]` if `output_final_state=True` else `None`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.gated_delta_rule import chunk_gated_delta_product
        # inputs with equal lengths
        >>> B, T, H, K, V = 4, 2048, 4, 512, 512
        >>> q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda')
        >>> k = F.normalize(torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda'), p=2, dim=-1)
        >>> v = torch.randn(B, T, H, V, dtype=torch.bfloat16, device='cuda')
        >>> beta = torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda').sigmoid()
        >>> g = F.logsigmoid(torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda'))
        >>> h0 = torch.randn(B, H, K, V, dtype=torch.bfloat16, device='cuda')
        >>> o, ht = chunk_gated_delta_product(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True
        )
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, beta, g = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, beta, g))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o, ht = chunk_gated_delta_product(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=cu_seqlens
        )
    """
    assert q.dtype != torch.float32, "ChunkGatedDeltaProductFunction does not support float32. Please use bfloat16."
    B, T, H, K, V = *q.shape, v.shape[-1]
    assert k.shape == (B, T*num_householder, H, K)
    assert v.shape == (B, T*num_householder, H, V)
    assert beta.shape == (B, T*num_householder, H)
    if g is not None:
        assert g.shape == (B, T, H)

    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing."
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}."
            )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    o, final_state = ChunkGatedDeltaProductFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        scale,
        num_householder,
        initial_state,
        output_final_state,
        use_qk_l2norm_in_kernel,
        cu_seqlens,
    )
    return o, final_state
