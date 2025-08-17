import torch
import math
from typing import Optional

# Import required functions for backward pass
from fla.ops.gated_delta_rule.wy_fast import (
    recompute_w_u_fwd as gdn_recompute_w_u_fwd,
)

from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from fla.ops.delta_rule.wy_fast import recompute_w_u_fwd as dn_recompute_w_u_fwd

from fla.ops.utils import chunk_local_cumsum, solve_tril
from fla.ops.common.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd


def naive_recurrent_gated_delta_product(q, k, v, g, beta, scale, cu_seqlens,
                                        initial_state=None, output_final_state=False,
                                        num_householder=1):
    q_original_dtype = q.dtype
    B, T, H, K = q.shape
    V = v.shape[-1]
    assert k.shape == (B, T*num_householder, H, K)
    assert v.shape == (B, T*num_householder, H, V)
    assert beta.shape == (B, T*num_householder, H)
    if g is not None:
        assert g.shape == (B, T, H)
    q, k, v, beta = map(lambda x: x.float(), (q, k, v, beta))

    h = torch.zeros(B, H, K, V, dtype=torch.float32, device=q.device)
    if initial_state is not None:
        h = initial_state

    o = torch.zeros(B, T, H, V, dtype=torch.float32, device=q.device)

    for i in range(T):
        if g is not None:
            h = h * g[:, i, :].exp()[..., None, None]
        # multiple state transition
        for j in range(num_householder):
            k_ij = k[:, i*num_householder+j, :, :]
            v_ij = v[:, i*num_householder+j, :, :]
            beta_ij = beta[:, i*num_householder+j, :]
            h = h + (v_ij - (h * k_ij[..., None]).sum(-2)).unsqueeze(-2) * k_ij[..., None] * beta_ij[..., None, None]
        # memory readout
        q_i = q[:, i, :, :]
        o_i = (h * q_i[..., None]).sum(-2)
        o[:, i] = o_i
    return o.to(q_original_dtype), h


# Helper Function 1: Direct gradient computation for U[t] - W[t]S[t]^T (PDF Section 1)
def helper_direct_gradient_u_minus_ws(q, k, do, num_householder):
    """
    Compute direct gradient ∂/∂(U[t] - W[t]S[t]^T) = (Q'[t:t+N]K[t:t+N]^T ⊙ M)^T · ∂/∂O'[t:t+N]
    
    Following PDF Section 1, this computes the gradient through the term (U[t] - W[t]S[t]^T)
    in the delta product rule equation.
    
    Args:
        q: Query tensor (B, T_true, H, K)  
        k: Key tensor (B, T_expanded, H, K)
        g: Gate tensor (B, T_true, H) - used to construct gamma
        do: Output gradient (B, T_true, H, V)
        s_state: State tensor S[t] (B, H, K, V)
        mask: Causal mask M
        g_expanded: Expanded gamma values (B, T_expanded, H)
        num_householder: Number of householder steps
        
    Returns:
        Gradient tensor of shape (B, T_expanded, H, V)
    """
    print(f"[HELPER1] helper_direct_gradient_u_minus_ws called")
    print(f"[HELPER1] Input shapes: q={q.shape}, k={k.shape}, do={do.shape}")
    print(f"[HELPER1] num_householder={num_householder}")
    
    B, T_true, H, K = q.shape
    T_expanded = k.shape[1]
    assert(T_true * num_householder == T_expanded)
    V = do.shape[-1]
    
    print(f"[HELPER1] Computed: B={B}, T_true={T_true}, T_expanded={T_expanded}, H={H}, K={K}, V={V}")
    
    # Initialize output gradient
    du_direct = torch.zeros(B, T_expanded, H, V, device=q.device, dtype=q.dtype)
    
    # Apply gamma scaling to Q and dO
    # if gamma_expanded is not None:
    #     q_expanded = q_expanded * gamma_expanded.unsqueeze(-1)
    #     do_expanded = do_expanded * gamma_expanded.unsqueeze(-1)
    
    # Compute A = Q'[t:t+N]K[t:t+N]^T ⊙ M for chunks
    BT = 64  # Chunk size
    num_chunks = math.ceil(T_expanded / BT)

    # Apply causal mask
    expanded_causal_mask = torch.tril(torch.ones(T_expanded, T_expanded, device=q.device))
    # take every numhouseholder row starting from the num_householder row 
    causal_mask = expanded_causal_mask[num_householder-1::num_householder, :] # (T_true, T_expanded)
    
    # BT chunks 
    for chunk_idx in range(num_chunks):
        # index of tokens of expanded tokens 
        expanded_t_start = chunk_idx * BT
        expanded_t_end = min(expanded_t_start + BT, T_expanded)
        expanded_chunk_size = expanded_t_end - expanded_t_start

        # index for actual tokens of q and do 
        t_start = (expanded_t_start // (num_householder * BT)) * BT  
        t_end = min(t_start + BT, T_true)    

        if expanded_chunk_size <= 0:
            continue
            
        # Extract chunks
        q_chunk = q[:, t_start:t_end]  # (B, chunk_size, H, K)
        k_chunk = k[:, expanded_t_start:expanded_t_end]           # (B, chunk_size, H, K)
        do_chunk = do[:, t_start:t_end] # (B, chunk_size, H, V)

        # TODO chec kthat q_chunk and k_chunk have same dimensions (can have different chunk dimensions other than dk)

        # TODO implement iterating over separating DK as well 
        
        # Compute QK^T for each batch and head
        for b in range(B):
            for h in range(H):
                # QK^T: (chunk_size, K) @ (K, chunk_size) -> (chunk_size, chunk_size)
                qk = torch.mm(q_chunk[b, :, h], k_chunk[b, :, h].t())
                
                masked_qk = qk * causal_mask[t_start:t_end, expanded_t_start:expanded_t_end]
                
                # Compute A^T @ dO: (chunk_size, chunk_size)^T @ (chunk_size, V) -> (chunk_size, V)
                du_chunk = torch.mm(masked_qk.t(), do_chunk[b, :, h])
                
                # Store result
                du_direct[b, expanded_t_start:expanded_t_end, h] = du_chunk
    
    return du_direct


# Helper Function 2: Final gradient computation for S[t] and U[t] - W[t]S[t]^T (PDF Section 2)
def helper_final_gradient_s_and_u(q, k, w, du_direct, do, ds_next, g, g_expanded, 
                                num_householder):
    """
    Compute final gradients following PDF Section 2:
    ∂/∂(U[t:t+N] - W[t:t+N]S[t]^T) = K[t:t+N](∂/∂S[t+1])^T + ∂_direct(U[t:t+N] - W[t:t+N]S[t]^T)
    ∂/∂S[t]^T = Q'[t:t+N]^T ∂/∂O'[t:t+N] + ∂/∂S[t+1]^T · γ'[t] - W[t:t+N]^T × ∂/∂(U[t:t+N] - W[t:t+N]S[t]^T)
    
    Args:
        q: Query tensor (B, T_true, H, K)
        k: Key tensor (B, T_expanded, H, K) 
        w: W matrix (B, T_expanded, H, K)
        du_direct: Direct gradient from helper function 1 (B, T_expanded, H, V)
        do: Output gradient (B, T_true, H, V)
        ds_next: Next state gradient ∂/∂S[t+1] (B, H, K, V)
        num_householder: Number of householder steps
        
    Returns:
        Tuple of (du_final, ds_final) gradients
    """
    print(f"[HELPER2] helper_final_gradient_s_and_u called")
    print(f"[HELPER2] Input shapes: q={q.shape}, k={k.shape}, w={w.shape}")
    print(f"[HELPER2] du_direct={du_direct.shape}, do={do.shape}, ds_next={ds_next.shape}")
    print(f"[HELPER2] g={g.shape}, g_expanded={g_expanded.shape}")

    ''' 
    returns derivative of non gated S 
    '''

    B, T_true, H, K = q.shape
    T_expanded = k.shape[1]
    V = do.shape[-1]
    
    print(f"[HELPER2] Computed: B={B}, T_true={T_true}, T_expanded={T_expanded}, H={H}, K={K}, V={V}")

    # Initialize outputs
    du_final = torch.zeros_like(du_direct) 

    gated_q = q * g.unsqueeze(-1).exp()
    gated_w = w * g_expanded.unsqueeze(-1).exp()
    
    # Use q and do directly instead of expanding - more efficient approach per PDF
    
    # Process chunks sequentially from T_true to 1 (backward pass) - iterate over T_true as per PDF
    BT_expanded = 64 
    BT_true = 64  # Chunk size for true time dimension 
    assert(BT_expanded == BT_true) 
    num_chunks = math.ceil(T_true / BT_true)

    ds_final = torch.zeros(B, num_chunks, H, K, V, device=q.device, dtype=q.dtype) 
    # prev_ds = ds_next.clone() 
    # prev_ds = torch.zeros(B, H, K, V, device=q.device, dtype=torch.float32)
    prev_ds = ds_next.clone()

    # TODO  replace so that we iterate over expanded token chunks to get closer ref implementation for kernel 
    for chunk_idx in range(num_chunks - 1, -1, -1):
        # True time chunk indices
        q_t_start = chunk_idx * BT_true
        q_t_end = min(q_t_start + BT_true, T_true)
        q_chunk_size = q_t_end - q_t_start
        
        if q_chunk_size <= 0:
            continue
        
        # Extract q and do chunks
        gated_q_chunk = gated_q[:, q_t_start:q_t_end] 
        do_chunk = do[:, q_t_start:q_t_end]     # (B, q_chunk_size, H, V)
        
        # Map to corresponding expanded indices
        # expanded_start to expanded_end corresponds to BT * num_householder expanded tokens
        expanded_start = q_t_start * num_householder
        expanded_end = min(q_t_end * num_householder, T_expanded)
        expanded_size = expanded_end - expanded_start
        
        chunk_ds = torch.zeros(B, H, K, V, device=q.device)

        
        # Step 1: Compute du_final using block multiplication for K matrices
        # Process K matrices in blocks of BT_expanded
        num_k_blocks = math.ceil(expanded_size / BT_expanded)
        
        for k_block_idx in range(num_k_blocks):
            k_t_start = expanded_start + k_block_idx * BT_expanded
            k_t_end = min(k_t_start + BT_expanded, expanded_end)
            k_block_size = k_t_end - k_t_start
            
            if k_block_size <= 0:
                continue
            
            # Extract K block
            k_chunk = k[:, k_t_start:k_t_end]
            # Apply gating: exp(g_expanded[t+N-1] - g_expanded[t:t+N])
            g_last = g_expanded[:, expanded_end-1, :, None]  # (B, 1, H, 1)
            g_current = g_expanded[:, k_t_start:k_t_end, :, None]  # (B, k_block_size, H, 1)
            gated_k_chunk = k_chunk * torch.exp(g_last - g_current)
            k_block = gated_k_chunk  # (B, k_block_size, H, K)

            # TODO can divide dk and dv into blocks 
            # Block multiply K @ current_ds^T
            for b in range(B):
                for h in range(H):
                    # K_block: (k_block_size, K), prev_ds: (K, V) -> (k_block_size, V)
                    k_ds_block = torch.mm(k_block[b, :, h], prev_ds[b, h])
                    
                    # Add direct gradient contribution
                    du_final[b, k_t_start:k_t_end, h] = k_ds_block + du_direct[b, k_t_start:k_t_end, h]
        # du_final is completely computed at this point 
        
        # Step 2: Compute ∂/∂S[t]^T contributions using q and do block multiplication
        
        # Now compute \leftarrow{Q}'^T @ dO' using block matrix multiplication
        for b in range(B):
            for h in range(H):
                # Q_chunk: (q_chunk_size, K), dO_chunk: (q_chunk_size, V) -> (K, V)
                q_do_contrib = torch.mm(gated_q_chunk[b, :, h].t(), do_chunk[b, :, h])
                chunk_ds[b, h] += q_do_contrib

        # Step 3: Compute -W^T @ du_final using block operations
        # Process W in blocks of BT_expanded to match du_final blocks
        for w_block_idx in range(num_k_blocks):
            w_t_start = expanded_start + w_block_idx * BT_expanded
            w_t_end = min(w_t_start + BT_expanded, expanded_end)
            w_block_size = w_t_end - w_t_start
            
            if w_block_size <= 0:
                continue
            
            # Extract gated W and du_final blocks
            w_block = gated_w[:, w_t_start:w_t_end]  # (B, w_block_size, H, K)
            du_block = du_final[:, w_t_start:w_t_end]  # (B, w_block_size, H, V)
            
            for b in range(B):
                for h in range(H):
                    # -W^T @ du_final: (K, w_block_size) @ (w_block_size, V) -> (K, V)
                    w_du_contrib = -torch.mm(w_block[b, :, h].t(), du_block[b, :, h]) 

                    # TODO instead of loading from memory and increment could just repeat num house holder times 
                    chunk_ds[b, h] += w_du_contrib
        
        # Step 4: Add recursive gradient term from S_{t+1} to S_{t}
        for b in range(B):
            for h in range(H):
                # TODO check whether this is correct 
                chunk_ds[b, h] += prev_ds[b, h] * g_expanded[b, expanded_end - 1, h].exp()
        
        # Store prev_ds in ds_final for this chunk  
        # TODO check 
        # ds_final is (B, num_chunks, H, K, V) each state corresponds to BT * num_householder tokens 
        ds_final[:, chunk_idx, :, :, :] = chunk_ds.clone()
        
        # Update prev_ds for next iteration
        # update S_{t+1} as S_{t}
        prev_ds = chunk_ds.clone()  
    
    return du_final, ds_final


# Helper Function 3: Final gradient computation for Q[t], W[t] and direct gradient for K[t], g[t] (PDF Section 3)  
def helper_gradient_qwkg(q, k, v_new, w, g, s, ds_final, dht, do, du_final, 
                        g_expanded, num_householder):
    """
    Compute gradients for Q, W, K, g following PDF Section 3.
    Based on helper_final_gradient_s_and_u chunking pattern and gamma scaling.
    
    Args:
        q: Query tensor (B, T_true, H, K)
        k: Key tensor (B, T_expanded, H, K) 
        v_new: V_new tensor (B, T_expanded, H, V)
        w: W matrix (B, T_expanded, H, K)
        g: Gate tensor (B, T_true, H)
        ds_final: State gradients (B, T_true, H, K, V)
        do: Output gradient (B, T_true, H, V)
        du_final: Final U gradient from helper function 2
        g_expanded: Expanded gamma values (B, T_expanded, H)
        num_householder: Number of householder steps
        
    Returns:
        Tuple of (dq, dw, dk, dg) gradients
    """
    print(f"[HELPER3] helper_gradient_qwkg called")
    print(f"[HELPER3] Input shapes: q={q.shape}, k={k.shape}, v_new={v_new.shape}")
    print(f"[HELPER3] w={w.shape}, g={g.shape}, s={s.shape}")
    print(f"[HELPER3] ds_final={ds_final.shape}, dht={dht.shape}, do={do.shape}")
    print(f"[HELPER3] du_final={du_final.shape}, g_expanded={g_expanded.shape}")
    
    B, T_true, H, K = q.shape
    T_expanded = k.shape[1]
    V = do.shape[-1]
    
    print(f"[HELPER3] Computed: B={B}, T_true={T_true}, T_expanded={T_expanded}, H={H}, K={K}, V={V}")
    
    # Mark unused variable
    _ = V
    
    # Initialize gradients
    dq = torch.zeros_like(q)
    dw = torch.zeros_like(w) 
    dk = torch.zeros_like(k)
    dg = torch.zeros_like(g) 
    dg_expanded = torch.zeros(B, T_expanded, H, device=g.device, dtype=g.dtype)
    
    # Chunking parameters (following helper_final_gradient_s_and_u pattern)
    BT_true = 64  # Same block size for both true and expanded sequences
    BT_expanded = 64 
    assert(BT_true == BT_expanded)
    
    # Process chunks based on true sequence
    num_chunks = math.ceil(T_true / BT_true)

    # Apply causal mask
    expanded_causal_mask = torch.tril(torch.ones(T_expanded, T_expanded, device=q.device))
    causal_mask = expanded_causal_mask[num_householder-1::num_householder, :] # (T_true, T_expanded)
    assert(causal_mask.shape == (T_true, T_expanded)) 

    
    for chunk_idx in range(num_chunks):
        # True sequence indices
        q_t_start = chunk_idx * BT_true
        q_t_end = min(q_t_start + BT_true, T_true)
        q_chunk_size = q_t_end - q_t_start
        
        if q_chunk_size <= 0:
            continue
            
        # Expanded sequence indices (same BT size)
        expanded_start = q_t_start * num_householder
        expanded_end = min(q_t_end * num_householder, T_expanded)
        expanded_size = expanded_end - expanded_start
        
        # Extract chunks
        q_chunk = q[:, q_t_start:q_t_end]  # (B, q_chunk_size, H, K)
        do_chunk = do[:, q_t_start:q_t_end]  # (B, q_chunk_size, H, V)

        du_chunk = du_final[:, expanded_start:expanded_end]  # (B, expanded_chunk_size, H, V)
        k_chunk = k[:, expanded_start:expanded_end]  # (B, expanded_chunk_size, H, K)
        w_chunk = w[:, expanded_start:expanded_end]
        
        # Extract state gradients for this chunk
        # s_next is not needed 
        s_chunk = s[:, chunk_idx]  # s is (B, num_chunks, H, K, V)
        gated_s_chunk = s_chunk * g_expanded[:, expanded_end-1, :, None, None].exp()
        
        # Mark unused variables to avoid warnings
        _ = k_chunk, w_chunk

        # ds_chunk_next can be dht 
        ds_chunk = ds_final[:, chunk_idx]  # (B, H, K, V) 
        if chunk_idx + 1 < num_chunks:
            ds_chunk_next = ds_final[:, chunk_idx + 1]  # (B, H, K, V)
        else: 
            # ds_chunk_next is the next state gradient 
            ds_chunk_next = dht 

        # TODO need to also account for b_dg_last of b dh 
        # add to dg_expanded
        # S[t] = 1/exp(g[t+N-1]) * \arrow{S}[t]
        # dL/d g[t+N-1] = dL/dS[t] * \arrow{S}[t] * (- 1/exp(g[t+N-1]))  
        # (B,H,K,V) -> (B,H) by summing dim 2 and 3 
        dg_expanded[:, expanded_end-1, :] += (ds_chunk * gated_s_chunk * - (1/g_expanded[:, expanded_end-1, :, None].exp())).sum(dim=(2,3))
        
        # Compute dQ_arrow following PDF Section 3
        # Term 1: (∂/∂O @ S[t]) ⊙ γ' - using ds_final as state gradients
        # Term 2: (∂/∂O × A^T ⊙ M) @ K but mulytiply gates for dq_arrow 
        for b in range(B):
            for h in range(H):
                # Term 1: do @ ds_chunk (state gradient contribution)
                # Direct matrix multiplication: do_chunk @ ds_chunk^T -> (q_chunk_size, K)
                # do_chunk: (q_chunk_size, V), ds_chunk: (K, V) -> ds_chunk^T: (V, K)
                dq_arrow= torch.mm(do_chunk[b, :, h], ds_chunk[b, h].t())  # (q_chunk_size, K)
            
                # dq_contrib = dq_contrib * causal_mask_chunk.unsqueeze(-1)
                # Term 2: Batched attention gradient over num_householder iterations
                # Following PDF Section 3: batch v_new computation in blocks
                num_v_blocks = math.ceil(expanded_size / BT_expanded)
        
                for v_new_block_idx in range(num_v_blocks):
                    v_new_t_start = expanded_start + v_new_block_idx * BT_expanded
                    v_new_t_end = min(v_new_t_start + BT_expanded, expanded_end)
                    v_new_block_size = v_new_t_end - v_new_t_start
                    
                    # causal mask 
                    # block multiplication of BT \times dv and dv \times BT and then masking by BT \times BT 
                    # index used for q_chunk and do_chunk is the index that should be used for the causal mask 
                    # O_expanded[num householder] corresponds to M_expanded[num householder]
                    causal_mask_chunk = causal_mask[q_t_start:q_t_end, v_new_t_start:v_new_t_end] 
                    
                    if v_new_block_size <= 0:
                        continue
                    
                    # Extract v_new block: (B, v_new_block_size, H, V)
                    v_new_block = v_new[:, v_new_t_start:v_new_t_end] 
                    
                    # computing A = dO * dvnew^T \cdot M 
                    dA = torch.mm(do_chunk[b, :, h], v_new_block[b, :, h].t()) * causal_mask_chunk
                    # block sum with BT \times BT and BT \times dk blocks multiplied by exp(-g[t:t+N]) (to compute dq_arrow)
                    dq_arrow += (dA @ k[b, v_new_t_start:v_new_t_end, h]) * torch.exp(-g[b, q_t_start:q_t_end, h])[:, None]
                
                dq[b, q_t_start:q_t_end, h] = dq_arrow
        
        # Compute dW following PDF: ∂/∂\overrightrrow{W} = -(∂/∂(U - WS^T)) × S^T 
        for b in range(B):
            for h in range(H):
                # Block matrix multiply du_final with state gradients
                # du_chunk: (expanded_chunk_size, V), ds_chunk: (K, V)
                # ds_chunk^T: (V, K) -> result: (expanded_chunk_size, K)
                # TODO need to batch and parallelize 
                dw_arrow = -torch.mm(du_chunk[b, :, h], ds_chunk[b, h].t())  # (expanded_chunk_size, K)
                
                dw[b, expanded_start:expanded_end, h] += dw_arrow
        
        # Compute d\overrightarrow{K} following PDF Section 3  
        # Term 1: A × ∂/∂S[t+1] (using ds_final)
        # Term 2: Attention gradient term and computing d\overrightarrow{K} from dK
        for b in range(B):
            for h in range(H):
                dk_arrow = torch.zeros(expanded_end - expanded_start, K, device=q.device)
                
                # Term 1: (Ũ - WS^T) × ∂/∂S[t+1] 
                # du_chunk represents (Ũ - WS^T): (expanded_chunk_size, V)
                # ds_chunk represents next ds ∂/∂S[t+1]: (K, V) if it exists and otherwise dht (last state gradient provided to bwd)
                # TODO: need to separate into block multiplication
                term1_contrib = torch.mm(du_chunk[b, :, h], ds_chunk_next[b, h].t())  # (expanded_chunk_size, K)
                
                dk_arrow += term1_contrib

                # TODO replace BT_expanded with BT_true 

                # Term 2: (∂/∂O × ((Ũ - WS^T)^T ⊙ M))^T × Q' and multiplied by e^(gamma_[t:t+N] - gamma_[t+N-1])
                num_v_blocks = math.ceil(expanded_size / BT_expanded)
                
                for v_new_block_idx in range(num_v_blocks):
                    v_new_t_start = expanded_start + v_new_block_idx * BT_expanded
                    v_new_t_end = min(v_new_t_start + BT_expanded, expanded_end)
                    v_new_block_size = v_new_t_end - v_new_t_start
                    
                    # causal mask for this block (M matrix from screenshot)
                    causal_mask_chunk = causal_mask[q_t_start:q_t_end, v_new_t_start:v_new_t_end] 
                    
                    if v_new_block_size <= 0:
                        continue
                    
                    # Extract A matrix block:  (Ũ - WS^T)
                    v_new_block = v_new[:, v_new_t_start:v_new_t_end]  
                    v_new_block_T = v_new_block[b, :, h].t()  # v_new^T: (dV, v_new_block_size) 

                    # computing A = dO * dvnew^T \cdot M as blocks (BT \times BV @ BV \times BT -> BT \times BT)
                    dA = torch.mm(do_chunk[b, :, h], v_new_block_T.t()) * causal_mask_chunk  # (q_chunk_size, v_new_block_size)
                    
                    # Final: (dA)^T × Q' -> (v_new_block_size, K)
                    term2_block = torch.mm(dA.t(), q_chunk[b, :, h])  # (BT_expanded, K)
                    
                    # computed blocks and so store as blocks of dk_arrow
                    # store in appropriate index of dk_arrow 
                    # and divide the gated term to compute derivative of dk_arrow from dk 
                    # TODO check if correct
                    dk_arrow[v_new_t_start-expanded_start:v_new_t_end-expanded_start] += term2_block * (g_expanded[b, v_new_t_start:v_new_t_end, h] - g_expanded[b, expanded_end - 1, h]).exp()[:, None]
                
                dk[b, expanded_start:expanded_end, h] = dk_arrow

    # TODO double check the dg implementation 

    # convert the gradients of the overrightarrow{q}, overrightarrow{w}, overrightarrow{k} to gradients of 
    # q, w, k 
    for chunk_idx in range(num_chunks):
        # converting from gated output derivatives to input derivatives by chain rule 
        # doing chain rule on q_rightarrow [i, :] = q [i, :] * exp(g) * scale 
        # dL / dq * scale = dL / df(g) so we do dL / df(g) * exp(g) 

        q_t_start = chunk_idx * BT_true
        q_t_end = min(q_t_start + BT_true, T_true)

        expanded_start = q_t_start * num_householder
        expanded_end = min(q_t_end * num_householder, T_expanded)
        expanded_size = expanded_end - expanded_start

        num_blocks = math.ceil(expanded_size / BT_expanded) 

        for b in range(B):
            for h in range(H):
                # Compute dg first before modifying dq
                # add over dim = 1 or add over rows because the ith row of q is scaled by a fixed g[i], so we need to sum over gradients of g[i] over each row 
                dg[b, q_t_start:q_t_end, h] += (dq[b, q_t_start:q_t_end, h] * q[b, q_t_start:q_t_end, h]).sum(dim=1) * torch.exp(g[b, q_t_start:q_t_end, h]) # reduce the second dimension or sum each rows 
                dq[b, q_t_start:q_t_end, h] = dq[b, q_t_start:q_t_end, h] * torch.exp(g[b, q_t_start:q_t_end, h])[:, None] 
                
                for block_idx in range(num_blocks):
                    block_start = expanded_start + block_idx * BT_expanded
                    block_end = min(block_start + BT_expanded, expanded_end)
                    block_size = block_end - block_start

                    if block_size <= 0:
                        continue

                g_expanded_current_block = g_expanded[b, block_start:block_end, h]
                g_expanded_last = g_expanded[b, expanded_end - 1, h]

                dg_expanded[b, block_start:block_end, h] += (dw[b, block_start:block_end, h] * w[b, block_start:block_end, h]).sum(dim=1) * g_expanded_current_block.exp()
                dw[b, block_start:block_end, h] = dw[b, block_start:block_end, h] * torch.exp(g_expanded_current_block)[:, None]

                # check gradient of last dg element 
                # expanded_end does not have to be q_t_end * num_householder but it corresponds to last expanded otken of 
                # the expanded chunk 
                if block_end > 0:
                    dg_expanded[b, block_end-1, h] += ((dk[b, block_start:block_end, h] * k[b, block_start:block_end, h]).sum(dim=1) * torch.exp(-g_expanded_current_block + g_expanded_last)).sum()
                dg_expanded[b, block_start:block_end, h] += (dk[b, block_start:block_end, h] * k[b, block_start:block_end, h]).sum(dim=1) * - torch.exp(-g_expanded_current_block + g_expanded_last)

                # multiply the gamma term of e^(gamma_[t+N-1] - gamma[t:t+N])
                dk[b, block_start:block_end, h] = dk[b, block_start:block_end, h] * torch.exp(-g_expanded_current_block + g_expanded_last)[:, None]

    # add 0s to dg to match the shape of dg_expanded correctly 
    # add index 0, index num householder and so on 
    dg_expanded[:, ::num_householder, :] += dg 

    return dq, dw, dk, dg_expanded


# Helper Function 4: Hidden gradient computation for K[t], g, V[t], β[t] (PDF Section 4)
def helper_hidden_gradient_kvgb(k, g_expanded, v, beta, dk_direct, dg_expanded_direct, dw, du, A):
    """
    Compute hidden gradients for K, g, V, β following PDF Section 4.
    Calls prepare_wy_repr_bwd and accumulates gradients as in chunk_gated_delta_rule_bwd.
    
    Args:
        k: Key tensor (B, T_expanded, H, K)
        g: Gate tensor (B, T_true, H)  
        v: Value tensor (B, T_expanded, H, V)
        beta: Beta tensor (B, T_expanded, H)
        dk_direct: Direct gradient of K from helper function 3
        dg_direct: Direct gradient of g from helper function 3
        dw: Gradient of W matrix (B, T_expanded, H, K)
        du: Gradient of U matrix (B, T_expanded, H, V) 
        A: A matrix from WY representation
        num_householder: Number of householder steps
        
    Returns:
        Tuple of (dk_final, dg_final, dv_final, dbeta_final) gradients
    """
    print(f"[HELPER4] helper_hidden_gradient_kvgb called")
    print(f"[HELPER4] Input shapes: k={k.shape}, g_expanded={g_expanded.shape}")
    print(f"[HELPER4] v={v.shape}, beta={beta.shape}")
    print(f"[HELPER4] dk_direct={dk_direct.shape}, dg_expanded_direct={dg_expanded_direct.shape}")
    print(f"[HELPER4] dw={dw.shape}, du={du.shape}, A={A.shape}")
    
    from fla.ops.gated_delta_rule.wy_fast import prepare_wy_repr_bwd
    
    # Call WY representation backward - following chunk_gated_delta_rule_bwd pattern
    dk2, dv_final, dbeta_final, dg2 = prepare_wy_repr_bwd(
        k=k, v=v, g=g_expanded, beta=beta, A=A, dw=dw, du=du, cu_seqlens=None
    )
    
    # Accumulate gradients: dk.add_(dk2) and dg.add_(dg2) from chunk.py:145-146
    dk_final = dk_direct.clone() 
    dk_final += dk2
    
    # Reduce g gradient from expanded back to true dimensions
    dg_final = dg_expanded_direct.clone() 
    dg_final += dg2

    return dk_final, dg_final, dv_final, dbeta_final


def naive_torch_delta_product_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: Optional[torch.Tensor],
    beta: torch.Tensor,
    scale: float,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    num_householder: int = 1,
    do: Optional[torch.Tensor] = None,
    dht: Optional[torch.Tensor] = None,
) -> tuple:
    """
    Naive backward function that matches the test interface.
    Returns gradients in the order expected by the test: (dq, dk, dv, dg, dbeta, dh0)
    """
    dq, dk_final, dv_final, dg_final, dbeta_final, ds0 = gated_naive_torch_delta_product_bwd(
        q, k, v, g, beta, scale, initial_state, output_final_state, 
        num_householder, do, dht
    )
    
    # Convert dg_final from expanded format back to true format for the test
    dg_true = dg_final[:, ::num_householder, :]
    
    return dq, dk_final, dv_final, dg_true, dbeta_final, ds0


def gated_naive_torch_delta_product_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: Optional[torch.Tensor],
    beta: torch.Tensor,
    scale: float,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    num_householder: int = 1,
    do: Optional[torch.Tensor] = None,
    dht: Optional[torch.Tensor] = None,
) -> tuple:
    """
    Main delta product backward function using the 4 helper functions.
    """
    print(f"[MAIN_BWD] gated_naive_torch_delta_product_bwd called")
    print(f"[MAIN_BWD] Input shapes: q={q.shape}, k={k.shape}, v={v.shape}")
    print(f"[MAIN_BWD] g={g.shape if g is not None else None}, beta={beta.shape}")
    print(f"[MAIN_BWD] do={do.shape if do is not None else None}, dht={dht.shape if dht is not None else None}")
    print(f"[MAIN_BWD] scale={scale}, num_householder={num_householder}")
    
    B, T_true, H, K = q.shape
    T_expanded = k.shape[1]
    V = v.shape[-1]
    
    print(f"[MAIN_BWD] Computed: B={B}, T_true={T_true}, T_expanded={T_expanded}, H={H}, K={K}, V={V}")
    
    # Use scale and output_final_state if needed
    _ = scale, output_final_state, T_true, K, V
    
    # Setup interleaved g for gamma computation
    assert(g is not None)
    g_expanded = torch.zeros(B, T_expanded, H, device=g.device, dtype=g.dtype)
    g_expanded[:, ::num_householder, :] = g

    g = chunk_local_cumsum(g, chunk_size=64)
    g_expanded = chunk_local_cumsum(g_expanded, chunk_size=64)

    # Compute A matrix first
    A = chunk_scaled_dot_kkt_fwd(
        k=k,
        g=g_expanded,
        beta=beta,
        output_dtype=g.dtype
    )
    A = solve_tril(
        A=A,
        output_dtype=k.dtype
    )

    if g_expanded is not None: 
        w, u = gdn_recompute_w_u_fwd(
            k=k,
            v=v,
            beta=beta,
            A=A,
            g=g_expanded,
        )
    else:
        w, u = dn_recompute_w_u_fwd(
            k=k,
            v=v,
            beta=beta,
            A=A,
        )
    
    h, v_new, _ = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g_expanded,
        initial_state=initial_state,
        output_final_state=False,
    )
    
    # Step 1: Compute direct gradient using Helper Function 1
    du_direct = helper_direct_gradient_u_minus_ws(
        q, k, do, num_householder
    )
    
    # Step 2: Compute final gradients using Helper Function 2
    du_final, ds_final = helper_final_gradient_s_and_u(
        q, k, w, du_direct, do, dht, g, g_expanded, num_householder
    )
    
    # Step 3: Compute Q, W, K, G gradients using Helper Function 3
    # TODO check how scale is used 
    dq, dw, dk_direct, dg_expanded_direct = helper_gradient_qwkg(
        q, k, v_new, w, g, h, ds_final, dht, do, du_final, g_expanded, num_householder
    )

    # g is gradients over expanded tokens 
    
    # Step 4: Compute hidden gradients using Helper Function 4 and add to direct gradients of dk and dg 
    dk_final, dg_final, dv_final, dbeta_final = helper_hidden_gradient_kvgb(
        k, g_expanded, v, beta, dk_direct, dg_expanded_direct, dw, du_final, A
    )

    # assert dg_final.dtype == torch.float32, "dg should be fp32"
    dg_final = chunk_local_cumsum(dg_final, chunk_size=64, reverse=True)

    ds0 = ds_final[:, 0]
    # only the first hidden state gradient needs to be returned ds0 for computing gradient of previous chunk 
    
    # Return in the expected order: (dq, dk, dv, dg, dbeta, dh0)
    return dq, dk_final, dv_final, dg_final, dbeta_final, ds0 





"""
modify existing cuda implementation 
"""

# def check_shared_mem(device_type=None, device_index=None):
#     """Check if device supports larger shared memory."""
#     if device_type == 'hopper':
#         return True
#     return torch.cuda.get_device_capability(device_index or 0)[0] >= 8


# def torch_chunk_gated_delta_product_bwd_dv_local(
#     q,
#     k,
#     g,
#     do,
#     scale,
#     cu_seqlens,
#     num_householder=1,
#     chunk_size=64
# ):
#     """
#     Compute direct gradient of v_new = U[t] - W[t]S[t]^T
#     Following PDF Section 1: Computing the direct gradient
#     """
#     B, T_expanded, H, K = k.shape
#     V = do.shape[-1]
#     T_true = q.shape[1]
    
#     # Grid configuration
#     BT = min(chunk_size, max(16, 2**int(math.log2(T_expanded)+0.5)))
#     chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    
#     # Memory tiling configuration
#     CONST_TILING = 128 if check_shared_mem('hopper') else 64 if check_shared_mem() else 32
#     BK = min(2**int(math.log2(K)+0.5), CONST_TILING)
#     BV = min(2**int(math.log2(V)+0.5), CONST_TILING)
    
#     NT = math.ceil(T_expanded / BT) if cu_seqlens is None else len(chunk_indices) // 2
    
#     dv = torch.zeros_like(k[:, :, :, :V])  # Shape: (B, T_expanded, H, V)
#     grid = (NT, B * H)
    
#     # Flatten arrays for pointer-like access
#     q_flat = q.contiguous().view(-1)
#     k_flat = k.contiguous().view(-1)
#     g_flat = g.contiguous().view(-1) if g is not None else None
#     do_flat = do.contiguous().view(-1)
#     dv_flat = dv.contiguous().view(-1)
#     cu_seqlens_flat = cu_seqlens.contiguous().view(-1) if cu_seqlens is not None else None
#     chunk_indices_flat = chunk_indices.contiguous().view(-1) if chunk_indices is not None else None
    
#     # Execute kernel grid
#     for i_t in range(grid[0]):
#         for i_bh in range(grid[1]):
#             torch_mock_kernel_chunk_gated_delta_product_bwd_dv_local(
#                 i_t, i_bh,
#                 q_flat, k_flat, g_flat, do_flat, dv_flat,
#                 cu_seqlens_flat, chunk_indices_flat,
#                 scale, T_expanded, T_true, H, K, V, BT, BK, BV, num_householder
#             )
    
#     return dv.view(B, T_expanded, H, V)


# def torch_mock_kernel_chunk_gated_delta_product_bwd_dv_local(
#     KERNEL_GRID_I, KERNEL_GRID_J,
#     q, k, g, do, dv,
#     cu_seqlens, chunk_indices,
#     scale, T_expanded, T_true, H, K, V, BT, BK, BV, num_householder
# ):
#     """
#     Mock kernel implementing direct gradient computation following PDF equations.
#     """
#     index_token_chunk, index_batch_head = KERNEL_GRID_I, KERNEL_GRID_J
#     index_batch, index_head = index_batch_head // H, index_batch_head % H
    
#     IS_VARLEN = cu_seqlens is not None
    
#     if IS_VARLEN:
#         i_n, i_t = chunk_indices[index_token_chunk * 2], chunk_indices[index_token_chunk * 2 + 1]
#         bos, eos = cu_seqlens[i_n], cu_seqlens[i_n + 1]
#         T_seq = eos - bos
#         bos_actual = bos // num_householder
#         eos_actual = eos // num_householder
#     else:
#         bos, eos = index_batch * T_expanded, (index_batch + 1) * T_expanded
#         T_seq = T_expanded
#         bos_actual = bos // num_householder
#         eos_actual = eos // num_householder
    
#     # Calculate offsets for flattened arrays
#     q_offset = (bos_actual * H + index_head) * K
#     do_offset = (bos_actual * H + index_head) * V
#     k_offset = (bos * H + index_head) * K
#     dv_offset = (bos * H + index_head) * V
    
#     # Chunk boundaries
#     t_start = index_token_chunk * BT
#     t_end = min(t_start + BT, T_seq)
    
#     if t_start >= T_seq:
#         return
    
#     chunk_length = t_end - t_start
    
#     # Process in blocks following PDF equations
#     # dv[t:t+N] = (Q'[t:t+N]K[t:t+N]^T ⊙ M)^T · dO'[t:t+N]
    
#     for i_k in range(math.ceil(K / BK)):
#         k_start = i_k * BK
#         k_end = min(k_start + BK, K)
#         k_size = k_end - k_start
        
#         # Load Q and K blocks
#         q_block = torch.zeros(chunk_length, k_size, dtype=torch.float32)
#         k_block = torch.zeros(chunk_length, k_size, dtype=torch.float32)
        
#         # Fill blocks with actual data (only at householder positions for Q)
#         for t_rel in range(chunk_length):
#             t_abs = t_start + t_rel
            
#             # Load K values
#             for k_idx in range(k_size):
#                 k_block[t_rel, k_idx] = k[k_offset + t_abs * K + k_start + k_idx]
            
#             # Load Q values (only at householder positions)
#             if t_abs % num_householder == (num_householder - 1):
#                 t_true_idx = t_abs // num_householder
#                 if t_true_idx < T_true:
#                     for k_idx in range(k_size):
#                         q_block[t_rel, k_idx] = q[q_offset + t_true_idx * K + k_start + k_idx]
        
#         # Compute QK^T with causal masking
#         qk_block = torch.mm(q_block, k_block.t())  # (chunk_length, chunk_length)
        
#         # Apply causal mask M
#         causal_mask = torch.tril(torch.ones(chunk_length, chunk_length))
#         masked_qk = qk_block * causal_mask
        
#         # Process V blocks
#         for i_v in range(math.ceil(V / BV)):
#             v_start = i_v * BV
#             v_end = min(v_start + BV, V)
#             v_size = v_end - v_start
            
#             # Load dO block (only at householder positions)
#             do_block = torch.zeros(chunk_length, v_size, dtype=torch.float32)
#             for t_rel in range(chunk_length):
#                 t_abs = t_start + t_rel
#                 if t_abs % num_householder == (num_householder - 1):
#                     t_true_idx = t_abs // num_householder
#                     if t_true_idx < T_true:
#                         for v_idx in range(v_size):
#                             do_block[t_rel, v_idx] = do[do_offset + t_true_idx * V + v_start + v_idx]
            
#             # Compute A^T @ dO where A = QK^T ⊙ M
#             dv_block = torch.mm(masked_qk.t(), do_block)  # (chunk_length, v_size)
            
#             # Store results
#             for t_rel in range(chunk_length):
#                 t_abs = t_start + t_rel
#                 for v_idx in range(v_size):
#                     dv[dv_offset + t_abs * V + v_start + v_idx] = dv_block[t_rel, v_idx]


# def torch_chunk_gated_delta_product_bwd_dhu(
#     q, k, w, g, h0, dht, do, dv, scale, cu_seqlens, num_householder, chunk_size=64
# ):
#     """
#     Compute gradients through hidden states following PDF Section 2.
#     """
#     B, T_expanded, H, K = k.shape
#     V = do.shape[-1] if do.dim() > 0 else dv.shape[-1]
#     T_true = q.shape[1]
    
#     # Grid configuration
#     BT = chunk_size
#     chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
#     chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT) if cu_seqlens is not None else None
    
#     if cu_seqlens is None:
#         N, NT = B, math.ceil(T_expanded / BT)
#     else:
#         N, NT = len(cu_seqlens) - 1, len(chunk_indices) // 2
    
#     BV = min(2**int(math.log2(V)+0.5), 64)
#     NV = math.ceil(V / BV)
    
#     # Initialize outputs
#     dh = torch.zeros(B, NT, H, K, V, device=k.device)
#     dh0 = torch.zeros_like(h0) if h0 is not None else None
#     du = torch.zeros_like(dv)
    
#     grid = (NV, N * H)
    
#     # Flatten arrays
#     q_flat = q.contiguous().view(-1)
#     k_flat = k.contiguous().view(-1)
#     w_flat = w.contiguous().view(-1) if w is not None else None
#     g_flat = g.contiguous().view(-1) if g is not None else None
#     do_flat = do.contiguous().view(-1)
#     dv_flat = dv.contiguous().view(-1)
#     dh_flat = dh.contiguous().view(-1)
#     du_flat = du.contiguous().view(-1)
#     h0_flat = h0.contiguous().view(-1) if h0 is not None else None
#     dht_flat = dht.contiguous().view(-1) if dht is not None else None
#     dh0_flat = dh0.contiguous().view(-1) if dh0 is not None else None
#     cu_seqlens_flat = cu_seqlens.contiguous().view(-1) if cu_seqlens is not None else None
#     chunk_offsets_flat = chunk_offsets.contiguous().view(-1) if chunk_offsets is not None else None
    
#     # Execute kernel grid
#     for i_v in range(grid[0]):
#         for i_nh in range(grid[1]):
#             torch_mock_kernel_chunk_gated_delta_product_bwd_dhu(
#                 i_v, i_nh,
#                 q_flat, k_flat, w_flat, g_flat, do_flat, dv_flat,
#                 dh_flat, du_flat, h0_flat, dht_flat, dh0_flat,
#                 cu_seqlens_flat, chunk_offsets_flat,
#                 scale, T_expanded, T_true, H, K, V, BT, BV, NT, num_householder
#             )
    
#     return dh.view(B, NT, H, K, V), dh0, du.view_as(dv)


# def torch_mock_kernel_chunk_gated_delta_product_bwd_dhu(
#     KERNEL_GRID_I, KERNEL_GRID_J,
#     q, k, w, g, do, dv, dh, du, h0, dht, dh0,
#     cu_seqlens, chunk_offsets, scale,
#     T_expanded, T_true, H, K, V, BT, BV, NT, num_householder
# ):
#     """
#     Mock kernel for hidden state gradients following PDF Section 2 formulas.
#     """
#     index_v_chunk, index_n_head = KERNEL_GRID_I, KERNEL_GRID_J
#     index_n, index_head = index_n_head // H, index_n_head % H
    
#     v_start = index_v_chunk * BV
#     v_end = min(v_start + BV, V)
#     v_size = v_end - v_start
    
#     if v_start >= V:
#         return
    
#     IS_VARLEN = cu_seqlens is not None
    
#     if IS_VARLEN:
#         seq_start, seq_end = cu_seqlens[index_n], cu_seqlens[index_n + 1]
#         seq_len = seq_end - seq_start
#         seq_start_actual = seq_start // num_householder
#         seq_end_actual = seq_end // num_householder
#     else:
#         seq_start, seq_end = index_n * T_expanded, (index_n + 1) * T_expanded
#         seq_len = T_expanded
#         seq_start_actual = seq_start // num_householder
#         seq_end_actual = seq_end // num_householder
    
#     # Calculate offsets
#     q_offset = (seq_start_actual * H + index_head) * K
#     k_offset = (seq_start * H + index_head) * K
#     do_offset = (seq_start_actual * H + index_head) * V
#     dv_offset = (seq_start * H + index_head) * V
#     dh_offset = (index_n * NT * H + index_head) * K * V
#     du_offset = dv_offset
    
#     # Initialize current hidden state gradient (from final state)
#     current_dh = torch.zeros(K, v_size, dtype=torch.float32)
#     if dht is not None:
#         for k_idx in range(K):
#             for v_idx in range(v_size):
#                 current_dh[k_idx, v_idx] = dht[index_n * H * K * V + index_head * K * V + k_idx * V + v_start + v_idx]
    
#     # Backward pass through time chunks
#     num_chunks = math.ceil(seq_len / BT)
#     for chunk_idx in range(num_chunks - 1, -1, -1):
#         t_start = chunk_idx * BT
#         t_end = min(t_start + BT, seq_len)
#         chunk_size = t_end - t_start
        
#         if chunk_size <= 0:
#             continue
        
#         # Load chunks
#         q_chunk = torch.zeros(chunk_size, K, dtype=torch.float32)
#         k_chunk = torch.zeros(chunk_size, K, dtype=torch.float32)
#         do_chunk = torch.zeros(chunk_size, v_size, dtype=torch.float32)
#         dv_chunk = torch.zeros(chunk_size, v_size, dtype=torch.float32)
        
#         # Fill chunks
#         for t_rel in range(chunk_size):
#             t_abs = t_start + t_rel
            
#             # Load K values
#             for k_idx in range(K):
#                 k_chunk[t_rel, k_idx] = k[k_offset + t_abs * K + k_idx]
            
#             # Load Q and dO values (only at householder positions)
#             if t_abs % num_householder == (num_householder - 1):
#                 t_true_idx = t_abs // num_householder
#                 if t_true_idx < (seq_end_actual - seq_start_actual):
#                     for k_idx in range(K):
#                         q_chunk[t_rel, k_idx] = q[q_offset + t_true_idx * K + k_idx]
#                     for v_idx in range(v_size):
#                         do_chunk[t_rel, v_idx] = do[do_offset + t_true_idx * V + v_start + v_idx]
            
#             # Load dV values
#             for v_idx in range(v_size):
#                 dv_chunk[t_rel, v_idx] = dv[dv_offset + t_abs * V + v_start + v_idx]
        
#         # Compute gradients following PDF formulas:
#         # dS[t]^T = Q'[t:t+N]^T @ dO'[t:t+N] + dS[t+1]^T * gamma'[t] - W[t:t+N]^T @ d(U[t:t+N] - W[t:t+N]S[t]^T)
        
#         # Direct contribution: Q^T @ dO
#         direct_contrib = torch.mm(q_chunk.t(), do_chunk)  # (K, v_size)
        
#         # Add hidden state propagation
#         chunk_dh = direct_contrib + current_dh
        
#         # Store hidden state gradients
#         for k_idx in range(K):
#             for v_idx in range(v_size):
#                 dh[dh_offset + chunk_idx * H * K * V + k_idx * V + v_start + v_idx] = chunk_dh[k_idx, v_idx]
        
#         # Update current_dh for next iteration and compute du
#         current_dh = chunk_dh
        
#         # Compute du from dv (simplified)
#         for t_rel in range(chunk_size):
#             t_abs = t_start + t_rel
#             for v_idx in range(v_size):
#                 du[du_offset + t_abs * V + v_start + v_idx] = dv_chunk[t_rel, v_idx]
    
#     # Set initial state gradient
#     if dh0 is not None:
#         for k_idx in range(K):
#             for v_idx in range(v_size):
#                 dh0[index_n * H * K * V + index_head * K * V + k_idx * V + v_start + v_idx] = current_dh[k_idx, v_idx]


# def torch_chunk_gated_delta_product_bwd_dqkwg(
#     q, k, v_new, w, g, h, dv, do, dh, scale, cu_seqlens, num_householder, chunk_size=64
# ):
#     """
#     Compute gradients for Q, K, W, and G following PDF Section 3.
#     """
#     B, T_true, H, K = q.shape
#     T_expanded = k.shape[1]
#     V = v_new.shape[-1] if v_new is not None else do.shape[-1]
    
#     # Grid configuration
#     BT = min(chunk_size, max(16, 2**int(math.log2(T_expanded)+0.5)))
#     chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
#     NT = math.ceil(T_expanded / BT) if cu_seqlens is None else len(chunk_indices) // 2
    
#     CONST_TILING = 64 if check_shared_mem() else 32
#     BK = min(2**int(math.log2(K)+0.5), CONST_TILING)
#     BV = min(2**int(math.log2(V)+0.5), CONST_TILING)
#     NK = math.ceil(K / BK)
    
#     # Initialize gradients
#     dq = torch.zeros_like(q)
#     dk = torch.zeros_like(k)
#     dg = torch.zeros(NK, *g.shape, dtype=torch.float32, device=g.device) if g is not None else None
#     dw = torch.zeros_like(w) if w is not None else None
    
#     grid = (NK, NT, B * H)
    
#     # Flatten arrays
#     arrays_to_flatten = [q, k, v_new, do, dq, dk]
#     if w is not None:
#         arrays_to_flatten.extend([w, dw])
#     if g is not None:
#         arrays_to_flatten.extend([g, dg])
#     if h is not None:
#         arrays_to_flatten.append(h)
#     if dv is not None:
#         arrays_to_flatten.append(dv)
#     if dh is not None:
#         arrays_to_flatten.append(dh)
    
#     flattened = [arr.contiguous().view(-1) if arr is not None else None for arr in arrays_to_flatten]
    
#     q_flat, k_flat, v_new_flat, do_flat, dq_flat, dk_flat = flattened[:6]
#     idx = 6
#     w_flat = flattened[idx] if w is not None else None
#     if w is not None:
#         idx += 1
#     dw_flat = flattened[idx] if w is not None else None
#     if w is not None:
#         idx += 1
#     g_flat = flattened[idx] if g is not None else None
#     if g is not None:
#         idx += 1
#     dg_flat = flattened[idx] if g is not None else None
#     if g is not None:
#         idx += 1
#     h_flat = flattened[idx] if h is not None else None
#     if h is not None:
#         idx += 1
#     dv_flat = flattened[idx] if dv is not None else None
#     if dv is not None:
#         idx += 1
#     dh_flat = flattened[idx] if dh is not None else None
    
#     cu_seqlens_flat = cu_seqlens.contiguous().view(-1) if cu_seqlens is not None else None
#     chunk_indices_flat = chunk_indices.contiguous().view(-1) if chunk_indices is not None else None
    
#     # Execute kernel grid
#     for i_k in range(grid[0]):
#         for i_t in range(grid[1]):
#             for i_bh in range(grid[2]):
#                 torch_mock_kernel_chunk_gated_delta_product_bwd_dqkwg(
#                     i_k, i_t, i_bh,
#                     q_flat, k_flat, v_new_flat, w_flat, g_flat, h_flat,
#                     dv_flat, do_flat, dh_flat, dq_flat, dk_flat, dw_flat, dg_flat,
#                     cu_seqlens_flat, chunk_indices_flat, scale,
#                     T_expanded, T_true, H, K, V, BT, BK, BV, num_householder
#                 )
    
#     # Handle gating gradients reduction
#     if g is not None and dg is not None:
#         dg_final = dg.sum(0)  # Reduce over NK dimension
#     else:
#         dg_final = None
    
#     return dq, dk, dw, dg_final


# def torch_mock_kernel_chunk_gated_delta_product_bwd_dqkwg(
#     KERNEL_GRID_I, KERNEL_GRID_J, KERNEL_GRID_K,
#     q, k, v_new, w, g, h, dv, do, dh, dq, dk, dw, dg,
#     cu_seqlens, chunk_indices, scale,
#     T_expanded, T_true, H, K, V, BT, BK, BV, num_householder
# ):
#     """
#     Mock kernel for Q, K, W, G gradients following PDF Section 3 formulas.
#     """
#     index_k_chunk, index_t_chunk, index_batch_head = KERNEL_GRID_I, KERNEL_GRID_J, KERNEL_GRID_K
#     index_batch, index_head = index_batch_head // H, index_batch_head % H
    
#     k_start = index_k_chunk * BK
#     k_end = min(k_start + BK, K)
#     k_size = k_end - k_start
    
#     if k_start >= K:
#         return
    
#     IS_VARLEN = cu_seqlens is not None
    
#     if IS_VARLEN:
#         i_n, i_t = chunk_indices[index_t_chunk * 2], chunk_indices[index_t_chunk * 2 + 1]
#         bos, eos = cu_seqlens[i_n], cu_seqlens[i_n + 1]
#         T_seq = eos - bos
#         bos_actual = bos // num_householder
#         eos_actual = eos // num_householder
#     else:
#         bos, eos = index_batch * T_expanded, (index_batch + 1) * T_expanded
#         T_seq = T_expanded
#         bos_actual = bos // num_householder
#         eos_actual = eos // num_householder
    
#     t_start = index_t_chunk * BT
#     t_end = min(t_start + BT, T_seq)
    
#     if t_start >= T_seq:
#         return
    
#     chunk_size = t_end - t_start
    
#     # Calculate offsets
#     q_offset = (bos_actual * H + index_head) * K
#     k_offset = (bos * H + index_head) * K
#     do_offset = (bos_actual * H + index_head) * V
#     dq_offset = q_offset
#     dk_offset = k_offset
    
#     # Process dQ and dK following PDF formulas
#     # dQ = (dO @ S[t]) ⊙ gamma + (dO @ A^T ⊙ M) @ K
#     # dK = first_term + second_term (direct + through hidden state)
    
#     for t_rel in range(chunk_size):
#         t_abs = t_start + t_rel
        
#         # Process dQ (only at householder positions)
#         if t_abs % num_householder == (num_householder - 1):
#             t_true_idx = t_abs // num_householder
#             if t_true_idx < (eos_actual - bos_actual):
                
#                 # Load current dO
#                 do_vec = torch.zeros(V, dtype=torch.float32)
#                 for v_idx in range(V):
#                     do_vec[v_idx] = do[do_offset + t_true_idx * V + v_idx]
                
#                 # Compute dQ contributions
#                 dq_contrib = torch.zeros(k_size, dtype=torch.float32)
                
#                 # First term: interaction with hidden state
#                 if h is not None:
#                     # Approximate S[t] from h
#                     for k_idx in range(k_size):
#                         h_sum = 0.0
#                         for v_idx in range(V):
#                             # Simplified h access - in practice this would be more complex
#                             h_sum += do_vec[v_idx]  # Simplified computation
#                         dq_contrib[k_idx] += h_sum * scale
                
#                 # Second term: masked attention
#                 if v_new is not None:
#                     for t_mask in range(t_abs + 1):  # Causal mask
#                         if t_mask < T_seq:
#                             # Load v_new and k
#                             v_vec = torch.zeros(V, dtype=torch.float32)
#                             k_vec = torch.zeros(k_size, dtype=torch.float32)
                            
#                             v_offset = (bos * H + index_head) * V + t_mask * V
#                             k_vec_offset = k_offset + t_mask * K + k_start
                            
#                             for v_idx in range(V):
#                                 v_vec[v_idx] = v_new[v_offset + v_idx]
#                             for k_idx in range(k_size):
#                                 k_vec[k_idx] = k[k_vec_offset + k_idx]
                            
#                             # Compute contribution
#                             dot_product = sum(do_vec[v_idx] * v_vec[v_idx] for v_idx in range(V))
#                             for k_idx in range(k_size):
#                                 dq_contrib[k_idx] += dot_product * k_vec[k_idx] * scale
                
#                 # Store dQ
#                 for k_idx in range(k_size):
#                     dq[dq_offset + t_true_idx * K + k_start + k_idx] += dq_contrib[k_idx]
        
#         # Process dK (all positions)
#         dk_contrib = torch.zeros(k_size, dtype=torch.float32)
        
#         # Find all query positions that attend to this key position
#         for q_true_idx in range(eos_actual - bos_actual):
#             q_expanded_idx = (q_true_idx + 1) * num_householder - 1
#             if q_expanded_idx >= t_abs:  # Causal constraint
                
#                 # Load q and do vectors
#                 q_vec = torch.zeros(k_size, dtype=torch.float32)
#                 do_vec = torch.zeros(V, dtype=torch.float32)
                
#                 for k_idx in range(k_size):
#                     q_vec[k_idx] = q[q_offset + q_true_idx * K + k_start + k_idx]
#                 for v_idx in range(V):
#                     do_vec[v_idx] = do[do_offset + q_true_idx * V + v_idx]
                
#                 if v_new is not None:
#                     # Load v_new for this position
#                     v_vec = torch.zeros(V, dtype=torch.float32)
#                     v_offset = (bos * H + index_head) * V + t_abs * V
#                     for v_idx in range(V):
#                         v_vec[v_idx] = v_new[v_offset + v_idx]
                    
#                     # Compute contribution: q @ dO @ v_new
#                     dot_product = sum(do_vec[v_idx] * v_vec[v_idx] for v_idx in range(V))
#                     for k_idx in range(k_size):
#                         dk_contrib[k_idx] += q_vec[k_idx] * dot_product * scale
        
#         # Store dK
#         for k_idx in range(k_size):
#             dk[dk_offset + t_abs * K + k_start + k_idx] += dk_contrib[k_idx]
        
#         # Process dG (if gating is used and at householder positions)
#         if g is not None and dg is not None and t_abs % num_householder == (num_householder - 1):
#             t_true_idx = t_abs // num_householder
#             if t_true_idx < (eos_actual - bos_actual) and t_true_idx > 0:
                
#                 # Gradient through gating of hidden state
#                 dg_contrib = 0.0
#                 if h is not None and dh is not None:
#                     # Simplified gradient computation
#                     for k_idx in range(k_size):
#                         for v_idx in range(V):
#                             # This is a simplified version - actual implementation would be more complex
#                             dg_contrib += 1.0  # Placeholder computation
                
#                 # Store dG with accumulation over K chunks
#                 g_offset = (index_batch * (eos_actual - bos_actual) * H + t_true_idx * H + index_head)
#                 dg[index_k_chunk * g.numel() // (NK if 'NK' in locals() else 1) + g_offset] += dg_contrib


# def torch_chunk_gated_delta_product_bwd(
#     q, k, v, g, g_interleaved, beta, A, h, v_new, do, dht, scale, cu_seqlens, initial_state, num_householder
# ):
#     """
#     Main backward pass orchestrating all gradient computations.
#     """
#     # Step 1: Compute direct gradient dv_new
#     dv_new_direct = torch_chunk_gated_delta_product_bwd_dv_local(
#         q, k, g_interleaved, do, scale, cu_seqlens, num_householder
#     )
    
#     # Step 2: Compute hidden state gradients
#     dh_computed, dh0, du = torch_chunk_gated_delta_product_bwd_dhu(
#         q, k, None, g_interleaved, initial_state, dht, do, dv_new_direct, scale, cu_seqlens, num_householder
#     )
    
#     # Step 3: Compute Q, K, W, G gradients
#     dq, dk_direct, dw, dg_direct = torch_chunk_gated_delta_product_bwd_dqkwg(
#         q, k, v_new, None, g, h, du, do, dh_computed, scale, cu_seqlens, num_householder
#     )
    
#     # Step 4: Accumulate final gradients
#     dk_final = dk_direct  # + dk_hidden_state_gradient (simplified)
#     dv_final = torch.zeros_like(v)  # Placeholder
#     dbeta = torch.zeros_like(beta)  # Placeholder
#     dg_final = dg_direct
    
#     return dq, dk_final, dv_final, dbeta, dg_final, dh0

