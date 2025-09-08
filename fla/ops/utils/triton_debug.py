# -*- coding: utf-8 -*-
"""
Triton tensor debugging utilities for saving intermediate tensors to CPU/disk.
Designed for debugging complex kernels like gated delta product.
"""

import torch
import triton
import triton.language as tl
import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime


class TritonDebugger:
    """
    Singleton debugger for saving and comparing Triton intermediate tensors.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, save_dir: str = "./triton_debug", enabled: bool = True):
        if self._initialized:
            return
        self.save_dir = Path(save_dir)
        self.enabled = enabled
        self.tensor_cache = {}
        self.comparison_results = []
        self._initialized = True
        
        if self.enabled:
            os.makedirs(self.save_dir, exist_ok=True)
            # Create subdirectories for organization
            os.makedirs(self.save_dir / "tensors", exist_ok=True)
            os.makedirs(self.save_dir / "comparisons", exist_ok=True)
            os.makedirs(self.save_dir / "metadata", exist_ok=True)
    
    def save_tensor(self, tensor: torch.Tensor, name: str, 
                   kernel_name: str = None, iteration: int = None,
                   block_id: Tuple[int, ...] = None, **metadata) -> str:
        """
        Save a tensor from Triton kernel execution.
        
        Args:
            tensor: Tensor to save (will be moved to CPU)
            name: Name of the tensor (e.g., 'b_q', 'b_k', 'b_dg_expanded')
            kernel_name: Name of the kernel (e.g., 'chunk_bwd_kernel_dqkwg')
            iteration: Loop iteration if applicable
            block_id: Block indices (i_k, i_t, i_bh)
            **metadata: Additional metadata
            
        Returns:
            Path where tensor was saved
        """
        if not self.enabled:
            return ""
        
        # Create unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        parts = [name]
        if kernel_name:
            parts.append(kernel_name)
        if iteration is not None:
            parts.append(f"iter{iteration}")
        if block_id:
            parts.append(f"block{'_'.join(map(str, block_id))}")
        
        filename = "_".join(parts) + f"_{timestamp}"
        
        # Move tensor to CPU and save
        cpu_tensor = tensor.detach().cpu() if tensor.is_cuda else tensor
        tensor_path = self.save_dir / "tensors" / f"{filename}.pt"
        torch.save(cpu_tensor, tensor_path)
        
        # Save metadata
        meta = {
            'name': name,
            'kernel_name': kernel_name,
            'iteration': iteration,
            'block_id': block_id,
            'shape': list(cpu_tensor.shape),
            'dtype': str(cpu_tensor.dtype),
            'device': str(tensor.device) if hasattr(tensor, 'device') else 'cpu',
            'timestamp': timestamp,
            'stats': {
                'min': float(cpu_tensor.min().item()) if cpu_tensor.numel() > 0 else None,
                'max': float(cpu_tensor.max().item()) if cpu_tensor.numel() > 0 else None,
                'mean': float(cpu_tensor.mean().item()) if cpu_tensor.numel() > 0 else None,
                'std': float(cpu_tensor.std().item()) if cpu_tensor.numel() > 1 else None,
                'nan_count': int(torch.isnan(cpu_tensor).sum().item()),
                'inf_count': int(torch.isinf(cpu_tensor).sum().item()),
            }
        }
        meta.update(metadata)
        
        meta_path = self.save_dir / "metadata" / f"{filename}.json"
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        
        # Cache for comparison
        cache_key = f"{kernel_name}_{name}_{iteration}_{block_id}"
        self.tensor_cache[cache_key] = cpu_tensor
        
        return str(tensor_path)
    
    def compare_tensors(self, tensor1: torch.Tensor, tensor2: torch.Tensor,
                       name1: str, name2: str, rtol: float = 1e-5, 
                       atol: float = 1e-8) -> Dict[str, Any]:
        """
        Compare two tensors and save comparison results.
        """
        if not self.enabled:
            return {}
        
        # Ensure both tensors are on CPU
        t1 = tensor1.detach().cpu() if tensor1.is_cuda else tensor1
        t2 = tensor2.detach().cpu() if tensor2.is_cuda else tensor2
        
        result = {
            'name1': name1,
            'name2': name2,
            'timestamp': datetime.now().isoformat(),
            'shapes_match': t1.shape == t2.shape,
            'shape1': list(t1.shape),
            'shape2': list(t2.shape),
        }
        
        if t1.shape != t2.shape:
            result['error'] = 'Shape mismatch'
            self.comparison_results.append(result)
            return result
        
        # Convert to float for comparison
        t1_f = t1.float()
        t2_f = t2.float()
        
        diff = t1_f - t2_f
        abs_diff = diff.abs()
        rel_diff = abs_diff / (t1_f.abs() + 1e-10)
        
        result.update({
            'allclose': torch.allclose(t1_f, t2_f, rtol=rtol, atol=atol),
            'max_abs_diff': float(abs_diff.max().item()),
            'mean_abs_diff': float(abs_diff.mean().item()),
            'max_rel_diff': float(rel_diff.max().item()),
            'rmse': float(torch.sqrt((diff ** 2).mean()).item()),
        })
        
        if not result['allclose']:
            # Find location of maximum difference
            max_idx = torch.argmax(abs_diff.flatten()).item()
            coords = np.unravel_index(max_idx, t1.shape)
            result['max_diff_location'] = coords
            result['value1_at_max'] = float(t1_f.flatten()[max_idx].item())
            result['value2_at_max'] = float(t2_f.flatten()[max_idx].item())
        
        self.comparison_results.append(result)
        
        # Save comparison
        comp_path = self.save_dir / "comparisons" / f"comp_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
        with open(comp_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        return result
    
    def print_summary(self):
        """Print summary of all comparisons."""
        if not self.enabled or not self.comparison_results:
            return
        
        print("\n" + "="*60)
        print("TRITON DEBUG SUMMARY")
        print("="*60)
        
        total = len(self.comparison_results)
        matching = sum(1 for r in self.comparison_results if r.get('allclose', False))
        
        print(f"Total comparisons: {total}")
        print(f"Matching: {matching}")
        print(f"Mismatched: {total - matching}")
        
        print("\nMismatched tensors:")
        for result in self.comparison_results:
            if not result.get('allclose', False):
                print(f"  - {result['name1']} vs {result['name2']}")
                print(f"    Max diff: {result.get('max_abs_diff', 'N/A'):.2e}")
                if 'max_diff_location' in result:
                    print(f"    At location: {result['max_diff_location']}")


# Global debugger instance
_debugger = TritonDebugger()


def debug_save_tensor(tensor: torch.Tensor, name: str, **kwargs):
    """Convenience function to save a tensor."""
    return _debugger.save_tensor(tensor, name, **kwargs)


def debug_compare(tensor1: torch.Tensor, tensor2: torch.Tensor, 
                 name1: str, name2: str, **kwargs):
    """Convenience function to compare tensors."""
    return _debugger.compare_tensors(tensor1, tensor2, name1, name2, **kwargs)


def debug_print_stats(tensor: torch.Tensor, name: str):
    """Print tensor statistics."""
    if not _debugger.enabled:
        return
    
    cpu_tensor = tensor.detach().cpu() if tensor.is_cuda else tensor
    print(f"\n[DEBUG] {name}:")
    print(f"  Shape: {list(cpu_tensor.shape)}")
    print(f"  Dtype: {cpu_tensor.dtype}")
    if cpu_tensor.numel() > 0:
        print(f"  Min: {cpu_tensor.min().item():.6e}")
        print(f"  Max: {cpu_tensor.max().item():.6e}")
        print(f"  Mean: {cpu_tensor.mean().item():.6e}")
        if torch.isnan(cpu_tensor).any():
            print(f"  ⚠️ Contains NaN!")
        if torch.isinf(cpu_tensor).any():
            print(f"  ⚠️ Contains Inf!")


# Triton kernel helper to save intermediate blocks
def save_triton_block(block, name: str, kernel_name: str = None,
                     i_k: int = None, i_t: int = None, i_bh: int = None,
                     i_nh: int = None, **metadata):
    """
    Helper to save Triton blocks from within Python wrapper functions.
    Call this from the Python function that calls the Triton kernel.
    
    Example usage in chunk_bwd_dqkwg function:
    
    # Before calling the kernel, set up debug tensors
    debug_tensors = {}
    
    # After kernel execution, save intermediate results
    if debug_enabled:
        save_triton_block(dq, 'dq_final', 'chunk_bwd_dqkwg')
        save_triton_block(dk, 'dk_final', 'chunk_bwd_dqkwg')
    """
    if not _debugger.enabled:
        return
    
    # Convert Triton tensor to PyTorch if needed
    if not isinstance(block, torch.Tensor):
        # This would need special handling for Triton language tensors
        # In practice, this is called from Python with PyTorch tensors
        return
    
    block_id = []
    if i_k is not None:
        block_id.append(i_k)
    if i_t is not None:
        block_id.append(i_t)
    if i_bh is not None:
        block_id.append(i_bh)
    
    _debugger.save_tensor(
        block, name, 
        kernel_name=kernel_name,
        iteration=i_nh,
        block_id=tuple(block_id) if block_id else None,
        **metadata
    )


# Modified kernel wrapper with debugging
def chunk_bwd_dqkwg_debug(
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
    debug: bool = False,
):
    """
    Wrapper for chunk_bwd_dqkwg with debugging capabilities.
    Set debug=True to save intermediate tensors.
    """
    from fla.ops.gated_delta_product.chunk_deltaproduct_o import chunk_bwd_dqkwg
    
    if debug:
        # Save inputs
        save_triton_block(q, 'q_input', 'chunk_bwd_dqkwg')
        save_triton_block(k, 'k_input', 'chunk_bwd_dqkwg')
        save_triton_block(v, 'v_input', 'chunk_bwd_dqkwg')
        save_triton_block(do, 'do_input', 'chunk_bwd_dqkwg')
        save_triton_block(h, 'h_input', 'chunk_bwd_dqkwg')
        save_triton_block(dh, 'dh_input', 'chunk_bwd_dqkwg')
        
        if g is not None:
            save_triton_block(g, 'g_input', 'chunk_bwd_dqkwg')
    
    # Call original function
    dq, dk, dw, dg = chunk_bwd_dqkwg(
        q=q, k=k, v=v, do=do, h=h, dh=dh,
        g=g, g_gamma=g_gamma, dv=dv, w=w,
        cu_seqlens=cu_seqlens, chunk_size=chunk_size,
        scale=scale, num_householder=num_householder
    )
    
    if debug:
        # Save outputs
        save_triton_block(dq, 'dq_output', 'chunk_bwd_dqkwg')
        save_triton_block(dk, 'dk_output', 'chunk_bwd_dqkwg')
        if dw is not None:
            save_triton_block(dw, 'dw_output', 'chunk_bwd_dqkwg')
        if dg is not None:
            save_triton_block(dg, 'dg_output', 'chunk_bwd_dqkwg')
        
        # Print summary
        _debugger.print_summary()
    
    return dq, dk, dw, dg


# Export functions
__all__ = [
    'TritonDebugger',
    'debug_save_tensor',
    'debug_compare',
    'debug_print_stats',
    'save_triton_block',
    'chunk_bwd_dqkwg_debug',
]