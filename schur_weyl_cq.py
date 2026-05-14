"""
schur_weyl_cq.py — Schur-Weyl implementation matching Parentin's approach.
Reads the exact Parentin blocks to understand the representation.
"""
import numpy as np, sys, os
from scipy.special import comb
import cvxpy as cp
sys.path.insert(0, '.')
from sa_engine import verify_parentin

def explore_parentin_blocks(n, data_dir):
    """Analyze the dimensions of the blocks in Parentin's data."""
    def _partition_ordering(n_sym):
        P = n_sym // 2 + 1
        return list(range(1, P)) + [0] if P > 1 else [0]
        
    print(f"Parentin blocks for n={n}")
    
    k = n // 2  # Examine the middle k (most complex)
    nmk = n - k
    j_k_order = _partition_ordering(k)
    j_nk_order = _partition_ordering(nmk)
    
    print(f"k={k}: j_k_order = {j_k_order}")
    print(f"nmk={nmk}: j_nk_order = {j_nk_order}")
    
    bobs = []; b = 0
    while os.path.exists(os.path.join(data_dir, f"bob_pov_{k}_block_{b}.npy")):
        bobs.append(np.load(os.path.join(data_dir, f"bob_pov_{k}_block_{b}.npy")))
        b += 1
        
    print(f"Found {len(bobs)} blocks for k={k}")
    
    n_blocks_nmk = len(j_nk_order)
    for idx in range(len(bobs)):
        j_k = j_k_order[idx // n_blocks_nmk]
        j_nk = j_nk_order[idx % n_blocks_nmk]
        print(f"  Block {idx}: j_k={j_k}, j_nk={j_nk}, shape={bobs[idx].shape}")
        
    return bobs

if __name__ == '__main__':
    data_dir = os.path.join(os.getcwd(), 'parentin_data')
    explore_parentin_blocks(17, data_dir)
