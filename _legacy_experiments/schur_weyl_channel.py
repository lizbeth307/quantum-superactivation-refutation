"""
schur_weyl_channel.py — Combinatorial evaluation of N^{⊗n} on Dicke states.
"""
import numpy as np, sys
from scipy.special import comb
from scipy import sparse
import time

def test_n_feasibility(n):
    """Test if we can build the projected channel for n directly."""
    print(f"Testing n={n} sparse Dicke state generation...")
    
    t0 = time.time()
    dim = 2**n
    dicke_sparse = []
    
    # Fast bit counting
    bits = np.zeros(dim, dtype=int)
    for i in range(1, dim):
        bits[i] = bits[i >> 1] + (i & 1)
        
    for k in range(n + 1):
        indices = np.where(bits == k)[0]
        val = 1.0 / np.sqrt(len(indices))
        vec = sparse.coo_matrix((np.full(len(indices), val), 
                               (indices, np.zeros(len(indices)))), 
                              shape=(dim, 1))
        dicke_sparse.append(vec.tocsc())
        
    print(f"n={n}: Built {n+1} sparse Dicke states in {time.time()-t0:.4f}s")
    return dicke_sparse

if __name__ == '__main__':
    for n in [8, 10, 12, 14, 16, 17]:
        test_n_feasibility(n)
