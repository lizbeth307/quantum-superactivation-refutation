"""
parallel_sa_exact.py — 30-core exact SA solver for N^{\otimes n}.
Uses the symmetric subspace (Dicke basis) for efficient exact evaluation up to n=11.
"""
import numpy as np, sys, time, os
import multiprocessing as mp
from scipy.special import comb
import cvxpy as cp
from itertools import combinations
sys.path.insert(0, '.')
from sa_engine import build_effective_channel

# ═══════════════════════════════════════════
#  DICKE BASIS BUILDER
# ═══════════════════════════════════════════
def dicke_basis(n):
    """Full Dicke basis for n qubits. Returns (2^n, n+1) matrix."""
    d = 2**n
    basis = np.zeros((d, n+1), dtype=complex)
    
    bits = np.zeros(d, dtype=int)
    for i in range(1, d):
        bits[i] = bits[i >> 1] + (i & 1)
        
    for k in range(n + 1):
        indices = np.where(bits == k)[0]
        val = 1.0 / np.sqrt(len(indices))
        basis[indices, k] = val
        
    return basis

# ═══════════════════════════════════════════
#  EXACT CHANNEL BUILDER (DICKE BASIS)
# ═══════════════════════════════════════════
def build_Nk_dicke_exact(n, k, Ks_single):
    """
    Build channel N_k (k erasures out of n) exactly in the Dicke basis.
    N_k = 1/C(n,k) * sum_{subsets of size k} (N_tilde on subset, I on rest)
    """
    d_sym = n + 1
    d_out = Ks_single[0].shape[0]  # Should be 4
    d_in = Ks_single[0].shape[1]   # Should be 2
    
    # Target dimension: output will have k noisy qubits (dim d_out each) 
    # and n-k intact qubits (dim d_in each)
    # Actually, if we want the output to remain in the symmetric subspace, 
    # this channel (which has C^2 -> C^4) takes us OUT of the symmetric subspace of C^2.
    
    # Wait, Parentin's effective channel N_tilde is C^2 -> C^2 !
    # But Ks = build_effective_channel() returns 2->4 operators.
    # Ah! The flag qubits...
    
    # If the output is C^4, then N^{\otimes n} has output (C^4)^{\otimes n}.
    # We cannot simply project the output to a C^2 Dicke basis!
    # The output space is huge.
    
    pass

if __name__ == '__main__':
    print("This code highlighted a major issue: N_tilde maps C^2 to C^4.")
    print("Projecting the OUTPUT to Dicke basis is mathematically invalid,")
    print("because the output is in (C^4)^n, not (C^2)^n.")
    print("We must trace out the flag systems or use the CQ structure properly.")
