import numpy as np

# ==============================================================================
# ⚛️ QuantumNEAT: Computable Cross-Norm or Realignment (CCNR) Criterion
# ==============================================================================
# The CCNR criterion (2002) is a powerful analytical tool to detect entanglement.
# It works by rearranging the indices of the density matrix and computing its
# trace norm (sum of singular values).
#
# If the sum of singular values > 1, the state is strictly ENTANGLED.
# This criterion can sometimes detect bound entangled states that the PPT criterion
# misses. We will test our NPT candidates with it.
# ==============================================================================

def ccnr_criterion(rho, d_A=3, d_B=3):
    """
    Computes the Realignment (CCNR) criterion for a bipartite state.
    """
    # 1. Reshape into tensor (i, j, k, l)
    # i, k belong to Alice (A)
    # j, l belong to Bob (B)
    rho_tensor = rho.reshape((d_A, d_B, d_A, d_B))
    
    # 2. Realign: Group Alice's indices (i, k) and Bob's indices (j, l)
    # Permute to (i, k, j, l)
    realigned_tensor = np.transpose(rho_tensor, (0, 2, 1, 3))
    
    # 3. Flatten into a new matrix R(rho)
    R_rho = realigned_tensor.reshape((d_A * d_A, d_B * d_B))
    
    # 4. Compute Singular Value Decomposition (SVD)
    # We only need the singular values
    singular_values = np.linalg.svd(R_rho, compute_uv=False)
    
    # 5. Sum the singular values (Trace Norm)
    trace_norm = np.sum(singular_values)
    
    return trace_norm

def test_candidates():
    print("🔬 Running Computable Cross-Norm / Realignment (CCNR) Test 🔬")
    print("-" * 60)
    print("Rule: If Trace Norm > 1.0, the state is DEFINITELY ENTANGLED.")
    print("-" * 60)
    
    candidates = ["candidate_mc.npy", "perfect_npt_be_hacked.npy", "candidate_npt_be_k2.npy"]
    
    for filename in candidates:
        try:
            rho = np.load(filename)
            norm = ccnr_criterion(rho)
            
            print(f"\nEvaluating: {filename}")
            print(f"CCNR Trace Norm: {norm:.6f}")
            
            if norm > 1.0 + 1e-10:
                print("💥 VIOLATION DETECTED! The CCNR criterion confirms this state is strictly entangled.")
            else:
                print("✅ No violation. CCNR cannot detect entanglement here (it might be bound, or highly mixed).")
                
        except FileNotFoundError:
            print(f"\nSkipping {filename} (not found).")

if __name__ == "__main__":
    test_candidates()
