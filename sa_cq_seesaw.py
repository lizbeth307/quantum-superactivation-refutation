"""
sa_cq_seesaw.py — Exploit CQ structure of Ñ^⊗n.

Key insight from Parentin et al.:
After n uses of Ñ, Bob measures n flag bits z ∈ {0,1}^n.
Conditioned on Hamming weight k = |z|:
  - (n-k) qubits pass through identity
  - k qubits pass through X^p ∘ Δ̄ (dephasing + bit-flip)

The effective channel for weight k is:
  N_k = id^⊗(n-k) ⊗ (X^p ∘ Δ̄)^⊗k  (up to permutation)

Probability: P(k) = C(n,k) / 2^n

F_c = Σ_k P(k) · F_D[k]
where F_D[k] = max_{enc,dec_k} F(Φ, (id⊗dec_k∘N_k∘enc)(Φ))

This splits into n+1 INDEPENDENT subproblems!
Each acts on n qubits → dim 2^n, but with permutation invariance → Schur-Weyl.
"""
import numpy as np, time
from scipy.special import comb

I2 = np.eye(2, dtype=complex)
sX = np.array([[0,1],[1,0]], dtype=complex)
sZ = np.array([[1,0],[0,-1]], dtype=complex)

p_flip = 1.0 / (1.0 + np.sqrt(2.0))

def noise_kraus():
    """Kraus for X^p ∘ Δ̄ on a single qubit."""
    Ks = []
    for kd in [np.sqrt(0.5)*I2, np.sqrt(0.5)*sZ]:  # dephasing
        for kx in [np.sqrt(1-p_flip)*I2, np.sqrt(p_flip)*sX]:  # bit-flip
            Ks.append(kx @ kd)
    return Ks

def build_N_k_kraus(n, k):
    """Build Kraus for N_k = id^⊗(n-k) ⊗ noise^⊗k on n qubits.
    
    For simplicity, noise acts on LAST k qubits.
    """
    noise = noise_kraus()
    
    # Identity on first (n-k) qubits, noise on last k
    if k == 0:
        return [np.eye(2**n, dtype=complex)]
    
    # Build noise^⊗k
    noise_k = noise
    for _ in range(k-1):
        noise_k = [np.kron(K1, K2) for K1 in noise_k for K2 in noise]
    
    # Tensor with identity on first (n-k) qubits
    d_id = 2**(n-k)
    return [np.kron(np.eye(d_id, dtype=complex), K) for K in noise_k]

def seesaw_for_k(n, k, d_R=2, n_iter=40, n_restarts=15):
    """Seesaw for F_D[k] = max F(Φ, dec_k ∘ N_k ∘ enc (Φ)).
    
    enc: C^d_R → C^(2^n)
    dec_k: C^(2^n) → C^d_R
    N_k acts on 2^n dimensional space
    """
    d = 2**n
    Ks = build_N_k_kraus(n, k)
    
    best_F = 0
    for restart in range(n_restarts):
        # Random encoder V: (d, d_R) isometry
        V = np.random.randn(d, d_R) + 1j*np.random.randn(d, d_R)
        U, s, Vh = np.linalg.svd(V, full_matrices=False)
        V = U @ Vh
        
        for it in range(n_iter):
            # σ_RB = Σ_K (I⊗K)|Ψ_V⟩⟨Ψ_V|(I⊗K†)
            sigma = np.zeros((d_R*d, d_R*d), dtype=complex)
            for K in Ks:
                KV = K @ V
                for i in range(d_R):
                    for j in range(d_R):
                        sigma[i*d:(i+1)*d, j*d:(j+1)*d] += \
                            np.outer(KV[:,i], KV[:,j].conj()) / d_R
            
            # Optimal decoder: top eigenvector of σ_RB
            evals, evecs = np.linalg.eigh(sigma)
            w = evecs[:, -1]
            W = w.reshape(d_R, d)
            U_w, s_w, Vh_w = np.linalg.svd(W, full_matrices=False)
            W = U_w @ Vh_w
            
            # Optimal encoder given W
            TK_list = [W @ K for K in Ks]
            M = np.zeros((d*d_R, d*d_R), dtype=complex)
            for TK in TK_list:
                t = TK.T.ravel('F')
                M += np.outer(t, t.conj())
            
            evals_v, evecs_v = np.linalg.eigh(M)
            v = evecs_v[:, -1]
            V = v.reshape(d, d_R, order='F')
            U_v, s_v, Vh_v = np.linalg.svd(V, full_matrices=False)
            V = U_v @ Vh_v
        
        # Final fidelity
        sigma = np.zeros((d_R*d, d_R*d), dtype=complex)
        for K in Ks:
            KV = K @ V
            for i in range(d_R):
                for j in range(d_R):
                    sigma[i*d:(i+1)*d, j*d:(j+1)*d] += \
                        np.outer(KV[:,i], KV[:,j].conj()) / d_R
        
        evals, evecs = np.linalg.eigh(sigma)
        w = evecs[:, -1]
        W = w.reshape(d_R, d)
        U_w, s_w, Vh_w = np.linalg.svd(W, full_matrices=False)
        W = U_w @ Vh_w
        
        F = 0
        for i in range(d_R):
            for j in range(d_R):
                block = sigma[i*d:(i+1)*d, j*d:(j+1)*d]
                F += (W @ block @ W.conj().T)[i,j]
        F = F.real / d_R
        
        best_F = max(best_F, F)
    
    return best_F

# ═══ MAIN ═══
if __name__ == '__main__':
    print("="*60)
    print("  SA CQ-SEESAW — Exploit Classical-Quantum Structure")
    print("="*60)
    
    for n in [1, 2, 3, 4, 5, 6, 7, 8]:
        d = 2**n
        if d > 256:
            print(f"\n  n={n}: d={d} too large for brute-force, need Schur-Weyl")
            break
        
        print(f"\n  n={n} (d={d}):")
        F_total = 0
        
        for k in range(n+1):
            pk = comb(n, k, exact=True) / (2**n)
            t0 = time.time()
            
            n_restarts = 20 if d <= 16 else 10
            Fk = seesaw_for_k(n, k, d_R=2, n_iter=40, n_restarts=n_restarts)
            dt = time.time() - t0
            
            F_total += pk * Fk
            print(f"    k={k:>2}: P={pk:.4f} F_D={Fk:.6f} contrib={pk*Fk:.6f} [{dt:.1f}s]")
        
        marker = " 🌟 SA!" if F_total > 0.75 else (" ↑" if F_total > 0.5 else "")
        print(f"  → F_c(Ñ^⊗{n}, 2) ≥ {F_total:.6f}{marker}")
    
    print(f"\n  Bounds: F_c(erasure) ≤ 0.75, F_c(PPT) ≤ 0.50")
    print(f"{'='*60}")
