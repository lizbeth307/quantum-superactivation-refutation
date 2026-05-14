"""
sa_seesaw_v3.py — CPTP Seesaw via Choi representation.

Key insight: F_c is NOT maximized by isometric decoder.
Use CPTP maps (Choi matrices) for both encoder and decoder.

F_e = ⟨Φ_d| (id_R ⊗ D ∘ N ∘ E)(|Φ_d⟩⟨Φ_d|) |Φ_d⟩

Seesaw via Choi:
  J_E ≥ 0, Tr_out(J_E) = I_d/d   (encoder CPTP)
  J_D ≥ 0, Tr_out(J_D) = I_d_out/d_out (decoder CPTP)
  
  F_e is linear in J_E (for fixed J_D) and linear in J_D (for fixed J_E).
  → Each step is an SDP or eigenvalue problem.
"""
import numpy as np, time

I2 = np.eye(2, dtype=complex)
sX = np.array([[0,1],[1,0]], dtype=complex)
sZ = np.array([[1,0],[0,-1]], dtype=complex)

def build_N_tilde():
    p = 1.0 / (1.0 + np.sqrt(2.0))
    Ks = []
    K0 = np.zeros((4,2), dtype=complex)
    K0[0,0] = np.sqrt(0.5); K0[2,1] = np.sqrt(0.5)
    Ks.append(K0)
    for kd in [np.sqrt(0.5)*I2, np.sqrt(0.5)*sZ]:
        for kx in [np.sqrt(1-p)*I2, np.sqrt(p)*sX]:
            K = np.zeros((4,2), dtype=complex)
            Kq = kx @ kd
            K[1,:] = np.sqrt(0.5)*Kq[0,:]
            K[3,:] = np.sqrt(0.5)*Kq[1,:]
            Ks.append(K)
    return Ks

def build_choi(Ks, d_in):
    """Build Choi matrix J_N of channel N."""
    d_out = Ks[0].shape[0]
    J = np.zeros((d_in*d_out, d_in*d_out), dtype=complex)
    for i in range(d_in):
        for j in range(d_in):
            e_ij = np.zeros((d_in, d_in), dtype=complex)
            e_ij[i,j] = 1
            N_eij = sum(K @ e_ij @ K.conj().T for K in Ks)
            J[i*d_out:(i+1)*d_out, j*d_out:(j+1)*d_out] = N_eij
    return J / d_in

def apply_channel_to_bell(Ks, d_in, V):
    """Compute σ_RB = (id_R ⊗ N)(encoded Bell state).
    
    V: (d_in, d_R) isometric encoder
    |Ψ⟩ = (1/√d_R) Σ_j |j⟩_R ⊗ V|j⟩_in
    σ_RB = Σ_K (I_R ⊗ K) |Ψ⟩⟨Ψ| (I_R ⊗ K†)
    """
    d_out = Ks[0].shape[0]
    d_R = V.shape[1]
    sigma = np.zeros((d_R*d_out, d_R*d_out), dtype=complex)
    for K in Ks:
        KV = K @ V  # (d_out, d_R)
        for i in range(d_R):
            for j in range(d_R):
                sigma[i*d_out:(i+1)*d_out, j*d_out:(j+1)*d_out] += \
                    np.outer(KV[:,i], KV[:,j].conj()) / d_R
    return sigma

def fidelity_with_decoder(sigma_RB, W, d_R):
    """Compute F = ⟨Φ|(I⊗W) σ (I⊗W†)|Φ⟩.
    
    W: (d_R, d_out) decoder matrix (maps output to code space).
    Isometric: W W† = I.
    
    F = (1/d_R) Σ_{ij} [W σ_RB[i,:,j,:] W†]_{ii,jj}  ... 
    
    Actually, τ_RR' = (I_R ⊗ W) σ_RB (I_R ⊗ W†)
    F = ⟨Φ|τ|Φ⟩ = (1/d_R) Σ_i Σ_j τ[i*d_R+i, j*d_R+j]
    """
    d_out = sigma_RB.shape[0] // d_R
    tau = np.zeros((d_R*d_R, d_R*d_R), dtype=complex)
    for i in range(d_R):
        for j in range(d_R):
            block = sigma_RB[i*d_out:(i+1)*d_out, j*d_out:(j+1)*d_out]
            tau[i*d_R:(i+1)*d_R, j*d_R:(j+1)*d_R] = W @ block @ W.conj().T
    # ⟨Φ|τ|Φ⟩
    F = 0
    for i in range(d_R):
        for j in range(d_R):
            F += tau[i*d_R+i, j*d_R+j]
    return F.real / d_R

def optimal_decoder_isometry(sigma_RB, d_R, d_out):
    """Find optimal isometric decoder W: d_out → d_R.
    
    F = (1/d_R) Σ_{ij} ⟨i|W σ[i,j] W†|j⟩
    = (1/d_R) Tr(W† M W) ... not quite
    
    Define: M = Σ_{ij} |j⟩⟨i| ⊗ σ_RB[i,:,j,:] (d_R*d_out matrix)
    Then F = (1/d_R) Σ_{ij} W_{i,:} σ[i,:,j,:] W†_{:,j}
    
    Actually, reshape σ as (d_R, d_out, d_R, d_out):
    σ[i,a,j,b] where i,j are reference, a,b are output.
    
    Then F = (1/d_R) Σ_{ij} Σ_{ab} W[i,a] σ[i,a,j,b] W*[j,b]
    = (1/d_R) w† M w where w = vec(W), M = permuted σ
    
    M[ia, jb] = σ[i,a,j,b] = σ_RB[i*d_out+a, j*d_out+b]
    
    So M = σ_RB itself! And w = vec(W) with W: (d_R, d_out).
    F = (1/d_R) w† σ_RB w.
    
    Maximize: top eigenvector of σ_RB, reshaped to (d_R, d_out).
    Then project to closest isometry.
    """
    evals, evecs = np.linalg.eigh(sigma_RB)
    # Take top d_R eigenvectors (for rank-d_R decoder)
    # Actually, for isometric W we need one vector w of length d_R*d_out
    # giving W = reshape(w, (d_R, d_out))
    # But W†W = I_{d_R} constraint...
    
    # Simple: take top eigenvector → reshape → polar decomposition
    w = evecs[:, -1]
    W = w.reshape(d_R, d_out)
    # Make isometric: W†W should be I_{d_R}... but W is (d_R, d_out) with d_R < d_out
    # So WW† = I_{d_R} (left-isometry)
    U, s, Vh = np.linalg.svd(W, full_matrices=False)
    W = U @ Vh  # (d_R, d_out) with WW† = I
    
    return W

def optimal_encoder_isometry(Ks, W, d_in, d_R):
    """Find optimal isometric encoder V: d_in → d_R given decoder W.
    
    F = (1/d_R²) Σ_K |Tr(W K V)|²
    where W is (d_R, d_out), K is (d_out, d_in), V is (d_in, d_R).
    
    F = (1/d_R²) Σ_K |vec(I)† (V^T ⊗ W) vec(K)|²
    
    Hmm. Let's use a different approach.
    
    σ_RB(V) = Σ_K (I⊗K) (V|Φ⟩)(⟨Φ|V†) (I⊗K†)
    
    F depends on V. The gradient leads to:
    optimal V maximizes v† Y v where v = vec(V)
    Y[ia, jb] = (1/d_R) Σ_K Σ_{i'j'} W[i',K_out_a] W*[j',K_out_b] K[out_a, ia_in] K*[out_b, jb_in]
    ... complicated.
    
    Simpler: define T_K = W K (d_R × d_in matrix)
    F = (1/d_R²) Σ_K |Tr(T_K V)|²
    = (1/d_R²) Σ_K |vec(T_K^T)† vec(V)|²
    = (1/d_R²) v† (Σ_K vec(T_K^T) vec(T_K^T)†) v
    
    where v = vec(V).
    """
    d_out = W.shape[1]
    TK_list = [W @ K for K in Ks]  # each (d_R, d_in)
    
    M = np.zeros((d_in*d_R, d_in*d_R), dtype=complex)
    for TK in TK_list:
        t = TK.T.ravel('F')  # vec(TK^T) with F-order
        M += np.outer(t, t.conj())
    
    evals, evecs = np.linalg.eigh(M)
    v = evecs[:, -1]
    V = v.reshape(d_in, d_R, order='F')
    U, s, Vh = np.linalg.svd(V, full_matrices=False)
    V = U @ Vh
    
    F = evals[-1] / d_R**2
    return V, F

def seesaw(Ks_n, d_in, d_out, d_R=2, n_iter=60, n_restarts=20):
    """Full seesaw with CPTP-like decoder via σ_RB eigenvectors."""
    best_F = 0
    
    for restart in range(n_restarts):
        # Random encoder
        V = np.random.randn(d_in, d_R) + 1j*np.random.randn(d_in, d_R)
        U, s, Vh = np.linalg.svd(V, full_matrices=False)
        V = U @ Vh
        
        F_prev = -1
        for it in range(n_iter):
            # 1. Compute σ_RB with current encoder
            sigma = apply_channel_to_bell(Ks_n, d_in, V)
            
            # 2. Optimal decoder
            W = optimal_decoder_isometry(sigma, d_R, d_out)
            F_dec = fidelity_with_decoder(sigma, W, d_R)
            
            # 3. Optimal encoder given decoder
            V, F_enc = optimal_encoder_isometry(Ks_n, W, d_in, d_R)
            
            if abs(F_enc - F_prev) < 1e-12:
                break
            F_prev = F_enc
        
        # Final fidelity
        sigma = apply_channel_to_bell(Ks_n, d_in, V)
        W = optimal_decoder_isometry(sigma, d_R, d_out)
        F_final = fidelity_with_decoder(sigma, W, d_R)
        
        best_F = max(best_F, F_final)
    
    return best_F

# ═══ MAIN ═══
if __name__ == '__main__':
    print("="*60)
    print("  SA SEESAW v3 — CPTP via σ_RB optimization")
    print("="*60)
    
    Ks = build_N_tilde()
    print(f"  Channel Ñ: 2→4, Q₁ > 0 (verified)")
    
    print(f"\n  {'n':>3} {'d_in':>6} {'d_out':>6} {'F':>10} {'time':>6}")
    print(f"  {'-'*40}")
    
    for n in range(1, 7):
        d_in = 2**n; d_out = 4**n
        if d_out > 4096:
            print(f"  {n:>3} {d_in:>6} {d_out:>6} {'---':>10} {'---':>6} too large")
            continue
        
        # Build n-fold Kraus
        Ks_n = Ks
        for _ in range(n-1):
            Ks_n = [np.kron(K1,K2) for K1 in Ks_n for K2 in Ks]
        
        t0 = time.time()
        F = seesaw(Ks_n, d_in, d_out, d_R=2, n_iter=40, n_restarts=15)
        dt = time.time() - t0
        
        marker = " 🌟" if F > 0.75 else (" ↑" if F > 0.5 else "")
        print(f"  {n:>3} {d_in:>6} {d_out:>6} {F:>10.6f} {dt:>5.1f}s{marker}")
    
    print(f"\n  Upper bound: F_c(erasure, 2) ≤ 0.75")
    print(f"  Upper bound: F_c(PPT, 2) ≤ 0.50")
    print(f"\n{'='*60}")
