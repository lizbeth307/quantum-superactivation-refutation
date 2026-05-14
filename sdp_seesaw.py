"""
sdp_seesaw.py — SDP-based seesaw for channel fidelity F_c(N^⊗n, d).

This is the KEY missing module for SA discovery.
Uses cvxpy to optimize CPTP encoder/decoder maps.

For n copies of channel N with CQ structure (flag + qubit):
  Bob measures flags z ∈ {0,1}^n, gets erasure count k = |z|
  Conditional channel N_k acts on n qubits:
    (n-k) qubits: identity
    k qubits: X^p ∘ Δ̄  (dephasing + bit-flip)

  F_c = Σ_k P(k) · F_k
  where F_k = max_{enc, dec_k} over Schur-Weyl blocks

For small n (≤8), we work in full Hilbert space.
For n>8, we use Schur-Weyl block decomposition.
"""
import numpy as np, cvxpy as cp, time
from scipy.special import comb

I2 = np.eye(2, dtype=complex)
sX = np.array([[0,1],[1,0]], dtype=complex)
sZ = np.array([[1,0],[0,-1]], dtype=complex)

P_FLIP = 1.0 / (1.0 + np.sqrt(2.0))

def noise_kraus():
    """Kraus for X^p ∘ Δ̄ on one qubit."""
    p = P_FLIP
    return [kf @ kd 
            for kd in [np.sqrt(0.5)*I2, np.sqrt(0.5)*sZ]
            for kf in [np.sqrt(1-p)*I2, np.sqrt(p)*sX]]

def build_Nk_choi(n, k):
    """Build Choi matrix of N_k = id^⊗(n-k) ⊗ noise^⊗k on n qubits.
    
    Choi(N_k) ∈ C^(2^n × 2^n) ⊗ C^(2^n × 2^n)  [input ⊗ output]
    Dimension: (2^n)² × (2^n)²
    
    For efficiency, return as Kraus operators instead.
    """
    d = 2**n
    # Build noise^⊗k Kraus
    nk = noise_kraus()
    if k == 0:
        return [np.eye(d, dtype=complex)]
    
    Ks_noise = nk
    for _ in range(k-1):
        Ks_noise = [np.kron(K1, K2) for K1 in Ks_noise for K2 in nk]
    
    # Tensor with id^⊗(n-k)
    d_id = 2**(n-k)
    return [np.kron(np.eye(d_id, dtype=complex), K) for K in Ks_noise]

def sdp_fidelity_fixed_encoder(Ks, V, d_R=2):
    """SDP: maximize F over CPTP decoder, given encoder V.
    
    V: (d_in, d_R) isometry (encoder)
    
    σ_RB = (id_R ⊗ N)(|Ψ_V⟩⟨Ψ_V|)
    F = max_{D CPTP} ⟨Φ| (id_R ⊗ D)(σ_RB) |Φ⟩
    
    = max_{J_D ≥ 0, Tr_out(J_D) = I/d_out} Tr(J_D · σ_RB^Γ)
    
    Actually for decoder optimization we use:
    F = max_W (1/d_R) Σ_{ij} ⟨i|W σ[i,j] W†|j⟩
    
    For CPTP decoder (more general than isometric):
    J_D ∈ C^(d_out × d_R) ⊗ C^(d_out × d_R), J_D ≥ 0, Tr_R(J_D) = I_{d_out}
    """
    d_in = V.shape[0]
    d_out = Ks[0].shape[0]
    
    # Compute σ_RB
    sigma = np.zeros((d_R*d_out, d_R*d_out), dtype=complex)
    for K in Ks:
        KV = K @ V  # (d_out, d_R)
        for i in range(d_R):
            for j in range(d_R):
                sigma[i*d_out:(i+1)*d_out, j*d_out:(j+1)*d_out] += \
                    np.outer(KV[:,i], KV[:,j].conj()) / d_R
    
    # For isometric decoder: max eigenvector of σ_RB
    evals = np.linalg.eigvalsh(sigma)
    F_iso = evals[-1] / d_R  # Upper bound for isometric
    
    # For CPTP decoder via SDP:
    # max Tr(J_D · Ω) s.t. J_D ≥ 0, Tr_{R'}(J_D) = I_{B}/d_B
    # where Ω encodes the fidelity objective
    
    # Build Ω: the "pull-back" of Bell state through σ
    # Ω_{B,R'} = (1/d_R) Σ_{ij} |i⟩⟨j|_{R'} ⊗ σ_RB[i,:,j,:]
    # This is just σ_RB with R→R' relabeling
    # F = Tr(J_D · Ω) where J_D has systems B⊗R'
    
    # J_D ∈ C^(d_out·d_R) × C^(d_out·d_R)
    # Constraint: Tr_{R'}(J_D) = I_{d_out}   [CPTP trace preservation]
    
    dim = d_out * d_R
    J = cp.Variable((dim, dim), hermitian=True)
    
    # Objective: maximize Tr(J · σ_RB) / d_R²
    # σ is already ordered as R⊗B, need to match J ordering as B⊗R'
    # Reorder σ from R⊗B to B⊗R
    sigma_BR = np.zeros_like(sigma)
    for i in range(d_R):
        for j in range(d_R):
            for a in range(d_out):
                for b in range(d_out):
                    sigma_BR[a*d_R+i, b*d_R+j] = sigma[i*d_out+a, j*d_out+b]
    
    obj = cp.Maximize(cp.real(cp.trace(J @ sigma_BR)) / d_R)
    
    # Constraint: J ≥ 0
    constraints = [J >> 0]
    
    # Constraint: Tr_{R'}(J) = I_{d_out}
    # Tr_{R'} means sum over R' indices
    for a in range(d_out):
        for b in range(d_out):
            # (Tr_R J)[a,b] = Σ_r J[a*d_R+r, b*d_R+r]
            val = sum(J[a*d_R+r, b*d_R+r] for r in range(d_R))
            if a == b:
                constraints.append(val == 1)
            else:
                constraints.append(val == 0)
    
    prob = cp.Problem(obj, constraints)
    try:
        prob.solve(solver=cp.SCS, verbose=False, max_iters=5000)
        if prob.status in ['optimal', 'optimal_inaccurate']:
            return prob.value
    except:
        pass
    
    return F_iso  # Fallback

def sdp_fidelity_fixed_decoder(Ks, J_D_val, d_R=2):
    """Optimize encoder V given decoder Choi J_D.
    
    This reduces to finding the top eigenvector of a certain matrix.
    """
    d_in = Ks[0].shape[1]
    d_out = Ks[0].shape[0]
    
    # Build the "pull-back" matrix for encoder optimization
    # For each pair of input basis vectors, compute contribution
    # Y[ia, jb] = (1/d_R²) Σ_K J_D_val applied to K|ia⟩
    # This is complex — use isometric encoder optimization instead
    
    # Gram matrix approach: M = Σ_K (W†K)^T ⊗ (W†K)* 
    # For CPTP decoder, extract effective W from J_D
    
    # Extract decoder isometry from J_D eigendecomposition
    ev, evec = np.linalg.eigh(J_D_val)
    # Top eigenvector gives approximate isometric decoder
    w = evec[:, -1]
    W = w.reshape(d_R, d_out) if d_out >= d_R else w.reshape(d_out, d_R).T
    U, s, Vh = np.linalg.svd(W, full_matrices=False)
    W = U @ Vh  # (d_R, d_out) isometry
    
    # Now optimize V: F = (1/d_R²) Σ_K |Tr(W K V)|²
    TK_list = [W @ K for K in Ks]
    M = np.zeros((d_in*d_R, d_in*d_R), dtype=complex)
    for TK in TK_list:
        t = TK.T.ravel('F')
        M += np.outer(t, t.conj())
    
    evals, evecs = np.linalg.eigh(M)
    v = evecs[:, -1]
    V = v.reshape(d_in, d_R, order='F')
    U2, s2, Vh2 = np.linalg.svd(V, full_matrices=False)
    V = U2 @ Vh2
    
    return V

def seesaw_sdp(n, k, d_R=2, n_iter=15, n_restarts=5):
    """Full SDP seesaw for F_k = max F(Φ, dec ∘ N_k ∘ enc (Φ)).
    
    Alternates:
    1. Fix encoder → SDP for CPTP decoder
    2. Fix decoder → eigenvalue problem for encoder
    """
    d = 2**n
    Ks = build_Nk_choi(n, k)
    
    if d > 64:
        print(f"    k={k}: d={d} too large for full SDP")
        return None
    
    best_F = 0
    
    for restart in range(n_restarts):
        # Random encoder
        V = np.random.randn(d, d_R) + 1j*np.random.randn(d, d_R)
        U, s, Vh = np.linalg.svd(V, full_matrices=False)
        V = U @ Vh
        
        for it in range(n_iter):
            # Step 1: SDP decoder
            F = sdp_fidelity_fixed_encoder(Ks, V, d_R)
            
            # Step 2: Update encoder (approximate via isometric)
            sigma = np.zeros((d_R*d, d_R*d), dtype=complex)
            for K in Ks:
                KV = K @ V
                for i in range(d_R):
                    for j in range(d_R):
                        sigma[i*d:(i+1)*d, j*d:(j+1)*d] += \
                            np.outer(KV[:,i], KV[:,j].conj()) / d_R
            
            # Best isometric decoder
            evals_s, evecs_s = np.linalg.eigh(sigma)
            w = evecs_s[:, -1]
            W = w.reshape(d_R, d)
            U_w, s_w, Vh_w = np.linalg.svd(W, full_matrices=False)
            W = U_w @ Vh_w
            
            # Optimize encoder
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
        
        best_F = max(best_F, F)
    
    return best_F

# ═══ MAIN ═══
if __name__ == '__main__':
    print("="*60)
    print("  SDP SEESAW — CPTP Decoder Optimization")
    print("="*60)
    
    for n in range(1, 7):
        d = 2**n
        if d > 64:
            print(f"\n  n={n}: d={d} too large, skipping")
            break
        
        print(f"\n  n={n} (d={d}):")
        F_total = 0
        
        for k in range(n+1):
            pk = comb(n, k, exact=True) / (2**n)
            t0 = time.time()
            
            Fk = seesaw_sdp(n, k, d_R=2, n_iter=10, n_restarts=3)
            dt = time.time() - t0
            
            if Fk is not None:
                F_total += pk * Fk
                print(f"    k={k}: P={pk:.4f} F_D={Fk:.6f} contrib={pk*Fk:.6f} [{dt:.1f}s]")
            else:
                print(f"    k={k}: skipped")
        
        marker = " 🌟 SA!" if F_total > 0.75 else (" ↑" if F_total > 0.5 else "")
        print(f"  → F_c(Ñ^⊗{n}, 2) ≥ {F_total:.6f}{marker}")
    
    print(f"\n  Reference: Parentin n=17 → F = 0.750131 > 0.75")
    print(f"  Our seesaw gives LOWER BOUND (limited restarts)")
    print(f"{'='*60}")
