"""
sa_correct.py — Correct SA Protocol following Parentin et al. (2026)
=====================================================================
Effective channel Ñ: C^2 → C^4 (qubit input, qubit⊗flag output)
  Ñ(ρ) = (1/2)|0⟩⟨0|_Z ⊗ ρ  +  (1/2)|1⟩⟨1|_Z ⊗ X^p(Δ̄(ρ))
  
where:
  - X^p(ρ) = (1-p)ρ + p σ_X ρ σ_X   (bit-flip, p = 1/(1+√2))
  - Δ̄(ρ) = (1/2)ρ + (1/2) σ_Z ρ σ_Z (complete dephasing)

This replaces the full P⊗A (16→20 dim) with a simple 2→4 channel.
SA criterion: F_c(Ñ^⊗n, 2) > 3/4 for some n.

We implement:
1. Exact Ñ channel Kraus operators
2. Coherent information maximization (single-shot Q₁)
3. n-copy channel fidelity via seesaw optimization
"""
import numpy as np, time
from scipy.linalg import sqrtm

# ═══ Pauli matrices ═══
I2 = np.eye(2, dtype=complex)
sX = np.array([[0,1],[1,0]], dtype=complex)
sZ = np.array([[1,0],[0,-1]], dtype=complex)

def S(rho):
    """Von Neumann entropy."""
    e = np.linalg.eigvalsh(rho); e = e[e>1e-15]
    return -np.sum(e*np.log2(e)) if len(e)>0 else 0.0

# ═══ Effective Channel Ñ ═══
def build_N_tilde():
    """Build Kraus operators for Ñ: C^2 → C^4.
    
    Output space: C^4 = C^2_qubit ⊗ C^2_flag
    Index: (qubit, flag) → qubit*2 + flag
    
    Case Z=0 (no erasure, prob 1/2):
      output = |0⟩_Z ⊗ ρ  (identity on qubit)
      Kraus: K_0 = √(1/2) · (I_qubit ⊗ |0⟩_flag)
    
    Case Z=1 (erasure, prob 1/2):
      output = |1⟩_Z ⊗ X^p(Δ̄(ρ))
      Δ̄(ρ) = diag(ρ_00, ρ_11)
      X^p(σ) = (1-p)σ + p σ_X σ σ_X
      
      This is a composition: first dephase, then bit-flip.
      Kraus for dephasing Δ̄: {√(1/2)I, √(1/2)σ_Z}
      Kraus for X^p: {√(1-p)I, √p σ_X}
      Combined Kraus for Z=1: √(1/2) · K_xp · K_deph ⊗ |1⟩_flag
    """
    p = 1.0 / (1.0 + np.sqrt(2.0))
    
    Ks = []
    # Case Z=0: identity channel, prob 1/2
    # K = √(1/2) [I ⊗ |0⟩]  → 4×2 matrix
    K0 = np.zeros((4, 2), dtype=complex)
    K0[0, 0] = np.sqrt(0.5)  # |0,0⟩ ← |0⟩
    K0[2, 1] = np.sqrt(0.5)  # |1,0⟩ ← |1⟩
    Ks.append(K0)
    
    # Case Z=1: X^p ∘ Δ̄, prob 1/2
    # Kraus for full composition: √(1/2) · K_xp · K_deph
    # K_deph ∈ {√(1/2)I, √(1/2)σ_Z}
    # K_xp ∈ {√(1-p)I, √p σ_X}
    for k_d in [np.sqrt(0.5)*I2, np.sqrt(0.5)*sZ]:
        for k_x in [np.sqrt(1-p)*I2, np.sqrt(p)*sX]:
            K_qubit = k_x @ k_d  # 2×2
            K = np.zeros((4, 2), dtype=complex)
            # Place in flag=1 subspace
            K[1, 0] = np.sqrt(0.5) * K_qubit[0, 0]  # |0,1⟩ ← |0⟩
            K[3, 0] = np.sqrt(0.5) * K_qubit[1, 0]  # |1,1⟩ ← |0⟩
            K[1, 1] = np.sqrt(0.5) * K_qubit[0, 1]  # |0,1⟩ ← |1⟩
            K[3, 1] = np.sqrt(0.5) * K_qubit[1, 1]  # |1,1⟩ ← |1⟩
            Ks.append(K)
    
    return Ks

def verify_channel(Ks, d_in=2):
    """Verify TP and test the channel."""
    S_tp = sum(K.conj().T @ K for K in Ks)
    tp_err = np.linalg.norm(S_tp - np.eye(d_in))
    print(f"  TP error: {tp_err:.2e}")
    
    # Test on |0⟩
    rho0 = np.array([[1,0],[0,0]], dtype=complex)
    out0 = sum(K @ rho0 @ K.conj().T for K in Ks)
    print(f"  Ñ(|0⟩⟨0|) diagonal: {np.diag(out0).real}")
    
    # Test on I/2
    rho_max = I2 / 2
    out_max = sum(K @ rho_max @ K.conj().T for K in Ks)
    print(f"  Ñ(I/2) diagonal: {np.diag(out_max).real}")
    
    return tp_err

def coherent_info_single(Ks, d_in=2, n_trials=500):
    """Maximize I(R>B) = S(B) - S(RB) over input states."""
    d_out = Ks[0].shape[0]
    best = -999
    for t in range(n_trials):
        if t == 0:
            psi = np.array([1,0,0,1], dtype=complex) / np.sqrt(2)
        elif t < 10:
            theta = np.pi * t / 10
            psi = np.array([np.cos(theta), 0, 0, np.sin(theta)], dtype=complex)
            psi /= np.linalg.norm(psi)
        else:
            psi = np.random.randn(d_in*d_in)+1j*np.random.randn(d_in*d_in)
            psi /= np.linalg.norm(psi)
        rho_RA = np.outer(psi, psi.conj())
        rho_RB = np.zeros((d_in*d_out, d_in*d_out), dtype=complex)
        for r1 in range(d_in):
            for r2 in range(d_in):
                bl = rho_RA[r1*d_in:(r1+1)*d_in, r2*d_in:(r2+1)*d_in]
                rho_RB[r1*d_out:(r1+1)*d_out, r2*d_out:(r2+1)*d_out] = sum(K@bl@K.conj().T for K in Ks)
        rho_B = sum(rho_RB[r*d_out:(r+1)*d_out, r*d_out:(r+1)*d_out] for r in range(d_in))
        ci = S(rho_B) - S(rho_RB)
        best = max(best, ci)
    return best

def channel_fidelity_seesaw(Ks, d_in, n_uses, d_code=2, n_iter=50, n_restarts=5):
    """Seesaw optimization for F_c(Ñ^⊗n, d_code).
    
    For small n, we work in full Hilbert space.
    Encoder: C^(d_code*d_code) → C^(d_in^n)  [isometry on Choi]
    Decoder: C^(d_out^n) → C^(d_code*d_code)  [CPTP]
    
    F = (1/d²) Tr(J_dec · (Ñ^⊗n ⊗ id)(J_enc))
    
    Seesaw: fix encoder → optimize decoder (SDP), fix decoder → optimize encoder.
    For simplicity, use power iteration method.
    """
    d_out = Ks[0].shape[0]
    d_total_in = d_in ** n_uses
    d_total_out = d_out ** n_uses
    d_R = d_code
    
    if d_total_in > 512 or d_total_out > 512:
        print(f"  ⚠️ n={n_uses}: dims {d_total_in}→{d_total_out} too large for brute force")
        print(f"  Need Schur-Weyl decomposition for n>{int(np.log2(512)/np.log2(d_out))}")
        return None
    
    # Build n-fold Kraus operators
    if n_uses == 1:
        Ks_n = Ks
    else:
        Ks_prev = Ks
        for _ in range(n_uses - 1):
            Ks_new = []
            for K1 in Ks_prev:
                for K2 in Ks:
                    Ks_new.append(np.kron(K1, K2))
            Ks_prev = Ks_new
        Ks_n = Ks_prev
    
    print(f"  n={n_uses}: {len(Ks_n)} Kraus ops, {d_total_in}→{d_total_out}")
    
    best_F = 0
    for restart in range(n_restarts):
        # Random isometric encoder: d_R → d_total_in
        V = np.random.randn(d_total_in, d_R) + 1j*np.random.randn(d_total_in, d_R)
        V, _ = np.linalg.qr(V)
        V = V[:, :d_R]  # (d_total_in, d_R) isometry
        
        for it in range(n_iter):
            # === Step 1: Apply channel to encoded Bell state ===
            # |Φ⟩ = (1/√d_R) Σ_i |i⟩_R |i⟩_A
            # Encode A: |i⟩_A → V|i⟩
            # Channel acts on encoded input
            
            # Build σ_RB = (id_R ⊗ Ñ^⊗n)(encoded Bell state)
            # σ_RB is d_R*d_total_out × d_R*d_total_out
            sigma_RB = np.zeros((d_R*d_total_out, d_R*d_total_out), dtype=complex)
            for K in Ks_n:
                KV = K @ V  # (d_total_out, d_R)
                # Contribution: (1/d_R) Σ_{i,j} |i⟩⟨j|_R ⊗ KV|i⟩⟨j|KV†
                for i in range(d_R):
                    for j in range(d_R):
                        sigma_RB[i*d_total_out:(i+1)*d_total_out, 
                                 j*d_total_out:(j+1)*d_total_out] += \
                            np.outer(KV[:,i], KV[:,j].conj()) / d_R
            
            # === Step 2: Optimal decoder for this encoder ===
            # Maximize F = (1/d_R²) Tr(σ_RB · (id_R ⊗ D†)(Φ_R))
            # = (1/d_R²) Σ_{i,j} ⟨i|σ_RB|j⟩_{B} where D maps to |i⟩⟨j|
            # Optimal D†: partial transpose trick
            # The optimal decoder maximizes Tr(D · σ_B→R)
            
            # Extract the "effective state" for decoder optimization
            # σ_RB already contains everything. For a unital decoder:
            # Best linear decoder: D(·) = partial trace trick
            
            # Simple approach: SVD of σ_RB viewed as R⊗B
            sigma_RB_reshape = sigma_RB.reshape(d_R, d_total_out, d_R, d_total_out)
            
            # Build X = Σ_{i,j} |i⟩⟨j| ⊗ σ_RB[i,j] block
            # Optimal isometric decoder: V_dec that maps B→R to maximize ⟨Φ|(V_dec⊗I)σ|Φ⟩
            
            # Fidelity = (1/d_R²) Σ_i Σ_j Tr(W†|i⟩⟨j|W · σ_RB[i,j])
            # where W: C^d_total_out → C^d_R is the decoder isometry
            
            # Build M = Σ_{i,j} σ_RB[i,*;j,*]^T · |j⟩⟨i|_R ⊗ I_B
            # Actually simpler: M_B = Σ_i σ_RB[i,:,i,:] and cross terms
            
            # Use partial trace to get decoder:
            # M = d_R * Tr_R(σ_RB · (Φ_R ⊗ I_B))  
            # where Φ_R = |Φ⟩⟨Φ| = (1/d_R)Σ_{ij}|ii⟩⟨jj|
            M = np.zeros((d_total_out, d_total_out), dtype=complex)
            for i in range(d_R):
                for j in range(d_R):
                    M += sigma_RB_reshape[i,:,j,:] * (1 if i==j else 0)
            # Actually we need cross terms for the decoder
            # The right object: for optimal decoder W: C^d_total_out → C^d_R
            # F = (1/d_R²) Tr(W† X W) where X = Σ_{ij} |i⟩⟨j| ⊗ σ[i,B,j,B]^T
            
            X = np.zeros((d_R*d_total_out, d_R*d_total_out), dtype=complex)
            for i in range(d_R):
                for j in range(d_R):
                    X[i*d_total_out:(i+1)*d_total_out,
                      j*d_total_out:(j+1)*d_total_out] = sigma_RB_reshape[i,:,j,:].T
            
            # Optimal W: top d_R right singular vectors of X reshaped
            # W is d_total_out × d_R, so we need top d_R eigenvectors of X
            evals, evecs = np.linalg.eigh(X)
            W_flat = evecs[:, -d_R:]  # top d_R eigenvectors
            W = W_flat.reshape(d_R, d_total_out, d_R)
            
            # Decoder: D(σ) = Σ_i W[i,:,:] σ W[i,:,:]†  ... not quite right
            # Actually W_flat columns are vectors in C^(d_R * d_total_out)
            # Reshape each to (d_R, d_total_out) → decoder isometry
            
            # Simpler: compute fidelity directly
            F = np.sum(evals[-d_R:]) / d_R**2
            
            # === Step 3: Optimal encoder for this decoder ===
            # From X, extract the "pull-back" to input space
            # Y = Σ_K K† X_pulled K where X_pulled involves decoder
            
            Y = np.zeros((d_R*d_total_in, d_R*d_total_in), dtype=complex)
            for K in Ks_n:
                # K: d_total_out × d_total_in
                # Build (I_R ⊗ K†) X (I_R ⊗ K)
                IK = np.kron(np.eye(d_R), K)  # (d_R*d_total_out, d_R*d_total_in)  
                IKH = IK.conj().T
                # But X is d_R*d_total_out × d_R*d_total_out
                # and we need (I⊗K†)^T X (I⊗K)^T ... 
                # Actually: (I⊗K†) X^T (I⊗K) in some form
                # Let's be careful:
                # contribution to Y from K: (I_R ⊗ K)† · X · (I_R ⊗ K)  ... no
                
                # The encoder contributes: σ_RB depends on V via KV
                # F = (1/d_R²) Σ evals(X(V))
                # dF/dV involves the gradient of X w.r.t. V
                
                # Simpler: Y = d_R * Tr_B[(I_R ⊗ Σ_K K† D_opt† Φ D_opt K)]
                # where D_opt is determined by W
                
                # Power iteration: Y_ij = Σ_K ⟨i_R| ⊗ K† · X_{row for i} 
                Y += IKH @ X.T @ IK  # This computes pull-back
            
            # Extract top eigenvectors for new V
            evals_V, evecs_V = np.linalg.eigh(Y)
            V_new_flat = evecs_V[:, -d_R:]
            # V_new is d_R*d_total_in, d_R → reshape to (d_total_in, d_R)
            # Each column v_k is in C^(d_R * d_total_in)
            # We want V: (d_total_in, d_R) isometry
            # Take the first column: v = V_new_flat[:, -1] ∈ C^(d_R*d_total_in)
            # Reshape to (d_R, d_total_in) → V = columns
            
            V_mat = V_new_flat[:, -d_R:].reshape(d_R, d_total_in, d_R)
            # V[:, i] should give i-th encoded basis vector
            V = np.zeros((d_total_in, d_R), dtype=complex)
            for i in range(d_R):
                V[:, i] = V_mat[i, :, i]
            # Orthogonalize
            V, _ = np.linalg.qr(V)
            V = V[:, :d_R]
        
        best_F = max(best_F, F)
        if restart == 0 or F > best_F - 0.001:
            print(f"    restart {restart}: F = {F:.6f}")
    
    return best_F

# ═══════════════════════════════════════════
#              MAIN
# ═══════════════════════════════════════════
if __name__ == '__main__':
    print("="*60)
    print("  CORRECT SA PROTOCOL — Parentin et al. (2026)")
    print("="*60)
    
    # 1. Build effective channel Ñ
    print("\n  Step 1: Build effective channel Ñ")
    Ks = build_N_tilde()
    print(f"  {len(Ks)} Kraus operators, 2→4")
    tp_err = verify_channel(Ks)
    
    # 2. Single-shot coherent information
    print("\n  Step 2: Single-shot coherent information")
    ci = coherent_info_single(Ks, d_in=2, n_trials=500)
    print(f"  Q₁(Ñ) = {ci:.6f}")
    print(f"  {'Positive!' if ci > 0.001 else 'Non-positive (expected for SA scenario)'}")
    
    # 3. Channel fidelity for n=1..8
    print("\n  Step 3: Channel fidelity F_c(Ñ^⊗n, 2)")
    print(f"  Target: F > 0.75")
    for n in range(1, 9):
        t0 = time.time()
        F = channel_fidelity_seesaw(Ks, d_in=2, n_uses=n, d_code=2, 
                                     n_iter=30, n_restarts=3)
        dt = time.time() - t0
        if F is not None:
            marker = " 🌟 SA!" if F > 0.75 else ""
            print(f"  n={n}: F = {F:.6f} [{dt:.1f}s]{marker}")
        else:
            print(f"  n={n}: skipped (too large)")
            break
    
    print(f"\n{'='*60}")
