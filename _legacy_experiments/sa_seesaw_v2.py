"""
sa_seesaw_v2.py — Correct seesaw for channel fidelity.

Standard Reimpell-Werner seesaw:
1. Fix encoder V (isometry d_R → d_in^n)
2. Compute σ_RB = (id_R ⊗ N^n)(Φ_V) where Φ_V = encoded Bell state
3. Optimal decoder: maximize F(Φ_d, (id⊗D)(σ_RB))
   → D_opt = argmax Tr(D · M) s.t. D CPTP
   → For isometric decoder W: W^⊗n → SVD of conditional state
4. Fix decoder, optimize encoder similarly
"""
import numpy as np, time

I2 = np.eye(2, dtype=complex)
sX = np.array([[0,1],[1,0]], dtype=complex)
sZ = np.array([[1,0],[0,-1]], dtype=complex)

def S(rho):
    e = np.linalg.eigvalsh(rho); e = e[e>1e-15]
    return -np.sum(e*np.log2(e)) if len(e)>0 else 0.0

def build_N_tilde():
    p = 1.0 / (1.0 + np.sqrt(2.0))
    Ks = []
    # Z=0 (identity, prob 1/2)
    K0 = np.zeros((4,2), dtype=complex)
    K0[0,0] = np.sqrt(0.5); K0[2,1] = np.sqrt(0.5)
    Ks.append(K0)
    # Z=1 (dephase + bit-flip, prob 1/2)
    for kd in [np.sqrt(0.5)*I2, np.sqrt(0.5)*sZ]:
        for kx in [np.sqrt(1-p)*I2, np.sqrt(p)*sX]:
            Kq = kx @ kd
            K = np.zeros((4,2), dtype=complex)
            K[1,:] = np.sqrt(0.5)*Kq[0,:]
            K[3,:] = np.sqrt(0.5)*Kq[1,:]
            Ks.append(K)
    return Ks

def build_n_kraus(Ks, n):
    """Build Kraus for N^⊗n."""
    if n == 1: return Ks
    result = Ks
    for _ in range(n-1):
        new = [np.kron(K1,K2) for K1 in result for K2 in Ks]
        result = new
    return result

def seesaw_fidelity(Ks_n, d_in, d_out, d_R=2, n_iter=40, n_restarts=10):
    """Seesaw for channel fidelity F_c(N, d_R).
    
    The entanglement fidelity:
    F = max_{V,W} (1/d_R) |Tr(W† N_choi V)|² ... no
    
    Actually: F_c = max_{enc,dec} F(Φ_d, (id⊗dec∘N∘enc)(Φ_d))
    
    For isometric encoder V: C^d_R → C^d_in and decoder W: C^d_out → C^d_R:
    
    F = (1/d_R²) |Σ_K Tr(W† K V)|²  ... no, this is for entanglement fidelity
    
    Actually for entanglement fidelity with isometric encoder V and decoder W:
    F_e = (1/d_R²) |Σ_K Tr(W† K V)|² only if there's one Kraus op.
    
    General formula:
    F_e(V,W) = (1/d_R) Σ_i ⟨Φ|(I_R⊗W) K_i V ⊗ I_R |Φ⟩|² ... still not right.
    
    Let me be precise:
    |Φ⟩ = (1/√d_R) Σ_j |j⟩_R |j⟩_A
    Encoded: (I_R ⊗ V)|Φ⟩ = (1/√d_R) Σ_j |j⟩_R V|j⟩
    After channel: σ_RB = Σ_K (I_R⊗K) V|Φ⟩⟨Φ|V† (I_R⊗K†)
    After decoder: τ_RR' = (I_R⊗W) σ_RB (I_R⊗W†)
    F = ⟨Φ_RR'|τ_RR'|Φ_RR'⟩ = (1/d_R) Σ_{ij} τ[i,i,j,j]
    where τ is in R⊗R' with both dim d_R
    
    = (1/d_R²) Σ_K |Tr(W† K V)|²
    
    YES! This is the correct formula for isometric enc/dec.
    """
    best_F = 0
    
    for restart in range(n_restarts):
        # Random isometric encoder V: d_R → d_in
        V = np.random.randn(d_in, d_R) + 1j*np.random.randn(d_in, d_R)
        V, _ = np.linalg.qr(V); V = V[:,:d_R]
        
        F_prev = 0
        for it in range(n_iter):
            # === Fix V, optimize W ===
            # F = (1/d_R²) Σ_K |Tr(W† K V)|²
            # = (1/d_R²) Σ_K Tr(W† K V V† K† W)
            # = (1/d_R²) Tr(W† M W)  where M = Σ_K (KV)(KV)†
            # Wait, that's wrong. Let me redo:
            # |Tr(W† KV)|² = Tr((W†KV)† W†KV) = Tr(V†K†W W†KV)
            # Summing: Σ_K Tr(V†K†W W†KV) = Tr(W†(Σ_K KV V†K†)W) 
            # Hmm, still complicated for matrix W.
            
            # Better: define A_K = W† K V which is d_R × d_R
            # F = (1/d_R²) Σ_K |Tr(A_K)|²
            # This is maximized when W† K V ∝ I for dominant K.
            
            # Standard approach: define M = Σ_K (KV)⊗(KV)*
            # Then F = (1/d_R²) vec(W†)† M vec(W†)
            # where vec is column-major vectorization.
            
            # M is d_out*d_R × d_out*d_R
            # Actually: Tr(W†KV) = vec(W)† vec(KV) where vec stacks columns
            # |Tr(W†KV)|² = |vec(W)†vec(KV)|²
            # Σ_K = vec(W)† (Σ_K vec(KV)vec(KV)†) vec(W)
            
            KV_list = [K @ V for K in Ks_n]
            M = sum(np.outer(KV.ravel('F'), KV.ravel('F').conj()) for KV in KV_list)
            
            # Maximize vec(W)† M vec(W) s.t. W†W = I
            # Top eigenvector of M gives optimal vec(W)
            evals, evecs = np.linalg.eigh(M)
            w_opt = evecs[:, -1]  # top eigenvector
            W = w_opt.reshape(d_out, d_R, order='F')
            # Ensure W is isometric (it should be approximately)
            U, s, Vh = np.linalg.svd(W, full_matrices=False)
            W = U @ Vh  # closest isometry
            
            F_dec = evals[-1] / d_R**2
            
            # === Fix W, optimize V ===
            # F = (1/d_R²) Σ_K |Tr(W†KV)|²
            # = (1/d_R²) vec(V)† (Σ_K vec(W†K)vec(W†K)†) vec(V)
            
            WK_list = [W.conj().T @ K for K in Ks_n]
            M2 = sum(np.outer(WK.ravel('F'), WK.ravel('F').conj()) for WK in WK_list)
            
            evals2, evecs2 = np.linalg.eigh(M2)
            v_opt = evecs2[:, -1]
            V = v_opt.reshape(d_in, d_R, order='F')
            U2, s2, Vh2 = np.linalg.svd(V, full_matrices=False)
            V = U2 @ Vh2
            
            F_enc = evals2[-1] / d_R**2
            
            if abs(F_enc - F_prev) < 1e-12:
                break
            F_prev = F_enc
        
        F_final = F_enc
        best_F = max(best_F, F_final)
    
    return best_F

# ═══ MAIN ═══
if __name__ == '__main__':
    print("="*60)
    print("  SA SEESAW v2 — Correct Entanglement Fidelity")
    print("="*60)
    
    Ks = build_N_tilde()
    tp = sum(K.conj().T@K for K in Ks)
    print(f"  Channel Ñ: {len(Ks)} Kraus, 2→4, TP err={np.linalg.norm(tp-I2):.2e}")
    
    # Coherent info
    from sa_correct import coherent_info_single
    ci = coherent_info_single(Ks, 2, 500)
    print(f"  Q₁(Ñ) = {ci:.6f} {'✅ >0' if ci>0 else ''}")
    
    # Channel fidelity for n=1..7
    print(f"\n  {'n':>3} {'d_in':>6} {'d_out':>6} {'#K':>8} {'F':>10} {'time':>6} status")
    print(f"  {'-'*55}")
    
    for n in range(1, 8):
        d_in = 2**n
        d_out = 4**n
        nk = 5**n
        
        if d_out > 1024:
            print(f"  {n:>3} {d_in:>6} {d_out:>6} {nk:>8} {'---':>10} {'---':>6} too large")
            continue
        
        Ks_n = build_n_kraus(Ks, n)
        t0 = time.time()
        F = seesaw_fidelity(Ks_n, d_in, d_out, d_R=2, n_iter=50, n_restarts=10)
        dt = time.time() - t0
        
        status = "🌟 SA!" if F > 0.75 else ("promising" if F > 0.5 else "")
        print(f"  {n:>3} {d_in:>6} {d_out:>6} {nk:>8} {F:>10.6f} {dt:>5.1f}s {status}")
    
    print(f"\n{'='*60}")
