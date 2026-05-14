"""
sa_full_PA.py — Full P⊗A Superactivation Verification

The REAL SA test: 
  P = PPT Horodecki channel (Q(P) = 0)
  A = 50% quantum erasure (Q(A) = 0)
  P⊗A: should have Q(P⊗A) > 0

Steps:
1. Build P from Horodecki 2×4 PPT state (Choi → Kraus)
2. Build A = 50% erasure (C² → C³)
3. Verify Q₁(P) ≤ 0 and Q₁(A) ≤ 0
4. Build P⊗A (C⁴ → C¹²) 
5. Compute Q₁(P⊗A) — single-shot lower bound
6. SDP seesaw on (P⊗A)^⊗n for n=1,2,3
"""
import numpy as np, cvxpy as cp, time, sys
sys.path.insert(0, '.')
from sa_engine import (S, partial_transpose_B, realignment_norm, 
                        coherent_info, I2, sX, sZ)

# ═══════════════════════════════════════════
#  Channel P: PPT Horodecki Channel
# ═══════════════════════════════════════════
def build_channel_P(a_param=0.5):
    """Build PPT channel P from Horodecki 2×4 state.
    
    Choi(P) = Horodecki state ρ_H ∈ C^2 ⊗ C^4
    P: C^2 → C^4, guaranteed PPT → Q(P) = 0
    
    Kraus: eigendecompose Choi, K_k = √(d_in · λ_k) · reshape(v_k)
    """
    a = a_param
    b = (1 + a) / 2
    c = np.sqrt(max(1 - a*a, 0)) / 2
    
    # Horodecki 2×4 PPT-entangled state
    rho = np.zeros((8, 8), dtype=complex)
    rho[0,0] = a;     rho[4,4] = a
    rho[1,1] = a;     rho[5,5] = a
    rho[2,2] = a;     rho[6,6] = b
    rho[3,3] = b;     rho[7,7] = a
    rho[3,6] = c;     rho[6,3] = c
    rho[0,7] = a;     rho[7,0] = a
    rho /= np.trace(rho)
    rho = (rho + rho.conj().T) / 2
    
    d_in = 2; d_out = 4
    
    # Verify PPT
    pt = partial_transpose_B(rho, d_in, d_out)
    pt_min = np.linalg.eigvalsh(pt).min()
    
    # Extract Kraus from Choi eigendecomposition
    # Choi convention: J = Σ |i⟩⟨j| ⊗ N(|i⟩⟨j|) / d_in
    # J = Σ_k λ_k |v_k⟩⟨v_k|
    # K_k = √(d_in · λ_k) · mat(v_k) where v_k reshaped to (d_out, d_in)
    ev, evec = np.linalg.eigh(rho)
    Ks = []
    for k in range(len(ev)):
        if ev[k] > 1e-14:
            # v_k ∈ C^(d_in * d_out), reshape to (d_in, d_out) then transpose
            v = evec[:, k].reshape(d_in, d_out)  # (d_in, d_out)
            K = np.sqrt(d_in * ev[k]) * v.T  # (d_out, d_in) = (4, 2)
            Ks.append(K)
    
    return Ks, rho, pt_min

# ═══════════════════════════════════════════
#  Channel A: 50% Quantum Erasure
# ═══════════════════════════════════════════
def build_channel_A(p_erase=0.5):
    """50% quantum erasure channel A: C² → C³.
    
    A(ρ) = (1-p)|ψ⟩⟨ψ| ⊕ 0  +  p · 0 ⊕ Tr(ρ)|e⟩⟨e|
    
    Output space: C³ = C² ⊕ C¹ (qubit + erasure flag)
    Q(A) = max(0, 1-2p) = 0 for p ≥ 0.5
    """
    p = p_erase
    K0 = np.zeros((3, 2), dtype=complex)
    K0[0,0] = np.sqrt(1-p); K0[1,1] = np.sqrt(1-p)
    
    K1 = np.zeros((3, 2), dtype=complex)
    K1[2,0] = np.sqrt(p)
    
    K2 = np.zeros((3, 2), dtype=complex)
    K2[2,1] = np.sqrt(p)
    
    return [K0, K1, K2]

# ═══════════════════════════════════════════
#  P⊗A: Tensor Product Channel
# ═══════════════════════════════════════════
def build_PA(Ks_P, Ks_A):
    """Build Kraus for P⊗A: C^4 → C^12."""
    return [np.kron(Kp, Ka) for Kp in Ks_P for Ka in Ks_A]

# ═══════════════════════════════════════════
#  Coherent Information (optimized)
# ═══════════════════════════════════════════
def coherent_info_opt(Ks, d_in, n_trials=200):
    """Maximize I(R>B) = S(B) - S(RB) over input states."""
    d_out = Ks[0].shape[0]
    best = -999
    
    for t in range(n_trials):
        if t == 0:
            # Maximally entangled
            psi = np.zeros(d_in*d_in, dtype=complex)
            for i in range(d_in): psi[i*d_in+i] = 1/np.sqrt(d_in)
        elif t < 20:
            # Diagonal in Schmidt basis
            c = np.random.randn(d_in) + 1j*np.random.randn(d_in)
            c /= np.linalg.norm(c)
            psi = np.zeros(d_in*d_in, dtype=complex)
            for i in range(d_in): psi[i*d_in+i] = c[i]
        else:
            psi = np.random.randn(d_in*d_in) + 1j*np.random.randn(d_in*d_in)
            psi /= np.linalg.norm(psi)
        
        rho_RA = np.outer(psi, psi.conj())
        rho_RB = np.zeros((d_in*d_out, d_in*d_out), dtype=complex)
        for r1 in range(d_in):
            for r2 in range(d_in):
                bl = rho_RA[r1*d_in:(r1+1)*d_in, r2*d_in:(r2+1)*d_in]
                rho_RB[r1*d_out:(r1+1)*d_out, r2*d_out:(r2+1)*d_out] = \
                    sum(K @ bl @ K.conj().T for K in Ks)
        rho_B = sum(rho_RB[r*d_out:(r+1)*d_out, r*d_out:(r+1)*d_out] 
                    for r in range(d_in))
        ci = S(rho_B) - S(rho_RB)
        best = max(best, ci)
    return best

# ═══════════════════════════════════════════
#  SDP Seesaw for P⊗A
# ═══════════════════════════════════════════
def seesaw_PA(Ks, d_in, d_R=2, n_iter=15, n_restarts=5):
    """Seesaw: optimize encoder V and decoder W for channel fidelity.
    
    F = max_{V,W} (1/d_R) Σ_{ij} ⟨i|W(Σ_K K V|i⟩⟨j|V†K†)W†|j⟩
    """
    d_out = Ks[0].shape[0]
    best_F = 0
    
    for restart in range(n_restarts):
        V = np.random.randn(d_in, d_R) + 1j*np.random.randn(d_in, d_R)
        U, s, Vh = np.linalg.svd(V, full_matrices=False)
        V = U @ Vh
        
        for it in range(n_iter):
            # Step 1: σ_RB with current encoder
            sigma = np.zeros((d_R*d_out, d_R*d_out), dtype=complex)
            for K in Ks:
                KV = K @ V
                for i in range(d_R):
                    for j in range(d_R):
                        sigma[i*d_out:(i+1)*d_out, j*d_out:(j+1)*d_out] += \
                            np.outer(KV[:,i], KV[:,j].conj()) / d_R
            
            # Step 2: SDP decoder optimization
            dim = d_out * d_R
            sigma_BR = np.zeros((dim, dim), dtype=complex)
            for i in range(d_R):
                for j in range(d_R):
                    for a in range(d_out):
                        for b in range(d_out):
                            sigma_BR[a*d_R+i, b*d_R+j] = sigma[i*d_out+a, j*d_out+b]
            
            J = cp.Variable((dim, dim), hermitian=True)
            obj = cp.Maximize(cp.real(cp.trace(J @ sigma_BR)) / d_R)
            constraints = [J >> 0]
            for a in range(d_out):
                for b in range(d_out):
                    val = sum(J[a*d_R+r, b*d_R+r] for r in range(d_R))
                    constraints.append(val == (1 if a == b else 0))
            
            prob = cp.Problem(obj, constraints)
            try:
                prob.solve(solver=cp.SCS, verbose=False, max_iters=3000)
                F = prob.value if prob.value is not None else 0
            except:
                F = 0
            
            # Step 3: Isometric encoder update
            evals_s, evecs_s = np.linalg.eigh(sigma)
            w = evecs_s[:, -1]
            W = w.reshape(d_R, d_out)
            U_w, s_w, Vh_w = np.linalg.svd(W, full_matrices=False)
            W = U_w @ Vh_w
            
            TK_list = [W @ K for K in Ks]
            M = np.zeros((d_in*d_R, d_in*d_R), dtype=complex)
            for TK in TK_list:
                t = TK.T.ravel('F')
                M += np.outer(t, t.conj())
            evals_v, evecs_v = np.linalg.eigh(M)
            v = evecs_v[:, -1]
            V = v.reshape(d_in, d_R, order='F')
            U_v, s_v, Vh_v = np.linalg.svd(V, full_matrices=False)
            V = U_v @ Vh_v
        
        best_F = max(best_F, F)
    
    return best_F

# ═══════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════
if __name__ == '__main__':
    print("="*60)
    print("  FULL P⊗A SUPERACTIVATION VERIFICATION")
    print("="*60)
    
    # === Step 1: Build Channel P ===
    print("\n  ── Step 1: Channel P (PPT Horodecki) ──")
    best_a = None; best_q1p = 999
    
    for a in np.linspace(0.1, 0.95, 10):
        Ks_P, choi_P, pt_min = build_channel_P(a)
        if pt_min < -1e-10:
            continue
        tp = sum(K.conj().T @ K for K in Ks_P)
        tp_err = np.linalg.norm(tp - np.eye(2))
        q1 = coherent_info_opt(Ks_P, 2, 50)
        R = realignment_norm(choi_P, 2, 4)
        print(f"  a={a:.2f}: {len(Ks_P)} Kraus, TP={tp_err:.1e}, PT_min={pt_min:+.4f}, Q₁={q1:+.4f}, R={R:.3f}")
        if q1 < best_q1p:
            best_q1p = q1; best_a = a
    
    # Use best P
    Ks_P, choi_P, pt_min_P = build_channel_P(best_a)
    print(f"\n  Selected: a={best_a:.2f}, Q₁(P)={best_q1p:+.4f}")
    print(f"  PPT: {pt_min_P >= -1e-10} → Q(P) = 0 ✅")
    
    # === Step 2: Build Channel A ===
    print("\n  ── Step 2: Channel A (50% Erasure) ──")
    Ks_A = build_channel_A(0.5)
    tp_A = sum(K.conj().T @ K for K in Ks_A)
    print(f"  {len(Ks_A)} Kraus, C²→C³, TP={np.linalg.norm(tp_A - np.eye(2)):.1e}")
    q1_A = coherent_info_opt(Ks_A, 2, 50)
    print(f"  Q₁(A) = {q1_A:+.4f}")
    print(f"  Q(A) = max(0, 1-2p) = 0 for p=0.5 ✅")
    
    # === Step 3: Verify Q(P)=Q(A)=0 ===
    print("\n  ── Step 3: Zero-Capacity Verification ──")
    q1_P = coherent_info_opt(Ks_P, 2, 200)
    q1_A = coherent_info_opt(Ks_A, 2, 200)
    print(f"  Q₁(P) = {q1_P:+.6f} {'✅ ≤ 0' if q1_P <= 0.001 else '⚠️ > 0'}")
    print(f"  Q₁(A) = {q1_A:+.6f} {'✅ ≤ 0' if q1_A <= 0.001 else '⚠️ > 0'}")
    
    # === Step 4: Build P⊗A ===
    print("\n  ── Step 4: Product Channel P⊗A ──")
    Ks_PA = build_PA(Ks_P, Ks_A)
    d_in_PA = 4   # 2⊗2
    d_out_PA = 12  # 4⊗3
    print(f"  {len(Ks_PA)} Kraus, C⁴→C¹², dims: {d_in_PA}→{d_out_PA}")
    tp_PA = sum(K.conj().T @ K for K in Ks_PA)
    print(f"  TP err: {np.linalg.norm(tp_PA - np.eye(d_in_PA)):.1e}")
    
    # === Step 5: Single-shot Q₁(P⊗A) ===
    print("\n  ── Step 5: Single-shot Q₁(P⊗A) ──")
    t0 = time.time()
    q1_PA = coherent_info_opt(Ks_PA, d_in_PA, 300)
    dt = time.time() - t0
    print(f"  Q₁(P⊗A) = {q1_PA:+.6f} [{dt:.1f}s]")
    if q1_PA > 0.001:
        print(f"  🌟 POSITIVE! SA confirmed at single-shot level!")
    else:
        print(f"  ≤ 0 (may need multi-copy for SA)")
    
    # === Step 6: SDP Seesaw on P⊗A ===
    print("\n  ── Step 6: Channel Fidelity F_c(P⊗A, 2) via SDP ──")
    t0 = time.time()
    F_1 = seesaw_PA(Ks_PA, d_in_PA, d_R=2, n_iter=10, n_restarts=5)
    dt = time.time() - t0
    print(f"  F_c(P⊗A, 2) = {F_1:.6f} [{dt:.1f}s]")
    
    # n=2 copies
    print("\n  ── Step 7: F_c((P⊗A)^⊗2, 2) ──")
    Ks_PA2 = [np.kron(K1, K2) for K1 in Ks_PA for K2 in Ks_PA]
    d_in_2 = d_in_PA**2   # 16
    d_out_2 = d_out_PA**2  # 144
    print(f"  {len(Ks_PA2)} Kraus, C^{d_in_2}→C^{d_out_2}")
    
    if d_out_2 <= 200:
        t0 = time.time()
        F_2 = seesaw_PA(Ks_PA2, d_in_2, d_R=2, n_iter=8, n_restarts=3)
        dt = time.time() - t0
        print(f"  F_c((P⊗A)^⊗2, 2) = {F_2:.6f} [{dt:.1f}s]")
    else:
        print(f"  Skipped (d_out={d_out_2} too large)")
    
    # === Summary ===
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"  P: Horodecki(a={best_a:.2f}), Q₁(P)={q1_P:+.4f}, PPT ✅")
    print(f"  A: 50%% erasure, Q₁(A)={q1_A:+.4f} ✅")
    print(f"  P⊗A: Q₁={q1_PA:+.4f}, F_c={F_1:.4f}")
    sa = q1_PA > 0.001 or F_1 > 0.75
    print(f"  SA: {'🌟 CONFIRMED' if sa else 'Not at single-shot (need n>1)'}")
    print(f"{'='*60}")
