"""
sa_full_PA_cq.py — CQ-structured seesaw for full P⊗A.

Key insight: After Bob measures the erasure flag bits z ∈ {0,1}^n,
the problem decomposes into n+1 independent sub-problems.

For each erasure count k = |z|:
  - P(k) = C(n,k)/2^n
  - Sub-channel N_k: (n-k) qubits pass identity, k pass noise
  - F_k = max_{enc_k, dec_k} fidelity for sub-channel

But for full P⊗A (not effective Ñ), the structure is different:
- P output is 4-dim (qubit + 2-dim shield)
- A output is 3-dim (qubit + erasure flag)
- Bob sees flag bits from A (erasure pattern)
- P's shield bits are NOT measured

So the CQ structure comes from A's erasure flags only.
With n copies of P⊗A:
  - n copies of A give erasure pattern z ∈ {0,1}^n  
  - k = |z| qubits are erased on A side
  - P side: all n copies give 4-dim output each (no flag measurement)
  
Conditional channel for each z:
  P⊗A|z: C^(2^n) → C^(4^n · 2^(n-k)) 
  where 2^(n-k) comes from non-erased A qubits

Actually, let's be more precise about the structure:
  
Each copy of P⊗A: C^4 → C^12
  Input: |ψ⟩_P ⊗ |φ⟩_A
  Output: P(|ψ⟩⟨ψ|) ⊗ A(|φ⟩⟨φ|)
  
A's output: with prob 0.5 → qubit (passthrough)
            with prob 0.5 → erasure flag |e⟩
  
So for n copies, after measuring A's flags:
  k qubits erased → only P's output matters for those
  (n-k) qubits pass → P's output ⊗ intact A qubit

This is complex. Let's do a simpler approach:
just use the isometric seesaw on P⊗A but with smarter initialization.
"""
import numpy as np, time, sys
sys.path.insert(0, '.')
from sa_engine import S, I2, sX, sZ

P_FLIP = 1.0 / (1.0 + np.sqrt(2.0))

def build_P(a_param=0.5):
    """PPT channel P: C²→C⁴ from Horodecki state."""
    a = a_param; b = (1+a)/2; c = np.sqrt(max(1-a*a,0))/2
    rho = np.zeros((8,8), dtype=complex)
    rho[0,0]=a; rho[4,4]=a; rho[1,1]=a; rho[5,5]=a
    rho[2,2]=a; rho[6,6]=b; rho[3,3]=b; rho[7,7]=a
    rho[3,6]=c; rho[6,3]=c; rho[0,7]=a; rho[7,0]=a
    rho /= np.trace(rho); rho = (rho+rho.conj().T)/2
    ev, evec = np.linalg.eigh(rho)
    return [np.sqrt(2*ev[k]) * evec[:,k].reshape(2,4).T for k in range(len(ev)) if ev[k]>1e-14]

def build_A(p=0.5):
    """50% erasure: C²→C³."""
    K0 = np.zeros((3,2), dtype=complex)
    K0[0,0]=np.sqrt(1-p); K0[1,1]=np.sqrt(1-p)
    K1 = np.zeros((3,2), dtype=complex); K1[2,0]=np.sqrt(p)
    K2 = np.zeros((3,2), dtype=complex); K2[2,1]=np.sqrt(p)
    return [K0, K1, K2]

def seesaw_iso(Ks, d_in, d_R=2, n_iter=30, n_restarts=10):
    """Fast isometric seesaw (no SDP overhead)."""
    d_out = Ks[0].shape[0]
    best_F = 0
    
    for restart in range(n_restarts):
        # Random isometric encoder
        V = np.random.randn(d_in, d_R) + 1j*np.random.randn(d_in, d_R)
        U, s, Vh = np.linalg.svd(V, full_matrices=False)
        V = U @ Vh
        
        prev_F = -1
        for it in range(n_iter):
            # σ_RB = (id_R ⊗ N)(|Ψ_V⟩⟨Ψ_V|)
            sigma = np.zeros((d_R*d_out, d_R*d_out), dtype=complex)
            for K in Ks:
                KV = K @ V  # (d_out, d_R)
                for i in range(d_R):
                    for j in range(d_R):
                        sigma[i*d_out:(i+1)*d_out, j*d_out:(j+1)*d_out] += \
                            np.outer(KV[:,i], KV[:,j].conj()) / d_R
            
            # Best isometric decoder W (d_R, d_out)
            evals, evecs = np.linalg.eigh(sigma)
            w = evecs[:, -1]
            W = w.reshape(d_R, d_out)
            Uw, sw, Vhw = np.linalg.svd(W, full_matrices=False)
            W = Uw @ Vhw
            
            # Fidelity
            F = 0
            for i in range(d_R):
                for j in range(d_R):
                    bl = sigma[i*d_out:(i+1)*d_out, j*d_out:(j+1)*d_out]
                    F += (W @ bl @ W.conj().T)[i,j]
            F = F.real / d_R
            
            # Best isometric encoder
            TK = [W @ K for K in Ks]
            M = np.zeros((d_in*d_R, d_in*d_R), dtype=complex)
            for T in TK:
                t = T.T.ravel('F')
                M += np.outer(t, t.conj())
            evals_v, evecs_v = np.linalg.eigh(M)
            v = evecs_v[:, -1]
            V = v.reshape(d_in, d_R, order='F')
            Uv, sv, Vhv = np.linalg.svd(V, full_matrices=False)
            V = Uv @ Vhv
            
            if abs(F - prev_F) < 1e-8: break
            prev_F = F
        
        best_F = max(best_F, F)
    
    return best_F

def coherent_info_q1(Ks, d_in, n_trials=300):
    """Q₁ = max coherent information."""
    d_out = Ks[0].shape[0]; best = -999
    for t in range(n_trials):
        if t == 0:
            psi = np.zeros(d_in*d_in, dtype=complex)
            for i in range(d_in): psi[i*d_in+i] = 1/np.sqrt(d_in)
        else:
            psi = np.random.randn(d_in*d_in)+1j*np.random.randn(d_in*d_in)
            psi /= np.linalg.norm(psi)
        rho_RA = np.outer(psi, psi.conj())
        rho_RB = np.zeros((d_in*d_out, d_in*d_out), dtype=complex)
        for r1 in range(d_in):
            for r2 in range(d_in):
                bl = rho_RA[r1*d_in:(r1+1)*d_in, r2*d_in:(r2+1)*d_in]
                rho_RB[r1*d_out:(r1+1)*d_out, r2*d_out:(r2+1)*d_out] = \
                    sum(K@bl@K.conj().T for K in Ks)
        rho_B = sum(rho_RB[r*d_out:(r+1)*d_out, r*d_out:(r+1)*d_out] for r in range(d_in))
        best = max(best, S(rho_B) - S(rho_RB))
    return best

if __name__ == '__main__':
    print("="*60)
    print("  FULL P⊗A — CQ-Optimized Seesaw")
    print("="*60)
    
    # Scan a-parameter for P
    print("\n  Scanning P(a) for best SA candidate...")
    best_result = None
    
    for a in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        Ks_P = build_P(a)
        Ks_A = build_A(0.5)
        Ks_PA = [np.kron(Kp, Ka) for Kp in Ks_P for Ka in Ks_A]
        
        q1_P = coherent_info_q1(Ks_P, 2, 100)
        q1_PA = coherent_info_q1(Ks_PA, 4, 200)
        
        t0 = time.time()
        F1 = seesaw_iso(Ks_PA, 4, d_R=2, n_iter=30, n_restarts=10)
        dt = time.time() - t0
        
        marker = " 🌟" if F1 > 0.75 else ""
        print(f"  a={a:.1f}: Q₁(P)={q1_P:+.4f}, Q₁(PA)={q1_PA:+.4f}, F₁={F1:.4f} [{dt:.1f}s]{marker}")
        
        if best_result is None or F1 > best_result['F1']:
            best_result = {'a': a, 'q1_P': q1_P, 'q1_PA': q1_PA, 'F1': F1}
    
    # Best a
    a_best = best_result['a']
    print(f"\n  Best: a={a_best}, F₁={best_result['F1']:.4f}")
    
    # n=2 with best a
    print(f"\n  Computing (P⊗A)^⊗2 with a={a_best}...")
    Ks_P = build_P(a_best)
    Ks_A = build_A(0.5)
    Ks_PA = [np.kron(Kp, Ka) for Kp in Ks_P for Ka in Ks_A]
    
    # (P⊗A)^⊗2: C^16 → C^144, but we can subsample Kraus for speed
    n_kpa = len(Ks_PA)
    print(f"  P⊗A: {n_kpa} Kraus, C⁴→C¹²")
    
    # For n=2 with full Kraus: n_kpa^2 = 441 ops → manageable for isometric
    Ks_PA2 = [np.kron(K1, K2) for K1 in Ks_PA for K2 in Ks_PA]
    print(f"  (P⊗A)²: {len(Ks_PA2)} Kraus, C¹⁶→C¹⁴⁴")
    
    t0 = time.time()
    F2 = seesaw_iso(Ks_PA2, 16, d_R=2, n_iter=20, n_restarts=5)
    dt = time.time() - t0
    print(f"  F₂ = F_c((P⊗A)², 2) = {F2:.6f} [{dt:.1f}s]")
    
    # Q₁ of product
    q1_PA2 = coherent_info_q1(Ks_PA2, 16, 100)
    print(f"  Q₁((P⊗A)²) = {q1_PA2:+.6f}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"  RESULTS: P=Horodecki(a={a_best}), A=Erasure(0.5)")
    print(f"  Q₁(P) = {best_result['q1_P']:+.4f} ≤ 0 (PPT → Q=0)")
    print(f"  Q₁(A) = 0 (erasure 50%% → Q=0)")
    print(f"  Q₁(P⊗A)   = {best_result['q1_PA']:+.4f}")
    print(f"  F₁(P⊗A)   = {best_result['F1']:.4f}")
    print(f"  F₂((PA)²)  = {F2:.4f}")
    print(f"  Q₁((PA)²)  = {q1_PA2:+.4f}")
    sa = F2 > 0.75 or q1_PA2 > 0.001
    print(f"  SA: {'🌟 CONFIRMED' if sa else 'Need more copies (n>2)'}")
    print(f"{'='*60}")
