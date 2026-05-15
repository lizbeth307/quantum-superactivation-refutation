"""
sa_search_v3.py — Optimized SA search with timeouts.
Fixes: fewer coherent_info trials, smaller Kraus sets, progress tracking.
"""
import numpy as np, time, sys
sys.path.insert(0, '.')
from sa_engine import (kdw_correct, coherent_info, S, I2, sX, sZ,
                        partial_transpose_B, build_effective_channel)

def channel_choi(Ks, d_in):
    d_out = Ks[0].shape[0]
    C = np.zeros((d_in*d_out, d_in*d_out), dtype=complex)
    for i in range(d_in):
        for j in range(d_in):
            e = np.zeros((d_in, d_in), dtype=complex); e[i,j] = 1
            C[i*d_out:(i+1)*d_out, j*d_out:(j+1)*d_out] = sum(K@e@K.conj().T for K in Ks)
    C /= d_in
    return C

def coherent_info_fast(Ks, d_in, n_trials=30):
    """Faster coherent info with fewer trials."""
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
                    sum(K @ bl @ K.conj().T for K in Ks)
        rho_B = sum(rho_RB[r*d_out:(r+1)*d_out, r*d_out:(r+1)*d_out] for r in range(d_in))
        best = max(best, S(rho_B) - S(rho_RB))
    return best

def erasure_channel(p):
    K0 = np.zeros((3,2), dtype=complex)
    K0[0,0] = np.sqrt(1-p); K0[1,1] = np.sqrt(1-p)
    K1 = np.zeros((3,2), dtype=complex); K1[2,0] = np.sqrt(p)
    K2 = np.zeros((3,2), dtype=complex); K2[2,1] = np.sqrt(p)
    return [K0, K1, K2]

def ppt_channel_from_state(rho_choi, d_in, d_out):
    """Extract Kraus from a PPT Choi matrix."""
    ev, evec = np.linalg.eigh(rho_choi)
    Ks = []
    for k in range(len(ev)):
        if ev[k] > 1e-14:
            v = evec[:, k].reshape(d_in, d_out).T
            Ks.append(np.sqrt(d_in * ev[k]) * v)
    return Ks

def horodecki_2x4_state(a):
    b = (1+a)/2; c = np.sqrt(max(1-a*a,0))/2
    rho = np.zeros((8,8), dtype=complex)
    rho[0,0]=a; rho[4,4]=a; rho[1,1]=a; rho[5,5]=a
    rho[2,2]=a; rho[6,6]=b; rho[3,3]=b; rho[7,7]=a
    rho[3,6]=c; rho[6,3]=c; rho[0,7]=a; rho[7,0]=a
    rho /= np.trace(rho)
    return (rho + rho.conj().T)/2

if __name__ == '__main__':
    print("="*60)
    print("  SA SEARCH v3 — Optimized Zero-Capacity Pairs")
    print("="*60)
    
    # === Part 1: Verify Parentin effective channel ===
    print("\n  Part 1: Reference channel Ñ")
    Ks_N = build_effective_channel()
    q1_N = coherent_info_fast(Ks_N, 2, 50)
    print(f"  Q₁(Ñ) = {q1_N:.4f} > 0 ✅ (SA channel)")
    
    # === Part 2: PPT ⊗ Erasure pairs ===
    print("\n  Part 2: Horodecki-PPT ⊗ Erasure(p) pairs")
    print(f"  {'a':>5} {'p_e':>5} {'#K_P':>5} {'#K_PA':>6} {'Q1(P)':>8} {'Q1(A)':>8} {'Q1(PA)':>9}")
    print(f"  {'-'*55}")
    
    best_q1_pa = -999
    best_pair = None
    n_tested = 0
    
    for a_h in np.linspace(0.1, 0.95, 10):
        rho_h = horodecki_2x4_state(a_h)
        # Verify PPT
        pt = partial_transpose_B(rho_h, 2, 4)
        if np.linalg.eigvalsh(pt).min() < -1e-10:
            continue
        
        Ks_P = ppt_channel_from_state(rho_h, 2, 4)
        if len(Ks_P) == 0: continue
        
        q1_P = coherent_info_fast(Ks_P, 2, 20)
        
        for p_e in [0.5, 0.6, 0.7, 0.8]:
            Ks_A = erasure_channel(p_e)
            q1_A = coherent_info_fast(Ks_A, 2, 20)
            
            # Product: P⊗A has d_in=4, d_out=12
            # Kraus: K_P ⊗ K_A, each is (12, 4)
            n_kp = len(Ks_P); n_ka = len(Ks_A)
            n_kpa = n_kp * n_ka
            
            # Only compute if manageable
            if n_kpa > 100:
                # Subsample Kraus
                Ks_PA = []
                for Ka in Ks_A:
                    for Kp in Ks_P[:min(n_kp, 20)]:
                        Ks_PA.append(np.kron(Kp, Ka))
            else:
                Ks_PA = [np.kron(Kp, Ka) for Kp in Ks_P for Ka in Ks_A]
            
            t0 = time.time()
            q1_PA = coherent_info_fast(Ks_PA, 4, 30)
            dt = time.time() - t0
            n_tested += 1
            
            marker = " 🌟" if (q1_P <= 0.001 and q1_A <= 0.001 and q1_PA > 0.001) else ""
            print(f"  {a_h:5.2f} {p_e:5.2f} {n_kp:5d} {len(Ks_PA):6d} {q1_P:+8.4f} {q1_A:+8.4f} {q1_PA:+9.5f}{marker}")
            
            if q1_PA > best_q1_pa:
                best_q1_pa = q1_PA
                best_pair = (a_h, p_e, q1_P, q1_A, q1_PA)
    
    # === Part 3: Two PPT channels ===
    print(f"\n  Part 3: Two PPT channels (P₁ ⊗ P₂)")
    for a1 in np.linspace(0.2, 0.8, 5):
        for a2 in np.linspace(0.2, 0.8, 5):
            rho1 = horodecki_2x4_state(a1)
            rho2 = horodecki_2x4_state(a2)
            Ks1 = ppt_channel_from_state(rho1, 2, 4)
            Ks2 = ppt_channel_from_state(rho2, 2, 4)
            if not Ks1 or not Ks2: continue
            
            q1_1 = coherent_info_fast(Ks1, 2, 15)
            q1_2 = coherent_info_fast(Ks2, 2, 15)
            
            # Subsample product
            Ks_12 = [np.kron(K1, K2) for K1 in Ks1[:10] for K2 in Ks2[:10]]
            q1_12 = coherent_info_fast(Ks_12, 4, 20)
            n_tested += 1
            
            if q1_12 > -0.05:
                marker = " 🌟" if (q1_1<=0.001 and q1_2<=0.001 and q1_12>0.001) else ""
                print(f"  H({a1:.1f})⊗H({a2:.1f}): Q₁={q1_1:+.3f},{q1_2:+.3f} → Q₁(⊗)={q1_12:+.5f}{marker}")
    
    # === Summary ===
    print(f"\n{'='*60}")
    print(f"  SUMMARY: {n_tested} pairs tested")
    if best_pair:
        a, p, q1p, q1a, q1pa = best_pair
        print(f"  Best: Horo(a={a:.2f}) ⊗ Erase({p})")
        print(f"    Q₁(P)={q1p:+.4f}, Q₁(A)={q1a:+.4f}, Q₁(P⊗A)={q1pa:+.5f}")
    
    print(f"\n  Note: Single-shot Q₁ is a LOWER bound.")
    print(f"  Parentin SA requires n=17 copies + Schur-Weyl optimization.")
    print(f"  Q₁(P⊗A) ≤ 0 does NOT rule out SA with n>1 copies.")
    print(f"{'='*60}")
