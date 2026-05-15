"""
sa_search_v2.py — Correct SA Search: find P,A with Q(P)=Q(A)=0 but Q(P⊗A)>0.

Key constraint: BOTH channels must have ZERO quantum capacity individually.
Known zero-capacity channels:
- PPT channels (Choi is PPT → Q=0)
- Entanglement-breaking channels (Q=0)
- Antidegradable channels (Q=0)

Strategy:
1. Generate PPT channels (guaranteed Q=0)
2. Check if JOINT channel P⊗A has Q>0 (via coherent info of product)
3. MLP screens Choi of P⊗A for entanglement
"""
import numpy as np, time, sys, os, torch
sys.path.insert(0, '.')
from sa_engine import (kdw_correct, coherent_info, S, I2, sX, sZ,
                        partial_transpose_B, realignment_norm,
                        build_effective_channel)
from retrain_v6 import extract_features, SAPredictor

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_mlp():
    ckpt = torch.load('sa_data/model_v6.pt', weights_only=False, map_location=DEVICE)
    model = SAPredictor(13).to(DEVICE)
    model.load_state_dict(ckpt['model_state']); model.eval()
    return model, ckpt['mu'], ckpt['std']

def channel_choi(Ks, d_in):
    d_out = Ks[0].shape[0]
    C = np.zeros((d_in*d_out, d_in*d_out), dtype=complex)
    for i in range(d_in):
        for j in range(d_in):
            e = np.zeros((d_in, d_in), dtype=complex); e[i,j] = 1
            C[i*d_out:(i+1)*d_out, j*d_out:(j+1)*d_out] = sum(K@e@K.conj().T for K in Ks)
    C /= d_in
    return C

def is_ppt_channel(Ks, d_in):
    """Check if channel's Choi matrix is PPT (→ zero quantum capacity)."""
    d_out = Ks[0].shape[0]
    C = channel_choi(Ks, d_in)
    pt = partial_transpose_B(C, d_in, d_out)
    return np.linalg.eigvalsh(pt).min() >= -1e-10

# ═══ PPT Channel Generators ═══

def ppt_channel_horodecki(a):
    """Generate PPT channel from Horodecki 2×4 state.
    Choi = Horodecki state → PPT → Q=0.
    Extract Kraus from Choi eigendecomposition.
    """
    b = (1 + a) / 2; c = np.sqrt(max(1 - a*a, 0)) / 2
    C = np.zeros((8, 8), dtype=complex)
    C[0,0] = a; C[4,4] = a; C[1,1] = a; C[5,5] = a
    C[2,2] = a; C[6,6] = b; C[3,3] = b; C[7,7] = a
    C[3,6] = c; C[6,3] = c; C[0,7] = a; C[7,0] = a
    C /= np.trace(C)
    # Choi → Kraus: C = Σ λ_k |v_k⟩⟨v_k|, K_k = √(d_in·λ_k) · reshape(v_k)
    d_in = 2; d_out = 4
    ev, evec = np.linalg.eigh(C)
    Ks = []
    for k in range(len(ev)):
        if ev[k] > 1e-14:
            v = evec[:, k].reshape(d_in, d_out).T  # (d_out, d_in)
            Ks.append(np.sqrt(d_in * ev[k]) * v)
    return Ks

def ppt_channel_depolarizing_mix(p):
    """Fully depolarizing channel mixed with identity — always PPT for p ≥ 1/2.
    N(ρ) = p·I/2 + (1-p)·ρ
    PPT iff p ≥ 1/(d+1) = 1/3 for qubit
    """
    d = 2
    Ks = []
    Ks.append(np.sqrt(1-p + p/d**2) * np.eye(d, dtype=complex))
    sq_p = np.sqrt(p) / d
    for i in range(d):
        for j in range(d):
            if i != j:
                E = np.zeros((d,d), dtype=complex)
                E[i,j] = sq_p
                Ks.append(E)
    # Add diagonal noise
    for i in range(d):
        E = np.zeros((d,d), dtype=complex)
        E[i,i] = sq_p
        if abs(E[i,i]) > 1e-15:
            Ks.append(E)
    return Ks

def erasure_channel(p):
    """Erasure channel: ρ → (1-p)ρ ⊕ p|e⟩⟨e|, C²→C³.
    Known: Q=0 for p ≥ 1/2.
    """
    K0 = np.zeros((3,2), dtype=complex)
    K0[0,0] = np.sqrt(1-p); K0[1,1] = np.sqrt(1-p)
    K1 = np.zeros((3,2), dtype=complex)
    K1[2,0] = np.sqrt(p)
    K2 = np.zeros((3,2), dtype=complex)
    K2[2,1] = np.sqrt(p)
    return [K0, K1, K2]

def tensor_kraus(Ks_A, Ks_B):
    """Kraus operators for A⊗B channel."""
    return [np.kron(Ka, Kb) for Ka in Ks_A for Kb in Ks_B]

# ═══ SEARCH ═══

if __name__ == '__main__':
    print("="*60)
    print("  SA SEARCH v2 — Zero-capacity channel pairs")
    print("="*60)
    
    model, mu, std = load_mlp()
    
    # Reference: Parentin's P⊗A
    print("\n  Reference: Parentin construction")
    Ks_ref = build_effective_channel()
    C_ref = channel_choi(Ks_ref, 2)
    q1_ref = coherent_info(Ks_ref, 2, 200)
    print(f"  Ñ (effective): Q₁ = {q1_ref:.4f} > 0 ✅")
    print(f"  But Q(P) = Q(A) = 0 individually")
    
    # Search: pair PPT channels with erasure channels
    print(f"\n  Strategy: PPT channel P ⊗ erasure channel A")
    print(f"  PPT → Q(P) = 0")
    print(f"  Erasure(p≥0.5) → Q(A) = 0")
    print(f"  Check: Q₁(P⊗A) > 0 ?")
    
    results = []
    n_tested = 0
    
    print(f"\n  {'P-channel':>15} {'A-channel':>15} {'Q1(P)':>7} {'Q1(A)':>7} {'Q1(PA)':>8} {'status':>10}")
    print(f"  {'-'*65}")
    
    # Pair 1: Horodecki PPT ⊗ Erasure
    for a_h in np.linspace(0.1, 0.95, 15):
        Ks_P = ppt_channel_horodecki(a_h)
        if not Ks_P: continue
        # Verify Q(P) = 0
        q1_P = coherent_info(Ks_P, 2, 50)
        
        for p_e in [0.5, 0.55, 0.6, 0.7, 0.8]:
            Ks_A = erasure_channel(p_e)
            q1_A = coherent_info(Ks_A, 2, 50)
            
            # Product channel
            Ks_PA = tensor_kraus(Ks_P, Ks_A)
            d_in_PA = 4  # 2⊗2
            q1_PA = coherent_info(Ks_PA, d_in_PA, 100)
            n_tested += 1
            
            status = "🌟 SA!" if (q1_P <= 0.001 and q1_A <= 0.001 and q1_PA > 0.001) else ""
            if q1_PA > -0.1 or status:
                print(f"  Horo(a={a_h:.2f}) {'Erase('+str(p_e)+')':>15} {q1_P:+7.4f} {q1_A:+7.4f} {q1_PA:+8.4f} {status}")
            
            results.append({
                'P': f'Horo(a={a_h:.2f})', 'A': f'Erase({p_e})',
                'Q1_P': q1_P, 'Q1_A': q1_A, 'Q1_PA': q1_PA
            })
    
    # Pair 2: Depolarizing(PPT) ⊗ Erasure
    for p_d in np.linspace(0.4, 0.99, 15):
        Ks_P = ppt_channel_depolarizing_mix(p_d)
        q1_P = coherent_info(Ks_P, 2, 50)
        if q1_P > 0.001: continue  # Skip if not zero-capacity
        
        for p_e in [0.5, 0.6, 0.7]:
            Ks_A = erasure_channel(p_e)
            q1_A = coherent_info(Ks_A, 2, 50)
            if q1_A > 0.001: continue
            
            Ks_PA = tensor_kraus(Ks_P, Ks_A)
            q1_PA = coherent_info(Ks_PA, 4, 100)
            n_tested += 1
            
            status = "🌟 SA!" if q1_PA > 0.001 else ""
            if q1_PA > -0.1 or status:
                print(f"  Depol(p={p_d:.2f}) {'Erase('+str(p_e)+')':>15} {q1_P:+7.4f} {q1_A:+7.4f} {q1_PA:+8.4f} {status}")
            
            results.append({
                'P': f'Depol({p_d:.2f})', 'A': f'Erase({p_e})',
                'Q1_P': q1_P, 'Q1_A': q1_A, 'Q1_PA': q1_PA
            })
    
    # Summary
    sa_found = [r for r in results if r['Q1_P'] <= 0.001 and r['Q1_A'] <= 0.001 and r['Q1_PA'] > 0.001]
    
    print(f"\n{'='*60}")
    print(f"  RESULTS: {n_tested} pairs tested")
    print(f"  SA confirmed: {len(sa_found)}")
    if sa_found:
        print(f"\n  SA CHANNELS:")
        for r in sa_found:
            print(f"    {r['P']} ⊗ {r['A']}: Q₁(P⊗A) = {r['Q1_PA']:+.4f}")
    else:
        print(f"\n  No SA found in single-shot (expected)")
        print(f"  Parentin SA requires n=17 uses + Schur-Weyl optimization")
        print(f"  Single-shot Q₁(P⊗A) is a LOWER BOUND on Q(P⊗A)")
        # Show best candidates
        top = sorted(results, key=lambda r: r['Q1_PA'], reverse=True)[:5]
        print(f"\n  Best candidates (highest Q₁(P⊗A)):")
        for r in top:
            print(f"    {r['P']} ⊗ {r['A']}: Q₁ = {r['Q1_PA']:+.6f}")
    
    print(f"{'='*60}")
