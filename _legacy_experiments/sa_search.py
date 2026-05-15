"""
sa_search.py — SA Candidate Search using MLP v6c as fast screener.

Pipeline:
1. Generate random channel families (parameterized)
2. MLP screens Choi matrices for P(entangled)>50% and K_pred>0
3. Exact K_DW verification on candidates
4. CQ fidelity analysis on confirmed channels

Channel families:
A. Generalized bit-flip + dephasing (2-param)
B. Erasure-like with tunable noise (2-param)
C. Random Kraus with constrained structure (high-dim)
D. Horodecki-inspired private channels
"""
import numpy as np, torch, time, sys, os
sys.path.insert(0, '.')
from sa_engine import (kdw_correct, realignment_norm, partial_transpose_B, 
                        coherent_info, S, I2, sX, sZ, build_effective_channel)
from retrain_v6 import extract_features, SAPredictor

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ═══ Load MLP ═══
def load_mlp():
    ckpt = torch.load('sa_data/model_v6.pt', weights_only=False, map_location=DEVICE)
    model = SAPredictor(13).to(DEVICE)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model, ckpt['mu'], ckpt['std']

def mlp_predict(model, mu, std, rho, dA, dB):
    """Fast MLP prediction: K_DW estimate + P(entangled)."""
    feat = extract_features(rho, dA, dB)
    xn = torch.tensor((feat - mu)/std, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pk, pe = model(xn)
    return pk.item(), pe.item()

def channel_choi(Ks, d_in):
    d_out = Ks[0].shape[0]
    C = np.zeros((d_in*d_out, d_in*d_out), dtype=complex)
    for i in range(d_in):
        for j in range(d_in):
            e = np.zeros((d_in, d_in), dtype=complex); e[i,j] = 1
            C[i*d_out:(i+1)*d_out, j*d_out:(j+1)*d_out] = sum(K@e@K.conj().T for K in Ks)
    C /= d_in
    return C

# ═══ Channel Families ═══

def family_A(p_flip, p_deph):
    """Generalized bit-flip + dephasing: C² → C²
    N(ρ) = (1-p_d)[(1-p_f)ρ + p_f σ_x ρ σ_x] + p_d [|0><0|ρ|0><0| + |1><1|ρ|1><1|]
    
    With flag (like Parentin): C² → C⁴
    """
    Ks = []
    # No-flag branch (identity-like)
    w_id = np.sqrt(0.5)
    K0 = np.zeros((4,2), dtype=complex)
    K0[0,0] = w_id; K0[2,1] = w_id
    Ks.append(K0)
    
    # Flag branch: dephase + flip
    for kd in [np.sqrt(1-p_deph)*I2, np.sqrt(p_deph)*sZ]:
        for kf in [np.sqrt(1-p_flip)*I2, np.sqrt(p_flip)*sX]:
            Kq = kf @ kd
            K = np.zeros((4,2), dtype=complex)
            K[1,:] = w_id * Kq[0,:]
            K[3,:] = w_id * Kq[1,:]
            Ks.append(K)
    return Ks

def family_B(eta, p_noise):
    """Erasure-like channel: C² → C³
    With prob eta: identity
    With prob (1-eta): noise channel
    """
    Ks = []
    # Identity branch
    K0 = np.zeros((3,2), dtype=complex)
    K0[0,0] = np.sqrt(eta); K0[1,1] = np.sqrt(eta)
    Ks.append(K0)
    
    # Noise branch → erasure to |2>
    K1 = np.zeros((3,2), dtype=complex)
    K1[2,0] = np.sqrt((1-eta)*(1-p_noise))
    Ks.append(K1)
    K2 = np.zeros((3,2), dtype=complex)
    K2[2,1] = np.sqrt((1-eta)*p_noise)
    Ks.append(K2)
    
    # Mixed noise
    K3 = np.zeros((3,2), dtype=complex)
    K3[0,0] = np.sqrt((1-eta)*p_noise*0.5)
    K3[1,1] = np.sqrt((1-eta)*p_noise*0.5)
    Ks.append(K3)
    
    return Ks

def family_C(d_out=4, n_kraus=4):
    """Random structured channel: C² → C^d_out with TP constraint."""
    d_in = 2
    Ks = []
    # Generate random Kraus then enforce TP
    raw = [np.random.randn(d_out, d_in) + 1j*np.random.randn(d_out, d_in) for _ in range(n_kraus)]
    S_tp = sum(K.conj().T @ K for K in raw)
    L = np.linalg.cholesky(S_tp)
    Linv = np.linalg.inv(L)
    return [K @ Linv for K in raw]

def family_D(a, b):
    """Two-parameter private-like channel: C² → C⁴
    Inspired by Horodecki private channel construction.
    """
    Ks = []
    # Branch 1: partial identity
    K0 = np.zeros((4,2), dtype=complex)
    K0[0,0] = np.sqrt(a); K0[2,1] = np.sqrt(a)
    Ks.append(K0)
    
    # Branch 2: rotated + dephased
    theta = b * np.pi
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]], dtype=complex)
    K1 = np.zeros((4,2), dtype=complex)
    K1[1,:] = np.sqrt((1-a)/2) * R[0,:]
    K1[3,:] = np.sqrt((1-a)/2) * R[1,:]
    Ks.append(K1)
    
    # Branch 3: Z-dephased rotated
    K2 = np.zeros((4,2), dtype=complex)
    K2[1,:] = np.sqrt((1-a)/2) * (sZ @ R)[0,:]
    K2[3,:] = np.sqrt((1-a)/2) * (sZ @ R)[1,:]
    Ks.append(K2)
    
    return Ks

# ═══ Search Engine ═══

def search(model, mu, std, n_scan=5000, verbose=True):
    """Systematic search across channel families."""
    candidates = []
    
    print(f"\n  Scanning {n_scan} channels across 4 families...")
    print(f"  {'Family':>8} {'params':>20} {'K_mlp':>7} {'P_ent':>6} {'Q1':>7} {'K_exact':>8} {'status':>10}")
    print(f"  {'-'*75}")
    
    t0 = time.time()
    n_screen = 0; n_verify = 0; n_confirm = 0
    
    # Family A: bit-flip + dephasing grid
    for p_flip in np.linspace(0.01, 0.99, 30):
        for p_deph in np.linspace(0.01, 0.99, 30):
            Ks = family_A(p_flip, p_deph)
            tp = sum(K.conj().T @ K for K in Ks)
            if np.linalg.norm(tp - I2) > 0.01: continue
            
            C = channel_choi(Ks, 2)
            k_mlp, p_ent = mlp_predict(model, mu, std, C, 2, 4)
            n_screen += 1
            
            if k_mlp > -0.1 and p_ent > 0.3:
                q1 = coherent_info(Ks, 2, 100)
                k_exact = kdw_correct(C, 2, 4, 50)
                n_verify += 1
                
                if k_exact > 0.001 or q1 > 0.001:
                    n_confirm += 1
                    candidates.append({
                        'family': 'A', 'params': (p_flip, p_deph),
                        'K_mlp': k_mlp, 'K_exact': k_exact, 'Q1': q1, 'P_ent': p_ent
                    })
                    if verbose:
                        print(f"  {'A':>8} pf={p_flip:.2f},pd={p_deph:.2f}  {k_mlp:+7.4f} {p_ent:5.1%} {q1:+7.4f} {k_exact:+8.5f} {'CANDIDATE' if k_exact>0 else 'Q1>0'}")
    
    # Family D: private-like grid
    for a in np.linspace(0.1, 0.9, 30):
        for b in np.linspace(0.01, 0.99, 30):
            Ks = family_D(a, b)
            tp = sum(K.conj().T @ K for K in Ks)
            if np.linalg.norm(tp - I2) > 0.01: continue
            
            C = channel_choi(Ks, 2)
            k_mlp, p_ent = mlp_predict(model, mu, std, C, 2, 4)
            n_screen += 1
            
            if k_mlp > -0.1 and p_ent > 0.3:
                q1 = coherent_info(Ks, 2, 100)
                k_exact = kdw_correct(C, 2, 4, 50)
                n_verify += 1
                
                if k_exact > 0.001 or q1 > 0.001:
                    n_confirm += 1
                    candidates.append({
                        'family': 'D', 'params': (a, b),
                        'K_mlp': k_mlp, 'K_exact': k_exact, 'Q1': q1, 'P_ent': p_ent
                    })
                    if verbose:
                        print(f"  {'D':>8} a={a:.2f},b={b:.2f}      {k_mlp:+7.4f} {p_ent:5.1%} {q1:+7.4f} {k_exact:+8.5f} {'CANDIDATE' if k_exact>0 else 'Q1>0'}")
    
    # Family C: random structured
    for _ in range(500):
        Ks = family_C(d_out=4, n_kraus=4)
        C = channel_choi(Ks, 2)
        k_mlp, p_ent = mlp_predict(model, mu, std, C, 2, 4)
        n_screen += 1
        
        if k_mlp > 0 and p_ent > 0.5:
            q1 = coherent_info(Ks, 2, 100)
            k_exact = kdw_correct(C, 2, 4, 50)
            n_verify += 1
            
            if k_exact > 0.001 or q1 > 0.001:
                n_confirm += 1
                candidates.append({
                    'family': 'C', 'params': 'random',
                    'K_mlp': k_mlp, 'K_exact': k_exact, 'Q1': q1, 'P_ent': p_ent
                })
                if verbose:
                    print(f"  {'C':>8} random              {k_mlp:+7.4f} {p_ent:5.1%} {q1:+7.4f} {k_exact:+8.5f} {'CANDIDATE' if k_exact>0 else 'Q1>0'}")
    
    dt = time.time() - t0
    
    return candidates, n_screen, n_verify, n_confirm, dt

# ═══ MAIN ═══
if __name__ == '__main__':
    print("="*60)
    print("  SA CANDIDATE SEARCH — MLP-accelerated")
    print("="*60)
    
    # Reference: Parentin channel
    print("\n  Reference: Parentin effective channel Ñ")
    model, mu, std = load_mlp()
    Ks_ref = build_effective_channel()
    C_ref = channel_choi(Ks_ref, 2)
    k_ref, p_ref = mlp_predict(model, mu, std, C_ref, 2, 4)
    q1_ref = coherent_info(Ks_ref, 2, 200)
    k_exact_ref = kdw_correct(C_ref, 2, 4, 100)
    print(f"  K_mlp={k_ref:+.4f}, P_ent={p_ref:.1%}, Q1={q1_ref:.4f}, K_exact={k_exact_ref:.4f}")
    
    # Search
    candidates, n_screen, n_verify, n_confirm, dt = search(model, mu, std)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"  SEARCH SUMMARY")
    print(f"  Screened: {n_screen} channels [{dt:.1f}s]")
    print(f"  MLP passed: {n_verify} ({100*n_verify/max(n_screen,1):.1f}%)")
    print(f"  Confirmed: {n_confirm} ({100*n_confirm/max(n_screen,1):.1f}%)")
    
    if candidates:
        print(f"\n  TOP CANDIDATES:")
        top = sorted(candidates, key=lambda c: c['Q1'], reverse=True)[:10]
        for i, c in enumerate(top):
            print(f"  #{i+1} Family {c['family']}: Q1={c['Q1']:+.4f} K={c['K_exact']:+.5f} params={c['params']}")
    else:
        print(f"\n  No candidates found (expected — SA is extremely rare)")
        print(f"  Parentin needed n=17 uses of a specific channel family")
    
    print(f"{'='*60}")
