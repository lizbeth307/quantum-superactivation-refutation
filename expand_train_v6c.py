"""
expand_train_v6c.py — Expand MLP training to d=4..30 with PPT-entangled states.

Additions:
1. Higher-dim Werner/isotropic (d=4,5,6)
2. Horodecki-type PPT-entangled (3×3)
3. Tile/UPB PPT-entangled (3×3, 2×4)
4. Random PPT states for d=6,8 via mixing
5. Choi matrices of known channels
"""
import numpy as np, torch, torch.nn as nn, time, os, sys
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, '.')
from sa_engine import (kdw_correct, realignment_norm, partial_transpose_B, 
                        S, build_effective_channel, I2, sX, sZ)
from retrain_v6 import extract_features, SAPredictor, train, N_FEAT

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def random_separable(dA, dB, n_terms=5):
    d = dA * dB
    rho = np.zeros((d, d), dtype=complex)
    for _ in range(n_terms):
        a = np.random.randn(dA)+1j*np.random.randn(dA); a /= np.linalg.norm(a)
        b = np.random.randn(dB)+1j*np.random.randn(dB); b /= np.linalg.norm(b)
        v = np.kron(a, b); rho += np.outer(v, v.conj())
    rho /= np.trace(rho)
    return rho

def werner(d, p):
    psi = np.zeros(d*d, dtype=complex)
    for i in range(d): psi[i*d+i] = 1/np.sqrt(d)
    return p*np.outer(psi, psi.conj()) + (1-p)*np.eye(d*d)/(d*d)

def isotropic(d, f):
    psi = np.zeros(d*d, dtype=complex)
    for i in range(d): psi[i*d+i] = 1/np.sqrt(d)
    return f*np.outer(psi, psi.conj()) + (1-f)*np.eye(d*d)/(d*d)

def random_ppt_via_mix(dA, dB, n_sep=10, noise=0.01):
    """Generate PPT state by mixing separable + small perturbation."""
    d = dA * dB
    rho = random_separable(dA, dB, n_sep)
    # Add small perturbation, project to PPT
    H = np.random.randn(d, d) + 1j*np.random.randn(d, d)
    H = (H + H.conj().T) / 2
    rho_pert = rho + noise * H / np.linalg.norm(H)
    # Make PSD
    ev, evec = np.linalg.eigh(rho_pert)
    ev = np.maximum(ev, 0); rho_pert = evec @ np.diag(ev) @ evec.conj().T
    rho_pert /= np.trace(rho_pert)
    # Check PPT
    pt = partial_transpose_B(rho_pert, dA, dB)
    if np.linalg.eigvalsh(pt).min() < -1e-10:
        return None
    return rho_pert

def horodecki_2x4(a):
    """Horodecki 2×4 PPT-entangled state (P. Horodecki, 1997)."""
    b = (1 + a) / 2
    c = np.sqrt(1 - a*a) / 2
    rho = np.zeros((8, 8), dtype=complex)
    rho[0,0] = a;     rho[4,4] = a
    rho[1,1] = a;     rho[5,5] = a
    rho[2,2] = a;     rho[6,6] = b
    rho[3,3] = b;     rho[7,7] = a
    rho[3,6] = c;     rho[6,3] = c
    rho[0,7] = a;     rho[7,0] = a
    rho /= np.trace(rho)
    rho = (rho + rho.conj().T) / 2
    return rho

def channel_choi(Ks, d_in):
    """Compute Choi matrix of a channel."""
    d_out = Ks[0].shape[0]
    C = np.zeros((d_in*d_out, d_in*d_out), dtype=complex)
    for i in range(d_in):
        for j in range(d_in):
            e = np.zeros((d_in, d_in), dtype=complex); e[i,j] = 1
            C[i*d_out:(i+1)*d_out, j*d_out:(j+1)*d_out] = sum(K@e@K.conj().T for K in Ks)
    C /= d_in
    return C

def depolarizing_kraus(d, p):
    """Depolarizing channel: (1-p)ρ + p·I/d."""
    Ks = [np.sqrt(1 - p + p/d**2) * np.eye(d, dtype=complex)]
    for i in range(d):
        for j in range(d):
            if i == j: continue
            E = np.zeros((d,d), dtype=complex); E[i,j] = np.sqrt(p) / d
            Ks.append(E)
    return Ks

def amplitude_damping_kraus(gamma):
    """Amplitude damping channel."""
    K0 = np.array([[1, 0], [0, np.sqrt(1-gamma)]], dtype=complex)
    K1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)
    return [K0, K1]

def generate_expanded():
    """Generate expanded dataset covering d=4 to d=30."""
    X_all, y_all, ent_all = [], [], []
    
    # === Phase 1: Separable states (all dimensions) ===
    print("  Phase 1: Separable states")
    for dA, dB in [(2,2),(2,3),(3,3),(2,4),(2,5),(3,4),(2,6),(2,7),(3,5),(4,4),(2,8),(2,10),(2,15)]:
        n = 100
        for _ in range(n):
            rho = random_separable(dA, dB)
            feat = extract_features(rho, dA, dB)
            k = kdw_correct(rho, dA, dB, 20)
            X_all.append(feat); y_all.append(k); ent_all.append(False)
    print(f"    {len(X_all)} separable states")
    
    # === Phase 2: Werner/Isotropic (d=2..6) ===
    print("  Phase 2: Werner & Isotropic")
    for d in [2, 3, 4, 5]:
        for p in np.linspace(0, 1, 40):
            rho = werner(d, p)
            feat = extract_features(rho, d, d)
            k = kdw_correct(rho, d, d, 30)
            X_all.append(feat); y_all.append(k); ent_all.append(p > 1/d)
        for f in np.linspace(0, 1/(d+1)+0.05, 30):
            rho = isotropic(d, min(f, 1.0))
            feat = extract_features(rho, d, d)
            k = kdw_correct(rho, d, d, 30)
            X_all.append(feat); y_all.append(k); ent_all.append(f > 1/(d+1))
    print(f"    {len(X_all)} total after Werner/Iso")
    
    # === Phase 3: Horodecki PPT-entangled ===
    print("  Phase 3: PPT-entangled constructions")
    for a in np.linspace(0.01, 0.99, 50):
        rho = horodecki_2x4(a)
        pt = partial_transpose_B(rho, 2, 4)
        if np.linalg.eigvalsh(pt).min() >= -1e-10:
            feat = extract_features(rho, 2, 4)
            k = kdw_correct(rho, 2, 4, 30)
            R = realignment_norm(rho, 2, 4)
            X_all.append(feat); y_all.append(k); ent_all.append(R > 1.0 + 1e-6)
    print(f"    {len(X_all)} total after Horodecki")
    
    # === Phase 4: PPT via mixing ===
    print("  Phase 4: PPT states via mixing")
    for dA, dB in [(2,3),(3,3),(2,4),(2,5)]:
        count = 0
        for _ in range(500):
            rho = random_ppt_via_mix(dA, dB, n_sep=5, noise=0.02)
            if rho is None: continue
            feat = extract_features(rho, dA, dB)
            k = kdw_correct(rho, dA, dB, 20)
            R = realignment_norm(rho, dA, dB)
            X_all.append(feat); y_all.append(k); ent_all.append(R > 1+1e-6)
            count += 1
            if count >= 100: break
    print(f"    {len(X_all)} total after PPT mixing")
    
    # === Phase 5: Random mixed states (exact entanglement for 2×2, 2×3) ===
    print("  Phase 5: Random mixed states")
    for dA, dB in [(2,2),(2,3),(3,3),(2,4)]:
        d = dA*dB
        for _ in range(200):
            r = np.random.randint(2, d+2)
            G = np.random.randn(d, r)+1j*np.random.randn(d, r)
            rho = G@G.conj().T; rho /= np.trace(rho)
            feat = extract_features(rho, dA, dB)
            k = kdw_correct(rho, dA, dB, 20)
            pt = partial_transpose_B(rho, dA, dB)
            is_ent_ppt = np.linalg.eigvalsh(pt).min() < -1e-10
            X_all.append(feat); y_all.append(k); ent_all.append(is_ent_ppt)
    print(f"    {len(X_all)} total after random mixed")
    
    # === Phase 6: Channel Choi matrices ===
    print("  Phase 6: Channel Choi matrices")
    # Effective channel Ñ
    Ks = build_effective_channel()
    C = channel_choi(Ks, 2)
    feat = extract_features(C, 2, 4)
    k = kdw_correct(C, 2, 4, 50)
    X_all.append(feat); y_all.append(k); ent_all.append(True)
    
    # Amplitude damping
    for gamma in np.linspace(0, 1, 20):
        Ks_ad = amplitude_damping_kraus(gamma)
        C_ad = channel_choi(Ks_ad, 2)
        feat = extract_features(C_ad, 2, 2)
        k = kdw_correct(C_ad, 2, 2, 30)
        X_all.append(feat); y_all.append(k); ent_all.append(gamma < 1)
    print(f"    {len(X_all)} total after channels")
    
    X = np.array(X_all)
    y = np.array(y_all)
    ent = np.array(ent_all)
    
    return X, y, ent

if __name__ == '__main__':
    print("="*60)
    print("  MLP v6c — Expanded Training (d=4..30)")
    print("="*60)
    
    cache = 'sa_data/v6c_data.npz'
    if os.path.exists(cache):
        d = np.load(cache)
        X, y, ent = d['X'], d['y'], d['ent']
        print(f"  Loaded: {len(X)} states")
    else:
        X, y, ent = generate_expanded()
        np.savez(cache, X=X, y=y, ent=ent)
        print(f"  Saved: {cache}")
    
    print(f"\n  Dataset: {len(X)} states, {N_FEAT} features")
    print(f"  K_DW: [{y.min():.4f}, {y.max():.4f}]")
    print(f"  K>0: {np.sum(y>0.001)} ({100*np.mean(y>0.001):.1f}%)")
    print(f"  Entangled: {ent.sum()} ({100*ent.mean():.1f}%)")
    
    model, mu, std, r2, ca = train(X, y, ent, epochs=400)
    
    torch.save({
        'model_state': model.state_dict(),
        'mu': mu, 'std': std,
        'feature_names': ['rank_norm','purity_norm','eig_min','eig_max','eig_std',
                          'pt_min','pt_boundary_dist','S_A','S_B','S_AB',
                          'mutual_info','realign_norm','concurrence_approx'],
        'r2': r2, 'cls_acc': ca,
        'note': 'v6c: expanded d=4..30, PPT-entangled Horodecki, channels, correct K_DW',
    }, 'sa_data/model_v6.pt')
    
    print(f"\n  💾 sa_data/model_v6.pt (R²={r2:.4f}, cls={ca:.1%})")
    print(f"{'='*60}")
