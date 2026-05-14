"""
fix_d4_d24.py — Fix confidence for d=4 (31%) and d=24 (54%)
d=4: Use Phase 0Q state (K_DW=1.263) as seed → wider K_DW range
d=24: Generate more path data from embedded seed
"""
import numpy as np
import os, time
from multiprocessing import Pool

def von_neumann(rho):
    e = np.linalg.eigvalsh(rho)
    e = e[e > 1e-15]
    return -np.sum(e * np.log2(e)) if len(e) > 0 else 0.0

def kdw_stinespring(rho, dA, dB, n_bases=50):
    d = dA * dB
    eigvals, eigvecs = np.linalg.eigh(rho)
    mask = eigvals > 1e-14
    lam = eigvals[mask]
    phi = eigvecs[:, mask]
    r = len(lam)
    if r == 0:
        return 0.0
    sqrt_lam = np.sqrt(lam)
    phi_r = phi.reshape(dA, dB, r)
    S_E_unc = von_neumann(np.diag(lam))
    rho_B_unc = sum(rho[a*dB:(a+1)*dB, a*dB:(a+1)*dB] for a in range(dA))
    S_B_unc = von_neumann(rho_B_unc)
    best = -999.0
    for trial in range(n_bases):
        if trial == 0:
            U = np.eye(dA, dtype=complex)
        else:
            H = np.random.randn(dA, dA) + 1j * np.random.randn(dA, dA)
            U, _ = np.linalg.qr(H)
        beta = np.einsum('ax,abk->xkb', U.conj(), phi_r)
        p_x = (np.sum(np.abs(beta)**2, axis=2)) @ lam
        sum_pSB = 0.0
        sum_pSE = 0.0
        for x in range(dA):
            if p_x[x] < 1e-15:
                continue
            wb = sqrt_lam[:, None] * beta[x]
            w = np.sum(wb, axis=0)
            rho_B_x = np.outer(w, w.conj()) / p_x[x]
            sum_pSB += p_x[x] * von_neumann(rho_B_x)
            gram = beta[x].conj() @ beta[x].T
            rho_E_x = np.outer(sqrt_lam, sqrt_lam) * gram.T / p_x[x]
            sum_pSE += p_x[x] * von_neumann(rho_E_x)
        kdw = (S_B_unc - sum_pSB) - (S_E_unc - sum_pSE)
        best = max(best, kdw)
    return best

def extract_features(rho, dA, dB):
    d = dA * dB
    eigvals = np.sort(np.linalg.eigvalsh(rho))[::-1]
    rho_r = rho.reshape(dA, dB, dA, dB)
    rho_pt = rho_r.transpose(0, 3, 2, 1).reshape(d, d)
    pt_min = np.linalg.eigvalsh(rho_pt).min()
    rho_A = np.zeros((dA, dA), dtype=complex)
    rho_B = np.zeros((dB, dB), dtype=complex)
    for a in range(dA):
        for b in range(dB):
            for ap in range(dA):
                rho_A[a, ap] += rho[a*dB+b, ap*dB+b]
            for bp in range(dB):
                rho_B[b, bp] += rho[a*dB+b, a*dB+bp]
    S_A = von_neumann(rho_A)
    S_B = von_neumann(rho_B)
    S_AB = von_neumann(rho)
    R = rho.reshape(dA, dB, dA, dB).transpose(0, 2, 1, 3).reshape(dA*dA, dB*dB)
    realign = np.linalg.norm(R, 'nuc')
    eig_nz = eigvals[eigvals > 1e-14]
    return {
        'rank_norm': len(eig_nz) / d**2,
        'purity_norm': np.sum(eigvals**2) * d,
        'eig_min': eigvals[-1] if len(eigvals) > 0 else 0,
        'eig_max': eigvals[0] if len(eigvals) > 0 else 0,
        'eig_std': np.std(eig_nz) if len(eig_nz) > 0 else 0,
        'pt_min': pt_min,
        'pt_boundary_dist': abs(pt_min),
        'pt_neg_count': 0,
        'S_A': S_A, 'S_B': S_B, 'S_AB': S_AB,
        'mutual_info': S_A + S_B - S_AB,
        'mutual_info_norm': (S_A + S_B - S_AB) / max(2 * np.log2(d), 1e-10),
        'realign_norm': realign,
        'A_max_mixed_dist': np.linalg.norm(rho_A - np.eye(dA)/dA),
        'B_max_mixed_dist': np.linalg.norm(rho_B - np.eye(dB)/dB),
    }

FEATURES = [
    'rank_norm', 'purity_norm', 'eig_min', 'eig_max', 'eig_std',
    'pt_min', 'pt_boundary_dist', 'pt_neg_count',
    'S_A', 'S_B', 'S_AB', 'mutual_info', 'mutual_info_norm',
    'realign_norm', 'A_max_mixed_dist', 'B_max_mixed_dist',
]

def noise_sweep(rho_seed, dA, dB, eps_values, seed, n_bases):
    np.random.seed(seed)
    d = dA * dB
    X_list, y_list = [], []
    for eps in eps_values:
        rho = (1 - eps) * rho_seed + eps * np.eye(d) / d
        rho_r = rho.reshape(dA, dB, dA, dB)
        rho_pt = rho_r.transpose(0, 3, 2, 1).reshape(d, d)
        pt_min = np.linalg.eigvalsh(rho_pt).min()
        if pt_min < -1e-8:
            continue
        feats = extract_features(rho, dA, dB)
        kdw = kdw_stinespring(rho, dA, dB, n_bases=n_bases)
        row = [feats.get(f, 0.0) for f in FEATURES]
        X_list.append(row)
        y_list.append(max(kdw, 0.0))
    return np.array(X_list) if X_list else np.zeros((0, len(FEATURES))), np.array(y_list)

def random_perturbation_sweep(rho_seed, dA, dB, n_pts, seed, n_bases):
    """Generate varied perturbations (not just mixing with identity)."""
    np.random.seed(seed)
    d = dA * dB
    X_list, y_list = [], []
    for i in range(n_pts):
        eps = np.random.uniform(0, 0.5)
        # Random perturbation direction (not just max-mixed)
        G = np.random.randn(d, d) + 1j * np.random.randn(d, d)
        noise = G @ G.conj().T
        noise /= np.trace(noise).real
        
        rho = (1 - eps) * rho_seed + eps * noise
        rho = (rho + rho.conj().T) / 2
        rho /= np.trace(rho).real
        
        # PPT check
        rho_r = rho.reshape(dA, dB, dA, dB)
        rho_pt = rho_r.transpose(0, 3, 2, 1).reshape(d, d)
        pt_min = np.linalg.eigvalsh(rho_pt).min()
        
        if pt_min < -1e-8:
            continue
        
        feats = extract_features(rho, dA, dB)
        kdw = kdw_stinespring(rho, dA, dB, n_bases=n_bases)
        row = [feats.get(f, 0.0) for f in FEATURES]
        X_list.append(row)
        y_list.append(max(kdw, 0.0))
    
    return np.array(X_list) if X_list else np.zeros((0, len(FEATURES))), np.array(y_list)

def worker(args):
    mode, rho_seed, dA, dB, param, seed, n_bases = args
    if mode == 'noise':
        return noise_sweep(rho_seed, dA, dB, param, seed, n_bases)
    else:
        return random_perturbation_sweep(rho_seed, dA, dB, param, seed, n_bases)


if __name__ == '__main__':
    print("=" * 60)
    print("  Fixing d=4 and d=24 confidence")
    print("=" * 60)
    
    n_cores = 15
    
    # ═══ d=4: Phase 0Q state (16×16, bipartite 4×4) ═══
    print("\n  ── d=4: Phase 0Q state (K_DW=1.263) ──")
    rho4 = np.load('sa_data/phase0q_2x2.npz')['rho']  # 16×16
    dA4, dB4 = 4, 4
    
    # 1) Noise sweep: eps=0..0.7
    eps_all = np.linspace(0, 0.7, 2000)
    eps_chunks = np.array_split(eps_all, n_cores)
    args4_noise = [('noise', rho4, dA4, dB4, chunk, 42+i, 50) for i, chunk in enumerate(eps_chunks)]
    
    # 2) Random perturbations: 200 per core
    args4_rand = [('rand', rho4, dA4, dB4, 200, 1000+i, 50) for i in range(n_cores)]
    
    t0 = time.time()
    with Pool(n_cores) as pool:
        res_noise = pool.map(worker, args4_noise)
        res_rand = pool.map(worker, args4_rand)
    
    X4_noise = np.vstack([r[0] for r in res_noise if len(r[0]) > 0])
    y4_noise = np.concatenate([r[1] for r in res_noise if len(r[1]) > 0])
    X4_rand = np.vstack([r[0] for r in res_rand if len(r[0]) > 0])
    y4_rand = np.concatenate([r[1] for r in res_rand if len(r[1]) > 0])
    
    X4 = np.vstack([X4_noise, X4_rand])
    y4 = np.concatenate([y4_noise, y4_rand])
    
    # Save as path_d_d4_0q.npz (separate from flower paths)
    save4 = {'X': X4, 'kdw': y4}
    for j, feat in enumerate(FEATURES):
        save4[feat] = X4[:, j]
    np.savez_compressed('sa_data/path_d_d4_0q.npz', **save4)
    print(f"  ✅ d=4 (0Q): {len(y4)} pts, K_DW [{y4.min():.4f}, {y4.max():.4f}] [{time.time()-t0:.0f}s]")
    
    # ═══ d=24: More noise sweep data ═══
    print("\n  ── d=24: Expanding from 1000 → 4000+ pts ──")
    # Load existing d=24 seed
    d24_data = np.load('sa_data/path_d_d24.npz')
    # The seed for d=24 is an embedded state — we need the rho
    # Check if embedded file has rho
    emb = np.load('sa_data/embedded_2x12.npz')
    if 'rho' in emb:
        rho24 = emb['rho']
    else:
        # Reconstruct from the best known state
        # Use native_d20_2x10 and embed into d=24
        # Actually, let's just use the existing path data values and generate around them
        # Use path_d_d24.npz which has feature columns
        print("  No rho for d=24, generating separable + random PPT instead")
        rho24 = None
    
    if rho24 is not None:
        dA24 = rho24.shape[0]
        # Figure out bipartite split
        for da in [2, 3, 4, 6]:
            if dA24 % da == 0:
                db = dA24 // da
                break
        print(f"  d=24 seed: {rho24.shape}, split {da}×{db}")
        
        eps24 = np.linspace(0, 0.6, 3000)
        eps_chunks24 = np.array_split(eps24, n_cores)
        args24 = [('noise', rho24, da, db, chunk, 500+i, 30) for i, chunk in enumerate(eps_chunks24)]
        
        t1 = time.time()
        with Pool(n_cores) as pool:
            res24 = pool.map(worker, args24)
        
        X24 = np.vstack([r[0] for r in res24 if len(r[0]) > 0])
        y24 = np.concatenate([r[1] for r in res24 if len(r[1]) > 0])
        
        save24 = {'X': X24, 'kdw': y24}
        for j, feat in enumerate(FEATURES):
            save24[feat] = X24[:, j]
        np.savez_compressed('sa_data/path_d_d24_extra.npz', **save24)
        print(f"  ✅ d=24: {len(y24)} pts, K_DW [{y24.min():.4f}, {y24.max():.4f}] [{time.time()-t1:.0f}s]")
    else:
        # Generate random separable + PPT for d=24 with varied noise
        print("  Generating diverse random states for d=24...")
        dA24, dB24 = 2, 12
        d24 = 24
        
        X_list, y_list = [], []
        np.random.seed(777)
        
        # Separable states
        for _ in range(2000):
            rho = np.zeros((d24, d24), dtype=complex)
            for _ in range(np.random.randint(2, 8)):
                a = np.random.randn(dA24) + 1j * np.random.randn(dA24)
                a /= np.linalg.norm(a)
                b = np.random.randn(dB24) + 1j * np.random.randn(dB24)
                b /= np.linalg.norm(b)
                psi = np.kron(a, b)
                rho += np.outer(psi, psi.conj())
            rho /= np.trace(rho).real
            feats = extract_features(rho, dA24, dB24)
            row = [feats.get(f, 0.0) for f in FEATURES]
            X_list.append(row)
            y_list.append(0.0)
        
        X24 = np.array(X_list)
        y24 = np.array(y_list)
        save24 = {'X': X24, 'kdw': y24}
        for j, feat in enumerate(FEATURES):
            save24[feat] = X24[:, j]
        np.savez_compressed('sa_data/path_d_d24_extra.npz', **save24)
        print(f"  ✅ d=24 (separable): {len(y24)} pts [{time.time()-t0:.0f}s]")
    
    print(f"\n{'='*60}")
    print("  DONE — retrain with new data")
    print(f"{'='*60}")
