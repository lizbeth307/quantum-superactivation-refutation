"""
fix_d24_interpolation.py — Fix d=24 with FULL K_DW range via interpolation
Strategy: rho(t) = (1-t)*rho_sep + t*rho_SA  for t in [0,1]
This gives K_DW from 0 (separable) to ~2.9 (SA) — full diversity!
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
        sum_pSB = sum_pSE = 0.0
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
        best = max(best, (S_B_unc - sum_pSB) - (S_E_unc - sum_pSE))
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
        'eig_min': eigvals[-1], 'eig_max': eigvals[0],
        'eig_std': np.std(eig_nz) if len(eig_nz) > 0 else 0,
        'pt_min': pt_min, 'pt_boundary_dist': abs(pt_min), 'pt_neg_count': 0,
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

def interpolation_worker(args):
    """Interpolate between separable and SA states."""
    rho_sa, dA, dB, t_values, seed, n_bases = args
    np.random.seed(seed)
    d = dA * dB
    
    X_list, y_list = [], []
    
    for t in t_values:
        # Generate random separable state
        rho_sep = np.zeros((d, d), dtype=complex)
        n_terms = np.random.randint(3, 12)
        for _ in range(n_terms):
            a = np.random.randn(dA) + 1j * np.random.randn(dA)
            a /= np.linalg.norm(a)
            b = np.random.randn(dB) + 1j * np.random.randn(dB)
            b /= np.linalg.norm(b)
            psi = np.kron(a, b)
            rho_sep += np.outer(psi, psi.conj())
        rho_sep /= np.trace(rho_sep).real
        
        # Interpolate
        rho = (1 - t) * rho_sep + t * rho_sa
        rho = (rho + rho.conj().T) / 2
        rho /= np.trace(rho).real
        
        # PPT check
        rho_r = rho.reshape(dA, dB, dA, dB)
        rho_pt = rho_r.transpose(0, 3, 2, 1).reshape(d, d)
        pt_min = np.linalg.eigvalsh(rho_pt).min()
        
        if pt_min < -1e-8:
            continue  # NPT — skip
        
        feats = extract_features(rho, dA, dB)
        kdw = kdw_stinespring(rho, dA, dB, n_bases=n_bases)
        
        row = [feats.get(f, 0.0) for f in FEATURES]
        X_list.append(row)
        y_list.append(max(kdw, 0.0))
    
    return np.array(X_list) if X_list else np.zeros((0, len(FEATURES))), np.array(y_list)


if __name__ == '__main__':
    print("=" * 60)
    print("  Fix d=24: Full K_DW range via interpolation")
    print("=" * 60)
    
    # Load d=24 SA seed
    emb = np.load('sa_data/embedded_2x12.npz')
    rho_sa = emb['rho']
    dA, dB = 2, 12
    d = 24
    print(f"  SA seed: {rho_sa.shape}, dA={dA}, dB={dB}")
    
    # Verify SA seed K_DW
    kdw_seed = kdw_stinespring(rho_sa, dA, dB, n_bases=50)
    print(f"  SA seed K_DW: {kdw_seed:.4f}")
    
    # Generate interpolation: t ∈ [0, 1]
    # t=0 → separable (K_DW=0), t=1 → SA (K_DW~2.7)
    n_cores = 15
    n_per_core = 300
    
    # Dense sampling across full range
    t_all = np.random.uniform(0, 1, n_per_core * n_cores)
    t_chunks = np.array_split(t_all, n_cores)
    
    args = [(rho_sa, dA, dB, chunk, 42 + i, 30) for i, chunk in enumerate(t_chunks)]
    
    t0 = time.time()
    print(f"  Generating {n_per_core * n_cores} interpolation points on {n_cores} cores...")
    
    with Pool(n_cores) as pool:
        results = pool.map(interpolation_worker, args)
    
    X_all = np.vstack([r[0] for r in results if len(r[0]) > 0])
    y_all = np.concatenate([r[1] for r in results if len(r[1]) > 0])
    
    elapsed = time.time() - t0
    
    # Also add pure separable (t=0)
    print(f"\n  Adding 500 pure separable states...")
    np.random.seed(999)
    X_sep, y_sep = [], []
    for _ in range(500):
        rho_sep = np.zeros((d, d), dtype=complex)
        for _ in range(np.random.randint(3, 10)):
            a = np.random.randn(dA) + 1j * np.random.randn(dA)
            a /= np.linalg.norm(a)
            b = np.random.randn(dB) + 1j * np.random.randn(dB)
            b /= np.linalg.norm(b)
            psi = np.kron(a, b)
            rho_sep += np.outer(psi, psi.conj())
        rho_sep /= np.trace(rho_sep).real
        feats = extract_features(rho_sep, dA, dB)
        X_sep.append([feats.get(f, 0.0) for f in FEATURES])
        y_sep.append(0.0)
    
    X_final = np.vstack([X_all, np.array(X_sep)])
    y_final = np.concatenate([y_all, np.array(y_sep)])
    
    # Save
    save = {'X': X_final, 'kdw': y_final}
    for j, feat in enumerate(FEATURES):
        save[feat] = X_final[:, j]
    np.savez_compressed('sa_data/path_d_d24_full.npz', **save)
    
    # Stats
    n_zero = np.sum(y_final < 0.01)
    n_mid = np.sum((y_final > 0.5) & (y_final < 2.0))
    n_high = np.sum(y_final > 2.0)
    
    print(f"\n  ✅ d=24 FULL: {len(y_final)} pts, K_DW [{y_final.min():.4f}, {y_final.max():.4f}] [{elapsed:.0f}s]")
    print(f"     K_DW ≈ 0:   {n_zero} ({100*n_zero/len(y_final):.0f}%)")
    print(f"     K_DW 0.5-2:  {n_mid} ({100*n_mid/len(y_final):.0f}%)")
    print(f"     K_DW > 2:    {n_high} ({100*n_high/len(y_final):.0f}%)")
    print(f"\n  💾 Saved: sa_data/path_d_d24_full.npz")
    print("=" * 60)
