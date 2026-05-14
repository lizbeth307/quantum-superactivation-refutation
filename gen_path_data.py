"""
gen_path_data.py — Generate path data for WEAK dimensions (d=8,9,12,14,15,18)
Uses existing SA candidates + Phase 0Q results as seeds.
Generates noise sweeps (like Path B/D) for each.
"""
import numpy as np
import os, time
from multiprocessing import Pool

def von_neumann(rho):
    e = np.linalg.eigvalsh(rho)
    e = e[e > 1e-15]
    return -np.sum(e * np.log2(e)) if len(e) > 0 else 0.0

def kdw_stinespring(rho, dA, dB, n_bases=50):
    """K_DW via Stinespring."""
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
    """16 features per plan."""
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

def noise_sweep_worker(args):
    """Generate noise sweep from a seed state."""
    rho_seed, dA, dB, eps_values, seed, n_bases = args
    np.random.seed(seed)
    d = dA * dB
    
    FEATURES = [
        'rank_norm', 'purity_norm', 'eig_min', 'eig_max', 'eig_std',
        'pt_min', 'pt_boundary_dist', 'pt_neg_count',
        'S_A', 'S_B', 'S_AB', 'mutual_info', 'mutual_info_norm',
        'realign_norm', 'A_max_mixed_dist', 'B_max_mixed_dist',
    ]
    
    X_list = []
    y_list = []
    
    for eps in eps_values:
        # Mix seed with maximally mixed state
        rho = (1 - eps) * rho_seed + eps * np.eye(d) / d
        
        # Check PPT
        rho_r = rho.reshape(dA, dB, dA, dB)
        rho_pt = rho_r.transpose(0, 3, 2, 1).reshape(d, d)
        pt_min = np.linalg.eigvalsh(rho_pt).min()
        
        if pt_min < -1e-8:
            continue  # NPT, skip
        
        feats = extract_features(rho, dA, dB)
        kdw = kdw_stinespring(rho, dA, dB, n_bases=n_bases)
        
        row = [feats.get(f, 0.0) for f in FEATURES]
        X_list.append(row)
        y_list.append(max(kdw, 0.0))
    
    return np.array(X_list), np.array(y_list)


if __name__ == '__main__':
    print("=" * 60)
    print("  Generating Path Data for Weak Dimensions")
    print("=" * 60)
    
    FEATURES = [
        'rank_norm', 'purity_norm', 'eig_min', 'eig_max', 'eig_std',
        'pt_min', 'pt_boundary_dist', 'pt_neg_count',
        'S_A', 'S_B', 'S_AB', 'mutual_info', 'mutual_info_norm',
        'realign_norm', 'A_max_mixed_dist', 'B_max_mixed_dist',
    ]
    
    # Seed states to use
    seeds = {}
    
    # d=8: from SDP (optimized_ppt_2x4.npz)
    f8 = 'sa_data/optimized_ppt_2x4.npz'
    if os.path.exists(f8):
        data = np.load(f8)
        seeds[8] = (data['rho'], 2, 4)
        print(f"  ✅ d=8 seed: {f8}")
    
    # d=9: from Phase 0Q (rho is d²×d² = 81×81, bipartite: 9×9)
    f9 = 'sa_data/phase0q_3x3.npz'
    if os.path.exists(f9):
        data = np.load(f9)
        rho9 = data['rho']
        # Phase 0Q stores rho in (dk*ds)²×(dk*ds)² = d²×d² space
        # bipartite split is d×d where d=dk*ds=9
        d9 = int(np.sqrt(rho9.shape[0]))
        seeds[9] = (rho9, d9, d9)
        print(f"  ✅ d=9 seed: {f9} (shape {rho9.shape[0]}×{rho9.shape[0]}, split {d9}×{d9})")
    
    # d=12: from native search
    for fname in ['native_d12_2x6.npz', 'native_d12_3x4.npz', 'native_d12_6x2.npz']:
        f12 = f'sa_data/{fname}'
        if os.path.exists(f12):
            data = np.load(f12)
            if 'rho' in data:
                d_total = data['rho'].shape[0]
                # Figure out dA, dB
                for dA in [2, 3, 4, 6]:
                    if d_total % dA == 0:
                        dB = d_total // dA
                        seeds[12] = (data['rho'], dA, dB)
                        print(f"  ✅ d=12 seed: {fname} ({dA}×{dB})")
                        break
            break
    
    # d=14: from native search
    f14 = 'sa_data/native_d14_2x7.npz'
    if os.path.exists(f14):
        data = np.load(f14)
        if 'rho' in data:
            seeds[14] = (data['rho'], 2, 7)
            print(f"  ✅ d=14 seed: {f14}")
    
    # d=15: from native search
    f15 = 'sa_data/native_d15_3x5.npz'
    if os.path.exists(f15):
        data = np.load(f15)
        if 'rho' in data:
            seeds[15] = (data['rho'], 3, 5)
            print(f"  ✅ d=15 seed: {f15}")
    
    # d=18: from native search
    f18 = 'sa_data/native_d18_2x9.npz'
    if os.path.exists(f18):
        data = np.load(f18)
        if 'rho' in data:
            seeds[18] = (data['rho'], 2, 9)
            print(f"  ✅ d=18 seed: {f18}")
    
    if not seeds:
        print("  ❌ No seed states found!")
        exit(1)
    
    print(f"\n  Seeds: {sorted(seeds.keys())}")
    
    # Generate noise sweeps for each seed
    n_eps = 200  # points per core
    n_cores = 15
    eps_all = np.linspace(0, 0.6, n_eps * n_cores)
    
    for d, (rho_seed, dA, dB) in sorted(seeds.items()):
        t0 = time.time()
        print(f"\n  ── d={d} ({dA}×{dB}) ── {n_eps * n_cores} noise points, {n_cores} cores")
        
        # Split epsilon values across cores
        eps_chunks = np.array_split(eps_all, n_cores)
        args = [(rho_seed, dA, dB, chunk, 42 + i, 30) for i, chunk in enumerate(eps_chunks)]
        
        with Pool(n_cores) as pool:
            results = pool.map(noise_sweep_worker, args)
        
        X_all = np.vstack([r[0] for r in results if len(r[0]) > 0])
        y_all = np.concatenate([r[1] for r in results if len(r[1]) > 0])
        
        if len(X_all) == 0:
            print(f"  ❌ d={d}: 0 PPT states generated")
            continue
        
        # Save
        save_dict = {'X': X_all, 'kdw': y_all}
        for j, feat in enumerate(FEATURES):
            save_dict[feat] = X_all[:, j]
        
        out_path = f'sa_data/path_d_d{d}.npz'
        np.savez_compressed(out_path, **save_dict)
        
        elapsed = time.time() - t0
        print(f"  ✅ d={d}: {len(y_all)} PPT pts, K_DW [{y_all.min():.4f}, {y_all.max():.4f}] [{elapsed:.0f}s]")
    
    print(f"\n{'='*60}")
    print("  PATH DATA GENERATION COMPLETE")
    print(f"{'='*60}")
