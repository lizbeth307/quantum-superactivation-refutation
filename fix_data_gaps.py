"""
fix_data_gaps.py — Fix diagnostic gaps + Phase 0.5 SA Finder
Per plan: search ALL d=4..20, ALL factorizations.
Generates NEGATIVE examples (K_DW≈0) + more data for weak dimensions.
"""
import numpy as np
import os, time, gc
from multiprocessing import Pool

# ══════════════════════════════════════════════════════════
# Core K_DW computation (Stinespring)
# ══════════════════════════════════════════════════════════

def von_neumann(rho):
    e = np.linalg.eigvalsh(rho)
    e = e[e > 1e-15]
    return -np.sum(e * np.log2(e)) if len(e) > 0 else 0.0

def kdw_stinespring(rho, dA, dB, n_bases=50):
    """Full K_DW via Stinespring purification."""
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
        norms_sq = np.sum(np.abs(beta)**2, axis=2)
        p_x = norms_sq @ lam
        
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
    """Extract 16 features per plan §0.2."""
    d = dA * dB
    eigvals = np.linalg.eigvalsh(rho)
    eigvals = np.sort(eigvals)[::-1]
    
    # PT eigenvalues
    rho_r = rho.reshape(dA, dB, dA, dB)
    rho_pt = rho_r.transpose(0, 3, 2, 1).reshape(d, d)
    pt_eigs = np.linalg.eigvalsh(rho_pt)
    pt_min = pt_eigs.min()
    
    # Reduced states
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
    
    # Realignment
    R = rho.reshape(dA, dB, dA, dB).transpose(0, 2, 1, 3).reshape(dA*dA, dB*dB)
    realign_norm = np.linalg.norm(R, 'nuc')
    
    eig_nz = eigvals[eigvals > 1e-14]
    rank = len(eig_nz)
    
    return {
        'rank_norm': rank / d**2,
        'purity_norm': np.sum(eigvals**2) * d,
        'eig_min': eigvals[-1] if len(eigvals) > 0 else 0,
        'eig_max': eigvals[0] if len(eigvals) > 0 else 0,
        'eig_std': np.std(eig_nz) if len(eig_nz) > 0 else 0,
        'pt_min': pt_min,
        'pt_boundary_dist': abs(pt_min),
        'pt_neg_count': np.sum(pt_eigs < -1e-10),
        'S_A': S_A,
        'S_B': S_B,
        'S_AB': S_AB,
        'mutual_info': S_A + S_B - S_AB,
        'mutual_info_norm': (S_A + S_B - S_AB) / (2 * np.log2(max(d, 2))),
        'realign_norm': realign_norm,
        'A_max_mixed_dist': np.linalg.norm(rho_A - np.eye(dA)/dA),
        'B_max_mixed_dist': np.linalg.norm(rho_B - np.eye(dB)/dB),
    }

# ══════════════════════════════════════════════════════════
# Negative Example Generators (plan §Issue 7)
# ══════════════════════════════════════════════════════════

def gen_separable(dA, dB, n=500):
    """Generate random separable states (K_DW = 0 guaranteed)."""
    d = dA * dB
    states = []
    for _ in range(n):
        # Mix of product states
        rho = np.zeros((d, d), dtype=complex)
        n_terms = np.random.randint(2, min(d, 10))
        for _ in range(n_terms):
            a = np.random.randn(dA) + 1j * np.random.randn(dA)
            a /= np.linalg.norm(a)
            b = np.random.randn(dB) + 1j * np.random.randn(dB)
            b /= np.linalg.norm(b)
            psi = np.kron(a, b)
            rho += np.outer(psi, psi.conj())
        rho /= np.trace(rho).real
        states.append(rho)
    return states

def gen_ppt_near_boundary(dA, dB, n=500):
    """Generate PPT states near boundary (K_DW ≈ 0, plan §Issue 7 Level 1)."""
    d = dA * dB
    states = []
    for _ in range(n * 3):  # generate more, filter PPT
        G = np.random.randn(d, d) + 1j * np.random.randn(d, d)
        rho = G @ G.conj().T
        rho /= np.trace(rho).real
        
        # Check PPT
        rho_r = rho.reshape(dA, dB, dA, dB)
        rho_pt = rho_r.transpose(0, 3, 2, 1).reshape(d, d)
        pt_min = np.linalg.eigvalsh(rho_pt).min()
        
        if pt_min > -1e-8:  # PPT
            states.append(rho)
            if len(states) >= n:
                break
    return states

def gen_worker(args):
    """Worker for parallel generation + feature extraction."""
    dA, dB, seed, n_sep, n_ppt, n_bases = args
    np.random.seed(seed)
    d = dA * dB
    results = []
    
    # Separable states (K_DW = 0)
    seps = gen_separable(dA, dB, n_sep)
    for rho in seps:
        feats = extract_features(rho, dA, dB)
        feats['kdw'] = 0.0  # guaranteed for separable
        feats['d'] = d
        feats['dA'] = dA
        feats['dB'] = dB
        feats['type'] = 'separable'
        results.append(feats)
    
    # PPT near-boundary (K_DW ≈ 0, compute to verify)
    ppts = gen_ppt_near_boundary(dA, dB, n_ppt)
    for rho in ppts:
        feats = extract_features(rho, dA, dB)
        kdw = kdw_stinespring(rho, dA, dB, n_bases=n_bases)
        feats['kdw'] = max(kdw, 0.0)
        feats['d'] = d
        feats['dA'] = dA
        feats['dB'] = dB
        feats['type'] = 'ppt_random'
        results.append(feats)
    
    return results

# ══════════════════════════════════════════════════════════
# Phase 0.5: SA Finder for all d, all factorizations
# ══════════════════════════════════════════════════════════

def sa_search_worker(args):
    """Search for SA in a specific (dA, dB) factorization."""
    dA, dB, seed, n_trials, n_bases = args
    np.random.seed(seed)
    d = dA * dB
    
    best_kdw = -999
    best_rho = None
    results = []
    
    for trial in range(n_trials):
        # Random Wishart → project to PPT
        G = np.random.randn(d, d) + 1j * np.random.randn(d, d)
        rho = G @ G.conj().T
        rho /= np.trace(rho).real
        
        # PPT check
        rho_r = rho.reshape(dA, dB, dA, dB)
        rho_pt = rho_r.transpose(0, 3, 2, 1).reshape(d, d)
        pt_min = np.linalg.eigvalsh(rho_pt).min()
        
        if pt_min < -1e-8:
            continue  # NPT, skip
        
        # Entanglement check (realignment)
        R = rho.reshape(dA, dB, dA, dB).transpose(0, 2, 1, 3).reshape(dA*dA, dB*dB)
        realign = np.linalg.norm(R, 'nuc')
        
        if realign <= 1.0 + 1e-8:
            continue  # separable
        
        # PPT + ENT → compute K_DW
        kdw = kdw_stinespring(rho, dA, dB, n_bases=n_bases)
        
        feats = extract_features(rho, dA, dB)
        feats['kdw'] = max(kdw, 0.0)
        feats['d'] = d
        feats['dA'] = dA
        feats['dB'] = dB
        feats['type'] = 'sa_search'
        results.append(feats)
        
        if kdw > best_kdw:
            best_kdw = kdw
            best_rho = rho.copy()
    
    return results, best_kdw, best_rho


if __name__ == '__main__':
    print("=" * 60)
    print("  DIAGNOSTIC FIX + Phase 0.5: SA Finder")
    print("=" * 60)
    
    FEATURES = [
        'rank_norm', 'purity_norm', 'eig_min', 'eig_max', 'eig_std',
        'pt_min', 'pt_boundary_dist', 'pt_neg_count',
        'S_A', 'S_B', 'S_AB', 'mutual_info', 'mutual_info_norm',
        'realign_norm', 'A_max_mixed_dist', 'B_max_mixed_dist',
    ]
    
    all_results = []
    
    # ── Step 1: Generate NEGATIVE examples (K_DW ≈ 0) ──
    print("\n  Step 1: Generating negative examples (K_DW ≈ 0)...")
    
    factorizations = []
    for d in range(4, 21):
        for dA in range(2, d):
            if d % dA == 0:
                dB = d // dA
                if dB >= 2:
                    factorizations.append((dA, dB))
    
    print(f"  Factorizations to process: {len(factorizations)}")
    for dA, dB in factorizations:
        print(f"    {dA}×{dB} = {dA*dB}")
    
    # Generate separable + random PPT for each factorization
    neg_args = [(dA, dB, i*100+42, 100, 50, 20) for i, (dA, dB) in enumerate(factorizations)]
    
    t0 = time.time()
    with Pool(min(16, len(neg_args))) as pool:
        neg_results = pool.map(gen_worker, neg_args)
    
    for batch in neg_results:
        all_results.extend(batch)
    
    n_neg = len(all_results)
    n_zero = sum(1 for r in all_results if r['kdw'] < 0.005)
    print(f"  Generated {n_neg} negative examples ({n_zero} with K_DW≈0) [{time.time()-t0:.0f}s]")
    
    # ── Step 2: Phase 0.5 — SA search for ALL d=4..20 ──
    print("\n  Step 2: Phase 0.5 — SA Finder (all d, all factorizations)...")
    
    sa_args = [(dA, dB, i*1000+7, 1000, 30) for i, (dA, dB) in enumerate(factorizations)]
    
    t1 = time.time()
    with Pool(min(16, len(sa_args))) as pool:
        sa_results = pool.map(sa_search_worker, sa_args)
    
    print(f"\n  Phase 0.5 SA Search Results:")
    print(f"  {'dA×dB':>6} {'d':>3} {'K_DW':>8} {'Found':>6}")
    print(f"  {'-'*30}")
    for (dA, dB), (results, best_kdw, best_rho) in zip(factorizations, sa_results):
        d = dA * dB
        n_found = len(results)
        status = "✅ SA!" if best_kdw > 0.001 else "❌"
        print(f"  {dA}×{dB:>3} {d:>3} {best_kdw:>8.4f} {n_found:>6} {status}")
        all_results.extend(results)
        
        # Save best SA candidate
        if best_rho is not None and best_kdw > 0.001:
            np.savez_compressed(f'sa_data/sa_phase05_{dA}x{dB}.npz', rho=best_rho, kdw=best_kdw)
    
    print(f"  SA search complete [{time.time()-t1:.0f}s]")
    
    # ── Step 3: Save combined dataset ──
    n_total = len(all_results)
    print(f"\n  Step 3: Saving {n_total} total examples...")
    
    X = np.zeros((n_total, len(FEATURES)))
    y = np.zeros(n_total)
    dims = np.zeros(n_total)
    
    for i, r in enumerate(all_results):
        for j, feat in enumerate(FEATURES):
            X[i, j] = r.get(feat, 0.0)
        y[i] = r['kdw']
        dims[i] = r['d']
    
    np.savez_compressed('sa_data/phase05_data.npz',
                       X=X, y=y, dims=dims, feature_names=FEATURES)
    
    n_pos = np.sum(y > 0.001)
    n_zero = np.sum(y < 0.005)
    print(f"  K_DW > 0: {n_pos} ({100*n_pos/n_total:.1f}%)")
    print(f"  K_DW ≈ 0: {n_zero} ({100*n_zero/n_total:.1f}%)")
    print(f"  Dimensions: {sorted(set(dims.astype(int)))}")
    print(f"\n  💾 Saved: sa_data/phase05_data.npz")
    print("=" * 60)
