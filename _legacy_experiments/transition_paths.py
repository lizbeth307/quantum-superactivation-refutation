"""
transition_paths.py — Generate transition path data for SA Discovery Engine.
Phase 0.1 of Master Plan v4.

Path A: alpha sweep (0 -> pi/4) for d=4 flower state.
At alpha=pi/8: known SA (K_DW=0.021).
At alpha=0: separable (no SA).
At alpha=pi/4: maximally entangled (NPT, no SA).

30 CPU cores, 10000 points.
Output: sa_data/path_a_d4.npz with features + K_DW for each point.
"""
import numpy as np
from multiprocessing import Pool
import time
import os
import sys

os.makedirs('sa_data', exist_ok=True)

# ── Core functions (top-level for pickle) ──

def bell_states_d4():
    """4 Bell states for dk=2."""
    def ket2(a, b):
        v = np.zeros(4, dtype=complex)
        v[a*2+b] = 1.0
        return v
    return [
        (ket2(0,0) + ket2(1,1)) / np.sqrt(2),  # Phi+
        (ket2(0,0) - ket2(1,1)) / np.sqrt(2),  # Phi-
        (ket2(0,1) + ket2(1,0)) / np.sqrt(2),  # Psi+
        (ket2(0,1) - ket2(1,0)) / np.sqrt(2),  # Psi-
    ]


def build_flower_alpha(alpha):
    """Build flower state parametrized by angle alpha.
    
    alpha = pi/8: original Horodecki flower (SA = yes).
    alpha = 0:    chi+ = |00>, chi- = -|11> (separable shield).
    alpha = pi/4: chi+ = chi- = Bell (NPT).
    
    Returns 16x16 density matrix.
    """
    dk, ds = 2, 2
    d = 4
    
    def ket2(a, b):
        v = np.zeros(4, dtype=complex)
        v[a*2+b] = 1.0
        return v
    
    bells = bell_states_d4()
    
    # Shield states from alpha
    psi_plus_sh = (ket2(0,1) + ket2(1,0)) / np.sqrt(2)
    psi_minus_sh = (ket2(0,1) - ket2(1,0)) / np.sqrt(2)
    
    a_plus = np.cos(alpha)
    b_plus = np.sin(alpha)
    chi_plus = a_plus * ket2(0,0) + b_plus * ket2(1,1)
    chi_minus = b_plus * ket2(0,0) - a_plus * ket2(1,1)
    
    shields = [
        0.5 * (np.outer(ket2(0,0), ket2(0,0)) + np.outer(psi_plus_sh, psi_plus_sh)),
        0.5 * (np.outer(ket2(1,1), ket2(1,1)) + np.outer(psi_minus_sh, psi_minus_sh)),
        np.outer(chi_plus, chi_plus.conj()),
        np.outer(chi_minus, chi_minus.conj()),
    ]
    
    # Mixing weights: need to find p1 that places state on PPT boundary
    # PT_min(p1) is NOT monotonic — has single peak near p1=0.586
    # Use golden section search to MAXIMIZE pt_min
    
    def build_rho(p1_val):
        p2_val = 1 - p1_val
        ww = [p1_val/2, p1_val/2, p2_val/2, p2_val/2]
        r = np.zeros((16, 16), dtype=complex)
        for i in range(4):
            psi = bells[i]
            rho_sh = shields[i]
            q = ww[i]
            for kA1 in range(2):
                for kB1 in range(2):
                    for kA2 in range(2):
                        for kB2 in range(2):
                            kv = psi[kA1*2+kB1] * psi[kA2*2+kB2].conj()
                            for sA1 in range(2):
                                for sB1 in range(2):
                                    for sA2 in range(2):
                                        for sB2 in range(2):
                                            sv = rho_sh[sA1*2+sB1, sA2*2+sB2]
                                            row = kA1*8 + sA1*4 + kB1*2 + sB1
                                            col = kA2*8 + sA2*4 + kB2*2 + sB2
                                            r[row, col] += q * kv * sv
        return (r + r.conj().T) / 2
    
    def pt_min_of(p1_val):
        r = build_rho(p1_val)
        pt = partial_transpose_4(r)
        return np.min(np.linalg.eigvalsh(pt))
    
    # Golden section search to maximize pt_min
    gr = (np.sqrt(5) + 1) / 2
    a, b = 0.01, 0.99
    for _ in range(100):  # 100 iters for machine precision
        c = b - (b - a) / gr
        d_pt = a + (b - a) / gr
        if pt_min_of(c) < pt_min_of(d_pt):
            a = c
        else:
            b = d_pt
    
    p1 = (a + b) / 2
    rho = build_rho(p1)
    return rho


def partial_transpose_4(rho):
    """PT for 4x4 bipartite system."""
    d = 4
    return rho.reshape(d,d,d,d).transpose(0,3,2,1).reshape(16,16)


def von_neumann(rho):
    eigs = np.linalg.eigvalsh(rho)
    eigs = eigs[eigs > 1e-15]
    return -np.sum(eigs * np.log2(eigs)) if len(eigs) > 0 else 0.0


def compute_kdw_d4(rho, n_bases=50):
    """K_DW for dk=2, ds=2 with correct 4-subsystem trace."""
    dk, ds = 2, 2
    best_k = -999.0
    
    for trial in range(n_bases):
        if trial == 0:
            U = np.eye(2, dtype=complex)
        elif trial == 1:
            U = np.array([[1,1],[1,-1]], dtype=complex) / np.sqrt(2)  # Hadamard
        else:
            H = np.random.randn(2, 2) + 1j * np.random.randn(2, 2)
            U, _ = np.linalg.qr(H)
        
        p_x = np.zeros(dk)
        S_B_x = np.zeros(dk)
        S_E_x = np.zeros(dk)
        
        for x in range(dk):
            rho_rest = np.zeros((ds*dk*ds, ds*dk*ds), dtype=complex)
            for kA1 in range(dk):
                for kA2 in range(dk):
                    coeff = U[kA1, x].conj() * U[kA2, x]
                    for sA1 in range(ds):
                        for kB1 in range(dk):
                            for sB1 in range(ds):
                                for sA2 in range(ds):
                                    for kB2 in range(dk):
                                        for sB2 in range(ds):
                                            i = kA1*8 + sA1*4 + kB1*2 + sB1
                                            j = kA2*8 + sA2*4 + kB2*2 + sB2
                                            ri = sA1*(dk*ds) + kB1*ds + sB1
                                            rj = sA2*(dk*ds) + kB2*ds + sB2
                                            rho_rest[ri, rj] += coeff * rho[i, j]
            
            p_x[x] = np.trace(rho_rest).real
            if p_x[x] > 1e-15:
                rho_rest /= p_x[x]
                
                rho_B = np.zeros((dk*ds, dk*ds), dtype=complex)
                for sA in range(ds):
                    for kB1 in range(dk):
                        for sB1 in range(ds):
                            for kB2 in range(dk):
                                for sB2 in range(ds):
                                    rho_B[kB1*ds+sB1, kB2*ds+sB2] += rho_rest[sA*(dk*ds)+kB1*ds+sB1, sA*(dk*ds)+kB2*ds+sB2]
                S_B_x[x] = von_neumann(rho_B)
                
                rho_E = np.zeros((ds, ds), dtype=complex)
                for kB in range(dk):
                    for sB in range(ds):
                        for sA1 in range(ds):
                            for sA2 in range(ds):
                                rho_E[sA1, sA2] += rho_rest[sA1*(dk*ds)+kB*ds+sB, sA2*(dk*ds)+kB*ds+sB]
                S_E_x[x] = von_neumann(rho_E)
        
        # Unconditioned
        rho_B_unc = np.zeros((dk*ds, dk*ds), dtype=complex)
        for kA in range(dk):
            for sA in range(ds):
                for kB1 in range(dk):
                    for sB1 in range(ds):
                        for kB2 in range(dk):
                            for sB2 in range(ds):
                                rho_B_unc[kB1*ds+sB1, kB2*ds+sB2] += rho[kA*8+sA*4+kB1*2+sB1, kA*8+sA*4+kB2*2+sB2]
        S_B = von_neumann(rho_B_unc)
        
        rho_E_unc = np.zeros((ds, ds), dtype=complex)
        for kA in range(dk):
            for kB in range(dk):
                for sB in range(ds):
                    for sA1 in range(ds):
                        for sA2 in range(ds):
                            rho_E_unc[sA1, sA2] += rho[kA*8+sA1*4+kB*2+sB, kA*8+sA2*4+kB*2+sB]
        S_E = von_neumann(rho_E_unc)
        
        I_XB = S_B - sum(p_x[x]*S_B_x[x] for x in range(dk) if p_x[x] > 1e-15)
        I_XE = S_E - sum(p_x[x]*S_E_x[x] for x in range(dk) if p_x[x] > 1e-15)
        
        best_k = max(best_k, I_XB - I_XE)
    
    return best_k


def extract_features(rho, d=4):
    """Extract unitarily invariant features from density matrix."""
    dk, ds = 2, 2
    n = d * d
    
    # Eigenvalues
    eigs = np.sort(np.linalg.eigvalsh(rho))
    rank = np.sum(eigs > 1e-10)
    purity = np.trace(rho @ rho).real
    
    # PT spectrum
    pt = partial_transpose_4(rho)
    pt_eigs = np.sort(np.linalg.eigvalsh(pt))
    pt_min = pt_eigs[0]
    is_ppt = pt_min >= -1e-5  # numerical tolerance for golden section optimization
    
    # Reduced states
    rho_A = np.trace(rho.reshape(d,d,d,d), axis1=1, axis2=3)
    rho_B = np.trace(rho.reshape(d,d,d,d), axis1=0, axis2=2)
    S_A = von_neumann(rho_A)
    S_B = von_neumann(rho_B)
    S_AB = von_neumann(rho)
    mutual_info = S_A + S_B - S_AB
    
    # Realignment
    R = rho.reshape(d,d,d,d).transpose(0,2,1,3).reshape(n,n)
    realign_norm = np.linalg.norm(R, 'nuc')
    
    # Max mixed distance
    A_dist = np.linalg.norm(rho_A - np.eye(d)/d)
    B_dist = np.linalg.norm(rho_B - np.eye(d)/d)
    
    return {
        'rank': rank, 'purity': purity,
        'eig_min': eigs[0], 'eig_max': eigs[-1], 'eig_std': np.std(eigs),
        'pt_min': pt_min, 'is_ppt': int(is_ppt),
        'pt_boundary_dist': abs(pt_min),
        'pt_neg_count': int(np.sum(pt_eigs < -1e-10)),
        'S_A': S_A, 'S_B': S_B, 'S_AB': S_AB,
        'mutual_info': mutual_info,
        'mutual_info_norm': mutual_info / (2 * np.log2(d)) if d > 1 else 0,
        'realign_norm': realign_norm,
        'A_max_mixed_dist': A_dist,
        'B_max_mixed_dist': B_dist,
        'purity_norm': purity * d,
        'rank_norm': rank / n,
    }


def process_point(args):
    """Process single alpha point: build state, extract features, compute K_DW."""
    alpha, idx = args
    
    rho = build_flower_alpha(alpha)
    features = extract_features(rho)
    
    # K_DW (20 bases for speed, key rate still found)
    kdw = compute_kdw_d4(rho, n_bases=20)
    
    features['alpha'] = alpha
    features['kdw'] = kdw
    features['d'] = 4
    features['dk'] = 2
    features['ds'] = 2
    features['idx'] = idx
    
    return features


# ── Main ──

if __name__ == '__main__':
    N_POINTS = 10000
    N_WORKERS = 30
    
    print("=" * 60)
    print("  PATH A: Alpha Sweep (d=4 flower)")
    print(f"  {N_POINTS} points, {N_WORKERS} CPU cores")
    print(f"  alpha: 0 -> pi/4 ({N_POINTS} steps)")
    print(f"  Expected SA region: alpha ~ pi/8 = {np.pi/8:.4f}")
    print("=" * 60)
    
    alphas = np.linspace(0.001, np.pi/4 - 0.001, N_POINTS)
    args = [(a, i) for i, a in enumerate(alphas)]
    
    t0 = time.time()
    
    # Process in chunks for progress reporting
    chunk_size = N_POINTS // 10
    all_results = []
    
    with Pool(N_WORKERS) as pool:
        for chunk_start in range(0, N_POINTS, chunk_size):
            chunk = args[chunk_start:chunk_start + chunk_size]
            results = pool.map(process_point, chunk)
            all_results.extend(results)
            
            elapsed = time.time() - t0
            done = len(all_results)
            rate = done / elapsed if elapsed > 0 else 0
            eta = (N_POINTS - done) / rate if rate > 0 else 0
            
            # Quick stats
            kdw_pos = sum(1 for r in all_results if r['kdw'] > 0.001)
            ppt_count = sum(1 for r in all_results if r['is_ppt'])
            
            print(f"  [{done:5d}/{N_POINTS}] {elapsed:5.1f}s "
                  f"({rate:.0f} pts/s, ETA {eta:.0f}s) "
                  f"PPT: {ppt_count} K_DW>0: {kdw_pos}")
    
    total_time = time.time() - t0
    
    # Convert to arrays
    feature_names = sorted(all_results[0].keys())
    data = {k: np.array([r[k] for r in all_results]) for k in feature_names}
    
    # Save
    np.savez('sa_data/path_a_d4.npz', **data)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"  COMPLETE: {N_POINTS} points in {total_time:.1f}s")
    print(f"  Saved: sa_data/path_a_d4.npz")
    print(f"{'='*60}")
    
    # Stats
    kdw = data['kdw']
    alpha = data['alpha']
    ppt = data['is_ppt']
    
    print(f"\n  PPT states: {np.sum(ppt)} / {N_POINTS} ({100*np.mean(ppt):.1f}%)")
    print(f"  K_DW > 0: {np.sum(kdw > 0.001)}")
    print(f"  K_DW max: {np.max(kdw):.6f} at alpha={alpha[np.argmax(kdw)]:.4f}")
    print(f"  K_DW range: [{np.min(kdw):.4f}, {np.max(kdw):.4f}]")
    
    # Transition point: where K_DW crosses 0
    for i in range(1, len(kdw)):
        if kdw[i-1] <= 0 and kdw[i] > 0:
            print(f"  Transition (0->+): alpha={alpha[i]:.6f}")
        if kdw[i-1] > 0 and kdw[i] <= 0:
            print(f"  Transition (+->0): alpha={alpha[i]:.6f}")
    
    print(f"\n  Features per point: {len(feature_names)}")
    print(f"  Feature list: {feature_names}")
