"""
prove_kdw_sb.py — Computational evidence for K_DW ≈ S(B) conjecture
Step 1: Extend data to d=40..100
Step 2: Analyze optimal U structure (is it always Hadamard?)
Step 3: Decompose K_DW = χ(B:X) - χ(E:X) analytically
"""
import numpy as np
from multiprocessing import Pool
import time, os, sys

def S(rho):
    e = np.linalg.eigvalsh(rho); e = e[e>1e-15]
    return -np.sum(e*np.log2(e)) if len(e)>0 else 0.0

def make_ppt_state(dA, dB, seed=None):
    """Generate random PPT state via channel construction (native method)."""
    if seed is not None: np.random.seed(seed)
    d = dA * dB
    # Random Kraus-like construction → guaranteed PPT at boundary
    n_kraus = d
    Ks = []
    for _ in range(n_kraus):
        K = np.random.randn(dB, dA) + 1j*np.random.randn(dB, dA)
        Ks.append(K)
    # Build Choi: ρ = (1/dA) Σ_{ij} |i><j| ⊗ N(|i><j|)
    rho = np.zeros((d, d), dtype=complex)
    for i in range(dA):
        for j in range(dA):
            eij = np.zeros((dA, dA), dtype=complex); eij[i,j] = 1
            N_eij = sum(K @ eij @ K.conj().T for K in Ks)
            rho[i*dB:(i+1)*dB, j*dB:(j+1)*dB] = N_eij
    rho /= np.trace(rho).real
    
    # Check PPT
    rho_pt = rho.reshape(dA,dB,dA,dB).transpose(0,3,2,1).reshape(d,d)
    if np.linalg.eigvalsh(rho_pt).min() < -1e-10:
        return None  # NPT, skip
    return rho

def kdw_with_best_U(rho, dA, dB, n_bases=200):
    """Returns K_DW AND the optimal U matrix."""
    ev, evec = np.linalg.eigh(rho)
    m = ev > 1e-14; lam = ev[m]; phi = evec[:, m]; r = len(lam)
    if r == 0: return 0, None, {}
    
    sq = np.sqrt(lam); phi_r = phi.reshape(dA, dB, r)
    SE = S(np.diag(lam))
    rB = sum(rho[a*dB:(a+1)*dB, a*dB:(a+1)*dB] for a in range(dA))
    SB = S(rB)
    
    # Also compute S(A)
    rA = np.zeros((dA, dA), dtype=complex)
    for a in range(dA):
        for ap in range(dA):
            rA[a, ap] = sum(rho[a*dB+b, ap*dB+b] for b in range(dB))
    SA = S(rA)
    
    best = -999; best_U = None
    chi_B_best = chi_E_best = 0
    
    for t in range(n_bases):
        if t == 0:
            U = np.eye(dA, dtype=complex)
        elif t == 1 and dA == 2:
            # Hadamard
            U = np.array([[1,1],[1,-1]], dtype=complex) / np.sqrt(2)
        else:
            U, _ = np.linalg.qr(np.random.randn(dA,dA)+1j*np.random.randn(dA,dA))
        
        beta = np.einsum('ax,abk->xkb', U.conj(), phi_r)
        px = (np.sum(np.abs(beta)**2, axis=2)) @ lam
        sSB = sSE = 0
        for x in range(dA):
            if px[x] < 1e-15: continue
            wb = sq[:, None] * beta[x]; w = np.sum(wb, 0)
            sSB += px[x] * S(np.outer(w, w.conj()) / px[x])
            sSE += px[x] * S(np.outer(sq, sq) * (beta[x].conj() @ beta[x].T).T / px[x])
        
        val = (SB - sSB) - (SE - sSE)
        if val > best:
            best = val; best_U = U.copy()
            chi_B_best = SB - sSB
            chi_E_best = SE - sSE
    
    info = {
        'SA': SA, 'SB': SB, 'SE': SE, 'S_AB': S(rho),
        'chi_B': chi_B_best, 'chi_E': chi_E_best,
        'ratio': best / SB if SB > 0.01 else 0,
        'delta': SB - best,  # K_DW = S(B) - delta
    }
    return best, best_U, info

def analyze_optimal_U(U, dA):
    """Check if optimal U is close to Hadamard."""
    if U is None or dA != 2:
        return {'type': 'unknown'}
    
    H = np.array([[1,1],[1,-1]]) / np.sqrt(2)
    # Check overlap with Hadamard (up to global phase)
    overlap = np.abs(np.trace(U.conj().T @ H)) / dA
    
    # Check if U is close to any Pauli
    paulis = {
        'I': np.eye(2),
        'H': H,
        'X': np.array([[0,1],[1,0]]) / np.sqrt(2) * np.sqrt(2),
        'σ_x': np.array([[0,1],[1,0]], dtype=complex),
    }
    best_match = 'none'; best_ov = 0
    for name, P in paulis.items():
        ov = np.abs(np.trace(U.conj().T @ P)) / dA
        if ov > best_ov:
            best_ov = ov; best_match = name
    
    return {'hadamard_overlap': overlap, 'best_match': best_match, 'best_overlap': best_ov}

def worker(args):
    dA, dB, seed = args
    rho = make_ppt_state(dA, dB, seed)
    if rho is None: return None
    kdw, U_opt, info = kdw_with_best_U(rho, dA, dB, n_bases=100)
    if kdw < 0.01: return None
    u_info = analyze_optimal_U(U_opt, dA)
    return {**info, 'kdw': kdw, 'dA': dA, 'dB': dB, 'd': dA*dB, **u_info}


if __name__ == '__main__':
    print("="*65)
    print("  PROVING K_DW ≈ S(B): Computational Evidence")
    print("="*65)
    
    # Phase 1: Analyze existing verified states
    print("\n  Phase 1: Re-analyze verified states with U decomposition")
    print("  " + "─"*55)
    
    existing = [
        ('optimized_ppt_2x4.npz', 2, 4),
        ('unstructured_3x3.npz', 3, 3),
        ('optimized_ppt_2x5.npz', 2, 5),
        ('native_d12_2x6.npz', 2, 6),
        ('native_d14_2x7.npz', 2, 7),
        ('native_d15_3x5.npz', 3, 5),
        ('native_d16_2x8.npz', 2, 8),
        ('native_d18_2x9.npz', 2, 9),
        ('native_d20_2x10.npz', 2, 10),
        ('native_d21_3x7.npz', 3, 7),
        ('embedded_2x12.npz', 2, 12),
        ('embedded_2x15.npz', 2, 15),
    ]
    
    print(f"\n  {'d':>3} {'dAxdB':>6} {'K_DW':>7} {'S(B)':>7} {'δ=SB-K':>7} {'K/SB':>6} {'χ(B:X)':>7} {'χ(E:X)':>7} {'U_opt':>8}")
    print(f"  {'─'*62}")
    
    deltas = []; ratios = []
    for fname, dA, dB in existing:
        fpath = f'sa_data/{fname}'
        if not os.path.exists(fpath): continue
        rho = np.load(fpath)['rho']
        if rho.shape[0] != dA*dB: continue
        
        kdw, U_opt, info = kdw_with_best_U(rho, dA, dB, n_bases=300)
        u_info = analyze_optimal_U(U_opt, dA)
        
        u_str = f"{u_info.get('best_match','?')}({u_info.get('best_overlap',0):.2f})" if dA==2 else "N/A"
        print(f"  {dA*dB:>3} {dA}x{dB:>2}  {kdw:>7.3f} {info['SB']:>7.3f} {info['delta']:>7.3f} {info['ratio']:>6.3f} {info['chi_B']:>7.3f} {info['chi_E']:>7.3f} {u_str:>8}")
        deltas.append(info['delta']); ratios.append(info['ratio'])
    
    print(f"\n  Average δ = S(B) - K_DW = {np.mean(deltas):.4f} ± {np.std(deltas):.4f}")
    print(f"  Average K_DW/S(B) = {np.mean(ratios):.4f} ± {np.std(ratios):.4f}")
    
    # Phase 2: Generate new high-d states
    print(f"\n  Phase 2: Generate SA states d=40..100 to test scaling")
    print("  " + "─"*55)
    
    high_d_results = []
    for dB in [20, 25, 30, 35, 40, 50]:
        dA = 2; d = dA * dB
        print(f"  Scanning d={d} ({dA}x{dB})...", end=" ", flush=True)
        found = 0; t0 = time.time()
        for seed in range(500):
            rho = make_ppt_state(dA, dB, seed=seed*137+dB)
            if rho is None: continue
            kdw, U_opt, info = kdw_with_best_U(rho, dA, dB, n_bases=50)
            if kdw > 0.1:
                u_info = analyze_optimal_U(U_opt, dA)
                high_d_results.append({**info, 'kdw': kdw, 'd': d, 'dA': dA, 'dB': dB, 'seed': seed})
                found += 1
                if found >= 3: break
        print(f"found {found} in {time.time()-t0:.1f}s")
    
    if high_d_results:
        print(f"\n  {'d':>4} {'K_DW':>7} {'S(B)':>7} {'δ':>7} {'K/SB':>6}")
        print(f"  {'─'*35}")
        for r in high_d_results:
            print(f"  {r['d']:>4} {r['kdw']:>7.3f} {r['SB']:>7.3f} {r['SB']-r['kdw']:>7.3f} {r['kdw']/r['SB']:>6.3f}")
    
    # Phase 3: Test the conjecture K_DW = S(B) - δ(d)
    all_results = []
    for fname, dA, dB in existing:
        fpath = f'sa_data/{fname}'
        if not os.path.exists(fpath): continue
        rho = np.load(fpath)['rho']
        if rho.shape[0] != dA*dB: continue
        kdw, _, info = kdw_with_best_U(rho, dA, dB, n_bases=300)
        all_results.append({'d': dA*dB, 'dA': dA, 'dB': dB, 'kdw': kdw, **info})
    all_results.extend(high_d_results)
    
    if len(all_results) > 5:
        ds = np.array([r['d'] for r in all_results])
        kdws = np.array([r['kdw'] for r in all_results])
        sbs = np.array([r['SB'] for r in all_results])
        deltas_all = sbs - kdws
        ratios_all = kdws / np.maximum(sbs, 0.01)
        
        print(f"\n{'='*65}")
        print(f"  CONJECTURE ANALYSIS: K_DW = S(B) - δ")
        print(f"{'='*65}")
        print(f"  n = {len(all_results)} data points, d = {ds.min()}..{ds.max()}")
        print(f"  δ = S(B) - K_DW:")
        print(f"    mean = {np.mean(deltas_all):.4f}")
        print(f"    std  = {np.std(deltas_all):.4f}")
        print(f"    min  = {np.min(deltas_all):.4f}")
        print(f"    max  = {np.max(deltas_all):.4f}")
        print(f"  K_DW/S(B) ratio:")
        print(f"    mean = {np.mean(ratios_all):.4f}")
        
        # Fit δ(d) 
        from numpy.linalg import lstsq
        X = np.column_stack([np.log2(ds), np.ones(len(ds))])
        c, _, _, _ = lstsq(X, deltas_all, rcond=None)
        pred_delta = X @ c
        ss_res = np.sum((deltas_all - pred_delta)**2)
        ss_tot = np.sum((deltas_all - deltas_all.mean())**2)
        r2_delta = 1 - ss_res/ss_tot
        
        print(f"\n  Fitted: δ(d) = {c[0]:.4f}·log₂(d) + {c[1]:.4f}  (R²={r2_delta:.4f})")
        print(f"\n  ══> CONJECTURE:")
        print(f"  ══> K_DW = S(ρ_B) - [{c[0]:.2f}·log₂(d) + {c[1]:.2f}]")
        print(f"  ══> or equivalently:")
        print(f"  ══> K_DW ≈ S(ρ_B) - {np.mean(deltas_all):.2f}  (constant δ, R²={1-np.var(deltas_all)/np.var(kdws):.3f})")
    
    print(f"\n{'='*65}")
