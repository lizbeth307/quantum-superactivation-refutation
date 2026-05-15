"""
bipartite_analysis.py — Systematic analysis of how bipartite decomposition affects K_DW.
For each d, test ALL valid decompositions and find the optimal split.
Also push to d=500 and d=1000 via deep embedding.
"""
import numpy as np
from multiprocessing import Pool
import time

def von_neumann(rho):
    e = np.linalg.eigvalsh(rho)
    e = e[e > 1e-15]
    return -np.sum(e * np.log2(e)) if len(e) > 0 else 0.0

def kdw_stinespring(rho, dA, dB, n_bases=50):
    d = dA * dB
    eigvals, eigvecs = np.linalg.eigh(rho)
    mask = eigvals > 1e-14
    lam = eigvals[mask]; phi = eigvecs[:, mask]; r = len(lam)
    sqrt_lam = np.sqrt(lam)
    S_E_unc = von_neumann(np.diag(lam))
    rho_B_unc = sum(rho[a*dB:(a+1)*dB, a*dB:(a+1)*dB] for a in range(dA))
    S_B_unc = von_neumann(rho_B_unc)
    best = -999.0
    for trial in range(n_bases):
        if trial == 0: U = np.eye(dA, dtype=complex)
        else:
            H = np.random.randn(dA,dA)+1j*np.random.randn(dA,dA)
            U, _ = np.linalg.qr(H)
        p_x = np.zeros(dA); S_B_x = np.zeros(dA); S_E_x = np.zeros(dA)
        for x in range(dA):
            beta = np.zeros((r, dB), dtype=complex)
            for k in range(r):
                for a in range(dA):
                    beta[k] += U[a,x].conj()*phi[a*dB:(a+1)*dB, k]
            norms_sq = np.array([np.dot(beta[k].conj(), beta[k]).real for k in range(r)])
            p_x[x] = np.dot(lam, norms_sq)
            if p_x[x] < 1e-15: continue
            rho_B_x = sum(sqrt_lam[k]*sqrt_lam[l]*np.outer(beta[k],beta[l].conj()) for k in range(r) for l in range(r))/p_x[x]
            S_B_x[x] = von_neumann(rho_B_x)
            rho_E_x = np.zeros((r,r), dtype=complex)
            for k in range(r):
                for l in range(r):
                    rho_E_x[k,l] = sqrt_lam[k]*sqrt_lam[l]*np.dot(beta[l].conj(), beta[k])
            rho_E_x /= p_x[x]
            S_E_x[x] = von_neumann(rho_E_x)
        I_XB = S_B_unc - sum(p_x[x]*S_B_x[x] for x in range(dA) if p_x[x]>1e-15)
        I_XE = S_E_unc - sum(p_x[x]*S_E_x[x] for x in range(dA) if p_x[x]>1e-15)
        best = max(best, I_XB - I_XE)
    return best

def kdw_worker(args):
    rho, dA, dB, seed, n = args
    np.random.seed(seed)
    return kdw_stinespring(rho, dA, dB, n)

if __name__ == '__main__':
    N = 30
    rho8 = np.load('sa_data/optimized_ppt_2x4.npz')['rho']
    rho10 = np.load('sa_data/optimized_ppt_2x5.npz')['rho']
    
    # ═══ PART 1: Optimal bipartite decomposition ═══
    print("=" * 65)
    print("  BIPARTITE DECOMPOSITION ANALYSIS")
    print("=" * 65)
    print(f"\n  Finding optimal dA for each d (fixed state from d=8/10 base)\n")
    
    # For each embedding level, test all valid dA values
    test_configs = [
        # (d_total, base_rho, base_dB, k_embed)
        (8, rho8, 4, 1),
        (16, rho8, 4, 2),
        (24, rho8, 4, 3),
        (40, rho8, 4, 5),
        (80, rho8, 4, 10),
        (10, rho10, 5, 1),
        (20, rho10, 5, 2),
        (30, rho10, 5, 3),
        (50, rho10, 5, 5),
        (100, rho10, 5, 10),
    ]
    
    print(f"  {'d':>5} {'dA':>4} {'dB':>5} {'K_DW':>10} {'log2(dB)':>10} {'K/log2(dB)':>12}")
    print(f"  {'-'*50}")
    
    bipartite_results = []
    for d_total, base_rho, base_dB, k in test_configs:
        if k == 1:
            rho = base_rho
        else:
            rho = np.kron(base_rho, np.eye(k)/k)
        
        actual_d = rho.shape[0]
        # Find all valid factorizations of actual_d
        factorizations = []
        for dA in range(2, min(actual_d, 11)):  # max dA=10
            if actual_d % dA == 0:
                dB = actual_d // dA
                factorizations.append((dA, dB))
        
        for dA, dB in factorizations:
            args = [(rho, dA, dB, seed, 100) for seed in range(N)]
            with Pool(N) as pool:
                kdw_list = pool.map(kdw_worker, args)
            best_kdw = max(kdw_list)
            ratio = best_kdw / np.log2(dB) if dB > 1 else 0
            print(f"  {d_total:5d} {dA:4d} {dB:5d} {best_kdw:10.4f} {np.log2(dB):10.3f} {ratio:12.4f}")
            bipartite_results.append((d_total, dA, dB, best_kdw))
    
    # ═══ PART 2: Extreme dimensions d=500, d=1000 ═══
    print(f"\n{'='*65}")
    print("  EXTREME DIMENSIONS: d=500, d=1000")
    print(f"{'='*65}")
    
    extreme_configs = [
        ('d=400', 2, 200, rho8, 50),
        ('d=500', 2, 250, rho10, 50),
        ('d=800', 2, 400, rho8, 100),
        ('d=1000', 2, 500, rho10, 100),
    ]
    
    for name, dA, dB_target, base_rho, k in extreme_configs:
        rho = np.kron(base_rho, np.eye(k)/k)
        actual_dB = rho.shape[0] // dA
        actual_d = dA * actual_dB
        
        # PPT check
        pt = rho.reshape(dA,actual_dB,dA,actual_dB).transpose(0,3,2,1).reshape(actual_d,actual_d)
        pt_min = np.min(np.linalg.eigvalsh(pt))
        ppt = pt_min >= -1e-6
        
        print(f"\n  {name} ({dA}x{actual_dB}, d={actual_d}):")
        print(f"    PPT: {ppt} (pt_min={pt_min:.2e})")
        
        if not ppt:
            print(f"    SKIP: not PPT")
            continue
        
        t0 = time.time()
        n_bases = max(30, 150 - actual_d//10)
        args = [(rho, dA, actual_dB, seed, n_bases) for seed in range(N)]
        with Pool(N) as pool:
            kdw_list = pool.map(kdw_worker, args)
        best_kdw = max(kdw_list)
        elapsed = time.time() - t0
        
        predicted = 0.901 * np.log2(actual_d) - 1.573
        print(f"    K_DW = {best_kdw:.6f} (predicted: {predicted:.4f}) ({elapsed:.1f}s)")
        print(f"    Ratio K/log2(d) = {best_kdw/np.log2(actual_d):.4f}")
    
    print(f"\n{'='*65}")
