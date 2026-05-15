"""
extend_extreme.py — Push SA to d=50, d=100, d=200 via deep embedding chains.
Uses ρ ⊗ I_k/k iterated.
"""
import numpy as np
from multiprocessing import Pool
import time

def von_neumann(rho):
    e = np.linalg.eigvalsh(rho)
    e = e[e > 1e-15]
    return -np.sum(e * np.log2(e)) if len(e) > 0 else 0.0

def kdw_stinespring(rho, dA, dB, n_bases=100):
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
            H = np.random.randn(dA, dA) + 1j * np.random.randn(dA, dA)
            U, _ = np.linalg.qr(H)
        p_x = np.zeros(dA); S_B_x = np.zeros(dA); S_E_x = np.zeros(dA)
        for x in range(dA):
            beta = np.zeros((r, dB), dtype=complex)
            for k in range(r):
                for a in range(dA):
                    beta[k] += U[a, x].conj() * phi[a*dB:(a+1)*dB, k]
            norms_sq = np.array([np.dot(beta[k].conj(), beta[k]).real for k in range(r)])
            p_x[x] = np.dot(lam, norms_sq)
            if p_x[x] < 1e-15: continue
            rho_B_x = sum(sqrt_lam[k]*sqrt_lam[l]*np.outer(beta[k], beta[l].conj())
                         for k in range(r) for l in range(r)) / p_x[x]
            S_B_x[x] = von_neumann(rho_B_x)
            rho_E_x = np.zeros((r, r), dtype=complex)
            for k in range(r):
                for l in range(r):
                    rho_E_x[k, l] = sqrt_lam[k]*sqrt_lam[l]*np.dot(beta[l].conj(), beta[k])
            rho_E_x /= p_x[x]
            S_E_x[x] = von_neumann(rho_E_x)
        I_XB = S_B_unc - sum(p_x[x]*S_B_x[x] for x in range(dA) if p_x[x] > 1e-15)
        I_XE = S_E_unc - sum(p_x[x]*S_E_x[x] for x in range(dA) if p_x[x] > 1e-15)
        best = max(best, I_XB - I_XE)
    return best

def kdw_worker(args):
    rho, dA, dB, seed, n = args
    np.random.seed(seed)
    return kdw_stinespring(rho, dA, dB, n)

if __name__ == '__main__':
    N = 30
    
    rho8 = np.load('sa_data/optimized_ppt_2x4.npz')['rho']   # 2x4
    rho10 = np.load('sa_data/optimized_ppt_2x5.npz')['rho']  # 2x5
    
    print("=" * 60)
    print("  EXTREME DIMENSION SA: d=4 to d=200")
    print("=" * 60)
    
    # Build embedding chain
    configs = [
        # (name, dA, dB, base_rho, k_embed)
        ('d=4 ref', 2, 4, rho8, 1),    # identity = no embedding, just rho8 as 2x4
        ('d=8 base', 2, 4, rho8, 1),
        ('d=10 base', 2, 5, rho10, 1),
        ('d=16', 2, 8, rho8, 2),
        ('d=20', 2, 10, rho10, 2),
        ('d=24', 2, 12, rho8, 3),
        ('d=30', 2, 15, rho10, 3),
        ('d=40', 2, 20, rho8, 5),
        ('d=50', 2, 25, rho10, 5),
        ('d=80', 2, 40, rho8, 10),
        ('d=100', 2, 50, rho10, 10),
        ('d=200', 2, 100, rho10, 20),
    ]
    
    results = []
    
    for name, dA, dB_target, base_rho, k in configs:
        d = dA * dB_target
        dB_base = base_rho.shape[0] // dA
        
        if k == 1:
            rho = base_rho
        else:
            rho = np.kron(base_rho, np.eye(k) / k)
        
        actual_dB = rho.shape[0] // dA
        actual_d = dA * actual_dB
        
        print(f"\n  {name} ({dA}x{actual_dB}, d={actual_d}):")
        
        # PPT check
        pt = rho.reshape(dA, actual_dB, dA, actual_dB).transpose(0, 3, 2, 1).reshape(actual_d, actual_d)
        pt_min = np.min(np.linalg.eigvalsh(pt))
        ppt = pt_min >= -1e-6
        
        print(f"    PPT: {ppt} (pt_min={pt_min:.2e})")
        
        if not ppt:
            print(f"    SKIP: not PPT")
            continue
        
        # K_DW
        t0 = time.time()
        n_bases = max(50, 200 - actual_d)  # fewer bases for large d
        args = [(rho, dA, actual_dB, seed, n_bases) for seed in range(N)]
        with Pool(N) as pool:
            kdw_list = pool.map(kdw_worker, args)
        best_kdw = max(kdw_list)
        elapsed = time.time() - t0
        
        sa = "SA!" if best_kdw > 0.001 else ""
        print(f"    K_DW = {best_kdw:.6f} ({elapsed:.1f}s) {sa}")
        
        results.append((actual_d, best_kdw, name))
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"  SCALING LAW VERIFICATION")
    print(f"{'='*60}")
    print(f"  {'d':>5}  {'K_DW':>10}  {'0.856*log2(d)-1.513':>20}  {'Name'}")
    print(f"  {'-'*50}")
    for d, kdw, name in results:
        predicted = 0.856 * np.log2(d) - 1.513
        print(f"  {d:5d}  {kdw:10.4f}  {predicted:20.4f}  {name}")
    
    # Fit log2 scaling
    ds = np.array([r[0] for r in results])
    ks = np.array([r[1] for r in results])
    from numpy.polynomial import polynomial as P
    log_ds = np.log2(ds)
    c = np.polyfit(log_ds, ks, 1)
    print(f"\n  Fit: K_DW = {c[0]:.4f} * log2(d) + {c[1]:.4f}")
    print(f"  (previous: 0.856 * log2(d) - 1.513)")
    print(f"\n{'='*60}")
