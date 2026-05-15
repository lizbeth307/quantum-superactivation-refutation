"""
fill_gaps.py — Find native SA for d=14,15,16,18,20 to complete the dimension map.
Also does noise robustness analysis on best states.
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

def search_native(args):
    dA, dB, seed = args
    d = dA * dB
    rng = np.random.RandomState(seed)
    best = None
    for trial in range(300):
        G = rng.randn(d, d) + 1j * rng.randn(d, d)
        rho = G @ G.conj().T; rho /= np.trace(rho).real
        for _ in range(30):
            pt = rho.reshape(dA,dB,dA,dB).transpose(0,3,2,1).reshape(d,d)
            evals, evecs = np.linalg.eigh(pt)
            if np.min(evals) >= -1e-10: break
            evals = np.maximum(evals, 0)
            pt_proj = evecs @ np.diag(evals) @ evecs.conj().T
            rho = pt_proj.reshape(dA,dB,dA,dB).transpose(0,3,2,1).reshape(d,d)
            rho = (rho + rho.conj().T) / 2
            evals2, evecs2 = np.linalg.eigh(rho)
            evals2 = np.maximum(evals2, 0)
            rho = evecs2 @ np.diag(evals2) @ evecs2.conj().T
            rho /= np.trace(rho).real
        pt = rho.reshape(dA,dB,dA,dB).transpose(0,3,2,1).reshape(d,d)
        pt_min = np.min(np.linalg.eigvalsh(pt))
        if pt_min < -1e-6: continue
        R = rho.reshape(dA,dB,dA,dB).transpose(0,2,1,3).reshape(d,d)
        realign = np.linalg.norm(R, 'nuc')
        if realign > 1.001:
            kdw = kdw_stinespring(rho, dA, dB, 30)
            if best is None or kdw > best[0]:
                best = (kdw, pt_min, realign, rho.copy())
    return best

def robustness_worker(args):
    rho0, dA, dB, noise_level, seed = args
    d = dA * dB
    rng = np.random.RandomState(seed)
    noise = rng.randn(d,d)+1j*rng.randn(d,d)
    noise = noise@noise.conj().T; noise /= np.trace(noise)
    rho = (1-noise_level)*rho0 + noise_level*(np.eye(d)/d)  # depolarizing noise
    rho = (rho+rho.conj().T)/2; rho /= np.trace(rho).real
    kdw = kdw_stinespring(rho, dA, dB, 30)
    return kdw

if __name__ == '__main__':
    N = 30
    
    # ═══ PART 1: Native search for gap dimensions ═══
    print("=" * 65)
    print("  NATIVE SA SEARCH: d=14,15,16,18,20,21,25")
    print("=" * 65)
    
    gap_dims = [
        (2, 7, 14), (3, 5, 15), (5, 3, 15), (2, 8, 16), (4, 4, 16),
        (2, 9, 18), (3, 6, 18), (2, 10, 20), (4, 5, 20), (3, 7, 21), (5, 5, 25),
    ]
    
    native_results = {}
    for dA, dB, d in gap_dims:
        args = [(dA, dB, seed) for seed in range(N)]
        t0 = time.time()
        with Pool(N) as pool:
            results = pool.map(search_native, args)
        elapsed = time.time() - t0
        
        hits = [r for r in results if r is not None]
        if hits:
            best = max(hits, key=lambda x: x[0])
            kdw, pt_min, realign, rho = best
            sa = "SA!" if kdw > 0.001 else ""
            print(f"  d={d:3d} ({dA}x{dB}): K_DW={kdw:.4f} real={realign:.4f} ({elapsed:.1f}s) {sa}")
            if kdw > 0.001:
                np.savez(f'sa_data/native_d{d}_{dA}x{dB}.npz', rho=rho)
                native_results[d] = (kdw, f'{dA}x{dB}')
        else:
            print(f"  d={d:3d} ({dA}x{dB}): No PPT+ENT ({elapsed:.1f}s)")
    
    # ═══ PART 2: Noise robustness ═══
    print(f"\n{'='*65}")
    print("  NOISE ROBUSTNESS ANALYSIS")
    print(f"{'='*65}")
    
    test_states = [
        ('d=8 (2x4)', np.load('sa_data/optimized_ppt_2x4.npz')['rho'], 2, 4),
        ('d=10 (2x5)', np.load('sa_data/optimized_ppt_2x5.npz')['rho'], 2, 5),
    ]
    if 'native_d12_2x6.npz' in [f for f in __import__('os').listdir('sa_data')]:
        test_states.append(('d=12 (2x6)', np.load('sa_data/native_d12_2x6.npz')['rho'], 2, 6))
    
    noise_levels = [0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
    
    print(f"\n  {'State':>15} | " + " | ".join(f"p={p:.2f}" for p in noise_levels))
    print(f"  {'-'*15}-+-" + "-+-".join("-"*6 for _ in noise_levels))
    
    for name, rho0, dA, dB in test_states:
        kdw_vs_noise = []
        for p in noise_levels:
            args = [(rho0, dA, dB, p, seed) for seed in range(N)]
            with Pool(N) as pool:
                results = pool.map(robustness_worker, args)
            kdw_vs_noise.append(max(results))
        
        vals = " | ".join(f"{k:6.3f}" for k in kdw_vs_noise)
        print(f"  {name:>15} | {vals}")
    
    # Critical noise threshold
    print(f"\n  Critical depolarizing noise level (K_DW → 0):")
    for name, rho0, dA, dB in test_states:
        for p in np.arange(0.0, 1.0, 0.02):
            rho = (1-p)*rho0 + p*np.eye(dA*dB)/(dA*dB)
            kdw = kdw_stinespring(rho, dA, dB, 20)
            if kdw < 0.01:
                print(f"    {name}: p_crit ≈ {p:.2f}")
                break
    
    print(f"\n{'='*65}")
