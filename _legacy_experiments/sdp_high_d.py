"""
sdp_high_d.py — SDP PPT search + K_DW for d=12,14,16.
Uses optimized SDP + perturbation + Stinespring pipeline.
"""
import numpy as np
from multiprocessing import Pool
import time
import os

try:
    import cvxpy as cp
except ImportError:
    print("CVXPY required"); exit(1)

os.makedirs('sa_data', exist_ok=True)

def von_neumann(rho):
    e = np.linalg.eigvalsh(rho)
    e = e[e > 1e-15]
    return -np.sum(e * np.log2(e)) if len(e) > 0 else 0.0

def sdp_find(dA, dB, n_trials=15):
    """SDP PPT entangled search."""
    d = dA * dB
    rho = cp.Variable((d, d), hermitian=True)
    pt_rho = cp.Variable((d, d), hermitian=True)
    
    constraints = [rho >> 0, cp.trace(rho) == 1, pt_rho >> 0]
    for i in range(dA):
        for j in range(dB):
            for k in range(dA):
                for l in range(dB):
                    constraints.append(pt_rho[i*dB+j, k*dB+l] == rho[i*dB+l, k*dB+j])
    
    best = None
    for trial in range(n_trials):
        np.random.seed(trial + 100)
        G = np.random.randn(d, d) + 1j * np.random.randn(d, d)
        W = G + G.conj().T; W /= np.linalg.norm(W)
        
        prob = cp.Problem(cp.Maximize(cp.real(cp.trace(W @ rho))), constraints)
        try:
            prob.solve(solver=cp.SCS, verbose=False, max_iters=10000)
        except:
            continue
        
        if prob.status in ['optimal', 'optimal_inaccurate'] and rho.value is not None:
            rv = rho.value
            pt = rv.reshape(dA,dB,dA,dB).transpose(0,3,2,1).reshape(d,d)
            pt_min = np.min(np.linalg.eigvalsh(pt))
            R = rv.reshape(dA,dB,dA,dB).transpose(0,2,1,3).reshape(d,d)
            realign = np.linalg.norm(R, 'nuc')
            
            if pt_min >= -1e-4 and realign > 1.001:
                if best is None or realign > best[1]:
                    best = (rv, realign, pt_min)
    return best

def perturb_for_ixb(rho0, dA, dB, seed, n_trials=300):
    """Perturb to maximize I(X;B)."""
    rng = np.random.RandomState(seed)
    d = dA * dB
    best = None
    for _ in range(n_trials):
        eps = rng.uniform(0.01, 0.5)
        G = rng.randn(d, d) + 1j * rng.randn(d, d)
        noise = G @ G.conj().T; noise /= np.trace(noise)
        rho = (1-eps)*rho0 + eps*noise
        rho = (rho + rho.conj().T) / 2; rho /= np.trace(rho).real
        
        pt = rho.reshape(dA,dB,dA,dB).transpose(0,3,2,1).reshape(d,d)
        pt_min = np.min(np.linalg.eigvalsh(pt))
        if pt_min < -1e-6:
            continue
        
        R = rho.reshape(dA,dB,dA,dB).transpose(0,2,1,3).reshape(d,d)
        realign = np.linalg.norm(R, 'nuc')
        if realign < 1.001:
            continue
        
        # Quick I(X;B)
        ixb = 0.0
        for trial in range(10):
            if trial == 0: U = np.eye(dA, dtype=complex)
            else:
                H = rng.randn(dA,dA) + 1j*rng.randn(dA,dA)
                U, _ = np.linalg.qr(H)
            p_x = np.zeros(dA); S_B_x = np.zeros(dA)
            for x in range(dA):
                rB = np.zeros((dB,dB), dtype=complex)
                for a1 in range(dA):
                    for a2 in range(dA):
                        c = U[a1,x].conj()*U[a2,x]
                        rB += c*rho[a1*dB:(a1+1)*dB, a2*dB:(a2+1)*dB]
                p_x[x] = np.trace(rB).real
                if p_x[x] > 1e-15:
                    rB /= p_x[x]; S_B_x[x] = von_neumann(rB)
            rBu = sum(rho[a*dB:(a+1)*dB, a*dB:(a+1)*dB] for a in range(dA))
            S_B = von_neumann(rBu)
            I = S_B - sum(p_x[x]*S_B_x[x] for x in range(dA) if p_x[x]>1e-15)
            ixb = max(ixb, I)
        
        if best is None or ixb > best[0]:
            best = (ixb, pt_min, realign, rho.copy())
    return best

def kdw_stinespring(rho, dA, dB, n_bases=100):
    """K_DW via Stinespring."""
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
                    beta[k] += U[a,x].conj() * phi[a*dB:(a+1)*dB, k]
            norms_sq = np.array([np.dot(beta[k].conj(), beta[k]).real for k in range(r)])
            p_x[x] = np.dot(lam, norms_sq)
            if p_x[x] < 1e-15: continue
            rho_B_x = sum(sqrt_lam[k]*sqrt_lam[l]*np.outer(beta[k],beta[l].conj()) for k in range(r) for l in range(r)) / p_x[x]
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

def stinespring_worker(args):
    rho, dA, dB, seed, n_bases = args
    np.random.seed(seed)
    return kdw_stinespring(rho, dA, dB, n_bases)

def perturb_wrapper(args):
    rho0, dA, dB, seed, n_trials = args
    return perturb_for_ixb(rho0, dA, dB, seed, n_trials)


if __name__ == '__main__':
    N = 30
    print("=" * 60)
    print("  HIGH-D SA SEARCH: d=12,14,16")
    print("=" * 60)
    
    configs = [
        (2, 6, 12), (3, 4, 12), (2, 7, 14), (2, 8, 16), (4, 4, 16),
    ]
    
    for dA, dB, d in configs:
        print(f"\n  --- {dA}x{dB} (d={d}) ---")
        
        # Step 1: SDP search
        print(f"    SDP search (15 trials)...")
        t0 = time.time()
        result = sdp_find(dA, dB, n_trials=15)
        t1 = time.time()
        
        if result is None:
            print(f"    No PPT+ENT found ({t1-t0:.1f}s)")
            continue
        
        rho0, realign, pt_min = result
        print(f"    SDP: realign={realign:.4f} pt_min={pt_min:.2e} ({t1-t0:.1f}s)")
        
        # Step 2: Perturbation for I(X;B)
        print(f"    Perturbation (30 cores x 300 trials)...")
        args = [(rho0, dA, dB, s, 300) for s in range(N)]
        t0 = time.time()
        with Pool(N) as pool:
            pert_results = pool.map(perturb_wrapper, 
                                   [(rho0, dA, dB, s, 300) for s in range(N)])
        t1 = time.time()
        
        hits = [r for r in pert_results if r is not None]
        if not hits:
            # Use SDP state directly
            rho_best = rho0
            print(f"    No perturbation improvement, using SDP state ({t1-t0:.1f}s)")
        else:
            best = max(hits, key=lambda x: x[0])
            rho_best = best[3]
            print(f"    Perturbed: I(X;B)={best[0]:.6f} realign={best[2]:.4f} ({t1-t0:.1f}s)")
        
        # Step 3: Stinespring K_DW
        print(f"    Stinespring K_DW (30 cores x 200 bases)...")
        t0 = time.time()
        args = [(rho_best, dA, dB, seed, 200) for seed in range(N)]
        with Pool(N) as pool:
            kdw_results = pool.map(stinespring_worker, args)
        t1 = time.time()
        
        best_kdw = max(kdw_results)
        sa = "*** SUPERACTIVATION ***" if best_kdw > 0.001 else ""
        print(f"    K_DW = {best_kdw:.6f} ({t1-t0:.1f}s) {sa}")
        
        if best_kdw > 0.001:
            np.savez(f'sa_data/sa_verified_d{d}_{dA}x{dB}.npz', rho=rho_best)
            print(f"    Saved: sa_data/sa_verified_d{d}_{dA}x{dB}.npz")
    
    print(f"\n{'='*60}")



