"""
gen_d810_data.py — Generate training data for d=8,10 via noise perturbation.
Uses Stinespring K_DW for accurate labels.
"""
import numpy as np
from multiprocessing import Pool
import time
import os

def von_neumann(rho):
    e = np.linalg.eigvalsh(rho)
    e = e[e > 1e-15]
    return -np.sum(e * np.log2(e)) if len(e) > 0 else 0.0

def kdw_stinespring(rho, dA, dB, n_bases=30):
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

def gen_worker(args):
    dA, dB, rho0, eps, idx = args
    d = dA*dB; n = d
    rng = np.random.RandomState(idx)
    
    # Perturb with noise
    noise = rng.randn(n,n) + 1j*rng.randn(n,n)
    noise = noise @ noise.conj().T; noise /= np.trace(noise)
    rho = (1-eps)*rho0 + eps*noise
    rho = (rho+rho.conj().T)/2; rho /= np.trace(rho).real
    
    # Features
    eigs = np.sort(np.linalg.eigvalsh(rho))
    pt = rho.reshape(dA,dB,dA,dB).transpose(0,3,2,1).reshape(n,n)
    pt_eigs = np.sort(np.linalg.eigvalsh(pt))
    rA = sum(rho.reshape(dA,dB,dA,dB)[:,:,i,i] for i in range(dB)) if dB <= dA else np.trace(rho.reshape(dA,dB,dA,dB), axis1=1, axis2=3)
    rB = sum(rho[a*dB:(a+1)*dB, a*dB:(a+1)*dB] for a in range(dA))
    S_A = von_neumann(rA); S_B = von_neumann(rB); S_AB = von_neumann(rho)
    R = rho.reshape(dA,dB,dA,dB).transpose(0,2,1,3).reshape(n,n)
    
    kdw = kdw_stinespring(rho, dA, dB, 30)
    
    return {
        'rank_norm': np.sum(eigs>1e-10)/n, 'purity_norm': np.trace(rho@rho).real*d,
        'eig_min': eigs[0], 'eig_max': eigs[-1], 'eig_std': np.std(eigs),
        'pt_min': pt_eigs[0], 'is_ppt': int(pt_eigs[0]>=-1e-5),
        'pt_boundary_dist': abs(pt_eigs[0]), 'pt_neg_count': int(np.sum(pt_eigs<-1e-10)),
        'S_A': S_A, 'S_B': S_B, 'S_AB': S_AB,
        'mutual_info': S_A+S_B-S_AB, 'mutual_info_norm': (S_A+S_B-S_AB)/(2*np.log2(d)),
        'realign_norm': np.linalg.norm(R, 'nuc'),
        'A_max_mixed_dist': np.linalg.norm(rA - np.eye(dA)/dA),
        'B_max_mixed_dist': np.linalg.norm(rB - np.eye(dB)/dB),
        'd': d, 'eps': eps, 'kdw': kdw,
    }


if __name__ == '__main__':
    N = 30
    os.makedirs('sa_data', exist_ok=True)
    
    print("Generating d=8,10 training data (Stinespring K_DW)")
    
    for dA, dB, fname in [(2,4,'optimized_ppt_2x4.npz'), (2,5,'optimized_ppt_2x5.npz')]:
        d = dA*dB
        rho0 = np.load(f'sa_data/{fname}')['rho']
        
        eps_vals = np.linspace(0, 0.6, 2500)
        args = [(dA, dB, rho0, eps, i) for i, eps in enumerate(eps_vals)]
        
        print(f"\n  d={d} ({dA}x{dB}): {len(args)} points...")
        t0 = time.time()
        with Pool(N) as pool:
            results = pool.map(gen_worker, args)
        elapsed = time.time() - t0
        
        keys = sorted(results[0].keys())
        data = {k: np.array([r[k] for r in results]) for k in keys}
        np.savez(f'sa_data/path_d_d{d}.npz', **data)
        
        ppt = np.sum(data['is_ppt'])
        kdw_pos = np.sum(data['kdw'] > 0.001)
        print(f"  {len(args)} pts in {elapsed:.1f}s")
        print(f"  PPT: {ppt}, K_DW>0: {kdw_pos}, K_DW: [{data['kdw'].min():.4f}, {data['kdw'].max():.4f}]")
    
    print("\n  DONE")
