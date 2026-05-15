"""
sa_high_d_v2.py — d=12: use d=4 flower structure on 2x2 key + noise in 3x3 shield.
Build flower ρ in d=dk*ds with dk=2,ds=6 or dk=3,ds=4 or dk=4,ds=3.
Then compute K_DW on the 4-subsystem (keyA,shieldA,keyB,shieldB) decomposition.
"""
import numpy as np
from multiprocessing import Pool
import time

def von_neumann(rho):
    e = np.linalg.eigvalsh(rho)
    e = e[e > 1e-15]
    return -np.sum(e * np.log2(e)) if len(e) > 0 else 0.0

def build_flower_state(dk, ds, alpha, shield_params=None, rng=None):
    """Build generalized flower state for dk x ds."""
    d = dk * ds
    n = d * d
    
    if rng is None:
        rng = np.random.RandomState(0)
    
    # Bell states for dk-dim key
    bells = []
    for m in range(dk):
        for nn_idx in range(dk):
            psi = np.zeros(dk*dk, dtype=complex)
            for j in range(dk):
                phase = np.exp(2j * np.pi * j * m / dk)
                psi[j*dk + ((j+nn_idx)%dk)] = phase / np.sqrt(dk)
            bells.append(psi)
    
    n_bell = dk * dk
    
    # Weights: convex combination
    w = np.abs(rng.randn(n_bell))
    w /= w.sum()
    
    # Shield states: random rank-1 in ds x ds
    shields = []
    for k in range(n_bell):
        v = rng.randn(ds) + 1j * rng.randn(ds)
        v /= np.linalg.norm(v)
        chi = np.kron(v, v)  # ds^2 dim
        shields.append(np.outer(chi, chi.conj()))
    
    # Build rho in order: kA, sA, kB, sB
    rho = np.zeros((n, n), dtype=complex)
    
    # Permutation: from (kA, kB) x (sA, sB) to (kA, sA, kB, sB)
    perm = np.zeros(n, dtype=int)
    for kA in range(dk):
        for sA in range(ds):
            for kB in range(dk):
                for sB in range(ds):
                    old = kA*(dk*ds*ds) + sA*(dk*ds) + kB*ds + sB  # target order
                    new_kron = (kA*dk+kB)*(ds*ds) + sA*ds + sB  # kron order (bell x shield)
                    perm[old] = new_kron
    
    for i in range(n_bell):
        pp = np.outer(bells[i], bells[i].conj())
        block = np.kron(pp, shields[i])
        rho += w[i] * block
    
    # Apply permutation  
    rho_perm = rho[np.ix_(perm, perm)]
    rho_perm = (rho_perm + rho_perm.conj().T) / 2
    rho_perm /= np.trace(rho_perm).real
    
    return rho_perm, w

def kdw_flower(rho, dk, ds, n_bases=50):
    """K_DW for flower state with 4-subsystem (kA,sA,kB,sB) structure."""
    d = dk * ds
    
    eigvals, eigvecs = np.linalg.eigh(rho)
    mask = eigvals > 1e-14
    lam = eigvals[mask]; phi = eigvecs[:, mask]; r = len(lam)
    sqrt_lam = np.sqrt(lam)
    
    # For flower: Alice has kA (dk dim), Bob has sA,kB,sB (ds*dk*ds dim)
    dA = dk; dB = ds * dk * ds
    n = dA * dB
    
    S_E_unc = von_neumann(np.diag(lam))
    rho_B_unc = np.zeros((dB, dB), dtype=complex)
    for a in range(dA):
        rho_B_unc += rho[a*dB:(a+1)*dB, a*dB:(a+1)*dB]
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

def search_worker(args):
    dk, ds, seed = args
    rng = np.random.RandomState(seed)
    d = dk * ds
    n = d * d
    
    best = None
    for trial in range(50):
        alpha = rng.uniform(0.01, np.pi/4)
        rho, w = build_flower_state(dk, ds, alpha, rng=rng)
        
        # Check PPT
        pt = rho.reshape(d,d,d,d).transpose(0,3,2,1).reshape(n,n)
        pt_min = np.min(np.linalg.eigvalsh(pt))
        
        R = rho.reshape(d,d,d,d).transpose(0,2,1,3).reshape(n,n)
        realign = np.linalg.norm(R, 'nuc')
        
        if pt_min >= -1e-6 and realign > 1.001:
            kdw = kdw_flower(rho, dk, ds, 30)
            if best is None or kdw > best[0]:
                best = (kdw, pt_min, realign, rho.copy())
    
    return best


if __name__ == '__main__':
    N = 30
    print("=" * 60)
    print("  FLOWER-BASED SA SEARCH for d=12,16,20")
    print("=" * 60)
    
    for dk, ds in [(2,6), (3,4), (4,3), (2,8), (4,4), (2,10), (4,5), (5,4)]:
        d = dk * ds
        print(f"\n  dk={dk}, ds={ds} (d={d}):")
        
        args = [(dk, ds, seed) for seed in range(N)]
        t0 = time.time()
        with Pool(N) as pool:
            results = pool.map(search_worker, args)
        elapsed = time.time() - t0
        
        hits = [r for r in results if r is not None]
        if hits:
            best = max(hits, key=lambda x: x[0])
            kdw, pt_min, realign, rho = best
            sa = "SA!" if kdw > 0.001 else ""
            print(f"    K_DW={kdw:.6f} pt_min={pt_min:.2e} realign={realign:.4f} ({elapsed:.1f}s) {sa}")
            if kdw > 0.001:
                np.savez(f'sa_data/flower_d{d}_dk{dk}_ds{ds}.npz', rho=rho)
        else:
            print(f"    No PPT+ENT found ({elapsed:.1f}s)")
    
    print(f"\n{'='*60}")
