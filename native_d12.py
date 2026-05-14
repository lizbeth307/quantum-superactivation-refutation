"""
native_d12.py — Find NATIVE (non-embedded) PPT entangled state in d=12.
Strategy: construct from known d=4 flower + noise in higher-d shield.
Uses direct key-shield decomposition: dk=2, ds=6 (d=12).
"""
import numpy as np
from multiprocessing import Pool
import time

def von_neumann(rho):
    e = np.linalg.eigvalsh(rho)
    e = e[e > 1e-15]
    return -np.sum(e * np.log2(e)) if len(e) > 0 else 0.0

def kdw_stinespring(rho, dA, dB, n_bases=80):
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

def build_and_test(args):
    """Build a candidate d=12 state and test it."""
    dA, dB, seed = args
    d = dA * dB
    rng = np.random.RandomState(seed)
    
    best = None
    
    for trial in range(200):
        # Strategy: random state → project to PPT cone → check entanglement
        G = rng.randn(d, d) + 1j * rng.randn(d, d)
        rho = G @ G.conj().T
        rho /= np.trace(rho).real
        
        # Project to PPT
        for ppt_iter in range(20):
            # Partial transpose
            pt = rho.reshape(dA, dB, dA, dB).transpose(0, 3, 2, 1).reshape(d, d)
            evals, evecs = np.linalg.eigh(pt)
            
            if np.min(evals) >= -1e-10:
                break  # Already PPT
            
            # Clip negative eigenvalues
            evals_clipped = np.maximum(evals, 0)
            pt_proj = evecs @ np.diag(evals_clipped) @ evecs.conj().T
            
            # Un-partial-transpose
            rho = pt_proj.reshape(dA, dB, dA, dB).transpose(0, 3, 2, 1).reshape(d, d)
            
            # Re-symmetrize and normalize
            rho = (rho + rho.conj().T) / 2
            
            # Project to PSD
            evals2, evecs2 = np.linalg.eigh(rho)
            evals2 = np.maximum(evals2, 0)
            rho = evecs2 @ np.diag(evals2) @ evecs2.conj().T
            rho /= np.trace(rho).real
        
        # Verify PPT
        pt = rho.reshape(dA, dB, dA, dB).transpose(0, 3, 2, 1).reshape(d, d)
        pt_min = np.min(np.linalg.eigvalsh(pt))
        if pt_min < -1e-6:
            continue
        
        # Check entanglement via realignment
        R = rho.reshape(dA, dB, dA, dB).transpose(0, 2, 1, 3).reshape(d, d)
        realign = np.linalg.norm(R, 'nuc')
        
        if realign > 1.001:
            # Compute K_DW
            kdw = kdw_stinespring(rho, dA, dB, 30)
            if best is None or kdw > best[0]:
                best = (kdw, pt_min, realign, rho.copy())
    
    return best

if __name__ == '__main__':
    N = 30
    print("=" * 60)
    print("  NATIVE d=12 SA SEARCH")
    print("=" * 60)
    
    decomps = [(2, 6), (3, 4), (4, 3), (6, 2)]
    
    for dA, dB in decomps:
        d = dA * dB
        print(f"\n  {dA}x{dB} (d={d}):")
        
        args = [(dA, dB, seed) for seed in range(N)]
        t0 = time.time()
        with Pool(N) as pool:
            results = pool.map(build_and_test, args)
        elapsed = time.time() - t0
        
        hits = [r for r in results if r is not None]
        if hits:
            best = max(hits, key=lambda x: x[0])
            kdw, pt_min, realign, rho = best
            sa = "SA!" if kdw > 0.001 else ""
            print(f"    Best: K_DW={kdw:.6f} pt_min={pt_min:.2e} realign={realign:.4f} ({elapsed:.1f}s) {sa}")
            if kdw > 0.001:
                np.savez(f'sa_data/native_d12_{dA}x{dB}.npz', rho=rho)
                print(f"    SAVED: sa_data/native_d12_{dA}x{dB}.npz")
        else:
            print(f"    No PPT+ENT found ({elapsed:.1f}s)")
    
    print(f"\n{'='*60}")
