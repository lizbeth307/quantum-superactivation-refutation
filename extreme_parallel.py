"""extreme_vectorized.py — d=400,500,1000 with VECTORIZED numpy + 30 cores."""
import numpy as np, time
from multiprocessing import Pool

def von_neumann(rho):
    e = np.linalg.eigvalsh(rho); e = e[e>1e-15]
    return -np.sum(e*np.log2(e)) if len(e)>0 else 0.0

def kdw_vectorized(rho, dA, dB, n_bases=5, seed=0):
    """Vectorized K_DW — O(r*dB) instead of O(r^2*dB)."""
    np.random.seed(seed)
    d = dA * dB
    eigvals, eigvecs = np.linalg.eigh(rho)
    mask = eigvals > 1e-14
    lam = eigvals[mask]
    phi = eigvecs[:, mask]  # (d, r)
    r = len(lam)
    sqrt_lam = np.sqrt(lam)
    
    # Reshape phi: (dA, dB, r)
    phi_reshaped = phi.reshape(dA, dB, r)
    
    S_E_unc = von_neumann(np.diag(lam))
    rho_B_unc = np.einsum('abk,abl,k,l->bl', phi_reshaped, phi_reshaped.conj(), sqrt_lam, sqrt_lam)
    # Actually simpler: rho_B = sum over a of rho[a*dB:(a+1)*dB, a*dB:(a+1)*dB]
    rho_B_unc2 = sum(rho[a*dB:(a+1)*dB, a*dB:(a+1)*dB] for a in range(dA))
    S_B_unc = von_neumann(rho_B_unc2)
    
    best = -999.0
    for trial in range(n_bases):
        if trial == 0:
            U = np.eye(dA, dtype=complex)
        else:
            H = np.random.randn(dA, dA) + 1j * np.random.randn(dA, dA)
            U, _ = np.linalg.qr(H)
        
        # beta[x, k, b] = sum_a U[a,x]* phi[a,b,k]
        # phi_reshaped: (dA, dB, r), U.conj(): (dA, dA)
        # beta: (dA, r, dB) — for each measurement outcome x
        beta = np.einsum('ax,abk->xkb', U.conj(), phi_reshaped)  # (dA, r, dB)
        
        # norms_sq[x, k] = sum_b |beta[x,k,b]|^2
        norms_sq = np.sum(np.abs(beta)**2, axis=2)  # (dA, r)
        
        # p_x[x] = sum_k lam[k] * norms_sq[x,k]
        p_x = norms_sq @ lam  # (dA,)
        
        I_XB = S_B_unc
        I_XE = S_E_unc
        sum_pSB = 0.0
        sum_pSE = 0.0
        
        for x in range(dA):
            if p_x[x] < 1e-15:
                continue
            
            # rho_B_x = sum_{k,l} sqrt_lam[k]*sqrt_lam[l] * outer(beta[x,k], beta[x,l].conj()) / p_x[x]
            # Vectorized: weighted_beta[k,b] = sqrt_lam[k] * beta[x,k,b]
            wb = sqrt_lam[:, None] * beta[x]  # (r, dB)
            rho_B_x = (wb.T @ wb.conj()) / p_x[x]  # (dB, dB) — FAST!
            sum_pSB += p_x[x] * von_neumann(rho_B_x)
            
            # rho_E_x[k,l] = sqrt_lam[k]*sqrt_lam[l] * <beta[x,l]|beta[x,k]> / p_x[x]
            # overlap[k,l] = beta[x,l].conj() @ beta[x,k]
            overlap = beta[x].conj() @ beta[x].T  # (r, r)
            rho_E_x = np.outer(sqrt_lam, sqrt_lam) * overlap / p_x[x]
            sum_pSE += p_x[x] * von_neumann(rho_E_x)
        
        kdw = (S_B_unc - sum_pSB) - (S_E_unc - sum_pSE)
        best = max(best, kdw)
    
    return best

def _worker(args):
    return kdw_vectorized(*args)

if __name__ == '__main__':
    rho8 = np.load('sa_data/optimized_ppt_2x4.npz')['rho']
    rho10 = np.load('sa_data/optimized_ppt_2x5.npz')['rho']

    configs = [
        ('d=400', rho8, 50),
        ('d=500', rho10, 50),
        ('d=800', rho8, 100),
        ('d=1000', rho10, 100),
    ]

    for name, base, k in configs:
        rho = np.kron(base, np.eye(k)/k)
        dA = 2; dB = rho.shape[0] // dA; d = dA * dB
        
        # PPT check
        pt = rho.reshape(dA, dB, dA, dB).transpose(0, 3, 2, 1).reshape(d, d)
        pt_min = np.min(np.linalg.eigvalsh(pt))
        
        print(f"{name} ({dA}x{dB}): 30 cores, pt_min={pt_min:.2e}", flush=True)
        
        t0 = time.time()
        args = [(rho, dA, dB, seed, 5) for seed in range(30)]
        with Pool(30) as pool:
            results = pool.map(_worker, args)
        best = max(results)
        elapsed = time.time() - t0
        pred = 0.901 * np.log2(d) - 1.573
        print(f"  K_DW = {best:.6f}  pred = {pred:.4f}  [{elapsed:.0f}s]", flush=True)

    print("DONE", flush=True)
