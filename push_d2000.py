"""push_d2000.py — Push to d=2000 and d=5000 with vectorized 30-core engine."""
import numpy as np, time
from multiprocessing import Pool

def von_neumann(rho):
    e = np.linalg.eigvalsh(rho); e = e[e>1e-15]
    return -np.sum(e*np.log2(e)) if len(e)>0 else 0.0

def kdw_vectorized(args):
    rho, dA, dB, seed, n_bases = args
    np.random.seed(seed)
    eigvals, eigvecs = np.linalg.eigh(rho)
    mask = eigvals > 1e-14
    lam = eigvals[mask]; phi = eigvecs[:,mask]; r = len(lam)
    sqrt_lam = np.sqrt(lam)
    phi_r = phi.reshape(dA, dB, r)
    S_E_unc = von_neumann(np.diag(lam))
    rho_B_unc = sum(rho[a*dB:(a+1)*dB, a*dB:(a+1)*dB] for a in range(dA))
    S_B_unc = von_neumann(rho_B_unc)
    best = -999.0
    for trial in range(n_bases):
        if trial==0: U=np.eye(dA,dtype=complex)
        else: H=np.random.randn(dA,dA)+1j*np.random.randn(dA,dA); U,_=np.linalg.qr(H)
        beta = np.einsum('ax,abk->xkb', U.conj(), phi_r)
        norms_sq = np.sum(np.abs(beta)**2, axis=2)
        p_x = norms_sq @ lam
        sum_pSB=0.0; sum_pSE=0.0
        for x in range(dA):
            if p_x[x]<1e-15: continue
            wb = sqrt_lam[:,None] * beta[x]
            w = np.sum(wb, axis=0)
            rho_B_x = np.outer(w, w.conj()) / p_x[x]
            sum_pSB += p_x[x] * von_neumann(rho_B_x)
            gram = beta[x].conj() @ beta[x].T
            rho_E_x = np.outer(sqrt_lam, sqrt_lam) * gram.T / p_x[x]
            sum_pSE += p_x[x] * von_neumann(rho_E_x)
        kdw = (S_B_unc-sum_pSB)-(S_E_unc-sum_pSE)
        best = max(best, kdw)
    return best

if __name__ == '__main__':
    rho8 = np.load('sa_data/optimized_ppt_2x4.npz')['rho']
    rho10 = np.load('sa_data/optimized_ppt_2x5.npz')['rho']
    
    configs = [
        ('d=2000', rho10, 200),
        ('d=4000', rho8, 500),
        ('d=5000', rho10, 500),
    ]
    
    for name, base, k in configs:
        rho = np.kron(base, np.eye(k)/k)
        dA = 2; dB = rho.shape[0] // 2; d = dA * dB
        print(f"{name} ({dA}x{dB}): 30 cores x 30 bases...", flush=True)
        t0 = time.time()
        args = [(rho, dA, dB, seed, 30) for seed in range(30)]
        with Pool(30) as pool:
            results = pool.map(kdw_vectorized, args)
        best = max(results)
        elapsed = time.time() - t0
        pred = 0.915 * np.log2(d) - 1.452
        print(f"  K_DW = {best:.6f}  pred = {pred:.4f}  [{elapsed:.0f}s]", flush=True)
    
    print("DONE", flush=True)
