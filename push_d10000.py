"""push_d10000.py — Push to d=10000 (break 10-bit barrier)."""
import numpy as np, time, gc
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
    del eigvals, eigvecs; gc.collect()
    sqrt_lam = np.sqrt(lam)
    phi_r = phi.reshape(dA, dB, r)
    del phi; gc.collect()
    S_E_unc = von_neumann(np.diag(lam))
    rho_B_unc = sum(rho[a*dB:(a+1)*dB, a*dB:(a+1)*dB] for a in range(dA))
    S_B_unc = von_neumann(rho_B_unc)
    del rho_B_unc; gc.collect()
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
            del rho_B_x, rho_E_x, gram, wb, w
        del beta
        kdw = (S_B_unc-sum_pSB)-(S_E_unc-sum_pSE)
        best = max(best, kdw)
    return best

if __name__ == '__main__':
    rho10 = np.load('sa_data/optimized_ppt_2x5.npz')['rho']  # 2x5
    
    # d=10000: rho10 x I_1000/1000 -> 2x5000
    k = 1000
    print(f"Building d=10000 matrix ({2}x{5*k})...", flush=True)
    rho = np.kron(rho10, np.eye(k)/k)
    dA = 2; dB = rho.shape[0] // 2; d = dA * dB
    mem_gb = rho.nbytes / 1e9
    print(f"  Matrix size: {rho.shape}, Memory: {mem_gb:.1f} GB", flush=True)
    print(f"d={d} ({dA}x{dB}): 2 workers x 3 bases...", flush=True)
    t0 = time.time()
    args = [(rho, dA, dB, s, 3) for s in range(2)]
    with Pool(2) as pool:
        results = pool.map(kdw_vectorized, args)
    best = max(results)
    elapsed = time.time() - t0
    pred = 0.932 * np.log2(d) - 1.532
    print(f"  K_DW = {best:.6f}  pred = {pred:.4f}  [{elapsed:.0f}s]", flush=True)
    print("DONE", flush=True)
