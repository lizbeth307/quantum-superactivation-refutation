"""chain_embed.py — Test chain embedding rho^{⊗n} vs identity embedding rho⊗I/k."""
import numpy as np, time
from multiprocessing import Pool

def von_neumann(rho):
    e = np.linalg.eigvalsh(rho); e = e[e>1e-15]
    return -np.sum(e*np.log2(e)) if len(e)>0 else 0.0

def kdw_vectorized(args):
    rho, dA, dB, seed, n_bases = args
    np.random.seed(seed)
    d = dA*dB
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
            wb = sqrt_lam[:,None]*beta[x]
            rho_B_x = (wb.T @ wb.conj())/p_x[x]
            sum_pSB += p_x[x]*von_neumann(rho_B_x)
            overlap = beta[x].conj() @ beta[x].T
            rho_E_x = np.outer(sqrt_lam,sqrt_lam)*overlap/p_x[x]
            sum_pSE += p_x[x]*von_neumann(rho_E_x)
        kdw = (S_B_unc-sum_pSB)-(S_E_unc-sum_pSE)
        best = max(best, kdw)
    return best

if __name__ == '__main__':
    rho8 = np.load('sa_data/optimized_ppt_2x4.npz')['rho']  # 2x4
    rho10 = np.load('sa_data/optimized_ppt_2x5.npz')['rho']  # 2x5
    
    print("IDENTITY EMBEDDING BREAKDOWN ANALYSIS", flush=True)
    print("Testing rho8 (2x4) ⊗ I_k/k for various k:\n", flush=True)
    
    for k in [2, 3, 5, 8, 10, 15, 20, 25, 30, 40, 50]:
        rho = np.kron(rho8, np.eye(k)/k)
        dA = 2; dB = rho.shape[0]//2; d = dA*dB
        args = [(rho, dA, dB, seed, 30) for seed in range(30)]
        with Pool(30) as pool: results = pool.map(kdw_vectorized, args)
        best = max(results)
        print(f"  k={k:3d}  d={d:5d}  K_DW={best:8.4f}", flush=True)
    
    print("\nTesting rho10 (2x5) ⊗ I_k/k:\n", flush=True)
    for k in [2, 3, 5, 8, 10, 15, 20, 25, 30, 40, 50]:
        rho = np.kron(rho10, np.eye(k)/k)
        dA = 2; dB = rho.shape[0]//2; d = dA*dB
        args = [(rho, dA, dB, seed, 30) for seed in range(30)]
        with Pool(30) as pool: results = pool.map(kdw_vectorized, args)
        best = max(results)
        print(f"  k={k:3d}  d={d:5d}  K_DW={best:8.4f}", flush=True)
    
    print("\nDONE", flush=True)
