"""extreme_v3.py — d=400..1000, vectorized, 30 cores, 50 bases each."""
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
            # wb[k,b] = sqrt_lam[k] * beta[x,k,b], shape (r, dB)
            wb = sqrt_lam[:,None] * beta[x]  # (r, dB)
            # w[b] = sum_k sqrt_lam[k] * beta[x,k,b] = sum over k of wb[k,b]
            w = np.sum(wb, axis=0)  # (dB,)
            # rho_B_x = |w><w| / p_x (rank-1 but this is the correct formula)
            rho_B_x = np.outer(w, w.conj()) / p_x[x]  # (dB, dB)
            sum_pSB += p_x[x] * von_neumann(rho_B_x)
            # rho_E_x[k,l] = sqrt_k sqrt_l <beta_l|beta_k> / p_x
            gram = beta[x].conj() @ beta[x].T  # gram[l,k] = <beta_l|beta_k>
            rho_E_x = np.outer(sqrt_lam, sqrt_lam) * gram.T / p_x[x]
            sum_pSE += p_x[x] * von_neumann(rho_E_x)
        kdw = (S_B_unc-sum_pSB)-(S_E_unc-sum_pSE)
        best = max(best, kdw)
    return best

if __name__ == '__main__':
    rho8=np.load('sa_data/optimized_ppt_2x4.npz')['rho']
    rho10=np.load('sa_data/optimized_ppt_2x5.npz')['rho']
    for name,base,k in [('d=400',rho8,50),('d=500',rho10,50),('d=800',rho8,100),('d=1000',rho10,100)]:
        rho=np.kron(base,np.eye(k)/k); dA=2; dB=rho.shape[0]//2; d=dA*dB
        print(f'{name} ({dA}x{dB}): 30 cores x 50 bases = 1500 total...', flush=True)
        t0=time.time()
        args=[(rho,dA,dB,seed,50) for seed in range(30)]
        with Pool(30) as pool: results=pool.map(kdw_vectorized, args)
        best=max(results); elapsed=time.time()-t0
        pred=0.901*np.log2(d)-1.573
        print(f'  K_DW={best:.6f} pred={pred:.4f} [{elapsed:.0f}s]', flush=True)
    print('DONE', flush=True)
