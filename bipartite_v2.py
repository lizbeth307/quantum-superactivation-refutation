"""bipartite_v2.py — Bipartite analysis + d=400,500"""
import numpy as np, time, sys
from multiprocessing import Pool

def von_neumann(rho):
    e = np.linalg.eigvalsh(rho); e = e[e>1e-15]
    return -np.sum(e*np.log2(e)) if len(e)>0 else 0.0

def kdw_stinespring(rho, dA, dB, n_bases=50):
    d = dA*dB
    eigvals, eigvecs = np.linalg.eigh(rho)
    mask = eigvals > 1e-14
    lam = eigvals[mask]; phi = eigvecs[:,mask]; r = len(lam)
    sqrt_lam = np.sqrt(lam)
    S_E_unc = von_neumann(np.diag(lam))
    rho_B_unc = sum(rho[a*dB:(a+1)*dB, a*dB:(a+1)*dB] for a in range(dA))
    S_B_unc = von_neumann(rho_B_unc)
    best = -999.0
    for trial in range(n_bases):
        if trial==0: U=np.eye(dA,dtype=complex)
        else: H=np.random.randn(dA,dA)+1j*np.random.randn(dA,dA); U,_=np.linalg.qr(H)
        p_x=np.zeros(dA); S_B_x=np.zeros(dA); S_E_x=np.zeros(dA)
        for x in range(dA):
            beta=np.zeros((r,dB),dtype=complex)
            for k in range(r):
                for a in range(dA):
                    beta[k]+=U[a,x].conj()*phi[a*dB:(a+1)*dB,k]
            norms_sq=np.array([np.dot(beta[k].conj(),beta[k]).real for k in range(r)])
            p_x[x]=np.dot(lam,norms_sq)
            if p_x[x]<1e-15: continue
            rho_B_x=sum(sqrt_lam[k]*sqrt_lam[l]*np.outer(beta[k],beta[l].conj()) for k in range(r) for l in range(r))/p_x[x]
            S_B_x[x]=von_neumann(rho_B_x)
            rho_E_x=np.zeros((r,r),dtype=complex)
            for k in range(r):
                for l in range(r):
                    rho_E_x[k,l]=sqrt_lam[k]*sqrt_lam[l]*np.dot(beta[l].conj(),beta[k])
            rho_E_x/=p_x[x]
            S_E_x[x]=von_neumann(rho_E_x)
        I_XB=S_B_unc-sum(p_x[x]*S_B_x[x] for x in range(dA) if p_x[x]>1e-15)
        I_XE=S_E_unc-sum(p_x[x]*S_E_x[x] for x in range(dA) if p_x[x]>1e-15)
        best=max(best, I_XB-I_XE)
    return best

def worker(args):
    rho,dA,dB,seed,n=args; np.random.seed(seed); return kdw_stinespring(rho,dA,dB,n)

if __name__ == '__main__':
    N=30
    rho8=np.load('sa_data/optimized_ppt_2x4.npz')['rho']
    rho10=np.load('sa_data/optimized_ppt_2x5.npz')['rho']

    print("BIPARTITE DECOMPOSITION ANALYSIS", flush=True)
    
    configs = [
        (8, rho8, 1), (16, rho8, 2), (40, rho8, 5),
        (10, rho10, 1), (20, rho10, 2), (50, rho10, 5),
    ]

    for d_total, base_rho, k in configs:
        rho = base_rho if k==1 else np.kron(base_rho, np.eye(k)/k)
        actual_d = rho.shape[0]
        for dA in range(2, min(actual_d, 9)):
            if actual_d % dA == 0:
                dB = actual_d // dA
                args = [(rho, dA, dB, seed, 60) for seed in range(N)]
                with Pool(N) as pool: kdw_list = pool.map(worker, args)
                best_kdw = max(kdw_list)
                ratio = best_kdw/np.log2(dB) if dB>1 else 0
                print(f"  d={d_total:4d}  {dA}x{dB:<5d}  K_DW={best_kdw:8.4f}  K/log2(dB)={ratio:.4f}", flush=True)

    # d=400 and d=500
    print("\nEXTREME DIMENSIONS", flush=True)
    for name, dA, base, k in [('d=400',2,rho8,50), ('d=500',2,rho10,50)]:
        rho = np.kron(base, np.eye(k)/k)
        actual_dB = rho.shape[0]//dA; actual_d = dA*actual_dB
        t0=time.time()
        args = [(rho, dA, actual_dB, seed, 30) for seed in range(N)]
        with Pool(N) as pool: kdw_list = pool.map(worker, args)
        best = max(kdw_list); elapsed=time.time()-t0
        pred = 0.901*np.log2(actual_d)-1.573
        print(f"  {name}: K_DW={best:.4f} (pred={pred:.4f}) ({elapsed:.1f}s)", flush=True)

    print("DONE", flush=True)
