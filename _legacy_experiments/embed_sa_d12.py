"""
embed_sa_d12.py — Embed known d=8 SA state into d=12 via tensor product.
If ρ_8 has K_DW > 0 in 2x4, then ρ_8 ⊗ σ_3 in 2x(4*3)=2x12 should also work.
"""
import numpy as np
from multiprocessing import Pool
import time

def von_neumann(rho):
    e = np.linalg.eigvalsh(rho)
    e = e[e>1e-15]
    return -np.sum(e*np.log2(e)) if len(e)>0 else 0.0

def kdw_stinespring(rho, dA, dB, n_bases=100):
    d = dA*dB
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

def kdw_worker(args):
    rho, dA, dB, seed, n_bases = args
    np.random.seed(seed)
    return kdw_stinespring(rho, dA, dB, n_bases)

if __name__ == '__main__':
    N = 30
    print("EMBEDDING SA STATES INTO HIGHER DIMENSIONS")
    
    # Strategy 1: ρ_d ⊗ |ψ+⟩⟨ψ+| in bigger space
    # If ρ is 2×4 PPT entangled, then ρ ⊗ I_k/k is 2×(4k) PPT
    # and has at least the same K_DW (tensor product doesn't reduce key rate)
    
    rho8 = np.load('sa_data/optimized_ppt_2x4.npz')['rho']  # 2x4, d=8
    rho10 = np.load('sa_data/optimized_ppt_2x5.npz')['rho']  # 2x5, d=10
    
    print("\n  Strategy 1: ρ ⊗ I_k/k embeddings")
    
    for base_name, rho_base, dA_base, dB_base in [
        ('d=8', rho8, 2, 4),
        ('d=10', rho10, 2, 5),
    ]:
        for k in [2, 3]:
            dA = dA_base  # Alice dimension stays same
            dB = dB_base * k  # Bob gets extended
            d = dA * dB
            
            # ρ_new = ρ_base ⊗ I_k/k
            I_k = np.eye(k) / k
            rho_new = np.kron(rho_base, I_k)
            
            # Verify PPT
            pt = rho_new.reshape(dA, dB, dA, dB).transpose(0,3,2,1).reshape(d,d)
            pt_min = np.min(np.linalg.eigvalsh(pt))
            R = rho_new.reshape(dA, dB, dA, dB).transpose(0,2,1,3).reshape(d,d)
            realign = np.linalg.norm(R, 'nuc')
            
            print(f"\n  {base_name} ⊗ I_{k}/{k} → {dA}x{dB} (d={d}):")
            print(f"    PPT: {pt_min >= -1e-6} (pt_min={pt_min:.2e})")
            print(f"    ENT: {realign > 1.001} (realign={realign:.4f})")
            
            if pt_min >= -1e-6:
                t0 = time.time()
                args = [(rho_new, dA, dB, seed, 200) for seed in range(N)]
                with Pool(N) as pool:
                    results = pool.map(kdw_worker, args)
                best_kdw = max(results)
                elapsed = time.time() - t0
                
                sa = "*** SA ***" if best_kdw > 0.001 else ""
                print(f"    K_DW = {best_kdw:.6f} ({elapsed:.1f}s) {sa}")
                
                if best_kdw > 0.001:
                    np.savez(f'sa_data/embedded_{dA}x{dB}.npz', rho=rho_new)
    
    # Strategy 2: Direct product of two SA states
    print("\n\n  Strategy 2: ρ_8 ⊗ ρ_10 → 4x20 (d=80)")
    rho_cross = np.kron(rho8, rho10)
    dA_c, dB_c = 4, 20  # (2*2) x (4*5)
    d_c = dA_c * dB_c
    pt = rho_cross.reshape(dA_c,dB_c,dA_c,dB_c).transpose(0,3,2,1).reshape(d_c,d_c)
    pt_min = np.min(np.linalg.eigvalsh(pt))
    print(f"  4x20 (d=80): pt_min={pt_min:.2e} {'PPT' if pt_min>=-1e-5 else 'NPT'}")
    
    print(f"\n{'='*60}")
