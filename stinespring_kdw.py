"""
stinespring_kdw.py — Full K_DW via Stinespring purification.
Computes K_DW = max_U [I(X;B) - I(X;E)] with Eve from purification.
For d=4,6,8,10 SA candidates.
"""
import numpy as np
from multiprocessing import Pool
import time
import os

def von_neumann(rho):
    e = np.linalg.eigvalsh(rho)
    e = e[e > 1e-15]
    return -np.sum(e * np.log2(e)) if len(e) > 0 else 0.0


def kdw_stinespring(rho, dA, dB, n_bases=500):
    """Full K_DW = I(X;B) - I(X;E) via Stinespring purification.
    
    Purification: |ψ⟩_ABE = Σ_k √λ_k |φ_k⟩_AB |k⟩_E
    After measurement x on A:
        |β_k^x⟩_B = ⟨u_x|_A |φ_k⟩_AB
        ρ_E^x = (1/p_x) Σ_{kl} √(λ_k λ_l) ⟨β_l^x|β_k^x⟩ |k⟩⟨l|
    """
    d = dA * dB
    
    # Eigendecomposition of ρ
    eigvals, eigvecs = np.linalg.eigh(rho)
    # Keep only positive eigenvalues
    mask = eigvals > 1e-14
    lam = eigvals[mask]
    phi = eigvecs[:, mask]  # d × r matrix, columns are |φ_k⟩
    r = len(lam)
    sqrt_lam = np.sqrt(lam)
    
    # S(E) = S(ρ_AB) since purification
    S_E_unc = von_neumann(np.diag(lam))
    
    # Unconditional ρ_B
    rho_B_unc = np.zeros((dB, dB), dtype=complex)
    for a in range(dA):
        rho_B_unc += rho[a*dB:(a+1)*dB, a*dB:(a+1)*dB]
    S_B_unc = von_neumann(rho_B_unc)
    
    best_kdw = -999.0
    
    for trial in range(n_bases):
        if trial == 0:
            U = np.eye(dA, dtype=complex)
        else:
            H = np.random.randn(dA, dA) + 1j * np.random.randn(dA, dA)
            U, _ = np.linalg.qr(H)
        
        p_x = np.zeros(dA)
        S_B_x = np.zeros(dA)
        S_E_x = np.zeros(dA)
        
        for x in range(dA):
            u_x = U[:, x]  # dA-dim measurement vector
            
            # |β_k^x⟩_B = Σ_a u_x[a]* φ_k[a*dB:(a+1)*dB]
            beta = np.zeros((r, dB), dtype=complex)
            for k in range(r):
                for a in range(dA):
                    beta[k] += u_x[a].conj() * phi[a*dB:(a+1)*dB, k]
            
            # p(x) = Σ_k λ_k ||β_k^x||²
            norms_sq = np.array([np.dot(beta[k].conj(), beta[k]).real for k in range(r)])
            p_x[x] = np.dot(lam, norms_sq)
            
            if p_x[x] < 1e-15:
                continue
            
            # ρ_B^x = (1/p_x) Σ_{kl} √(λ_k λ_l) |β_k^x⟩⟨β_l^x|
            rho_B_x = np.zeros((dB, dB), dtype=complex)
            for k in range(r):
                for l in range(r):
                    rho_B_x += sqrt_lam[k] * sqrt_lam[l] * np.outer(beta[k], beta[l].conj())
            rho_B_x /= p_x[x]
            S_B_x[x] = von_neumann(rho_B_x)
            
            # ρ_E^x: r × r matrix
            # (ρ_E^x)_{kl} = √(λ_k λ_l) ⟨β_l^x|β_k^x⟩ / p_x
            rho_E_x = np.zeros((r, r), dtype=complex)
            for k in range(r):
                for l in range(r):
                    rho_E_x[k, l] = sqrt_lam[k] * sqrt_lam[l] * np.dot(beta[l].conj(), beta[k])
            rho_E_x /= p_x[x]
            S_E_x[x] = von_neumann(rho_E_x)
        
        I_XB = S_B_unc - sum(p_x[x] * S_B_x[x] for x in range(dA) if p_x[x] > 1e-15)
        I_XE = S_E_unc - sum(p_x[x] * S_E_x[x] for x in range(dA) if p_x[x] > 1e-15)
        
        K = I_XB - I_XE
        best_kdw = max(best_kdw, K)
    
    return best_kdw


def worker(args):
    """Parallel worker for K_DW computation."""
    rho, dA, dB, seed, n_bases = args
    np.random.seed(seed)
    return kdw_stinespring(rho, dA, dB, n_bases)


if __name__ == '__main__':
    N_WORKERS = 30
    BASES_PER_WORKER = 200
    
    print("=" * 60)
    print("  STINESPRING K_DW — Full Verification")
    print(f"  {N_WORKERS} cores × {BASES_PER_WORKER} bases = {N_WORKERS*BASES_PER_WORKER} total bases")
    print("=" * 60)
    
    candidates = []
    
    # d=4 flower (reference)
    if os.path.exists('sa_data/path_a_d4.npz'):
        # Build flower state at alpha=pi/8
        import sys
        sys.path.insert(0, '.')
        try:
            from flower_general import build_state
            rho4, info = build_state(2, 2, np.pi/8)
            candidates.append(('d=4 flower', 2, 2, rho4))
        except:
            pass
    
    # d=6 projection
    if os.path.exists('sa_data/sa_candidate_d6.npz'):
        rho6 = np.load('sa_data/sa_candidate_d6.npz')['rho']
        candidates.append(('d=6 proj', 6, 6, rho6))  # bipartite 6x6
    
    # d=8 SDP+perturb
    if os.path.exists('sa_data/optimized_ppt_2x4.npz'):
        rho8 = np.load('sa_data/optimized_ppt_2x4.npz')['rho']
        candidates.append(('d=8 SDP', 2, 4, rho8))
    
    # d=10 SDP+perturb
    if os.path.exists('sa_data/optimized_ppt_2x5.npz'):
        rho10 = np.load('sa_data/optimized_ppt_2x5.npz')['rho']
        candidates.append(('d=10 SDP', 2, 5, rho10))
    
    results = {}
    
    for name, dA, dB, rho in candidates:
        d = dA * dB
        print(f"\n  --- {name} (dA={dA}, dB={dB}, d={d}) ---")
        
        # Verify PPT
        pt = rho.reshape(dA, dB, dA, dB).transpose(0, 3, 2, 1).reshape(d, d)
        pt_min = np.min(np.linalg.eigvalsh(pt))
        R = rho.reshape(dA, dB, dA, dB).transpose(0, 2, 1, 3).reshape(d, d)
        realign = np.linalg.norm(R, 'nuc')
        
        print(f"    PPT: pt_min={pt_min:.2e} {'PASS' if pt_min >= -1e-5 else 'FAIL'}")
        print(f"    ENT: realign={realign:.4f} {'PASS' if realign > 1.001 else 'FAIL'}")
        
        # Parallel K_DW
        t0 = time.time()
        args = [(rho, dA, dB, seed, BASES_PER_WORKER) for seed in range(N_WORKERS)]
        with Pool(N_WORKERS) as pool:
            kdw_results = pool.map(worker, args)
        elapsed = time.time() - t0
        
        best_kdw = max(kdw_results)
        
        results[name] = {
            'dA': dA, 'dB': dB, 'd': d,
            'pt_min': pt_min, 'realign': realign,
            'kdw': best_kdw,
        }
        
        sa_str = "*** SUPERACTIVATION ***" if best_kdw > 0.001 else ""
        print(f"    K_DW = {best_kdw:.6f} ({elapsed:.1f}s, {N_WORKERS*BASES_PER_WORKER} bases)")
        print(f"    {sa_str}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"  STINESPRING K_DW RESULTS")
    print(f"{'='*60}")
    print(f"  {'Name':<15} {'d':>3} {'PPT':>5} {'ENT':>5} {'K_DW':>10} {'SA':>4}")
    print(f"  {'-'*45}")
    for name, r in results.items():
        ppt = 'YES' if r['pt_min'] >= -1e-5 else 'no'
        ent = 'YES' if r['realign'] > 1.001 else 'no'
        sa = 'YES' if r['kdw'] > 0.001 else 'no'
        print(f"  {name:<15} {r['d']:3d} {ppt:>5} {ent:>5} {r['kdw']:10.6f} {sa:>4}")
    print(f"{'='*60}")
