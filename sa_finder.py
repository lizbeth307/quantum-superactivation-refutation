"""
sa_finder.py — Phase 0.5: Universal PPT Entangled State Finder.
Projection-based search + K_DW verification.
30 CPU cores, d=3..12.
"""
import numpy as np
from multiprocessing import Pool
import time
import os

os.makedirs('sa_data', exist_ok=True)


def search_ppt_ent(args):
    """Search for PPT entangled states via projection method."""
    d, seed, n_trials = args
    rng = np.random.RandomState(seed)
    n = d * d
    best = None
    
    for trial in range(n_trials):
        # Strategy 1: Random Wishart state, project to PPT
        k = rng.randint(d, 3*d)
        G = rng.randn(n, k) + 1j * rng.randn(n, k)
        rho = G @ G.conj().T
        rho /= np.trace(rho)
        
        # PT
        pt = rho.reshape(d,d,d,d).transpose(0,3,2,1).reshape(n,n)
        eigs, vecs = np.linalg.eigh(pt)
        
        if np.min(eigs) >= -1e-10:
            # Already PPT — check entanglement
            R = rho.reshape(d,d,d,d).transpose(0,2,1,3).reshape(n,n)
            realign = np.linalg.norm(R, 'nuc')
            if realign > 1.001:
                if best is None or realign > best[0]:
                    best = (realign, np.min(eigs), rho.copy())
            continue
        
        # Project to PPT: clip negative eigenvalues
        eigs_clip = np.maximum(eigs, 0)
        pt_fixed = vecs @ np.diag(eigs_clip) @ vecs.conj().T
        pt_fixed = (pt_fixed + pt_fixed.conj().T) / 2
        pt_fixed /= np.trace(pt_fixed)
        
        # Undo PT
        rho_ppt = pt_fixed.reshape(d,d,d,d).transpose(0,3,2,1).reshape(n,n)
        rho_ppt = (rho_ppt + rho_ppt.conj().T) / 2
        
        # Check PSD
        state_eigs = np.linalg.eigvalsh(rho_ppt)
        if np.min(state_eigs) < -1e-8:
            # Project to PSD: clip and rebuild
            state_eigs = np.maximum(state_eigs, 0)
            V = np.linalg.eigh(rho_ppt)[1]
            rho_ppt = V @ np.diag(state_eigs) @ V.conj().T
        rho_ppt /= np.trace(rho_ppt).real
        
        # Verify PPT still holds
        pt2 = rho_ppt.reshape(d,d,d,d).transpose(0,3,2,1).reshape(n,n)
        pt_min = np.min(np.linalg.eigvalsh(pt2))
        if pt_min < -1e-8:
            continue
        
        # Check entanglement
        R = rho_ppt.reshape(d,d,d,d).transpose(0,2,1,3).reshape(n,n)
        realign = np.linalg.norm(R, 'nuc')
        
        if realign > 1.001:
            if best is None or realign > best[0]:
                best = (realign, pt_min, rho_ppt.copy())
    
    return best


def compute_kdw_general(rho, dk, ds, n_bases=100):
    """K_DW for general dk,ds with correct 4-subsystem trace."""
    d = dk * ds
    best_k = -999.0
    
    for trial in range(n_bases):
        if trial == 0:
            U = np.eye(dk, dtype=complex)
        else:
            H = np.random.randn(dk, dk) + 1j * np.random.randn(dk, dk)
            U, _ = np.linalg.qr(H)
        
        p_x = np.zeros(dk)
        S_B_x = np.zeros(dk)
        S_E_x = np.zeros(dk)
        
        for x in range(dk):
            dim_rest = ds * dk * ds
            rho_rest = np.zeros((dim_rest, dim_rest), dtype=complex)
            for kA1 in range(dk):
                for kA2 in range(dk):
                    c = U[kA1,x].conj() * U[kA2,x]
                    for sA1 in range(ds):
                        for kB1 in range(dk):
                            for sB1 in range(ds):
                                for sA2 in range(ds):
                                    for kB2 in range(dk):
                                        for sB2 in range(ds):
                                            i = kA1*(ds*dk*ds)+sA1*(dk*ds)+kB1*ds+sB1
                                            j = kA2*(ds*dk*ds)+sA2*(dk*ds)+kB2*ds+sB2
                                            ri = sA1*(dk*ds)+kB1*ds+sB1
                                            rj = sA2*(dk*ds)+kB2*ds+sB2
                                            rho_rest[ri,rj] += c * rho[i,j]
            
            p_x[x] = np.trace(rho_rest).real
            if p_x[x] > 1e-15:
                rho_rest /= p_x[x]
                rB = np.zeros((dk*ds,dk*ds), dtype=complex)
                for sA in range(ds):
                    for kB1 in range(dk):
                        for sB1 in range(ds):
                            for kB2 in range(dk):
                                for sB2 in range(ds):
                                    rB[kB1*ds+sB1,kB2*ds+sB2] += rho_rest[sA*(dk*ds)+kB1*ds+sB1,sA*(dk*ds)+kB2*ds+sB2]
                e = np.linalg.eigvalsh(rB); e = e[e>1e-15]
                S_B_x[x] = -np.sum(e*np.log2(e)) if len(e)>0 else 0
                
                rE = np.zeros((ds,ds), dtype=complex)
                for kB in range(dk):
                    for sB in range(ds):
                        for sA1 in range(ds):
                            for sA2 in range(ds):
                                rE[sA1,sA2] += rho_rest[sA1*(dk*ds)+kB*ds+sB,sA2*(dk*ds)+kB*ds+sB]
                e = np.linalg.eigvalsh(rE); e = e[e>1e-15]
                S_E_x[x] = -np.sum(e*np.log2(e)) if len(e)>0 else 0
        
        rBu = np.zeros((dk*ds,dk*ds), dtype=complex)
        for kA in range(dk):
            for sA in range(ds):
                for kB1 in range(dk):
                    for sB1 in range(ds):
                        for kB2 in range(dk):
                            for sB2 in range(ds):
                                rBu[kB1*ds+sB1,kB2*ds+sB2] += rho[kA*(ds*dk*ds)+sA*(dk*ds)+kB1*ds+sB1,kA*(ds*dk*ds)+sA*(dk*ds)+kB2*ds+sB2]
        e = np.linalg.eigvalsh(rBu); e = e[e>1e-15]
        S_B = -np.sum(e*np.log2(e)) if len(e)>0 else 0
        
        rEu = np.zeros((ds,ds), dtype=complex)
        for kA in range(dk):
            for kB in range(dk):
                for sB in range(ds):
                    for sA1 in range(ds):
                        for sA2 in range(ds):
                            rEu[sA1,sA2] += rho[kA*(ds*dk*ds)+sA1*(dk*ds)+kB*ds+sB,kA*(ds*dk*ds)+sA2*(dk*ds)+kB*ds+sB]
        e = np.linalg.eigvalsh(rEu); e = e[e>1e-15]
        S_E = -np.sum(e*np.log2(e)) if len(e)>0 else 0
        
        I_XB = S_B - sum(p_x[x]*S_B_x[x] for x in range(dk) if p_x[x]>1e-15)
        I_XE = S_E - sum(p_x[x]*S_E_x[x] for x in range(dk) if p_x[x]>1e-15)
        best_k = max(best_k, I_XB - I_XE)
    
    return best_k


if __name__ == '__main__':
    N_WORKERS = 30
    
    print("=" * 60)
    print(f"  SA FINDER — Phase 0.5")
    print(f"  {N_WORKERS} cores, projection search")
    print("=" * 60)
    
    results = {}
    
    for d in [3, 4, 5, 6, 8, 9, 10, 12]:
        print(f"\n  d={d}: searching...")
        n_trials_per_core = 1000 if d <= 6 else 500
        
        args = [(d, seed, n_trials_per_core) for seed in range(N_WORKERS)]
        
        t0 = time.time()
        with Pool(N_WORKERS) as pool:
            raw = pool.map(search_ppt_ent, args)
        elapsed = time.time() - t0
        
        hits = [r for r in raw if r is not None]
        total_trials = N_WORKERS * n_trials_per_core
        
        if hits:
            best = max(hits, key=lambda x: x[0])
            realign, pt_min, rho = best
            
            # Check all factorizations for K_DW
            best_kdw = -999.0
            best_dk_ds = None
            
            # All (dk, ds) factorizations of d
            for dk in range(2, d):
                if d % dk == 0:
                    ds = d // dk
                    if ds >= 2:
                        kdw = compute_kdw_general(rho, dk, ds, n_bases=50)
                        if kdw > best_kdw:
                            best_kdw = kdw
                            best_dk_ds = (dk, ds)
            
            results[d] = {
                'realign': realign, 'pt_min': pt_min,
                'kdw': best_kdw, 'dk_ds': best_dk_ds,
                'n_hits': len(hits),
            }
            
            sa_str = "SA!" if best_kdw > 0.001 else ""
            fac_str = f"dk={best_dk_ds[0]},ds={best_dk_ds[1]}" if best_dk_ds else "none"
            print(f"    {len(hits)}/{N_WORKERS} cores found PPT+ENT")
            print(f"    Best realign={realign:.4f} pt_min={pt_min:.2e}")
            print(f"    K_DW={best_kdw:.6f} ({fac_str}) {sa_str}")
            print(f"    {elapsed:.1f}s ({total_trials} trials)")
            
            np.savez(f'sa_data/sa_candidate_d{d}.npz', rho=rho)
        else:
            results[d] = None
            print(f"    No PPT+ENT found ({total_trials} trials, {elapsed:.1f}s)")
    
    print(f"\n{'='*60}")
    print(f"  SA FINDER RESULTS")
    print(f"{'='*60}")
    print(f"  {'d':>3} {'PPT+ENT':>8} {'realign':>8} {'K_DW':>8} {'fact':>10} {'SA':>4}")
    print(f"  {'-'*45}")
    for d in sorted(results.keys()):
        r = results[d]
        if r:
            sa = "YES" if r['kdw'] > 0.001 else "no"
            fac = f"{r['dk_ds'][0]}x{r['dk_ds'][1]}" if r['dk_ds'] else "-"
            print(f"  {d:3d} {r['n_hits']:8d} {r['realign']:8.4f} {r['kdw']:8.4f} {fac:>10} {sa:>4}")
        else:
            print(f"  {d:3d}        0        -        -          -    -")
