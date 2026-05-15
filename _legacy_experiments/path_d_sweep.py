"""
path_d_sweep.py — Phase 0.1 Path D: Dimension sweep.
Generate training data for d=6 by perturbing the validated SA candidate.
Also try UPB construction for d=8,9.
30 CPU cores.
"""
import numpy as np
from multiprocessing import Pool
import time
import os

os.makedirs('sa_data', exist_ok=True)

# ── Shared ──

def pt_d(rho, d):
    n = d*d
    return rho.reshape(d,d,d,d).transpose(0,3,2,1).reshape(n,n)

def von_neumann(rho):
    e = np.linalg.eigvalsh(rho)
    e = e[e > 1e-15]
    return -np.sum(e * np.log2(e)) if len(e) > 0 else 0.0

def realignment_norm(rho, d):
    n = d*d
    R = rho.reshape(d,d,d,d).transpose(0,2,1,3).reshape(n,n)
    return np.linalg.norm(R, 'nuc')

def extract_features_general(rho, d, extra=None):
    n = d*d
    eigs = np.sort(np.linalg.eigvalsh(rho))
    pt = pt_d(rho, d)
    pt_eigs = np.sort(np.linalg.eigvalsh(pt))
    rho_A = np.trace(rho.reshape(d,d,d,d), axis1=1, axis2=3)
    rho_B = np.trace(rho.reshape(d,d,d,d), axis1=0, axis2=2)
    S_A = von_neumann(rho_A); S_B = von_neumann(rho_B)
    S_AB = von_neumann(rho)
    R = rho.reshape(d,d,d,d).transpose(0,2,1,3).reshape(n,n)
    f = {
        'rank': np.sum(eigs>1e-10), 'purity': np.trace(rho@rho).real,
        'eig_min': eigs[0], 'eig_max': eigs[-1], 'eig_std': np.std(eigs),
        'pt_min': pt_eigs[0], 'is_ppt': int(pt_eigs[0] >= -1e-5),
        'pt_boundary_dist': abs(pt_eigs[0]),
        'pt_neg_count': int(np.sum(pt_eigs < -1e-10)),
        'S_A': S_A, 'S_B': S_B, 'S_AB': S_AB,
        'mutual_info': S_A + S_B - S_AB,
        'mutual_info_norm': (S_A+S_B-S_AB) / (2*np.log2(d)) if d>1 else 0,
        'realign_norm': np.linalg.norm(R, 'nuc'),
        'A_max_mixed_dist': np.linalg.norm(rho_A - np.eye(d)/d),
        'B_max_mixed_dist': np.linalg.norm(rho_B - np.eye(d)/d),
        'purity_norm': np.trace(rho@rho).real * d,
        'rank_norm': np.sum(eigs>1e-10) / n,
        'd': d,
    }
    if extra:
        f.update(extra)
    return f

def kdw_general(rho, dk, ds, n_bases=30):
    """Fast K_DW for any dk,ds."""
    d = dk*ds; best = -999.0
    for trial in range(n_bases):
        if trial == 0: U = np.eye(dk, dtype=complex)
        else:
            H = np.random.randn(dk,dk)+1j*np.random.randn(dk,dk)
            U, _ = np.linalg.qr(H)
        p_x = np.zeros(dk); S_B_x = np.zeros(dk); S_E_x = np.zeros(dk)
        for x in range(dk):
            dim_r = ds*dk*ds
            rr = np.zeros((dim_r,dim_r), dtype=complex)
            for k1 in range(dk):
                for k2 in range(dk):
                    c = U[k1,x].conj()*U[k2,x]
                    for s1 in range(ds):
                        for b1 in range(dk):
                            for t1 in range(ds):
                                for s2 in range(ds):
                                    for b2 in range(dk):
                                        for t2 in range(ds):
                                            i=k1*(ds*dk*ds)+s1*(dk*ds)+b1*ds+t1
                                            j=k2*(ds*dk*ds)+s2*(dk*ds)+b2*ds+t2
                                            rr[s1*(dk*ds)+b1*ds+t1,s2*(dk*ds)+b2*ds+t2] += c*rho[i,j]
            p_x[x] = np.trace(rr).real
            if p_x[x]>1e-15:
                rr /= p_x[x]
                rB = np.zeros((dk*ds,dk*ds), dtype=complex)
                for s in range(ds):
                    for b1 in range(dk):
                        for t1 in range(ds):
                            for b2 in range(dk):
                                for t2 in range(ds):
                                    rB[b1*ds+t1,b2*ds+t2] += rr[s*(dk*ds)+b1*ds+t1,s*(dk*ds)+b2*ds+t2]
                S_B_x[x] = von_neumann(rB)
                rE = np.zeros((ds,ds), dtype=complex)
                for b in range(dk):
                    for t in range(ds):
                        for s1 in range(ds):
                            for s2 in range(ds):
                                rE[s1,s2] += rr[s1*(dk*ds)+b*ds+t,s2*(dk*ds)+b*ds+t]
                S_E_x[x] = von_neumann(rE)
        rBu = np.zeros((dk*ds,dk*ds), dtype=complex)
        for k in range(dk):
            for s in range(ds):
                for b1 in range(dk):
                    for t1 in range(ds):
                        for b2 in range(dk):
                            for t2 in range(ds):
                                rBu[b1*ds+t1,b2*ds+t2] += rho[k*(ds*dk*ds)+s*(dk*ds)+b1*ds+t1,k*(ds*dk*ds)+s*(dk*ds)+b2*ds+t2]
        S_B = von_neumann(rBu)
        rEu = np.zeros((ds,ds), dtype=complex)
        for k in range(dk):
            for b in range(dk):
                for t in range(ds):
                    for s1 in range(ds):
                        for s2 in range(ds):
                            rEu[s1,s2] += rho[k*(ds*dk*ds)+s1*(dk*ds)+b*ds+t,k*(ds*dk*ds)+s2*(dk*ds)+b*ds+t]
        S_E = von_neumann(rEu)
        I_XB = S_B - sum(p_x[x]*S_B_x[x] for x in range(dk) if p_x[x]>1e-15)
        I_XE = S_E - sum(p_x[x]*S_E_x[x] for x in range(dk) if p_x[x]>1e-15)
        best = max(best, I_XB - I_XE)
    return best


# ── Path D1: Noise perturbation of d=6 SA candidate ──

RHO_6 = np.load('sa_data/sa_candidate_d6.npz')['rho']

def path_d6_worker(args):
    eps, idx = args
    d = 6; n = 36
    rho = (1 - eps) * RHO_6 + eps * np.eye(n) / n
    k = kdw_general(rho, 3, 2, n_bases=20)
    return extract_features_general(rho, d, {'eps': eps, 'kdw': k, 'idx': idx, 'dk': 3, 'ds': 2, 'path': 'D6'})


# ── UPB for d=8 (2x4 system: Feng UPB) ──

def build_upb_state(dA, dB):
    """Build PPT entangled state from Tiles-like UPB for dA x dB."""
    d = dA * dB
    n = d * d
    
    # Generalized Tiles UPB for dA x dB
    # Product vectors that form UPB
    upb = []
    
    if dA == 3 and dB == 3:
        # Standard Tiles UPB (Bennett et al.)
        e0 = np.array([1,0,0], dtype=complex)
        e1 = np.array([0,1,0], dtype=complex)
        e2 = np.array([0,0,1], dtype=complex)
        
        upb.append((e0, (e0-e1)/np.sqrt(2)))
        upb.append(((e0-e1)/np.sqrt(2), e2))
        upb.append((e2, (e1-e2)/np.sqrt(2)))
        upb.append(((e1-e2)/np.sqrt(2), e0))
        upb.append(((e0+e1+e2)/np.sqrt(3), (e0+e1+e2)/np.sqrt(3)))
    
    elif dA == 2 and dB == 4:
        # Pyramid UPB for 2x4
        e0 = np.array([1,0], dtype=complex)
        e1 = np.array([0,1], dtype=complex)
        f0 = np.array([1,0,0,0], dtype=complex)
        f1 = np.array([0,1,0,0], dtype=complex)
        f2 = np.array([0,0,1,0], dtype=complex)
        f3 = np.array([0,0,0,1], dtype=complex)
        
        upb.append((e0, f0))
        upb.append((e1, f1))
        upb.append(((e0+e1)/np.sqrt(2), (f2+f3)/np.sqrt(2)))
        upb.append(((e0-e1)/np.sqrt(2), (f2-f3)/np.sqrt(2)))
        upb.append(((e0+1j*e1)/np.sqrt(2), (f0+f1+f2+f3)/2))
    
    elif dA == 4 and dB == 4:
        # GenTiles for 4x4
        e = [np.zeros(4, dtype=complex) for _ in range(4)]
        for i in range(4): e[i][i] = 1.0
        
        upb.append((e[0], (e[0]-e[1])/np.sqrt(2)))
        upb.append(((e[0]-e[1])/np.sqrt(2), e[2]))
        upb.append((e[2], (e[2]-e[3])/np.sqrt(2)))
        upb.append(((e[2]-e[3])/np.sqrt(2), e[0]))
        v = sum(e)/2
        upb.append((v, v))
    
    if not upb:
        return None
    
    # Build UPB projector
    P_upb = np.zeros((d, d), dtype=complex)
    for (a, b) in upb:
        psi = np.kron(a, b)
        P_upb += np.outer(psi, psi.conj())
    
    # PPT entangled state = (I - P_upb) / trace
    rho = np.eye(d) - P_upb
    eigs = np.linalg.eigvalsh(rho)
    if np.min(eigs) < -1e-10:
        # Project to PSD
        E, V = np.linalg.eigh(rho)
        E = np.maximum(E, 0)
        rho = V @ np.diag(E) @ V.conj().T
    
    tr = np.trace(rho).real
    if tr < 1e-10:
        return None
    rho /= tr
    
    # Check PPT
    # For dA x dB: PT over B
    rho_2d = rho.reshape(dA, dB, dA, dB)
    pt = rho_2d.transpose(0,3,2,1).reshape(d,d)
    pt_min = np.min(np.linalg.eigvalsh(pt))
    
    # Check entanglement
    R = rho.reshape(dA,dB,dA,dB).transpose(0,2,1,3).reshape(d,d)
    realign = np.linalg.norm(R, 'nuc')
    
    return rho, pt_min, realign


def search_upb_worker(args):
    dA, dB, idx = args
    result = build_upb_state(dA, dB)
    if result is None:
        return None
    rho, pt_min, realign = result
    d = dA * dB
    if pt_min >= -1e-8 and realign > 1.001:
        return {'dA': dA, 'dB': dB, 'd': d, 'pt_min': pt_min, 
                'realign': realign, 'rho': rho}
    return None


# ── Main ──

if __name__ == '__main__':
    N_WORKERS = 30
    
    print("=" * 60)
    print("  PATH D + UPB SEARCH")
    print("=" * 60)
    
    # Path D: d=6 noise perturbation
    print("\n  --- Path D: d=6 noise sweep ---")
    eps_vals = np.linspace(0, 0.5, 5000)
    args_d6 = [(eps, i) for i, eps in enumerate(eps_vals)]
    
    t0 = time.time()
    with Pool(N_WORKERS) as pool:
        results_d6 = pool.map(path_d6_worker, args_d6)
    elapsed = time.time() - t0
    
    keys = sorted(results_d6[0].keys())
    data = {k: np.array([r[k] for r in results_d6]) for k in keys}
    np.savez('sa_data/path_d_d6.npz', **data)
    
    ppt = np.sum(data['is_ppt'])
    kdw_pos = np.sum(data['kdw'] > 0.001)
    print(f"  5000 pts in {elapsed:.1f}s")
    print(f"  PPT: {ppt}, K_DW>0: {kdw_pos}, K_DW max: {np.max(data['kdw']):.4f}")
    
    # Transition: where K_DW crosses 0
    kdw_arr = data['kdw']
    eps_arr = data['eps']
    for i in range(1, len(kdw_arr)):
        if kdw_arr[i-1] > 0.001 and kdw_arr[i] <= 0.001:
            print(f"  K_DW transition at eps={eps_arr[i]:.4f}")
            break
    
    # UPB search
    print("\n  --- UPB Construction ---")
    for dA, dB in [(3,3), (2,4), (4,4)]:
        d = dA * dB
        result = build_upb_state(dA, dB)
        if result is not None:
            rho, pt_min, realign = result
            ppt = pt_min >= -1e-8
            ent = realign > 1.001
            print(f"  {dA}x{dB} (d={d}): PT_min={pt_min:.2e} realign={realign:.4f} PPT={ppt} ENT={ent}")
            if ppt and ent:
                np.savez(f'sa_data/upb_{dA}x{dB}.npz', rho=rho)
                # Compute K_DW
                for dk in range(2, d):
                    if d % dk == 0:
                        ds = d // dk
                        if ds >= 2:
                            k = kdw_general(rho, dk, ds, 50)
                            if k > 0.001:
                                print(f"    K_DW(dk={dk},ds={ds}) = {k:.6f} SA!")
        else:
            print(f"  {dA}x{dB} (d={d}): construction failed")
    
    print(f"\n{'='*60}")
    print("  COMPLETE")
    print(f"{'='*60}")
