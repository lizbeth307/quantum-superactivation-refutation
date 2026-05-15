"""
paths_bce.py — Transition Paths B, C, E for d=4 flower.
Phase 0.1 continued. 30 CPU cores.

Path B: Noise sweep (eps=0 -> 0.5), 5000 pts
  Mix flower with maximally mixed: rho(eps) = (1-eps)*flower + eps*I/d^2
  
Path C: PPT boundary approach, 5000 pts
  Vary p1 around optimal: p1 = p1_opt * (1 + delta), delta in [-0.3, 0.3]
  
Path E: Shield quality (U: Hadamard -> Identity), 5000 pts
  U(t) = expm(t * logm(H)), t in [0, 1]
"""
import numpy as np
from multiprocessing import Pool
from scipy.linalg import expm, logm
import time
import os

os.makedirs('sa_data', exist_ok=True)


# ── Shared functions ──

def ket2(a, b):
    v = np.zeros(4, dtype=complex)
    v[a*2+b] = 1.0
    return v

BELLS = [
    (ket2(0,0) + ket2(1,1)) / np.sqrt(2),
    (ket2(0,0) - ket2(1,1)) / np.sqrt(2),
    (ket2(0,1) + ket2(1,0)) / np.sqrt(2),
    (ket2(0,1) - ket2(1,0)) / np.sqrt(2),
]

PSI_PLUS_SH = (ket2(0,1) + ket2(1,0)) / np.sqrt(2)
PSI_MINUS_SH = (ket2(0,1) - ket2(1,0)) / np.sqrt(2)

# Optimal flower params
ALPHA_OPT = np.pi / 8
P1_OPT = np.sqrt(2) / (1 + np.sqrt(2))


def build_flower_full(alpha, p1):
    """Build flower with explicit alpha and p1."""
    a_p = np.cos(alpha)
    b_p = np.sin(alpha)
    chi_plus = a_p * ket2(0,0) + b_p * ket2(1,1)
    chi_minus = b_p * ket2(0,0) - a_p * ket2(1,1)
    
    shields = [
        0.5 * (np.outer(ket2(0,0), ket2(0,0)) + np.outer(PSI_PLUS_SH, PSI_PLUS_SH)),
        0.5 * (np.outer(ket2(1,1), ket2(1,1)) + np.outer(PSI_MINUS_SH, PSI_MINUS_SH)),
        np.outer(chi_plus, chi_plus.conj()),
        np.outer(chi_minus, chi_minus.conj()),
    ]
    
    p2 = 1 - p1
    weights = [p1/2, p1/2, p2/2, p2/2]
    
    rho = np.zeros((16,16), dtype=complex)
    for i in range(4):
        psi = BELLS[i]; rho_sh = shields[i]; q = weights[i]
        for kA1 in range(2):
            for kB1 in range(2):
                for kA2 in range(2):
                    for kB2 in range(2):
                        kv = psi[kA1*2+kB1] * psi[kA2*2+kB2].conj()
                        for sA1 in range(2):
                            for sB1 in range(2):
                                for sA2 in range(2):
                                    for sB2 in range(2):
                                        sv = rho_sh[sA1*2+sB1, sA2*2+sB2]
                                        row = kA1*8 + sA1*4 + kB1*2 + sB1
                                        col = kA2*8 + sA2*4 + kB2*2 + sB2
                                        rho[row, col] += q * kv * sv
    return (rho + rho.conj().T) / 2


def pt4(rho):
    return rho.reshape(4,4,4,4).transpose(0,3,2,1).reshape(16,16)


def von_neumann(rho):
    eigs = np.linalg.eigvalsh(rho)
    eigs = eigs[eigs > 1e-15]
    return -np.sum(eigs * np.log2(eigs)) if len(eigs) > 0 else 0.0


def compute_kdw_fast(rho, n_bases=20):
    """K_DW for dk=2, ds=2 with correct 4-subsystem trace."""
    dk, ds = 2, 2
    best_k = -999.0
    
    for trial in range(n_bases):
        if trial == 0:
            U = np.eye(2, dtype=complex)
        elif trial == 1:
            U = np.array([[1,1],[1,-1]], dtype=complex) / np.sqrt(2)
        else:
            H = np.random.randn(2,2) + 1j*np.random.randn(2,2)
            U, _ = np.linalg.qr(H)
        
        p_x = np.zeros(dk)
        S_B_x = np.zeros(dk)
        S_E_x = np.zeros(dk)
        
        for x in range(dk):
            dim_rest = ds*dk*ds  # = 8 for dk=2,ds=2
            rho_rest = np.zeros((dim_rest, dim_rest), dtype=complex)
            for kA1 in range(2):
                for kA2 in range(2):
                    c = U[kA1,x].conj() * U[kA2,x]
                    for sA1 in range(2):
                        for kB1 in range(2):
                            for sB1 in range(2):
                                for sA2 in range(2):
                                    for kB2 in range(2):
                                        for sB2 in range(2):
                                            i = kA1*8+sA1*4+kB1*2+sB1
                                            j = kA2*8+sA2*4+kB2*2+sB2
                                            ri = sA1*(dk*ds)+kB1*ds+sB1
                                            rj = sA2*(dk*ds)+kB2*ds+sB2
                                            rho_rest[ri,rj] += c * rho[i,j]
            
            p_x[x] = np.trace(rho_rest).real
            if p_x[x] > 1e-15:
                rho_rest /= p_x[x]
                rB = np.zeros((dk*ds, dk*ds), dtype=complex)
                for sA in range(2):
                    for kB1 in range(2):
                        for sB1 in range(2):
                            for kB2 in range(2):
                                for sB2 in range(2):
                                    rB[kB1*2+sB1, kB2*2+sB2] += rho_rest[sA*(dk*ds)+kB1*ds+sB1, sA*(dk*ds)+kB2*ds+sB2]
                S_B_x[x] = von_neumann(rB)
                rE = np.zeros((2,2), dtype=complex)
                for kB in range(2):
                    for sB in range(2):
                        for sA1 in range(2):
                            for sA2 in range(2):
                                rE[sA1,sA2] += rho_rest[sA1*(dk*ds)+kB*ds+sB, sA2*(dk*ds)+kB*ds+sB]
                S_E_x[x] = von_neumann(rE)
        
        rBu = np.zeros((4,4), dtype=complex)
        for kA in range(2):
            for sA in range(2):
                for kB1 in range(2):
                    for sB1 in range(2):
                        for kB2 in range(2):
                            for sB2 in range(2):
                                rBu[kB1*2+sB1,kB2*2+sB2] += rho[kA*8+sA*4+kB1*2+sB1, kA*8+sA*4+kB2*2+sB2]
        S_B = von_neumann(rBu)
        
        rEu = np.zeros((2,2), dtype=complex)
        for kA in range(2):
            for kB in range(2):
                for sB in range(2):
                    for sA1 in range(2):
                        for sA2 in range(2):
                            rEu[sA1,sA2] += rho[kA*8+sA1*4+kB*2+sB, kA*8+sA2*4+kB*2+sB]
        S_E = von_neumann(rEu)
        
        I_XB = S_B - sum(p_x[x]*S_B_x[x] for x in range(2) if p_x[x]>1e-15)
        I_XE = S_E - sum(p_x[x]*S_E_x[x] for x in range(2) if p_x[x]>1e-15)
        best_k = max(best_k, I_XB - I_XE)
    
    return best_k


def extract_features(rho, extra=None):
    d = 4; n = 16
    eigs = np.sort(np.linalg.eigvalsh(rho))
    pt = pt4(rho)
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
        'mutual_info_norm': (S_A + S_B - S_AB) / (2*np.log2(d)),
        'realign_norm': np.linalg.norm(R, 'nuc'),
        'A_max_mixed_dist': np.linalg.norm(rho_A - np.eye(d)/d),
        'B_max_mixed_dist': np.linalg.norm(rho_B - np.eye(d)/d),
        'purity_norm': np.trace(rho@rho).real * d,
        'rank_norm': np.sum(eigs>1e-10) / n,
        'd': 4, 'dk': 2, 'ds': 2,
    }
    if extra:
        f.update(extra)
    return f


# ── Path B: Noise ──

def path_b_worker(args):
    eps, idx = args
    flower = build_flower_full(ALPHA_OPT, P1_OPT)
    rho = (1 - eps) * flower + eps * np.eye(16) / 16
    kdw = compute_kdw_fast(rho, n_bases=20)
    return extract_features(rho, {'eps': eps, 'kdw': kdw, 'idx': idx, 'path': 'B'})


# ── Path C: PPT boundary ──

def path_c_worker(args):
    delta, idx = args
    p1 = P1_OPT * (1 + delta)
    p1 = np.clip(p1, 0.01, 0.99)
    rho = build_flower_full(ALPHA_OPT, p1)
    kdw = compute_kdw_fast(rho, n_bases=20)
    return extract_features(rho, {'delta': delta, 'p1': p1, 'kdw': kdw, 'idx': idx, 'path': 'C'})


# ── Path E: Shield quality ──

def path_e_worker(args):
    t, idx = args
    # U(t) interpolation: Identity -> Hadamard
    H = np.array([[1,1],[1,-1]], dtype=complex) / np.sqrt(2)
    if t <= 0.001:
        U = np.eye(2, dtype=complex)
    elif t >= 0.999:
        U = H
    else:
        U = expm(t * logm(H))
    
    # Build flower with modified chi states using U
    # chi_plus = U[0,0]*|00> + U[1,0]*|11>
    # chi_minus = U[0,1]*|00> + U[1,1]*|11>
    chi_plus = U[0,0] * ket2(0,0) + U[1,0] * ket2(1,1)
    chi_minus = U[0,1] * ket2(0,0) + U[1,1] * ket2(1,1)
    
    shields = [
        0.5 * (np.outer(ket2(0,0), ket2(0,0)) + np.outer(PSI_PLUS_SH, PSI_PLUS_SH)),
        0.5 * (np.outer(ket2(1,1), ket2(1,1)) + np.outer(PSI_MINUS_SH, PSI_MINUS_SH)),
        np.outer(chi_plus, chi_plus.conj()),
        np.outer(chi_minus, chi_minus.conj()),
    ]
    
    # Find optimal p1 for this shield via golden section
    def build_rho_p1(p1v):
        w = [p1v/2, p1v/2, (1-p1v)/2, (1-p1v)/2]
        r = np.zeros((16,16), dtype=complex)
        for i in range(4):
            psi = BELLS[i]; rho_sh = shields[i]; q = w[i]
            for kA1 in range(2):
                for kB1 in range(2):
                    for kA2 in range(2):
                        for kB2 in range(2):
                            kv = psi[kA1*2+kB1] * psi[kA2*2+kB2].conj()
                            for sA1 in range(2):
                                for sB1 in range(2):
                                    for sA2 in range(2):
                                        for sB2 in range(2):
                                            sv = rho_sh[sA1*2+sB1, sA2*2+sB2]
                                            row = kA1*8+sA1*4+kB1*2+sB1
                                            col = kA2*8+sA2*4+kB2*2+sB2
                                            r[row,col] += q * kv * sv
        return (r + r.conj().T) / 2
    
    def pt_min_of(p1v):
        return np.min(np.linalg.eigvalsh(pt4(build_rho_p1(p1v))))
    
    gr = (np.sqrt(5)+1)/2
    a, b = 0.01, 0.99
    for _ in range(80):
        c = b-(b-a)/gr; d2 = a+(b-a)/gr
        if pt_min_of(c) < pt_min_of(d2): a=c
        else: b=d2
    p1 = (a+b)/2
    
    rho = build_rho_p1(p1)
    kdw = compute_kdw_fast(rho, n_bases=20)
    return extract_features(rho, {'t': t, 'p1_opt': p1, 'kdw': kdw, 'idx': idx, 'path': 'E'})


# ── Main ──

if __name__ == '__main__':
    N_B = 5000
    N_C = 5000
    N_E = 5000
    N_WORKERS = 30
    
    print("=" * 60)
    print("  PATHS B, C, E (d=4 flower)")
    print(f"  Path B: {N_B} pts (noise eps=0->0.5)")
    print(f"  Path C: {N_C} pts (PPT boundary delta=-0.3->0.3)")
    print(f"  Path E: {N_E} pts (shield quality t=0->1)")
    print(f"  Workers: {N_WORKERS}")
    print("=" * 60)
    
    for path_name, worker_fn, args_gen, n_pts, filename in [
        ("B (noise)", path_b_worker,
         lambda n: [(eps, i) for i, eps in enumerate(np.linspace(0, 0.5, n))],
         N_B, "path_b_d4.npz"),
        ("C (PPT boundary)", path_c_worker,
         lambda n: [(d, i) for i, d in enumerate(np.linspace(-0.3, 0.3, n))],
         N_C, "path_c_d4.npz"),
        ("E (shield quality)", path_e_worker,
         lambda n: [(t, i) for i, t in enumerate(np.linspace(0.001, 0.999, n))],
         N_E, "path_e_d4.npz"),
    ]:
        print(f"\n  --- Path {path_name} ---")
        args = args_gen(n_pts)
        
        t0 = time.time()
        with Pool(N_WORKERS) as pool:
            results = pool.map(worker_fn, args)
        elapsed = time.time() - t0
        
        keys = sorted(results[0].keys())
        data = {k: np.array([r[k] for r in results]) for k in keys}
        np.savez(f'sa_data/{filename}', **data)
        
        ppt = np.sum(data['is_ppt'])
        kdw_pos = np.sum(data['kdw'] > 0.001)
        kdw_max = np.max(data['kdw'])
        
        print(f"  {n_pts} pts in {elapsed:.1f}s ({n_pts/elapsed:.0f} pts/s)")
        print(f"  PPT: {ppt}, K_DW>0: {kdw_pos}, K_DW max: {kdw_max:.4f}")
        print(f"  Saved: sa_data/{filename}")
    
    print(f"\n{'='*60}")
    print("  ALL PATHS COMPLETE")
    print(f"{'='*60}")
