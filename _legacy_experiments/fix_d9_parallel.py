"""
fix_d9_parallel.py — Unstructured Cholesky for d=9 on 30 cores
Each core runs independent restarts, then we pick the global best.
"""
import numpy as np
import torch
import torch.nn as nn
import time
from multiprocessing import Pool

def von_neumann(rho):
    e = np.linalg.eigvalsh(rho)
    e = e[e > 1e-15]
    return -np.sum(e * np.log2(e)) if len(e) > 0 else 0.0

def kdw_stinespring(rho, dA, dB, n_bases=300):
    d = dA * dB
    eigvals, eigvecs = np.linalg.eigh(rho)
    mask = eigvals > 1e-14
    lam = eigvals[mask]
    phi = eigvecs[:, mask]
    r = len(lam)
    if r == 0:
        return 0.0
    sqrt_lam = np.sqrt(lam)
    phi_r = phi.reshape(dA, dB, r)
    S_E_unc = von_neumann(np.diag(lam))
    rho_B_unc = sum(rho[a*dB:(a+1)*dB, a*dB:(a+1)*dB] for a in range(dA))
    S_B_unc = von_neumann(rho_B_unc)
    best = -999.0
    for trial in range(n_bases):
        if trial == 0:
            U = np.eye(dA, dtype=complex)
        else:
            H = np.random.randn(dA, dA) + 1j * np.random.randn(dA, dA)
            U, _ = np.linalg.qr(H)
        beta = np.einsum('ax,abk->xkb', U.conj(), phi_r)
        p_x = (np.sum(np.abs(beta)**2, axis=2)) @ lam
        sum_pSB = sum_pSE = 0.0
        for x in range(dA):
            if p_x[x] < 1e-15:
                continue
            wb = sqrt_lam[:, None] * beta[x]
            w = np.sum(wb, axis=0)
            rho_B_x = np.outer(w, w.conj()) / p_x[x]
            sum_pSB += p_x[x] * von_neumann(rho_B_x)
            gram = beta[x].conj() @ beta[x].T
            rho_E_x = np.outer(sqrt_lam, sqrt_lam) * gram.T / p_x[x]
            sum_pSE += p_x[x] * von_neumann(rho_E_x)
        best = max(best, (S_B_unc - sum_pSB) - (S_E_unc - sum_pSE))
    return best


def worker(args):
    """Single core: run n_restarts of unstructured Cholesky for d=9."""
    core_id, n_restarts, steps, seed = args
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    dA, dB = 3, 3
    d = 9
    
    best_realign = 0.0
    best_rho = None
    best_pt = None
    
    for restart in range(n_restarts):
        # Fresh random init each restart
        L_re = torch.randn(d, d, dtype=torch.float64) * 0.1
        L_im = torch.randn(d, d, dtype=torch.float64) * 0.1
        L_re.requires_grad_(True)
        L_im.requires_grad_(True)
        
        optimizer = torch.optim.Adam([L_re, L_im], lr=0.003)
        
        for step in range(steps):
            L = torch.complex(L_re, L_im)
            rho = L @ L.conj().T
            rho = (rho + rho.conj().T) / 2
            rho = rho / (torch.real(torch.trace(rho)) + 1e-30)
            
            rho_r = rho.reshape(dA, dB, dA, dB)
            rho_pt = rho_r.permute(0, 3, 2, 1).reshape(d, d)
            pt_min = torch.linalg.eigvalsh(rho_pt).min()
            
            R = rho.reshape(dA, dB, dA, dB).permute(0, 2, 1, 3).reshape(dA*dA, dB*dB)
            realign = torch.linalg.norm(R, ord='nuc')
            
            ppt_penalty = torch.relu(-pt_min + 1e-12) * 50000
            loss = -realign + ppt_penalty
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([L_re, L_im], 5.0)
            optimizer.step()
            
            with torch.no_grad():
                pt_val = pt_min.item()
                re_val = realign.item()
                if pt_val >= 0 and re_val > best_realign:
                    best_realign = re_val
                    best_rho = rho.detach().numpy()
                    best_pt = pt_val
    
    return best_rho, best_realign, best_pt, core_id


if __name__ == '__main__':
    print("=" * 60)
    print("  Fix d=9: Unstructured Cholesky — 30 CORES")
    print("=" * 60)
    
    n_cores = 30
    restarts_per_core = 20  # 30×20 = 600 total restarts
    steps = 5000
    
    args = [(i, restarts_per_core, steps, 42 + i * 1000) for i in range(n_cores)]
    
    print(f"  Cores: {n_cores}")
    print(f"  Restarts per core: {restarts_per_core}")
    print(f"  Total restarts: {n_cores * restarts_per_core}")
    print(f"  Steps per restart: {steps}")
    print(f"  Running...\n")
    
    t0 = time.time()
    with Pool(n_cores) as pool:
        results = pool.map(worker, args)
    elapsed = time.time() - t0
    
    # Find global best
    global_best_re = 0.0
    global_best_rho = None
    global_best_pt = None
    global_best_core = -1
    
    for rho, re_val, pt_val, core_id in results:
        if rho is not None and re_val > global_best_re:
            global_best_re = re_val
            global_best_rho = rho
            global_best_pt = pt_val
            global_best_core = core_id
    
    print(f"  Search done in {elapsed:.0f}s")
    print(f"  Best from core {global_best_core}: realign={global_best_re:.6f} pt_min={global_best_pt:.2e}")
    
    # Validate with K_DW
    if global_best_rho is not None and global_best_re > 1.0:
        kdw = kdw_stinespring(global_best_rho, 3, 3, n_bases=300)
        
        eigs = np.linalg.eigvalsh(global_best_rho)
        rho_pt = global_best_rho.reshape(3,3,3,3).transpose(0,3,2,1).reshape(9,9)
        pt_min_final = np.linalg.eigvalsh(rho_pt).min()
        
        print(f"\n{'='*60}")
        print(f"  FINAL RESULT")
        print(f"{'='*60}")
        print(f"  Realignment: {global_best_re:.6f} (entangled=YES)")
        print(f"  PPT: min_eig(PT) = {pt_min_final:.2e}")
        print(f"  Physicality: min_eig = {eigs.min():.2e}, trace = {np.trace(global_best_rho).real:.10f}")
        print(f"  K_DW = {kdw:.6f} bits")
        
        if kdw > 0.001 and pt_min_final >= -1e-10:
            np.savez_compressed('sa_data/unstructured_3x3.npz', rho=global_best_rho,
                                kdw=kdw, realign=global_best_re, pt_min=pt_min_final)
            print(f"\n  ✅ d=9 SA FOUND! K_DW={kdw:.4f} bits")
            print(f"  💾 Saved: sa_data/unstructured_3x3.npz")
        else:
            print(f"\n  ⚠️ PPT entangled but K_DW={kdw:.6f} (borderline)")
    else:
        print(f"\n  ❌ No PPT entangled state found")
    
    print("=" * 60)
