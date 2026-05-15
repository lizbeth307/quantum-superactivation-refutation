"""
fix_d9_unstructured.py — Fix d=9 via unstructured Cholesky (general L, float64)
No Kronecker constraint — searches the FULL 9×9 density matrix space
"""
import numpy as np
import torch
import torch.nn as nn
import time

def von_neumann(rho):
    e = np.linalg.eigvalsh(rho)
    e = e[e > 1e-15]
    return -np.sum(e * np.log2(e)) if len(e) > 0 else 0.0

def kdw_stinespring(rho, dA, dB, n_bases=200):
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


class UnstructuredPPT(nn.Module):
    def __init__(self, d, rank=None):
        super().__init__()
        self.d = d
        r = rank or d
        self.L_re = nn.Parameter(torch.randn(d, r, dtype=torch.float64) * 0.1)
        self.L_im = nn.Parameter(torch.randn(d, r, dtype=torch.float64) * 0.1)
    
    def forward(self):
        L = torch.complex(self.L_re, self.L_im)
        rho = L @ L.conj().T
        rho = (rho + rho.conj().T) / 2
        rho = rho / (torch.real(torch.trace(rho)) + 1e-30)
        return rho


if __name__ == '__main__':
    print("=" * 60)
    print("  Fix d=9: Unstructured Cholesky (general L, float64)")
    print("=" * 60)
    
    dA, dB = 3, 3
    d = 9
    n_restarts = 300
    steps = 5000
    
    best_realign = 0.0
    best_rho = None
    best_pt = None
    
    t0 = time.time()
    
    for restart in range(n_restarts):
        model = UnstructuredPPT(d, rank=d)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
        
        local_best_re = 0.0
        local_best_rho = None
        local_best_pt = None
        
        for step in range(steps):
            rho = model()
            
            rho_r = rho.reshape(dA, dB, dA, dB)
            rho_pt = rho_r.permute(0, 3, 2, 1).reshape(d, d)
            pt_eigs = torch.linalg.eigvalsh(rho_pt)
            pt_min = pt_eigs.min()
            
            R = rho.reshape(dA, dB, dA, dB).permute(0, 2, 1, 3).reshape(dA*dA, dB*dB)
            realign = torch.linalg.norm(R, ord='nuc')
            
            ppt_penalty = torch.relu(-pt_min + 1e-12) * 50000
            loss = -realign + ppt_penalty
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
            with torch.no_grad():
                pt_val = pt_min.item()
                re_val = realign.item()
                if pt_val >= 0 and re_val > local_best_re:
                    local_best_re = re_val
                    local_best_rho = rho.detach().numpy()
                    local_best_pt = pt_val
        
        if local_best_rho is not None and local_best_re > best_realign:
            best_realign = local_best_re
            best_rho = local_best_rho
            best_pt = local_best_pt
        
        if (restart + 1) % 25 == 0:
            elapsed = time.time() - t0
            pt_str = f"{best_pt:.2e}" if best_pt is not None else "N/A"
            print(f"  restart {restart+1}/{n_restarts}: best_realign={best_realign:.6f} pt_min={pt_str} [{elapsed:.0f}s]")
    
    elapsed = time.time() - t0
    
    # Validate
    if best_rho is not None:
        kdw = kdw_stinespring(best_rho, dA, dB, n_bases=300)
        
        eigs = np.linalg.eigvalsh(best_rho)
        rho_pt = best_rho.reshape(3,3,3,3).transpose(0,3,2,1).reshape(9,9)
        pt_min_final = np.linalg.eigvalsh(rho_pt).min()
        
        print(f"\n{'='*60}")
        print(f"  FINAL RESULT")
        print(f"{'='*60}")
        print(f"  Realignment: {best_realign:.6f} (entangled={'YES' if best_realign > 1 else 'NO'})")
        print(f"  PPT: min_eig(PT) = {pt_min_final:.2e}")
        print(f"  Physicality: min_eig = {eigs.min():.2e}, trace = {np.trace(best_rho).real:.10f}")
        print(f"  K_DW = {kdw:.6f} bits")
        print(f"  Time: {elapsed:.0f}s")
        
        if kdw > 0.001 and pt_min_final >= -1e-10:
            np.savez_compressed('sa_data/unstructured_3x3.npz', rho=best_rho,
                                kdw=kdw, realign=best_realign, pt_min=pt_min_final)
            print(f"\n  ✅ d=9 SA FOUND! K_DW={kdw:.4f} bits — saved to sa_data/unstructured_3x3.npz")
        else:
            print(f"\n  ⚠️ d=9: realign>1 but K_DW={kdw:.6f}")
    else:
        print(f"\n  ❌ No PPT entangled state found for d=9 [{elapsed:.0f}s]")
    
    print("=" * 60)
