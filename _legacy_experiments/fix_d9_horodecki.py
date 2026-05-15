"""
fix_d9_horodecki.py — Fix d=9 via two approaches:
1. Horodecki 3⊗3 PPT entangled family (from literature)
2. Unstructured Cholesky optimization (general L, no Kronecker constraint)
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

print("=" * 60)
print("  Fix d=9: Horodecki + Unstructured Cholesky")
print("=" * 60)

# ══════════════════════════════════════════════════════════
# Approach 1: Horodecki 3⊗3 PPT entangled states
# From Horodecki 1997 (PRL 80, 5239)
# ══════════════════════════════════════════════════════════
print("\n  ── Approach 1: Horodecki 3⊗3 Family ──")

best_kdw_h = 0
best_rho_h = None
best_a_h = None

for a in np.linspace(0.01, 0.99, 200):
    rho = np.zeros((9, 9))
    norm = 8*a + 1
    
    # Horodecki family (real, PPT entangled for a ∈ (0,1))
    rho[0,0] = a;    rho[0,4] = a;    rho[0,8] = a
    rho[1,1] = a
    rho[2,2] = a
    rho[3,3] = a
    rho[4,0] = a;    rho[4,4] = a;    rho[4,8] = a
    rho[5,5] = a
    rho[6,6] = (1+a)/2;  rho[6,8] = np.sqrt(1-a**2)/2
    rho[7,7] = a
    rho[8,0] = a;    rho[8,4] = a;    rho[8,6] = np.sqrt(1-a**2)/2;  rho[8,8] = (1+a)/2
    
    rho /= norm
    
    # PPT check (3⊗3 split)
    dA, dB = 3, 3
    rho_r = rho.reshape(dA, dB, dA, dB)
    rho_pt = rho_r.transpose(0, 3, 2, 1).reshape(9, 9)
    pt_min = np.linalg.eigvalsh(rho_pt).min()
    
    if pt_min < -1e-10:
        continue
    
    # Quick K_DW
    kdw = kdw_stinespring(rho, dA, dB, n_bases=50)
    
    if kdw > best_kdw_h:
        best_kdw_h = kdw
        best_rho_h = rho.copy()
        best_a_h = a

if best_rho_h is not None:
    # Refine with 200 bases
    kdw_refined = kdw_stinespring(best_rho_h, 3, 3, n_bases=200)
    print(f"  Best Horodecki: a={best_a_h:.4f}, K_DW={kdw_refined:.6f}")
    
    rho_r = best_rho_h.reshape(3, 3, 3, 3)
    rho_pt = rho_r.transpose(0, 3, 2, 1).reshape(9, 9)
    pt_min = np.linalg.eigvalsh(rho_pt).min()
    print(f"  PPT: min_eig(PT)={pt_min:.2e}")
else:
    print("  No Horodecki state found with K_DW > 0")
    kdw_refined = 0

# ══════════════════════════════════════════════════════════
# Approach 2: Unstructured Cholesky (general L, float64)
# No Kronecker constraint — full 9×9 density matrix search
# ══════════════════════════════════════════════════════════
print("\n  ── Approach 2: Unstructured Cholesky (float64) ──")

class UnstructuredPPT(nn.Module):
    """General density matrix via Cholesky: rho = LL†/Tr(LL†)."""
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

def optimize_unstructured(dA, dB, n_restarts=200, steps=5000, lr=0.003):
    d = dA * dB
    best_realign = 0.0
    best_rho = None
    best_pt = None
    
    for restart in range(n_restarts):
        model = UnstructuredPPT(d, rank=d)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        local_best_re = 0.0
        local_best_rho = None
        local_best_pt = None
        
        for step in range(steps):
            rho = model()
            
            # Partial transpose
            rho_r = rho.reshape(dA, dB, dA, dB)
            rho_pt = rho_r.permute(0, 3, 2, 1).reshape(d, d)
            pt_eigs = torch.linalg.eigvalsh(rho_pt)
            pt_min = pt_eigs.min()
            
            # Realignment
            R = rho.reshape(dA, dB, dA, dB).permute(0, 2, 1, 3).reshape(dA*dA, dB*dB)
            realign = torch.linalg.norm(R, ord='nuc')
            
            # Objective: maximize realignment, strictly enforce PPT
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
        
        if (restart + 1) % 20 == 0:
            pt_str = f"{best_pt:.2e}" if best_pt is not None else "N/A"
            print(f"  restart {restart+1}/{n_restarts}: best_realign={best_realign:.6f} pt_min={pt_str}")
    
    return best_rho, best_realign, best_pt

t0 = time.time()
rho9_u, realign9_u, pt9_u = optimize_unstructured(3, 3, n_restarts=200, steps=5000)
elapsed = time.time() - t0

if rho9_u is not None:
    kdw9_u = kdw_stinespring(rho9_u, 3, 3, n_bases=200)
    print(f"\n  Unstructured: realign={realign9_u:.6f} pt_min={pt9_u:.2e} K_DW={kdw9_u:.6f} [{elapsed:.0f}s]")
else:
    kdw9_u = 0
    print(f"\n  Unstructured: no PPT entangled state found [{elapsed:.0f}s]")

# ══════════════════════════════════════════════════════════
# Pick the best result
# ══════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  RESULTS COMPARISON")
print(f"{'='*60}")
print(f"  Horodecki:     K_DW = {kdw_refined:.6f}")
print(f"  Unstructured:  K_DW = {kdw9_u:.6f}")

# Save the best
best_kdw = max(kdw_refined, kdw9_u)
if best_kdw > 0.001:
    if kdw_refined >= kdw9_u and best_rho_h is not None:
        np.savez_compressed('sa_data/horodecki_3x3.npz', rho=best_rho_h,
                            kdw=kdw_refined, a=best_a_h)
        print(f"\n  ✅ d=9 SA via Horodecki! K_DW={kdw_refined:.4f}")
    elif rho9_u is not None:
        np.savez_compressed('sa_data/unstructured_3x3.npz', rho=rho9_u,
                            kdw=kdw9_u, realign=realign9_u, pt_min=pt9_u)
        print(f"\n  ✅ d=9 SA via unstructured! K_DW={kdw9_u:.4f}")
else:
    print(f"\n  ⚠️ d=9: Neither approach found SA with K_DW > 0.001")
    print(f"  This may be a fundamental limitation of 3⊗3 systems")

print("=" * 60)
