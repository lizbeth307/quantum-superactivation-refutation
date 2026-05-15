"""
fix_phase0q_states.py — Fix d=6 and d=9 Phase 0Q states
d=6: project rho onto PSD cone (clip negative eigs to 0)
d=9: re-optimize with tighter PPT penalty on CPU (float64)
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
print("  Fixing Phase 0Q states: d=6 and d=9")
print("=" * 60)

# ═══ Fix d=6: Project onto PSD cone ═══
print("\n  ── Fix d=6 (phase0q_2x3.npz) ──")
data6 = np.load('sa_data/phase0q_2x3.npz')
rho6 = data6['rho']
print(f"  Before: shape={rho6.shape}, min_eig={np.linalg.eigvalsh(rho6).min():.2e}")

# Project: clip negative eigenvalues to 0, renormalize
eigvals, eigvecs = np.linalg.eigh(rho6)
eigvals = np.maximum(eigvals, 0)  # Clip negatives
rho6_fixed = eigvecs @ np.diag(eigvals) @ eigvecs.conj().T
rho6_fixed = (rho6_fixed + rho6_fixed.conj().T) / 2
rho6_fixed /= np.trace(rho6_fixed).real

print(f"  After:  min_eig={np.linalg.eigvalsh(rho6_fixed).min():.2e}, trace={np.trace(rho6_fixed).real:.10f}")

# Check PPT
d6 = 6
dA6, dB6 = 6, 6  # Phase 0Q: d²×d² format
rho6_r = rho6_fixed.reshape(dA6, dB6, dA6, dB6)
rho6_pt = rho6_r.transpose(0, 3, 2, 1).reshape(36, 36)
pt_min6 = np.linalg.eigvalsh(rho6_pt).min()
print(f"  PPT: min_eig(PT)={pt_min6:.2e}, is_PPT={pt_min6 >= -1e-10}")

# K_DW
kdw6 = kdw_stinespring(rho6_fixed, dA6, dB6, n_bases=200)
print(f"  K_DW = {kdw6:.6f} bits")

if pt_min6 >= -1e-10 and kdw6 > 0.001:
    np.savez_compressed('sa_data/phase0q_2x3.npz', rho=rho6_fixed,
                        kdw=kdw6, realign=data6['realign'], pt_min=pt_min6)
    print(f"  ✅ d=6 FIXED and saved! K_DW={kdw6:.4f}")
else:
    print(f"  ❌ d=6 still fails after PSD projection")


# ═══ Fix d=9: Re-optimize with float64 + tighter PPT ═══
print("\n  ── Fix d=9: Re-optimize with float64 + tight PPT ──")

class StructuredPPT64(nn.Module):
    """float64 version with tighter PPT enforcement."""
    def __init__(self, dk, ds, n_terms=4):
        super().__init__()
        self.dk = dk
        self.ds = ds
        self.d = dk * ds
        self.n_terms = n_terms
        rk = dk * dk
        rs = ds * ds
        self.L_key_re = nn.ParameterList([
            nn.Parameter(torch.randn(rk, rk, dtype=torch.float64) * 0.1) for _ in range(n_terms)
        ])
        self.L_key_im = nn.ParameterList([
            nn.Parameter(torch.randn(rk, rk, dtype=torch.float64) * 0.1) for _ in range(n_terms)
        ])
        self.L_shield_re = nn.ParameterList([
            nn.Parameter(torch.randn(rs, rs, dtype=torch.float64) * 0.1) for _ in range(n_terms)
        ])
        self.L_shield_im = nn.ParameterList([
            nn.Parameter(torch.randn(rs, rs, dtype=torch.float64) * 0.1) for _ in range(n_terms)
        ])
    
    def forward(self):
        L_terms = []
        for i in range(self.n_terms):
            Lk = torch.complex(self.L_key_re[i], self.L_key_im[i])
            Ls = torch.complex(self.L_shield_re[i], self.L_shield_im[i])
            Li = torch.kron(Lk, Ls)
            L_terms.append(Li)
        L = sum(L_terms)
        rho = L @ L.conj().T
        rho = (rho + rho.conj().T) / 2
        trace = torch.real(torch.trace(rho))
        rho = rho / (trace + 1e-30)
        return rho
    
    def partial_transpose(self, rho):
        d = self.d
        rho_r = rho.reshape(d, d, d, d)
        rho_pt = rho_r.permute(0, 3, 2, 1).reshape(d * d, d * d)
        return rho_pt
    
    def pt_min_eigenvalue(self, rho):
        rho_pt = self.partial_transpose(rho)
        eigs = torch.linalg.eigvalsh(rho_pt)
        return eigs.min()
    
    def realignment_norm(self, rho):
        d = self.d
        R = rho.reshape(d, d, d, d).permute(0, 2, 1, 3).reshape(d * d, d * d)
        return torch.linalg.norm(R, ord='nuc')

def optimize_d9(n_restarts=100, steps=3000):
    """Re-optimize d=9 with float64 and strict PPT."""
    dk, ds = 3, 3
    best_realign = 0.0
    best_rho = None
    best_pt = None
    
    for restart in range(n_restarts):
        model = StructuredPPT64(dk, ds, n_terms=6)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        
        local_best_realign = 0.0
        local_best_rho = None
        local_best_pt = None
        
        for step in range(steps):
            rho = model()
            pt_min = model.pt_min_eigenvalue(rho)
            realign = model.realignment_norm(rho)
            
            # STRICT PPT: must be >= 0 (not just >= -1e-7)
            ppt_penalty = torch.relu(-pt_min + 1e-10) * 10000  
            loss = -realign + ppt_penalty
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
            with torch.no_grad():
                pt_val = pt_min.item()
                re_val = realign.item()
                if pt_val >= 0 and re_val > local_best_realign:
                    local_best_realign = re_val
                    local_best_rho = rho.detach().numpy()
                    local_best_pt = pt_val
        
        if local_best_rho is not None and local_best_realign > best_realign:
            best_realign = local_best_realign
            best_rho = local_best_rho
            best_pt = local_best_pt
        
        if (restart + 1) % 10 == 0:
            print(f"  restart {restart+1}/{n_restarts}: best_realign={best_realign:.6f} pt_min={best_pt}")
    
    return best_rho, best_realign, best_pt

t0 = time.time()
rho9, realign9, pt9 = optimize_d9(n_restarts=100, steps=3000)
elapsed = time.time() - t0

if rho9 is not None and pt9 is not None and pt9 >= 0:
    # Validate K_DW
    dA9, dB9 = 9, 9
    kdw9 = kdw_stinespring(rho9, dA9, dB9, n_bases=200)
    print(f"\n  d=9: realign={realign9:.6f} pt_min={pt9:.2e} K_DW={kdw9:.6f} [{elapsed:.0f}s]")
    
    if kdw9 > 0.001:
        np.savez_compressed('sa_data/phase0q_3x3.npz', rho=rho9,
                            kdw=kdw9, realign=realign9, pt_min=pt9)
        print(f"  ✅ d=9 FIXED and saved! K_DW={kdw9:.4f}")
    else:
        print(f"  ⚠️ d=9 is PPT but K_DW={kdw9:.6f} (near zero)")
else:
    print(f"  ❌ d=9 could not find strictly PPT entangled state [{elapsed:.0f}s]")

print("=" * 60)
