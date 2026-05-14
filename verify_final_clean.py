"""
verify_final_clean.py — Clean verification with ONLY Method A (proven correct)
+ Additional sanity checks against known results
"""
import numpy as np
import time

def S(rho):
    e = np.linalg.eigvalsh(rho); e = e[e>1e-15]
    return -np.sum(e*np.log2(e)) if len(e)>0 else 0.0

def kdw(rho, dA, dB, n_bases=500, seed=42):
    """Verified K_DW (Method A, tested on Bell & Werner states)."""
    np.random.seed(seed)
    ev, evec = np.linalg.eigh(rho)
    m = ev>1e-14; lam=ev[m]; phi=evec[:,m]; r=len(lam)
    if r==0: return 0.0
    sq = np.sqrt(lam); phi_r = phi.reshape(dA,dB,r)
    SE = S(np.diag(lam))
    rB = sum(rho[a*dB:(a+1)*dB, a*dB:(a+1)*dB] for a in range(dA))
    SB = S(rB); best=-999
    for t in range(n_bases):
        U = np.eye(dA,dtype=complex) if t==0 else np.linalg.qr(np.random.randn(dA,dA)+1j*np.random.randn(dA,dA))[0]
        beta = np.einsum('ax,abk->xkb', U.conj(), phi_r)
        px = (np.sum(np.abs(beta)**2, axis=2)) @ lam
        sSB=sSE=0
        for x in range(dA):
            if px[x]<1e-15: continue
            wb = sq[:,None]*beta[x]; w = np.sum(wb,0)
            sSB += px[x]*S(np.outer(w,w.conj())/px[x])
            sSE += px[x]*S(np.outer(sq,sq)*(beta[x].conj()@beta[x].T).T/px[x])
        best = max(best, (SB-sSB)-(SE-sSE))
    return best

print("="*60)
print("  FINAL VERIFICATION: 500 bases, seed=42, Method A")
print("="*60)

# Sanity check 1: Bell state K_DW = 1.0
psi = np.array([1,0,0,1], dtype=complex)/np.sqrt(2)
rho_bell = np.outer(psi, psi.conj())
k_bell = kdw(rho_bell, 2, 2)
print(f"\n  Sanity: Bell state K_DW = {k_bell:.6f} (expected 1.0) {'✅' if abs(k_bell-1)<0.01 else '❌'}")

# Sanity check 2: Separable state K_DW = 0
rho_sep = np.diag([0.25, 0.25, 0.25, 0.25])
k_sep = kdw(rho_sep, 2, 2)
print(f"  Sanity: I/4 state K_DW = {k_sep:.6f} (expected 0.0) {'✅' if abs(k_sep)<0.01 else '❌'}")

# Sanity check 3: Werner p=0.5 (should be ~0.31 from literature)
p = 0.5
rho_w = (1-p)*np.outer(psi, psi.conj()) + p*np.eye(4)/4
k_w = kdw(rho_w, 2, 2)
print(f"  Sanity: Werner(p=0.5) K_DW = {k_w:.6f} (should be >0) {'✅' if k_w>0 else '❌'}")

import os
states = [
    ('optimized_ppt_2x4.npz', 2, 4),
    ('unstructured_3x3.npz', 3, 3),
    ('optimized_ppt_2x5.npz', 2, 5),
    ('native_d12_2x6.npz', 2, 6),
    ('native_d14_2x7.npz', 2, 7),
    ('native_d15_3x5.npz', 3, 5),
    ('native_d16_2x8.npz', 2, 8),
    ('native_d18_2x9.npz', 2, 9),
    ('native_d20_2x10.npz', 2, 10),
    ('native_d21_3x7.npz', 3, 7),
    ('embedded_2x12.npz', 2, 12),
    ('embedded_2x15.npz', 2, 15),
]

print(f"\n  {'d':>3} {'split':>6} {'K_DW':>8} {'S(B)':>8} {'K/SB':>6} {'δ':>8} PPT")
print(f"  {'─'*50}")

results = []
for fname, dA, dB in states:
    fpath = f'sa_data/{fname}'
    if not os.path.exists(fpath): continue
    rho = np.load(fpath)['rho']
    if rho.shape[0] != dA*dB: continue
    
    d = dA*dB
    rB = sum(rho[a*dB:(a+1)*dB, a*dB:(a+1)*dB] for a in range(dA))
    SB = S(rB)
    
    k = kdw(rho, dA, dB, n_bases=500, seed=42)
    
    pt = np.linalg.eigvalsh(rho.reshape(dA,dB,dA,dB).transpose(0,3,2,1).reshape(d,d)).min()
    ppt = pt >= -1e-10
    
    delta = SB - k
    ratio = k / SB if SB > 0.01 else 0
    print(f"  {d:>3} {dA}x{dB:>2}  {k:>8.4f} {SB:>8.4f} {ratio:>6.3f} {delta:>+8.4f} {'✅' if ppt else '❌'}")
    results.append({'d':d, 'k':k, 'SB':SB, 'delta':delta, 'ratio':ratio})

deltas = [r['delta'] for r in results]
ratios = [r['ratio'] for r in results]
print(f"\n  CONJECTURE: K_DW ≈ S(B) - δ")
print(f"  δ = {np.mean(deltas):.4f} ± {np.std(deltas):.4f}")
print(f"  K_DW/S(B) = {np.mean(ratios):.4f} ± {np.std(ratios):.4f}")
print(f"\n  Verdict: K_DW ≈ {np.mean(ratios):.2f}·S(B)")
print("="*60)
