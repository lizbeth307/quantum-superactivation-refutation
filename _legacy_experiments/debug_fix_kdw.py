"""
debug_fix_kdw.py — Fix the fundamental Stinespring bug and re-verify
BUG: ρ_B|x was computed as |Σ_k w_k⟩⟨Σ_k w_k| (rank-1 pure state)
FIX: ρ_B|x = Σ_k λ_k |β_xk⟩⟨β_xk| / p_x  (correct mixed state)
"""
import numpy as np, os

def S(rho):
    e = np.linalg.eigvalsh(rho); e = e[e>1e-15]
    return -np.sum(e*np.log2(e)) if len(e)>0 else 0.0

def kdw_FIXED(rho, dA, dB, n_bases=300, seed=42):
    """FIXED K_DW: correct ρ_B|x = Σ_k λ_k |β_xk⟩⟨β_xk| / p_x"""
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
            wb = sq[:,None]*beta[x]  # (r, dB)
            # FIX: ρ_B|x = wb† @ wb / p_x  (mixed state, not pure!)
            rho_Bx = wb.conj().T @ wb / px[x]  # (dB, dB)
            sSB += px[x]*S(rho_Bx)
            # ρ_E|x unchanged (was correct)
            gram = beta[x].conj() @ beta[x].T
            sSE += px[x]*S(np.outer(sq,sq) * gram.T / px[x])
        best = max(best, (SB-sSB)-(SE-sSE))
    return best

# === SANITY CHECKS ===
print("="*60)
print("  K_DW BUG FIX: Sanity Checks")
print("="*60)

psi = np.array([1,0,0,1], dtype=complex)/np.sqrt(2)
rho_bell = np.outer(psi, psi.conj())
print(f"  Bell |Φ+⟩:    K_DW = {kdw_FIXED(rho_bell, 2, 2):.6f}  (expect 1.0)")

rho_sep = np.eye(4)/4
print(f"  I/4 (sep):    K_DW = {kdw_FIXED(rho_sep, 2, 2):.6f}  (expect 0.0)")

p=0.5; rho_w = (1-p)*rho_bell + p*np.eye(4)/4
print(f"  Werner(0.5):  K_DW = {kdw_FIXED(rho_w, 2, 2):.6f}  (expect ~0.31)")

p=0.9; rho_w9 = (1-p)*rho_bell + p*np.eye(4)/4
print(f"  Werner(0.9):  K_DW = {kdw_FIXED(rho_w9, 2, 2):.6f}  (expect ~0)")

# === RE-VERIFY ALL SA STATES ===
print(f"\n{'='*60}")
print("  RE-VERIFICATION: All SA states with FIXED K_DW")
print("="*60)

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

print(f"\n  {'d':>3} {'split':>6} {'K_DW_old':>9} {'K_DW_fix':>9} {'S(B)':>7} {'K/SB':>6} PPT SA?")
print(f"  {'─'*62}")

results = []
for fname, dA, dB in states:
    fpath = f'sa_data/{fname}'
    if not os.path.exists(fpath): continue
    rho = np.load(fpath)['rho']
    d = dA*dB
    if rho.shape[0] != d: continue
    rB = sum(rho[a*dB:(a+1)*dB, a*dB:(a+1)*dB] for a in range(dA))
    SB = S(rB)
    k_fix = kdw_FIXED(rho, dA, dB, n_bases=500, seed=42)
    pt = np.linalg.eigvalsh(rho.reshape(dA,dB,dA,dB).transpose(0,3,2,1).reshape(d,d)).min()
    ppt = pt >= -1e-10
    ratio = k_fix/SB if SB>0.01 else 0
    sa = k_fix > 0.01 and ppt
    print(f"  {d:>3} {dA}x{dB:>2}  {'???':>9} {k_fix:>9.4f} {SB:>7.3f} {ratio:>6.3f} {'✅' if ppt else '❌'} {'🌟' if sa else '❌'}")
    results.append({'d':d, 'k':k_fix, 'SB':SB, 'ratio':ratio, 'sa':sa})

sa_count = sum(1 for r in results if r['sa'])
print(f"\n  SA states confirmed: {sa_count}/{len(results)}")
if results:
    ratios = [r['ratio'] for r in results if r['sa']]
    if ratios:
        print(f"  K_DW/S(B) = {np.mean(ratios):.4f} ± {np.std(ratios):.4f}")
print("="*60)
