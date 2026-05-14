"""
debug_kdw.py — Find the bug: which K_DW method is correct?
Test on a KNOWN state with analytical K_DW.
"""
import numpy as np

def S(rho):
    e = np.linalg.eigvalsh(rho); e = e[e>1e-15]
    return -np.sum(e*np.log2(e)) if len(e)>0 else 0.0

# Test 1: Maximally entangled state |Φ+> = (|00>+|11>)/√2
# K_DW should = 1 bit (perfect correlation)
print("="*55)
print("  TEST 1: Bell state |Φ+> (K_DW should = 1.0)")
print("="*55)
psi = np.array([1,0,0,1], dtype=complex) / np.sqrt(2)
rho = np.outer(psi, psi.conj())
dA, dB = 2, 2

# Method A
def kdw_A(rho, dA, dB, n_bases=100):
    ev, evec = np.linalg.eigh(rho)
    m = ev>1e-14; lam=ev[m]; phi=evec[:,m]; r=len(lam)
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

# Method B
def kdw_B(rho, dA, dB, n_bases=100):
    ev, evec = np.linalg.eigh(rho)
    m = ev>1e-14; lam=ev[m]; vecs=evec[:,m]; r=len(lam)
    rB = sum(rho[a*dB:(a+1)*dB, a*dB:(a+1)*dB] for a in range(dA))
    HB = S(rB); HE = S(np.diag(lam))
    phi_r = vecs.reshape(dA,dB,r)
    best=-999
    for t in range(n_bases):
        U = np.eye(dA,dtype=complex) if t==0 else np.linalg.qr(np.random.randn(dA,dA)+1j*np.random.randn(dA,dA))[0]
        phi_U = np.einsum('ax,abk->xbk', U.conj(), phi_r)
        sHBx=sHEx=0
        for x in range(dA):
            coeff = np.sqrt(lam)[None,:] * phi_U[x]  # (dB, r)
            px = np.sum(np.abs(coeff)**2)
            if px < 1e-15: continue
            coeff_n = coeff / np.sqrt(px)
            rho_Bx = coeff_n @ coeff_n.conj().T
            rho_Ex = coeff_n.conj().T @ coeff_n
            sHBx += px * S(rho_Bx)
            sHEx += px * S(rho_Ex)
        IXB = HB - sHBx
        IXE = HE - sHEx
        best = max(best, IXB - IXE)
    return best

np.random.seed(42)
kA = kdw_A(rho, dA, dB)
np.random.seed(42)
kB = kdw_B(rho, dA, dB)
print(f"  Method A: {kA:.6f}")
print(f"  Method B: {kB:.6f}")
print(f"  Expected: 1.000000")

# Test 2: Werner state ρ = (1-p)|Φ+><Φ+| + p*I/4
print(f"\n{'='*55}")
print("  TEST 2: Werner state p=0.3 (known K_DW > 0)")
print("="*55)
p = 0.3
rho_w = (1-p) * np.outer(psi, psi.conj()) + p * np.eye(4) / 4
np.random.seed(42)
kA = kdw_A(rho_w, 2, 2)
np.random.seed(42) 
kB = kdw_B(rho_w, 2, 2)
print(f"  Method A: {kA:.6f}")
print(f"  Method B: {kB:.6f}")

# Test 3: Detailed trace through Method B for Bell state
print(f"\n{'='*55}")
print("  TEST 3: Debug Method B step-by-step (Bell state)")
print("="*55)
ev, evec = np.linalg.eigh(rho)
m = ev>1e-14; lam=ev[m]; vecs=evec[:,m]; r=len(lam)
print(f"  eigenvalues: {lam}")
print(f"  rank: {r}")
rB = sum(rho[a*dB:(a+1)*dB, a*dB:(a+1)*dB] for a in range(dA))
print(f"  ρ_B = {np.diag(rB).real}")
print(f"  H(B) = {S(rB):.6f}")
print(f"  H(E) = {S(np.diag(lam)):.6f}")

phi_r = vecs.reshape(dA, dB, r)
print(f"  phi_r shape: {phi_r.shape}")
print(f"  phi_r[0,:,0] = {phi_r[0,:,0]}")
print(f"  phi_r[1,:,0] = {phi_r[1,:,0]}")

U = np.eye(dA, dtype=complex)
phi_U = np.einsum('ax,abk->xbk', U.conj(), phi_r)
for x in range(dA):
    coeff = np.sqrt(lam)[None,:] * phi_U[x]
    px = np.sum(np.abs(coeff)**2)
    print(f"\n  x={x}: p(x)={px:.4f}")
    coeff_n = coeff / np.sqrt(px)
    rBx = coeff_n @ coeff_n.conj().T
    rEx = coeff_n.conj().T @ coeff_n
    print(f"    ρ_B|x diag = {np.diag(rBx).real}")
    print(f"    H(B|x) = {S(rBx):.6f}")
    print(f"    ρ_E|x diag = {np.diag(rEx).real}")
    print(f"    H(E|x) = {S(rEx):.6f}")

# Check: for Bell state with Z measurement, 
# p(0) = p(1) = 0.5, ρ_B|0 = |0><0|, ρ_B|1 = |1><1|
# So H(B|X) = 0, I(X:B) = H(B) - 0 = 1
# ρ_E = pure → H(E) = 0, H(E|X) = 0 → I(X:E) = 0
# K_DW = 1 - 0 = 1 ✓
