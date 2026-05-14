"""
kdw_correct.py — Correct Devetak-Winter key rate from first principles
K_DW = max_M [H(X|E) - H(X|B)]
     = max_M [H(A|E) - H(A|B)]  for projective M on A

For state ρ_AB with purification |ψ⟩_ABE:
  H(X|B) = H(XB) - H(B) 
  H(X|E) = H(XE) - H(E)
"""
import numpy as np, os

def S(rho):
    e = np.linalg.eigvalsh(rho); e = e[e>1e-15]
    return -np.sum(e*np.log2(e)) if len(e)>0 else 0.0

def kdw_correct(rho, dA, dB, n_bases=300, seed=42):
    """Correct K_DW = max_U [H(X|E) - H(X|B)]
    
    After measurement on A in basis U, outcome x:
    ρ_XB = Σ_x p(x)|x><x| ⊗ ρ_B|x
    ρ_XE = Σ_x p(x)|x><x| ⊗ ρ_E|x
    
    H(X|B) = H(XB) - H(B) = H(Σ p_x|x><x|⊗ρ_Bx) - H(ρ_B)
    H(X|E) = H(XE) - H(E) = H(Σ p_x|x><x|⊗ρ_Ex) - H(ρ_E)
    """
    np.random.seed(seed)
    d = dA * dB
    
    # Eigendecompose
    ev, evec = np.linalg.eigh(rho)
    m = ev > 1e-14; lam = ev[m]; vecs = evec[:, m]; r = len(lam)
    if r == 0: return 0.0
    
    phi_r = vecs.reshape(dA, dB, r)
    
    # Marginals
    rB = sum(rho[a*dB:(a+1)*dB, a*dB:(a+1)*dB] for a in range(dA))
    HB = S(rB)
    HE = S(np.diag(lam))  # ρ_E = Tr_AB(|ψ><ψ|) = diag(λ)
    
    best = -999
    for t in range(n_bases):
        U = np.eye(dA, dtype=complex) if t == 0 else \
            np.linalg.qr(np.random.randn(dA,dA)+1j*np.random.randn(dA,dA))[0]
        
        # Build ρ_XB and ρ_XE block-diagonal
        rho_XB = np.zeros((dA*dB, dA*dB), dtype=complex)
        rho_XE = np.zeros((dA*r, dA*r), dtype=complex)
        
        for x in range(dA):
            # β[k,b] = Σ_a U*[a,x] φ[a,b,k]
            beta_x = np.einsum('a,abk->kb', U[:, x].conj(), phi_r)  # (r, dB)
            
            # ρ_B|x = Σ_k λ_k |β_k><β_k| (correct!)
            # In matrix form: diag(λ) is (r,r), beta_x is (r,dB)
            # ρ_B|x = beta_x† @ diag(λ) @ beta_x
            rBx = beta_x.conj().T @ np.diag(lam) @ beta_x  # (dB, dB)
            # p(x) = Tr(ρ_B|x)
            px = np.trace(rBx).real
            
            # ρ_E|x = Σ_b (λ_k λ_k')^{1/2}... 
            # Actually: ρ_E|x[k,k'] = Σ_b β[k,b] β*[k',b] √λ_k √λ_k'... no
            # ρ_E|x = diag(√λ) @ (β_x @ β_x†) @ diag(√λ) 
            # Wait... let me re-derive.
            # |φ_x⟩_BE = Σ_k √λ_k |β_xk⟩_B |k⟩_E
            # ρ_E|x = Tr_B(|φ_x><φ_x|) = Σ_k,k' √λ_k √λ_k' <β_xk'|β_xk> |k><k'|
            # = diag(√λ) @ Gram @ diag(√λ)
            # where Gram[k,k'] = <β_xk'|β_xk> = Σ_b β*[k',b] β[k,b] = (β_x @ β_x†)^T?
            # No: Gram[k,k'] = Σ_b β_x[k,b] β_x*[k',b] = (β_x @ β_x.conj().T)[k,k']
            Gram = beta_x @ beta_x.conj().T  # (r, r)
            sq = np.sqrt(lam)
            rEx = np.outer(sq, sq) * Gram  # (r, r)
            
            if px > 1e-15:
                rho_XB[x*dB:(x+1)*dB, x*dB:(x+1)*dB] = rBx  # unnormalized block
                rho_XE[x*r:(x+1)*r, x*r:(x+1)*r] = rEx  # unnormalized block
        
        # H(XB) and H(XE) - these are block-diagonal, so entropy = Σ contributions
        HXB = S(rho_XB)
        HXE = S(rho_XE)
        
        HX_given_B = HXB - HB
        HX_given_E = HXE - HE
        
        kdw_val = HX_given_E - HX_given_B
        best = max(best, kdw_val)
    
    return best


# === SANITY CHECKS ===
print("="*60)
print("  CORRECT K_DW: H(X|E) - H(X|B)")
print("="*60)

psi = np.array([1,0,0,1], dtype=complex)/np.sqrt(2)
rho_bell = np.outer(psi, psi.conj())

k = kdw_correct(rho_bell, 2, 2, n_bases=100)
print(f"  Bell |Φ+⟩:    K_DW = {k:.6f}  (expect 1.0) {'✅' if abs(k-1)<0.05 else '❌'}")

rho_sep = np.eye(4)/4
k = kdw_correct(rho_sep, 2, 2, n_bases=100)
print(f"  I/4 (sep):    K_DW = {k:.6f}  (expect ≤0) {'✅' if k<=0.01 else '❌'}")

p=0.5; rho_w = (1-p)*rho_bell + p*np.eye(4)/4
k = kdw_correct(rho_w, 2, 2, n_bases=200)
print(f"  Werner(0.5):  K_DW = {k:.6f}  (expect ~0.31)")

p=0.3; rho_w3 = (1-p)*rho_bell + p*np.eye(4)/4
k3 = kdw_correct(rho_w3, 2, 2, n_bases=200)
print(f"  Werner(0.3):  K_DW = {k3:.6f}  (expect ~0.59)")

# === RE-VERIFY SA STATES ===
print(f"\n{'='*60}")
print("  SA STATES — Correct K_DW")
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

print(f"\n  {'d':>3} {'split':>6} {'K_DW':>8} {'S(B)':>7} PPT  SA?")
print(f"  {'─'*42}")

sa_count = 0
for fname, dA, dB in states:
    fpath = f'sa_data/{fname}'
    if not os.path.exists(fpath): continue
    rho = np.load(fpath)['rho']
    d = dA*dB
    if rho.shape[0] != d: continue
    rB = sum(rho[a*dB:(a+1)*dB, a*dB:(a+1)*dB] for a in range(dA))
    SB = S(rB)
    k = kdw_correct(rho, dA, dB, n_bases=500, seed=42)
    pt = np.linalg.eigvalsh(rho.reshape(dA,dB,dA,dB).transpose(0,3,2,1).reshape(d,d)).min()
    ppt = pt >= -1e-10
    sa = k > 0.01 and ppt
    if sa: sa_count += 1
    print(f"  {d:>3} {dA}x{dB:>2}  {k:>8.4f} {SB:>7.3f} {'✅' if ppt else '❌'}  {'🌟' if sa else '❌'}")

print(f"\n  Confirmed SA: {sa_count}/12")
print("="*60)
