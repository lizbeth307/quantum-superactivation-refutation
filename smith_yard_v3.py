"""
smith_yard_v3.py — Correct private state with explicit 4-index tensor.
Following Horodecki et al. exactly.

A "private bit" γ on A_k A_s B_k B_s (all qubits):
γ = 1/2 Σ_{ij} |i⟩⟨j|_{Ak} ⊗ |i⟩⟨j|_{Bk} ⊗ (I_As ⊗ U_i^{Bs}) σ_{AsBs} (I_As ⊗ U_j^{Bs})†

Total system: 2⊗2⊗2⊗2 = 16 dims
Alice = Ak⊗As (dim 4), Bob = Bk⊗Bs (dim 4)
"""
import numpy as np, time

def S(rho):
    e = np.linalg.eigvalsh(rho); e = e[e>1e-15]
    return -np.sum(e*np.log2(e)) if len(e)>0 else 0.0

def make_pbit_explicit(theta, alpha=np.pi/4):
    """Build γ as explicit 16x16 matrix with clear A|B split."""
    # 4-index tensor γ[ak*2+as, bk*2+bs, ak'*2+as', bk'*2+bs']
    # Flatten: row = (ak*2+as)*4 + (bk*2+bs) for dA=4, dB=4
    
    psi_s = np.array([np.cos(alpha), 0, 0, np.sin(alpha)], dtype=complex)
    sigma = np.outer(psi_s, psi_s.conj())  # (4,4) on As⊗Bs
    
    U0 = np.eye(2, dtype=complex)
    U1 = np.diag([1.0, np.exp(1j*theta)])
    
    gamma = np.zeros((16, 16), dtype=complex)
    
    for ik in range(2):  # key bit Alice
        for jk in range(2):  # key bit Alice'
            # Twisting
            Ui = U0 if ik == 0 else U1  # on Bs
            Uj = U0 if jk == 0 else U1
            
            for a_s in range(2):
                for b_s in range(2):
                    for a_s2 in range(2):
                        for b_s2 in range(2):
                            # Twisted shield: (I_As ⊗ Ui) σ (I_As ⊗ Uj†)
                            # σ[as*2+bs, as'*2+bs'] -> multiply bs by Ui, bs' by Uj†
                            val = 0
                            for bs_orig in range(2):
                                for bs2_orig in range(2):
                                    val += Ui[b_s, bs_orig] * sigma[a_s*2+bs_orig, a_s2*2+bs2_orig] * Uj[b_s2, bs2_orig].conj()
                            
                            # |ik⟩⟨jk|_Ak ⊗ |ik⟩⟨jk|_Bk means:
                            # Ak=ik, Bk=ik for row;  Ak=jk, Bk=jk for col
                            row_A = ik*2 + a_s  # Alice index
                            row_B = ik*2 + b_s  # Bob index (key bit = ik!)
                            col_A = jk*2 + a_s2
                            col_B = jk*2 + b_s2
                            
                            row = row_A * 4 + row_B  # (Alice)*dB + (Bob)
                            col = col_A * 4 + col_B
                            
                            gamma[row, col] += 0.5 * val
    
    return gamma

print("="*60)
print("  PRIVATE STATE: Explicit Construction")
print("="*60)

# Test basic properties first
gamma = make_pbit_explicit(theta=np.pi/3, alpha=np.pi/4)
print(f"\n  Tr(γ) = {np.trace(gamma).real:.6f}")
print(f"  Hermitian: {np.linalg.norm(gamma-gamma.conj().T):.2e}")
eigs = np.linalg.eigvalsh(gamma)
print(f"  min eig = {eigs.min():.6e}")
print(f"  rank = {np.sum(eigs > 1e-10)}")

# PPT check
dA=4; dB=4
pt = gamma.reshape(dA,dB,dA,dB).transpose(0,3,2,1).reshape(16,16)
pt_eigs = np.linalg.eigvalsh(pt)
print(f"  PT min eig = {pt_eigs.min():.6e}")

# Scan
print(f"\n  Scanning α, θ...")
for alpha_deg in [15, 20, 25, 30, 35, 40, 45]:
    alpha = np.radians(alpha_deg)
    for theta_deg in range(0, 360, 3):
        theta = np.radians(theta_deg)
        g = make_pbit_explicit(theta, alpha)
        if np.linalg.eigvalsh(g).min() < -1e-10: continue
        pt = g.reshape(4,4,4,4).transpose(0,3,2,1).reshape(16,16)
        ptm = np.linalg.eigvalsh(pt).min()
        if ptm >= -1e-10:
            R = g.reshape(4,4,4,4).transpose(0,2,1,3).reshape(16,16)
            Rn = np.linalg.norm(R, 'nuc')
            print(f"  ✅ PPT: α={alpha_deg}° θ={theta_deg}° ptmin={ptm:.4e} ||R||₁={Rn:.4f}")

# Also try: key dim > 2
print(f"\n  Trying different approach: direct Smith-Yard channel")
# Smith-Yard use the "Horodecki channel" which is known analytically
# It maps C^3 -> C^3 and its Choi state is PPT entangled
# From Horodecki 1997: the 3x3 PPT entangled state family

def horodecki_3x3(a):
    """Horodecki 3⊗3 PPT entangled state, a ∈ (0,1)."""
    rho = np.array([
        [a, 0, 0, 0, a, 0, 0, 0, a],
        [0, a, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, a, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, a, 0, 0, 0, 0, 0],
        [a, 0, 0, 0, a, 0, 0, 0, a],
        [0, 0, 0, 0, 0, a, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, (1+a)/2, 0, np.sqrt(1-a**2)/2],
        [0, 0, 0, 0, 0, 0, 0, a, 0],
        [a, 0, 0, 0, a, 0, np.sqrt(1-a**2)/2, 0, (1+a)/2],
    ], dtype=complex) / (8*a + 1)
    return rho

print(f"\n  Horodecki 3x3 family (a ∈ (0,1)):")
for a in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
    rho = horodecki_3x3(a)
    tr = np.trace(rho).real
    eigs = np.linalg.eigvalsh(rho)
    pt = rho.reshape(3,3,3,3).transpose(0,3,2,1).reshape(9,9)
    ptm = np.linalg.eigvalsh(pt).min()
    R = rho.reshape(3,3,3,3).transpose(0,2,1,3).reshape(9,9)
    Rn = np.linalg.norm(R, 'nuc')
    ppt = ptm >= -1e-10
    ent = Rn > 1+1e-6
    print(f"  a={a:.1f}: Tr={tr:.3f} PSD={eigs.min():.2e} PT={ptm:.4e} ||R||={Rn:.4f} {'PPT' if ppt else 'NPT'} {'ENT' if ent else 'SEP?'}")

print("="*60)
