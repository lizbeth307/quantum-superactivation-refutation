"""
smith_yard_v2.py — Correct private state with proper tensor structure.
Systems: A_key(2) ⊗ B_key(2) ⊗ A_shield(2) ⊗ B_shield(2)
Bipartition: A = A_key ⊗ A_shield, B = B_key ⊗ B_shield
"""
import numpy as np, time

def S(rho):
    e = np.linalg.eigvalsh(rho); e = e[e>1e-15]
    return -np.sum(e*np.log2(e)) if len(e)>0 else 0.0

def make_pbit(theta, alpha=np.pi/4):
    """Private bit on (A_key⊗A_shield) ⊗ (B_key⊗B_shield) = C^4 ⊗ C^4.
    
    γ = (1/2) Σ_{i,j} |i⟩⟨j|_{A_key} ⊗ |i⟩⟨j|_{B_key} ⊗ (I ⊗ U_i) σ (I ⊗ U_j†)
    
    Index ordering: (a_key, a_shield, b_key, b_shield)
    Flatten A = a_key*2 + a_shield (dim 4)
    Flatten B = b_key*2 + b_shield (dim 4)
    """
    # Shield state |ψ⟩ = cos(α)|00⟩ + sin(α)|11⟩ on A_shield ⊗ B_shield
    psi_s = np.array([np.cos(alpha), 0, 0, np.sin(alpha)], dtype=complex)
    sigma_shield = np.outer(psi_s, psi_s.conj())  # (4,4) on A_s⊗B_s
    
    # Twisting: U_0 = I, U_1 on B_shield only
    U0 = np.eye(2, dtype=complex)
    U1 = np.diag([1.0, np.exp(1j*theta)])
    
    dA = 4; dB = 4; d = 16
    gamma = np.zeros((d, d), dtype=complex)
    
    for i in range(2):
        for j in range(2):
            Ui = U0 if i == 0 else U1
            Uj = U0 if j == 0 else U1
            # (I_{A_s} ⊗ U_i) σ (I_{A_s} ⊗ U_j†) on (A_s, B_s)
            IUi = np.kron(np.eye(2), Ui)
            IUj = np.kron(np.eye(2), Uj)
            twisted_shield = IUi @ sigma_shield @ IUj.conj().T  # (4,4)
            
            # Place: row = (a_key=i, a_s)*4 + (b_key=i, b_s)
            #         col = (a_key=j, a_s)*4 + (b_key=j, b_s)
            for a_s in range(2):
                for b_s in range(2):
                    for a_s2 in range(2):
                        for b_s2 in range(2):
                            row_A = i*2 + a_s      # A = a_key*2 + a_shield
                            row_B = i*2 + b_s      # B = b_key*2 + b_shield
                            col_A = j*2 + a_s2
                            col_B = j*2 + b_s2
                            row = row_A * dB + row_B
                            col = col_A * dB + col_B
                            gamma[row, col] += 0.5 * twisted_shield[a_s*2+b_s, a_s2*2+b_s2]
    
    return gamma

def partial_transpose_B(rho, dA, dB):
    """Partial transpose on B subsystem."""
    return rho.reshape(dA,dB,dA,dB).transpose(0,3,2,1).reshape(dA*dB, dA*dB)

print("="*60)
print("  SMITH-YARD: Private State Search")
print("="*60)

dA = 4; dB = 4; d = 16
found_ppt = []

for alpha_deg in [10, 20, 30, 40, 45, 50, 60, 70, 80]:
    for theta_deg in range(0, 360, 5):
        alpha = np.radians(alpha_deg)
        theta = np.radians(theta_deg)
        gamma = make_pbit(theta, alpha)
        
        # Basic checks
        tr = np.trace(gamma).real
        if abs(tr - 1) > 1e-10: continue
        eigs = np.linalg.eigvalsh(gamma)
        if eigs.min() < -1e-10: continue
        
        # PPT
        pt = partial_transpose_B(gamma, dA, dB)
        pt_min = np.linalg.eigvalsh(pt).min()
        
        if pt_min >= -1e-10:
            # Realignment
            R = gamma.reshape(dA,dB,dA,dB).transpose(0,2,1,3).reshape(dA*dA, dB*dB)
            Rn = np.linalg.norm(R, 'nuc')
            rB = sum(gamma[a*dB:(a+1)*dB, a*dB:(a+1)*dB] for a in range(dA))
            ci = S(rB) - S(gamma)
            found_ppt.append({
                'alpha': alpha_deg, 'theta': theta_deg,
                'pt_min': pt_min, 'R_norm': Rn, 'CI': ci,
                'gamma': gamma
            })

print(f"\n  Found {len(found_ppt)} PPT private states")
if found_ppt:
    # Show unique
    seen = set()
    unique = []
    for f in found_ppt:
        key = (f['alpha'], round(f['R_norm'],3))
        if key not in seen:
            seen.add(key)
            unique.append(f)
            print(f"  α={f['alpha']:>2}° θ={f['theta']:>3}° PT={f['pt_min']:.2e} ||R||₁={f['R_norm']:.4f} I(A>B)={f['CI']:.4f}")
    
    # Test SA for best candidate (highest realignment norm)
    best = max(unique, key=lambda x: x['R_norm'])
    print(f"\n  Best candidate: α={best['alpha']}° θ={best['theta']}° ||R||₁={best['R_norm']:.4f}")
    
    gamma = best['gamma']
    print(f"  Testing Q(N_PPT ⊗ N_erasure)...")
    
    # Build channel Kraus
    C = gamma * dA
    ev, U = np.linalg.eigh(C)
    K_ppt = [np.sqrt(max(ev[k],0))*U[:,k].reshape(dA,dB).T 
             for k in range(len(ev)) if ev[k]>1e-14]
    Sm = sum(K.conj().T@K for K in K_ppt)
    e2, U2 = np.linalg.eigh(Sm); e2 = np.maximum(e2, 1e-15)
    fix = U2 @ np.diag(1/np.sqrt(e2)) @ U2.conj().T
    K_ppt = [K @ fix for K in K_ppt]
    print(f"  TP check: {np.linalg.norm(sum(K.conj().T@K for K in K_ppt)-np.eye(dA)):.2e}")
    
    # Erasure
    d_era = dA+1
    K_era = []; K0=np.zeros((d_era,dA),dtype=complex)
    K0[:dA,:dA]=np.sqrt(0.5)*np.eye(dA); K_era.append(K0)
    for i in range(dA):
        Ki=np.zeros((d_era,dA),dtype=complex); Ki[dA,i]=np.sqrt(0.5); K_era.append(Ki)
    
    d_in=dA*dA; d_out=dB*d_era; np.random.seed(42); best_ci=-999
    t0 = time.time()
    for t in range(100):
        if t==0:
            psi = np.zeros(d_in*d_in, dtype=complex)
            for i in range(d_in): psi[i*d_in+i]=1/np.sqrt(d_in)
        else:
            psi = np.random.randn(d_in*d_in)+1j*np.random.randn(d_in*d_in)
            psi /= np.linalg.norm(psi)
        rho_in = np.outer(psi, psi.conj())
        rho_out = np.zeros((d_in*d_out, d_in*d_out), dtype=complex)
        for r1 in range(d_in):
            for r2 in range(d_in):
                b = rho_in[r1*d_in:(r1+1)*d_in, r2*d_in:(r2+1)*d_in]
                ob = np.zeros((d_out,d_out),dtype=complex)
                for K1 in K_ppt:
                    for K2 in K_era:
                        Kj = np.kron(K1,K2); ob += Kj@b@Kj.conj().T
                rho_out[r1*d_out:(r1+1)*d_out, r2*d_out:(r2+1)*d_out] = ob
        rB = sum(rho_out[r*d_out:(r+1)*d_out, r*d_out:(r+1)*d_out] for r in range(d_in))
        ci = S(rB) - S(rho_out)
        best_ci = max(best_ci, ci)
    
    print(f"  Q(N_PPT ⊗ N_erasure) = {best_ci:.6f} [{time.time()-t0:.1f}s]")
    if best_ci > 0.001:
        print(f"  🌟 SUPERACTIVATION CONFIRMED!")
    else:
        print(f"  ❌ No SA found (may need more trials or different input)")
else:
    print("  ❌ No PPT states found — construction may need different form")

print(f"\n{'='*60}")
