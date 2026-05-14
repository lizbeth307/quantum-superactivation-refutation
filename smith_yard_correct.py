"""
smith_yard_correct.py — Correct Smith-Yard superactivation from first principles.

Private state γ = U (|Φ+⟩⟨Φ+|_key ⊗ σ_shield) U†
where U is a "twisting" unitary that makes γ PPT.

Construction from Horodecki et al. (2005/2008):
Key: qubit (d_K=2), Shield: qubit (d_S=2)
Total: 4x4 state on (A_key⊗A_shield) ⊗ (B_key⊗B_shield)
"""
import numpy as np, time

def S(rho):
    e = np.linalg.eigvalsh(rho); e = e[e>1e-15]
    return -np.sum(e*np.log2(e)) if len(e)>0 else 0.0

def make_private_state(theta):
    """Construct PPT private state (pbit) with parameter theta.
    
    γ = (1/2) Σ_{i,j∈{0,1}} |ii⟩⟨jj|_{A_k B_k} ⊗ U_i σ U_j†
    
    Key: A_k, B_k (qubits), Shield: A_s, B_s (qubits)
    Total dim: 4x4 = 16 (but arranged as 2x2 key ⊗ 2x2 shield)
    
    For PPT: use U_0 = I, U_1 = diag(1, e^{iθ}) on shield B_s.
    σ = |ψ(α)⟩⟨ψ(α)| where |ψ⟩ = cos(α)|00⟩ + sin(α)|11⟩
    """
    # Shield state
    alpha = np.pi/4  # maximally entangled shield
    psi_shield = np.array([np.cos(alpha), 0, 0, np.sin(alpha)], dtype=complex)
    sigma = np.outer(psi_shield, psi_shield.conj())  # 4x4 on A_s⊗B_s
    
    # Twisting unitaries on B_shield
    U0 = np.eye(2, dtype=complex)
    U1 = np.diag([1.0, np.exp(1j*theta)])
    
    # Key system: |00⟩, |01⟩, |10⟩, |11⟩ on A_k⊗B_k
    # Build γ on (A_k⊗A_s) ⊗ (B_k⊗B_s) = 4⊗4 = 16×16
    # Index: (a_k*2+a_s)*4 + (b_k*2+b_s) ... actually (a_k,a_s,b_k,b_s)
    # Flatten: idx = a_k*8 + a_s*4 + b_k*2 + b_s
    
    d = 16  # 4⊗4
    gamma = np.zeros((d, d), dtype=complex)
    
    for i in range(2):  # key bit i
        for j in range(2):  # key bit j
            # |ii⟩⟨jj|_key = |i⟩⟨j|_{A_k} ⊗ |i⟩⟨j|_{B_k}
            # U_i σ U_j† on shield: (I_{A_s} ⊗ U_i) σ (I_{A_s} ⊗ U_j†)
            Ui = U0 if i == 0 else U1
            Uj = U0 if j == 0 else U1
            
            # shield block: (I⊗Ui) σ (I⊗Uj†)
            IUi = np.kron(np.eye(2), Ui)  # 4x4
            IUj = np.kron(np.eye(2), Uj)  # 4x4
            shield_block = IUi @ sigma @ IUj.conj().T  # 4x4
            
            # Place in γ: rows (a_k=i, a_s, b_k=i, b_s), cols (a_k=j, a_s, b_k=j, b_s)
            for a_s in range(2):
                for b_s in range(2):
                    for a_s2 in range(2):
                        for b_s2 in range(2):
                            row = i*8 + a_s*4 + i*2 + b_s
                            col = j*8 + a_s2*4 + j*2 + b_s2
                            gamma[row, col] += 0.5 * shield_block[a_s*2+b_s, a_s2*2+b_s2]
    
    return gamma

def check_state(gamma, name, dA=4, dB=4):
    d = dA*dB
    print(f"\n  ══ {name} ══")
    print(f"  Tr = {np.trace(gamma).real:.6f}")
    print(f"  Hermitian err = {np.linalg.norm(gamma-gamma.conj().T):.2e}")
    eigs = np.linalg.eigvalsh(gamma)
    print(f"  PSD: min_eig = {eigs.min():.6e}")
    
    # PPT (partial transpose on B = B_key⊗B_shield)
    gamma_pt = gamma.reshape(dA,dB,dA,dB).transpose(0,3,2,1).reshape(d,d)
    pt_min = np.linalg.eigvalsh(gamma_pt).min()
    print(f"  PPT: min_eig(PT) = {pt_min:.6e} {'✅ PPT' if pt_min>=-1e-10 else '❌ NPT'}")
    
    # Realignment
    R = gamma.reshape(dA,dB,dA,dB).transpose(0,2,1,3).reshape(dA*dA, dB*dB)
    R_norm = np.linalg.norm(R, 'nuc')
    print(f"  ||R||₁ = {R_norm:.6f} {'→ entangled' if R_norm>1+1e-6 else ''}")
    
    # Key rate via correct K_DW (H(X|E) - H(X|B))
    # For private state: key rate = log2(K) = 1 bit
    # Let's verify by measuring key subsystem
    rB = sum(gamma[a*dB:(a+1)*dB, a*dB:(a+1)*dB] for a in range(dA))
    print(f"  S(B) = {S(rB):.4f}")
    print(f"  S(AB) = {S(gamma):.4f}")
    print(f"  I(A>B) = {S(rB) - S(gamma):.4f}")
    
    return pt_min >= -1e-10

def test_superactivation(gamma, dA=4, dB=4, n_trials=100):
    """Test Q(N_γ ⊗ N_erasure) > 0."""
    # Build Kraus for PPT channel from Choi state γ
    # Channel maps C^dA → C^dB
    C = gamma * dA
    ev, U = np.linalg.eigh(C)
    K_ppt = []
    for i in range(len(ev)):
        if ev[i] > 1e-14:
            K = np.sqrt(ev[i]) * U[:, i].reshape(dA, dB).T  # (dB, dA)
            K_ppt.append(K)
    # TP fix
    Sm = sum(K.conj().T@K for K in K_ppt)
    e2, U2 = np.linalg.eigh(Sm); e2 = np.maximum(e2, 1e-15)
    fix = U2 @ np.diag(1/np.sqrt(e2)) @ U2.conj().T
    K_ppt = [K @ fix for K in K_ppt]
    
    # Erasure channel on dA
    d_era = dA + 1
    K_era = []
    K0 = np.zeros((d_era, dA), dtype=complex)
    K0[:dA,:dA] = np.sqrt(0.5)*np.eye(dA); K_era.append(K0)
    for i in range(dA):
        Ki = np.zeros((d_era, dA), dtype=complex)
        Ki[dA, i] = np.sqrt(0.5); K_era.append(Ki)
    
    d_in = dA * dA
    d_out = dB * d_era
    d_ref = d_in
    
    np.random.seed(42)
    best = -999
    for t in range(n_trials):
        if t == 0:
            psi = np.zeros(d_ref*d_in, dtype=complex)
            for i in range(d_in): psi[i*d_in+i] = 1/np.sqrt(d_in)
        else:
            psi = np.random.randn(d_ref*d_in)+1j*np.random.randn(d_ref*d_in)
            psi /= np.linalg.norm(psi)
        
        rho_in = np.outer(psi, psi.conj())
        rho_out = np.zeros((d_ref*d_out, d_ref*d_out), dtype=complex)
        for r1 in range(d_ref):
            for r2 in range(d_ref):
                block = rho_in[r1*d_in:(r1+1)*d_in, r2*d_in:(r2+1)*d_in]
                out_block = np.zeros((d_out, d_out), dtype=complex)
                for K1 in K_ppt:
                    for K2 in K_era:
                        Kj = np.kron(K1, K2)
                        out_block += Kj @ block @ Kj.conj().T
                rho_out[r1*d_out:(r1+1)*d_out, r2*d_out:(r2+1)*d_out] = out_block
        
        rho_B = sum(rho_out[r*d_out:(r+1)*d_out, r*d_out:(r+1)*d_out] for r in range(d_ref))
        ci = S(rho_B) - S(rho_out)
        best = max(best, ci)
    
    return best

# === MAIN ===
print("="*60)
print("  SMITH-YARD PRIVATE STATE CONSTRUCTION")
print("="*60)

# Scan theta to find PPT private state
best_theta = None; best_ci = -999
for theta_deg in range(0, 360, 15):
    theta = np.radians(theta_deg)
    gamma = make_private_state(theta)
    d = 16; dA = 4; dB = 4
    gamma_pt = gamma.reshape(dA,dB,dA,dB).transpose(0,3,2,1).reshape(d,d)
    pt_min = np.linalg.eigvalsh(gamma_pt).min()
    if pt_min >= -1e-10:  # PPT!
        print(f"  θ={theta_deg:>3}°: PPT ✅ (pt_min={pt_min:.4e})")
        if best_theta is None:
            best_theta = theta

if best_theta is not None:
    print(f"\n  Best PPT private state: θ={np.degrees(best_theta):.0f}°")
    gamma = make_private_state(best_theta)
    is_ppt = check_state(gamma, f"Private State θ={np.degrees(best_theta):.0f}°")
    
    if is_ppt:
        print(f"\n  Testing superactivation...")
        t0 = time.time()
        ci = test_superactivation(gamma, dA=4, dB=4, n_trials=50)
        print(f"  Q(N_PPT ⊗ N_erasure) = {ci:.6f} [{time.time()-t0:.1f}s]")
        print(f"  {'🌟 SUPERACTIVATION!' if ci > 0.001 else '❌ No SA'}")
else:
    print("\n  ❌ No PPT private state found — trying different construction")
    # Try with non-maximally entangled shield
    for alpha_deg in range(10, 90, 10):
        for theta_deg in range(0, 360, 30):
            alpha = np.radians(alpha_deg)
            theta = np.radians(theta_deg)
            psi_s = np.array([np.cos(alpha),0,0,np.sin(alpha)],dtype=complex)
            sigma = np.outer(psi_s, psi_s.conj())
            U1 = np.diag([1.0, np.exp(1j*theta)])
            
            d=16; gamma = np.zeros((d,d),dtype=complex)
            for i in range(2):
                for j in range(2):
                    Ui = np.eye(2) if i==0 else U1
                    Uj = np.eye(2) if j==0 else U1
                    sb = np.kron(np.eye(2),Ui) @ sigma @ np.kron(np.eye(2),Uj).conj().T
                    for as1 in range(2):
                        for bs1 in range(2):
                            for as2 in range(2):
                                for bs2 in range(2):
                                    r = i*8+as1*4+i*2+bs1
                                    c = j*8+as2*4+j*2+bs2
                                    gamma[r,c] += 0.5*sb[as1*2+bs1, as2*2+bs2]
            
            pt = gamma.reshape(4,4,4,4).transpose(0,3,2,1).reshape(16,16)
            if np.linalg.eigvalsh(pt).min() >= -1e-10:
                print(f"  Found PPT: α={alpha_deg}°, θ={theta_deg}°")
                check_state(gamma, f"α={alpha_deg}°,θ={theta_deg}°")
                ci = test_superactivation(gamma, 4, 4, 30)
                print(f"  Q(joint) = {ci:.6f} {'🌟 SA!' if ci>0.001 else ''}")
                if ci > 0.001: break

print(f"\n{'='*60}")
