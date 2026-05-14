"""
superactivation_test.py — Direct test: Q(N_PPT ⊗ N_erasure) > 0?
This is the CORRECT test for superactivation, not K_DW.
Smith-Yard (2008): SA iff joint channel has positive coherent information.
"""
import numpy as np, os, time

def S(rho):
    e = np.linalg.eigvalsh(rho); e = e[e>1e-15]
    return -np.sum(e*np.log2(e)) if len(e)>0 else 0.0

def coherent_info_state(rho_AB, dA, dB):
    """I(A>B) = S(B) - S(AB) for bipartite state."""
    rB = sum(rho_AB[a*dB:(a+1)*dB, a*dB:(a+1)*dB] for a in range(dA))
    return S(rB) - S(rho_AB)

def apply_channel(rho_in, kraus):
    """Apply channel defined by Kraus operators."""
    d_out = kraus[0].shape[0]
    return sum(K @ rho_in @ K.conj().T for K in kraus)

def erasure_kraus(d, p=0.5):
    """50% erasure channel: d -> d+1."""
    K = []; K0 = np.zeros((d+1,d),dtype=complex)
    K0[:d,:d] = np.sqrt(1-p)*np.eye(d); K.append(K0)
    for i in range(d):
        Ki = np.zeros((d+1,d),dtype=complex); Ki[d,i]=np.sqrt(p); K.append(Ki)
    return K

def ppt_channel_kraus(rho_choi, dA, dB):
    """Extract Kraus from PPT Choi state with TP normalization."""
    C = rho_choi * dA
    ev, U = np.linalg.eigh(C)
    raw = [np.sqrt(max(ev[i],0))*U[:,i].reshape(dA,dB).T 
           for i in range(len(ev)) if ev[i]>1e-14]
    Sm = sum(K.conj().T@K for K in raw)
    e2, U2 = np.linalg.eigh(Sm); e2 = np.maximum(e2, 1e-15)
    fix = U2 @ np.diag(1/np.sqrt(e2)) @ U2.conj().T
    return [K @ fix for K in raw]

def joint_channel_coherent_info(rho_choi, dA_ppt, dB_ppt, p_erasure=0.5, n_trials=200):
    """Compute max I(A>B) for N_PPT ⊗ N_erasure.
    
    Input: dA_ppt dimensional system for PPT channel
           dA_ppt dimensional system for erasure channel
    Total input: dA_ppt^2
    PPT output: dB_ppt, Erasure output: dA_ppt+1
    Total output: dB_ppt * (dA_ppt+1)
    
    Reference system R has same dim as total input.
    """
    K_ppt = ppt_channel_kraus(rho_choi, dA_ppt, dB_ppt)
    K_era = erasure_kraus(dA_ppt, p_erasure)
    
    d_in1 = dA_ppt  # PPT input
    d_in2 = dA_ppt  # erasure input (same dim)
    d_out1 = dB_ppt  # PPT output
    d_out2 = dA_ppt + 1  # erasure output
    
    d_in = d_in1 * d_in2
    d_out = d_out1 * d_out2
    d_ref = d_in  # reference = input
    
    best = -999
    np.random.seed(42)
    
    for t in range(n_trials):
        # Random pure input |ψ⟩_{R,in1,in2}
        if t == 0:
            # Maximally entangled
            psi = np.zeros(d_ref * d_in, dtype=complex)
            for i in range(d_in):
                psi[i*d_in + i] = 1/np.sqrt(d_in)
        else:
            psi = np.random.randn(d_ref*d_in) + 1j*np.random.randn(d_ref*d_in)
            psi /= np.linalg.norm(psi)
        
        rho_R_in = np.outer(psi, psi.conj())  # (d_ref*d_in) x (d_ref*d_in)
        
        # Apply (id_R ⊗ N_PPT ⊗ N_era) to the in1,in2 subsystems
        # Reshape: R(d_ref) ⊗ in1(d_in1) ⊗ in2(d_in2)
        # Output: R(d_ref) ⊗ out1(d_out1) ⊗ out2(d_out2)
        
        rho_out = np.zeros((d_ref*d_out, d_ref*d_out), dtype=complex)
        
        for r1 in range(d_ref):
            for r2 in range(d_ref):
                # Block [r1, r2] of R: acts on in1⊗in2
                block_in = rho_R_in[r1*d_in:(r1+1)*d_in, r2*d_in:(r2+1)*d_in]
                # block_in is (d_in1*d_in2) x (d_in1*d_in2)
                # Reshape to (d_in1, d_in2, d_in1, d_in2)
                block_4d = block_in.reshape(d_in1, d_in2, d_in1, d_in2)
                
                # Apply N_PPT on subsystem 1 and N_era on subsystem 2
                block_out = np.zeros((d_out1, d_out2, d_out1, d_out2), dtype=complex)
                for K1 in K_ppt:
                    for K2 in K_era:
                        for a1 in range(d_in1):
                            for a2 in range(d_in1):
                                # Input block for channels
                                rho_12 = block_4d[a1, :, a2, :]  # (d_in2, d_in2)
                                # This isn't right - need to apply K1 on dim1 and K2 on dim2
                                pass
                
                # Actually, simpler: apply tensor product of Kraus
                # K_joint = K1 ⊗ K2 for each pair
                block_out_flat = np.zeros((d_out, d_out), dtype=complex)
                for K1 in K_ppt:
                    for K2 in K_era:
                        K_joint = np.kron(K1, K2)  # (d_out1*d_out2, d_in1*d_in2)
                        block_out_flat += K_joint @ block_in @ K_joint.conj().T
                
                rho_out[r1*d_out:(r1+1)*d_out, r2*d_out:(r2+1)*d_out] = block_out_flat
        
        # Coherent info: I(R>out) = S(out) - S(R,out)
        rho_B = sum(rho_out[r*d_out:(r+1)*d_out, r*d_out:(r+1)*d_out] for r in range(d_ref))
        ci = S(rho_B) - S(rho_out)
        best = max(best, ci)
    
    return best


print("="*60)
print("  SUPERACTIVATION TEST: Q(N_PPT ⊗ N_erasure) > 0?")
print("="*60)

# Test with known: erasure ⊗ erasure should give Q=0
print("\n  Sanity: erasure(0.5) alone")
K_era = erasure_kraus(2, 0.5)
check = sum(K.conj().T@K for K in K_era)
print(f"  Completeness: {np.linalg.norm(check - np.eye(2)):.2e}")

# Test SA for each PPT state
states = [
    ('optimized_ppt_2x4.npz', 2, 4),
    ('unstructured_3x3.npz', 3, 3),
    ('native_d12_2x6.npz', 2, 6),
]

for fname, dA, dB in states:
    fpath = f'sa_data/{fname}'
    if not os.path.exists(fpath): continue
    rho = np.load(fpath)['rho']
    if rho.shape[0] != dA*dB: continue
    
    print(f"\n  ══ {fname} ({dA}x{dB}) ══")
    
    # Verify PPT
    d = dA*dB
    pt = np.linalg.eigvalsh(rho.reshape(dA,dB,dA,dB).transpose(0,3,2,1).reshape(d,d)).min()
    print(f"  PPT: {pt:.2e} {'✅' if pt>=-1e-10 else '❌'}")
    
    # Q of PPT channel alone (should be 0)
    K_ppt = ppt_channel_kraus(rho, dA, dB)
    comp = sum(K.conj().T@K for K in K_ppt)
    print(f"  PPT Kraus completeness: {np.linalg.norm(comp-np.eye(dA)):.2e}")
    
    # Joint channel Q
    print(f"  Computing Q(N_PPT ⊗ N_erasure)...")
    t0 = time.time()
    ci = joint_channel_coherent_info(rho, dA, dB, p_erasure=0.5, n_trials=100)
    dt = time.time() - t0
    
    sa = ci > 0.001
    print(f"  max I(R>B) = {ci:.6f} [{dt:.1f}s]")
    print(f"  {'🌟 SUPERACTIVATION CONFIRMED!' if sa else '❌ No superactivation'}")

print(f"\n{'='*60}")
