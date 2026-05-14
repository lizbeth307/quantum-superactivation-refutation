"""
sa_horodecki.py — Test Smith-Yard SA with the actual Horodecki 3x3 PPT state.
This is the state used (or similar to) the original Smith-Yard construction.
"""
import numpy as np, time

def S(rho):
    e = np.linalg.eigvalsh(rho); e = e[e>1e-15]
    return -np.sum(e*np.log2(e)) if len(e)>0 else 0.0

def horodecki_3x3(a):
    """Horodecki 1997 PPT entangled state on C^3⊗C^3."""
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

def choi_to_kraus(rho_choi, dA, dB):
    """Extract TP Kraus operators from Choi state."""
    C = rho_choi * dA
    ev, U = np.linalg.eigh(C)
    Ks = [np.sqrt(max(ev[k],0))*U[:,k].reshape(dA,dB).T 
          for k in range(len(ev)) if ev[k]>1e-14]
    Sm = sum(K.conj().T@K for K in Ks)
    e2, U2 = np.linalg.eigh(Sm); e2 = np.maximum(e2, 1e-15)
    fix = U2 @ np.diag(1/np.sqrt(e2)) @ U2.conj().T
    return [K @ fix for K in Ks]

def erasure_kraus(d, p=0.5):
    K = []; K0 = np.zeros((d+1,d),dtype=complex)
    K0[:d,:d] = np.sqrt(1-p)*np.eye(d); K.append(K0)
    for i in range(d):
        Ki = np.zeros((d+1,d),dtype=complex); Ki[d,i]=np.sqrt(p); K.append(Ki)
    return K

def test_SA(K_ppt, K_era, dA_in, dB_out, d_era_out, n_trials=200):
    """Q(N_PPT ⊗ N_erasure) via coherent information optimization."""
    d_in = dA_in * dA_in  # two copies of input
    d_out = dB_out * d_era_out
    
    np.random.seed(42)
    best = -999
    
    for t in range(n_trials):
        # Input state on reference ⊗ channel_input
        if t == 0:
            # Maximally entangled
            psi = np.zeros(d_in*d_in, dtype=complex)
            for i in range(d_in): psi[i*d_in+i]=1/np.sqrt(d_in)
        elif t < 20:
            # Structured: |ψ⟩ = Σ c_i |i⟩|i⟩
            c = np.random.randn(d_in) + 1j*np.random.randn(d_in)
            c /= np.linalg.norm(c)
            psi = np.zeros(d_in*d_in, dtype=complex)
            for i in range(d_in): psi[i*d_in+i] = c[i]
        else:
            psi = np.random.randn(d_in*d_in)+1j*np.random.randn(d_in*d_in)
            psi /= np.linalg.norm(psi)
        
        rho_in = np.outer(psi, psi.conj())
        rho_out = np.zeros((d_in*d_out, d_in*d_out), dtype=complex)
        
        for r1 in range(d_in):
            for r2 in range(d_in):
                b = rho_in[r1*d_in:(r1+1)*d_in, r2*d_in:(r2+1)*d_in]
                ob = np.zeros((d_out,d_out), dtype=complex)
                for K1 in K_ppt:
                    for K2 in K_era:
                        Kj = np.kron(K1, K2)
                        ob += Kj @ b @ Kj.conj().T
                rho_out[r1*d_out:(r1+1)*d_out, r2*d_out:(r2+1)*d_out] = ob
        
        rB = sum(rho_out[r*d_out:(r+1)*d_out, r*d_out:(r+1)*d_out] for r in range(d_in))
        ci = S(rB) - S(rho_out)
        best = max(best, ci)
    
    return best

print("="*60)
print("  SUPERACTIVATION: Horodecki 3x3 + 50% Erasure")
print("="*60)

for a in [0.1, 0.2, 0.3, 0.5]:
    rho = horodecki_3x3(a)
    dA = 3; dB = 3
    
    K_ppt = choi_to_kraus(rho, dA, dB)
    tp_err = np.linalg.norm(sum(K.conj().T@K for K in K_ppt) - np.eye(dA))
    
    K_era = erasure_kraus(dA, 0.5)
    d_era = dA + 1  # = 4
    
    print(f"\n  a = {a}: TP err = {tp_err:.2e}")
    t0 = time.time()
    ci = test_SA(K_ppt, K_era, dA, dB, d_era, n_trials=200)
    dt = time.time() - t0
    
    print(f"  Q(N_PPT ⊗ N_erasure) = {ci:.6f}  [{dt:.1f}s]")
    if ci > 0.001:
        print(f"  🌟 SUPERACTIVATION FOUND!")
    else:
        print(f"  ❌ No SA (coherent info ≤ 0)")

print(f"\n{'='*60}")
