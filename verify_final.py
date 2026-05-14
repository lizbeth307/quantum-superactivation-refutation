"""verify_final.py — Final PPT+I(X;B) verification for d=8,10 candidates."""
import numpy as np

def von_neumann(rho):
    e = np.linalg.eigvalsh(rho)
    e = e[e>1e-15]
    return -np.sum(e*np.log2(e)) if len(e)>0 else 0.0

print("FINAL VERIFICATION")
for dA, dB in [(2,4), (2,5)]:
    d = dA*dB
    rho = np.load(f'sa_data/optimized_ppt_{dA}x{dB}.npz')['rho']
    
    pt = rho.reshape(dA,dB,dA,dB).transpose(0,3,2,1).reshape(d,d)
    pt_min = np.min(np.linalg.eigvalsh(pt))
    psd_min = np.min(np.linalg.eigvalsh(rho))
    R = rho.reshape(dA,dB,dA,dB).transpose(0,2,1,3).reshape(d,d)
    realign = np.linalg.norm(R, 'nuc')
    
    print(f"\n  d={d} ({dA}x{dB}):")
    print(f"    PSD: min eig = {psd_min:.2e}")
    print(f"    PPT: pt_min = {pt_min:.6f} {'PASS' if pt_min >= -1e-5 else 'FAIL'}")
    print(f"    ENT: realign = {realign:.4f} {'PASS' if realign > 1.001 else 'FAIL'}")
    print(f"    Trace = {np.trace(rho).real:.8f}")
    
    best_ixb = -999.0
    for trial in range(500):
        if trial == 0:
            U = np.eye(dA, dtype=complex)
        else:
            H = np.random.randn(dA,dA)+1j*np.random.randn(dA,dA)
            U, _ = np.linalg.qr(H)
        p_x = np.zeros(dA)
        S_B_x = np.zeros(dA)
        for x in range(dA):
            rB = np.zeros((dB,dB), dtype=complex)
            for a1 in range(dA):
                for a2 in range(dA):
                    c = U[a1,x].conj()*U[a2,x]
                    rB += c*rho[a1*dB:(a1+1)*dB, a2*dB:(a2+1)*dB]
            p_x[x] = np.trace(rB).real
            if p_x[x] > 1e-15:
                rB /= p_x[x]
                S_B_x[x] = von_neumann(rB)
        rBu = sum(rho[a*dB:(a+1)*dB, a*dB:(a+1)*dB] for a in range(dA))
        S_B = von_neumann(rBu)
        I = S_B - sum(p_x[x]*S_B_x[x] for x in range(dA) if p_x[x]>1e-15)
        best_ixb = max(best_ixb, I)
    
    rBu = sum(rho[a*dB:(a+1)*dB, a*dB:(a+1)*dB] for a in range(dA))
    I_coh = von_neumann(rBu) - von_neumann(rho)
    
    print(f"    I(X;B) = {best_ixb:.6f}")
    print(f"    I_coh = {I_coh:.6f}")
    if best_ixb > 0.001:
        print(f"    >>> SA CANDIDATE CONFIRMED")
