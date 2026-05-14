"""verify_sdp.py — Verify SDP PPT candidates for d=8,10"""
import numpy as np

def von_neumann(rho):
    e = np.linalg.eigvalsh(rho)
    e = e[e>1e-15]
    return -np.sum(e*np.log2(e)) if len(e)>0 else 0.0

def kdw_bipartite(rho, dA, dB, n_bases=200):
    best_ixb = -999.0
    for trial in range(n_bases):
        if trial == 0: U = np.eye(dA, dtype=complex)
        else:
            H = np.random.randn(dA,dA)+1j*np.random.randn(dA,dA)
            U, _ = np.linalg.qr(H)
        p_x = np.zeros(dA); S_B_x = np.zeros(dA)
        for x in range(dA):
            rB = np.zeros((dB,dB), dtype=complex)
            for a1 in range(dA):
                for a2 in range(dA):
                    c = U[a1,x].conj()*U[a2,x]
                    rB += c * rho[a1*dB:(a1+1)*dB, a2*dB:(a2+1)*dB]
            p_x[x] = np.trace(rB).real
            if p_x[x]>1e-15:
                rB /= p_x[x]; S_B_x[x] = von_neumann(rB)
        rBu = sum(rho[a*dB:(a+1)*dB, a*dB:(a+1)*dB] for a in range(dA))
        S_B = von_neumann(rBu)
        I_XB = S_B - sum(p_x[x]*S_B_x[x] for x in range(dA) if p_x[x]>1e-15)
        best_ixb = max(best_ixb, I_XB)
    
    rBu = sum(rho[a*dB:(a+1)*dB, a*dB:(a+1)*dB] for a in range(dA))
    I_coh = von_neumann(rBu) - von_neumann(rho)
    return best_ixb, I_coh

print("SDP CANDIDATE VERIFICATION")
for dA, dB in [(2,4), (2,5)]:
    d = dA*dB
    rho = np.load(f'sa_data/sdp_ppt_{dA}x{dB}.npz')['rho']
    pt = rho.reshape(dA,dB,dA,dB).transpose(0,3,2,1).reshape(d,d)
    pt_min = np.min(np.linalg.eigvalsh(pt))
    R = rho.reshape(dA,dB,dA,dB).transpose(0,2,1,3).reshape(d,d)
    realign = np.linalg.norm(R, 'nuc')
    I_XB, I_coh = kdw_bipartite(rho, dA, dB, 200)
    
    ppt_ok = "PASS" if pt_min >= -1e-5 else "FAIL"
    ent_ok = "PASS" if realign > 1.001 else "FAIL"
    
    print(f"\n  d={d} ({dA}x{dB}):")
    print(f"    PPT: {ppt_ok} (pt_min={pt_min:.2e})")
    print(f"    ENT: {ent_ok} (realign={realign:.4f})")
    print(f"    I(X;B) = {I_XB:.6f}")
    print(f"    I_coh = {I_coh:.6f}")
    
    if I_XB > 0.001:
        print(f"    >>> I(X;B) > 0: SA CANDIDATE (pending I(X;E) check)")
