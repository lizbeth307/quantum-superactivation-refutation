"""optimize_ppt.py — Optimize I(X;B) for SDP PPT states d=8,10 via perturbation."""
import numpy as np
from multiprocessing import Pool
import time

def von_neumann(rho):
    e = np.linalg.eigvalsh(rho)
    e = e[e>1e-15]
    return -np.sum(e*np.log2(e)) if len(e)>0 else 0.0

def ixb_bipartite(rho, dA, dB, n_bases=20):
    best = -999.0
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
                    rB += c*rho[a1*dB:(a1+1)*dB, a2*dB:(a2+1)*dB]
            p_x[x] = np.trace(rB).real
            if p_x[x]>1e-15:
                rB /= p_x[x]; S_B_x[x] = von_neumann(rB)
        rBu = sum(rho[a*dB:(a+1)*dB, a*dB:(a+1)*dB] for a in range(dA))
        S_B = von_neumann(rBu)
        I = S_B - sum(p_x[x]*S_B_x[x] for x in range(dA) if p_x[x]>1e-15)
        best = max(best, I)
    return best

def perturb_search(args):
    dA, dB, rho0, seed, n_trials = args
    rng = np.random.RandomState(seed)
    d = dA*dB
    best = None
    for _ in range(n_trials):
        eps = rng.uniform(0.01, 0.5)
        G = rng.randn(d,d)+1j*rng.randn(d,d)
        noise = G@G.conj().T; noise /= np.trace(noise)
        rho = (1-eps)*rho0 + eps*noise
        rho = (rho+rho.conj().T)/2; rho /= np.trace(rho).real
        
        pt = rho.reshape(dA,dB,dA,dB).transpose(0,3,2,1).reshape(d,d)
        pt_min = np.min(np.linalg.eigvalsh(pt))
        if pt_min < -1e-6: continue
        
        R = rho.reshape(dA,dB,dA,dB).transpose(0,2,1,3).reshape(d,d)
        realign = np.linalg.norm(R, 'nuc')
        if realign < 1.001: continue
        
        ixb = ixb_bipartite(rho, dA, dB, 20)
        if best is None or ixb > best[0]:
            best = (ixb, pt_min, realign, rho.copy())
    return best

if __name__ == '__main__':
    print("Optimizing I(X;B) for SDP PPT states, 30 cores")
    for dA, dB in [(2,4), (2,5)]:
        d = dA*dB
        rho0 = np.load(f'sa_data/sdp_ppt_{dA}x{dB}.npz')['rho']
        print(f'\n  {dA}x{dB} (d={d}):')
        args = [(dA, dB, rho0, s, 200) for s in range(30)]
        t0 = time.time()
        with Pool(30) as pool:
            results = pool.map(perturb_search, args)
        hits = [r for r in results if r is not None]
        if hits:
            best = max(hits, key=lambda x: x[0])
            ixb, pt, rl, rho = best
            print(f'    Best I(X;B)={ixb:.6f} realign={rl:.4f} pt={pt:.2e}')
            print(f'    ({time.time()-t0:.1f}s, {len(hits)} hits from 6000 trials)')
            np.savez(f'sa_data/optimized_ppt_{dA}x{dB}.npz', rho=rho)
        else:
            print(f'    No PPT+ENT perturbed states found ({time.time()-t0:.1f}s)')
