"""
sdp_mosek.py — SDP seesaw with MOSEK solver (reliable, interior-point).

Compares SCS vs MOSEK vs CLARABEL on the same problem.
Resolves whether F_D=1.0 was a numerical artifact or real.
"""
import numpy as np, cvxpy as cp, time, sys
sys.path.insert(0, '.')
from sa_engine import S, I2, sX, sZ, build_effective_channel
from sdp_seesaw import build_Nk_choi
from scipy.special import comb

def sdp_decoder(Ks, V, d_R=2, solver='MOSEK'):
    """SDP decoder optimization with specified solver."""
    d_in = V.shape[0]
    d_out = Ks[0].shape[0]
    
    # σ_RB
    sigma = np.zeros((d_R*d_out, d_R*d_out), dtype=complex)
    for K in Ks:
        KV = K @ V
        for i in range(d_R):
            for j in range(d_R):
                sigma[i*d_out:(i+1)*d_out, j*d_out:(j+1)*d_out] += \
                    np.outer(KV[:,i], KV[:,j].conj()) / d_R
    
    # Reorder σ: R⊗B → B⊗R
    dim = d_out * d_R
    sigma_BR = np.zeros((dim, dim), dtype=complex)
    for i in range(d_R):
        for j in range(d_R):
            for a in range(d_out):
                for b in range(d_out):
                    sigma_BR[a*d_R+i, b*d_R+j] = sigma[i*d_out+a, j*d_out+b]
    
    J = cp.Variable((dim, dim), hermitian=True)
    obj = cp.Maximize(cp.real(cp.trace(J @ sigma_BR)) / d_R)
    constraints = [J >> 0]
    for a in range(d_out):
        for b in range(d_out):
            val = sum(J[a*d_R+r, b*d_R+r] for r in range(d_R))
            constraints.append(val == (1 if a == b else 0))
    
    prob = cp.Problem(obj, constraints)
    try:
        prob.solve(solver=solver, verbose=False)
        return prob.value if prob.value is not None else -999
    except Exception as e:
        return -999

def isometric_fidelity(Ks, V, d_R=2):
    """Isometric decoder fidelity (ground truth)."""
    d_out = Ks[0].shape[0]
    sigma = np.zeros((d_R*d_out, d_R*d_out), dtype=complex)
    for K in Ks:
        KV = K @ V
        for i in range(d_R):
            for j in range(d_R):
                sigma[i*d_out:(i+1)*d_out, j*d_out:(j+1)*d_out] += \
                    np.outer(KV[:,i], KV[:,j].conj()) / d_R
    evals = np.linalg.eigvalsh(sigma)
    return evals[-1]  # max eigenvalue = F for isometric decoder

def seesaw_full(Ks, d_in, d_R=2, solver='MOSEK', n_iter=15, n_restarts=8):
    """Full seesaw: alternate SDP decoder + isometric encoder."""
    d_out = Ks[0].shape[0]
    best_F = 0
    
    for restart in range(n_restarts):
        V = np.random.randn(d_in, d_R) + 1j*np.random.randn(d_in, d_R)
        U, s, Vh = np.linalg.svd(V, full_matrices=False); V = U@Vh
        
        for it in range(n_iter):
            # SDP decoder
            F_sdp = sdp_decoder(Ks, V, d_R, solver)
            
            # Isometric encoder update (via σ_RB eigenvector → decoder → M matrix)
            sigma = np.zeros((d_R*d_out, d_R*d_out), dtype=complex)
            for K in Ks:
                KV = K @ V
                for i in range(d_R):
                    for j in range(d_R):
                        sigma[i*d_out:(i+1)*d_out, j*d_out:(j+1)*d_out] += \
                            np.outer(KV[:,i], KV[:,j].conj()) / d_R
            
            ev, evc = np.linalg.eigh(sigma)
            w = evc[:, -1]; W = w.reshape(d_R, d_out)
            Uw, sw, Vhw = np.linalg.svd(W, full_matrices=False); W = Uw@Vhw
            
            TK = [W @ K for K in Ks]
            M = np.zeros((d_in*d_R, d_in*d_R), dtype=complex)
            for T in TK:
                t = T.T.ravel('F'); M += np.outer(t, t.conj())
            ev2, evc2 = np.linalg.eigh(M)
            v = evc2[:, -1]; V = v.reshape(d_in, d_R, order='F')
            Uv, sv, Vhv = np.linalg.svd(V, full_matrices=False); V = Uv@Vhv
        
        best_F = max(best_F, F_sdp)
    
    return best_F

if __name__ == '__main__':
    print("="*65)
    print("  SDP SEESAW — MOSEK vs SCS Comparison")
    print("="*65)
    
    # === Part 1: Solver comparison on n=2, k=1 ===
    print("\n  Part 1: Solver comparison on Ñ^⊗2, k=1")
    n, k = 2, 1
    Ks = build_Nk_choi(n, k)
    
    # Use same optimized encoder for fair comparison
    np.random.seed(42)
    V = np.random.randn(4, 2) + 1j*np.random.randn(4, 2)
    U, s, Vh = np.linalg.svd(V, full_matrices=False); V = U@Vh
    
    # Run isometric seesaw first to get good encoder
    for it in range(30):
        d = 4; d_R = 2; d_out = 4
        sigma = np.zeros((d_R*d_out, d_R*d_out), dtype=complex)
        for K in Ks:
            KV = K@V
            for i in range(d_R):
                for j in range(d_R):
                    sigma[i*d_out:(i+1)*d_out, j*d_out:(j+1)*d_out] += np.outer(KV[:,i],KV[:,j].conj())/d_R
        ev,evc = np.linalg.eigh(sigma); w=evc[:,-1]
        W=w.reshape(d_R,d_out); Uw,sw,Vhw=np.linalg.svd(W,full_matrices=False); W=Uw@Vhw
        TK=[W@K for K in Ks]
        M=np.zeros((d*d_R,d*d_R),dtype=complex)
        for T in TK: t=T.T.ravel('F'); M+=np.outer(t,t.conj())
        ev2,evc2=np.linalg.eigh(M); v=evc2[:,-1]
        V=v.reshape(d,d_R,order='F')
        Uv,sv,Vhv=np.linalg.svd(V,full_matrices=False); V=Uv@Vhv
    
    F_iso = isometric_fidelity(Ks, V, d_R)
    print(f"  Isometric F = {F_iso:.6f}")
    
    for solver in ['SCS', 'MOSEK', 'CLARABEL']:
        t0 = time.time()
        F = sdp_decoder(Ks, V, d_R, solver)
        dt = time.time() - t0
        gap = F - F_iso
        print(f"  {solver:>8s}: F = {F:.6f}  gap={gap:+.6f}  [{dt:.2f}s]")
    
    # === Part 2: Full seesaw with MOSEK for n=1..4 ===
    print("\n  Part 2: Full MOSEK seesaw on Ñ^⊗n")
    print(f"  {'n':>3} {'k':>3} {'P(k)':>7} {'F_D(M)':>8} {'F_D(iso)':>8} {'contrib':>8}")
    print(f"  {'-'*50}")
    
    for n in range(1, 5):
        d = 2**n
        if d > 16: break
        F_total_M = 0; F_total_I = 0
        
        for k in range(n+1):
            pk = comb(n, k, exact=True) / (2**n)
            Ks = build_Nk_choi(n, k)
            
            # MOSEK seesaw
            F_M = seesaw_full(Ks, d, d_R=2, solver='MOSEK', n_iter=10, n_restarts=5)
            
            # Isometric seesaw
            best_iso = 0
            for restart in range(10):
                np.random.seed(restart*100+k+n*1000)
                V = np.random.randn(d,2)+1j*np.random.randn(d,2)
                U,s,Vh = np.linalg.svd(V, full_matrices=False); V=U@Vh
                for it in range(25):
                    sig = np.zeros((2*d,2*d), dtype=complex)
                    for K in Ks:
                        KV=K@V
                        for i in range(2):
                            for j in range(2):
                                sig[i*d:(i+1)*d,j*d:(j+1)*d]+=np.outer(KV[:,i],KV[:,j].conj())/2
                    ev,evc=np.linalg.eigh(sig); w=evc[:,-1]
                    W=w.reshape(2,d); Uw,sw,Vhw=np.linalg.svd(W,full_matrices=False); W=Uw@Vhw
                    F=sum((W@sig[i*d:(i+1)*d,j*d:(j+1)*d]@W.conj().T)[i,j] for i in range(2) for j in range(2)).real/2
                    TK=[W@K for K in Ks]; M=np.zeros((d*2,d*2),dtype=complex)
                    for T in TK: t=T.T.ravel('F'); M+=np.outer(t,t.conj())
                    ev2,evc2=np.linalg.eigh(M); v=evc2[:,-1]
                    V=v.reshape(d,2,order='F')
                    Uv,sv,Vhv=np.linalg.svd(V,full_matrices=False); V=Uv@Vhv
                best_iso = max(best_iso, F)
            
            F_total_M += pk * F_M
            F_total_I += pk * best_iso
            print(f"  {n:3d} {k:3d} {pk:7.4f} {F_M:8.4f} {best_iso:8.4f} {pk*F_M:8.4f}")
        
        sa_M = " 🌟" if F_total_M > 0.75 else ""
        sa_I = " 🌟" if F_total_I > 0.75 else ""
        print(f"  → n={n}: F_MOSEK={F_total_M:.6f}{sa_M}  F_iso={F_total_I:.6f}{sa_I}")
    
    print(f"\n{'='*65}")
