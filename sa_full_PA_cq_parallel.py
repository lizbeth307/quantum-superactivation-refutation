"""
sa_full_PA_cq_parallel.py — CQ-Optimized Seesaw for full P⊗A.
Uses 30 CPU Cores for maximum performance.
"""
import numpy as np, time, sys
import multiprocessing as mp
sys.path.insert(0, '.')
from sa_engine import S

def build_P(a_param=0.5):
    """PPT channel P: C²→C⁴ from Horodecki state."""
    a = a_param; b = (1+a)/2; c = np.sqrt(max(1-a*a,0))/2
    rho = np.zeros((8,8), dtype=complex)
    rho[0,0]=a; rho[4,4]=a; rho[1,1]=a; rho[5,5]=a
    rho[2,2]=a; rho[6,6]=b; rho[3,3]=b; rho[7,7]=a
    rho[3,6]=c; rho[6,3]=c; rho[0,7]=a; rho[7,0]=a
    rho /= np.trace(rho); rho = (rho+rho.conj().T)/2
    ev, evec = np.linalg.eigh(rho)
    return [np.sqrt(2*ev[k]) * evec[:,k].reshape(2,4).T for k in range(len(ev)) if ev[k]>1e-14]

def build_A(p=0.5):
    """50% erasure: C²→C³."""
    K0 = np.zeros((3,2), dtype=complex)
    K0[0,0]=np.sqrt(1-p); K0[1,1]=np.sqrt(1-p)
    K1 = np.zeros((3,2), dtype=complex); K1[2,0]=np.sqrt(p)
    K2 = np.zeros((3,2), dtype=complex); K2[2,1]=np.sqrt(p)
    return [K0, K1, K2]

# ═══════════════════════════════════════════
#  PARALLEL SEESAW (ISOMETRIC)
# ═══════════════════════════════════════════
def _seesaw_iso_worker(args):
    idx, Ks, d_in, d_R, n_iter, seed = args
    np.random.seed(seed)
    d_out = Ks[0].shape[0]
    
    # Random isometric encoder
    V = np.random.randn(d_in, d_R) + 1j*np.random.randn(d_in, d_R)
    U, _, Vh = np.linalg.svd(V, full_matrices=False)
    V = U @ Vh
    
    prev_F = -1
    best_F = 0
    for it in range(n_iter):
        sigma = np.zeros((d_R*d_out, d_R*d_out), dtype=complex)
        for K in Ks:
            KV = K @ V
            for i in range(d_R):
                for j in range(d_R):
                    sigma[i*d_out:(i+1)*d_out, j*d_out:(j+1)*d_out] += \
                        np.outer(KV[:,i], KV[:,j].conj()) / d_R
        
        evals, evecs = np.linalg.eigh(sigma)
        w = evecs[:, -1]
        W = w.reshape(d_R, d_out)
        Uw, _, Vhw = np.linalg.svd(W, full_matrices=False)
        W = Uw @ Vhw
        
        F = sum((W @ sigma[i*d_out:(i+1)*d_out, j*d_out:(j+1)*d_out] @ W.conj().T)[i,j] 
                for i in range(d_R) for j in range(d_R)).real / d_R
        
        TK = [W @ K for K in Ks]
        M = np.zeros((d_in*d_R, d_in*d_R), dtype=complex)
        for T in TK:
            t = T.T.ravel('F')
            M += np.outer(t, t.conj())
            
        evals_v, evecs_v = np.linalg.eigh(M)
        v = evecs_v[:, -1]
        V_new = v.reshape(d_in, d_R, order='F')
        Uv, _, Vhv = np.linalg.svd(V_new, full_matrices=False)
        V = Uv @ Vhv
        
        if abs(F - prev_F) < 1e-8: break
        prev_F = F
        best_F = max(best_F, F)
        
    return best_F

def seesaw_iso_parallel(Ks, d_in, d_R=2, n_iter=30, n_restarts=60, n_jobs=30):
    args_list = [(i, Ks, d_in, d_R, n_iter, int(time.time()*1000)%1000000 + i) 
                 for i in range(n_restarts)]
    with mp.Pool(n_jobs) as pool:
        results = pool.map(_seesaw_iso_worker, args_list)
    return max(results)

# ═══════════════════════════════════════════
#  PARALLEL COHERENT INFO Q1
# ═══════════════════════════════════════════
def _q1_worker(args):
    idx, Ks, d_in, n_trials, seed = args
    np.random.seed(seed)
    d_out = Ks[0].shape[0]
    best = -999
    
    for t in range(n_trials):
        if idx == 0 and t == 0:
            psi = np.zeros(d_in*d_in, dtype=complex)
            for i in range(d_in): psi[i*d_in+i] = 1/np.sqrt(d_in)
        else:
            psi = np.random.randn(d_in*d_in)+1j*np.random.randn(d_in*d_in)
            psi /= np.linalg.norm(psi)
            
        rho_RA = np.outer(psi, psi.conj())
        rho_RB = np.zeros((d_in*d_out, d_in*d_out), dtype=complex)
        for r1 in range(d_in):
            for r2 in range(d_in):
                bl = rho_RA[r1*d_in:(r1+1)*d_in, r2*d_in:(r2+1)*d_in]
                rho_RB[r1*d_out:(r1+1)*d_out, r2*d_out:(r2+1)*d_out] = \
                    sum(K@bl@K.conj().T for K in Ks)
        rho_B = sum(rho_RB[r*d_out:(r+1)*d_out, r*d_out:(r+1)*d_out] for r in range(d_in))
        best = max(best, S(rho_B) - S(rho_RB))
        
    return best

def coherent_info_q1_parallel(Ks, d_in, total_trials=900, n_jobs=30):
    trials_per_worker = total_trials // n_jobs
    args_list = [(i, Ks, d_in, trials_per_worker, int(time.time()*1000)%1000000 + i) 
                 for i in range(n_jobs)]
    with mp.Pool(n_jobs) as pool:
        results = pool.map(_q1_worker, args_list)
    return max(results)

# ═══════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════
if __name__ == '__main__':
    print("="*60)
    print("  FULL P⊗A — 30-Core Parallel Seesaw & Q1")
    print("="*60)
    
    # Let's fix a=0.7 which we found was best
    a_best = 0.7
    print(f"\n  Testing P(a={a_best}) ⊗ A(0.5)")
    
    Ks_P = build_P(a_best)
    Ks_A = build_A(0.5)
    Ks_PA = [np.kron(Kp, Ka) for Kp in Ks_P for Ka in Ks_A]
    
    t0 = time.time()
    F1 = seesaw_iso_parallel(Ks_PA, 4, d_R=2, n_iter=40, n_restarts=60)
    q1_PA = coherent_info_q1_parallel(Ks_PA, 4, total_trials=900)
    print(f"  n=1 (P⊗A) : F₁ = {F1:.4f}, Q₁ = {q1_PA:+.4f} [{time.time()-t0:.1f}s]")
    
    print(f"\n  Computing (P⊗A)^⊗2...")
    Ks_PA2 = [np.kron(K1, K2) for K1 in Ks_PA for K2 in Ks_PA]
    
    t0 = time.time()
    F2 = seesaw_iso_parallel(Ks_PA2, 16, d_R=2, n_iter=30, n_restarts=90)
    dt2 = time.time() - t0
    print(f"  F₂ = F_c((P⊗A)², 2) = {F2:.6f} [{dt2:.1f}s]")
    
    t0 = time.time()
    q1_PA2 = coherent_info_q1_parallel(Ks_PA2, 16, total_trials=600)
    dt_q2 = time.time() - t0
    print(f"  Q₁((P⊗A)²) = {q1_PA2:+.6f} [{dt_q2:.1f}s]")
    
    # Test n=3 if possible
    print(f"\n  Computing (P⊗A)^⊗3...")
    Ks_PA3 = [np.kron(K1, K2) for K1 in Ks_PA2 for K2 in Ks_PA]
    print(f"  (P⊗A)³: {len(Ks_PA3)} Kraus, C^64 → C^1728")
    t0 = time.time()
    # Less restarts for n=3, just testing if it runs
    F3 = seesaw_iso_parallel(Ks_PA3, 64, d_R=2, n_iter=15, n_restarts=30)
    print(f"  F₃ = F_c((P⊗A)³, 2) = {F3:.6f} [{time.time()-t0:.1f}s]")
    
    print(f"\n{'='*60}")
