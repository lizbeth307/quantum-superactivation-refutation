"""
parallel_q1.py — 30-core optimization for Coherent Information Q1(N)
"""
import numpy as np, sys, time
import multiprocessing as mp
sys.path.insert(0, '.')
from sa_engine import S, partial_trace_A, partial_trace_B
from sa_full_PA_cq import build_P, build_A

def _worker_q1(args):
    idx, d_in, d_out, Ks, n_trials, seed_offset = args
    np.random.seed(seed_offset + idx)
    best = -999
    
    # We do a subset of trials in this worker
    for t in range(n_trials):
        if idx == 0 and t == 0:
            # First worker, first trial: maximally entangled state
            psi = np.zeros(d_in*d_in, dtype=complex)
            for i in range(d_in): psi[i*d_in+i] = 1/np.sqrt(d_in)
        else:
            psi = np.random.randn(d_in*d_in) + 1j*np.random.randn(d_in*d_in)
            psi /= np.linalg.norm(psi)
            
        rho_RA = np.outer(psi, psi.conj())
        rho_RB = np.zeros((d_in*d_out, d_in*d_out), dtype=complex)
        
        # Apply channel N = sum_k K @ rho @ K.dag
        for r1 in range(d_in):
            for r2 in range(d_in):
                bl = rho_RA[r1*d_in:(r1+1)*d_in, r2*d_in:(r2+1)*d_in]
                rho_RB[r1*d_out:(r1+1)*d_out, r2*d_out:(r2+1)*d_out] = \
                    sum(K @ bl @ K.conj().T for K in Ks)
                    
        rho_B = sum(rho_RB[r*d_out:(r+1)*d_out, r*d_out:(r+1)*d_out] for r in range(d_in))
        
        # Q1 for this pure state input
        q1 = S(rho_B) - S(rho_RB)
        if q1 > best: best = q1
        
    return best

def coherent_info_Q1_parallel(Ks, d_in, d_out, total_trials=3000, n_jobs=30):
    """Parallel computation of Q1 using 30 cores."""
    trials_per_worker = total_trials // n_jobs
    args_list = [(i, d_in, d_out, Ks, trials_per_worker, int(time.time()*1000)%1000000) 
                 for i in range(n_jobs)]
    
    with mp.Pool(n_jobs) as pool:
        results = pool.map(_worker_q1, args_list)
        
    return max(results)

if __name__ == '__main__':
    print('='*65)
    print('  CHANNEL Q1 ANALYSIS (30 Cores)')
    print('='*65)

    Ks_P = build_P(a_param=0.48)
    q1_P = coherent_info_Q1_parallel(Ks_P, 2, 4, total_trials=3000)
    print(f'Channel P (Horodecki): Q1={q1_P:+.4f}')

    Ks_A = build_A(p=0.5)
    q1_A = coherent_info_Q1_parallel(Ks_A, 2, 3, total_trials=3000)
    print(f'Channel A (Erasure):   Q1={q1_A:+.4f}')

    Ks_PA = [np.kron(Kp, Ka) for Kp in Ks_P for Ka in Ks_A]
    t0 = time.time()
    q1_PA = coherent_info_Q1_parallel(Ks_PA, 4, 12, total_trials=3000)
    dt = time.time() - t0
    print(f'Channel P⊗A:           Q1={q1_PA:+.4f} [{dt:.1f}s]')

    # n=2 (P⊗A ⊗ P⊗A)
    print(f'\n  Computing (P⊗A)⊗(P⊗A)... (16 -> 144 dimensions)')
    Ks_PA2 = [np.kron(K1, K2) for K1 in Ks_PA for K2 in Ks_PA]
    t0 = time.time()
    # Fewer trials for n=2 because the space is huge, but with 30 cores we can do 300 easily
    q1_PA2 = coherent_info_Q1_parallel(Ks_PA2, 16, 144, total_trials=300, n_jobs=30)
    dt = time.time() - t0
    print(f'Channel (P⊗A)²:        Q1={q1_PA2:+.4f} [{dt:.1f}s]')
    
    # Check effective channel for comparison
    from sa_engine import build_effective_channel
    Ks_Ntilde = build_effective_channel()
    q1_Ntilde = coherent_info_Q1_parallel(Ks_Ntilde, 2, 4, total_trials=3000)
    print(f'\nEffective Channel Ñ:   Q1={q1_Ntilde:+.4f}')
    
    print('='*65)
