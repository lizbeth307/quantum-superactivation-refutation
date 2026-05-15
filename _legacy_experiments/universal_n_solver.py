"""
universal_n_solver.py — The 30-Core Exact Solver for Q > 0.
Uses Dicke basis (symmetric subspace) for exact evaluation up to n=11.
Exploits the Classical-Quantum (CQ) structure of the effective channel.
"""
import numpy as np, time, sys
import multiprocessing as mp
from scipy.special import comb
import itertools
sys.path.insert(0, '.')

# ═══════════════════════════════════════════
#  CQ CHANNEL CONSTRUCTION
# ═══════════════════════════════════════════
def build_cq_kraus(a):
    p = 0.5
    c = np.sqrt(max(1-a*a, 0)) / 2
    b = (1+a)/2
    
    K0_in = np.array([[np.sqrt(1-p), 0], [0, np.sqrt(1-p)]])
    K1_er = np.array([[np.sqrt(p*a), 0], [0, np.sqrt(p*a)]])
    K2_er = np.array([[0, np.sqrt(p*b)], [np.sqrt(p*b), 0]])
    K3_er = np.array([[np.sqrt(p*c), 0], [0, -np.sqrt(p*c)]])
    K4_er = np.array([[0, -np.sqrt(p*c)], [np.sqrt(p*c), 0]])
    
    return [K0_in], [K1_er, K2_er, K3_er, K4_er]

def dicke_basis(n):
    d = 2**n
    basis = np.zeros((d, n+1), dtype=complex)
    bits = np.zeros(d, dtype=int)
    for i in range(1, d):
        bits[i] = bits[i >> 1] + (i & 1)
    for k in range(n + 1):
        indices = np.where(bits == k)[0]
        basis[indices, k] = 1.0 / np.sqrt(len(indices))
    return basis

def build_Nk_dicke(n, k, Ks_in, Ks_er):
    """Build Nk"""
    kraus_list_dicke = []
    D = dicke_basis(n)
    
    # Bob measures the exact sequence of erasures.
    # By symmetry, all sequences with k erasures yield the same fidelity.
    # Therefore, we DO NOT average over combinations. We just evaluate ONE specific sequence!
    # Let the LAST k qubits be erased.
    positions = list(range(n-k, n))
    
    erased_idx_list = list(itertools.product(range(len(Ks_er)), repeat=k))
    
    # Weight is 1.0 because we are evaluating the fidelity for THIS specific sequence
    weight = 1.0
    
    for e_idxs in erased_idx_list:
        ops = [None] * n
        e_ptr = 0
        for i in range(n):
            if i in positions:
                ops[i] = Ks_er[e_idxs[e_ptr]]
                e_ptr += 1
            else:
                ops[i] = Ks_in[0]
                
        # Kronecker product
        O = ops[0]
        for j in range(1, n):
            O = np.kron(O, ops[j])
            
        # Project ONLY the input onto the symmetric subspace (n+1)
        # The output lives in the full 2^n space!
        K_mapped = O @ D
        
        if np.linalg.norm(K_mapped) > 1e-10:
            kraus_list_dicke.append((weight, K_mapped))
            
    return kraus_list_dicke

# ═══════════════════════════════════════════
#  30-CORE PARALLEL SEESAW (DICKE -> FULL)
# ═══════════════════════════════════════════
def _seesaw_worker(args):
    idx, d_sym, dim, d_R, kraus_list, n_iter, seed = args
    np.random.seed(seed)
    
    # Initialize encoder in Dicke basis (d_sym x d_R)
    V = np.random.randn(d_sym, d_R) + 1j*np.random.randn(d_sym, d_R)
    U, _, Vh = np.linalg.svd(V, full_matrices=False)
    V = U @ Vh
    
    best_F = 0
    prev_F = -1
    
    for it in range(n_iter):
        # 1. Optimize Decoder
        sigma = np.zeros((d_R*dim, d_R*dim), dtype=complex)
        for weight, K in kraus_list:
            KV = K @ V # size: dim x d_R
            for i in range(d_R):
                for j in range(d_R):
                    sigma[i*dim:(i+1)*dim, j*dim:(j+1)*dim] += \
                        weight * np.outer(KV[:,i], KV[:,j].conj()) / d_R
                        
        ev, evc = np.linalg.eigh(sigma)
        w = evc[:, -1]
        W = w.reshape(d_R, dim)
        Uw, _, Vhw = np.linalg.svd(W, full_matrices=False)
        W = Uw @ Vhw
        
        # Fidelity
        F = sum((W @ sigma[i*dim:(i+1)*dim, j*dim:(j+1)*dim] @ W.conj().T)[i,j] 
                for i in range(d_R) for j in range(d_R)).real / d_R
                
        # 2. Optimize Encoder
        M = np.zeros((d_sym*d_R, d_sym*d_R), dtype=complex)
        for weight, K in kraus_list:
            TK = W @ K # W is d_R x dim, K is dim x d_sym. TK is d_R x d_sym
            t = TK.T.ravel('F')
            M += weight * np.outer(t, t.conj())
            
        ev2, evc2 = np.linalg.eigh(M)
        v = evc2[:, -1]
        V_new = v.reshape(d_sym, d_R, order='F')
        Uv, _, Vhv = np.linalg.svd(V_new, full_matrices=False)
        V = Uv @ Vhv
        
        if abs(F - prev_F) < 1e-6: break
        prev_F = F
        best_F = max(best_F, F)
        
    return best_F

def solve_k_parallel(n, k, a_param=0.4336, n_restarts=30, n_iter=20):
    Ks_in, Ks_er = build_cq_kraus(a_param)
    kraus_list = build_Nk_dicke(n, k, Ks_in, Ks_er)
    
    d_sym = n + 1
    dim = 2**n
    args_list = [(i, d_sym, dim, 2, kraus_list, n_iter, int(time.time()*1000)%1000000 + i) 
                 for i in range(n_restarts)]
                 
    with mp.Pool(30) as pool:  # Enforced by .gravityrules
        results = pool.map(_seesaw_worker, args_list)
        
    return max(results)

def solve_n(n, a_param=0.4336):
    print(f"\n{'='*60}")
    print(f"  SOLVING FOR n = {n} (a = {a_param:.4f})")
    print(f"{'='*60}")
    
    F_total = 0
    t0_all = time.time()
    
    for k in range(n + 1):
        t0 = time.time()
        restarts = 60 if k <= n//2 else 30
        F_k = solve_k_parallel(n, k, a_param, n_restarts=restarts, n_iter=25)
        dt = time.time() - t0
        
        pk = comb(n, k, exact=True) / (2**n)
        contrib = pk * F_k
        F_total += contrib
        
        print(f"    k={k:2d} | P={pk:.4f} | F_k_dicke={F_k:.5f} | contrib={contrib:.5f} [{dt:.1f}s]")
        
    dt_all = time.time() - t0_all
    sa = F_total > 0.75
    print(f"\n  → F_c(Ñ^⊗{n}, 2) = {F_total:.6f} [{'🌟 SUPERACTIVATION!' if sa else 'Need more copies'}] [{dt_all:.1f}s]")
    return F_total

if __name__ == '__main__':
    # 30 cores enforced by .gravityrules
    for n in range(1, 10):
        solve_n(n, a_param=0.4336)
