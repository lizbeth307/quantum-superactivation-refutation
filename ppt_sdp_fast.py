"""
ppt_sdp_fast.py — Fast PPT entangled state search via SDP.
Uses permutation matrix for PT constraint (not element-wise).
30 cores parallel.
"""
import numpy as np
import cvxpy as cp
from multiprocessing import Pool
import time


def pt_permutation(d):
    """Build permutation matrix P such that vec(rho^TB) = P @ vec(rho).
    For d x d bipartite system, PT swaps Bob's indices.
    rho^TB[iA*d+iB, jA*d+jB] = rho[iA*d+jB, jA*d+iB]
    """
    n = d * d
    P = np.zeros((n, n), dtype=int)
    for iA in range(d):
        for iB in range(d):
            row = iA * d + iB
            new_row = iA * d + iB  # same
            # But we need to express rho_PT as P @ rho @ P.T
            # rho_PT[a,b] = rho[perm(a), perm(b)] where perm swaps Bob
            # perm(iA*d + iB) = iA*d + iB (identity for Alice)
            # NO: PT is NOT a simple permutation of rows/cols
            # PT[iA*d+iB, jA*d+jB] = rho[iA*d+jB, jA*d+iB]
            pass
    
    # Actually: build swap operator on Bob's space
    # S_B[iB, jB] swaps iB <-> jB: S_B = identity (trivial)
    # PT = (I_A ⊗ T_B) applied to rho viewed as (A⊗B) x (A⊗B)
    # rho_PT = (I_A ⊗ T_B) rho (I_A ⊗ T_B)  where T_B is transpose on B
    # But T_B is NOT unitary for complex matrices
    
    # Correct approach: build the linear map explicitly
    # rho_PT[i,j] = rho[f(i,j)] where f swaps Bob indices
    # i = iA*d+iB, j = jA*d+jB
    # f: (iA*d+iB, jA*d+jB) -> (iA*d+jB, jA*d+iB)
    
    # For CVXPY: express rho_PT as a matrix obtained by 
    # reshaping and permuting indices
    return None  # not simple


def find_ppt_entangled(dk, ds, seed=0):
    """Find PPT entangled state using SDP with clever formulation."""
    d = dk * ds
    n = d * d
    rng = np.random.RandomState(seed)
    
    # Use a different approach: parametrize rho directly and 
    # check PPT numerically after solving
    # 
    # SDP: maximize Tr(W @ rho) subject to rho >= 0, Tr(rho) = 1
    # where W is an entanglement witness
    #
    # Then check if the optimal rho is PPT
    # If PPT + entangled -> SA candidate!
    
    # Better approach for finding PPT entangled:
    # Known construction: use the Choi map for d=3
    # For d=3x3: the Choi witness detects BE states
    
    # Simplest approach that works: 
    # Generate random PPT states and check entanglement
    
    best = None
    
    for trial in range(200):
        # Generate random state via partial transpose trick:
        # Start with random state, project onto PPT set
        
        # Random Wishart matrix -> density matrix
        G = rng.randn(n, n) + 1j * rng.randn(n, n)
        rho = G @ G.conj().T
        rho /= np.trace(rho)
        
        # Project onto PPT: 
        # rho_PT = partial_transpose(rho)
        # If not PSD, fix eigenvalues
        rho_pt = rho.reshape(d,d,d,d).transpose(0,3,2,1).reshape(n,n)
        eigs_pt, vecs_pt = np.linalg.eigh(rho_pt)
        
        if np.min(eigs_pt) < 0:
            # Clip negative eigenvalues
            eigs_pt = np.maximum(eigs_pt, 0)
            rho_pt_fixed = vecs_pt @ np.diag(eigs_pt) @ vecs_pt.conj().T
            rho_pt_fixed /= np.trace(rho_pt_fixed)
            
            # Undo PT to get the state
            rho_ppt = rho_pt_fixed.reshape(d,d,d,d).transpose(0,3,2,1).reshape(n,n)
            rho_ppt = (rho_ppt + rho_ppt.conj().T) / 2
            
            # Check if PSD
            eigs = np.linalg.eigvalsh(rho_ppt)
            if np.min(eigs) < -1e-8:
                continue
            
            rho_ppt = np.maximum(rho_ppt, 0)  # won't help for complex
            # Re-check
            eigs = np.linalg.eigvalsh(rho_ppt)
            if np.min(eigs) < -1e-8:
                continue
            
            rho_ppt /= np.trace(rho_ppt)
            rho = rho_ppt
        
        # Verify PPT
        rho_pt2 = rho.reshape(d,d,d,d).transpose(0,3,2,1).reshape(n,n)
        pt_min = np.min(np.linalg.eigvalsh(rho_pt2))
        
        if pt_min < -1e-8:
            continue
        
        # Check entanglement via realignment
        R = rho.reshape(d,d,d,d).transpose(0,2,1,3).reshape(n,n)
        realign = np.linalg.norm(R, 'nuc')  # nuclear norm
        
        rank = np.sum(np.linalg.eigvalsh(rho) > 1e-8)
        
        if realign > 1.001:
            # PPT entangled!
            if best is None or realign > best['realign']:
                best = {
                    'dk': dk, 'ds': ds, 'd': d,
                    'pt_min': pt_min, 'rank': rank,
                    'realign': realign,
                    'rho': rho.copy(), 'seed': seed,
                    'trial': trial,
                }
    
    return best


def worker(args):
    return find_ppt_entangled(*args)


if __name__ == '__main__':
    print("Fast PPT Entangled State Search (random + projection)")
    
    tasks = []
    for dk, ds in [(2,3), (3,2), (2,4), (4,2), (3,3), (2,5), (5,2), (2,6)]:
        for seed in range(30):
            tasks.append((dk, ds, seed))
    
    print(f"Tasks: {len(tasks)}, Workers: 30")
    print(f"Each: 200 random states -> project to PPT -> check entanglement")
    print(f"Total: {len(tasks) * 200} states\n")
    
    t0 = time.time()
    with Pool(30) as pool:
        results = pool.map(worker, tasks)
    elapsed = time.time() - t0
    
    print(f"Done in {elapsed:.1f}s\n")
    
    # Collect by (dk, ds)
    from collections import defaultdict
    by_d = defaultdict(list)
    for r in results:
        if r is not None:
            by_d[(r['dk'], r['ds'])].append(r)
    
    print(f"{'d':>3} {'dk':>3} {'ds':>3} {'found':>6} {'best_realign':>12} {'rank':>5} {'pt_min':>10}")
    print("-" * 55)
    
    for dk, ds in [(2,3), (3,2), (2,4), (4,2), (3,3), (2,5), (5,2), (2,6)]:
        d = dk * ds
        hits = by_d.get((dk, ds), [])
        if hits:
            best = max(hits, key=lambda x: x['realign'])
            print(f"{d:3d} {dk:3d} {ds:3d} {len(hits):6d} {best['realign']:12.4f} {best['rank']:5d} {best['pt_min']:10.6f}")
            if best['realign'] > 1.001:
                np.savez(f'sa_data/ppt_entangled_d{d}_dk{dk}.npz',
                         rho=best['rho'], dk=dk, ds=ds)
                print(f"    -> SAVED sa_data/ppt_entangled_d{d}_dk{dk}.npz")
        else:
            print(f"{d:3d} {dk:3d} {ds:3d}      0           -     -          -")
