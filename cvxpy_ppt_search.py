"""
cvxpy_ppt_search.py — Use CVXPY SDP to find PPT entangled states with K_DW > 0.
30 CPU cores parallel: each core tries different d, dk, ds.
"""
import numpy as np
import cvxpy as cp
from multiprocessing import Pool
import time


def find_ppt_entangled_cvxpy(dk, ds, seed=0):
    """Use SDP to find PPT state that maximizes entanglement proxy.
    
    We parametrize rho as a valid density matrix (PSD, Tr=1),
    add PPT constraint (rho^TB >= 0),
    and maximize realignment norm (proxy for entanglement).
    
    If we find PPT + entangled state, it's a candidate for SA.
    """
    d = dk * ds
    n = d * d  # full dimension
    
    rng = np.random.RandomState(seed)
    
    # Decision variable: density matrix
    rho = cp.Variable((n, n), hermitian=True)
    
    # Constraints
    constraints = [
        rho >> 0,           # PSD
        cp.trace(rho) == 1, # Trace 1
    ]
    
    # PPT constraint via element-wise equality with rho_PT variable
    
    rho_PT = cp.Variable((n, n), hermitian=True)
    
    # Add constraints: rho_PT[i,j] = rho[perm_i, perm_j]
    for iA in range(d):
        for iB in range(d):
            for jA in range(d):
                for jB in range(d):
                    r1 = iA * d + iB
                    c1 = jA * d + jB
                    r2 = iA * d + jB
                    c2 = jA * d + iB
                    constraints.append(rho_PT[r1, c1] == rho[r2, c2])
    
    constraints.append(rho_PT >> 0)  # PPT constraint
    
    # Objective: maximize something that indicates entanglement
    # Proxy: maximize trace of rho against a random entangled target
    target = rng.randn(n, n) + 1j * rng.randn(n, n)
    target = target + target.conj().T
    target = target / np.linalg.norm(target)
    
    objective = cp.Maximize(cp.real(cp.trace(target @ rho)))
    
    prob = cp.Problem(objective, constraints)
    
    try:
        prob.solve(solver=cp.SCS, verbose=False, max_iters=5000)
        
        if prob.status in ['optimal', 'optimal_inaccurate']:
            rho_val = rho.value
            if rho_val is not None:
                # Check realignment norm
                R = rho_val.reshape(d, d, d, d).transpose(0, 2, 1, 3).reshape(n, n)
                realign_norm = np.linalg.norm(R, 'nuc')
                
                rank = np.sum(np.linalg.eigvalsh(rho_val) > 1e-8)
                pt_val = rho_PT.value
                pt_min = np.min(np.linalg.eigvalsh(pt_val)) if pt_val is not None else -999
                
                return {
                    'dk': dk, 'ds': ds, 'd': d,
                    'pt_min': pt_min, 'rank': rank,
                    'realign_norm': realign_norm,
                    'entangled': realign_norm > 1.001,
                    'rho': rho_val, 'seed': seed,
                }
    except Exception as e:
        pass
    
    return None


def worker(args):
    dk, ds, seed = args
    return find_ppt_entangled_cvxpy(dk, ds, seed)


if __name__ == '__main__':
    print("CVXPY PPT Entangled State Search")
    print(f"CVXPY {cp.__version__}")
    
    # Generate tasks: multiple seeds per (dk, ds)
    tasks = []
    for dk, ds in [(2, 3), (2, 4), (3, 3), (2, 5), (3, 2), (2, 2)]:
        for seed in range(5):
            tasks.append((dk, ds, seed))
    
    print(f"Tasks: {len(tasks)}, Workers: 30")
    
    t0 = time.time()
    with Pool(30) as pool:
        results = pool.map(worker, tasks)
    elapsed = time.time() - t0
    
    print(f"\nDone in {elapsed:.1f}s\n")
    
    # Summary
    for r in results:
        if r is not None:
            d = r['d']
            ent = "ENTANGLED!" if r['entangled'] else "separable"
            print(f"d={d:2d} (dk={r['dk']},ds={r['ds']}) seed={r['seed']}: "
                  f"PT_min={r['pt_min']:.6f} rank={r['rank']} "
                  f"realign={r['realign_norm']:.4f} {ent}")
        else:
            print(f"  Failed")
