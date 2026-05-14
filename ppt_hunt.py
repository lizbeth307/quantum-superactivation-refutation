"""
ppt_hunt.py — Hunt for PPT entangled states, dimension by dimension.
Sequential per d, parallel within d. Memory-safe.
"""
import numpy as np
from multiprocessing import Pool
import time, os

os.makedirs('sa_data', exist_ok=True)


def partial_transpose(rho, d):
    n = d * d
    return rho.reshape(d,d,d,d).transpose(0,3,2,1).reshape(n,n)


def realignment_norm(rho, d):
    n = d * d
    R = rho.reshape(d,d,d,d).transpose(0,2,1,3).reshape(n,n)
    return np.linalg.norm(R, 'nuc')


def hunt_worker(args):
    d, seed, n_trials = args
    rng = np.random.RandomState(seed)
    n = d * d
    best = None
    
    for _ in range(n_trials):
        # Random density matrix (Wishart)
        k = max(n, 2*d)  # rank control
        G = rng.randn(n, k) + 1j * rng.randn(n, k)
        rho = G @ G.conj().T
        rho /= np.trace(rho)
        
        # Check PPT
        rho_pt = partial_transpose(rho, d)
        eigs_pt = np.linalg.eigvalsh(rho_pt)
        
        if eigs_pt[0] >= -1e-10:
            # PPT! Check entanglement
            rnorm = realignment_norm(rho, d)
            rank = np.sum(np.linalg.eigvalsh(rho) > 1e-10)
            
            if rnorm > 1.001:
                if best is None or rnorm > best['realign']:
                    best = {
                        'd': d, 'pt_min': eigs_pt[0],
                        'rank': rank, 'realign': rnorm,
                        'rho': rho, 'seed': seed,
                    }
    return best


def hunt_dimension(d, n_workers=15, n_trials_per_worker=500):
    print(f"\n  d={d}: {n_workers} workers x {n_trials_per_worker} trials = {n_workers*n_trials_per_worker} states")
    
    args = [(d, seed, n_trials_per_worker) for seed in range(n_workers)]
    
    t0 = time.time()
    with Pool(n_workers) as pool:
        results = pool.map(hunt_worker, args)
    elapsed = time.time() - t0
    
    hits = [r for r in results if r is not None]
    
    if hits:
        best = max(hits, key=lambda x: x['realign'])
        print(f"  FOUND {len(hits)} PPT entangled states!")
        print(f"  Best: realign={best['realign']:.4f} rank={best['rank']} pt_min={best['pt_min']:.2e}")
        np.savez(f'sa_data/ppt_entangled_d{d}.npz', rho=best['rho'])
        return best
    else:
        print(f"  No PPT entangled found ({elapsed:.1f}s)")
        return None


if __name__ == '__main__':
    print("=" * 50)
    print("  PPT ENTANGLED STATE HUNT")
    print("=" * 50)
    
    all_results = {}
    
    for d in [4, 6, 8, 9, 10, 12]:
        # More trials for smaller d (faster per trial)
        if d <= 6:
            n_trials = 2000
        elif d <= 10:
            n_trials = 500
        else:
            n_trials = 200
        
        r = hunt_dimension(d, n_workers=15, n_trials_per_worker=n_trials)
        all_results[d] = r
    
    print("\n" + "=" * 50)
    print("  SUMMARY")
    print("=" * 50)
    for d, r in all_results.items():
        if r:
            print(f"  d={d:2d}: PPT ENTANGLED  realign={r['realign']:.4f} rank={r['rank']}")
        else:
            print(f"  d={d:2d}: not found")
