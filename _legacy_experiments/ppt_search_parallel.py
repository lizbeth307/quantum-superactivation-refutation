"""
ppt_search_parallel.py — Massive parallel PPT boundary search for d>4.
Uses 30 CPU cores to scan random unitaries + weights simultaneously.
"""
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial
import time
import sys

# ── Core functions (must be top-level for pickle) ──

def bell_states(dk):
    states = []
    for m in range(dk):
        for n in range(dk):
            psi = np.zeros(dk * dk, dtype=complex)
            for j in range(dk):
                phase = np.exp(2j * np.pi * j * m / dk)
                idx = j * dk + ((j + n) % dk)
                psi[idx] = phase / np.sqrt(dk)
            states.append(psi)
    return states


def full_index(key_idx, shield_idx, dk, ds):
    kA = key_idx // dk
    kB = key_idx % dk
    sA = shield_idx // ds
    sB = shield_idx % ds
    return kA * (ds * dk * ds) + sA * (dk * ds) + kB * ds + sB


def random_unitary(n):
    """Haar-random unitary matrix."""
    H = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    Q, R = np.linalg.qr(H)
    D = np.diag(np.diag(R) / np.abs(np.diag(R)))
    return Q @ D


def build_state(dk, ds, bells, shields, weights):
    """Build density matrix from Bell states, shields, and weights."""
    d = dk * ds
    rho = np.zeros((d * d, d * d), dtype=complex)
    n_bell = len(bells)
    n_shields = len(shields)
    
    for i in range(n_bell):
        psi = bells[i]
        rho_sh = shields[i % n_shields]
        q = weights[i]
        
        for a in range(dk * dk):
            for b in range(dk * dk):
                kv = psi[a] * psi[b].conj()
                if abs(kv) < 1e-20:
                    continue
                for c in range(ds * ds):
                    for e in range(ds * ds):
                        sv = rho_sh[c, e]
                        if abs(sv) < 1e-20:
                            continue
                        row = full_index(a, c, dk, ds)
                        col = full_index(b, e, dk, ds)
                        rho[row, col] += q * kv * sv
    
    rho = (rho + rho.conj().T) / 2
    return rho


def partial_transpose(rho, d):
    n = d * d
    return rho.reshape(d, d, d, d).transpose(0, 3, 2, 1).reshape(n, n)


def make_shields(dk, ds, U, strategy):
    """Generate shield states using different strategies."""
    def ket(a, b):
        v = np.zeros(ds * ds, dtype=complex)
        v[a * ds + b] = 1.0
        return v
    
    n_bell = dk * dk
    shields = []
    
    if strategy == 'pure':
        # Pure shields from U columns: |chi_k> = sum_j U[j,k] |j,j>
        for k in range(min(ds, n_bell)):
            chi = np.zeros(ds * ds, dtype=complex)
            for j in range(ds):
                chi[j * ds + j] = U[j, k]
            chi /= max(np.linalg.norm(chi), 1e-15)
            shields.append(np.outer(chi, chi.conj()))
        # Pad with mixed states if needed
        while len(shields) < n_bell:
            k1 = len(shields) % ds
            k2 = (k1 + 1) % ds
            chi1 = np.zeros(ds * ds, dtype=complex)
            chi2 = np.zeros(ds * ds, dtype=complex)
            for j in range(ds):
                chi1[j * ds + j] = U[j, k1]
                chi2[j * ds + j] = U[j, k2]
            chi1 /= max(np.linalg.norm(chi1), 1e-15)
            chi2 /= max(np.linalg.norm(chi2), 1e-15)
            shields.append(0.5 * (np.outer(chi1, chi1.conj()) + np.outer(chi2, chi2.conj())))
    
    elif strategy == 'mixed':
        # Mixed: combine basis states and Bell-like states
        for k in range(n_bell):
            idx = k % ds
            diag = ket(idx, idx)
            j1 = idx
            j2 = (idx + 1) % ds
            bell_like = (ket(j1, j2) + ket(j2, j1)) / np.sqrt(2)
            shields.append(0.5 * (np.outer(diag, diag) + np.outer(bell_like, bell_like)))
    
    elif strategy == 'horodecki_gen':
        # Generalized Horodecki: first ds pure from U, rest mixed
        for k in range(ds):
            chi = np.zeros(ds * ds, dtype=complex)
            for j in range(ds):
                chi[j * ds + j] = U[j, k]
            chi /= max(np.linalg.norm(chi), 1e-15)
            shields.append(np.outer(chi, chi.conj()))
        # Additional mixed shields
        for k in range(ds, n_bell):
            idx = k % ds
            diag = ket(idx, idx)
            j2 = (idx + 1) % ds
            bell_like = (ket(idx, j2) + ket(j2, idx)) / np.sqrt(2)
            shields.append(0.5 * (np.outer(diag, diag) + np.outer(bell_like, bell_like)))
    
    return shields[:n_bell]


def search_worker(args):
    """Single worker: try random unitary + weights + strategy."""
    dk, ds, seed = args
    rng = np.random.RandomState(seed)
    d = dk * ds
    bells = bell_states(dk)
    n_bell = dk * dk
    
    best_pt_min = -999
    best_result = None
    
    strategies = ['pure', 'mixed', 'horodecki_gen']
    
    for trial in range(50):  # 50 trials per worker
        U = random_unitary(ds)  # random shield unitary
        strategy = strategies[trial % 3]
        shields = make_shields(dk, ds, U, strategy)
        
        # Try multiple weight distributions
        for w_trial in range(30):
            if w_trial == 0:
                weights = np.ones(n_bell) / n_bell  # uniform
            elif w_trial == 1:
                p1 = np.sqrt(ds) / (1 + np.sqrt(ds))
                w = np.zeros(n_bell)
                half = n_bell // 2
                w[:half] = p1 / half
                w[half:] = (1 - p1) / (n_bell - half)
                weights = w
            else:
                weights = rng.dirichlet(np.ones(n_bell) * 0.5)  # sparser
            
            rho = build_state(dk, ds, bells, shields, weights)
            pt = partial_transpose(rho, d)
            pt_min = np.min(np.linalg.eigvalsh(pt))
            
            # Track closest to PPT boundary from PPT side
            if pt_min >= -1e-10:
                if best_result is None or pt_min < best_result['pt_min']:
                    rank = np.sum(np.linalg.eigvalsh(rho) > 1e-10)
                    best_result = {
                        'pt_min': pt_min, 'rank': rank,
                        'weights': weights.copy(), 'strategy': strategy,
                        'U': U.copy(), 'seed': seed, 'rho': rho
                    }
            elif pt_min > best_pt_min:
                best_pt_min = pt_min
    
    return best_result, best_pt_min


def run_search(dk, ds, n_workers=30, n_tasks_per_worker=1):
    d = dk * ds
    print(f"\n{'='*60}")
    print(f"  PPT SEARCH: dk={dk}, ds={ds}, d={d}")
    print(f"  Workers: {n_workers}, Tasks: {n_workers * n_tasks_per_worker}")
    print(f"  Each worker: 50 unitaries x 20 weight sets = 1000 states")
    print(f"  Total: {n_workers * 1000} states")
    print(f"{'='*60}")
    
    seeds = list(range(n_workers * n_tasks_per_worker))
    args = [(dk, ds, s) for s in seeds]
    
    t0 = time.time()
    with Pool(n_workers) as pool:
        results = pool.map(search_worker, args)
    elapsed = time.time() - t0
    
    # Collect best PPT result
    best_ppt = None
    best_npt = -999
    n_ppt = 0
    
    for result, pt_min in results:
        if result is not None:
            n_ppt += 1
            if best_ppt is None or result['pt_min'] < best_ppt['pt_min']:
                best_ppt = result
        if pt_min > best_npt:
            best_npt = pt_min
    
    print(f"\n  Time: {elapsed:.1f}s")
    print(f"  PPT states found: {n_ppt}/{len(results)} workers")
    
    if best_ppt is not None:
        print(f"  BEST PPT:")
        print(f"    PT_min = {best_ppt['pt_min']:.10f}")
        print(f"    Rank = {best_ppt['rank']}")
        print(f"    Strategy = {best_ppt['strategy']}")
        print(f"    Weights = {best_ppt['weights']}")
        
        # Save state for K_DW computation
        np.savez(f'sa_data/ppt_d{d}.npz', 
                 rho=best_ppt['rho'], dk=dk, ds=ds,
                 weights=best_ppt['weights'], U=best_ppt['U'])
        print(f"    Saved: sa_data/ppt_d{d}.npz")
        return best_ppt
    else:
        print(f"  No PPT state found! Best NPT: pt_min = {best_npt:.6f}")
        return None


if __name__ == '__main__':
    import os
    os.makedirs('sa_data', exist_ok=True)
    
    N_WORKERS = 30
    
    # Search for each factorization
    results = {}
    
    for dk, ds in [(2, 3), (2, 4), (3, 3), (2, 5)]:
        d = dk * ds
        r = run_search(dk, ds, n_workers=N_WORKERS)
        results[(dk, ds)] = r
    
    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    for (dk, ds), r in results.items():
        d = dk * ds
        if r is not None:
            print(f"  d={d:2d} (dk={dk},ds={ds}): PPT! pt_min={r['pt_min']:.8f} rank={r['rank']}")
        else:
            print(f"  d={d:2d} (dk={dk},ds={ds}): NO PPT FOUND")
