"""
sdp_ppt_search.py — Phase 0.5+: SDP-based PPT entangled state search.
Uses CVXPY to find states maximizing realignment norm subject to PPT.
For d=8,9,10 where random projection fails.
"""
import numpy as np
import time
import os

os.makedirs('sa_data', exist_ok=True)

try:
    import cvxpy as cp
    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False
    print("CVXPY not installed. Install: pip install cvxpy")


def sdp_find_ppt_entangled(dA, dB, verbose=True):
    """Find PPT entangled state for dA x dB using SDP.
    
    Maximize trace norm of realignment subject to:
    - rho >= 0 (PSD)
    - trace(rho) = 1
    - PT(rho) >= 0 (PPT)
    
    Returns rho if PPT entangled found, None otherwise.
    """
    if not HAS_CVXPY:
        return None
    
    d = dA * dB
    
    # Variable
    rho = cp.Variable((d, d), hermitian=True)
    
    # Constraints
    constraints = [
        rho >> 0,           # PSD
        cp.trace(rho) == 1, # Trace 1
    ]
    
    # PPT: partial transpose over B >= 0
    # Build PT explicitly using permutation
    # For dA x dB: PT[i*dB+j, k*dB+l] = rho[i*dB+l, k*dB+j]
    # Use cp.partial_transpose if available, otherwise build manually
    
    # Build permutation for PT
    perm = np.zeros((d*d,), dtype=int)
    for i in range(dA):
        for j in range(dB):
            for k in range(dA):
                for l in range(dB):
                    old = (i*dB+j)*d + (k*dB+l)
                    new = (i*dB+l)*d + (k*dB+j)
                    perm[old] = new
    
    # PT as matrix: pt = P @ vec(rho), where P is permutation
    # Then reshape to matrix and constrain PSD
    # Simpler: build pt_rho as expression
    pt_rho = cp.Variable((d, d), hermitian=True)
    
    for i in range(dA):
        for j in range(dB):
            for k in range(dA):
                for l in range(dB):
                    constraints.append(
                        pt_rho[i*dB+j, k*dB+l] == rho[i*dB+l, k*dB+j]
                    )
    
    constraints.append(pt_rho >> 0)  # PPT
    
    # Objective: maximize some entanglement measure
    # Use trace norm of (rho - I/d) as proxy
    # Or maximize off-diagonal coherence
    # Or: maximize a random linear objective to explore boundary
    
    # Strategy: generate random witness W and maximize Tr(W*rho)
    # Repeat with different W to find entangled states
    
    best_result = None
    
    for trial in range(10):
        # Random entanglement witness
        np.random.seed(trial)
        G = np.random.randn(d, d) + 1j * np.random.randn(d, d)
        W = G + G.conj().T  # Hermitian
        W = W / np.linalg.norm(W)
        
        objective = cp.Maximize(cp.real(cp.trace(W @ rho)))
        prob = cp.Problem(objective, constraints)
        
        try:
            prob.solve(solver=cp.SCS, verbose=False, max_iters=5000)
        except:
            continue
        
        if prob.status in ['optimal', 'optimal_inaccurate']:
            rho_val = rho.value
            if rho_val is None:
                continue
            
            # Check entanglement via realignment
            R = rho_val.reshape(dA, dB, dA, dB).transpose(0, 2, 1, 3).reshape(d, d)
            realign = np.linalg.norm(R, 'nuc')
            
            # Verify PPT
            pt = rho_val.reshape(dA, dB, dA, dB).transpose(0, 3, 2, 1).reshape(d, d)
            pt_min = np.min(np.linalg.eigvalsh(pt))
            
            if verbose:
                print(f"  trial {trial}: realign={realign:.4f} pt_min={pt_min:.2e} "
                      f"{'PPT' if pt_min>=-1e-5 else 'NPT'} "
                      f"{'ENT' if realign>1.001 else 'SEP'}")
            
            if pt_min >= -1e-5 and realign > 1.001:
                if best_result is None or realign > best_result[1]:
                    best_result = (rho_val, realign, pt_min)
    
    return best_result


if __name__ == '__main__':
    print("=" * 60)
    print("  SDP PPT ENTANGLED SEARCH")
    print("=" * 60)
    
    if not HAS_CVXPY:
        print("  ERROR: CVXPY not available")
        exit(1)
    
    for dA, dB in [(2, 4), (3, 3), (4, 4), (2, 5)]:
        d = dA * dB
        print(f"\n  --- {dA}x{dB} (d={d}) ---")
        
        t0 = time.time()
        result = sdp_find_ppt_entangled(dA, dB)
        elapsed = time.time() - t0
        
        if result:
            rho, realign, pt_min = result
            print(f"  FOUND! realign={realign:.4f} pt_min={pt_min:.2e} ({elapsed:.1f}s)")
            np.savez(f'sa_data/sdp_ppt_{dA}x{dB}.npz', rho=rho)
        else:
            print(f"  No PPT+ENT found ({elapsed:.1f}s)")
    
    print(f"\n{'='*60}")
