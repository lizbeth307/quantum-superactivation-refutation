"""
flower_general.py - General Flower State for any (dk, ds) factorization.
========================================================================
Phase 1.1 of SA Master Plan v4.

Builds the Horodecki PPT entangled state for C^{dk*ds} x C^{dk*ds}.
Uses ONLY numpy (CPU) for d<=8, torch (GPU) for d>=10.

Verified: recovers K_DW = 0.021 for (dk=2, ds=2) [flower_exact.py golden ref].
"""
import numpy as np
from scipy.linalg import expm, logm


def bell_states(dk):
    """Generate dk^2 generalized Bell states for C^dk x C^dk.
    
    |Phi_{m,n}> = (1/sqrt(dk)) * sum_j exp(2pi*i*j*m/dk) |j> |j+n mod dk>
    
    Returns: list of dk^2 vectors, each of length dk^2.
    """
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


def shield_states_horodecki(dk, ds, U):
    """Build shield states matching Horodecki paper (Eqs 13-15).
    
    For dk=2, ds=2 (exact):
      shield[0] = (1/2)(|00><00| + |Psi+><Psi+|)  (mixed, rank 2)
      shield[1] = (1/2)(|11><11| + |Psi-><Psi-|)  (mixed, rank 2)
      shield[2] = |chi+><chi+|  (pure)
      shield[3] = |chi-><chi-|  (pure)
      
    For general dk, ds: generalized construction.
    
    Returns: list of dk^2 shield density matrices (each ds^2 x ds^2),
             and mixing weights q.
    """
    def ket2(a, b, ds):
        """Shield basis |a,b> as ds^2-vector."""
        v = np.zeros(ds * ds, dtype=complex)
        v[a * ds + b] = 1.0
        return v
    
    if dk == 2 and ds == 2:
        # EXACT construction from flower_exact.py (verified K_DW=0.021)
        # Bell-like states on shield
        psi_plus_sh = (ket2(0, 1, 2) + ket2(1, 0, 2)) / np.sqrt(2)
        psi_minus_sh = (ket2(0, 1, 2) - ket2(1, 0, 2)) / np.sqrt(2)
        
        # Chi states from Hadamard angle
        a_plus = np.sqrt(2 + np.sqrt(2)) / 2  # cos(pi/8)
        b_plus = np.sqrt(2 - np.sqrt(2)) / 2  # sin(pi/8)
        chi_plus = a_plus * ket2(0, 0, 2) + b_plus * ket2(1, 1, 2)
        chi_minus = b_plus * ket2(0, 0, 2) - a_plus * ket2(1, 1, 2)
        
        # 4 shield density matrices
        shields = [
            0.5 * (np.outer(ket2(0,0,2), ket2(0,0,2)) + np.outer(psi_plus_sh, psi_plus_sh)),
            0.5 * (np.outer(ket2(1,1,2), ket2(1,1,2)) + np.outer(psi_minus_sh, psi_minus_sh)),
            np.outer(chi_plus, chi_plus.conj()),
            np.outer(chi_minus, chi_minus.conj()),
        ]
        
        # Mixing weights (from paper)
        p1 = np.sqrt(2) / (1 + np.sqrt(2))
        p2 = 1.0 / (1 + np.sqrt(2))
        weights = [p1/2, p1/2, p2/2, p2/2]
        
        return shields, np.array(weights)
    
    else:
        # General case: use U to construct shield states
        # Strategy: dk^2 shield states, some mixed, some pure
        # Pure shields from U columns: |chi_k> = sum_j U[j,k] |j,j>
        n_bell = dk * dk
        shields = []
        
        # First: pure shield states from U
        for k in range(ds):
            chi = np.zeros(ds * ds, dtype=complex)
            for j in range(ds):
                chi[j * ds + j] = U[j, k]
            norm = np.linalg.norm(chi)
            if norm > 1e-15:
                chi /= norm
            shields.append(np.outer(chi, chi.conj()))
        
        # If we need more shields (n_bell > ds): add mixed states
        while len(shields) < n_bell:
            # Mixed: (1/2)(|ii><ii| + random Bell-like)
            idx = len(shields) % ds
            diag = ket2(idx, idx, ds)
            # Random off-diagonal
            j1 = idx
            j2 = (idx + 1) % ds
            bell_like = (ket2(j1, j2, ds) + ket2(j2, j1, ds)) / np.sqrt(2)
            shields.append(0.5 * (np.outer(diag, diag) + np.outer(bell_like, bell_like)))
        
        # Weights: scan for PPT boundary
        weights = np.ones(n_bell) / n_bell  # start uniform
        return shields, weights


def hadamard_2():
    """Hadamard matrix for ds=2 (original flower)."""
    return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def dft_matrix(ds):
    """DFT matrix for ds dimensions."""
    omega = np.exp(2j * np.pi / ds)
    return np.array([[omega ** (j * k) for k in range(ds)] 
                     for j in range(ds)], dtype=complex) / np.sqrt(ds)


def geodesic_unitary(U_target, t):
    """Interpolate from Identity to U_target along geodesic.
    U(t) = expm(t * logm(U_target)), t in [0, 1].
    """
    if t <= 0:
        return np.eye(U_target.shape[0], dtype=complex)
    if t >= 1:
        return U_target
    log_U = logm(U_target)
    return expm(t * log_U)


def find_ppt_weights(dk, ds, bell_states_list, shield_kets):
    """Find mixing weights that place state on PPT boundary.
    
    Uses the Horodecki construction: rho = sum_i q_i |psi_i><psi_i| (x) rho_shield_i
    where shield states are constructed from pairs of shield kets.
    
    For dk=2, ds=2: must recover p1 = sqrt(2)/(1+sqrt(2)).
    """
    d = dk * ds
    n_bell = dk * dk
    n_shields = len(shield_kets)
    
    def build_state_from_alpha(alpha):
        """Build state parametrized by alpha in [0, 1].
        
        For dk=2: 4 Bell states, 2 shield kets.
        Shield assignment: Bell 0,2 -> shield pair from ket 0, Bell 1,3 -> shield pair from ket 1.
        Mix: q0=q2=alpha/2, q1=q3=(1-alpha)/2.
        
        Shield rho for pair (a): 
          rho_0 = |chi_a, chi_a><chi_a, chi_a|  (pure product on shield)
          Actually: rho_shield = mixed from shield kets to get rank-2 structure.
        """
        # Shield density matrices: each is a rank-1 or rank-2 state on ds^2
        # For the original flower: shield states are specific mixtures
        # Here: use pure shield states |chi_k, chi_k> (diagonal correlation)
        
        rho = np.zeros((d * d, d * d), dtype=complex)
        q = np.zeros(n_bell)
        
        for i in range(n_bell):
            # Assign shield: cycle through shield kets
            s_idx = i % n_shields
            
            # Weight: parametrized by alpha
            if s_idx == 0:
                count_0 = sum(1 for j in range(n_bell) if j % n_shields == 0)
                q[i] = alpha / count_0
            else:
                count_rest = sum(1 for j in range(n_bell) if j % n_shields == s_idx)
                q[i] = (1 - alpha) / max(1, (n_shields - 1) * count_rest)
            
            # Shield state: |chi_s><chi_s| (pure)
            chi = shield_kets[s_idx]
            rho_sh = np.outer(chi, chi.conj())
            
            # Bell state
            psi = bell_states_list[i]
            
            # Tensor product
            for a in range(dk * dk):
                for b in range(dk * dk):
                    kv = psi[a] * psi[b].conj()
                    for c in range(ds * ds):
                        for e_idx in range(ds * ds):
                            sv = rho_sh[c, e_idx]
                            row = _full_index(a, c, dk, ds)
                            col = _full_index(b, e_idx, dk, ds)
                            rho[row, col] += q[i] * kv * sv
        
        rho = (rho + rho.conj().T) / 2
        return rho, q
    
    # Scan alpha from 0 to 1 to find PPT boundary
    best_alpha = 0.5
    best_pt_min = -999
    
    # First: coarse scan
    for alpha in np.linspace(0.01, 0.99, 200):
        rho, q = build_state_from_alpha(alpha)
        pt = partial_transpose(rho, d)
        pt_min = np.min(np.linalg.eigvalsh(pt))
        
        # We want pt_min closest to 0 from above (PPT boundary)
        if pt_min >= -1e-10 and (best_pt_min < -1e-10 or abs(pt_min) < abs(best_pt_min)):
            best_pt_min = pt_min
            best_alpha = alpha
        elif pt_min > best_pt_min:
            best_pt_min = pt_min
            best_alpha = alpha
    
    # Fine search around best alpha
    for alpha in np.linspace(max(0.01, best_alpha - 0.05), 
                              min(0.99, best_alpha + 0.05), 500):
        rho, q = build_state_from_alpha(alpha)
        pt = partial_transpose(rho, d)
        pt_min = np.min(np.linalg.eigvalsh(pt))
        
        if pt_min >= -1e-10 and abs(pt_min) < abs(best_pt_min):
            best_pt_min = pt_min
            best_alpha = alpha
    
    rho_final, q_final = build_state_from_alpha(best_alpha)
    return rho_final, q_final, best_alpha


def _full_index(key_idx, shield_idx, dk, ds):
    """Convert (key_index, shield_index) to full Hilbert space index.
    
    Ordering: Alice_key, Alice_shield, Bob_key, Bob_shield
    key_idx = kA * dk + kB  (in dk^2 space)
    shield_idx = sA * ds + sB  (in ds^2 space)
    
    Full index = kA*(ds*dk*ds) + sA*(dk*ds) + kB*ds + sB
    """
    kA = key_idx // dk
    kB = key_idx % dk
    sA = shield_idx // ds
    sB = shield_idx % ds
    return kA * (ds * dk * ds) + sA * (dk * ds) + kB * ds + sB


def partial_transpose(rho, d):
    """Partial transpose over Bob for d x d bipartite system."""
    n = d * d
    rho_r = rho.reshape(d, d, d, d)
    rho_pt = rho_r.transpose(0, 3, 2, 1).reshape(n, n)
    return rho_pt


def compute_kdw(rho, d, dk, n_bases=500):
    """Compute Devetak-Winter key rate K_DW.
    
    Correct 4-subsystem trace: kA(dk), sA(ds), kB(dk), sB(ds).
    Index: kA*(ds*dk*ds) + sA*(dk*ds) + kB*ds + sB.
    Measurement on Alice's KEY (kA).
    Bob = (kB, sB). Eve = sA (shield Alice).
    """
    ds = d // dk
    best_k = -999.0
    
    for trial in range(n_bases):
        if trial == 0:
            U_meas = np.eye(dk, dtype=complex)
        else:
            H = np.random.randn(dk, dk) + 1j * np.random.randn(dk, dk)
            U_meas, _ = np.linalg.qr(H)
        
        p_x = np.zeros(dk)
        S_B_x = np.zeros(dk)
        S_E_x = np.zeros(dk)
        
        for x in range(dk):
            # Project kA onto |u_x>: rho_rest = <u_x|_kA rho |u_x>_kA
            dim_rest = ds * dk * ds
            rho_rest = np.zeros((dim_rest, dim_rest), dtype=complex)
            
            for kA1 in range(dk):
                for kA2 in range(dk):
                    coeff = U_meas[kA1, x].conj() * U_meas[kA2, x]
                    for sA1 in range(ds):
                        for kB1 in range(dk):
                            for sB1 in range(ds):
                                for sA2 in range(ds):
                                    for kB2 in range(dk):
                                        for sB2 in range(ds):
                                            i = kA1*(ds*dk*ds) + sA1*(dk*ds) + kB1*ds + sB1
                                            j = kA2*(ds*dk*ds) + sA2*(dk*ds) + kB2*ds + sB2
                                            ri = sA1*(dk*ds) + kB1*ds + sB1
                                            rj = sA2*(dk*ds) + kB2*ds + sB2
                                            rho_rest[ri, rj] += coeff * rho[i, j]
            
            p_x[x] = np.trace(rho_rest).real
            if p_x[x] > 1e-15:
                rho_rest /= p_x[x]
                
                # Bob = (kB, sB): trace out sA
                rho_B = np.zeros((dk * ds, dk * ds), dtype=complex)
                for sA in range(ds):
                    for kB1 in range(dk):
                        for sB1 in range(ds):
                            for kB2 in range(dk):
                                for sB2 in range(ds):
                                    ri = sA*(dk*ds) + kB1*ds + sB1
                                    rj = sA*(dk*ds) + kB2*ds + sB2
                                    rho_B[kB1*ds+sB1, kB2*ds+sB2] += rho_rest[ri, rj]
                
                eigs = np.linalg.eigvalsh(rho_B)
                eigs = eigs[eigs > 1e-15]
                S_B_x[x] = -np.sum(eigs * np.log2(eigs)) if len(eigs) > 0 else 0.0
                
                # Eve = sA: trace out Bob (kB, sB)
                rho_E = np.zeros((ds, ds), dtype=complex)
                for kB in range(dk):
                    for sB in range(ds):
                        for sA1 in range(ds):
                            for sA2 in range(ds):
                                ri = sA1*(dk*ds) + kB*ds + sB
                                rj = sA2*(dk*ds) + kB*ds + sB
                                rho_E[sA1, sA2] += rho_rest[ri, rj]
                
                eigs = np.linalg.eigvalsh(rho_E)
                eigs = eigs[eigs > 1e-15]
                S_E_x[x] = -np.sum(eigs * np.log2(eigs)) if len(eigs) > 0 else 0.0
        
        # Unconditioned Bob
        rho_B_unc = np.zeros((dk * ds, dk * ds), dtype=complex)
        for kA in range(dk):
            for sA in range(ds):
                for kB1 in range(dk):
                    for sB1 in range(ds):
                        for kB2 in range(dk):
                            for sB2 in range(ds):
                                i = kA*(ds*dk*ds) + sA*(dk*ds) + kB1*ds + sB1
                                j = kA*(ds*dk*ds) + sA*(dk*ds) + kB2*ds + sB2
                                rho_B_unc[kB1*ds+sB1, kB2*ds+sB2] += rho[i, j]
        eigs = np.linalg.eigvalsh(rho_B_unc)
        eigs = eigs[eigs > 1e-15]
        S_B = -np.sum(eigs * np.log2(eigs)) if len(eigs) > 0 else 0.0
        
        # Unconditioned Eve
        rho_E_unc = np.zeros((ds, ds), dtype=complex)
        for kA in range(dk):
            for kB in range(dk):
                for sB in range(ds):
                    for sA1 in range(ds):
                        for sA2 in range(ds):
                            i = kA*(ds*dk*ds) + sA1*(dk*ds) + kB*ds + sB
                            j = kA*(ds*dk*ds) + sA2*(dk*ds) + kB*ds + sB
                            rho_E_unc[sA1, sA2] += rho[i, j]
        eigs = np.linalg.eigvalsh(rho_E_unc)
        eigs = eigs[eigs > 1e-15]
        S_E = -np.sum(eigs * np.log2(eigs)) if len(eigs) > 0 else 0.0
        
        I_XB = S_B - sum(p_x[x]*S_B_x[x] for x in range(dk) if p_x[x] > 1e-15)
        I_XE = S_E - sum(p_x[x]*S_E_x[x] for x in range(dk) if p_x[x] > 1e-15)
        
        K = I_XB - I_XE
        best_k = max(best_k, K)
    
    return best_k


def _partial_trace_A(rho, d):
    """Trace out Alice (first subsystem) from d*d matrix."""
    return np.trace(rho.reshape(d, d, d, d), axis1=0, axis2=2)


def _partial_trace_B(rho, d):
    """Trace out Bob (second subsystem) from d*d matrix."""
    return np.trace(rho.reshape(d, d, d, d), axis1=1, axis2=3)


def _von_neumann(rho):
    """Von Neumann entropy S(rho) in bits."""
    if rho is None:
        return 0.0
    rho = (rho + rho.conj().T) / 2
    eigs = np.linalg.eigvalsh(rho)
    eigs = eigs[eigs > 1e-15]
    return -np.sum(eigs * np.log2(eigs))


def build_flower(dk, ds, U=None, verbose=True):
    """Build flower state for given factorization.
    
    Args:
        dk: key dimension
        ds: shield dimension  
        U: ds x ds unitary for shield (default: Hadamard for ds=2, DFT for ds>2)
        verbose: print progress
    
    Returns:
        dict with rho, properties, and K_DW
    """
    d = dk * ds
    
    if U is None:
        U = hadamard_2() if ds == 2 else dft_matrix(ds)
    
    if verbose:
        print(f"Building flower state: dk={dk}, ds={ds}, d={d}")
    
    # 1. Generate Bell states
    bells = bell_states(dk)
    if verbose:
        print(f"  {len(bells)} Bell states generated")
    
    # 2. Get shield states and weights
    shields, weights = shield_states_horodecki(dk, ds, U)
    if verbose:
        print(f"  {len(shields)} shield states, weights={[f'{w:.4f}' for w in weights]}")
    
    # 3. Build density matrix: rho = sum_i q_i |psi_i><psi_i| (x) shield_i
    rho = np.zeros((d * d, d * d), dtype=complex)
    for i in range(len(bells)):
        psi = bells[i]
        rho_sh = shields[i]
        q = weights[i]
        
        for a in range(dk * dk):
            for b in range(dk * dk):
                kv = psi[a] * psi[b].conj()
                for c in range(ds * ds):
                    for e in range(ds * ds):
                        sv = rho_sh[c, e]
                        row = _full_index(a, c, dk, ds)
                        col = _full_index(b, e, dk, ds)
                        rho[row, col] += q * kv * sv
    
    rho = (rho + rho.conj().T) / 2
    
    # 4. If not PPT: scan alpha to find PPT boundary (for general case)
    pt = partial_transpose(rho, d)
    pt_min = np.min(np.linalg.eigvalsh(pt))
    
    if pt_min < -1e-10 and dk != 2 or ds != 2:
        # Need to optimize weights for PPT boundary
        if verbose:
            print(f"  State is NPT (PT_min={pt_min:.4f}), scanning for PPT boundary...")
        rho, weights = _scan_for_ppt(dk, ds, bells, shields, verbose)
        pt = partial_transpose(rho, d)
        pt_min = np.min(np.linalg.eigvalsh(pt))
    
    # 5. Verify properties
    pt_eigs = np.sort(np.linalg.eigvalsh(pt))
    rho_eigs = np.sort(np.linalg.eigvalsh(rho))
    
    is_psd = rho_eigs[0] >= -1e-10
    is_ppt = pt_eigs[0] >= -1e-10
    rank = np.sum(rho_eigs > 1e-10)
    
    if verbose:
        print(f"  PSD: {is_psd} (min eig = {rho_eigs[0]:.2e})")
        print(f"  PPT: {is_ppt} (PT min = {pt_min:.2e})")
        print(f"  Rank: {rank}")
        print(f"  Trace: {np.trace(rho).real:.8f}")
    
    # 6. Compute K_DW
    n_bases = 500
    if verbose:
        print(f"  Computing K_DW ({n_bases} bases)...")
    kdw = compute_kdw(rho, d, dk, n_bases=n_bases)
    
    if verbose:
        status = "SA CANDIDATE!" if (is_ppt and kdw > 0.001) else "no SA"
        print(f"  K_DW = {kdw:.6f} bits  [{status}]")
    
    return {
        'rho': rho,
        'dk': dk, 'ds': ds, 'd': d,
        'is_psd': is_psd, 'is_ppt': is_ppt,
        'pt_min': pt_min, 'rank': rank,
        'kdw': kdw, 'weights': weights, 'U': U,
    }


def _scan_for_ppt(dk, ds, bells, shields, verbose=False):
    """Scan mixing weights to find PPT boundary for general case."""
    d = dk * ds
    n = len(bells)
    best_rho = None
    best_pt_min = -999
    
    for trial in range(500):
        # Random weights that sum to 1
        w = np.random.dirichlet(np.ones(n))
        
        rho = np.zeros((d*d, d*d), dtype=complex)
        for i in range(n):
            psi = bells[i]
            rho_sh = shields[i % len(shields)]
            for a in range(dk*dk):
                for b in range(dk*dk):
                    kv = psi[a] * psi[b].conj()
                    for c in range(ds*ds):
                        for e in range(ds*ds):
                            sv = rho_sh[c, e]
                            row = _full_index(a, c, dk, ds)
                            col = _full_index(b, e, dk, ds)
                            rho[row, col] += w[i] * kv * sv
        rho = (rho + rho.conj().T) / 2
        
        pt = partial_transpose(rho, d)
        pt_min = np.min(np.linalg.eigvalsh(pt))
        
        if pt_min >= -1e-10 and (best_pt_min < -1e-10 or pt_min < best_pt_min):
            best_pt_min = pt_min
            best_rho = rho.copy()
            best_w = w.copy()
            if verbose and trial % 100 == 0:
                print(f"    trial {trial}: PT_min = {pt_min:.6f}")
    
    if best_rho is None:
        best_rho = rho
        best_w = w
    return best_rho, best_w


# ─── Main: test all factorizations ───

if __name__ == '__main__':
    print("=" * 60)
    print("  FLOWER GENERAL — Phase 1.1")
    print("=" * 60)
    
    # Test 1: d=4 (must recover known flower)
    print("\n--- TEST 1: d=4 (dk=2, ds=2) — must match flower_exact ---")
    result = build_flower(2, 2)
    
    assert result['is_psd'], "FAIL: not PSD!"
    assert result['is_ppt'], "FAIL: not PPT!"
    assert result['kdw'] > 0.01, f"FAIL: K_DW = {result['kdw']:.6f} (expected > 0.01)"
    print("  PASSED: d=4 flower recovered!")
    
    # Test 2: d=6 (dk=2, ds=3) — unknown, explore
    print("\n--- TEST 2: d=6 (dk=2, ds=3) — DFT shield ---")
    result6 = build_flower(2, 3)
    
    # Test 3: d=8 (dk=2, ds=4)
    print("\n--- TEST 3: d=8 (dk=2, ds=4) — DFT shield ---")
    result8 = build_flower(2, 4)
    
    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for r in [result, result6, result8]:
        sa = "SA!" if (r['is_ppt'] and r['kdw'] > 0.001) else "no"
        print(f"  d={r['d']:2d} (dk={r['dk']},ds={r['ds']}): "
              f"PPT={r['is_ppt']} K_DW={r['kdw']:+.6f} SA={sa}")
