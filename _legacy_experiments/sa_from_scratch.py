"""
sa_from_scratch.py — Build SA from first principles.

What Smith-Yard (2008) did with pen & paper, and Parentin (2026) with
months of computation — we do with verified math + SDP + GPU.

Architecture:
  1. Schur-Weyl basis: (C²)^⊗n → ⊕_j V_j ⊗ W_j
  2. Dicke states: |D_n^k⟩ for symmetric subspace (j=n/2)
  3. Permutation-invariant encoder: |0_L⟩, |1_L⟩ ∈ Sym^n(C²)
  4. CQ channel N_k in Dicke basis (block-diagonal!)
  5. SDP decoder per erasure count k
  6. Total fidelity: F = Σ_k P(k) · F_k

Key insight: Ñ^⊗n is permutation-covariant, so encoding in
the symmetric subspace (dim n+1) reduces the problem from
dim 2^n to dim (n+1). This is the Schur-Weyl reduction.
"""
import numpy as np, cvxpy as cp, time, sys
from scipy.special import comb
from itertools import combinations

# ═══════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════
P_FLIP = 1.0 / (1.0 + np.sqrt(2.0))  # ≈ 0.2929

# ═══════════════════════════════════════════
#  DICKE STATES (Symmetric Subspace Basis)
# ═══════════════════════════════════════════

def dicke_state(n, hw):
    """Dicke state |D_n^hw⟩ = symmetric superposition of all n-qubit 
    computational basis states with Hamming weight hw.
    
    Returns: (2^n,) vector in the full Hilbert space.
    """
    d = 2**n
    vec = np.zeros(d, dtype=complex)
    # Enumerate all n-bit strings with hw ones
    for positions in combinations(range(n), hw):
        idx = 0
        for pos in positions:
            idx |= (1 << (n - 1 - pos))
        vec[idx] = 1.0
    # Normalize
    norm = np.sqrt(comb(n, hw, exact=True))
    return vec / norm

def dicke_basis(n):
    """Full Dicke basis for n qubits.
    Returns: (2^n, n+1) matrix where column j = |D_n^j⟩.
    """
    d = 2**n
    basis = np.zeros((d, n+1), dtype=complex)
    for hw in range(n+1):
        basis[:, hw] = dicke_state(n, hw)
    return basis

# ═══════════════════════════════════════════
#  NOISE CHANNEL IN DICKE BASIS
# ═══════════════════════════════════════════

def noise_single_qubit():
    """Pauli channel for X^p ∘ Δ̄ on one qubit.
    N(ρ) = (1-p)/2·ρ + (1-p)/2·ZρZ + p/2·XρX + p/2·YρY
    
    Returns list of (prob, Pauli) pairs.
    """
    p = P_FLIP
    return [
        ((1-p)/2, np.eye(2, dtype=complex)),     # I
        ((1-p)/2, np.diag([1,-1]).astype(complex)), # Z
        (p/2, np.array([[0,1],[1,0]], dtype=complex)),  # X
        (p/2, np.array([[0,-1j],[1j,0]], dtype=complex)), # Y
    ]

def build_Nk_dicke(n, k):
    """Build channel N_k in Dicke basis.
    
    N_k: (n-k) identity qubits ⊗ k noisy qubits
    The noisy qubits are in positions {n-k, ..., n-1} (last k).
    
    Since the encoder is permutation-invariant, we average over
    all k-subsets (which positions get noise).
    
    Returns: list of (prob, Kraus_dicke) in the Dicke basis.
    """
    d_sym = n + 1  # dimension of symmetric subspace
    D = dicke_basis(n)  # (2^n, n+1)
    d = 2**n
    
    # For each Pauli string on k noisy qubits: 
    # P = P_1 ⊗ P_2 ⊗ ... ⊗ P_k applied to k specific positions
    # Then average over all C(n,k) choices of positions
    
    noise = noise_single_qubit()  # 4 terms
    
    # Build Kraus in Dicke basis
    # For k noisy qubits, the Pauli string has 4^k terms
    # Each term: probability p_1·p_2·...·p_k, operator P_1⊗...⊗P_k on k qubits
    
    if k == 0:
        return [(1.0, np.eye(d_sym, dtype=complex))]
    
    # For efficiency: enumerate Pauli strings on k qubits
    n_paulis = len(noise)  # 4
    
    kraus_list = []
    
    # Average over all C(n,k) position sets
    n_subsets = int(comb(n, k, exact=True))
    
    for positions in combinations(range(n), k):
        # For each Pauli string on these k positions
        for pauli_idx in np.ndindex(*([n_paulis]*k)):
            prob = 1.0
            # Build full n-qubit Pauli
            ops = [np.eye(2, dtype=complex)] * n
            for q, pi in enumerate(pauli_idx):
                p_q, P_q = noise[pi]
                prob *= p_q
                ops[positions[q]] = P_q
            
            # Full operator: O = ops[0] ⊗ ops[1] ⊗ ... ⊗ ops[n-1]
            O = ops[0]
            for j in range(1, n):
                O = np.kron(O, ops[j])
            
            # Project to Dicke basis: K = D† O D  (d_sym × d_sym)
            K_dicke = D.conj().T @ O @ D
            
            kraus_list.append((prob / n_subsets, K_dicke))
    
    return kraus_list

def build_Nk_dicke_fast(n, k):
    """Faster version: precompute Dicke basis action of Paulis.
    
    Since Pauli X flips bits and Z adds phases, their action on 
    Dicke states can be computed combinatorially.
    """
    d_sym = n + 1
    
    if k == 0:
        return [(1.0, np.eye(d_sym, dtype=complex))]
    
    if k > n:
        return []
    
    # For small n, use the explicit construction
    if 2**n <= 2048:  # n <= 11
        return build_Nk_dicke(n, k)
    
    # For larger n: need combinatorial approach
    # TODO: implement for n > 11
    raise NotImplementedError(f"n={n} too large for explicit construction")

# ═══════════════════════════════════════════
#  CHANNEL FIDELITY IN DICKE BASIS
# ═══════════════════════════════════════════

def fidelity_dicke(n, k, V, use_sdp=False):
    """Compute entanglement fidelity for encoder V in Dicke basis.
    
    V: (n+1, 2) isometry (encoder maps qubit to symmetric subspace)
    N_k: channel in Dicke basis
    
    F_k = max_D (1/4) Σ_{ij} Tr(D(σ[i,j]) · |j⟩⟨i|)
    where σ[i,j] = Σ_{K,p} p · K V|i⟩⟨j|V† K†
    """
    d_sym = n + 1
    d_R = 2
    
    kraus_list = build_Nk_dicke_fast(n, k)
    
    # Build σ_RB
    sigma = np.zeros((d_R * d_sym, d_R * d_sym), dtype=complex)
    for prob, K in kraus_list:
        KV = K @ V  # (d_sym, d_R)
        for i in range(d_R):
            for j in range(d_R):
                sigma[i*d_sym:(i+1)*d_sym, j*d_sym:(j+1)*d_sym] += \
                    prob * np.outer(KV[:,i], KV[:,j].conj()) / d_R
    
    if use_sdp:
        # SDP decoder
        dim = d_sym * d_R
        sigma_BR = np.zeros((dim, dim), dtype=complex)
        for i in range(d_R):
            for j in range(d_R):
                for a in range(d_sym):
                    for b in range(d_sym):
                        sigma_BR[a*d_R+i, b*d_R+j] = sigma[i*d_sym+a, j*d_sym+b]
        
        J = cp.Variable((dim, dim), hermitian=True)
        obj = cp.Maximize(cp.real(cp.trace(J @ sigma_BR)) / (d_R**2))
        constraints = [J >> 0]
        for a in range(d_sym):
            for b in range(d_sym):
                val = sum(J[a*d_R+r, b*d_R+r] for r in range(d_R))
                constraints.append(val == (1 if a == b else 0))
        
        prob_sdp = cp.Problem(obj, constraints)
        prob_sdp.solve(solver=cp.CLARABEL, verbose=False)
        return prob_sdp.value if prob_sdp.value is not None else 0
    else:
        # Isometric decoder
        ev, evc = np.linalg.eigh(sigma)
        # F_e = max eigval
        # But need to account for entanglement fidelity formula:
        # F_e = (1/d_R) Σ_{ij} ⟨i|W σ[i,j] W†|j⟩
        w = evc[:, -1]
        W = w.reshape(d_R, d_sym)
        Uw, sw, Vhw = np.linalg.svd(W, full_matrices=False)
        W = Uw @ Vhw
        
        F = 0
        for i in range(d_R):
            for j in range(d_R):
                bl = sigma[i*d_sym:(i+1)*d_sym, j*d_sym:(j+1)*d_sym]
                F += (W @ bl @ W.conj().T)[i, j]
        return F.real / d_R

def seesaw_dicke(n, k, n_iter=30, n_restarts=20, use_sdp=False):
    """Seesaw in Dicke basis: alternate encoder/decoder optimization.
    
    Encoder: V ∈ C^(n+1, 2) isometry
    Decoder: W ∈ C^(2, n+1) isometry or CPTP via SDP
    """
    d_sym = n + 1
    d_R = 2
    best_F = 0
    best_V = None
    
    kraus_list = build_Nk_dicke_fast(n, k)
    
    for restart in range(n_restarts):
        np.random.seed(restart)
        V = np.random.randn(d_sym, d_R) + 1j*np.random.randn(d_sym, d_R)
        U, s, Vh = np.linalg.svd(V, full_matrices=False); V = U @ Vh
        
        for it in range(n_iter):
            # σ_RB
            sigma = np.zeros((d_R*d_sym, d_R*d_sym), dtype=complex)
            for prob, K in kraus_list:
                KV = K @ V
                for i in range(d_R):
                    for j in range(d_R):
                        sigma[i*d_sym:(i+1)*d_sym, j*d_sym:(j+1)*d_sym] += \
                            prob * np.outer(KV[:,i], KV[:,j].conj()) / d_R
            
            # Decoder: isometric
            ev, evc = np.linalg.eigh(sigma)
            w = evc[:, -1]
            W = w.reshape(d_R, d_sym)
            Uw, sw, Vhw = np.linalg.svd(W, full_matrices=False)
            W = Uw @ Vhw
            
            # Fidelity
            F = sum((W @ sigma[i*d_sym:(i+1)*d_sym, j*d_sym:(j+1)*d_sym] @ W.conj().T)[i,j] 
                    for i in range(d_R) for j in range(d_R)).real / d_R
            
            # Encoder update
            TK_list = []
            for prob, K in kraus_list:
                TK_list.append((prob, W @ K))
            
            M = np.zeros((d_sym*d_R, d_sym*d_R), dtype=complex)
            for prob, TK in TK_list:
                t = TK.T.ravel('F')
                M += prob * np.outer(t, t.conj())
            
            ev2, evc2 = np.linalg.eigh(M)
            v = evc2[:, -1]
            V = v.reshape(d_sym, d_R, order='F')
            Uv, sv, Vhv = np.linalg.svd(V, full_matrices=False)
            V = Uv @ Vhv
        
        if F > best_F:
            best_F = F
            best_V = V.copy()
    
    # Final SDP decoder with best encoder
    if use_sdp and best_V is not None:
        F_sdp = fidelity_dicke(n, k, best_V, use_sdp=True)
        return max(best_F, F_sdp), best_V
    
    return best_F, best_V

# ═══════════════════════════════════════════
#  MAIN: FIND MINIMUM n* FOR SA
# ═══════════════════════════════════════════

if __name__ == '__main__':
    print("="*65)
    print("  SA FROM SCRATCH — Schur-Weyl Symmetric Subspace Codes")
    print("  Building our own quantum error correcting codes for SA")
    print("="*65)
    
    results = {}
    
    for n in range(1, 12):
        d_sym = n + 1
        d_full = 2**n
        
        if d_full > 2048:
            print(f"\n  n={n}: 2^n={d_full} too large, stopping")
            break
        
        print(f"\n  ═══ n={n} (sym dim={d_sym}, full dim={d_full}) ═══")
        
        F_total = 0
        t0 = time.time()
        
        for k in range(n+1):
            pk = comb(n, k, exact=True) / (2**n)
            
            t1 = time.time()
            Fk, Vk = seesaw_dicke(n, k, n_iter=25, n_restarts=15, use_sdp=False)
            dt = time.time() - t1
            
            contrib = pk * Fk
            F_total += contrib
            
            print(f"    k={k:2d}: P={pk:.4f} F_D={Fk:.6f} contrib={contrib:.6f} [{dt:.1f}s]")
        
        dt_total = time.time() - t0
        sa = F_total > 0.75
        marker = " 🌟 SA!" if sa else ""
        print(f"  → F_c(Ñ^⊗{n}, 2) ≥ {F_total:.6f} [{dt_total:.1f}s]{marker}")
        results[n] = F_total
    
    # Summary
    print(f"\n{'='*65}")
    print(f"  SCALING LAW")
    print(f"  {'n':>3} {'F_c':>8} {'SA':>5}")
    print(f"  {'-'*20}")
    for n, F in sorted(results.items()):
        print(f"  {n:3d} {F:8.4f} {'🌟' if F > 0.75 else ''}")
    
    print(f"\n  Reference: Parentin n=17 → F=0.750131")
    print(f"  Our codes use Dicke (symmetric subspace) encoding")
    print(f"{'='*65}")
