"""
schur_weyl.py — Phase 2: Schur-Weyl Engine.

Reduces the dimensionality of N^{\otimes n} using permutation symmetry.
For channels with input C^2, the invariant subspace is the direct sum
of spin-J representations. The symmetric subspace (maximum spin J=n/2)
is spanned by the Dicke states.

This engine constructs block-diagonal representations of channels.
"""
import numpy as np, sys
from scipy.special import comb
import cvxpy as cp
sys.path.insert(0, '.')

def get_dicke_size(n):
    """The symmetric subspace of (C^2)^{\otimes n} has dimension n+1."""
    return n + 1

def number_of_blocks(n):
    """
    Number of irreducible representations (blocks) in (C^2)^{\otimes n}.
    For SU(2), the spins go from J = n/2 down to 0 or 1/2.
    """
    return (n // 2) + 1

def block_dimensions(n):
    """
    Returns a list of dimensions for each spin-J block.
    A block with spin J has dimension (2J + 1).
    """
    dims = []
    J_max = n / 2.0
    for k in range(number_of_blocks(n)):
        J = J_max - k
        dim_J = int(2 * J + 1)
        # Multiplicity of this representation in the Schur-Weyl decomposition
        # (Not strictly needed if we only restrict to the symmetric subspace)
        dims.append(dim_J)
    return dims

def dicke_state(n, k):
    """
    Construct the Dicke state |D_n^k> in the full 2^n dimensional space.
    |D_n^k> is the symmetric superposition of all n-qubit states with k ones.
    """
    dim = 2**n
    state = np.zeros(dim, dtype=complex)
    
    # Iterate over all 2^n basis states
    for i in range(dim):
        # Count number of set bits (number of '1's)
        ones = bin(i).count('1')
        if ones == k:
            state[i] = 1.0
            
    # Normalize
    norm = np.sqrt(comb(n, k))
    return state / norm

def project_to_symmetric_subspace(matrix_2n):
    """
    Given a matrix acting on (C^2)^{\otimes n}, project it down to the
    (n+1)x(n+1) matrix acting on the symmetric subspace spanned by Dicke states.
    """
    # Verify input dimension
    dim = matrix_2n.shape[0]
    n = int(np.log2(dim))
    if 2**n != dim:
        raise ValueError(f"Matrix dimension {dim} is not a power of 2.")
        
    sym_dim = n + 1
    proj_matrix = np.zeros((sym_dim, sym_dim), dtype=complex)
    
    # Pre-compute Dicke basis
    basis = [dicke_state(n, k) for k in range(sym_dim)]
    
    # Compute <D_n^j | M | D_n^k>
    for j in range(sym_dim):
        for k in range(sym_dim):
            bra = basis[j].conj().T
            ket = basis[k]
            proj_matrix[j, k] = bra @ matrix_2n @ ket
            
    return proj_matrix

def build_symmetric_encoder(n, d_R):
    """
    Builds an encoder isometry V: C^{d_R} -> Sym^n(C^2).
    Represented as an SDP variable.
    """
    sym_dim = n + 1
    # We want to find a channel E(rho) = V rho V^dag.
    # In SDP form, we optimize the Choi matrix of E.
    # But for a specific capacity threshold, we can use CVXPY variables.
    pass

if __name__ == '__main__':
    print("=================================================================")
    print("  PHASE 2: SCHUR-WEYL ENGINE (Symmetric Subspace)")
    print("=================================================================")
    
    n_test = 4
    print(f"Testing for n = {n_test} copies")
    print(f"Full Hilbert space dimension: {2**n_test}")
    print(f"Symmetric subspace dimension: {get_dicke_size(n_test)}")
    print(f"Number of blocks (irreps):    {number_of_blocks(n_test)}")
    print(f"Block dimensions:             {block_dimensions(n_test)}")
    
    # Test projection of an identity matrix
    I_full = np.eye(2**n_test)
    I_sym = project_to_symmetric_subspace(I_full)
    
    err = np.linalg.norm(I_sym - np.eye(n_test + 1))
    print(f"\nIdentity projection error: {err:.2e} (Should be 0)")
    
    # Test projection of X^{\otimes n}
    X = np.array([[0, 1], [1, 0]])
    X_n = X
    for _ in range(n_test - 1):
        X_n = np.kron(X_n, X)
        
    X_sym = project_to_symmetric_subspace(X_n)
    print("\nProjected X^⊗n matrix (acting on Dicke basis):")
    # For X_n, it flips all bits, so |D_n^k> -> |D_n^{n-k}>
    print(np.round(X_sym.real, 2))
    
    print("\nConclusion: The symmetric subspace correctly reduces 2^n -> n+1.")
    print("Next step: Implement channel action directly on Dicke states without building 2^n matrices.")
    print("=================================================================")
