import numpy as np
import sympy as sp
import os

path_sparse = r'C:\Users\playm\OneDrive\Робочий стіл\QuantumNEAT\candidate_sparse.npy'
mat = np.load(path_sparse)

# Find a nice common denominator
# Looking at the original matrix, the values are things like 0.168, -0.035, etc.
# Let's map to Rational.
# What if we just use nsimplify with rational=True and a small denominator?

dim = mat.shape[0]
sym_mat = sp.zeros(dim, dim)

for i in range(dim):
    for j in range(dim):
        val = mat[i, j]
        if np.iscomplexobj(val) or isinstance(val, complex):
            real_part = sp.nsimplify(val.real, tolerance=0.01, rational=True)
            imag_part = sp.nsimplify(val.imag, tolerance=0.01, rational=True)
            sym_mat[i, j] = real_part + sp.I * imag_part
        else:
            sym_mat[i, j] = sp.nsimplify(val, tolerance=0.01, rational=True)

# Make hermitian
for i in range(dim):
    sym_mat[i, i] = sp.re(sym_mat[i, i])
    for j in range(i+1, dim):
        sym_mat[j, i] = sp.conjugate(sym_mat[i, j])

# Normalize trace
tr = sp.trace(sym_mat)
sym_mat = sym_mat / tr

print("Rational Density Matrix (rho):")
sp.pprint(sym_mat)

def partial_transpose(rho_sym, dim_A=3, dim_B=3):
    pt = sp.zeros(dim_A*dim_B, dim_A*dim_B)
    for i in range(dim_A):
        for j in range(dim_B):
            for k in range(dim_A):
                for l in range(dim_B):
                    # swap A subsystem indices (i, k) -> (k, i) or B (j, l) -> (l, j)
                    # Let's transpose subsystem B
                    pt[i*dim_B + l, k*dim_B + j] = rho_sym[i*dim_B + j, k*dim_B + l]
    return pt

rho_pt = partial_transpose(sym_mat)

print("\nEvaluating PT matrix negative eigenvalue...")
char_poly = rho_pt.charpoly()
# The characteristic polynomial P(lambda)
print("Characteristic polynomial:", char_poly)

# We can find roots numerically to confirm, but char_poly gives the certificate.
# If P(0) has the opposite sign to what we expect for a positive semidefinite matrix, 
# then there is at least one negative eigenvalue. Wait, det(rho_pt) = P(0) * (-1)^d.
det_val = rho_pt.det()
print(f"\nDeterminant of PT matrix: {det_val}")

# We can also output the exact roots if there are any rational ones or we can just show det.
# But for dimension 9, determinant being negative implies at least one negative root.

# Output this exact matrix to a file
with open("exact_rational_matrix.txt", "w") as f:
    f.write(sp.srepr(sym_mat))

