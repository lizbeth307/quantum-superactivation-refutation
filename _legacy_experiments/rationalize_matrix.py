import numpy as np
import sympy as sp
from fractions import Fraction
import os

path_sparse = r'C:\Users\playm\OneDrive\Робочий стіл\QuantumNEAT\candidate_sparse.npy'
path_full = r'C:\Users\playm\OneDrive\Робочий стіл\QuantumNEAT\candidate_full_rank.npy'

def load_and_rationalize(path, max_denominator=1000):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return None
    mat = np.load(path)
    print(f"\nOriginal Matrix from {os.path.basename(path)} shape {mat.shape}:\n", mat.round(4))
    
    # We want to convert to rational numbers
    # We can use Fraction(x).limit_denominator(max_denominator)
    rat_mat = np.empty(mat.shape, dtype=object)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            # Handling complex numbers if any
            if isinstance(val, complex) or np.iscomplexobj(val):
                real_part = Fraction(val.real).limit_denominator(max_denominator)
                imag_part = Fraction(val.imag).limit_denominator(max_denominator)
                rat_mat[i, j] = real_part + imag_part * 1j
            else:
                rat_mat[i, j] = Fraction(val).limit_denominator(max_denominator)
                
    # Normalize trace to exactly 1
    tr = sum(rat_mat[i, i] for i in range(mat.shape[0]))
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            rat_mat[i, j] = rat_mat[i, j] / tr
            
    print(f"\nRational Matrix (max_denom={max_denominator}):")
    for i in range(mat.shape[0]):
        row_str = " ".join([str(rat_mat[i, j]) for j in range(mat.shape[1])])
        print(row_str)
        
    return rat_mat

def partial_transpose(rho, dim_A=3, dim_B=3):
    rho_tensor = rho.reshape((dim_A, dim_B, dim_A, dim_B))
    rho_pt_tensor = rho_tensor.transpose((0, 3, 2, 1))
    return rho_pt_tensor.reshape((dim_A * dim_B, dim_A * dim_B))

def check_pt(rat_mat):
    print("\nChecking PT eigenvalues via SymPy...")
    # Convert to sympy Matrix
    sp_mat = sp.Matrix(rat_mat)
    sp_mat_pt = sp.Matrix(partial_transpose(rat_mat))
    
    # Check eigenvalues of PT
    # To avoid slow eigenvalue computation, maybe we can just calculate characteristic polynomial
    # and look for negative roots, or evaluate numerically to high precision first.
    eigenvals_evalf = sp_mat_pt.evalf().eigenvals()
    print("Numerical eigenvalues of Rational PT:")
    min_eig = float('inf')
    for ev in eigenvals_evalf:
        if ev.is_real:
            val = float(ev)
        else:
            val = float(ev.as_real_imag()[0])
        print(f"  {val:.6f}")
        if val < min_eig:
            min_eig = val
            
    if min_eig < -1e-8:
        print("=> State is strictly NPT!")
    else:
        print("=> State is PPT (or weakly NPT).")

print("Processing sparse candidate:")
r_sparse = load_and_rationalize(path_sparse, max_denominator=100)
if r_sparse is not None:
    check_pt(r_sparse)

print("\nProcessing full rank candidate:")
r_full = load_and_rationalize(path_full, max_denominator=100)
if r_full is not None:
    check_pt(r_full)

