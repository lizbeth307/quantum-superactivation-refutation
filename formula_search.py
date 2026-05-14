"""
formula_search.py — Find best closed-form for K_DW of PPT states
Uses verified data (12 states, 300 bases, seed=42)
"""
import numpy as np
from itertools import product

# Verified data from verify_mpmath.py
data = [
    # d, dA, dB, K_DW, S_A, S_B, rank, ||R||₁
    (8, 2, 4, 1.145, 0.827, 1.434, 8, 0.861),
    (9, 3, 3, 1.516, 1.577, 1.577, 9, 1.184),
    (10, 2, 5, 1.066, 0.650, 1.377, 10, 0.856),
    (12, 2, 6, 2.341, 0.993, 2.435, 12, 0.670),
    (14, 2, 7, 2.892, 0.960, 2.681, 14, 0.607),
    (15, 3, 5, 1.847, 1.569, 2.257, 15, 0.789),
    (16, 2, 8, 2.686, 0.999, 2.867, 16, 0.586),
    (18, 2, 9, 2.990, 0.998, 2.990, 18, 0.545),
    (20, 2, 10, 2.830, 0.999, 3.173, 20, 0.514),
    (21, 3, 7, 2.224, 1.561, 2.737, 21, 0.666),
    (24, 2, 12, 2.568, 0.827, 3.019, 24, 0.497),
    (30, 2, 15, 2.679, 0.650, 2.962, 30, 0.494),
]

d_arr = np.array([x[0] for x in data])
dA_arr = np.array([x[1] for x in data])
dB_arr = np.array([x[2] for x in data])
kdw_arr = np.array([x[3] for x in data])
SA_arr = np.array([x[4] for x in data])
SB_arr = np.array([x[5] for x in data])
R_arr = np.array([x[7] for x in data])

print("="*65)
print("  SYMBOLIC FORMULA SEARCH FOR K_DW")
print("  12 verified PPT states, d=8..30")
print("="*65)

# Try many candidate formulas
candidates = []

def try_formula(name, pred):
    """Evaluate a candidate formula."""
    residuals = kdw_arr - pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((kdw_arr - kdw_arr.mean())**2)
    r2 = 1 - ss_res / ss_tot
    mae = np.mean(np.abs(residuals))
    candidates.append((r2, mae, name, pred))

# === Simple formulas ===
try_formula("log₂(d) - 1", np.log2(d_arr) - 1)
try_formula("0.9·log₂(d) - 1.19", 0.9*np.log2(d_arr) - 1.19)
try_formula("log₂(dB)", np.log2(dB_arr))

# === Entropy-based ===
try_formula("S(B) - S(A) + 1", SB_arr - SA_arr + 1)
try_formula("S(B) - 0.3·S(A)", SB_arr - 0.3*SA_arr)
try_formula("S(B)", SB_arr)

# === Optimize: a·S(B) + b·S(A) + c ===
from numpy.linalg import lstsq
X = np.column_stack([SB_arr, SA_arr, np.ones(len(data))])
coeffs, _, _, _ = lstsq(X, kdw_arr, rcond=None)
pred = X @ coeffs
try_formula(f"{coeffs[0]:.3f}·S(B) + {coeffs[1]:.3f}·S(A) + {coeffs[2]:.3f}", pred)

# === log₂(dB) formulas ===
X2 = np.column_stack([np.log2(dB_arr), np.ones(len(data))])
c2, _, _, _ = lstsq(X2, kdw_arr, rcond=None)
try_formula(f"{c2[0]:.3f}·log₂(dB) + {c2[1]:.3f}", X2 @ c2)

# === log₂(dB) + log₂(dA) ===
X3 = np.column_stack([np.log2(dB_arr), np.log2(dA_arr), np.ones(len(data))])
c3, _, _, _ = lstsq(X3, kdw_arr, rcond=None)
try_formula(f"{c3[0]:.3f}·log₂(dB) + {c3[1]:.3f}·log₂(dA) + {c3[2]:.3f}", X3 @ c3)

# === S(B) + σ_eig formula ===
X4 = np.column_stack([SB_arr, SA_arr, R_arr, np.ones(len(data))])
c4, _, _, _ = lstsq(X4, kdw_arr, rcond=None)
try_formula(f"{c4[0]:.3f}·S(B) + {c4[1]:.3f}·S(A) + {c4[2]:.3f}·||R|| + {c4[3]:.3f}", X4 @ c4)

# === Nonlinear: log₂(d) * f(dA) ===
try_formula("log₂(dB) · (1 - 1/dA)", np.log2(dB_arr) * (1 - 1/dA_arr))
try_formula("log₂(dB) · log₂(dA)/log₂(dA+1)", np.log2(dB_arr) * np.log2(dA_arr)/np.log2(dA_arr+1))

# === Information-theoretic ===
try_formula("log₂(dB) - log₂(dA)/2", np.log2(dB_arr) - np.log2(dA_arr)/2)
try_formula("log₂(d) - log₂(dA) - 0.5", np.log2(d_arr) - np.log2(dA_arr) - 0.5)
try_formula("log₂(dB/dA) + 0.5", np.log2(dB_arr/dA_arr) + 0.5)

# === Power-law ===
X5 = np.column_stack([np.log(dB_arr), np.ones(len(data))])
c5, _, _, _ = lstsq(X5, np.log(kdw_arr + 0.01), rcond=None)
try_formula(f"exp({c5[1]:.3f}) · dB^{c5[0]:.3f}", np.exp(c5[1]) * dB_arr**c5[0])

# === Mixed entropy + dimension ===
X6 = np.column_stack([SB_arr, np.log2(dA_arr), np.ones(len(data))])
c6, _, _, _ = lstsq(X6, kdw_arr, rcond=None)
try_formula(f"{c6[0]:.3f}·S(B) + {c6[1]:.3f}·log₂(dA) + {c6[2]:.3f}", X6 @ c6)

# === Capacity-like: log₂(1 + dB/dA) ===
try_formula("log₂(1 + dB)", np.log2(1 + dB_arr))
try_formula("log₂(1 + dB/dA)", np.log2(1 + dB_arr/dA_arr))

# === Best linear in SB only ===
X7 = np.column_stack([SB_arr, np.ones(len(data))])
c7, _, _, _ = lstsq(X7, kdw_arr, rcond=None)
try_formula(f"{c7[0]:.3f}·S(B) + {c7[1]:.3f}", X7 @ c7)

# === Quadratic in log ===
X8 = np.column_stack([np.log2(dB_arr), np.log2(dB_arr)**2, np.ones(len(data))])
c8, _, _, _ = lstsq(X8, kdw_arr, rcond=None)
try_formula(f"{c8[0]:.3f}·log₂(dB) + {c8[1]:.3f}·log₂²(dB) + {c8[2]:.3f}", X8 @ c8)

# Sort by R²
candidates.sort(key=lambda x: -x[0])

print(f"\n  {'R²':>6}  {'MAE':>6}  Formula")
print(f"  {'─'*60}")
for r2, mae, name, _ in candidates[:20]:
    marker = ' 🌟' if r2 > 0.9 else ''
    print(f"  {r2:>6.4f}  {mae:>6.3f}  {name}{marker}")

# Show best
best = candidates[0]
print(f"\n{'='*65}")
print(f"  BEST FORMULA: {best[2]}")
print(f"  R² = {best[0]:.4f}, MAE = {best[1]:.3f} bits")
print(f"{'='*65}")

# Show predictions vs actual for top 3
for idx, (r2, mae, name, pred) in enumerate(candidates[:3]):
    print(f"\n  [{idx+1}] {name} (R²={r2:.4f})")
    print(f"  {'d':>4} {'dA×dB':>6} {'K_DW':>8} {'Pred':>8} {'Err':>8}")
    for i, row in enumerate(data):
        d, dA, dB, kdw = row[0], row[1], row[2], row[3]
        print(f"  {d:>4} {dA}×{dB:>2}   {kdw:>8.3f} {pred[i]:>8.3f} {pred[i]-kdw:>+8.3f}")
