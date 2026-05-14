"""
symbolic_kdw.py — Phase 2: Symbolic Regression for K_DW formula.
Uses PySR to discover analytic expressions for K_DW from features.
"""
import numpy as np
import time
import os

# ── Load all training data ──
features = [
    'rank_norm', 'purity_norm', 'eig_min', 'eig_max', 'eig_std',
    'pt_min', 'pt_boundary_dist', 'pt_neg_count',
    'S_A', 'S_B', 'S_AB', 'mutual_info', 'mutual_info_norm',
    'realign_norm', 'A_max_mixed_dist', 'B_max_mixed_dist',
]

X_all, y_all = [], []
for f in ['path_a_d4.npz', 'path_b_d4.npz', 'path_c_d4.npz', 'path_e_d4.npz',
          'path_d_d6.npz', 'path_d_d8.npz', 'path_d_d10.npz']:
    try:
        data = np.load(f'sa_data/{f}')
        n = len(data['kdw'])
        X = np.zeros((n, len(features)))
        for j, feat in enumerate(features):
            if feat in data:
                X[:, j] = data[feat].real
        y = data['kdw'].real
        X_all.append(X)
        y_all.append(y)
        d = data['d'][0] if 'd' in data else '?'
        print(f"  Loaded {f}: {n} pts, d={d}")
    except Exception as e:
        print(f"  Skip {f}: {e}")

X = np.vstack(X_all)
y = np.concatenate(y_all)
print(f"\n  Total: {len(X)} pts, {X.shape[1]} features")
print(f"  K_DW range: [{y.min():.4f}, {y.max():.4f}]")

# ── Subsample for PySR (too slow on 35k) ──
np.random.seed(42)
idx = np.random.choice(len(X), size=min(5000, len(X)), replace=False)
X_sub = X[idx]
y_sub = y[idx]
print(f"  Subsampled to {len(X_sub)} pts for PySR")

# ── Run PySR ──
from pysr import PySRRegressor

# Select most informative features
# Based on physics: mutual_info, realign_norm, S_AB, pt_min, purity should matter
key_features = [
    'mutual_info', 'realign_norm', 'S_AB', 'pt_min',
    'purity_norm', 'eig_std', 'S_A', 'S_B',
]
key_idx = [features.index(f) for f in key_features]
X_key = X_sub[:, key_idx]

print(f"\n  PySR: {len(key_features)} features: {key_features}")
print(f"  Starting symbolic regression...")

model = PySRRegressor(
    niterations=100,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["sqrt", "log", "exp", "abs"],
    populations=30,
    population_size=50,
    maxsize=25,
    parsimony=0.001,
    timeout_in_seconds=300,
    turbo=True,
    progress=True,
    temp_equation_file="sa_data/pysr_kdw.csv",
)

t0 = time.time()
model.fit(X_key, y_sub, variable_names=key_features)
elapsed = time.time() - t0

print(f"\n  PySR complete: {elapsed:.1f}s")
print(f"\n  Best equations:")
print(model)

# Save best equation
best = model.get_best()
print(f"\n  BEST FORMULA:")
print(f"  K_DW = {best['equation']}")
print(f"  Complexity: {best['complexity']}")
print(f"  Loss: {best['loss']:.6f}")

# Evaluate on full dataset
X_key_full = X[:, key_idx]
y_pred = model.predict(X_key_full)
ss_res = np.sum((y - y_pred) ** 2)
ss_tot = np.sum((y - y.mean()) ** 2)
r2 = 1 - ss_res / ss_tot
mae = np.mean(np.abs(y - y_pred))

print(f"\n  Full dataset performance:")
print(f"  R² = {r2:.4f}")
print(f"  MAE = {mae:.6f}")
print(f"\n  Saved equations to sa_data/pysr_kdw.csv")
