"""
pysr_v2.py — Improved PySR with dimension as explicit feature.
Also includes log(d), 1/d for better scaling discovery.
"""
import numpy as np
import time

features = [
    'mutual_info', 'realign_norm', 'S_AB', 'pt_min',
    'purity_norm', 'eig_std', 'S_A', 'S_B', 'd',
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
        X_all.append(X); y_all.append(y)
    except:
        pass

X = np.vstack(X_all); y = np.concatenate(y_all)
print(f"Total: {len(X)} pts")

# Add derived features
log_d = np.log2(X[:, features.index('d')])
inv_d = 1.0 / np.maximum(X[:, features.index('d')], 1)
X = np.column_stack([X, log_d, inv_d])
features_ext = features + ['log2_d', 'inv_d']

# Subsample
np.random.seed(42)
idx = np.random.choice(len(X), size=min(8000, len(X)), replace=False)
X_sub = X[idx]; y_sub = y[idx]

print(f"Features: {features_ext}")
print(f"Subsampled: {len(X_sub)} pts")

from pysr import PySRRegressor

model = PySRRegressor(
    niterations=150,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["sqrt", "log", "exp", "abs", "square"],
    populations=40,
    population_size=60,
    maxsize=30,
    parsimony=0.0005,
    timeout_in_seconds=500,
    turbo=True,
    progress=True,
    temp_equation_file="sa_data/pysr_kdw_v2.csv",
)

t0 = time.time()
model.fit(X_sub, y_sub, variable_names=features_ext)
elapsed = time.time() - t0

print(f"\nPySR v2 complete: {elapsed:.1f}s")
print(f"\nBest equations:")
print(model)

best = model.get_best()
print(f"\nBEST FORMULA:")
print(f"  K_DW = {best['equation']}")
print(f"  Complexity: {best['complexity']}")
print(f"  Loss: {best['loss']:.6f}")

# Evaluate on full dataset
y_pred = model.predict(X)
ss_res = np.sum((y - y_pred)**2)
ss_tot = np.sum((y - y.mean())**2)
r2 = 1 - ss_res / ss_tot
print(f"\nFull dataset R² = {r2:.4f}")
print(f"MAE = {np.mean(np.abs(y - y_pred)):.6f}")
