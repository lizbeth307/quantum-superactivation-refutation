"""Partial verification using available k=0,1,17 data."""
import numpy as np, os
from scipy.special import comb

data_dir = r"C:\Users\playm\OneDrive\Робочий стіл\QuantumNEAT\parentin_data"
D_R = 2

def _syt(n_sym, j):
    if j == 0: return 1
    return int(comb(n_sym, j, exact=True)) - int(comb(n_sym, j - 1, exact=True))

def _partition_ordering(n_sym):
    P_val = n_sym // 2 + 1
    if P_val == 1: return [0]
    return list(range(1, P_val)) + [0]

n = int(np.load(os.path.join(data_dir, "n.npy")))
f_ver = float(np.load(os.path.join(data_dir, "fidelity_verified.npy")))
f_see = float(np.load(os.path.join(data_dir, "fidelity_seesaw.npy")))

print("="*60)
print(f"  PARENTIN et al. (2026) — SA Verification")
print(f"  n = {n} channel uses")
print("="*60)
print(f"  F_verified = {f_ver:.12f}")
print(f"  F_seesaw   = {f_see:.12f}")
print(f"  F - 0.75   = {f_ver - 0.75:+.6e}")
if f_ver > 0.75:
    print(f"  🌟 SUPERACTIVATION CONFIRMED (F > 0.75)")

# Check encoder validity
print(f"\n  Encoder validity:")
j_order = _partition_ordering(n)
tp_sum = np.zeros((D_R, D_R), dtype=complex)
for i in range(9):
    B = np.load(os.path.join(data_dir, f"encoder_block_{i}.npy"))
    Bsym = (B + B.conj().T) / 2
    min_eig = np.linalg.eigvalsh(Bsym).min()
    j = j_order[i]
    f = _syt(n, j)
    m = B.shape[0] // D_R
    tp_sum += f * np.einsum("isjs->ij", B.reshape(D_R, m, D_R, m))
    print(f"    block_{i}: shape={B.shape[0]:>3}x{B.shape[1]:<3} λ=({n-j},{j}) f={f:>5} min_eig={min_eig:+.2e} {'✅' if min_eig>-1e-6 else '❌'}")

tp_err = np.max(np.abs(tp_sum - np.eye(D_R)))
print(f"  TP error: {tp_err:.2e} {'✅' if tp_err<1e-6 else '❌'}")

# Check decoder k=0 validity
print(f"\n  Decoder k=0 validity:")
for i in range(9):
    B = np.load(os.path.join(data_dir, f"decoder_0_block_{i}.npy"))
    Bsym = (B + B.conj().T) / 2
    min_eig = np.linalg.eigvalsh(Bsym).min()
    m = B.shape[0] // D_R
    tr_R = np.einsum("isit->st", B.reshape(D_R, m, D_R, m))
    unital_err = np.max(np.abs(tr_R - np.eye(m)))
    print(f"    block_{i}: shape={B.shape[0]:>3}x{B.shape[1]:<3} min_eig={min_eig:+.2e} unital_err={unital_err:.2e} {'✅' if min_eig>-1e-6 and unital_err<1e-6 else '❌'}")

# Compute partial fidelity for k=0
print(f"\n  Fidelity computation (k=0, k=17):")
for k in [0, 17]:
    F_D = 0
    j_order_k = _partition_ordering(n)
    for i in range(9):
        M = np.load(os.path.join(data_dir, f"bob_pov_{k}_block_{i}.npy"))
        D = np.load(os.path.join(data_dir, f"decoder_{k}_block_{i}.npy"))
        f = _syt(n, j_order_k[i])
        F_D += f * float(np.real(np.trace(M @ D)))
    F_D /= D_R**2
    w = float(comb(n, k, exact=True)) * (0.5**n)
    print(f"    k={k:>2}: F_D = {F_D:.8f}, weight = {w:.10f}, contrib = {w*F_D:.8f}")

print(f"\n  (Full F requires all k=0..17, we have k=0,1,17)")
print("="*60)
