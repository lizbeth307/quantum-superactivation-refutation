"""Download Parentin et al. supplementary data from arXiv."""
import urllib.request, os

base = "https://arxiv.org/src/2604.27042v2/anc/"
out = r"C:\Users\playm\OneDrive\Робочий стіл\QuantumNEAT\parentin_data"

# Key files
files = ["n.npy", "fidelity_seesaw.npy", "fidelity_verified.npy"]

# Encoder blocks (9)
for i in range(9):
    files.append(f"encoder_block_{i}.npy")

# Decoder and bob_pov for k=0 (simplest case: 9 blocks each)
for i in range(9):
    files.append(f"decoder_0_block_{i}.npy")
    files.append(f"bob_pov_0_block_{i}.npy")

# k=17 (all-erasure case)
for i in range(9):
    files.append(f"decoder_17_block_{i}.npy")
    files.append(f"bob_pov_17_block_{i}.npy")

# k=1 (9 blocks? check)
for i in range(9):
    files.append(f"decoder_1_block_{i}.npy")
    files.append(f"bob_pov_1_block_{i}.npy")

print(f"Downloading {len(files)} files...")
ok = 0; fail = 0
for f in files:
    url = base + f
    dst = os.path.join(out, f)
    if os.path.exists(dst):
        ok += 1; continue
    try:
        urllib.request.urlretrieve(url, dst)
        sz = os.path.getsize(dst)
        print(f"  ✓ {f} ({sz} bytes)")
        ok += 1
    except Exception as e:
        print(f"  ✗ {f}: {e}")
        fail += 1

print(f"\nDone: {ok} ok, {fail} failed")

# Load and inspect
import numpy as np
n = np.load(os.path.join(out, "n.npy"))
f_ver = np.load(os.path.join(out, "fidelity_verified.npy"))
f_see = np.load(os.path.join(out, "fidelity_seesaw.npy"))
print(f"\nn = {n}")
print(f"fidelity_verified = {f_ver}")
print(f"fidelity_seesaw = {f_see}")

enc0 = np.load(os.path.join(out, "encoder_block_0.npy"))
print(f"\nencoder_block_0 shape: {enc0.shape}")
print(f"encoder_block_0 dtype: {enc0.dtype}")
print(f"min eig: {np.linalg.eigvalsh((enc0+enc0.conj().T)/2).min():.6e}")
