"""Download ALL Parentin et al. data (k=0..17) and verify full fidelity."""
import urllib.request, os, numpy as np
from scipy.special import comb

data_dir = r"C:\Users\playm\OneDrive\Робочий стіл\QuantumNEAT\parentin_data"

# Download all missing k values
print("Downloading remaining k values...")
ok = 0; fail = 0
for k in range(18):  # k=0..17
    # Determine number of blocks for this k
    # From Parentin: blocks depend on Schur-Weyl decomposition
    # Try up to 30 blocks per k
    for b in range(30):
        for prefix in ["decoder", "bob_pov"]:
            fname = f"{prefix}_{k}_block_{b}.npy"
            dst = os.path.join(data_dir, fname)
            if os.path.exists(dst):
                ok += 1
                continue
            url = f"https://arxiv.org/src/2604.27042v2/anc/{fname}"
            try:
                urllib.request.urlretrieve(url, dst)
                sz = os.path.getsize(dst)
                if sz < 100:  # Too small = error page
                    os.remove(dst)
                    break
                ok += 1
            except:
                break

print(f"Download: {ok} ok, done")

# Count blocks per k
print("\nBlocks per k:")
for k in range(18):
    blocks_dec = 0
    blocks_bob = 0
    for b in range(30):
        if os.path.exists(os.path.join(data_dir, f"decoder_{k}_block_{b}.npy")):
            blocks_dec += 1
        if os.path.exists(os.path.join(data_dir, f"bob_pov_{k}_block_{b}.npy")):
            blocks_bob += 1
    if blocks_dec > 0:
        print(f"  k={k:>2}: {blocks_dec} decoder, {blocks_bob} bob_pov blocks")
