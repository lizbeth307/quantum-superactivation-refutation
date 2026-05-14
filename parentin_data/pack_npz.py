"""Pack all .npy files into single .npz and run verification."""
import numpy as np, os, glob

data_dir = r"C:\Users\playm\OneDrive\Робочий стіл\QuantumNEAT\parentin_data"
npz_path = os.path.join(data_dir, "superactivation_operators_n17.npz")

# Collect all .npy files
arrays = {}
for f in glob.glob(os.path.join(data_dir, "*.npy")):
    key = os.path.splitext(os.path.basename(f))[0]
    arrays[key] = np.load(f)
    
print(f"Packed {len(arrays)} arrays into .npz")
for k in sorted(arrays.keys())[:10]:
    print(f"  {k}: shape={arrays[k].shape if hasattr(arrays[k],'shape') else 'scalar'}")
print(f"  ...")

np.savez(npz_path, **arrays)
print(f"\nSaved to {npz_path}")
print(f"Size: {os.path.getsize(npz_path)} bytes")
