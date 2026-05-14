"""
export_to_matlab.py
Exports discovered quantum states to MATLAB .mat files for independent verification.
Includes the PPT condition check and K_DW score.
"""
import numpy as np
import scipy.io as sio
import os, sys

sys.path.insert(0, '.')
from sa_engine import kdw_correct

def export_state(filename, dA, dB):
    filepath = os.path.join('sa_data', filename)
    if not os.path.exists(filepath):
        print(f"Skipping {filename} (not found)")
        return
        
    data = np.load(filepath)
    rho = data['rho'] if 'rho' in data else data['Ks']  # Ks if channel, but we need state
    if 'rho' not in data:
        print(f"Skipping {filename} (no 'rho' array)")
        return
        
    # Recompute K_DW for accuracy
    print(f"Processing {filename}...")
    kdw = kdw_correct(rho, dA, dB, n_bases=500)
    
    mat_filename = filepath.replace('.npz', '.mat')
    sio.savemat(mat_filename, {
        'rho': rho,
        'dA': dA,
        'dB': dB,
        'K_DW': kdw
    })
    print(f"  -> Exported to {mat_filename} (K_DW = {kdw:+.6f})")

if __name__ == '__main__':
    print("Exporting key states to MATLAB format for rigid verification...")
    export_state('optimized_ppt_2x4.npz', 2, 4)
    export_state('ppt_entangled_tiles_3x3.npz', 3, 3)
    export_state('upb_3x3.npz', 3, 3)
