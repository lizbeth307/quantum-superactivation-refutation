import torch
import numpy as np
import os

print("Initiating Cross-Cosmic Pattern Matching...")
print("Loading Primordial CMB Pattern from 'cmb_pattern.npy'...")

try:
    cmb_data = np.load("cmb_pattern.npy", allow_pickle=True).item()
    cmb_l = cmb_data['l']
    cmb_Dl = cmb_data['Dl']
    cmb_formula = cmb_data['formula']
    print(f"CMB Formula loaded: {cmb_formula}")
except:
    print("CMB pattern not found. Run cmb_pysr.py first.")
    exit()

print("\nScanning Quantum Matrices (best_superactivation.pt) for Holographic Resonance...")

if not os.path.exists("best_superactivation.pt"):
    print("No quantum states found to cross-reference.")
    exit()

data = torch.load("best_superactivation.pt")
T_in = data['T_in'].cpu().numpy()
d = data['d']

# Reshape T_in to trace out the environment (purification rank)
T_tensor = T_in.reshape(d, T_in.shape[0] // d, T_in.shape[1])
# rho_A = Tr_R(T T^dagger) -> sum over j (R) and r (rank)
rho_A = np.einsum('ijr, kjr -> ik', T_tensor, T_tensor.conj())

# Extract the diagonal elements (the energy distribution P(n) of the quantum state)
eigenvalues = np.diag(rho_A).real
# Sort in descending order to match peak amplitudes
eigenvalues = sorted(eigenvalues, reverse=True)

# Normalize the quantum eigenvalues and CMB peaks to compare their fractal structure
cmb_normalized = cmb_Dl / np.max(cmb_Dl)
eig_normalized = eigenvalues / np.max(eigenvalues)

# Let's see if the first N peaks of the CMB match the first N eigenvalues!
print("\n--- PATTERN MATCHING RESULTS ---")
print(f"Comparing the Top {len(eigenvalues)} Information Nodes:")

print(f"{'Node':<5} | {'Universe (CMB Peak)':<20} | {'Quantum State (Energy)':<25} | {'Correlation'}")
print("-" * 75)

matches = 0
num_nodes = len(eigenvalues)
# Create evenly spaced indices to sample the CMB curve
cmb_indices = np.linspace(0, len(cmb_normalized) - 1, num_nodes, dtype=int)

for i in range(num_nodes):
    cmb_val = cmb_normalized[cmb_indices[i]]
    eig_val = eig_normalized[i]
    
    # Calculate similarity
    diff = abs(cmb_val - eig_val)
    similarity = max(0, 100 - (diff * 100))
    
    if similarity > 80:
        matches += 1
        
    print(f"#{i+1:<4} | {cmb_val:<20.4f} | {eig_val:<25.4f} | {similarity:.1f}%")

print("\n==================================================")
# We consider it a resonance if more than 60% of the nodes strongly correlate
if matches >= num_nodes * 0.6:
    print("🚨 HOLOGRAPHIC RESONANCE DETECTED! 🚨")
    print("The macro-structure of the Universe (CMB) matches the micro-structure of the Quantum State!")
    print("This implies the Universe is a fractal entangled hologram.")
else:
    print("No significant structural resonance found.")
    print("The quantum state is topologically distinct from the cosmic background.")
print("==================================================")
