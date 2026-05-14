import torch
import numpy as np
import time
import os
from QuantumApp.backend.quantum_core import project_to_trace_preserving, evaluate_stinespring_capacity
from QuantumApp.backend.channels import build_depolarizing_channel, build_phase_damping_channel, build_amplitude_damping_channel

# Import evaluate_npt_penalty correctly from phase21
from phase21_super_synthesizer import evaluate_npt_penalty

def batch_kron(A, B):
    Na, da_out, da_in = A.shape
    Nb, db_out, db_in = B.shape
    A_exp = A.view(Na, 1, da_out, 1, da_in, 1)
    B_exp = B.view(1, Nb, 1, db_out, 1, db_in)
    return (A_exp * B_exp).reshape(Na * Nb, da_out * db_out, da_in * db_in)

def autonomous_hunt():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Initializing Global Superactivation Hunter on {device} (Nanometric Precision: complex128)")
    os.makedirs('discoveries', exist_ok=True)
    
    hunt_id = 0
    while True:
        hunt_id += 1
        d = np.random.choice([3, 4])
        print(f"\n--- HUNT #{hunt_id} (Dimension: {d}) ---")
        
        # 1. AI randomly constructs a novel Zero-Capacity environment
        # We blend two different zero-capacity channels to create an undiscovered noise model!
        p1 = np.random.uniform(0.5, 0.8) # >0.5 guarantees 0 capacity for these
        p2 = np.random.uniform(0.5, 0.8)
        
        funcs = [build_depolarizing_channel, build_phase_damping_channel, build_amplitude_damping_channel]
        f1, f2 = np.random.choice(funcs, 2, replace=False)
        
        K_Noise_1 = f1(d, p1).to(device)
        K_Noise_2 = f2(d, p2).to(device)
        
        # The novel noise is the sequence of the two noises
        K_Novel_list = []
        for i in range(K_Noise_1.shape[0]):
            for j in range(K_Noise_2.shape[0]):
                K_Novel_list.append(K_Noise_2[j] @ K_Noise_1[i])
        K_Novel = torch.stack(K_Novel_list)
        
        print(f"Environment: {f1.__name__} (p={p1:.2f}) -> {f2.__name__} (p={p2:.2f})")
        print(f"Novel Noise Model constructed with {K_Novel.shape[0]} pathways.")
        
        # 2. AI initializes a fully free PPT Channel (The Ancilla Repeater)
        K_PPT_real = torch.randn((d*d, d, d), dtype=torch.float64, device=device).requires_grad_(True)
        K_PPT_imag = torch.randn((d*d, d, d), dtype=torch.float64, device=device).requires_grad_(True)
        
        # 3. AI initializes the Input State Tensor
        T_in_real = torch.randn((d*d, d*d), dtype=torch.float64, device=device).requires_grad_(True)
        T_in_imag = torch.randn((d*d, d*d), dtype=torch.float64, device=device).requires_grad_(True)
        
        optimizer = torch.optim.Adam([T_in_real, T_in_imag, K_PPT_real, K_PPT_imag], lr=0.02)
        
        epochs = 300
        best_ic = 0.0
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            K_PPT = torch.complex(K_PPT_real, K_PPT_imag)
            K_PPT_TP, sum_K_K = project_to_trace_preserving(K_PPT, d, device)
            
            # Combine Novel Noise with the PPT Ancilla
            Ks = batch_kron(K_Novel, K_PPT_TP)
            
            T_in = torch.complex(T_in_real, T_in_imag)
            state_norm = torch.trace(T_in.conj().T @ T_in).real
            T_in_norm = T_in / torch.sqrt(torch.clamp(state_norm, min=1e-12))
            
            Ic = evaluate_stinespring_capacity(Ks, T_in_norm, device)
            
            # Use phase21 function correctly
            npt_penalty = evaluate_npt_penalty(K_PPT, d, d)
            tp_penalty = torch.sum(torch.abs(sum_K_K - torch.eye(d, dtype=torch.complex128, device=device)))
            
            loss = -Ic + 1000.0 * npt_penalty + 10.0 * tp_penalty
            loss.backward()
            optimizer.step()
            
            if npt_penalty.item() < 1e-6 and tp_penalty.item() < 1e-4 and Ic.item() > best_ic:
                best_ic = Ic.item()
                
            if epoch % 100 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch:3d} | Capacity: {Ic.item():.5f} | NPT Pen: {npt_penalty.item():.2e}")
                
        if best_ic > 0.01:
            print(f"🚨🚨 BREAKTHROUGH! Superactivation Found! Capacity = {best_ic:.5f} 🚨🚨")
            torch.save({
                'K_Novel': K_Novel.detach(),
                'K_PPT': K_PPT_TP.detach(),
                'T_in': T_in_norm.detach(),
                'Capacity': best_ic
            }, f"discoveries/novel_superactivation_{hunt_id}.pt")
            print("Discovery saved to disk!")
        else:
            print("No viable superactivation found in this sector. Moving to next...")
            
        import gc
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == '__main__':
    autonomous_hunt()
