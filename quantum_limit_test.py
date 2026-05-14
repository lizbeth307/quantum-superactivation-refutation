import torch
import time
import os
import psutil

def test_quantum_limit():
    print("=== Quantum Simulator Benchmark ===")
    print("Testing maximum qubits on Ryzen 9 CPU...\n")
    
    # Start from 15 qubits
    max_safe_qubits = 30 # 30 qubits = ~16 GB RAM
    
    for n_qubits in range(15, max_safe_qubits + 1):
        # Calculate required memory for statevector
        # 2^N elements, each complex128 (16 bytes)
        elements = 2**n_qubits
        mem_gb = elements * 16 / (1024**3)
        
        # Check system available memory before allocating
        available_mem_gb = psutil.virtual_memory().available / (1024**3)
        if mem_gb * 2 > available_mem_gb: # We need roughly 2x memory for operations
            print(f"\n[!] STOPPING at {n_qubits} qubits. Require {mem_gb*2:.2f} GB RAM, but only {available_mem_gb:.2f} GB available.")
            print("Your laptop has reached the classical memory limit!")
            break
            
        print(f"Allocating {n_qubits} qubits (State space: {elements:,} states)... Memory needed: {mem_gb:.4f} GB", end="", flush=True)
        
        start_time = time.time()
        
        # 1. Allocate quantum state (all 0s initially)
        # Using float64/complex128 to match real quantum simulators
        try:
            state = torch.zeros(elements, dtype=torch.complex128)
            state[0] = 1.0 + 0.0j # Initial state |0...0>
            
            # 2. Simulate applying a global phase gate (or simple math op) across all states
            # This forces the CPU to actually touch every single state in memory
            state = state * torch.exp(torch.tensor(1j * 0.5))
            
            # 3. Calculate probability sum (should be 1.0)
            prob_sum = torch.sum(torch.abs(state)**2)
            
            del state # Free memory
        except RuntimeError as e:
            print(f"\n[X] FAILED due to Memory Error: {e}")
            break
            
        end_time = time.time()
        calc_time = end_time - start_time
        
        print(f" -> Done in {calc_time:.4f} sec")
        
        # Give memory a moment to free up
        time.sleep(0.5)

if __name__ == "__main__":
    test_quantum_limit()
