import torch
import time
from quantum_core.simulator import sample_circuit, NUM_QUBITS

print(f"NUM_QUBITS = {NUM_QUBITS}")
inputs = torch.ones(NUM_QUBITS)
weights = torch.ones(NUM_QUBITS * 2, requires_grad=True)

start = time.time()
print("Starting forward...")
out = sample_circuit(inputs, weights)
print(f"Forward done in {time.time()-start:.2f}s. Output:", out)

start = time.time()
print("Starting backward...")
out.sum().backward()
print(f"Backward done in {time.time()-start:.2f}s. Grad shape:", weights.grad.shape)
