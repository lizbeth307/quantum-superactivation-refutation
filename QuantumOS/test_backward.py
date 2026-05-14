import sys
import os
import torch
import torch.nn as nn
torch.set_default_dtype(torch.float64)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Temporarily override NUM_QUBITS for faster testing
import quantum_core.simulator
import quantum_core.chunked_simulator
quantum_core.simulator.NUM_QUBITS = 10
quantum_core.chunked_simulator.PyTorchChunkedSimulator.num_qubits = 10

from ui.quantum_os import QuantumStabilizerAI, shared_state

print("Testing forward and backward pass on GPU chunked simulator...")
model = QuantumStabilizerAI()
inputs = torch.ones((1, 10))
target = torch.ones((1, 10))
loss_fn = nn.MSELoss()

print("Forward pass...")
outputs = model(inputs)
print("Outputs device:", outputs.device)

loss = loss_fn(outputs, target)
print("Loss device:", loss.device)

print("Backward pass...")
loss.backward()
print("Backward pass complete!")

print("All device mismatches fixed!")
