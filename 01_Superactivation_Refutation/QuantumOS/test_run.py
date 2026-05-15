import sys
import os
import time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from quantum_core.simulator import sample_circuit, NUM_QUBITS

class QuantumStabilizerAI(nn.Module):
    def __init__(self):
        super(QuantumStabilizerAI, self).__init__()
        self.fc1 = nn.Linear(NUM_QUBITS, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, NUM_QUBITS * 2) 

    def forward(self, inputs):
        x = self.fc1(inputs)
        x = self.relu(x)
        weights = self.fc2(x)
        q_outputs = []
        for i in range(inputs.shape[0]):
            q_out = sample_circuit(inputs[i], weights[i])
            q_outputs.append(torch.stack(q_out))
        return torch.stack(q_outputs)

try:
    print("Initializing Qiskit Aer 30-qubit model...")
    torch.set_default_dtype(torch.float64)
    model = QuantumStabilizerAI()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    inputs = torch.ones((1, NUM_QUBITS))
    target = torch.ones((1, NUM_QUBITS))

    start = time.time()
    print(f"[{time.time()-start:.2f}s] Starting forward pass...")
    noisy_outputs = model(inputs)
    print(f"[{time.time()-start:.2f}s] Forward pass completed! Output shape: {noisy_outputs.shape}")
    
    loss = loss_fn(noisy_outputs, target)
    print(f"[{time.time()-start:.2f}s] Starting backward pass (parameter-shift rule on 60 parameters = ~120 circuits)...")
    loss.backward()
    print(f"[{time.time()-start:.2f}s] Backward pass completed! Gradients computed.")
    
    optimizer.step()
    print(f"[{time.time()-start:.2f}s] Success!")
except Exception as e:
    import traceback
    traceback.print_exc()
