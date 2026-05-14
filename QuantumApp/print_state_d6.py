import torch
import numpy as np

data = torch.load("best_superactivation.pt", weights_only=True)
T_in = data['T_in']
d = data['d']

dim_in = T_in.shape[0]
T_tensor = T_in.reshape(d, dim_in // d, T_in.shape[1])
rho_A = torch.einsum('ijr, kjr -> ik', T_tensor, T_tensor.conj())
probs = torch.real(torch.diag(rho_A)).cpu().numpy()

print(f"Exact Probabilities: {probs}")
