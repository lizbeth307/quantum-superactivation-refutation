import torch
Ks = torch.randn(2, 4, 3) # num_k=2, d_out=4, d_in_sys=3
T_in = torch.randn(3, 5)  # rank_in=5

W_tensor = torch.einsum('kij, jr -> kir', Ks, T_in)

# Expected rho_B: \sum_k W[k] @ W[k].T
rho_B_expected = torch.einsum('kir, kjr -> ij', W_tensor, W_tensor.conj())

# Original method
num_k = 2; rank_in = 5; d_out = 4
W_B = W_tensor.reshape(num_k * rank_in, d_out)
rho_B_original = W_B.T @ W_B.conj()

print("Match:", torch.allclose(rho_B_expected, rho_B_original))
