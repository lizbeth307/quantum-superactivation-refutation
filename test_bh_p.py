import torch
import numpy as np

def build_black_hole_channel(d: int, p: float = 0.5) -> torch.Tensor:
    q_erase = p
    q_dephase = min(p * 1.5, 0.99)
    Ks = []
    K0 = torch.zeros((d+1, d), dtype=torch.complex128)
    for i in range(d):
        K0[i, i] = np.sqrt(1.0 - q_erase) * np.sqrt(1.0 - q_dephase)
    Ks.append(K0)
    for i in range(d):
        Ki = torch.zeros((d+1, d), dtype=torch.complex128)
        Ki[i, i] = np.sqrt(1.0 - q_erase) * np.sqrt(q_dephase)
        Ks.append(Ki)
    for i in range(d):
        Ei = torch.zeros((d+1, d), dtype=torch.complex128)
        Ei[d, i] = np.sqrt(q_erase)
        Ks.append(Ei)
    return torch.stack(Ks)

def von_neumann_entropy(rho):
    ev = torch.linalg.eigvalsh(rho)
    ev = ev[ev > 1e-12]
    return -torch.sum(ev * torch.log2(ev))

for d in [4, 5, 6]:
    for p in [0.30, 0.31, 0.32, 0.33, 0.34, 0.35]:
        Ks = build_black_hole_channel(d, p)
        psi = torch.zeros(d*d, dtype=torch.complex128)
        for i in range(d):
            psi[i*d + i] = 1.0 / np.sqrt(d)
        T_in = psi.reshape(d, d)
        W_tensor = torch.einsum('kij, jr -> kir', Ks, T_in)
        rho_B = torch.einsum('kir, kjr -> ij', W_tensor, W_tensor.conj())
        num_k = Ks.shape[0]
        W_E = W_tensor.reshape(num_k, (d + 1) * d)
        rho_E = W_E @ W_E.conj().T
        S_B = von_neumann_entropy(rho_B)
        S_E = von_neumann_entropy(rho_E)
        print(f"BH d={d} p={p:.2f} -> Ic: {S_B - S_E:.5f}")
