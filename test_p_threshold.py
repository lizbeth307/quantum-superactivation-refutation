import torch
import numpy as np

def build_depolarizing_channel(d: int, p: float = 0.5) -> torch.Tensor:
    Ks = []
    omega = np.exp(2j * np.pi / d)
    for a in range(d):
        for b in range(d):
            U = np.zeros((d, d), dtype=np.complex128)
            for k in range(d):
                U[k, (k + b) % d] = omega ** (a * k)
            if a == 0 and b == 0:
                coeff = np.sqrt(1 - p + p/(d*d))
            else:
                coeff = np.sqrt(p/(d*d))
            Ks.append(coeff * U)
    return torch.tensor(np.array(Ks), dtype=torch.complex128)

def von_neumann_entropy(rho):
    ev = torch.linalg.eigvalsh(rho)
    ev = ev[ev > 1e-12]
    return -torch.sum(ev * torch.log2(ev))

for d in [4, 5, 6]:
    for p in [0.30, 0.31, 0.32]:
        Ks = build_depolarizing_channel(d, p)
        psi = torch.zeros(d*d, dtype=torch.complex128)
        for i in range(d):
            psi[i*d + i] = 1.0 / np.sqrt(d)
        T_in = psi.reshape(d, d)
        W_tensor = torch.einsum('kij, jr -> kir', Ks, T_in)
        rho_B = torch.einsum('kir, kjr -> ij', W_tensor, W_tensor.conj())
        num_k = Ks.shape[0]
        W_E = W_tensor.reshape(num_k, d * d)
        rho_E = W_E @ W_E.conj().T
        S_B = von_neumann_entropy(rho_B)
        S_E = von_neumann_entropy(rho_E)
        print(f"d={d} p={p:.2f} -> Ic: {S_B - S_E:.5f}")
