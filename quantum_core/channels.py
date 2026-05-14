import torch
import numpy as np

def build_erasure_channel(d: int, p: float = 0.5) -> torch.Tensor:
    """Builds the quantum erasure channel."""
    Ks = []
    # Intact channel mapping d -> d+1
    K_intact = np.zeros((d+1, d), dtype=np.complex128)
    for i in range(d):
        K_intact[i, i] = np.sqrt(1 - p)
    Ks.append(K_intact)
    
    # Error pathways mapping d -> d+1 (erasure flag at d)
    for i in range(d):
        K_err = np.zeros((d+1, d), dtype=np.complex128)
        K_err[d, i] = np.sqrt(p)
        Ks.append(K_err)
    return torch.tensor(np.array(Ks), dtype=torch.complex128)

def build_depolarizing_channel(d: int, p: float = 0.5) -> torch.Tensor:
    """Builds the generalized depolarizing channel using Weyl operator basis."""
    Ks = []
    # Weyl operators: U_{ab} = sum_k exp(2 pi i a k / d) |k><k+b|
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

def build_smith_yard_ppt(d: int = 4) -> torch.Tensor:
    """Constructs the theoretical Smith-Yard PPT channel manifold."""
    Swap = torch.zeros((d*d, d*d), dtype=torch.complex128)
    for i in range(d):
        for j in range(d):
            Swap[i*d + j, j*d + i] = 1.0
            
    P_sym = 0.5 * (torch.eye(d*d, dtype=torch.complex128) + Swap)
    P_anti = 0.5 * (torch.eye(d*d, dtype=torch.complex128) - Swap)
    
    Ks = []
    factor_k = torch.sqrt(torch.tensor(1.0 / (d * (d - 1)), dtype=torch.float64))
    for k in range(d):
        Vk = torch.zeros((d*d, d), dtype=torch.complex128)
        for a in range(d):
            Vk[a*d + k, a] = 1.0
        K = factor_k * P_anti @ Vk
        Ks.append(K)
        
    rho_0 = (2.0 / (d * (d + 1))) * P_sym
    ev, evec = torch.linalg.eigh(rho_0)
    for m in range(d*d):
        if ev[m] > 1e-10:
            phi_m = evec[:, m]
            for l in range(d):
                bra_l = torch.zeros(d, dtype=torch.complex128)
                bra_l[l] = 1.0
                L = torch.sqrt(torch.tensor(0.5)) * torch.sqrt(ev[m]) * torch.outer(phi_m, bra_l)
                Ks.append(L)
                
    return torch.stack(Ks)

def build_phase_damping_channel(d: int, p: float = 0.5) -> torch.Tensor:
    Ks = []
    K0 = torch.zeros((d, d), dtype=torch.complex128)
    K0[0, 0] = 1.0
    for i in range(1, d):
        K0[i, i] = np.sqrt(1.0 - p)
    Ks.append(K0)
    for i in range(1, d):
        Ki = torch.zeros((d, d), dtype=torch.complex128)
        Ki[i, i] = np.sqrt(p)
        Ks.append(Ki)
    return torch.stack(Ks)

def build_amplitude_damping_channel(d: int, p: float = 0.5) -> torch.Tensor:
    Ks = []
    K0 = torch.zeros((d, d), dtype=torch.complex128)
    K0[0, 0] = 1.0
    for i in range(1, d):
        K0[i, i] = np.sqrt(1.0 - p)
    Ks.append(K0)
    for i in range(1, d):
        Ki = torch.zeros((d, d), dtype=torch.complex128)
        Ki[0, i] = np.sqrt(p)
        Ks.append(Ki)
    return torch.stack(Ks)

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

def build_wormhole_channel(d: int, p: float = 0.5) -> torch.Tensor:
    # A wormhole is modeled as a bipartite Dephrasure channel
    # K_wormhole = K_BH \otimes K_BH
    K_BH = build_black_hole_channel(d, p)
    # We compute the kronecker product of all pairs of Kraus operators
    K_wormhole = []
    for k1 in K_BH:
        for k2 in K_BH:
            K_wormhole.append(torch.kron(k1, k2))
    return torch.stack(K_wormhole)
