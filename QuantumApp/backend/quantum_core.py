import sys
import os
import torch

from quantum_core.math import evaluate_npt_penalty

def project_to_trace_preserving(K_PPT, d, device):
    """
    Strict mathematical projection onto the Trace-Preserving manifold.
    Returns detached inverted square root matrix to prevent NaN gradients.
    """
    num_PPT = K_PPT.shape[0]
    sum_K_K = torch.zeros((d, d), dtype=torch.complex128, device=device)
    for k in range(num_PPT):
        sum_K_K += K_PPT[k].conj().T @ K_PPT[k]
        
    # Force exact Hermiticity to prevent eigh convergence failure
    sum_K_K = (sum_K_K + sum_K_K.conj().T) / 2.0
    
    # STABILIZATION: Add a tiny identity matrix to prevent ill-conditioning
    sum_K_K = sum_K_K + torch.eye(d, dtype=torch.complex128, device=device) * 1e-6
    
    ev_tp, evec_tp = torch.linalg.eigh(sum_K_K)
    diag_val = 1.0 / torch.sqrt(torch.clamp(ev_tp, min=1e-8))
    inv_sqrt_M = evec_tp @ torch.diag(diag_val).to(dtype=torch.complex128) @ evec_tp.conj().T
    
    # DETACH inv_sqrt_M to prevent NaN gradients from repeated eigenvalues!
    inv_sqrt_M = inv_sqrt_M.detach()
    
    K_PPT_TP = torch.zeros_like(K_PPT)
    for k in range(num_PPT):
        K_PPT_TP[k] = K_PPT[k] @ inv_sqrt_M
        
    return K_PPT_TP, sum_K_K

def evaluate_stinespring_capacity(Ks, T_in_norm, device):
    """
    Rigorous Stinespring calculation of coherent information.
    """
    # W_tensor: (num_k, d_out, rank_in)
    W_tensor = torch.einsum('kij, jr -> kir', Ks, T_in_norm)
    
    # B Output State
    W_B = W_tensor.permute(0, 2, 1).reshape(Ks.shape[0] * T_in_norm.shape[1], Ks.shape[1])
    rho_B = W_B.conj().T @ W_B
    # Enforce Hermiticity and stabilize
    rho_B = (rho_B + rho_B.conj().T) / 2.0
    rho_B = rho_B + torch.eye(rho_B.shape[0], dtype=torch.complex128, device=device) * 1e-8
    ev_B = torch.linalg.eigvalsh(rho_B)
    ev_B = ev_B[ev_B > 1e-12]
    S_B = -torch.sum(ev_B * torch.log2(ev_B))
    
    # Environment State
    W_E = W_tensor.reshape(Ks.shape[0], Ks.shape[1] * T_in_norm.shape[1])
    rho_E = W_E @ W_E.conj().T
    rho_E = (rho_E + rho_E.conj().T) / 2.0
    rho_E = rho_E + torch.eye(rho_E.shape[0], dtype=torch.complex128, device=device) * 1e-8
    ev_E = torch.linalg.eigvalsh(rho_E)
    ev_E = ev_E[ev_E > 1e-12]
    S_E = -torch.sum(ev_E * torch.log2(ev_E))
    
    Ic = S_B - S_E
    return Ic

def evaluate_complementary_capacity(Ks, T_in_norm, device):
    """
    Evaluates the Coherent Information of the Complementary Channel.
    If the forward channel represents a Black Hole, the complementary channel 
    represents the Hawking radiation (the Environment).
    I_c(Env) = S_E - S_B
    """
    W_tensor = torch.einsum('kij, jr -> kir', Ks, T_in_norm)
    
    # B Output State (The Black Hole remnant)
    W_B = W_tensor.permute(0, 2, 1).reshape(Ks.shape[0] * T_in_norm.shape[1], Ks.shape[1])
    rho_B = W_B.conj().T @ W_B
    rho_B = (rho_B + rho_B.conj().T) / 2.0
    rho_B = rho_B + torch.eye(rho_B.shape[0], dtype=torch.complex128, device=device) * 1e-8
    ev_B = torch.linalg.eigvalsh(rho_B)
    ev_B = ev_B[ev_B > 1e-12]
    S_B = -torch.sum(ev_B * torch.log2(ev_B))
    
    # Environment State (Hawking Radiation)
    W_E = W_tensor.reshape(Ks.shape[0], Ks.shape[1] * T_in_norm.shape[1])
    rho_E = W_E @ W_E.conj().T
    rho_E = (rho_E + rho_E.conj().T) / 2.0
    rho_E = rho_E + torch.eye(rho_E.shape[0], dtype=torch.complex128, device=device) * 1e-8
    ev_E = torch.linalg.eigvalsh(rho_E)
    ev_E = ev_E[ev_E > 1e-12]
    S_E = -torch.sum(ev_E * torch.log2(ev_E))
    
    Ic_env = S_E - S_B
    return Ic_env

def evaluate_metrology_distance(Ks, T_in_norm, d, device):
    """
    Evaluates the metrological utility of the channel by calculating the distinguishability
    (Frobenius distance) of the output state after a small phase shift on the input.
    """
    dim_in = T_in_norm.shape[0]
    
    # The phase shift MUST ONLY happen on the probe (the first d dimensions)
    # This prevents the AI from cheating by hiding the phase shift in the Ancilla
    H_probe = torch.zeros((d, d), dtype=torch.complex128, device=device)
    for i in range(d):
        H_probe[i, i] = 1.0 if i % 2 == 0 else -1.0
        
    I_ancilla = torch.eye(dim_in // d, dtype=torch.complex128, device=device)
    H = torch.kron(H_probe, I_ancilla)
    
    # Apply a small phase shift theta = 0.05
    theta = 0.05
    U_phase = torch.matrix_exp(-1j * theta * H)
    T_in_shifted = U_phase @ T_in_norm
    
    # Pass baseline state through channel
    W_0 = torch.einsum('kij, jr -> kir', Ks, T_in_norm)
    W_B0 = W_0.permute(0, 2, 1).reshape(Ks.shape[0] * T_in_norm.shape[1], Ks.shape[1])
    rho_B0 = W_B0.conj().T @ W_B0
    
    # Pass shifted state through channel
    W_1 = torch.einsum('kij, jr -> kir', Ks, T_in_shifted)
    W_B1 = W_1.permute(0, 2, 1).reshape(Ks.shape[0] * T_in_shifted.shape[1], Ks.shape[1])
    rho_B1 = W_B1.conj().T @ W_B1
    
    # Calculate Distinguishability (Frobenius norm scaled up for UI readability)
    dist = torch.norm(rho_B0 - rho_B1, p='fro')
    
    # We return dist * 10.0 to make it visually similar in magnitude to Capacity for the UI graph
    return dist * 10.0

def evaluate_retrocausal_fidelity(Ks, T_in_norm, d, device):
    """
    Evaluates the probability of successfully closing a Post-Selected Closed Timelike Curve (P-CTC).
    Based on Seth Lloyd's 2011 model: P_success = <Phi^+ | E(rho) | Phi^+>
    where Phi^+ is the maximally entangled state.
    """
    dim_in = Ks.shape[2]
    
    # Pass baseline state through channel
    W_0 = torch.einsum('kij, jr -> kir', Ks, T_in_norm)
    W_B0 = W_0.permute(0, 2, 1).reshape(Ks.shape[0] * T_in_norm.shape[1], Ks.shape[1])
    out_state = W_B0.conj().T @ W_B0
    
    # Construct the maximally entangled Bell state |Phi+>
    # |Phi+> = 1/sqrt(d) sum |i, i>
    phi_plus = torch.zeros(out_state.shape[0], dtype=torch.complex128, device=device)
    for i in range(d):
        if i * d + i < out_state.shape[0]:
            phi_plus[i * d + i] = (d) ** -0.5
        
    fidelity = torch.real(phi_plus.conj() @ out_state @ phi_plus)
    return fidelity
