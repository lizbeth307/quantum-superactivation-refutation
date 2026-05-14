import torch
import numpy as np
from typing import Tuple, Optional

def von_neumann_entropy(rho: torch.Tensor) -> torch.Tensor:
    """
    Computes the Von Neumann entropy of a density matrix.
    S(rho) = -Tr(rho log2 rho)
    """
    ev, _ = custom_eigh(rho)
    ev = ev[ev > 1e-12]
    if len(ev) == 0:
        return torch.tensor(0.0, dtype=torch.float64, device=rho.device)
    return -torch.sum(ev * torch.log2(ev))

def batch_kron(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Computes the Kronecker product of two sets of Kraus operators.
    A: (Na, da_out, da_in)
    B: (Nb, db_out, db_in)
    Returns: (Na*Nb, da_out*db_out, da_in*db_in)
    """
    Na, da_out, da_in = A.shape
    Nb, db_out, db_in = B.shape
    A_exp = A.view(Na, 1, da_out, 1, da_in, 1)
    B_exp = B.view(1, Nb, 1, db_out, 1, db_in)
    return (A_exp * B_exp).reshape(Na * Nb, da_out * db_out, da_in * db_in)

def make_tp_kraus(K_raw: torch.Tensor) -> torch.Tensor:
    """
    Projects a set of Kraus operators onto the Trace-Preserving (TP) manifold.
    """
    num_k, d_out, d_in = K_raw.shape
    M = torch.zeros((d_in, d_in), dtype=torch.complex128, device=K_raw.device)
    for i in range(num_k):
        M += K_raw[i].conj().T @ K_raw[i]
        
    jitter = torch.randn_like(M) * 1e-8
    jitter = (jitter + jitter.conj().T) / 2
    if torch.isnan(M).any():
        print("CRITICAL: M contains NaN!")
    ev, evec = custom_eigh(M + jitter)
    diag_val = 1.0 / torch.sqrt(torch.clamp(ev, min=1e-8))
    diag_mat = torch.diag(diag_val).to(dtype=torch.complex128)
    inv_sqrt_M = evec @ diag_mat @ evec.conj().T
    
    K_tp = torch.zeros_like(K_raw)
    for i in range(num_k):
        K_tp[i] = K_raw[i] @ inv_sqrt_M
    return K_tp

def evaluate_stinespring_capacity(Ks: torch.Tensor, T_in: torch.Tensor) -> torch.Tensor:
    """
    Differentiable evaluation of Quantum Capacity I_c = S(B) - S(E)
    using the Stinespring dilation.
    """
    num_k, d_out, d_in = Ks.shape
    dim_in_state = T_in.shape[1]
    
    W_tensor = torch.einsum('kij, jr -> kir', Ks, T_in)
    
    # Correct computation of rho_B: Trace out E (k) and R (r)
    rho_B = torch.einsum('kir, kjr -> ij', W_tensor, W_tensor.conj())
    
    # Correct computation of rho_E: Trace out B (i) and R (r)
    # W_tensor is (num_k, d_out, dim_in_state)
    W_E = W_tensor.reshape(num_k, d_out * dim_in_state)
    rho_E = W_E @ W_E.conj().T
    
    S_B = von_neumann_entropy(rho_B)
    S_E = von_neumann_entropy(rho_E)
    return S_B - S_E

def evaluate_npt_penalty(Ks: torch.Tensor, d_out: int, d_in: int) -> torch.Tensor:
    """
    Computes the PPT penalty for a set of Kraus operators.
    If the channel is NPT (contains useful entanglement), returns a >0 penalty.
    """
    J_tensor = torch.einsum('kai, kbj -> iajb', Ks, Ks.conj())
    Choi_PT_tensor = J_tensor.permute(0, 3, 2, 1)
    Choi_PT = Choi_PT_tensor.reshape(d_in * d_out, d_in * d_out)
    ev, _ = custom_eigh(Choi_PT)
    return torch.sum(torch.abs(torch.clamp(ev, max=0.0))**3)

class MemoryEfficientEigh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        jitter = torch.randn_like(A) * 1e-8
        jitter = (jitter + jitter.conj().T) / 2
        L, V = torch.linalg.eigh(A + jitter)
        ctx.save_for_backward(L, V)
        return L, V

    @staticmethod
    def backward(ctx, grad_L, grad_V):
        L, V = ctx.saved_tensors
        grad_A = None
        if ctx.needs_input_grad[0]:
            N = L.shape[0]
            device = L.device
            dtype = V.dtype
            with torch.no_grad():
                if grad_V is not None:
                    M = torch.matmul(V.mH, grad_V)
                    M_anti = 0.5 * (M - M.mH)
                    L_diff = L.unsqueeze(1) - L.unsqueeze(0)
                    mask = torch.abs(L_diff) > 1e-10
                    F = torch.zeros_like(L_diff, dtype=dtype)
                    F[mask] = (1.0 / L_diff[mask]).to(F.dtype)
                    H = F * M_anti
                else:
                    H = torch.zeros((N, N), dtype=dtype, device=device)
                if grad_L is not None:
                    H.diagonal().add_(grad_L.to(dtype))
                VH = torch.matmul(V, H)
                grad_A = torch.matmul(VH, V.mH)
        return grad_A

def custom_eigh(A):
    return MemoryEfficientEigh.apply(A)
