import torch
import numpy as np
from scipy.optimize import basinhopping
import sys

def build_erasure_channel(d=4, p=0.5):
    Ks = []
    Ks.append(np.sqrt(1 - p) * np.eye(d))
    for i in range(d):
        E_i = np.zeros((d, d))
        E_i[0, i] = 1.0  
        Ks.append(np.sqrt(p) * E_i)
    K_out = []
    K0 = np.zeros((d+1, d), dtype=np.complex128)
    K0[:d, :d] = np.sqrt(1 - p) * np.eye(d)
    K_out.append(K0)
    for i in range(d):
        Ki = np.zeros((d+1, d), dtype=np.complex128)
        Ki[d, i] = np.sqrt(p)
        K_out.append(Ki)
    return torch.tensor(np.array(K_out), dtype=torch.complex128)

def build_smith_yard_ppt(d=4):
    Swap = np.zeros((d*d, d*d), dtype=np.complex128)
    for i in range(d):
        for j in range(d):
            Swap[i*d + j, j*d + i] = 1.0
    P_sym = 0.5 * (np.eye(d*d) + Swap)
    P_anti = 0.5 * (np.eye(d*d) - Swap)
    Ks = []
    factor_k = np.sqrt(1.0 / (d * (d - 1)))
    for k in range(d):
        Vk = np.zeros((d*d, d), dtype=np.complex128)
        for a in range(d):
            Vk[a*d + k, a] = 1.0
        K = factor_k * P_anti @ Vk
        Ks.append(K)
    rho_0 = (2.0 / (d * (d + 1))) * P_sym
    ev, evec = np.linalg.eigh(rho_0)
    for m in range(d*d):
        if ev[m] > 1e-10:
            phi_m = evec[:, m]
            for l in range(d):
                bra_l = np.zeros(d, dtype=np.complex128)
                bra_l[l] = 1.0
                L = np.sqrt(0.5) * np.sqrt(ev[m]) * np.outer(phi_m, bra_l)
                Ks.append(L)
    return torch.tensor(np.array(Ks), dtype=torch.complex128)

def batch_kron(A, B):
    Na, da_out, da_in = A.shape
    Nb, db_out, db_in = B.shape
    A_exp = A.view(Na, 1, da_out, 1, da_in, 1)
    B_exp = B.view(1, Nb, 1, db_out, 1, db_in)
    return (A_exp * B_exp).reshape(Na * Nb, da_out * db_out, da_in * db_in)

def custom_eigh(A, eps=1e-12):
    L, V = torch.linalg.eigh(A)
    return L, V

class EntropyFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rho):
        L, V = custom_eigh(rho)
        L = torch.clamp(L, min=1e-12)
        entropy = -torch.sum(L * torch.log2(L))
        ctx.save_for_backward(L, V)
        return entropy

    @staticmethod
    def backward(ctx, grad_output):
        L, V = ctx.saved_tensors
        dL = - (torch.log2(L) + 1.0 / np.log(2.0))
        dL_complex = torch.complex(dL, torch.zeros_like(dL))
        grad_rho = V @ torch.diag(dL_complex) @ V.conj().T
        return grad_output * grad_rho

def von_neumann_entropy(rho):
    return EntropyFunc.apply(rho)

print("Building exact Smith-Yard channels for d=4...", flush=True)
K_Erasure = build_erasure_channel(d=4, p=0.5)
K_PPT = build_smith_yard_ppt(d=4)
Ks_joint = batch_kron(K_Erasure, K_PPT)
num_k, d_out, d_in = Ks_joint.shape

def objective_with_grad(x_np):
    x = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
    
    real_part = x[:d_in*d_in]
    imag_part = x[d_in*d_in:]
    T_in = torch.complex(real_part, imag_part).reshape(d_in, d_in)
    
    norm = torch.sqrt(torch.sum(torch.abs(T_in)**2))
    T_in = T_in / norm
    
    W_tensor = torch.einsum('kij, jr -> kir', Ks_joint, T_in)
    rho_B = torch.einsum('kir, kjr -> ij', W_tensor, W_tensor.conj())
    
    W_E = W_tensor.reshape(num_k, d_out * d_in)
    rho_E = W_E @ W_E.conj().T
    
    S_B = von_neumann_entropy(rho_B)
    S_E = von_neumann_entropy(rho_E)
    
    Ic = S_B - S_E
    loss = -Ic
    
    loss.backward()
    grad = x.grad.numpy().astype(np.float64)
    return loss.item(), grad

global_best = 0.0

def callback(x, f, accept):
    global global_best
    if -f > global_best:
        global_best = -f
        print(f"New global best Ic: {global_best:.6f}", flush=True)

print(f"Joint Channel: {num_k} operators, output dim {d_out}, input dim {d_in}", flush=True)
print("Starting Basin Hopping with L-BFGS-B gradients...", flush=True)

x0 = np.random.randn(2 * d_in * d_in)
minimizer_kwargs = {"method": "L-BFGS-B", "jac": True}
res = basinhopping(objective_with_grad, x0, minimizer_kwargs=minimizer_kwargs, niter=50, stepsize=0.5, callback=callback, seed=42)

print("\n" + "="*50)
print("SEARCH COMPLETE")
print(f"Best Ic found for d=4: {-res.fun:.6f}")
print("="*50)
