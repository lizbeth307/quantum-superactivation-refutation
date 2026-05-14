import torch
import numpy as np
import sys
sys.path.append(r'C:\Users\playm\OneDrive\Робочий стіл\QuantumNEAT')
from phase21_super_synthesizer import build_smith_yard_ppt, evaluate_Ic_joint_gram, evaluate_npt_penalty

def build_depolarizing_channel(d, p):
    Ks = []
    I_mat = torch.eye(d, dtype=torch.complex128)
    Ks.append(np.sqrt(1 - p) * I_mat)
    norm_val = np.sqrt(p / d)
    for i in range(d):
        for j in range(d):
            K = torch.zeros((d, d), dtype=torch.complex128)
            K[i, j] = norm_val
            Ks.append(K)
    return torch.stack(Ks)

d = 4
p = 0.35
epochs = 200
device = "cuda" if torch.cuda.is_available() else "cpu"
d_ancilla = 2
dim_in = d_ancilla * d * d

K_PPT_initial = build_smith_yard_ppt(d).to(device)
num_PPT = K_PPT_initial.shape[0]
K_PPT_real = K_PPT_initial.real.clone().requires_grad_(True)
K_PPT_imag = K_PPT_initial.imag.clone().requires_grad_(True)

K_Era = build_depolarizing_channel(d, p=p).to(device)
d_out_A = d
num_Era = K_Era.shape[0]

rank_in = dim_in
T_in_real_val = torch.randn((dim_in, rank_in), dtype=torch.float64, device=device) * 0.1
T_in_imag_val = torch.randn((dim_in, rank_in), dtype=torch.float64, device=device) * 0.1

T_in_real = T_in_real_val.requires_grad_(True)
T_in_imag = T_in_imag_val.requires_grad_(True)

optimizer = torch.optim.Adam([T_in_real, T_in_imag, K_PPT_real, K_PPT_imag], lr=0.01)

best_ic = -999.0
print("Running rigorous optimization...")

for epoch in range(epochs):
    optimizer.zero_grad()
    K_PPT = torch.complex(K_PPT_real, K_PPT_imag)
    
    K_Era_exp = K_Era.view(num_Era, 1, d_out_A, d, 1, 1)
    K_PPT_exp = K_PPT.view(1, num_PPT, 1, 1, d, d)
    Ks_joint = (K_Era_exp * K_PPT_exp).reshape(num_Era * num_PPT, d_out_A*d, d*d)
    
    I_C = torch.eye(d_ancilla, dtype=torch.complex128, device=device)
    num_k = Ks_joint.shape[0]
    d_out_AB = Ks_joint.shape[1]
    d_in_AB = Ks_joint.shape[2]
    Ks = torch.zeros((num_k, d_ancilla * d_out_AB, d_ancilla * d_in_AB), dtype=torch.complex128, device=device)
    for k in range(num_k):
        Ks[k] = torch.kron(I_C, Ks_joint[k])
        
    T_in = torch.complex(T_in_real, T_in_imag)
    Ic = evaluate_Ic_joint_gram(Ks, T_in)
    
    npt_penalty = evaluate_npt_penalty(K_PPT, d, d)
    sum_K_K = torch.zeros((d, d), dtype=torch.complex128, device=device)
    for k in range(num_PPT):
        sum_K_K += K_PPT[k].conj().T @ K_PPT[k]
    tp_penalty = torch.sum(torch.abs(sum_K_K - torch.eye(d, device=device))**2)
    
    loss = -Ic + 100.0 * npt_penalty + 100.0 * tp_penalty
    loss.backward()
    optimizer.step()
    
    if npt_penalty.item() < 1e-6 and tp_penalty.item() < 1e-4 and Ic.item() > best_ic:
        best_ic = Ic.item()
        torch.save({
            'K_PPT': K_PPT.detach(),
            'T_in': T_in.detach(),
            'Ic': Ic.item(),
            'npt': npt_penalty.item(),
            'tp': tp_penalty.item()
        }, "verify_matrices.pt")
        print(f"Epoch {epoch}: New Best Ic = {best_ic:.5f} (NPT={npt_penalty.item():.2e}, TP={tp_penalty.item():.2e})")

print("Optimization complete. Run verification script next.")
