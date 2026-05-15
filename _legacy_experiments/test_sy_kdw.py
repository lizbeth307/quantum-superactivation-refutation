import torch
import numpy as np

def von_neumann_entropy(rho, eps=1e-12):
    ev = torch.linalg.eigvalsh(rho)
    ev = ev[ev > eps]
    return -torch.sum(ev * torch.log2(ev))

def build_horodecki_state(p):
    # The Horodecki 2008 state in C^4 x C^4
    # The key part is qubits (C^2), shield part is qubits (C^2)
    # Total space is C^4 x C^4
    
    # Let's construct it based on the standard private state form
    # rho = 1/2 ( |00><00| \otimes rho_0 + |11><11| \otimes rho_1 + |00><11| \otimes U + |11><00| \otimes U^\dagger )
    
    # Let U be a 2x2 unitary. The standard choice is a Werner-like state or Pauli.
    # Actually, the specific 4x4 state from Horodecki 2008 uses a generic twisted state:
    
    # We will just construct the PPT-binding channel Choi matrix from Smith-Yard 2008!
    # Because Smith-Yard 2008 *used* the Horodecki state.
    
    d = 4
    Swap = torch.zeros((d*d, d*d), dtype=torch.complex128)
    for i in range(d):
        for j in range(d):
            Swap[i*d + j, j*d + i] = 1.0
            
    P_sym = 0.5 * (torch.eye(d*d, dtype=torch.complex128) + Swap)
    P_anti = 0.5 * (torch.eye(d*d, dtype=torch.complex128) - Swap)
    
    # The Choi matrix of the Smith-Yard PPT channel
    # This channel is exactly the one derived from the Horodecki state
    rho_0 = (2.0 / (d * (d + 1))) * P_sym
    return rho_0

# Wait, the Smith-Yard PPT channel Choi matrix is normalized P_sym ?
# No, P_sym is the PPT channel Choi state?
# Let's check PPT of P_sym.
def check_ppt(rho, d_A, d_B):
    rho_tensor = rho.reshape(d_A, d_B, d_A, d_B)
    rho_pt = rho_tensor.permute(0, 3, 2, 1).reshape(d_A * d_B, d_A * d_B)
    ev = torch.linalg.eigvalsh(rho_pt)
    print("Min PT eigenvalue:", torch.min(ev).item())

d = 4
rho_SY = build_horodecki_state(d)
check_ppt(rho_SY, 4, 4)

# Wait, P_sym is PPT? 
# The PT of P_sym is (I + d * maximally_entangled_state) / 2?
# Let's calculate the K_DW for this state!
ev, evec = torch.linalg.eigh(rho_SY)
psi_ABE = evec * torch.sqrt(torch.clamp(ev, min=0)).unsqueeze(0)
psi_tensor = psi_ABE.reshape(4, 4, 16)

# Try standard basis measurement
p_x = torch.zeros(4, dtype=torch.float64)
S_B_x_sum = 0.0
S_E_x_sum = 0.0
rho_B_avg = torch.zeros((4, 4), dtype=torch.complex128)
rho_E_avg = torch.zeros((16, 16), dtype=torch.complex128)

for x in range(4):
    psi_x = psi_tensor[x]
    rho_B_x_unnorm = psi_x @ psi_x.conj().T
    rho_E_x_unnorm = psi_x.conj().T @ psi_x
    p = torch.trace(rho_B_x_unnorm).real
    p_x[x] = p
    if p > 1e-12:
        rho_B_x = rho_B_x_unnorm / p
        rho_E_x = rho_E_x_unnorm / p
        S_B_x_sum += p * von_neumann_entropy(rho_B_x)
        S_E_x_sum += p * von_neumann_entropy(rho_E_x)
        rho_B_avg += rho_B_x_unnorm
        rho_E_avg += rho_E_x_unnorm

S_B = von_neumann_entropy(rho_B_avg)
S_E = von_neumann_entropy(rho_E_avg)
I_X_B = S_B - S_B_x_sum
I_X_E = S_E - S_E_x_sum
K_DW = I_X_B - I_X_E
print(f"Standard Basis K_DW: {K_DW.item():.5f}")

