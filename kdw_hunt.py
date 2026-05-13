import torch
import torch.optim as optim
import numpy as np

def von_neumann_entropy(rho, eps=1e-12):
    ev = torch.linalg.eigvalsh(rho)
    ev = ev[ev > eps]
    return -torch.sum(ev * torch.log2(ev))

def project_to_density_matrix(T):
    rho = T @ T.conj().T
    rho = rho / torch.trace(rho)
    return rho

def npt_penalty(rho, d_A, d_B):
    # Partial transpose over B
    rho_tensor = rho.reshape(d_A, d_B, d_A, d_B)
    rho_pt = rho_tensor.permute(0, 3, 2, 1).reshape(d_A * d_B, d_A * d_B)
    ev = torch.linalg.eigvalsh(rho_pt)
    neg_ev = ev[ev < 0]
    return torch.sum(neg_ev**2)

def evaluate_kdw(rho_AB, U_A, d_A, d_B, device):
    # 1. Purify rho_AB -> |psi>_ABE
    ev, evec = torch.linalg.eigh(rho_AB)
    ev_sqrt = torch.sqrt(torch.clamp(ev, min=0.0))
    # psi_ABE shape: (d_A * d_B, d_E) where d_E = d_A * d_B
    psi_ABE = evec * ev_sqrt.unsqueeze(0)
    
    # 2. Measurement on A
    # U_A is a d_A x d_A unitary
    # We measure in computational basis of A after applying U_A
    
    # Reshape psi to (d_A, d_B, d_E)
    psi_tensor = psi_ABE.reshape(d_A, d_B, d_A * d_B)
    
    # Apply U_A to A subsystem
    # psi_tensor_U = sum_i U_A[x, i] psi_tensor[i, b, e]
    psi_tensor_U = torch.einsum('xi, ibe -> xbe', U_A, psi_tensor)
    
    # Now x is the measurement outcome
    p_x = torch.zeros(d_A, dtype=torch.float64, device=device)
    S_B_x_sum = torch.tensor(0.0, dtype=torch.float64, device=device)
    S_E_x_sum = torch.tensor(0.0, dtype=torch.float64, device=device)
    
    rho_B_avg = torch.zeros((d_B, d_B), dtype=torch.complex128, device=device)
    rho_E_avg = torch.zeros((d_A*d_B, d_A*d_B), dtype=torch.complex128, device=device)
    
    for x in range(d_A):
        # unnormalized post-measurement state on BE
        psi_x = psi_tensor_U[x] # shape (d_B, d_E)
        
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
    return K_DW

def run_kdw_hunt(d_A, d_B, epochs=2000):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize rho_AB
    T_real = torch.randn((d_A*d_B, d_A*d_B), dtype=torch.float64, device=device).requires_grad_(True)
    T_imag = torch.randn((d_A*d_B, d_A*d_B), dtype=torch.float64, device=device).requires_grad_(True)
    
    # Initialize U_A generator (skew-Hermitian matrix H)
    H_real = torch.randn((d_A, d_A), dtype=torch.float64, device=device).requires_grad_(True)
    H_imag = torch.randn((d_A, d_A), dtype=torch.float64, device=device).requires_grad_(True)
    
    optimizer = optim.Adam([T_real, T_imag, H_real, H_imag], lr=0.05)
    
    best_kdw = -999.0
    best_npt = 999.0
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        T = torch.complex(T_real, T_imag)
        rho_AB = project_to_density_matrix(T)
        
        # Make U_A
        H = torch.complex(H_real, H_imag)
        H = H - H.conj().T # skew-Hermitian
        U_A = torch.linalg.matrix_exp(H)
        
        kdw = evaluate_kdw(rho_AB, U_A, d_A, d_B, device)
        npt = npt_penalty(rho_AB, d_A, d_B)
        
        # Loss: maximize kdw, minimize npt
        loss = -kdw + 1000000.0 * npt
        
        loss.backward()
        optimizer.step()
        
        if npt.item() < 1e-7 and kdw.item() > best_kdw:
            best_kdw = kdw.item()
            best_npt = npt.item()
            
        if epoch % 100 == 0:
            print(f"Epoch {epoch:4d} | Loss: {loss.item():.4f} | K_DW: {kdw.item():.5f} | NPT Penalty: {npt.item():.2e}")
            
    print(f"\nFINAL BEST VALID PPT K_DW (d_A={d_A}, d_B={d_B}): {best_kdw:.5f}")

if __name__ == "__main__":
    print("Hunting for PPT states with K_DW > 0...")
    print("Testing d_A=4, d_B=4")
    run_kdw_hunt(4, 4, 3000)
    print("Testing d_A=6, d_B=6")
    run_kdw_hunt(6, 6, 3000)
