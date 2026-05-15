import torch
import torch.nn as nn
import numpy as np

# ==============================================================================
# ⚛️ QuantumNEAT: NPT Bound Entanglement GAN (V2 - Deep Hunt)
# ==============================================================================

def partial_transpose(rho, dim_A, dim_B):
    rho_tensor = rho.view(dim_A, dim_B, dim_A, dim_B)
    rho_pt_tensor = rho_tensor.permute(0, 3, 2, 1)
    return rho_pt_tensor.reshape(dim_A * dim_B, dim_A * dim_B)

class Generator(nn.Module):
    def __init__(self, dim=9):
        super().__init__()
        self.T = nn.Parameter(torch.randn(dim, dim, dtype=torch.cfloat))
        
    def forward(self):
        rho = self.T @ self.T.mH
        return rho / torch.trace(rho).real

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.A_params = nn.Parameter(torch.randn(3, 2, dtype=torch.cfloat))
        self.B_params = nn.Parameter(torch.randn(3, 2, dtype=torch.cfloat))
        
    def forward(self, rho):
        Q_A, _ = torch.linalg.qr(self.A_params)
        Q_B, _ = torch.linalg.qr(self.B_params)
        V_joint = torch.kron(Q_A, Q_B)
        
        rho_qubit = V_joint.mH @ rho @ V_joint
        rho_qubit = rho_qubit / torch.trace(rho_qubit).real
        rho_q_pt = partial_transpose(rho_qubit, 2, 2)
        
        # We want to minimize the smallest eigenvalue
        eigvals = torch.linalg.eigvalsh(rho_q_pt)
        return eigvals[0]

def run_deep_npt_hunt():
    print("🚀 Initiating Deep GAN Hunt for NPT Bound Entanglement Candidate...")
    
    G = Generator()
    D = Discriminator()
    
    opt_G = torch.optim.Adam(G.parameters(), lr=0.005)
    opt_D = torch.optim.Adam(D.parameters(), lr=0.05)
    
    target_npt = -0.05
    best_candidate_rho = None
    best_defense_score = -float('inf')
    
    for epoch in range(3000):
        # 1. Train Discriminator (The Hacker)
        rho = G().detach()
        for _ in range(10): # D gets 10 tries to crack the state
            opt_D.zero_grad()
            loss_D = D(rho)
            loss_D.backward()
            opt_D.step()
            
        # 2. Train Generator (The Hider)
        opt_G.zero_grad()
        rho = G()
        
        rho_pt = partial_transpose(rho, 3, 3)
        min_eig_overall = torch.linalg.eigvalsh(rho_pt)[0]
        
        loss_NPT = torch.relu(min_eig_overall - target_npt)
        
        score_G_vs_D = D(rho)
        loss_Adversarial = -score_G_vs_D
        
        loss_G = 20.0 * loss_NPT + loss_Adversarial
        loss_G.backward()
        opt_G.step()
        
        # Save the best candidate that is strictly NPT but beats the discriminator
        if min_eig_overall.item() < -0.01 and score_G_vs_D.item() > best_defense_score:
            best_defense_score = score_G_vs_D.item()
            best_candidate_rho = rho.detach().cpu().numpy()
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch:4d} | NPT Eig = {min_eig_overall.item():.4f} | D's Best Eig = {score_G_vs_D.item():.6f}")

    print("\n✅ Hunt Complete.")
    print(f"Best Candidate Defense Score (D's Eigenvalue): {best_defense_score:.6f}")
    if best_defense_score >= 0:
        print("🏆 SUCCESS! Generator created an NPT state that D could NOT distill.")
        np.save("candidate_npt_be.npy", best_candidate_rho)
        print("Candidate state saved to 'candidate_npt_be.npy' for further SDP analysis.")
    else:
        print("D eventually cracked all states. We need a bigger neural network.")

if __name__ == "__main__":
    torch.manual_seed(999)
    run_deep_npt_hunt()
