import torch
import torch.nn as nn
import numpy as np

# ==============================================================================
# ⚛️ QuantumNEAT: 2-Copy NPT Bound Entanglement GAN
# ==============================================================================
# We upgrade the Discriminator to attack TWO COPIES of the state simultaneously.
# The state is ρ ⊗ ρ (81x81 matrix).
# Alice has 9 dimensions (3x3), Bob has 9 dimensions (3x3).
# D tries to project this 81x81 state down to a 4x4 (2-qubit) state to distill it.
# If G wins, we have a 2-undistillable NPT state!
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

class DiscriminatorK2(nn.Module):
    def __init__(self):
        super().__init__()
        # Alice and Bob now project from 9 dimensions down to 2!
        self.A_params = nn.Parameter(torch.randn(9, 2, dtype=torch.cfloat))
        self.B_params = nn.Parameter(torch.randn(9, 2, dtype=torch.cfloat))
        
    def forward(self, rho):
        # 1. Compute ρ ⊗ ρ
        rho_tensor = torch.kron(rho, rho) # 81 x 81 matrix
        
        # 2. Rearrange subsystems from (A1, B1, A2, B2) to (A1, A2, B1, B2)
        # Original shape: 3 (A1), 3 (B1), 3 (A2), 3 (B2) x 3 (A1'), 3 (B1'), 3 (A2'), 3 (B2')
        rt = rho_tensor.view(3, 3, 3, 3, 3, 3, 3, 3)
        # Permute: A1(0), A2(2), B1(1), B2(3), A1'(4), A2'(6), B1'(5), B2'(7)
        rt_perm = rt.permute(0, 2, 1, 3, 4, 6, 5, 7)
        # Reshape to bipartite 9x9 system
        rho_bipartite_k2 = rt_perm.reshape(81, 81)
        
        # 3. Apply local projections (9->2 for Alice, 9->2 for Bob)
        Q_A, _ = torch.linalg.qr(self.A_params)
        Q_B, _ = torch.linalg.qr(self.B_params)
        V_joint = torch.kron(Q_A, Q_B) # 81 x 4 matrix
        
        rho_qubit = V_joint.mH @ rho_bipartite_k2 @ V_joint
        rho_qubit = rho_qubit / torch.trace(rho_qubit).real
        
        # 4. Check if it's distillable
        rho_q_pt = partial_transpose(rho_qubit, 2, 2)
        eigvals = torch.linalg.eigvalsh(rho_q_pt)
        
        return eigvals[0]

def run_k2_npt_hunt():
    print("🚀 Initiating K=2 (Two-Copy) GAN Hunt for NPT Bound Entanglement...")
    
    G = Generator()
    D = DiscriminatorK2()
    
    # We lower the learning rates because 81x81 landscape is extremely chaotic
    opt_G = torch.optim.Adam(G.parameters(), lr=0.002)
    opt_D = torch.optim.Adam(D.parameters(), lr=0.02)
    
    target_npt = -0.05
    best_candidate_rho = None
    best_defense_score = -float('inf')
    
    for epoch in range(1501):
        # 1. Train Discriminator (The Hacker)
        rho = G().detach()
        for _ in range(10): # D gets 10 tries
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
        
        loss_G = 30.0 * loss_NPT + loss_Adversarial
        loss_G.backward()
        opt_G.step()
        
        if min_eig_overall.item() < -0.02 and score_G_vs_D.item() > best_defense_score:
            best_defense_score = score_G_vs_D.item()
            best_candidate_rho = rho.detach().cpu().numpy()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch:4d} | NPT Eig = {min_eig_overall.item():.4f} | K=2 D's Best Eig = {score_G_vs_D.item():.6f}")

    print("\n✅ K=2 Hunt Complete.")
    print(f"Best K=2 Candidate Defense Score: {best_defense_score:.6f}")
    if best_defense_score >= 0:
        print("🏆 INCREDIBLE! G created an NPT state that D could not distill EVEN WITH 2 COPIES.")
        np.save("candidate_npt_be_k2.npy", best_candidate_rho)
    else:
        print("❌ D cracked all states using 2 copies. NPT bound entanglement remains elusive.")

if __name__ == "__main__":
    torch.manual_seed(123)
    run_k2_npt_hunt()
