import torch
import torch.nn as nn
import numpy as np

# ==============================================================================
# ⚛️ QuantumNEAT: 2-Undistillable NPT Bound Entanglement (Monte Carlo Search)
# ==============================================================================
# We bypass the GAN instability and SDP limitations.
# To prove a state is 2-undistillable, we must show that NO local projections 
# from ρ⊗ρ down to 2 qubits can produce a state with a negative partial transpose.
# 
# We optimize ρ. In every step, we test ρ⊗ρ against thousands of random 
# distillation protocols (Monte Carlo projections). 
# We penalize ρ if ANY protocol succeeds.
# ==============================================================================

def partial_transpose_2q(rho_batch):
    # rho_batch: (B, 4, 4)
    rt = rho_batch.view(-1, 2, 2, 2, 2)
    rt_pt = rt.permute(0, 1, 4, 3, 2)
    return rt_pt.reshape(-1, 4, 4)

class RobustNPTHunter(nn.Module):
    def __init__(self, dim=9, num_samples=2000):
        super().__init__()
        self.dim = dim
        self.num_samples = num_samples
        self.T = nn.Parameter(torch.randn(dim, dim, dtype=torch.cfloat))

    def forward(self):
        # 1. Generate State
        rho = self.T @ self.T.mH
        rho = rho / torch.trace(rho).real
        
        # 2. Check Overall NPT condition
        rt = rho.view(3, 3, 3, 3)
        rt_pt = rt.permute(0, 3, 2, 1).reshape(9, 9)
        min_eig_rho = torch.linalg.eigvalsh(rt_pt)[0]
        
        # We want rho to be clearly NPT (e.g. eigenvalue <= -0.05)
        loss_npt = torch.relu(min_eig_rho - (-0.05))**2
        
        # 3. Construct 2-copy state (81x81)
        rho2 = torch.kron(rho, rho)
        r2t = rho2.view(3, 3, 3, 3, 3, 3, 3, 3)
        # Permute (A1, B1, A2, B2) -> (A1, A2, B1, B2)
        r2t_perm = r2t.permute(0, 2, 1, 3, 4, 6, 5, 7)
        rho2_bip = r2t_perm.reshape(81, 81)
        
        # 4. Generate Random Projections (Monte Carlo Distillation)
        # We generate random 9x2 isometries
        A_rand = torch.randn(self.num_samples, 9, 2, dtype=torch.cfloat, device=rho.device)
        B_rand = torch.randn(self.num_samples, 9, 2, dtype=torch.cfloat, device=rho.device)
        
        Q_A, _ = torch.linalg.qr(A_rand)
        Q_B, _ = torch.linalg.qr(B_rand)
        
        # We can't do a full kron batch easily without einsum, so we do it via einsum
        # Q_A is (B, 9, 2), Q_B is (B, 9, 2). V_joint should be (B, 81, 4)
        V_joint = torch.einsum('bij,bkl->bikjl', Q_A, Q_B).reshape(self.num_samples, 81, 4)
        
        # Project state: V^dagger @ rho2 @ V
        # V_joint.mH is (B, 4, 81)
        rho_q = V_joint.mH @ rho2_bip.unsqueeze(0) @ V_joint
        
        # Normalize
        traces = torch.diagonal(rho_q, dim1=-2, dim2=-1).sum(-1).real
        rho_q = rho_q / traces.view(-1, 1, 1)
        
        # 5. Check distillability (Partial Transpose of the 2-qubit states)
        rho_q_pt = partial_transpose_2q(rho_q)
        eigvals = torch.linalg.eigvalsh(rho_q_pt)
        
        # Find the absolute worst case (minimum eigenvalue across all samples)
        # If this is < 0, the state was distilled!
        min_eig_distill = torch.min(eigvals[:, 0])
        
        # We want min_eig_distill to be >= 0 (undistillable)
        loss_distill = torch.relu(-min_eig_distill)**2
        
        total_loss = 100.0 * loss_npt + loss_distill
        return total_loss, min_eig_rho, min_eig_distill

def run_monte_carlo_hunt():
    print("🎯 Starting Monte Carlo K=2 Undistillability Hunt 🎯")
    print("Testing 2000 random distillation protocols per step.")
    
    model = RobustNPTHunter(num_samples=2000)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
    
    best_defense = -float('inf')
    
    for step in range(1500):
        optimizer.zero_grad()
        loss, npt_eig, distill_eig = model()
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if distill_eig.item() > best_defense and npt_eig.item() < -0.02:
            best_defense = distill_eig.item()
            
        if step % 50 == 0:
            print(f"Step {step:4d} | NPT Eig = {npt_eig.item():.4f} | Worst Distill Eig = {distill_eig.item():.6f}")
            
        if distill_eig.item() >= 0 and npt_eig.item() <= -0.05:
            print("\n🎉 PERFECT ROBUST CANDIDATE FOUND!")
            rho = model.T @ model.T.mH
            rho = rho / torch.trace(rho).real
            np.save("candidate_mc.npy", rho.detach().cpu().numpy())
            break

    print("\nOptimization Complete.")
    print(f"Best Defense Score (should be >= 0): {best_defense:.6f}")

if __name__ == "__main__":
    torch.manual_seed(888)
    run_monte_carlo_hunt()
