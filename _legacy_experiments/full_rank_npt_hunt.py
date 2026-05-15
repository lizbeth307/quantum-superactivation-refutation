import torch
import torch.nn as nn
import numpy as np
import sys

# ==============================================================================
# ⚛️ QuantumNEAT: Full-Rank NPT Bound Entangled Candidate Hunt
# ==============================================================================
# The AI previously "cheated" by making P_11 = P_22 = 0 to trivially satisfy the 
# NPT condition. We now forbid this boundary trick.
# We demand that the state has FULL RANK (all eigenvalues >= 0.05).
# This forces the AI to use genuine quantum structure rather than edge loopholes.
# ==============================================================================

def partial_transpose_2q(rho_batch):
    rt = rho_batch.view(-1, 2, 2, 2, 2)
    rt_pt = rt.permute(0, 1, 4, 3, 2)
    return rt_pt.reshape(-1, 4, 4)

class FullRankNPTHunter(nn.Module):
    def __init__(self, dim=9, num_samples=2000):
        super().__init__()
        self.dim = dim
        self.num_samples = num_samples
        self.T = nn.Parameter(torch.randn(dim, dim, dtype=torch.cfloat))

    def forward(self):
        rho = self.T @ self.T.mH
        rho = rho / torch.trace(rho).real
        
        # 1. NPT Penalty
        rt = rho.view(3, 3, 3, 3)
        rt_pt = rt.permute(0, 3, 2, 1).reshape(9, 9)
        min_eig_rho_pt = torch.linalg.eigvalsh(rt_pt)[0]
        loss_npt = torch.relu(min_eig_rho_pt - (-0.05))**2
        
        # 2. FULL-RANK Penalty (No zeros allowed on the diagonal / eigenvalues)
        # We force all eigenvalues of rho to be >= 0.03
        eigvals_rho = torch.linalg.eigvalsh(rho)
        min_eig_rho = eigvals_rho[0]
        loss_full_rank = torch.relu(0.03 - min_eig_rho)**2
        
        # 3. Distillation Penalty (Monte Carlo Hacker)
        rho2 = torch.kron(rho, rho)
        r2t = rho2.view(3, 3, 3, 3, 3, 3, 3, 3)
        rho2_bip = r2t.permute(0, 2, 1, 3, 4, 6, 5, 7).reshape(81, 81)
        
        A_rand = torch.randn(self.num_samples, 9, 2, dtype=torch.cfloat, device=rho.device)
        B_rand = torch.randn(self.num_samples, 9, 2, dtype=torch.cfloat, device=rho.device)
        
        Q_A, _ = torch.linalg.qr(A_rand)
        Q_B, _ = torch.linalg.qr(B_rand)
        
        V_joint = torch.einsum('bij,bkl->bikjl', Q_A, Q_B).reshape(self.num_samples, 81, 4)
        
        rho_q = V_joint.mH @ rho2_bip.unsqueeze(0) @ V_joint
        traces = torch.diagonal(rho_q, dim1=-2, dim2=-1).sum(-1).real
        rho_q = rho_q / (traces.view(-1, 1, 1) + 1e-12)
        
        rho_q_pt = partial_transpose_2q(rho_q)
        eigvals = torch.linalg.eigvalsh(rho_q_pt)
        min_eig_distill = torch.min(eigvals[:, 0])
        
        loss_distill = torch.relu(-min_eig_distill)**2
        
        # Combine losses: The AI must satisfy all three strictly!
        total_loss = 100.0 * loss_distill + 50.0 * loss_npt + 200.0 * loss_full_rank
        return total_loss, min_eig_rho_pt, min_eig_distill, min_eig_rho, rho

def run_full_rank_hunt():
    print("🛡️ Starting Full-Rank 'No Cheating' NPT Hunt 🛡️")
    print("Forcing the AI to build a state with NO zeroes (all eigenvalues >= 0.03).")
    
    model = FullRankNPTHunter(num_samples=2000)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    for step in range(3000):
        optimizer.zero_grad()
        loss, npt_eig, distill_eig, min_rho_eig, rho = model()
        
        loss.backward()
        optimizer.step()
        
        if step % 200 == 0:
            print(f"Step {step:4d} | NPT = {npt_eig.item():.4f} | Distill = {distill_eig.item():.4f} | Min State Eig = {min_rho_eig.item():.4f}")
            sys.stdout.flush()
            
        if distill_eig.item() >= 0 and npt_eig.item() <= -0.05 and min_rho_eig.item() >= 0.03:
            print("\n🎉 FULL-RANK ALGEBRAIC CANDIDATE FOUND! THE AI BEAT THE RULES!")
            np.save("candidate_full_rank.npy", rho.detach().cpu().numpy())
            break
            
    print("\nOptimization Complete.")
    np.set_printoptions(precision=3, suppress=True, linewidth=120)
    print("\nState Diagonal (Notice, no zeros!):")
    print(torch.diagonal(rho).real.detach().cpu().numpy())

if __name__ == "__main__":
    torch.manual_seed(999)
    run_full_rank_hunt()
