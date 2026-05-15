import torch
import torch.nn as nn
import numpy as np

import sys

# ==============================================================================
# ⚛️ QuantumNEAT: Sparse 2-Undistillable NPT Hunt
# ==============================================================================
# To find an algebraic formula, we need a CLEAN matrix, not a noisy one.
# We add an L1 penalty to force the matrix to be as sparse as possible (lots of zeros).
# If the AI finds a state that is mostly zeros but still NPT and undistillable,
# we can look at the remaining numbers and derive the exact formula!
# ==============================================================================

def partial_transpose_2q(rho_batch):
    rt = rho_batch.view(-1, 2, 2, 2, 2)
    rt_pt = rt.permute(0, 1, 4, 3, 2)
    return rt_pt.reshape(-1, 4, 4)

class SparseNPTHunter(nn.Module):
    def __init__(self, dim=9, num_samples=2000):
        super().__init__()
        self.dim = dim
        self.num_samples = num_samples
        self.T = nn.Parameter(torch.randn(dim, dim, dtype=torch.cfloat))

    def forward(self):
        rho = self.T @ self.T.mH
        rho = rho / torch.trace(rho).real
        
        # --- NPT Penalty ---
        rt = rho.view(3, 3, 3, 3)
        rt_pt = rt.permute(0, 3, 2, 1).reshape(9, 9)
        min_eig_rho = torch.linalg.eigvalsh(rt_pt)[0]
        loss_npt = torch.relu(min_eig_rho - (-0.05))**2
        
        # --- L1 Sparsity Penalty (The key to finding a formula) ---
        # We penalize the absolute values of the matrix elements.
        # This forces the network to push irrelevant numbers to exactly 0.0.
        loss_sparsity = torch.mean(torch.abs(rho))
        
        # --- Distillation Penalty ---
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
        
        # Combine losses: aggressively force sparsity
        total_loss = 100.0 * loss_distill + 50.0 * loss_npt + 25.0 * loss_sparsity
        return total_loss, min_eig_rho, min_eig_distill, rho

def run_sparse_hunt():
    print("🔍 Starting Sparse Pattern Recognition Hunt 🔍")
    print("Forcing the AI to strip away noise and reveal the pure mathematical structure...")
    
    model = SparseNPTHunter()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    for step in range(3000):
        optimizer.zero_grad()
        loss, npt_eig, distill_eig, rho = model()
        
        loss.backward()
        optimizer.step()
        
        if step % 250 == 0:
            print(f"Step {step:4d} | NPT = {npt_eig.item():.4f} | Distill = {distill_eig.item():.4f} | Sparsity = {torch.mean(torch.abs(rho)).item():.4f}")
            sys.stdout.flush()
            
        if distill_eig.item() >= 0 and npt_eig.item() <= -0.05 and torch.mean(torch.abs(rho)).item() < 0.03:
            print("\n🎉 EXTREME SPARSE ALGEBRAIC CANDIDATE FOUND!")
            break
            
    print("\nOptimization Complete.")
    
    # Thresholding: snap small values to 0 to reveal the clean skeleton
    clean_rho = rho.detach().cpu().numpy()
    clean_rho[np.abs(clean_rho) < 0.015] = 0.0
    np.save("candidate_extreme_sparse.npy", clean_rho)
    
    print("\nCleaned Skeleton Matrix (Real Part):")
    np.set_printoptions(precision=3, suppress=True, linewidth=120)
    print(clean_rho.real)

if __name__ == "__main__":
    torch.manual_seed(42)
    run_sparse_hunt()
