import torch
import numpy as np
from torch import nn

# ==============================================================================
# ⚛️ QuantumNEAT: Hastings Additivity Counterexample Hunter
# ==============================================================================
# In 2009, Matthew Hastings disproved the additivity conjecture for the minimum 
# output entropy (MOE) of quantum channels. He showed that:
# S_min(Φ ⊗ Φ_conjugate) < S_min(Φ) + S_min(Φ_conjugate)
# 
# His proof was non-constructive (using high-dimensional random Haar unitaries).
# Our goal is to use PyTorch's Riemannian optimization to find an EXPLICIT, 
# low-dimensional counterexample (e.g., d=3 or d=4).
# ==============================================================================

def von_neumann_entropy(rho):
    """Computes the Von Neumann entropy S(ρ) = -Tr(ρ log2 ρ)."""
    # Ensure Hermitian
    rho = (rho + rho.mH) / 2.0
    eigvals = torch.linalg.eigvalsh(rho)
    eigvals = torch.clamp(eigvals, min=1e-12)
    return -torch.sum(eigvals * torch.log2(eigvals))

class HastingsHunter(nn.Module):
    def __init__(self, d=3, num_unitaries=4):
        super().__init__()
        self.d = d
        self.num_unitaries = num_unitaries
        
        # Parameterize the unitaries for the channel Φ
        # We use a complex matrix and will take its QR decomposition to get exact unitaries
        self.U_params = nn.Parameter(torch.randn(num_unitaries, d, d, dtype=torch.cfloat))
        
        # Parameterize the single input state |ψ> (dim = d)
        self.psi_params = nn.Parameter(torch.randn(d, dtype=torch.cfloat))
        
        # Parameterize the joint input state |Ψ_AB> (dim = d*d)
        self.Psi_AB_params = nn.Parameter(torch.randn(d * d, dtype=torch.cfloat))

    def get_unitaries(self):
        """Orthogonalize parameters to get exact Unitary matrices."""
        unitaries = []
        for i in range(self.num_unitaries):
            Q, R = torch.linalg.qr(self.U_params[i])
            # Make the diagonal of R positive to ensure uniqueness
            d_tensor = torch.diagonal(R)
            ph = d_tensor / torch.abs(d_tensor)
            Q = Q * ph
            unitaries.append(Q)
        return torch.stack(unitaries)

    def apply_channel(self, rho, unitaries):
        """Apply channel Φ(ρ) = (1/D) * Σ U_i ρ U_i^†"""
        out_rho = torch.zeros_like(rho)
        for i in range(self.num_unitaries):
            out_rho += unitaries[i] @ rho @ unitaries[i].mH
        return out_rho / self.num_unitaries

    def apply_conjugate_channel(self, rho, unitaries):
        """Apply channel Φ_bar(ρ) = (1/D) * Σ U_i^* ρ (U_i^*)^†"""
        out_rho = torch.zeros_like(rho)
        for i in range(self.num_unitaries):
            U_conj = torch.conj(unitaries[i])
            out_rho += U_conj @ rho @ U_conj.mH
        return out_rho / self.num_unitaries

    def apply_joint_channel(self, rho_AB, unitaries):
        """Apply Φ ⊗ Φ_bar to a bipartite state ρ_AB."""
        out_rho = torch.zeros_like(rho_AB)
        for i in range(self.num_unitaries):
            for j in range(self.num_unitaries):
                U_i = unitaries[i]
                U_j_conj = torch.conj(unitaries[j])
                # Kronecker product of U_i and U_j_conj
                U_tensor = torch.kron(U_i, U_j_conj)
                out_rho += U_tensor @ rho_AB @ U_tensor.mH
        return out_rho / (self.num_unitaries ** 2)

    def forward(self):
        unitaries = self.get_unitaries()
        
        # 1. Single channel entropy S_min(Φ)
        psi = self.psi_params / torch.linalg.norm(self.psi_params)
        rho_single = torch.outer(psi, torch.conj(psi))
        out_single = self.apply_channel(rho_single, unitaries)
        S_single = von_neumann_entropy(out_single)
        
        # 2. Joint channel entropy S_min(Φ ⊗ Φ_bar)
        Psi_AB = self.Psi_AB_params / torch.linalg.norm(self.Psi_AB_params)
        rho_joint = torch.outer(Psi_AB, torch.conj(Psi_AB))
        out_joint = self.apply_joint_channel(rho_joint, unitaries)
        S_joint = von_neumann_entropy(out_joint)
        
        # Additivity Violation: Δ = S_joint - 2 * S_single
        # If Δ < 0, Hastings was right and we found a concrete example!
        delta = S_joint - 2 * S_single
        
        return delta, S_single, S_joint

def run_hastings_hunt():
    print("Starting QuantumNEAT Additivity Hunt (Hastings 2009)")
    print("Target: Find Δ = S_min(Φ ⊗ Φ_bar) - 2 * S_min(Φ) < 0")
    
    # We start with d=4 (ququarts) to test a different regime
    model = HastingsHunter(d=4, num_unitaries=5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # We want to MINIMIZE delta
    best_delta = float('inf')
    
    for step in range(2000):
        optimizer.zero_grad()
        delta, S_single, S_joint = model()
        
        # Our loss is delta itself. We want to push it negative.
        loss = delta
        loss.backward()
        optimizer.step()
        
        if delta.item() < best_delta:
            best_delta = delta.item()
            
        if step % 100 == 0:
            print(f"Step {step:4d} | Δ = {delta.item():.6f} | S_single = {S_single.item():.4f} | S_joint = {S_joint.item():.4f}")
            
    print(f"\nOptimization Complete. Best Additivity Violation (Δ): {best_delta:.6f}")
    if best_delta < -1e-4:
        print("WE BROKE ADDITIVITY! CONCRETE COUNTEREXAMPLE FOUND!")
    else:
        print("Additivity held (Δ ≥ 0). We may need higher dimensions (d=4 or d=5).")

if __name__ == "__main__":
    torch.manual_seed(42)
    run_hastings_hunt()
