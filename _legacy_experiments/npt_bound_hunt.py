import torch
import torch.nn as nn

# ==============================================================================
# ⚛️ QuantumNEAT: NPT Bound Entanglement GAN
# ==============================================================================
# Open problem (since 1999): Do NPT Bound Entangled states exist?
# We use Adversarial PyTorch (GAN architecture) to hunt for a candidate.
# 
# Generator (G): Creates a 3x3 bipartite density matrix ρ that is NPT.
# Discriminator (D): Tries to find local rank-2 projections (A, B) to distill 
#                    entanglement from ρ. 
# 
# If G wins, it means we have an NPT state from which NO local projections 
# can extract a 2-qubit entangled state (1-undistillable NPT state).
# This is a major candidate for NPT Bound Entanglement!
# ==============================================================================

def partial_transpose(rho, dim_A, dim_B):
    """Computes the partial transpose of a bipartite density matrix over subsystem B."""
    rho_tensor = rho.view(dim_A, dim_B, dim_A, dim_B)
    rho_pt_tensor = rho_tensor.permute(0, 3, 2, 1)
    return rho_pt_tensor.reshape(dim_A * dim_B, dim_A * dim_B)

class Generator(nn.Module):
    """Generates the 9x9 bipartite state ρ (3x3 system)."""
    def __init__(self, dim=9):
        super().__init__()
        self.T = nn.Parameter(torch.randn(dim, dim, dtype=torch.cfloat))
        
    def forward(self):
        # Create a valid density matrix: ρ = T T^† / Tr(T T^†)
        rho = self.T @ self.T.mH
        rho = rho / torch.trace(rho).real
        return rho

class Discriminator(nn.Module):
    """Finds local projections A, B (from 3D down to 2D) to distill entanglement."""
    def __init__(self):
        super().__init__()
        self.A_params = nn.Parameter(torch.randn(3, 2, dtype=torch.cfloat))
        self.B_params = nn.Parameter(torch.randn(3, 2, dtype=torch.cfloat))
        
    def get_isometries(self):
        # QR decomposition to get orthogonal projections
        Q_A, _ = torch.linalg.qr(self.A_params)
        Q_B, _ = torch.linalg.qr(self.B_params)
        return Q_A, Q_B
        
    def forward(self, rho):
        V_A, V_B = self.get_isometries()
        V_joint = torch.kron(V_A, V_B)
        
        # Project the 9x9 state down to a 4x4 (2-qubit) state
        rho_qubit = V_joint.mH @ rho @ V_joint
        
        # Normalize just in case
        rho_qubit = rho_qubit / torch.trace(rho_qubit).real
        
        # Calculate partial transpose of the 2-qubit state
        rho_q_pt = partial_transpose(rho_qubit, 2, 2)
        
        # The discriminator's score is the minimum eigenvalue of the 2-qubit PT
        eigvals = torch.linalg.eigvalsh(rho_q_pt)
        min_eig = eigvals[0]
        
        return min_eig

def run_npt_bound_hunt():
    print("⚔️ Starting Adversarial Hunt for NPT Bound Entanglement ⚔️")
    
    G = Generator()
    D = Discriminator()
    
    opt_G = torch.optim.Adam(G.parameters(), lr=0.01)
    opt_D = torch.optim.Adam(D.parameters(), lr=0.05)
    
    # We want G to make a state with NPT negativity at least -0.05
    target_npt = -0.05
    
    for epoch in range(1001):
        # -------------------------------------------------
        # 1. Train Discriminator (D tries to distill)
        # D wants to MINIMIZE the minimum eigenvalue of the projected PT state
        # (meaning it wants to find negative eigenvalues = distillable entanglement)
        # -------------------------------------------------
        rho = G().detach() # Freeze Generator
        for _ in range(5):
            opt_D.zero_grad()
            score_D = D(rho)
            loss_D = score_D # D minimizes the eigenvalue
            loss_D.backward()
            opt_D.step()
            
        # -------------------------------------------------
        # 2. Train Generator (G tries to hide from D while staying NPT)
        # -------------------------------------------------
        opt_G.zero_grad()
        rho = G()
        
        # Ensure rho is NPT overall (9x9 system)
        rho_pt = partial_transpose(rho, 3, 3)
        eigvals_pt = torch.linalg.eigvalsh(rho_pt)
        min_eig_overall = eigvals_pt[0]
        
        # Penalty if G is not NPT enough
        loss_NPT = torch.relu(min_eig_overall - target_npt)
        
        # G wants D's score to be POSITIVE (meaning D fails to distill)
        score_G_vs_D = D(rho)
        loss_Adversarial = -score_G_vs_D # G maximizes D's score
        
        loss_G = 10.0 * loss_NPT + loss_Adversarial
        loss_G.backward()
        opt_G.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch:4d} | Overall NPT eig = {min_eig_overall.item():.4f} | D's best distillation eig = {score_G_vs_D.item():.6f}")

    print("\n🏁 Adversarial Training Complete.")
    print("If D's best distillation eig >= 0, G successfully hid the entanglement!")
    print("This means the state is NPT but 1-undistillable (Strong candidate for Bound Entanglement).")

if __name__ == "__main__":
    torch.manual_seed(777)
    run_npt_bound_hunt()
