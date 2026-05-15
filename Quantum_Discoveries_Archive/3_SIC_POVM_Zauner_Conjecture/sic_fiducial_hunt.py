import torch
import torch.nn as nn
import numpy as np
import sys

# ==============================================================================
# ⚛️ QuantumNEAT: Weyl-Heisenberg Fiducial Vector Hunter
# ==============================================================================
# Instead of searching for d^2 vectors, we search for exactly ONE vector |ψ_0>.
# We apply the Weyl-Heisenberg displacement operators X^m Z^n to generate the rest.
# We also apply an L1 Sparsity Penalty to extract the algebraic roots!
# ==============================================================================

def generate_weyl_heisenberg_operators(d):
    """Generate the d^2 Weyl-Heisenberg operators X^m Z^n."""
    # Clock matrix Z
    omega = np.exp(2j * np.pi / d)
    Z = torch.zeros((d, d), dtype=torch.cfloat)
    for j in range(d):
        Z[j, j] = omega**j
        
    # Shift matrix X
    X = torch.zeros((d, d), dtype=torch.cfloat)
    for j in range(d):
        X[(j+1)%d, j] = 1.0
        
    operators = []
    # Generate all combinations of X^m Z^n (excluding m=0, n=0 which is Identity)
    for m in range(d):
        for n in range(d):
            if m == 0 and n == 0:
                continue
            
            X_m = torch.linalg.matrix_power(X, m)
            Z_n = torch.linalg.matrix_power(Z, n)
            D = X_m @ Z_n
            operators.append(D)
            
    return torch.stack(operators)

class FiducialHunter(nn.Module):
    def __init__(self, d=6):
        super().__init__()
        self.d = d
        # A single vector parameterized by d complex numbers
        self.v_params = nn.Parameter(torch.randn(d, dtype=torch.cfloat))
        
        # Pre-compute the Weyl-Heisenberg group (excluding Identity)
        self.D_ops = generate_weyl_heisenberg_operators(d)

    def get_normalized_vector(self):
        norm = torch.linalg.norm(self.v_params)
        return self.v_params / (norm + 1e-12)

    def forward(self):
        v = self.get_normalized_vector()
        
        # Calculate overlaps: |<v | D_mn | v>|^2 for all D_mn
        # D_ops is shape (d^2-1, d, d), v is shape (d,)
        # D_v = D_ops @ v -> shape (d^2-1, d)
        D_v = torch.einsum('nij,j->ni', self.D_ops.to(v.device), v)
        
        # Overlaps = |v.mH @ D_v|^2
        overlaps = torch.abs(torch.einsum('i,ni->n', v.conj(), D_v))**2
        
        target_val = 1.0 / (self.d + 1.0)
        
        # Deviation from exact SIC-POVM condition
        loss_sic = torch.mean((overlaps - target_val)**2)
        
        # L1 Sparsity Penalty: force absolute values of the components to be zero or very uniform
        # We penalize the L1 norm of the amplitudes to make the vector sparse
        loss_sparsity = torch.mean(torch.abs(v))
        
        # We start with pure SIC loss, and add sparsity slowly
        return loss_sic, loss_sparsity, v

def run_fiducial_hunt(d=6):
    print(f"🌟 Starting Fiducial Vector Hunter for d={d} 🌟")
    print("Using Weyl-Heisenberg symmetry + L1 Sparsity Penalty.")
    
    model = FiducialHunter(d=d)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    for step in range(15000):
        optimizer.zero_grad()
        loss_sic, loss_sparsity, v = model()
        
        # Increase sparsity penalty gradually
        sparsity_weight = min(0.05, step / 100000.0) 
        total_loss = loss_sic + sparsity_weight * loss_sparsity
        
        total_loss.backward()
        optimizer.step()
        
        if step % 1000 == 0:
            print(f"Step {step:5d} | SIC Loss = {loss_sic.item():.2e} | Sparsity = {loss_sparsity.item():.4f}")
            sys.stdout.flush()
            
        if loss_sic.item() < 1e-12:
            print(f"\n🎉 EXACT ALGEBRAIC FIDUCIAL VECTOR FOUND! SIC Loss = {loss_sic.item():.2e}")
            break
            
    print("\nOptimization Complete.")
    v_clean = v.detach().cpu().numpy()
    
    # Thresholding to reveal zeroes (Sparsity)
    v_clean[np.abs(v_clean) < 1e-4] = 0.0
    
    np.set_printoptions(precision=4, suppress=True)
    print("\nFiducial Vector Amplitudes (Magnitudes):")
    print(np.abs(v_clean))
    print("\nFiducial Vector (Complex):")
    print(v_clean)
    
    np.save(f"fiducial_d{d}.npy", v_clean)

if __name__ == "__main__":
    torch.manual_seed(999)
    run_fiducial_hunt(d=6)
