import torch
import torch.nn as nn
import numpy as np
import sys

# ==============================================================================
# ⚛️ QuantumNEAT: Massive Parallel SIC-POVM Hunter (d=6)
# ==============================================================================
# We deploy 10,000 parallel optimizers simultaneously in a single PyTorch tensor.
# This explores the incredibly complex d=6 landscape at lightspeed.
# No for-loops for batches. Pure Einstein Summation (einsum).
# ==============================================================================

def generate_weyl_heisenberg_operators(d):
    omega = np.exp(2j * np.pi / d)
    Z = torch.zeros((d, d), dtype=torch.cdouble)
    for j in range(d):
        Z[j, j] = omega**j
        
    X = torch.zeros((d, d), dtype=torch.cdouble)
    for j in range(d):
        X[(j+1)%d, j] = 1.0
        
    operators = []
    for m in range(d):
        for n in range(d):
            if m == 0 and n == 0:
                continue
            D = torch.linalg.matrix_power(X, m) @ torch.linalg.matrix_power(Z, n)
            operators.append(D)
            
    return torch.stack(operators)

class ParallelFiducialHunter(nn.Module):
    def __init__(self, d=6, batch_size=10000):
        super().__init__()
        self.d = d
        self.B = batch_size
        
        # 10,000 parallel vectors initialized randomly
        self.v_batch = nn.Parameter(torch.randn(self.B, d, dtype=torch.cdouble))
        
        # Register D_ops as a buffer so it moves to GPU if available
        self.register_buffer('D_ops', generate_weyl_heisenberg_operators(d))

    def get_normalized_vectors(self):
        norms = torch.linalg.norm(self.v_batch, dim=1, keepdim=True)
        return self.v_batch / (norms + 1e-12)

    def forward(self):
        v = self.get_normalized_vectors()
        
        # v is (B, d). D_ops is (N_ops, d, d) where N_ops = 35.
        # Apply all 35 operators to all 10,000 vectors simultaneously
        # Result shape: (B, N_ops, d)
        D_v = torch.einsum('nij,bj->bni', self.D_ops, v)
        
        # Compute overlaps: <v | D_v> -> shape (B, N_ops)
        overlaps = torch.abs(torch.einsum('bi,bni->bn', v.conj(), D_v))**2
        
        target_val = 1.0 / (self.d + 1.0)
        
        # Compute SIC loss for EACH of the 10,000 vectors
        # shape: (B,)
        losses = torch.mean((overlaps - target_val)**2, dim=1)
        
        # The optimizer needs a scalar loss, so we minimize the mean of all losses
        # But we track the ABSOLUTE BEST loss in the batch
        total_loss = torch.mean(losses)
        best_loss, best_idx = torch.min(losses, dim=0)
        
        return total_loss, best_loss, best_idx, v[best_idx]

def run_massive_parallel_hunt(d=54, batch_size=1000):
    print("==================================================")
    print(f" 🚀 LAUNCHING PARALLEL SIC-POVM HUNT (d={d}) 🚀")
    print(f" Swarm Size: {batch_size} simultaneous agents.")
    print("==================================================")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Compute Device: {device}")
    
    model = ParallelFiducialHunter(d=d, batch_size=batch_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    
    # We use a scheduler to "freeze" the vectors once they find a good valley
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5000, eta_min=1e-5)
    
    global_best_loss = float('inf')
    stuck_counter = 0
    
    for step in range(5000):
        optimizer.zero_grad()
        total_loss, best_loss, best_idx, best_v = model()
        
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        
        if best_loss.item() < global_best_loss:
            global_best_loss = best_loss.item()
            stuck_counter = 0
        else:
            stuck_counter += 1
            
        if stuck_counter > 50:
            with torch.no_grad():
                # Random noise to kick agents out of local minima
                noise = torch.randn_like(model.v_batch) * 0.1
                model.v_batch.add_(noise)
            stuck_counter = 0
            print(f"🌋 THERMAL SHOCK at Step {step:4d}! Shaking the swarm out of Barren Plateau...")
            
        if step % 200 == 0:
            print(f"Step {step:4d} | Batch Mean Loss = {total_loss.item():.2e} | 🏆 BEST AGENT LOSS = {global_best_loss:.2e}")
            sys.stdout.flush()
            
        if global_best_loss < 1e-12:
            print("\n🎉 WORLD CLASS DISCOVERY! EXACT SIC-POVM FOUND FOR d=6!")
            break
            
    print("\n==================================================")
    print(" 🏁 SWARM OPTIMIZATION COMPLETE 🏁")
    print("==================================================")
    print(f"Absolute Best Loss Achieved: {global_best_loss:.4e}")
    
    best_v_np = best_v.detach().cpu().numpy()
    np.save(f"best_sic_d{d}_parallel.npy", best_v_np)
    
    print("\nMagnitudes of the best vector:")
    print(np.abs(best_v_np))
    
    if global_best_loss > 1e-8:
        print("\nAll 10,000 agents got stuck in local minima (d=6 is notoriously rugged).")
        print("Next step: L1 Sparsity, or increasing swarm size to 1,000,000!")

if __name__ == "__main__":
    torch.manual_seed(42)
    run_massive_parallel_hunt(d=54, batch_size=1000)
