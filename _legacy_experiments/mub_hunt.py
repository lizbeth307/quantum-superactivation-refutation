import torch
import torch.nn as nn
import math

# ==============================================================================
# ⚛️ QuantumNEAT: The MUBs in d=6 Hunter
# ==============================================================================
# Open problem: Does there exist a set of 4 (or up to 7) Mutually Unbiased Bases 
# (MUBs) in dimension d=6? 
# So far, humanity has only found 3.
#
# Condition: Two orthonormal bases A and B are MUBs if for all i,j:
# |<A_i | B_j>|^2 = 1/d
# 
# We fix Basis 0 to be the Identity matrix. We optimize Unitaries U_1, U_2...
# Loss = sum_{a<b} sum_{i,j} ( |(U_a^dagger U_b)_{i,j}|^2 - 1/d )^2
# ==============================================================================

class MUBHunter(nn.Module):
    def __init__(self, d=6, num_bases=4):
        super().__init__()
        self.d = d
        self.num_bases = num_bases
        
        # We need (num_bases - 1) unitaries, since U_0 is fixed as Identity
        self.U_params = nn.Parameter(torch.randn(num_bases - 1, d, d, dtype=torch.cfloat))

    def get_unitaries(self):
        """Map unconstrained parameters to exact Unitary matrices via QR decomposition."""
        unitaries = [torch.eye(self.d, dtype=torch.cfloat)] # U_0 is Identity
        
        for k in range(self.num_bases - 1):
            Q, R = torch.linalg.qr(self.U_params[k])
            # Ensure uniqueness of QR to make gradient descent smooth
            d_tensor = torch.diagonal(R)
            ph = d_tensor / torch.abs(d_tensor + 1e-12)
            Q = Q * ph
            unitaries.append(Q)
            
        return torch.stack(unitaries)

    def forward(self):
        unitaries = self.get_unitaries()
        target_val = 1.0 / self.d
        total_loss = 0.0
        
        # Check all pairs of bases (a < b)
        for a in range(self.num_bases):
            for b in range(a + 1, self.num_bases):
                # Inner products are elements of U_a^dagger @ U_b
                V = unitaries[a].mH @ unitaries[b]
                
                # Modulus squared of each element
                P = torch.abs(V)**2
                
                # Deviation from 1/d
                loss_matrix = (P - target_val)**2
                total_loss += torch.sum(loss_matrix)
                
        return total_loss, unitaries

def run_mub_hunt(d=6, target_bases=4):
    print(f"🚀 Hunting for {target_bases} MUBs in dimension d={d}")
    print(f"Target: Minimizing Overlap Variance (Loss -> 0)")
    
    model = MUBHunter(d=d, num_bases=target_bases)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    best_loss = float('inf')
    
    for step in range(5000):
        optimizer.zero_grad()
        loss, _ = model()
        
        loss.backward()
        optimizer.step()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            
        if step % 500 == 0:
            print(f"Step {step:4d} | Loss = {loss.item():.8f}")
            
        # If loss is functionally zero, we found them!
        if loss.item() < 1e-10:
            print(f"\n🎉 EXACT MATCH FOUND AT STEP {step}! LOSS = {loss.item()}")
            break
            
    print(f"\nOptimization Complete. Best Loss: {best_loss:.8f}")
    if best_loss < 1e-6:
        print(f"🚨 WE FOUND {target_bases} MUBs IN d={d}! THIS IS A SCIENTIFIC BREAKTHROUGH! 🚨")
    else:
        print(f"Failed to find exact {target_bases} MUBs. Loss plateaued. The optimizer is stuck in a local minimum, or they don't exist.")

if __name__ == "__main__":
    torch.manual_seed(101)
    # Start by trying to find 4 MUBs in d=6 (current world record is 3)
    run_mub_hunt(d=6, target_bases=4)
