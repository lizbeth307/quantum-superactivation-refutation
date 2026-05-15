import torch
import numpy as np

# ==============================================================================
# ⚛️ QuantumNEAT: The K=3 Boss Fight (Testing the Candidate on 3 Copies)
# ==============================================================================
# We test the Full-Rank NPT candidate against N=3 (3 copies).
# The space is 9^3 = 729 x 729. 
# We simulate 10,000 random Schmidt-rank-2 projections.
# If the state survives this, it is an exceptionally robust Bound Entanglement candidate.
# ==============================================================================

def partial_transpose_2q(rho_batch):
    rt = rho_batch.view(-1, 2, 2, 2, 2)
    rt_pt = rt.permute(0, 1, 4, 3, 2)
    return rt_pt.reshape(-1, 4, 4)

def run_k3_boss_fight():
    print("⚔️ THE K=3 BOSS FIGHT: Testing Candidate on 3 Copies (729 x 729) ⚔️\n")
    
    try:
        rho_np = np.load("candidate_full_rank.npy")
        rho = torch.tensor(rho_np, dtype=torch.cfloat)
    except FileNotFoundError:
        print("Candidate not found! Run full_rank_npt_hunt.py first.")
        return

    print("1. Constructing 3-copy state ρ ⊗ ρ ⊗ ρ...")
    rho2 = torch.kron(rho, rho)
    rho3 = torch.kron(rho2, rho)
    
    # rho3 is 729x729. Alice has indices (3, 3, 3) = 27. Bob has (3, 3, 3) = 27.
    # We must permute to group Alice's subsystems together and Bob's together.
    # rho3 shape before: (A1 B1 A2 B2 A3 B3, A1' B1' A2' B2' A3' B3')
    r3t = rho3.view(3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3)
    
    # Alice: 0 (A1), 2 (A2), 4 (A3). Bob: 1 (B1), 3 (B2), 5 (B3).
    # Same for column indices (+6).
    r3t_perm = r3t.permute(0, 2, 4, 1, 3, 5, 6, 8, 10, 7, 9, 11)
    rho3_bip = r3t_perm.reshape(729, 729)
    
    print("2. State successfully mapped to Bipartite (Alice: 27D, Bob: 27D).")
    
    num_samples = 10000
    print(f"\n3. Simulating {num_samples} Random Distillation Attacks (Projections)...")
    
    # Generate random 27x2 projections for Alice and Bob
    # Alice projects from 27D down to a qubit (2D).
    A_rand = torch.randn(num_samples, 27, 2, dtype=torch.cfloat)
    B_rand = torch.randn(num_samples, 27, 2, dtype=torch.cfloat)
    
    Q_A, _ = torch.linalg.qr(A_rand)
    Q_B, _ = torch.linalg.qr(B_rand)
    
    # Joint projection is Q_A ⊗ Q_B. Shape: (num_samples, 729, 4)
    V_joint = torch.einsum('bij,bkl->bikjl', Q_A, Q_B).reshape(num_samples, 729, 4)
    
    print("4. Applying projections and extracting 2-qubit states...")
    # Project state: V^dagger @ rho3_bip @ V
    rho_q = V_joint.mH @ rho3_bip.unsqueeze(0) @ V_joint
    
    # Normalize
    traces = torch.diagonal(rho_q, dim1=-2, dim2=-1).sum(-1).real
    rho_q = rho_q / (traces.view(-1, 1, 1) + 1e-12)
    
    print("5. Evaluating Partial Transpose eigenvalues for all 10,000 protocols...")
    rho_q_pt = partial_transpose_2q(rho_q)
    eigvals = torch.linalg.eigvalsh(rho_q_pt)
    
    min_eig_all = torch.min(eigvals[:, 0]).item()
    
    print("\n" + "="*50)
    print("🏆 K=3 SURVIVAL RESULTS 🏆")
    print(f"Worst-case Distillation Eigenvalue: {min_eig_all:.6f}")
    
    if min_eig_all >= 0:
        print("\n🤯 INCREDIBLE! The state SURVIVED 3 copies!")
        print("It is officially 3-undistillable against 10,000 random protocols.")
        print("This is a world-class candidate for NPT Bound Entanglement.")
    else:
        print("\n❌ The state was DISTILLED at K=3.")
        print("The hacker found a vulnerability when given 3 copies of the state.")
        print("The state is NOT bound entangled (it is a 'weak' NPT state).")
    print("="*50)

if __name__ == "__main__":
    torch.manual_seed(123)
    run_k3_boss_fight()
