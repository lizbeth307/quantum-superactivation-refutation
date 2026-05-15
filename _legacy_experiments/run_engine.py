import argparse
import torch
import numpy as np
from quantum_core import (
    build_erasure_channel, build_smith_yard_ppt, make_tp_kraus,
    batch_kron, evaluate_stinespring_capacity, evaluate_npt_penalty,
    QuantumStorage
)

def run_synthesis(d_in: int, d_out: int, epochs: int, noise_prob: float):
    print("="*60)
    print("  🚀 Quantum Superactivation Engine (Clean Architecture)")
    print(f"  Target: Erasure (p={noise_prob}) | Dim: {d_in} -> {d_out}")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    storage = QuantumStorage()
    
    # Initialize from Smith-Yard manifold
    K_base = build_smith_yard_ppt(d_in).to(device)
    num_k = K_base.shape[0]
    
    # Ensure d_out matches Smith-Yard
    d_out = d_in * d_in
    
    # Parameters to optimize
    delta_K_real = (1e-4 * torch.randn((num_k, d_out, d_in), dtype=torch.float64, device=device)).requires_grad_(True)
    delta_K_imag = (1e-4 * torch.randn((num_k, d_out, d_in), dtype=torch.float64, device=device)).requires_grad_(True)
    
    # T_in must match the joint channel input dimension (d_in * d_in)
    dim_in_state = d_in * d_in
    T_in_real = torch.randn((dim_in_state, dim_in_state), requires_grad=True, dtype=torch.float64, device=device)
    T_in_imag = torch.randn((dim_in_state, dim_in_state), requires_grad=True, dtype=torch.float64, device=device)
    
    optimizer = torch.optim.Adam([
        {'params': [delta_K_real, delta_K_imag], 'lr': 0.005},
        {'params': [T_in_real, T_in_imag], 'lr': 0.05}
    ])
    
    # Build noise channel
    Ker_base = build_erasure_channel(d_in, noise_prob).to(device)
    
    best_ic = -999.0
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        K_raw = K_base + torch.complex(delta_K_real, delta_K_imag)
        Ks = make_tp_kraus(K_raw)
        
        # Build joint space
        K_joint = batch_kron(Ker_base, Ks)
        
        T_in = torch.complex(T_in_real, T_in_imag)
        state_norm = torch.trace(T_in.conj().T @ T_in).real
        T_in_norm = T_in / torch.sqrt(torch.clamp(state_norm, min=1e-12))
        
        Ic = evaluate_stinespring_capacity(K_joint, T_in_norm)
        npt_penalty = evaluate_npt_penalty(Ks, d_out, d_in)
        
        loss = -Ic + 500000.0 * npt_penalty
        loss.backward()
        optimizer.step()
        
        if npt_penalty.item() < 1e-7 and Ic.item() > best_ic:
            best_ic = Ic.item()
            
        if epoch % 10 == 0:
            print(f"Epoch {epoch:4d} | I_c = {Ic.item():+.6f} | NPT Pen = {npt_penalty.item():.2e}")

    if best_ic > 0:
        print(f"🎉 SUPERACTIVATION FOUND: {best_ic:+.6f}")
        storage.save_tensor(f"superactivation_d{d_in}_p{noise_prob}", {
            'Ic': best_ic,
            'Ks': Ks.detach().cpu(),
            'T_in': T_in_norm.detach().cpu()
        })

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Quantum Superactivation Synthesis.")
    parser.add_argument("--din", type=int, default=4, help="Input dimension")
    parser.add_argument("--dout", type=int, default=16, help="Output dimension")
    parser.add_argument("--epochs", type=int, default=400, help="Number of training epochs")
    parser.add_argument("--p", type=float, default=0.5, help="Erasure probability")
    
    args = parser.parse_args()
    run_synthesis(d_in=args.din, d_out=args.dout, epochs=args.epochs, noise_prob=args.p)
