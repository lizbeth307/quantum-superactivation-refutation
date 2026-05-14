import sys
import os
import torch
import numpy as np
import asyncio
from starlette.websockets import WebSocketState
import json

from quantum_core import (
    build_depolarizing_channel, build_amplitude_damping_channel, 
    build_erasure_channel, build_smith_yard_ppt, 
    build_phase_damping_channel, build_black_hole_channel, build_wormhole_channel,
    make_tp_kraus, evaluate_stinespring_capacity, evaluate_npt_penalty
)

def project_to_trace_preserving(K_raw, d, device):
    """Wrapper to match optimizer's expected tuple return and device argument."""
    # We ignore d and device since make_tp_kraus handles them internally
    K_tp = make_tp_kraus(K_raw)
    # Return dummy sum_K_K_raw to satisfy the optimizer unpacking
    return K_tp, torch.zeros((d, d), device=device)

# Retrocausal / Metrology placeholders if not in core yet
def evaluate_retrocausal_fidelity(K, T_in, d, device):
    # Dummy implementation for retrocausal
    return evaluate_stinespring_capacity(K, T_in) * 0.9

def evaluate_metrology_distance(K, T_in, d, device):
    # Dummy implementation for metrology
    return evaluate_stinespring_capacity(K, T_in) * 1.1

async def run_optimization_loop(websocket, params, force_cpu: bool = False):
    try:
        d = int(params.get("d", 12))
        energy_limit = float(params.get("energy", 15.0))
        p = float(params.get("p", 0.5))
        epochs = int(params.get("epochs", 100))
        use_ancilla = bool(params.get("ancilla", False))
        noise_model = params.get("noise", "erasure")
        topology = params.get("topology", "bipartite")
        objective = params.get("objective", "capacity")
            
        if force_cpu:
            device = "cpu"
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if noise_model == "wormhole":
            base_in = d * d
        else:
            base_in = d
            
        if objective == "additivity_violation":
            if topology == "tripartite":
                dim_in = base_in * base_in * base_in
            else:
                dim_in = base_in * base_in
        else:
            if topology == "tripartite":
                dim_in = base_in * d * d
            else:
                dim_in = base_in * d
        
        await websocket.send_text(json.dumps({"type": "log", "message": f"Initializing Topology={topology}, Noise={noise_model}, d={d}, p={p}..."}))
        await asyncio.sleep(0.1)
        
        # Initialize K_PPT
        if noise_model == "amplitude_damping":
            K_PPT_initial = build_smith_yard_ppt(d).to(device)
            num_PPT = K_PPT_initial.shape[0]
            noise = torch.randn_like(K_PPT_initial) * 0.1
            K_PPT_initial = K_PPT_initial + noise
            K_PPT_initial, _ = project_to_trace_preserving(K_PPT_initial, d, device)
        else:
            K_PPT_initial = build_smith_yard_ppt(d).to(device)
            num_PPT = K_PPT_initial.shape[0]
        
        K_PPT_real = K_PPT_initial.real.clone().requires_grad_(True)
        K_PPT_imag = K_PPT_initial.imag.clone().requires_grad_(True)
        
        # Initialize K_Era (noisy channel)
        if noise_model == "erasure":
            K_Era = build_erasure_channel(d, p=p).to(device)
        elif noise_model == "amplitude_damping":
            K_Era = build_amplitude_damping_channel(d, p=p).to(device)
        elif noise_model == "phase_damping":
            K_Era = build_phase_damping_channel(d, p=p).to(device)
        elif noise_model == "black_hole":
            K_Era = build_black_hole_channel(d, p=p).to(device)
        elif noise_model == "wormhole":
            K_Era = build_wormhole_channel(d, p=p).to(device)
        elif noise_model == "ppt_channel":
            K_Era = None # Will be set dynamically to K_PPT_TP
        else:
            K_Era = build_depolarizing_channel(d, p=p).to(device)
            
        def batch_kron(A, B):
            Na, da_out, da_in = A.shape
            Nb, db_out, db_in = B.shape
            A_exp = A.view(Na, 1, da_out, 1, da_in, 1)
            B_exp = B.view(1, Nb, 1, db_out, 1, db_in)
            return (A_exp * B_exp).reshape(Na * Nb, da_out * db_out, da_in * db_in)
        
        # Initialize T_in
        if use_ancilla:
            dim_in_state = dim_in * d
        else:
            dim_in_state = dim_in
            
        rank_in = dim_in_state
        T_in_real_val = torch.randn((dim_in_state, rank_in), dtype=torch.float64, device=device) * 0.1
        T_in_imag_val = torch.randn((dim_in_state, rank_in), dtype=torch.float64, device=device) * 0.1
        
        T_in_real = T_in_real_val.requires_grad_(True)
        T_in_imag = T_in_imag_val.requires_grad_(True)
        
        optimizer = torch.optim.Adam([T_in_real, T_in_imag, K_PPT_real, K_PPT_imag], lr=0.01)
        
        best_ic = -999.0
        
        for epoch in range(epochs):
            # Yield to event loop and check if client disconnected
            await asyncio.sleep(0)
            if websocket.client_state == WebSocketState.DISCONNECTED:
                print("Halt signal received. Stopping optimization loop.")
                break
                
            optimizer.zero_grad()
            
            K_PPT = torch.complex(K_PPT_real, K_PPT_imag)
            
            K_PPT_TP, sum_K_K_raw = project_to_trace_preserving(K_PPT, d, device)
            
            K_Noise = K_Era if K_Era is not None else K_PPT_TP
            
            if objective == "additivity_violation":
                # The Holy Grail: Q(N x N). We don't use PPT channel. We test two identical noise channels.
                if topology == "tripartite":
                    Ks_part = batch_kron(K_Noise, K_Noise)
                    Ks = batch_kron(Ks_part, K_Noise)
                else:
                    Ks = batch_kron(K_Noise, K_Noise)
            else:
                if topology == "tripartite":
                    Ks_part = batch_kron(K_Noise, K_PPT_TP)
                    Ks = batch_kron(Ks_part, K_PPT_TP)
                else:
                    Ks = batch_kron(K_Noise, K_PPT_TP)
                    
            if use_ancilla:
                # To use an Ancilla, we tensor the channel with an Identity channel of dimension d.
                # This allows the AI to encode into a larger space A x Ancilla.
                K_Id = torch.eye(d, dtype=torch.complex128, device=device).unsqueeze(0)
                Ks = batch_kron(Ks, K_Id)
            
            T_in = torch.complex(T_in_real, T_in_imag)
            # Normalize T_in perfectly
            state_norm = torch.trace(T_in.conj().T @ T_in).real
            T_in_norm = T_in / torch.sqrt(torch.clamp(state_norm, min=1e-12))
            
            if objective == "metrology":
                Ic = evaluate_metrology_distance(Ks, T_in_norm, d, device)
            elif objective == "retrocausality":
                Ic = evaluate_retrocausal_fidelity(Ks, T_in_norm, d, device)
            elif objective == "additivity_violation":
                Ic = evaluate_stinespring_capacity(Ks, T_in_norm)
            else:
                Ic = evaluate_stinespring_capacity(Ks, T_in_norm)
            
            npt_penalty = evaluate_npt_penalty(K_PPT_TP, d*d, d)
            
            # Calculate the real difference for telemetry
            sum_K_K_TP = torch.zeros((d, d), dtype=torch.complex128, device=device)
            for k in range(num_PPT):
                sum_K_K_TP += K_PPT_TP[k].conj().T @ K_PPT_TP[k]
            tp_penalty = torch.norm(sum_K_K_TP - torch.eye(d, device=device))
            
            # Calculate Energy Penalty
            # T_in_norm has shape (dim_in_state, rank_in)
            T_tensor = T_in_norm.reshape(d, T_in_norm.shape[0] // d, T_in_norm.shape[1])
            # rho_A = Tr_R(T T^dagger) -> sum over j (R) and r (rank)
            rho_A = torch.einsum('ijr, kjr -> ik', T_tensor, T_tensor.conj())
            probs = torch.real(torch.diag(rho_A))
            
            n_hat = torch.arange(d, dtype=torch.float64, device=device)
            avg_energy = torch.sum(n_hat * probs)
            
            energy_penalty = 0.0
            if energy_limit < 14.9:
                energy_penalty = torch.relu(avg_energy - energy_limit)**2
                
            # Use MASSIVE penalties to strictly enforce physics laws
            loss = -Ic + 1000000.0 * npt_penalty + 100000.0 * tp_penalty + 100.0 * energy_penalty
            loss.backward()
            if torch.isnan(K_PPT_real.grad).any():
                print(f"CRITICAL: K_PPT_real.grad contains NaN at epoch {epoch}!")
            optimizer.step()
            
            is_best = False
            if npt_penalty.item() < 1e-6 and tp_penalty.item() < 1e-4 and Ic.item() > best_ic and (energy_limit >= 14.9 or avg_energy.item() <= energy_limit + 0.1):
                best_ic = Ic.item()
                is_best = True
                
                # Save the absolute best matrices for rigorous verification
                torch.save({
                    'K_PPT': K_PPT_TP.detach(),
                    'T_in': T_in_norm.detach(),
                    'Ic': Ic.item(),
                    'npt_penalty': npt_penalty.item(),
                    'tp_penalty': tp_penalty.item(),
                    'd': d,
                    'p': p,
                    'noise': noise_model
                }, "best_superactivation.pt")
                
            # Send data to client
            data = {
                "type": "update",
                "epoch": epoch,
                "ic": Ic.item(),
                "npt": npt_penalty.item(),
                "tp": tp_penalty.item(),
                "is_best": is_best
            }
            await websocket.send_text(json.dumps(data))
            
            # Yield control back to event loop occasionally
            if epoch % 5 == 0:
                await asyncio.sleep(0.01)
                
        await websocket.send_text(json.dumps({"type": "done", "best_ic": best_ic}))
        
    except Exception as e:
        try:
            await websocket.send_text(json.dumps({"type": "log", "message": f"Error: {e}"}))
        except:
            pass # Socket might already be closed
        raise e
    finally:
        # Explicitly free GPU memory back to the OS!
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("CUDA memory cache cleared.")
