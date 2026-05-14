import torch
import numpy as np

def analyze_best():
    try:
        data = torch.load("best_superactivation.pt", weights_only=True)
    except FileNotFoundError:
        print("No best_superactivation.pt found.")
        return
        
    print(f"=== Analysis of Saved State ===")
    print(f"Noise Model: {data.get('noise', 'unknown')}")
    print(f"Dimension d: {data['d']}")
    print(f"Noise Prob p: {data['p']}")
    print(f"Metric (Ic/Dist): {data['Ic']:.5f}")
    
    T_in = data['T_in']
    K_PPT = data['K_PPT']
    
    print("\n--- Input State T_in ---")
    # T_in has shape (dim_in, rank_in), which is flattened A tensor R.
    # We want the reduced density matrix of A, which has dimension d.
    d = data['d']
    T_mat = T_in.reshape(d, d) # Assuming rank_in = dim_in, wait, T_in is (d*d, d*d)
    # Actually T_in is (d*d, rank_in). We reshape it to (d, d, rank_in)
    T_tensor = T_in.reshape(d, d, T_in.shape[1])
    # rho_A = Tr_R(T T^dagger) -> sum over j (R) and r (rank)
    # rho_A_ik = sum_{j,r} T_{i,j,r} T^*_{k,j,r}
    rho_A = torch.einsum('ijr, kjr -> ik', T_tensor, T_tensor.conj())
    
    # Calculate probabilities of each basis state for probe A
    probs = torch.real(torch.diag(rho_A))
    
    print("Photon Number Probabilities (Probe A):")
    for i in range(d):
        if probs[i] > 0.001:
            print(f"  |{i}> : {probs[i].item()*100:.2f}%")
            
    n_hat = torch.arange(d, dtype=torch.float64, device=T_in.device)
    avg_energy = torch.sum(n_hat * probs).item()
    print(f"Average Energy (Photons): {avg_energy:.3f}")
    
    print("\n--- PPT Channel K_PPT ---")
    print(f"Number of Kraus Operators: {K_PPT.shape[0]}")
    
    # Check trace preservation exactly
    sum_K_K = torch.zeros((data['d'], data['d']), dtype=torch.complex128, device=K_PPT.device)
    for k in range(K_PPT.shape[0]):
        sum_K_K += K_PPT[k].conj().T @ K_PPT[k]
        
    trace_err = torch.norm(sum_K_K - torch.eye(data['d'], device=K_PPT.device)).item()
    print(f"Trace-Preservation Error: {trace_err:.2e}")

if __name__ == "__main__":
    analyze_best()
