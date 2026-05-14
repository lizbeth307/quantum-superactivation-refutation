import torch
import numpy as np
try:
    from pysr import PySRRegressor
except ImportError:
    print("PySR is not installed. Please pip install pysr")
    exit()

def run_pysr():
    try:
        data = torch.load("best_superactivation.pt", weights_only=True)
    except FileNotFoundError:
        print("No saved state found. Run optimization first.")
        return
        
    print(f"Starting PySR Symbolic Extraction for {data.get('noise', 'unknown')} channel...")
    
    T_in = data['T_in']
    d = data['d']
    
    # Calculate probabilities of the probe
    dim_in = T_in.shape[0]
    T_tensor = T_in.reshape(d, dim_in // d, T_in.shape[1])
    rho_A = torch.einsum('ijr, kjr -> ik', T_tensor, T_tensor.conj())
    probs = torch.real(torch.diag(rho_A)).detach().cpu().numpy()
    
    # Filter out near-zero probabilities to help PySR
    X_train = []
    y_train = []
    for i in range(d):
        if probs[i] > 1e-4:
            X_train.append([i])
            y_train.append(probs[i])
            
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    if len(X_train) < 2:
        print("State is too trivial (single peak). No complex formula to extract.")
        return
        
    model = PySRRegressor(
        niterations=100,  # Increased for better convergence
        binary_operators=["+", "*", "-", "/"],
        unary_operators=["exp", "square"], # Removed sin/cos to prevent Fourier-like overfitting
        parsimony=0.01, # Stronger penalty for complex formulas (Occam's Razor)
        verbosity=0
    )
    
    print("Fitting symbolic equations (Strict Mode: No trig functions, high parsimony)...")
    model.fit(X_train, y_train)
    
    print("\n" + "="*50)
    print("🎯 PySR EXTRACTION COMPLETE!")
    print("Best algebraic formula for the DFS Probability Distribution:")
    print(f"P(n) = {model.sympy()}")
    print("="*50 + "\n")

if __name__ == "__main__":
    run_pysr()
