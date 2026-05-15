import asyncio
import torch
import time

class MockWebsocket:
    async def send_text(self, text):
        pass # print(f"WS SEND: {text}")

from QuantumApp.backend.optimizer import run_optimization_loop

async def main():
    ws = MockWebsocket()
    params = {
        "d": 4,
        "energy": 100.0,
        "p": 0.5,
        "epochs": 1500,
        "ancilla": False,
        "noise": "erasure",
        "topology": "bipartite",
        "objective": "quantum"
    }
    
    print("Starting Deep Search for Smith-Yard True Superactivation (d=4)...")
    start = time.time()
    try:
        await run_optimization_loop(ws, params)
    except Exception as e:
        import traceback
        traceback.print_exc()
    print(f"Finished in {time.time() - start:.2f} seconds.")

    # Load and analyze the best state
    try:
        data = torch.load("best_superactivation.pt", weights_only=True)
        T_in = data['T_in']
        d = data['d']
        dim_in = T_in.shape[0]
        T_tensor = T_in.reshape(d, dim_in // d, T_in.shape[1])
        rho_A = torch.einsum('ijr, kjr -> ik', T_tensor, T_tensor.conj())
        probs = torch.real(torch.diag(rho_A)).cpu().numpy()
        print(f"\nDiscovered Optimal State Probabilities (d=4):")
        print(probs)
    except Exception as e:
        print("Could not load or analyze the state.")

asyncio.run(main())
