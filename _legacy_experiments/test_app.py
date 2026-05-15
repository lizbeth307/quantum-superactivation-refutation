import asyncio
import json
import torch
import sys
import os

class MockWebsocket:
    async def send_text(self, text):
        print(f"WS SEND: {text}")

from QuantumApp.backend.optimizer import run_optimization_loop

async def main():
    ws = MockWebsocket()
    params = {
        "d": 3,
        "energy": 15.0,
        "p": 0.5,
        "epochs": 10,
        "ancilla": False,
        "noise": "wormhole",
        "topology": "bipartite",
        "objective": "retrocausality"
    }
    
    # Let's import the math functions and hook them to print grad
    import quantum_core.math as qm
    
    try:
        await run_optimization_loop(ws, params)
    except Exception as e:
        import traceback
        traceback.print_exc()
        import traceback
        traceback.print_exc()

asyncio.run(main())
