import asyncio
import torch
import json

class MockWebsocket:
    class State:
        DISCONNECTED = 0
    client_state = 1
    async def send_text(self, text):
        data = json.loads(text)
        if data["type"] == "done":
            print(f"BEST IC FOUND: {data['best_ic']}")

from QuantumApp.backend.optimizer import run_optimization_loop

async def main():
    ws = MockWebsocket()
    params = {
        "d": 5,
        "energy": 100.0,
        "p": 0.5,
        "epochs": 200,
        "ancilla": False,
        "noise": "erasure",
        "topology": "bipartite",
        "objective": "quantum"
    }
    
    await run_optimization_loop(ws, params)

asyncio.run(main())
