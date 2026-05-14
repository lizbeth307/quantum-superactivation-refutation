import os
import torch
import numpy as np
from typing import Any, Dict

class QuantumStorage:
    """Handles unified saving and loading of quantum states and models."""
    def __init__(self, root_dir: str = None):
        if root_dir is None:
            # Point to QuantumNEAT/sa_data by default
            self.root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'sa_data'))
        else:
            self.root_dir = root_dir
        os.makedirs(self.root_dir, exist_ok=True)
        
    def save_tensor(self, name: str, data: Dict[str, Any]):
        path = os.path.join(self.root_dir, f"{name}.pt")
        torch.save(data, path)
        print(f"[QuantumStorage] Saved tensor to {path}")
        
    def load_tensor(self, name: str) -> Dict[str, Any]:
        path = os.path.join(self.root_dir, f"{name}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} not found.")
        return torch.load(path)
        
    def save_numpy(self, name: str, **kwargs):
        path = os.path.join(self.root_dir, f"{name}.npz")
        np.savez(path, **kwargs)
        print(f"[QuantumStorage] Saved numpy array to {path}")
