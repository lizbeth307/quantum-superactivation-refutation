import sys
import os
import torch
import numpy as np

try:
    from quantum_core.channels import (
        build_erasure_channel, build_smith_yard_ppt
    )
except ImportError as e:
    print(f"Warning: {e}")
    
def build_depolarizing_channel(d, p):
    Ks = []
    I_mat = torch.eye(d, dtype=torch.complex128)
    Ks.append(np.sqrt(1 - p) * I_mat)
    
    norm_val = np.sqrt(p / d)
    for i in range(d):
        for j in range(d):
            K = torch.zeros((d, d), dtype=torch.complex128)
            K[i, j] = norm_val
            Ks.append(K)
    return torch.stack(Ks)

def build_phase_damping_channel(d, p):
    Ks = []
    # K0 = diag(1, sqrt(1-p), sqrt(1-p), ...)
    K0 = torch.zeros((d, d), dtype=torch.complex128)
    K0[0, 0] = 1.0
    for i in range(1, d):
        K0[i, i] = np.sqrt(1.0 - p)
    Ks.append(K0)
    
    # K_i = diag(0, ..., sqrt(p), 0...)
    for i in range(1, d):
        Ki = torch.zeros((d, d), dtype=torch.complex128)
        Ki[i, i] = np.sqrt(p)
        Ks.append(Ki)
        
    return torch.stack(Ks)

def build_black_hole_channel(d, p):
    # Dephrasure Channel (Erasure + Dephasing)
    # Output dimension is d+1
    q_erase = p
    q_dephase = min(p * 1.5, 0.99) # Make dephasing slightly stronger
    
    Ks = []
    
    # 1. Dephasing part (weight 1 - q_erase)
    # K0 = sqrt(1 - q_erase) * sqrt(1 - q_dephase) * I
    K0 = torch.zeros((d+1, d), dtype=torch.complex128)
    for i in range(d):
        K0[i, i] = np.sqrt(1.0 - q_erase) * np.sqrt(1.0 - q_dephase)
    Ks.append(K0)
    
    # Ki = sqrt(1 - q_erase) * sqrt(q_dephase) * |i><i|
    for i in range(d):
        Ki = torch.zeros((d+1, d), dtype=torch.complex128)
        Ki[i, i] = np.sqrt(1.0 - q_erase) * np.sqrt(q_dephase)
        Ks.append(Ki)
        
    # 2. Erasure part (weight q_erase)
    # E_i = sqrt(q_erase) * |e><i|
    for i in range(d):
        Ei = torch.zeros((d+1, d), dtype=torch.complex128)
        Ei[d, i] = np.sqrt(q_erase)
        Ks.append(Ei)
        
    return torch.stack(Ks)
def build_amplitude_damping_channel(d, p):
    Ks = []
    K0 = torch.zeros((d, d), dtype=torch.complex128)
    K0[0, 0] = 1.0
    for i in range(1, d):
        K0[i, i] = np.sqrt(1.0 - p)
    Ks.append(K0)
    
    for i in range(1, d):
        Ki = torch.zeros((d, d), dtype=torch.complex128)
        Ki[0, i] = np.sqrt(p)
        Ks.append(Ki)
        
    return torch.stack(Ks)
