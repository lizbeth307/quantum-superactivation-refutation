"""Quick SA benchmark — 30 seconds max."""
import time, sys, os
os.environ['PYTHONIOENCODING'] = 'utf-8'
import numpy as np
import torch

print("=== SA BENCHMARK (quick) ===")
print(f"GPU: {torch.cuda.get_device_name(0)}")
dev = torch.device('cuda')

for d in [4, 6, 8, 10, 12]:
    n = d * d
    # GPU batch eigendecomp
    A = torch.randn(100, n, n, dtype=torch.cfloat, device=dev)
    A = A + A.conj().transpose(-1, -2)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(5):
        torch.linalg.eigh(A)
    torch.cuda.synchronize()
    gpu_ms = (time.perf_counter() - t0) / 5 / 100 * 1000

    # CPU single
    B = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    B = B + B.conj().T
    t0 = time.perf_counter()
    for _ in range(20):
        np.linalg.eigh(B)
    cpu_ms = (time.perf_counter() - t0) / 20 * 1000

    print(f"d={d:2d} ({n:3d}x{n:3d}): GPU={gpu_ms:.3f}ms  CPU={cpu_ms:.3f}ms  speedup={cpu_ms/gpu_ms:.0f}x")

# Autograd test
print("\n--- Autograd (gradient of entropy) ---")
for d in [4, 6, 8]:
    n = d * d
    L = torch.randn(n, 6, dtype=torch.cfloat, device=dev, requires_grad=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(50):
        rho = L @ L.conj().T
        rho = rho / rho.trace()
        eigs = torch.linalg.eigvalsh(rho)
        loss = -(eigs.clamp(min=1e-12) * torch.log2(eigs.clamp(min=1e-12))).sum()
        loss.backward()
        L.grad.zero_()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / 50 * 1000
    print(f"d={d:2d}: {ms:.2f} ms/gradient")

# CPU cores
import multiprocessing
print(f"\nCPU cores: {multiprocessing.cpu_count()}")

# Memory
print("\n--- VRAM per batch ---")
for d in [4, 6, 8, 10, 12, 16]:
    mb = 1000 * (d*d)**2 * 8 / 1024**2
    print(f"d={d:2d}: 1000 states = {mb:.0f} MB ({mb/12226*100:.1f}% VRAM)")

print("\nDONE")
