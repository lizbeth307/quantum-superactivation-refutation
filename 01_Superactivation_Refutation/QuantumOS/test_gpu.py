from qiskit_aer import AerSimulator
simulator_gpu = AerSimulator(device='GPU')
print("Backend info:", simulator_gpu.name)
print("Available devices:", simulator_gpu.available_devices())
