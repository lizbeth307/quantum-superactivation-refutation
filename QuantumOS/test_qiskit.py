import pennylane as qml

print("Importing qiskit noise...")
from qiskit_aer.noise import NoiseModel, depolarizing_error
noise_model = NoiseModel()
error_1 = depolarizing_error(0.001, 1)
noise_model.add_all_qubit_quantum_error(error_1, ['rx', 'ry', 'rz'])

print("Creating device with method='statevector'...")
try:
    dev = qml.device('qiskit.aer', wires=2, noise_model=noise_model, method='statevector')
    print("Device created successfully!")
except Exception as e:
    print(f"Error: {e}")

print("Creating device with backend_options...")
try:
    dev2 = qml.device('qiskit.aer', wires=2, noise_model=noise_model, backend_options={'method': 'statevector'})
    print("Device 2 created successfully!")
except Exception as e:
    print(f"Error: {e}")

print("Creating QNode...")
@qml.qnode(dev)
def test_circuit():
    qml.RX(0.1, wires=0)
    return qml.expval(qml.PauliZ(0))

print("Executing circuit...")
try:
    res = test_circuit()
    print("Result:", res)
except Exception as e:
    print(f"Error: {e}")
