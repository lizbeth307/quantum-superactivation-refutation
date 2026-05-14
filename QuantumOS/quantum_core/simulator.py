import pennylane as qml
import torch
from .noise_model import default_noise, get_qiskit_noise_model

NUM_QUBITS = 30

# Перехід на Qiskit Aer для симуляції 30 кубітів
try:
    q_noise = get_qiskit_noise_model()
    # ПРИМУСОВО вказуємо method='statevector' для Монте-Карло, інакше Qiskit спробує створити 
    # матрицю густини 30 кубітів (що неможливо і призводить до 0% навантаження CPU / зависання)
    dev = qml.device('qiskit.aer', wires=NUM_QUBITS, noise_model=q_noise, method='statevector')
    USE_QISKIT = True
except Exception as e:
    dev = qml.device('default.qubit', wires=NUM_QUBITS)
    USE_QISKIT = False

def noisy_gate(gate_func, wires, *args, **kwargs):
    """
    Обгортка для гейтів. Якщо використовуємо Qiskit, шум автоматично накладається симулятором.
    """
    gate_func(*args, wires=wires, **kwargs)
    
    if not USE_QISKIT:
        if isinstance(wires, int) or len(wires) == 1:
            default_noise.apply_single_qubit_noise(wires)
        elif len(wires) == 2:
            default_noise.apply_two_qubit_noise(wires)

def measure_with_error(wire):
    """
    Додає помилку зчитування перед фінальним вимірюванням.
    """
    default_noise.apply_readout_error(wire)
    return qml.expval(qml.PauliZ(wire))

# Приклад створення кастомного ланцюжка з підтримкою GPU-Chunking та автодиференціювання
import math
from .chunked_simulator import PyTorchChunkedSimulator

def run_circuit_forward(inputs, weights, erasure_flags):
    """
    Допоміжна функція для запуску симулятора 30 кубітів.
    Використовує наш оптимізований ChunkedSimulator.
    """
    sim = PyTorchChunkedSimulator(NUM_QUBITS, chunk_bits=24, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Ініціалізація стану (Encoder)
    for i in range(NUM_QUBITS):
        sim.apply_ry(inputs[i], i)
        
    # 2. Шар оптимізації (Parametrized Gates), який контролює ШІ
    weight_idx = 0
    for i in range(NUM_QUBITS):
        sim.apply_rx(weights[weight_idx], i)
        sim.apply_rz(weights[weight_idx+1], i)
        weight_idx += 2
        
    # 3. Заплутування (Двокубітні гейти)
    for i in range(NUM_QUBITS - 1):
        sim.apply_cnot(i, i+1)
        
    # 4. Joint System Noise (PPT ⊗ Erasure with Classical Flags)
    # Згідно зі Смітом-Ярдом та Парентіном: спільна система двох каналів з нульовою ємністю
    import random
    pi_tensor = torch.tensor(math.pi, dtype=torch.float64, device='cpu')
    
    for i in range(NUM_QUBITS):
        if i < 15:
            # Кубіти 0..14: Erasure Channel
            # Застосовуємо стирання (повна деполяризація) ТІЛЬКИ якщо класичний прапорець == 1
            if erasure_flags[i].item() == 1.0:
                error_prob = 0.75
                if random.random() < error_prob:
                    pauli = random.choice(['X', 'Y', 'Z'])
                    if pauli == 'X': sim.apply_rx(pi_tensor, i)
                    elif pauli == 'Y': sim.apply_ry(pi_tensor, i)
                    elif pauli == 'Z': sim.apply_rz(pi_tensor, i)
        else:
            # Кубіти 15..29: PPT/Horodecki шум (завжди працює, немає прапорця)
            error_prob = 0.75
            if random.random() < error_prob:
                pauli = random.choice(['X', 'Y', 'Z'])
                if pauli == 'X': sim.apply_rx(pi_tensor, i)
                elif pauli == 'Y': sim.apply_ry(pi_tensor, i)
                elif pauli == 'Z': sim.apply_rz(pi_tensor, i)
        
    return sim.expval_z().cpu()

class QuantumCircuitFunction(torch.autograd.Function):
    """
    Кастомна PyTorch функція для обчислення градієнтів за допомогою
    правила зсуву параметрів (Parameter-Shift Rule) без переповнення пам'яті (VRAM).
    """
    @staticmethod
    def forward(ctx, inputs, weights, erasure_flags):
        ctx.save_for_backward(inputs, weights, erasure_flags)
        # Виконуємо прямий прохід
        return run_circuit_forward(inputs, weights, erasure_flags)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, weights, erasure_flags = ctx.saved_tensors
        
        # --- SPSA (Simultaneous Perturbation Stochastic Approximation) ---
        # Повертаємо швидкий SPSA для 200-300 епох
        delta = torch.randint(0, 2, weights.shape, device=weights.device, dtype=weights.dtype) * 2 - 1
        c = 0.1
        
        weights_p = weights + c * delta
        weights_n = weights - c * delta
        
        res_p = run_circuit_forward(inputs, weights_p, erasure_flags)
        res_n = run_circuit_forward(inputs, weights_n, erasure_flags)
        
        grad_scalar = torch.sum(grad_output * (res_p - res_n))
        grad_weights = (grad_scalar / (2 * c)) * delta
        
        return None, grad_weights, None

def sample_circuit(inputs, weights, erasure_flags):
    """
    Інтерфейс для PyTorch, який замінює QNode з PennyLane.
    """
    return QuantumCircuitFunction.apply(inputs, weights, erasure_flags)
