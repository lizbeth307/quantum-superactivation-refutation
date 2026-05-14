import pennylane as qml
import numpy as np

class IBMNoiseModel:
    """
    Реалістична модель шуму для квантового симулятора, заснована на 
    характеристиках топових квантових комп'ютерів IBM (2026).
    """
    def __init__(self, 
                 single_qubit_error=0.001,  # 0.1%
                 two_qubit_error=0.03,      # 3% (середнє між 1% і 5%)
                 readout_error=0.02,        # 2% (середнє між 1% і 3%)
                 t1_time=100e-6,            # 100 мікросекунд (T1)
                 t2_time=100e-6,            # 100 мікросекунд (T2)
                 gate_time=100e-9):         # 100 наносекунд на 1 гейт
        
        self.single_qubit_error = single_qubit_error
        self.two_qubit_error = two_qubit_error
        self.readout_error = readout_error
        
        # Обчислюємо ймовірності затухання на основі часу
        # p = 1 - exp(-t / T)
        self.amp_damping_prob = 1 - np.exp(-gate_time / t1_time)
        self.phase_damping_prob = 1 - np.exp(-gate_time / t2_time)

    def apply_single_qubit_noise(self, wire):
        """Застосовує шум до одного кубіта після операції."""
        qml.DepolarizingChannel(self.single_qubit_error, wires=wire)
        qml.AmplitudeDamping(self.amp_damping_prob, wires=wire)
        qml.PhaseDamping(self.phase_damping_prob, wires=wire)

    def apply_two_qubit_noise(self, wires):
        """Застосовує шум до двох кубітів після двокубітної операції."""
        # Для спрощення застосовуємо DepolarizingChannel до обох кубітів
        # PennyLane має Multi-qubit Depolarizing Channel, але для 2 кубітів ми можемо
        # застосувати індивідуальні з вищою ймовірністю, або 2-qubit канал.
        qml.DepolarizingChannel(self.two_qubit_error, wires=wires[0])
        qml.DepolarizingChannel(self.two_qubit_error, wires=wires[1])
        
        # Декогеренція також діє під час двокубітного гейту
        qml.AmplitudeDamping(self.amp_damping_prob, wires=wires[0])
        qml.PhaseDamping(self.phase_damping_prob, wires=wires[0])
        qml.AmplitudeDamping(self.amp_damping_prob, wires=wires[1])
        qml.PhaseDamping(self.phase_damping_prob, wires=wires[1])

    def apply_readout_error(self, wire):
        """Застосовує помилку вимірювання (bit flip)."""
        qml.BitFlip(self.readout_error, wires=wire)

# Глобальний інстанс моделі шуму для швидкого доступу
default_noise = IBMNoiseModel()

def get_qiskit_noise_model():
    """Створює еквівалентну модель шуму для симулятора qiskit.aer"""
    try:
        from qiskit_aer.noise import NoiseModel, depolarizing_error
        noise_model = NoiseModel()
        error_1 = depolarizing_error(0.001, 1)
        error_2 = depolarizing_error(0.03, 2)
        # Додаємо помилки до стандартних гейтів
        noise_model.add_all_qubit_quantum_error(error_1, ['rx', 'ry', 'rz'])
        noise_model.add_all_qubit_quantum_error(error_2, ['cx'])
        return noise_model
    except ImportError:
        return None
