import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

torch.set_default_dtype(torch.float64)

# Додаємо кореневу папку для імпорту quantum_core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quantum_core.simulator import sample_circuit, NUM_QUBITS

class QuantumStabilizerAI(nn.Module):
    """
    Гібридна квантово-класична нейронна мережа.
    Класична частина (PyTorch) керує параметрами (weights) 
    зашумленого квантового ланцюжка, щоб мінімізувати вплив шуму 
    (Error Mitigation).
    """
    def __init__(self):
        super(QuantumStabilizerAI, self).__init__()
        
        # Класична нейромережа (пре-процесинг)
        self.fc1 = nn.Linear(NUM_QUBITS, 64)
        self.relu = nn.ReLU()
        # Генеруємо 2 параметри обертання (RX, RZ) для кожного кубіта
        self.fc2 = nn.Linear(64, NUM_QUBITS * 2) 
        
        # Ініціалізація ваг
        nn.init.uniform_(self.fc2.weight, -0.1, 0.1)

    def forward(self, inputs):
        # 1. Класичний пре-процесинг (визначаємо, як скомпенсувати помилки)
        x = self.fc1(inputs)
        x = self.relu(x)
        weights = self.fc2(x) # [batch_size, NUM_QUBITS * 2]
        
        # 2. Квантова екзекуція (зашумлена)
        # PennyLane qnode з інтерфейсом torch підтримує батчінг
        # Проте для простоти обробимо по одному зразку
        q_outputs = []
        for i in range(inputs.shape[0]):
            # Проганяємо через симулятор з шумом
            q_out = sample_circuit(inputs[i], weights[i])
            q_outputs.append(q_out)
            
        return torch.stack(q_outputs)

import time

def train_stabilizer(epochs=50, batch_size=16):
    """
    Функція для тренування ШІ працювати "крізь бруд".
    """
    print(f"Ініціалізація QuantumOS Stabilizer (Ryzen 9 Backend)...")
    model = QuantumStabilizerAI()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss() # Прагнемо отримати ідеальний чистий стан
    
    print("Початок навчання ШІ-компенсатора шуму (Error Mitigation)...")
    
    for epoch in range(epochs):
        start_time = time.time()
        # Генеруємо "ідеальні" бажані стани (наприклад, стан |1> для всіх кубітів)
        # У реальності тут будуть складніші квантові стани
        target_states = torch.ones((batch_size, NUM_QUBITS))
        
        # Вхідні дані: що ми хочемо згенерувати (чистий сигнал)
        inputs = target_states.clone()
        
        optimizer.zero_grad()
        
        # Пропускаємо через гібридну модель (PyTorch + Noisy PennyLane)
        # ШІ намагатиметься підібрати такі `weights`, щоб на виході вийшли одиниці, 
        # незважаючи на 5% помилки CNOT та декогеренцію
        noisy_outputs = model(inputs)
        
        # Обчислюємо втрату: наскільки зашумлений вихід відрізняється від ідеалу
        loss = loss_fn(noisy_outputs, target_states)
        
        loss.backward()
        optimizer.step()
        
        print(f"Епоха {epoch+1}/{epochs} | Втрата (Хаос): {loss.item():.4f} | Час виконання: {time.time() - start_time:.2f} сек")
        
        # Збереження чекпойнту
        checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"stabilizer_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
        }, checkpoint_path)
        print(f"Чекпойнт збережено: {checkpoint_path}")
        
    print("Навчання завершено. ШІ адаптувався до шуму IBM-рівня.")
    return model

if __name__ == "__main__":
    train_stabilizer(epochs=10, batch_size=1)
