import time
import random
import threading
import sys
import os

import plotext as plt
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box
from rich.ansi import AnsiDecoder
from rich.console import Group

# Додаємо кореневу папку для імпорту quantum_core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from quantum_core.simulator import sample_circuit, NUM_QUBITS

# Глобальний стан для синхронізації UI та PyTorch
shared_state = {
    "epoch": 0,
    "max_epochs": 30,
    "loss": 0.0,
    "loss_history": [],
    "logs": ["Ініціалізація QuantumOS..."],
    "qubit_states": [(f"|0⟩", "0.0%") for _ in range(NUM_QUBITS)],
    "is_training": True,
    "epoch_start_time": None
}

# -----------------------------------------------------
# PyTorch AI Engine (Фоновий потік)
# -----------------------------------------------------
class QuantumStabilizerAI(nn.Module):
    def __init__(self):
        super(QuantumStabilizerAI, self).__init__()
        # Приймаємо 30 кубітів + 15 класичних прапорців стирання = 45
        self.fc1 = nn.Linear(NUM_QUBITS + 15, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, NUM_QUBITS * 2) 
        nn.init.uniform_(self.fc2.weight, -0.1, 0.1)

    def forward(self, inputs, erasure_flags):
        # Гібридна передача знань: ШІ бачить, які кубіти стерті
        x = torch.cat([inputs, erasure_flags], dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        weights = self.fc2(x)
        q_outputs = []
        for i in range(inputs.shape[0]):
            q_out = sample_circuit(inputs[i], weights[i], erasure_flags[i])
            q_outputs.append(q_out)
        return torch.stack(q_outputs)

def train_loop():
    try:
        torch.set_default_dtype(torch.float64)
        model = QuantumStabilizerAI()
        
        # Дефолтний learning rate
        lr = 0.01 
        optimizer = optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        
        # Зменшуємо батч для швидкості оновлення UI
        batch_size = 1
        epochs = shared_state["max_epochs"]
        
        start_epoch = 1
        # Логіка автоматичного підвантаження чекпойнтів
        checkpoint_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ai_engine", "checkpoints")
        # Checkpoint loading
        start_epoch = 1
        checkpoint_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ai_engine", "checkpoints")
        if os.path.exists(checkpoint_dir):
            checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
            if checkpoints:
                checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
                latest_ckpt = checkpoints[-1]
                ckpt_path = os.path.join(checkpoint_dir, latest_ckpt)
                try:
                    checkpoint = torch.load(ckpt_path)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    start_epoch = checkpoint['epoch'] + 1
                    shared_state["loss"] = checkpoint['loss']
                    shared_state["logs"].append(f"[УСПІХ] Відновлено ваги з {latest_ckpt} (SPSA Атака)")
                except Exception as e:
                    shared_state["logs"].append(f"[ПОМИЛКА] Не вдалося завантажити чекпойнт: {e}")
        
        # SPSA може потребувати 200-300 епох для стабілізації
        epochs = 300
        shared_state["max_epochs"] = epochs
        
        shared_state["logs"].append("Початок атаки на Спільну Систему Сміта-Ярда...")
        time.sleep(0.1)
        
        for epoch in range(start_epoch, epochs + 1):
            shared_state["epoch_start_time"] = time.time()
            target_states = torch.ones((batch_size, NUM_QUBITS))
            inputs = target_states.clone()
            
            # Генерація прапорців стирання (50% ймовірність для перших 15 кубітів)
            erasure_flags = torch.randint(0, 2, (batch_size, 15), dtype=torch.float64)
            
            optimizer.zero_grad()
            noisy_outputs = model(inputs, erasure_flags)
            loss = loss_fn(noisy_outputs, target_states)
            
            shared_state["logs"].append(f"[ОБЧИСЛЕННЯ] Градієнт для епохи {epoch}...")
            time.sleep(0.1) # Оновлюємо UI
            
            loss.backward()
            optimizer.step()
            
            # Оновлюємо стан для UI
            shared_state["epoch"] = epoch
            val_loss = loss.item()
            shared_state["loss"] = val_loss
            shared_state["loss_history"].append(val_loss)
            
            # Збереження чекпойнту
            os.makedirs(checkpoint_dir, exist_ok=True)
            new_ckpt_path = os.path.join(checkpoint_dir, f"stabilizer_epoch_{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }, new_ckpt_path)
            shared_state["logs"].append(f"[СИСТЕМА] Збережено чекпойнт епохи {epoch}")
            
            # Симулюємо потік сирих даних
            bin_stream = "".join(str(random.randint(0, 1)) for _ in range(40))
            shared_state["logs"].append(f"[{time.strftime('%H:%M:%S')}] Loss {val_loss:.4f} | {bin_stream}")
            if len(shared_state["logs"]) > 10:
                shared_state["logs"].pop(0)
                
            # Оновлюємо стани кубітів (помилка корелює з Loss)
            new_states = []
            for q in range(NUM_QUBITS):
                err = max(0.1, (val_loss * 5) + random.uniform(-0.5, 0.5)) 
                state = f"0.{random.randint(100, 999)}|0⟩ + 0.{random.randint(100, 999)}|1⟩"
                new_states.append((state, f"{err:.1f}%"))
            shared_state["qubit_states"] = new_states
            
        shared_state["logs"].append("Навчання успішно завершено!")
        shared_state["is_training"] = False
    except Exception as e:
        shared_state["logs"].append(f"[CRITICAL ERROR] {str(e)}")
        shared_state["is_training"] = False

# -----------------------------------------------------
# UI Engine (Головний потік)
# -----------------------------------------------------
def create_layout():
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main")
    )
    layout["main"].split_row(
        Layout(name="quantum_core", ratio=1),
        Layout(name="ai_engine", ratio=1)
    )
    layout["main"]["quantum_core"].split_column(
        Layout(name="qubits", ratio=2),
        Layout(name="graph", ratio=1)
    )
    return layout

def make_plot():
    """Створює графік Plotext та повертає його для Rich"""
    plt.clf()
    plt.plotsize(40, 10)
    plt.title("Реальна Loss-функція (PyTorch)")
    # Видалили plt.colorless(), Rich сам відрендерить ANSI-кольори Plotext
    
    if len(shared_state["loss_history"]) > 0:
        plt.plot(shared_state["loss_history"], marker="dot")
    else:
        plt.plot([0], [0]) # Пустий графік
        
    plot_str = plt.build()
    decoder = AnsiDecoder()
    lines = list(decoder.decode(plot_str))
    return Panel(Group(*lines), title="[bold green]Крива Навчання ШІ[/bold green]", border_style="green")

def generate_qubit_status():
    table = Table(box=box.SIMPLE_HEAVY, border_style="green", expand=True)
    table.add_column("Qubit", justify="center", style="green")
    table.add_column("Live State |ψ⟩", justify="center")
    table.add_column("Error Rate", justify="right")

    for i in range(min(10, NUM_QUBITS)): 
        state, error = shared_state["qubit_states"][i]
        err_val = float(error[:-1]) if "%" in error else 0.0
        color = "red" if err_val > 3.0 else "green"
        table.add_row(f"Q{i}", Text(state, style="bold green"), Text(error, style=color))
        
    return Panel(table, title=f"[bold green]Квантові Стани (Live)[/bold green]", border_style="green")

def generate_ai_log():
    text = Text()
    text.append("Нейромережа стабілізації працює у фоновому потоці...\n\n", style="bold green")
    epoch = shared_state["epoch"]
    loss = shared_state["loss"]
    text.append(f"=> Епоха навчання: {epoch}/{shared_state['max_epochs']}\n", style="bold cyan")
    
    if shared_state["is_training"] and shared_state.get("epoch_start_time") is not None:
        elapsed = time.time() - shared_state["epoch_start_time"]
        text.append(f"=> Тривалість епохи: {elapsed:.1f} сек (Обчислення градієнта CUDA)\n", style="bold yellow")
        
    text.append(f"=> Втрата (MSE Loss): {loss:.4f}\n\n", style="bold red" if loss > 0.825 else "bold green")
    
    text.append("[RAW DATA STREAM & LOGS]\n", style="dim green")
    for log in shared_state["logs"]:
        text.append(f"{log}\n", style="dim green")
        
    return Panel(text, title="[bold green]ШІ Стабілізатор (Neural Mitigator)[/bold green]", border_style="green")

def main():
    # Запускаємо PyTorch у фоновому потоці (Daemon Thread)
    trainer = threading.Thread(target=train_loop, daemon=True)
    trainer.start()

    layout = create_layout()
    layout["header"].update(Panel(Text("QUANTUM OS v2.0 | LIVE SYNC | PLOTEXT GRAPH", style="bold green", justify="center"), style="bold green", box=box.MINIMAL))
    
    # Запуск UI з частотою оновлення 10 кадрів/сек
    with Live(layout, refresh_per_second=10, screen=True) as live:
        # Змінив 'or' на 'and', щоб UI закривався, якщо потік "вмер" (trainer.is_alive() == False)
        while shared_state["is_training"] and trainer.is_alive():
            layout["main"]["quantum_core"]["qubits"].update(generate_qubit_status())
            layout["main"]["quantum_core"]["graph"].update(make_plot())
            layout["main"]["ai_engine"].update(generate_ai_log())
            time.sleep(0.1)
            
        # Фінальне оновлення після завершення
        layout["main"]["quantum_core"]["qubits"].update(generate_qubit_status())
        layout["main"]["quantum_core"]["graph"].update(make_plot())
        layout["main"]["ai_engine"].update(generate_ai_log())
        
        # Даємо користувачу час роздивитися результат
        time.sleep(10)

if __name__ == "__main__":
    main()
