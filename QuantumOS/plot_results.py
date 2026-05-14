import os
import torch
import matplotlib.pyplot as plt
import numpy as np

checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ai_engine", "checkpoints")
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "QuantumNEAT", "paper_pipeline", "spsa_joint_loss.pdf")

# Переконаємось, що папка існує
os.makedirs(os.path.dirname(output_path), exist_ok=True)

epochs = []
losses = []

# Зчитуємо всі чекпойнти
for filename in os.listdir(checkpoint_dir):
    if filename.startswith("stabilizer_epoch_") and filename.endswith(".pt"):
        epoch_str = filename.replace("stabilizer_epoch_", "").replace(".pt", "")
        try:
            epoch = int(epoch_str)
            # Беремо тільки епохи >= 16 (там де почалася атака на Сміта-Ярда)
            if epoch >= 16:
                ckpt_path = os.path.join(checkpoint_dir, filename)
                checkpoint = torch.load(ckpt_path, weights_only=False, map_location='cpu')
                if 'loss' in checkpoint:
                    epochs.append(epoch)
                    losses.append(checkpoint['loss'])
        except Exception as e:
            print(f"Помилка читання {filename}: {e}")

# Сортуємо по епохах
sorted_indices = np.argsort(epochs)
epochs = np.array(epochs)[sorted_indices]
losses = np.array(losses)[sorted_indices]

if len(epochs) == 0:
    print("Немає даних для побудови графіка (епохи >= 16).")
else:
    # Побудова графіка
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, marker='o', linestyle='-', color='red', markersize=4, label='SPSA Optimization Loss')
    plt.axhline(y=1.0, color='gray', linestyle='--', label='Theoretical Max Entropy (Zero Capacity)')
    
    plt.title('SPSA Optimization on Smith-Yard Joint System ($N=15$)', fontsize=14)
    plt.xlabel('Epoch (Training Iteration)', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, format='pdf', dpi=300)
    print(f"Графік успішно збережено у {output_path}")
    print(f"Дані: мінімальний Loss = {np.min(losses):.4f}, максимальний Loss = {np.max(losses):.4f}")
