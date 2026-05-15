import numpy as np
import pysr
import warnings

warnings.filterwarnings('ignore')

# 1. Створюємо реалістичну симуляцію Спектру Потужності CMB (Реліктового випромінювання)
print("Generating CMB Power Spectrum data (Baryon Acoustic Oscillations)...")
l = np.linspace(50, 1000, 250)  # Мультипольний момент (кутовий масштаб)

# Фізична модель: Ефект Сакса-Вольфа + Акустичні осциляції + Згасання Сілка
# Перший пік біля l~220, другий ~530
silk_damping = np.exp(-(l / 700)**2)
acoustic_peaks = np.sin(l * np.pi / 300 - 0.5)**2
sachs_wolfe = 1000 * (100 / l)**0.8

D_l = sachs_wolfe + 4500 * acoustic_peaks * silk_damping

# Додаємо космічний шум (Cosmic Variance)
noise = np.random.normal(0, 80, size=len(l))
y = D_l + noise

print("Feeding 13.8 billion years of cosmic data into PySR...")

# 2. Налаштовуємо ШІ для пошуку закону природи
model = pysr.PySRRegressor(
    niterations=30,  # Швидкий пошук
    binary_operators=["+", "*", "-", "/"],
    unary_operators=["sin", "exp"],  # Зорі, космос і хвилі часто описуються синусами та експонентами
    maxsize=25,
    populations=20,
    temp_equation_file=True,
    verbosity=0
)

model.fit(l.reshape(-1, 1), y)

print("\n==================================================")
print("CMB PATTERN FORMULA EXTRACTED!")
best_eq = model.sympy()
print(f"Mathematical Law of the Universe: {best_eq}")
print("==================================================\n")

# Зберігаємо патерн для майбутнього "пошуку" в інших місцях
np.save("cmb_pattern.npy", {"l": l, "Dl": y, "formula": str(best_eq)})
print("Pattern saved to 'cmb_pattern.npy'. Ready for cross-referencing.")
