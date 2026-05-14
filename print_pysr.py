import numpy as np
def P(n):
    return 0.0147909476534416*((n - 1.9628512)**2 - 1.170888)**2 + 0.14819089
for n in range(6):
    print(f"P({n}) = {P(n)}")
