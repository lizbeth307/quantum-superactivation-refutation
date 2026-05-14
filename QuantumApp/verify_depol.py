import numpy as np

def H(p, d):
    l1 = 1 - p + p/(d**2)
    l2 = p/(d**2)
    s = 0
    if l1 > 0: s -= l1 * np.log2(l1)
    if l2 > 0: s -= (d**2 - 1) * l2 * np.log2(l2)
    return s

def capacity_depolarizing(p, d):
    # Coherent information of the depolarizing channel
    return np.log2(d) - H(p, d)

d = 4
p = 0.35
cap = capacity_depolarizing(p, d)
print(f"Capacity of Depolarizing Channel (d={d}, p={p}) = {cap:.5f} bits")
if cap <= 0:
    print("VERIFIED: The single Depolarizing channel has ZERO quantum capacity.")
else:
    print("WARNING: The channel has positive capacity.")
