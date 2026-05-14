# ⚛️ QuantumNEAT & QuantumOS — Computational Refutation of Quantum Superactivation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20189708.svg)](https://doi.org/10.5281/zenodo.20189708)

> **A rigorous computational disproof of the Smith-Yard (2008) superactivation hypothesis using exact Stinespring purifications and hybrid AI simulation.**

## 🏆 Key Results

This repository contains the exact source code, algorithms, and data required to computationally reproduce the refutation of quantum superactivation via bound entanglement in low dimensions.

### 1. The Analytical Flaw (Micro-level)
Using high-precision global optimization (Basin Hopping + Adam) and exact double-precision LAPACK eigensolvers, we traverse the full 512-dimensional Positive Partial Transpose (PPT) manifold for $d=4 \times 4$. We establish a strict upper bound:
```text
max K_DW ≤ -0.6825 bits (for d=4x4)
max K_DW ≤ -2.08 bits (for d=6x6)
```
This proves that the foundational Horodecki private states are incapable of establishing a positive key rate, resulting in profound leakage to the environment and breaking the quantum one-time pad protocol.

### 2. The Macroscopic Failure (N=15 Joint System)
To address the finite-blocklength claims of Parentin et al. (2026), we developed **QuantumOS**, a hybrid 30-qubit machine learning simulator.
Using a Simultaneous Perturbation Stochastic Approximation (SPSA) attack on the Joint Channel topology ($\mathcal{N}_{PPT} \otimes \mathcal{N}_{Erasure}$), we demonstrate a complete failure to achieve positive capacity. 

**The Prescient Encoder Bound:** Even when the AI is granted unphysical, omniscient prescience of the classical Erasure Flags *before* encoding, the optimization hits a strict Barren Plateau at absolute depolarization (Loss ~1.0).

## 📁 Repository Structure

```text
├── kdw_hunt.py            # Exact d=4x4 global optimization script
├── requirements.txt       # Python dependencies
└── /QuantumOS/            # (Associated module for N=15 AI attack)
    ├── quantum_core/      # 30-qubit simulator logic
    ├── ai_engine/         # Hybrid PyTorch stabilization models
    └── ui/                # Terminal Dashboard (quantum_os.py)
```

## 🚀 How to Run

### 1. Exact Analytical Bound ($d=4 \times 4$)
To reproduce the $-0.68$ bound using LAPACK purifications:
```bash
python kdw_hunt.py
```

### 2. The $N=15$ Macroscopic Joint Channel Attack
To run the 30-qubit SPSA simulation with classical erasure flags:
```bash
python QuantumOS/ui/quantum_os.py
```

## 📖 Citation

```bibtex
@misc{murai_2026_20189708,
  author       = {Murai, Yaroslav},
  title        = {Computational Refutation of Quantum Superactivation via Bound Entanglement in Low Dimensions},
  month        = may,
  year         = 2026,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.20189708},
  url          = {https://doi.org/10.5281/zenodo.20189708}
}
```
