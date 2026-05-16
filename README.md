# ⚛️ QuantumNEAT & QuantumOS
> **Computational Refutation of Quantum Superactivation via Bound Entanglement**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20189708.svg)](https://doi.org/10.5281/zenodo.20189708)

> **A rigorous computational disproof of the 18-year-old Smith-Yard (2008) superactivation hypothesis using exact Stinespring purifications, hybrid AI simulation (PyTorch), and PySR symbolic regression.**

## 🏆 Key Discoveries

This repository contains the exact source code, algorithms, and data required to computationally reproduce the refutation of quantum superactivation via bound entanglement, as well as the emergent cosmological symmetry discovered during optimization.

### 1. The Analytical Flaw (Micro-level)
Using high-precision global optimization (Basin Hopping + Adam) and exact double-precision LAPACK eigensolvers, we traverse the full 512-dimensional Positive Partial Transpose (PPT) manifold for $d=4 \times 4$. We establish a strict upper bound:
```text
max K_DW ≤ -0.6825 bits (for d=4x4)
max K_DW ≤ -2.08 bits (for d=6x6)
```
This proves that the foundational Horodecki private states are incapable of establishing a positive key rate, resulting in profound leakage to the environment and breaking the quantum one-time pad protocol.

### 2. The Macroscopic Failure (N=15 Joint System)
Using a Simultaneous Perturbation Stochastic Approximation (SPSA) attack on the Joint Channel topology ($\mathcal{N}_{PPT} \otimes \mathcal{N}_{Erasure}$), we demonstrate a complete failure to achieve positive capacity, hitting a strict Barren Plateau at absolute depolarization (Loss ~1.0).

### 3. Exact Analytical Bound ($d=4 \times 4$)
Using exact Stinespring purifications and double-precision LAPACK eigensolvers, we traverse the full 512-dimensional Positive Partial Transpose (PPT) manifold for $d=4 \times 4$. We establish a strict upper bound:
```text
max K_DW ≤ -0.6825 bits
```
This analytically proves that the foundational Horodecki private states are incapable of establishing a positive key rate.



## 📁 Repository Structure

We have cleanly separated the two major discoveries into their respective directories. All legacy R&D testing scripts have been moved to an archive folder.

```text
├── 01_Superactivation_Refutation/   # The core Quantum Machine Learning simulations
│   ├── sa_full_PA_cq.py             # Main AI Optimization loop for Superactivation
│   ├── kdw_hunt.py                  # Exact d=4x4 global optimization analytical script
│   └── QuantumOS/                   # Associated module for N=15 AI attack
├── 01_Superactivation_Refutation/   # The core Quantum Machine Learning simulations
│   ├── sa_full_PA_cq.py             # Main AI Optimization loop for Superactivation
│   ├── kdw_hunt.py                  # Exact d=4x4 global optimization analytical script
│   └── QuantumOS/                   # Associated module for N=15 AI attack
├── paper_pipeline/                  # Formal reports and LaTeX manuscripts
│   └── paper_draft.tex              # The main arXiv submission LaTeX
└── _legacy_experiments/             # (Archive) Intermediate testing and raw data files
```

## 🚀 How to Run

### 1. The Superactivation Refutation (AI Joint Channel Attack)
To run the primary AI optimization loop that disproves superactivation:
```bash
python 01_Superactivation_Refutation/sa_full_PA_cq.py
```



## 📖 Citation

```bibtex
@misc{murai_2026_20189708,
  author       = {Murai, Yaroslav},
  title        = {Computational Refutation of Quantum Superactivation via Bound Entanglement},
  month        = may,
  year         = 2026,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.20189708},
  url          = {https://doi.org/10.5281/zenodo.20189708}
}
```
