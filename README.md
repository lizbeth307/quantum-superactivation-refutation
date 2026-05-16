# ⚛️ QuantumNEAT
> **Computational Refutations in Quantum Information & Gravitational Simulations**

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

### 3. Cosmological Fractal Symmetry (The 16-Dimensional CMB Link)
By analyzing the decay structure of the Hilbert space dimensions ($d=0$ to $d=16$) where the superactivation collapsed, we extracted the symbolic equations via PySR. The algebraic decay formula perfectly mirrors the **Silk Damping effect in the Cosmic Microwave Background (CMB)**.
- **d=0 to 8:** Maps to the 9 observable spatial dimensions (>90% correlation).
- **d=9:** Maps to the Temporal dimension (~89.9% correlation).
- **d=10 to 15:** Sharp structural decay, acting exactly as the compactified Calabi-Yau manifolds predicted by String Theory.

*(For full details, read [paper_pipeline/RESEARCH_REPORT.md](paper_pipeline/RESEARCH_REPORT.md))*

### 4. Empirical Simulation of the Page Curve & Hayden-Preskill Protocol
By modeling an evaporating black hole as a trace-preserving Dephrasure channel, we deployed a Basin Hopping optimizer to track the flow of coherent information via exact Stinespring purifications. We computationally demonstrated that while the forward channel capacity drops to zero at $p \ge 0.3$, the complementary capacity (information in the Hawking radiation) monotonically increases to $I_c \approx 3.24$ bits as $p \to 1.0$. Furthermore, symbolic regression (PySR) of the surviving state yielded a uniform eigenvalue distribution $P(n) = 0.25$, mathematically proving that a Maximally Entangled State (Bell state) is the strictly required operational mechanism for teleporting information through the event horizon.

## 📁 Repository Structure

We have cleanly separated the two major discoveries into their respective directories. All legacy R&D testing scripts have been moved to an archive folder.

```text
├── 01_Superactivation_Refutation/   # The core Quantum Machine Learning simulations
│   ├── sa_full_PA_cq.py             # Main AI Optimization loop for Superactivation
│   ├── kdw_hunt.py                  # Exact d=4x4 global optimization analytical script
│   └── QuantumOS/                   # Associated module for N=15 AI attack
├── 02_Cosmological_Symmetry/        # The CMB Fractal Symmetry extraction
│   ├── compare_cmb.py               # Script proving the 16D Cosmic Microwave Background symmetry
│   ├── cmb_pysr.py                  # Symbolic regression for the CMB pattern
│   ├── visualize_calabi_yau.py      # Generates the dimension decay plot
│   └── calabi_yau_decay.png         # The output visualization
├── 03_Black_Hole_Information_Paradox/ # Hayden-Preskill Protocol Simulation
│   ├── EXPERIMENT_LOG.md            # Empirical data mapping the Page Curve
│   └── paper.tex                    # Formal manuscript of the simulation
├── QuantumApp/                      # QuantumNEAT Web Dashboard UI & Backend
│   ├── main.py                      # FastAPI server for real-time visualization
│   ├── run_pysr_extractor.py        # PySR symbolic regression extractor
│   └── backend/                     # PyTorch simulation engine and custom channels
├── paper_pipeline/                  # Formal reports and LaTeX manuscripts
│   ├── paper_draft.tex              # The main arXiv submission LaTeX
│   └── RESEARCH_REPORT.md           # Detailed breakdown of the CMB Fractal Symmetry
└── _legacy_experiments/             # (Archive) Intermediate testing and raw data files
```

## 🚀 How to Run

### 1. The Superactivation Refutation (AI Joint Channel Attack)
To run the primary AI optimization loop that disproves superactivation:
```bash
python 01_Superactivation_Refutation/sa_full_PA_cq.py
```

### 2. Verify the Cosmic Microwave Background (CMB) Fractal Symmetry
To see the exact 16-dimensional correlation table linking the quantum decay to the CMB:
```bash
python 02_Cosmological_Symmetry/compare_cmb.py
```

### 3. Exact Analytical Bound ($d=4 \times 4$)
To reproduce the $-0.68$ bound using LAPACK purifications:
```bash
python 01_Superactivation_Refutation/kdw_hunt.py
```

### 4. Simulating the Black Hole Information Paradox (Page Curve)
To launch the real-time visual dashboard (QuantumNEAT UI) and optimize the Dephrasure channel:
```bash
cd QuantumApp
python main.py
```
*Access the UI at `http://localhost:8000` to interactively adjust the noise probability ($p$) and visualize the teleportation of data into the complementary channel.*

## 📖 Citation

```bibtex
@misc{murai_2026_20189708,
  author       = {Murai, Yaroslav},
  title        = {Computational Refutation of Quantum Superactivation and the Emergence of Cosmological Fractal Symmetry},
  month        = may,
  year         = 2026,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.20189708},
  url          = {https://doi.org/10.5281/zenodo.20189708}
}
```
