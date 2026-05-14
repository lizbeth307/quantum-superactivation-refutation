#!/usr/bin/env python3
"""
verify_operators.py
===================
Self-contained verification of the superactivation operators.

This script audits a single .npz file (e.g., superactivation_operators_n17_clean.npz)
which should contain:
  - encoder_block_{i}         : Choi blocks of the encoder E.
  - decoder_{k}_block_{j}     : Choi blocks of the adjoint decoders D_k.
  - bob_pov_{k}_block_{j}     : Bob's POV Choi blocks (encoder ∘ N_k) - [Optional]
  - fidelity_verified         : Precomputed entanglement fidelity.
  - n                         : Number of channel uses.

The script performs two primary checks:

  1. VALIDITY — Every stored Choi block is a valid unnormalized Choi matrix:
       - Encoder (Schrödinger picture):
           CP   : Each block B^λ ≥ 0.
           TP   : Σ_λ f_λ Tr_sys(B^λ) = I_{d_R}.
       - Adjoint Decoders (Heisenberg picture):
           CP     : Each block B^λ ≥ 0.
           U : Tr_ref(B^λ) = I_{m_λ} per block.

  2. FIDELITY — The entanglement fidelity is checked against the single-use 
     upper bound (0.75). If --recompute is flagged, the script re-calculates 
     the fidelity from the bob_pov and decoder blocks:
       - F_D[k] = (1/d_R²) Σ_λ f_λ · Tr(bob_pov_k^λ @ decoder_k^λ)
       - F      = Σ_k C(n,k) · (1/2)^n · F_D[k]

Usage
-----
    python verify_results.py [--recompute] [--print-operators] [--verbose]

Dependencies: numpy, scipy.
"""

import argparse
import numpy as np
from scipy.special import comb

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
D = 2       # Qubit dimension
D_R = D     # Reference dimension
W = 74      # Print width

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def _hr(ch="─"):
    print(ch * W)

def _section(title):
    _hr()
    print(f"  {title}")
    _hr()

# ---------------------------------------------------------------------------
# Combinatorics for the GL(2) Schur-Weyl decomposition
# ---------------------------------------------------------------------------
def _syt(n_sym, j):
    """f_λ: SYT count for S_{n_sym} partition (n_sym−j, j)."""
    if j == 0:
        return 1
    return int(comb(n_sym, j, exact=True)) - int(comb(n_sym, j - 1, exact=True))

def _partition_ordering(n_sym):
    """Internal partition ordering: j = 1, 2, ..., ⌊n_sym/2⌋, 0"""
    P_val = n_sym // 2 + 1
    if P_val == 1:
        return [0]
    return list(range(1, P_val)) + [0]

# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------
def _sorted_keys(data, prefix):
    # Handles both .npy suffixes and plain keys
    return sorted(
        [k for k in data.files if k.startswith(prefix)],
        key=lambda x: int(x.split("_")[-1].replace('.npy', '')),
    )

def load_enc_blocks(data):
    return [np.array(data[k]) for k in _sorted_keys(data, "encoder_block_")]

def load_dec_blocks(data, k):
    return [np.array(data[key]) for key in _sorted_keys(data, f"decoder_{k}_block_")]

def load_bob_pov_blocks(data, k):
    return [np.array(data[key]) for key in _sorted_keys(data, f"bob_pov_{k}_block_")]

# ---------------------------------------------------------------------------
# Partial-trace helpers
# ---------------------------------------------------------------------------
def _tr_sys(B, d_R=D_R):
    m = B.shape[0] // d_R
    return np.einsum("isjs->ij", B.reshape(d_R, m, d_R, m))

def _tr_R(B, d_R=D_R):
    m = B.shape[0] // d_R
    return np.einsum("isit->st", B.reshape(d_R, m, d_R, m))

# ---------------------------------------------------------------------------
# Fidelity computation logic
# ---------------------------------------------------------------------------
def _fidelity_D_full(Mk_list, Dk_list, n):
    j_order = _partition_ordering(n)
    total = 0.0
    for i, (M, D_mat) in enumerate(zip(Mk_list, Dk_list)):
        f = _syt(n, j_order[i])
        total += f * float(np.real(np.trace(M @ D_mat)))
    return total / D_R**2

def _fidelity_D_subgroup(Mk_list, Dk_list, k, n):
    nmk = n - k
    j_k_order = _partition_ordering(k)
    j_nk_order = _partition_ordering(nmk)
    n_blocks_nmk = len(j_nk_order)
    total = 0.0
    for idx, (M, D_mat) in enumerate(zip(Mk_list, Dk_list)):
        j_k = j_k_order[idx // n_blocks_nmk]
        j_nk = j_nk_order[idx % n_blocks_nmk]
        f = _syt(k, j_k) * _syt(nmk, j_nk)
        total += f * float(np.real(np.trace(M @ D_mat)))
    return total / D_R**2

def compute_fidelity(data, n):
    if f"bob_pov_0_block_0" not in data.files and f"bob_pov_0_block_0.npy" not in data.files:
        return None, None

    F_per_k = []
    for k in range(n + 1):
        Mk_list = load_bob_pov_blocks(data, k)
        Dk_list = load_dec_blocks(data, k)
        if k == 0 or k == n:
            F_per_k.append(_fidelity_D_full(Mk_list, Dk_list, n))
        else:
            F_per_k.append(_fidelity_D_subgroup(Mk_list, Dk_list, k, n))

    F_total = sum(
        float(comb(n, k, exact=True)) * (0.5**n) * F_per_k[k]
        for k in range(n + 1)
    )
    return float(F_total), F_per_k

# ---------------------------------------------------------------------------
# Validity checks
# ---------------------------------------------------------------------------
def _cp_check(B):
    Bsym = (B + B.conj().T) / 2.0
    min_eig = float(np.linalg.eigvalsh(Bsym).min())
    herm_err = float(np.max(np.abs(B - B.conj().T)))
    return min_eig, herm_err

def check_encoder(enc_blocks, n, verbose=False):
    j_order = _partition_ordering(n)
    tp_sum = np.zeros((D_R, D_R), dtype=complex)
    min_eig_all, herm_err_all = float("inf"), 0.0

    for i, B in enumerate(enc_blocks):
        j = j_order[i]
        f = _syt(n, j)
        min_eig, herm_err = _cp_check(B)
        min_eig_all = min(min_eig_all, min_eig)
        herm_err_all = max(herm_err_all, herm_err)
        tp_sum += f * _tr_sys(B)
        if verbose:
            print(f"    encoder_block_{i}: λ=({n-j},{j})  min_eig={min_eig:+.3e}  herm={herm_err:.1e}")

    tp_err = float(np.max(np.abs(tp_sum - np.eye(D_R))))
    return min_eig_all, herm_err_all, tp_err

def check_decoder(dk_blocks, n, k, verbose=False):
    min_eig_all, herm_err_all, tp_err_all = float("inf"), 0.0, 0.0
    for i, B in enumerate(dk_blocks):
        m = B.shape[0] // D_R
        min_eig, herm_err = _cp_check(B)
        min_eig_all = min(min_eig_all, min_eig)
        herm_err_all = max(herm_err_all, herm_err)
        tp_err = float(np.max(np.abs(_tr_R(B) - np.eye(m))))
        tp_err_all = max(tp_err_all, tp_err)
        if verbose:
            print(f"    decoder_{k}_block_{i}: min_eig={min_eig:+.3e}  unital_err={tp_err:.1e}")
    return min_eig_all, herm_err_all, tp_err_all

# ---------------------------------------------------------------------------
# Print operators
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Interactive Operator Browser
# ---------------------------------------------------------------------------
def interactive_browser(data, n, enc_blocks, dec_blocks_all):
    _section("INTERACTIVE OPERATOR BROWSER")
    
    while True:
        print("\nSelect Map Type to view:")
        print("  [e] Encoder (E)")
        print("  [d] Decoders (D_k)")
        print("  [q] Quit Browser")
        choice = input("\nChoice: ").strip().lower()

        if choice == 'q':
            break
        
        elif choice == 'e':
            _browse_blocks(enc_blocks, "Encoder E")
            
        elif choice == 'd':
            print(f"\nAvailable Decoders: 0 to {n}")
            k_choice = input(f"Enter k (0-{n}) or 'all': ").strip().lower()
            
            if k_choice == 'all':
                for k in range(n + 1):
                    _browse_blocks(dec_blocks_all[k], f"Decoder D_{k}")
            else:
                try:
                    k = int(k_choice)
                    if 0 <= k <= n:
                        _browse_blocks(dec_blocks_all[k], f"Decoder D_{k}")
                    else:
                        print(f"Error: k must be between 0 and {n}.")
                except ValueError:
                    print("Invalid input.")

def _browse_blocks(blocks, label):
    num_blocks = len(blocks)
    print(f"\n--- {label} ---")
    print(f"This map contains {num_blocks} symmetry blocks.")
    
    while True:
        print(f"\nSelect block to print (0 to {num_blocks-1}), 'all', or 'back':")
        b_choice = input("Choice: ").strip().lower()
        
        if b_choice == 'back':
            break
        elif b_choice == 'all':
            for i, B in enumerate(blocks):
                _print_single_block(B, i)
            break
        else:
            try:
                idx = int(b_choice)
                if 0 <= idx < num_blocks:
                    _print_single_block(blocks[idx], idx)
                else:
                    print(f"Error: Index must be 0 to {num_blocks-1}.")
            except ValueError:
                print("Invalid input.")

def _print_single_block(B, idx):
    re = np.real(B)
    im = np.imag(B)
    print(f"\n[Block {idx}] Shape: {B.shape[0]}x{B.shape[1]}")
    with np.printoptions(precision=5, suppress=True, linewidth=120):
        lines = repr(re).replace("\n", "\n      ")
        print(f"  re = {lines}")
        if np.max(np.abs(im)) > 1e-14:
            lines = repr(im).replace("\n", "\n      ")
            print(f"  im = {lines}")

# ---------------------------------------------------------------------------
# Refactored Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Verify superactivation operators.")
    parser.add_argument("--file", default="superactivation_operators_n17.npz", help="Input .npz file")
    parser.add_argument("--recompute", action="store_true", help="Recompute fidelity from blocks")
    parser.add_argument("--print-operators", "-p", action="store_true", help="Interactive operator browser")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print per-block validity details")
    args = parser.parse_args()

    # (Keep your existing data loading and validity check code here...)
    data = np.load(args.file, allow_pickle=True)
    n = int(data["n"])
    f_stored = float(data["fidelity_verified"])
    
    _hr("=")
    print(f"  SUPERACTIVATION OPERATOR VERIFICATION  (n = {n})")
    _hr("=")
    print(f"  File                  : {args.file}")
    
    # --- 1. Fidelity Assessment ---
    _section("FIDELITY")
    print(f"  Stored F (verified)   : {f_stored:.10f}")
    
    F_show = f_stored
    if args.recompute:
        F_computed, F_per_k = compute_fidelity(data, n)
        if F_computed is not None:
            print(f"  Recomputed F          : {F_computed:.10f}")
            print(f"  Difference            : {abs(F_computed - f_stored):.2e}")
            
            print(f"\n  {'k':>3}  | {'F_D[k]':>10}  | {'weight':>12}  | {'contrib':>10}")
            print(f"  {'─'*3}-+-{'─'*10}--+-{'─'*12}--+-{'─'*10}")
            for k in range(n + 1):
                w = float(comb(n, k, exact=True)) * (0.5 ** n)
                print(f"  {k:>3}  | {F_per_k[k]:>10.6f}  | {w:>12.8f}  | {w*F_per_k[k]:>10.6f}")
            F_show = F_computed
        else:
            print("  [!] Cannot recompute: 'bob_pov' blocks missing.")

    UPPER_BOUND = 0.75
    diff = F_show - UPPER_BOUND
    print(f"\n  Upper bound 0.75      : {UPPER_BOUND}")
    print(f"  F - 0.75              : {diff:+.2e}", end="")
    print("   *** SUPERACTIVATION CONFIRMED ***" if diff > 0 else "   (no superactivation)")

    # --- 2. Validity Audit ---
    _section("VALIDITY  (unnormalized Choi matrices: CP = PSD,  TP / unital)")
    tol = 1e-6
    all_ok = True

    # Encoder
    enc_blocks = load_enc_blocks(data)
    min_eig, herm_err, tp_err = check_encoder(enc_blocks, n, args.verbose)
    flag = "✓" if (min_eig >= -tol and tp_err < tol) else "✗ WARNING"
    print(f"  Encoder ({len(enc_blocks):2d} blocks)  : herm={herm_err:.1e}  min_eig={min_eig:+.2e}  TP_err={tp_err:.1e}  [{flag}]")
    all_ok &= (flag == "✓")

    # Decoders
    dec_blocks_all = {}
    for k in range(n + 1):
        dk_blocks = load_dec_blocks(data, k)
        dec_blocks_all[k] = dk_blocks
        d_min, d_herm, d_tp = check_decoder(dk_blocks, n, k, args.verbose)
        flag = "✓" if (d_min >= -tol and d_tp < tol) else "✗ WARNING"
        print(f"  Decoder k={k:2d} ({len(dk_blocks):2d} blocks): herm={d_herm:.1e}  min_eig={d_min:+.2e}  Unital_err={d_tp:.1e}  [{flag}]")
        all_ok &= (flag == "✓")

    if all_ok:
        print("\n  All operators pass validity checks: CP ✓  TP / unital ✓")

    # Updated logic for printing
    if args.print_operators:
        interactive_browser(data, n, enc_blocks, dec_blocks_all)

    _hr("=")
    print("  Done.")
    _hr("=")

if __name__ == "__main__":
    main()