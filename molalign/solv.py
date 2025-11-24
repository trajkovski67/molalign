#!/usr/bin/env python3
"""
solv_correction_soft.py — Generalized damped Boltzmann solvation correction

Functional:
-----------
C_add = c1 * F_soft(κ)
       + α * Var_w[F_i]
       + γ * Ā
       + δ * C(ℓ)

Definitions:
------------
F_soft(κ) = - κ kT ln Σ_i w_i exp(-F_i / (κ kT))
Var_w[F_i] = weighted variance of site free energies
Ā          = weighted mean orientation anisotropy
C(ℓ)        = Σ_{i,j} w_i w_j (F_i - ⟨F⟩_w)(F_j - ⟨F⟩_w) exp[-r_ij² / (2ℓ²)]

CLI example:
------------
python solv_correction_soft.py OUT \
    --T 298 --site-T 3000 \
    --c1 1.0 --alpha 0.2 --gamma 0.1 --delta 0.1 \
    --kappa 1.5 --corr-length 2.0
"""

import os
import json
import argparse
import math
import numpy as np
from glob import glob
from itertools import combinations

# ---------- Constants ----------
kB = 3.166811563e-6  # Hartree/K
HARTREE_TO_KCAL = 627.509

# ---------- I/O ----------
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def extract_per_site(data):
    """Extract per-site orientation energies and optional grid coordinates."""
    if "solute" not in data or "solvent" not in data:
        raise ValueError("Missing solute/solvent energies")

    E_solute = float(data["solute"]["energy_Eh"])
    E_solvent = float(data["solvent"]["energy_Eh"])

    coords = np.array(data.get("molA_grids_xyz", []), float) if "molA_grids_xyz" in data else None

    site_map = {}
    for k, v in data.items():
        if not (k.startswith("gp") and isinstance(v, dict)):
            continue
        try:
            idx = int(k[2:])
        except ValueError:
            continue
        Es = []
        for ck, cv in v.items():
            if isinstance(cv, dict) and "energy_Eh" in cv:
                Es.append(float(cv["energy_Eh"]))
        if Es:
            site_map[idx] = np.array(Es, float)

    return E_solute, E_solvent, site_map, coords

# ---------- Core calculations ----------

def per_site_stats(E_solute, E_solvent, site_map, T):
    """Compute per-site orientation-collapsed free energies F_i and anisotropies A_i."""
    beta = 1.0 / (kB * T)
    idxs = sorted(site_map.keys())
    F_i_list, A_i_list = [], []

    for i in idxs:
        E_complex = site_map[i]
        E_int = E_complex - (E_solute + E_solvent)
        Emin = float(np.min(E_int))
        Z_orient = np.sum(np.exp(-beta * (E_int - Emin)))
        F_i = Emin - kB * T * math.log(Z_orient)
        A_i = float(np.var(E_int)) if len(E_int) > 1 else 0.0
        F_i_list.append(F_i)
        A_i_list.append(A_i)

    return np.array(idxs, int), np.array(F_i_list, float), np.array(A_i_list, float)

def boltzmann_site_weights(F_i, T_site):
    """Boltzmann site weights."""
    beta = 1.0 / (kB * T_site)
    Fmin = float(np.min(F_i))
    w = np.exp(-beta * (F_i - Fmin))
    w /= np.sum(w)
    return w

def F_soft(F_i, w, T, kappa):
    """Compute damped Boltzmann (soft-min) free energy."""
    tau = kappa * kB * T
    F_shift = np.min(F_i)
    logsum = np.log(np.sum(w * np.exp(-(F_i - F_shift) / tau)))
    return -tau * logsum + F_shift

def weighted_variance(F_i, w):
    """Weighted variance of site free energies."""
    mean = np.sum(w * F_i)
    return np.sum(w * (F_i - mean) ** 2)

def weighted_mean(x, w):
    return np.sum(w * x)

def correlation_term(F_i, coords, w, ell):
    """Compute short-range correlation correction C(ℓ)."""
    if coords is None or ell <= 0.0:
        return 0.0
    n = len(F_i)
    meanF = np.sum(w * F_i)
    F_dev = F_i - meanF
    corr = 0.0
    for i, j in combinations(range(n), 2):
        r_ij = np.linalg.norm(coords[i] - coords[j])
        weight = math.exp(-r_ij ** 2 / (2 * ell ** 2))
        corr += w[i] * w[j] * F_dev[i] * F_dev[j] * weight
    # Double count pairs (i,j) and (j,i)
    corr *= 2.0
    # Add diagonal self terms
    corr += np.sum(w * w * F_dev * F_dev)
    return corr

# ---------- Directory aggregation ----------

def analyze_directory(directory, T=298.15, T_site=None,
                      c1=1.0, alpha=0.0, gamma=0.0, delta=0.0,
                      kappa=1.0, corr_length=0.0):
    files = sorted(glob(os.path.join(directory, "*.json")))
    if not files:
        raise RuntimeError(f"No JSON files in {directory}")

    F_blocks, A_blocks, coords_all = [], [], []
    for jf in files:
        data = load_json(jf)
        E_sol, E_solvent, site_map, coords = extract_per_site(data)
        if not site_map:
            continue
        idxs, F_i, A_i = per_site_stats(E_sol, E_solvent, site_map, T)
        F_blocks.append(F_i)
        A_blocks.append(A_i)
        if coords is not None:
            coords_all.append(coords[:len(F_i)])

    F = np.concatenate(F_blocks)
    A = np.concatenate(A_blocks)
    coords_all = np.concatenate(coords_all) if coords_all else None

    T_site = T_site or T
    w = boltzmann_site_weights(F, T_site)

    # Compute terms
    Fsoft = F_soft(F, w, T, kappa)
    VarF = weighted_variance(F, w)
    Abar = weighted_mean(A, w)
    Corr = correlation_term(F, coords_all, w, corr_length) if delta != 0 else 0.0

    # Total correction
    Cadd = c1 * Fsoft + alpha * VarF + gamma * Abar + delta * Corr

    return {
        "N_files": len(files),
        "N_sites": len(F),
        "T": T,
        "T_site": T_site,
        "c1": c1,
        "alpha": alpha,
        "gamma": gamma,
        "delta": delta,
        "kappa": kappa,
        "corr_length": corr_length,
        "F_soft_Eh": Fsoft,
        "VarF_Eh2": VarF,
        "Abar_Eh2": Abar,
        "Corr_Eh2": Corr,
        "C_total_Eh": Cadd,
        "C_total_kcal": Cadd * HARTREE_TO_KCAL
    }

# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description="Damped Boltzmann solvation correction functional.")
    ap.add_argument("directory", help="Directory with JSON files (e.g., OUT)")
    ap.add_argument("--T", type=float, default=298.15, help="Orientation temperature (K)")
    ap.add_argument("--site-T", type=float, default=None, help="Site aggregation temperature (K)")
    ap.add_argument("--c1", type=float, default=1.0, help="Coefficient for F_soft(κ)")
    ap.add_argument("--alpha", type=float, default=0.0, help="Coefficient for Var_w[F_i]")
    ap.add_argument("--gamma", type=float, default=0.0, help="Coefficient for ⟨A_i⟩")
    ap.add_argument("--delta", type=float, default=0.0, help="Coefficient for correlation term C(ℓ)")
    ap.add_argument("--kappa", type=float, default=1.0, help="Soft-min temperature scaling (κ)")
    ap.add_argument("--corr-length", type=float, default=0.0, help="Correlation length ℓ (Å)")
    args = ap.parse_args()

    res = analyze_directory(
        args.directory,
        T=args.T, T_site=args.site_T,
        c1=args.c1, alpha=args.alpha, gamma=args.gamma, delta=args.delta,
        kappa=args.kappa, corr_length=args.corr_length
    )

    print("\n=== Damped Boltzmann Solvation Correction ===")
    print(f"Files             : {res['N_files']}")
    print(f"Grid sites        : {res['N_sites']}")
    print(f"T (K)             : {res['T']}")
    print(f"T_site (K)        : {res['T_site']}")
    print(f"κ (soft-min)      : {res['kappa']}")
    print(f"ℓ (Å)             : {res['corr_length']}")
    print("---------------------------------------------------")
    print(f"F_soft(κ) (Eh)    : {res['F_soft_Eh']:.6f}")
    print(f"Var_w[F_i] (Eh²)  : {res['VarF_Eh2']:.6e}")
    print(f"⟨A⟩ (Eh²)         : {res['Abar_Eh2']:.6e}")
    print(f"C(ℓ) (Eh²)        : {res['Corr_Eh2']:.6e}")
    print("---------------------------------------------------")
    print(f"C_total (Eh)      : {res['C_total_Eh']:.6f}")
    print(f"C_total (kcal/mol): {res['C_total_kcal']:.3f}")
    print("===================================================")

if __name__ == "__main__":
    main()

