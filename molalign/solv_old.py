#!/usr/bin/env python3
"""
solv_correction.py  —  Solvation correction functional with homogeneous baseline

Physics:
--------
For each grid site i and its orientations o:
    E_int(i,o) = E_complex(i,o) - (E_solute + E_solvent)
    F_i = -kT ln ∑_o exp(-β E_int(i,o))
         = E_min - kT ln ∑_o exp[-β(E_int(i,o) - E_min)]

Global terms:
    F_B   = ∑_i p_i F_i ,   p_i ∝ exp(-β_site F_i)
    μ     = mean_i F_i
    H     = Var[F_i]
    Abar  = ⟨Var_o[E_int(i,o)]⟩_p
    Layer = variance of radial-bin means vs global mean (weighted by p_i)

Correction functional:
    C = (F_B - μ) + α·H + γ·Abar + β_r·Layer

For a homogeneous field (all F_i equal, no anisotropy), C → 0.

CLI Example:
    python solv_correction.py OUT \
        --T 298 --site-T 3000 \
        --alpha 0.2 --gamma 0.1 --beta_r 0.2 \
        --clip-kcal 80 --winsor-quantiles 0.01,0.99
"""

import os, json, argparse, math
import numpy as np
from glob import glob

kB = 3.166811563e-6  # Hartree/K
HARTREE_TO_KCAL = 627.509

# ---------- JSON parsing ----------

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def extract_per_site(data):
    """
    Extract site-wise orientation energies and optional grid radii.
    """
    if "solute" not in data or "solvent" not in data:
        raise ValueError("Missing solute/solvent energies")
    E_solute = float(data["solute"]["energy_Eh"])
    E_solvent = float(data["solvent"]["energy_Eh"])

    radii = None
    if "molA_grids_xyz" in data:
        coords = np.array(data["molA_grids_xyz"], float)
        center = coords.mean(axis=0)
        radii = np.linalg.norm(coords - center, axis=1)

    site_map = {}
    for k, v in data.items():
        if not (k.startswith("gp") and isinstance(v, dict)):
            continue
        try:
            idx = int(k[2:])
        except:
            continue
        Es = []
        for ck, cv in v.items():
            if isinstance(cv, dict) and "energy_Eh" in cv:
                Es.append(float(cv["energy_Eh"]))
        if Es:
            site_map[idx] = np.array(Es, float)

    return E_solute, E_solvent, site_map, radii

# ---------- Core math ----------

def per_site_stats(E_solute, E_solvent, site_map, T, clip_kcal=None, winsor=None):
    """
    Compute per-site orientation-collapsed free energies F_i and anisotropies A_i.
    """
    beta = 1.0 / (kB * T)
    idxs = sorted(site_map.keys())
    F_i_list, A_i_list, counts = [], [], []

    for i in idxs:
        E_complex = site_map[i]
        E_int = E_complex - (E_solute + E_solvent)

        # Optional robustification
        if clip_kcal is not None:
            clip_eh = clip_kcal / HARTREE_TO_KCAL
            E_int = np.clip(E_int, -clip_eh, clip_eh)

        if winsor is not None:
            qlow, qhigh = winsor
            lo, hi = np.quantile(E_int, [qlow, qhigh])
            E_int = np.clip(E_int, lo, hi)

        # Stable log-sum-exp
        Emin = float(np.min(E_int))
        Z_orient = np.sum(np.exp(-beta * (E_int - Emin)))
        F_i = Emin - kB * T * math.log(Z_orient)

        A_i = float(np.var(E_int)) if len(E_int) > 1 else 0.0

        F_i_list.append(F_i)
        A_i_list.append(A_i)
        counts.append(len(E_int))

    return np.array(idxs, int), np.array(F_i_list, float), np.array(A_i_list, float), np.array(counts, int)

def boltzmann_site_weights(F_i, T_site):
    """Site-level weights with optional tempering."""
    beta = 1.0 / (kB * T_site)
    Fmin = float(np.min(F_i))
    w = np.exp(-beta * (F_i - Fmin))
    Z = np.sum(w)
    return w / Z, Z

def radial_layering_term(r_i, F_i, p_i, nbins=24):
    """Weighted variance of radial-bin means vs global mean."""
    if r_i is None:
        return 0.0
    r, F, p = np.asarray(r_i), np.asarray(F_i), np.asarray(p_i)
    m = np.isfinite(r) & np.isfinite(F) & np.isfinite(p) & (p > 0)
    if not np.any(m):
        return 0.0
    r, F, p = r[m], F[m], p[m]
    p /= np.sum(p)

    bins = np.linspace(np.min(r), np.max(r), nbins + 1)
    idx = np.digitize(r, bins) - 1

    F_global = np.sum(p * F)
    num = 0.0
    for b in range(nbins):
        mask = idx == b
        if not np.any(mask):
            continue
        pb = np.sum(p[mask])
        if pb <= 0:
            continue
        Fb = np.sum(p[mask] * F[mask]) / pb
        num += pb * (Fb - F_global) ** 2
    return num

# ---------- Directory aggregation ----------

def analyze_directory(directory, T=298.15, T_site=None,
                      alpha=0.0, gamma=0.0, beta_r=0.0,
                      nbins=24, clip_kcal=None, winsor=None):
    files = sorted(glob(os.path.join(directory, "*.json")))
    if not files:
        raise RuntimeError(f"No JSON files in {directory}")

    F_blocks, A_blocks, r_blocks = [], [], []
    for jf in files:
        data = load_json(jf)
        E_sol, E_solvent, site_map, radii = extract_per_site(data)
        if not site_map:
            continue
        idxs, F_i, A_i, counts = per_site_stats(E_sol, E_solvent, site_map, T,
                                                clip_kcal=clip_kcal, winsor=winsor)
        F_blocks.append(F_i)
        A_blocks.append(A_i)
        if radii is not None:
            r_i = np.array([radii[i] if i < len(radii) else np.nan for i in idxs])
            r_blocks.append(r_i)

    if not F_blocks:
        raise RuntimeError("No valid grid sites found.")
    F = np.concatenate(F_blocks)
    A = np.concatenate(A_blocks)
    r = np.concatenate(r_blocks) if r_blocks else None

    T_site = T_site or T
    p, _ = boltzmann_site_weights(F, T_site)

    F_B = np.sum(p * F)
    mu = np.mean(F)
    H = np.var(F)
    Abar = np.sum(p * A)
    Layer = radial_layering_term(r, F, p, nbins) if (r is not None and beta_r != 0) else 0.0

    #C1 = F_B - mu
    C2 = alpha * H
    C3 = gamma * Abar
    #C4 = beta_r * Layer
    C4 = 0
    C1 = F_B*beta_r
    Ctot = C1 + C2 + C3 + C4

    return {
        "N_files": len(files),
        "N_sites": len(F),
        "T": T, "T_site": T_site,
        "alpha": alpha, "gamma": gamma, "beta_r": beta_r,
        "F_B_Eh": F_B,
        "mu_Eh": mu,
        "H_Eh2": H,
        "Abar_Eh2": Abar,
        "Layer_Eh2": Layer,
        "C1_Eh": C1,
        "C2_Eh": C2,
        "C3_Eh": C3,
        "C4_Eh": C4,
        "C_total_Eh": Ctot,
        "C_total_kcal": Ctot * HARTREE_TO_KCAL
    }

# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description="Solvation correction functional (homogeneous baseline).")
    ap.add_argument("directory", help="Directory with JSON files (e.g., OUT)")
    ap.add_argument("--T", type=float, default=298.15, help="Orientation temperature (K)")
    ap.add_argument("--site-T", type=float, default=None,
                    help="Site aggregation temperature (K). Default = T")
    ap.add_argument("--alpha", type=float, default=0.0, help="Weight for spatial variance")
    ap.add_argument("--gamma", type=float, default=0.0, help="Weight for orientation anisotropy")
    ap.add_argument("--beta_r", type=float, default=0.0, help="Weight for radial layering")
    ap.add_argument("--nbins", type=int, default=24, help="Radial bins")
    ap.add_argument("--clip-kcal", type=float, default=None,
                    help="Symmetric clipping of |E_int| before collapse (kcal/mol)")
    ap.add_argument("--winsor-quantiles", type=str, default=None,
                    help="Winsorization quantiles, e.g. '0.01,0.99'")
    args = ap.parse_args()

    winsor = None
    if args.winsor_quantiles:
        winsor = tuple(map(float, args.winsor_quantiles.split(",")))

    res = analyze_directory(args.directory,
                            T=args.T, T_site=args.site_T,
                            alpha=args.alpha, gamma=args.gamma, beta_r=args.beta_r,
                            nbins=args.nbins, clip_kcal=args.clip_kcal, winsor=winsor)

    print("\n=== Solvation Correction (homogeneous baseline) ===")
    print(f"Files            : {res['N_files']}")
    print(f"Grid sites       : {res['N_sites']}")
    print(f"T (K)            : {res['T']}")
    print(f"T_site (K)       : {res['T_site']}")
    print(f"α (spatial var)  : {res['alpha']}")
    print(f"γ (anisotropy)   : {res['gamma']}")
    print(f"β_r (radial)     : {res['beta_r']}")
    print("---------------------------------------------------")
    print(f"F_B   (Eh)       : {res['F_B_Eh']:.6f}")
    print(f"μ     (Eh)       : {res['mu_Eh']:.6f}")
    print(f"H=Var[F_i] (Eh²) : {res['H_Eh2']:.6e}")
    print(f"⟨A⟩ (Eh²)        : {res['Abar_Eh2']:.6e}")
    print(f"Layer (Eh²)      : {res['Layer_Eh2']:.6e}")
    print("---------------------------------------------------")
    print(f"C1=F_B-μ (Eh)    : {res['C1_Eh']:.6f}")
    print(f"C2=α·Var[F]      : {res['C2_Eh']:.6f}")
    print(f"C3=γ·⟨A⟩          : {res['C3_Eh']:.6f}")
    print(f"C4=β_r·Layer     : {res['C4_Eh']:.6f}")
    print("---------------------------------------------------")
    print(f"Total Correction : {res['C_total_Eh']:.6f} Eh")
    print(f"Total Correction : {res['C_total_kcal']:.3f} kcal/mol")
    print("===================================================")

if __name__ == "__main__":
    main()

