#!/usr/bin/env python3
"""
Kirkwood–Buff functional from TB-lite grids (Z-based hybrid model)

Z_i = Σ_α exp(-β ΔU_iα)
Z_ref = <Z> = average over gridpoints
g_i = Z_i / Z_ref

ENERGY-WEIGHTED KB:

⟨ΔU_i⟩ = Σ_α ΔU_iα exp(-β ΔU_iα) / Z_i

G12 = Σ_i (g_i - 1) · ρ · ΔV_i · ⟨ΔU_i⟩

ΔG = -kB T · G12
"""

import os
import re
import glob
import json
import argparse
import numpy as np
import pandas as pd

# ------------------ CONSTANTS ------------------

HARTREE_TO_KCAL = 627.5095
kB = 0.0019872041
RHO_N_DEFAULT = 0.033456
COORD_TOL = 1e-4


# ------------------ HELPERS ------------------

def detect_scale(fn):
    m = re.search(r"scale([0-9p]+)", fn)
    if not m:
        return None
    return float(m.group(1).replace("p", "."))


def load_dV(fn):
    df = pd.read_csv(fn)
    required = ["scale", "x", "y", "z", "dV_A3"]
    for c in required:
        if c not in df.columns:
            raise RuntimeError("dV file missing column: " + c)
    return df


def match_gp(scale, x, y, z, dV_df):
    block = dV_df[np.isclose(dV_df["scale"], scale)]
    if block.empty:
        return None
    dx = np.abs(block["x"].values - x)
    dy = np.abs(block["y"].values - y)
    dz = np.abs(block["z"].values - z)
    m = (dx < COORD_TOL) & (dy < COORD_TOL) & (dz < COORD_TOL)
    idx = block.index[m]
    return int(idx[0]) if len(idx) else None


# ------------------ ENERGY EXTRACTION ------------------

def extract_interaction_energies(jdata):
    coords = np.array(jdata["molA_grids_xyz"], float)
    E_sol = float(jdata["solute"]["energy_Eh"])
    E_solv = float(jdata["solvent"]["energy_Eh"])

    deltaE = {}

    for k, v in jdata.items():
        if not k.startswith("gp"):
            continue
        try:
            i = int(k[2:])
        except:
            continue

        block = v.values() if isinstance(v, dict) else v
        dEs = []

        for entry in block:
            try:
                Ec = float(entry["energy_Eh"])
                dU = (Ec - (E_sol + E_solv)) * HARTREE_TO_KCAL + 0.679
                dEs.append(dU)
            except:
                continue

        if dEs:
            deltaE[i] = np.array(dEs, float)

    return coords, deltaE


# ------------------ MAIN ------------------

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--dv-file", default="dV_all.csv")
    parser.add_argument("--json-glob", default="tb_lite_results_scale*.json")
    parser.add_argument("--temperature", type=float, default=298.15)
    parser.add_argument("--steepness", type=float, default=1.0)
    parser.add_argument("--rho-n", type=float, default=RHO_N_DEFAULT)
    parser.add_argument("--out-per-grid", default="kb_per_grid.csv")
    parser.add_argument("--out-per-scale", default="kb_per_scale.csv")
    args = parser.parse_args()

    T = args.temperature
    beta = args.steepness / (kB * T)
    rho = args.rho_n

    print("\n=========== KB RUN (ENERGY-WEIGHTED HYBRID MODEL) ===========")
    print(f"T = {T} K")
    print(f"β = {beta:.6f}")
    print("g_i = Z_i / <Z>")
    print("G12 = Σ (g_i - 1) · ρ · ΔV · <ΔU>")
    print("============================================================\n")

    dV_df = load_dV(args.dv_file)
    files = sorted(glob.glob(args.json_glob))

    entries = []
    Z_all = []

    # -------- PASS 1: COMPUTE Z_i AND <ΔU_i> --------
    for fn in files:
        scale = detect_scale(fn) or 1.0
        print(f"[INFO] Loading {fn} (scale={scale})")

        with open(fn) as f:
            jdata = json.load(f)

        coords, energies = extract_interaction_energies(jdata)

        for i, dEs in energies.items():

            if i >= coords.shape[0]:
                continue

            x, y, z = coords[i]
            idx = match_gp(scale, x, y, z, dV_df)
            if idx is None:
                continue

            dV = float(dV_df.loc[idx, "dV_A3"])
            #Ui_avg = float(np.mean(dEs))

            # beta from your code: beta = 1/(kBT) or 1/(RT) depending on units
                       
           
            weights = np.exp(-beta * dEs)
            Zi = float(np.sum(weights))

            # --- NEW: skip grids with zero / invalid partition function ---
            if Zi <= 0.0 or not np.isfinite(Zi):
                continue

            Ui_avg = float(np.sum(dEs * weights) / Zi)
            Zi = float(np.exp(-beta * Ui_avg))

            # also skip if Ui_avg is NaN/inf (paranoia)
            if not np.isfinite(Ui_avg):
                continue

            entries.append((scale, i, x, y, z, Zi, dV, Ui_avg))
            Z_all.append(Zi)

    if not Z_all:
        raise RuntimeError("No valid gridpoints found (all Z_i underflowed / invalid).")

    Z_avg = float(np.sum(Z_all))
    print(f"[INFO] Global <Z> = {Z_avg:.6e}\n")

    # -------- PASS 2: ENERGY-WEIGHTED KB --------
    per_grid = []
    per_scale = {}
    G12 = 0.0

    for scale, i, x, y, z, Zi, dV, Ui_avg in entries:

        g = Zi #/ Z_avg
        dG = (g - 0.0) * dV * rho # Ui_avg #* rho

        G12 += dG
        per_scale.setdefault(scale, 0.0)
        per_scale[scale] += dG

        r = float(np.sqrt(x*x + y*y + z*z))

        per_grid.append({
            "scale": scale,
            "gp": i,
            "r": r,
            "Z_i": Zi,
            "U_avg": Ui_avg,
            "g_i": g,
            "dV_A3": dV,
            "rho": rho,
            "dG12_i": dG
        })

    pd.DataFrame(per_grid).to_csv(args.out_per_grid, index=False)
    print(f"[INFO] Per-grid → {args.out_per_grid}")

    rows = []
    for s in sorted(per_scale):
        rows.append({
            "scale": s,
            "G12_scale": per_scale[s]
        })

    pd.DataFrame(rows).to_csv(args.out_per_scale, index=False)
    print(f"[INFO] Per-scale → {args.out_per_scale}")

    # -------- FINAL ENERGY-WEIGHTED FREE ENERGY --------
    dG_total = -kB * T * G12

    print("\n============ FINAL RESULT ============")
    print(f"G12(energy) = {G12:.6f}")
    print(f"ΔG = {dG_total:.6f} kcal/mol")
    print("=====================================\n")


if __name__ == "__main__":
    main()

