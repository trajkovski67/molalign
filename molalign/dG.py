#!/usr/bin/env python3
"""
Kirkwood–Buff integral from TB-lite grids + dV_all.csv
GLOBAL-NORMALIZED VERSION (Option A)

g_i = Z_i / <Z>

Where <Z> is the global average over ALL gridpoints and ALL scales.
This produces extremely stable g-values.

Energy capping:
    For scales > 1.0:
        ΔU_iα = max(ΔU_iα, cap_low)
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
kB = 0.0019872041              # kcal/mol/K
RHO_N_DEFAULT = 0.033456       # water number density [1/Å^3]
COORD_TOL = 1e-4               # Å tolerance for matching


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
            raise RuntimeError("dV file missing required column: " + c)
    return df


def match_gp(scale, x, y, z, dV_df):
    df_s = dV_df[np.isclose(dV_df["scale"], scale)]
    if df_s.empty:
        return None
    dx = np.abs(df_s["x"].values - x)
    dy = np.abs(df_s["y"].values - y)
    dz = np.abs(df_s["z"].values - z)
    mask = (dx < COORD_TOL) & (dy < COORD_TOL) & (dz < COORD_TOL)
    idx = df_s.index[mask]
    return int(idx[0]) if len(idx) else None


# ------------------ ENERGY EXTRACTION WITH CAP ------------------

def extract_interaction_energies(jdata, scale, cap_low):
    """
    Extract ΔU_iα (interaction energies), apply cap only for scale > 1.0:

        if ΔU_iα > cap_low   → set ΔU_iα = cap_low (usually cap_low = -1 kcal/mol)
    """

    coords = np.array(jdata["molA_grids_xyz"], float)
    E_sol = float(jdata["solute"]["energy_Eh"])
    E_solv = float(jdata["solvent"]["energy_Eh"])

    deltaE = {}

    for key, val in jdata.items():
        if not key.startswith("gp"):
            continue
        try:
            i = int(key[2:])
        except:
            continue

        block = val.values() if isinstance(val, dict) else val
        dEs = []

        for entry in block:
            try:
                Ec = float(entry["energy_Eh"])
                dU = (Ec - (E_sol + E_solv)) * HARTREE_TO_KCAL
            except:
                continue

            # CAP APPLIED ONLY FOR OUTER SCALES
            if scale > 10.0:
                if dU > cap_low:
                    dU = cap_low

            dEs.append(dU)

        if dEs:
            deltaE[i] = np.array(dEs, float)
            deltaE[i]=deltaE[i]+0.69

    return coords, deltaE


# ------------------ Boltzmann Z_i ------------------

def compute_Z(deltaE, beta):
    Z = {}
    for i, dEs in deltaE.items():
        w = np.exp(-beta * dEs)
        Z[i] = float(np.sum(w))
    return Z


# ------------------ MAIN ------------------

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--dv-file", default="dV_all.csv")
    parser.add_argument("--json-glob", default="tb_lite_results_scale*.json")
    parser.add_argument("--temperature", type=float, default=298.15)
    parser.add_argument("--steepness", type=float, default=1.0)
    parser.add_argument("--cap-low", type=float, default=-5.0,
                        help="Minimum allowed ΔU for scale>1.0 (default -5 kcal/mol)")
    parser.add_argument("--rho-n", type=float, default=RHO_N_DEFAULT)
    parser.add_argument("--out-per-grid", default="kb_per_grid.csv")
    parser.add_argument("--out-per-scale", default="kb_per_scale.csv")
    args = parser.parse_args()

    T = args.temperature
    beta_eff = args.steepness / (kB * T)
    cap_low = args.cap_low
    rho_n = args.rho_n

    print("\n=========== KB RUN (GLOBAL NORMALIZATION, cap) ===========")
    print(f"T = {T} K  (kBT = {kB*T:.6f} kcal/mol)")
    print(f"β_eff = {beta_eff:.6f}")
    print(f"Energy cap for scales > 1.0:  ΔU > {cap_low} → {cap_low}")
    print("===========================================================\n")

    dV_df = load_dV(args.dv_file)
    json_files = sorted(glob.glob(args.json_glob))

    entries = []   # (scale, i, x,y,z, Z_i)

    # ---------- PASS 1: Compute all Z_i ----------
    for jf in json_files:
        scale = detect_scale(os.path.basename(jf)) or 1.0
        print(f"[INFO] Reading {jf}   scale={scale}")

        with open(jf) as f:
            jdata = json.load(f)

        coords, deltaE = extract_interaction_energies(jdata, scale, cap_low)
        
        Z = compute_Z(deltaE, beta_eff)

        for i, Zi in Z.items():
            if i < coords.shape[0]:
                x, y, z = coords[i]
                entries.append((scale, i, x, y, z, Zi))

    # ----------- GLOBAL NORMALIZATION -----------
    Z_all = [Zi for (_, _, _, _, _, Zi) in entries]
    Z_avg = float(np.mean(Z_all))

    print(f"\n[INFO] Global Z_avg = {Z_avg:.6e}\n")

    # ---------- PASS 2: Compute g_i = Z_i / Z_avg, accumulate G12 ----------
    per_grid = []
    per_scale_acc = {}

    G12_total = 0.0

    for scale, i, x, y, z, Zi in entries:

        idx = match_gp(scale, x, y, z, dV_df)
        if idx is None:
            continue

        dV = float(dV_df.loc[idx, "dV_A3"])

        g = Zi #/ Z_avg
        gm1 = g-1
        dG12_i = gm1 * dV

        G12_total += dG12_i
        per_scale_acc.setdefault(scale, 0.0)
        per_scale_acc[scale] += dG12_i

        r = float(np.sqrt(x*x + y*y + z*z))

        per_grid.append({
            "scale": scale,
            "gp": i,
            "r": r,
            "Z_i": Zi,
            "g_i": g,
            "g_minus_1": gm1,
            "dV_A3": dV,
            "dG12_i_A3": dG12_i
        })

    # Save per-grid CSV
    pd.DataFrame(per_grid).to_csv(args.out_per_grid, index=False)
    print(f"[INFO] Per-grid KB written → {args.out_per_grid}")

    # ---------- PER-SCALE SUMMARY ----------
    rows_sc = []
    for s in sorted(per_scale_acc.keys()):
        Gs = per_scale_acc[s]
        dGs = -kB * T * rho_n * Gs
        rows_sc.append({
            "scale": s,
            "G12_scale_A3": Gs,
            "dG_KB_scale_kcal": dGs
        })

    pd.DataFrame(rows_sc).to_csv(args.out_per_scale, index=False)
    print(f"[INFO] Per-scale summary → {args.out_per_scale}")

    # ---------- FINAL KB RESULT ----------
    dG_KB_total = -kB * T * rho_n * G12_total

    print("\n============= KB RESULTS =============")
    print(f"Total G12     = {G12_total:.6f} Å³")
    print(f"ΔG_KB (total) = {dG_KB_total:.6f} kcal/mol")
    print("======================================\n")

if __name__ == "__main__":
    main()

