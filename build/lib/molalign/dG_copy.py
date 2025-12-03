#!/usr/bin/env python3
"""
grid_boltzmann_dV.py  (a.k.a. dG)

Combine:
  - dV_all.csv  (per-grid volume elements ΔV_i)
  - TB-lite JSON files (per-grid multiple orientations & energies)

For each grid point i at each scale s:

  1) Extract ΔE_{i,α} = E_complex(i, α) − (E_solute + E_solvent) [kcal/mol]
  2) Boltzmann-average over orientations α:
       β = 1 / (k_B T)
       Z_i = Σ_α exp(-β ΔE_{i,α})
       u_i_raw = Σ_α ΔE_{i,α} exp(-β ΔE_{i,α}) / Z_i   (if Z_i > 0)

  3) Apply physical cutoff (far-away interactions):
       if u_i_raw > -1.0 kcal/mol:
           u_i = 0.0
       else:
           u_i = u_i_raw

  4) Define:
       g_i        = exp(-u_i / (k_B T))
       g_minus_1_i = g_i - 1

  5) Look up ΔV_i from dV_all.csv by matching (scale, x, y, z)

  6) Compute two ΔG contributions per grid:

       A) "excess" form:
            dG_A_i = ρ_n * (g_i - 1) * u_i * ΔV_i

       B) "raw" form:
            dG_B_i = ρ_n * g_i * u_i * ΔV_i

  7) Sum over all grid points and JSON files:

       ΔG_A_total = Σ_i ρ_n (g_i - 1) u_i ΔV_i
       ΔG_B_total = Σ_i ρ_n g_i u_i ΔV_i

No per-shell dG is used to compute these totals. RDF g(r) is just a
volume-weighted average of discrete grid g_i in radial bins.

Outputs
=======

  * dG_per_grid.csv:
      json_file, scale, gp_index,
      x, y, z, r,
      u_int_raw_kcal, u_int_clipped_kcal,
      g_i, g_minus_1_i,
      dV_A3,
      dG_A_kcal, dG_B_kcal

  * rdf.csv:
      r_center_A,
      g_r (volume-weighted mean of g_i),
      g_minus1_r (volume-weighted mean of g_i - 1),
      n_points,
      shell_volume_A3

  * Terminal summary:
      - per JSON: stats on u_raw, counts matched/unmatched
      - global totals: ΔG_A_total, ΔG_B_total
"""

import os
import re
import json
import glob
import numpy as np
import pandas as pd
import argparse

HARTREE_TO_KCAL = 627.5095
kB = 0.0019872041       # kcal/mol/K
RHO_N_WATER = 0.033456  # molecules / Å^3
COORD_TOL = 1e-4        # Å tolerance for matching coordinates

# cut-off for interaction energy (above this → set to 0)
U_CUTOFF_KCAL = -1.5    # kcal/mol


# ================================================================
# Utility functions
# ================================================================

def detect_scale_from_filename(fn: str):
    """
    Extract scale factor from filenames like:
      tb_lite_results_scale1p05.json  -> 1.05
      tb_lite_results_scale0p9.json   -> 0.9
    Returns float or None.
    """
    m = re.search(r"scale([0-9p]+)", fn)
    if not m:
        return None
    return float(m.group(1).replace("p", "."))


def load_dV(dv_filename: str) -> pd.DataFrame:
    """
    Load dV_all.csv produced by the volume script.

    Required columns:
      scale, x, y, z, dV_A3
    """
    if not os.path.isfile(dv_filename):
        raise FileNotFoundError(f"dV file '{dv_filename}' not found")

    df = pd.read_csv(dv_filename)
    required = ["scale", "x", "y", "z", "dV_A3"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"dV file '{dv_filename}' must contain column '{col}'")

    print(f"[INFO] Loaded dV file '{dv_filename}' with {len(df)} rows.")
    print(f"       Scales present in dV: {sorted(df['scale'].unique())}\n")
    return df


def match_gridpoint(scale: float, x: float, y: float, z: float,
                    dV_df: pd.DataFrame):
    """
    Match (scale, x, y, z) to a row in dV_df within a small tolerance.
    Returns the integer index or None if not found.
    """
    df_s = dV_df[np.isclose(dV_df["scale"], scale)]
    if df_s.empty:
        return None

    dx = np.abs(df_s["x"].values - x)
    dy = np.abs(df_s["y"].values - y)
    dz = np.abs(df_s["z"].values - z)
    mask = (dx < COORD_TOL) & (dy < COORD_TOL) & (dz < COORD_TOL)
    idxs = df_s.index[mask]
    if len(idxs) == 0:
        return None
    # If multiple, just pick the first
    return int(idxs[0])


def extract_gp_energies(jdata):
    """
    From a TB-lite JSON object, extract:

      coords:  np.array (N, 3) from "molA_grids_xyz" (Å)
      deltaEs: dict i -> [ΔE_{i,α}] in kcal/mol
      n_orient_all: list of orientation counts per grid point

    ΔE_{i,α} = E_complex(i,α) – (E_solute + E_solvent)
    """
    coords = np.array(jdata["molA_grids_xyz"], dtype=float)
    E_sol = float(jdata["solute"]["energy_Eh"])
    E_solv = float(jdata["solvent"]["energy_Eh"])

    deltaEs = {}
    n_orient_all = []

    for key, val in jdata.items():
        if not isinstance(key, str):
            continue
        if not key.startswith("gp"):
            continue

        try:
            i = int(key[2:])
        except ValueError:
            continue

        gp_block = val
        if isinstance(gp_block, dict):
            entries = list(gp_block.values())
        elif isinstance(gp_block, list):
            entries = list(gp_block)
        else:
            continue

        dEs = []
        for entry in entries:
            try:
                Ec = float(entry["energy_Eh"])
                dE = (Ec - (E_sol + E_solv)) * HARTREE_TO_KCAL
                dEs.append(dE)
            except Exception:
                continue

        if dEs:
            deltaEs[i] = dEs
            n_orient_all.append(len(dEs))

    return coords, deltaEs, n_orient_all


def per_grid_boltzmann(deltaEs, T: float):
    """
    Given deltaEs: dict i -> list of ΔE_{i,α} [kcal/mol],
    compute for each grid point i:

      Z_i      = Σ_α exp(-β ΔE_{i,α})
      u_i_raw  = Σ_α ΔE_{i,α} exp(-β ΔE_{i,α}) / Z_i   (if Z_i != 0)

    Returns:
      u_int_raw: dict i -> u_i_raw  [kcal/mol]
      Z_i:       dict i -> Z_i      [dimensionless]
    """
    beta = 1.0 / (kB * T)
    u_int_raw = {}
    Z_i = {}

    for i, dEs in deltaEs.items():
        dEs_arr = np.array(dEs, dtype=float)
        w = np.exp(-beta * dEs_arr)
        Z = np.sum(w)

        if Z == 0.0:
            u_int_raw[i] = 0.0
            Z_i[i] = 0.0
        else:
            u_int_raw[i] = float(np.sum(dEs_arr * w) / Z)
            Z_i[i] = float(Z)

    return u_int_raw, Z_i


# ================================================================
# Main
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Combine TB-lite JSONs with dV_all.csv to compute per-grid "
            "Boltzmann-averaged interaction energies, g_i, and two ΔG forms:\n"
            "  A: dG_A = rho_n * (g-1) * u * dV\n"
            "  B: dG_B = rho_n * g * u * dV,\n"
            "with u clipped to 0 when u > -1 kcal/mol."
        )
    )
    parser.add_argument("--dv-file", default="dV_all.csv",
                        help="CSV file with per-grid dV (default: dV_all.csv)")
    parser.add_argument("--json-glob", default="*.json",
                        help="Glob pattern for TB-lite JSON files (default: '*.json')")
    parser.add_argument("--temperature", type=float, default=298.15,
                        help="Temperature in K (default: 298.15)")
    parser.add_argument("--rho-n", type=float, default=RHO_N_WATER,
                        help="Number density rho_n in 1/Å^3 (default: water)")
    parser.add_argument("--rdf-dr", type=float, default=0.25,
                        help="Bin width for RDF g(r) in Å (default: 0.25)")
    parser.add_argument("--out-per-grid", default="dG_per_grid.csv",
                        help="Output CSV with per-grid contributions")
    parser.add_argument("--out-rdf", default="rdf.csv",
                        help="Output CSV with radial distribution function g(r)")
    args = parser.parse_args()

    T = args.temperature
    beta = 1.0 / (kB * T)
    rho_n = args.rho_n

    print("========== GRID BOLTZMANN + dV (A & B, with cutoff) ==========")
    print(f"T            : {T:.2f} K")
    print(f"kB           : {kB:.7f} kcal/mol/K")
    print(f"beta         : {beta:.7f} 1/kcal")
    print(f"rho_n        : {rho_n:.6f} 1/Å^3")
    print(f"u cutoff     : {U_CUTOFF_KCAL:.2f} kcal/mol (u > cutoff → u = 0)")
    print(f"dV file      : {args.dv_file}")
    print(f"JSON pattern : {args.json_glob}")
    print("-------------------------------------------------------------\n")
    print("Definitions per grid point i:")
    print("  u_i_raw     = Boltzmann-avg interaction energy [kcal/mol]")
    print("  u_i         = { u_i_raw   if u_i_raw <= -1 kcal/mol")
    print("                  0.0       if u_i_raw >  -1 kcal/mol }")
    print("  g_i         = exp(-u_i / (kB * T))")
    print("  g_minus_1_i = g_i - 1")
    print("  dG_A_i      = rho_n * (g_i - 1) * u_i * dV_i")
    print("  dG_B_i      = rho_n * g_i * u_i * dV_i")
    print("  ΔG_A/B      = sum over all grid points and all JSON files\n")

    # 1) Load dV_all.csv
    dV_df = load_dV(args.dv_file)

    # 2) Collect JSON files
    json_files = sorted(glob.glob(args.json_glob))
    if not json_files:
        print("[ERROR] No JSON files found matching pattern:", args.json_glob)
        return

    print(f"[INFO] Found {len(json_files)} JSON files:")
    for f in json_files:
        print("       ", f)
    print()

    all_rows = []  # for dG_per_grid.csv
    all_r = []     # for RDF
    all_g = []
    all_g_minus_1 = []
    all_dV = []

    total_dG_A = 0.0
    total_dG_B = 0.0
    total_points = 0
    total_matched = 0
    total_unmatched = 0

    # ============================================================
    # Loop over JSON files
    # ============================================================
    for jf in json_files:
        print(f"--- Processing JSON: {jf} ---")
        with open(jf, "r") as fh:
            jdata = json.load(fh)

        scale = detect_scale_from_filename(os.path.basename(jf))
        if scale is None:
            print("  [WARN] Could not detect scale from filename; assuming scale=1.0")
            scale = 1.0

        coords, deltaEs, n_orient_all = extract_gp_energies(jdata)

        if not deltaEs:
            print("  [WARN] No gridpoint energies found; skipping this file.\n")
            continue

        n_grids = len(deltaEs)
        total_points += n_grids

        print(f"  scale                 : {scale:.4f}")
        print(f"  grid points with data : {n_grids}")
        if n_orient_all:
            print(f"  orientations per gp   : mean={np.mean(n_orient_all):.2f}, "
                  f"min={np.min(n_orient_all)}, max={np.max(n_orient_all)}")
        print("  -> Boltzmann-averaging over orientations...")

        # 1) Boltzmann average per grid (raw u)
        u_int_raw, Z_i = per_grid_boltzmann(deltaEs, T=T)
        u_vals = np.array(list(u_int_raw.values()), dtype=float)
        Z_vals = np.array(list(Z_i.values()), dtype=float)

        print(f"     u_int_raw stats [kcal/mol]: "
              f"mean={u_vals.mean():.4f}, min={u_vals.min():.4f}, max={u_vals.max():.4f}")
        print(f"     Z_i stats                : "
              f"mean={Z_vals.mean():.4e}, min={Z_vals.min():.4e}, max={Z_vals.max():.4e}")

        matched_here = 0
        unmatched_here = 0

        # 2) Per-grid contributions (A and B) with cutoff
        for i in deltaEs.keys():
            if i >= coords.shape[0]:
                unmatched_here += 1
                continue

            x, y, z = coords[i]
            idx = match_gridpoint(scale, x, y, z, dV_df)
            if idx is None:
                unmatched_here += 1
                continue

            dV = float(dV_df.loc[idx, "dV_A3"])

            # raw and clipped interaction energy
            u_raw = u_int_raw[i]
            if u_raw > U_CUTOFF_KCAL:
                u = 0.0
            else:
                u = u_raw

            # g_i = exp(-u / kT)
            g = float(np.exp(-beta * u))
            if abs(u) < 1e-12:
                g = 1.0  # enforce exactly 1 for zero interaction

            g_minus_1 = g - 1.0

            # Two ΔG definitions
            dG_A = rho_n * g_minus_1 * u * dV  # kcal/mol
            dG_B = rho_n * g * u * dV          # kcal/mol

            total_dG_A += dG_A
            total_dG_B += dG_B
            matched_here += 1

            r = float(np.sqrt(x * x + y * y + z * z))

            all_rows.append({
                "json_file": os.path.basename(jf),
                "scale": scale,
                "gp_index": i,
                "x": x,
                "y": y,
                "z": z,
                "r": r,
                "u_int_raw_kcal": u_raw,
                "u_int_clipped_kcal": u,
                "g_i": g,
                "g_minus_1_i": g_minus_1,
                "dV_A3": dV,
                "dG_A_kcal": dG_A,
                "dG_B_kcal": dG_B,
            })

            all_r.append(r)
            all_g.append(g)
            all_g_minus_1.append(g_minus_1)
            all_dV.append(dV)

        total_matched += matched_here
        total_unmatched += unmatched_here

        print(f"  matched grid points    : {matched_here}")
        print(f"  unmatched grid points  : {unmatched_here}\n")

    if not all_rows:
        print("[ERROR] No per-grid data accumulated. Check JSON ↔ dV consistency.")
        return

    # ============================================================
    # Save per-grid table
    # ============================================================
    per_grid_df = pd.DataFrame(all_rows)
    per_grid_df.to_csv(args.out_per_grid, index=False)
    print(f"[INFO] Wrote per-grid contributions to '{args.out_per_grid}'")

    # ============================================================
    # Discrete RDF (volume-weighted g and g-1 in radial shells)
    # ============================================================
    all_r_arr = np.array(all_r)
    all_g_arr = np.array(all_g)
    all_gm1_arr = np.array(all_g_minus_1)
    all_dV_arr = np.array(all_dV)

    r_min = float(all_r_arr.min())
    r_max = float(all_r_arr.max())
    dr = args.rdf_dr

    bins = np.arange(r_min, r_max + dr, dr)
    centers = 0.5 * (bins[:-1] + bins[1:])

    g_bin = []
    gm1_bin = []
    n_bin = []
    V_bin_list = []

    for b0, b1 in zip(bins[:-1], bins[1:]):
        mask = (all_r_arr >= b0) & (all_r_arr < b1)
        n_here = int(np.sum(mask))

        if not np.any(mask):
            g_bin.append(np.nan)
            gm1_bin.append(np.nan)
            n_bin.append(0)
            V_bin_list.append(0.0)
            continue

        V_shell = float(np.sum(all_dV_arr[mask]))
        if V_shell <= 0:
            g_bin.append(np.nan)
            gm1_bin.append(np.nan)
        else:
            # volume-weighted averages in this shell
            g_val = float(np.sum(all_g_arr[mask] * all_dV_arr[mask]) / V_shell)
            gm1_val = float(np.sum(all_gm1_arr[mask] * all_dV_arr[mask]) / V_shell)
            g_bin.append(g_val)
            gm1_bin.append(gm1_val)

        n_bin.append(n_here)
        V_bin_list.append(V_shell)

    rdf_df = pd.DataFrame({
        "r_center_A": centers,
        "g_r": g_bin,
        "g_minus1_r": gm1_bin,
        "n_points": n_bin,
        "shell_volume_A3": V_bin_list,
    })
    rdf_df.to_csv(args.out_rdf, index=False)
    print(f"[INFO] Wrote RDF g(r) to '{args.out_rdf}'")

    # ============================================================
    # Global summary
    # ============================================================
    print("\n============= GLOBAL SUMMARY =============")
    print(f"Total JSON files          : {len(json_files)}")
    print(f"Total grid points (ΔE)    : {total_points}")
    print(f"Total matched points      : {total_matched}")
    print(f"Total unmatched points    : {total_unmatched}")
    print("------------------------------------------")
    print(f"Total ΔG_A = Σ rho_n (g-1) u dV  = {total_dG_A:.6f} kcal/mol")
    print(f"Total ΔG_B = Σ rho_n g u dV      = {total_dG_B:.6f} kcal/mol")
    print("==========================================\n")


if __name__ == "__main__":
    main()

