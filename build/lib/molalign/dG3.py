#!/usr/bin/env python3
"""
dG — PES-based RDF solvation correction
---------------------------------------

Uses:
    - PES from TB-lite JSON files
    - RDF from Boltzmann occupation
    - Interaction energy (not total energy)
    - Bulk defined from LAST SCALE ONLY
    - g (not g-1)
    - Number density weighting
    - Final normalization by effective solvent count

Formula:

    Z_i = Σ exp(-β_eff ΔU_iα)
    g_i = Z_i / Z_bulk
    u_i = Σ ΔU_iα exp(-β_eff ΔU_iα) / Z_i

    ΔG_raw = ρ Σ g_i u_i ΔV_i
    N_eff  = ρ Σ g_i ΔV_i

    ΔG_norm = ΔG_raw / N_eff
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
RHO_N = 0.033456
COORD_TOL = 1e-4

# ------------------ HELPERS ------------------

def detect_scale(fn):
    m = re.search(r"scale([0-9p]+)", fn)
    return float(m.group(1).replace("p",".")) if m else None


def load_dV(fn):
    df = pd.read_csv(fn)
    for c in ["scale","x","y","z","dV_A3"]:
        if c not in df.columns:
            raise RuntimeError("dV_all.csv must have: scale,x,y,z,dV_A3")
    return df


def match_gp(scale,x,y,z, df):
    sub = df[np.isclose(df["scale"], scale)]
    m = (
        (np.abs(sub["x"]-x) < COORD_TOL) &
        (np.abs(sub["y"]-y) < COORD_TOL) &
        (np.abs(sub["z"]-z) < COORD_TOL)
    )
    idx = sub.index[m]
    return int(idx[0]) if len(idx) else None


def extract_PES(jdata):
    coords = np.array(jdata["molA_grids_xyz"], dtype=float)
    E_sol = float(jdata["solute"]["energy_Eh"])
    E_solv = float(jdata["solvent"]["energy_Eh"])

    deltaE = {}

    for k,val in jdata.items():
        if not k.startswith("gp"):
            continue
        try:
            i = int(k[2:])
        except:
            continue

        entries = val.values() if isinstance(val,dict) else val
        dEs = []

        for e in entries:
            try:
                Ec = float(e["energy_Eh"])
                dEs.append((Ec - (E_sol + E_solv)) * HARTREE_TO_KCAL)
            except:
                pass

        if dEs:
            deltaE[i] = np.array(dEs, float)

    return coords, deltaE


def boltzmann(deltaE, beta):
    Z = {}
    U = {}
    for i,dE in deltaE.items():
        w = np.exp(-beta * dE)
        Zi = np.sum(w)
        if Zi > 0:
            Z[i] = float(Zi)
            U[i] = float(np.sum(dE*w)/Zi)
        else:
            Z[i] = 0.0
            U[i] = 0.0
    return Z, U


# ------------------ MAIN ------------------

def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--dv-file", default="dV_all.csv")
    ap.add_argument("--json-glob", default="tb_lite_results_scale*.json")
    ap.add_argument("--temperature", type=float, default=298.15)
    ap.add_argument("--steepness", type=float, default=1.0)
    ap.add_argument("--rho-n", type=float, default=RHO_N)
    ap.add_argument("--rdf-dr", type=float, default=0.25)
    ap.add_argument("--out-per-grid", default="dG_per_grid.csv")
    ap.add_argument("--out-rdf", default="rdf.csv")
    args = ap.parse_args()

    beta_eff = args.steepness / (kB * args.temperature)
    rho = args.rho_n

    print("=========== dG RUN (Normalized, last-scale bulk) ===========")
    print(f"T = {args.temperature} K")
    print(f"β_eff = {beta_eff:.6f}")
    print("============================================================")

    dV_df = load_dV(args.dv_file)
    json_files = sorted(glob.glob(args.json_glob))
    if not json_files:
        raise RuntimeError("No JSON files found.")

    entries = []

    # ---------- PASS 1: Collect Z and u ----------
    for jf in json_files:
        scale = detect_scale(jf) or 1.0
        print(f"[INFO] Processing {os.path.basename(jf)}")

        with open(jf) as f:
            jdata = json.load(f)

        coords, deltaE = extract_PES(jdata)
        Z,U = boltzmann(deltaE, beta_eff)

        for i,(Zi,Ui) in enumerate(zip([Z[k] for k in Z],[U[k] for k in U])):
            pass

        for i in Z:
            if i >= coords.shape[0]:
                continue
            entries.append((jf, scale, i, coords[i], Z[i], U[i]))


    # ---------- BULK FROM LAST SCALE ----------
    last_scale = max(e[1] for e in entries)
    Zb = [Zi for (_,s,_,_,Zi,_) in entries if abs(s-last_scale) < 1e-6]

    if not Zb:
        raise RuntimeError("No gridpoints in last scale for Z_bulk")

    Z_bulk = float(np.mean(Zb))
    print(f"\n[INFO] Z_bulk from last scale (scale={last_scale}) = {Z_bulk:.6e}")

    # ---------- PASS 2: Integrate ----------
    rows = []
    total = 0.0

    all_r, all_g, all_dV = [], [], []

    for jf,scale,i,(x,y,z),Zi,Ui in entries:

        idx = match_gp(scale,x,y,z,dV_df)
        if idx is None:
            continue

        dV = float(dV_df.loc[idx,"dV_A3"])
        g  = Zi / Z_bulk
        dG = rho * g * Ui * dV

        r = np.sqrt(x*x + y*y + z*z)

        total += dG
        all_r.append(r)
        all_g.append(g)
        all_dV.append(dV)

        rows.append({
            "json": os.path.basename(jf),
            "scale": scale,
            "gp": i,
            "r": r,
            "Z_i": Zi,
            "g_i": g,
            "u_i": Ui,
            "dV": dV,
            "dG": dG
        })

    df = pd.DataFrame(rows)
    df.to_csv(args.out_per_grid, index=False)
    print(f"[INFO] Per-grid written → {args.out_per_grid}")

    # ---------- RDF ----------
    r = np.array(all_r)
    g = np.array(all_g)
    dV = np.array(all_dV)

    bins = np.arange(r.min(), r.max()+args.rdf_dr, args.rdf_dr)
    centers = 0.5*(bins[:-1]+bins[1:])

    Gbin, Nbin, Vbin = [],[],[]

    for a,b in zip(bins[:-1], bins[1:]):
        m = (r>=a) & (r<b)
        V = np.sum(dV[m])
        if V > 0:
            Gbin.append(np.sum(g[m]*dV[m]) / V)
        else:
            Gbin.append(np.nan)
        Nbin.append(int(np.sum(m)))
        Vbin.append(V)

    rdf = pd.DataFrame({
        "r_center": centers,
        "g_r": Gbin,
        "n_points": Nbin,
        "shell_volume": Vbin
    })

    rdf.to_csv(args.out_rdf, index=False)
    print(f"[INFO] RDF written → {args.out_rdf}")

    # ---------- NORMALIZATION ----------
    Neff = rho * np.sum(np.array(all_g) * np.array(all_dV))

    print("\n========== SUMMARY ==========")
    print(f"ΔG_raw   = {total:.6f} kcal/mol")
    print(f"N_eff    = {Neff:.6f}")
    if Neff > 0:
        print(f"ΔG_norm  = {total/Neff:.6f} kcal/mol")
    else:
        print("ΔG_norm  = undefined (N_eff = 0)")
    print("============================")


if __name__ == "__main__":
    main()

