#!/usr/bin/env python3
"""
dG : PES-based RDF correction
Bulk defined from LAST SCALE ONLY.

Correction formula:
    ΔG_corr = ρ Σ (g_i - 1) u_i dV

Definitions:
    ΔU_{iα} = E_complex - (E_solute + E_solvent)
    beta_eff = s / (kB T)
    Z_i     = Σ exp(-beta_eff ΔU)
    g_i     = Z_i / Z_bulk
    u_i     = Σ ΔU exp(-beta_eff ΔU) / Z_i
"""

import os, re, glob, json, argparse
import numpy as np
import pandas as pd

# ================= CONSTANTS =================

HARTREE_TO_KCAL = 627.5095
kB = 0.0019872041
RHO_N = 0.033456
COORD_TOL = 1e-4

# ================= UTILITIES =================

def detect_scale(fn):
    m = re.search(r"scale([0-9p]+)", fn)
    return float(m.group(1).replace("p",".")) if m else None


def load_dV(filename):
    df = pd.read_csv(filename)
    for c in ["scale","x","y","z","dV_A3"]:
        if c not in df.columns:
            raise ValueError("dV_all.csv needs columns: scale,x,y,z,dV_A3")
    return df


def match_gp(scale,x,y,z,dV_df):
    df = dV_df[np.isclose(dV_df["scale"], scale)]
    m = (
        (np.abs(df["x"]-x) < COORD_TOL) &
        (np.abs(df["y"]-y) < COORD_TOL) &
        (np.abs(df["z"]-z) < COORD_TOL)
    )
    idx = df.index[m]
    return int(idx[0]) if len(idx) else None


def extract_energies(jdata):
    coords = np.array(jdata["molA_grids_xyz"], dtype=float)
    E_sol = float(jdata["solute"]["energy_Eh"])
    E_solv = float(jdata["solvent"]["energy_Eh"])

    deltaEs = {}
    for k,v in jdata.items():
        if not k.startswith("gp"):
            continue
        try:
            i = int(k[2:])
        except:
            continue

        entries = v.values() if isinstance(v,dict) else v
        dEs = []
        for e in entries:
            try:
                Ec = float(e["energy_Eh"])
                dEs.append((Ec - (E_sol+E_solv)) * HARTREE_TO_KCAL)
            except:
                pass

        if dEs:
            deltaEs[i] = dEs

    return coords, deltaEs


def boltzmann(deltaEs, beta):
    Z = {}
    U = {}
    for i, dEs in deltaEs.items():
        d = np.array(dEs)
        w = np.exp(-beta * d)
        z = np.sum(w)

        if z > 0:
            Z[i] = float(z)
            U[i] = float(np.sum(d*w)/z)
        else:
            Z[i] = 0.0
            U[i] = 0.0
    return Z, U


# ================= MAIN =================

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

    beta = args.steepness / (kB * args.temperature)
    rho = args.rho_n

    print("=========== dG RUN (bulk = last scale) ===========")
    print(f"T = {args.temperature} K")
    print(f"steepness = {args.steepness}")
    print("=================================================\n")

    dV_df = load_dV(args.dv_file)
    jsons = sorted(glob.glob(args.json_glob))
    if not jsons:
        raise RuntimeError("No JSON files found")

    entries = []

    # ---------- PASS 1: collect (Z, u) ----------
    for jf in jsons:
        scale = detect_scale(jf) or 1.0
        with open(jf) as f:
            data = json.load(f)

        coords, deltaEs = extract_energies(data)
        Z, U = boltzmann(deltaEs, beta)

        for i in Z:
            if i >= coords.shape[0]:
                continue
            entries.append((jf, scale, i, coords[i], Z[i], U[i]))

    # ---------- BULK FROM LAST SHELL ----------
    last_scale = max([e[1] for e in entries])
    Z_bulk_vals = [Zi for (_,s,_,_,Zi,_) in entries if abs(s-last_scale) < 1e-6]

    if not Z_bulk_vals:
        raise RuntimeError("No points at final scale for Z_bulk")

    Z_bulk = float(np.mean(Z_bulk_vals))
    print(f"[INFO] Z_bulk from last scale (scale={last_scale}) = {Z_bulk:.6e}\n")

    # ---------- PASS 2: integrate correction ----------
    rows = []
    total = 0.0
    all_r, all_g, all_W, all_dV = [], [], [], []

    for jf, scale, i, (x,y,z), Zi, Ui in entries:
        idx = match_gp(scale,x,y,z,dV_df)
        if idx is None:
            continue

        dV = float(dV_df.loc[idx,"dV_A3"])

        g = Zi / Z_bulk
        W = -(1/beta) * np.log(Zi + 1e-300)

        dG = rho * (g - 1) * Ui * dV
        total += dG

        r = np.sqrt(x*x + y*y + z*z)

        rows.append({
            "json": os.path.basename(jf),
            "scale": scale,
            "gp": i,
            "r": r,
            "Z_i": Zi,
            "g_i": g,
            "W_i": W,
            "u_i": Ui,
            "dV": dV,
            "dG": dG
        })

        all_r.append(r)
        all_g.append(g)
        all_W.append(W)
        all_dV.append(dV)

    # ---------- Save ----------
    pd.DataFrame(rows).to_csv(args.out_per_grid, index=False)
    print(f"[INFO] Per-grid output → {args.out_per_grid}")

    # ---------- RDF ----------
    r = np.array(all_r)
    g = np.array(all_g)
    W = np.array(all_W)
    dV = np.array(all_dV)

    bins = np.arange(r.min(), r.max()+args.rdf_dr, args.rdf_dr)
    centers = 0.5*(bins[:-1]+bins[1:])

    Gbin, Wbin, Nbin, Vbin = [],[],[],[]

    for a,b in zip(bins[:-1], bins[1:]):
        m = (r>=a) & (r<b)
        V = np.sum(dV[m])
        if V > 0:
            Gbin.append(np.sum(g[m]*dV[m]) / V)
            Wbin.append(np.sum(W[m]*dV[m]) / V)
        else:
            Gbin.append(np.nan)
            Wbin.append(np.nan)
        Nbin.append(np.sum(m))
        Vbin.append(V)

    rdf = pd.DataFrame({
        "r_center": centers,
        "g_r": Gbin,
        "W_r": Wbin,
        "n_points": Nbin,
        "shell_volume": Vbin
    })

    rdf.to_csv(args.out_rdf, index=False)
    print(f"[INFO] RDF → {args.out_rdf}")

    print("\n========== SUMMARY ==========")
    print(f"ΔG_corr = {total:.6f} kcal/mol")
    print("============================")


if __name__ == "__main__":
    main()

