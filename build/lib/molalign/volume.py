#!/usr/bin/env python3
import os
import re
import numpy as np
import pandas as pd

BOHR_TO_ANG = 0.52917721092
BOHR2_TO_ANG2 = BOHR_TO_ANG**2
ANG3_TO_BOHR3 = (1.0/BOHR_TO_ANG)**3


# ----------------------------------------------------------
# Extract numeric scale: e.g. water_scale1p05_cavity.csv → 1.05
# ----------------------------------------------------------
def detect_scale(fn):
    m = re.search(r"scale([0-9p]+)", fn)
    if not m:
        return None
    return float(m.group(1).replace("p", "."))


# ----------------------------------------------------------
# Robust cavity.csv reader (CRITICAL FIX)
#   • Never uses pandas.read_csv
#   • Removes trailing commas
#   • Ensures EXACT 10 columns
#   • Converts owner → int
#   • Converts units BOHR → Å
# ----------------------------------------------------------
def load_csv(path):

    rows = []
    with open(path, "r") as f:
        header = f.readline().strip().split(",")   # ignore header

        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(",")

            # remove unlimited trailing empty columns
            while parts and parts[-1] == "":
                parts.pop()

            # enforce EXACT column count:
            # ngrid, numbering, x, y, z, owner, radius, area, w_leb, f
            if len(parts) < 10:
                parts += ["0"] * (10 - len(parts))
            parts = parts[:10]

            rows.append(parts)

    df = pd.DataFrame(rows, columns=[
        "ngrid","numbering","x","y","z",
        "owner","radius","area","w_leb","f"
    ])

    # Convert everything to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Owner MUST be integer
    df["owner"] = df["owner"].astype(int)

    # Convert to Å
    df["x"] *= BOHR_TO_ANG
    df["y"] *= BOHR_TO_ANG
    df["z"] *= BOHR_TO_ANG
    df["radius"] *= BOHR_TO_ANG
    df["area"] *= BOHR2_TO_ANG2

    return df


# ----------------------------------------------------------
# MAIN PROGRAM
# ----------------------------------------------------------
def main():

    root = os.getcwd()
    print(f"\nScanning: {root}\n")

    # Identify cavity CSVs
    files = []
    for fn in os.listdir(root):
        if fn.endswith("_cavity.csv"):
            sc = detect_scale(fn)
            if sc is not None:
                files.append((sc, fn))

    if not files:
        print("No cavity CSV files found.")
        return

    # Sort by scale
    files.sort(key=lambda x: x[0])
    scales = [s for s, _ in files]

    print("Detected scales:")
    for s, fn in files:
        print(f"  {s:.4f}   {fn}")
    print()

    m = len(scales)

    # ------------------------------------------------------
    # Load CSVs
    # ------------------------------------------------------
    data = []
    for s, fn in files:
        df = load_csv(fn)
        data.append((s, df))

    # ------------------------------------------------------
    # Determine which atoms exist in ANY grid
    # ------------------------------------------------------
    atoms = set()
    for _, df in data:
        atoms |= set(df["owner"].unique())
    atoms = sorted(list(atoms))

    # ------------------------------------------------------
    # Get base radii R_A^(0) at scale=1.0
    # ------------------------------------------------------
    idx_1 = np.argmin([abs(s - 1.0) for s in scales])
    scale1, df1 = data[idx_1]

    R0 = {}
    for atom in atoms:
        rvals = df1.loc[df1["owner"] == atom, "radius"]
        if len(rvals) == 0:
            raise RuntimeError(f"Atom {atom} missing at scale=1.0")
        # radius at scale = 1.0
        R0[atom] = float(rvals.iloc[0])

    # ------------------------------------------------------
    # Compute radii for all scales: R_A,k = s_k * R_A^(0)
    # ------------------------------------------------------
    R = {atom: np.array([s * R0[atom] for s in scales]) for atom in atoms}

    # ------------------------------------------------------
    # Compute ΔR per atom using midpoint rule
    # ------------------------------------------------------
    dR = {}
    for atom in atoms:
        dRk = np.zeros(m)
        dRk[0] = 0.5 * (R[atom][1] - R[atom][0])
        for k in range(1, m-1):
            dRk[k] = 0.5 * (R[atom][k+1] - R[atom][k-1])
        dRk[m-1] = 0.5 * (R[atom][m-1] - R[atom][m-2])
        dR[atom] = dRk

    # ------------------------------------------------------
    # Compute dV for each grid point: A_i,k * ΔR_{A(i),k}
    # ------------------------------------------------------
    total_A3 = 0.0
    rows = []

    print("\n=== Shell Volume Contributions (owner-wise ΔR) ===\n")

    for k, (scale, df) in enumerate(data):

        A_vals = df["area"].values
        owners = df["owner"].values

        # get ΔR for each grid point according to owner atom
        dR_vals = np.array([dR[int(a)][k] for a in owners])

        dV_A3 = A_vals * dR_vals
        shell_A3 = float(np.sum(dV_A3))
        total_A3 += shell_A3

        print(f"scale {scale:.4f}:  ΔV = {shell_A3:.6f} Å³   "
              f"({shell_A3*ANG3_TO_BOHR3:.6f} bohr³)")

        # store per-grid data
        for i in range(len(df)):
            rows.append({
                "scale": scale,
                "owner": int(owners[i]),
                "x": float(df["x"].iloc[i]),
                "y": float(df["y"].iloc[i]),
                "z": float(df["z"].iloc[i]),
                "area_A2": A_vals[i],
                "radius_A": float(df["radius"].iloc[i]),
                "dR_A": float(dR_vals[i]),
                "dV_A3": float(dV_A3[i]),
                "dV_bohr3": float(dV_A3[i] * ANG3_TO_BOHR3)
            })

    total_bohr3 = total_A3 * ANG3_TO_BOHR3

    print("\n============================================")
    print(f"Total volume = {total_A3:.6f} Å³")
    print(f"Total volume = {total_bohr3:.6f} bohr³")
    print("============================================\n")

    # Write output CSV
    out = pd.DataFrame(rows)
    out.to_csv("dV_all.csv", index=False)
    print("Wrote dV_all.csv\n")


if __name__ == "__main__":
    main()

