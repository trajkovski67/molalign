#!/usr/bin/env python3
"""
all_grid_tb.py

Loop over all grid points of molA, align molB, run xTB SP,
and save all results in a single JSON file.
"""
import os
import sys
import json
import argparse
import numpy as np
import tempfile
import subprocess
import re
import molalign.align_many_to_one_xtb as align_mod

BOHR_TO_ANG = 0.52917721092

# ---------- xTB call ----------
def run_xtb_sp(coords_angstrom, atomic_numbers, charge=0):
    with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as tmp:
        xyz_file = tmp.name
    with open(xyz_file, "w") as fh:
        fh.write(f"{len(atomic_numbers)}\n")
        fh.write("xtb input\n")
        for (x, y, z), Z in zip(coords_angstrom, atomic_numbers):
            fh.write(f"{int(Z)}  {x:.8f}  {y:.8f}  {z:.8f}\n")

    cmd = ["xtb", xyz_file, "--alpb", "water", "--charge", str(charge), "--sp"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = proc.stdout + proc.stderr

    m = re.search(r"total energy\s+(-?\d+\.\d+)", out)
    if not m:
        raise RuntimeError("Cannot parse xTB output:\n" + out)

    return float(m.group(1))


def atoms_to_xyz_list(coords, atomic_numbers):
    return [{"element": str(int(Z)), "x": float(p[0]), "y": float(p[1]), "z": float(p[2])}
            for p, Z in zip(coords, atomic_numbers)]


# ---------- MAIN ----------
def main(fileA, fileB, angles_csv, charge=0, out_json="tb_lite_results.json"):
    fileA = os.path.abspath(fileA)
    fileB = os.path.abspath(fileB)
    angles = [float(a) for a in angles_csv.split(",")]

    dataA = np.load(fileA, allow_pickle=True)
    dataB = np.load(fileB, allow_pickle=True)

    atomsA = dataA["atoms"][:, :3]
    Z_A = dataA["atoms"][:, 3].astype(int)
    atomsB = dataB["atoms"][:, :3]
    Z_B = dataB["atoms"][:, 3].astype(int)

    results = {"molA_grids_xyz": dataA["grids"][:, :3].tolist()}

    # Solute energy
    e_solute = run_xtb_sp(atomsA, Z_A, charge=charge)
    results["solute"] = {
        "energy_Eh": float(e_solute),
        "charge": charge,
        "xyz": atoms_to_xyz_list(atomsA, Z_A)
    }

    # Solvent energy (neutral)
    e_solvent = run_xtb_sp(atomsB, Z_B, charge=0)
    results["solvent"] = {
        "energy_Eh": float(e_solvent),
        "charge": 0,
        "xyz": atoms_to_xyz_list(atomsB, Z_B)
    }

    num_grids = dataA["grids"].shape[0]
    for i in range(num_grids):
        print(f"Processing grid {i}/{num_grids-1}...")
        try:
            gp_results = align_mod.main(fileA, fileB, i, angles, charge=charge)
            results[f"gp{i}"] = gp_results
        except Exception as e:
            print(f"Skipped grid {i} due to error: {e}", file=sys.stderr)

    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        return obj

    with open(out_json, "w") as fh:
        json.dump(results, fh, indent=2, default=convert_numpy)

    print(f"*** Saved all SP results to {out_json}")
    return results


def cli_main():
    parser = argparse.ArgumentParser(description="Generate complexes and run xTB SP for all grids.")
    parser.add_argument("fileA")
    parser.add_argument("fileB")
    parser.add_argument("angles_csv")
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--out", default="tb_lite_results.json")
    args = parser.parse_args()
    main(args.fileA, args.fileB, args.angles_csv, charge=args.charge, out_json=args.out)


if __name__ == "__main__":
    cli_main()

