#!/usr/bin/env python3
"""
all_grid_orca.py  (OPI version; no files kept)

Loop over all grid points of molA, align molB, run ORCA SPs via OPI, and save JSON.
"""

import os
import sys
import json
import argparse
import numpy as np
import molalign.align_many_to_one_orca as align_mod

def atoms_to_xyz_list(coords, atomic_numbers):
    return [
        {"element": str(int(Z)), "x": float(p[0]), "y": float(p[1]), "z": float(p[2])}
        for p, Z in zip(coords, atomic_numbers)
    ]

def run_orca_sp_wrapper(coords_ang, atomic_numbers, charge=0, mult=1, ncores=8, method="r2scan-3c"):
    return align_mod.run_orca_sp_opi(
        coords_ang, atomic_numbers, charge=charge, mult=mult, ncores=ncores, method=method
    )

def main(fileA, fileB, angles_csv, charge=0, mult=1, solvent_mult=1,
         out_json="orca_results.json", ncores=8, method="r2scan-3c"):

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

    # Solute SP
    try:
        e_solute = run_orca_sp_wrapper(atomsA, Z_A, charge=charge, mult=mult, ncores=ncores, method=method)
        results["solute"] = {"energy_Eh": float(e_solute), "charge": int(charge), "mult": int(mult)}
    except Exception as e:
        results["solute"] = {"error": str(e), "charge": int(charge), "mult": int(mult)}

    # Solvent SP (neutral)
    try:
        e_solvent = run_orca_sp_wrapper(atomsB, Z_B, charge=0, mult=solvent_mult, ncores=ncores, method=method)
        results["solvent"] = {"energy_Eh": float(e_solvent), "charge": 0, "mult": int(solvent_mult)}
    except Exception as e:
        results["solvent"] = {"error": str(e), "charge": 0, "mult": int(solvent_mult)}

    num_grids = dataA["grids"].shape[0]
    for i in range(num_grids):
        print(f"Processing grid {i}/{num_grids-1}...")
        try:
            gp_results = align_mod.main(
                fileA, fileB, i, angles, charge=charge, mult=mult, ncores=ncores, method=method
            )
            results[f"gp{i}"] = gp_results
        except Exception as e:
            print(f"Skipped grid {i}: {e}", file=sys.stderr)

    with open(out_json, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"*** Saved ORCA SP results to {out_json}")
    return results

def cli_main():
    parser = argparse.ArgumentParser(description="Run ORCA SP for all grids via OPI (no files kept).")
    parser.add_argument("fileA")
    parser.add_argument("fileB")
    parser.add_argument("angles_csv")
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--mult", type=int, default=1)
    parser.add_argument("--solvent-mult", type=int, default=1)
    parser.add_argument("--out", default="orca_results.json")
    parser.add_argument("--ncores", type=int, default=8, help="Cores per ORCA SP (OPI)")
    parser.add_argument("--method", default="r2scan-3c", help="DFT method key (e.g. r2scan-3c, pbe0, b3lyp)")
    args = parser.parse_args()
    main(args.fileA, args.fileB, args.angles_csv,
         charge=args.charge, mult=args.mult, solvent_mult=args.solvent_mult,
         out_json=args.out, ncores=args.ncores, method=args.method)

if __name__ == "__main__":
    cli_main()

