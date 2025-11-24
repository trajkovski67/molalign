#!/usr/bin/env python3
"""
all_grid_tb.py

Loop over all grid points of molA, align molB, run TB-lite SP,
and save all results in a single JSON file.
"""
import os
import sys
import json
import argparse
import numpy as np
import molalign.align_many_to_one_tb as align_mod

BOHR_TO_ANG = 0.52917721092

def atoms_to_xyz_list(coords, atomic_numbers):
    return [{"element": str(int(Z)), "x": float(p[0]), "y": float(p[1]), "z": float(p[2])}
            for p, Z in zip(coords, atomic_numbers)]

# --- MODIFIED: added charge parameter ---
def run_tblite_sp(coords_angstrom, atomic_numbers, method="GFN2-xTB", charge=0):
    """
    Run a single-point calculation using tblite.

    Parameters:
        coords_angstrom (array-like): Atomic coordinates in Angstrom.
        atomic_numbers (array-like): Atomic numbers of atoms.
        method (str): The xTB method to use (default "GFN2-xTB").
        charge (int): Total charge of the molecule.

    Returns:
        float: Energy in Hartree.
    """
    import numpy as np
    from tblite.interface import Calculator
    import contextlib, io

    # Convert coordinates to Bohr
    coords_bohr = np.array(coords_angstrom) / BOHR_TO_ANG

    # Initialize calculator with charge (correct way)
    calc = Calculator(method, np.array(atomic_numbers), coords_bohr, charge=charge)
    calc.add("alpb-solvation", 78.4)
    # Capture stdout to avoid printing
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        res = calc.singlepoint()

    # Return energy
    return res.get("energy")

def main(fileA, fileB, angles_csv, charge=0, out_json="tb_lite_results.json"):
    import os
    import numpy as np
    import molalign.align_many_to_one_tb as align_mod

    fileA = os.path.abspath(fileA)
    fileB = os.path.abspath(fileB)
    angles = [float(a) for a in angles_csv.split(",")]

    dataA = np.load(fileA, allow_pickle=True)
    dataB = np.load(fileB, allow_pickle=True)

    atomsA = dataA["atoms"][:, :3]
    Z_A = dataA["atoms"][:, 3].astype(int)
    atomsB = dataB["atoms"][:, :3]
    Z_B = dataB["atoms"][:, 3].astype(int)

    # --- NEW: Include molA grid points in the results JSON ---
    results = {
        "molA_grids_xyz": dataA["grids"][:, :3].tolist()
    }

    # --- MODIFIED: solute uses user-specified charge ---
    e_solute = run_tblite_sp(atomsA, Z_A, charge=charge)
    results["solute"] = {
        "energy_Eh": float(e_solute),
        "charge": charge,
        "xyz": atoms_to_xyz_list(atomsA, Z_A)
    }

    # --- MODIFIED: solvent always neutral (charge=0) ---
    e_solvent = run_tblite_sp(atomsB, Z_B, charge=0)
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

    # Helper to convert numpy types to Python-native types
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        return obj

    # Save results to JSON safely
    with open(out_json, "w") as fh:
        import json
        json.dump(results, fh, indent=2, default=convert_numpy)

    print(f"*** Saved all SP results to {out_json}")
    return results

def cli_main():
    parser = argparse.ArgumentParser(description="Generate complexes and run tblite SP for all grids.")
    parser.add_argument("fileA", help="NPZ file for solute")
    parser.add_argument("fileB", help="NPZ file for solvent")
    parser.add_argument("angles_csv", help="Comma-separated rotation angles")
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--out", default="tb_lite_results.json")
    args = parser.parse_args()
    main(args.fileA, args.fileB, args.angles_csv, charge=args.charge, out_json=args.out)

if __name__ == "__main__":
    cli_main()

