#!/usr/bin/env python3
import os
import sys
import numpy as np
from . import allign_many_to_one as align
import argparse

def to_xyz(filename, coords, atomic_numbers):
    with open(filename,'w') as f:
        f.write(f"{len(coords)}\n{filename}\n")
        for p,Z in zip(coords, atomic_numbers):
            f.write(f"{Z} {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n")

def save_mol_xyz(npz_file, folder):
    data = np.load(npz_file, allow_pickle=True)   # <-- add allow_pickle=True
    atoms = data["atoms"]       # x, y, z, Z, index
    coords = atoms[:, :3]
    Z = atoms[:, 3].astype(int)
    os.makedirs(folder, exist_ok=True)
    xyz_path = os.path.join(folder, os.path.basename(npz_file).replace(".npz", ".xyz"))
    to_xyz(xyz_path, coords, Z)


def main():
    parser = argparse.ArgumentParser(description="Generate complexes for all grid points of molA.")
    parser.add_argument("fileA", help="NPZ file for molecule A")
    parser.add_argument("fileB", help="NPZ file for molecule B")
    parser.add_argument("angles_csv", help="Comma-separated rotation angles, e.g., 0,90,180,270")
    args = parser.parse_args()

    fileA = os.path.abspath(args.fileA)
    fileB = os.path.abspath(args.fileB)
    angles = [float(a) for a in args.angles_csv.split(",")]

    # Base folder to store results
    base_prefix = f"{os.path.splitext(os.path.basename(fileA))[0]}_vs_{os.path.splitext(os.path.basename(fileB))[0]}_complexes"
    os.makedirs(base_prefix, exist_ok=True)

    # --- Save mol1 and mol2 XYZ ---
    solute_folder = os.path.join(base_prefix, "solute")
    solvent_folder = os.path.join(base_prefix, "solvent")
    save_mol_xyz(fileA, solute_folder)
    save_mol_xyz(fileB, solvent_folder)
    # --------------------------------

    num_grids = np.load(fileA)["grids"].shape[0]

    for i in range(num_grids):
        #print(f"\n=== Aligning grid point {i}/{num_grids-1} ===")
        try:
            grid_folder = os.path.join(base_prefix, f"gp{i}")
            os.makedirs(grid_folder, exist_ok=True)

            cwd = os.getcwd()
            os.chdir(grid_folder)

            align.main(fileA, fileB, i, angles, f"gp{i}")

            os.chdir(cwd)

        except Exception as e:
            print(f"  Skipped grid {i} due to error: {e}")

if __name__ == "__main__":
    main()

