#!/usr/bin/env python3
import os
import sys
import numpy as np
from . import allign_many_to_one as align
import argparse

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

    num_grids = np.load(fileA)["grids"].shape[0]

    for i in range(num_grids):
        print(f"\n=== Aligning grid point {i}/{num_grids-1} ===")
        try:
            grid_folder = os.path.join(base_prefix, f"gp{i}")
            os.makedirs(grid_folder, exist_ok=True)

            cwd = os.getcwd()
            os.chdir(grid_folder)

            align.main(fileA, fileB, i, angles, f"gp{i}")

            os.chdir(cwd)

        except Exception as e:
            print(f"⚠️ Skipped grid {i} due to error: {e}")

if __name__ == "__main__":
    main()

