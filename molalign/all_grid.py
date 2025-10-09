#!/usr/bin/env python3
import os
import sys
import numpy as np
import allign_many_to_one as align  # import your align script

if len(sys.argv) < 3:
    print("Usage: ./all_grid molA.npz molB.npz [angles_csv]")
    print("Example: ./all_grid helium_data.npz water_data.npz 0,90,180,270")
    sys.exit(1)

# Get absolute paths
fileA = os.path.abspath(sys.argv[1])
fileB = os.path.abspath(sys.argv[2])

# Parse rotation angles
angles_csv = sys.argv[3] if len(sys.argv) >= 4 else "0"
angles = [float(a) for a in angles_csv.split(",")]

# Base folder to store all results
base_prefix = f"{os.path.splitext(os.path.basename(fileA))[0]}_vs_{os.path.splitext(os.path.basename(fileB))[0]}_complexes"
os.makedirs(base_prefix, exist_ok=True)

# Number of grid points in molA
num_grids = np.load(fileA)["grids"].shape[0]

for i in range(num_grids):
    print(f"\n=== Aligning grid point {i}/{num_grids-1} ===")
    try:
        # Create folder for this grid point
        grid_folder = os.path.join(base_prefix, f"gp{i}")
        os.makedirs(grid_folder, exist_ok=True)

        # Change working directory to save files here
        cwd = os.getcwd()
        os.chdir(grid_folder)

        # Call your align script with rotation angles
        align.main(fileA, fileB, i, angles, f"gp{i}")

        # Return to original directory
        os.chdir(cwd)

    except Exception as e:
        print(f"⚠️ Skipped grid {i} due to error: {e}")

