#!/usr/bin/env python3
import os
import subprocess
import glob

# Folder containing all XYZ files (or adjust as needed)
xyz_root = "water_data_vs_proton_data_complexes"

# xTB command template (single-point, GFN2-xTB, output to .out)
xtb_cmd = "xtb {xyz} --chrg 1 > {out}"

# Find all xyz files recursively
xyz_files = glob.glob(os.path.join(xyz_root, "**", "*.xyz"), recursive=True)

print(f"Found {len(xyz_files)} XYZ files. Running xTB SP calculations...")

for xyz in xyz_files:
    out_file = os.path.splitext(xyz)[0] + ".out"
    print(f"Running xTB for {xyz} → {out_file}")
    try:
        subprocess.run(f"xtb {xyz} --sp > {out_file}", shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"⚠️ xTB failed for {xyz}: {e}")

print("✅ All calculations attempted.")
