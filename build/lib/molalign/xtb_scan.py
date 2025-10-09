#!/usr/bin/env python3
import os
import subprocess
import glob
import argparse
import sys


def run_xtb_scans(xyz_root: str, charge: int = 0):
    """Run xTB single-point calculations on all .xyz files in a folder tree."""
    if not os.path.isdir(xyz_root):
        print(f"❌ Error: '{xyz_root}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    # Find all xyz files recursively
    xyz_files = glob.glob(os.path.join(xyz_root, "**", "*.xyz"), recursive=True)

    if not xyz_files:
        print(f"⚠️ No .xyz files found in {xyz_root}")
        return

    print(f"Found {len(xyz_files)} XYZ files in '{xyz_root}'. Running xTB SP calculations...")

    for xyz in xyz_files:
        out_file = os.path.splitext(xyz)[0] + ".out"
        print(f"▶ Running xTB for {xyz} → {out_file}")
        try:
            subprocess.run(
                ["xtb", xyz, "--sp", f"--chrg", str(charge)],
                stdout=open(out_file, "w"),
                stderr=subprocess.STDOUT,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"  ⚠️ xTB failed for {xyz}: {e}")

    print("✅ All calculations attempted.")


def main():
    parser = argparse.ArgumentParser(
        description="Run xTB single-point calculations for all .xyz files in a folder tree."
    )
    parser.add_argument(
        "xyz_root",
        help="Root directory containing .xyz files (searched recursively).",
    )
    parser.add_argument(
        "--chrg",
        type=int,
        default=0,
        help="Molecular charge for xTB (default: 0).",
    )

    args = parser.parse_args()
    run_xtb_scans(args.xyz_root, args.chrg)


if __name__ == "__main__":
    main()

