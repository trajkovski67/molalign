#!/usr/bin/env python3
import subprocess
import os
import argparse
import sys


def run_command(cmd, cwd=None):
    """Run a shell command and stop if it fails."""
    print(f"\n*** Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, cwd=cwd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"COMMAND FAILED: {' '.join(cmd)}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Run the full molalign pipeline: read → align → scan → extract."
    )
    parser.add_argument(
        "molA_cpcm",
        help="First .cpcm file (molecule A, defines grid points)."
    )
    parser.add_argument(
        "molB_cpcm",
        help="Second .cpcm file (molecule B, to align)."
    )
    parser.add_argument(
        "--angles",
        help="Comma-separated rotation angles (default: 0,90,180,270).",
        default="0,90,180,270"
    )
    parser.add_argument(
        "--charge",
        type=int,
        default=0,
        help="Charge for xTB calculations (default: 0)."
    )

    args = parser.parse_args()

    prefixA = os.path.splitext(os.path.basename(args.molA_cpcm))[0]
    prefixB = os.path.splitext(os.path.basename(args.molB_cpcm))[0]

    # Step 1: Convert CPCM to NPZ
    run_command(["cpcm-reader", args.molA_cpcm, prefixA])
    run_command(["cpcm-reader", args.molB_cpcm, prefixB])

    # Step 2: Generate complexes (all-grid)
    run_command([
        "align-grid",
        f"{prefixA}_data.npz",
        f"{prefixB}_data.npz",
        args.angles
    ])

    # Step 3: Run xTB SP scans
    complexes_dir = f"{prefixA}_data_vs_{prefixB}_data_complexes"
    run_command(["xtb-scan", complexes_dir, "--chrg", str(args.charge)])

    # Step 4: Extract results
    run_command(["xtb-extract", complexes_dir])

    print("\n*** All pipeline steps completed successfully!")


if __name__ == "__main__":
    main()

