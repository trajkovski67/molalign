#!/usr/bin/env python3
"""
Wrapper to run molalign TB-lite workflow:
1. Convert solute/solvent CPCM → NPZ.
2. Align grids and run TB-lite single-point calculations with optional charge.
3. Save all results in one JSON file inside output directory.
"""
import argparse
import subprocess
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Align molecules and compute TB-lite SP energies.")
    parser.add_argument("solute_cpcm", help="CPCM file for solute")
    parser.add_argument("solvent_cpcm", help="CPCM file for solvent")
    parser.add_argument("--out", default="OUT", help="Output directory")
    parser.add_argument("--angles", default="0,90,180,270", help="Comma-separated rotation angles")
    parser.add_argument("--charge", type=int, default=0, help="Total charge for the TB-lite calculation")
    args = parser.parse_args()

    solute = os.path.abspath(args.solute_cpcm)
    solvent = os.path.abspath(args.solvent_cpcm)
    out_dir = os.path.abspath(args.out)
    angles = args.angles
    charge = args.charge

    os.makedirs(out_dir, exist_ok=True)

    # Step 1: CPCM → NPZ
    for molfile in [solute, solvent]:
        molname = os.path.splitext(os.path.basename(molfile))[0]
        print(f"\n*** Running: cpcm-reader {molfile} {molname}")
        try:
            subprocess.run(["cpcm-reader", molfile, molname], check=True)
        except subprocess.CalledProcessError as e:
            print(f"ERROR: cpcm-reader failed for {molfile}: {e}", file=sys.stderr)
            sys.exit(1)

        npzfile = f"{molname}_data.npz"
        if not os.path.exists(npzfile):
            print(f"ERROR: Missing expected {npzfile}", file=sys.stderr)
            sys.exit(1)
        else:
            # Move NPZ to output directory
            os.rename(npzfile, os.path.join(out_dir, npzfile))

    # Step 2: Align grids and run TB-lite SP calculations
    solute_npz = os.path.join(out_dir, f"{os.path.splitext(os.path.basename(solute))[0]}_data.npz")
    solvent_npz = os.path.join(out_dir, f"{os.path.splitext(os.path.basename(solvent))[0]}_data.npz")

    print(f"\n*** Running: align-grid-tb {solute_npz} {solvent_npz} {angles} --charge {charge}")
    try:
        # Pass charge to align-grid-tb CLI
        subprocess.run(["align-grid-tb", solute_npz, solvent_npz, angles, "--charge", str(charge)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: align-grid-tb failed: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\n*** All computations complete. Results saved in {out_dir}\n")
    print(f"*** Check tb_lite_results.json inside {out_dir} for energies and geometries.")

if __name__ == "__main__":
    main()

