#!/usr/bin/env python3
"""
run_molalign_shifted_tb.py

Wrapper for molalign + xTB:
- Converts CPCM → NPZ
- Shifts solute grids along atom normals
- Runs xTB SP calculations for all shifts
- Saves results in separate NPZ and JSON files
"""
import argparse
import subprocess
import os
import sys
import numpy as np

def compute_normals(grids, atoms):
    atom_indices = grids[:, 4].astype(int)
    atom_coords = atoms[:, :3]
    grid_coords = grids[:, :3]
    normals = grid_coords - atom_coords[atom_indices]
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    return normals

def shift_grid_points(grids, atoms, shift):
    normals = compute_normals(grids, atoms)
    shifted = grids.copy()
    shifted[:, :3] = grids[:, :3] + normals * shift
    return shifted

def save_shifted_npz(orig_npz, shift, out_file):
    data = np.load(orig_npz, allow_pickle=True)
    atoms = data["atoms"]
    grids = data["grids"]
    shifted = shift_grid_points(grids, atoms, shift)
    np.savez(out_file, atoms=atoms, grids=shifted)
    print(f"*** Saved shifted NPZ ({shift:+} Å): {out_file}")
    return out_file

def sanitize_filename(value):
    s = str(value).replace(".", "p").replace("-", "n")
    return s

def main():
    parser = argparse.ArgumentParser(description="Run molalign xTB with shifted solute grids.")
    parser.add_argument("solute_cpcm")
    parser.add_argument("solvent_cpcm")
    parser.add_argument("--out", default="OUT")
    parser.add_argument("--angles", default="0,90,180,270")
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--shifts", type=float, nargs="+",
        default=[-1.0, 0.0, 1.0])
    args = parser.parse_args()

    solute = os.path.abspath(args.solute_cpcm)
    solvent = os.path.abspath(args.solvent_cpcm)
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    # Step 1: CPCM → NPZ
    for molfile in [solute, solvent]:
        molname = os.path.splitext(os.path.basename(molfile))[0]
        try:
            subprocess.run(["cpcm-reader", molfile, molname], check=True)
        except subprocess.CalledProcessError as e:
            print(f"ERROR: cpcm-reader failed for {molfile}", file=sys.stderr)
            sys.exit(1)

        npzfile = f"{molname}_data.npz"
        os.rename(npzfile, os.path.join(out_dir, npzfile))

    solute_npz = os.path.join(out_dir, f"{os.path.splitext(os.path.basename(solute))[0]}_data.npz")
    solvent_npz = os.path.join(out_dir, f"{os.path.splitext(os.path.basename(solvent))[0]}_data.npz")

    # Step 2: Shift solute grids
    variant_files = []
    for s in args.shifts:
        suffix = f"shift{sanitize_filename(s)}" if s != 0 else "original"
        out_npz = os.path.join(out_dir, f"solute_data_{suffix}.npz")
        save_shifted_npz(solute_npz, s, out_npz)
        out_json = os.path.join(out_dir, f"tb_lite_results_{suffix}.json")
        variant_files.append((out_npz, out_json))

    # Step 3: Run align-grid-tb (now using xTB inside)
    for npz_file, out_json in variant_files:
        try:
            subprocess.run([
                "align-grid-xtb", npz_file, solvent_npz,
                args.angles, "--charge", str(args.charge),
                "--out", out_json
            ], check=True)
        except subprocess.CalledProcessError:
            print(f"ERROR running align-grid-tb for {npz_file}", file=sys.stderr)

    print("\n*** All computations complete.")

if __name__ == "__main__":
    main()

