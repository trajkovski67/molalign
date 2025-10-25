#!/usr/bin/env python3
"""
molalign-extended-orca  (OPI version; no files kept)

Workflow:
  1. Convert CPCM → NPZ using an external `cpcm-reader`
  2. Shift solute grids along atom normals (for given shifts)
  3. Run all-grid ORCA SPs via the `all_grid_orca.py` wrapper (OPI backend)
  4. Save ONLY final JSON results
"""

import argparse
import subprocess
import os
import sys
import numpy as np

# ---------- Utility: compute normals and shifts ----------
def compute_normals(grids, atoms):
    atom_indices = grids[:, 4].astype(int)
    atom_coords = atoms[:, :3]
    grid_coords = grids[:, :3]
    normals = grid_coords - atom_coords[atom_indices]
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
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
    shifted_grids = shift_grid_points(grids, atoms, shift)
    np.savez(out_file, atoms=atoms, grids=shifted_grids)
    print(f"*** Saved shifted NPZ ({shift:+} Å): {out_file}")
    return out_file

def sanitize_filename(value):
    return str(value).replace(".", "p").replace("-", "n")

# ---------- Main driver ----------
def main():
    parser = argparse.ArgumentParser(description="Run molalign ORCA SPs via OPI (no files kept).")
    parser.add_argument("solute_cpcm", help="CPCM file for solute")
    parser.add_argument("solvent_cpcm", help="CPCM file for solvent")
    parser.add_argument("--out", default="OUT", help="Output directory")
    parser.add_argument("--angles", default="0,90,180,270", help="Comma-separated rotation angles (deg)")
    parser.add_argument("--charge", type=int, default=0, help="Total charge for solute/complex")
    parser.add_argument("--mult", type=int, default=1, help="Multiplicity for solute/complex")
    parser.add_argument("--solvent-mult", type=int, default=1, help="Multiplicity for solvent")
    parser.add_argument("--shifts", type=float, nargs="+", default=[-1.0, 0.0, 1.0], help="Grid shifts (Å)")
    parser.add_argument("--ncores", type=int, default=8, help="Cores per ORCA SP (OPI)")
    parser.add_argument("--method", default="r2scan-3c", help="DFT method (e.g. r2scan-3c, pbe, b3lyp)")
    parser.add_argument("--align-grid-orca", default="align-grid-orca", help="CLI or path to all_grid_orca.py")
    parser.add_argument("--cpcm-reader", default="cpcm-reader", help="CLI tool to convert CPCM → *_data.npz")
    args = parser.parse_args()

    solute = os.path.abspath(args.solute_cpcm)
    solvent = os.path.abspath(args.solvent_cpcm)
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    # ---------- Step 1: Convert CPCM → NPZ ----------
    for molfile in [solute, solvent]:
        molname = os.path.splitext(os.path.basename(molfile))[0]
        print(f"\n*** Running: {args.cpcm_reader} {molfile} {molname}")
        subprocess.run([args.cpcm_reader, molfile, molname], check=True)
        npzfile = f"{molname}_data.npz"
        if not os.path.exists(npzfile):
            sys.exit(f"ERROR: Missing expected {npzfile}")
        os.replace(npzfile, os.path.join(out_dir, npzfile))

    solute_npz = os.path.join(out_dir, f"{os.path.splitext(os.path.basename(solute))[0]}_data.npz")
    solvent_npz = os.path.join(out_dir, f"{os.path.splitext(os.path.basename(solvent))[0]}_data.npz")

    # ---------- Step 2: Generate shifted solute grids ----------
    variant_files = []
    for s in args.shifts:
        suffix = f"shift{sanitize_filename(s)}" if s != 0 else "original"
        out_npz = os.path.join(out_dir, f"solute_data_{suffix}.npz")
        save_shifted_npz(solute_npz, s, out_npz)
        out_json = os.path.join(out_dir, f"orca_results_{suffix}.json")
        variant_files.append((out_npz, out_json))

    # ---------- Step 3: Run all-grid ORCA SPs ----------
    for npz_file, out_json in variant_files:
        cmd = [
            args.align_grid_orca, npz_file, solvent_npz, args.angles,
            "--charge", str(args.charge),
            "--mult", str(args.mult),
            "--solvent-mult", str(args.solvent_mult),
            "--out", out_json,
            "--ncores", str(args.ncores),
            "--method", args.method,
        ]
        print(f"\n*** Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    print("\n*** All computations complete. Results saved in:")
    for _, out_json in variant_files:
        print(f"    {out_json}")

if __name__ == "__main__":
    main()

