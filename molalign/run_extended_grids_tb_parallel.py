#!/usr/bin/env python3
"""
run_extended_grids_tb_parallel2.py

Wrapper for molalign + TB-lite with shifted solute grids.
Now with:
- tqdm progress bar over shifts
- cleaner logging
"""

import argparse
import subprocess
import os
import sys
import numpy as np
from tqdm import tqdm


# ---------- Grid shift utilities ----------

def compute_normals(grids, atoms):
    atom_indices = grids[:, 4].astype(int)
    atom_coords = atoms[:, :3]
    grid_coords = grids[:, :3]
    normals = grid_coords - atom_coords[atom_indices]
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.where(norms > 0, norms, 1.0)
    return normals


def shift_grid_points(grids, atoms, shift):
    normals = compute_normals(grids, atoms)
    shifted = grids.copy()
    shifted[:, :3] = grids[:, :3] + normals * shift
    return shifted


def save_shifted_npz(orig_npz, shift, out_file, atoms=None, grids=None, normals=None):
    """
    Save a shifted NPZ file.

    If atoms/grids/normals are provided, uses those directly (no reload / recompute).
    Otherwise, falls back to loading orig_npz and computing normals.
    """
    if atoms is None or grids is None:
        data = np.load(orig_npz, allow_pickle=True)
        atoms = data["atoms"]
        grids = data["grids"]

    if normals is None:
        normals = compute_normals(grids, atoms)

    shifted_grids = grids.copy()
    shifted_grids[:, :3] = grids[:, :3] + normals * shift

    np.savez(out_file, atoms=atoms, grids=shifted_grids)
    print(f"    → Saved shifted NPZ ({shift:+.2f} Å): {out_file}")
    return out_file


def sanitize_filename(value):
    """Convert float shift to a safe string for filenames."""
    s = str(value).replace(".", "p").replace("-", "n")
    return s


# ---------- Main workflow ----------

def main():
    parser = argparse.ArgumentParser(description="Run molalign TB-lite with shifted solute grids.")
    parser.add_argument("solute_cpcm", help="CPCM file for solute")
    parser.add_argument("solvent_cpcm", help="CPCM file for solvent")
    parser.add_argument("--out", default="OUT", help="Output directory")
    parser.add_argument("--angles", default="0,90,180,270", help="Comma-separated rotation angles")
    parser.add_argument("--charge", type=int, default=0, help="Total charge for TB-lite calculation")
    parser.add_argument(
        "--shifts", type=float, nargs="+",
        default=[-1.0, 0.0, 1.0],
        help="Shifts along normals in Å (space-separated, e.g. -1 0 1)"
    )
    args = parser.parse_args()

    solute = os.path.abspath(args.solute_cpcm)
    solvent = os.path.abspath(args.solvent_cpcm)
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    angles = args.angles
    charge = args.charge
    shifts = args.shifts

    print("\n========== MOLALIGN + TBLITE (SHIFTED GRIDS) ==========")
    print(f"Solute CPCM : {solute}")
    print(f"Solvent CPCM: {solvent}")
    print(f"Output dir  : {out_dir}")
    print(f"Angles      : {angles}")
    print(f"Charge      : {charge}")
    print(f"Shifts (Å)  : {shifts}")
    print("-------------------------------------------------------")

    # ---------- Step 1: CPCM → NPZ ----------
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
        os.rename(npzfile, os.path.join(out_dir, npzfile))

    solute_npz = os.path.join(out_dir, f"{os.path.splitext(os.path.basename(solute))[0]}_data.npz")
    solvent_npz = os.path.join(out_dir, f"{os.path.splitext(os.path.basename(solvent))[0]}_data.npz")

    # ---------- Step 2: Generate shifted solute grids (preload once) ----------
    print("\nLoading solute NPZ and precomputing normals…")
    solute_data = np.load(solute_npz, allow_pickle=True)
    atoms_solute = solute_data["atoms"]
    grids_solute = solute_data["grids"]
    normals_solute = compute_normals(grids_solute, atoms_solute)

    variant_files = []
    print("\nGenerating shifted solute NPZ variants:")
    for s in shifts:
        suffix = f"shift{sanitize_filename(s)}" if s != 0 else "original"
        out_npz = os.path.join(out_dir, f"solute_data_{suffix}.npz")
        save_shifted_npz(
            solute_npz,
            s,
            out_npz,
            atoms=atoms_solute,
            grids=grids_solute,
            normals=normals_solute,
        )
        out_json = os.path.join(out_dir, f"tb_lite_results_{suffix}.json")
        variant_files.append((s, out_npz, out_json))

    # ---------- Step 3: Run align-grid-tb for each shift ----------
    print("\nRunning align-grid-tb-parallel-2 for all shifts:\n")
    for s, npz_file, out_json in tqdm(variant_files, desc="Shifts", unit="shift"):
        print(f"\n*** Shift {s:+.2f} Å")
        print(f"    NPZ : {npz_file}")
        print(f"    OUT : {out_json}")
        try:
            subprocess.run([
                "align-grid-tb-parallel-2", npz_file, solvent_npz,
                angles, "--charge", str(charge),
                "--out", out_json
            ], check=True)
        except subprocess.CalledProcessError as e:
            print(f"ERROR: align-grid-tb failed for {npz_file}: {e}", file=sys.stderr)

    print("\n*** All computations complete. Results saved in:")
    for _, _, out_json in variant_files:
        print(f"    {out_json}")


if __name__ == "__main__":
    main()

