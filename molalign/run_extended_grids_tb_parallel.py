#!/usr/bin/env python3
"""
run_extended_grids_tb_sequential.py

- Fully sequential execution
- Shifts → sequential
- Grids → sequential
- Angles → sequential
- Single global progress bar
- Keeps maximum speed without oversubscription
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

# ---------- Worker: single calculation ----------
def run_single_calc(grid_idx, angle, solute_npz, solvent_npz, charge):
    import molalign.align_many_to_one_tb as align_mod
    # Run calculation for a single grid and angle
    return align_mod.main(solute_npz, solvent_npz, grid_idx, [angle], charge=charge)

# ---------- Main workflow ----------
def main():
    parser = argparse.ArgumentParser(description="Fully sequential molalign TB-lite runner")
    parser.add_argument("solute_cpcm", help="CPCM file for solute")
    parser.add_argument("solvent_cpcm", help="CPCM file for solvent")
    parser.add_argument("--out", default="OUT", help="Output directory")
    parser.add_argument("--angles", default="0,90,180,270", help="Comma-separated rotation angles")
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--shifts", type=float, nargs="+", default=[0.0])
    args = parser.parse_args()

    solute = os.path.abspath(args.solute_cpcm)
    solvent = os.path.abspath(args.solvent_cpcm)
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    angles = [float(a) for a in args.angles.split(",")]
    charge = args.charge
    shifts = args.shifts

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

    # ---------- Step 2: Sequential shifts ----------
    for shift in shifts:
        print(f"\n*** Processing shift {shift:+} Å")
        suffix = f"shift{sanitize_filename(shift)}" if shift != 0 else "original"
        shifted_npz = os.path.join(out_dir, f"solute_data_{suffix}.npz")
        save_shifted_npz(solute_npz, shift, shifted_npz)

        data = np.load(shifted_npz, allow_pickle=True)
        num_grids = data["grids"].shape[0]

        # ---------- Step 3: Sequential grids and angles ----------
        total_calcs = num_grids * len(angles)
        global_progress = tqdm(total=total_calcs, desc=f"Shift {shift:+} Å grids")
        for grid_idx in range(num_grids):
            for angle in angles:
                run_single_calc(grid_idx, angle, shifted_npz, solvent_npz, charge)
                global_progress.update(1)
        global_progress.close()
        print(f"*** Completed shift {shift:+} Å")

if __name__ == "__main__":
    main()

