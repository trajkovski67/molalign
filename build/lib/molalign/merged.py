#!/usr/bin/env python3
"""
Merged molalign + TB-lite workflow with parallelization.
- align_many_to_one_tb.py
- all_grid_tb.py
- run_extended_grids_tb.py

All original logic preserved.
Added:
    • Parallelization limited to 6 CPU cores
    • tqdm progress bars for grid points & alignment jobs
"""

import os
import sys
import json
import argparse
import subprocess
import numpy as np
import contextlib, io
from multiprocessing import Pool, cpu_count
from tblite.interface import Calculator
from tqdm import tqdm      # <-- NEW

BOHR_TO_ANG = 0.52917721092
NCORES = min(6, cpu_count())   # <-- NEW: Limit cores

# ============================================================
# Shared utility
# ============================================================

def atoms_to_xyz_list(coords, atomic_numbers):
    return [{"element": str(int(Z)), "x": float(p[0]), "y": float(p[1]), "z": float(p[2])}
            for p, Z in zip(coords, atomic_numbers)]


# ============================================================
# TB-lite wrapper
# ============================================================

def run_tblite_sp(coords_angstrom, atomic_numbers, method="GFN2-xTB", charge=0):
    coords_bohr = np.array(coords_angstrom) / BOHR_TO_ANG
    calc = Calculator(method, np.array(atomic_numbers), coords_bohr, charge=charge)
    calc.add("alpb-solvation", "water")
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        res = calc.singlepoint()
    return res.get("energy")


# ============================================================
# Alignment helpers
# ============================================================

def rotation_matrix_from_vectors(a, b, eps=1e-12):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < eps or nb < eps:
        return np.eye(3)
    a /= na
    b /= nb
    dot = np.dot(a, b)
    if dot > 1 - eps:
        return np.eye(3)
    if dot < -1 + eps:
        arbitrary = np.array([1, 0, 0])
        if abs(a[0]) > 0.9:
            arbitrary = np.array([0, 1, 0])
        axis = np.cross(a, arbitrary)
        axis /= np.linalg.norm(axis)
        return -np.eye(3) + 2 * np.outer(axis, axis)
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K.dot(K) * ((1 - dot) / (s**2))


def rotate_around_axis(coords, axis, angle_deg):
    theta = np.radians(angle_deg)
    k = axis / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    return coords @ R.T


# ============================================================
# PARALLEL worker for align_many_to_one
# ============================================================

def _worker_align(params):
    (idxB, pA, nA, atomsA, Z_A, atomsB, Z_B, gridsB, normalsB, angles, charge) = params

    results = {}

    pB = gridsB[idxB]
    nB = normalsB[idxB]
    target = -nA

    R_align = rotation_matrix_from_vectors(nB, target)
    atomsB_rot = atomsB @ R_align.T
    gridsB_rot = gridsB @ R_align.T

    translation = pA - gridsB_rot[idxB]
    atomsB_rot += translation
    gridsB_rot += translation

    for angle in angles:
        axis = target
        atomsB_final = rotate_around_axis(atomsB_rot - pA, axis, angle) + pA

        atoms_combined = np.vstack((atomsA, atomsB_final))
        Z_combined = np.hstack((Z_A, Z_B))

        key = f"b{idxB}_rot{int(angle)}"
        energy = run_tblite_sp(atoms_combined, Z_combined, charge=charge)

        results[key] = {
            "energy_Eh": float(energy),
            "xyz": atoms_to_xyz_list(atoms_combined, Z_combined)
        }

    return (idxB, results)


# ============================================================
# align_many_to_one_tb → PARALLEL + PROGRESS BAR
# ============================================================

def align_many_to_one_main(fileA, fileB, idxA, angles=[0], charge=0):
    dataA = np.load(fileA, allow_pickle=True)
    dataB = np.load(fileB, allow_pickle=True)

    atomsA = dataA["atoms"][:, :3]
    Z_A = dataA["atoms"][:, 3].astype(int)
    gridsA = dataA["grids"][:, :3]
    normalsA = gridsA - atomsA[dataA["grids"][:, 4].astype(int)]
    normalsA /= np.linalg.norm(normalsA, axis=1, keepdims=True)

    atomsB = dataB["atoms"][:, :3]
    Z_B = dataB["atoms"][:, 3].astype(int)
    gridsB = dataB["grids"][:, :3]
    normalsB = gridsB - atomsB[dataB["grids"][:, 4].astype(int)]
    normalsB /= np.linalg.norm(normalsB, axis=1, keepdims=True)

    pA = gridsA[idxA]
    nA = normalsA[idxA]

    # Prepare job list
    params_list = [
        (idxB, pA, nA, atomsA, Z_A, atomsB, Z_B, gridsB, normalsB, angles, charge)
        for idxB in range(len(gridsB))
    ]

    # Parallel with progress bar
    with Pool(NCORES) as pool:
        out = list(
            tqdm(pool.imap(_worker_align, params_list),
                 total=len(params_list),
                 desc=f"Aligning grid {idxA}")
        )

    # Merge results
    all_complexes = {}
    for idxB, r in out:
        all_complexes.update(r)

    return all_complexes



# ============================================================
# all_grid_tb.py → PARALLEL + PROGRESS BAR
# ============================================================

def _worker_grid(params):
    (fileA, fileB, idx, angles, charge) = params
    try:
        return (idx, align_many_to_one_main(fileA, fileB, idx, angles, charge))
    except Exception as e:
        return (idx, {"ERROR": str(e)})


def all_grid_tb_main(fileA, fileB, angles_csv, charge=0, out_json="tb_lite_results.json"):
    fileA = os.path.abspath(fileA)
    fileB = os.path.abspath(fileB)
    angles = [float(a) for a in angles_csv.split(",")]

    dataA = np.load(fileA, allow_pickle=True)
    dataB = np.load(fileB, allow_pickle=True)

    atomsA = dataA["atoms"][:, :3]
    Z_A = dataA["atoms"][:, 3].astype(int)
    atomsB = dataB["atoms"][:, :3]
    Z_B = dataB["atoms"][:, 3].astype(int)

    results = {"molA_grids_xyz": dataA["grids"][:, :3].tolist()}

    e_solute = run_tblite_sp(atomsA, Z_A, charge=charge)
    results["solute"] = {"energy_Eh": float(e_solute), "charge": charge,
                         "xyz": atoms_to_xyz_list(atomsA, Z_A)}

    e_solvent = run_tblite_sp(atomsB, Z_B, charge=0)
    results["solvent"] = {"energy_Eh": float(e_solvent), "charge": 0,
                          "xyz": atoms_to_xyz_list(atomsB, Z_B)}

    num_grids = dataA["grids"].shape[0]

    params_list = [(fileA, fileB, i, angles, charge) for i in range(num_grids)]

    # Parallel with progress bar
    with Pool(NCORES) as pool:
        out = list(
            tqdm(pool.imap(_worker_grid, params_list),
                 total=num_grids,
                 desc="Processing all grid points")
        )

    for idx, r in out:
        results[f"gp{idx}"] = r

    # Save
    with open(out_json, "w") as fh:
        json.dump(results, fh, indent=2)

    print(f"*** Saved all SP results to {out_json}")
    return results



# ============================================================
# run_extended_grids_tb.py (unchanged)
# ============================================================

def compute_normals(grids, atoms):
    atom_indices = grids[:, 4].astype(int)
    normals = grids[:, :3] - atoms[atom_indices, :3]
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
    shifted_grids = shift_grid_points(grids, atoms, shift)
    np.savez(out_file, atoms=atoms, grids=shifted_grids)
    print(f"*** Saved shifted NPZ ({shift:+} Å): {out_file}")
    return out_file

def sanitize_filename(value):
    return str(value).replace(".", "p").replace("-", "n")


# ============================================================
# MAIN CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Merged molalign TB-lite workflow with parallel SP evaluations.")
    parser.add_argument("solute_cpcm")
    parser.add_argument("solvent_cpcm")
    parser.add_argument("--out", default="OUT")
    parser.add_argument("--angles", default="0,90,180,270")
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--shifts", type=float, nargs="+", default=[-1.0, 0.0, 1.0])
    args = parser.parse_args()

    solute = os.path.abspath(args.solute_cpcm)
    solvent = os.path.abspath(args.solvent_cpcm)
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    angles = args.angles
    charge = args.charge
    shifts = args.shifts

    # Step 1: CPCM → NPZ
    for molfile in [solute, solvent]:
        molname = os.path.splitext(os.path.basename(molfile))[0]
        print(f"\n*** Running: cpcm-reader {molfile} {molname}")
        subprocess.run(["cpcm-reader", molfile, molname], check=True)
        npzfile = f"{molname}_data.npz"
        os.rename(npzfile, os.path.join(out_dir, npzfile))

    solute_npz = os.path.join(out_dir, f"{os.path.splitext(os.path.basename(solute))[0]}_data.npz")
    solvent_npz = os.path.join(out_dir, f"{os.path.splitext(os.path.basename(solvent))[0]}_data.npz")

    # Step 2: shifted grids
    variant_files = []
    for s in shifts:
        suffix = f"shift{sanitize_filename(s)}" if s != 0 else "original"
        out_npz = os.path.join(out_dir, f"solute_data_{suffix}.npz")
        save_shifted_npz(solute_npz, s, out_npz)
        out_json = os.path.join(out_dir, f"tb_lite_results_{suffix}.json")
        variant_files.append((out_npz, out_json))

    # Step 3: run full grid alignment TB-lite
    for npz_file, out_json in variant_files:
        print(f"\n*** Running parallel align-grid-tb: {npz_file} {solvent_npz}")
        all_grid_tb_main(npz_file, solvent_npz, angles, charge=charge, out_json=out_json)

    print("\n*** All computations complete. Results saved:")
    for _, out_json in variant_files:
        print("   ", out_json)


if __name__ == "__main__":
    main()

