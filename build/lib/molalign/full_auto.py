#!/usr/bin/env python3
"""
molalign_fast.py
================

Optimized + parallelized replacement for:
    - align_many_to_one_tb.py
    - all_grid_tb.py

Features:
    ✔ Loads NPZ once
    ✔ Precomputes normals once
    ✔ Parallel processing of grid points
    ✔ Efficient solvent–solute alignment
    ✔ TB-lite SP calls in parallel
    ✔ Outputs identical JSON format

Usage:
    python molalign_fast.py solute_data.npz solvent_data.npz "0,90,180,270" \
           --charge 0 --out tb_lite_results.json
"""

import os
import json
import numpy as np
import argparse
from multiprocessing import Pool, cpu_count
import functools
import contextlib, io
from tblite.interface import Calculator

# =====================================================================
# Utility
# =====================================================================

BOHR_TO_ANG = 0.52917721092

def atoms_to_xyz_list(coords, atomic_numbers):
    return [{"element": str(int(Z)),
             "x": float(p[0]),
             "y": float(p[1]),
             "z": float(p[2])}
            for p, Z in zip(coords, atomic_numbers)]

def run_tblite_sp(coords_angstrom, atomic_numbers, method="GFN2-xTB", charge=0):
    """Run TB-lite SP and return energy."""
    coords_bohr = np.array(coords_angstrom) / BOHR_TO_ANG
    calc = Calculator(method, np.array(atomic_numbers), coords_bohr, charge=charge)
    calc.add("alpb-solvation", 78.4)

    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        res = calc.singlepoint()
    return res.get("energy")


# =====================================================================
# Geometry: alignment + rotation
# =====================================================================

def rotation_matrix_from_vectors(a, b, eps=1e-12):
    """Return rotation matrix that rotates vector a → vector b."""
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
        # opposite direction
        arbitrary = np.array([1, 0, 0])
        if abs(a[0]) > 0.9:
            arbitrary = np.array([0, 1, 0])
        axis = np.cross(a, arbitrary)
        axis /= np.linalg.norm(axis)
        return -np.eye(3) + 2 * np.outer(axis, axis)

    v = np.cross(a, b)
    s = np.linalg.norm(v)
    K = np.array([[0, -v[2], v[1]],
                  [v[2], 0, -v[0]],
                  [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - dot) / (s * s))

def rotate_around_axis(coords, axis, angle_deg):
    theta = np.radians(angle_deg)
    k = axis / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]])
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    return coords @ R.T


# =====================================================================
# Single-grid alignment & SP evaluation
# =====================================================================

def process_single_grid(idxA, atomsA, Z_A, gridsA, normalsA,
                        atomsB, Z_B, gridsB, normalsB,
                        angles, charge):
    """
    Perform alignment of solvent to solute grid idxA and compute SP energies.
    """
    pA = gridsA[idxA]
    nA = normalsA[idxA]
    target = -nA

    results = {}

    for idxB, (pB, nB) in enumerate(zip(gridsB, normalsB)):
        # Align normals
        R_align = rotation_matrix_from_vectors(nB, target)

        atomsB_rot = atomsB @ R_align.T
        gridsB_rot = gridsB @ R_align.T

        # Translate so solvent grid idxB -> solute grid idxA
        translation = pA - gridsB_rot[idxB]
        atomsB_rot += translation
        gridsB_rot += translation

        # Sweep rotations
        for angle in angles:
            axis = target
            atomsB_final = rotate_around_axis(atomsB_rot - pA, axis, angle) + pA

            # Combine with solute
            atoms_combined = np.vstack([atomsA, atomsB_final])
            Z_combined = np.hstack([Z_A, Z_B])

            key = f"b{idxB}_rot{int(angle)}"
            energy = run_tblite_sp(atoms_combined, Z_combined, charge=charge)

            results[key] = {
                "energy_Eh": float(energy),
                "xyz": atoms_to_xyz_list(atoms_combined, Z_combined)
            }

    return results


# =====================================================================
# Parallel wrapper
# =====================================================================

def worker(idxA, atomsA, Z_A, gridsA, normalsA,
           atomsB, Z_B, gridsB, normalsB,
           angles, charge):
    try:
        res = process_single_grid(
            idxA,
            atomsA, Z_A, gridsA, normalsA,
            atomsB, Z_B, gridsB, normalsB,
            angles, charge
        )
        return idxA, res
    except Exception as e:
        return idxA, {"error": str(e)}


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Parallel grid-based molalign + TB-lite SP")
    parser.add_argument("fileA", help="solute NPZ")
    parser.add_argument("fileB", help="solvent NPZ")
    parser.add_argument("angles_csv", help="comma-separated rotation angles")
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--out", default="tb_lite_results.json")
    args = parser.parse_args()

    angles = [float(a) for a in args.angles_csv.split(",")]

    # =============================================================
    # Load NPZ once
    # =============================================================
    dataA = np.load(args.fileA, allow_pickle=True)
    dataB = np.load(args.fileB, allow_pickle=True)

    atomsA = dataA["atoms"][:, :3]
    Z_A = dataA["atoms"][:, 3].astype(int)
    gridsA = dataA["grids"][:, :3]
    ownersA = dataA["grids"][:, 4].astype(int)

    atomsB = dataB["atoms"][:, :3]
    Z_B = dataB["atoms"][:, 3].astype(int)
    gridsB = dataB["grids"][:, :3]
    ownersB = dataB["grids"][:, 4].astype(int)

    # =============================================================
    # Precompute normals once
    # =============================================================
    normalsA = gridsA - atomsA[ownersA]
    normalsA /= np.linalg.norm(normalsA, axis=1, keepdims=True)

    normalsB = gridsB - atomsB[ownersB]
    normalsB /= np.linalg.norm(normalsB, axis=1, keepdims=True)

    # =============================================================
    # Prepare results dictionary
    # =============================================================
    results = {
        "molA_grids_xyz": gridsA.tolist(),
        "solute": {
            "energy_Eh": float(run_tblite_sp(atomsA, Z_A, charge=args.charge)),
            "charge": args.charge,
            "xyz": atoms_to_xyz_list(atomsA, Z_A),
        },
        "solvent": {
            "energy_Eh": float(run_tblite_sp(atomsB, Z_B, charge=0)),
            "charge": 0,
            "xyz": atoms_to_xyz_list(atomsB, Z_B),
        }
    }

    num_grids = gridsA.shape[0]

    # =============================================================
    # Parallel execution
    # =============================================================
    print(f"Running with {cpu_count()} CPUs...")
    pool = Pool(cpu_count())

    worker_func = functools.partial(
        worker,
        atomsA=atomsA, Z_A=Z_A, gridsA=gridsA, normalsA=normalsA,
        atomsB=atomsB, Z_B=Z_B, gridsB=gridsB, normalsB=normalsB,
        angles=angles, charge=args.charge
    )

    for idxA, r in pool.imap(worker_func, range(num_grids)):
        results[f"gp{idxA}"] = r
        print(f"Completed grid {idxA}/{num_grids - 1}")

    pool.close()
    pool.join()

    # =============================================================
    # Save JSON
    # =============================================================
    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)

    print(f"\n*** Saved full TB-lite results to {args.out}")


# =====================================================================
# Entry point
# =====================================================================

if __name__ == "__main__":
    main()

