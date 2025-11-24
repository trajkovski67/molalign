#!/usr/bin/env python3
"""
align_many_to_one_tb.py

Align molB to molA grid points, run TB-lite SP, and return results.
"""

import os
import numpy as np
from tblite.interface import Calculator
import contextlib, io

BOHR_TO_ANG = 0.52917721092

# Control whether to store full xyz data for each complex
WRITE_XYZ = os.environ.get("ALIGN_TB_WRITE_XYZ", "1") != "0"


def atoms_to_xyz_list(coords, atomic_numbers):
    """Convert coordinates + atomic numbers to XYZ list (dicts)."""
    return [
        {
            "element": str(int(Z)),
            "x": float(p[0]),
            "y": float(p[1]),
            "z": float(p[2]),
        }
        for p, Z in zip(coords, atomic_numbers)
    ]


def run_tblite_sp(coords_angstrom, atomic_numbers, method="GFN2-xTB", charge=0):
    """Run TB-lite single-point, suppress verbose output, supports charge."""
    coords_bohr = np.array(coords_angstrom) / BOHR_TO_ANG
    calc = Calculator(method, np.array(atomic_numbers), coords_bohr, charge=charge)
    # calc.set("accuracy", 1.0)
    # calc.set("max-iter", 500)
    calc.add("alpb-solvation", "water")
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        res = calc.singlepoint()
    return res.get("energy")


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
    K = np.array(
        [
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ]
    )
    return np.eye(3) + K + K.dot(K) * ((1 - dot) / (s ** 2))


def rotate_around_axis(coords, axis, angle_deg):
    theta = np.radians(angle_deg)
    k = axis / np.linalg.norm(axis)
    K = np.array(
        [
            [0, -k[2], k[1]],
            [k[2], 0, -k[0]],
            [-k[1], k[0], 0],
        ]
    )
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    return coords @ R.T


# ---------------------------------------------------------------------
# NEW: precomputation helper
# ---------------------------------------------------------------------
def prepare_data(data):
    """
    Precompute atoms, atomic numbers, grid positions and normals.

    Parameters
    ----------
    data : dict- or npz-like
        Must have keys "atoms" and "grids".

    Returns
    -------
    dict
        { "atoms", "Z", "grid_xyz", "atom_idx", "normals" }
    """
    atoms = data["atoms"][:, :3]
    Z = data["atoms"][:, 3].astype(int)

    grids = data["grids"]
    grid_xyz = grids[:, :3]
    atom_idx = grids[:, 4].astype(int)

    normals = grid_xyz - atoms[atom_idx]
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    # Correct, robust normalization
    safe_norms = np.where(norms > 0, norms, 1.0)
    normals = normals / safe_norms

    return {
        "atoms": atoms,
        "Z": Z,
        "grid_xyz": grid_xyz,
        "atom_idx": atom_idx,
        "normals": normals,
    }


# ---------------------------------------------------------------------
# NEW: core that works on precomputed data (for parallel workers)
# ---------------------------------------------------------------------
def main_from_prepared(preA, preB, idxA, angles=[0], charge=0):
    """
    Align molB to grid idxA of molA, run TB-lite SP, return dict of results.

    Parameters
    ----------
    preA, preB : dict
        Result of prepare_data(dataA) / prepare_data(dataB).
    idxA : int
        Grid index on molA.
    angles : list of float
        Rotation angles (deg) around surface normal.
    charge : int
        Total charge for the combined complex.

    Returns
    -------
    dict
        keyed by 'b{idxB}_rot{angle}', each value contains
        'energy_Eh', and optionally 'xyz' (combined atoms).
    """
    atomsA = preA["atoms"]
    Z_A = preA["Z"]
    gridsA = preA["grid_xyz"]
    normalsA = preA["normals"]

    atomsB = preB["atoms"]
    Z_B = preB["Z"]
    gridsB = preB["grid_xyz"]
    normalsB = preB["normals"]

    pA = gridsA[idxA]
    nA = normalsA[idxA]
    target = -nA  # fixed for this grid point

    all_complexes = {}

    for idxB, (pB, nB) in enumerate(zip(gridsB, normalsB)):
        # Align normal of B at grid idxB to -nA
        R_align = rotation_matrix_from_vectors(nB, target)
        atomsB_rot = atomsB @ R_align.T
        gridsB_rot = gridsB @ R_align.T

        # Translate so grid point B (after rotation) sits at pA
        translation = pA - gridsB_rot[idxB]
        atomsB_rot_shift = atomsB_rot + translation

        for angle in angles:
            axis = target
            atomsB_final = rotate_around_axis(atomsB_rot_shift - pA, axis, angle) + pA
            atoms_combined = np.vstack((atomsA, atomsB_final))
            Z_combined = np.hstack((Z_A, Z_B))

            key = f"b{idxB}_rot{int(angle)}"
            energy = run_tblite_sp(atoms_combined, Z_combined, charge=charge)

            entry = {"energy_Eh": float(energy)}
            if WRITE_XYZ:
                entry["xyz"] = atoms_to_xyz_list(atoms_combined, Z_combined)
            all_complexes[key] = entry

    return all_complexes


# ---------------------------------------------------------------------
# Helper that works with raw (npz-like) data
# ---------------------------------------------------------------------
def main_from_data(dataA, dataB, idxA, angles=[0], charge=0):
    """
    Align molB to grid idxA of molA, run TB-lite SP, return dict of results.

    This version accepts npz-like data and internally uses precomputed arrays.
    """
    preA = prepare_data(dataA)
    preB = prepare_data(dataB)
    return main_from_prepared(preA, preB, idxA, angles=angles, charge=charge)


# ---------------------------------------------------------------------
# Original API: still available, wrapper around main_from_data.
# ---------------------------------------------------------------------
def main(fileA, fileB, idxA, angles=[0], charge=0):
    """
    Backwards-compatible wrapper: loads NPZ files and delegates to main_from_data.
    """
    dataA = np.load(fileA, allow_pickle=True)
    dataB = np.load(fileB, allow_pickle=True)
    return main_from_data(dataA, dataB, idxA, angles=angles, charge=charge)

