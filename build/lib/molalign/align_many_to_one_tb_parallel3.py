#!/usr/bin/env python3
"""
align_many_to_one_tb_parallel.py (rewritten, structured)

Core responsibilities:
- Prepare NPZ-style atoms + grids into convenient arrays
- For one chosen solute grid point idxA:
  - Align every solvent grid point normal to -nA
  - Rotate around that axis by all requested angles
  - Build combined complexes (A + rotated B)
  - Call a user-provided energy function (e.g. TB-lite) for each complex
- Return a structured dict of results for that idxA

This file contains *no* parallel logic and *no* tblite-specific code
beyond the optional default run_tblite_sp for standalone calls.
Parallel + reuse-of-Calculator logic is handled in all_grid_tb_parallel.py.
"""

import os
import numpy as np
import contextlib
import io

BOHR_TO_ANG = 0.52917721092

WRITE_XYZ = os.environ.get("ALIGN_TB_WRITE_XYZ", "1") != "0"

# ---------------------- optional Numba ---------------------- #

try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


def atoms_to_xyz_list(coords, atomic_numbers):
    """Convert coordinates + atomic numbers to simple XYZ-style dicts."""
    return [
        {
            "element": str(int(Z)),
            "x": float(p[0]),
            "y": float(p[1]),
            "z": float(p[2]),
        }
        for p, Z in zip(coords, atomic_numbers)
    ]


# ---------------------- default TB-lite wrapper ---------------------- #

def run_tblite_sp(coords_angstrom, atomic_numbers, method="GFN2-xTB", charge=0):
    """
    Simple TB-lite single-point wrapper.

    NOTE: This is only used if no custom energy_fn is passed.
    For parallel runs, all_grid_tb_parallel.py will inject a more
    efficient per-worker energy function instead.
    """
    from tblite.interface import Calculator

    coords_angstrom = np.asarray(coords_angstrom, dtype=float)
    Z = np.asarray(atomic_numbers, dtype=int)
    coords_bohr = coords_angstrom / BOHR_TO_ANG

    calc = Calculator(method, Z, coords_bohr, charge=charge)
    calc.add("alpb-solvation", "water")

    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        res = calc.singlepoint()
    return res.get("energy")


# ---------------------- rotations (NumPy + optional numba) ---------------------- #

def rotation_matrix_from_vectors_np(a, b, eps=1e-12):
    """
    Return rotation matrix R such that R @ a ≈ b.
    """
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < eps or nb < eps:
        return np.eye(3)

    a /= na
    b /= nb
    dot = np.dot(a, b)

    if dot > 1 - eps:
        return np.eye(3)
    if dot < -1 + eps:
        # 180° rotation: choose arbitrary orthogonal axis
        arbitrary = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            arbitrary = np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, arbitrary)
        axis /= np.linalg.norm(axis)
        # Householder-like reflection
        return -np.eye(3) + 2.0 * np.outer(axis, axis)

    v = np.cross(a, b)
    s = np.linalg.norm(v)
    K = np.array([
        [0.0,    -v[2],  v[1]],
        [v[2],   0.0,   -v[0]],
        [-v[1],  v[0],  0.0 ],
    ])
    return np.eye(3) + K + K @ K * ((1.0 - dot) / (s ** 2))


def rotate_around_axis_np(coords, axis, angle_deg):
    """
    Rotate coords around given axis by angle_deg.
    """
    coords = np.asarray(coords, dtype=float)
    k = np.asarray(axis, dtype=float)
    k /= np.linalg.norm(k)
    theta = np.deg2rad(angle_deg)

    K = np.array([
        [0.0,   -k[2],  k[1]],
        [k[2],   0.0,  -k[0]],
        [-k[1],  k[0],  0.0],
    ])
    I = np.eye(3)
    R = I + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)
    return coords @ R.T


if _HAS_NUMBA:
    @njit
    def _rotation_matrix_from_vectors_nb(a, b, eps=1e-12):
        a = a.astype(np.float64)
        b = b.astype(np.float64)
        na = np.sqrt((a * a).sum())
        nb = np.sqrt((b * b).sum())
        if na < eps or nb < eps:
            return np.eye(3)

        a = a / na
        b = b / nb
        dot = (a * b).sum()

        if dot > 1 - eps:
            return np.eye(3)
        if dot < -1 + eps:
            arbitrary = np.array((1.0, 0.0, 0.0))
            if abs(a[0]) > 0.9:
                arbitrary = np.array((0.0, 1.0, 0.0))
            axis = np.array((
                a[1] * arbitrary[2] - a[2] * arbitrary[1],
                a[2] * arbitrary[0] - a[0] * arbitrary[2],
                a[0] * arbitrary[1] - a[1] * arbitrary[0],
            ))
            norm_axis = np.sqrt((axis * axis).sum())
            axis = axis / norm_axis
            R = -np.eye(3)
            for i in range(3):
                for j in range(3):
                    R[i, j] += 2.0 * axis[i] * axis[j]
            return R

        v = np.array((
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ))
        s = np.sqrt((v * v).sum())
        K = np.zeros((3, 3))
        K[0, 1] = -v[2]
        K[0, 2] =  v[1]
        K[1, 0] =  v[2]
        K[1, 2] = -v[0]
        K[2, 0] = -v[1]
        K[2, 1] =  v[0]

        I = np.eye(3)
        KK = K @ K
        factor = (1.0 - dot) / (s * s)
        return I + K + KK * factor

    @njit
    def _rotate_around_axis_nb(coords, axis, angle_rad):
        coords = coords.astype(np.float64)
        k = axis.astype(np.float64)
        k = k / np.sqrt((k * k).sum())
        K = np.zeros((3, 3))
        K[0, 1] = -k[2]
        K[0, 2] =  k[1]
        K[1, 0] =  k[2]
        K[1, 2] = -k[0]
        K[2, 0] = -k[1]
        K[2, 1] =  k[0]
        I = np.eye(3)
        KK = K @ K
        R = I + np.sin(angle_rad) * K + (1.0 - np.cos(angle_rad)) * KK
        return coords @ R.T

    def rotation_matrix_from_vectors(a, b, eps=1e-12):
        return _rotation_matrix_from_vectors_nb(
            np.array(a, dtype=np.float64),
            np.array(b, dtype=np.float64),
            eps,
        )

    def rotate_around_axis(coords, axis, angle_deg):
        theta = np.deg2rad(angle_deg)
        return _rotate_around_axis_nb(
            np.array(coords, dtype=np.float64),
            np.array(axis, dtype=np.float64),
            theta,
        )

else:
    rotation_matrix_from_vectors = rotation_matrix_from_vectors_np
    rotate_around_axis = rotate_around_axis_np


# ---------------------- geometry preparation ---------------------- #

def prepare_data(data):
    """
    Precompute atoms, atomic numbers, grid positions and normals.

    Parameters
    ----------
    data : dict-like (e.g. np.load result)
        Must contain "atoms" and "grids" arrays:
          atoms: (N_atoms, 5): x,y,z,Z,index
          grids: (N_grid, 5):  x,y,z,Z,atom_index

    Returns
    -------
    dict
        {
          "atoms"    : (Na,3),
          "Z"        : (Na,),
          "grid_xyz" : (Ng,3),
          "atom_idx" : (Ng,),
          "normals"  : (Ng,3),
        }
    """
    atoms_all = data["atoms"]
    grids = data["grids"]

    atoms = atoms_all[:, :3]
    Z = atoms_all[:, 3].astype(int)

    grid_xyz = grids[:, :3]
    atom_idx = grids[:, 4].astype(int)

    # normal = (grid - atom) normalized
    normals = grid_xyz - atoms[atom_idx]
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    safe_norms = np.where(norms > 0.0, norms, 1.0)
    normals = normals / safe_norms

    return {
        "atoms": atoms,
        "Z": Z,
        "grid_xyz": grid_xyz,
        "atom_idx": atom_idx,
        "normals": normals,
    }


# ---------------------- core alignment routine ---------------------- #

def main_from_prepared(
    preA,
    preB,
    idxA,
    angles=[0.0],
    charge=0,
    energy_fn=None,
):
    """
    Align molB to grid idxA of molA, rotate around surface normal,
    run energy_fn for each complex, return dict of results.

    Parameters
    ----------
    preA, preB : dict
        Output from prepare_data(dataA/dataB).
    idxA : int
        Solute grid index on molA.
    angles : list[float]
        Rotation angles (degrees) around normal.
    charge : int
        Total charge for the combined complex.
    energy_fn : callable or None
        Must accept (coords_angstrom, atomic_numbers, method="GFN2-xTB", charge=0)
        and return an energy in Eh. If None, run_tblite_sp is used.

    Returns
    -------
    dict
        keys like "b<idxB>_rot<int(angle)>"
        each value: {"energy_Eh": ..., "xyz": ... (optional)}
    """
    if energy_fn is None:
        energy_fn = run_tblite_sp

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
    target = -nA  # solvent normal should point toward solute

    all_complexes = {}

    # Loop over all solvent grid points
    for idxB, (pB, nB) in enumerate(zip(gridsB, normalsB)):
        # Align B-normal to target
        R_align = rotation_matrix_from_vectors(nB, target)
        atomsB_rot = atomsB @ R_align.T
        gridsB_rot = gridsB @ R_align.T

        # Translation: bring grid point B (after rotation) to pA
        translation = pA - gridsB_rot[idxB]
        atomsB_rot_shift = atomsB_rot + translation

        # Now rotate around the shared axis (target)
        for angle in angles:
            atomsB_final = rotate_around_axis(
                atomsB_rot_shift - pA, axis=target, angle_deg=angle
            ) + pA

            atoms_combined = np.vstack((atomsA, atomsB_final))
            Z_combined = np.hstack((Z_A, Z_B))

            energy = energy_fn(atoms_combined, Z_combined, charge=charge)
            key = f"b{idxB}_rot{int(round(angle))}"

            entry = {"energy_Eh": float(energy)}
            if WRITE_XYZ:
                entry["xyz"] = atoms_to_xyz_list(atoms_combined, Z_combined)

            all_complexes[key] = entry

    return all_complexes


def main_from_data(dataA, dataB, idxA, angles=[0.0], charge=0, energy_fn=None):
    """
    Convenience wrapper: takes npz-like data, prepares, then calls main_from_prepared.
    """
    preA = prepare_data(dataA)
    preB = prepare_data(dataB)
    return main_from_prepared(preA, preB, idxA, angles=angles, charge=charge, energy_fn=energy_fn)


def main(fileA, fileB, idxA, angles=[0.0], charge=0):
    """
    Backwards-compatible CLI-style entry: load NPZ files, run main_from_data.
    """
    dataA = np.load(fileA, allow_pickle=True)
    dataB = np.load(fileB, allow_pickle=True)
    return main_from_data(dataA, dataB, idxA, angles=angles, charge=charge)


if __name__ == "__main__":
    # Light manual test if you want
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("fileA")
    p.add_argument("fileB")
    p.add_argument("idxA", type=int)
    p.add_argument("--angles", default="0,90,180,270")
    p.add_argument("--charge", type=int, default=0)
    args = p.parse_args()

    angles = [float(a) for a in args.angles.split(",")]
    res = main(args.fileA, args.fileB, args.idxA, angles=angles, charge=args.charge)
    print(f"Computed {len(res)} complexes for idxA = {args.idxA}")

