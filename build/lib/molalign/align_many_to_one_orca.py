#!/usr/bin/env python3
"""
align_many_to_one_orca_parallel3.py

Geometry + alignment core, analogous to align_many_to_one_tb_parallel3.py,
but using ORCA via OPI to evaluate single-point energies (r2SCAN-3c + CPCM(water)).

Differences vs TB-lite:
  - Energy backend is ORCA (OPI) with:
        ! r2SCAN-3c CPCM(water) TightSCF
  - No OPI ncores are set; ORCA uses environment defaults.
  - Uses TemporaryDirectory internally so no files are left behind.

Parallelization is handled in all_grid_orca_parallel3.py.
"""

import os
import numpy as np
import contextlib
import io
import tempfile
from pathlib import Path

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


# ---------------------- ORCA OPI wrapper ---------------------- #

Z2SYM = {
    1: "H",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    15: "P",
    16: "S",
    17: "Cl",
    35: "Br",
}


def run_orca_sp(
    coords_angstrom,
    atomic_numbers,
    charge=0,
    multiplicity=1,
):
    """
    ORCA OPI single-point with:
        ! r2SCAN-3c CPCM(water) TightSCF

    - Uses a TemporaryDirectory per call.
    - Parses "FINAL SINGLE POINT ENERGY" from the .out file.
    """
    from opi.core import Calculator
    from opi.input.structures.atom import Atom
    from opi.input.structures.structure import Structure

    coords = np.asarray(coords_angstrom, float)
    Z = np.asarray(atomic_numbers, int)

    atoms = []
    for (x, y, z), Zi in zip(coords, Z):
        Zi = int(Zi)
        if Zi not in Z2SYM:
            raise ValueError(f"Atomic number {Zi} not supported by Z2SYM")
        atoms.append(
            Atom(
                element=Z2SYM[Zi],
                coordinates=(float(x), float(y), float(z)),
            )
        )

    structure = Structure(
        atoms=atoms,
        charge=int(charge),
        multiplicity=int(multiplicity),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        workdir = Path(tmpdir)
        basename = "sp"

        calc = Calculator(basename=basename, working_dir=workdir)
        calc.structure = structure

        # IMPORTANT: same as standalone script
        calc.input.add_arbitrary_string("!pal4 r2SCAN-3c CPCM(water) TightSCF")

        # DO NOT set calc.input.ncores here → ORCA uses environment defaults

        # Run ORCA via OPI
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            calc.write_input()
            calc.run()

        outfile = workdir / f"{basename}.out"
        energy = None
        with open(outfile, "r") as fh:
            for line in fh:
                if "FINAL SINGLE POINT ENERGY" in line:
                    energy = float(line.split()[-1])
                    break

        if energy is None:
            raise RuntimeError(f"Energy not found in {outfile}")

        return energy


# ---------------------- rotations (NumPy + optional numba) ---------------------- #

def rotation_matrix_from_vectors_np(a, b, eps=1e-12):
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
        arbitrary = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            arbitrary = np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, arbitrary)
        axis /= np.linalg.norm(axis)
        return -np.eye(3) + 2.0 * np.outer(axis, axis)

    v = np.cross(a, b)
    s = np.linalg.norm(v)
    K = np.array([
        [0.0,    -v[2],  v[1]],
        [v[2],   0.0,   -v[0]],
        [-v[1],  v[0],  0.0],
    ])
    return np.eye(3) + K + K @ K * ((1.0 - dot) / (s ** 2))


def rotate_around_axis_np(coords, axis, angle_deg):
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
    atoms_all = data["atoms"]
    grids = data["grids"]

    atoms = atoms_all[:, :3]
    Z = atoms_all[:, 3].astype(int)

    grid_xyz = grids[:, :3]
    atom_idx = grids[:, 4].astype(int)

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
    multiplicity=1,
):
    """
    Align molB to grid idxA of molA, rotate around surface normal,
    run energy_fn for each complex, return dict of results.
    """
    if energy_fn is None:
        def energy_fn(coords, Z, charge=charge, multiplicity=multiplicity):
            return run_orca_sp(coords, Z, charge=charge, multiplicity=multiplicity)

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
    target = -nA

    all_complexes = {}

    for idxB, (pB, nB) in enumerate(zip(gridsB, normalsB)):
        R_align = rotation_matrix_from_vectors(nB, target)
        atomsB_rot = atomsB @ R_align.T
        gridsB_rot = gridsB @ R_align.T

        translation = pA - gridsB_rot[idxB]
        atomsB_rot_shift = atomsB_rot + translation

        for angle in angles:
            atomsB_final = rotate_around_axis(
                atomsB_rot_shift - pA, axis=target, angle_deg=angle
            ) + pA

            atoms_combined = np.vstack((atomsA, atomsB_final))
            Z_combined = np.hstack((Z_A, Z_B))

            energy = energy_fn(
                atoms_combined,
                Z_combined,
                charge=charge,
                multiplicity=multiplicity,
            )
            key = f"b{idxB}_rot{int(round(angle))}"

            entry = {"energy_Eh": float(energy)}
            if WRITE_XYZ:
                entry["xyz"] = atoms_to_xyz_list(atoms_combined, Z_combined)
            all_complexes[key] = entry

    return all_complexes


def main_from_data(
    dataA,
    dataB,
    idxA,
    angles=[0.0],
    charge=0,
    energy_fn=None,
    multiplicity=1,
):
    preA = prepare_data(dataA)
    preB = prepare_data(dataB)
    return main_from_prepared(
        preA,
        preB,
        idxA,
        angles=angles,
        charge=charge,
        energy_fn=energy_fn,
        multiplicity=multiplicity,
    )


def main(fileA, fileB, idxA, angles=[0.0], charge=0, multiplicity=1):
    dataA = np.load(fileA, allow_pickle=True)
    dataB = np.load(fileB, allow_pickle=True)
    return main_from_data(
        dataA,
        dataB,
        idxA,
        angles=angles,
        charge=charge,
        multiplicity=multiplicity,
    )


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("fileA")
    p.add_argument("fileB")
    p.add_argument("idxA", type=int)
    p.add_argument("--angles", default="0,90,180,270")
    p.add_argument("--charge", type=int, default=0)
    p.add_argument("--multiplicity", type=int, default=1)
    args = p.parse_args()

    angles = [float(a) for a in args.angles.split(",")]
    res = main(
        args.fileA,
        args.fileB,
        args.idxA,
        angles=angles,
        charge=args.charge,
        multiplicity=args.multiplicity,
    )
    print(f"Computed {len(res)} complexes for idxA = {args.idxA}")

