#!/usr/bin/env python3
"""
align_many_to_one_tb.py

Align molB to molA grid points, run TB-lite SP, and return results.
"""
import numpy as np
import os
from tblite.interface import Calculator
import contextlib, io

BOHR_TO_ANG = 0.52917721092

def atoms_to_xyz_list(coords, atomic_numbers):
    """Convert coordinates + atomic numbers to XYZ list (dicts)."""
    return [{"element": str(int(Z)), "x": float(p[0]), "y": float(p[1]), "z": float(p[2])}
            for p, Z in zip(coords, atomic_numbers)]

def run_tblite_sp(coords_angstrom, atomic_numbers, method="GFN2-xTB", charge=0):
    """Run TB-lite single-point, suppress verbose output, supports charge."""
    coords_bohr = np.array(coords_angstrom) / BOHR_TO_ANG
    calc = Calculator(method, np.array(atomic_numbers), coords_bohr, charge=charge)
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
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K.dot(K) * ((1 - dot) / (s ** 2))

def rotate_around_axis(coords, axis, angle_deg):
    theta = np.radians(angle_deg)
    k = axis / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    return coords @ R.T

def main(fileA, fileB, idxA, angles=[0], charge=0):
    """
    Align molB to grid idxA of molA, run TB-lite SP, return dict of results.
    Returns:
        dict: keyed by 'b{idxB}_rot{angle}', each value contains
              'energy_Eh', 'xyz' (combined atoms)
    """
    dataA = np.load(fileA, allow_pickle=True)
    dataB = np.load(fileB, allow_pickle=True)

    atomsA = dataA["atoms"][:, :3]
    Z_A = dataA["atoms"][:, 3].astype(int)
    gridsA = dataA["grids"][:, :3]
    normalsA = gridsA - atomsA[dataA["grids"][:, 4].astype(int)]
    norms = np.linalg.norm(normalsA, axis=1, keepdims=True)
    normalsA /= np.where(norms > 0, norms, 1.0)

    atomsB = dataB["atoms"][:, :3]
    Z_B = dataB["atoms"][:, 3].astype(int)
    gridsB = dataB["grids"][:, :3]
    normalsB = gridsB - atomsB[dataB["grids"][:, 4].astype(int)]
    norms = np.linalg.norm(normalsB, axis=1, keepdims=True)
    normalsB /= np.where(norms > 0, norms, 1.0)

    pA = gridsA[idxA]
    nA = normalsA[idxA]

    all_complexes = {}

    for idxB, (pB, nB) in enumerate(zip(gridsB, normalsB)):
        target = -nA
        R_align = rotation_matrix_from_vectors(nB, target)
        atomsB_rot = atomsB @ R_align.T
        gridsB_rot = gridsB @ R_align.T
        normalsB_rot = normalsB @ R_align.T

        translation = pA - gridsB_rot[idxB]
        atomsB_rot += translation
        gridsB_rot += translation

        for angle in angles:
            axis = target
            atomsB_final = rotate_around_axis(atomsB_rot - pA, axis, angle) + pA
            atoms_combined = np.vstack((atomsA, atomsB_final))
            Z_combined = np.hstack((Z_A, Z_B))

            key = f"b{idxB}_rot{int(angle)}"
            # Run TB-lite SP for this complex with charge
            energy = run_tblite_sp(atoms_combined, Z_combined, charge=charge)
            all_complexes[key] = {
                "energy_Eh": float(energy),
                "xyz": atoms_to_xyz_list(atoms_combined, Z_combined)
            }

    return all_complexes

