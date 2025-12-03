#!/usr/bin/env python3
import os
import numpy as np
import subprocess
import tempfile
import re

BOHR_TO_ANG = 0.52917721092
WRITE_XYZ = os.environ.get("ALIGN_TB_WRITE_XYZ", "1") != "0"
ORCA_EXE = os.environ.get("ORCA_EXE", "orca")


def atoms_to_xyz_list(coords, atomic_numbers):
    return [{"element": str(int(Z)), "x": float(x), "y": float(y), "z": float(z)}
            for (x, y, z), Z in zip(coords, atomic_numbers)]


# ====================== ORCA ENERGY ====================== #

def write_orca_input(path, coords, Z, charge):
    with open(path, "w") as f:
        f.write("! r2scan-3c TightSCF\n")
        f.write("%cpcm\n")
        f.write("  smd false\n")
        f.write("end\n")
        f.write(f"* xyz {charge} 1\n")
        for (x, y, z), Z in zip(coords, Z):
            f.write(f"{int(Z)} {x:.8f} {y:.8f} {z:.8f}\n")
        f.write("*\n")


def parse_energy(outfile):
    txt = open(outfile).read()
    m = re.search(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)", txt)
    if not m:
        raise RuntimeError("ORCA energy not found")
    return float(m.group(1))


def run_orca_sp(coords_angstrom, atomic_numbers, charge=0):
    coords = np.asarray(coords_angstrom)
    Z = np.asarray(atomic_numbers, dtype=int)

    with tempfile.TemporaryDirectory(prefix="orca_") as tmp:
        inp = os.path.join(tmp, "job.inp")
        out = os.path.join(tmp, "job.out")

        write_orca_input(inp, coords, Z, charge)

        subprocess.run(
            [ORCA_EXE, inp],
            stdout=open(out, "w"),
            stderr=subprocess.STDOUT,
            cwd=tmp,
            check=True,
        )

        return parse_energy(out)


# ====================== ROTATIONS ====================== #

def rotation_matrix_from_vectors(a, b, eps=1e-12):
    a = np.array(a, float)
    b = np.array(b, float)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = np.dot(a, b)
    if s < eps:
        return np.eye(3)
    k = np.array([[0, -v[2], v[1]],
                  [v[2], 0, -v[0]],
                  [-v[1], v[0], 0]])
    return np.eye(3) + k + k @ k * ((1 - c) / (s ** 2))


def rotate_around_axis(coords, axis, angle_deg):
    axis = axis / np.linalg.norm(axis)
    theta = np.deg2rad(angle_deg)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    return coords @ R.T


# ====================== DATA ====================== #

def prepare_data(data):
    atoms = data["atoms"][:, :3]
    Z = data["atoms"][:, 3].astype(int)
    grids = data["grids"][:, :3]
    atom_idx = data["grids"][:, 4].astype(int)
    normals = grids - atoms[atom_idx]
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    return dict(atoms=atoms, Z=Z, grid_xyz=grids, normals=normals)


# ====================== MAIN PER GRID ====================== #

def main_from_prepared(preA, preB, idxA, angles, charge, energy_fn=None):
    if energy_fn is None:
        energy_fn = run_orca_sp

    A = preA["atoms"]
    ZA = preA["Z"]
    gridsA = preA["grid_xyz"]
    normalsA = preA["normals"]

    B = preB["atoms"]
    ZB = preB["Z"]
    gridsB = preB["grid_xyz"]
    normalsB = preB["normals"]

    pA = gridsA[idxA]
    target = -normalsA[idxA]

    results = {}

    for idxB, nB in enumerate(normalsB):
        R = rotation_matrix_from_vectors(nB, target)
        Brot = B @ R.T
        gridB_rot = gridsB @ R.T
        shift = pA - gridB_rot[idxB]
        Brot += shift

        for ang in angles:
            Bfin = rotate_around_axis(Brot - pA, target, ang) + pA
            coords = np.vstack((A, Bfin))
            Z = np.hstack((ZA, ZB))
            E = energy_fn(coords, Z, charge=charge)

            key = f"b{idxB}_rot{int(round(ang))}"
            entry = {"energy_Eh": float(E)}
            if WRITE_XYZ:
                entry["xyz"] = atoms_to_xyz_list(coords, Z)

            results[key] = entry

    return results

