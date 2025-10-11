#!/usr/bin/env python3
"""
allign_many_to_one_tb.py

Align molB to all grid points of molA, compute TB-Lite energies,
and save everything into a single JSON file.
"""

import numpy as np
import json
from multiprocessing import Pool, cpu_count
from tblite.interface import Calculator

# -------------------------------
# Top-level functions (for Pool)
# -------------------------------

def compute_tb_energy(atoms_combined, Z_combined):
    """Compute TB-Lite energy of a complex."""
    calc = Calculator(method="GFN2-xTB", numbers=Z_combined, positions=atoms_combined)
    result = calc.singlepoint()
    energy = float(result["energy"])
    return energy

def atoms_to_xyz_dict(coords, atomic_numbers):
    return [{"element": int(Z), "x": float(x), "y": float(y), "z": float(z)}
            for Z, (x, y, z) in zip(atomic_numbers, coords)]

def worker(task):
    gp_key, key, atoms_combined, Z_combined = task
    energy = compute_tb_energy(atoms_combined, Z_combined)
    xyz_dict = atoms_to_xyz_dict(atoms_combined, Z_combined)
    print(f"Processed {gp_key} {key}: energy={energy:.6f}", flush=True)
    return gp_key, key, energy, xyz_dict

# -------------------------------
# Helper functions
# -------------------------------

def load_npz_file(npz_file):
    data = np.load(npz_file)
    atoms_array = data["atoms"]       # x, y, z, Z, index
    grids_array = data["grids"]       # x, y, z, Z, index

    atom_coords = atoms_array[:, :3]
    atom_numbers = atoms_array[:, 3].astype(int)
    atom_indices = atoms_array[:, 4].astype(int)

    grid_coords = grids_array[:, :3]
    grid_indices = grids_array[:, 4].astype(int)
    grid_numbers = atom_numbers[grid_indices]

    normals = grid_coords - atom_coords[grid_indices]
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.where(norms > 0, norms, 1.0)

    return atom_coords, atom_numbers, atom_indices, grid_coords, grid_numbers, grid_indices, normals

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

# -------------------------------
# Main function
# -------------------------------

def main(fileA, fileB, angles=[0], output_json="complexes_tb.json"):
    atomsA, Z_A, idxsA, gridsA, Z_gridsA, idxs_gridsA, normalsA = load_npz_file(fileA)
    atomsB, Z_B, idxsB, gridsB, Z_gridsB, idxs_gridsB, normalsB = load_npz_file(fileB)

    results = {}
    tasks = []

    for idxA, (pA, nA) in enumerate(zip(gridsA, normalsA)):
        gp_key = f"gp{idxA}"
        results[gp_key] = {}

        for idxB, (pB, nB) in enumerate(zip(gridsB, normalsB)):
            target = -nA
            R_align = rotation_matrix_from_vectors(nB, target)
            atomsB_rot = atomsB @ R_align.T
            gridsB_rot = gridsB @ R_align.T

            translation = pA - gridsB_rot[idxB]
            atomsB_rot += translation

            for angle in angles:
                atomsB_final = rotate_around_axis(atomsB_rot - pA, target, angle) + pA
                atoms_combined = np.vstack((atomsA, atomsB_final))
                Z_combined = np.hstack((Z_A, Z_B))
                key = f"b{idxB}_rot{int(angle)}"
                tasks.append((gp_key, key, atoms_combined, Z_combined))

    # --- Multiprocessing or serial fallback ---
    n_cpus = min(cpu_count(), 4)
    if len(tasks) < 2:
        for task in tasks:
            gp_key, key, energy, xyz_dict = worker(task)
            results[gp_key][key] = {"energy_Eh": energy, "xyz": xyz_dict}
    else:
        with Pool(n_cpus) as pool:
            for gp_key, key, energy, xyz_dict in pool.imap_unordered(worker, tasks):
                results[gp_key][key] = {"energy_Eh": energy, "xyz": xyz_dict}

    # --- Save JSON ---
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"*** Saved all complexes with energies to {output_json}", flush=True)
    print(f"*** Total grid points: {len(results)}", flush=True)

# -------------------------------
# CLI entry
# -------------------------------

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: ./align_many_to_one_tb.py molA.npz molB.npz [angles_csv] [output_json]")
        sys.exit(1)

    fileA = sys.argv[1]
    fileB = sys.argv[2]
    angles_csv = sys.argv[3] if len(sys.argv) >= 4 else "0"
    angles = [float(a) for a in angles_csv.split(",")]
    output_json = sys.argv[4] if len(sys.argv) >= 5 else "complexes_tb.json"

    main(fileA, fileB, angles, output_json)
