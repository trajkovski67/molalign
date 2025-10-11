#!/usr/bin/env python3
import numpy as np
import sys
import os
import subprocess
import json
import re
import multiprocessing as mp

def load_npz_file(npz_file):
    data = np.load(npz_file)
    atoms_array = data["atoms"]
    grids_array = data["grids"]
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
    return np.eye(3) + K + K.dot(K) * ((1 - dot) / (s**2))

def rotate_around_axis(coords, axis, angle_deg):
    theta = np.radians(angle_deg)
    k = axis / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    return coords @ R.T

def to_xyz(filename, coords, atomic_numbers):
    with open(filename, 'w') as f:
        f.write(f"{len(coords)}\n{filename}\n")
        for p, Z in zip(coords, atomic_numbers):
            f.write(f"{Z} {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n")

def parse_tblite_output(out_path):
    energy = None
    grad_norm = None
    try:
        with open(out_path, 'r') as f:
            for line in f:
                if "Total Energy" in line:
                    match = re.search(r"([-]?\d+\.\d+)", line)
                    if match:
                        energy = float(match.group(1))
                elif "Gradient norm" in line:
                    match = re.search(r"([-]?\d+\.\d+)", line)
                    if match:
                        grad_norm = float(match.group(1))
    except FileNotFoundError:
        return None, None
    return energy, grad_norm

def run_tblite_job(job):
    """Worker function for multiprocessing pool."""
    xyz_path, key = job
    out_path = xyz_path.replace(".xyz", ".out")
    try:
        subprocess.run(
            ["tblite", xyz_path, "--sp", "--output", out_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
        )
        energy, grad_norm = parse_tblite_output(out_path)
    except subprocess.CalledProcessError:
        return key, None

    # Read atoms for JSON entry
    atoms = []
    with open(xyz_path) as f:
        lines = f.readlines()[2:]
        for line in lines:
            parts = line.split()
            if len(parts) < 4:
                continue
            atoms.append({
                "element": parts[0],
                "x": float(parts[1]),
                "y": float(parts[2]),
                "z": float(parts[3])
            })

    if energy is None:
        return key, None

    return key, {"energy_Eh": energy, "grad_norm_Eh_per_a": grad_norm, "xyz": atoms}

def main(fileA, fileB, idxA, angles=[0], prefix="complex"):
    atomsA, Z_A, idxsA, gridsA, Z_gridsA, idxs_gridsA, normalsA = load_npz_file(fileA)
    atomsB, Z_B, idxsB, gridsB, Z_gridsB, idxs_gridsB, normalsB = load_npz_file(fileB)

    if idxA < 0 or idxA >= len(gridsA):
        raise IndexError(f"idxA out of range 0..{len(gridsA)-1}")

    pA = gridsA[idxA]
    nA = normalsA[idxA]
    target = -nA

    xyz_folder = f"{prefix}_xyz_files"
    os.makedirs(xyz_folder, exist_ok=True)
    results = {f"gp{idxA}": {}}

    jobs = []

    # Generate all complexes and queue jobs
    for idxB, (pB, nB) in enumerate(zip(gridsB, normalsB)):
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
            xyz_path = os.path.join(xyz_folder, f"{prefix}_{key}_atoms.xyz")
            to_xyz(xyz_path, atoms_combined, Z_combined)
            jobs.append((xyz_path, key))

    # Run tblite jobs in parallel
    print(f"🧠 Launching {len(jobs)} tblite jobs on {mp.cpu_count()} cores...")
    with mp.Pool(processes=mp.cpu_count()) as pool:
        for key, result in pool.imap_unordered(run_tblite_job, jobs):
            if result:
                results[f"gp{idxA}"][key] = result

    with open(f"{prefix}_results.json", "w") as jf:
        json.dump(results, jf, indent=2)
    print(f"✅ Finished — results saved to {prefix}_results.json")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: ./align_one_to_many_tblite.py molA.npz molB.npz idxA [angles_csv] [output_prefix]")
        sys.exit(1)

    fileA = sys.argv[1]
    fileB = sys.argv[2]
    idxA = int(sys.argv[3])
    angles_csv = sys.argv[4] if len(sys.argv) >= 5 else "0"
    angles = [float(a) for a in angles_csv.split(",")]
    prefix = sys.argv[5] if len(sys.argv) >= 6 else "complex"

    main(fileA, fileB, idxA, angles, prefix)

