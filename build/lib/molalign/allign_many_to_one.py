#!/usr/bin/env python3
"""
align_one_to_many_npz.py

For each grid point j of molB aligned to chosen grid idxA of molA,
generate combined atom positions, grid points, normals, and save everything
in a single .npz, plus one xyz file for the combined atoms per complex.
"""
import numpy as np
import sys
import os
def load_npz_file(npz_file):
    data = np.load(npz_file)
    atoms_array = data["atoms"]       # x, y, z, Z, index
    grids_array = data["grids"]       # x, y, z, Z, index
    atom_coords = atoms_array[:, :3]
    atom_numbers = atoms_array[:, 3].astype(int)
    atom_indices = atoms_array[:, 4].astype(int)

    grid_coords = grids_array[:, :3]
    grid_indices = grids_array[:, 4].astype(int)
    grid_numbers = atom_numbers[grid_indices]  # map atomic number via atom index

    normals = grid_coords - atom_coords[grid_indices]
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.where(norms>0, norms, 1.0)

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
        arbitrary = np.array([1,0,0])
        if abs(a[0])>0.9:
            arbitrary = np.array([0,1,0])
        axis = np.cross(a, arbitrary)
        axis /= np.linalg.norm(axis)
        return -np.eye(3) + 2*np.outer(axis, axis)
    v = np.cross(a,b)
    s = np.linalg.norm(v)
    K = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    return np.eye(3) + K + K.dot(K)*((1-dot)/(s**2))

def rotate_around_axis(coords, axis, angle_deg):
    theta = np.radians(angle_deg)
    k = axis/np.linalg.norm(axis)
    K = np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]])
    R = np.eye(3) + np.sin(theta)*K + (1-np.cos(theta))*(K@K)
    return coords @ R.T

def to_xyz(filename, coords, atomic_numbers):
    with open(filename,'w') as f:
        f.write(f"{len(coords)}\n{filename}\n")
        for p,Z in zip(coords, atomic_numbers):
            f.write(f"{Z} {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n")

def main(fileA, fileB, idxA, angles=[0], prefix="complex"):
    atomsA, Z_A, idxsA, gridsA, Z_gridsA, idxs_gridsA, normalsA = load_npz_file(fileA)
    atomsB, Z_B, idxsB, gridsB, Z_gridsB, idxs_gridsB, normalsB = load_npz_file(fileB)

    if idxA<0 or idxA>=len(gridsA):
        raise IndexError(f"idxA out of range 0..{len(gridsA)-1}")

    pA = gridsA[idxA]
    nA = normalsA[idxA]

    all_complexes = {}  # dict keyed by "gp{idxA}_b{idxB}_rot{angle}"

    # Create folder for XYZ files
    xyz_folder = f"{prefix}_xyz_files"
    os.makedirs(xyz_folder, exist_ok=True)


    for idxB, (pB, nB) in enumerate(zip(gridsB, normalsB)):
        # Rotate molB so nB points opposite to nA
        target = -nA
        R_align = rotation_matrix_from_vectors(nB, target)
        atomsB_rot = atomsB @ R_align.T
        gridsB_rot = gridsB @ R_align.T
        normalsB_rot = normalsB @ R_align.T

        # Translate molB so idxB touches idxA
        translation = pA - gridsB_rot[idxB]
        atomsB_rot += translation
        gridsB_rot += translation

        for angle in angles:
            axis = target
            atomsB_final = rotate_around_axis(atomsB_rot - pA, axis, angle) + pA
            gridsB_final = rotate_around_axis(gridsB_rot - pA, axis, angle) + pA
            normalsB_final = rotate_around_axis(normalsB_rot, axis, angle)

            atoms_combined = np.vstack((atomsA, atomsB_final))
            Z_combined = np.hstack((Z_A, Z_B))
            idx_combined = np.hstack((idxsA, idxsB))

            grids_combined = np.vstack((gridsA, gridsB_final))
            Z_grids_combined = np.hstack((Z_gridsA, Z_gridsB))
            idx_grids_combined = np.hstack((idxs_gridsA, idxs_gridsB))

            normals_combined = np.vstack((normalsA, normalsB_final))

            key = f"gp{idxA}_b{idxB}_rot{int(angle)}"
            all_complexes[key] = {
                "atoms": np.column_stack((atoms_combined, Z_combined, idx_combined)),
                "grids": np.column_stack((grids_combined, Z_grids_combined, idx_grids_combined)),
                "normals": normals_combined
            }

            # write XYZ for atoms
            xyz_path = os.path.join(xyz_folder, f"{prefix}_{key}_atoms.xyz")
            to_xyz(xyz_path, atoms_combined, Z_combined)

    # Save all complexes in one npz file
    np.savez(f"{prefix}_all_complexes.npz", **all_complexes)
    print(f"✅ Saved {prefix}_all_complexes.npz and XYZ files for each complex.")

if __name__=="__main__":
    if len(sys.argv)<4:
        print("Usage: ./align_one_to_many_npz.py molA.npz molB.npz idxA [angles_csv] [output_prefix]")
        sys.exit(1)

    fileA = sys.argv[1]
    fileB = sys.argv[2]
    idxA = int(sys.argv[3])
    angles_csv = sys.argv[4] if len(sys.argv)>=5 else "0"
    angles = [float(a) for a in angles_csv.split(",")]
    prefix = sys.argv[5] if len(sys.argv)>=6 else "complex"

    main(fileA, fileB, idxA, angles, prefix)
