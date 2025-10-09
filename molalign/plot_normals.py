#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def load_npz_file(npz_file):
    """Load .npz file and compute normals from atom-grid mapping."""
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
    normals /= np.where(norms > 0, norms, 1.0)

    return atom_coords, atom_numbers, atom_indices, grid_coords, grid_numbers, grid_indices, normals


def main():
    if len(sys.argv) < 2:
        print("Usage: ./plot_normals.py file.npz [scale]")
        sys.exit(1)

    npz_file = sys.argv[1]
    scale = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0

    atom_coords, atom_numbers, _, grid_coords, grid_numbers, _, normals = load_npz_file(npz_file)

    # Element color map
    colors = {
        1: "white",  # H
        6: "gray",   # C
        7: "blue",   # N
        8: "red",    # O
        16: "yellow" # S
    }
    atom_colors = [colors.get(Z, "green") for Z in atom_numbers]

    # 3D Plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(atom_coords[:, 0], atom_coords[:, 1], atom_coords[:, 2],
               c=atom_colors, s=200, edgecolor="k", label="Atoms")

    ax.scatter(grid_coords[:, 0], grid_coords[:, 1], grid_coords[:, 2],
               c="cyan", s=30, alpha=0.5, label="Grid points")

    ax.quiver(grid_coords[:, 0], grid_coords[:, 1], grid_coords[:, 2],
              normals[:, 0], normals[:, 1], normals[:, 2],
              length=scale, color="purple", alpha=0.6, normalize=True)

    ax.set_xlabel("X [Å]")
    ax.set_ylabel("Y [Å]")
    ax.set_zlabel("Z [Å]")
    ax.set_title(f"Atoms and computed normals\n{os.path.basename(npz_file)}")
    ax.legend()

    plt.show()


if __name__ == "__main__":
    main()

