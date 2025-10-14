#!/usr/bin/env python3
"""
visualize_grid.py — simple 3D visualization of molalign .npz grid files

Usage:
    python visualize_grid.py file.npz [--downsample N]
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
import argparse

def visualize_npz(npz_file, downsample=None):
    # Load NPZ data
    data = np.load(npz_file, allow_pickle=True)
    atoms = data["atoms"]
    grids = data["grids"]

    # Extract coordinates
    atom_coords = atoms[:, :3].astype(float)
    grid_coords = grids[:, :3].astype(float)

    # Atom index for each grid point (if exists)
    atom_idx = None
    if grids.shape[1] >= 5:
        atom_idx = grids[:, 4].astype(int)

    # Downsample if needed
    if downsample is not None and grid_coords.shape[0] > downsample:
        indices = np.random.choice(grid_coords.shape[0], downsample, replace=False)
        grid_coords = grid_coords[indices]
        if atom_idx is not None:
            atom_idx = atom_idx[indices]

    # Setup figure
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(f"Grid visualization: {npz_file}")

    # Scatter grid points
    if atom_idx is not None:
        scatter = ax.scatter(
            grid_coords[:, 0],
            grid_coords[:, 1],
            grid_coords[:, 2],
            c=atom_idx,
            cmap="rainbow",
            s=5,
            alpha=0.6,
        )
        fig.colorbar(scatter, ax=ax, label="Atom index")
    else:
        ax.scatter(
            grid_coords[:, 0],
            grid_coords[:, 1],
            grid_coords[:, 2],
            c="blue",
            s=5,
            alpha=0.6,
            label="Grid points",
        )

    # Plot atoms as larger black points
    ax.scatter(
        atom_coords[:, 0],
        atom_coords[:, 1],
        atom_coords[:, 2],
        c="black",
        s=60,
        depthshade=True,
        label="Atoms",
    )

    ax.set_xlabel("x (Å)")
    ax.set_ylabel("y (Å)")
    ax.set_zlabel("z (Å)")
    ax.legend()
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Visualize molalign .npz grid points.")
    parser.add_argument("npz_file", help="Path to .npz file")
    parser.add_argument("--downsample", type=int, default=None,
                        help="Randomly show only N grid points (for speed)")
    args = parser.parse_args()
    visualize_npz(args.npz_file, args.downsample)

if __name__ == "__main__":
    main()

