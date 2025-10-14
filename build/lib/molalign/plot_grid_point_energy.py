#!/usr/bin/env python3
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import sys

def main():
    if len(sys.argv) < 3:
        print("Usage: plot-grid-energy mol1.npz xtb_results.json")
        sys.exit(1)

    npz_file = sys.argv[1]
    json_file = sys.argv[2]

    # Load mol1 npz
    data = np.load(npz_file)
    atoms = data["atoms"]      # x, y, z, Z, index
    grids = data["grids"]      # x, y, z, Z, atom_index

    atom_coords = atoms[:, :3]
    atom_Z = atoms[:, 3].astype(int)
    grid_coords = grids[:, :3]
    grid_indices = grids[:, 4].astype(int)

    # Load JSON energy results
    with open(json_file, "r") as f:
        xtb_results = json.load(f)
    E_solute = xtb_results["solute"]["energy_Eh"]
    E_solvent = xtb_results["solvent"]["energy_Eh"]

    #print(f"Solute energy: {E_solute} Eh")
    #print(f"Solvent energy: {E_solvent} Eh")
    # Compute lowest energy per grid point
    lowest_energy = []
    for i in range(len(grid_coords)):
        gp_key = f"gp{i}"
        if gp_key in xtb_results:
            energies = [v["energy_Eh"] for v in xtb_results[gp_key].values()]
            lowest_energy.append((min(energies)-E_solute-E_solvent)*627.5)
        else:
            lowest_energy.append(np.nan)
    lowest_energy = np.array(lowest_energy)

    # Normalize energies for coloring
    norm = Normalize(vmin=np.nanmin(lowest_energy), vmax=np.nanmax(lowest_energy))
    vmin, vmax = np.nanpercentile(lowest_energy, [5, 95])
    norm = Normalize(vmin=vmin, vmax=vmax)

    # Simple color map for atom elements
    element_colors = {
        1: 'white',   # H
        6: 'gray',    # C
        7: 'blue',    # N
        8: 'red',     # O
        16: 'yellow', # S
    }
    atom_colors = [element_colors.get(Z, 'green') for Z in atom_Z]

    # Plot
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot grid points colored by lowest energy
    sc = ax.scatter(grid_coords[:,0], grid_coords[:,1], grid_coords[:,2],
                    c=lowest_energy, cmap='coolwarm', s=80, edgecolor='k', norm=norm, alpha=0.8)

    # Plot atoms as spheres colored by element
    ax.scatter(atom_coords[:,0], atom_coords[:,1], atom_coords[:,2],
               c=atom_colors, s=300, marker='o', edgecolor='k', label='Atoms')

    ax.set_xlabel("X [A]")
    ax.set_ylabel("Y [A]")
    ax.set_zlabel("Z [A]")
    ax.set_title("Mol1 grid points and atoms")

    # Add colorbar for energies
    cbar = plt.colorbar(sc, ax=ax, shrink=0.1)
    cbar.set_label("Lowest energy [Eh]")

    # Fix coordinate box (adjust if needed)
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_zlim(-10, 10)

    # Optional legend
    ax.legend()

    plt.show()

if __name__ == "__main__":
    main()

