#!/usr/bin/env python3
import re
import numpy as np
import sys

BOHR_TO_ANG = 0.52917721092
def read_atoms(filename):
    """Read atoms with coordinates and atomic numbers."""
    coords = []
    atomic_numbers = []
    pattern = re.compile(
        r'\s*([-+]?\d*\.\d+(?:[Ee][-+]?\d+)?)\s+'  # x
        r'([-+]?\d*\.\d+(?:[Ee][-+]?\d+)?)\s+'      # y
        r'([-+]?\d*\.\d+(?:[Ee][-+]?\d+)?)\s+'      # z
        r'([-+]?\d*\.\d+(?:[Ee][-+]?\d+)?)\s+'      # radius (ignored)
        r'(\d+)'                                     # atomic number
    )

    read_atoms = False
    with open(filename, 'r') as f:
        for line in f:
            if "CARTESIAN COORDINATES" in line:
                read_atoms = True
                continue
            if read_atoms:
                if line.strip().startswith('#') or not line.strip():
                    continue
                m = pattern.match(line)
                if m:
                    x, y, z, _, atm = m.groups()
                    coords.append([float(x), float(y), float(z)])
                    atomic_numbers.append(int(atm))
                else:
                    break
    coords = np.array(coords)*BOHR_TO_ANG
    atomic_numbers = np.array(atomic_numbers)
    indices = np.arange(len(coords))
    atoms_array = np.column_stack((coords, atomic_numbers, indices))
    return atoms_array

def read_gridpoints(filename, atoms_array):
    """Read gridpoints and map atomic number using atom index."""
    coords = []
    atom_indices = []
    # first 3 floats = x,y,z, skip next 6 floats, last int = atom index
    pattern = re.compile(
        r'\s*([-+]?\d*\.\d+(?:[Ee][-+]?\d+)?)\s+'  # X
        r'([-+]?\d*\.\d+(?:[Ee][-+]?\d+)?)\s+'      # Y
        r'([-+]?\d*\.\d+(?:[Ee][-+]?\d+)?)\s+'      # Z
        r'(?:[\dEe\.\-+]+\s+){6}'                   # skip 6 floats
        r'(\d+)'                                    # atom index
    )

    read_surface = False
    with open(filename, 'r') as f:
        for line in f:
            if "SURFACE POINTS" in line:
                read_surface = True
                continue
            if read_surface:
                if line.strip().startswith('#') or not line.strip():
                    continue
                m = pattern.match(line)
                if m:
                    x, y, z, idx = m.groups()
                    idx = int(idx)
                    coords.append([float(x), float(y), float(z)])
                    atom_indices.append(idx)
    coords = np.array(coords)*BOHR_TO_ANG
    atom_indices = np.array(atom_indices)
    # Map atomic number from atoms_array
    atomic_numbers = atoms_array[atom_indices, 3]
    grid_array = np.column_stack((coords, atomic_numbers, atom_indices))
    return grid_array

def save_txt(filename, atoms_array, grid_array):
    """Save human-readable TXT with atoms and grids arrays in neatly aligned columns."""
    with open(filename, 'w') as f:
        # Header for atoms
        f.write("# Atoms array:\n")
        f.write("{:^11} {:^11} {:^14} {:>12} {:>8}\n".format("x", "y", "z", "Z", "indx"))
        for row in atoms_array:
            f.write("{:12.6f} {:12.6f} {:12.6f} {:12d} {:6d}\n".format(
                row[0], row[1], row[2], int(row[3]), int(row[4])
            ))

        f.write("\n# Gridpoints array:\n")
        f.write("{:^11} {:^11} {:^14} {:>12} {:>8}\n".format("x", "y", "z", "Z", "indx"))
        for row in grid_array:
            f.write("{:12.6f} {:12.6f} {:12.6f} {:12d} {:6d}\n".format(
                row[0], row[1], row[2], int(row[3]), int(row[4])
            ))

def main():
    if len(sys.argv) != 3:
        print("Usage: ./cpcm_to_npz.py <cpcm_file> <output_prefix>")
        sys.exit(1)

    filename = sys.argv[1]
    prefix = sys.argv[2]

    atoms_array = read_atoms(filename)
    grid_array = read_gridpoints(filename, atoms_array)

    # Save NPZ
    np.savez(f"{prefix}_data.npz", atoms=atoms_array, grids=grid_array)
    print(f"✅ Saved NPZ with {atoms_array.shape[0]} atoms and {grid_array.shape[0]} gridpoints.")

    # Save TXT
    txt_file = f"{prefix}_data.txt"
    save_txt(txt_file, atoms_array, grid_array)
    print(f"✅ Saved human-readable TXT: {txt_file}")

if __name__ == "__main__":
    main()

