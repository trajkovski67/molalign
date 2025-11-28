#!/usr/bin/env python3
"""
molalign_scaled.py

Solute: XYZ + moist (cavity.csv, multiple vdW scales)
Solvent: CPCM + cpcm-reader
For each scale: build NPZ/TXT and run align-grid-tb-parallel-2.
"""

import argparse
import csv
import os
import subprocess
import sys
import numpy as np
from tqdm import tqdm

BOHR_TO_ANG = 0.52917721092

# ---------------- Periodic table (enough for now) ---------------- #
PERIODIC_TABLE = {
    "H": 1, "He": 2,
    "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9,
    "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17,
    "Ar": 18,
}


# ---------------- XYZ → atoms array ---------------- #
def read_xyz_atoms(filename):
    with open(filename, "r") as f:
        lines = f.readlines()

    if len(lines) < 3:
        raise ValueError(f"XYZ file {filename} too short")

    try:
        n_atoms = int(lines[0].strip())
    except ValueError:
        raise ValueError(f"First line of {filename} must be atom count")

    atom_lines = lines[2:2 + n_atoms]
    coords = []
    Zs = []

    for line in atom_lines:
        parts = line.split()
        if len(parts) < 4:
            continue
        sym = parts[0]
        if sym not in PERIODIC_TABLE:
            raise ValueError(f"Unknown element symbol '{sym}' in {filename}")
        x, y, z = map(float, parts[1:4])
        coords.append([x, y, z])
        Zs.append(PERIODIC_TABLE[sym])

    coords = np.array(coords, dtype=float)          # Å
    Zs = np.array(Zs, dtype=int)
    idx = np.arange(len(coords), dtype=int)

    atoms_array = np.column_stack((coords, Zs, idx))
    return atoms_array


# ---------------- moist cavity.csv → grids array ---------------- #
def read_moist_csv(filename, atoms_array):
    """
    cavity.csv format:
    ngrid,numbering,x,y,z,owner,radius,area,w_leb,f

    x,y,z and radius are in BOHR → converted to Å.
    owner is 1-based atom index.
    """
    coords_bohr = []
    atom_indices = []
    atomic_numbers = []

    with open(filename, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        header = [h.strip().lower() for h in header]

        try:
            ix = header.index("x")
            iy = header.index("y")
            iz = header.index("z")
            iowner = header.index("owner")
        except ValueError:
            raise ValueError("cavity.csv missing x,y,z,owner columns")

        for row in reader:
            if len(row) <= iowner:
                continue
            try:
                x = float(row[ix])
                y = float(row[iy])
                z = float(row[iz])
                owner = int(row[iowner])  # 1-based
            except ValueError:
                continue

            atom_idx = owner - 1
            if atom_idx < 0 or atom_idx >= atoms_array.shape[0]:
                raise ValueError(
                    f"Owner index {owner} out of range for {atoms_array.shape[0]} atoms"
                )

            Z = int(atoms_array[atom_idx, 3])
            coords_bohr.append([x, y, z])
            atomic_numbers.append(Z)
            atom_indices.append(atom_idx)

    coords_bohr = np.array(coords_bohr, dtype=float)
    coords_ang = coords_bohr * BOHR_TO_ANG  # convert to Å
    atomic_numbers = np.array(atomic_numbers, dtype=int)
    atom_indices = np.array(atom_indices, dtype=int)

    grid_array = np.column_stack((coords_ang, atomic_numbers, atom_indices))
    return grid_array


# ---------------- TXT writer (same format as cpcm-reader) ---------------- #
def save_txt(filename, atoms_array, grid_array):
    with open(filename, "w") as f:
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


# ---------------- Run moist (writes cavity.csv) ---------------- #
def run_moist(xyz_file_abs, nleb, scale, workdir, solute_name, scale_tag):
    """
    Run moist iswig in 'workdir' so cavity.csv is created there.
    Then rename cavity.csv → <solute_name>_scale<scale_tag>_cavity.csv
    and return the new path.
    """
    cmd = [
        "moist", "iswig", xyz_file_abs,
        "--nleb", str(nleb),
        "--rad-multiplier", str(scale),
    ]

    try:
        subprocess.run(cmd, check=True, cwd=workdir)
    except FileNotFoundError:
        print("ERROR: moist not found in PATH", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"ERROR running moist for scale {scale}: {e}", file=sys.stderr)
        sys.exit(1)

    cav_path = os.path.join(workdir, "cavity.csv")
    if not os.path.exists(cav_path):
        print("ERROR: cavity.csv not created by moist!", file=sys.stderr)
        sys.exit(1)

    new_name = os.path.join(
        workdir, f"{solute_name}_scale{scale_tag}_cavity.csv"
    )
    os.replace(cav_path, new_name)
    return new_name


def sanitize_scale(value):
    """Convert float scale to a safe string for filenames."""
    s = str(value)
    s = s.replace(".", "p").replace("-", "n")
    return s


# ---------------- CLI ---------------- #
def parse_args():
    p = argparse.ArgumentParser(
        description="Molalign-like pipeline with moist scaled vdW radii."
    )
    p.add_argument("solute_xyz", help="Solute XYZ file")
    p.add_argument("solvent_cpcm", help="Solvent CPCM file")
    p.add_argument(
        "--out", default="OUT",
        help="Output directory (default: OUT)"
    )
    p.add_argument(
        "--angles", default="0,90,180,270",
        help="Comma-separated rotation angles string"
    )
    p.add_argument(
        "--charge", type=int, default=0,
        help="Total charge for TB-lite calculation"
    )
    p.add_argument(
        "--nleb", type=int, default=26,
        help="Lebedev grid size for moist"
    )
    p.add_argument(
        "--scales", type=float, nargs="+",
        default=[1.0],
        help="vdW scaling factors (e.g. --scales 0.8 0.9 1.0 1.1 1.2)"
    )
    return p.parse_args()


# ---------------- Main workflow ---------------- #
def main():
    args = parse_args()

    solute_xyz = os.path.abspath(args.solute_xyz)
    solvent_cpcm = os.path.abspath(args.solvent_cpcm)
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    solute_name = os.path.splitext(os.path.basename(solute_xyz))[0]
    solvent_name = os.path.splitext(os.path.basename(solvent_cpcm))[0]

    print("\n========== MOIST + MOLALIGN + TBLITE (SCALED vdW RADII) ==========")
    print(f"Solute XYZ   : {solute_xyz}")
    print(f"Solvent CPCM : {solvent_cpcm}")
    print(f"Output dir   : {out_dir}")
    print(f"Angles       : {args.angles}")
    print(f"Charge       : {args.charge}")
    print(f"nleb         : {args.nleb}")
    print(f"Scales       : {args.scales}")
    print("-------------------------------------------------------")

    # ----- Step 1: Solvent CPCM → NPZ (unchanged) ----- #
    print(f"\n*** Running: cpcm-reader {solvent_cpcm} {solvent_name}")
    try:
        subprocess.run(["cpcm-reader", solvent_cpcm, solvent_name], check=True)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: cpcm-reader failed for {solvent_cpcm}: {e}", file=sys.stderr)
        sys.exit(1)

    solvent_npz_local = f"{solvent_name}_data.npz"
    if not os.path.exists(solvent_npz_local):
        print(f"ERROR: Missing expected {solvent_npz_local}", file=sys.stderr)
        sys.exit(1)

    solvent_npz = os.path.join(out_dir, solvent_npz_local)
    os.replace(solvent_npz_local, solvent_npz)
    print(f"    → Solvent NPZ moved to: {solvent_npz}")

    # ----- Step 2: Read solute atoms from XYZ ----- #
    print("\nReading solute atoms from XYZ…")
    atoms_solute = read_xyz_atoms(solute_xyz)
    print(f"    → Found {atoms_solute.shape[0]} atoms.")

    # ----- Step 3: For each scale, run moist + build NPZ/TXT ----- #
    variant_files = []  # (scale, solute_npz, out_json)

    print("\nGenerating solute NPZ/TXT variants for all scales:\n")
    for scale in tqdm(args.scales, desc="Scales", unit="scale"):
        scale_tag = sanitize_scale(scale)
        print(f"\n*** Scale {scale:.3f}")

        # 3a. Run moist in out_dir, rename cavity.csv
        csv_path = run_moist(solute_xyz, args.nleb, scale,
                             workdir=out_dir,
                             solute_name=solute_name,
                             scale_tag=scale_tag)
        print(f"    → cavity CSV: {csv_path}")

        # 3b. Parse cavity CSV into grids
        grid_array = read_moist_csv(csv_path, atoms_solute)
        print(f"    → Parsed {grid_array.shape[0]} grid points.")

        # 3c. Save NPZ
        solute_npz = os.path.join(out_dir,
                                  f"{solute_name}_data_scale{scale_tag}.npz")
        np.savez(solute_npz, atoms=atoms_solute, grids=grid_array)
        print(f"    → Saved NPZ: {solute_npz}")

        # 3d. Save TXT
        solute_txt = os.path.join(out_dir,
                                  f"{solute_name}_data_scale{scale_tag}.txt")
        save_txt(solute_txt, atoms_solute, grid_array)
        print(f"    → Saved TXT: {solute_txt}")

        # 3e. TB-lite result JSON path
        out_json = os.path.join(out_dir,
                                f"tb_lite_results_scale{scale_tag}.json")
        variant_files.append((scale, solute_npz, out_json))

    # ----- Step 4: Run align-grid-tb-parallel-2 for each scale ----- #
    print("\nRunning align-grid-tb-parallel-2 for all scales:\n")
    for scale, npz_file, out_json in tqdm(variant_files, desc="TB-lite", unit="run"):
        print(f"\n*** Scale {scale:.3f}")
        print(f"    NPZ : {npz_file}")
        print(f"    OUT : {out_json}")
        try:
            subprocess.run([
                "align-grid-tb-parallel-2", npz_file, solvent_npz,
                args.angles, "--charge", str(args.charge),
                "--out", out_json
            ], check=True)
        except subprocess.CalledProcessError as e:
            print(f"ERROR: align-grid-tb-parallel-2 failed for {npz_file}: {e}", file=sys.stderr)

    print("\n*** All computations complete. Results saved in:")
    for scale, _, out_json in variant_files:
        print(f"    scale {scale:.3f}: {out_json}")


if __name__ == "__main__":
    main()

