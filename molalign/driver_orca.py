#!/usr/bin/env python3
"""
driver3_orca.py

Molalign extended driver using ORCA OPI backend instead of TB-lite.

Pipeline:
  1. Read solute XYZ
  2. For each vdW scaling factor:
        - Run moist iswig -> cavity.csv
        - Convert cavity.csv -> solute NPZ/TXT
  3. Convert solvent CPCM -> NPZ via cpcm-reader
  4. For each scale:
        - Run align-grid-orca-parallel-3 (parallel ORCA OPI grid scan)
        - JSON structure identical to TB-lite version
"""

import os

# Same single-thread forcing as original driver3.py
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import csv
import subprocess
import sys
import numpy as np
from tqdm import tqdm

BOHR_TO_ANG = 0.52917721092

PERIODIC_TABLE = {
    "H": 1, "He": 2,
    "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9,
    "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,
    "S": 16, "Cl": 17, "Ar": 18, "Br": 35,
}


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

    coords = np.array(coords, dtype=float)
    Zs = np.array(Zs, dtype=int)
    idx = np.arange(len(coords), dtype=int)

    return np.column_stack((coords, Zs, idx))


def read_moist_csv(filename, atoms_array):
    with open(filename, "r") as f:
        header = next(csv.reader(f))

    header = [h.strip().lower() for h in header]
    ix = header.index("x")
    iy = header.index("y")
    iz = header.index("z")
    iowner = header.index("owner")

    coords_bohr = []
    atomic_numbers = []
    atom_indices = []

    with open(filename, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            x = float(row[ix])
            y = float(row[iy])
            z = float(row[iz])
            owner = int(row[iowner])
            idx = owner - 1
            if idx < 0 or idx >= atoms_array.shape[0]:
                raise ValueError(f"Owner index {owner} out of range")
            Z = int(atoms_array[idx, 3])
            coords_bohr.append([x, y, z])
            atomic_numbers.append(Z)
            atom_indices.append(idx)

    coords_bohr = np.array(coords_bohr, float)
    coords_ang = coords_bohr * BOHR_TO_ANG

    return np.column_stack(
        (coords_ang, np.array(atomic_numbers, int), np.array(atom_indices, int))
    )


def save_txt(filename, atoms_array, grid_array):
    with open(filename, "w") as f:
        f.write("# Atoms array:\n")
        f.write("{:^11} {:^11} {:^14} {:>12} {:>8}\n".format(
            "x", "y", "z", "Z", "indx"
        ))
        for row in atoms_array:
            f.write("{:12.6f} {:12.6f} {:12.6f} {:12d} {:6d}\n".format(
                row[0], row[1], row[2], int(row[3]), int(row[4])
            ))
        f.write("\n# Gridpoints array:\n")
        f.write("{:^11} {:^11} {:^14} {:>12} {:>8}\n".format(
            "x", "y", "z", "Z", "indx"
        ))
        for row in grid_array:
            f.write("{:12.6f} {:12.6f} {:12.6f} {:12d} {:6d}\n".format(
                row[0], row[1], row[2], int(row[3]), int(row[4])
            ))


def run_moist(xyz_file, nleb, scale, workdir, solute_name, tag):
    cmd = [
        "moist", "iswig", xyz_file,
        "--nleb", str(nleb),
        "--rad-multiplier", str(scale),
    ]
    subprocess.run(cmd, check=True, cwd=workdir)

    old = os.path.join(workdir, "cavity.csv")
    if not os.path.exists(old):
        raise RuntimeError("cavity.csv missing after moist")

    new = os.path.join(workdir, f"{solute_name}_scale{tag}_cavity.csv")
    os.replace(old, new)
    return new


def sanitize_scale(scale):
    return str(scale).replace(".", "p").replace("-", "n")


def parse_args():
    p = argparse.ArgumentParser(
        description="Solute XYZ + moist (scaled vdW) + solvent CPCM + ORCA-OPI molalign."
    )
    p.add_argument("solute_xyz")
    p.add_argument("solvent_cpcm")
    p.add_argument("--out", default="OUT")
    p.add_argument("--angles", default="0,90,180,270")
    p.add_argument("--charge", type=int, default=0)
    p.add_argument("--multiplicity", type=int, default=1)
    p.add_argument("--nleb", type=int, default=26)
    p.add_argument("--scales", type=float, nargs="+", default=[1.0])
    p.add_argument("--ncores", type=int, default=None,
                   help="Max Python workers for parallel grid scan "
                        "(sets ALIGN_TB_MAX_WORKERS).")
    return p.parse_args()


def main():
    args = parse_args()

    solute_xyz = os.path.abspath(args.solute_xyz)
    solvent_cpcm = os.path.abspath(args.solvent_cpcm)
    outdir = os.path.abspath(args.out)
    os.makedirs(outdir, exist_ok=True)

    solute_name = os.path.splitext(os.path.basename(solute_xyz))[0]
    solvent_name = os.path.splitext(os.path.basename(solvent_cpcm))[0]

    if args.ncores is not None and args.ncores > 0:
        os.environ["ALIGN_TB_MAX_WORKERS"] = str(args.ncores)

    print("\n========== MOIST + MOLALIGN + ORCA OPI (SCALED vdW RADII) ==========")
    print(f"Solute XYZ     : {solute_xyz}")
    print(f"Solvent CPCM   : {solvent_cpcm}")
    print(f"Output dir     : {outdir}")
    print(f"Angles         : {args.angles}")
    print(f"Charge         : {args.charge}")
    print(f"Multiplicity   : {args.multiplicity}")
    print(f"Scales         : {args.scales}")
    print(f"Max workers    : {os.environ.get('ALIGN_TB_MAX_WORKERS','default')}")
    print("--------------------------------------------------------------------")

    # 1. CPCM → NPZ
    print(f"\n*** Running: cpcm-reader {solvent_cpcm} {solvent_name}")
    subprocess.run(["cpcm-reader", solvent_cpcm, solvent_name], check=True)

    solvent_npz_local = f"{solvent_name}_data.npz"
    if not os.path.exists(solvent_npz_local):
        raise RuntimeError(f"Expected {solvent_npz_local} not found")

    solvent_npz = os.path.join(outdir, solvent_npz_local)
    os.replace(solvent_npz_local, solvent_npz)
    print(f"    → Solvent NPZ: {solvent_npz}")

    # 2. solute XYZ
    print("\nReading solute atoms...")
    atoms_solute = read_xyz_atoms(solute_xyz)
    print(f"    → {atoms_solute.shape[0]} atoms")

    # 3. scaled solute NPZ/TXT
    variant_files = []

    print("\nGenerating solute NPZ/TXT for all scales:")
    for scale in tqdm(args.scales, desc="Scales"):
        tag = sanitize_scale(scale)
        print(f"\n*** Scale {scale:.3f} (tag={tag})")

        csv_path = run_moist(
            solute_xyz,
            args.nleb,
            scale,
            outdir,
            solute_name,
            tag,
        )
        print(f"    cavity CSV → {csv_path}")

        grid_array = read_moist_csv(csv_path, atoms_solute)
        print(f"    Parsed {grid_array.shape[0]} grid points")

        solute_npz = os.path.join(outdir, f"{solute_name}_data_scale{tag}.npz")
        np.savez(solute_npz, atoms=atoms_solute, grids=grid_array)
        print(f"    Solute NPZ → {solute_npz}")

        solute_txt = os.path.join(outdir, f"{solute_name}_data_scale{tag}.txt")
        save_txt(solute_txt, atoms_solute, grid_array)
        print(f"    Solute TXT → {solute_txt}")

        out_json = os.path.join(outdir, f"orca_opi_results_scale{tag}.json")
        variant_files.append((scale, solute_npz, out_json))

    # 4. run ORCA-parallel molalign
    print("\nRunning align-grid-orca-parallel-3 for all scales:\n")
    for scale, solute_npz, out_json in tqdm(variant_files, desc="ORCA-OPI"):
        print(f"\n*** Scale {scale:.3f}")
        print(f"    Solute NPZ : {solute_npz}")
        print(f"    Out JSON   : {out_json}")

        subprocess.run([
            "align-grid-orca",
            solute_npz,
            solvent_npz,
            args.angles,
            "--charge", str(args.charge),
            "--multiplicity", str(args.multiplicity),
            "--out", out_json,
        ], check=True)

    print("\n*** COMPLETE. Output JSON files:")
    for scale, _, out_json in variant_files:
        print(f"    scale {scale}: {out_json}")


if __name__ == "__main__":
    main()

