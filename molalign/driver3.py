#!/usr/bin/env python3
"""
molalign-extended-parallel-3 (driver3.py)

Pipeline:
  1. Read solute XYZ
  2. For each vdW scaling factor:
        - Run moist iswig -> cavity.csv (in OUT dir)
        - Convert cavity.csv -> solute NPZ/TXT (atoms + grids in Å)
  3. Convert solvent CPCM -> NPZ via cpcm-reader
  4. For each scale:
        - Run align-grid-tb-parallel-2 (which scans all grids in parallel)
        - JSON output is identical in structure to the original code.

New in this version:
  - Hard limits OpenMP thread counts (OMP/MKL/BLAS/etc) to 1
  - Adds a --ncores option, which sets ALIGN_TB_MAX_WORKERS
    so align-grid-tb-parallel-2 does NOT use more than that many workers.
"""

import os

# -------------------------------------------------------------------
# FORCE all heavy libraries to single-thread BEFORE importing numpy/tblite
# -------------------------------------------------------------------
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

# -------------------------------------------------------------------
# Periodic table (minimal but sufficient)
# -------------------------------------------------------------------
PERIODIC_TABLE = {
    "H": 1, "He": 2,
    "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9,
    "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,
    "S": 16, "Cl": 17, "Ar": 18,"Br":35,
}


# -------------------------------------------------------------------
# XYZ → atoms array
# -------------------------------------------------------------------
def read_xyz_atoms(filename):
    """
    Read atoms from an XYZ file.

    Returns an array with columns:
        [x, y, z, Z, idx]
    where idx is 0-based.
    """
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
    idx = np.arange(len(coords), dtype=int)  # ← IMPORTANT: dtype=int, not 'int' as stop

    atoms_array = np.column_stack((coords, Zs, idx))
    return atoms_array


# -------------------------------------------------------------------
# moist cavity.csv → grids array
# -------------------------------------------------------------------
def read_moist_csv(filename, atoms_array):
    """
    Read moist cavity.csv with header:
        ngrid,numbering,x,y,z,owner,radius,area,w_leb,f

    x, y, z, radius are in BOHR → converted to Å.
    owner is a 1-based atom index.
    Returns grid_array: [x, y, z, Z, atom_idx]
    """
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
        raise ValueError("cavity.csv missing x, y, z, owner columns")

    coords_bohr = []
    atomic_numbers = []
    atom_indices = []

    with open(filename, "r") as f:
        reader = csv.reader(f)
        next(reader)  # skip header again

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

            idx = owner - 1  # convert to 0-based
            if idx < 0 or idx >= atoms_array.shape[0]:
                raise ValueError(
                    f"Owner index {owner} out of range for {atoms_array.shape[0]} atoms"
                )

            Z = int(atoms_array[idx, 3])
            coords_bohr.append([x, y, z])
            atomic_numbers.append(Z)
            atom_indices.append(idx)

    coords_bohr = np.array(coords_bohr, dtype=float)
    coords_ang = coords_bohr * BOHR_TO_ANG

    atomic_numbers = np.array(atomic_numbers, dtype=int)
    atom_indices = np.array(atom_indices, dtype=int)

    grid_array = np.column_stack((coords_ang, atomic_numbers, atom_indices))
    return grid_array


# -------------------------------------------------------------------
# TXT writer (same format as cpcm-reader output)
# -------------------------------------------------------------------
def save_txt(filename, atoms_array, grid_array):
    with open(filename, "w") as f:
        f.write("# Atoms array:\n")
        f.write("{:^11} {:^11} {:^14} {:>12} {:>8}\n".format(
            "x", "y", "z", "Z", "indx"))
        for row in atoms_array:
            f.write("{:12.6f} {:12.6f} {:12.6f} {:12d} {:6d}\n".format(
                row[0], row[1], row[2], int(row[3]), int(row[4])
            ))

        f.write("\n# Gridpoints array:\n")
        f.write("{:^11} {:^11} {:^14} {:>12} {:>8}\n".format(
            "x", "y", "z", "Z", "indx"))
        for row in grid_array:
            f.write("{:12.6f} {:12.6f} {:12.6f} {:12d} {:6d}\n".format(
                row[0], row[1], row[2], int(row[3]), int(row[4])
            ))


# -------------------------------------------------------------------
# Run moist iswig
# -------------------------------------------------------------------
def run_moist(xyz_file_abs, nleb, scale, workdir, solute_name, scale_tag):
    """
    Run moist iswig in workdir, which writes cavity.csv there.
    Then rename cavity.csv → <solute_name>_scale<scale_tag}_cavity.csv
    and return that path.
    """
    cmd = [
        "moist", "iswig", xyz_file_abs,
        "--nleb", str(nleb),
        "--rad-multiplier", str(scale),
    ]

    try:
        subprocess.run(cmd, check=True, cwd=workdir)
    except FileNotFoundError:
        print("ERROR: 'moist' not found in PATH", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: moist failed for scale {scale}: {e}", file=sys.stderr)
        sys.exit(1)

    cav_old = os.path.join(workdir, "cavity.csv")
    if not os.path.exists(cav_old):
        print("ERROR: cavity.csv not generated by moist", file=sys.stderr)
        sys.exit(1)

    cav_new = os.path.join(
        workdir, f"{solute_name}_scale{scale_tag}_cavity.csv"
    )
    os.replace(cav_old, cav_new)
    return cav_new


def sanitize_scale(scale):
    """
    Turn a float like 1.0, 1.2, 0.8 into a safe tag:
    1.0  -> 1p0
    1.2  -> 1p2
    0.8  -> 0p8
    """
    return str(scale).replace(".", "p").replace("-", "n")


# -------------------------------------------------------------------
# CLI + main
# -------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Solute XYZ + moist (scaled vdW) + solvent CPCM + alignment."
    )
    p.add_argument("solute_xyz", help="Solute XYZ file")
    p.add_argument("solvent_cpcm", help="Solvent CPCM file")
    p.add_argument("--out", default="OUT", help="Output directory (default: OUT)")
    p.add_argument("--angles", default="0,90,180,270",
                   help="Comma-separated rotation angles (default: 0,90,180,270)")
    p.add_argument("--charge", type=int, default=0,
                   help="Total charge for TB-lite calculation (default: 0)")
    p.add_argument("--nleb", type=int, default=26,
                   help="Lebedev grid size for moist (default: 26)")
    p.add_argument("--scales", type=float, nargs="+",
                   default=[1.0],
                   help="vdW scaling factors, e.g. --scales 0.8 1.0 1.2")
    p.add_argument("--ncores", type=int, default=None,
                   help="Max number of Python workers for align-grid-tb-parallel-2 "
                        "(sets ALIGN_TB_MAX_WORKERS). If omitted, the align "
                        "script's own default (4) is used.")
    return p.parse_args()


def main():
    args = parse_args()

    solute_xyz = os.path.abspath(args.solute_xyz)
    solvent_cpcm = os.path.abspath(args.solvent_cpcm)
    outdir = os.path.abspath(args.out)
    os.makedirs(outdir, exist_ok=True)

    solute_name = os.path.splitext(os.path.basename(solute_xyz))[0]
    solvent_name = os.path.splitext(os.path.basename(solvent_cpcm))[0]

    # If user requested explicit core limit, propagate to align-grid driver
    if args.ncores is not None and args.ncores > 0:
        os.environ["ALIGN_TB_MAX_WORKERS"] = str(args.ncores)

    print("\n========== MOIST + MOLALIGN + TBLITE (SCALED vdW RADII) ==========")
    print(f"Solute XYZ   : {solute_xyz}")
    print(f"Solvent CPCM : {solvent_cpcm}")
    print(f"Output dir   : {outdir}")
    print(f"Angles       : {args.angles}")
    print(f"Charge       : {args.charge}")
    print(f"nleb         : {args.nleb}")
    print(f"Scales       : {args.scales}")
    if args.ncores is not None:
        print(f"Max workers  : {args.ncores} (ALIGN_TB_MAX_WORKERS)")
    else:
        print("Max workers  : (align-grid default, typically 4)")
    print("------------------------------------------------------------------")

    # ---------------- 1. Solvent CPCM → NPZ ----------------
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

    solvent_npz = os.path.join(outdir, solvent_npz_local)
    os.replace(solvent_npz_local, solvent_npz)
    print(f"    → Solvent NPZ moved to: {solvent_npz}")

    # ---------------- 2. Read solute XYZ ----------------
    print("\nReading solute atoms from XYZ…")
    atoms_solute = read_xyz_atoms(solute_xyz)
    print(f"    → Found {atoms_solute.shape[0]} atoms.")

    # ---------------- 3. For each scale: moist + NPZ/TXT ----------------
    variant_files = []  # (scale, solute_npz, out_json)

    print("\nGenerating solute NPZ/TXT variants for all scales:\n")
    for scale in tqdm(args.scales, desc="Scales", unit="scale"):
        tag = sanitize_scale(scale)
        print(f"\n*** Scale {scale:.3f} (tag: {tag})")

        csv_path = run_moist(
            solute_xyz,
            args.nleb,
            scale,
            workdir=outdir,
            solute_name=solute_name,
            scale_tag=tag,
        )
        print(f"    → cavity CSV: {csv_path}")

        grid_array = read_moist_csv(csv_path, atoms_solute)
        print(f"    → Parsed {grid_array.shape[0]} grid points.")

        solute_npz = os.path.join(outdir, f"{solute_name}_data_scale{tag}.npz")
        np.savez(solute_npz, atoms=atoms_solute, grids=grid_array)
        print(f"    → Saved NPZ: {solute_npz}")

        solute_txt = os.path.join(outdir, f"{solute_name}_data_scale{tag}.txt")
        save_txt(solute_txt, atoms_solute, grid_array)
        print(f"    → Saved TXT: {solute_txt}")

        out_json = os.path.join(outdir, f"tb_lite_results_scale{tag}.json")
        variant_files.append((scale, solute_npz, out_json))

    # ---------------- 4. Run align-grid-tb-parallel-2 for each scale ----------------
    print("\nRunning align-grid-tb-parallel-2 for all scales:\n")
    for scale, npz_file, out_json in tqdm(variant_files, desc="TB-lite", unit="run"):
        print(f"\n*** Scale {scale:.3f}")
        print(f"    NPZ : {npz_file}")
        print(f"    OUT : {out_json}")

        try:
            subprocess.run([
                "align-grid-tb-parallel-3",
                npz_file,
                solvent_npz,
                args.angles,
                "--charge", str(args.charge),
                "--out", out_json,
            ], check=True)
        except subprocess.CalledProcessError as e:
            print(f"ERROR: align-grid-tb-parallel-2 failed for {npz_file}: {e}",
                  file=sys.stderr)

    print("\n*** All computations complete. Results saved in:")
    for scale, _, out_json in variant_files:
        print(f"    scale {scale:.3f}: {out_json}")


if __name__ == "__main__":
    main()

