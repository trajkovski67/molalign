#!/usr/bin/env python3
"""
all_grid_orca_parallel3.py  (JSON-compatible, ORCA OPI backend)

Parallel grid scan using ORCA OPI with:
    ! r2SCAN-3c CPCM(water) TightSCF

Key points:
  - Same JSON structure as all_grid_tb_parallel3.py
  - Does NOT set OPI ncores; ORCA uses env defaults
  - Uses TemporaryDirectory per worker; no files left behind
"""

import os
import sys
import json
import argparse
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from pathlib import Path
import tempfile

import molalign.align_many_to_one_orca as align_mod

WRITE_XYZ = os.environ.get("ALIGN_TB_WRITE_XYZ", "1") != "0"

_WORKER = {}
Z2SYM = align_mod.Z2SYM


def atoms_to_xyz_list(coords, atomic_numbers):
    return [
        {
            "element": str(int(Z)),
            "x": float(p[0]),
            "y": float(p[1]),
            "z": float(p[2]),
        }
        for p, Z in zip(coords, atomic_numbers)
    ]


def run_orca_sp_single(
    coords_angstrom,
    atomic_numbers,
    charge=0,
    multiplicity=1,
):
    """
    One-off ORCA OPI SP (for isolated solute/solvent).
    Same settings as grid scan: r2SCAN-3c CPCM(water) TightSCF.
    """
    from opi.core import Calculator
    from opi.input.structures.atom import Atom
    from opi.input.structures.structure import Structure

    coords = np.asarray(coords_angstrom, float)
    Z = np.asarray(atomic_numbers, int)

    atoms = []
    for (x, y, z), Zi in zip(coords, Z):
        Zi = int(Zi)
        if Zi not in Z2SYM:
            raise ValueError(f"Atomic number {Zi} not supported by Z2SYM")
        atoms.append(
            Atom(
                element=Z2SYM[Zi],
                coordinates=(float(x), float(y), float(z)),
            )
        )

    structure = Structure(
        atoms=atoms,
        charge=int(charge),
        multiplicity=int(multiplicity),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        workdir = Path(tmpdir)
        basename = "sp"

        calc = Calculator(basename=basename, working_dir=workdir)
        calc.structure = structure
        calc.input.add_arbitrary_string("!pal4 r2SCAN-3c CPCM(water) TightSCF")

        # No ncores set here

        calc.write_input()
        calc.run()

        outfile = workdir / f"{basename}.out"
        energy = None
        with open(outfile, "r") as fh:
            for line in fh:
                if "FINAL SINGLE POINT ENERGY" in line:
                    energy = float(line.split()[-1])
                    break

        if energy is None:
            raise RuntimeError(f"Energy not found in {outfile}")

        return energy


# ---------------- per-worker energy function ---------------- #

def _worker_energy(
    coords_angstrom,
    atomic_numbers,
    charge=0,
    multiplicity=1,
):
    """
    Energy function passed into align_mod.main_from_prepared.

    Each worker uses a shared TemporaryDirectory (created once in _init_worker),
    and different basenames per job.
    """
    from opi.core import Calculator
    from opi.input.structures.atom import Atom
    from opi.input.structures.structure import Structure

    coords = np.asarray(coords_angstrom, float)
    Z = np.asarray(atomic_numbers, int)

    atoms = []
    for (x, y, z), Zi in zip(coords, Z):
        Zi = int(Zi)
        if Zi not in Z2SYM:
            raise ValueError(f"Atomic number {Zi} not supported by Z2SYM")
        atoms.append(
            Atom(
                element=Z2SYM[Zi],
                coordinates=(float(x), float(y), float(z)),
            )
        )

    structure = Structure(
        atoms=atoms,
        charge=int(charge),
        multiplicity=int(multiplicity),
    )

    tmpdir_path: Path = _WORKER["tmpdir_path"]
    job_counter = _WORKER["job_counter"]
    _WORKER["job_counter"] = job_counter + 1
    basename = f"sp_{job_counter}"

    calc = Calculator(basename=basename, working_dir=tmpdir_path)
    calc.structure = structure
    calc.input.add_arbitrary_string("! r2SCAN-3c CPCM(water) TightSCF")

    calc.write_input()
    calc.run()

    outfile = tmpdir_path / f"{basename}.out"
    energy = None
    with open(outfile, "r") as fh:
        for line in fh:
            if "FINAL SINGLE POINT ENERGY" in line:
                energy = float(line.split()[-1])
                break

    if energy is None:
        raise RuntimeError(f"Energy not found in {outfile}")

    return energy


# ---------------- worker initializer ---------------- #

def _init_worker(dataA, dataB, angles, charge, multiplicity):
    """
    Called once in each worker process.
    Precomputes geometry + allocates one TemporaryDirectory per worker.
    """
    global _WORKER

    preA = align_mod.prepare_data(dataA)
    preB = align_mod.prepare_data(dataB)

    tmpdir_obj = tempfile.TemporaryDirectory()
    tmpdir_path = Path(tmpdir_obj.name)

    _WORKER = {
        "preA": preA,
        "preB": preB,
        "angles": list(angles),
        "charge": int(charge),
        "multiplicity": int(multiplicity),
        "tmpdir_obj": tmpdir_obj,
        "tmpdir_path": tmpdir_path,
        "job_counter": 0,
    }


# ---------------- worker job ---------------- #

def worker_one_grid(idxA):
    global _WORKER
    preA = _WORKER["preA"]
    preB = _WORKER["preB"]
    angles = _WORKER["angles"]
    charge = _WORKER["charge"]
    multiplicity = _WORKER["multiplicity"]

    try:
        gp_results = align_mod.main_from_prepared(
            preA,
            preB,
            idxA,
            angles=angles,
            charge=charge,
            energy_fn=_worker_energy,
            multiplicity=multiplicity,
        )
        return idxA, gp_results, None
    except Exception as e:
        return idxA, None, repr(e)


# ---------------- main orchestration ---------------- #

def main(
    fileA,
    fileB,
    angles_csv,
    charge=0,
    multiplicity=1,
    out_json="orca_opi_results.json",
):
    fileA = os.path.abspath(fileA)
    fileB = os.path.abspath(fileB)
    charge = int(charge)
    multiplicity = int(multiplicity)

    print("\n========== ORCA OPI GRID SCAN ==========")
    print(f"Solute NPZ   : {fileA}")
    print(f"Solvent NPZ  : {fileB}")
    print(f"Angles       : {angles_csv}")
    print(f"Charge       : {charge}")
    print(f"Multiplicity : {multiplicity}")
    print("----------------------------------------")

    angles = [float(a) for a in angles_csv.split(",")]

    rawA = np.load(fileA, allow_pickle=True)
    rawB = np.load(fileB, allow_pickle=True)

    dataA = {k: rawA[k] for k in rawA.files}
    dataB = {k: rawB[k] for k in rawB.files}

    atomsA = dataA["atoms"][:, :3]
    Z_A = dataA["atoms"][:, 3].astype(int)
    atomsB = dataB["atoms"][:, :3]
    Z_B = dataB["atoms"][:, 3].astype(int)

    results = {}

    # 1) molA_grids_xyz
    results["molA_grids_xyz"] = dataA["grids"][:, :3].tolist()

    # 2) solute
    print("\nComputing solute energy (ORCA OPI)…")
    e_solute = run_orca_sp_single(
        atomsA,
        Z_A,
        charge=charge,
        multiplicity=multiplicity,
    )
    solute_entry = {
        "energy_Eh": float(e_solute),
        "charge": charge,
    }
    if WRITE_XYZ:
        solute_entry["xyz"] = atoms_to_xyz_list(atomsA, Z_A)
    results["solute"] = solute_entry

    # 3) solvent
    print("Computing solvent energy (ORCA OPI)…")
    e_solvent = run_orca_sp_single(
        atomsB,
        Z_B,
        charge=0,
        multiplicity=1,
    )
    solvent_entry = {
        "energy_Eh": float(e_solvent),
        "charge": 0,
    }
    if WRITE_XYZ:
        solvent_entry["xyz"] = atoms_to_xyz_list(atomsB, Z_B)
    results["solvent"] = solvent_entry

    num_grids = dataA["grids"].shape[0]
    print(f"\nTotal solute grid points: {num_grids}")

    # Parallel worker count uses same env var as TB-lite
    default_max = int(os.environ.get("ALIGN_TB_MAX_WORKERS", 4))
    ncores_workers = min(default_max, mp.cpu_count(), num_grids)
    print(f"Using {ncores_workers} worker processes")

    ctx = mp.get_context("spawn")

    if num_grids <= ncores_workers:
        chunksize = 1
    else:
        chunksize = max(1, num_grids // (ncores_workers * 4))

    print(f"Chunk size: {chunksize}")
    print("----------------------------------------\n")

    pbar = tqdm(total=num_grids, desc="Grid scanning (ORCA)", smoothing=0.1)

    with ctx.Pool(
        processes=ncores_workers,
        initializer=_init_worker,
        initargs=(dataA, dataB, angles, charge, multiplicity),
    ) as pool:
        for grid_index, gp_result, err in pool.imap_unordered(
            worker_one_grid, range(num_grids), chunksize=chunksize
        ):
            if err is not None:
                print(f"[WARNING] grid {grid_index} failed: {err}", file=sys.stderr)
            else:
                results[f"gp{grid_index}"] = gp_result
            pbar.update(1)

    pbar.close()

    print("\nSaving results…")

    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        return obj

    with open(out_json, "w") as fh:
        json.dump(results, fh, indent=2, default=convert_numpy)

    print(f"\n*** DONE. Results saved to {out_json}\n")
    print("========================================\n")

    return results


def cli_main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate complexes and run ORCA OPI SP for all grids "
            "(parallel, JSON-compatible)."
        )
    )
    parser.add_argument("fileA", help="NPZ file for solute")
    parser.add_argument("fileB", help="NPZ file for solvent")
    parser.add_argument(
        "angles_csv",
        help="Comma-separated rotation angles, e.g. 0,90,180,270",
    )
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--multiplicity", type=int, default=1)
    parser.add_argument("--out", default="orca_opi_results.json")
    args = parser.parse_args()

    main(
        args.fileA,
        args.fileB,
        args.angles_csv,
        charge=args.charge,
        multiplicity=args.multiplicity,
        out_json=args.out,
    )


if __name__ == "__main__":
    cli_main()

