#!/usr/bin/env python3
"""
all_grid_tb_parallel.py  (JSON-compatible, slightly optimized)

This is intended to be used as the backend for the executable:

    align-grid-tb-parallel-2

JSON structure and key order are kept identical to the original script:

Top-level keys in insertion order:
  1. "molA_grids_xyz"
  2. "solute"
  3. "solvent"
  4. "gp0", "gp1", ..., "gp<N-1>"

Each "gpX" contains:
  "b<idxB>_rot<angle_int>" → {"energy_Eh": float, "xyz": [...]}

We only optimize internal Calculator reuse and keep everything else the same.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")  # avoid oversubscription in tblite/OpenMP

import sys
import json
import argparse
import numpy as np
import multiprocessing as mp
from tqdm import tqdm

import molalign.align_many_to_one_tb_parallel3 as align_mod

BOHR_TO_ANG = 0.52917721092
WRITE_XYZ = os.environ.get("ALIGN_TB_WRITE_XYZ", "1") != "0"

# Per-worker state (lives only inside each subprocess)
_WORKER = {}


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


def run_tblite_sp_single(coords_angstrom, atomic_numbers, method="GFN2-xTB", charge=0):
    """
    Simple TB-lite SP wrapper for solute/solvent energies (few calls).
    """
    from tblite.interface import Calculator
    import contextlib, io

    coords_angstrom = np.asarray(coords_angstrom, dtype=float)
    Z = np.asarray(atomic_numbers, dtype=int)
    coords_bohr = coords_angstrom / BOHR_TO_ANG

    calc = Calculator(method, Z, coords_bohr, charge=charge)
    calc.add("alpb-solvation", "water")

    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        res = calc.singlepoint()

    return res.get("energy")


# ---------------- per-worker energy function ---------------- #

def _worker_energy(coords_angstrom, atomic_numbers, method="GFN2-xTB", charge=0):
    """
    Energy function passed into align_mod.main_from_prepared.

    Tries to reuse a Calculator per worker:
      - If Z and charge match previous call: update coordinates and singlepoint()
      - Otherwise: rebuild Calculator once and cache it.

    If the tblite API for calc.set("coordinates", ...) is not supported,
    we fall back to rebuilding the Calculator.
    """
    from tblite.interface import Calculator
    import contextlib, io

    coords_angstrom = np.asarray(coords_angstrom, dtype=float)
    Z = np.asarray(atomic_numbers, dtype=int)
    coords_bohr = coords_angstrom / BOHR_TO_ANG

    calc = _WORKER.get("calc")
    cached_Z = _WORKER.get("Z_template", None)
    cached_charge = _WORKER.get("charge", None)

    # Try reuse if Z and charge match
    if calc is not None and cached_Z is not None:
        if cached_charge == int(charge) and np.array_equal(cached_Z, Z):
            try:
                f = io.StringIO()
                with contextlib.redirect_stdout(f):
                    calc.set("coordinates", coords_bohr)
                    res = calc.singlepoint()
                return res.get("energy")
            except Exception:
                # Fall back to rebuild below
                pass

    # Rebuild Calculator from scratch
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        calc = Calculator(method, Z, coords_bohr, charge=charge)
        calc.add("alpb-solvation", "water")
        res = calc.singlepoint()

    _WORKER["calc"] = calc
    _WORKER["Z_template"] = Z
    _WORKER["charge"] = int(charge)

    return res.get("energy")


# ---------------- worker initializer ---------------- #

def _init_worker(dataA, dataB, angles, charge):
    """
    Called once in each worker process.

    - Precompute preA/preB (atoms, grids, normals)
    - Optionally build one initial Calculator for the combined system
      (for potential reuse later).
    """
    global _WORKER
    from tblite.interface import Calculator
    import contextlib, io

    preA = align_mod.prepare_data(dataA)
    preB = align_mod.prepare_data(dataB)

    atomsA = preA["atoms"]
    Z_A = preA["Z"]
    atomsB = preB["atoms"]
    Z_B = preB["Z"]

    atoms_combined = np.vstack((atomsA, atomsB))
    Z_combined = np.hstack((Z_A, Z_B))
    coords_bohr = atoms_combined / BOHR_TO_ANG

    calc = None
    Z_template = None
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        try:
            calc = Calculator("GFN2-xTB", Z_combined, coords_bohr, charge=charge)
            calc.add("alpb-solvation", "water")
            Z_template = Z_combined
        except Exception:
            calc = None
            Z_template = None

    _WORKER = {
        "preA": preA,
        "preB": preB,
        "angles": list(angles),
        "charge": int(charge),
        "calc": calc,
        "Z_template": Z_template,
    }


# ---------------- worker job ---------------- #

def worker_one_grid(idxA):
    """
    Worker job: handle all alignments for a single solute grid point idxA.
    Returns (idxA, gp_results, err_string_or_None).
    """
    global _WORKER
    preA = _WORKER["preA"]
    preB = _WORKER["preB"]
    angles = _WORKER["angles"]
    charge = _WORKER["charge"]

    try:
        gp_results = align_mod.main_from_prepared(
            preA,
            preB,
            idxA,
            angles=angles,
            charge=charge,
            energy_fn=_worker_energy,
        )
        return idxA, gp_results, None
    except Exception as e:
        return idxA, None, repr(e)


# ---------------- main orchestration ---------------- #

def main(fileA, fileB, angles_csv, charge=0, out_json="tb_lite_results.json"):

    fileA = os.path.abspath(fileA)
    fileB = os.path.abspath(fileB)
    charge = int(charge)

    print("\n========== TB-LITE GRID SCAN ==========")
    print(f"Solute NPZ  : {fileA}")
    print(f"Solvent NPZ : {fileB}")
    print(f"Angles      : {angles_csv}")
    print(f"Charge      : {charge}")
    print("---------------------------------------")

    angles = [float(a) for a in angles_csv.split(",")]

    rawA = np.load(fileA, allow_pickle=True)
    rawB = np.load(fileB, allow_pickle=True)

    dataA = {k: rawA[k] for k in rawA.files}
    dataB = {k: rawB[k] for k in rawB.files}

    atomsA = dataA["atoms"][:, :3]
    Z_A = dataA["atoms"][:, 3].astype(int)
    atomsB = dataB["atoms"][:, :3]
    Z_B = dataB["atoms"][:, 3].astype(int)

    # IMPORTANT: insertion order of keys preserved
    results = {}

    # 1) molA_grids_xyz
    results["molA_grids_xyz"] = dataA["grids"][:, :3].tolist()

    # 2) solute
    print("\nComputing solute energy…")
    e_solute = run_tblite_sp_single(atomsA, Z_A, charge=charge)
    solute_entry = {
        "energy_Eh": float(e_solute),
        "charge": charge,
    }
    if WRITE_XYZ:
        solute_entry["xyz"] = atoms_to_xyz_list(atomsA, Z_A)
    results["solute"] = solute_entry

    # 3) solvent
    print("Computing solvent energy…")
    e_solvent = run_tblite_sp_single(atomsB, Z_B, charge=0)
    solvent_entry = {
        "energy_Eh": float(e_solvent),
        "charge": 0,
    }
    if WRITE_XYZ:
        solvent_entry["xyz"] = atoms_to_xyz_list(atomsB, Z_B)
    results["solvent"] = solvent_entry

    num_grids = dataA["grids"].shape[0]
    print(f"\nTotal solute grid points: {num_grids}")

    # Parallel settings
    default_max = int(os.environ.get("ALIGN_TB_MAX_WORKERS", 4))
    ncores = min(default_max, mp.cpu_count(), num_grids)
    print(f"Using {ncores} worker processes")

    ctx = mp.get_context("spawn")

    if num_grids <= ncores:
        chunksize = 1
    else:
        chunksize = max(1, num_grids // (ncores * 4))

    print(f"Chunk size: {chunksize}")
    print("---------------------------------------\n")

    pbar = tqdm(total=num_grids, desc="Grid scanning", smoothing=0.1)

    with ctx.Pool(
        processes=ncores,
        initializer=_init_worker,
        initargs=(dataA, dataB, angles, charge),
    ) as pool:
        # gp keys will be inserted in the order in which results arrive.
        # This matches the behaviour of imap_unordered in your original script.
        for grid_index, gp_result, err in pool.imap_unordered(
            worker_one_grid, range(num_grids), chunksize=chunksize
        ):
            if err is not None:
                print(f"[WARNING] grid {grid_index} failed: {err}", file=sys.stderr)
            else:
                # 4) gpX entries
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
    print("=======================================\n")

    return results


def cli_main():
    parser = argparse.ArgumentParser(
        description="Generate complexes and run tblite SP for all grids (parallel, JSON-compatible)."
    )
    parser.add_argument("fileA", help="NPZ file for solute")
    parser.add_argument("fileB", help="NPZ file for solvent")
    parser.add_argument("angles_csv", help="Comma-separated rotation angles, e.g. 0,90,180,270")
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--out", default="tb_lite_results.json")
    args = parser.parse_args()

    main(args.fileA, args.fileB, args.angles_csv, charge=args.charge, out_json=args.out)


if __name__ == "__main__":
    cli_main()

