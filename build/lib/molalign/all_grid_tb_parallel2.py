#!/usr/bin/env python3
"""
all_grid_tb.py  (high-performance parallel version)

Loop over all grid points of molA, align molB, run TB-lite SP,
and save all results in a single JSON file.

Key improvements:
- Preloads NPZ data once (no repeated disk I/O).
- Uses a spawn-based multiprocessing context (robust with console entrypoints).
- Uses a worker initializer to send dataA/dataB/angles/charge to workers only once.
- Sets OMP_NUM_THREADS=1 to avoid OpenMP oversubscription inside tblite.
- Precomputes normals and geometry once per worker (via prepare_data).
- Optional skipping of xyz output via ALIGN_TB_WRITE_XYZ.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")  # avoid oversubscription in tblite / OpenMP

import sys
import json
import argparse
import numpy as np
import multiprocessing as mp

import molalign.align_many_to_one_tb_parallel2 as align_mod


BOHR_TO_ANG = 0.52917721092

WRITE_XYZ = os.environ.get("ALIGN_TB_WRITE_XYZ", "1") != "0"


# --------------------------- utilities ---------------------------

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


def run_tblite_sp(coords_angstrom, atomic_numbers, method="GFN2-xTB", charge=0):
    """
    Run a single-point calculation using tblite.
    """
    from tblite.interface import Calculator
    import contextlib, io

    coords_bohr = np.array(coords_angstrom) / BOHR_TO_ANG
    calc = Calculator(method, np.array(atomic_numbers), coords_bohr, charge=charge)
    calc.add("alpb-solvation", "water")

    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        res = calc.singlepoint()

    return res.get("energy")


# --------------------------- worker globals ---------------------------

_WORKER_DATA = {}


def _init_worker(dataA, dataB, angles, charge):
    """
    Initializer for worker processes.

    Each worker gets its own prepared data:
    - preA, preB: dicts from align_mod.prepare_data
    - angles (list of floats)
    - charge (int)
    """
    global _WORKER_DATA
    preA = align_mod.prepare_data(dataA)
    preB = align_mod.prepare_data(dataB)
    _WORKER_DATA = {
        "preA": preA,
        "preB": preB,
        "angles": angles,
        "charge": charge,
    }


def worker_one_grid(idxA):
    """
    Worker function that computes all complexes at a single grid index of molA.
    """
    global _WORKER_DATA
    preA = _WORKER_DATA["preA"]
    preB = _WORKER_DATA["preB"]
    angles = _WORKER_DATA["angles"]
    charge = _WORKER_DATA["charge"]

    try:
        gp_results = align_mod.main_from_prepared(preA, preB, idxA, angles=angles, charge=charge)
        return (idxA, gp_results, None)
    except Exception as e:
        return (idxA, None, str(e))


# --------------------------- main logic ---------------------------

def main(fileA, fileB, angles_csv, charge=0, out_json="tb_lite_results.json"):

    fileA = os.path.abspath(fileA)
    fileB = os.path.abspath(fileB)
    angles = [float(a) for a in angles_csv.split(",")]

    # Preload NPZ data ONCE
    rawA = np.load(fileA, allow_pickle=True)
    rawB = np.load(fileB, allow_pickle=True)

    # convert to plain dicts (pickle-friendly, independent of npz object)
    dataA = {k: rawA[k] for k in rawA.files}
    dataB = {k: rawB[k] for k in rawB.files}

    atomsA = dataA["atoms"][:, :3]
    Z_A = dataA["atoms"][:, 3].astype(int)

    atomsB = dataB["atoms"][:, :3]
    Z_B = dataB["atoms"][:, 3].astype(int)

    results = {
        "molA_grids_xyz": dataA["grids"][:, :3].tolist()
    }

    # --- Solute and solvent SP ---
    e_solute = run_tblite_sp(atomsA, Z_A, charge=charge)
    solute_entry = {
        "energy_Eh": float(e_solute),
        "charge": charge,
    }
    if WRITE_XYZ:
        solute_entry["xyz"] = atoms_to_xyz_list(atomsA, Z_A)
    results["solute"] = solute_entry

    e_solvent = run_tblite_sp(atomsB, Z_B, charge=0)
    solvent_entry = {
        "energy_Eh": float(e_solvent),
        "charge": 0,
    }
    if WRITE_XYZ:
        solvent_entry["xyz"] = atoms_to_xyz_list(atomsB, Z_B)
    results["solvent"] = solvent_entry

    num_grids = dataA["grids"].shape[0]

    # ------------- PARALLEL EXECUTION -------------
    print(f"Launching {num_grids} grid jobs using multiprocessing...")

    # Choose a reasonable number of workers: allow override via env var
    default_max = 4
    max_workers_env = os.environ.get("ALIGN_TB_MAX_WORKERS")
    if max_workers_env is not None:
        try:
            default_max = max(1, int(max_workers_env))
        except ValueError:
            pass

    max_workers = default_max
    ncores = min(max_workers, mp.cpu_count(), num_grids)
    print(f"  → using {ncores} worker processes")

    ctx = mp.get_context("spawn")

    # Choose a chunksize to amortize overhead for many grid points
    if num_grids <= ncores:
        chunksize = 1
    else:
        chunksize = max(1, num_grids // (ncores * 4))

    with ctx.Pool(
        processes=ncores,
        initializer=_init_worker,
        initargs=(dataA, dataB, angles, charge),
    ) as pool:

        for grid_index, gp_result, err in pool.imap_unordered(
            worker_one_grid, range(num_grids), chunksize=chunksize
        ):
            if err is not None:
                print(f"[WARNING] grid {grid_index} failed: {err}", file=sys.stderr)
            else:
                results[f"gp{grid_index}"] = gp_result

    # ------------- Save JSON safely -------------

    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        return obj

    with open(out_json, "w") as fh:
        json.dump(results, fh, indent=2, default=convert_numpy)

    print(f"\n*** Saved all SP results to {out_json}")
    return results


# --------------------------- CLI ---------------------------

def cli_main():
    parser = argparse.ArgumentParser(
        description="Generate complexes and run tblite SP for all grids (parallel, optimized)."
    )
    parser.add_argument("fileA", help="NPZ file for solute")
    parser.add_argument("fileB", help="NPZ file for solvent")
    parser.add_argument("angles_csv", help="Comma-separated rotation angles")
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--out", default="tb_lite_results.json")
    args = parser.parse_args()

    main(args.fileA, args.fileB, args.angles_csv, charge=args.charge, out_json=args.out)


if __name__ == "__main__":
    cli_main()

