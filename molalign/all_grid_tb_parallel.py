#!/usr/bin/env python3
"""
all_grid_tb_parallel2.py  (Improved)

- Cleaner console output
- tqdm progress bar with ETA
- Per-worker reusable Calculator for TB-lite (where supported)
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")  # avoid oversubscription in tblite/OpenMP

import sys
import json
import argparse
import numpy as np
import multiprocessing as mp
from tqdm import tqdm

import molalign.align_many_to_one_tb_parallel2 as align_mod


BOHR_TO_ANG = 0.52917721092
WRITE_XYZ = os.environ.get("ALIGN_TB_WRITE_XYZ", "1") != "0"

_WORKER_DATA = {}  # shared within each worker process only


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


def run_tblite_sp_single(coords_angstrom, atomic_numbers, method="GFN2-xTB", charge=0):
    """
    Simple single-point wrapper for *one* molecule (solute / solvent).
    Creating a fresh Calculator is fine here (only called a few times).
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


# ------------------- per-worker energy function -------------------

def _worker_energy(coords_angstrom, atomic_numbers, method="GFN2-xTB", charge=0):
    """
    Energy function used inside workers, passed into align_mod.main_from_prepared.

    Tries to re-use a single Calculator per worker by updating coordinates.
    Falls back to creating a new Calculator if the API call fails.
    """
    global _WORKER_DATA
    import numpy as np
    import contextlib, io
    from tblite.interface import Calculator

    coords_angstrom = np.array(coords_angstrom, dtype=float)
    Z = np.array(atomic_numbers, dtype=int)

    coords_bohr = coords_angstrom / BOHR_TO_ANG

    calc = _WORKER_DATA.get("calc")

    # 1) Try to reuse calculator
    if calc is not None:
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            try:
                # Assumes tblite Calculator supports these keys.
                # If not, this will throw and we rebuild once.
                calc.set("coordinates", coords_bohr)
                if charge != _WORKER_DATA.get("charge", charge):
                    calc.set("charge", int(charge))
                res = calc.singlepoint()
                _WORKER_DATA["charge"] = int(charge)
                return res.get("energy")
            except Exception:
                # Fallback to rebuilding
                pass

    # 2) Fallback: build a new calculator and store it
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        calc = Calculator(method, Z, coords_bohr, charge=charge)
        calc.add("alpb-solvation", "water")
        res = calc.singlepoint()

    _WORKER_DATA["calc"] = calc
    _WORKER_DATA["charge"] = int(charge)
    return res.get("energy")


# --------------------------- worker init ---------------------------

def _init_worker(dataA, dataB, angles, charge):
    """
    Initializer for worker processes.

    - Prepares preA/preB (geometry, normals, etc.)
    - Optionally creates a first Calculator instance for the combined system.
    """
    global _WORKER_DATA
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

    # Try creating a reusable Calculator once
    calc = None
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        try:
            calc = Calculator("GFN2-xTB", Z_combined, coords_bohr, charge=charge)
            calc.add("alpb-solvation", "water")
        except Exception:
            # If this fails for whatever reason, we'll lazily construct in _worker_energy
            calc = None

    _WORKER_DATA = {
        "preA": preA,
        "preB": preB,
        "angles": angles,
        "charge": int(charge),
        "calc": calc,
    }


# --------------------------- worker job ---------------------------

def worker_one_grid(idxA):
    """
    Worker function that computes all complexes at a single grid index of molA.
    Uses align_mod.main_from_prepared with the custom _worker_energy function.
    """
    global _WORKER_DATA
    preA = _WORKER_DATA["preA"]
    preB = _WORKER_DATA["preB"]
    angles = _WORKER_DATA["angles"]
    charge = _WORKER_DATA["charge"]

    try:
        gp_results = align_mod.main_from_prepared(
            preA,
            preB,
            idxA,
            angles=angles,
            charge=charge,
            energy_fn=_worker_energy,
        )
        return (idxA, gp_results, None)
    except Exception as e:
        return (idxA, None, repr(e))


# --------------------------- main logic ---------------------------

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

    results = {
        "molA_grids_xyz": dataA["grids"][:, :3].tolist()
    }

    # --- Solute and solvent SP (simple, few calls) ---
    print("\nComputing solute energy…")
    e_solute = run_tblite_sp_single(atomsA, Z_A, charge=charge)
    solute_entry = {
        "energy_Eh": float(e_solute),
        "charge": charge,
    }
    if WRITE_XYZ:
        solute_entry["xyz"] = atoms_to_xyz_list(atomsA, Z_A)
    results["solute"] = solute_entry

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

    # ------------- PARALLEL EXECUTION -------------
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
    print("=======================================\n")

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

