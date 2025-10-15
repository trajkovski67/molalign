#!/usr/bin/env python3
"""
all_grid_tb_parallel.py

Run TB-lite calculations per-grid in batches of 4 parallel, then next batch, etc.
"""
import argparse
import numpy as np
from molalign import align_many_to_one_tb as align_mod
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

BOHR_TO_ANG = 0.52917721092

def run_single_grid(args):
    """Worker for a single grid."""
    fileA, fileB, grid_index, angles, charge = args
    try:
        return align_mod.main(fileA, fileB, grid_index, angles, charge=charge)
    except Exception as e:
        print(f"Skipped grid {grid_index} due to error: {e}")
        return None

def main(fileA, fileB, angles_csv, charge=0, out_json="tb_lite_results.json", batch_size=4):
    import os
    import json

    fileA = os.path.abspath(fileA)
    fileB = os.path.abspath(fileB)
    angles = [float(a) for a in angles_csv.split(",")]

    dataA = np.load(fileA, allow_pickle=True)
    dataB = np.load(fileB, allow_pickle=True)

    atomsA = dataA["atoms"][:, :3]
    Z_A = dataA["atoms"][:, 3].astype(int)
    atomsB = dataB["atoms"][:, :3]
    Z_B = dataB["atoms"][:, 3].astype(int)

    # Prepare results
    results = {"molA_grids_xyz": dataA["grids"][:, :3].tolist()}

    from molalign.all_grid_tb import run_tblite_sp
    results["solute"] = {"energy_Eh": float(run_tblite_sp(atomsA, Z_A, charge=charge)), "charge": charge}
    results["solvent"] = {"energy_Eh": float(run_tblite_sp(atomsB, Z_B, charge=0)), "charge": 0}

    num_grids = dataA["grids"].shape[0]
    worker_args = [(fileA, fileB, i, angles, charge) for i in range(num_grids)]

    # Run in batches of `batch_size`
    grid_results = []
    for i in tqdm(range(0, num_grids, batch_size), desc="Grids"):
        batch = worker_args[i:i+batch_size]
        with Pool(processes=len(batch)) as pool:
            batch_results = pool.map(run_single_grid, batch)
        grid_results.extend(batch_results)

    # Collect results
    for i, res in enumerate(grid_results):
        if res is not None:
            results[f"gp{i}"] = res

    # Save JSON
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        return obj

    with open(out_json, "w") as fh:
        json.dump(results, fh, indent=2, default=convert_numpy)

    print(f"*** Saved all SP results to {out_json}")
    return results

def cli_main():
    parser = argparse.ArgumentParser(description="Run TB-lite per-grid calculations in parallel batches")
    parser.add_argument("fileA")
    parser.add_argument("fileB")
    parser.add_argument("angles_csv")
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--out", default="tb_lite_results.json")
    parser.add_argument("--batch-size", type=int, default=4, help="Number of grids to run in parallel per batch")
    args = parser.parse_args()
    main(args.fileA, args.fileB, args.angles_csv, charge=args.charge, out_json=args.out, batch_size=args.batch_size)

if __name__ == "__main__":
    cli_main()

