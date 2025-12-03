#!/usr/bin/env python3
import os, json, argparse, numpy as np, multiprocessing as mp
from tqdm import tqdm
import molalign.align_many_to_one_orca as align_mod

WRITE_XYZ = os.environ.get("ALIGN_TB_WRITE_XYZ", "1") != "0"


_WORKER = {}

def _init_worker(dataA, dataB, angles, charge):
    _WORKER["preA"] = align_mod.prepare_data(dataA)
    _WORKER["preB"] = align_mod.prepare_data(dataB)
    _WORKER["angles"] = angles
    _WORKER["charge"] = charge


def worker_one_grid(idx):
    try:
        r = align_mod.main_from_prepared(
            _WORKER["preA"], _WORKER["preB"],
            idx, _WORKER["angles"], _WORKER["charge"]
        )
        return idx, r, None
    except Exception as e:
        return idx, None, repr(e)


def atoms_to_xyz_list(coords, Z):
    return [{"element": str(int(Zi)), "x": float(x), "y": float(y), "z": float(z)}
            for (x,y,z),Zi in zip(coords,Z)]


def main(fileA, fileB, angles_csv, charge, out_json):
    dataA = dict(np.load(fileA, allow_pickle=True))
    dataB = dict(np.load(fileB, allow_pickle=True))
    angles = [float(a) for a in angles_csv.split(",")]

    atomsA = dataA["atoms"][:, :3]
    ZA = dataA["atoms"][:, 3].astype(int)
    atomsB = dataB["atoms"][:, :3]
    ZB = dataB["atoms"][:, 3].astype(int)

    results = {}
    results["molA_grids_xyz"] = dataA["grids"][:, :3].tolist()

    print("Solute energy...")
    results["solute"] = {"energy_Eh": align_mod.run_orca_sp(atomsA, ZA, charge), "charge": charge}
    if WRITE_XYZ:
        results["solute"]["xyz"] = atoms_to_xyz_list(atomsA, ZA)

    print("Solvent energy...")
    results["solvent"] = {"energy_Eh": align_mod.run_orca_sp(atomsB, ZB, 0), "charge": 0}
    if WRITE_XYZ:
        results["solvent"]["xyz"] = atoms_to_xyz_list(atomsB, ZB)

    ncores = min(os.cpu_count(), len(dataA["grids"]))
    ctx = mp.get_context("spawn")

    with ctx.Pool(ncores, initializer=_init_worker, initargs=(dataA, dataB, angles, charge)) as pool:
        for idx, res, err in tqdm(pool.imap_unordered(worker_one_grid, range(len(dataA["grids"]))),
                                  total=len(dataA["grids"])):
            if err:
                print(f"Grid {idx} failed: {err}")
            else:
                results[f"gp{idx}"] = res

    json.dump(results, open(out_json, "w"), indent=2)
    print(f"DONE: {out_json}")


def cli_main():
    p = argparse.ArgumentParser()
    p.add_argument("fileA")
    p.add_argument("fileB")
    p.add_argument("angles")
    p.add_argument("--charge", type=int, default=0)
    p.add_argument("--out", default="orca_results.json")
    a = p.parse_args()
    main(a.fileA, a.fileB, a.angles, a.charge, a.out)


if __name__ == "__main__":
    cli_main()

