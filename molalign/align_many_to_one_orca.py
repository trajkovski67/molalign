#!/usr/bin/env python3
"""
align_many_to_one_orca.py  (Parallel OPI version; no files kept)

Align molB to molA grid points, run ORCA SP via OPI (default Dft.R2SCAN_3C),
and return results in memory — no persistent files left on disk.
Parallelizes all SPs per solute grid, runs grids serially.
"""

import json
import numpy as np
import tempfile
from pathlib import Path
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from opi.core import Calculator
from opi.input.simple_keywords import Dft, Task
from opi.input.structures.structure import Structure


# ---------------- ORCA/OPI wrapper ----------------

def _pick_dft(method_name: str):
    """Map a CLI method string to Dft enum; default r2scan-3c."""
    if not method_name:
        return Dft.R2SCAN_3C
    key = method_name.strip().upper().replace("-", "_")
    aliases = {
        "R2SCAN3C": "R2SCAN_3C",
        "R2SCAN_3C": "R2SCAN_3C",
        "PBE": "PBE",
        "PBE0": "PBE0",
        "B3LYP": "B3LYP",
        "BP86": "BP86",
        "TPSS": "TPSS",
        "WB97X": "WB97X",
    }
    key = aliases.get(key, key)
    return getattr(Dft, key, Dft.R2SCAN_3C)


def run_orca_sp_opi(coords_angstrom, atomic_numbers,
                    charge=0, mult=1, ncores=8, method="r2scan-3c"):
    """
    Run an ORCA single-point via OPI (orca-pi) fully in-memory.
    Returns energy in Hartree (float). Cleans up temp files.
    """
    # ensure ORCA respects requested cores
    os.environ["OMP_NUM_THREADS"] = str(ncores)
    os.environ["ORCA_NUM_PROCS"] = str(ncores)

    sym_from_Z = {
        1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O",
        9: "F", 10: "Ne", 11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P",
        16: "S", 17: "Cl", 18: "Ar", 35: "Br", 53: "I"
    }

    # Build temporary XYZ
    lines = [str(len(atomic_numbers)), ""]
    for Z, (x, y, z) in zip(atomic_numbers, coords_angstrom):
        sym = sym_from_Z.get(int(Z), str(int(Z)))
        lines.append(f"{sym:<3s} {x: .8f} {y: .8f} {z: .8f}")
    xyz_text = "\n".join(lines) + "\n"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        xyz_path = tmpdir / "geom.xyz"
        xyz_path.write_text(xyz_text)

        structure = Structure.from_xyz(xyz_path)
        structure.charge = int(charge)
        structure.multiplicity = int(mult)

        calc = Calculator(basename="sp", working_dir=tmpdir)
        calc.structure = structure
        calc.input.add_simple_keywords(_pick_dft(method), Task.SP)

        # Manually add %pal block (per ORCA 6.1 manual §2.5)
        calc.input.text_blocks = {"%pal": f"nprocs {int(ncores)}\nend"}

        calc.write_input()
        calc.run()

        output = calc.get_output()
        if not output.terminated_normally():
            out_file = calc.working_dir / f"{calc.basename}.out"
            tail = out_file.read_text().splitlines()[-50:] if out_file.exists() else []
            raise RuntimeError("ORCA did not terminate normally.\n" + "\n".join(tail))

        output.parse()
        return float(output.results_properties.geometries[0].single_point_data.finalenergy)


# ---------------- Geometry utilities ----------------

def atoms_to_xyz_list(coords, atomic_numbers):
    return [
        {"element": str(int(Z)), "x": float(p[0]), "y": float(p[1]), "z": float(p[2])}
        for p, Z in zip(coords, atomic_numbers)
    ]


def rotation_matrix_from_vectors(a, b, eps=1e-12):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < eps or nb < eps:
        return np.eye(3)
    a /= na
    b /= nb
    dot = np.dot(a, b)
    if dot > 1 - eps:
        return np.eye(3)
    if dot < -1 + eps:
        arbitrary = np.array([1, 0, 0]) if abs(a[0]) <= 0.9 else np.array([0, 1, 0])
        axis = np.cross(a, arbitrary)
        axis /= np.linalg.norm(axis)
        return -np.eye(3) + 2 * np.outer(axis, axis)
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    K = np.array([[0, -v[2], v[1]],
                  [v[2], 0, -v[0]],
                  [-v[1], v[0], 0]])
    return np.eye(3) + K + K.dot(K) * ((1 - dot) / (s ** 2))


def rotate_around_axis(coords, axis, angle_deg):
    theta = np.radians(angle_deg)
    k = axis / (np.linalg.norm(axis) + 1e-20)
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]])
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    return coords @ R.T


# ---------------- Parallel helper ----------------

def _compute_single_sp(args):
    """Worker function to compute one SP energy (single solvent grid + rotation)."""
    (
        atomsA, Z_A,
        atomsB, Z_B,
        pA, nA,
        pB, nB,
        idxB, angle,
        charge, mult, ncores, method
    ) = args

    key = f"b{idxB}_rot{int(angle)}"
    target = -nA
    try:
        # Align solvent normal to solute normal
        R_align = rotation_matrix_from_vectors(nB, target)
        atomsB_rot = atomsB @ R_align.T
        gridsB_rot = pB @ R_align.T if isinstance(pB, np.ndarray) else pB
        translation = pA - (gridsB_rot if isinstance(gridsB_rot, np.ndarray) else pB)
        atomsB_rot += translation

        # Rotate around solute normal
        atomsB_final = rotate_around_axis(atomsB_rot - pA, target, angle) + pA

        atoms_combined = np.vstack((atomsA, atomsB_final))
        Z_combined = np.hstack((Z_A, Z_B))

        energy = run_orca_sp_opi(
            atoms_combined, Z_combined,
            charge=charge, mult=mult, ncores=ncores, method=method
        )

        return key, {
            "energy_Eh": float(energy),
            "xyz": atoms_to_xyz_list(atoms_combined, Z_combined),
        }

    except Exception as e:
        return key, {"error": str(e)}


# ---------------- Main workflow ----------------

def main(fileA, fileB, idxA, angles=[0], charge=0, mult=1,
         ncores=8, method="r2scan-3c", max_workers=None):
    """
    Align molB to grid idxA of molA, run ORCA SPs via OPI in parallel per grid.
    Returns dict of results.
    """

    dataA = np.load(fileA, allow_pickle=True)
    dataB = np.load(fileB, allow_pickle=True)

    atomsA = dataA["atoms"][:, :3]
    Z_A = dataA["atoms"][:, 3].astype(int)
    gridsA = dataA["grids"][:, :3]
    normalsA = gridsA - atomsA[dataA["grids"][:, 4].astype(int)]
    normalsA /= np.linalg.norm(normalsA, axis=1, keepdims=True) + 1e-12

    atomsB = dataB["atoms"][:, :3]
    Z_B = dataB["atoms"][:, 3].astype(int)
    gridsB = dataB["grids"][:, :3]
    normalsB = gridsB - atomsB[dataB["grids"][:, 4].astype(int)]
    normalsB /= np.linalg.norm(normalsB, axis=1, keepdims=True) + 1e-12

    pA = gridsA[idxA]
    nA = normalsA[idxA]

    # Prepare all independent SP jobs for this solute grid
    tasks = []
    for idxB, (pB, nB) in enumerate(zip(gridsB, normalsB)):
        for angle in angles:
            tasks.append((
                atomsA, Z_A,
                atomsB, Z_B,
                pA, nA,
                pB, nB,
                idxB, angle,
                charge, mult, ncores, method
            ))

    all_complexes = {}

    # Determine number of parallel workers (auto if not given)
    if max_workers is None:
        total_cores = os.cpu_count() or 8
        max_workers = max(1, total_cores // ncores)

    print(f"[Grid {idxA}] Running {len(tasks)} SPs in parallel (max_workers={max_workers}, ncores/job={ncores})")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_compute_single_sp, t) for t in tasks]
        for fut in as_completed(futures):
            key, result = fut.result()
            all_complexes[key] = result

    return all_complexes


# ---------------- CLI ----------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Align molB to one molA grid site and run ORCA SPs via OPI (parallel per grid)."
    )
    parser.add_argument("fileA", help="NPZ file for solute")
    parser.add_argument("fileB", help="NPZ file for solvent")
    parser.add_argument("idxA", type=int, help="Index of grid point on molA")
    parser.add_argument("--angles", default="0", help="Comma-separated rotation angles (deg)")
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--mult", type=int, default=1)
    parser.add_argument("--ncores", type=int, default=8, help="Cores per ORCA SP (OPI)")
    parser.add_argument("--method", default="r2scan-3c", help="DFT method key (e.g. r2scan-3c, pbe0, b3lyp)")
    parser.add_argument("--max-workers", type=int, default=None,
                        help="Max parallel SP jobs per grid (auto if not set)")
    args = parser.parse_args()

    angles = [float(a) for a in args.angles.split(",")]
    results = main(
        args.fileA, args.fileB, args.idxA,
        angles=angles,
        charge=args.charge,
        mult=args.mult,
        ncores=args.ncores,
        method=args.method,
        max_workers=args.max_workers
    )
    print(json.dumps(results, indent=2))

