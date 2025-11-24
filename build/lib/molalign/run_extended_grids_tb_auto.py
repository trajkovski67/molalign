#!/usr/bin/env python3
"""Auto-scan solute grid shifts for TB-lite calculations based on interaction energy thresholds."""
import argparse
import json
import os
import subprocess
import sys
from typing import Dict, List, Tuple

import numpy as np

EH_TO_KCAL_MOL = 627.509474


# ---------- Grid shift utilities ----------
def compute_normals(grids: np.ndarray, atoms: np.ndarray) -> np.ndarray:
    atom_indices = grids[:, 4].astype(int)
    atom_coords = atoms[:, :3]
    grid_coords = grids[:, :3]
    normals = grid_coords - atom_coords[atom_indices]
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.where(norms > 0, norms, 1.0)
    return normals


def shift_grid_points(grids: np.ndarray, atoms: np.ndarray, shift: float) -> np.ndarray:
    normals = compute_normals(grids, atoms)
    shifted = grids.copy()
    shifted[:, :3] = grids[:, :3] + normals * shift
    return shifted


def save_shifted_npz(orig_npz: str, shift: float, out_file: str) -> str:
    data = np.load(orig_npz, allow_pickle=True)
    atoms = data["atoms"]
    grids = data["grids"]
    shifted_grids = shift_grid_points(grids, atoms, shift)
    np.savez(out_file, atoms=atoms, grids=shifted_grids)
    print(f"*** Saved shifted NPZ ({shift:.3f} Å): {out_file}")
    return out_file


def sanitize_filename(value: float) -> str:
    return str(value).replace(".", "p").replace("-", "n")


# ---------- Interaction energy utilities ----------
def compute_interaction_energies(json_file: str) -> Tuple[np.ndarray, Dict[str, float]]:
    with open(json_file) as fh:
        data = json.load(fh)

    try:
        e_solute = float(data["solute"]["energy_Eh"])
        e_solvent = float(data["solvent"]["energy_Eh"])
        num_grids = len(data["molA_grids_xyz"])
    except KeyError as exc:
        raise KeyError(f"Missing key in {json_file}: {exc}") from exc

    interaction_kcal = []
    for idx in range(num_grids):
        gp_key = f"gp{idx}"
        if gp_key not in data:
            raise KeyError(f"Grid entry {gp_key} absent in {json_file}")
        rotations = data[gp_key]
        if not rotations:
            raise ValueError(f"No rotations stored for {gp_key} in {json_file}")
        e_min = min(float(rot["energy_Eh"]) for rot in rotations.values())
        delta_e = (e_min - e_solute - e_solvent) * EH_TO_KCAL_MOL
        interaction_kcal.append(delta_e)

    arr = np.array(interaction_kcal, dtype=float)
    stats = {
        "count": int(arr.size),
        "min_kcalmol": float(np.min(arr)),
        "max_kcalmol": float(np.max(arr)),
        "mean_kcalmol": float(np.mean(arr)),
        "median_kcalmol": float(np.median(arr)),
    }
    return arr, stats


# ---------- Core workflow ----------
def run_align_grid_tb(
    npz_file: str,
    solvent_npz: str,
    angles: str,
    charge: int,
    out_json: str,
    skip_existing: bool,
) -> None:
    if skip_existing and os.path.exists(out_json):
        print(f"*** Skipping align-grid-tb; using existing {out_json}")
        return

    cmd = [
        "align-grid-tb",
        npz_file,
        solvent_npz,
        angles,
        "--charge",
        str(charge),
        "--out",
        out_json,
    ]
    print(f"\n*** Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"align-grid-tb failed for shift NPZ {npz_file}") from exc


def process_shift(
    shift: float,
    solute_npz: str,
    solvent_npz: str,
    out_dir: str,
    angles: str,
    charge: int,
    skip_existing: bool,
) -> Dict[str, object]:
    suffix = "original" if abs(shift) < 1e-12 else f"shift{sanitize_filename(shift)}"
    out_npz = os.path.join(out_dir, f"solute_data_{suffix}.npz")
    out_json = os.path.join(out_dir, f"tb_lite_results_{suffix}.json")

    if not (skip_existing and os.path.exists(out_npz)):
        save_shifted_npz(solute_npz, shift, out_npz)

    run_align_grid_tb(out_npz, solvent_npz, angles, charge, out_json, skip_existing)

    if not os.path.exists(out_json):
        raise FileNotFoundError(f"Expected results JSON not found: {out_json}")

    grid_minima, stats = compute_interaction_energies(out_json)
    print(
        "    → min ΔE = {min:.2f} kcal/mol | mean ΔE = {mean:.2f} kcal/mol"
        .format(min=stats["min_kcalmol"], mean=stats["mean_kcalmol"])
    )

    record = {
        "shift_A": float(shift),
        "npz_file": out_npz,
        "json_file": out_json,
        "stats": stats,
        "grid_minima_kcalmol": grid_minima.tolist(),
    }
    return record


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Auto-scan solute grid shifts until interaction energy thresholds "
            "are reached (TB-lite backend)."
        )
    )
    parser.add_argument("solute_cpcm", help="CPCM file for solute")
    parser.add_argument("solvent_cpcm", help="CPCM file for solvent")
    parser.add_argument("--out", default="OUT", help="Output directory")
    parser.add_argument("--angles", default="0,90,180,270", help="Comma-separated rotation angles")
    parser.add_argument("--charge", type=int, default=0, help="Total charge for TB-lite calculation")
    parser.add_argument("--skip-existing", action="store_true", help="Reuse existing NPZ/JSON outputs if present")

    parser.add_argument("--inward-step", type=float, default=0.25, help="Step size (Å) for inward scans (equal spacing)")
    parser.add_argument("--max-inward-steps", type=int, default=12, help="Maximum inward shift steps")
    parser.add_argument(
        "--inward-threshold", type=float, default=50.0,
        help="Stop inward scanning when min interaction energy exceeds this value (kcal/mol)"
    )

    parser.add_argument(
        "--outward-initial-step", type=float, default=0.25,
        help="Initial outward step size (Å)"
    )
    parser.add_argument(
        "--outward-multiplier", type=float, default=1.7,
        help="Multiplier applied to the outward step after each iteration"
    )
    parser.add_argument("--max-outward-steps", type=int, default=12, help="Maximum outward shift steps")
    parser.add_argument(
        "--outward-threshold", type=float, default=-1.0,
        help="Stop outward scanning when min interaction energy rises above this value (kcal/mol)"
    )

    args = parser.parse_args()

    if args.outward_multiplier <= 1.0:
        parser.error("--outward-multiplier must be greater than 1.0 for exponential spacing")
    if args.inward_step <= 0 or args.outward_initial_step <= 0:
        parser.error("Step sizes must be positive")

    solute = os.path.abspath(args.solute_cpcm)
    solvent = os.path.abspath(args.solvent_cpcm)
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    for molfile in [solute, solvent]:
        molname = os.path.splitext(os.path.basename(molfile))[0]
        print(f"\n*** Running: cpcm-reader {molfile} {molname}")
        try:
            subprocess.run(["cpcm-reader", molfile, molname], check=True)
        except subprocess.CalledProcessError as exc:
            print(f"ERROR: cpcm-reader failed for {molfile}: {exc}", file=sys.stderr)
            sys.exit(1)

        npzfile = f"{molname}_data.npz"
        if not os.path.exists(npzfile):
            print(f"ERROR: Missing expected {npzfile}", file=sys.stderr)
            sys.exit(1)
        os.replace(npzfile, os.path.join(out_dir, npzfile))

    solute_npz = os.path.join(out_dir, f"{os.path.splitext(os.path.basename(solute))[0]}_data.npz")
    solvent_npz = os.path.join(out_dir, f"{os.path.splitext(os.path.basename(solvent))[0]}_data.npz")

    shifts_summary: List[Dict[str, object]] = []

    print("\n=== Running base (0.0 Å) shift ===")
    base_record = process_shift(
        0.0,
        solute_npz,
        solvent_npz,
        out_dir,
        args.angles,
        args.charge,
        args.skip_existing,
    )
    shifts_summary.append(base_record)

    # Inward scanning (negative shifts, equal spacing)
    current_shift = 0.0
    for step_index in range(1, args.max_inward_steps + 1):
        current_shift -= args.inward_step
        print(f"\n=== Inward shift {step_index}: {current_shift:.3f} Å ===")
        record = process_shift(
            current_shift,
            solute_npz,
            solvent_npz,
            out_dir,
            args.angles,
            args.charge,
            args.skip_existing,
        )
        shifts_summary.append(record)
        min_energy = float(min(record["grid_minima_kcalmol"]))
        if min_energy >= args.inward_threshold:
            print(
                f"Reached inward threshold: min ΔE {min_energy:.2f} ≥ {args.inward_threshold} kcal/mol"
            )
            break
    else:
        print(
            f"⚠️ Inward threshold of {args.inward_threshold} kcal/mol not reached within "
            f"{args.max_inward_steps} steps."
        )

    # Outward scanning (positive shifts, exponentially increasing spacing)
    current_shift = 0.0
    current_step = args.outward_initial_step
    for step_index in range(1, args.max_outward_steps + 1):
        current_shift += current_step
        print(f"\n=== Outward shift {step_index}: {current_shift:.3f} Å ===")
        record = process_shift(
            current_shift,
            solute_npz,
            solvent_npz,
            out_dir,
            args.angles,
            args.charge,
            args.skip_existing,
        )
        shifts_summary.append(record)
        min_energy = float(min(record["grid_minima_kcalmol"]))
        if min_energy >= args.outward_threshold:
            print(
                f"Reached outward threshold: min ΔE {min_energy:.2f} ≥ {args.outward_threshold} kcal/mol"
            )
            break
        current_step *= args.outward_multiplier
    else:
        print(
            f"⚠️ Outward threshold of {args.outward_threshold} kcal/mol not reached within "
            f"{args.max_outward_steps} steps."
        )

    summary_path = os.path.join(out_dir, "shift_scan_summary.json")
    with open(summary_path, "w") as fh:
        json.dump(shifts_summary, fh, indent=2)
    print(f"\n*** Saved shift summary to {summary_path}")
    print("*** Individual results saved to:")
    for entry in shifts_summary:
        print(f"    {entry['json_file']}")


if __name__ == "__main__":
    main()

