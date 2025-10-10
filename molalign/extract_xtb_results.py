#!/usr/bin/env python3
import os
import re
import json
import glob
import argparse
import sys

def parse_xtb_output(out_file):
    """Extract total energy (Eh) and gradient norm (Eh/α) from xTB output."""
    energy = None
    grad_norm = None
    with open(out_file, "r") as f:
        for line in f:
            if "TOTAL ENERGY" in line:
                match = re.search(r"([-+]?\d*\.\d+|\d+)", line)
                if match:
                    energy = float(match.group(1))
            elif "GRADIENT NORM" in line:
                match = re.search(r"([-+]?\d*\.\d+|\d+)", line)
                if match:
                    grad_norm = float(match.group(1))
    return energy, grad_norm


def read_xyz(xyz_file):
    """Read XYZ file and return list of atoms as dicts with element symbol and coordinates."""
    atoms = []
    try:
        with open(xyz_file, "r") as f:
            lines = f.readlines()
        for line in lines[2:]:  # skip atom count + comment
            parts = line.strip().split()
            if len(parts) != 4:
                continue
            sym, x, y, z = parts
            atoms.append({
                "element": sym,
                "x": float(x),
                "y": float(y),
                "z": float(z)
            })
    except FileNotFoundError:
        pass
    return atoms


def read_sp_energy(folder):
    """Return total energy (Eh) from a single-point .out file in the folder."""
    out_files = glob.glob(os.path.join(folder, "*.out"))
    if not out_files:
        return None
    energy, _ = parse_xtb_output(out_files[0])
    return energy


def collect_xtb_results(base_dir):
    """Collect xTB results into dict, with solute/solvent energies and xyz for complexes."""
    results = {}
    pattern = re.compile(r"gp(\d+)_gp(\d+)_b(\d+)_rot(\d+)_atoms\.out")

    # --- Solute and solvent energies ---
    solute_folder = os.path.join(base_dir, "solute")
    solvent_folder = os.path.join(base_dir, "solvent")

    solute_energy = read_sp_energy(solute_folder)
    solvent_energy = read_sp_energy(solvent_folder)

    if solute_energy is not None:
        results["solute"] = {"energy_Eh": solute_energy}
    if solvent_energy is not None:
        results["solvent"] = {"energy_Eh": solvent_energy}

    # --- Complex results ---
    for out_file in glob.glob(os.path.join(base_dir, "**", "*.out"), recursive=True):
        filename = os.path.basename(out_file)
        match = pattern.match(filename)
        if not match:
            continue

        gpA, gpB, b_index, rot = match.groups()
        gp_key = f"gp{gpA}"
        energy, grad = parse_xtb_output(out_file)
        if energy is None:
            continue

        xyz_file = out_file.replace(".out", ".xyz")
        atoms = read_xyz(xyz_file)

        results.setdefault(gp_key, {})[f"b{b_index}_rot{rot}"] = {
            "energy_Eh": energy,
            "grad_norm_Eh_per_a": grad,
            "xyz": atoms
        }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Extract xTB total energies, gradient norms, and coordinates from output files."
    )
    parser.add_argument(
        "base_dir",
        help="Root folder containing xTB .out files (searched recursively).",
    )
    parser.add_argument(
        "--out",
        help="Output JSON filename (default: xtb_results.json inside base_dir).",
        default=None,
    )

    args = parser.parse_args()

    base_dir = os.path.abspath(args.base_dir)
    if not os.path.isdir(base_dir):
        print(f"ERROR: '{base_dir}' is not a valid directory.", file=sys.stderr)
        sys.exit(1)

    results = collect_xtb_results(base_dir)
    if not results:
        print(f"⚠️ No valid xTB results found in {base_dir}")
        sys.exit(0)

    out_json = args.out or os.path.join(base_dir, "xtb_results.json")

    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"*** Extracted results from {base_dir}")
    print(f"*** Saved summary to {out_json}")
    print(f"*** Total grid points processed: {len([k for k in results if k.startswith('gp')])}")


if __name__ == "__main__":
    main()

