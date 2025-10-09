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


def collect_xtb_results(base_dir):
    """Walk through directories and collect xTB results into a nested dictionary."""
    results = {}
    pattern = re.compile(r"gp(\d+)_gp(\d+)_b(\d+)_rot(\d+)_atoms\.out")

    for out_file in glob.glob(os.path.join(base_dir, "**", "*.out"), recursive=True):
        filename = os.path.basename(out_file)
        match = pattern.match(filename)
        if not match:
            continue

        gpA, gpB, b_index, rot = match.groups()
        energy, grad = parse_xtb_output(out_file)
        if energy is None:
            continue

        results.setdefault(f"gp{gpA}", {})[f"b{b_index}_rot{rot}"] = {
            "energy_Eh": energy,
            "grad_norm_Eh_per_a": grad
        }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Extract xTB total energies and gradient norms from output files."
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
        print(f"❌ Error: '{base_dir}' is not a valid directory.", file=sys.stderr)
        sys.exit(1)

    results = collect_xtb_results(base_dir)
    if not results:
        print(f"⚠️ No valid xTB results found in {base_dir}")
        sys.exit(0)

    out_json = args.out or os.path.join(base_dir, "xtb_results.json")

    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✅ Extracted results from {base_dir}")
    print(f"📁 Saved summary to {out_json}")
    print(f"📊 Total grid points processed: {len(results)}")


if __name__ == "__main__":
    main()

