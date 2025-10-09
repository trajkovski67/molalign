#!/usr/bin/env python3
import os
import re
import json
import glob

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


def collect_xtb_results(base_dir="water_data_vs_proton_data_complexes"):
    """Walk through directories and collect xTB results."""
    results = {}

    for out_file in glob.glob(os.path.join(base_dir, "**", "*.out"), recursive=True):
        filename = os.path.basename(out_file)
        # Example: gp2_gp0_b0_rot40_atoms.out
        match = re.match(r"gp(\d+)_gp(\d+)_b(\d+)_rot(\d+)_atoms\.out", filename)
        if not match:
            continue

        gpA, gpB, b_index, rot = match.groups()
        energy, grad = parse_xtb_output(out_file)
        if energy is None:
            continue

        # Nested dict
        if f"gp{gpA}" not in results:
            results[f"gp{gpA}"] = {}

        results[f"gp{gpA}"][f"b{b_index}_rot{rot}"] = {
            "energy_Eh": energy,
            "grad_norm_Eh_per_a": grad
        }

    return results


def main():
    base_dir = "water_data_vs_proton_data_complexes"
    results = collect_xtb_results(base_dir)

    # Save as JSON
    out_json = os.path.join(base_dir, "xtb_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✅ Extracted results from {base_dir}")
    print(f"Saved summary to {out_json}")
    print(f"Total grid points processed: {len(results)}")


if __name__ == "__main__":
    main()
