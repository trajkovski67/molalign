#!/usr/bin/env python3
"""
boltzmann_multi.py

Boltzmann analysis for multiple TB-lite JSON result files,
based on the minimum interaction energy per grid point.

For each JSON:
- ΔE = E_complex − (E_solute + E_solvent) [kcal/mol]
- For each grid point gpX, we take the minimum ΔE among its complexes
- Boltzmann analysis is done over these per-grid-point minima
- Can count how many grid points are within X% of the global minimum
- Saves top N grid-point minima (their best complex) as XYZ per temperature
"""

import json
import numpy as np
import os
import sys
import glob

HARTREE_TO_KCAL = 627.5095
kB = 0.0019872041  # kcal/mol/K

# Atomic number → element symbol mapping
Z_TO_SYMBOL = {1: "H", 6: "C", 7: "N", 8: "O", 16: "S", 17: "Cl", 35: "Br"}


# --------------------------------------------------------------
# Utilities
# --------------------------------------------------------------

def boltzmann_weights(energies, T=298.15):
    """Return normalized Boltzmann weights for a 1D array of energies (kcal/mol)."""
    energies = np.array(energies, dtype=float)
    min_E = energies.min()
    delta_E = energies - min_E
    weights = np.exp(-delta_E / (kB * T))
    return weights / weights.sum()


def save_xyz(xyz, filename, comment=""):
    """Write a list of atoms with 'element', 'x', 'y', 'z' to XYZ format."""
    with open(filename, "w") as fh:
        fh.write(f"{len(xyz)}\n")
        fh.write(comment + "\n")
        for atom in xyz:
            Z = int(atom["element"])
            symbol = Z_TO_SYMBOL.get(Z, str(Z))
            fh.write(f"{symbol} {atom['x']:.6f} {atom['y']:.6f} {atom['z']:.6f}\n")


def collect_gridpoint_minima(json_files):
    """
    For each JSON file and each grid point gpX, find the complex with
    minimum ΔE = E_complex − (E_solute + E_solvent) [kcal/mol].

    Returns:
        list of (label, min_deltaE_kcal, xyz_of_min_complex)
        where label = "<basename.json>_<gpX>"
    """
    gp_minima = []

    for f in json_files:
        try:
            with open(f) as fh:
                data = json.load(fh)
        except Exception:
            continue

        # Extract solute and solvent energies
        try:
            E_solute = float(data["solute"]["energy_Eh"])
            E_solvent = float(data["solvent"]["energy_Eh"])
        except KeyError:
            continue

        for gp_key, gp_data in data.items():
            # Skip non-gridpoint keys
            if gp_key in ("solute", "solvent"):
                continue

            # Only accept dict or list
            if isinstance(gp_data, dict):
                complex_items = gp_data.items()
            elif isinstance(gp_data, list):
                complex_items = enumerate(gp_data)
            else:
                continue

            min_deltaE = None
            min_xyz = None

            for complex_key, entry in complex_items:
                try:
                    E_complex = float(entry["energy_Eh"])
                    deltaE_kcal = (E_complex - (E_solute + E_solvent)) * HARTREE_TO_KCAL
                    xyz = entry.get("xyz", [])
                except Exception:
                    continue

                if (min_deltaE is None) or (deltaE_kcal < min_deltaE):
                    min_deltaE = deltaE_kcal
                    min_xyz = xyz

            if min_deltaE is not None:
                label = f"{os.path.basename(f)}_{gp_key}"
                gp_minima.append((label, min_deltaE, min_xyz))

    return gp_minima


# --------------------------------------------------------------
# Per-temperature analysis using grid-point minima
# --------------------------------------------------------------

def run_single_temperature(json_files, T, top_n, out_dir, within_percent=None):
    """
    Perform Boltzmann analysis at temperature T using minimum ΔE per grid point.

    Args:
        json_files: list of TB-lite JSON files.
        T: temperature in K.
        top_n: how many best grid points to save as XYZ.
        out_dir: base output directory.
        within_percent: if not None, count grid points within X% of global minimum.
    """
    gp_minima = collect_gridpoint_minima(json_files)
    if not gp_minima:
        print("No valid grid-point minima found in the input JSON files!")
        sys.exit(1)

    labels, deltaE_list, xyz_list = zip(*gp_minima)
    deltaE_list = np.array(deltaE_list, dtype=float)

    # Boltzmann weights over grid-point minima
    weights = boltzmann_weights(deltaE_list, T=T)
    E_avg = np.sum(deltaE_list * weights)

    print(f"\n=== Temperature: {T:.2f} K ===")
    print(f"Number of grid points (with minima) = {len(deltaE_list)}")
    print(f"Boltzmann-weighted ⟨ΔE_min(gp)⟩ = {E_avg:.3f} kcal/mol")

    # Temperature-specific directory
    T_dir = os.path.join(out_dir, f"T_{int(T)}K")
    os.makedirs(T_dir, exist_ok=True)

    # Save top N grid points by Boltzmann weight (their minimum complexes)
    top_idx = np.argsort(-weights)[:top_n]
    for i in top_idx:
        label = labels[i]
        deltaE = deltaE_list[i]
        xyz_file = os.path.join(T_dir, f"{label}.xyz")
        save_xyz(
            xyz_list[i],
            xyz_file,
            comment=f"{label} ΔE_min(gp)={deltaE:.3f} kcal/mol"
        )

    # Also write full distribution as text
    dist_file = os.path.join(T_dir, "gridpoint_minima_distribution.txt")
    with open(dist_file, "w") as fh:
        fh.write("# label   DeltaE_min_kcal   Boltzmann_weight\n")
        for lbl, dE, w in zip(labels, deltaE_list, weights):
            fh.write(f"{lbl}   {dE:.6f}   {w:.6e}\n")
    print(f"Full weight distribution written to {dist_file}")

    # ----------------------------------------------------------
    # Count grid points within X% of global minimum, if requested
    # ----------------------------------------------------------
    if within_percent is not None:
        DeltaE_min_global = deltaE_list.min()
        threshold = DeltaE_min_global * (1 + within_percent / 100.0)
        mask = deltaE_list <= threshold

        count_within = int(np.sum(mask))
        total = len(deltaE_list)
        frac = count_within / total * 100.0

        print(f"\nWithin {within_percent:.1f}% of global minimum ΔE_min(gp):")
        print(f"  Global minimum ΔE_min(gp) = {DeltaE_min_global:.3f} kcal/mol")
        print(f"  Threshold                 = {threshold:.3f} kcal/mol")
        print(f"  Grid points               = {count_within}/{total} ({frac:.1f}%)")

        # Save detailed list
        summary_file = os.path.join(T_dir, f"within_{within_percent}pct_of_min_gp.txt")
        with open(summary_file, "w") as fh:
            fh.write(f"Global minimum ΔE_min(gp): {DeltaE_min_global:.6f} kcal/mol\n")
            fh.write(f"Threshold: {threshold:.6f} kcal/mol\n")
            fh.write(f"Grid points within {within_percent}%: {count_within}/{total}\n\n")
            fh.write("Grid points within threshold:\n")
            for lbl, dE, ok in zip(labels, deltaE_list, mask):
                if ok:
                    fh.write(f"{lbl}   ΔE_min(gp)={dE:.4f} kcal/mol\n")

        print(f"List of grid points within X% saved to {summary_file}")


# --------------------------------------------------------------
# Flexible temperature parsing (single, multi, ranges)
# --------------------------------------------------------------

def parse_temperature_argument(temp_arg):
    """
    Flexible temperature parser:
    - Single value:     298.15
    - Multiple values:  298.15 310 350
    - Range:            250:400:10  (start:end:step)
    """
    temps = []

    for token in temp_arg:
        if ":" in token:
            parts = token.split(":")
            if len(parts) != 3:
                raise ValueError("Range format must be start:end:step")
            start, end, step = map(float, parts)
            temps.extend(list(np.arange(start, end + 1e-8, step)))
        else:
            temps.append(float(token))

    return sorted(set(temps))


# --------------------------------------------------------------
# CLI
# --------------------------------------------------------------

def cli_main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Boltzmann analysis of TB-lite JSON results "
                    "using minimum ΔE per grid point."
    )

    parser.add_argument(
        "--temperature", nargs="+", default=["298.15"],
        help="Temperature(s): e.g. 298.15 OR 298 310 350 OR 250:400:10"
    )

    parser.add_argument(
        "--top_n", type=int, default=10,
        help="Number of top grid points (by Boltzmann weight) to save as XYZ"
    )

    parser.add_argument(
        "--within_percent", type=float,
        help="Count grid points whose minimum ΔE is within X%% of the global minimum"
    )

    parser.add_argument(
        "--out_dir", default="top_complexes",
        help="Base output directory"
    )

    parser.add_argument(
        "--recursive", action="store_true",
        help="Search subdirectories for JSON files"
    )

    args = parser.parse_args()

    # Collect JSON files
    if args.recursive:
        json_files = glob.glob("**/*.json", recursive=True)
    else:
        json_files = glob.glob("*.json")

    if not json_files:
        print("No JSON files found.")
        sys.exit(1)

    print(f"Found {len(json_files)} JSON files.")

    # Temperatures
    try:
        temperatures = parse_temperature_argument(args.temperature)
    except Exception as e:
        print(f"Error parsing --temperature: {e}")
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    for T in temperatures:
        run_single_temperature(
            json_files=json_files,
            T=T,
            top_n=args.top_n,
            out_dir=args.out_dir,
            within_percent=args.within_percent,
        )


if __name__ == "__main__":
    cli_main()

