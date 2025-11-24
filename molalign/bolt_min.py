#!/usr/bin/env python3
"""
Compute Boltzmann weights over minimum ΔE of each gpX,
and optionally print diagnostic info for gp1.
"""

import json
import numpy as np
import glob
import argparse
import sys

HARTREE_TO_KCAL = 627.5095
kB = 0.0019872041  # kcal/mol/K


# ---------------------------------------------------------
# Boltzmann weight function
# ---------------------------------------------------------
def boltzmann_weights(E, T):
    E = np.array(E)
    Emin = E.min()
    dE = E - Emin
    w = np.exp(-dE / (kB * T))
    return w / w.sum()


# ---------------------------------------------------------
# Print gp1 diagnostics optionally
# ---------------------------------------------------------
def print_gp1(json_file, E_solute, E_solvent, gp_data):
    """Print solute, solvent, best gp1 complex energy and ΔE."""
    best_E = None

    if isinstance(gp_data, dict):
        entries = gp_data.values()
    else:
        entries = gp_data

    for entry in entries:
        try:
            Ec = float(entry["energy_Eh"])
        except:
            continue
        if best_E is None or Ec < best_E:
            best_E = Ec

    if best_E is None:
        print(f"[WARN] No valid complexes in gp1 for {json_file}")
        return

    dE = (best_E - (E_solute + E_solvent)) * HARTREE_TO_KCAL

    print("\n================ gp1 CHECK =====================")
    print(f"File: {json_file}")
    print("-----------------------------------------------")
    print(f"Solute energy     (Eh): {E_solute: .12f}")
    print(f"Solvent energy    (Eh): {E_solvent: .12f}")
    print(f"Best gp1 complex  (Eh): {best_E: .12f}")
    print(f"ΔE interaction (kcal/mol): {dE: .6f}")
    print("=================================================\n")


# ---------------------------------------------------------
# Extract minima for all gpX in one JSON file
# ---------------------------------------------------------
def extract_gp_minima(json_file, check_gp1=False):
    try:
        data = json.load(open(json_file))
    except Exception as e:
        print(f"[ERROR] Could not read {json_file}: {e}")
        return []

    try:
        E_solute = float(data["solute"]["energy_Eh"])
        E_solvent = float(data["solvent"]["energy_Eh"])
    except:
        print(f"[WARN] Missing solute/solvent in {json_file}")
        return []

    minima = []  # list of (gp_name, ΔE_kcal)

    for key, gp_data in data.items():
        if key in ("solute", "solvent"):
            continue
        if not key.startswith("gp"):
            continue

        # optional gp1 check
        if check_gp1 and key == "gp1":
            print_gp1(json_file, E_solute, E_solvent, gp_data)

        # find best complex in this gp
        best_Ec = None

        if isinstance(gp_data, dict):
            entries = gp_data.values()
        else:
            entries = gp_data

        for entry in entries:
            try:
                Ec = float(entry["energy_Eh"])
            except:
                continue
            if best_Ec is None or Ec < best_Ec:
                best_Ec = Ec

        if best_Ec is None:
            continue

        # ΔE
        dE_kcal = (best_Ec - (E_solute + E_solvent)) * HARTREE_TO_KCAL
        minima.append((key, dE_kcal))

    return minima


# ---------------------------------------------------------
# Temperature parser
# ---------------------------------------------------------
def parse_temps(tokens):
    temps = []
    for t in tokens:
        if ":" in t:
            a, b, s = map(float, t.split(":"))
            temps.extend(list(np.arange(a, b + 1e-8, s)))
        else:
            temps.append(float(t))
    return sorted(set(temps))


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Boltzmann over gp minima, optional gp1 check")
    p.add_argument("--temperature", nargs="+", required=True,
                   help="Temperatures: e.g. 300 500 OR 200:600:50")
    p.add_argument("--recursive", action="store_true")
    p.add_argument("--check_gp1", action="store_true",
                   help="Print solute/solvent/gp1 minimum energies")
    args = p.parse_args()

    # collect JSON files
    if args.recursive:
        json_files = glob.glob("**/*.json", recursive=True)
    else:
        json_files = glob.glob("*.json")

    if not json_files:
        print("No JSON files found.")
        sys.exit(1)

    temps = parse_temps(args.temperature)

    # collect all minima from all files
    all_gp_minima = []

    for f in sorted(json_files):
        minima = extract_gp_minima(f, check_gp1=args.check_gp1)
        all_gp_minima.extend(minima)

    if not all_gp_minima:
        print("No gp minima found.")
        sys.exit(1)

    # Sort energy list (for printing)
    gp_names, dEs = zip(*all_gp_minima)
    dEs = np.array(dEs)

    # Print ranked gp minima
    ranked_idx = np.argsort(dEs)

    print("\n=============== GRID-POINT MINIMA (SORTED) ===============")
    print(f"{'GP':10s} {'ΔE (kcal/mol)':>15s}")
    print("----------------------------------------------------------")
    for i in ranked_idx:
        print(f"{gp_names[i]:10s} {dEs[i]:15.6f}")
    print("==========================================================\n")

    # Boltzmann over minima for each temperature
    for T in temps:
        w = boltzmann_weights(dEs, T)
        Eavg = np.sum(w * dEs)
        print(f"T = {T:.1f} K : Boltzmann ⟨ΔE⟩ = {Eavg:.6f} kcal/mol")


if __name__ == "__main__":
    main()

