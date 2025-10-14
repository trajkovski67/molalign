#!/usr/bin/env python3
"""
boltzmann_multi.py

Boltzmann analysis for multiple TB-lite JSON result files.
- Computes ΔE = E_complex − (E_solute + E_solvent) [kcal/mol]
- Computes Boltzmann weights over all complexes in all files
- Saves top N complexes as XYZ files
"""

import json
import numpy as np
import os
import sys

HARTREE_TO_KCAL = 627.5095
kB = 0.0019872041  # kcal/mol/K

# Atomic number → element symbol mapping
Z_TO_SYMBOL = {1:"H", 6:"C", 7:"N", 8:"O", 16:"S", 17:"Cl", 35:"Br"}

def boltzmann_weights(energies, T=298.15):
    energies = np.array(energies)
    min_E = energies.min()
    delta_E = energies - min_E
    weights = np.exp(-delta_E / (kB * T))
    return weights / weights.sum()

def save_xyz(xyz, filename, comment=""):
    with open(filename, "w") as fh:
        fh.write(f"{len(xyz)}\n")
        fh.write(comment + "\n")
        for atom in xyz:
            Z = int(atom['element'])
            symbol = Z_TO_SYMBOL.get(Z, str(Z))
            fh.write(f"{symbol} {atom['x']:.6f} {atom['y']:.6f} {atom['z']:.6f}\n")

def collect_complexes(json_files):
    """Return list of (key, ΔE, xyz) tuples from multiple JSONs"""
    complexes = []
    for f in json_files:
        with open(f) as fh:
            data = json.load(fh)  # <- fixed

        # Read solute and solvent energies
        try:
            E_solute = float(data['solute']['energy_Eh'])
            E_solvent = float(data['solvent']['energy_Eh'])
        except KeyError:
            print(f"Skipping file {f}: missing solute/solvent energies")
            continue

        for gp_key, gp_data in data.items():
            if gp_key in ['solute', 'solvent']:
                continue

            # gp_data may be a dict or a list
            if isinstance(gp_data, dict):
                complex_items = gp_data.items()
            elif isinstance(gp_data, list):
                complex_items = enumerate(gp_data)
            else:
                print(f"Skipping unexpected entry {gp_key} in {f}")
                continue

            for complex_key, complex_entry in complex_items:
                try:
                    E_complex = float(complex_entry['energy_Eh'])
                    deltaE_kcal = (E_complex - (E_solute + E_solvent)) * HARTREE_TO_KCAL
                    complexes.append((f"{os.path.basename(f)}_{gp_key}_{complex_key}",
                                      deltaE_kcal,
                                      complex_entry.get('xyz', [])))
                except Exception as e:
                    x=1
                    #print(f"Skipping complex {gp_key}_{complex_key} in {f}: {e}")
    return complexes

def main(json_files, T=298.15, top_n=10, out_dir="top_complexes"):
    if not json_files:
        print("No JSON files provided.")
        sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)

    complexes = collect_complexes(json_files)
    if not complexes:
        print("No complexes found in the input JSON files!")
        sys.exit(1)

    keys, deltaE_list, xyz_list = zip(*complexes)
    deltaE_list = np.array(deltaE_list)
    weights = boltzmann_weights(deltaE_list, T=T)
    E_avg = np.sum(deltaE_list * weights)

    print(f"\nBoltzmann-weighted average interaction energy at T={T} K: {E_avg:.3f} kcal/mol")

    # Top N complexes
    top_indices = np.argsort(-weights)[:top_n]
    print(f"\nTop {top_n} complexes by Boltzmann weight:")
    for i in top_indices:
        key = keys[i]
        deltaE = deltaE_list[i]
        weight = weights[i]
        print(f"{key}: ΔE={deltaE:.3f}, weight={weight:.3f}")
        xyz_file = os.path.join(out_dir, f"{key}.xyz")
        save_xyz(xyz_list[i], xyz_file, comment=f"{key} ΔE={deltaE:.3f} kcal/mol")
        print(f"Saved XYZ: {xyz_file}")

def cli_main():
    import argparse
    parser = argparse.ArgumentParser(description="Boltzmann analysis of multiple TB-lite JSON results")
    parser.add_argument("json_files", nargs="+", help="TB-lite JSON result files")
    parser.add_argument("--temperature", type=float, default=298.15, help="Temperature in K")
    parser.add_argument("--top_n", type=int, default=10, help="Number of top complexes to save")
    parser.add_argument("--out_dir", default="top_complexes", help="Directory to save XYZ files")
    args = parser.parse_args()

    main(args.json_files, T=args.temperature, top_n=args.top_n, out_dir=args.out_dir)

if __name__ == "__main__":
    cli_main()

