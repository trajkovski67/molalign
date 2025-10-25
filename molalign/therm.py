#!/usr/bin/env python3
"""
boltzmann_thermo.py

Boltzmann analysis for multiple TB-lite JSON result files.
Computes:
- ΔE = E_complex − (E_solute + E_solvent) [kcal/mol]
- Partition function Z
- Boltzmann weights
- Average energy <E>
- Entropy S
- Solvation free energy Gsolv
- Saves top N complexes as XYZ files
"""

import json
import numpy as np
import os
from pathlib import Path

HARTREE_TO_KCAL = 627.5095
kB = 0.0019872041  # kcal/mol/K

# Atomic number → element symbol mapping
Z_TO_SYMBOL = {1:"H", 6:"C", 7:"N", 8:"O", 16:"S", 17:"Cl", 35:"Br"}

def boltzmann_weights(energies, T=298.15):
    """Compute Boltzmann weights for a list of energies."""
    energies = np.array(energies)
    min_E = energies.min()
    delta_E = energies - min_E
    weights = np.exp(-delta_E / (kB * T))
    return weights / weights.sum()

def save_xyz(xyz, filename, comment=""):
    """Save XYZ coordinates to a file."""
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
            data = json.load(fh)

        try:
            E_solute = float(data['solute']['energy_Eh'])
            E_solvent = float(data['solvent']['energy_Eh'])
        except KeyError:
            print(f"[warn] Skipping file {f}: missing solute/solvent energies")
            continue

        for gp_key, gp_data in data.items():
            if gp_key in ['solute', 'solvent']:
                continue

            # gp_data can be a list or dict
            if isinstance(gp_data, dict):
                complex_items = gp_data.items()
            elif isinstance(gp_data, list):
                complex_items = enumerate(gp_data)
            else:
                print(f"[warn] Skipping unexpected entry {gp_key} in {f}")
                continue

            for complex_key, complex_entry in complex_items:
                try:
                    E_complex = float(complex_entry['energy_Eh'])
                    deltaE_kcal = (E_complex - (E_solute + E_solvent)) * HARTREE_TO_KCAL
                    complexes.append((f"{Path(f).name}_{gp_key}_{complex_key}",
                                      deltaE_kcal,
                                      complex_entry.get('xyz', [])))
                except Exception as e:
                    none=9
                    #print(f"[warn] Skipping complex {gp_key}_{complex_key} in {f}: {e}")
    return complexes

def compute_thermo(deltaE_list, T=298.15):
    """Compute partition function, <E>, S, and Gsolv from energy list."""
    deltaE_array = np.array(deltaE_list)
    Z = np.sum(np.exp(-deltaE_array / (kB * T)))
    weights = np.exp(-deltaE_array / (kB * T)) / Z
    E_avg = np.sum(weights * deltaE_array)
    S = -kB * np.sum(weights * np.log(weights + 1e-20))  # add small number to avoid log(0)
    G_solv = -kB * T * np.log(Z)
    return Z, weights, E_avg, S, G_solv

def main(T=298.15, top_n=10, out_dir="top_complexes"):
    """Main analysis for all JSON files in current directory."""
    json_files = list(Path(".").glob("*.json"))
    if not json_files:
        print("No JSON files found in the current directory.")
        return

    os.makedirs(out_dir, exist_ok=True)
    complexes = collect_complexes(json_files)
    if not complexes:
        print("No complexes found.")
        return

    keys, deltaE_list, xyz_list = zip(*complexes)
    Z, weights, E_avg, S, G_solv = compute_thermo(deltaE_list, T=T)

    print(f"\nPartition function Z = {Z:.6f}")
    print(f"Boltzmann-weighted <E> = {E_avg:.3f} kcal/mol")
    print(f"Entropy S = {S:.3f} kcal/mol/K")
    print(f"Solvation free energy G_solv = {G_solv:.3f} kcal/mol")

    # Save top N complexes by weight
    #top_indices = np.argsort(-weights)[:top_n]
    #print(f"\nTop {top_n} complexes by Boltzmann weight:")
    #for i in top_indices:
        #key = keys[i]
        #deltaE = deltaE_list[i]
        #weight = weights[i]
        #print(f"{key}: ΔE={deltaE:.3f}, weight={weight:.3f}")
        #xyz_file = os.path.join(out_dir, f"{key}.xyz")
        #save_xyz(xyz_list[i], xyz_file, comment=f"{key} ΔE={deltaE:.3f} kcal/mol")
        #print(f"Saved XYZ: {xyz_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Boltzmann analysis + thermodynamics from TB-lite JSON files")
    parser.add_argument("--temperature", type=float, default=298.15, help="Temperature in K")
    parser.add_argument("--top_n", type=int, default=10, help="Number of top complexes to save")
    parser.add_argument("--out_dir", default="top_complexes", help="Directory to save XYZ files")
    args = parser.parse_args()
    main(T=args.temperature, top_n=args.top_n, out_dir=args.out_dir)

