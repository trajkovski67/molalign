#!/usr/bin/env python3
"""
boltzmann_analysis_relative.py

Compute Boltzmann weights of TB-lite complexes using relative energies 
(E_complex - E_solute - E_solvent) in kcal/mol,
select top N complexes by weight, and visualize their geometries.
"""
import json
import numpy as np
import py3Dmol
import argparse

HARTREE_TO_KCAL = 627.5095
kB = 0.0019872041  # kcal/mol/K

def boltzmann_weights(energies, T=298.15):
    """
    Compute Boltzmann weights from a list of energies in kcal/mol.
    Returns normalized weights.
    """
    energies = np.array(energies)
    min_E = energies.min()
    delta_E = energies - min_E
    weights = np.exp(-delta_E / (kB * T))
    return weights / weights.sum()

def get_all_complexes_relative(data):
    """
    Flatten the JSON into a list of (key, relative_energy_kcal, xyz) tuples.
    E_rel = E_complex - (E_solute + E_solvent)
    """
    E_solute = float(data['solute']['energy_Eh'])
    E_solvent = float(data['solvent']['energy_Eh'])

    complexes = []
    for gp_key, gp_data in data.items():
        if gp_key in ["solute", "solvent"]:
            continue
        for complex_key, complex_data in gp_data.items():
            E_complex = float(complex_data["energy_Eh"])
            E_rel = (E_complex - (E_solute + E_solvent)) * HARTREE_TO_KCAL
            xyz = complex_data["xyz"]
            complexes.append((f"{gp_key}_{complex_key}", E_rel, xyz))
    return complexes

def visualize_complex(xyz, title="Complex"):
    """
    Show a single complex in py3Dmol.
    """
    xyz_str = ""
    for atom in xyz:
        xyz_str += f"{atom['element']} {atom['x']} {atom['y']} {atom['z']}\n"

    view = py3Dmol.view(width=400, height=400)
    view.addModel(xyz_str, 'xyz')
    view.setStyle({'stick': {}})
    view.zoomTo()
    view.setTitle(title)
    view.show()

def main(json_file, temperature=298.15, top_n=10):
    with open(json_file) as f:
        data = json.load(f)

    complexes = get_all_complexes_relative(data)
    if len(complexes) == 0:
        print("No complexes found in JSON!")
        return

    keys, energies, xyz_list = zip(*complexes)
    energies = np.array(energies)
    weights = boltzmann_weights(energies, T=temperature)

    # Select top N complexes by weight
    top_indices = np.argsort(-weights)[:top_n]

    print(f"Top {top_n} complexes by Boltzmann weight at T={temperature} K (relative energies in kcal/mol):")
    for i in top_indices:
        print(f"{keys[i]}: ΔE={energies[i]:.2f} kcal/mol, weight={weights[i]:.3f}")

    # Visualize top complexes
    for i in top_indices:
        visualize_complex(xyz_list[i], title=f"{keys[i]} (ΔE={energies[i]:.2f} kcal/mol)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Boltzmann analysis using relative energies (complex - solute - solvent)")
    parser.add_argument("json_file", help="TB-lite results JSON file")
    parser.add_argument("--temperature", type=float, default=298.15, help="Temperature in K")
    parser.add_argument("--top_n", type=int, default=10, help="Number of top complexes to visualize")
    args = parser.parse_args()
    main(args.json_file, temperature=args.temperature, top_n=args.top_n)

