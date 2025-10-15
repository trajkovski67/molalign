import os
import json
import numpy as np

def extract_interaction_energies(data):
    """
    Compute per-grid interaction energies:
    ΔE = E_solute+solvent - E_solute - E_solvent
    For each gpX, pick the minimum energy among rotations.
    """
    E_solute = data["solute"]["energy_Eh"]
    E_solvent = data["solvent"]["energy_Eh"]

    num_grids = len(data["molA_grids_xyz"])
    interaction_energies = []

    for i in range(num_grids):
        gp_key = f"gp{i}"
        if gp_key not in data:
            raise KeyError(f"{gp_key} not found in JSON.")
        gp_data = data[gp_key]

        # Minimum energy among all rotations
        E_min = min([rot["energy_Eh"] for rot in gp_data.values()])
        deltaE = E_min - E_solute - E_solvent
        interaction_energies.append(deltaE)

    return np.array(interaction_energies, dtype=float)

def partition_function_stable(energies, kB, T):
    """
    Numerically stable Boltzmann partition function
    """
    E_min = np.min(energies)
    boltz = np.exp(-(energies - E_min) / (kB * T))
    Z = np.sum(boltz)
    return Z, E_min

def free_energy_correction(energies, kB, T):
    Z, E_min = partition_function_stable(energies, kB, T)
    return E_min - kB * T * np.log(Z)

def main():
    """
    Main CLI for computing local free energy correction ΔG_loc
    """
    # Constants
    kB = 3.1668114e-6  # Hartree/K
    T = 298.15         # Kelvin

    # Find all JSON files in the current directory
    all_files = [f for f in os.listdir('.') if f.endswith('.json')]
    if not all_files:
        print("No JSON files found in the current directory.")
        return

    all_layers_energies = []

    for fname in sorted(all_files):
        with open(fname) as f:
            data = json.load(f)
        interaction_energies = extract_interaction_energies(data)
        all_layers_energies.append(interaction_energies)

    # Concatenate all layers to form full 3D cavity energies
    all_energies = np.concatenate(all_layers_energies)

    # Compute ΔG_loc
    dG_loc = free_energy_correction(all_energies, kB, T)

    print(f"Local free energy correction ΔG_loc = {dG_loc:.6f} Eh")
    print(f"ΔG_loc ≈ {dG_loc*627.509:.2f} kcal/mol")

if __name__ == "__main__":
    main()

