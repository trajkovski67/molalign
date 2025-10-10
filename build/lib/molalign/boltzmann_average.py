#!/usr/bin/env python3
import json
import numpy as np
import argparse
import sys

KCAL_PER_EH = 627.5095
R_KCAL = 0.0019872041  # kcal/(mol·K)

def compute_boltzmann_weights(energies_kcal, T=298.15):
    """Return normalized Boltzmann weights from energies in kcal/mol."""
    energies = np.array(energies_kcal)
    dE = energies - np.min(energies)  # relative energies
    weights = np.exp(-dE / (R_KCAL * T))
    weights /= np.sum(weights)
    return weights

def main():
    parser = argparse.ArgumentParser(
        description="Compute binding energies and Boltzmann weights from xtb_results.json"
    )
    parser.add_argument("json_file", help="xtb_results.json file")
    parser.add_argument("--T", type=float, default=298.15, help="Temperature in K (default 298.15)")
    parser.add_argument("--top", type=int, default=10, help="Number of top complexes to report (default 10)")
    args = parser.parse_args()

    with open(args.json_file, "r") as f:
        data = json.load(f)

    if "solute" not in data or "solvent" not in data:
        print("❌ JSON must contain 'solute' and 'solvent' keys.", file=sys.stderr)
        sys.exit(1)

    E_solute = data["solute"]["energy_Eh"]
    E_solvent = data["solvent"]["energy_Eh"]

    complexes = []
    for gp_key, gp_data in data.items():
        if not gp_key.startswith("gp"):
            continue
        for conf_key, conf_data in gp_data.items():
            E_complex = conf_data.get("energy_Eh")
            if E_complex is None:
                continue
            E_bind_kcal = (E_complex - E_solute - E_solvent) * KCAL_PER_EH
            complexes.append({
                "id": f"{gp_key}_{conf_key}",
                "E_complex_Eh": E_complex,
                "E_bind_kcal": E_bind_kcal
            })

    if not complexes:
        print("⚠️ No complexes found in the JSON file.", file=sys.stderr)
        sys.exit(1)

    # Compute Boltzmann weights
    energies_kcal = [c["E_bind_kcal"] for c in complexes]
    weights = compute_boltzmann_weights(energies_kcal, args.T)

    for c, w in zip(complexes, weights):
        c["boltzmann_weight"] = w

    # Sort by descending Boltzmann weight
    complexes_sorted = sorted(complexes, key=lambda x: x["boltzmann_weight"], reverse=True)

    # Report top N
    print(f"\n📊 Boltzmann-weighted binding energies at {args.T:.2f} K")
    print(f"{'Complex':<25} {'E_bind (kcal/mol)':>18} {'Weight':>12}")
    print("-" * 60)
    for c in complexes_sorted[:args.top]:
        print(f"{c['id']:<25} {c['E_bind_kcal']:>18.4f} {c['boltzmann_weight']:>12.4e}")

    # Boltzmann-averaged binding energy
    avg_Ebind = np.sum(weights * energies_kcal)
    print("\n🔹 Boltzmann-averaged binding energy: %.4f kcal/mol" % avg_Ebind)

if __name__ == "__main__":
    main()

