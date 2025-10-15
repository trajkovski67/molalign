#!/usr/bin/env python3
import os
import json
import subprocess
import numpy as np
import pandas as pd
import argparse
import re

# Constants
HARTREE_TO_KCAL = 627.509
R_kcal = 0.0019872041
T = 298.15  # K

def write_xyz(atom_list, filename):
    """Write xyz file from list of dicts: [{element, x, y, z}, ...]"""
    with open(filename, "w") as f:
        f.write(f"{len(atom_list)}\n\n")
        for atom in atom_list:
            elem = atom.get("element")
            if isinstance(elem, (int, float)):
                elem = str(int(elem))
            f.write(f"{elem} {atom['x']} {atom['y']} {atom['z']}\n")

def run_xtb(xyz_file, charge=0, solvent=None):
    """Run xTB Hessian calculation and extract total free energy"""
    base = os.path.splitext(xyz_file)[0]
    cmd = ["xtb", xyz_file, "--hess", "--chrg", str(charge)]
    if solvent:
        cmd += ["--alpb", solvent]

    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout + result.stderr
    with open(f"{base}.xtbout", "w") as f:
        f.write(output)

    # Extract the total free energy robustly
    G_value = None
    for line in output.splitlines():
        if "total free energy" in line.lower():
            match = re.search(r"(-?\d+\.\d+)", line)
            if match:
                G_value = float(match.group(1))
                break

    if G_value is None:
        raise RuntimeError(f"Cannot find G in xTB output for {xyz_file}")
    return G_value

def cluster_xtb_dG_json(json_dir=".", top_n=5, out_csv="dG_xtb_summary.csv", charge=0):
    # Collect all gp complexes from all JSON files
    all_gp = []
    json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]
    if not json_files:
        raise RuntimeError(f"No JSON files found in {json_dir}")

    for jf in json_files:
        path = os.path.join(json_dir, jf)
        with open(path) as f:
            data = json.load(f)

        solute = data.get("solute")
        solvent = data.get("solvent")
        gp_entries = {k: v for k, v in data.items() if k.startswith("gp")}
        if not gp_entries:
            continue

        # Flatten gp entries (b0_rot0, b1_rot90, etc.)
        for gp_name, gp_data in gp_entries.items():
            for sub_name, sub_data in gp_data.items():
                energy = None
                for key in ["energy_Eh", "energy", "E", "Etot", "energyEh"]:
                    if key in sub_data:
                        energy = sub_data[key]
                        break
                if energy is None:
                    continue
                all_gp.append({
                    "file": jf,
                    "gp": f"{gp_name}_{sub_name}",
                    "energy_Eh": energy,
                    "xyz": sub_data["xyz"],
                    "solute": solute,
                    "solvent": solvent
                })

    if not all_gp:
        raise RuntimeError("No grid-point complexes found in JSON files.")

    # Pick top_n lowest-energy complexes globally
    sorted_all = sorted(all_gp, key=lambda x: x["energy_Eh"])
    best_gp = sorted_all[:top_n]

    results = []
    for entry in best_gp:
        base = f"{entry['file'].replace('.json','')}_{entry['gp']}"
        complex_file = f"{base}_complex.xyz"
        solute_file = f"{base}_solute.xyz"
        solvent_file = f"{base}_solvent.xyz"

        write_xyz(entry["xyz"], complex_file)
        write_xyz(entry["solute"]["xyz"], solute_file)
        write_xyz(entry["solvent"]["xyz"], solvent_file)

        print(f"Running xTB for {base} ...")
        try:
            G_complex_gas = run_xtb(complex_file, charge=charge)
            G_complex_solv = run_xtb(complex_file, charge=charge, solvent="water")
            G_solute_gas = run_xtb(solute_file, charge=charge)
            G_solvent_gas = run_xtb(solvent_file, charge=0)
        except RuntimeError as e:
            print(f"⚠️ Skipping {base} due to xTB error: {e}")
            continue

        dG_gas = (G_complex_gas - G_solute_gas - G_solvent_gas) * HARTREE_TO_KCAL
        dG_solv = (G_complex_solv - G_complex_gas) * HARTREE_TO_KCAL
        dG_final = dG_gas + dG_solv + 2.0

        results.append({
            "file": entry["file"],
            "gp": entry["gp"],
            "G_complex_gas": G_complex_gas,
            "G_complex_solv": G_complex_solv,
            "G_solute_gas": G_solute_gas,
            "G_solvent_gas": G_solvent_gas,
            "dG_gas_kcal": dG_gas,
            "dG_solv_kcal": dG_solv,
            "dG_final_kcal": dG_final
        })

    if not results:
        raise RuntimeError("No complexes could be processed with xTB.")

    df = pd.DataFrame(results)
    minE = df["dG_final_kcal"].min()
    weights = np.exp(-(df["dG_final_kcal"] - minE) / (R_kcal * T))
    df["boltz_weight"] = weights / np.sum(weights)
    boltz_avg = np.sum(df["dG_final_kcal"] * df["boltz_weight"])

    # Add summary row
    df.loc[len(df)] = ["---"] * (len(df.columns) - 1) + [boltz_avg]
    df.to_csv(out_csv, index=False)
    print(f"\n✅ Done. xTB ΔG summary saved to {out_csv}")
    print(f"📊 Boltzmann-averaged ΔG_final = {boltz_avg:.2f} kcal/mol")


def main():
    parser = argparse.ArgumentParser(description="Compute ΔGsolv and ΔGgas for top 5 complexes across all JSON files")
    parser.add_argument("--json-dir", default=".", help="Directory with JSON files (default: current)")
    parser.add_argument("--top", type=int, default=5, help="Top N complexes globally by energy")
    parser.add_argument("--out", default="dG_xtb_summary.csv", help="Output CSV file")
    parser.add_argument("--charge", type=int, default=0, help="Charge of the solute/complex")
    args = parser.parse_args()

    cluster_xtb_dG_json(json_dir=args.json_dir, top_n=args.top, out_csv=args.out, charge=args.charge)


if __name__ == "__main__":
    main()

