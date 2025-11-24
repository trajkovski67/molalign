#!/usr/bin/env python3
"""
molalign.element_scan
---------------------
Adaptive M–O dissociation scan with optional solvent models.

• Coarse scan (0.1 Å) to find rough minimum.
• Detect slope change (ΔE/Δr sign change).
• Fine resample (0.02 Å) near minimum and quadratic-fit refinement.
• Supports any implicit solvation model in ORCA (CPCM, SMD, etc.).
• Cleans intermediate files, saves .dat, .json, and lowest .xyz.
"""

import subprocess, json
import numpy as np
from pathlib import Path

# --- constants ---
HARTREE_TO_KCAL = 627.509474
THRESH_KCAL = 0.1
THRESH_EH = THRESH_KCAL / HARTREE_TO_KCAL
STOP_DISTANCE_MIN = 8
ORCA_CMD = "orca"

# --- reference water geometry ---
WATER_RAW = np.array([
    [-1.80646061197347, 4.63817703015447,  0.62399693269628],
    [-1.76753267988609, 3.79897080586617,  0.15770859148542],
    [-0.86504870814043, 3.73932216397935, -0.16662152418169],
])

# =============================================================
# Geometry and I/O helpers
# =============================================================

def orient_water_O_facing(coords):
    """Rotate H2O so O faces +z."""
    H1, O, H2 = coords
    H1 -= O; H2 -= O; O[:] = 0.0
    bis = 0.5 * (H1 + H2); bis /= np.linalg.norm(bis)
    target = np.array([0.0, 0.0, -1.0])
    v = np.cross(bis, target); s = np.linalg.norm(v); c = np.dot(bis, target)
    if s < 1e-8: R = np.eye(3)
    else:
        vx = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
        R = np.eye(3)+vx+vx@vx*((1-c)/(s**2))
    coords = coords @ R.T
    coords[:,1] -= np.mean(coords[:,1])
    return coords

def write_input(metal, charge, mult, dist, coords, outdir, method_line):
    inp = outdir / f"{metal}_z{dist:.2f}.inp"
    with inp.open("w") as f:
        f.write(f"{method_line}\n")
        f.write(f"*xyz {charge} {mult}\n")
        f.write("O   0.000000   0.000000   0.000000\n")
        for (x,y,z) in coords[[0,2]]:
            f.write(f"H   {x: .6f}  {y: .6f}  {z: .6f}\n")
        f.write(f"{metal:2s}   0.000000   0.000000   {dist:.6f}\n*\n")
    return inp

def run_orca(inp):
    out = inp.with_suffix(".out")
    subprocess.run([ORCA_CMD, str(inp)], stdout=out.open("w"),
                   stderr=subprocess.STDOUT, check=False)
    for line in out.read_text().splitlines():
        if "FINAL SINGLE POINT ENERGY" in line:
            return float(line.split()[-1])
    return None

def save_xyz(path, geom, metal):
    with open(path, "w") as f:
        f.write("4\nLowest-energy complex (O facing metal)\n")
        f.write(f"O  {geom['O'][0]:10.6f} {geom['O'][1]:10.6f} {geom['O'][2]:10.6f}\n")
        f.write(f"H  {geom['H1'][0]:10.6f} {geom['H1'][1]:10.6f} {geom['H1'][2]:10.6f}\n")
        f.write(f"H  {geom['H2'][0]:10.6f} {geom['H2'][1]:10.6f} {geom['H2'][2]:10.6f}\n")
        f.write(f"{metal:2s}  {geom['metal'][0]:10.6f} {geom['metal'][1]:10.6f} {geom['metal'][2]:10.6f}\n")

def cleanup(folder, keep):
    for f in folder.iterdir():
        if f not in keep:
            f.unlink(missing_ok=True)

# =============================================================
# Main adaptive scan
# =============================================================

def main(element="Li", charge=1, mult=1,
         solvation_model="CPCM", solvent="water"):

    # Build method line
    if solvation_model.lower() in {"none", "gas", "vacuum"}:
        method_line = "! wb97x-3c TightSCF"
    else:
        method_line = f"! wb97x-3c TightSCF {solvation_model}({solvent})"

    coords_H2O = orient_water_O_facing(WATER_RAW.copy())
    workdir = Path(f"{element}_scan")
    workdir.mkdir(exist_ok=True)

    results = []
    dat_path = workdir / f"{element}_scan.dat"
    fdat = dat_path.open("w")
    fdat.write(f"# Adaptive {element}–O scan ({method_line})\n# dist(A)     E(Ha)\n")

    dist = 1.1
    step = 0.1
    last_E = None
    dE = -999
    print(f"=== {element}–O adaptive scan ({method_line}) ===\n")

    # --- coarse adaptive scan ---
    while last_E is None or dist <= STOP_DISTANCE_MIN or dE < -THRESH_EH:
        inp = write_input(element, charge, mult, dist, coords_H2O, workdir, method_line)
        E = run_orca(inp)
        fdat.write(f"{dist:8.3f}  {E if E is not None else float('nan'): .12f}\n")
        results.append({"dist_A": dist, "energy_Eh": E})
        if last_E is not None and E is not None:
            dE = E - last_E
            print(f"dist={dist:.2f}  ΔE={dE*HARTREE_TO_KCAL: .3f} kcal/mol")
            if dist > STOP_DISTANCE_MIN and dE > -THRESH_EH:
                print(f"Stop condition reached at {dist:.2f} Å\n")
                break
        last_E = E
        dist += step

    fdat.close()

    # --- fine re-sampling around minimum ---
    energies = np.array([r["energy_Eh"] for r in results])
    dists = np.array([r["dist_A"] for r in results])
    slopes = np.diff(energies) / np.diff(dists)
    fine_results = []
    for i in range(1, len(slopes)):
        if slopes[i-1] < 0 < slopes[i]:
            r1, r2 = dists[i-1], dists[i]
            print(f"Refining minimum region between {r1:.2f} and {r2:.2f} Å ...")
            fine_pts = np.linspace(r1, r2, 6)[1:-1]
            for rf in fine_pts:
                inp = write_input(element, charge, mult, rf, coords_H2O, workdir, method_line)
                E = run_orca(inp)
                fine_results.append({"dist_A": rf, "energy_Eh": E})
                fdat = open(dat_path, "a")
                fdat.write(f"{rf:8.3f}  {E if E is not None else float('nan'): .12f}\n")
                fdat.close()
            break

    # Merge and sort results
    all_results = results + fine_results
    all_results.sort(key=lambda x: x["dist_A"])

    # --- quadratic fit near min ---
    valid = [r for r in all_results if r["energy_Eh"] is not None]
    E = np.array([r["energy_Eh"] for r in valid])
    R = np.array([r["dist_A"] for r in valid])
    i_min = np.argmin(E)
    window = slice(max(0, i_min-1), min(len(E), i_min+2))
    if len(E[window]) == 3:
        coeffs = np.polyfit(R[window], E[window], 2)
        a, b, c = coeffs
        r_fit = -b / (2 * a)
        e_fit = np.polyval(coeffs, r_fit)
        print(f"Quadratic refined minimum: {r_fit:.3f} Å, {e_fit:.8f} Eh")
    else:
        r_fit, e_fit = R[i_min], E[i_min]

    # --- save lowest geometry ---
    geom = {
        "O": [0,0,0],
        "H1": coords_H2O[0].tolist(),
        "H2": coords_H2O[2].tolist(),
        "metal": [0,0,r_fit]
    }
    xyz_path = workdir / f"{element}_lowest.xyz"
    save_xyz(xyz_path, geom, element)

    # --- save JSON ---
    json_path = workdir / f"{element}_scan.json"
    with json_path.open("w") as f:
        json.dump({
            "element": element,
            "charge": charge,
            "mult": mult,
            "solvation_model": solvation_model,
            "solvent": solvent,
            "method": "wB97X-3c",
            "refined_min": {"dist_A": r_fit, "energy_Eh": e_fit},
            "results": all_results
        }, f, indent=2)

    cleanup(workdir, [dat_path, json_path, xyz_path])
    print(f"\n>>> Results → {json_path}, {dat_path}, {xyz_path}")

# =============================================================
# CLI entry point
# =============================================================

def cli_main():
    import argparse
    p = argparse.ArgumentParser(description="Adaptive M–O scan with refinement and solvation models.")
    p.add_argument("--element", required=True)
    p.add_argument("--charge", type=int, required=True)
    p.add_argument("--mult", type=int, required=True)
    p.add_argument("--solvation-model", default="CPCM")
    p.add_argument("--solvent", default="water")
    args = p.parse_args()
    main(args.element, args.charge, args.mult, args.solvation_model, args.solvent)

if __name__ == "__main__":
    cli_main()

