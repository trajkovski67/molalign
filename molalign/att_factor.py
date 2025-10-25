"""
molalign.att_factor
-------------------

Computes charge–density–based attenuation factors from .cpcm files.

Usage:
    att_factor <file.cpcm> <alpha> <beta> <charge>

Example:
    att_factor water_anion.cpcm 2.0 3.0 -1
"""

import math
import sys
from pathlib import Path

def parse_cpcm_file(filename: str):
    """Extract CPCM Volume and Area from file."""
    volume = None
    area = None
    with open(filename, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[-1].lower().startswith("volume"):
                try:
                    volume = float(parts[0])
                except ValueError:
                    continue
            elif parts[-1].lower().startswith("area"):
                try:
                    area = float(parts[0])
                except ValueError:
                    continue
    if volume is None or area is None:
        raise ValueError(f"Could not find 'Volume' and 'Area' entries in {filename}")
    return volume, area


def compute_effective_radius(volume_bohr3: float) -> float:
    """Compute effective radius in Å from CPCM volume (bohr³)."""
    bohr_to_ang = 0.529177
    V_ang3 = volume_bohr3 * (bohr_to_ang ** 3)
    return (3 * V_ang3 / (4 * math.pi)) ** (1 / 3)


def attenuation(alpha: float, beta: float, q: float, reff: float) -> float:
    """Return attenuation factor A = 1 / (1 + α(|q|/Reff)^β)."""
    return 1.0 / (1.0 + alpha * (abs(q) / reff) ** beta)


def main():
    if len(sys.argv) < 5:
        print("Usage: att_factor <file.cpcm> <alpha> <beta> <charge>")
        sys.exit(1)

    filename = Path(sys.argv[1])
    alpha = float(sys.argv[2])
    beta = float(sys.argv[3])
    charge = float(sys.argv[4])

    if not filename.exists():
        sys.exit(f"Error: file '{filename}' not found")

    volume, area = parse_cpcm_file(filename)
    reff = compute_effective_radius(volume)
    A = attenuation(alpha, beta, charge, reff)

    print(f"\nFile: {filename}")
    print(f"Volume (bohr³): {volume:.3f}")
    print(f"Area (bohr²): {area:.3f}")
    print(f"Effective radius R_eff = {reff:.3f} Å")
    print(f"Attenuation factor A = {A:.5f}  (α={alpha}, β={beta}, q={charge})\n")


if __name__ == "__main__":
    main()

