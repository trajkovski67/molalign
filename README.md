molalign is a toolkit for automating molecular alignment, parsing CPCM data, and extracting xTB results. It provides utilities for molecular surface scanning, and energy map visualization.

Installation:
pip install -e .

Usage:
molalign-run [-h] [--angles ANGLES] [--charge CHARGE] molA_cpcm molB_cpcm
molA_cpcm is the solute CPCM file, molB_cpcm is the solvent CPCM file.
--angles specifies rotation angles (e.g. 0,45,90) for solvent rotations around the aligned normal for each grid point.
--charge sets the molecular charge value.

Extended Run (3D Scan):
molalign-extended [-h] [--out OUT] [--angles ANGLES] [--charge CHARGE] [--shifts SHIFTS [SHIFTS ...]] solute_cpcm solvent_cpcm
Performs a 3D scan along the normal direction. The SHIFTS are given in angstroms (e.g. -0.5 0.0 0.5 1.0). Negative shifts are valid but may lead to localized spheres. After positive shifts, the union of grid points is not taken, which can distort the cavity instead of scaling it.

Visualization and Analysis Tools:
plot-grid-energy mol1.npz xtb_results.json
Plots energy grids based on CPCM and xTB results data.

slice [-h] [--tol TOL] [--sigma SIGMA] args [args ...]
Generates 2D slices of the interaction energy map to visualize local variations.

boltzmann [-h] [--temperature TEMPERATURE] [--top_n TOP_N] [--out_dir OUT_DIR] json_files [json_files ...] polish
Performs Boltzmann-weighted averaging of energy distributions with optional filtering by top conformations and output directory specification.

Example Workflow:
Step 1: Align molecules
molalign-run solute.cpcm solvent.cpcm --angles 0,45,90
Step 2: Perform 3D scan
molalign-extended solute.cpcm solvent.cpcm --angles 0,90,180 --shifts -0.5 0.0 0.5 1.0
Step 3: Visualize energy grids and slices
plot-grid-energy mol1.npz xtb_results.json
slice xtb_results.json --sigma 0.1
Step 4: Compute Boltzmann averages
boltzmann --temperature 298 xtb_results_*.json

