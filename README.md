**molalign** is a toolkit for automating molecular alignment, CPCM data parsing, and xTB result extraction.
## Installation
```bash
pip install -e .

usage: molalign-run [-h] [--angles ANGLES] [--charge CHARGE] molA_cpcm molB_cpcm
where ANGLES is csv of angles (e.g. 0,45,90) the solvent rotates around the aligned normal for each grid point.
mol1 - solute, mol2 - solvent

to visualize: plot-grid-energy mol1.npz xtb_results.json

