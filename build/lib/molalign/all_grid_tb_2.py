#!/usr/bin/env python3
"""
Align two copies of the same solvent to each solute grid point and evaluate TB-
Lite energies.
"""
  import argparse
  import json
  import os
  from multiprocessing import Pool, cpu_count

  import numpy as np
  from tblite.interface import Structure, Calculator

  from .allign_many_to_one_tb import (
      load_npz_file,
      rotation_matrix_from_vectors,
      rotate_around_axis,
      atoms_to_xyz_dict,
  )

  def _tb_energy_from_npz(npz_file):
      atoms = np.load(npz_file, allow_pickle=True)["atoms"]
      coords = atoms[:, :3]
      numbers = atoms[:, 3].astype(int)

      mol = Structure(numbers=numbers, positions=coords)
      calc = Calculator(method="GFN2-xTB", structure=mol)
      result = calc.singlepoint()

      energy = float(result["energy"])
      gradients = np.asarray(result["gradients"])
      grad_norm = float(np.linalg.norm(gradients))
      return energy, grad_norm, atoms_to_xyz_dict(coords, numbers)

  def _align_copy(atoms, grids, normals, idx, anchor, target):
      rot = rotation_matrix_from_vectors(normals[idx], target)
      atoms_rot = atoms @ rot.T
      grids_rot = grids @ rot.T
      normals_rot = normals @ rot.T

      translation = anchor - grids_rot[idx]
      atoms_rot += translation
      grids_rot += translation
      return atoms_rot, grids_rot, normals_rot

  def _tasks(solute, solvent, angles1, angles2):
      atomsA, ZA, _, gridsA, _, _, normalsA = solute
      atomsB, ZB, _, gridsB, _, _, normalsB = solvent

      tasks = []
      for idxA, (pA, nA) in enumerate(zip(gridsA, normalsA)):
          gp_key = f"gp{idxA}"
          target_axis = -nA
          if np.linalg.norm(target_axis) < 1e-8:
              target_axis = np.array([0.0, 0.0, 1.0])

          for idxB1 in range(len(gridsB)):
              b1_atoms_aligned, b1_grids_aligned, b1_normals_aligned = _align_copy(
                  atomsB, gridsB, normalsB, idxB1, pA, target_axis
              )

              for angle1 in angles1:
                  b1_atoms_rot = rotate_around_axis(b1_atoms_aligned - pA,
  target_axis, angle1) + pA

                  for idxB2 in range(len(gridsB)):
                      b2_atoms_aligned, b2_grids_aligned, b2_normals_aligned =
  _align_copy(
                          atomsB, gridsB, normalsB, idxB2, pA, target_axis
                      )

                      for angle2 in angles2:
                          b2_atoms_rot = rotate_around_axis(b2_atoms_aligned - pA,
  target_axis, angle2) + pA

                          atoms_combined = np.vstack((atomsA, b1_atoms_rot,
  b2_atoms_rot))
                          numbers_combined = np.hstack((ZA, ZB, ZB))
                          key = (
                              f"b{idxB1}_rot{int(angle1)}"
                              f"__c{idxB2}_rot{int(angle2)}"
                          )
                          tasks.append((gp_key, key, atoms_combined,
  numbers_combined))
      return tasks

  def _worker(task):
      gp_key, key, coords, numbers = task
      mol = Structure(numbers=numbers, positions=coords)
      calc = Calculator(method="GFN2-xTB", structure=mol)
      result = calc.singlepoint()

      energy = float(result["energy"])
      gradients = np.asarray(result["gradients"])
      grad_norm = float(np.linalg.norm(gradients))
      xyz = atoms_to_xyz_dict(coords, numbers)
      return gp_key, key, energy, grad_norm, xyz

  def main(fileA, fileB, angles1, angles2, output_json=None, max_procs=None):
      fileA = os.path.abspath(fileA)
      fileB = os.path.abspath(fileB)

      baseA = os.path.splitext(os.path.basename(fileA))[0]
      baseB = os.path.splitext(os.path.basename(fileB))[0]
      if output_json is None:
          output_json = f"{baseA}_vs_{baseB}_two_tb.json"

      solute = load_npz_file(fileA)
      solvent = load_npz_file(fileB)

      results = {}
      energy, grad_norm, xyz = _tb_energy_from_npz(fileA)
      results["solute"] = {"energy_Eh": energy, "grad_norm_Eh_per_a": grad_norm,
  "xyz": xyz}

      energy, grad_norm, xyz = _tb_energy_from_npz(fileB)
      results["solvent"] = {"energy_Eh": energy, "grad_norm_Eh_per_a": grad_norm,
  "xyz": xyz}

      tasks = _tasks(solute, solvent, angles1, angles2)

      if max_procs is None:
          max_procs = min(cpu_count(), 8)
      with Pool(max_procs) as pool:
          for gp_key, key, energy, grad_norm, xyz in pool.imap_unordered(_worker,
  tasks):
              results.setdefault(gp_key, {})[key] = {
                  "energy_Eh": energy,
                  "grad_norm_Eh_per_a": grad_norm,
                  "xyz": xyz,
              }

      with open(output_json, "w") as handle:
          json.dump(results, handle, indent=2)

      print(f"*** Saved two-solvent TB-Lite results to {output_json}")
      print(f"*** Grid points processed: {len([k for k in results if
  k.startswith('gp')])}")

  def cli_main():
      parser = argparse.ArgumentParser(
          description="Align two copies of a solvent molecule to each solute grid
  point and compute TB-Lite energies."
      )
      parser.add_argument("fileA", help="Solute NPZ file.")
      parser.add_argument("fileB", help="Solvent NPZ file (used twice).")
      parser.add_argument(
          "--angles1",
          default="0,90,180,270",
          help="Comma-separated rotation angles for copy #1 (default:
  0,90,180,270).",
      )
      parser.add_argument(
          "--angles2",
          default=None,
          help="Comma-separated rotation angles for copy #2 (default: same as
  --angles1).",
      )
      parser.add_argument(
          "--out",
          default=None,
          help="Output JSON filename (default: <solute>_vs_<solvent>_two_tb.json).",
      )
      parser.add_argument(
          "--max-procs",
          type=int,
          default=None,
          help="Maximum worker processes (default: min(cpu_count(), 8)).",
      )
      args = parser.parse_args()

      angles1 = [float(a) for a in args.angles1.split(",") if a.strip()]
      if not angles1:
          angles1 = [0.0]
      if args.angles2 is None:
          angles2 = angles1
      else:
          angles2 = [float(a) for a in args.angles2.split(",") if a.strip()]
          if not angles2:
              angles2 = [0.0]

      main(args.fileA, args.fileB, angles1, angles2, args.out, args.max_procs)

  if __name__ == "__main__":
      cli_main()
