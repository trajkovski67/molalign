#!/usr/bin/env python3
"""
Full molalign pipeline for two identical solvent copies using TB-Lite.
"""
  import argparse
  import os
  import subprocess
  import sys


  def run_command(cmd, cwd=None):
      print(f"\n*** Running: {' '.join(cmd)}")
      try:
          subprocess.run(cmd, cwd=cwd, check=True)
      except subprocess.CalledProcessError:
          print(f"COMMAND FAILED: {' '.join(cmd)}", file=sys.stderr)
          sys.exit(1)


  def main():
      parser = argparse.ArgumentParser(
          description="Run molalign TB-Lite pipeline with two copies of the same
  solvent.",
      )
      parser.add_argument("molA_cpcm", help="Solute CPCM file (defines grid).")
      parser.add_argument("molB_cpcm", help="Solvent CPCM file (used twice).")
      parser.add_argument(
          "--angles1",
          default="0,90,180,270",
          help="Comma-separated rotation angles for solvent copy #1.",
      )
      parser.add_argument(
          "--angles2",
          default=None,
          help="Comma-separated rotation angles for solvent copy #2 (default: same
  as --angles1).",
      )
      parser.add_argument(
          "--out",
      )
      parser.add_argument(
          "--max-procs",
          type=int,
          default=None,
          help="Maximum worker processes for TB-Lite (default: min(cpu_count(),
  8)).",
      )
      args = parser.parse_args()

      prefixA = os.path.splitext(os.path.basename(args.molA_cpcm))[0]
      prefixB = os.path.splitext(os.path.basename(args.molB_cpcm))[0]

      run_command(["cpcm-reader", args.molA_cpcm, prefixA])
      run_command(["cpcm-reader", args.molB_cpcm, prefixB])

      npzA = f"{prefixA}_data.npz"
      npzB = f"{prefixB}_data.npz"
      output_json = args.out or f"{prefixA}_vs_{prefixB}_two_tb.json"

      cmd = [
          "align-grid-tb-two-same",
          npzA,
          npzB,
          "--angles1",
          args.angles1,
          "--out",
          output_json,
      ]
      if args.angles2 is not None:
          cmd.extend(["--angles2", args.angles2])
      if args.max_procs is not None:
          cmd.extend(["--max-procs", str(args.max_procs)])

      run_command(cmd)
      print(f"\n*** Two-solvent TB-Lite pipeline completed. JSON saved to
  {output_json}")


  if __name__ == "__main__":
      main()
