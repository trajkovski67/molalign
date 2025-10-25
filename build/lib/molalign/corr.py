#!/usr/bin/env python3
"""
molalign.corr
-------------
Compute implicit–explicit corrected solute–solvent interaction energies
for all JSON grid files in the current directory, and perform a global
Boltzmann average across all grids (e.g. concentric shells).

Usage:
    $ corr
"""

import json, csv, math, os, glob
import numpy as np
from math import pi, tanh
from collections import defaultdict

# =========================
# Constants
# =========================
K_COULOMB = 332.06371          # kcal·Å/(mol·e²)
HARTREE_TO_KCAL = 627.509474
R_kcal = 0.0019872041          # kcal/(mol·K)
EPS_R = 78.5
T_DEFAULT = 298.15

# =========================
# Element data
# =========================
Z_TO_EL = {1:"H",6:"C",7:"N",8:"O",9:"F",15:"P",16:"S",17:"Cl",35:"Br",53:"I"}
BONDI = {"H":1.20,"C":1.70,"N":1.55,"O":1.52,"F":1.47,
         "P":1.80,"S":1.80,"Cl":1.75,"Br":1.85,"I":1.98}

# OBC-II parameters
OBC_ALPHA, OBC_BETA, OBC_GAMMA = 1.0, 0.8, 4.85
DESCREEN_SCALE = 0.8

# =========================
# Geometry / GB helpers
# =========================
def sphere_overlap(a,b,d):
    if d>=a+b: return 0.0
    if d<=abs(a-b): return (4/3)*pi*min(a,b)**3
    return (pi*(a+b-d)**2*(d**2 + 2*d*(a+b) - 3*(a-b)**2))/(12*d)

def obc_born_radii(coords, elements):
    coords=np.asarray(coords,float)
    n=len(coords)
    r_intr=np.array([BONDI.get(el,1.7) for el in elements])
    V=(4/3)*pi*r_intr**3
    psi=np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i==j: continue
            d=np.linalg.norm(coords[i]-coords[j])
            psi[i]+=sphere_overlap(r_intr[i],DESCREEN_SCALE*r_intr[j],d)/V[i]
    t=np.tanh(OBC_ALPHA*psi - OBC_BETA*psi**2 + OBC_GAMMA*psi**3)
    return r_intr/(1.0-t)

def f_gb(r,a_i,a_j):
    return math.sqrt(r*r + a_i*a_j*math.exp(-r*r/(4*a_i*a_j)))

def gb_cross_energy(sol_xyz,sol_q,sol_alpha,solv_xyz,solv_q,solv_alpha,eps_r=EPS_R):
    scale=-(1.0-1.0/eps_r)*K_COULOMB
    e=0.0
    for i in range(len(sol_q)):
        for j in range(len(solv_q)):
            rij=np.linalg.norm(sol_xyz[i]-solv_xyz[j])
            fij=f_gb(rij,sol_alpha[i],solv_alpha[j])
            e+=sol_q[i]*solv_q[j]/fij
    return scale*e

# =========================
# Boltzmann helpers
# =========================
def boltzmann_weights(energies,T=T_DEFAULT):
    E=np.array(energies,float)
    Emin=np.min(E)
    beta=1.0/(R_kcal*T)
    w_unnorm=np.exp(-beta*(E-Emin))
    Z=np.sum(w_unnorm)
    return w_unnorm/Z, Z, Emin

def boltzmann_average(values,energies,T=T_DEFAULT):
    w,Z,Emin=boltzmann_weights(energies,T)
    v=np.array(values,float)
    mean=float(np.sum(w*v))
    var=float(np.sum(w*(v-mean)**2))
    return mean, math.sqrt(var), {"Z":float(Z),"Emin_kcal":float(Emin)}

# =========================
# Core per-file processor
# =========================
def process_single_json(json_in, solute_charges, solvent_charges=None, temperature_K=T_DEFAULT):
    with open(json_in) as f: data=json.load(f)
    solute=data["solute"]; solvent=data["solvent"]

    solute_xyz=np.array([[a["x"],a["y"],a["z"]] for a in solute["xyz"]])
    solvent_xyz=np.array([[a["x"],a["y"],a["z"]] for a in solvent["xyz"]])
    solute_el=[Z_TO_EL.get(int(a["element"]),str(a["element"])) for a in solute["xyz"]]
    solvent_el=[Z_TO_EL.get(int(a["element"]),str(a["element"])) for a in solvent["xyz"]]

    solute_q=np.asarray(solute_charges,float)
    if len(solute_q)!=len(solute_el):
        raise ValueError(f"{json_in}: charge count ({len(solute_q)}) != atom count ({len(solute_el)})")

    if solvent_charges is None:
        if len(solvent_el)==3 and set(solvent_el)>={"O","H"}:
            solvent_q=np.array([-0.834,0.417,0.417])
        else:
            raise ValueError("Provide solvent_charges for non-water solvent")
    else:
        solvent_q=np.asarray(solvent_charges,float)

    solute_alpha=obc_born_radii(solute_xyz,solute_el)
    solvent_alpha=obc_born_radii(solvent_xyz,solvent_el)

    E_sol_ref=solute.get("energy_Eh",0.0)*HARTREE_TO_KCAL
    E_solv_ref=solvent.get("energy_Eh",0.0)*HARTREE_TO_KCAL
    n_sol=len(solute_el)

    all_corrected=[]
    for gp_name,gp_data in data.items():
        if gp_name in ("solute","solvent"): continue
        if isinstance(gp_data,dict): pose_iter=gp_data.items()
        elif isinstance(gp_data,list): pose_iter=[(f"pose_{i}",p) for i,p in enumerate(gp_data)]
        else: continue
        for pose_name,pose_data in pose_iter:
            while isinstance(pose_data,list) and len(pose_data)>0:
                pose_data=pose_data[0]
            if not isinstance(pose_data,dict): continue

            e_explicit=float(pose_data.get("energy_Eh",0.0))*HARTREE_TO_KCAL
            xyz_all=pose_data.get("xyz",[])
            coords=np.array([[a["x"],a["y"],a["z"]] for a in xyz_all])
            sol_xyz=coords[:n_sol]; solv_xyz=coords[n_sol:]
            e_cross=gb_cross_energy(sol_xyz,solute_q,solute_alpha,
                                    solv_xyz,solvent_q,solvent_alpha)
            e_corr=e_explicit - e_cross
            e_int_corr=e_corr - (E_sol_ref + E_solv_ref)
            all_corrected.append(e_int_corr)
    return all_corrected

# =========================
# CLI entry point
# =========================
def main():
    """Entry point for the `corr` command."""
    charge_file="charges"
    if not os.path.exists(charge_file):
        raise FileNotFoundError(f"Charge file '{charge_file}' not found in current directory.")
    with open(charge_file) as f:
        solute_charges=[float(line.strip()) for line in f if line.strip()]
    print(f"✅ Loaded {len(solute_charges)} solute charges from '{charge_file}'")

    json_files=sorted(glob.glob("*.json"))
    if not json_files:
        raise FileNotFoundError("No .json files found in current directory.")
    print(f"📂 Found {len(json_files)} JSON files to process.\n")

    all_energies=[]
    for fn in json_files:
        try:
            E = process_single_json(fn, solute_charges)
            all_energies.extend(E)
            print(f"✅ {fn}: collected {len(E)} corrected interaction energies.")
        except Exception as e:
            print(f"❌ Error in {fn}: {e}")

    if not all_energies:
        raise RuntimeError("No corrected energies found in any JSON file.")

    mean,std,meta=boltzmann_average(all_energies,all_energies,T=T_DEFAULT)
    print("\n=== GLOBAL BOLTZMANN AVERAGE OVER ALL JSON FILES ===")
    print(f"T = {T_DEFAULT:.2f} K")
    print(f"Global ⟨E_int,corr⟩_B = {mean:.6f} ± {std:.6f} kcal/mol")
    print(f"(Z={meta['Z']:.6g}, Emin={meta['Emin_kcal']:.6f} kcal/mol)")
    print(f"Total samples: {len(all_energies)}")

    # Write a summary text file for convenience
    with open("global_average.txt","w") as f:
        f.write(f"Temperature: {T_DEFAULT:.2f} K\n")
        f.write(f"Global <E_int,corr>_B = {mean:.6f} ± {std:.6f} kcal/mol\n")
        f.write(f"Samples: {len(all_energies)}\n")
    print("\n📄 Saved global average to 'global_average.txt'.")

if __name__ == "__main__":
    main()

