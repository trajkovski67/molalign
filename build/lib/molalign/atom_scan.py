#!/usr/bin/env python3
"""
align_many_to_one_orca.py  (Parallel OPI version; CPCM water + ΔE-step scan)

Enhancements:
  • CPCM(Water) implicit solvent on every SP.
  • Monomer (solute + solvent) SPs computed first.
  • For each orientation:
        - INSIDE (toward solute): +0.2 Å steps until ΔE_step ≥ +10 kcal/mol.
        - OUTSIDE (away): start 0.2 Å, grow ×1.5 until ΔE_step > −0.5 kcal/mol.
  • ΔE_step is the *change* in interaction energy between consecutive steps.
  • Interaction energy: E_int = E_complex − E_solute − E_solvent.
"""

import json, os, tempfile
from pathlib import Path
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

from opi.core import Calculator
from opi.input.simple_keywords import Dft, Task
from opi.input.structures.structure import Structure

HARTREE_TO_KCAL_MOL = 627.509474

INSIDE_STEP_ANG = 0.2
INSIDE_STOP_dE_KCAL = +10.0
OUTSIDE_START_STEP_ANG = 0.2
OUTSIDE_EXP_MULT = 1.5
OUTSIDE_STOP_dE_KCAL = -0.5
MAX_INSIDE_STEPS = 300
MAX_OUTSIDE_STEPS = 200
MAX_TOTAL_STEPS_PER_ORIENTATION = 600


# ---------- ORCA/OPI helpers ----------

def _pick_dft(name):
    if not name:
        return Dft.R2SCAN_3C
    key = name.strip().upper().replace("-", "_")
    aliases = {"R2SCAN3C":"R2SCAN_3C"}
    key = aliases.get(key, key)
    return getattr(Dft, key, Dft.R2SCAN_3C)

def _enable_cpcm_water(calc):
    try: calc.input.add_keywords(["CPCM(Water)"])
    except Exception: pass
    calc.input.text_blocks = getattr(calc.input, "text_blocks", {})
    calc.input.text_blocks["%cpcm"] = "epsilon 80.4\nend"

def run_orca_sp_opi(coords, Z, charge=0, mult=1, ncores=8, method="r2scan-3c"):
    os.environ["OMP_NUM_THREADS"] = os.environ["ORCA_NUM_PROCS"] = str(ncores)
    sym = {1:"H",6:"C",7:"N",8:"O",9:"F",16:"S",17:"Cl",35:"Br",53:"I"}
    lines=[str(len(Z)),""]+[f"{sym.get(int(z),str(z)):<3} {x: .8f} {y: .8f} {z: .8f}" for z,(x,y,z) in zip(Z,coords)]
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp); xyz=p/"geom.xyz"; xyz.write_text("\n".join(lines)+"\n")
        s=Structure.from_xyz(xyz); s.charge=int(charge); s.multiplicity=int(mult)
        calc=Calculator(basename="sp",working_dir=p); calc.structure=s
        calc.input.add_simple_keywords(_pick_dft(method),Task.SP)
        calc.input.text_blocks={"%pal":f"nprocs {ncores}\nend"}
        _enable_cpcm_water(calc)
        calc.write_input(); calc.run()
        out=calc.get_output()
        if not out.terminated_normally():
            raise RuntimeError("ORCA failed")
        out.parse()
        return float(out.results_properties.geometries[0].single_point_data.finalenergy)


# ---------- geometry utilities ----------

def rotation_matrix_from_vectors(a,b,eps=1e-12):
    a,b=np.array(a,float),np.array(b,float)
    na,nb=np.linalg.norm(a),np.linalg.norm(b)
    if na<eps or nb<eps: return np.eye(3)
    a/=na; b/=nb
    dot=np.dot(a,b)
    if dot>1-eps: return np.eye(3)
    if dot<-1+eps:
        axis=np.cross(a,[1,0,0] if abs(a[0])<=0.9 else [0,1,0]); axis/=np.linalg.norm(axis)
        return -np.eye(3)+2*np.outer(axis,axis)
    v=np.cross(a,b); s=np.linalg.norm(v)
    K=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    return np.eye(3)+K+K.dot(K)*((1-dot)/(s**2))

def rotate_around_axis(coords,axis,angle_deg):
    t=np.radians(angle_deg); k=axis/ (np.linalg.norm(axis)+1e-20)
    K=np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]])
    return coords@(np.eye(3)+np.sin(t)*K+(1-np.cos(t))*(K@K)).T


# ---------- core math ----------

def _align_and_rotate_B(atA,atB,pA,nA,pB,nB,ang):
    target=-nA; R=rotation_matrix_from_vectors(nB,target)
    atB=atB@R.T; pB=(pB@R.T); atB+=(pA-pB)
    atB=rotate_around_axis(atB-pA,target,ang)+pA
    return atB, target/np.linalg.norm(target), pA

def _interaction_energy_kcal(Ec,EA,EB): return (Ec-EA-EB)*HARTREE_TO_KCAL_MOL

def _compute_monomer_energies(atA,ZA,atB,ZB,cA,mA,cB,mB,ncores,method):
    EA=run_orca_sp_opi(atA,ZA,cA,mA,ncores,method)
    EB=run_orca_sp_opi(atB,ZB,cB,mB,ncores,method)
    return EA,EB


# ---------- orientation scan ----------

def _scan_single_orientation(args):
    (atA,ZA,atB,ZB,pA,nA,pB,nB,idxB,ang,
     cA,mA,cB,mB,ncores,method,EA,EB)=args
    key=f"b{idxB}_rot{int(ang)}"
    try:
        B0,axis,pivot=_align_and_rotate_B(atA,atB,pA,nA,pB,nB,ang)
        res={"orientation":key,"inside":[],"outside":[]}
        steps=0

        def run_complex(offset,dir):
            nonlocal steps
            disp=axis*(offset if dir=="inside" else -offset)
            B=B0+disp
            coords=np.vstack((atA,B)); Z=np.hstack((ZA,ZB))
            Ec=run_orca_sp_opi(coords,Z,charge=cA+cB,mult=max(1,mA*mB),
                               ncores=ncores,method=method)
            Eint=_interaction_energy_kcal(Ec,EA,EB)
            steps+=1
            return Eint,{"offset_A":float(offset),
                         "energy_Eh":float(Ec),
                         "E_int_kcal_mol":float(Eint)}

        # --- inside scan ---
        prev=None; offset=0.0
        for i in range(MAX_INSIDE_STEPS):
            Eint,rec=run_complex(offset,"inside"); res["inside"].append(rec)
            if prev is not None:
                dE=Eint-prev
                res["inside"][-1]["dE_step_kcal_mol"]=float(dE)
                if dE>=INSIDE_STOP_dE_KCAL: break
            prev=Eint; offset+=INSIDE_STEP_ANG
            if steps>=MAX_TOTAL_STEPS_PER_ORIENTATION: break

        # --- outside scan ---
        prev=None; offset=0.0; step=OUTSIDE_START_STEP_ANG
        for i in range(MAX_OUTSIDE_STEPS):
            Eint,rec=run_complex(offset,"outside"); res["outside"].append(rec)
            if prev is not None:
                dE=Eint-prev
                res["outside"][-1]["dE_step_kcal_mol"]=float(dE)
                if dE>OUTSIDE_STOP_dE_KCAL: break
            prev=Eint; offset+=step; step*=OUTSIDE_EXP_MULT
            if steps>=MAX_TOTAL_STEPS_PER_ORIENTATION: break

        return key,res
    except Exception as e:
        return key,{"error":str(e)}


# ---------- main ----------

def main(fileA,fileB,idxA,angles=[0],charge=0,mult=1,
         ncores=8,method="r2scan-3c",max_workers=None,
         chargeA=None,multA=None,chargeB=None,multB=None):
    dA=np.load(fileA,allow_pickle=True); dB=np.load(fileB,allow_pickle=True)
    atA,Z_A=dA["atoms"][:,:3],dA["atoms"][:,3].astype(int)
    grA=dA["grids"][:,:3]; nA=grA-atA[dA["grids"][:,4].astype(int)]
    nA/=np.linalg.norm(nA,axis=1,keepdims=True)+1e-12
    atB,Z_B=dB["atoms"][:,:3],dB["atoms"][:,3].astype(int)
    grB=dB["grids"][:,:3]; nB=grB-atB[dB["grids"][:,4].astype(int)]
    nB/=np.linalg.norm(nB,axis=1,keepdims=True)+1e-12

    cA=int(0 if chargeA is None else chargeA)
    mA=int(1 if multA is None else multA)
    cB=int(0 if chargeB is None else chargeB)
    mB=int(1 if multB is None else multB)

    pA,na=grA[idxA],nA[idxA]

    EA,EB=_compute_monomer_energies(atA,Z_A,atB,Z_B,cA,mA,cB,mB,ncores,method)

    tasks=[]
    for idxb,(pB,nb) in enumerate(zip(grB,nB)):
        for ang in angles:
            tasks.append((atA,Z_A,atB,Z_B,pA,na,pB,nb,idxb,ang,
                          cA,mA,cB,mB,ncores,method,EA,EB))

    if max_workers is None:
        total=os.cpu_count() or 8
        max_workers=max(1,total//ncores)

    print(f"[Grid {idxA}] {len(tasks)} orientations, {ncores} cores/job, {max_workers} workers")

    out={"meta":{"idxA":idxA,"angles_deg":angles,"method":method,
                 "implicit":"CPCM(Water)","inside_step_A":INSIDE_STEP_ANG,
                 "inside_stop_dE_kcal":INSIDE_STOP_dE_KCAL,
                 "outside_start_step_A":OUTSIDE_START_STEP_ANG,
                 "outside_exp_mult":OUTSIDE_EXP_MULT,
                 "outside_stop_dE_kcal":OUTSIDE_STOP_dE_KCAL},
         "monomers":{"E_solute_Eh":EA,"E_solvent_Eh":EB},
         "orientations":{}}

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs=[ex.submit(_scan_single_orientation,t) for t in tasks]
        for f in as_completed(futs):
            k,v=f.result(); out["orientations"][k]=v
    return out


# ---------- CLI ----------

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("fileA"); p.add_argument("fileB"); p.add_argument("idxA",type=int)
    p.add_argument("--angles",default="0"); p.add_argument("--ncores",type=int,default=8)
    p.add_argument("--method",default="r2scan-3c"); p.add_argument("--max-workers",type=int,default=None)
    p.add_argument("--chargeA",type=int,default=0); p.add_argument("--multA",type=int,default=1)
    p.add_argument("--chargeB",type=int,default=0); p.add_argument("--multB",type=int,default=1)
    a=p.parse_args()
    angs=[float(x) for x in a.angles.split(",")]
    r=main(a.fileA,a.fileB,a.idxA,angs,ncores=a.ncores,method=a.method,
           max_workers=a.max_workers,chargeA=a.chargeA,multA=a.multA,
           chargeB=a.chargeB,multB=a.multB)
    print(json.dumps(r,indent=2))

