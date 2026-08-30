#!/usr/bin/env python3
"""Post-hoc matched response using the frozen Stage 1 Python simulator."""
from __future__ import annotations
import csv, sys
from pathlib import Path
import numpy as np
from probeing.experiments.decision_sufficiency import TargetCase, simulate_population

source=Path(sys.argv[1]); destination=Path(sys.argv[2])
with source.open(newline="",encoding="utf-8") as f: rows=list(csv.DictReader(f))
groups: dict[str,list[dict[str,str]]]={}
for row in rows: groups.setdefault(row["target_id"],[]).append(row)
output=[]
for target_id, values in groups.items():
    time=np.asarray([float(v["time_s"]) for v in values]); force=np.asarray([float(v["force"]) for v in values])
    case=TargetCase(target_id,"matched",0,0,float(values[0]["k"]),float(values[0]["c"]),float(values[0]["m"]))
    sim=simulate_population((case,),time,force,contact_mode="unilateral")
    for i,row in enumerate(values):
        output.append({**row,"python_x":sim.displacement_m[0,i],"python_v":sim.velocity_m_per_s[0,i],"python_a":sim.acceleration_m_per_s2[0,i]})
with destination.open("x",newline="",encoding="utf-8") as f:
    writer=csv.DictWriter(f,fieldnames=list(output[0]));writer.writeheader();writer.writerows(output)
