"""June 2008 record flood (USGS 05454500 peak 41,100 cfs = 1,163.8 m3/s, stage 31.53 ft on 2008-06-15).

Used (a) as an extra stage validation point for the calibrated channel and (b) as the positive
control for the GPT-4o-mini perception test (widespread visible urban flooding).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import solver
from calibrate_channel import gage, model_stage
from grids import build

Q = 41100 * 0.0283168
OUT = Path(__file__).resolve().parents[1] / "data" / "c1" / "runs"
RES = Path(__file__).resolve().parent / "results"

x, y, _ = gage()
obs_stage_m = (617.10 + 31.53) * 0.3048
rows = []
for dx in (1.0, 0.5):
    g = build(dx)
    r = solver.run(g.z - 190.0, g.manning, g.inflow, Q, dx, h0=g.h0, t_min=3600.0, t_max=6 * 3600.0,
                   check_every=600.0, log=lambda m: print(m, flush=True))
    tag = f"record2008_dx{dx:g}"
    np.savez_compressed(OUT / f"{tag}.npz", h=r.h.astype(np.float32), u=r.u.astype(np.float32), v=r.v.astype(np.float32))
    s = model_stage(g, r.h, x, y)
    rows.append(dict(discharge="record2008", Q_m3s=round(Q, 1), dx=dx, observed_peak_stage_m=obs_stage_m,
                     model_stage_m=s, error_m=s - obs_stage_m, sim_time_s=r.sim_time_s, wall_time_s=r.wall_time_s))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(RES / "stage_validation_2008_peak.csv", index=False)
