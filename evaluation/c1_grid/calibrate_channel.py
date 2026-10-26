"""Calibrate the one unknown channel parameter against the USGS Iowa City gage, then validate.

3DEP lidar is hydro-flattened, so the river bed below the water surface is unknown. The bed is
lowered by a uniform burn depth. That depth is chosen so the modelled stage at USGS 05454500
(Iowa River at Iowa City) matches the USGS rating at the normal discharge (171 m3/s) on the 1 m grid.
The flood and extreme discharges are then used as an independent check against the rating.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

import solver
from grids import build

HERE = Path(__file__).resolve().parent
DATA2 = HERE.parents[0] / "data" / "c2" / "usgs"
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
CFS = {"normal": 6045.0, "flood": 20068.0, "extreme": 24922.0}
Q = {"normal": 171.0, "flood": 568.0, "extreme": 705.0}


def gage():
    site = pd.read_csv(DATA2 / "site_05454500.rdb", sep="\t", comment="#").iloc[1:].iloc[0]
    rating = pd.read_csv(DATA2 / "rating_05454500.rdb", sep="\t", comment="#").iloc[1:]
    x, y = Transformer.from_crs(4326, 26915, always_xy=True).transform(float(site.dec_long_va), float(site.dec_lat_va))
    datum_m = float(site.alt_va) * 0.3048  # NAVD88, same datum as 3DEP
    obs = {k: datum_m + 0.3048 * float(np.interp(c, rating.DEP.astype(float), rating.INDEP.astype(float)))
           for k, c in CFS.items()}
    return x, y, obs


def model_stage(g, h, x, y, half=10):
    col, row = ~g.transform * (x, y)
    r, c = int(row), int(col)
    k = int(round(half / g.dx))
    rs = slice(r - k, r + k + 1)
    m = g.river[rs] & (h[rs] > 0.05)
    return float(np.median((g.z[rs] + h[rs])[m]))


def run(burn, name, dx=1.0):
    g = build(dx, burn=burn)
    r = solver.run(g.z - 190.0, g.manning, g.inflow, Q[name], dx, h0=g.h0, t_min=3600.0, t_max=5 * 3600.0,
                   check_every=600.0, log=lambda m: None)
    return g, r


def main():
    x, y, obs = gage()
    rows = []
    for burn in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
        g, r = run(burn, "normal")
        rows.append(dict(burn_m=burn, model_stage_m=model_stage(g, r.h, x, y)))
        print(rows[-1], flush=True)
    cal = pd.DataFrame(rows)
    err = cal.model_stage_m - obs["normal"]
    # stage falls monotonically with burn depth; interpolate to zero error
    burn = float(np.interp(0.0, err.values[::-1], cal.burn_m.values[::-1]))
    val = []
    for name in ("normal", "flood", "extreme"):
        g, r = run(burn, name)
        s = model_stage(g, r.h, x, y)
        val.append(dict(discharge=name, Q_m3s=Q[name], rating_stage_m=obs[name], model_stage_m=s,
                        error_m=s - obs[name], role="calibration" if name == "normal" else "validation"))
        print(val[-1], flush=True)
    cal.assign(obs_normal_m=obs["normal"]).to_csv(OUT / "channel_calibration_sweep.csv", index=False)
    pd.DataFrame(val).to_csv(OUT / "stage_validation_05454500.csv", index=False)
    (OUT / "channel_calibration.json").write_text(json.dumps(dict(
        burn_m=round(burn, 3), gage="USGS 05454500 Iowa River at Iowa City",
        calibrated_on="normal 171 m3/s, 1 m grid", validation=val), indent=2))


if __name__ == "__main__":
    main()
