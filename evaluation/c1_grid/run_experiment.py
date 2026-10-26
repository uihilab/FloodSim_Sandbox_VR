"""Run the solver to steady state for every (cell size, discharge) pair and save the fields."""
import json
import sys
from pathlib import Path

import numpy as np

import solver
from grids import build

OUT = Path(__file__).resolve().parents[1] / "data" / "c1" / "runs"
OUT.mkdir(parents=True, exist_ok=True)

CELL_SIZES = [0.5, 1.0, 2.0, 4.0]
# Manuscript discharges (Section 2.4.1): normal, flood and extreme stages at USGS 05453520
DISCHARGES = {"normal": 171.0, "flood": 568.0, "extreme": 705.0}


def main(only=None):
    for name, Q in DISCHARGES.items():
        for dx in CELL_SIZES:
            tag = f"{name}_dx{dx:g}"
            if only and tag not in only:
                continue
            if (OUT / f"{tag}.npz").exists():
                print("skip", tag)
                continue
            g = build(dx)
            print(f"== {tag}: grid {g.z.shape}, {g.z.size/1e6:.2f} M cells", flush=True)
            r = solver.run(g.z - 190.0, g.manning, g.inflow, Q, dx, h0=g.h0,
                           t_min=3600.0, t_max=5 * 3600.0, check_every=600.0,
                           log=lambda m: print(m, flush=True))
            np.savez_compressed(OUT / f"{tag}.npz", h=r.h.astype(np.float32),
                                u=r.u.astype(np.float32), v=r.v.astype(np.float32))
            meta = dict(tag=tag, discharge=name, Q=Q, dx=dx, shape=list(g.z.shape),
                        sim_time_s=r.sim_time_s, wall_time_s=r.wall_time_s, steps=r.steps,
                        outflow_m3s=r.outflow_m3s, mass_clip_rel=r.mass_error_rel,
                        converged=bool(r.history[-1]["imbalance"] < 0.005 and r.history[-1]["dvol_rel"] < 0.001),
                        history=r.history)
            (OUT / f"{tag}.json").write_text(json.dumps(meta, indent=1))
            print(f"   done in {r.wall_time_s:.0f} s wall, {r.steps} steps", flush=True)


if __name__ == "__main__":
    main(set(sys.argv[1:]) or None)
