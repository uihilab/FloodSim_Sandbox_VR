"""Analyse the in-engine Fluid Flux runs described in INENGINE_PROTOCOL.md.

Inputs (evaluation/data/c1/inengine/):
  buildings_<discharge>_<cellsize>.csv   from export_building_heights.py, e.g. buildings_flood_0.5m.csv
  profiler CSVs from csvprofile, named <discharge>_<cellsize>.csv in a profiling/ subfolder (optional)
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

IN = Path(__file__).resolve().parents[1] / "data" / "c1" / "inengine"
OUT = Path(__file__).resolve().parent / "results" / "inengine"
WET_CM = 10.0  # same 0.10 m threshold as the solver study


def cellsize(tag):
    return float(re.search(r"_([\d.]+)m$", tag).group(1))


def main():
    files = sorted(IN.glob("buildings_*.csv"))
    if not files:
        raise SystemExit(f"No in-engine exports in {IN}. Follow INENGINE_PROTOCOL.md first.")
    OUT.mkdir(parents=True, exist_ok=True)
    runs = {}
    for f in files:
        tag = f.stem.removeprefix("buildings_")
        discharge = tag.split("_")[0]
        runs.setdefault(discharge, {})[cellsize(tag)] = pd.read_csv(f).set_index("actor")

    rows = []
    for discharge, by_dx in runs.items():
        ref_dx = min(by_dx)
        ref = by_dx[ref_dx]
        for dx, df in sorted(by_dx.items()):
            j = ref[["LocalHeight"]].join(df[["LocalHeight"]], lsuffix="_ref", how="inner")
            wet_r, wet_c = j.LocalHeight_ref > WET_CM, j.LocalHeight > WET_CM
            err_m = (j.LocalHeight - j.LocalHeight_ref)[wet_r | wet_c] / 100.0
            rows.append(dict(discharge=discharge, dx=dx, ref_dx=ref_dx, buildings=len(j),
                             wet_ref=int(wet_r.sum()), wet=int(wet_c.sum()),
                             classification_agreement=float((wet_r == wet_c).mean()),
                             depth_bias_m=float(err_m.mean()) if len(err_m) else np.nan,
                             depth_abs_err_median_m=float(err_m.abs().median()) if len(err_m) else np.nan,
                             depth_abs_err_p90_m=float(err_m.abs().quantile(0.9)) if len(err_m) else np.nan))
    res = pd.DataFrame(rows)

    prof_dir = IN / "profiling"
    perf = []
    for f in sorted(prof_dir.glob("*.csv")) if prof_dir.exists() else []:
        p = pd.read_csv(f, low_memory=False)
        p = p.apply(pd.to_numeric, errors="coerce")
        frame = p.get("FrameTime")
        gpu = p.get("GPUTime")
        perf.append(dict(run=f.stem, frame_ms_mean=frame.mean() if frame is not None else np.nan,
                         frame_ms_p95=frame.quantile(0.95) if frame is not None else np.nan,
                         gpu_ms_mean=gpu.mean() if gpu is not None else np.nan,
                         fps_mean=1000.0 / frame.mean() if frame is not None else np.nan))
    res.to_csv(OUT / "inengine_building_metrics.csv", index=False)
    if perf:
        pd.DataFrame(perf).to_csv(OUT / "inengine_frame_cost.csv", index=False)
    print(res.round(3).to_string())
    if perf:
        print(pd.DataFrame(perf).round(2).to_string())


if __name__ == "__main__":
    main()
