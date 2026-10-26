"""Label-independent headline metrics from the scored responses (run after score.py).

- Overbank detection: does the structured probe report overbank flooding (tier >= 2 or a non-zero
  depth) on scenes where the simulation has water on streets or around buildings (visual tier >= 2)?
- Discharge discrimination: Spearman correlation between predicted tier and upstream discharge.
- Escalation of recommendations with discharge, per condition.
"""
import csv
import json

import numpy as np
import pandas as pd
from scipy import stats

from common import HERE

df = pd.read_csv(HERE / "results" / "scored" / "responses_scored.csv")
man = {r["scene_id"]: r for r in csv.DictReader(open(HERE / "scenes" / "manifest.csv", encoding="utf-8"))}
df["Q"] = df.scene_id.map(lambda s: float(man[s]["discharge_m3s"]))
df["visual_tier"] = df.scene_id.map(lambda s: int(man[s]["severity_tier_true"]))

out = {}
p = df[df.condition.isin(["C0", "C1"])].copy()
p["reports_overbank"] = (p.tier_pred >= 2) | (p.depth_pred.fillna(0) > 0)
pos, neg = p[p.visual_tier >= 2], p[p.visual_tier <= 1]
out["probe_overbank"] = dict(
    responses_on_overbank_scenes=int(len(pos)), detected=int(pos.reports_overbank.sum()),
    responses_on_other_scenes=int(len(neg)), false_reports=int(neg.reports_overbank.sum()))
out["probe_tier_counts_by_scenario"] = pd.crosstab(p.scenario, p.tier_pred).to_dict()
nd = p[p.scenario != "dry"]
rho, pval = stats.spearmanr(nd.tier_pred, nd.Q)
out["probe_tier_vs_discharge_spearman"] = dict(rho=float(rho), p=float(pval), n=int(len(nd)))

esc = {}
for c in ("A", "B", "C0", "C1"):
    a = df[df.condition == c]
    esc[c] = dict(
        high_severity_per_response=a.groupby("scenario").n_high_severity.mean().round(3).to_dict(),
        evacuation_share=a.groupby("scenario").cat_evacuation.apply(lambda s: float((s > 0).mean())).round(3).to_dict(),
        truncated_share=float((a.finish_reason == "length").mean()))
    nd_c = a[a.scenario != "dry"].groupby("scene_id").agg(h=("n_high_severity", "mean"), Q=("Q", "first"))
    r, pv = stats.spearmanr(nd_c.h, nd_c.Q)
    esc[c]["scene_high_severity_vs_discharge_spearman"] = dict(rho=float(r), p=float(pv))
out["escalation"] = esc

path = HERE / "results" / "scored" / "headline_metrics.json"
path.write_text(json.dumps(out, indent=2, default=str))
print(json.dumps(out, indent=2, default=str))
