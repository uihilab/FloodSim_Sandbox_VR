"""Modality ablation and image/data conflict test, per model (run after score.py for each model).

Arms (same 30 flooded scenes, same manuscript-style state JSON with no severity label):
    image only  : A (deployed prompt) and C0 (structured probe, temperature 0)
    data only   : J (deployed-style prompt) and PJ (probe)
    image + data: K (deployed prompt) and PK (probe)

Conflict test: images paired with the state of a different scenario (dry image + 2008 data, 2008 image +
normal-flow data, normal-flow image + 2008 data) under K and PK. If outputs follow the data whatever
the image shows, the image is not contributing to the assessment.

Usage: python ablation.py --models gpt-4o-mini gpt-6.1-sol gpt-5.6-terra
Expects results/<model>/scored/responses_scored.csv (gpt-4o-mini: results/scored/) and
results/<model>/scored_mismatch/responses_scored.csv.
"""
import argparse
import csv
import json
import re

import numpy as np
import pandas as pd
from scipy import stats

from common import HERE, PRODUCTION_MODEL

ARMS = {"image only": ("A", "C0"), "data only": ("J", "PJ"), "image + data": ("K", "PK")}
CONTRADICTION = re.compile(r"(image|picture|photo|visual)[^.\n]{0,120}(does not|doesn't|do not|no (visible )?(sign|flood|water)|"
                           r"appears? (to be )?(dry|normal)|not (show|indicate|visible)|inconsistent|discrepan|contradict)", re.I)


def model_dir(m):
    return HERE / "results" if m == PRODUCTION_MODEL else HERE / "results" / m


def load(m, sub):
    df = pd.read_csv(model_dir(m) / sub / "responses_scored.csv")
    return df


def manifest(name):
    return {r["scene_id"]: r for r in csv.DictReader(open(HERE / "scenes" / name, encoding="utf-8-sig"))}


def boot_ci(x, n=2000, seed=0):
    x = np.asarray(x, float)
    if len(x) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    b = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return (float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)))


def ablation(m, df, man):
    df = df.copy()
    df["Q"] = df.scene_id.map(lambda s: float(man[s]["discharge_m3s"]))
    df["vis"] = df.scene_id.map(lambda s: int(man[s]["severity_tier_true"]))
    df = df[df.scenario != "dry"]
    rows, scene_tabs = [], {}
    for arm, (rec_c, probe_c) in ARMS.items():
        p = df[df.condition == probe_c].copy()
        r = df[df.condition == rec_c].copy()
        if p.empty or r.empty:
            continue
        p["correct"] = (p.tier_pred == p.vis).astype(float)
        p["overbank"] = (p.tier_pred >= 2) | (p.depth_pred.fillna(0) > 0)
        pos, neg = p[p.vis >= 2], p[p.vis <= 1]
        rho_t, p_t = stats.spearmanr(p.tier_pred, p.Q)
        sm = r.groupby("scene_id").agg(h=("n_high_severity", "mean"), Q=("Q", "first"))
        rho_h, p_h = stats.spearmanr(sm.h, sm.Q)
        sc_p = p.groupby("scene_id").correct.mean()
        sc_cov = r.groupby("scene_id").expected_coverage.mean()
        sc_inap = r.groupby("scene_id").any_inappropriate.mean()
        scene_tabs[arm] = dict(correct=sc_p, coverage=sc_cov, inappropriate=sc_inap)
        normal = r[r.scenario == "normal"]
        high = r[r.scenario.isin(["flood", "extreme", "record2008"])]
        rows.append({
            "model": m, "arm": arm, "probe_condition": probe_c, "recommendation_condition": rec_c,
            "tier_accuracy": p.correct.mean(), "tier_accuracy_ci": boot_ci(sc_p.values),
            "overbank_detected": f"{int(pos.overbank.sum())}/{len(pos)}",
            "overbank_recall": pos.overbank.mean() if len(pos) else np.nan,
            "overbank_false_report": f"{int(neg.overbank.sum())}/{len(neg)}",
            "tier_vs_discharge_rho": rho_t, "tier_vs_discharge_p": p_t,
            "depth_mae_m_on_overbank": float((p.loc[p.vis >= 2, "depth_pred"].fillna(0) - p.loc[p.vis >= 2, "depth_true"]).abs().mean()) if "depth_true" in p else np.nan,
            "expected_coverage": r.expected_coverage.mean(), "expected_coverage_ci": boot_ci(sc_cov.values),
            "any_inappropriate": r.any_inappropriate.mean(),
            "evacuation_normal_flow": float((normal.cat_evacuation > 0).mean()) if len(normal) else np.nan,
            "evacuation_flood_or_higher": float((high.cat_evacuation > 0).mean()) if len(high) else np.nan,
            "escalation_rho": rho_h, "escalation_p": p_h,
            "truncated": float((r.finish_reason == "length").mean()),
            "unverified_name_rate": float((r.n_unverified > 0).mean()),
            "latency_median_s_recommendation": float(r.latency_s.median()),
            "latency_median_s_probe": float(p.latency_s.median()),
            "prompt_tokens_median": float(r.prompt_tokens.median()),
        })
    tests = []
    for a, b in (("image + data", "data only"), ("image + data", "image only"), ("data only", "image only")):
        if a not in scene_tabs or b not in scene_tabs:
            continue
        for metric in ("correct", "coverage", "inappropriate"):
            x, y = scene_tabs[a][metric], scene_tabs[b][metric]
            j = pd.concat([x, y], axis=1, keys=["a", "b"]).dropna()
            d = j.a - j.b
            pv = stats.wilcoxon(j.a, j.b).pvalue if (d != 0).any() else np.nan
            tests.append(dict(model=m, comparison=f"{a} vs {b}", metric=metric, n_scenes=len(j),
                              mean_a=j.a.mean(), mean_b=j.b.mean(), diff=d.mean(), wilcoxon_p=pv))
    return rows, tests


def conflict(m, mm_df, main_df, mm_man):
    """For each mismatch scene compare with the image-source and data-source scenes under matched input."""
    rows = []
    for cond, col in (("PK", "tier_pred"), ("K", "n_high_severity")):
        for sid, g in mm_df[mm_df.condition == cond].groupby("scene_id"):
            img_src = mm_man[sid]["base_scene_id"]
            json_src = mm_man[sid]["json_source"]
            mm = g[col].mean()
            # image source under matched input; dry images have no state, so use the image-only arm
            img_cond = cond if not img_src.startswith("dry") else ("C0" if cond == "PK" else "A")
            img_val = main_df[(main_df.scene_id == img_src) & (main_df.condition == img_cond)][col].mean()
            json_val = main_df[(main_df.scene_id == json_src) & (main_df.condition == cond)][col].mean()
            follow = (mm - img_val) / (json_val - img_val) if abs(json_val - img_val) > 1e-9 else np.nan
            flagged = float(g.response_text.fillna("").str.contains(CONTRADICTION).mean())
            rows.append(dict(model=m, condition=cond, metric=col, mismatch_scene=sid, image_source=img_src,
                             data_source=json_src, image_source_value=img_val, data_source_value=json_val,
                             mismatch_value=mm, data_following_index=follow,
                             responses_noting_image_data_conflict=flagged))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=[PRODUCTION_MODEL])
    args = ap.parse_args()
    man, mm_man = manifest("manifest.csv"), manifest("manifest_mismatch.csv")
    abl, tests, conf = [], [], []
    for m in args.models:
        main_df = load(m, "scored")
        a, t = ablation(m, main_df, man)
        abl += a
        tests += t
        mm_path = model_dir(m) / "scored_mismatch" / "responses_scored.csv"
        if mm_path.exists():
            conf += conflict(m, pd.read_csv(mm_path), main_df, mm_man)
    out = HERE / "results" / "ablation"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(abl).to_csv(out / "ablation_by_arm.csv", index=False)
    pd.DataFrame(tests).to_csv(out / "ablation_paired_tests.csv", index=False)
    cdf = pd.DataFrame(conf)
    cdf.to_csv(out / "conflict_test.csv", index=False)
    summ = cdf.groupby(["model", "condition"]).agg(
        median_data_following_index=("data_following_index", "median"),
        mean_data_following_index=("data_following_index", "mean"),
        noted_conflict=("responses_noting_image_data_conflict", "mean")).reset_index() if len(cdf) else pd.DataFrame()
    summ.to_csv(out / "conflict_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(abl).drop(columns=["tier_accuracy_ci", "expected_coverage_ci"]).round(3).to_string())
    print(pd.DataFrame(tests).round(3).to_string())
    print(summ.round(3).to_string())


if __name__ == "__main__":
    main()
