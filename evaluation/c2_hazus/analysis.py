"""Validation and uncertainty of the HAZUS-based damage estimates.

Run fetch_data.py first. Outputs go to ./out/ and ./summary.md.

A. Document the in-app damage model (constants read from BP_SimControl.uasset).
B. Validate Hazus 4.0 RES1 depth-damage functions against June 2008 NFIP claims
   (Johnson County = Iowa City/Coralville, Linn County = Cedar Rapids; same event).
C. Monte Carlo propagation of stage, first-floor-elevation, curve-choice and
   empirical-bias uncertainty onto the in-app aggregate dollar estimates.
D. Why the per-building height-ratio overlay overstates damage for procedural buildings.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data", "c2"))
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(20261023)
N_BOOT = 2000
N_MC = 100_000

# ---------------------------------------------------------------- A. in-app model
# Literal constants in Content/Blueprints/Core/BP_SimControl.uasset (both project copies):
# StructureDamage = 5688154 * stage_ft - 134925113, ContentDamage = 3172460 * stage_ft - 74897629,
# with fluid height converted from Unreal units (cm) to ft by 0.032808.
STRUCT_A, STRUCT_B = 5_688_154.0, -134_925_113.0
CONT_A, CONT_B = 3_172_460.0, -74_897_629.0
# The manuscript's stages (53.5 / 60.83 / 63.0 ft) are gage heights at USGS 05453520 (Iowa River below
# Coralville Dam). The IFIS stage-loss fit reaches zero at 23.7 ft, which is on the scale of the downtown
# gage USGS 05454500 (Iowa River at Iowa City), so the loss lines are evaluated at the 05454500 stage
# for the same discharge, read from the current USGS rating (data/c2/usgs/rating_05454500.rdb).
DISCHARGES_CFS = {"Normal (171 m3/s)": 6045.0, "Flood (568 m3/s)": 20068.0, "Extreme (705 m3/s)": 24922.0}
MANUSCRIPT_STAGE_05453520 = {"Normal (171 m3/s)": 53.5, "Flood (568 m3/s)": 60.83, "Extreme (705 m3/s)": 63.0}


def _rating_stage(site, cfs):
    r = pd.read_csv(os.path.join(DATA, "usgs", f"rating_{site}.rdb"), sep="	", comment="#").iloc[1:]
    return float(np.interp(cfs, r["DEP"].astype(float), r["INDEP"].astype(float)))


STAGES = {k: round(_rating_stage("05454500", q), 2) for k, q in DISCHARGES_CFS.items()}


def in_app(stage_ft):
    s = np.maximum(0.0, STRUCT_A * stage_ft + STRUCT_B)
    c = np.maximum(0.0, CONT_A * stage_ft + CONT_B)
    return s, c


def part_a():
    rows = []
    for name, st in STAGES.items():
        s, c = in_app(st)
        rows.append({"scenario": name, "discharge_cfs": DISCHARGES_CFS[name],
                     "stage_05453520_manuscript_ft": MANUSCRIPT_STAGE_05453520[name],
                     "check_05453520_from_rating_ft": round(_rating_stage("05453520", DISCHARGES_CFS[name]), 2),
                     "stage_ft": st, "structure_usd": s, "content_usd": c, "total_usd": s + c})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "A_in_app_model_at_manuscript_stages.csv"), index=False)
    info = {"zero_damage_stage_structure_ft": -STRUCT_B / STRUCT_A,
            "zero_damage_stage_content_ft": -CONT_B / CONT_A,
            "structure_usd_per_ft": STRUCT_A, "content_usd_per_ft": CONT_A}
    return df, info


# ---------------------------------------------------------------- Hazus curves
DEPTH_COLS = [f"m{i}" for i in (4, 3, 2, 1)] + [f"p{i}" for i in range(25)]
DEPTHS_FT = np.array([-4, -3, -2, -1] + list(range(25)), dtype=float)
STORY_KEY = {1: "1", 2: "2", 3: "3", 4: "S"}  # NFIP numberOfFloors code -> Hazus RES1 id letter


def load_default_curves():
    """Hazus 4.0 default riverine RES1 structure curves, keyed by (story, basement)."""
    lut = pd.read_csv(os.path.join(DATA, "Building_DDF_Riverine_LUT_Hazus4p0.csv"))
    lut = lut[lut["Occupancy"] == "RES1"]
    curves = {}
    for _, r in lut.iterrows():
        sid = r["SpecificOccupId"]  # e.g. R12B
        curves[(sid[2], sid[3] == "B")] = (r[DEPTH_COLS].to_numpy(float) / 100.0, f"{sid} {r['Source']}: {r['Description']}")
    return curves


def load_alternative_families():
    """Other RES1 structure curves in the Hazus library, grouped by source, freshwater/riverine only."""
    fn = pd.read_csv(os.path.join(DATA, "flBldgStructDmgFn.csv"))
    fn = fn[(fn["Occupancy"] == "RES1") & fn["Description"].str.contains("Structure", case=False)]
    fn = fn[~fn["Description"].str.contains("V-Zone|salt", case=False)]
    fn = fn[~fn["Source"].str.contains("Galveston|New Orleans", case=False)]  # Gulf-coast specific
    fams = {}
    for _, r in fn.iterrows():
        d = r["Description"].lower()
        story = ("S" if "split" in d else "3" if "three" in d else "2" if ("two" in d or "1/2" in d) else
                 "1" if "one" in d else None)
        if story is None:
            continue
        base = "w/ basement" in d
        fam = r["Source"].strip()
        fams.setdefault(fam, {}).setdefault((story, base), (r[DEPTH_COLS].to_numpy(float) / 100.0, int(r["BldgDmgFnID"])))
    return fams


def interp_curve(curve, depth_ft):
    return np.interp(depth_ft, DEPTHS_FT, curve)


# ---------------------------------------------------------------- B. claims
def load_claims():
    frames = []
    for c in ("johnson", "linn"):
        d = pd.DataFrame(json.load(open(os.path.join(DATA, f"nfip_claims_{c}_county_IA.json"), encoding="utf-8")))
        d["county"] = c.title()
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    log = [("all claims, both counties, all years", len(d))]
    d = d[(d["yearOfLoss"] == 2008) & d["dateOfLoss"].str[:7].isin(["2008-05", "2008-06", "2008-07"])]
    log.append(("May-July 2008 (Iowa/Cedar River flood)", len(d)))
    d = d[d["occupancyType"].isin([1, 11])]
    log.append(("single-family residential (occupancyType 1 or 11)", len(d)))
    d = d[~d["elevatedBuildingIndicator"].fillna(False).astype(bool)]
    log.append(("not elevated", len(d)))
    d = d[d["numberOfFloorsInTheInsuredBuilding"].isin([1, 2, 3, 4])]
    log.append(("1, 2, 3+ floors or split level", len(d)))

    bec = d["basementEnclosureCrawlspaceType"]
    found = d["foundationType"].astype(str)
    basement = pd.Series(np.nan, index=d.index, dtype=object)
    basement[bec.isin([1, 2])] = True
    basement[bec.isin([0, 3, 4])] = False
    basement[bec.isna() & (found == "2")] = True
    basement[bec.isna() & found.isin(["1", "3"])] = False
    d = d.assign(basement=basement)

    wd = pd.to_numeric(d["waterDepth"], errors="coerce")
    d = d.assign(depth_ft=wd / 12.0)
    d = d[wd.notna() & (wd != 9999)]
    log.append(("waterDepth reported", len(d)))
    d = d[(d["depth_ft"] >= -4) & (d["depth_ft"] <= 24)]
    log.append(("depth within Hazus curve domain (-4 to 24 ft)", len(d)))
    dmg = pd.to_numeric(d["buildingDamageAmount"], errors="coerce")
    rc = pd.to_numeric(d["buildingReplacementCost"], errors="coerce")
    acv = pd.to_numeric(d["buildingPropertyValue"], errors="coerce")
    d = d.assign(dmg=dmg, rc=rc, acv=acv)
    d = d[(d["dmg"] > 0) & (d["rc"] > 0)]
    log.append(("building damage > 0 and replacement cost > 0", len(d)))
    log.append(("  of which basement status unknown (excluded from primary sample)", int(d["basement"].isna().sum())))
    log.append(("  of which waterDepth exactly 0 (ambiguous; excluded from primary sample)", int((d["depth_ft"] == 0).sum())))
    return d, log


def predict(d, curves):
    out = np.full(len(d), np.nan)
    for i, (st, b, z) in enumerate(zip(d["numberOfFloorsInTheInsuredBuilding"], d["basement"], d["depth_ft"])):
        if b is None or (isinstance(b, float) and np.isnan(b)):
            continue
        key = (STORY_KEY[int(st)], bool(b))
        if key in curves:
            out[i] = interp_curve(curves[key][0], z)
    return out


def metrics(obs, pred, dmg_usd, rc):
    e = pred - obs
    agg = np.sum(pred * rc) / np.sum(dmg_usd)
    rho = stats.spearmanr(obs, pred).statistic if len(obs) > 2 else np.nan
    return {"n": len(obs), "mean_observed_DR": obs.mean(), "mean_predicted_DR": pred.mean(),
            "bias_pred_minus_obs": e.mean(), "MAE": np.abs(e).mean(), "RMSE": np.sqrt((e ** 2).mean()),
            "spearman_obs_vs_pred": rho, "aggregate_pred_over_obs_usd": agg}


def boot_ci(obs, pred, dmg_usd, rc):
    n = len(obs)
    keys = ["bias_pred_minus_obs", "MAE", "RMSE", "spearman_obs_vs_pred", "aggregate_pred_over_obs_usd"]
    samples = {k: [] for k in keys}
    for _ in range(N_BOOT):
        idx = RNG.integers(0, n, n)
        m = metrics(obs[idx], pred[idx], dmg_usd[idx], rc[idx])
        for k in keys:
            samples[k].append(m[k])
    return {k: (np.nanpercentile(v, 2.5), np.nanpercentile(v, 97.5)) for k, v in samples.items()}, samples


DEPTH_BINS = [-4, 0, 1, 2, 4, 6, 10, 24]


def part_b(curves):
    d, log = load_claims()
    d = d.copy()
    d["DR_obs_raw"] = d["dmg"] / d["rc"]
    d["DR_obs"] = d["DR_obs_raw"].clip(upper=1.0)
    d["DR_obs_acv"] = (d["dmg"] / d["acv"].where(d["acv"] > 0)).clip(upper=1.0)
    d["DR_pred"] = predict(d, curves)
    d["DR_obs_capped"] = d["DR_obs_raw"] > 1.0
    withzero = d[d["DR_pred"].notna()].copy()
    # waterDepth == 0 is ambiguous (unrecorded vs water exactly at the floor); primary sample drops it.
    prim = withzero[withzero["depth_ft"] != 0].copy()

    results = {}
    def run(name, sub, obs_col="DR_obs"):
        sub = sub[sub[obs_col].notna()]
        o, p = sub[obs_col].to_numpy(), sub["DR_pred"].to_numpy()
        usd = (sub[obs_col] * (sub["rc"] if obs_col == "DR_obs" else sub["acv"])).to_numpy()
        base = sub["rc"].to_numpy() if obs_col == "DR_obs" else sub["acv"].to_numpy()
        m = metrics(o, p, usd, base)
        ci, samples = boot_ci(o, p, usd, base)
        for k, (lo, hi) in ci.items():
            m[f"{k}_CI95_lo"], m[f"{k}_CI95_hi"] = lo, hi
        m["spearman_obs_vs_depth"] = stats.spearmanr(o, sub["depth_ft"]).statistic
        results[name] = m
        return samples

    agg_samples = run("Primary: both counties, replacement-cost basis", prim)["aggregate_pred_over_obs_usd"]
    run("Including claims with waterDepth = 0", withzero)
    run("Johnson County only (Iowa City, Coralville)", prim[prim["county"] == "Johnson"])
    run("Linn County only (Cedar Rapids)", prim[prim["county"] == "Linn"])
    run("Riverine overflow only (causeOfDamage 2)", prim[prim["causeOfDamage"].astype(str) == "2"])
    run("Actual-cash-value basis", prim, obs_col="DR_obs_acv")
    run("Above-first-floor depths only (depth >= 0)", prim[prim["depth_ft"] >= 0])
    # The data dictionary warns some depths were entered in feet; values of 1-11 "inches" are the suspect range.
    agg_screened = run("Excluding 0-1 ft depths (possible feet-as-inches entries)",
                       prim[(prim["depth_ft"] < 0) | (prim["depth_ft"] >= 1)])["aggregate_pred_over_obs_usd"]
    for assume in (True, False):
        alt = d.copy()
        alt.loc[alt["basement"].isna(), "basement"] = assume
        alt["DR_pred"] = predict(alt, curves)
        run(f"Unknown basement assigned {'basement' if assume else 'no basement'}",
            alt[alt["DR_pred"].notna() & (alt["depth_ft"] != 0)])

    metr = pd.DataFrame(results).T
    metr.to_csv(os.path.join(OUT, "B_validation_metrics.csv"))

    prim["depth_bin"] = pd.cut(prim["depth_ft"], DEPTH_BINS, right=False)
    binned = prim.groupby("depth_bin", observed=True).agg(
        n=("DR_obs", "size"), obs_mean=("DR_obs", "mean"), obs_median=("DR_obs", "median"),
        pred_mean=("DR_pred", "mean"), pred_median=("DR_pred", "median")).reset_index()
    binned["mean_bias_pred_minus_obs"] = binned["pred_mean"] - binned["obs_mean"]
    binned["depth_bin"] = binned["depth_bin"].astype(str)
    binned.to_csv(os.path.join(OUT, "B_binned_by_depth.csv"), index=False)

    cls = prim.groupby(["numberOfFloorsInTheInsuredBuilding", "basement"]).agg(
        n=("DR_obs", "size"), obs_mean=("DR_obs", "mean"), pred_mean=("DR_pred", "mean")).reset_index()
    cls.to_csv(os.path.join(OUT, "B_by_building_class.csv"), index=False)
    prim[["county", "nfipCommunityName", "dateOfLoss", "causeOfDamage", "numberOfFloorsInTheInsuredBuilding",
          "basement", "depth_ft", "dmg", "rc", "acv", "DR_obs_raw", "DR_obs", "DR_pred"]].to_csv(
        os.path.join(OUT, "B_claims_sample.csv"), index=False)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    for b, mk in ((False, "o"), (True, "^")):
        s = prim[prim["basement"] == b]
        ax[0].scatter(s["depth_ft"], s["DR_obs"], s=12, alpha=0.45, marker=mk,
                      label=f"NFIP 2008 claims, {'with' if b else 'no'} basement (n={len(s)})")
    xs = np.linspace(-4, 24, 200)
    for key, lab in ((("1", False), "R11N"), (("2", False), "R12N"), (("1", True), "R11B"), (("2", True), "R12B")):
        ax[0].plot(xs, interp_curve(curves[key][0], xs), lw=1.6, label=f"Hazus default {lab}")
    ax[0].set_xlabel("Water depth relative to first floor (ft)")
    ax[0].set_ylabel("Building damage ratio")
    ax[0].set_xlim(-4.5, 15)
    ax[0].legend(fontsize=7)
    ax[0].set_title("Observed vs Hazus default curves")
    mid = [(a + b) / 2 for a, b in zip(DEPTH_BINS[:-1], DEPTH_BINS[1:])][:len(binned)]
    ax[1].plot(mid, binned["obs_mean"], "o-", label="Observed mean")
    ax[1].plot(mid, binned["pred_mean"], "s--", label="Hazus predicted mean")
    for i, (x, n) in enumerate(zip(mid, binned["n"])):
        # stagger labels so neighbouring narrow bins do not overlap
        ax[1].annotate(f"n={n}", (x, 0.02 + 0.05 * (i % 2)), fontsize=7, ha="center")
    ax[1].set_xlabel("Depth bin midpoint (ft)")
    ax[1].set_ylabel("Mean damage ratio")
    ax[1].set_ylim(0, 1)
    ax[1].legend()
    ax[1].set_title("Binned comparison")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "B_observed_vs_hazus.png"), dpi=170)
    plt.close(fig)

    contents = {"claims_with_DR_above_1_capped": int(prim["DR_obs_capped"].sum()),
                "spearman_obs_vs_depth_p": float(stats.spearmanr(prim["DR_obs"], prim["depth_ft"]).pvalue),
                "claims_2008_with_contents_damage": int(pd.to_numeric(d["contentsDamageAmount"], errors="coerce").gt(0).sum()),
                "claims_2008_with_contents_value_reported": int(pd.to_numeric(d["contentsPropertyValue"], errors="coerce").gt(0).sum())}
    return prim, metr, binned, cls, log, contents, {"primary": np.array(agg_samples), "screened": np.array(agg_screened)}


# ---------------------------------------------------------------- C. Monte Carlo
SIGMA_STAGE_FT = 0.5   # assumption, see summary; sensitivity at 0.25 and 1.0
SIGMA_FFE_FT = 1.0     # assumption, see summary; sensitivity at 0.5 and 2.0


def curve_family_factors(prim, curves, fams):
    """Sum of damage predicted by each alternative curve family / default, over the claims sample."""
    base = np.sum(prim["DR_pred"] * prim["rc"])
    out = {}
    for fam, fc in fams.items():
        pred = []
        for st, b, z in zip(prim["numberOfFloorsInTheInsuredBuilding"], prim["basement"], prim["depth_ft"]):
            key = (STORY_KEY[int(st)], bool(b))
            c = fc[key][0] if key in fc else curves[key][0]
            pred.append(interp_curve(c, z))
        out[fam] = {"factor": float(np.sum(np.array(pred) * prim["rc"]) / base),
                    "classes_covered": len([k for k in fc if k in curves]), "curve_ids": sorted(v[1] for v in fc.values())}
    return out


def part_c(fam_factors, agg_samples):
    fam_k = np.array([v["factor"] for v in fam_factors.values()])
    emp_k = 1.0 / agg_samples["primary"]  # observed / predicted, bootstrap distribution
    emp_k_scr = 1.0 / agg_samples["screened"]
    rows = []
    for sig_s, sig_f, tag in ((SIGMA_STAGE_FT, SIGMA_FFE_FT, "central"), (0.25, 0.5, "low"), (1.0, 2.0, "high")):
        for name, st in STAGES.items():
            eff = st + RNG.normal(0, sig_s, N_MC) - RNG.normal(0, sig_f, N_MC)
            s, c = in_app(eff)
            tot = s + c
            for variant, k in (("stage + FFE only", np.ones(N_MC)),
                               ("+ Hazus curve choice", RNG.choice(fam_k, N_MC)),
                               ("+ empirical 2008 bias, primary sample", RNG.choice(emp_k, N_MC)),
                               ("+ empirical 2008 bias, screened sample", RNG.choice(emp_k_scr, N_MC))):
                v = tot * k
                p5, p50, p95 = np.percentile(v, [5, 50, 95])
                rows.append({"sigma_set": tag, "sigma_stage_ft": sig_s, "sigma_ffe_ft": sig_f, "scenario": name,
                             "stage_ft": st, "variant": variant, "deterministic_total_usd": float(sum(in_app(st))),
                             "p5_usd": p5, "p50_usd": p50, "p95_usd": p95})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "C_monte_carlo_total_damage.csv"), index=False)
    return df


def part_c_depths(curves, fams, prim):
    """Damage-ratio spread at fixed depths above first floor: library curves vs 2008 observations."""
    rows = []
    for depth in (1.0, 3.0, 6.0):
        for key, lab in ((("1", False), "1 story, no basement"), (("2", False), "2 story, no basement"),
                         (("1", True), "1 story, basement"), (("2", True), "2 story, basement")):
            vals = [interp_curve(curves[key][0], depth)] + [interp_curve(fc[key][0], depth) for fc in fams.values() if key in fc]
            near = prim[(prim["depth_ft"].sub(depth).abs() <= 1.0) &
                        (prim["numberOfFloorsInTheInsuredBuilding"].map(STORY_KEY) == key[0]) &
                        (prim["basement"] == key[1])]["DR_obs"]
            rows.append({"depth_above_first_floor_ft": depth, "class": lab,
                         "hazus_default": interp_curve(curves[key][0], depth),
                         "library_min": min(vals), "library_max": max(vals), "n_library_curves": len(vals),
                         "observed_n_within_1ft": len(near),
                         "observed_p5": near.quantile(0.05) if len(near) >= 5 else np.nan,
                         "observed_median": near.median() if len(near) >= 5 else np.nan,
                         "observed_p95": near.quantile(0.95) if len(near) >= 5 else np.nan})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "C_damage_ratio_at_fixed_depths.csv"), index=False)
    return df


# ---------------------------------------------------------------- D. height-ratio overlay
def part_d(curves):
    """AC_DamageCalculator colours a building by clamp(local fluid height / mesh height, 0, 1)."""
    rows = []
    for stories, h_m, key in ((1, 3.0, ("1", False)), (2, 6.0, ("2", False)), (3, 9.0, ("3", False))):
        for depth_m in (0.5, 1.0, 2.0):
            hz = interp_curve(curves[key][0], depth_m * 3.28084)
            for err in (-0.3, 0.0, 0.3):
                mesh_h = h_m * (1 + err)
                rows.append({"stories": stories, "true_height_m": h_m, "mesh_height_error": err,
                             "mesh_height_m": mesh_h, "depth_m": depth_m,
                             "overlay_ratio": min(1.0, depth_m / mesh_h), "hazus_default_DR": hz})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "D_overlay_ratio_vs_hazus.csv"), index=False)
    osm = json.load(open(os.path.join(DATA, "osm_buildings_floodmap_bbox.json"), encoding="utf-8"))
    els = osm["elements"]
    cov = {"osm_buildings_in_bbox": len(els),
           "with_height_tag": sum("height" in e.get("tags", {}) for e in els),
           "with_levels_tag": sum("building:levels" in e.get("tags", {}) for e in els),
           "bbox_s_w_n_e": osm["bbox_s_w_n_e"]}
    return df, cov


# ---------------------------------------------------------------- main
def main():
    curves = load_default_curves()
    fams = load_alternative_families()
    a_df, a_info = part_a()
    prim, metr, binned, cls, log, contents, agg_samples = part_b(curves)
    fam_factors = curve_family_factors(prim, curves, fams)
    c_df = part_c(fam_factors, agg_samples)
    cd_df = part_c_depths(curves, fams, prim)
    d_df, osm_cov = part_d(curves)
    res = {"A": {"table": a_df.to_dict("records"), **a_info}, "B_filter_log": log,
           "B_metrics": metr.to_dict("index"), "B_other": contents,
           "C_curve_families": fam_factors, "D_osm_coverage": osm_cov,
           "default_curves": {f"{k[0]}{'B' if k[1] else 'N'}": v[1] for k, v in curves.items()}}
    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    import write_summary
    write_summary.write(res, a_df, metr, binned, cls, c_df, cd_df, d_df, os.path.join(HERE, "summary.md"))
    print("done")


if __name__ == "__main__":
    main()
