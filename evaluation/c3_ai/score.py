"""Score the logged GPT-4o-mini responses. Every number written comes from the logs and the
scene manifest; nothing is typed in.

Usage:
    python score.py                       (real logs in results/, uses the gpt-4o judge)
    python score.py --no-judge            (keyword coder only, fully offline)
    python score.py --embeddings          (adds text-embedding-3-small consistency)
    python score.py --dry-run             (scores dry_run/responses_MOCK.jsonl with the MOCK client)
"""

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import common as C
import metrics as M

TIERS = [0, 1, 2, 3]


# ---------------------------------------------------------------- loading

def load_logs(path, dry_run):
    recs = {}
    n_err = 0
    for line in Path(path).open(encoding="utf-8"):
        r = json.loads(line)
        if r.get("source") == "MOCK" and not dry_run:
            sys.exit(f"{path} contains MOCK records; refusing to score them as real results.")
        if r.get("source") != "MOCK" and dry_run:
            sys.exit(f"{path} contains real records; dry run scores MOCK logs only.")
        if r.get("error"):
            n_err += 1
            continue
        recs[(r["scene_id"], r["condition"], r["rep"])] = r  # last successful record wins
    return list(recs.values()), n_err


def parse_structured(text):
    try:
        d = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    out = {}
    try:
        out["tier"] = int(d.get("severity_tier"))
        if out["tier"] not in TIERS:
            return None
    except (TypeError, ValueError):
        return None
    ff = d.get("flooded_features") or []
    out["flooded_features"] = [str(x) for x in ff] if isinstance(ff, list) else []
    try:
        out["depth"] = float(d.get("max_depth_estimate_m"))
    except (TypeError, ValueError):
        out["depth"] = None
    acts = d.get("actions") or []
    out["actions"] = [str(x) for x in acts] if isinstance(acts, list) else []
    return out


# ---------------------------------------------------------------- coding

class Judge:
    """gpt-4o (temperature 0) content coder with an on-disk cache keyed by the text hash."""

    def __init__(self, client, cache_path, enabled):
        self.client, self.enabled, self.cache_path = client, enabled, cache_path
        self.cache = {}
        if cache_path.exists():
            for line in cache_path.open(encoding="utf-8"):
                r = json.loads(line)
                self.cache[r["key"]] = r["result"]
        self.rubric = C.load_prompt("judge_rubric.txt")
        import threading
        self.lock = threading.Lock()

    def code(self, text):
        if not self.enabled:
            return None
        key = hashlib.sha256((C.JUDGE_MODEL + "\n" + self.rubric + "\n" + text).encode()).hexdigest()
        if key in self.cache:
            return self.cache[key]
        prompt = self.rubric.format(taxonomy=C.taxonomy_text(), text=text)
        d = None
        for attempt in range(5):
            try:
                resp = self.client.chat.completions.create(
                    model=C.JUDGE_MODEL, temperature=0, response_format={"type": "json_object"}, max_tokens=1500,
                    messages=[{"role": "user", "content": prompt}])
                d = json.loads(resp.choices[0].message.content)
                break
            except json.JSONDecodeError:
                continue  # truncated or malformed coder output; ask again
            except Exception as e:  # noqa: BLE001
                if attempt == 4:
                    raise
                import time
                time.sleep(2 ** attempt)
        if d is None:
            # Coder failed repeatedly: fall back to the keyword coder for this text and do not cache it
            return {"categories": C.keyword_code(text), "named_locations": C.regex_named_locations(text),
                    "fallback": "keyword"}
        cats = {k: int(bool(d.get("categories", {}).get(k, 0))) for k in C.CATEGORY_KEYS}
        result = {"categories": cats, "named_locations": [str(x) for x in d.get("named_locations", [])]}
        with self.lock:
            self.cache[key] = result
            with self.cache_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "result": result}) + "\n")
        return result


def expected_matrix():
    df = pd.read_csv(C.HERE / "expected_actions.csv")
    return {(int(r.tier), r.category): r.rating for r in df.itertuples()}


# ---------------------------------------------------------------- per-response table

def build_rows(records, scenes, judge, gaz):
    exp = expected_matrix()
    rows = []
    for r in records:
        s = scenes.get(r["scene_id"])
        if s is None:
            continue
        cond = r["condition"]
        text = r.get("response_text") or ""
        row = {
            "scene_id": s["scene_id"], "base_scene_id": s["base_scene_id"], "perturbation": s["perturbation"],
            "scenario": s["scenario"], "tier_true": s["severity_tier_true"], "condition": cond, "rep": r["rep"],
            "latency_s": r["latency_s"], "prompt_tokens": r["usage"]["prompt_tokens"],
            "completion_tokens": r["usage"]["completion_tokens"], "finish_reason": r.get("finish_reason"),
            "response_text": text,
        }
        coded_text = text
        if C.CONDITIONS[cond]["structured"]:
            p = parse_structured(text)
            row["parse_ok"] = p is not None
            if p:
                row["tier_pred"] = p["tier"]
                row["depth_pred"] = p["depth"]
                row["pred_features_raw"] = "|".join(p["flooded_features"])
                coded_text = "\n".join(p["actions"])
            row["depth_true"] = s["probe_max_depth_m"]
        row["coded_text"] = coded_text

        kw = C.keyword_code(coded_text)
        j = judge.code(coded_text) if coded_text.strip() else {"categories": {k: 0 for k in C.CATEGORY_KEYS}, "named_locations": []}
        cats = j["categories"] if j else kw
        row["coder"] = "judge" if j else "keyword"
        for k in C.CATEGORY_KEYS:
            row[f"cat_{k}"] = cats[k]
            row[f"kw_{k}"] = kw[k]

        tier = s["severity_tier_true"]
        expected = [k for k in C.CATEGORY_KEYS if exp.get((tier, k)) == "expected"]
        inapp = [k for k in C.CATEGORY_KEYS if exp.get((tier, k)) == "inappropriate"]
        row["n_categories"] = sum(cats.values())
        row["n_high_severity"] = sum(cats[k] for k in C.HIGH_SEVERITY)
        row["expected_coverage"] = (sum(cats[k] for k in expected) / len(expected)) if expected else np.nan
        row["n_inappropriate"] = sum(cats[k] for k in inapp)
        row["any_inappropriate"] = int(row["n_inappropriate"] > 0)

        # Grounding: every named place the response mentions (whole text, not only actions).
        names = (judge.code(text)["named_locations"] if j and text.strip() else C.regex_named_locations(text))
        if C.CONDITIONS[cond]["structured"] and row.get("pred_features_raw"):
            names = list(names) + row["pred_features_raw"].split("|")
        names = sorted(set(n.strip() for n in names if n.strip()))
        matched = {n: C.match_gazetteer(n, gaz) for n in names}
        unverified = [n for n, m in matched.items() if m is None]
        offscene = [n for n, m in matched.items() if m and s["visible_features"] and m not in s["visible_features"]]
        row["names"] = "|".join(names)
        row["n_names"] = len(names)
        row["unverified_names"] = "|".join(unverified)
        row["n_unverified"] = len(unverified)
        row["n_offscene"] = len(offscene)
        row["any_unverified"] = int(bool(unverified))

        if C.CONDITIONS[cond]["structured"] and row.get("parse_ok"):
            pred_c = {C.match_gazetteer(n, gaz) for n in (row.get("pred_features_raw") or "").split("|") if n}
            n_unmatched_pred = sum(1 for n in (row.get("pred_features_raw") or "").split("|") if n and C.match_gazetteer(n, gaz) is None)
            pred_c.discard(None)
            tp, fp, fn, _, _ = M.set_prf(pred_c, s["wet_features"])
            row["feat_tp"], row["feat_fp"], row["feat_fn"] = tp, fp + n_unmatched_pred, fn
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- analyses

def by_scene(df, col):
    return [g[col].dropna().tolist() for _, g in df.groupby("scene_id")]


def ci_str(point, lo, hi, pct=False):
    if np.isnan(point):
        return "n/a"
    f = (lambda v: f"{100 * v:.1f}%") if pct else (lambda v: f"{v:.3f}")
    return f"{f(point)} [{f(lo)}, {f(hi)}]"


def perception(df, out, figs):
    res, conf = [], {}
    for cond in [c for c in ("C0", "C1") if c in df.condition.unique()]:
        d = df[df.condition == cond]
        ok = d[d.parse_ok == True]  # noqa: E712
        d_ok = ok.assign(correct=(ok.tier_pred == ok.tier_true).astype(float))
        acc = M.cluster_bootstrap_ci(by_scene(d_ok, "correct"))
        dry = ok[ok.scenario == "dry"]
        fa = M.cluster_bootstrap_ci(by_scene(dry.assign(fa=(dry.tier_pred >= 1).astype(float)), "fa")) if len(dry) else (np.nan,) * 3
        under = ok[ok.tier_true >= 2]
        miss = float((under.tier_pred < under.tier_true).mean()) if len(under) else np.nan
        tp, fp, fn = ok.feat_tp.sum(), ok.feat_fp.sum(), ok.feat_fn.sum()
        dep = ok.dropna(subset=["depth_pred", "depth_true"])
        err = (dep.depth_pred - dep.depth_true)
        res.append({
            "condition": cond, "n_responses": len(d), "parse_rate": float(d.parse_ok.mean()),
            "tier_accuracy": acc[0], "tier_accuracy_lo": acc[1], "tier_accuracy_hi": acc[2],
            "macro_f1": M.macro_f1(ok.tier_true.tolist(), ok.tier_pred.tolist(), TIERS),
            "kappa_vs_truth": M.cohen_kappa(ok.tier_true.tolist(), ok.tier_pred.tolist(), TIERS),
            "kappa_quadratic_vs_truth": M.cohen_kappa(ok.tier_true.tolist(), ok.tier_pred.tolist(), TIERS, "quadratic"),
            "dry_false_alarm_rate": fa[0], "dry_false_alarm_lo": fa[1], "dry_false_alarm_hi": fa[2],
            "underestimate_rate_tier2plus": miss,
            "feature_precision": tp / (tp + fp) if tp + fp else np.nan,
            "feature_recall": tp / (tp + fn) if tp + fn else np.nan,
            "depth_n": len(dep), "depth_mae_m": float(err.abs().mean()) if len(dep) else np.nan,
            "depth_bias_m": float(err.mean()) if len(dep) else np.nan,
        })
        cm = M.confusion(ok.tier_true.tolist(), ok.tier_pred.tolist(), TIERS)
        conf[cond] = cm
        pd.DataFrame(cm, index=[f"true_{C.TIER_NAMES[t]}" for t in TIERS],
                     columns=[f"pred_{C.TIER_NAMES[t]}" for t in TIERS]).to_csv(out / f"confusion_{cond}.csv")
    pdf = pd.DataFrame(res)
    pdf.to_csv(out / "perception.csv", index=False)
    if conf:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(conf), figsize=(4.2 * len(conf), 3.8), squeeze=False)
        for ax, (cond, cm) in zip(axes[0], conf.items()):
            ax.imshow(cm, cmap="Blues")
            for i in range(4):
                for j in range(4):
                    ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=9)
            ax.set_xticks(range(4), [C.TIER_NAMES[t] for t in TIERS], rotation=30)
            ax.set_yticks(range(4), [C.TIER_NAMES[t] for t in TIERS])
            ax.set_xlabel("Predicted tier")
            ax.set_ylabel("True tier")
            ax.set_title(C.CONDITIONS[cond]["label"], fontsize=9)
        fig.tight_layout()
        fig.savefig(figs / "confusion_tiers.png", dpi=200)
        plt.close(fig)
    return pdf


def actions(df, out, figs):
    res = []
    for cond, d in df.groupby("condition"):
        cov = M.cluster_bootstrap_ci(by_scene(d, "expected_coverage"))
        inap = M.cluster_bootstrap_ci(by_scene(d, "any_inappropriate"))
        rho, p = (stats.spearmanr(d.tier_true, d.n_high_severity) if d.tier_true.nunique() > 1 else (np.nan, np.nan))
        sm = d.groupby("scene_id").agg(t=("tier_true", "first"), h=("n_high_severity", "mean"))
        rho_s, p_s = (stats.spearmanr(sm.t, sm.h) if sm.t.nunique() > 1 else (np.nan, np.nan))
        dry = d[d.scenario == "dry"]
        row = {
            "condition": cond, "n_responses": len(d),
            "expected_coverage": cov[0], "expected_coverage_lo": cov[1], "expected_coverage_hi": cov[2],
            "inappropriate_rate": inap[0], "inappropriate_lo": inap[1], "inappropriate_hi": inap[2],
            "escalation_spearman_rho_responses": rho, "escalation_p_responses": p,
            "escalation_spearman_rho_scene_means": rho_s, "escalation_p_scene_means": p_s,
            "dry_overreaction_rate": float((dry.n_high_severity > 0).mean()) if len(dry) else np.nan,
            "truncated_rate": float((d.finish_reason == "length").mean()),
            "mean_categories_per_response": float(d.n_categories.mean()),
        }
        for t in TIERS:
            dt_ = d[d.tier_true == t]
            row[f"mean_high_severity_tier{t}"] = float(dt_.n_high_severity.mean()) if len(dt_) else np.nan
        for k in C.CATEGORY_KEYS:
            row[f"share_{k}"] = float(d[f"cat_{k}"].mean())
        res.append(row)
    adf = pd.DataFrame(res)
    adf.to_csv(out / "actions.csv", index=False)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    w = 0.8 / max(len(adf), 1)
    for i, r in enumerate(adf.itertuples()):
        vals = [getattr(r, f"mean_high_severity_tier{t}") for t in TIERS]
        ax.bar(np.arange(4) + i * w, vals, w, label=r.condition)
    ax.set_xticks(np.arange(4) + w * (len(adf) - 1) / 2, [C.TIER_NAMES[t] for t in TIERS])
    ax.set_xlabel("True severity tier")
    ax.set_ylabel("High-severity action categories\nper response (mean)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figs / "escalation.png", dpi=200)
    plt.close(fig)
    return adf


def grounding(df, out):
    res = []
    for cond, d in df.groupby("condition"):
        u = M.cluster_bootstrap_ci(by_scene(d, "any_unverified"))
        res.append({"condition": cond, "n_responses": len(d), "mean_names": float(d.n_names.mean()),
                    "responses_with_unverified_name": u[0], "unverified_lo": u[1], "unverified_hi": u[2],
                    "mean_offscene_names": float(d.n_offscene.mean())})
    gdf = pd.DataFrame(res)
    gdf.to_csv(out / "grounding.csv", index=False)
    names = Counter(n for v in df.unverified_names for n in v.split("|") if n)
    pd.DataFrame(names.most_common(), columns=["name", "count"]).assign(
        author_verdict="", note="real Iowa City place in the scene? add to gazetteer.csv if so and re-score"
    ).to_csv(out / "unverified_names_for_review.csv", index=False)
    return gdf


def consistency(df, out, client, use_embeddings):
    rows = []
    for (scene, cond), d in df.groupby(["scene_id", "condition"]):
        if len(d) < 2:
            continue
        texts = d.coded_text.tolist()
        r = {"scene_id": scene, "condition": cond, "k": len(d),
             "tfidf_cosine": M.mean_pairwise_cosine(M.tfidf_matrix(texts))}
        if use_embeddings:
            emb = client.embeddings.create(model=C.EMBED_MODEL, input=[t or " " for t in texts])
            r["embedding_cosine"] = M.mean_pairwise_cosine([e.embedding for e in emb.data])
        counts = [[len(d) - d[f"cat_{k}"].sum(), d[f"cat_{k}"].sum()] for k in C.CATEGORY_KEYS]
        r["fleiss_kappa_categories"] = M.fleiss_kappa(counts)
        r["category_agreement"] = float(np.mean([max(c) / len(d) for c in counts]))
        if "tier_pred" in d and d.tier_pred.notna().any():
            tp = d.tier_pred.dropna()
            r["tier_modal_agreement"] = tp.value_counts().iloc[0] / len(tp)
        rows.append(r)
    cdf = pd.DataFrame(rows)
    cdf.to_csv(out / "consistency_by_scene.csv", index=False)
    summ = cdf.drop(columns="scene_id").groupby("condition").mean(numeric_only=True).reset_index()
    summ.to_csv(out / "consistency.csv", index=False)
    return summ


def robustness(df, out):
    pert = df[df.perturbation != "none"]
    rows = []
    for (scene, cond), d in pert.groupby(["scene_id", "condition"]):
        base = df[(df.scene_id == d.base_scene_id.iloc[0]) & (df.condition == cond)]
        if base.empty:
            continue
        maj = lambda x: {k for k in C.CATEGORY_KEYS if x[f"cat_{k}"].mean() >= 0.5}  # noqa: E731
        r = {"scene_id": scene, "base_scene_id": d.base_scene_id.iloc[0], "perturbation": d.perturbation.iloc[0],
             "condition": cond, "majority_category_jaccard": M.jaccard(maj(d), maj(base)),
             "high_severity_mean_diff": float(d.n_high_severity.mean() - base.n_high_severity.mean())}
        if "tier_pred" in d and d.tier_pred.notna().any() and base.tier_pred.notna().any():
            mt, mb = d.tier_pred.mode().iloc[0], base.tier_pred.mode().iloc[0]
            r.update(modal_tier_variant=mt, modal_tier_base=mb, tier_changed=int(mt != mb))
        rows.append(r)
    rdf = pd.DataFrame(rows)
    rdf.to_csv(out / "robustness.csv", index=False)
    return rdf


def comparisons(df, out):
    rows = []
    sm = df.groupby(["scene_id", "condition"]).mean(numeric_only=True).reset_index()

    def paired(c1, c2, col):
        a = sm[sm.condition == c1].set_index("scene_id")[col]
        b = sm[sm.condition == c2].set_index("scene_id")[col]
        j = a.to_frame("a").join(b.to_frame("b"), how="inner").dropna()
        p, n = M.wilcoxon_paired(j.a, j.b)
        rows.append({"comparison": f"{c1} vs {c2}", "metric": col + " (scene mean)", "n_scenes": n,
                     "mean_first": float(j.a.mean()) if n else np.nan, "mean_second": float(j.b.mean()) if n else np.nan,
                     "test": "Wilcoxon signed-rank", "p_value": p})

    for col in ("expected_coverage", "any_inappropriate", "n_high_severity", "any_unverified", "latency_s", "completion_tokens"):
        for c1, c2 in (("A", "B"), ("A", "C1"), ("C0", "C1")):
            if {c1, c2} <= set(df.condition):
                paired(c1, c2, col)
    if {"C0", "C1"} <= set(df.condition):
        ok = df[df.parse_ok == True]  # noqa: E712
        mod = ok.groupby(["scene_id", "condition"]).agg(t=("tier_true", "first"), p=("tier_pred", lambda x: x.mode().iloc[0]))
        mod["correct"] = mod.t == mod.p
        w = mod.correct.unstack("condition").dropna()
        if {"C0", "C1"} <= set(w.columns):
            b = int((w.C0 & ~w.C1).sum())
            c = int((~w.C0 & w.C1).sum())
            rows.append({"comparison": "C0 vs C1", "metric": "modal tier correct (per scene)", "n_scenes": len(w),
                         "mean_first": float(w.C0.mean()), "mean_second": float(w.C1.mean()),
                         "test": f"exact McNemar (b={b}, c={c})", "p_value": M.mcnemar_exact(b, c)})
    cdf = pd.DataFrame(rows)
    cdf.to_csv(out / "comparisons.csv", index=False)
    return cdf


def latency(df, out, figs):
    rows = []
    for cond, d in df.groupby("condition"):
        for col in ("latency_s", "prompt_tokens", "completion_tokens"):
            rows.append({"condition": cond, "variable": col, **M.describe(d[col].tolist())})
    ldf = pd.DataFrame(rows)
    ldf.to_csv(out / "latency_tokens.csv", index=False)
    import matplotlib.pyplot as plt
    conds = sorted(df.condition.unique())
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.boxplot([df[df.condition == c].latency_s for c in conds], tick_labels=conds, showfliers=True)
    ax.set_ylabel("API round-trip latency (s)")
    ax.set_xlabel("Condition")
    fig.tight_layout()
    fig.savefig(figs / "latency.png", dpi=200)
    plt.close(fig)
    return ldf


def coder_agreement(df, out, hand_dir):
    rows = []
    for k in C.CATEGORY_KEYS:
        rows.append({"pair": "keyword vs primary coder", "category": k,
                     "kappa": M.cohen_kappa(df[f"kw_{k}"].tolist(), df[f"cat_{k}"].tolist(), [0, 1]),
                     "agreement": float((df[f"kw_{k}"] == df[f"cat_{k}"]).mean()), "n": len(df)})
    key_path = out / "hand_coding_key.csv"
    c1p, c2p = hand_dir / "hand_coding_coder1.csv", hand_dir / "hand_coding_coder2.csv"
    if key_path.exists() and c1p.exists() and c2p.exists():
        key = pd.read_csv(key_path)
        c1, c2 = pd.read_csv(c1p), pd.read_csv(c2p)
        m = key.merge(c1, on="item_id", suffixes=("", "_c1")).merge(c2, on="item_id", suffixes=("_c1", "_c2"))
        m = m.merge(df[["scene_id", "condition", "rep"] + [f"cat_{k}" for k in C.CATEGORY_KEYS]],
                    on=["scene_id", "condition", "rep"])
        for k in C.CATEGORY_KEYS:
            a, b = m[f"{k}_c1"], m[f"{k}_c2"]
            ok = a.notna() & b.notna()
            if ok.sum():
                rows.append({"pair": "coder1 vs coder2", "category": k, "n": int(ok.sum()),
                             "kappa": M.cohen_kappa(a[ok].astype(int).tolist(), b[ok].astype(int).tolist(), [0, 1]),
                             "agreement": float((a[ok] == b[ok]).mean())})
                for cname, col in (("coder1", a), ("coder2", b)):
                    rows.append({"pair": f"{cname} vs primary coder", "category": k, "n": int(ok.sum()),
                                 "kappa": M.cohen_kappa(col[ok].astype(int).tolist(), m.loc[ok, f"cat_{k}"].tolist(), [0, 1]),
                                 "agreement": float((col[ok] == m.loc[ok, f"cat_{k}"]).mean())})
    adf = pd.DataFrame(rows)
    adf.to_csv(out / "coder_agreement.csv", index=False)
    return adf


def export_hand_sample(df, out, frac=0.2, seed=42):
    """Blind 20% stratified sample for two authors to code. Never overwritten once created."""
    sample_path, key_path = out / "hand_coding_sample.csv", out / "hand_coding_key.csv"
    if sample_path.exists():
        return
    parts = []
    for _, g in df.groupby(["condition", "scenario"]):
        n = max(1, int(round(frac * len(g))))
        parts.append(g.sample(n=min(n, len(g)), random_state=seed))
    s = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    s["item_id"] = [f"item_{i:04d}" for i in range(len(s))]
    s[["item_id", "scene_id", "condition", "rep"]].to_csv(key_path, index=False)
    blind = s[["item_id", "coded_text"]].rename(columns={"coded_text": "text"})
    for k in C.CATEGORY_KEYS:
        blind[k] = ""
    blind.to_csv(sample_path, index=False)


# ---------------------------------------------------------------- report

def fmt(v, pct=False, nd=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return f"{100 * v:.1f}%" if pct else f"{v:.{nd}f}"


def write_report(path, mock, meta, perc, act, grd, cons, rob, comp, lat, agree):
    L = []
    if mock:
        L += ["# MOCK PIPELINE TEST - NOT RESULTS", "",
              "Every number below comes from a MOCK client that returns random text. It only shows that the",
              "pipeline runs end to end. Do not cite anything in this file.", ""]
    else:
        L += ["# GPT-4o-mini technical evaluation: computed results", ""]
    L += [f"Generated by score.py from `{meta['log']}` ({meta['n_ok']} successful calls, {meta['n_err']} failed calls excluded), "
          f"{meta['n_scenes']} scenes. Action coder: {meta['coder']}. Intervals are 95% percentile bootstrap CIs resampling scenes.", ""]
    L += ["## Conditions", ""] + [f"- **{k}**: {v['label']}" for k, v in C.CONDITIONS.items() if k in meta["conditions"]] + [""]

    if len(perc):
        L += ["## Perception accuracy (structured probe)", "",
              "| Condition | Parse rate | Tier accuracy | Macro-F1 | Kappa vs truth (quadratic) | Dry false alarm | Under-estimate (tier 2-3) | Feature precision | Feature recall | Depth MAE (m) | Depth bias (m) |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in perc.itertuples():
            L.append(f"| {r.condition} | {fmt(r.parse_rate, True)} | {ci_str(r.tier_accuracy, r.tier_accuracy_lo, r.tier_accuracy_hi, True)} | "
                     f"{fmt(r.macro_f1)} | {fmt(r.kappa_vs_truth)} ({fmt(r.kappa_quadratic_vs_truth)}) | "
                     f"{ci_str(r.dry_false_alarm_rate, r.dry_false_alarm_lo, r.dry_false_alarm_hi, True)} | {fmt(r.underestimate_rate_tier2plus, True)} | "
                     f"{fmt(r.feature_precision, True)} | {fmt(r.feature_recall, True)} | {fmt(r.depth_mae_m, nd=2)} (n={r.depth_n}) | {fmt(r.depth_bias_m, nd=2)} |")
        L += ["", "Confusion matrices: `confusion_C0.csv`, `confusion_C1.csv`, `figures/confusion_tiers.png`.", ""]

    L += ["## Recommendations: coverage, appropriateness, escalation", "",
          "| Condition | Expected-action coverage | Any inappropriate action | Escalation rho (scene means, p) | Dry over-reaction | Truncated at max_tokens | High-severity per response by tier (none/normal/flood/extreme) |",
          "|---|---|---|---|---|---|---|"]
    for r in act.itertuples():
        L.append(f"| {r.condition} | {ci_str(r.expected_coverage, r.expected_coverage_lo, r.expected_coverage_hi, True)} | "
                 f"{ci_str(r.inappropriate_rate, r.inappropriate_lo, r.inappropriate_hi, True)} | "
                 f"{fmt(r.escalation_spearman_rho_scene_means)} (p={fmt(r.escalation_p_scene_means)}) | {fmt(r.dry_overreaction_rate, True)} | "
                 f"{fmt(r.truncated_rate, True)} | {' / '.join(fmt(getattr(r, f'mean_high_severity_tier{t}'), nd=2) for t in TIERS)} |")
    L += ["", "Expected/inappropriate categories per tier are defined in `expected_actions.csv` (author rubric). Figure: `figures/escalation.png`.", ""]

    L += ["## Grounding", "", "| Condition | Named places per response | Responses with an unverifiable name | Off-scene names per response |", "|---|---|---|---|"]
    for r in grd.itertuples():
        L.append(f"| {r.condition} | {fmt(r.mean_names, nd=2)} | {ci_str(r.responses_with_unverified_name, r.unverified_lo, r.unverified_hi, True)} | {fmt(r.mean_offscene_names, nd=2)} |")
    L += ["", "Unverifiable = not in `gazetteer.csv`. Review `unverified_names_for_review.csv` before reporting: genuine local names must be added to the gazetteer and the scoring re-run.", ""]

    L += ["## Consistency across repeated calls", ""]
    cols = [c for c in ("tfidf_cosine", "embedding_cosine", "fleiss_kappa_categories", "category_agreement", "tier_modal_agreement") if c in cons.columns]
    L += ["| Condition | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for r in cons.itertuples():
        L.append(f"| {r.condition} | " + " | ".join(fmt(getattr(r, c)) for c in cols) + " |")
    L += ["", "Per-scene values: `consistency_by_scene.csv`.", ""]

    if len(rob):
        L += ["## Robustness to camera perturbation", ""]
        g = rob.groupby("condition").agg(n=("scene_id", "count"), jac=("majority_category_jaccard", "mean"),
                                          **({"chg": ("tier_changed", "mean")} if "tier_changed" in rob else {}))
        L += ["| Condition | Perturbed scenes | Majority-category Jaccard vs base | Modal tier changed |", "|---|---|---|---|"]
        for c, r in g.iterrows():
            L.append(f"| {c} | {int(r.n)} | {fmt(r.jac)} | {fmt(r.get('chg', np.nan), True)} |")
        L += [""]

    if len(comp):
        L += ["## Paired comparisons between conditions", "", "| Comparison | Metric | n scenes | First | Second | Test | p |", "|---|---|---|---|---|---|---|"]
        for r in comp.itertuples():
            L.append(f"| {r.comparison} | {r.metric} | {r.n_scenes} | {fmt(r.mean_first)} | {fmt(r.mean_second)} | {r.test} | {fmt(r.p_value)} |")
        L += ["", "p-values are unadjusted; apply Holm correction if several are reported together.", ""]

    L += ["## Latency and tokens", "", "| Condition | Variable | n | Mean | Median | p95 | Max |", "|---|---|---|---|---|---|---|"]
    for r in lat.itertuples():
        L.append(f"| {r.condition} | {r.variable} | {r.n} | {fmt(r.mean, nd=2)} | {fmt(r.median, nd=2)} | {fmt(r.p95, nd=2)} | {fmt(r.max, nd=2)} |")
    L += ["", "Latency is the API round trip measured by run_eval.py from this machine; it excludes in-engine capture and encoding.", ""]

    L += ["## Coder agreement", ""]
    for pair, g in agree.groupby("pair"):
        L.append(f"- {pair}: mean kappa {fmt(g.kappa.mean())} over {len(g)} categories (n={int(g.n.iloc[0])} responses)")
    if not (agree.pair == "coder1 vs coder2").any():
        L.append("- Hand coding not yet available: two authors fill `hand_coding_coder1.csv` and `hand_coding_coder2.csv` (copies of `hand_coding_sample.csv`) and re-run.")
    L.append("")
    Path(path).write_text("\n".join(L), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", default=str(C.HERE / "results" / "responses.jsonl"))
    ap.add_argument("--manifest", default=str(C.HERE / "scenes" / "manifest.csv"))
    ap.add_argument("--out", default=str(C.HERE / "results" / "scored"))
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--embeddings", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.dry_run:
        root = C.HERE / "dry_run"
        args.log, args.manifest, args.out = root / "responses_MOCK.jsonl", root / "mock_scenes" / "manifest.csv", root / "results_MOCK"
    out = Path(args.out)
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        (out / "MOCK_README.txt").write_text("MOCK pipeline test output. Not results. Do not cite.\n", encoding="utf-8")

    scenes = {s["scene_id"]: s for s in C.load_manifest(args.manifest)}
    records, n_err = load_logs(args.log, args.dry_run)
    if not records:
        sys.exit("no successful records in the log")
    client = C.make_client(args.dry_run) if (not args.no_judge or args.embeddings) else None
    judge = Judge(client, out / "judge_cache.jsonl", enabled=not args.no_judge)
    gaz = C.load_gazetteer()

    # Judge calls are independent per response; code them in parallel (results are cached, order kept)
    from concurrent.futures import ThreadPoolExecutor
    chunks = [records[i::16] for i in range(16)]
    with ThreadPoolExecutor(max_workers=16) as pool:
        parts = list(pool.map(lambda ch: build_rows(ch, scenes, judge, gaz), chunks))
    df = pd.concat(parts, ignore_index=True).sort_values(["scene_id", "condition", "rep"]).reset_index(drop=True)
    df.to_csv(out / "responses_scored.csv", index=False)
    perc = perception(df, out, figs)
    act = actions(df, out, figs)
    grd = grounding(df, out)
    cons = consistency(df, out, client, args.embeddings)
    rob = robustness(df, out)
    comp = comparisons(df, out)
    lat = latency(df, out, figs)
    export_hand_sample(df, out)
    agree = coder_agreement(df, out, out)
    meta = {"log": str(args.log), "n_ok": len(df), "n_err": n_err, "n_scenes": df.scene_id.nunique(),
            "coder": (f"{C.JUDGE_MODEL} judge (temperature 0)" + (" [MOCK]" if args.dry_run else "")) if not args.no_judge else "keyword patterns only",
            "conditions": set(df.condition)}
    write_report(out / "results.md", args.dry_run, meta, perc, act, grd, cons, rob, comp, lat, agree)
    print(f"wrote {out / 'results.md'}")


if __name__ == "__main__":
    main()
