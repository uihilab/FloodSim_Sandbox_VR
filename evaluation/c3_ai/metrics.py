"""Pure metric functions used by score.py. Checked against hand-computed values in tests/."""

import math
import re
from collections import Counter
from itertools import combinations

import numpy as np
from scipy import stats


def confusion(y_true, y_pred, labels):
    idx = {l: i for i, l in enumerate(labels)}
    m = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            m[idx[t], idx[p]] += 1
    return m


def accuracy(y_true, y_pred):
    return float(np.mean([t == p for t, p in zip(y_true, y_pred)])) if len(y_true) else float("nan")


def macro_f1(y_true, y_pred, labels):
    """Mean F1 over labels that occur in y_true (labels absent from truth are skipped)."""
    f1s = []
    for l in labels:
        tp = sum(t == l and p == l for t, p in zip(y_true, y_pred))
        fp = sum(t != l and p == l for t, p in zip(y_true, y_pred))
        fn = sum(t == l and p != l for t, p in zip(y_true, y_pred))
        if tp + fn == 0:
            continue
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn)
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s)) if f1s else float("nan")


def cohen_kappa(a, b, labels=None, weights=None):
    """Cohen's kappa; weights=None, 'linear' or 'quadratic' (for ordinal tiers)."""
    if labels is None:
        labels = sorted(set(a) | set(b))
    m = confusion(a, b, labels).astype(float)
    n = m.sum()
    if n == 0:
        return float("nan")
    k = len(labels)
    i, j = np.meshgrid(range(k), range(k), indexing="ij")
    if weights is None:
        w = (i != j).astype(float)
    elif weights == "linear":
        w = np.abs(i - j) / max(k - 1, 1)
    else:
        w = ((i - j) / max(k - 1, 1)) ** 2
    expected = np.outer(m.sum(1), m.sum(0)) / n
    denom = (w * expected).sum()
    if denom == 0:
        return float("nan")
    return float(1 - (w * m).sum() / denom)


def fleiss_kappa(counts):
    """counts: items x categories matrix, each row summing to the number of raters."""
    counts = np.asarray(counts, dtype=float)
    n_items, _ = counts.shape
    r = counts.sum(1)
    if n_items == 0 or not np.allclose(r, r[0]) or r[0] < 2:
        return float("nan")
    r = r[0]
    p_j = counts.sum(0) / (n_items * r)
    P_i = ((counts ** 2).sum(1) - r) / (r * (r - 1))
    P_bar, P_e = P_i.mean(), (p_j ** 2).sum()
    if P_e == 1:
        return float("nan")
    return float((P_bar - P_e) / (1 - P_e))


def set_prf(pred, true):
    pred, true = set(pred), set(true)
    tp = len(pred & true)
    prec = tp / len(pred) if pred else float("nan")
    rec = tp / len(true) if true else float("nan")
    return tp, len(pred) - tp, len(true) - tp, prec, rec


def jaccard(a, b):
    a, b = set(a), set(b)
    return 1.0 if not a and not b else len(a & b) / len(a | b)


_TOKEN = re.compile(r"[a-z][a-z0-9'-]+")
STOP = set("the a an and or of to in on for with at by from is are be this that it as into near any all should can will may".split())


def tfidf_matrix(texts):
    docs = [[t for t in _TOKEN.findall((x or "").lower()) if t not in STOP] for x in texts]
    vocab = sorted({t for d in docs for t in d})
    if not vocab:
        return np.zeros((len(texts), 0))
    vi = {t: i for i, t in enumerate(vocab)}
    df = Counter(t for d in docs for t in set(d))
    n = len(docs)
    idf = np.array([math.log((1 + n) / (1 + df[t])) + 1 for t in vocab])
    m = np.zeros((n, len(vocab)))
    for r, d in enumerate(docs):
        for t, c in Counter(d).items():
            m[r, vi[t]] = c
    m *= idf
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return m / norms


def mean_pairwise_cosine(vectors):
    v = np.asarray(vectors, dtype=float)
    if len(v) < 2:
        return float("nan")
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1
    v = v / norms
    sims = [float(v[i] @ v[j]) for i, j in combinations(range(len(v)), 2)]
    return float(np.mean(sims))


def mcnemar_exact(b, c):
    """Exact two-sided McNemar test from discordant counts b (only first correct), c (only second)."""
    n = b + c
    if n == 0:
        return float("nan")
    return float(min(1.0, 2 * stats.binom.cdf(min(b, c), n, 0.5)))


def wilcoxon_paired(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if len(x) < 2 or np.allclose(x, y):
        return float("nan"), len(x)
    return float(stats.wilcoxon(x, y, zero_method="wilcox").pvalue), len(x)


def cluster_bootstrap_ci(values_by_cluster, stat=np.mean, n_boot=2000, alpha=0.05, seed=0):
    """Percentile CI resampling whole clusters (scenes), so repeated calls on one scene are not
    treated as independent. values_by_cluster: list of lists of per-call values."""
    clusters = [np.asarray(v, float) for v in values_by_cluster if len(v)]
    if not clusters:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    point = float(stat(np.concatenate(clusters)))
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(clusters), len(clusters))
        boots.append(stat(np.concatenate([clusters[i] for i in pick])))
    lo, hi = np.nanpercentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def describe(x):
    x = np.asarray([v for v in x if v is not None and not (isinstance(v, float) and math.isnan(v))], float)
    if not len(x):
        return {"n": 0}
    return {"n": int(len(x)), "mean": float(x.mean()), "sd": float(x.std(ddof=1)) if len(x) > 1 else float("nan"),
            "median": float(np.median(x)), "p90": float(np.percentile(x, 90)), "p95": float(np.percentile(x, 95)),
            "min": float(x.min()), "max": float(x.max())}
