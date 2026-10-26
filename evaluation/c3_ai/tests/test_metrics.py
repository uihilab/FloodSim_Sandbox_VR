"""Metric functions checked against values computed by hand. Run: python tests/test_metrics.py"""

import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import common as C  # noqa: E402
import metrics as M  # noqa: E402


def close(a, b, tol=1e-9):
    return abs(a - b) < tol


def test_cohen_kappa():
    # 2x2 table [[20, 5], [10, 15]]: po = 35/50 = 0.70; pe = (25*30 + 25*20)/2500 = 0.50; kappa = 0.40
    a = [0] * 25 + [1] * 25
    b = [0] * 20 + [1] * 5 + [0] * 10 + [1] * 15
    assert close(M.cohen_kappa(a, b, [0, 1]), 0.4)


def test_fleiss_kappa():
    # Fleiss (1971)-style toy: 3 items, 3 raters, 2 categories.
    # rows [3,0], [0,3], [2,1]: P_i = 1, 1, (4+1-3)/6 = 1/3 -> P_bar = 7/9
    # p_j = (5/9, 4/9) -> P_e = 41/81; kappa = (7/9 - 41/81)/(1 - 41/81) = (22/81)/(40/81) = 0.55
    assert close(M.fleiss_kappa([[3, 0], [0, 3], [2, 1]]), 0.55)


def test_macro_f1_and_accuracy():
    yt = [0, 0, 1, 1]
    yp = [0, 1, 1, 1]
    # class 0: P=1, R=0.5, F1=2/3; class 1: P=2/3, R=1, F1=0.8 -> macro 0.7333...
    assert close(M.macro_f1(yt, yp, [0, 1, 2, 3]), (2 / 3 + 0.8) / 2)
    assert close(M.accuracy(yt, yp), 0.75)


def test_mcnemar():
    # b=1, c=6: p = 2 * P(X<=1 | n=7, 0.5) = 2 * 8/128 = 0.125
    assert close(M.mcnemar_exact(1, 6), 0.125)


def test_cosine():
    assert close(M.mean_pairwise_cosine([[1, 0], [0, 1], [1, 1]]), (0 + 2 * math.sqrt(0.5)) / 3)


def test_keyword_coder():
    c = C.keyword_code("Evacuate residents and close Riverside Drive; deploy sandbags.")
    assert c["evacuation"] == 1 and c["road_closure"] == 1 and c["barriers"] == 1 and c["pumping"] == 0


def test_production_prompt_matches_cpp():
    cpp = Path(__file__).resolve().parents[3] / "Source" / "VR_VIS" / "DataAnalysis.cpp"
    if not cpp.exists():
        print("  (DataAnalysis.cpp not found, skipped)")
        return
    m = re.search(r'SetStringField\("text",\s*"([^"]+)"\)', cpp.read_text(encoding="utf-8", errors="ignore"))
    assert m and m.group(1) == C.load_prompt("production_prompt.txt"), "production prompt drifted from DataAnalysis.cpp"
    src = cpp.read_text(encoding="utf-8", errors="ignore")
    assert '"gpt-4o-mini"' in src and 'SetNumberField("max_tokens", 300)' in src and "temperature" not in src


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
