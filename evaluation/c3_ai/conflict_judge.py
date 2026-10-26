"""Ask gpt-4o (temperature 0) whether each free-text answer (condition K) on an image/data mismatch scene
explicitly points out that the image and the simulation data disagree. Replaces a keyword heuristic that
produced false positives."""
import json
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from openai import OpenAI

from ablation import model_dir

PROMPT = """Below is an answer an AI assistant gave after receiving an aerial image of a city and a JSON block of flood-simulation data for the same view.

Does the answer explicitly state that the image and the simulation data disagree, are inconsistent, or that the image does not show the flooding the data describe (or vice versa)? General caveats such as "the image alone cannot establish X" do NOT count; only an explicit statement of a mismatch between what the image shows and what the data say counts.

Return JSON: {{"explicit_conflict": true or false, "quote": "<the sentence that states it, or empty>"}}

ANSWER:
{text}"""

cl = OpenAI()


def ask(text):
    r = cl.chat.completions.create(model="gpt-4o", temperature=0, response_format={"type": "json_object"},
                                   max_tokens=300, messages=[{"role": "user", "content": PROMPT.format(text=text)}])
    return json.loads(r.choices[0].message.content)


rows = []
for m in ("gpt-4o-mini", "gpt-5.6-terra", "gpt-6.1-sol"):
    d = pd.read_csv(model_dir(m) / "scored_mismatch" / "responses_scored.csv")
    d = d[d.condition == "K"].copy()
    with ThreadPoolExecutor(16) as pool:
        res = list(pool.map(ask, d.response_text.fillna("").tolist()))
    d["explicit_conflict"] = [bool(x.get("explicit_conflict")) for x in res]
    d["quote"] = [x.get("quote", "") for x in res]
    d["model"] = m
    d["pair"] = d.scene_id.str.replace(r"MM_(\w+?)_CAM\d__(\w+)", r"\1 image + \2 data", regex=True)
    rows.append(d[["model", "scene_id", "pair", "rep", "explicit_conflict", "quote"]])
out = pd.concat(rows)
out.to_csv(model_dir("gpt-4o-mini") / "ablation" / "conflict_judged.csv", index=False)
print(out.groupby(["model", "pair"]).explicit_conflict.mean().round(2).unstack())
print(out.groupby("model").explicit_conflict.mean().round(3))
print(out[out.explicit_conflict].groupby("model").quote.first())
