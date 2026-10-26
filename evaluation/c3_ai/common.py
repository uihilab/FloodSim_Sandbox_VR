"""Shared definitions for the GPT-4o-mini technical evaluation.

Nothing in this module produces results. It holds the scene manifest loader, the
action taxonomy, the gazetteer, prompt loading and the MOCK client used only by
--dry-run to exercise the pipeline.
"""

import csv
import hashlib
import json
import random
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROMPTS = HERE / "prompts"

PRODUCTION_MODEL = "gpt-4o-mini"
PRODUCTION_MAX_TOKENS = 300
UPGRADE_MAX_COMPLETION_TOKENS = 4000
JUDGE_MODEL = "gpt-4o"
EMBED_MODEL = "text-embedding-3-small"

MOCK_MARK = "[MOCK RESPONSE - NOT REAL DATA]"

SCENARIO_DISCHARGE = {"dry": 0.0, "normal": 171.0, "flood": 568.0, "extreme": 705.0, "record2008": 1163.8}
TIER_NAMES = {0: "none", 1: "normal", 2: "flood", 3: "extreme"}

# Conditions. A = as deployed in DataAnalysis.cpp. B = A plus the simulation-state JSON the
# manuscript describes. C0/C1 = structured probe at temperature 0 / API default.
CONDITIONS = {
    "A": {"label": "As deployed (image + production prompt)", "structured": False},
    "B": {"label": "Grounded (A + simulation-state JSON)", "structured": False},
    "C0": {"label": "Structured probe, temperature 0", "structured": True, "temperature": 0.0},
    "C1": {"label": "Structured probe, default temperature", "structured": True},
    # Modality ablation (image-only vs structured-data-only vs combined). All three arms use the
    # same manuscript-style state JSON (state_v2: no severity label). Image-only arms are A and C0.
    "J": {"label": "Data only (state JSON, no image), production-style prompt", "structured": False, "image": False, "state": True},
    "K": {"label": "Image + state JSON, production prompt", "structured": False, "image": True, "state": True},
    "PJ": {"label": "Structured probe, data only, temperature 0", "structured": True, "temperature": 0.0, "image": False, "state": True},
    "PK": {"label": "Structured probe, image + data, temperature 0", "structured": True, "temperature": 0.0, "image": True, "state": True},
}
ABLATION_CONDITIONS = ("J", "K", "PJ", "PK")

# In-app loss lines (BP_SimControl) at the Iowa City gage stage, see evaluation/c2_hazus
def estimated_loss_usd(stage_ft):
    s = max(0.0, 5_688_154 * stage_ft - 134_925_113)
    c = max(0.0, 3_172_460 * stage_ft - 74_897_629)
    return round(s + c, -4)

# Action taxonomy. Keyword patterns are a transparent fallback coder; the LLM judge is the
# primary coder and the two are compared in score.py. high_severity marks actions that are
# only appropriate once people or property are threatened (used for escalation and
# over-reaction metrics).
TAXONOMY = [
    ("monitor_assess", "Monitor water levels/gages, inspect or assess conditions", False,
     r"monitor|gage|gauge|inspect|assess|surveil|sensor|observ|track(ing)? (water|river|levels)"),
    ("road_closure", "Close or barricade roads, bridges, underpasses; detour or reroute traffic", False,
     r"clos(e|ure|ing)\b[^.;\n]{0,40}\b(road|street|drive|bridge|underpass|lane|route|traffic|crossing)|barricad|detour|re-?rout|traffic control|road closure"),
    ("evacuation", "Order or advise evacuation of people", True,
     r"evacuat"),
    ("barriers", "Deploy sandbags, temporary barriers, floodwalls, levees, flood gates", False,
     r"sandbag|barrier|levee|flood ?wall|hesco|tiger dam|aqua ?dam|berm|\bdike|bulkhead|flood ?gate|temporary wall"),
    ("pumping", "Deploy pumps or dewatering", False,
     r"\bpump|dewater"),
    ("utilities", "De-energize or protect electrical, gas or other utilities", True,
     r"de-?energi|shut ?(off|down)[^.;\n]{0,30}(power|electric|gas|utilit)|(power|electric|gas|utilit)\w*[^.;\n]{0,30}shut ?(off|down)|transformer|substation|utility|utilities|electrical"),
    ("rescue_staging", "Stage rescue teams, boats, emergency responders", True,
     r"rescue|swift.?water|\bboats?\b|first responders?|emergency (services|responders|crews|teams)"),
    ("drainage", "Clear or protect drainage, storm sewers, culverts, backflow valves", False,
     r"drain|storm ?sewer|culvert|backflow|check valve|flap gate|catch basin|outfall|sewer"),
    ("debris", "Clear debris or obstructions", False,
     r"debris|logjam|obstruction"),
    ("shelter", "Open shelters, relocate people, move to higher ground", True,
     r"shelter|relocat|higher ground|safe zone|temporary housing"),
    ("critical_facilities", "Protect hospitals, schools, emergency, water or power facilities", True,
     r"hospital|clinic|school|critical (infrastructure|facilit)|fire station|police station|water treatment|wastewater|nursing"),
    ("public_comm", "Warn or inform the public, alerts, outreach", False,
     r"alert|warn|notify|notification|communicat|inform (the )?(public|residents)|social media|broadcast|announce|outreach|awareness|signage"),
    ("long_term", "Long-term mitigation: zoning, elevating or floodproofing buildings, green infrastructure, buyouts", False,
     r"zoning|land.?use|elevat(e|ing|ion of) (the )?(buildings|structures|homes)|floodproof|buy.?out|green infrastructure|rain garden|wetland|retention pond|detention|permeable|resilien|building code|retrofit|insurance"),
]
CATEGORY_KEYS = [t[0] for t in TAXONOMY]
HIGH_SEVERITY = [t[0] for t in TAXONOMY if t[2]]
_KEYWORD_RE = {t[0]: re.compile(t[3], re.I) for t in TAXONOMY}


def taxonomy_text():
    return "\n".join(f"- {k}: {d}" for k, d, _, _ in TAXONOMY)


def keyword_code(text):
    return {k: int(bool(_KEYWORD_RE[k].search(text or ""))) for k in CATEGORY_KEYS}


def load_prompt(name):
    return (PROMPTS / name).read_text(encoding="utf-8")


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def split_list(cell):
    if cell is None:
        return []
    cell = str(cell).strip()
    if not cell or cell.lower() == "nan":
        return []
    return [x.strip() for x in cell.split("|") if x.strip()]


MANIFEST_COLUMNS = [
    "scene_id", "image_file", "scenario", "discharge_m3s", "camera_bookmark", "view_type",
    "base_scene_id", "perturbation", "severity_tier_true", "gage_stage_ft",
    "visible_features", "wet_features", "probe_ids", "probe_depths_m", "probe_max_depth_m",
    "sim_time_s", "capture_width", "capture_height", "projection", "captured_by", "capture_date", "notes",
]


def load_manifest(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    missing = [c for c in ("scene_id", "image_file", "scenario", "severity_tier_true") if c not in (rows[0] if rows else {})]
    if missing:
        raise SystemExit(f"manifest {path} is missing columns: {missing}")
    base = Path(path).resolve().parent
    scenes = []
    for r in rows:
        if not r["scene_id"].strip():
            continue
        if r["scenario"] not in SCENARIO_DISCHARGE:
            raise SystemExit(f"{r['scene_id']}: scenario must be one of {list(SCENARIO_DISCHARGE)}")
        tier = r["severity_tier_true"].strip()
        if tier == "":
            raise SystemExit(f"{r['scene_id']}: severity_tier_true is blank; fill it per CAPTURE_PROTOCOL.md")
        img = Path(r["image_file"])
        if not img.is_absolute():
            img = base / img
        s = dict(r)
        s["image_path"] = str(img)
        s["severity_tier_true"] = int(tier)
        s["visible_features"] = split_list(r.get("visible_features"))
        s["wet_features"] = split_list(r.get("wet_features"))
        pmd = (r.get("probe_max_depth_m") or "").strip()
        s["probe_max_depth_m"] = float(pmd) if pmd else None
        s["base_scene_id"] = (r.get("base_scene_id") or "").strip() or r["scene_id"]
        s["perturbation"] = (r.get("perturbation") or "").strip() or "none"
        scenes.append(s)
    return scenes


def simulation_state(scene):
    """The JSON the manuscript says accompanies the image (condition B)."""
    def num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None
    return {
        "scenario": scene["scenario"],
        "upstream_discharge_m3s": SCENARIO_DISCHARGE[scene["scenario"]],
        "gage_stage_ft": num(scene.get("gage_stage_ft")),
        "usgs_gage": "05453520",
        "inundated_features": scene["wet_features"],
        "max_probe_depth_m": scene["probe_max_depth_m"],
    }


def simulation_state_v2(scene):
    """Manuscript-style simulation state (Section 2.4.4): discharge, gage stage, affected infrastructure,
    water depth and damage estimate. Deliberately carries no scenario name or severity label, so it
    does not hand the model the answer. Mismatch scenes take these fields from a different scenario."""
    def num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None
    stage = num(scene.get("gage_stage_ft"))
    return {
        "upstream_discharge_m3s": num(scene.get("discharge_m3s")),
        "usgs_gage": "05454500 Iowa River at Iowa City",
        "gage_stage_ft": stage,
        "inundated_infrastructure": scene["wet_features"],
        "max_water_depth_outside_mapped_channel_m": scene["probe_max_depth_m"],
        "fraction_of_area_in_view_inundated": num(scene.get("floodplain_wet_fraction")),
        "estimated_damage_usd_2023": estimated_loss_usd(stage) if stage is not None else None,
    }


# ---------------------------------------------------------------- gazetteer

def load_gazetteer(path=HERE / "gazetteer.csv"):
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    gaz = []
    for r in rows:
        names = [r["canonical_name"]] + split_list(r.get("aliases"))
        gaz.append({
            "canonical": r["canonical_name"],
            "patterns": [re.compile(r"\b" + re.escape(n) + r"\b", re.I) for n in names],
            "in_domain": (r.get("in_domain") or "").strip().lower(),
        })
    return gaz


def match_gazetteer(name, gaz):
    for g in gaz:
        if any(p.search(name) for p in g["patterns"]):
            return g["canonical"]
    return None


def gazetteer_mentions(text, gaz):
    return sorted({g["canonical"] for g in gaz if any(p.search(text or "") for p in g["patterns"])})


_NAMED_RE = re.compile(
    r"\b(?:[A-Z][a-zA-Z.']+\s){0,3}(?:Street|St\.?|Drive|Dr\.?|Avenue|Ave\.?|Road|Rd\.?|Boulevard|Blvd|"
    r"Bridge|Hall|Building|Hospital|Center|Centre|Park|Dam|Lake|Union|Auditorium|School|Station)\b"
    r"|\b(?:Building|Block|Zone|Sector)\s+[A-Z]?\d+[A-Z]?\b"
)


def regex_named_locations(text):
    """Offline fallback extraction of named places; the LLM judge is the primary extractor."""
    out = set()
    for m in _NAMED_RE.finditer(text or ""):
        s = m.group(0).strip()
        if len(s.split()) >= 2:
            out.add(s)
    return sorted(out)


# ---------------------------------------------------------------- MOCK client (dry run only)

class _Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class MockClient:
    """Stand-in for openai.OpenAI used only by --dry-run. Every text it returns starts with
    MOCK_MARK so mock output can never be mistaken for model output."""

    is_mock = True

    def __init__(self, seed=0):
        self.rng = random.Random(seed)
        self.chat = _Obj(completions=_Obj(create=self._chat))
        self.embeddings = _Obj(create=self._embed)

    def _chat(self, model, messages, **kw):
        text_parts = [c["text"] for m in messages for c in (m["content"] if isinstance(m["content"], list) else [{"type": "text", "text": m["content"]}]) if c.get("type") == "text"]
        prompt = "\n".join(text_parts)
        r = self.rng
        if "Return a single JSON object" in prompt and "ACTION CATEGORIES" in prompt:
            body = prompt.split("<<<", 1)[-1]
            cats = keyword_code(body)
            content = json.dumps({"categories": cats, "named_locations": regex_named_locations(body), "mock": MOCK_MARK})
        elif "severity_tier" in prompt:
            pool = ["monitor gage levels", "close Riverside Drive", "deploy sandbags", "evacuate low-lying buildings",
                    "stage pumps", "issue public alerts", "de-energize transformers", "clear storm drains"]
            content = json.dumps({
                "severity_tier": r.randint(0, 3),
                "flooded_features": r.sample(["Riverside Drive", "Iowa Memorial Union", "Main Street", "Building A34"], r.randint(0, 2)),
                "max_depth_estimate_m": round(r.uniform(0, 2), 2),
                "actions": r.sample(pool, r.randint(2, 5)),
                "note": MOCK_MARK,
            })
        else:
            pool = ["Monitor the river gage.", "Close Riverside Drive to traffic.", "Deploy sandbags near the IMU.",
                    "Evacuate residents of low-lying blocks.", "Install pumps.", "Alert the public via social media.",
                    "Clear storm drains of debris.", "Consider long-term zoning changes.", "Protect Mercy Hospital."]
            content = MOCK_MARK + " " + " ".join(r.sample(pool, r.randint(3, 6)))
        pt, ct = r.randint(800, 1200), r.randint(80, 300)
        return _Obj(
            id="mock-" + str(r.random())[2:10], model="MOCK-" + model, system_fingerprint="mock",
            choices=[_Obj(message=_Obj(content=content), finish_reason="stop")],
            usage=_Obj(prompt_tokens=pt, completion_tokens=ct, total_tokens=pt + ct),
        )

    def _embed(self, model, input):
        data = []
        for t in input:
            rr = random.Random(hashlib.md5(t.encode()).hexdigest())
            data.append(_Obj(embedding=[rr.gauss(0, 1) for _ in range(32)]))
        return _Obj(data=data)


def make_client(dry_run):
    if dry_run:
        return MockClient()
    from openai import OpenAI
    return OpenAI()  # reads OPENAI_API_KEY from the environment
