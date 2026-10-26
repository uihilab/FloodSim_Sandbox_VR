"""Query GPT-4o-mini on the captured FloodSim scenes and log every raw response.

Usage:
    set OPENAI_API_KEY=...            (never stored in the logs)
    python run_eval.py --manifest scenes/manifest.csv --k 10
    python run_eval.py --dry-run      (MOCK client, mock scenes, writes only to dry_run/)

Conditions (see common.CONDITIONS):
    A   as deployed: the exact request built in DataAnalysis.cpp (image + production prompt,
        max_tokens 300, no temperature, PNG base64 data URL, gpt-4o-mini)
    B   A plus the simulation-state JSON the manuscript describes
    C0  structured JSON probe at temperature 0
    C1  structured JSON probe at the API default temperature

One JSONL line per call. Re-running resumes: (scene, condition, rep) cells that already
have a successful record are skipped.
"""

import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import base64
import datetime as dt
import json
import sys
import time
from pathlib import Path

import common as C

RETRYABLE = ("RateLimitError", "APITimeoutError", "APIConnectionError", "InternalServerError", "ServiceUnavailableError")


def image_data_url(path):
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode("ascii")


def build_request(scene, condition, model=C.PRODUCTION_MODEL):
    """Return (kwargs for chat.completions.create, prompt_text) with no secrets inside."""
    spec = C.CONDITIONS[condition]
    use_image = spec.get("image", True)
    if condition in ("A", "B", "K"):
        prompt = C.load_prompt("production_prompt.txt")
    elif condition == "J":
        prompt = C.load_prompt("production_prompt_data_only.txt")
    elif condition == "PJ":
        prompt = C.load_prompt("structured_probe_data.txt")
    elif condition == "PK":
        prompt = C.load_prompt("structured_probe_fused.txt")
    else:
        prompt = C.load_prompt("structured_probe.txt")
    if condition == "B":
        prompt = prompt + C.load_prompt("grounded_suffix.txt").format(state_json=json.dumps(C.simulation_state(scene), indent=2))
    elif spec.get("state"):
        prompt = prompt + C.load_prompt("grounded_suffix.txt").format(state_json=json.dumps(C.simulation_state_v2(scene), indent=2))
    # Same structure as DataAnalysis.cpp::MakeHTTPRequest: one user message, text then image.
    content = [{"type": "text", "text": prompt}]
    if use_image:
        content.append({"type": "image_url", "image_url": {"url": image_data_url(scene["image_path"])}})
    kwargs = {"model": model, "messages": [{"role": "user", "content": content}]}
    legacy = model.startswith("gpt-4o")
    if not spec["structured"]:
        if legacy:
            kwargs["max_tokens"] = C.PRODUCTION_MAX_TOKENS  # production value; temperature left unset
        else:
            # Newer models count reasoning in the completion budget, so a 300-token cap would return
            # empty answers. An upgraded app would set this budget; documented in summary.md.
            kwargs["max_completion_tokens"] = C.UPGRADE_MAX_COMPLETION_TOKENS
    else:
        kwargs["max_tokens" if legacy else "max_completion_tokens"] = 600 if legacy else C.UPGRADE_MAX_COMPLETION_TOKENS
        kwargs["response_format"] = {"type": "json_object"}
        if "temperature" in spec:
            kwargs["temperature"] = spec["temperature"]
    return kwargs, prompt


def call_with_retry(client, kwargs, max_tries=6, meta=None):
    delay = 2.0
    for attempt in range(1, max_tries + 1):
        t0 = time.perf_counter()
        try:
            resp = client.chat.completions.create(**kwargs)
            return resp, time.perf_counter() - t0, attempt, None
        except Exception as e:  # noqa: BLE001 - classify by name so the openai import stays optional
            name = type(e).__name__
            if name == "BadRequestError" and "temperature" in str(e) and "temperature" in kwargs:
                # Some newer models accept only their default temperature; record and retry without it
                kwargs.pop("temperature")
                if meta is not None:
                    meta["temperature_dropped"] = True
                continue
            if name in RETRYABLE and attempt < max_tries:
                time.sleep(delay)
                delay = min(delay * 2, 60)
                continue
            return None, time.perf_counter() - t0, attempt, f"{name}: {e}"


def done_cells(log_path):
    done = set()
    if log_path.exists():
        for line in log_path.open(encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("error") is None:
                done.add((r["scene_id"], r["condition"], r["rep"]))
    return done


def make_mock_scenes(out_dir):
    """Small synthetic PNGs + manifest so --dry-run can exercise every code path."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        ("MOCK_dry_CAM1", "dry", "CAM1", "", "none", 0, "", "", "0"),
        ("MOCK_normal_CAM1", "normal", "CAM1", "", "none", 1, "Iowa River|Riverside Drive", "Iowa River", "0.0"),
        ("MOCK_flood_CAM1", "flood", "CAM1", "", "none", 2, "Iowa River|Riverside Drive|Iowa Memorial Union", "Iowa River|Riverside Drive", "0.6"),
        ("MOCK_flood_CAM1_P1", "flood", "CAM1", "MOCK_flood_CAM1", "shift_10m", 2, "Iowa River|Riverside Drive|Iowa Memorial Union", "Iowa River|Riverside Drive", "0.6"),
        ("MOCK_extreme_CAM1", "extreme", "CAM1", "", "none", 3, "Iowa River|Riverside Drive|Iowa Memorial Union", "Iowa River|Riverside Drive|Iowa Memorial Union", "1.8"),
    ]
    lines = [",".join(C.MANIFEST_COLUMNS)]
    for sid, scen, cam, base, pert, tier, vis, wet, depth in rows:
        img = np.zeros((90, 160, 3))
        img[..., 1] = 0.5
        level = {"dry": 0, "normal": 15, "flood": 40, "extreme": 70}[scen]
        img[90 - level:, :, 2] = 0.8
        plt.imsave(out_dir / f"{sid}.png", img)
        vals = {c: "" for c in C.MANIFEST_COLUMNS}
        vals.update(scene_id=sid, image_file=f"{sid}.png", scenario=scen, discharge_m3s=str(C.SCENARIO_DISCHARGE[scen]),
                    camera_bookmark=cam, view_type="aerial", base_scene_id=base, perturbation=pert,
                    severity_tier_true=str(tier), gage_stage_ft="", visible_features=vis, wet_features=wet,
                    probe_max_depth_m=depth, notes="MOCK scene for pipeline test only")
        lines.append(",".join(vals[c] for c in C.MANIFEST_COLUMNS))
    (out_dir / "manifest.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_dir / "manifest.csv"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=str(C.HERE / "scenes" / "manifest.csv"))
    ap.add_argument("--out", default=str(C.HERE / "results" / "responses.jsonl"))
    ap.add_argument("--k", type=int, default=10, help="repetitions per scene and condition")
    ap.add_argument("--conditions", default="A,B,C0,C1")
    ap.add_argument("--scenes", default="", help="comma-separated scene_ids to restrict to")
    ap.add_argument("--dry-run", action="store_true", help="MOCK client + mock scenes; output to dry_run/")
    ap.add_argument("--workers", type=int, default=1, help="parallel requests (latency is still per request)")
    ap.add_argument("--model", default=C.PRODUCTION_MODEL, help="model to evaluate (default: the deployed gpt-4o-mini)")
    args = ap.parse_args()

    if args.dry_run:
        root = C.HERE / "dry_run"
        manifest = make_mock_scenes(root / "mock_scenes")
        out = root / "responses_MOCK.jsonl"
        k = min(args.k, 3)
        (root / "MOCK_README.txt").write_text(
            "Everything in this folder was produced by a MOCK client to test the pipeline.\n"
            "It is not GPT-4o-mini output and must never be reported as a result.\n", encoding="utf-8")
    else:
        manifest, out, k = Path(args.manifest), Path(args.out), args.k
        if args.model != C.PRODUCTION_MODEL and args.out == ap.get_default("out"):
            out = C.HERE / "results" / args.model / "responses.jsonl"
        if not manifest.exists():
            sys.exit(f"manifest not found: {manifest} (see CAPTURE_PROTOCOL.md)")

    scenes = C.load_manifest(manifest)
    if args.scenes:
        keep = set(args.scenes.split(","))
        scenes = [s for s in scenes if s["scene_id"] in keep]
    for s in scenes:
        if not Path(s["image_path"]).exists():
            sys.exit(f"image missing for {s['scene_id']}: {s['image_path']}")

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    for c in conditions:
        if c not in C.CONDITIONS:
            sys.exit(f"unknown condition {c}")

    client = C.make_client(args.dry_run)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = done_cells(out)
    def applicable(s, c):
        # Data arms need a simulation state: skip flood-free controls unless they are mismatch scenes
        return not (c in C.ABLATION_CONDITIONS and s["scenario"] == "dry" and not s["scene_id"].startswith("MM_"))
    todo = [(s, c, rep) for s in scenes for c in conditions for rep in range(k)
            if applicable(s, c) and (s["scene_id"], c, rep) not in done]
    print(f"{len(scenes)} scenes x {len(conditions)} conditions x k={k}: {len(todo)} calls to make, {len(done)} already logged")

    def one(s, cond, rep):
        kwargs, prompt = build_request(s, cond, args.model)
        meta = {}
        resp, latency, attempts, err = call_with_retry(client, kwargs, meta=meta)
        temperature_dropped = bool(meta.get("temperature_dropped"))
        rec = {
            "source": "MOCK" if args.dry_run else "openai",
            "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "scene_id": s["scene_id"], "condition": cond, "rep": rep,
            "image_sha256": C.sha256_file(s["image_path"]),
            "request": {k2: v for k2, v in kwargs.items() if k2 != "messages"},
            "prompt_text": prompt,
            "latency_s": round(latency, 4), "attempts": attempts, "error": err,
            "concurrent_workers": args.workers, "model_requested": args.model,
            "temperature_dropped": temperature_dropped,
        }
        if resp is not None:
            ch = resp.choices[0]
            rec.update({
                "response_id": resp.id, "response_model": resp.model,
                "system_fingerprint": getattr(resp, "system_fingerprint", None),
                "finish_reason": ch.finish_reason, "response_text": ch.message.content,
                "usage": {"prompt_tokens": resp.usage.prompt_tokens, "completion_tokens": resp.usage.completion_tokens,
                          "total_tokens": resp.usage.total_tokens},
            })
        return rec

    lock = threading.Lock()
    with out.open("a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(one, *t) for t in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            rec = fut.result()
            with lock:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
            if i % 50 == 0 or rec["error"]:
                print(f"  {i}/{len(todo)} {rec['scene_id']} {rec['condition']} rep{rec['rep']} "
                      f"{'ERROR ' + rec['error'] if rec['error'] else 'ok'}", flush=True)
    print(f"log: {out}")


if __name__ == "__main__":
    main()
