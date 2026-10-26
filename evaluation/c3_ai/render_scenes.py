"""Render the 27 evaluation scenes and their ground truth from the hydrodynamic runs.

Each scene is an orthographic aerial view (as captured by the app's SceneCapture2D) made from
NAIP 0.5 m orthoimagery with the simulated floodwater from the 0.5 m reference runs of the
grid study (evaluation/c1_grid) composited on top. Because the water comes from a known model
state, the ground truth (which named streets and buildings are wet, how deep) is exact.

Scenes: 3 discharges x 6 cameras, 3 dry controls (imagery with no simulated water), and
6 camera perturbations of 2 flood scenes.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageEnhance
from shapely.geometry import LineString, Polygon, box
from shapely.ops import transform as stransform
from pyproj import Transformer
from rasterio.features import rasterize

HERE = Path(__file__).resolve().parent
R3 = HERE.parent
sys.path.insert(0, str(R3 / "c1_grid"))
from grids import build  # noqa: E402

SCENES = HERE / "scenes"
IMG = SCENES / "images"
IMG.mkdir(parents=True, exist_ok=True)
DATA3 = R3 / "data" / "c3"
RUNS = R3 / "data" / "c1" / "runs"

PX = 0.5
WET_M = 0.15          # a named feature counts as wet when water next to it exceeds this depth
SCENARIO_TIER = {"dry": 0, "normal": 1, "flood": 2, "extreme": 3, "record2008": 3}
Q = {"dry": 0.0, "normal": 171.0, "flood": 568.0, "extreme": 705.0, "record2008": 1163.8}
# Stage at USGS 05454500 Iowa River at Iowa City from the current USGS rating (evaluation/data/c2/usgs)
STAGE_IC_FT = {"normal": 15.95, "flood": 24.93, "extreme": 26.9, "record2008": 31.53}  # 2008: observed peak

# Camera windows: centre (feature name used to place it), width x height in metres
CAMERAS = {
    "CAM1": dict(center=None, w=None, h=None, desc="overview of the whole reach"),
    "CAM2": dict(center="Iowa Memorial Union", w=400, h=300, desc="Iowa Memorial Union riverfront"),
    "CAM3": dict(center="Art Building West", w=400, h=300, desc="Art Building West / North Riverside Drive"),
    "CAM4": dict(center="C. Maxwell Hydraulics Laboratory", w=400, h=300, desc="Burlington Street dam and bridge"),
    "CAM5": dict(center="Rise at River Front Crossings", w=400, h=300, desc="South Riverside / Riverfront Crossings"),
    "CAM6": dict(center="Old Capitol", w=400, h=300, desc="downtown east of the river"),
}
PERTURB = {"P1": dict(dx=40, dy=-30, rot=0, zoom=1.0, bright=1.0),
           "P2": dict(dx=0, dy=0, rot=12, zoom=1.0, bright=1.0),
           "P3": dict(dx=0, dy=0, rot=0, zoom=1.25, bright=0.85)}

_to_utm = Transformer.from_crs(4326, 26915, always_xy=True).transform


def named_features():
    els = json.loads((DATA3 / "osm_named.json").read_text(encoding="utf-8"))["elements"]
    feats = {}
    for e in els:
        name = e["tags"]["name"]
        if name == "unnamed alley":
            continue
        pts = [(p["lon"], p["lat"]) for p in e.get("geometry", [])] or \
              [(p["lon"], p["lat"]) for m in e.get("members", []) for p in m.get("geometry", [])]
        if len(pts) < 2:
            continue
        kind = "street" if "highway" in e["tags"] else "building" if "building" in e["tags"] else "place"
        g = Polygon(pts) if kind != "street" and pts[0] == pts[-1] and len(pts) >= 4 else LineString(pts)
        g = stransform(_to_utm, g)
        if name in feats:
            feats[name]["geoms"].append(g)
        else:
            feats[name] = dict(kind=kind, geoms=[g])
    return feats


WET_URBAN_M = 0.10
URBAN_MIN_M2 = 50.0      # tier 2: at least this much inundated street / building-perimeter area in view
WIDESPREAD_FRAC = 0.15   # tier 3: at least this fraction of the visible non-channel area inundated


def urban_mask(g):
    """Streets (OSM highways, bridges excluded, 4 m buffer) and 3 m rings around buildings, off the channel."""
    els = json.loads((DATA3 / "osm_named.json").read_text(encoding="utf-8"))["elements"]
    roads = [stransform(_to_utm, LineString([(p["lon"], p["lat"]) for p in e["geometry"]])).buffer(4.0)
             for e in els if "highway" in e["tags"] and e["tags"].get("bridge") != "yes" and len(e.get("geometry", [])) >= 2]
    from grids import building_polygons
    rings = [p.buffer(3.0).difference(p) for _, p in building_polygons()]
    m = rasterize(roads + rings, out_shape=g.z.shape, transform=g.transform, fill=0, default_value=1,
                  dtype="uint8", all_touched=True).astype(bool)
    return m & ~g.river & ~g.building


def visual_tier(scenario, d, view, urban, fp):
    """Tier by the probe's own visual definitions, computed from the simulation:
    0 no simulated water; 1 water only in/along the channel; 2 street or building-perimeter inundation;
    3 widespread inundation of the view."""
    if scenario == "dry":
        return 0
    urban_wet = float(((d > WET_URBAN_M) & urban & view).sum() * PX * PX)
    frac = float((d[fp] > WET_URBAN_M).mean()) if fp.any() else 0.0
    if frac >= WIDESPREAD_FRAC:
        return 3
    if urban_wet >= URBAN_MIN_M2:
        return 2
    return 1


def feature_depth(geoms, depth, tr, river):
    """Max floodplain depth within 3 m of a feature (outside the river channel)."""
    shapes = [(g.buffer(3.0).difference(g) if isinstance(g, Polygon) else g.buffer(3.0)) for g in geoms]
    m = rasterize(shapes, out_shape=depth.shape, transform=tr, fill=0, default_value=1,
                  dtype="uint8", all_touched=True).astype(bool) & ~river
    return float(depth[m].max()) if m.any() else 0.0


def composite(naip, h, g):
    """Muddy floodwater over the imagery wherever the simulated water surface is above the
    photographed ground (lidar surface), including inside the channel, so exposed bars and banks
    are covered at high flow. Opacity grows with depth above the photographed surface."""
    rgb = naip.astype(np.float32)
    above = np.where(g.building, 0.0, (g.z + h) - g.ground)
    water = (h > 0.02) & (above > 0.02)
    a = np.clip(0.55 + 0.4 * np.minimum(above / 1.5, 1.0), 0, 0.95) * water
    col = np.array([112, 100, 74], np.float32) - 30 * np.minimum(above / 2.0, 1.0)[..., None]
    out = rgb * (1 - a[..., None]) + col * a[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def crop(img, tr, cx, cy, w, h, rot=0.0, zoom=1.0, out_scale=2):
    W, H = int(w / PX), int(h / PX)
    col, row = ~tr * (cx, cy)
    pil = Image.fromarray(img)
    big = pil.crop((int(col - W), int(row - H), int(col + W), int(row + H)))
    if rot:
        big = big.rotate(rot, resample=Image.BICUBIC)
    cw, ch = int(W / zoom), int(H / zoom)
    c = big.crop((W - cw // 2, H - ch // 2, W - cw // 2 + cw, H - ch // 2 + ch))
    return c.resize((W * out_scale, H * out_scale), Image.BICUBIC)


def main():
    g = build(PX)
    ny, nx = g.z.shape
    with rasterio.open(DATA3 / "naip_utm15n_0p5m.tif") as src:
        naip = np.moveaxis(src.read(), 0, -1)[:ny, :nx]
    tr = g.transform
    feats = named_features()
    x0, y1 = tr.c, tr.f
    domain = box(x0, y1 - ny * PX, x0 + nx * PX, y1)
    centers = {}
    for cam, c in CAMERAS.items():
        if c["center"]:
            centers[cam] = feats[c["center"]]["geoms"][0].centroid.coords[0]

    depths = {"dry": np.zeros((ny, nx))}  # imagery as photographed, no simulated water
    for s in ("normal", "flood", "extreme", "record2008"):
        depths[s] = np.load(RUNS / f"{s}_dx0.5.npz")["h"].astype(np.float64)
        depths[s][g.building] = 0.0

    rows = []

    def add(scene_id, scenario, cam, img, window, base=None, pert="none"):
        img.save(IMG / f"{scene_id}.png")
        d = depths[scenario]
        cx, cy, w, h = window
        win = box(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2).intersection(domain)
        visible, wet = [], []
        for name, f in feats.items():
            gs = [gg for gg in f["geoms"] if gg.intersects(win)]
            if not gs:
                continue
            visible.append(name)
            if scenario != "dry" and feature_depth([gg.intersection(win.buffer(5)) for gg in gs], d, tr, g.river) > WET_M:
                wet.append(name)
        wm = rasterize([win], out_shape=d.shape, transform=tr, fill=0, default_value=1, dtype="uint8").astype(bool)
        fp = wm & ~g.river & ~g.building
        rows.append(dict(
            scene_id=scene_id, image_file=f"images/{scene_id}.png", scenario=scenario,
            discharge_m3s=Q[scenario], camera_bookmark=cam, view_type="aerial",
            base_scene_id=base or "", perturbation=pert,
            severity_tier_true=visual_tier(scenario, d, wm, urban, fp), scenario_tier=SCENARIO_TIER[scenario],
            urban_wet_area_m2=round(float(((d > WET_URBAN_M) & urban & wm).sum() * PX * PX), 1),
            floodplain_wet_fraction=round(float((d[fp] > WET_URBAN_M).mean()), 4) if fp.any() else 0.0,
            gage_stage_ft=STAGE_IC_FT.get(scenario, ""), visible_features="|".join(sorted(visible)),
            wet_features="|".join(sorted(wet)), probe_ids="",
            probe_depths_m="", probe_max_depth_m=round(float(d[fp].max()), 2) if fp.any() else 0.0,
            sim_time_s="", capture_width=img.size[0], capture_height=img.size[1], projection="orthographic",
            captured_by="render_scenes.py", capture_date="",
            notes=f"floodplain wet fraction {float((d[fp] > 0.1).mean()):.3f}" if fp.any() else ""))

    urban = urban_mask(g)
    for scenario in ("normal", "flood", "extreme", "dry", "record2008"):
        img = composite(naip, depths[scenario], g)
        cams = CAMERAS if scenario != "dry" else {k: CAMERAS[k] for k in ("CAM1", "CAM2", "CAM4")}
        for cam, c in cams.items():
            if cam == "CAM1":
                pil = Image.fromarray(img).resize((nx // 2, ny // 2), Image.BICUBIC)
                window = (x0 + nx * PX / 2, y1 - ny * PX / 2, nx * PX, ny * PX)
            else:
                cx, cy = centers[cam]
                pil = crop(img, tr, cx, cy, c["w"], c["h"])
                window = (cx, cy, c["w"], c["h"])
            add(f"{scenario}_{cam}", scenario, cam, pil, window)
        if scenario == "flood":
            for cam in ("CAM2", "CAM4"):
                for pk, p in PERTURB.items():
                    cx, cy = centers[cam][0] + p["dx"], centers[cam][1] + p["dy"]
                    c = CAMERAS[cam]
                    pil = crop(img, tr, cx, cy, c["w"], c["h"], rot=p["rot"], zoom=p["zoom"])
                    if p["bright"] != 1.0:
                        pil = ImageEnhance.Brightness(pil).enhance(p["bright"])
                    add(f"flood_{cam}_{pk}", "flood", cam, pil,
                        (cx, cy, c["w"] / p["zoom"], c["h"] / p["zoom"]), base=f"flood_{cam}", pert=pk)

    with open(SCENES / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Gazetteer: every named OSM feature in the domain is a real, in-domain name
    gz = HERE / "gazetteer.csv"
    seed = list(csv.DictReader(open(gz, encoding="utf-8")))
    have = {r["canonical_name"].lower() for r in seed}
    for r in seed:
        if r["in_domain"] == "TODO":
            r["in_domain"] = "yes" if any(r["canonical_name"].lower() in n.lower() or n.lower() in r["canonical_name"].lower()
                                          for n in feats) else "no"
    for name, f in sorted(feats.items()):
        if name.lower() not in have:
            seed.append(dict(canonical_name=name, aliases="", type=f["kind"], in_domain="yes",
                             source="OpenStreetMap (data/c3/osm_named.json)"))
    with open(gz, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["canonical_name", "aliases", "type", "in_domain", "source"])
        w.writeheader()
        w.writerows(seed)
    print(f"{len(rows)} scenes written; gazetteer has {len(seed)} names")
    for r in rows:
        print(r["scene_id"], r["severity_tier_true"], r["probe_max_depth_m"], len(r["visible_features"].split("|")),
              len([x for x in r["wet_features"].split("|") if x]), r["notes"])


if __name__ == "__main__":
    main()
