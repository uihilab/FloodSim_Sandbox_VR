"""Download the public data used for the damage-model validation.

(a) OpenFEMA NFIP Redacted Claims v3 (the FimaNfipClaims v2 endpoint has been retired), Johnson County,
    Iowa (county FIPS 19103, Iowa City) and Linn County (19113, Cedar Rapids, same June 2008 event), all years.
(b) Hazus 4.0 riverine depth-damage function lookup tables from FEMA's FAST repository.
(c) Hazus 4.0 structure/content DDF tables (flBldgStructDmgFn / flBldgContDmgFn) for curve metadata.

Raw files and provenance.json are written to ../data/c2/.
"""

import json
import os
from datetime import datetime, timezone

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "data", "c2"))
os.makedirs(OUT, exist_ok=True)

NFIP_URL = "https://www.fema.gov/api/open/v3/NfipClaims"
NFIP_FIELDS_URL = "https://www.fema.gov/api/open/v1/OpenFemaDataSetFields"
COUNTIES = {"19103": "Johnson", "19113": "Linn"}
FAST_TREE = "https://api.github.com/repos/nhrap-hazus/FAST/git/trees/HEAD?recursive=1"
FAST_RAW = "https://raw.githubusercontent.com/nhrap-hazus/FAST/{sha}/{path}"
FAST_FILES = [
    "Lookuptables/Building_DDF_Riverine_LUT_Hazus4p0.csv",
    "Lookuptables/Content_DDF_Riverine_LUT_Hazus4p0.csv",
    "Lookuptables/flBldgStructDmgFn.csv",
    "Lookuptables/flBldgContDmgFn.csv",
    "Lookuptables/DefaultDDFAssignmentTables.xlsx",
]


def fetch_nfip():
    out = []
    for code, name in COUNTIES.items():
        flt = f"countyCode eq '{code}'"
        rows, skip, top, total = [], 0, 1000, None
        while True:
            r = requests.get(NFIP_URL, params={"$filter": flt, "$top": top, "$skip": skip,
                                               "$inlinecount": "allpages"}, timeout=120)
            r.raise_for_status()
            js = r.json()
            batch = js.get("NfipClaims", [])
            rows.extend(batch)
            total = js.get("metadata", {}).get("count", total)
            skip += top
            if not batch or (total is not None and skip >= total):
                break
        fn = os.path.join(OUT, f"nfip_claims_{name.lower()}_county_IA.json")
        with open(fn, "w", encoding="utf-8") as f:
            json.dump(rows, f)
        out.append({"url": NFIP_URL, "filter": flt, "county": name, "file": os.path.basename(fn),
                    "rows": len(rows), "reported_count": total,
                    "rows_2008": sum(1 for x in rows if x.get("yearOfLoss") == 2008)})
    # Data dictionary (field units and code lists) for the fields used in the analysis.
    r = requests.get(NFIP_FIELDS_URL, params={
        "$filter": "openFemaDataSet eq 'NfipClaims' and datasetVersion eq 3", "$top": 500}, timeout=120)
    r.raise_for_status()
    with open(os.path.join(OUT, "nfip_claims_v3_data_dictionary.json"), "w", encoding="utf-8") as f:
        json.dump(r.json()["OpenFemaDataSetFields"], f, indent=1)
    return out


def fetch_fast():
    tree = requests.get(FAST_TREE, timeout=60)
    tree.raise_for_status()
    sha = tree.json()["sha"]
    out = []
    for p in FAST_FILES:
        url = FAST_RAW.format(sha=sha, path=p)
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        fn = os.path.join(OUT, os.path.basename(p))
        with open(fn, "wb") as f:
            f.write(r.content)
        n = r.content.count(b"\n") if p.endswith(".csv") else None
        out.append({"url": url, "file": os.path.basename(p), "bytes": len(r.content), "lines": n})
    return {"repo": "https://github.com/nhrap-hazus/FAST", "commit": sha, "files": out}


def fetch_osm_buildings():
    """OSM building ways (tags only) inside the bounding box of the app's IowaCity.geojson flood map."""
    geo = os.path.normpath(os.path.join(HERE, "..", "..", "FloodSim_Sandbox_VR 5.7", "IowaCity.geojson"))
    with open(geo, "r", encoding="utf-8") as f:
        g = json.load(f)
    xs = [p[0] for ft in g["features"] for p in ft["geometry"]["coordinates"][0]]
    ys = [p[1] for ft in g["features"] for p in ft["geometry"]["coordinates"][0]]
    bbox = (min(ys), min(xs), max(ys), max(xs))
    q = f'[out:json][timeout:90];way["building"]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});out tags center;'
    els, url, err = None, None, None
    for url in ("https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter",
                "https://overpass.private.coffee/api/interpreter"):
        try:
            r = requests.post(url, data={"data": q}, timeout=180,
                              headers={"User-Agent": "FloodSim-analysis/1.0 (research use)"})
            r.raise_for_status()
            els = r.json()["elements"]
            break
        except Exception as e:  # mirror busy or down; try the next one
            err = e
    if els is None:
        raise RuntimeError(f"All Overpass mirrors failed: {err}")
    with open(os.path.join(OUT, "osm_buildings_floodmap_bbox.json"), "w", encoding="utf-8") as f:
        json.dump({"query": q, "bbox_s_w_n_e": bbox, "elements": els}, f)
    return {"url": url, "query": q, "rows": len(els)}


if __name__ == "__main__":
    prov = {"retrieved_utc": datetime.now(timezone.utc).isoformat(),
            "nfip": fetch_nfip(), "hazus_fast": fetch_fast(), "osm": fetch_osm_buildings()}
    with open(os.path.join(OUT, "provenance.json"), "w", encoding="utf-8") as f:
        json.dump(prov, f, indent=2)
    print(json.dumps(prov, indent=2))
