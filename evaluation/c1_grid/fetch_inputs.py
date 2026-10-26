"""Download the public inputs for the grid-resolution study.

- USGS 3DEP 1 m bare-earth DEM (NAVD88, metres), clipped via the 3DEP ImageServer.
- OpenStreetMap building footprints and water areas via the Overpass API.

The domain is the GeoJSON flood-map extent shipped with FloodSim Sandbox
(IowaCity.geojson) plus a buffer, projected to UTM 15N (EPSG:26915).
"""
import json
import datetime as dt
from pathlib import Path

import requests
from pyproj import Transformer

OUT = Path(__file__).resolve().parents[1] / "data" / "c1"
OUT.mkdir(parents=True, exist_ok=True)

# Flood-map extent from IowaCity.geojson is lon -91.5419..-91.5365, lat 41.6511..41.6621.
# Buffer it so the river enters and leaves the domain well away from the area of interest.
LON0, LON1, LAT0, LAT1 = -91.5480, -91.5310, 41.6460, 41.6680

to_utm = Transformer.from_crs(4326, 26915, always_xy=True)
x0, y0 = to_utm.transform(LON0, LAT0)
x1, y1 = to_utm.transform(LON1, LAT1)
x0, y0, x1, y1 = [round(v) for v in (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))]
W, H = x1 - x0, y1 - y0


def fetch_dem():
    url = "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage"
    params = {
        "bbox": f"{x0},{y0},{x1},{y1}", "bboxSR": 26915, "imageSR": 26915,
        "size": f"{W},{H}", "format": "tiff", "pixelType": "F32",
        "noDataInterpretation": "esriNoDataMatchAny", "interpolation": "RSP_BilinearInterpolation",
        "f": "image",
    }
    r = requests.get(url, params=params, timeout=300)
    r.raise_for_status()
    path = OUT / "dem_3dep_1m_utm15n.tif"
    path.write_bytes(r.content)
    return url, params, path


def fetch_osm():
    q = f"""[out:json][timeout:120];
(
  way["building"]({LAT0},{LON0},{LAT1},{LON1});
  relation["building"]({LAT0},{LON0},{LAT1},{LON1});
  way["natural"="water"]({LAT0},{LON0},{LAT1},{LON1});
  relation["natural"="water"]({LAT0},{LON0},{LAT1},{LON1});
  way["waterway"="riverbank"]({LAT0},{LON0},{LAT1},{LON1});
  way["waterway"="river"]({LAT0},{LON0},{LAT1},{LON1});
);
out geom;"""
    for url in ("https://overpass-api.de/api/interpreter",
                "https://overpass.kumi.systems/api/interpreter",
                "https://overpass.private.coffee/api/interpreter"):
        try:
            r = requests.post(url, data={"data": q}, timeout=300,
                              headers={"User-Agent": "FloodSim-Sandbox-research/1.0"})
            r.raise_for_status()
            break
        except requests.RequestException as e:
            print(f"Overpass mirror failed ({url}): {e}")
    else:
        raise RuntimeError("All Overpass mirrors failed")
    path = OUT / "osm_buildings_water.json"
    path.write_text(r.text, encoding="utf-8")
    return url, q, path


if __name__ == "__main__":
    dem_url, dem_params, dem_path = fetch_dem()
    osm_url, osm_q, osm_path = fetch_osm()
    prov = {
        "retrieved_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "domain_wgs84": [LON0, LAT0, LON1, LAT1],
        "domain_utm15n": [x0, y0, x1, y1],
        "dem": {"source": "USGS 3DEP 1/3 arc-second and 1 m composite, ImageServer exportImage",
                "url": dem_url, "params": dem_params, "file": dem_path.name},
        "osm": {"source": "OpenStreetMap contributors via Overpass API (ODbL)",
                "url": osm_url, "query": osm_q, "file": osm_path.name},
    }
    (OUT / "provenance.json").write_text(json.dumps(prov, indent=2))
    print(json.dumps({k: prov[k] for k in ("domain_utm15n",)}), W, H)
