"""Download inputs for rendering the evaluation scenes: NAIP orthoimagery and named OSM features."""
import json
import datetime as dt
from pathlib import Path

import requests

OUT = Path(__file__).resolve().parents[1] / "data" / "c3"
OUT.mkdir(parents=True, exist_ok=True)
PROV = json.loads((OUT.parent / "c1" / "provenance.json").read_text())
X0, Y0, X1, Y1 = PROV["domain_utm15n"]
LON0, LAT0, LON1, LAT1 = PROV["domain_wgs84"]


def naip(px=0.5, tile=1000):
    """Mosaic NAIP natural-colour imagery at 0.5 m in tiles (the service caps request size)."""
    import numpy as np
    import rasterio
    from rasterio.io import MemoryFile
    from rasterio.transform import from_origin
    url = "https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPImagery/ImageServer/exportImage"
    w, h = int((X1 - X0) / px), int((Y1 - Y0) / px)
    rgb = np.zeros((3, h, w), np.uint8)
    for r0 in range(0, h, tile):
        for c0 in range(0, w, tile):
            tw, th = min(tile, w - c0), min(tile, h - r0)
            bx0, by1 = X0 + c0 * px, Y1 - r0 * px
            params = dict(bbox=f"{bx0},{by1 - th * px},{bx0 + tw * px},{by1}", bboxSR=26915, imageSR=26915,
                          size=f"{tw},{th}", format="tiff", pixelType="U8", bandIds="0,1,2", f="image")
            r = requests.get(url, params=params, timeout=600)
            r.raise_for_status()
            with MemoryFile(r.content) as mf, mf.open() as src:
                rgb[:, r0:r0 + th, c0:c0 + tw] = src.read()[:3]
    with rasterio.open(OUT / "naip_utm15n_0p5m.tif", "w", driver="GTiff", width=w, height=h, count=3,
                       dtype="uint8", crs="EPSG:26915", transform=from_origin(X0, Y1, px, px)) as dst:
        dst.write(rgb)
    return url, dict(px=px, tile=tile, bandIds="0,1,2")


def osm_named():
    q = f"""[out:json][timeout:120];
(
  way["highway"]["name"]({LAT0},{LON0},{LAT1},{LON1});
  way["building"]["name"]({LAT0},{LON0},{LAT1},{LON1});
  relation["building"]["name"]({LAT0},{LON0},{LAT1},{LON1});
  way["leisure"]["name"]({LAT0},{LON0},{LAT1},{LON1});
  way["amenity"]["name"]({LAT0},{LON0},{LAT1},{LON1});
);
out geom;"""
    for url in ("https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter",
                "https://overpass.private.coffee/api/interpreter"):
        try:
            r = requests.post(url, data={"data": q}, timeout=300, headers={"User-Agent": "FloodSim-Sandbox-research/1.0"})
            r.raise_for_status()
            (OUT / "osm_named.json").write_text(r.text, encoding="utf-8")
            return url, q
        except requests.RequestException as e:
            print("mirror failed", url, e)
    raise RuntimeError("Overpass failed")


if __name__ == "__main__":
    n_url, n_params = naip()
    o_url, o_q = osm_named()
    (OUT / "provenance.json").write_text(json.dumps(dict(
        retrieved_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        naip=dict(source="USDA NAIP via USGS The National Map ImageServer", url=n_url, params=n_params),
        osm=dict(source="OpenStreetMap contributors via Overpass (ODbL)", url=o_url, query=o_q)), indent=2))
