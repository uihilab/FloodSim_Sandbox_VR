"""Shared loaders: DEM, OSM building footprints and river polygon in UTM 15N."""
import json
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from shapely.geometry import Polygon, MultiPolygon, LineString
from shapely.ops import transform, unary_union, polygonize

DATA = Path(__file__).resolve().parents[1] / "data" / "c1"
_to_utm = Transformer.from_crs(4326, 26915, always_xy=True).transform


def load_dem():
    with rasterio.open(DATA / "dem_3dep_1m_utm15n.tif") as src:
        return src.read(1).astype(np.float64), src.transform, src.bounds


def _ring(geom):
    return [(p["lon"], p["lat"]) for p in geom]


def _relation_polys(el):
    outers = [LineString(_ring(m["geometry"])) for m in el.get("members", [])
              if m.get("role") == "outer" and m.get("geometry")]
    inners = [LineString(_ring(m["geometry"])) for m in el.get("members", [])
              if m.get("role") == "inner" and m.get("geometry")]
    out = unary_union(list(polygonize(unary_union(outers))))
    if inners:
        out = out.difference(unary_union(list(polygonize(unary_union(inners)))))
    return out


def load_osm():
    """Return (buildings, water) as lists of shapely polygons in UTM 15N."""
    els = json.loads((DATA / "osm_buildings_water.json").read_text(encoding="utf-8"))["elements"]
    buildings, water = [], []
    for el in els:
        tags = el.get("tags", {})
        if el["type"] == "way" and "geometry" in el and len(el["geometry"]) >= 4 \
                and el["geometry"][0] == el["geometry"][-1]:
            poly = Polygon(_ring(el["geometry"]))
        elif el["type"] == "relation":
            poly = _relation_polys(el)
        else:
            continue
        if poly.is_empty:
            continue
        poly = transform(_to_utm, poly.buffer(0))
        polys = list(poly.geoms) if isinstance(poly, MultiPolygon) else [poly]
        if "building" in tags:
            buildings += [(f"{el['type'][0]}{el['id']}_{k}", p) for k, p in enumerate(polys)]
        elif tags.get("natural") == "water" or tags.get("waterway") == "riverbank":
            water += polys
    return buildings, water
