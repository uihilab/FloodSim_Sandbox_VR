"""Build the model grid (terrain, buildings, roughness, inflow) at a given cell size.

Every resolution uses the same DEM, the same building footprints, the same channel
treatment, roughness and boundary conditions. Only the cell size changes.
"""
from dataclasses import dataclass

import numpy as np
from affine import Affine
from rasterio.features import rasterize
from scipy.ndimage import zoom
from shapely.geometry import box
from shapely.ops import unary_union

from inputs import load_dem, load_osm

BLOCK = 4  # all grids are cropped to a multiple of the coarsest cell (4 m) so they nest exactly
CHANNEL_BURN_M = 3.0      # depth burned below the hydro-flattened lidar surface; set by calibrate_channel.py
BUILDING_RAISE_M = 30.0   # buildings are solid blocks in the heightfield, as in Fluid Flux
N_CHANNEL, N_FLOODPLAIN = 0.035, 0.030  # Manning's n (Chow, 1959 typical values)


@dataclass
class Grid:
    dx: float
    transform: Affine
    z: np.ndarray          # bed incl. buildings and channel burn (m, NAVD88)
    ground: np.ndarray     # bed without buildings (m)
    building: np.ndarray   # bool
    river: np.ndarray      # bool
    manning: np.ndarray
    inflow: np.ndarray     # bool, north-edge river cells
    h0: np.ndarray         # initial depth: channel filled to the flattened surface
    burn: float = CHANNEL_BURN_M


_cache = {}


def _base():
    if "base" not in _cache:
        dem, tr, bounds = load_dem()
        ny, nx = (dem.shape[0] // BLOCK) * BLOCK, (dem.shape[1] // BLOCK) * BLOCK
        dem = dem[:ny, :nx]
        buildings, water = load_osm()
        extent = box(tr.c, tr.f - ny * tr.a, tr.c + nx * tr.a, tr.f)
        water = unary_union(water).intersection(extent)
        _cache["base"] = (dem, tr, buildings, water, extent)
    return _cache["base"]


def _resample(dem1, dx):
    if dx == 1.0:
        return dem1.copy()
    if dx < 1.0:
        return zoom(dem1, 1.0 / dx, order=1)
    k = int(dx)
    ny, nx = dem1.shape
    return dem1.reshape(ny // k, k, nx // k, k).mean(axis=(1, 3))


def _calibrated_burn():
    import json
    from pathlib import Path
    f = Path(__file__).resolve().parent / "results" / "channel_calibration.json"
    return json.loads(f.read_text())["burn_m"] if f.exists() else CHANNEL_BURN_M


def build(dx, burn=None):
    dem1, tr1, buildings, water, _ = _base()
    ground = _resample(dem1, dx)
    ny, nx = ground.shape
    tr = Affine(dx, 0, tr1.c, 0, -dx, tr1.f)
    building = rasterize([p for _, p in buildings], out_shape=(ny, nx), transform=tr,
                         fill=0, default_value=1, dtype="uint8").astype(bool)
    river = rasterize([water], out_shape=(ny, nx), transform=tr,
                      fill=0, default_value=1, dtype="uint8").astype(bool)
    building &= ~river
    z = ground.copy()
    burn = _calibrated_burn() if burn is None else burn
    z[river] -= burn
    z[building] += BUILDING_RAISE_M
    manning = np.where(river, N_CHANNEL, N_FLOODPLAIN)
    inflow = np.zeros_like(river)
    inflow[0, :] = river[0, :]
    h0 = np.where(river, burn, 0.0)
    return Grid(dx, tr, z, ground, building, river, manning, inflow, h0, burn)


def building_polygons():
    return _base()[2]
