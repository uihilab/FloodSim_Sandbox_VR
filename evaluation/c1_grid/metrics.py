"""Compare every run against the 0.5 m reference and write tables, figures and summary numbers."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt

from grids import build, building_polygons
from run_experiment import CELL_SIZES, DISCHARGES

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / "data" / "c1" / "runs"
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)

REF_DX = 0.5
WET = 0.10        # m, depth threshold for "inundated"
RING_M = 3.0      # m, ring around each footprint used to read the water at a building
CANYON_M = 5.0    # m, open cells this close to a building count as street canyon


def load(name, dx):
    d = np.load(RUNS / f"{name}_dx{dx:g}.npz")
    meta = json.loads((RUNS / f"{name}_dx{dx:g}.json").read_text())
    return d["h"].astype(np.float64), d["u"].astype(np.float64), d["v"].astype(np.float64), meta


def to_ref(a, dx):
    k = int(round(dx / REF_DX))
    return a if k == 1 else np.repeat(np.repeat(a, k, axis=0), k, axis=1)


_rings = {}


def ring_cells(grid, polys):
    """Flat indices of open cells in a ring around each footprint, cached per cell size."""
    if grid.dx not in _rings:
        cells = []
        for bid, p in polys:
            ring = p.buffer(RING_M).difference(p)
            minx, miny, maxx, maxy = ring.bounds
            inv = ~grid.transform
            c0, r0 = [int(np.floor(v)) - 1 for v in inv * (minx, maxy)]
            c1, r1 = [int(np.ceil(v)) + 1 for v in inv * (maxx, miny)]
            r0, c0 = max(r0, 0), max(c0, 0)
            r1, c1 = min(r1, grid.z.shape[0]), min(c1, grid.z.shape[1])
            if r1 <= r0 or c1 <= c0:
                cells.append((bid, np.array([], dtype=np.int64)))
                continue
            win = grid.transform * grid.transform.translation(c0, r0)
            m = rasterize([ring], out_shape=(r1 - r0, c1 - c0), transform=win, fill=0,
                          default_value=1, dtype="uint8", all_touched=True).astype(bool)
            m &= ~grid.building[r0:r1, c0:c1]
            rr, cc = np.nonzero(m)
            cells.append((bid, (rr + r0) * grid.z.shape[1] + (cc + c0)))
        _rings[grid.dx] = cells
    return _rings[grid.dx]


def building_water(grid, h, polys):
    """Max depth and max water-surface elevation in a ring around each footprint, on the run's own grid."""
    hf = h.ravel()
    bed = np.where(grid.building, grid.ground, grid.z).ravel()
    wse = np.where(hf > 0, bed + hf, np.nan)
    rows = []
    for bid, idx in ring_cells(grid, polys):
        if idx.size == 0:
            rows.append((bid, np.nan, np.nan))
            continue
        w = wse[idx]
        rows.append((bid, float(hf[idx].max()), float(np.nanmax(w)) if np.isfinite(w).any() else np.nan))
    return pd.DataFrame(rows, columns=["building", "depth", "wse"]).set_index("building")


def gci(f1, f2, f3, r=2.0, fs=1.25):
    """Grid Convergence Index (Roache, 1994) for fine f1, medium f2, coarse f3."""
    e21, e32 = f2 - f1, f3 - f2
    if e21 == 0 or e32 == 0 or np.sign(e21) != np.sign(e32):
        return dict(p=np.nan, gci_fine_pct=np.nan, note="non-monotonic convergence")
    p = np.log(abs(e32 / e21)) / np.log(r)
    gci_fine = fs * abs(e21 / f1) / (r ** p - 1) * 100
    return dict(p=p, gci_fine_pct=gci_fine, note="monotonic convergence")


def main():
    polys = building_polygons()
    grids = {dx: build(dx) for dx in CELL_SIZES}
    ref_grid = grids[REF_DX]
    open_ref = ~ref_grid.building
    floodplain = open_ref & ~ref_grid.river
    canyon = floodplain & (distance_transform_edt(~ref_grid.building) * REF_DX <= CANYON_M)
    cell_a = REF_DX ** 2

    # Footprint representation at each cell size, against the true footprints clipped to the
    # domain and outside the river polygon (the part of each building the grids can represent)
    from shapely.geometry import box as _box
    from shapely.ops import unary_union as _uu
    ny0, nx0 = ref_grid.z.shape
    extent = _box(ref_grid.transform.c, ref_grid.transform.f - ny0 * REF_DX,
                  ref_grid.transform.c + nx0 * REF_DX, ref_grid.transform.f)
    from inputs import load_osm
    river_poly = _uu(load_osm()[1])
    clipped = [(bid, p.intersection(extent).difference(river_poly)) for bid, p in polys]
    clipped = [(bid, p) for bid, p in clipped if not p.is_empty and p.area > 0]
    true_area = sum(p.area for _, p in clipped)
    fp_rows = []
    for dx, g in grids.items():
        lost, small_lost = 0, 0
        for _, p in clipped:
            m = rasterize([p], out_shape=g.building.shape, transform=g.transform, fill=0,
                          default_value=1, dtype="uint8")
            if m.sum() == 0:
                lost += 1
        # open (non-building) cells in the 0.5 m grid that a coarser grid turns into building, and vice versa
        b = to_ref(g.building, dx)
        fp_rows.append(dict(dx=dx, n_buildings_in_domain=len(clipped),
                            rasterized_building_area_m2=g.building.sum() * dx * dx,
                            true_footprint_area_m2=true_area,
                            area_error_pct=100 * (g.building.sum() * dx * dx - true_area) / true_area,
                            buildings_not_represented=lost,
                            open_area_blocked_vs_0p5m_m2=float((b & ~ref_grid.building).sum() * REF_DX ** 2),
                            blocked_area_opened_vs_0p5m_m2=float((~b & ref_grid.building).sum() * REF_DX ** 2)))
    pd.DataFrame(fp_rows).to_csv(OUT / "footprint_representation.csv", index=False)

    rows, bld_rows, perf_rows = [], [], []
    for name in DISCHARGES:
        h_ref, u_ref, v_ref, _ = load(name, REF_DX)
        s_ref = np.hypot(u_ref, v_ref)
        wse_ref = np.where(ref_grid.building, ref_grid.ground, ref_grid.z) + h_ref
        b_ref = building_water(ref_grid, h_ref, polys)
        for dx in CELL_SIZES:
            h, u, v, meta = load(name, dx)
            perf_rows.append(dict(discharge=name, dx=dx, cells=int(np.prod(meta["shape"])),
                                  steps=meta["steps"], sim_time_s=meta["sim_time_s"],
                                  wall_time_s=meta["wall_time_s"],
                                  ms_per_step=1000 * meta["wall_time_s"] / meta["steps"],
                                  sim_s_per_wall_s=meta["sim_time_s"] / meta["wall_time_s"],
                                  converged=meta["converged"], outflow_m3s=meta["outflow_m3s"]))
            g = grids[dx]
            hr, sr = to_ref(h, dx), to_ref(np.hypot(u, v), dx)
            wser = to_ref(np.where(g.building, g.ground, g.z), dx) + hr
            wet_ref, wet = (h_ref > WET) & floodplain, (hr > WET) & floodplain
            tp, fp_, fn = (wet_ref & wet).sum(), (~wet_ref & wet).sum(), (wet_ref & ~wet).sum()
            both = wet_ref & wet
            cb = both & canyon
            dd, dw, ds = hr[both] - h_ref[both], wser[both] - wse_ref[both], sr[both] - s_ref[both]
            rows.append(dict(
                discharge=name, Q=DISCHARGES[name], dx=dx,
                flooded_area_ha=wet.sum() * cell_a / 1e4,
                flooded_area_ref_ha=wet_ref.sum() * cell_a / 1e4,
                area_change_pct=100 * (wet.sum() - wet_ref.sum()) / max(wet_ref.sum(), 1),
                csi=tp / max(tp + fp_ + fn, 1), hit_rate=tp / max(tp + fn, 1),
                false_alarm_ratio=fp_ / max(tp + fp_, 1), extent_bias=(tp + fp_) / max(tp + fn, 1),
                depth_bias_m=dd.mean(), depth_mae_m=np.abs(dd).mean(), depth_rmse_m=np.sqrt((dd ** 2).mean()),
                wse_bias_m=dw.mean(), wse_rmse_m=np.sqrt((dw ** 2).mean()),
                speed_rmse_ms=np.sqrt((ds ** 2).mean()),
                speed_p95_ms=np.percentile(sr[wet], 95) if wet.any() else np.nan,
                speed_p95_ref_ms=np.percentile(s_ref[wet_ref], 95) if wet_ref.any() else np.nan,
                canyon_depth_rmse_m=np.sqrt(((hr[cb] - h_ref[cb]) ** 2).mean()) if cb.any() else np.nan,
                canyon_speed_rmse_ms=np.sqrt(((sr[cb] - s_ref[cb]) ** 2).mean()) if cb.any() else np.nan,
                volume_floodplain_m3=(hr * floodplain).sum() * cell_a,
            ))
            b = building_water(g, h, polys)
            j = b_ref.join(b, lsuffix="_ref").dropna(subset=["depth_ref"])
            wet_r, wet_c = j["depth_ref"] > WET, j["depth"].fillna(0) > WET
            err = (j["depth"] - j["depth_ref"])[wet_r | wet_c].dropna()
            bld_rows.append(dict(
                discharge=name, dx=dx, buildings_compared=len(j),
                wet_buildings_ref=int(wet_r.sum()), wet_buildings=int(wet_c.sum()),
                newly_wet=int((~wet_r & wet_c).sum()), newly_dry=int((wet_r & ~wet_c).sum()),
                classification_agreement=float((wet_r == wet_c).mean()),
                no_cells_in_ring=int(j["depth"].isna().sum()),
                depth_err_median_m=float(err.median()) if len(err) else np.nan,
                depth_abs_err_median_m=float(err.abs().median()) if len(err) else np.nan,
                depth_abs_err_p90_m=float(err.abs().quantile(0.9)) if len(err) else np.nan,
                buildings_abs_err_gt_0p3m=int((err.abs() > 0.3).sum()),
            ))
            if name == "flood" and dx in (1.0, 4.0):
                np.save(OUT / f"_diff_{name}_dx{dx:g}.npy", np.where(open_ref, hr - h_ref, np.nan).astype(np.float32))

    ext = pd.DataFrame(rows)
    bld = pd.DataFrame(bld_rows)
    perf = pd.DataFrame(perf_rows)
    ext.to_csv(OUT / "extent_depth_velocity.csv", index=False)
    bld.to_csv(OUT / "building_metrics.csv", index=False)
    perf.to_csv(OUT / "solver_cost.csv", index=False)

    gci_rows = []
    for name in DISCHARGES:
        e = ext[ext.discharge == name].set_index("dx")
        for col in ("flooded_area_ha", "volume_floodplain_m3"):
            gci_rows.append(dict(discharge=name, quantity=col, f_0p5=e.loc[0.5, col], f_1=e.loc[1.0, col],
                                 f_2=e.loc[2.0, col], **gci(e.loc[0.5, col], e.loc[1.0, col], e.loc[2.0, col])))
    pd.DataFrame(gci_rows).to_csv(OUT / "gci.csv", index=False)

    figures(ext, bld, ref_grid)
    print(ext.round(3).to_string())
    print(bld.round(3).to_string())
    print(perf.round(2).to_string())
    print(pd.DataFrame(gci_rows).round(3).to_string())
    print(pd.DataFrame(fp_rows).round(2).to_string())


def figures(ext, bld, ref_grid):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for name, grp in ext.groupby("discharge", sort=False):
        axes[0].plot(grp.dx, grp.csi, "o-", label=name)
        axes[1].plot(grp.dx, grp.depth_rmse_m, "o-", label=name)
    for name, grp in bld.groupby("discharge", sort=False):
        axes[2].plot(grp.dx, grp.depth_abs_err_median_m, "o-", label=name)
    for ax, t in zip(axes, ["Inundation extent CSI vs 0.5 m", "Depth RMSE vs 0.5 m (m)",
                            "Median |depth error| at wet buildings (m)"]):
        ax.set_xscale("log", base=2)
        ax.set_xticks(CELL_SIZES, [f"{d:g}" for d in CELL_SIZES])
        ax.set_xlabel("Cell size (m)")
        ax.set_title(t, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig_metrics_vs_cellsize.png", dpi=200)

    fig, axes = plt.subplots(1, 2, figsize=(9, 8))
    for ax, dx in zip(axes, (1.0, 4.0)):
        d = np.load(OUT / f"_diff_flood_dx{dx:g}.npy")
        im = ax.imshow(d, cmap="RdBu_r", vmin=-0.5, vmax=0.5)
        ax.contour(ref_grid.building, levels=[0.5], colors="k", linewidths=0.2)
        ax.set_title(f"Depth difference, {dx:g} m minus 0.5 m\n(flood, 568 m³/s)", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes, shrink=0.6, label="m")
    fig.savefig(OUT / "fig_depth_difference_flood.png", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
