"""Real site outlines for US sites, replacing the capacity-based footprint guess.

* Solar: array polygons from the US Large-Scale Solar Photovoltaic Database
  (USPVDB, USGS / Berkeley Lab), keyed by EIA plant code. Public domain.
* Wind: the turbine field from the US Wind Turbine Database: convex hull of a
  project's turbines, widened by one rotor diameter.

``build(sites, turbines)`` returns one row per outlined site (area, radius from
the site point, array centroid) and writes footprints.parquet with each outline
as GeoJSON (simplified to ~10 m) for the warehouse and the map.
The USPVDB file is only parsed as data.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from shapely.geometry import MultiPoint, mapping, shape

from .. import config

KM_PER_DEG = 111.32
SIMPLIFY_DEG = 1e-4      # ~10 m: plenty for a map, and keeps the payload small
ROTOR_DEFAULT_M = 110.0  # typical modern rotor when USWTDB lacks one
CLUSTER_GAP_KM = 2.0     # turbines further apart than this belong to separate fields
HYBRID_MAX_KM = 20.0     # at solar+wind sites, ignore turbine fields this far from the arrays


def uspvdb_zip() -> Path | None:
    zips = sorted((config.RAW / "uspvdb").glob("uspvdb*.zip"))
    return zips[-1] if zips else None


def _to_km(geom, lon0: float, lat0: float):
    """Lon/lat geometry -> local km (equirectangular around lon0, lat0)."""
    c = np.cos(np.radians(lat0))
    return shapely.transform(geom, lambda xy: np.column_stack(((xy[:, 0] - lon0) * KM_PER_DEG * c,
                                                               (xy[:, 1] - lat0) * KM_PER_DEG)))


def _to_deg(geom, lon0: float, lat0: float):
    c = np.cos(np.radians(lat0))
    return shapely.transform(geom, lambda xy: np.column_stack((xy[:, 0] / (KM_PER_DEG * c) + lon0,
                                                               xy[:, 1] / KM_PER_DEG + lat0)))


def load_solar() -> pd.DataFrame:
    """One row per USPVDB facility: EIA plant code, outline, area and centroid."""
    z = uspvdb_zip()
    if z is None:
        print("[footprints] no USPVDB file in data/raw/uspvdb; solar sites keep estimated footprints")
        return pd.DataFrame(columns=["plant_code", "geom", "area_km2", "pv_lat", "pv_lon"])
    with zipfile.ZipFile(z) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".geojson"))
        with zf.open(name) as fh:
            feats = json.load(fh)["features"]
    rows = []
    for f in feats:
        p = f.get("properties") or {}
        code = pd.to_numeric(p.get("eia_id"), errors="coerce")
        if pd.isna(code) or not f.get("geometry"):
            continue
        g = shape(f["geometry"])
        if g.is_empty:
            continue
        rows.append({"plant_code": int(code), "geom": g,
                     "area_km2": (p.get("p_area") or 0) / 1e6 or None,
                     "pv_lat": p.get("ylat"), "pv_lon": p.get("xlong")})
    df = pd.DataFrame(rows)
    # A plant listed twice: merge its parts.
    df = df.groupby("plant_code", as_index=False).agg(
        geom=("geom", lambda s: shapely.union_all(list(s))), area_km2=("area_km2", "sum"),
        pv_lat=("pv_lat", "mean"), pv_lon=("pv_lon", "mean"))
    print(f"[footprints] USPVDB {z.name}: {len(df):,} solar outlines")
    return df


def wind_outlines(turbines: pd.DataFrame) -> pd.DataFrame:
    """Turbine-field outline per wind site_id.

    Turbines closer than CLUSTER_GAP_KM form one field; each field is the convex
    hull of its turbines widened by one rotor diameter. Scattered portfolios and
    USWTDB's per-county "unknown" projects so become several small fields rather
    than one hull spanning a county.
    """
    rows = []
    for sid, d in turbines.dropna(subset=["lat", "lon"]).groupby("site_id"):
        lon0, lat0 = float(d["lon"].mean()), float(d["lat"].mean())
        rotor_km = float(d["t_rd"].max()) / 1000 if d["t_rd"].notna().any() else ROTOR_DEFAULT_M / 1000
        pts = shapely.get_parts(_to_km(MultiPoint(d[["lon", "lat"]].to_numpy()), lon0, lat0))
        blobs = shapely.get_parts(shapely.union_all(shapely.buffer(pts, CLUSTER_GAP_KM / 2, quad_segs=2)))
        tree = shapely.STRtree(pts)
        fields = [MultiPoint(pts[tree.query(b, predicate="intersects")]).convex_hull.buffer(rotor_km, quad_segs=4)
                  for b in blobs]
        geom = shapely.union_all(fields)
        rows.append({"site_id": sid, "geom": _to_deg(geom, lon0, lat0), "area_km2": geom.area})
    return pd.DataFrame(rows, columns=["site_id", "geom", "area_km2"])


def _radius_km(geom, lon0: float, lat0: float) -> float:
    """Distance from the site point to the farthest point of its outline."""
    xy = shapely.get_coordinates(_to_km(geom, lon0, lat0))
    return float(np.hypot(xy[:, 0], xy[:, 1]).max()) if len(xy) else 0.0


def build(sites: pd.DataFrame, turbines: pd.DataFrame) -> pd.DataFrame:
    """Outline every US site we can. ``sites`` needs site_id, plant_code, lat, lon.

    Returns site_id, fp_kind, fp_area_km2, pv_lat, pv_lon, geom (lon/lat).
    Radius is computed later, once the final site point is known (see ``finish``).
    """
    solar = load_solar()
    solar["site_id"] = "eia:" + solar["plant_code"].astype(str)
    solar = solar[solar["site_id"].isin(sites["site_id"])]
    wind = wind_outlines(turbines[turbines["site_id"].isin(sites["site_id"])])
    s = solar.set_index("site_id")
    w = wind.set_index("site_id")
    ids = s.index.union(w.index)
    out = pd.DataFrame(index=ids)
    out["fp_kind"] = np.select(
        [ids.isin(s.index) & ids.isin(w.index), ids.isin(s.index)],
        ["solar arrays + turbine field", "solar arrays"], "turbine field")
    geoms, dropped, wind_area = [], 0, w["area_km2"].copy()
    for sid in ids:
        if sid in s.index and sid in w.index:
            # Hybrid: keep turbine fields near the arrays (a stray turbine with a
            # wrong plant id would otherwise stretch the site across a state).
            pv = s.at[sid, "geom"]
            c = pv.centroid
            pv_km = _to_km(pv, c.x, c.y)
            near = [f for f in shapely.get_parts(w.at[sid, "geom"])
                    if _to_km(f, c.x, c.y).distance(pv_km) <= HYBRID_MAX_KM]
            dropped += len(shapely.get_parts(w.at[sid, "geom"])) - len(near)
            geoms.append(shapely.union_all([pv, *near]))
            wind_area[sid] = sum(_to_km(f, c.x, c.y).area for f in near)
        else:
            geoms.append((s if sid in s.index else w).at[sid, "geom"])
    if dropped:
        print(f"[footprints] ignored {dropped} turbine field(s) more than {HYBRID_MAX_KM:.0f} km from their solar arrays")
    out["geom"] = geoms
    out["fp_area_km2"] = s["area_km2"].reindex(ids).fillna(0) + wind_area.reindex(ids).fillna(0)
    out["pv_lat"] = s["pv_lat"].reindex(ids)
    out["pv_lon"] = s["pv_lon"].reindex(ids)
    print(f"[footprints] {len(out):,} US sites outlined "
          f"({(out['fp_kind'] == 'solar arrays').sum():,} solar, {(out['fp_kind'] == 'turbine field').sum():,} wind, "
          f"{(out['fp_kind'] == 'solar arrays + turbine field').sum():,} both)")
    return out.reset_index(names="site_id")


def finish(fp: pd.DataFrame, sites: pd.DataFrame) -> pd.DataFrame:
    """Radius from each site's final point; write footprints.parquet. Returns
    site_id, fp_kind, fp_area_km2, fp_radius_km for joining onto the site table."""
    pts = sites.set_index("site_id")[["lat", "lon"]]
    fp = fp[fp["site_id"].isin(pts.index)].copy()
    fp["fp_radius_km"] = [_radius_km(g, pts.at[sid, "lon"], pts.at[sid, "lat"])
                          for sid, g in zip(fp["site_id"], fp["geom"])]
    simple = [g.simplify(SIMPLIFY_DEG, preserve_topology=True) for g in fp["geom"]]
    fp["geojson"] = [json.dumps(mapping(shapely.set_precision(g, 1e-5)), separators=(",", ":")) for g in simple]
    fp[["site_id", "fp_kind", "fp_area_km2", "fp_radius_km", "geojson"]].rename(
        columns={"fp_kind": "kind", "fp_area_km2": "area_km2", "fp_radius_km": "radius_km"}
    ).to_parquet(config.PROCESSED / "footprints.parquet", index=False)
    r = fp["fp_radius_km"]
    print(f"[footprints] radius from site point: median {r.median():.2f} km, 95th pct {r.quantile(.95):.1f} km, "
          f"max {r.max():.0f} km; outline file {sum(map(len, fp['geojson'])) / 1e6:.1f} MB")
    return fp[["site_id", "fp_kind", "fp_area_km2", "fp_radius_km"]]
