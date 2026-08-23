#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 14 (multi-hazard risk-factor heatmap) — clean replot.

Rows = the 14 Marmara provinces; columns = risk factors COMPUTED from your data
by per-province zonal statistics, each min-max normalised to [0,1], plus a
weighted composite column. Sequential, colour-blind-safe colormap; no red-green,
no self-citation.

Factors (each optional; a missing dataset just drops its column):
  Wind hazard        mean 100 m wind speed          (GWA GeoTIFF)
  Seismicity         M>=4 event density per 1000 km^2 (catalogue CSV)
  Coastal exposure   share of land <= 10 m elevation (GEBCO GeoTIFF)
  Population density persons per km^2                (TÜİK CSV + area)
  Urban built-up     CLC+ Sealed fraction            (CLC+ GeoTIFF)

Run:  python3 plot_riskheatmap_marmara.py     # -> Figure_15.png (300 dpi)
Requires: rasterio, numpy, pandas, geopandas, matplotlib
"""

import os, sys, csv, unicodedata

# --- PROJ/GDAL guard --------------------------------------------------------
for _v in ("PROJ_LIB", "PROJ_DATA", "GDAL_DATA"):
    if "gmt" in os.environ.get(_v, "").lower():
        os.environ.pop(_v, None)
_cp = os.path.join(sys.prefix, "share", "proj")
if os.path.exists(os.path.join(_cp, "proj.db")):
    os.environ["PROJ_DATA"] = _cp; os.environ["PROJ_LIB"] = _cp
_cg = os.path.join(sys.prefix, "share", "gdal")
if os.path.isdir(_cg):
    os.environ["GDAL_DATA"] = _cg
# ----------------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import geopandas as gpd
import rasterio
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from rasterio.features import geometry_mask
from affine import Affine

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
DATA_ROOT = os.environ.get(
    "DATA_ROOT",
    os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")),
)
TURKEY    = os.path.join(DATA_ROOT, "TURKEY")
PROV_JSON = os.path.join(DATA_ROOT, "geoBoundaries-TUR-ADM1.geojson")
WIND_TIF  = os.path.join(TURKEY, "TUR_wind-speed_100m.tif")
CLC_TIF   = os.path.join(DATA_ROOT, "CORINE_Land_Cover_Types",
                         "CLMS_CLCplus_RASTER_2018_010m_eu_03035_V1_1", "Data",
                         "CLMS_CLCplus_RASTER_2018_010m_eu_03035_V1_1.tif")
GEBCO_TIF = os.path.join(TURKEY, "GEBCO_Marmara", "gebco_2026_n42.0_s39.0_w25.0_e31.0_geotiff.tif")
CATALOG   = os.path.join(TURKEY, "IEB_Marmara.csv")
POP_CSV   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tuik_pop_2022.csv")

EXTENT    = (25.9, 31.1, 39.2, 41.9)
SEALED    = 1            # CLC+ Sealed code
SEIS_MINMAG = 4.0
WEIGHTS   = {}           # e.g. {"Seismicity":0.4,"Coastal exposure":0.3,...}; empty = equal
CMAP      = "YlOrRd"     # sequential, colour-blind-safe
OUT_NAME  = "Figure_15.png"

# ----------------------------------------------------------------------
def tr_norm(s):
    if s is None: return ""
    s = str(s)
    for a, b in (("İ","i"),("I","i"),("ı","i"),("Ş","s"),("ş","s"),("Ğ","g"),("ğ","g"),
                 ("Ç","c"),("ç","c"),("Ö","o"),("ö","o"),("Ü","u"),("ü","u")):
        s = s.replace(a, b)
    return unicodedata.normalize("NFKD", s).encode("ascii","ignore").decode().strip().lower()

MARMARA = {"istanbul","tekirdag","edirne","kirklareli","balikesir","canakkale","bursa",
           "eskisehir","bilecik","kocaeli","sakarya","duzce","bolu","yalova"}

def load_provinces():
    g = gpd.read_file(PROV_JSON).to_crs(4326)
    namef = next((c for c in ("shapeName","NAME","name","adm1_name") if c in g.columns), None)
    g["name"] = g[namef].astype(str); g["norm"] = g["name"].apply(tr_norm)
    g = g[g["norm"].isin(MARMARA)].copy()
    g["area_km2"] = g.to_crs(3035).area.values / 1e6
    lon0, lon1, lat0, lat1 = EXTENT
    return g.cx[lon0:lon1, lat0:lat1]

# ----------------------------------------------------------------------
def _read_window(path, categorical):
    lon0, lon1, lat0, lat1 = EXTENT
    with rasterio.open(path) as ds:
        if str(ds.crs).upper() not in ("EPSG:4326",):
            l, b, r, t = transform_bounds("EPSG:4326", ds.crs, lon0, lat0, lon1, lat1, densify_pts=21)
        else:
            l, b, r, t = lon0, lat0, lon1, lat1
        win = from_bounds(l, b, r, t, ds.transform).round_offsets().round_lengths()
        ww, wh = int(win.width), int(win.height)
        sc = min((3000 if categorical else 1800)/max(ww, 1), 1.0)
        ow, oh = max(int(ww*sc), 1), max(int(wh*sc), 1)
        arr = ds.read(1, window=win, out_shape=(oh, ow),
                      resampling=(Resampling.nearest if categorical else Resampling.average),
                      boundless=True, fill_value=(255 if categorical else np.nan)).astype("float64")
        tr = ds.window_transform(win) * Affine.scale(ww/ow, wh/oh)
        return arr, tr, ds.crs, ds.nodata

def zonal(prov, path, kind):
    """kind: 'mean' | 'sealed_frac' | 'lowland_frac'"""
    arr, tr, rcrs, nod = _read_window(path, categorical=(kind == "sealed_frac"))
    g = prov.to_crs(rcrs); oh, ow = arr.shape; out = {}
    for name, geom in zip(g["name"], g.geometry):
        m = geometry_mask([geom], (oh, ow), tr, invert=True)
        v = arr[m]
        if kind == "sealed_frac":
            v = v[v != 255]
            out[name] = float(np.mean(v == SEALED)) if v.size else np.nan
        elif kind == "lowland_frac":
            v = v[np.isfinite(v)]
            out[name] = float(np.mean((v >= 0) & (v <= 10))) if v.size else np.nan
        else:
            if nod is not None: v = v[v != nod]
            v = v[np.isfinite(v)]
            out[name] = float(np.nanmean(v)) if v.size else np.nan
    return pd.Series(out)

def seismicity_density(prov):
    df = pd.read_csv(CATALOG, sep=None, engine="python")
    low = {c.lower(): c for c in df.columns}
    lon = next((low[k] for k in low if "lon" in k), None)
    lat = next((low[k] for k in low if "lat" in k), None)
    mag = next((low[k] for k in low if k in ("mag","mw","ml") or "mag" in k), None)
    pts = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(pd.to_numeric(df[lon], errors="coerce"),
                                                           pd.to_numeric(df[lat], errors="coerce")), crs=4326)
    pts = pts[pd.to_numeric(df[mag], errors="coerce") >= SEIS_MINMAG]
    j = gpd.sjoin(pts, prov[["name", "geometry"]], predicate="within", how="inner")
    cnt = j.groupby("name").size()
    dens = (cnt / prov.set_index("name")["area_km2"] * 1000.0)          # per 1000 km^2
    return dens.reindex(prov["name"]).fillna(0.0)

def population_density(prov):
    pop = {}
    with open(POP_CSV, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            k = {c.lower(): c for c in row}
            pk = next((k[c] for c in ("province","il","name") if c in k), None)
            vk = next((k[c] for c in ("population","pop","nufus") if c in k), None)
            if pk and vk: pop[tr_norm(row[pk])] = float(str(row[vk]).replace(",", ""))
    dens = prov.set_index("norm").apply(lambda r: pop.get(r.name, np.nan)/r["area_km2"], axis=1)
    dens.index = prov["name"].values
    return dens

# ----------------------------------------------------------------------
def build_matrix(prov):
    cols = {}
    def add(label, fn):
        try:
            s = fn(); 
            if s.notna().any(): cols[label] = s; print(f"  ok: {label}")
            else: print(f"  skip (all NaN): {label}")
        except Exception as ex:
            print(f"  skip: {label} ({ex})")
    if os.path.exists(WIND_TIF):  add("Wind hazard",       lambda: zonal(prov, WIND_TIF, "mean"))
    if os.path.exists(CATALOG):   add("Seismicity",        lambda: seismicity_density(prov))
    if os.path.exists(GEBCO_TIF): add("Coastal exposure",  lambda: zonal(prov, GEBCO_TIF, "lowland_frac"))
    if os.path.exists(POP_CSV):   add("Population density", lambda: population_density(prov))
    if os.path.exists(CLC_TIF):   add("Urban built-up",    lambda: zonal(prov, CLC_TIF, "sealed_frac"))
    if not cols:
        sys.exit("No factors could be computed — check dataset paths at the top of the script.")
    M = pd.DataFrame(cols)
    M.index = M.index.map(lambda n: n)                 # province names
    # normalise each column to [0,1] (all factors: higher = higher risk)
    N = (M - M.min()) / (M.max() - M.min()).replace(0, np.nan)
    N = N.fillna(0.0)
    w = np.array([WEIGHTS.get(c, 1.0) for c in N.columns]); w = w / w.sum()
    N["Composite"] = (N.values * w).sum(axis=1)
    return N.sort_values("Composite", ascending=False)

# ----------------------------------------------------------------------
def render(N, out_path):
    provs = list(N.index); cols = list(N.columns)
    fig_h = max(5.5, 0.42 * len(provs) + 1.5)
    fig, ax = plt.subplots(figsize=(1.15*len(cols) + 3.5, fig_h))
    data = N.values
    im = ax.imshow(data, cmap=CMAP, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(provs))); ax.set_yticklabels(provs, fontsize=9)
    # separate the composite column
    ax.axvline(len(cols) - 1.5, color="white", lw=3)
    for i in range(len(provs)):
        for j in range(len(cols)):
            val = data[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if val > 0.6 else "#222",
                    fontweight="bold" if cols[j] == "Composite" else "normal")
    ax.set_xticks(np.arange(-.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(provs), 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.0); ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("Normalised factor score (0 = low, 1 = high)", fontsize=9)
    ax.set_title("Multi-hazard risk-factor heatmap — Marmara provinces",
                 fontsize=13, fontweight="bold", pad=10)
    ax.text(0.0, -0.14, "Each factor computed from source data by per-province zonal statistics and "
            "min-max normalised; composite = weighted mean of factors.",
            transform=ax.transAxes, ha="left", va="top", fontsize=7.5, color="#666")
    fig.tight_layout(); fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    print("written:", out_path)

def main():
    prov = load_provinces()
    print(f"  provinces: {len(prov)}")
    N = build_matrix(prov)
    render(N, os.path.join(os.path.dirname(os.path.abspath(__file__)), OUT_NAME))

if __name__ == "__main__":
    main()
