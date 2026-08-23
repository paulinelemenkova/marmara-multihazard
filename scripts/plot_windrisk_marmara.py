#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 12 (wind risk) of the Marmara basin — clean replot.

Wind RISK = wind HAZARD x urban EXPOSURE (no icons / infographic panels):
  * Hazard   -> Global Wind Atlas mean wind speed (100 m) GeoTIFF.
  * Exposure -> built-up fraction from Copernicus CLC+ Backbone (Sealed class).
  * Both are reprojected onto one WGS84 grid; risk = norm(wind) x builtup.
  * Province outlines + city labels -> geoBoundaries ADM1.

This is a transparent first-order hazard x exposure index (relative, unitless),
NOT a calibrated loss model — the caption says so.

Run:  python3 plot_windrisk_marmara.py        # -> Figure_13.png (300 dpi)
Requires: rasterio, numpy, matplotlib, geopandas
"""

import os, sys

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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
import geopandas as gpd
import rasterio
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds, reproject
from rasterio.transform import from_bounds as tfb
from affine import Affine

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
DATA_ROOT = os.environ.get(
    "DATA_ROOT",
    os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")),
)
TURKEY    = os.path.join(DATA_ROOT, "TURKEY")
WIND_TIF  = os.path.join(TURKEY, "TUR_wind-speed_100m.tif")                 # GWA 100 m
CLC_TIF   = os.path.join(DATA_ROOT, "CORINE_Land_Cover_Types",
                         "CLMS_CLCplus_RASTER_2018_010m_eu_03035_V1_1", "Data",
                         "CLMS_CLCplus_RASTER_2018_010m_eu_03035_V1_1.tif")
PROV_JSON = os.path.join(DATA_ROOT, "geoBoundaries-TUR-ADM1.geojson")

EXTENT    = (25.9, 31.1, 39.2, 41.9)
TARGET_W  = 720          # analysis grid width (cells)
SEALED_CODE = 1          # CLC+ Backbone: 1 = Sealed (built-up)
W_EXP     = 0.5          # exposure exponent (0.5 = geometric-ish weighting)
CMAP      = "YlOrRd"     # sequential, colour-blind-safe
OUT_NAME  = "Figure_13.png"

CITIES = [("İstanbul", 28.98, 41.02), ("İzmit", 29.92, 40.77), ("Bursa", 29.06, 40.19),
          ("Tekirdağ", 27.51, 40.98), ("Yalova", 29.28, 40.65), ("Bandırma", 27.97, 40.35),
          ("Çanakkale", 26.41, 40.15)]

# ----------------------------------------------------------------------
def geo_aspect(e): return 1.0 / np.cos(np.deg2rad(0.5 * (e[2] + e[3])))

def dst_grid():
    lon0, lon1, lat0, lat1 = EXTENT
    w = TARGET_W
    h = max(int(round(w * (lat1-lat0) / (lon1-lon0) / np.cos(np.deg2rad(0.5*(lat0+lat1))))), 1)
    return w, h, tfb(lon0, lat0, lon1, lat1, w, h)

def read_wind(dw, dh, dtr):
    lon0, lon1, lat0, lat1 = EXTENT
    with rasterio.open(WIND_TIF) as ds:
        l, b, r, t = transform_bounds("EPSG:4326", ds.crs, lon0, lat0, lon1, lat1, densify_pts=21)
        win = from_bounds(l, b, r, t, ds.transform).round_offsets().round_lengths()
        ww, wh = int(win.width), int(win.height)
        sc = min(1600/max(ww, 1), 1.0); ow, oh = max(int(ww*sc), 1), max(int(wh*sc), 1)
        arr = ds.read(1, window=win, out_shape=(oh, ow), resampling=Resampling.average,
                      boundless=True, fill_value=np.nan).astype("float64")
        if ds.nodata is not None:
            arr[arr == ds.nodata] = np.nan
        str_ = ds.window_transform(win) * Affine.scale(ww/ow, wh/oh)
        crs = ds.crs
    dst = np.full((dh, dw), np.nan)
    reproject(arr, dst, src_transform=str_, src_crs=crs, dst_transform=dtr,
              dst_crs="EPSG:4326", resampling=Resampling.average,
              src_nodata=np.nan, dst_nodata=np.nan)
    return dst

def read_builtup(dw, dh, dtr):
    lon0, lon1, lat0, lat1 = EXTENT
    with rasterio.open(CLC_TIF) as ds:
        l, b, r, t = transform_bounds("EPSG:4326", ds.crs, lon0, lat0, lon1, lat1, densify_pts=21)
        win = from_bounds(l, b, r, t, ds.transform).round_offsets().round_lengths()
        ww, wh = int(win.width), int(win.height)
        sc = min(3000/max(ww, 1), 1.0); ow, oh = max(int(ww*sc), 1), max(int(wh*sc), 1)
        codes = ds.read(1, window=win, out_shape=(oh, ow), resampling=Resampling.nearest,
                        boundless=True, fill_value=255)
        sealed = (codes == SEALED_CODE).astype("float64")
        str_ = ds.window_transform(win) * Affine.scale(ww/ow, wh/oh)
        crs = ds.crs
    dst = np.zeros((dh, dw))
    reproject(sealed, dst, src_transform=str_, src_crs=crs, dst_transform=dtr,
              dst_crs="EPSG:4326", resampling=Resampling.average)     # -> built-up fraction
    return dst

def compute_risk():
    dw, dh, dtr = dst_grid()
    wind = read_wind(dw, dh, dtr)
    built = read_builtup(dw, dh, dtr)
    land = np.isfinite(wind)
    p2, p98 = np.nanpercentile(wind[land], [2, 98]) if land.any() else (0, 1)
    wnorm = np.clip((wind - p2) / max(p98 - p2, 1e-6), 0, 1)
    risk = wnorm * np.power(np.clip(built, 0, 1), W_EXP)
    risk[~land] = np.nan
    risk[built <= 0] = np.nan                                        # no exposure -> no risk shown
    return np.ma.masked_invalid(risk)

# ----------------------------------------------------------------------
def add_scalebar(ax, e, km=50, x0=0.055, y0=0.06):
    lon0, lon1, lat0, lat1 = e
    deg = km / (111.320 * np.cos(np.deg2rad(0.5*(lat0+lat1))))
    x = lon0+x0*(lon1-lon0); y = lat0+y0*(lat1-lat0); seg = deg/2
    for i in range(2):
        ax.add_patch(plt.Rectangle((x+i*seg, y), seg, (lat1-lat0)*0.012,
                     facecolor="black" if i == 0 else "white", edgecolor="black", lw=0.8, zorder=25))
    ax.text(x, y+(lat1-lat0)*0.03, "0", ha="center", va="bottom", fontsize=8, zorder=25)
    ax.text(x+deg, y+(lat1-lat0)*0.03, f"{km} km", ha="center", va="bottom", fontsize=8, zorder=25)

def add_north(ax, e, x0=0.955, y0=0.90):
    lon0, lon1, lat0, lat1 = e
    x = lon0+x0*(lon1-lon0); y = lat0+y0*(lat1-lat0); dy = (lat1-lat0)*0.06
    ax.annotate("", xy=(x, y+dy), xytext=(x, y),
                arrowprops=dict(arrowstyle="-|>", lw=1.6, color="black"), zorder=25)
    ax.text(x, y+dy*1.15, "N", ha="center", va="bottom", fontsize=11, fontweight="bold", zorder=25)

def render(risk, prov, out_path):
    lon0, lon1, lat0, lat1 = EXTENT
    fig, ax = plt.subplots(figsize=(13, 8)); ax.set_facecolor("#cfe0ec")
    ax.set_xlim(lon0, lon1); ax.set_ylim(lat0, lat1); ax.set_aspect(geo_aspect(EXTENT))
    if prov is not None and len(prov):
        prov.plot(ax=ax, facecolor="#eef1ec", edgecolor="#9aa0a6", linewidth=0.6, zorder=2)
    vmax = np.nanpercentile(risk.compressed(), 98) if risk.count() else 1.0
    im = ax.imshow(risk, extent=EXTENT, origin="upper", cmap=CMAP,
                   vmin=0, vmax=max(vmax, 1e-3), zorder=3)
    if prov is not None and len(prov):
        prov.boundary.plot(ax=ax, color="#333", linewidth=0.6, zorder=5)
    for nm, lx, ly in CITIES:
        if lon0 < lx < lon1 and lat0 < ly < lat1:
            ax.scatter([lx], [ly], s=12, color="black", zorder=10)
            ax.annotate(nm, xy=(lx, ly), xytext=(3, 3), textcoords="offset points",
                        fontsize=8, fontweight="bold", zorder=11,
                        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.7))
    for nm, (lx, ly) in {"MARMARA SEA": (28.05, 40.55), "Black Sea": (28.6, 41.78),
                         "Aegean Sea": (26.15, 39.55)}.items():
        if lon0 < lx < lon1 and lat0 < ly < lat1:
            ax.text(lx, ly, nm, fontsize=9.5, style="italic", color="#14406e",
                    ha="center", va="center", zorder=9)
    ax.set_xticks(np.arange(np.ceil(lon0), lon1+0.01, 1.0))
    ax.set_yticks(np.arange(np.ceil(lat0), lat1+0.01, 1.0))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}°E"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}°N"))
    ax.grid(True, color="0.5", lw=0.4, ls=":", alpha=0.45, zorder=6)
    ax.tick_params(labelsize=9)
    cb = fig.colorbar(ScalarMappable(norm=mcolors.Normalize(0, max(vmax, 1e-3)), cmap=CMAP),
                      ax=ax, fraction=0.030, pad=0.02)
    cb.set_label("Wind risk index (relative)", fontsize=9)
    add_scalebar(ax, EXTENT); add_north(ax, EXTENT)
    ax.set_title("Wind risk of the Marmara basin (hazard × urban exposure)",
                 fontsize=13, fontweight="bold", pad=8)
    ax.text(0.5, -0.06, "Relative index: normalised GWA 100 m wind speed × CLC+ built-up fraction; "
            "not a calibrated loss model.", transform=ax.transAxes, ha="center", va="top",
            fontsize=7.3, color="#555")
    fig.tight_layout(); fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    print("written:", out_path)

def main():
    for f, nm in ((WIND_TIF, "WIND_TIF (Global Wind Atlas)"), (CLC_TIF, "CLC_TIF (CORINE CLC+)")):
        if not os.path.exists(f):
            sys.exit(f"Missing {nm}: {f}")
    lon0, lon1, lat0, lat1 = EXTENT
    prov = gpd.read_file(PROV_JSON).to_crs(4326).cx[lon0:lon1, lat0:lat1] if os.path.exists(PROV_JSON) else None
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUT_NAME)
    render(compute_risk(), prov, out)

if __name__ == "__main__":
    main()
