#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 5 (land cover) of the Marmara basin — clean replot.

Source: Copernicus CLMS CLC+ Backbone raster, 2018, 10 m, EPSG:3035
  $DATA_ROOT/CORINE_Land_Cover_Types/
      CLMS_CLCplus_RASTER_2018_010m_eu_03035_V1_1/Data/
      CLMS_CLCplus_RASTER_2018_010m_eu_03035_V1_1.tif

The full product is pan-European at 10 m, so this script:
  1) reprojects the Marmara bounding box (WGS84) into EPSG:3035,
  2) reads ONLY that window, decimated to ~TARGET_W px (nearest, categorical),
  3) reprojects the window to WGS84 for a lon/lat map,
  4) draws it with the CLC+ legend (full 11 classes) or a grouped scheme.

Run:
    python3 plot_landcover_marmara.py          # -> Figure_05.png (300 dpi)

Requires: rasterio, numpy, matplotlib   (geopandas optional, for provinces)
          pip install rasterio numpy matplotlib geopandas
"""

import os
import sys

# ---------------------------------------------------------------------------
# PROJ/GDAL database fix (must run BEFORE importing rasterio)
# GMT.app exports an OLD proj.db via PROJ_LIB/PROJ_DATA (layout MINOR=2), which
# breaks rasterio's EPSG lookups. Remove that override and let rasterio use its
# OWN (conda) PROJ database — never point it at a different library's proj.db,
# or you get a layout-version mismatch.
# ---------------------------------------------------------------------------
for _v in ("PROJ_LIB", "PROJ_DATA", "GDAL_DATA"):
    if "gmt" in os.environ.get(_v, "").lower():
        os.environ.pop(_v, None)
_conda_proj = os.path.join(sys.prefix, "share", "proj")     # e.g. anaconda3/share/proj
if os.path.exists(os.path.join(_conda_proj, "proj.db")):
    os.environ["PROJ_DATA"] = _conda_proj
    os.environ["PROJ_LIB"] = _conda_proj
_conda_gdal = os.path.join(sys.prefix, "share", "gdal")
if os.path.isdir(_conda_gdal):
    os.environ["GDAL_DATA"] = _conda_gdal
# ---------------------------------------------------------------------------

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import LightSource  # (kept for optional relief; unused by default)

import rasterio
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds, reproject
from rasterio.transform import from_bounds as transform_from_bounds
from affine import Affine

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
DATA_ROOT = os.environ.get(
    "DATA_ROOT",
    os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")),
)
CLC_TIF = os.path.join(
    DATA_ROOT, "CORINE_Land_Cover_Types",
    "CLMS_CLCplus_RASTER_2018_010m_eu_03035_V1_1", "Data",
    "CLMS_CLCplus_RASTER_2018_010m_eu_03035_V1_1.tif")

PROV_JSON = os.path.join(DATA_ROOT, "geoBoundaries-TUR-ADM1.geojson")  # optional overlay

EXTENT   = (26.0, 31.0, 39.2, 41.8)   # lon_min, lon_max, lat_min, lat_max (WGS84)
TARGET_W = 2600                        # decimated width in pixels (raise for more detail)
MODE     = "full"                      # "full" (11 CLC+ classes) or "grouped"
SHOW_PROVINCES = True
OUT_NAME = "Figure_05.png"

SEA_CODE, NODATA = 0, 255              # CLC+: 0 = outside area (used here as sea)

# ----------------------------------------------------------------------
# CLC+ Backbone class scheme (codes 1-11) + fallback colours
# ----------------------------------------------------------------------
CODE_NAME = {
    1: "Sealed (built-up)",
    2: "Woody – needle-leaved trees",
    3: "Woody – broadleaved deciduous",
    4: "Woody – broadleaved evergreen",
    5: "Low-growing woody plants",
    6: "Permanent herbaceous",
    7: "Periodically herbaceous (cropland)",
    8: "Lichens and mosses",
    9: "Non- and sparsely vegetated",
    10: "Water",
    11: "Snow and ice",
}
CODE_COLOR = {                      # used if the GeoTIFF has no embedded colour table
    1:  "#E31A1C", 2:  "#1B7837", 3:  "#4DAC26", 4:  "#2E8B57",
    5:  "#A6D96A", 6:  "#D9EF8B", 7:  "#FEE08B", 8:  "#C7B9A5",
    9:  "#B0B0B0", 10: "#2C7FB8", 11: "#FFFFFF",
}
SEA_COLOR = "#ABD3E6"

# grouped scheme (paper-style broad categories)
GROUP_DEF = [
    ("Artificial / sealed surfaces", [1],           "#E31A1C"),
    ("Forest & woody vegetation",    [2, 3, 4, 5],   "#2E8B57"),
    ("Herbaceous / grassland",       [6, 8],         "#C7E9A0"),
    ("Cropland (agricultural)",      [7],            "#FEE08B"),
    ("Bare / sparsely vegetated",    [9, 11],        "#BEBEBE"),
    ("Water",                        [10],           "#2C7FB8"),
]

# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def hex2rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))

def geo_aspect(ext):
    return 1.0 / np.cos(np.deg2rad(0.5 * (ext[2] + ext[3])))

def add_scalebar(ax, ext, km=50, x0=0.06, y0=0.06):
    lon0, lon1, lat0, lat1 = ext
    deg = km / (111.320 * np.cos(np.deg2rad(0.5 * (lat0 + lat1))))
    x = lon0 + x0 * (lon1 - lon0); y = lat0 + y0 * (lat1 - lat0); seg = deg / 2
    for i in range(2):
        ax.add_patch(plt.Rectangle((x + i * seg, y), seg, (lat1 - lat0) * 0.012,
                     facecolor="black" if i == 0 else "white", edgecolor="black",
                     lw=0.8, zorder=20))
    ax.text(x, y + (lat1 - lat0) * 0.03, "0", ha="center", va="bottom", fontsize=8, zorder=20)
    ax.text(x + deg, y + (lat1 - lat0) * 0.03, f"{km} km", ha="center", va="bottom",
            fontsize=8, zorder=20)

def add_north(ax, ext, x0=0.95, y0=0.90):
    lon0, lon1, lat0, lat1 = ext
    x = lon0 + x0 * (lon1 - lon0); y = lat0 + y0 * (lat1 - lat0); dy = (lat1 - lat0) * 0.06
    ax.annotate("", xy=(x, y + dy), xytext=(x, y),
                arrowprops=dict(arrowstyle="-|>", lw=1.6, color="black"), zorder=20)
    ax.text(x, y + dy * 1.15, "N", ha="center", va="bottom", fontsize=11,
            fontweight="bold", zorder=20)

# ----------------------------------------------------------------------
# read + reproject the CLC+ window
# ----------------------------------------------------------------------
def read_clc():
    lon0, lon1, lat0, lat1 = EXTENT
    with rasterio.open(CLC_TIF) as ds:
        # window in the raster CRS (EPSG:3035)
        l, b, r, t = transform_bounds("EPSG:4326", ds.crs, lon0, lat0, lon1, lat1,
                                      densify_pts=21)
        win = from_bounds(l, b, r, t, ds.transform).round_offsets().round_lengths()
        win_w, win_h = int(win.width), int(win.height)
        scale = min(TARGET_W / win_w, 1.0)
        out_w, out_h = max(int(win_w * scale), 1), max(int(win_h * scale), 1)
        arr = ds.read(1, window=win, out_shape=(out_h, out_w),
                      resampling=Resampling.nearest, boundless=True, fill_value=NODATA)
        src_tr = ds.window_transform(win) * Affine.scale(win_w / out_w, win_h / out_h)
        src_crs = ds.crs
        # try embedded colour table
        try:
            cmap = ds.colormap(1)          # {code: (r,g,b,a)}
        except Exception:
            cmap = None

    # reproject the small window to WGS84 over EXTENT
    dst_w, dst_h = out_w, out_h
    dst = np.full((dst_h, dst_w), NODATA, dtype=arr.dtype)
    dst_tr = transform_from_bounds(lon0, lat0, lon1, lat1, dst_w, dst_h)
    reproject(source=arr, destination=dst,
              src_transform=src_tr, src_crs=src_crs,
              dst_transform=dst_tr, dst_crs="EPSG:4326",
              resampling=Resampling.nearest, src_nodata=NODATA, dst_nodata=NODATA)
    return dst, cmap

# ----------------------------------------------------------------------
# colour mapping -> RGBA image
# ----------------------------------------------------------------------
def build_rgba(codes, cmap):
    LUT = np.zeros((256, 4), dtype=np.uint8)      # default transparent
    if MODE == "grouped":
        for _, members, col in GROUP_DEF:
            r, g, b = hex2rgb(col)
            for c in members:
                LUT[c] = (r, g, b, 255)
    else:
        for c, name in CODE_NAME.items():
            if cmap and c in cmap and cmap[c][:3] != (0, 0, 0):
                r, g, b = cmap[c][:3]
            else:
                r, g, b = hex2rgb(CODE_COLOR[c])
            LUT[c] = (r, g, b, 255)
    sr, sg, sb = hex2rgb(SEA_COLOR)
    LUT[SEA_CODE] = (sr, sg, sb, 255)             # outside area -> sea
    LUT[NODATA] = (0, 0, 0, 0)                     # nodata transparent
    return LUT[codes]

def legend_handles(codes):
    present = set(np.unique(codes).tolist())
    handles = []
    if MODE == "grouped":
        for name, members, col in GROUP_DEF:
            if present & set(members):
                handles.append(Patch(facecolor=col, edgecolor="#444", lw=0.4, label=name))
    else:
        for c, name in CODE_NAME.items():
            if c in present:
                col = CODE_COLOR[c]
                handles.append(Patch(facecolor=col, edgecolor="#444", lw=0.4, label=name))
    handles.append(Patch(facecolor=SEA_COLOR, edgecolor="#444", lw=0.4, label="Sea"))
    return handles

# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main():
    codes, cmap = read_clc()
    rgba = build_rgba(codes, cmap)
    lon0, lon1, lat0, lat1 = EXTENT

    fig, ax = plt.subplots(figsize=(13, 8))
    ax.set_xlim(lon0, lon1); ax.set_ylim(lat0, lat1)
    ax.set_aspect(geo_aspect(EXTENT))
    ax.imshow(rgba, extent=EXTENT, origin="upper", interpolation="nearest", zorder=2)

    # provinces (optional)
    if SHOW_PROVINCES and os.path.exists(PROV_JSON):
        try:
            import geopandas as gpd
            prov = gpd.read_file(PROV_JSON).to_crs(4326).cx[lon0:lon1, lat0:lat1]
            prov.boundary.plot(ax=ax, color="#1a1a1a", linewidth=0.7, zorder=6)
            for _, r in prov.iterrows():
                c = r.geometry.representative_point()
                nm = r.get("shapeName", None)
                if nm and lon0 < c.x < lon1 and lat0 < c.y < lat1:
                    ax.text(c.x, c.y, str(nm).upper(), fontsize=8, ha="center",
                            va="center", color="#111", zorder=9,
                            bbox=dict(boxstyle="round,pad=0.12", fc="white",
                                      ec="none", alpha=0.5))
        except Exception as e:
            print("provinces skipped:", e)

    # sea labels
    for name, (lx, ly) in {"MARMARA SEA": (28.2, 40.62), "Black Sea": (29.2, 41.6),
                            "Aegean Sea": (26.2, 39.5)}.items():
        if lon0 < lx < lon1 and lat0 < ly < lat1:
            ax.text(lx, ly, name, fontsize=10, style="italic", color="#14406e",
                    ha="center", va="center", zorder=9)

    # graticule
    ax.set_xticks(np.arange(np.ceil(lon0), lon1 + 0.01, 1.0))
    ax.set_yticks(np.arange(np.ceil(lat0), lat1 + 0.01, 1.0))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}°E"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}°N"))
    ax.grid(True, color="0.5", lw=0.4, ls=":", alpha=0.5, zorder=5)
    ax.tick_params(labelsize=9)

    leg = ax.legend(handles=legend_handles(codes), loc="upper left", fontsize=8.0,
                    title="Land cover (CLC+ Backbone 2018, 10 m)", title_fontsize=9,
                    framealpha=0.93, borderpad=0.7)
    leg.set_zorder(21)

    add_scalebar(ax, EXTENT); add_north(ax, EXTENT)
    ax.set_title("Land cover of the Marmara basin (Copernicus CLC+ Backbone, 2018)",
                 fontsize=13, fontweight="bold", pad=8)

    outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUT_NAME)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight", facecolor="white")
    print("written:", outpath)

if __name__ == "__main__":
    main()
