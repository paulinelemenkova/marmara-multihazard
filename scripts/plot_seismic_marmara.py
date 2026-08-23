#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 10 (seismic hazard / seismotectonics) of the Marmara basin — clean replot.

Layers (no icons / infographic panels):
  * Earthquake epicentres  -> your catalogue CSV (IEB_Marmara.csv): sized by
    magnitude, coloured by depth.
  * Active faults          -> Turkey 500k Faults V1.TAB (MapInfo).
  * Hillshaded relief/sea  -> GEBCO_Marmara GeoTIFF (optional base).
  * Province outlines      -> geoBoundaries ADM1.
  * Documented large events marked with stars (magnitudes from the literature).

First run  --inspect  if the catalogue columns aren't auto-detected:
    python3 plot_seismic_marmara.py --inspect
then set LON/LAT/MAG/DEPTH _FIELD below if needed and run:
    python3 plot_seismic_marmara.py            # -> Figure_11.png (300 dpi)

Requires: pandas, geopandas, rasterio, numpy, matplotlib
"""

import os, sys, argparse

# --- PROJ/GDAL guard (avoids GMT's stale proj.db) ---------------------------
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
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.colors import LightSource
import geopandas as gpd

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
DATA_ROOT = os.environ.get(
    "DATA_ROOT",
    os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")),
)
TURKEY      = os.path.join(DATA_ROOT, "TURKEY")
CATALOG_CSV = os.path.join(TURKEY, "IEB_Marmara.csv")
FAULT_TAB   = os.path.join(TURKEY, "Geological map data of Turkey", "Turkey 500k Faults V1.TAB")
RELIEF_TIF  = os.path.join(TURKEY, "GEBCO_Marmara", "gebco_2026_n42.0_s39.0_w25.0_e31.0_geotiff.tif")
PROV_JSON   = os.path.join(DATA_ROOT, "geoBoundaries-TUR-ADM1.geojson")

EXTENT   = (25.9, 31.1, 39.2, 41.9)
MIN_MAG  = 2.5                 # drop micro-seismicity below this
DEPTH_CMAP = "plasma"          # perceptually-uniform, colour-blind-safe
OUT_NAME = "Figure_11.png"

# set these only if --inspect shows auto-detection picked the wrong column
LON_FIELD = None; LAT_FIELD = None; MAG_FIELD = None; DEPTH_FIELD = None

# documented large events (label, lon, lat, Mw) — verified magnitudes; edit freely
HISTORICAL = [
    ("1999 İzmit, M$_w$7.4",  29.86, 40.75, 7.4),
    ("1999 Düzce, M$_w$7.2",  31.16, 40.77, 7.2),
    ("1912 Mürefte, M$_s$7.4", 27.20, 40.75, 7.4),
]

# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def detect(cols, cands):
    low = {c.lower(): c for c in cols}
    for want in cands:
        for lc, orig in low.items():
            if want in lc:
                return orig
    return None

def geo_aspect(e): return 1.0 / np.cos(np.deg2rad(0.5 * (e[2] + e[3])))

def hillshade(elev, az=315, alt=45):
    return LightSource(azdeg=az, altdeg=alt).hillshade(np.nan_to_num(elev, nan=0.0),
                                                       vert_exag=0.00035, dx=1, dy=1)

def mag_size(m):                       # marker area from magnitude
    return 0.7 * np.power(m, 3.2)

def add_scalebar(ax, e, km=50, x0=0.055, y0=0.06):
    lon0, lon1, lat0, lat1 = e
    deg = km / (111.320 * np.cos(np.deg2rad(0.5 * (lat0 + lat1))))
    x = lon0 + x0*(lon1-lon0); y = lat0 + y0*(lat1-lat0); seg = deg/2
    for i in range(2):
        ax.add_patch(plt.Rectangle((x+i*seg, y), seg, (lat1-lat0)*0.012,
                     facecolor="black" if i == 0 else "white", edgecolor="black", lw=0.8, zorder=30))
    ax.text(x, y+(lat1-lat0)*0.03, "0", ha="center", va="bottom", fontsize=8, zorder=30)
    ax.text(x+deg, y+(lat1-lat0)*0.03, f"{km} km", ha="center", va="bottom", fontsize=8, zorder=30)

def add_north(ax, e, x0=0.955, y0=0.90):
    lon0, lon1, lat0, lat1 = e
    x = lon0+x0*(lon1-lon0); y = lat0+y0*(lat1-lat0); dy = (lat1-lat0)*0.06
    ax.annotate("", xy=(x, y+dy), xytext=(x, y),
                arrowprops=dict(arrowstyle="-|>", lw=1.6, color="black"), zorder=30)
    ax.text(x, y+dy*1.15, "N", ha="center", va="bottom", fontsize=11, fontweight="bold", zorder=30)

# ----------------------------------------------------------------------
# data loading
# ----------------------------------------------------------------------
def load_catalog():
    df = pd.read_csv(CATALOG_CSV, sep=None, engine="python")
    lon = LON_FIELD or detect(df.columns, ["longitude", "lon", "boylam", "x"])
    lat = LAT_FIELD or detect(df.columns, ["latitude", "lat", "enlem", "y"])
    mag = MAG_FIELD or detect(df.columns, ["mag", "mw", "ml", "büyük", "xm", "m"])
    dep = DEPTH_FIELD or detect(df.columns, ["depth", "derin", "dep", "z"])
    if not (lon and lat and mag):
        print("Columns found:", list(df.columns))
        sys.exit("Could not auto-detect lon/lat/mag — set LON_FIELD/LAT_FIELD/MAG_FIELD at top.")
    print(f"  catalogue fields -> lon:{lon}  lat:{lat}  mag:{mag}  depth:{dep}")
    out = pd.DataFrame({"lon": pd.to_numeric(df[lon], errors="coerce"),
                        "lat": pd.to_numeric(df[lat], errors="coerce"),
                        "mag": pd.to_numeric(df[mag], errors="coerce")})
    out["depth"] = pd.to_numeric(df[dep], errors="coerce") if dep else np.nan
    lon0, lon1, lat0, lat1 = EXTENT
    out = out.dropna(subset=["lon", "lat", "mag"])
    out = out[(out.lon.between(lon0, lon1)) & (out.lat.between(lat0, lat1)) & (out.mag >= MIN_MAG)]
    print(f"  {len(out)} events in extent with M>={MIN_MAG}")
    return out.sort_values("mag")           # small first so big plot on top

def load_optional():
    lon0, lon1, lat0, lat1 = EXTENT
    faults = relief = r_ext = prov = None
    if os.path.exists(FAULT_TAB):
        try:
            f = gpd.read_file(FAULT_TAB)
            if f.crs is None: f.set_crs(4326, inplace=True)
            faults = f.to_crs(4326).cx[lon0:lon1, lat0:lat1]
        except Exception as ex:
            print("faults skipped:", ex)
    if os.path.exists(PROV_JSON):
        prov = gpd.read_file(PROV_JSON).to_crs(4326).cx[lon0:lon1, lat0:lat1]
    if os.path.exists(RELIEF_TIF):
        import rasterio
        from rasterio.windows import from_bounds
        with rasterio.open(RELIEF_TIF) as ds:
            try:
                win = from_bounds(lon0, lat0, lon1, lat1, ds.transform)
                a = ds.read(1, window=win); t = ds.window_transform(win)
                h, w = a.shape; r_ext = (t.c, t.c+w*t.a, t.f+h*t.e, t.f)
            except Exception:
                a = ds.read(1); b = ds.bounds; r_ext = (b.left, b.right, b.bottom, b.top)
            relief = a.astype("float32"); relief[relief < -12000] = np.nan
    return faults, relief, r_ext, prov

# ----------------------------------------------------------------------
# renderer
# ----------------------------------------------------------------------
def render(cat, faults, relief, r_ext, prov, out_path):
    lon0, lon1, lat0, lat1 = EXTENT
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.set_xlim(lon0, lon1); ax.set_ylim(lat0, lat1); ax.set_aspect(geo_aspect(EXTENT))

    if relief is not None:
        ax.imshow(hillshade(relief), extent=r_ext, origin="upper", cmap="gray",
                  vmin=0, vmax=1, alpha=0.55, zorder=1)
        sea = np.ma.masked_where(relief >= 0, relief)
        ax.imshow(sea, extent=r_ext, origin="upper", cmap="Blues_r", alpha=0.9, zorder=2)
    else:
        ax.set_facecolor("#dbe7f0")

    if prov is not None and len(prov):
        prov.boundary.plot(ax=ax, color="#555", linewidth=0.5, zorder=3)
    if faults is not None and len(faults):
        faults.plot(ax=ax, color="#b2182b", linewidth=1.3, zorder=6)

    # epicentres: size ~ magnitude, colour ~ depth (unless depth is a placeholder)
    d = cat["depth"]
    depth_ok = (d.notna().sum() > 0 and d.dropna().nunique() > 3
                and float(d.std(skipna=True) or 0) > 1.0)
    if depth_ok:
        norm = mcolors.Normalize(vmin=0, vmax=max(np.nanpercentile(d, 95), 10))
        ax.scatter(cat["lon"], cat["lat"], s=mag_size(cat["mag"]), c=d, cmap=DEPTH_CMAP,
                   norm=norm, alpha=0.8, edgecolor="k", linewidth=0.25, zorder=8)
    else:
        print("  note: focal depth is constant/placeholder — using a uniform colour instead.")
        ax.scatter(cat["lon"], cat["lat"], s=mag_size(cat["mag"]), facecolor="#c0392b",
                   alpha=0.6, edgecolor="k", linewidth=0.25, zorder=8)

    # documented large events as stars
    for name, lx, ly, mw in HISTORICAL:
        if lon0 < lx < lon1 and lat0 < ly < lat1:
            ax.scatter([lx], [ly], marker="*", s=430, facecolor="#ffd400",
                       edgecolor="black", linewidth=0.9, zorder=10)
            ax.annotate(name, xy=(lx, ly), xytext=(4, 6), textcoords="offset points",
                        fontsize=8, fontweight="bold", zorder=11,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))

    # sea labels
    for nm, (lx, ly) in {"MARMARA SEA": (28.0, 40.6), "Black Sea": (28.6, 41.78),
                         "Aegean Sea": (26.15, 39.55)}.items():
        if lon0 < lx < lon1 and lat0 < ly < lat1:
            ax.text(lx, ly, nm, fontsize=9.5, style="italic", color="#14406e",
                    ha="center", va="center", zorder=9)
    if lon0 < 28.4 < lon1:
        ax.text(28.4, 40.78, "North Anatolian Fault Zone", fontsize=8.5, color="#7b241c",
                style="italic", ha="center", va="center", zorder=9)

    # graticule
    ax.set_xticks(np.arange(np.ceil(lon0), lon1+0.01, 1.0))
    ax.set_yticks(np.arange(np.ceil(lat0), lat1+0.01, 1.0))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}°E"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}°N"))
    ax.grid(True, color="0.5", lw=0.4, ls=":", alpha=0.5, zorder=5)
    ax.tick_params(labelsize=9)

    # colourbar (depth) + magnitude size legend + fault/event legend
    if depth_ok:
        cb = fig.colorbar(ScalarMappable(norm=norm, cmap=DEPTH_CMAP), ax=ax,
                          fraction=0.030, pad=0.02)
        cb.set_label("Focal depth (km)", fontsize=9)
    mag_handles = [Line2D([0], [0], marker="o", linestyle="", markerfacecolor="#888",
                   markeredgecolor="k", markersize=np.sqrt(mag_size(m))/1.6, label=f"M {m:.0f}")
                   for m in (3, 4, 5, 6, 7)]
    leg1 = ax.legend(handles=mag_handles, loc="upper left", fontsize=8, title="Magnitude",
                     title_fontsize=9, framealpha=0.92, labelspacing=1.5, borderpad=0.9,
                     handletextpad=1.3)
    leg1.set_zorder(31); ax.add_artist(leg1)
    extra = [Line2D([0], [0], color="#b2182b", lw=1.6, label="Active fault"),
             Line2D([0], [0], marker="*", linestyle="", markerfacecolor="#ffd400",
                    markeredgecolor="k", markersize=13, label="Large historical event")]
    leg2 = ax.legend(handles=extra, loc="lower right", fontsize=8, framealpha=0.92)
    leg2.set_zorder(31)

    add_scalebar(ax, EXTENT); add_north(ax, EXTENT)
    ax.set_title("Seismicity and active tectonics of the Marmara basin",
                 fontsize=13, fontweight="bold", pad=8)
    ax.text(0.5, -0.06, "Epicentres from the earthquake catalogue; faults after the Türkiye 1:500,000 map.",
            transform=ax.transAxes, ha="center", va="top", fontsize=7.5, color="#555")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    print("written:", out_path)

# ----------------------------------------------------------------------
def inspect():
    df = pd.read_csv(CATALOG_CSV, sep=None, engine="python", nrows=5)
    print("columns:", list(df.columns)); print(df.head())

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--inspect", action="store_true")
    if ap.parse_args().inspect:
        inspect(); return
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUT_NAME)
    render(load_catalog(), *load_optional(), out)

if __name__ == "__main__":
    main()
