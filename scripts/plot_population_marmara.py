#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 5 (population) of the Marmara basin — clean replot.

Province-level population-density choropleth (persons km-2) with graduated
proportional symbols for total population. No icons / infographic panels.

Data:
  * Province polygons : geoBoundaries-TUR-ADM1.geojson  (ADM1 = provinces)
  * Population values : your TÜİK CSV  (columns: province,population)
                        -> set POP_CSV.  If absent, a flagged APPROXIMATE
                           2022 fallback table is used so the script still runs.

Optional raster mode (a true gridded density surface, if you have GHS-POP or
WorldPop as a population-COUNT GeoTIFF): set POP_RASTER and MODE="raster".

Run:
    python3 plot_population_marmara.py        # -> Figure_06.png (300 dpi)

Requires: geopandas, numpy, matplotlib   (rasterio only for MODE="raster")
          pip install geopandas numpy matplotlib rasterio
"""

import os, sys, csv, unicodedata

# --- PROJ/GDAL guard (harmless if not needed; avoids GMT's stale proj.db) ----
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
from matplotlib.lines import Line2D
import geopandas as gpd

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
DATA_ROOT = os.environ.get(
    "DATA_ROOT",
    os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")),
)
PROV_JSON = os.path.join(DATA_ROOT, "geoBoundaries-TUR-ADM1.geojson")
POP_CSV   = ""                       # e.g. os.path.join(DATA_ROOT,"TURKEY","tuik_pop_2022.csv")
POP_RASTER= ""                       # e.g. GHS-POP / WorldPop count GeoTIFF
RELIEF_TIF= ""                       # optional GEBCO tif for a sea backdrop
EXTENT    = (25.9, 31.1, 39.2, 41.9) # lon_min, lon_max, lat_min, lat_max
MODE      = "choropleth"             # "choropleth" or "raster"
CMAP      = "YlOrRd"                 # sequential, colour-blind-safe
OUT_NAME  = "Figure_06.png"

# ----------------------------------------------------------------------
# APPROXIMATE TÜİK-2022 populations (fallback only) -- VERIFY / REPLACE
# with your own CSV before publication. Values ~2022 ADNKS.
# ----------------------------------------------------------------------
POP_FALLBACK = {
    "Istanbul": 15907951, "Bursa": 3194720, "Kocaeli": 2079072,
    "Balikesir": 1257590, "Tekirdag": 1142451, "Sakarya": 1080080,
    "Eskisehir": 906617,  "Canakkale": 561411, "Edirne": 411528,
    "Duzce": 404486,      "Kirklareli": 369347, "Bolu": 320824,
    "Yalova": 296333,     "Bilecik": 228673,
}

# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def tr_norm(s):
    """Normalise a Turkish province name for matching."""
    if s is None:
        return ""
    s = str(s)
    for a, b in (("İ", "i"), ("I", "i"), ("ı", "i"), ("Ş", "s"), ("ş", "s"),
                 ("Ğ", "g"), ("ğ", "g"), ("Ç", "c"), ("ç", "c"),
                 ("Ö", "o"), ("ö", "o"), ("Ü", "u"), ("ü", "u")):
        s = s.replace(a, b)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return s.strip().lower()

POP_BY_NORM = {tr_norm(k): v for k, v in POP_FALLBACK.items()}
TARGET = set(POP_BY_NORM.keys())     # the 14 Marmara provinces

def load_population_csv(path):
    out = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            keys = {k.lower(): k for k in row}
            pk = next((keys[k] for k in ("province", "il", "name", "adm1") if k in keys), None)
            vk = next((keys[k] for k in ("population", "pop", "nufus", "value") if k in keys), None)
            if pk and vk and row[vk]:
                out[tr_norm(row[pk])] = int(float(str(row[vk]).replace(",", "").replace(" ", "")))
    return out

def geo_aspect(ext):
    return 1.0 / np.cos(np.deg2rad(0.5 * (ext[2] + ext[3])))

def add_scalebar(ax, ext, km=50, x0=0.055, y0=0.09):
    lon0, lon1, lat0, lat1 = ext
    deg = km / (111.320 * np.cos(np.deg2rad(0.5 * (lat0 + lat1))))
    x = lon0 + x0 * (lon1 - lon0); y = lat0 + y0 * (lat1 - lat0); seg = deg / 2
    for i in range(2):
        ax.add_patch(plt.Rectangle((x + i * seg, y), seg, (lat1 - lat0) * 0.012,
                     facecolor="black" if i == 0 else "white", edgecolor="black",
                     lw=0.8, zorder=25))
    ax.text(x, y + (lat1 - lat0) * 0.03, "0", ha="center", va="bottom", fontsize=8, zorder=25)
    ax.text(x + deg, y + (lat1 - lat0) * 0.03, f"{km} km", ha="center", va="bottom",
            fontsize=8, zorder=25)

def add_north(ax, ext, x0=0.955, y0=0.90):
    lon0, lon1, lat0, lat1 = ext
    x = lon0 + x0 * (lon1 - lon0); y = lat0 + y0 * (lat1 - lat0); dy = (lat1 - lat0) * 0.06
    ax.annotate("", xy=(x, y + dy), xytext=(x, y),
                arrowprops=dict(arrowstyle="-|>", lw=1.6, color="black"), zorder=25)
    ax.text(x, y + dy * 1.15, "N", ha="center", va="bottom", fontsize=11,
            fontweight="bold", zorder=25)

def sea_and_frame(ax, ext):
    ax.set_facecolor("#dbe7f0")                       # sea backdrop
    for name, (lx, ly) in {"MARMARA SEA": (28.0, 40.62), "Black Sea": (28.6, 41.75),
                            "Aegean Sea": (26.35, 39.80)}.items():
        if ext[0] < lx < ext[1] and ext[2] < ly < ext[3]:
            ax.text(lx, ly, name, fontsize=9.5, style="italic", color="#14406e",
                    ha="center", va="center", zorder=9)

def graticule(ax, ext):
    ax.set_xticks(np.arange(np.ceil(ext[0]), ext[1] + 0.01, 1.0))
    ax.set_yticks(np.arange(np.ceil(ext[2]), ext[3] + 0.01, 1.0))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}°E"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}°N"))
    ax.grid(True, color="0.5", lw=0.4, ls=":", alpha=0.5, zorder=5)
    ax.tick_params(labelsize=9)

# ----------------------------------------------------------------------
# choropleth
# ----------------------------------------------------------------------
def render_choropleth(prov, out_path):
    lon0, lon1, lat0, lat1 = EXTENT
    # density in an equal-area CRS
    prov_ea = prov.to_crs(3035)
    prov = prov.copy()
    prov["area_km2"] = prov_ea.area.values / 1e6
    prov["dens"] = prov["pop"].values / prov["area_km2"].values

    vmin = max(prov["dens"].min(), 1.0); vmax = prov["dens"].max()
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(CMAP)

    fig, ax = plt.subplots(figsize=(13, 8))
    ax.set_xlim(lon0, lon1); ax.set_ylim(lat0, lat1); ax.set_aspect(geo_aspect(EXTENT))
    sea_and_frame(ax, EXTENT)

    prov.plot(ax=ax, color=[cmap(norm(d)) for d in prov["dens"]],
              edgecolor="#3a3a3a", linewidth=0.6, zorder=3)

    # graduated proportional symbols (total population)
    cent = prov.geometry.representative_point()
    pmax = prov["pop"].max()
    smax = 1200.0                                       # max marker area (pt^2)
    sizes = smax * (prov["pop"].values / pmax)
    ax.scatter(cent.x, cent.y, s=sizes, facecolor="#2c3e50", alpha=0.55,
               edgecolor="white", linewidth=0.6, zorder=8)

    # province labels — placed just BELOW each symbol (offset by its radius in
    # points, so text never sits on the circle regardless of figure scaling)
    import math
    for i, ((_, r), c) in enumerate(zip(prov.iterrows(), cent)):
        if lon0 < c.x < lon1 and lat0 < c.y < lat1:
            rad_pt = math.sqrt(max(sizes[i], 1.0) / math.pi)
            ax.annotate(r["name"], xy=(c.x, c.y), xytext=(0, -(rad_pt + 4.5)),
                        textcoords="offset points", ha="center", va="top",
                        fontsize=8, color="#111", zorder=10,
                        bbox=dict(boxstyle="round,pad=0.12", fc="white",
                                  ec="none", alpha=0.75))

    graticule(ax, EXTENT); add_scalebar(ax, EXTENT); add_north(ax, EXTENT)

    # density colourbar
    sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.030, pad=0.02)
    cb.set_label("Population density (persons km$^{-2}$, log scale)", fontsize=9)

    # proportional-symbol size legend
    refs = [r for r in (500000, 3000000, 15000000) if r <= pmax] or [pmax]
    handles = [Line2D([0], [0], marker="o", linestyle="", markerfacecolor="#2c3e50",
                      markeredgecolor="white", alpha=0.55,
                      markersize=np.sqrt(smax * (r / pmax)) / 1.4,
                      label=f"{r/1e6:.1f} M") for r in refs]
    leg = ax.legend(handles=handles, loc="upper left", fontsize=8,
                    title="Total population", title_fontsize=9,
                    framealpha=0.92, labelspacing=1.6, borderpad=0.9, handletextpad=1.4)
    leg.set_zorder(26)

    ax.set_title("Population distribution of the Marmara basin (province level)",
                 fontsize=13, fontweight="bold", pad=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    print("written:", out_path)

# ----------------------------------------------------------------------
# raster density mode (optional)
# ----------------------------------------------------------------------
def render_raster(out_path):
    import rasterio
    from rasterio.windows import from_bounds
    from rasterio.enums import Resampling
    from rasterio.warp import transform_bounds, reproject
    from rasterio.transform import from_bounds as tfb
    from affine import Affine
    lon0, lon1, lat0, lat1 = EXTENT
    with rasterio.open(POP_RASTER) as ds:
        l, b, r, t = transform_bounds("EPSG:4326", ds.crs, lon0, lat0, lon1, lat1, densify_pts=21)
        win = from_bounds(l, b, r, t, ds.transform).round_offsets().round_lengths()
        ww, wh = int(win.width), int(win.height)
        sc = min(3000 / ww, 1.0); ow, oh = max(int(ww*sc), 1), max(int(wh*sc), 1)
        arr = ds.read(1, window=win, out_shape=(oh, ow), resampling=Resampling.average,
                      boundless=True, fill_value=0).astype("float64")
        src_tr = ds.window_transform(win) * Affine.scale(ww/ow, wh/oh)
        src_crs = ds.crs
    dst = np.zeros((oh, ow)); dst_tr = tfb(lon0, lat0, lon1, lat1, ow, oh)
    reproject(arr, dst, src_transform=src_tr, src_crs=src_crs,
              dst_transform=dst_tr, dst_crs="EPSG:4326", resampling=Resampling.average)
    # counts per cell -> persons per km2 (approx cell area in km2)
    dlat = (lat1 - lat0) / oh; dlon = (lon1 - lon0) / ow
    cell_km2 = (dlat * 111.32) * (dlon * 111.32 * np.cos(np.deg2rad(0.5*(lat0+lat1))))
    dens = np.ma.masked_less_equal(dst / max(cell_km2, 1e-9), 1.0)

    fig, ax = plt.subplots(figsize=(13, 8))
    ax.set_xlim(lon0, lon1); ax.set_ylim(lat0, lat1); ax.set_aspect(geo_aspect(EXTENT))
    sea_and_frame(ax, EXTENT)
    im = ax.imshow(dens, extent=EXTENT, origin="upper", cmap=CMAP,
                   norm=mcolors.LogNorm(vmin=10, vmax=np.nanmax(dens)), zorder=3)
    if os.path.exists(PROV_JSON):
        gpd.read_file(PROV_JSON).to_crs(4326).cx[lon0:lon1, lat0:lat1].boundary.plot(
            ax=ax, color="#222", linewidth=0.6, zorder=6)
    graticule(ax, EXTENT); add_scalebar(ax, EXTENT); add_north(ax, EXTENT)
    cb = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.02)
    cb.set_label("Population density (persons km$^{-2}$, log scale)", fontsize=9)
    ax.set_title("Population density of the Marmara basin (gridded)",
                 fontsize=13, fontweight="bold", pad=8)
    fig.tight_layout(); fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    print("written:", out_path)

# ----------------------------------------------------------------------
def load_provinces():
    g = gpd.read_file(PROV_JSON).to_crs(4326)
    namef = next((c for c in ("shapeName", "NAME", "name", "adm1_name") if c in g.columns), None)
    g["name"] = g[namef].astype(str)
    g["norm"] = g["name"].apply(tr_norm)
    g = g[g["norm"].isin(TARGET)].copy()
    # population source
    pop = load_population_csv(POP_CSV) if POP_CSV and os.path.exists(POP_CSV) else {}
    if not pop:
        print("! No POP_CSV -> using APPROXIMATE built-in 2022 values. "
              "Replace with your TÜİK CSV before publishing.")
        pop = POP_BY_NORM
    g["pop"] = g["norm"].map(pop)
    missing = g[g["pop"].isna()]["name"].tolist()
    if missing:
        print("! No population for:", missing, "(check name spelling in the CSV)")
        g = g.dropna(subset=["pop"])
    print(f"  provinces plotted: {len(g)}  total pop: {int(g['pop'].sum()):,}")
    return g

def main():
    outdir = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(outdir, OUT_NAME)
    if MODE == "raster" and POP_RASTER:
        render_raster(out)
    else:
        render_choropleth(load_provinces(), out)

if __name__ == "__main__":
    main()
