#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 11 (tsunami) of the Marmara basin — clean replot.

Bathymetry + coastal-exposure map (no icons / infographic panels):
  * Bathymetry of the Sea of Marmara        -> GEBCO GeoTIFF (water depth).
  * Low-lying coastal zones (elevation-based exposure, <=5 m and <=10 m within
    a coastal buffer)                        -> derived from the same DEM.
  * Main Marmara Fault (tsunamigenic source) -> Turkey 500k Faults V1.TAB.
  * Indicative tsunami-propagation arrows    -> schematic, from the source.
  * Province outlines + coastal city labels  -> geoBoundaries ADM1.

NOTE: the coastal zones are an ELEVATION-BASED exposure proxy ("bathtub"),
NOT a hydrodynamic inundation model. If you have a scenario max-inundation
GeoTIFF, point INUNDATION_TIF at it to render modelled depths instead.

Run:  python3 plot_tsunami_marmara.py        # -> Figure_12.png (300 dpi)
Requires: rasterio, numpy, scipy, matplotlib, geopandas
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
from matplotlib.patches import Patch, FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib.colors import LightSource, to_rgba
from scipy.ndimage import distance_transform_edt
import geopandas as gpd

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
DATA_ROOT = os.environ.get(
    "DATA_ROOT",
    os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")),
)
TURKEY     = os.path.join(DATA_ROOT, "TURKEY")
RELIEF_TIF = os.path.join(TURKEY, "GEBCO_Marmara", "gebco_2026_n42.0_s39.0_w25.0_e31.0_geotiff.tif")
FAULT_TAB  = os.path.join(TURKEY, "Geological map data of Turkey", "Turkey 500k Faults V1.TAB")
PROV_JSON  = os.path.join(DATA_ROOT, "geoBoundaries-TUR-ADM1.geojson")
INUNDATION_TIF = ""      # optional: scenario max-inundation-depth GeoTIFF (metres)

EXTENT     = (25.9, 31.1, 39.2, 41.9)
ELEV_LOW   = 5.0         # m  -> highest-exposure coastal band
ELEV_MID   = 10.0        # m  -> secondary coastal band
COAST_BUFFER_KM = 6.0    # only flag low land within this distance of the sea
OUT_NAME   = "Figure_12.png"

CITIES = [("İstanbul", 28.98, 41.02), ("Bakırköy", 28.87, 40.98), ("İzmit", 29.92, 40.77),
          ("Yalova", 29.28, 40.65), ("Bandırma", 27.97, 40.35), ("Tekirdağ", 27.51, 40.98),
          ("Gemlik", 29.16, 40.43)]

# indicative propagation arrows: (from_lon, from_lat) -> (to_lon, to_lat)
SOURCE = (28.85, 40.85)          # Prince's Islands segment of the Main Marmara Fault
TARGETS = [(29.00, 40.99), (28.60, 40.97), (29.30, 40.66), (29.90, 40.75), (27.6, 40.55)]

# ----------------------------------------------------------------------
def geo_aspect(e): return 1.0 / np.cos(np.deg2rad(0.5 * (e[2] + e[3])))
def hillshade(z): return LightSource(315, 45).hillshade(np.nan_to_num(z, nan=0.0),
                                                        vert_exag=0.00035, dx=1, dy=1)

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
def load_relief():
    import rasterio
    from rasterio.windows import from_bounds
    lon0, lon1, lat0, lat1 = EXTENT
    with rasterio.open(RELIEF_TIF) as ds:
        try:
            win = from_bounds(lon0, lat0, lon1, lat1, ds.transform)
            a = ds.read(1, window=win); t = ds.window_transform(win)
        except Exception:
            a = ds.read(1); t = ds.transform
        h, w = a.shape
        r_ext = (t.c, t.c + w*t.a, t.f + h*t.e, t.f)
        px_km_x = abs(t.a) * 111.320 * np.cos(np.deg2rad(0.5*(lat0+lat1)))
        px_km_y = abs(t.e) * 111.320
    z = a.astype("float32"); z[z < -12000] = np.nan
    return z, r_ext, (px_km_y, px_km_x)

def coastal_exposure(z, sampling):
    sea = z < 0
    dist_km = distance_transform_edt(~sea, sampling=sampling)     # km to nearest sea cell
    land = z >= 0
    near = dist_km <= COAST_BUFFER_KM
    band_low = land & near & (z <= ELEV_LOW)
    band_mid = land & near & (z > ELEV_LOW) & (z <= ELEV_MID)
    depth = np.where(sea, -z, np.nan)                             # water depth (m)
    return band_low, band_mid, depth

# ----------------------------------------------------------------------
def render(z, r_ext, band_low, band_mid, depth, faults, prov, inund, out_path):
    lon0, lon1, lat0, lat1 = EXTENT
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.set_xlim(lon0, lon1); ax.set_ylim(lat0, lat1); ax.set_aspect(geo_aspect(EXTENT))

    # land hillshade + bathymetry
    ax.imshow(hillshade(z), extent=r_ext, origin="upper", cmap="gray", vmin=0, vmax=1,
              alpha=0.5, zorder=1)
    dnorm = mcolors.Normalize(vmin=0, vmax=np.nanpercentile(depth, 99) if np.isfinite(depth).any() else 1200)
    ax.imshow(np.ma.masked_invalid(depth), extent=r_ext, origin="upper", cmap="Blues",
              norm=dnorm, zorder=2)

    # coastal exposure (or modelled inundation, if provided)
    if inund is not None:
        inorm = mcolors.Normalize(vmin=0, vmax=np.nanpercentile(inund.compressed(), 98) if inund.count() else 4)
        ax.imshow(inund, extent=r_ext, origin="upper", cmap="OrRd", norm=inorm, alpha=0.85, zorder=4)
    else:
        for mask, col in ((band_mid, "#fdae61"), (band_low, "#d7301f")):
            rgba = np.zeros((*mask.shape, 4)); rgba[mask] = to_rgba(col, 0.85)
            ax.imshow(rgba, extent=r_ext, origin="upper", zorder=4)

    if prov is not None and len(prov):
        prov.boundary.plot(ax=ax, color="#444", linewidth=0.6, zorder=5)
    if faults is not None and len(faults):
        faults.plot(ax=ax, color="#b2182b", linewidth=1.3, zorder=6)

    # indicative propagation arrows from the source
    for tx, ty in TARGETS:
        ax.add_patch(FancyArrowPatch(SOURCE, (tx, ty), arrowstyle="-|>", mutation_scale=14,
                     lw=1.6, color="#1f6fb2", alpha=0.9, zorder=7,
                     connectionstyle="arc3,rad=0.08"))
    ax.scatter([SOURCE[0]], [SOURCE[1]], marker="*", s=260, facecolor="#ffd400",
               edgecolor="black", linewidth=0.8, zorder=8)
    ax.annotate("Main Marmara Fault\n(tsunami source)", xy=SOURCE, xytext=(4, -14),
                textcoords="offset points", fontsize=8, fontweight="bold", zorder=9,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))

    # city labels
    for nm, lx, ly in CITIES:
        if lon0 < lx < lon1 and lat0 < ly < lat1:
            ax.scatter([lx], [ly], s=12, color="black", zorder=10)
            ax.annotate(nm, xy=(lx, ly), xytext=(3, 3), textcoords="offset points",
                        fontsize=8, fontweight="bold", zorder=11,
                        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.7))
    for nm, (lx, ly) in {"MARMARA SEA": (28.05, 40.55), "Black Sea": (28.6, 41.78),
                         "Aegean Sea": (26.15, 39.55)}.items():
        if lon0 < lx < lon1 and lat0 < ly < lat1:
            ax.text(lx, ly, nm, fontsize=9.5, style="italic", color="#0b2e57",
                    ha="center", va="center", zorder=9)

    ax.set_xticks(np.arange(np.ceil(lon0), lon1+0.01, 1.0))
    ax.set_yticks(np.arange(np.ceil(lat0), lat1+0.01, 1.0))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}°E"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}°N"))
    ax.grid(True, color="0.5", lw=0.4, ls=":", alpha=0.45, zorder=5)
    ax.tick_params(labelsize=9)

    cb = fig.colorbar(ScalarMappable(norm=dnorm, cmap="Blues"), ax=ax, fraction=0.030, pad=0.02)
    cb.set_label("Water depth (m)", fontsize=9)

    if inund is not None:
        exp_handles = [Patch(facecolor="#d7301f", edgecolor="none", label="Modelled inundation")]
    else:
        exp_handles = [Patch(facecolor="#d7301f", edgecolor="none", label=f"Coastal exposure ≤ {ELEV_LOW:.0f} m"),
                       Patch(facecolor="#fdae61", edgecolor="none", label=f"Coastal exposure ≤ {ELEV_MID:.0f} m")]
    handles = exp_handles + [
        Line2D([0], [0], color="#b2182b", lw=1.6, label="Active fault"),
        Line2D([0], [0], color="#1f6fb2", lw=1.6, label="Indicative propagation"),
        Line2D([0], [0], marker="*", linestyle="", markerfacecolor="#ffd400",
               markeredgecolor="k", markersize=12, label="Tsunami source")]
    leg = ax.legend(handles=handles, loc="upper left", fontsize=8, title="Tsunami exposure",
                    title_fontsize=9, framealpha=0.93, borderpad=0.7)
    leg.set_zorder(31)

    add_scalebar(ax, EXTENT); add_north(ax, EXTENT)
    ax.set_title("Tsunami exposure of the Marmara basin", fontsize=13, fontweight="bold", pad=8)
    note = ("Coastal zones are an elevation-based exposure proxy from the DEM, not a hydrodynamic model; "
            "propagation arrows are indicative.") if inund is None else \
           "Modelled maximum inundation depth; bathymetry from GEBCO."
    ax.text(0.5, -0.06, note, transform=ax.transAxes, ha="center", va="top", fontsize=7.3, color="#555")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    print("written:", out_path)

# ----------------------------------------------------------------------
def load_vectors():
    lon0, lon1, lat0, lat1 = EXTENT
    faults = prov = None
    if os.path.exists(FAULT_TAB):
        try:
            f = gpd.read_file(FAULT_TAB)
            if f.crs is None: f.set_crs(4326, inplace=True)
            faults = f.to_crs(4326).cx[lon0:lon1, lat0:lat1]
        except Exception as ex:
            print("faults skipped:", ex)
    if os.path.exists(PROV_JSON):
        prov = gpd.read_file(PROV_JSON).to_crs(4326).cx[lon0:lon1, lat0:lat1]
    return faults, prov

def load_inundation(r_ext, shape):
    if not INUNDATION_TIF or not os.path.exists(INUNDATION_TIF):
        return None
    import rasterio
    from rasterio.warp import reproject, Resampling
    from rasterio.transform import from_bounds as tfb
    lon0, lon1, lat0, lat1 = EXTENT
    with rasterio.open(INUNDATION_TIF) as ds:
        src = ds.read(1).astype("float64"); src_crs = ds.crs; src_tr = ds.transform
        nod = ds.nodata
    dst = np.full(shape, np.nan)
    reproject(src, dst, src_transform=src_tr, src_crs=src_crs,
              dst_transform=tfb(lon0, lat0, lon1, lat1, shape[1], shape[0]),
              dst_crs="EPSG:4326", src_nodata=nod, dst_nodata=np.nan, resampling=Resampling.bilinear)
    dst[dst <= 0] = np.nan
    return np.ma.masked_invalid(dst)

def main():
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUT_NAME)
    z, r_ext, sampling = load_relief()
    band_low, band_mid, depth = coastal_exposure(z, sampling)
    faults, prov = load_vectors()
    inund = load_inundation(r_ext, z.shape)
    render(z, r_ext, band_low, band_mid, depth, faults, prov, inund, out)

if __name__ == "__main__":
    main()
