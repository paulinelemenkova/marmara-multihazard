#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 7 (wind hazard) of the Marmara basin — clean replot.

A single wind-speed surface (no icons / climate infographic panels), from
EITHER of two sources:

  RASTER mode  — a wind-speed GeoTIFF (set WIND_TIF). Best option: the
     Global Wind Atlas mean wind speed at 100 m (free, per-country GeoTIFF):
     https://globalwindatlas.info  ->  Download  ->  GIS files (mean wind speed).

  ERA5 mode    — an ERA5 NetCDF with 10-m components u10/v10 (set ERA5_NC).
     The script computes wind speed = sqrt(u^2+v^2) and reduces over time
     (STAT = "mean" or a percentile like "p95" as an extreme-wind proxy),
     and can overlay the mean wind-direction vectors.

Base map = your local geoBoundaries provinces; sea shown as backdrop.

Run:
    python3 plot_windhazard_marmara.py       # -> Figure_08.png (300 dpi)

Requires: geopandas, numpy, matplotlib, and
          rasterio (RASTER mode)  /  xarray + netCDF4 (ERA5 mode)
"""

import os, sys

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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
import geopandas as gpd

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
DATA_ROOT = os.environ.get(
    "DATA_ROOT",
    os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")),
)
PROV_JSON  = os.path.join(DATA_ROOT, "geoBoundaries-TUR-ADM1.geojson")

WIND_TIF   = os.path.join(DATA_ROOT, "TURKEY", "TUR_wind-speed_100m.tif")  # Global Wind Atlas 100 m
ERA5_NC    = ""      # e.g. os.path.join(DATA_ROOT,"TURKEY","era5_marmara_10m_wind.nc")
STAT       = "mean"  # ERA5 only: "mean" or "p95"/"p98" (percentile as extreme proxy)
SHOW_VECTORS = True   # ERA5 only: overlay mean wind-direction arrows

# If neither WIND_TIF nor ERA5_NC is set, auto-download the Global Wind Atlas
# mean wind-speed GeoTIFF for the country below and cache it next to this script.
GWA_AUTO    = True
GWA_COUNTRY = "TUR"   # ISO-3166 alpha-3
GWA_HEIGHT  = 100     # 10 / 50 / 100 / 150 / 200 m

EXTENT     = (25.9, 31.1, 39.2, 41.9)
CMAP       = "turbo"                            # rainbow (Google Turbo)
CBAR_LABEL = "Mean wind speed at 100 m (m s$^{-1}$)"
TITLE      = "Wind hazard of the Marmara basin"
OUT_NAME   = "Figure_08.png"

CITIES = [("İstanbul", 28.98, 41.02), ("İzmit", 29.92, 40.77), ("Bursa", 29.06, 40.19),
          ("Tekirdağ", 27.51, 40.98), ("Çanakkale", 26.41, 40.15), ("Bandırma", 27.97, 40.35)]

# ----------------------------------------------------------------------
# cartographic helpers
# ----------------------------------------------------------------------
def geo_aspect(e): return 1.0 / np.cos(np.deg2rad(0.5 * (e[2] + e[3])))

def add_scalebar(ax, e, km=50, x0=0.055, y0=0.06):
    lon0, lon1, lat0, lat1 = e
    deg = km / (111.320 * np.cos(np.deg2rad(0.5 * (lat0 + lat1))))
    x = lon0 + x0*(lon1-lon0); y = lat0 + y0*(lat1-lat0); seg = deg/2
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

def decorate(ax):
    lon0, lon1, lat0, lat1 = EXTENT
    ax.set_xlim(lon0, lon1); ax.set_ylim(lat0, lat1); ax.set_aspect(geo_aspect(EXTENT))
    if os.path.exists(PROV_JSON):
        prov = gpd.read_file(PROV_JSON).to_crs(4326).cx[lon0:lon1, lat0:lat1]
        prov.boundary.plot(ax=ax, color="#333333", linewidth=0.7, zorder=8)
    for nm, lx, ly in CITIES:
        if lon0 < lx < lon1 and lat0 < ly < lat1:
            ax.scatter([lx], [ly], s=12, color="black", zorder=10)
            ax.annotate(nm, xy=(lx, ly), xytext=(3, 3), textcoords="offset points",
                        fontsize=8.5, fontweight="bold", zorder=11,
                        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.7))
    for name, (lx, ly) in {"MARMARA SEA": (28.0, 40.55), "Black Sea": (28.6, 41.78),
                           "Aegean Sea": (26.15, 39.55)}.items():
        if lon0 < lx < lon1 and lat0 < ly < lat1:
            ax.text(lx, ly, name, fontsize=9.5, style="italic", color="#14406e",
                    ha="center", va="center", zorder=9)
    ax.set_xticks(np.arange(np.ceil(lon0), lon1+0.01, 1.0))
    ax.set_yticks(np.arange(np.ceil(lat0), lat1+0.01, 1.0))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}°E"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}°N"))
    ax.grid(True, color="0.5", lw=0.4, ls=":", alpha=0.45, zorder=6)
    ax.tick_params(labelsize=9)
    add_scalebar(ax, EXTENT); add_north(ax, EXTENT)
    ax.set_title(TITLE, fontsize=13, fontweight="bold", pad=8)

def finish(fig, ax, mappable, source_note):
    cb = fig.colorbar(mappable, ax=ax, fraction=0.030, pad=0.02)
    cb.set_label(CBAR_LABEL, fontsize=9)
    ax.text(0.5, -0.06, source_note, transform=ax.transAxes, ha="center", va="top",
            fontsize=7.5, color="#555")
    outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUT_NAME)
    fig.tight_layout(); fig.savefig(outpath, dpi=300, bbox_inches="tight", facecolor="white")
    print("written:", outpath)

# ----------------------------------------------------------------------
# RASTER mode (GeoTIFF -> WGS84 grid over EXTENT)
# ----------------------------------------------------------------------
def render_raster(out_source="Wind data: Global Wind Atlas."):
    import rasterio
    from rasterio.windows import from_bounds
    from rasterio.enums import Resampling
    from rasterio.warp import transform_bounds, reproject
    from rasterio.transform import from_bounds as tfb
    from affine import Affine
    lon0, lon1, lat0, lat1 = EXTENT
    with rasterio.open(WIND_TIF) as ds:
        l, b, r, t = transform_bounds("EPSG:4326", ds.crs, lon0, lat0, lon1, lat1, densify_pts=21)
        win = from_bounds(l, b, r, t, ds.transform).round_offsets().round_lengths()
        ww, wh = int(win.width), int(win.height)
        sc = min(2000 / max(ww, 1), 1.0); ow, oh = max(int(ww*sc), 1), max(int(wh*sc), 1)
        arr = ds.read(1, window=win, out_shape=(oh, ow), resampling=Resampling.average,
                      boundless=True, fill_value=np.nan).astype("float64")
        nod = ds.nodata
        if nod is not None:
            arr[arr == nod] = np.nan
        src_tr = ds.window_transform(win) * Affine.scale(ww/ow, wh/oh)
        src_crs = ds.crs
    dst = np.full((oh, ow), np.nan); dst_tr = tfb(lon0, lat0, lon1, lat1, ow, oh)
    reproject(arr, dst, src_transform=src_tr, src_crs=src_crs,
              dst_transform=dst_tr, dst_crs="EPSG:4326", resampling=Resampling.average,
              src_nodata=np.nan, dst_nodata=np.nan)
    grid = np.ma.masked_invalid(dst)

    fig, ax = plt.subplots(figsize=(13, 8)); ax.set_facecolor("#cfe0ec")
    vmin, vmax = np.nanpercentile(grid.compressed(), [2, 98]) if grid.count() else (0, 1)
    im = ax.imshow(grid, extent=EXTENT, origin="upper", cmap=CMAP,
                   vmin=vmin, vmax=vmax, zorder=3)
    decorate(ax); finish(fig, ax, im, out_source)

# ----------------------------------------------------------------------
# ERA5 mode (NetCDF u10/v10)
# ----------------------------------------------------------------------
def _pick(ds, names):
    for n in names:
        if n in ds.variables or n in getattr(ds, "coords", {}):
            return n
    return None

def render_era5(out_source="Wind data: ERA5 (ECMWF)."):
    import xarray as xr
    lon0, lon1, lat0, lat1 = EXTENT
    ds = xr.open_dataset(ERA5_NC)
    uname = _pick(ds, ["u10", "10u", "u10m", "U10"]);  vname = _pick(ds, ["v10", "10v", "v10m", "V10"])
    lonn = _pick(ds, ["longitude", "lon", "x"]);       latn = _pick(ds, ["latitude", "lat", "y"])
    if not all([uname, vname, lonn, latn]):
        raise SystemExit(f"Could not identify u10/v10/lon/lat in {ERA5_NC}. Vars: {list(ds.variables)}")
    u, v = ds[uname], ds[vname]
    # normalise longitudes if 0..360
    if float(ds[lonn].max()) > 180:
        ds = ds.assign_coords({lonn: (((ds[lonn] + 180) % 360) - 180)}).sortby(lonn)
        u, v = ds[uname], ds[vname]
    u = u.sel({lonn: slice(lon0, lon1), latn: slice(max(lat0, lat1), min(lat0, lat1))
               if float(ds[latn][0]) > float(ds[latn][-1]) else slice(lat0, lat1)})
    v = v.sel({lonn: u[lonn], latn: u[latn]})
    spd = np.sqrt(u**2 + v**2)
    tdim = _pick(ds, ["time", "valid_time"]) or next((d for d in spd.dims if d not in (lonn, latn)), None)
    if tdim and tdim in spd.dims:
        if STAT.startswith("p"):
            q = float(STAT[1:]) / 100.0
            speed = spd.quantile(q, dim=tdim); um = u.mean(tdim); vm = v.mean(tdim)
        else:
            speed = spd.mean(tdim); um = u.mean(tdim); vm = v.mean(tdim)
    else:
        speed, um, vm = spd, u, v
    lons = np.asarray(speed[lonn]); lats = np.asarray(speed[latn]); Z = np.asarray(speed)

    fig, ax = plt.subplots(figsize=(13, 8)); ax.set_facecolor("#cfe0ec")
    vmin, vmax = np.nanpercentile(Z, [2, 98])
    pm = ax.pcolormesh(lons, lats, Z, cmap=CMAP, vmin=vmin, vmax=vmax,
                       shading="auto", zorder=3)
    if SHOW_VECTORS:
        step = max(1, len(lons) // 22)
        ax.quiver(lons[::step], lats[::step],
                  np.asarray(um)[::step, ::step], np.asarray(vm)[::step, ::step],
                  color="#222222", scale=250, width=0.0022, alpha=0.7, zorder=7)
    decorate(ax); finish(fig, ax, pm, out_source)

# ----------------------------------------------------------------------
def download_gwa(country, height, dest):
    import requests
    url = f"https://globalwindatlas.info/api/gis/country/{country}/wind-speed/{height}"
    print("Downloading Global Wind Atlas mean wind speed:", url)
    try:
        r = requests.get(url, timeout=300, headers={"User-Agent": "marmara-wind-map/1.0"})
    except Exception as ex:
        raise SystemExit(f"GWA download failed ({ex}).\n" + _GWA_MANUAL)
    if r.status_code != 200 or r.content[:4] not in (b"II*\x00", b"MM\x00*"):
        raise SystemExit(f"GWA download failed (HTTP {r.status_code}, not a GeoTIFF).\n" + _GWA_MANUAL)
    open(dest, "wb").write(r.content)
    print(f"  saved {dest} ({len(r.content)//(1024*1024)} MB)")
    return dest

_GWA_MANUAL = (
    "Download it manually instead:\n"
    "  https://globalwindatlas.info  ->  pan to Türkiye / the Marmara region on the map\n"
    "  ->  open the GIS tab in the right sidebar  ->  download the mean wind-speed TIF,\n"
    "  then set WIND_TIF at the top of this script to that file.")

def main():
    global WIND_TIF
    if not WIND_TIF and not ERA5_NC and GWA_AUTO:
        cache = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             f"gwa_windspeed_{GWA_HEIGHT}m_{GWA_COUNTRY}.tif")
        WIND_TIF = cache if os.path.exists(cache) else download_gwa(GWA_COUNTRY, GWA_HEIGHT, cache)
        if os.path.exists(cache):
            print("using cached", os.path.basename(cache), "(delete it to re-download)")
    if WIND_TIF:
        render_raster(out_source=f"Wind data: Global Wind Atlas (DTU), mean speed at {GWA_HEIGHT} m.")
    elif ERA5_NC:
        render_era5()
    else:
        sys.exit("Set WIND_TIF (Global Wind Atlas GeoTIFF) or ERA5_NC (ERA5 u10/v10 NetCDF), "
                 "or leave GWA_AUTO=True to fetch it automatically.")

if __name__ == "__main__":
    main()
