#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 4 (geology + active tectonics) of the Marmara basin — clean replot.

Reads local datasets on the local data drive, no icons/infographics:
  * Lithology polygons   : Turkey 500k Lithology V1.TAB   (MapInfo)
  * Faults               : Turkey 500k Faults V1.TAB      (MapInfo)
  * Relief / bathymetry  : GEBCO_Marmara/*.tif            (GeoTIFF)
  * Province outlines    : geoBoundaries-TUR-ADM1.geojson

Two steps:
  1)  python3 plot_geology_marmara.py --inspect
      Prints the columns / unique values of the lithology & fault tables so
      you can confirm which field holds the rock type (LITHO_FIELD below).
  2)  python3 plot_geology_marmara.py
      Renders Figure_04.png (300 dpi) next to this script.

Self-test (no data needed, just checks the plotting pipeline):
      python3 plot_geology_marmara.py --selftest

Requires: geopandas, rasterio, shapely, matplotlib, numpy
          (pip install geopandas rasterio shapely matplotlib)
"""

import os, sys, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.colors import LightSource

# ----------------------------------------------------------------------
# CONFIG  — edit paths/fields here
# ----------------------------------------------------------------------
DATA_ROOT = os.environ.get(
    "DATA_ROOT",
    os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")),
)
TURKEY    = os.path.join(DATA_ROOT, "TURKEY")

LITHO_TAB = os.path.join(TURKEY, "Geological map data of Turkey", "Turkey 500k Lithology V1.TAB")
FAULT_TAB = os.path.join(TURKEY, "Geological map data of Turkey", "Turkey 500k Faults V1.TAB")
RELIEF_TIF= os.path.join(TURKEY, "GEBCO_Marmara", "gebco_2026_n42.0_s39.0_w25.0_e31.0_geotiff.tif")
PROV_JSON = os.path.join(DATA_ROOT, "geoBoundaries-TUR-ADM1.geojson")

# Map extent (deg): lon_min, lon_max, lat_min, lat_max  (Marmara study area)
EXTENT = (26.0, 31.0, 39.3, 41.7)

# Field in the lithology table that carries the rock/age description.
# Run --inspect first; set this to the right column name (e.g. "LITHOLOGY",
# "DESCRIPTIO", "AGE", "ROCK_TYPE"). None -> auto-detect a likely text field.
LITHO_FIELD = None

OUT_NAME = "Figure_04.png"

# ----------------------------------------------------------------------
# Paper legend: six units + colours (mirrors the original legend, corrected)
# ----------------------------------------------------------------------
CLASSES = [
    "Quaternary alluvium",
    "Cenozoic sedimentary",
    "Tertiary volcanic",
    "Mesozoic limestone",
    "Palaeozoic metamorphic",
    "Precambrian crystalline basement",
    "Undifferentiated",
]
CLASS_COLOR = {
    "Quaternary alluvium":              "#FBF3D5",
    "Cenozoic sedimentary":             "#CBA46A",
    "Tertiary volcanic":                "#E67E22",
    "Mesozoic limestone":               "#A2D9CE",
    "Palaeozoic metamorphic":           "#2E9E5B",
    "Precambrian crystalline basement": "#8E7CA1",
    "Undifferentiated":                 "#DCDCDC",
}

def classify_by_keywords(text):
    """Map a free-text lithology/age string to one of the six paper units."""
    if text is None:
        return "Undifferentiated"
    t = str(text).lower()
    def has(*ws): return any(w in t for w in ws)
    if has("alluv", "quatern", "fluvial", "delta", "colluv", "beach", "terrace"):
        return "Quaternary alluvium"
    if has("volcan", "tuff", "basalt", "andesite", "ignimbrit", "pyroclast", "lava", "dacite"):
        return "Tertiary volcanic"
    if has("gneiss", "granit", "crystalline", "migmat", "granodior", "pluton"):
        return "Precambrian crystalline basement"
    if has("schist", "metamorph", "marble", "phyllit", "amphibolit", "slate", "quartzit", "mica"):
        return "Palaeozoic metamorphic"
    if has("limestone", "carbonate", "dolomit", "recif", "reef"):
        return "Mesozoic limestone"
    if has("sediment", "sandstone", "conglomer", "clastic", "flysch", "shale",
            "clay", "marl", "silt", "molasse", "evaporit", "gypsum"):
        return "Cenozoic sedimentary"
    return "Undifferentiated"

# ----------------------------------------------------------------------
# Cartographic helpers
# ----------------------------------------------------------------------
def hillshade(elev, az=315, alt=45):
    ls = LightSource(azdeg=az, altdeg=alt)
    return ls.hillshade(np.nan_to_num(elev, nan=0.0), vert_exag=0.00035, dx=1, dy=1)

def add_scalebar(ax, extent, km=50, x0=0.06, y0=0.07):
    lon0, lon1, lat0, lat1 = extent
    latm = 0.5 * (lat0 + lat1)
    deg = km / (111.320 * np.cos(np.deg2rad(latm)))      # km -> deg lon
    x = lon0 + x0 * (lon1 - lon0)
    y = lat0 + y0 * (lat1 - lat0)
    seg = deg / 2.0
    for i in range(2):
        ax.add_patch(plt.Rectangle((x + i*seg, y), seg, (lat1-lat0)*0.012,
                     facecolor="black" if i == 0 else "white",
                     edgecolor="black", lw=0.8, zorder=20))
    ax.text(x, y + (lat1-lat0)*0.03, "0", ha="center", va="bottom", fontsize=8, zorder=20)
    ax.text(x + deg, y + (lat1-lat0)*0.03, f"{km} km", ha="center", va="bottom", fontsize=8, zorder=20)

def add_north(ax, extent, x0=0.95, y0=0.90):
    lon0, lon1, lat0, lat1 = extent
    x = lon0 + x0 * (lon1 - lon0); y = lat0 + y0 * (lat1 - lat0)
    dy = (lat1 - lat0) * 0.06
    ax.annotate("", xy=(x, y + dy), xytext=(x, y),
                arrowprops=dict(arrowstyle="-|>", lw=1.6, color="black"), zorder=20)
    ax.text(x, y + dy*1.15, "N", ha="center", va="bottom", fontsize=11, fontweight="bold", zorder=20)

def geo_aspect(extent):
    lat0, lat1 = extent[2], extent[3]
    return 1.0 / np.cos(np.deg2rad(0.5 * (lat0 + lat1)))

# ----------------------------------------------------------------------
# Data-agnostic renderer
# ----------------------------------------------------------------------
def render_map(relief, relief_extent, litho_gdf, faults_gdf, prov_gdf, out_path):
    lon0, lon1, lat0, lat1 = EXTENT
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.set_xlim(lon0, lon1); ax.set_ylim(lat0, lat1)
    ax.set_aspect(geo_aspect(EXTENT))

    # 1) relief base: grey hillshade on land, blue on sea
    if relief is not None:
        elev = relief
        hs = hillshade(elev)
        ax.imshow(hs, extent=relief_extent, origin="upper", cmap="gray",
                  vmin=0, vmax=1, alpha=0.55, zorder=1)
        sea = np.ma.masked_where(elev >= 0, elev)
        ax.imshow(sea, extent=relief_extent, origin="upper", cmap="Blues_r",
                  alpha=0.9, zorder=2)
    else:
        ax.set_facecolor("#eef2f4")

    # 2) lithology polygons (semi-transparent so relief texture shows)
    if litho_gdf is not None and len(litho_gdf):
        for cls in CLASSES:
            sub = litho_gdf[litho_gdf["CLASS"] == cls]
            if len(sub):
                sub.plot(ax=ax, facecolor=CLASS_COLOR[cls], edgecolor="none",
                         alpha=0.82, linewidth=0, zorder=3)
        litho_gdf.boundary.plot(ax=ax, color="#5b5b5b", linewidth=0.15, zorder=4)

    # 3) province outlines + labels
    if prov_gdf is not None and len(prov_gdf):
        prov_gdf.boundary.plot(ax=ax, color="#2b2b2b", linewidth=0.7, zorder=6)
        namef = next((c for c in ("shapeName", "NAME", "name", "adm1_name")
                      if c in prov_gdf.columns), None)
        if namef:
            for _, r in prov_gdf.iterrows():
                c = r.geometry.representative_point()
                if lon0 < c.x < lon1 and lat0 < c.y < lat1:
                    ax.text(c.x, c.y, str(r[namef]).upper(), fontsize=8.5,
                            ha="center", va="center", color="#111111", zorder=9,
                            path_effects=None,
                            bbox=dict(boxstyle="round,pad=0.12", fc="white",
                                      ec="none", alpha=0.45))

    # 4) faults
    if faults_gdf is not None and len(faults_gdf):
        faults_gdf.plot(ax=ax, color="#c0392b", linewidth=1.4, zorder=8)

    # 5) sea labels
    for name, (lx, ly, rot) in {
        "MARMARA SEA": (28.2, 40.62, 0),
        "Black Sea":   (29.2, 41.55, 0),
        "Aegean Sea":  (26.15, 39.6, 0),
    }.items():
        if lon0 < lx < lon1 and lat0 < ly < lat1:
            ax.text(lx, ly, name, fontsize=10, style="italic", color="#1a3e6e",
                    ha="center", va="center", zorder=9)
    # NAFZ annotation
    if lon0 < 28.4 < lon1:
        ax.text(28.4, 40.75, "North Anatolian Fault Zone", fontsize=8.5,
                color="#7b241c", style="italic", ha="center", va="center", zorder=9)

    # 6) graticule
    ax.set_xticks(np.arange(np.ceil(lon0), lon1 + 0.01, 1.0))
    ax.set_yticks(np.arange(np.ceil(lat0), lat1 + 0.01, 1.0))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}°E"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}°N"))
    ax.grid(True, color="0.5", lw=0.4, ls=":", alpha=0.6, zorder=5)
    ax.tick_params(labelsize=9)

    # 7) legend
    handles = [Patch(facecolor=CLASS_COLOR[c], edgecolor="#5b5b5b", lw=0.4, label=c)
               for c in CLASSES[:-1]]
    handles.append(Line2D([0], [0], color="#c0392b", lw=1.6, label="Active fault"))
    leg = ax.legend(handles=handles, loc="upper left", fontsize=8.2,
                    title="Geological units & structures", title_fontsize=9,
                    framealpha=0.92, borderpad=0.7)
    leg.set_zorder(21)

    add_scalebar(ax, EXTENT); add_north(ax, EXTENT)
    ax.set_title("Geological and active-tectonic setting of the Marmara basin",
                 fontsize=13, fontweight="bold", pad=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    print("written:", out_path)

# ----------------------------------------------------------------------
# Real data loading
# ----------------------------------------------------------------------
def load_real():
    import geopandas as gpd
    import rasterio
    from rasterio.windows import from_bounds
    lon0, lon1, lat0, lat1 = EXTENT

    # relief
    relief, r_ext = None, None
    if os.path.exists(RELIEF_TIF):
        with rasterio.open(RELIEF_TIF) as ds:
            try:
                win = from_bounds(lon0, lat0, lon1, lat1, ds.transform)
                arr = ds.read(1, window=win)
                t = ds.window_transform(win)
                h, w = arr.shape
                r_ext = (t.c, t.c + w * t.a, t.f + h * t.e, t.f)
            except Exception:
                arr = ds.read(1); b = ds.bounds
                r_ext = (b.left, b.right, b.bottom, b.top)
            relief = arr.astype("float32")
            relief[relief < -12000] = np.nan
    else:
        print("! relief not found:", RELIEF_TIF)

    # lithology
    litho = None
    if os.path.exists(LITHO_TAB):
        litho = gpd.read_file(LITHO_TAB)
        if litho.crs is None:
            print("! lithology has no CRS; assuming EPSG:4326")
            litho.set_crs(4326, inplace=True)
        litho = litho.to_crs(4326).cx[lon0:lon1, lat0:lat1]
        field = LITHO_FIELD or _guess_text_field(litho)
        print(f"  lithology field used: {field}")
        litho["CLASS"] = litho[field].apply(classify_by_keywords)
    else:
        print("! lithology not found:", LITHO_TAB)

    # faults
    faults = None
    if os.path.exists(FAULT_TAB):
        faults = gpd.read_file(FAULT_TAB)
        if faults.crs is None:
            faults.set_crs(4326, inplace=True)
        faults = faults.to_crs(4326).cx[lon0:lon1, lat0:lat1]
    else:
        print("! faults not found:", FAULT_TAB)

    # provinces
    prov = None
    if os.path.exists(PROV_JSON):
        prov = gpd.read_file(PROV_JSON).to_crs(4326).cx[lon0:lon1, lat0:lat1]
    else:
        print("! provinces not found:", PROV_JSON)

    return relief, r_ext, litho, faults, prov

def _guess_text_field(gdf):
    cands = [c for c in gdf.columns if gdf[c].dtype == object and c.lower() != "geometry"]
    for pref in ("litholog", "descrip", "rock", "unit", "age", "formation", "type", "name"):
        for c in cands:
            if pref in c.lower():
                return c
    return cands[0] if cands else gdf.columns[0]

def inspect():
    import geopandas as gpd
    for label, path in (("LITHOLOGY", LITHO_TAB), ("FAULTS", FAULT_TAB)):
        print("=" * 70, f"\n{label}: {path}")
        if not os.path.exists(path):
            print("  (not found)"); continue
        g = gpd.read_file(path)
        print("  CRS:", g.crs, "| rows:", len(g), "| geom:", g.geom_type.unique())
        print("  columns:", list(g.columns))
        for c in g.columns:
            if g[c].dtype == object and c.lower() != "geometry":
                vals = g[c].dropna().unique()
                print(f"   - {c}: {len(vals)} unique -> {list(vals)[:12]}")

# ----------------------------------------------------------------------
# Self-test with synthetic data
# ----------------------------------------------------------------------
def make_synthetic():
    import geopandas as gpd
    from shapely.geometry import Polygon, LineString, box
    lon0, lon1, lat0, lat1 = EXTENT
    ny, nx = 160, 260
    ys = np.linspace(lat1, lat0, ny); xs = np.linspace(lon0, lon1, nx)
    X, Y = np.meshgrid(xs, ys)
    elev = 600*np.exp(-(((X-27)**2+(Y-40)**2)/0.6)) + 500*np.exp(-(((X-30)**2+(Y-40.2)**2)/0.8))
    elev += 50*np.sin(X*3)*np.cos(Y*3)
    elev[(Y > 40.35) & (Y < 40.95)] = -80    # Marmara Sea strip
    elev[Y > 41.45] = -60                     # Black Sea
    r_ext = (lon0, lon1, lat0, lat1)

    cells, cls = [], []
    import random; random.seed(1)
    for i in np.linspace(lon0, lon1, 11)[:-1]:
        for j in np.linspace(lat0, lat1, 9)[:-1]:
            cells.append(box(i, j, i+0.5, j+0.3))
            cls.append(random.choice(CLASSES[:-1]))
    litho = gpd.GeoDataFrame({"CLASS": cls}, geometry=cells, crs=4326)
    litho = litho[litho.intersects(box(lon0, lat0, lon1, lat1))]
    # carve out the sea from land polygons visually: drop cells fully in sea strip
    litho = litho[~litho.geometry.apply(lambda g: 40.4 < g.centroid.y < 40.9)]

    faults = gpd.GeoDataFrame(geometry=[
        LineString([(26.2, 40.5), (28.0, 40.55), (29.2, 40.75), (30.6, 40.72)]),
        LineString([(26.5, 40.2), (27.6, 40.35)]),
    ], crs=4326)

    prov = gpd.GeoDataFrame({"shapeName": ["Istanbul", "Kocaeli", "Bursa", "Tekirdag"]},
        geometry=[box(28.5, 40.9, 29.4, 41.3), box(29.6, 40.6, 30.4, 40.9),
                  box(28.6, 39.9, 29.6, 40.4), box(26.8, 40.9, 28.0, 41.2)], crs=4326)
    return elev, r_ext, litho, faults, prov

# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    outdir = os.path.dirname(os.path.abspath(__file__))

    if a.inspect:
        inspect(); return
    if a.selftest:
        data = make_synthetic()
        render_map(*data, os.path.join(outdir, "Figure_04_SELFTEST.png"))
        return
    data = load_real()
    render_map(*data, os.path.join(outdir, OUT_NAME))

if __name__ == "__main__":
    main()
