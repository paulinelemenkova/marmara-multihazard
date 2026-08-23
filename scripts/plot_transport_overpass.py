#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 6 (transport network) of the Marmara basin — clean replot.
Direct Overpass API version (uses `requests`; no osmnx).

One Overpass query for the whole bbox (motorways/trunk/primary, railways,
ferry routes, airports, ports) instead of osmnx's ~52 sub-queries, so it is
usually much faster. Base map = your local geoBoundaries provinces.

Run:
    python3 plot_transport_overpass.py       # -> Figure_07.png (300 dpi)

Requires: requests, geopandas, shapely, numpy, matplotlib
© OpenStreetMap contributors (ODbL).
"""

import os, sys, json, time

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
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import geopandas as gpd
from shapely.geometry import LineString, Point

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
DATA_ROOT = os.environ.get(
    "DATA_ROOT",
    os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")),
)
PROV_JSON = os.path.join(DATA_ROOT, "geoBoundaries-TUR-ADM1.geojson")
EXTENT    = (25.9, 31.1, 39.2, 41.9)      # lon_min, lon_max, lat_min, lat_max
OUT_NAME  = "Figure_07.png"
CACHE     = os.path.join(DATA_ROOT, "overpass_transport.json")     # raw response cached here (delete to refresh)

OVERPASS_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]

CITIES = [("İstanbul", 28.98, 41.02), ("İzmit", 29.92, 40.77), ("Bursa", 29.06, 40.19),
          ("Tekirdağ", 27.51, 40.98), ("Yalova", 29.28, 40.65), ("Balıkesir", 27.89, 39.65),
          ("Çanakkale", 26.41, 40.15), ("Sakarya", 30.40, 40.77), ("Edirne", 26.56, 41.68)]

# ----------------------------------------------------------------------
# Overpass fetch
# ----------------------------------------------------------------------
def overpass_query(e):
    w, ee, s, n = e[0], e[1], e[2], e[3]
    bbox = f"({s},{w},{n},{ee})"
    return f"""[out:json][timeout:180];
(
  way["highway"~"^(motorway|motorway_link|trunk|trunk_link|primary)$"]{bbox};
  way["railway"="rail"]{bbox};
  way["route"="ferry"]{bbox};
  relation["route"="ferry"]{bbox};
  way["aeroway"="aerodrome"]{bbox};
  node["aeroway"="aerodrome"]{bbox};
  way["amenity"="ferry_terminal"]{bbox};
  node["amenity"="ferry_terminal"]{bbox};
  way["harbour"="yes"]{bbox};
);
out geom;"""

def fetch(e):
    if os.path.exists(CACHE):
        print("using cached", CACHE, "(delete it to re-download)")
        return json.load(open(CACHE, encoding="utf-8"))
    q = overpass_query(e)
    for url in OVERPASS_MIRRORS:
        try:
            print("querying", url, "…")
            r = requests.post(url, data={"data": q}, timeout=200,
                              headers={"User-Agent": "marmara-transport-map/1.0"})
            if r.status_code == 200 and r.text.strip().startswith("{"):
                data = r.json()
                json.dump(data, open(CACHE, "w", encoding="utf-8"))
                print(f"  got {len(data.get('elements', []))} elements  (cached -> {CACHE})")
                return data
            print("  HTTP", r.status_code, "- trying next mirror")
        except Exception as ex:
            print("  failed:", ex, "- trying next mirror")
        time.sleep(2)
    raise SystemExit("All Overpass mirrors failed. Try again later or reduce EXTENT.")

# ----------------------------------------------------------------------
# parse Overpass JSON -> layers
# ----------------------------------------------------------------------
def parse(data):
    m, p, rail, ferry, air, port = [], [], [], [], [], []
    for el in data.get("elements", []):
        t = el.get("type"); tags = el.get("tags", {}) or {}
        if t == "way" and el.get("geometry"):
            coords = [(g["lon"], g["lat"]) for g in el["geometry"]]
            if len(coords) < 2:
                continue
            line = LineString(coords)
            hw, rw, rt = tags.get("highway"), tags.get("railway"), tags.get("route")
            if hw in ("motorway", "motorway_link"):
                m.append(line)
            elif hw in ("trunk", "trunk_link", "primary"):
                p.append(line)
            elif rw == "rail":
                rail.append(line)
            elif rt == "ferry":
                ferry.append(line)
            elif tags.get("aeroway") == "aerodrome":
                air.append(line.centroid)
            elif tags.get("amenity") == "ferry_terminal" or tags.get("harbour") == "yes":
                port.append(line.centroid)
        elif t == "node":
            pt = Point(el["lon"], el["lat"])
            if tags.get("aeroway") == "aerodrome":
                air.append(pt)
            elif tags.get("amenity") == "ferry_terminal":
                port.append(pt)
        elif t == "relation" and tags.get("route") == "ferry":
            for mem in el.get("members", []):
                g = mem.get("geometry")
                if g and len(g) >= 2:
                    ferry.append(LineString([(q["lon"], q["lat"]) for q in g]))

    def L(geoms): return gpd.GeoDataFrame(geometry=geoms, crs=4326) if geoms else None
    return {"motorway": L(m), "primary": L(p), "rail": L(rail), "ferry": L(ferry),
            "airport": L(air), "port": L(port)}

# ----------------------------------------------------------------------
# cartographic helpers + renderer  (identical styling to the osmnx version)
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

def graticule(ax, e):
    ax.set_xticks(np.arange(np.ceil(e[0]), e[1]+0.01, 1.0))
    ax.set_yticks(np.arange(np.ceil(e[2]), e[3]+0.01, 1.0))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}°E"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}°N"))
    ax.grid(True, color="0.5", lw=0.4, ls=":", alpha=0.5, zorder=6)
    ax.tick_params(labelsize=9)

def render(layers, prov, out_path):
    lon0, lon1, lat0, lat1 = EXTENT
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.set_xlim(lon0, lon1); ax.set_ylim(lat0, lat1); ax.set_aspect(geo_aspect(EXTENT))
    ax.set_facecolor("#cfe0ec")
    if prov is not None and len(prov):
        prov.plot(ax=ax, facecolor="#eef1ec", edgecolor="#9aa0a6", linewidth=0.6, zorder=2)

    def plot(key, **kw):
        g = layers.get(key)
        if g is not None and len(g):
            g.plot(ax=ax, **kw)
    plot("ferry",    color="#1f6fb2", linewidth=1.0, linestyle=(0, (4, 3)), zorder=4)
    plot("rail",     color="#333333", linewidth=0.9, linestyle=(0, (5, 2)), zorder=5)
    plot("primary",  color="#E8912A", linewidth=1.1, zorder=6)
    plot("motorway", color="#D7191C", linewidth=2.2, zorder=7)

    for key, marker, fc in (("airport", "^", "white"), ("port", "s", "#20486b")):
        g = layers.get(key)
        if g is not None and len(g):
            g = g.cx[lon0:lon1, lat0:lat1]
            if len(g):
                ax.scatter(g.geometry.x, g.geometry.y, s=55 if marker == "^" else 32,
                           marker=marker, facecolor=fc, edgecolor="black", linewidth=0.8, zorder=9)

    for nm, lx, ly in CITIES:
        if lon0 < lx < lon1 and lat0 < ly < lat1:
            ax.scatter([lx], [ly], s=14, color="black", zorder=10)
            ax.annotate(nm, xy=(lx, ly), xytext=(3, 3), textcoords="offset points",
                        fontsize=8.5, fontweight="bold", zorder=11,
                        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.7))

    for name, (lx, ly) in {"MARMARA SEA": (28.0, 40.55), "Black Sea": (28.6, 41.75),
                           "Aegean Sea": (26.15, 39.55)}.items():
        if lon0 < lx < lon1 and lat0 < ly < lat1:
            ax.text(lx, ly, name, fontsize=9.5, style="italic", color="#14406e",
                    ha="center", va="center", zorder=8)

    graticule(ax, EXTENT); add_scalebar(ax, EXTENT); add_north(ax, EXTENT)
    handles = [
        Line2D([0], [0], color="#D7191C", lw=2.2, label="Motorway"),
        Line2D([0], [0], color="#E8912A", lw=1.4, label="Trunk / primary road"),
        Line2D([0], [0], color="#333333", lw=1.0, ls=(0, (5, 2)), label="Railway"),
        Line2D([0], [0], color="#1f6fb2", lw=1.0, ls=(0, (4, 3)), label="Ferry route"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor="white",
               markeredgecolor="black", markersize=8, label="Airport"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#20486b",
               markeredgecolor="black", markersize=7, label="Port / ferry terminal"),
    ]
    leg = ax.legend(handles=handles, loc="upper left", fontsize=8.2,
                    title="Transport network", title_fontsize=9, framealpha=0.93, borderpad=0.7)
    leg.set_zorder(26)
    ax.set_title("Transport network of the Marmara basin", fontsize=13, fontweight="bold", pad=8)
    ax.text(0.5, -0.06, "Road, rail and ferry data © OpenStreetMap contributors (ODbL).",
            transform=ax.transAxes, ha="center", va="top", fontsize=7.5, color="#555")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    print("written:", out_path)

# ----------------------------------------------------------------------
def load_provinces():
    if not os.path.exists(PROV_JSON):
        print("! provinces not found:", PROV_JSON); return None
    lon0, lon1, lat0, lat1 = EXTENT
    return gpd.read_file(PROV_JSON).to_crs(4326).cx[lon0:lon1, lat0:lat1]

def main():
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUT_NAME)
    layers = parse(fetch(EXTENT))
    for k, g in layers.items():
        print(f"  {k:9s}: {0 if g is None else len(g)} features")
    render(layers, load_provinces(), out)

if __name__ == "__main__":
    main()
