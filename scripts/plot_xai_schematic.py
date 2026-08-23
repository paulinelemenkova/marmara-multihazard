#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 9 (Explainable AI / SHAP) — clean schematic replot.

A pure-matplotlib schematic of the XAI workflow used to interpret the
multi-hazard ML models: input feature groups -> trained models ->
SHAP explainer (TreeSHAP / DeepSHAP) -> global + local explanations.

No icons / clip-art / raster pictures. Colour-blind-safe Okabe-Ito palette.
The global ranking and local force panels are ILLUSTRATIVE (schematic), not
model outputs — a note on the figure states this.

Run from your script folder.  Output: Figure_10.png (300 dpi) next to this file.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib import mathtext as _mathtext

def safe_math(s, fallback):
    """Return s if matplotlib's mathtext can parse it, else a plain fallback.
    (Older matplotlib lacks \\setminus / \\dfrac, which would crash rendering.)"""
    try:
        _mathtext.MathTextParser("agg").parse(s)
        return s
    except Exception:
        return fallback

plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"]})

OK = {"orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "yellow": "#F0E442",
      "blue": "#0072B2", "verm": "#D55E00", "purple": "#CC79A7", "grey": "#4D4D4D"}
ARROW = "#3A3A3A"

def tint(h, f):
    h = h.lstrip("#"); r, g, b = (int(h[i:i+2], 16) for i in (0, 2, 4))
    r = int(r+(255-r)*f); g = int(g+(255-g)*f); b = int(b+(255-b)*f)
    return f"#{r:02x}{g:02x}{b:02x}"

fig, ax = plt.subplots(figsize=(15, 8.6))
ax.set_xlim(0, 160); ax.set_ylim(0, 100); ax.axis("off")

def rbox(x, y, w, h, fc, ec, lw=1.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.5",
                                mutation_aspect=0.62, lw=lw, edgecolor=ec, facecolor=fc, zorder=2))

def header(x, y, w, h, title, hc, bc, tfs=11.5):
    rbox(x, y, w, h, bc, hc)
    hh = 6.5
    ax.add_patch(FancyBboxPatch((x, y+h-hh), w, hh, boxstyle="round,pad=0,rounding_size=1.5",
                                mutation_aspect=0.62, lw=0, facecolor=hc, zorder=3))
    ax.text(x+w/2, y+h-hh/2, title, ha="center", va="center", color="white",
            fontsize=tfs, fontweight="bold", zorder=4)

def arrow(p1, p2, rad=0.0, lw=1.8):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=16, lw=lw,
                                 color=ARROW, zorder=1, connectionstyle=f"arc3,rad={rad}"))

def txt(x, y, s, fs=9, **kw):
    ax.text(x, y, s, fontsize=fs, zorder=5, **kw)

# ---------------------------------------------------------------- title
txt(80, 96, "Explainable AI (XAI): interpreting the multi-hazard models with SHAP",
    fs=16.5, ha="center", va="center", fontweight="bold", color="#111")
txt(80, 91.5, "Shapley additive explanations decompose each prediction into per-feature contributions",
    fs=11, ha="center", va="center", color="#555")

# ---------------------------------------------------------------- (1) inputs
ix, iw, iy, ih = 3, 36, 24, 60
header(ix, iy, iw, ih, "INPUT FEATURES", OK["blue"], tint(OK["blue"], 0.9))
groups = [("Spectral & SAR", "NDBI · NDVI · GLCM\ncoherence · backscatter"),
          ("Geotechnical", "V$_{s30}$ · soil class\ndepth to bedrock"),
          ("Structural", "age · storeys\nsystem · material")]
gy = iy + ih - 6.5 - 3
for name, items in groups:
    rbox(ix+2.5, gy-13.5, iw-5, 13, tint(OK["blue"], 0.78), OK["blue"], lw=1.0)
    txt(ix+iw/2, gy-3.0, name, fs=9.5, ha="center", va="center", fontweight="bold", color="#123")
    txt(ix+iw/2, gy-9.0, items, fs=8, ha="center", va="center", color="#123", linespacing=1.15)
    gy -= 16
txt(ix+iw/2, iy+2.6, "→ 47-D feature tensor", fs=9, ha="center", va="center",
    style="italic", color="#123")

# ---------------------------------------------------------------- (2) models + SHAP
mx, mw = 45, 40
# trained models
header(mx, 54, mw, 30, "TRAINED MODELS", OK["green"], tint(OK["green"], 0.9))
rbox(mx+2.5, 62.5, mw-5, 8.5, tint(OK["green"], 0.8), OK["green"], lw=1.0)
txt(mx+mw/2, 66.8, "Tree ensembles\nXGBoost · Random Forest", fs=8.6, ha="center", va="center",
    color="#0a3", linespacing=1.15)
rbox(mx+2.5, 55.5, mw-5, 6.5, tint(OK["green"], 0.8), OK["green"], lw=1.0)
txt(mx+mw/2, 58.7, "Neural nets: CNN · LSTM · GNN · Transformer", fs=8.4, ha="center", va="center",
    color="#0a3")
# shap explainer
header(mx, 24, mw, 26, "SHAP EXPLAINER", OK["purple"], tint(OK["purple"], 0.9))
txt(mx+mw/2, 40.5, "TreeSHAP  (tree models)\nDeepSHAP  (neural nets)", fs=9, ha="center",
    va="center", fontweight="bold", color="#623", linespacing=1.2)
_eq = safe_math(
    r"$\phi_j=\sum_{S\subseteq F\backslash\{j\}}\frac{|S|!\,(|F|-|S|-1)!}{|F|!}\,[f(S\cup\{j\})-f(S)]$",
    "phi_j = sum_S  w(S) [ f(S \u222a {j}) \u2212 f(S) ]")
txt(mx+mw/2, 31.5, _eq, fs=10.5, ha="center", va="center", color="#222")
txt(mx+mw/2, 26.2, "contribution of feature $j$ to a single prediction", fs=7.8,
    ha="center", va="center", style="italic", color="#555")

# ---------------------------------------------------------------- (3) global importance
gx, gw, gyb, gh = 92, 65, 50, 34
header(gx, gyb, gw, gh, "GLOBAL FEATURE IMPORTANCE", OK["orange"], tint(OK["orange"], 0.92))
feat = ["V$_{s30}$ (site amplification)", "Peak ground acceleration", "Distance to Main Marmara Fault",
        "Soil class / depth to bedrock", "Pre-1975 building age", "50-yr design gust speed",
        "Coastal elevation / inundation"]
vals = [1.00, 0.86, 0.72, 0.60, 0.48, 0.34, 0.22]           # illustrative, unitless
bx0 = gx + 33; bxmax = gx + gw - 3; by = gyb + gh - 6.5 - 3.2
bar_h = (by - (gyb+3)) / len(feat)
for name, v in zip(feat, vals):
    ax.add_patch(Rectangle((bx0, by-bar_h*0.68), (bxmax-bx0)*v, bar_h*0.62,
                 facecolor=OK["orange"], edgecolor="none", zorder=4))
    txt(bx0-1.5, by-bar_h*0.37, name, fs=7.6, ha="right", va="center", color="#222")
    by -= bar_h
ax.annotate("", xy=(bxmax, gyb+2.6), xytext=(bx0, gyb+2.6),
            arrowprops=dict(arrowstyle="-|>", lw=1.0, color="#666"), zorder=4)
txt((bx0+bxmax)/2, gyb+1.0, "mean |SHAP value|  (relative)", fs=7.5, ha="center", va="top",
    style="italic", color="#555")

# ---------------------------------------------------------------- (4) local explanation
lx, lw_, lyb, lh = 92, 65, 8, 36
header(lx, lyb, lw_, lh, "LOCAL EXPLANATION  (force plot)", OK["sky"], tint(OK["sky"], 0.92))
base = lx + 30                          # base-value x
mid_y = lyb + lh/2 - 3
# baseline / axis
ax.plot([lx+4, lx+lw_-4], [mid_y, mid_y], color="#999", lw=0.8, zorder=3)
ax.plot([base, base], [mid_y-6, mid_y+6], color="#333", lw=1.0, ls=(0, (3, 2)), zorder=3)
txt(base, mid_y+7.5, "base value  $E[f(x)]$", fs=7.6, ha="center", va="bottom", color="#333")
# positive (push risk up, red, to the right)
pos = [("low $V_{s30}$", 11), ("pre-1975 building", 12), ("near-fault", 8)]
x = base
for i, (name, d) in enumerate(pos):
    ax.add_patch(Rectangle((x, mid_y-1.6), d, 3.2, facecolor=OK["verm"],
                 edgecolor="white", lw=0.6, zorder=4))
    ly = mid_y - 2.8 - (2.6 if i % 2 else 0)
    txt(x+d/2, ly, name, fs=6.6, ha="center", va="top", color=OK["verm"])
    x += d
fx = x
ax.annotate("", xy=(fx, mid_y), xytext=(fx, mid_y+5.5),
            arrowprops=dict(arrowstyle="-", lw=0.8, color="#333"), zorder=4)
txt(fx, mid_y+6.2, "prediction  $f(x)$", fs=7.6, ha="center", va="bottom", fontweight="bold", color="#111")
# negative (push risk down, blue, to the left)
neg = [("firm bedrock", 10), ("high elevation", 11)]
x = base
for i, (name, d) in enumerate(neg):
    ax.add_patch(Rectangle((x-d, mid_y-1.6), d, 3.2, facecolor=OK["blue"],
                 edgecolor="white", lw=0.6, zorder=4))
    ly = mid_y - 2.8 - (2.6 if i % 2 else 0)
    txt(x-d/2, ly, name, fs=6.6, ha="center", va="top", color=OK["blue"])
    x -= d
# legend for force plot
ax.add_patch(Rectangle((lx+4, lyb+3.2), 3, 2.2, facecolor=OK["verm"], edgecolor="none", zorder=4))
txt(lx+8, lyb+4.3, "increases risk", fs=7, ha="left", va="center", color="#333")
ax.add_patch(Rectangle((lx+34, lyb+3.2), 3, 2.2, facecolor=OK["blue"], edgecolor="none", zorder=4))
txt(lx+38, lyb+4.3, "decreases risk", fs=7, ha="left", va="center", color="#333")

# ---------------------------------------------------------------- arrows between stages
arrow((ix+iw, 54), (mx, 66))                       # inputs -> models
arrow((mx+mw/2, 54), (mx+mw/2, 50))                # models -> shap
arrow((mx+mw, 40), (gx, 62), rad=0.12)             # shap -> global
arrow((mx+mw, 34), (lx, 26), rad=-0.12)            # shap -> local

# ---------------------------------------------------------------- footnote
txt(80, 3.2, "Schematic of the SHAP interpretation workflow; the global ranking and local "
    "contributions shown are illustrative, not model outputs.",
    fs=8, ha="center", va="center", style="italic", color="#777")

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Figure_10.png")
plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.1, facecolor="white")
print("written:", out)
