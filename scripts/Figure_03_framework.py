#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure: Integrated multi-hazard risk and cascading-impact framework
(earthquake, tsunami, strong wind) for the Marmara basin / Istanbul.

Pure-matplotlib schematic: rounded boxes, arrows and text only
(no icons, clip-art or raster pictures). Colour-blind-safe Okabe-Ito palette.

Run from:
<project root>
Output: Figure_03.png (300 dpi) written next to this script.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ----------------------------------------------------------------------
# Style
# ----------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none",
})

# Okabe-Ito (colour-blind-safe)
OKB = {
    "orange":  "#E69F00",
    "skyblue": "#56B4E9",
    "green":   "#009E73",
    "yellow":  "#F0E442",
    "blue":    "#0072B2",
    "verm":    "#D55E00",
    "purple":  "#CC79A7",
    "grey":    "#4D4D4D",
}

def tint(hexc, f):
    """Blend colour hexc with white by fraction f (0=colour,1=white)."""
    h = hexc.lstrip("#")
    r, g, b = (int(h[i:i+2], 16) for i in (0, 2, 4))
    r = int(r + (255 - r) * f); g = int(g + (255 - g) * f); b = int(b + (255 - b) * f)
    return f"#{r:02x}{g:02x}{b:02x}"

ARROW = "#3A3A3A"

# ----------------------------------------------------------------------
# Canvas
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(15, 9))
ax.set_xlim(0, 160)
ax.set_ylim(0, 108)
ax.axis("off")

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def panel(x, y, w, h, title, items, head_c, body_c,
          head_h=7.0, title_fs=12, item_fs=9.2, title_color="white"):
    """Rounded panel with a coloured header strip and bulleted body items."""
    r = "round,pad=0,rounding_size=1.6"
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=r, mutation_aspect=0.62,
                                lw=1.6, edgecolor=head_c, facecolor=body_c, zorder=2))
    ax.add_patch(FancyBboxPatch((x, y + h - head_h), w, head_h, boxstyle=r,
                                mutation_aspect=0.62, lw=0, facecolor=head_c, zorder=3))
    ax.text(x + w / 2, y + h - head_h / 2, title, ha="center", va="center",
            color=title_color, fontsize=title_fs, fontweight="bold", zorder=4)
    top = y + h - head_h - 2.6
    bot = y + 2.6
    n = len(items)
    for i, it in enumerate(items):
        yy = top - (i + 0.5) * (top - bot) / n
        ax.text(x + 2.8, yy, it, ha="left", va="center", fontsize=item_fs,
                color="#1a1a1a", zorder=4)

def node(x, y, w, h, text, face, edge, fs=11.5, fc_text="white", bold=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.6",
                                mutation_aspect=0.62, lw=1.8, edgecolor=edge,
                                facecolor=face, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=fc_text, fontweight="bold" if bold else "normal", zorder=4)

def arrow(p1, p2, rad=0.0, lw=1.8, color=ARROW, ms=16):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=ms,
                                 lw=lw, color=color, zorder=1,
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=1, shrinkB=1))

def coup_label(x, y, text):
    ax.text(x, y, text, ha="center", va="center", fontsize=7.6, color="#333333",
            zorder=5, linespacing=1.05,
            bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#bdbdbd", lw=0.6))

# ----------------------------------------------------------------------
# Title
# ----------------------------------------------------------------------
ax.text(80, 105.2,
        "Integrated multi-hazard risk and cascading-impact framework",
        ha="center", va="center", fontsize=16.5, fontweight="bold", color="#111111")
ax.text(80, 101.3,
        "Earthquake  ·  Tsunami  ·  Strong wind — Marmara basin / Istanbul",
        ha="center", va="center", fontsize=11.5, color="#555555")

# ----------------------------------------------------------------------
# Hazard panels
# ----------------------------------------------------------------------
py, ph = 66, 30
panel(3, py, 46, ph, "EARTHQUAKE",
      ["•  Ground shaking",
       "•  Seismic moment deficit (NAFZ locked)",
       "•  Soft-soil site amplification (V$_{s30}$)",
       "•  PGA maps (475-yr return period)"],
      OKB["verm"], tint(OKB["verm"], 0.86))

panel(57, py, 46, ph, "TSUNAMI",
      ["•  Seafloor displacement (Okada model)",
       "•  Nonlinear shallow-water propagation",
       "•  Near-field arrival (3–6 min)",
       "•  Coastal inundation depth (≤ 3.8 m)"],
      OKB["blue"], tint(OKB["blue"], 0.86))

panel(111, py, 46, ph, "STRONG WIND",
      ["•  ERA5 extreme-value analysis",
       "•  Topographic & urban-canyon channelling",
       "•  Gumbel return-period gust speeds"],
      OKB["green"], tint(OKB["green"], 0.86))

# inter-hazard couplings (two one-way arrows per gap)
arrow((49, 88), (57, 88), rad=0.0)                       # EQ -> TS (top)
coup_label(53, 93.2, "Submarine\nfault rupture")
arrow((57, 72), (49, 72), rad=0.0)                       # TS -> EQ (bottom)
coup_label(53, 77.0, "Coastal infra.\nfailure (liquef.)")

arrow((103, 88), (111, 88), rad=0.0)                     # TS -> WD (top)
coup_label(107, 93.2, "Post-EQ fire\n(Poyraz winds)")
arrow((111, 72), (103, 72), rad=0.0)                     # WD -> TS (bottom)
coup_label(107, 77.0, "Wind-driven\nsurge amplif.")

# ----------------------------------------------------------------------
# Vulnerability + exposure layer
# ----------------------------------------------------------------------
node(3, 49, 44, 10.5, "Structural vulnerability\n(pre-1975 building stock)",
     tint(OKB["orange"], 0.55), OKB["orange"], fs=10, fc_text="#1a1a1a")
node(113, 49, 44, 10.5, "Structural vulnerability\n(pre-1975 building stock)",
     tint(OKB["orange"], 0.55), OKB["orange"], fs=10, fc_text="#1a1a1a")
node(52, 47.5, 56, 12.5, "URBAN SYSTEMS EXPOSURE",
     OKB["grey"], "#2b2b2b", fs=13, fc_text="white")

# hazards -> layer
arrow((26, py), (26, 59.5))                              # EQ -> left vuln
arrow((134, py), (134, 59.5))                            # WD -> right vuln
arrow((80, py), (80, 60.2))                              # TS -> exposure

# vuln -> exposure (inward)
arrow((47, 54.3), (52, 54.3))
arrow((113, 54.3), (108, 54.3))
ax.text(80, 63.2,
        "Bayesian fragility updating  ·  ML models  (CNN, LSTM, GNN, Transformer, XGBoost, RF)",
        ha="center", va="center", fontsize=8.4, style="italic", color="#333333", zorder=6,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", lw=0.6))

# ----------------------------------------------------------------------
# Outcomes layer
# ----------------------------------------------------------------------
oy, oh = 6, 28
panel(3, oy, 74, oh, "CASCADING & COMPOUND IMPACTS",
      ["•  Lifeline failures (Bosphorus crossings, rail network)",
       "•  Post-earthquake fire conflagrations",
       "•  Simultaneous coastal & inland road closures",
       "•  Non-additive loss amplification (Gaussian copula)"],
      OKB["purple"], tint(OKB["purple"], 0.85), title_fs=12, item_fs=9.6)

panel(83, oy, 74, oh, "RISK MITIGATION & RESILIENCE PLANNING",
      ["•  Retrofit prioritisation (non-ductile RC & unreinforced masonry)",
       "•  Evidence-based urban planning (tsunami run-up zones)",
       "•  Optimal emergency response & evacuation routing"],
      OKB["skyblue"], tint(OKB["skyblue"], 0.85), title_fs=12, item_fs=9.6)

# exposure -> outcomes
arrow((70, 47.5), (45, oy + oh), rad=0.12)
arrow((90, 47.5), (115, oy + oh), rad=-0.12)

# ----------------------------------------------------------------------
# Save
# ----------------------------------------------------------------------
outdir = os.path.dirname(os.path.abspath(__file__))
outpath = os.path.join(outdir, "Figure_03.png")
plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
fig.savefig(outpath, dpi=300, bbox_inches="tight", pad_inches=0.10, facecolor="white")
print("written:", outpath)
