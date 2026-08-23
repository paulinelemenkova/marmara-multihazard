#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bring every figure PNG under a size cap (default 1.5 MB), keeping a reserve
copy of the originals.

  1. Copies each PNG in figures/ into figures_ORIGINAL/ (true originals are
     preserved; a file already backed up is never overwritten, so re-runs are
     safe).
  2. Any PNG over the cap is compressed IN PLACE:
        a) re-save with optimisation,
        b) if still too big, downscale (LANCZOS) in 10% steps — physical size
           is preserved by lowering the stored DPI — down to a floor,
        c) if still too big, reduce to a 256-colour adaptive palette (dithered).
  Files already under the cap are left untouched (but are still backed up).

Run:  python3 compress_figures.py
Requires: Pillow  (already in Anaconda)
"""

import os, shutil
from PIL import Image

BASE      = os.environ.get(
    "PROJECT_ROOT",
    os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")),
)
FIG_DIR   = os.path.join(BASE, "figures")
BACKUP    = os.path.join(BASE, "figures_ORIGINAL")
TARGET    = int(1.5 * 1024 * 1024)     # 1.5 MB
MIN_LONG  = 2000                        # px floor for the long edge (~150 dpi at 13")

def human(n): return f"{n/1024/1024:.2f} MB"

def flatten(path):
    im = Image.open(path)
    dpi = im.info.get("dpi", (300, 300))
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im).convert("RGB")
    else:
        im = im.convert("RGB")
    return im, dpi

def compress(path, target):
    full, dpi0 = flatten(path)
    w0 = full.width
    def save(img):
        scale = img.width / w0
        d = (max(int(dpi0[0]*scale), 72), max(int(dpi0[1]*scale), 72))
        img.save(path, format="PNG", optimize=True, dpi=d)
        return os.path.getsize(path)

    if save(full) <= target:
        return "optimised"
    cur = full
    while max(cur.size) > MIN_LONG:
        cur = cur.resize((max(int(cur.width*0.9), 1), max(int(cur.height*0.9), 1)), Image.LANCZOS)
        if save(cur) <= target:
            return f"downscaled to {cur.width}x{cur.height}"
    # last resort: 256-colour adaptive palette (dithered)
    pal = cur.convert("P", palette=Image.ADAPTIVE, colors=256)
    scale = cur.width / w0
    d = (max(int(dpi0[0]*scale), 72), max(int(dpi0[1]*scale), 72))
    pal.save(path, format="PNG", optimize=True, dpi=d)
    return "palette-256" if os.path.getsize(path) <= target else "STILL OVER CAP"

def main():
    os.makedirs(BACKUP, exist_ok=True)
    pngs = sorted(f for f in os.listdir(FIG_DIR)
                  if f.lower().endswith(".png") and os.path.isfile(os.path.join(FIG_DIR, f)))
    print(f"{len(pngs)} PNGs in {FIG_DIR}\n")
    for f in pngs:
        src = os.path.join(FIG_DIR, f)
        bak = os.path.join(BACKUP, f)
        if not os.path.exists(bak):                 # preserve TRUE originals only
            shutil.copy2(src, bak)
        before = os.path.getsize(src)
        if before <= TARGET:
            print(f"  keep       {f:18s} {human(before)}")
            continue
        how = compress(src, TARGET)
        print(f"  compressed {f:18s} {human(before)} -> {human(os.path.getsize(src))}  [{how}]")
    print(f"\nOriginals backed up in: {BACKUP}")

if __name__ == "__main__":
    main()
