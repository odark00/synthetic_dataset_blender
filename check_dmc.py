"""Verify that the Data Matrix codes in the RENDERED images really carry the
payload recorded in the label JSON.

    python3 check_dmc.py [gray_dir] [labels_dir] [n_images]

For every parcel with a "dmc" entry it takes the four recorded corners,
perspective-rectifies that quad out of the grayscale render, samples the module
grid, and compares it with a fresh ECC200 encoding of the recorded payload.
This checks the whole chain - encoder -> texture -> UVs -> render -> label
geometry - not just the encoder.

A code only survives if it is big enough on screen (roughly >= 2 px per
module), so small/steeply-viewed codes are reported as "too small" rather than
as failures.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import cv2

import dmc_ecc200

HERE = Path(__file__).resolve().parent
gray = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "gray_images"
lab = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "labels"
limit = int(sys.argv[3]) if len(sys.argv) > 3 else 20

OVERSAMPLE = 8          # pixels per module in the rectified patch


def _sample(img, src, size, quiet, dx, dy, scale):
    n = size + 2 * quiet
    side = n * OVERSAMPLE
    # corners are (u,v) = (0,0) (1,0) (1,1) (0,1); v runs bottom-up in UV
    # space, so v=1 must land on the TOP row of the rectified patch
    dst = np.float32([[-dx * OVERSAMPLE, side + dy * OVERSAMPLE],
                      [side - dx * OVERSAMPLE, side + dy * OVERSAMPLE],
                      [side - dx * OVERSAMPLE, dy * OVERSAMPLE],
                      [-dx * OVERSAMPLE, dy * OVERSAMPLE]])
    c = side * 0.5
    dst = (dst - c) * scale + c
    M = cv2.getPerspectiveTransform(src, dst)
    patch = cv2.warpPerspective(img, M, (side, side))
    # Shading across a code is far from uniform (falloff, specular), so divide
    # out the illumination first. The kernel has to be much wider than a module
    # (~10) or it flattens away the code itself.
    k = max(3, int(OVERSAMPLE * 10) | 1)
    bg = cv2.GaussianBlur(patch, (k, k), 0)
    flat = patch / np.maximum(bg, 1e-3)
    # sample the middle of each module only - the borders bleed into their
    # neighbours through the renderer's antialiasing
    blk = flat.reshape(n, OVERSAMPLE, n, OVERSAMPLE)
    lo = OVERSAMPLE // 4
    hi = OVERSAMPLE - lo
    cells = blk[:, lo:hi, :, lo:hi].mean(axis=(1, 3))
    u8 = np.clip((cells - cells.min())
                 / max(float(cells.max() - cells.min()), 1e-6) * 255,
                 0, 255).astype(np.uint8)
    thr, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (u8 < thr).astype(int)[quiet:quiet + size, quiet:quiet + size]


def check(img, corners, text):
    """Best module accuracy over a small sub-module alignment search.

    A real decoder locks onto the finder pattern and clock track to derive its
    sampling grid; we only have the four outline corners, so we search a little
    offset/scale window instead. Without it, a fraction-of-a-module
    misregistration alone drops the score to ~0.8.
    """
    size, grid = dmc_ecc200.encode(text)
    ref = np.array(grid)
    src = np.float32(corners)
    best = 0.0
    for quiet in (1, 2):
        steps = [i / 10.0 for i in range(-4, 5)]
        for dx in steps:
            for dy in steps:
                for scale in (0.97, 1.0, 1.03):
                    got = _sample(img, src, size, quiet, dx, dy, scale)
                    if got.shape != ref.shape:
                        continue
                    m = float((got == ref).mean())
                    if m > best:
                        best = m
                        if best == 1.0:
                            return True, quiet, 1.0
    return False, None, best


MIN_PX_PER_MODULE = 3.5     # below this the render simply cannot resolve modules

tot = ok = small = 0
worst = []
ppm_all = []
acc = []
for jf in sorted(lab.glob("img_*.json"))[:limit]:
    meta = json.loads(jf.read_text())
    f = gray / (jf.stem + ".png")
    if not f.exists():
        continue
    img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE).astype(np.float32)
    for p in meta["parcels"]:
        d = p.get("dmc")
        if not d:
            continue
        pts = d.get("corners_2d")
        if not pts or any(q is None for q in pts):
            continue
        tot += 1
        e0 = math.dist(pts[0], pts[1])
        e1 = math.dist(pts[1], pts[2])
        ppm = min(e0, e1) / d["symbol_modules"]
        ppm_all.append(ppm)
        if ppm < MIN_PX_PER_MODULE:
            small += 1
            continue
        good, quiet, m = check(img, pts, d["text"])
        acc.append(float(m) if not good else 1.0)
        if good:
            ok += 1
        elif float(m) < 0.99:
            worst.append((jf.stem, d["text"], f"match={float(m):.3f}",
                          f"px/module={ppm:.1f}"))

print(f"codes with corners in frame : {tot}")
if ppm_all:
    ppm_all.sort()
    print(f"  px per module   min/median/max: {ppm_all[0]:.1f} / "
          f"{ppm_all[len(ppm_all) // 2]:.1f} / {ppm_all[-1]:.1f}")
print(f"  below {MIN_PX_PER_MODULE} px/module (unresolvable): {small}")
checked = tot - small
print(f"  checked                   : {checked}")
if checked:
    acc.sort()
    print(f"  module accuracy min/median: {acc[0]:.3f} / {acc[len(acc) // 2]:.3f}")
    print(f"  exact (100% of modules)   : {ok}/{checked}")
    print(f"  >= 99% of modules         : "
          f"{sum(1 for a in acc if a >= 0.99)}/{checked}")
for w in worst[:10]:
    print("   MISMATCH", w)
