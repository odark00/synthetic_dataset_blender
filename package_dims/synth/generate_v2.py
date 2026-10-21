"""Synthetic dataset v2 for overhead package dimensioning (camera cam_000CDF0AD90F).

Same output schema and camera framing as generate.py (v1), more realistic scenes:
  * z-buffered renderer (per-pixel plane depth) instead of the painter's algorithm, so forks
    under a pallet deck, gaps between cartons and see-through cage mesh occlude correctly
  * one light direction per scene -> consistent Lambert shading of every face, darkening of
    side faces towards the floor, directional + contact shadows on the floor
  * compositional geometry: EUR-style block pallets (deck boards, cross boards, blocks),
    loads built from individual cartons (uniform / irregular stacks, mixed cartons, single big
    box, plywood crate, panel, strapped machine, small item), gitterbox cages with posts
  * pallet truck engaged under most pallets (forks inside the pallet, heel, tilted drawbar
    with loop handle), as in the real frames
  * carton "SKUs" with brand text / oval logos / small print, flap seam + tape, worn boxes;
    stretch film as an alpha overlay with edge-biased crinkle highlights; straps
  * sun patch entering from the bottom-right edge (size range of the real frames)

Labels follow v1 exactly: the unit box is the OBB of pallet + load in pallet axes, H from the
floor to the highest point; the truck is not part of the unit.

    nice -n 19 python synth/generate_v2.py --n 3000 --workers 8 --seed 0
"""
import argparse
import json
import math
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from camera import Camera, pad_square  # noqa: E402
from generate import CAM_DIR, LABEL_M, SHARD, TEX_PPM, dmc_symbol, rect_corners, smooth_noise  # noqa: E402,F401

PALLET_H = 0.144
FONTS = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_TRIPLEX,
         cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, cv2.FONT_HERSHEY_SCRIPT_COMPLEX]
# box faces as corner indices; corners 0-3 bottom (rect_corners order), 4-7 top.
# face 0 = top, 1-4 = sides (texture y runs top -> bottom), 5 = bottom
FACES = [(4, 5, 6, 7)] + [(4 + i, 4 + (i + 1) % 4, (i + 1) % 4, i) for i in range(4)] + [(3, 2, 1, 0)]


def px(m):
    return max(4, int(round(m * TEX_PPM)))


# ---------------------------------------------------------------------------------- textures
def fine(rng, h, w, sigma):
    return rng.normal(0, sigma, (h, w)).astype(np.float32)


def grain_noise(rng, h, w, along_px, across_px):
    """Noise elongated along texture x (wood grain, film streaks)."""
    small = rng.uniform(-1, 1, (max(2, h // max(1, across_px) + 2), max(2, w // max(1, along_px) + 2)))
    return cv2.resize(small.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)


def pseudo_word(rng):
    c, v = "bcdfgjklmnprstvz", "aeiouy"
    s = "".join(rng.choice(list(c)) + rng.choice(list(v)) for _ in range(rng.integers(2, 4)))
    s = s + rng.choice(list(c)) if rng.random() < 0.4 else s
    return s.upper() if rng.random() < 0.4 else s.capitalize()


def draw_text(t, text, font, tone, box, thick_rel=0.08, italic=False, flip=False):
    """Fit `text` into box=(x, y, w, h) of texture t, blend with an antialiased mask."""
    x, y, w, h = [int(v) for v in box]
    if w < 8 or h < 5:
        return
    f = font | (cv2.FONT_ITALIC if italic else 0)
    (tw, th), base = cv2.getTextSize(text, f, 1.0, 2)
    scale = 4.0
    canvas = np.zeros((int((th + base) * scale) + 8, int(tw * scale) + 8), np.uint8)
    cv2.putText(canvas, text, (4, int(th * scale) + 4), f, scale, 255, max(1, int(th * scale * thick_rel)), cv2.LINE_AA)
    r = min(w / canvas.shape[1], h / canvas.shape[0])
    m = cv2.resize(canvas, (max(1, int(canvas.shape[1] * r)), max(1, int(canvas.shape[0] * r))),
                   interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    if flip:
        m = m[::-1, ::-1]
    mh, mw = m.shape
    ox, oy = x + (w - mw) // 2, y + (h - mh) // 2
    ox, oy = max(0, ox), max(0, oy)
    mh, mw = min(mh, t.shape[0] - oy), min(mw, t.shape[1] - ox)
    if mh <= 0 or mw <= 0:
        return
    reg = t[oy:oy + mh, ox:ox + mw]
    reg[:] = reg * (1 - m[:mh, :mw]) + tone * m[:mh, :mw]


def make_sku(rng):
    """Appearance of one carton type; all cartons of a SKU look alike (as on real pallets)."""
    dark = rng.random() < 0.3
    base = rng.uniform(30, 70) if dark else rng.uniform(85, 165)
    ink = rng.uniform(170, 235) if dark else rng.uniform(20, 70)
    if rng.random() < 0.15:  # light ink on mid cardboard (white labels / logos)
        ink = rng.uniform(190, 240)
    return {"base": base, "ink": ink, "dark": dark,
            "style": rng.choice(["plain", "oval", "text", "band", "text"], p=[0.25, 0.25, 0.25, 0.1, 0.15]),
            "word": pseudo_word(rng), "font": int(rng.choice(FONTS)), "italic": bool(rng.random() < 0.3),
            "small_print": rng.random() < 0.5, "flaps": rng.random() < 0.75, "tape": rng.random() < 0.6,
            "tape_tone": float(np.clip(base + rng.choice([-1, 1]) * rng.uniform(15, 50), 10, 235)),
            "worn": rng.random() < 0.15, "mottle": rng.uniform(1.5, 4), "grain": rng.uniform(1.0, 3.0),
            "side_logo": rng.random() < 0.7, "logo_m": rng.uniform(0.1, 0.35)}


def carton_face(rng, sku, w, h, top):
    t = np.full((h, w), sku["base"] + rng.normal(0, 3), np.float32)
    t += smooth_noise(rng, h, w, px(0.6)) * sku["mottle"] + fine(rng, h, w, sku["grain"])
    flip = rng.random() < 0.5
    if top or sku["side_logo"]:
        st = sku["style"]
        # print sized in metres (logos are ~0.1-0.35 m), centred on the face
        lw_, lh_ = min(0.7 * w, px(sku["logo_m"])), min(0.4 * h, px(sku["logo_m"] * 0.4))
        bx = ((w - lw_) / 2, (h - lh_) / 2, lw_, lh_)
        if st == "oval":
            c, ax = (w // 2, h // 2), (max(3, int(lw_ * 0.55)), max(2, int(lh_ * 0.75)))
            cv2.ellipse(t, c, ax, 0, 0, 360, float(sku["ink"]), -1, cv2.LINE_AA)
            draw_text(t, sku["word"], sku["font"], sku["base"], (c[0] - ax[0] * 0.8, c[1] - ax[1] * 0.7,
                      ax[0] * 1.6, ax[1] * 1.4), italic=sku["italic"], flip=flip)
        elif st == "text":
            draw_text(t, sku["word"], sku["font"], sku["ink"], bx, italic=sku["italic"], flip=flip)
        elif st == "band":
            y0, bh = int(h * rng.uniform(0.2, 0.6)), max(3, min(int(h * 0.3), px(rng.uniform(0.04, 0.12))))
            t[y0:y0 + bh] = sku["ink"]
            draw_text(t, sku["word"], sku["font"], sku["base"], ((w - lw_) / 2, y0 + bh * 0.1, lw_, bh * 0.8), flip=flip)
    if sku["small_print"]:
        for _ in range(rng.integers(1, 4)):
            lh = max(4, int(px(0.012) * rng.uniform(0.8, 1.6)))
            y = int(rng.uniform(0.05, 0.9) * h)
            draw_text(t, " ".join(pseudo_word(rng).upper() for _ in range(2)), cv2.FONT_HERSHEY_SIMPLEX,
                      sku["ink"] * 0.6 + sku["base"] * 0.4, (w * rng.uniform(0.05, 0.4), y, w * 0.5, lh), flip=flip)
    if top and sku["flaps"]:  # RSC carton: flap seam along the long axis, often taped
        long_x = w >= h
        if sku["tape"]:
            tw = px(0.05)
            if long_x:
                t[max(0, h // 2 - tw // 2):h // 2 + tw // 2] = sku["tape_tone"]
            else:
                t[:, max(0, w // 2 - tw // 2):w // 2 + tw // 2] = sku["tape_tone"]
        if long_x:
            t[h // 2:h // 2 + 1] *= 0.55
        else:
            t[:, w // 2:w // 2 + 1] *= 0.55
    if sku["worn"]:
        dirt = cv2.GaussianBlur((rng.random((h, w)) < 0.004).astype(np.float32), (0, 0), rng.uniform(1, 4))
        t -= dirt / (dirt.max() + 1e-6) * rng.uniform(15, 40)
        t += smooth_noise(rng, h, w, px(0.1)) * rng.uniform(5, 12)
    e = max(1, w // 150)
    t[:e] *= 0.7
    t[-e:] *= 0.7
    t[:, :e] *= 0.7
    t[:, -e:] *= 0.7
    return np.clip(t, 0, 255)


def wood(rng, w, h, tone):
    t = tone + grain_noise(rng, h, w, px(0.25), max(2, px(0.02))) * rng.uniform(8, 20) + fine(rng, h, w, rng.uniform(2, 5))
    for _ in range(rng.poisson(0.6)):  # knots
        cv2.ellipse(t, (int(rng.integers(0, w)), int(rng.integers(0, h))), (px(0.012), px(0.008)), 0, 0, 360,
                    float(tone * 0.55), -1, cv2.LINE_AA)
    if rng.random() < 0.3:  # dirt
        t -= np.clip(smooth_noise(rng, h, w, px(0.08)), 0, 1) * rng.uniform(15, 50)
    e = max(1, min(h, w) // 12)
    t[:e] *= 0.8
    t[-e:] *= 0.8
    return np.clip(t, 0, 255)


def plain(rng, w, h, tone, noise=3.0, mottle=4.0):
    return np.clip(tone + smooth_noise(rng, h, w, px(0.3)) * mottle + fine(rng, h, w, noise), 0, 255)


def plywood(rng, w, h):
    tone = rng.uniform(110, 185)
    t = tone + grain_noise(rng, h, w, px(0.5), px(0.05)) * rng.uniform(6, 14) + fine(rng, h, w, 3)
    if rng.random() < 0.6:  # stencil text
        draw_text(t, pseudo_word(rng).upper(), cv2.FONT_HERSHEY_SIMPLEX, tone * 0.3, (w * 0.2, h * 0.35, w * 0.6, h * 0.3))
    return np.clip(t, 0, 255)


def machine_face(rng, w, h, tone):
    t = plain(rng, w, h, tone, 2, 3)
    for _ in range(rng.integers(1, 4)):  # light panels / windows / foam
        pw, ph = int(w * rng.uniform(0.2, 0.6)), int(h * rng.uniform(0.2, 0.6))
        x, y = rng.integers(0, max(1, w - pw)), rng.integers(0, max(1, h - ph))
        t[y:y + ph, x:x + pw] = rng.uniform(130, 215)
        cv2.rectangle(t, (int(x), int(y)), (int(x + pw), int(y + ph)), float(tone * 0.6), max(1, px(0.01)))
    return np.clip(t, 0, 255)


def film_tex(rng, w, h, top):
    """Stretch film: mostly transparent, crinkle highlights biased to the edges; returns (tone, alpha)."""
    tone = np.full((h, w), rng.uniform(190, 245), np.float32)
    alpha = np.full((h, w), rng.uniform(0.02, 0.10), np.float32)
    area = (w / TEX_PPM) * (h / TEX_PPM)
    for _ in range(int(area * rng.uniform(8, 35)) + 2):
        if rng.random() < 0.85:  # near an edge
            side = rng.integers(0, 4)
            d = abs(rng.normal(0, 0.06))
            x = rng.uniform(0, w) if side < 2 else (d * w if side == 2 else (1 - d) * w)
            y = rng.uniform(0, h) if side >= 2 else (d * h if side == 0 else (1 - d) * h)
        else:
            x, y = rng.uniform(0, w), rng.uniform(0, h)
        ang, ln = rng.uniform(0, math.pi), px(rng.uniform(0.02, 0.25))
        k = int(rng.integers(3, 6))
        steps = np.cumsum(rng.normal(0, 0.5, k)) + ang
        pts = np.stack([x + np.cumsum(np.cos(steps)) * ln / k, y + np.cumsum(np.sin(steps)) * ln / k], 1)
        cv2.polylines(alpha, [pts.astype(np.int32)], False, float(rng.uniform(0.12, 0.45)), 1,
                      cv2.LINE_AA)
    if not top and rng.random() < 0.5:  # rope of bunched film at the bottom of the wrap
        bh = px(rng.uniform(0.03, 0.09))
        bh = min(bh, h)
        alpha[h - bh:] = np.maximum(alpha[h - bh:], rng.uniform(0.3, 0.6) + grain_noise(rng, bh, w, px(0.1), 2) * 0.15)
    if rng.random() < 0.5:  # broad glare
        g = np.clip(smooth_noise(rng, h, w, px(0.2)) - 0.5, 0, 1) * rng.uniform(0.2, 0.6)
        alpha = np.maximum(alpha, g)
    return tone, np.clip(cv2.GaussianBlur(alpha, (0, 0), 0.7), 0, 1)


def mesh_tex(rng, w, h, step_m, tone):
    t = np.full((h, w), tone, np.float32) + fine(rng, h, w, 6)
    a = np.zeros((h, w), np.float32)
    step = max(4, px(step_m))
    th = max(1, px(0.005))
    for x in range(0, w, step):
        cv2.line(a, (x, 0), (x, h), 1.0, th)
    for y in range(0, h, step):
        cv2.line(a, (0, y), (w, y), 1.0, th)
    return t, a


def place_dmc(rng, t):
    """White shipping label with a DMC pasted into a top texture -> symbol corners (texture px) or None."""
    h, w = t.shape
    s = px(LABEL_M)
    pad = px(rng.uniform(0.01, 0.04))
    lw, lh = s + 2 * pad + px(rng.uniform(0, 0.10)), s + 2 * pad + px(rng.uniform(0, 0.06))
    if lw + 4 >= w or lh + 4 >= h:
        return None
    patch = np.full((lh, lw), rng.uniform(185, 240), np.float32) + fine(rng, lh, lw, 2)
    patch[pad:pad + s, pad:pad + s] = np.minimum(dmc_symbol(rng, s), patch[0, 0])
    if lw - s - 2 * pad > 10:  # human-readable text beside the code
        for k in range(3):
            draw_text(patch, pseudo_word(rng).upper(), cv2.FONT_HERSHEY_SIMPLEX, 40,
                      (s + 1.5 * pad, pad + k * s / 3, lw - s - 2.5 * pad, s / 4))
    ang = rng.uniform(0, 360) if rng.random() < 0.5 else rng.choice([0, 90, 180, 270]) + rng.normal(0, 3)
    hx = hy = math.hypot(lw, lh) / 2 + 1
    if 2 * hx + 4 >= min(w, h):
        ang, hx, hy = 0, lw / 2, lh / 2
    cx, cy = rng.uniform(hx + 2, w - hx - 2), rng.uniform(hy + 2, h - hy - 2)
    M = cv2.getRotationMatrix2D((lw / 2, lh / 2), ang, 1.0)
    M[:, 2] += (cx - lw / 2, cy - lh / 2)
    alpha = cv2.warpAffine(np.ones_like(patch), M, (w, h))
    t[:] = t * (1 - alpha) + cv2.warpAffine(patch, M, (w, h)) * alpha
    sym = np.array([[pad, pad], [pad + s, pad], [pad + s, pad + s], [pad, pad + s]], np.float64)
    return (sym @ M[:, :2].T + M[:, 2]).tolist()


# ---------------------------------------------------------------------------------- geometry
def prism(cx, cy, l, w, z0, z1, yaw=0.0):
    """Vertical box in (x, y, height) coordinates -> (8, 3) corners, bottom 0-3 / top 4-7."""
    xy = rect_corners(cx, cy, l, w, yaw)
    return np.concatenate([np.c_[xy, np.full(4, z0)], np.c_[xy, np.full(4, z1)]])


def oriented_box(center, a, b, c):
    """Box from centre and half-axis vectors (a: length, b: width, c: 'up') -> (8, 3)."""
    s = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    return np.array([center + i * a + j * b - c for i, j in s] + [center + i * a + j * b + c for i, j in s])


def part(P, tex, ao=0.0, cull=False, top_tex=None, shade=1.0, mat="matte"):
    """mat: material name, only used by the path-traced renderer (generate_v3.py)."""
    return {"P": np.asarray(P, np.float64), "tex": tex, "ao": ao, "cull": cull, "top_tex": top_tex, "shade": shade,
            "mat": mat}


class Light:
    def __init__(self, rng):
        self.L = np.array([rng.normal(0, 0.35), rng.normal(0, 0.35), -1.0])  # camera frame, towards the light
        self.L /= np.linalg.norm(self.L)
        self.amb = rng.uniform(0.45, 0.8)
        self.top = self.amb + (1 - self.amb) * max(0.0, -self.L[2])
        self.floor_dir = -self.L[:2] / -self.L[2]  # shadow offset per metre of height (floor plane)

    def shade(self, n):
        return (self.amb + (1 - self.amb) * max(0.0, float(n @ self.L))) / self.top


class ZRenderer:
    """Planar textured quads into a pinhole canvas with a z-buffer (camera depth Z)."""

    def __init__(self, cam):
        self.cam = cam
        W, H = cam.size
        self.img = np.zeros((H, W), np.float32)
        self.alpha = np.zeros((H, W), np.float32)
        self.z = np.full((H, W), cam.mount_h + 1.0, np.float32)
        self.shadow = np.zeros((H, W), np.float32)
        self.rx = ((np.arange(W) - cam.cx) / cam.fx).astype(np.float32)
        self.ry = ((np.arange(H) - cam.cy) / cam.fy).astype(np.float32)

    def w2c(self, P):
        P = np.asarray(P, np.float64)
        return np.c_[P[:, :2], self.cam.mount_h - P[:, 2]]

    def quad(self, q, tex, shade=1.0, alpha=None, blend=False):
        """q: (4,3) camera frame; texture corners (0,0),(w,0),(w,h),(0,h) map to q[0..3]."""
        if (q[:, 2] < 0.2).any():
            return
        dst = self.cam.project(q).astype(np.float32)
        W, H = self.cam.size
        x0, y0 = np.floor(dst.min(0)).astype(int)
        x1, y1 = np.ceil(dst.max(0)).astype(int) + 1
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
        if x1 <= x0 or y1 <= y0:
            return
        th, tw = tex.shape
        src = np.float32([[0, 0], [tw, 0], [tw, th], [0, th]])
        local = (dst - (x0, y0)).astype(np.float32)
        Hm = cv2.getPerspectiveTransform(src, local)
        size = (x1 - x0, y1 - y0)
        roi = cv2.warpPerspective(tex, Hm, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        m = np.zeros((size[1], size[0]), np.uint8)
        cv2.fillConvexPoly(m, np.round(local * 16).astype(np.int32), 1, cv2.LINE_8, 4)
        m = m.astype(bool)
        n = np.cross(q[1] - q[0], q[3] - q[0])
        den = n[0] * self.rx[None, x0:x1] + n[1] * self.ry[y0:y1, None] + n[2]
        with np.errstate(divide="ignore", invalid="ignore"):
            Z = float(n @ q[0]) / den
        zb, im = self.z[y0:y1, x0:x1], self.img[y0:y1, x0:x1]
        if blend:  # semi-transparent overlay (film): in front of or on the surface, no depth write
            a = cv2.warpPerspective(alpha, Hm, size, flags=cv2.INTER_LINEAR)
            vis = m & (Z < zb + 0.03) & (Z > 0)
            im[vis] = im[vis] * (1 - a[vis]) + roi[vis] * shade * a[vis]
            return
        vis = m & (Z < zb - 1e-4) & (Z > 0)
        if alpha is not None:
            vis &= cv2.warpPerspective(alpha, Hm, size, flags=cv2.INTER_NEAREST) > 0.5
        im[vis] = roi[vis] * shade
        zb[vis] = Z[vis]
        self.alpha[y0:y1, x0:x1][vis] = 1

    def box(self, P, tex_fn, light, ao=0.0, top_tex=None, skip=None, shade=1.0):
        """P: (8,3) in (x, y, height); tex_fn(face, w_px, h_px) -> tex or (tex, alpha)."""
        Pc = self.w2c(P)
        c = Pc.mean(0)
        for fi, idx in enumerate(FACES):
            q = Pc[list(idx)]
            n = np.cross(q[1] - q[0], q[3] - q[0])
            nn = np.linalg.norm(n)
            if nn < 1e-12:
                continue
            n /= nn
            fc = q.mean(0)
            if n @ (fc - c) < 0:
                n = -n
            if n @ fc >= 0:  # back face
                continue
            if skip is not None and skip(P[list(idx)].mean(0), fi):
                continue
            w_px, h_px = px(np.linalg.norm(q[1] - q[0])), px(np.linalg.norm(q[3] - q[0]))
            t = top_tex if (fi == 0 and top_tex is not None) else tex_fn(fi, w_px, h_px)
            a = None
            if isinstance(t, tuple):
                t, a = t
            if 1 <= fi <= 4 and ao > 0:
                t = t * (1 - ao * np.linspace(0, 1, t.shape[0], dtype=np.float32) ** 2)[:, None]
            self.quad(q, t.astype(np.float32), light.shade(n) * shade, a)

    def floor_shadow(self, foot_xy, H, light, rng):
        """Directional shadow of a unit (footprint swept by its height) + contact darkening."""
        pts = np.concatenate([foot_xy, foot_xy + light.floor_dir * H])
        hull = cv2.convexHull(np.round(self.cam.project(self.w2c(np.c_[pts, np.zeros(len(pts))])) * 4).astype(np.int32))
        m = np.zeros_like(self.shadow)
        cv2.fillConvexPoly(m, hull, 1.0, cv2.LINE_AA, 2)
        m = cv2.GaussianBlur(m, (0, 0), rng.uniform(8, 35)) * rng.uniform(0.08, 0.35)
        f = np.zeros_like(self.shadow)
        cv2.fillConvexPoly(f, np.round(self.cam.project(self.w2c(np.c_[foot_xy, np.zeros(4)])) * 4).astype(np.int32),
                           1.0, cv2.LINE_AA, 2)
        f = cv2.GaussianBlur(cv2.dilate(f, np.ones((5, 5), np.uint8)), (0, 0), rng.uniform(3, 8)) * rng.uniform(0.15, 0.4)
        self.shadow = np.maximum(self.shadow, np.maximum(m, f))


# ---------------------------------------------------------------------------------- builders
def build_pallet(rng, L, W):
    """Block pallet in local frame (x along L): bottom boards, 9 blocks, 3 cross boards, deck boards."""
    style = rng.choice(["wood", "old", "plastic"], p=[0.55, 0.35, 0.10])
    tone = {"wood": rng.uniform(140, 200), "old": rng.uniform(80, 140), "plastic": rng.uniform(25, 70)}[style]
    if style == "plastic":
        tex = lambda f, w, h: plain(rng, w, h, tone, 2, 3)  # noqa: E731
    else:
        tex = lambda f, w, h: wood(rng, w, h, float(np.clip(tone + rng.normal(0, 12), 30, 230)))  # noqa: E731
    mat = "plastic" if style == "plastic" else "wood"
    parts = []
    xs = [-L / 2 + 0.0725, 0.0, L / 2 - 0.0725]
    ys = [-W / 2 + 0.05, 0.0, W / 2 - 0.05]
    for y, bw in zip(ys, (0.1, 0.145, 0.1)):  # bottom boards
        parts.append(part(prism(0, y, L, bw, 0.0, 0.022), tex, mat=mat))
    for x in xs:
        for y, bw in zip(ys, (0.1, 0.145, 0.1)):
            parts.append(part(prism(x, y, 0.145, bw, 0.022, 0.1), tex, mat=mat))
        parts.append(part(prism(x, 0, 0.145, W, 0.1, 0.122), tex, mat=mat))  # cross board
    n = 5 if W < 0.9 else 7
    widths = np.array([0.145 if i % 2 == 0 else 0.1 for i in range(n)])
    gap = (W - widths.sum()) / (n - 1)
    y = -W / 2
    for bw in widths:  # deck boards along L
        parts.append(part(prism(0, y + bw / 2, L, bw, 0.122, PALLET_H), tex, mat=mat))
        y += bw + gap
    return parts


def build_truck(rng, tilt=None):
    """Pallet truck in its own frame: forks along +x from the heel (x=0) to the tips (x=1.15)."""
    ft = rng.uniform(110, 215)
    body = ft * rng.uniform(0.85, 1.05) if rng.random() < 0.7 else rng.uniform(20, 60)
    dark = rng.uniform(12, 45)
    ftex = lambda f, w, h: plain(rng, w, h, ft, 3, 5)  # noqa: E731
    btex = lambda f, w, h: plain(rng, w, h, body, 3, 5)  # noqa: E731
    dtex = lambda f, w, h: plain(rng, w, h, dark, 2, 2)  # noqa: E731
    parts = [part(prism(0.575, y, 1.15, 0.16, 0.025, 0.085), ftex, ao=0.2, mat="paint") for y in (-0.19, 0.19)]
    parts.append(part(prism(-0.12, 0, 0.26, 0.56, 0.0, 0.2), btex, ao=0.3, mat="paint"))  # heel / chassis
    parts.append(part(prism(-0.13, 0, 0.16, 0.22, 0.2, 0.45), btex if rng.random() < 0.5 else dtex,
                      mat="paint"))  # pump
    for y in (-0.12, 0.12):  # steering wheels
        parts.append(part(prism(-0.2, y, 0.16, 0.06, 0.0, 0.17), dtex, mat="rubber"))
    th = rng.uniform(0.05, 1.3) if tilt is None else tilt
    d = np.array([-math.sin(th), 0.0, math.cos(th)])
    perp = np.array([math.cos(th), 0.0, math.sin(th)])
    B = np.array([-0.13, 0.0, 0.45])
    Lb = rng.uniform(0.9, 1.1)
    parts.append(part(oriented_box(B + d * Lb / 2, d * Lb / 2, np.array([0, 0.02, 0]), perp * 0.02), dtex,
                      mat="paint"))
    T = B + d * Lb
    grip = rng.uniform(150, 220) if rng.random() < 0.3 else dark
    ring = [T + d * (0.1 - 0.1 * math.cos(p)) + np.array([0, 0.16 * math.sin(p), 0]) for p in np.linspace(0, 2 * math.pi, 9)]
    for a, b in zip(ring[:-1], ring[1:]):
        v = (b - a) / 2
        side = np.cross(v, perp)
        side = side / (np.linalg.norm(side) + 1e-9) * 0.015
        parts.append(part(oriented_box((a + b) / 2, v, side, perp * 0.015),
                          lambda f, w, h: plain(rng, w, h, grip, 2, 2), mat="rubber"))
    return parts


def transform(P, yaw, cx, cy, ox=0.0, oy=0.0):
    c, s = math.cos(yaw), math.sin(yaw)
    out = P.copy()
    out[:, :2] = (P[:, :2] + (ox, oy)) @ np.array([[c, s], [-s, c]]) + (cx, cy)
    return out


def carton_part(rng, sku, cx, cy, l, w, z0, z1, yaw=0.0, cull=True):
    return part(prism(cx, cy, l, w, z0, z1, yaw), lambda f, ww, hh: carton_face(rng, sku, ww, hh, f == 0),
                ao=rng.uniform(0.05, 0.2), cull=cull, mat="cardboard")


def load_stack(rng, lx, ly, H, base, skus):
    """Grid of equal cartons, optionally with an incomplete top layer."""
    nx = max(1, round(lx / rng.uniform(0.25, 0.6)))
    ny = max(1, round(ly / rng.uniform(0.2, 0.45)))
    nz = max(1, round(H / rng.uniform(0.15, 0.45)))
    dx, dy, dz = lx / nx, ly / ny, H / nz
    ragged = rng.random() < 0.35
    parts = []
    for k in range(nz):
        layer = []
        for i in range(nx):
            for j in range(ny):
                if k == nz - 1 and ragged and rng.random() < 0.35:
                    continue
                s = rng.uniform(0.003, 0.015)
                layer.append(carton_part(rng, skus[rng.integers(len(skus))],
                                         -lx / 2 + (i + 0.5) * dx + rng.normal(0, 0.005),
                                         -ly / 2 + (j + 0.5) * dy + rng.normal(0, 0.005),
                                         dx - s, dy - s, base + k * dz, base + (k + 1) * dz - rng.uniform(0, 0.004)))
        if not layer:
            layer.append(carton_part(rng, skus[0], 0, 0, dx, dy, base + k * dz, base + (k + 1) * dz))
        parts += layer
    if nx * ny > 1:  # dark core: gaps between cartons look into shadow, not onto the pallet
        parts.append(dict(part(prism(0, 0, lx - 0.03, ly - 0.03, base, base + max(dz, (nz - 1) * dz) - 0.01),
                               lambda f, w, h: plain(rng, w, h, rng.uniform(15, 40), 2, 2)), dmc_ok=False))
    return parts


def load_mixed(rng, lx, ly, H, base, skus):
    """Columns of different cartons and heights, a few loose cartons askew on top (real package 4)."""
    nx, ny = max(1, round(lx / rng.uniform(0.3, 0.6))), max(1, round(ly / rng.uniform(0.25, 0.5)))
    dx, dy = lx / nx, ly / ny
    parts, tops = [], []
    tall = (rng.integers(nx), rng.integers(ny))
    for i in range(nx):
        for j in range(ny):
            ht = H if (i, j) == tall else H * rng.uniform(0.4, 1.0)
            z = base
            sku = skus[rng.integers(len(skus))]
            sl, sw = dx * rng.uniform(0.75, 0.98), dy * rng.uniform(0.75, 0.98)
            cx0, cy0 = -lx / 2 + (i + 0.5) * dx, -ly / 2 + (j + 0.5) * dy
            while z < base + ht - 0.05:
                ch = min(rng.uniform(0.15, 0.4), base + ht - z)
                parts.append(carton_part(rng, sku, cx0 + rng.normal(0, 0.01), cy0 + rng.normal(0, 0.01), sl, sw,
                                         z, z + ch, rng.normal(0, 0.04)))
                z += ch
            tops.append((cx0, cy0, z))
    for _ in range(rng.integers(0, 4) if rng.random() < 0.5 else 0):
        cx0, cy0, z = tops[rng.integers(len(tops))]
        sku = skus[rng.integers(len(skus))]
        parts.append(carton_part(rng, sku, cx0 + rng.normal(0, 0.05), cy0 + rng.normal(0, 0.05),
                                 rng.uniform(0.2, 0.4), rng.uniform(0.15, 0.3), z, z + rng.uniform(0.1, 0.25),
                                 rng.uniform(-0.7, 0.7)))
    parts.append(dict(part(prism(0, 0, lx * 0.7, ly * 0.7, base, base + H * 0.35),
                           lambda f, w, h: plain(rng, w, h, rng.uniform(15, 40), 2, 2)), dmc_ok=False))
    return parts


def strap_parts(rng, lo, hi, z0, z1, axis):
    """1-3 straps over the top of the hull [lo, hi] x [z0, z1] running along `axis` (0 = x)."""
    tone = rng.uniform(170, 235) if rng.random() < 0.7 else rng.uniform(25, 60)
    tex = lambda f, w, h: plain(rng, w, h, tone, 2, 2)  # noqa: E731
    out, e, sw = [], 0.004, rng.uniform(0.012, 0.02)
    n = rng.integers(1, 4)
    other = 1 - axis
    for k in range(n):
        p = lo[other] + (hi[other] - lo[other]) * (k + 1) / (n + 1) + rng.normal(0, 0.02)
        ext = hi[axis] - lo[axis] + 2 * e
        mid = (lo[axis] + hi[axis]) / 2
        if axis == 0:
            out.append(part(prism(mid, p, ext, sw, z1, z1 + e), tex, mat="strap"))
            for x in (lo[0] - e / 2, hi[0] + e / 2):
                out.append(part(prism(x, p, e, sw, z0, z1 + e), tex, mat="strap"))
        else:
            out.append(part(prism(p, mid, sw, ext, z1, z1 + e), tex, mat="strap"))
            for y in (lo[1] - e / 2, hi[1] + e / 2):
                out.append(part(prism(p, y, sw, e, z0, z1 + e), tex, mat="strap"))
    return out


def film_quads(rng, lo, hi, z0, z1, top):
    """Stretch-film hull (local frame) as overlay quads (4,3) with (tone, alpha) textures."""
    e = 0.008
    xy = np.array([[lo[0] - e, lo[1] - e], [hi[0] + e, lo[1] - e], [hi[0] + e, hi[1] + e], [lo[0] - e, hi[1] + e]])
    P = np.concatenate([np.c_[xy, np.full(4, z0)], np.c_[xy, np.full(4, z1 + e)]])
    faces = list(range(1, 5)) + ([0] if top else [])
    return [(P[list(FACES[f])], f == 0) for f in faces]


def make_unit(rng):
    """One shipping unit in its local frame -> dict(parts, quads, kind, type, pallet, ...)."""
    r = rng.random()
    kind = "carton" if r < 0.15 else "cage" if r < 0.23 else "pallet_load"
    parts, quads, pallet, base, sub = [], [], None, 0.0, kind
    if kind != "carton":
        q = rng.random()
        pallet = (1.2, 0.8) if q < 0.6 else (1.2, 1.0) if q < 0.85 else \
            tuple(sorted([rng.uniform(0.8, 1.6), rng.uniform(0.6, 1.2)], reverse=True))
    if kind == "carton":
        lx, ly = sorted([rng.uniform(0.3, 1.4), rng.uniform(0.25, 1.0)], reverse=True)
        H = rng.uniform(0.2, 1.2)
        sku = make_sku(rng)
        if rng.random() < 0.25 and H > 0.4:  # 2-3 cartons stacked
            k = int(rng.integers(2, 4))
            for i in range(k):
                parts.append(carton_part(rng, sku, rng.normal(0, 0.01), rng.normal(0, 0.01), lx, ly,
                                         H * i / k, H * (i + 1) / k, rng.normal(0, 0.02), cull=False))
        else:
            parts.append(carton_part(rng, sku, 0, 0, lx, ly, 0, H, 0, cull=False))
        load = [lx, ly]
    elif kind == "cage":
        L, W = pallet
        H = rng.uniform(0.7, 1.3)
        steel = rng.uniform(60, 160)
        stex = lambda f, w, h: plain(rng, w, h, steel, 4, 6)  # noqa: E731
        parts.append(part(prism(0, 0, L, W, 0.0, 0.12), lambda f, w, h: plain(rng, w, h, steel * 0.5, 3, 4),
                          mat="steel"))
        parts.append(part(prism(0, 0, L - 0.04, W - 0.04, 0.12, 0.15),
                          lambda f, w, h: (wood(rng, w, h, rng.uniform(80, 160)) if rng.random() < 0.5
                                           else plain(rng, w, h, steel * 0.8, 4, 8)), mat="steel"))
        for sx in (-1, 1):
            for sy in (-1, 1):
                parts.append(part(prism(sx * (L / 2 - 0.02), sy * (W / 2 - 0.02), 0.04, 0.04, 0.0, H), stex,
                                  mat="steel"))
        P = prism(0, 0, L - 0.01, W - 0.01, 0.15, H - 0.02)
        for f in range(1, 5):
            quads.append(("mesh", P[list(FACES[f])], rng.uniform(0.04, 0.1), rng.uniform(120, 210)))
        for i in range(4):  # rim
            a, b = P[4 + i, :2], P[4 + (i + 1) % 4, :2]
            m, d = (a + b) / 2, b - a
            parts.append(part(prism(m[0], m[1], np.linalg.norm(d) + 0.03, 0.03, H - 0.03, H, math.atan2(-d[1], d[0])),
                              stex, mat="steel"))
        if rng.random() < 0.6:  # contents
            sku = make_sku(rng)
            parts += load_stack(rng, L - 0.1, W - 0.1, rng.uniform(0.2, H - 0.25), 0.15, [sku])
        load, base = [L, W], 0.0
        sub = "cage"
    else:
        L, W = pallet
        parts += build_pallet(rng, L, W)
        base = PALLET_H
        q = rng.random()
        if q < 0.7:
            load = [L + rng.uniform(-0.15, 0.1), W + rng.uniform(-0.15, 0.1)]
        elif q < 0.85:
            load = [L + rng.uniform(0.1, 0.6), rng.uniform(0.3, W)]
        else:
            load = [rng.uniform(0.3, L), rng.uniform(0.25, W)]
        H = rng.uniform(0.15, 2.05)
        lx, ly = load
        ox, oy = rng.normal(0, 0.03), rng.normal(0, 0.03)
        sub = rng.choice(["stack", "mixed", "single", "panel", "machine", "small"],
                         p=[0.35, 0.12, 0.23, 0.1, 0.1, 0.1])
        if q >= 0.7 and q < 0.85:
            sub = "panel" if rng.random() < 0.6 else "single"
        lp = []
        if sub == "stack":
            skus = [make_sku(rng)] if rng.random() < 0.7 else [make_sku(rng) for _ in range(rng.integers(2, 4))]
            lp = load_stack(rng, lx, ly, H, base, skus)
        elif sub == "mixed":
            lp = load_mixed(rng, lx, ly, H, base, [make_sku(rng) for _ in range(rng.integers(2, 5))])
        elif sub == "single":
            if rng.random() < 0.25:
                lp = [part(prism(0, 0, lx, ly, base, base + H), lambda f, w, h: plywood(rng, w, h), ao=0.15, mat="wood")]
            else:
                sku = make_sku(rng)
                if rng.random() < 0.6:  # big boxes are mostly plain, taped cardboard
                    sku.update(style="plain", side_logo=False)
                lp = [part(prism(0, 0, lx, ly, base, base + H), lambda f, w, h: carton_face(rng, sku, w, h, f == 0),
                           ao=0.15, mat="cardboard")]
        elif sub == "panel":
            tone = rng.uniform(110, 180)
            H = rng.uniform(0.05, 1.0)
            lp = [part(prism(0, 0, lx, ly, base, base + H), lambda f, w, h: plain(rng, w, h, tone, 2, 4), ao=0.1, mat="panel")]
        elif sub == "machine":
            tone = rng.uniform(30, 110)
            lp = [part(prism(0, 0, lx, ly, base, base + H), lambda f, w, h: machine_face(rng, w, h, tone), ao=0.15, mat="paint")]
        else:  # small item(s) on a big pallet
            sku = make_sku(rng)
            lx, ly = rng.uniform(0.3, L), rng.uniform(0.25, W)
            H = rng.uniform(0.15, 1.0)
            ox, oy = rng.uniform(-(L - lx) / 2, (L - lx) / 2), rng.uniform(-(W - ly) / 2, (W - ly) / 2)
            lp = [carton_part(rng, sku, 0, 0, lx, ly, base, base + H, rng.normal(0, 0.05), cull=False)]
            load = [lx, ly]
        yaw_off = rng.normal(0, 0.03) if rng.random() < 0.5 else 0.0
        for p in lp:
            p["P"] = transform(p["P"], yaw_off, ox, oy)
        parts += lp
        lo = np.array([ox - lx / 2, oy - ly / 2])
        hi = np.array([ox + lx / 2, oy + ly / 2])
        if sub in ("stack", "single", "machine", "mixed") and rng.random() < {"stack": 0.5, "mixed": 0.3}.get(sub, 0.25):
            for qd, top in film_quads(rng, lo, hi, base, base + H, rng.random() < 0.6):
                quads.append(("film", qd, top))
        if sub in ("single", "machine", "stack") and rng.random() < (0.5 if sub == "machine" else 0.2):
            parts += strap_parts(rng, lo, hi, base, base + H, int(rng.integers(2)))
    return {"kind": sub if kind == "pallet_load" else kind, "type": "parcel" if kind == "carton" else "palletised_load",
            "parts": parts, "quads": quads, "pallet": list(pallet) if pallet else None, "load": [float(v) for v in load]}


def unit_extent(u):
    """Overall footprint (lo, hi) in the local frame and total height from the geometry itself."""
    pts = np.concatenate([p["P"] for p in u["parts"]] + [q[1] for q in u["quads"]])
    return pts[:, :2].min(0), pts[:, :2].max(0), float(pts[:, 2].max())


def add_dmc(rng, u, H):
    """Put the shipping label on (one of) the highest top faces; returns local (4,3) symbol corners."""
    cands = [p for p in u["parts"] if p["P"][4:, 2].max() > H - 0.02 and p.get("dmc_ok", True)
             and np.linalg.norm(p["P"][5] - p["P"][4]) > 0.2 and np.linalg.norm(p["P"][7] - p["P"][4]) > 0.2]
    if not cands:
        return None
    p = cands[rng.integers(len(cands))]
    top = p["P"][4:]
    tw, th = px(np.linalg.norm(top[1] - top[0])), px(np.linalg.norm(top[3] - top[0]))
    t = p["tex"](0, tw, th)
    if isinstance(t, tuple):
        return None
    t = t.copy()
    c = place_dmc(rng, t)
    if c is None:
        return None
    p["top_tex"] = t
    uv = np.array(c) / (tw, th)
    return top[0] + uv[:, :1] * (top[1] - top[0]) + uv[:, 1:] * (top[3] - top[0])


def cull_fn(cartons):
    """skip(face_centre, face) for faces touching another carton (interior of a stack)."""
    boxes = [(p["P"][:4, :2], p["P"][0, 2], p["P"][4, 2]) for p in cartons]

    def inside(pt):
        for xy, z0, z1 in boxes:
            if not (z0 - 1e-6 <= pt[2] <= z1 + 1e-6):
                continue
            d = np.roll(xy, -1, 0) - xy
            cr = d[:, 0] * (pt[1] - xy[:, 1]) - d[:, 1] * (pt[0] - xy[:, 0])
            if (cr >= 0).all() or (cr <= 0).all():
                return True
        return False

    def skip(fc, fi, P):
        cen = P.mean(0)
        if fi == 0:
            probe = fc + (0, 0, 0.025)
        elif fi == 5:
            return True
        else:
            n = fc[:2] - cen[:2]
            probe = np.r_[fc[:2] + n / (np.linalg.norm(n) + 1e-9) * 0.025, fc[2]]
        return inside(probe)
    return skip


# ---------------------------------------------------------------------------------- scene
def sample_pose(rng, cam):
    yaw = rng.uniform(0, math.pi) if rng.random() < 0.5 else rng.choice([0, math.pi / 2]) + rng.normal(0, 0.12)
    half_x = cam.mount_h * cam.size[0] / 2 / cam.fx * 0.62
    half_y = cam.mount_h * cam.size[1] / 2 / cam.fy * 0.58
    return rng.uniform(-half_x, half_x), rng.uniform(-half_y, half_y), yaw


def render_parts(R, parts, light, yaw, cx, cy):
    cart = [p for p in parts if p["cull"]]
    world = [dict(p, P=transform(p["P"], yaw, cx, cy)) for p in parts]
    skip_local = cull_fn(cart) if cart else None
    for p, pl in zip(world, parts):
        skip = None
        if pl["cull"] and skip_local is not None:
            skip = (lambda P: (lambda fc_world, fi: skip_local(P[list(FACES[fi])].mean(0), fi, P)))(pl["P"])
        R.box(p["P"], p["tex"], light, p["ao"], p["top_tex"], skip, p["shade"])


def render_quads(R, rng, quads, light, yaw, cx, cy):
    for q in quads:
        Q = transform(q[1], yaw, cx, cy)
        Qc = R.w2c(Q)
        w_px, h_px = px(np.linalg.norm(Q[1] - Q[0])), px(np.linalg.norm(Q[3] - Q[0]))
        n = np.cross(Qc[1] - Qc[0], Qc[3] - Qc[0])
        n /= np.linalg.norm(n) + 1e-12
        n = n if n @ Qc.mean(0) < 0 else -n
        if q[0] == "film":
            tone, a = film_tex(rng, w_px, h_px, q[2])
            R.quad(Qc, tone, 1.0, a, blend=True)
        else:
            t, a = mesh_tex(rng, w_px, h_px, q[2], q[3])
            R.quad(Qc, t, light.shade(n), a)


def floor_clutter(rng, R, light):
    for _ in range(rng.integers(1, 4)):
        x, y = rng.uniform(-2.5, 2.5), rng.uniform(-2, 2)
        l, w = rng.uniform(0.05, 0.4), rng.uniform(0.02, 0.3)
        tone = rng.uniform(110, 200) if rng.random() < 0.7 else rng.uniform(20, 45)
        R.box(prism(x, y, l, w, 0.0, 0.002, rng.uniform(0, math.pi)),
              lambda f, ww, hh: plain(rng, ww, hh, tone, 4, 12) * (0.75 + 0.25 * grain_noise(rng, hh, ww, 30, 30)), light)


def sun_patch(rng, w, h):
    """Saturated sunlight entering from the bottom-right edge (0-5 % of the frame in the real data)."""
    m = np.zeros((h, w), np.float32)
    c = (int(rng.uniform(1.0, 1.1) * w), int(rng.uniform(0.85, 1.05) * h))
    ax = (int(rng.uniform(0.03, 0.14) * w), int(rng.uniform(0.08, 0.3) * h))
    cv2.ellipse(m, c, ax, rng.uniform(-25, 25), 0, 360, 1.0, -1, cv2.LINE_AA)
    return cv2.GaussianBlur(m, (0, 0), rng.uniform(1, 5))


def finish(rng, bg, R, flip):
    b = bg.astype(np.float32)
    if rng.random() < 0.5:
        b = b * rng.uniform(0.85, 1.15) + rng.uniform(-8, 8)
    floor = R.alpha < 0.5
    out = np.where(floor, b * (1 - R.shadow), R.img)
    h, w = out.shape
    if rng.random() < 0.8:  # 15 of the 19 real frames have it
        m = sun_patch(rng, w, h)
        if flip is not None:
            m = cv2.flip(m, flip)
        out = out * (1 - m) + 255 * m
    if rng.random() < 0.3:  # vignetting
        yy, xx = np.mgrid[0:h, 0:w]
        r2 = ((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2
        out *= 1 - rng.uniform(0.05, 0.2) * r2 / 2
    out = out * rng.uniform(0.88, 1.12) + rng.uniform(-8, 8)
    out = 255 * np.clip(out / 255, 0, 1) ** rng.uniform(0.88, 1.15)
    out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.35, 1.0))
    out += rng.normal(0, rng.uniform(0.8, 3), out.shape)
    out[bg == 0] = 0
    return np.clip(out, 0, 255).astype(np.uint8)


def make_sample(seed, cam, bg):
    rng = np.random.default_rng(seed)
    R = ZRenderer(cam)
    light = Light(rng)
    flip = None
    if rng.random() < 0.5:
        flip = int(rng.integers(-1, 2))
        bg = cv2.flip(bg, flip)
    n = 0 if rng.random() < 0.08 else 2 if rng.random() < 0.10 else 1
    objects, placed = [], []
    for _ in range(n):
        u = make_unit(rng)
        lo, hi, H = unit_extent(u)
        truck = None
        if u["pallet"] and u["kind"] != "cage" and rng.random() < 0.7:
            L = u["pallet"][0]
            end = 1 if rng.random() < 0.5 else -1
            root = L / 2 + (rng.uniform(0.0, 0.08) if rng.random() < 0.75 else rng.uniform(0.1, 0.6))
            truck = (build_truck(rng), end, root, rng.normal(0, 0.02))
        reach = math.hypot(*(hi - lo)) / 2 + (1.4 if truck else 0.2)
        foot_local = rect_corners(*(lo + hi) / 2, *(hi - lo), 0.0)
        for _try in range(30):
            cx, cy, yaw = sample_pose(rng, cam)
            foot = transform(np.c_[foot_local, np.zeros(4)], yaw, cx, cy)[:, :2]
            uv = cam.project(np.concatenate([R.w2c(np.c_[foot, np.full(4, z)]) for z in (0.0, H)]))
            inside = (uv.min(0) >= 2).all() and (uv.max(0) <= np.array(cam.size) - 2).all()
            if inside and all(math.hypot(cx - px_, cy - py_) > reach + r_ for px_, py_, r_ in placed):
                break
        else:
            continue
        placed.append((cx, cy, reach))
        dmc = add_dmc(rng, u, H) if (u["kind"] != "cage" and rng.random() < 0.9) else None
        R.floor_shadow(foot, H, light, rng)
        render_parts(R, u["parts"], light, yaw, cx, cy)
        if truck:
            parts, end, root, dy = truck
            # truck frame: forks along +x from the heel; heel sits beyond the pallet end `end`
            tyaw = yaw + (math.pi if end > 0 else 0.0)
            tx, ty = transform(np.array([[end * root, dy, 0.0]]), yaw, cx, cy)[0, :2]
            render_parts(R, parts, light, tyaw, tx, ty)
        render_quads(R, rng, u["quads"], light, yaw, cx, cy)
        bottom, top = R.w2c(np.c_[foot, np.zeros(4)]), R.w2c(np.c_[foot, np.full(4, H)])
        pal_top = None
        if u["pallet"]:
            pal_top = R.w2c(transform(np.c_[rect_corners(0, 0, *u["pallet"], 0), np.full(4, PALLET_H)], yaw, cx, cy))
        objects.append({
            "name": f"unit_{len(objects):02d}",
            "type": u["type"],
            "object_dimensions_m": {"width_m": float(np.linalg.norm(foot[1] - foot[0])),
                                    "length_m": float(np.linalg.norm(foot[3] - foot[0])),
                                    "height_m": H},
            "boxes_3d_cam": np.round(np.concatenate([bottom, top]), 6).tolist(),
            "synth": {"kind": u["kind"], "top_face_3d": np.round(top, 6).tolist(),
                      "pallet_top_3d": np.round(pal_top, 6).tolist() if pal_top is not None else None,
                      "dmc_3d": np.round(R.w2c(transform(dmc, yaw, cx, cy)), 6).tolist() if dmc is not None else None,
                      "load_lw_m": u["load"], "load_h_m": H - (PALLET_H if u["pallet"] and u["kind"] != "cage" else 0.0),
                      "pallet_lw_m": u["pallet"], "truck": truck is not None},
        })
    if n == 0 or rng.random() < 0.15:  # stray pallet truck
        cx, cy, yaw = sample_pose(rng, cam)
        if all(math.hypot(cx - px_, cy - py_) > 1.0 + r_ for px_, py_, r_ in placed):
            render_parts(R, build_truck(rng), light, yaw, cx, cy)
    if rng.random() < 0.3:
        floor_clutter(rng, R, light)
    return finish(rng, bg, R, flip), objects


def worker(args):
    start, count, seed, out, scale, canvas, cam_dir = args
    cam = Camera(cam_dir, scale)
    bg = cv2.resize(cv2.imread(os.path.join(ROOT, "assets", "background_median.png"), cv2.IMREAD_GRAYSCALE),
                    cam.size, interpolation=cv2.INTER_AREA)
    bg = cam.undistort_image(bg)
    for i in range(start, start + count):
        img, objects = make_sample(seed * 1_000_003 + i, cam, bg)
        shard = os.path.join(out, f"shard_{i // SHARD * SHARD:08d}")
        os.makedirs(shard, exist_ok=True)
        cv2.imwrite(os.path.join(shard, f"{i:08d}.gray.png"), pad_square(img, canvas))
        label = {"camera_intrinsics": cam.intrinsics_json(canvas), "num_objects": len(objects), "objects": objects,
                 "camera_pose": {"height_m": cam.mount_h, "tilt_deg": 0.0, "roll_deg": 0.0},
                 "valid_region": [0, 0, cam.size[0], cam.size[1]], "seed": int(seed * 1_000_003 + i),
                 "generator": "synth/generate_v2.py"}
        with open(os.path.join(shard, f"{i:08d}.json"), "w") as f:
            json.dump(label, f, indent=1)
    return count


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cam-dir", default=CAM_DIR, help="folder with cam_info.json and raw_*.json")
    ap.add_argument("--out", default=os.path.join(ROOT, "datasets", "SYNTH_CAM_000CDF0AD90F_v2"))
    ap.add_argument("--n", type=int, default=3000, help="number of images")
    ap.add_argument("--scale", type=float, default=0.5, help="image scale vs the 2560x2048 sensor")
    ap.add_argument("--canvas", type=int, default=1280, help="square output side; frame goes top-left")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    chunk = 25
    jobs = [(s, min(chunk, args.n - s), args.seed, args.out, args.scale, args.canvas, args.cam_dir)
            for s in range(0, args.n, chunk)]
    from multiprocessing import Pool
    done = 0
    with Pool(args.workers) as pool:
        for c in pool.imap_unordered(worker, jobs):
            done += c
            print(f"{done}/{args.n}", flush=True)


if __name__ == "__main__":
    main()
