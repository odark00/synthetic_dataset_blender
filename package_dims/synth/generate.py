"""Procedural synthetic dataset for overhead package dimensioning (camera cam_000CDF0AD90F).

Every image is rendered in the real camera geometry: real intrinsics, real mounting height,
real empty-floor background (median of the real frames). Objects are textured 3D cuboids:
cartons, shrink-wrapped pallet loads, carton stacks, white panels, cages, plus pallet trucks
as distractors. Because the scene is known, labels are exact.

Images are PINHOLE (lens distortion removed, like custom_3d_model expects); real frames
must get the same treatment at test time (see prepare_real.py).

Output = custom_3d_model "objects" schema (same as TASLER_MULTI / TASLER_PACK):
    <out>/shard_XXXXXXXX/{idx:08d}.gray.png   1280x1024 frame top-left on a 1280x1280 black canvas
    <out>/shard_XXXXXXXX/{idx:08d}.json       camera_intrinsics + objects[]:
        boxes_3d_cam  8 corners, OpenCV camera frame, bottom 0-3 (floor) / top 4-7
        object_dimensions_m  width_m = |c1-c0|, length_m = |c3-c0|, height_m from the floor
        -> the whole shipping unit, PALLET INCLUDED (same as the Readme ground truth)
        synth  extra info ignored by the loader (top face, pallet top, DMC corners, ...)

Run (CPU only, low priority so it does not disturb other jobs):
    nice -n 19 python synth/generate.py --n 4000 --workers 4
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
from camera import Camera, pad_square  # noqa: E402

CAM_DIR = os.path.join(os.path.dirname(ROOT), "cam_000CDF0AD90F")
LABEL_M = 0.0948      # DMC symbol side, estimated from the real frames
TEX_PPM = 350         # texture resolution, px per metre (image is ~320 px/m at 4 m)
PALLET_H = 0.144
SHARD = 250          # images per shard_XXXXXXXX folder (as in TASLER_*)


# ---------------------------------------------------------------------------------- textures
def smooth_noise(rng, h, w, cell):
    """Low-frequency noise in [-1, 1] with features of ~cell px."""
    sh, sw = max(2, h // cell + 2), max(2, w // cell + 2)
    return cv2.resize(rng.uniform(-1, 1, (sh, sw)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)


def tex_size(a_m, b_m):
    return max(8, int(a_m * TEX_PPM)), max(8, int(b_m * TEX_PPM))  # (w, h)


def cardboard(rng, w, h, base=None):
    base = rng.uniform(70, 175) if base is None else base
    t = base + smooth_noise(rng, h, w, 60) * rng.uniform(4, 15) + rng.normal(0, rng.uniform(1, 4), (h, w))
    if rng.random() < 0.6:  # packing tape along one axis
        tw = max(3, int(0.05 * TEX_PPM))
        tone = base + rng.choice([-1, 1]) * rng.uniform(15, 45)
        if rng.random() < 0.5:
            y = h // 2 + int(rng.uniform(-0.05, 0.05) * h)
            t[max(0, y - tw // 2):y + tw // 2] = tone
        else:
            x = w // 2 + int(rng.uniform(-0.05, 0.05) * w)
            t[:, max(0, x - tw // 2):x + tw // 2] = tone
    for _ in range(rng.integers(0, 4)):  # prints / small labels
        pw, ph = int(rng.uniform(0.04, 0.25) * TEX_PPM), int(rng.uniform(0.02, 0.15) * TEX_PPM)
        x, y = rng.integers(0, max(1, w - pw)), rng.integers(0, max(1, h - ph))
        tone = rng.uniform(170, 225) if rng.random() < 0.5 else rng.uniform(25, 70)
        if rng.random() < 0.5:
            t[y:y + ph, x:x + pw] = tone
        else:
            cv2.ellipse(t, (x + pw // 2, y + ph // 2), (pw // 2, ph // 2), 0, 0, 360, float(tone), -1)
    if rng.random() < 0.3:  # creases / flap seams
        for _ in range(rng.integers(1, 3)):
            p0 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
            p1 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
            cv2.line(t, p0, p1, float(base - rng.uniform(10, 30)), int(rng.integers(1, 3)))
    return t


def carton_grid(rng, w, h):
    """Top of a stack of cartons: grid of boxes with dark seams and optional logos."""
    dark = rng.random() < 0.35
    base = rng.uniform(35, 80) if dark else rng.uniform(80, 175)
    cw, ch = int(rng.uniform(0.25, 0.6) * TEX_PPM), int(rng.uniform(0.2, 0.5) * TEX_PPM)
    t = np.zeros((h, w), np.float32)
    logo = rng.random() < 0.5
    for y0 in range(0, h, ch):
        for x0 in range(0, w, cw):
            cell = cardboard(rng, min(cw, w - x0), min(ch, h - y0), base + rng.normal(0, 8))
            if logo and cell.shape[0] > 20 and cell.shape[1] > 20:
                c = (cell.shape[1] // 2, cell.shape[0] // 2)
                ax = (max(3, cell.shape[1] // 5), max(2, cell.shape[0] // 8))
                cv2.ellipse(cell, c, ax, 0, 0, 360, float(rng.uniform(170, 240) if dark else rng.uniform(20, 60)), -1)
            t[y0:y0 + cell.shape[0], x0:x0 + cell.shape[1]] = cell
            seam = float(base * rng.uniform(0.3, 0.6))
            cv2.line(t, (x0, y0), (x0 + cw, y0), seam, int(rng.integers(2, 5)))
            cv2.line(t, (x0, y0), (x0, y0 + ch), seam, int(rng.integers(2, 5)))
    return t


def white_panel(rng, w, h):
    return rng.uniform(140, 210) + smooth_noise(rng, h, w, 120) * rng.uniform(3, 10) + rng.normal(0, 2, (h, w))


def shrink_wrap(rng, t):
    """Lower contrast + specular streaks of stretch film."""
    h, w = t.shape
    t = t * rng.uniform(0.8, 0.95) + rng.uniform(10, 35)
    streak = np.zeros_like(t)
    for _ in range(rng.integers(2, 8)):
        pts = np.stack([rng.uniform(0, w, 4), rng.uniform(0, h, 4)], 1).astype(np.int32)
        cv2.polylines(streak, [pts], False, float(rng.uniform(15, 60)), int(rng.integers(1, 6)))
    streak = cv2.GaussianBlur(streak, (0, 0), rng.uniform(1, 4))
    return np.clip(t + streak, 0, 255)


def pallet_top(rng, w, h):
    """Wooden slats running along the long axis (texture x-axis), dark gaps between them."""
    t = np.full((h, w), rng.uniform(15, 50), np.float32)
    n = int(rng.integers(5, 9))
    slat = h / n * rng.uniform(0.55, 0.8)
    wood = rng.uniform(100, 190)
    for i in range(n):
        y0 = int(np.clip(i * (h - slat) / max(1, n - 1), 0, h - slat))
        grain = smooth_noise(rng, int(slat), w, 30)
        grain = cv2.resize(grain, (w, int(slat)))
        t[y0:y0 + int(slat)] = wood + rng.normal(0, 10) + grain * 12
    for x0 in (0, w // 2 - int(0.07 * TEX_PPM), w - int(0.145 * TEX_PPM)):  # cross boards
        t[:, x0:x0 + int(0.145 * TEX_PPM)] = np.maximum(t[:, x0:x0 + int(0.145 * TEX_PPM)], wood * 0.9)
    return t + rng.normal(0, 3, t.shape)


def pallet_side(rng, w, h):
    t = np.full((h, w), rng.uniform(10, 40), np.float32)
    wood = rng.uniform(90, 170)
    bw = int(0.145 * TEX_PPM)
    for x0 in (0, w // 2 - bw // 2, w - bw):  # blocks
        t[:, x0:x0 + bw] = wood
    t[:max(2, h // 5)] = wood
    t[-max(2, h // 5):] = wood
    return t + rng.normal(0, 4, t.shape)


def mesh(rng, w, h):
    """Wire mesh of a cage wall / floor."""
    t = np.full((h, w), rng.uniform(30, 80), np.float32) + smooth_noise(rng, h, w, 40) * 15
    step = max(4, int(rng.uniform(0.04, 0.1) * TEX_PPM))
    tone = float(rng.uniform(140, 220))
    for x in range(0, w, step):
        cv2.line(t, (x, 0), (x, h), tone, 2)
    for y in range(0, h, step):
        cv2.line(t, (0, y), (w, y), tone, 2)
    return t


def dmc_symbol(rng, px):
    """16x16 DataMatrix-like symbol (L finder + clock track + random data), side px."""
    m = rng.integers(0, 2, (16, 16)).astype(np.uint8)
    m[:, 0] = 1
    m[-1, :] = 1
    m[0, ::2] = 1
    m[0, 1::2] = 0
    m[1::2, -1] = 1
    m[::2, -1] = 0
    img = cv2.resize((1 - m) * 255, (px, px), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    return np.rot90(img, rng.integers(0, 4)).copy()


def place_dmc(rng, t):
    """Paste a white shipping label with a DMC into a top texture; return symbol corners (texture px)."""
    h, w = t.shape
    s = int(LABEL_M * TEX_PPM)
    pad = int(rng.uniform(0.01, 0.05) * TEX_PPM)
    lw, lh = s + 2 * pad + int(rng.uniform(0, 0.08) * TEX_PPM), s + 2 * pad + int(rng.uniform(0, 0.08) * TEX_PPM)
    if lw + 4 >= w or lh + 4 >= h:
        return None
    patch = np.full((lh, lw), rng.uniform(180, 235), np.float32)
    patch[pad:pad + s, pad:pad + s] = np.minimum(dmc_symbol(rng, s), patch[0, 0])
    ang = rng.uniform(0, 360) if rng.random() < 0.5 else rng.choice([0, 90, 180, 270]) + rng.normal(0, 3)
    hx = hy = math.hypot(lw, lh) / 2 + 1
    if 2 * hx + 4 >= min(w, h):  # face too small for a rotated label: keep it axis-aligned
        ang, hx, hy = 0, lw / 2, lh / 2
    cx = rng.uniform(hx + 2, w - hx - 2)
    cy = rng.uniform(hy + 2, h - hy - 2)
    M = cv2.getRotationMatrix2D((lw / 2, lh / 2), ang, 1.0)
    M[:, 2] += (cx - lw / 2, cy - lh / 2)
    alpha = cv2.warpAffine(np.ones_like(patch), M, (w, h))
    warped = cv2.warpAffine(patch, M, (w, h))
    t[:] = t * (1 - alpha) + warped * alpha
    sym = np.array([[pad, pad], [pad + s, pad], [pad + s, pad + s], [pad, pad + s]], np.float64)
    return (sym @ M[:, :2].T + M[:, 2]).tolist()


# ---------------------------------------------------------------------------------- geometry
def rect_corners(cx, cy, l, w, yaw):
    """Floor-plane rectangle corners (4,2), ordered along +length then +width."""
    c, s = math.cos(yaw), math.sin(yaw)
    loc = np.array([[-l / 2, -w / 2], [l / 2, -w / 2], [l / 2, w / 2], [-l / 2, w / 2]])
    return loc @ np.array([[c, s], [-s, c]]) + (cx, cy)


class Renderer:
    """Paints planar textured quads into an undistorted (pinhole) canvas with an alpha mask."""

    def __init__(self, cam):
        self.cam = cam
        w, h = cam.size
        self.img = np.zeros((h, w), np.float32)
        self.alpha = np.zeros((h, w), np.float32)
        self.shadow = np.zeros((h, w), np.float32)

    def to3d(self, xy, height):
        return np.c_[xy, np.full(len(xy), self.cam.mount_h - height)]

    def quad(self, p3d, tex, shade=1.0):
        """Map texture corners (0,0),(w,0),(w,h),(0,h) onto 3D quad p3d (4,3)."""
        dst = self.cam.project(p3d).astype(np.float32)
        th, tw = tex.shape
        src = np.float32([[0, 0], [tw, 0], [tw, th], [0, th]])
        W, H = self.cam.size
        x0, y0 = np.floor(dst.min(0)).astype(int)
        x1, y1 = np.ceil(dst.max(0)).astype(int)
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
        if x1 <= x0 or y1 <= y0:
            return
        Hm = cv2.getPerspectiveTransform(src, (dst - (x0, y0)).astype(np.float32))
        roi = cv2.warpPerspective(tex * shade, Hm, (x1 - x0, y1 - y0), flags=cv2.INTER_AREA)
        m = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillConvexPoly(m, np.round((dst - (x0, y0)) * 16).astype(np.int32), 1, cv2.LINE_AA, 4)
        m = m.astype(bool)
        self.img[y0:y1, x0:x1][m] = roi[m]
        self.alpha[y0:y1, x0:x1][m] = 1

    def floor_shadow(self, rng, xy, strength):
        pts = np.round(self.cam.project(self.to3d(xy, 0.0))).astype(np.int32)
        m = np.zeros_like(self.shadow)
        cv2.fillPoly(m, [pts], 1.0)
        m = cv2.GaussianBlur(m, (0, 0), rng.uniform(8, 30))
        self.shadow = np.maximum(self.shadow, m * strength)

    def cuboid(self, rng, xy, z0, z1, top_tex, side_tex_fn, shades=None):
        """Draw visible side faces then the top. xy: floor corners (4,2) in the order of rect_corners."""
        bot, top = self.to3d(xy, z0), self.to3d(xy, z1)
        faces = []
        for i in range(4):
            j = (i + 1) % 4
            quad = np.array([top[i], top[j], bot[j], bot[i]])
            centre = quad.mean(0)
            normal = np.r_[xy[j] - xy[i], 0.0]
            normal = np.array([normal[1], -normal[0], 0.0])
            if np.dot(normal[:2], centre[:2] - xy.mean(0)) < 0:
                normal = -normal
            if np.dot(normal, -centre) > 0:  # face looks towards the camera
                faces.append((np.linalg.norm(centre), quad, i))
        for _, quad, i in sorted(faces, key=lambda f: -f[0]):
            edge = np.linalg.norm(xy[(i + 1) % 4] - xy[i])
            self.quad(quad, side_tex_fn(*tex_size(edge, z1 - z0)),
                      shades[i] if shades is not None else rng.uniform(0.5, 0.9))
        if top_tex is not None:
            self.quad(top, top_tex, rng.uniform(0.9, 1.1))
        return top


# ---------------------------------------------------------------------------------- scene
def sample_pose(rng, l, w, cam):
    yaw = rng.uniform(0, math.pi) if rng.random() < 0.5 else rng.choice([0, math.pi / 2]) + rng.normal(0, 0.12)
    half_x = cam.mount_h * cam.size[0] / 2 / cam.fx * 0.62
    half_y = cam.mount_h * cam.size[1] / 2 / cam.fy * 0.58
    return rng.uniform(-half_x, half_x), rng.uniform(-half_y, half_y), yaw


def make_object(rng):
    r = rng.random()
    if r < 0.2:
        kind = "carton"
    elif r < 0.85:
        kind = "pallet_load"
    else:
        kind = "cage"
    o = {"kind": kind}
    if kind == "carton":
        o["load"] = sorted([rng.uniform(0.3, 1.4), rng.uniform(0.25, 1.0)], reverse=True)
        o["load_h"] = rng.uniform(0.2, 1.2)
        o["pallet"] = None
    else:
        q = rng.random()
        pl = (1.2, 0.8) if q < 0.6 else (1.2, 1.0) if q < 0.85 else \
            tuple(sorted([rng.uniform(0.8, 1.6), rng.uniform(0.6, 1.2)], reverse=True))
        o["pallet"] = list(pl)
        if kind == "cage":
            o["load"] = [pl[0] + rng.uniform(0, 0.05), pl[1] + rng.uniform(0, 0.05)]
            o["load_h"] = rng.uniform(0.6, 1.3)
        else:
            q = rng.random()
            if q < 0.7:    # load ~ pallet footprint, small over/underhang
                o["load"] = [pl[0] + rng.uniform(-0.15, 0.1), pl[1] + rng.uniform(-0.15, 0.1)]
            elif q < 0.85:  # long item overhanging the pallet (like a panel)
                o["load"] = [pl[0] + rng.uniform(0.1, 0.6), rng.uniform(0.3, pl[1])]
            else:           # small carton on a big pallet
                o["load"] = [rng.uniform(0.3, pl[0]), rng.uniform(0.25, pl[1])]
            o["load_h"] = rng.uniform(0.15, 2.1)
    o["load_off"] = [rng.normal(0, 0.03), rng.normal(0, 0.03)] if o["pallet"] else [0, 0]
    o["load_yaw_off"] = rng.normal(0, 0.03) if o["pallet"] and rng.random() < 0.5 else 0.0
    return o


def overall_dims(o):
    """Overall L, W (union of pallet and load footprints, in pallet axes) and total height."""
    base = PALLET_H if o["pallet"] else 0.0
    pts = [rect_corners(o["load_off"][0], o["load_off"][1], *o["load"], o["load_yaw_off"])]
    if o["pallet"]:
        pts.append(rect_corners(0, 0, *o["pallet"], 0.0))
    pts = np.concatenate(pts)
    ext = pts.max(0) - pts.min(0)
    return float(max(ext)), float(min(ext)), base + o["load_h"]


def draw_truck(rng, R, cx, cy, yaw, length=1.15):
    """Pallet truck: two light forks + dark body/handle at one end (floor level distractor)."""
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, s], [-s, c]])
    fork_tex = lambda w, h: np.full((h, w), rng.uniform(150, 225), np.float32) + rng.normal(0, 4, (h, w))  # noqa: E731
    for off in (-0.19, 0.19):
        xy = rect_corners(0, off, length, 0.16, 0) @ rot + (cx, cy)
        R.cuboid(rng, xy, 0.0, 0.08, fork_tex(*tex_size(length, 0.16)), fork_tex)
    body = rect_corners(length / 2 + 0.12, 0, 0.25, 0.55, 0) @ rot + (cx, cy)
    R.cuboid(rng, body, 0.0, 0.25, fork_tex(*tex_size(0.25, 0.55)), fork_tex)
    hx, hy = np.array([length / 2 + 0.3, 0]) @ rot + (cx, cy)
    handle = rect_corners(0, 0, 0.5, 0.06, 0) @ rot + (hx, hy)
    dark = lambda w, h: np.full((h, w), rng.uniform(10, 40), np.float32)  # noqa: E731
    R.cuboid(rng, handle, 0.8, 0.9, dark(*tex_size(0.5, 0.06)), dark)


def render_object(rng, R, o, cx, cy, yaw):
    """Render one package; return its ground truth record (3D, camera frame)."""
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, s], [-s, c]])
    base = 0.0
    rec = {"kind": o["kind"]}
    if o["pallet"]:
        pxy = rect_corners(0, 0, *o["pallet"], 0) @ rot + (cx, cy)
        R.floor_shadow(rng, pxy, rng.uniform(0.15, 0.4))
        if rng.random() < 0.35:
            draw_truck(rng, R, cx, cy, yaw + (0 if rng.random() < 0.7 else math.pi), o["pallet"][0])
        R.cuboid(rng, pxy, 0.0, PALLET_H, pallet_top(rng, *tex_size(*o["pallet"])), lambda w, h: pallet_side(rng, w, h))
        rec["pallet_top"] = R.to3d(pxy, PALLET_H).tolist()
        base = PALLET_H
    lxy = rect_corners(*o["load_off"], *o["load"], o["load_yaw_off"]) @ rot + (cx, cy)
    if not o["pallet"]:
        R.floor_shadow(rng, lxy, rng.uniform(0.15, 0.4))
    z1 = base + o["load_h"]
    tw, th = tex_size(*o["load"])
    dmc = None
    if o["kind"] == "cage":
        floor = R.to3d(lxy, base + 0.02)
        R.quad(floor, cardboard(rng, tw, th) if rng.random() < 0.6 else mesh(rng, tw, th), rng.uniform(0.6, 0.9))
        top3, bot3 = R.to3d(lxy, z1), R.to3d(lxy, base)
        walls = [np.array([top3[i], top3[(i + 1) % 4], bot3[(i + 1) % 4], bot3[i]]) for i in range(4)]
        for wq in sorted(walls, key=lambda q: -np.linalg.norm(q.mean(0))):
            R.quad(wq, mesh(rng, *tex_size(np.linalg.norm(wq[1] - wq[0]), o["load_h"])), rng.uniform(0.6, 1.0))
        top_rec = top3  # rim itself is drawn by draw_cage_frame()
    else:
        if o["kind"] == "carton":
            tt, side = cardboard(rng, tw, th), (lambda w, h: cardboard(rng, w, h))
        else:
            q = rng.random()
            if q < 0.4:
                tt, side = carton_grid(rng, tw, th), (lambda w, h: carton_grid(rng, w, h))
            elif q < 0.8:
                tt, side = cardboard(rng, tw, th), (lambda w, h: cardboard(rng, w, h))
            else:
                tt, side = white_panel(rng, tw, th), (lambda w, h: white_panel(rng, w, h))
            if rng.random() < 0.5:
                tt = shrink_wrap(rng, tt)
                side = (lambda f: (lambda w, h: shrink_wrap(rng, f(w, h))))(side)
        if rng.random() < 0.9:
            dmc_tex = place_dmc(rng, tt)
            if dmc_tex is not None:
                top3 = R.to3d(lxy, z1)
                u = np.array(dmc_tex) / (tw, th)
                dmc = (top3[0] + u[:, :1] * (top3[1] - top3[0]) + u[:, 1:] * (top3[3] - top3[0])).tolist()
        top_rec = R.cuboid(rng, lxy, base, z1, tt, side)
    L, W, Ht = overall_dims(o)
    rec.update({"top": np.asarray(top_rec).tolist(), "dmc": dmc, "L": L, "W": W, "H": Ht,
                "load": o["load"], "load_h": o["load_h"], "pallet": o["pallet"], "yaw": yaw,
                "center": [cx, cy]})
    return rec


def draw_cage_frame(R, rec, rng):
    """Cage rim as four thin bars (drawn after walls so the open interior stays visible)."""
    top = np.array(rec["top"])
    for i in range(4):
        a, b = top[i], top[(i + 1) % 4]
        d = (b - a)[:2]
        n = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-9) * 0.035
        q = np.array([a, b, b + np.r_[n, 0], a + np.r_[n, 0]])
        tex = np.full((8, max(8, int(np.linalg.norm(d) * TEX_PPM))), rng.uniform(130, 210), np.float32)
        R.quad(q, tex)


# ---------------------------------------------------------------------------------- output
def unit_footprint(o, cx, cy, yaw):
    """Floor corners (4,2) of the whole shipping unit: OBB of pallet + load in pallet axes."""
    pts = [rect_corners(*o["load_off"], *o["load"], o["load_yaw_off"])]
    if o["pallet"]:
        pts.append(rect_corners(0, 0, *o["pallet"], 0.0))
    pts = np.concatenate(pts)
    lo, hi = pts.min(0), pts.max(0)
    c, s = math.cos(yaw), math.sin(yaw)
    return rect_corners(*(lo + hi) / 2, *(hi - lo), 0.0) @ np.array([[c, s], [-s, c]]) + (cx, cy)


def finish(rng, bg, R):
    b = bg.astype(np.float32)
    if rng.random() < 0.5:  # global light change of the background
        b = b * rng.uniform(0.8, 1.2) + rng.uniform(-10, 10)
    out = b * (1 - R.shadow) * (1 - R.alpha) + R.img * R.alpha
    out[bg == 0] = 0  # keep the black corners left by undistortion
    h, w = out.shape
    if rng.random() < 0.25:  # sunlight patch saturating a corner (seen in the real frames)
        m = np.zeros((h, w), np.float32)
        cx, cy = rng.uniform(0.9, 1.05) * w, rng.uniform(0.75, 1.0) * h  # bottom-right, as in the real frames
        cv2.ellipse(m, (int(cx), int(cy)), (int(rng.uniform(0.08, 0.25) * w), int(rng.uniform(0.15, 0.4) * h)),
                    rng.uniform(0, 180), 0, 360, 1.0, -1)
        m = cv2.GaussianBlur(m, (0, 0), rng.uniform(1, 6))
        out = out * (1 - m) + 255 * m
    out = out * rng.uniform(0.85, 1.15) + rng.uniform(-12, 12)
    out = 255 * np.clip(out / 255, 0, 1) ** rng.uniform(0.85, 1.2)
    if rng.random() < 0.4:
        out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.4, 1.2))
    out += rng.normal(0, rng.uniform(1, 4), out.shape)
    out[bg == 0] = 0
    return np.clip(out, 0, 255).astype(np.uint8)


def make_sample(seed, cam, bg):
    rng = np.random.default_rng(seed)
    R = Renderer(cam)
    if rng.random() < 0.5:  # mirrored floors: more background variety, geometry is unaffected
        flip = int(rng.integers(-1, 2))
        bg = cv2.flip(bg, flip)
    n = 0 if rng.random() < 0.08 else 2 if rng.random() < 0.10 else 1
    objects, placed = [], []
    for _ in range(n):
        o = make_object(rng)
        L, W, _ = overall_dims(o)
        for _try in range(30):
            cx, cy, yaw = sample_pose(rng, L, W, cam)
            uv = cam.project(np.concatenate([R.to3d(unit_footprint(o, cx, cy, yaw), z) for z in (0.0, overall_dims(o)[2])]))
            inside = (uv.min(0) >= 2).all() and (uv.max(0) <= np.array(cam.size) - 2).all()
            if inside and all(math.hypot(cx - px, cy - py) > (L + pl) / 2 + 0.3 for px, py, pl in placed):
                break
        else:
            continue
        placed.append((cx, cy, L))
        rec = render_object(rng, R, o, cx, cy, yaw)
        if o["kind"] == "cage":
            draw_cage_frame(R, rec, rng)
        foot = unit_footprint(o, cx, cy, yaw)
        bottom, top = R.to3d(foot, 0.0), R.to3d(foot, rec["H"])
        objects.append({
            "name": f"unit_{len(objects):02d}",
            "type": "parcel" if o["kind"] == "carton" else "palletised_load",
            "object_dimensions_m": {"width_m": float(np.linalg.norm(foot[1] - foot[0])),
                                    "length_m": float(np.linalg.norm(foot[3] - foot[0])),
                                    "height_m": float(rec["H"])},
            "boxes_3d_cam": np.round(np.concatenate([bottom, top]), 6).tolist(),
            "synth": {"kind": o["kind"], "top_face_3d": np.round(rec["top"], 6).tolist(),
                      "pallet_top_3d": np.round(rec["pallet_top"], 6).tolist() if rec.get("pallet_top") else None,
                      "dmc_3d": np.round(rec["dmc"], 6).tolist() if rec["dmc"] else None,
                      "load_lw_m": o["load"], "load_h_m": o["load_h"], "pallet_lw_m": o["pallet"]},
        })
    if n == 0 or rng.random() < 0.15:  # stray pallet truck on the floor
        cx, cy, yaw = sample_pose(rng, 1.5, 0.6, cam)
        if all(math.hypot(cx - px, cy - py) > (1.5 + pl) / 2 + 0.2 for px, py, pl in placed):
            draw_truck(rng, R, cx, cy, yaw)
    return finish(rng, bg, R), objects


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
                 "valid_region": [0, 0, cam.size[0], cam.size[1]], "seed": int(seed * 1_000_003 + i)}
        with open(os.path.join(shard, f"{i:08d}.json"), "w") as f:
            json.dump(label, f, indent=1)
    return count


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cam-dir", default=CAM_DIR, help="folder with cam_info.json and raw_*.json")
    ap.add_argument("--out", default=os.path.join(ROOT, "datasets", "SYNTH_CAM_000CDF0AD90F"))
    ap.add_argument("--n", type=int, default=4000, help="number of images")
    ap.add_argument("--scale", type=float, default=0.5, help="image scale vs the 2560x2048 sensor")
    ap.add_argument("--canvas", type=int, default=1280, help="square output side; frame goes top-left")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    chunk = 50
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
