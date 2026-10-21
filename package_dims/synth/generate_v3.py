"""Synthetic dataset v4: path-traced with Blender Cycles over the real empty floor (dataset v3 = the same renderer
with the plain v2 scenes; this version adds the shipping units of units_v3.py).

Same output schema, framing and label convention as generate_v2.py. Scenes:
  * 60 % v2 units (cartons, pallet loads, cages), upgraded: stretch film draped over the real load outline in any
    style (clear, bubble wrap, black / white opaque, tinted) instead of a box hull, soft cartons (rounded edges,
    bulging sides, sagging / domed tops, dents) in 35 %, edge protectors + top sheet + straps
  * 40 % units_v3 kinds: sacks, big bags, beverage pallets, crate stacks on dollies, roll containers, film-wrapped
    machines / furniture, octabins, pallet collars, polybags, slatted crates, IBCs, buckets / canisters, long goods
    and (rare, ~8 % of all units) round goods: drums, cable drums, coils, tyres / wheels, paper / carpet rolls, tubes
  * smooth curved surfaces (per-corner normals), shared textures for identical items
  * shipping label = sticker with the DataMatrix on a flat top spot (on top of the film), exact dmc_3d corners
Image formation:
  * physically based light transport (Cycles, GPU): soft area lights + ambient, real soft shadows, contact
    occlusion under/inside pallets and in the gaps between cartons, light bounced from the real floor
  * the floor is a shadow catcher textured with the real background, so shadows and occlusion are applied
    to the real floor image; objects are composited over it with antialiased edges
  * warehouse lighting: mostly ambient + a grid of ceiling lamps (or a few lamps), optional soft dock-door
    light from the bottom right; normalised so a white top at the image centre always gets the same light
  * materials: rough cardboard (bulging faces, handled rims, scuffs), wood, paint, steel, rubber, glossy PE, fabric;
    film = transparent / milky / opaque per style, with patchy crinkles that catch lamp glints
  * pallet truck with round parts: tapered forks, pump, wheels, drawbar, rounded-triangle handle loop
  * camera: linear response, gain, gamma, sun patch, vignetting, slight blur, sensor noise at the real level

Needs Blender >= 4.2 (tested 5.2) on PATH or via --blender; renders on the GPU (OptiX).
CPU workers build scenes / composite while Blender renders (~2 s per image on an RTX 4070 laptop).

    nice -n 10 python3 synth/generate_v3.py --n 3000 --workers 6 --seed 0   # -> datasets/SYNTH_CAM_000CDF0AD90F_v4
Interrupted runs resume: images whose png + json already exist are skipped.
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import generate_v2 as G  # noqa: E402
import units_v3 as U  # noqa: E402
from camera import Camera, pad_square  # noqa: E402
from generate import CAM_DIR, SHARD, rect_corners  # noqa: E402

ALB = 220.0       # image value of a white (albedo 1) horizontal surface under the nominal light, before camera gain
EXPOSURE = 0.5    # blender_render.py stores radiance * EXPOSURE
MAX_ALBEDO = 0.92
# material -> (roughness, metallic, unevenness of the faces)
MATS = {"cardboard": (0.85, 0.0, 1.0), "wood": (0.9, 0.0, 0.6), "plastic": (0.55, 0.0, 0.2), "paint": (0.4, 0.0, 0.1),
        "rubber": (0.7, 0.0, 0.3), "strap": (0.45, 0.0, 0.3), "steel": (0.45, 0.6, 0.1), "panel": (0.6, 0.0, 0.3),
        "matte": (0.8, 0.0, 0.5), "tyre": (0.75, 0.0, 0.25), "glossy": (0.25, 0.0, 0.15), "fabric": (0.95, 0.0, 0.8)}
V2_SHARE = 0.6   # share of v2 units (cartons / pallet loads / cages, upgraded); the rest are units_v3 kinds
SOFT_P = 0.35    # v2 units with soft cartons
PROTECT_P = 0.25  # v2 stacks / mixed / single loads with edge protectors
HIDDEN_RES = 8    # faces the camera cannot see get textures at 1/HIDDEN_RES resolution
MAX_TILE = 3000   # px, longest side of one texture tile
FILM_GLOSS = {"clear": 1.0, "bubble": 0.8, "black": 0.9, "white": 0.5, "tinted": 0.8}


# ---------------------------------------------------------------------------------- scene buffer
def to_blender(q):
    """(N, 3) points in v2 world (x, y = camera axes, height) -> Blender world (x, -y, height)."""
    return np.asarray(q, np.float64) * (1.0, -1.0, 1.0)


def pack_atlas(tiles, pad=2):
    """Shelf-pack (h, w, c) uint8 tiles -> atlas, top-left offsets of every tile."""
    total = sum((t.shape[0] + 2 * pad) * (t.shape[1] + 2 * pad) for t in tiles)
    widest = max(t.shape[1] for t in tiles) + 2 * pad
    W = int(max(widest, min(8192, 2 ** math.ceil(math.log2(max(64.0, math.sqrt(total) * 1.1))))))
    order = sorted(range(len(tiles)), key=lambda i: -tiles[i].shape[0])
    pos, x, y, row = [None] * len(tiles), 0, 0, 0
    for i in order:
        h, w = tiles[i].shape[:2]
        if x + w + 2 * pad > W:
            x, y, row = 0, y + row, 0
        pos[i] = (x + pad, y + pad)
        x, row = x + w + 2 * pad, max(row, h + 2 * pad)
    atlas = np.zeros((y + row, W, tiles[0].shape[2]), np.uint8)
    for t, (px_, py_) in zip(tiles, pos):
        h, w = t.shape[:2]
        atlas[py_ - pad:py_ + h + pad, px_ - pad:px_ + w + pad] = \
            cv2.copyMakeBorder(t, pad, pad, pad, pad, cv2.BORDER_REPLICATE).reshape(h + 2 * pad, w + 2 * pad, -1)
    return atlas, pos


def newell(P):
    """Normal (not normalised) of a planar polygon (k, 3)."""
    return np.cross(P, np.roll(P, -1, 0)).sum(0)


class Group:
    """Textured polygons sharing one Blender material, with per-corner normals (flat polygons: the face normal;
    smooth surfaces: interpolated). Tiles are packed into one atlas; surfaces with the same key share a tile."""

    def __init__(self):
        self.batches, self.tiles, self.keys = [], [], {}

    def tile(self, tile, key=None):
        """-> index of the tile (registered once per key)."""
        if key is not None and key in self.keys:
            return self.keys[key]
        self.tiles.append(tile if tile.ndim == 3 else tile[..., None])
        if key is not None:
            self.keys[key] = len(self.tiles) - 1
        return len(self.tiles) - 1

    def add(self, P_v2, inside_v2, tile, uv_px=None):
        """Flat polygon with its own tile. P_v2: (k, 3); inside_v2: a point behind it (normals point away from it);
        uv_px: (k, 2) tile pixel coords of the vertices (default: quad -> tile corners)."""
        tile = tile if tile.ndim == 3 else tile[..., None]
        if uv_px is None:
            h, w = tile.shape[:2]
            uv_px = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float64)
        Pb, cb = to_blender(P_v2), to_blender(np.asarray(inside_v2)[None])[0]
        n = newell(Pb)
        if n @ (Pb.mean(0) - cb) < 0:  # outward normals (Blender winding)
            Pb, uv_px, n = Pb[::-1], uv_px[::-1], -n
        n = n / (np.linalg.norm(n) + 1e-12)
        self.batches.append((Pb[None], np.asarray(uv_px, np.float64)[None], np.repeat(n[None, None], len(Pb), 1),
                             self.tile(tile)))

    def add_quads(self, Q_v2, N_v2, uv_px, ti):
        """Smooth quads (m, 4, 3) with corner normals (m, 4, 3) and tile pixel coords (m, 4, 2) on tile ti;
        the winding follows the normals."""
        Qb, Nb, uv = to_blender(Q_v2.reshape(-1, 3)).reshape(Q_v2.shape), N_v2 * (1.0, -1.0, 1.0), uv_px.copy()
        nq = np.cross(Qb, np.roll(Qb, -1, 1)).sum(1)
        flip = (nq * Nb.mean(1)).sum(-1) < 0
        Qb[flip], Nb[flip], uv[flip] = Qb[flip, ::-1], Nb[flip, ::-1], uv[flip, ::-1]
        self.batches.append((Qb, uv, Nb, ti))

    def arrays(self, name):
        if not self.batches:
            return {}
        atlas, pos = pack_atlas(self.tiles)
        AH, AW = atlas.shape[:2]
        V, UV, N, S = [], [], [], []
        for Pb, uv, Nb, ti in self.batches:
            x, y = pos[ti]
            m, k = Pb.shape[:2]
            V.append(Pb.reshape(-1, 3))
            N.append(Nb.reshape(-1, 3))
            uv = uv.reshape(-1, 2)
            UV.append(np.c_[(x + uv[:, 0]) / AW, 1.0 - (y + uv[:, 1]) / AH])
            S.append(np.full(m, k, np.int32))
        return {f"{name}_v": np.concatenate(V).astype(np.float32), f"{name}_fs": np.concatenate(S),
                f"{name}_n": np.concatenate(N).astype(np.float32),
                f"{name}_uv": np.concatenate(UV).astype(np.float32), f"{name}_atlas": atlas}


def weather(rng, t):
    """Handled cardboard: broad blotches, darker rims along the edges, a few scuffs (tones, in place)."""
    h, w = t.shape
    if min(h, w) < 12:
        return t
    t += G.smooth_noise(rng, h, w, G.px(0.15)) * rng.uniform(2, 7)
    yy, xx = np.mgrid[0:h, 0:w]
    edge = np.minimum(np.minimum(xx, w - 1 - xx), np.minimum(yy, h - 1 - yy)).astype(np.float32)
    t -= np.exp(-edge / G.px(rng.uniform(0.01, 0.04))) * rng.uniform(0, 14)
    for _ in range(rng.poisson(1.2)):
        p0 = np.array([rng.uniform(0, w), rng.uniform(0, h)])
        ang, ln = rng.uniform(0, math.pi), G.px(rng.uniform(0.02, 0.15))
        p1 = p0 + ln * np.array([math.cos(ang), math.sin(ang)])
        m = np.zeros_like(t)
        cv2.line(m, tuple(p0.astype(int)), tuple(p1.astype(int)), 1.0, int(rng.integers(1, 3)), cv2.LINE_AA)
        t += cv2.GaussianBlur(m, (0, 0), 0.8) * rng.choice([-1, 1]) * rng.uniform(8, 25)
    return t


def solid_tile(t, mat):
    """Tone texture -> RGBA tile (albedo, roughness, metallic, unevenness)."""
    alb = np.clip(t / ALB, 0, MAX_ALBEDO) * 255
    return np.round(np.stack([alb] + [np.full_like(alb, v * 255) for v in MATS.get(mat, MATS["matte"])], -1)
                    ).astype(np.uint8)


def film_tile(rng, w, h, f):
    """Film surface tile: R = milky haze (crinkles edge-biased, rope at the bottom of the wrap), G = film albedo,
    B = gloss amount; f = film dict from units_v3.film_wrap (style, tone, base, gain)."""
    _, a = G.film_tex(rng, w, h, f["kind"] == "top")
    if f["kind"] == "side":
        a = a[::-1]  # tile rows run bottom -> top on the wrap; film_tex puts the bunched rope at the last rows
    haze = f["base"] + a * f["gain"]
    if f["style"] == "bubble":  # air cushions: a hexagonal grid of small rings
        p = max(4, G.px(0.012))
        cell = np.zeros((2 * p, p), np.float32)
        for cx, cy in ((p / 2, p / 2), (0, 1.5 * p), (p, 1.5 * p)):
            cv2.circle(cell, (int(cx), int(cy)), max(1, p // 3), 1.0, 1, cv2.LINE_AA)
        b = np.tile(cell, (h // (2 * p) + 1, w // p + 1))[:h, :w]
        haze = haze + b * 0.15
    return np.round(np.stack([np.clip(haze, 0, 1), np.full_like(a, f["tone"]),
                              np.full_like(a, FILM_GLOSS[f["style"]])], -1) * 255).astype(np.uint8)


def grid_quads(A):
    """(a, b, c) grid -> (m, 4, c) quads (i, j), (i, j+1), (i+1, j+1), (i+1, j)."""
    return np.stack([A[:-1, :-1], A[:-1, 1:], A[1:, 1:], A[1:, :-1]], 2).reshape(-1, 4, A.shape[-1])


class SceneBuf:
    def __init__(self, cam, rng):
        self.cam_pos = np.array([0.0, 0.0, cam.mount_h])
        self.rng = rng
        self.solid, self.film, self.mesh = Group(), Group(), Group()

    def facing(self, P, inside):
        n = newell(P)
        n = n if n @ (P.mean(0) - inside) > 0 else -n
        return n @ (P.mean(0) - self.cam_pos) < 0

    @staticmethod
    def tex(fn, fi, w, h, visible):
        if not visible:
            w, h = max(4, w // HIDDEN_RES), max(4, h // HIDDEN_RES)
        t = fn(fi, w, h)
        return t[0] if isinstance(t, tuple) else t

    def add_parts(self, parts, yaw, cx, cy):
        """v2 box parts (P = 8 corners), polygon parts (polys = [(poly (k, 3), inside point)]) and smooth surfaces
        (surf = (a, b, 3) grid with normals N and uv in metres, units_v3)."""
        cart = [p for p in parts if p.get("cull") and "surf" not in p and "polys" not in p]
        skip_local = G.cull_fn(cart) if cart else None
        for p in parts:
            mat = p.get("mat", "matte")
            if "surf" in p:
                self.add_surf(p, yaw, cx, cy)
                continue
            if "polys" in p:
                for fi, (poly, inside) in enumerate(p["polys"]):
                    poly = G.transform(poly, yaw, cx, cy)
                    inside = G.transform(np.asarray(inside, np.float64)[None], yaw, cx, cy)[0]
                    self.add_poly(poly, inside, p["tex"], fi, mat)
                continue
            Pw = G.transform(p["P"], yaw, cx, cy)
            cen = Pw.mean(0)
            for fi, idx in enumerate(G.FACES):
                q = Pw[list(idx)]
                if np.linalg.norm(newell(q)) < 1e-12:
                    continue
                hidden = fi == 5 or (p["cull"] and skip_local is not None
                                     and skip_local(p["P"][list(idx)].mean(0), fi, p["P"]))
                visible = not hidden and self.facing(q, cen)
                if fi == 0 and p["top_tex"] is not None:
                    t = p["top_tex"].copy()
                else:
                    t = self.tex(p["tex"], fi, G.px(np.linalg.norm(q[1] - q[0])), G.px(np.linalg.norm(q[3] - q[0])),
                                 visible)
                if mat == "cardboard" and visible:
                    t = weather(self.rng, np.asarray(t, np.float32))
                self.solid.add(q, cen, solid_tile(t, mat))

    def add_poly(self, P, inside, tex_fn, fi, mat):
        """Planar polygon with a texture tile spanning its bounding box (in-plane axes: first edge, normal x it)."""
        n = newell(P)
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            return
        e1 = P[1] - P[0]
        e1 /= np.linalg.norm(e1) + 1e-12
        e2 = np.cross(n / nn, e1)
        st = np.c_[(P - P[0]) @ e1, (P - P[0]) @ e2]
        st -= st.min(0)
        w, h = (G.px(v) for v in st.max(0))
        t = self.tex(tex_fn, fi, w, h, self.facing(P, inside))
        uv = st * (t.shape[1] / max(st[:, 0].max(), 1e-9), t.shape[0] / max(st[:, 1].max(), 1e-9))
        self.solid.add(P, inside, solid_tile(t, mat), uv)

    def surf_geometry(self, P, N, yaw, cx, cy):
        a, b = P.shape[:2]
        Pw = G.transform(P.reshape(-1, 3), yaw, cx, cy).reshape(a, b, 3)
        Nw = G.transform(N.reshape(-1, 3), yaw, 0.0, 0.0).reshape(a, b, 3)
        Q, NQ = grid_quads(Pw), grid_quads(Nw)
        ok = np.isfinite(Q).all((1, 2)) & np.isfinite(NQ).all((1, 2)) & \
            (np.linalg.norm(np.cross(Q, np.roll(Q, -1, 1)).sum(1), axis=1) > 1e-10)
        visible = bool(((Nw * (Pw - self.cam_pos)).sum(-1) < 0).any())
        return Q, NQ, ok, visible

    @staticmethod
    def surf_uv(uv, w, h):
        uv = uv - uv.reshape(-1, 2).min(0)
        span = np.maximum(uv.reshape(-1, 2).max(0), 1e-9)
        return grid_quads(uv * (w / span[0], h / span[1])), span

    def add_surf(self, p, yaw, cx, cy):
        """Smooth surface: one texture tile over the whole (u, v) grid, shared by surfaces with the same key."""
        Q, NQ, ok, visible = self.surf_geometry(p["surf"], p["N"], yaw, cx, cy)
        if not ok.any():
            return
        span = np.ptp(p["uv"].reshape(-1, 2), 0)
        w, h = (min(MAX_TILE, G.px(v)) for v in span)
        key = p.get("key")
        visible = visible and not p.get("hidden")
        if key is not None and key in self.solid.keys:
            ti = self.solid.keys[key]
        else:
            t = self.tex(p["tex"], p.get("face", 0), w, h, visible or key is not None)
            if p.get("mat") == "cardboard" and visible:
                t = weather(self.rng, np.asarray(t, np.float32))
            ti = self.solid.tile(solid_tile(np.asarray(t, np.float32), p.get("mat", "matte")), key)
        th, tw = self.solid.tiles[ti].shape[:2]
        UV, _ = self.surf_uv(p["uv"], tw, th)
        self.solid.add_quads(Q[ok], NQ[ok], UV[ok], ti)

    def add_films(self, films, yaw, cx, cy):
        """Stretch film surfaces (units_v3.film_wrap): own tile each (haze, albedo, gloss)."""
        for f in films:
            Q, NQ, ok, _ = self.surf_geometry(f["P"], f["N"], yaw, cx, cy)
            if not ok.any():
                continue
            span = np.ptp(f["uv"].reshape(-1, 2), 0)
            w, h = (min(MAX_TILE, G.px(v)) for v in span)
            ti = self.film.tile(film_tile(self.rng, w, h, f))
            UV, _ = self.surf_uv(f["uv"], w, h)
            self.film.add_quads(Q[ok], NQ[ok], UV[ok], ti)

    def add_quads(self, rng, quads, yaw, cx, cy):
        if not quads:
            return
        Qs = [G.transform(q[1], yaw, cx, cy) for q in quads]
        cen = np.concatenate(Qs).mean(0)
        for q, Q in zip(quads, Qs):
            w_px, h_px = G.px(np.linalg.norm(Q[1] - Q[0])), G.px(np.linalg.norm(Q[3] - Q[0]))
            if q[0] == "film":  # v2 film hull quad: clear film, tile y runs top -> bottom already
                _, a = G.film_tex(rng, w_px, h_px, q[2])
                haze = np.clip(rng.uniform(0.0, 0.04) + a * rng.uniform(0.6, 1.2), 0, 1)
                self.film.add(Q, cen, np.round(np.stack([haze, np.full_like(a, 0.85), np.ones_like(a)], -1) * 255
                                               ).astype(np.uint8))
            else:
                t, a = G.mesh_tex(rng, w_px, h_px, q[2], q[3])
                alb = np.clip(t / ALB, 0, MAX_ALBEDO) * 255
                self.mesh.add(Q, cen, np.round(np.stack([alb, a * 255], -1)).astype(np.uint8))

    def arrays(self):
        out = {}
        for name in ("solid", "film", "mesh"):
            out.update(getattr(self, name).arrays(name))
        return out


# ---------------------------------------------------------------------------------- lighting
def sample_lights(rng):
    """Ambient (world) + ceiling lamps (+ daylight from the dock door at the bottom right of the image).

    Powers are set so that a white horizontal surface at the image centre receives ~1.0 in total
    (Cycles: area light of power P at distance d -> P cos(a) cos(b) / (pi^2 d^2) on a white Lambertian).
    The lamps also give the specular glints on film, tape and paint."""
    amb = rng.uniform(0.6, 0.88)  # real frames: soft, mostly ambient light, faint shadows
    lamps = []  # x, y, z, size_x, size_y (pointing down)
    if rng.random() < 0.85:  # regular grid of high-bay lamps / light strips
        dx, dy, z = rng.uniform(2.5, 5), rng.uniform(2.5, 6), rng.uniform(6.5, 10)
        sx, sy = rng.uniform(0.3, 1.4), rng.uniform(0.1, 0.5)
        a, ox, oy = rng.choice([0.0, math.pi / 2]) + rng.normal(0, 0.05), rng.uniform(0, dx), rng.uniform(0, dy)
        for i in range(-4, 5):
            for j in range(-4, 5):
                x, y = ox + i * dx, oy + j * dy
                x, y = x * math.cos(a) - y * math.sin(a), x * math.sin(a) + y * math.cos(a)
                if math.hypot(x, y) < 10 and rng.random() < 0.9:  # some lamps are off
                    lamps.append([x, y, z, sx, sy])
    for _ in range(int(rng.integers(0 if lamps else 2, 5))):  # extra lamps / skylights
        sx, sy = rng.uniform(0.5, 2.5), rng.uniform(0.3, 1.2)
        lamps.append([rng.uniform(-4, 4), rng.uniform(-4, 4), rng.uniform(6.5, 9.5)] + ([sx, sy] if rng.random() < 0.5
                                                                                      else [sy, sx]))
    lamps = np.array(lamps)
    d2 = (lamps[:, :3] ** 2).sum(1)
    w = rng.uniform(0.5, 1.5, len(lamps))  # relative lamp powers
    contrib = w * (lamps[:, 2] ** 2 / d2) / (math.pi ** 2 * d2)  # per unit power, on a white top at the origin
    power = w * (1 - amb) / contrib.sum()
    rows = [[x, y, z, sx, sy, P, x, y, 0.0] for (x, y, z, sx, sy), P in zip(lamps, power)]
    if rng.random() < 0.35:  # dock door: big, soft, mostly lights the faces turned towards it
        ang = -math.pi / 4 + rng.uniform(-0.6, 0.6)  # Blender (+x, -y) = image bottom-right
        dist, z = rng.uniform(6, 10), rng.uniform(1.5, 3.0)
        d2 = dist * dist + z * z
        share = rng.uniform(0.02, 0.08)
        rows.append([dist * math.cos(ang), dist * math.sin(ang), z, rng.uniform(3, 6), rng.uniform(2.5, 4),
                     share * math.pi ** 2 * d2 / (z / math.sqrt(d2)), 0.0, 0.0, 1.0])
    return amb, np.array(rows, np.float64)


# ---------------------------------------------------------------------------------- scene
def w2c(cam, P):
    P = np.asarray(P, np.float64)
    return np.c_[P[:, :2], cam.mount_h - P[:, 2]]


def floor_clutter(rng):
    """Flat litter on the floor: paper / label scraps, cardboard pieces, dark rubber or plastic bits."""
    parts = []
    for _ in range(rng.integers(1, 4)):
        x, y = rng.uniform(-2.5, 2.5), rng.uniform(-2, 2)
        l, w = rng.uniform(0.04, 0.3), rng.uniform(0.02, 0.2)
        tone = rng.choice([rng.uniform(170, 230), rng.uniform(90, 150), rng.uniform(20, 45)], p=[0.5, 0.3, 0.2])
        parts.append(G.part(G.prism(x, y, l, w, 0.0, rng.uniform(0.001, 0.004), rng.uniform(0, math.pi)),
                            lambda f, ww, hh, tone=tone: G.plain(rng, ww, hh, tone, 3, 6), mat="matte"))
    return parts


def v2_unit(rng):
    """v2 carton / pallet load / cage, upgraded: stretch film draped over the real load outline in any style (clear,
    bubble, black, white, tinted) instead of a box hull, sometimes soft cartons (rounded, bulging, sagging) and
    edge protectors + top sheet. When anything changed, the label is a sticker on top (on the film)."""
    u = G.make_unit(rng)
    u["parts"] = [p for p in u["parts"] if p.get("dmc_ok", True)]  # v2's dark core: path tracing shadows the gaps
    base = G.PALLET_H if u["pallet"] and u["kind"] != "cage" else 0.0
    film_q = [q for q in u["quads"] if q[0] == "film"]
    u.update(quads=[q for q in u["quads"] if q[0] != "film"], films=[], labels=None, carrier_h=base)
    if u["kind"] == "cage":
        return u
    straps = [p for p in u["parts"] if p.get("mat") == "strap"]
    rest = [p for p in u["parts"] if p.get("mat") != "strap"]
    load = [p for p in rest if p["P"][:, 2].min() >= base - 1e-6]
    pallet = [p for p in rest if p["P"][:, 2].min() < base - 1e-6]
    changed = bool(film_q)
    if rng.random() < SOFT_P and any(p.get("mat") == "cardboard" for p in load):
        load = U.soften(rng, load)
        changed = True
    if u["kind"] in ("stack", "mixed", "single") and rng.random() < PROTECT_P:
        lo, hi = U.extent(load)
        load += U.edge_protectors(rng, lo[:2], hi[:2], base, hi[2])
        if not straps:
            straps = G.strap_parts(rng, lo[:2] - 0.006, hi[:2] + 0.006, base, hi[2] + 0.004, int(rng.integers(2)))
        changed = True
    top_film = None
    if film_q:
        top_film = (min(q[1][:, 2].min() for q in film_q), max(q[1][:, 2].max() for q in film_q) - 0.008,
                    any(q[2] for q in film_q))
    elif u["kind"] == "carton" and rng.random() < 0.1:  # single carton wrapped in film
        top_film = (0.0, U.extent(load)[1][2], rng.random() < 0.6)
    if top_film:
        z0, z1, top = top_film
        u["films"] = U.film_wrap(rng, U.comps_of(load), max(0.02, z0 - rng.uniform(0, 0.06)), z1, top)
        changed = True
    if changed:
        u["labels"] = U.top_label_spots(rng, load, U.extent(load)[1][2])
    u["parts"] = pallet + load + straps
    return u


def make_unit(rng):
    """One shipping unit: v2 cartons / pallet loads / cages (upgraded) or one of the units_v3 kinds."""
    return v2_unit(rng) if rng.random() < V2_SHARE else U.make_unit(rng)


def unit_extent(u):
    """Overall footprint (lo, hi) in the local frame and total height, from parts, mesh quads and film."""
    pts = np.concatenate([p["P"] for p in u["parts"]] + [q[1] for q in u["quads"]] +
                         [f["P"].reshape(-1, 3) for f in u["films"]])
    return pts[:, :2].min(0), pts[:, :2].max(0), float(pts[:, 2].max())


def add_label(rng, u, H):
    """Shipping label with the DataMatrix -> local (4, 3) symbol corners or None. v2 units without changes get it
    printed into a top face texture; the others a sticker on a flat spot, on top of the film if there is one."""
    if u["labels"] is None:
        return G.add_dmc(rng, u, H)
    if not u["labels"]:
        return None
    x, y, z, ms = u["labels"][rng.integers(len(u["labels"]))]
    # lie on the highest point under the sticker (domed sacks, dents, straps), ignoring much taller neighbours
    pts = np.concatenate([p["P"] for p in u["parts"]])
    near = (np.hypot(pts[:, 0] - x, pts[:, 1] - y) < 0.09) & (pts[:, 2] < z + 0.03)
    if near.any():
        z = max(z, float(pts[near, 2].max()) + 0.001)
    fz = U.film_top_z(u["films"], x, y)
    if fz is not None:
        z = max(z, fz + 0.002)
    part, dmc = U.label_part(rng, x, y, z, max_size=max(ms, 0.12))
    u["parts"].append(part)
    return dmc


def make_scene(seed, cam, samples):
    """-> (scene arrays for blender_render.py, objects[] labels, flip code)."""
    rng = np.random.default_rng(seed)
    S = SceneBuf(cam, rng)
    flip = int(rng.integers(-1, 2)) if rng.random() < 0.5 else -2
    n = 0 if rng.random() < 0.08 else 2 if rng.random() < 0.10 else 1
    objects, placed = [], []
    for _ in range(n):
        u = make_unit(rng)
        dmc = add_label(rng, u, unit_extent(u)[2]) if (u["kind"] != "cage" and rng.random() < 0.9) else None
        lo, hi, H = unit_extent(u)
        truck = None
        if u["pallet"] and u["kind"] != "cage" and rng.random() < 0.7:
            L = u["pallet"][0]
            end = 1 if rng.random() < 0.5 else -1
            root = L / 2 + (rng.uniform(0.0, 0.08) if rng.random() < 0.75 else rng.uniform(0.1, 0.6))
            truck = (U.build_truck(rng), end, root, rng.normal(0, 0.02))
        reach = math.hypot(*(hi - lo)) / 2 + (1.4 if truck else 0.2)
        foot_local = rect_corners(*(lo + hi) / 2, *(hi - lo), 0.0)
        for _try in range(30):
            cx, cy, yaw = G.sample_pose(rng, cam)
            foot = G.transform(np.c_[foot_local, np.zeros(4)], yaw, cx, cy)[:, :2]
            uv = cam.project(np.concatenate([w2c(cam, np.c_[foot, np.full(4, z)]) for z in (0.0, H)]))
            inside = (uv.min(0) >= 2).all() and (uv.max(0) <= np.array(cam.size) - 2).all()
            if inside and all(math.hypot(cx - px_, cy - py_) > reach + r_ for px_, py_, r_ in placed):
                break
        else:
            continue
        placed.append((cx, cy, reach))
        S.add_parts(u["parts"], yaw, cx, cy)
        if truck:
            parts, end, root, dy = truck
            tyaw = yaw + (math.pi if end > 0 else 0.0)
            tx, ty = G.transform(np.array([[end * root, dy, 0.0]]), yaw, cx, cy)[0, :2]
            S.add_parts(parts, tyaw, tx, ty)
        S.add_quads(rng, u["quads"], yaw, cx, cy)
        S.add_films(u["films"], yaw, cx, cy)
        bottom, top = w2c(cam, np.c_[foot, np.zeros(4)]), w2c(cam, np.c_[foot, np.full(4, H)])
        pal_top = None
        if u["pallet"]:
            pal_top = w2c(cam, G.transform(np.c_[rect_corners(0, 0, *u["pallet"], 0), np.full(4, G.PALLET_H)],
                                           yaw, cx, cy))
        objects.append({
            "name": f"unit_{len(objects):02d}",
            "type": u["type"],
            "object_dimensions_m": {"width_m": float(np.linalg.norm(foot[1] - foot[0])),
                                    "length_m": float(np.linalg.norm(foot[3] - foot[0])),
                                    "height_m": H},
            "boxes_3d_cam": np.round(np.concatenate([bottom, top]), 6).tolist(),
            "synth": {"kind": u["kind"], "top_face_3d": np.round(top, 6).tolist(),
                      "pallet_top_3d": np.round(pal_top, 6).tolist() if pal_top is not None else None,
                      "dmc_3d": np.round(w2c(cam, G.transform(dmc, yaw, cx, cy)), 6).tolist() if dmc is not None
                      else None,
                      "load_lw_m": u["load"], "load_h_m": H - u["carrier_h"],
                      "pallet_lw_m": u["pallet"], "truck": truck is not None,
                      "film": u["films"][0]["style"] if u["films"] else None,
                      "soft": any(p.get("soft") for p in u["parts"])},
        })
    if n == 0 or rng.random() < 0.15:  # stray pallet truck
        cx, cy, yaw = G.sample_pose(rng, cam)
        if all(math.hypot(cx - px_, cy - py_) > 1.0 + r_ for px_, py_, r_ in placed):
            S.add_parts(U.build_truck(rng), yaw, cx, cy)
    if rng.random() < 0.2:
        S.add_parts(floor_clutter(rng), 0.0, 0.0, 0.0)
    amb, lights = sample_lights(rng)
    scene = S.arrays()
    scene.update(cam=np.array([cam.fx, cam.fy, cam.cx, cam.cy, cam.size[0], cam.size[1], cam.mount_h]),
                 lights=lights.reshape(-1, 9), world=np.float64(amb), flip=np.int64(flip),
                 samples=np.int64(samples), seed=np.int64(seed),
                 # film (haze, albedo, gloss come per surface from the atlas): noise scale, bump strength, gloss
                 # roughness, gloss gain, wrap bands, fold radius over the load edges, crinkle patch threshold
                 film_params=np.array([rng.uniform(25, 80), rng.uniform(0.5, 1.0), rng.uniform(0.03, 0.12),
                                       rng.uniform(0.8, 1.0), rng.uniform(15, 50), rng.uniform(0.02, 0.06),
                                       rng.uniform(0.4, 0.6)]))
    return scene, objects, flip


# ---------------------------------------------------------------------------------- camera / compositing
def composite(rng, rgba, bg, flip):
    """16-bit RGBA render (straight alpha) over the real floor + camera effects -> uint8 image."""
    if flip != -2:
        bg = cv2.flip(bg, flip)
    a = rgba[..., 3].astype(np.float32) / 65535
    rad = rgba[..., :3].astype(np.float32).mean(-1) / 65535 / EXPOSURE
    b = bg.astype(np.float32)
    if rng.random() < 0.5:
        b = b * rng.uniform(0.85, 1.15) + rng.uniform(-8, 8)
    out = rad * a * ALB * rng.uniform(0.9, 1.1) + b * (1 - a)
    h, w = out.shape
    if rng.random() < 0.8:  # 15 of the 19 real frames have it
        m = G.sun_patch(rng, w, h)
        if flip != -2:
            m = cv2.flip(m, flip)
        out = out * (1 - m) + 255 * m
    if rng.random() < 0.3:  # vignetting
        yy, xx = np.mgrid[0:h, 0:w]
        r2 = ((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2
        out *= 1 - rng.uniform(0.05, 0.2) * r2 / 2
    out = out * rng.uniform(0.9, 1.1) + rng.uniform(-6, 6)
    out = 255 * np.clip(out / 255, 0, 1) ** rng.uniform(0.9, 1.12)
    if rng.random() < 0.5:
        out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.3, 0.7))
    # sensor noise, shot ~ sqrt(signal) + read; real frames (2x downsampled): sigma ~0.8 grey levels on the floor
    out += rng.normal(0, 1, out.shape) * np.sqrt(rng.uniform(0.002, 0.012) * np.clip(out, 0, None) + rng.uniform(0.1, 0.5))
    out[bg == 0] = 0
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------------- orchestration
_W = {}


def init_worker(cam_dir, scale):
    _W["cam"] = Camera(cam_dir, scale)
    _W["bg"] = load_background(_W["cam"])


def load_background(cam):
    bg = cv2.resize(cv2.imread(os.path.join(ROOT, "assets", "background_median.png"), cv2.IMREAD_GRAYSCALE),
                    cam.size, interpolation=cv2.INTER_AREA)
    return cam.undistort_image(bg)


def paths(out, i):
    shard = os.path.join(out, f"shard_{i // SHARD * SHARD:08d}")
    return os.path.join(shard, f"{i:08d}.gray.png"), os.path.join(shard, f"{i:08d}.json")


def build_job(i, seed, work, samples):
    """Worker: scene i -> work/{i}.npz (atomically); returns (objects, flip) or None on failure."""
    try:
        scene, objects, flip = make_scene(seed * 1_000_003 + i, _W["cam"], samples)
        tmp = os.path.join(work, f"{i:08d}.npz.tmp")
        with open(tmp, "wb") as f:
            np.savez(f, **scene)
        os.replace(tmp, os.path.join(work, f"{i:08d}.npz"))
        return objects, flip
    except Exception as e:  # never block the renderer on one bad scene
        print(f"scene {i} failed: {e!r}", flush=True)
        open(os.path.join(work, f"{i:08d}.skip"), "w").close()
        return None


def finish_job(i, seed, work, out, canvas, objects, flip):
    """Worker: render + real floor -> final png + json."""
    cam, bg = _W["cam"], _W["bg"]
    render = os.path.join(work, f"{i:08d}.render.png")
    rgba = cv2.imread(render, cv2.IMREAD_UNCHANGED)
    img = composite(np.random.default_rng([seed, i, 7]), rgba, bg, flip)
    png, js = paths(out, i)
    os.makedirs(os.path.dirname(png), exist_ok=True)
    cv2.imwrite(png, pad_square(img, canvas))
    label = {"camera_intrinsics": cam.intrinsics_json(canvas), "num_objects": len(objects), "objects": objects,
             "camera_pose": {"height_m": cam.mount_h, "tilt_deg": 0.0, "roll_deg": 0.0},
             "valid_region": [0, 0, cam.size[0], cam.size[1]], "seed": int(seed * 1_000_003 + i),
             "generator": "synth/generate_v3.py (v4 scenes, units_v3)"}
    with open(js + ".tmp", "w") as f:
        json.dump(label, f, indent=1)
    os.replace(js + ".tmp", js)
    os.remove(render)
    return i


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cam-dir", default=CAM_DIR, help="folder with cam_info.json and raw_*.json")
    ap.add_argument("--out", default=os.path.join(ROOT, "datasets", "SYNTH_CAM_000CDF0AD90F_v4"))
    ap.add_argument("--n", type=int, default=3000, help="number of images")
    ap.add_argument("--start", type=int, default=0, help="first image index")
    ap.add_argument("--scale", type=float, default=0.5, help="image scale vs the 2560x2048 sensor")
    ap.add_argument("--canvas", type=int, default=1280, help="square output side; frame goes top-left")
    ap.add_argument("--workers", type=int, default=6, help="CPU processes building scenes / compositing")
    ap.add_argument("--blender", default=shutil.which("blender") or "blender")
    ap.add_argument("--blender-procs", type=int, default=2, help="concurrent Blender renderers (overlap CPU/GPU)")
    ap.add_argument("--device", default="OPTIX", help="Cycles device: OPTIX | CUDA | CPU")
    ap.add_argument("--samples", type=int, default=48, help="Cycles samples per pixel (adaptive, denoised)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep-work", action="store_true", help="keep the work folder (scenes / raw renders)")
    args = ap.parse_args()

    todo = [i for i in range(args.start, args.start + args.n) if not all(map(os.path.exists, paths(args.out, i)))]
    print(f"{args.n - len(todo)} done already, {len(todo)} to render", flush=True)
    if not todo:
        return
    work = os.path.join(args.out, "_work")
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    cam = Camera(args.cam_dir, args.scale)
    np.save(os.path.join(work, "floor_albedo.npy"), load_background(cam).astype(np.float32) / ALB)
    with open(os.path.join(work, "todo.txt"), "w") as f:
        f.write("\n".join(map(str, todo)) + "\n")

    procs = []
    for k in range(args.blender_procs):
        log = open(os.path.join(work, f"blender_{k}.log"), "w")
        procs.append(subprocess.Popen(
            [args.blender, "-b", "--factory-startup", "-noaudio", "--python-exit-code", "1", "-P", os.path.join(HERE, "blender_render.py"), "--",
             "--work", work, "--worker", str(k), "--nworkers", str(args.blender_procs), "--device", args.device],
            stdout=log, stderr=subprocess.STDOUT))

    from multiprocessing import Pool
    pool = Pool(args.workers, initializer=init_worker, initargs=(args.cam_dir, args.scale))
    window = max(8, 3 * args.workers)
    built, finishing = {}, []
    nxt, done, t0 = 0, 0, time.time()
    try:
        while done < len(todo):
            while nxt < len(todo) and nxt - done < window:
                built[todo[nxt]] = pool.apply_async(build_job, (todo[nxt], args.seed, work, args.samples))
                nxt += 1
            progressed = False
            for i in list(built):  # renders finish out of order (several Blender processes)
                r = built.get(i)
                if r is None or not r.ready():
                    continue
                res = r.get()
                if res is None:
                    del built[i]
                    finishing.append((i, None))
                    progressed = True
                elif os.path.exists(os.path.join(work, f"{i:08d}.render.png")):
                    del built[i]
                    finishing.append((i, pool.apply_async(finish_job, (i, args.seed, work, args.out, args.canvas, *res))))
                    progressed = True
            while finishing and (finishing[0][1] is None or finishing[0][1].ready()):
                i, r = finishing.pop(0)
                if r is not None:
                    r.get()
                done += 1
                if done % 25 == 0 or done == len(todo):
                    dt = time.time() - t0
                    print(f"{done}/{len(todo)}  {dt / done:.2f} s/img  eta {dt / done * (len(todo) - done) / 60:.0f} min",
                          flush=True)
            if not progressed:
                codes = [p.poll() for p in procs]
                if any(c not in (None, 0) for c in codes) or (None not in codes and not finishing):
                    raise RuntimeError(f"Blender exited (codes {codes}), see {work}/blender_*.log")
                time.sleep(0.05)
    finally:
        open(os.path.join(work, "STOP"), "w").close()
        for p in procs:
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill()
        pool.terminate()
    if not args.keep_work:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
