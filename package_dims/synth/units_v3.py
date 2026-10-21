"""Shipping units for the path-traced generator (generate_v3.py) beyond v2's cartons / pallet loads / cages.

A Dachser cross-dock does not only see cartons: drums, IBCs, cable drums and coils, tyres and wheels, sacks and
big bags, paper and carpet rolls, buckets / canisters, beverage pallets, crate stacks on dollies, roll containers,
long goods on bearers and irregular items (machines, furniture) under clear, black, white or bubble film.

Builders return units in the generate_v2.make_unit format (local frame: x = length, y = width, z = height above
the floor; one entry per physical part) plus
    films   stretch film around the load: surfaces that follow its real outline (rounded, draped over uneven tops)
    labels  flat spots on top for the shipping label: (x, y, z, max_size_m)
Part kinds (all carry "P" = points for the extent, "mat", "tex"):
    box   v2 part, P = 8 corners
    poly  {"polys": [(polygon (k, 3), point inside)]}  flat-shaded planar polygons
    surf  {"surf": (a, b, 3) grid, "N": normals, "uv": (a, b, 2) metres}  smooth-shaded, one texture over the grid;
          parts with the same "key" share one texture (identical items)
"""
import math

import cv2
import numpy as np

import generate_v2 as G
from generate import rect_corners

PALLET_H = G.PALLET_H
px = G.px
U = None  # set per call: rng.uniform


# ---------------------------------------------------------------------------------- transforms
def rotx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def roty(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rotz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def xform(parts, R=None, t=(0.0, 0.0, 0.0)):
    """Rotate (3x3) then translate parts in place; returns them."""
    R = np.eye(3) if R is None else np.asarray(R, np.float64)
    t = np.asarray(t, np.float64)
    for p in parts:
        if "surf" in p:
            p["surf"] = p["surf"] @ R.T + t
            p["N"] = p["N"] @ R.T
            p["P"] = p["surf"].reshape(-1, 3)
        elif "polys" in p:
            p["polys"] = [(q @ R.T + t, c @ R.T + t) for q, c in p["polys"]]
            p["P"] = np.concatenate([q for q, _ in p["polys"]])
        else:
            p["P"] = p["P"] @ R.T + t
    return parts


def extent(parts):
    pts = np.concatenate([p["P"] for p in parts])
    return pts.min(0), pts.max(0)


# ---------------------------------------------------------------------------------- textures
def plain(rng, tone, noise=3.0, mottle=5.0):
    return lambda f, w, h: G.plain(rng, w, h, tone, noise, mottle)


def stripes(t, pos_px, width_px, tone, axis=0):
    """Paint bands of constant v (axis 0: rows) or u (axis 1: columns) into t."""
    for p in pos_px:
        a, b = int(round(p - width_px / 2)), int(round(p + width_px / 2)) + 1
        if axis == 0:
            t[max(0, a):max(0, b)] = tone
        else:
            t[:, max(0, a):max(0, b)] = tone
    return t


def cumlen(pr):
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(np.asarray(pr, np.float64), axis=0), axis=1))]


def hazard_diamond(t, cx, cy, size, rng):
    """ADR-style diamond label (white, dark border, pictogram blob) centred at (cx, cy) px."""
    s = size / 2
    pts = np.array([[cx, cy - s], [cx + s, cy], [cx, cy + s], [cx - s, cy]], np.int32)
    cv2.fillConvexPoly(t, pts, float(rng.uniform(190, 235)), cv2.LINE_AA)
    cv2.polylines(t, [pts], True, float(rng.uniform(20, 60)), max(1, int(size * 0.05)), cv2.LINE_AA)
    cv2.ellipse(t, (int(cx), int(cy - s * 0.25)), (max(1, int(s * 0.3)), max(1, int(s * 0.2))), 0, 0, 360,
                float(rng.uniform(20, 60)), -1, cv2.LINE_AA)


# ---------------------------------------------------------------------------------- surfaces
def surf(P, N, uv, tex, mat, key=None):
    P = np.asarray(P, np.float64)
    return {"surf": P, "N": np.asarray(N, np.float64), "uv": np.asarray(uv, np.float64), "tex": tex, "mat": mat,
            "key": key, "P": P.reshape(-1, 3), "cull": False, "top_tex": None}


def grid_normals(P, closed_u=False, center=None):
    """Numeric normals of a (rows, cols, 3) grid: cross(d/dcol, d/drow); flipped away from `center` if given."""
    du = np.gradient(P, axis=1)
    if closed_u:
        du[:, 0] = du[:, -1] = (P[:, 1] - P[:, -2]) / 2
    dv = np.gradient(P, axis=0)
    N = np.cross(du, dv)
    n = np.linalg.norm(N, axis=-1, keepdims=True)
    bad = n[..., 0] < 1e-12
    N = N / np.maximum(n, 1e-12)
    if bad.any():  # poles: borrow from the neighbouring row
        for i in np.unique(np.nonzero(bad)[0]):
            j = i + 1 if i + 1 < len(P) else i - 1
            N[i] = np.where(bad[i][:, None], N[j], N[i])
    if center is not None:
        s = np.sign(((P - center) * N).sum(-1, keepdims=True))
        N = N * np.where(s == 0, 1, s)
    return N


def grid_uv(P):
    """Arc-length coordinates (metres) of a grid: u along columns, v along rows (averaged over the other axis)."""
    du = np.linalg.norm(np.diff(P, axis=1), axis=-1).mean(0)
    dv = np.linalg.norm(np.diff(P, axis=0), axis=-1).mean(1)
    u, v = np.r_[0, np.cumsum(du)], np.r_[0, np.cumsum(dv)]
    return np.stack(np.broadcast_arrays(u[None, :], v[:, None]), -1)


def sharpen(pr, angle_deg=35.0):
    """Duplicate profile vertices where the profile turns by more than angle_deg (-> sharp edge when smooth-shaded)."""
    pr = np.asarray(pr, np.float64)
    out = [pr[0]]
    for i in range(1, len(pr) - 1):
        a, b = pr[i] - pr[i - 1], pr[i + 1] - pr[i]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        out.append(pr[i])
        if na > 1e-9 and nb > 1e-9 and np.dot(a, b) / (na * nb) < math.cos(math.radians(angle_deg)):
            out.append(pr[i])
    out.append(pr[-1])
    return np.array(out)


def lathe(profile, tex, mat, n=40, key=None, sharp=35.0):
    """Surface of revolution about +z. profile (k, 2) = (r, t) points, counter-clockwise in the (r, t) plane
    (outer side upwards, top inwards ...) -> outward normals. Texture: u around (at the largest radius), v along."""
    pr = sharpen(profile, sharp) if sharp else np.asarray(profile, np.float64)
    th = np.linspace(0, 2 * math.pi, n + 1)
    radial = np.stack([np.cos(th), np.sin(th), np.zeros_like(th)], -1)
    z = np.array([0.0, 0.0, 1.0])
    P = pr[:, 0, None, None] * radial[None] + pr[:, 1, None, None] * z
    seg = np.diff(pr, axis=0)
    ln = np.linalg.norm(seg, axis=1, keepdims=True)
    seg = np.where(ln > 1e-9, seg / np.maximum(ln, 1e-12), 0)
    tan = np.zeros_like(pr)
    tan[:-1] += seg
    tan[1:] += seg
    tan /= np.maximum(np.linalg.norm(tan, axis=1, keepdims=True), 1e-12)
    n2 = np.stack([tan[:, 1], -tan[:, 0]], -1)  # (dt, -dr): outward for a CCW profile
    N = n2[:, 0, None, None] * radial[None] + n2[:, 1, None, None] * z
    rmax = max(pr[:, 0].max(), 1e-3)
    v = cumlen(pr)
    uv = np.stack(np.broadcast_arrays((th * rmax)[None, :], v[:, None]), -1)
    return surf(P, N, uv, tex, mat, key)


def lathe_at(profile, tex, mat, R=None, t=(0, 0, 0), **kw):
    return xform([lathe(profile, tex, mat, **kw)], R, t)[0]


def rbox(size, tex, mat, e1=0.2, e2=0.2, n=10, bulge=0.0, dome=0.0, noise=0.0, rng=None, key=None, clust=1.0,
         flat_bottom=False):
    """Rounded box around the origin (generalised superellipsoid, cube-sphere grids). One surf per face, so the
    top can carry prints; e1 = roundness in z (0 = sharp), e2 = in x-y. bulge: sides out at mid height,
    dome: top/bottom up/down in the middle; noise: random dents (relative). clust > 1 packs the grid lines
    towards the face borders (small roundings stay round instead of becoming a chamfer); flat_bottom: the
    bottom stays flat on z = -c (stands on the floor / the layer below, no dome or dents downwards)."""
    a, b, c = np.asarray(size, np.float64) / 2
    t = np.linspace(-1, 1, n + 1)
    s = np.sign(t) * (1 - (1 - np.abs(t)) ** clust)
    uu, vv = np.meshgrid(s, s)
    parts = []
    faces = [(2, 1), (2, -1), (0, 1), (0, -1), (1, 1), (1, -1)]  # top, bottom, +x, -x, +y, -y
    for fi, (ax, sg) in enumerate(faces):
        p = np.zeros(uu.shape + (3,))
        o = [i for i in range(3) if i != ax]
        p[..., ax] = sg
        p[..., o[0]], p[..., o[1]] = uu, vv
        F = (np.abs(p[..., 0]) ** (2 / e2) + np.abs(p[..., 1]) ** (2 / e2)) ** (e2 / e1) + np.abs(p[..., 2]) ** (2 / e1)
        q = p * F[..., None] ** (-e1 / 2)
        if bulge:
            q[..., :2] *= 1 + bulge * (1 - q[..., 2:3] ** 2)
        if dome:
            up = q[..., 2] > 0.5 if flat_bottom else np.abs(q[..., 2]) > 0.5
            q[..., 2] += dome * np.sign(q[..., 2]) * (1 - q[..., 0] ** 2) * (1 - q[..., 1] ** 2) * up
        if noise and rng is not None:
            q *= 1 + noise * cv2.resize(rng.uniform(-1, 1, (4, 4)).astype(np.float32), (n + 1, n + 1),
                                         interpolation=cv2.INTER_CUBIC)[..., None]
        if flat_bottom:
            q[..., 2] = np.maximum(q[..., 2], -1.0)
        P = q * (a, b, c)
        N = grid_normals(P, center=np.zeros(3))
        parts.append(surf(P, N, grid_uv(P), tex, mat, None if key is None else f"{key}_{fi}"))
        parts[-1]["face"] = fi
    return parts


def tube(path, r, tex, mat, closed=False, n=8, key=None):
    """Smooth round tube of radius r along a 3D polyline."""
    path = np.asarray(path, np.float64)
    if closed:
        path = np.r_[path, path[:1]]
    tang = np.gradient(path, axis=0)
    if closed:
        tang[0] = tang[-1] = path[1] - path[-2]
    tang /= np.linalg.norm(tang, axis=1, keepdims=True)
    ref = np.array([0.0, 0.0, 1.0]) if abs(tang[0][2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    a = np.linspace(0, 2 * math.pi, n + 1)
    rings, normals, u = [], [], None
    for p, t in zip(path, tang):  # parallel transport of the ring frame
        u = np.cross(np.cross(t, ref), t) if u is None else np.cross(t, np.cross(u, t))
        u /= np.linalg.norm(u)
        v = np.cross(t, u)
        nr = np.cos(a)[:, None] * u + np.sin(a)[:, None] * v
        rings.append(p + r * nr)
        normals.append(nr)
    P, N = np.array(rings), np.array(normals)
    uv = np.stack(np.broadcast_arrays((a * r)[None, :], cumlen(path)[:, None]), -1)
    return surf(P, N, uv, tex, mat, key)


def band(ring_xy, z0, z1, tex, mat, key=None):
    """Vertical band (strap) following a closed ring (k, 2); normals point away from the ring centre."""
    ring = np.r_[ring_xy, ring_xy[:1]]
    P = np.stack([np.c_[ring, np.full(len(ring), z)] for z in (z0, z1)])
    N = grid_normals(P, closed_u=True)
    N[..., 2] = 0
    N /= np.maximum(np.linalg.norm(N, axis=-1, keepdims=True), 1e-12)
    N *= np.where(((P[..., :2] - ring.mean(0)) * N[..., :2]).sum(-1, keepdims=True) < 0, -1, 1)
    return surf(P, N, grid_uv(P), tex, mat, key)


# ---------------------------------------------------------------------------------- flat polygon solids
def extrude(outline, z0, z1):
    """Vertical prism over a convex (k, 2) outline -> polys [(poly, inside)] (top + sides)."""
    o = np.asarray(outline, np.float64)
    c = np.r_[o.mean(0), (z0 + z1) / 2]
    polys = [(np.c_[o, np.full(len(o), z1)], c), (np.c_[o, np.full(len(o), z0)][::-1], c)]
    for a, b in zip(o, np.roll(o, -1, 0)):
        polys.append((np.array([[*a, z1], [*b, z1], [*b, z0], [*a, z0]]), c))
    return polys


def circle(cx, cy, r, n=14):
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return np.c_[cx + r * np.cos(t), cy + r * np.sin(t)]


def chaikin(P, it=3):
    P = np.asarray(P, np.float64)
    for _ in range(it):
        Q = np.roll(P, -1, 0)
        new = np.empty((2 * len(P), P.shape[1]))
        new[0::2], new[1::2] = 0.75 * P + 0.25 * Q, 0.25 * P + 0.75 * Q
        P = new
    return P


def poly_part(polys, tex, mat):
    return {"polys": polys, "tex": tex, "mat": mat, "cull": False, "top_tex": None,
            "P": np.concatenate([q for q, _ in polys])}


def box(cx, cy, l, w, z0, z1, tex, mat, yaw=0.0):
    return G.part(G.prism(cx, cy, l, w, z0, z1, yaw), tex, mat=mat)


# ---------------------------------------------------------------------------------- stretch film
def convex_hull(xy):
    xy = np.asarray(xy, np.float32).reshape(-1, 2)
    return cv2.convexHull(xy).reshape(-1, 2).astype(np.float64)


def offset_hull(xy, r, k=12):
    """Convex hull of the points grown by r (Minkowski sum with a disc)."""
    h = convex_hull(xy)
    if r <= 0:
        return h
    a = np.linspace(0, 2 * math.pi, k, endpoint=False)
    return convex_hull((h[:, None] + r / math.cos(math.pi / k) * np.stack([np.cos(a), np.sin(a)], -1)[None]).reshape(-1, 2))


def radial(poly, c, th):
    """Distance from c to the boundary of the convex polygon along the directions th."""
    d = np.stack([np.cos(th), np.sin(th)], -1)[:, None]
    p = poly[None]
    e = np.roll(poly, -1, 0)[None] - p
    w = p - c
    cr = lambda a, b: a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]  # noqa: E731
    den = cr(d, -e)
    with np.errstate(divide="ignore", invalid="ignore"):
        t, s = cr(w, -e) / den, cr(d, w) / den
    ok = (np.abs(den) > 1e-12) & (t > 0) & (s >= -1e-9) & (s <= 1 + 1e-9)
    return np.where(ok, t, np.inf).min(1)


def inside_convex(pts, hull):
    """(M, 2) points -> bool, inside the convex polygon hull (k, 2) (either orientation)."""
    e = np.roll(hull, -1, 0) - hull
    w = pts[:, None, :] - hull[None]
    cr = e[None, :, 0] * w[..., 1] - e[None, :, 1] * w[..., 0]
    return (cr >= -1e-9).all(1) | (cr <= 1e-9).all(1)


def film_style(rng, clear=None):
    """Film look: clear stretch film (most), bubble wrap, black / white opaque film or hood, tinted (blue) film.
    tone = diffuse albedo, base = milky haze everywhere, gain = haze of the crinkles, gloss = specular amount."""
    r = rng.random() if clear is None else (0.0 if clear else rng.uniform(0.7, 1.0))
    if r < 0.7:
        return {"style": "clear", "tone": 0.85, "base": rng.uniform(0.0, 0.04), "gain": rng.uniform(0.6, 1.2)}
    if r < 0.8:
        return {"style": "bubble", "tone": 0.85, "base": rng.uniform(0.15, 0.35), "gain": 0.8}
    if r < 0.88:
        return {"style": "black", "tone": rng.uniform(0.02, 0.06), "base": 1.0, "gain": 0.0}
    if r < 0.95:
        return {"style": "white", "tone": rng.uniform(0.55, 0.85), "base": rng.uniform(0.85, 1.0), "gain": 0.3}
    return {"style": "tinted", "tone": rng.uniform(0.12, 0.35), "base": rng.uniform(0.5, 0.8), "gain": 0.5}


def comps_of(parts):
    """Film support: one (xy points, top height) per part."""
    return [(p["P"][:, :2], float(p["P"][:, 2].max())) for p in parts]


def film_wrap(rng, comps, z0, z1, top, rounding=None, style=None, n=160):
    """Stretch film around a load. comps = [(xy points, top height)]. The side follows the convex outline of
    everything still reaching each height, grown by `rounding`, and drapes inwards over lower parts at a
    limited slope; with top=True a cover over the highest parts. -> list of film surfaces."""
    rounding = rng.uniform(0.006, 0.02) if rounding is None else rounding
    style = film_style(rng) if style is None else style
    hmax = max(t for _, t in comps)
    z1 = min(z1, hmax)
    tops = sorted({round(t, 4) for _, t in comps if z0 < t < z1})
    zs = np.unique(np.r_[np.linspace(z0, z1, 10), tops, [min(z1, t + 0.01) for t in tops]])
    k = rng.uniform(0.6, 1.5)  # max inward draping per metre of height
    hulls = []
    for i, z in enumerate(zs):  # outline per level: nested, each inside the one below
        sel = [xy for xy, t in comps if t >= z - 1e-6] or [xy for xy, t in comps if t >= hmax - 1e-6]
        h = offset_hull(np.concatenate(sel), rounding)
        if hulls:
            prev = hulls[-1]
            cp = prev.mean(0)
            r = np.linalg.norm(prev - cp, axis=1, keepdims=True)
            shrunk = cp + (prev - cp) * np.clip(1 - k * (z - zs[i - 1]) / np.maximum(r, 1e-9), 0, 1)
            h = convex_hull(np.r_[h, shrunk])
        hulls.append(h)
    c = hulls[-1].mean(0)  # inside every level (levels are nested)
    th = np.linspace(0, 2 * math.pi, n + 1)
    R = np.array([radial(h, c, th) for h in hulls])
    R = np.where(np.isfinite(R), R, np.nanmax(np.where(np.isfinite(R), R, np.nan)))
    d = np.stack([np.cos(th), np.sin(th)], -1)
    P = np.stack([np.c_[c + R[i][:, None] * d, np.full(len(th), z)] for i, z in enumerate(zs)])
    N = grid_normals(P, closed_u=True)
    N *= np.where(((P[..., :2] - c) * N[..., :2]).sum(-1, keepdims=True) < 0, -1, 1)  # outwards
    films = [dict(P=P, N=N, uv=grid_uv(P), kind="side", **style)]
    if top:
        ss = np.linspace(1, 0, 10)
        ring = P[-1, :, :2]
        xy = c + ss[:, None, None] * (ring - c)
        h = np.full(xy.shape[:2], z0)
        flat = xy.reshape(-1, 2)
        for cxy, t in comps:
            if t <= z0:
                continue
            inside = inside_convex(flat, offset_hull(cxy, rounding * 0.5)).reshape(h.shape)
            h = np.where(inside, np.maximum(h, t), h)
        for _ in range(4):  # film under tension spans gaps between the high points
            h = np.maximum(h, cv2.GaussianBlur(h.astype(np.float32), (0, 0), 1.5).astype(np.float64))
        h[0] = z1
        Pc = np.concatenate([xy, h[..., None] + 0.004 * (np.arange(len(ss)) > 0)[:, None, None]], -1)
        Nc = grid_normals(Pc, closed_u=True)
        Nc *= np.where(Nc[..., 2:] < 0, -1, 1)  # upwards
        films.append(dict(P=Pc, N=Nc, uv=grid_uv(Pc), kind="top", **style))
    return films


def film_top_z(films, x, y):
    """Height of the film cover above (x, y), or None without a cover."""
    for f in films:
        if f["kind"] == "top":
            P = f["P"].reshape(-1, 3)
            return float(P[np.argmin(np.hypot(P[:, 0] - x, P[:, 1] - y)), 2])
    return None


def box_film(rng, lo, hi, z0, z1, top, style=None):
    """Film around a box-shaped load (v2 loads)."""
    xy = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
    return film_wrap(rng, [(xy, z1)], z0, z1, top, style=style)


# ---------------------------------------------------------------------------------- unit helpers
def unit(kind, parts, pallet=None, films=(), labels=(), quads=(), typ=None, carrier_h=0.0):
    lo, hi = extent(parts + [{"P": f["P"].reshape(-1, 3)} for f in films])
    load = [p for p in parts if not p.get("carrier")]
    llo, lhi = extent(load) if load else (lo, hi)
    return {"kind": kind, "type": typ or ("palletised_load" if pallet or carrier_h else "parcel"), "parts": parts,
            "quads": list(quads), "films": list(films), "pallet": list(pallet) if pallet else None,
            "load": [float(lhi[0] - llo[0]), float(lhi[1] - llo[1])], "labels": list(labels),
            "carrier_h": float(PALLET_H if pallet else carrier_h)}


def carrier(parts):
    for p in parts:
        p["carrier"] = True
    return parts


def pallet_for(rng, sizes):
    """(parts, L, W, base height) for one of the given pallet footprints."""
    L, W = sizes[rng.integers(len(sizes))]
    return carrier(G.build_pallet(rng, L, W)), L, W, PALLET_H


def grid_xy(L, W, d_x, d_y, jitter=0.0, rng=None):
    nx, ny = max(1, int((L + 1e-6) // d_x)), max(1, int((W + 1e-6) // d_y))
    xs = (np.arange(nx) - (nx - 1) / 2) * d_x
    ys = (np.arange(ny) - (ny - 1) / 2) * d_y
    pts = [(x, y) for x in xs for y in ys]
    if jitter and rng is not None:
        pts = [(x + rng.normal(0, jitter), y + rng.normal(0, jitter)) for x, y in pts]
    return pts


def maybe_film(rng, parts, z0, p=0.5, top_p=0.5, style=None):
    if rng.random() >= p:
        return []
    H = max(pp["P"][:, 2].max() for pp in parts)
    top = rng.random() < top_p
    z1 = H if top else H - rng.uniform(0.0, 0.15)
    return film_wrap(rng, comps_of(parts), max(0.02, z0 - rng.uniform(0, 0.06)), max(z1, z0 + 0.05), top, style=style)


def paint_tone(rng):
    return float(rng.choice([rng.uniform(40, 70), rng.uniform(50, 85), rng.uniform(14, 30), rng.uniform(45, 80),
                             rng.uniform(110, 160), rng.uniform(170, 215)]))


def castor(rng, x, y, d=0.1, w=0.032, tone=60.0, metal=150.0):
    """Swivel castor: wheel (axis along y) + fork + top plate; height d + 0.035."""
    Ri, Ro = d * 0.18, d / 2
    prof = [(Ri, -w / 2), (Ro - 0.006, -w / 2), (Ro, -w / 2 + 0.006), (Ro, w / 2 - 0.006), (Ro - 0.006, w / 2),
            (Ri, w / 2), (Ri, -w / 2)]
    wheel = lathe_at(prof, plain(rng, tone, 2, 2), "rubber", R=rotx(math.pi / 2), t=(x, y, d / 2), n=20)
    fork = [box(x, y + s * (w / 2 + 0.005), d * 0.55, 0.004, d * 0.3, d + 0.02, plain(rng, metal, 3, 3), "steel")
            for s in (-1, 1)]
    plate = box(x - d * 0.12, y, d * 0.75, w + 0.03, d + 0.02, d + 0.035, plain(rng, metal, 3, 3), "steel")
    return [wheel, plate] + fork


def label_part(rng, x, y, z, yaw=None, max_size=0.3):
    """White shipping label with the DataMatrix lying on a flat spot -> (part, symbol corners (4, 3))."""
    s = px(G.LABEL_M)
    pad = px(rng.uniform(0.008, 0.03))
    lw = s + 2 * pad + px(rng.uniform(0, 0.08))
    lh = s + 2 * pad + px(rng.uniform(0, 0.05))
    lim = px(max_size)
    lw, lh = min(lw, max(lim, s + 2 * pad)), min(lh, max(lim, s + 2 * pad))
    patch = np.full((lh, lw), rng.uniform(185, 240), np.float32) + rng.normal(0, 2, (lh, lw)).astype(np.float32)
    patch[pad:pad + s, pad:pad + s] = np.minimum(G.dmc_symbol(rng, s), patch[0, 0])
    if lw - s - 2 * pad > 10:
        for k in range(3):
            G.draw_text(patch, G.pseudo_word(rng).upper(), cv2.FONT_HERSHEY_SIMPLEX, 40,
                        (s + 1.5 * pad, pad + k * s / 3, lw - s - 2.5 * pad, s / 4))
    yaw = rng.uniform(0, 2 * math.pi) if yaw is None else yaw
    Lm, Wm = lw / G.TEX_PPM, lh / G.TEX_PPM
    p = G.part(G.prism(x, y, Lm, Wm, z, z + 0.0006, yaw), lambda f, w, h: np.full((h, w), 215.0), mat="matte")
    p["top_tex"] = patch
    top = p["P"][4:]
    uv = np.array([[pad, pad], [pad + s, pad], [pad + s, pad + s], [pad, pad + s]], np.float64) / (lw, lh)
    return p, top[0] + uv[:, :1] * (top[1] - top[0]) + uv[:, 1:] * (top[3] - top[0])


# ---------------------------------------------------------------------------------- builders
def drums(rng):
    """Steel / plastic drums (200 l, sometimes 60-120 l) on a pallet, single drum on the floor."""
    plastic = rng.random() < 0.35
    small = rng.random() < 0.2
    D = rng.uniform(0.4, 0.48) if small else (rng.uniform(0.56, 0.6) if plastic else 0.585)
    Hd = rng.uniform(0.55, 0.75) if small else (rng.uniform(0.9, 0.95) if plastic else 0.88)
    q = rng.random()
    if q < 0.45:
        pal, L, W, base = pallet_for(rng, [(1.2, 1.2), (1.14, 1.14), (1.2, 1.0)])
    elif q < 0.85:
        pal, L, W, base = pallet_for(rng, [(1.2, 0.8)])
    else:
        pal, L, W, base = [], D, D, 0.0
    R = D / 2
    if plastic:
        prof = [(0, 0), (R - 0.04, 0), (R, 0.04), (R, Hd * 0.3), (R + 0.008, Hd * 0.32), (R, Hd * 0.34),
                (R, Hd * 0.66), (R + 0.008, Hd * 0.68), (R, Hd * 0.7), (R, Hd - 0.05), (R - 0.03, Hd - 0.005),
                (R - 0.045, Hd), (R - 0.06, Hd - 0.012), (0, Hd - 0.012)]
        mat, tone = "plastic", float(rng.choice([rng.uniform(50, 80), rng.uniform(15, 30), rng.uniform(150, 200)]))
    else:
        h1, h2 = Hd / 3, 2 * Hd / 3
        prof = [(0, 0), (R - 0.012, 0), (R, 0.012), (R, h1 - 0.02), (R + 0.006, h1 - 0.01), (R + 0.011, h1),
                (R + 0.006, h1 + 0.01), (R, h1 + 0.02), (R, h2 - 0.02), (R + 0.006, h2 - 0.01), (R + 0.011, h2),
                (R + 0.006, h2 + 0.01), (R, h2 + 0.02), (R, Hd - 0.015), (R + 0.004, Hd), (R - 0.012, Hd),
                (R - 0.02, Hd - 0.012), (0, Hd - 0.012)]
        mat, tone = "paint", paint_tone(rng)
    vc = cumlen(sharpen(prof))
    diamond = rng.random() < 0.5

    def tex(f, w, h, tone=tone):
        t = G.plain(rng, w, h, tone, 2.5, 4)
        if rng.random() < 0.3:  # old drum: dirt / rust streaks
            t -= np.clip(G.smooth_noise(rng, h, w, px(0.1)), 0, 1) * rng.uniform(10, 40)
        if diamond:
            v = px(vc[3] + (vc[-6] - vc[3]) * rng.uniform(0.3, 0.6))
            hazard_diamond(t, rng.uniform(0.1, 0.9) * w, v, px(0.1), rng)
        if rng.random() < 0.5:
            G.draw_text(t, G.pseudo_word(rng).upper(), cv2.FONT_HERSHEY_SIMPLEX, float(np.clip(255 - tone, 20, 235)),
                        (rng.uniform(0, 0.6) * w, px(vc[3] + 0.05), w * 0.25, px(0.06)))
        return t

    pos = grid_xy(L, W, D + 0.01, D + 0.01) if pal else [(0.0, 0.0)]
    if pal and len(pos) > 1 and rng.random() < 0.15:
        pos = pos[:-1]
    parts, labels = list(pal), []
    key = f"drum{rng.integers(1 << 30)}"
    for x, y in pos:
        x, y = x + rng.normal(0, 0.005), y + rng.normal(0, 0.005)
        parts.append(lathe_at(prof, tex, mat, t=(x, y, base), n=48, key=key))
        a = rng.uniform(0, 2 * math.pi)
        for rr, rb in ((R * 0.65, 0.035), (R * 0.65, 0.025)):  # bungs on the lid
            bp = [(0, 0), (rb, 0), (rb, 0.014), (0, 0.014)]
            parts.append(lathe_at(bp, plain(rng, tone * 0.9, 2, 2), mat,
                                  t=(x + rr * math.cos(a), y + rr * math.sin(a), base + Hd - 0.013), n=16))
            a += math.pi
        labels.append((x - R * 0.35 * math.sin(a), y + R * 0.35 * math.cos(a), base + Hd - 0.012, 0.2))
    films = maybe_film(rng, parts[len(pal):], base, p=0.3) if pal else []
    if pal and not films and rng.random() < 0.3:  # straps around the drums
        ring = offset_hull(np.concatenate([circle(x, y, R, 24) for x, y in pos]), 0.004)
        ring = ring[np.argsort(np.arctan2(*(ring - ring.mean(0)).T[::-1]))]
        st = plain(rng, rng.uniform(25, 220), 2, 2)
        for z in (base + Hd * 0.25, base + Hd * 0.75):
            parts.append(band(ring, z, z + 0.016, st, "strap"))
    return unit("drums", parts, (L, W) if pal else None, films, labels)


def ibc(rng):
    """1000 l IBC: translucent HDPE tank in a galvanised tube cage on a pallet base."""
    pal, L, W, base = pallet_for(rng, [(1.2, 1.0)])
    Ht = rng.uniform(0.95, 1.0)
    tank_tone = rng.uniform(165, 215)
    tank = xform(rbox((1.13, 0.95, Ht), plain(rng, tank_tone, 3, 6), "plastic", e1=0.25, e2=0.12, n=12),
                 t=(0, 0, base + 0.01 + Ht / 2))
    steel = rng.uniform(140, 200)
    st = plain(rng, steel, 3, 4)
    parts = list(pal) + tank
    Hc = base + Ht + 0.03
    xs, ys = np.linspace(-0.585, 0.585, 7), np.linspace(-0.49, 0.49, 6)
    r = 0.0095
    for x in xs:  # vertical bars
        for y in (-0.49, 0.49):
            parts.append(tube([(x, y, base), (x, y, Hc)], r, st, "steel"))
    for y in ys[1:-1]:
        for x in (-0.585, 0.585):
            parts.append(tube([(x, y, base), (x, y, Hc)], r, st, "steel"))
    rect = [(-0.585, -0.49), (0.585, -0.49), (0.585, 0.49), (-0.585, 0.49)]
    for z in np.linspace(base + 0.25, Hc, 4):  # horizontal rings
        pts = [(x, y, z) for x, y in rect]
        parts.append(tube(chaikin(np.array(pts), 2), r, st, "steel", closed=True))
    for x in np.linspace(-0.4, 0.4, rng.integers(2, 4)):  # top bars across
        parts.append(tube([(x, -0.49, Hc), (x, 0.49, Hc)], r, st, "steel"))
    cx, cy = rng.choice([-0.25, 0.0, 0.25]), rng.choice([-0.2, 0.0])
    cap_r = rng.uniform(0.075, 0.115)
    parts.append(lathe_at([(0, 0), (cap_r, 0), (cap_r, 0.045), (0, 0.045)],
                          plain(rng, rng.choice([rng.uniform(15, 40), rng.uniform(150, 210)]), 2, 2), "plastic",
                          t=(cx, cy, base + 0.01 + Ht - 0.01), n=24))
    parts.append(box(0.55, 0.0, 0.12, 0.14, base + 0.05, base + 0.2, plain(rng, rng.uniform(20, 120), 2, 2),
                     "plastic"))  # outlet valve
    lx = -cx * 1.6 if cx else rng.choice([-0.35, 0.35])
    return unit("ibc", parts, (L, W), maybe_film(rng, parts[len(pal):], base, p=0.08),
                [(lx, 0.15, base + 0.01 + Ht, 0.18)])


def plank_disc(rng, R, tone, hole_r):
    """Top texture of a wooden reel flange: parallel boards, centre hole, bolts."""
    def tex(f, w, h):
        t = np.zeros((h, w), np.float32)
        nb = max(3, int(2 * R / rng.uniform(0.1, 0.2)))
        for i in range(nb):
            y0, y1 = int(i * h / nb), int((i + 1) * h / nb)
            t[y0:y1] = G.wood(rng, w, max(1, y1 - y0), float(np.clip(tone + rng.normal(0, 12), 40, 220)))
            t[y0:y0 + 1] *= 0.5
        cv2.circle(t, (w // 2, h // 2), max(1, px(hole_r)), float(tone * 0.25), -1, cv2.LINE_AA)
        for a in np.linspace(0, 2 * math.pi, 4, endpoint=False) + rng.uniform(0, 1):
            cv2.circle(t, (int(w / 2 + px(R * 0.45) * math.cos(a)), int(h / 2 + px(R * 0.45) * math.sin(a))),
                       max(1, px(0.012)), float(tone * 0.4), -1, cv2.LINE_AA)
        return t
    return tex


def winding(rng, tone, pitch, along_u=True, tilt=0.0):
    """Cable windings: thin dark grooves every `pitch` metres (rows = constant v, or columns)."""
    def tex(f, w, h):
        t = G.plain(rng, w, h, tone, 2, 4)
        p = max(2.0, px(pitch))
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        coord = (yy + tilt * xx) if along_u else (xx + tilt * yy)
        g = np.abs(((coord / p) % 1.0) - 0.5) * 2
        return t * (0.55 + 0.45 * np.clip(g * 1.6, 0, 1))
    return tex


def cable_drum(rng):
    """Wooden cable reel, standing on a flange (on a pallet or floor) or lying on its rim."""
    Df = rng.uniform(0.5, 1.6)
    Wd = rng.uniform(0.25, 0.4) + Df * rng.uniform(0.2, 0.45)
    tf = rng.uniform(0.025, 0.06)
    Dw = Df * rng.uniform(0.8, 0.94)
    Rf, Rw = Df / 2, Dw / 2
    tone = rng.uniform(90, 190)
    cable = float(rng.choice([rng.uniform(12, 35), rng.uniform(50, 90), rng.uniform(110, 170)]))
    flange_tex = plank_disc(rng, Rf, tone, rng.uniform(0.04, 0.08))
    parts = []
    for z0 in (0.0, tf + Wd):
        parts.append(poly_part(extrude(circle(0, 0, Rf, 48), z0, z0 + tf), flange_tex, "wood"))
    parts.append(lathe_at([(Rw, 0.0), (Rw, Wd)], winding(rng, cable, rng.uniform(0.012, 0.04)), "rubber",
                          t=(0, 0, tf), n=48))
    standing = Df < 1.0 and rng.random() < 0.45
    labels, pal, base = [], None, 0.0
    if standing:
        if rng.random() < 0.6:
            pp, L, W, base = pallet_for(rng, [(1.2, 0.8), (1.2, 1.0), (0.8, 0.6)] if Df < 0.8 else [(1.2, 1.0), (1.2, 1.2)])
            pal = (L, W)
            parts = xform(parts, t=(0, 0, base)) + pp
        a = rng.uniform(0, 2 * math.pi)
        labels.append((Rf * 0.6 * math.cos(a), Rf * 0.6 * math.sin(a), base + 2 * tf + Wd, 0.2))
    else:
        parts = xform(parts, rotx(math.pi / 2), (0, (2 * tf + Wd) / 2, Rf))
        for s in (-1, 1):  # chocks
            parts.append(box(s * Rf * 0.8, 0, 0.12, 2 * tf + Wd + 0.1, 0.0, 0.08, plain(rng, rng.uniform(90, 160), 3, 6),
                             "wood"))
    return unit("cable_drum", parts, pal, maybe_film(rng, [p for p in parts if not p.get("carrier")], base, p=0.1),
                labels)


def ring_profile(Ri, Ro, h, r):
    """Counter-clockwise rounded-rectangle cross-section (r, t) for coils, rolls ends etc."""
    r = min(r, (Ro - Ri) / 2 - 1e-4, h / 2 - 1e-4)
    pts = []
    for cx, cz, a0 in ((Ro - r, r, -90), (Ro - r, h - r, 0), (Ri + r, h - r, 90), (Ri + r, r, 180)):
        for a in np.radians(np.linspace(a0, a0 + 90, 5)):
            pts.append((cx + r * math.cos(a), cz + r * math.sin(a)))
    return pts + [pts[0]]


def coils(rng):
    """Cable / hose coils lying flat, stacked on a pallet (often film-wrapped) or on the floor."""
    Ro = rng.uniform(0.22, 0.55)
    Ri = Ro * rng.uniform(0.45, 0.75)
    hc = rng.uniform(0.08, 0.3)
    n = int(rng.integers(1, 5))
    tone = float(rng.choice([rng.uniform(12, 35), rng.uniform(45, 90), rng.uniform(100, 180)]))
    tex = winding(rng, tone, rng.uniform(0.008, 0.03), along_u=False, tilt=rng.uniform(-0.15, 0.15))
    if rng.random() < 0.7:
        pal, L, W, base = pallet_for(rng, [(1.2, 0.8), (1.2, 1.0), (1.2, 1.2), (0.8, 0.6)] if Ro < 0.4
                                     else [(1.2, 1.2), (1.2, 1.0)])
    else:
        pal, L, W, base = [], 2 * Ro, 2 * Ro, 0.0
    cols = grid_xy(L, W, 2 * Ro + 0.01, 2 * Ro + 0.01) if pal else [(0.0, 0.0)]
    parts, labels, key = list(pal), [], f"coil{rng.integers(1 << 30)}"
    prof = ring_profile(Ri, Ro, hc, rng.uniform(0.2, 0.5) * min(Ro - Ri, hc))
    for x, y in cols:
        for k in range(n):
            z = base + k * hc
            parts.append(lathe_at(prof, tex, "rubber", t=(x + rng.normal(0, 0.01), y + rng.normal(0, 0.01), z), n=48,
                                  key=key))
        a = rng.uniform(0, 2 * math.pi)
        labels.append((x + (Ri + Ro) / 2 * math.cos(a), y + (Ri + Ro) / 2 * math.sin(a), base + n * hc,
                       0.8 * (Ro - Ri)))
    return unit("coils", parts, (L, W) if pal else None, maybe_film(rng, parts[len(pal):], base, p=0.5), labels)


def tyre_profile(Ri, Ro, w):
    """Counter-clockwise tyre cross-section (r, t), t centred on the tread; bulging sidewalls, rounded shoulders."""
    bead, cr = w * 0.72, min(0.035, w * 0.25)
    pts = []
    for s in np.linspace(0, 1, 7):  # lower sidewall, outwards
        r = Ri + (Ro - cr - Ri) * s
        pts.append((r, -(bead + (w - bead) * math.sin(s * math.pi / 2) + 0.012 * math.sin(s * math.pi))))
    for a in np.radians(np.linspace(-90, 0, 5))[1:]:
        pts.append((Ro - cr + cr * math.cos(a), -w + cr + cr * math.sin(a)))
    pts.append((Ro, w - cr))
    for a in np.radians(np.linspace(0, 90, 5))[1:]:
        pts.append((Ro - cr + cr * math.cos(a), w - cr + cr * math.sin(a)))
    for s in np.linspace(1, 0, 7)[1:]:  # upper sidewall, inwards
        r = Ri + (Ro - cr - Ri) * s
        pts.append((r, bead + (w - bead) * math.sin(s * math.pi / 2) + 0.012 * math.sin(s * math.pi)))
    pts.append(pts[0])
    return pts


def tyre_tex(rng, prof, tone):
    v = cumlen(sharpen(prof))
    pr = sharpen(prof)
    tread = np.nonzero(pr[:, 0] >= pr[:, 0].max() - 1e-6)[0]
    v0, v1 = v[tread.min()], v[tread.max()]
    brand, size_txt = G.pseudo_word(rng).upper(), f"{rng.integers(175, 315)}/{rng.integers(4, 8)}5 R{rng.integers(14, 22)}"

    def tex(f, w, h):
        t = G.plain(rng, w, h, tone, 1.5, 2)
        a, b = px(v0), px(v1)
        pitch = max(4, px(rng.uniform(0.025, 0.045)))
        for x in range(0, w, pitch):  # tread blocks: lateral grooves + 2-4 circumferential grooves
            cv2.line(t, (x, a), (x + pitch // 2, b), float(tone * 0.35), max(1, px(0.006)), cv2.LINE_AA)
        for k in range(rng.integers(2, 5)):
            y = int(a + (b - a) * (k + 1) / 5)
            t[y:y + max(1, px(0.008))] *= 0.35
        for sv in (v0 * 0.55, v1 + (v[-1] - v1) * 0.45):  # sidewall lettering, repeated around
            for k in range(2):
                G.draw_text(t, brand if k == 0 else size_txt, cv2.FONT_HERSHEY_DUPLEX, tone * 1.6 + 8,
                            (w * (0.1 + 0.5 * k), px(sv) - px(0.012), w * 0.25, px(0.024)))
        return t
    return tex


def rim_parts(rng, Ri, w, base_z):
    """Wheel rim inside a lying tyre: barrel + face with spoke windows (metal or black steel)."""
    metal = rng.random() < 0.6
    tone = rng.uniform(140, 210) if metal else rng.uniform(18, 40)
    Rh = rng.uniform(0.03, 0.05)
    prof = [(Rh, w * 0.4), (Ri * 0.97, w * 0.45), (Ri, w * 0.72), (Ri * 0.98, -w * 0.72), (Rh, -w * 0.3)]
    prof = prof[::-1]  # bottom -> top on the inside of the tyre is clockwise; reverse to face up / inwards
    nsp = int(rng.integers(5, 11))

    def tex(f, ww, hh):
        t = G.plain(rng, ww, hh, tone, 2, 3)
        vv = cumlen(sharpen(prof))
        a, b = px(vv[-2] * 0.25 + vv[-1] * 0.0), px(vv[-1] * 0.75)
        for k in range(nsp):
            x0 = int(ww * (k + 0.15) / nsp)
            cv2.rectangle(t, (x0, min(a, b)), (x0 + int(ww * 0.5 / nsp), max(a, b)), float(tone * 0.15), -1)
        return t
    return [lathe_at(prof, tex, "steel" if metal else "paint", t=(0, 0, base_z), n=40)]


def tyres(rng):
    """Car / van tyres (or complete wheels) lying in columns on a pallet, or a single column on the floor."""
    Ro = rng.uniform(0.28, 0.4)
    w = rng.uniform(0.085, 0.16)
    Ri = rng.uniform(0.19, 0.24)
    tone = rng.uniform(12, 28)
    prof = tyre_profile(Ri, Ro, w)
    tex = tyre_tex(rng, prof, tone)
    wheels = rng.random() < 0.3
    nh = int(rng.integers(3, 9 if not wheels else 6))
    if rng.random() < 0.7:
        pal, L, W, base = pallet_for(rng, [(1.2, 0.8), (1.2, 1.0), (1.2, 1.2)])
    else:
        pal, L, W, base = [], 2 * Ro, 2 * Ro, 0.0
    cols = grid_xy(L, W, 2 * Ro + 0.005, 2 * Ro + 0.005) if pal else [(0.0, 0.0)]
    parts, labels, key = list(pal), [], f"tyre{rng.integers(1 << 30)}"
    hmax = 0.0
    for x, y in cols:
        nk = nh if rng.random() < 0.8 else int(rng.integers(1, nh + 1))
        for k in range(nk):
            z = base + w + k * 2 * w
            xx, yy = x + rng.normal(0, 0.008), y + rng.normal(0, 0.008)
            parts.append(lathe_at(prof, tex, "tyre", R=rotz(rng.uniform(0, 2 * math.pi)), t=(xx, yy, z), n=56, key=key))
            if wheels:
                parts += xform(rim_parts(rng, Ri, w, 0.0), t=(xx, yy, z))
        top = base + nk * 2 * w
        hmax = max(hmax, top)
        a = rng.uniform(0, 2 * math.pi)
        rr = (Ri + Ro) / 2
        labels.append((x + rr * math.cos(a), y + rr * math.sin(a), top - 0.004, 0.7 * (Ro - Ri)))
    labels = [lb for lb in labels if lb[2] >= hmax - 0.01]
    return unit("wheels" if wheels else "tyres", parts, (L, W) if pal else None,
                maybe_film(rng, parts[len(pal):], base, p=0.35), labels)


def sack_tex(rng, paper):
    tone = rng.uniform(110, 170) if paper else float(rng.choice([rng.uniform(170, 225), rng.uniform(40, 120)]))
    ink = rng.uniform(20, 70) if tone > 100 else rng.uniform(170, 230)
    word = G.pseudo_word(rng).upper()
    band_c = rng.random() < 0.6

    def tex(f, w, h):
        t = G.plain(rng, w, h, tone, 2.5, 5)
        if f == 0 or rng.random() < 0.5:
            if band_c:
                y0 = int(h * rng.uniform(0.25, 0.45))
                t[y0:y0 + max(2, int(h * 0.15))] = ink
            G.draw_text(t, word, cv2.FONT_HERSHEY_DUPLEX, ink if not band_c else tone, (w * 0.2, h * 0.3, w * 0.6, h * 0.25))
        return t
    return tex, tone


def sacks(rng):
    """25 kg sacks (paper or PE) in interlocked layers on a pallet, often stretch-wrapped / hooded."""
    pal, L, W, base = pallet_for(rng, [(1.2, 0.8), (1.2, 1.0)])
    paper = rng.random() < 0.6
    sl, sw, sh = 0.6 * rng.uniform(0.95, 1.05), 0.4 * rng.uniform(0.95, 1.08), rng.uniform(0.09, 0.15)
    tex, _ = sack_tex(rng, paper)
    mat = "cardboard" if paper else "glossy"
    e1, e2 = rng.uniform(0.5, 0.8), rng.uniform(0.15, 0.3)
    nl = int(rng.integers(3, 11))
    if W > 0.9:  # 5 per layer: 2 lengthwise + 3 crosswise
        pat = [(-0.3, -0.3, 0), (0.3, -0.3, 0), (-0.4, 0.2, 1), (0.0, 0.2, 1), (0.4, 0.2, 1)]
    else:
        pat = [(-0.3, -0.2, 0), (0.3, -0.2, 0), (-0.3, 0.2, 0), (0.3, 0.2, 0)]
    parts, labels = list(pal), []
    key = f"sack{rng.integers(1 << 30)}"
    for k in range(nl):
        flip = -1 if k % 2 else 1
        z = base + sh / 2 + k * sh * 0.92
        for x, y, rot in pat:
            sp = rbox((sl, sw, sh), tex, mat, e1, e2, n=8, noise=0.04, dome=0.15, rng=rng, key=key)
            yaw = (math.pi / 2 if rot else 0.0) + rng.normal(0, 0.03)
            parts += xform(sp, rotz(yaw), (x * flip + rng.normal(0, 0.01), y * flip + rng.normal(0, 0.01), z))
            if k == nl - 1:
                labels.append((x * flip, y * flip, z + sh * 0.5 * (1 + 0.15), 0.15))
    style = film_style(rng, clear=rng.random() < 0.8)
    films = maybe_film(rng, parts[len(pal):], base, p=0.6, top_p=0.6, style=style)
    return unit("sacks", parts, (L, W), films, labels)


def big_bag(rng):
    """FIBC big bag: bulging woven body, lifting loops, filling spout; on a pallet or the floor."""
    S = rng.uniform(0.85, 1.0)
    Hb = rng.uniform(0.7, 1.4)
    tone = float(rng.choice([rng.uniform(170, 220), rng.uniform(120, 165)]))
    word = G.pseudo_word(rng).upper()

    def tex(f, w, h):
        t = G.plain(rng, w, h, tone, 4, 6)
        yy, xx = np.mgrid[0:h, 0:w]
        t *= 0.95 + 0.05 * ((xx // 2 + yy // 2) % 2)  # weave
        t[:, :2] *= 0.75
        t[:, -2:] *= 0.75
        if f >= 2 and rng.random() < 0.6:
            cv2.rectangle(t, (int(w * 0.3), int(h * 0.3)), (int(w * 0.7), int(h * 0.6)), float(tone * 0.5), 2)
            G.draw_text(t, word, cv2.FONT_HERSHEY_SIMPLEX, tone * 0.4, (w * 0.32, h * 0.35, w * 0.36, h * 0.2))
        return t
    if rng.random() < 0.7:
        pal, L, W, base = pallet_for(rng, [(1.2, 1.0), (1.2, 1.2), (1.0, 1.2)])
    else:
        pal, L, W, base = [], S, S, 0.0
    body = xform(rbox((S, S, Hb), tex, "fabric", rng.uniform(0.3, 0.5), rng.uniform(0.25, 0.45), n=12,
                      bulge=rng.uniform(0.04, 0.1), dome=rng.uniform(-0.05, 0.08), noise=0.03, rng=rng),
                 rotz(rng.normal(0, 0.05)), (0, 0, base + Hb / 2))
    parts = list(pal) + body
    top = max(p["P"][:, 2].max() for p in body)
    lt = plain(rng, rng.choice([tone, rng.uniform(40, 120)]), 2, 2)
    hl = rng.uniform(0.12, 0.3)
    for sx in (-1, 1):  # lifting loops on the top corners
        for sy in (-1, 1):
            c = np.array([sx * S * 0.4, sy * S * 0.4])
            d = np.array([sx, -sy]) / math.sqrt(2) * 0.08
            arc = [(*(c + d * math.cos(a)), top - 0.03 + hl * math.sin(a)) for a in np.linspace(0, math.pi, 9)]
            parts.append(tube(arc, 0.012, lt, "fabric"))
    labels = []
    if rng.random() < 0.5:  # filling spout, tied
        r = rng.uniform(0.12, 0.2)
        hs = rng.uniform(0.08, 0.3)
        parts.append(lathe_at([(r, 0), (r * 0.8, hs * 0.7), (r * 0.3, hs), (0, hs)], plain(rng, tone * 0.95, 4, 8),
                              "fabric", t=(0, 0, top - 0.04), n=24))
        labels.append((S * 0.28, -S * 0.1, top - 0.01, 0.15))
    else:
        labels.append((rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2), top, 0.2))
    return unit("big_bag", parts, (L, W) if pal else None, [], labels if rng.random() < 0.6 else [])


def roll_end_tex(rng, prof, tone_side, tone_end, wrap):
    """Paper / textile roll: wrapped side, end face with fine concentric rings, dark core."""
    pr = sharpen(prof)
    v = cumlen(pr)

    def tex(f, w, h):
        t = G.plain(rng, w, h, tone_side if wrap else tone_end, 2, 4)
        for i in range(len(pr) - 1):
            a, b = pr[i], pr[i + 1]
            if abs(a[1] - b[1]) < 1e-6 and abs(a[0] - b[0]) > 1e-6:  # flat end face
                y0, y1 = px(v[i]), px(v[i + 1])
                t[y0:y1] = tone_end + G.grain_noise(rng, max(1, y1 - y0), w, w, 2) * 10
                for y in range(y0, y1, max(2, px(0.006))):
                    t[y] *= rng.uniform(0.85, 1.0)
            elif abs(a[0] - b[0]) < 1e-6 and a[0] < pr[:, 0].max() - 1e-6:  # core
                t[px(v[i]):px(v[i + 1])] = tone_end * 0.3
        return t
    return tex


def rolls_standing(rng):
    """Paper / film reels standing on a pallet or the floor."""
    R = rng.uniform(0.2, 0.6)
    Hr = rng.uniform(0.4, 1.4)
    rc = rng.uniform(0.04, 0.08)
    prof = [(rc, 0), (R, 0), (R, Hr), (rc, Hr), (rc, Hr * 0.3)]
    wrap = rng.random() < 0.6
    tex = roll_end_tex(rng, prof, rng.uniform(110, 200), rng.uniform(150, 225), wrap)
    if rng.random() < 0.6:
        pal, L, W, base = pallet_for(rng, [(1.2, 0.8), (1.2, 1.0), (1.2, 1.2)] if R < 0.5 else [(1.2, 1.2)])
    else:
        pal, L, W, base = [], 2 * R, 2 * R, 0.0
    pos = grid_xy(L, W, 2 * R + 0.01, 2 * R + 0.01) if pal else [(0.0, 0.0)]
    parts, labels, key = list(pal), [], f"roll{rng.integers(1 << 30)}"
    for x, y in pos:
        parts.append(lathe_at(prof, tex, "cardboard" if wrap else "matte", t=(x, y, base), n=56, key=key))
        a = rng.uniform(0, 2 * math.pi)
        labels.append((x + (R + rc) / 2 * math.cos(a), y + (R + rc) / 2 * math.sin(a), base + Hr, 0.7 * (R - rc)))
    return unit("rolls_standing", parts, (L, W) if pal else None, maybe_film(rng, parts[len(pal):], base, p=0.4),
                labels)


def rolls_lying(rng):
    """Carpet / textile / insulation rolls lying in a bundle (pyramid or one layer), on bearers or a pallet."""
    Lr = rng.uniform(1.0, 2.4)
    R = rng.uniform(0.08, 0.28)
    rc = R * rng.uniform(0.15, 0.4)
    prof = [(rc, 0), (R - 0.01, 0), (R, 0.01), (R, Lr - 0.01), (R - 0.01, Lr), (rc, Lr), (rc, Lr - 0.05)]
    glossy = rng.random() < 0.6  # individually wrapped in PE film
    tex = roll_end_tex(rng, prof, rng.uniform(40, 220), rng.uniform(60, 200), True)
    n0 = int(rng.integers(2, 6))
    pyramid = rng.random() < 0.6
    parts, key = [], f"rroll{rng.integers(1 << 30)}"
    on_pallet = Lr <= 1.3 and rng.random() < 0.5
    if on_pallet:
        pal, L, W, base = pallet_for(rng, [(1.2, 0.8), (1.2, 1.0)])
        n0 = max(1, min(n0, int(W // (2 * R))))
        parts += pal
    else:
        base = 0.08
        tone_b = rng.uniform(90, 170)
        for x in np.linspace(-Lr * 0.35, Lr * 0.35, 2 if Lr < 1.8 else 3):
            parts.append(box(x, 0, 0.08, n0 * 2 * R + 0.1, 0.0, 0.08, plain(rng, tone_b, 3, 6), "wood"))
        carrier(parts)
    rows = [(n0 - k) if pyramid else n0 for k in range(n0 if pyramid else int(rng.integers(1, 3)))]
    labels = []
    for k, m in enumerate(rows):
        z = base + R + k * R * (math.sqrt(3) if pyramid else 2)
        for i in range(m):
            y = (i - (m - 1) / 2) * 2 * R
            parts.append(lathe_at(prof, tex, "glossy" if glossy else "fabric", R=roty(math.pi / 2),
                                  t=(-Lr / 2 + rng.normal(0, 0.02), y, z), n=40, key=key))
        if k == len(rows) - 1:
            labels.append((rng.uniform(-0.3, 0.3) * Lr, 0.0, z + R - 0.003, 0.12))
    films = maybe_film(rng, [p for p in parts if not p.get("carrier")], base, p=0.25)
    return unit("rolls_lying", parts, (L, W) if on_pallet else None, films, labels,
                carrier_h=0.0 if on_pallet else 0.08)


def buckets(rng):
    """Buckets (lid + handle) or canisters in layers with tier sheets, stretch-wrapped on a pallet."""
    pal, L, W, base = pallet_for(rng, [(1.2, 0.8), (1.2, 1.0)])
    tone = float(rng.choice([rng.uniform(170, 220), rng.uniform(15, 35), rng.uniform(50, 160)]))
    parts, labels = list(pal), []
    sheet = plain(rng, rng.uniform(110, 170), 2, 4)
    can = rng.random() < 0.4
    nl = int(rng.integers(2, 6))
    z = base
    key = f"bkt{rng.integers(1 << 30)}"
    if can:
        cl, cw, ch = rng.uniform(0.22, 0.32), rng.uniform(0.14, 0.22), rng.uniform(0.25, 0.4)
        pos = grid_xy(L, W, cl + 0.01, cw + 0.01)
    else:
        Rt = rng.uniform(0.11, 0.17)
        ch = rng.uniform(0.2, 0.38)
        pos = grid_xy(L, W, 2 * Rt + 0.01, 2 * Rt + 0.01)
        prof = [(0, 0), (Rt * 0.88, 0), (Rt, ch - 0.02), (Rt + 0.006, ch - 0.015), (Rt + 0.006, ch), (Rt - 0.01, ch),
                (Rt - 0.015, ch - 0.006), (0, ch - 0.006)]
    for k in range(nl):
        for x, y in pos:
            if can:
                body = rbox((cl, cw, ch), plain(rng, tone, 2, 4), "plastic", 0.18, 0.22, n=8, key=key)
                parts += xform(body, t=(x, y, z + ch / 2))
                if k == nl - 1:
                    hx = x - cl * 0.15
                    parts.append(tube([(hx - cl * 0.25, y, z + ch - 0.01), (hx - cl * 0.2, y, z + ch + 0.035),
                                       (hx + cl * 0.2, y, z + ch + 0.035), (hx + cl * 0.25, y, z + ch - 0.01)],
                                      0.012, plain(rng, tone, 2, 2), "plastic"))
                    parts.append(lathe_at([(0, 0), (0.025, 0), (0.025, 0.03), (0, 0.03)], plain(rng, tone * 0.8, 2, 2),
                                          "plastic", t=(x + cl * 0.32, y, z + ch - 0.01), n=16))
                    labels.append((x - cl * 0.15, y + cw * 0.25, z + ch, 0.1))
            else:
                parts.append(lathe_at(prof, plain(rng, tone, 2, 4), "plastic", t=(x, y, z), n=40, key=key))
                if k == nl - 1 and rng.random() < 0.7:  # handle lying on the lid
                    a = rng.uniform(0, math.pi)
                    hp = [(x + Rt * 0.95 * math.cos(a + s), y + Rt * 0.95 * math.sin(a + s), z + ch + 0.006)
                          for s in np.linspace(-0.6, math.pi + 0.6, 10)]
                    parts.append(tube(hp, 0.004, plain(rng, 180, 2, 2), "steel"))
                if k == nl - 1:
                    labels.append((x, y, z + ch - 0.006, 0.15))
        z += ch + 0.003
        if k < nl - 1 and rng.random() < 0.7:  # tier sheet
            parts.append(box(0, 0, L - 0.02, W - 0.02, z - 0.003, z, sheet, "cardboard"))
    return unit("canisters" if can else "buckets", parts, (L, W), maybe_film(rng, parts[len(pal):], base, p=0.8),
                labels)


def beverages(rng):
    """Beverage pallet: shrink-packed bottles in layers with tier sheets; top layer shows caps and shoulders."""
    pal, L, W, base = pallet_for(rng, [(1.2, 0.8), (0.8, 0.6), (1.2, 1.0)])
    rb = rng.uniform(0.03, 0.045)
    hb = rng.uniform(0.2, 0.33)
    nl = int(rng.integers(2, 6))
    parts, labels = list(pal), []
    dark = rng.uniform(25, 90)
    sheet = plain(rng, rng.uniform(110, 170), 2, 4)
    side = rng.uniform(40, 120)
    nx, ny = int(L // (2 * rb)), int(W // (2 * rb))
    lx, ly = nx * 2 * rb, ny * 2 * rb

    def block_tex(f, w, h):  # sides of the lower layers: bottle bodies behind shrink film
        t = G.plain(rng, w, h, side, 2, 4)
        p = max(3, px(2 * rb))
        xx = np.arange(w)[None, :]
        t *= 0.75 + 0.25 * np.abs(np.sin(np.pi * xx / p))
        return t
    z = base
    for k in range(nl - 1):
        parts.append(box(0, 0, lx, ly, z, z + hb, block_tex, "glossy"))
        z += hb
        parts.append(box(0, 0, lx + 0.01, ly + 0.01, z, z + 0.003, sheet, "cardboard"))
        z += 0.003
    prof = [(0, 0), (rb, 0), (rb, hb * 0.62), (rb * 0.35, hb * 0.86), (rb * 0.33, hb * 0.95), (rb * 0.42, hb * 0.95),
            (rb * 0.42, hb), (0, hb)]
    cap = rng.uniform(20, 220)
    vc = cumlen(sharpen(prof))

    def bottle_tex(f, w, h):
        t = G.plain(rng, w, h, dark, 2, 2)
        t[px(vc[-4]):] = cap
        return t
    key = f"btl{rng.integers(1 << 30)}"
    for i in range(nx):
        for j in range(ny):
            x, y = -lx / 2 + (i + 0.5) * 2 * rb, -ly / 2 + (j + 0.5) * 2 * rb
            parts.append(lathe_at(prof, bottle_tex, "glossy", t=(x, y, z), n=14, key=key))
    labels.append((0.0, 0.0, z + hb + 0.012, 0.25))  # on the top film
    films = film_wrap(rng, comps_of(parts[len(pal):]), base, z + hb, True,
                      style=film_style(rng, clear=True)) if rng.random() < 0.85 else []
    if not films:
        labels = [(-lx / 2 + rb, -ly / 2 + rb, z + hb, 0.06)]
    return unit("beverages", parts, (L, W), films, labels)


def crate_parts(rng, cl, cw, ch, tone, open_top, key):
    """Stackable plastic crate (E2 style): ribbed walls with hand holes; open crates are hollow."""
    def wall(f, w, h):
        t = G.plain(rng, w, h, tone, 2, 3)
        for x in range(0, w, max(3, px(0.05))):
            t[:, x:x + max(1, px(0.006))] *= 0.8
        if f in (1, 3) or w > h * 2:
            cv2.ellipse(t, (w // 2, int(h * 0.2)), (max(2, px(0.05)), max(1, px(0.015))), 0, 0, 360, float(tone * 0.2), -1)
        return t
    if not open_top:
        return [box(0, 0, cl, cw, 0, ch, wall, "plastic")]
    th = 0.01
    tt = plain(rng, tone, 2, 3)
    return [box(0, 0, cl, cw, 0, 0.012, tt, "plastic"),
            box(0, -cw / 2 + th / 2, cl, th, 0, ch, wall, "plastic"), box(0, cw / 2 - th / 2, cl, th, 0, ch, wall, "plastic"),
            box(-cl / 2 + th / 2, 0, th, cw - 2 * th, 0, ch, wall, "plastic"),
            box(cl / 2 - th / 2, 0, th, cw - 2 * th, 0, ch, wall, "plastic")]


def crates(rng):
    """Plastic crate stacks: one stack on a dolly with castors, or 2x2 stacks on a pallet."""
    cl, cw = (0.6, 0.4) if rng.random() < 0.75 else (0.4, 0.3)
    ch = rng.uniform(0.12, 0.32)
    tone = float(rng.choice([rng.uniform(40, 80), rng.uniform(100, 160), rng.uniform(15, 35), rng.uniform(60, 120)]))
    dolly = rng.random() < 0.5
    parts, labels = [], []
    if dolly:
        tone_d = rng.uniform(20, 120)
        dh = 0.135
        parts += carrier([box(0, 0, cl + 0.02, cw + 0.02, dh - 0.035, dh, plain(rng, tone_d, 2, 3), "plastic")]
                         + sum([castor(rng, sx * (cl / 2 - 0.05), sy * (cw / 2 - 0.05), d=0.1, tone=rng.uniform(15, 60))
                                for sx in (-1, 1) for sy in (-1, 1)], []))
        for p in parts:
            p["carrier"] = True
        pos, base, pal = [(0.0, 0.0)], dh, None
    else:
        pal_parts, L, W, base = pallet_for(rng, [(1.2, 0.8)])
        parts += pal_parts
        pos, pal = [(sx * cl / 2, sy * cw / 2) for sx in (-1, 1) for sy in (-1, 1)] if cl < 0.5 else \
            [(sx * 0.3, sy * 0.2) for sx in (-1, 1) for sy in (-1, 1)], (L, W)
        if cl > 0.5:
            cl, cw = 0.6, 0.4
    for x, y in pos:
        n = int(rng.integers(2, 9 if dolly else 6))
        key = f"crate{rng.integers(1 << 30)}"
        for k in range(n):
            top = k == n - 1
            cp = crate_parts(rng, cl, cw, ch, tone, top, key if not top else None)
            parts += xform(cp, t=(x, y, base + k * ch))
            if top and rng.random() < 0.6:  # some content in the open crate
                for _ in range(rng.integers(1, 5)):
                    s = rng.uniform(0.06, 0.18)
                    parts += xform(rbox((s, s * rng.uniform(0.6, 1), ch * 0.5), plain(rng, rng.uniform(40, 200), 3, 6),
                                        "matte", 0.4, 0.4, n=6),
                                   rotz(rng.uniform(0, 3)), (x + rng.uniform(-cl / 3, cl / 3), y + rng.uniform(-cw / 4, cw / 4),
                                                             base + k * ch + 0.012 + ch * 0.25))
        labels.append((x + cl * 0.3, y, base + n * ch, 0.0))  # labels sit on the crate side; none on top
    return unit("crates", parts, pal, [], [], carrier_h=0.0 if pal else 0.135)


def roll_container(rng):
    """Retail roll container: base on 4 castors, 2-4 mesh sides on a tube frame, cartons inside."""
    L, W = (0.8, 0.72) if rng.random() < 0.6 else (0.8, 0.67)
    Hc = rng.uniform(1.5, 1.8)
    bh = 0.17
    steel = rng.uniform(110, 190)
    st = plain(rng, steel, 3, 4)
    parts = carrier([box(0, 0, L, W, bh - 0.03, bh, plain(rng, rng.choice([steel * 0.7, rng.uniform(15, 50)]), 3, 4),
                         "steel")] + sum([castor(rng, sx * (L / 2 - 0.07), sy * (W / 2 - 0.07), d=0.125,
                                                 tone=rng.uniform(15, 60), metal=steel)
                                          for sx in (-1, 1) for sy in (-1, 1)], []))
    sides = [s for s in range(4) if rng.random() < (0.9 if s < 2 else 0.5)]
    cs = [(-L / 2, -W / 2), (L / 2, -W / 2), (L / 2, W / 2), (-L / 2, W / 2)]
    quads = []
    frame = []
    for s in sides:
        (x0, y0), (x1, y1) = cs[s], cs[(s + 1) % 4]
        frame.append(tube([(x0, y0, bh), (x0, y0, Hc), (x1, y1, Hc), (x1, y1, bh)], 0.011, st, "steel"))
        q = np.array([[x0, y0, Hc - 0.01], [x1, y1, Hc - 0.01], [x1, y1, bh + 0.01], [x0, y0, bh + 0.01]])
        quads.append(("mesh", q, rng.uniform(0.05, 0.1), steel))
    parts += carrier(frame)
    if rng.random() < 0.4:  # shelf
        parts += carrier([box(0, 0, L - 0.04, W - 0.04, Hc * 0.55, Hc * 0.55 + 0.01, st, "steel")])
    sku = G.make_sku(rng)
    z = bh
    labels = []
    hfill = rng.uniform(0.3, 1.0) * (Hc - bh)
    while z < bh + hfill - 0.1:  # cartons inside
        ch = rng.uniform(0.15, 0.4)
        for x, y in ((-L / 4, -W / 4), (L / 4, -W / 4), (-L / 4, W / 4), (L / 4, W / 4)):
            if rng.random() < 0.8:
                parts.append(G.carton_part(rng, sku, x + rng.normal(0, 0.01), y + rng.normal(0, 0.01), L / 2 - 0.04,
                                           W / 2 - 0.04, z, z + ch, rng.normal(0, 0.04), cull=False))
        z += ch
    labels.append((L / 4, W / 4, z, 0.0))
    return unit("roll_container", parts, None, [], [], quads=quads, carrier_h=bh)


def long_goods(rng):
    """Bundle of pipes or boards (1.5-3 m) on wooden bearers, strapped."""
    Lg = rng.uniform(1.5, 3.0)
    boards = rng.random() < 0.35
    parts = []
    tone_b = rng.uniform(90, 170)
    nb = 2 if Lg < 2.2 else 3
    base = rng.uniform(0.06, 0.1)
    if boards:
        bw, bt = rng.uniform(0.08, 0.2), rng.uniform(0.02, 0.05)
        nx, nz = int(rng.integers(3, 8)), int(rng.integers(3, 10))
        Wb = nx * bw
        tone = rng.uniform(110, 200)
        for k in range(nz):
            for i in range(nx):
                parts.append(box(rng.normal(0, 0.01), -Wb / 2 + (i + 0.5) * bw, Lg, bw - 0.004, base + k * bt,
                                 base + (k + 1) * bt, lambda f, w, h: G.wood(rng, w, h, float(tone + rng.normal(0, 15))),
                                 "wood"))
        Hb, half = nz * bt, Wb / 2
    else:
        r = rng.uniform(0.016, 0.06)
        steel = rng.random() < 0.4
        tone = rng.uniform(120, 200) if steel else float(rng.choice([rng.uniform(80, 160), rng.uniform(15, 35),
                                                                     rng.uniform(170, 220)]))
        ri = r * rng.uniform(0.7, 0.9)
        prof = [(ri, Lg), (ri, 0), (r, 0), (r, Lg), (ri, Lg)][::-1]
        nrow = int(rng.integers(2, 7))
        key = f"pipe{rng.integers(1 << 30)}"
        hmax = 0
        for k in range(nrow):
            m = nrow + 1 - (k % 2) if k < nrow else nrow
            m = max(1, nrow + 2 - k) if rng.random() < 0.3 else nrow + (k % 2 == 0)
            for i in range(m):
                y = (i - (m - 1) / 2) * 2 * r
                z = base + r + k * r * math.sqrt(3)
                parts.append(lathe_at(prof, plain(rng, tone, 2, 3), "steel" if steel else "plastic", R=roty(math.pi / 2),
                                      t=(-Lg / 2 + rng.normal(0, 0.01), y, z), n=16, key=key))
                hmax = max(hmax, z + r)
        Hb, half = hmax - base, (nrow + 1) * r
    tb = plain(rng, tone_b, 3, 6)
    bear = [box(x, 0, 0.08, 2 * half + 0.15, 0.0, base, tb, "wood") for x in np.linspace(-Lg * 0.35, Lg * 0.35, nb)]
    parts = carrier(bear) + parts
    stc = plain(rng, rng.uniform(25, 220), 2, 2)
    for x in np.linspace(-Lg * 0.3, Lg * 0.3, nb):  # straps around the bundle
        ring = [(-half - 0.004, base), (half + 0.004, base), (half + 0.004, base + Hb + 0.004), (-half - 0.004, base + Hb + 0.004)]
        loop = chaikin(np.array([(x, y, z) for y, z in ring]), 3)
        parts.append(tube(loop, 0.004, stc, "strap", closed=True))
    return unit("long_goods", parts, None, [], [(rng.uniform(-0.3, 0.3) * Lg, 0.0, base + Hb, 0.1)] if boards else [],
                carrier_h=base)


def wrapped(rng):
    """Irregular item(s) - machine, furniture, parts - on a pallet or skids under opaque / clear / bubble film."""
    parts = []
    if rng.random() < 0.65:
        pal, L, W, base = pallet_for(rng, [(1.2, 0.8), (1.2, 1.0), (1.2, 1.2), (1.6, 1.0)])
        parts += pal
    else:
        L, W = rng.uniform(0.8, 1.8), rng.uniform(0.6, 1.2)
        base = rng.uniform(0.08, 0.12)
        tb = plain(rng, rng.uniform(90, 170), 3, 6)
        parts += carrier([box(0, y, L, 0.1, 0.0, base, tb, "wood") for y in (-W * 0.35, W * 0.35)])
        pal = None
    comps = []
    sofa = rng.random() < 0.25
    tone = rng.uniform(30, 200)
    if sofa:  # seat + back + arms
        h = rng.uniform(0.35, 0.5)
        comps += xform(rbox((L * 0.95, W * 0.9, h), plain(rng, tone, 3, 5), "fabric", 0.4, 0.3, n=8), t=(0, 0, base + h / 2))
        hb = rng.uniform(0.35, 0.5)
        comps += xform(rbox((L * 0.95, 0.2, hb), plain(rng, tone, 3, 5), "fabric", 0.4, 0.3, n=8),
                       t=(0, W * 0.35, base + h + hb / 2))
        for s in (-1, 1):
            comps += xform(rbox((0.18, W * 0.9, 0.2), plain(rng, tone, 3, 5), "fabric", 0.4, 0.3, n=6),
                           t=(s * L * 0.4, 0, base + h + 0.1))
    else:
        for _ in range(rng.integers(1, 5)):
            l, w, h = rng.uniform(0.2, L), rng.uniform(0.2, W), rng.uniform(0.15, 1.4)
            x, y = rng.uniform(-(L - l) / 2, (L - l) / 2), rng.uniform(-(W - w) / 2, (W - w) / 2)
            if rng.random() < 0.3:
                r = min(l, w) / 2
                comps.append(lathe_at([(0, 0), (r, 0), (r, h), (0, h)], plain(rng, rng.uniform(30, 200), 3, 5), "paint",
                                      t=(x, y, base), n=32))
            else:
                comps += xform(rbox((l, w, h), plain(rng, rng.uniform(30, 200), 3, 5), "paint",
                                    rng.uniform(0.05, 0.3), rng.uniform(0.05, 0.3), n=6), t=(x, y, base + h / 2))
    parts += comps
    style = film_style(rng) if rng.random() < 0.4 else film_style(rng, clear=False)
    H = max(p["P"][:, 2].max() for p in comps)
    films = film_wrap(rng, comps_of(comps), max(0.02, base - 0.03), H, True, rounding=rng.uniform(0.03, 0.1),
                      style=style)
    top = films[-1]["P"]
    i = np.unravel_index(np.argmax(top[..., 2]), top.shape[:2])
    labels = [(float(top[i][0]), float(top[i][1]), float(top[i][2]) - 0.003, 0.15)]
    return unit("wrapped", parts, (L, W) if pal else None, films, labels, carrier_h=0.0 if pal else base)


def octagon(L, W, ch):
    a, b = L / 2, W / 2
    return np.array([(-a + ch, -b), (a - ch, -b), (a, -b + ch), (a, b - ch), (a - ch, b), (-a + ch, b), (-a, b - ch),
                     (-a, -b + ch)])


def octabin(rng):
    """Octabin: octagonal corrugated bulk container on a pallet, with a lid or open (liner bag), strapped."""
    pal, L, W, base = pallet_for(rng, [(1.2, 1.0), (1.2, 0.8), (1.14, 1.14)])
    Lb, Wb = L - rng.uniform(0.0, 0.04), W - rng.uniform(0.0, 0.04)
    ch = rng.uniform(0.15, 0.28) * min(Lb, Wb)
    Hb = rng.uniform(0.55, 1.1)
    sku = G.make_sku(rng)
    sku.update(flaps=False, tape=False, style=str(rng.choice(["plain", "text", "band"])), side_logo=rng.random() < 0.5)
    tex = lambda f, w, h: G.carton_face(rng, sku, w, h, False)  # noqa: E731
    parts, labels = list(pal), []
    parts.append(poly_part(extrude(octagon(Lb, Wb, ch), base, base + Hb)[2:], tex, "cardboard"))  # walls
    top = base + Hb
    if rng.random() < 0.6:  # lid with a skirt over the walls
        lt = top + 0.006
        parts.append(poly_part(extrude(octagon(Lb + 0.016, Wb + 0.016, ch + 0.003), top - rng.uniform(0.04, 0.1), lt),
                               lambda f, w, h: G.carton_face(rng, sku, w, h, f == 0), "cardboard"))
        labels.append((rng.uniform(-0.25, 0.25) * Lb, rng.uniform(-0.2, 0.2) * Wb, lt, 0.3))
    else:  # open: the liner bag with the goods, below the rim
        fill = base + Hb * rng.uniform(0.55, 0.95)
        tone = float(rng.choice([rng.uniform(170, 225), rng.uniform(90, 150)]))
        bag = rbox((Lb - 0.08, Wb - 0.08, 0.12), plain(rng, tone, 3, 8), "glossy", 0.6, 0.55, n=10, dome=0.3,
                   noise=0.05, rng=rng)
        parts += xform(bag, t=(0, 0, fill - 0.06))
    st = plain(rng, rng.uniform(25, 220), 2, 2)
    ring = octagon(Lb + 0.006, Wb + 0.006, ch + 0.001)
    for z in (base + Hb * rng.uniform(0.2, 0.35), base + Hb * rng.uniform(0.65, 0.8)):
        parts.append(band(ring, z, z + 0.016, st, "strap"))
    return unit("octabin", parts, (L, W), [], labels)


def collars(rng):
    """Pallet collars (hinged wooden frames, 0.2 m each) stacked on a pallet; goods inside; open or with a lid."""
    pal, L, W, base = pallet_for(rng, [(1.2, 0.8), (1.2, 1.0)])
    n, hc, t = int(rng.integers(1, 6)), 0.195, 0.02
    tone = rng.uniform(110, 200)
    wtex = lambda f, w, h: G.wood(rng, w, h, float(np.clip(tone + rng.normal(0, 15), 40, 230)))  # noqa: E731
    steel = plain(rng, rng.uniform(60, 150), 3, 4)
    parts = list(pal)
    for k in range(n):
        z = base + k * hc
        for s in (-1, 1):
            parts.append(box(0, s * (W / 2 - t / 2), L, t, z + 0.003, z + hc, wtex, "wood"))
            parts.append(box(s * (L / 2 - t / 2), 0, t, W - 2 * t, z + 0.003, z + hc, wtex, "wood"))
        for sx in (-1, 1):  # hinges: steel plates round the corners
            for sy in (-1, 1):
                for zh in (z + 0.03, z + hc - 0.07):
                    parts.append(box(sx * (L / 2 - 0.04), sy * (W / 2 + 0.0015), 0.08, 0.003, zh, zh + 0.04, steel,
                                     "steel"))
                    parts.append(box(sx * (L / 2 + 0.0015), sy * (W / 2 - 0.04), 0.003, 0.08, zh, zh + 0.04, steel,
                                     "steel"))
    top = base + n * hc
    labels, content = [], []
    fill = rng.uniform(0.3, 1.0) * (top - base)
    if rng.random() < 0.75 and fill > 0.08:
        content = [p for p in G.load_stack(rng, L - 2 * t - 0.03, W - 2 * t - 0.03, fill, base, [G.make_sku(rng)])
                   if p.get("dmc_ok", True)]
        parts += content
    if rng.random() < 0.5:  # lid: plywood / chipboard sheet
        lt = rng.uniform(0.008, 0.02)
        parts.append(box(0, 0, L, W, top, top + lt, lambda f, w, h: G.plywood(rng, w, h), "wood"))
        labels.append((rng.uniform(-0.3, 0.3), rng.uniform(-0.2, 0.2), top + lt, 0.3))
    elif content and fill > top - base - 0.25:
        labels = top_label_spots(rng, content, base + fill)
    return unit("collars", parts, (L, W), [], labels)


def bag_tex(rng, tone):
    """Polybag / mailing bag: soft sheen streaks, welded seams at the ends, print on the top."""
    word = G.pseudo_word(rng).upper()
    printed = rng.random() < 0.5
    ink = rng.uniform(20, 70) if tone > 120 else rng.uniform(170, 230)
    font = int(rng.choice(G.FONTS[:4]))

    def tex(f, w, h):
        t = G.plain(rng, w, h, tone, 2.5, 7)
        for _ in range(rng.poisson(4)):  # wrinkles
            m = np.zeros_like(t)
            p0 = (int(rng.uniform(0, w)), int(rng.uniform(0, h)))
            a, ln = rng.uniform(0, math.pi), px(rng.uniform(0.05, 0.25))
            cv2.line(m, p0, (int(p0[0] + ln * math.cos(a)), int(p0[1] + ln * math.sin(a))), 1.0, max(1, px(0.008)),
                     cv2.LINE_AA)
            t += cv2.GaussianBlur(m, (0, 0), 1.5) * rng.choice([-1, 1]) * rng.uniform(6, 20)
        if f == 0 and printed:
            G.draw_text(t, word, font, ink, (w * 0.2, h * 0.35, w * 0.6, h * 0.3))
        e = max(1, px(0.012))
        t[:, :e] *= 0.85
        t[:, -e:] *= 0.85
        return np.clip(t, 0, 255)
    return tex


def polybags(rng):
    """Soft parcels: polybags / mailing bags / textile packs, one loose on the floor or a heap on a pallet."""
    tone = float(rng.choice([rng.uniform(175, 230), rng.uniform(80, 140), rng.uniform(15, 35)], p=[0.5, 0.3, 0.2]))
    tex = bag_tex(rng, tone)
    mat = "glossy" if rng.random() < 0.7 else "fabric"

    def bag(l, w, h):
        return rbox((l, w, h), tex, mat, rng.uniform(0.45, 0.9), rng.uniform(0.15, 0.4), n=10,
                    dome=rng.uniform(0.0, 0.25), noise=rng.uniform(0.02, 0.07), rng=rng, flat_bottom=True)

    def top_of(faces):
        P = faces[0]["surf"]
        return P.reshape(-1, 3)[np.argmax(P[..., 2])]

    if rng.random() < 0.55:
        l, w = sorted([rng.uniform(0.3, 0.8), rng.uniform(0.2, 0.55)], reverse=True)
        h = rng.uniform(0.06, 0.28)
        parts = xform(bag(l, w, h), rotz(rng.normal(0, 0.05)), (0, 0, h / 2))
        x, y, z = top_of(parts)
        return unit("polybag", parts, None, [], [(x, y, z, 0.2)])
    pal, L, W, base = pallet_for(rng, [(1.2, 0.8), (1.2, 1.0)])
    parts, z, best = list(pal), base, None
    for k in range(int(rng.integers(2, 7))):
        hl = 0.0
        for _ in range(int(rng.integers(3, 8))):
            l, w = sorted([rng.uniform(0.3, 0.7), rng.uniform(0.2, 0.5)], reverse=True)
            h = rng.uniform(0.08, 0.25)
            x, y = rng.uniform(-(L - l) / 2, (L - l) / 2), rng.uniform(-(W - w) / 2, (W - w) / 2)
            b = xform(bag(l, w, h), rotz(rng.uniform(0, math.pi)), (x, y, z + h / 2))
            parts += b
            hl = max(hl, h)
            tp = top_of(b)
            if best is None or tp[2] > best[2]:
                best = tp
        z += hl * rng.uniform(0.6, 0.85)
    films = maybe_film(rng, parts[len(pal):], base, p=0.5)
    return unit("polybags", parts, (L, W), films, [(*best, 0.2)])


def tubes(rng):
    """Mailing / carpet tubes with end caps: one tube or a taped bundle lying on the floor (no top label)."""
    Lt, r = rng.uniform(0.5, 1.6), rng.uniform(0.035, 0.08)
    tone = rng.uniform(100, 175) if rng.random() < 0.7 else rng.uniform(185, 230)
    cap = rng.uniform(15, 40) if rng.random() < 0.5 else rng.uniform(185, 230)
    c = 0.03
    prof = [(0, 0), (r + 0.003, 0), (r + 0.003, c), (r, c), (r, Lt - c), (r + 0.003, Lt - c), (r + 0.003, Lt), (0, Lt)]
    vc = cumlen(sharpen(prof))
    pitch = rng.uniform(0.08, 0.2)

    def tex(f, w, h):
        t = G.plain(rng, w, h, tone, 2, 4)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        t *= np.where(np.abs(((yy + xx * 0.6) / px(pitch)) % 1.0 - 0.5) < 0.02, 0.8, 1.0)  # spiral winding seam
        a, b = px(vc[3]), px(vc[-5])
        t[:a] = cap
        t[b:] = cap
        return t
    m = int(rng.choice([1, 1, 3, 4, 5, 6, 7]))
    rows = [m] if m < 3 else [(m + 1) // 2, m // 2]
    parts, key = [], f"tube{rng.integers(1 << 30)}"
    for k, nr in enumerate(rows):
        for i in range(nr):
            y = (i - (nr - 1) / 2) * 2 * r
            parts.append(lathe_at(prof, tex, "cardboard", R=roty(math.pi / 2),
                                  t=(-Lt / 2 + rng.normal(0, 0.01 if m > 1 else 0), y, r + k * r * math.sqrt(3)), n=24,
                                  key=key))
    if m > 1:  # tape round the bundle
        half, hb = rows[0] * r, (r * 2 + (len(rows) - 1) * r * math.sqrt(3))
        tp = plain(rng, rng.uniform(25, 220), 2, 2)
        for x in (-Lt * 0.3, Lt * 0.3):
            ring = [(-half - 0.004, 0.0), (half + 0.004, 0.0), (half + 0.004, hb + 0.004), (-half - 0.004, hb + 0.004)]
            parts.append(tube(chaikin(np.array([(x, y, z) for y, z in ring]), 3), 0.003, tp, "strap", closed=True))
    return unit("tubes", parts, None, [], [])


def slat_crate(rng):
    """Wooden slatted crate on runners: slats with gaps on the sides (goods visible), plywood or slatted lid."""
    L, W = sorted([rng.uniform(0.7, 2.0), rng.uniform(0.5, 1.2)], reverse=True)
    Hc, rb, t, post = rng.uniform(0.45, 1.5), 0.1, 0.022, 0.07
    tone = rng.uniform(120, 200)
    wtex = lambda f, w, h: G.wood(rng, w, h, float(np.clip(tone + rng.normal(0, 15), 40, 230)))  # noqa: E731
    ys = (-W / 2 + 0.06, W / 2 - 0.06) if W < 0.9 else (-W / 2 + 0.06, 0.0, W / 2 - 0.06)
    parts = carrier([box(0, y, L, 0.09, 0.0, rb, wtex, "wood") for y in ys])
    parts.append(box(0, 0, L, W, rb, rb + 0.02, wtex, "wood"))
    z0, z1 = rb + 0.02, rb + Hc
    for sx in (-1, 1):
        for sy in (-1, 1):
            parts.append(box(sx * (L / 2 - post / 2), sy * (W / 2 - post / 2), post, post, z0, z1, wtex, "wood"))
    sw = rng.uniform(0.07, 0.12)
    nsl = max(2, int(Hc / rng.uniform(0.18, 0.3)))
    for zc in np.linspace(z0 + sw / 2, z1 - sw / 2, nsl):
        for s in (-1, 1):
            parts.append(box(0, s * (W / 2 + t / 2), L, t, zc - sw / 2, zc + sw / 2, wtex, "wood"))
            parts.append(box(s * (L / 2 + t / 2), 0, t, W + 2 * t, zc - sw / 2, zc + sw / 2, wtex, "wood"))
    mtone = rng.uniform(30, 150)
    mh = (z1 - z0 - 0.02) * rng.uniform(0.5, 0.95)
    parts.append(box(rng.normal(0, 0.03), rng.normal(0, 0.03), L * rng.uniform(0.5, 0.85), W * rng.uniform(0.5, 0.85),
                     z0, z0 + mh, lambda f, w, h: G.machine_face(rng, w, h, mtone), "paint"))
    labels = []
    if rng.random() < 0.5:
        parts.append(box(0, 0, L + 2 * t, W + 2 * t, z1, z1 + 0.015, lambda f, w, h: G.plywood(rng, w, h), "wood"))
        labels.append((rng.uniform(-0.3, 0.3) * L, rng.uniform(-0.25, 0.25) * W, z1 + 0.015, 0.3))
    else:
        for x in np.linspace(-L / 2 + sw / 2, L / 2 - sw / 2, max(3, int(L / rng.uniform(0.15, 0.3)))):
            parts.append(box(x, 0, sw, W + 2 * t, z1, z1 + t, wtex, "wood"))
    return unit("slat_crate", parts, None, [], labels, carrier_h=rb)


# ---------------------------------------------------------------------------------- v2 loads: soft cartons etc.
def is_box(p):
    return "surf" not in p and "polys" not in p and len(p["P"]) == 8


def soften(rng, parts):
    """Cartons of a v2 load (8-corner cardboard boxes) -> rounded boxes: rounded edges / corners, sides bulging
    under the load, sagging or domed tops, dents; one look for the whole unit. Prints / tape stay on the faces."""
    e = rng.uniform(0.05, 0.2)
    bulge = rng.uniform(0.0, 0.035) if rng.random() < 0.6 else 0.0
    dome = rng.uniform(-0.05, 0.06) if rng.random() < 0.6 else 0.0
    noise = rng.uniform(0.0, 0.012)
    cart = [p for p in parts if p.get("cull") and is_box(p)]
    skip = G.cull_fn(cart) if cart else None
    out = []
    for p in parts:
        if p.get("mat") != "cardboard" or not is_box(p) or p.get("top_tex") is not None:
            out.append(p)
            continue
        P = p["P"]
        ax = P[1] - P[0]
        size = (np.linalg.norm(ax), np.linalg.norm(P[3] - P[0]), P[4, 2] - P[0, 2])
        if min(size) < 0.03:
            out.append(p)
            continue
        faces = rbox(size, p["tex"], "cardboard", e * rng.uniform(0.8, 1.25), e * rng.uniform(0.8, 1.25), n=8,
                     bulge=bulge * rng.uniform(0.5, 1.0), dome=dome * rng.uniform(0.5, 1.0), noise=noise, rng=rng,
                     clust=2.0, flat_bottom=True)
        xform(faces, rotz(math.atan2(ax[1], ax[0])), P.mean(0))
        for f in faces:
            fi = {0: 0, 1: 5}.get(f["face"], 1)
            f["hidden"] = fi == 5 or bool(p.get("cull") and skip is not None
                                           and skip(f["surf"].reshape(-1, 3).mean(0), fi, P))
            f["soft"] = True
        out += faces
    return out


def top_label_spots(rng, parts, H, margin=0.04):
    """Flat spots on the highest box tops (v2 boxes or soft cartons) for a stick-on label: (x, y, z, max size)."""
    spots = []
    for p in parts:
        if p.get("dmc_ok") is False or p.get("mat") == "strap":
            continue
        if "surf" in p:
            if p.get("face") != 0 or not p.get("soft"):
                continue
            S = p["surf"]
            q0, q1, q3 = S[0, 0], S[0, -1], S[-1, 0]
            pts = S.reshape(-1, 3)
        elif is_box(p):
            q0, q1, q3 = p["P"][4], p["P"][5], p["P"][7]
            pts = p["P"][4:]
        else:
            continue
        if pts[:, 2].max() < H - 0.03:
            continue
        a, b = q1 - q0, q3 - q0
        la, lb = np.linalg.norm(a[:2]), np.linalg.norm(b[:2])
        if min(la, lb) < 0.2:
            continue
        fa, fb = [rng.uniform(-1, 1) * max(0.0, 0.5 - (margin + 0.12) / ln) for ln in (la, lb)]
        x, y = (q0 + a / 2 + b / 2 + fa * a + fb * b)[:2]
        near = np.hypot(pts[:, 0] - x, pts[:, 1] - y) < 0.1
        z = float(pts[near, 2].max() if near.any() else pts[:, 2].max())
        spots.append((float(x), float(y), z, min(la, lb) - 2 * margin))
    zmax = max((s[2] for s in spots), default=0.0)
    return [s for s in spots if s[2] > zmax - 0.005]  # not under a top sheet / a higher layer


def edge_protectors(rng, lo, hi, z0, z1):
    """Cardboard edge protectors (L-angles) on the 4 vertical edges of a load, often with a top sheet."""
    tone = float(rng.choice([rng.uniform(100, 165), rng.uniform(185, 230), rng.uniform(25, 50)], p=[0.6, 0.25, 0.15]))
    tex = plain(rng, tone, 2.5, 4)
    leg, t = rng.uniform(0.035, 0.06), rng.uniform(0.003, 0.005)
    parts = []
    if rng.random() < 0.6:
        z1 += 0.003
        parts.append(box((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, hi[0] - lo[0] + 0.01, hi[1] - lo[1] + 0.01, z1 - 0.003,
                         z1, plain(rng, rng.uniform(100, 170), 2.5, 4), "cardboard"))
    for sx in (-1, 1):
        for sy in (-1, 1):
            x, y = (hi[0] if sx > 0 else lo[0]), (hi[1] if sy > 0 else lo[1])
            parts.append(box(x - sx * (leg - t) / 2, y + sy * t / 2, leg + t, t, z0, z1, tex, "cardboard"))
            parts.append(box(x + sx * t / 2, y - sy * leg / 2, t, leg, z0, z1, tex, "cardboard"))
    return parts


# kind -> (builder, relative frequency). Round goods (drums ... tubes) together ~20 % of these units, ~8 % of all.
BUILDERS = {"sacks": (sacks, 1.0), "big_bag": (big_bag, 0.6), "beverages": (beverages, 0.6), "crates": (crates, 0.6),
            "roll_container": (roll_container, 0.5), "wrapped": (wrapped, 1.3), "octabin": (octabin, 0.6),
            "collars": (collars, 0.8), "polybags": (polybags, 0.9), "slat_crate": (slat_crate, 0.6), "ibc": (ibc, 0.4),
            "buckets": (buckets, 0.5), "long_goods": (long_goods, 0.35),
            "drums": (drums, 0.5), "cable_drum": (cable_drum, 0.3), "coils": (coils, 0.3), "tyres": (tyres, 0.4),
            "rolls_standing": (rolls_standing, 0.25), "rolls_lying": (rolls_lying, 0.3), "tubes": (tubes, 0.2)}


def make_unit(rng, kinds=None):
    names = [k for k in BUILDERS if kinds is None or k in kinds]
    w = np.array([BUILDERS[k][1] for k in names])
    return BUILDERS[names[rng.choice(len(names), p=w / w.sum())]][0](rng)


# ---------------------------------------------------------------------------------- pallet truck (smooth parts)
def build_truck(rng, tilt=None):
    """Hand pallet truck in its own frame (forks along +x from the heel at x=0 to the tips at x=1.15):
    tapered forks, chassis, round pump with head, steering wheels, tilted drawbar, rounded-triangle handle."""
    paint = rng.uniform(40, 150)                       # yellow / red / blue / grey paint in mono
    dark = rng.uniform(12, 40)
    grip = dark if rng.random() < 0.7 else paint
    pt = plain(rng, paint, 3, 5)
    parts = []
    for y in (-0.19, 0.19):
        o = [(0.0, y - 0.08), (1.07, y - 0.08), (1.15, y - 0.04), (1.15, y + 0.04), (1.07, y + 0.08), (0.0, y + 0.08)]
        parts.append(poly_part(extrude(o, 0.03, 0.085), pt, "paint"))
    chassis = [(-0.33, -0.2), (-0.25, -0.28), (0.06, -0.28), (0.06, 0.28), (-0.25, 0.28), (-0.33, 0.2)]
    parts.append(poly_part(extrude(chassis, 0.05, 0.19), pt, "paint"))
    px_ = -0.17
    parts.append(lathe_at([(0, 0), (0.065, 0), (0.065, 0.23), (0.09, 0.23), (0.09, 0.28), (0, 0.28)],
                          plain(rng, paint * rng.uniform(0.6, 1.0), 3, 5), "paint", t=(px_, 0, 0.19), n=24))
    for y in (-0.13, 0.13):  # steering wheels (axis along y)
        prof = [(0.02, -0.025), (0.085, -0.025), (0.09, -0.02), (0.09, 0.02), (0.085, 0.025), (0.02, 0.025),
                (0.02, -0.025)]
        parts.append(lathe_at(prof, plain(rng, dark, 2, 2), "rubber", R=rotx(math.pi / 2), t=(px_ - 0.02, y, 0.09), n=20))
    th = rng.uniform(0.05, 1.3) if tilt is None else tilt  # drawbar tilt from vertical, towards -x
    d = np.array([-math.sin(th), 0.0, math.cos(th)])
    B = np.array([px_, 0.0, 0.47])
    T = B + d * rng.uniform(0.9, 1.1)
    parts.append(tube([B, (B + T) / 2, T], 0.018, pt, "paint"))
    hl, hw = rng.uniform(0.2, 0.28), rng.uniform(0.11, 0.15)
    tri = np.array([T, T + d * hl + np.array([0, -hw, 0]), T + d * hl + np.array([0, hw, 0])])
    parts.append(tube(chaikin(tri, 3), 0.014, plain(rng, grip, 2, 2), "rubber", closed=True))
    return parts
