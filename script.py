"""
Diverse synthetic parcel dataset generator (Blender 5.x, headless).

    blender -b -P script.py          (or ./run.sh bg)

Parcels and shapes
------------------
* TWENTY-TWO PACKAGE TYPES, including:
      lying_tube   a cylinder resting on its side (rolled goods, poster tubes)
      barrel       a cylinder with a bulged or waisted profile
      tray_box     an open-topped box / crate with a raised rim
      crate        a timber-surfaced box with a slatted, banded lid
      envelope     a rigid document mailer, only a few millimetres thick
      tub          a strongly tapered bucket with a rim
  Cuboid parcels also draw lid profiles - `arch` (barrel vault), `hip`
  (hipped roof), `double_step` (two stacked lids) and `rim` (open crate) -
  and round ones `rim` and `bulge`. The analytic top profile is exact, so
  tape, labels and Data Matrix codes follow it.
* MORE SHAPES: n-gon prisms (3..12 sides, optionally irregular),
  truncated-pyramid boxes, gabled (ridged) boxes, wedges (one side higher
  than the other) and stepped/lidded boxes.
* NON-FLAT TOPS. Every cuboid parcel draws a top profile - flat, domed,
  truncated-pyramid, gabled, sloped, stepped or sagging - so roughly half the
  boxes have no flat lid. The profile is analytic (`Parcel.top_off`),
  so the tape, the shipping label and the Data Matrix follow the real surface,
  and the tape's wrap-over arc is anchored to the local top height.
* WIDE SIZE / ASPECT RANGE, plus a per-parcel stretch on one axis, so
  footprints run from small cubes to long flat slabs.

Surfaces and light
------------------
* PRINTED GRAPHICS ON THE PARCELS. A decal layer prints a generated
  graphic - shipping glyphs (fragile / umbrella / this-way-up arrows),
  stencil bars, warning triangles, hatch blocks, rings, barcodes and
  text-like address blocks - onto the box faces, clipped so it appears once
  rather than tiling. 64 unique bitmaps are generated per run.
* PATTERNS. The printed-pattern layers pick from eight sources (checker,
  bands, voronoi, brick, magic, gradient, noise blotches and a photographic
  pattern image), at a random blend mode, and a second independent layer can
  be laid over the first (plaids, grids, cell-over-stripes ...).
* TEXTURES. ~330 CC0 PBR sets (cardboard, chipboard, cork, paper, plastic,
  foil, fabric, wicker, styrofoam, wallpapers, leather, timber for crates)
  plus a wide floor pool including warehouse line markings, gratings and
  plate floors. The parcel shader occasionally crosses kinds for surfaces
  that are not "obvious".
* FLOORS: every image picks a different floor - all the floor materials
  already present in scene.blend (concrete, granite, marble, carpet, tiles,
  metal, plastic, wood veneer, ...) plus procedural ones (worn concrete,
  epoxy, rubber, tiled, lined/grid and a painted warehouse line marking).
  Scale/rotation/offset/brightness/roughness are re-randomized per image.
* LIGHT VARIATION. Lighting is drawn from seven named scenarios (soft studio,
  hard single source, overhead scanner flash, low backlight, dappled/gobo,
  dim ambient, mixed colour) over a pool of six lights that includes spots.
  Colour gels, elevation, hardness, world ambient and exposure ranges are all
  wide, and ~30% of frames put a patterned blocker above the scene so the
  light falls in patches. The scenario name is written into the label file.

Scene
-----
* PARCEL SHAPES are generated procedurally with bmesh instead of reusing the 4
  meshes baked into scene.blend, so every parcel in every image is a unique
  piece of geometry:
      box, box_soft, flat_box, tall_box   - cuboids (crisp .. very rounded)
      padded_mailer, poly_bag, sack       - soft/pillowed parcels
      tube, drum, round_tin, hex_box      - ROUND parcels (cylinders/prisms)
  All of them get per-vertex dents, bevelled edges and (for the flat-topped
  ones) a bulged top, so no two parcels share a silhouette.
* TAPE: real geometry, not a texture. Strips follow the parcel's top surface,
  wrap over the rounded edge along the true bevel arc and hang down the side.
  Variants: single seam, H-tape (3 strips), cross tape, plus plastic strapping
  bands. Tape materials: clear/glossy, brown kraft, white, dark strap.
* SHIPPING LABELS: a paper quad on the top face with a procedurally generated
  barcode + address block (reads well in grayscale).
* STACKING: parcels are sometimes stacked on top of each other (with the
  stacked parcel guaranteed to sit fully on its support).
* CAMERA jitter (height + small tilt), so the model does not overfit to one
  exact top-down pose. Labels stay correct: they are camera-space 3D boxes.

Outputs:
    gray_images/img_XXXX.png     640x640 grayscale
    labels/img_XXXX.json         camera intrinsics + 8 3D bbox corners/parcel

The 3D boxes are read back from the evaluated mesh bounds, so every new shape
and top profile is labelled correctly without any extra bookkeeping.

Everything is driven by the CONFIG block below. A few knobs can also be
overridden from the environment for quick tests, e.g.

    DS_NUM_IMAGES=4 DS_OUT_TAG=_smoke blender -b -P script.py
"""

import bpy
import bmesh
import json
import math
import os
import random
import sys
import mathutils
from mathutils import Vector
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent


def _env(name, default, cast=str):
    v = os.environ.get(name)
    return cast(v) if v not in (None, "") else default


NUM_IMAGES  = _env("DS_NUM_IMAGES", 3000, int)
START_INDEX = _env("DS_START_INDEX", 0, int)
OUT_TAG     = _env("DS_OUT_TAG", "")             # gray_images<TAG> / labels<TAG>
SEED        = _env("DS_SEED", 20260806, int)
# CYCLES on the GPU (OptiX/CUDA) is both prettier AND ~12x faster than EEVEE
# here, because headless EEVEE has no hardware GL context and falls back to
# software rasterization. Use DS_ENGINE=EEVEE if no GPU is available.
ENGINE      = _env("DS_ENGINE", "CYCLES").upper()  # CYCLES | EEVEE
CYCLES_SAMPLES = _env("DS_CYCLES_SAMPLES", 64, int)

# Square output. FX scales with RES so the field of view (~0.87 m of floor)
# stays identical - render at DS_RES=1280 if you need the Data Matrix codes to
# be big enough in pixels to actually decode (see check_dmc.py).
RES = _env("DS_RES", 640, int)
FX  = 1100.0 * RES / 640.0
EEVEE_SAMPLES = 64

# --- parcels per image ---------------------------------------------------
MIN_PARCELS = 2
MAX_PARCELS = 10
PLACE_ATTEMPTS = 120
GAP_MIN, GAP_MAX = 0.002, 0.030     # random clearance between footprints (m)

# Stacking: smaller packages resting on bigger ones.
STACK_PROB = 0.55                   # chance an image contains any stack
STACK_MAX_TOWERS = 2                # how many parcels can get something on top
STACK_MAX_ON_TOP = 4                # up to this many packages per support -
                                    # fewer if they don't fit
STACK_SECOND_LEVEL_PROB = 0.30      # ... and a third level on top of that
STACK_SIZE_SCALE = (0.40, 0.80)     # stacked parcels are scaled down by this
STACK_MIN_SCALE = 0.40              # ... but never below this

CORNER_DENT_PROB = 0.30             # rigid boxes with one corner knocked in
PATTERN_PROB = 0.62                 # printed pattern on the box surface
PATTERN2_PROB = 0.40                # ... and a second pattern layer over it
# A generated graphic (shipping glyph, stencil bar, warning triangle, barcode,
# address block, ...) printed onto the faces. Clipped rather than tiled, so it
# reads as printing on the carton and not as a wallpaper.
GRAPHIC_PROB = 0.45
GRAPHIC_POOL = 64                   # distinct generated bitmaps per run
GRAPHIC_PX = 192                    # bitmap resolution (square)
DAMAGE_PROB  = 0.38                 # torn patches showing the liner underneath
RIPPED_EDGE_PROB = 0.22             # geometry: one ragged/torn top edge
CRUMPLED_PROB = 0.32                # noticeably un-smooth surfaces
MIN_VISIBLE_CORNERS = 4             # 8 = fully inside frame; 4 allows truncation
TRUNCATION_FREE_PROB = 0.75         # share of parcels required to be fully visible

# One axis of a parcel is often stretched or squashed after the base size has
# been drawn, which widens the aspect-ratio distribution a lot (long slabs,
# near-cubes, narrow columns) without touching the per-type ranges.
ASPECT_STRETCH_PROB = 0.55
ASPECT_STRETCH = (0.60, 1.70)

# --- parcel type mix (weights) ------------------------------------------
# ~26% round/prismatic, ~17% soft (mailer/bag/sack), ~24% odd-topped boxes.
TYPE_WEIGHTS = {
    "box":            14,
    "box_soft":       10,
    "flat_box":        7,
    "tall_box":        5,
    "padded_mailer":   7,
    "poly_bag":        6,
    "sack":            5,
    "tube":            6,
    "drum":            7,
    "round_tin":       6,
    "hex_box":         4,
    "prism":           9,    # n-gon prism, 3..12 sides, sometimes irregular
    "tapered_box":     7,    # truncated pyramid (narrower or wider at the top)
    "gable_box":       6,    # ridged / roof top
    "wedge_box":       6,    # one side higher than the other
    "stepped_box":     6,    # smaller lid or step on the top face
    # --- wide-format / non-box packages -----------------------------------
    "lying_tube":      6,    # cylinder resting on its side
    "barrel":          5,    # bulged / waisted cylinder
    "tray_box":        5,    # open-topped box with a raised rim
    "crate":           5,    # timber crate, slatted lid
    "envelope":        5,    # rigid document mailer, millimetres thick
    "tub":             4,    # strongly tapered bucket with a rim
}
# "round" = built from a cone/prism primitive rather than a subdivided cube
ROUND_TYPES = ("tube", "drum", "round_tin", "hex_box", "prism", "barrel", "tub")
SOFT_TYPES = ("poly_bag", "sack")
# built by their own branch: neither a cone primitive nor a deformed cube
LYING_TYPES = ("lying_tube",)
# cuboid-derived types whose top profile is forced rather than drawn
BOX_TYPES = ("box", "box_soft", "flat_box", "tall_box", "tapered_box",
             "gable_box", "wedge_box", "stepped_box", "tray_box", "crate",
             "envelope")

# --- parcel size ranges (metres) ----------------------------------------
# The camera sees ~0.85-1.15 m of floor (camera height is randomized), so
# parcels have to stay small - but the spread is deliberately wide.
SIZE = {
    #                 min_w  max_w  min_d  max_d  min_h  max_h
    "box":           (0.085, 0.34,  0.085, 0.34,  0.05,  0.26),
    "box_soft":      (0.095, 0.31,  0.095, 0.31,  0.06,  0.24),
    "flat_box":      (0.14,  0.38,  0.10,  0.30,  0.022, 0.085),
    "tall_box":      (0.07,  0.19,  0.07,  0.19,  0.16,  0.38),
    "padded_mailer": (0.12,  0.32,  0.085, 0.24,  0.025, 0.095),
    "poly_bag":      (0.11,  0.32,  0.085, 0.26,  0.035, 0.13),
    "sack":          (0.12,  0.30,  0.10,  0.24,  0.075, 0.20),
    "tube":          (0.040, 0.115, 0.040, 0.115, 0.15,  0.36),   # w=d=diameter
    "drum":          (0.12,  0.28,  0.12,  0.28,  0.10,  0.30),
    "round_tin":     (0.095, 0.28,  0.095, 0.28,  0.030, 0.12),
    "hex_box":       (0.095, 0.26,  0.095, 0.26,  0.06,  0.22),
    "prism":         (0.085, 0.28,  0.085, 0.28,  0.05,  0.30),
    "tapered_box":   (0.10,  0.32,  0.10,  0.32,  0.06,  0.26),
    "gable_box":     (0.105, 0.33,  0.095, 0.29,  0.06,  0.22),
    "wedge_box":     (0.105, 0.33,  0.095, 0.29,  0.05,  0.20),
    "stepped_box":   (0.115, 0.33,  0.105, 0.31,  0.06,  0.22),
    # For lying_tube w is the LENGTH along the roll and h the diameter;
    # d is overwritten with h in build_parcel so the cross-section is round.
    "lying_tube":    (0.16,  0.40,  0.05,  0.16,  0.05,  0.16),
    "barrel":        (0.10,  0.26,  0.10,  0.26,  0.10,  0.30),
    "tray_box":      (0.13,  0.36,  0.11,  0.30,  0.045, 0.15),
    "crate":         (0.13,  0.36,  0.11,  0.31,  0.07,  0.24),
    "envelope":      (0.15,  0.36,  0.11,  0.27,  0.004, 0.020),
    "tub":           (0.10,  0.26,  0.10,  0.26,  0.08,  0.24),
}

# --- shape detail --------------------------------------------------------
# Sides of an n-gon prism. 4 is in there on purpose: a "prism" with 4 sides is
# a diamond-oriented box, which is a genuinely different silhouette from a
# cube because the footprint is rotated 45 degrees relative to its own axes.
PRISM_SIDES = [3, 4, 5, 6, 7, 8, 9, 10, 12]
PRISM_SIDE_WEIGHTS = [7, 8, 12, 14, 9, 14, 7, 10, 9]
PRISM_IRREGULAR_PROB = 0.30      # per-corner radius jitter -> irregular n-gon
PRISM_TAPER_PROB = 0.35          # top narrower (or wider) than the base

# Top profiles for cuboid parcels. "flat" still gets a small bulge + noise,
# so it is not literally flat - it is just not a deliberate shape.
TOP_STYLE_WEIGHTS = {
    "flat":    28,
    "dome":    14,
    "pyramid":  9,
    "gable":    9,
    "slope":    9,
    "step":     8,
    "sag":      5,
    "twin":     4,      # two ridges / a valley down the middle
    # --- curved / multi-facet lids ----------------------------------------
    "arch":     6,      # barrel vault along one axis
    "hip":      5,      # hipped roof: a ridge with four sloped facets
    "double_step": 5,   # a lid on a lid
}

# Cylinder profile: the radius is modulated over the height, which turns the
# straight-walled drums into barrels, waisted tins and stepped canisters.
BARREL_PROFILE_PROB = 0.30      # non-"barrel" round types that get a profile
BARREL_BULGE = (0.06, 0.20)     # amplitude as a fraction of the radius

# --- tape / labels -------------------------------------------------------
TAPE_PROB       = 0.82     # parcel carries packing tape
STRAP_PROB      = 0.16     # ... plus plastic strapping bands
LABEL_PROB      = 0.55     # ... plus a printed shipping label
WRAP_FILM_PROB  = 0.14     # ... or is shrink-wrapped in clear film

# Every package gets exactly one Data Matrix (ECC200) code on its top face.
# Payloads are kept short on purpose so the symbols stay 10x10 / 12x12: at this
# field of view (~736 px/m at RES=640) a 16x16 symbol smears into noise.
DMC_PROB      = 1.0
DMC_SIZE_MIN  = 0.026
DMC_SIZE_MAX  = 0.050
DMC_BEND_PROB = 0.35       # stuck over a crease/bulge instead of dead flat
DMC_SIZE_FLOOR = 0.006     # smallest code we will still print (narrow tube end caps)
TAPE_WIDTH_MIN  = 0.010
TAPE_WIDTH_MAX  = 0.024
# How far each layer floats above the parcel surface (m). The order matters:
# tape and shrink wrap go on first, then the shipping label and the Data Matrix
# are stuck on top of them - exactly as on a real parcel, and it stops the tape
# from covering the code.
TAPE_LIFT  = 0.00025
LIFT_FILM  = 0.00050
LIFT_LABEL = 0.00075
LIFT_DMC   = 0.00090

# --- camera --------------------------------------------------------------
# A wider height range means the same parcel can cover very different pixel
# areas from image to image, which is the cheapest apparent-scale augmentation
# there is. The label is a camera-space 3D box, so it follows automatically.
CAM_Z_MIN, CAM_Z_MAX = 1.18, 1.95
CAM_TILT_MAX = math.radians(7.0)
CAM_SHIFT    = 0.06        # lateral jitter (m)

# --- lighting ------------------------------------------------------------
LIGHT_DIST_MIN, LIGHT_DIST_MAX = 0.7, 4.0
LIGHT_ELEV_MIN, LIGHT_ELEV_MAX = math.radians(8), math.radians(89)
# Energies are quoted for a 1.5 m throw and rescaled by (dist/1.5)^2, then
# split across however many lights are on, so total exposure stays sane.
POINT_ENERGY_MIN, POINT_ENERGY_MAX = 18.0, 190.0
AREA_ENERGY_MIN, AREA_ENERGY_MAX = 14.0, 165.0
SPOT_ENERGY_MIN, SPOT_ENERGY_MAX = 25.0, 220.0
SUN_ENERGY_MIN, SUN_ENERGY_MAX = 0.3, 3.4
WORLD_STRENGTH_MIN, WORLD_STRENGTH_MAX = 0.0, 0.9
# Clamps on the per-scenario exposure ranges below. The upper clamp stops a
# pale floor under a bright ambient from washing the frame out completely.
EXPOSURE_MIN, EXPOSURE_MAX = -1.6, 0.45

# Lighting scenarios. Sampling a named setup and then jittering inside it
# gives structured variety (hard raking light, flat scanner light, patchy
# shadow, gloom) instead of one blurred-together average of everything.
LIGHT_SCENARIOS = {
    "studio_soft":     18,   # 2-4 big soft sources, low contrast
    "hard_single":     18,   # one small hard source, long shadows
    "overhead_flash":  14,   # near-vertical hard light, scanner-tunnel look
    "backlit_low":     12,   # grazing light from one side, elevation < 25 deg
    "dappled":         14,   # a patterned blocker overhead -> patchy light
    "dim_ambient":     12,   # weak sources, most of the light from the world
    "colour_mix":      12,   # two or three strongly gelled sources
}
# A blocker plane above the scene, invisible to the camera but casting a
# patterned shadow. Always on for "dappled", occasionally elsewhere.
GOBO_PROB = 0.12
GOBO_HEIGHT = (0.55, 1.05)      # above the floor, i.e. below the camera
COLOUR_GEL_PROB = 0.18          # a strongly coloured light outside colour_mix

# --- floor ---------------------------------------------------------------
FLOOR_SCALE = 2.6          # enlarge the plane so tilted views never see void
FLOOR_TEX_SCALE_MIN, FLOOR_TEX_SCALE_MAX = 1.0, 14.0

GRAY_DIR   = (SCRIPT_DIR / f"gray_images{OUT_TAG}").resolve()
LABEL_DIR  = (SCRIPT_DIR / f"labels{OUT_TAG}").resolve()
TEXTURE_DIR = (SCRIPT_DIR / "textures").resolve()

GRAY_DIR.mkdir(parents=True, exist_ok=True)
LABEL_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)

# ============================================================
# SCENE
# ============================================================
bpy.ops.wm.open_mainfile(filepath=str(SCRIPT_DIR / "scene.blend"))
scene = bpy.context.scene

scene.render.resolution_x = RES
scene.render.resolution_y = RES
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = 'PNG'
scene.render.image_settings.color_mode = 'BW'      # render straight to grayscale
scene.render.image_settings.color_depth = '8'
scene.render.use_persistent_data = False
scene.render.film_transparent = False

if ENGINE == "CYCLES":
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = CYCLES_SAMPLES
    scene.cycles.use_denoising = True
    scene.cycles.max_bounces = 6
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.02
    # GPU (OptiX > CUDA > HIP > ONEAPI > METAL), falls back to CPU silently
    try:
        prefs = bpy.context.preferences.addons['cycles'].preferences
        for dev_type in ('OPTIX', 'CUDA', 'HIP', 'ONEAPI', 'METAL'):
            try:
                prefs.compute_device_type = dev_type
            except TypeError:
                continue
            prefs.get_devices()
            gpus = [d for d in prefs.devices if d.type == dev_type]
            if gpus:
                for d in prefs.devices:
                    d.use = (d.type == dev_type)
                scene.cycles.device = 'GPU'
                print(f"cycles: {dev_type} on {[d.name for d in gpus]}")
                break
    except Exception as exc:
        print("cycles: GPU setup failed, using CPU:", exc)
else:
    scene.render.engine = 'BLENDER_EEVEE_NEXT' if 'BLENDER_EEVEE_NEXT' in {
        i.identifier for i in
        bpy.types.RenderSettings.bl_rna.properties['engine'].enum_items
    } else 'BLENDER_EEVEE'
    ev = scene.eevee
    for attr, val in (("taa_render_samples", EEVEE_SAMPLES),
                      ("use_raytracing", True),        # EEVEE Next: AO/reflections
                      ("use_shadows", True),
                      ("use_soft_shadows", True),
                      ("use_gtao", True),              # legacy EEVEE
                      ("shadow_cube_size", '2048'),
                      ("shadow_cascade_size", '2048')):
        if hasattr(ev, attr):
            try:
                setattr(ev, attr, val)
            except Exception:
                pass

camera = bpy.data.objects["Camera"]
camera.data.sensor_width = 36.0
camera.data.lens = 36.0 * FX / RES
camera.data.shift_x = 0.0
camera.data.shift_y = 0.0

floor = bpy.data.objects["Plane"]

# Hide every pre-existing mesh except the floor: the trolley / forklift / rig
# widgets must never leak into a frame.
for ob in bpy.data.objects:
    if ob.type == 'MESH' and ob is not floor:
        ob.hide_render = True
        ob.hide_viewport = True

# Enlarge the floor so a tilted camera can never see past its edge.
floor.scale = tuple(s * FLOOR_SCALE for s in floor.scale)
bpy.context.view_layer.update()

_dg = bpy.context.evaluated_depsgraph_get()
_fe = floor.evaluated_get(_dg)
_fc = [_fe.matrix_world @ Vector(c) for c in _fe.bound_box]
FLOOR_Z = max(c.z for c in _fc)

# The collection every generated object goes into.
WORK_COLL = scene.collection

# ============================================================
# TEXTURES
# ============================================================
# Photographic patterns: any loaded texture can also be used as a PRINT on a
# parcel rather than as the surface itself (the "photo" source of the pattern
# stack further down). Filled in once the textures are loaded.
PATTERN_IMAGES = []

_EXCLUDE = ("symbol", "icon", "sticker", "vector", "fragile", "logo",
            "base_color", "ware_", "box_wood", "package-vector",
            # stock-library previews: these carry visible watermarks, which
            # end up printed across the parcels and teach the detector to
            # look for text that is not there
            "stock-photo", "stockcake", "istockphoto", "shutterstock",
            "1000_f_", "360_f_", "260nw", "612x612", "-mpb4ax")


def _load_textures():
    cardboard, tape, paper = [], [], []
    if not TEXTURE_DIR.exists():
        return cardboard, tape, paper
    for f in sorted(os.listdir(TEXTURE_DIR)):
        low = f.lower()
        if not low.endswith((".png", ".jpg", ".jpeg")):
            continue
        if any(x in low for x in _EXCLUDE):
            continue
        try:
            img = bpy.data.images.load(str(TEXTURE_DIR / f))
        except Exception:
            print(f"  ! failed to load {f}")
            continue
        img.use_fake_user = True
        if "tape" in low or "adhesive" in low:
            tape.append(img)
        elif "paper" in low or "manila" in low or "card.jpg" in low:
            paper.append(img)
            cardboard.append(img)
        else:
            cardboard.append(img)
    return cardboard, tape, paper


TEX_CARDBOARD, TEX_TAPE, TEX_PAPER = _load_textures()
print(f"textures: {len(TEX_CARDBOARD)} cardboard, {len(TEX_TAPE)} tape, "
      f"{len(TEX_PAPER)} paper")

# ------------------------------------------------------------------
# PBR SETS (textures_pbr/, fetched by fetch_textures.py - CC0/ambientCG)
# Each set is a folder with Color.jpg + Roughness.jpg + Normal.jpg.
# ------------------------------------------------------------------
PBR_DIR = (SCRIPT_DIR / "textures_pbr").resolve()
_FLOOR_PREFIXES = ("concrete", "tiles", "asphalt", "road", "pavingstones",
                   "woodfloor", "planks", "metal", "corrugatedsteel",
                   "rubber", "carpet", "terrazzo", "marble", "granite",
                   "bricks", "ground", "gravel", "diamondplate",
                   # stone / architectural surfaces
                   "onyx", "travertine", "rock", "facade", "officeceiling",
                   "snow")
# folder prefix -> parcel surface family. Anything unrecognised is treated as
# cardboard, which is the safe default for a parcel.
# NOTE: "wallpaper" has to be tested after "paper" would have matched it -
# the loop below keeps the LAST match, and startswith("paper") is false for
# "Wallpaper001A" anyway, so order only matters for genuine prefixes.
_KIND_PREFIXES = (("cardboard", "cardboard"), ("chipboard", "cardboard"),
                  ("cork", "cardboard"),
                  ("paper", "paper"), ("wallpaper", "paper"),
                  ("plastic", "plastic"), ("foil", "foil"),
                  ("fabric", "fabric"), ("wicker", "fabric"),
                  ("leather", "fabric"),
                  ("styrofoam", "foam"),
                  # bare timber, for crates and trays. "woodfloor" is caught
                  # by _FLOOR_PREFIXES before this ever runs.
                  ("wood", "wood"))


def _load_pbr_sets():
    parcel, floor, sticker = [], [], []
    if not PBR_DIR.exists():
        return parcel, floor, sticker
    for d in sorted(PBR_DIR.iterdir()):
        if not d.is_dir() or not (d / "Color.jpg").exists():
            continue
        entry = {"name": d.name, "kind": "cardboard"}
        low = d.name.lower()
        for prefix, kind in _KIND_PREFIXES:
            if low.startswith(prefix):
                entry["kind"] = kind
        ok = True
        for key, fname, non_color in (("color", "Color.jpg", False),
                                      ("rough", "Roughness.jpg", True),
                                      ("normal", "Normal.jpg", True)):
            p = d / fname
            if not p.exists():
                entry[key] = None
                continue
            try:
                img = bpy.data.images.load(str(p))
            except Exception:
                ok = False
                break
            img.use_fake_user = True
            if non_color:
                img.colorspace_settings.name = 'Non-Color'
            entry[key] = img
        if not ok:
            continue
        if low.startswith("sticker"):
            sticker.append(entry)
        elif any(low.startswith(p) for p in _FLOOR_PREFIXES):
            floor.append(entry)
        else:
            parcel.append(entry)
    return parcel, floor, sticker


PBR_PARCEL, PBR_FLOOR, PBR_STICKER = _load_pbr_sets()
PBR_BY_KIND = {}
for _s in PBR_PARCEL:
    PBR_BY_KIND.setdefault(_s["kind"], []).append(_s)
print(f"pbr sets: {len(PBR_PARCEL)} parcel "
      f"({ {k: len(v) for k, v in PBR_BY_KIND.items()} }), "
      f"{len(PBR_FLOOR)} floor, {len(PBR_STICKER)} sticker")

# Any loaded image can also serve as a PRINTED pattern on a parcel (see the
# "photo" source in the pattern stack): a fabric weave printed on a carton, a
# tile pattern on a wrap, a wicker mesh on a paper sleeve. Floor sets are the
# richest source of large-scale structure, so they are in the pool too.
PATTERN_IMAGES.extend(
    [s["color"] for s in PBR_PARCEL + PBR_FLOOR if s.get("color") is not None]
    + list(TEX_CARDBOARD))
print(f"pattern images: {len(PATTERN_IMAGES)}")

# ============================================================
# DATA MATRIX (ECC200) ENCODER
# ============================================================
# Lives in dmc_ecc200.py so check_dmc.py can use the exact same encoder to
# verify the rendered codes. Blender does not put the script's directory on
# sys.path, so add it.
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import dmc_ecc200

dmc_encode = dmc_ecc200.encode

# ============================================================
# PRINTED GRAPHICS (generated bitmaps)
# ============================================================
# Real parcels are printed on: handling glyphs, stencilled part numbers,
# warning triangles, hatch bands, ruled address blocks. None of that comes out
# of a procedural noise texture, so it is drawn here into small RGBA bitmaps
# (transparent background, one ink colour) and stuck onto the box faces as a
# clipped decal. The pool is generated once per run.


class _Canvas:
    """Tiny 1-channel raster with just enough primitives to draw a print."""

    def __init__(self, n):
        self.n = n
        self.a = [0.0] * (n * n)

    def _put(self, x, y):
        if 0 <= x < self.n and 0 <= y < self.n:
            self.a[y * self.n + x] = 1.0

    def rect(self, x0, y0, x1, y1):
        for y in range(max(0, int(y0)), min(self.n, int(y1) + 1)):
            for x in range(max(0, int(x0)), min(self.n, int(x1) + 1)):
                self._put(x, y)

    def frame(self, x0, y0, x1, y1, t):
        self.rect(x0, y0, x1, y0 + t)
        self.rect(x0, y1 - t, x1, y1)
        self.rect(x0, y0, x0 + t, y1)
        self.rect(x1 - t, y0, x1, y1)

    def line(self, x0, y0, x1, y1, t):
        dx, dy = x1 - x0, y1 - y0
        ll = dx * dx + dy * dy
        r = t * 0.5
        for y in range(max(0, int(min(y0, y1) - r - 1)),
                       min(self.n, int(max(y0, y1) + r + 2))):
            for x in range(max(0, int(min(x0, x1) - r - 1)),
                           min(self.n, int(max(x0, x1) + r + 2))):
                if ll < 1e-9:
                    u = 0.0
                else:
                    u = max(0.0, min(1.0, ((x - x0) * dx + (y - y0) * dy) / ll))
                px, py = x0 + u * dx, y0 + u * dy
                if (x - px) ** 2 + (y - py) ** 2 <= r * r:
                    self._put(x, y)

    def ring(self, cx, cy, rad, t):
        r0, r1 = rad - t * 0.5, rad + t * 0.5
        for y in range(max(0, int(cy - r1 - 1)), min(self.n, int(cy + r1 + 2))):
            for x in range(max(0, int(cx - r1 - 1)),
                           min(self.n, int(cx + r1 + 2))):
                dd = math.hypot(x - cx, y - cy)
                if r0 <= dd <= r1:
                    self._put(x, y)

    def disc(self, cx, cy, rad):
        self.ring(cx, cy, rad * 0.5, rad)

    def arc(self, cx, cy, rad, a0, a1, t):
        steps = max(8, int(rad * abs(a1 - a0)))
        pts = [(cx + rad * math.cos(a0 + (a1 - a0) * k / steps),
                cy + rad * math.sin(a0 + (a1 - a0) * k / steps))
               for k in range(steps + 1)]
        for (xa, ya), (xb, yb) in zip(pts, pts[1:]):
            self.line(xa, ya, xb, yb, t)

    def poly(self, pts, t, close=True):
        seq = list(pts) + ([pts[0]] if close else [])
        for (xa, ya), (xb, yb) in zip(seq, seq[1:]):
            self.line(xa, ya, xb, yb, t)


def _g_bars(c, x0, y0, x1, y1, t):
    """A ruled text block: a few lines of varying length."""
    rows = random.randint(3, 7)
    step = (y1 - y0) / rows
    th = max(2.0, min(t, step * 0.55))
    for i in range(rows):
        yy = y0 + step * (i + 0.25)
        c.rect(x0, yy, x0 + (x1 - x0) * random.uniform(0.35, 1.0), yy + th)


def _g_barcode(c, x0, y0, x1, y1):
    """Vertical bars of random widths - reads as a printed code strip."""
    x = x0
    while x < x1:
        w = random.uniform(2.0, 7.0)
        if random.random() < 0.62:
            c.rect(x, y0, min(x + w, x1), y1)
        x += w + random.uniform(2.0, 6.0)


def _g_arrow(c, cx, cy, s, t):
    """One 'this way up' arrow."""
    c.line(cx, cy - s, cx, cy + s, t)
    c.line(cx, cy + s, cx - s * 0.55, cy + s * 0.42, t)
    c.line(cx, cy + s, cx + s * 0.55, cy + s * 0.42, t)


def _g_glass(c, cx, cy, s, t):
    """The 'fragile' stemmed glass."""
    c.poly([(cx - s * 0.45, cy + s), (cx - s * 0.28, cy - s * 0.1),
            (cx + s * 0.28, cy - s * 0.1), (cx + s * 0.45, cy + s)],
           t, close=False)
    c.line(cx, cy - s * 0.1, cx, cy - s * 0.72, t)
    c.line(cx - s * 0.42, cy - s * 0.72, cx + s * 0.42, cy - s * 0.72, t)


def _g_umbrella(c, cx, cy, s, t):
    """'Keep dry': a dome with a hooked stem."""
    c.arc(cx, cy, s * 0.85, 0.0, math.pi, t)
    c.line(cx - s * 0.85, cy, cx + s * 0.85, cy, t)
    c.line(cx, cy, cx, cy - s * 0.8, t)
    c.arc(cx - s * 0.2, cy - s * 0.8, s * 0.2, -math.pi, 0.0, t)


def _g_triangle(c, cx, cy, s, t):
    """Warning triangle with an exclamation bar."""
    c.poly([(cx, cy + s), (cx + s * 0.95, cy - s * 0.75),
            (cx - s * 0.95, cy - s * 0.75)], t)
    c.rect(cx - t * 0.6, cy - s * 0.35, cx + t * 0.6, cy + s * 0.45)
    c.rect(cx - t * 0.6, cy - s * 0.62, cx + t * 0.6, cy - s * 0.5)


def _g_hatch(c, x0, y0, x1, y1, t):
    step = random.uniform(10.0, 26.0)
    x = x0 - (y1 - y0)
    while x < x1:
        c.line(x, y0, x + (y1 - y0), y1, t)
        x += step


def _g_grid(c, x0, y0, x1, y1, t):
    nx = random.randint(2, 5)
    ny = random.randint(2, 5)
    for i in range(nx + 1):
        x = x0 + (x1 - x0) * i / nx
        c.line(x, y0, x, y1, t)
    for j in range(ny + 1):
        y = y0 + (y1 - y0) * j / ny
        c.line(x0, y, x1, y, t)


_SEVEN_SEG = {
    0: "abcdef", 1: "bc", 2: "abdeg", 3: "abcdg", 4: "bcfg",
    5: "acdfg", 6: "acdefg", 7: "abc", 8: "abcdefg", 9: "abcdfg",
}


def _g_digits(c, x0, y0, x1, y1, t):
    """A row of stencilled seven-segment digits."""
    n = random.randint(3, 6)
    gap = (x1 - x0) / n
    w, h = gap * 0.62, (y1 - y0)
    for k in range(n):
        bx = x0 + gap * k + gap * 0.19
        segs = _SEVEN_SEG[random.randint(0, 9)]
        if "a" in segs:
            c.rect(bx, y0 + h - t, bx + w, y0 + h)
        if "g" in segs:
            c.rect(bx, y0 + h * 0.5 - t * 0.5, bx + w, y0 + h * 0.5 + t * 0.5)
        if "d" in segs:
            c.rect(bx, y0, bx + w, y0 + t)
        if "f" in segs:
            c.rect(bx, y0 + h * 0.5, bx + t, y0 + h)
        if "b" in segs:
            c.rect(bx + w - t, y0 + h * 0.5, bx + w, y0 + h)
        if "e" in segs:
            c.rect(bx, y0, bx + t, y0 + h * 0.5)
        if "c" in segs:
            c.rect(bx + w - t, y0, bx + w, y0 + h * 0.5)


_GLYPHS = (_g_arrow, _g_glass, _g_umbrella, _g_triangle)


def make_graphic_image(name):
    """Draw one printed graphic into a transparent RGBA image."""
    n = GRAPHIC_PX
    c = _Canvas(n)
    t = random.uniform(0.020, 0.055) * n            # stroke weight
    m = random.uniform(0.06, 0.16) * n              # margin
    lo, hi = m, n - m

    kind = random.choices(
        ["glyphs", "block", "code", "band", "seal", "stencil"],
        [24, 20, 14, 12, 14, 16])[0]

    if kind == "glyphs":
        # one to three handling symbols in a row
        k = random.choices([1, 2, 3], [46, 34, 20])[0]
        fn = random.choice(_GLYPHS)
        s = min((hi - lo) / (2.2 * k), (hi - lo) * 0.45)
        for i in range(k):
            cx = lo + (hi - lo) * (i + 0.5) / k
            fn(c, cx, (lo + hi) * 0.5, s * 0.9, t)
        if random.random() < 0.45:
            c.frame(lo * 0.6, lo * 0.6, n - lo * 0.6, n - lo * 0.6, t * 0.8)

    elif kind == "block":
        # a framed address / description block
        if random.random() < 0.7:
            c.frame(lo, lo, hi, hi, t * 0.9)
            c.rect(lo, hi - (hi - lo) * 0.22, hi, hi - (hi - lo) * 0.19)
            _g_bars(c, lo + t * 2, lo + t * 2, hi - t * 2,
                    hi - (hi - lo) * 0.26, t)
        else:
            _g_bars(c, lo, lo, hi, hi, t)

    elif kind == "code":
        _g_barcode(c, lo, lo + (hi - lo) * 0.30, hi, hi)
        _g_digits(c, lo, lo, hi, lo + (hi - lo) * 0.22, max(2.0, t * 0.7))

    elif kind == "band":
        y0 = lo + (hi - lo) * random.uniform(0.18, 0.42)
        y1 = y0 + (hi - lo) * random.uniform(0.20, 0.38)
        c.rect(lo, y0 - t * 0.6, hi, y0)
        c.rect(lo, y1, hi, y1 + t * 0.6)
        if random.random() < 0.55:
            _g_hatch(c, lo, y0, hi, y1, t)
        else:
            _g_grid(c, lo, y0, hi, y1, t * 0.7)

    elif kind == "seal":
        cx = cy = n * 0.5
        rad = (hi - lo) * 0.46
        c.ring(cx, cy, rad, t)
        if random.random() < 0.6:
            c.ring(cx, cy, rad * random.uniform(0.62, 0.82), t * 0.7)
        if random.random() < 0.7:
            _g_bars(c, cx - rad * 0.5, cy - rad * 0.35,
                    cx + rad * 0.5, cy + rad * 0.35, t * 0.8)
        else:
            random.choice(_GLYPHS)(c, cx, cy, rad * 0.55, t)

    else:   # stencil
        _g_digits(c, lo, lo + (hi - lo) * 0.55, hi, hi, t)
        if random.random() < 0.6:
            _g_bars(c, lo, lo, hi, lo + (hi - lo) * 0.42, t * 0.8)
        else:
            _g_hatch(c, lo, lo, hi, lo + (hi - lo) * 0.42, t)

    ink = (random.uniform(0.03, 0.30) if random.random() < 0.8
           else random.uniform(0.72, 0.95))
    px = [0.0] * (n * n * 4)
    for i, a in enumerate(c.a):
        o = i * 4
        px[o] = px[o + 1] = px[o + 2] = ink
        px[o + 3] = a
    img = bpy.data.images.new(name, width=n, height=n, alpha=True)
    img.pixels = px
    img.use_fake_user = True
    return img


PRINT_GRAPHICS = [make_graphic_image(f"print_{i:02d}")
                  for i in range(GRAPHIC_POOL)]
print(f"printed graphics: {len(PRINT_GRAPHICS)}")


# ============================================================
# SHADER HELPERS
# ============================================================


def _find(sockets, name, kind=None):
    """Socket lookup by name (+ optional type) - ShaderNodeMix has several
    same-named sockets, so the type filter matters."""
    for s in sockets:
        if s.name == name and (kind is None or s.type == kind):
            return s
    return None


def _set(node, name, value, kind=None):
    s = _find(node.inputs, name, kind)
    if s is None:
        return False
    try:
        if hasattr(s.default_value, "__len__") and not isinstance(value, (list, tuple)):
            n = len(s.default_value)
            s.default_value = tuple([value] * min(n, 3)) + ((1.0,) if n == 4 else ())
        else:
            s.default_value = value
        return True
    except Exception:
        return False


def _mix_rgb(nt, blend='MIX'):
    """ShaderNodeMix wired for colours; returns (node, fac, a, b, out)."""
    n = nt.nodes.new("ShaderNodeMix")
    n.data_type = 'RGBA'
    n.blend_type = blend
    fac = _find(n.inputs, "Factor", 'VALUE')
    a = _find(n.inputs, "A", 'RGBA')
    b = _find(n.inputs, "B", 'RGBA')
    out = _find(n.outputs, "Result", 'RGBA')
    return n, fac, a, b, out


def _gray(v, a=1.0):
    return (v, v, v, a)


# ============================================================
# PRINTED-PATTERN STACK
# ============================================================
# One "stack" = own mapping + eight pattern sources + a one-hot selector +
# a colour ramp. Built once per material, re-parameterized per parcel.
PATTERN_SOURCES = ("checker", "bands", "voronoi", "brick", "magic",
                   "gradient", "blotch", "photo")
PATTERN_WEIGHTS = [20, 22, 12, 14, 9, 8, 8, 14]


def _pattern_stack(nt, tc, index, y):
    """Build one printed-pattern layer; returns the refs needed to drive it."""
    N, L = nt.nodes, nt.links
    x0 = -1500

    mp = N.new("ShaderNodeMapping");        mp.location = (x0, y)
    L.new(tc.outputs["Object"], mp.inputs["Vector"])

    chk = N.new("ShaderNodeTexChecker");    chk.location = (x0 + 220, y + 240)
    L.new(mp.outputs["Vector"], chk.inputs["Vector"])

    wv = N.new("ShaderNodeTexWave");        wv.location = (x0 + 220, y + 60)
    wv.wave_type = 'BANDS'
    L.new(mp.outputs["Vector"], wv.inputs["Vector"])

    vor = N.new("ShaderNodeTexVoronoi");    vor.location = (x0 + 220, y - 120)
    L.new(mp.outputs["Vector"], vor.inputs["Vector"])

    brk = N.new("ShaderNodeTexBrick");      brk.location = (x0 + 220, y - 300)
    L.new(mp.outputs["Vector"], brk.inputs["Vector"])

    mag = N.new("ShaderNodeTexMagic");      mag.location = (x0 + 220, y - 480)
    L.new(mp.outputs["Vector"], mag.inputs["Vector"])

    grd = N.new("ShaderNodeTexGradient");   grd.location = (x0 + 220, y - 660)
    L.new(mp.outputs["Vector"], grd.inputs["Vector"])

    blo = N.new("ShaderNodeTexNoise");      blo.location = (x0 + 220, y - 840)
    L.new(mp.outputs["Vector"], blo.inputs["Vector"])

    pho = N.new("ShaderNodeTexImage");      pho.location = (x0 + 220, y - 1040)
    pho.projection = 'BOX'
    pho.projection_blend = 0.3
    pho.extension = 'REPEAT'
    L.new(mp.outputs["Vector"], pho.inputs["Vector"])

    # one-hot selector: setting f_k = 1 picks source k and shadows everything
    # to its left, so only the last factor set to 1 matters
    s1, f1, a1, b1, o1 = _mix_rgb(nt);      s1.location = (x0 + 460, y + 150)
    L.new(chk.outputs["Color"], a1)
    L.new(wv.outputs["Color"], b1)

    s2, f2, a2, b2, o2 = _mix_rgb(nt);      s2.location = (x0 + 620, y + 60)
    L.new(o1, a2)
    L.new(_find(vor.outputs, "Distance", 'VALUE') or vor.outputs[0], b2)

    s3, f3, a3, b3, o3 = _mix_rgb(nt);      s3.location = (x0 + 780, y - 30)
    L.new(o2, a3)
    L.new(brk.outputs["Color"], b3)

    s4, f4, a4, b4, o4 = _mix_rgb(nt);      s4.location = (x0 + 940, y - 120)
    L.new(o3, a4)
    L.new(mag.outputs["Color"], b4)

    s5, f5, a5, b5, o5 = _mix_rgb(nt);      s5.location = (x0 + 1100, y - 210)
    L.new(o4, a5)
    L.new(grd.outputs["Color"], b5)

    s6, f6, a6, b6, o6 = _mix_rgb(nt);      s6.location = (x0 + 1260, y - 300)
    L.new(o5, a6)
    L.new(blo.outputs["Fac"], b6)

    s7, f7, a7, b7, o7 = _mix_rgb(nt);      s7.location = (x0 + 1420, y - 390)
    L.new(o6, a7)
    L.new(pho.outputs["Color"], b7)

    ramp = N.new("ShaderNodeValToRGB");     ramp.location = (x0 + 1600, y)
    L.new(o7, ramp.inputs["Fac"])

    return dict(index=index, mp=mp, chk=chk, wv=wv, vor=vor, brk=brk, mag=mag,
                grd=grd, blo=blo, pho=pho,
                facs=(f1, f2, f3, f4, f5, f6, f7), ramp=ramp,
                out=ramp.outputs["Color"])


def randomize_pattern_stack(p, strength_mix):
    """Point one pattern layer at a random source with random parameters.

    `strength_mix` is the ShaderNodeMix that folds the layer into the base
    colour; its blend type and factor are randomized here too.
    """
    src = random.choices(PATTERN_SOURCES, PATTERN_WEIGHTS)[0]
    if src == "photo" and not PATTERN_IMAGES:
        src = "bands"
    # one-hot from the right: the last factor at 1.0 wins
    onehot = {"checker":  (0, 0, 0, 0, 0, 0, 0),
              "bands":    (1, 0, 0, 0, 0, 0, 0),
              "voronoi":  (0, 1, 0, 0, 0, 0, 0),
              "brick":    (0, 0, 1, 0, 0, 0, 0),
              "magic":    (0, 0, 0, 1, 0, 0, 0),
              "gradient": (0, 0, 0, 0, 1, 0, 0),
              "blotch":   (0, 0, 0, 0, 0, 1, 0),
              "photo":    (0, 0, 0, 0, 0, 0, 1)}[src]
    for fac, v in zip(p["facs"], onehot):
        fac.default_value = float(v)

    s = math.exp(random.uniform(math.log(2.0), math.log(30.0)))
    p["mp"].inputs["Scale"].default_value = (
        s * random.uniform(0.25, 1.0), s * random.uniform(0.25, 1.0), s)
    p["mp"].inputs["Rotation"].default_value = (
        0.0, 0.0, random.uniform(0, math.tau))
    p["mp"].inputs["Location"].default_value = (
        random.uniform(-10, 10), random.uniform(-10, 10),
        random.uniform(-10, 10))

    if src == "checker":
        p["chk"].inputs["Scale"].default_value = random.uniform(1.2, 9.0)
    elif src == "bands":
        p["wv"].inputs["Scale"].default_value = random.uniform(0.4, 6.0)
        p["wv"].inputs["Distortion"].default_value = random.uniform(0.0, 6.0)
        p["wv"].wave_profile = random.choice(['SIN', 'SAW', 'TRI'])
        p["wv"].wave_type = random.choices(
            ['BANDS', 'RINGS'], [82, 18])[0]
        p["wv"].bands_direction = random.choice(['X', 'Y', 'DIAGONAL'])
    elif src == "voronoi":
        p["vor"].feature = random.choice(
            ['F1', 'SMOOTH_F1', 'DISTANCE_TO_EDGE', 'DISTANCE_TO_EDGE'])
        p["vor"].inputs["Scale"].default_value = random.uniform(2.0, 20.0)
        if "Randomness" in p["vor"].inputs:
            p["vor"].inputs["Randomness"].default_value = random.uniform(0.2, 1.0)
    elif src == "brick":
        b = p["brk"]
        b.inputs["Color1"].default_value = _gray(random.uniform(0.25, 0.95))
        b.inputs["Color2"].default_value = _gray(random.uniform(0.25, 0.95))
        b.inputs["Mortar"].default_value = _gray(random.uniform(0.05, 0.6))
        b.inputs["Scale"].default_value = random.uniform(1.5, 12.0)
        b.inputs["Mortar Size"].default_value = random.uniform(0.005, 0.05)
        b.inputs["Brick Width"].default_value = random.uniform(0.3, 1.4)
        b.inputs["Row Height"].default_value = random.uniform(0.15, 0.6)
        b.offset = random.choice([0.0, 0.25, 0.5, 0.5])
    elif src == "magic":
        p["mag"].inputs["Scale"].default_value = random.uniform(2.0, 14.0)
        p["mag"].inputs["Distortion"].default_value = random.uniform(0.5, 3.0)
        p["mag"].turbulence_depth = random.randint(1, 4)
    elif src == "gradient":
        # a printed fade / colour block across the panel. The generic mapping
        # scale above is far too fine for a gradient - it would tile into
        # stripes - so it is overridden here.
        p["grd"].gradient_type = random.choice(
            ['LINEAR', 'QUADRATIC', 'EASING', 'DIAGONAL', 'SPHERICAL',
             'RADIAL'])
        g = random.uniform(0.6, 2.5)
        p["mp"].inputs["Scale"].default_value = (g, g, g)
    elif src == "blotch":
        b = p["blo"]
        b.inputs["Scale"].default_value = random.uniform(1.0, 7.0)
        b.inputs["Detail"].default_value = random.uniform(0.0, 4.0)
        if "Roughness" in b.inputs:
            b.inputs["Roughness"].default_value = random.uniform(0.3, 0.8)
        if "Distortion" in b.inputs:
            b.inputs["Distortion"].default_value = random.uniform(0.0, 2.5)
    else:   # photo: another texture used as a print rather than as the board
        p["pho"].image = random.choice(PATTERN_IMAGES)
        p["pho"].projection_blend = random.uniform(0.15, 0.45)
        g = random.uniform(1.5, 9.0)
        p["mp"].inputs["Scale"].default_value = (
            g * random.uniform(0.7, 1.3), g * random.uniform(0.7, 1.3), g)

    ramp = p["ramp"].color_ramp
    ramp.interpolation = random.choices(
        ['CONSTANT', 'LINEAR', 'EASE'], [46, 40, 14])[0]
    lo, hi = sorted((random.uniform(0.15, 0.6), random.uniform(0.4, 0.9)))
    ramp.elements[0].position = lo
    ramp.elements[1].position = max(hi, lo + 0.03)
    # printed ink is darker than the board; occasionally it is a light overprint
    if random.random() < 0.78:
        ramp.elements[0].color = _gray(random.uniform(0.12, 0.65))
        ramp.elements[1].color = _gray(1.0)
    else:
        ramp.elements[0].color = _gray(1.0)
        ramp.elements[1].color = _gray(random.uniform(0.6, 0.95))

    strength_mix.blend_type = random.choices(
        ['MULTIPLY', 'MIX', 'OVERLAY', 'SOFT_LIGHT', 'BURN'],
        [40, 22, 20, 12, 6])[0]


# ============================================================
# PARCEL MATERIALS
# ============================================================
# One flexible graph, re-parameterized per parcel per frame:
#
#   TexCoord(Object) -> Mapping -> ImageTexture(BOX proj) -> HSV ----.
#                                                                   |-> Mix -> Mix(mult grunge) -> Base Color
#                                          flat colour (RGB) -------'
#   Noise + Wave -> Mix -> Bump  ->  Bump(from image)  -> Normal
#   Noise -> MapRange -> Roughness
#
# Pools are re-parameterized per frame instead of being rebuilt, so they must
# be big enough that two objects alive in the SAME frame never share an entry
# (they share the datablock, so the second randomize would restyle the first).
# A frame can hold up to 9 parcels on the floor plus two stacks, so the pools
# are sized well above that - two objects alive in the same frame must never
# land on the same slot.
PARCEL_MAT_POOL = 28
TAPE_MAT_POOL = 20
STRAP_MAT_POOL = 14
LABEL_MAT_POOL = 18
DMC_MAT_POOL = 24


def build_parcel_material(i):
    mat = bpy.data.materials.new(f"P_Mat_{i:02d}")
    # Pooled materials have zero users between frames (their objects are
    # deleted); without a fake user the periodic orphans_purge() FREES them and
    # the cached node references below dangle -> hard crash mid-run.
    mat.use_fake_user = True
    mat.use_nodes = True
    nt = mat.node_tree
    N, L = nt.nodes, nt.links
    N.clear()

    out = N.new("ShaderNodeOutputMaterial");      out.location = (1000, 0)
    bsdf = N.new("ShaderNodeBsdfPrincipled");     bsdf.location = (700, 0)
    L.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    tc = N.new("ShaderNodeTexCoord");             tc.location = (-1400, 0)
    mp = N.new("ShaderNodeMapping");              mp.location = (-1200, 100)
    L.new(tc.outputs["Object"], mp.inputs["Vector"])

    img = N.new("ShaderNodeTexImage");            img.location = (-1000, 300)
    img.projection = 'BOX'
    img.projection_blend = 0.3
    img.extension = 'REPEAT'
    if TEX_CARDBOARD:
        img.image = TEX_CARDBOARD[0]
    L.new(mp.outputs["Vector"], img.inputs["Vector"])

    hsv = N.new("ShaderNodeHueSaturation");       hsv.location = (-760, 300)
    L.new(img.outputs["Color"], hsv.inputs["Color"])

    flat = N.new("ShaderNodeRGB");                flat.location = (-760, 60)
    flat.outputs[0].default_value = _gray(0.75)

    mixf, mixf_fac, mixf_a, mixf_b, mixf_out = _mix_rgb(nt)
    mixf.location = (-520, 200)
    L.new(hsv.outputs["Color"], mixf_a)
    L.new(flat.outputs[0], mixf_b)

    # grunge / dirt
    gn = N.new("ShaderNodeTexNoise");             gn.location = (-1000, -260)
    gn.noise_dimensions = '3D'
    L.new(mp.outputs["Vector"], gn.inputs["Vector"])
    gramp = N.new("ShaderNodeValToRGB");          gramp.location = (-800, -260)
    L.new(gn.outputs["Fac"], gramp.inputs["Fac"])

    mixg, mixg_fac, mixg_a, mixg_b, mixg_out = _mix_rgb(nt, 'MULTIPLY')
    mixg.location = (-260, 120)
    L.new(mixf_out, mixg_a)
    L.new(gramp.outputs["Color"], mixg_b)

    # ---- printed pattern -------------------------------------------------
    # Two independent layers, each of which can be any of five sources
    # (checker, bands, voronoi cells, brick/grid, magic turbulence). The
    # source is selected with one-hot mix factors instead of rebuilding the
    # graph, and the layers are combined at a random blend mode, so a single
    # material datablock covers stripes, checks, plaids, grids, mosaics and
    # blotches.
    pat1 = _pattern_stack(nt, tc, 1, 620)
    pat2 = _pattern_stack(nt, tc, 2, 1180)

    pmix, pmix_fac, pmix_a, pmix_b, pmix_out = _mix_rgb(nt, 'MULTIPLY')
    pmix.location = (-40, 200)
    L.new(mixg_out, pmix_a)
    L.new(pat1["out"], pmix_b)

    qmix, qmix_fac, qmix_a, qmix_b, qmix_out = _mix_rgb(nt, 'MULTIPLY')
    qmix.location = (110, 240)
    L.new(pmix_out, qmix_a)
    L.new(pat2["out"], qmix_b)

    # ---- printed graphic decal -------------------------------------------
    # A generated bitmap (handling glyph, stencil, barcode, address block)
    # printed onto the faces. CLIP extension means the image appears ONCE per
    # box-projection face instead of tiling, so it reads as printing and not
    # as a wrapping paper. Its alpha drives the mix, so the transparent
    # background leaves the board untouched.
    gmp = N.new("ShaderNodeMapping");             gmp.location = (-1200, -1500)
    L.new(tc.outputs["Object"], gmp.inputs["Vector"])
    gimg = N.new("ShaderNodeTexImage");           gimg.location = (-1000, -1500)
    gimg.projection = 'BOX'
    gimg.projection_blend = 0.0
    gimg.extension = 'CLIP'
    L.new(gmp.outputs["Vector"], gimg.inputs["Vector"])

    gstr = N.new("ShaderNodeMath");               gstr.location = (-700, -1560)
    gstr.operation = 'MULTIPLY'
    L.new(gimg.outputs["Alpha"], gstr.inputs[0])
    gstr.inputs[1].default_value = 0.0

    gmix, gmix_fac, gmix_a, gmix_b, gmix_out = _mix_rgb(nt)
    gmix.location = (180, 260)
    L.new(qmix_out, gmix_a)
    L.new(gimg.outputs["Color"], gmix_b)
    L.new(gstr.outputs[0], gmix_fac)

    # ---- damage: torn patches showing the lighter liner underneath
    dn = N.new("ShaderNodeTexNoise");             dn.location = (-1000, -1000)
    L.new(mp.outputs["Vector"], dn.inputs["Vector"])
    dramp = N.new("ShaderNodeValToRGB");          dramp.location = (-760, -1000)
    dramp.color_ramp.interpolation = 'CONSTANT'   # hard torn edges
    L.new(dn.outputs["Fac"], dramp.inputs["Fac"])
    ripc = N.new("ShaderNodeRGB");                ripc.location = (-760, -1200)

    dmix, dmix_fac, dmix_a, dmix_b, dmix_out = _mix_rgb(nt)
    dmix.location = (330, 120)
    L.new(gmix_out, dmix_a)
    L.new(ripc.outputs[0], dmix_b)
    L.new(dramp.outputs["Color"], dmix_fac)
    L.new(dmix_out, bsdf.inputs["Base Color"])

    # micro relief: fine noise + corrugation wave
    fn = N.new("ShaderNodeTexNoise");             fn.location = (-1000, -560)
    L.new(mp.outputs["Vector"], fn.inputs["Vector"])
    wv = N.new("ShaderNodeTexWave");              wv.location = (-1000, -800)
    wv.wave_type = 'BANDS'
    wv.bands_direction = 'X'
    L.new(mp.outputs["Vector"], wv.inputs["Vector"])

    mixh, mixh_fac, mixh_a, mixh_b, mixh_out = _mix_rgb(nt)
    mixh.location = (-700, -650)
    L.new(fn.outputs["Fac"], mixh_a)
    L.new(wv.outputs["Fac"], mixh_b)

    bump_d = N.new("ShaderNodeBump");             bump_d.location = (-420, -600)
    L.new(mixh_out, bump_d.inputs["Height"])
    bump_i = N.new("ShaderNodeBump");             bump_i.location = (-200, -400)
    L.new(img.outputs["Color"], bump_i.inputs["Height"])
    L.new(bump_d.outputs["Normal"], bump_i.inputs["Normal"])
    L.new(bump_i.outputs["Normal"], bsdf.inputs["Normal"])

    # roughness: procedural breakup, or a real Roughness map when the surface
    # comes from a PBR set. (No normal map here on purpose: these meshes are
    # unwrapped-free and use BOX projection, so tangent-space normals would be
    # wrong - the colour-driven Bump above does that job instead.)
    rr = N.new("ShaderNodeMapRange");             rr.location = (-420, -140)
    L.new(gn.outputs["Fac"], rr.inputs["Value"])

    img_r = N.new("ShaderNodeTexImage");          img_r.location = (-1000, -80)
    img_r.projection = 'BOX'
    img_r.projection_blend = 0.3
    img_r.extension = 'REPEAT'
    L.new(mp.outputs["Vector"], img_r.inputs["Vector"])
    rr2 = N.new("ShaderNodeMapRange");            rr2.location = (-700, -80)
    L.new(img_r.outputs["Color"], rr2.inputs["Value"])

    rmix = N.new("ShaderNodeMix");                rmix.location = (200, -140)
    rmix.data_type = 'FLOAT'
    r_fac = _find(rmix.inputs, "Factor", 'VALUE')
    r_a = [s for s in rmix.inputs if s.name == "A" and s.type == 'VALUE'][0]
    r_b = [s for s in rmix.inputs if s.name == "B" and s.type == 'VALUE'][0]
    L.new(rr.outputs["Result"], r_a)
    L.new(rr2.outputs["Result"], r_b)
    L.new(_find(rmix.outputs, "Result", 'VALUE'), bsdf.inputs["Roughness"])

    return mat, dict(img=img, mp=mp, hsv=hsv, flat=flat, mixf=mixf,
                     mixf_fac=mixf_fac, gn=gn, gramp=gramp, mixg_fac=mixg_fac,
                     fn=fn, wv=wv, mixh_fac=mixh_fac, bump_d=bump_d,
                     bump_i=bump_i, rr=rr, bsdf=bsdf,
                     img_r=img_r, rr2=rr2, r_fac=r_fac,
                     pat1=pat1, pat2=pat2,
                     pmix=pmix, pmix_fac=pmix_fac,
                     qmix=qmix, qmix_fac=qmix_fac,
                     gmp=gmp, gimg=gimg, gstr=gstr, gmix=gmix,
                     dn=dn, dramp=dramp, ripc=ripc)


PARCEL_MATS = [build_parcel_material(i) for i in range(PARCEL_MAT_POOL)]

# Surface presets: the list is the draw pool, so repeats are weights.
SURFACE_PRESETS = ["kraft", "kraft", "kraft", "kraft",
                   "kraft_dark", "kraft_dark", "white_board", "white_board",
                   "bleached", "paper", "paper", "plastic", "foam", "foil",
                   "wood"]

# which family of PBR sets each preset should draw from
PRESET_PBR_KIND = {
    "kraft": "cardboard", "kraft_dark": "cardboard",
    "white_board": "paper", "bleached": "paper", "paper": "paper",
    "plastic": "plastic", "fabric": "fabric", "foam": "foam", "foil": "foil",
    "wood": "wood",
}
PBR_TEXTURE_PROB = 0.72     # else fall back to the photo textures in textures/
PBR_CROSS_KIND_PROB = 0.22  # ... and sometimes ignore the preset's own family


def randomize_parcel_material(entry, preset=None, cyl=False):
    mat, r = entry
    preset = preset or random.choice(SURFACE_PRESETS)

    # Surface source: either a downloaded PBR set (tileable colour + a real
    # roughness map) or one of the plain photo textures.
    pbr_pool = PBR_BY_KIND.get(PRESET_PBR_KIND.get(preset, "cardboard"), [])
    if not pbr_pool or random.random() < PBR_CROSS_KIND_PROB:
        # a cardboard box wrapped in patterned paper, a foam box in a plastic
        # sleeve, ... - the family is a default, not a rule
        pbr_pool = PBR_PARCEL or pbr_pool
    if pbr_pool and random.random() < PBR_TEXTURE_PROB:
        s = random.choice(pbr_pool)
        r["img"].image = s["color"]
        r["img_r"].image = s["rough"]
        r["r_fac"].default_value = 1.0 if s["rough"] is not None else 0.0
    else:
        if TEX_CARDBOARD:
            pool = TEX_PAPER if (preset in ("paper", "bleached") and TEX_PAPER) \
                else TEX_CARDBOARD
            r["img"].image = random.choice(pool)
        r["img_r"].image = None
        r["r_fac"].default_value = 0.0
    r["img"].projection_blend = random.uniform(0.15, 0.45)
    r["img_r"].projection_blend = r["img"].projection_blend

    s = random.uniform(2.5, 11.0)
    r["mp"].inputs["Scale"].default_value = (
        s * random.uniform(0.8, 1.25), s * random.uniform(0.8, 1.25), s)
    r["mp"].inputs["Location"].default_value = (
        random.uniform(-50, 50), random.uniform(-50, 50), random.uniform(-50, 50))
    # BOX projection assigns each face to the mapping axis its normal is
    # closest to. If the mapping is rotated much, a face lands between two
    # axes and gets a visible diagonal ghost seam - so keep the projection
    # frame nearly aligned with the parcel and get variety from offset/scale.
    r["mp"].inputs["Rotation"].default_value = (
        random.uniform(-0.05, 0.05),
        random.uniform(-0.05, 0.05),
        random.choice([0.0, math.pi * 0.5, math.pi, math.pi * 1.5])
        + random.uniform(-0.10, 0.10))

    hue = 0.5 + random.uniform(-0.02, 0.02)

    if preset == "kraft":
        sat, val = random.uniform(0.75, 1.25), random.uniform(0.45, 1.05)
        flat_c, flat_fac = _gray(random.uniform(0.22, 0.45)), random.uniform(0.0, 0.25)
        rough = (random.uniform(0.62, 0.78), random.uniform(0.82, 0.98))
        bump_img, bump_det = random.uniform(0.15, 0.45), random.uniform(0.05, 0.22)
        spec = random.uniform(0.25, 0.45)
    elif preset == "kraft_dark":
        sat, val = random.uniform(0.8, 1.3), random.uniform(0.35, 0.7)
        flat_c, flat_fac = _gray(random.uniform(0.14, 0.3)), random.uniform(0.05, 0.35)
        rough = (random.uniform(0.6, 0.8), random.uniform(0.85, 1.0))
        bump_img, bump_det = random.uniform(0.2, 0.5), random.uniform(0.08, 0.25)
        spec = random.uniform(0.2, 0.4)
    elif preset == "white_board":
        sat, val = random.uniform(0.05, 0.3), random.uniform(0.9, 1.35)
        flat_c, flat_fac = _gray(random.uniform(0.5, 0.75)), random.uniform(0.3, 0.6)
        rough = (random.uniform(0.45, 0.6), random.uniform(0.7, 0.9))
        bump_img, bump_det = random.uniform(0.15, 0.35), random.uniform(0.05, 0.2)
        spec = random.uniform(0.35, 0.55)
    elif preset == "bleached":
        sat, val = random.uniform(0.1, 0.45), random.uniform(0.8, 1.25)
        flat_c, flat_fac = _gray(random.uniform(0.45, 0.7)), random.uniform(0.3, 0.65)
        rough = (random.uniform(0.55, 0.7), random.uniform(0.8, 0.95))
        bump_img, bump_det = random.uniform(0.1, 0.3), random.uniform(0.04, 0.18)
        spec = random.uniform(0.3, 0.5)
    elif preset == "plastic":
        sat, val = random.uniform(0.0, 0.25), random.uniform(0.7, 1.2)
        flat_c, flat_fac = _gray(random.uniform(0.2, 0.7)), random.uniform(0.8, 1.0)
        rough = (random.uniform(0.10, 0.22), random.uniform(0.28, 0.5))
        bump_img, bump_det = random.uniform(0.0, 0.08), random.uniform(0.15, 0.5)
        spec = random.uniform(0.5, 0.75)
    elif preset == "fabric":            # woven sacks
        sat, val = random.uniform(0.4, 1.1), random.uniform(0.5, 1.1)
        flat_c, flat_fac = _gray(random.uniform(0.2, 0.55)), random.uniform(0.0, 0.3)
        rough = (random.uniform(0.7, 0.85), random.uniform(0.9, 1.0))
        bump_img, bump_det = random.uniform(0.3, 0.6), random.uniform(0.1, 0.3)
        spec = random.uniform(0.15, 0.35)
    elif preset == "foam":              # expanded polystyrene shippers
        sat, val = random.uniform(0.0, 0.2), random.uniform(1.0, 1.45)
        flat_c, flat_fac = _gray(random.uniform(0.66, 0.9)), random.uniform(0.35, 0.75)
        rough = (random.uniform(0.72, 0.88), random.uniform(0.9, 1.0))
        bump_img, bump_det = random.uniform(0.2, 0.5), random.uniform(0.1, 0.35)
        spec = random.uniform(0.15, 0.35)
    elif preset == "foil":              # metallized / mylar wrapped
        sat, val = random.uniform(0.0, 0.3), random.uniform(0.75, 1.3)
        flat_c, flat_fac = _gray(random.uniform(0.35, 0.85)), random.uniform(0.4, 0.9)
        rough = (random.uniform(0.06, 0.2), random.uniform(0.22, 0.48))
        bump_img, bump_det = random.uniform(0.05, 0.3), random.uniform(0.2, 0.7)
        spec = random.uniform(0.6, 0.9)
    elif preset == "wood":              # timber crates / trays
        sat, val = random.uniform(0.5, 1.15), random.uniform(0.45, 1.0)
        flat_c, flat_fac = _gray(random.uniform(0.25, 0.5)), random.uniform(0.0, 0.25)
        rough = (random.uniform(0.5, 0.68), random.uniform(0.75, 0.92))
        bump_img, bump_det = random.uniform(0.25, 0.6), random.uniform(0.08, 0.25)
        spec = random.uniform(0.25, 0.45)
    else:  # paper
        sat, val = random.uniform(0.3, 0.9), random.uniform(0.65, 1.15)
        flat_c, flat_fac = _gray(random.uniform(0.32, 0.62)), random.uniform(0.2, 0.55)
        rough = (random.uniform(0.55, 0.72), random.uniform(0.78, 0.95))
        bump_img, bump_det = random.uniform(0.05, 0.22), random.uniform(0.05, 0.2)
        spec = random.uniform(0.3, 0.5)

    _set(r["hsv"], "Hue", hue)
    _set(r["hsv"], "Saturation", sat)
    _set(r["hsv"], "Value", val)
    r["flat"].outputs[0].default_value = flat_c
    r["mixf_fac"].default_value = 1.0 if not TEX_CARDBOARD else flat_fac

    # dirt / stains
    r["gn"].inputs["Scale"].default_value = random.uniform(1.5, 9.0)
    r["gn"].inputs["Detail"].default_value = random.uniform(2.0, 8.0)
    if "Roughness" in r["gn"].inputs:
        r["gn"].inputs["Roughness"].default_value = random.uniform(0.4, 0.7)
    e0, e1 = r["gramp"].color_ramp.elements[0], r["gramp"].color_ramp.elements[1]
    dirt = random.uniform(0.55, 1.0)
    e0.position = random.uniform(0.15, 0.45)
    e0.color = _gray(dirt)
    e1.position = random.uniform(0.55, 0.9)
    e1.color = _gray(1.0)
    r["mixg_fac"].default_value = random.uniform(0.25, 0.9)

    # micro relief
    r["fn"].inputs["Scale"].default_value = random.uniform(20.0, 90.0)
    r["fn"].inputs["Detail"].default_value = random.uniform(2.0, 6.0)
    r["wv"].inputs["Scale"].default_value = random.uniform(25.0, 120.0)
    r["wv"].inputs["Distortion"].default_value = random.uniform(0.0, 3.0)
    r["mixh_fac"].default_value = (random.uniform(0.0, 0.45)
                                   if preset != "plastic" else 0.0)
    _set(r["bump_d"], "Strength", bump_det)
    _set(r["bump_i"], "Strength", bump_img)
    _set(r["bump_d"], "Distance", random.uniform(0.001, 0.004))
    _set(r["bump_i"], "Distance", random.uniform(0.002, 0.008))

    _set(r["rr"], "From Min", 0.15)
    _set(r["rr"], "From Max", 0.85)
    _set(r["rr"], "To Min", rough[0])
    _set(r["rr"], "To Max", rough[1])
    # remap the roughness MAP into the same range, with a little spread so two
    # parcels using the same set still differ
    _set(r["rr2"], "From Min", random.uniform(0.0, 0.25))
    _set(r["rr2"], "From Max", random.uniform(0.75, 1.0))
    _set(r["rr2"], "To Min", max(0.02, rough[0] - random.uniform(0.0, 0.15)))
    _set(r["rr2"], "To Max", min(1.0, rough[1] + random.uniform(0.0, 0.1)))

    _set(r["bsdf"], "Metallic",
         random.uniform(0.45, 0.95) if preset == "foil" else 0.0)
    _set(r["bsdf"], "Specular IOR Level", spec)
    _set(r["bsdf"], "IOR", 1.45)

    # ---- printed pattern (up to two layers) ------------------------------
    if random.random() < PATTERN_PROB:
        randomize_pattern_stack(r["pat1"], r["pmix"])
        r["pmix_fac"].default_value = random.uniform(0.2, 0.85)
        if random.random() < PATTERN2_PROB:
            randomize_pattern_stack(r["pat2"], r["qmix"])
            r["qmix_fac"].default_value = random.uniform(0.15, 0.7)
        else:
            r["qmix_fac"].default_value = 0.0
    else:
        r["pmix_fac"].default_value = 0.0
        r["qmix_fac"].default_value = 0.0

    # ---- printed graphic -------------------------------------------------
    if PRINT_GRAPHICS and random.random() < GRAPHIC_PROB:
        r["gimg"].image = random.choice(PRINT_GRAPHICS)
        # printed size in metres. The material has no idea how big the parcel
        # is, so the range is chosen to look sane on everything from a mailer
        # to a drum; on the small ones the print is simply cropped by the face.
        size = random.uniform(0.055, 0.20)
        gs = 1.0 / size
        r["gmp"].inputs["Scale"].default_value = (gs, gs, gs)
        # centre the [0,1] tile on the object, then push it around a bit
        r["gmp"].inputs["Location"].default_value = (
            0.5 + random.uniform(-0.28, 0.28),
            0.5 + random.uniform(-0.28, 0.28),
            0.5 + random.uniform(-0.28, 0.28))
        r["gmp"].inputs["Rotation"].default_value = (
            0.0, 0.0, random.choice([0.0, math.pi * 0.5, math.pi,
                                     math.pi * 1.5]) + random.uniform(-0.12, 0.12))
        r["gmix"].blend_type = random.choices(
            ['MIX', 'MULTIPLY', 'OVERLAY'], [56, 30, 14])[0]
        r["gstr"].inputs[1].default_value = random.uniform(0.55, 1.0)
    else:
        r["gstr"].inputs[1].default_value = 0.0

    # ---- rips / damage ---------------------------------------------------
    dramp = r["dramp"].color_ramp
    if random.random() < DAMAGE_PROB:
        r["dn"].inputs["Scale"].default_value = random.uniform(6.0, 22.0)
        r["dn"].inputs["Detail"].default_value = random.uniform(8.0, 15.0)
        if "Distortion" in r["dn"].inputs:
            r["dn"].inputs["Distortion"].default_value = random.uniform(0.3, 2.0)
        dramp.elements[0].position = 0.0
        dramp.elements[0].color = _gray(0.0)
        # high threshold -> a few small ragged patches, not big blobs
        dramp.elements[1].position = random.uniform(0.72, 0.92)
        dramp.elements[1].color = _gray(random.uniform(0.45, 0.95))
        # exposed liner is usually lighter; sometimes a dark scuff
        r["ripc"].outputs[0].default_value = _gray(
            random.uniform(0.45, 0.78) if random.random() < 0.7
            else random.uniform(0.06, 0.26))
    else:
        dramp.elements[0].color = _gray(0.0)
        dramp.elements[1].color = _gray(0.0)     # mask off -> no damage
    return mat


# ============================================================
# TAPE + STRAP MATERIALS
# ============================================================
def build_tape_material(i):
    mat = bpy.data.materials.new(f"T_Mat_{i:02d}")
    mat.use_fake_user = True        # see build_parcel_material()
    mat.use_nodes = True
    nt = mat.node_tree
    N, L = nt.nodes, nt.links
    N.clear()

    out = N.new("ShaderNodeOutputMaterial");   out.location = (700, 0)
    bsdf = N.new("ShaderNodeBsdfPrincipled");  bsdf.location = (420, 0)
    L.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    tc = N.new("ShaderNodeTexCoord");          tc.location = (-900, 0)
    mp = N.new("ShaderNodeMapping");           mp.location = (-700, 0)
    L.new(tc.outputs["Object"], mp.inputs["Vector"])

    img = N.new("ShaderNodeTexImage");         img.location = (-480, 250)
    img.projection = 'BOX'
    img.projection_blend = 0.3
    if TEX_TAPE:
        img.image = TEX_TAPE[0]
    L.new(mp.outputs["Vector"], img.inputs["Vector"])
    hsv = N.new("ShaderNodeHueSaturation");    hsv.location = (-260, 250)
    L.new(img.outputs["Color"], hsv.inputs["Color"])

    flat = N.new("ShaderNodeRGB");             flat.location = (-260, 30)
    mix, fac, a, b, res = _mix_rgb(nt);        mix.location = (60, 150)
    L.new(hsv.outputs["Color"], a)
    L.new(flat.outputs[0], b)
    L.new(res, bsdf.inputs["Base Color"])

    # Relief: stretched noise (wrinkles) x voronoi edges (crease lines and
    # trapped air bubbles). The same signal drives Alpha, so clear tape looks
    # like film with creases rather than a flat translucent band.
    wr = N.new("ShaderNodeTexNoise");          wr.location = (-480, -300)
    L.new(mp.outputs["Vector"], wr.inputs["Vector"])
    vor = N.new("ShaderNodeTexVoronoi");       vor.location = (-480, -560)
    vor.feature = 'DISTANCE_TO_EDGE'
    L.new(mp.outputs["Vector"], vor.inputs["Vector"])
    vramp = N.new("ShaderNodeValToRGB");       vramp.location = (-280, -560)
    L.new(vor.outputs["Distance"], vramp.inputs["Fac"])

    hmix, hfac, ha, hb, hout = _mix_rgb(nt, 'MULTIPLY')
    hmix.location = (-40, -420)
    L.new(wr.outputs["Fac"], ha)
    L.new(vramp.outputs["Color"], hb)

    bump = N.new("ShaderNodeBump");            bump.location = (200, -420)
    L.new(hout, bump.inputs["Height"])
    L.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    amr = N.new("ShaderNodeMapRange");         amr.location = (200, -650)
    L.new(hout, amr.inputs["Value"])
    L.new(amr.outputs["Result"], bsdf.inputs["Alpha"])

    return mat, dict(img=img, hsv=hsv, flat=flat, fac=fac, mp=mp, wr=wr,
                     bump=bump, bsdf=bsdf, vor=vor, vramp=vramp, hfac=hfac,
                     amr=amr)


TAPE_MATS = [build_tape_material(i) for i in range(TAPE_MAT_POOL)]
STRAP_MATS = [build_tape_material(100 + i) for i in range(STRAP_MAT_POOL)]


def _set_blend(mat, blended):
    if hasattr(mat, "surface_render_method"):
        mat.surface_render_method = 'BLENDED' if blended else 'DITHERED'
    elif hasattr(mat, "blend_method"):
        mat.blend_method = 'BLEND' if blended else 'OPAQUE'


def randomize_tape_material(entry, kind=None):
    mat, r = entry
    kind = kind or random.choices(
        ["clear", "brown", "white", "printed"], [34, 40, 18, 8])[0]

    r["mp"].inputs["Scale"].default_value = tuple(
        [random.uniform(6.0, 22.0)] * 3)
    r["mp"].inputs["Location"].default_value = (
        random.uniform(-40, 40), random.uniform(-40, 40), random.uniform(-40, 40))
    r["mp"].inputs["Rotation"].default_value = (0, 0, random.uniform(0, math.tau))

    if TEX_TAPE:
        r["img"].image = random.choice(TEX_TAPE)

    alpha = 1.0
    if kind == "clear":
        # mostly see-through: it should read as a glossy sheen over the box,
        # not as a white band painted on top of it
        r["flat"].outputs[0].default_value = _gray(random.uniform(0.7, 0.95))
        r["fac"].default_value = random.uniform(0.55, 0.9)
        rough = random.uniform(0.04, 0.18)
        alpha = random.uniform(0.12, 0.45)
        spec = 0.8
        bump_s = random.uniform(0.25, 0.6)
    elif kind == "brown":
        r["flat"].outputs[0].default_value = _gray(random.uniform(0.3, 0.55))
        r["fac"].default_value = random.uniform(0.0, 0.45) if TEX_TAPE else 1.0
        _set(r["hsv"], "Value", random.uniform(0.7, 1.3))
        _set(r["hsv"], "Saturation", random.uniform(0.6, 1.2))
        rough = random.uniform(0.18, 0.45)
        spec = 0.55
        bump_s = random.uniform(0.1, 0.35)
    elif kind == "white":
        r["flat"].outputs[0].default_value = _gray(random.uniform(0.72, 0.95))
        r["fac"].default_value = random.uniform(0.7, 1.0)
        rough = random.uniform(0.2, 0.5)
        spec = 0.5
        bump_s = random.uniform(0.08, 0.3)
    else:  # printed / coloured tape
        r["flat"].outputs[0].default_value = _gray(random.uniform(0.18, 0.42))
        r["fac"].default_value = random.uniform(0.5, 0.95)
        rough = random.uniform(0.12, 0.4)
        spec = 0.6
        bump_s = random.uniform(0.1, 0.3)

    _set_blend(mat, kind == "clear")
    _set(r["bsdf"], "Roughness", rough)
    _set(r["bsdf"], "Specular IOR Level", spec)
    _set(r["bsdf"], "Metallic", 0.0)
    _set(r["bsdf"], "IOR", 1.5)

    # wrinkles: stretched along the strip so creases run across it
    r["wr"].inputs["Scale"].default_value = random.uniform(20.0, 120.0)
    r["wr"].inputs["Detail"].default_value = random.uniform(1.0, 5.0)
    r["vor"].inputs["Scale"].default_value = random.uniform(8.0, 60.0)
    bubbles = random.random() < 0.5
    r["vramp"].color_ramp.elements[0].position = (
        random.uniform(0.0, 0.06) if bubbles else 0.0)
    r["vramp"].color_ramp.elements[1].position = random.uniform(0.10, 0.45)
    r["hfac"].default_value = random.uniform(0.25, 0.85)
    _set(r["bump"], "Strength", bump_s)
    _set(r["bump"], "Distance", random.uniform(0.0004, 0.0018))

    # Alpha is driven by the crease signal (linked), so it must be set here
    # rather than on the BSDF socket.
    _set(r["amr"], "From Min", 0.0)
    _set(r["amr"], "From Max", 1.0)
    if kind == "clear":
        lo = alpha * random.uniform(0.35, 0.8)
        _set(r["amr"], "To Min", lo)
        _set(r["amr"], "To Max", min(1.0, alpha * random.uniform(1.0, 1.8)))
    else:
        _set(r["amr"], "To Min", 1.0)
        _set(r["amr"], "To Max", 1.0)
    return mat


def randomize_strap_material(entry):
    """Plastic strapping band: dark, glossy, near-flat."""
    mat, r = entry
    r["fac"].default_value = 1.0
    v = random.choice([random.uniform(0.03, 0.12), random.uniform(0.6, 0.9)])
    r["flat"].outputs[0].default_value = _gray(v)
    _set_blend(mat, False)
    _set(r["bsdf"], "Roughness", random.uniform(0.12, 0.4))
    _set(r["bsdf"], "Specular IOR Level", 0.7)
    _set(r["amr"], "To Min", 1.0)
    _set(r["amr"], "To Max", 1.0)
    r["wr"].inputs["Scale"].default_value = random.uniform(60.0, 200.0)
    r["vor"].inputs["Scale"].default_value = random.uniform(30.0, 120.0)
    r["hfac"].default_value = random.uniform(0.0, 0.3)
    _set(r["bump"], "Strength", random.uniform(0.05, 0.2))
    return mat


# ============================================================
# SHIPPING-LABEL MATERIAL  (procedural barcode + address block)
# ============================================================
def build_label_material(i):
    mat = bpy.data.materials.new(f"L_Mat_{i:02d}")
    mat.use_fake_user = True        # see build_parcel_material()
    mat.use_nodes = True
    nt = mat.node_tree
    N, L = nt.nodes, nt.links
    N.clear()

    out = N.new("ShaderNodeOutputMaterial");   out.location = (900, 0)
    bsdf = N.new("ShaderNodeBsdfPrincipled");  bsdf.location = (650, 0)
    L.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    tc = N.new("ShaderNodeTexCoord");          tc.location = (-1100, 0)
    sep = N.new("ShaderNodeSeparateXYZ");      sep.location = (-900, 0)
    L.new(tc.outputs["UV"], sep.inputs["Vector"])

    # --- barcode: vertical stripes, restricted to a horizontal band --------
    bmap = N.new("ShaderNodeMapping");         bmap.location = (-900, 350)
    L.new(tc.outputs["UV"], bmap.inputs["Vector"])
    bw = N.new("ShaderNodeTexWave");           bw.location = (-700, 350)
    bw.wave_type = 'BANDS'
    bw.bands_direction = 'X'
    bw.wave_profile = 'SAW'
    L.new(bmap.outputs["Vector"], bw.inputs["Vector"])
    bramp = N.new("ShaderNodeValToRGB");       bramp.location = (-500, 350)
    bramp.color_ramp.interpolation = 'CONSTANT'
    bramp.color_ramp.elements[0].position = 0.0
    bramp.color_ramp.elements[0].color = _gray(0.0)
    bramp.color_ramp.elements[1].position = 0.45
    bramp.color_ramp.elements[1].color = _gray(1.0)
    L.new(bw.outputs["Fac"], bramp.inputs["Fac"])

    bband = N.new("ShaderNodeValToRGB");       bband.location = (-700, 100)
    bband.color_ramp.interpolation = 'CONSTANT'
    e = bband.color_ramp.elements
    e[0].position = 0.0
    e[0].color = _gray(0.0)
    e[1].position = 0.10
    e[1].color = _gray(1.0)
    e2 = bband.color_ramp.elements.new(0.38)
    e2.color = _gray(0.0)
    L.new(sep.outputs["Y"], bband.inputs["Fac"])

    bmul, bfac, ba, bb, bout = _mix_rgb(nt, 'MULTIPLY')
    bmul.location = (-260, 250)
    bfac.default_value = 1.0
    L.new(bramp.outputs["Color"], ba)
    L.new(bband.outputs["Color"], bb)

    # --- text block: horizontal bars in the upper part ---------------------
    tmap = N.new("ShaderNodeMapping");         tmap.location = (-900, -350)
    L.new(tc.outputs["UV"], tmap.inputs["Vector"])
    tw = N.new("ShaderNodeTexWave");           tw.location = (-700, -350)
    tw.wave_type = 'BANDS'
    tw.bands_direction = 'Y'
    tw.wave_profile = 'SAW'
    L.new(tmap.outputs["Vector"], tw.inputs["Vector"])
    tramp = N.new("ShaderNodeValToRGB");       tramp.location = (-500, -350)
    tramp.color_ramp.interpolation = 'CONSTANT'
    tramp.color_ramp.elements[0].position = 0.0
    tramp.color_ramp.elements[0].color = _gray(0.0)
    tramp.color_ramp.elements[1].position = 0.35
    tramp.color_ramp.elements[1].color = _gray(1.0)
    L.new(tw.outputs["Fac"], tramp.inputs["Fac"])

    tband = N.new("ShaderNodeValToRGB");       tband.location = (-700, -600)
    tband.color_ramp.interpolation = 'CONSTANT'
    te = tband.color_ramp.elements
    te[0].position = 0.0
    te[0].color = _gray(0.0)
    te[1].position = 0.52
    te[1].color = _gray(1.0)
    te2 = tband.color_ramp.elements.new(0.94)
    te2.color = _gray(0.0)
    L.new(sep.outputs["Y"], tband.inputs["Fac"])

    xband = N.new("ShaderNodeValToRGB");       xband.location = (-700, -820)
    xband.color_ramp.interpolation = 'CONSTANT'
    xe = xband.color_ramp.elements
    xe[0].position = 0.0
    xe[0].color = _gray(0.0)
    xe[1].position = 0.08
    xe[1].color = _gray(1.0)
    xe2 = xband.color_ramp.elements.new(0.72)
    xe2.color = _gray(0.0)
    L.new(sep.outputs["X"], xband.inputs["Fac"])

    tm1, tf1, ta1, tb1, to1 = _mix_rgb(nt, 'MULTIPLY')
    tm1.location = (-320, -450)
    tf1.default_value = 1.0
    L.new(tramp.outputs["Color"], ta1)
    L.new(tband.outputs["Color"], tb1)

    tm2, tf2, ta2, tb2, to2 = _mix_rgb(nt, 'MULTIPLY')
    tm2.location = (-120, -500)
    tf2.default_value = 1.0
    L.new(to1, ta2)
    L.new(xband.outputs["Color"], tb2)

    # marks = barcode + text
    add, afac, aa, ab, aout = _mix_rgb(nt, 'ADD')
    add.location = (60, 0)
    afac.default_value = 1.0
    L.new(bout, aa)
    L.new(to2, ab)

    # paper colour <- marks
    paper = N.new("ShaderNodeRGB");            paper.location = (60, 300)
    paper.outputs[0].default_value = _gray(0.9)
    ink = N.new("ShaderNodeRGB");              ink.location = (60, 180)
    ink.outputs[0].default_value = _gray(0.05)

    final, ffac, fa, fb, fout = _mix_rgb(nt)
    final.location = (380, 100)
    L.new(aout, ffac)
    L.new(paper.outputs[0], fa)
    L.new(ink.outputs[0], fb)
    L.new(fout, bsdf.inputs["Base Color"])

    _set(bsdf, "Roughness", 0.55)
    _set(bsdf, "Specular IOR Level", 0.4)

    return mat, dict(bmap=bmap, bw=bw, bramp=bramp, tmap=tmap, tw=tw,
                     tramp=tramp, paper=paper, ink=ink, bsdf=bsdf,
                     bband=bband, tband=tband)


LABEL_MATS = [build_label_material(i) for i in range(LABEL_MAT_POOL)]


# ============================================================
# DATA MATRIX DECAL  (one per package, on the top face)
# ============================================================
def build_dmc_material(i):
    mat = bpy.data.materials.new(f"D_Mat_{i:02d}")
    mat.use_fake_user = True
    mat.use_nodes = True
    nt = mat.node_tree
    N, L = nt.nodes, nt.links
    N.clear()

    out = N.new("ShaderNodeOutputMaterial");   out.location = (600, 0)
    bsdf = N.new("ShaderNodeBsdfPrincipled");  bsdf.location = (350, 0)
    L.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    tc = N.new("ShaderNodeTexCoord");          tc.location = (-600, 0)
    img = N.new("ShaderNodeTexImage");         img.location = (-380, 0)
    img.interpolation = 'Closest'     # keep module edges crisp
    img.extension = 'EXTEND'
    L.new(tc.outputs["UV"], img.inputs["Vector"])

    hsv = N.new("ShaderNodeHueSaturation");    hsv.location = (-120, 0)
    L.new(img.outputs["Color"], hsv.inputs["Color"])
    L.new(hsv.outputs["Color"], bsdf.inputs["Base Color"])

    _set(bsdf, "Roughness", 0.55)
    _set(bsdf, "Specular IOR Level", 0.35)
    return mat, dict(img=img, hsv=hsv, bsdf=bsdf)


DMC_MATS = [build_dmc_material(i) for i in range(DMC_MAT_POOL)]

# Short payloads -> 10x10 or 12x12 symbols (3 and 5 ASCII codewords). Digit
# pairs cost one codeword each, so 6 digits = 3 codewords = the smallest symbol.
_DMC_TEMPLATES = [
    lambda: f"{random.randint(0, 999999):06d}",                  # 10x10
    lambda: f"{random.randint(0, 99999999):08d}",                # 12x12
    lambda: f"{random.randint(0, 9999999999):010d}",             # 12x12
    lambda: f"P{random.randint(0, 9999):04d}",                   # 10x10
    lambda: f"{random.choice('ABCDEFGHJKLMNPRSTUVWXYZ')}{random.randint(0, 999999):06d}",
    lambda: f"{random.randint(0, 99):02d}-{random.randint(0, 9999):04d}",
]


def make_dmc_image(name):
    """Encode a random payload and return (image, payload, modules)."""
    payload = random.choice(_DMC_TEMPLATES)()
    size, grid = dmc_encode(payload)

    quiet = random.choice([1, 1, 2])            # quiet zone in modules
    n = size + 2 * quiet
    ink = random.uniform(0.02, 0.16)
    paper = random.uniform(0.72, 0.97)

    px = [0.0] * (n * n * 4)
    for r in range(n):
        # image rows run bottom-to-top in Blender; the grid is top-to-bottom
        gr = r - quiet
        for c in range(n):
            gc = c - quiet
            dark = (0 <= gr < size and 0 <= gc < size
                    and grid[size - 1 - gr][gc])
            v = ink if dark else paper
            o = (r * n + c) * 4
            px[o] = px[o + 1] = px[o + 2] = v
            px[o + 3] = 1.0

    img = bpy.data.images.new(name, width=n, height=n, alpha=False)
    img.pixels = px
    return img, payload, size


def _dmc_spot(parcel, side, reach, bands):
    """Pick a centre for the code.

    Usually near the middle of the top face, sometimes off towards a side, and
    never overlapping a tape strip - a code half-covered by tape is unreadable.
    On a parcel whose lid is not flat (gable, step, truncated pyramid, ...) the
    code is kept inside the locally flat patch, so it does not fold over a
    ridge. Returns (cx, cy) or None if the tape leaves no room.
    """
    hd = 0.5 * math.hypot(side, side)        # rotation-safe half extent

    zone = parcel.decal_zone
    zx = zy = 0.0
    rx = ry = None
    if zone is not None:
        zx, zy, zhx, zhy = zone
        rx, ry = zhx - hd, zhy - hd
        if rx < 0.0 or ry < 0.0:
            rx = ry = None                   # flat patch smaller than the code

    def clear_of_tape(cx, cy):
        for axis, c, w in bands:
            # a strip running along 'y' occupies a band in x, and vice versa
            p = cx if axis == 'y' else cy
            if abs(p - c) < w * 0.5 + hd:
                return False
        return True

    for attempt in range(60):
        if rx is not None and attempt < 45:
            cx = zx + random.uniform(-rx, rx)
            cy = zy + random.uniform(-ry, ry)
            if math.hypot(cx, cy) > reach:
                continue
        else:
            # centre-biased: ~70% of codes sit in the middle third
            if random.random() < 0.7:
                f = random.uniform(0.0, 0.35)
            else:
                f = random.uniform(0.35, 1.0)
            ang = random.uniform(0, math.tau)
            cx = math.cos(ang) * f * reach
            cy = math.sin(ang) * f * reach
        if clear_of_tape(cx, cy):
            return cx, cy
    return None


def build_dmc(parcel, entry, name, avoid=None):
    """Put one Data Matrix decal on the parcel's top face.

    Kept clear of the tape strips already on the parcel, biased towards the
    centre, and sometimes slightly bent (as if stuck over a crease or a bulge).
    `avoid` is an optional (x, y, radius) keep-out. Returns (object, info).
    """
    mat, r = entry
    fx = parcel.flat_half('x', 0.0)
    fy = parcel.flat_half('y', 0.0)
    lim = min(fx, fy) * 0.98
    # Small parcels (especially the scaled-down stacked ones) still get a code,
    # just a smaller one - otherwise ~20% of packages end up with none.
    side0 = min(random.uniform(DMC_SIZE_MIN, DMC_SIZE_MAX), lim * 1.35)
    if side0 < DMC_SIZE_FLOOR:
        return None, None

    # Shrink until the code fits between the tape strips.
    side = side0
    spot = None
    for _ in range(4):
        half_diag = 0.5 * math.hypot(side, side)
        if half_diag <= lim:
            reach = max(0.0, lim - half_diag)
            spot = _dmc_spot(parcel, side, reach, parcel.tape_bands)
            if spot is not None:
                break
        side *= 0.78
        if side < DMC_SIZE_FLOOR:
            break
    if spot is None:
        # On a small top the tape can cover every candidate spot. Rather than
        # leave the package without a code, go back to the full size and drop
        # the keep-out: the code is drawn above the tape anyway, so it stays
        # fully visible - it just sits on the tape.
        side = side0
        half_diag = 0.5 * math.hypot(side, side)
        reach = max(0.0, lim - half_diag)
        spot = _dmc_spot(parcel, side, reach, []) or (0.0, 0.0)
    cx, cy = spot
    half_diag = 0.5 * math.hypot(side, side)

    # keep off the shipping label too
    if avoid is not None:
        ax, ay, ar = avoid
        for _ in range(20):
            if math.hypot(cx - ax, cy - ay) > (ar + half_diag) * 0.95:
                break
            alt = _dmc_spot(parcel, side, max(0.0, lim - half_diag),
                            parcel.tape_bands)
            if alt is None:
                break
            cx, cy = alt

    img, payload, modules = make_dmc_image(name)
    r["img"].image = img
    _set(r["hsv"], "Value", random.uniform(0.85, 1.15))

    yaw = random.uniform(0, math.tau)
    ca, sa = math.cos(yaw), math.sin(yaw)

    # Optional bend: a shallow bulge/crease across the sticker. It is zero at
    # the outline, so the four recorded corners stay exact.
    bend_amp = 0.0
    bend_axis = 0
    if random.random() < DMC_BEND_PROB:
        bend_amp = random.uniform(0.0006, 0.0035) * random.choice([1.0, -1.0])
        bend_axis = random.choice([0, 1])

    def bend_z(fu, fv):
        """fu, fv in [-0.5, 0.5] across the sticker."""
        if bend_amp == 0.0:
            return 0.0
        t = fu if bend_axis == 0 else fv
        return bend_amp * max(0.0, 1.0 - (2.0 * t) ** 2)

    bmv = bmesh.new()
    uv_layer = bmv.loops.layers.uv.new("UVMap")
    n = 6                                   # enough segments to show the bend
    grid_v = []
    for i in range(n + 1):
        row = []
        for j in range(n + 1):
            u, v = i / n, j / n
            fu, fv = u - 0.5, v - 0.5
            lx, ly = fu * side, fv * side
            x = cx + lx * ca - ly * sa
            y = cy + lx * sa + ly * ca
            z = parcel.top_z(x, y) + LIFT_DMC + bend_z(fu, fv)
            row.append((bmv.verts.new((x, y, z)), (u, v)))
        grid_v.append(row)
    for i in range(n):
        for j in range(n):
            quad = [grid_v[i][j], grid_v[i + 1][j],
                    grid_v[i + 1][j + 1], grid_v[i][j + 1]]
            try:
                f = bmv.faces.new([q[0] for q in quad])
            except ValueError:
                continue
            for loop, (_, uv) in zip(f.loops, quad):
                loop[uv_layer].uv = uv
    if not bmv.faces:
        bmv.free()
        bpy.data.images.remove(img)
        return None, None
    bmesh.ops.recalc_face_normals(bmv, faces=bmv.faces)
    ob = _make_object("dmc", bmv, mat)
    ob.parent = parcel.obj
    ob.matrix_parent_inverse = mathutils.Matrix.Identity(4)

    # local-space corners, for the label file
    corners_local = []
    for fu, fv in ((-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)):
        x = cx + fu * side * ca - fv * side * sa
        y = cy + fu * side * sa + fv * side * ca
        corners_local.append((x, y, parcel.top_z(x, y) + LIFT_DMC))

    info = {"text": payload, "modules": modules, "size_m": side,
            "bent": bend_amp != 0.0, "corners_local": corners_local,
            "image": img}
    return ob, info


def randomize_label_material(entry):
    mat, r = entry
    r["bmap"].inputs["Scale"].default_value = (
        random.uniform(28.0, 70.0), 1.0, 1.0)
    r["bmap"].inputs["Location"].default_value = (
        random.uniform(0, 10), 0.0, 0.0)
    r["bw"].inputs["Distortion"].default_value = random.uniform(0.0, 1.5)
    r["bramp"].color_ramp.elements[1].position = random.uniform(0.35, 0.6)

    r["tmap"].inputs["Scale"].default_value = (
        1.0, random.uniform(9.0, 22.0), 1.0)
    r["tmap"].inputs["Location"].default_value = (
        0.0, random.uniform(0, 10), 0.0)
    r["tramp"].color_ramp.elements[1].position = random.uniform(0.25, 0.5)

    v = random.uniform(0.72, 0.97)
    r["paper"].outputs[0].default_value = _gray(v)
    r["ink"].outputs[0].default_value = _gray(random.uniform(0.02, 0.22))
    _set(r["bsdf"], "Roughness", random.uniform(0.35, 0.7))
    return mat


# ============================================================
# FLOOR MATERIALS
# ============================================================
# a) every floor-ish material already in scene.blend, made randomizable by
#    injecting TexCoord->Mapping into unlinked image-texture vectors and an
#    HSV node into the base-colour path;
# b) 4 fully procedural floors built here.
SCENE_FLOOR_NAMES = [
    "Concrete floor", "Concreate", "Simple concrete", "Ground",
    "Gray Granite", "Gray marble", "Light Gray Marble", "Gray Carpet",
    "Modern Gray Tile", "Floor Tile Dark Gray 600×600mm",
    "Dark Gray 600×600mm White Grout", "Metal Floor Panels Industrial",
    "Scratched grey plastic", "Grey metal", "Grey plastic",
    "Gray Wood Veneer",
]


# Per-image floor brightness window. The main loop narrows it to the dark end
# when the whole image is pale parcels, so a "white mailers on a white floor"
# frame can't come out as a contrast-free blur.
FLOOR_VALUE_RANGE = [0.45, 1.70]


def _floor_value():
    return random.uniform(*FLOOR_VALUE_RANGE)


def _floor_base_grey():
    """Base grey for the procedural floors, tracking the same bias."""
    return random.uniform(0.10, 0.62) * min(1.0, FLOOR_VALUE_RANGE[1] / 1.7)


def prepare_scene_floor(mat):
    """Inject Mapping + brightness control; return a randomizer closure."""
    if not mat.use_nodes:
        return None
    nt = mat.node_tree
    N, L = nt.nodes, nt.links

    tc = N.new("ShaderNodeTexCoord")
    mp = N.new("ShaderNodeMapping")
    L.new(tc.outputs["UV"], mp.inputs["Vector"])
    n_tex = 0
    for n in list(N):
        if n.type == 'TEX_IMAGE':
            n.extension = 'REPEAT'
            if not n.inputs["Vector"].is_linked:
                L.new(mp.outputs["Vector"], n.inputs["Vector"])
                n_tex += 1
    if n_tex == 0:
        N.remove(mp)
        N.remove(tc)
        mp = None

    bsdf = next((n for n in N if n.type == 'BSDF_PRINCIPLED'), None)
    hsv = None
    base_rough = None
    if bsdf is not None:
        bc = bsdf.inputs.get("Base Color")
        if bc is not None and bc.is_linked:
            src = bc.links[0].from_socket
            hsv = N.new("ShaderNodeHueSaturation")
            L.new(src, hsv.inputs["Color"])
            L.new(hsv.outputs["Color"], bc)
        rr = bsdf.inputs.get("Roughness")
        if rr is not None and not rr.is_linked:
            base_rough = float(rr.default_value)

    def rnd():
        if mp is not None:
            s = math.exp(random.uniform(math.log(FLOOR_TEX_SCALE_MIN),
                                        math.log(FLOOR_TEX_SCALE_MAX)))
            mp.inputs["Scale"].default_value = (
                s * random.uniform(0.85, 1.18), s * random.uniform(0.85, 1.18), s)
            mp.inputs["Location"].default_value = (
                random.uniform(-20, 20), random.uniform(-20, 20), 0.0)
            mp.inputs["Rotation"].default_value = (0, 0, random.uniform(0, math.tau))
        if hsv is not None:
            _set(hsv, "Value", _floor_value())
            _set(hsv, "Saturation", random.uniform(0.2, 1.3))
        if bsdf is not None and base_rough is not None:
            _set(bsdf, "Roughness",
                 min(1.0, max(0.03, base_rough + random.uniform(-0.25, 0.25))))

    return rnd


def build_procedural_floor(kind, idx):
    """kind: concrete | epoxy | rubber | painted"""
    mat = bpy.data.materials.new(f"F_Proc_{kind}_{idx}")
    mat.use_fake_user = True        # see build_parcel_material()
    mat.use_nodes = True
    nt = mat.node_tree
    N, L = nt.nodes, nt.links
    N.clear()

    out = N.new("ShaderNodeOutputMaterial");  out.location = (900, 0)
    bsdf = N.new("ShaderNodeBsdfPrincipled"); bsdf.location = (650, 0)
    L.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    tc = N.new("ShaderNodeTexCoord");         tc.location = (-1200, 0)
    mp = N.new("ShaderNodeMapping");          mp.location = (-1000, 0)
    L.new(tc.outputs["Object"], mp.inputs["Vector"])

    n1 = N.new("ShaderNodeTexNoise");         n1.location = (-800, 200)
    L.new(mp.outputs["Vector"], n1.inputs["Vector"])
    r1 = N.new("ShaderNodeValToRGB");         r1.location = (-600, 200)
    L.new(n1.outputs["Fac"], r1.inputs["Fac"])

    vor = N.new("ShaderNodeTexVoronoi");      vor.location = (-800, -150)
    vor.feature = 'DISTANCE_TO_EDGE'
    L.new(mp.outputs["Vector"], vor.inputs["Vector"])
    r2 = N.new("ShaderNodeValToRGB");         r2.location = (-600, -150)
    L.new(vor.outputs["Distance"], r2.inputs["Fac"])

    m1, f1, a1, b1, o1 = _mix_rgb(nt, 'MULTIPLY')
    m1.location = (-320, 60)
    f1.default_value = 0.6
    L.new(r1.outputs["Color"], a1)
    L.new(r2.outputs["Color"], b1)

    # tiled / brick-laid floor: own mapping so the tile pitch is independent
    # of the surface-detail scale
    bmp = N.new("ShaderNodeMapping");         bmp.location = (-1000, 380)
    L.new(tc.outputs["Object"], bmp.inputs["Vector"])
    brk = N.new("ShaderNodeTexBrick");        brk.location = (-800, 380)
    L.new(bmp.outputs["Vector"], brk.inputs["Vector"])
    mb, fb, ab, bb, ob_ = _mix_rgb(nt, 'MULTIPLY')
    mb.location = (-160, 180)
    fb.default_value = 0.0
    L.new(o1, ab)
    L.new(brk.outputs["Color"], bb)
    o1 = ob_                       # everything downstream sees the tiles too

    # painted line marking - own mapping, otherwise it inherits the (much
    # finer) surface-detail scale and turns the floor into stripey fabric.
    wmp = N.new("ShaderNodeMapping");         wmp.location = (-1000, -450)
    L.new(tc.outputs["Object"], wmp.inputs["Vector"])
    wave = N.new("ShaderNodeTexWave");        wave.location = (-800, -450)
    wave.wave_type = 'BANDS'
    L.new(wmp.outputs["Vector"], wave.inputs["Vector"])
    r3 = N.new("ShaderNodeValToRGB");         r3.location = (-600, -450)
    r3.color_ramp.interpolation = 'CONSTANT'
    r3.color_ramp.elements[0].position = 0.0
    r3.color_ramp.elements[0].color = _gray(0.0)
    r3.color_ramp.elements[1].position = 0.9
    r3.color_ramp.elements[1].color = _gray(1.0)
    L.new(wave.outputs["Fac"], r3.inputs["Fac"])

    paint = N.new("ShaderNodeRGB");           paint.location = (-320, -420)
    paint.outputs[0].default_value = _gray(0.85)
    m2, f2, a2, b2, o2 = _mix_rgb(nt)
    m2.location = (60, 0)
    L.new(o1, a2)
    L.new(paint.outputs[0], b2)
    if kind == "painted":
        L.new(r3.outputs["Color"], f2)
    else:
        f2.default_value = 0.0
    L.new(o2, bsdf.inputs["Base Color"])

    bump = N.new("ShaderNodeBump");           bump.location = (300, -300)
    L.new(o1, bump.inputs["Height"])
    L.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    refs = dict(mp=mp, n1=n1, r1=r1, vor=vor, r2=r2, f1=f1, wave=wave, r3=r3,
                f2=f2, paint=paint, bump=bump, bsdf=bsdf, kind=kind)

    def rnd():
        s = random.uniform(3.0, 30.0)
        mp.inputs["Scale"].default_value = (s, s, s)
        mp.inputs["Location"].default_value = (
            random.uniform(-30, 30), random.uniform(-30, 30), 0.0)
        mp.inputs["Rotation"].default_value = (0, 0, random.uniform(0, math.tau))

        n1.inputs["Scale"].default_value = random.uniform(1.0, 12.0)
        n1.inputs["Detail"].default_value = random.uniform(2.0, 12.0)
        base = _floor_base_grey()
        spread = random.uniform(0.05, 0.35)
        r1.color_ramp.elements[0].color = _gray(max(0.0, base - spread))
        r1.color_ramp.elements[1].color = _gray(min(1.0, base + spread))
        r1.color_ramp.elements[0].position = random.uniform(0.1, 0.4)
        r1.color_ramp.elements[1].position = random.uniform(0.6, 0.95)

        vor.inputs["Scale"].default_value = random.uniform(2.0, 25.0)
        r2.color_ramp.elements[0].position = random.uniform(0.0, 0.06)
        r2.color_ramp.elements[1].position = random.uniform(0.08, 0.3)
        r2.color_ramp.elements[0].color = _gray(random.uniform(0.2, 0.8))
        f1.default_value = random.uniform(0.15, 0.85)

        if kind in ("tiles", "brick"):
            k = math.exp(random.uniform(math.log(1.5), math.log(9.0)))
            bmp.inputs["Scale"].default_value = (k, k, k)
            bmp.inputs["Location"].default_value = (
                random.uniform(-8, 8), random.uniform(-8, 8), 0.0)
            bmp.inputs["Rotation"].default_value = (
                0, 0, random.uniform(0, math.tau))
            tile = _floor_base_grey() + random.uniform(0.05, 0.35)
            brk.inputs["Color1"].default_value = _gray(min(1.0, tile))
            brk.inputs["Color2"].default_value = _gray(
                min(1.0, max(0.02, tile * random.uniform(0.72, 1.25))))
            brk.inputs["Mortar"].default_value = _gray(
                max(0.01, tile * random.uniform(0.25, 0.7)))
            brk.inputs["Scale"].default_value = 1.0
            brk.inputs["Mortar Size"].default_value = random.uniform(0.004, 0.03)
            brk.inputs["Bias"].default_value = random.uniform(-0.4, 0.4)
            if kind == "brick":
                brk.inputs["Brick Width"].default_value = random.uniform(0.4, 0.9)
                brk.inputs["Row Height"].default_value = random.uniform(0.15, 0.35)
                brk.offset = 0.5
            else:
                sq = random.uniform(0.45, 1.0)
                brk.inputs["Brick Width"].default_value = sq
                brk.inputs["Row Height"].default_value = sq
                brk.offset = random.choice([0.0, 0.0, 0.5])
            fb.default_value = random.uniform(0.7, 1.0)
        else:
            fb.default_value = 0.0

        if kind == "epoxy":
            rough = random.uniform(0.08, 0.3)
            bstr = random.uniform(0.02, 0.12)
        elif kind == "rubber":
            rough = random.uniform(0.6, 0.95)
            bstr = random.uniform(0.2, 0.6)
        elif kind == "tiles":
            rough = random.uniform(0.12, 0.55)
            bstr = random.uniform(0.05, 0.3)
        elif kind == "brick":
            rough = random.uniform(0.5, 0.9)
            bstr = random.uniform(0.15, 0.5)
        elif kind == "painted":
            rough = random.uniform(0.3, 0.7)
            bstr = random.uniform(0.05, 0.3)
            # one painted lane line every ~1-5 m, 5-20 cm wide
            wmp.inputs["Scale"].default_value = tuple(
                [random.uniform(0.35, 1.6)] * 3)
            wmp.inputs["Location"].default_value = (
                random.uniform(-5, 5), random.uniform(-5, 5), 0.0)
            wmp.inputs["Rotation"].default_value = (0, 0, random.uniform(0, math.tau))
            wave.inputs["Scale"].default_value = 1.0
            wave.inputs["Distortion"].default_value = random.uniform(0.0, 0.4)
            r3.color_ramp.elements[1].position = random.uniform(0.90, 0.975)
            paint.outputs[0].default_value = _gray(random.uniform(0.55, 0.95))
        else:  # concrete
            rough = random.uniform(0.45, 0.9)
            bstr = random.uniform(0.1, 0.4)
        _set(bsdf, "Roughness", rough)
        _set(bsdf, "Metallic", 0.0)
        _set(bump, "Strength", bstr)
        _set(bump, "Distance", random.uniform(0.002, 0.02))

    return mat, rnd, refs


def build_pbr_floor_material(i):
    """Floor driven by a downloaded PBR set. Unlike the parcels, the floor
    plane has a real UV layout, so a tangent-space normal map is valid here."""
    mat = bpy.data.materials.new(f"F_PBR_{i}")
    mat.use_fake_user = True
    mat.use_nodes = True
    nt = mat.node_tree
    N, L = nt.nodes, nt.links
    N.clear()

    out = N.new("ShaderNodeOutputMaterial");  out.location = (900, 0)
    bsdf = N.new("ShaderNodeBsdfPrincipled"); bsdf.location = (650, 0)
    L.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    tc = N.new("ShaderNodeTexCoord");         tc.location = (-1000, 0)
    mp = N.new("ShaderNodeMapping");          mp.location = (-800, 0)
    L.new(tc.outputs["UV"], mp.inputs["Vector"])

    col = N.new("ShaderNodeTexImage");        col.location = (-560, 300)
    col.extension = 'REPEAT'
    L.new(mp.outputs["Vector"], col.inputs["Vector"])
    hsv = N.new("ShaderNodeHueSaturation");   hsv.location = (-260, 300)
    L.new(col.outputs["Color"], hsv.inputs["Color"])
    L.new(hsv.outputs["Color"], bsdf.inputs["Base Color"])

    rgh = N.new("ShaderNodeTexImage");        rgh.location = (-560, 20)
    rgh.extension = 'REPEAT'
    L.new(mp.outputs["Vector"], rgh.inputs["Vector"])
    rmap = N.new("ShaderNodeMapRange");       rmap.location = (-260, 20)
    L.new(rgh.outputs["Color"], rmap.inputs["Value"])
    L.new(rmap.outputs["Result"], bsdf.inputs["Roughness"])

    nrm = N.new("ShaderNodeTexImage");        nrm.location = (-560, -280)
    nrm.extension = 'REPEAT'
    L.new(mp.outputs["Vector"], nrm.inputs["Vector"])
    nmap = N.new("ShaderNodeNormalMap");      nmap.location = (-260, -280)
    L.new(nrm.outputs["Color"], nmap.inputs["Color"])
    L.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])

    def rnd():
        s = random.choice(PBR_FLOOR)
        col.image = s["color"]
        rgh.image = s["rough"]
        nrm.image = s["normal"]
        # one texture tile ~= 0.25 .. 2.5 m of floor
        k = math.exp(random.uniform(math.log(1.2), math.log(11.0)))
        mp.inputs["Scale"].default_value = (
            k * random.uniform(0.9, 1.1), k * random.uniform(0.9, 1.1), k)
        mp.inputs["Location"].default_value = (
            random.uniform(-20, 20), random.uniform(-20, 20), 0.0)
        mp.inputs["Rotation"].default_value = (0, 0, random.uniform(0, math.tau))
        _set(hsv, "Value", _floor_value())
        _set(hsv, "Saturation", random.uniform(0.15, 1.2))
        _set(rmap, "From Min", random.uniform(0.0, 0.2))
        _set(rmap, "From Max", random.uniform(0.8, 1.0))
        _set(rmap, "To Min", random.uniform(0.1, 0.4))
        _set(rmap, "To Max", random.uniform(0.6, 1.0))
        _set(nmap, "Strength",
             0.0 if s["normal"] is None else random.uniform(0.4, 1.5))
        _set(bsdf, "Metallic", 0.0)

    return mat, rnd


FLOORS = []          # list of (material, randomizer)
for nm in SCENE_FLOOR_NAMES:
    m = bpy.data.materials.get(nm)
    if m is None:
        continue
    fn = prepare_scene_floor(m)
    if fn is not None:
        FLOORS.append((m, fn))
for nm in SCENE_FLOOR_NAMES:
    m = bpy.data.materials.get(nm)
    if m is not None:
        m.use_fake_user = True      # survives the periodic orphans_purge()
for _i, _k in enumerate(("concrete", "epoxy", "rubber", "painted",
                         "tiles", "brick")):
    _m, _fn, _r = build_procedural_floor(_k, _i)
    FLOORS.append((_m, _fn))
# Downloaded PBR floors: a handful of slots, each re-pointed at a random set
# per image, so all of PBR_FLOOR is reachable. Weighted by repeating the
# slots, since these are the most realistic floors we have.
PBR_FLOOR_SLOTS = 12 if PBR_FLOOR else 0
for _i in range(PBR_FLOOR_SLOTS):
    FLOORS.append(build_pbr_floor_material(_i))
print(f"floors: {len(FLOORS)} materials "
      f"({len(PBR_FLOOR)} PBR sets behind the {PBR_FLOOR_SLOTS} PBR slots)")

floor.data.materials.clear()
floor.data.materials.append(FLOORS[0][0])

# ============================================================
# LIGHTS
# ============================================================
for lt in [o for o in bpy.data.objects if o.type == 'LIGHT']:
    bpy.data.objects.remove(lt, do_unlink=True)

LIGHTS = []
for name, ltype in (("Key_Area", 'AREA'), ("Fill_Area", 'AREA'),
                    ("Point", 'POINT'), ("Point2", 'POINT'),
                    ("Spot", 'SPOT'), ("Sun", 'SUN')):
    ld = bpy.data.lights.new(name, ltype)
    ob = bpy.data.objects.new(name, ld)
    WORK_COLL.objects.link(ob)
    LIGHTS.append(ob)

SOFT_LIGHTS = [o for o in LIGHTS if o.data.type == 'AREA']
HARD_LIGHTS = [o for o in LIGHTS if o.data.type in ('POINT', 'SPOT', 'SUN')]

WORLD_BG = None
if scene.world is None:
    scene.world = bpy.data.worlds.new("World")
scene.world.use_nodes = True
WORLD_BG = next((n for n in scene.world.node_tree.nodes
                 if n.type == 'BACKGROUND'), None)


def aim(ob, target=Vector((0, 0, 0.1))):
    d = target - ob.location
    ob.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()


# ------------------------------------------------------------------
# GOBO: a patterned plane above the scene, invisible to the camera but
# blocking light, so the parcels are lit in patches (window bars, foliage,
# racking overhead). Cycles honours visible_camera; EEVEE does not, so the
# gobo is only used under Cycles.
# ------------------------------------------------------------------
def build_gobo():
    mat = bpy.data.materials.new("Gobo")
    mat.use_fake_user = True
    mat.use_nodes = True
    nt = mat.node_tree
    N, L = nt.nodes, nt.links
    N.clear()

    out = N.new("ShaderNodeOutputMaterial");    out.location = (600, 0)
    trans = N.new("ShaderNodeBsdfTransparent"); trans.location = (300, 120)
    diff = N.new("ShaderNodeBsdfDiffuse");      diff.location = (300, -80)
    diff.inputs["Color"].default_value = _gray(0.0)
    mix = N.new("ShaderNodeMixShader");         mix.location = (450, 0)
    L.new(trans.outputs["BSDF"], mix.inputs[1])
    L.new(diff.outputs["BSDF"], mix.inputs[2])
    L.new(mix.outputs["Shader"], out.inputs["Surface"])

    tc = N.new("ShaderNodeTexCoord");           tc.location = (-800, 0)
    mp = N.new("ShaderNodeMapping");            mp.location = (-620, 0)
    L.new(tc.outputs["Object"], mp.inputs["Vector"])

    noi = N.new("ShaderNodeTexNoise");          noi.location = (-420, 180)
    L.new(mp.outputs["Vector"], noi.inputs["Vector"])
    wav = N.new("ShaderNodeTexWave");           wav.location = (-420, 0)
    wav.wave_type = 'BANDS'
    L.new(mp.outputs["Vector"], wav.inputs["Vector"])
    vor = N.new("ShaderNodeTexVoronoi");        vor.location = (-420, -180)
    L.new(mp.outputs["Vector"], vor.inputs["Vector"])

    s1, f1, a1, b1, o1 = _mix_rgb(nt);          s1.location = (-200, 90)
    L.new(noi.outputs["Fac"], a1)
    L.new(wav.outputs["Fac"], b1)
    s2, f2, a2, b2, o2 = _mix_rgb(nt);          s2.location = (-40, 0)
    L.new(o1, a2)
    L.new(_find(vor.outputs, "Distance", 'VALUE') or vor.outputs[0], b2)

    ramp = N.new("ShaderNodeValToRGB");         ramp.location = (120, 0)
    ramp.color_ramp.interpolation = 'CONSTANT'
    L.new(o2, ramp.inputs["Fac"])
    L.new(ramp.outputs["Color"], mix.inputs[0])

    me = bpy.data.meshes.new("gobo_mesh")
    s = 4.0
    me.from_pydata([(-s, -s, 0.0), (s, -s, 0.0), (s, s, 0.0), (-s, s, 0.0)],
                   [], [(0, 1, 2, 3)])
    me.update()
    me.materials.append(mat)
    ob = bpy.data.objects.new("Gobo", me)
    WORK_COLL.objects.link(ob)
    ob.hide_render = True
    for attr in ("visible_camera", "visible_diffuse", "visible_glossy"):
        try:
            setattr(ob, attr, False)
        except Exception:
            pass
    return ob, dict(mp=mp, noi=noi, wav=wav, vor=vor,
                    facs=(f1, f2), ramp=ramp)


GOBO_OB, GOBO_R = build_gobo()


def randomize_gobo(center, height):
    """Point the blocker at a random pattern and put it over the scene."""
    GOBO_OB.location = (center.x + random.uniform(-0.3, 0.3),
                        center.y + random.uniform(-0.3, 0.3),
                        FLOOR_Z + height)
    GOBO_OB.rotation_euler = (random.uniform(-0.12, 0.12),
                              random.uniform(-0.12, 0.12),
                              random.uniform(0, math.tau))
    src = random.choices(["noise", "bands", "cells"], [34, 40, 26])[0]
    for fac, v in zip(GOBO_R["facs"],
                      {"noise": (0, 0), "bands": (1, 0),
                       "cells": (0, 1)}[src]):
        fac.default_value = float(v)

    # Feature size on the plane is ~1/scale metres; the camera only sees about
    # 0.9 m of floor, so anything below ~1.5 is one soft edge across the whole
    # frame rather than a pattern.
    sc = random.uniform(1.5, 10.0)
    GOBO_R["mp"].inputs["Scale"].default_value = (sc, sc, sc)
    GOBO_R["mp"].inputs["Rotation"].default_value = (
        0.0, 0.0, random.uniform(0, math.tau))
    GOBO_R["noi"].inputs["Scale"].default_value = random.uniform(0.8, 5.0)
    GOBO_R["noi"].inputs["Detail"].default_value = random.uniform(0.0, 3.0)
    GOBO_R["wav"].inputs["Scale"].default_value = random.uniform(0.3, 3.0)
    GOBO_R["wav"].inputs["Distortion"].default_value = random.uniform(0.0, 3.0)
    GOBO_R["wav"].bands_direction = random.choice(['X', 'Y', 'DIAGONAL'])
    GOBO_R["vor"].inputs["Scale"].default_value = random.uniform(1.0, 8.0)

    e0, e1 = GOBO_R["ramp"].color_ramp.elements
    # how much of the plane is opaque: a narrow slot vs a light haze
    cut = random.uniform(0.28, 0.72)
    e0.position = 0.0
    e0.color = _gray(0.0)               # 0 -> transparent
    e1.position = cut
    e1.color = _gray(1.0)               # 1 -> opaque blocker
    GOBO_R["ramp"].color_ramp.interpolation = random.choices(
        ['CONSTANT', 'LINEAR'], [72, 28])[0]


# Per-scenario parameter envelope. `energy` is a multiplier on the base ranges,
# `soft` the source size in metres (area size / point radius), `elev` degrees.
LIGHT_SETUPS = {
    "studio_soft":    dict(n=(2, 4), elev=(40, 88), soft=(0.55, 2.2),
                           energy=(0.7, 1.2), world=(0.15, 0.60),
                           expo=(-0.5, 0.4), gel=0.10, gobo=0.05, soft_bias=0.8),
    "hard_single":    dict(n=(1, 1), elev=(18, 58), soft=(0.02, 0.18),
                           energy=(0.9, 1.6), world=(0.0, 0.12),
                           expo=(-0.7, 0.3), gel=0.15, gobo=0.10, soft_bias=0.1),
    "overhead_flash": dict(n=(1, 2), elev=(70, 89), soft=(0.03, 0.35),
                           energy=(0.9, 1.5), world=(0.02, 0.25),
                           expo=(-0.6, 0.2), gel=0.05, gobo=0.05, soft_bias=0.3),
    "backlit_low":    dict(n=(1, 2), elev=(8, 26), soft=(0.05, 0.9),
                           energy=(1.0, 1.7), world=(0.05, 0.35),
                           expo=(-0.8, 0.2), gel=0.20, gobo=0.05, soft_bias=0.4),
    "dappled":        dict(n=(1, 2), elev=(35, 85), soft=(0.02, 0.5),
                           energy=(1.0, 1.7), world=(0.05, 0.35),
                           expo=(-0.5, 0.3), gel=0.15, gobo=1.0, soft_bias=0.2),
    "dim_ambient":    dict(n=(1, 2), elev=(15, 88), soft=(0.3, 1.6),
                           energy=(0.20, 0.5), world=(0.45, 0.90),
                           expo=(-0.2, 0.6), gel=0.10, gobo=0.05, soft_bias=0.6),
    "colour_mix":     dict(n=(2, 3), elev=(15, 85), soft=(0.05, 1.4),
                           energy=(0.7, 1.3), world=(0.02, 0.40),
                           expo=(-0.6, 0.3), gel=0.95, gobo=0.10, soft_bias=0.5),
}


def _gel_colour(strong):
    """A light colour: warm/cool white, or a saturated gel."""
    warm = Vector((1.0, 0.82, 0.62))
    cool = Vector((0.74, 0.86, 1.0))
    if not strong:
        t = random.random()
        return tuple(warm * (1 - t) + cool * t)
    hue = random.random()
    sat = random.uniform(0.35, 0.85)
    c = mathutils.Color()
    c.hsv = (hue, sat, 1.0)
    return (c.r, c.g, c.b)


def randomize_lights(center):
    """Light the frame from one of the named scenarios. Returns its name."""
    names = list(LIGHT_SCENARIOS)
    scen = random.choices(names, [LIGHT_SCENARIOS[n] for n in names])[0]
    cfg = LIGHT_SETUPS[scen]

    use_gobo = (ENGINE == "CYCLES"
                and random.random() < max(cfg["gobo"], GOBO_PROB))
    gobo_h = random.uniform(*GOBO_HEIGHT) if use_gobo else 0.0
    GOBO_OB.hide_render = not use_gobo
    if use_gobo:
        randomize_gobo(center, gobo_h)

    n_on = random.randint(*cfg["n"])
    # pick the sources: soft scenarios favour the area lights, hard ones the
    # point/spot/sun
    pool = list(LIGHTS)
    ws = [cfg["soft_bias"] if o in SOFT_LIGHTS else 1.0 - cfg["soft_bias"]
          for o in pool]
    on = []
    for _ in range(min(n_on, len(pool))):
        i = random.choices(range(len(pool)), [max(w, 0.03) for w in ws])[0]
        on.append(pool.pop(i))
        ws.pop(i)
    n_on = len(on)
    for ob in LIGHTS:
        ob.hide_render = ob not in on
    share = 1.0 / max(n_on, 1)

    el_lo, el_hi = (math.radians(cfg["elev"][0]), math.radians(cfg["elev"][1]))
    e_lo, e_hi = cfg["energy"]

    for ob in on:
        az = random.uniform(0, math.tau)
        el = random.uniform(el_lo, el_hi)
        dist = random.uniform(LIGHT_DIST_MIN, LIGHT_DIST_MAX)
        if use_gobo and ob.data.type != 'SUN':
            # a source under the blocker would just be switched off by it
            need = gobo_h + 0.35
            dist = max(dist, need / 0.92)
            el = max(el, math.asin(min(0.95, need / dist)))
        ob.location = (center.x + dist * math.cos(el) * math.cos(az),
                       center.y + dist * math.cos(el) * math.sin(az),
                       FLOOR_Z + dist * math.sin(el))
        aim(ob, center)

        strong_gel = random.random() < max(cfg["gel"], COLOUR_GEL_PROB)
        ob.data.color = _gel_colour(strong_gel)

        falloff = (dist / 1.5) ** 2
        boost = random.uniform(e_lo, e_hi) * falloff * share
        size = random.uniform(*cfg["soft"])
        if ob.data.type == 'AREA':
            ob.data.size = size
            ob.data.energy = random.uniform(
                AREA_ENERGY_MIN, AREA_ENERGY_MAX) * boost
        elif ob.data.type == 'POINT':
            ob.data.shadow_soft_size = min(size, 0.5)
            ob.data.energy = random.uniform(
                POINT_ENERGY_MIN, POINT_ENERGY_MAX) * boost
        elif ob.data.type == 'SPOT':
            ob.data.shadow_soft_size = min(size, 0.4)
            ob.data.spot_size = random.uniform(math.radians(28),
                                               math.radians(115))
            ob.data.spot_blend = random.uniform(0.05, 0.9)
            ob.data.energy = random.uniform(
                SPOT_ENERGY_MIN, SPOT_ENERGY_MAX) * boost
        else:  # SUN: distance is irrelevant, only the direction counts
            ob.data.angle = random.uniform(math.radians(0.3),
                                           math.radians(3.0 + 22.0 * size))
            ob.data.energy = random.uniform(
                SUN_ENERGY_MIN, SUN_ENERGY_MAX) * random.uniform(e_lo, e_hi) * share

    if WORLD_BG is not None:
        w_lo, w_hi = cfg["world"]
        _set(WORLD_BG, "Strength", random.uniform(
            max(WORLD_STRENGTH_MIN, w_lo), min(WORLD_STRENGTH_MAX, w_hi)))
        g = random.uniform(0.25, 1.0)
        if random.random() < 0.22:          # a tinted room / sky ambient
            c = mathutils.Color()
            c.hsv = (random.random(), random.uniform(0.1, 0.45), 1.0)
            WORLD_BG.inputs["Color"].default_value = (c.r * g, c.g * g,
                                                      c.b * g, 1.0)
        else:
            tint = random.uniform(-0.08, 0.08)
            WORLD_BG.inputs["Color"].default_value = (
                g * (1 + tint), g, g * (1 - tint), 1.0)

    scene.view_settings.exposure = random.uniform(
        max(EXPOSURE_MIN, cfg["expo"][0]), min(EXPOSURE_MAX, cfg["expo"][1]))
    transform = random.choices(["AgX", "Standard", "Filmic"], [50, 32, 18])[0]
    # The valid `look` values depend on the active view transform, so reset it
    # before switching - a stale look raises and would stick around.
    try:
        scene.view_settings.look = 'None'
        scene.view_settings.view_transform = transform
    except Exception:
        transform = scene.view_settings.view_transform
    if transform == "AgX" and random.random() < 0.40:
        try:
            scene.view_settings.look = random.choice(
                ["AgX - Medium Contrast", "AgX - High Contrast",
                 "AgX - Low Contrast", "AgX - Punchy"])
        except Exception:
            pass
    return scen, use_gobo


# ============================================================
# GEOMETRY
# ============================================================
def _noise3(x, y, z, seed):
    return mathutils.noise.noise(Vector((x, y, z + seed)))


class Parcel:
    """Generated parcel geometry + the analytic description of its top face,
    which the tape/label geometry needs in order to lie exactly on it."""

    def __init__(self, obj, ptype, w, d, h, bevel, round_r, top_fn,
                 top_w=None, top_d=None, decal_zone=None, top_flat=True,
                 shape=None):
        self.obj = obj
        self.type = ptype
        self.w, self.d, self.h = w, d, h
        # Extents of the TOP face. They differ from w/d on tapered parcels, and
        # everything that sits on the lid (tape, label, code) works off them.
        self.top_w = w if top_w is None else top_w
        self.top_d = d if top_d is None else top_d
        self.bevel = bevel
        self.radius = round_r          # None for cuboids
        self.top_off = top_fn          # f(x, y) -> extra z on top of h/2
        # Rectangle (cx, cy, hx, hy) on the lid that is locally flat enough to
        # take a sticker - one roof facet, the raised step, the plateau of a
        # truncated pyramid. None means "the whole flat area is fine".
        self.decal_zone = decal_zone
        self.top_flat = top_flat       # may other parcels be stacked on it?
        self.shape = shape or {}       # extra description for the label file
        self.dmc = None                # Data Matrix info, filled by decorate()
        # Tape strips already on the top face, as (running axis, offset on the
        # perpendicular axis, width). The Data Matrix keeps clear of these.
        self.tape_bands = []

    # ---- top-surface helpers -------------------------------------------
    def top_z(self, x, y):
        return self.h * 0.5 + self.top_off(x, y)

    def flat_half(self, axis, other):
        """Half-extent of the FLAT part of the top along `axis` ('x'|'y'),
        at the given coordinate on the other axis."""
        if self.radius is None:
            half = (self.top_w if axis == 'x' else self.top_d) * 0.5
            return max(half - self.bevel, 1e-4)
        rin = max(self.radius - self.bevel, 1e-4)
        return math.sqrt(max(rin * rin - other * other, 1e-8))

    def edge_half(self, axis, other):
        """Half-extent of the top-face outline along `axis`."""
        if self.radius is None:
            return (self.top_w if axis == 'x' else self.top_d) * 0.5
        return math.sqrt(max(self.radius ** 2 - other ** 2, 1e-8))


def _sharp_edges(bm, min_angle=math.radians(20.0)):
    out = []
    for e in bm.edges:
        try:
            if e.calc_face_angle(0.0) > min_angle:
                out.append(e)
        except Exception:
            pass
    return out


def _bevel_group(bm, edges, width, segments, profile):
    """Bevel one set of edges. Earlier passes can invalidate later edges, so
    every group is re-filtered through is_valid before use."""
    edges = [e for e in edges if e.is_valid]
    if not edges or width <= 1e-5:
        return
    kw = dict(offset=width, offset_type='OFFSET', segments=max(1, segments),
              profile=profile, clamp_overlap=True)
    try:
        bmesh.ops.bevel(bm, geom=edges, affect='EDGES', **kw)
    except TypeError:
        bmesh.ops.bevel(bm, geom=edges, **kw)


def _edge_groups(bm):
    """sharp edges split into (vertical, top horizontal, bottom horizontal)."""
    vert, top, bot = [], [], []
    for e in _sharp_edges(bm):
        a, b = e.verts[0].co, e.verts[1].co
        if abs(a.z - b.z) > math.hypot(a.x - b.x, a.y - b.y):
            vert.append(e)
        elif 0.5 * (a.z + b.z) > 0.0:
            top.append(e)
        else:
            bot.append(e)
    return vert, top, bot


def _segments():
    """1 = hard chamfer, 2 = crude round, 6 = soft roll."""
    return random.choices([1, 2, 3, 4, 5, 6], [22, 18, 20, 16, 12, 12])[0]


def _round_corners(bm, base_w, small, style=None):
    """Apply the parcel's corner rounding and return the TOP edge radius.

    Corner rounding is deliberately different from parcel to parcel: which
    edges get rounded, how wide, how many segments (chamfer vs soft roll) and
    the bevel profile (<0.5 hollow, 0.5 circular, >0.5 bulged). The returned
    top radius is what the tape geometry uses for its wrap-over arc, so it has
    to be the width actually applied to the top edges.
    """
    style = style or random.choices(
        ["uniform", "soft_top", "soft_verticals", "chamfer", "mixed", "worn"],
        [26, 18, 18, 12, 16, 10])[0]

    cap = 0.45 * small

    def clamp(x):
        return max(0.0005, min(x, cap))

    vert, top, bot = _edge_groups(bm)

    if style == "uniform":
        w = clamp(base_w)
        s, p = _segments(), random.uniform(0.35, 0.65)
        _bevel_group(bm, top, w, s, p)
        _bevel_group(bm, vert, w, s, p)
        _bevel_group(bm, bot, w, s, p)
        return w

    if style == "soft_top":
        wt = clamp(base_w * random.uniform(1.4, 2.2))
        wv = clamp(base_w * random.uniform(0.5, 1.0))
        wb = clamp(base_w * random.uniform(0.3, 0.8))
        _bevel_group(bm, top, wt, max(2, _segments()), random.uniform(0.4, 0.7))
        _bevel_group(bm, vert, wv, _segments(), random.uniform(0.35, 0.65))
        _bevel_group(bm, bot, wb, _segments(), random.uniform(0.3, 0.6))
        return wt

    if style == "soft_verticals":
        wv = clamp(base_w * random.uniform(1.5, 2.4))
        wt = clamp(base_w * random.uniform(0.35, 0.8))
        _bevel_group(bm, top, wt, _segments(), random.uniform(0.35, 0.65))
        _bevel_group(bm, vert, wv, max(2, _segments()), random.uniform(0.4, 0.7))
        _bevel_group(bm, bot, clamp(base_w * 0.5), 1, 0.5)
        return wt

    if style == "chamfer":
        w = clamp(base_w * random.uniform(1.1, 1.8))
        p = random.uniform(0.2, 0.5)
        _bevel_group(bm, top, w, 1, p)
        _bevel_group(bm, vert, w * random.uniform(0.6, 1.2), 1, p)
        _bevel_group(bm, bot, w * random.uniform(0.4, 1.0), 1, p)
        return w

    if style == "mixed":
        wt = clamp(base_w * random.uniform(0.5, 2.0))
        _bevel_group(bm, top, wt, _segments(), random.uniform(0.2, 0.8))
        _bevel_group(bm, vert, clamp(base_w * random.uniform(0.5, 2.0)),
                     _segments(), random.uniform(0.2, 0.8))
        _bevel_group(bm, bot, clamp(base_w * random.uniform(0.3, 1.5)),
                     _segments(), random.uniform(0.2, 0.8))
        return wt

    # "worn": one pass everywhere, then a wider second pass on a random
    # subset, so some corners are visibly softer than others on the same box
    w = clamp(base_w * random.uniform(0.5, 0.9))
    s, p = _segments(), random.uniform(0.35, 0.65)
    _bevel_group(bm, top, w, s, p)
    _bevel_group(bm, vert, w, s, p)
    _bevel_group(bm, bot, w, s, p)
    again = [e for e in _sharp_edges(bm, math.radians(35.0))
             if random.random() < 0.45]
    _bevel_group(bm, again, clamp(base_w * random.uniform(0.6, 1.3)),
                 max(2, _segments()), random.uniform(0.4, 0.7))
    return w


def _rip_edge(bm, w, d, h, amount):
    """Make one top edge ragged, as if the box has been torn open there.

    Only vertices within a thin band along the chosen edge move, and they move
    mostly sideways/downwards, so the middle of the top face - where the tape
    and the codes sit - is untouched.
    """
    axis = random.choice(['x', 'y'])
    sgn = random.choice([-1.0, 1.0])
    edge_pos = sgn * (w if axis == 'x' else d) * 0.5
    band = random.uniform(0.10, 0.22) * (w if axis == 'x' else d)
    top = h * 0.5
    depth = random.uniform(0.25, 0.6) * h
    freq = random.uniform(30.0, 90.0)

    for v in bm.verts:
        p = v.co.x if axis == 'x' else v.co.y
        if abs(p - edge_pos) > band or v.co.z < top - depth:
            continue
        near = 1.0 - abs(p - edge_pos) / band          # 1 at the edge
        high = max(0.0, (v.co.z - (top - depth)) / max(depth, 1e-6))
        k = (near * high) ** 1.5
        jag = mathutils.noise.noise(
            Vector((v.co.x * freq, v.co.y * freq, v.co.z * freq)))
        if axis == 'x':
            v.co.x -= sgn * abs(jag) * amount * k
        else:
            v.co.y -= sgn * abs(jag) * amount * k
        v.co.z -= abs(jag) * amount * k * 0.7


def _crush_corner(bm, w, d, h, amount):
    """Dent one vertical corner inwards. Purely lateral so the analytic top
    surface (and therefore the tape/label placement) stays exact."""
    cx = random.choice([-1.0, 1.0]) * w * 0.5
    cy = random.choice([-1.0, 1.0]) * d * 0.5
    cz = random.uniform(-0.3, 0.5) * h
    centre = Vector((cx, cy, cz))
    reach = random.uniform(0.18, 0.34) * min(w, d)
    push = Vector((-cx, -cy, 0.0)).normalized() * amount
    for v in bm.verts:
        t = (v.co - centre).length
        if t < reach:
            k = (1.0 - t / reach) ** 2
            v.co.x += push.x * k
            v.co.y += push.y * k


def _pick_top_style(ptype):
    """Which deliberate shape the lid gets. 'flat' still means the small bulge
    + noise applied, i.e. not literally planar."""
    if ptype == "gable_box":
        return random.choices(["gable", "twin"], [78, 22])[0]
    if ptype == "wedge_box":
        return "slope"
    if ptype == "stepped_box":
        return "step"
    if ptype == "tapered_box":
        return random.choices(["flat", "dome", "pyramid"], [48, 27, 25])[0]
    if ptype in ("tray_box", "crate"):
        return "rim"
    if ptype == "envelope":
        # a document mailer is flat except for the glued flap across one end
        return random.choices(["flat", "slope", "arch"], [64, 22, 14])[0]
    if ptype in ("padded_mailer", "poly_bag", "sack"):
        return random.choices(["dome", "sag", "slope", "flat"],
                              [58, 13, 13, 16])[0]
    names = list(TOP_STYLE_WEIGHTS)
    return random.choices(names, [TOP_STYLE_WEIGHTS[n] for n in names])[0]


def _smooth01(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def _box_top_shape(style, w, d, h):
    """Analytic lid profile for a cuboid parcel.

    Returns (fn, decal_zone, flat_enough_to_stack_on). `fn(x, y)` is the extra
    height of the lid over h/2 - the mesh is deformed with exactly this
    function, so tape, labels and codes placed with it sit on the real
    surface.
    """
    ex, ey = w * 0.5, d * 0.5

    if style == "dome":
        amp = random.uniform(0.10, 0.34) * h
        p = random.uniform(0.6, 1.8)

        def fn(x, y):
            fx = max(0.0, 1.0 - (x / ex) ** 2)
            fy = max(0.0, 1.0 - (y / ey) ** 2)
            return amp * (fx * fy) ** p

        return fn, None, amp < 0.06 * h

    if style == "sag":
        amp = random.uniform(0.05, 0.17) * h

        def fn(x, y):
            fx = max(0.0, 1.0 - (x / ex) ** 2)
            fy = max(0.0, 1.0 - (y / ey) ** 2)
            return -amp * fx * fy

        return fn, None, False

    if style == "pyramid":
        amp = random.uniform(0.12, 0.42) * h
        t0 = random.uniform(0.15, 0.6)          # plateau fraction
        zone = (0.0, 0.0, ex * t0 * 0.9, ey * t0 * 0.9)

        def fn(x, y):
            t = max(abs(x) / ex, abs(y) / ey)
            k = (1.0 - t) / max(1.0 - t0, 1e-6)
            return amp * min(1.0, max(0.0, k))

        return fn, zone, False

    if style == "gable":
        ra = random.choice(['x', 'y'])          # the ridge runs along this axis
        e = ey if ra == 'x' else ex             # ... and varies across that one
        p0 = random.uniform(-0.35, 0.35) * e
        amp = random.uniform(0.16, 0.5) * h

        def fn(x, y, ra=ra, e=e, p0=p0, amp=amp):
            u = y if ra == 'x' else x
            if u >= p0:
                k = 1.0 - (u - p0) / max(e - p0, 1e-6)
            else:
                k = 1.0 - (p0 - u) / max(e + p0, 1e-6)
            return amp * max(0.0, min(1.0, k))

        # a sticker goes on one roof facet, clear of the ridge
        side = random.choice([-1.0, 1.0])
        lo, hi = (p0, e) if side > 0 else (-e, p0)
        cu = 0.5 * (lo + hi)
        hu = 0.42 * (hi - lo)
        if ra == 'x':
            zone = (0.0, cu, ex * 0.8, hu)
        else:
            zone = (cu, 0.0, hu, ey * 0.8)
        return fn, zone, False

    if style == "twin":
        # a valley down the middle: the lid rises towards both edges
        ra = random.choice(['x', 'y'])
        e = ey if ra == 'x' else ex
        p0 = random.uniform(-0.2, 0.2) * e
        amp = random.uniform(0.12, 0.35) * h

        def fn(x, y, ra=ra, e=e, p0=p0, amp=amp):
            u = y if ra == 'x' else x
            return amp * min(1.0, abs(u - p0) / max(e, 1e-6))

        side = random.choice([-1.0, 1.0])
        cu = p0 + side * e * 0.62
        hu = e * 0.3
        zone = ((0.0, cu, ex * 0.8, hu) if ra == 'x'
                else (cu, 0.0, hu, ey * 0.8))
        return fn, zone, False

    if style == "slope":
        ax = random.choice(['x', 'y'])
        sgn = random.choice([-1.0, 1.0])
        amp = random.uniform(0.18, 0.65) * h

        def fn(x, y, ax=ax, sgn=sgn, amp=amp):
            u = x if ax == 'x' else y
            e = ex if ax == 'x' else ey
            return amp * max(0.0, min(1.0, 0.5 + 0.5 * sgn * u / e))

        # the whole lid stays planar, so anything can be stuck anywhere on it
        return fn, None, False

    if style == "step":
        amp = random.uniform(0.10, 0.36) * h
        hx = ex * random.uniform(0.32, 0.72)
        hy = ey * random.uniform(0.32, 0.72)
        cx = random.uniform(-1.0, 1.0) * max(0.0, ex - hx) * 0.7
        cy = random.uniform(-1.0, 1.0) * max(0.0, ey - hy) * 0.7
        ramp = max(0.004, min(hx, hy) * random.uniform(0.08, 0.25))

        def fn(x, y):
            sx = _smooth01((hx - abs(x - cx)) / ramp)
            sy = _smooth01((hy - abs(y - cy)) / ramp)
            return amp * min(sx, sy)

        zone = (cx, cy, max(hx - ramp, 0.006), max(hy - ramp, 0.006))
        return fn, zone, False

    if style == "arch":
        # a barrel vault: the lid is a half-cylinder across one axis
        ra = random.choice(['x', 'y'])       # the vault runs along this axis
        e = ey if ra == 'x' else ex
        amp = random.uniform(0.18, 0.55) * h

        def fn(x, y, ra=ra, e=e, amp=amp):
            u = (y if ra == 'x' else x) / max(e, 1e-6)
            return amp * math.sqrt(max(0.0, 1.0 - min(1.0, u * u)))

        # the crown of the vault is the flattest strip on it
        hu = e * 0.22
        zone = ((0.0, 0.0, ex * 0.8, hu) if ra == 'x'
                else (0.0, 0.0, hu, ey * 0.8))
        return fn, zone, False

    if style == "hip":
        # hipped roof: a ridge along the long axis with four sloped facets
        ra = 'x' if w >= d else 'y'
        amp = random.uniform(0.14, 0.45) * h
        # how far the ridge runs, as a fraction of the long half-extent
        rl = random.uniform(0.15, 0.6)

        def fn(x, y, ra=ra, amp=amp, rl=rl):
            if ra == 'x':
                a, b, ea, eb = x, y, ex, ey
            else:
                a, b, ea, eb = y, x, ey, ex
            ka = max(0.0, (abs(a) / max(ea, 1e-6) - rl) / max(1.0 - rl, 1e-6))
            kb = abs(b) / max(eb, 1e-6)
            return amp * max(0.0, 1.0 - max(ka, kb))

        zone = ((0.0, 0.0, ex * rl * 0.8, ey * 0.14) if ra == 'x'
                else (0.0, 0.0, ex * 0.14, ey * rl * 0.8))
        return fn, zone, False

    if style == "double_step":
        # a lid on a lid: two concentric raised plateaus
        a1 = random.uniform(0.08, 0.22) * h
        a2 = a1 * random.uniform(0.5, 1.1)
        h1x, h1y = ex * random.uniform(0.55, 0.85), ey * random.uniform(0.55, 0.85)
        h2x, h2y = h1x * random.uniform(0.35, 0.7), h1y * random.uniform(0.35, 0.7)
        cx = random.uniform(-1.0, 1.0) * max(0.0, ex - h1x) * 0.6
        cy = random.uniform(-1.0, 1.0) * max(0.0, ey - h1y) * 0.6
        ramp = max(0.003, min(h2x, h2y) * random.uniform(0.10, 0.28))

        def fn(x, y):
            s1 = min(_smooth01((h1x - abs(x - cx)) / ramp),
                     _smooth01((h1y - abs(y - cy)) / ramp))
            s2 = min(_smooth01((h2x - abs(x - cx)) / ramp),
                     _smooth01((h2y - abs(y - cy)) / ramp))
            return a1 * s1 + a2 * s2

        zone = (cx, cy, max(h2x - ramp, 0.006), max(h2y - ramp, 0.006))
        return fn, zone, False

    if style == "rim":
        # an open crate / tray: the walls stand proud of a recessed floor
        amp = random.uniform(0.25, 0.65) * h
        tx = ex * random.uniform(0.10, 0.20)        # wall thickness
        ty = ey * random.uniform(0.10, 0.20)
        ramp = max(0.002, min(tx, ty) * 0.45)

        def fn(x, y):
            inx = _smooth01(((ex - tx) - abs(x)) / ramp)
            iny = _smooth01(((ey - ty) - abs(y)) / ramp)
            return amp * (1.0 - min(inx, iny))

        # the recessed floor is where anything printed or stuck ends up.
        # Nothing is stacked on an open crate: a child would rest on the rim
        # and float over the recess.
        zone = (0.0, 0.0, max(ex - tx - ramp, 0.006),
                max(ey - ty - ramp, 0.006))
        return fn, zone, False

    return None, None, True             # "flat" - handled by the caller


def _round_top_shape(style, radius, h):
    """Same idea for cylinders / prisms: (fn, decal_zone, stackable).

    Amplitudes stay well under the parcel height on purpose - a drum with a
    tall dome stops reading as a parcel and starts reading as an egg.
    """
    if style == "dome":
        amp = random.uniform(0.05, 0.20) * h
        p = random.uniform(0.7, 1.6)

        def fn(x, y):
            q = max(0.0, 1.0 - (x * x + y * y) / (radius * radius))
            return amp * q ** p

        return fn, None, amp < 0.05 * h

    if style == "cone":
        amp = random.uniform(0.08, 0.28) * h
        t0 = random.uniform(0.15, 0.55)
        zone = (0.0, 0.0, radius * t0 * 0.65, radius * t0 * 0.65)

        def fn(x, y):
            t = math.hypot(x, y) / radius
            return amp * min(1.0, max(0.0, (1.0 - t) / max(1.0 - t0, 1e-6)))

        return fn, zone, False

    if style == "slant":
        amp = random.uniform(0.12, 0.5) * h
        ang = random.uniform(0, math.tau)
        ca, sa = math.cos(ang), math.sin(ang)

        def fn(x, y):
            u = (x * ca + y * sa) / radius
            return amp * max(0.0, min(1.0, 0.5 + 0.5 * u))

        return fn, None, False

    if style == "lid":
        # a smaller disc sitting proud of the top, like a screw cap
        amp = random.uniform(0.06, 0.22) * h
        rr = radius * random.uniform(0.35, 0.72)
        ramp = max(0.003, rr * random.uniform(0.10, 0.30))
        zone = (0.0, 0.0, max(rr - ramp, 0.004) * 0.7,
                max(rr - ramp, 0.004) * 0.7)

        def fn(x, y):
            return amp * _smooth01((rr - math.hypot(x, y)) / ramp)

        return fn, zone, False

    if style == "rim":
        # an open tub / bucket: an annular wall around a recessed floor
        amp = random.uniform(0.20, 0.55) * h
        rin = radius * random.uniform(0.76, 0.9)
        ramp = max(0.002, (radius - rin) * 0.5)

        def fn(x, y):
            return amp * (1.0 - _smooth01((rin - math.hypot(x, y)) / ramp))

        zone = (0.0, 0.0, max(rin - ramp, 0.005) * 0.7,
                max(rin - ramp, 0.005) * 0.7)
        return fn, zone, False

    return None, None, True


def _top_weight(z, zt, bevel, band):
    """How much of the lid profile a vertex at height `z` receives.

    The top face itself moves rigidly (weight 1), the rounded rim and the
    upper part of the side walls blend it away, so a gable really is a gable
    and not a bulge.
    """
    z1 = zt - min(bevel * 1.3, band * 0.5)
    z0 = zt - band
    if z >= z1:
        return 1.0
    if z <= z0:
        return 0.0
    return _smooth01((z - z0) / max(z1 - z0, 1e-6))


def _tessellate_cap(bm, zt):
    """Give the top n-gon of a cylinder/prism interior vertices, so a domed or
    coned lid has something to actually move."""
    caps = [f for f in bm.faces
            if len(f.verts) > 3 and all(v.co.z > zt - 1e-4 for v in f.verts)]
    if not caps:
        return
    try:
        res = bmesh.ops.poke(bm, faces=caps)
        new_faces = res.get("faces", [])
        edges = {e for f in new_faces if f.is_valid for e in f.edges}
        if edges:
            bmesh.ops.subdivide_edges(bm, edges=list(edges), cuts=2)
    except Exception:
        pass


def _make_object(name, bm, mat):
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    me.materials.append(mat)
    for poly in me.polygons:
        poly.use_smooth = True
    ob = bpy.data.objects.new(name, me)
    WORK_COLL.objects.link(ob)
    return ob


def build_parcel(ptype, mat, size_scale=1.0):
    lo_w, hi_w, lo_d, hi_d, lo_h, hi_h = SIZE[ptype]
    w = random.uniform(lo_w, hi_w) * size_scale
    d = random.uniform(lo_d, hi_d) * size_scale
    h = random.uniform(lo_h, hi_h) * size_scale

    cyl = ptype in ROUND_TYPES
    lying = ptype in LYING_TYPES
    if cyl:
        d = w                                   # diameter
    elif lying:
        d = h                                   # round cross-section on its side

    # One axis stretched or squashed: the same type covers near-cubes, long
    # slabs and narrow columns. Round parcels only stretch in height, so they
    # stay circular in plan; a parcel on its side only stretches in length.
    if random.random() < ASPECT_STRETCH_PROB:
        k = random.uniform(*ASPECT_STRETCH)
        if cyl:
            axis = 'h'
        elif lying:
            axis = 'w'
        else:
            axis = random.choices(['w', 'd', 'h'], [40, 40, 20])[0]
        if axis == 'w':
            w = max(0.05, min(w * k, 0.45))
        elif axis == 'd':
            d = max(0.05, min(d * k, 0.42))
        else:
            h = max(0.02, min(h * k, 0.40))

    bm = bmesh.new()
    seed = random.uniform(0, 500)
    shape = {}

    if lying:
        # A cylinder resting on its side: rolled goods, a poster tube that has
        # been laid down, a taped-up roll of underlay. Built along Z and then
        # rotated onto its side, so the analytic top surface is the exact
        # circular section  z = sqrt(r^2 - y^2).
        rad = h * 0.5
        length = w
        segs = random.choice([28, 40, 56])
        if random.random() < 0.18:              # a rolled-up n-gon bundle
            segs = random.choice([8, 10, 12])
        try:
            bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False,
                                  segments=segs, radius1=rad, radius2=rad,
                                  depth=length)
        except TypeError:
            bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False,
                                  segments=segs, diameter1=rad * 2,
                                  diameter2=rad * 2, depth=length)
        bmesh.ops.subdivide_edges(
            bm, edges=[e for e in bm.edges
                       if abs(e.verts[0].co.z - e.verts[1].co.z) > 1e-6],
            cuts=random.choice([2, 3, 4]))
        # lay it down: local +Z becomes +X
        rot = mathutils.Matrix.Rotation(math.pi * 0.5, 4, 'Y')
        bmesh.ops.transform(bm, matrix=rot, verts=bm.verts[:])

        cap = 0.3 * min(rad, length * 0.5)
        bevel = min(random.uniform(0.002, 0.020), cap)
        _bevel_group(bm, _sharp_edges(bm), bevel,
                     random.choices([1, 2, 3, 4], [18, 26, 30, 26])[0],
                     random.uniform(0.25, 0.75))

        # ends are rarely perfectly cut: taper the last stretch a little
        end_taper = random.uniform(0.0, 0.14)
        dent = random.uniform(0.0, 0.0025)
        nsc = random.uniform(6.0, 20.0)
        hl = length * 0.5
        for v in bm.verts:
            k = max(0.0, (abs(v.co.x) / max(hl, 1e-6) - 0.82) / 0.18)
            f = 1.0 - end_taper * k
            rr = math.hypot(v.co.y, v.co.z)
            if rr > 1e-6:
                g = f + _noise3(v.co.x * nsc, v.co.y * nsc,
                                v.co.z * nsc, seed) * dent / max(rr, 1e-6)
                v.co.y *= g
                v.co.z *= g

        radius = None
        # Extents of the strip along the crown that the tape / label / code may
        # use: full length minus the end caps, and the upper part of the roll.
        top_w = length * (1.0 - end_taper) * 0.94
        top_d = 2.0 * rad * 0.62

        def top_fn(x, y, rad=rad):
            return math.sqrt(max(rad * rad - y * y, 0.0)) - rad

        zone = (0.0, 0.0, top_w * 0.42, rad * 0.30)
        top_flat = False
        shape = {"sides": segs, "lying": True,
                 "end_taper": round(end_taper, 3)}

    elif cyl:
        if ptype == "hex_box":
            segs = 6
        elif ptype == "prism":
            segs = random.choices(PRISM_SIDES, PRISM_SIDE_WEIGHTS)[0]
        elif ptype == "drum" and random.random() < 0.12:
            segs = 8
        else:
            segs = random.choice([32, 48, 64])
        r = w * 0.5
        taper = 1.0
        if ptype == "drum":
            taper = random.uniform(0.86, 1.0)
        elif ptype == "round_tin":
            taper = random.uniform(0.95, 1.0)
        elif ptype == "tub":
            taper = (random.uniform(1.10, 1.38) if random.random() < 0.7
                     else random.uniform(0.62, 0.85))
        elif ptype == "barrel":
            taper = random.uniform(0.92, 1.06)
        elif ptype == "hex_box" and random.random() < 0.25:
            taper = random.uniform(0.78, 0.96)
        elif ptype == "prism" and random.random() < PRISM_TAPER_PROB:
            taper = (random.uniform(0.58, 0.92) if random.random() < 0.75
                     else random.uniform(1.06, 1.26))
        r_bot, r_top = r, r * taper
        try:
            bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False,
                                  segments=segs, radius1=r_bot, radius2=r_top,
                                  depth=h)
        except TypeError:
            bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False,
                                  segments=segs, diameter1=r_bot * 2,
                                  diameter2=r_top * 2, depth=h)
        # a couple of horizontal loops so the side can dent
        bmesh.ops.subdivide_edges(
            bm, edges=[e for e in bm.edges
                       if abs(e.verts[0].co.z - e.verts[1].co.z) > 1e-6],
            cuts=random.choice([1, 2, 3]))

        # irregular n-gon: every corner gets its own radius, so a "hexagonal"
        # parcel is not a textbook hexagon
        jitter_min = 1.0
        if (ptype in ("prism", "hex_box") and segs <= 12
                and random.random() < PRISM_IRREGULAR_PROB):
            step = math.tau / segs
            offs = [random.uniform(-0.13, 0.13) for _ in range(segs)]
            jitter_min = 1.0 + min(offs)
            for v in bm.verts:
                rad = math.hypot(v.co.x, v.co.y)
                if rad < 1e-6:
                    continue
                k = int(round(math.atan2(v.co.y, v.co.x) / step)) % segs
                f = 1.0 + offs[k]
                v.co.x *= f
                v.co.y *= f

        # Profiled wall: the radius is modulated over the height, which turns a
        # straight-walled drum into a barrel, a waisted tin or a stepped
        # canister. `taper` above stays a straight cone on top of this.
        profile = "straight"
        if ptype == "barrel" or random.random() < BARREL_PROFILE_PROB:
            profile = random.choices(["bulge", "waist", "step"],
                                     [52, 28, 20])[0]
            amp = random.uniform(*BARREL_BULGE)
            if profile == "waist":
                amp *= -0.8
            t_step = random.uniform(0.35, 0.7)
            for v in bm.verts:
                t = (v.co.z + h * 0.5) / max(h, 1e-6)
                t = max(0.0, min(1.0, t))
                if profile == "step":
                    f = 1.0 + amp * _smooth01((t - t_step) / 0.06)
                else:
                    f = 1.0 + amp * math.sin(math.pi * t)
                v.co.x *= f
                v.co.y *= f
            # sin(pi t) vanishes at both caps, so only the step profile
            # actually changes the radius of the top face
            if profile == "step":
                r_top *= 1.0 + amp

        # rim rounding: top and bottom rims get independent width / segments /
        # profile, so lids read differently from bases
        cap = 0.3 * min(r_top, h)
        w_top = min(random.uniform(0.003, 0.024), cap)
        w_bot = min(random.uniform(0.002, 0.020), cap)
        rims_top = [e for e in _sharp_edges(bm)
                    if 0.5 * (e.verts[0].co.z + e.verts[1].co.z) > 0]
        rims_bot = [e for e in _sharp_edges(bm)
                    if 0.5 * (e.verts[0].co.z + e.verts[1].co.z) <= 0]
        _bevel_group(bm, rims_top, w_top,
                     random.choices([1, 2, 3, 4, 5], [14, 20, 26, 22, 18])[0],
                     random.uniform(0.25, 0.75))
        _bevel_group(bm, rims_bot, w_bot,
                     random.choices([1, 2, 3, 4], [22, 26, 28, 24])[0],
                     random.uniform(0.25, 0.75))
        bevel = w_top

        dent = random.uniform(0.0, 0.0022)
        nsc = random.uniform(6.0, 22.0)
        for v in bm.verts:
            if v.co.z < -h * 0.5 + 1e-5:
                continue
            k = _noise3(v.co.x * nsc, v.co.y * nsc, v.co.z * nsc, seed) * dent
            rad = math.hypot(v.co.x, v.co.y)
            if rad > 1e-6:
                v.co.x += v.co.x / rad * k
                v.co.y += v.co.y / rad * k

        # Analytic top radius. For a low-sided prism the inscribed radius
        # (the apothem) is what keeps a sticker on the face rather than
        # hanging over a corner.
        radius = r_top * (math.cos(math.pi / segs) if segs <= 12 else 1.0)
        radius = max(radius * jitter_min, 1e-3)

        if ptype == "tub":
            tstyle = random.choices(["rim", "flat", "lid"], [70, 18, 12])[0]
        elif ptype == "tube" or segs <= 4:
            # a coned lid on a 3- or 4-sided prism turns it into a pyramid,
            # which is no longer a package
            tstyle = random.choices(["flat", "dome", "slant"], [70, 18, 12])[0]
        else:
            tstyle = random.choices(
                ["flat", "dome", "cone", "slant", "lid", "rim"],
                [40, 19, 11, 11, 10, 9])[0]
        shape_fn, zone, top_flat = _round_top_shape(tstyle, radius, h)
        if shape_fn is None:
            top_fn = (lambda x, y: 0.0)
        else:
            top_fn = shape_fn
            _tessellate_cap(bm, h * 0.5)
            zt = h * 0.5
            band = max(2.5 * bevel, random.uniform(0.15, 0.45) * h)
            for v in bm.verts:
                wgt = _top_weight(v.co.z, zt, bevel, band)
                if wgt > 0.0:
                    v.co.z += wgt * shape_fn(v.co.x, v.co.y)
        top_w = top_d = 2.0 * r_top
        shape = {"sides": segs, "taper": round(taper, 3), "top": tstyle,
                 "irregular": jitter_min != 1.0, "wall": profile}

    else:
        radius = None
        style = _pick_top_style(ptype)
        # a step or a ridge needs vertices to sit on
        fine = style in ("step", "pyramid", "gable", "twin", "dome",
                         "arch", "hip", "double_step", "rim")
        cuts = random.choice([9, 11, 13]) if fine else random.choice([4, 5, 6, 8])
        bmesh.ops.create_cube(bm, size=1.0)
        for v in bm.verts:
            v.co.x *= w
            v.co.y *= d
            v.co.z *= h
        bmesh.ops.subdivide_edges(bm, edges=bm.edges[:], cuts=cuts,
                                  use_grid_fill=True)

        # truncated pyramid: the walls lean in (or, less often, out)
        taper = 1.0
        if ptype == "tapered_box":
            taper = (random.uniform(0.52, 0.88) if random.random() < 0.75
                     else random.uniform(1.08, 1.32))
        elif ptype in ("box", "box_soft", "tall_box") and random.random() < 0.14:
            taper = random.uniform(0.82, 0.97)
        if taper != 1.0:
            for v in bm.verts:
                k = (v.co.z + h * 0.5) / h
                s = 1.0 + (taper - 1.0) * k
                v.co.x *= s
                v.co.y *= s
        top_w, top_d = w * taper, d * taper

        # Bevel is clamped relative to the parcel's own size, otherwise small
        # parcels turn into pillows and stop reading as boxes.
        small = min(w, d, h)
        if ptype == "sack":
            # not too round, or it stops reading as a bag and becomes an egg
            base_w = random.uniform(0.20, 0.32) * small
            corner_style = random.choice(["uniform", "soft_top"])
        elif ptype in ("padded_mailer", "poly_bag"):
            base_w = random.uniform(0.22, 0.40) * small
            corner_style = random.choice(["uniform", "soft_top",
                                          "soft_verticals"])
        elif ptype == "box_soft":
            base_w = min(random.uniform(0.010, 0.040), 0.20 * small)
            corner_style = None
        elif ptype == "flat_box":
            base_w = min(random.uniform(0.002, 0.010), 0.16 * small)
            corner_style = random.choice(["uniform", "chamfer",
                                          "soft_verticals", "mixed", "worn"])
        elif ptype == "envelope":
            # millimetres thick: the bevel has to stay tiny or the whole
            # mailer turns into a lens
            base_w = min(random.uniform(0.0008, 0.004), 0.28 * small)
            corner_style = random.choice(["uniform", "chamfer", "worn"])
        elif ptype in ("tray_box", "crate"):
            # crates are crisp: sawn timber and stapled board, not rolled card
            base_w = min(random.uniform(0.001, 0.007), 0.10 * small)
            corner_style = random.choice(["uniform", "chamfer"])
        else:
            base_w = min(random.uniform(0.002, 0.022), 0.12 * small)
            corner_style = None
        bevel = _round_corners(bm, base_w, small, corner_style)

        # a dented corner on some rigid boxes
        if ptype in BOX_TYPES and random.random() < CORNER_DENT_PROB:
            _crush_corner(bm, w, d, h,
                          min(random.uniform(0.006, 0.030), 0.22 * small))

        # ... and a torn-open edge on some of them
        if ptype in BOX_TYPES and random.random() < RIPPED_EDGE_PROB:
            _rip_edge(bm, w, d, h,
                      min(random.uniform(0.008, 0.028), 0.3 * small))

        # lateral dents / soft-parcel wrinkles
        # Surface irregularity. A third of the rigid boxes are "crumpled":
        # visibly bowed/creased panels rather than perfectly flat card.
        # sawn timber and stapled crates do not crumple
        crumpled = (random.random() < CRUMPLED_PROB
                    and ptype not in ("tray_box", "crate"))
        if ptype in ("tray_box", "crate", "envelope"):
            dent, nsc = random.uniform(0.0, 0.0015), random.uniform(6.0, 22.0)
        elif ptype == "sack":
            dent, nsc = random.uniform(0.008, 0.020), random.uniform(5.0, 14.0)
        elif ptype == "poly_bag":
            dent, nsc = random.uniform(0.004, 0.012), random.uniform(8.0, 26.0)
        elif ptype == "padded_mailer":
            dent, nsc = random.uniform(0.001, 0.005), random.uniform(6.0, 18.0)
        elif crumpled:
            dent, nsc = random.uniform(0.003, 0.009), random.uniform(4.0, 16.0)
        else:
            dent, nsc = random.uniform(0.0, 0.0025), random.uniform(5.0, 20.0)
        for v in bm.verts:
            if v.co.z < -h * 0.5 + 1e-5:
                continue
            v.co.x += _noise3(v.co.x * nsc, v.co.y * nsc, v.co.z * nsc, seed) * dent
            v.co.y += _noise3(v.co.y * nsc, v.co.z * nsc, v.co.x * nsc,
                              seed + 31.7) * dent

        # top shaping, in two parts:
        #  * the base bulge + noise, faded to 0 at the edges;
        #  * the deliberate lid profile (gable / slope / step / ...), which is
        #    applied to the top face rigidly and blended away down the walls.
        if ptype in ("padded_mailer", "poly_bag", "sack"):
            dome = random.uniform(0.10, 0.30) * h
            tn = random.uniform(0.15, 0.5) * dome
        elif ptype == "box_soft":
            dome = random.uniform(0.0, 0.055) * min(w, d) / 0.2
            tn = random.uniform(0.0, 0.4) * dome
        else:
            dome = random.uniform(0.0, 0.02) * min(w, d) / 0.2
            tn = random.uniform(0.0, 0.3) * dome
        if crumpled:
            dome = max(dome, random.uniform(0.004, 0.014))
            tn = max(tn, random.uniform(0.3, 0.9) * dome)
        tsc = random.uniform(6.0, 20.0)

        def base_fn(x, y, w=top_w, d=top_d, dome=dome, tn=tn, tsc=tsc,
                    seed=seed):
            fx = max(0.0, 1.0 - (2.0 * x / w) ** 2)
            fy = max(0.0, 1.0 - (2.0 * y / d) ** 2)
            fall = fx * fy
            return fall * (dome + _noise3(x * tsc, y * tsc, 0.0, seed + 77.3) * tn)

        shape_fn, zone, top_flat = _box_top_shape(style, top_w, top_d, h)
        if shape_fn is None:
            top_fn = base_fn
        else:
            def top_fn(x, y, base_fn=base_fn, shape_fn=shape_fn):
                return base_fn(x, y) + shape_fn(x, y)

        zt = h * 0.5
        band = max(2.5 * bevel, random.uniform(0.20, 0.55) * h)
        for v in bm.verts:
            z = v.co.z                          # weights use the original z
            dz = 0.0
            if z > 0.0:
                dz += min(1.0, (z / zt) ** 2) * base_fn(v.co.x, v.co.y)
            if shape_fn is not None:
                wgt = _top_weight(z, zt, bevel, band)
                if wgt > 0.0:
                    dz += wgt * shape_fn(v.co.x, v.co.y)
            v.co.z += dz

        shape = {"top": style, "taper": round(taper, 3), "cuts": cuts,
                 "crumpled": crumpled}

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    ob = _make_object(f"parcel_{ptype}", bm, mat)
    return Parcel(ob, ptype, w, d, h, bevel, radius, top_fn,
                  top_w=top_w, top_d=top_d, decal_zone=zone,
                  top_flat=top_flat, shape=shape)


# ---- tape ---------------------------------------------------------------
def _strip_geometry(bmv, parcel, axis, center, width, flap, lift, wobble):
    """One tape strip: flat top -> bevel arc -> flap down the side.

    axis 'y' means the strip runs along Y and is `width` wide in X.
    """
    ncols = max(3, min(12, int(width / 0.012) + 3))
    nlen = 14
    narc = 4
    r = max(parcel.bevel, 1e-4)
    h2 = parcel.h * 0.5

    cols = [center - width * 0.5 + width * i / (ncols - 1) for i in range(ncols)]
    grid = []                                   # [col][row] -> vert

    for c in cols:
        flat = parcel.flat_half('y' if axis == 'y' else 'x', c)
        edge = parcel.edge_half('y' if axis == 'y' else 'x', c)
        pts = []
        # flat run
        for j in range(nlen):
            t = -1.0 + 2.0 * j / (nlen - 1)
            u = t * flat
            x, y = (c, u) if axis == 'y' else (u, c)
            z = parcel.top_z(x, y) + lift
            z += wobble * _noise3(x * 40, y * 40, 0.0, 11.0)
            pts.append((x, y, z))
        # arcs at both ends + flaps. The arc is anchored to the lid height at
        # the point where the flat top meets the bevel, so the tape follows a
        # sloped or gabled lid over the edge instead of jumping to h/2.
        head, tail = [], []
        for sgn, dst in ((-1.0, head), (1.0, tail)):
            xe, ye = ((c, sgn * flat) if axis == 'y' else (sgn * flat, c))
            z_e = parcel.top_z(xe, ye)
            for k in range(1, narc + 1):
                t = (math.pi * 0.5) * k / narc
                u = sgn * (flat + (edge - flat) * math.sin(t))
                zz = z_e - r + r * math.cos(t)
                nx = math.sin(t) * sgn
                nz = math.cos(t)
                x, y = (c, u) if axis == 'y' else (u, c)
                if axis == 'y':
                    dst.append((x, y + nx * lift, zz + nz * lift))
                else:
                    dst.append((x + nx * lift, y, zz + nz * lift))
            if flap > 1e-4:
                u = sgn * edge
                x, y = (c, u) if axis == 'y' else (u, c)
                zf = min(z_e - r - flap, h2 - r * 0.25)
                if axis == 'y':
                    dst.append((x, y + sgn * lift, zf))
                else:
                    dst.append((x + sgn * lift, y, zf))
        grid.append(list(reversed(head)) + pts + tail)

    verts = [[bmv.verts.new(p) for p in col] for col in grid]
    for i in range(len(verts) - 1):
        for j in range(len(verts[i]) - 1):
            try:
                bmv.faces.new((verts[i][j], verts[i + 1][j],
                               verts[i + 1][j + 1], verts[i][j + 1]))
            except ValueError:
                pass


def build_tape(parcel, mat, style=None, width=None, thin=False):
    """Returns a tape object (child of the parcel) or None."""
    style = style or random.choices(
        ["seam", "h", "cross", "double", "wrapped"], [40, 12, 23, 12, 13])[0]

    long_axis = 'y' if parcel.d >= parcel.w else 'x'
    short_axis = 'x' if long_axis == 'y' else 'y'
    span = parcel.flat_half(short_axis, 0.0)

    lift = TAPE_LIFT * random.uniform(0.85, 1.25)
    wobble = random.uniform(0.0, 0.00015)

    if style == "film":
        # shrink wrap: one sheet covering the whole top, wrapped over two edges
        w = 2.0 * parcel.flat_half(short_axis, 0.0) * 0.99
        strips = [(long_axis, 0.0, w)]
        flap = random.uniform(0.45, 0.95) * parcel.h * 0.5
        bmv = bmesh.new()
        _strip_geometry(bmv, parcel, long_axis, 0.0, w, flap, LIFT_FILM, wobble)
        if not bmv.faces:
            bmv.free()
            return None
        bmesh.ops.recalc_face_normals(bmv, faces=bmv.faces)
        ob = _make_object("wrapfilm", bmv, mat)
        ob.parent = parcel.obj
        ob.matrix_parent_inverse = mathutils.Matrix.Identity(4)
        return ob

    w = width or random.uniform(TAPE_WIDTH_MIN, TAPE_WIDTH_MAX)
    if thin:
        w = random.uniform(0.006, 0.011)
    # keep the tape inside the flat part of the top
    max_w = 0.75 * min(
        parcel.flat_half('x', 0.0), parcel.flat_half('y', 0.0)) * 2.0
    w = min(w, max(max_w, 0.004))

    flap = random.uniform(0.0, 0.55) * parcel.h * 0.5

    strips = []
    if style == "seam":
        strips.append((long_axis, random.uniform(-0.12, 0.12) * span, w))
    elif style == "h":
        off = span * random.uniform(0.5, 0.78)
        w = w * 0.8                     # 3 strips: keep each one narrow
        strips += [(long_axis, 0.0, w), (long_axis, off, w), (long_axis, -off, w)]
    elif style == "cross":
        strips += [(long_axis, random.uniform(-0.1, 0.1) * span, w),
                   (short_axis, random.uniform(-0.1, 0.1) *
                    parcel.flat_half(long_axis, 0.0), w * random.uniform(0.8, 1.1))]
    elif style == "double":
        off = span * random.uniform(0.25, 0.5)
        strips += [(long_axis, off, w), (long_axis, -off, w)]
    else:   # "wrapped": parcel bound up in several rounds of tape
        w *= random.uniform(0.7, 1.0)
        n_long = random.randint(2, 4)
        n_short = random.randint(1, 3)
        span_s = parcel.flat_half(long_axis, 0.0)
        for i in range(n_long):
            off = ((i - (n_long - 1) * 0.5) / max(n_long, 1)) * 2.0 * span * 0.85
            strips.append((long_axis, off, w))
        for i in range(n_short):
            off = ((i - (n_short - 1) * 0.5) / max(n_short, 1)) * 2.0 * span_s * 0.85
            strips.append((short_axis, off, w * random.uniform(0.8, 1.1)))

    bmv = bmesh.new()
    for i, (axis, c, ww) in enumerate(strips):
        # `c` is an offset on the axis perpendicular to the strip direction
        if abs(c) + ww * 0.5 > parcel.flat_half('x' if axis == 'y' else 'y', 0.0):
            c = 0.0
            strips[i] = (axis, c, ww)
        _strip_geometry(bmv, parcel, axis, c, ww, flap, lift, wobble)
    parcel.tape_bands.extend(strips)
    if not bmv.faces:
        bmv.free()
        return None
    bmesh.ops.recalc_face_normals(bmv, faces=bmv.faces)
    ob = _make_object("tape", bmv, mat)
    ob.parent = parcel.obj
    ob.matrix_parent_inverse = mathutils.Matrix.Identity(4)
    return ob


def build_strap(parcel, mat):
    return build_tape(parcel, mat,
                      style=random.choice(["seam", "cross", "double"]),
                      thin=True)


# ---- shipping label ------------------------------------------------------
def build_label(parcel, mat, avoid=None):
    fx = parcel.flat_half('x', 0.0)
    fy = parcel.flat_half('y', 0.0)
    if min(fx, fy) < 0.022:
        return None

    lw = min(random.uniform(0.045, 0.085), 1.5 * fx)
    lh = lw * random.uniform(0.55, 0.8)
    # Keep the label inside the flat part of the top: top_z() is only valid
    # there, so an overhanging label would float over the rounded edge.
    lim = min(fx, fy) * 0.98
    hd = 0.5 * math.hypot(lw, lh)
    if hd > lim:
        lw *= lim / hd
        lh *= lim / hd
    if lw < 0.02:
        return None
    yaw = random.choice([0.0, math.pi * 0.5]) + random.uniform(-0.25, 0.25)
    ca, sa = math.cos(yaw), math.sin(yaw)

    # centre, kept inside the flat area (and inside the locally flat patch of
    # a shaped lid) and off the Data Matrix
    half_diag = 0.5 * math.hypot(lw, lh)
    rx = max(0.0, fx - half_diag)
    ry = max(0.0, fy - half_diag)
    lo_x, hi_x, lo_y, hi_y = -rx, rx, -ry, ry
    if parcel.decal_zone is not None:
        zx, zy, zhx, zhy = parcel.decal_zone
        m = half_diag * 0.5
        ax, bx = max(lo_x, zx - zhx + m), min(hi_x, zx + zhx - m)
        ay, by = max(lo_y, zy - zhy + m), min(hi_y, zy + zhy - m)
        if ax <= bx and ay <= by:
            lo_x, hi_x, lo_y, hi_y = ax, bx, ay, by
    cx = cy = 0.0
    for _ in range(10):
        cx = random.uniform(lo_x, hi_x)
        cy = random.uniform(lo_y, hi_y)
        if avoid is None:
            break
        ax, ay, ar = avoid
        if math.hypot(cx - ax, cy - ay) > (ar + half_diag) * 0.95:
            break
    else:
        return None                     # no room next to the code

    n = 4
    bmv = bmesh.new()
    uv_layer = bmv.loops.layers.uv.new("UVMap")
    grid = []
    for i in range(n + 1):
        row = []
        for j in range(n + 1):
            u, v = i / n, j / n
            lx = (u - 0.5) * lw
            ly = (v - 0.5) * lh
            x = cx + lx * ca - ly * sa
            y = cy + lx * sa + ly * ca
            z = parcel.top_z(x, y) + LIFT_LABEL
            row.append((bmv.verts.new((x, y, z)), (u, v)))
        grid.append(row)

    for i in range(n):
        for j in range(n):
            quad = [grid[i][j], grid[i + 1][j], grid[i + 1][j + 1], grid[i][j + 1]]
            try:
                f = bmv.faces.new([q[0] for q in quad])
            except ValueError:
                continue
            for loop, (_, uv) in zip(f.loops, quad):
                loop[uv_layer].uv = uv

    if not bmv.faces:
        bmv.free()
        return None
    bmesh.ops.recalc_face_normals(bmv, faces=bmv.faces)
    ob = _make_object("label", bmv, mat)
    ob.parent = parcel.obj
    ob.matrix_parent_inverse = mathutils.Matrix.Identity(4)
    return ob


# ============================================================
# PLACEMENT / PROJECTION
# ============================================================
def discard(ob):
    """Remove an object and its (now unused) mesh."""
    me = ob.data if ob.type == 'MESH' else None
    try:
        bpy.data.objects.remove(ob, do_unlink=True)
    except Exception:
        return
    if me is not None and me.users == 0:
        try:
            bpy.data.meshes.remove(me)
        except Exception:
            pass


def bbox_world(obj):
    return [obj.matrix_world @ Vector(c) for c in obj.bound_box]


def view_bounds():
    """Axis-aligned bounds of the camera's view footprint on the floor."""
    frame = [camera.matrix_world @ v for v in camera.data.view_frame(scene=scene)]
    origin = camera.matrix_world.translation
    pts = []
    for f in frame:
        dv = f - origin
        if abs(dv.z) < 1e-9:
            return None
        t = (FLOOR_Z - origin.z) / dv.z
        if t <= 0:
            return None
        pts.append(origin + dv * t)
    xs = [p.x for p in pts]
    ys = [p.y for p in pts]
    return min(xs), max(xs), min(ys), max(ys)


def project(pw):
    p = camera.matrix_world.inverted() @ pw
    if p.z >= 0:
        return None
    u = FX * (p.x / -p.z) + RES * 0.5
    v = FX * (-p.y / -p.z) + RES * 0.5
    if 0 <= u <= RES and 0 <= v <= RES:
        return (u, v)
    return None


def visible_corners(obj):
    return [p for p in (project(c) for c in bbox_world(obj)) if p is not None]


def is_visible(obj, need=None):
    return len(visible_corners(obj)) >= (need or MIN_VISIBLE_CORNERS)


def bbox_reasonable(obj):
    pts = visible_corners(obj)
    if len(pts) < 4:
        return False
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (max(xs) - min(xs)) <= RES * 1.2 and (max(ys) - min(ys)) <= RES * 1.2


def footprint_xy(obj):
    return [(c.x, c.y) for c in bbox_world(obj)]


def obb_axes(obj):
    m = obj.matrix_world
    ax = mathutils.Vector((m[0][0], m[1][0]))
    ay = mathutils.Vector((m[0][1], m[1][1]))
    if ax.length:
        ax.normalize()
    if ay.length:
        ay.normalize()
    return [ax, ay]


def footprints_overlap(a, b, gap=0.0):
    pa, pb = footprint_xy(a), footprint_xy(b)
    for ax in obb_axes(a) + obb_axes(b):
        if ax.length == 0:
            continue
        pra = [ax.x * x + ax.y * y for x, y in pa]
        prb = [ax.x * x + ax.y * y for x, y in pb]
        if max(pra) + gap < min(prb) or max(prb) + gap < min(pra):
            return False
    return True


def rest_on(obj, z_surface):
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    oe = obj.evaluated_get(dg)
    min_z = min((oe.matrix_world @ Vector(c)).z for c in oe.bound_box)
    obj.location.z += (z_surface - min_z)


def place(parcel, placed, bounds, gap):
    """Most parcels must be fully inside the frame; a minority are allowed to
    be truncated at the border (realistic, and useful training signal)."""
    x0, x1, y0, y1 = bounds
    mx = (x1 - x0) * 0.06
    my = (y1 - y0) * 0.06
    ob = parcel.obj
    need = 8 if random.random() < TRUNCATION_FREE_PROB else MIN_VISIBLE_CORNERS
    for attempt in range(PLACE_ATTEMPTS):
        if attempt == PLACE_ATTEMPTS // 2:
            need = MIN_VISIBLE_CORNERS      # relax rather than drop the parcel
        ob.location.x = random.uniform(x0 + mx, x1 - mx)
        ob.location.y = random.uniform(y0 + my, y1 - my)
        ob.rotation_euler[2] = random.uniform(0, math.tau)
        bpy.context.view_layer.update()
        if (is_visible(ob, need) and bbox_reasonable(ob)
                and not any(footprints_overlap(ob, o.obj, gap) for o in placed)):
            return True
    return False


# ============================================================
# STACKING
# ============================================================
TYPES = list(TYPE_WEIGHTS.keys())
WEIGHTS = [TYPE_WEIGHTS[t] for t in TYPES]

# Surface presets used in the frame being built, so the floor brightness can
# be picked against them (see the main loop).
FRAME_PRESETS = []
PALE_PRESETS = ("white_board", "bleached", "foam", "paper")


def parcel_surface(ptype, slot, family, fam_preset):
    """Pick + randomize the material for one parcel."""
    if ptype == "poly_bag":
        preset = "plastic"
    elif ptype == "sack":
        preset = "fabric"
    elif ptype in ("crate", "tray_box"):
        # timber most of the time, otherwise heavy board
        preset = random.choices(["wood", "kraft_dark", "kraft"],
                                [64, 20, 16])[0]
    elif ptype == "envelope":
        preset = random.choices(["white_board", "paper", "bleached", "kraft",
                                 "plastic"], [30, 24, 18, 18, 10])[0]
    elif ptype == "tub":
        preset = random.choices(["plastic", "white_board", "foil", "kraft"],
                                [46, 24, 16, 14])[0]
    else:
        preset = fam_preset if family else random.choice(SURFACE_PRESETS)
    FRAME_PRESETS.append(preset)
    return randomize_parcel_material(
        PARCEL_MATS[slot % len(PARCEL_MATS)], preset, ptype in ROUND_TYPES)


def decorate(parcel, slot, family, fam_tape_kind):
    """Tape / shrink wrap / straps / shipping label / Data Matrix.

    Exactly one Data Matrix goes on every package's top face; the shipping
    label is then placed clear of it. Returns the objects created.
    """
    objs = []
    soft = parcel.type in SOFT_TYPES
    # An open crate or a bucket is not taped shut, and a roll on its side is
    # usually only banded - so the tape chance is scaled per type.
    tape_scale = 0.45 if soft else 1.0
    if parcel.type in ("tray_box", "tub"):
        tape_scale = 0.22
    elif parcel.type in LYING_TYPES:
        tape_scale = 0.65

    if random.random() < TAPE_PROB * tape_scale:
        t = build_tape(parcel, randomize_tape_material(
            TAPE_MATS[slot % len(TAPE_MATS)],
            fam_tape_kind if family else None))
        if t:
            objs.append(t)
    if not soft and random.random() < WRAP_FILM_PROB:
        f = build_tape(parcel, randomize_tape_material(
            TAPE_MATS[(slot + 1) % len(TAPE_MATS)], "clear"), style="film")
        if f:
            objs.append(f)
    if not soft and random.random() < STRAP_PROB:
        s = build_strap(parcel, randomize_strap_material(
            STRAP_MATS[slot % len(STRAP_MATS)]))
        if s:
            objs.append(s)

    keep_out = None
    if random.random() < DMC_PROB:
        ob, info = build_dmc(parcel, DMC_MATS[slot % len(DMC_MATS)],
                             f"dmc_{slot:02d}")
        if ob is not None:
            objs.append(ob)
            parcel.dmc = info
            cx = sum(c[0] for c in info["corners_local"]) * 0.25
            cy = sum(c[1] for c in info["corners_local"]) * 0.25
            keep_out = (cx, cy, info["size_m"] * 0.75)

    if random.random() < LABEL_PROB:
        lb = build_label(parcel, randomize_label_material(
            LABEL_MATS[slot % len(LABEL_MATS)]), avoid=keep_out)
        if lb:
            objs.append(lb)
    return objs


def stack_on(support, slot, family, fam_preset, fam_tape_kind, want=None):
    """Place several smaller packages on the support's top face.

    Positions are sampled in the support's own yaw-rotated frame with each
    child's half-diagonal subtracted from the half-extents, so a child stays
    on its support whatever its own yaw is. Children are kept apart with the
    same SAT footprint test used on the floor. Asking for N children yields N
    or fewer - whatever actually fits.

    Returns (parcels, objects).
    """
    want = want or random.randint(1, STACK_MAX_ON_TOP)
    yaw = support.obj.rotation_euler[2]
    ca, sa = math.cos(yaw), math.sin(yaw)
    if support.radius is not None:
        # keep a stacked parcel inside the round lid, not hanging off it
        hx = hy = support.radius * 0.72
    else:
        hx, hy = support.top_w * 0.5, support.top_d * 0.5
    z_top = max(c.z for c in bbox_world(support.obj))

    kids, objs = [], []
    for _ in range(want):
        seated = False
        for _attempt in range(5):
            ptype = random.choices(TYPES, WEIGHTS)[0]
            pmat = parcel_surface(ptype, slot, family, fam_preset)
            scale = random.uniform(*STACK_SIZE_SCALE)
            if want > 1:
                scale *= 0.82 ** (want - 1)     # more on top -> smaller each
            # don't shrink into doll-house parcels: below this the top face is
            # too small to carry a Data Matrix at all
            scale = max(scale, STACK_MIN_SCALE)
            top = build_parcel(ptype, pmat, size_scale=scale)

            hd = 0.5 * math.hypot(top.w, top.d)
            rx, ry = hx - hd, hy - hd
            if rx <= 0.0 or ry <= 0.0:
                discard(top.obj)
                continue

            top.obj.rotation_euler = (0.0, 0.0, random.uniform(0, math.tau))
            spot = False
            for _try in range(24):
                lx = random.uniform(-rx, rx)
                ly = random.uniform(-ry, ry)
                top.obj.location.x = support.obj.location.x + lx * ca - ly * sa
                top.obj.location.y = support.obj.location.y + lx * sa + ly * ca
                bpy.context.view_layer.update()
                if any(footprints_overlap(top.obj, k.obj, 0.004) for k in kids):
                    continue
                spot = True
                break
            if not spot:
                discard(top.obj)
                continue

            rest_on(top.obj, z_top - 0.0008)
            if not (is_visible(top.obj) and bbox_reasonable(top.obj)):
                discard(top.obj)
                continue

            kids.append(top)
            objs.append(top.obj)
            objs.extend(decorate(top, slot, family, fam_tape_kind))
            slot += 1
            seated = True
            break
        if not seated:
            break               # no room left on this support
    return kids, objs


# ============================================================
# MAIN LOOP
# ============================================================

print(f"rendering {NUM_IMAGES} images -> {GRAY_DIR}")

for i in range(NUM_IMAGES):
    idx = START_INDEX + i
    random.seed(SEED * 7919 + idx)          # reproducible / resumable per frame

    # ---- camera ---------------------------------------------------------
    camera.location = (random.uniform(-CAM_SHIFT, CAM_SHIFT),
                       random.uniform(-CAM_SHIFT, CAM_SHIFT),
                       random.uniform(CAM_Z_MIN, CAM_Z_MAX))
    camera.rotation_euler = (random.uniform(-CAM_TILT_MAX, CAM_TILT_MAX),
                             random.uniform(-CAM_TILT_MAX, CAM_TILT_MAX),
                             random.uniform(0, math.tau))
    bpy.context.view_layer.update()
    bounds = view_bounds()
    if bounds is None:
        bounds = (-0.4, 0.4, -0.4, 0.4)

    # ---- parcels --------------------------------------------------------
    FRAME_PRESETS.clear()
    spawned = []          # every generated object, for cleanup
    placed = []           # Parcel instances actually in frame
    gap = random.uniform(GAP_MIN, GAP_MAX)
    n_target = random.randint(MIN_PARCELS, MAX_PARCELS)

    # a whole image is often a single parcel "family" (same tape, same look)
    family = random.random() < 0.4
    fam_preset = random.choice(SURFACE_PRESETS)
    fam_tape_kind = random.choice(["clear", "brown", "brown", "white", "printed"])

    for k in range(n_target):
        ptype = random.choices(TYPES, WEIGHTS)[0]

        # A "family" image shares the look (preset + tape kind), but every
        # parcel still gets its own material datablock so the texture crop
        # differs - otherwise all parcels come out pixel-identical.
        pmat = parcel_surface(ptype, k, family, fam_preset)

        parcel = build_parcel(ptype, pmat)
        parcel.obj.rotation_euler = (0, 0, 0)
        rest_on(parcel.obj, FLOOR_Z)

        if not place(parcel, placed, bounds, gap):
            # place() leaves the object at its last attempted spot, which is
            # inside the view. Keeping it would render an unlabelled parcel -
            # a false negative the detector gets punished for.
            discard(parcel.obj)
            continue

        spawned.append(parcel.obj)
        spawned.extend(decorate(parcel, k, family, fam_tape_kind))
        placed.append(parcel)

    # ---- stacking: smaller packages on top of bigger ones ----------------
    if placed and random.random() < STACK_PROB:
        # only rigid, reasonably wide, flat-topped parcels make believable
        # supports - nothing gets stacked on a gabled or coned lid
        supports = [p for p in placed
                    if p.type not in SOFT_TYPES and p.top_flat]
        supports.sort(key=lambda p: -(p.w * p.d))
        supports = supports[:max(1, len(supports) // 2 + 1)]
        random.shuffle(supports)

        slot = n_target
        for support in supports[:random.randint(1, STACK_MAX_TOWERS)]:
            kids, objs = stack_on(support, slot, family, fam_preset,
                                  fam_tape_kind)
            slot += max(1, len(kids))
            if not kids:
                continue
            spawned.extend(objs)
            placed.extend(kids)

            # a third level on one of the parcels we just stacked
            if random.random() < STACK_SECOND_LEVEL_PROB:
                kids2, objs2 = stack_on(random.choice(kids), slot, family,
                                        fam_preset, fam_tape_kind,
                                        want=random.randint(1, 2))
                slot += max(1, len(kids2))
                spawned.extend(objs2)
                placed.extend(kids2)

    # ---- floor ----------------------------------------------------------
    # Chosen after the parcels so its brightness can be set against them: a
    # frame full of white mailers gets a floor from the dark end of the range,
    # which stops "pale on pale" contrast-free images.
    pale = sum(p in PALE_PRESETS for p in FRAME_PRESETS)
    pale_share = pale / max(len(FRAME_PRESETS), 1)
    if pale_share > 0.6:
        FLOOR_VALUE_RANGE[:] = [0.35, 1.00]
    elif pale_share < 0.2:
        FLOOR_VALUE_RANGE[:] = [0.55, 1.70]
    else:
        FLOOR_VALUE_RANGE[:] = [0.45, 1.70]
    fmat, frnd = random.choice(FLOORS)
    floor.data.materials.clear()
    floor.data.materials.append(fmat)
    frnd()

    # ---- lights ---------------------------------------------------------
    if placed:
        cx = sum(p.obj.location.x for p in placed) / len(placed)
        cy = sum(p.obj.location.y for p in placed) / len(placed)
    else:
        cx = cy = 0.0
    scen, gobo_on = randomize_lights(Vector((cx, cy, FLOOR_Z + 0.08)))

    bpy.context.view_layer.update()

    # ---- render ---------------------------------------------------------
    gray_path = str(GRAY_DIR / f"img_{idx:04d}.png")
    scene.render.filepath = gray_path
    bpy.ops.render.render(write_still=True)

    # ---- labels ---------------------------------------------------------
    cam_inv = camera.matrix_world.inverted()
    parcels_json = []
    for p in placed:
        corners_cam = [cam_inv @ c for c in bbox_world(p.obj)]
        entry = {
            "name": p.obj.name,
            "type": p.type,
            "shape": p.shape,          # lid style, sides, taper, ...
            "location": list(p.obj.location),
            "rotation": list(p.obj.rotation_euler),
            "dimensions": list(p.obj.dimensions),
            "bbox_3d_camera": [[c.x, c.y, c.z] for c in corners_cam],
        }
        if p.dmc is not None:
            m = p.obj.matrix_world
            world = [m @ Vector(c) for c in p.dmc["corners_local"]]
            cam = [cam_inv @ c for c in world]
            entry["dmc"] = {
                "text": p.dmc["text"],
                "symbol_modules": p.dmc["modules"],
                "size_m": p.dmc["size_m"],
                "bent": p.dmc["bent"],
                "corners_3d_camera": [[c.x, c.y, c.z] for c in cam],
                # 2D corners, None where the corner falls outside the frame
                "corners_2d": [project(c) for c in world],
            }
        parcels_json.append(entry)

    meta = {
        "num_parcels": len(placed),
        "camera_intrinsics": {
            "fx": FX, "fy": FX,
            "cx": RES * 0.5, "cy": RES * 0.5,
            "width": RES, "height": RES,
        },
        "camera_location": list(camera.location),
        "camera_rotation": list(camera.rotation_euler),
        "floor_material": fmat.name,
        "lighting": {
            "scenario": scen,
            "gobo": bool(gobo_on),
            "exposure": scene.view_settings.exposure,
            "view_transform": scene.view_settings.view_transform,
        },
        "parcels": parcels_json,
    }
    with open(LABEL_DIR / f"img_{idx:04d}.json", "w") as f:
        json.dump(meta, f, indent=4)

    # ---- cleanup --------------------------------------------------------
    for ob in spawned:
        discard(ob)
    for p in placed:            # the per-parcel Data Matrix bitmaps
        if p.dmc is not None:
            try:
                bpy.data.images.remove(p.dmc["image"])
            except Exception:
                pass

    print(f"Rendered {idx}  ({i + 1}/{NUM_IMAGES}, {len(placed)} parcels, "
          f"floor={fmat.name}, light={scen}{'+gobo' if gobo_on else ''})")

    if (i + 1) % 50 == 0:
        bpy.data.orphans_purge(do_local_ids=True, do_linked_ids=True,
                               do_recursive=True)
        # Drop decoded pixel buffers of the ~200 loaded textures; Blender
        # re-reads them from disk on demand. Keeps RSS flat over a long run.
        for _im in bpy.data.images:
            if _im.filepath:
                try:
                    _im.buffers_free()
                except Exception:
                    pass

print("DONE")
