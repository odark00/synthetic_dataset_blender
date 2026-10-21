"""Cycles renderer for generate_v3.py -- runs INSIDE Blender, started by generate_v3.py:

    blender -b --factory-startup -P synth/blender_render.py -- --work DIR --worker k --nworkers K

Reads DIR/todo.txt (one image index per line) and renders every k-th entry (k = worker). For each index it
waits for DIR/{i:08d}.npz (scene written by generate_v3.py; DIR/{i:08d}.skip = nothing to render), renders
it and writes DIR/{i:08d}.render.png:
    16-bit RGBA, straight alpha, linear radiance * 0.5 (a white horizontal surface under the nominal light
    = 0.5); alpha = objects + shadow-catcher shadows on the floor, i.e. final = rgb * a + background * (1 - a)

Blender frame: X = camera x (image right), Y = -camera y (image up), Z = height above the floor; the camera
sits at (0, 0, mounting height) looking straight down.
"""
import argparse
import os
import sys
import time

import bpy
import numpy as np
from mathutils import Vector

EXPOSURE = 0.5  # stored value = radiance * EXPOSURE (headroom for highlights above white)


def parse():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    ap.add_argument("--device", default="OPTIX", help="OPTIX | CUDA | CPU")
    return ap.parse_args(argv)


# ---------------------------------------------------------------------------------- setup
def setup_render(device, W, H):
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    if device != "CPU":
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = device
        prefs.refresh_devices()
        for d in prefs.devices:
            d.use = d.type == device
        sc.cycles.device = "GPU"
    c = sc.cycles
    c.samples = 48
    c.use_adaptive_sampling = True
    c.adaptive_threshold = 0.02
    c.use_denoising = True
    c.denoiser = "OPENIMAGEDENOISE"
    c.denoising_use_gpu = device != "CPU"
    c.denoising_input_passes = "RGB_ALBEDO_NORMAL"
    c.denoising_prefilter = "ACCURATE"
    c.max_bounces, c.diffuse_bounces, c.glossy_bounces = 8, 3, 3
    c.transmission_bounces, c.transparent_max_bounces = 4, 24
    c.caustics_reflective = c.caustics_refractive = False
    c.filter_width = 1.2
    c.seed = 0
    sc.render.use_persistent_data = True
    sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = W, H, 100
    sc.render.film_transparent = True
    sc.render.dither_intensity = 0.0
    sc.view_settings.view_transform = "Raw"
    sc.view_settings.look = "None"
    sc.view_settings.exposure = float(np.log2(EXPOSURE))
    sc.view_settings.gamma = 1.0
    im = sc.render.image_settings
    im.file_format, im.color_mode, im.color_depth, im.compression = "PNG", "RGBA", "16", 15
    sc.render.use_file_extension = False


def setup_camera(fx, fy, cx, cy, W, H, mount_h):
    """Pinhole camera with the OpenCV intrinsics (pixel centres at integer coordinates)."""
    sc = bpy.context.scene
    cam = bpy.data.cameras.new("cam")
    ob = bpy.data.objects.new("cam", cam)
    sc.collection.objects.link(ob)
    sc.camera = ob
    ob.location = (0.0, 0.0, mount_h)
    cam.sensor_fit = "HORIZONTAL"
    cam.sensor_width = 36.0
    cam.lens = fx * cam.sensor_width / W
    # Blender keeps the smaller aspect at 1: fy > fx -> wider pixels (checked against Camera.project: < 0.1 px)
    sc.render.pixel_aspect_x, sc.render.pixel_aspect_y = (fy / fx, 1.0) if fy >= fx else (1.0, fx / fy)
    cam.shift_x = (W / 2 - (cx + 0.5)) / max(W, H)
    cam.shift_y = ((cy + 0.5) - H / 2) / max(W, H)
    cam.clip_start, cam.clip_end = 0.05, 50.0


def image_from_array(name, a, is_float=False):
    """(h, w) or (h, w, c<=4) array in [0, 1], row 0 = top -> Blender image (non-colour data, channels packed)."""
    a = np.asarray(a, np.float32)
    if a.ndim == 2:
        a = a[..., None]
    h, w, c = a.shape
    px = np.ones((h, w, 4), np.float32)
    px[..., :min(c, 4)] = a[..., :4] if c > 1 else a
    img = bpy.data.images.new(name, w, h, alpha=True, float_buffer=is_float, is_data=True)
    img.colorspace_settings.is_data = True
    img.alpha_mode = "CHANNEL_PACKED"
    img.pixels.foreach_set(px[::-1].ravel())
    img.update()
    return img


def new_material(name):
    mat = bpy.data.materials.new(name)
    if mat.node_tree is None:  # Blender < 5: node trees are opt-in
        mat.use_nodes = True
    return mat


def node(nt, kind, loc=(0, 0), **inputs):
    n = nt.nodes.new(kind)
    n.location = loc
    for k, v in inputs.items():  # by identifier: name lookup misses sockets hidden by the node's current mode
        next(i for i in n.inputs if i.identifier == k).default_value = v
    return n


def setup_floor(bg_path, cam_params):
    """Shadow-catcher floor carrying the real empty-floor image (for light bounced onto the objects)."""
    fx, fy, cx, cy, W, H, mount_h = cam_params
    bg = np.load(bg_path).astype(np.float32)  # floor albedo, same scale as the object textures
    imgs = {}
    for code in (-2, -1, 0, 1):  # -2 = no flip, else cv2.flip code
        a = bg if code == -2 else (bg[::-1, ::-1] if code == -1 else bg[::-1] if code == 0 else bg[:, ::-1])
        imgs[code] = image_from_array(f"floor_{code}", np.clip(a, 0, 1.2), is_float=True)
    R = 40.0
    xy = np.array([[-R, -R], [R, -R], [R, R], [-R, R]])
    u = (fx * xy[:, 0] / mount_h + cx + 0.5) / W          # Blender y = -camera y
    v = 1.0 - (fy * -xy[:, 1] / mount_h + cy + 0.5) / H
    me = bpy.data.meshes.new("floor")
    me.from_pydata([(x, y, 0.0) for x, y in xy], [], [(0, 1, 2, 3)])
    uv = me.uv_layers.new()
    uv.data.foreach_set("uv", np.c_[u, v].astype(np.float32).ravel())
    ob = bpy.data.objects.new("floor", me)
    bpy.context.scene.collection.objects.link(ob)
    ob.is_shadow_catcher = True
    mat = new_material("floor")
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = 0.9
    tex = node(nt, "ShaderNodeTexImage", (-400, 0))
    tex.extension = "EXTEND"
    tex.image = imgs[-2]
    nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    me.materials.append(mat)
    return tex, imgs


def solid_material():
    """Atlas RGBA = (albedo, roughness, metallic, unevenness); rounded edges (bevel) + bulging faces (bump)."""
    mat = new_material("solid")
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    tex = node(nt, "ShaderNodeTexImage", (-700, 0))
    tex.interpolation, tex.extension = "Linear", "EXTEND"
    sep = node(nt, "ShaderNodeSeparateColor", (-400, 0))
    bev = node(nt, "ShaderNodeBevel", (-400, -300), Radius=0.006)
    bev.samples = 4
    nt.links.new(tex.outputs["Color"], sep.inputs["Color"])
    nt.links.new(sep.outputs["Red"], bsdf.inputs["Base Color"])
    nt.links.new(sep.outputs["Green"], bsdf.inputs["Roughness"])
    nt.links.new(sep.outputs["Blue"], bsdf.inputs["Metallic"])
    tc = node(nt, "ShaderNodeTexCoord", (-900, -500))
    wav = node(nt, "ShaderNodeTexNoise", (-650, -500), Scale=3.0, Detail=3.0, Roughness=0.55)
    amt = node(nt, "ShaderNodeMath", (-400, -650), Value_001=0.8)
    amt.operation = "MULTIPLY"
    bump = node(nt, "ShaderNodeBump", (-200, -300), Strength=0.1, Distance=0.06)
    nt.links.new(tc.outputs["Object"], wav.inputs["Vector"])
    nt.links.new(tex.outputs["Alpha"], amt.inputs[0])
    nt.links.new(amt.outputs["Value"], bump.inputs["Strength"])
    nt.links.new(wav.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bev.outputs["Normal"], bump.inputs["Normal"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return mat, tex


def film_material():
    """Stretch film: atlas RGB = (milky haze / opacity, albedo, gloss amount): clear film is transparent with milky
    crinkles, opaque black / white film is diffuse; all with a crinkled glossy reflection."""
    mat = new_material("film")
    nt = mat.node_tree
    for n in list(nt.nodes):
        if n.type == "BSDF_PRINCIPLED":
            nt.nodes.remove(n)
    out = nt.nodes["Material Output"]
    tex = node(nt, "ShaderNodeTexImage", (-900, 300))
    tex.interpolation, tex.extension = "Linear", "EXTEND"
    sep = node(nt, "ShaderNodeSeparateColor", (-650, 300))
    haze = node(nt, "ShaderNodeMath", (-450, 300), Value_001=1.0, Value_002=0.0)
    haze.operation, haze.use_clamp = "MULTIPLY_ADD", True
    gmask = node(nt, "ShaderNodeMath", (-50, -150))
    gmask.operation = "MULTIPLY"
    tc = node(nt, "ShaderNodeTexCoord", (-1300, -200))
    mp = node(nt, "ShaderNodeMapping", (-1100, -200))
    mp.inputs["Scale"].default_value = (1.0, 1.0, 4.0)  # wrap runs horizontally: crinkles stretched along it
    noise = node(nt, "ShaderNodeTexNoise", (-900, -200), Scale=40.0, Detail=10.0, Roughness=0.7, Distortion=1.0)
    wave = node(nt, "ShaderNodeTexWave", (-900, -450), Scale=30.0, Distortion=6.0, Detail=4.0)
    wave.wave_type, wave.bands_direction = "BANDS", "Z"  # overlapping wraps around the load
    height = node(nt, "ShaderNodeMath", (-700, -300), Value_001=0.3)
    height.operation = "MULTIPLY_ADD"
    # crinkles come in patches: a low-frequency mask switches them on over part of the film
    patch = node(nt, "ShaderNodeTexNoise", (-1100, -650), Scale=3.0, Detail=2.0)
    ramp = node(nt, "ShaderNodeMapRange", (-900, -800), **{"From Min": 0.45, "From Max": 0.6})
    masked = node(nt, "ShaderNodeMath", (-500, -400))
    masked.operation = "MULTIPLY"
    bump = node(nt, "ShaderNodeBump", (-650, -200), Strength=0.5, Distance=0.008)
    fold = node(nt, "ShaderNodeBevel", (-900, -650), Radius=0.04)  # film rounds over the load edges
    fold.samples = 4
    fres = node(nt, "ShaderNodeFresnel", (-400, -50), IOR=1.8)
    gloss_amt = node(nt, "ShaderNodeMath", (-200, -50), Value_001=1.0)
    gloss_amt.operation, gloss_amt.use_clamp = "MULTIPLY", True
    transp = node(nt, "ShaderNodeBsdfTransparent", (-400, 150))
    diff = node(nt, "ShaderNodeBsdfDiffuse", (-400, 50), Color=(0.85, 0.85, 0.85, 1.0))
    mix1 = node(nt, "ShaderNodeMixShader", (-150, 200))
    gloss = node(nt, "ShaderNodeBsdfGlossy", (-200, -250), Roughness=0.12)
    mix2 = node(nt, "ShaderNodeMixShader", (100, 100))
    L = nt.links.new
    L(tex.outputs["Color"], sep.inputs["Color"])
    L(sep.outputs["Red"], haze.inputs[0])
    L(tc.outputs["Object"], mp.inputs["Vector"])
    L(mp.outputs["Vector"], noise.inputs["Vector"])
    L(mp.outputs["Vector"], wave.inputs["Vector"])
    L(wave.outputs["Fac"], height.inputs[0])
    L(noise.outputs["Fac"], height.inputs[2])
    L(tc.outputs["Object"], patch.inputs["Vector"])
    L(patch.outputs["Fac"], ramp.inputs["Value"])
    L(height.outputs["Value"], masked.inputs[0])
    L(ramp.outputs["Result"], masked.inputs[1])
    L(masked.outputs["Value"], bump.inputs["Height"])
    L(fold.outputs["Normal"], bump.inputs["Normal"])
    L(bump.outputs["Normal"], fres.inputs["Normal"])
    L(bump.outputs["Normal"], gloss.inputs["Normal"])
    L(fres.outputs["Fac"], gloss_amt.inputs[0])
    L(haze.outputs["Value"], mix1.inputs["Fac"])
    L(sep.outputs["Green"], diff.inputs["Color"])
    L(transp.outputs["BSDF"], mix1.inputs[1])
    L(diff.outputs["BSDF"], mix1.inputs[2])
    L(gloss_amt.outputs["Value"], gmask.inputs[0])
    L(sep.outputs["Blue"], gmask.inputs[1])
    L(gmask.outputs["Value"], mix2.inputs["Fac"])
    L(mix1.outputs["Shader"], mix2.inputs[1])
    L(gloss.outputs["BSDF"], mix2.inputs[2])
    L(mix2.outputs["Shader"], out.inputs["Surface"])
    return mat, {"tex": tex, "haze": haze, "noise": noise, "wave": wave, "bump": bump, "gloss": gloss,
                 "gloss_amt": gloss_amt, "fold": fold, "patch": ramp}


def mesh_material():
    """Cage wire mesh: atlas RG = (albedo, coverage); galvanised steel where covered."""
    mat = new_material("mesh")
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Metallic"].default_value = 0.7
    bsdf.inputs["Roughness"].default_value = 0.45
    out = nt.nodes["Material Output"]
    tex = node(nt, "ShaderNodeTexImage", (-700, 0))
    tex.interpolation, tex.extension = "Linear", "EXTEND"
    sep = node(nt, "ShaderNodeSeparateColor", (-450, 0))
    transp = node(nt, "ShaderNodeBsdfTransparent", (-200, 200))
    mix = node(nt, "ShaderNodeMixShader", (100, 100))
    nt.links.new(tex.outputs["Color"], sep.inputs["Color"])
    nt.links.new(sep.outputs["Red"], bsdf.inputs["Base Color"])
    nt.links.new(sep.outputs["Green"], mix.inputs["Fac"])
    nt.links.new(transp.outputs["BSDF"], mix.inputs[1])
    nt.links.new(bsdf.outputs["BSDF"], mix.inputs[2])
    nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])
    return mat, tex


# ---------------------------------------------------------------------------------- per scene
class Renderer:
    def __init__(self, args, first):
        self.args = args
        cam = first["cam"]
        fx, fy, cx, cy, W, H, mh = [float(v) for v in cam]
        for o in list(bpy.data.objects):
            bpy.data.objects.remove(o)
        setup_render(args.device, int(W), int(H))
        setup_camera(fx, fy, cx, cy, int(W), int(H), mh)
        self.floor_tex, self.floor_imgs = setup_floor(os.path.join(args.work, "floor_albedo.npy"),
                                                     (fx, fy, cx, cy, W, H, mh))
        self.mats = {"solid": solid_material(), "film": film_material(), "mesh": mesh_material()}
        world = bpy.data.worlds.new("world")
        world.use_nodes = True
        bpy.context.scene.world = world
        self.world_bg = world.node_tree.nodes["Background"]
        self.world_bg.inputs["Color"].default_value = (1, 1, 1, 1)
        self.objects, self.images, self.lights = [], [], []

    def clear(self):
        for o in self.objects:
            me = o.data
            bpy.data.objects.remove(o)
            if isinstance(me, bpy.types.Mesh):
                bpy.data.meshes.remove(me)
            elif isinstance(me, bpy.types.Light):
                bpy.data.lights.remove(me)
        for im in self.images:
            bpy.data.images.remove(im)
        self.objects, self.images = [], []

    def add_mesh(self, name, V, sizes, UV, N, mat):
        """V: vertices of all polygons back to back (no sharing), sizes: vertex count per polygon,
        N: normal per polygon corner (flat faces: the face normal; smooth surfaces: interpolated)."""
        me = bpy.data.meshes.new(name)
        me.vertices.add(len(V))
        me.vertices.foreach_set("co", V.astype(np.float32).ravel())
        me.loops.add(len(V))
        me.loops.foreach_set("vertex_index", np.arange(len(V), dtype=np.int32))
        me.polygons.add(len(sizes))
        me.polygons.foreach_set("loop_start", np.r_[0, np.cumsum(sizes)[:-1]].astype(np.int32))
        me.polygons.foreach_set("loop_total", sizes.astype(np.int32))
        me.update(calc_edges=True)
        uv = me.uv_layers.new()
        uv.data.foreach_set("uv", UV.astype(np.float32).ravel())
        N = N.astype(np.float32)
        try:  # Blender >= 5.0: free custom normals as a corner attribute (fast)
            me.attributes.new("custom_normal", "FLOAT_VECTOR", "CORNER").data.foreach_set("vector", N.ravel())
        except (RuntimeError, TypeError):
            me.shade_smooth()
            me.normals_split_custom_set(N)
        me.materials.append(mat)
        me.update()
        ob = bpy.data.objects.new(name, me)
        bpy.context.scene.collection.objects.link(ob)
        self.objects.append(ob)

    def add_light(self, x, y, z, sx, sy, power, tx, ty, tz):
        L = bpy.data.lights.new("lamp", "AREA")
        L.shape, L.size, L.size_y, L.energy = "RECTANGLE", float(sx), float(sy), float(power)
        ob = bpy.data.objects.new("lamp", L)
        ob.location = (x, y, z)
        d = Vector((tx - x, ty - y, tz - z))
        ob.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
        bpy.context.scene.collection.objects.link(ob)
        self.objects.append(ob)

    def build(self, s):
        self.clear()
        sc = bpy.context.scene
        sc.cycles.samples = int(s["samples"])
        sc.cycles.seed = int(s["seed"]) % (2 ** 31)
        self.world_bg.inputs["Strength"].default_value = float(s["world"])
        self.floor_tex.image = self.floor_imgs[int(s["flip"])]
        for row in s["lights"]:
            self.add_light(*row)
        for g in ("solid", "film", "mesh"):
            if f"{g}_v" not in s or len(s[f"{g}_fs"]) == 0:
                continue
            mat, extra = self.mats[g]
            img = image_from_array(f"{g}_atlas", s[f"{g}_atlas"].astype(np.float32) / 255.0)
            self.images.append(img)
            tex = extra["tex"] if isinstance(extra, dict) else extra
            tex.image = img
            if g == "film":
                # noise scale, bump strength, gloss roughness, gloss gain, wrap bands, fold radius, crinkle patch
                # threshold (haze, albedo and gloss amount come per surface from the atlas)
                fp = s["film_params"]
                extra["noise"].inputs["Scale"].default_value = float(fp[0])
                extra["bump"].inputs["Strength"].default_value = float(fp[1])
                extra["gloss"].inputs["Roughness"].default_value = float(fp[2])
                extra["gloss_amt"].inputs[1].default_value = float(fp[3])
                extra["wave"].inputs["Scale"].default_value = float(fp[4])
                extra["fold"].inputs["Radius"].default_value = float(fp[5])
                extra["patch"].inputs["From Min"].default_value = float(fp[6])  # higher = fewer crinkled patches
                extra["patch"].inputs["From Max"].default_value = float(fp[6]) + 0.12
            self.add_mesh(g, s[f"{g}_v"], s[f"{g}_fs"], s[f"{g}_uv"], s[f"{g}_n"], mat)

    def render(self, out):
        sc = bpy.context.scene
        tmp = out + ".tmp.png"
        sc.render.filepath = tmp
        bpy.ops.render.render(write_still=True)
        os.replace(tmp, out)


def main():
    args = parse()
    todo = [int(l) for l in open(os.path.join(args.work, "todo.txt")) if l.strip()]
    mine = todo[args.worker::args.nworkers]
    parent = os.getppid()
    R = None
    t0, done = time.time(), 0
    for i in mine:
        npz = os.path.join(args.work, f"{i:08d}.npz")
        skip = os.path.join(args.work, f"{i:08d}.skip")
        out = os.path.join(args.work, f"{i:08d}.render.png")
        while not (os.path.exists(npz) or os.path.exists(skip)):
            if os.path.exists(os.path.join(args.work, "STOP")) or os.getppid() != parent:
                return
            time.sleep(0.05)
        if not os.path.exists(npz):
            continue
        s = dict(np.load(npz))
        if R is None:
            R = Renderer(args, s)
        R.build(s)
        R.render(out)
        os.remove(npz)
        done += 1
        if done % 10 == 0:
            print(f"[blender {args.worker}] {done}/{len(mine)}  {(time.time() - t0) / done:.2f} s/img", flush=True)


if __name__ == "__main__":
    main()
