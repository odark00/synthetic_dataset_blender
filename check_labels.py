"""Sanity-check a rendered dataset: draw the labelled 3D boxes over the images.

    python3 check_labels.py [gray_dir] [labels_dir] [out_dir] [n]

Uses the same projection as convert_to_moca3d.py, so if the wireframes line up
with the parcels the labels are consistent with the training converter.
"""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
gray = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "gray_images"
lab = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "labels"
out = Path(sys.argv[3]) if len(sys.argv) > 3 else HERE / "_label_check"
n = int(sys.argv[4]) if len(sys.argv) > 4 else 8
out.mkdir(parents=True, exist_ok=True)

# bound_box corner order -> the 12 edges of the box
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
         (0, 4), (1, 5), (2, 6), (3, 7)]

for jf in sorted(lab.glob("img_*.json"))[:n]:
    meta = json.loads(jf.read_text())
    K = meta["camera_intrinsics"]
    img_f = gray / (jf.stem + ".png")
    if not img_f.exists():
        continue
    im = Image.open(img_f).convert("RGB")
    dr = ImageDraw.Draw(im)
    for p in meta["parcels"]:
        pts = []
        for X, Y, Z in p["bbox_3d_camera"]:
            if Z >= 0:
                pts.append(None)
                continue
            pts.append((K["fx"] * X / -Z + K["cx"], K["fy"] * -Y / -Z + K["cy"]))
        for a, b in EDGES:
            if pts[a] and pts[b]:
                dr.line([pts[a], pts[b]], fill=(255, 40, 40), width=2)

        # Data Matrix quad, from the same camera-space corners
        dmc = p.get("dmc")
        if dmc:
            q = []
            for X, Y, Z in dmc["corners_3d_camera"]:
                if Z >= 0:
                    q = []
                    break
                q.append((K["fx"] * X / -Z + K["cx"], K["fy"] * -Y / -Z + K["cy"]))
            if len(q) == 4:
                dr.line(q + [q[0]], fill=(40, 220, 90), width=2)
    im.save(out / img_f.name)
    print("wrote", out / img_f.name)
