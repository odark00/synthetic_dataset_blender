"""
Convert per-image label JSONs into MoCA3D image-dataloader format.

Input labels (dataset/labels/img_XXXX.json):
  - camera_intrinsics: fx, fy, cx, cy, width, height
  - parcels[].bbox_3d_camera: 8 x [X, Y, Z] corners in camera space

Output: datasets/MoCA3D/CustomDataset_{split}.json
  - images[]: id, file_path, K, width, height
  - annotations[]: image_id, bbox2D_tight, projected_corners, depth, bbox3D_cam, quality, ...
"""

import json
import math
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABELS_DIR = PROJECT_ROOT / "dataset" / "labels"
JSON_OUT_DIR = PROJECT_ROOT / "datasets" / "MoCA3D"
# file_path is relative to data_root = datasets/
IMAGE_PATH_PREFIX = "CustomDataset/gray_images"

TRAIN_FRAC = 0.85
SEED = 42


def project_3d_to_2d(corners_3d, fx, fy, cx, cy):
    """Project 8 camera-space corners to image plane.

    Blender convention: camera looks along -Z, so Z < 0 for objects in front.
    Matches the project() function in dataset/script.py:
      u = fx * X / Z + cx
      v = fy * Y / (-Z) + cy
    """
    projected = []
    for X, Y, Z in corners_3d:
        u = fx * X / -Z + cx
        v = fy * -Y / (-Z) + cy
        projected.append((u, v))
    return projected


def bbox2d_from_corners(projected_corners, img_w, img_h):
    """Compute tight 2D bbox [x1, y1, x2, y2] from projected corners."""
    us = [p[0] for p in projected_corners]
    vs = [p[1] for p in projected_corners]
    x1 = max(0.0, min(us))
    y1 = max(0.0, min(vs))
    x2 = min(float(img_w), max(us))
    y2 = min(float(img_h), max(vs))
    return [x1, y1, x2, y2]


def compute_dimensions(corners_3d):
    """Estimate box dimensions from 8 corners as axis-aligned extent in cam space."""
    xs = [c[0] for c in corners_3d]
    ys = [c[1] for c in corners_3d]
    zs = [c[2] for c in corners_3d]
    return [max(ys) - min(ys), max(xs) - min(xs), max(zs) - min(zs)]  # h, w, d


def build_split_json(label_paths, split_name):
    images = []
    annotations = []
    ann_id = 0

    for img_id, label_path in enumerate(label_paths):
        label = json.loads(label_path.read_text())
        intr = label["camera_intrinsics"]
        fx, fy = intr["fx"], intr["fy"]
        cx, cy = intr["cx"], intr["cy"]
        w, h = intr["width"], intr["height"]
        K = [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]

        stem = label_path.stem  # e.g. img_0000
        file_path = f"{IMAGE_PATH_PREFIX}/{stem}.png"

        images.append({"id": img_id, "file_path": file_path, "K": K, "width": w, "height": h})

        for parcel in label.get("parcels", []):
            corners_3d = parcel["bbox_3d_camera"]  # list of 8 [X, Y, Z]

            # Blender: objects in front have Z < 0; depth = -Z
            zs = [c[2] for c in corners_3d]
            if any(z >= 0 for z in zs):
                continue

            projected = project_3d_to_2d(corners_3d, fx, fy, cx, cy)
            bbox2d = bbox2d_from_corners(projected, w, h)

            # Skip if box is degenerate
            if bbox2d[2] <= bbox2d[0] or bbox2d[3] <= bbox2d[1]:
                continue

            depth = float(-sum(zs) / len(zs))  # positive depth

            # Convert Blender camera space → OpenCV camera space.
            # Blender: +X right, +Y up, -Z forward.
            # OpenCV:  +X right, +Y down, +Z forward.
            # So: X_ocv = X_b, Y_ocv = -Y_b, Z_ocv = -Z_b.
            corners_ocv = [[X, -Y, -Z] for X, Y, Z in corners_3d]

            center_cam = [
                sum(c[0] for c in corners_ocv) / 8,
                sum(c[1] for c in corners_ocv) / 8,
                depth,
            ]
            dims = compute_dimensions(corners_ocv)
            if any(d <= 0 for d in dims):
                continue

            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_name": "parcel",
                "bbox2D_tight": bbox2d,
                "projected_corners": [{"u": u, "v": v} for u, v in projected],
                "depth": depth,
                "bbox3D_cam": corners_ocv,
                "center_cam": center_cam,
                "dimensions": dims,
                "quality": "Good",
                "valid3D": True,
                "behind_camera": False,
                "iscrowd": 0,
                "ignore": False,
                "lidar_pts": 1,
                "segmentation_pts": 1,
                "depth_error": 0.0,
                "truncation": 0.0,
                "visibility": 1.0,
                "index_mapping": list(range(8)),
            })
            ann_id += 1

    return {"images": images, "annotations": annotations}


def main():
    label_paths = sorted(LABELS_DIR.glob("*.json"))
    if not label_paths:
        raise RuntimeError(f"No label JSONs found in {LABELS_DIR}")

    random.seed(SEED)
    indices = list(range(len(label_paths)))
    random.shuffle(indices)
    split = int(len(indices) * TRAIN_FRAC)
    train_paths = [label_paths[i] for i in sorted(indices[:split])]
    val_paths = [label_paths[i] for i in sorted(indices[split:])]

    JSON_OUT_DIR.mkdir(parents=True, exist_ok=True)

    for split_name, paths in [("train", train_paths), ("val", val_paths)]:
        data = build_split_json(paths, split_name)
        out = JSON_OUT_DIR / f"CustomDataset_{split_name}.json"
        out.write_text(json.dumps(data, indent=2))
        print(f"[{split_name}] {len(paths)} images, {len(data['annotations'])} annotations -> {out}")


if __name__ == "__main__":
    main()
