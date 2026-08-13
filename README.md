# synthetic_dataset_blender

Synthetic parcel dataset rendered headlessly with Blender: 640x640 grayscale
images plus camera intrinsics and 8-corner 3D bounding boxes per parcel.

    ./run.sh          # render (see run.sh header for bg/watch/stop/smoke/check)

## What's in git

`scene.blend` (203 MB) is stored with [Git LFS](https://git-lfs.com), so you
need it installed before cloning:

    git lfs install
    git clone git@github.com:odark00/synthetic_dataset_blender.git

Cloning without LFS leaves a small text pointer in place of the real file and
`script.py` will fail to open it.

## What's not in git, and how to get it back

| Path | Size | How to restore |
|---|---|---|
| `textures_pbr/` | ~1 GB | `python3 fetch_textures.py` (CC0 sets from ambientCG) |
| `gray_images/`, `labels/` | ~737 MB | `./run.sh` |
| `_label_check/` | small | `./run.sh check` |
| `assets/` | ~138 MB | Not referenced by any script; content is baked into `scene.blend` |

The 170 photo textures in `textures/` (cardboard, tape and paper) *are*
tracked, as ordinary git objects rather than LFS — each is well under
GitHub's 100 MB per-file limit, so they cost no LFS quota.
