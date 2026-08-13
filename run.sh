#!/usr/bin/env bash
# Render the synthetic parcel dataset with Blender (headless).
#
# Package types include boxes, cylinders lying on their side, barrels, open
# trays, timber crates, rigid envelopes and tubs, with generated printed
# graphics (handling glyphs, stencils, barcodes, address blocks), eight
# pattern sources, ~190 CC0 PBR sets, and seven named lighting scenarios over
# six lights plus an optional shadow gobo. Default run is 3000 images.
#
# Usage:
#   ./run.sh          Render in the foreground (Ctrl-C to stop)
#   ./run.sh bg       Render in the background, logging to render.log
#   ./run.sh watch    Tail the background log
#   ./run.sh stop     Stop a running render
#   ./run.sh status   Show whether a render is running + how many frames exist
#   ./run.sh smoke    Render 8 test frames into gray_images_smoke/
#   ./run.sh check    Draw the labelled 3D boxes over the images (_label_check/)
#
# Output goes to gray_images/ + labels/.
#
# Textures: photos in textures/ plus CC0 PBR sets in textures_pbr/. Fetch or
# extend the latter with:  python3 fetch_textures.py
#
# Env overrides (also work with bg/fg):
#   DS_NUM_IMAGES, DS_START_INDEX, DS_OUT_TAG, DS_SEED, DS_ENGINE=EEVEE|CYCLES
# Resuming an interrupted run, e.g. after 1200 frames:
#   DS_START_INDEX=1200 DS_NUM_IMAGES=1800 ./run.sh bg

set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$DIR/script.py"
LOG="$DIR/render.log"
PATTERN="blender -b -P $SCRIPT"

cmd="${1:-fg}"

case "$cmd" in
  fg)
    echo "Rendering in foreground (Ctrl-C to stop)..."
    exec blender -b -P "$SCRIPT"
    ;;
  bg)
    if pgrep -af "$PATTERN" >/dev/null; then
      echo "A render is already running. Use './run.sh stop' first."
      exit 1
    fi
    nohup blender -b -P "$SCRIPT" > "$LOG" 2>&1 &
    echo "Started render in background (pid $!). Log: $LOG"
    echo "Watch it with: ./run.sh watch"
    ;;
  watch)
    tail -f "$LOG"
    ;;
  stop)
    if pkill -f "$PATTERN"; then
      echo "Stopped the running render."
    else
      echo "No render was running."
    fi
    ;;
  status)
    if pgrep -af "$PATTERN" >/dev/null; then
      echo "Render RUNNING:"
      pgrep -af "$PATTERN"
    else
      echo "Render not running."
    fi
    echo "Images in gray_images/: $(ls "$DIR/gray_images" 2>/dev/null | grep -cE 'img_[0-9]+\.png' || echo 0)"
    echo "Labels in labels/:     $(ls "$DIR/labels" 2>/dev/null | grep -cE 'img_[0-9]+\.json' || echo 0)"
    ;;
  smoke)
    rm -rf "$DIR/gray_images_smoke" "$DIR/labels_smoke"
    DS_NUM_IMAGES="${DS_NUM_IMAGES:-8}" DS_OUT_TAG=_smoke \
      blender -b -P "$SCRIPT" | grep -E '^Rendered|^DONE|^pbr|^textures|^printed'
    echo "Wrote $DIR/gray_images_smoke"
    ;;
  check)
    python3 "$DIR/check_labels.py" "${2:-$DIR/gray_images}" \
      "${3:-$DIR/labels}" "$DIR/_label_check" "${4:-8}"
    ;;
  *)
    echo "Usage: ./run.sh [fg|bg|watch|stop|status|smoke|check]"
    exit 1
    ;;
esac
