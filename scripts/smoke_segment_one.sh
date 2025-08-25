#!/usr/bin/env bash
set -euo pipefail

# Edit paths before running
EP_DIR="/full/path/to/episode"
OUT_DIR="/full/path/to/out/grasp_segments"
CAMS="images,images_wrist"

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  echo "[info] using venv: $VIRTUAL_ENV"
fi

python -m pip install --quiet --upgrade pip
python -m pip install --quiet "imageio[ffmpeg]" pillow

python -m scripts.segment_grasp \
  --episode_dir "$EP_DIR" \
  --out_dir "$OUT_DIR" \
  --cams "$CAMS" \
  --save_videos \
  --close_drop_thr 0.004 --h_lift 0.10

SEG_COUNT=$(ls -1 "$OUT_DIR"/seg_*.pkl 2>/dev/null | wc -l | tr -d ' ')
VID_COUNT=$(ls -1 "$OUT_DIR"/videos/*.mp4 2>/dev/null | wc -l | tr -d ' ')

echo "[smoke] segments=$SEG_COUNT videos=$VID_COUNT"
echo "[smoke] out: $OUT_DIR"
