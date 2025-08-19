#!/usr/bin/env python3
"""
make_videos.py

Create MP4 videos from RLBench dataset image sequences for selected camera views.

Usage:
  python make_videos.py --root /home/naren/HRL_part/rlbench_data \
                        --views wrist_rgb overhead_rgb \
                        --fps 12

It will traverse tasks/variation*/episode* and produce one MP4 per episode per view.
Videos are saved alongside the images as <view>.mp4. If --views is omitted, it
auto-detects any view folders containing images.
"""
import argparse
from pathlib import Path
import sys

import imageio.v2 as iio

IMG_EXTS = ('.png', '.jpg', '.jpeg')


def _list_frames(img_dir: Path):
    files = []
    for ext in IMG_EXTS:
        files.extend(img_dir.glob(f'*{ext}'))
    # Sort by numeric stem if possible, else lexicographically
    def sort_key(p: Path):
        try:
            return (0, int(p.stem))
        except ValueError:
            return (1, p.name)
    return sorted(files, key=sort_key)


def build_video_from_images(img_dir: Path, out_path: Path, fps: int, frames) -> str:
    """
    Returns:
      'ok' on success,
      'no_frames' if the directory has no images,
      'encode_fail' if a writer couldn't be created or failed.
    """
    if not frames:
        return 'no_frames'

    # Read first frame to get size
    first = iio.imread(frames[0])
    height, width = first.shape[:2]

    # Try FFMPEG explicitly; fall back to default if available
    try:
        writer = iio.get_writer(out_path, fps=fps, format='FFMPEG')
    except Exception:
        try:
            writer = iio.get_writer(out_path, fps=fps)
        except Exception as e:
            print(f"[FAIL] Could not open video writer for {out_path}: {e}")
            print("       Hint: pip install 'imageio[ffmpeg]' or 'imageio-ffmpeg'")
            return 'encode_fail'

    try:
        for f in frames:
            img = iio.imread(f)
            if img.shape[0] != height or img.shape[1] != width:
                # Simple safeguard: skip mismatched frames
                continue
            writer.append_data(img)
        return 'ok'
    except Exception as e:
        print(f"[FAIL] Encoding error for {out_path}: {e}")
        return 'encode_fail'
    finally:
        try:
            writer.close()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='Path to rlbench_data root')
    ap.add_argument('--views', nargs='+', default=None,
                    help='Camera view folders to convert (e.g., wrist_rgb overhead_rgb).\n'
                         'If omitted, auto-detects any subfolders with images.')
    ap.add_argument('--fps', type=int, default=12, help='Frames per second for videos')
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        print(f"[ERR] Root not found: {root}")
        sys.exit(1)

    # Supported directory shapes:
    # - tasks/<task>/variation*/episode*/<view>/*.{png,jpg}
    # - tasks/<task>/variation*/episodes/episode*/<view>/*.{png,jpg}
    # - tasks/<task>/scene_*/variation*/episodes/episode*/<view>/*.{png,jpg}
    episodes = []
    episodes.extend(root.glob('*/variation*/episode*'))
    episodes.extend(root.glob('*/variation*/episodes/episode*'))
    episodes.extend(root.glob('*/scene_*/variation*/episodes/episode*'))
    episodes = sorted(set(episodes))
    if not episodes:
        print(f"[ERR] No episodes found under {root}")
        # Print a quick hint of the structure
        top = list(root.glob('*'))[:10]
        print('[HINT] Top-level entries:', [p.name for p in top])
        sys.exit(2)

    total = 0
    made = 0
    for ep in episodes:
        # Determine views to use
        if args.views is None:
            # Auto-detect any immediate subfolders that contain at least one image
            candidate_dirs = [d for d in ep.iterdir() if d.is_dir()]
            views = []
            for d in candidate_dirs:
                if any(d.glob(f'*{ext}') for ext in IMG_EXTS):
                    views.append(d.name)
        else:
            views = args.views

        for view in views:
            img_dir = ep / view
            if not img_dir.is_dir():
                continue
            out_path = ep / f'{view}.mp4'
            total += 1
            frames = _list_frames(img_dir)
            result = build_video_from_images(img_dir, out_path, args.fps, frames)
            if result == 'ok':
                made += 1
                print(f"[OK] {out_path}")
            elif result == 'no_frames':
                print(f"[SKIP] No frames in {img_dir}")
            else:
                print(f"[FAIL] Could not create {out_path}")
    print(f"Done. Created {made}/{total} videos.")


if __name__ == '__main__':
    main()
