import os
import sys
import argparse
import json
import re
from typing import List

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio

# Ensure repo root is importable when executed directly
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from hrl_grasp.segmentation.episode_io import (
    load_episode,
    frame_to_step_indices,
    next_segid,
    append_manifest_row,
)

DOC = """
Manual segment by image frames.

Examples:
  # Grasp 1: frames 0–196 on images_wrist
  python -m scripts.segment_by_frames \
    --episode_dir "/home/naren/SEER/HRL_part/data/LHManip/LHManip_partial/long_horizon_manipulation_dataset/place_the_bowls_on_the_appropriate_plate/place_the_bowls_on_the_appropriate_plate/4" \
    --out_dir "/home/naren/SEER/HRL_part/.../4/segments" \
    --skill grasp \
    --seg_id 1 \
    --map_cam images_wrist \
    --frame_start 0 \
    --frame_end 196 \
    --cams images,images_wrist \
    --save_videos \
    --render_fps 10

  # Grasp 2: frames 370–582 on images_wrist
  python -m scripts.segment_by_frames \
    --episode_dir "/home/naren/SEER/HRL_part/data/LHManip/LHManip_partial/long_horizon_manipulation_dataset/place_the_bowls_on_the_appropriate_plate/place_the_bowls_on_the_appropriate_plate/4" \
    --out_dir "/home/naren/SEER/HRL_part/.../4/segments" \
    --skill grasp \
    --seg_id 2 \
    --map_cam images_wrist \
    --frame_start 370 \
    --frame_end 582 \
    --cams images,images_wrist \
    --save_videos \
    --render_fps 10
"""


def _natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def _draw_footer(img: Image.Image, lines: List[str]) -> Image.Image:
    w, h = img.size
    pad = 6
    bar_h = 18 + 14 * (len(lines) - 1)
    overlay = Image.new("RGBA", (w, h))
    draw = ImageDraw.Draw(overlay)
    draw.rectangle([(0, h - bar_h), (w, h)], fill=(0, 0, 0, 140))
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    y = h - bar_h + pad
    for line in lines:
        draw.text((pad, y), line, fill=(255, 255, 255, 255), font=font)
        y += 14
    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


def _render_cam_segment(ep: dict, cam: str, t_start: int, t_end: int, out_mp4: str, fps: int, segment_name: str):
    states = ep["states"]
    T = int(states.shape[0])

    frames = list(ep["frame_paths"].get(cam, []) or [])
    frames.sort(key=_natural_key)
    if not frames:
        print(f"[render] No frames in {cam}; skip {out_mp4}")
        return

    idx_all = frame_to_step_indices(len(frames), T)
    z = states[:, 2].astype(float)
    g = states[:, 14:16].mean(axis=1).astype(float)

    sel = [i for i in range(len(frames)) if t_start <= int(idx_all[i]) < t_end]
    if not sel:
        print(f"[render] No frames for {cam} in steps [{t_start},{t_end}); skip")
        return

    os.makedirs(os.path.dirname(out_mp4), exist_ok=True)

    tried = []
    writer_options = [
        {"fps": int(fps), "codec": "libx264", "macro_block_size": None},
        {"fps": int(fps), "codec": "mpeg4"},
        {"fps": int(fps)},
    ]
    for writer_kwargs in writer_options:
        try:
            kwargs = {"mode": "I", "fps": writer_kwargs.get("fps", int(fps))}
            if "codec" in writer_kwargs:
                kwargs["codec"] = writer_kwargs["codec"]
            if "macro_block_size" in writer_kwargs:
                kwargs["macro_block_size"] = writer_kwargs["macro_block_size"]
            w_ctx = imageio.get_writer(out_mp4, **kwargs)
            with w_ctx as w:  # type: ignore[assignment]
                for k, i in enumerate(sel):
                    step = int(idx_all[i])
                    img = Image.open(frames[i]).convert("RGB")
                    lines = [
                        f"{segment_name}",
                        f"frame {k+1}/{len(sel)} | step {step} | H {t_end - t_start}",
                        f"z={z[step]:.3f} m | grip={g[step]:.3f} m",
                    ]
                    append = getattr(w, "append_data", None)
                    if append is None:
                        raise RuntimeError("imageio writer has no append_data")
                    append(np.asarray(_draw_footer(img, lines)))
            print(f"[render] wrote {out_mp4} ({len(sel)} frames, steps [{t_start},{t_end}))")
            break
        except Exception as e:
            tried.append((writer_kwargs, str(e)))
            continue
    else:
        print(f"[render][ERROR] all writer attempts failed: {tried}")


def main():
    ap = argparse.ArgumentParser(description="Extract a segment by image frame indices.", epilog=DOC)
    ap.add_argument("--episode_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--skill", required=True)
    ap.add_argument("--seg_id", type=int, default=None)
    ap.add_argument("--map_cam", default=None, help="Camera whose frames define [start,end]; default wrist if exists else images")
    ap.add_argument("--frame_start", type=int, required=True)
    ap.add_argument("--frame_end", type=int, required=True)
    ap.add_argument("--cams", default="images,images_wrist")
    ap.add_argument("--save_videos", action="store_true")
    ap.add_argument("--render_fps", type=int, default=10)

    args = ap.parse_args()

    ep = load_episode(args.episode_dir)
    states = np.asarray(ep.get("states"))
    actions = ep.get("actions")
    actions = np.asarray(actions) if actions is not None else None
    T_states = int(states.shape[0])
    T_actions = int(actions.shape[0]) if actions is not None else T_states
    T = min(T_states, T_actions)
    if T_states != T_actions and actions is not None:
        # clip to the shorter length
        states = states[:T]
        actions = actions[:T]

    # Choose mapping camera
    map_cam = args.map_cam
    if not map_cam:
        map_cam = "images_wrist" if (ep["frame_paths"].get("images_wrist") or []) else "images"
    frames_map = list(ep["frame_paths"].get(map_cam, []) or [])
    if not frames_map and map_cam != "images":
        # fallback
        map_cam = "images"
        frames_map = list(ep["frame_paths"].get(map_cam, []) or [])
    frames_map.sort(key=_natural_key)
    if not frames_map:
        raise SystemExit(f"No frames found in map_cam '{map_cam}'. Aborting.")

    F = len(frames_map)
    f0 = max(0, min(F - 1, int(args.frame_start)))
    f1 = max(0, min(F - 1, int(args.frame_end)))
    if f0 > f1:
        raise SystemExit(f"frame_start ({f0}) > frame_end ({f1}).")

    idx_all = frame_to_step_indices(F, T)
    t_start = int(idx_all[f0])
    t_end = int(min(T, idx_all[f1] + 1))
    H = int(t_end - t_start)
    if H <= 0:
        raise SystemExit(f"Mapped segment has non-positive H: t_start={t_start}, t_end={t_end}")

    states_seg = states[t_start:t_end]
    actions_seg = actions[t_start:t_end] if actions is not None else None

    # Determine seg_id / segment_name
    out_skill_dir = os.path.join(os.path.abspath(args.out_dir), args.skill)
    os.makedirs(out_skill_dir, exist_ok=True)
    seg_id = int(args.seg_id) if args.seg_id is not None else next_segid(out_skill_dir, args.skill)
    segment_name = f"{args.skill}{seg_id:02d}"

    payload = {
        "kind": "manual_segment_v1",
        "skill": str(args.skill),
        "segment_name": segment_name,
        "episode_dir": ep.get("episode_dir", os.path.abspath(args.episode_dir)),
        "language": ep.get("language", ""),
        "source": {
            "mode": "frames",
            "map_cam": map_cam,
            "frame_start": int(f0),
            "frame_end": int(f1),
        },
        "t_start": int(t_start),
        "t_end": int(t_end),
        "H": int(H),
        "states": states_seg,
        "actions": actions_seg,
        "derived": {
            "ee_pos": states_seg[:, 0:3],
            "ee_quat": states_seg[:, 3:7],
            "gripper_opening": states_seg[:, 14:16].mean(axis=1),
        },
    }

    # Save pickle and params.json for reproducibility
    pkl_path = os.path.join(out_skill_dir, f"{segment_name}.pkl")
    with open(pkl_path, "wb") as f:
        import pickle
        pickle.dump(payload, f, protocol=5)

    with open(os.path.join(out_skill_dir, f"{segment_name}.params.json"), "w") as f:
        json.dump({
            "episode_dir": args.episode_dir,
            "out_dir": args.out_dir,
            "skill": args.skill,
            "seg_id": seg_id if args.seg_id is not None else None,
            "map_cam": map_cam,
            "frame_start": int(f0),
            "frame_end": int(f1),
            "cams": args.cams,
            "save_videos": bool(args.save_videos),
            "render_fps": int(args.render_fps),
        }, f, indent=2)

    # Append manifest row
    manifest_csv = os.path.join(out_skill_dir, "manifest.csv")
    row = {
        "episode_dir": ep.get("episode_dir", os.path.abspath(args.episode_dir)),
        "skill": args.skill,
        "segment_name": segment_name,
        "map_cam": map_cam,
        "frame_start": int(f0),
        "frame_end": int(f1),
        "t_start": int(t_start),
        "t_end": int(t_end),
        "H": int(H),
    }
    header = [
        "episode_dir",
        "skill",
        "segment_name",
        "map_cam",
        "frame_start",
        "frame_end",
        "t_start",
        "t_end",
        "H",
    ]
    append_manifest_row(manifest_csv, row, header)

    # Render videos if requested
    if args.save_videos:
        videos_dir = os.path.join(out_skill_dir, "videos")
        os.makedirs(videos_dir, exist_ok=True)
        for cam in [c.strip() for c in args.cams.split(",") if c.strip()]:
            if not (ep["frame_paths"].get(cam) or []):
                print(f"[render] Cam '{cam}' missing; skip")
                continue
            out_mp4 = os.path.join(videos_dir, f"{segment_name}_{cam}.mp4")
            _render_cam_segment(ep, cam, t_start, t_end, out_mp4, int(args.render_fps), segment_name)

    print("Saved:")
    print(f"  {pkl_path}")
    if args.save_videos:
        print(f"  videos -> {os.path.join(out_skill_dir, 'videos')}")
    print(f"  manifest -> {manifest_csv}")


if __name__ == "__main__":
    main()
