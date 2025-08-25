import os
import re
from typing import List, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio

from hrl_grasp.segmentation.episode_io import load_episode, frame_to_step_indices


def _natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def _draw_footer_overlay(img: Image.Image, lines: List[str]) -> Image.Image:
    # Draw a semi-transparent black footer with white text
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


def render_segment_video(ep_dir: str, cam_dirname: str, t_start: int, t_end: int, out_mp4: str, success: Optional[bool] = None):
    ep = load_episode(ep_dir)
    states = ep["states"]
    T = int(states.shape[0])

    frames = ep["frame_paths"].get(cam_dirname, [])
    if not frames:
        print(f"[render] No frames in {cam_dirname} for {ep_dir}; skipping {out_mp4}")
        return

    F = len(frames)
    idx_all = frame_to_step_indices(F, T)

    # select frames whose mapped step within [t_start, t_end)
    sel = [i for i in range(F) if t_start <= int(idx_all[i]) < t_end]
    if not sel:
        print(f"[render] No frames fall within steps [{t_start},{t_end}) for {cam_dirname}; skipping")
        return

    os.makedirs(os.path.dirname(out_mp4), exist_ok=True)

    z = states[:, 2].astype(float)
    g = states[:, 14:16].mean(axis=1).astype(float)

    # Try preferred codec, then fall back if missing
    tried = []
    for writer_kwargs in (
        dict(fps=10, codec="libx264", format="FFMPEG", macro_block_size=None),
        dict(fps=10, codec="mpeg4", format="FFMPEG"),
        dict(fps=10, format="FFMPEG"),
    ):
        try:
            with imageio.get_writer(out_mp4, **writer_kwargs) as w:
                for k, i in enumerate(sel):
                    step = int(idx_all[i])
                    img = Image.open(frames[i]).convert("RGB")
                    ok = f"success={success}" if success is not None else None
                    lines = [f"frame {k+1}/{len(sel)} | step {step}"]
                    lines.append(f"z={z[step]:.3f} m | grip={g[step]:.3f} m")
                    if ok:
                        lines.append(ok)
                    img = _draw_footer_overlay(img, lines)
                    w.append_data(np.asarray(img))
            break
        except Exception as e:
            tried.append((writer_kwargs, str(e)))
            continue
    else:
        raise RuntimeError(f"All writer attempts failed: {tried}")

    print(f"[render] wrote {out_mp4} ({len(sel)} frames, steps [{t_start},{t_end}))")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode_dir", required=True)
    ap.add_argument("--cam", required=True)
    ap.add_argument("--t_start", type=int, required=True)
    ap.add_argument("--t_end", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    render_segment_video(args.episode_dir, args.cam, args.t_start, args.t_end, args.out)
