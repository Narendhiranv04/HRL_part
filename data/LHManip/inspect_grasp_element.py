#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inspect a grasp segment pickle in depth.

Examples:
  # just print stats
  python inspect_grasp_segment.py /path/to/grasp/grasp01.pkl

  # also save plots + CSV next to the pkl
  python inspect_grasp_segment.py /path/to/grasp/grasp01.pkl --write-plots --dump-csv

  # also render a video from images_wrist (needs: pip install "imageio[ffmpeg]" pillow)
  python inspect_grasp_segment.py /path/to/grasp/grasp01.pkl --video --cam images_wrist --fps 10
"""

import argparse, os, sys, pickle, json, glob, re
from typing import List, Dict, Any, Tuple
import numpy as np

# --- lazy optionals for video/overlay ---
try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    Image = None
try:
    import imageio.v2 as imageio
except Exception:
    imageio = None


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]
def natural_sorted(paths: List[str]) -> List[str]:
    return sorted(paths, key=natural_key)

def pct(x: np.ndarray, qs=(0,5,50,95,100)) -> Dict[int, float]:
    if x.size == 0: return {q: float("nan") for q in qs}
    return {int(q): float(np.percentile(x, q)) for q in qs}

def frame_to_step_indices(num_frames: int, T: int) -> np.ndarray:
    if num_frames <= 0 or T <= 0: return np.zeros((0,), dtype=int)
    return np.floor(np.linspace(0, T-1, num_frames)).astype(int)

def load_pickle(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)

def load_episode_T(episode_dir: str) -> int:
    p = os.path.join(episode_dir, "data.pkl")
    if not os.path.isfile(p):
        raise FileNotFoundError(f"Original episode data.pkl not found: {p}")
    with open(p, "rb") as f:
        D = pickle.load(f)
    S = np.asarray(D.get("observations", {}).get("states"))
    return int(len(S))

def draw_footer(im, lines: List[str]):
    if Image is None: return im
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("DejaVuSansMono.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    w, h = im.size
    pad = 6
    text = "   ".join(lines)
    # compute text height
    th = 18
    bar_h = th + 2*pad
    overlay = Image.new("RGBA", (w, bar_h), (0,0,0,140))
    im = im.convert("RGBA")
    im.paste(overlay, (0, h-bar_h), overlay)
    draw = ImageDraw.Draw(im)
    draw.text((pad, h-bar_h+pad), text, fill=(255,255,255,255), font=font)
    return im.convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pkl", help="Path to grasp segment pickle (e.g., grasp01.pkl)")
    ap.add_argument("--video", action="store_true", help="Render MP4 preview for the segment")
    ap.add_argument("--cam", default="", help="images_wrist | images | images_left (auto if empty)")
    ap.add_argument("--fps", type=int, default=10, help="Video FPS")
    ap.add_argument("--write-plots", action="store_true", help="Save z/grip plots as PNG")
    ap.add_argument("--dump-csv", action="store_true", help="Dump per-step CSV next to the pkl")
    args = ap.parse_args()

    seg = load_pickle(args.pkl)
    base = os.path.splitext(args.pkl)[0]
    out_dir = os.path.dirname(args.pkl)

    # ---- Basic header ----
    print(f"[FILE] {args.pkl}")
    for k in ["kind","skill","segment_name","episode_dir","language"]:
        if k in seg: print(f"{k:>14}: {seg[k]}")
    print(f"{'source':>14}: {seg.get('source',{})}")

    # ---- Time window ----
    t_start = int(seg.get("t_start", 0))
    t_end   = int(seg.get("t_end", 0))
    H       = int(seg.get("H", max(0, t_end - t_start)))
    t_close_rel = seg.get("t_close_rel", None)
    t_lift_hit  = seg.get("t_lift_hit", None)
    print(f"\n[WINDOW] t_start={t_start}  t_end={t_end}  H={H}")
    if t_close_rel is not None:
        print(f"[EVENTS] t_close_rel={t_close_rel}")
    if t_lift_hit is not None:
        print(f"[EVENTS] t_lift_hit={t_lift_hit}")

    # ---- States / actions ----
    S = np.asarray(seg.get("states", np.zeros((0,25),dtype=np.float32)))
    A = np.asarray(seg.get("actions", np.zeros((0,8),dtype=np.float32)))
    if S.shape[0] != H:
        print(f"[WARN] states length {len(S)} != H {H}")
    if A.shape[0] != H:
        print(f"[WARN] actions length {len(A)} != H {H}")
    print(f"\n[SHAPES] states={S.shape} actions={A.shape}")

    # ---- Derived ----
    if "derived" in seg:
        D = seg["derived"]
        ee_pos = np.asarray(D.get("ee_pos", S[:,0:3] if S.size else np.zeros((0,3))))
        ee_quat = np.asarray(D.get("ee_quat", S[:,3:7] if S.size else np.zeros((0,4))))
        g_open = np.asarray(D.get("gripper_opening", S[:,14:16].mean(axis=1) if S.size else np.zeros((0,))))
    else:
        ee_pos = S[:,0:3] if S.size else np.zeros((0,3))
        ee_quat = S[:,3:7] if S.size else np.zeros((0,4))
        g_open = S[:,14:16].mean(axis=1) if S.size else np.zeros((0,))
    z = ee_pos[:,2] if ee_pos.size else np.zeros((0,))

    # ---- Stats ----
    print("\n[STATS] ee.z (m):", pct(z))
    print("[STATS] grip (m):", pct(g_open))
    if A.size:
        pos3 = A[:,0:3] if A.shape[1] >= 3 else np.zeros((H,3))
        print("[STATS] Δpos (action[0:3]) per step (mean±std):",
              np.round(pos3.mean(axis=0),4), "±", np.round(pos3.std(axis=0),4))
        print("[STATS] action min/max:", float(A.min()), float(A.max()))

    # ---- Thresholds (if any) ----
    thr = seg.get("thresholds", {})
    if thr:
        print("\n[THRESHOLDS]", json.dumps(thr, indent=2))

    # ---- CSV dump ----
    if args.dump_csv:
        csv_path = base + ".csv"
        try:
            import csv
            with open(csv_path, "w", newline="") as f:
                w = csv.writer(f)
                cols = ["step","ee_x","ee_y","ee_z","grip"] + [f"a{i}" for i in range(A.shape[1])]
                w.writerow(cols)
                for i in range(H):
                    row = [t_start + i, ee_pos[i,0], ee_pos[i,1], ee_pos[i,2], g_open[i]] + \
                          (A[i,:].tolist() if A.size else [])
                    w.writerow(row)
            print(f"[CSV]  {csv_path}")
        except Exception as e:
            print(f"[CSV] failed: {e}")

    # ---- Plots ----
    if args.write_plots:
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.rcParams["figure.figsize"] = (9,4)
            x = np.arange(H)
            # z plot
            plt.figure()
            plt.plot(x, z, label="ee.z (m)")
            if t_close_rel is not None and 0 <= t_close_rel < H:
                plt.axvline(t_close_rel, color="k", linestyle="--", label="t_close_rel")
            plt.xlabel("segment step"); plt.ylabel("z (m)"); plt.title("EE height")
            plt.legend(); plt.tight_layout()
            out_png1 = base + "_z.png"; plt.savefig(out_png1); plt.close()
            print(f"[PLOT] {out_png1}")
            # grip plot
            plt.figure()
            plt.plot(x, g_open, label="grip opening (m)")
            if t_close_rel is not None and 0 <= t_close_rel < H:
                plt.axvline(t_close_rel, color="k", linestyle="--", label="t_close_rel")
            plt.xlabel("segment step"); plt.ylabel("m"); plt.title("Gripper opening")
            plt.legend(); plt.tight_layout()
            out_png2 = base + "_grip.png"; plt.savefig(out_png2); plt.close()
            print(f"[PLOT] {out_png2}")
        except Exception as e:
            print(f"[PLOT] skipped (matplotlib missing?): {e}")

    # ---- Video preview of the segment ----
    if args.video:
        if imageio is None or Image is None:
            print('[VIDEO] missing deps. Install: pip install "imageio[ffmpeg]" pillow')
        else:
            episode_dir = seg.get("episode_dir", "")
            if not episode_dir or not os.path.isdir(episode_dir):
                print("[VIDEO] episode_dir missing in pickle or not found on disk.")
            else:
                # choose camera
                cam = args.cam
                cams_pref = [cam] if cam else ["images_wrist","images","images_left"]
                cam_used = ""
                fpaths = []
                for c in cams_pref:
                    pdir = os.path.join(episode_dir, c)
                    if os.path.isdir(pdir):
                        imgs = glob.glob(os.path.join(pdir, "*.png")) + glob.glob(os.path.join(pdir, "*.jpg"))
                        imgs = natural_sorted(imgs)
                        if imgs:
                            cam_used, fpaths = c, imgs
                            break
                if not fpaths:
                    print("[VIDEO] no frames found in episode cams.")
                else:
                    T_total = load_episode_T(episode_dir)
                    idx_all = frame_to_step_indices(len(fpaths), T_total)
                    # select frames whose mapped step falls inside [t_start, t_end-1]
                    keep = [i for i, t in enumerate(idx_all) if t_start <= t < t_end]
                    if not keep:
                        print("[VIDEO] no frames mapped inside the segment window for cam:", cam_used)
                    else:
                        out_mp4 = base + f"_{cam_used}.mp4"
                        try:
                            w = imageio.get_writer(out_mp4, fps=args.fps, format="FFMPEG", macro_block_size=None)
                            for j, i in enumerate(keep, start=1):
                                try:
                                    im = Image.open(fpaths[i]).convert("RGB")
                                except Exception as e:
                                    print(f"[VIDEO] skip frame {fpaths[i]}: {e}")
                                    continue
                                # overlay
                                seg_step = (idx_all[i] - t_start)
                                ee_z = z[seg_step] if 0 <= seg_step < H else float("nan")
                                g = g_open[seg_step] if 0 <= seg_step < H else float("nan")
                                lines = [
                                    f"{os.path.basename(args.pkl)}  cam={cam_used}",
                                    f"frame {j}/{len(keep)}  step {seg_step+1}/{H}  ee.z={ee_z:+.3f}  grip={g:+.3f}"
                                ]
                                im = draw_footer(im, lines)
                                w.append_data(np.asarray(im))
                            w.close()
                            print(f"[VIDEO] {out_mp4}")
                            meta = {
                                "segment_pkl": os.path.abspath(args.pkl),
                                "episode_dir": os.path.abspath(episode_dir),
                                "cam": cam_used, "fps": args.fps,
                                "frames_in_segment": len(keep),
                                "t_start": t_start, "t_end": t_end, "H": H
                            }
                            with open(os.path.splitext(out_mp4)[0] + ".json","w") as f:
                                json.dump(meta, f, indent=2)
                        except Exception as e:
                            print(f"[VIDEO] failed to write mp4: {e}\n[HINT] pip install \"imageio[ffmpeg]\" pillow")

    print("\n[DONE]")


if __name__ == "__main__":
    main()
