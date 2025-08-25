import os
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import argparse
import pickle
from typing import List

import numpy as np

from hrl_grasp.segmentation.episode_io import load_episode, save_segment_pkl, write_manifest_csv
from hrl_grasp.segmentation.grasp_segmenter import detect_grasps, detect_cross_only, slice_segment, build_segment_payload


def _parse_cams(cams: str) -> List[str]:
    return [c.strip() for c in cams.split(",") if c.strip()]


def main():
    ap = argparse.ArgumentParser(description="Detect and save grasp segments for one episode")
    ap.add_argument("--episode_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--cams", default="images,images_wrist,images_left")
    ap.add_argument("--save_videos", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--fallback_cross_only", action="store_true")
    ap.add_argument("--autotune", action="store_true")
    # thresholds/config
    ap.add_argument("--W", type=int, default=5)
    ap.add_argument("--close_drop_thr", type=float, default=0.003)
    ap.add_argument("--h_lift", type=float, default=0.06)
    ap.add_argument("--lift_window", type=int, default=30)
    ap.add_argument("--H_pre", type=int, default=12)
    ap.add_argument("--H_post", type=int, default=12)
    ap.add_argument("--H_refrac", type=int, default=20)
    ap.add_argument("--closed_frac_min", type=float, default=0.7)
    ap.add_argument("--nearband_relax", action="store_true")
    # expansion controls
    ap.add_argument("--target_frames", type=int, default=28, help="Target number of video frames per segment")
    ap.add_argument("--pre_lookback_max", type=int, default=120)
    ap.add_argument("--H_pre_min", type=int, default=24)
    ap.add_argument("--H_pre_cap", type=int, default=180)
    ap.add_argument("--pre_xy_min", type=float, default=0.10)
    ap.add_argument("--post_mode", type=str, choices=["fixed", "to_lift", "to_transport"], default="to_transport")
    ap.add_argument("--H_post_min", type=int, default=24)
    ap.add_argument("--H_post_cap", type=int, default=180)
    ap.add_argument("--transport_z", type=float, default=0.18)
    ap.add_argument("--post_dwell", type=int, default=8)

    args = ap.parse_args()

    ep = load_episode(args.episode_dir)

    cfg = dict(
        W=args.W,
        close_drop_thr=args.close_drop_thr,
        h_lift=args.h_lift,
        lift_window=args.lift_window,
        H_pre=args.H_pre,
        H_post=args.H_post,
        H_refrac=args.H_refrac,
        closed_frac_min=args.closed_frac_min,
        nearband_relax=args.nearband_relax,
        # expansion
        target_frames=args.target_frames,
        pre_lookback_max=args.pre_lookback_max,
        H_pre_min=args.H_pre_min,
        H_pre_cap=args.H_pre_cap,
        pre_xy_min=args.pre_xy_min,
        post_mode=args.post_mode,
        H_post_min=args.H_post_min,
        H_post_cap=args.H_post_cap,
        transport_z=args.transport_z,
        post_dwell=args.post_dwell,
    )

    # Episode context for expansion logic
    # Choose a camera to estimate number of frames; prefer wrist if present
    frame_paths = ep.get("frame_paths", {})
    num_frames = 0
    for cam_pref in ("images_wrist", "images", "images_left"):
        fps = frame_paths.get(cam_pref) or []
        if fps:
            num_frames = len(fps)
            break
    cfg["_num_frames"] = int(num_frames)
    cfg["_T"] = int(ep["states"].shape[0])

    # Save cfg inside meta for payload building
    ep_meta = dict(ep)
    ep_meta["cfg"] = cfg

    segments = detect_grasps(ep["states"], cfg, debug=args.debug)
    chosen_cfg = dict(cfg)
    if not segments and args.fallback_cross_only:
        print("[SEGMENTER] no segments with main detector; trying fallback crossings-only...")
        segments = detect_cross_only(ep["states"], cfg)
    if not segments and args.autotune:
        print("[SEGMENTER] no segments; autotuning thresholds...")
        grid = [
            (0.05, 40, 0.003, 0.7),
            (0.04, 50, 0.003, 0.7),
            (0.04, 50, 0.0025, 0.6),
        ]
        for h_lift, lift_window, close_thr, closed_frac_min in grid:
            trial = dict(cfg)
            trial.update(dict(h_lift=h_lift, lift_window=lift_window, close_drop_thr=close_thr, closed_frac_min=closed_frac_min))
            segs2 = detect_grasps(ep["states"], trial, debug=args.debug)
            if segs2:
                segments = segs2
                chosen_cfg = trial
                print(f"[SEGMENTER][autotune] chosen: h_lift={h_lift}, lift_window={lift_window}, close_drop_thr={close_thr}, closed_frac_min={closed_frac_min}")
                break

    os.makedirs(args.out_dir, exist_ok=True)
    out_segments = []
    rows = []

    videos_dir = os.path.join(args.out_dir, "videos")
    cams = _parse_cams(args.cams)

    agg_payloads = []

    print(f"[SEGMENTER] episode={ep['episode_dir']} segments={len(segments)}")

    for i, seg in enumerate(segments):
        t_start, t_close, t_end = seg["t_start"], seg["t_close"], seg["t_end"]
        H = int(t_end - t_start)
        print(f"  seg_{i+1:06d}: t=[{t_start}..{t_end}] H={H} success={seg['success']}")

        sl = slice_segment(ep["states"], ep.get("actions"), t_start, t_end)
        payload = build_segment_payload(ep_meta, seg, sl)

        seg_id = f"seg_{i+1:06d}"
        pkl_path = os.path.join(args.out_dir, f"{seg_id}.pkl")
        save_segment_pkl(pkl_path, payload)

        if args.aggregate:
            agg_payloads.append(payload)

        row = {
            "episode_dir": ep["episode_dir"],
            "seg_id": seg_id,
            "t_start": t_start,
            "t_close": t_close,
            "t_close_rel": int(payload["t_close_rel"]),
            "t_end": t_end,
            "H": H,
            "success": bool(seg["success"]),
            "z_table": float(payload["thresholds"]["z_table"]),
            "z_low": float(payload["thresholds"].get("z_low", 0.0)),
            "z_high": float(payload["thresholds"].get("z_high", 0.0)),
            "g_thr": float(payload["thresholds"]["g_thr"]),
            "close_drop_thr": float(payload["thresholds"]["close_drop_thr"]),
            "h_lift": float(payload["thresholds"]["h_lift"]),
            "lift_window": int(payload["thresholds"]["lift_window"]),
            "closed_frac_min": float(payload["thresholds"].get("closed_frac_min", 0.7)),
        }
        # Expansion fields if present
        exp = payload.get("expansion", {})
        if exp:
            row.update({
                "H_target": int(exp.get("H_target", 0)),
                "H_pre_dyn": int(exp.get("H_pre_dyn", 0)),
                "H_post_dyn": int(exp.get("H_post_dyn", 0)),
                "frames_est": int(exp.get("frames_est", 0)),
            })
        rows.append(row)

        if args.save_videos:
            try:
                from scripts.render_grasp_segments import render_segment_video
                for cam in cams:
                    out_mp4 = os.path.join(videos_dir, f"{seg_id}_{cam}.mp4")
                    render_segment_video(ep["episode_dir"], cam, t_start, t_end, out_mp4, success=bool(seg["success"]))
            except Exception as e:
                print(f"[WARN] video rendering failed: {e}")

    if args.aggregate and agg_payloads:
        agg_path = os.path.join(args.out_dir, "grasp_segments_agg.pkl")
        with open(agg_path, "wb") as f:
            pickle.dump(agg_payloads, f, protocol=5)

    manifest_path = os.path.join(args.out_dir, "manifest.csv")
    write_manifest_csv(manifest_path, rows)
    # Save chosen params for reproducibility
    import json
    with open(os.path.join(args.out_dir, "params.json"), "w") as f:
        json.dump(chosen_cfg, f, indent=2)

    print("Outputs:")
    print(f"  *.pkl under {args.out_dir}")
    if args.save_videos:
        print(f"  videos under {videos_dir}")
    print(f"  manifest.csv appended at {manifest_path}")


if __name__ == "__main__":
    main()
