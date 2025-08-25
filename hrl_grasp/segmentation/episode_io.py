import os
import re
import glob
import csv
import pickle
from typing import Dict, List

import numpy as np


_DEF_CAM_DIRS = ("images", "images_wrist", "images_left")


def _natural_key(s: str):
    # "img_2.png" -> ["img_", 2, ".png"] so sorting is numeric on numbers
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def _sorted_abs_paths(pattern: str) -> List[str]:
    paths = glob.glob(pattern)
    paths.sort(key=_natural_key)
    return [os.path.abspath(p) for p in paths]


def natural_sort(paths: List[str]) -> List[str]:
    """Digit-aware sort helper exposed for callers.

    Returns a new list sorted using the same natural ordering as used internally.
    """
    return sorted(paths, key=_natural_key)


def load_episode(ep_dir: str) -> Dict:
    """Load an episode directory containing data.pkl and image folders.

    Returns a dict with keys: states, actions, language, frame_paths{cam}.
    """
    ep_dir = os.path.abspath(ep_dir)
    pkl_path = os.path.join(ep_dir, "data.pkl")
    if not os.path.isfile(pkl_path):
        raise FileNotFoundError(f"Missing data.pkl under {ep_dir}")
    with open(pkl_path, "rb") as f:
        D = pickle.load(f)

    obs = D.get("observations", {})
    states = np.asarray(obs.get("states"))
    actions = np.asarray(D.get("actions")) if D.get("actions") is not None else None

    lang = D.get("language_instruction")
    if isinstance(lang, (list, tuple)) and len(lang) > 0:
        language = lang[0]
    elif isinstance(lang, str):
        language = lang
    else:
        language = ""

    frame_paths = {}
    for cam in _DEF_CAM_DIRS:
        frame_paths[cam] = _sorted_abs_paths(os.path.join(ep_dir, cam, "*.png"))
        if not frame_paths[cam]:
            # also check jpg
            frame_paths[cam] = _sorted_abs_paths(os.path.join(ep_dir, cam, "*.jpg"))

    return {
        "states": states,
        "actions": actions,
        "language": language,
        "frame_paths": frame_paths,
        "episode_dir": ep_dir,
    }


def frame_to_step_indices(num_frames: int, T: int) -> np.ndarray:
    """Monotonic mapping from frames to control steps using floor(linspace).

    Returns an array of length F with values in [0, T-1].
    """
    if num_frames <= 0:
        return np.zeros((0,), dtype=np.int32)
    if T <= 0:
        return np.zeros((num_frames,), dtype=np.int32)
    idx = np.floor(np.linspace(0, max(T - 1, 0), num_frames)).astype(np.int32)
    # Enforce monotonic non-decreasing
    idx = np.maximum.accumulate(idx)
    return idx


def save_segment_pkl(path: str, payload: Dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=5)


def next_segid(out_dir_skill: str, skill: str) -> int:
    """Return the next 1-based seg id by scanning existing {skill}*.pkl files.

    E.g., if grasp01.pkl, grasp02.pkl exist, returns 3.
    """
    os.makedirs(out_dir_skill, exist_ok=True)
    patt = os.path.join(out_dir_skill, f"{skill}*.pkl")
    files = glob.glob(patt)
    max_id = 0
    for p in files:
        base = os.path.basename(p)
        m = re.match(fr"{re.escape(skill)}(\d+).*\\.pkl$", base)
        if m:
            try:
                n = int(m.group(1))
                max_id = max(max_id, n)
            except ValueError:
                continue
    return max_id + 1


def append_manifest_row(manifest_csv: str, row: Dict, header: List[str]):
    """Append a row dict to CSV, creating file and writing header if missing."""
    os.makedirs(os.path.dirname(manifest_csv), exist_ok=True)
    write_header = not os.path.exists(manifest_csv)
    with open(manifest_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k) for k in header})


def write_manifest_csv(path: str, rows: List[Dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    header = [
        "episode_dir",
        "seg_id",
        "t_start",
        "t_close",
        "t_close_rel",
        "t_end",
        "H",
        "success",
        "z_table",
        "z_low",
        "z_high",
        "g_thr",
        "close_drop_thr",
        "h_lift",
        "lift_window",
        "closed_frac_min",
        # expansion (optional)
        "H_target",
        "H_pre_dyn",
        "H_post_dyn",
        "frames_est",
    ]
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if write_header:
            writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in header})
