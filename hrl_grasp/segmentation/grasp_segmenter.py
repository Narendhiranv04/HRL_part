from typing import Dict, List, Optional
import numpy as np
from typing import Tuple


def smooth(x: np.ndarray, W: int = 5) -> np.ndarray:
    if x.size == 0:
        return x.copy()
    if W <= 1:
        return x.copy()
    W = int(W)
    pad = W // 2
    xpad = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(W, dtype=np.float32) / float(W)
    y = np.convolve(xpad, kernel, mode="valid")
    return y.astype(x.dtype)


def estimate_thresholds(z: np.ndarray, g: np.ndarray) -> Dict[str, float]:
    # Robust global thresholds: ignore zeros/near-zeros
    z_valid = z[z > 0.05] if z.size else z
    if z_valid.size < 100:
        z_valid = z
    z_table_global = float(np.percentile(z_valid, 2)) if z_valid.size else 0.0
    z_low_global = z_table_global + 0.02
    z_high_global = z_table_global + 0.10

    g_valid = g[g > 1e-4] if g.size else g
    if g_valid.size < 100:
        g_valid = g
    g_open = float(np.percentile(g_valid, 95)) if g_valid.size else 0.0
    g_closed = float(np.percentile(g_valid, 5)) if g_valid.size else 0.0
    g_thr = 0.5 * (g_open + g_closed)

    return {
        "z_table": float(z_table_global),
        "z_low": float(z_low_global),
        "z_high": float(z_high_global),
        "g_thr": float(g_thr),
    }


def _mean_gripper_opening(states: np.ndarray) -> np.ndarray:
    # mean of s[:,14:16]
    if states.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    return states[:, 14:16].mean(axis=1)


# --- Expansion utilities ---
def frames_per_step(num_frames: int, T: int) -> float:
    return (float(num_frames) / float(T)) if T > 0 else 0.0


def H_for_target_frames(num_frames: int, T: int, target_frames: int) -> int:
    r = frames_per_step(num_frames, T)
    return int(np.ceil(float(target_frames) / max(r, 1e-6)))


def find_approach_start(states: np.ndarray, t_close: int, cfg: dict, z_table: float) -> int:
    t0 = max(0, t_close - int(cfg.get("pre_lookback_max", 120)))
    ee = states[:, 0:3]
    g = states[:, 14:16].mean(axis=1)
    g_thr = float(cfg.get("_g_thr", 0.0))
    z_low = z_table + 0.02
    xy_close = ee[t_close, 0:2]
    best = None
    for t in range(t0, t_close):
        if g[t] <= g_thr:  # open before close
            continue
        if ee[t, 2] <= z_low:  # above near-table band => approach
            continue
        if np.linalg.norm(ee[t, 0:2] - xy_close) < float(cfg.get("pre_xy_min", 0.10)):
            continue
        best = t
        break
    if best is None:
        best = max(0, t_close - int(cfg.get("H_pre_min", 24)))
    return best


def extend_post(states: np.ndarray, t_close: int, t_lift_hit: Optional[int], cfg: dict, z_table: float) -> int:
    T = len(states)
    z = states[:, 2]
    # minimal end
    t_end = min(T, t_close + int(cfg.get("H_post_min", 24)))
    post_mode = cfg.get("post_mode", "to_transport")
    if post_mode == "to_lift":
        if t_lift_hit is not None:
            t_end = max(t_end, min(T, t_lift_hit + int(cfg.get("post_dwell", 8))))
    elif post_mode == "to_transport":
        z_thresh = z_table + float(cfg.get("transport_z", 0.18))
        t = t_close
        while t < min(T, t_close + int(cfg.get("H_post_cap", 180))):
            if z[t] >= z_thresh:
                t_end = max(t_end, min(T, t + int(cfg.get("post_dwell", 8))))
                break
            t += 1
    # else fixed: keep H_post_min
    return min(t_end, min(T, t_close + int(cfg.get("H_post_cap", 180))))


def detect_grasps(states: np.ndarray, cfg: Dict, debug: bool = False) -> List[Dict]:
    T = states.shape[0]
    if T == 0:
        return []

    W = int(cfg.get("W", 5))
    close_drop_thr = float(cfg.get("close_drop_thr", 0.004))
    h_lift = float(cfg.get("h_lift", 0.06))
    lift_window = int(cfg.get("lift_window", 30))
    H_pre = int(cfg.get("H_pre", 12))
    H_post = int(cfg.get("H_post", 12))
    H_refrac = int(cfg.get("H_refrac", 20))
    closed_frac_min = float(cfg.get("closed_frac_min", 0.7))
    nearband_relax = bool(cfg.get("nearband_relax", True))
    # Set expansion defaults if not present
    cfg.setdefault("target_frames", 25)
    cfg.setdefault("pre_lookback_max", 120)
    cfg.setdefault("H_pre_min", 24)
    cfg.setdefault("H_pre_cap", 180)
    cfg.setdefault("pre_xy_min", 0.10)
    cfg.setdefault("post_mode", "to_transport")
    cfg.setdefault("H_post_min", 24)
    cfg.setdefault("H_post_cap", 180)
    cfg.setdefault("transport_z", 0.18)
    cfg.setdefault("post_dwell", 8)

    z = states[:, 2].astype(np.float32)
    g = _mean_gripper_opening(states).astype(np.float32)

    # Smooth signals before computing deltas
    z_s = smooth(z, W=W)
    g_s = smooth(g, W=W)

    thr = estimate_thresholds(z_s, g_s)
    z_table_global, g_thr = thr["z_table"], thr["g_thr"]
    # Stash globals for expansion helpers
    cfg["_g_thr"] = g_thr
    cfg["_T"] = T
    cfg.setdefault("_num_frames", 0)

    # differences on smoothed signals
    dg = np.diff(g_s, prepend=g_s[0])
    dz = np.diff(z_s, prepend=z_s[0])

    segments: List[Dict] = []
    t = 1
    dbg_shown = 0
    while t < T - 1:
        # Local near-table band from a window before t
        t0_loc = max(0, t - 60)
        z_loc = z[t0_loc:t]
        z_loc_valid = z_loc[z_loc > 0.05]
        if z_loc_valid.size >= 20:
            z_table = float(np.percentile(z_loc_valid, 5))
        else:
            z_table = z_table_global
        z_low = z_table + 0.02
        z_high = z_table + 0.10

        # Close event: either sharp drop or threshold crossing from open->closed
        g_margin = float(cfg.get("g_margin", 0.002))
        crossed = (g_s[t] < g_thr) and (g_s[max(0, t - 3):t].max(initial=g_s[t]) > (g_thr + g_margin))
        cond_close = (dg[t] < -close_drop_thr) or crossed
        if not cond_close:
            t += 1
            continue

        # approach-near (soft if nearband_relax): either at close is in band or was in band recently
        in_band_now = (z[t] >= z_low) and (z[t] <= z_high)
        if nearband_relax:
            t0 = max(0, t - 10)
            recent_min = float(np.min(z[t0:t+1])) if t0 < t+1 else z[t]
            near_recent = (recent_min >= z_low) and (recent_min <= z_high)
            pre_ok = in_band_now or near_recent
        else:
            pre_ok = in_band_now

        # Lift confirmation within window
        z_close = z_s[t]
        t2 = min(T, t + 1 + lift_window)
        z_after = z_s[t + 1:t2]
        # First index that reaches required lift
        t_lift_hit = None
        for k in range(t + 1, t2):
            if (z_s[k] - z_close) >= h_lift:
                t_lift_hit = k
                break
        dz_max = (np.max(z_after) - z_close) if z_after.size else 0.0
        frac_closed = float((g_s[t:min(T, t + lift_window)] < g_thr).mean()) if T > t else 0.0
        success = (dz_max >= h_lift) and (frac_closed >= closed_frac_min)

        # Dynamic windowing to target frames and include approach/transport
        # Local z_table around candidate for near-table and transport height
        z_table_local = z_table
        H_target = H_for_target_frames(int(cfg.get("_num_frames", 0)), int(cfg.get("_T", T)), int(cfg.get("target_frames", 25)))

        # PRE: dynamic approach start
        t_pre = find_approach_start(states, t, cfg, z_table_local)
        # ensure at least H_pre_min and not exceeding caps around t_close
        H_pre_min = int(cfg.get("H_pre_min", 24))
        H_pre_cap = int(cfg.get("H_pre_cap", 180))
        t_min_cap = max(0, t - H_pre_cap)
        t_pre = max(t_min_cap, t_pre)
        if (t - t_pre) < H_pre_min:
            t_pre = max(t_min_cap, t - H_pre_min)

        # POST: extend per mode
        t_post = extend_post(states, t, t_lift_hit, cfg, z_table_local)

        # Pad to hit H_target symmetrically within caps and bounds
        H_now = t_post - t_pre
        need = max(0, H_target - H_now)
        # pre padding limited by cap and episode start
        pre_cap_start = max(0, t - H_pre_cap)
        max_pre_pad = t_pre - pre_cap_start
        pad_pre = min(need // 2, max_pre_pad)
        # post padding limited by cap and episode end
        post_cap_end = min(T, t + int(cfg.get("H_post_cap", 180)))
        max_post_pad = post_cap_end - t_post
        pad_post = min(need - pad_pre, max_post_pad)
        t_start = max(0, t_pre - pad_pre)
        t_end = min(T, t_post + pad_post)

        if pre_ok:
            seg = {
                "t_start": int(t_start),
                "t_close": int(t),
                "t_end": int(t_end),
                "success": bool(success),
                "z_table": float(thr["z_table"]),
                "g_thr": float(g_thr),
            }
            # expansion metadata
            seg["t_lift_hit"] = int(t_lift_hit) if t_lift_hit is not None else None
            seg["H_target"] = int(H_target)
            seg["H_pre_dyn"] = int(t - t_start)
            seg["H_post_dyn"] = int(t_end - t)
            seg["frames_est"] = int(
                np.ceil(
                    frames_per_step(int(cfg.get("_num_frames", 0)), int(cfg.get("_T", T)))
                    * (t_end - t_start)
                )
            )
            segments.append(seg)
            # Refractory period: skip ahead to avoid overlaps
            t = int(t_end + H_refrac)
            continue
        else:
            if debug and dbg_shown < 10:
                reason = []
                if not pre_ok:
                    reason.append("not_near_table")
                if not success:
                    if dz_max < h_lift:
                        reason.append("no_lift")
                    if frac_closed < closed_frac_min:
                        reason.append("not_closed_after")
                print(
                    f"[detect] reject@t={t}: reasons={','.join(reason)} z_close={z_close:.3f} dz_max={dz_max:.3f} frac_closed={frac_closed:.2f}"
                )
                dbg_shown += 1

        t += 1

    return segments


def detect_cross_only(states: np.ndarray, cfg: Dict) -> List[Dict]:
    """Fallback detector: use open->closed threshold crossings as anchors.
    Returns segments with success=False and fixed pre/post horizons.
    """
    T = states.shape[0]
    if T == 0:
        return []
    W = int(cfg.get("W", 5))
    H_pre = int(cfg.get("H_pre", 12))
    H_post = int(cfg.get("H_post", 12))
    H_refrac = int(cfg.get("H_refrac", 20))

    z = states[:, 2].astype(np.float32)
    g = _mean_gripper_opening(states).astype(np.float32)
    # thresholds
    thr = estimate_thresholds(smooth(z, W=W), smooth(g, W=max(1, min(W, 5))))
    g_thr = thr["g_thr"]

    segs: List[Dict] = []
    t = 1
    while t < T:
        crossed = (g[t] < g_thr) and (g[max(0, t - 1)] > g_thr + 0.002)
        if crossed:
            t_close = int(t)
            t_start = max(0, t_close - H_pre)
            t_end = min(T, t_close + H_post)
            segs.append({
                "t_start": t_start,
                "t_close": t_close,
                "t_end": t_end,
                "success": False,
                "z_table": float(thr["z_table"]),
                "g_thr": float(g_thr),
            })
            t = int(t_end + H_refrac)
            continue
        t += 1
    return segs


def slice_segment(states: np.ndarray, actions: Optional[np.ndarray], t_start: int, t_end: int) -> Dict:
    sl_states = states[t_start:t_end].copy()
    sl_actions = actions[t_start:t_end].copy() if actions is not None else None
    return {
        "states": sl_states,
        "actions": sl_actions,
    }


def build_segment_payload(ep_meta: Dict, seg_meta: Dict, slice_data: Dict) -> Dict:
    states = slice_data["states"]
    actions = slice_data["actions"]
    H = int(states.shape[0])

    ee_pos = states[:, 0:3]
    ee_quat = states[:, 3:7]
    g_open = states[:, 14:16].mean(axis=1)

    t_close = int(seg_meta["t_close"]) - int(seg_meta["t_start"])  # local index
    t_close_glob = int(seg_meta["t_close"]) 

    # Recompute thresholds for bookkeeping (episode-wide)
    ep_states = ep_meta.get("states")
    cfg = ep_meta.get("cfg", {})
    W = int(cfg.get("W", 5))
    if ep_states is not None and getattr(ep_states, "ndim", 0) >= 2 and ep_states.shape[0] > 0:
        z_ep = smooth(ep_states[:, 2].astype(np.float32), W=W)
        g_ep = smooth(ep_states[:, 14:16].mean(axis=1).astype(np.float32), W=W)
        thr = estimate_thresholds(z_ep, g_ep)
    else:
        thr = {"z_table": float(seg_meta.get("z_table", 0.0)), "z_low": 0.0, "z_high": 0.0, "g_thr": float(seg_meta.get("g_thr", 0.0))}

    if ep_states is not None and ep_states.shape[0] > t_close_glob:
        tp_x = float(ep_states[t_close_glob, 0])
        tp_y = float(ep_states[t_close_glob, 1])
    else:
        tp_x = 0.0
        tp_y = 0.0
    tp_z = float(thr.get("z_table", seg_meta.get("z_table", 0.0) or 0.0))
    target_proxy = np.array([tp_x, tp_y, tp_z], dtype=np.float32)

    cfg_fields = {
        "z_table": float(thr.get("z_table", seg_meta.get("z_table", 0.0) or 0.0)),
        "z_low": float(thr.get("z_low", 0.0)),
        "z_high": float(thr.get("z_high", 0.0)),
        "g_thr": float(thr.get("g_thr", seg_meta.get("g_thr", 0.0) or 0.0)),
        "close_drop_thr": float(cfg.get("close_drop_thr", 0.003)),
        "h_lift": float(cfg.get("h_lift", 0.06)),
        "lift_window": int(cfg.get("lift_window", 30)),
        "closed_frac_min": float(cfg.get("closed_frac_min", 0.7)),
        "nearband_relax": bool(cfg.get("nearband_relax", True)),
    }

    payload = {
        "kind": "grasp_segment_v1",
        "episode_dir": ep_meta.get("episode_dir", ""),
        "language": ep_meta.get("language", ""),
        "t_start": int(seg_meta["t_start"]),
        "t_close": int(seg_meta["t_close"]),
        "t_end": int(seg_meta["t_end"]),
        "t_close_rel": int(t_close),
        "H": H,
        "thresholds": cfg_fields,
        "states": states,
        "actions": actions,
        "derived": {
            "ee_pos": ee_pos,
            "ee_quat": ee_quat,
            "gripper_opening": g_open,
            "target_proxy": target_proxy,
        },
    }
    # Include expansion metadata if available in seg_meta
    exp_keys = ("t_lift_hit", "H_target", "H_pre_dyn", "H_post_dyn", "frames_est")
    exp = {k: seg_meta.get(k) for k in exp_keys if k in seg_meta}
    if exp:
        payload["expansion"] = exp
    return payload
