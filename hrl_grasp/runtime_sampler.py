# hrl_grasp/runtime_sampler.py
from __future__ import annotations
from typing import List, Dict, Tuple, Optional

import os, csv, math, random, time
import numpy as np

from pyrep import PyRep
from pyrep.robots.arms.panda import Panda
from pyrep.robots.end_effectors.panda_gripper import PandaGripper
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.objects.shape import Shape
import imageio

from tools.arrangement_lib import (
    place_objects, remove_all, ensure_dir, XY_BOUNDS, MIN_SEP, TABLE_Z,
)

CATALOG_DEFAULT = [
    "Cup", "Plate", "Knife", "Block", "USBStick", "WineBottle",
    "Shoes", "Sponge", "WateringCan", "PuzzlePiece", "Meat",
]

RLBench_BASE_SCENE = "/home/naren/HRL_part/rlbench/rlbench/task_design.ttt"


def _rng(seed: Optional[int] = None) -> random.Random:
    r = random.Random()
    if seed is not None:
        r.seed(seed)
    return r


def fixed_eval_seeds(n: int = 200, base_seed: int = 12345) -> List[int]:
    return [base_seed + i for i in range(n)]


class RuntimeSampler:
    """Hybrid arrangement scheduler with B sub-episodes per arrangement.

    API for training:
      - env.reset() -> (obs_vec, info)
      - env.step(action_vec[dx,dy,dz,grip]) -> (obs_vec, reward, done, info)
      - env.finish_episode(success: bool)
      - env.eval_mode (bool), env.set_phase(phase_id)
    """

    # Reward parameters
    ALPHA_APPROACH = 2.0
    BETA_GRASP = 0.75
    GAMMA_LIFT = 5.0
    SUCCESS_BONUS = 10.0
    STEP_PENALTY = 0.001
    H0_BIAS = 0.03
    H_SUCCESS = 0.12

    # Action limits
    MAX_DPOS = 0.03
    Z_MIN = TABLE_Z + 0.05
    Z_MAX = 1.20
    X_MIN, X_MAX = XY_BOUNDS[0]
    Y_MIN, Y_MAX = XY_BOUNDS[1]

    def __init__(self,
                 headless: bool = True,
                 episodes_per_arrangement: int = 4,
                 min_objects: int = 3,
                 max_objects: int = 6,
                 catalog_categories: Optional[List[str]] = None,
                 eval_mode: bool = False,
                 fixed_eval_seeds_list: Optional[List[int]] = None,
                 horizon: int = 180,
                 log_csv: Optional[str] = None,
                 camera_res: Tuple[int, int] = (640, 480),
                 internal_steps_per_control: int = 6,
                 debug_steps: bool = False,
                 render_root: str = "/home/naren/HRL_part/rlbench_data/renders",
                 save_scene_images: bool = True,
                 save_eval_images: bool = True,
                 image_debug_prints: bool = True):
        # Config
        self.headless = headless
        self.B_base = episodes_per_arrangement
        self.min_objects = min_objects
        self.max_objects = max_objects
        self.catalog = catalog_categories or list(CATALOG_DEFAULT)
        self.eval_mode = eval_mode
        self.eval_seeds = fixed_eval_seeds_list or fixed_eval_seeds(200)
        self.eval_seed_idx = 0
        self.horizon = horizon
        self.log_csv = log_csv
        self.camera_res = camera_res
        self.n_internal = max(1, int(internal_steps_per_control))
        self.debug_steps = bool(debug_steps)
        self.render_root = render_root
        self.save_scene_images = bool(save_scene_images)
        self.save_eval_images = bool(save_eval_images)
        self.image_debug_prints = bool(image_debug_prints)
        self._phase_override = None
        self._obs_debug_count = 0

        # Simulator
        self.pr = PyRep()
        self.pr.launch(RLBench_BASE_SCENE, headless=headless)
        self.pr.start()
        self.arm = Panda()
        self.gripper = PandaGripper()
        self.tip = self.arm.get_tip()

        # Cameras
        self.cam_over = self._ensure_cam("cam_overhead", [0, 0, 1.30], [0, 0, 0])
        if VisionSensor.exists("cam_wrist"):
            self.cam_wrist = VisionSensor("cam_wrist")
        else:
            self.cam_wrist = VisionSensor.create([self.camera_res[0], self.camera_res[1]])
            try:
                self.cam_wrist.set_parent(self.tip)
                self.cam_wrist.set_position([0.0, 0.0, 0.0], relative_to=self.tip)
                self.cam_wrist.set_orientation([0.0, 0.0, 0.0], relative_to=self.tip)
            except Exception:
                self.cam_wrist.set_position([0, 0, 1.0])
                self.cam_wrist.set_orientation([0, 0, 0])
        self.cam_wrist.set_resolution(self.camera_res)

        # CSV header
        if log_csv:
            ensure_dir(os.path.dirname(log_csv))
            if not os.path.exists(log_csv):
                with open(log_csv, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["arrangement_id", "subep_idx", "mode", "K", "B", "target_pid",
                                "target_category", "success", "steps", "return"])  # episode rows

        # Arrangement-level state
        self.arrangement_id = 0
        self.subep_idx = 0
        self.active_arrangement = False
        self.objects = []  # List[Shape]
        self.project_map = {}
        self.target_queue = []
        self.snapshot_robot_start = {
            "joints": self.arm.get_joint_positions(),
            "grip_open": True,
        }
        self.snapshot_object_poses = {}

        # Episode state
        self.curr_target_pid = None
        self.curr_target_shape = None
        self.prev_dist = 0.0
        self.t_step = 0
        self.ep_return = 0.0
        self.total_subepisodes = 0

        # Summaries
        self.hist_K = {}
        self.cat_counts = {}
        self.success_counts_cat = {}
        self.success_counts_total = 0
        self.reset_times_ms = []
        # Render counters
        self.renders_written_train = 0
        self.renders_written_eval = 0
        self.blank_image_retries = 0
        self.blank_image_fallbacks = 0

    def _ensure_cam(self, name: str, pos, euler) -> VisionSensor:
        if VisionSensor.exists(name):
            cam = VisionSensor(name)
        else:
            cam = VisionSensor.create([self.camera_res[0], self.camera_res[1]])
            cam.set_position(pos)
            cam.set_orientation(euler)
        cam.set_resolution(self.camera_res)
        return cam

    def shutdown(self):
        self.pr.stop(); self.pr.shutdown()

    # --------------------------- Curriculum ---------------------------
    def _phase_for(self, total_subeps: int) -> Tuple[int, int]:
        # Returns K_range (lo, hi), B is locked to K elsewhere
        override = self._phase_override
        if override == 1:
            return (2, 3)
        if override == 2:
            return (3, 4)
        if override == 3:
            return (5, 6)
        if total_subeps <= 500:
            return (2, 3)
        if total_subeps <= 2500:
            return (3, 4)
        return (5, 6)

    # ---------------- Arrangement (spawn) and sub-episode (restore) ---
    def _arrangement_reset(self) -> None:
        # Determine K by curriculum (K-only). B is always set to K.
        K_rng = self._phase_for(self.total_subepisodes)

        # Sample RNG seed
        if self.eval_mode:
            seed = self.eval_seeds[self.arrangement_id % len(self.eval_seeds)]
        else:
            seed = int(time.time() * 1e6) & 0xFFFFFFFF
        rng = _rng(seed)

        # Sample K uniformly in [min_objects, max_objects] intersect phase range
        lo = max(self.min_objects, K_rng[0])
        hi = min(self.max_objects, K_rng[1])
        K = int(rng.randint(lo, hi))

        # Force include a target category globally, then fill remaining without replacement
        target_cat_global = rng.choice(self.catalog)
        pool = [c for c in self.catalog if c != target_cat_global]
        others = rng.sample(pool, k=max(0, K - 1))
        cats = [target_cat_global] + others

        # Spawn with rejection sampling
        attempts = 0
        while True:
            attempts += 1
            try:
                placed, _mapping = place_objects(cats)
                if len(placed) != K:
                    raise RuntimeError("spawn count mismatch")
                break
            except Exception:
                if attempts >= 5:
                    raise
                continue

        # Record objects and mapping
        self.objects = [obj for (_pid, _cat, obj, _pos) in placed]
        self.project_map = {}
        for pid, cat, obj, pos in placed:
            self.project_map[pid] = {
                "category": cat,
                "handle": int(obj.get_handle()),
                "position": list(obj.get_position()),
                "orientation": list(obj.get_orientation()),
            }

        # Lock B to K and build target queue
        B_effective = K
        if self.eval_mode:
            # Deterministic order 0..K-1
            self.target_queue = list(range(K))
        else:
            # Random permutation of all targets
            pids = list(range(K))
            rng.shuffle(pids)
            self.target_queue = pids

        # Snapshots
        self.snapshot_robot_start = {
            "joints": self.arm.get_joint_positions(),
            "grip_open": True,
        }
        self.snapshot_object_poses = {
            pid: {
                "position": list(Shape(self.project_map[pid]["handle"]).get_position()),
                "orientation": list(Shape(self.project_map[pid]["handle"]).get_orientation())
            } for pid in self.project_map
        }

        # Small settle before capture
        for _ in range(3):
            self.pr.step()

        # Scene-wise image capture (once per arrangement)
        try:
            self._capture_scene_images(arrangement_id=self.arrangement_id + 1, mode=('eval' if self.eval_mode else 'train'))
        except Exception as e:
            print(f"[WARN] IMAGE_CAPTURE_FAILED err={e}")

        # Update arrangement meta
        self.subep_idx = 0
        self.active_arrangement = True
        self.arrangement_id += 1

        # Logging per object
        print(f"[ARR {self.arrangement_id:06d}] mode={'eval' if self.eval_mode else 'train'} K={K} B={B_effective}")
        for pid, data in self.project_map.items():
            pos = data["position"]; ori = data["orientation"]; yaw = float(ori[2]) if len(ori) >= 3 else 0.0
            print(f"  PID={pid:02d} CAT={data['category']:12s} HANDLE={data['handle']} POS=({pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:+.3f}) YAW={yaw:+.3f}")

        # Bookkeeping
        self.hist_K[K] = self.hist_K.get(K, 0) + 1
        for pid, data in self.project_map.items():
            self.cat_counts[data["category"]] = self.cat_counts.get(data["category"], 0) + 1

        self._K_current = K
        self._B_current = B_effective

    def _subepisode_reset(self) -> Tuple[np.ndarray, Dict]:
        # Restore robot
        try:
            self.arm.set_joint_positions(self.snapshot_robot_start["joints"])  # type: ignore[index]
            self.gripper.release()
        except Exception:
            pass

        # Restore objects
        invalid = False
        for pid, data in self.snapshot_object_poses.items():
            try:
                sh = Shape(self.project_map[pid]["handle"])  # re-wrap
                sh.set_position(data["position"])  # type: ignore[index]
                sh.set_orientation(data["orientation"])  # type: ignore[index]
                try:
                    sh.set_velocity([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
                except Exception:
                    pass
            except Exception:
                invalid = True
        for _ in range(6):
            self.pr.step()
        if invalid:
            # Force a new arrangement
            print("[WARN] Invalid handle during restore; respawning arrangement.")
            self.active_arrangement = False
            return self.reset()

        # Select current target
        self.curr_target_pid = int(self.target_queue[self.subep_idx])
        self.curr_target_shape = Shape(self.project_map[self.curr_target_pid]["handle"])  # type: ignore[index]

        # Initialize episode accumulators
        self.t_step = 0
        self.ep_return = 0.0
        self.prev_dist = self._ee_target_dist()

        # Log target
        cat = self.project_map[self.curr_target_pid]["category"]  # type: ignore[index]
        print(f"TARGET ARR={self.arrangement_id:06d} SUB={self.subep_idx}/{self._B_current} PID={self.curr_target_pid} CAT={cat}")

        obs, info = self.build_observation()
        return obs, info

    # --------------------------- Public API ---------------------------
    def reset(self) -> Tuple[np.ndarray, Dict]:
        t0 = time.time()
        if (not self.active_arrangement) or (self.subep_idx >= getattr(self, "_B_current", self.B_base)):
            self._arrangement_reset()
        obs, info = self._subepisode_reset()
        t1 = time.time()
        self.reset_times_ms.append((t1 - t0) * 1000.0)
        self.total_subepisodes += 1
        if self.total_subepisodes % 100 == 0:
            self._print_summaries()
        # Debug: feature min/max for first few episodes
        if self._obs_debug_count < 10:
            self._obs_debug_count += 1
            vmin = float(np.min(obs))
            vmax = float(np.max(obs))
            print(f"[OBS DEBUG] dim={obs.shape[0]} min={vmin:.3f} max={vmax:.3f}")
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        # Parse action
        dx, dy, dz, grip = [float(x) for x in action]
        # Clamp delta
        clamped = {"dx": False, "dy": False, "dz": False}
        if dx < -self.MAX_DPOS or dx > self.MAX_DPOS:
            clamped["dx"] = True
        if dy < -self.MAX_DPOS or dy > self.MAX_DPOS:
            clamped["dy"] = True
        if dz < -self.MAX_DPOS or dz > self.MAX_DPOS:
            clamped["dz"] = True
        dx = max(-self.MAX_DPOS, min(self.MAX_DPOS, dx))
        dy = max(-self.MAX_DPOS, min(self.MAX_DPOS, dy))
        dz = max(-self.MAX_DPOS, min(self.MAX_DPOS, dz))

        # Move EE by IK
        ee_pos = list(self.tip.get_position())
        tgt_pos = [ee_pos[0] + dx, ee_pos[1] + dy, ee_pos[2] + dz]
        # Workspace clamps
        before_ws = list(tgt_pos)
        tgt_pos[0] = max(self.X_MIN, min(self.X_MAX, tgt_pos[0]))
        tgt_pos[1] = max(self.Y_MIN, min(self.Y_MAX, tgt_pos[1]))
        tgt_pos[2] = max(self.Z_MIN, min(self.Z_MAX, tgt_pos[2]))
        if self.debug_steps:
            for k, was in clamped.items():
                if was:
                    print(f"[CLAMP] {k} to ±{self.MAX_DPOS}")
            if before_ws != tgt_pos:
                print(f"[CLAMP] workspace {before_ws} -> {tgt_pos}")

        try:
            j = self.arm.solve_ik(position=tgt_pos, euler=[0.0, math.pi, 0.0])
            self.arm.set_joint_positions(j)
        except Exception:
            pass

        # Gripper logic and auto-grasp when close
        grasp_flag = False
        if grip > 0.0:
            try:
                if self.curr_target_shape is not None:
                    tpos = self.curr_target_shape.get_position()
                    ee = self.tip.get_position()
                    if math.dist([ee[0], ee[1], ee[2]], [tpos[0], tpos[1], tpos[2]]) < 0.025:
                        self.gripper.grasp(self.curr_target_shape)
                        grasp_flag = True
            except Exception:
                pass
        else:
            self.gripper.release()

        # Step physics
        for _ in range(self.n_internal):
            self.pr.step()

        # Rewards
        dist = self._ee_target_dist()
        d_dist = self.prev_dist - dist
        r_approach = self.ALPHA_APPROACH * d_dist
        # Held proxy: object very close to tip and off-table
        held_proxy = False
        try:
            if self.curr_target_shape is not None:
                tpos = self.curr_target_shape.get_position()
                ee = self.tip.get_position()
                near = math.dist([ee[0], ee[1], ee[2]], [tpos[0], tpos[1], tpos[2]]) < 0.03
                held_proxy = bool(near and (tpos[2] > (TABLE_Z + self.H0_BIAS)))
        except Exception:
            held_proxy = False
        r_grasp = self.BETA_GRASP if (grasp_flag or held_proxy) and (grip > 0.0) else 0.0
        z_obj = self._target_z()
        r_lift = self.GAMMA_LIFT * max(0.0, z_obj - (TABLE_Z + self.H0_BIAS))
        success = z_obj > (TABLE_Z + self.H_SUCCESS)
        r_success = self.SUCCESS_BONUS if success else 0.0
        r = r_approach + r_grasp + r_lift + r_success - self.STEP_PENALTY

        # Update state
        self.prev_dist = dist
        self.t_step += 1
        self.ep_return += r
        done = success or (self.t_step >= self.horizon)

        obs, info = self.build_observation()
        info.update(dict(
            dist=dist, d_dist=d_dist, grasp_flag=grasp_flag, held=held_proxy, z_obj=z_obj,
            r_approach=r_approach, r_grasp=r_grasp, r_lift=r_lift, r_success=r_success,
            success=success,
        ))
        # Safety: NaN/Inf check
        if not np.isfinite([r, dist, d_dist, z_obj]).all():
            print("[WARN] NaN/Inf encountered. Aborting episode.")
            done = True
        if self.debug_steps:
            print({"dist": dist, "d_dist": d_dist, "grasp_flag": grasp_flag, "z_obj": z_obj, "reward": float(r)})
        return obs, float(r), bool(done), info

    def finish_episode(self, success: bool) -> None:
        # Episode CSV row
        if self.log_csv is not None and self.curr_target_pid is not None:
            with open(self.log_csv, "a", newline="") as f:
                w = csv.writer(f)
                cat = self.project_map[self.curr_target_pid]["category"]
                w.writerow([self.arrangement_id, self.subep_idx, 'eval' if self.eval_mode else 'train',
                            getattr(self, "_K_current", 0), getattr(self, "_B_current", 0),
                            self.curr_target_pid, cat, int(bool(success)), self.t_step, self.ep_return])

        # Success stats
        if self.curr_target_pid is not None:
            cat = self.project_map[self.curr_target_pid]["category"]
            if success:
                self.success_counts_total += 1
                self.success_counts_cat[cat] = self.success_counts_cat.get(cat, 0) + 1

        # Advance sub-episode pointer; cleanup if arrangement exhausted
        self.subep_idx += 1
        if self.subep_idx >= getattr(self, "_B_current", self.B_base):
            # Remove objects and deactivate arrangement
            if self.objects:
                try:
                    remove_all([(pid, self.project_map[pid]["category"], Shape(self.project_map[pid]["handle"]), None) for pid in self.project_map])
                except Exception:
                    pass
            self.active_arrangement = False
        # else: remain active and reuse on next reset

    # ----------------------- Observation & helpers --------------------
    def build_observation(self) -> Tuple[np.ndarray, Dict]:
        # EE pose
        ee_p = np.array(self.tip.get_position(), dtype=np.float32)
        ee_q = np.array(self.tip.get_quaternion(), dtype=np.float32)  # xyzw
        # Gripper open scalar (approx): 1.0=open, 0.0=closed
        grip_open = 1.0  # simple proxy
        # Target pose
        if self.curr_target_shape is not None:
            t_p = np.array(self.curr_target_shape.get_position(), dtype=np.float32)
        else:
            t_p = np.zeros(3, dtype=np.float32)
        rel = t_p - ee_p

        # Clip to workspace (documented scaling)
        ee_p[0] = np.clip(ee_p[0], self.X_MIN, self.X_MAX)
        ee_p[1] = np.clip(ee_p[1], self.Y_MIN, self.Y_MAX)
        ee_p[2] = np.clip(ee_p[2], self.Z_MIN, self.Z_MAX)

        obs = np.concatenate([ee_p, ee_q, np.array([grip_open], dtype=np.float32), t_p, rel], axis=0)
        if not np.isfinite(obs).all():
            print("[WARN] Non-finite obs detected; sanitizing.")
            obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        info = {
            "obs_dim": int(obs.shape[0]),
            "ee_pos": ee_p.tolist(),
            "ee_quat": ee_q.tolist(),
            "grip_open": float(grip_open),
            "target_pos": t_p.tolist(),
            "rel_pos": rel.tolist(),
            "dist": float(self._ee_target_dist()),
            "z_obj": float(self._target_z()),
        }
        return obs.astype(np.float32), info

    def _ee_target_dist(self) -> float:
        if self.curr_target_shape is None:
            return 0.0
        ee = self.tip.get_position()
        tp = self.curr_target_shape.get_position()
        return float(math.dist([ee[0], ee[1], ee[2]], [tp[0], tp[1], tp[2]]))

    def _target_z(self) -> float:
        if self.curr_target_shape is None:
            return TABLE_Z
        return float(self.curr_target_shape.get_position()[2])

    def _print_summaries(self) -> None:
        n = self.total_subepisodes
        avg_ms = (sum(self.reset_times_ms[-100:]) / max(1, len(self.reset_times_ms[-100:]))) if self.reset_times_ms else 0.0
        cats = {k: self.cat_counts.get(k, 0) for k in sorted(self.cat_counts.keys())}
        succ_cats = {k: self.success_counts_cat.get(k, 0) for k in sorted(self.success_counts_cat.keys())}
        print("[SUMMARY] subeps=%d avg_reset_ms=%.1f K_hist=%s succ_total=%d cat_counts=%s succ_by_cat=%s renders_train=%d renders_eval=%d blank_retries=%d blank_fallbacks=%d" % (
            n, avg_ms, dict(sorted(self.hist_K.items())), self.success_counts_total, cats, succ_cats,
            self.renders_written_train, self.renders_written_eval, self.blank_image_retries, self.blank_image_fallbacks))

    # ----------------------- Control APIs ----------------------------
    def set_phase(self, phase_id: Optional[int]) -> None:
        """Force curriculum phase: 1->(2-3), 2->(3-4), 3->(5-6), None->auto."""
        self._phase_override = phase_id

    # ----------------------- Image capture helpers -------------------
    def _sensor_capture_rgb(self, sensor: VisionSensor) -> Optional[np.ndarray]:
        try:
            img = sensor.capture_rgb()  # [H,W,3] float [0,1]
            if img is None:
                return None
            if not np.isfinite(img).all():
                return None
            return img
        except Exception:
            return None

    def _save_png(self, out_path: str, img_rgb_float: np.ndarray) -> bool:
        try:
            img = np.clip(img_rgb_float, 0.0, 1.0)
            img8 = (img * 255).astype(np.uint8)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            try:
                import cv2  # type: ignore
                cv2.imwrite(out_path, img8[..., ::-1])
            except Exception:
                imageio.v2.imwrite(out_path, img8)
            return True
        except Exception as e:
            print(f"[WARN] SAVE_FAILED path={out_path} err={e}")
            return False

    def _capture_with_guard(self, sensor: VisionSensor) -> Tuple[Optional[np.ndarray], bool]:
        img = self._sensor_capture_rgb(sensor)
        if img is None:
            return None, False
        mean_intensity = float(np.mean(img) * 255.0)
        if mean_intensity < 5.0:
            # Retry after a few physics steps
            for _ in range(3):
                self.pr.step()
            self.blank_image_retries += 1
            img2 = self._sensor_capture_rgb(sensor)
            if img2 is None:
                self.blank_image_fallbacks += 1
                print("[WARN] BLANK_IMAGE (None) after retry; continuing.")
                return None, True
            if float(np.mean(img2) * 255.0) < 5.0:
                self.blank_image_fallbacks += 1
                print("[WARN] BLANK_IMAGE after retry; continuing.")
                return None, True
            return img2, True
        return img, False

    def _capture_scene_images(self, arrangement_id: int, mode: str) -> None:
        # Toggle checks
        if not self.save_scene_images:
            return
        if self.eval_mode and not self.save_eval_images:
            return
        # Ensure sensors are present
        if self.cam_over is None:
            self.cam_over = self._ensure_cam("cam_overhead", [0, 0, 1.30], [0, 0, 0])
        if self.cam_wrist is None:
            if VisionSensor.exists("cam_wrist"):
                self.cam_wrist = VisionSensor("cam_wrist")
            else:
                self.cam_wrist = VisionSensor.create([self.camera_res[0], self.camera_res[1]])
                try:
                    self.cam_wrist.set_parent(self.tip)
                    self.cam_wrist.set_position([0.0, 0.0, 0.0], relative_to=self.tip)
                    self.cam_wrist.set_orientation([0.0, 0.0, 0.0], relative_to=self.tip)
                except Exception:
                    self.cam_wrist.set_position([0, 0, 1.0])
                    self.cam_wrist.set_orientation([0, 0, 0])
            self.cam_wrist.set_resolution(self.camera_res)

        subdir = "eval" if self.eval_mode else "train"
        base = os.path.join(self.render_root, subdir, f"arr_{arrangement_id:06d}")
        oh_path = os.path.join(base, "overhead.png")
        wr_path = os.path.join(base, "wrist.png")

        # Overhead
        img_oh, retried_oh = self._capture_with_guard(self.cam_over)
        if img_oh is not None:
            if self._save_png(oh_path, img_oh):
                if self.image_debug_prints:
                    print(f"SAVED IMG mode={subdir} ARR={arrangement_id:06d} path={oh_path}")
                if self.eval_mode:
                    self.renders_written_eval += 1
                else:
                    self.renders_written_train += 1

        # Wrist
        img_wr, retried_wr = self._capture_with_guard(self.cam_wrist)
        if img_wr is not None:
            if self._save_png(wr_path, img_wr):
                if self.image_debug_prints:
                    print(f"SAVED IMG mode={subdir} ARR={arrangement_id:06d} path={wr_path}")
                if self.eval_mode:
                    self.renders_written_eval += 1
                else:
                    self.renders_written_train += 1
