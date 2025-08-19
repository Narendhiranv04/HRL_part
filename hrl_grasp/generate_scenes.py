#!/usr/bin/env python3
"""
generate_scenes.py

Create additional "scene_*" datasets for PickAndLift by randomizing object
positions and injecting extra objects imported via PyRep from CoppeliaSim models.

Output directory structure (one extra scene_ level):
  <save_path>/pick_and_lift/scene_0001/variation0/episodes/episode0/<views>/*.png

Example:
    python generate_scenes.py \
        --save_path /home/naren/HRL_part/rlbench_data \
        --scenes 3 --variations 1 \
        --min_objects 3 --max_objects 5 \
        --categories Cup Plate Knife Block "USB Stick" "Wine Bottle" Shoes Sponge "Watering Can" "Puzzle Piece" \
        --renderer opengl --image_size 128 128

Notes:
- We DO NOT replace the target block (to keep success conditions intact).
- We add distractors (e.g., cups, small household items) placed within the
  task's boundary using SpawnBoundary sampling.
"""
import argparse
import os
import random
import sys
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import numpy as np
from PIL import Image
from pyrep.const import RenderMode
from pyrep.objects.shape import Shape

from rlbench import ObservationConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.backend.spawn_boundary import SpawnBoundary
from rlbench.backend.const import (
    LEFT_SHOULDER_RGB_FOLDER, LEFT_SHOULDER_DEPTH_FOLDER, LEFT_SHOULDER_MASK_FOLDER,
    RIGHT_SHOULDER_RGB_FOLDER, RIGHT_SHOULDER_DEPTH_FOLDER, RIGHT_SHOULDER_MASK_FOLDER,
    OVERHEAD_RGB_FOLDER, OVERHEAD_DEPTH_FOLDER, OVERHEAD_MASK_FOLDER,
    WRIST_RGB_FOLDER, WRIST_DEPTH_FOLDER, WRIST_MASK_FOLDER,
    FRONT_RGB_FOLDER, FRONT_DEPTH_FOLDER, FRONT_MASK_FOLDER,
    DEPTH_SCALE, VARIATIONS_FOLDER, VARIATION_DESCRIPTIONS,
    EPISODES_FOLDER, EPISODE_FOLDER, LOW_DIM_PICKLE)
from rlbench.backend import utils as rb_utils
from rlbench.environment import Environment
from rlbench.tasks.pick_and_lift import PickAndLift
from rlbench.backend.conditions import DetectedCondition, ConditionSet, GraspedCondition

# Desired categories in RLBench
DEFAULT_CATEGORIES = [
    'Cup', 'Plate', 'Knife', 'Block', 'USB Stick', 'Wine Bottle',
    'Shoes', 'Sponge', 'Watering Can', 'Puzzle Piece'
]
from rlb_object_library import ensure_assets


def check_and_make(dir_path: str):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def _slugify(text: str) -> str:
    return ''.join(c.lower() if c.isalnum() else '_' for c in text).strip('_')


def _resolve_rlb_assets(categories: List[str], headless: bool = True) -> Dict[str, Path]:
    mapping = ensure_assets(headless=headless)
    # Filter to requested categories only
    out: Dict[str, Path] = {}
    for c in categories:
        if c in mapping:
            out[c] = Path(mapping[c])
    return out


def _choose_balanced(categories: List[str], per_scene_n: int, cat_cursor: int) -> Tuple[List[str], int]:
    chosen = []
    for i in range(per_scene_n):
        chosen.append(categories[(cat_cursor + i) % len(categories)])
    return chosen, (cat_cursor + per_scene_n) % len(categories)


def _save_demo_to(example_path: str, demo) -> None:
    # Based on rlbench/rlbench/dataset_generator.py:save_demo
    left_shoulder_rgb_path = os.path.join(example_path, LEFT_SHOULDER_RGB_FOLDER)
    left_shoulder_depth_path = os.path.join(example_path, LEFT_SHOULDER_DEPTH_FOLDER)
    left_shoulder_mask_path = os.path.join(example_path, LEFT_SHOULDER_MASK_FOLDER)
    right_shoulder_rgb_path = os.path.join(example_path, RIGHT_SHOULDER_RGB_FOLDER)
    right_shoulder_depth_path = os.path.join(example_path, RIGHT_SHOULDER_DEPTH_FOLDER)
    right_shoulder_mask_path = os.path.join(example_path, RIGHT_SHOULDER_MASK_FOLDER)
    overhead_rgb_path = os.path.join(example_path, OVERHEAD_RGB_FOLDER)
    overhead_depth_path = os.path.join(example_path, OVERHEAD_DEPTH_FOLDER)
    overhead_mask_path = os.path.join(example_path, OVERHEAD_MASK_FOLDER)
    wrist_rgb_path = os.path.join(example_path, WRIST_RGB_FOLDER)
    wrist_depth_path = os.path.join(example_path, WRIST_DEPTH_FOLDER)
    wrist_mask_path = os.path.join(example_path, WRIST_MASK_FOLDER)
    front_rgb_path = os.path.join(example_path, FRONT_RGB_FOLDER)
    front_depth_path = os.path.join(example_path, FRONT_DEPTH_FOLDER)
    front_mask_path = os.path.join(example_path, FRONT_MASK_FOLDER)

    for p in [left_shoulder_rgb_path, left_shoulder_depth_path, left_shoulder_mask_path,
              right_shoulder_rgb_path, right_shoulder_depth_path, right_shoulder_mask_path,
              overhead_rgb_path, overhead_depth_path, overhead_mask_path,
              wrist_rgb_path, wrist_depth_path, wrist_mask_path,
              front_rgb_path, front_depth_path, front_mask_path]:
        check_and_make(p)

    for i, obs in enumerate(demo):
        left_shoulder_rgb = Image.fromarray(obs.left_shoulder_rgb)
        left_shoulder_depth = rb_utils.float_array_to_rgb_image(obs.left_shoulder_depth, scale_factor=DEPTH_SCALE)
        left_shoulder_mask = Image.fromarray((obs.left_shoulder_mask * 255).astype(np.uint8))
        right_shoulder_rgb = Image.fromarray(obs.right_shoulder_rgb)
        right_shoulder_depth = rb_utils.float_array_to_rgb_image(obs.right_shoulder_depth, scale_factor=DEPTH_SCALE)
        right_shoulder_mask = Image.fromarray((obs.right_shoulder_mask * 255).astype(np.uint8))
        overhead_rgb = Image.fromarray(obs.overhead_rgb)
        overhead_depth = rb_utils.float_array_to_rgb_image(obs.overhead_depth, scale_factor=DEPTH_SCALE)
        overhead_mask = Image.fromarray((obs.overhead_mask * 255).astype(np.uint8))
        wrist_rgb = Image.fromarray(obs.wrist_rgb)
        wrist_depth = rb_utils.float_array_to_rgb_image(obs.wrist_depth, scale_factor=DEPTH_SCALE)
        wrist_mask = Image.fromarray((obs.wrist_mask * 255).astype(np.uint8))
        front_rgb = Image.fromarray(obs.front_rgb)
        front_depth = rb_utils.float_array_to_rgb_image(obs.front_depth, scale_factor=DEPTH_SCALE)
        front_mask = Image.fromarray((obs.front_mask * 255).astype(np.uint8))

        fmt = '%d.png'
        left_shoulder_rgb.save(os.path.join(left_shoulder_rgb_path, fmt % i))
        left_shoulder_depth.save(os.path.join(left_shoulder_depth_path, fmt % i))
        left_shoulder_mask.save(os.path.join(left_shoulder_mask_path, fmt % i))
        right_shoulder_rgb.save(os.path.join(right_shoulder_rgb_path, fmt % i))
        right_shoulder_depth.save(os.path.join(right_shoulder_depth_path, fmt % i))
        right_shoulder_mask.save(os.path.join(right_shoulder_mask_path, fmt % i))
        overhead_rgb.save(os.path.join(overhead_rgb_path, fmt % i))
        overhead_depth.save(os.path.join(overhead_depth_path, fmt % i))
        overhead_mask.save(os.path.join(overhead_mask_path, fmt % i))
        wrist_rgb.save(os.path.join(wrist_rgb_path, fmt % i))
        wrist_depth.save(os.path.join(wrist_depth_path, fmt % i))
        wrist_mask.save(os.path.join(wrist_mask_path, fmt % i))
        front_rgb.save(os.path.join(front_rgb_path, fmt % i))
        front_depth.save(os.path.join(front_depth_path, fmt % i))
        front_mask.save(os.path.join(front_mask_path, fmt % i))

        # Set large image arrays to None before pickling
        obs.left_shoulder_rgb = None
        obs.left_shoulder_depth = None
        obs.left_shoulder_point_cloud = None
        obs.left_shoulder_mask = None
        obs.right_shoulder_rgb = None
        obs.right_shoulder_depth = None
        obs.right_shoulder_point_cloud = None
        obs.right_shoulder_mask = None
        obs.overhead_rgb = None
        obs.overhead_depth = None
        obs.overhead_point_cloud = None
        obs.overhead_mask = None
        obs.wrist_rgb = None
        obs.wrist_depth = None
        obs.wrist_point_cloud = None
        obs.wrist_mask = None
        obs.front_rgb = None
        obs.front_depth = None
        obs.front_point_cloud = None
        obs.front_mask = None

    # Save low-dim data
    import pickle
    with open(os.path.join(example_path, LOW_DIM_PICKLE), 'wb') as f:
        pickle.dump(demo, f)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--save_path', required=True, help='Root where scene_* datasets will be stored')
    ap.add_argument('--scenes', type=int, default=3, help='Number of scene_* groups to generate')
    ap.add_argument('--episodes_per_scene', type=int, default=-1, help='Episodes per scene; -1 => equals number of placed objects')
    ap.add_argument('--variations', type=int, default=1, help='Variations per scene (<= task.variation_count)')
    ap.add_argument('--min_objects', type=int, default=3, help='Min objects per scene')
    ap.add_argument('--max_objects', type=int, default=5, help='Max objects per scene')
    ap.add_argument('--categories', nargs='+', default=DEFAULT_CATEGORIES, help='Desired object categories to sample from')
    ap.add_argument('--image_size', nargs=2, type=int, default=[128, 128], help='Image size W H')
    ap.add_argument('--renderer', choices=['opengl', 'opengl3'], default='opengl3', help='Renderer')
    ap.add_argument('--headless', action='store_true', default=True, help='Headless mode for CoppeliaSim')
    ap.add_argument('--models_root', type=str, default='', help='Override path to CoppeliaSim models dir (defaults to $COPPELIASIM_ROOT/models)')
    ap.add_argument('--arm_max_velocity', type=float, default=1.0)
    ap.add_argument('--arm_max_acceleration', type=float, default=4.0)
    ap.add_argument('--distractor_scale', type=float, default=0.85, help='Uniform scale to apply to imported distractors.')
    ap.add_argument('--min_distance', type=float, default=0.30, help='Min planar distance between placed distractors and others.')
    ap.add_argument('--name_episodes_by_object', action='store_true', default=True, help='Name episode folders with category slug.')
    ap.add_argument('--make_named_videos', action='store_true', default=True, help='Export wrist/overhead MP4s named scene and object.')
    ap.add_argument('--objects_per_scene', type=int, default=4, help='Number of objects to place per scene.')
    ap.add_argument('--all_categories_per_scene', action='store_true', default=False, help='Place all requested categories in every scene.')
    ap.add_argument('--simple_scene_names', action='store_true', default=True, help='Use 1-based scene names (scene_1, scene_2, ...)')
    ap.add_argument('--per_episode_single_target', action='store_true', default=False, help='Only place the intended object for each episode (if True). If False, keep all scene objects present each episode.')
    ap.add_argument('--fixed_layout', action='store_true', default=True, help='Place objects at fixed safe offsets around the target to maximize stability.')
    return ap.parse_args()


def main():
    args = parse_args()

    save_root = Path(args.save_path)
    save_root.mkdir(parents=True, exist_ok=True)

    # Build RLBench single-object asset library
    requested_cats = args.categories
    asset_map = _resolve_rlb_assets(requested_cats, headless=args.headless)
    available_cats = [c for c in requested_cats if c in asset_map]
    missing_cats = [c for c in requested_cats if c not in asset_map]
    if missing_cats:
        print('[ERR] Missing RLBench assets for categories:', missing_cats)
        sys.exit(2)

    # Observation configuration similar to dataset_generator
    img_size = list(map(int, args.image_size))
    obs_config = ObservationConfig()
    obs_config.set_all(True)
    obs_config.right_shoulder_camera.image_size = img_size
    obs_config.left_shoulder_camera.image_size = img_size
    obs_config.overhead_camera.image_size = img_size
    obs_config.wrist_camera.image_size = img_size
    obs_config.front_camera.image_size = img_size
    for cam in [obs_config.right_shoulder_camera,
                obs_config.left_shoulder_camera,
                obs_config.overhead_camera,
                obs_config.wrist_camera,
                obs_config.front_camera]:
        cam.depth_in_meters = False
        cam.masks_as_one_channel = False

    if args.renderer == 'opengl':
        rm = RenderMode.OPENGL
    else:
        rm = RenderMode.OPENGL3
    obs_config.right_shoulder_camera.render_mode = rm
    obs_config.left_shoulder_camera.render_mode = rm
    obs_config.overhead_camera.render_mode = rm
    obs_config.wrist_camera.render_mode = rm
    obs_config.front_camera.render_mode = rm

    env = Environment(
        action_mode=MoveArmThenGripper(JointVelocity(), Discrete()),
        obs_config=obs_config,
        headless=args.headless,
        arm_max_velocity=args.arm_max_velocity,
        arm_max_acceleration=args.arm_max_acceleration,
    )
    env.launch()

    try:
        task_env = env.get_task(PickAndLift)
        max_var = min(args.variations, task_env.variation_count())

        cat_cursor = 0
        for s in range(args.scenes):
            scene_idx = s + 1 if args.simple_scene_names else s
            scene_name = f'scene_{scene_idx}' if args.simple_scene_names else f'scene_{s:04d}'
            print(f'\n[SCENE] {scene_name}')

            for var in range(max_var):
                task_env.set_variation(var)
                var_root = save_root / task_env.get_name() / scene_name / (VARIATIONS_FOLDER % var)
                var_root.mkdir(parents=True, exist_ok=True)

                # Save descriptions for this variation (robust retries)
                reset_ok = False
                last_err: Optional[Exception] = None
                for _try in range(10):
                    try:
                        desc, _ = task_env.reset()
                        reset_ok = True
                        break
                    except Exception as e:
                        last_err = e
                        print(f"[WARN] Reset failed (try {_try+1}/10): {e}")
                        continue
                if not reset_ok:
                    print(f"[SKIP] Could not reset variation {var} for {scene_name}: {last_err}")
                    continue
                import pickle
                with open(var_root / VARIATION_DESCRIPTIONS, 'wb') as f:
                    pickle.dump(desc, f)

                episodes_path = var_root / EPISODES_FOLDER
                episodes_path.mkdir(parents=True, exist_ok=True)

                # Decide number of objects and categories for this scene
                if args.all_categories_per_scene:
                    chosen_cats = list(available_cats)
                else:
                    # Choose exactly objects_per_scene categories; allow replacement if needed
                    k = max(1, int(args.objects_per_scene))
                    if len(available_cats) == 0:
                        chosen_cats = []
                    elif k <= len(available_cats):
                        chosen_cats = random.sample(available_cats, k)
                    else:
                        chosen_cats = random.choices(available_cats, k=k)

                # Resolve model files for each chosen category (from RLBench assets)
                chosen_models: List[Tuple[str, Path]] = []
                for cat in chosen_cats:
                    mp = asset_map.get(cat)
                    if mp is not None:
                        chosen_models.append((cat, mp))

                # We'll import/place per-episode after each reset to avoid conflicts with task resets
                # Ensure one episode per object when not overridden
                episodes_this_scene = args.episodes_per_scene if args.episodes_per_scene > 0 else len(chosen_models)

                # Create episodes; rotate intended target index across objects
                for ep in range(episodes_this_scene):

                    # Reset robot/task (fresh task scene)
                    ok = False
                    for _try in range(4):
                        try:
                            task_env.reset()
                            ok = True
                            break
                        except Exception as e:
                            print(f'[WARN] Reset before episode failed (try {_try+1}/4): {e}')
                    if not ok:
                        print('[SKIP] Episode skipped due to repeated reset failure.')
                        continue

                    # Decide intended index relative to chosen_models for this scene
                    intended_idx_global = ep % max(1, len(chosen_models)) if len(chosen_models) > 0 else 0

                    # Import and place objects for this episode
                    # Track (category, base_obj, shape, model_path)
                    placed_objs: List[Tuple[str, object, Shape, str]] = []
                    models_this_ep = chosen_models if not args.per_episode_single_target else (
                        [chosen_models[intended_idx_global]] if len(chosen_models) > 0 else []
                    )
                    # Recreate boundary after reset to avoid stale handles
                    boundary = None
                    try:
                        boundary_shape = Shape('pick_and_lift_boundary')
                        boundary = SpawnBoundary([boundary_shape])
                    except Exception:
                        boundary = None

                    for cat, mp in models_this_ep:
                        try:
                            base_obj = task_env._scene.pyrep.import_model(str(mp))
                            shp = Shape(base_obj.get_handle())
                            # Optional scale down to ease planning
                            try:
                                s = max(0.2, float(args.distractor_scale))
                                shp.set_scale([s, s, s])
                            except Exception:
                                pass
                            placed = False
                            if args.fixed_layout:
                                try:
                                    # Deterministic offsets around the target block
                                    t_shape = Shape('pick_and_lift_target')
                                    tpos = t_shape.get_position()
                                    # 4 slots forming a cross around target (x,y offsets in meters)
                                    slots_xy = [(0.16, 0.00), (-0.16, 0.00), (0.00, 0.16), (0.00, -0.16)]
                                    # Choose next free slot by count so far
                                    idx = len(placed_objs) % len(slots_xy)
                                    dx, dy = slots_xy[idx]
                                    shp.set_position([tpos[0] + dx, tpos[1] + dy, tpos[2] + 0.005])
                                    # Upright with random yaw
                                    try:
                                        yaw = random.uniform(-3.14, 3.14)
                                        shp.set_orientation([0.0, 0.0, yaw])
                                    except Exception:
                                        pass
                                    placed = True
                                except Exception:
                                    placed = False
                            if not placed:
                                # Fallback to sampling within boundary
                                for _place_try in range(20):
                                    try:
                                        if boundary is not None:
                                            boundary.sample(
                                                shp,
                                                ignore_collisions=False,
                                                min_distance=max(0.1, float(args.min_distance)),
                                                min_rotation=(0, 0, -3.14),
                                                max_rotation=(0, 0, 3.14),
                                            )
                                        else:
                                            raise RuntimeError('Boundary unavailable')
                                        # Slightly lift above table to avoid micro-collisions
                                        try:
                                            pos = shp.get_position()
                                            shp.set_position([pos[0], pos[1], pos[2] + 0.005])
                                        except Exception:
                                            pass
                                        placed = True
                                        break
                                    except Exception:
                                        continue
                                if not placed:
                                    raise RuntimeError('Failed to place after multiple attempts.')
                            placed_objs.append((cat, base_obj, shp, str(mp)))
                        except Exception as e:
                            print(f'[WARN] Failed to place {cat} from {mp}: {e}')
                            try:
                                base_obj.remove()
                            except Exception:
                                pass
                            continue

                    # Step simulation a few times to stabilize imported shapes
                    try:
                        for _ in range(3):
                            task_env._scene.pyrep.step()
                    except Exception:
                        pass

                    print(f'[SCENE {scene_name}] variation {var} episode {ep} (objects={len(placed_objs)})')

                    # Determine intended object within placed list and prepare scene
                    intended_idx = ep % max(1, len(placed_objs))
                    target_shape = None
                    try:
                        target_shape = Shape('pick_and_lift_target')
                    except Exception:
                        target_shape = None
                    # Disable physics for distractors; attach intended to target and hide block visual
                    for j, (_c, _base, shp, _mp) in enumerate(placed_objs):
                        if j == intended_idx:
                            # Intended: ensure visible and attached to the block so it moves with expert
                            if target_shape is not None:
                                try:
                                    shp.set_parent(target_shape, keep_in_place=True)  # type: ignore[arg-type]
                                except Exception:
                                    pass
                                try:
                                    target_shape.set_renderable(False)
                                except Exception:
                                    pass
                            try:
                                shp.set_dynamic(True)
                            except Exception:
                                pass
                            try:
                                shp.set_respondable(True)
                            except Exception:
                                pass
                        else:
                            # Distractors: disable physics/collisions
                            for fn_name in ('set_dynamic', 'set_respondable', 'set_collidable', 'set_measurable', 'set_detectable'):
                                try:
                                    fn = getattr(shp, fn_name, None)
                                    if callable(fn):
                                        fn(False)
                                except Exception:
                                    pass

                    # Keep the default PickAndLift success conditions intact.
                    # We'll still record which object is intended via metadata and naming,
                    # but we won't modify the task's target/block to preserve expert stability.
                    # intended_idx already computed above

                    # Small grace steps before asking for a demo
                    try:
                        for _ in range(2):
                            task_env._scene.pyrep.step()
                    except Exception:
                        pass
                    # Collect expert demo (still picks the block for now; videos/names reflect intended target)
                    demo = None
                    demo_ok = False
                    last_err = None
                    for _dt in range(3):
                        try:
                            demo, = task_env.get_demos(amount=1, live_demos=True)
                            demo_ok = True
                            break
                        except Exception as e:
                            last_err = e
                            print(f'[WARN] Expert demo failed (try {_dt+1}/3): {e}')
                    if not demo_ok or demo is None:
                        print(f'[SKIP] Episode skipped; expert demo failed repeatedly: {last_err}')
                        continue

                    # Name episodes with object slug if enabled
                    if args.name_episodes_by_object and len(placed_objs) > 0:
                        intended_idx = ep % len(placed_objs)
                        intended_cat = placed_objs[intended_idx][0]
                        slug = _slugify(intended_cat)
                        ep_dir = episodes_path / f"episode_{ep:03d}__{slug}"
                    else:
                        ep_dir = episodes_path / (EPISODE_FOLDER % ep)
                    check_and_make(str(ep_dir))
                    _save_demo_to(str(ep_dir), demo)

                    # Save metadata of placed objects and intended target index (for training)
                    def _tolist(x):
                        try:
                            import numpy as _np
                            if isinstance(x, _np.ndarray):
                                return x.tolist()
                        except Exception:
                            pass
                        return x

                    meta = {
                        'scene': scene_name,
                        'variation': var,
                        'episode': ep,
                        'objects': [
                            {
                                'category': c,
                                'name': sh.get_name(),
                                'model_path': mp,
                                'position': _tolist(getattr(sh, 'get_position')() if hasattr(sh, 'get_position') else None),
                                'quaternion': _tolist(getattr(sh, 'get_quaternion')() if hasattr(sh, 'get_quaternion') else None),
                            } for (c, _base, sh, mp) in placed_objs
                        ],
                        'intended_target_index': intended_idx,
                    }
                    import json
                    with open(ep_dir / 'meta.json', 'w') as f:
                        json.dump(meta, f, indent=2)

                    # Optionally create named videos for wrist and overhead
                    if args.make_named_videos:
                        try:
                            import imageio
                            import re

                            def _frames_in(dir_path: Path):
                                files = [p for p in dir_path.glob('*.png')]
                                files.sort(key=lambda p: int(re.sub(r'\D', '', p.stem) or 0))
                                return files

                            def _write_mp4(frames, out_path: Path, fps=15):
                                if not frames:
                                    return
                                writer = imageio.get_writer(str(out_path), fps=fps, codec='libx264', quality=7)
                                try:
                                    for f in frames:
                                        writer.append_data(imageio.v2.imread(f))
                                finally:
                                    writer.close()

                            intended_cat = placed_objs[intended_idx][0] if placed_objs else 'none'
                            slug = _slugify(intended_cat)
                            base = f"{scene_name}_picking_{slug}"
                            # Wrist video
                            wrist_frames = _frames_in(ep_dir / WRIST_RGB_FOLDER)
                            _write_mp4(wrist_frames, ep_dir / f"{base}_wrist_rgb.mp4")
                            # Overhead video
                            ov_frames = _frames_in(ep_dir / OVERHEAD_RGB_FOLDER)
                            _write_mp4(ov_frames, ep_dir / f"{base}_overhead_rgb.mp4")
                        except Exception as _ve:
                            print(f"[WARN] Failed to write named videos: {_ve}")

                    # Cleanup: restore target visibility
                    try:
                        if target_shape is not None:
                            target_shape.set_renderable(True)
                    except Exception:
                        pass
                    # Remove placed objects before next episode/reset
                    for _, base_obj, _, _ in placed_objs:
                        try:
                            base_obj.remove()
                        except Exception:
                            pass
        print('\nDone generating scenes.')
    finally:
        env.shutdown()


if __name__ == '__main__':
    main()
