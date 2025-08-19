#!/usr/bin/env python3
"""
Manual four-object pick script using RLBench scene (no Task).
- Launches base RLBench scene with Panda.
- Imports 4 RLBench assets (Cup, Plate, Block, Sponge) and places around the target area.
- For each object: plan a simple pick (approach -> grasp -> lift) and record wrist/overhead frames.
- Saves videos and meta per object in same scene directory.

This bypasses PickAndLift task constraints that prevent extra objects from being added to the task tree.
"""
from __future__ import annotations
import os
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
from pyrep.const import RenderMode
from pyrep.objects.shape import Shape
from pyrep.objects.vision_sensor import VisionSensor
from math import cos, sin

from rlbench import ObservationConfig
from rlbench.environment import Environment
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointPosition
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.backend.scene import Scene

from rlb_object_library import ensure_assets

ASSET_ORDER = ['Cup', 'Plate', 'Block', 'Sponge']
SLOTS = [(0.16, 0.00), (-0.16, 0.00), (0.00, 0.16), (0.00, -0.16)]
SAVE_ROOT = Path('/home/naren/HRL_part/rlbench_data/custom_four_pick/scene_1')


def setup_env() -> Tuple[Environment, Scene]:
    obs = ObservationConfig()
    obs.set_all(True)
    for cam in [obs.right_shoulder_camera, obs.left_shoulder_camera, obs.overhead_camera, obs.wrist_camera, obs.front_camera]:
        cam.image_size = (128, 128)
        cam.depth_in_meters = False
        cam.masks_as_one_channel = False
        cam.render_mode = RenderMode.OPENGL3
    env = Environment(MoveArmThenGripper(JointPosition(), Discrete()), obs_config=obs, headless=True)
    env.launch()
    return env, env._scene


def import_objects(scene: Scene, categories: List[str]) -> List[Tuple[str, object, Shape]]:
    mapping = ensure_assets(headless=True)
    placed: List[Tuple[str, object, Shape]] = []
    # Use target position as center
    target = Shape('pick_and_lift_target')
    tpos = target.get_position()
    for i, cat in enumerate(categories):
        mp = mapping[cat]
        base = scene.pyrep.import_model(str(mp))
        shp = Shape(base.get_handle())
        # scale down a bit
        try:
            shp.set_scale([0.8, 0.8, 0.8])
        except Exception:
            pass
        dx, dy = SLOTS[i % len(SLOTS)]
        shp.set_position([tpos[0] + dx, tpos[1] + dy, tpos[2] + 0.005])
        try:
            shp.set_orientation([0.0, 0.0, np.random.uniform(-3.14, 3.14)])
        except Exception:
            pass
        placed.append((cat, base, shp))
    return placed


def euler_to_quat(roll: float, pitch: float, yaw: float) -> List[float]:
    cr = cos(roll * 0.5)
    sr = sin(roll * 0.5)
    cp = cos(pitch * 0.5)
    sp = sin(pitch * 0.5)
    cy = cos(yaw * 0.5)
    sy = sin(yaw * 0.5)
    w = cr*cp*cy + sr*sp*sy
    x = sr*cp*cy - cr*sp*sy
    y = cr*sp*cy + sr*cp*sy
    z = cr*cp*sy - sr*sp*cy
    return [x, y, z, w]


def look_at(obj: Shape) -> Tuple[List[float], List[float]]:
    pos = obj.get_position()
    # approach pose slightly above
    approach = [pos[0], pos[1], pos[2] + 0.18]
    # gripper pointing down; Panda tool frame is roughly aligned with -Z world
    # Use a quaternion for downward orientation
    quat = euler_to_quat(np.pi, 0.0, 0.0)
    return approach, quat


def plan_and_execute(scene: Scene, target: Shape) -> bool:
    arm = scene.robot.arm
    gripper = scene.robot.gripper
    try:
        approach, quat = look_at(target)
        path1 = arm.get_path(approach, quaternion=quat, ignore_collisions=True)
        path1.step()
        path1.set_to_end()
        # descend to grasp
        grasp = target.get_position()
        grasp[2] = max(grasp[2] + 0.01, 0.78)  # ensure slightly above table height
        path2 = arm.get_path(grasp, quaternion=quat, ignore_collisions=True)
        path2.step()
        path2.set_to_end()
        # close gripper
        done = False
        for _ in range(50):
            done = gripper.actuate(0.0, 0.04)
            scene.step()
            if done:
                break
        # lift
        lift = [grasp[0], grasp[1], grasp[2] + 0.20]
        path3 = arm.get_path(lift, quaternion=quat, ignore_collisions=True)
        path3.step()
        path3.set_to_end()
        # small hold
        for _ in range(10):
            scene.step()
        # open and return
        for _ in range(50):
            done = gripper.actuate(1.0, 0.04)
            scene.step()
            if done:
                break
        return True
    except Exception as e:
        print('[PICK] Failed:', e)
        return False


def record_frame(cam: VisionSensor, path: Path, idx: int):
    import imageio
    img = cam.capture_rgb()
    (path / 'rgb').mkdir(parents=True, exist_ok=True)
    imageio.v2.imwrite(str((path / 'rgb' / f'{idx}.png')), (img * 255).astype(np.uint8))


def main():
    SAVE_ROOT.mkdir(parents=True, exist_ok=True)
    env, scene = setup_env()
    try:
        placed = import_objects(scene, ASSET_ORDER)
        wrist = scene._cam_wrist
        overhead = scene._cam_overhead
        # One episode per object; keep all present
        for ep, (cat, _base, shp) in enumerate(placed):
            ep_dir = SAVE_ROOT / f'episode_{ep:03d}__{cat.lower()}'
            (ep_dir / 'wrist_rgb').mkdir(parents=True, exist_ok=True)
            (ep_dir / 'overhead_rgb').mkdir(parents=True, exist_ok=True)
            frames = 0
            # small pre-roll
            for _ in range(5):
                scene.step()
                record_frame(wrist, ep_dir / 'wrist', frames)
                record_frame(overhead, ep_dir / 'overhead', frames)
                frames += 1
            ok = plan_and_execute(scene, shp)
            # capture some more frames
            for _ in range(30):
                scene.step()
                record_frame(wrist, ep_dir / 'wrist', frames)
                record_frame(overhead, ep_dir / 'overhead', frames)
                frames += 1
            meta = {
                'episode': ep,
                'category': cat,
                'intended_target_index': ep,
            }
            with open(ep_dir / 'meta.json', 'w') as f:
                json.dump(meta, f, indent=2)
        # videos
        try:
            import imageio, re
            for ep, (cat, _, __) in enumerate(placed):
                ep_dir = SAVE_ROOT / f'episode_{ep:03d}__{cat.lower()}'
                base = f'scene_1_picking_{cat.lower()}'
                def frames_in(d: Path):
                    files = list((d / 'rgb').glob('*.png'))
                    files.sort(key=lambda p: int(re.sub(r'\D', '', p.stem) or 0))
                    return files
                def write_mp4(frames, out):
                    if not frames:
                        return
                    w = imageio.get_writer(str(out), fps=15, codec='libx264', quality=7)
                    try:
                        for f in frames:
                            w.append_data(imageio.v2.imread(f))
                    finally:
                        w.close()
                write_mp4(frames_in(ep_dir / 'wrist'), ep_dir / f'{base}_wrist_rgb.mp4')
                write_mp4(frames_in(ep_dir / 'overhead'), ep_dir / f'{base}_overhead_rgb.mp4')
        except Exception as e:
            print('[VIDEO] Failed:', e)
    finally:
        env.shutdown()


if __name__ == '__main__':
    main()
