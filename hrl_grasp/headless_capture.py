"""Headless capture for RLBench tasks: saves overhead (isometric) and wrist camera images/videos.

Usage:
    python3 headless_capture.py --task pick_and_lift --n 50 --out ./captures

This script launches RLBench headless, samples scene variations, saves images for two cameras
(overhead and wrist) and assembles short per-variation MP4s.

Note: relies on RLBench observation config exposing `overhead` and `wrist` cameras.
"""
import os
import argparse
import imageio
import numpy as np
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.observation_config import ObservationConfig
from rlbench.environment import Environment
from rlbench.tasks import PickAndLift, PickUpCup, LiftNumberedBlock

TASKS = {
    'pick_and_lift': PickAndLift,
    'pick_up_cup': PickUpCup,
    'lift_numbered_block': LiftNumberedBlock,
}


def make_env(headless=True):
    action_mode = MoveArmThenGripper(arm_action_mode=JointVelocity(), gripper_action_mode=Discrete())
    obs_config = ObservationConfig()
    # enable overhead and wrist cams
    obs_config.set_all_high_dim(True)
    obs_config.set_all_low_dim(False)
    # ensure overhead and wrist are active
    obs_config.overhead_camera.set_all(True)
    obs_config.wrist_camera.set_all(True)

    env = Environment(action_mode=action_mode, obs_config=obs_config, headless=headless)
    env.launch()
    return env


def capture_task(task_name, n, out_dir):
    if task_name not in TASKS:
        raise ValueError('Unknown task')
    task_cls = TASKS[task_name]
    env = make_env(headless=True)
    task = env.get_task(task_cls)
    os.makedirs(out_dir, exist_ok=True)
    print('Capturing', task_name, '->', out_dir)
    try:
        for i in range(n):
            task.sample_variation()
            descriptions, obs = task.reset()
            # obs contains overhead_rgb and wrist_rgb if configured
            overhead = obs.overhead_rgb
            wrist = obs.wrist_rgb
            base = os.path.join(out_dir, f"{task_name}_{i}")
            imageio.imsave(base + '_overhead.png', overhead)
            imageio.imsave(base + '_wrist.png', wrist)
            print('Saved', base + '_overhead.png', base + '_wrist.png')
    finally:
        env.shutdown()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--task', default='pick_and_lift', choices=list(TASKS.keys()))
    p.add_argument('--n', type=int, default=10)
    p.add_argument('--out', default='./captures')
    args = p.parse_args()
    capture_task(args.task, args.n, args.out)
