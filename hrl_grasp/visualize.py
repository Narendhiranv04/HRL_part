"""Visualize RLBench scenes for the 3 grasp tasks and save wrist & side views.

Usage: run `python visualize.py` and follow prompt. This script launches RLBench,
samples scene variations, renders the wrist camera and left_shoulder camera views,
and saves images to ./visuals.

Note: Requires CoppeliaSim and PyRep installed and RLBench available.
"""
import os
import random
import numpy as np
import imageio
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.tasks import PickAndLift, PickUpCup, LiftNumberedBlock

OUT_DIR = os.path.join(os.path.dirname(__file__), 'visuals')
os.makedirs(OUT_DIR, exist_ok=True)

TASKS = {
    'pick_and_lift': PickAndLift,
    'pick_up_cup': PickUpCup,
    'lift_numbered_block': LiftNumberedBlock,
}


def make_env(task_cls, headless=False):
    action_mode = MoveArmThenGripper(arm_action_mode=JointVelocity(), gripper_action_mode=Discrete())
    obs_config = ObservationConfig()
    obs_config.set_all(True)
    env = Environment(action_mode=action_mode, obs_config=obs_config, headless=headless)
    env.launch()
    task = env.get_task(task_cls)
    return env, task


def save_obs_images(obs, prefix):
    # left_shoulder and wrist available
    l = obs.left_shoulder_rgb
    w = obs.wrist_rgb
    imageio.imsave(f"{prefix}_left_shoulder.png", l)
    imageio.imsave(f"{prefix}_wrist.png", w)


def visualize_task(task_name, n=5):
    task_cls = TASKS[task_name]
    env, task = make_env(task_cls, headless=False)
    for i in range(n):
        task.sample_variation()
        descriptions, obs = task.reset()
        prefix = os.path.join(OUT_DIR, f"{task_name}_{i}")
        save_obs_images(obs, prefix)
        print(f"Saved visuals for {task_name} variation {i} -> {prefix}_*.png")
    env.shutdown()


if __name__ == '__main__':
    print('Available tasks:', list(TASKS.keys()))
    t = input('Enter task name to visualize (or Enter for pick_and_lift): ').strip() or 'pick_and_lift'
    if t not in TASKS:
        print('Unknown task')
    else:
        visualize_task(t, n=3)
