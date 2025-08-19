#!/usr/bin/env python3
import os
from rlbench.environment import Environment
from rlbench import ObservationConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.tasks.pick_and_lift import PickAndLift
from pyrep.const import RenderMode

def main():
    obs = ObservationConfig()
    obs.set_all(True)
    for cam in [obs.right_shoulder_camera, obs.left_shoulder_camera, obs.overhead_camera, obs.wrist_camera, obs.front_camera]:
        cam.image_size = (128, 128)
        cam.depth_in_meters = False
        cam.masks_as_one_channel = False
        cam.render_mode = RenderMode.OPENGL3
    env = Environment(
        MoveArmThenGripper(JointVelocity(), Discrete()),
        obs_config=obs,
        headless=True,
    )
    env.launch()
    try:
        task = env.get_task(PickAndLift)
        desc, _ = task.reset()
        print('Reset OK. Variation descriptions:', len(desc))
        demo, = task.get_demos(amount=1, live_demos=True)
        print('Demo OK. Length:', len(demo))
    finally:
        env.shutdown()

if __name__ == '__main__':
    main()
