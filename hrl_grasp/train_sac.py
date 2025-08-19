"""Skeleton training script for SAC on RLBench tasks using state observations.

This script uses the RLBench Gym wrapper (rlbench.rlbench.gym.RLBenchEnv) to create
an env with observation_mode='state' and trains a Stable-Baselines3 SAC agent
on the selected task. It logs progress to stdout and saves model + periodic
evaluation rollouts (images) using utils.record_obs_images.

Notes:
- This is a scaffold. Tune hyperparams and add replay/dataset logic as needed.
- Requires RLBench, PyRep, Stable-Baselines3, and CoppeliaSim installed.
"""
import os
import numpy as np
import time
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.observation_config import ObservationConfig
from rlbench.environment import Environment
from rlbench.tasks import PickAndLift, PickUpCup, LiftNumberedBlock

from utils import record_obs_images

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
import gymnasium as gym
from rlbench.gym import RLBenchEnv
from stable_baselines3 import SAC

# Create an RLBench environment
env = RLBenchEnv(
    task_class="reach_target",
    observation_mode="state",
    render_mode="human"
)

model = SAC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000, log_interval=4)
model.save("sac_rlbench")

del model # remove to demonstrate saving and loading

model = SAC.load("sac_rlbench")

obs, info = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
    gym_env = Monitor(gym_env)
