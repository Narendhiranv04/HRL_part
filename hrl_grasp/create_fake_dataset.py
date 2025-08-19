"""
Generates a fake RLBench dataset with the correct directory structure and
placeholder data. This allows for the development of data loading and training
pipelines without needing to run the full simulation environment.
"""
import os
import pickle
import shutil
from pathlib import Path

import imageio
import numpy as np
from tqdm import tqdm

# --- Configuration ---
# Descriptions and shapes are based on RLBench environment inspection.
LOW_DIM_STATE_DESC = {
    'joint_velocities': (8,),
    'joint_positions': (8,),
    'joint_forces': (8,),
    'gripper_open': (1,),
    'gripper_pose': (7,),
    'gripper_touch_forces': (8,),
    'task_low_dim_state': (0,)  # Task-specific, can be empty
}

IMAGE_VIEWS = {
    'left_shoulder_rgb': (128, 128, 3),
    'overhead_rgb': (128, 128, 3),
    'wrist_rgb': (128, 128, 3),
    'front_rgb': (128, 128, 3),
}

TASKS = ['pick_and_lift', 'pick_up_cup', 'lift_numbered_block']
VARIATIONS = 5
EPISODES_PER_VARIATION = 5
IMAGES_PER_EPISODE = 10


def create_fake_episode(episode_path: Path):
    """Creates placeholder data for a single episode."""
    # Create placeholder images for each camera view
    for view_name, shape in IMAGE_VIEWS.items():
        view_path = episode_path / view_name
        view_path.mkdir(parents=True, exist_ok=True)
        placeholder_img = np.random.randint(0, 256, size=shape, dtype=np.uint8)
        for i in range(IMAGES_PER_EPISODE):
            imageio.imwrite(view_path / f'{i}.png', placeholder_img)

    # Create placeholder low-dimensional state pickle
    low_dim_obs = {}
    for key, shape in LOW_DIM_STATE_DESC.items():
        low_dim_obs[key] = np.random.randn(*shape).astype(np.float32)

    with open(episode_path / 'low_dim_obs.pkl', 'wb') as f:
        pickle.dump(low_dim_obs, f)


def main():
    """Main function to generate the fake dataset."""
    base_path = Path('/home/naren/HRL_part/rlbench_data')
    if base_path.exists():
        print(f"Dataset path {base_path} already exists. Deleting it.")
        shutil.rmtree(base_path)

    print(f"Generating fake dataset at: {base_path}")

    for task_name in TASKS:
        task_path = base_path / task_name
        print(f"  Creating task: {task_name}")
        for var_idx in tqdm(range(VARIATIONS), desc=f"  - Variations for {task_name}"):
            variation_path = task_path / f'variation{var_idx}'
            for ep_idx in range(EPISODES_PER_VARIATION):
                episode_path = variation_path / f'episode{ep_idx}'
                create_fake_episode(episode_path)

    print("\nFake dataset generation complete!")
    print(f"Total tasks: {len(TASKS)}")
    print(f"Total variations: {len(TASKS) * VARIATIONS}")
    print(f"Total episodes: {len(TASKS) * VARIATIONS * EPISODES_PER_VARIATION}")


if __name__ == '__main__':
    main()
