import pickle
from pathlib import Path

import imageio
import numpy as np
import torch
from torch.utils.data import Dataset


class RLBenchDataset(Dataset):
    """
    A PyTorch Dataset for loading RLBench demonstration data.
    This class is designed to work with the directory structure created by
    the RLBench dataset generator (or the fake one).
    """

    def __init__(self, data_path: str, tasks: list, image_views: list = None):
        """
        Args:
            data_path (str): The root path to the generated dataset.
            tasks (list): A list of task names to include.
            image_views (list, optional): A list of camera views to load.
                                          Defaults to ['wrist_rgb', 'overhead_rgb'].
        """
        self.data_path = Path(data_path)
        self.tasks = tasks
        self.image_views = image_views if image_views is not None else ['wrist_rgb', 'overhead_rgb']
        self.episodes = self._find_episodes()

        if not self.episodes:
            raise RuntimeError(f"No episodes found for tasks {tasks} in {data_path}")

    def _find_episodes(self):
        """Scans the data path and collects all episode paths, including scene_* layout."""
        episode_paths = []
        for task_name in self.tasks:
            task_path = self.data_path / task_name
            if not task_path.is_dir():
                continue
            # Default RLBench layout: task/variation*/episodes/episode*
            for ep_path in task_path.glob('variation*/episodes/episode*'):
                episode_paths.append(ep_path)
            # Extended layout with scenes: task/scene_*/variation*/episodes/episode*
            for ep_path in task_path.glob('scene_*/variation*/episodes/episode*'):
                episode_paths.append(ep_path)
            # Also handle object-named episodes (episode_xxx__slug)
            for ep_path in task_path.glob('scene_*/variation*/episodes/episode_*__*'):
                episode_paths.append(ep_path)
        return sorted(set(episode_paths))

    def __len__(self):
        """Returns the total number of episodes."""
        return len(self.episodes)

    def __getitem__(self, idx):
        """
        Loads a single episode's data.

        For simplicity in this example, we will load the entire trajectory
        for an episode. A more advanced implementation might load individual
        timesteps.

        Returns:
            A dictionary containing:
            - 'low_dim_state': The low-dimensional state data.
            - 'images': A dictionary of image sequences for the requested views.
        """
        episode_path = self.episodes[idx]

        # Load low-dimensional observations
        with open(episode_path / 'low_dim_obs.pkl', 'rb') as f:
            low_dim_obs = pickle.load(f)

        # Load image sequences
        images = {}
        for view in self.image_views:
            image_path = episode_path / view
            # Find all PNGs and sort them numerically
            img_files = sorted(image_path.glob('*.png'), key=lambda x: int(x.stem))
            # Stack images into a single numpy array (T, H, W, C)
            imgs = np.stack([imageio.v2.imread(f) for f in img_files])
            # Convert to tensor (T, C, H, W) and normalize to [0, 1]
            imgs_tensor = torch.from_numpy(imgs).permute(0, 3, 1, 2).float() / 255.0
            images[view] = imgs_tensor

        # Optional meta with intended target and objects
        meta = None
        meta_path = episode_path / 'meta.json'
        if meta_path.exists():
            try:
                import json
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
            except Exception:
                meta = None

        # For offline RL, you typically want state, action, reward, next_state
        # Here, we are just providing the state observations (low_dim and image)
        # The actions would need to be extracted from the demo file if available.
        # For this example, we focus on loading the observations.
        sample = {
            'low_dim_state': low_dim_obs,
            'images': images
        }
        if meta is not None:
            sample['meta'] = meta
            sample['intended_target_index'] = meta.get('intended_target_index')

        return sample
