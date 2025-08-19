import random
from collections import deque, namedtuple
from typing import Tuple
import numpy as np

Transition = namedtuple('Transition', ['obs', 'action', 'reward', 'next_obs', 'done'])

class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, action_dim: int):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def push(self, obs: np.ndarray, action: np.ndarray, reward: float, next_obs: np.ndarray, done: bool):
        self.buffer.append(Transition(obs.astype(np.float32), action.astype(np.float32), float(reward), next_obs.astype(np.float32), bool(done)))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch = random.sample(self.buffer, batch_size)
        obs = np.stack([b.obs for b in batch], axis=0)
        actions = np.stack([b.action for b in batch], axis=0)
        rewards = np.array([b.reward for b in batch], dtype=np.float32)[:, None]
        next_obs = np.stack([b.next_obs for b in batch], axis=0)
        dones = np.array([b.done for b in batch], dtype=np.float32)[:, None]
        return obs, actions, rewards, next_obs, dones
