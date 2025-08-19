#!/usr/bin/env python3
import os, numpy as np
from hrl_grasp.runtime_sampler import RuntimeSampler, fixed_eval_seeds

LOG = "/home/naren/HRL_part/rlbench_data/runtime_logs/episodes.csv"


def run_subepisode(rs: RuntimeSampler, max_steps: int = 40):
    obs, info = rs.reset()
    steps = 0
    done = False
    success = False
    while not done and steps < max_steps:
        # Random small delta and random grip
        action = np.array([
            np.random.uniform(-0.02, 0.02),
            np.random.uniform(-0.02, 0.02),
            np.random.uniform(-0.02, 0.02),
            np.random.uniform(-1.0, 1.0),
        ], dtype=np.float32)
        obs, reward, done, info = rs.step(action)
        steps += 1
        success = bool(info.get("success", False))
    rs.finish_episode(success=success)


def main():
    rs = RuntimeSampler(headless=True, log_csv=LOG, camera_res=(640, 480))
    try:
        # Run a few train sub-episodes
        for _ in range(3):
            run_subepisode(rs, max_steps=50)

        # Switch to eval mode and force a new arrangement next reset
        rs.eval_mode = True
        rs.active_arrangement = False
        seeds = fixed_eval_seeds(n=3, base_seed=4242)
        for _ in seeds:
            run_subepisode(rs, max_steps=50)
    finally:
        rs.shutdown()
    print(f"[DONE] Wrote CSV rows to: {LOG}")


if __name__ == "__main__":
    main()
