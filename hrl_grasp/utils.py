"""Helpers for recording rollouts and saving videos/images during evaluation.

This file provides a minimal `record_episode` that runs an RLBench task environment
and writes wrist and left_shoulder frames to disk. It assumes RLBench is installed.
"""
import os
import imageio


def record_obs_images(obs, out_prefix):
    # obs is RLBench Observation object with left_shoulder_rgb and wrist_rgb
    if not os.path.exists(os.path.dirname(out_prefix)):
        os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    imageio.imsave(out_prefix + '_left_shoulder.png', obs.left_shoulder_rgb)
    imageio.imsave(out_prefix + '_wrist.png', obs.wrist_rgb)
