# scripts/run_hybrid_scheduler_demo.py
import os
from hrl_grasp.runtime_sampler import RuntimeSampler, fixed_eval_seeds

LOG = "/home/naren/HRL_part/rlbench_data/runtime_logs/hybrid_episodes.csv"

def main():
    rs = RuntimeSampler(
        headless=True,
        log_csv=LOG,
        episodes_per_arrangement=4,
        min_objects=3,
        max_objects=6,
        eval_mode=False,
    )
    try:
        # One arrangement reused for 4 sub-episodes
        info = rs.reset(mode="train", seed=None, debug_capture_images=False)
        for _ in range(4):
            # Use returned mapping/target for your policy here
            print(f"[DEMO] ARR={info['arrangement_id']} SUB={info['subep_idx']}/{info['B']} K={info['K']} target={info['target_cat']} pid={info['target_project_id']}")
            # End episode -> advance sub-episode
            rs.finish_episode(success=None)
            info = rs.reset(mode="train", seed=None, debug_capture_images=False)

        # Eval mode with seeds, iterate all targets per arrangement
        rs.eval_mode = True
        rs.eval_seeds = fixed_eval_seeds(3, 999)
        for _ in range(2):
            info = rs.reset(mode="eval", seed=None, debug_capture_images=False)
            B = info['B']
            for _ in range(B):
                rs.finish_episode(success=None)
                info = rs.reset(mode="eval", seed=None, debug_capture_images=False)
    finally:
        rs.shutdown()
    print(f"[DONE] Hybrid schedule demo complete. CSV at {LOG}")

if __name__ == "__main__":
    main()
