HRL_part - Grasp skill scaffold

This workspace contains scaffolding to train and visualize 'Grasp' low-level skills using RLBench tasks: PickAndLift, PickUpCup, LiftNumberedBlock.

Structure:
- `visualize.py` - view random scene variations and save wrist (end-effector) and side views.
- `train_sac.py` - skeleton to train a SAC agent on RLBench tasks using state observations.
- `utils.py` - helpers to record evaluation rollouts and save videos/images.
- `requirements.txt` - Python deps to install.

New utility:
- `headless_capture.py` - headless-only tool to capture and save images/videos from two cameras (overhead/isometric + wrist end-effector) for the configured tasks. Use this to produce datasets of scene variations without opening any GUI.

See comments in scripts for usage notes. You must install RLBench and CoppeliaSim as described in the RLBench README before running.
