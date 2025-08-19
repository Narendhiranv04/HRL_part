# scripts/make_arrangements_images.py
import os, argparse, math, time
import numpy as np

from pyrep import PyRep
from pyrep.robots.arms.panda import Panda
from pyrep.objects.vision_sensor import VisionSensor

from tools.arrangement_lib import place_objects, remove_all, snapshot, save_mapping_json, ensure_dir

ALL_CATS = [
    "Cup","Plate","Knife","Block","USBStick","WineBottle","Shoes","Sponge","WateringCan","PuzzlePiece","Meat"
]

SCENE = "/home/naren/HRL_part/rlbench/rlbench/task_design.ttt"


def move_wrist_observation(arm: Panda, center=(0.0, 0.0, 0.95)):
    # Point tool downwards (Z-)
    euler = (0.0, math.pi, 0.0)
    j = arm.solve_ik(position=list(center), euler=euler)
    arm.set_joint_positions(j)


def ensure_cam(name: str, default_pos=None, default_euler=None, res=(640, 480)) -> VisionSensor:
    if VisionSensor.exists(name):
        cam = VisionSensor(name)
    else:
        cam = VisionSensor.create([res[0], res[1]])
        if default_pos is not None:
            cam.set_position(default_pos)
        if default_euler is not None:
            cam.set_orientation(default_euler)
    cam.set_resolution(res)
    return cam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--num_arrangements", type=int, default=10)
    ap.add_argument("--min_objects", type=int, default=3)
    ap.add_argument("--max_objects", type=int, default=6)
    ap.add_argument("--cats", type=str, nargs="*", default=ALL_CATS)
    ap.add_argument("--headless", action="store_true", default=True)
    args = ap.parse_args()

    pr = PyRep(); pr.launch(SCENE, headless=args.headless); pr.start()
    try:
        arm = Panda()
        # Cameras
        cam_over = ensure_cam("cam_overhead", default_pos=[0, 0, 1.30], default_euler=[0, 0, 0])
        cam_wrist = VisionSensor("cam_wrist") if VisionSensor.exists("cam_wrist") else ensure_cam("cam_wrist_fallback", [0,0,1.0], [0,0,0])

        ensure_dir(args.out_root)

        for i in range(args.num_arrangements):
            k = int(np.random.randint(args.min_objects, args.max_objects + 1))
            if k > len(args.cats):
                k = len(args.cats)
            cats = list(np.random.choice(args.cats, size=k, replace=False))

            placed, mapping = place_objects(cats)

            # Move wrist to an observation pose over table center
            try:
                move_wrist_observation(arm, (0.0, 0.0, 0.95))
            except Exception:
                pass

            # Step a few frames to ensure proper render
            for _ in range(5):
                pr.step()

            out_dir = os.path.join(args.out_root, f"arr_{i:04d}")
            ensure_dir(out_dir)
            # Save images and mapping
            snapshot(cam_over, os.path.join(out_dir, "overhead.png"))
            snapshot(cam_wrist, os.path.join(out_dir, "wrist.png"))
            save_mapping_json(os.path.join(out_dir, "mapping.json"), mapping)

            remove_all(placed)
    finally:
        pr.stop(); pr.shutdown()

    print(f"[DONE] Wrote {args.num_arrangements} arrangements with overhead & wrist images to {args.out_root}")


if __name__ == "__main__":
    main()
