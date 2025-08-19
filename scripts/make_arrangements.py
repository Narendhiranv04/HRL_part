# scripts/make_arrangements.py
import os, argparse
import numpy as np

from pyrep import PyRep
from pyrep.robots.arms.panda import Panda
from pyrep.objects.vision_sensor import VisionSensor

from tools.arrangement_lib import place_objects, remove_all, snapshot, save_mapping_json, ensure_dir

ALL_CATS = [
    "Cup","Plate","Knife","Block","USBStick","WineBottle","Shoes","Sponge","WateringCan","PuzzlePiece","Meat"
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--num_arrangements", type=int, default=20)
    ap.add_argument("--min_objects", type=int, default=3)
    ap.add_argument("--max_objects", type=int, default=6)
    ap.add_argument("--cats", type=str, nargs="*", default=ALL_CATS)
    ap.add_argument("--headless", action="store_true", default=False)
    args = ap.parse_args()

    scene = "/home/naren/HRL_part/rlbench/rlbench/task_design.ttt"
    pr = PyRep(); pr.launch(scene, headless=args.headless); pr.start()
    try:
        _ = Panda()  # load the robot; not used for images but okay to keep consistent
        if not VisionSensor.exists("cam_overhead"):
            cam_over = VisionSensor.create([640, 480])
            cam_over.set_position([0.0, 0.0, 1.30])
            cam_over.set_orientation([0, 0, 0])
        else:
            cam_over = VisionSensor("cam_overhead")

        ensure_dir(args.out_root)

        for i in range(args.num_arrangements):
            k = int(np.random.randint(args.min_objects, args.max_objects + 1))
            if k > len(args.cats):
                k = len(args.cats)
            cats = list(np.random.choice(args.cats, size=k, replace=False))
            placed, mapping = place_objects(cats)

            out_dir = os.path.join(args.out_root, f"arr_{i:04d}")
            ensure_dir(out_dir)
            snapshot(cam_over, os.path.join(out_dir, "overhead.png"))
            save_mapping_json(os.path.join(out_dir, "mapping.json"), mapping)

            print(f"[ARR {i:04d}] objects={k}")
            for pid, cat, obj, _ in placed:
                print(f"  project_id={pid:02d}  name={cat:12s} handle={obj.get_handle()} pos={obj.get_position()}")

            remove_all(placed)
    finally:
        pr.stop(); pr.shutdown()
    print(f"[DONE] Wrote {args.num_arrangements} arrangements to {args.out_root}")


if __name__ == "__main__":
    main()
