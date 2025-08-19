# scripts/smoke_pick_demo.py
import os, sys, math
import numpy as np

from pyrep import PyRep
from pyrep.robots.arms.panda import Panda
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.objects.dummy import Dummy

from tools.arrangement_lib import place_objects, remove_all, snapshot, save_mapping_json, ensure_dir

# Use RLBench's base scene (same one Environment uses)
SCENE = "/home/naren/HRL_part/rlbench/rlbench/task_design.ttt"
OUT = "/home/naren/HRL_part/rlbench_data/_smoke"
CATS = ["Cup", "Sponge", "WateringCan", "Block"]


def solve_ik_waypoint(arm: Panda, pos, euler=(0, math.pi, 0)):
    j = arm.solve_ik(position=pos, euler=euler)
    return j


def main():
    ensure_dir(OUT)
    headless = False
    pr = PyRep(); pr.launch(SCENE, headless=headless); pr.start()
    try:
        arm = Panda()
        if VisionSensor.exists("cam_overhead"):
            cam_over = VisionSensor("cam_overhead")
        else:
            cam_over = VisionSensor.create([640, 480])
            cam_over.set_position([0.0, 0.0, 1.30])
            cam_over.set_orientation([0.0, 0.0, 0.0])

        placed, mapping = place_objects(CATS)
        print("[SMOKE] project_id → {category, handle}:")
        for pid, cat, obj, _ in placed:
            print(f"  {pid}: {cat}, handle={obj.get_handle()}, pos={obj.get_position()}")

        # Save mapping and static snapshot
        snapshot(cam_over, os.path.join(OUT, "arrangement.png"))
        save_mapping_json(os.path.join(OUT, "arrangement.json"), mapping)

        # Attempt a naive IK pick
        try:
            import cv2
            H, W = 480, 640
            target_pid, _, tgt, _ = placed[np.random.randint(len(placed))]
            print(f"[SMOKE] Target project_id={target_pid}")
            vid = cv2.VideoWriter(os.path.join(OUT, "smoke_traj.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 20, (W, H))
            cam = VisionSensor("cam_wrist") if VisionSensor.exists("cam_wrist") else cam_over

            above = np.array(tgt.get_position()); above[2] += 0.15
            down  = above.copy(); down[2] -= 0.12

            j = solve_ik_waypoint(arm, above.tolist()); arm.set_joint_positions(j)
            for _ in range(30): pr.step(); frame = (cam.capture_rgb()[..., ::-1]*255).astype(np.uint8); vid.write(frame)
            j = solve_ik_waypoint(arm, down.tolist()); arm.set_joint_positions(j)
            for _ in range(30): pr.step(); frame = (cam.capture_rgb()[..., ::-1]*255).astype(np.uint8); vid.write(frame)
            arm.gripper.grasp(tgt)
            for _ in range(10): pr.step(); frame = (cam.capture_rgb()[..., ::-1]*255).astype(np.uint8); vid.write(frame)
            j = solve_ik_waypoint(arm, above.tolist()); arm.set_joint_positions(j)
            for _ in range(40): pr.step(); frame = (cam.capture_rgb()[..., ::-1]*255).astype(np.uint8); vid.write(frame)
            vid.release()
        except Exception as e:
            print(f"[SMOKE] IK/video skipped due to error: {e}")
        finally:
            remove_all(placed)
    finally:
        pr.stop(); pr.shutdown()
    print(f"[SMOKE] Wrote: {OUT}/arrangement.png, {OUT}/arrangement.json and optionally smoke_traj.mp4")


if __name__ == "__main__":
    main()
