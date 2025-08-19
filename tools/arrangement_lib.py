# tools/arrangement_lib.py
from __future__ import annotations
from typing import List, Tuple, Dict

import os, json, math, random
import numpy as np

try:
    import cv2  # optional; used if available
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False
    import imageio

from pyrep.objects.shape import Shape, PrimitiveShape
from pyrep.objects.vision_sensor import VisionSensor

# Adjust to match your RLBench scene table height
TABLE_Z = 0.78
# Table bounds (meters) for X and Y placement
XY_BOUNDS = ((-0.40, 0.40), (-0.20, 0.20))
# Minimum center-to-center separation between objects (meters)
MIN_SEP = 0.06

# Category → primitive proxy (+ nominal size in meters)
# For CUBOID: size=(sx, sy, sz)
# For CYLINDER: size=(radius, radius, height)
CAT2PRIM: Dict[str, Tuple[str, Tuple[float, float, float]]] = {
    "Cup":          ("CYLINDER", (0.035, 0.035, 0.09)),
    "Sponge":       ("CUBOID",   (0.09, 0.06, 0.02)),
    "WateringCan":  ("CYLINDER", (0.05, 0.05, 0.16)),
    "Block":        ("CUBOID",   (0.05, 0.05, 0.05)),
    "Plate":        ("CYLINDER", (0.10, 0.10, 0.02)),
    "Knife":        ("CUBOID",   (0.20, 0.02, 0.01)),
    "USBStick":     ("CUBOID",   (0.06, 0.02, 0.01)),
    "WineBottle":   ("CYLINDER", (0.04, 0.04, 0.30)),
    "Shoes":        ("CUBOID",   (0.24, 0.09, 0.08)),
    "PuzzlePiece":  ("CUBOID",   (0.06, 0.06, 0.01)),
    "Meat":         ("CUBOID",   (0.10, 0.07, 0.03)),
}


def _rand_xy() -> Tuple[float, float]:
    return (
        float(np.random.uniform(*XY_BOUNDS[0])),
        float(np.random.uniform(*XY_BOUNDS[1])),
    )


def _too_close(p: Tuple[float, float], others: List[Tuple[float, float]]) -> bool:
    for o in others:
        d = math.hypot(p[0] - o[0], p[1] - o[1])
        if d < MIN_SEP:
            return True
    return False


def spawn_object(category: str, color=None) -> Tuple[Shape, Tuple[float, float, float]]:
    kind, size = CAT2PRIM[category]
    if color is None:
        color = [random.random(), random.random(), random.random()]
    if kind == "CUBOID":
        shape = Shape.create(
            type=PrimitiveShape.CUBOID,
            size=size,  # (sx, sy, sz)
            color=color,
            static=False,
            respondable=True,
        )
        sz = size[2]
    else:
        # CYLINDER expects diameters in x,y and height in z
        r, _, h = size
        shape = Shape.create(
            type=PrimitiveShape.CYLINDER,
            size=(r * 2.0, r * 2.0, h),
            color=color,
            static=False,
            respondable=True,
        )
        sz = h
    return shape, (size[0], size[1], sz)


def place_objects(categories: List[str], z: float = TABLE_Z):
    """Spawn and place objects with rejection sampling to avoid overlaps.
    Returns: (placed, mapping)
      placed: list of (project_id, category, shape, position)
      mapping: dict project_id -> info
    """
    placed: List[Tuple[int, str, Shape, Tuple[float, float, float]]] = []
    xy_list: List[Tuple[float, float]] = []
    mapping: Dict[int, dict] = {}

    for idx, cat in enumerate(categories):
        obj, size = spawn_object(cat)
        sx, sy, sz = size
        # rejection sampling in XY; ensure above table by half-height
        for _ in range(200):
            x, y = _rand_xy()
            if not _too_close((x, y), xy_list):
                xy_list.append((x, y))
                obj.set_position([x, y, z + (sz * 0.5)])
                # random yaw only
                obj.set_orientation([0.0, 0.0, random.uniform(-math.pi, math.pi)])
                break
        pid = int(idx)
        pos = tuple(obj.get_position())
        placed.append((pid, cat, obj, pos))
        mapping[pid] = {
            "category": cat,
            "handle": int(obj.get_handle()),
            "position": list(pos),
            "orientation": list(obj.get_orientation()),
            "size": [sx, sy, sz],
        }
    return placed, mapping


def remove_all(placed) -> None:
    for _, _, obj, _ in placed:
        try:
            obj.remove()
        except Exception:
            pass


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def snapshot(sensor: VisionSensor, out_png: str) -> None:
    img = sensor.capture_rgb()  # float [H,W,3] in [0,1], RGB
    img8 = (img * 255).astype(np.uint8)
    if _HAS_CV2:
        # OpenCV expects BGR
        import cv2
        cv2.imwrite(out_png, img8[..., ::-1])
    else:
        imageio.v2.imwrite(out_png, img8)


def save_mapping_json(out_json: str, mapping: dict) -> None:
    with open(out_json, "w") as f:
        json.dump(mapping, f, indent=2)
