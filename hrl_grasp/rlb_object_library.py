"""
rlb_object_library.py

Extract single-object TTMs from RLBench task scenes so we can import
graspable items directly (without loading entire task scenes) when
generating custom datasets.

Categories supported:
  Cup, Plate, Knife, Block, USB Stick, Wine Bottle, Shoes,
  Sponge, Watering Can, Puzzle Piece.

Usage:
  from rlb_object_library import ensure_assets
  mapping = ensure_assets()
  # mapping: {category: /abs/path/to/asset.ttm}

Notes:
  - Requires COPPELIASIM_ROOT to be set (PyRep uses this).
  - Uses RLBench's task_design.ttt as a base empty scene to import from.
  - We save extracted objects under hrl_grasp/rlb_assets/<slug>.ttm
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple, Optional

from pyrep import PyRep
from pyrep.objects.shape import Shape


# Map requested category to (source_ttm, object_name_inside_scene)

def _rlb_repo_root() -> Path:
    # Locate the RLBench repo in the workspace by relative path
    this = Path(__file__).resolve()
    root = this.parents[1] / 'rlbench' / 'rlbench'
    if not root.exists():
        # fallback: if layout differs, allow RLBench installed as package (unsupported for extraction)
        raise FileNotFoundError('RLBench sources not found under workspace/rlbench/rlbench')
    return root


def _category_mapping() -> Dict[str, Tuple[Path, str]]:
    rlb = _rlb_repo_root()
    task_ttms = rlb / 'task_ttms'
    assets = rlb / 'assets'
    return {
        'Cup': (task_ttms / 'pick_up_cup.ttm', 'cup'),
        'Plate': (assets / 'plate.ttm', 'plate'),
        'Knife': (task_ttms / 'put_knife_on_chopping_board.ttm', 'knife'),
        'Block': (task_ttms / 'lift_numbered_block.ttm', 'block1'),
        'USB Stick': (task_ttms / 'insert_usb_in_computer.ttm', 'usb'),
        'Wine Bottle': (task_ttms / 'stack_wine.ttm', 'wine_bottle'),
        'Shoes': (task_ttms / 'put_shoes_in_box.ttm', 'shoe1'),
        'Sponge': (task_ttms / 'wipe_desk.ttm', 'sponge'),
        'Watering Can': (task_ttms / 'water_plants.ttm', 'waterer'),
        'Puzzle Piece': (task_ttms / 'solve_puzzle.ttm', 'solve_puzzle_piece1'),
    }


def _slug(name: str) -> str:
    return name.lower().replace(' ', '_')


def _launch_pyrep(scene_ttt: Path, headless: bool = True) -> PyRep:
    if not os.environ.get('COPPELIASIM_ROOT'):
        raise EnvironmentError('COPPELIASIM_ROOT not set; required for PyRep.')
    pr = PyRep()
    pr.launch(str(scene_ttt), headless=headless)
    return pr


def _find_shape_or_none(shape_name: str) -> Optional[Shape]:
    try:
        return Shape(shape_name)
    except Exception:
        return None


def _save_shape_model(shape: Shape, out_path: Path) -> None:
    # Directly save this shape as a model (will include its children/geom)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Ensure it is flagged as a model before saving
        shape.set_model(True)
    except Exception:
        pass
    shape.save_model(str(out_path))


def ensure_assets(out_dir: Optional[Path] = None, headless: bool = True) -> Dict[str, str]:
    """Ensure single-object TTMs exist for all categories.

    Returns a mapping {category: absolute_path_to_ttm}
    """
    rlb = _rlb_repo_root()
    base_scene = rlb / 'task_design.ttt'
    if out_dir is None:
        out_dir = Path(__file__).resolve().parent / 'rlb_assets'
    out_dir = Path(out_dir)

    cat_map = _category_mapping()
    out_map: Dict[str, str] = {}

    # Launch one PyRep session for all extractions
    pr = _launch_pyrep(base_scene, headless=headless)
    try:
        for cat, (src_ttm, obj_name) in cat_map.items():
            dst = out_dir / f'{_slug(cat)}.ttm'
            if dst.exists():
                out_map[cat] = str(dst)
                continue

            if not src_ttm.exists():
                raise FileNotFoundError(f'Source TTM missing for {cat}: {src_ttm}')

            # Import source scene/model (task or asset)
            base = pr.import_model(str(src_ttm))

            # Try to fetch the named shape; fallback: try lowercase/variant names
            shape = _find_shape_or_none(obj_name)
            if shape is None:
                # try a few variants
                candidates = [
                    obj_name,
                    obj_name.lower(),
                    obj_name.replace(' ', '_'),
                    obj_name.replace(' ', '').lower(),
                ]
                for cand in candidates:
                    shape = _find_shape_or_none(cand)
                    if shape is not None:
                        break

            if shape is None:
                # As a last resort, pick any Shape child of imported base whose name contains a token
                token = obj_name.split('_')[0].lower()
                try:
                    from pyrep.objects.object import Object
                    children = base.get_objects_in_tree()
                    for ch in children:
                        if isinstance(ch, Shape) and token in ch.get_name().lower():
                            shape = ch
                            break
                except Exception:
                    pass

            if shape is None:
                # Cleanup and fail this category
                base.remove()
                raise RuntimeError(f'Could not find shape "{obj_name}" inside {src_ttm}')

            # Save shape as standalone TTM
            _save_shape_model(shape, dst)
            out_map[cat] = str(dst)

            # Cleanup imported source
            try:
                base.remove()
            except Exception:
                pass

    finally:
        pr.shutdown()

    return out_map


if __name__ == '__main__':
    mapping = ensure_assets()
    print('Extracted/verified assets:')
    for k, v in mapping.items():
        print(f'  - {k}: {v}')
