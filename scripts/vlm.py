#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate a minimal JSON (data-only) for VLM multiple-choice scene classification.

Each record corresponds to one clip:
  - clip_id
  - images: first-frame 6V paths (CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT, CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT)

Dataset layout (example):
$PATH_TO_DATASET$/Data_indoor/camera/CAM_BACK/<clip_id>/xxxxx.jpeg
"""

import os
import json
import glob
import argparse
from typing import Dict, List, Optional

CAM_ORDER = [
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
]

INDOOR_OPTIONS = [
    {"id": "factory",   "name": "工厂"},
    {"id": "medical",   "name": "医疗"},
    {"id": "lab",       "name": "实验室"},
    {"id": "home",      "name": "家居"},
    {"id": "office",    "name": "办公室"},
    {"id": "classroom", "name": "教室"},
    {"id": "retail",    "name": "零售"},
    {"id": "lobby",     "name": "大厅"},
]

OUTDOOR_OPTIONS = [
    {"id": "park",            "name": "公园"},
    {"id": "sports_ground",   "name": "运动场"},
    {"id": "square",          "name": "广场"},
    {"id": "street",          "name": "街道"},
    {"id": "campus",          "name": "校园"},
    {"id": "road",            "name": "马路"},
    {"id": "industrial_park", "name": "园区"},
]

IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")


def list_clip_ids(camera_root: str, anchor_cam: str = "CAM_BACK") -> List[str]:
    """
    List clip_id directories under camera_root/<anchor_cam>/.
    """
    anchor_dir = os.path.join(camera_root, anchor_cam)
    if not os.path.isdir(anchor_dir):
        raise FileNotFoundError(f"Anchor cam dir not found: {anchor_dir}")

    clip_ids = []
    for entry in os.listdir(anchor_dir):
        p = os.path.join(anchor_dir, entry)
        if os.path.isdir(p):
            clip_ids.append(entry)
    clip_ids.sort()
    return clip_ids


def pick_first_frame_image(clip_cam_dir: str) -> Optional[str]:
    """
    Pick the first frame by lexicographic order of filename within clip_cam_dir.
    """
    files = []
    for ext in IMG_EXTS:
        files.extend(glob.glob(os.path.join(clip_cam_dir, ext)))
    if not files:
        return None
    files.sort()
    return os.path.abspath(files[0])


def build_minimal_data_only(
    data_root: str,
    split: str,
    allow_missing_cam: bool = False,
    max_clips: int = -1,
) -> List[Dict]:
    camera_root = os.path.join(data_root, "camera")
    if not os.path.isdir(camera_root):
        raise FileNotFoundError(f"camera dir not found: {camera_root}")

    # NOTE: kept for "other things unchanged" (even though not used in data-only output)
    _options = INDOOR_OPTIONS if split == "indoor" else OUTDOOR_OPTIONS
    _ = _options

    clip_ids = list_clip_ids(camera_root, anchor_cam="CAM_BACK")
    if max_clips and max_clips > 0:
        clip_ids = clip_ids[:max_clips]

    data: List[Dict] = []

    for clip_id in clip_ids:
        images: Dict[str, str] = {}
        missing = []

        for cam in CAM_ORDER:
            clip_cam_dir = os.path.join(camera_root, cam, clip_id)
            img0 = pick_first_frame_image(clip_cam_dir)
            if img0 is None:
                missing.append(cam)
            else:
                images[cam] = img0

        if missing and not allow_missing_cam:
            continue

        data.append({
            "clip_id": clip_id,
            "images": images
        })

    return data


def main():
    parser = argparse.ArgumentParser(
        description="Generate data-only JSON for VLM scene classification (first-frame 6V per clip)."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root of Data_indoor or Data_outdoor, e.g. $PATH_TO_DATASET$/Data_indoor",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["indoor", "outdoor"],
        required=True,
        help="Choose option set for scene classification.",
    )
    parser.add_argument(
        "--out_json",
        type=str,
        required=True,
        help="Output json file path.",
    )
    parser.add_argument(
        "--allow_missing_cam",
        action="store_true",
        help="Allow clips with missing camera views (will include whatever exists). Default: skip if not full 6V.",
    )
    parser.add_argument(
        "--max_clips",
        type=int,
        default=-1,
        help="Debugging: if >0, only process first N clips.",
    )

    args = parser.parse_args()

    data_only = build_minimal_data_only(
        data_root=args.data_root,
        split=args.split,
        allow_missing_cam=args.allow_missing_cam,
        max_clips=args.max_clips,
    )

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(data_only, f, ensure_ascii=False, indent=2)

    print(f"[OK] wrote: {args.out_json}")
    print(f"     records={len(data_only)}")


if __name__ == "__main__":
    main()