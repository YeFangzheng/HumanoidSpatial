#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
from collections import Counter

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description="RoboPanda JSON integrity checker (read-only).")
    parser.add_argument(
        "--root",
        type=str,
        default="$PATH_TO_DATASET$/Data_outdoor",  # <- 修改这里
        help="Dataset root containing clips.json and frames.json",
    )
    parser.add_argument("--show", type=int, default=20,
                        help="How many example entries to show for each error type")
    args = parser.parse_args()

    root = args.root
    clips_path = os.path.join(root, "clips.json")
    frames_path = os.path.join(root, "frames.json")

    if not os.path.exists(clips_path) or not os.path.exists(frames_path):
        print(f"[ERROR] Missing clips.json or frames.json under: {root}")
        print(f"  clips.json : {clips_path} (exists={os.path.exists(clips_path)})")
        print(f"  frames.json: {frames_path} (exists={os.path.exists(frames_path)})")
        return 2

    print(f"[ROOT] {root}")
    print("[LOAD] clips.json ...")
    clips = load_json(clips_path)
    print("[LOAD] frames.json ...")
    frames = load_json(frames_path)

    print(f"[INFO] #clips={len(clips)}  #frames={len(frames)}")

    # -----------------------------
    # 0) token 唯一性检查
    # -----------------------------
    clip_tokens = [c.get("token") for c in clips]
    frame_tokens = [f.get("token") for f in frames]

    dup_clip = [t for t, cnt in Counter(clip_tokens).items() if t is not None and cnt > 1]
    dup_frame = [t for t, cnt in Counter(frame_tokens).items() if t is not None and cnt > 1]

    if dup_clip:
        print("============================================================")
        print(f"[ERROR] Duplicate clip tokens detected: {len(dup_clip)}")
        for t in dup_clip[:args.show]:
            print(f"  dup clip token: {t}")
        print("============================================================")
    else:
        print("[OK] clip token unique")

    if dup_frame:
        print("============================================================")
        print(f"[ERROR] Duplicate frame tokens detected: {len(dup_frame)}")
        for t in dup_frame[:args.show]:
            print(f"  dup frame token: {t}")
        print("============================================================")
    else:
        print("[OK] frame token unique")

    clip_token_set = set([t for t in clip_tokens if t is not None])
    frame_token_set = set([t for t in frame_tokens if t is not None])

    # -----------------------------
    # 1) 孤儿帧：frame.scene_token 不在 clips.token
    # -----------------------------
    orphan_frames = []
    for fr in frames:
        st = fr.get("scene_token")
        if st not in clip_token_set:
            orphan_frames.append((fr.get("token"), st))

    if orphan_frames:
        print("============================================================")
        print(f"[ERROR] Orphan frames (frames.scene_token not in clips.token): {len(orphan_frames)}")
        for ft, st in orphan_frames[:args.show]:
            print(f"  orphan frame_token={ft}  scene_token={st}")
        print("============================================================")
    else:
        print("[OK] no orphan frames (forward)")

    # -----------------------------
    # 2) 反向孤儿：clips.frames 里引用的 frame token 不在 frames.token
    # -----------------------------
    missing_frame_refs = []  # (clip_token, missing_frame_token)
    total_refs = 0

    for c in clips:
        ct = c.get("token")
        fr_list = c.get("frames", [])
        if not isinstance(fr_list, list):
            print("============================================================")
            print(f"[ERROR] clip.frames is not a list! clip_token={ct} type={type(fr_list)}")
            print("============================================================")
            continue

        for ft in fr_list:
            total_refs += 1
            if ft not in frame_token_set:
                missing_frame_refs.append((ct, ft))

    if missing_frame_refs:
        print("============================================================")
        print(f"[ERROR] Missing frame references (clips.frames not found in frames.token): {len(missing_frame_refs)}")
        print(f"[INFO] total clip->frame references scanned: {total_refs}")
        for ct, ft in missing_frame_refs[:args.show]:
            print(f"  clip_token={ct}  missing_frame_token={ft}")
        print("============================================================")
    else:
        print("[OK] no reverse-orphan (all clip frame refs exist in frames.json)")
        print(f"[INFO] total clip->frame references scanned: {total_refs}")

    # -----------------------------
    # 3) 返回码：有错误则非 0
    # -----------------------------
    has_error = bool(dup_clip or dup_frame or orphan_frames or missing_frame_refs)
    if has_error:
        print("[RESULT] FAIL: integrity issues detected.")
        return 1
    else:
        print("[RESULT] PASS: no integrity issues detected.")
        return 0

if __name__ == "__main__":
    raise SystemExit(main())
