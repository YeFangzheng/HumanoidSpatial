#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import time
from typing import Dict, List, Any, Tuple


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def backup_file(path: str) -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    bak = f"{path}.bak_{ts}"
    # 只复制，不删除原文件
    with open(path, "rb") as fsrc, open(bak, "wb") as fdst:
        fdst.write(fsrc.read())
    return bak


def index_by_token(items: List[Dict[str, Any]], name: str) -> Dict[str, Dict[str, Any]]:
    idx = {}
    for it in items:
        if "token" not in it:
            raise RuntimeError(f"[{name}] missing 'token' field in item: {it.keys()}")
        tok = it["token"]
        if tok in idx:
            raise RuntimeError(f"[{name}] duplicate token inside same file: {tok}")
        idx[tok] = it
    return idx


def check_no_overlap(a: Dict[str, Any], b: Dict[str, Any], what: str):
    inter = set(a.keys()) & set(b.keys())
    if inter:
        # 只打印少量示例，避免刷屏
        sample = list(sorted(inter))[:10]
        raise RuntimeError(f"[CONFLICT] {what} token overlap detected: count={len(inter)} sample={sample}")


def compute_clip_id_offset(dst_clips: List[Dict[str, Any]]) -> int:
    max_id = -1
    for c in dst_clips:
        if "clip_id" in c:
            try:
                max_id = max(max_id, int(c["clip_id"]))
            except Exception:
                pass
    return max_id + 1


def remap_clip_ids(src_clips: List[Dict[str, Any]], start_id: int) -> Tuple[List[Dict[str, Any]], int]:
    """
    将 src 的 clip_id 重新映射到 [start_id, start_id+len-1]，保证不撞。
    """
    new = []
    cur = start_id
    for c in src_clips:
        cc = dict(c)
        cc["clip_id"] = cur
        cur += 1
        new.append(cc)
    return new, cur


def sanity_check_links(
    clips: List[Dict[str, Any]],
    frames: List[Dict[str, Any]],
    max_check_clips: int = 50,
    max_check_frames_per_clip: int = 50,
) -> List[str]:
    """
    保守一致性检查（抽样），发现问题就返回错误列表（不直接 raise，方便统一报）。
    """
    errs = []
    frame_idx = index_by_token(frames, "frames(merged)")

    # 抽样检查前 max_check_clips 个 clip，每个 clip 抽样前 max_check_frames_per_clip 个 frame token
    for ci, clip in enumerate(clips[:max_check_clips]):
        clip_tok = clip.get("token")
        fr_list = clip.get("frames", [])
        if not isinstance(fr_list, list):
            errs.append(f"[clip {clip_tok}] frames field is not a list")
            continue

        for ftok in fr_list[:max_check_frames_per_clip]:
            fr = frame_idx.get(ftok)
            if fr is None:
                errs.append(f"[clip {clip_tok}] frame token missing in frames.json: {ftok}")
                continue
            st = fr.get("scene_token")
            if st != clip_tok:
                errs.append(f"[frame {ftok}] scene_token mismatch: {st} != clip_token {clip_tok}")

    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dst", required=True, help="Destination dataset root, e.g. .../Data_indoor")
    ap.add_argument("--src", required=True, help="Source dataset root, e.g. .../Data_indoor2")
    ap.add_argument("--apply", action="store_true", help="Actually write merged json into dst (with backups)")
    ap.add_argument("--no_remap_clip_id", action="store_true", help="Do NOT remap src clip_id (NOT recommended)")
    ap.add_argument("--check", action="store_true", help="Run sanity checks (sampled) after merge in memory")
    ap.add_argument("--max_check_clips", type=int, default=50)
    ap.add_argument("--max_check_frames_per_clip", type=int, default=50)
    args = ap.parse_args()

    dst_clips_path = os.path.join(args.dst, "clips.json")
    dst_frames_path = os.path.join(args.dst, "frames.json")
    src_clips_path = os.path.join(args.src, "clips.json")
    src_frames_path = os.path.join(args.src, "frames.json")

    for p in [dst_clips_path, dst_frames_path, src_clips_path, src_frames_path]:
        if not os.path.exists(p):
            raise RuntimeError(f"Missing file: {p}")

    print(f"[LOAD] dst clips:  {dst_clips_path}")
    print(f"[LOAD] dst frames: {dst_frames_path}")
    print(f"[LOAD] src clips:  {src_clips_path}")
    print(f"[LOAD] src frames: {src_frames_path}")

    dst_clips = load_json(dst_clips_path)
    dst_frames = load_json(dst_frames_path)
    src_clips = load_json(src_clips_path)
    src_frames = load_json(src_frames_path)

    if not isinstance(dst_clips, list) or not isinstance(src_clips, list):
        raise RuntimeError("clips.json should be a JSON array (list).")
    if not isinstance(dst_frames, list) or not isinstance(src_frames, list):
        raise RuntimeError("frames.json should be a JSON array (list).")

    dst_clip_idx = index_by_token(dst_clips, "dst_clips")
    src_clip_idx = index_by_token(src_clips, "src_clips")
    dst_frame_idx = index_by_token(dst_frames, "dst_frames")
    src_frame_idx = index_by_token(src_frames, "src_frames")

    # 1) token 绝不允许重叠（你说 token 不冲突，这里就用来“验证假设”）
    check_no_overlap(dst_clip_idx, src_clip_idx, "clip")
    check_no_overlap(dst_frame_idx, src_frame_idx, "frame")

    # 2) clip_id：即便 token 不冲突，也可能 clip_id 撞（很多下游会用 clip_id 分组）
    if args.no_remap_clip_id:
        # 仍然检查一下是否有撞（撞了就拒绝）
        dst_ids = set()
        for c in dst_clips:
            if "clip_id" in c:
                dst_ids.add(int(c["clip_id"]))
        inter = []
        for c in src_clips:
            if "clip_id" in c and int(c["clip_id"]) in dst_ids:
                inter.append(int(c["clip_id"]))
        if inter:
            inter = sorted(set(inter))
            raise RuntimeError(f"[CONFLICT] clip_id overlap detected (use remap): count={len(inter)} sample={inter[:10]}")
        src_clips_merged = src_clips
        remap_info = "[clip_id] keep original (no remap)"
    else:
        start_id = compute_clip_id_offset(dst_clips)
        src_clips_merged, end_id = remap_clip_ids(src_clips, start_id)
        remap_info = f"[clip_id] remapped src clip_id to range [{start_id}, {end_id-1}]"

    merged_clips = dst_clips + src_clips_merged
    merged_frames = dst_frames + src_frames

    print("============================================================")
    print(f"[OK] no clip token overlap:  dst={len(dst_clips)} src={len(src_clips)} -> merged={len(merged_clips)}")
    print(f"[OK] no frame token overlap: dst={len(dst_frames)} src={len(src_frames)} -> merged={len(merged_frames)}")
    print(f"{remap_info}")
    print("============================================================")

    # 3) 可选一致性检查（抽样）
    if args.check:
        errs = sanity_check_links(
            merged_clips,
            merged_frames,
            max_check_clips=args.max_check_clips,
            max_check_frames_per_clip=args.max_check_frames_per_clip,
        )
        if errs:
            print("[CHECK] sanity check FAILED (show up to 20):")
            for e in errs[:20]:
                print("  ", e)
            raise RuntimeError(f"Sanity check failed: {len(errs)} issues found.")
        else:
            print("[CHECK] sanity check PASSED (sampled).")

    if not args.apply:
        print("[DRY-RUN] Not writing anything. Re-run with --apply to write merged json (with backups).")
        return

    # 4) 写入：先备份再写（双保险）
    print("[APPLY] backing up dst json files...")
    bak1 = backup_file(dst_clips_path)
    bak2 = backup_file(dst_frames_path)
    print(f"[BACKUP] {bak1}")
    print(f"[BACKUP] {bak2}")

    print("[APPLY] writing merged json...")
    save_json(dst_clips_path, merged_clips)
    save_json(dst_frames_path, merged_frames)
    print("[DONE] merged json written successfully.")


if __name__ == "__main__":
    main()
