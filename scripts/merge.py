#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import List


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def ensure_dir(p: Path, apply: bool, logf):
    if p.exists():
        return
    logf(f"[MKDIR] {p}")
    if apply:
        p.mkdir(parents=True, exist_ok=True)


def walk_all_files(root: Path) -> List[Path]:
    files = []
    for p in root.rglob("*"):
        if p.is_file():
            files.append(p)
    files.sort()
    return files


def copy_one_file(
    src_file: Path,
    dst_file: Path,
    apply: bool,
    logf,
    verify_hash_on_conflict: bool = False,
) -> None:
    if not dst_file.exists():
        logf(f"[COPY] {src_file}  ->  {dst_file}")
        if apply:
            ensure_dir(dst_file.parent, apply=True, logf=logf)
            shutil.copy2(src_file, dst_file)
        return

    # dst exists: strict handling
    src_stat = src_file.stat()
    dst_stat = dst_file.stat()

    if src_stat.st_size == dst_stat.st_size and verify_hash_on_conflict:
        src_h = sha256_file(src_file)
        dst_h = sha256_file(dst_file)
        if src_h == dst_h:
            logf(f"[SKIP-SAME] {src_file}  ==  {dst_file}")
            return

    logf(f"[CONFLICT] dst exists and differs: {dst_file}")
    logf(f"           src: {src_file} (size={src_stat.st_size})")
    logf(f"           dst: {dst_file} (size={dst_stat.st_size})")
    raise RuntimeError(f"Conflict detected (refuse to overwrite): {dst_file}")


def merge_json_unique_by_key(
    dst_json: Path,
    src_json: Path,
    key: str,
    apply: bool,
    logf,
) -> None:
    if not src_json.exists():
        logf(f"[JSON] src not found, skip: {src_json}")
        return

    if not dst_json.exists():
        logf(f"[JSON] dst not found, will copy whole file: {src_json} -> {dst_json}")
        if apply:
            ensure_dir(dst_json.parent, apply=True, logf=logf)
            shutil.copy2(src_json, dst_json)
        return

    with dst_json.open("r", encoding="utf-8") as f:
        dst_data = json.load(f)
    with src_json.open("r", encoding="utf-8") as f:
        src_data = json.load(f)

    if not isinstance(dst_data, list) or not isinstance(src_data, list):
        raise RuntimeError(f"JSON structure not list: {dst_json} or {src_json}")

    dst_keys = set()
    for item in dst_data:
        if isinstance(item, dict) and key in item:
            dst_keys.add(item[key])

    new_items = []
    conflicts = 0
    for item in src_data:
        if not isinstance(item, dict) or key not in item:
            continue
        k = item[key]
        if k in dst_keys:
            conflicts += 1
        else:
            new_items.append(item)

    if conflicts > 0:
        # 你的前提是 token 不冲突，如果这里冲突，说明数据假设被破坏，必须停
        raise RuntimeError(f"JSON token conflicts found in {dst_json.name}: {conflicts} items (refuse to merge)")

    if not new_items:
        logf(f"[JSON] no new items to merge for {dst_json.name}")
        return

    merged = dst_data + new_items
    logf(f"[JSON] merge {dst_json.name}: +{len(new_items)} new items (total={len(merged)})")

    if apply:
        backup = dst_json.with_suffix(dst_json.suffix + f".bak_{int(time.time())}")
        shutil.copy2(dst_json, backup)
        logf(f"[JSON] backup created: {backup}")
        with dst_json.open("w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False)
        logf(f"[JSON] written: {dst_json}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dst", required=True, type=str, help="Existing dataset root, e.g. Data_indoor")
    parser.add_argument("--src", required=True, type=str, help="New dataset root, e.g. Data_indoor2")
    parser.add_argument("--apply", action="store_true", help="Actually copy files (otherwise dry-run)")
    parser.add_argument("--log", type=str, default="", help="Log file path (default: ./merge_log_TIMESTAMP.txt)")
    parser.add_argument("--quiet", action="store_true", help="Less stdout; still write full log to file")
    parser.add_argument("--verify_hash_on_conflict", action="store_true",
                        help="If dst exists and size same, compare sha256 to allow skip-same; otherwise conflict.")
    parser.add_argument("--merge_json", action="store_true",
                        help="Merge clips.json and frames.json by unique token (creates backup on apply).")
    args = parser.parse_args()

    dst_root = Path(args.dst).resolve()
    src_root = Path(args.src).resolve()

    if not dst_root.exists() or not dst_root.is_dir():
        print(f"[FATAL] dst root not found/dir: {dst_root}")
        sys.exit(2)
    if not src_root.exists() or not src_root.is_dir():
        print(f"[FATAL] src root not found/dir: {src_root}")
        sys.exit(2)

    ts = int(time.time())
    log_path = Path(args.log) if args.log else Path(f"./merge_log_{ts}.txt")

    def logf(msg: str):
        # always write log file
        with log_path.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")
        # stdout depends on quiet
        if not args.quiet:
            print(msg)

    # Always print a tiny header even in quiet mode
    if args.quiet:
        print(f"[START] dst={dst_root}")
        print(f"[START] src={src_root}")
        print(f"[MODE]  {'APPLY' if args.apply else 'DRY-RUN'}")
        print(f"[LOG]   {log_path.resolve()}")

    logf("============================================================")
    logf(f"[START] dst={dst_root}")
    logf(f"[START] src={src_root}")
    logf(f"[MODE]  {'APPLY' if args.apply else 'DRY-RUN'}")
    logf(f"[LOG]   {log_path.resolve()}")
    logf("============================================================")

    # 关键：默认跳过 clips.json / frames.json 的“文件级复制”
    skip_relpaths = {Path("clips.json"), Path("frames.json")}

    src_files = walk_all_files(src_root)
    logf(f"[SCAN] src files: {len(src_files)}")

    copied = 0
    skipped_meta = 0
    skipped_same = 0

    try:
        for sf in src_files:
            rel = sf.relative_to(src_root)

            # skip meta json copy (handled by --merge_json)
            if rel in skip_relpaths:
                skipped_meta += 1
                logf(f"[SKIP-META] {rel} (use --merge_json to merge)")
                continue

            df = dst_root / rel

            # We only know skip-same after hashing (optional)
            if df.exists() and args.verify_hash_on_conflict and sf.stat().st_size == df.stat().st_size:
                # copy_one_file will hash and maybe skip
                pass

            copy_one_file(
                src_file=sf,
                dst_file=df,
                apply=args.apply,
                logf=logf,
                verify_hash_on_conflict=args.verify_hash_on_conflict,
            )

            # If no exception, either copied or skipped-same; we can approx count by existence:
            # Not perfect, but good enough for summary.
            if not df.exists():
                # dry-run case: file doesn't exist, so count as "would copy"
                copied += 1
            else:
                # could be copied in apply, or could have already existed (conflict would have raised)
                # For apply, still count as copied if we logged COPY.
                pass

        logf("------------------------------------------------------------")
        logf(f"[DONE] file stage finished. (meta json skipped: {skipped_meta})")
        logf("------------------------------------------------------------")

        if args.merge_json:
            logf("[JSON] merging clips.json / frames.json ...")
            merge_json_unique_by_key(
                dst_json=dst_root / "clips.json",
                src_json=src_root / "clips.json",
                key="token",
                apply=args.apply,
                logf=logf,
            )
            merge_json_unique_by_key(
                dst_json=dst_root / "frames.json",
                src_json=src_root / "frames.json",
                key="token",
                apply=args.apply,
                logf=logf,
            )
            logf("[JSON] merge done.")
        else:
            logf("[JSON] not merged (use --merge_json if needed).")

        logf("============================================================")
        logf("[SUCCESS] merge completed with zero conflicts.")
        logf("============================================================")

        if args.quiet:
            print("[SUCCESS] merge completed with zero conflicts.")
            print("Check log for details:", str(log_path.resolve()))

    except Exception as e:
        logf("============================================================")
        logf(f"[FAILED] {type(e).__name__}: {e}")
        logf("[ABORT] refusing to proceed further to avoid corruption.")
        logf("============================================================")

        if args.quiet:
            print(f"[FAILED] {type(e).__name__}: {e}")
            print("Check log for details:", str(log_path.resolve()))
        sys.exit(1)


if __name__ == "__main__":
    main()
