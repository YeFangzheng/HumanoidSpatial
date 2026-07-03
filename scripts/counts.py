#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import numpy as np
from collections import Counter, defaultdict

IGNORE_LABEL = 255

def iter_npz_files(root_dir: str):
    """
    Iterate all .npz under:
      {root_dir}/annotation/occ/**.npz
    """
    occ_root = os.path.join(root_dir, "annotation", "occ")
    if not os.path.isdir(occ_root):
        raise FileNotFoundError(f"Not found occ dir: {occ_root}")

    for dirpath, _, filenames in os.walk(occ_root):
        for fn in filenames:
            if fn.endswith(".npz"):
                yield os.path.join(dirpath, fn)

def load_occ_array(npz_path: str) -> np.ndarray:
    """
    Load occupancy array from npz.
    Prefer key 'occ', else fall back to the first key.
    """
    data = np.load(npz_path)
    if "occ" in data.files:
        occ = data["occ"]
    else:
        # fallback to the first key
        occ = data[data.files[0]]
    return occ

def update_counter(counter: Counter, occ: np.ndarray):
    """
    Update class counts from occ array, excluding IGNORE_LABEL.
    """
    flat = occ.reshape(-1)
    # count ignore separately
    ignore_cnt = int(np.sum(flat == IGNORE_LABEL))
    valid = flat[flat != IGNORE_LABEL]
    if valid.size > 0:
        # bincount is faster than Counter for int labels
        maxv = int(valid.max())
        bc = np.bincount(valid.astype(np.int64), minlength=maxv + 1)
        for cls_id, cnt in enumerate(bc):
            if cnt:
                counter[int(cls_id)] += int(cnt)
    return ignore_cnt, int(valid.size)

def summarize(counter: Counter, total_valid: int):
    """
    Return list of dict rows sorted by class_id.
    """
    rows = []
    for cls_id in sorted(counter.keys()):
        cnt = counter[cls_id]
        ratio = (cnt / total_valid) if total_valid > 0 else 0.0
        rows.append({
            "class_id": int(cls_id),
            "voxel_count": int(cnt),
            "ratio": float(ratio),
        })
    return rows

def run_one_split(root_dir: str, split_name: str, max_files: int = -1, verbose: bool = True):
    counter = Counter()
    n_files = 0
    total_ignore = 0
    total_valid = 0

    for npz_path in iter_npz_files(root_dir):
        occ = load_occ_array(npz_path)
        ignore_cnt, valid_cnt = update_counter(counter, occ)
        total_ignore += ignore_cnt
        total_valid += valid_cnt
        n_files += 1

        if verbose and n_files % 200 == 0:
            print(f"[{split_name}] processed {n_files} files... valid_vox={total_valid:,} ignore255={total_ignore:,}")

        if max_files > 0 and n_files >= max_files:
            break

    rows = summarize(counter, total_valid)
    result = {
        "split": split_name,
        "root_dir": root_dir,
        "num_files": n_files,
        "total_valid_voxels_excluding_255": int(total_valid),
        "total_255_voxels": int(total_ignore),
        "class_stats": rows,
        "raw_counts": {str(k): int(v) for k, v in sorted(counter.items())},
    }
    return result

def save_json(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def save_csv(class_stats, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("class_id,voxel_count,ratio\n")
        for r in class_stats:
            f.write(f"{r['class_id']},{r['voxel_count']},{r['ratio']}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--indoor", default="$PATH_TO_DATASET$/Data_indoor", type=str)
    parser.add_argument("--outdoor", default="$PATH_TO_DATASET$/Data_outdoor", type=str)
    parser.add_argument("--max_files", default=-1, type=int, help="debug: only process first N files per split")
    parser.add_argument("--out_dir", default="./occ_class_freq", type=str, help="where to save json/csv")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    verbose = not args.quiet

    indoor_res = run_one_split(args.indoor, "indoor", max_files=args.max_files, verbose=verbose)
    outdoor_res = run_one_split(args.outdoor, "outdoor", max_files=args.max_files, verbose=verbose)

    # print brief summary
    def print_top(res, topk=10):
        stats = sorted(res["class_stats"], key=lambda x: x["voxel_count"], reverse=True)
        print(f"\n===== {res['split']} =====")
        print(f"root: {res['root_dir']}")
        print(f"files: {res['num_files']}")
        print(f"valid_vox(excl.255): {res['total_valid_voxels_excluding_255']:,}")
        print(f"vox(255): {res['total_255_voxels']:,}")
        print(f"num_classes_present: {len(res['class_stats'])}")
        print(f"Top-{topk} classes by voxel_count:")
        for r in stats[:topk]:
            print(f"  cls={r['class_id']:>3}  cnt={r['voxel_count']:<12}  ratio={r['ratio']:.6f}")

    if verbose:
        print_top(indoor_res, topk=15)
        print_top(outdoor_res, topk=15)

    # save outputs
    save_json(indoor_res, os.path.join(args.out_dir, "indoor_occ_class_freq.json"))
    save_json(outdoor_res, os.path.join(args.out_dir, "outdoor_occ_class_freq.json"))
    save_csv(indoor_res["class_stats"], os.path.join(args.out_dir, "indoor_occ_class_freq.csv"))
    save_csv(outdoor_res["class_stats"], os.path.join(args.out_dir, "outdoor_occ_class_freq.csv"))

    # also save merged pack
    save_json({"indoor": indoor_res, "outdoor": outdoor_res},
              os.path.join(args.out_dir, "occ_class_freq_all.json"))

    if verbose:
        print(f"\n[OK] saved to: {args.out_dir}")

if __name__ == "__main__":
    main()