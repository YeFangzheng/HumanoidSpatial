"""
将 pred occ 的 frame_XXXXXX.npz 重命名为与 GT occ 一致的 LIDAR 文件名。

通过读取 GT occ 目录获取正确的文件名列表，按帧序号一一对应重命名。

用法：
    python tools/rename_pred_occ.py \
        --pred-dir vis_results/pred_occ \
        --gt-dir $PATH_TO_DATASET$/Data_indoor/annotation/occ

    # 先预览（不实际执行）
    python tools/rename_pred_occ.py \
        --pred-dir vis_results/pred_occ \
        --gt-dir $PATH_TO_DATASET$/Data_indoor/annotation/occ \
        --dry-run
"""

import os
import re
import shutil
import argparse


def get_sorted_npz(directory):
    """获取目录下所有 .npz 文件，按名称排序"""
    files = [f for f in os.listdir(directory) if f.endswith('.npz')]
    files.sort()
    return files


def main():
    parser = argparse.ArgumentParser(description='Rename pred occ files to match GT naming')
    parser.add_argument('--pred-dir', required=True, help='pred occ 根目录（包含 scene_token 子目录）')
    parser.add_argument('--gt-dir', required=True, help='GT occ 根目录（包含 scene_token 子目录）')
    parser.add_argument('--dry-run', action='store_true', help='只打印不执行')
    args = parser.parse_args()

    # 遍历 pred 下的每个 scene_token 子目录
    scene_tokens = [d for d in os.listdir(args.pred_dir)
                    if os.path.isdir(os.path.join(args.pred_dir, d))]

    if not scene_tokens:
        print(f"No scene_token directories found in {args.pred_dir}")
        return

    total_renamed = 0
    total_errors = 0

    for scene_token in sorted(scene_tokens):
        pred_scene_dir = os.path.join(args.pred_dir, scene_token)
        gt_scene_dir = os.path.join(args.gt_dir, scene_token)

        if not os.path.isdir(gt_scene_dir):
            print(f"[SKIP] GT dir not found: {gt_scene_dir}")
            continue

        pred_files = get_sorted_npz(pred_scene_dir)
        gt_files = get_sorted_npz(gt_scene_dir)

        # 检查数量是否匹配
        if len(pred_files) != len(gt_files):
            print(f"[WARN] {scene_token}: pred has {len(pred_files)} files, "
                  f"GT has {len(gt_files)} files — 数量不匹配，跳过")
            total_errors += 1
            continue

        print(f"\n[{scene_token}] {len(pred_files)} files")

        for pred_name, gt_name in zip(pred_files, gt_files):
            old_path = os.path.join(pred_scene_dir, pred_name)
            new_path = os.path.join(pred_scene_dir, gt_name)

            if pred_name == gt_name:
                # 已经是正确名称
                continue

            if args.dry_run:
                print(f"  {pred_name} -> {gt_name}")
            else:
                os.rename(old_path, new_path)

            total_renamed += 1

    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Done! "
          f"Renamed {total_renamed} files, {total_errors} errors")


if __name__ == '__main__':
    main()