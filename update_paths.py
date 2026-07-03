#!/usr/bin/env python3
"""Helper script to update dataset and checkpoint paths in all configuration files.

This script replaces placeholder paths ($PATH_TO_XXX$) with actual paths.

Usage:
    python update_paths.py \
        --dataset-root /path/to/dataset \
        --benchmark-root /path/to/Occupancy_Giga-benchmark \
        --ckpts-root /path/to/checkpoints

Or interactive mode:
    python update_paths.py
"""

import argparse
import os
import re
from pathlib import Path


def update_paths_in_file(filepath, dataset_root, benchmark_root, ckpts_root, dry_run=False):
    """Update paths in a single file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Replace placeholder paths with actual paths
    # Data paths
    if dataset_root:
        content = content.replace('$PATH_TO_DATASET$/Data_indoor', f'{dataset_root}/Data_indoor')
        content = content.replace('$PATH_TO_DATASET$/Data_outdoor', f'{dataset_root}/Data_outdoor')
        content = content.replace('$PATH_TO_DATASET$/Data_mix', f'{dataset_root}/Data_mix')
    
    # Benchmark root path
    if benchmark_root:
        content = content.replace('$PATH_TO_OCCUPANCY_GIGA_BENCHMARK$', benchmark_root)
    
    # Checkpoint path
    if ckpts_root:
        content = content.replace('$PATH_TO_CKPTS$', ckpts_root)
    
    if content != original_content:
        if not dry_run:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
        return True
    return False


def main():
    parser = argparse.ArgumentParser(
        description='Update dataset and checkpoint paths in configuration files from placeholders to actual paths',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Update all paths
  python update_paths.py \\
      --dataset-root /mnt/dataset/xgr94a/RoboPanda \\
      --benchmark-root /mnt/dataset/xgr94a/RoboPanda/Code/fangzheng/Occupancy_Giga-benchmark \\
      --ckpts-root /mnt/dataset/xgr94a/RoboPanda/Code/fangzheng/checkpoint

  # Preview changes without applying
  python update_paths.py --dry-run \\
      --dataset-root /path/to/data \\
      --benchmark-root /path/to/benchmark \\
      --ckpts-root /path/to/ckpts

  # Only update dataset paths
  python update_paths.py --dataset-root /path/to/data

  # Only update checkpoint paths
  python update_paths.py --ckpts-root /path/to/checkpoints
        """
    )
    parser.add_argument(
        '--dataset-root',
        help='Root path to dataset directory (contains Data_indoor, Data_outdoor, Data_mix)'
    )
    parser.add_argument(
        '--benchmark-root',
        help='Root path to Occupancy_Giga-benchmark code directory'
    )
    parser.add_argument(
        '--ckpts-root',
        help='Root path to checkpoint directory (for pre-trained weights like resnet50-0676ba61.pth)'
    )
    parser.add_argument(
        '--config-dir',
        default='configs/exp',
        help='Directory containing config files (default: configs/exp)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be changed without making changes'
    )
    
    args = parser.parse_args()
    
    # At least one path must be specified
    if not any([args.dataset_root, args.benchmark_root, args.ckpts_root]):
        parser.print_help()
        print("\n❌ Error: At least one of --dataset-root, --benchmark-root, or --ckpts-root must be specified.")
        return
    
    # Expand user home directory for all paths
    dataset_root = os.path.expanduser(args.dataset_root) if args.dataset_root else None
    benchmark_root = os.path.expanduser(args.benchmark_root) if args.benchmark_root else None
    ckpts_root = os.path.expanduser(args.ckpts_root) if args.ckpts_root else None
    
    # Verify paths exist (unless dry-run)
    if not args.dry_run:
        if dataset_root and not os.path.exists(dataset_root):
            print(f"⚠️  Warning: Dataset root does not exist: {dataset_root}")
        if benchmark_root and not os.path.exists(benchmark_root):
            print(f"⚠️  Warning: Benchmark root does not exist: {benchmark_root}")
        if ckpts_root and not os.path.exists(ckpts_root):
            print(f"⚠️  Warning: Checkpoints root does not exist: {ckpts_root}")
            print(f"   Please create it and download pre-trained weights.")
    
    # Check for ResNet-50 weights
    if ckpts_root and os.path.exists(ckpts_root):
        resnet_path = Path(ckpts_root) / 'resnet50-0676ba61.pth'
        if not resnet_path.exists():
            print(f"\n⚠️  Warning: ResNet-50 pretrained weights not found at {resnet_path}")
            print(f"   Download with: wget -P {ckpts_root} https://download.pytorch.org/models/resnet50-0676ba61.pth")
    
    # Find all config files
    config_dir = Path(args.config_dir)
    if not config_dir.exists():
        print(f"❌ Config directory not found: {config_dir}")
        return
    
    config_files = list(config_dir.glob('*.py'))
    
    if not config_files:
        print(f"❌ No .py files found in {config_dir}")
        return
    
    print(f"{'[DRY RUN] ' if args.dry_run else ''}Updating paths in {len(config_files)} files...")
    if dataset_root:
        print(f"  Dataset root:   {dataset_root}")
    if benchmark_root:
        print(f"  Benchmark root: {benchmark_root}")
    if ckpts_root:
        print(f"  Checkpoints:    {ckpts_root}")
    print()
    
    updated = 0
    for config_file in sorted(config_files):
        if update_paths_in_file(config_file, dataset_root, benchmark_root, ckpts_root, args.dry_run):
            print(f"  {'[DRY RUN] ' if args.dry_run else ''}✓ {config_file.name}")
            updated += 1
        else:
            print(f"  {'[DRY RUN] ' if args.dry_run else ''}○ {config_file.name} (no changes)")
    
    print()
    if args.dry_run:
        print(f"[DRY RUN] Would update {updated} files. Run without --dry-run to apply changes.")
    else:
        print(f"✅ Updated {updated} configuration files successfully!")
        
    # Final instructions
    if not args.dry_run and ckpts_root and not os.path.exists(ckpts_root):
        print(f"\n📋 Next steps:")
        print(f"   1. Create checkpoint directory: mkdir -p {ckpts_root}")
        print(f"   2. Download ResNet-50: wget -P {ckpts_root} https://download.pytorch.org/models/resnet50-0676ba61.pth")
        if dataset_root and not os.path.exists(f"{dataset_root}/Data_indoor"):
            print(f"   3. Download datasets from HuggingFace (see README.md)")


if __name__ == '__main__':
    main()
