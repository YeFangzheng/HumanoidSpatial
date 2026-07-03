# HSP-benchmark

A comprehensive benchmark for 3D occupancy prediction with 10 state-of-the-art models on humanoid robot perception tasks.

## Supported Models

| Model | Indoor | Outdoor | Mix |
|-------|--------|---------|-----|
| [BEVDet](https://github.com/fundamentalvision/BEVDet) | ✅ | ✅ | ✅ |
| [BEVDet-COTR](https://github.com/zzelloss/BEVDet-COTR) | ✅ | ✅ | ✅ |
| [BEVFormer](https://github.com/fundamentalvision/BEVFormer) | ✅ | ✅ | ✅ |
| [FB-Occ](https://github.com/NVlabs/FB-BEV) | ✅ | ✅ | ✅ |
| [FlashOcc](https://github.com/Yzichen/FlashOCC) | ✅ | ✅ | ✅ |
| [GaussianFormer](https://github.com/NVLabs/GaussianFormer) | ✅ | ✅ | ✅ |
| [OccFormer](https://github.com/zhangyp15/OccFormer) | ✅ | ✅ | ✅ |
| [SparseOcc](https://github.com/MCG-NJU/SparseOcc) | ✅ | ✅ | ✅ |
| [SurroundOcc](https://github.com/weiyithu/SurroundOcc) | ✅ | ✅ | ✅ |
| [VoxFormer](https://github.com/NVlabs/VoxFormer) | ✅ | ✅ | ✅ |

## Installation

### Option 1: Conda (Recommended)

```bash
conda env create -f environment.yml
conda activate occupancy-giga

# Install CUDA extensions
cd mmdet3d/ops && python setup.py develop
cd ../..

# Install additional extensions for GaussianFormer
cd mmdet3d/models/GaussianFormer/model/head/localagg && pip install -e .
cd ../localagg_prob && pip install -e .
cd ../localagg_prob_fast && pip install -e .

pip install -e .
```

### Option 2: pip

```bash
pip install -r requirements.txt
pip install -e .
```

## Quick Start

### 1. Prepare Data

Download our pre-processed dataset from [HuggingFace](https://huggingface.co/datasets) (coming soon), or prepare your own data following [scripts/README.md](scripts/README.md).

Expected structure:
```
/path/to/dataset/
├── Data_indoor/
│   ├── train_frames.json
│   ├── val_frames.json
│   └── annotation/occ/
├── Data_outdoor/
└── Data_mix/
```

### 2. Update Paths

```bash
python update_paths.py \
    --dataset-root /path/to/dataset \
    --ckpts-root /path/to/checkpoints \
    --benchmark-root /path/to/Occupancy_Giga-benchmark
```

**Note**: Download [ResNet-50 pretrained weights](https://download.pytorch.org/models/resnet50-0676ba61.pth) to your checkpoints directory.

### 3. Training

```bash
export PYTHONPATH=$PWD:$PYTHONPATH

# Example: Train BEVDet on indoor data
torchrun --nproc_per_node=8 tools/train.py \
    configs/exp/bevdet_indoor.py \
    --launcher pytorch \
    --work-dir output/bevdet_indoor
```

Replace `bevdet_indoor.py` with other configs (e.g., `bevformer_outdoor.py`, `gaussianformer_mix.py`).

**Note for GaussianFormer**: Requires `mmsegmentation`:
```bash
pip install 'mmsegmentation>=1.2.0,<1.3.0'
```

### 4. Testing

```bash
# Same domain evaluation
torchrun --nproc_per_node=8 tools/test.py \
    configs/exp/bevdet_indoor.py \
    output/bevdet_indoor/epoch_20.pth \
    --launcher pytorch \
    --work-dir output_test/bevdet_indoor

# Cross-domain evaluation (indoor model on outdoor data)
torchrun --nproc_per_node=8 tools/test.py \
    configs/exp/bevdet_outdoor.py \
    output/bevdet_indoor/epoch_20.pth \
    --launcher pytorch \
    --work-dir output_test/bevdet_inout
```

### 5. Single GPU

Replace `torchrun --nproc_per_node=8` with `python`:
```bash
python tools/train.py configs/exp/bevdet_indoor.py --work-dir output/bevdet_indoor
```

## Pre-trained Weights

Download our trained model weights from [HuggingFace](https://huggingface.co) (coming soon):

| Model | Indoor | Outdoor | Mix |
|-------|--------|---------|-----|
| BEVDet | [🤗]() | [🤗]() | [🤗]() |
| BEVDet-COTR | [🤗]() | [🤗]() | [🤗]() |
| BEVFormer | [🤗]() | [🤗]() | [🤗]() |
| FB-Occ | [🤗]() | [🤗]() | [🤗]() |
| FlashOcc | [🤗]() | [🤗]() | [🤗]() |
| GaussianFormer | [🤗]() | [🤗]() | [🤗]() |
| OccFormer | [🤗]() | [🤗]() | [🤗]() |
| SparseOcc | [🤗]() | [🤗]() | [🤗]() |
| SurroundOcc | [🤗]() | [🤗]() | [🤗]() |
| VoxFormer | [🤗]() | [🤗]() | [🤗]() |

## Project Structure

```
Occupancy_Giga-benchmark/
├── configs/exp/          # Model configs (indoor/outdoor/mix)
├── mmdet3d/              # Models & evaluation
├── tools/                # train.py, test.py
└── scripts/              # Data preprocessing
```

## Evaluation Metrics

- **RayIoU**: IoU along camera rays
- **mIoU**: Mean IoU across all classes (excludes classes with 0 IoU)
- **Per-class IoU**: Individual class IoU

Results saved to `{work_dir}/raymetric_eval.json`.

## Troubleshooting

**CUDA kernel errors**: Reinstall mmcv with correct CUDA version:
```bash
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu124/torch2.5/index.html
```

**Multi-GPU hangs**: Set `export NCCL_IB_DISABLE=1`

## Citation

```bibtex
@article{HumanoidSpatial,
  title={HumanoidSpatial},
  year={2026}
}
```

## License

Apache License 2.0
