#!/bin/bash
# Installation script for Occupancy Giga-benchmark

set -e

echo "========================================"
echo "Occupancy Giga-benchmark Installation"
echo "========================================"
echo ""

# Check Python version
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "Detected Python version: $PYTHON_VERSION"

# Check CUDA availability
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA available:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
    echo "WARNING: nvidia-smi not found. CUDA may not be available."
fi

echo ""
echo "Step 1: Installing core dependencies..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo ""
echo "Step 2: Installing OpenMMLab dependencies..."
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install "mmdet>=3.0.0"
mim install "mmdet3d>=1.4.0"

echo ""
echo "Step 3: Installing additional dependencies..."
pip install timm spconv-cu121

echo ""
echo "Step 4: Installing package in development mode..."
pip install -e .

echo ""
echo "Step 5: Compiling CUDA extensions..."
cd mmdet3d/ops
python setup.py develop
cd ../..

echo ""
echo "Step 6: Installing GaussianFormer CUDA extensions..."
cd mmdet3d/models/GaussianFormer/model/head/localagg
pip install -e .
cd -
cd mmdet3d/models/GaussianFormer/model/head/localagg_prob
pip install -e .
cd -
cd mmdet3d/models/GaussianFormer/model/head/localagg_prob_fast
pip install -e .
cd -

echo ""
echo "Step 7: Verifying installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import mmdet3d; print(f'MMDet3D: {mmdet3d.__version__}')"
python -c "import mmengine; print(f'MMEngine: {mmengine.__version__}')"

echo ""
echo "========================================"
echo "Installation completed successfully!"
echo "========================================"
echo ""
echo "To get started:"
echo "  1. Prepare your data in Data_indoor/, Data_outdoor/, Data_mix/"
echo "  2. Run training: torchrun --nproc_per_node=8 tools/train.py configs/exp/bevdet_indoor.py --launcher pytorch --work-dir output/bevdet_indoor"
echo "  3. Run testing: torchrun --nproc_per_node=8 tools/test.py configs/exp/bevdet_indoor.py output/bevdet_indoor/epoch_20.pth --launcher pytorch --work-dir output_test/bevdet_indoor"
