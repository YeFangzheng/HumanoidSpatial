#!/usr/bin/env python3
"""Environment check script for Occupancy Giga-benchmark."""

import sys

def check_python_version():
    """Check Python version."""
    print("Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"  ✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"  ✗ Python {version.major}.{version.minor}.{version.micro} (requires >=3.8)")
        return False

def check_torch():
    """Check PyTorch installation."""
    print("Checking PyTorch...")
    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__}")
        
        # Check CUDA
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"    CUDA version: {torch.version.cuda}")
        else:
            print("  ⚠ CUDA not available (CPU only)")
        return True
    except ImportError:
        print("  ✗ PyTorch not found")
        return False

def check_openmmlab():
    """Check OpenMMLab packages."""
    print("Checking OpenMMLab packages...")
    packages = {
        'mmengine': 'MMEngine',
        'mmcv': 'MMCV',
        'mmdet': 'MMDet',
        'mmdet3d': 'MMDet3D',
        'mmseg': 'MMSegmentation'
    }
    
    all_ok = True
    for module, name in packages.items():
        try:
            mod = __import__(module)
            version = getattr(mod, '__version__', 'unknown')
            print(f"  ✓ {name}: {version}")
        except ImportError:
            print(f"  ✗ {name}: not found")
            all_ok = False
    
    return all_ok

def check_additional_deps():
    """Check additional dependencies."""
    print("Checking additional dependencies...")
    deps = [
        ('numpy', 'NumPy'),
        ('scipy', 'SciPy'),
        ('cv2', 'OpenCV'),
        ('PIL', 'Pillow'),
        ('matplotlib', 'Matplotlib'),
        ('timm', 'Timm'),
        ('spconv', 'SpConv'),
    ]
    
    all_ok = True
    for module, name in deps:
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name}: not found")
            all_ok = False
    
    return all_ok

def check_cuda_extensions():
    """Check CUDA extensions."""
    print("Checking CUDA extensions...")
    
    extensions = [
        ('mmdet3d.ops.voxelize', 'voxelize_ext'),
        ('mmdet3d.ops.dvr', 'dvr_ext'),
        ('mmdet3d.ops.bev_pool_v2', 'bev_pool_v2_ext'),
    ]
    
    all_ok = True
    for module, name in extensions:
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError as e:
            print(f"  ⚠ {name}: {e}")
    
    return all_ok

def check_gaussianformer_deps():
    """Check GaussianFormer specific dependencies."""
    print("Checking GaussianFormer dependencies...")
    
    try:
        import local_aggregate
        print("  ✓ local_aggregate")
    except ImportError:
        print("  ⚠ local_aggregate (optional, for GaussianFormer)")
    
    try:
        import local_aggregate_prob
        print("  ✓ local_aggregate_prob")
    except ImportError:
        print("  ⚠ local_aggregate_prob (optional, for GaussianFormer)")
    
    try:
        import local_aggregate_prob_fast
        print("  ✓ local_aggregate_prob_fast")
    except ImportError:
        print("  ⚠ local_aggregate_prob_fast (optional, for GaussianFormer)")

def main():
    """Main check function."""
    print("=" * 50)
    print("Occupancy Giga-benchmark Environment Check")
    print("=" * 50)
    print()
    
    results = []
    
    results.append(("Python", check_python_version()))
    print()
    
    results.append(("PyTorch", check_torch()))
    print()
    
    results.append(("OpenMMLab", check_openmmlab()))
    print()
    
    results.append(("Additional", check_additional_deps()))
    print()
    
    results.append(("CUDA Extensions", check_cuda_extensions()))
    print()
    
    check_gaussianformer_deps()
    print()
    
    # Summary
    print("=" * 50)
    print("Summary")
    print("=" * 50)
    
    for name, ok in results:
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"  {name}: {status}")
    
    all_ok = all(r[1] for r in results)
    print()
    if all_ok:
        print("✓ Environment check passed! Ready to train and evaluate.")
        return 0
    else:
        print("✗ Some checks failed. Please install missing dependencies.")
        print("  Run: bash install.sh")
        return 1

if __name__ == '__main__':
    sys.exit(main())
