# Release Guide

Complete guide to release this codebase to GitHub.

## Prerequisites

1. **GitHub Account**: Have a GitHub account
2. **Personal Access Token**: Create at https://github.com/settings/tokens
   - Required scopes: `repo` (full control of private repositories)
   - Save the token securely

## Quick Release (Automated)

### Step 1: Configure

Create `release.config` file:

```bash
GITHUB_USERNAME="your_username"
GITHUB_REPO="Occupancy-Giga-benchmark"
GITHUB_TOKEN="ghp_xxxxxxxxxxxxxxxxxxxx"  # Your PAT
```

### Step 2: Run Release Script

```bash
chmod +x release.sh
./release.sh
```

This will:
- Initialize git repository
- Create initial commit
- Create GitHub repo (public or private)
- Push all code
- Create first release tag

## Manual Release (Step-by-Step)

### 1. Initialize Git Repository

```bash
cd /mnt/dataset/xgr94a/RoboPanda/Code/fangzheng/Occupancy_Giga-benchmark

# Initialize git
git init

# Configure git (if not already done)
git config user.name "Your Name"
git config user.email "your.email@example.com"

# Add all files
git add .

# Create initial commit
git commit -m "Initial release: Occupancy Giga-benchmark with 10 models

Features:
- 10 state-of-the-art occupancy prediction models
- Support for indoor, outdoor, and mixed datasets
- Unified training/testing interface
- Cross-domain evaluation support
- RayMetric evaluation with improved mIoU calculation

Models included:
- BEVDet, BEVDet-COTR, BEVFormer
- FB-Occ, FlashOcc, GaussianFormer
- OccFormer, SparseOcc, SurroundOcc, VoxFormer"
```

### 2. Create GitHub Repository

**Option A: Using GitHub CLI (gh)**

```bash
# Install gh if not available
# https://cli.github.com/

# Login
echo "YOUR_TOKEN" | gh auth login --with-token

# Create repository (public)
gh repo create Occupancy-Giga-benchmark --public --source=. --push

# Or private
gh repo create Occupancy-Giga-benchmark --private --source=. --push
```

**Option B: Using curl**

```bash
# Create repository via API
curl -H "Authorization: token YOUR_TOKEN" \
     -H "Accept: application/vnd.github.v3+json" \
     https://api.github.com/user/repos \
     -d '{
       "name": "Occupancy-Giga-benchmark",
       "description": "A comprehensive benchmark for 3D occupancy prediction with 10 state-of-the-art models",
       "private": false,
       "has_issues": true,
       "has_wiki": false,
       "has_downloads": true
     }'

# Add remote and push
git remote add origin https://github.com/YOUR_USERNAME/Occupancy-Giga-benchmark.git
git branch -M main
git push -u origin main
```

**Option C: Manual on GitHub Website**

1. Go to https://github.com/new
2. Repository name: `Occupancy-Giga-benchmark`
3. Description: `A comprehensive benchmark for 3D occupancy prediction with 10 state-of-the-art models`
4. Choose public or private
5. Don't initialize with README (we already have one)
6. Click "Create repository"
7. Follow the instructions on the page to push existing code

### 3. Create First Release

```bash
# Create a tag
git tag -a v1.0.0 -m "Initial release v1.0.0"

# Push tag
git push origin v1.0.0
```

Then on GitHub:
1. Go to repository → Releases → "Create a new release"
2. Choose tag: `v1.0.0`
3. Title: `v1.0.0 - Initial Release`
4. Description: Copy from the release notes below

## Release Notes Template

```markdown
## 🎉 Occupancy Giga-benchmark v1.0.0

A comprehensive benchmark for 3D occupancy prediction on humanoid robot perception tasks.

### ✨ Features

- **10 State-of-the-art Models**: BEVDet, BEVDet-COTR, BEVFormer, FB-Occ, FlashOcc, GaussianFormer, OccFormer, SparseOcc, SurroundOcc, VoxFormer
- **3 Dataset Domains**: Indoor, outdoor, and mixed scenarios
- **Unified Interface**: All models use the same training/testing commands
- **Cross-domain Evaluation**: Easy to evaluate models on different domains
- **RayMetric**: Improved mIoU calculation excluding classes with 0 IoU

### 📦 What's Included

- Complete source code for all 10 models
- Configuration files for indoor/outdoor/mix datasets
- Training and testing scripts with unified interface
- Data preprocessing scripts
- Environment setup files (conda/pip)

### 🚀 Quick Start

```bash
# Install
conda env create -f environment.yml
conda activate occupancy-giga
pip install -e .

# Update paths
python update_paths.py \
    --dataset-root /path/to/dataset \
    --ckpts-root /path/to/checkpoints \
    --benchmark-root /path/to/Occupancy_Giga-benchmark

# Train
torchrun --nproc_per_node=8 tools/train.py \
    configs/exp/bevdet_indoor.py \
    --launcher pytorch \
    --work-dir output/bevdet_indoor
```

### 📚 Documentation

- [README.md](README.md) - Full documentation
- [update_paths.py](update_paths.py) - Path configuration helper

### 📝 Citation

```bibtex
@article{occupancy_giga,
  title={Occupancy Giga-benchmark: A Comprehensive Benchmark for 3D Occupancy Prediction on Humanoid Robots},
  year={2024}
}
```

### 🔗 Links

- [Datasets on HuggingFace]() (Coming Soon)
- [Pre-trained Weights on HuggingFace]() (Coming Soon)

---
**Full Changelog**: https://github.com/YOUR_USERNAME/Occupancy-Giga-benchmark/commits/v1.0.0
```

## Post-Release Checklist

- [ ] Verify code is pushed to GitHub
- [ ] Verify README renders correctly
- [ ] Create first release with notes
- [ ] (Optional) Enable GitHub Pages for documentation
- [ ] (Optional) Add repository to HuggingFace datasets card
- [ ] Upload datasets to HuggingFace and update links in README
- [ ] Upload pre-trained weights to HuggingFace and update links

## Troubleshooting

**Large files**: If you have files > 100MB:
```bash
# Install git-lfs
git lfs install

# Track large files
git lfs track "*.pth"
git lfs track "*.ckpt"
git add .gitattributes
```

**Push rejected**: If push fails due to large files:
```bash
# Check what large files exist
find . -type f -size +50M

# Add to .gitignore if they shouldn't be in repo
echo "*.pth" >> .gitignore
echo "*.ckpt" >> .gitignore
echo "output/" >> .gitignore

# Remove from git cache if already added
git rm --cached path/to/large/file
```

## Next Steps After Release

1. **Upload to HuggingFace**:
   - Create dataset repositories for Data_indoor, Data_outdoor, Data_mix
   - Create model repositories for all 30 trained models
   - Update README.md with actual HuggingFace links

2. **Documentation**:
   - Add more examples to wiki or docs folder
   - Create tutorial notebooks (optional)

3. **Community**:
   - Share on social media / forums
   - Submit to paper reading lists
   - Respond to issues and PRs
