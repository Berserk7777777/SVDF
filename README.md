# Few-Shot Action Recognition via SVD Features

Few-shot video action recognition framework leveraging Stable Video Diffusion (SVD) features with spatio-temporal meta-learning.

## Overview

This project extracts multi-layer, multi-timestep features from pre-trained Stable Video Diffusion models and trains a **SpatioTemporalTRXNet** meta-learner for few-shot action recognition on HMDB51, UCF101, and Kinetics400.

**Key Features:**
- Multi-layer UNet feature extraction (layers 5, 7, 12) at multiple diffusion timesteps (15, 23 out of 25 total steps)
- 3×3 spatial grid features for fine-grained spatio-temporal modeling
- Uniform temporal segmentation into 3 phases for efficient GPU processing
- Prototype-based matching with learnable spatial attention

---

## Installation

### 1. Clone Repository
```bash
git clone https://github.com/your-username/ActionDiff.git
cd ActionDiff/generative-models
```

### 2. Create Environment
```bash
conda create -n actiondiff python=3.10
conda activate actiondiff
```

### 3. Install Dependencies
```bash
# Install PyTorch (CUDA 11.8)
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# Install project dependencies
cd ..
pip install -r requirements.txt

# Install generative-models package
cd generative-models
pip install -e .
```

### 4. Download SVD Pretrained Weights

Download Stable Video Diffusion checkpoints and place in `generative-models/checkpoints/`:

```bash
mkdir -p checkpoints
cd checkpoints

# SVD (14 frames)
wget https://huggingface.co/stabilityai/stable-video-diffusion-img2vid/resolve/main/svd.safetensors

# SVD-XT (25 frames, recommended)
wget https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/resolve/main/svd_xt.safetensors

cd ..
```

**Required files:**
- `checkpoints/svd.safetensors` (9.56 GB)
- `checkpoints/svd_xt.safetensors` (9.56 GB)

---

## Dataset Preparation

### Directory Structure
```
ActionDiff/
├── generative-models/          # Main codebase
│   ├── scripts/
│   ├── models/
│   ├── datasets/
│   ├── checkpoints/            # SVD weights
│   │   ├── svd.safetensors
│   │   └── svd_xt.safetensors
│   └── results/                # Extracted features & trained models
├── splits/                     # Dataset splits
│   ├── hmdb_ARN/
│   ├── ucf_ARN/
│   └── kinetics_CMN/
├── hmdb51/
│   └── videos/                 # HMDB51 raw videos
│       ├── brush_hair/
│       ├── cartwheel/
│       └── ... (51 classes)
├── ucf101/
│   └── videos/                 # UCF101 raw videos
│       ├── ApplyEyeMakeup/
│       ├── Archery/
│       └── ... (101 classes)
└── kinetics400/
    └── videos/                 # Kinetics400 raw videos (optional)
```

### Download Datasets

**HMDB51:**
```bash
wget http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar
unrar x hmdb51_org.rar -d ../hmdb51/videos/
```

**UCF101:**
```bash
wget https://www.crcv.ucf.edu/data/UCF101/UCF101.rar
unrar x UCF101.rar -d ../ucf101/videos/
```

**Kinetics400:** Follow [official instructions](https://github.com/cvdfoundation/kinetics-dataset)

**Dataset Splits:**

The project uses standard few-shot splits:
- **HMDB51:** 31 base / 10 val / 10 test classes (ARN split)
- **UCF101:** 70 base / 10 val / 21 test classes (ARN split)
- **Kinetics400:** 64 base / 12 val / 24 test classes (CMN split)

Splits are included in `splits/` directory.

---

## Feature Extraction

Extract multi-layer SVD features at multiple timesteps and layers **simultaneously** in a single pass.

### Parameters
- **UNet Layers:** 5, 7 (mid layers), 12 (deep layer)
- **Diffusion Timesteps:** 23 (primary), 15 (secondary) — extracted simultaneously
- **Total diffusion steps:** 25
- **Feature stride:** 4 (temporal subsampling)
- **Offset frames:** 8 (starting frame offset)
- **Grid size:** 3×3 spatial patches
- **Output:** `(T, 9, 1920)` per video
  - `T`: temporal frames (subsampled with stride 4)
  - `9`: spatial patches (3×3 grid)
  - `1920`: concatenated channels (layer5@t23 + layer7@t23 + layer12@t23 + layer5@t15 + layer7@t15 + layer12@t15)

### HMDB51

```bash
cd generative-models

python scripts/generate_svd_maps.py \
    --dataset hmdb51 \
    --data_root ../hmdb51/videos \
    --output_folder results/hmdb51 \
    --exp_name extract_hmdb51 \
    --diffusion_step 23 \
    --diffusion_step2 15 \
    --mid_layer_idxs 5 7 \
    --deep_layer_idxs 12 \
    --feature_stride 4 \
    --offset_frames 8 \
    --device cuda
```

### UCF101

```bash
python scripts/generate_svd_maps.py \
    --dataset ucf101 \
    --data_root ../ucf101/videos \
    --output_folder results/ucf101 \
    --exp_name extract_ucf101 \
    --diffusion_step 23 \
    --diffusion_step2 15 \
    --mid_layer_idxs 5 7 \
    --deep_layer_idxs 12 \
    --feature_stride 4 \
    --offset_frames 8 \
    --device cuda
```

### Kinetics400

```bash
python scripts/generate_svd_maps.py \
    --dataset kinetics400 \
    --data_root ../kinetics400/videos \
    --output_folder results/kinetics400 \
    --exp_name extract_kinetics \
    --diffusion_step 23 \
    --diffusion_step2 15 \
    --mid_layer_idxs 5 7 \
    --deep_layer_idxs 12 \
    --feature_stride 4 \
    --offset_frames 8 \
    --device cuda
```

**Output:** Features saved as `{class}__{video}_grid.pt` in `results/{dataset}/extract_{dataset}_grid/feats/grid_3x3/`

---

## Training

Train SpatioTemporalTRXNet meta-learner on extracted features.

### HMDB51

**5-way 1-shot:**
```bash
python scripts/train_fsar.py \
    --dataset hmdb51 \
    --features_root results/hmdb51/extract_hmdb51_grid/feats/grid_3x3 \
    --splits_root ../splits/hmdb_ARN \
    --n_way 5 --k_shot 1 --n_query 15 \
    --hidden_dim 256 --n_phases 3 \
    --num_epochs 50 --batch_size 4 --lr 5e-5 \
    --save_dir results/fsar_hmdb51_5way1shot \
    --device cuda
```

**5-way 5-shot:**
```bash
python scripts/train_fsar.py \
    --dataset hmdb51 \
    --features_root results/hmdb51/extract_hmdb51_grid/feats/grid_3x3 \
    --splits_root ../splits/hmdb_ARN \
    --n_way 5 --k_shot 5 --n_query 15 \
    --hidden_dim 256 --n_phases 3 \
    --num_epochs 50 --batch_size 2 --lr 5e-5 \
    --save_dir results/fsar_hmdb51_5way5shot \
    --device cuda
```

### UCF101

**5-way 1-shot:**
```bash
python scripts/train_fsar.py \
    --dataset ucf101 \
    --features_root results/ucf101/extract_ucf101_grid/feats/grid_3x3 \
    --splits_root ../splits/ucf_ARN \
    --n_way 5 --k_shot 1 --n_query 15 \
    --hidden_dim 256 --n_phases 3 \
    --num_epochs 50 --batch_size 4 --lr 5e-5 \
    --save_dir results/fsar_ucf101_5way1shot \
    --device cuda
```

**5-way 5-shot:**
```bash
python scripts/train_fsar.py \
    --dataset ucf101 \
    --features_root results/ucf101/extract_ucf101_grid/feats/grid_3x3 \
    --splits_root ../splits/ucf_ARN \
    --n_way 5 --k_shot 5 --n_query 15 \
    --hidden_dim 256 --n_phases 3 \
    --num_epochs 50 --batch_size 2 --lr 5e-5 \
    --save_dir results/fsar_ucf101_5way5shot \
    --device cuda
```

### Kinetics400

**5-way 1-shot:**
```bash
python scripts/train_fsar.py \
    --dataset kinetics400 \
    --features_root results/kinetics400/extract_kinetics_grid/feats/grid_3x3 \
    --splits_root ../splits/kinetics_CMN \
    --n_way 5 --k_shot 1 --n_query 15 \
    --hidden_dim 256 --n_phases 3 \
    --num_epochs 50 --batch_size 4 --lr 5e-5 \
    --save_dir results/fsar_kinetics_5way1shot \
    --device cuda
```

**5-way 5-shot:**
```bash
python scripts/train_fsar.py \
    --dataset kinetics400 \
    --features_root results/kinetics400/extract_kinetics_grid/feats/grid_3x3 \
    --splits_root ../splits/kinetics_CMN \
    --n_way 5 --k_shot 5 --n_query 15 \
    --hidden_dim 256 --n_phases 3 \
    --num_epochs 50 --batch_size 2 --lr 5e-5 \
    --save_dir results/fsar_kinetics_5way5shot \
    --device cuda
```

**Training outputs:**
- `best_model.pth`: Best validation accuracy checkpoint
- `latest_model.pth`: Latest epoch checkpoint
- `training_log.txt`: Training metrics log

---

## Evaluation

Test trained models with **10,000 episodes** for robust 95% confidence intervals.

### HMDB51

**5-way 1-shot:**
```bash
python scripts/test_fsar.py \
    --ckpt_path results/fsar_hmdb51_5way1shot/best_model.pth \
    --dataset hmdb51 \
    --features_root results/hmdb51/extract_hmdb51_grid/feats/grid_3x3 \
    --splits_root ../splits/hmdb_ARN \
    --split test \
    --n_way 5 --k_shot 1 --n_query 15 \
    --n_episodes 10000 \
    --device cuda
```

**5-way 5-shot:**
```bash
python scripts/test_fsar.py \
    --ckpt_path results/fsar_hmdb51_5way5shot/best_model.pth \
    --dataset hmdb51 \
    --features_root results/hmdb51/extract_hmdb51_grid/feats/grid_3x3 \
    --splits_root ../splits/hmdb_ARN \
    --split test \
    --n_way 5 --k_shot 5 --n_query 15 \
    --n_episodes 10000 \
    --device cuda
```

### UCF101

**5-way 1-shot:**
```bash
python scripts/test_fsar.py \
    --ckpt_path results/fsar_ucf101_5way1shot/best_model.pth \
    --dataset ucf101 \
    --features_root results/ucf101/extract_ucf101_grid/feats/grid_3x3 \
    --splits_root ../splits/ucf_ARN \
    --split test \
    --n_way 5 --k_shot 1 --n_query 15 \
    --n_episodes 10000 \
    --device cuda
```

**5-way 5-shot:**
```bash
python scripts/test_fsar.py \
    --ckpt_path results/fsar_ucf101_5way5shot/best_model.pth \
    --dataset ucf101 \
    --features_root results/ucf101/extract_ucf101_grid/feats/grid_3x3 \
    --splits_root ../splits/ucf_ARN \
    --split test \
    --n_way 5 --k_shot 5 --n_query 15 \
    --n_episodes 10000 \
    --device cuda
```

### Kinetics400

**5-way 1-shot:**
```bash
python scripts/test_fsar.py \
    --ckpt_path results/fsar_kinetics_5way1shot/best_model.pth \
    --dataset kinetics400 \
    --features_root results/kinetics400/extract_kinetics_grid/feats/grid_3x3 \
    --splits_root ../splits/kinetics_CMN \
    --split test \
    --n_way 5 --k_shot 1 --n_query 15 \
    --n_episodes 10000 \
    --device cuda
```

**5-way 5-shot:**
```bash
python scripts/test_fsar.py \
    --ckpt_path results/fsar_kinetics_5way5shot/best_model.pth \
    --dataset kinetics400 \
    --features_root results/kinetics400/extract_kinetics_grid/feats/grid_3x3 \
    --splits_root ../splits/kinetics_CMN \
    --split test \
    --n_way 5 --k_shot 5 --n_query 15 \
    --n_episodes 10000 \
    --device cuda
```

**Output format:**
```
============================================================
  5-way 1-shot  |  10000 episodes
  Accuracy : 72.45% ± 0.28%  (95% CI)
============================================================
```

---

## Additional Tools

### 1. Complexity Analysis

Profile model parameters, FLOPs, latency, and GPU memory:

```bash
python scripts/profile_fsar.py \
    --ckpt_path results/fsar_hmdb51_5way1shot/best_model.pth \
    --n_patches 9 --T 8 \
    --hidden_dim 256 --n_phases 3 \
    --device cuda
```

**Output:**
- Parameters: ~1.0M
- FLOPs: ~0.7 GFLOPs/episode
- Latency: ~14ms/episode (GPU)
- Peak GPU memory: ~150 MB
- Theoretical complexity analysis

### 2. Cluster Visualization

Generate t-SNE visualizations showing feature clustering across 3 stages:

```bash
python scripts/visualize_clusters.py \
    --dataset hmdb51 \
    --features_root results/hmdb51/extract_hmdb51_grid/feats/grid_3x3 \
    --ckpt_path results/fsar_hmdb51_5way1shot/best_model.pth \
    --split test \
    --best_of 20 \
    --device cuda
```

**Output:** 3-stage t-SNE plots saved in `results/cluster_vis/`:
- Stage 1: Raw pixel features (before SVD)
- Stage 2: SVD-extracted features
- Stage 3: Model embeddings (after SpatioTemporalTRXNet)

---

## Model Architecture

### SpatioTemporalTRXNet

**Encoder (SpatioTemporalEncoder):**
1. Linear projection: D → H (1920 → 256)
2. Temporal + Spatial positional encoding
3. Spatio-temporal self-attention over T×P tokens (T=8 frames, P=9 patches)
4. Uniform temporal segmentation into K=3 phases (GPU-efficient, no CPU sync)
5. Phase-wise temporal pooling → (B, K, P, H)

**Matching Head (SpatioProtoHead):**
1. Learnable spatial patch attention weights (9 parameters)
2. Weighted spatial aggregation → (B, K, H)
3. Per-phase cosine similarity: query vs support prototypes
4. Multi-scale matching: per-phase + global similarity
5. Learnable temperature scaling

**Complexity:**
- **Time:** O(B·(T·P)²·H) for self-attention (dominant term)
- **Space:** O(B·T·P·H) activations + O(D·H + H²) parameters
- **Parameters:** ~1.0M
- **FLOPs:** ~0.7 GFLOPs/episode (5-way 1-shot)
- **Temporal segmentation:** Uniform 3-way split (O(1) per sample, fully GPU-parallelized)

---

## Requirements

**Core dependencies:**
- Python 3.10
- PyTorch 2.0.1
- torchvision 0.15.2
- CUDA 11.8+

**Key packages:**
- einops >= 0.6.1
- omegaconf >= 2.3.0
- pytorch-lightning == 2.0.1
- wandb >= 0.15.6 (optional, for logging)
- thop (optional, for FLOPs profiling)

See `requirements.txt` for complete list.

---

## Citation

If you use this code, please cite:

```bibtex
@article{fsar2024,
  title={Few-Shot Action Recognition via Stable Video Diffusion Features},
  author={Your Name},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year=2024}
}
```

---

## Acknowledgments

- [Stable Video Diffusion](https://github.com/Stability-AI/generative-models) for pre-trained models
- [ARN](https://github.com/zhangxueting/ARN) for HMDB51/UCF101 few-shot splits
- [CMN](https://github.com/ffmpbgrnn/CMN) for Kinetics400 few-shot splits

---

## License

This project is licensed under the MIT License.
