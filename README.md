# ViPFast

ViPFast is a dynamic 3D Gaussian Splatting project for reconstructing and rendering time-varying scenes. The codebase extends a canonical Gaussian representation with learned deformation fields, affine velocity fields, FastGS-guided densification/pruning, and optional MPM-based physics regularization.

The main goal is to improve dynamic Gaussian training and rendering efficiency while keeping motion physically more plausible for dynamic objects and indoor dynamic scenes.

## What This Project Solves

Dynamic 3D Gaussian Splatting often faces three practical problems:

- **Inefficient Gaussian growth**: standard gradient-only densification can create many low-value Gaussians, increasing rasterization and optimizer cost.
- **Weak dynamic consistency**: learned deformation fields can fit training views but produce unstable or implausible motion, especially outside observed times.
- **High training/rendering cost**: dense splat sets and broad rasterization footprints slow down training, rendering, and evaluation.

ViPFast addresses these issues with:

- **FastGS-guided densification and pruning**: high-gradient Gaussians are only cloned or split when they consistently contribute to high-error pixels across sampled views.
- **FastGS rasterizer integration**: the custom rasterizer supports a footprint multiplier, metric-map driven per-Gaussian contribution counts, and optimized CUDA rendering/backward paths.
- **Dynamic canonical Gaussian modeling**: each canonical Gaussian is deformed according to frame time through learned code, deformation, and velocity fields.
- **12-DOF affine velocity field**: the default velocity model supports translation, rotation, stretching, and shearing, making it more expressive than a pure SE(3) field.
- **MPM physics regularization**: optional material-aware physical losses regularize deformation through volume, boundary, and elastic-energy terms.

## Method Overview

At training time, the system maintains a canonical `GaussianModel` and predicts a dynamic state for each camera timestamp:

```text
canonical xyz
    -> CodeField(xyz)
    -> DeformNetwork(xyz, time, code)
    -> VelocityField / AffineVelocityField
    -> dynamic xyz, rotation, scale
    -> FastGS Gaussian rasterizer
    -> reconstruction loss + optional MPM loss
```

The training loop has two stages:

1. **Warm-up stage**: train static Gaussians before enabling full dynamic deformation.
2. **Dynamic stage**: predict time-dependent Gaussian states and optimize reconstruction, deformation, velocity, and optional MPM material regularization.

During densification intervals, FastGS performs extra scoring passes on sampled cameras:

1. Render a view normally.
2. Build a high-error pixel mask from normalized per-pixel L1 error.
3. Render again with `metric_map` enabled.
4. Accumulate how often each Gaussian contributes to high-error pixels.
5. Use this score to gate clone/split and prune low-value Gaussians.

## Repository Structure

```text
.
├── train_gui.py                         # Main training entry, with optional DearPyGui viewer
├── render.py                            # Render train/val/test and interpolation videos
├── metrics.py                           # SSIM, PSNR, MAE-PSNR, LPIPS evaluation
├── seg.py                               # Gaussian clustering/segmentation utility
├── arguments/                           # CLI argument groups
├── gaussian_renderer/                   # Python render wrapper around CUDA rasterizer
├── scene/                               # Dataset loading, cameras, GaussianModel, DeformModel
├── utils/                               # Losses, FastGS scoring, physics/PINNs/velocity utilities
├── mpm_core/                            # MPM simulator, physics state, materials, kernels
├── submodules/
│   ├── diff-gaussian-rasterization_fastgs/
│   ├── simple-knn/
│   └── fused-ssim/
├── tests/                               # Lightweight utility tests
├── env.yml                              # Conda environment skeleton
├── requirements.txt                     # Python package pins
└── train.sh                             # Example training/evaluation commands
```

## Requirements

Recommended environment:

- Linux
- NVIDIA GPU with CUDA support
- CUDA toolkit with `nvcc`
- Conda or Miniconda
- Python 3.10
- PyTorch matching your CUDA version

The CUDA extensions are required for practical training:

- `diff-gaussian-rasterization_fastgs`
- `simple-knn`
- `fused-ssim`

MPM support additionally requires `warp-lang`. If Warp is not installed, the code prints a warning and disables MPM integration.

## Environment Setup

### 1. Create Conda Environment

```bash
conda create -n vipfast python=3.10 -y
conda activate vipfast
```

### 2. Install PyTorch

Install the PyTorch build that matches your CUDA driver/toolkit. Example for CUDA 12.4:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 3. Install Runtime Utilities

Some scripts import optional utilities that are not pinned in `requirements.txt`:

```bash
pip install dearpygui opencv-python warp-lang
```

### 4. Build CUDA Submodules

From the repository root:

```bash
pip install ./submodules/simple-knn
pip install ./submodules/fused-ssim
pip install ./submodules/diff-gaussian-rasterization_fastgs
pip install -r requirements.txt
```

## Data Preparation

The loader supports several dataset layouts:

- COLMAP: directory contains `sparse/`
- Blender/D-NeRF: directory contains `transforms_train.json`
- DTU-style: directory contains `cameras_sphere.npz`
- Nerfies/HyperNeRF: directory contains `dataset.json`
- Plenoptic Video: directory contains `poses_bounds.npy`

Set `-s` or `--source_path` to the dataset root. Outputs are written to `-m` or `--model_path`.

## Training

Basic command:

```bash
bash train.sh
```

Useful training options:

- `--iterations`: total training iterations. Default: `40000`.
- `--warm_up`: static warm-up iterations. Default: `3000`.
- `--max_time`: maximum normalized training time.
- `--fastgs_mult`: rasterizer footprint multiplier.
- `--fastgs_score_cameras`: number of sampled cameras for FastGS scoring.
- `--fastgs_loss_thresh`: high-error pixel threshold for metric-map construction.
- `--mpm_weight`: global MPM regularization weight.
- `--mpm_elastic_weight`: relative elastic-energy weight.
- `--mpm_grid_res`: MPM grid resolution.

Example commands for dynamic object and indoor scenes are collected in `train.sh`.

## Rendering

Render a trained model:

```bash
python render.py \
  -m output/your_experiment \
  --mode render \
  --skip_train
```

Other render modes:

```bash
python render.py -m output/your_experiment --mode time
python render.py -m output/your_experiment --mode view
python render.py -m output/your_experiment --mode all
python render.py -m output/your_experiment --mode pose
```

Rendered images are saved under:

```text
output/your_experiment/{train,val,test}/ours_<iteration>/
├── renders/
├── gt/
└── depth/
```

## Evaluation

Compute metrics on the rendered test split:

```bash
python metrics.py -m output/your_experiment -s test --half_res
```

For validation:

```bash
python metrics.py -m output/your_experiment -s val --half_res
```

The script writes:

```text
results_<split>_halfres.json
per_view_<split>_halfres.json
```

## Segmentation Utility

Gaussian clustering can be run after training:

```bash
python seg.py -m output/your_experiment --K 3 --vis
```

This is useful for inspecting motion groups or object-level dynamic components.

## Current Implementation Notes

- FastGS is enabled by default in the optimization parameters.
- The training script constructs `DeformModel` with affine velocity enabled.
- MPM loss is active when `warp-lang` is available and MPM is requested.
- PINNs modules exist in `utils/`, but the PINNs loss block in `train_gui.py` is currently commented out.
- `output/` contains generated experiments and should usually be excluded from source control in a clean release.

## Citation

This repository uses and modifies components from 3D Gaussian Splatting and its CUDA rasterization pipeline. If you use this code for research, also cite the original 3DGS work and any relevant upstream methods used by your experiment.

