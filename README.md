# 3D DVC Dataset Generator

Standalone dataset generation library for **3D Digital Volume Correlation (DVC)**. Generates pairs of volumetric images with ground-truth deformation fields (flow fields) for training optical flow / DVC models.

Extracted from [RAFT-DVC](https://github.com/your-repo/RAFT-DVC). No PyTorch dependency.

---

## Installation

**Requirements:** [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html), Python 3.9+

```bash
# 1. Create a conda environment (any name you like)
conda create -n 3D_DVC_Dataset_Generator python=3.11

# 2. Activate the environment
conda activate 3D_DVC_Dataset_Generator

# 3. Install the package and all core dependencies
pip install -e .

# 4. (Optional) Install visualization tools
pip install -e ".[viz]"
```

All dependencies are declared in [`pyproject.toml`](pyproject.toml). No other installation files are needed.

---

## Quick Start

Double-click **`run.bat`** (or run it from terminal) for an interactive menu:

```text
============================================================
  3D DVC Dataset Generator
============================================================

  1. Generate dataset
  2. Diagnose TIF files
  3. Custom config path
  4. Exit

Select [1-4]:
> 1

Available configs:
  1. exp_Franck_128
  2. synthetic_128_v2
  ...
Select config [1-N]:
```

Or run directly from the command line:

```bash
# After pip install -e .:
generate-dataset --config configs/synthetic_128_v2.yaml

# Or via python -m:
python -m dvc_generator.cli --config configs/exp_Franck_128.yaml
python -m dvc_generator.analysis.diagnostics --tif_dir data_tif/my_experiment
```

---

## Directory Structure

```text
.
├── run.bat                                     <- interactive menu launcher (Windows)
├── dvc_generator/                              <- main Python package
│   ├── cli.py                                  <- CLI entry point (generate-dataset)
│   ├── generators/                             <- orchestration layer
│   │   ├── base.py                             <- abstract BaseGenerator
│   │   ├── synthetic.py                        <- SyntheticGenerator (Pipeline A)
│   │   └── experimental.py                     <- ExperimentalGenerator (Pipeline B)
│   ├── simulation/                             <- physics & imaging simulation
│   │   ├── deformation.py                      <- flow field generation (4 types)
│   │   ├── warping.py                          <- backward / forward / bead warping
│   │   ├── beads.py                            <- particle generation & rendering
│   │   └── effects.py                          <- noise & PSF simulation
│   ├── io/                                     <- data loading & saving
│   │   ├── tif_loader.py                       <- TIF / NumPy loader
│   │   ├── preprocessor.py                     <- intensity normalization & denoising
│   │   ├── extractor.py                        <- random sub-volume extraction
│   │   └── writer.py                           <- unified output writer
│   └── analysis/                               <- quality control & diagnostics
│       ├── quality_control.py                  <- sample validation & visualization
│       └── diagnostics.py                      <- TIF file diagnostic tool
└── configs/
    ├── synthetic_128_v1.yaml
    ├── synthetic_128_v2.yaml                   <- recommended for synthetic
    ├── exp_Franck_32.yaml
    ├── exp_Franck_64.yaml
    ├── exp_Franck_64_small_disp.yaml
    └── exp_Franck_128.yaml
```

---

## Two Pipelines

Both pipelines are accessed through a **single entry point**. The `source_type` field in the YAML config determines which pipeline runs.

### Pipeline A -- Synthetic Generation (`source_type: synthetic`)

Programmatically generate particle image volume pairs from scratch. Fully controlled and reproducible.

```text
YAML config
  -> generate random particles (position, radius, intensity)
  -> generate deformation field (affine / bspline / localized / combined)
  -> forward warp particle coordinates
  -> render vol0 (undeformed) and vol1 (deformed)
  -> apply imaging effects (Poisson noise + Gaussian blur)
  -> quality check -> save .npy files
```

**Run:**

```bash
python generate_dataset.py --config configs/synthetic_128_v2.yaml
```

### Pipeline B -- Crop from Experimental Images (`source_type: crop_from_image`)

Randomly crop sub-volumes from real microscopy images (TIF or NumPy), apply a synthetic deformation field, and build a semi-synthetic dataset.

```text
Image directory (TIF / .npy files)
  -> load and filter valid files (size checks)
  -> randomly crop sub-volumes (with quality filtering)
  -> intensity normalization / denoising
  -> apply synthetic deformation field
  -> backward_swap or forward warp to generate image pair
  -> save .npy + metadata
```

**Run:**

```bash
python generate_dataset.py --config configs/exp_Franck_128.yaml
```

---

## Unified Config Schema

Both pipelines share common top-level fields:

```yaml
source_type: synthetic          # "synthetic" | "crop_from_image"

dataset:
  name: my_dataset
  output_dir: data/my_dataset
  volume_shape: [128, 128, 128]

generation:
  seed: 42
  parallel: false
  num_workers: 4
  save_metadata: true

splits:
  train: 800
  val: 100
  test: 100

deformations:                   # identical schema for both pipelines
  type_probabilities: { affine: 0.5, bspline: 0.5, localized: 0.0, combined: 0.0 }
  affine: { ... }
  bspline: { ... }

quality:
  max_displacement_ratio: 0.5

# --- synthetic only ---
particles: { ... }
imaging: { ... }

# --- crop_from_image only ---
input: { format: tif, raw_data_dir: ..., file_pattern: "*.tif" }
extraction: { volume_size: [128,128,128], strategy: random, total_volumes: 50 }
preprocessing: { normalize: true, clip_percentile: [1, 99] }
warp_mode: backward_swap
```

---

## Output Format

Each dataset is split into `train/`, `val/`, and `test/`. Per-split structure:

```text
<dataset_name>/
├── train/
│   ├── vol0/           sample_0000.npy   # (D,H,W) float32 -- reference frame
│   ├── vol1/           sample_0000.npy   # (D,H,W) float32 -- deformed frame
│   ├── flow/           sample_0000.npy   # (3,D,H,W) float32 -- GT flow [dz,dy,dx]
│   ├── metadata/       sample_0000.json  # generation parameters & statistics
│   └── visualizations/ sample_0000.png   # optional slice visualization
├── val/
├── test/
├── generation_config.yaml                # full config snapshot (for reproduction)
└── generation_summary.yaml               # summary (file count, timing, etc.)
```

**Flow field convention:** `flow[c, z, y, x]` is the displacement from vol0 to vol1 in voxels. `c=0` -> z, `c=1` -> y, `c=2` -> x.

---

## Warp Strategies (Pipeline B)

| Method | Config value | Implementation | Accuracy | Speed |
| --- | --- | --- | --- | --- |
| Backward + Swap | `backward_swap` | `scipy.ndimage.map_coordinates` | Exact (no holes) | Fast |
| Forward | `forward` | Trilinear splatting (pure numpy) | Approximate (may have holes) | Slow |

**backward_swap** (default, recommended): generates flow as backward flow, warps, swaps vol0/vol1 roles, saves negated flow as forward direction.

---

## Deformation Types

| Type | Description | Typical Use |
| --- | --- | --- |
| `affine` | Global affine (rotation, translation, shear) | Rigid-body global motion |
| `bspline` | B-spline smooth random field | Continuous soft-tissue deformation |
| `localized` | Locally concentrated deformation at random centers | Stress concentration |
| `combined` | Random weighted mix of affine + bspline | Maximum diversity |

---

## Configuration Reference

### Pipeline A Configs (synthetic)

| Config | Volume | Particle Radius | Count Range | Deformations | Notes |
| --- | --- | --- | --- | --- | --- |
| `synthetic_128_v1` | 128^3 | 2.5-3.0 vx | 400-1500 | affine, bspline | Dense, large particles |
| `synthetic_128_v2` | 128^3 | 0.8-3.0 vx | 200-2000 | all 4 types | Recommended |

### Pipeline B Configs (Franck Lab data)

| Config | Volume | Notes |
| --- | --- | --- |
| `exp_Franck_32` | 32^3 | Small, fast experiments |
| `exp_Franck_64` | 64^3 | Medium size |
| `exp_Franck_64_small_disp` | 64^3 | Small displacements (matching 32^3 training) |
| `exp_Franck_128` | 126x128x128 | Full size, production |

---

## Dependencies

All dependencies are managed via [`pyproject.toml`](pyproject.toml):

| Package | Required | Purpose |
| --- | --- | --- |
| `numpy`, `scipy` | Core | Array ops, deformation, warping |
| `pyyaml`, `tqdm` | Core | Config loading, progress bars |
| `tifffile` | Core | Pipeline B TIF input |
| `matplotlib` | Optional (`[viz]`) | Slice visualization |
| `pyvista` | Optional (`[viz]`) | 3D volume rendering |

No PyTorch. The original RAFT-DVC `grid_sample` backward warp has been replaced with `scipy.ndimage.map_coordinates`.

---

## Development

See [DEVELOPMENT_PLAN_EN.md](DEVELOPMENT_PLAN_EN.md) for the full 7-module development roadmap, task checklists, testing requirements, and contribution guidelines.

For the Chinese version of this README, see [README_CN.md](README_CN.md).
