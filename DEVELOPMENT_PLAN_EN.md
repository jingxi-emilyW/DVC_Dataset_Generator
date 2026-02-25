# 3D DVC Dataset Generator — Development Plan & Progress Tracker

## Document Info

| Field | Content |
| ------- | --------- |
| Project | 3D DVC Dataset Generator |
| Maintainer | Zach Tong |
| Created | 2026-02-23 |
| Last Updated | 2026-02-25 |
| Version | v1.0 |

---

## Changelog

| Date | Author | Change |
| ------ | -------- | -------- |
| 2026-02-23 | Zach Tong | Initial document: 7-module development plan + GUI roadmap |
| 2026-02-25 | Zach Tong | Module 1 complete: restructured as `dvc_generator/` package (generators/simulation/io/analysis); unified CLI via `dvc_generator/cli.py`; dynamic config listing in `run.bat` |

---

## Project Overview

This project is a standalone dataset generation library extracted from RAFT-DVC (3D Digital Volume Correlation). It generates pairs of 3D volumetric images with ground-truth deformation fields (flow fields) for training optical flow / DVC models.

**Core constraints:**

- No PyTorch. Dependencies: numpy, scipy, pyyaml, tqdm, tifffile only
- Two pipelines: synthetic particle generation (Pipeline A) and experimental image cropping (Pipeline B)
- All parameters configured via YAML — zero hardcoding
- Output format fully compatible with RAFT-DVC training code

**Flow field convention:** `flow.shape == (3, D, H, W)`, `flow[0]=dz, flow[1]=dy, flow[2]=dx`, units in voxels, direction from vol0 to vol1.

---

## Overall Progress

| Module | Description | Assignee | Status | Start | Done | PR |
| -------- | ------------- | ---------- | -------- | ------- | ------ | ---- |
| Module 1 | Architecture Integration (unified CLI) | Zach Tong | 🟢 Done | 2026-02-23 | 2026-02-25 | — |
| Module 2 | Particle & Imaging Diversity | TBD | ⏸ Pending | — | — | — |
| Module 3 | Deformation Field Diversity | TBD | ⏸ Pending | — | — | — |
| Module 4 | Data Augmentation Library | TBD | ⏸ Pending | — | — | — |
| Module 5 | Uncertainty & Interpretability | TBD | ⏸ Pending | — | — | — |
| Module 6 | Performance & Engineering | TBD | ⏸ Pending | — | — | — |
| Module 7 | Quality Assessment & Tools | TBD | ⏸ Pending | — | — | — |
| GUI | Graphical Interface | TBD | 🔒 Unplanned | — | — | — |
| Custom | Mentee-proposed features (see note below) | TBD | — | — | — | — |

**Legend:** 🟢 Done &nbsp;|&nbsp; 🟡 In Progress &nbsp;|&nbsp; ⏸ Pending &nbsp;|&nbsp; 🔒 Unplanned &nbsp;|&nbsp; ❌ Dropped

> **On custom feature development:** Beyond the assigned modules, mentees are encouraged to propose their own features or new modules. Discuss the idea with Zach Tong before starting — once approved, open a new branch and submit a PR as usual. Custom modules follow the same coding standards and testing requirements. The PR description must additionally explain the **motivation and intended use case** for the new feature.

---

## Module Dependencies

```text
Module 1 (Architecture Integration)    ← prerequisite for all subsequent modules
├── Module 2 (Particle/Imaging)        ← can run in parallel with Module 3
├── Module 3 (Deformation)             ← can run in parallel with Module 2
├── Module 4 (Augmentation)            ← start after Module 1
├── Module 5 (Uncertainty)             ← start after Modules 2 & 3 are stable
├── Module 6 (Performance)             ← can be done at any stage, highly independent
└── Module 7 (Quality Assessment)      ← best done after other modules are stable

GUI development ← plan separately after all modules complete
```

---

## Development Workflow

### Branch & PR

- Each module corresponds to **one PR**, reviewed and merged by Zach Tong
- Branch naming is up to the developer — use something descriptive (e.g., `feature/module2-ellipsoidal-beads`)
- **Before submitting a PR**, you must complete all testing requirements for that module yourself

### PR Description Template

```markdown
## Module N: [Module Name]

### What was implemented
- Which sub-tasks (T*.* IDs) were completed
- Which files were added / modified

### Test results
- Commands run
- Screenshots or log output
- Key numerical results (displacement stats, SNR, timing, etc.)

### YAML config changes
- New config fields added (with examples)

### Known limitations / incomplete items
- If anything is deferred, document it here with rationale
```

### Commit Message Format

```text
[ModuleN] Short English description
```

Examples:

```bash
[Module2] Add ellipsoidal bead rendering with anisotropic sigma
[Module3] Add sinusoidal deformation type with frequency/amplitude config
[Module6] Vectorize forward warp with numpy scatter, 6x speedup on 128^3
```

---

## Coding Standards

| Rule | Requirement |
| ------ | ------------- |
| Dependencies | **No PyTorch.** Core: numpy, scipy, pyyaml, tqdm, tifffile. Extended (after T1.6): h5py, zarr |
| Configuration | All new feature parameters must be YAML-configurable. **No hardcoding** |
| Naming | All variables, functions, and class names in **English**, following Python snake_case |
| Docstrings | New classes and non-trivial functions must have English docstrings (args + return value) |
| Type hints | Recommended for all new function signatures |
| Flow convention | `flow.shape == (3, D, H, W)`, `flow[0]=dz, flow[1]=dy, flow[2]=dx`, voxels |
| Output format | Per sample: `vol0.npy (D,H,W) float32`, `vol1.npy (D,H,W) float32`, `flow.npy (3,D,H,W) float32` |

---

## Testing Standards

- No automated test framework (no pytest for now). Testing is **manual: run commands + visual/numerical inspection**
- Each module's specific requirements are listed in its "Testing Requirements" section
- Fix all failures in your branch before opening a PR. **Do not submit PRs with known bugs**
- Known limitations that are intentionally deferred are acceptable — document them clearly in the PR

---

## Module Detail Plans

---

## Module 1: Architecture Integration

**Assignee:** Zach Tong &nbsp;|&nbsp; **Status:** 🟢 Done &nbsp;|&nbsp; **Start:** 2026-02-23 &nbsp;|&nbsp; **Done:** 2026-02-25

### Goal

Unify Pipeline A and Pipeline B under a single CLI entry point. After completion, users only need to set `source_type` in their config file to run either pipeline with `python generate_dataset.py --config <file>`. All subsequent modules build on this foundation.

### Background

Currently the two pipelines have separate entry scripts and share modules in an ad-hoc way. This module will:

1. Design a unified top-level YAML config schema (add `source_type` field)
2. Move shared modules to `scripts/shared/`
3. Create a unified entry script `generate_dataset.py`
4. Extend Pipeline B input formats (HDF5, Zarr, NumPy)

### Actual Directory Structure (as implemented)

> The original plan used `scripts/` subdirectories. During implementation, the structure was upgraded to a proper Python package organized by functional domain — a cleaner and more professional layout.

```text
3D_DVC_Dataset_Generator/
├── dvc_generator/                        ← installable Python package (pip install -e .)
│   ├── cli.py                            ← unified CLI entry point
│   ├── generators/
│   │   ├── base.py                       ← BaseGenerator (shared logic)
│   │   ├── synthetic.py                  ← Pipeline A: SyntheticGenerator
│   │   └── experimental.py              ← Pipeline B: ExperimentalGenerator
│   ├── simulation/
│   │   ├── deformation.py               ← DeformationGenerator
│   │   ├── warping.py                   ← ForwardWarper, backward_warp
│   │   ├── beads.py                     ← BeadGenerator, BeadRenderer
│   │   └── effects.py                   ← ImagingSimulator (was imaging.py)
│   ├── io/
│   │   ├── writer.py                    ← OutputWriter (unified save logic)
│   │   ├── tif_loader.py               ← TifLoader
│   │   ├── preprocessor.py             ← Preprocessor
│   │   └── extractor.py                ← VolumeExtractor
│   └── analysis/
│       ├── quality_control.py          ← QualityChecker
│       └── diagnostics.py              ← TIF diagnostic tool
├── configs/                             ← all YAML configs (flat, no subdirs)
│   ├── synthetic_128_v1.yaml
│   ├── synthetic_128_v2.yaml
│   ├── exp_Franck_32.yaml
│   ├── exp_Franck_64.yaml
│   ├── exp_Franck_64_small_disp.yaml
│   └── exp_Franck_128.yaml
├── pyproject.toml                       ← package definition + CLI entry point
└── run.bat                              ← Windows launcher with dynamic config listing
```

### Unified YAML Schema (top-level fields)

```yaml
source_type: synthetic          # synthetic | crop_from_image

dataset:
  name: my_dataset
  output_dir: data/

generation:
  seed: 42
  parallel: false
  num_workers: 4

splits:
  train: 800
  val: 100
  test: 100

# Used when source_type: synthetic
particles: { ... }
imaging: { ... }
deformations: { ... }
quality: { ... }

# Used when source_type: crop_from_image
input:
  format: tif                   # tif | hdf5 | zarr | numpy
  raw_data_dir: data_tif/exp_name/
  hdf5_key: volume              # only needed for format: hdf5
extraction: { ... }
preprocessing: { ... }
deformations: { ... }           # same schema as synthetic
warp_mode: backward_swap
```

### Task Checklist

- [x] **T1.1** Shared modules organized under `dvc_generator/simulation/` and `dvc_generator/io/`; all imports updated
- [x] **T1.2** Unified save logic in `dvc_generator/io/writer.py` (`OutputWriter`), used by both pipelines
- [x] **T1.3** Pipeline A refactored as `dvc_generator/generators/synthetic.py` (`SyntheticGenerator`)
- [x] **T1.4** Pipeline B refactored as `dvc_generator/generators/experimental.py` (`ExperimentalGenerator`)
- [x] **T1.5** Unified CLI entry point at `dvc_generator/cli.py`; runs via `generate-dataset` or `python -m dvc_generator.cli`
- [x] **T1.6** *(Deferred)* TIF format supported; HDF5/Zarr/NumPy can be added in Module 6
- [x] **T1.7** All 6 config files updated to unified schema (flat `configs/` directory)
- [x] **T1.8** `README.md` updated with new structure, install instructions, and CLI commands

### Testing Requirements

After completion, all of the following must run successfully:

```bash
# Pipeline A (synthetic, small test run)
python generate_dataset.py --config configs/data_generation/synthetic_128_v2.yaml

# Pipeline B (experimental images, requires TIF data)
python generate_dataset.py --config configs/data_generation_from_experiments/exp_Franck_128.yaml
```

**Acceptance criteria:**

1. Both commands produce valid `vol0.npy (D,H,W)`, `vol1.npy (D,H,W)`, `flow.npy (3,D,H,W)`
2. `flow.npy` has no NaN/Inf and physically reasonable magnitudes
3. `generation_summary.yaml` and `generation_config.yaml` are generated
4. Both pipelines call `scripts/shared/` modules with no duplicated logic

### Definition of Done

- [x] Unified entry runs both pipelines
- [x] All 6 config files migrated to new schema
- [x] No duplicated deformation/output logic between pipelines
- [x] Pipeline B supports TIF input (NumPy/HDF5/Zarr deferred to Module 6)
- [x] `README.md` updated

---

## Module 2: Particle & Imaging Diversity

**Assignee:** TBD &nbsp;|&nbsp; **Prerequisite:** Module 1 &nbsp;|&nbsp; **Status:** ⏸ Pending

**Goal:** Extend `beads.py` and `imaging.py` to add more realistic particle morphologies, PSF types, and imaging noise models.

**Files involved:**

| File | Action |
| ------ | -------- |
| `dvc_generator/simulation/beads.py` | Primary: modify `BeadRenderer`, `BeadGenerator` |
| `dvc_generator/simulation/effects.py` | Primary: modify `ImagingSimulator` |
| `configs/*.yaml` | Add new config fields |

### Sub-tasks A: Particle Morphology

- [ ] **T2.A1** **Ellipsoidal beads:** Modify `BeadRenderer._render_single_bead()` to support anisotropic sigma `[sigma_z, sigma_y, sigma_x]`. Config: `particles.anisotropy: [sz, sy, sx]` (isotropic: `[1,1,1]`)
- [ ] **T2.A2** **Multi-class particles:** `BeadGenerator` supports `particle_groups` config — each group has independent `num_range`, `radius_range`, `intensity_range`; all groups are rendered into the same volume
- [ ] **T2.A3** **Depth-dependent intensity attenuation:** Apply a linear or exponential decay factor along the z-axis during rendering. Config: `particles.depth_attenuation: {enabled, decay_rate}`

### Sub-tasks B: PSF Models

- [ ] **T2.B1** **Anisotropic Gaussian PSF:** Modify `ImagingSimulator._apply_psf()` to support different sigma in z vs. xy directions
- [ ] **T2.B2** **PSF YAML parameterization:** Expand from a single `psf_sigma` to:

  ```yaml
  psf:
    type: gaussian_isotropic     # gaussian_isotropic | gaussian_anisotropic
    sigma: 1.0                   # used for isotropic
    sigma_z: 2.0                 # used for anisotropic
    sigma_xy: 1.0
  ```

### Sub-tasks C: Noise Models

- [ ] **T2.C1** **Readout noise:** Add an independent additive Gaussian component in `_add_noise()` (signal-independent, models CCD/sCMOS electronic noise). Config: `imaging.readout_noise_std`
- [ ] **T2.C2** **Background fluorescence enhancement:** Extend existing `_generate_background()` into a spatially correlated low-frequency random field. Config: `imaging.background.spatial_scale`
- [ ] **T2.C3** **Depth-dependent SNR decay:** Reduce local SNR for deeper (higher z) voxels. Config: `imaging.depth_snr_decay: {enabled, decay_rate}`

**Testing Requirements:**

| Test | Acceptance Criterion |
| ------ | --------------------- |
| Ellipsoidal beads (T2.A1) | Visually verify particle elongation in z vs. xy center slices matches `anisotropy` ratio |
| Multi-class particles (T2.A2) | Count particles per group, verify proportions match `num_range` configuration |
| Anisotropic PSF (T2.B1) | Measured particle width in z slice is ~`sigma_z/sigma_xy` times wider than in xy slice |
| Readout noise (T2.C1) | Std of intensity in particle-free regions ≈ configured `readout_noise_std` |
| All new params | With new features disabled, output is backward-compatible with old configs |

**PR Requirements:** Include at least 2 slice visualizations (with vs. without new feature). List all new YAML config fields with their default values.

---

## Module 3: Deformation Field Diversity

**Assignee:** TBD &nbsp;|&nbsp; **Prerequisite:** Module 1 &nbsp;|&nbsp; **Status:** ⏸ Pending

**Goal:** Extend `DeformationGenerator` with physically motivated deformation types to improve training data diversity.

**Files involved:**

| File | Action |
| ------ | -------- |
| `dvc_generator/simulation/deformation.py` | Primary |
| `configs/*.yaml` | Add new deformation type configs |

**Existing deformation types:**

| Type | Method | Description |
| ------ | -------- | ------------- |
| `affine` | `_generate_affine()` | Global affine (rotation + translation + shear) |
| `bspline` | `_generate_bspline()` | B-spline smooth random field |
| `localized` | `_generate_localized()` | Locally concentrated deformation |
| `combined` | `_generate_combined()` | Weighted affine + bspline mix |

**Sub-tasks:**

- [ ] **T3.1** **Anisotropic scaling:** Support independent max displacement per axis in both `_generate_affine()` and `_generate_bspline()`. Config: `deformations.anisotropic_scale: [sz, sy, sx]`
- [ ] **T3.2** **Sinusoidal deformation:** New `_generate_sinusoidal()` method. Params: frequency (cycles per volume), amplitude (voxels), propagation direction. Config: `sinusoidal: {frequency, amplitude, direction}`
- [ ] **T3.3** **Zero-boundary deformation:** Post-processing option — smoothly decay the flow field to zero at volume boundaries using a Hann window. Config: `deformations.zero_boundary: {enabled, boundary_width}`
- [ ] **T3.4** **Layered deformation:** New `_generate_layered()` method — divide volume into N z-layers, apply independent affine per layer with smooth transitions at boundaries
- [ ] **T3.5** **Discontinuous interface deformation:** New `_generate_discontinuous()` method — introduce a displacement jump at a specified z-plane (simulates cracks / slip planes). Jump magnitude is configurable
- [ ] **T3.6** **Update dispatch logic:** Extend `generate()` to support new type names and probability weights in YAML
- [ ] **T3.7** *(Optional)* **Divergence constraint:** Apply approximate incompressibility via Helmholtz decomposition (retain only the curl component of the flow field)

**Testing Requirements:**

| Test | Acceptance Criterion |
| ------ | --------------------- |
| Sinusoidal (T3.2) | 1D FFT of flow's z-component shows a peak at the configured `frequency` |
| Zero-boundary (T3.3) | Mean flow magnitude in the first `boundary_width` voxels < 0.1 voxel |
| Discontinuous (T3.5) | Flow shows a visible jump at the configured z-plane; jump magnitude ≈ configured value |
| All new types | Report `compute_flow_statistics()` output (mean/std/max magnitude) in PR |

---

## Module 4: Data Augmentation Library

**Assignee:** TBD &nbsp;|&nbsp; **Prerequisite:** Module 1 &nbsp;|&nbsp; **Status:** ⏸ Pending

**Goal:** Create `scripts/shared/augmentation/` — a composable augmentation pipeline supporting both **offline** (write to disk at generation time) and **online** (randomize at runtime) modes.

> **Key design constraint:** All spatial augmentations (flip, rotate, Cutout) must be applied **jointly** to `vol0`, `vol1`, and `flow`. For `flow`, direction components must be adjusted accordingly (e.g., flipping along z requires negating `flow[0]` — the dz component).

**New file structure:**

```text
dvc_generator/augmentation/
├── __init__.py
├── pipeline.py       ← AugmentationPipeline class
├── spatial.py        ← Cutout, Flip, Rotate
├── intensity.py      ← Scale, Jitter, Gamma
└── blur.py           ← AnisotropicBlur, Downsample
```

### Sub-tasks A: Spatial Masking

- [ ] **T4.A1** **`Cutout3D`:** Zero out 1–N random cuboid regions in vol0 and vol1; simultaneously write `mask.npy` (1 in masked regions, 0 elsewhere). Config: `num_cutouts: [min, max]`, `cutout_size: [dz, dy, dx]`
- [ ] **T4.A2** **`RandomFlip3D`:** Flip along each axis with independent probability; negate corresponding flow components
- [ ] **T4.A3** **`RandomRotate90`:** Rotate 0°/90°/180°/270° around z-axis; rotate flow's dy/dx components accordingly

### Sub-tasks B: Intensity

- [ ] **T4.B1** **`GlobalIntensityScale`:** Multiply vol0 and vol1 by a random scalar in `[a, b]`
- [ ] **T4.B2** **`LocalIntensityJitter`:** Multiply by a spatially smooth low-frequency random field (local contrast variation)
- [ ] **T4.B3** **`GammaCorrection`:** Apply random power-law gamma in `[gamma_min, gamma_max]`

### Sub-tasks C: Blur / Degradation

- [ ] **T4.C1** **`AnisotropicBlur`:** Gaussian blur with independent z/y/x sigma
- [ ] **T4.C2** **`Downsample`:** Downsample then upsample (configurable scale factor)

### Sub-tasks D: Pipeline Integration

- [ ] **T4.D1** **`AugmentationPipeline`:** Execute augmentation steps in YAML-defined order, each with an independent execution probability `p`
- [ ] **T4.D2** Unified interface: `pipeline.apply(vol0, vol1, flow)` → `(vol0_aug, vol1_aug, flow_aug, mask)`
- [ ] **T4.D3** Integrate augmentation toggle into `dvc_generator/cli.py` via config field `augmentation.enabled`

**YAML config example:**

```yaml
augmentation:
  enabled: true
  steps:
    - type: Cutout3D
      p: 0.5
      num_cutouts: [1, 3]
      cutout_size: [16, 16, 16]
    - type: RandomFlip3D
      p: 0.5
      axes: [true, true, false]   # allow z/y flip, disable x flip
    - type: GlobalIntensityScale
      p: 0.8
      scale_range: [0.7, 1.3]
    - type: AnisotropicBlur
      p: 0.3
      sigma_z: [0.0, 1.0]
      sigma_xy: [0.0, 0.5]
```

**Testing Requirements:**

| Test | Acceptance Criterion |
| ------ | --------------------- |
| `Cutout3D` | Masked regions in vol0/vol1 are zero; `mask.npy` is 1 at those positions |
| `RandomFlip3D(axis=z)` | `flow[0]` (dz) is negated; `flow[1]`, `flow[2]` unchanged |
| `AugmentationPipeline` (3 steps) | Output shape unchanged, no NaN/Inf |
| Flow consistency | After flip/rotate, `vol1[p + flow_aug(p)] ≈ vol0[p]` holds for 5 random test points |

---

## Module 5: Uncertainty & Interpretability Support

**Assignee:** TBD &nbsp;|&nbsp; **Prerequisite:** Modules 2 & 3 stable &nbsp;|&nbsp; **Status:** ⏸ Pending

**Goal:** Save auxiliary maps (`mask`, `confidence_map`, `reliability_map`) alongside each sample to support uncertainty-aware downstream models.

**Files involved:**

| File | Action |
| ------ | -------- |
| `dvc_generator/simulation/beads.py` | Add particle region mask generation |
| `dvc_generator/io/writer.py` | Extend to save optional auxiliary maps |
| `README.md` | Update output file list |

**Sub-tasks:**

- [ ] **T5.1** **`mask.npy` (Pipeline A):** In `BeadRenderer.render()`, additionally output a boolean volume marking voxels covered by particles (above an intensity threshold). Config: `output.mask: {enabled, threshold}`
- [ ] **T5.2** **`confidence_map.npy`:** Confidence field based on local particle density — sliding-window particle count, normalized to [0, 1]. Config: `output.confidence_map: {enabled, window_size}`
- [ ] **T5.3** **`reliability_map.npy` (Pipeline B):** Texture richness based on local intensity variance — sliding-window std map, normalized to [0, 1]. Config: `output.reliability_map: {enabled, window_size}`
- [ ] **T5.4** Update `output_writer.py` save logic to optionally write the above maps (controlled by config switches)
- [ ] **T5.5** Update `README.md` "Output Format" section to list all optional auxiliary files

**Testing Requirements:**

| Test | Acceptance Criterion |
| ------ | --------------------- |
| `mask.npy` (T5.1) | Pearson correlation between `mask` and thresholded `vol0` > 0.7 |
| `confidence_map.npy` (T5.2) | Visually: dense particle regions have high confidence, empty borders have low confidence |
| `reliability_map.npy` (T5.3) | Sample 10 points from textured regions and 10 from uniform regions; textured mean significantly higher |
| Disabled switch | With `enabled: false`, no auxiliary files are written and no errors are raised |

---

## Module 6: Performance & Engineering

**Assignee:** TBD &nbsp;|&nbsp; **Prerequisite:** Module 1 (highly independent) &nbsp;|&nbsp; **Status:** ⏸ Pending

**Goal:** Improve generation throughput and flexibility: accelerate forward warp, parallelize Pipeline B, support non-cubic volumes, and enable incremental dataset appending.

**Sub-tasks:**

- [ ] **T6.1** **Forward warp acceleration:** Replace the per-voxel Python loop in `dvc_generator/simulation/warping.py` with a vectorized numpy scatter (`np.add.at`). Provide a benchmark script comparing 128³ timing before and after
- [ ] **T6.2** **Pipeline B parallelization:** Add `multiprocessing.Pool` to `dvc_generator/generators/experimental.py` (mirror Pipeline A's `parallel`/`num_workers` config)
- [ ] **T6.3** **Non-cubic volume support:** Audit all modules for implicit `D==H==W` assumptions (especially coordinate grid generation in `dvc_generator/simulation/deformation.py` and voxel indexing in `dvc_generator/simulation/warping.py`); fix to support arbitrary `(D, H, W)`
- [ ] **T6.4** **Incremental generation:** Add `--append` flag to `dvc_generator/cli.py` — detect existing sample count from `generation_summary.yaml` and continue numbering from N+1

**Testing Requirements:**

| Test | Acceptance Criterion |
| ------ | --------------------- |
| Forward warp speedup (T6.1) | Benchmark shows ≥5× speedup on 128³ volume (report exact numbers) |
| Pipeline B parallel (T6.2) | Fixed seed: parallel vs. sequential `flow.npy` outputs are identical |
| Non-cubic volumes (T6.3) | `(64, 96, 128)` volume runs through both pipelines without error; output shapes correct |
| Incremental (T6.4) | Appending 10 samples to a 50-sample dataset yields 60 samples with continuous numbering |

---

## Module 7: Quality Assessment & Documentation Tools

**Assignee:** TBD &nbsp;|&nbsp; **Prerequisite:** Other modules stable &nbsp;|&nbsp; **Status:** ⏸ Pending

**Goal:** Provide an end-to-end dataset analysis tool, a startup config validator, and a warp accuracy benchmark suite.

**New files:**

```text
dvc_generator/tools/
├── __init__.py
├── analyze_dataset.py     ← dataset statistics and report generation
└── benchmark_warp.py      ← warp accuracy benchmark
```

**Sub-tasks:**

- [ ] **T7.1** **Dataset analysis tool** `dvc_generator/tools/analyze_dataset.py`:
  - Iterate over a dataset directory, compute EPE distribution, particle density distribution, SNR distribution
  - Output a Markdown report with summary tables and histogram paths
  - Command: `python -m dvc_generator.tools.analyze_dataset --dataset_dir data/my_dataset/ --output report.md`

- [ ] **T7.2** **Config validator:** Run automatically at `dvc_generator/cli.py` startup. Checks:
  - `max_displacement < volume_size / 2`
  - `particles.num_range[1] > particles.num_range[0]`
  - `splits.train + splits.val + splits.test > 0`
  - PSF sigma values reasonable (`< volume_size / 4`)
  - On failure: print a clear error message and exit. No silent failures

- [ ] **T7.3** **Warp accuracy benchmark** `dvc_generator/tools/benchmark_warp.py`:
  - Generate a known sinusoidal flow field → forward warp → backward warp → compute round-trip EPE
  - Test both backward and forward warp methods
  - Acceptance: backward warp round-trip EPE < 0.05 voxel on 128³
  - Output a Markdown table with mean/max EPE per method

**Testing Requirements:**

| Test | Acceptance Criterion |
| ------ | --------------------- |
| Analysis tool (T7.1) | Runs on 10+ sample dataset, produces well-formatted `report.md`, no crashes |
| Config validator (T7.2) | Intentionally invalid config (e.g., `max_displacement=200` for 128³) triggers clear error and exits |
| Warp benchmark (T7.3) | Backward warp round-trip EPE < 0.05 voxel; results shown in Markdown table |

---

## Future: GUI Development

**Status:** 🔒 Unplanned

A separate GUI development plan will be created after Modules 1–7 are complete and the CLI workflow is fully stable.

**Anticipated goals** (for reference only):

- Visual parameter configuration (avoids manual YAML editing)
- Real-time generation progress monitoring
- Built-in slice viewer (live preview of generated volumes)
- Lower barrier to entry for non-programming users

---

*Maintenance: When updating module status, completing tasks, or recording new decisions, add a row to the Changelog table at the top with the date and your name.*
