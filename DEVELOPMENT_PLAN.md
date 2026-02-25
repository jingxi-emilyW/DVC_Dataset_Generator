# 3D DVC Dataset Generator — 开发计划与进度文档

## 文档信息

| 字段 | 内容 |
|------|------|
| 项目名称 | 3D DVC Dataset Generator |
| 项目负责人 | Zach Tong |
| 创建日期 | 2026-02-23 |
| 最近更新 | 2026-02-23 |
| 文档版本 | v1.0 |

---

## 修改记录

| 日期 | 执行人 | 修改内容 |
|------|--------|----------|
| 2026-02-23 | Zach Tong | 初始文档创建，制定 7 个模块开发计划及 GUI 未来路线图 |

---

## 项目简介

本项目是从 RAFT-DVC（3D Digital Volume Correlation）项目中独立出来的数据集生成工具库，用于生成带有已知真值变形场（ground-truth flow field）的三维体积图像对，供光流/DVC 模型训练使用。

**核心特性**：
- 无 PyTorch 依赖，仅依赖 numpy / scipy / pyyaml / tqdm / tifffile
- 两条 Pipeline：合成粒子生成（Pipeline A）+ 从实验图像裁剪生成（Pipeline B）
- 所有参数通过 YAML 配置，完全可重现
- 输出格式与 RAFT-DVC 训练代码完全兼容

**流场约定**：`flow.shape == (3, D, H, W)`，`flow[0]=dz, flow[1]=dy, flow[2]=dx`，单位 voxels，方向为从 vol0 到 vol1。

---

## 总体进度

| 模块 | 内容 | 负责人 | 状态 | 开始日期 | 完成日期 | PR |
|------|------|--------|------|----------|----------|----|
| 模块一 | 架构整合（统一 CLI 接口） | Zach Tong | 🟡 进行中 | 2026-02-23 | — | — |
| 模块二 | 粒子与图像生成多样性 | TBD | ⏸ 待开始 | — | — | — |
| 模块三 | 变形场多样性 | TBD | ⏸ 待开始 | — | — | — |
| 模块四 | 数据增强库 | TBD | ⏸ 待开始 | — | — | — |
| 模块五 | 不确定性与可解释性支持 | TBD | ⏸ 待开始 | — | — | — |
| 模块六 | 性能与工程优化 | TBD | ⏸ 待开始 | — | — | — |
| 模块七 | 质量评估与文档工具 | TBD | ⏸ 待开始 | — | — | — |
| GUI | 图形界面开发 | TBD | 🔒 未规划 | — | — | — |
| 自定义 | Mentee 自提功能（见下方说明） | TBD | — | — | — | — |

**状态说明**：🟢 已完成 &nbsp;|&nbsp; 🟡 进行中 &nbsp;|&nbsp; ⏸ 待开始 &nbsp;|&nbsp; 🔒 未规划 &nbsp;|&nbsp; ❌ 已放弃

> **关于自定义功能开发**：Mentee 在完成指定模块的同时，也欢迎自行提出新功能或新模块的想法。提议前请先在 PR 之外与 Zach Tong 沟通确认方向，经批准后可在新分支上独立开发并提交 PR。自定义模块需遵守相同的代码规范和测试要求，PR 描述中需额外说明功能的**动机和使用场景**。

---

## 模块依赖关系

```
模块一（架构整合）                ← 所有后续模块的前置依赖
├── 模块二（粒子/成像多样性）     ← 可与模块三并行
├── 模块三（变形场多样性）        ← 可与模块二并行
├── 模块四（数据增强库）          ← 建议在模块一完成后进行
├── 模块五（不确定性支持）        ← 建议在模块二、三基本稳定后进行
├── 模块六（性能优化）            ← 可在任意阶段进行，独立性强
└── 模块七（质量评估）            ← 建议在其他模块完成后进行

GUI 开发 ← 待所有模块完成、流程稳定后单独规划
```

---

## 开发流程规范

### 分支与 PR

- 每个模块对应**一个 PR**，由 Zach Tong 审核后合并
- 分支命名由开发者自行决定，但应能清晰表达内容（例：`feature/module2-ellipsoidal-beads`）
- PR 提交前，开发者**必须本人完成本模块所有测试要求**并确认通过

### PR 描述模板

提交 PR 时，描述中需包含以下内容：

```
## 模块 N：[模块名称]

### 实现内容
- 实现了哪些子任务（T*.* 编号）
- 新增/修改了哪些文件

### 测试结果
- 运行的测试命令
- 输出截图或日志片段
- 关键数值结果（如位移统计、SNR、运行时间）

### YAML 配置变更
- 新增了哪些配置字段（附示例）

### 已知局限 / 未完成项
- 如有遗留问题，在此说明
```

### 提交信息格式

```
[ModuleN] 简短英文描述
```

示例：
```
[Module2] Add ellipsoidal bead rendering with anisotropic sigma
[Module3] Add sinusoidal deformation type with frequency/amplitude config
```

---

## 代码规范

| 规范项 | 要求 |
|--------|------|
| 依赖限制 | **无 PyTorch**。核心依赖：numpy、scipy、pyyaml、tqdm、tifffile。扩展依赖（T1.6 后）：h5py、zarr |
| 参数配置 | 所有新功能参数必须在 YAML 中可配置，**禁止硬编码** |
| 变量/函数命名 | 全部使用**英文**，遵循 Python snake_case |
| 代码注释 | 新增的类和关键函数必须有英文 docstring，注明参数类型和含义 |
| 类型注解 | 新增函数建议在签名层面加 type hints |
| 流场约定 | `flow.shape == (3, D, H, W)`，`flow[0]=dz, flow[1]=dy, flow[2]=dx`，单位 voxels |
| 输出格式 | 每个样本：`vol0.npy (D,H,W) float32`、`vol1.npy (D,H,W) float32`、`flow.npy (3,D,H,W) float32` |

---

## 测试规范

- 本项目暂不引入自动化测试框架（如 pytest），测试以**手动运行命令 + 视觉/数值检查**为主
- 每个模块的测试要求见各模块"测试要求"章节
- 测试失败须在当前分支修复后再提交 PR，**不得在 PR 中保留已知 bug**
- 如遇问题，在 PR 中记录到"已知局限"，并说明原因和后续计划

---

---

# 模块详细计划

---

## 模块一：架构整合

**负责人**：Zach Tong &nbsp;|&nbsp; **状态**：🟡 进行中 &nbsp;|&nbsp; **开始日期**：2026-02-23

### 目标

将 Pipeline A 和 Pipeline B 整合为统一的命令行接口。完成后，用户只需指定配置文件中的 `source_type`，即可通过同一入口脚本运行两条 Pipeline，后续所有模块在此基础上进行功能扩展。

### 背景说明

当前两条 Pipeline 各有独立入口（`generate_confocal_dataset.py` 和 `generate_from_tif.py`），变形模块（`deformation.py`）被 Pipeline B 通过相对路径直接导入，共享方式不规范。本模块目标：

1. 设计统一的顶层 YAML config schema（新增 `source_type` 字段）
2. 整理共享模块到 `scripts/shared/` 公共目录
3. 创建统一入口脚本 `generate_dataset.py`
4. 扩展 Pipeline B 的输入格式（HDF5、Zarr、NumPy）

### 目标目录结构

```
3D_DVC_Dataset_Generator/
├── generate_dataset.py                          ← 统一入口（新建）
├── scripts/
│   ├── shared/                                  ← 共享模块（新建）
│   │   ├── __init__.py
│   │   ├── deformation.py                       ← 迁移自 data_generation/modules/
│   │   ├── quality_control.py                   ← 迁移自 data_generation/modules/
│   │   └── output_writer.py                     ← 统一输出逻辑（新建）
│   ├── data_generation/                         ← Pipeline A（source_type: synthetic）
│   │   ├── synthetic_generator.py               ← 重构后的 Pipeline A 主类
│   │   └── modules/
│   │       ├── beads.py
│   │       ├── imaging.py
│   │       └── warping.py
│   └── data_generation_from_experiments/        ← Pipeline B（source_type: crop_from_image）
│       ├── experimental_generator.py            ← 重构后的 Pipeline B 主类
│       └── modules/
│           ├── tif_loader.py                    ← 扩展支持 HDF5/Zarr/NumPy
│           ├── preprocessor.py
│           └── volume_extractor.py
└── configs/
    ├── data_generation/
    │   ├── synthetic_128_v1.yaml                ← 更新为新 schema
    │   └── synthetic_128_v2.yaml                ← 更新为新 schema
    └── data_generation_from_experiments/
        ├── exp_Franck_32.yaml                   ← 更新为新 schema
        ├── exp_Franck_64.yaml
        ├── exp_Franck_64_small_disp.yaml
        └── exp_Franck_128.yaml
```

### 统一 YAML Schema 设计（顶层字段）

```yaml
# 所有配置文件共用的顶层结构
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

# source_type: synthetic 时使用
particles: { ... }
imaging: { ... }
deformations: { ... }
quality: { ... }

# source_type: crop_from_image 时使用
input:
  format: tif                   # tif | hdf5 | zarr | numpy
  raw_data_dir: data_tif/exp_name/
  hdf5_key: volume              # 仅 format: hdf5 时需要
extraction: { ... }
preprocessing: { ... }
deformations: { ... }           # 与 synthetic 共用同一 schema
warp_mode: backward_swap
```

### 任务清单

- [ ] **T1.1** 新建 `scripts/shared/` 目录，将 `deformation.py`、`quality_control.py` 迁移至此，更新所有 import 路径
- [ ] **T1.2** 新建 `scripts/shared/output_writer.py`，统一样本保存逻辑（`vol0/vol1/flow/metadata`），供两条 Pipeline 调用
- [ ] **T1.3** 重构 Pipeline A 为 `scripts/data_generation/synthetic_generator.py`（`SyntheticGenerator` 类），使用 `output_writer`
- [ ] **T1.4** 重构 Pipeline B 为 `scripts/data_generation_from_experiments/experimental_generator.py`（`ExperimentalGenerator` 类），使用 `output_writer`
- [ ] **T1.5** 新建顶层 `generate_dataset.py`，按 `source_type` 路由到对应 Generator
- [ ] **T1.6** 扩展 `tif_loader.py`，支持 HDF5（`h5py`）、Zarr、NumPy（`.npy`）输入（通过 `input.format` 字段切换）
- [ ] **T1.7** 更新全部 6 个配置文件（2 个 Pipeline A + 4 个 Pipeline B）符合新 schema
- [ ] **T1.8** 更新 `README.md`，反映新目录结构、新入口命令和新配置格式

### 测试要求

完成后，以下命令必须全部成功运行：

```bash
# Pipeline A（合成数据，小规模测试）
python generate_dataset.py --config configs/data_generation/synthetic_128_v2.yaml

# Pipeline B（实验图像，需有 TIF 数据文件）
python generate_dataset.py --config configs/data_generation_from_experiments/exp_Franck_128.yaml
```

**验收标准**：
1. 每条命令均输出格式正确的 `vol0.npy (D,H,W)`、`vol1.npy (D,H,W)`、`flow.npy (3,D,H,W)` 文件
2. `flow.npy` 无 NaN/Inf，值域物理合理
3. 生成 `generation_summary.yaml` 和 `generation_config.yaml`
4. 两条 Pipeline 均正常调用 `scripts/shared/` 中的共享模块（无重复代码）

### 完成标准（Definition of Done）

- [ ] 统一入口可运行两条 Pipeline
- [ ] 全部 6 个配置文件更新为新 schema
- [ ] `scripts/shared/` 中无重复的变形/输出逻辑
- [ ] Pipeline B 支持至少 TIF + NumPy 两种输入格式（HDF5/Zarr 可后续补充）
- [ ] `README.md` 更新完成

---

## 模块二：粒子与图像生成多样性

**负责人**：TBD &nbsp;|&nbsp; **前置条件**：模块一完成 &nbsp;|&nbsp; **状态**：⏸ 待开始

### 目标

扩展 `beads.py` 和 `imaging.py`，增加更真实的粒子形态、PSF 类型和成像噪声模型，提升合成数据的物理真实性和多样性。

### 涉及文件

| 文件 | 操作 |
|------|------|
| `scripts/data_generation/modules/beads.py` | 主要修改：`BeadRenderer`、`BeadGenerator` |
| `scripts/data_generation/modules/imaging.py` | 主要修改：`ImagingSimulator` |
| `configs/data_generation/*.yaml` | 新增配置字段 |

### 子任务 A：粒子形态扩展

- [ ] **T2.A1** **椭球形粒子**：修改 `BeadRenderer._render_single_bead()`，支持各向异性 sigma `[sigma_z, sigma_y, sigma_x]`。配置字段：`particles.anisotropy: [sz, sy, sx]`（各向同性时 `[1,1,1]`）
- [ ] **T2.A2** **多类粒子共存**：`BeadGenerator` 支持 `particle_groups` 配置，每组独立指定 `num_range`、`radius_range`、`intensity_range`，各组粒子混合渲染
- [ ] **T2.A3** **深度相关强度衰减**：渲染时按 z 坐标对粒子强度施加衰减（线性或指数），配置字段：`particles.depth_attenuation: {enabled, decay_rate}`

### 子任务 B：PSF 模型扩展

- [ ] **T2.B1** **各向异性 PSF**：修改 `ImagingSimulator._apply_psf()`，支持 z/xy 方向不同 sigma 的 Gaussian PSF
- [ ] **T2.B2** **PSF YAML 参数化**：配置字段从单一 `psf_sigma` 扩展为：
  ```yaml
  psf:
    type: gaussian_isotropic     # gaussian_isotropic | gaussian_anisotropic
    sigma: 1.0                   # 各向同性时使用
    sigma_z: 2.0                 # 各向异性时使用
    sigma_xy: 1.0
  ```

### 子任务 C：噪声模型扩展

- [ ] **T2.C1** **读出噪声**：在 `ImagingSimulator._add_noise()` 中新增独立的加性高斯分量（不依赖信号强度），配置字段：`imaging.readout_noise_std`
- [ ] **T2.C2** **背景荧光增强**：`_generate_background()` 已有基础实现，扩展为空间相关随机场（低频平滑场），配置字段：`imaging.background.spatial_scale`
- [ ] **T2.C3** **深度相关 SNR 衰减**：深层体素（大 z 坐标）自动降低 SNR，配置字段：`imaging.depth_snr_decay: {enabled, decay_rate}`

### 测试要求

| 测试项 | 验收标准 |
|--------|---------|
| 椭球粒子（T2.A1）| 在 z/y/x 三个中心切片上目视验证粒子拉伸方向符合 `anisotropy` 配置 |
| 多类粒子（T2.A2）| 统计各组粒子数量，与配置的 `num_range` 比例基本一致 |
| 各向异性 PSF（T2.B1）| z 方向切片的粒子宽度明显大于 xy 切片（比例接近 `sigma_z/sigma_xy`）|
| 读出噪声（T2.C1）| 空白区域（无粒子区）的强度标准差接近 `readout_noise_std` 配置值 |
| 所有新参数 | YAML 中可配置，关闭新特性时输出与原版本兼容 |

### PR 说明要求

- 提供至少 **2 张**切片可视化（对比有/无新特性的效果）
- 列出新增的所有 YAML 配置字段及其默认值

---

## 模块三：变形场多样性

**负责人**：TBD &nbsp;|&nbsp; **前置条件**：模块一完成 &nbsp;|&nbsp; **状态**：⏸ 待开始

### 目标

扩展 `DeformationGenerator`，新增多种物理上更真实的变形类型，提升训练数据的变形多样性。

### 涉及文件

| 文件 | 操作 |
|------|------|
| `scripts/shared/deformation.py` | 主要修改（模块一完成后路径更新） |
| `configs/*.yaml` | 新增 deformation 类型配置 |

### 当前已有变形类型

| 类型 | 方法 | 说明 |
|------|------|------|
| `affine` | `_generate_affine()` | 全局仿射（旋转 + 平移 + 剪切） |
| `bspline` | `_generate_bspline()` | B-spline 平滑随机场 |
| `localized` | `_generate_localized()` | 局部区域集中变形 |
| `combined` | `_generate_combined()` | affine + bspline 加权混合 |

### 子任务

- [ ] **T3.1** **各向异性变形**：在 `_generate_affine()` 和 `_generate_bspline()` 中支持 x/y/z 方向最大位移独立配置（`deformations.anisotropic_scale: [sz, sy, sx]`）
- [ ] **T3.2** **周期性/正弦变形**：新增 `_generate_sinusoidal()` 方法，参数：频率（cycles per volume）、幅度（voxels）、传播方向（x/y/z）。YAML 配置字段：`sinusoidal: {frequency, amplitude, direction}`
- [ ] **T3.3** **约束边界变形**：作为现有方法的后处理选项，在体积边界处用平滑窗函数（如 Hann 窗）将流场衰减至零。配置字段：`deformations.zero_boundary: {enabled, boundary_width}`
- [ ] **T3.4** **分层变形**：新增 `_generate_layered()` 方法，按 z 坐标将体积分为 N 层，每层施加独立的 affine 变形并在层间平滑过渡
- [ ] **T3.5** **不连续界面变形**：新增 `_generate_discontinuous()` 方法，在指定 z 平面处引入位移跳变（模拟材料裂缝/滑移面），跳变幅度可配置
- [ ] **T3.6** **更新类型分发逻辑**：在 `generate()` 方法中支持新类型的名称和概率权重配置
- [ ] **T3.7** （可选）**散度约束**：对生成流场施加近似不可压缩约束（Helmholtz 分解，仅保留旋度分量）

### 测试要求

| 测试项 | 验收标准 |
|--------|---------|
| 正弦变形（T3.2）| 对流场 z 分量做 1D FFT，主频率峰值与配置的 `frequency` 一致 |
| 约束边界（T3.3）| 边界处（前 `boundary_width` 个 voxel）流场幅度均值 < 0.1 voxel |
| 不连续界面（T3.5）| 在界面 z 处流场出现明显跳变，跳变幅度接近配置值 |
| 所有新类型 | 输出 `compute_flow_statistics()` 的统计报告（mean/std/max magnitude） |

---

## 模块四：数据增强库

**负责人**：TBD &nbsp;|&nbsp; **前置条件**：模块一完成 &nbsp;|&nbsp; **状态**：⏸ 待开始

### 目标

新建 `scripts/shared/augmentation/` 模块，实现可组合的增强流水线，支持**离线**（生成时写入磁盘）和**在线**（运行时随机化）两种模式。

> **关键设计约束**：所有空间增强（翻转、旋转、Cutout）必须**同步作用于 vol0、vol1 和 flow**。对 flow 的方向分量也需做对应调整（例如沿 z 轴翻转后，`flow[0]`（dz）需取反）。

### 新建文件结构

```
scripts/shared/augmentation/
├── __init__.py
├── pipeline.py       ← AugmentationPipeline 类（流水线管理）
├── spatial.py        ← 空间类增强（Cutout、翻转、旋转）
├── intensity.py      ← 强度类增强（缩放、抖动、Gamma）
└── blur.py           ← 模糊/退化类增强
```

### 子任务 A：空间遮蔽类

- [ ] **T4.A1** **`Cutout3D`**：随机遮蔽 1~N 个长方体区域（强度置零），同步生成 `mask.npy`（被遮蔽区域为 1，其余为 0）。配置字段：`num_cutouts: [min, max]`、`cutout_size: [dz, dy, dx]`
- [ ] **T4.A2** **`RandomFlip3D`**：沿 x/y/z 方向各自以独立概率随机翻转。翻转 flow 时同步取反对应方向分量
- [ ] **T4.A3** **`RandomRotate90`**：沿 z 轴随机旋转 0°/90°/180°/270°。旋转 flow 时同步调整 dy/dx 分量

### 子任务 B：强度类

- [ ] **T4.B1** **`GlobalIntensityScale`**：vol0 和 vol1 分别（或共同）乘以随机系数 ∈ [a, b]
- [ ] **T4.B2** **`LocalIntensityJitter`**：生成低频空间随机场，与 vol0/vol1 相乘（局部对比度变化）
- [ ] **T4.B3** **`GammaCorrection`**：随机 gamma 校正，gamma ∈ [gamma_min, gamma_max]

### 子任务 C：模糊退化类

- [ ] **T4.C1** **`AnisotropicBlur`**：各向异性高斯模糊（z/y/x 方向独立 sigma）
- [ ] **T4.C2** **`Downsample`**：下采样再上采样（缩放因子 ∈ [factor_min, factor_max]）

### 子任务 D：流水线集成

- [ ] **T4.D1** **`AugmentationPipeline`**：按 YAML 列表顺序组合增强步骤，每步有独立的执行概率 `p`
- [ ] **T4.D2** 输入接口统一：`pipeline.apply(vol0, vol1, flow)` → `(vol0_aug, vol1_aug, flow_aug, mask)`
- [ ] **T4.D3** 在 `generate_dataset.py` 中集成增强开关（配置字段 `augmentation.enabled`）

### YAML 配置示例

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
      axes: [true, true, false]  # 允许 z/y 翻转，禁止 x 翻转
    - type: GlobalIntensityScale
      p: 0.8
      scale_range: [0.7, 1.3]
    - type: AnisotropicBlur
      p: 0.3
      sigma_z: [0.0, 1.0]
      sigma_xy: [0.0, 0.5]
```

### 测试要求

| 测试项 | 验收标准 |
|--------|---------|
| `Cutout3D` | 遮蔽区域 vol0/vol1 均为 0，`mask.npy` 对应位置为 1 |
| `RandomFlip3D(axis=z)` | vol0/vol1 沿 z 翻转，`flow[0]`（dz 分量）取反，flow[1]/flow[2] 不变 |
| `AugmentationPipeline` | 3 步流水线串行执行后，输出 shape 不变，无 NaN/Inf |
| 流场一致性 | 翻转/旋转后，`vol1[p + flow_aug(p)] ≈ vol0[p]` 仍成立（验证 5 个随机点） |

---

## 模块五：不确定性与可解释性支持

**负责人**：TBD &nbsp;|&nbsp; **前置条件**：模块二、三基本稳定 &nbsp;|&nbsp; **状态**：⏸ 待开始

### 目标

在每个样本中额外保存辅助图（mask、confidence map、reliability map），供下游模型学习不确定性估计。

### 涉及文件

| 文件 | 操作 |
|------|------|
| `scripts/data_generation/modules/beads.py` | 新增粒子区域 mask 生成 |
| `scripts/shared/output_writer.py` | 扩展保存逻辑，支持可选辅助图 |
| `README.md` | 更新输出文件列表 |

### 子任务

- [ ] **T5.1** **`mask.npy`（Pipeline A）**：在 `BeadRenderer.render()` 中额外输出布尔体积，标记每个 voxel 是否被粒子覆盖（强度超过阈值的区域为 True）
- [ ] **T5.2** **`confidence_map.npy`**：基于局部粒子密度的置信度场。方案：在 3D 体素格上统计局部窗口内的粒子数量，归一化至 [0, 1]。配置字段：`output.confidence_map: {enabled, window_size}`
- [ ] **T5.3** **`reliability_map.npy`（Pipeline B）**：基于局部强度方差评估纹理丰富度。方案：滑动窗口局部方差图，归一化至 [0, 1]。配置字段：`output.reliability_map: {enabled, window_size}`
- [ ] **T5.4** 更新 `output_writer.py` 的保存逻辑，支持以上辅助图的可选输出（通过配置项开关）
- [ ] **T5.5** 更新 `README.md` 的"输出格式"章节，列出所有可选辅助图文件

### 测试要求

| 测试项 | 验收标准 |
|--------|---------|
| `mask.npy`（T5.1）| `mask` 为 1 的区域与 `vol0` 高强度区域的 Pearson 相关系数 > 0.7 |
| `confidence_map.npy`（T5.2）| 目视验证粒子密集区域置信度高，边界/空白区域置信度低 |
| `reliability_map.npy`（T5.3）| 对有纹理区域和均匀区域各取 10 个样本点，有纹理区域均值显著高于均匀区域 |
| 关闭开关 | 配置项 `enabled: false` 时，不生成对应辅助文件，无报错 |

---

## 模块六：性能与工程优化

**负责人**：TBD &nbsp;|&nbsp; **前置条件**：模块一完成（可独立进行）&nbsp;|&nbsp; **状态**：⏸ 待开始

### 目标

提升数据生成速度，支持非正方形体积和增量生成，扩展更多输入格式。

### 子任务

- [ ] **T6.1** **前向 warp 加速**：将 `warping.py` 中的逐体素 Python 循环替换为向量化实现。推荐方案：`np.add.at` scatter 操作（纯 numpy，无需额外依赖）。需提供 benchmark 脚本，对比 128³ 体积的运行时间
- [ ] **T6.2** **Pipeline B 并行化**：在 `experimental_generator.py` 中加入 `multiprocessing.Pool`（参考 Pipeline A 的 `parallel` 配置项），配置字段：`generation.parallel`、`generation.num_workers`
- [ ] **T6.3** **非正方形体积支持**：检查所有模块中隐式假设 D=H=W 的代码，修改为支持任意 `(D, H, W)`。关键检查点：`deformation.py` 中的坐标网格生成、`warping.py` 中的体素索引
- [ ] **T6.4** **增量生成**：`generate_dataset.py` 支持 `--append` 标志，检测输出目录中已有的样本数（通过 `generation_summary.yaml`），从 N+1 开始编号追加

### 测试要求

| 测试项 | 验收标准 |
|--------|---------|
| 前向 warp 加速（T6.1）| Benchmark 脚本输出：128³ 体积加速后 < 原版 1/5 时间（或给出实测对比）|
| Pipeline B 并行（T6.2）| 固定 seed，parallel=true 与 parallel=false 输出的 `flow.npy` 完全一致 |
| 非正方形体积（T6.3）| 用 `(64, 96, 128)` 体积运行 Pipeline A 和 B，输出形状分别正确 |
| 增量生成（T6.4）| 已有 50 个样本的目录追加 10 个，最终目录有 60 个样本，编号连续 |

---

## 模块七：质量评估与文档工具

**负责人**：TBD &nbsp;|&nbsp; **前置条件**：其他模块基本稳定 &nbsp;|&nbsp; **状态**：⏸ 待开始

### 目标

提供端到端的数据集分析工具、启动时配置验证器和 warp 精度基准测试套件。

### 新建文件

```
scripts/tools/
├── __init__.py
├── analyze_dataset.py    ← 数据集统计分析与报告生成
└── benchmark_warp.py     ← Warp 精度基准测试
```

### 子任务

- [ ] **T7.1** **数据集分析工具** `scripts/tools/analyze_dataset.py`：
  - 遍历指定数据集目录，统计 EPE 分布（flow magnitude）、粒子密度分布（mean intensity）、SNR 分布
  - 输出 Markdown 报告（含统计表格和直方图路径）
  - 命令：`python scripts/tools/analyze_dataset.py --dataset_dir data/my_dataset/ --output report.md`

- [ ] **T7.2** **配置合法性验证器**：在 `generate_dataset.py` 启动时自动运行，检查：
  - `max_displacement < volume_size / 2`（避免过大位移）
  - `particles.num_range[1] > particles.num_range[0]`
  - `splits.train + splits.val + splits.test > 0`
  - PSF sigma 值合理（< volume_size / 4）
  - 验证失败时打印明确错误信息并终止，不静默失败

- [ ] **T7.3** **Warp 精度基准测试** `scripts/tools/benchmark_warp.py`：
  - 生成已知的正弦流场 → 正向 warp → 反向 warp → 计算往返 EPE
  - 对 backward warp + forward warp 两种方法分别测试
  - 验收标准：backward warp 往返 EPE < 0.05 voxel（128³ 体积）
  - 输出 Markdown 表格，含各方法的 mean/max EPE

### 测试要求

| 测试项 | 验收标准 |
|--------|---------|
| 分析工具（T7.1）| 对 10+ 样本的数据集运行，生成格式正确的 `report.md`，无崩溃 |
| 配置验证器（T7.2）| 故意写入非法配置（如 `max_displacement=200` for 128³），验证输出明确错误信息并退出 |
| Warp 基准（T7.3）| backward warp 往返 EPE < 0.05 voxel，结果在 Markdown 表格中展示 |

---

## 未来计划：GUI 开发

**状态**：🔒 未规划

本阶段不做详细规划。在模块一至七全部完成、整体命令行流程完全稳定后，将单独制定 GUI 开发计划文档。

**预期目标**（仅供参考）：
- 可视化的 YAML 参数配置界面（避免手动编辑配置文件）
- 实时生成进度监控（样本数、预计剩余时间）
- 内置切片可视化窗口（实时预览生成效果）
- 降低非编程用户的使用门槛

---

*文档维护：如更新模块状态、修改任务清单或记录新决策，请在顶部"修改记录"表格中添加一行，注明日期和执行人。*
