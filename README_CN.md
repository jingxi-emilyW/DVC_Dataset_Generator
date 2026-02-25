# 3D 体积变形数据集生成工具

从 RAFT-DVC 项目中提取的独立数据集生成模块，用于生成带有已知真值变形场（ground-truth flow）的三维体积图像对。
本模块可作为独立项目使用，无需 RAFT-DVC 的模型训练代码。

---

## 目录结构

```
.
├── scripts/
│   ├── data_generation/                    # Pipeline A：合成数据生成
│   │   ├── generate_confocal_dataset.py    # 主入口脚本
│   │   └── modules/
│   │       ├── beads.py                    # 粒子生成与渲染
│   │       ├── deformation.py              # 变形场生成（4种类型）
│   │       ├── imaging.py                  # 成像噪声模拟
│   │       ├── warping.py                  # 前向 warp（trilinear splatting）
│   │       └── quality_control.py          # 质量检查与可视化
│   │
│   └── data_generation_from_experiments/   # Pipeline B：从现有图像裁剪生成
│       ├── generate_from_tif.py            # 主入口脚本
│       ├── diagnose_tif.py                 # TIF 文件诊断工具
│       └── modules/
│           ├── tif_loader.py               # TIF 文件加载与过滤
│           ├── preprocessor.py             # 强度归一化、去噪预处理
│           ├── volume_extractor.py         # 随机裁剪子体积
│           └── dataset_builder.py          # 变形应用与数据集保存
│
└── configs/
    ├── data_generation/                    # Pipeline A 典型配置（2个）
    │   ├── confocal_particles_128_v1.yaml  # 128³，稠密粒子
    │   └── confocal_particles_128_v2.yaml  # 128³，稀疏粒子（推荐起点）
    └── data_generation_from_experiments/   # Pipeline B 配置（Franck 实验，4个）
        ├── exp_Franck_32.yaml
        ├── exp_Franck_64.yaml
        ├── exp_Franck_64_small_disp.yaml
        └── exp_Franck_128.yaml
```

---

## 两条 Pipeline 简介

### Pipeline A — 合成数据生成

**用途**：从零开始程序化生成粒子图像体积对，完全受控、可重现。

**流程**：
```
配置文件
  → 随机生成粒子（位置、半径、强度）
  → 生成变形场（affine / bspline / localized / combined）
  → 前向 warp 粒子坐标
  → 渲染 vol0（变形前）和 vol1（变形后）
  → 叠加成像噪声（泊松噪声 + 高斯模糊）
  → 质量检查 → 保存 .npy 文件
```

**运行**：
```bash
python scripts/data_generation/generate_confocal_dataset.py \
    --config configs/data_generation/confocal_particles_128_v2.yaml
```

**关键参数**（在 yaml 中配置）：
| 参数 | 说明 |
|------|------|
| `particles.num_range` | 粒子数量范围 |
| `particles.radius_range` | 粒子半径范围（voxels） |
| `particles.gaussian_sharpness` | 粒子高斯弥散系数（1.0=尖锐，2.0=标准） |
| `deformations.types` | 启用的变形类型列表 |
| `deformations.max_displacement` | 最大位移（voxels） |
| `imaging.noise_level` | 泊松噪声强度 |

---

### Pipeline B — 从现有图像裁剪生成

**用途**：从真实实验图像（confocal、荧光等）中随机裁剪子体积，施加合成变形场后构建训练数据集。适用于真实图像纹理但已知 GT 变形的半合成数据生成。

**流程**：
```
TIF 文件目录
  → 加载并过滤有效 TIF 文件（尺寸、通道检查）
  → 随机裁剪多个子体积（质量过滤）
  → 强度归一化 / 去噪预处理
  → 施加合成变形场（同 Pipeline A 的变形模块）
  → 前向/反向 warp 生成图像对
  → 保存 .npy + 元数据
```

**运行**：
```bash
python scripts/data_generation_from_experiments/generate_from_tif.py \
    --config configs/data_generation_from_experiments/exp_Franck_128.yaml

# TIF 文件诊断（运行前先检查数据）
python scripts/data_generation_from_experiments/diagnose_tif.py \
    --input_dir /path/to/tif_files
```

**目录约定**：
- 输入 TIF 文件：`data_tif/exp_<name>/` 目录（在 yaml 中配置）
- 输出数据集：`data/<dataset_name>/` 目录

---

## Warp 策略说明

### 反向 warp + Swap（默认，推荐）

`warp_mode: backward_swap`（在 yaml 中配置或 `DatasetBuilder` 构造时传入）

```
生成流场 F（作为反向流）
  ↓
反向 warp：vol_warped[q] = vol_original[q - F(q)]
  （scipy.ndimage.map_coordinates，trilinear，精确）
  ↓
Swap 角色：vol0 = vol_warped，vol1 = vol_original
  ↓
保存前向流：saved_flow = -F（从 vol0 到 vol1，数学精确）
```

**优点**：数学精确，无需迭代求逆。生成的 `(vol0, vol1, flow)` 三元组满足：
`vol1[p + flow(p)] ≈ vol0[p]`

### 前向 warp（可选，供对比）

`warp_mode: forward`

```
生成流场 F（作为前向流）
  ↓
前向 warp：trilinear splatting（vol1 = forward_warp(vol0, F)）
  ↓
保存前向流：saved_flow = F
```

**注意**：前向 warp 可能产生空洞（无源 voxel 映射的区域），适用于稀疏粒子场景。

### 实现对比

| 方法 | 实现 | 精度 | 速度 |
|------|------|------|------|
| 反向 warp | `scipy.ndimage.map_coordinates` | 精确（无空洞） | 快 |
| 前向 warp | Trilinear splatting（纯 numpy） | 近似（可能有空洞） | 慢（逐体素） |

---

## 输出格式

每个数据集分为 `train/` `val/` `test/` 三个 split，每个 split 结构：

```
<dataset_name>/
├── train/
│   ├── vol0/           sample_0000.npy  # (D,H,W) float32，参考帧
│   ├── vol1/           sample_0000.npy  # (D,H,W) float32，变形帧
│   ├── flow/           sample_0000.npy  # (3,D,H,W) float32，GT 变形场 [dz,dy,dx]
│   ├── metadata/       sample_0000.json # 生成参数、统计信息
│   └── visualizations/ sample_0000.png  # 可选，切片可视化
├── val/
├── test/
├── generation_config.yaml               # 完整生成配置（复现用）
└── generation_summary.yaml             # 生成摘要（文件数、用时等）
```

**流场约定**：`flow[c, z, y, x]` 表示从 vol0 到 vol1 的位移，`c=0` 是 z 方向，`c=1` 是 y 方向，`c=2` 是 x 方向（单位：voxels）。

---

## 依赖

```
numpy
scipy
pyyaml
tqdm
tifffile        # 仅 Pipeline B 需要
matplotlib      # 可选，用于可视化
pyvista         # 可选，用于 3D 渲染
```

**无 PyTorch 依赖**。原 RAFT-DVC 项目中 `dataset_builder.py` 使用 `torch.nn.functional.grid_sample` 实现反向 warp，此处已替换为等价的 `scipy.ndimage.map_coordinates` 实现。

---

## 变形类型

由 `scripts/data_generation/modules/deformation.py` 中的 `DeformationGenerator` 实现，支持 4 种变形类型（在 yaml 的 `deformations.types` 中启用）：

| 类型 | 说明 | 适用场景 |
|------|------|---------|
| `affine` | 全局仿射变形（旋转、平移、剪切） | 全局刚体运动 |
| `bspline` | B-spline 平滑随机场 | 连续平滑非刚体变形 |
| `localized` | 局部区域集中变形 | 局部应力集中 |
| `combined` | 以上类型的随机组合 | 最大多样性 |

---

## 粒子渲染控制（Pipeline A）

`beads.py` 中的 `BeadRenderer` 使用高斯函数渲染粒子：

```
intensity(r) = peak_intensity * exp(-r² / (gaussian_sharpness * radius²))
```

| `gaussian_sharpness` 值 | 效果 |
|------------------------|------|
| 1.0（V2/V3 使用） | 更尖锐、边界清晰的粒子 |
| 2.0（默认值） | 标准高斯弥散 |
| >2.0 | 更弥散、低对比度的粒子 |

---

## 已有配置文件参考

### Pipeline A 配置

| 配置文件 | 体积尺寸 | 粒子半径 | 粒子数量范围 | 变形类型 | 特点 |
| --- | --- | --- | --- | --- | --- |
| `confocal_particles_128_v1` | 128³ | 2.5–3.0 | 400–1500 | affine, bspline | **稠密**，大粒子，标准高斯弥散 |
| `confocal_particles_128_v2` | 128³ | 0.8–3.0 | 200–2000 | 全部4种 | **稀疏**，粒子大小多样，更尖锐（推荐） |

> 需要其他尺寸（32³、64³）或更多变形类型时，参考以上配置新建即可。

### Pipeline B 配置（Franck 实验室数据）

| 配置文件 | 体积尺寸 | 说明 |
| --- | --- | --- |
| `exp_Franck_32` | 32³ | 小尺寸，快速实验 |
| `exp_Franck_64` | 64³ | 中等尺寸 |
| `exp_Franck_64_small_disp` | 64³ | 中等尺寸，小位移变形 |
| `exp_Franck_128` | 128³ | 完整尺寸，正式训练用 |

---

## 扩展开发计划

本节是本项目的完整路线图，按模块分类列出，供 AI 辅助开发时作为参考文档。

---

### 模块一：架构整合（高优先级）

- [ ] **统一 Pipeline A 与 Pipeline B 为单一数据生成接口**
  - 目前两条 Pipeline 有各自独立的入口脚本，但共用变形、质量控制、输出模块
  - 目标：设计统一的顶层 API（如 `generate_dataset.py`），用户只需在配置中指定 `source_type: synthetic` 或 `source_type: crop_from_image`，其余流程完全共用
  - Pipeline B 视为"用真实图像替换合成粒子渲染"的特殊情形，而非独立功能
  - 扩展 Pipeline B 的输入格式：除 TIF 外，支持 HDF5、Zarr、NumPy 体积文件
  - 支持在同一配置中混合多个数据源（合成 + 多个实验数据集），自动合并为单一训练集

---

### 模块二：粒子与图像生成多样性（`beads.py` / `imaging.py`）

#### 粒子形态扩展

- [ ] 椭球形粒子（各向异性 sigma）：模拟显微镜 z 方向分辨率低的 PSF 展宽
- [ ] 多类粒子共存（不同半径分布、强度分布的子群混合，如细胞核 + 细胞质标记）
- [ ] 粒子空间聚集/排斥模型（代替纯随机分布），模拟真实细胞排列
- [ ] 粒子强度非均匀性（空间位置相关的强度衰减，模拟远离焦平面的强度下降）

#### 成像模式（点扩散函数 PSF）扩展

- [ ] 宽场显微镜 PSF（较大 z 方向弥散）
- [ ] 共聚焦 PSF（当前默认，各向同性高斯）
- [ ] 双光子 PSF（更深组织，强 z 方向拉伸）
- [ ] 光片显微镜 PSF（单侧激发，z 方向梯形 PSF）
- [ ] PSF 参数化为 YAML 配置，支持用户自定义 PSF 核

#### 噪声模型扩展（`imaging.py` 中的 `ImagingSimulator`）

- [ ] 泊松噪声（光子散粒噪声，当前已有）
- [ ] 读出噪声（加性高斯，相机 CCD/sCMOS 的电路噪声）
- [ ] 背景荧光（空间平滑的低频随机场，模拟非特异性荧光）
- [ ] 条纹/环状伪影（模拟结构照明显微镜的特征噪声）
- [ ] 光漂白效应（时序数据中粒子强度随时间衰减）
- [ ] 成像深度相关的信噪比衰减（深层体素自动降低 SNR）

---

### 模块三：变形场多样性（`deformation.py`）

#### 新增变形类型

- [ ] 周期性变形（正弦波叠加），模拟机械振动或循环载荷
- [ ] 各向异性变形（x/y/z 方向位移幅度独立控制）
- [ ] 约束边界变形（体积边界处位移为零，模拟固定边界条件）
- [ ] 分层变形（不同深度层施加不同变形，模拟层状材料）
- [ ] 不连续/界面变形（在指定平面处位移场有跳变，模拟裂缝或滑移面）
- [ ] 基于物理模型的变形（线性弹性，基于 Lamé 方程的解析解）

#### 变形场质量控制增强

- [ ] 可压缩性约束（控制 divergence，模拟不可压缩材料）
- [ ] 光滑性约束（最大梯度限制，避免生成不物理的大梯度）
- [ ] 变形场可视化增强（彩色涡旋图、应变张量可视化）

---

### 模块四：数据增强库（新增 `augmentation/` 模块）

以下增强操作应设计为可组合的流水线，支持离线生成（写入磁盘）和在线增强（运行时随机化）两种模式。

#### 空间遮蔽类

- [ ] Cutout Augmentation：随机遮蔽 1–N 个长方体区域（强度置零），模拟稀疏特征输入；遮蔽区域同步记录为 `mask.npy`，供模型学习不确定性
- [ ] 随机裁剪/翻转/旋转（90° 倍数），扩大数据多样性

#### 强度增强类

- [ ] 全局强度缩放（随机乘以 [0.7, 1.3] 范围系数）
- [ ] 局部强度抖动（空间平滑的随机强度倍增场）
- [ ] 对比度增强/降低（gamma 校正）
- [ ] 通道/时间方向的强度漂移（模拟采集过程中光源不稳定）

#### 模糊与退化类

- [ ] 各向异性高斯模糊（z/y/x 方向不同 sigma，模拟 PSF 不对称）
- [ ] 运动模糊（沿主变形方向的方向性模糊）
- [ ] 下采样再上采样（模拟低分辨率采集）

#### 混合增强

- [ ] MixUp / CutMix 的 3D 版本（用于提升泛化性）
- [ ] 增强配置通过 YAML 描述，支持各增强步骤的概率和参数范围

---

### 模块五：不确定性与可解释性支持

- [ ] 生成时同步保存 `mask.npy`（标记哪些区域有真实粒子、哪些是空白/遮蔽区）
- [ ] 生成时同步保存 `confidence_map.npy`（基于局部粒子密度估计的置信度场）
- [ ] 为 Pipeline B 生成 `reliability_map.npy`（标记裁剪区域内纹理是否足够丰富）
- [ ] 输出中包含局部 SNR 图，供下游模型使用

---

### 模块六：性能与工程优化

- [ ] **前向 warp 加速**：当前逐体素 Python 循环 O(D·H·W) 较慢；方案：Numba JIT 或向量化 scatter（numpy advanced indexing + bincount）
- [ ] **并行数据生成**：当前 Pipeline A 已有 multiprocessing；Pipeline B 尚未并行化，统一后扩展
- [ ] **支持非正方形体积**：当前假设 D=H=W，扩展为任意 (D, H, W)
- [ ] **支持更多输入格式**：HDF5（h5py）、Zarr、NIFTI (.nii)、Imaris (.ims)
- [ ] **增量生成**：支持在已有数据集上追加样本，而非每次重新生成

---

### 模块七：质量评估与文档

- [ ] **端到端数据集分析工具**：生成后自动统计 EPE 分布、粒子密度分布、SNR 分布，输出 Markdown/HTML 报告
- [ ] **配置合法性验证**：YAML 加载后自动检查参数范围和相互约束（如 `max_displacement < volume_size / 2`）
- [ ] **基准测试套件**：用标准合成数据验证 warp 精度（backward_warp + forward_warp 的往返误差应趋近于零）

---

## 来源项目

本模块提取自 [RAFT-DVC](https://github.com/your-repo/RAFT-DVC)（3D Digital Volume Correlation）项目。
RAFT-DVC 是一个基于 RAFT 光流网络架构的三维体积变形追踪模型，用于数字体积相关（DVC）分析。

**主要变动**（相比原项目）：
- 移除了对 RAFT-DVC 模型代码的依赖
- `dataset_builder.py` 中的反向 warp 由 `torch.nn.functional.grid_sample` 替换为 `scipy.ndimage.map_coordinates`
- `dataset_builder.py` 中新增独立的 `forward_warp()` 静态方法（trilinear splatting）
- 新增 `warp_mode` 配置项，支持在两种 warp 策略间切换
