# S2Diff: HSI Super-Resolution Experimental Scaffold

当前仓库用于 HSI-MSI Fusion 超分实验。基础代码提供数据读取、SRF、指标、可切换 LR-HSI 退化和渐进物理退化轨迹；具体扩散网络尚未接入。

## 当前代码

| 文件/目录 | 说明 |
|---|---|
| `data_loader.py` | HSI 读取、patch 构建、LR-HSI 退化、HR-MSI 构建、DataLoader |
| `degradations/` | Bicubic、Gaussian+Bicubic、Physical 退化及 progressive trajectory |
| `check_degradation_trajectory.py` | 在接网络前检查各时间步退化状态 |
| `losses.py` | 当前已实现 SAMLoss；其他一致性损失将在模型阶段补充 |
| `metrics.py` | PSNR / RMSE / SAM / ERGAS / SSIM / CC |
| `srf_utils.py` | SRF 加载、插值、离散积分权重、HSI→MSI |
| `config.py` | 数据、退化和训练通用配置 |
| `main.py` | 检查数据与退化配置是否正确串联 |

## Degradation v1

### 1. LR-HSI 退化模式

三种退化通过 `--degradation_mode` 切换：

- `bicubic`：纯 Bicubic 下采样；
- `gaussian_bicubic`：Gaussian blur + Bicubic，下采样基线；
- `physical`：Gaussian PSF/MTF + detector area averaging + sampling，作为当前主实验退化。

新实验默认：

```text
degradation_mode = physical
scale_ratio = 4
mtf_nyquist = 0.2
psf_truncate = 3.0
```

旧 `make_lr_hsi()` 在未传入退化算子时仍保持 Gaussian+Bicubic 概念基线，避免旧脚本静默改变结果。

### 2. Progressive degradation

默认：

```text
T = 12
scale stages = 1 -> 2 -> 4
lift = auto
```

`auto` 会解析为：

- `physical` → `normalized_adjoint`；
- `bicubic / gaussian_bicubic` → `bilinear`。

物理模式下：

```math
D_t = B_{r_t} P_t
```

```math
\tilde D_t = U_t D_t
```

逆过程预留更新：

```math
x_{t-1}=x_t+\tilde D_{t-1}(\hat X_0)-\tilde D_t(\hat X_0)
```

所有 diffusion state 均位于 HR 网格。

### 3. 数据端与扩散终点闭合

`build_datasets()` 只构建一个 `degradation_operator`，训练集、测试集和 `ProgressiveDegradation` 共用该对象。

必须满足：

```math
D_T(X)=Y_{LR-HSI}
```

对应测试位于：

```text
tests/test_degradation_closure.py
tests/test_data_pipeline_degradation.py
```

## 退化轨迹 sanity check

先把原始 HSI 放入：

```text
data/raw/PaviaU.mat
data/raw/Houston13.mat
data/raw/Chikusei.mat
```

### Physical 主实验轨迹

```bash
python check_degradation_trajectory.py \
  --dataset PaviaU \
  --mode physical \
  --lift_mode auto \
  --crop_size 128 \
  --scale_ratio 4 \
  --total_steps 12 \
  --mtf_nyquist 0.2
```

### Gaussian+Bicubic 对照

```bash
python check_degradation_trajectory.py \
  --dataset PaviaU \
  --mode gaussian_bicubic \
  --lift_mode auto \
  --crop_size 128 \
  --scale_ratio 4 \
  --total_steps 12 \
  --legacy_sigma 2.0 \
  --legacy_kernel 5
```

### Bicubic 对照

```bash
python check_degradation_trajectory.py \
  --dataset PaviaU \
  --mode bicubic \
  --lift_mode auto \
  --crop_size 128 \
  --scale_ratio 4 \
  --total_steps 12
```

脚本逐时间步输出：

```text
t, scale, strength, scale_transition, step_l1,
PSNR, SAM(deg), mean, std, HF ratio
```

重点先检查：

1. `terminal_closure_max_abs_error` 是否接近 0；
2. `mean` 是否在 1→2→4 切换时出现不合理幅值跳变；
3. `step_l1` 在尺度切换点是否远高于邻近时间步；
4. PSNR、HF ratio 是否总体随退化逐渐下降；
5. SAM 是否保持合理，不出现由 lift 人为造成的突变。

输出保存在：

```text
outputs/degradation_trajectory/
```

## HSI-MSI Fusion 数据

正式融合实验建议使用真实 SRF：

```bash
python main.py \
  --dataset PaviaU \
  --degradation_mode physical \
  --msi_mode srf
```

仓库已经包含 PaviaU、Houston13、Chikusei 的波长文件和 WorldView-2 SRF CSV；原始 HSI `.mat` 不提交到仓库。

## 当前阶段

当前先完成退化轨迹与数据闭合验证。只有真实 HSI 上的 trajectory sanity check 通过后，再接 clean HR-HSI predictor 和扩散训练循环；MSI 高频可迁移引导留到创新点 1 稳定后加入。
