# S2Diff：高光谱超分辨率实验代码

当前仓库正在实现高光谱超分辨率中的两个创新方向。现阶段已经进入 **Innovation 1：传感器退化一致性渐进扩散** 的模型训练阶段；Innovation 2（光谱安全的 MSI 高频可迁移引导）暂不接入 predictor。

## 当前 Innovation 1 定义

前向渐进状态：

```text
D_t: HR-HSI -> 当前原生退化状态
D~_t = U_t o D_t
x_t = D~_t(X)
```

物理退化主路径使用 normalized-adjoint lift，使所有 diffusion state 保持在 HR 网格。

逆过程固定为：

```text
x_{t-1} = x_t + D~_{t-1}(X0_hat) - D~_t(X0_hat)
```

当前默认：

- scale trajectory: `1 -> 2 -> 4`
- `T = 12`
- degradation main mode: `physical`
- physical lift: `normalized_adjoint`
- predictor input: `x_t, t`
- predictor output: clean HR-HSI `X0_hat`
- first-stage loss: `L1 + lambda_sam * SAM`
- `lambda_deg = 0` by default; sensor-domain degradation consistency is implemented but disabled initially
- timestep sampling: 80% global uniform + 20% transition-neighborhood emphasis

## 关键文件

| 文件 | 说明 |
|------|------|
| `degradations/` | physical / Gaussian+Bicubic / Bicubic 退化与渐进轨迹 |
| `check_degradation_trajectory.py` | 前向渐进退化 sanity check |
| `models/predictor.py` | Innovation 1 时间条件 clean-HSI predictor |
| `innovation1.py` | 时间步采样、训练、可选 L_deg、完整 T->0 逆推与评估 |
| `main.py` | Innovation 1 训练 / 测试入口 |
| `data_loader.py` | HSI 数据读取、patch、SRF 协议与通用数据字段 |
| `losses.py` | 当前提供 SAMLoss |
| `metrics.py` | PSNR / RMSE / SAM / ERGAS / SSIM / CC |
| `srf_utils.py` | SRF 读取、插值、权重构建、HSI->MSI |
| `tests/test_degradation_closure.py` | 退化终点闭合与伴随算子测试 |
| `tests/test_innovation1_training.py` | predictor / training / reverse inference smoke tests |

## 固定传感器协议

后续 HSI-MSI 融合实验统一使用：

- PaviaU: IKONOS 4-band SRF
- Houston13: WorldView-2 8-band SRF
- Chikusei: WorldView-2 8-band SRF

Innovation 1 当前不将 HR-MSI 输入模型，但 dataloader 继续保留这套协议，便于 Innovation 2 无缝接入。

## 训练 Innovation 1

PaviaU 物理退化主实验：

```bash
python main.py \
  --stage train \
  --dataset PaviaU \
  --degradation_mode physical \
  --diffusion_steps 12 \
  --scale_ratio 4 \
  --lambda_l1 1.0 \
  --lambda_sam 0.1 \
  --lambda_deg 0.0
```

Houston13：

```bash
python main.py \
  --stage train \
  --dataset Houston13 \
  --degradation_mode physical
```

Chikusei：

```bash
python main.py \
  --stage train \
  --dataset Chikusei \
  --degradation_mode physical
```

普通 Gaussian+Bicubic 对照：

```bash
python main.py \
  --stage train \
  --dataset PaviaU \
  --degradation_mode gaussian_bicubic
```

纯 Bicubic 对照：

```bash
python main.py \
  --stage train \
  --dataset PaviaU \
  --degradation_mode bicubic
```

## 测试 / 完整逆推

默认读取：

```text
checkpoints/innovation1/<dataset>_innovation1_<degradation_mode>.pth
```

运行：

```bash
python main.py \
  --stage test \
  --dataset PaviaU \
  --degradation_mode physical
```

也可显式指定 checkpoint：

```bash
python main.py \
  --stage test \
  --dataset PaviaU \
  --degradation_mode physical \
  --resume checkpoints/innovation1/your_checkpoint.pth
```

## 重要实验约束

Innovation 1 训练阶段只使用 `batch["gt"]` 构造：

```text
x_t = D~_t(X)
```

不会把 `hr_msi` 输入 predictor。

通用 dataloader 当前仍保留历史 `lr_hsi` 字段，该字段由旧 Gaussian+Bicubic 路径生成。**Innovation 1 的物理退化训练与评估不会使用这个字段。** 评估时 terminal LR 会直接由当前 `process.terminal_observation(gt)` 生成，从而继续保证：

```text
D_T(X) = Y_LR-HSI
```

推理初始化为：

```text
x_T = U_T(Y_LR-HSI)
```

并从 `t=T` 迭代到 `t=0`。

## 可选第二阶段 L_deg

基础 predictor 稳定收敛后，可直接开启：

```bash
--lambda_deg 0.1
```

对应：

```text
L_deg = ||D_t(X0_hat) - D_t(X)||_1
```

该损失在原生传感器退化域计算，与 lift 选择解耦。

## 测试

```bash
pytest -q tests/test_degradation_closure.py
pytest -q tests/test_innovation1_training.py
```

## 数据目录

```text
project/
├── data/
│   ├── raw/
│   ├── wavelengths/
│   ├── srf/
│   └── srf_weights/
├── checkpoints/
├── logs/
├── outputs/
├── degradations/
├── models/
└── tests/
```
