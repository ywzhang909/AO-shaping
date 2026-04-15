# AO-Shaping 模型训练指南

## 快速开始

### 基础训练命令

```bash
# 使用默认参数 (angular loss, LSGAN, GPU)
uv run python src/ao_shaping/ml/train.py train \
    --data-dir data/slm_capture/20260403_173337 \
    --epochs 100 \
    --batch-size 8 \
    --output-dir checkpoints
```

### 完整参数列表

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--data-dir` | 训练数据目录 | 必填 |
| `--output-dir` | 模型保存目录 | `checkpoints` |
| `--epochs` | 训练轮数 | `100` |
| `--batch-size` | 批次大小 | `8` |
| `--lr` | 学习率 | `2e-4` |
| `--lambda-l1` | L1/Angular损失权重 | `100.0` |
| `--lambda-adv` | 对抗损失权重 | `1.0` |
| `--target-size` | 目标图像尺寸 (HxW) | `256x256` |
| `--train-split` | 训练集比例 | `0.8` |
| `--num-workers` | DataLoader工作进程数 | `0` |
| `--use-daheng/--no-daheng` | 使用Daheng相机数据 | `True` |
| `--use-miicam/--no-miicam` | 使用MiiCam相机数据 | `True` |
| `--device` | 设备 (cuda/cpu) | `auto` |
| `--seed` | 随机种子 | `42` |
| `--gan-mode` | GAN损失类型 (`vanilla`/`lsgan`) | `lsgan` |
| `--loss-type` | 主损失类型 (`l1`/`angular`) | `angular` |
| `--use-wandb/--no-wandb` | 使用wandb记录 | `False` |
| `--wandb-project` | wandb项目名称 | `ao-shaping` |
| `--wandb-name` | wandb run名称 | `auto` |
| `--log-images/--no-log-images` | 记录验证图像 | `True` |

## 损失函数说明

### Angular Loss (默认)
适用于相位预测，因为相位值在 [0, 2π] 范围内循环:
```python
diff = torch.remainder(pred - target + π, 2π) - π
loss = mean(diff²)
```

### L1 Loss
标准的L1损失，用于像素级回归。

## 使用示例

### 1. 使用Angular损失训练 (推荐)
```bash
uv run python src/ao_shaping/ml/train.py train \
    --data-dir data/slm_capture/20260403_173337 \
    --epochs 100 \
    --batch-size 8 \
    --loss-type angular \
    --gan-mode lsgan \
    --output-dir checkpoints_angular
```

### 2. 使用L1损失训练
```bash
uv run python src/ao_shaping/ml/train.py train \
    --data-dir data/slm_capture/20260403_173337 \
    --epochs 100 \
    --batch-size 8 \
    --loss-type l1 \
    --output-dir checkpoints_l1
```

### 3. 使用WandB记录训练 (离线模式)
```bash
uv run python src/ao_shaping/ml/train.py train \
    --data-dir data/slm_capture/20260403_173337 \
    --epochs 100 \
    --batch-size 8 \
    --use-wandb \
    --wandb-project ao-shaping \
    --wandb-name exp001 \
    --output-dir checkpoints_wandb
```

### 4. 从检查点恢复训练
```bash
uv run python src/ao_shaping/ml/train.py train \
    --data-dir data/slm_capture/20260403_173337 \
    --epochs 50 \
    --resume checkpoints/best.pt \
    --output-dir checkpoints
```

### 5. 推理预测
```bash
# 单张图像预测
uv run python src/ao_shaping/ml/train.py predict \
    --checkpoint checkpoints/best.pt \
    --image input.npy \
    --output result.npy

# 批量预测
uv run python src/ao_shaping/ml/train.py predict \
    --checkpoint checkpoints/best.pt \
    --data-dir data/slm_capture/20260403_173337 \
    --output-dir predictions
```

## 输出文件

训练完成后，在 `--output-dir` 下会生成:

```
checkpoints/
├── best.pt          # 最佳验证损失模型
├── checkpoint_10.pt # 第10轮检查点
├── checkpoint_20.pt # 第20轮检查点
├── ...
├── final.pt         # 最终模型
├── config.json      # 训练配置
└── training_history.json # 训练历史
```

## 模型架构

- **Generator**: U-Net (124M参数)
  - Input: 2通道图像 (Daheng + MiiCam)
  - Output: 1通道相位图 (256x256)
- **Discriminator**: PatchGAN (2.7M参数)

## 注意事项

1. 确保GPU可用 (CUDA 13.0 + PyTorch 2.10+)
2. 数据目录需包含 `sample_XXXX/sample.pt` 文件
3. wandb默认使用离线模式，需要网络时去掉相关设置
