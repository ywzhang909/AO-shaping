# AO-Shaping 模型训练指南

## 支持的模型类型

| 模型类型 | 输出 | 说明 |
|---------|------|------|
| `unet` | 相位图 (256x256) | U-Net+GAN 图像到图像翻译 |
| `resnet18` | Zernike系数 | ResNet18骨干网络回归 |
| `resnet34` | Zernike系数 | ResNet34骨干网络回归 |
| `simple_cnn` | Zernike系数 | 轻量级CNN回归 |

## 快速开始

### 统一训练命令

```bash
# 训练 Zernike 系数预测模型 (resnet18)
uv run python src/ao_shaping/ml/train.py train \
    --data-dir data/slm_capture \
    --model-type resnet18 \
    --epochs 100 \
    --batch-size 8

# 训练 UNet 相位图预测模型
uv run python src/ao_shaping/ml/train.py train \
    --data-dir data/slm_capture \
    --model-type unet \
    --epochs 100 \
    --batch-size 8
```

### 完整参数列表

#### 通用参数
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--data-dir` | 训练数据目录 | 必填 |
| `--output-dir` | 模型保存目录 | `checkpoints` |
| `--model-type` | 模型类型 | `resnet18` |
| `--epochs` | 训练轮数 | `100` |
| `--batch-size` | 批次大小 | `8` |
| `--lr` | 学习率 | `1e-3` |
| `--weight-decay` | 权重衰减 | `1e-4` |
| `--train-split` | 训练集比例 | `0.7` |
| `--val-split` | 验证集比例 | `0.15` |
| `--target-size` | 目标图像尺寸 | `256x256` |
| `--device` | 设备 (cuda/cpu) | `auto` |
| `--seed` | 随机种子 | `42` |
| `--use-wandb/--no-wandb` | 使用wandb | `False` |
| `--wandb-project` | wandb项目 | `ao-shaping` |
| `--resume` | 恢复检查点 | `None` |

#### Zernike模型参数
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input-mode` | 输入模式 | `combined` |
| `--n-zernike-terms` | Zernike项数 | `55` |
| `--n-max` | 径向阶 | `10` |

#### UNet模型参数
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--lambda-l1` | L1损失权重 | `100.0` |
| `--lambda-adv` | 对抗损失权重 | `1.0` |
| `--gan-mode` | GAN模式 | `lsgan` |
| `--loss-type` | 损失类型 | `angular` |

## 使用示例

### 1. 训练 Zernike 系数模型
```bash
uv run python src/ao_shaping/ml/train.py train \
    --data-dir data/slm_capture \
    --model-type resnet18 \
    --epochs 100 \
    --lr 1e-3 \
    --n-zernike-terms 55 \
    --input-mode combined
```

### 2. 训练 UNet 相位图模型
```bash
uv run python src/ao_shaping/ml/train.py train \
    --data-dir data/slm_capture \
    --model-type unet \
    --epochs 100 \
    --lr 2e-4 \
    --lambda-l1 100 \
    --gan-mode lsgan
```

### 3. 训练 Simple CNN
```bash
uv run python src/ao_shaping/ml/train.py train \
    --data-dir data/slm_capture \
    --model-type simple_cnn \
    --epochs 100 \
    --lr 5e-4
```

### 4. WandB 记录
```bash
uv run python src/ao_shaping/ml/train.py train \
    --data-dir data/slm_capture \
    --model-type resnet18 \
    --epochs 100 \
    --use-wandb \
    --wandb-project ao-shaping \
    --wandb-name exp001
```

### 5. 从检查点恢复
```bash
uv run python src/ao_shaping/ml/train.py train \
    --data-dir data/slm_capture \
    --model-type resnet18 \
    --epochs 50 \
    --resume checkpoints/best.pt
```

### 6. 推理预测
```bash
# 单张图像
uv run python src/ao_shaping/ml/train.py predict \
    --checkpoint checkpoints/best.pt \
    --image input.npy \
    --output result.npy

# 批量预测
uv run python src/ao_shaping/ml/train.py predict \
    --checkpoint checkpoints/best.pt \
    --data-dir data/slm_capture \
    --output-dir predictions
```

## WandB 超参数搜索

统一支持所有模型类型的超参数搜索:

```bash
uv run python src/ao_shaping/ml/train.py sweep \
    --data-dir data/slm_capture \
    --project ao-shaping-sweep \
    --epochs 30 \
    --count 10
```

搜索参数:
- `lr`: [1e-4, 5e-4, 1e-3, 5e-3]
- `model_type`: [resnet18, resnet34, simple_cnn, unet]
- `input_mode`: [focus, pupil, combined] (Zernike)
- `n_zernike_terms`: [28, 55] (Zernike)
- `lambda_l1`: [50.0, 100.0, 200.0] (UNet)
- `gan_mode`: [lsgan, vanilla] (UNet)

## 输出文件

```
checkpoints/
├── best.pt          # 最佳验证损失模型
├── final.pt        # 最终模型
├── config.json    # 训练配置
└── history.json   # 训练历史
```

## 模型架构

- **UNet**: U-Net 生成器 + PatchGAN 判别器
  - Input: 2通道图像 (Daheng + MiiCam)
  - Output: 1通道相位图 (256x256)

- **ResNet18/34**: ResNet 骨干 + 全连接头
  - Input: 1-2通道图像
  - Output: n_zernike 系数

- **SimpleCNN**: 轻量级 CNN + 全连接头
  - Input: 1-2通道图像
  - Output: n_zernike 系数

## 注意事项

1. 确保 GPU 可用 (CUDA + PyTorch)
2. 数据目录结构:
   - Zernike: 包含 `sample_XXXX/` 子目录，每个包含 `focus.pt`, `pupil.pt`, `coefficients.pt`
   - UNet: 包含 `sample_XXXX/` 子目录，每个包含相机图像和相位图
3. wandb 默认离线模式，需要网络时去掉相关设置