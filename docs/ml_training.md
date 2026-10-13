# ML 训练指南

> 本文件从 README.md 拆出（2026-10-13）。

ML 代码在独立包 `src/ml/` (可选依赖组), 与硬件层解耦。训练器、模型与指标的
**权威说明在实验报告里** (报告是结论的唯一事实源, 脚本只做渲染):

| 任务 | 入口 | 报告 |
|---|---|---|
| Zernike 系数 → 远场前向模型 (`ml/zernike/`) | `python -m ml.zernike.train_coeff` | [`report/zernike_coeff2amp/report.md`](../report/zernike_coeff2amp/report.md) |
| 前向模型搜索 (physics × U-Net × loss) | `python scripts/forward_search.py` | [`report/loss_defects/forward_search_report.md`](../report/loss_defects/forward_search_report.md) |
| U-Net+GAN 相位预测 (`ml/phase/`) | `python -m ml.train train --output-mode phase --data-dir <dir>` (click CLI, 见 `src/ml/train.py`) | — |
| FourierGSNet 仿真矩阵 (phase→far-field 逆问题) | `python scripts/fouriergsnet_sim_train.py` | [`report/fouriergsnet_sim/report.md`](../report/fouriergsnet_sim/report.md) |
| 逆向光束整形 PhasorUNet (`ml/shaping/`, 目标远场 → SLM 相位, 带物理 round-trip 损失) | `python -m ml.shaping.train` (评估: `python -m ml.shaping.evaluate`) | [`report/inverse_shaping_unet/report.md`](../report/inverse_shaping_unet/report.md) |

两个关键口径 (完整论证见报告):

- **只用 R² / corr / 光斑域指标判断拟合质量**, 不要用 MSE/PSNR/SSIM。实测同一配置下
  `normalization="sum"` 给出 PSNR 72.1 dB / SSIM 0.9996, 而 R² 反而更差 —— 任何含 sum
  归一化的指标都在度量归一化本身。
- **fold 与 seed 不能混配**: 同 seed 的 N 折共享同一条初始化随机流, 配对检验的 p 值
  偏乐观。判定前跑多种子。

`ml/hwdataset/` 把 `data/debug/` 下的硬件调试产物转成 PyTorch Dataset
(输入 = SLM 相位 + 曝光, 输出 = 相机画面); 语料统计与抽样自检:

```bash
python -m ml.hwdataset.inspect --index-cache data/hw_index_cache.json --sample 5
python -m ml.hwdataset.inspect --index-cache data/hw_index_cache.json --build-cache
```
