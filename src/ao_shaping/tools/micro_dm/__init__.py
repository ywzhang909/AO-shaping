"""Micro-DM (R50Power) 台架工具。

包含:
- micro_dm_image_collect.py — 逐单元图像采集 (CLI:
  `python -m ao_shaping.tools.micro_dm.micro_dm_image_collect`)。遍历控制器 IP,
  对每个通道下发电压 → 相机采集 → 归位。图像按 `IP-通道号` 落盘, 磁盘布局
  (data/micro_dm_images/) 承重于 `scripts/` 侧的 Micro-DM diff 分析流水线,
  **改动落盘命名/目录前先看 `scripts/md_img_*.py`**。

本包**不做 eager re-export**: 子模块在模块作用域 import 了 `click` / 相机与 DM 驱动 /
`utils.image.hardware_utils`, 包级再导出会让 `import ao_shaping.tools.micro_dm` 付出这些代价。
需要什么直接从子模块导入。
"""