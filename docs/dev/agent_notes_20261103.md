# Agent 注意事项（2026-11-03 当日修改）

> 外迁自根 `AGENTS.md`；按日追加，过期条目迁入 `docs/dev/` 对应专题。

## sim2f / 仿真
- 采集帧可视化与下一轮闭环新增**零级漂移 / 低信噪比守卫**；物理残差拟合新增契约测试。
- 报告与产物更新：`report/shape_trials/.../physical_refine` 新增 CCD 统计散点、整形能量图与采集序列 GIF；溯源登记同步。

## transfer-desktop
- PySide6 依赖、ABBA 复选框、实验截图与用户手册落地。
- 设备自检与**只读零级读取**；截图脚本在无显示环境需 `QT_QPA_PLATFORM=offscreen`（见 `docs/environment.md`）。

## beam-quality / ccd
- 自动曝光闭环 + 特征计算多帧平均。
- `BaseCamera` 通用取流层接入 GUI 并兼容 Daheng；MiiCam callback stream 修复；Daheng 型号读取改用 `model_name`。

## 文档与门禁
- `docs/dev/module_inventory.md`、`docs/dev/report_subdir_index.md` 重钉计数与登记。
- `scripts/_common/module_size_baseline.json` 更新。
- 根 `AGENTS.md` 与 `report/AGENTS.md` 压缩：规则外迁至 `docs/dev/*`，编码规范见 `coding_conventions.md`，包位置见 `module_inventory.md`。

## 代理执行提醒
- 修改报告/移动文件后同步 `scripts/_common/provenance.py::REPORTS` 与 `tests/ao_shaping/scripts/test_report_provenance.py::_PREEXISTING_BROKEN`，并跑 `python scripts/sync_report_provenance.py`。
- 新增环境变量必须登记到 `docs/environment.md`（`test_env_names_read_by_code_are_documented` 门禁）。
- 代码改动保持绝对导入、`from __future__ import annotations`、类型注解与 ruff 门禁。