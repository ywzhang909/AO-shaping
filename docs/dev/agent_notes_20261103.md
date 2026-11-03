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

## Windows / CUDA DLL
- 迁移学习会话在 `fit` 段**静默退出**: 事件日志 `id=1000` 显示 `python.exe` 崩在
  `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\x64\cublasLt64_13.dll`
  (`0xc0000005`, 6.14.11.1340)。根因是 `import cupy` 按裸名加载 cuBLASLt, PATH 上的
  toolkit 副本顶替了 torch 2.14.1+cu130 自带的 6.14.11.1300。修法在
  `src/ao_shaping/_cublas_preload.py` (在 `import cupy` 之前按全路径预载 torch 那份);
  `AO_SKIP_CUBLAS_PRELOAD` 是逃生开关。**判据**: torch 的 `cublas64_13.dll` 是**按裸名**
  绑定 cuBLASLt 的, 所以决定崩不崩的是**先进进程的是哪一份** —— toolkit 副本先进
  (PATH 默认如此) 则 torch 的大 matmul 必崩; 预载后进程里只剩 torch 一份。

## 文档与门禁
- `docs/dev/module_inventory.md`、`docs/dev/report_subdir_index.md` 重钉计数与登记。
- `scripts/_common/module_size_baseline.json` 更新。
- 根 `AGENTS.md` 与 `report/AGENTS.md` 压缩：规则外迁至 `docs/dev/*`，编码规范见 `coding_conventions.md`，包位置见 `module_inventory.md`。

## 代理执行提醒
- 修改报告/移动文件后同步 `scripts/_common/provenance.py::REPORTS` 与 `tests/ao_shaping/scripts/test_report_provenance.py::_PREEXISTING_BROKEN`，并跑 `python scripts/sync_report_provenance.py`。
- 新增环境变量必须登记到 `docs/environment.md`（`test_env_names_read_by_code_are_documented` 门禁）。
- 代码改动保持绝对导入、`from __future__ import annotations`、类型注解与 ruff 门禁。