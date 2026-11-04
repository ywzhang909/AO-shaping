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
  (`0xc0000005`, 6.14.11.1340)。根因是 `import cupy` 按**裸名**加载 cuBLAS, PATH 上的
  toolkit 副本与 torch 2.14.1+cu130 自带的 6.14.11.1300 **各绑一个名字**; 决定生死的是
  「两个裸名是否落到同一个 build」, 不是「谁的版本新」。
- ⚠️ **首版修法 (`8014f579`) 方向是错的, 已由 `pin_cublas()` 取代**: 它只把 `cublasLt`
  钉到 torch 那份, 于是 `cublas` 仍走 PATH → 混合对, 实测 cupy 的 matmul / `linalg.inv` /
  `random` 与 torch 的 matmul **全部报错** (`CUBLAS_STATUS_INVALID_VALUE` /
  `CUSOLVER_STATUS_INTERNAL_ERROR`)。反过来把 torch 的一对全钉住也不通: torch 只 ship
  `cublas/cublasLt/cudart`, cupy 还要 `cusolver/cufft/curand`, 而这些**只存在于 CUDA 安装里**
  (OOPAO 的 `LiFT.py` 真用 `xp.linalg.pinv`)。
- 现行做法: `pin_cublas()` 把**两个裸名一起钉到 PATH 上第一套完整 CUDA** (toolkit 版本 ≥
  torch 自带版本时), torch 靠 CUDA minor 兼容跑在更新的 cuBLAS 上 —— 四种绑定组合的实测表见
  `docs/environment.md` §cuBLAS DLL 与 PATH。入口规则与禁止项见根 `AGENTS.md` §CUDA 原生库入口,
  门禁 `tests/ao_shaping/test_native_cuda_import_gate.py`, 逃生开关仍是 `AO_SKIP_CUBLAS_PRELOAD`。
- **已实测无效的"修法", 别再提**: `.venv` 隔离与 `uv run` 都不清理 `PATH`;
  `os.add_dll_directory(torch\lib)` 对 ctypes 裸名加载无效; 从 `PATH` 删 CUDA 目录后 scoop
  那份 (6.14.11.1380) 仍会被绑。把 `torch\lib` 置入 `PATH` 首位**有效**但是机器级配置。
  根治 = 机器只留一套 CUDA minor 且与 torch extra 同源。
- ✅ **本机已切 `--extra torch-cu132` (2026-11-03 收口)**: `torch 2.14.0+cu132` 自带的
  `cublasLt64_13.dll` = `6.14.11.1340` = PATH 首位 v13.2, 两个裸名**字面同版本**, 任何导入
  顺序都不再分叉 (实测先 `import torch` 与先 `pin_cublas()` 两种顺序, cupy
  `matmul/inv/pinv` + torch `matmul` 四算子全 OK)。代价: torch 2.14.1 → 2.14.0、
  torchvision 0.29.1 → 0.29.0。因此 **extra 选型是主修法**, `pin_cublas()` 退居兜底
  (给 minor 不匹配的机器和未来 extra)。

## utils/image · axis_order (#223 tier 1)
- 2D 点的轴序有了名字、准入清单与非方帧守卫：新增 `src/ao_shaping/utils/image/axis_order.py`（`AxisOrder`、`VALID_AXIS_ORDERS`、`normalize_axis_order`、`assert_square_frame`）。
- 涉及 `resample.py`、`beam_metrics.py`、`spots_calc.py`；所有新代码必须用 `AxisOrder` 命名轴序，非方帧必须经 `assert_square_frame` 守卫；注记统一用术语「全频随机相位」。
- 门禁：`tests/ao_shaping/utils/test_axis_order.py`。

## optimizer/wf · Recorder(mark=)
- `Recorder(mark=)` 必须传目标列名；`greedy_zernike.py` 与 `ga_zernike.py` 已修，启动即崩回归已闭合。
- 修改 Recorder 调用时务必确认 `mark` 列名与输出列一致。

## module-size gate / SSoT
- 模块规模门禁的复位按钮现在会报棘棘轮方向（`scripts/_common/module_size_baseline.json` 需与代码同步）。
- 两处 SSoT 单源违规收口：shape-vocab 与 recorder key；新增 shape/Recorder 字段时只允许一个源头。
- `tests/ao_shaping/test_conventions.py::test_all_ruff_pins_agree_and_are_exact`、`tests/ao_shaping/ml/test_train_cli_runnable.py::test_ruff_excludes_nothing_under_src` 仍为硬门禁。

## transfer / 工具链
- 曝光阶梯走不到接受带时续档重试，不再 abort 整条采集 lane（`tools/slm/transfer/capture.py` / `capture_response.py` / `validate.py`）。
- fit 段改回内存 payload：`build_fit_model` 必须保留 `samples=`（`tools/slm/transfer/fit.py`）。
- 归档列表分列 + 行内归档按钮 + fps 自动播放；归档回放扫描整个 debug 根下所有历史 pkl（`gui/slm/transfer_desktop/archive_replay.py`、`live_view.py`、`page.py`）。
- 实时画面改为逐帧刷新（`transfer_desktop/live_view.py`）。

## transfer-desktop / GUI
- PySide6 依赖补 `pyside6-addons`（QtCharts 只在 Addons 里）；`pyproject.toml` 已更新。
- 设备自检与只读零级读取；截图脚本无显示环境需 `QT_QPA_PLATFORM=offscreen`（见 `docs/environment.md`）。
- 用户手册与实验截图落地：`docs/GUI/slm/transfer_desktop/`。

## beam-quality / ccd
- 自动曝光闭环 + 特征计算多帧平均；`BaseCamera` 通用取流层接入 GUI 并兼容 Daheng。
- MiiCam callback stream 修复；Daheng 型号读取改用 `model_name`（`drivers/ccd/miicam/driver.py`、`drivers/ccd/daheng/driver.py`）。
- `drivers/sim/bench_twin.py`、`drivers/slm/santec/probe.py` 登记到 `module_inventory.md`。

## gs-refine / 调用点
- `_run_gs_refine_session` 调用点残留 cam 占位参数已修复；新增会话时勿再传占位 cam。
- `gs_control_corrected_arm.py` 尚未实现，docstring 已明示（跟进 `e496d08a`），勿当作可用 API。

## sim2f / 仿真（补充）
- 采集帧可视化与下一轮闭环新增零级漂移 / 低信噪比守卫；物理残差拟合新增契约测试。
- offline diagnostics + 11-03 报告（CCD GIFs）；`report/shape_trials/.../physical_refine` 新增 CCD 统计散点、整形能量图与采集序列 GIF；溯源登记同步。

## docs / drivers / wiki
- `docs/drivers` doc-tree 漂移修复：ccd 幻影 `_miicam_sdk`、slm 漏登 `probe.py`。
- call_graph wiki 行号重钉（optimizer_greedy 78/403 → 355/519、slm-transfer-shapes 721 → 723）。
- 术语门禁 4 文件 17 行机械统一（`--apply`）；新词登记 §3。
- `docs/dev/module_inventory.md`、`docs/dev/report_subdir_index.md` 重钉计数与登记。

## report / 溯源
- local_forward 报告 8 个 gitignored npz 改行内 code，修 relative_links 断链。
- prune 3 phantom REPORTS entries；sync README — 3 登记报告产物未入库移入「产物尚未提交」清单。
- 新增/移动报告后同步 `provenance.py::REPORTS` 与 `test_report_provenance.py::_PREEXISTING_BROKEN`，跑 `sync_report_provenance.py`。

## 文档与门禁
- `scripts/_common/module_size_baseline.json` 更新。
- 根 `AGENTS.md` 与 `report/AGENTS.md` 压缩：规则外迁至 `docs/dev/*`（CUDA → `docs/dev/cuda_native_libs.md`，溯源 → `docs/dev/report_provenance.md`），编码规范见 `coding_conventions.md`，包位置见 `module_inventory.md`。
- 新增环境变量必须登记到 `docs/environment.md`（`test_env_names_read_by_code_are_documented` 门禁）。

## 代理执行提醒
- 修改报告/移动文件后同步 `scripts/_common/provenance.py::REPORTS` 与 `tests/ao_shaping/scripts/test_report_provenance.py::_PREEXISTING_BROKEN`，并跑 `python scripts/sync_report_provenance.py`。
- 代码改动保持绝对导入、`from __future__ import annotations`、类型注解与 ruff 门禁。
- CUDA 原生库入口规则见 [docs/dev/cuda_native_libs.md](cuda_native_libs.md)；`pin_cublas()` 必须在 `import cupy`/OOPAO 之前。

## 代理执行提醒
- 修改报告/移动文件后同步 `scripts/_common/provenance.py::REPORTS` 与 `tests/ao_shaping/scripts/test_report_provenance.py::_PREEXISTING_BROKEN`，并跑 `python scripts/sync_report_provenance.py`。
- 新增环境变量必须登记到 `docs/environment.md`（`test_env_names_read_by_code_are_documented` 门禁）。
- 代码改动保持绝对导入、`from __future__ import annotations`、类型注解与 ruff 门禁。