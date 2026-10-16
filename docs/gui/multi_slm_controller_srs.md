# Multi-SLM 控制器 (multi_slm_controller.py) — 软件需求规格说明书 (SRS)

## 1. 概述

### 1.1 文档目的

本文件定义 `src/ao_shaping/gui/slm/multi_slm_controller.py` 的功能需求、非功能需求、接口规范与关键设计约束。本模块是 Santec SLM-200 空间光调制器的 Streamlit 前端页面，支持最多两台 SLM，每台可独立连接、配置和相位编程，并内置闭环远场光束整形（GS+SPGD 与正向模型闭环）。

### 1.2 启动方式

```bash
streamlit run src/ao_shaping/gui/slm/multi_slm_controller.py
```

### 1.3 依赖模块

| 模块 | 职责 |
|------|------|
| `ao_shaping.drivers.slm.santec.Santec` | SLM 驱动（open/close/write_phase/display_memory/create_phase_from_array/apply_shift 等） |
| `ao_shaping.gui.slm.pattern_controls` | 图案注册表 `PATTERN_REGISTRY`、`generate_phase_gray`、`refresh_phase_preview`、`PatternControl` 基类 |
| `ao_shaping.gui.slm.shaping_controls` | 闭环整形区 `render_closed_loop_shaping`、`ShapingMethod` 适配器、装饰器 `shaping_session` |
| `ao_shaping.utils.wavefront.pattern_helper.PatternHelper` | 按设备缓存的图案计算助手（面板分辨率、像素间距） |
| `ao_shaping.runners.runner_common` | 共用 runner 参数对象与保存器（与 CLI 同一代码路径） |

### 1.4 用户画像

- 光学实验操作员：连接 SLM、下发相位图案、进行远场图像观测
- 闭环整形用户：运行 GS+SPGD 或正向模型闭环整形，获取最佳相位与远场均匀性

---

## 2. 功能需求

### 2.1 设备发现与连接

| ID | 需求 | 优先级 |
|----|------|--------|
| FR-1.1 | 支持扫描 SLM 编号 1..8，探测设备是否空闲（SLM_Ctrl_Open 返回码） | 必须 |
| FR-1.2 | 空闲设备读取序列号、Panel_Res、Pitch_um、波长等设备参数 | 必须 |
| FR-1.3 | 已占用设备（ret > 0）标记为“使用中”，不可连接 | 必须 |
| FR-1.4 | 无设备（ret < 0）跳过，不报错 | 必须 |
| FR-1.5 | 支持最多两台 SLM 同时连接，各自独立 session state 命名空间 `slm1` / `slm2` | 必须 |
| FR-1.6 | 探测运行在守护线程中，单台超时 3s，避免 Santec SDK 挂起阻塞 UI | 必须 |

### 2.2 SLM 参数配置

| ID | 需求 | 优先级 |
|----|------|--------|
| FR-2.1 | 支持设置波长（nm），EEPROM 写入约需 40s | 必须 |
| FR-2.2 | 支持设置显示模式（内存模式 / DVI 模式），DVI 模式被拒绝（已禁用） | 必须 |
| FR-2.3 | 支持设置 2π 灰度值 `_max_gray` | 必须 |
| FR-2.4 | 支持加载/保存设备配置（JSON） | 可选 |
| FR-2.5 | 支持加载/清除波前矫正 CSV | 可选 |
| FR-2.6 | 支持设置平移 (shift_x/shift_y) 并应用 | 可选 |

### 2.3 相位图案生成与下发

| ID | 需求 | 优先级 |
|----|------|--------|
| FR-3.1 | 支持多种全息相位图案：平场、线性光栅、圆形光栅、透镜、全息光栅、闪耀光栅、棋盘格、二元光栅、微透镜阵列、湍流相位屏、Zernike、达曼光栅、涡旋相位、半半相位 | 必须 |
| FR-3.2 | 图案生成通过 `PatternHelper` 生成弧度相位，经 `Santec.create_phase_from_array()`（弧度→灰度+矫正+LUT+平移）下发 | 必须 |
| FR-3.3 | 平场灰度走原始 uint16 路径（`np.full((h,w), gray, dtype=np.uint16)`），不经弧度转换 | 必须 |
| FR-3.4 | 内存槽自动轮换：连续写同一槽位固件判 no-op，每次选取 2..125 中不同于当前显示槽的槽位 | 必须 |
| FR-3.5 | 支持“从 CSV 加载相位”：CSV 格式 1200×1920，值 0~1023，首行/首列为 Y/X 索引；管线 `load_gray_from_csv` → `csv_to_phase` → `display_phase` | 必须 |
| FR-3.6 | 支持“导出当前相位为 CSV”：数据区为弧度值，保留 Y/X 行列索引；导出文件不可再交给灰度加载管线 | 必须 |
| FR-3.7 | 支持“应用平移”：读取当前相位、撤销旧偏移、施加新偏移、轮换槽位、重新显示 | 可选 |
| FR-3.8 | 支持“保存当前相位为底相位”，并在下发时自动叠加底相位（开关可控） | 可选 |

### 2.4 周期切换（持续任务）

| ID | 需求 | 优先级 |
|----|------|--------|
| FR-4.1 | 支持两相位 A/B 周期切换，频率由用户设定 | 必须 |
| FR-4.2 | 切换时钟基于 `time.time()` 墙钟差值：`int(elapsed * 2.0 * freq) % 2 == 0`，无累积误差 | 必须 |
| FR-4.3 | 后台线程绝不直接碰 `st.session_state`，通过 `threading.Event` 传递停止信号，通过可变列表（`slm_container`、`freq_ref`）传递快照 | 必须 |
| FR-4.4 | 每次切换选取不同于当前显示槽的内存槽位，确保 LCOS 面板刷新 | 必须 |
| FR-4.5 | 支持停止周期切换，释放线程与事件 | 必须 |

### 2.5 闭环远场整形

| ID | 需求 | 优先级 |
|----|------|--------|
| FR-5.1 | 支持 GS+SPGD 闭环整形（已验证配方），与 CLI `slm-gs-refine` 同一代码路径 | 必须 |
| FR-5.2 | 支持正向模型闭环整形（前沿，需 torch） | 可选 |
| FR-5.3 | 运行期间暂时释放目标 SLM 句柄，结束后重新获取（Santec SDK 句柄不可重入） | 必须 |
| FR-5.4 | 运行期间停止后台周期切换线程，避免冲突 | 必须 |
| FR-5.5 | 结束后把最佳相位（raw 弧度）经常规 `display_phase` 路径下发显示 | 必须 |
| FR-5.6 | 显示 flat/GS/best 各阶段得分、PIB、CV，以及 GS bake-off 判定（adopted/rejected） | 必须 |
| FR-5.7 | debug 模式下产物存于 `data/debug/slm_gs_refine/<ts>/` | 可选 |

---

## 3. 非功能需求

| ID | 需求 | 说明 |
|----|------|------|
| NF-1 | 会话状态隔离 | 每个 SLM 以 `slm{N}`（N=1 或 2）为命名空间，互不干扰 |
| NF-2 | 线程安全 | 后台周期切换线程不直接操作 `st.session_state`；所有 UI 更新在主线程完成 |
| NF-3 | 错误隔离 | 优化器崩溃不应导致页面设备句柄泄露，`shaping_session` 装饰器在 `finally` 中重新获取句柄 |
| NF-4 | 字节级一致 | GUI 预览与硬件下发共用 `create_phase_from_array()`，确保显示与实际下发一致 |
| NF-5 | 无 DVI 模式 | DVI 模式 `open()` 可能挂起 120s/300s，且挂起后 memory 模式 `open()` 也挂起，需物理断电恢复；因此仅提供内存模式 |

---

## 4. 接口规范

### 4.1 Session State Schema（每台 SLM）

| 键 | 类型 | 含义 |
|----|------|------|
| `slm{N}` | Santec | SLM 对象（断开时为 None） |
| `slm{N}_connected` | bool | 连接标志 |
| `slm{N}_wavelength` | int | 工作波长（nm） |
| `slm{N}_video_mode` | str | 显示模式（内存模式） |
| `slm{N}_next_memory` | int | 下一个内存槽提示 |
| `slm{N}_phase_preview` | ndarray | 缓存的预览图像 |
| `slm{N}_phase_source` | str | 缓存相位的来源标签 |
| `slm{N}_shift_x/y` | int | 平移值 |
| `slm{N}_use_correction` | bool | 波前矫正开关 |
| `slm{N}_toggle_phase_a/b` | ndarray | 周期切换存储的相位 |
| `slm{N}_toggle_active` | bool | 周期切换是否运行 |
| `slm{N}_toggle_frequency` | float | 切换频率（Hz） |
| `slm{N}_toggle_thread` | Thread | 后台线程句柄 |
| `slm{N}_toggle_stop_event` | threading.Event | 停止信号 |
| `slm{N}_toggle_freq_ref` | list | 可变频率引用（单元素） |
| `slm{N}_toggle_slm_container` | list | 可变 SLM 引用（单元素） |
| `slm{N}_base_phase` | ndarray | 用于叠加的底相位 |
| `slm{N}_overlay_base` | bool | 叠加开关 |
| `slm{N}_width/height` | int | 面板分辨率 |
| `slm{N}_pixel_pitch_um` | float | 像素间距 |
| `slm{N}_bits` | int | 位深 |
| `slm{N}_loaded_config` | dict | 缓存的配置文件 |

### 4.2 主要公共函数

| 函数 | 签名 | 说明 |
|------|------|------|
| `main()` | 无 | Streamlit 页面入口 |
| `render_slm_sidebar(slm_num)` | 无 | 设备发现、连接、参数设置侧边栏 |
| `render_phase_control(slm_num)` | 无 | 相位图案生成、CSV 加载/导出、平移、底相位、周期切换控制面板 |
| `display_slm_status(slm_num)` | 无 | 显示 SLM 连接状态 |
| `_stop_toggle(slm_num)` | 无 | 停止指定 SLM 的周期切换线程 |
| `_toggle_phases_task(container, phase_a, phase_b, stop_event, freq_ref)` | 无 | 后台周期切换线程主体 |
| `_scan_available_slms()` | list[dict] | 扫描 SLM 设备列表（守护线程，3s 超时） |

---

## 5. 关键设计规则（来自 AGENTS.md）

1. **内存槽轮换**：对同一槽位连续 `write_phase` + `display_memory` 会被固件判为 no-op，LCOS 面板不刷新。换槽在 2..125 中随机选取，不等于当前显示槽；进程重启后仍有效（首次 `get_displayed_memory_number()` 读取当前槽）。
2. **DVI 模式已禁用**：`video_mode=1` 的 `open()` 可能挂起，且挂起后 memory 模式 `open()` 也挂起。仅提供内存模式。
3. **平场灰度 RAW 路径**：始终通过 `np.full((h,w), gray, dtype=np.uint16)` 发送原始 uint16 灰度。切勿将平场相位走 `create_phase_from_array()`。
4. **平移数学**：唯一实现在驱动层的静态 `Santec.shift_phase`。重写缓存相位必须走驱动层 `apply_shift()`，绝不能对缓存相位再跑 `write_phase`（会二次叠加矫正）。
5. **后台线程**：绝不直接碰 `st.session_state`。`_toggle_phases_task` 使用 `slm_container`（列表）和 `freq_ref`（列表）作为可变快照，并用 `threading.Event` 传递停止信号。
6. **时序**：优先用 `time.time()` 墙钟差值而非计数器 —— `int(elapsed * 2.0 * freq) % 2 == 0` 按精确频率切换，无累积误差。
7. **0 级光斑**：绝不假设 0 级光斑位于相机画面中心 —— 用 `argmax` 定位。

---

## 6. 与 CLI 的对应关系

| GUI 功能 | CLI 命令 | 代码路径 |
|----------|----------|----------|
| 相位图案生成与下发 | — | `pattern_controls.generate_phase_gray` → `Santec.create_phase_from_array` |
| GS+SPGD 闭环整形 | `slm-gs-refine` | `SlmGsRefineParams` → `build_gs_refine_config` → `optimize_slm_gs_refine` → `save_gs_refine_results` |
| 正向模型闭环整形 | `slm-model-in-loop` | `SlmModelInLoopParams` → `build_model_in_loop_config` → `optimize_slm_model_in_loop` → `save_model_in_loop_results` |

---

## 7. 相关文档

- 用户手册: [`docs/gui/multi_slm_controller_user_manual.md`](./multi_slm_controller_user_manual.md)
- SLM GUI 操作时长分类与路线图: [`docs/slm/slm_gui_operation_map.md`](../slm/slm_gui_operation_map.md)
- GUI 索引: [`docs/gui.md`](../gui.md)
- 闭环整形实测报告: [`report/shape_trials/gs_refine_live_20261012.md`](../../report/shape_trials/gs_refine_live_20261012.md)
- 正向模型闭环实测报告: [`report/shape_trials/model_in_loop_live_20261013.md`](../../report/shape_trials/model_in_loop_live_20261013.md)
