# Multi-SLM 控制器 (multi_slm_controller.py) — 用户手册

## 1. 概述

`multi_slm_controller.py` 是 Santec SLM-200 空间光调制器的 Streamlit 前端页面。支持最多两台 SLM，每台可独立连接、配置和相位编程，并内置闭环远场光束整形（GS+SPGD 与正向模型闭环）。

启动方式：

```bash
streamlit run src/ao_shaping/gui/slm/multi_slm_controller.py
```

完整技术细节（类图、时序图、Session State Schema、图案注册表、设计规则）见 [`docs/gui/multi_slm_controller_srs.md`](./multi_slm_controller_srs.md)。

---

## 2. 快速上手

### 2.1 连接 SLM

1. 在侧边栏点击“刷新设备列表”，扫描 SLM 编号 1..8。
2. 对空闲设备点击“连接”，系统自动读取序列号、面板分辨率、像素间距、波长。
3. 连接成功后，SLM 对象存入 `session_state`，可进行相位下发。

### 2.2 生成并下发相位

1. 在相位控制面板选择图案类型（平场、线性光栅、透镜、Zernike 等）。
2. 调整参数后点击“从模式生成器生成相位”。
3. 系统通过 `PatternHelper` 生成弧度相位，经 `Santec.create_phase_from_array()` 转为 uint16 灰度（含矫正、LUT、平移），下发到 SLM 内存槽并显示。

### 2.3 从 CSV 加载相位

CSV 格式要求：1200×1920，值 0~1023，首行/首列为 Y/X 索引。

管线：`load_gray_from_csv`（驱动层格式校验）→ `csv_to_phase`（灰度→弧度）→ `display_phase`（`create_phase_from_array()` 弧度→灰度+矫正+LUT+平移）。

> ⚠️ 注意：导出的弧度 CSV（`Santec.save_phase_to_csv()`）数据区为弧度值，**不可**再交给 `load_gray_from_csv()` 或 `csv_to_phase()`；重新使用时应按弧度读取并传入 `create_phase_from_array()`。

### 2.4 周期切换（A/B 相位）

1. 先将两个相位分别设为“相位 A”与“相位 B”。
2. 设置切换频率（Hz），点击“开始周期切换”。
3. 后台线程按 `int(elapsed * 2.0 * freq) % 2 == 0` 精确频率切换，每次选取不同于当前显示槽的内存槽位（2..125），确保 LCOS 面板刷新。
4. 点击“停止周期切换”释放线程。

---

## 3. 闭环远场整形

`render_closed_loop_shaping()` 在「闭环整形」区渲染，取代已移除的开环方形整形。两种方法均在 GUI 内调用 canonical 优化器管线（与 CLI 同一代码路径），并在结束后把最佳相位（raw 弧度）经常规 `display_phase` 路径下发显示。

| 方法 | 代码路径（与 CLI 等价） | 说明 |
|------|-------------------------|------|
| **GS+SPGD 闭环整形（已验证配方）** | `SlmGsRefineParams` → `build_gs_refine_config` → `optimize_slm_gs_refine` → `save_gs_refine_results`（同 CLI `slm-gs-refine`） | 默认值即交替重放验证过的配方 |
| **正向模型闭环整形（前沿）** | `SlmModelInLoopParams` → `build_model_in_loop_config` → `optimize_slm_model_in_loop`（同 CLI `slm-model-in-loop`） | 需要 torch |

### 3.1 GS+SPGD 闭环整形（已验证配方）

GS 开环预矫正（bake-off，仅当实测优于平场才采用）+ 无感知 SPGD 细化。

GUI 默认值取自交替重放验证过的配方（[`report/shape_trials/gs_refine_live_20261012.md`](../../report/shape_trials/gs_refine_live_20261012.md)）：

- `cam_type=daheng`、曝光 `0.05 ms`（大恒曝光上限按会话：`AO_DAHENG_MAX_EXPOSURE_MS`，0 = 画面探测）
- `pupil_center=960,690`、`beam_radius_px=450`、`target_side=100`、`cam_size=250`
- `gs_iters=12`、`far_field_padding=3`、`phase_grid=12`、`epochs=12`、`seed=0`、`confirm_best` 开、`gs_method=plain`
- 验证结果：**flat 0.527 → plain GS 0.775 → SPGD best 0.780**（两次一致的交替重放）

### 3.2 正向模型闭环整形（前沿）

每轮用强随机探针重拟合正向模型的 Zernike 像差（Step A），再冻结该像差合成目标方斑相位（Step B），带 trust region 与逐轮验收。

需 `torch`。目前最佳均匀性 **CV 0.643 / EE 0.786**，但尚未成为被接受的均匀方形，见 [`report/shape_trials/model_in_loop_live_20261013.md`](../../report/shape_trials/model_in_loop_live_20261013.md)。

### 3.3 设备生命周期注意

上述优化器在**内部**自行打开/关闭 SLM 与相机（Santec SDK 句柄**不可重入**），因此 GUI 在整形运行期间会**暂时释放目标 SLM 句柄**、运行结束后重新获取（只释放目标 SLM，另一台 SLM 保持连接）。GUI 显示 flat/GS/best 各阶段得分、PIB、CV 与 GS bake-off 判定（adopted/rejected）；debug 开启时产物存于 `data/debug/slm_gs_refine/<ts>/`。

---

## 4. 关键设计规则

| 规则 | 说明 |
|------|------|
| 内存槽轮换 | 对同一槽位连续 `write_phase` + `display_memory` 会被固件判为 no-op，LCOS 面板不刷新。换槽在 2..125 中随机选取，不等于当前显示槽。 |
| DVI 模式已禁用 | `video_mode=1` 的 `open()` 可能挂起 120s/300s，且挂起后 memory 模式 `open()` 也挂起，需物理断电恢复。仅提供内存模式。 |
| 平场灰度 RAW 路径 | 始终通过 `np.full((h,w), gray, dtype=np.uint16)` 发送原始 uint16 灰度。切勿将平场相位走 `create_phase_from_array()`。 |
| 平移数学 | 唯一实现在驱动层的静态 `Santec.shift_phase`。重写缓存相位必须走驱动层 `apply_shift()`，绝不能对缓存相位再跑 `write_phase`（会二次叠加矫正）。 |
| 后台线程 | 绝不直接碰 `st.session_state`。`_toggle_phases_task` 使用可变列表作为快照，并用 `threading.Event` 传递停止信号。 |
| 时序 | 优先用 `time.time()` 墙钟差值而非计数器 —— `int(elapsed * 2.0 * freq) % 2 == 0` 按精确频率切换，无累积误差。 |
| 0 级光斑 | 绝不假设 0 级光斑位于相机画面中心 —— 用 `argmax` 定位。 |

---

## 5. 已移除功能

旧的开环方形整形功能（**GS方形整形** / **GS方形整形(高斯版)** / **稳像法整形**，含 `_gs_square_phase_radians` / `generate_gs_square_phase`）已移除。

原因：在本 2f-Fourier 台架上，用未标定台架参数做开环方形整形已被实测证明行不通 —— Zernike SPGD 与未标定 GS 从未产出过均匀方形，见 [`report/shape_trials/README.md`](../../report/shape_trials/README.md) 与 [`report/shape_trials/online_20261010.md`](../../report/shape_trials/online_20261010.md)。

取而代之的是上方第 3 节「闭环整形」区（GS+SPGD 闭环 / 正向模型闭环，均走已验证的闭环管线）。

---

## 6. 相关文档

- 需求规格说明书: [`docs/gui/multi_slm_controller_srs.md`](./multi_slm_controller_srs.md)
- SLM GUI 操作时长分类与路线图: [`docs/slm/slm_gui_operation_map.md`](../slm/slm_gui_operation_map.md)
- GUI 索引: [`docs/gui.md`](../gui.md)
- 闭环整形实测报告: [`report/shape_trials/gs_refine_live_20261012.md`](../../report/shape_trials/gs_refine_live_20261012.md)
- 正向模型闭环实测报告: [`report/shape_trials/model_in_loop_live_20261013.md`](../../report/shape_trials/model_in_loop_live_20261013.md)
