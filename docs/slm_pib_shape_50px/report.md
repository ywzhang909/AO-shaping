# slm-pib 仿真运行报告 (SPGD 方形目标, 2f-Fourier 数字孪生)

**生成时间**: 2026-09-26 17:41:32

> ⚠️ **本报告的数值已过时且结论方向写反，勿引用 (2026-10-01 复核)**：
> 1. **`shape` 是最大化目标**（`slm_zernike_pib.py:710-712`：`objective_mode = "max" if objective in ("pib","avg_radiu","shape","roi_pib","rms_pib") else "min"`）。
>    下文 §2「解读」称其为最小化 —— 这是 `generate_slm_pib_sim_report.py:424-429`
>    自己标注的 **2026-10-01 缺陷**。按真实方向，`-1.2017 → -0.9127` 是**改善**（更接近 0），
>    而框内能量 `_p%` 下降 0.4942 → 0.3893 说明**均匀性换能量**，正是 §3.3 所述的取舍。
> 2. 本次运行**早于 `FAR_FIELD_PADDING` 过采样修复**。0 级光斑当时只有约 1.8 px FWHM，
>    与 [`slm_pib_sim/report.md`](../slm_pib_sim/report.md:47) 等填充后报告的数值
>    **不可直接比较**。
> 3. 下文的图片链接使用了 Windows 反斜杠，Markdown 渲染为文本（已修正为正斜杠）。
> 需要当前可信数据请读 `docs/slm_pib_sim/report.md`（2026-10-01 生成）。

**Fully offline** — 本报告由 `scripts/generate_slm_pib_sim_report.py` 离线生成, 仅读取 `slm_pib_runner --debug` 保存的 PKL/JSON 调试产物, 不打开任何硬件。

## 1. 运行说明

本运行使用 `src/ao_shaping/runners/slm_pib_runner.py` 的 **SPGD** 子命令, 在纯 numpy 2f-Fourier 仿真 (`src/ao_shaping/drivers/sim/slm_pib_sim.py`) 下执行, 无硬件。

- **相机**: `--cam_type sim` (注册到相机注册表, 读取仿真远场)
- **SLM**: `Santec` 被 monkeypatch 为 `SimSLMPib` (FFT 远场, 0 级光斑位于帧中心)
- **目标**: 方形 (`--target_shape square`, `--target_size` 相机像素)
- **搜索**: SPGD 梯度法 (`--optimizer_type adamod`), Zernike 系数 n≤4 (15 个模式)

光学模型: SLM 位于 2f 光路前焦面, CCD 位于后焦面, 因此 CCD 图像 = SLM 瞳孔场的 2D FFT (夫琅禾费远场)。输入为高斯光束, 施加 Zernike 相位后经 `np.fft.fft2` 传播到远场。

## 2. 运行 `run0` (目录 `20260926_172355`)

| 项目 | 值 |
|---|---|
| epoch 数 | 101 |
| 初始 J (shape) | -1.2017 |
| 最终 J (shape) | -0.9127 |
| 初始框内能量 _p% | 0.4942 |
| 最终框内能量 _p% | 0.3893 |
| 搜索配置 | {'epochs': 100, 'delta': 0.0005, 'lr': 0.5} |

![run0_objective](figures/run0_objective.png)

![run0_zernike](figures/run0_zernike.png)

![run0_phase_evolution](figures/run0_phase_evolution.png)

![run0_spot_evolution](figures/run0_spot_evolution.png)

![run0_phase](gifs/run0_phase.gif)

![run0_spot](gifs/run0_spot.gif)

### 解读

- 目标 J 从 -1.2017 变化到 -0.9127。**`shape` 是最大化目标**（见文首警示），故这是**改善** 0.2890
  （`slm_zernike_pib.py:710-712`；下方原「最小化」表述为 2026-10-01 确认的生成器缺陷）。
- 框内能量 `_p%` 从 0.4942 降到 0.3893：**J 的改善伴随框内能量下降**，即优化器用能量换了
  均匀性/形状分数。这个方向性 trade-off 只有在知道目标是 maximize 之后才读得出来。
- 低阶 Zernike 主要做波前校正/聚焦, 并非真正的方形成形 (方形需要全像素自由度, 见 AGENTS.md 反模式)。
- 相位图展示 15 个 Zernike 模式 (n≤4) 的加权合成; 远场图展示 0 级光斑的 FFT 传播结果。

## 3. 结论

1. **管线验证通过**: `slm-pib` 的 SPGD 闭环在纯仿真下可端到端运行, 无需硬件, 调试产物 (PKL/JSON/PNG) 与硬件运行格式一致, 可直接用于离线报告生成。
2. **仿真模型合理**: 2f-Fourier FFT 远场使 Zernike 相位对光斑产生真实可测的影响 (非零梯度), 与硬件 2f 光路 (SLM 前焦面 → 透镜 → CCD 后焦面) 一致。
3. **低阶 Zernike 的局限**: 与 AGENTS.md 反模式一致, n≤4 的 Zernike 是圆对称光滑基, 无法合成真正的方形远场; 本报告的方形目标用于验证 **目标函数 + 闭环反馈链路**, 而非真正的方形成形 (后者需 freeform/全像素相位, 如 `spgd-square --basis freeform`)。
4. **可复用**: 报告生成器完全离线, 任何一次 `slm-pib --debug` 运行 (仿真或硬件) 的调试产物都能用本脚本重新出报告。
