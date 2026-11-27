# Agent 当日记录 2026-11-27（beam quality 整形链路降本，离线）

本轮**未占用硬件**；上机验证清单见文末，须用户确认后再执行。
①「隐藏页停渲染」已提交 `f43b4bc8`，其硬件 pytest 固化见 `19e88928`。

## 1. ② 逐帧计算降本（离线已改并自检）

只改 `src/ao_shaping/gui/ccd/beam_quality/analyzer.py`：

1. `_background_estimate`：≥ 20 万像素的帧按 4 像素步进取中值；小帧仍走精确全幅中值。
2. `_largest_component`：只在非零像素包围盒内做 `ndimage.label`，再还原成全幅布尔图；
   盒外全为背景，连通关系与全幅标注一致。
3. `fit_profile`：给 `curve_fit` 传解析雅可比 `_gaussian_1d_jac`，保留 `maxfev=4000`
   的失败兜底口径。

合成帧（640×480 / 1344×760 / 斜坡背景 / 饱和帧）离线对比，中位数：

| 场景 | 改前 ms | 改后 ms | 提速 |
| --- | ---: | ---: | ---: |
| 640×480 平背景 | 12.85 | 6.95 | 46.0% |
| 1344×760 平背景 | 30.86 | 16.50 | 46.5% |
| 1344×760 斜坡背景 | 36.77 | 25.57 | 30.5% |
| 640×480 饱和 | 12.97 | 6.97 | 46.2% |

数值差异：拟合参数 ≤ 5e-9，`center_spread` ≤ 0.025 px；未改变质心/椭圆口径。

自检：`.venv\Scripts\ruff.exe check` all passed；
`pytest tests/ao_shaping/gui/ccd/ -q --import-mode=importlib` 102 passed / 1 skipped；
`pytest tests/ao_shaping/gui/slm/test_transfer_desktop_live_view.py -q --import-mode=importlib`
24 passed；`git diff --check` 针对改动文件 exit 0。

## 2. GPU 批量可行性结论（离线实测，不采用）

`torch` 2.14+cu132（1 GPU）与 `cupy` 14.2 均可用。逐帧 reduction / elementwise
算子搬 GPU 后，单帧含 H2D+D2H 的收益很小；把 stats 融合成 bundle 后，
计入传输的批量均值在 640×480 为 n=4/8/16 → 2.00/1.53/1.18 ms/帧，
1344×760 为 4.25/3.30/3.28 ms/帧；n=32 不再改善。

- 单帧：640×480 bundle CPU 3.55 ms vs GPU 含传输约 2.85 ms；1344×760 CPU
  10.32 ms vs GPU 约 2.28 ms，但这是纯 stats，不含 `ndimage.label` / `curve_fit` /
  `to_uint8`——后者才是主热点且不是 GPU 批处理候选。
- 30 Hz 下缓冲 8–16 帧等于新增 0.27–0.53 s 延迟，与实时整形目标冲突。
- `torch.median` 对偶数个元素取较小中间值（探针中差 13 counts），`torch.quantile`
  与 `np.median` 一致但更慢；两种都不能无损替换现有中值口径。
- **裁定**：不采用 GPU 批量；继续用 CPU 抽样中值 + ROI 标注 + 解析雅可比。
  原始基准脚本与日志留在临时目录
  `C:\Users\zhangh\AppData\Local\Temp\opencode\gpu_batch_bench\`。

## 3. 上机清单（待用户确认，本轮未执行）

1. 设备：占用 SLM #1 + 大恒 CCD cam 0（页 1 真实闭环）；页 0 仿真源不占相机。
   开始前、结束后须在回复中明确报告占用与释放。
2. 命令：

   ```bash
   .venv\Scripts\python.exe -m pytest tests/ao_shaping/gui/ccd/test_beam_quality_hardware.py --hardware -v -q --import-mode=importlib
   ```

3. 观察：`visible` / `hidden` 两段的 `cpu_per_wall`、`processed_per_wall`、
   `poll_interval_ms`、`analyze_ms_median`；运行中 checkpoint 仍须 `render_active=false`；
   页 1 会话 `journal.json` / `optical_probe.json` 存在，退出后 `beam_source=None`、
   `transfer_running=false`。
4. 证据：`data/debug/beam_quality/hidden_gate_hardware.json`；会话目录
   `data/debug/slm_transfer/<时间戳>/`。
5. 若失败：先看 `current_problems()` 与运行按钮状态，不要重复启动第二个占设备的进程。
