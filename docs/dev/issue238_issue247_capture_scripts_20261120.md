# #238 功率参考 + #247 几何采集：两条离线车道（2026-11-20）

未开 SLM/CCD/激光、未执行 `--execute`、未在 `data/` 建目录、未 push、未改 issue 状态。任务路径：`scripts/slm/issue247_geometry_capture_20261120.py`（新）、`tests/ao_shaping/scripts/test_issue247_geometry_capture_20261120.py`（新）、`scripts/slm/issue238_gray_phase_probes.py`（改）、`scripts/experiments/run_issue238_period_sweep_20261119.py`（改）、`tests/ao_shaping/scripts/test_issue238_power_reference_20261120.py`（新）。

## #238 漂移解耦

- 采集脚本新增 `--power-reference {off,abba}`（默认 `off`，历史 plan 逐字不变）+ `--power-reference-frames N`（默认 2）；窗口新增 `--period-order {as-given,palindrome}` 并透传两旗标。
- `abba`：每个 sweep block 前后各插 `N` 帧 `flat`(φ=0) 参考帧（`role=pre/post`、`block_index`、`ref_slot`、单调 UTC 时间戳、同 uint16+sha256）。全程单一曝光，出现第二个曝光立即中止并写 `power_reference_abort.json` 后正常释放。`summary.json` 仅在启用时增描述性 `power_reference` 段（pre/post 原始总计数、`drift_ratio_post_over_pre`、逐帧 `scale`、`descriptive_only: true`、`dark_subtracted: false`），**不在脚本内计算或声明新 g₂π**。
- 纯函数 `interpolate_power_scale(frame_times, ref_times, ref_totals)`：`r_j = ref_total_j / mean(所有 ref_total)`，帧时刻按括号参考线性插值，返回 1.0 = 全程平均功率；区间外**常数保持**最近一侧比值。
- `palindrome` 把周期展开为如 `32,48,64,64,48,32`（奇数长度中心周期单份），使 period 与漂移可分离；`validate()` 拒绝 <2 个不同周期的回文、`abba` 下 `--power-reference-frames < 1`；`off` 不转发两旗标，plan-only 无副作用。
- 细节见 `issue238_binary_gray_capture_plan_20261119.md` §6.8。

## #247 几何采集

- plan-only 默认 / `--self-test` 无设备自检 / `--execute` 才开设备；11 图样 44 帧，schema `issue247-geometry-capture/v2`。
- 旧 `TILT_AMPLITUDE_RAD=0.5` 全跨仅 1.0 rad（≈ period 12063.7 px），远场位移比可测小约 190 倍，已判 `rejected_weak_design`。改真实棱镜斜坡 `--tilt-period-px 128.0`（短轴 1200 行为保 ≥10 条纹钳到 120 px，plan 同记 `requested_period_px`/`period_px`）。悲观几何（pitch 20 µm / focal 100 mm / CCD 3.45 µm / λ=1064 nm）位移 x 12.05 / y 12.85 CCD px，`--self-test` 门槛 ≥5 px。
- `chiral_L` 三块 π 相位在傅里叶面不可判读，已换成单向不等阶 blaze 光斑梯（`2π/96`→+1、`2π/32`→+3，spot `[0,+1,+3]` 单侧、非镜像/非 180° 不变）以打破 180° 分支；对称光栅对保留为比例参考。
- 仍缺物理常数：`camera_pixel_size_um`/`slm_pixel_pitch_um`/`focal_magnification`，plan `missing` 明列不猜；`--execute` 时从驱动读取 pitch/pixel size，读不到即 `null` + `missing`，不改 `data/slm_geometry_calib.npz`。
- 细节见 `next_hw_plan_transfer_candidates_issue247_20261117.md` 任务 B「几何采集脚本」小节。

## 本轮离线验证（orchestrator 复核，非仅子代理自述）

- `test_issue247_geometry_capture_20261120.py` 16 passed；`test_issue238_power_reference_20261120.py` + `test_issue238_gray_phase_probes.py` 30 passed；`test_script_imports_resolve` + `test_script_file_form_imports` 915 passed。
- 5 个任务文件 `py_compile` exit 0、`ruff All checks passed!`、无 BOM；`--self-test` 与 `--help` exit 0；plan-only `dir_exists=False`。
- #247 plan 独立确认：11 图样/44 帧、`tilt`/`ladder` 子对象与 5 条 `missing`（`camera_pixel_size_um`、`slm_pixel_pitch_um`、`focal_magnification`、`tilt_expected_ccd_px`、`ladder_expected_ccd_px`）。

## 既有基线失败（非本轮引入）

- `test_script_repo_root_bootstrap.py` 2 failed：旧脚本 `evaluate_multiplicative_gain_b.py` parents 深度、`issue247_capture_tilt_20261117.py` 硬编码机器路径。
- `test_cli_argv_binding` / `test_probe_flags` 2 failed：HEAD census 漂移，属 #251 范围。
- `pytest tests/ao_shaping/scripts -q` 本机挂起（>900 s 被 SIGTERM），已改定点用例，勿再整目录跑。

## 硬件前置（三 issue 共同）

区间内未采数；#234/#236 仍受 2026-11-20 双门 fail 门控；#238/#247 需用户明确授权窗口 + 唯一新 `--out`（`mkdir(exist_ok=False)`）+ 六步光路探测 + `DeviceLock`/`--wait-until-idle` + 设备占用报告。授权前一切保持 OPEN、不升默认、不冻结物理映射。
