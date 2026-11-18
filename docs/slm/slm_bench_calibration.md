# SLM 实验系统标定与历史说明（原 model-in-loop 保留部分）

> 本文是一份**历史 / 标定说明**：记录从原 `model-in-loop`（MIL）链条中保留下来的
> 实验系统测量与标定知识。MIL 优化循环本身已经退役，本文**不再**描述如何运行闭环整形，
> 也不提供已删除脚本 / 路径的命令。
>
> 数值结论以报告为准，不在本文复述为「当前有效值」：
> - 探针离线总览：[`report/slm/bench_probe/report.md`](../../report/slm/bench_probe/report.md)
> - 几何标定制动报告：[`report/slm/model_in_loop_hw_runbook/model_in_loop_bench_calibration.md`](../../report/slm/model_in_loop_hw_runbook/model_in_loop_bench_calibration.md)
> - 曝光 / 焦面基础标定：[`report/slm/bench/bench_calibration_20261001.md`](../../report/slm/bench/bench_calibration_20261001.md)

## 1. 为什么保留这份文档（MIL 退役原因）

原 `slm-model-in-loop` 命令及其闭环（Step A 强随机探针逐轮重拟合共享 Zernike 像差 →
Step B 冻结像差合成目标相位 → 逐轮验收）已经从源码、CLI 与 GUI 中移除。退役依据是实测：
在这台 LCOS 实验系统上，高空间频率被强烈低通，「随机瞳面相位 ⇒ 远场散斑」这一模型前提
不成立——实测帧是一个紧致光斑 + 一圈暗晕，而模型预测铺满画幅的散斑，散斑相关度上限约
0.07，低于几何门限 0.5；即使越过几何标定，全频随机相位验证集相关性仍低于逐样本门限，
运行始终被门控拒绝（`bake_off_rejected`）。对比测试中迁移学习 / 多形状链路
（`transfer shapes`）优于 MIL，后者成形帧峰值仅 18–21/255、CV 未达验收。门槛从未被调低或
绕过——保留的标定知识用于支撑仍在使用的几何复用、指纹比对与迁移学习流程，而非复活 MIL。

保留的具体部分：
- **入射（瞳面）定位**：面板光斑中心 / 覆盖半径的实测；
- **曝光窗口探测**：每个会话重新 bracket，历史值不可跨光路复用；
- **sweep 几何标定 / 指纹**：相位—光斑扫描反解像素比例、tilt 焦比与束腰，并比对标定后
  光路是否变过；
- **runbook 中标定部分的使用**：采集与几何求解阶段；
- **退役原因**（本节）。

## 2. 入射（瞳面）定位

面板坐标与相机坐标**不是同一套**：两轴互换 90° 且尺度差 10 倍以上，因此 `--pupil-center`
一类参数只能在**面板上实测**，不能从相机位置推断。0 阶一律用 `argmax` 定位，不要用几何
中心。面板光斑中心 / 半径由半平面随机相位边界扫描实测（0 阶浓度掉下去的拐点即光斑边缘），
具体数值与出处见几何标定报告与 [`report/slm/bench_probe/report.md`](../../report/slm/bench_probe/report.md)。
同一会话内须确认光源 / 衰减片稳定；功率变化后须重新定位。

聚焦、彗差与对称性诊断另见 [`slm_alignment_probe.md`](slm_alignment_probe.md)；
采集前的完整设备 / 曝光纪律见 [`pre_run_characterization.md`](pre_run_characterization.md)。

## 3. 曝光窗口探测

曝光必须**每次运行前重新 bracket**，不得沿用历史值：

| 曝光区间（历史实测） | 结论 |
|---|---|
| ≥ 0.4 ms | 重复一致、单调、近线性，可用 |
| < 0.4 ms | 读数紊乱（同一档两次读到 80 与 255），不可用 |
| 饱和拐点 ≈ 2.0–2.2 ms | 到达量程饱和 |
| 建议工作点 1.0–1.5 ms | 峰值有余量（历史参考） |

历史 `1.5 ms` 只是某次会话的参考：同一次会话内平场峰值 8 分钟内从 202.5 涨到 246（+21%），
因此不能写死。正曝光上限由当前会话约束 `AO_DAHENG_MAX_EXPOSURE_MS` 给出，未设置或为零时走
画面探测；平相位很亮、全频随机相位很暗时应分别确定可测曝光。`--exposure-time-ms`（原 MIL
cli 选项）曾接受三态：正数 ms / `auto`（画面探测）/ `default`（保持设备当前曝光），裸 `0`
被拒——它经驱动会被钳到设备最小值 ~0.02 ms，比可用区间暗 20–75 倍。这些语义保留在现行
标定 / 整形入口中，具体以当前 CLI 为准；数值出处见
[`report/slm/bench/bench_calibration_20261001.md`](../../report/slm/bench/bench_calibration_20261001.md)。

## 4. sweep 几何标定与指纹

扫掠几何由相位—光斑扫描反解相机 / 模型像素比例、tilt 焦比与束腰；实现分布在
`src/ao_shaping/optimizer/wfless/` 的 `bench_geometry.py`（`BenchGeometry`）、
`bench_sweep.py`（`SweepRecord`）与 `bench_sweep_fit.py`（`calibrate_bench_geometry_from_sweep`）
（共享叶子见 `bench_geometry_common.py`），只做测量、不含闭环。设备动作由
`scripts/slm/model_in_loop_hw_runbook.py` 的 `--stage sweep` / `--stage calibrate` 驱动
（`--stage collect` / `sweep` / `calibrate` 仍是标定用途；`fit` / `shape` 属于已退役的闭环判决，
不再作为标定步骤使用）。

- 焦面尺度由相位斜坡扫描确定，与 Zernike 倾斜独立路线互证；两轴互换、相位相关位移在该
  实验系统不可用（shift_corr 峰值 0.041 < 0.30），位移测量必须回退到光斑质心。
- 宽度响应联合相对 RMS 超阈值时束腰不可辨识，标定器应拒绝输出并标注；不能仅凭几何拟合
  相关性就宣称模型可靠。
- **光路指纹**：标定结果只对产生它的光路有效。`src/ao_shaping/tools/slm/panel/bench_fingerprint.py`
  用平场 FWHM / hollowness / centroid 与「平场 + 已知 Zernike 倾斜」的 centroid 三个数判断
  光路相对标定是否变过（FWHM 表轴向位置、tilt_shift 表焦面尺度、centroid 表光轴），
  亮度刻意不参与判据。换光路 / 换 SLM / 换相机后须重新标定，不要静默复用旧几何。

四项 swept 探针的结论与数值见总览报告
[`report/slm/bench_probe/report.md`](../../report/slm/bench_probe/report.md)；几何求解的
实测常数见
[`report/slm/model_in_loop_hw_runbook/model_in_loop_bench_calibration.md`](../../report/slm/model_in_loop_hw_runbook/model_in_loop_bench_calibration.md)。

## 5. runbook 中标定部分的使用

在装配好光路、确认设备互斥后，用 runbook 的标定阶段采集并求解几何（仅列标定用途；
参数占位符须替换为本次探测值，不要把历史中心 / 曝光 / 半径写死）：

```powershell
python scripts/slm/model_in_loop_hw_runbook.py --stage sweep --out data/debug/model_in_loop_hw_runbook/session --pupil-center <x,y> --region 128 --far-field-size 1024 --collect-disc 450 --zernike-radius 450 --exposure-ms <本次探测值>
python scripts/slm/model_in_loop_hw_runbook.py --stage calibrate --out data/debug/model_in_loop_hw_runbook/session --region 128 --far-field-size 1024 --collect-disc 450 --zernike-radius 450
```

核对 `phase_offset.npy`、Recorder 的实际总相位与 `bench_geometry.json`；曝光、中心、半径都
按本次画面调整。这两条命令只产出标定证据，不含 MIL 闭环整形。更早的 MIL 复现命令
（含已删除的 `--phase-offset-file` 等开关与 `slm-model-in-loop` CLI）已随源码移除，本文不再
给出。
