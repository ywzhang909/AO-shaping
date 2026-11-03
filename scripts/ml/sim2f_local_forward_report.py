"""Render the November 3 local-forward CCD experiment and its audit trail."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts._common.provenance import provenance_block_for
from scripts.ml.sim2f_local_forward_experiment import metrics

REPORT_KEY = "report/shape_trials/local_forward/local_forward_20261103.md"
OUT = ROOT / "report/shape_trials/local_forward"
SESSION = ROOT / "data/debug/local_forward_cone_20261103_2242b"
FAILED = ROOT / "data/debug/local_forward_ring_20261103_2230"


def main() -> None:
    """Recompute every CCD metric before reporting independently replayed gains."""
    (OUT / "artifacts").mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    summary = json.loads((SESSION / "summary.json").read_text(encoding="utf-8"))
    failed = json.loads((FAILED / "summary.json").read_text(encoding="utf-8"))
    target = np.load(SESSION / "target.npy")
    support = np.load(SESSION / "support.npy")
    cx, cy = summary["fixed_center_xy"]

    def crop(name: str) -> np.ndarray:
        frame = np.load(SESSION / "frames" / f"{name}.npy")
        return frame[cy - 128 : cy + 128, cx - 128 : cx + 128]

    max_error = 0.0
    for record in summary["captures"]:
        recomputed = metrics(crop(record["name"]), target, support)
        max_error = max(
            max_error,
            *(abs(value - record["metrics"][key]) for key, value in recomputed.items()),
        )
    if max_error > 1e-5:
        raise ValueError(f"saved metrics do not match frames: {max_error}")
    replay = summary["replay"]
    gains = np.array([row["gain"] for row in replay])
    means = {
        arm: {
            key: float(np.mean([row[arm][key] for row in replay]))
            for key in replay[0][arm]
        }
        for arm in ("reference", "candidate")
    }
    a = np.mean(
        [crop(f"replay{i:02d}_{k}") for i in range(5) for k in ("a1", "a2")], axis=0
    )
    b = np.mean(
        [crop(f"replay{i:02d}_{k}") for i in range(5) for k in ("b1", "b2")], axis=0
    )
    radial = np.hypot(*np.indices((256, 256)) - 127.5)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    for ax, image, title in zip(
        axes[:2],
        (a, b),
        ("Continuous cone: replay A", "Differentiable update: replay B"),
        strict=True,
    ):
        artist = ax.imshow(image, cmap="magma", vmin=0, vmax=18)
        ax.contour(support, levels=[0.5], colors=["cyan"], linewidths=0.6)
        ax.set_title(title)
        ax.set_xlabel("CCD pixel")
    fig.colorbar(artist, ax=list(axes[:2]), label="Counts (0.35 ms)", shrink=0.7)
    for image, label in ((a, "A: seed"), (b, "B: optimised")):
        profile = [image[(radial >= r) & (radial < r + 1)].mean() for r in range(125)]
        axes[2].plot(profile, label=label)
    axes[2].axvspan(0.3 * 256, 0.42 * 256, color="gray", alpha=0.15)
    axes[2].set(
        xlabel="Radius (CCD px)",
        ylabel="Azimuthal mean (counts)",
        title="Radial profile; target shaded",
    )
    axes[2].legend()
    noise = np.array([row["noise"] for row in replay])
    axes[3].bar(
        np.arange(1, 6), gains, color="#0072B2", label="Measured objective gain"
    )
    axes[3].plot(
        np.arange(1, 6),
        3 * noise,
        "o",
        color="#D55E00",
        label="3 x within-block spread",
    )
    axes[3].set(
        xlabel="Independent ABBA block",
        ylabel="Objective gain",
        title="Final replay: 5/5 accepted",
    )
    axes[3].legend(fontsize=8)
    fig.savefig(OUT / "figures/ccd_replay.png", dpi=180)
    plt.close(fig)
    artifacts = {
        "session": str(SESSION.relative_to(ROOT)),
        "summary_sha256": hashlib.sha256(
            (SESSION / "summary.json").read_bytes()
        ).hexdigest(),
        "metric_recompute_max_error": max_error,
        "captures": len(summary["captures"]),
        "rounds": [
            {
                key: row[key]
                for key in ("round", "delta_max_rad", "validation", "trials")
            }
            for row in summary["rounds"]
        ],
        "replay": replay,
        "replay_means": means,
        "gain_mean": float(gains.mean()),
        "gain_std": float(gains.std(ddof=1)),
        "gain_range": [float(gains.min()), float(gains.max())],
        "devices_closed": summary.get("devices_closed"),
        "restored_entry_slot": summary.get("restored_entry_slot"),
        "failed_seed_replay_gains": [row["gain"] for row in failed["replay"]],
    }
    (OUT / "artifacts/results.json").write_text(
        json.dumps(artifacts, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    # Keep a self-contained compact copy of the physical commands and model
    # responses; the 4GB native CCD record stays in its original session.
    for name in (
        "accepted_phase.npy",
        "basis.npy",
        "target.npy",
        "support.npy",
        "r00_jacobian.npy",
        "r01_jacobian.npy",
        "r02_jacobian.npy",
    ):
        data = np.load(SESSION / name)
        np.savez_compressed(OUT / "artifacts" / (Path(name).stem + ".npz"), data=data)
    np.savez_compressed(OUT / "artifacts/replay_means.npz", reference=a, candidate=b)
    before, after = means["reference"], means["candidate"]
    metric_rows = "\n".join(
        f"| {key} | {before[key]:.6f} | {after[key]:.6f} |"
        for key in ("objective", "l1", "ee", "ncc", "centroid_error_px")
    )
    round_rows = "\n".join(
        f"| {row['round'] + 1} | {row['trials'][0]['gain']:.6f} | {row['trials'][0]['noise']:.6f} | {row['validation']['predicted_gain']:.6f} | {row['validation']['measured_gain']:.6f} |"
        for row in summary["rounds"]
    )
    body = f"""# 三日报告诊断与局部物理预测模型可微整形（2026-11-03）

{provenance_block_for(REPORT_KEY)}

## 实测结论

在 Santec #1、1064 nm、2π=993 灰度与大恒 CCD 上完成 ring 的三轮可微优化。
连续锥相位作为起点，当前 CCD 成对探针辨识图像响应，Adam 对正向预测图像自动微分。
每轮 A–B–B–A 验收均通过；最终五组独立 A–B–B–A 重放也全部通过。
平均目标函数改善 {gains.mean():.6f}（{100 * gains.mean() / before["objective"]:.2f}%），
目标内能量提高 {100 * (after["ee"] - before["ee"]):.2f} 个百分点。
这些是同曝光 CCD 读数，不是模型自身评分。

| 指标 | 起始相位 A（5 组均值） | 优化相位 B（5 组均值） |
|---|---:|---:|
{metric_rows}

![实测重放对照](figures/ccd_replay.png)

圆环仍有明显角向亮度不均和中心残余光；NCC 约 0.40，不宣称均匀理想圆环或任意形状能力。
本次采用**实测标定的局部一阶正向响应模型**，不是已经修复并验证的全局 FFT 物理预测模型。
仅验证当前光路、当前相位邻域及 ring；不把其改善外推到 cross、frame 或新光路。

## 三天报告中的算法问题

1. **11 月 1 日：辨识与泛化并不等价。** `fov_generalization` 是纯仿真；两个 96×96 窗口虽拟合成功，确认误差比未校正更大（1.75–1.76 倍）。低残差和 `ok=True` 不能代替独立确认；主动探测同预算误差还差 17–42%。
2. **11 月 2 日：评分、划分与相位族泛化混在一起。** 去背景、增益投影、几何支持域将平/探针 NCC 纠正到约 0.93，但 ring/cross/frame 仍只有约 0.33–0.40；25 个验证行有 16 行与训练命令重复，不能作为新命令泛化证据。MTF/零级拟合验证增益仅约 0.0037 NCC。
3. **11 月 3 日：历史 GS 相位物理尺寸错配。** `sim2f_fov_hw_session._gs_phase` 将 256 CCD 目标直接 resize 到 512 FFT，而每个 FFT 样本为约 2.099 CCD 像素，整幅缩放将目标放大约 4.20 倍。修复为按 `sampling_ratio` 将 CCD 目标采样到 FFT 坐标。
4. **包裹后标量插值改变光场。** GS 返回 `angle(back)` 的 [-π,π] 相位，128→900 标量双线性插值在 ±π 边界引入错误相位。修复为确定性二维解包裹后嵌入，再由驱动唯一包裹。历史 ring 命令的标量插值与圆周插值在标称高斯照明权重下 RMS 相位差约 1.056 rad；这说明插值影响显著，不把它当成光路的完整归因。
5. **离线诊断的“实际相位”并不实际。** 历史诊断硬编码 pupil（瞳面）角点 (510,150)，近期会话中心 (960,720) 对应角点 (510,270)；对包裹灰度反解弧度做面积平均，也不等价于复光场。该诊断中的目标分数接近不能证明物理预测模型已匹配。
6. **零级定位受入口状态与双峰影响。** 显式曝光入口在写平相位前定位当前显示图案，且使用被 ROI 钳位的中心作为零级；同一个中心不能随候选相位重定位。本次明确写平相位后在原生帧固定中心 ({cx},{cy})，所有探针、验收、重放共享窗口。
7. **单轮改善不够。** 首次以历史 ring 命令做局部优化，某轮改善 0.00553，但最终五组重放四组变差、一组微小改善；全部未过门，作为负结果保留。本次连续相位初始化后独立重放才稳定通过。

## 模型、测量与验收

模型为 `I(c) = clamp(I0 + sum(J_k*c_k), 0)`，输出 256×256 CCD 计数图像。
27 个二维余弦相位模式（总阶数 ≤6、无 piston）；每个模式 ±0.25 rad 中心差分辨识 `J_k`。
每轮以接受相位为原点重新采集响应；不把被拒相位累积为新起点。
以共享 `objective_terms` 的 L1 + 0.25×(1−EE) 自动微分，Adam 120 步；组合更新最大绝对值 ≤0.8 rad。
独立混合命令（优化方向的 1/4 步）只作预测诊断，不参与 Jacobian 估计；三轮预测与实测改善均同号。
强度 NCC 容易被中心亮斑主导，因此同时报告改善幅度，不以 NCC 单独批准模型。

硬件参数：面板中心 (960,600)，900×900 面板区域；CCD 原生 2592×1944，2.2 µm 像元；
曝光 0.35 ms，单次捕获平均 16 帧，写入后稳定 0.3 s，分析窗口 256×256。
共 {len(summary["captures"])} 个已登记原生捕获，另有定位平相位帧。
起点 `0.55*hypot(x−63.5,y−63.5)` 是连续弧度锥相位，非 GS 输出。
探针、候选都经过相同面板嵌入/灰度/内存槽路径，保存实际下发灰度。

| 轮 | ABBA 实测改善 | 组内最大重复波动 | 独立混合命令预测改善 | 同命令实测改善 |
|---|---:|---:|---:|---:|
{round_rows}

接受条件：目标函数改善 > max(0.003, 3×组内 A/A、B/B 最大差)，EE 不下降，质心误差增加不超过 1 px。
最终五组全部通过，改善范围 [{gains.min():.6f}, {gains.max():.6f}]，组间标准差 {gains.std(ddof=1):.6f}。
“3×波动”是工程阈值，不解释为标准差或统计显著性。
最终实验系统恢复槽 {summary.get("restored_entry_slot")}，设备关闭记录为 `{summary.get("devices_closed")}`。

## 产物与复现

- [逐轮、逐重放结果与摘要哈希](artifacts/results.json)。全部 {len(summary["captures"])} 个捕获的指标从原生帧重新计算，最大误差 {max_error:.3g}。
- [最终相位](artifacts/accepted_phase.npz)、[相位基](artifacts/basis.npz)、[目标](artifacts/target.npz)、[支持域](artifacts/support.npz)、[重放平均帧](artifacts/replay_means.npz)。NPZ 数组键为 `data`，重放平均帧键为 `reference`/`candidate`。
- 三轮完整图像响应：[第一轮](artifacts/r00_jacobian.npz)、[第二轮](artifacts/r01_jacobian.npz)、[第三轮](artifacts/r02_jacobian.npz)。
- 原生帧、逐条相位、实际下发灰度及 summary：`{SESSION.relative_to(ROOT).as_posix()}`；失败对照：`{FAILED.relative_to(ROOT).as_posix()}`。
- 连续锥相位中心/斜率的探索记录：`data/debug/continuous_cone_scan_20261103/scan.json`；这些是探索值，结论只取后续独立重放。

```powershell
$env:PYTHONPATH='src;libs'
$env:AO_DAHENG_MAX_EXPOSURE_MS='18.8846'
# 起点文件为 128×128 的上述连续锥相位，先保存到 data/debug/。
.venv/Scripts/python.exe scripts/ml/sim2f_local_forward_experiment.py --out data/debug/local_forward_cone_NEW --phase data/debug/continuous_cone055_20261103.npy --pupil-center 960 600 --exposure-ms 0.35 --samples 16 --order 6 --rounds 3 --replays 5 --radius 0.8
.venv/Scripts/python.exe scripts/ml/sim2f_local_forward_report.py
```

GS 修复的仿真回归检查物理 CCD 目标内能量、NCC 与连续相位范围；新模型另有独立非线性响应测试与梯度数值一致性测试。
历史会话与历史分数保持原样，不用修复后的相位重写过去的结果。
"""
    (ROOT / REPORT_KEY).write_text(body, encoding="utf-8")
    logger.info("Report written: {}", REPORT_KEY)


if __name__ == "__main__":
    main()
