"""绝对强度保真度探针：模型预测 vs 实测 CCD（固定确定性相位集）。

本脚本回答一个问题：**冻结前向模型（``FrozenDefocusForward``）预测的强度，与同一
相位在 CCD 上实测的强度，绝对值上差多少？** 判据不是"优化能否收敛"，而是逐图样的
*绝对*保真度指标（issue #221 / #234 / #236）：

* 采样窗核心区 CV：模型侧 ``model_cv_core`` vs 实测侧 ``measured_cv_core_window``；
* 平均强度比：``mean_intensity_ratio_core = 实测核心均值 / 模型核心均值``；
* 全帧核心 CV（``measured_cv_core_full``）作为窗内指标的旁证。

与 :mod:`shape_transition_timing_20261106` 一致的约定（复用而非重写）：
* **同帧定心**：先显示零相位、抓一帧，用这一帧的 argmax 重建模型采样窗与两侧掩膜；
  离线（无 ``--execute``）时用 ``--spot-x/--spot-y``（默认取
  ``anchored_m3_20261116/plan.json`` 记录的 ``spot_xy``）。
* **曝光与锚定运行同口径**：``--exposure-ms`` 默认 0.5，与
  ``anchored_m3_20261116/plan.json`` 和 ``anchored_m3_rs15_20261116/plan.json`` 一致。
* **相位极性从检查点读**：``phase_sign`` 来自 ``frozen_parameters.json``，不硬编码。

相位集（12 个，全部确定性、可复现）：
``zero``；``rand_rms0p5_seed{0,1}``、``rand_rms1p5_seed{0,1}``（RMS 精确归一）；
``anchored_m3_best``、``anchored_m3_rs15_best``（两次锚定运行的最终最优网格，存在才纳入）；
``tilt_x_{1,3}rad``、``tilt_y_{1,3}rad``（Noll 2/3）；``defocus_z4_1rad``（Noll 4）。
Zernike 图样用 :func:`generate_noll_polynomial` 生成，单位圆外置 0（Noll 归一化下
``amplitude`` 即圆内 RMS 弧度）。

设备纪律（AGENTS.md）：
* 没 ``--execute`` 时不 import 驱动、不打开任何设备；仍加载检查点、构图样、算模型预测、
  写 ``plan.json`` / ``model/*.npy`` / ``phase/*.npy`` / ``summary.json``（实测列为
  ``null``），然后打印 ``skip: requires --execute`` 干净退出。
* 有 ``--execute`` 时单进程打开 Santec #N（内存模式 ``video_mode=0``）与大恒 CCD 一次，
  ``finally`` 释放；每次 ``display_data`` 与 ``grab`` 写进 ``device_log.json``。
* ``--wait-until-idle`` 语义同 ``shape_transition_timing``（本脚本自带实现，把"自身
  进程"排除在占用判定之外——上游实现的自排除标记是 ``shape_transition_timing``，
  对本脚本不成立，而本脚本的命令行含 ``defocus`` 检查点路径时会误判为自己占用自己）。

硬件上的 PYTHONPATH（离线验证不需要）::

    $env:PYTHONPATH = "D:\\Projects\\TIFO\\AO-shaping\\libs"

用法::

    python scripts/slm/absolute_fidelity_probe_20261116.py --out <dir>          # 离线
    python scripts/slm/absolute_fidelity_probe_20261116.py --execute --out <dir>  # 硬件
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from loguru import logger

from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
    FrozenDefocusForward,
    FrozenDefocusParameters,
    load_frozen_parameters,
    recentre_window,
)
from ao_shaping.algorithm.signal_processing.multishape_objective import (
    ShapeObjective,
    target_at,
)
from ao_shaping.utils.wavefront.zernike_calc import generate_noll_polynomial
from shape_transition_timing_20261106 import (
    CAMERA_SENSOR_PX,
    WINDOW_STEP_PX,
    clock,
    close_devices,
    display_phase,
    full_frame_raster,
    grab_average,
    monitor_cv,
    open_devices,
    sample_ccd_to_window,
    sha256_file,
)

#: 经标定的 M3 照明检查点（grid 128 / fft 512 / phase_sign -1.0 均从文件读出）。
DEFAULT_CHECKPOINT = (
    ROOT
    / "data/debug/slm_transfer/defocus_forward_20261107_focus_r2_fit"
    / "checkpoints/M3_illumination/frozen_parameters.json"
)

#: 提供"最终最优网格"的两次锚定运行（``plan.json`` 里 ``exposure_ms=0.5``）。
ANCHORED_RUNS: tuple[tuple[str, Path], ...] = (
    ("anchored_m3", ROOT / "data/debug/slm_transfer/anchored_m3_20261116"),
    ("anchored_m3_rs15", ROOT / "data/debug/slm_transfer/anchored_m3_rs15_20261116"),
)
ANCHORED_BEST_NPY = "phase/stage0_square_best_grid.npy"

#: 离线（无同帧定心）时的光斑中心 ``(col, row)``：取 ``anchored_m3_20261116/plan.json``
#: 记录的 ``spot_xy``。``--execute`` 时被同帧定心的实测值覆盖。
DEFAULT_SPOT_XY = (1224.0, 1093.0)

#: 与两次锚定运行同口径的采集参数。
DEFAULT_EXPOSURE_MS = 0.5
DEFAULT_FRAMES = 5
DEFAULT_SETTLE_S = 0.4
DEFAULT_SIDE = 400.0
DEFAULT_WINDOW_HALF_SPAN_PX = 260.0

#: 定心帧达到满量程这个比例即判为饱和。``np.argmax`` 返回**首个**最大值下标
#: (行优先), 一帧里整片 4095 时它取到的是饱和块的**顶边**而不是光斑中心 ——
#: 2026-11-16 实测: 12-bit 帧被 auto_exposure 推到 1000 ms 过曝后饱和, ``spot_xy``
#: 由正确的 ``[1224, 1093]`` 静默漂到 ``[1220, 606]``, 核心掩膜落到暗区, 模型侧
#: ``model_cv_core`` 塌缩到 0 (均值恰好等于 checkpoint 的 ``background``)。
#: 定心帧一饱和, argmax 就不可信, 必须报错而不是静默给出错误窗心。
CENTRING_SATURATION_FRAC = 0.98

#: Noll 索引 → ``(n, m)``。
NOLL_TILT_X = (1, 1)
NOLL_TILT_Y = (1, -1)
NOLL_DEFOCUS = (2, 0)


@dataclass(frozen=True, eq=False)
class Pattern:
    """一个待测相位图样（``phase`` 形状 ``(grid, grid)``，单位弧度）。"""

    pattern_id: str
    source: str
    phase: np.ndarray
    note: str = ""


def _torch():
    import torch

    return torch


def _num(value: Any, ndigits: int = 6) -> float | None:
    """把标量转成可 JSON 序列化的有限浮点；非有限/非法一律 ``None``。"""
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result):
        return None
    return round(result, ndigits)


# --------------------------------------------------------------------------- #
# 相位集
# --------------------------------------------------------------------------- #
def random_phase(grid: int, rms: float, seed: int) -> np.ndarray:
    """标准正态噪声缩放到**精确** RMS = ``rms``（种子固定 ⇒ 完全可复现）。"""
    rng = np.random.default_rng(int(seed))
    raw = rng.standard_normal((int(grid), int(grid)))
    std = float(raw.std())
    if std <= 0:
        raise ValueError("随机相位退化（std=0），无法归一化 RMS")
    return (raw * (float(rms) / std)).astype(np.float32)


def noll_phase(grid: int, nm: tuple[int, int], amplitude: float) -> np.ndarray:
    """单个 Noll 多项式（``amplitude`` 为 Noll 归一化系数 = 圆内 RMS 弧度）。

    :func:`generate_noll_polynomial` 在单位圆外返回 ``NaN``；SLM 上圆外命令 0 相位，
    故此处显式置 0（并在 plan.json 里记录该约定）。
    """
    n, m = int(nm[0]), int(nm[1])
    raw = generate_noll_polynomial(n, m, (int(grid), int(grid)), float(amplitude))
    return np.nan_to_num(
        np.asarray(raw, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )


def load_anchored_phase(path: Path, grid: int) -> np.ndarray:
    """读入锚定运行保存的最优控制网格，并校验形状与模型 ``grid_size`` 一致。"""
    phase = np.load(path)
    if phase.ndim != 2 or phase.shape != (int(grid), int(grid)):
        raise ValueError(
            f"锚定相位形状 {phase.shape} 与检查点 grid_size={grid} 不符: {path}"
        )
    return np.nan_to_num(
        np.asarray(phase, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )


def build_patterns(grid: int) -> list[Pattern]:
    """按固定顺序构造确定性相位集（缺失的锚定网格跳过并记录）。"""
    patterns: list[Pattern] = [
        Pattern(
            pattern_id="zero",
            source="synthetic:zero",
            phase=np.zeros((int(grid), int(grid)), dtype=np.float32),
            note="零相位基准（同帧定心也用它）",
        )
    ]

    for rms, tag in ((0.5, "0p5"), (1.5, "1p5")):
        for seed in (0, 1):
            patterns.append(
                Pattern(
                    pattern_id=f"rand_rms{tag}_seed{seed}",
                    source=f"synthetic:random_rms{tag}_seed{seed}",
                    phase=random_phase(grid, rms, seed),
                    note=f"高斯随机相位，RMS 精确归一到 {rms} rad",
                )
            )

    for tag, run_dir in ANCHORED_RUNS:
        path = run_dir / ANCHORED_BEST_NPY
        if not path.is_file():
            logger.warning(f"[pattern] 锚定最优网格缺失，跳过 {tag}: {path}")
            continue
        patterns.append(
            Pattern(
                pattern_id=f"{tag}_best",
                source=f"file:{path.relative_to(ROOT).as_posix()}",
                phase=load_anchored_phase(path, grid),
                note=f"{tag} 锚定运行的最终最优控制网格",
            )
        )

    for axis, nm in (("x", NOLL_TILT_X), ("y", NOLL_TILT_Y)):
        for amplitude in (1.0, 3.0):
            patterns.append(
                Pattern(
                    pattern_id=f"tilt_{axis}_{int(amplitude)}rad",
                    source=f"synthetic:noll{2 if axis == 'x' else 3}_amp{amplitude:g}",
                    phase=noll_phase(grid, nm, amplitude),
                    note=f"纯 {axis} 倾斜（Noll {2 if axis == 'x' else 3}），"
                    f"圆内 RMS {amplitude:g} rad，圆外 0",
                )
            )

    patterns.append(
        Pattern(
            pattern_id="defocus_z4_1rad",
            source="synthetic:noll4_amp1",
            phase=noll_phase(grid, NOLL_DEFOCUS, 1.0),
            note="离焦型 Noll-4，圆内 RMS 1 rad，圆外 0",
        )
    )

    ids = [p.pattern_id for p in patterns]
    if len(set(ids)) != len(ids):
        raise ValueError(f"相位图样 id 重复: {ids}")
    return patterns


# --------------------------------------------------------------------------- #
# 模型 / 掩膜
# --------------------------------------------------------------------------- #
def build_forward(
    params: FrozenDefocusParameters,
    spot_xy: tuple[float, float],
    *,
    half_span_px: float,
    step_px: float,
) -> tuple[FrozenDefocusForward, np.ndarray, tuple[int, int]]:
    """以 ``spot_xy=(col,row)`` 为窗心重建采样窗并构造冻结前向。"""
    camera_xy = recentre_window(
        (float(spot_xy[0]), float(spot_xy[1])),
        half_span_px=float(half_span_px),
        step_px=float(step_px),
    )
    forward = FrozenDefocusForward(params, camera_xy, device="cpu")
    centre_model = forward.sample_grid_index(float(spot_xy[0]), float(spot_xy[1]))
    return forward, camera_xy, (int(centre_model[0]), int(centre_model[1]))


def predict_window(
    forward: FrozenDefocusForward, phase: np.ndarray, exposure_ms: float
) -> np.ndarray:
    """``forward.predict`` 在采样窗上的强度预测（形状与 ``camera_xy[..., 0]`` 同）。"""
    torch = _torch()
    with torch.no_grad():
        predicted = forward.predict(
            torch.as_tensor(np.asarray(phase, dtype=np.float32)[None]),
            exposure_ms=float(exposure_ms),
        )
    return predicted[0].detach().cpu().numpy().astype(np.float32)


def masked_mean(image: np.ndarray, core: np.ndarray) -> float:
    """核心掩膜内的平均强度（掩膜空/退化时返回 ``nan``）。"""
    mask = np.asarray(core, dtype=bool)
    values = np.asarray(image, dtype=np.float64)[mask]
    if values.size == 0:
        return float("nan")
    return float(np.mean(values))


# --------------------------------------------------------------------------- #
# CLI / plan
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="绝对强度保真度探针：模型预测 vs 实测 CCD（固定确定性相位集）"
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="打开硬件执行")
    parser.add_argument(
        "--wait-until-idle",
        type=float,
        default=0.0,
        help="开设备前等待光路空闲的最长秒数（0=不等待）",
    )
    parser.add_argument("--shape", default="square", help="核心区光栅形状")
    parser.add_argument("--side", type=float, default=DEFAULT_SIDE, help="核心形状边长（CCD px）")
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES, help="每次抓取平均的帧数")
    parser.add_argument(
        "--exposure-ms",
        type=float,
        default=DEFAULT_EXPOSURE_MS,
        help="CCD 曝光（与两次锚定运行一致，默认 0.5）",
    )
    parser.add_argument(
        "--bit-depth",
        type=int,
        default=12,
        choices=(8, 12, 16),
        help="CCD 输出位深（默认 12；8-bit 会饱和零级、截断整形光，见 8-bit 证据）",
    )
    parser.add_argument(
        "--auto-exposure",
        action="store_true",
        help="打开设备后用 auto_exposure 把峰值压到目标灰度，避免零级饱和",
    )
    parser.add_argument(
        "--auto-exposure-target",
        type=float,
        default=180.0,
        help="auto_exposure 目标峰值（0-255 灰度，默认 180，约 12-bit 满量程七成）",
    )
    parser.add_argument(
        "--settle-s", type=float, default=DEFAULT_SETTLE_S, help="显示相位后的静置秒数"
    )
    parser.add_argument(
        "--window-half-span-px",
        type=float,
        default=DEFAULT_WINDOW_HALF_SPAN_PX,
        help="模型采样窗单边跨度（CCD px）",
    )
    parser.add_argument(
        "--spot-x",
        type=float,
        default=DEFAULT_SPOT_XY[0],
        help="离线时的光斑 col（默认取 anchored_m3_20261116/plan.json 的 spot_xy）",
    )
    parser.add_argument(
        "--spot-y",
        type=float,
        default=DEFAULT_SPOT_XY[1],
        help="离线时的光斑 row（同上）",
    )
    parser.add_argument(
        "--spot-threshold",
        type=float,
        default=1e3,
        help="同帧定心的最低峰值；低于则沿用 --spot-x/--spot-y",
    )
    parser.add_argument("--slm-number", type=int, default=1)
    parser.add_argument("--cam-id", type=int, default=0)
    parser.add_argument("--camera", choices=("daheng",), default="daheng")
    parser.add_argument("--camera-width", type=int, default=None)
    parser.add_argument("--camera-height", type=int, default=None)
    return parser.parse_args(argv)


def wait_for_idle(seconds: float) -> None:
    """轮询：没有别的进程持有本光路设备才继续。

    逻辑同 :func:`shape_transition_timing_20261106.wait_for_idle`，唯一差别是"自身
    进程"的排除标记换成本脚本名——上游实现只排除 ``shape_transition_timing``，而本
    脚本的命令行里 ``--checkpoint`` 默认路径含 ``defocus``，会被上游的占用标记命中，
    导致自己等自己直到超时。
    """
    if seconds <= 0:
        return
    import subprocess

    marker_keys = (
        "hw_",
        "defocus",
        "differentiable",
        "diffraction_efficiency",
        "capture_",
        "multishape",
        "shape_transition_timing",
        "absolute_fidelity_probe",
    )
    deadline = time.time() + float(seconds)
    while time.time() < deadline:
        listing = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                "Select-Object -ExpandProperty CommandLine",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
        busy = [
            line
            for line in listing.splitlines()
            if any(key in line for key in marker_keys)
            and "absolute_fidelity_probe" not in line
        ]
        if not busy:
            return
        logger.info("[wait] 仍有进程占用设备，等待…")
        time.sleep(15)
    logger.warning(f"[wait] {seconds}s 后仍有进程占用设备，继续（可能冲突）")


def build_plan(
    args: argparse.Namespace, params: FrozenDefocusParameters, patterns: list[Pattern]
) -> dict:
    """设备/相机打开之前的全部运行决定（离线也可写）。"""
    sensor = (
        (args.camera_width, args.camera_height)
        if (args.camera_width and args.camera_height)
        else CAMERA_SENSOR_PX[args.camera]
    )
    return dict(
        script=str(Path(__file__).resolve()),
        purpose="absolute intensity fidelity: frozen-model prediction vs measured CCD frame",
        issues=[221, 234, 236],
        timestamp=clock()["timestamp_utc"],
        checkpoint=str(args.checkpoint),
        checkpoint_sha256=sha256_file(Path(args.checkpoint)),
        out=str(args.out),
        model=dict(
            grid_size=int(params.grid_size),
            fft_size=int(params.fft_size),
            pixel_samples=int(params.pixel_samples),
            phase_sign=float(params.phase_sign),
            gain=float(params.gain),
            background=float(params.background),
            sigma_ccd_px=float(params.sigma_ccd_px),
            defocus_diopter=float(params.defocus_diopter),
        ),
        pattern_ids=[p.pattern_id for p in patterns],
        patterns=[
            dict(
                pattern_id=p.pattern_id,
                source=p.source,
                note=p.note,
                phase_shape=list(p.phase.shape),
                phase_rms_rad=_num(float(p.phase.std()), 6),
                phase_ptp_rad=_num(float(np.ptp(p.phase)), 6),
            )
            for p in patterns
        ],
        metrics=[
            "per-pattern: model_cv_core vs measured_cv_core_window (sampling-window core)",
            "per-pattern: mean_intensity_ratio_core = measured/model mean over core",
            "per-pattern: measured_cv_core_full (full-frame raster core, cross-check)",
        ],
        core=dict(shape=str(args.shape), side=float(args.side)),
        acquisition=dict(
            frames=int(args.frames),
            exposure_ms=float(args.exposure_ms),
            settle_s=float(args.settle_s),
            window_half_span_px=float(args.window_half_span_px),
            window_step_px=float(WINDOW_STEP_PX),
        ),
        centring=dict(
            spot_xy_offline=[float(args.spot_x), float(args.spot_y)],
            spot_xy_source="anchored_m3_20261116/plan.json:spot_xy",
            spot_threshold=float(args.spot_threshold),
            policy="same-frame: display zero -> grab -> argmax -> rebuild window+masks",
        ),
        execute=bool(args.execute),
        wait_until_idle=float(args.wait_until_idle),
        device=dict(camera=str(args.camera), cam_id=int(args.cam_id), slm_number=int(args.slm_number)),
        camera_sensor_px=[int(sensor[0]), int(sensor[1])],
        zernike_convention="generate_noll_polynomial, amplitude = in-aperture RMS rad, outside aperture = 0",
        pythonpath_note=r"hardware: $env:PYTHONPATH = 'D:\Projects\TIFO\AO-shaping\libs'",
    )


def build_summary(plan: dict, *, mode: str, rows: list[dict], context: dict) -> dict:
    """``summary.json``：逐图样指标 + ``shape_transition_timing`` 格式的 device_log。"""
    return dict(plan, mode=mode, **context, patterns_metrics=rows)


# --------------------------------------------------------------------------- #
# 逐图样测量
# --------------------------------------------------------------------------- #
def probe_pattern(
    *,
    pattern: Pattern,
    forward: FrozenDefocusForward,
    camera_xy: np.ndarray,
    spot_xy: tuple[float, float],
    exposure_ms: float,
    core_model: np.ndarray,
    shape: str,
    side: float,
    measured_frame: np.ndarray | None,
    probe_start: float,
) -> tuple[np.ndarray, dict]:
    """算模型预测（始终）与实测指标（``measured_frame`` 非 None 时），返回 ``(预测, 行)``。"""
    predicted = predict_window(forward, pattern.phase, exposure_ms)
    model_core_image = np.maximum(predicted, 0.0)
    model_mean_core = masked_mean(model_core_image, core_model)

    row: dict[str, Any] = dict(
        pattern_id=pattern.pattern_id,
        source=pattern.source,
        note=pattern.note,
        phase_rms_rad=_num(float(pattern.phase.std()), 6),
        phase_ptp_rad=_num(float(np.ptp(pattern.phase)), 6),
        model_mean_core=_num(model_mean_core, 6),
        model_cv_core=_num(monitor_cv(model_core_image, core_model), 6),
        measured_mean_core_window=None,
        measured_cv_core_window=None,
        measured_mean_core_full=None,
        measured_cv_core_full=None,
        measured_frame_mean=None,
        mean_intensity_ratio_core=None,
        mean_intensity_ratio_window=None,
        seconds_from_probe_start=round(time.monotonic() - probe_start, 4),
        wall=clock(),
    )

    if measured_frame is None:
        return predicted, row

    frame = np.asarray(measured_frame, dtype=np.float32)
    measured_window = sample_ccd_to_window(frame, camera_xy)
    measured_mean_core_window = masked_mean(measured_window, core_model)

    _, core_full = full_frame_raster(
        shape, frame.shape, (float(spot_xy[1]), float(spot_xy[0])), float(side)
    )
    measured_mean_core_full = masked_mean(frame, core_full)

    row.update(
        measured_frame_shape=list(frame.shape),
        measured_mean_core_window=_num(measured_mean_core_window, 6),
        measured_cv_core_window=_num(monitor_cv(measured_window, core_model), 6),
        measured_mean_core_full=_num(measured_mean_core_full, 6),
        measured_cv_core_full=_num(monitor_cv(frame, core_full), 6),
        measured_frame_mean=_num(float(frame.mean()), 6),
        mean_intensity_ratio_core=_num(
            measured_mean_core_window / model_mean_core if model_mean_core else float("nan"), 6
        ),
        mean_intensity_ratio_window=_num(
            float(measured_window.mean()) / float(predicted.mean())
            if float(predicted.mean())
            else float("nan"),
            6
        ),
    )
    return predicted, row


def run_probe(
    args: argparse.Namespace,
    params: FrozenDefocusParameters,
    patterns: list[Pattern],
    out: Path,
    *,
    spot_xy: tuple[float, float],
    spot_source: str,
    slm: Any = None,
    cam: Any = None,
    exposure_ms: float | None = None,
    device_log: list[dict] | None = None,
) -> tuple[list[dict], dict]:
    """构图样→（可选）上屏+抓帧→逐图样算指标，落盘 ``phase/``、``model/``、``frames/``。"""
    used_exposure = float(exposure_ms if exposure_ms is not None else args.exposure_ms)
    forward, camera_xy, centre_model = build_forward(
        params,
        spot_xy,
        half_span_px=float(args.window_half_span_px),
        step_px=float(WINDOW_STEP_PX),
    )
    target_model = target_at(
        camera_xy, str(args.shape), spot_xy, float(args.side)
    ).astype(np.float32)
    core_model = ShapeObjective(target_model).core

    (out / "phase").mkdir(parents=True, exist_ok=True)
    (out / "model").mkdir(parents=True, exist_ok=True)
    (out / "frames").mkdir(parents=True, exist_ok=True)

    probe_start = time.monotonic()
    rows: list[dict] = []
    for index, pattern in enumerate(patterns, start=1):
        np.save(out / "phase" / f"{pattern.pattern_id}.npy", pattern.phase)

        measured_frame: np.ndarray | None = None
        if slm is not None and cam is not None:
            display_phase(slm, pattern.phase, device_log=device_log)
            time.sleep(max(0.0, float(args.settle_s)))
            measured_frame = grab_average(
                cam, int(args.frames), height=args.camera_height
            )
            if device_log is not None:
                device_log.append(
                    dict(
                        action="capture",
                        pattern_id=pattern.pattern_id,
                        index=index,
                        exposure_ms=used_exposure,
                        frames=int(args.frames),
                        **clock(),
                    )
                )
            np.save(
                out / "frames" / f"{pattern.pattern_id}.npy",
                np.asarray(measured_frame, dtype=np.float32),
            )

        predicted, row = probe_pattern(
            pattern=pattern,
            forward=forward,
            camera_xy=camera_xy,
            spot_xy=spot_xy,
            exposure_ms=used_exposure,
            core_model=core_model,
            shape=str(args.shape),
            side=float(args.side),
            measured_frame=measured_frame,
            probe_start=probe_start,
        )
        np.save(out / "model" / f"{pattern.pattern_id}.npy", predicted)
        rows.append(row)
        logger.info(
            f"[probe {index}/{len(patterns)}] {pattern.pattern_id} · "
            f"model_cv={row['model_cv_core']} measured_cv={row['measured_cv_core_window']} "
            f"ratio_core={row['mean_intensity_ratio_core']}"
        )

    context = dict(
        exposure_ms=used_exposure,
        frames=int(args.frames),
        settle_s=float(args.settle_s),
        shape=str(args.shape),
        side=float(args.side),
        spot_xy=[float(spot_xy[0]), float(spot_xy[1])],
        spot_xy_source=spot_source,
        centre_model=[int(centre_model[0]), int(centre_model[1])],
        window_half_span_px=float(args.window_half_span_px),
        window_step_px=float(WINDOW_STEP_PX),
        window_shape=list(camera_xy[..., 0].shape),
        core_model_pixels=int(np.count_nonzero(core_model)),
        model=dict(
            grid_size=int(params.grid_size),
            fft_size=int(params.fft_size),
            phase_sign=float(params.phase_sign),
        ),
        probe_seconds=round(time.monotonic() - probe_start, 4),
        device_log=device_log if device_log is not None else [],
    )
    return rows, context


def _frame_full_scale(cam: Any, frame: np.ndarray) -> float:
    """定心帧的满量程, 按**生效位深**而非 ``frame.dtype``。

    12-bit 帧的 ``dtype`` 是 ``uint16``, 按 dtype 会得到 65535, 而真实满量程是
    4095 —— 与 :class:`DahengCamera` 的 ``_full_scale`` 同口径。位深不可得时回落
    ``common.full_scale`` (它认 ``AveragedFrame.full_scale`` 与整数 dtype)。
    """
    bits = getattr(cam, "bit_depth", None)
    if bits:
        return float((1 << int(bits)) - 1)
    from ao_shaping.drivers.ccd.common import full_scale

    return float(full_scale(frame))


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # ``frames/`` must exist before the execute path's same-frame centring save,
    # which runs prior to ``run_probe`` (the function that otherwise creates it).
    (out / "frames").mkdir(parents=True, exist_ok=True)

    params = load_frozen_parameters(args.checkpoint)
    grid = int(params.grid_size)
    patterns = build_patterns(grid)
    plan = build_plan(args, params, patterns)
    (out / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    logger.info(
        f"[plan] {len(patterns)} 个图样 · grid={grid} fft={params.fft_size} "
        f"phase_sign={params.phase_sign} exposure={args.exposure_ms}ms → {out / 'plan.json'}"
    )

    if not args.execute:
        rows, context = run_probe(
            args,
            params,
            patterns,
            out,
            spot_xy=(float(args.spot_x), float(args.spot_y)),
            spot_source="cli/offline-default (anchored_m3_20261116/plan.json:spot_xy)",
        )
        summary = build_summary(plan, mode="dry-run", rows=rows, context=context)
        (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("skip: requires --execute")
        logger.info(
            f"[dry-run] 模型预测与 plan/summary 已写入 {out}；未打开任何设备（no --execute）"
        )
        return 0

    wait_for_idle(float(args.wait_until_idle))

    device_log: list[dict] = []
    slm = cam = None
    exit_code = 1
    try:
        cam, slm, exposure_ms = open_devices(args, centre_ij=None, device_log=device_log)
        sensor_w = int(getattr(cam, "cam_width", 0) or 0)
        sensor_h = int(getattr(cam, "cam_height", 0) or 0)
        if sensor_w and sensor_h:
            plan["camera_sensor_live"] = [sensor_w, sensor_h]
            args.camera_width, args.camera_height = sensor_w, sensor_h

        # ---- 自动曝光：把峰值压到目标灰度，避免零级饱和摧毁动态范围 ----
        if args.auto_exposure:
            from ao_shaping.drivers.ccd.common import get_camera_exposure_ms

            note_ae = dict(action="auto_exposure", target_max=args.auto_exposure_target,
                           **clock())
            device_log.append(note_ae)
            logger.info(f"[device] auto_exposure target_max={args.auto_exposure_target}")
            cam.auto_exposure(target_max=float(args.auto_exposure_target))
            exposure_ms = float(get_camera_exposure_ms(cam))
            plan["exposure_ms"] = exposure_ms
            plan["exposure_source"] = "auto_exposure"
            logger.info(f"[device] auto_exposure 生效曝光 {exposure_ms:.4f} ms")

        # ---- 同帧定心：显示零相位 → 抓帧 → 用该帧 argmax 重建采样窗与掩膜 ----
        display_phase(slm, np.zeros((grid, grid), dtype=np.float32), device_log=device_log)
        time.sleep(max(0.0, float(args.settle_s)))
        base_frame = grab_average(cam, int(args.frames), height=args.camera_height)
        device_log.append(
            dict(action="capture", tag="centring_zero", exposure_ms=exposure_ms, **clock())
        )
        base_frame = np.asarray(base_frame, dtype=np.float32)
        np.save(out / "frames" / "_centring_zero.npy", base_frame)
        spot_xy = (float(args.spot_x), float(args.spot_y))
        spot_source = "cli (same-frame centring below threshold)"
        peak = float(base_frame.max()) if base_frame.size else float("nan")
        full = _frame_full_scale(cam, base_frame)
        sat_frac = peak / full if full > 0 and np.isfinite(peak) else float("nan")
        if np.isfinite(sat_frac) and sat_frac >= CENTRING_SATURATION_FRAC:
            # 饱和帧上 argmax 取的是首个饱和像素 (顶边), 不是光斑中心 —— 静默
            # 定心会给出错误窗心, 故这里直接报错 (设备由 finally 释放)。
            raise RuntimeError(
                f"[centring] 定心帧饱和 peak={peak:.0f}/{full:.0f}"
                f" ({sat_frac:.1%}) —— argmax 会落到首个饱和像素而非光斑中心, "
                f"定心不可靠。请降低曝光 (--auto-exposure 或更小 --exposure-ms) 后重跑。"
            )
        if np.isfinite(base_frame).all() and peak > float(args.spot_threshold):
            row_index, col_index = np.unravel_index(int(np.argmax(base_frame)), base_frame.shape)
            spot_xy = (float(col_index), float(row_index))
            spot_source = f"same-frame-centring (peak={peak:.1f}/{full:.0f})"
        plan["spot_xy"] = [spot_xy[0], spot_xy[1]]
        plan["spot_xy_source"] = spot_source
        logger.info(f"[centring] spot_xy={spot_xy} 来源={spot_source}")

        rows, context = run_probe(
            args,
            params,
            patterns,
            out,
            spot_xy=spot_xy,
            spot_source=spot_source,
            slm=slm,
            cam=cam,
            exposure_ms=exposure_ms,
            device_log=device_log,
        )
        summary = build_summary(plan, mode="execute", rows=rows, context=context)
        (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        plan["complete"] = True
        exit_code = 0
        logger.info(f"[probe] 完成 {len(rows)} 个图样 → {out / 'summary.json'}")
    except Exception as exc:
        logger.error(f"[probe] 失败: {exc!r}")
        exit_code = 1
        raise
    finally:
        close_devices(
            slm,
            cam,
            device_log,
            slm_number=int(args.slm_number),
            camera=str(args.camera),
        )
        (out / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
        (out / "device_log.json").write_text(
            json.dumps(device_log, indent=2, default=str), encoding="utf-8"
        )
        logger.info(
            f"[release] 设备已释放，device_log {len(device_log)} 条 → "
            f"{out / 'device_log.json'}"
        )
    logger.info(
        f"start={plan.get('timestamp')} end={datetime.now(timezone.utc).isoformat(timespec='microseconds')}"
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
