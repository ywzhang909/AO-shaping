"""固定目标保持 · 真设备验收采集（2026-11-23）。

为什么要有这个脚本
------------------
``b681033f`` 把"全程闭环目标保持"做进了 GUI：固定 CCD 像素参考、形状与位置分开
评分、有界修正、无效/陈旧帧冻结。但它在**实验系统上**还没有跑过一次 —— 页面能证明
代码路径没崩，证明不了零相位 8 帧够不够估噪声、推导出的阈值在这个光路上合不合
试、缓变漂移下固定的像素质心是否真被守住。

本脚本把同一条生产路径（``HardwareBench`` → ``ClosedLoopRunner`` → 
``TargetLockSupervisor``）搬到命令行，分两段采集，每段都留可复核的证据：

``calibrate``
    零相位、N 帧（默认 8），算 :class:`CalibrationSample` 并按固定倍数推导
    :class:`LockConfig`。饱和或信噪比不足时**直接失败**，不写出一份"看起来能用"
    的阈值。原始帧落盘，公式可以事后用同一段帧重算。

``hold``
    目标保持。目标默认是分析窗口上的矩形；也可以给一份**已保存的实测帧**
    （``--target-from-frame``）当作"要守住的最终形状"。相位候选来自 transfer
    检查点（frozen defocus 物理预测模型），可选 ``--source-dataset`` 打开在线适配：
    只有适配被晋级，物理预测模型才会跟着漂移走，闭环才可能把漂移补回来。

可选漂移注入（``--drift tilt-x|tilt-y``）
    用 SLM 自身叠一条缓慢张开的线性相位斜坡，把焦斑在 CCD 上推走 —— 这是
    **物理上真实发生的光斑移动**，又比推镜子可复现。注入量按相位 rad 计，
    逐次写入线性张开；脚本同时记录每次写入的注入值和实测质心，因此
    "rad → CCD px" 的换算来自当天实测，不是模型假设。

设备纪律
--------
* 默认 **plan-only**：不 import 设备驱动、不建输出目录，只打印解析后的计划。
  只有显式 ``--execute`` 才打开 SLM/CCD。
* ``--execute`` 时 SLM + CCD 各打开**一次**，``try/finally`` 释放；标定与保持
  在同一进程内共用一次打开，避免背对背 open/close 触发驱动板假死。
* ``--wait-until-idle``（默认 300 s；0 = 同时关闭等待与加锁）在开设备前走
  :mod:`ao_shaping.tools.kernels.device_mutex`，并在整个采集期间持有设备锁。
* 实验前先按 ``libs/skills/ao-optical-path-calibration/SKILL.md`` 跑
  ``scripts/slm/probe_optical_path_validity.py``；本脚本不重复那六步探测，
  只要求探测判 ``ok`` 之后再跑。

用法::

    # 离线计划（不碰硬件、不建目录）
    python scripts/slm/target_lock_ondevice_20261123.py --out /tmp/lock \\
        --checkpoint data/debug/slm_transfer/20261104_192122

    # 标定 + 保持（一次打开设备）
    python scripts/slm/target_lock_ondevice_20261123.py --execute --stage both \\
        --out data/debug/target_lock/20261123 --checkpoint data/debug/slm_transfer/20261104_192122 \\
        --exposure-ms 20 --drift tilt-x --drift-rad 2.0 --wait-until-idle 300

退出码
------
``0`` 采集完成；``2`` 命令行/计划无效（含 plan-only 的输入错误）；``3`` 设备或
帧质量让本次结论不可用（饱和、信噪比不足、门禁失败、保持提前冻结）。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import threading
import time
from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:  # 直跑脚本时也能 import 本包
    sys.path.insert(0, str(_REPO_ROOT / "src"))

#: ``LockConfig`` 的默认位置容差，仅用于"没标定就跑"的告警里做对照。
DEFAULT_HOLD_TOLERANCE_CCD_PX = 5.0


# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #


def _json_default(value: Any) -> Any:
    """把 numpy / dataclass / Path 统一成可写 JSON 的值。"""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        item = value.item()
        if isinstance(item, float) and not math.isfinite(item):
            return None
        return item
    if is_dataclass(value):
        return {field.name: getattr(value, field.name) for field in fields(value)}
    return str(value)


def _finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _print_plan(plan: dict[str, Any]) -> None:
    print(json.dumps(plan, ensure_ascii=False, indent=2, default=_json_default))


# --------------------------------------------------------------------------- #
# 检查点解析：中心、窗口、控制网格都从冻结参数里取，避免手抄错
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CheckpointSpec:
    """:class:`OnlineTransferAdapter` 需要的东西 + 分析窗口的默认参数。"""

    session_dir: Path
    params_path: Path
    span: int
    centre: tuple[float, float]
    grid_size: int
    origin_spacing_note: str = ""


def _find_params(path: Path) -> Path:
    """接受 会话目录 / fit 目录 / frozen_parameters.json 三种写法。"""
    if path.is_file():
        if path.suffix.lower() != ".json":
            raise ValueError(f"--checkpoint 指向的文件应为 JSON：{path}")
        return path
    for candidate in (
        path / "frozen_parameters.json",
        path / "fit" / "frozen_parameters.json",
        path / "initial" / "frozen_parameters.json",
    ):
        if candidate.is_file():
            return candidate
    raise ValueError(
        f"{path} 下没有 frozen_parameters.json（试过 ./fit 与 ./initial）"
    )


def resolve_checkpoint(path: Path, *, span: int | None) -> CheckpointSpec:
    params_path = _find_params(Path(path).expanduser())
    params = json.loads(params_path.read_text(encoding="utf-8"))
    try:
        origin = params["origin"]
        grid_size = int(params["grid_size"])
    except KeyError as exc:  # 冻结参数缺字段 = 不是本脚本认得的检查点
        raise ValueError(f"{params_path} 缺少字段 {exc}") from exc

    resolved_span = span
    note = ""
    if resolved_span is None:
        provenance = params_path.parent / "provenance.json"
        if provenance.is_file():
            data = json.loads(provenance.read_text(encoding="utf-8"))
            window_span = data.get("window_span_px")
            if window_span is not None:
                resolved_span = int(window_span)
                note = "span 取自 provenance.window_span_px"
        if resolved_span is None:
            resolved_span = 120
            note = "provenance 没有 window_span_px，退回 span=120（必须与训练窗口一致）"
    if int(resolved_span) < 2 or int(resolved_span) % 2:
        raise ValueError("span 必须是 >= 2 的偶数（分析网格步长 2 px）")

    session_dir = params_path.parent
    if session_dir.name in {"fit", "initial"}:
        session_dir = session_dir.parent
    return CheckpointSpec(
        session_dir=session_dir,
        params_path=params_path,
        span=int(resolved_span),
        centre=(float(origin[0]), float(origin[1])),
        grid_size=grid_size,
        origin_spacing_note=note,
    )


# --------------------------------------------------------------------------- #
# 目标与相位
# --------------------------------------------------------------------------- #


def _parse_centre(text: str, fallback: tuple[float, float]) -> tuple[float, float]:
    if text.strip().lower() in {"", "auto"}:
        return fallback
    parts = [piece for piece in text.replace(";", ",").split(",") if piece.strip()]
    if len(parts) != 2:
        raise ValueError("--centre 需写成 'x,y'（CCD 像素）")
    return float(parts[0]), float(parts[1])


def _load_array(path: Path) -> np.ndarray:
    """读 .npy / .npz；npz 取第一个数组（也认键名 frame/target）。"""
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.asarray(np.load(path))
    if suffix == ".npz":
        with np.load(path) as data:
            for key in ("frame", "target", "data"):
                if key in data:
                    return np.asarray(data[key])
            first = next(iter(data.files), None)
            if first is None:
                raise ValueError(f"{path} 里没有数组")
            return np.asarray(data[first])
    raise ValueError(f"只支持 .npy / .npz：{path}")


def build_target(
    args: argparse.Namespace,
    *,
    xy: np.ndarray,
) -> np.ndarray:
    """``field`` 矩形 / ``npy`` 网格目标 / ``frame`` 已保存实测帧。"""
    from ao_shaping.gui.slm.closed_loop_shaping import sample_on_grid, target_field

    mode = args.target
    if mode == "field":
        return target_field(
            xy,
            centre_xy=(float(xy[..., 0].mean()), float(xy[..., 1].mean())),
            half_width=float(args.half_width),
            half_height=float(args.half_height),
        )
    if mode == "npy":
        if args.target_path is None:
            raise ValueError("--target npy 需要 --target-path")
        target = _load_array(Path(args.target_path).expanduser()).astype(np.float64)
        if target.shape != xy.shape[:2]:
            raise ValueError(
                f"--target-path 形状 {target.shape} 与分析网格 {xy.shape[:2]} 不一致"
            )
        return target
    if mode == "frame":
        if args.target_path is None:
            raise ValueError("--target frame 需要 --target-path（已保存的 CCD 帧）")
        frame = _load_array(Path(args.target_path).expanduser()).astype(np.float64)
        if frame.ndim != 2:
            raise ValueError(f"目标帧必须是二维图，实际 {frame.shape}")
        sampled = sample_on_grid(frame, xy)
        # 采样后的目标统一归一化到 0/1，仅用于形状比较（绝对能量另由窗口占比报告）。
        peak = float(np.max(np.abs(sampled)))
        if not np.isfinite(peak) or peak <= 0.0:
            raise ValueError("目标帧在分析窗口内没有正信号")
        return sampled / peak
    raise ValueError(f"未知 --target：{mode}")


def build_start_phase(args: argparse.Namespace, spec: CheckpointSpec | None) -> np.ndarray:
    if args.start_phase is not None:
        phase = _load_array(Path(args.start_phase).expanduser()).astype(np.float64)
        if phase.ndim != 2:
            raise ValueError(f"起始相位必须是二维，实际 {phase.shape}")
        return phase
    grid_size = int(args.grid_size or (spec.grid_size if spec else 128))
    return np.zeros((grid_size, grid_size), dtype=np.float64)


# --------------------------------------------------------------------------- #
# 漂移注入：同一条 Bench 接口，只在写入前叠一条缓慢张开的相位斜坡
# --------------------------------------------------------------------------- #


class DriftBench:
    """包一层真实 Bench，按写入次数把线性相位斜坡叠进相位。

    为什么用相位斜坡而不是推镜子：**光斑确实在 CCD 上被推走**（物理效应），
    但幅度、起止与复现性都在软件控制内。注入量以"控制网格上的峰谷 rad"计，
    逐次线性张开；:class:`TargetLockSupervisor` 看到的是同一个固定像素质心，
    所以它必须真的把漂移补回来才算守住。
    """

    def __init__(
        self,
        inner: Any,
        *,
        axis: str,
        total_rad: float,
        start_write: int,
        ramp_writes: int,
        grid_size: int,
    ) -> None:
        if axis not in {"tilt-x", "tilt-y"}:
            raise ValueError(f"未知漂移轴：{axis}")
        if not math.isfinite(total_rad):
            raise ValueError("--drift-rad 必须是有限值")
        if start_write < 0 or ramp_writes < 1:
            raise ValueError("--drift-start-write >= 0 且 --drift-ramp-writes >= 1")
        self._inner = inner
        self.axis = axis
        self.total_rad = float(total_rad)
        self.start_write = int(start_write)
        self.ramp_writes = int(ramp_writes)
        self.grid_size = int(grid_size)
        self.write_index = 0
        self.trace: list[dict[str, Any]] = []

    # -- Bench 接口 -------------------------------------------------------- #
    def open(self) -> None:
        self._inner.open()

    def describe(self) -> dict[str, Any]:
        info = dict(self._inner.describe())
        info["drift"] = {
            "axis": self.axis,
            "total_rad": self.total_rad,
            "start_write": self.start_write,
            "ramp_writes": self.ramp_writes,
        }
        return info

    def capture(self) -> Any:
        return self._inner.capture()

    def close(self) -> None:
        self._inner.close()

    # -- 注入 -------------------------------------------------------------- #
    def injected_peak_to_peak(self, write_index: int) -> float:
        progress = (write_index - self.start_write) / float(self.ramp_writes)
        return self.total_rad * float(np.clip(progress, 0.0, 1.0))

    def tilt(self, write_index: int) -> np.ndarray:
        amplitude = self.injected_peak_to_peak(write_index)
        if amplitude == 0.0:
            return np.zeros((self.grid_size, self.grid_size), dtype=np.float64)
        axis = np.arange(self.grid_size, dtype=np.float64) / max(self.grid_size - 1, 1)
        ramp = amplitude * axis
        if self.axis == "tilt-x":
            return np.broadcast_to(ramp[None, :], (self.grid_size, self.grid_size)).copy()
        return np.broadcast_to(ramp[:, None], (self.grid_size, self.grid_size)).copy()

    def apply_phase(self, phase: np.ndarray) -> str:
        values = np.asarray(phase, dtype=np.float64)
        if values.shape != (self.grid_size, self.grid_size):
            raise ValueError(
                f"相位形状 {values.shape} 与注入网格 {(self.grid_size, self.grid_size)} 不一致"
            )
        index = self.write_index
        tilt = self.tilt(index)
        self.write_index += 1
        self.trace.append(
            {
                "write_index": index,
                "peak_to_peak_rad": float(tilt.max() - tilt.min()),
                "axis": self.axis,
            }
        )
        return self._inner.apply_phase(values + tilt)


# --------------------------------------------------------------------------- #
# Bench 工厂与"已打开"包装
# --------------------------------------------------------------------------- #


def make_hardware_bench(args: argparse.Namespace) -> Any:
    """裸 HardwareBench：标定用它，保证零相位采集不掺漂移。"""
    from ao_shaping.gui.slm.closed_loop_bench import HardwareBench

    return HardwareBench(
        slm_number=args.slm_number,
        wavelength=args.wavelength,
        camera_type=args.camera,
        cam_id=args.cam_id,
        exposure_ms=args.exposure_ms,
        settle_s=args.settle_s,
        discard_frames=args.discard_frames,
    )


def make_bench(args: argparse.Namespace, *, grid_size: int, base: Any | None = None) -> Any:
    """保持阶段的 bench：``--drift none`` 时不包漂移层。

    ``base`` 给 ``--stage both`` 复用已打开的裸 bench，避免背对背 open/close。
    """
    inner = base if base is not None else make_hardware_bench(args)
    if args.drift == "none":
        return inner
    return DriftBench(
        inner,
        axis=args.drift,
        total_rad=args.drift_rad,
        start_write=args.drift_start_write,
        ramp_writes=args.drift_ramp_writes,
        grid_size=grid_size,
    )


class PreopenedBench:
    """把"已经打开的 bench"喂给 runner：``open`` 变 no-op，``close`` 照常释放。

    ``ClosedLoopRunner.run`` 固定会调一次 ``open``；``--stage both`` 时标定已经
    打开过设备，直接复用会触发 ``HardwareBench already opened``。这层包装让
    两个阶段真正共用同一次设备打开，而不是连续开关驱动板。
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def open(self) -> None:
        return None

    def describe(self) -> dict[str, Any]:
        return self._inner.describe()

    def apply_phase(self, phase: np.ndarray) -> str:
        return self._inner.apply_phase(phase)

    def capture(self) -> Any:
        return self._inner.capture()

    def close(self) -> None:
        self._inner.close()

    def __getattr__(self, name: str) -> Any:  # trace 等附加信息透传
        return getattr(self._inner, name)


# --------------------------------------------------------------------------- #
# 阶段 1：标定
# --------------------------------------------------------------------------- #


def run_calibration(
    args: argparse.Namespace,
    *,
    bench: Any,
    xy: np.ndarray,
    target: np.ndarray,
    grid_size: int,
    out: Path,
) -> tuple[Any, Any, dict[str, Any]]:
    """零相位 N 帧 → CalibrationSample → LockConfig；返回 (sample, config, 证据)。"""
    from ao_shaping.gui.slm.closed_loop_shaping import grid_spacing_px
    from ao_shaping.gui.slm.target_lock import (
        calibrate_lock_thresholds,
        calibration_stats_from_frames,
        target_on_frame,
    )

    frames: list[np.ndarray] = []          # 原始 dtype，落盘不丢 12-bit 精度
    frames_float: list[np.ndarray] = []    # 统一 float，统计入口用
    captured_meta: list[dict[str, Any]] = []
    t0 = time.monotonic()
    for index in range(args.frames):
        phase = np.zeros((grid_size, grid_size), dtype=np.float64)
        bench.apply_phase(phase)
        captured = bench.capture()
        frames.append(np.asarray(captured.frame))
        frames_float.append(np.asarray(captured.frame, dtype=np.float64))
        captured_meta.append(
            {
                "frame_index": index,
                "frame_id": int(captured.frame_id),
                "exposure_ms": float(captured.exposure_ms),
                "full_scale": float(captured.full_scale),
                "monotonic_ns": int(captured.monotonic_ns),
                "peak_counts": float(np.max(captured.frame)),
            }
        )
    elapsed = time.monotonic() - t0

    target_frame = target_on_frame(xy, target, frames[0].shape)
    sample = calibration_stats_from_frames(
        frames_float, target=target_frame, full_scale=float(captured_meta[0]["full_scale"])
    )
    spacing = grid_spacing_px(xy)
    if spacing is None:
        raise RuntimeError("分析网格没有正的有限步长，无法把位置容差换算成网格单位")
    config = calibrate_lock_thresholds(
        sample,
        position_tolerance_ccd_px=float(args.hold_tolerance),
        grid_spacing_px=spacing,
    )

    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "calibration_frames.npz",
        frames=np.asarray(frames),  # 保留原始 dtype，不把 12-bit/浮点帧截成 uint16
        frame_ids=np.asarray([item["frame_id"] for item in captured_meta], dtype=np.int64),
        exposure_ms=np.asarray([item["exposure_ms"] for item in captured_meta]),
    )
    evidence = {
        "created_at": _utc_now(),
        "phase": "zero",
        "frame_count": len(frames),
        "elapsed_s": round(elapsed, 3),
        "frames": captured_meta,
        "sample": asdict(sample),
        "grid_spacing_px": spacing,
        "position_tolerance_ccd_px": float(args.hold_tolerance),
        "lock_config": asdict(config),
    }
    _write_json(out / "calibration.json", evidence)

    # 帧质量是标定能不能用的前提：饱和与低信噪都必须让本次结论不可用。
    problems: list[str] = []
    if sample.saturated_fraction > max(0.01, 5.0 * args.saturation_limit):
        problems.append(
            f"饱和像素占比 {sample.saturated_fraction:.3%} 超过 1% —— 降低曝光后重采"
        )
    if sample.peak_counts <= sample.dark_mean + 10.0 * max(sample.dark_sigma, 1e-9):
        problems.append(
            f"峰值 {sample.peak_counts:.1f} 未明显高于暗场 "
            f"{sample.dark_mean:.1f}+10σ —— 提高曝光或检查光路"
        )
    if sample.dark_sigma <= 0.0:
        problems.append("暗场标准差为 0 —— 背景区被信号占满或相机在做数值裁剪")
    if problems:
        raise FrameQualityError("；".join(problems), evidence)
    return sample, config, evidence


class FrameQualityError(RuntimeError):
    """帧质量让标定/保持在物理上不可用（不是脚本 bug）。"""

    def __init__(self, message: str, evidence: dict[str, Any]) -> None:
        super().__init__(message)
        self.evidence = evidence


# --------------------------------------------------------------------------- #
# 阶段 2：保持
# --------------------------------------------------------------------------- #


def _round_to_dict(record: Any) -> dict[str, Any]:
    gate = record.gate
    return {
        "round_index": int(record.round_index),
        "phase_id": record.phase_id,
        "frame_id": int(record.frame_id),
        "captured_at_s": _finite_or_none(record.captured_at_s),
        "exposure_ms": float(record.exposure_ms),
        "gate_ok": bool(gate.ok),
        "gate_reasons": list(gate.reasons),
        "centroid_xy": list(gate.centroid_xy) if gate.centroid_xy else None,
        "peak_counts": _finite_or_none(gate.peak),
        "full_scale": _finite_or_none(gate.full_scale),
        "saturation_pixels": int(gate.saturation_pixels),
        "metrics": dict(record.metrics) if record.metrics else None,
        "measured_score": _finite_or_none(record.measured_score),
        "predicted_score": _finite_or_none(record.predicted_score),
        "phase_step_rms": _finite_or_none(record.phase_step_rms),
        "window_fraction": _finite_or_none(record.window_fraction),
        "decision": record.decision,
        "note": record.note,
    }


def _event_to_dict(event: Any) -> dict[str, Any]:
    payload = event.payload
    if isinstance(payload, dict):
        data = payload
    elif is_dataclass(payload):
        data = _round_to_dict(payload) if payload.__class__.__name__ == "RoundRecord" else asdict(payload)
    else:
        data = {"value": payload}
    return {"kind": event.kind, "payload": data}


def _summarise(rounds: list[dict[str, Any]], events: list[dict[str, Any]]) -> dict[str, Any]:
    """把逐轮记录压成可复核的统计；不做"通过/不通过"裁决。"""
    good = [row for row in rounds if row["gate_ok"]]
    centroids = [row["centroid_xy"] for row in good if row["centroid_xy"]]
    scores = [row["measured_score"] for row in good if row["measured_score"] is not None]
    locks = [event["payload"] for event in events if event["kind"] == "lock"]
    errors = [
        abs(float(item["centroid_error_ccd_px"]))
        for item in locks
        if item.get("centroid_error_ccd_px") is not None
        and math.isfinite(float(item["centroid_error_ccd_px"]))
    ]
    actions: dict[str, int] = {}
    for item in locks:
        action = str(item.get("action", "unknown"))
        actions[action] = actions.get(action, 0) + 1

    summary: dict[str, Any] = {
        "rounds": len(rounds),
        "gate_ok_rounds": len(good),
        "invalid_rounds": len(rounds) - len(good),
        "actions": actions,
        "centroid_first": centroids[0] if centroids else None,
        "centroid_last": centroids[-1] if centroids else None,
        "centroid_drift_px": (
            float(
                np.hypot(
                    centroids[-1][0] - centroids[0][0],
                    centroids[-1][1] - centroids[0][1],
                )
            )
            if len(centroids) >= 2
            else None
        ),
        "centroid_error_max_ccd_px": max(errors) if errors else None,
        "centroid_error_rms_ccd_px": (
            float(np.sqrt(np.mean(np.square(errors)))) if errors else None
        ),
        "score_first": scores[0] if scores else None,
        "score_last": scores[-1] if scores else None,
        "score_best": min(scores) if scores else None,
    }
    return summary


def run_hold(
    args: argparse.Namespace,
    *,
    spec: CheckpointSpec,
    xy: np.ndarray,
    target: np.ndarray,
    start_phase: np.ndarray,
    lock_config: Any,
    bench: Any,
    out: Path,
) -> dict[str, Any]:
    from ao_shaping.gui.slm.closed_loop_shaping import (
        ClosedLoopConfig,
        ClosedLoopRunner,
        OnlineTransferAdapter,
    )

    out.mkdir(parents=True, exist_ok=True)
    adapter = None
    forward = None
    if args.source_dataset is not None:
        (out / "adapt_session").mkdir(parents=True, exist_ok=True)
        adapter = OnlineTransferAdapter(
            checkpoint=spec.params_path,
            session_dir=out / "adapt_session",
            span=spec.span,
            source_dataset=args.source_dataset,
            device=args.device,
        )
        adapter_note = (
            f"适配已就绪：{len(adapter.rows)} 行种子"
            if adapter.seeded
            else f"适配未就绪：{adapter.seed_error}"
        )
    elif args.adapt:
        raise ValueError("--adapt 需要 --source-dataset（否则没有可用的标定行）")
    else:
        # 不适配也要有物理预测模型，否则 hold 在自检阶段就 FAULT。
        from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
            FrozenDefocusForward,
            load_frozen_parameters,
        )

        params = load_frozen_parameters(spec.params_path)
        forward = FrozenDefocusForward(params, xy, device=args.device)
        adapter_note = "未启用在线适配：使用冻结 defocus 物理预测模型（不随漂移重拟合）"

    config = ClosedLoopConfig(
        xy=xy,
        target=target,
        start_phase=start_phase,
        max_rounds=int(args.max_rounds),
        mode="hold",
        candidate_policy="auto",
        hold_target_ccd_px=float(args.hold_tolerance),
        lock_config=lock_config,
        adapt=bool(args.adapt),
        exposure_ms=float(args.exposure_ms),
        validation_every=int(args.validation_every),
    )

    rounds_jsonl = out / "rounds.jsonl"
    events_jsonl = out / "events.jsonl"
    rounds: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    raw_dir = out / "frames"
    lock = threading.Lock()

    def on_event(event: Any) -> None:
        row = _event_to_dict(event)
        with lock:
            events.append(row)
            with events_jsonl.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")
        if event.kind in {"state", "fault", "bench"}:
            print(f"[{event.kind}] {event.payload}", flush=True)

    def on_frame(captured: Any, record: Any) -> None:
        if args.save_raw_every > 0 and captured.frame_id % args.save_raw_every == 0:
            raw_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                raw_dir / f"frame_{captured.frame_id:05d}.npz",
                frame=np.asarray(captured.frame),
                exposure_ms=float(captured.exposure_ms),
                full_scale=float(captured.full_scale),
            )
        if record is not None:
            row = _round_to_dict(record)
            with lock:
                rounds.append(row)
                with rounds_jsonl.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")

    runner = ClosedLoopRunner(
        bench, config=config, forward=forward, adapter=adapter,
        on_event=on_event, on_frame=on_frame,
    )
    events_jsonl.write_text("", encoding="utf-8")
    rounds_jsonl.write_text("", encoding="utf-8")
    started = time.monotonic()
    try:
        summary = runner.run()
    except Exception as exc:  # 设备异常也要留下已采到的证据
        _write_json(
            out / "summary.json",
            {
                "state": "exception",
                "reason": str(exc),
                "adapter": adapter_note,
                "rounds": len(rounds),
                "elapsed_s": round(time.monotonic() - started, 3),
            },
        )
        raise

    elapsed = time.monotonic() - started
    stats = _summarise(rounds, events)
    drift_trace = getattr(bench, "trace", None)
    payload: dict[str, Any] = {
        "created_at": _utc_now(),
        "elapsed_s": round(elapsed, 3),
        "state": summary.state.value,
        "reason": summary.reason,
        "adapter": adapter_note,
        "best_score": _finite_or_none(summary.best_score),
        "applied_score": _finite_or_none(summary.applied_score),
        "config": {
            "max_rounds": int(args.max_rounds),
            "hold_target_ccd_px": float(args.hold_tolerance),
            "lock_config": asdict(lock_config) if lock_config is not None else None,
            "adapt": bool(args.adapt),
            "exposure_ms": float(args.exposure_ms),
        },
        "window": {
            "span": spec.span,
            "centre": [float(xy[..., 0].mean()), float(xy[..., 1].mean())],
            "grid_size": int(np.asarray(start_phase).shape[0]),
            "target_peak": float(np.max(target)),
            "target_fraction": float(np.mean(np.asarray(target) > 0.0)),
        },
        "drift": {
            "axis": args.drift,
            "total_rad": float(args.drift_rad),
            "start_write": int(args.drift_start_write),
            "ramp_writes": int(args.drift_ramp_writes),
            "writes": len(drift_trace) if drift_trace else 0,
            "final_injected_peak_to_peak_rad": (
                float(drift_trace[-1]["peak_to_peak_rad"]) if drift_trace else 0.0
            ),
        },
        "stats": stats,
        "verdict": "evidence-only：物理验收由操作员按 stats 与原始帧判断，脚本不代替验收",
    }
    _write_json(out / "summary.json", payload)

    with (out / "rounds.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "round", "frame_id", "gate_ok", "decision", "score",
                "centroid_x", "centroid_y", "peak_counts", "saturated_fraction",
                "phase_step_rms", "window_fraction",
            ]
        )
        for row in rounds:
            centroid = row["centroid_xy"] or [None, None]
            writer.writerow(
                [
                    row["round_index"], row["frame_id"], row["gate_ok"],
                    row["decision"], row["measured_score"],
                    centroid[0], centroid[1], row["peak_counts"],
                    row["saturated_fraction"], row["phase_step_rms"],
                    row["window_fraction"],
                ]
            )
    return payload


# --------------------------------------------------------------------------- #
# 命令行
# --------------------------------------------------------------------------- #


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="固定目标保持·实验验收（标定 + 保持）", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--out", type=Path, required=True, help="输出目录（plan-only 不创建）")
    parser.add_argument("--stage", choices=("calibrate", "hold", "both"), default="both")
    parser.add_argument("--execute", action="store_true", help="真正打开 SLM/CCD；缺省只打印计划")
    parser.add_argument(
        "--wait-until-idle", type=float, default=300.0,
        help="开设备前等待其他会话释放设备的秒数；0 = 不等待也不加锁",
    )
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="transfer 会话目录 / fit 目录 / frozen_parameters.json")
    parser.add_argument("--source-dataset", type=Path, default=None,
                        help="既有标定数据集（含 manifest.json），给了才可能启用在线适配")
    parser.add_argument("--adapt", action="store_true", help="在线适配（需要 --source-dataset）")
    parser.add_argument("--device", default=None, help="适配/物理预测模型的 torch 设备（默认由适配器决定）")

    # 窗口与目标
    parser.add_argument("--span", type=int, default=None, help="分析窗口半宽（CCD px，偶数）")
    parser.add_argument("--centre", default="auto", help="分析窗口中心 'x,y'；auto = 检查点 origin")
    parser.add_argument("--grid-size", type=int, default=None, help="SLM 控制网格边长；缺省取检查点")
    parser.add_argument("--target", choices=("field", "npy", "frame"), default="field")
    parser.add_argument("--target-path", type=Path, default=None, help="npy/frame 目标来源")
    parser.add_argument("--half-width", type=float, default=10.0, help="矩形目标半宽（CCD px）")
    parser.add_argument("--half-height", type=float, default=10.0, help="矩形目标半高（CCD px）")
    parser.add_argument("--start-phase", type=Path, default=None, help="起始相位 .npy（rad，控制网格）")

    # 设备参数
    parser.add_argument("--slm-number", type=int, default=1)
    parser.add_argument("--wavelength", type=int, default=1064)
    parser.add_argument("--camera", default="daheng")
    parser.add_argument("--cam-id", default=0)
    parser.add_argument("--exposure-ms", type=float, default=20.0)
    parser.add_argument("--settle-s", type=float, default=0.15)
    parser.add_argument("--discard-frames", type=int, default=3)

    # 标定
    parser.add_argument("--frames", type=int, default=8, help="标定帧数（零相位）")
    parser.add_argument("--hold-tolerance", type=float, default=DEFAULT_HOLD_TOLERANCE_CCD_PX,
                        help="位置容差（CCD px）")
    parser.add_argument("--saturation-limit", type=float, default=0.002,
                        help="标定允许的饱和占比下限（实际取 max(1%%, 5×该值)）")

    # 保持
    parser.add_argument("--max-rounds", type=int, default=40)
    parser.add_argument("--validation-every", type=int, default=0)
    parser.add_argument("--lock-config", type=Path, default=None,
                        help="calibration.json（只跑 hold 时用它复用已标定阈值）")
    parser.add_argument("--save-raw-every", type=int, default=0,
                        help="每 N 帧保存一张原始帧；0 = 不保存")

    # 漂移注入
    parser.add_argument("--drift", choices=("none", "tilt-x", "tilt-y"), default="none")
    parser.add_argument("--drift-rad", type=float, default=2.0,
                        help="注入相位斜坡的峰谷总幅（rad）")
    parser.add_argument("--drift-start-write", type=int, default=4)
    parser.add_argument("--drift-ramp-writes", type=int, default=20)
    return parser.parse_args(argv)


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    """把参数解析成可复核的计划；plan-only 走同一条路径，确保计划与执行一致。"""
    if args.frames < 2:
        raise ValueError("--frames 至少 2（否则估不了噪声与重复性）")
    if args.max_rounds < 1:
        raise ValueError("--max-rounds >= 1")
    if args.hold_tolerance <= 0 or not math.isfinite(args.hold_tolerance):
        raise ValueError("--hold-tolerance 必须是正的有限值")
    if args.exposure_ms <= 0 or not math.isfinite(args.exposure_ms):
        raise ValueError("--exposure-ms 必须是正的有限值")
    if args.drift != "none" and args.drift_rad < 0:
        raise ValueError("--drift-rad 不能为负")

    spec = None
    if args.checkpoint is not None:
        spec = resolve_checkpoint(args.checkpoint, span=args.span)
    elif args.stage in {"hold", "both"}:
        raise ValueError("hold/both 需要 --checkpoint（保持模式必须有物理预测模型）")

    grid_size = int(args.grid_size or (spec.grid_size if spec else 128))
    span = int(args.span or (spec.span if spec else 120))
    if span < 2 or span % 2:
        raise ValueError("--span 必须是 >= 2 的偶数")
    centre = _parse_centre(args.centre, spec.centre if spec else (0.0, 0.0))
    if spec is None and args.centre.strip().lower() in {"", "auto"}:
        raise ValueError("没有 --checkpoint 时必须显式 --centre x,y（CCD 像素）")
    if args.target == "field" and max(args.half_width, args.half_height) >= span:
        raise ValueError("--half-width/--half-height 必须小于 --span（窗口装不下目标）")
    if args.target in {"npy", "frame"}:
        if args.target_path is None:
            raise ValueError(f"--target {args.target} 需要 --target-path")
        if not Path(args.target_path).expanduser().is_file():
            raise ValueError(f"--target-path 不存在：{args.target_path}")
    if args.start_phase is not None and not Path(args.start_phase).expanduser().is_file():
        raise ValueError(f"--start-phase 不存在：{args.start_phase}")
    if args.lock_config is not None and not Path(args.lock_config).expanduser().is_file():
        raise ValueError(f"--lock-config 不存在：{args.lock_config}")
    if args.adapt and args.source_dataset is None:
        raise ValueError("--adapt 需要 --source-dataset（否则没有可用的标定行）")
    if args.source_dataset is not None and not Path(args.source_dataset).expanduser().is_dir():
        raise ValueError(f"--source-dataset 不存在：{args.source_dataset}")
    if args.source_dataset is not None and not (
        Path(args.source_dataset).expanduser() / "manifest.json"
    ).is_file():
        raise ValueError(f"--source-dataset 缺少 manifest.json：{args.source_dataset}")

    warnings: list[str] = []
    if args.stage == "hold" and args.lock_config is None:
        warnings.append(
            "只跑 hold 且没有 --lock-config：沿用默认 LockConfig（仅为 8/12-bit 相机起点）"
        )
    if args.drift != "none":
        if args.drift_rad > 2.0 * math.pi:
            warnings.append("--drift-rad 超过 2π，相位会被 LUT 环绕多次，换算不单调")
        if args.drift_start_write < 1:
            warnings.append("--drift-start-write < 1：基线帧之后立即开始漂移")
        if not args.adapt:
            warnings.append(
                "漂移测试建议 --adapt（需 --source-dataset）：否则物理预测模型不随漂移重拟合"
            )
    if spec is not None and spec.origin_spacing_note.startswith("provenance 没有"):
        warnings.append(spec.origin_spacing_note)

    return {
        "generated_at": _utc_now(),
        "execute": bool(args.execute),
        "stage": args.stage,
        "out": str(Path(args.out).expanduser()),
        "checkpoint": (
            None
            if spec is None
            else {
                "session_dir": str(spec.session_dir),
                "params": str(spec.params_path),
                "grid_size": spec.grid_size,
                "span_from_checkpoint": spec.span,
                "origin": list(spec.centre),
                "note": spec.origin_spacing_note,
            }
        ),
        "window": {"span": span, "centre": list(centre), "grid_size": grid_size},
        "target": {
            "mode": args.target,
            "path": str(args.target_path) if args.target_path else None,
            "half_width": args.half_width,
            "half_height": args.half_height,
        },
        "device": {
            "slm_number": args.slm_number,
            "wavelength": args.wavelength,
            "camera": f"{args.camera}:{args.cam_id}",
            "exposure_ms": args.exposure_ms,
            "settle_s": args.settle_s,
            "discard_frames": args.discard_frames,
            "wait_until_idle_s": args.wait_until_idle,
        },
        "calibration": {
            "frames": args.frames,
            "hold_tolerance_ccd_px": args.hold_tolerance,
            "saturation_limit": args.saturation_limit,
        },
        "hold": {
            "max_rounds": args.max_rounds,
            "adapt": bool(args.adapt),
            "source_dataset": str(args.source_dataset) if args.source_dataset else None,
            "validation_every": args.validation_every,
            "save_raw_every": args.save_raw_every,
        },
        "drift": {
            "axis": args.drift,
            "total_rad": args.drift_rad,
            "start_write": args.drift_start_write,
            "ramp_writes": args.drift_ramp_writes,
        },
        "warnings": warnings,
    }


def _load_lock_config(path: Path) -> Any:
    from ao_shaping.gui.slm.target_lock import LockConfig

    data = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    payload = data.get("lock_config", data)
    allowed = {field.name for field in fields(LockConfig)}
    missing = {"max_centroid_error_px"} - set(payload)
    if missing:
        raise ValueError(f"{path} 缺少 LockConfig 字段：{sorted(missing)}")
    return LockConfig(**{key: payload[key] for key in payload if key in allowed})


def _wait_and_lock(args: argparse.Namespace) -> tuple[Any, dict[str, Any], Any]:
    """``--wait-until-idle`` → 持锁；返回 (lock, 证据, 释放回调)。"""
    if args.wait_until_idle <= 0:
        return None, {"status": "skipped", "budget_s": 0.0}, lambda: None
    from ao_shaping.tools.kernels import device_mutex
    from ao_shaping.tools.kernels.device_mutex import DeviceLock

    verdict = device_mutex.wait_until_idle(
        args.wait_until_idle,
        slm_number=args.slm_number,
        cam_id=args.cam_id,
        cam_type=args.camera,
        own_paths=[Path(__file__).resolve()],
    )
    evidence = {
        "status": verdict.status,
        "budget_s": args.wait_until_idle,
        "waited_s": round(verdict.waited_s, 3),
        "holder_pid": verdict.holder.pid if verdict.holder is not None else None,
        "detail": verdict.detail,
        "remedy": verdict.remedy,
    }
    if not verdict.ok:
        raise SystemExit(
            f"device unavailable ({verdict.kind or verdict.status}): "
            f"{verdict.reason or verdict.detail or verdict.status} -- {verdict.remedy}"
        )
    lock = verdict.lock or DeviceLock.default()
    if verdict.lock is None and not lock.acquire(timeout=0.0):
        raise SystemExit("device lock just taken by another session; rerun")
    return lock, evidence, lock.release


def execute(args: argparse.Namespace, plan: dict[str, Any]) -> int:
    """唯一碰设备的路径：开一次、跑完、finally 释放。返回退出码。"""
    from loguru import logger

    from ao_shaping.gui.slm.closed_loop_shaping import analysis_grid

    out: Path = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)  # 只有 --execute 才建目录
    _write_json(out / "plan.json", plan)

    spec = None
    if args.checkpoint is not None:
        spec = resolve_checkpoint(args.checkpoint, span=args.span)
    window = plan["window"]
    xy = analysis_grid(tuple(window["centre"]), int(window["span"]))
    target = build_target(args, xy=xy)
    start_phase = build_start_phase(args, spec)
    grid_size = int(np.asarray(start_phase).shape[0])
    if spec is not None and grid_size != spec.grid_size:
        raise ValueError(
            f"起始相位网格 {grid_size} 与检查点 grid_size {spec.grid_size} 不一致；"
            "物理预测模型只认训练时的控制网格"
        )

    lock, lock_evidence, release = _wait_and_lock(args)
    plan["device_lock"] = lock_evidence
    _write_json(out / "plan.json", plan)

    lock_config = _load_lock_config(args.lock_config) if args.lock_config else None
    # 全流程只有一个 HardwareBench 实例：标定用裸 bench，保持在外面叠漂移层；
    # stage=both 时用 PreopenedBench 让 runner 的 open() 变成 no-op。
    base_bench = make_hardware_bench(args)
    exit_code = 0
    try:
        if args.stage in {"calibrate", "both"}:
            logger.info("打开 SLM + CCD（一次），阶段={}", args.stage)
            base_bench.open()
            try:
                sample, derived, evidence = run_calibration(
                    args, bench=base_bench, xy=xy, target=target,
                    grid_size=grid_size, out=out / "calibration",
                )
                logger.info("标定：{}", evidence["sample"])
                logger.info("推导阈值：{}", derived)
                lock_config = derived
            except FrameQualityError as exc:
                logger.error("标定帧质量不可用：{}", exc)
                print(f"标定失败：{exc}")
                exit_code = 3
            finally:
                if args.stage == "calibrate":
                    base_bench.close()

        if exit_code == 0 and args.stage in {"hold", "both"}:
            hold_bench = make_bench(args, grid_size=grid_size, base=base_bench)
            if args.stage == "both":
                # 标定与保持共用同一次设备打开，避免背对背 open/close 假死。
                hold_bench = PreopenedBench(hold_bench)
            payload = run_hold(
                args, spec=spec, xy=xy, target=target, start_phase=start_phase,
                lock_config=lock_config, bench=hold_bench, out=out / "hold",
            )
            print(
                f"保持结束：state={payload['state']} reason={payload['reason']} "
                f"rounds={payload['stats']['rounds']} "
                f"位置漂移={payload['stats']['centroid_drift_px']}"
            )
            if "fault" in str(payload["state"]).lower():
                exit_code = 3
    finally:
        try:
            base_bench.close()
        except Exception:  # 关闭失败不掩盖主结论
            logger.warning("关闭设备时出错（已忽略，保住主结论）")
        finally:
            release()
    return exit_code


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        plan = build_plan(args)
    except ValueError as exc:
        print(f"计划无效：{exc}", file=sys.stderr)
        return 2
    if not args.execute:
        _print_plan(plan)
        print("\nplan-only：未打开任何设备、未创建输出目录。加 --execute 才会采集。")
        return 0
    try:
        return execute(args, plan)
    except SystemExit:
        raise
    except Exception as exc:  # 设备/配置异常统一成退出码 3，保留堆栈便于追责
        import traceback

        traceback.print_exc()
        print(f"执行失败：{exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
