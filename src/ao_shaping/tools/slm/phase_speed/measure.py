"""相位变化速度测量的**硬件后端** (Santec SLM + 相机注册表).

分层
----
:mod:`~ao_shaping.tools.slm.phase_speed.core` 是纯算法层 (0 级指标 + 阶跃时序
解算), 本模块只负责"把图案写上面板、把远场帧取回来", 以及实验纪律: 内存模式
槽位轮换、raw uint16 平场 (严禁 ``create_phase_from_array``)、相机满量程饱和
检查、设备参数落盘.

测量序列
--------
1. 写图案 A (gray-a), 用 :class:`~ao_shaping.drivers.slm.santec.SlotRotator`
   轮换到一个**新**内存槽, ``wait_time_s=settle_s`` 让像素翻转;
2. 采集 ``baseline_window`` 帧基线;
3. 写图案 B (gray-b) 到**另一个**新槽, ``wait_time_s=0`` 立即开始采集;
4. 逐帧 ``get_numpy_image(1, skip_first=False)`` 采集 ``n_frames`` 帧, 用
   ``time.perf_counter()`` 记录每帧墙钟时间戳;
5. 每帧取 :func:`~ao_shaping.tools.slm.phase_speed.core.far_field_metric`,
   把 "基线帧 + 阶跃后帧" 拼成一条 trace 交给 :func:`analyze_step`;
   报告的响应时间/稳定时间**减去基线帧前缀**, 使 ``t=0`` 恰好是阶跃帧.

⚠️ ``Santec.display_data(wait_time_s=...)`` **内部已经 sleep**; 本模块不再额外
``time.sleep`` (避免双重等待). 槽位轮换只用 ``SlotRotator.next_slot()``.

⚠️ 帧率限制: 测的是 SLM+CCD 组合响应, fps 受相机采集速率约束, 不是孤立的
SLM 上升时间. 这一点写进 JSON 的 ``caveats`` 与终端摘要.

⚠️ 真开设备前必须先拿台面跨进程互斥锁 (``DeviceLock``): SLM 的 USB 句柄独占,
第二个进程 open 会被驱动拒绝, 现象却像"设备没插".
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
from loguru import logger

from ao_shaping.drivers.ccd.common import create_camera, full_scale, is_saturated
from ao_shaping.drivers.slm.factory import require_drive_board_entry
from ao_shaping.drivers.slm.santec import SlotRotator
from ao_shaping.utils.slm.panel_geometry import flat_gray

from .core import PhaseStepResult, analyze_step, far_field_metric

if TYPE_CHECKING:
    from ao_shaping.drivers.ccd.base import BaseCamera
    from ao_shaping.drivers.slm.santec import Santec

    from .__main__ import SpeedConfig

__all__ = [
    "SpeedMeasurementResult",
    "build_pattern",
    "run_speed_measurement",
    "write_pattern",
]


#: 槽位轮换范围 (与 Santec 驱动常量一致).
_SLOT_MIN, _SLOT_MAX = 2, 125
#: raw 灰度编码上限 (10-bit, 0..1023; 写 1024 会回绕成 0, 实测固化).
_MAX_GRAY = 1023
#: 饱和预警阈值 (占满量程比例).
_SAT_FRACTION = 0.95
#: 台面跨进程互斥锁的等待上限 (s).
_LOCK_TIMEOUT_S = 1800.0


@contextmanager
def _bench_lock_context(
    *, lock_factory: Callable[[], Any] | None, injected: bool
) -> Iterator[Any]:
    """真开设备前拿台面跨进程互斥锁; 注入设备工厂时 (离线/测试) 不占锁."""
    if lock_factory is None and injected:
        yield None
        return

    from ao_shaping.tools.kernels.device_mutex import DeviceLock

    make_lock = lock_factory or (lambda: DeviceLock.default())
    lock = make_lock()
    if not lock.acquire(timeout=_LOCK_TIMEOUT_S):
        raise RuntimeError(
            f"台面设备被其它进程占用 (holder={lock.holder()}), "
            f"锁文件 {lock.path}; 等它释放后再重试"
        )
    logger.info("phase_speed: 已获得台面互斥锁 {}", lock.path)
    try:
        yield lock
    finally:
        lock.release()
        logger.info("phase_speed: 已释放台面互斥锁")


class SpeedMeasurementResult:
    """一次测量的全部产物 (供 CLI 与测试读取)."""

    def __init__(
        self,
        result: PhaseStepResult,
        payload: dict[str, Any],
        out_dir: Path,
    ) -> None:
        self.result = result
        self.payload = payload
        self.out_dir = out_dir

    @property
    def fps_measured(self) -> float:
        return float(self.payload["fps_measured"])

    @property
    def fps_requested(self) -> float:
        return float(self.payload["fps_requested"])


def build_pattern(
    pattern: str,
    shape: tuple[int, int],
    gray: int,
    period_px: float,
) -> np.ndarray:
    """生成 raw uint16 灰度图案 (绝不走弧度→灰度转换).

    Args:
        pattern: ``"flat"`` 平场, 或 ``"blaze"`` 闪耀光栅 (深度 = ``gray`` 码).
        shape: ``(height, width)`` 面板尺寸.
        gray: 灰度值 / blaze 深度码 (0..1023).
        period_px: blaze 的光栅周期 (像素); flat 时忽略.

    Returns:
        np.ndarray: ``(height, width)`` uint16 raw 灰度图.
    """
    gray = int(np.clip(int(gray), 0, _MAX_GRAY))
    if pattern == "flat":
        return flat_gray(shape, gray_value=gray)
    if pattern == "blaze":
        height, width = int(shape[0]), int(shape[1])
        period = max(2.0, float(period_px))
        # 一个周期内第 j 个像元取 gray*j/P (j=1..P), 斜坡整步长 gray/P.
        ramp = ((np.arange(width, dtype=np.float64) % period) + 1.0) / period
        phase = np.broadcast_to(ramp, (height, width))
        return np.mod(np.rint(phase * gray), _MAX_GRAY + 1).astype(np.uint16)
    raise ValueError(f"unknown pattern {pattern!r}; expected 'flat' or 'blaze'")


def write_pattern(
    slm: "Santec",
    rotator: SlotRotator,
    pattern: np.ndarray,
    wait_s: float,
) -> int:
    """把 raw 灰度图案写到**新**内存槽并显示.

    ``Santec.display_data(wait_time_s=...)`` 内部已 sleep, 这里不再额外等待.
    """
    slot = rotator.next_slot()
    slm.display_data(pattern, memory_number=slot, wait_time_s=wait_s)
    logger.info("写入槽 {} 图案 min={} max={}", slot, int(pattern.min()), int(pattern.max()))
    return slot


def _open_slm(config: "SpeedConfig") -> "Santec":
    from ao_shaping.drivers.slm.santec import Santec

    return Santec(
        slm_number=config.slm_number,
        wavelength=config.wavelength_nm,
        video_mode=0,  # 内存模式, 严禁 DVI
    )


def _open_camera(config: "SpeedConfig") -> "BaseCamera":
    return create_camera(
        config.cam_type,
        config.cam_id,
        exposure_time_ms=config.exposure_ms,
    )


def run_speed_measurement(
    config: "SpeedConfig",
    *,
    slm_factory: Callable[["SpeedConfig"], Any] | None = None,
    camera_factory: Callable[["SpeedConfig"], Any] | None = None,
    lock_factory: Callable[[], Any] | None = None,
) -> SpeedMeasurementResult:
    """执行相位阶跃速度测量并落盘.

    Args:
        config: CLI 参数对象.
        slm_factory: 返回 SLM 上下文管理器的工厂 (默认打开真实 Santec; 测试注入).
        camera_factory: 返回相机上下文管理器的工厂 (默认走相机注册表; 测试注入).
        lock_factory: 台面互斥锁工厂 (默认 ``DeviceLock.default()``; 测试注入).
            只有注入它才会在注入设备工厂的同时仍去拿锁.

    Returns:
        SpeedMeasurementResult: 含分析结果与完整元数据.
    """
    injected_devices = slm_factory is not None or camera_factory is not None
    slm_factory = slm_factory or _open_slm
    camera_factory = camera_factory or _open_camera

    with ExitStack() as stack:
        # SLM 的 USB 句柄独占: 真开设备前必须先拿台面跨进程锁, 否则第二个
        # 进程 open 会被驱动拒绝 (表现得像"设备没插")。
        stack.enter_context(
            _bench_lock_context(lock_factory=lock_factory, injected=injected_devices)
        )
        slm = stack.enter_context(slm_factory(config))
        cam = stack.enter_context(camera_factory(config))
        return _run_session(config, slm, cam)


def _run_session(config: "SpeedConfig", slm: Any, cam: Any) -> SpeedMeasurementResult:
    """已打开设备后的测量会话 (设备生命周期由 :func:`run_speed_measurement` 管理)."""
    # 采集准入: 驱动板必须应答 (测试替身/仿真面板无该方法 ⇒ no-op).
    drive_board = require_drive_board_entry(slm)

    w, h = int(slm.Panel_Res[0]), int(slm.Panel_Res[1])
    shape = (h, w)
    n_baseline = max(1, int(config.baseline_window))
    n_frames = max(2, int(config.n_frames))

    rotator = SlotRotator(slm, _SLOT_MIN, _SLOT_MAX)
    pattern_a = build_pattern(config.pattern, shape, config.gray_a, config.period_px)
    pattern_b = build_pattern(config.pattern, shape, config.gray_b, config.period_px)

    # 1) 写 A + 稳定 + 采基线.
    slot_a = write_pattern(slm, rotator, pattern_a, config.settle_s)
    baseline_metrics: list[float] = []
    baseline_frames: list[np.ndarray] = []
    for _ in range(n_baseline):
        frame = np.asarray(cam.get_numpy_image(1, skip_first=False))
        baseline_frames.append(frame.copy())
        baseline_metrics.append(far_field_metric(frame))

    # 2) 写 B (不等额外稳定) + 立即逐帧采集并记录时间戳.
    slot_b = write_pattern(slm, rotator, pattern_b, 0.0)
    step_metrics: list[float] = []
    step_frames: list[np.ndarray] = []
    timestamps: list[float] = []
    for _ in range(n_frames):
        t0 = time.perf_counter()
        frame = np.asarray(cam.get_numpy_image(1, skip_first=False))
        t1 = time.perf_counter()
        step_frames.append(frame.copy())
        step_metrics.append(far_field_metric(frame))
        timestamps.append((t0 + t1) / 2.0)

    # 3) 帧率: 实测取阶跃段相邻帧间隔中位数; 请求值按曝光给乐观上界.
    intervals_ms = np.diff(timestamps) * 1000.0
    if intervals_ms.size and np.median(intervals_ms) > 0.0:
        fps_measured = 1000.0 / float(np.median(intervals_ms))
    else:
        fps_measured = 0.0
    fps_requested = 1000.0 / float(config.exposure_ms) if config.exposure_ms > 0 else 0.0

    # 4) 阶跃时序解算: 基线帧 + 阶跃后帧拼成一条 trace, before 落在基线帧上.
    # 时间戳分辨率不足导致 fps 为 0 时, 退回请求帧率, 保证 analyze_step 可算.
    fps_for_analysis = fps_measured if fps_measured > 0.0 else fps_requested
    if fps_for_analysis <= 0.0:
        fps_for_analysis = 1.0
    trace = baseline_metrics + step_metrics
    result = analyze_step(
        trace,
        fps=fps_for_analysis,
        settle_fraction=config.settle_fraction,
        baseline_window=n_baseline,
        settle_window=max(1, int(config.settle_window)),
        response_fraction=config.response_fraction,
    )

    # 5) 把 t=0 对齐到阶跃帧 (减去基线前缀时长).
    if fps_measured > 0.0:
        offset_ms = n_baseline * (1000.0 / fps_measured)
        if result.t_response_ms is not None:
            result.t_response_ms = max(0.0, result.t_response_ms - offset_ms)
        if result.t_settle_ms is not None:
            result.t_settle_ms = max(0.0, result.t_settle_ms - offset_ms)
    result.notes.append(
        "measure.py 已把响应时间/稳定时间减去基线前缀时长 (t=0 = 写入图案 B 后的首帧)."
    )

    # 6) 饱和检查 (相机感知满量程, 避免 uint16 容器被当成 16-bit).
    full_scale_value = full_scale(step_frames[-1], cam)
    saturated = [i for i, f in enumerate(step_frames) if is_saturated(f, cam)]
    if saturated:
        result.notes.append(
            f"检测到 {len(saturated)} 帧接近/达到满量程 (前几帧索引 {saturated[:5]}); "
            "0 级峰值被削顶会低估响应幅度, 请降低曝光重测."
        )

    exposure_effective = _call_exposure(cam)
    payload: dict[str, Any] = {
        "instrument": "SLM phase-change speed (SLM+CCD combined response)",
        "config": {
            "out_dir": str(config.out_dir),
            "slm_number": config.slm_number,
            "wavelength_nm": config.wavelength_nm,
            "cam_type": config.cam_type,
            "cam_id": config.cam_id,
            "exposure_ms": config.exposure_ms,
            "gray_a": config.gray_a,
            "gray_b": config.gray_b,
            "pattern": config.pattern,
            "period_px": config.period_px,
            "n_frames": n_frames,
            "n_baseline": n_baseline,
            "settle_s": config.settle_s,
            "baseline_window": n_baseline,
            "settle_window": int(config.settle_window),
            "settle_fraction": config.settle_fraction,
            "response_fraction": config.response_fraction,
        },
        "device": {
            "panel_shape": list(shape),
            "slot_a": slot_a,
            "slot_b": slot_b,
            "drive_board": drive_board,
            "exposure_effective_ms": exposure_effective,
            "full_scale": full_scale_value,
        },
        "fps_requested": fps_requested,
        "fps_measured": fps_measured,
        "frame_intervals_ms": [float(v) for v in intervals_ms],
        "timestamps_unix_s": [float(v) for v in timestamps],
        "trace": result.trace,
        "step_metric_before": result.step_metric_before,
        "step_metric_after": result.step_metric_after,
        "t_response_ms": result.t_response_ms,
        "t_settle_ms": result.t_settle_ms,
        "frames_to_settle": result.frames_to_settle,
        "saturated_frames": saturated,
        "notes": result.notes,
        "caveats": [
            "本工具测量的是 SLM+CCD 组合系统的响应, 不是孤立的 SLM 上升时间.",
            "时间分辨率受相机帧率限制: 单帧间隔约 "
            f"{(1000.0 / fps_measured if fps_measured > 0 else float('nan')):.3f} ms "
            "(实测), 任何快于该间隔的响应都无法分辨.",
            "请求帧率 1000/exposure_ms 仅按曝光估算的乐观上界, 未计读出与传输时间.",
        ],
    }

    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = str(getattr(config, "tag", "phase_speed") or "phase_speed")
    _save_json(out_dir / f"{tag}_result.json", payload)
    _save_plot(out_dir / f"{tag}_trace.png", result, n_baseline, fps_measured)
    if getattr(config, "save_frames", False):
        np.savez_compressed(
            out_dir / f"{tag}_frames.npz",
            baseline=np.stack(baseline_frames) if baseline_frames else np.empty((0, h, w)),
            step=np.stack(step_frames) if step_frames else np.empty((0, h, w)),
        )

    _print_summary(result, fps_requested, fps_measured, out_dir)
    return SpeedMeasurementResult(result, payload, out_dir)


def _call_exposure(cam: Any) -> float | None:
    getter = getattr(cam, "get_exposure_time_ms", None)
    if getter is None:
        return None
    try:
        return float(getter())
    except Exception as exc:  # pragma: no cover - 后端差异的防御
        logger.warning("读取生效曝光失败: {}", exc)
        return None


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("结果已保存: {}", path)


def _save_plot(path: Path, result: PhaseStepResult, n_baseline: int, fps: float) -> None:
    """保存指标-时间曲线 PNG (matplotlib Agg, 无界面)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - matplotlib 视为可选依赖
        logger.warning("matplotlib 不可用, 跳过 PNG 绘制")
        return

    if fps <= 0.0:
        return
    dt = 1000.0 / fps
    times = [idx * dt - n_baseline * dt for idx in range(len(result.trace))]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(times, result.trace, marker="o", ms=3, label="0-order metric")
    ax.axvline(0.0, color="gray", ls="--", label="phase step (t=0)")
    if result.t_response_ms is not None:
        ax.axvline(result.t_response_ms, color="tab:orange", label="response")
    if result.t_settle_ms is not None:
        ax.axvline(result.t_settle_ms, color="tab:green", label="settle")
    ax.set_xlabel("time relative to phase step (ms)")
    ax.set_ylabel("0-order peak - background (ADU)")
    ax.set_title("SLM phase-step response (SLM+CCD combined)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("曲线已保存: {}", path)


def _print_summary(
    result: PhaseStepResult,
    fps_requested: float,
    fps_measured: float,
    out_dir: Path,
) -> None:
    import click

    click.echo("=== SLM 相位变化速度测量 ===")
    click.echo(f"请求帧率(按曝光估算上界): {fps_requested:.2f} fps")
    frame_ms = 1000.0 / fps_measured if fps_measured > 0 else float("nan")
    click.echo(f"实测帧率: {fps_measured:.2f} fps (单帧 {frame_ms:.2f} ms)")
    click.echo(f"阶跃前电平: {result.step_metric_before:.3f}  ->  阶跃后电平: {result.step_metric_after:.3f}")
    click.echo(f"响应时间 (90%): {result.t_response_ms}")
    click.echo(f"稳定时间 (±10%): {result.t_settle_ms}  (帧数={result.frames_to_settle})")
    click.echo("⚠️  测量受相机帧率限制, 反映的是 SLM+CCD 组合系统响应, 不是孤立 SLM 上升时间.")
    click.echo(f"产物目录: {out_dir.resolve()}")
    for note in result.notes:
        click.echo(f"  - {note}")
