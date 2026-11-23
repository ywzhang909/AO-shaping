"""hold 模式漂移反馈（Jacobian + P 控制器）的离线契约。

不打开 SLM / CCD / 激光。钉住四条：

* ``drift_feedback=False``（默认）：既不改写下发相位，记录里也不出现反馈字段，
  现有行为逐位不变；
* 启用 + 合成常量漂移：实测质心残差被反馈压向收敛（不再发散），
  ``applied_phase`` 仍记纯候选相位（反馈不并进结论）；
* 单轮反馈倾斜超上限即判故障（回滚到纯形状并收尾）；
* ``drift_feedback_gain`` / ``drift_feedback_max_rad`` 的配置校验。

数值约定（与 ``test_closed_loop_shaping`` 同一替身几何）：传感器 40x40，分析窗口
5x5 位于 (8..12, 8..12)，控制网格 8x8，基线帧无漂移；从第二次采集起注入常量
质心漂移。
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from ao_shaping.gui.slm.closed_loop_shaping import (
    CapturedFrame,
    ClosedLoopConfig,
    ClosedLoopRunner,
    LoopState,
    analysis_grid,
    target_field,
)

_SPAN = 4
_SENSOR = 40
_CENTRE = (10.0, 10.0)
_GAUSS_SIGMA = 0.9
_GRID_N = 8


def _xy() -> np.ndarray:
    return analysis_grid(_CENTRE, _SPAN)


def _target() -> np.ndarray:
    return target_field(_xy(), half_width=2.0, half_height=2.0)


def _embed(pattern: np.ndarray) -> np.ndarray:
    """把窗口图案散到取景框的采样点上，其余像素为 0（与旧测试同一约定）。"""
    frame = np.zeros((_SENSOR, _SENSOR), dtype=np.float64)
    grid = _xy()
    rows = grid[..., 1].astype(int)
    cols = grid[..., 0].astype(int)
    frame[rows, cols] = np.asarray(pattern, dtype=np.float64)
    return frame


def _ramp(n: int, axis: int) -> np.ndarray:
    r = np.arange(n, dtype=np.float64) / float(n - 1)
    return r[None, :] if axis == 1 else r[:, None]


def _extract_tilt(phase) -> tuple[float, float]:
    """把相位投影到 x / y 单位斜坡上，返回 (coef_x, coef_y)。

    斜坡沿整个 (H, W) 网格广播，因此按全网格做最小二乘，单位峰谷斜坡的系数
    恰为 1（与 Jacobian 探针、反馈相位构造一致）。
    """
    if hasattr(phase, "detach"):
        phase = phase.detach().cpu().numpy()
    p = np.asarray(phase, dtype=np.float64)

    def coeff(ramp: np.ndarray) -> float:
        r = np.broadcast_to(ramp - ramp.mean(), p.shape)
        denom = float((r * r).sum())
        if denom <= 0.0:
            return 0.0
        return float((p * r).sum() / denom)

    return coeff(_ramp(p.shape[1], 1)), coeff(_ramp(p.shape[0], 0))


def _spot_pattern(sx: float, sy: float) -> np.ndarray:
    """分析窗口上的偏心高斯斑（峰 ~80，无底：质心位移 ≈ 注入漂移）。"""
    grid = _xy()
    gx = grid[..., 0] - _CENTRE[0] - sx
    gy = grid[..., 1] - _CENTRE[1] - sy
    return 80.0 * np.exp(-(gx**2 + gy**2) / (2 * _GAUSS_SIGMA**2))


class _TiltForward:
    """合成光路：相位含 tilt 分量 c 时，光斑质心平移 k·c（px/rad）。

    与 ``_TimedDriftBench`` 用同一套斜坡投影，保证 Jacobian 探针测到的灵敏度
    就是硬件侧真实响应 —— 反馈才可能真正收敛。输出用 torch 算子从输入相位
    现算（可微），``refine_phase`` 的反向传播才走得通。
    """

    def __init__(self, xy: np.ndarray, kx: float = 2.5, ky: float = 2.5) -> None:
        self.xy = np.asarray(xy, dtype=np.float64)
        self.kx, self.ky = kx, ky

    def predict(self, phase, exposure_ms=None):
        import torch

        p = (
            phase
            if torch.is_tensor(phase)
            else torch.as_tensor(np.asarray(phase, dtype=np.float32))
        )
        if p.ndim == 2:
            p = p[None]
        height, width = int(p.shape[-2]), int(p.shape[-1])
        rx = torch.linspace(0.0, 1.0, width, device=p.device, dtype=p.dtype)
        ry = torch.linspace(0.0, 1.0, height, device=p.device, dtype=p.dtype)

        def coeff(ramp: torch.Tensor) -> torch.Tensor:
            r = ramp - ramp.mean()
            r_full = r.expand_as(p)
            denom = (r_full * r_full).sum()
            return (p * r_full).sum() / denom

        cx = coeff(rx[None, None, :])
        cy = coeff(ry[None, :, None])
        grid = torch.as_tensor(self.xy, dtype=p.dtype, device=p.device)
        gx = grid[..., 0] - _CENTRE[0] - self.kx * cx
        gy = grid[..., 1] - _CENTRE[1] - self.ky * cy
        intensity = torch.exp(-(gx**2 + gy**2) / (2 * _GAUSS_SIGMA**2))
        return intensity[None]  # (1, H, W)


class _TimedDriftBench:
    """假光路：真实单调时钟（配对判定）+ 从第二次采集起注入常量漂移。

    下发相位里含 tilt 时，光斑质心按 ``_TiltForward`` 同一灵敏度平移，使
    "反馈相位 → 质心" 的正向链路与 Jacobian 探针完全一致。
    """

    def __init__(
        self,
        *,
        drift_xy: tuple[float, float] = (3.0, 0.0),
        kx: float = 2.5,
        ky: float = 2.5,
    ) -> None:
        self.applied: list[np.ndarray] = []
        self._frame_id = 0
        self.closed = False
        self.opened = 0
        self.exposure_ms = 1.0
        self.full_scale = 255.0
        self.drift_xy = drift_xy
        self.kx, self.ky = kx, ky

    def open(self) -> None:
        self.opened += 1

    def describe(self) -> dict:
        return {"name": "drift", "sensor": _SENSOR}

    def apply_phase(self, phase: np.ndarray) -> str:
        self.applied.append(np.array(phase, dtype=np.float64, copy=True))
        return f"phase-{len(self.applied)}"

    def capture(self) -> CapturedFrame:
        self._frame_id += 1
        phase = self.applied[-1] if self.applied else np.zeros((_GRID_N, _GRID_N))
        cx, cy = _extract_tilt(phase)
        drift = self.drift_xy if self._frame_id >= 2 else (0.0, 0.0)
        sx = drift[0] + self.kx * cx
        sy = drift[1] + self.ky * cy
        return CapturedFrame(
            frame=_embed(_spot_pattern(sx, sy)),
            frame_id=self._frame_id,
            exposure_ms=self.exposure_ms,
            full_scale=self.full_scale,
            monotonic_ns=time.monotonic_ns(),
        )

    def close(self) -> None:
        self.closed = True


def _config(**overrides) -> ClosedLoopConfig:
    base = dict(
        xy=_xy(),
        target=_target(),
        start_phase=np.zeros((_GRID_N, _GRID_N), dtype=np.float64),
        max_rounds=4,
        candidate_policy="auto",
        adapt=False,
        exposure_ms=1.0,
        mode="hold",
    )
    base.update(overrides)
    return ClosedLoopConfig(**base)


# --------------------------------------------------------------------------- #
# 默认关闭：行为与旧版一致
# --------------------------------------------------------------------------- #


def test_drift_feedback_disabled_by_default_and_changes_nothing() -> None:
    cfg = _config()
    assert cfg.drift_feedback is False
    assert cfg.drift_feedback_gain == 0.5
    assert cfg.drift_feedback_max_rad == 10.0

    bench = _TimedDriftBench(drift_xy=(3.0, 0.0))
    runner = ClosedLoopRunner(bench, config=cfg, forward=_TiltForward(_xy()))
    summary = runner.run()

    assert summary.state is LoopState.STOPPED
    assert "轮数上限" in summary.reason
    rounds = summary.rounds
    assert len(rounds) == 5  # 基线 + 4 轮

    # 反馈字段一律不出现：既没算反馈，也没写进记录。
    for record in rounds[1:]:
        assert record.drift_feedback_tilt_rad is None
        assert record.drift_residual_px is None

    # 漂移没有被校正：每轮实测质心都停留在漂移位置（相对基线 > 2 px）。
    baseline_xy = rounds[0].gate.centroid_xy
    for record in rounds[1:]:
        dx = record.gate.centroid_xy[0] - baseline_xy[0]
        assert dx > 2.0

    # 下发相位始终是纯候选相位（起点），从未叠加反馈项。
    assert np.allclose(summary.applied_phase, np.zeros((_GRID_N, _GRID_N)))
    assert bench.closed


# --------------------------------------------------------------------------- #
# 启用：残差收敛、applied_phase 保持纯候选
# --------------------------------------------------------------------------- #


def test_drift_feedback_converges_residual_and_keeps_applied_phase_pure() -> None:
    events: list = []
    gain = 0.5
    bench = _TimedDriftBench(drift_xy=(3.0, 0.0))
    runner = ClosedLoopRunner(
        bench,
        config=_config(
            drift_feedback=True,
            drift_feedback_gain=gain,
        ),
        forward=_TiltForward(_xy()),
        on_event=events.append,
    )
    summary = runner.run()

    assert summary.state is LoopState.STOPPED
    rounds = summary.rounds
    assert len(rounds) == 5  # 基线 + 4 轮（再长会撞上监督器的 non_improving 冻结）

    # 实测证据看门禁质心（记录里的 drift_residual_px 是"本轮反馈据以计算"的滞后
    # 残差，来自上一帧），两者都留在记录里，这里用实测值判定收敛。
    baseline_xy = np.asarray(rounds[0].gate.centroid_xy, dtype=np.float64)
    measured = [
        float(np.asarray(r.gate.centroid_xy, dtype=np.float64)[0] - baseline_xy[0])
        for r in rounds[1:]
    ]
    drift_px = measured[0]  # 第 1 轮帧已含漂移：开环残差（≈3.8 px）
    assert drift_px > 2.0

    # 纯 P + 单位延迟的离散闭环不动点：e = d - gain·e ⇒ 稳态残差 d/(1+gain)。
    steady = drift_px / (1.0 + gain)
    # 反馈从不把误差放大到开环之上（不发散）。
    assert all(m <= drift_px + 0.2 for m in measured)
    # 后两轮已经落在稳态附近（有界收敛，而非归零）。
    assert all(abs(m - steady) < 0.8 for m in measured[-2:])
    assert abs(measured[-1]) < abs(measured[0])

    # 每轮都有有限且受控的反馈倾斜（未触发超限故障）。
    residuals = [r.drift_residual_px for r in rounds[1:]]
    assert all(r is not None for r in residuals)
    tilts = [r.drift_feedback_tilt_rad for r in rounds[1:]]
    assert all(t is not None and 0.0 <= t < 10.0 for t in tilts)

    # applied_phase 保持纯候选（起点相位）：反馈从不并进结论。
    assert np.allclose(summary.applied_phase, np.zeros((_GRID_N, _GRID_N)))
    assert np.allclose(bench.applied[-1], np.zeros((_GRID_N, _GRID_N)))

    # 页面 lock 事件带上反馈字段。
    lock_payloads = [e.payload for e in events if e.kind == "lock"]
    assert lock_payloads
    assert "drift_feedback_tilt_rad" in lock_payloads[-1]
    assert "drift_residual_px" in lock_payloads[-1]

    # Jacobian 数值量级：x 灵敏度 ~2.4 px/rad（高斯在 2px 采样网格上的质心）。
    jac = runner._drift_jac[id(runner.forward)]
    assert jac is not None
    assert np.all(np.isfinite(jac))
    assert 0.5 < abs(jac[0, 0]) < 5.0
    assert 0.5 < abs(jac[1, 1]) < 5.0
    assert abs(jac[0, 1]) < 0.2  # 两轴解耦


# --------------------------------------------------------------------------- #
# 超限：走 FAULT 收尾
# --------------------------------------------------------------------------- #


def test_drift_feedback_over_limit_ends_in_fault() -> None:
    bench = _TimedDriftBench(drift_xy=(3.0, 0.0))
    runner = ClosedLoopRunner(
        bench,
        config=_config(drift_feedback=True, drift_feedback_max_rad=0.01),
        forward=_TiltForward(_xy()),
    )
    summary = runner.run()

    assert summary.state is LoopState.FAULT
    assert "超过上限" in summary.reason
    # 收尾回滚到纯形状（起点相位）。
    assert np.allclose(summary.applied_phase, np.zeros((_GRID_N, _GRID_N)))
    assert bench.closed


def test_drift_feedback_config_validation() -> None:
    with pytest.raises(ValueError, match="drift_feedback_gain"):
        _config(drift_feedback=True, drift_feedback_gain=0.0)
    with pytest.raises(ValueError, match="drift_feedback_max_rad"):
        _config(drift_feedback=True, drift_feedback_max_rad=0.0)
    # 关闭时允许任意数值（保持旧行为），不校验。
    _config(drift_feedback=False, drift_feedback_gain=-1.0, drift_feedback_max_rad=-1.0)
