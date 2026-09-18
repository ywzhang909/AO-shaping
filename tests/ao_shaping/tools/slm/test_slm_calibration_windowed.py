# -*- coding: utf-8 -*-
"""SLMCCDCalibrator.align 窗口化对准的离线测试 (无硬件).

Fake 契约:
  FakeSLM          Panel_Res=(1920,1200) 对齐 Santec; display_phase 记录相位.
  FakeWindowedCCD  镜像 DahengCamera.reset_window/get_numpy_image 契约:
                   center=(x,y), size=(sx,sy); size=(0,0) -> 全幅;
                   宽高与 offset 按 inc 向下取整; 断言 offset>=0;
                   get_numpy_image 返回当前窗口裁剪帧.
                   zero_order_sequence 提供逐帧 0 级位置(取到末尾则保持),
                   相位经 _describe_phase 推断 flat/blaze 后渲染 0级(+ ±1级).

离线确定性参数 (手算验证):
  sensor (1200,1920), inc=4, K=5021, spot_sigma=1.0 -> fwhm≈3px -> f0=3,
  margin=12; P=16 -> Δ=round(5021/16)=314 -> sx=sy=2*(314+12)=652;
  0级 (700,800) 首窗: center=(800,700) -> offset=(472,372), 返回中心=(328,328);
  漂移序列 (3px/帧) 使 stage B 的 +1级y 位移=314+6=320 -> sy=664,
  首窗 (800,709) -> offset=(472,376), 返回中心=(328,333).
"""
from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from ao_shaping.tools.slm.calibration import SLMCCDCalibrator

TWO_PI = 2.0 * np.pi


def _blaze_step(axis: str, phase: np.ndarray, panel_res: tuple[int, int]) -> float:
    """从 (未包裹) 闪耀相位反推周期: 沿轴向 diff(mod 2π) 的中位正步长."""
    h, w = panel_res
    if axis == "x":
        line = phase[0]
    else:
        line = phase[:, 0]
    wrapped = np.mod(line, TWO_PI)
    diffs = np.diff(wrapped)
    pos = diffs[diffs > 1e-6]
    if len(pos) == 0:
        return 0.0
    step = float(np.median(pos))
    return TWO_PI / step if step > 0 else 0.0


def _describe_phase(phase: Any) -> tuple[str, str | None, float | None]:
    """返回 (kind, axis, period): ('flat',None,None) / ('blaze','x',P) 等."""
    if phase is None:
        return ("flat", None, None)
    p = np.asarray(phase, np.float64)
    if p.size == 0:
        return ("flat", None, None)
    if np.allclose(p, p.ravel()[0]):
        return ("flat", None, None)
    if np.allclose(p, p[0, :][None, :]):  # 每行相同 -> 相位沿 x 变化
        p_period = _blaze_step("x", p, p.shape)
        return ("blaze", "x", p_period if p_period else None)
    if np.allclose(p, p[:, 0][:, None]):  # 每列相同 -> 相位沿 y 变化
        p_period = _blaze_step("y", p, p.shape)
        return ("blaze", "y", p_period if p_period else None)
    return ("other", None, None)


class FakeSLM:
    """最小 SLM 假体: 仅暴露 calibrator 用到的接口."""

    def __init__(self) -> None:
        self.Panel_Res = (1920, 1200)
        self.last_phase: np.ndarray | None = None

    def display_phase(self, phase: np.ndarray) -> None:
        self.last_phase = np.asarray(phase)

    def display_data(self, pattern: np.ndarray) -> None:
        self.last_phase = None


class FakeWindowedCCD:
    """镜像 DahengCamera 开窗契约的假相机 (确定性, 无噪声)."""

    def __init__(
        self,
        sensor_hw: tuple[int, int] = (1200, 1920),
        *,
        inc: int = 4,
        K: float = 5021.0,
        zero_order_sequence: list[tuple[int, int]] | None = None,
        spot_sigma: float = 1.0,
        peak: float = 200.0,
    ) -> None:
        self.sensor_hw = sensor_hw
        self.inc = inc
        self.K = K
        self.zero_order_sequence = zero_order_sequence or [(700, 800)] * 30
        self.spot_sigma = spot_sigma
        self.peak = peak
        self.exposure_time_ms: float = 1.0
        self._offset = (0, 0)
        self._window_size = (0, 0)
        self._calls = 0
        self._current_phase: np.ndarray | None = None  # 由光学链路注入

    # ---- 鼠标垫式渲染: 全幅高斯光斑 -> 裁剪 ----
    @property
    def window_size(self) -> tuple[int, int]:
        return self._window_size

    @property
    def window_offset(self) -> tuple[int, int]:
        return self._offset

    def _next_pos(self) -> tuple[int, int]:
        self._calls += 1
        idx = min(self._calls - 1, len(self.zero_order_sequence) - 1)
        return self.zero_order_sequence[idx]

    def _render_full(self, pos: tuple[int, int]) -> np.ndarray:
        cy, cx = pos
        H, W = self.sensor_hw
        yy, xx = np.mgrid[:H, :W]
        frame = np.zeros((H, W), np.float64)
        kind, axis, period = _describe_phase(self._current_phase)
        spots: list[tuple[int, int]] = [(cy, cx)]  # 0 级恒在
        if kind == "blaze" and period:
            delta = int(round(self.K / period))
            if axis == "x":
                spots.append((cy, cx + delta))
            else:
                spots.append((cy + delta, cx))
        for syy, sxx in spots:
            g = self.peak * np.exp(
                -(((yy - syy) ** 2 + (xx - sxx) ** 2) / (2.0 * self.spot_sigma**2))
            )
            frame += g
        return np.minimum(frame, 65535.0)

    # ---- Daheng 驱动契约镜像 ----
    def reset_window(self, center, size) -> tuple[tuple[int, int], tuple[int, int]]:
        """center=(x,y); size=(sx,sy); size=(0,0) -> 全幅.

        返回 ((w,h), (cx-x_offset, cy-y_offset)), 与真实驱动一致.
        """
        cx, cy = int(round(center[0])), int(round(center[1]))
        sx, sy = int(round(size[0])), int(round(size[1]))
        inc = self.inc
        if sx <= 0 and sy <= 0:
            self._window_size = (0, 0)
            self._offset = (0, 0)
            return ((0, 0), (0, 0))
        width = int(sx // inc * inc)
        height = int(sy // inc * inc)
        self._window_size = (width, height)
        x_offset = int((cx - width // 2) // inc * inc)
        y_offset = int((cy - height // 2) // inc * inc)
        assert x_offset >= 0 and y_offset >= 0, "ROI offset out of sensor"
        self._offset = (x_offset, y_offset)
        return ((width, height), (cx - x_offset, cy - y_offset))

    def get_numpy_image(
        self, n_sample: int = 1, skip_first: bool = True, denoise: bool = False
    ) -> np.ndarray:
        cy, cx = self._next_pos()
        full = self._render_full((cy, cx))
        ww, wh = self.window_size
        x0, y0 = self.window_offset
        if ww <= 0 or wh <= 0:
            return full
        return full[y0 : y0 + wh, x0 : x0 + ww]

    def is_connected(self) -> bool:
        return True

    def close(self) -> None:
        pass


# =====================================================================
# reset_window 契约
# =====================================================================

DRIFT_SEQ = [
    (700, 800),
    (703, 800),
    (706, 800),
    (709, 800),
    (712, 800),
    (715, 800),
]


def test_reset_window_full_frame_restores_full_size() -> None:
    ccd = FakeWindowedCCD()

    ret = ccd.reset_window((0, 0), (0, 0))

    assert ret == ((0, 0), (0, 0))
    assert ccd.window_size == (0, 0)
    assert ccd.window_offset == (0, 0)
    assert ccd.get_numpy_image().shape == (1200, 1920)


def test_reset_window_quantizes_offset_and_returns_center() -> None:
    ccd = FakeWindowedCCD()

    ret = ccd.reset_window((101, 100), (64, 64))

    # width=64 -> x_offset=(101-32)=69 -> 68; y_offset=(100-32)=68 -> 68
    assert ret == ((64, 64), (33, 32))
    assert ccd.window_size == (64, 64)
    assert ccd.window_offset == (68, 68)


def test_reset_window_rejects_outside_sensor() -> None:
    ccd = FakeWindowedCCD()

    with pytest.raises(AssertionError):
        ccd.reset_window((5, 5), (64, 64))  # x_offset=-28 < 0


def test_get_numpy_image_returns_window_crop() -> None:
    ccd = FakeWindowedCCD(zero_order_sequence=[(700, 800)] * 10)
    ccd.reset_window((800, 700), (652, 652))  # offset=(472, 372)

    frame = ccd.get_numpy_image()

    assert frame.shape == (652, 652)
    assert float(frame.max()) == pytest.approx(200.0)
    # 窗内 0级 位于 (700-372, 800-472) = (328, 328)
    assert float(frame[328, 328]) == pytest.approx(200.0)


# =====================================================================
# align 窗口化对准
# =====================================================================


def _make_calibrator(
    seq: list[tuple[int, int]] | None = None,
) -> tuple[SLMCCDCalibrator, FakeSLM, FakeWindowedCCD]:
    slm = FakeSLM()
    ccd = FakeWindowedCCD(zero_order_sequence=seq or [(700, 800)] * 30)

    def _optical_link(phase: np.ndarray) -> None:
        """模拟光学链路: SLM 上显示的相位决定了 CCD 帧的衍射图案."""
        phase = np.asarray(phase)
        slm.last_phase = phase
        ccd._current_phase = phase

    slm.display_phase = _optical_link  # type: ignore[method-assign]
    geo = SLMCCDCalibrator(cast(Any, slm), cast(Any, ccd), settle_s=0.0)
    return geo, slm, ccd


def test_align_success_sets_window_and_calib_keys() -> None:
    geo, slm, ccd = _make_calibrator()

    report = geo.align()

    assert report["ok"] is True
    assert report["period"] == pytest.approx(16.0)
    assert len(report["rounds"]) == 2  # 阶段B(1) + 阶段C开窗(1), 收敛轮不追加
    assert report["center_full"] == pytest.approx([700.0, 800.0])
    calib = geo.calib or {}
    assert calib["align_ok"] is True
    assert calib["align_period"] == pytest.approx(16.0)
    assert np.allclose(calib["align_center_full"], [700.0, 800.0])
    assert np.allclose(calib["align_window_center_xy"], [800.0, 700.0])
    assert np.allclose(calib["align_window_size_xy"], [652.0, 652.0])
    # 驱动侧状态: 成功保留窗口
    assert ccd.window_size == (652, 652)
    assert ccd.window_offset == (472, 372)


def test_align_converges_with_drift_tolerance() -> None:
    geo, _slm, ccd = _make_calibrator(DRIFT_SEQ)

    report = geo.align(max_rounds=10, center_tol_px=4.0)

    # 漂移污染 stage B 的 +1级y 位移 (314+6=320) -> sy=664
    assert report["ok"] is True
    assert report["period"] == pytest.approx(16.0)
    assert len(report["rounds"]) == 2  # B(1) + C(1: rnd0 开窗; rnd1 收敛不追加)
    assert np.allclose(report["center_full"], [712.0, 800.0])
    calib = geo.calib or {}
    assert calib["align_ok"] is True
    assert np.allclose(calib["align_center_full"], [712.0, 800.0])
    assert np.allclose(calib["align_window_center_xy"], [800.0, 709.0])
    assert np.allclose(calib["align_window_size_xy"], [652.0, 664.0])
    assert ccd.window_size == (652, 664)
    assert ccd.window_offset == (472, 376)


def test_align_not_converged_restores_full_frame() -> None:
    geo, _slm, ccd = _make_calibrator(DRIFT_SEQ)

    report = geo.align(max_rounds=3, center_tol_px=2.0)

    assert report["ok"] is False
    assert report["reason"] == "not_converged"
    assert len(report["rounds"]) == 4  # B(1) + C(3 轮开窗)
    calib = geo.calib or {}
    assert calib["align_ok"] is False
    assert "align_window_center_xy" not in calib
    assert "align_window_size_xy" not in calib
    # 失败路径: 恢复全幅
    assert ccd.window_size == (0, 0)
    assert ccd.window_offset == (0, 0)


def test_align_no_spot_restores_full_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    geo, _slm, ccd = _make_calibrator()

    def _sparse(pos: tuple[int, int]) -> np.ndarray:
        frame = np.zeros((1200, 1920), np.float64)
        frame[5, 5] = 100.0  # 阈值内像素 1 < 5 -> _moments 抛 RuntimeError
        return frame

    monkeypatch.setattr(ccd, "_render_full", _sparse)

    report = geo.align()

    assert report["ok"] is False
    assert report["reason"] == "no_spot"
    assert (geo.calib or {})["align_ok"] is False
    assert ccd.window_size == (0, 0)
    assert ccd.window_offset == (0, 0)


def test_align_no_period_fits_restores_full_frame() -> None:
    geo, _slm, ccd = _make_calibrator([(5, 5)] * 30)

    report = geo.align()

    # 0级在 (5,5): margin(12) > 5 -> 全部候选被跳过
    assert report["ok"] is False
    assert report["reason"] == "no_period"
    assert len(report["rounds"]) == 6  # 6 个候选周期各一轮
    assert (geo.calib or {})["align_ok"] is False
    assert ccd.window_size == (0, 0)
    assert ccd.window_offset == (0, 0)


# =====================================================================
# apply_stored_window
# =====================================================================


def test_apply_stored_window_restores_saved_window() -> None:
    geo, _slm, ccd = _make_calibrator()
    geo.calib = {
        "align_window_center_xy": np.array([800.0, 700.0]),
        "align_window_size_xy": np.array([652.0, 652.0]),
    }

    assert geo.apply_stored_window() is True
    assert ccd.window_size == (652, 652)
    assert ccd.window_offset == (472, 372)


def test_apply_stored_window_no_calib_returns_false() -> None:
    geo, _slm, ccd = _make_calibrator()

    assert geo.apply_stored_window() is False

    geo.calib = {}  # 无 align 窗口 keys
    assert geo.apply_stored_window() is False
    # 相机保持现状(全幅), 不主动复位
    assert ccd.window_size == (0, 0)


# =====================================================================
# calibrate 保留 align keys
# =====================================================================


def test_calibrate_preserves_align_keys() -> None:
    geo, _slm, _ccd = _make_calibrator()
    geo.calib = {
        "align_ok": True,
        "align_period": 16.0,
        "align_center_full": np.array([700.0, 800.0]),
        "align_window_center_xy": np.array([800.0, 700.0]),
        "align_window_size_xy": np.array([652.0, 652.0]),
    }

    result = geo.calibrate(periods=(16,))

    assert result["Kx"] == pytest.approx(5021.0, rel=0.01)
    assert result["Ky"] == pytest.approx(5021.0, rel=0.01)
    assert np.allclose(result["center"], [700.0, 800.0])
    # align 结果不被标定流程丢弃
    assert result["align_ok"] is True
    assert result["align_period"] == pytest.approx(16.0)
    assert np.allclose(result["align_center_full"], [700.0, 800.0])
    assert np.allclose(result["align_window_center_xy"], [800.0, 700.0])
    assert np.allclose(result["align_window_size_xy"], [652.0, 652.0])