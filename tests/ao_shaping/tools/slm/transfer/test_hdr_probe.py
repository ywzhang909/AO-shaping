"""``hdr_probe.capture_hdr_display`` 的离线验证: 假相机, 不碰硬件。

钉住探测契约: 任何失败 (相机不支持 / 融合异常 / 无 fused / 空数组) 都返回
``None`` 且不冒泡 —— 探测是 best-effort, 不能让一次记录采集失败。
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from ao_shaping.tools.slm.transfer.hdr_probe import (
    DEFAULT_HDR_EXPOSURES_MS,
    capture_hdr_display,
)


class _FakeCam:
    """capture_hdr 返回 fused 或抛异常, 记录收到的曝光阶梯。"""

    def __init__(self, fused=None, exc: Exception | None = None):
        self._fused = fused
        self._exc = exc
        self.calls: list[list[float]] = []

    def capture_hdr(self, exposures_ms, n_sample: int = 1):
        self.calls.append(list(exposures_ms))
        if self._exc is not None:
            raise self._exc
        return types.SimpleNamespace(fused=self._fused)


def test_capture_hdr_display_returns_fused_float32():
    fused = np.linspace(0, 5000.0, 16, dtype=np.float64).reshape(4, 4)
    cam = _FakeCam(fused=fused)
    out = capture_hdr_display(cam, (1.0, 4.0, 16.0))
    assert out is not None
    assert out.dtype == np.float32
    assert out.shape == (4, 4)
    assert cam.calls == [[1.0, 4.0, 16.0]]


def test_capture_hdr_display_defaults_exposure_ladder():
    cam = _FakeCam(fused=np.ones((2, 2), dtype=np.float64))
    assert capture_hdr_display(cam) is not None
    assert cam.calls == [list(DEFAULT_HDR_EXPOSURES_MS)]


def test_capture_hdr_display_returns_none_on_exception():
    cam = _FakeCam(exc=RuntimeError("HDR unsupported"))
    assert capture_hdr_display(cam) is None


def test_capture_hdr_display_none_on_missing_fused():
    cam = _FakeCam(fused=None)
    assert capture_hdr_display(cam) is None


def test_capture_hdr_display_none_on_empty():
    cam = _FakeCam(fused=np.zeros((0, 0), dtype=np.float64))
    assert capture_hdr_display(cam) is None


def test_capture_hdr_display_passes_n_sample():
    cam = _FakeCam(fused=np.ones((1, 1), dtype=np.float64))
    capture_hdr_display(cam, (1.0, 2.0), n_sample=3)
    assert cam.calls == [[1.0, 2.0]]


def test_capture_hdr_display_preserves_fused_values():
    """fused 是宽动态 radiance×曝光, 不能把 NaN/Inf 悄悄吞掉 (留给色调映射处理)。"""
    fused = np.array([[0.0, 100.0], [1000.0, 10000.0]], dtype=np.float64)
    out = capture_hdr_display(_FakeCam(fused=fused))
    assert out is not None
    assert np.allclose(np.asarray(out, dtype=np.float64), fused, rtol=1e-6)
