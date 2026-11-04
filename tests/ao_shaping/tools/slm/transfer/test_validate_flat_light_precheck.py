"""平面相位验光前置: 先证光路有光, 再恢复整形相位/曝光, 不把整形掩模低峰值当 blocked。"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from ao_shaping.tools.slm.transfer import validate
from ao_shaping.tools.slm.transfer.light_guard import OpticalPathInvalid


def _bright_frame() -> np.ndarray:
    frame = np.zeros((32, 32), dtype=np.uint8)
    frame[12:19, 12:19] = 80
    return frame


def _dark_frame() -> np.ndarray:
    return np.zeros((32, 32), dtype=np.uint8)


class _Camera:
    def __init__(self, frame: np.ndarray, prior: float = 0.5) -> None:
        self._frame = np.asarray(frame)
        self.readback_ms = float(prior)
        self.reset_calls: list[float] = []
        self.read_exposures: list[float] = []

    def reset_exposure_time(self, value: float) -> float:
        self.readback_ms = float(value)
        self.reset_calls.append(float(value))
        return float(value)

    def get_numpy_image(self, **kwargs) -> np.ndarray:
        self.read_exposures.append(self.readback_ms)
        return self._frame.copy()


class _Slm:
    def __init__(self) -> None:
        self.displays: list[int] = []

    def create_phase_from_array(self, phase) -> np.ndarray:
        arr = np.asarray(phase)
        if np.allclose(arr, 0.0):
            return np.zeros_like(arr, dtype=np.uint16)
        return np.full(arr.shape, 7, dtype=np.uint16)

    def display_data(self, gray) -> None:
        self.displays.append(int(np.asarray(gray).max()))


def _patch_exposure(monkeypatch, cam: _Camera) -> None:
    monkeypatch.setattr(
        "ao_shaping.drivers.ccd.common.get_camera_exposure_ms",
        lambda c: c.readback_ms,
    )


def test_flat_light_precheck_success_restores_phase_and_exposure(monkeypatch, tmp_path):
    cam = _Camera(_bright_frame(), prior=0.5)
    slm = _Slm()
    _patch_exposure(monkeypatch, cam)
    phase = np.ones((8, 8), dtype=np.float32)
    args = SimpleNamespace(out=tmp_path)
    seen: list[dict] = []

    result = validate.flat_phase_light_precheck(
        cam=cam, slm=slm, args=args, observer=seen.append,
        label="demo", phase=phase, floor_ms=0.01, ceiling_ms=10.0,
    )

    assert result["ok"] is True
    # 平面相位先上屏, 随后恢复整形相位。
    assert slm.displays == [0, 7]
    # 最后恢复进入验光前的曝光。
    assert cam.reset_calls[-1] == pytest.approx(0.5)
    assert any(p.name == "demo_flat_light_phase.npy" for p in tmp_path.iterdir())
    assert any("demo_flat_light_read_" in p.name for p in tmp_path.iterdir())
    assert any(p.name == "demo_flat_light_reads.json" for p in tmp_path.iterdir())
    # 平场帧使用独立 light_check_flat 标签, 不冒充整形帧。
    assert seen and all(payload["phase"] == "light_check_flat" for payload in seen)


def test_flat_light_precheck_dark_raises_and_still_restores(monkeypatch, tmp_path):
    cam = _Camera(_dark_frame(), prior=0.7)
    slm = _Slm()
    _patch_exposure(monkeypatch, cam)
    phase = np.ones((8, 8), dtype=np.float32)
    args = SimpleNamespace(out=tmp_path)

    with pytest.raises(OpticalPathInvalid):
        validate.flat_phase_light_precheck(
            cam=cam, slm=slm, args=args, observer=None,
            label="dark", phase=phase, floor_ms=0.01, ceiling_ms=10.0,
        )

    assert slm.displays == [0, 7]
    assert cam.reset_calls[-1] == pytest.approx(0.7)


def test_flat_light_does_not_create_shaping_exposure_evidence(monkeypatch, tmp_path):
    cam = _Camera(_bright_frame(), prior=0.3)
    slm = _Slm()
    _patch_exposure(monkeypatch, cam)
    phase = np.ones((8, 8), dtype=np.float32)
    args = SimpleNamespace(out=tmp_path)

    validate.flat_phase_light_precheck(
        cam=cam, slm=slm, args=args, observer=None,
        label="probe", phase=phase, floor_ms=0.01, ceiling_ms=10.0,
    )

    assert not any(p.name == "probe_exposure_phase.npy" for p in tmp_path.iterdir())
    assert not any(p.name.startswith("probe_exposure_read_") for p in tmp_path.iterdir())
