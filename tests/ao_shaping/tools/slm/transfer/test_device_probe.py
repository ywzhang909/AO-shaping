"""``device_probe`` 的离线验证: 假相机/假 SLM, 不碰硬件。

钉住探测契约: 任何失败都返回 ``{"ok": False, ...}`` 且不冒泡; 正常路径
返回 CCD profile / 自检结果 / 0级基线。
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from ao_shaping.tools.slm.transfer.device_probe import (
    detect_ccd_type,
    device_selfcheck,
    probe_environment,
    zero_order_probe,
)


class _FakeCam:
    """假相机: 读帧序列, 支持 format/bit_depth/pixel_size/model 注入。"""

    def __init__(self, frames=None, exc: Exception | None = None,
                 fmt: str = "MONO8", bit_depth: int = 8,
                 pixel_size_um: float = 2.2, model: str = "MER2-507-23GM",
                 exposure_ms: float = 10.0):
        self._frames = list(frames) if frames else [np.ones((4, 4), dtype=np.uint8)]
        self._exc = exc
        self._idx = 0
        self.pixel_size_um = pixel_size_um
        self.bit_depth = bit_depth
        self._fmt = fmt
        self._model = model
        self._exposure_ms = exposure_ms
        self.cam = types.SimpleNamespace(
            PixelFormat=types.SimpleNamespace(get=lambda: fmt),
            DeviceModelName=types.SimpleNamespace(get=lambda: model),
            PixelSize=types.SimpleNamespace(get=lambda: bit_depth),
        )

    def get_numpy_image(self, n_sample: int = 1, skip_first: bool = False):
        if self._exc is not None:
            raise self._exc
        f = self._frames[self._idx % len(self._frames)]
        self._idx += 1
        return np.asarray(f)

    def get_exposure_time_ms(self):
        return self._exposure_ms


class _FakeSlm:
    def __init__(self):
        self.calls: list = []

    def set_phase_overlay(self, overlay):
        self.calls.append(overlay)


def test_detect_ccd_type_returns_profile():
    frame = np.arange(16, dtype=np.uint8).reshape(4, 4)
    cam = _FakeCam(frames=[frame], fmt="MONO8", bit_depth=8,
                   pixel_size_um=2.2, model="MER2-507-23GM")
    profile = detect_ccd_type(cam, cam_id=0, cam_type="daheng")
    assert profile["ok"] is True
    assert profile["format"] == "MONO8"
    assert profile["bit_depth"] == 8
    assert profile["pixel_size_um"] == 2.2
    assert profile["model"] == "MER2-507-23GM"
    assert profile["shape"] == (4, 4)
    assert "MONO8" in profile["profile"]


def test_detect_ccd_type_tolerates_failures():
    cam = _FakeCam(exc=RuntimeError("no sdk"))
    profile = detect_ccd_type(cam)
    assert profile["ok"] is False
    assert profile["shape"] is None
    # 其他字段仍尽力读 (cam 对象上仍有属性), 不抛
    assert "profile" in profile


def test_device_selfcheck_ok_with_light():
    frame = np.zeros((4, 4), dtype=np.uint8)
    frame[1, 1] = 100
    cam = _FakeCam(frames=[frame, frame, frame])
    res = device_selfcheck(cam, n_frame=3)
    assert res["sensor_ok"] is True
    assert res["light_ok"] is True
    assert res["focus_ok"] is True
    assert res["ok"] is True
    assert res["centroid"] == (1, 1)
    assert res["frames"] == 3


def test_device_selfcheck_no_light():
    frame = np.zeros((4, 4), dtype=np.uint8)
    cam = _FakeCam(frames=[frame, frame, frame])
    res = device_selfcheck(cam, n_frame=3)
    assert res["sensor_ok"] is True
    assert res["light_ok"] is False
    assert res["ok"] is False


def test_device_selfcheck_error_does_not_raise():
    cam = _FakeCam(exc=RuntimeError("read failed"))
    res = device_selfcheck(cam)
    assert res["ok"] is False
    assert "error" in res


def test_zero_order_probe_returns_baseline():
    frame = np.zeros((4, 4), dtype=np.uint8)
    frame[2, 3] = 50
    cam = _FakeCam(frames=[frame, frame, frame], exposure_ms=5.0)
    slm = _FakeSlm()
    res = zero_order_probe(cam, slm, n_frame=3, settle_s=0.0)
    assert res["ok"] is True
    assert res["peak"] == 50.0
    assert res["mean"] > 0
    assert res["centroid"] == (3, 2)
    assert res["counts_per_ms"] == 10.0  # 50 / 5.0
    assert slm.calls == [None]  # set_phase_overlay(None) 平场


def test_zero_order_probe_without_slm():
    frame = np.ones((2, 2), dtype=np.uint8) * 7
    cam = _FakeCam(frames=[frame])
    res = zero_order_probe(cam, None, n_frame=1, settle_s=0.0)
    assert res["ok"] is True
    assert res["peak"] == 7.0
    assert res["counts_per_ms"] == 0.7  # 7 / 10.0


def test_zero_order_probe_error():
    cam = _FakeCam(exc=RuntimeError("boom"))
    res = zero_order_probe(cam, None, n_frame=1, settle_s=0.0)
    assert res["ok"] is False
    assert "error" in res


def test_probe_environment_aggregates():
    frame = np.zeros((4, 4), dtype=np.uint8)
    frame[0, 0] = 20
    cam = _FakeCam(frames=[frame, frame, frame])
    slm = _FakeSlm()
    out = probe_environment(cam, slm, cam_id=2, cam_type="daheng", n_frame=2)
    assert set(out.keys()) == {"ccd_profile", "selfcheck", "zero_order"}
    assert out["ccd_profile"]["cam_id"] == 2
    assert out["selfcheck"]["ok"] is True
    assert out["zero_order"]["ok"] is True