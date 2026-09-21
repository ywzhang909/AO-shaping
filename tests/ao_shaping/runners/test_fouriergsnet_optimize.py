# -*- coding: utf-8 -*-
"""fouriergsnet_optimize.py 的离线 mock-SLM 测试 (不打开任何硬件).

该脚本是独立 CLI (不属于 ao_shaping 包), 因此用 importlib 从仓库根加载, 并
通过 FakeSLM/FakeCCD 驱动其显示助手与 ShapingSystem, 断言 AGENTS.md 中的设备契约:

* 每次写入都做内存槽轮换 (2..125) 且使用 MEMORY_MODE_INTERNAL;
* 平场 / 自定义 LUT 通道发送 **原始 uint16 灰度**, 绝不走弧度转换;
* ShapingSystem 使用 ``Santec.Panel_Res`` 解析面板尺寸
  (回归: 旧代码引用了不存在的 ``SantecSLM200``).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO_ROOT / "fouriergsnet_optimize.py"
_SRC = _REPO_ROOT / "src"

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ao_shaping.drivers.slm.santec import MEMORY_MODE_INTERNAL  # noqa: E402
from ao_shaping.utils.slm import phase_display  # noqa: E402


def _load_module():
    """从磁盘加载独立脚本 (不加入 sys.modules 缓存之外的位置)."""
    spec = importlib.util.spec_from_file_location("fouriergsnet_optimize", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fg = _load_module()


class FakeSLM:
    """Santec 最小替身: 记录每次显示调用, 不接触硬件."""

    Panel_Res = (1920, 1200)

    def __init__(self):
        self.displayed_memory_number: int | None = 7
        self.phase_calls: list[dict] = []
        self.data_calls: list[dict] = []

    def get_displayed_memory_number(self) -> int:
        return int(self.displayed_memory_number or 1)

    def display_phase(self, phase, wait_time_s=None, memory_number=None,
                      memory_mode=None) -> int:
        self.phase_calls.append(dict(phase=np.array(phase), wait_time_s=wait_time_s,
                                     memory_number=memory_number,
                                     memory_mode=memory_mode))
        self.displayed_memory_number = memory_number
        return 0

    def display_data(self, data, wait_time_s=None, memory_number=None,
                     memory_mode=None) -> int:
        self.data_calls.append(dict(data=np.array(data), wait_time_s=wait_time_s,
                                    memory_number=memory_number,
                                    memory_mode=memory_mode))
        self.displayed_memory_number = memory_number
        return 0


class FakeCCD:
    def get_numpy_image(self, n_sample: int = 1) -> np.ndarray:
        return np.zeros((64, 64), np.float64)


def _calib() -> dict:
    """ShapingSystem 构造 + display() 所需的最小标定字典."""
    return dict(
        Kx=80.0, Ky=80.0,
        center=np.array([60.0, 60.0]),
        crop_side=120,
        rotation_deg=0.0,
        beam_center=np.array([600.0, 960.0]),
    )


def _lut_result() -> dict:
    return dict(phase_range=2.0 * np.pi,
                gray_of_phase=np.linspace(0.0, 255.0, 256))


def _freeze(monkeypatch, slot: int) -> None:
    """固定槽位 + 消除 settle 睡眠, 让测试确定且快速."""
    monkeypatch.setattr(phase_display, "pick_slm_slot", lambda slm: slot)
    monkeypatch.setattr(fg.time, "sleep", lambda *_: None)


# --------------------------------------------------------------------------
# 显示助手契约
# --------------------------------------------------------------------------
def test_display_radians_rotates_slot_and_uses_memory_mode(monkeypatch):
    _freeze(monkeypatch, 42)
    slm = FakeSLM()
    phase = np.linspace(0.0, 2 * np.pi, 64, dtype=np.float32).reshape(8, 8)

    slot = fg._display_radians(slm, phase, 0.0)

    assert slot == 42
    assert len(slm.phase_calls) == 1
    assert slm.data_calls == []
    call = slm.phase_calls[0]
    assert call["memory_number"] == 42
    assert call["memory_mode"] == MEMORY_MODE_INTERNAL == 0
    assert call["wait_time_s"] == 0.0
    np.testing.assert_allclose(call["phase"], phase)


def test_display_grayscale_sends_raw_uint16(monkeypatch):
    _freeze(monkeypatch, 55)
    slm = FakeSLM()
    gray = np.arange(16, dtype=np.uint16).reshape(4, 4)

    slot = fg._display_grayscale(slm, gray, 0.0)

    assert slot == 55
    assert len(slm.data_calls) == 1
    assert slm.phase_calls == []
    call = slm.data_calls[0]
    assert call["data"].dtype == np.uint16
    np.testing.assert_array_equal(call["data"], gray)
    assert call["memory_number"] == 55
    assert call["memory_mode"] == MEMORY_MODE_INTERNAL
    assert call["wait_time_s"] == 0.0


def test_show_flat_uses_grayscale_path(monkeypatch):
    _freeze(monkeypatch, 60)
    slm = FakeSLM()
    cal = fg.SLMCCDCalibrator(slm, FakeCCD(), lambda ccd: np.zeros((64, 64)),
                              slm.Panel_Res, settle_s=0.0)

    cal._show_flat()

    assert len(slm.data_calls) == 1
    assert slm.phase_calls == []
    data = slm.data_calls[0]["data"]
    assert data.dtype == np.uint16
    assert data.shape == tuple(slm.Panel_Res)
    assert not data.any()


# --------------------------------------------------------------------------
# ShapingSystem 契约
# --------------------------------------------------------------------------
def test_shaping_system_display_uses_lut_grayscale(monkeypatch):
    _freeze(monkeypatch, 70)
    slm = FakeSLM()
    sys_ = fg.ShapingSystem(slm, FakeCCD(), lambda ccd: np.zeros((64, 64)),
                            _calib(), _lut_result())
    phi = fg.torch.full((fg.N, fg.N), np.pi, device=fg.DEV)

    sys_.display(phi)

    assert len(slm.data_calls) == 1
    assert slm.phase_calls == []
    data = slm.data_calls[0]["data"]
    assert data.dtype == np.uint16
    assert data.shape == tuple(slm.Panel_Res)
    assert 0 < int(data.max()) <= 255
    assert slm.data_calls[0]["memory_mode"] == MEMORY_MODE_INTERNAL


def test_shaping_system_uses_santec_panel_res():
    from ao_shaping.drivers.slm import Santec

    slm = FakeSLM()
    sys_ = fg.ShapingSystem(slm, FakeCCD(), lambda ccd: np.zeros((64, 64)),
                            _calib(), _lut_result())

    assert tuple(Santec.Panel_Res) == sys_.geo.panel_res
    assert tuple(Santec.Panel_Res) == sys_.lut.panel_res
