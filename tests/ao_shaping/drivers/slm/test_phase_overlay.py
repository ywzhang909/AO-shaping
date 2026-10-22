"""Santec 驱动发送管线叠加层 (T4) 的离线测试.

两个物理上不同的叠加层在同一处 (``create_phase_from_array``) 应用，空间不同故
分两个槽：

* **相位叠加** ``_phase_overlay`` (弧度, 离焦) —— 在 ``_resize_to_panel`` 之后、
  弧度→灰度转换之前加到输入相位上，因此**先于** 矫正 / LUT / 平移；
* **底相位** ``_base_phase`` (灰度) —— 在 矫正 / LUT 之后、最终 ``mod`` 之前
  以灰度相加，因此**独立于** LUT 重映射.

测试在 Linux 上运行：``_slm_win`` (Windows SDK) 在模块顶层 import 时即触发
``ctypes.WinDLL``，所以按 ``test_gray_csv.py`` 的既有做法，在 import 驱动前先
把 SDK 塞进 ``sys.modules``。驱动**不连接** (``is_open=False``)，但
``create_phase_from_array`` 是纯 numpy 管线，不触碰 ``self._slm``，可离线运行.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import numpy as np

sys.modules["ao_shaping.drivers.slm.santec._slm_win"] = MagicMock()

from ao_shaping.drivers.slm.santec import Santec


def _gray(phase: float, base: int = 0) -> int:
    """与驱动同序的灰度计算：(phase/2π + base/max_gray) mod 1，再 ×max_gray."""
    mg = Santec.MAX_GRAYSCALE_VALUE
    return int(np.mod(phase / (2 * np.pi) + base / mg, 1.0) * mg)


def _panel() -> np.ndarray:
    return np.zeros((Santec.Panel_Res[1], Santec.Panel_Res[0]))


def test_overlays_start_absent() -> None:
    slm = Santec(slm_number=1)
    assert slm.get_phase_overlay() is None
    assert slm.get_base_phase() is None


def test_set_get_returns_independent_copies() -> None:
    slm = Santec(slm_number=1)
    ov = _panel()
    ov[10, 10] = 0.5
    slm.set_phase_overlay(ov)
    got = slm.get_phase_overlay()
    assert got is not ov
    assert np.allclose(got, ov)
    # 副本独立：改原数组不影响驱动内部值
    ov[10, 10] = 99.0
    assert slm.get_phase_overlay()[10, 10] == 0.5

    base = np.zeros_like(ov, dtype=np.uint16)
    base[0, 0] = 7
    slm.set_base_phase(base)
    gotb = slm.get_base_phase()
    assert gotb is not base
    assert gotb.dtype == np.uint16
    assert gotb[0, 0] == 7


def test_none_clears_both() -> None:
    slm = Santec(slm_number=1)
    slm.set_phase_overlay(_panel())
    slm.set_base_phase(np.zeros(_panel().shape, dtype=np.uint16))
    assert slm.get_phase_overlay() is not None
    assert slm.get_base_phase() is not None
    slm.set_phase_overlay(None)
    slm.set_base_phase(None)
    assert slm.get_phase_overlay() is None
    assert slm.get_base_phase() is None


def test_phase_overlay_is_added_in_radians_before_gray_conversion() -> None:
    slm = Santec(slm_number=1, shift_x=0, shift_y=0)
    out = slm.create_phase_from_array(_panel())
    assert np.all(out == 0)
    slm.set_phase_overlay(np.full_like(_panel(), np.pi / 2))
    out = slm.create_phase_from_array(_panel())
    assert int(np.unique(out)[0]) == _gray(np.pi / 2)
    assert int(np.unique(out)[0]) == 255  # π/2 → 1/4 量程


def test_base_phase_is_added_in_grayscale_space() -> None:
    slm = Santec(slm_number=1, shift_x=0, shift_y=0)
    base = np.zeros((Santec.Panel_Res[1], Santec.Panel_Res[0]), dtype=np.uint16)
    base[:] = 40
    slm.set_base_phase(base)
    out = slm.create_phase_from_array(_panel())
    # 零相位 + 底相位 40 → 灰度 40（底相位原样出现）
    assert int(np.unique(out)[0]) == 40
    # 相位 π/2 (→255) + 底相位 900 → (255+900) mod 1023，回绕生效
    base[:] = 900
    slm.set_base_phase(base)
    out = slm.create_phase_from_array(np.full_like(_panel(), np.pi / 2))
    assert int(np.unique(out)[0]) == (255 + 900) % 1023


def test_phase_overlay_applies_before_lut_remap() -> None:
    """离焦是相位层：必须先于 LUT 重映射，否则相位-灰度补偿会被破坏."""
    slm = Santec(slm_number=1, shift_x=0, shift_y=0)
    # 非恒等 LUT：灰度 g → max - g（真正的翻转，能区分施加顺序）
    mg = Santec.MAX_GRAYSCALE_VALUE
    slm._lut = (mg - 1 - np.arange(mg, dtype=np.int64)).astype(np.uint16)

    slm.set_phase_overlay(np.full_like(_panel(), np.pi / 2))
    out = slm.create_phase_from_array(_panel())
    raw = _gray(np.pi / 2)  # 255
    # 若 overlay 在 LUT 之前 (正确)：gray=255 → lut[255]=768
    # 若在 LUT 之后 (错误)：会再 +255
    assert int(np.unique(out)[0]) == mg - 1 - raw


def test_base_phase_applies_after_lut_remap() -> None:
    """底相位是灰度层：在 LUT 之后加，不被 LUT 重映射改变."""
    slm = Santec(slm_number=1, shift_x=0, shift_y=0)
    mg = Santec.MAX_GRAYSCALE_VALUE
    # 翻转 LUT (真正的置换)：能区分底相位在 LUT 前后
    slm._lut = (mg - 1 - np.arange(mg, dtype=np.int64)).astype(np.uint16)
    # 零相位 → raw=0 → lut[0]=1022；底相位 40 应在其后原样加 → (1022+40)%1023
    base = np.zeros((Santec.Panel_Res[1], Santec.Panel_Res[0]), dtype=np.uint16)
    base[:] = 40
    slm.set_base_phase(base)
    out = slm.create_phase_from_array(_panel())
    expected = int(np.mod(slm._lut[0] + 40, mg))
    assert int(np.unique(out)[0]) == expected
    # 若底相位在 LUT 之前 (错误)：lut[40]=982，与正确值不同
    assert int(np.unique(out)[0]) != int(slm._lut[40])


def test_phase_overlay_and_base_phase_compose() -> None:
    slm = Santec(slm_number=1, shift_x=0, shift_y=0)
    slm.set_phase_overlay(np.full_like(_panel(), np.pi / 2))  # → 255
    base = np.zeros((Santec.Panel_Res[1], Santec.Panel_Res[0]), dtype=np.uint16)
    base[:] = 40
    slm.set_base_phase(base)
    out = slm.create_phase_from_array(_panel())
    assert int(np.unique(out)[0]) == (255 + 40) % 1023


def test_shape_mismatched_overlay_is_skipped_not_fatal() -> None:
    """尺寸不符的叠加层应跳过 (告警) 而非崩溃/错位."""
    slm = Santec(slm_number=1, shift_x=0, shift_y=0)
    bad = np.full((3, 4), np.pi / 2)
    slm.set_phase_overlay(bad)
    out = slm.create_phase_from_array(_panel())
    # 被跳过 → 等价于无叠加
    assert np.all(out == 0)

    slm2 = Santec(slm_number=1, shift_x=0, shift_y=0)
    slm2.set_base_phase(np.zeros((3, 4), dtype=np.uint16))
    out2 = slm2.create_phase_from_array(_panel())
    assert np.all(out2 == 0)


def test_small_input_is_resized_before_overlay_applies() -> None:
    """叠加是面板坐标层、在 resize 之后施加，故按面板尺度覆盖整个面板.

    输入 100×100（小于面板）先 resize 到面板；叠加层是固定面板尺度模板，
    resize 后铺满整个 1200×1920（含补零边框）。若叠加在 resize 之前 / 按输入
    尺度施加，则边框不该被覆盖 —— 整板 = π/2 即证明叠加在 resize 之后、面板尺度.
    """
    slm = Santec(slm_number=1, shift_x=0, shift_y=0)
    small = np.zeros((100, 100))
    slm.set_phase_overlay(np.full_like(_panel(), np.pi / 2))
    out = slm.create_phase_from_array(small)
    assert out.shape == (Santec.Panel_Res[1], Santec.Panel_Res[0])
    # 整板（含补零边框）都 = π/2 → 255
    assert np.all(out == 255)
