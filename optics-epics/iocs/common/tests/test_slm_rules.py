"""SLM 安全规则单测(规则 1 灰度 RAW / 规则 2 槽位轮换)。"""
from __future__ import annotations

import numpy as np
import pytest

from ao_epics_common.slm_rules import (
    DEFAULT_SLOTS,
    GRAYSCALE_MAX,
    GRAYSCALE_MIN,
    SLM_PANEL_HEIGHT,
    SLM_PANEL_WIDTH,
    MemorySlotRotator,
    SlmRuleError,
    flat_phase_grayscale,
    validate_grayscale,
    validate_phase_array,
)


class TestMemorySlotRotator:
    def test_default_slots(self) -> None:
        r = MemorySlotRotator()
        assert r.slots == DEFAULT_SLOTS == (3, 4, 5)

    def test_consecutive_never_repeats(self) -> None:
        r = MemorySlotRotator((3, 4, 5))
        prev = r.next()
        for _ in range(20):  # 多轮循环
            cur = r.next()
            assert cur != prev, "连续两次写入不能同槽位(规则 2)"
            assert cur in (3, 4, 5)
            prev = cur

    def test_cycle_wraps(self) -> None:
        r = MemorySlotRotator((3, 4, 5))
        assert [r.next() for _ in range(6)] == [3, 4, 5, 3, 4, 5]

    def test_accepts_list(self) -> None:
        r = MemorySlotRotator([1, 2])
        assert r.slots == (1, 2)

    def test_requires_at_least_two_slots(self) -> None:
        with pytest.raises(SlmRuleError):
            MemorySlotRotator((3,))

    def test_rejects_out_of_range_slot(self) -> None:
        with pytest.raises(SlmRuleError):
            MemorySlotRotator((3, 0))
        with pytest.raises(SlmRuleError):
            MemorySlotRotator((3, 128))


class TestValidateGrayscale:
    def test_boundaries(self) -> None:
        assert validate_grayscale(GRAYSCALE_MIN) == 0
        assert validate_grayscale(GRAYSCALE_MAX) == 1023

    def test_out_of_range(self) -> None:
        with pytest.raises(SlmRuleError):
            validate_grayscale(-1)
        with pytest.raises(SlmRuleError):
            validate_grayscale(1024)

    def test_float_rounds(self) -> None:
        assert validate_grayscale(100.4) == 100
        assert validate_grayscale(100.6) == 101


class TestFlatPhaseGrayscale:
    def test_shape_dtype(self) -> None:
        arr = flat_phase_grayscale(gray=500)
        assert arr.shape == (SLM_PANEL_HEIGHT, SLM_PANEL_WIDTH)
        assert arr.dtype == np.uint16
        assert np.all(arr == 500)

    def test_custom_size(self) -> None:
        arr = flat_phase_grayscale(height=4, width=8, gray=0)
        assert arr.shape == (4, 8)
        assert arr.dtype == np.uint16

    def test_invalid_gray_raises(self) -> None:
        with pytest.raises(SlmRuleError):
            flat_phase_grayscale(gray=4096)


class TestValidatePhaseArray:
    def test_uint16_passthrough(self) -> None:
        arr = np.zeros((4, 4), dtype=np.uint16)
        out = validate_phase_array(arr)
        assert out.dtype == np.uint16
        assert out is arr

    def test_int32_in_range_converted(self) -> None:
        arr = np.full((4, 4), 512, dtype=np.int32)
        out = validate_phase_array(arr)
        assert out.dtype == np.uint16
        assert np.all(out == 512)

    def test_float_rejected(self) -> None:
        """浮点输入 = 弧度:拒绝(规则 1 防误用)。"""
        arr = np.zeros((4, 4), dtype=np.float64)
        with pytest.raises(SlmRuleError):
            validate_phase_array(arr)

    def test_out_of_range_rejected(self) -> None:
        arr = np.full((4, 4), 2048, dtype=np.int32)
        with pytest.raises(SlmRuleError):
            validate_phase_array(arr)

    def test_1d_rejected(self) -> None:
        arr = np.zeros(16, dtype=np.uint16)
        with pytest.raises(SlmRuleError):
            validate_phase_array(arr)
