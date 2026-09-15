"""SantecSLM200 平移功能非硬件测试.

覆盖:
- ``SantecSLM200.shift_phase`` 静态纯函数 (平移数学唯一实现)
- ``SantecSLM200.apply_shift`` 高层编排 (绝对定位/无累积/槽轮换/配置保存)

不触碰硬件: 构造时用 ``sys.modules`` stub 替换 ``_slm_win`` SDK 模块,
编排测试 monkeypatch 驱动内部方法 (``_ensure_open``/``get_displayed_phase``/
``_write_to_memory``/``_display_memory``/``save_config``/``_pick_next_memory_slot``)。
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from ao_shaping.drivers.slm.santec_slm200 import SantecSLM200, VideoMode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _install_sdk_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """用空模块替换 _slm_win, 使 SantecSLM200 构造不需要真实 SLM SDK DLL."""
    stub = types.ModuleType("ao_shaping.drivers.slm._slm_win")
    monkeypatch.setitem(sys.modules, "ao_shaping.drivers.slm._slm_win", stub)


def _phase_with_square(
    width: int = 1920, height: int = 1200, value: int = 512
) -> np.ndarray:
    """参考相位: 中心 64x64 方块 = ``value``, 其余 0 (uint16)."""
    phase = np.zeros((height, width), dtype=np.uint16)
    y0 = height // 2 - 32
    x0 = width // 2 - 32
    phase[y0 : y0 + 64, x0 : x0 + 64] = value
    return phase


def _nonzero_cols(out: np.ndarray) -> np.ndarray:
    return np.flatnonzero(out.sum(axis=0))


def _nonzero_rows(out: np.ndarray) -> np.ndarray:
    return np.flatnonzero(out.sum(axis=1))


def _reference_shift_slice(phase: np.ndarray, shift_x: int, shift_y: int) -> np.ndarray:
    """参考实现: 手工 slice-copy + 零填充 (与旧 GUI ``_apply_shift`` 语义相同)."""
    shifted = np.zeros_like(phase)
    h, w = phase.shape
    y_src_start = max(0, -shift_y)
    y_src_end = min(h, h - shift_y)
    y_dst_start = max(0, shift_y)
    y_dst_end = min(h, h + shift_y)
    x_src_start = max(0, -shift_x)
    x_src_end = min(w, w - shift_x)
    x_dst_start = max(0, shift_x)
    x_dst_end = min(w, w + shift_x)
    if y_src_end > y_src_start and x_src_end > x_src_start:
        shifted[y_dst_start:y_dst_end, x_dst_start:x_dst_end] = phase[
            y_src_start:y_src_end, x_src_start:x_src_end
        ]
    return shifted


# ---------------------------------------------------------------------------
# TestShiftPhase — 纯数学, 零硬件依赖
# ---------------------------------------------------------------------------


class TestShiftPhase:
    """``SantecSLM200.shift_phase`` 静态纯函数."""

    def test_zero_shift_returns_input_object(self) -> None:
        phase = _phase_with_square()
        out = SantecSLM200.shift_phase(phase, 0, 0)
        assert out is phase  # (0,0) 原样返回, 与旧 _apply_shift 语义一致

    def test_positive_x_moves_right_zero_fill(self) -> None:
        phase = _phase_with_square()
        shift = 10
        out = SantecSLM200.shift_phase(phase, shift, 0)
        assert out.dtype == np.uint16
        assert out.shape == phase.shape
        assert np.all(out[:, :shift] == 0)  # 左侧 vacated 填 0
        cols = _nonzero_cols(out)
        # 方块原 col 928..991, 右移 10 后 col 938..1001
        assert cols[0] == 928 + shift

    def test_positive_y_moves_down_zero_fill(self) -> None:
        phase = _phase_with_square()
        shift = 10
        out = SantecSLM200.shift_phase(phase, 0, shift)
        assert np.all(out[:shift, :] == 0)
        rows = _nonzero_rows(out)
        assert rows[0] == 568 + shift

    def test_negative_shifts(self) -> None:
        phase = _phase_with_square()
        out = SantecSLM200.shift_phase(phase, -10, -10)
        cols = _nonzero_cols(out)
        rows = _nonzero_rows(out)
        assert cols[0] == 928 - 10
        assert rows[0] == 568 - 10

    @pytest.mark.parametrize(
        "sx,sy",
        [(7, 0), (0, -13), (5, -5), (-3, 9)],
    )
    def test_matches_reference_slice_shift(self, sx: int, sy: int) -> None:
        """与旧 GUI slice-copy 平移语义字节级一致 (整数平移)."""
        phase = _phase_with_square()
        out = SantecSLM200.shift_phase(phase, sx, sy)
        ref = _reference_shift_slice(phase, sx, sy)
        assert np.array_equal(out, ref)

    def test_dtype_preserved_for_float_input(self) -> None:
        phase = _phase_with_square().astype(np.float64)
        out = SantecSLM200.shift_phase(phase, 3, 3)
        assert out.dtype == np.float64

    def test_large_shift_fully_zero(self) -> None:
        phase = _phase_with_square()
        out = SantecSLM200.shift_phase(phase, 5000, 5000)
        assert np.all(out == 0)


# ---------------------------------------------------------------------------
# TestApplyShift — 编排逻辑, 不触碰硬件
# ---------------------------------------------------------------------------


class TestApplyShift:
    """``SantecSLM200.apply_shift`` 高层编排测试."""

    def _make_slm(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        shift_x: int = 0,
        shift_y: int = 0,
        video_mode: VideoMode = VideoMode.Memory,
    ) -> tuple[SantecSLM200, dict[str, list]]:
        _install_sdk_stub(monkeypatch)
        slm = SantecSLM200(slm_number=1, shift_x=shift_x, shift_y=shift_y)
        slm.video_mode = video_mode

        monkeypatch.setattr(slm, "_ensure_open", lambda: None)
        monkeypatch.setattr(slm, "_pick_next_memory_slot", lambda: 7)

        calls: dict[str, list] = {"write": [], "display": [], "save": []}
        monkeypatch.setattr(
            slm,
            "_write_to_memory",
            lambda phase, slot, memory_mode=0: calls["write"].append(
                (phase.copy(), slot)
            ),
        )
        monkeypatch.setattr(
            slm,
            "_display_memory",
            lambda slot: calls["display"].append(slot),
        )
        monkeypatch.setattr(slm, "save_config", lambda: calls["save"].append(1))
        return slm, calls

    @staticmethod
    def _set_displayed(
        slm: SantecSLM200,
        phase: np.ndarray | None,
        source: str = "内存槽 5",
    ) -> None:
        slm.get_displayed_phase = lambda: (
            (phase.copy() if phase is not None else None, source)
        )

    # ---- absolute positioning ------------------------------------------------

    def test_applies_absolute_shift_and_redisplays(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        slm, calls = self._make_slm(monkeypatch, shift_x=3, shift_y=-2)
        phase = _phase_with_square()
        self._set_displayed(slm, phase)

        result = slm.apply_shift(
            shift_x=8, shift_y=5, wait_time_s=0.0, save_config=True
        )

        assert slm.shift_x == 8 and slm.shift_y == 5
        assert len(calls["write"]) == 1
        written, slot = calls["write"][0]
        assert slot == 7  # 轮换内存槽

        # 绝对定位: 从 current_phase (内含旧偏移 3,-2) 反移旧偏移再施加新偏移
        expected = SantecSLM200.shift_phase(
            SantecSLM200.shift_phase(phase, -3, 2), 8, 5
        )
        assert np.array_equal(written, expected)
        assert calls["display"] == [7]
        assert len(calls["save"]) == 1
        assert result is not None
        assert np.array_equal(result, expected)

    # ---- no accumulation across consecutive calls ----------------------------

    def test_no_accumulation_across_calls(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        slm, calls = self._make_slm(monkeypatch, shift_x=0, shift_y=0)
        phase = _phase_with_square()
        cache: dict = {"phase": phase.copy()}
        slm.get_displayed_phase = lambda: (cache["phase"].copy(), "内存槽 5")

        r1 = slm.apply_shift(shift_x=5, shift_y=4, wait_time_s=0.0, save_config=False)
        cache["phase"] = r1
        calls["write"].clear()

        r2 = slm.apply_shift(shift_x=10, shift_y=6, wait_time_s=0.0, save_config=False)

        assert slm.shift_x == 10 and slm.shift_y == 6
        assert r2 is not None
        # 第二次: displayed=含旧偏移(5,4)的相位, 反移→原始, 再施加新偏移(10,6)
        expected = SantecSLM200.shift_phase(SantecSLM200.shift_phase(r1, -5, -4), 10, 6)
        written, _slot = calls["write"][0]
        assert np.array_equal(written, expected)
        # 累积错误会得到 r1@(10,6)=P@(15,10), 与绝对定位结果不同
        naive = SantecSLM200.shift_phase(r1, 10, 6)
        assert not np.array_equal(written, naive)

    # ---- no cached phase → params only, no rewrite --------------------------

    def test_no_cached_phase_updates_params_only(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        slm, calls = self._make_slm(monkeypatch)
        self._set_displayed(slm, phase=None, source="未缓存")

        result = slm.apply_shift(
            shift_x=-4, shift_y=7, wait_time_s=0.0, save_config=True
        )

        assert result is None
        assert slm.shift_x == -4 and slm.shift_y == 7
        assert calls["write"] == [] and calls["display"] == []
        assert len(calls["save"]) == 1

    # ---- unchanged shift → skip rewrite -------------------------------------

    def test_unchanged_shift_skips_rewrite(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        slm, calls = self._make_slm(monkeypatch, shift_x=6, shift_y=-3)
        phase = _phase_with_square()
        self._set_displayed(slm, phase)

        result = slm.apply_shift(
            shift_x=6, shift_y=-3, wait_time_s=0.0, save_config=True
        )

        assert result is not None
        assert np.array_equal(result, phase)
        assert calls["write"] == [] and calls["display"] == []
        assert len(calls["save"]) == 1  # config still saved

    # ---- DVI mode → params only, no rewrite ---------------------------------

    def test_dvi_mode_updates_params_only(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        slm, calls = self._make_slm(monkeypatch, video_mode=VideoMode.DVI)
        phase = _phase_with_square()
        self._set_displayed(slm, phase, source="直接显示缓存")

        result = slm.apply_shift(
            shift_x=2, shift_y=2, wait_time_s=0.0, save_config=False
        )

        assert slm.shift_x == 2 and slm.shift_y == 2
        assert calls["write"] == [] and calls["display"] == []
        assert result is not None
        assert np.array_equal(result, phase)

    # ---- save_config failure is non-fatal ------------------------------------

    def test_save_config_failure_is_non_fatal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        slm, calls = self._make_slm(monkeypatch, shift_x=1, shift_y=1)
        phase = _phase_with_square()
        self._set_displayed(slm, phase)

        def boom() -> None:
            raise OSError("config 只读")

        monkeypatch.setattr(slm, "save_config", boom)

        result = slm.apply_shift(
            shift_x=9, shift_y=9, wait_time_s=0.0, save_config=True
        )
        assert result is not None
        assert len(calls["write"]) == 1
