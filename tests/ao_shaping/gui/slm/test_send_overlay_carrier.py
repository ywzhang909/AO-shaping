"""载波成为发送叠加层之后的行为契约。

这个文件针对的是 2026-10-23 报告的那个缺陷：「侧栏『载波』没有叠加倾斜相位」。
根因是载波只作为 ``blaze_period_px`` 喂给整形优化器，由优化器自己在相位里加，
于是侧栏「一次下发」的 PatternHelper 相位从来没带上它。

现在载波是 :func:`build_send_overlay` 的一层，经同一个 ``PhaseOverlay`` 推到句柄，
所以任何走 ``create_phase_from_array`` 的发送都会带上。这里钉住三条：

1. **勾选且周期有效 → 叠加层里有倾斜**（不是空数组、不是全零）；
2. **未勾选 / 周期 -1 / 周期 0 → 不叠加**（0 是「实测自动」，需要相机反馈，
   叠加层没有画面可依，只能明确不加而不是猜）；
3. **与底相位/离焦相加而不是互相覆盖**（多选时三层都生效）。

不碰硬件：overlay 构造是纯 numpy。
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pytest
import streamlit as st

from ao_shaping.gui.slm import bench_settings as bs


@pytest.fixture(autouse=True)
def state(monkeypatch):
    state: dict[str, Any] = {}
    monkeypatch.setattr(st, "session_state", state)
    return state


class _Slm:
    """Just enough device surface for the overlay builders."""

    slm_number = 1
    wavelength = 1064.0
    Pitch_um = 8.0
    _max_gray = 993
    Panel_Res = (1920, 1200)
    is_open = True


def _enable_carrier(state, period: int = 128, direction: str = "vertical") -> None:
    state["slm1_blaze_enabled"] = True
    state["bench_blaze_period_px"] = period
    state["bench_blaze_direction"] = direction


class TestCarrierOverlay:
    def test_enabled_carrier_produces_a_tilt(self, state) -> None:
        _enable_carrier(state)
        out = bs.build_carrier_overlay("slm1", _Slm())
        assert out is not None
        assert out.shape == (1200, 1920)
        # A tilt is a ramp: variation along the grating axis, none across it.
        assert float(np.ptp(out[:, :64])) > 0.5

    def test_the_ramp_runs_along_the_chosen_axis(self, state) -> None:
        _enable_carrier(state, direction="vertical")
        vertical = bs.build_carrier_overlay("slm1", _Slm())
        state["bench_blaze_direction"] = "horizontal"
        horizontal = bs.build_carrier_overlay("slm1", _Slm())
        assert vertical is not None and horizontal is not None
        # "vertical" = ramp along panel x (see OverlayParams docstring).
        assert float(np.ptp(vertical[600, :256])) > float(np.ptp(vertical[:256, 600]))
        assert float(np.ptp(horizontal[:256, 600])) > float(np.ptp(horizontal[600, :256]))

    @pytest.mark.parametrize("period", [-1, 0])
    def test_no_tilt_when_the_period_says_off_or_auto(
        self, state, period: int
    ) -> None:
        """0 = 实测自动 —— 需要相机反馈，叠加层没有画面可依，不能猜一个周期。"""
        _enable_carrier(state, period=period)
        assert bs.build_carrier_overlay("slm1", _Slm()) is None

    def test_no_tilt_when_unchecked(self, state) -> None:
        state["bench_blaze_period_px"] = 128
        assert bs.build_carrier_overlay("slm1", _Slm()) is None


class TestSendOverlayComposition:
    def test_carrier_alone_reaches_the_overlay(self, state) -> None:
        _enable_carrier(state)
        overlay = bs.build_send_overlay("slm1", _Slm())
        assert overlay is not None
        assert overlay.phase_rad is not None
        assert float(np.ptp(overlay.phase_rad[:, :64])) > 0.5

    def test_carrier_and_defocus_stack(self, state) -> None:
        """The user asked for multi-select: selected layers all add, none replaces."""
        _enable_carrier(state)
        carrier_only = bs.build_send_overlay("slm1", _Slm())

        state["slm1_defocus_enabled"] = True
        state["slm1_defocus_amount"] = 10.0
        both = bs.build_send_overlay("slm1", _Slm())

        assert carrier_only is not None and both is not None
        diff = both.phase_rad - carrier_only.phase_rad
        assert np.any(np.abs(diff) > 0.1), "离焦层没有叠加到载波之上"

    def test_carrier_and_base_phase_both_survive(self, state) -> None:
        """底相位在灰度槽、载波在弧度槽 —— 合成分不能把其中一个吃掉。"""
        _enable_carrier(state)
        base = np.full((1200, 1920), 17, dtype=np.uint16)
        state["slm1_base_phase"] = base
        state["slm1_overlay_base"] = True

        overlay = bs.build_send_overlay("slm1", _Slm())
        assert overlay is not None
        assert overlay.phase_rad is not None, "载波层丢失"
        assert overlay.base_gray is not None, "底相位层丢失"

    def test_nothing_selected_means_no_overlay(self, state) -> None:
        assert bs.build_send_overlay("slm1", _Slm()) is None


class _RecordingHandle:
    """Stand-in handle that records what the overlay funnel pushed onto it."""

    slm_number = 1
    is_open = True
    wavelength = 1064.0
    Pitch_um = 8.0
    _max_gray = 993
    Panel_Res = (1920, 1200)

    def __init__(self) -> None:
        self.phase_overlay = "unset"
        self.base_phase = "unset"

    def set_phase_overlay(self, arr) -> None:
        self.phase_overlay = None if arr is None else np.asarray(arr)

    def set_base_phase(self, arr) -> None:
        self.base_phase = None if arr is None else np.asarray(arr)


class TestPushedToTheHandle:
    """``_apply_slm_overlays`` is what makes the send path actually carry it."""

    def test_carrier_checkbox_pushes_a_tilt_onto_the_handle(self, state) -> None:
        from ao_shaping.gui.slm import multi_slm_controller as gui

        handle = _RecordingHandle()
        state["slm1"] = handle
        _enable_carrier(state)

        gui._apply_slm_overlays(1)

        assert handle.phase_overlay is not None
        assert float(np.ptp(handle.phase_overlay[:, :64])) > 0.5

    def test_unchecking_clears_the_tilt(self, state) -> None:
        from ao_shaping.gui.slm import multi_slm_controller as gui

        handle = _RecordingHandle()
        state["slm1"] = handle
        _enable_carrier(state)
        gui._apply_slm_overlays(1)
        assert handle.phase_overlay is not None

        state["slm1_blaze_enabled"] = False
        gui._apply_slm_overlays(1)
        assert handle.phase_overlay is None

    def test_base_phase_gating_is_independent(self, state) -> None:
        """底相位勾选与载波勾选互不影响（多选语义）。"""
        from ao_shaping.gui.slm import multi_slm_controller as gui

        handle = _RecordingHandle()
        state["slm1"] = handle
        _enable_carrier(state)
        state["slm1_base_phase"] = np.full((1200, 1920), 5, dtype=np.uint16)
        state["slm1_overlay_base"] = True

        gui._apply_slm_overlays(1)

        assert handle.phase_overlay is not None
        assert handle.base_phase is not None


class TestShapingDoesNotDoubleApply:
    def test_shaping_params_carry_no_blaze_of_their_own(self) -> None:
        """载波由叠加层施加；优化器再传一次就会把倾斜加两遍。

        该层在优化器打开句柄时装到它的 SLM 上，所以整轮优化与最终相位都带载波，
        口径一致；``blaze_period_px`` 必须保持 -1。
        """
        from ao_shaping.gui.slm import shaping_controls as sc

        values = dict(sc.GS_REFINE_RECIPE)
        values["slm_number"] = 1
        values["blaze_period_px"] = 128  # 侧栏里选了一个具体周期
        values["bench_prefix"] = "slm1"

        gs = sc.GsRefineMethod("slm1", 1).build_params(values)
        assert gs.blaze_period_px == -1

        mil_values = {
            "n_rounds": 1,
            "seed": 0,
            "target_shape": "square",
            "target_side": 60,
            "w_uniformity": 0.4,
            "w_efficiency": 0.6,
            "probe_count": 6,
            "probe_spread": 4.0,
            "step_a_iterations": 80,
            "step_b_iterations": 300,
            "holdout_probe_count": 2,
            "min_holdout_correlation": 0.5,
            "region": 64,
            "far_field_padding": 8,
            "panel_span_px": 900,
            "pupil_center": "960,720",
            "cam_type": "daheng",
            "cam_id": 0,
            "camera_pixel_um": 2.2,
            "exposure_time_ms": 0.2,
            "max_exposure_ms": 0.6,
            "cam_size": 300,
            "slm_number": 1,
            "blaze_period_px": 128,
            "blaze_direction": "vertical",
            "bench_prefix": "slm1",
        }
        mil = sc.ModelInLoopMethod("slm1", 1).build_params(mil_values)
        assert mil.blaze_period_px == -1


class TestTransferSkipsTheSendOverlay:
    """迁移模型的相位自带 ``ramp_matrix`` 载波，所以它**不接**侧栏发送叠加层。

    接了这一层就是把同一个倾斜加两遍（0 级会跑两倍远），而且 ``transfer.validate.run``
    在测量期间还会主动清掉驱动叠加层 —— 运行期与下发期的口径必须一致。
    """

    def test_the_flag_is_an_explicit_opt_out(self) -> None:
        from ao_shaping.gui.slm import shaping_controls as sc

        assert sc.TransferMethod.applies_send_overlay is False
        # The base default stays True: the flag is opt-out, not opt-in.
        assert sc.ShapingMethod.applies_send_overlay is True
        assert sc.GsRefineMethod.applies_send_overlay is True
        assert sc.ModelInLoopMethod.applies_send_overlay is True

    def test_prepare_config_never_assigns_send_overlay(self, monkeypatch) -> None:
        from ao_shaping.gui.slm import shaping_controls as sc
        from ao_shaping.tools.slm.transfer.workflow import TransferSession

        # ``TransferSession`` is frozen and has no such field, so a base-class
        # ``prepare_config`` (which assigns ``config.send_overlay``) could only
        # crash here — nothing to catch it at the mapping layer.
        assert "send_overlay" not in {f.name for f in dataclasses.fields(TransferSession)}

        assigned: dict[str, Any] = {}

        class _RecordingConfig:
            """Records every attribute write (a frozen dataclass cannot)."""

            stages = ["capture", "fit", "summary", "direction_check"]

            def __init__(self) -> None:
                object.__setattr__(self, "assigned", assigned)

            def plan(self) -> dict[str, Any]:
                return {"stages": list(self.stages)}

            def __setattr__(self, name, value) -> None:
                self.assigned[name] = value
                object.__setattr__(self, name, value)

        monkeypatch.setattr(st, "progress", lambda value, text=None: None)

        config = _RecordingConfig()
        sc.TransferMethod("slm1", 1).prepare_config(config, {"bench_prefix": "slm1"})

        assert "send_overlay" not in assigned, assigned

        # Guard the guard: the same recorder must catch the assignment on a
        # method that *does* take the overlay, otherwise this test could pass
        # because the recorder observes nothing at all.
        gs_config = _RecordingConfig()
        sc.ShapingMethod.prepare_config(
            sc.GsRefineMethod("slm1", 1), gs_config, {"bench_prefix": "slm1"}
        )
        assert "send_overlay" in gs_config.assigned
