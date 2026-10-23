"""Backend tests for the Fresnel focus probe (``--cam-type sim``).

The probe used to construct the panel directly (``Santec(...)``), so the
simulation backend could not reach it. It now goes through the bench factory
for the simulation while the hardware session is left exactly as it was.

What is pinned here
-------------------

1. **the hardware session** -- the panel is still constructed with the same
   arguments, in the same order (panel first, camera second), and the evidence
   file keeps the same field set;
2. **the simulation session** -- the factory is asked for a ``sim`` panel, the
   camera is the simulation camera, and the evidence says *loudly* that the
   winning 1/f is the twin's own value (not a bench measurement);
3. **the overlay/wait call** -- the simulation panel has no LCOS flip time, so
   it must not be handed a wait it cannot honour.

No device is touched: the panel and the camera are fakes.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from ao_shaping.tools.slm.panel import focus_probe as fp

FRAME_SHAPE = (120, 160)


class FakePanel:
    """假面板: 记录调用, 记住最后写上去的相位。"""

    def __init__(self, events: list[tuple[str, Any]]) -> None:
        self.events = events
        self.calls: list[tuple[Any, ...]] = []
        self.last_phase: np.ndarray | None = None

    def __enter__(self) -> "FakePanel":
        self.events.append(("panel_enter", None))
        return self

    def __exit__(self, *exc: Any) -> None:
        self.events.append(("panel_exit", None))

    def create_phase_from_array(self, phase_rad: np.ndarray) -> np.ndarray:
        self.last_phase = np.asarray(phase_rad, dtype=np.float64)
        self.calls.append(("create_phase_from_array", tuple(self.last_phase.shape)))
        return np.zeros(self.last_phase.shape, dtype=np.uint16)

    def display_data(self, gray: np.ndarray, *args: Any, **kwargs: Any) -> int:
        self.calls.append(
            (
                "display_data",
                tuple(args),
                tuple(sorted((str(k), v) for k, v in kwargs.items())),
            )
        )
        return 0


class FakeCamera:
    """假相机: 亮度随瞳面二次相位的幅度下降 (离焦把光摊开)。"""

    def __init__(self, panel: FakePanel, events: list[tuple[str, Any]]) -> None:
        self.panel = panel
        self.events = events
        self.calls: list[tuple[Any, ...]] = []
        self.exposure_ms = 0.1

    def __enter__(self) -> "FakeCamera":
        self.events.append(("camera_enter", None))
        return self

    def __exit__(self, *exc: Any) -> None:
        self.events.append(("camera_exit", None))

    def get_numpy_image(self, n_sample: int = 1, skip_first: bool = False) -> np.ndarray:
        self.calls.append(("get_numpy_image", int(n_sample)))
        rms = (
            float(np.std(self.panel.last_phase))
            if self.panel.last_phase is not None
            else 0.0
        )
        peak = max(20.0, 220.0 - 3.0 * rms)
        frame = np.zeros(FRAME_SHAPE, dtype=np.float32)
        frame[55:65, 75:85] = peak
        return frame

    def reset_exposure_time(self, time_ms: float) -> float:
        self.calls.append(("reset_exposure_time", float(time_ms)))
        self.exposure_ms = float(time_ms)
        return float(time_ms)


def _install_fakes(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[FakePanel, FakeCamera, list[tuple[str, Any]], dict[str, Any]]:
    """把面板与相机换成假的, 并记下构造参数。"""
    from ao_shaping.drivers.ccd import common as ccd_common
    from ao_shaping.drivers.slm import factory as slm_factory
    from ao_shaping.drivers.slm import santec as santec_pkg

    events: list[tuple[str, Any]] = []
    panel = FakePanel(events)
    camera = FakeCamera(panel, events)
    seen: dict[str, Any] = {}

    class _FakeSantec(FakePanel):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(events)
            seen["santec_kwargs"] = dict(kwargs)

    def _open_slm(params: Any) -> FakePanel:
        seen["factory_params"] = params
        return panel

    def _create_camera(
        camera_type: Any,
        cam_id: Any = 0,
        exposure_time_ms: Any = 20.0,
        **kwargs: Any,
    ) -> FakeCamera:
        seen["camera_type"] = camera_type
        seen["camera_id"] = cam_id
        seen["camera_exposure_ms"] = exposure_time_ms
        seen["camera_kwargs"] = dict(kwargs)
        return camera

    monkeypatch.setattr(slm_factory, "open_slm", _open_slm)
    monkeypatch.setattr(ccd_common, "create_camera", _create_camera)
    monkeypatch.setattr(santec_pkg, "Santec", _FakeSantec)
    return panel, camera, events, seen


class TestHardwareSession:
    """硬件路径: 参数、顺序与产物字段一律不动。"""

    def test_the_panel_is_still_constructed_with_the_same_arguments(
        self, monkeypatch, tmp_path
    ):
        _panel, _camera, _events, seen = _install_fakes(monkeypatch)
        result = fp.probe_session_focus(
            slm_number=2,
            cam_type="daheng",
            cam_id=1,
            wavelength_nm=532,
            debug_dir=tmp_path,
        )
        assert "error" not in result, result.get("error")
        # 构造参数逐字不变: 编号 / 波长 / memory 模式 (video_mode=0)。
        assert seen["santec_kwargs"] == {
            "slm_number": 2,
            "wavelength": 532,
            "video_mode": 0,
        }
        # 硬件那条**没有**走工厂。
        assert "factory_params" not in seen

    def test_the_camera_is_constructed_after_the_panel_session(
        self, monkeypatch, tmp_path
    ):
        _panel, _camera, events, seen = _install_fakes(monkeypatch)
        fp.probe_session_focus(cam_type="daheng", cam_id=0, debug_dir=tmp_path)
        names = [name for name, _ in events]
        assert names.index("panel_enter") < names.index("camera_enter")
        assert seen["camera_type"] == "daheng"
        assert seen["camera_kwargs"] == {}

    def test_the_evidence_keeps_the_hardware_field_set(self, monkeypatch, tmp_path):
        _install_fakes(monkeypatch)
        result = fp.probe_session_focus(cam_type="daheng", debug_dir=tmp_path)
        # 硬件那轮的证据文件**不许**因为「加了仿真分支」而多出字段。
        assert "backend" not in result
        assert "result_kind" not in result
        assert "sim_seed" not in result
        assert not any("仿真" in note for note in result.get("notes", []))

    def test_the_evidence_file_is_written(self, monkeypatch, tmp_path):
        _install_fakes(monkeypatch)
        fp.probe_session_focus(cam_type="daheng", debug_dir=tmp_path)
        files = sorted(tmp_path.glob("*.json"))
        assert files, "证据文件必须落盘"
        payload = json.loads(files[0].read_text(encoding="utf-8"))
        assert payload["cam_type"] == "daheng"
        assert "backend" not in payload


class TestSimulationSession:
    """仿真路径: 同一套 CLI, 同一套产物结构, 外加「这是仿真」的标记。"""

    def test_the_factory_is_asked_for_the_simulated_panel(self, monkeypatch, tmp_path):
        _panel, _camera, _events, seen = _install_fakes(monkeypatch)
        fp.probe_session_focus(cam_type="sim", debug_dir=tmp_path)
        assert "factory_params" in seen, "仿真面板必须走工厂"
        assert seen["factory_params"].cam_type == "sim"
        assert seen["factory_params"].slm_number == 1
        assert seen["factory_params"].slm_wavelength == 1064
        # 硬件那条 Santec 构造**没有**被调用。
        assert "santec_kwargs" not in seen

    def test_the_camera_is_the_simulation_camera(self, monkeypatch, tmp_path):
        _panel, _camera, _events, seen = _install_fakes(monkeypatch)
        fp.probe_session_focus(cam_type="sim", debug_dir=tmp_path)
        assert seen["camera_type"] == "sim"

    def test_the_simulated_panel_is_not_asked_to_wait(self, monkeypatch, tmp_path):
        """仿真面板没有翻转时间 ⇒ 只把灰度图交下去, 不带等待。"""
        panel, _camera, _events, _seen = _install_fakes(monkeypatch)
        fp.probe_session_focus(cam_type="sim", debug_dir=tmp_path)
        displays = [call for call in panel.calls if call[0] == "display_data"]
        assert displays, "探针必须显示相位"
        for call in displays:
            assert call[1] == (), "仿真路径不许把 0.4 s 的等待当成位置参数塞进去"
            assert "wait_time_s" not in dict(call[2])

    def test_the_evidence_marks_the_simulation(self, monkeypatch, tmp_path):
        _install_fakes(monkeypatch)
        result = fp.probe_session_focus(cam_type="sim", debug_dir=tmp_path)
        assert result["backend"] == "sim"
        assert result["result_kind"] == "simulation"
        assert result["sim_seed"] == fp.SIM_SEED
        assert result["sim_far_field_padding"] == fp.SIM_FAR_FIELD_PADDING
        assert any("仿真" in note for note in result.get("notes", []))

    def test_the_evidence_file_carries_the_mark(self, monkeypatch, tmp_path):
        _install_fakes(monkeypatch)
        fp.probe_session_focus(cam_type="sim", debug_dir=tmp_path)
        files = sorted(tmp_path.glob("*.json"))
        assert files
        payload = json.loads(files[0].read_text(encoding="utf-8"))
        assert payload["backend"] == "sim"
        assert any("仿真" in note for note in payload.get("notes", []))

    def test_the_same_field_set_is_present_in_both_backends(
        self, monkeypatch, tmp_path
    ):
        """两个后端的产物结构一致 (只是多了几个「这是仿真」的键)。"""
        _install_fakes(monkeypatch)
        hardware = fp.probe_session_focus(cam_type="daheng", debug_dir=tmp_path / "hw")
        _install_fakes(monkeypatch)
        simulation = fp.probe_session_focus(cam_type="sim", debug_dir=tmp_path / "sim")
        assert set(hardware) <= set(simulation)
        for key in ("best_focal_mm", "best_diopter_m1", "points", "range_m1"):
            assert key in hardware
            assert key in simulation
