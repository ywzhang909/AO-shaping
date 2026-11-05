"""采集段历史/理想模型预测：本次定位与几何、网格转换及 best-effort 降级。"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from ao_shaping.algorithm.signal_processing import (
    differentiable_defocus_uniformity as model_module,
)
from ao_shaping.drivers.ccd import common
from ao_shaping.tools.slm.panel import focus_probe
from ao_shaping.tools.slm.transfer import (
    capture as capture_module,
    fit as fit_module,
    hardware,
)


class _Camera:
    """常亮、无饱和的小图，只用于驱动真实 capture.read 漏斗。"""

    def __init__(self) -> None:
        self.readback_ms = 0.123

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def open(self) -> None:
        pass

    def close(self) -> None:
        pass

    def get_numpy_image(self, n_sample=1, skip_first=False):
        yy, xx = np.mgrid[:50, :60]
        return (1 + 80 * np.exp(-((xx - 30) ** 2 + (yy - 25) ** 2) / 20)).astype(
            np.uint8
        )

    def reset_exposure_time(self, value) -> None:
        self.readback_ms = value


class _Slm:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def open(self) -> None:
        pass

    def close(self) -> None:
        pass

    def set_phase_overlay(self, phase) -> None:
        pass

    def create_phase_from_array(self, phase):
        return np.zeros(phase.shape, dtype=np.uint16)

    def display_data(self, gray) -> None:
        pass


@dataclass
class _Setting:
    status: str = "ok"
    exposure_ms: float = 0.123
    peak: float = 0.0
    fraction: float = 0.48
    reads: int = 1


class _ArrayTensor:
    """predict 返回值的极小 torch 协议替身，避免测试分配真实模型。"""

    def __init__(self, array: np.ndarray) -> None:
        self.array = array

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array


class _RecordingForward:
    instances = []

    def __init__(self, params, camera_xy, *, device=None) -> None:
        self.params = params
        self.camera_xy = camera_xy
        self.device = device
        self.inputs = []
        self.__class__.instances.append(self)

    def predict(self, phase):
        self.inputs.append(np.array(phase, copy=True))
        return [_ArrayTensor(np.full((2, 3), 7, dtype=np.float32))]


def _checkpoint(tmp_path, *, grid_size: int):
    path = tmp_path / "frozen_parameters.json"
    path.write_text(
        json.dumps(
            dict(
                defocus_diopter=0.1,
                origin=[999.0, 888.0],
                amplitude_center=[32.0, 32.0],
                amplitude_width=[20.0, 20.0],
                residual_noll_radians={},
                gain=1.0,
                background=0.0,
                sigma_ccd_px=0.0,
                ramp_matrix=[[0.0, 5000.0], [-5000.0, 0.0]],
                grid_size=grid_size,
                fft_size=64,
                pixel_samples=1,
            )
        ),
        encoding="utf-8",
    )
    return path


def _geometry_rows(matrix: np.ndarray) -> list[dict]:
    """由指定几何反推 signed-ramp 光斑，避免测试另造一套几何公式。"""
    rows = []
    origin = np.array([23.0, 31.0])
    for axis_index, axis in enumerate(("x", "y")):
        for period in (64, 128):
            for sign in (-1, 1):
                rows.append(
                    dict(
                        label=f"ramp_{axis}_{period}_{sign}",
                        spot_xy=(
                            origin + sign * matrix[:, axis_index] / period
                        ).tolist(),
                        exposure_ms=0.123,
                        peak=80,
                    )
                )
    return rows


def _patch_capture(monkeypatch):
    camera = _Camera()
    monkeypatch.setattr(hardware, "prepare_camera_sdk", lambda _: None)
    monkeypatch.setattr(common, "create_camera", lambda *args, **kwargs: camera)
    monkeypatch.setattr(common, "get_camera_exposure_range", lambda _: (0.02, 10.0))
    monkeypatch.setattr(common, "get_camera_exposure_ms", lambda cam: cam.readback_ms)
    monkeypatch.setitem(
        sys.modules,
        "ao_shaping.drivers.slm.santec",
        SimpleNamespace(Santec=lambda **kwargs: _Slm()),
    )
    monkeypatch.setattr(capture_module, "locate", lambda frame: (23.0, 31.0))

    def adapt(setter, read, **kwargs):
        setter(0.123)
        read()
        return _Setting()

    monkeypatch.setattr(focus_probe, "adapt_exposure", adapt)
    monkeypatch.setattr(capture_module.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        capture_module,
        "probes",
        lambda *args, **kwargs: iter(
            (
                ("reference_start", "calibration", np.zeros((16, 16), np.float32)),
                ("probe_000_test", "train", np.ones((16, 16), np.float32)),
            )
        ),
    )


def _capture_args(tmp_path, checkpoint=None):
    return SimpleNamespace(
        out=tmp_path / "capture",
        camera="custom",
        cam_id=0,
        slm_number=1,
        grid=16,
        count=1,
        frames=1,
        seed=1,
        execute=True,
        exposure_max_steps=2,
        save_dataset=False,
        checkpoint=checkpoint,
        device="cpu",
    )


def test_reference_recentres_historical_forward_once(tmp_path, monkeypatch):
    _RecordingForward.instances = []
    monkeypatch.setattr(model_module, "FrozenDefocusForward", _RecordingForward)
    predictor = capture_module._LivePrediction(
        _checkpoint(tmp_path, grid_size=16), capture_grid=16, device="cpu"
    )

    predictor.attach_reference({"spot_xy": [23.0, 31.0]})
    predictor.attach_reference({"spot_xy": [1000.0, 1000.0]})

    assert len(_RecordingForward.instances) == 1
    forward = _RecordingForward.instances[0]
    assert forward.device == "cpu"
    # forward 的频率映射也使用 origin；不仅建窗，参数原点同样须换成本次值。
    assert forward.params.origin_xy == (23.0, 31.0)
    np.testing.assert_allclose(forward.camera_xy[..., 0].mean(), 23.0)
    np.testing.assert_allclose(forward.camera_xy[..., 1].mean(), 31.0)


def test_different_capture_grid_uses_the_fit_panel_convention(tmp_path, monkeypatch):
    _RecordingForward.instances = []
    monkeypatch.setattr(model_module, "FrozenDefocusForward", _RecordingForward)
    predictor = capture_module._LivePrediction(
        _checkpoint(tmp_path, grid_size=32), capture_grid=16, device="cpu"
    )
    predictor.attach_reference({"spot_xy": [23.0, 31.0]})

    predicted = predictor.predict(np.full((16, 16), 2.5, dtype=np.float32))

    forward = _RecordingForward.instances[0]
    assert forward.inputs[0].shape == (32, 32)
    expected = np.concatenate((np.full((20, 32), 2.5), np.zeros((12, 32)))).astype(
        np.float32
    )
    np.testing.assert_allclose(forward.inputs[0], expected)
    np.testing.assert_array_equal(predicted, np.full((2, 3), 7, dtype=np.float32))


def test_no_checkpoint_builds_ideal_model_from_current_reference_and_ramps(monkeypatch):
    """无历史参数时用本次平场与 signed-ramp 直接构造零校正理想模型。"""
    _RecordingForward.instances = []
    monkeypatch.setattr(model_module, "FrozenDefocusForward", _RecordingForward)
    predictor = capture_module._LivePrediction(None, capture_grid=16, device="cpu")
    predictor.attach_reference({"spot_xy": [23.0, 31.0]})
    assert predictor.predict(np.zeros((16, 16), np.float32)) is None  # 几何尚未到齐

    matrix = np.array([[2.0, 0.5], [-0.25, 3.0]])
    predictor.attach_geometry(_geometry_rows(matrix))

    assert len(_RecordingForward.instances) == 1
    forward = _RecordingForward.instances[0]
    params = forward.params
    assert params.origin_xy == (23.0, 31.0)
    assert params.amplitude_center_xy == (960.0, 600.0)
    assert params.amplitude_width_xy == (400.0, 300.0)
    assert params.defocus_diopter == 0.0
    assert params.residual_noll_radians == {j: 0.0 for j in range(5, 12)}
    assert params.gain == 1.0
    assert params.background == 0.0
    assert params.sigma_ccd_px == 0.0
    assert params.grid_size == 16
    np.testing.assert_allclose(params.ramp_matrix, matrix)
    np.testing.assert_allclose(forward.camera_xy[..., 0].mean(), 23.0)
    np.testing.assert_allclose(forward.camera_xy[..., 1].mean(), 31.0)
    np.testing.assert_array_equal(predictor.predict(np.zeros((16, 16), np.float32)), 7)


def test_probe_frames_carry_predicted_after_reference(tmp_path, monkeypatch):
    _patch_capture(monkeypatch)
    checkpoint = _checkpoint(tmp_path, grid_size=16)
    instances = []

    class Prediction:
        def __init__(self, path, *, capture_grid, device) -> None:
            self.attached = None
            self.capture_grid = capture_grid
            self.device = device
            instances.append(self)

        def attach_reference(self, row) -> None:
            self.attached = row["spot_xy"]

        def predict(self, phase_grid):
            if self.attached is None:
                return None
            return np.full((2, 2), float(phase_grid.mean()), dtype=np.float32)

    monkeypatch.setattr(capture_module, "_LivePrediction", Prediction)
    frames = []
    capture_module.capture(_capture_args(tmp_path, checkpoint), on_frame=frames.append)

    reference = [
        frame for frame in frames if frame["label"].endswith("reference_start")
    ]
    probes = [frame for frame in frames if frame["label"].endswith("probe_000_test")]
    assert reference and all(frame["predicted"] is None for frame in reference)
    assert probes and all(frame["predicted"] is not None for frame in probes)
    assert all(frame["phase_grid"].shape == (16, 16) for frame in probes)
    assert instances[0].attached == [23.0, 31.0]
    assert instances[0].capture_grid == 16
    assert instances[0].device == "cpu"


def test_cold_start_emits_ideal_prediction_as_soon_as_ramp_geometry_is_ready(
    tmp_path, monkeypatch
):
    _patch_capture(monkeypatch)
    _RecordingForward.instances = []
    monkeypatch.setattr(model_module, "FrozenDefocusForward", _RecordingForward)
    # 此测试钉住 capture 的建模特时序；几何数值本身由上一测试用真实 geometry 验证。
    monkeypatch.setattr(fit_module, "geometry", lambda rows: (np.eye(2), {}))
    zero = np.zeros((16, 16), np.float32)
    plan = [("reference_start", "calibration", zero)]
    for axis in ("x", "y"):
        for period in (64, 128):
            for sign in (-1, 1):
                plan.append((f"ramp_{axis}_{period}_{sign}", "calibration", zero))
    plan.append(("probe_000_test", "train", np.ones((16, 16), np.float32)))
    monkeypatch.setattr(capture_module, "probes", lambda *args, **kwargs: iter(plan))

    frames = []
    capture_module.capture(_capture_args(tmp_path), on_frame=frames.append)

    reference = [
        frame for frame in frames if frame["label"].endswith("reference_start")
    ]
    ramps = [frame for frame in frames if "ramp_" in frame["label"]]
    probes = [frame for frame in frames if frame["label"].endswith("probe_000_test")]
    assert reference and all(frame["predicted"] is None for frame in reference)
    assert ramps and all(
        frame["predicted"] is None for frame in ramps
    )  # 等几何，非永久留空
    assert probes and all(frame["predicted"] is not None for frame in probes)
    assert all(frame["predicted"].shape == (2, 3) for frame in probes)
    assert len(_RecordingForward.instances) == 1  # 参数/xy/forward 均只建一次
    assert _RecordingForward.instances[0].params.origin_xy == (23.0, 31.0)
    np.testing.assert_allclose(
        _RecordingForward.instances[0].params.ramp_matrix, np.eye(2)
    )


def test_prediction_failure_degrades_without_aborting_capture(tmp_path, monkeypatch):
    _patch_capture(monkeypatch)
    checkpoint = _checkpoint(tmp_path, grid_size=16)
    warnings = []
    failures = []

    class BrokenPrediction:
        def __init__(self, *args, **kwargs) -> None:
            self.attached = False

        def attach_reference(self, row) -> None:
            self.attached = True

        def predict(self, phase_grid):
            if not self.attached:
                return None
            failures.append(1)
            raise RuntimeError("forward failed")

    monkeypatch.setattr(capture_module, "_LivePrediction", BrokenPrediction)
    monkeypatch.setattr(capture_module.logger, "warning", warnings.append)
    frames = []

    capture_module.capture(_capture_args(tmp_path, checkpoint), on_frame=frames.append)

    assert (
        len(
            json.loads((tmp_path / "capture/manifest.json").read_text(encoding="utf-8"))
        )
        == 2
    )
    assert len(failures) == 1  # 首次失败后停用，不在后续每帧重复失败。
    assert all(frame["predicted"] is None for frame in frames)
    assert any("forward failed" in message for message in warnings)
