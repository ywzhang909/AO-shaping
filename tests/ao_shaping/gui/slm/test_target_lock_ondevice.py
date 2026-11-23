"""``scripts/slm/target_lock_ondevice_20261123.py`` 的离线测试。

全部用假 bench / 临时检查点：不碰驱动、不碰 torch、不碰 Qt。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from ao_shaping.gui.slm.closed_loop_shaping import analysis_grid, target_field
from ao_shaping.gui.slm.target_lock import target_on_frame

_REPO = Path(__file__).resolve().parents[4]
_SCRIPT = _REPO / "scripts" / "slm" / "target_lock_ondevice_20261123.py"

spec = importlib.util.spec_from_file_location("target_lock_ondevice", _SCRIPT)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod  # dataclass 处理要按 __module__ 回查 sys.modules
spec.loader.exec_module(mod)


def _write_checkpoint(root: Path, *, span: int | None = 120) -> Path:
    fit = root / "fit"
    fit.mkdir(parents=True)
    (fit / "frozen_parameters.json").write_text(
        json.dumps({"origin": [1304.0, 942.0], "grid_size": 128}),
        encoding="utf-8",
    )
    if span is not None:
        (fit / "provenance.json").write_text(
            json.dumps({"window_span_px": span}), encoding="utf-8"
        )
    return root


def _plan_args(out: Path, *extra: str) -> object:
    return mod.parse_args(["--out", str(out), *extra])


# --------------------------------------------------------------------------- #
# target_on_frame
# --------------------------------------------------------------------------- #


def test_target_on_frame_places_window_on_full_frame() -> None:
    xy = analysis_grid((32.0, 32.0), 4)  # 坐标 28..36，步长 2
    target = target_field(xy, half_width=2.0, half_height=2.0)
    frame = target_on_frame(xy, target, (64, 64))
    assert frame.shape == (64, 64)
    yi = np.rint(xy[..., 1]).astype(int)
    xi = np.rint(xy[..., 0]).astype(int)
    assert np.array_equal(frame[yi, xi], target)
    assert frame[0, 0] == 0.0  # 窗口外必须是 0，暗场掩膜才落在背景上


def test_target_on_frame_rejects_out_of_bounds_and_shape_mismatch() -> None:
    xy = analysis_grid((32.0, 32.0), 4)
    target = np.ones(xy.shape[:2])
    with pytest.raises(ValueError, match="超出帧范围"):
        target_on_frame(xy, target, (16, 16))
    with pytest.raises(ValueError, match="不一致"):
        target_on_frame(xy, np.ones((3, 3)), (64, 64))


# --------------------------------------------------------------------------- #
# DriftBench / PreopenedBench
# --------------------------------------------------------------------------- #


class _RecordingBench:
    def __init__(self) -> None:
        self.phases: list[np.ndarray] = []
        self.opened = 0
        self.closed = 0

    def open(self) -> None:
        self.opened += 1

    def describe(self) -> dict:
        return {"name": "fake"}

    def apply_phase(self, phase: np.ndarray) -> str:
        self.phases.append(np.asarray(phase, dtype=np.float64).copy())
        return f"phase-{len(self.phases)}"

    def capture(self):  # pragma: no cover - 漂移测试不采帧
        raise NotImplementedError

    def close(self) -> None:
        self.closed += 1


def test_drift_bench_ramps_linear_tilt_and_logs_each_write() -> None:
    inner = _RecordingBench()
    bench = mod.DriftBench(
        inner, axis="tilt-x", total_rad=2.0, start_write=1, ramp_writes=2, grid_size=4
    )
    zero = np.zeros((4, 4))
    for _ in range(4):
        bench.apply_phase(zero)

    # write0：未开始；write1：起点；write2：一半；write3：满幅。
    assert [float(p.max()) for p in inner.phases] == [0.0, 0.0, 1.0, 2.0]
    ramp = inner.phases[-1]
    assert np.allclose(ramp[0], [0.0, 2.0 / 3.0, 4.0 / 3.0, 2.0])  # 沿列（x）张开
    assert np.allclose(ramp[:, 0], 0.0)
    assert [row["peak_to_peak_rad"] for row in bench.trace] == [0.0, 0.0, 1.0, 2.0]
    assert bench.describe()["drift"]["axis"] == "tilt-x"


def test_drift_bench_tilt_y_runs_along_rows() -> None:
    inner = _RecordingBench()
    bench = mod.DriftBench(
        inner, axis="tilt-y", total_rad=1.0, start_write=0, ramp_writes=1, grid_size=4
    )
    bench.apply_phase(np.zeros((4, 4)))  # write0：progress=0，尚未张开
    bench.apply_phase(np.zeros((4, 4)))  # write1：满幅
    ramp = inner.phases[-1]
    assert np.allclose(ramp[:, 0], [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
    with pytest.raises(ValueError, match="不一致"):
        bench.apply_phase(np.zeros((2, 2)))


def test_preopened_bench_skips_open_but_closes_once() -> None:
    inner = _RecordingBench()
    bench = mod.PreopenedBench(inner)
    bench.open()
    assert inner.opened == 0
    bench.apply_phase(np.zeros((2, 2)))
    assert len(inner.phases) == 1
    bench.close()
    assert inner.closed == 1


# --------------------------------------------------------------------------- #
# 检查点解析与计划
# --------------------------------------------------------------------------- #


def test_resolve_checkpoint_reads_origin_grid_and_provenance_span(tmp_path) -> None:
    root = _write_checkpoint(tmp_path / "session")
    spec = mod.resolve_checkpoint(root, span=None)
    assert spec.params_path == root / "fit" / "frozen_parameters.json"
    assert spec.session_dir == root
    assert spec.span == 120
    assert spec.centre == (1304.0, 942.0)
    assert spec.grid_size == 128


def test_resolve_checkpoint_requires_frozen_parameters(tmp_path) -> None:
    with pytest.raises(ValueError, match="frozen_parameters"):
        mod.resolve_checkpoint(tmp_path, span=None)


def test_build_plan_rejects_invalid_inputs(tmp_path) -> None:
    with pytest.raises(ValueError, match="checkpoint"):
        mod.build_plan(_plan_args(tmp_path / "o"))  # both 必须有检查点
    with pytest.raises(ValueError, match="centre"):
        mod.build_plan(_plan_args(tmp_path / "o", "--stage", "calibrate"))
    with pytest.raises(ValueError, match="frames"):
        mod.build_plan(
            _plan_args(tmp_path / "o", "--stage", "calibrate", "--centre", "32,32", "--frames", "1")
        )
    root = _write_checkpoint(tmp_path / "session")
    with pytest.raises(ValueError, match="target-path"):
        mod.build_plan(_plan_args(tmp_path / "o", "--checkpoint", str(root), "--target", "npy"))
    with pytest.raises(ValueError, match="偶数"):
        mod.build_plan(_plan_args(tmp_path / "o", "--checkpoint", str(root), "--span", "9"))


def test_build_plan_rejects_adapt_without_dataset(tmp_path) -> None:
    root = _write_checkpoint(tmp_path / "session")
    with pytest.raises(ValueError, match="source-dataset"):
        mod.build_plan(
            _plan_args(tmp_path / "o", "--checkpoint", str(root), "--adapt")
        )


def test_build_plan_warns_when_drift_without_adaptation(tmp_path) -> None:
    root = _write_checkpoint(tmp_path / "session")
    plan = mod.build_plan(
        _plan_args(tmp_path / "o", "--checkpoint", str(root), "--drift", "tilt-x")
    )
    assert any("--adapt" in warning for warning in plan["warnings"])
    assert plan["checkpoint"]["span_from_checkpoint"] == 120
    assert plan["window"]["centre"] == [1304.0, 942.0]


def test_main_plan_only_creates_no_directory(tmp_path, capsys) -> None:
    root = _write_checkpoint(tmp_path / "session")
    out = tmp_path / "not_created"
    code = mod.main(["--out", str(out), "--checkpoint", str(root)])
    assert code == 0
    assert not out.exists()
    assert "plan-only" in capsys.readouterr().out


def test_main_reports_invalid_plan_with_exit_code_2(tmp_path, capsys) -> None:
    code = mod.main(["--out", str(tmp_path / "o")])
    assert code == 2
    assert "计划无效" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# 标定阶段（假 bench + 合成帧）
# --------------------------------------------------------------------------- #


class _SyntheticBench:
    """零相位下给一个高斯光斑 + 背景噪声；帧质量可由参数控制。"""

    def __init__(self, *, peak: float = 1000.0, background: float = 100.0,
                 full_scale: float = 4095.0, noise: float = 1.0) -> None:
        self.peak = peak
        self.background = background
        self.full_scale = full_scale
        self.noise = noise
        self.frame_id = 0
        yy, xx = np.mgrid[0:64, 0:64]
        self.spot = np.exp(-(((xx - 32.0) ** 2 + (yy - 32.0) ** 2) / (2.0 * 4.0**2)))
        self.rng = np.random.default_rng(7)

    def apply_phase(self, phase: np.ndarray) -> str:
        return "zero"

    def capture(self):
        self.frame_id += 1
        frame = self.background + self.peak * self.spot
        frame = frame + self.rng.normal(0.0, self.noise, frame.shape)
        frame = np.clip(frame, 0.0, self.full_scale)

        class Captured:
            pass

        captured = Captured()
        captured.frame = frame
        captured.frame_id = self.frame_id
        captured.exposure_ms = 20.0
        captured.full_scale = self.full_scale
        captured.monotonic_ns = self.frame_id * 10**6
        return captured


def _calibration_args(tmp_path: Path):
    return _plan_args(
        tmp_path / "out", "--stage", "calibrate", "--centre", "32,32", "--span", "8"
    )


def test_run_calibration_derives_config_and_saves_frames(tmp_path) -> None:
    args = _calibration_args(tmp_path)
    xy = analysis_grid((32.0, 32.0), 8)
    target = target_field(xy, half_width=4.0, half_height=4.0)
    sample, config, evidence = mod.run_calibration(
        args, bench=_SyntheticBench(), xy=xy, target=target, grid_size=16,
        out=tmp_path / "calibration",
    )
    assert sample.dark_sigma > 0.0
    assert config.min_peak_counts > sample.dark_mean
    assert config.max_centroid_error_px == pytest.approx(5.0 / 2.0)  # 容差 5 px / 步长 2 px
    saved = np.load(tmp_path / "calibration" / "calibration_frames.npz")
    assert saved["frames"].shape == (8, 64, 64)
    on_disk = json.loads((tmp_path / "calibration" / "calibration.json").read_text("utf-8"))
    assert on_disk["lock_config"]["min_peak_counts"] == pytest.approx(config.min_peak_counts)

    reloaded = mod._load_lock_config(tmp_path / "calibration" / "calibration.json")
    assert reloaded.max_centroid_error_px == pytest.approx(config.max_centroid_error_px)


def test_run_calibration_keeps_raw_frame_dtype_on_disk(tmp_path) -> None:
    args = _calibration_args(tmp_path)
    xy = analysis_grid((32.0, 32.0), 8)
    target = target_field(xy, half_width=4.0, half_height=4.0)

    class UInt16Bench(_SyntheticBench):
        def __init__(self) -> None:
            self._inner = _SyntheticBench()

        def capture(self):
            captured = self._inner.capture()
            captured.frame = captured.frame.astype(np.uint16)
            return captured

    mod.run_calibration(
        args, bench=UInt16Bench(), xy=xy, target=target, grid_size=16,
        out=tmp_path / "calibration",
    )
    saved = np.load(tmp_path / "calibration" / "calibration_frames.npz")
    assert saved["frames"].dtype == np.uint16


def test_run_calibration_rejects_saturated_frames(tmp_path) -> None:
    args = _calibration_args(tmp_path)
    xy = analysis_grid((32.0, 32.0), 8)
    target = target_field(xy, half_width=4.0, half_height=4.0)
    bench = _SyntheticBench(peak=0.0, background=4095.0, noise=0.0)  # 全帧顶满
    with pytest.raises(mod.FrameQualityError, match="饱和"):
        mod.run_calibration(
            args, bench=bench, xy=xy, target=target, grid_size=16,
            out=tmp_path / "calibration",
        )
    # 失败也要留下证据，便于事后复核"为什么不能用"。
    assert (tmp_path / "calibration" / "calibration.json").is_file()


def test_summarise_ignores_nan_lock_errors() -> None:
    events = [
        {"kind": "lock", "payload": {"action": "hold", "centroid_error_ccd_px": float("nan")}},
        {"kind": "lock", "payload": {"action": "correct", "centroid_error_ccd_px": 1.5}},
    ]
    stats = mod._summarise([], events)
    assert stats["centroid_error_max_ccd_px"] == pytest.approx(1.5)
    assert stats["actions"] == {"hold": 1, "correct": 1}
