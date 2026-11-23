"""``scripts/slm/target_lock_ondevice_20261123.py`` 的离线测试。

全部用假 bench / 临时检查点：不碰驱动、不碰 torch、不碰 Qt。
"""

from __future__ import annotations

import importlib.util
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from ao_shaping.gui.slm.closed_loop_shaping import (
    RoundRecord,
    analysis_grid,
    check_frame,
    sample_on_grid,
    target_field,
)
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
        # σ=2：窗口口径标定后，暗环（窗口外圈）必须不被光斑翼部淹没，
        # 否则峰值"未明显高于暗场"的帧质量门禁会把合格光路也判死。
        self.spot = np.exp(-(((xx - 32.0) ** 2 + (yy - 32.0) ** 2) / (2.0 * 2.0**2)))
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
    # 窗口口径：统计的是 sample_on_grid 后的窗口帧（9×9），不是整帧（64×64）。
    assert sample.pixel_count == 9 * 9
    assert evidence["stats_basis"] == "analysis_window"
    assert evidence["window_shape"] == [9, 9]


def test_run_calibration_thresholds_are_window_scale(tmp_path) -> None:
    """回归（run b 根因）：min_total_counts 按窗口 total 的 2% 推导。

    run b（target_lock_20261123b）整帧统计给 5M px 的 2% ≈ 30,917，而 hold
    监督器看 121×121 窗口图（帧和 ~22k）→ 第 0 帧就 low_signal。标定必须
    与门禁同口径：窗口帧总量级决定阈值量级，阈值远低于窗口帧和。
    """
    args = _calibration_args(tmp_path)
    xy = analysis_grid((32.0, 32.0), 8)
    target = target_field(xy, half_width=4.0, half_height=4.0)
    sample, config, _evidence = mod.run_calibration(
        args, bench=_SyntheticBench(), xy=xy, target=target, grid_size=16,
        out=tmp_path / "calibration",
    )
    # 窗口帧（9×9，含高斯光斑顶部）的帧和：
    saved = np.load(tmp_path / "calibration" / "calibration_frames.npz")
    window_sum = float(np.median([sample_on_grid(f, xy).sum() for f in saved["frames"]]))
    assert config.min_total_counts == pytest.approx(
        max(0.02 * sample.total_counts, 1.0)
    )
    # 口径一致性：sample.total_counts 本身就是窗口帧和的中位数。
    assert sample.total_counts == pytest.approx(window_sum)
    # 阈值必须远低于窗口帧和（否则真实窗口帧必判 low_signal）。
    assert config.min_total_counts < 0.5 * window_sum
    assert sample.pixel_count == 9 * 9  # 窗口尺度，不是 64×64 整帧


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


# --------------------------------------------------------------------------- #
# hold 回调记录：saturated_fraction 必须存在且是真实比例（拒绝/饱和帧也要能落盘）
# --------------------------------------------------------------------------- #


class _Captured:
    def __init__(self, frame: np.ndarray, full_scale: float) -> None:
        self.frame = frame
        self.frame_id = 1
        self.exposure_ms = 20.0
        self.full_scale = full_scale
        self.monotonic_ns = 10**6


def _record_for_frame(frame: np.ndarray, full_scale: float) -> RoundRecord:
    gate = check_frame(frame, full_scale=full_scale, exposure_ms=20.0)
    return RoundRecord(
        round_index=0,
        phase_id="zero",
        frame_id=1,
        exposure_ms=20.0,
        gate=gate,
    )


def test_record_row_reports_real_saturated_fraction_on_rejected_frame() -> None:
    # 首帧即饱和（用户现场：hold 起始帧饱和，监督器正确拒绝）
    frame = np.zeros((64, 64))
    frame[10:14, 10:14] = 4095.0  # 16 px 饱和 → 16/4096
    record = _record_for_frame(frame, full_scale=4095.0)
    assert not record.gate.ok
    assert "saturated" in record.gate.reasons

    row = mod._record_row(_Captured(frame, 4095.0), record)
    assert row["saturated_fraction"] == pytest.approx(16.0 / (64 * 64))
    assert row["gate_ok"] is False
    assert row["gate_reasons"] and "saturated" in row["gate_reasons"]
    # 事件路径拿同一对象，序列化结果不再退回 null
    assert mod._round_to_dict(record)["saturated_fraction"] == pytest.approx(16.0 / 4096)


def test_record_row_reports_zero_when_clean_frame() -> None:
    frame = np.zeros((64, 64))
    frame[30, 30] = 1000.0
    record = _record_for_frame(frame, full_scale=4095.0)
    assert record.gate.ok
    row = mod._record_row(_Captured(frame, 4095.0), record)
    assert row["saturated_fraction"] == 0.0  # 测了是 0 → 0，不是 null


def test_record_row_writes_null_when_inputs_missing() -> None:
    frame = np.zeros((64, 64))
    record = _record_for_frame(frame, full_scale=4095.0)

    class _NoFrame:
        frame_id = 1
        exposure_ms = 20.0
        full_scale = 4095.0
        monotonic_ns = 10**6
        frame = None  # 必要输入不可用

    row = mod._record_row(_NoFrame(), record)
    assert row["saturated_fraction"] is None  # 未测 ≠ 0，不伪造
    # 未经过 on_frame 的事件路径（record 上从未算过）同样保持 null
    fresh = _record_for_frame(frame, full_scale=4095.0)
    assert mod._round_to_dict(fresh)["saturated_fraction"] is None


def test_write_rounds_csv_survives_rejected_saturated_rows(tmp_path) -> None:
    # 回归：此前 on_frame 产出的行缺 saturated_fraction 键，写出直接 KeyError
    saturated = np.zeros((64, 64))
    saturated[10:14, 10:14] = 4095.0
    record = _record_for_frame(saturated, full_scale=4095.0)
    row = mod._record_row(_Captured(saturated, 4095.0), record)

    clean_frame = np.zeros((64, 64))
    clean_frame[30, 30] = 1000.0
    clean_row = mod._record_row(
        _Captured(clean_frame, 4095.0), _record_for_frame(clean_frame, 4095.0)
    )

    path = tmp_path / "rounds.csv"
    mod._write_rounds_csv(path, [row, clean_row])  # 不应抛 KeyError
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert float(rows[0]["saturated_fraction"]) == pytest.approx(16.0 / 4096)
    assert rows[0]["gate_ok"] == "False"
    assert rows[1]["saturated_fraction"] == "0.0"
    assert rows[1]["gate_ok"] == "True"


def test_write_rounds_csv_allows_null_fraction(tmp_path) -> None:
    # 未经 on_frame 的行（必要输入缺失）→ 空列，不伪造 0
    frame = np.zeros((64, 64))
    frame[30, 30] = 1000.0
    record = _record_for_frame(frame, full_scale=4095.0)
    row = mod._round_to_dict(record)
    assert "saturated_fraction" in row
    path = tmp_path / "rounds.csv"
    mod._write_rounds_csv(path, [row])  # 不应抛 KeyError
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["saturated_fraction"] == ""  # null → 空单元格


# --------------------------------------------------------------------------- #
# auto-spot 窗口 + 整数 registration（Lane B）
# --------------------------------------------------------------------------- #


def _spot_frame(cx: float, cy: float, *, sigma: float = 3.0) -> np.ndarray:
    """带暗地板的合成光斑帧，用于测定位；质心故意落在非整数像素。"""
    yy, xx = np.mgrid[0:128, 0:128]
    return 20.0 + 900.0 * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma ** 2)))


def test_locate_spot_xy_centroid_argmax_and_rounded_window_centre() -> None:
    spot = mod.locate_spot_xy(_spot_frame(57.4, 41.7))
    assert spot.centroid_xy[0] == pytest.approx(57.4, abs=0.2)
    assert spot.centroid_xy[1] == pytest.approx(41.7, abs=0.2)
    # argmax 落在离质心最近的整数像素：质心/argmax 一致
    assert spot.argmax_xy == (57.0, 42.0)
    assert abs(spot.argmax_xy[0] - spot.centroid_xy[0]) < 1.0
    assert abs(spot.argmax_xy[1] - spot.centroid_xy[1]) < 1.0
    # 窗口中心 = round(质心)，与 analysis_grid 的 np.rint 同口径
    assert spot.window_centre == (57, 42)
    assert spot.signal_pixels > 0
    assert spot.threshold > spot.dark_level


def test_locate_spot_xy_rejects_frame_without_signal() -> None:
    with pytest.raises(ValueError, match="未检出光斑"):
        mod.locate_spot_xy(np.full((16, 16), 7.0))


def test_spot_registration_integer_dx_dy_and_order_dy_dx() -> None:
    """run b 实测：spot (1227.7,1103.2) vs origin (1304.27,942.63)。"""
    reg = mod.spot_registration((1227.6604290304301, 1103.1570674277286),
                                 (1304.2752685546875, 942.6307373046875))
    assert reg.dx == -76
    assert reg.dy == 160
    # registration_shift_ccd_px 的顺序是 (dy, dx)，不是 (dx, dy)
    assert reg.registration_shift_ccd_px == (160.0, -76.0)
    assert reg.fractional_dx_dy[0] == pytest.approx(-76.61, abs=0.02)
    assert reg.fractional_dx_dy[1] == pytest.approx(160.53, abs=0.02)


def test_integer_shift_grid_is_exact_but_fractional_is_not() -> None:
    """整数平移 + origin 网格 ≡ spot 网格；分数平移留下 >1% 形状差。

    用高斯光斑的解析采样代替 torch forward（本测试文件不碰 torch）：
    整数时两网格逐点相同 → rel-err 恰为 0；分数时采样点错位 ~0.6 px →
    锐利 PSF 的归一化相对误差远大于 1%（探针实测 11-18%）。
    """
    origin = (1304.2752685546875, 942.6307373046875)
    spot = (1227.6604290304301, 1103.1570674277286)
    reg = mod.spot_registration(spot, origin)
    xy_origin = analysis_grid(origin, 120)
    xy_spot = analysis_grid(spot, 120)
    dx_s, dy_s = reg.registration_shift_ccd_px[1], reg.registration_shift_ccd_px[0]

    # analysis_grid 用 np.rint：整数平移后的采样点与 origin 网格逐点相同
    assert np.array_equal(xy_spot - np.array([dx_s, dy_s]), xy_origin)

    def intensity(xy: np.ndarray) -> np.ndarray:
        return np.exp(-(((xy[..., 0] - origin[0]) ** 2
                         + (xy[..., 1] - origin[1]) ** 2) / (2.0 * 2.0 ** 2)))

    base = intensity(xy_origin)
    shifted = intensity(xy_spot - np.array([dx_s, dy_s]))
    assert np.linalg.norm(base - shifted) / np.linalg.norm(base) == pytest.approx(0.0)

    fdx, fdy = reg.fractional_dx_dy
    fractional = intensity(xy_spot - np.array([fdx, fdy]))
    frac_rel = np.linalg.norm(base - fractional) / np.linalg.norm(base)
    assert frac_rel > 0.01  # 实测 ~0.28；严禁分数平移


def test_registered_forward_params_uses_dy_dx_tuple() -> None:
    import dataclasses

    @dataclasses.dataclass(frozen=True)
    class _Params:
        registration_shift_ccd_px: tuple[float, float] | None = None
        grid_size: int = 128

    reg = mod.spot_registration((1227.66, 1103.16), (1304.27, 942.63))
    original = _Params()
    patched = mod.registered_forward_params(original, reg)
    assert patched.registration_shift_ccd_px == (160.0, -76.0)
    assert original.registration_shift_ccd_px is None  # 不原地改冻结参数
    assert patched.grid_size == original.grid_size


def test_build_plan_auto_spot_keeps_origin_as_provisional_when_offline(tmp_path) -> None:
    root = _write_checkpoint(tmp_path / "session")
    args = mod.parse_args(["--out", str(tmp_path / "o"), "--checkpoint", str(root),
                           "--centre", mod.AUTO_SPOT])
    plan = mod.build_plan(args)
    assert plan["window"]["centre_mode"] == "auto-spot"
    assert plan["window"]["centre_is_provisional"] is True
    # 离线计划仍是检查点 origin，执行期会用定位结果覆盖
    assert plan["window"]["centre"] == [1304.0, 942.0]


def test_build_plan_default_and_explicit_centre_unchanged(tmp_path) -> None:
    root = _write_checkpoint(tmp_path / "session")
    default_plan = mod.build_plan(
        mod.parse_args(["--out", str(tmp_path / "o"), "--checkpoint", str(root)])
    )
    assert default_plan["window"]["centre_mode"] == "explicit/auto"
    assert default_plan["window"]["centre_is_provisional"] is False
    assert default_plan["window"]["centre"] == [1304.0, 942.0]

    explicit = mod.build_plan(mod.parse_args(
        ["--out", str(tmp_path / "o"), "--checkpoint", str(root), "--centre", "11,22"]
    ))
    assert explicit["window"]["centre"] == [11.0, 22.0]

    with pytest.raises(ValueError, match="需要 --checkpoint"):
        mod.build_plan(mod.parse_args(
            ["--out", str(tmp_path / "o"), "--centre", mod.AUTO_SPOT]
        ))
