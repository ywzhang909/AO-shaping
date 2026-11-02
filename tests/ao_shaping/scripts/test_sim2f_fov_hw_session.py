"""Tests for the resumable one-camera sim2f FOV session script.

The script is the seam between a single hardware session and the offline work
that follows it, so these tests pin the properties that make that seam safe:

* ``plan`` is deterministic (byte-identical phases and ``plan.json``);
* ``capture`` resumes: it skips valid frames, re-captures missing ones, and
  re-captures a corrupt ``.npy`` instead of aborting the whole session;
* ``finalize`` writes the repository's standard recording, and
  ``load_bench_session`` can read it back;
* ``analyze`` produces finite metrics on a dry session and says
  ``not_implemented`` rather than inventing the parts it does not do.

Everything here runs on ``SimInstrument`` (``--dry-run``); no device is opened.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "ml" / "sim2f_fov_hw_session.py"


def _load_session_module():
    """Import the script by path (``scripts/`` is not an installable package)."""
    spec = importlib.util.spec_from_file_location(
        "sim2f_fov_hw_session_under_test", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


session_mod = _load_session_module()


def _plan(session: Path, *, screens: int = 1, repeats: int = 2) -> dict:
    session_mod.main(
        [
            "plan",
            "--out",
            str(session),
            "--screens",
            str(screens),
            "--repeats",
            str(repeats),
            "--targets",
            "ring",
        ]
    )
    return json.loads((session / session_mod.PLAN_NAME).read_text(encoding="utf-8"))


def _capture(session: Path) -> dict:
    session_mod.main(["capture", "--session", str(session), "--dry-run"])
    return json.loads((session / session_mod.MANIFEST_NAME).read_text(encoding="utf-8"))


def _finalize(session: Path) -> dict:
    session_mod.main(["finalize", "--session", str(session)])
    return json.loads((session / session_mod.SESSION_NAME).read_text(encoding="utf-8"))


class TestPlan:
    """The plan is data: same arguments in, same bytes out."""

    def test_two_runs_are_byte_identical(self, tmp_path) -> None:
        first = tmp_path / "first"
        second = tmp_path / "second"
        plan_a = _plan(first)
        plan_b = _plan(second)

        assert (first / session_mod.PLAN_NAME).read_bytes() == (
            second / session_mod.PLAN_NAME
        ).read_bytes()
        assert plan_a["items"] == plan_b["items"]
        for item in plan_a["items"]:
            assert (first / item["phase"]).read_bytes() == (
                second / item["phase"]
            ).read_bytes()

    def test_it_records_the_required_stages(self, tmp_path) -> None:
        plan = _plan(tmp_path / "session", screens=2)

        assert plan["schema"] == session_mod.PLAN_SCHEMA
        assert plan["screens"] == 2
        assert session_mod.CALIB_SCREENS == 24
        assert plan["probe"]["noll"] == list(range(4, 12))
        assert plan["probe"]["amplitude_rad"] == pytest.approx(0.35)

        calib = [item for item in plan["items"] if item["kind"] == "calib_screen"]
        probes = [item for item in plan["items"] if item["kind"] == "probe"]
        shapes = [item for item in plan["items"] if item["kind"] == "shape"]
        assert len(calib) == 2
        assert len(probes) == 16  # +/- for Noll 4..11
        assert {item["sign"] for item in probes} == {"plus", "minus"}
        assert {item["target"] for item in shapes} == {"ring"}
        assert {item["variant"] for item in shapes} == {"nominal"}

        # Every row maps into the conditions ``load_bench_session`` can read.
        assert {item["condition"] for item in plan["items"]} <= {
            "a1",
            "a2",
            "b1",
            "b2",
        }
        assert all(item["phase_sha256"] for item in plan["items"])

    def test_repeats_must_be_two_or_three(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="repeats"):
            session_mod.build_plan(
                out=tmp_path / "bad",
                grid=16,
                window_px=256,
                seed=0,
                screens=1,
                repeats=4,
                targets=("ring",),
                calibration=None,
                fit_iters=1,
            )


class TestCaptureResume:
    """Ctrl-C leaves a session that resumes instead of restarting."""

    def test_it_writes_native_frames_and_a_manifest(self, tmp_path) -> None:
        session = tmp_path / "session"
        plan = _plan(session)
        manifest = _capture(session)

        assert manifest["schema"] == session_mod.MANIFEST_SCHEMA
        assert manifest["native_shape"] == [256, 256]
        assert len(manifest["items"]) == len(plan["items"])
        for item in plan["items"]:
            entry = manifest["items"][item["id"]]
            assert entry["status"] == "done"
            frame = np.load(session / entry["frame"])
            assert frame.dtype == np.float32
            assert frame.shape == (256, 256)
            assert entry["phase_sha256"] == item["phase_sha256"]
            assert entry["kind"] == item["kind"]

    def test_it_skips_valid_and_recaptures_missing_or_corrupt(
        self, tmp_path, capsys
    ) -> None:
        session = tmp_path / "session"
        plan = _plan(session)
        manifest = _capture(session)

        ids = [item["id"] for item in plan["items"]]
        missing = manifest["items"][ids[0]]
        corrupt = manifest["items"][ids[1]]
        stale = manifest["items"][ids[2]]
        untouched = manifest["items"][ids[3]]
        untouched_bytes = (session / untouched["frame"]).read_bytes()

        (session / missing["frame"]).unlink()
        (session / corrupt["frame"]).write_bytes(b"not a npy file")
        np.save(session / stale["frame"], np.zeros((256, 256), dtype=np.float32))

        resumed = _capture(session)
        output = capsys.readouterr().out
        assert "3/" in output  # exactly the three broken items were pending

        assert (session / untouched["frame"]).read_bytes() == untouched_bytes
        for item_id in (ids[0], ids[1], ids[2]):
            entry = resumed["items"][item_id]
            reloaded = np.load(session / entry["frame"])
            assert reloaded.shape == (256, 256)
            assert session_mod._array_sha256(reloaded) == entry["frame_sha256"]

        # A third run has nothing to do.
        _capture(session)
        assert "already valid" in capsys.readouterr().out


class TestFinalize:
    """``finalize`` is the bridge to the repository's standard recording."""

    def test_the_recording_loads_through_the_standard_reader(self, tmp_path) -> None:
        from ml.sim2f.calibrate import load_bench_session

        session = tmp_path / "session"
        plan = _plan(session)
        _capture(session)
        payload = _finalize(session)

        assert payload["schema"] == session_mod.SESSION_SCHEMA
        assert payload["items"] == len(plan["items"])
        assert payload["window_px"] == 256

        recording = load_bench_session(session)
        assert recording.sim_config.ccd_px == 256
        assert len(recording.samples) == len(plan["items"])
        assert all(sample.frame.shape == (256, 256) for sample in recording.samples)

        # Re-running finalize must not append duplicate rows.
        _finalize(session)
        again = load_bench_session(session)
        assert len(again.samples) == len(recording.samples)


class TestAnalyze:
    """``analyze`` reports numbers or says it does not implement them."""

    def test_a_dry_session_produces_finite_metrics(self, tmp_path) -> None:
        session = tmp_path / "session"
        out = tmp_path / "analysis"
        _plan(session)
        _capture(session)
        _finalize(session)

        session_mod.main(["analyze", "--session", str(session), "--out", str(out)])
        summary = json.loads(
            (out / session_mod.SUMMARY_NAME).read_text(encoding="utf-8")
        )

        assert summary["schema"] == session_mod.SUMMARY_SCHEMA
        assert summary["window_px"] == 256
        assert len(summary["items"]) == len(
            json.loads((session / session_mod.PLAN_NAME).read_text(encoding="utf-8"))[
                "items"
            ]
        )
        for row in summary["items"]:
            metrics = {
                key: value for key, value in row.items() if isinstance(value, float)
            }
            assert metrics, row
            assert all(np.isfinite(value) for value in metrics.values()), row
            assert {"l1_calib", "ncc_calib", "gain_calib"} <= set(row)
        flat = next(row for row in summary["items"] if row["kind"] == "flat")
        assert flat["l1_in"] < 1e-3
        assert flat["ncc_in"] > 0.999
        groups = summary["groups"]
        assert set(groups) == {row["group"] for row in summary["items"]}
        for block in groups.values():
            assert block["items"] > 0
            assert set(block["metrics"]) == set(session_mod.METRIC_KEYS)
            for key in ("l1_calib", "ncc_calib", "gain_calib"):
                metric = block["metrics"][key]
                assert set(metric) == {"mean", "n"}
                assert 0 <= metric["n"] <= block["items"]
                assert metric["mean"] is None or np.isfinite(metric["mean"])
        assert "ok" in summary["fit"]
        assert summary["not_implemented"]
        assert all(
            value.startswith("not_implemented")
            for value in summary["not_implemented"].values()
        )

    def test_the_summary_records_the_crop_and_support(self, tmp_path) -> None:
        session = tmp_path / "session"
        out = tmp_path / "analysis"
        _plan(session)
        _capture(session)
        _finalize(session)

        session_mod.main(["analyze", "--session", str(session), "--out", str(out)])
        summary = json.loads(
            (out / session_mod.SUMMARY_NAME).read_text(encoding="utf-8")
        )

        half = summary["native_shape"][0] // 2
        crop_sq = summary["crop_sq_zero_order_xy"]
        assert crop_sq == [
            summary["zero_order_xy"][0] - summary["crop_origin_xy"][0],
            summary["zero_order_xy"][1] - summary["crop_origin_xy"][1],
        ]
        assert summary["zero_order_offset_xy"] == [
            crop_sq[0] - half,
            crop_sq[1] - half,
        ]
        assert all(row["score_mask_count"] > 0 for row in summary["items"])
        assert "NaN" not in json.dumps(summary)


class TestAnalyzeZeroOrder:
    """The measured zero-order is reported, or said to be unmeasurable."""

    def test_the_measured_zero_order_reports_its_offset(self, tmp_path) -> None:
        session = tmp_path / "session"
        out = tmp_path / "analysis"
        _plan(session)
        manifest = _capture(session)
        center = manifest["zero_order_xy"]
        height, width = manifest["native_shape"]
        yy, xx = np.mgrid[0:height, 0:width]
        shifted = np.exp(
            -((xx - center[0]) ** 2 + (yy - (center[1] + 2)) ** 2) / (2.0 * 2.0**2)
        )
        flat = _set_flat_frame(session, shifted)
        _finalize(session)

        session_mod.main(["analyze", "--session", str(session), "--out", str(out)])
        summary = json.loads(
            (out / session_mod.SUMMARY_NAME).read_text(encoding="utf-8")
        )

        assert summary["zero_order_measured_source"] == flat["id"]
        measured = summary["zero_order_measured_xy"]
        assert measured[0] == pytest.approx(center[0], abs=0.1)
        assert measured[1] == pytest.approx(center[1] + 2.0, abs=0.1)
        assert summary["zero_order_residual_px"] == pytest.approx(2.0, abs=0.1)
        assert summary["zero_order_aligned_xy"] == [
            float(value) for value in summary["crop_sq_zero_order_xy"]
        ]
        assert summary["zero_order_align_residual_px"] == pytest.approx(2.0, abs=0.1)
        assert "zero_order_residual_px" not in summary["undefined_metrics"]

    def test_an_empty_flat_frame_leaves_the_measurement_undefined(
        self, tmp_path
    ) -> None:
        session = tmp_path / "session"
        out = tmp_path / "analysis"
        _plan(session)
        manifest = _capture(session)
        _set_flat_frame(
            session, np.zeros(tuple(manifest["native_shape"]), dtype=np.float32)
        )
        _finalize(session)

        session_mod.main(["analyze", "--session", str(session), "--out", str(out)])
        summary = json.loads(
            (out / session_mod.SUMMARY_NAME).read_text(encoding="utf-8")
        )

        assert summary["zero_order_measured_xy"] is None
        assert summary["zero_order_measured_source"] is None
        assert summary["zero_order_residual_px"] is None
        assert summary["zero_order_align_residual_px"] is None
        assert {
            "zero_order_measured_xy",
            "zero_order_measured_source",
            "zero_order_residual_px",
            "zero_order_align_residual_px",
        } <= set(summary["undefined_metrics"])
        assert "NaN" not in json.dumps(summary)


def _set_flat_frame(session: Path, frame: np.ndarray) -> dict:
    """Overwrite the first flat frame and keep its manifest hash consistent."""
    plan = json.loads((session / session_mod.PLAN_NAME).read_text(encoding="utf-8"))
    manifest = json.loads(
        (session / session_mod.MANIFEST_NAME).read_text(encoding="utf-8")
    )
    flat = next(item for item in plan["items"] if item["kind"] == "flat")
    entry = manifest["items"][flat["id"]]
    path = session / entry["frame"]
    np.save(path, np.asarray(frame, dtype=np.float32))
    entry["frame_sha256"] = session_mod._array_sha256(session_mod._load_npy(path))
    (session / session_mod.MANIFEST_NAME).write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return flat


def _spot(size: int, center: tuple[int, int], sigma: float = 2.0) -> np.ndarray:
    """A small Gaussian spot on a square frame."""
    yy, xx = np.mgrid[0:size, 0:size]
    return np.exp(-((xx - center[0]) ** 2 + (yy - center[1]) ** 2) / (2.0 * sigma**2))


def _window(full_px: int, sq_center: tuple[int, int]):
    return session_mod.CameraWindow(
        center=sq_center, size=(full_px, full_px), frame=(full_px, full_px)
    )


class TestSupportAlignment:
    """The prediction is aligned to the recorded zero-order, not the crop centre."""

    def test_a_clipped_crop_reports_its_offset(self) -> None:
        col0, row0, sq_center = session_mod._crop_layout((256, 300), (10, 128), 256)

        assert (col0, row0) == (0, 0)
        assert sq_center == (10, 128)

    def test_a_roll_puts_the_prediction_on_the_recorded_zero_order(self) -> None:
        full_px = 256
        half = full_px // 2
        sq_center = (10, 128)
        predicted = _spot(full_px, (half, half))
        truth = _spot(full_px, sq_center)

        rolled, valid = session_mod._roll_to_zero_order(predicted, sq_center, full_px)
        support = valid & (rolled > 0)
        metrics = session_mod._support_metrics(
            rolled, truth, _window(full_px, sq_center), support
        )

        assert np.unravel_index(int(np.argmax(rolled)), rolled.shape) == (128, 10)
        assert metrics["ncc_all"] > 0.999
        assert metrics["l1_all"] < 1e-3

    def test_wrapped_pixels_never_reach_the_scores(self) -> None:
        full_px = 256
        half = full_px // 2
        sq_center = (10, 128)
        # dx = -118, so input column 62 wraps to output column 200, which is
        # inside the frame but explicitly outside the valid support.
        predicted = _spot(full_px, (half, half))
        truth = _spot(full_px, sq_center)
        window = _window(full_px, sq_center)

        clean_rolled, clean_valid = session_mod._roll_to_zero_order(
            predicted, sq_center, full_px
        )
        clean = session_mod._support_metrics(
            clean_rolled, truth, window, clean_valid & (clean_rolled > 0)
        )

        garbage = predicted.copy()
        garbage[half, 62] = 1e6
        rolled, valid = session_mod._roll_to_zero_order(garbage, sq_center, full_px)
        dirty = session_mod._support_metrics(
            rolled, truth, window, valid & (rolled > 0)
        )

        assert not valid[:, 200].any()
        assert dirty["ncc_all"] == pytest.approx(clean["ncc_all"])
        assert dirty["l1_all"] == pytest.approx(clean["l1_all"])

    def test_pixels_the_prediction_never_supports_are_ignored(self) -> None:
        full_px = 256
        half = full_px // 2
        sq_center = (10, 128)
        predicted = _spot(full_px, (half, half))
        rolled, valid = session_mod._roll_to_zero_order(predicted, sq_center, full_px)
        support = valid & (rolled > 0)
        window = _window(full_px, sq_center)

        truth = _spot(full_px, sq_center)
        clean = session_mod._support_metrics(rolled, truth, window, support)

        noisy_truth = truth.copy()
        assert rolled[10, 100] == 0
        assert not support[10, 100]
        noisy_truth[10, 100] = 5.0
        noisy = session_mod._support_metrics(rolled, noisy_truth, window, support)

        assert noisy["ncc_all"] == pytest.approx(clean["ncc_all"])
        assert noisy["l1_all"] == pytest.approx(clean["l1_all"])

    def test_identical_centred_frames_score_perfectly(self) -> None:
        full_px = 256
        centre = (full_px // 2, full_px // 2)
        frame = _spot(full_px, centre)
        _, valid = session_mod._roll_to_zero_order(frame, centre, full_px)
        support = valid & (frame > 0)

        metrics = session_mod._support_metrics(
            frame, frame, _window(full_px, centre), support
        )

        assert metrics["ncc_all"] == pytest.approx(1.0)
        assert metrics["l1_all"] == pytest.approx(0.0, abs=1e-12)

    def test_the_window_follows_the_recorded_zero_order(self) -> None:
        full_px = 256
        sq_center = (100, 128)
        window = session_mod.CameraWindow(
            center=sq_center, size=(64, 64), frame=(full_px, full_px)
        )

        assert window.center == sq_center
        assert window.offset((full_px, full_px)) == (68, 96)

    def test_empty_support_is_json_safe(self) -> None:
        diagnostics = session_mod._support_diagnostics(np.zeros((4, 4), dtype=bool))

        assert diagnostics["score_mask_count"] == 0
        assert diagnostics["valid_support_fraction"] == 0.0
        assert diagnostics["valid_support_bbox"] is None
        json.dumps(diagnostics)


class TestGeometricSupport:
    """Support is the model's physical window, not the lit prediction pixels."""

    @staticmethod
    def _coarse_cfg():
        # 22 um pixels put the ~1.1 mm model window at ~54 px, so a 256 px
        # frame contains it and the boundary is testable.
        return replace(session_mod.Sim2fConfig.calibrated(), ccd_pixel_um=22.0)

    def test_half_width_is_the_far_field_window_in_camera_pixels(self) -> None:
        cfg = session_mod.Sim2fConfig.calibrated()
        expected = (
            cfg.fft_length * cfg.far_field_pitch_m / (2.0 * cfg.ccd_pixel_um * 1e-6)
        )

        assert session_mod._geometric_half_width_px(cfg) == pytest.approx(expected)

    def test_the_mask_is_centred_on_the_zero_order(self) -> None:
        cfg = self._coarse_cfg()
        full_px = 256
        centre = (100, 130)
        mask = session_mod._geometric_support(cfg, centre, full_px)
        half = session_mod._geometric_half_width_px(cfg)
        yy, xx = np.mgrid[0:full_px, 0:full_px]
        expected = (np.abs(xx - centre[0]) <= half) & (np.abs(yy - centre[1]) <= half)

        np.testing.assert_array_equal(mask, expected)
        assert mask[centre[1], centre[0]]
        assert not mask[0, 0]

    def test_positive_pixels_outside_the_model_window_do_not_move_scores(
        self,
    ) -> None:
        cfg = self._coarse_cfg()
        full_px = 256
        centre = (128, 128)
        support = session_mod._geometric_support(cfg, centre, full_px)
        truth = _spot(full_px, centre)
        clean = _spot(full_px, centre)
        bright = clean.copy()
        bright[:8, :8] = 1e6
        assert not support[:8, :8].any()
        window = _window(full_px, centre)

        a = session_mod._support_metrics(clean, truth, window, support)
        b = session_mod._support_metrics(bright, truth, window, support)
        for key in (
            "l1_all",
            "ncc_all",
            "l1_calib",
            "ncc_calib",
            "gain_calib",
            "centroid_err_px",
            "d86_err_px",
        ):
            assert b[key] == pytest.approx(a[key], rel=1e-12, abs=1e-12)

    def test_energy_fractions_use_global_unit_sum_bases(self) -> None:
        full_px = 64
        support = np.zeros((full_px, full_px), dtype=bool)
        support[16:48, 16:48] = True
        truth = np.zeros((full_px, full_px))
        truth[24:40, 24:40] = 1.0
        pred = truth.copy()
        pred[0:4, 0:4] = 1.0

        metrics = session_mod._support_metrics(
            pred, truth, _window(full_px, (32, 32)), support
        )

        assert metrics["truth_energy_in_support"] == pytest.approx(1.0)
        assert metrics["energy_total_ratio"] == pytest.approx(256.0 / (256.0 + 16.0))


class TestCalibrationMetrics:
    """Calibration metrics are the same numbers ``calibrate._scores`` reports."""

    def test_prepare_frame_removes_the_pedestal_before_fitting_gain(self) -> None:
        full_px = 64
        truth_spot = _spot(full_px, (32, 32), sigma=2.5)
        truth = 5.0 + truth_spot
        pred = 5.0 + 2.0 * truth_spot
        support = np.ones((full_px, full_px), dtype=bool)

        metrics = session_mod._support_metrics(
            pred, truth, _window(full_px, (32, 32)), support
        )

        assert metrics["gain_calib"] == pytest.approx(0.5)
        assert metrics["l1_calib"] == pytest.approx(0.0, abs=1e-4)
        assert metrics["ncc_calib"] == pytest.approx(1.0)


class _ForwardRecorder:
    def __init__(self) -> None:
        self.kwargs = None

    def forward(self, phase, **kwargs):
        self.kwargs = kwargs
        return np.zeros((1, 1, 4, 4))


def test_analyze_predicts_without_detector_photometry() -> None:
    recorder = _ForwardRecorder()

    session_mod._predict_full_frame(recorder, np.zeros((8, 8)), None)

    assert recorder.kwargs["detector"] is False
    assert recorder.kwargs["aberration"] is None


class TestCalibrate:
    """``calibrate`` exposes both post-affine stages and reports both."""

    @staticmethod
    def _recording(tmp_path: Path) -> Path:
        session = tmp_path / "session"
        _plan(session)
        _capture(session)
        _finalize(session)
        return session

    def test_both_post_affine_flags_default_off(self) -> None:
        args = session_mod._build_parser().parse_args(
            ["calibrate", "--session", "s", "--out", "o"]
        )

        assert args.fit_measurement is False
        assert args.fit_zero_order is False

    def test_the_summary_carries_the_transfer_and_the_zero_order(
        self, tmp_path
    ) -> None:
        session = self._recording(tmp_path)
        out = tmp_path / "calibration"

        session_mod.main(
            [
                "calibrate",
                "--session",
                str(session),
                "--out",
                str(out),
                "--iters",
                "2",
                "--fit-measurement",
                "--fit-zero-order",
            ]
        )
        summary = json.loads(
            (out / "calibrate_summary.json").read_text(encoding="utf-8")
        )

        assert summary["schema"] == "sim2f_fov_hw_session/calibrate/v1"
        assert "psf_sigma_px" in summary["measurement"]
        assert summary["zero_order"].keys() == {"eta"}
        assert 0.0 <= summary["zero_order"]["eta"] <= 0.5
        assert summary["score"].keys() == summary["score_start"].keys()

    def test_with_both_flags_off_the_summary_is_unchanged(self, tmp_path) -> None:
        session = self._recording(tmp_path)
        out = tmp_path / "calibration"

        session_mod.main(
            ["calibrate", "--session", str(session), "--out", str(out), "--iters", "2"]
        )
        summary = json.loads(
            (out / "calibrate_summary.json").read_text(encoding="utf-8")
        )

        assert summary["measurement"]["psf_sigma_px"] == 0.0
        assert summary["zero_order"] == {}
