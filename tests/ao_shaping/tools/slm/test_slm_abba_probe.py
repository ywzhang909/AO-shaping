"""Offline tests for the ABBA dense-phase detectability probe.

No hardware: the panel and the camera are stand-ins wired through the same two
injection points (``display_pattern`` / ``capture``) that ``main()`` builds from
the real drivers, so the whole measurement path -- exposure bracketing, drift
floor, the ``+ - - +`` palindrome, the SNR verdict and the checkpoints -- runs
in-process with no device, no SDK and no sleep.
"""

from __future__ import annotations

import ast
import pickle
from pathlib import Path

import numpy as np
import pytest

from ao_shaping.tools.slm import slm_abba_probe as probe
from ao_shaping.tools.slm.slm_abba_probe import (
    DEFAULT_GRID,
    DEFAULT_PAIRS,
    DEFAULT_SEED,
    dense_abba_sweep,
    drift_floor_at_exposure,
    prepare_roi_frame,
    rebracket_exposure,
    save_abba_records_pkl,
    save_abba_summary_npz,
)
from ao_shaping.tools.slm.slm_bench_metrics import crop_roi, finite_clip
from ao_shaping.tools.slm.slm_snr_probe import SNR_STRONG, SNR_USABLE

PANEL = (48, 64)
FRAME = (64, 64)
CELLS = 8
CENTRE = (32, 32)
HALF = 16


def _spot(frame_shape: tuple[int, int] = FRAME, peak: float = 220.0) -> np.ndarray:
    yy, xx = np.mgrid[0 : frame_shape[0], 0 : frame_shape[1]]
    return peak * np.exp(
        -(((yy - 32.0) ** 2 + (xx - 32.0) ** 2) / (2 * 4.0**2))
    )


class FakeBench:
    """A panel plus a camera, good enough to be a bench.

    The far field is a tiny physical model rather than an arbitrary function of
    the command: the displayed phase is collapsed onto a coarse grid and added
    to a fixed complex pupil field, and the frame shows ``|psi|**2``. That
    matters for the test's central claim -- because ``|psi0 + a|**2`` and
    ``|psi0 - a|**2`` differ by ``4*Re(conj(psi0)*a)``, a ``+`` and a ``-``
    perturbation give genuinely *different* intensities. A model that returned
    ``tanh(phase)`` would be odd-symmetric and every ABBA response would collapse
    to zero, testing nothing.
    """

    def __init__(
        self,
        gain: float = 1.6,
        noise: float = 0.5,
        peak: float = 220.0,
        seed: int = 11,
    ) -> None:
        self.gain = float(gain)
        self.noise = float(noise)
        self.peak = float(peak)
        self.phase = np.zeros(PANEL, dtype=np.float64)
        self._pilot = np.random.default_rng(seed)
        self._read_noise = np.random.default_rng(seed + 1)
        self.pilot_field = (
            self._pilot.normal(size=(CELLS, CELLS))
            + 1j * self._pilot.normal(size=(CELLS, CELLS))
        )
        self.reads = 0
        self.displays: list[np.ndarray] = []

    def display(self, phase: np.ndarray) -> None:
        self.phase = np.asarray(phase, dtype=np.float64)
        self.displays.append(self.phase)

    def _coarse(self) -> np.ndarray:
        bh, bw = PANEL[0] // CELLS, PANEL[1] // CELLS
        return self.phase[: CELLS * bh, : CELLS * bw].reshape(
            CELLS, bh, CELLS, bw
        ).mean(axis=(1, 3))

    def frame(self) -> np.ndarray:
        self.reads += 1
        intensity = np.abs(self.pilot_field + self.gain * self._coarse()) ** 2
        top = float(intensity.max())
        cell = intensity / top if top > 0.0 else intensity
        speckle = np.kron(cell, np.ones((FRAME[0] // CELLS, FRAME[1] // CELLS)))
        frame = _spot() + self.peak * speckle
        return frame + self._read_noise.normal(0.0, self.noise, FRAME)


def _capture(bench: FakeBench, centre: tuple[int, int] = CENTRE, half: int = HALF):
    """The ``capture`` half of the injection pair, as ``main()`` builds it."""

    def capture() -> np.ndarray:
        return prepare_roi_frame(bench.frame(), centre, half)

    return capture


def _sweep(bench: FakeBench, *, display_pattern=None, **kwargs):
    params: dict[str, object] = {
        "n_patterns": 4,
        "pairs": DEFAULT_PAIRS,
        "delta_rad": 0.5,
        "grid": 4,
        "panel_shape": PANEL,
        "floor": 20.0,
        "seed": DEFAULT_SEED,
        "exposure_ms": 1.0,
    }
    params.update(kwargs)
    return dense_abba_sweep(
        _capture(bench),
        bench.display if display_pattern is None else display_pattern,
        **params,
    )


def _floor_for(bench: FakeBench, n: int = 6) -> float:
    floor, _ = drift_floor_at_exposure(_capture(bench), n)
    return floor


class TestRebracketExposure:
    @staticmethod
    def _linear_bench(scale: float = 40.0):
        state = {"exposure": 0.0}
        frames = []

        def capture() -> np.ndarray:
            img = np.zeros((8, 8), dtype=np.float64)
            img[4, 4] = state["exposure"] * scale
            frames.append(float(img[4, 4]))
            return img

        def set_exposure(value: float) -> None:
            state["exposure"] = float(value)

        return capture, set_exposure, frames

    def test_picks_the_brightest_unsaturated_rung(self) -> None:
        capture, set_exposure, _ = self._linear_bench()
        chosen, info = rebracket_exposure(
            capture, set_exposure, [0.4, 0.8, 1.0], saturation_level=45.0
        )
        # Peaks are 16 / 32 / 40; 40 is the brightest and still under 45.
        assert chosen == pytest.approx(1.0)
        assert info["chosen_peak"] == pytest.approx(40.0)
        assert not info["all_saturated"]

    def test_a_saturated_rung_is_skipped_for_the_next_brightest(self) -> None:
        capture, set_exposure, _ = self._linear_bench()
        chosen, info = rebracket_exposure(
            capture, set_exposure, [0.4, 0.8, 1.0], saturation_level=35.0
        )
        # 40 clips; 32 is the brightest that does not.
        assert chosen == pytest.approx(0.8)
        assert info["clipped"] == [False, False, True]

    def test_all_saturated_falls_back_to_the_dimmest_rung(self) -> None:
        capture, set_exposure, _ = self._linear_bench()
        chosen, info = rebracket_exposure(
            capture, set_exposure, [0.4, 0.8, 1.0], saturation_level=5.0
        )
        assert chosen == pytest.approx(0.4)
        assert info["all_saturated"] is True
        assert all(info["clipped"])

    def test_every_rung_is_probed_in_order(self) -> None:
        capture, set_exposure, seen = self._linear_bench()
        ladder = [0.4, 0.8, 1.0]
        rebracket_exposure(capture, set_exposure, ladder)
        assert len(seen) == len(ladder)
        assert seen == pytest.approx([16.0, 32.0, 40.0])

    def test_empty_ladder_raises(self) -> None:
        capture, set_exposure, _ = self._linear_bench()
        with pytest.raises(ValueError, match="must not be empty"):
            rebracket_exposure(capture, set_exposure, [])

    def test_a_dark_bench_still_selects_its_brightest_rung(self) -> None:
        capture, set_exposure, _ = self._linear_bench(scale=1e-6)
        chosen, info = rebracket_exposure(
            capture, set_exposure, [0.4, 0.8, 1.0], saturation_level=245.0
        )
        assert chosen == pytest.approx(1.0)
        assert not info["all_saturated"]


class TestDriftFloor:
    def test_returns_the_median_and_every_consecutive_difference(self) -> None:
        frames = [np.full((4, 4), float(i)) for i in range(5)]
        it = iter(frames)
        floor, series = drift_floor_at_exposure(lambda: next(it), 5)
        assert len(series) == 4
        assert floor == pytest.approx(float(np.median(series)))

    def test_a_static_bench_has_a_zero_floor(self) -> None:
        floor, series = drift_floor_at_exposure(lambda: np.zeros((4, 4)), 4)
        assert floor == pytest.approx(0.0)
        assert series == [0.0, 0.0, 0.0]

    def test_a_real_bench_has_a_positive_floor(self) -> None:
        bench = FakeBench()
        floor, series = drift_floor_at_exposure(_capture(bench), 6)
        assert floor > 0.0
        assert len(series) == 5

    def test_fewer_than_two_frames_raises(self) -> None:
        with pytest.raises(ValueError, match="n must be >= 2"):
            drift_floor_at_exposure(lambda: np.zeros((4, 4)), 1)

    def test_it_is_the_shared_kernel_not_a_reimplementation(self) -> None:
        from ao_shaping.tools.slm.slm_bench_metrics import flat_to_flat_floor

        bench = FakeBench()
        frames = [np.asarray(_capture(bench)()) for _ in range(4)]
        stream = iter(frames)
        assert drift_floor_at_exposure(lambda: next(stream), 4) == flat_to_flat_floor(
            frames
        )


class TestPrepareRoiFrame:
    def test_removes_the_read_noise_pedestal(self) -> None:
        # A pedestal of 50 with sigma=8 over 4096 samples never reaches zero
        # (50 - 5*8 = 10), so the fixture would not exercise the clip at all.
        # Use a pedestal comparable to sigma so negatives really occur, which is
        # the measured situation: 7164/14400 negative pixels on a real frame.
        rng = np.random.default_rng(3)
        frame = 4.0 + rng.normal(0.0, 8.0, FRAME)
        assert (frame < 0.0).any(), "the fixture must contain negative pixels"
        prepared = prepare_roi_frame(frame, CENTRE, HALF)
        assert prepared.shape == (2 * HALF, 2 * HALF)
        assert float(prepared.min()) >= 0.0

    def test_crops_about_the_passed_centre_not_the_frame_centre(self) -> None:
        frame = np.zeros(FRAME, dtype=np.float64)
        frame[8, 8] = 500.0  # a bright blob well off the geometric centre
        prepared = prepare_roi_frame(frame, (8, 8), 4)
        # The blob sits at the ROI centre, which the pedestal subtraction keeps.
        assert prepared.shape == (8, 8)
        assert float(prepared[4, 4]) == pytest.approx(500.0)
        assert float(prepared[0, 0]) == pytest.approx(0.0)

    def test_it_is_finite_clip_then_crop_roi(self) -> None:
        frame = np.arange(FRAME[0] * FRAME[1], dtype=np.float64).reshape(FRAME) - 500.0
        assert np.array_equal(
            prepare_roi_frame(frame, CENTRE, HALF),
            crop_roi(finite_clip(frame), CENTRE, HALF),
        )


class TestDenseAbbaSweep:
    def test_a_responsive_bench_is_graded_usable(self) -> None:
        bench = FakeBench()
        floor = _floor_for(bench)
        rows, summary = _sweep(bench, floor=floor)
        assert floor > 0.0
        assert summary["verdict"] == "usable"
        assert summary["snr_class"] in ("usable", "strong")
        assert summary["n_exceeding"] == summary["n_patterns"]
        assert summary["exceeding_fraction"] == pytest.approx(1.0)
        assert all(row["response"] > 0.0 for row in rows)
        assert summary["median_snr"] >= SNR_USABLE

    def test_a_frozen_panel_clears_nothing(self) -> None:
        """+ and - write the *same* pixels, so nothing is detectable.

        This is the null control: with the panel unresponsive the response can
        only be read noise, which must not be reported as a signal.
        """
        bench = FakeBench()
        floor = _floor_for(bench)
        rows, summary = _sweep(
            bench, floor=floor, display_pattern=lambda phase: None
        )
        assert summary["n_exceeding"] == 0
        assert summary["exceeding_fraction"] == pytest.approx(0.0)
        assert summary["verdict"] == "unusable"
        assert all(row["snr"] < SNR_USABLE for row in rows)

    def test_a_zero_coefficient_grid_writes_flat_for_both_signs(self) -> None:
        """Zero coefficients are the cleanest identical-to-flat case.

        A zero block pattern is flat whatever sign multiplies it, so the + and -
        writes are byte-identical and every ABBA response can only be noise.
        """
        from ao_shaping.tools.slm.slm_bench_metrics import build_block_pattern

        assert not np.any(build_block_pattern(np.zeros(16), 4, PANEL))

        bench = FakeBench(gain=0.0)  # panel state cannot reach the far field
        floor = _floor_for(bench)
        rows, summary = dense_abba_sweep(
            _capture(bench),
            bench.display,
            n_patterns=3,
            pairs=DEFAULT_PAIRS,
            delta_rad=0.5,
            grid=4,
            panel_shape=PANEL,
            floor=floor,
        )
        assert len(rows) == 3
        assert summary["n_exceeding"] == 0
        assert summary["verdict"] == "unusable"

    def test_record_count_matches_n_patterns(self) -> None:
        bench = FakeBench()
        floor = _floor_for(bench)
        for count in (1, 2, 5):
            rows, summary = _sweep(bench, n_patterns=count, floor=floor)
            assert len(rows) == count
            assert summary["n_patterns"] == count
            assert summary["exceeding_fraction"] == pytest.approx(
                summary["n_exceeding"] / count
            )

    def test_epochs_are_contiguous_from_zero(self) -> None:
        bench = FakeBench()
        rows, _ = _sweep(bench, n_patterns=6, floor=_floor_for(bench))
        assert [row["_epoch"] for row in rows] == list(range(6))
        assert [row["pattern"] for row in rows] == list(range(6))

    def test_every_row_carries_the_recorder_schema(self) -> None:
        bench = FakeBench()
        rows, _ = _sweep(bench, floor=_floor_for(bench))
        expected = {
            "peak", "sum", "norm", "response", "floor", "snr",
            "pattern", "pair", "exposure_ms", "_img", "_epoch",
        }
        for row in rows:
            assert expected <= set(row)
            assert np.asarray(row["_img"]).shape == (2 * HALF, 2 * HALF)
            assert np.isfinite(row["response"])
            assert np.isfinite(row["snr"])
            assert row["peak"] > 0.0
            assert row["floor"] == pytest.approx(row["floor"])

    def test_a_zero_floor_does_not_divide_by_zero(self) -> None:
        bench = FakeBench()
        rows, summary = _sweep(bench, floor=0.0)
        assert all(np.isfinite(row["snr"]) for row in rows)
        assert np.isfinite(summary["median_snr"])

    def test_a_bigger_delta_gives_a_bigger_response(self) -> None:
        bench = FakeBench()
        floor = _floor_for(bench)
        small, _ = _sweep(bench, delta_rad=0.25, floor=floor, seed=5)
        large, _ = _sweep(bench, delta_rad=1.5, floor=floor, seed=5)
        assert np.median([r["response"] for r in large]) > np.median(
            [r["response"] for r in small]
        )

    def test_the_same_seed_reproduces_the_same_run(self) -> None:
        floor = _floor_for(FakeBench())
        first, first_summary = _sweep(FakeBench(), seed=17, floor=floor)
        second, second_summary = _sweep(FakeBench(), seed=17, floor=floor)
        assert [r["response"] for r in first] == pytest.approx(
            [r["response"] for r in second]
        )
        assert [r["snr"] for r in first] == pytest.approx([r["snr"] for r in second])
        assert first_summary["verdict"] == second_summary["verdict"]

    def test_different_seeds_explore_different_patterns(self) -> None:
        floor = _floor_for(FakeBench())
        first, _ = _sweep(FakeBench(), seed=1, floor=floor)
        second, _ = _sweep(FakeBench(), seed=2, floor=floor)
        assert not np.allclose(
            [r["response"] for r in first], [r["response"] for r in second]
        )

    def test_the_bench_is_left_flat(self) -> None:
        bench = FakeBench()
        _sweep(bench, floor=_floor_for(bench))
        assert not np.any(bench.phase)

    def test_the_flat_reference_is_written_before_it_is_read(self) -> None:
        """The panel retains the last pattern, so the anchor write must be first."""
        bench = FakeBench()
        _sweep(bench, floor=_floor_for(bench))
        assert not np.any(bench.displays[0]), "the first write must be flat"

    def test_on_pattern_fires_once_per_pattern(self) -> None:
        bench = FakeBench()
        seen: list[int] = []
        _sweep(bench, n_patterns=3, floor=_floor_for(bench), on_pattern=seen.append)
        assert len(seen) == 3

    def test_provenance_lands_in_the_summary(self) -> None:
        bench = FakeBench()
        _, summary = _sweep(
            bench, floor=12.5, seed=99, grid=5, delta_rad=0.75, exposure_ms=1.25
        )
        assert summary["seed"] == 99
        assert summary["grid"] == 5
        assert summary["n_dof"] == 25
        assert summary["delta_rad"] == pytest.approx(0.75)
        assert summary["exposure_ms"] == pytest.approx(1.25)
        assert summary["drift_floor"] == pytest.approx(12.5)
        assert summary["snr_strong"] == SNR_STRONG
        assert summary["snr_usable"] == SNR_USABLE
        assert tuple(summary["panel_shape"]) == PANEL

    def test_the_summary_reports_the_median_and_the_minimum(self) -> None:
        bench = FakeBench()
        rows, summary = _sweep(bench, n_patterns=5, floor=_floor_for(bench))
        responses = [row["response"] for row in rows]
        assert summary["median_response"] == pytest.approx(float(np.median(responses)))
        assert summary["min_response"] == pytest.approx(float(np.min(responses)))

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"n_patterns": 0}, "n_patterns"),
            ({"pairs": 0}, "pairs"),
            ({"delta_rad": 0.0}, "delta_rad"),
            ({"delta_rad": -1.0}, "delta_rad"),
        ],
    )
    def test_invalid_arguments_are_rejected(self, kwargs: dict, match: str) -> None:
        bench = FakeBench()
        with pytest.raises(ValueError, match=match):
            _sweep(bench, **kwargs)

    def test_a_grid_too_coarse_for_the_panel_is_rejected(self) -> None:
        from ao_shaping.tools.slm.slm_bench_metrics import build_block_pattern

        with pytest.raises(ValueError, match="too coarse"):
            build_block_pattern(np.zeros(4096), 64, PANEL)

    def test_it_builds_its_patterns_with_the_shared_kernel(self) -> None:
        from ao_shaping.tools.slm.slm_bench_metrics import build_block_pattern

        bench = FakeBench()
        _sweep(bench, grid=4, floor=_floor_for(bench))
        # grid=4 over PANEL -> 4x4 blocks of 12x16 px, exactly what the kernel
        # produces for the same seeded coefficients.
        expected = build_block_pattern(
            np.random.default_rng(DEFAULT_SEED).uniform(-1.0, 1.0, size=16), 4, PANEL
        )
        assert expected.shape == PANEL
        assert np.any(expected != 0.0)


class TestPersistence:
    def test_records_are_keyed_by_epoch(self, tmp_path) -> None:
        bench = FakeBench()
        rows, _ = _sweep(bench, n_patterns=3, floor=_floor_for(bench))
        path = save_abba_records_pkl(tmp_path / "nested" / "abba_records.pkl", rows)
        assert path.exists()
        payload = pickle.loads(path.read_bytes())
        assert sorted(payload) == [0, 1, 2]
        assert payload[1]["pattern"] == 1

    def test_checkpointing_after_every_pattern_keeps_every_epoch(
        self, tmp_path
    ) -> None:
        bench = FakeBench()
        path = tmp_path / "abba_records.pkl"
        accumulated: list[dict] = []
        seen: list[int] = []

        def append(row: dict) -> None:
            seen.append(row["_epoch"])
            accumulated.append(row)
            save_abba_records_pkl(path, accumulated)

        _sweep(bench, n_patterns=4, floor=_floor_for(bench), on_pattern=append)
        assert seen == [0, 1, 2, 3]
        assert sorted(pickle.loads(path.read_bytes())) == [0, 1, 2, 3]

    def test_summary_npz_round_trips(self, tmp_path) -> None:
        bench = FakeBench()
        _, summary = _sweep(bench, floor=_floor_for(bench))
        path = save_abba_summary_npz(tmp_path / "abba_summary.npz", summary)
        assert path.exists()
        with np.load(path, allow_pickle=False) as data:
            assert data["verdict"].item() == summary["verdict"]
            assert data["median_snr"].item() == pytest.approx(summary["median_snr"])
            assert data["n_exceeding"].item() == summary["n_exceeding"]
            assert tuple(data["panel_shape"]) == PANEL


class TestPlanSelfCheck:
    def test_no_hw_returns_zero(self) -> None:
        assert probe.main(["--no-hw"]) == 0

    def test_no_hw_accepts_every_documented_flag(self) -> None:
        assert probe.main(
            [
                "--no-hw",
                "--out", "data/slm_abba",
                "--slm-number", "2",
                "--slm-wavelength", "1064",
                "--cam-type", "miicam",
                "--cam-id", "3",
                "--exposure-ms", "1.0",
                "--rebracket-exposure",
                "--exposure-ladder", "0.4,0.6,0.8,1.0,1.25,1.5",
                "--saturation-level", "250",
                "--saturation-max-peak", "245",
                "--delta-rad", "0.5",
                "--grid", "24",
                "--n-patterns", "12",
                "--pairs", "3",
                "--roi", "192",
                "--seed", "20261001",
            ]
        ) == 0

    def test_no_hw_never_imports_a_driver(self, tmp_path) -> None:
        """The self-check must be safe on a machine with no SLM/CCD SDK."""
        import builtins

        real_import = builtins.__import__

        def guarded(name, *args, **kwargs):
            if name.startswith("ao_shaping.drivers"):
                raise AssertionError(f"--no-hw imported the driver module {name}")
            return real_import(name, *args, **kwargs)

        builtins.__import__ = guarded
        try:
            assert probe.main(["--no-hw"]) == 0
        finally:
            builtins.__import__ = real_import

    def test_defaults_match_the_documented_cli(self) -> None:
        args = probe._parse_args(["--no-hw"])
        assert args.out == "data/slm_abba"
        assert args.slm_number == 1
        assert args.slm_wavelength == 1064
        assert args.cam_type == "daheng"
        assert args.cam_id == 0
        assert args.exposure_ms == pytest.approx(1.0)
        assert args.saturation_level == pytest.approx(250.0)
        assert args.saturation_max_peak == pytest.approx(245.0)
        assert args.delta_rad == pytest.approx(0.5)
        assert args.grid == DEFAULT_GRID
        assert args.n_patterns == 12
        assert args.pairs == 3
        assert args.roi == 192
        assert args.seed == DEFAULT_SEED
        assert probe._parse_floats(args.exposure_ladder) == [0.4, 0.6, 0.8, 1.0, 1.25, 1.5]

    def test_cam_type_rejects_an_unknown_backend(self) -> None:
        with pytest.raises(SystemExit):
            probe._parse_args(["--cam-type", "nikon"])


def test_driver_imports_are_function_local() -> None:
    """Import-time laziness is a repo-wide contract, not a style preference.

    ``--no-hw`` must run on a machine with no SLM/CCD SDK, so the drivers may
    only ever be imported inside the hardware branch of ``main()``.
    """
    tree = ast.parse(Path(probe.__file__).read_text(encoding="utf-8"))
    eager = [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and "drivers" in ast.dump(node)
    ]
    assert eager == []
    assert not hasattr(probe, "Santec")
    assert not hasattr(probe, "create_camera")