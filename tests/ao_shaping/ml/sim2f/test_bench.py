"""Tests for :mod:`ml.sim2f.bench` -- the panel mapping, the quadruple, the gates.

No hardware: the SLM/CCD paths are exercised through fake devices (the pattern of
``tests/ao_shaping/tools/slm/test_slm_phase_response_logic.py``: a ``FakeSLM`` that
records writes, a ``FakeCamera`` that serves frames), and the twin path through
:class:`ml.sim2f.bench.SimInstrument`.

The important test is :class:`TestPublishedReproduction`: it re-measures the
published ``runs/sim2f_fair/loop/a_no_noise`` ``flat`` / ``net_open`` rows through
this module on ``--samples loop``.  If that does not reproduce, the measurement
chain differs from ``loop.run_loop`` and the hardware session would be measuring
something the report never described -- so the failure is the finding, and the
tolerance must not be widened to make it go away.
"""

from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch lives in the optional ml group")

from ml.sim2f.bench import (
    INSUFFICIENT_POWER,
    NET_DOES_NOT_RECOVER,
    NET_RECOVERS,
    PANEL_SHAPE,
    PUPIL_RADIUS_PX,
    BenchmarkError,
    HwInstrument,
    SimInstrument,
    build_panel_phase,
    default_exposure_ladder,
    gate0,
    measure_sample,
    panel_corner,
    pupil_center,
    recorded_grayscale,
    run_session,
    select_indices,
    verdict,
)
from ml.sim2f.config import OpticsErrors, Sim2fConfig, SynthConfig, TeacherConfig
from ml.sim2f.optics import Sim2f

_GRID = 64
_CCD = 64


def _ideal_bench() -> Sim2fConfig:
    """A small, ideal bench: no errors, no detector, fast FFT."""
    return replace(
        Sim2fConfig(),
        grid=_GRID,
        ccd_px=_CCD,
        far_field_padding=2,
        errors=OpticsErrors(),
        exposure_ms=0.0,
    )


def _blob(ccd_px: int = _CCD, *, row: int = 20, col: int = 30, sigma: float = 4.0) -> np.ndarray:
    """A smooth bright blob on a dark frame, as ``(1, 1, ccd, ccd)`` torch."""
    axis = np.arange(ccd_px, dtype=np.float64)
    rr, cc = np.meshgrid(axis, axis, indexing="ij")
    image = np.exp(-(((rr - row) ** 2) + ((cc - col) ** 2)) / (2.0 * sigma**2))
    return torch.from_numpy(image.astype(np.float32))[None, None]


def _scrambled(ccd_px: int = _CCD) -> np.ndarray:
    """A frame that is uncorrelated with :func:`_blob` (a low-ncc 'worse' net)."""
    axis = np.arange(ccd_px, dtype=np.float64)
    rr, cc = np.meshgrid(axis, axis, indexing="ij")
    image = 1.0 + 0.5 * np.sin(rr / 2.0) * np.cos(cc / 3.0)
    return torch.from_numpy(image.astype(np.float32))[None, None]


class FakeInstrument:
    """Serves a flat frame for a zero command and a shaped frame otherwise.

    ``flat_alt`` is served on the *second* flat read instead of ``flat``, which
    is how a noise floor is injected into a digital measurement: the two ``A``
    readings are the same condition, so any difference between them is the
    sample's floor and nothing else.
    """

    def __init__(
        self,
        flat: "torch.Tensor",
        net: "torch.Tensor",
        *,
        flat_alt: "torch.Tensor | None" = None,
        twin: "torch.Tensor | None" = None,
    ) -> None:
        self.flat = flat
        self.flat_alt = flat if flat_alt is None else flat_alt
        self.net = net
        self.twin_frame = net if twin is None else twin
        self._flats = 0
        self.closed = False

    def capture(self, phase_rad) -> np.ndarray:
        array = np.asarray(phase_rad)
        if not np.any(array):
            self._flats += 1
            frame = self.flat if self._flats % 2 else self.flat_alt
            return frame[0, 0].numpy().astype(np.float32)
        return self.net[0, 0].numpy().astype(np.float32)

    def twin(self, phase_rad) -> np.ndarray:
        return self.twin_frame[0, 0].numpy().astype(np.float32)

    def close(self) -> None:
        self.closed = True

    def describe(self) -> dict:
        return {"backend": "fake"}


class TestBuildPanelPhase:
    """The panel mapping of README §1.1 -- the one mistake that ruins a session."""

    def test_places_a_centred_square_and_nothing_else(self) -> None:
        dof = np.linspace(-1.0, 1.0, 128)[None, :] * np.ones((128, 1))
        panel = build_panel_phase(dof)

        assert panel.shape == (1200, 1920)
        assert np.issubdtype(panel.dtype, np.floating)
        # pupil_side(450) = 900 -> rows 150..1049, cols 510..1409 on a 1200x1920
        # panel, centred at (600, 960).
        block = panel[150:1050, 510:1410]
        assert np.any(block != 0.0)
        outside = panel.copy()
        outside[150:1050, 510:1410] = 0.0
        assert np.count_nonzero(outside) == 0
        assert PUPIL_RADIUS_PX * 2 == 900

    def test_inside_is_a_bilinear_upsample_to_the_pupil_square(self) -> None:
        from ao_shaping.utils.image.resample import upsample_bilinear_to_shape

        dof = np.linspace(-0.5, 0.5, 128)[None, :] * np.ones((128, 1))
        panel = build_panel_phase(dof, radius_px=64.0)
        side = int(round(2 * 64.0))
        block = panel[600 - side // 2 : 600 + side // 2, 960 - side // 2 : 960 + side // 2]

        expected = upsample_bilinear_to_shape(dof, (side, side))
        assert np.allclose(block, expected, rtol=1e-12, atol=1e-12)
        # Not a zero-padded copy of the 128 grid: every row/column carries shaped
        # values, so the whole 128 map is stretched over the pupil square.
        assert np.count_nonzero(block[-1]) > 0

    def test_accepts_a_singleton_batch_channel(self) -> None:
        dof = np.zeros((1, 128, 128), dtype=np.float32)
        dof[0, :, :64] = 1.0
        panel = build_panel_phase(torch.from_numpy(dof))
        assert panel.shape == (1200, 1920)

    @pytest.mark.parametrize("shape", [(64, 128), (128, 64), (128, 128, 128), (128,)])
    def test_rejects_a_non_square_map(self, shape) -> None:
        with pytest.raises(ValueError):
            build_panel_phase(np.zeros(shape))


class TestPupilPlacement:
    """Where the 900 px pupil square goes on the panel (README §1.1, G1).

    A wrong centre is not a shifted result: the beam then sees a cropped, off-axis
    piece of the phase map, so the far field is *wrong*.  These tests pin the
    placement and the refusal.
    """

    def test_the_default_is_the_panel_centre(self) -> None:
        panel = build_panel_phase(np.ones((128, 128)))
        rows, cols = np.nonzero(panel)
        assert (rows.min(), rows.max()) == (150, 1049)
        assert (cols.min(), cols.max()) == (510, 1409)
        assert pupil_center() == (960, 600)
        assert panel_corner() == (510, 150)

    def test_an_off_centre_pupil_lands_where_asked(self) -> None:
        panel = build_panel_phase(np.ones((128, 128)), center_xy=(600, 700))
        rows, cols = np.nonzero(panel)
        assert (rows.min(), rows.max()) == (250, 1149)
        assert (cols.min(), cols.max()) == (150, 1049)
        assert pupil_center(center_xy=(600, 700)) == (600, 700)
        assert panel_corner(center_xy=(600, 700)) == (150, 250)
        # And the *content* is the same map, just moved: the block equals the
        # default-centre block.
        default = build_panel_phase(np.ones((128, 128)))
        assert np.array_equal(
            panel[250:1150, 150:1050], default[150:1050, 510:1410]
        )

    @pytest.mark.parametrize("center", [(700, 400), (100, 600), (960, 1100)])
    def test_a_pupil_that_runs_off_the_panel_is_rejected(self, center) -> None:
        with pytest.raises(ValueError, match="runs off"):
            build_panel_phase(np.ones((128, 128)), center_xy=center)


class TestResampleConvention:
    """Measure -- do not assume -- what the 128 -> 900 upsample convention costs.

    ``compose_pupil_dof`` resamples with ``upsample_bilinear_to_shape``
    (``scipy.ndimage.zoom(order=1)`` + ``fit_exact``), which is **endpoint
    aligned**: input sample 0 lands on output pixel 0 and input 127 on pixel 899.
    The twin, by contrast, treats the 128 samples as pixel *centres* of the 900 px
    region.  The two differ by a small, edge-weighted displacement plus a stretch;
    these tests measure it instead of asserting a bound.
    """

    #: Measured on the actual embedding (both axes agree to 1e-14), panel px.
    MAX_SHIFT_PX = 3.0156
    MEAN_SHIFT_PX = 1.5095

    @staticmethod
    def _shift_px() -> dict[str, np.ndarray]:
        """Signed displacement per output pixel, panel px, for each axis.

        A **ramp** is used on purpose: a bilinear resample of a linear ramp is
        exact under any affine convention, so the resampled value *is* the input
        coordinate that convention implies.  The displacement is then read off
        directly -- no second interpolation and no correlation fit to argue about.
        """
        n_in = 128
        side = int(round(2 * PUPIL_RADIUS_PX))
        axis = np.arange(n_in, dtype=np.float64)
        column = np.arange(side, dtype=np.float64)
        pixel_centre = (column + 0.5) * n_in / side - 0.5
        px_per_sample = side / n_in

        ramp_x = np.broadcast_to(axis[None, :], (n_in, n_in)).copy()
        ramp_y = np.broadcast_to(axis[:, None], (n_in, n_in)).copy()
        block_x = build_panel_phase(ramp_x)[150:1050, 510:1410]
        block_y = build_panel_phase(ramp_y)[150:1050, 510:1410]

        return {
            "x": (block_x[0, :] - pixel_centre) * px_per_sample,
            "y": (block_y[:, 0] - pixel_centre) * px_per_sample,
        }

    def test_the_map_is_endpoint_aligned(self) -> None:
        """``out[0] == in[0]`` and ``out[-1] == in[-1]`` -- i.e. not pixel-centred."""
        n_in = 128
        side = int(round(2 * PUPIL_RADIUS_PX))
        ramp = np.broadcast_to(np.arange(n_in, dtype=np.float64)[None, :], (n_in, n_in))
        block = build_panel_phase(ramp.copy())[150:1050, 510:1410]
        row = block[0, :]
        endpoint = np.arange(side, dtype=np.float64) * (n_in - 1) / (side - 1)

        assert row[0] == pytest.approx(0.0, abs=1e-9)
        assert row[-1] == pytest.approx(n_in - 1, abs=1e-9)
        # ... and the whole curve is the endpoint-aligned line, not something else.
        assert np.abs(row - endpoint).max() == pytest.approx(0.0, abs=1e-9)

    def test_measured_displacement_between_the_two_conventions(self) -> None:
        """Max and mean |shift| in panel px, on the real 128 -> 900 map."""
        for axis, shift in self._shift_px().items():
            assert np.abs(shift).max() == pytest.approx(self.MAX_SHIFT_PX, abs=1e-3), axis
            assert np.abs(shift).mean() == pytest.approx(self.MEAN_SHIFT_PX, abs=1e-3), axis
            # The two conventions agree in the middle and disagree at the edges,
            # with opposite signs -- a stretch about the centre, not a translation.
            assert shift[0] > 3.0 and shift[-1] < -3.0
            assert abs(shift[shift.size // 2]) < 0.05

    def test_the_embedded_pitch_is_899_over_127_not_900_over_128(self) -> None:
        """The convention is a **pitch** change: 899/127 px per sample, not 900/128.

        Endpoint alignment maps input sample ``i`` to output pixel ``i * 899/127``,
        so the embedded map's effective pitch is 899/127 = 7.07874 px per grid
        sample while the twin's pixel-centre grid uses 900/128 = 7.03125.  The
        ratio 1.006755 is the whole difference; it is read off the ramp (whose
        resampled value *is* the input coordinate), not assumed.
        """
        n_in, side = 128, int(round(2 * PUPIL_RADIUS_PX))
        axis = np.arange(n_in, dtype=np.float64)
        ramp = np.broadcast_to(axis[None, :], (n_in, n_in)).copy()
        row = build_panel_phase(ramp)[150:1050, 510:1410][0, :]

        samples_per_px = float(row[1] - row[0])  # 127/899
        pitch_px = 1.0 / samples_per_px
        assert pitch_px == pytest.approx(899.0 / 127.0, rel=1e-9)
        assert pitch_px != pytest.approx(900.0 / 128.0, rel=1e-9)

        ratio = pitch_px / (side / n_in)
        assert ratio == pytest.approx(1.006755, abs=1e-6)
        # The map is *stretched* by this much (its pitch is larger, not smaller).
        assert ratio - 1.0 == pytest.approx(0.006755, abs=1e-6)
        # The edge displacement is the same number seen from the centre: half the
        # map (63.5 samples) scaled by the stretch.
        assert abs(ratio - 1.0) * (n_in - 1) / 2.0 * (side / n_in) == pytest.approx(
            self.MAX_SHIFT_PX, abs=0.05
        )

    def test_the_residual_far_field_error_is_a_scale_error_below_one_percent(self) -> None:
        """A +0.6755 % pupil stretch -> a -0.671 % far-field scale error, not a shift.

        Displacing a *shaping* phase map does not translate the focal pattern (it
        changes the map's local gradient), so the consequence of the endpoint
        convention is a **scale** error: the focal pattern shrinks by the reciprocal
        of the pupil pitch ratio, ``1/1.006755 = 0.993291``, i.e. -0.671 %.  On the
        corpus' finest feature (``SynthConfig.min_feature_px`` = 24 CCD px) that is
        ``0.00671 * 24 = 0.16`` CCD px.
        """
        n_in, side = 128, int(round(2 * PUPIL_RADIUS_PX))
        axis = np.arange(n_in, dtype=np.float64)
        ramp = np.broadcast_to(axis[None, :], (n_in, n_in)).copy()
        row = build_panel_phase(ramp)[150:1050, 510:1410][0, :]
        pitch_ratio = (1.0 / float(row[1] - row[0])) / (side / n_in)
        far_field_scale = 1.0 / pitch_ratio
        scale_error = abs(1.0 - far_field_scale)

        assert scale_error < 0.01, (
            f"measured far-field scale error {scale_error:.5f} is not below 1 %"
        )
        assert scale_error == pytest.approx(0.006709, abs=1e-6)
        feature_px = SynthConfig().min_feature_px
        assert feature_px == 24.0
        assert scale_error * feature_px == pytest.approx(0.161, abs=0.002)
        assert scale_error * feature_px < 1.0  # far below one camera pixel


def _sent_grayscale_defs(source: str) -> list[ast.FunctionDef]:
    """Every ``sent_grayscale`` definition in ``source``."""
    return [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "sent_grayscale"
    ]


def _calls_recorded_grayscale(func: ast.FunctionDef) -> bool:
    """Whether ``func``'s body calls ``recorded_grayscale`` by name."""
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "recorded_grayscale"
        for node in ast.walk(func)
    )


class TestRecordedGrayscale:
    """The single panel-grayscale path both instruments share.

    ``recorded_grayscale`` used to be *called* by both ``sent_grayscale`` methods
    without existing anywhere in any revision -- an undefined name that ruff flags
    as F821 and that raises ``NameError`` the moment either method runs.  No test
    covered the recording path, so the suites stayed green while both instruments
    were broken.  These tests call it, pin what each instrument records, and pin
    that neither method re-inlines the composition.
    """

    def test_twin_record_is_the_pure_math_conversion_of_the_embedding(self) -> None:
        from ao_shaping.utils.slm.phase_display import phase_to_slm_grayscale

        phase = np.full((_GRID, _GRID), 0.7, dtype=np.float32)
        recorded = recorded_grayscale(phase)

        assert recorded.dtype == np.uint16
        assert recorded.shape == PANEL_SHAPE
        np.testing.assert_array_equal(
            recorded, phase_to_slm_grayscale(build_panel_phase(phase), slm=None)
        )

    def test_the_record_is_black_outside_the_pupil_square(self) -> None:
        phase = np.full((_GRID, _GRID), 0.7, dtype=np.float32)
        recorded = recorded_grayscale(phase)
        x0, y0 = panel_corner(PANEL_SHAPE, PUPIL_RADIUS_PX, None)
        side = 2 * int(PUPIL_RADIUS_PX)

        assert (x0, y0) != (0, 0)
        assert (recorded[:y0, :] == 0).all(), "rows above the pupil carry the command"
        assert (recorded[y0 + side :, :] == 0).all(), "rows below the pupil carry it"
        assert (recorded[:, :x0] == 0).all(), "columns left of the pupil carry it"

    def test_the_twin_instrument_delegates_to_the_shared_helper(self) -> None:
        instrument = SimInstrument(bench=Sim2f(_ideal_bench(), device="cpu"), device="cpu")
        phase = np.full((_GRID, _GRID), 0.7, dtype=np.float32)

        np.testing.assert_array_equal(
            instrument.sent_grayscale(phase), recorded_grayscale(phase)
        )

    def test_a_live_device_is_forwarded_to_the_conversion(self, monkeypatch) -> None:
        import ao_shaping.utils.slm.phase_display as phase_display

        seen: dict[str, object] = {}

        def _recorder(phase, max_grayscale=None, slm=None):
            seen["panel"] = np.asarray(phase, dtype=np.float64)
            seen["slm"] = slm
            return np.zeros(phase.shape, dtype=np.uint16)

        monkeypatch.setattr(phase_display, "phase_to_slm_grayscale", _recorder)
        device = object()
        phase = np.linspace(0.0, 2.0, _GRID * _GRID, dtype=np.float32).reshape(
            _GRID, _GRID
        )

        recorded_grayscale(
            phase,
            slm=device,
            panel_shape=(256, 256),
            radius_px=64.0,
            center_xy=(128, 128),
        )

        assert seen["slm"] is device, "the live SLM never reached the conversion"
        np.testing.assert_array_equal(
            seen["panel"],
            build_panel_phase(
                phase, panel_shape=(256, 256), radius_px=64.0, center_xy=(128, 128)
            ),
        )

    def test_both_instruments_delegate_instead_of_re_inlining(self) -> None:
        source = Path(recorded_grayscale.__code__.co_filename).read_text(encoding="utf-8")
        bodies = _sent_grayscale_defs(source)

        assert len(bodies) == 2, f"expected twin + hardware, found {len(bodies)}"
        for func in bodies:
            assert _calls_recorded_grayscale(func), (
                f"sent_grayscale at line {func.lineno} re-inlines the grayscale "
                "composition instead of calling recorded_grayscale"
            )

    def test_the_delegation_scan_is_not_vacuous(self) -> None:
        planted = (
            "class HwInstrument:\n"
            "    def sent_grayscale(self, phase_rad):\n"
            "        return phase_to_slm_grayscale(build_panel_phase(phase_rad))\n"
        )
        bodies = _sent_grayscale_defs(planted)

        assert len(bodies) == 1, "the scan saw no planted method"
        assert not _calls_recorded_grayscale(bodies[0]
                                             ), "the planted copy slipped through"


class TestSimInstrument:
    """The twin instrument IS ``Sim2f.forward`` -- nothing more, nothing less."""

    @pytest.fixture(scope="class")
    def bench(self):
        return Sim2f(_ideal_bench(), device="cpu")

    def test_zero_phase_equals_the_plain_forward_pass(self, bench) -> None:
        instrument = SimInstrument(bench=bench, device="cpu")
        zeros = np.zeros((_GRID, _GRID), dtype=np.float32)

        measured = instrument.capture(zeros)
        expected = bench.forward(torch.zeros(1, _GRID, _GRID))[0, 0].numpy()

        assert measured.shape == (_CCD, _CCD)
        np.testing.assert_allclose(measured, expected, rtol=1e-6, atol=0.0)

    def test_a_nonzero_phase_changes_the_frame(self, bench) -> None:
        instrument = SimInstrument(bench=bench, device="cpu")
        zeros = np.zeros((_GRID, _GRID), dtype=np.float32)
        bump = zeros.copy()
        bump[_GRID // 2 - 8 : _GRID // 2 + 8, _GRID // 2 - 8 : _GRID // 2 + 8] = 1.5

        flat = instrument.capture(zeros)
        shaped = instrument.capture(bump)

        assert not np.allclose(flat, shaped)
        assert np.isfinite(shaped).all()

    def test_twin_prediction_is_the_measured_frame(self, bench) -> None:
        instrument = SimInstrument(bench=bench, device="cpu")
        phase = np.full((_GRID, _GRID), 0.7, dtype=np.float32)
        np.testing.assert_array_equal(instrument.capture(phase), instrument.twin(phase))


class TestMeasureSample:
    """A1/B1/B2/A2 -- order, drift, and the noise-floor flag."""

    def _measure(self, instrument, *, index: int = 3, kind: str = "disc") -> dict:
        target = _blob()
        return measure_sample(
            instrument,
            net_phase=np.ones((_GRID, _GRID), dtype=np.float32),
            target=target,
            support=(target > 0.05).to(torch.float32),
            teacher=TeacherConfig(),
            index=index,
            kind=kind,
        )

    def test_identical_reads_give_zero_drift(self) -> None:
        row = self._measure(FakeInstrument(_blob(), _blob()))
        assert row["ncc_a_drift"] == 0.0
        assert row["ncc_flat"] == pytest.approx(row["conditions"]["a1"]["ncc"])
        assert row["ncc_net"] == pytest.approx(row["conditions"]["b1"]["ncc"])
        assert row["indistinguishable"] is False
        assert row["index"] == 3 and row["kind"] == "disc"
        assert set(row["frames"]) == {"a1", "b1", "b2", "a2"}

    def test_net_answering_with_the_target_recovers(self) -> None:
        target = _blob()
        support = (target > 0.05).to(torch.float32)
        rows = []
        for index in range(8):
            row = measure_sample(
                FakeInstrument(_scrambled(), target),
                net_phase=np.ones((_GRID, _GRID), dtype=np.float32),
                target=target,
                support=support,
                teacher=TeacherConfig(),
                index=index,
                kind="square",
            )
            rows.append(row)

        assert all(row["ncc_net"] > row["ncc_flat"] for row in rows)
        assert verdict(rows) == NET_RECOVERS

    def test_a_worse_net_does_not_recover(self) -> None:
        target = _blob()
        support = (target > 0.05).to(torch.float32)
        rows = []
        for index in range(8):
            row = measure_sample(
                FakeInstrument(target, _scrambled()),
                net_phase=np.ones((_GRID, _GRID), dtype=np.float32),
                target=target,
                support=support,
                teacher=TeacherConfig(),
                index=index,
                kind="square",
            )
            rows.append(row)

        assert all(row["ncc_net"] < row["ncc_flat"] for row in rows)
        assert verdict(rows) == NET_DOES_NOT_RECOVER

    def test_a_noisy_floor_makes_the_sample_indistinguishable(self) -> None:
        target = _blob()
        support = (target > 0.05).to(torch.float32)
        rows = []
        for index in range(8):
            row = measure_sample(
                # A1 reads the target, A2 reads something else entirely: the
                # sample's own floor is then as large as the effect being tested.
                FakeInstrument(target, target, flat_alt=_scrambled()),
                net_phase=np.ones((_GRID, _GRID), dtype=np.float32),
                target=target,
                support=support,
                teacher=TeacherConfig(),
                index=index,
                kind="square",
            )
            rows.append(row)

        assert all(row["ncc_a_drift"] > 0.0 for row in rows)
        assert all(row["indistinguishable"] for row in rows)
        # Every sample excluded => no power, which is a different claim from
        # "net does not recover" and must not be reported as one.
        assert verdict(rows) == INSUFFICIENT_POWER

    def test_verdict_counts_wins_and_delta(self) -> None:
        def rows(wins: int, delta: float, loss: float = -0.01, n: int = 8) -> list[dict]:
            made = []
            for index in range(n):
                net = 0.6 + (delta if index < wins else loss)
                made.append(
                    {
                        "index": index,
                        "kind": "square",
                        "ncc_flat": 0.6,
                        "ncc_net": net,
                        "ncc_a_drift": 0.0,
                    }
                )
            return made

        assert verdict(rows(6, 0.10)) == NET_RECOVERS
        assert verdict(rows(5, 0.10)) == NET_DOES_NOT_RECOVER
        # 6 wins but a mean delta of only 0.01: fails the second condition.
        assert verdict(rows(8, 0.01)) == NET_DOES_NOT_RECOVER
        assert verdict(rows(6, 0.10)[:4]) == INSUFFICIENT_POWER

    def test_ncc_agrees_with_the_shared_metric_code(self) -> None:
        from ml.sim2f.loop import _score

        target = _blob()
        support = (target > 0.05).to(torch.float32)
        instrument = FakeInstrument(target, _scrambled())
        row = self._measure(instrument)

        reference = _score(target, target, support, teacher=TeacherConfig())
        assert row["conditions"]["a1"]["ncc"] == pytest.approx(reference["ncc"])
        assert row["conditions"]["a1"]["objective"] == pytest.approx(reference["objective"])

    def test_zero_order_offset_is_a_diagnostic_between_the_two_flat_frames(self) -> None:
        target = _blob()
        # The fake's twin prediction is the flat frame rolled 5 px sideways, so the
        # measured-vs-predicted 0-order offset must read ~5 px.
        rolled = torch.roll(target, 5, dims=-1)
        instrument = FakeInstrument(target, target, twin=rolled)
        row = self._measure(instrument)

        assert 4.0 < row["zero_order_offset_px"] < 6.0
        # Nothing else may move because of it: it is not a criterion.
        assert row["indistinguishable"] is False
        assert verdict([row], min_wins=1) in (NET_RECOVERS, NET_DOES_NOT_RECOVER)

    def test_zero_order_offset_is_zero_on_the_twin(self) -> None:
        bench = Sim2f(_ideal_bench(), device="cpu")
        instrument = SimInstrument(bench=bench, device="cpu")
        target = _blob()
        row = measure_sample(
            instrument,
            net_phase=np.full((_GRID, _GRID), 0.3, dtype=np.float32),
            target=target,
            support=(target > 0.05).to(torch.float32),
            teacher=TeacherConfig(),
            index=0,
            kind="square",
        )
        # The twin *is* the instrument here, so its prediction of the flat frame is
        # the measured flat frame: the offset is exactly 0 by construction.
        assert row["zero_order_offset_px"] == 0.0


class TestGate0:
    """The four pre-registered checks, on synthetic frames."""

    def _frames(self, *, drift: float = 0.0, net_scale: float = 2.0) -> dict:
        axis = np.arange(_CCD, dtype=np.float64)
        rr, cc = np.meshgrid(axis, axis, indexing="ij")
        flat = np.exp(-(((rr - 32) ** 2) + ((cc - 32) ** 2)) / (2 * 3.0**2))
        net = flat.copy()
        net[10:20, 10:30] += net_scale
        block = {
            "a1": flat.copy(),
            "b1": net.copy(),
            "b2": net.copy(),
            "a2": np.roll(flat, int(round(drift)), axis=1),
        }
        return {0: {name: value.astype(np.float32) for name, value in block.items()}}

    @staticmethod
    def _rows(indices: list[int]) -> list[dict]:
        return [
            {
                "index": index,
                "kind": "square",
                "ncc_flat": 0.6,
                "ncc_net": 0.5,
                "ncc_a_drift": 0.0,
            }
            for index in indices
        ]

    def test_a_healthy_digital_session_passes(self) -> None:
        frames = self._frames()
        gate = gate0(
            self._rows([0]),
            frames,
            full_scale=None,
            twin_frames={0: frames[0]["b1"]},
        )

        assert gate["pass"] is True
        assert gate["g0_1_exposure"]["applicable"] is False
        assert gate["g0_3_phase_on_panel"]["pass"] is True
        assert gate["g0_4_geometry"]["min_ncc"] == pytest.approx(1.0, abs=1e-6)

    def test_zero_order_drift_fails_the_stability_gate(self) -> None:
        frames = self._frames(drift=6.0)
        gate = gate0(
            self._rows([0]), frames, full_scale=None, twin_frames={0: frames[0]["b1"]}
        )

        assert gate["g0_2_zero_order"]["pass"] is False
        assert gate["g0_2_zero_order"]["max_drift_px"] > 2.0
        assert gate["pass"] is False

    def test_a_frozen_panel_fails_the_phase_gate(self) -> None:
        frames = self._frames(net_scale=0.0)
        for name in ("b1", "b2"):
            frames[0][name] = frames[0]["a1"].copy()
        gate = gate0(
            self._rows([0]), frames, full_scale=None, twin_frames={0: frames[0]["b1"]}
        )

        assert gate["g0_3_phase_on_panel"]["pass"] is False
        assert gate["pass"] is False

    def test_a_geometry_mismatch_fails_the_identifiability_gate(self) -> None:
        frames = self._frames()
        gate = gate0(
            self._rows([0]),
            frames,
            full_scale=None,
            twin_frames={0: np.roll(frames[0]["b1"], 40, axis=0)},
        )

        assert gate["g0_4_geometry"]["pass"] is False
        assert gate["pass"] is False

    def test_a_missing_twin_prediction_fails_rather_than_passes(self) -> None:
        gate = gate0(self._rows([0]), self._frames(), full_scale=None)
        assert gate["g0_4_geometry"]["pass"] is False
        assert gate["pass"] is False

    def test_the_exposure_gate_reads_a_counted_frame(self) -> None:
        frames = self._frames()
        for name in ("a1", "a2"):
            frames[0][name] = (frames[0][name] / frames[0][name].max() * 255.0).astype(
                np.float32
            )
        gate = gate0(
            self._rows([0]), frames, full_scale=255.0, twin_frames={0: frames[0]["b1"]}
        )
        assert gate["g0_1_exposure"]["applicable"] is True
        # Peak sits at full scale => over the [0.2, 0.8] band.
        assert gate["g0_1_exposure"]["pass"] is False

        dim = self._frames()
        for name in ("a1", "a2"):
            dim[0][name] = (dim[0][name] / dim[0][name].max() * 100.0).astype(np.float32)
        ok = gate0(
            self._rows([0]), dim, full_scale=255.0, twin_frames={0: dim[0]["b1"]}
        )
        assert ok["g0_1_exposure"]["pass"] is True


class _FakeSLM:
    """Minimal Santec stand-in: records every write with its memory slot."""

    def __init__(self, *, panel=(1920, 1200)) -> None:
        # ``Panel_Res`` is (width, height), like the driver's PANEL_RES.
        self.Panel_Res = panel
        self.slm_number = 1
        self.wavelength = 1064
        self.video_mode = 0
        self._serial_number = "FAKE-0001"
        self.displayed_slot = 2
        self.is_open = False
        self.closed = False
        self.writes: list[tuple[np.ndarray, int]] = []

    def open(self) -> None:
        self.is_open = True

    def close(self) -> None:
        self.closed = True

    def get_displayed_memory_number(self) -> int:
        return self.displayed_slot

    def display_phase(self, phase_rad: np.ndarray, memory_number: int | None = None) -> None:
        slot = 0 if memory_number is None else int(memory_number)
        self.displayed_slot = slot
        self.writes.append((np.asarray(phase_rad, dtype=np.float64), slot))


class _FakeCamera:
    """Camera stand-in: a bright 0-order whose height follows the exposure.

    ``counts_per_ms`` is the fake's photometry: the spot's grey level is
    ``min(full_scale, counts_per_ms * exposure_ms)``, so an exposure ladder sees a
    monotone trace and a chosen exposure is visible in the frames.  That is what
    makes "the session uses the exposure the ladder picked" checkable.
    """

    def __init__(
        self,
        *,
        spot=(300, 400),
        full_shape=(600, 800),
        exposure_ms=0.02,
        counts_per_ms=10000.0,
    ) -> None:
        self.spot = spot
        self.full_shape = full_shape
        self.exposure_ms = float(exposure_ms)
        self.counts_per_ms = float(counts_per_ms)
        self.window: tuple[int, int] = full_shape
        self.center = (full_shape[1] // 2, full_shape[0] // 2)
        self.closed = False
        self.exposures: list[float] = []

    def open(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True

    def reset_exposure_time(self, time_ms: float) -> float:
        self.exposures.append(float(time_ms))
        self.exposure_ms = float(time_ms)
        return float(time_ms)

    @property
    def exposure_time_ms(self) -> float:
        """The ABC's name for the cached exposure -- what ``ccd.common`` reads back."""
        return self.exposure_ms

    def reset_window(self, center, size):
        width, height = int(size[0]), int(size[1])
        self.window = (height, width)
        self.center = (int(center[0]), int(center[1]))
        left = max(0, min(self.center[0] - width // 2, self.full_shape[1] - width))
        top = max(0, min(self.center[1] - height // 2, self.full_shape[0] - height))
        self.center = (int(left + width // 2), int(top + height // 2))
        return (width, height), self.center

    @property
    def peak_counts(self) -> int:
        """Spot level at the current exposure, uint8, clipped at full scale."""
        return int(round(min(255.0, self.counts_per_ms * self.exposure_ms)))

    def _frame(self) -> np.ndarray:
        height, width = self.window
        left = self.center[0] - width // 2
        top = self.center[1] - height // 2
        frame = np.zeros((height, width), dtype=np.uint8)
        row = int(self.spot[0]) - top
        col = int(self.spot[1]) - left
        if 0 <= row < height and 0 <= col < width:
            frame[row, col] = self.peak_counts
        return frame

    def get_numpy_image(self, n_sample: int = 1, skip_first: bool = True) -> np.ndarray:
        return self._frame()


@pytest.fixture
def fake_devices(monkeypatch):
    """Patch the two device factories HwInstrument imports at call time."""
    import ao_shaping.drivers.ccd.common as ccd_common
    import ao_shaping.drivers.slm.santec as santec_pkg
    import ao_shaping.utils.slm.phase_display as phase_display

    made: dict[str, object] = {}

    def _slm(**kwargs):
        slm = _FakeSLM()
        slm.video_mode = int(kwargs.get("video_mode", 0))
        made["slm"] = slm
        return slm

    def _cam(*args, **kwargs):
        cam = _FakeCamera(exposure_ms=kwargs.get("exposure_time_ms", 0.02))
        made["cam"] = cam
        return cam

    monkeypatch.setattr(santec_pkg, "Santec", _slm)
    monkeypatch.setattr(ccd_common, "create_camera", _cam)
    monkeypatch.setattr(phase_display, "time", type("T", (), {"sleep": staticmethod(lambda _: None)}))
    monkeypatch.setattr(phase_display, "_last_slm_slot", None)
    return made


class TestHwInstrument:
    """The hardware path, on fake devices: guards, slot rotation, framing."""

    def test_it_refuses_without_the_exposure_ceiling(self, monkeypatch, fake_devices) -> None:
        monkeypatch.delenv("AO_DAHENG_MAX_EXPOSURE_MS", raising=False)

        with pytest.raises(BenchmarkError, match="AO_DAHENG_MAX_EXPOSURE_MS"):
            HwInstrument(exposure_ms=0.02)

        assert fake_devices == {}

    def test_it_refuses_an_exposure_above_the_ceiling(self, monkeypatch, fake_devices) -> None:
        monkeypatch.setenv("AO_DAHENG_MAX_EXPOSURE_MS", "0.5")

        with pytest.raises(BenchmarkError, match="exceeds"):
            HwInstrument(exposure_ms=1.0)

        assert fake_devices == {}

    def test_no_exposure_and_no_ladder_uses_the_default_geometric_ladder(
        self, monkeypatch, fake_devices
    ) -> None:
        monkeypatch.setenv("AO_DAHENG_MAX_EXPOSURE_MS", "0.8")
        instrument = HwInstrument(ccd_px=64, settle_s=0.0)
        try:
            trace = instrument.exposure_trace
            assert trace["source"] == "ladder"
            default = default_exposure_ladder(0.8)
            assert trace["requested_ms"] == pytest.approx(list(default))
            assert len(trace["applied_ms"]) == len(default)
            assert min(trace["applied_ms"]) == pytest.approx(default[0])
            assert trace["chosen_ms"] in trace["applied_ms"]
        finally:
            instrument.close()

    def test_it_writes_the_panel_and_rotates_the_memory_slot(
        self, monkeypatch, fake_devices
    ) -> None:
        monkeypatch.setenv("AO_DAHENG_MAX_EXPOSURE_MS", "0.8")
        instrument = HwInstrument(exposure_ms=0.02, ccd_px=64, settle_s=0.0)
        try:
            slm = fake_devices["slm"]
            assert slm.video_mode == 0  # memory mode only, never DVI
            zeros = np.zeros((_GRID, _GRID), dtype=np.float32)
            first = instrument.capture(zeros)
            second = instrument.capture(np.full((_GRID, _GRID), 2.0, dtype=np.float32))

            assert len(slm.writes) == 2
            # The panel is always the full 1200x1920 radians map, and the pupil
            # square is the 900 px one (450 radius), not a pupil_fill disc.
            panel = slm.writes[0][0]
            assert panel.shape == (1200, 1920)
            assert np.count_nonzero(panel[150:1050, 510:1410]) >= 0
            assert np.count_nonzero(panel[:150]) == 0
            # Consecutive writes must not reuse a slot: the LCOS does not refresh.
            assert slm.writes[0][1] != slm.writes[1][1]

            assert first.shape == (64, 64) and second.shape == (64, 64)
            assert fake_devices["cam"].exposures == [0.02]
        finally:
            instrument.close()
        assert fake_devices["slm"].closed is True
        assert fake_devices["cam"].closed is True

    def test_it_puts_the_pupil_square_where_the_operator_asked(
        self, monkeypatch, fake_devices
    ) -> None:
        """G1: an off-centre pupil lands off-centre, and the metadata says so."""
        monkeypatch.setenv("AO_DAHENG_MAX_EXPOSURE_MS", "0.8")
        instrument = HwInstrument(
            exposure_ms=0.02, ccd_px=64, pupil_center_xy=(600, 700), settle_s=0.0
        )
        try:
            described = instrument.describe()
            assert described["pupil_center_xy"] == [600, 700]
            assert described["pupil_center_is_default"] is False
            assert described["pupil_corner_xy"] == [150, 250]

            instrument.capture(np.ones((128, 128), dtype=np.float32))
            panel = np.asarray(fake_devices["slm"].writes[-1][0])
            rows, cols = np.nonzero(panel)
            assert (rows.min(), rows.max()) == (250, 1149)
            assert (cols.min(), cols.max()) == (150, 1049)
        finally:
            instrument.close()

    def test_the_default_pupil_centre_is_the_panel_centre(
        self, monkeypatch, fake_devices
    ) -> None:
        monkeypatch.setenv("AO_DAHENG_MAX_EXPOSURE_MS", "0.8")
        instrument = HwInstrument(exposure_ms=0.02, ccd_px=64, settle_s=0.0)
        try:
            described = instrument.describe()
            assert described["pupil_center_xy"] == [960, 600]
            assert described["pupil_center_is_default"] is True
            assert described["pupil_corner_xy"] == [510, 150]
        finally:
            instrument.close()

    def test_it_windows_the_camera_on_the_zero_order(self, monkeypatch, fake_devices) -> None:
        monkeypatch.setenv("AO_DAHENG_MAX_EXPOSURE_MS", "0.8")
        instrument = HwInstrument(exposure_ms=0.02, ccd_px=64)
        try:
            described = instrument.describe()
            # The 0-order sits at (300, 400), not at the frame centre (600, 400).
            assert described["camera"]["center_xy"] == [400, 300]
            assert described["camera"]["window"] == [64, 64]
            assert described["camera"]["full_scale"] == 255.0
            assert described["pupil_side_px"] == 900
            frame = instrument.capture(np.zeros((_GRID, _GRID), dtype=np.float32))
            assert frame.max() == pytest.approx(200.0)
        finally:
            instrument.close()

    def test_twin_prediction_comes_from_the_naked_calibrated_twin(
        self, monkeypatch, fake_devices
    ) -> None:
        monkeypatch.setenv("AO_DAHENG_MAX_EXPOSURE_MS", "0.8")
        instrument = HwInstrument(exposure_ms=0.02, ccd_px=64, device="cpu")
        grid = Sim2fConfig.calibrated().grid
        try:
            prediction = instrument.twin(np.zeros((grid, grid), dtype=np.float32))
        finally:
            instrument.close()

        expected = Sim2f(Sim2fConfig.calibrated(), device="cpu").forward(
            torch.zeros(1, grid, grid), detector=False
        )[0, 0].numpy()
        assert prediction.shape == (256, 256)
        np.testing.assert_allclose(prediction, expected, rtol=1e-6, atol=0.0)


class TestPupilPlacementRecord:
    """``session.json`` must say where the pupil went, and whether it was defaulted."""

    def test_hw_record_carries_the_requested_centre(self) -> None:
        from ml.sim2f.bench import _pupil_placement

        record = _pupil_placement("hw", {"panel_shape": [1200, 1920]}, None, (600, 700))
        assert record["pupil_center_xy"] == [600, 700]
        assert record["pupil_center_is_default"] is False
        assert record["pupil_corner_xy"] == [150, 250]
        assert record["panel_center_xy"] == [960, 600]
        assert "panel embedding" in record["applies_to"]

    def test_the_default_record_is_marked_as_the_default(self) -> None:
        from ml.sim2f.bench import _pupil_placement

        record = _pupil_placement("sim", {}, (1200, 1920), None)
        assert record["pupil_center_is_default"] is True
        assert record["pupil_center_xy"] == [960, 600]
        assert "not used" in record["applies_to"]


class TestExposureLadder:
    """G7: the exposure is measured, not guessed -- and the choice is recorded."""

    _LADDER = (0.002, 0.005, 0.01, 0.02, 0.05, 0.1)

    def test_an_explicit_exposure_skips_the_ladder(self, monkeypatch, fake_devices) -> None:
        monkeypatch.setenv("AO_DAHENG_MAX_EXPOSURE_MS", "0.8")
        instrument = HwInstrument(exposure_ms=0.03, ccd_px=64, settle_s=0.0)
        try:
            trace = instrument.exposure_trace
            assert trace["source"] == "explicit"
            assert trace["chosen_ms"] == 0.03
            assert trace["applied_ms"] == []  # nothing was probed
            # Exactly one exposure write: the constructor's.
            assert fake_devices["cam"].exposures == [0.03]
            assert instrument.exposure_ms == 0.03
        finally:
            instrument.close()

    def test_the_ladder_picks_the_step_closest_to_half_full_scale(
        self, monkeypatch, fake_devices
    ) -> None:
        monkeypatch.setenv("AO_DAHENG_MAX_EXPOSURE_MS", "0.8")
        instrument = HwInstrument(
            exposure_ladder=self._LADDER, ccd_px=64, settle_s=0.0
        )
        try:
            trace = instrument.exposure_trace
            cam = fake_devices["cam"]
            # counts_per_ms = 10000 -> peaks 20, 50, 100, 200, 255, 255 of 255.
            assert trace["peak_counts"] == [20.0, 50.0, 100.0, 200.0, 255.0, 255.0]
            assert trace["full_scale"] == 255.0
            assert trace["chosen_ms"] == pytest.approx(0.01)  # |100/255 - 0.5| wins
            assert trace["chosen_peak_fraction"] == pytest.approx(100.0 / 255.0)
            assert trace["target_fraction"] == 0.5
            # The camera was left at the chosen step, not at the ladder's last one.
            assert cam.exposure_ms == pytest.approx(0.01)
            assert cam.exposures[-1] == pytest.approx(0.01)
        finally:
            instrument.close()

    def test_the_session_uses_the_chosen_exposure(self, monkeypatch, fake_devices) -> None:
        monkeypatch.setenv("AO_DAHENG_MAX_EXPOSURE_MS", "0.8")
        instrument = HwInstrument(
            exposure_ladder=self._LADDER, ccd_px=64, settle_s=0.0
        )
        try:
            chosen = instrument.exposure_ms
            frame = instrument.capture(np.zeros((_GRID, _GRID), dtype=np.float32))
            assert instrument.describe()["camera"]["exposure_ms"] == chosen
            # The spot height IS the exposure: frames are taken at the chosen step.
            assert float(frame.max()) == pytest.approx(
                min(255.0, 10000.0 * chosen), abs=0.5
            )
            assert chosen != pytest.approx(self._LADDER[-1])
        finally:
            instrument.close()

    def test_the_chosen_exposure_sits_inside_the_g0_1_band(
        self, monkeypatch, fake_devices
    ) -> None:
        monkeypatch.setenv("AO_DAHENG_MAX_EXPOSURE_MS", "0.8")
        instrument = HwInstrument(
            exposure_ladder=self._LADDER, ccd_px=64, settle_s=0.0
        )
        try:
            trace = instrument.exposure_trace
            assert 0.2 <= trace["chosen_peak_fraction"] <= 0.8
        finally:
            instrument.close()

    def test_an_out_of_band_ladder_fails_g0_1_with_the_trace(
        self, monkeypatch, fake_devices
    ) -> None:
        """A ladder that cannot reach 0.2 x full scale must fail, not be patched."""
        monkeypatch.setenv("AO_DAHENG_MAX_EXPOSURE_MS", "0.8")
        instrument = HwInstrument(
            exposure_ladder=(0.0002, 0.0005), ccd_px=64, settle_s=0.0
        )
        try:
            trace = instrument.exposure_trace
            assert trace["chosen_ms"] == pytest.approx(0.0005)
            assert trace["chosen_peak_fraction"] < 0.2
            # The out-of-band exposure is kept, not silently replaced: G0-1 is
            # what reports it, with the whole trace attached.
            dim = np.zeros((_CCD, _CCD), dtype=np.float32)
            dim[32, 32] = float(trace["chosen_peak_fraction"]) * 255.0
            bright = dim.copy()
            bright[10, 10] = 40.0
            frames = {0: {"a1": dim, "b1": bright, "b2": bright, "a2": dim}}
            gate = gate0(
                [{"index": 0, "kind": "square", "ncc_flat": 0.6, "ncc_net": 0.5,
                  "ncc_a_drift": 0.0}],
                frames,
                full_scale=255.0,
                twin_frames={0: bright},
                exposure_ladder=trace,
            )
            assert gate["g0_1_exposure"]["pass"] is False
            assert gate["g0_1_exposure"]["exposure_ladder"] is not None
            assert gate["g0_1_exposure"]["exposure_ladder"]["chosen_ms"] == pytest.approx(
                trace["chosen_ms"]
            )
            assert gate["pass"] is False
        finally:
            instrument.close()

    @pytest.mark.parametrize("ladder", [(0.02,), (0.0, 0.02), (0.02, float("nan"))])
    def test_a_bad_ladder_is_refused(self, monkeypatch, fake_devices, ladder) -> None:
        monkeypatch.setenv("AO_DAHENG_MAX_EXPOSURE_MS", "0.8")
        with pytest.raises(BenchmarkError):
            HwInstrument(exposure_ladder=ladder, ccd_px=64)
        assert fake_devices == {}  # refused before any device was touched

    def test_a_ladder_above_the_cap_is_refused(self, monkeypatch, fake_devices) -> None:
        monkeypatch.setenv("AO_DAHENG_MAX_EXPOSURE_MS", "0.1")
        with pytest.raises(BenchmarkError, match="exceeds"):
            HwInstrument(exposure_ladder=(0.01, 0.2), ccd_px=64)
        assert fake_devices == {}

    def test_the_default_ladder_spans_cap_64_to_cap(self) -> None:
        ladder = default_exposure_ladder(0.8)
        assert len(ladder) == 8
        assert ladder[0] == pytest.approx(0.8 / 64.0)
        assert ladder[-1] == pytest.approx(0.8)
        assert all(right > left for left, right in zip(ladder, ladder[1:]))


class TestSantecConstruction:
    """G3: the constructor HwInstrument uses must be the driver's real one."""

    def test_the_kwargs_match_the_real_driver_signature(self) -> None:
        import inspect

        try:
            from ao_shaping.drivers.slm.santec import Santec
        except (ImportError, OSError) as exc:  # vendor DLL not installed here
            pytest.skip(f"SLM driver unavailable: {exc}")

        parameters = inspect.signature(Santec.__init__).parameters
        for name in ("slm_number", "wavelength", "video_mode"):
            assert name in parameters, (
                f"Santec.__init__ has no {name!r} parameter; HwInstrument passes it. "
                f"Real signature: {list(parameters)}"
            )
        # ``video_mode`` must default to memory mode, and 0 IS VideoMode.Memory:
        # DVI mode can hang the controller until a physical power cycle.
        assert parameters["video_mode"].default == 0
        assert "from_params" in dir(Santec)

    def test_from_params_is_the_alternative_and_is_equivalent(self) -> None:
        """Documents why direct construction was chosen over ``from_params``."""
        import inspect

        try:
            from ao_shaping.drivers.slm.santec import Santec
        except (ImportError, OSError) as exc:
            pytest.skip(f"SLM driver unavailable: {exc}")

        factory = inspect.signature(Santec.from_params)
        assert "params" in factory.parameters
        # from_params reads slm_number / slm_wavelength off a params object and maps
        # wavelength 0 -> None, exactly what HwInstrument does by hand; it needs a
        # carrier object this module has no other use for, hence direct construction.
        assert "slm_wavelength" in inspect.getsource(Santec.from_params)


class TestSelectIndices:
    """``--samples`` selection must match the published §5 set exactly."""

    class _Corpus:
        n_samples = 64

        def indices(self, split: str):
            assert split == "test"
            return torch.arange(0, 8, dtype=torch.long)

    def test_test_split(self) -> None:
        assert select_indices(self._Corpus(), "test") == list(range(8))

    def test_loop_set_is_the_published_one(self) -> None:
        expected = torch.randperm(64, generator=torch.Generator().manual_seed(7))[:24]
        got = select_indices(self._Corpus(), "loop")
        assert got == expected.tolist()
        assert got[:4] == [47, 14, 15, 18]

    def test_limit_and_unknown_name(self) -> None:
        assert select_indices(self._Corpus(), "test", limit=3) == [0, 1, 2]
        with pytest.raises(ValueError, match="unknown sample set"):
            select_indices(self._Corpus(), "nope")


class TestPublishedReproduction:
    """``--samples loop`` on ``--backend sim`` must reproduce the published table.

    The reference is ``runs/sim2f_fair/loop/a_no_noise/eval_loop.json`` (the
    ``flat`` and ``net_open`` arms).  It lives under the gitignored ``runs/``
    tree, so the test skips when the artefact is absent rather than inventing a
    substitute: a stand-in would only pin the stand-in.

    A mismatch is a finding, not a tolerance problem -- do not widen it.
    """

    _PUBLISHED = "runs/sim2f_fair/loop/a_no_noise/eval_loop.json"
    _CHECKPOINT = "runs/sim2f_fair/train/model_best.pt"
    _CACHE = "runs/sim2f_fair/corpus"
    #: The floor the task sets: relative 1e-4 on ``objective`` and ``ncc``.
    _TOLERANCE = 1e-4

    @pytest.fixture(scope="class")
    def payload(self):
        from pathlib import Path

        root = Path(__file__).resolve().parents[4]
        for relative in (self._PUBLISHED, self._CHECKPOINT, self._CACHE):
            if not (root / relative).exists():
                pytest.skip(f"{relative} is absent (gitignored runs/ tree)")
        return run_session(
            checkpoint=root / self._CHECKPOINT,
            cache=root / self._CACHE,
            samples="loop",
            out_dir=root / "runs" / "sim2f_bench" / "_pytest",
            backend="sim",
        )

    def test_flat_and_net_open_reproduce(self, payload) -> None:
        import json
        from pathlib import Path

        published = json.loads(
            (Path(__file__).resolve().parents[4] / self._PUBLISHED).read_text(
                encoding="utf-8"
            )
        )["aggregate"]
        mine = payload["aggregate"]

        for arm, reference in (("flat", "flat"), ("net", "net_open")):
            for column in ("objective", "ncc"):
                got = mine[arm][column]
                want = published[reference][column]
                assert got == pytest.approx(want, rel=self._TOLERANCE, abs=0.0), (
                    f"{arm}.{column}: {got!r} vs published {want!r} "
                    f"(rel {abs(got - want) / abs(want):.2e})"
                )

    def test_it_is_bit_exact_on_the_published_run_s_device(self, payload) -> None:
        import json
        from pathlib import Path

        if not torch.cuda.is_available():
            pytest.skip("published run used CUDA; float32 kernels differ on CPU")

        published = json.loads(
            (Path(__file__).resolve().parents[4] / self._PUBLISHED).read_text(
                encoding="utf-8"
            )
        )["aggregate"]
        mine = payload["aggregate"]

        assert mine["flat"]["objective"] == published["flat"]["objective"]
        assert mine["flat"]["ncc"] == published["flat"]["ncc"]
        assert mine["net"]["objective"] == published["net_open"]["objective"]
        assert mine["net"]["ncc"] == published["net_open"]["ncc"]

    def test_the_session_reports_the_published_verdict(self, payload) -> None:
        # On the twin, ``net`` loses to ``flat``: that is the published §5
        # conclusion (7/7 families), so a dry run that "recovers" would mean the
        # measurement, not the bench, changed.
        assert payload["gate0"]["pass"] is True
        assert payload["verdict"] == NET_DOES_NOT_RECOVER
        assert payload["criterion"]["counted"] == 24

    def test_artefacts_exist(self, payload) -> None:
        from pathlib import Path

        out = Path(__file__).resolve().parents[4] / "runs" / "sim2f_bench" / "_pytest"
        for name in ("session.json", "rows.csv", "frames.npz", "fetch.png", "VERDICT.md"):
            assert (out / name).is_file(), name
        text = (out / "VERDICT.md").read_text(encoding="utf-8")
        assert payload["verdict"] in text

    def test_the_session_records_where_the_pupil_went(self, payload) -> None:
        placement = payload["pupil_placement"]
        assert placement["pupil_center_is_default"] is True
        assert placement["pupil_center_xy"] == [960, 600]
        assert placement["pupil_corner_xy"] == [510, 150]
        assert placement["panel_shape"] == [1200, 1920]
        # The sim backend never embeds into a panel, and the JSON says so rather
        # than implying a placement that did not happen.
        assert "not used" in placement["applies_to"]

    def test_rows_carry_the_zero_order_diagnostic(self, payload) -> None:
        for row in payload["rows"]:
            assert "zero_order_offset_px" in row
            assert row["zero_order_offset_px"] == pytest.approx(0.0, abs=1e-6)
        from pathlib import Path

        rows_csv = (
            Path(__file__).resolve().parents[4] / "runs" / "sim2f_bench" / "_pytest" / "rows.csv"
        ).read_text(encoding="utf-8")
        assert "zero_order_offset_px" in rows_csv.splitlines()[0]
