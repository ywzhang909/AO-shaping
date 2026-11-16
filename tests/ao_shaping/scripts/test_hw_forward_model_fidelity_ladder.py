"""Contracts for the hardware forward-model fidelity ladder.

Everything here is offline. The ladder's whole point is a *hardware* run, so the
tests deliberately pin the pieces a broken dry-run would silently paper over:

* the ``--dry-run`` end-to-end path really touches no driver package (exit 0 is
  the proof: ``run_dry_run`` calls ``assert_no_drivers``, which exits 3 on any
  ``ao_shaping.drivers.*`` import). Run in a **subprocess** so a sibling test
  that imported a driver cannot mask the guarantee;
* the zero-fill coverage mask excludes exactly the cells ``resample_ccd_to_model``
  fabricated as 0, and is a no-op when the model far-field fits on the sensor;
* ``spearman_signal`` (the primary judge) recovers a frame synthesised *from* the
  prediction, while the coverage-wide Spearman stays noise-floor dominated — the
  reason the primary metric is not ``spearman_coverage``;
* Noll blocks are finite (``generate_noll_polynomial`` returns NaN outside the
  unit circle) and foreign-grid model phases resample centre-anchored.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.slm import _hw_fidelity_ladder_sections as sec
from scripts.slm.hw_forward_model_fidelity_ladder import build_parser, main

ROOT = Path(__file__).resolve().parents[3]
LADDER_SCRIPT = ROOT / "scripts" / "slm" / "hw_forward_model_fidelity_ladder.py"

# A small grid keeps every test well under a second; padding=2 is the default and
# makes ``block_centre == (M-1)/2`` — the identity ``resample_model_grid`` leans on.
N_PUPIL = 64
PADDING = 2
PUPIL_RADIUS = 384.0
PUPIL_CENTRE = (480.0, 400.0)
W0_PX = 150.0
K_MEAN = 7510.61

# Frame big enough that the model far-field array fits entirely on the sensor.
BIG_FRAME = (800, 800)
BIG_CENTRE = (400.0, 400.0)
# Frame too small: the outer model cells fall off the sensor and get zero-filled.
SMALL_FRAME = (300, 300)
SMALL_CENTRE = (150.0, 150.0)


@pytest.fixture(scope="module")
def opt():
    """The optimisation module the ladder reuses (loaded by path, driver-free)."""
    return sec.load_opt_module()


@pytest.fixture(scope="module")
def spec(opt):
    return opt.build_model_spec(
        pupil_radius=PUPIL_RADIUS,
        pupil_center=PUPIL_CENTRE,
        n_pupil=N_PUPIL,
        padding=PADDING,
        w0_px=W0_PX,
    )


def _gaussian_pred(spec, sigma: float) -> np.ndarray:
    """A centred Gaussian on the model grid, i.e. a stand-in far-field intensity."""
    m = int(spec.m_grid)
    c = int(spec.centre_index)
    ax = np.arange(m, dtype=np.float64) - c
    yy, xx = np.meshgrid(ax, ax, indexing="ij")
    return np.exp(-(xx**2 + yy**2) / (2.0 * float(sigma) ** 2))


# ---------------------------------------------------------------------------
# CLI / token parsing
# ---------------------------------------------------------------------------
def test_parse_pattern_tokens_builtin_all_file_and_dir(tmp_path: Path) -> None:
    """``--patterns`` accepts builtin names, ``all``, ``name=path``, bare paths and ``dir:``."""
    phase = tmp_path / "myphase.npy"
    np.save(phase, np.zeros((N_PUPIL, N_PUPIL)))

    tokens = sec.parse_pattern_tokens("flat,tilt_x")
    assert tokens == [("builtin", "flat"), ("builtin", "tilt_x")]

    all_tokens = sec.parse_pattern_tokens("all")
    assert [t[1] for t in all_tokens] == list(sec.BUILTIN_PATTERN_IDS)

    named = sec.parse_pattern_tokens(f"custom={phase}")
    assert named == [("file", ("custom", phase))]

    bare = sec.parse_pattern_tokens(str(phase))
    assert bare == [("file", (phase.stem, phase))]

    dirs = sec.parse_pattern_tokens(f"dir:{tmp_path}")
    assert dirs == [("dir", tmp_path)]

    # --custom-phase-dir is sugar for appending one more dir: token.
    with_dir = sec.parse_pattern_tokens("flat", custom_phase_dir=tmp_path)
    assert with_dir == [("builtin", "flat"), ("dir", tmp_path)]


def test_parse_pattern_tokens_rejects_unknown_and_empty() -> None:
    """A typo must fail loudly, not be silently dropped from the ladder."""
    with pytest.raises(ValueError, match="未知图样"):
        sec.parse_pattern_tokens("flat,not_a_pattern")
    with pytest.raises(ValueError, match="为空"):
        sec.parse_pattern_tokens("  ,  ")


# ---------------------------------------------------------------------------
# Noll blocks and grid resampling
# ---------------------------------------------------------------------------
def test_noll_phase_block_is_finite_and_zero_outside_the_pupil(spec) -> None:
    """``generate_noll_polynomial`` returns NaN outside the unit circle; the block must not.

    A NaN would reach ``process_pattern`` and blow up (or, worse, survive into the
    model prediction) — the SLM commands 0 phase outside the aperture, so 0 is the
    documented convention (same as ``absolute_fidelity_probe_20261116.noll_phase``).
    """
    for name, (zn, zm, sign) in sec.NOLL_SPECS.items():
        block = sec.noll_phase_block(zn, zm, N_PUPIL, sign * 1.0)
        assert block.shape == (N_PUPIL, N_PUPIL), name
        assert np.isfinite(block).all(), f"{name} still carries NaN/Inf"
        # Corners are outside the unit circle -> exactly 0.
        assert block[0, 0] == 0.0 and block[-1, -1] == 0.0, name
        # And the pattern is not identically zero (the amplitude survived).
        assert float(np.abs(block).max()) > 0.0, name


def test_resample_model_grid_keeps_the_optical_axis_centred(spec) -> None:
    """A foreign-grid phase resamples by scaling about ``(M-1)/2``, not about a corner.

    Corner-anchored zoom would shift the pupil by half the grid and every
    ``optimized_final`` metric from a 1024-grid history would be garbage.
    """
    m_dst = int(spec.m_grid)
    m_src = 4 * m_dst
    ii, jj = np.mgrid[0:m_src, 0:m_src]
    # Distance from the *geometric* centre of the source grid.
    src = np.abs(ii - (m_src - 1) / 2.0) + np.abs(jj - (m_src - 1) / 2.0)

    out = sec.resample_model_grid(src, spec)
    assert out.shape == (m_dst, m_dst)
    # Centre-anchored nearest-neighbour zoom: a source that is symmetric about its
    # own centre stays symmetric about the destination centre, up to the one
    # nearest-neighbour step that an even-sized grid costs. (A corner-anchored
    # ``src = i * ratio`` would be off by ratio/2 == 2 samples here.)
    assert np.abs(out - out[::-1, :]).max() <= 1.0, "rows lost the centre anchor"
    assert np.abs(out - out[:, ::-1]).max() <= 1.0, "cols lost the centre anchor"
    # The optical axis (the source's minimum) lands on the destination centre.
    assert int(np.argmin(out.sum(axis=1))) in (
        int(np.floor((m_dst - 1) / 2.0)),
        int(np.ceil((m_dst - 1) / 2.0)),
    )
    # ...and the full source extent is sampled: the corner keeps the corner's value
    # (the source's largest distance), not some cropped inner ring.
    assert float(out[0, 0]) == pytest.approx(float(src[0, 0]), abs=2.0 * (m_src / m_dst))


def test_resample_model_grid_identity_and_shape_guard(spec) -> None:
    """Same grid => an equal *copy*; non-square input => a clear error."""
    same = np.arange(int(spec.m_grid) ** 2, dtype=np.float64).reshape(
        int(spec.m_grid), int(spec.m_grid)
    )
    out = sec.resample_model_grid(same, spec)
    assert out is not same
    assert np.array_equal(out, same)

    with pytest.raises(ValueError, match="方形"):
        sec.resample_model_grid(np.zeros((8, 16)), spec)


# ---------------------------------------------------------------------------
# Coverage mask <-> zero-fill pairing
# ---------------------------------------------------------------------------
def test_coverage_mask_is_a_noop_when_the_far_field_fits_on_the_sensor(opt, spec) -> None:
    """The dry-run stand-in geometry: 100 % coverage, so the mask must not clip anything."""
    cov = sec.coverage_mask(
        frame_shape=BIG_FRAME, centre_xy=BIG_CENTRE, k_mean=K_MEAN, spec=spec
    )
    assert cov.shape == (int(spec.m_grid), int(spec.m_grid))
    assert cov.all(), "model far-field fits inside the frame; nothing should be masked"


def test_coverage_mask_excludes_exactly_the_zero_filled_cells(opt, spec) -> None:
    """Out-of-sensor model cells are fabricated 0s; the mask must flag them.

    A random frame (not a constant one — ``prepare_frame`` subtracts the median)
    makes the zero-fill visible: ``meas`` is exactly 0 off-coverage and non-zero
    well inside it.
    """
    rng = np.random.default_rng(7)
    frame = rng.uniform(0.0, 100.0, size=SMALL_FRAME)

    cov = sec.coverage_mask(
        frame_shape=SMALL_FRAME, centre_xy=SMALL_CENTRE, k_mean=K_MEAN, spec=spec
    )
    assert not cov.all(), "this frame is deliberately too small"
    assert cov.any()

    meas, cov2 = sec.resample_with_coverage(
        opt, frame, SMALL_CENTRE, K_MEAN, spec
    )
    assert np.array_equal(cov, cov2)
    assert meas.shape == cov.shape
    # Every excluded cell is a zero-fill, never a measurement.
    assert np.count_nonzero(meas[~cov]) == 0
    # ...and the flagged cells really are outside the sensor, so the interior is
    # not being thrown away.
    interior = cov.copy()
    interior[0, :] = interior[-1, :] = False
    interior[:, 0] = interior[:, -1] = False
    assert float((meas[interior] > 0).mean()) > 0.5


# ---------------------------------------------------------------------------
# Metrics: the reason spearman_signal is the primary judge
# ---------------------------------------------------------------------------
def test_signal_floor_spearman_recovers_a_frame_built_from_the_prediction(
    opt, spec
) -> None:
    """A frame synthesised *from* the prediction must score ~1 on the primary judge.

    The same pair, ranked over the whole coverage mask, is dominated by the noise
    floor (>99 % of covered cells sit below 1e-3 of peak) and scores far lower —
    which is exactly why ``spearman_coverage`` must not be used as the verdict.
    """
    pred = _gaussian_pred(spec, sigma=8.0)
    frame = sec.synthetic_ccd_frame(
        pred,
        frame_shape=BIG_FRAME,
        centre_xy=BIG_CENTRE,
        k_mean=K_MEAN,
        spec=spec,
        seed=3,
    )
    meas, cov = sec.resample_with_coverage(opt, frame, BIG_CENTRE, K_MEAN, spec)
    assert cov.all()

    row = sec.pattern_metrics(
        opt, pred=pred, meas=meas, coverage=cov, k_mean=K_MEAN, spec=spec
    )

    assert row["signal_pixels"] > 0
    assert row["signal_floor"] == sec.DEFAULT_SIGNAL_FLOOR
    # Signal region: the synthetic frame is the prediction itself, so the rank
    # order must survive the CCD noise (peak 2000 counts, std 6).
    assert row["spearman_signal"] > 0.95, row["spearman_signal"]
    assert row["pearson_signal"] > 0.9, row["pearson_signal"]
    # Coverage-wide: mostly noise floor => the documented failure mode.
    assert row["spearman_coverage"] < row["spearman_signal"], (
        row["spearman_coverage"],
        row["spearman_signal"],
    )
    assert row["spearman_coverage"] < 0.9, row["spearman_coverage"]
    # A frame built from the prediction has no FWHM/offset disagreement to speak of.
    assert row["fwhm_ratio"] == pytest.approx(1.0, abs=0.35)
    assert row["peak_offset_model_px"] < 2.0
    # Identity orientation: the synthetic canvas is generated in model orientation.
    assert row["orientation_match"] == "identity"


def test_signal_floor_is_tighter_than_the_coverage_metric(opt, spec) -> None:
    """Raising the floor shrinks the signal set and cannot loosen the noise floor in."""
    pred = _gaussian_pred(spec, sigma=8.0)
    frame = sec.synthetic_ccd_frame(
        pred,
        frame_shape=BIG_FRAME,
        centre_xy=BIG_CENTRE,
        k_mean=K_MEAN,
        spec=spec,
        seed=11,
    )
    meas, cov = sec.resample_with_coverage(opt, frame, BIG_CENTRE, K_MEAN, spec)

    loose = sec.pattern_metrics(
        opt, pred=pred, meas=meas, coverage=cov, k_mean=K_MEAN, spec=spec,
        signal_floor=1e-4,
    )
    tight = sec.pattern_metrics(
        opt, pred=pred, meas=meas, coverage=cov, k_mean=K_MEAN, spec=spec,
        signal_floor=1e-2,
    )
    assert tight["signal_pixels"] < loose["signal_pixels"]
    assert tight["signal_frac_of_coverage"] < loose["signal_frac_of_coverage"]


def test_pattern_metrics_rejects_a_flat_prediction(opt, spec) -> None:
    """No signal at all => an all-zero signal mask, not a division by zero."""
    pred = np.zeros((int(spec.m_grid), int(spec.m_grid)))
    meas = np.full((int(spec.m_grid), int(spec.m_grid)), 3.0)
    cov = np.ones_like(pred, dtype=bool)
    row = sec.pattern_metrics(
        opt, pred=pred, meas=meas, coverage=cov, k_mean=K_MEAN, spec=spec
    )
    assert row["signal_pixels"] == 0
    # Degenerate input must not crash the ladder: every key is present, values may
    # legitimately be NaN (a flat prediction carries no rank information).
    for key in ("spearman_signal", "pearson_signal", "spearman_coverage", "fwhm_ratio"):
        assert key in row
    assert row["signal_frac_of_coverage"] == 0.0


# ---------------------------------------------------------------------------
# End-to-end dry run (subprocess: a clean interpreter is the only honest way to
# assert "no driver was ever imported").
# ---------------------------------------------------------------------------
def test_dry_run_end_to_end_touches_no_driver(tmp_path: Path) -> None:
    """``--dry-run`` runs the whole ladder offline and exits 0 with its artifacts.

    Exit 0 *is* the no-driver proof: ``run_dry_run`` finishes with
    ``assert_no_drivers``, which raises ``SystemExit(3)`` the moment any
    ``ao_shaping.drivers.*`` module is in ``sys.modules``.
    """
    out = tmp_path / "ladder"
    proc = subprocess.run(
        [
            sys.executable,
            str(LADDER_SCRIPT),
            "--dry-run",
            "--patterns",
            "flat,tilt_x",
            "--n-pupil",
            str(N_PUPIL),
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "DRY-RUN 违约" not in proc.stderr
    assert "ao_shaping.drivers" not in proc.stdout

    # Artifacts the report and the next device session depend on.
    for rel in (
        "ladder_summary.md",
        "metrics.json",
        "metrics.csv",
        "plan.json",
        "figures/fidlad_correlation_bars.png",
        "figures/fidlad_radial_profiles.png",
    ):
        assert (out / rel).is_file(), f"missing artifact {rel}"

    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    rows = metrics["rows"] if isinstance(metrics, dict) else metrics
    ids = [r["pattern_id"] for r in rows]
    assert ids == ["flat", "tilt_x"]
    for row in rows:
        # The primary judge is present, finite and in range for a synthetic frame.
        assert 0.0 <= float(row["spearman_signal"]) <= 1.0, row["pattern_id"]
        assert int(row["signal_pixels"]) >= 0
        assert float(row["signal_floor"]) == sec.DEFAULT_SIGNAL_FLOOR

    plan = json.loads((out / "plan.json").read_text(encoding="utf-8"))
    assert plan["mode"] == "dry-run"
    assert plan["predict_from"] == "model"
    assert float(plan["signal_floor"]) == sec.DEFAULT_SIGNAL_FLOOR
    assert plan["model"]["n_pupil"] == N_PUPIL
    # The plan's grid is internally consistent with the padding the parser defaults to.
    padding = int(build_parser().parse_args([]).model_padding)
    assert plan["model"]["m_grid"] == N_PUPIL * padding

    # Per-pattern phase / prediction / frame dumps.
    for pid in ids:
        assert (out / "phase" / f"{pid}.npy").is_file(), pid
        assert (out / "pred" / f"{pid}.npy").is_file(), pid
        assert (out / "frames" / f"{pid}.npy").is_file(), pid


def test_execute_and_dry_run_are_mutually_exclusive() -> None:
    """The two modes cannot both be claimed on one command line."""
    with pytest.raises(SystemExit) as exc:
        main(["--execute", "--dry-run"])
    assert "互斥" in str(exc.value)


def test_dry_run_is_the_default_mode() -> None:
    """No ``--execute`` means no device is opened — the parser default must be off."""
    args = build_parser().parse_args([])
    assert args.execute is False
    assert args.dry_run is False
