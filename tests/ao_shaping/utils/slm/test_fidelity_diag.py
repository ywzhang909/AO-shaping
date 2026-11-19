"""Tests for the seven-factor fidelity decomposition (issue #156, R-87).

Reverse-validation mirrors the source paper: a perfect simulator must score
every factor at 1.0, and each injected defect must depress exactly the factor
that names it — that is what makes the diagnostic actionable.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from ao_shaping.utils.slm.fidelity_diag import (
    DEFAULT_THRESHOLD,
    MITIGATIONS,
    FidelityReport,
    diagnose,
    gamma_amplitude_only,
    gamma_calibration,
    gamma_decorrelation,
    gamma_jitter,
    gamma_mode_coupling,
    gamma_noise,
    gamma_unmodulated,
    step_dft,
    theoretical_ceiling,
)

MODULE_PATH = (
    Path(__file__).resolve().parents[4]
    / "src"
    / "ao_shaping"
    / "utils"
    / "slm"
    / "fidelity_diag.py"
)


class TestCeiling:
    def test_single_mode_has_no_enhancement(self) -> None:
        assert theoretical_ceiling(1) == pytest.approx(1.0)

    def test_pi_over_four_bound(self) -> None:
        assert theoretical_ceiling(2) == pytest.approx(1.0 + np.pi / 4.0)
        assert theoretical_ceiling(100) == pytest.approx(1.0 + np.pi / 4.0 * 99.0)
        # ~21 % below N, the whole point of the corrected acceptance criterion
        assert theoretical_ceiling(100) / 100.0 == pytest.approx(0.7875, abs=1e-4)

    def test_rejects_nonpositive_modes(self) -> None:
        with pytest.raises(ValueError):
            theoretical_ceiling(0)


class TestFactorKernels:
    def test_mode_coupling_clips_into_unit_interval(self) -> None:
        assert gamma_mode_coupling(0.42) == pytest.approx(0.42)
        assert gamma_mode_coupling(1.3) == pytest.approx(1.0)
        assert gamma_mode_coupling(-0.2) == pytest.approx(0.0)
        with pytest.raises(ValueError):
            gamma_mode_coupling(float("nan"))

    def test_amplitude_only_uniform_transmission_is_lossless(self) -> None:
        assert gamma_amplitude_only([1.0, 1.0, 1.0, 1.0]) == pytest.approx(1.0)

    def test_amplitude_only_half_dark_pupil_costs_half(self) -> None:
        assert gamma_amplitude_only([1.0, 0.0, 1.0, 0.0]) == pytest.approx(0.5)

    def test_amplitude_only_rejects_degenerate_input(self) -> None:
        with pytest.raises(ValueError):
            gamma_amplitude_only([])
        with pytest.raises(ValueError):
            gamma_amplitude_only([0.0, 0.0])

    def test_unmodulated_round_trip(self) -> None:
        for gamma0_sq in (0.0, 0.01, 0.0625):
            gamma0 = float(np.sqrt(gamma0_sq))
            # |F1|/|F0| = gamma0 * sqrt(1 - gamma0**2) for pure unmodulated light
            ratio = gamma0 * float(np.sqrt(1.0 - gamma0_sq))
            f1 = complex(ratio, 0.0)
            assert gamma_unmodulated(1.0 + 0j, f1) == pytest.approx(gamma0_sq, abs=1e-9)

    def test_unmodulated_rejects_non_stepping_series(self) -> None:
        with pytest.raises(ValueError):
            gamma_unmodulated(0j, 1.0 + 0j)
        with pytest.raises(ValueError):
            gamma_unmodulated(1.0 + 0j, 0.8 + 0j)

    def test_calibration_ideal_ramp_is_lossless(self) -> None:
        ramp = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
        assert gamma_calibration(ramp) == pytest.approx(1.0)
        assert gamma_calibration(ramp, ramp) == pytest.approx(1.0)

    def test_calibration_constant_offset_is_not_a_loss(self) -> None:
        ramp = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
        assert gamma_calibration(ramp + 0.7, ramp) == pytest.approx(1.0)

    def test_calibration_penalises_nonlinear_response(self) -> None:
        n = 256
        ideal = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        steps = np.arange(n, dtype=np.float64)
        # the paper's injected defect: arange(256)/128*pi*1.2 (1.2x too steep)
        measured = steps / 128.0 * np.pi * 1.2
        value = gamma_calibration(measured, ideal)
        assert value < DEFAULT_THRESHOLD
        assert value == pytest.approx(np.sinc(1.23 / (2.0 * np.pi)) ** 2, abs=2e-2)

    def test_calibration_rejects_mismatched_inputs(self) -> None:
        with pytest.raises(ValueError):
            gamma_calibration([0.0, 1.0], [0.0])

    def test_jitter_matches_sinc_squared(self) -> None:
        assert gamma_jitter(0.0) == pytest.approx(1.0)
        assert gamma_jitter(0.5) == pytest.approx((np.sin(0.5) / 0.5) ** 2)
        assert gamma_jitter(1.0) < gamma_jitter(0.5) < 1.0
        with pytest.raises(ValueError):
            gamma_jitter(-0.1)

    def test_noise_is_snr_over_one_plus_snr(self) -> None:
        assert gamma_noise(1.0) == pytest.approx(0.5)
        assert gamma_noise(9.0) == pytest.approx(0.9)
        assert gamma_noise(1e9) == pytest.approx(1.0)
        with pytest.raises(ValueError):
            gamma_noise(-1.0)

    def test_decorrelation_detects_changed_scene(self) -> None:
        rng = np.random.default_rng(0)
        a = rng.normal(size=(32, 32))
        assert gamma_decorrelation(a, a) == pytest.approx(1.0)
        assert gamma_decorrelation(a, 3.0 * a + 5.0) == pytest.approx(1.0)
        assert gamma_decorrelation(a, rng.normal(size=(32, 32))) < 0.05

    def test_decorrelation_rejects_degenerate_input(self) -> None:
        with pytest.raises(ValueError):
            gamma_decorrelation(np.ones((4, 4)), np.ones((4, 4)))
        with pytest.raises(ValueError):
            gamma_decorrelation(np.zeros((4, 4)), np.zeros((4, 5)))


class TestStepDft:
    def test_dc_and_first_harmonic_amplitudes(self) -> None:
        steps = 8
        n = np.arange(steps)
        dc, amp = 2.0, 0.5
        frames = dc + amp * np.cos(2.0 * np.pi * n / steps)
        frames = np.repeat(frames[:, None], 3, axis=1)
        spec = step_dft(frames)
        assert abs(spec[0, 0]) == pytest.approx(steps * dc)
        assert abs(spec[1, 0]) == pytest.approx(steps * amp / 2.0)
        assert abs(spec[2, 0]) == pytest.approx(0.0, abs=1e-9)

    def test_rejects_single_step(self) -> None:
        with pytest.raises(ValueError):
            step_dft(np.zeros((1, 4, 4)))


class TestDiagnose:
    def test_perfect_conditions_hit_the_ceiling_exactly(self) -> None:
        report = diagnose(100)
        assert isinstance(report, FidelityReport)
        assert report.combined == pytest.approx(1.0)
        assert report.expected_enhancement == pytest.approx(report.ceiling)
        assert report.shortfall == pytest.approx(0.0)
        assert report.warnings == ()

    def test_each_defect_depresses_only_its_own_factor(self) -> None:
        rng = np.random.default_rng(1)
        base = rng.normal(size=(16, 16))
        cases = {
            "gamma_mode_coupling": dict(mode_fraction=0.5),
            "gamma_amplitude_only": dict(transmissions=[1.0, 0.0, 1.0, 0.0]),
            "gamma_unmodulated": dict(f0=1.0 + 0j, f1=0.19 + 0j),
            "gamma_jitter": dict(phase_error_rad=1.0),
            "gamma_noise": dict(snr=0.05),
            "gamma_decorrelation": dict(before=base, after=rng.normal(size=(16, 16))),
        }
        for name, kwargs in cases.items():
            report = diagnose(50, **kwargs)
            depressed = [k for k, v in report.factors.items() if v < DEFAULT_THRESHOLD]
            assert depressed == [name], (name, report.factors)
            assert any(name in w for w in report.warnings)
            assert report.expected_enhancement < report.ceiling

    def test_injected_phase_response_error_is_localised(self) -> None:
        n = 256
        ideal = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        measured = np.arange(n, dtype=np.float64) / 128.0 * np.pi * 1.2
        report = diagnose(100, measured_phase_rad=measured, ideal_phase_rad=ideal, snr=1e6)
        assert report.factors["gamma_calibration"] < DEFAULT_THRESHOLD
        assert report.factors["gamma_noise"] == pytest.approx(1.0)
        assert "LUT" in " ".join(report.warnings)

    def test_combined_equals_product_of_factors(self) -> None:
        report = diagnose(
            20,
            mode_fraction=0.9,
            transmissions=[1.0, 0.8, 1.0, 0.9],
            phase_error_rad=0.2,
            snr=20.0,
        )
        assert report.combined == pytest.approx(float(np.prod(list(report.factors.values()))))
        assert report.expected_enhancement == pytest.approx(
            1.0 + (report.ceiling - 1.0) * report.combined
        )

    def test_half_supplied_pairs_are_rejected(self) -> None:
        with pytest.raises(ValueError):
            diagnose(10, f0=1.0 + 0j)
        with pytest.raises(ValueError):
            diagnose(10, before=np.ones((2, 2)))


def test_every_mitigation_key_matches_a_reported_factor() -> None:
    report = diagnose(10, snr=1.0)
    assert set(MITIGATIONS) >= set(report.factors)
    assert set(MITIGATIONS) == {
        "gamma_mode_coupling",
        "gamma_amplitude_only",
        "gamma_unmodulated",
        "gamma_calibration",
        "gamma_jitter",
        "gamma_noise",
        "gamma_decorrelation",
    }


def test_leaf_module_imports_no_ao_shaping_package() -> None:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    offenders = [name for name in imported if name.split(".")[0] == "ao_shaping"]
    assert offenders == [], f"utils/slm leaf must not import ao_shaping: {offenders}"
