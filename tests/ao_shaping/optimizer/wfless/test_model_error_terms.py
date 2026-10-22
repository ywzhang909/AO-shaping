"""正向模型的误差项: LUT 非线性 (gamma) + 相机离焦 + 光路倾斜。

被测对象:

* ``ao_shaping/utils/wavefront/model_error_terms.py`` —— 误差项容器与装载器;
* ``ao_shaping/algorithm/signal_processing/zernike_coefficient_optimizer.py``
  —— 误差项在正向模型里的**作用位置**与数值;
* ``ao_shaping/optimizer/wfless/model_in_loop_shaping.py`` —— Step B 与 Step A
  共用同一个前向, 以及 gamma 的拟合。

物理口径 (**不要重复计入**, 这里是唯一的说明处):

* **面板离焦 (瞳面二次相位) ≡ 相机离焦 (焦面角谱)** —— 孪生实测二者逐位
  等价 (焦面复场的频谱就是瞳面)。所以 MIL 模型里的 **Noll 4 系数已经能表达
  相机离焦**, 不要既加「面板 Zernike 离焦」又加「相机离焦」。
* **光路倾斜**在硬件上被**几何标定**吸收 (ROI 中心按 0 级实测); 孪生保留该级
  只是为了让仿真路径完整。
* 真正非冗余、MIL 原本**没有**的误差级是 **gamma (LUT 非线性)**: 孪生实测
  gamma = 2.07, 即「命令相位 = 实际相位」这个前提错得很厉害。

⇒ 误差项的作用位置: gamma 在**瞳面相位之前** (命令 → 实际), 倾斜与离焦在
**焦面复场**上、``|·|²`` 之前。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from ao_shaping.algorithm.signal_processing.zernike_coefficient_optimizer import (
    ZernikeCoefficientOptimizer,
    apply_lut_nonlinearity,
)
from ao_shaping.optimizer.wfless.model_in_loop_shaping import (
    StepBConfig,
    fit_lut_gamma,
    shape_phase_with_frozen_aberration,
)
from ao_shaping.utils.slm.phase_display import phase_to_slm_grayscale
from ao_shaping.utils.wavefront.model_error_terms import (
    ModelErrorTerms,
    load_error_terms,
    save_error_terms,
)

REGION = 48
WAVELENGTH_M = 1064e-9
FOCAL_LENGTH_M = 0.125
SLM_PIXEL_UM = 8.0
#: The twin's gray depth. 16 bit is the most a uint16 gray map can carry, so
#: it is the *smallest* quantisation the twin can express.
GRAY_LEVELS = 65535


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _twin(
    *,
    gamma: float = 1.0,
    defocus_m: float = 0.0,
    tilt_x_rad: float = 0.0,
    tilt_y_rad: float = 0.0,
    gray_levels: int = GRAY_LEVELS,
):
    """A bench twin on a grid the model can reproduce exactly."""
    from ao_shaping.drivers.sim.bench_twin import (
        BenchTwin,
        BenchTwinConfig,
        TwinErrorModel,
    )

    errors = TwinErrorModel(
        lut_gamma=gamma,
        defocus_m=defocus_m,
        tilt_x_rad=tilt_x_rad,
        tilt_y_rad=tilt_y_rad,
        gray_levels=gray_levels,
    )
    config = BenchTwinConfig(
        region=REGION,
        region_span_px=REGION,
        far_field_padding=1,
        beam_w0_px=12.0,
        crop_px=REGION,
        wavelength_m=WAVELENGTH_M,
        focal_length_m=FOCAL_LENGTH_M,
        slm_pixel_um=SLM_PIXEL_UM,
        errors=errors,
    )
    return BenchTwin(config), config


def _aperture() -> np.ndarray:
    """The model's pupil aperture (the disc the phase is masked to)."""
    optimizer = ZernikeCoefficientOptimizer(
        n_orders=1, region=REGION, dtype="float64", device="cpu"
    )
    return optimizer.generate_basis()[0] > 0.0


def _command_phase(seed: int = 0, amplitude: float = 2.0, sigma: float = 6.0) -> np.ndarray:
    """A smooth command phase, supported **inside** the pupil aperture.

    The support matters: the model masks the phase outside the aperture (the
    twin does not), so a command that reaches beyond it would compare two
    different pupils. The smoothness keeps the far field a clean focus, which
    is what makes the grey quantisation a negligible perturbation instead of a
    speckle scrambler.
    """
    from scipy.ndimage import gaussian_filter

    aperture = _aperture()
    rng = np.random.default_rng(seed)
    field = gaussian_filter(rng.normal(0.0, 1.0, (REGION, REGION)), sigma=sigma, mode="reflect")
    field -= field.mean()
    field = field / max(float(field.std()), 1e-12) * amplitude
    phase = np.zeros((REGION, REGION), dtype=np.float64)
    phase[aperture] = np.mod(field[aperture], 2.0 * math.pi)
    return phase


def _terms(
    *,
    gamma: float = 1.0,
    defocus_m: float = 0.0,
    tilt_x_rad: float = 0.0,
    tilt_y_rad: float = 0.0,
    pitch_m: float = 0.0,
) -> ModelErrorTerms:
    """Error terms carrying this bench's optical prescription."""
    return ModelErrorTerms(
        lut_gamma=gamma,
        camera_defocus_m=defocus_m,
        tilt_x_rad=tilt_x_rad,
        tilt_y_rad=tilt_y_rad,
        wavelength_m=WAVELENGTH_M,
        focal_length_m=FOCAL_LENGTH_M,
        far_field_pitch_m=pitch_m,
    )


def _model(terms: ModelErrorTerms, amplitude: np.ndarray) -> ZernikeCoefficientOptimizer:
    """A Step A optimizer on the twin's grid, carrying the twin's illumination."""
    optimizer = ZernikeCoefficientOptimizer(
        n_orders=1,
        region=REGION,
        dtype="float64",
        device="cpu",
        error_terms=terms,
    )
    optimizer.set_source_amplitude(amplitude)
    return optimizer


def _grey_equivalent_command(phase: np.ndarray, gray_levels: int = GRAY_LEVELS) -> np.ndarray:
    """The command that makes the panel display *exactly* the same phase.

    The twin's LUT chain is ``command -> gray -> 2π·(gray/levels)^gamma``; the
    model's is ``command -> 2π·(wrap(command)/2π)^gamma`` (the grey
    quantisation is deliberately omitted from the differentiable path). Feeding
    the model the *grey-equivalent* command — the phase the panel actually
    displays — removes the quantisation from the comparison without touching
    either implementation.
    """
    gray = phase_to_slm_grayscale(phase, max_grayscale=gray_levels)
    return 2.0 * math.pi * gray.astype(np.float64) / float(gray_levels)


def _normalised(image: np.ndarray) -> np.ndarray:
    """Peak-normalised far field."""
    peak = float(np.max(image))
    return image / peak if peak > 0.0 else image


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation of two flat arrays."""
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    x = x - x.mean()
    y = y - y.mean()
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(x @ y / denominator) if denominator > 0.0 else float("nan")


def _second_moment(image: np.ndarray) -> float:
    """The normalised second central moment of a far-field frame."""
    arr = np.asarray(image, dtype=np.float64)
    peak = float(arr.max())
    if peak <= 0.0:
        return float("nan")
    arr = arr / peak
    yy, xx = np.mgrid[0 : arr.shape[0], 0 : arr.shape[1]]
    cx = float((arr * xx).sum() / arr.sum())
    cy = float((arr * yy).sum() / arr.sum())
    return float((arr * ((xx - cx) ** 2 + (yy - cy) ** 2)).sum() / arr.sum())


def _centroid(image: np.ndarray) -> tuple[float, float]:
    """The intensity centroid of a far-field frame."""
    arr = np.asarray(image, dtype=np.float64)
    yy, xx = np.mgrid[0 : arr.shape[0], 0 : arr.shape[1]]
    total = float(arr.sum())
    if total <= 0.0:
        return float("nan"), float("nan")
    return float((arr * xx).sum() / total), float((arr * yy).sum() / total)


# ---------------------------------------------------------------------------
# The default path must not move
# ---------------------------------------------------------------------------


class TestDefaultPathIsUnchanged:
    """With no error terms the forward model must be exactly what it was."""

    def test_the_default_container_is_the_identity(self) -> None:
        terms = ModelErrorTerms()
        assert terms.is_identity
        assert not terms.needs_focal_geometry

    def test_the_forward_model_is_bitwise_the_ideal_formula(self) -> None:
        """Zero error terms -> the canonical FFT formula, bit for bit.

        The reference is computed independently here (the pupil field, the
        centred FFT and the intensity), so the test pins the *formula*, not
        just "the code did not change".
        """
        import torch

        optimizer = ZernikeCoefficientOptimizer(
            n_orders=4, region=32, dtype="float64", device="cpu"
        )
        rng = np.random.default_rng(7)
        phase = rng.normal(0.0, 2.0, (32, 32))
        coefficients = np.zeros(optimizer.n_coefficients)
        coefficients[3] = 0.25

        produced = optimizer.forward_intensity(coefficients, phase)

        basis = optimizer.generate_basis()
        aperture = (basis[0] > 0.0).astype(np.float64)
        amplitude = ZernikeCoefficientOptimizer.native_amplitude(32)
        patch = (phase + np.tensordot(coefficients, basis, axes=(0, 0))) * aperture
        field = amplitude * np.exp(1j * patch)
        reference = np.abs(
            np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field), norm="ortho"))
        ) ** 2
        # The numpy FFT and the torch FFT round in the last bits, so the two
        # chains agree to ~1e-13 rather than bit for bit. The *torch* chain is
        # the model's own, and that one is compared bit for bit below.
        np.testing.assert_allclose(produced, reference, rtol=1e-12, atol=0.0)

        # ... and the torch chain agrees with the model's own output bit for
        # bit: this is the formula, spelled out independently.
        torch_patch = torch.from_numpy(patch)
        torch_field = torch.from_numpy(amplitude) * torch.exp(1j * torch_patch)
        torch_reference = (
            torch.fft.fftshift(
                torch.fft.fft2(torch.fft.ifftshift(torch_field), norm="ortho")
            )
            .real.pow(2)
            + torch.fft.fftshift(
                torch.fft.fft2(torch.fft.ifftshift(torch_field), norm="ortho")
            )
            .imag.pow(2)
        ).numpy()
        assert np.array_equal(produced, torch_reference)

    def test_step_b_drives_the_optimizer_forward(self) -> None:
        """Step B must call the optimizer's own forward, not a copy of it.

        That is what makes "the calibration and the synthesis share one
        forward" structural rather than a convention: if Step B ever grew a
        second implementation, this test would fail.
        """
        optimizer = ZernikeCoefficientOptimizer(
            n_orders=4, region=32, dtype="float64", device="cpu"
        )
        calls: list[int] = []
        original = optimizer.far_field_intensity_tensor

        def spy(coefficients, phase_slm, amplitude):
            calls.append(1)
            return original(coefficients, phase_slm, amplitude)

        optimizer.far_field_intensity_tensor = spy  # type: ignore[method-assign]
        target = np.zeros((32, 32), dtype=np.float64)
        target[12:20, 12:20] = 1.0
        shape_phase_with_frozen_aberration(
            optimizer,
            np.zeros(optimizer.n_coefficients),
            target,
            np.zeros((32, 32), dtype=np.float64),
            StepBConfig(iterations=3, lr=0.05, target_side=8),
            dtype="float64",
            device="cpu",
            seed=0,
        )
        assert len(calls) == 3

    def test_the_tensor_forward_matches_the_numpy_one_bitwise(self) -> None:
        import torch

        optimizer = ZernikeCoefficientOptimizer(
            n_orders=4, region=32, dtype="float64", device="cpu"
        )
        rng = np.random.default_rng(11)
        phase = rng.normal(0.0, 2.0, (32, 32))
        coefficients = np.zeros(optimizer.n_coefficients)
        coefficients[3] = 0.25
        produced = optimizer.forward_intensity(coefficients, phase)
        tensor = optimizer.far_field_intensity_tensor(
            torch.from_numpy(coefficients),
            torch.from_numpy(phase),
            torch.from_numpy(ZernikeCoefficientOptimizer.native_amplitude(32)),
        )
        assert np.array_equal(produced, tensor.numpy())


# ---------------------------------------------------------------------------
# The model must equal the twin
# ---------------------------------------------------------------------------


class TestModelEqualsTwin:
    """The model's error terms must reproduce the twin's forward chain."""

    GAMMA = 1.6
    DEFOCUS_M = 2.0e-4
    TILT_X = 3.0e-4
    TILT_Y = -1.0e-4

    def _compare(self, *, grey_equivalent: bool) -> float:
        twin, _ = _twin(
            gamma=self.GAMMA,
            defocus_m=self.DEFOCUS_M,
            tilt_x_rad=self.TILT_X,
            tilt_y_rad=self.TILT_Y,
        )
        command = _command_phase(3)
        measured = twin.far_field(command)
        model = _model(
            _terms(
                gamma=self.GAMMA,
                defocus_m=self.DEFOCUS_M,
                tilt_x_rad=self.TILT_X,
                tilt_y_rad=self.TILT_Y,
                pitch_m=twin.p_fft,
            ),
            twin.pupil_amplitude(),
        )
        phase = (
            _grey_equivalent_command(command) if grey_equivalent else command
        )
        predicted = model.forward_intensity(
            np.zeros(model.n_coefficients), phase
        )
        return float(
            np.max(np.abs(_normalised(measured) - _normalised(predicted)))
        )

    def test_the_model_reproduces_the_twin_with_error_terms(self) -> None:
        """gamma + camera defocus + tilt: the model must equal the twin.

        The comparison uses the *grey-equivalent* command, i.e. the phase the
        panel actually displays. The twin quantises the command onto its gray
        map (an unavoidable, non-differentiable step) while the model omits
        that step on purpose; feeding the model the grey-equivalent command
        removes the quantisation from the comparison without touching either
        implementation, so what is left is exactly the chain under test.
        """
        deviation = self._compare(grey_equivalent=True)
        assert deviation < 1e-6, (
            f"the model and the twin disagree by {deviation:.3g} "
            "(tolerance 1e-6 for float64)"
        )

    def test_the_raw_command_residual_is_the_grey_quantisation(self) -> None:
        """With the raw command the only difference left is the quantisation.

        The model deliberately omits the grey quantisation (it is a staircase,
        so it has no gradient), so the residual has to be *the quantisation's
        size* and nothing else. It also has to shrink when the twin's gray map
        gets finer -- that is what makes the attribution to the quantisation
        falsifiable rather than a story.
        """
        coarse = self._compare(grey_equivalent=False)
        fine = self._compare(grey_equivalent=False)
        assert coarse == fine  # the comparison is deterministic
        assert coarse < 1e-4, (
            f"the raw-command residual is {coarse:.3g}; it must be the "
            "quantisation, not a modelling difference"
        )

        # A finer gray map shrinks the residual: the quantisation is its cause.
        residuals = {}
        for levels in (1023, GRAY_LEVELS):
            twin, _ = _twin(
                gamma=self.GAMMA,
                defocus_m=self.DEFOCUS_M,
                tilt_x_rad=self.TILT_X,
                tilt_y_rad=self.TILT_Y,
                gray_levels=levels,
            )
            command = _command_phase(3)
            measured = twin.far_field(command)
            model = _model(
                _terms(
                    gamma=self.GAMMA,
                    defocus_m=self.DEFOCUS_M,
                    tilt_x_rad=self.TILT_X,
                    tilt_y_rad=self.TILT_Y,
                    pitch_m=twin.p_fft,
                ),
                twin.pupil_amplitude(),
            )
            predicted = model.forward_intensity(
                np.zeros(model.n_coefficients), command
            )
            residuals[levels] = float(
                np.max(np.abs(_normalised(measured) - _normalised(predicted)))
            )
        assert residuals[GRAY_LEVELS] < residuals[1023], (
            "a finer gray map must shrink the residual, otherwise it is not "
            f"the quantisation: {residuals}"
        )


# ---------------------------------------------------------------------------
# The direction of each error term
# ---------------------------------------------------------------------------


class TestErrorTermDirections:
    """Each error term must push the far field the way the physics says."""

    def test_gamma_follows_the_power_law(self) -> None:
        """``actual = phi_max · (wrap(cmd) / 2π) ** gamma``."""
        import torch

        command = torch.tensor([[0.3, 1.7, 3.9, 6.0]], dtype=torch.float64)
        for gamma in (1.0, 1.6, 2.07):
            actual = apply_lut_nonlinearity(command, gamma)
            wrapped = torch.remainder(command, 2.0 * math.pi)
            expected = 2.0 * math.pi * (wrapped / (2.0 * math.pi)) ** gamma
            assert torch.allclose(actual, expected, rtol=0, atol=1e-12)
        # A deeper gamma compresses the phase: the panel displays less phase
        # than it was told to.
        shallow = apply_lut_nonlinearity(command, 1.0)
        deep = apply_lut_nonlinearity(command, 2.07)
        assert float(deep[0, 1]) < float(shallow[0, 1])
        assert float(deep[0, 1]) == pytest.approx(
            2.0 * math.pi * (1.7 / (2.0 * math.pi)) ** 2.07, rel=1e-12
        )

    def test_gamma_one_is_the_identity_of_the_raw_phase(self) -> None:
        """``gamma == 1`` must not even wrap: the old output must not move."""
        import torch

        command = torch.tensor([[-1.0, 0.5, 7.0]], dtype=torch.float64)
        assert torch.equal(apply_lut_nonlinearity(command, 1.0), command)

    def test_defocus_widens_the_spot(self) -> None:
        """A defocused camera sees a wider spot than a focused one.

        The effect is *monotone* in the defocus, which is the falsifiable
        statement: a single positive sign would also be satisfied by a model
        that merely added noise. The magnitudes are small at this bench's
        geometry (the pupil is 24 model px across, so 50 mm of camera defocus
        is 0.35 rad of quadratic phase at the pupil edge), so the test
        compares the growth rather than pinning a number.
        """
        twin, _ = _twin(defocus_m=5.0e-2)
        amplitude = twin.pupil_amplitude()
        sharp_model = _model(_terms(pitch_m=twin.p_fft), amplitude)
        command = _command_phase(5)

        def moment(defocus_m: float) -> float:
            model = _model(
                _terms(defocus_m=defocus_m, pitch_m=twin.p_fft), amplitude
            )
            return _second_moment(
                model.forward_intensity(np.zeros(model.n_coefficients), command)
            )

        sharp = moment(0.0)
        near = moment(5.0e-3)
        far = moment(5.0e-2)
        assert sharp < near < far, (
            f"the spot must widen monotonically with the defocus: "
            f"{sharp:.4f} -> {near:.4f} -> {far:.4f}"
        )
        # ... and the same holds on the twin, which is what makes the
        # comparison meaningful.
        sharp_twin, _ = _twin()
        assert _second_moment(twin.far_field(command)) > _second_moment(
            sharp_twin.far_field(command)
        )

    def test_tilt_shifts_the_spot_by_f_times_tilt(self) -> None:
        """The focal pattern must move by ``f · tilt`` — like the twin's."""
        tilt_x = 3.0e-4
        tilted_twin, _ = _twin(tilt_x_rad=tilt_x)
        straight_twin, _ = _twin()
        amplitude = tilted_twin.pupil_amplitude()
        straight_model = _model(_terms(pitch_m=tilted_twin.p_fft), amplitude)
        tilted_model = _model(
            _terms(tilt_x_rad=tilt_x, pitch_m=tilted_twin.p_fft), amplitude
        )
        command = _command_phase(7)
        straight = straight_model.forward_intensity(
            np.zeros(straight_model.n_coefficients), command
        )
        tilted = tilted_model.forward_intensity(
            np.zeros(tilted_model.n_coefficients), command
        )

        expected_cells = FOCAL_LENGTH_M * tilt_x / tilted_twin.p_fft
        dx_model = _centroid(tilted)[0] - _centroid(straight)[0]
        dx_twin = _centroid(tilted_twin.far_field(command))[0] - _centroid(
            straight_twin.far_field(command)
        )[0]
        # The model must move the pattern exactly like the twin does ...
        assert dx_model == pytest.approx(dx_twin, rel=0.02, abs=0.05)
        # ... and by the physical amount f * tilt.
        assert dx_model == pytest.approx(expected_cells, rel=0.05, abs=0.05)


# ---------------------------------------------------------------------------
# The fidelity gain -- "the fine tuning really helps"
# ---------------------------------------------------------------------------


class TestFidelityGain:
    """A fitted gamma must predict the measurement better than gamma = 1."""

    GAMMA_TRUE = 1.6

    def _samples(self, count: int = 6):
        twin, _ = _twin(gamma=self.GAMMA_TRUE)
        amplitude = twin.pupil_amplitude()
        frames = []
        for index in range(count):
            command = _command_phase(100 + index)
            frames.append((twin.far_field(command), command))
        return twin, amplitude, frames

    def test_the_fitted_gamma_predicts_better_than_the_ideal_model(self) -> None:
        twin, amplitude, frames = self._samples()
        model = _model(_terms(gamma=1.0, pitch_m=twin.p_fft), amplitude)

        ideal = [
            _correlation(
                model.forward_intensity(np.zeros(model.n_coefficients), command),
                measured,
            )
            for measured, command in frames
        ]

        fitted, history = fit_lut_gamma(
            model,
            frames,
            gammas=np.linspace(1.0, 3.0, 21),
            source_amplitude=amplitude,
        )
        # The fit restores the optimizer's error terms, so the tuned
        # prediction has to be made with the fitted gamma installed.
        from dataclasses import replace

        model.set_error_terms(replace(model.error_terms, lut_gamma=fitted))
        tuned = [
            _correlation(
                model.forward_intensity(np.zeros(model.n_coefficients), command),
                measured,
            )
            for measured, command in frames
        ]

        assert fitted == pytest.approx(self.GAMMA_TRUE, abs=0.06), (
            f"the fitted gamma {fitted} should land on the truth "
            f"{self.GAMMA_TRUE} (history: {history})"
        )
        mean_ideal = float(np.mean(ideal))
        mean_tuned = float(np.mean(tuned))
        assert mean_tuned > mean_ideal, (
            f"the fitted gamma must predict better: {mean_tuned:.4f} vs "
            f"{mean_ideal:.4f}"
        )
        # The ideal model's prediction is *materially* wrong, not slightly.
        assert mean_ideal < 0.99, (
            f"gamma = 1 should be clearly wrong on a gamma = {self.GAMMA_TRUE} "
            f"bench, got {mean_ideal:.4f}"
        )
        assert mean_tuned > 0.999, (
            f"the fitted gamma should predict the bench well, got {mean_tuned:.4f}"
        )
        # The scan is monotone around the truth: the loss falls with the gamma
        # up to the truth and rises after it.
        losses = dict(history)
        assert losses[1.0] > losses[self.GAMMA_TRUE]
        assert losses[3.0] > losses[self.GAMMA_TRUE]

    def test_the_fit_restores_the_optimizer_error_terms(self) -> None:
        twin, amplitude, frames = self._samples(count=2)
        terms = _terms(gamma=1.0, pitch_m=twin.p_fft)
        model = _model(terms, amplitude)
        fit_lut_gamma(model, frames, gammas=[1.0, 1.5, 2.0], source_amplitude=amplitude)
        assert model.error_terms == terms


# ---------------------------------------------------------------------------
# The container and the loader
# ---------------------------------------------------------------------------


class TestContainerAndLoader:
    def test_the_container_round_trips_through_json(self, tmp_path: Path) -> None:
        terms = _terms(gamma=2.07, defocus_m=1e-4, pitch_m=3.4e-4)
        path = save_error_terms(terms, tmp_path / "model_error_terms.json")
        restored = load_error_terms(path)
        assert restored == terms

    def test_the_loader_reads_a_twin_fit_result(self, tmp_path: Path) -> None:
        """The twin's ``save_fit`` output must load without conversion."""
        twin, config = _twin(gamma=2.07, defocus_m=1.5e-4, tilt_x_rad=2e-4)
        payload = {
            "config": {
                "wavelength_m": config.wavelength_m,
                "focal_length_m": config.focal_length_m,
                "region": config.region,
                "region_span_px": config.region_span_px,
                "far_field_padding": config.far_field_padding,
                "slm_pixel_um": config.slm_pixel_um,
                "errors": {
                    "lut_gamma": 2.07,
                    "lut_phase_max_rad": 2.0 * math.pi,
                    "defocus_m": 1.5e-4,
                    "tilt_x_rad": 2e-4,
                    "tilt_y_rad": -3e-4,
                },
            },
            "loss": 0.1,
        }
        path = tmp_path / "fit.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        terms = load_error_terms(path)
        assert terms.lut_gamma == pytest.approx(2.07)
        assert terms.camera_defocus_m == pytest.approx(1.5e-4)
        assert terms.tilt_x_rad == pytest.approx(2e-4)
        assert terms.tilt_y_rad == pytest.approx(-3e-4)
        # The geometry comes from the twin's own configuration.
        assert terms.wavelength_m == pytest.approx(config.wavelength_m)
        assert terms.focal_length_m == pytest.approx(config.focal_length_m)
        assert terms.far_field_pitch_m == pytest.approx(twin.p_fft, rel=1e-12)

    def test_a_foreign_payload_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "foreign.json"
        path.write_text(json.dumps({"something": "else"}), encoding="utf-8")
        with pytest.raises(ValueError, match="does not contain error terms"):
            load_error_terms(path)

    def test_the_geometry_can_be_re_anchored_on_another_bench(self) -> None:
        terms = _terms(gamma=2.07, defocus_m=1e-4, pitch_m=3.4e-4)
        moved = terms.with_optics(
            wavelength_m=WAVELENGTH_M,
            focal_length_m=0.311,
            far_field_pitch_m=1.1e-4,
        )
        assert moved.lut_gamma == terms.lut_gamma
        assert moved.camera_defocus_m == terms.camera_defocus_m
        assert moved.focal_length_m == pytest.approx(0.311)
        assert moved.far_field_pitch_m == pytest.approx(1.1e-4)

    def test_a_defocus_without_a_wavelength_is_refused(self) -> None:
        terms = ModelErrorTerms(camera_defocus_m=1e-4)
        with pytest.raises(ValueError, match="needs a wavelength"):
            terms.validate_focal_geometry()

    def test_a_tilt_without_a_focal_length_is_refused(self) -> None:
        terms = ModelErrorTerms(tilt_x_rad=1e-4)
        with pytest.raises(ValueError, match="needs the focal length"):
            terms.validate_focal_geometry()

    def test_a_non_positive_gamma_is_refused(self) -> None:
        with pytest.raises(ValueError, match="lut_gamma"):
            ModelErrorTerms(lut_gamma=0.0)


# ---------------------------------------------------------------------------
# The hardware path: from a calibration file to the model
# ---------------------------------------------------------------------------


class TestHardwareWiring:
    """The runner must be able to load the terms and to write them back."""

    def _config(self, error_params: str | None):
        from ao_shaping.optimizer.wfless.slm_model_in_loop import (
            SlmModelInLoopConfig,
        )

        return SlmModelInLoopConfig(
            region=64,
            far_field_padding=4,
            panel_span_px=900.0,
            slm_pixel_um=8.0,
            wavelength_nm=1064,
            error_params=error_params,
        )

    def test_the_config_loads_the_terms_and_re_anchors_the_geometry(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "errors.json"
        path.write_text(json.dumps({"lut_gamma": 2.07}), encoding="utf-8")
        config = self._config(str(path))
        terms = config.error_terms
        assert terms is not None
        assert terms.lut_gamma == pytest.approx(2.07)
        # The geometry is THIS run's, not the calibration file's.
        assert terms.wavelength_m == pytest.approx(1064e-9)
        assert terms.focal_length_m > 0.0
        expected_pitch = (
            1064e-9
            * config.focal_length_m
            / (config.region * config.far_field_padding * (900.0 / config.region) * 8e-6)
        )
        assert terms.far_field_pitch_m == pytest.approx(expected_pitch, rel=1e-12)

    def test_a_missing_error_params_file_is_refused_before_any_device(self) -> None:
        with pytest.raises(ValueError, match="error_params does not exist"):
            self._config("C:/nowhere/errors.json")

    def test_the_optimizer_the_runner_builds_carries_the_terms(
        self, tmp_path: Path
    ) -> None:
        from ao_shaping.optimizer.wfless.slm_model_in_loop import _make_optimizer

        path = tmp_path / "errors.json"
        path.write_text(json.dumps({"lut_gamma": 2.07}), encoding="utf-8")
        config = self._config(str(path))
        optimizer = _make_optimizer(config, None)
        assert optimizer.error_terms.lut_gamma == pytest.approx(2.07)

    def test_a_run_without_error_params_keeps_the_ideal_model(self) -> None:
        config = self._config(None)
        assert config.error_terms is None
        from ao_shaping.optimizer.wfless.slm_model_in_loop import _make_optimizer

        optimizer = _make_optimizer(config, None)
        assert optimizer.error_terms.is_identity

    def test_the_terms_are_written_next_to_the_run(self, tmp_path: Path) -> None:
        """The next run must be able to reuse what this one used."""
        from ao_shaping.optimizer.wfless.slm_model_in_loop import (
            ModelInLoopResult,
            SlmModelInLoopConfig,
        )
        from ao_shaping.runners.runner_common import (
            SlmModelInLoopParams,
            save_model_in_loop_results,
        )
        from ao_shaping.utils.io.file import Recorder

        source = tmp_path / "errors.json"
        source.write_text(json.dumps({"lut_gamma": 2.07}), encoding="utf-8")
        config = SlmModelInLoopConfig(
            region=64,
            far_field_padding=4,
            panel_span_px=900.0,
            error_params=str(source),
        )
        result = ModelInLoopResult(
            recorder=Recorder(),
            status="completed",
            best_phase=np.zeros((64, 64)),
            best_frame=None,
            geometry=None,
            target_side_camera=60,
            target_side_model=30,
            flat_score=0.1,
            best_score=0.2,
            config=config,
        )
        saved = save_model_in_loop_results(
            result,
            SlmModelInLoopParams(),
            root_dir=tmp_path,
            debug=False,
        )
        written = saved.save_dir / "model_error_terms.json"
        assert written.is_file()
        restored = load_error_terms(written)
        assert restored.lut_gamma == pytest.approx(2.07)
        # ... and the run config carries it in a loadable form too.
        run_config = json.loads(
            (saved.save_dir / "run_config.json").read_text(encoding="utf-8")
        )
        assert run_config["error_terms"]["lut_gamma"] == pytest.approx(2.07)
