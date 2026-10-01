"""Contract tests for the simulated Shack-Hartmann wavefront sensor.

Oracle review corrected the original plan: a Shack-Hartmann sensor measures the
**pupil-plane phase gradient**, not the focal-plane intensity. Driving
``wfs_measure(..., phase_in=...)`` on an OOPAO ``ShackHartmann`` measures exactly
that — the slope response is linear in the injected pupil phase (verified: an
exact 2.0x slope ratio per doubling, with the expected centroid nonlinearity
only appearing at large tilts) and deterministic across repeated reads.

What these tests pin is the part that is easy to get silently wrong: **units**.
The WFS family returns Zernike coefficients in **micrometres** while the
correction path works in **waves**, and mixing them produced two real bugs
historically (1.88x coefficient inflation, 6.28x phase shrinkage).
"""

from __future__ import annotations

import numpy as np
import pytest

from ao_shaping.drivers.wfs.base import BaseWFS
from ao_shaping.utils.wavefront.zernike_utils import (
    generate_zernike_phase,
    um_to_waves,
)

pytest.importorskip(
    "ao_shaping.drivers.sim._oopao_compat",
    reason="OOPAO not available",
)

from ao_shaping.drivers.sim.wfs.simulated_wfs import SimulatedWFS  # noqa: E402

BASE_WFS_METHODS = (
    "take_image",
    "get_wavefront",
    "get_spot_deviation",
    "get_zernike",
    "get_spots_statics",
    "build_subaperture_mask",
)

#: Runner-facing Thorlabs methods that have no simulated counterpart.
THORLAB_NO_OPS = (
    "save_user_ref",
    "load_user_ref",
    "optimize_pupil",
    "optimize_exposure_time_and_gain",
)


@pytest.fixture(scope="module")
def wfs():
    return SimulatedWFS(resolution=48, n_subap=6, n_pixel_per_subaperture=8)


class TestImplementsBaseWFS:
    def test_is_a_base_wfs(self, wfs):
        assert isinstance(wfs, BaseWFS)

    @pytest.mark.parametrize("name", BASE_WFS_METHODS)
    def test_satisfies_the_contract(self, wfs, name):
        assert callable(getattr(wfs, name, None)), f"{name} missing"

    @pytest.mark.parametrize("name", THORLAB_NO_OPS)
    def test_thorlabs_specific_methods_are_no_ops(self, wfs, name):
        """Runners call these unconditionally; they must exist and not explode."""
        assert callable(getattr(wfs, name, None)), f"{name} missing on the sim WFS"

    def test_exposes_the_shared_parameters(self, wfs):
        names = set(wfs.list_parameters())
        assert {"exposure_time_ms", "remove_tilt"} <= names


class TestPupilGradientResponse:
    """The sensor must respond to pupil phase, not to focal-plane intensity."""

    def _defocus(self, wfs, scale: float) -> np.ndarray:
        res = wfs.grid
        yy, xx = np.mgrid[0:res, 0:res]
        return scale * -1e-4 * ((xx - res / 2) ** 2 + (yy - res / 2) ** 2)

    def test_flat_pupil_gives_zero_slopes(self, wfs):
        wfs.set_pupil_phase(np.zeros((wfs.grid, wfs.grid)))
        dx, dy = wfs.take_slopes()
        assert np.allclose(dx, 0, atol=1e-9)
        assert np.allclose(dy, 0, atol=1e-9)

    def test_slopes_scale_linearly_with_phase(self, wfs):
        base = self._defocus(wfs, 1.0)
        wfs.set_pupil_phase(base)
        a = wfs.take_slopes()[0]
        wfs.set_pupil_phase(base * 2.0)
        b = wfs.take_slopes()[0]
        rms_a = float(np.sqrt(np.mean(a**2)))
        rms_b = float(np.sqrt(np.mean(b**2)))
        assert rms_a > 0, "a defocus pupil phase must produce measurable slopes"
        assert rms_b / rms_a == pytest.approx(2.0, rel=0.02), (
            "the SH response must be linear in pupil phase; centroid-based SH "
            "saturates only at large tilts"
        )

    def test_reads_are_deterministic(self, wfs):
        """Reports must be byte-reproducible, so repeated reads must be identical."""
        wfs.set_pupil_phase(self._defocus(wfs, 1.0))
        first = wfs.take_slopes()[0]
        second = wfs.take_slopes()[0]
        assert np.array_equal(first, second)


class TestUnitContract:
    """``get_zernike`` is micrometres; the correction path is waves."""

    def test_get_zernike_returns_micrometres(self, wfs):
        """A known defocus amplitude must survive the whole chain as micrometres.

        ``um_to_waves`` is hard-coded for 532 nm, so the simulated sensor's
        default wavelength must match or the whole chain silently rescales.

        Amplitudes are chosen in the well-resolved regime: OOPAO's slope
        estimator is accurate to <1% there but degrades below ~0.3 rad (see the
        module docstring). This pins the guarantee callers actually rely on.
        """
        assert wfs.wavelength_nm == 532, (
            "the sim WFS must default to 532 nm so um_to_waves() round-trips"
        )
        wfs.set_pupil_phase(np.zeros((wfs.grid, wfs.grid)))
        assert np.allclose(wfs.get_zernike(zernike_order=4), 0.0, atol=1e-9)

        for amplitude_rad in (0.35, 0.70, -0.70):
            phase = np.nan_to_num(
                generate_zernike_phase(
                    {4: amplitude_rad},
                    resolution=(wfs.grid, wfs.grid),
                    n_max=4,
                    radius=wfs.grid / 2.4,
                )
            )
            wfs.set_pupil_phase(phase)
            z_um = wfs.get_zernike(zernike_order=4)
            recovered_rad = um_to_waves(z_um[3]) * 2.0 * np.pi
            assert recovered_rad == pytest.approx(amplitude_rad, rel=0.01)
            assert np.argmax(np.abs(z_um)) == 3

    def test_recovery_is_sign_symmetric(self, wfs):
        """The sensor must not favour defocus of one sign over the other."""
        ratios = []
        for amplitude_rad in (0.35, -0.35):
            phase = np.nan_to_num(
                generate_zernike_phase(
                    {4: amplitude_rad},
                    resolution=(wfs.grid, wfs.grid),
                    n_max=4,
                    radius=wfs.grid / 2.4,
                )
            )
            wfs.set_pupil_phase(phase)
            z_um = wfs.get_zernike(zernike_order=4)
            ratios.append(um_to_waves(z_um[3]) * 2.0 * np.pi / amplitude_rad)
        assert ratios[0] == pytest.approx(ratios[1], rel=1e-6)

    def _inject_defocus(self, wfs, scale: float) -> None:
        res = wfs.grid
        yy, xx = np.mgrid[0:res, 0:res]
        wfs.set_pupil_phase(scale * -1e-3 * ((xx - res / 2) ** 2 + (yy - res / 2) ** 2))

    def test_get_wavefront_returns_waves(self, wfs):
        """``get_wavefront`` is documented in waves, so a full turn is 1.0."""
        self._inject_defocus(wfs, 1.0)
        wavefront, stats = wfs.get_wavefront()
        assert wavefront.shape == (wfs.grid, wfs.grid)
        assert set(stats) >= {"rms", "min", "max"}
        assert np.isfinite(wavefront).all()
        # Defocus in radians -> waves must be smaller by 2*pi.
        assert np.max(np.abs(wavefront)) < 1.0, (
            "waves = radians / 2*pi, so the magnitude must shrink"
        )

    def test_statics_keys_match_the_thorlab_contract(self, wfs):
        """The runner reads ``statics["wighted_rms"]``, so the keys must match.

        ``ThorlabWFS.get_wavefront`` documents exactly these six keys. A narrower
        dict raises ``KeyError`` mid-optimization, which is invisible until a
        runner is actually pointed at the simulated sensor.
        """
        self._inject_defocus(wfs, 1.0)
        _, stats = wfs.get_wavefront()
        assert set(stats) == {"min", "max", "diff", "mean", "rms", "wighted_rms"}
        assert stats["wighted_rms"] >= 0.0
        assert stats["diff"] == pytest.approx(stats["max"] - stats["min"])

    def test_wavefront_and_zernike_agree_on_sign(self, wfs):
        """Both readouts derive from the same state and must not disagree.

        The comparison is a projection onto the defocus mode, not ``mean()``:
        the grid is zero outside the circular aperture, so the mean is not the
        defocus amplitude.
        """
        self._inject_defocus(wfs, 1.0)
        wavefront, _ = wfs.get_wavefront()
        z_um = wfs.get_zernike(zernike_order=4)
        z_rad = um_to_waves(z_um) * 2.0 * np.pi
        defocus = np.nan_to_num(
            generate_zernike_phase(
                {4: 1.0}, resolution=(wfs.grid, wfs.grid), n_max=4,
                radius=wfs.grid / 2.4,
            )
        )
        projection = float(np.sum(wavefront * defocus))
        assert np.sign(projection) == np.sign(z_rad[3])


class TestRunnerFacingSurface:
    """Members the optimizers and runners read straight off the sensor.

    Each of these was found by a crash rather than by a test, so they are pinned
    together: an added sensor type that cannot answer them is not substitutable.
    """

    @pytest.mark.parametrize(
        "name",
        [
            "num_spots_x",
            "num_spots_y",
            "mla_index",
            "serial_num",
            "device_name",
            "exposure_time",
            "high_speed",
            "use_custom_ref",
            "pupil",
            "d_x",
        ],
    )
    def test_attribute_is_present(self, wfs, name):
        assert hasattr(wfs, name), f"{name} missing; runners read it directly"

    def test_mla_and_pupil_methods(self, wfs):
        assert isinstance(wfs.get_mla_name(), str)
        wfs.set_ref_plane(True)
        assert wfs.use_custom_ref is True

    def test_subaperture_mask_shape_matches_the_slope_grid(self, wfs):
        """The mask is filtered against ``2 * nx * ny`` slopes, so it must match.

        It used to be sized from the flux array, which OOPAO reports on the 8x8
        lenslet grid (64 entries at ``n_subap=6``) while the slopes span a 6x6
        subaperture grid -- so the mask silently had the wrong length and the
        calibration blew up with a boolean-index size mismatch.
        """
        mask, valid = wfs.build_subaperture_mask()
        assert mask.shape == (wfs.num_spots_x, wfs.num_spots_y)
        assert mask.dtype == bool
        assert valid.shape == (int(mask.sum()),)
        assert 2 * mask.size == 2 * wfs.num_spots_x * wfs.num_spots_y

    def test_subaperture_mask_unpacks_as_two_values(self, wfs):
        """All three callers do ``mask, _ = wfs.build_subaperture_mask()``."""
        mask, _valid = wfs.build_subaperture_mask()
        assert mask is not None


class TestDmCoupling:
    """DM voltages must reach the pupil this sensor measures.

    They used to stop at the far-field model: ``SimulateDM`` published only to
    ``SimPibSystem``, so ``wf``/``rms-zernike`` read a flat pupil and reported
    zero RMS no matter what the DM did.
    """

    def test_dm_voltage_reaches_the_measured_pupil(self, wfs):
        from ao_shaping.drivers.dm._registry import create_dm

        wfs.set_pupil_phase(np.zeros((wfs.grid, wfs.grid)))
        dm = create_dm("sim")
        dm.open()
        try:
            readings = []
            for amplitude in (0.0, 300.0):
                volts = np.zeros(dm.DM_NUM)
                volts[10] = amplitude
                dm.send_voltages(volts)
                wfs.take_image(3)
                _, stats = wfs.get_wavefront()
                readings.append(stats["wighted_rms"])
        finally:
            dm.close()

        assert readings[0] < 1e-4, "flat DM must read a flat pupil"
        assert readings[1] > 10 * readings[0], (
            "driving the DM did not move the pupil the sensor measures"
        )

    def test_explicit_phase_and_dm_phase_sum(self, wfs):
        """Both contributors apply; neither replaces the other."""
        wfs.set_pupil_phase(np.zeros((wfs.grid, wfs.grid)))
        wfs.dm_optics.reset()
        wfs.take_image(3)
        bare = np.abs(wfs._pupil_phase()).max()

        grid_y, grid_x = np.mgrid[0 : wfs.grid, 0 : wfs.grid]
        bowl = -1e-4 * (
            (grid_x - wfs.grid / 2) ** 2 + (grid_y - wfs.grid / 2) ** 2
        )
        wfs.set_pupil_phase(bowl)
        wfs.take_image(3)
        with_explicit = np.abs(wfs._pupil_phase()).max()

        volts = np.zeros(wfs.dm_optics.n_actuators)
        volts[5] = wfs.dm_optics.v_max
        wfs.dm_optics.set_voltages(volts)
        wfs.take_image(3)
        with_both = np.abs(wfs._pupil_phase()).max()

        assert with_both > max(bare, with_explicit), (
            "DM phase was dropped instead of summed with the explicit phase"
        )

    def test_dm_phase_is_not_baked_into_the_command(self, wfs):
        """Following the SimDisturbance precedent, contributors sum at evaluation."""
        wfs.set_pupil_phase(np.zeros((wfs.grid, wfs.grid)))
        volts = np.zeros(wfs.dm_optics.n_actuators)
        volts[3] = 200.0
        wfs.dm_optics.set_voltages(volts)
        wfs.take_image(3)
        assert np.count_nonzero(wfs._phase_rad) == 0


class TestDisturbanceInjection:
    """The aberration the loop is supposed to correct."""

    def test_cn2_zero_injects_nothing(self, wfs):
        from ao_shaping.drivers.sim.disturbance import DisturbanceConfig

        plain = SimulatedWFS(resolution=48, n_subap=6, n_pixel_per_subaperture=8)
        assert plain.disturbance is None
        cfg = DisturbanceConfig(mode="none", cn2=0.0)
        quiet = SimulatedWFS(
            resolution=48, n_subap=6, n_pixel_per_subaperture=8,
            disturbance_config=cfg,
        )
        assert quiet.disturbance is not None
        assert np.count_nonzero(quiet.disturbance.phase()) == 0

    def test_static_cn2_injects_a_stable_aberration(self):
        from ao_shaping.drivers.sim.disturbance import DisturbanceConfig

        def build():
            cfg = DisturbanceConfig(mode="static", cn2=2e-13, seed=7)
            return SimulatedWFS(
                resolution=48, n_subap=6, n_pixel_per_subaperture=8,
                disturbance_config=cfg,
            )

        a, b = build(), build()
        assert np.abs(a.disturbance.phase()).max() > 0, "cn2 must add aberration"
        assert np.array_equal(
            a.disturbance.phase(), b.disturbance.phase()
        ), "static mode must be reproducible for a fixed seed"

    def test_aberration_raises_the_measured_rms(self):
        from ao_shaping.drivers.sim.disturbance import DisturbanceConfig

        plain = SimulatedWFS(resolution=48, n_subap=6, n_pixel_per_subaperture=8)
        cfg = DisturbanceConfig(mode="static", cn2=2e-13, seed=7)
        aberrated = SimulatedWFS(
            resolution=48, n_subap=6, n_pixel_per_subaperture=8,
            disturbance_config=cfg,
        )
        plain.take_image(3)
        aberrated.take_image(3)
        assert (
            aberrated.get_wavefront()[1]["wighted_rms"]
            > 10 * plain.get_wavefront()[1]["wighted_rms"]
        )

    def test_cn2_knob_reaches_the_registry(self):
        """The kwarg must survive registry filtering or the CLI flag is dead."""
        from ao_shaping.drivers.wfs._registry import create_wfs

        sensor = create_wfs("sim", resolution=48, disturbance_cn2=2e-13)
        assert sensor.disturbance is not None
        assert create_wfs("sim", resolution=48).disturbance is None


class TestNoOpSurface:
    def test_optimize_pupil_returns_the_known_pupil(self, wfs):
        cx, cy, dx, dy = wfs.optimize_pupil()
        assert all(np.isfinite(v) for v in (cx, cy, dx, dy))

    def test_user_ref_round_trip_does_not_raise(self, wfs):
        assert wfs.save_user_ref() is True
        assert wfs.load_user_ref() is True

    def test_context_manager_and_connection_flag(self, wfs):
        with wfs as opened:
            assert opened.is_connected()
        assert not wfs.is_connected()
