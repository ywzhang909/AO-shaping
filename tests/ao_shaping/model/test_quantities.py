"""Physical units and centred-grid conversion tests."""

import numpy as np
import pytest

from ao_shaping.model import (
    AmplitudeMap,
    ComplexField,
    CoordinateGrid2D,
    DmCommands,
    OPDMap,
    PSFImage,
    PhaseMap,
    WfsSlopes,
    ZernikeCoefficients,
)
from ao_shaping.model.spatial import resample_physical_grid
from ao_shaping.utils.slm.phase_display import phase_to_slm_grayscale


def test_phase_opd_roundtrip_preserves_units_and_metadata():
    phase = PhaseMap(np.array([[0.0, np.pi], [2 * np.pi, -np.pi]]), 8.0)
    opd = phase.to_opd_map(532e-9)
    assert isinstance(opd, OPDMap)
    np.testing.assert_allclose(opd.opd, phase.phase * 532e-9 / (2 * np.pi))
    np.testing.assert_allclose(opd.to_phase().phase, phase.phase)
    assert opd.metadata is phase.metadata


def test_wfs_slopes_flat_roundtrip_and_validation():
    values = np.arange(8, dtype=float)
    slopes = WfsSlopes.from_flat(values, n_subaperture=2, units="rad")
    np.testing.assert_array_equal(slopes.to_flat(), values)
    np.testing.assert_allclose(slopes.to_units("arcsec").to_units("rad").to_flat(), values)
    with pytest.raises(ValueError):
        WfsSlopes.from_flat(values, n_subaperture=3)


def test_zernike_unit_conversion_keeps_noll_indices():
    coeffs = ZernikeCoefficients(np.array([0.25, -0.5]), np.array([2, 5]), "waves")
    radians = coeffs.to_radians()
    np.testing.assert_allclose(radians.coefficients, [np.pi / 2, -np.pi])
    np.testing.assert_array_equal(radians.indices, [2, 5])
    np.testing.assert_allclose(radians.to_waves().coefficients, coeffs.coefficients)
    with pytest.raises(ValueError):
        ZernikeCoefficients([1, 2], [1, 1])


def test_physical_interpolation_tracks_coordinates_not_image_fraction():
    # 5 source samples at x = -2,-1,0,1,2 um; target samples at -1,0,1 um.
    source = np.tile(np.arange(-2.0, 3.0), (5, 1))
    target = resample_physical_grid(source, 1.0, (3, 3), 1.0)
    np.testing.assert_allclose(target, np.tile([-1.0, 0.0, 1.0], (3, 1)))
    coarse = resample_physical_grid(source, 1.0, (3, 3), 2.0)
    np.testing.assert_allclose(coarse, np.tile([-2.0, 0.0, 2.0], (3, 1)))
    grid = CoordinateGrid2D.centered((3, 3), 2.0)
    np.testing.assert_allclose(grid.x, np.tile([-2.0, 0.0, 2.0], (3, 1)))
    np.testing.assert_allclose(grid.y[:, 0], [-2.0, 0.0, 2.0])
    np.testing.assert_allclose(grid.to_metres().x, grid.x * 1e-6)


def test_maps_share_physical_resampling_convention():
    phase = PhaseMap(np.tile(np.arange(-2.0, 3.0), (5, 1)), 1.0)
    opd = phase.to_opd_map(1e-6)
    reduced_phase = phase.resample_to((3, 3), 1.0)
    reduced_opd = opd.resample_to((3, 3), 1.0)
    np.testing.assert_allclose(reduced_opd.to_phase().phase, reduced_phase.phase)
    amplitude = AmplitudeMap(np.abs(phase.phase), 1.0)
    np.testing.assert_allclose(amplitude.resample_to((3, 3), 1.0).amplitude, np.abs(reduced_phase.phase))
    field = ComplexField(np.ones((5, 5)) * (1 + 2j), 1.0)
    np.testing.assert_allclose(field.resample_to((3, 3), 1.0).field, 1 + 2j)
    psf = PSFImage(np.ones((5, 5)), 1.0, 1e-6)
    np.testing.assert_allclose(psf.resample_to((3, 3), 1.0).intensity, 1.0)


def test_intensity_amplitude_conversion_and_grid_match():
    psf = PSFImage(np.full((3, 3), 9.0), 4.0, 532e-9)
    amplitude = psf.to_amplitude()
    np.testing.assert_allclose(amplitude.amplitude, 3.0)
    np.testing.assert_allclose(amplitude.intensity, psf.intensity)
    phase = PhaseMap(np.zeros((3, 3)), 4.0)
    field = ComplexField.from_maps(phase, amplitude)
    np.testing.assert_allclose(field.intensity, psf.intensity)
    with pytest.raises(ValueError, match="pixel pitch"):
        ComplexField.from_maps(PhaseMap(np.zeros((3, 3)), 8.0), amplitude)


def test_phase_map_grayscale_matches_raw_path():
    raw = np.array([[0.0, np.pi], [2 * np.pi, -np.pi]])
    typed = PhaseMap(raw, 8.0)
    np.testing.assert_array_equal(phase_to_slm_grayscale(typed), phase_to_slm_grayscale(raw))


def test_sim_micro_dm_accepts_typed_and_raw_commands():
    from ao_shaping.drivers.sim.dm.simulated_micro_dm import SimMicroDM

    dm = SimMicroDM(safety_mode=False)
    typed = DmCommands(np.full(dm.DM_Num, 130.0), dm.V_Min, dm.V_Max, dm.DM_Num)
    result = dm.send_voltages(typed)
    assert isinstance(result, DmCommands)
    np.testing.assert_allclose(result.voltages, dm.V_Max)
    raw_result = dm.send_voltages(np.zeros(dm.DM_Num))
    assert isinstance(raw_result, np.ndarray)
    with pytest.raises(ValueError):
        dm.send_voltages(DmCommands(np.zeros(dm.DM_Num), dm.V_Min - 1, dm.V_Max, dm.DM_Num))


def test_generic_sim_dm_accepts_typed_and_raw_commands():
    from ao_shaping.drivers.sim.dm.simulated_dm import SimulateDM

    dm = SimulateDM(max_iter_diff=0, noise_level=0)
    typed = DmCommands(np.full(dm.channel, 600.0), dm.v_min, dm.v_max, dm.channel)
    result = dm.send_voltages(typed)
    assert isinstance(result, DmCommands)
    np.testing.assert_allclose(result.voltages, dm.v_max)
    assert isinstance(dm.send_voltages(np.zeros(dm.channel)), np.ndarray)
