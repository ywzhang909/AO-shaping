from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from ao_shaping.tools.slm.slm_zernike_common import (
    added_tilt,
    collect_device_info,
    correction_artifact_name,
    flat_gray,
    index_of,
    make_phase,
    nm_of,
    safe_pinv,
    um_to_waves,
)


class FakePatternHelper:
    def __init__(self):
        self.calls = []

    def generate_zernike_polynomial(self, **kwargs) -> np.ndarray:
        self.calls.append(kwargs)
        return np.full((4, 4), 0.25, dtype=np.float64)


class FakeSLM:
    Panel_Res = (6, 8)
    MAX_GRAYSCALE_VALUE = 1023
    shift_x = 3
    shift_y = -2
    correction_enabled = False
    correction_csv_path = None
    lut = None
    current_grayscale = 17
    is_open = True
    Response_time_ms = 12.0
    MAX_PIXEL_FLIP_TIME_MS = 200.0

    def __init__(self):
        self.serial_number = "SLM-1"
        self.display_name = "test"
        self.version = {"dll": 1}
        self.temperature = (20.0, 21.0)
        self.video_mode = 0

    def get_wavelength_info(self):
        return 532, 993


class FakeWFS:
    exposure_time = 0.2
    pupil = (1.0, 2.0, 3.0, 4.0)
    num_spots_x = 27
    num_spots_y = 27
    mla_index = "768"
    use_custom_ref = False
    high_speed = True
    master_gain = 1.0

    def get_hardware_info(self):
        return {
            "serial_number": "WFS-1",
            "device_name": "test wfs",
            "manufacturer": "Thorlabs",
            "model": "WFS",
            "firmware_version": "1",
        }

    def get_mla_name(self):
        return "768"


def test_wfs_units_convert_micrometres_to_waves() -> None:
    assert um_to_waves(np.array([0.0, 0.532, 1.064])) == pytest.approx([0.0, 1.0, 2.0])


def test_dll_zernike_index_mapping_is_reversible() -> None:
    assert nm_of(5) == (2, 0)
    assert nm_of(13) == (4, 0)
    assert index_of((2, 0)) == 5
    assert index_of((4, 0)) == 13
    assert index_of((11, 0)) is None


def test_safe_pinv_ignores_zero_columns() -> None:
    matrix = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])

    pinv = safe_pinv(matrix)

    assert pinv.shape == (2, 3)
    assert pinv[0] == pytest.approx([1.0 / 14.0, 2.0 / 14.0, 3.0 / 14.0])
    assert pinv[1] == pytest.approx([0.0, 0.0, 0.0])


def test_make_phase_preserves_coefficients_radius_and_order() -> None:
    helper = FakePatternHelper()

    phase = make_phase(cast(Any, helper), {(2, 0): 1.5}, radius=12.0, n_max=3)

    assert phase.shape == (4, 4)
    assert helper.calls == [{"coefficients": {(2, 0): 1.5}, "radius": 12.0, "n_max": 4}]


def test_correction_artifact_name_contains_replay_metadata() -> None:
    name = correction_artifact_name(
        serial="abc",
        wavelength_nm=532,
        shift_x=1,
        shift_y=-2,
        zernike_radius=300.0,
        ts="20260917_010203",
    )

    assert name == "slm_corr_abc_532nm_shift1_-2_R300_20260917_010203.csv"


def test_flat_gray_uses_direct_uint16_path() -> None:
    phase = flat_gray()

    assert phase.shape == (1200, 1920)
    assert phase.dtype == np.uint16
    assert np.all(phase == 0)


def test_added_tilt_uses_waves_relative_to_baseline() -> None:
    z = np.array([0.0, 0.0, 0.532, 1.064])
    base = np.zeros_like(z)

    norm, components = added_tilt(z, base)

    assert components == pytest.approx([1.0, 2.0])
    assert norm == pytest.approx(np.sqrt(5.0))


def test_collect_device_info_records_available_fields() -> None:
    info = collect_device_info(cast(Any, FakeSLM()), cast(Any, FakeWFS()), slm_number=2)

    assert info["slm"]["slm_number"] == 2
    assert info["slm"]["wavelength_nm"] == 532
    assert info["slm"]["two_pi_gray"] == 993
    assert info["slm"]["panel_res"] == [6, 8]
    assert info["wfs"]["serial_number"] == "WFS-1"
    assert info["wfs"]["pupil_center_mm"] == [1.0, 2.0]
