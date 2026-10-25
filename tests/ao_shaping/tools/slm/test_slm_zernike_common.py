from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from ao_shaping.tools.zernike.common import (
    added_tilt,
    collect_device_info,
    correction_artifact_name,
    flat_gray,
    index_of,
    make_phase,
    measure_tilt_defocus,
    nm_of,
    safe_pinv,
    um_to_waves,
    wfs_channels_lambda,
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


class _SlotProbeWFS:
    """A WFS stub returning one fixed driver-shaped reading (67 long, index 0 dead).

    Every slot carries a different value so any off-by-one in the reader is visible
    instead of hiding inside a plausible-looking number.
    """

    def __init__(self, vector: np.ndarray) -> None:
        self._vector = vector

    def take_image(self, n_sample: int = 1, dynamicNoiseCut: bool = True) -> None:
        return None

    def get_zernike(self, zernike_order: int = 10) -> np.ndarray:
        return self._vector.copy()


def _driver_shaped_reading() -> np.ndarray:
    z = np.zeros(67, dtype=float)
    z[2], z[3] = 0.532, -1.064      # (1,-1), (1,1)  -> +1.0 λ, -2.0 λ
    z[4], z[5] = 9.000, 1.596       # (2,-2) astig, (2,0) defocus -> 3.0 λ
    return z


def test_measure_tilt_defocus_reads_the_slots_its_own_table_documents() -> None:
    """Defocus is the (2,0) slot, which is 5 -- not the Noll number 4.

    Noll numbering and the driver's 1-based sequential-m enumeration agree for piston,
    tip and tilt and diverge from slot 4 onward, so a literal ``[4]`` here reads
    astigmatism and still looks like a sane coefficient.
    """
    assert index_of((2, 0)) == 5 and nm_of(4) == (2, -2), (
        "the enumeration changed; this test's literals must follow the authoritative map"
    )

    z_tilt, z_defocus = measure_tilt_defocus(cast(Any, _SlotProbeWFS(_driver_shaped_reading())), n_avg=1)

    assert z_tilt is not None and z_defocus is not None
    assert z_tilt == pytest.approx([1.0, -2.0])
    assert z_defocus == pytest.approx(3.0), (
        "defocus must come from the (2,0) slot; the astigmatism slot would give 9.0/0.532 "
        f"= {9.0 / 0.532:.4f}"
    )


def test_measure_tilt_defocus_drops_a_frame_whose_defocus_slot_is_not_finite() -> None:
    """A non-finite defocus slot cannot slip past the guard into the median."""
    z = _driver_shaped_reading()
    z[5] = np.nan

    z_tilt, z_defocus = measure_tilt_defocus(cast(Any, _SlotProbeWFS(z)), n_avg=1)

    assert z_tilt is None and z_defocus is None, (
        "the only frame was invalid, so the wrapper must report no measurement "
        f"(got tilt={z_tilt!r}, defocus={z_defocus!r})"
    )


#: A channel vector whose index 0 is the uncorrectable piston offset.
_CHANNELS = [5.0, 7.0, 11.0, 13.0, 17.0]


def test_wfs_channels_lambda_zeroes_piston_and_keeps_the_rest() -> None:
    w = wfs_channels_lambda(list(_CHANNELS), len(_CHANNELS))

    assert w == pytest.approx([0.0, 7.0, 11.0, 13.0, 17.0])


def test_wfs_channels_lambda_drops_the_leading_padding_slot() -> None:
    raw = [999.0, *_CHANNELS]                      # 0 号是未写的填充槽
    w = wfs_channels_lambda(raw, len(_CHANNELS))

    assert w.size == len(_CHANNELS)
    assert w == pytest.approx([0.0, 7.0, 11.0, 13.0, 17.0]), (
        "slot 0 is padding, so the piston is the ORIGINAL index 1 -- and it is the value zeroed"
    )


@pytest.mark.parametrize("key", ["w", "w_before", "wavefront"])
def test_wfs_channels_lambda_accepts_each_saved_dict_key(key: str) -> None:
    """The corpus writes this vector under three different names over time."""
    w = wfs_channels_lambda({key: list(_CHANNELS)}, len(_CHANNELS))

    assert w == pytest.approx([0.0, 7.0, 11.0, 13.0, 17.0])


def test_wfs_channels_lambda_reports_the_source_when_input_is_unusable() -> None:
    for bad, why in (({"other": [1.0]}, "no recognized key"), ([1.0, 2.0], "wrong length")):
        with pytest.raises(ValueError, match="--w-file") as exc:
            wfs_channels_lambda(bad, len(_CHANNELS), source="--w-file")
        assert str(exc.value), why


def test_wfs_channels_lambda_never_aliases_the_caller_buffer() -> None:
    """A slice is a VIEW, so the copy is what keeps piston-zeroing local to the result."""
    source = np.asarray(_CHANNELS, dtype=float)

    w = wfs_channels_lambda(source, len(_CHANNELS))
    w[1] = 99.0

    assert source == pytest.approx(_CHANNELS), "the caller's vector was mutated"
    assert not np.shares_memory(w, source)


def test_wfs_channels_lambda_treats_one_extra_slot_as_padding_at_the_real_length() -> None:
    """The rule is "one longer than the matrix expects", not a hardcoded width.

    The response-matrix h5 stores 66 WFS channels, so a 67-long saved reading is the same
    wavefront plus the slot the DLL never writes -- and the same reasoning holds at any
    other order, where a fixed 67 would reject a legitimate file.
    """
    channels = np.arange(1.0, 67.0)
    raw = np.concatenate(([0.0], channels))

    assert len(channels) == 66 and len(raw) == 67
    assert wfs_channels_lambda(raw, 66) == pytest.approx(wfs_channels_lambda(channels, 66))


def test_wfs_channels_lambda_rejects_a_length_beyond_one_padding_slot() -> None:
    with pytest.raises(ValueError, match="波前长度 68"):
        wfs_channels_lambda(np.arange(1.0, 69.0), 66)
