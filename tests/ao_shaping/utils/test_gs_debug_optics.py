from __future__ import annotations

import numpy as np
import pytest

from scripts.slm.generate_gif_from_pkl import _pupil_geometry
from ao_shaping.utils.slm.gs_debug_optics import grayscale_to_radians, idealized_far_field


def test_grayscale_conversion_validates_integer_lut_range() -> None:
    np.testing.assert_allclose(grayscale_to_radians(np.array([[0, 1023]], dtype=np.uint16)), [[0, 2 * np.pi]])
    with pytest.raises(ValueError, match="integer"):
        grayscale_to_radians(np.zeros((2, 2), dtype=float))
    with pytest.raises(ValueError, match="outside"):
        grayscale_to_radians(np.array([[1024]], dtype=np.int32))


def test_far_field_uses_off_center_pupil() -> None:
    phase = np.zeros((32, 40), dtype=float)
    image = idealized_far_field(phase, pupil_center=(10, 20), beam_radius_px=6)
    assert image.shape == phase.shape
    assert np.isclose(image.max(), 1.0)
    with pytest.raises(ValueError, match="radius"):
        idealized_far_field(phase, pupil_center=(10, 20), beam_radius_px=0)


def test_gif_geometry_reads_sidecar_and_legacy_override(tmp_path) -> None:
    pkl = tmp_path / "slm_gs_refine_20261012_180222.pkl"
    pkl.with_suffix(".json").write_text(
        '{"beam_radius_px": 450, "pupil_center": [960, 690]}', encoding="utf-8"
    )
    assert _pupil_geometry(pkl, None) == ((960.0, 690.0), 450.0)
    assert _pupil_geometry(pkl, (900, 600)) == ((900.0, 600.0), 450.0)
    pkl.with_suffix(".json").write_text('{"beam_radius_px": 450}', encoding="utf-8")
    assert _pupil_geometry(pkl, None) is None
