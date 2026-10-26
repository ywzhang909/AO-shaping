"""Offline tests for the 「远场尺寸换算」 popover helper (issue #204, GUI step).

``ao_shaping.gui.slm.link_design_assist`` is the pure half of the popover: it turns
the operator's units (cm / mm / m at the target distance) into the diagnostic
camera's µm / px and back, and it owns the locked design point that the
「光路参数」 expander shows. The Streamlit half (``ShapingMethod.render_link_assist``)
only places widgets, so everything worth pinning is testable here — no server, no
hardware, and **no streamlit import** in this module either.

What these tests hold down:

1. **The golden design point** (issue #204): far field 10 cm @ 1 km → CCD
   200 µm = 91 px, Airy first zero 28.85 µm = 13 px, coherent cells N = 8.46
   (90 mm exit pupil) / 9.40 (100 mm), scale ``L/(f_c·M_t)`` = 500. If the helper
   ever drifts from the notebook these numbers stop matching, which is exactly
   what an operator would mis-read off the popover.
2. **Single source of truth** — the conversion is not re-derived here: every
   length comes from :mod:`ao_shaping.utils.wavefront.link_design`
   (``far_to_ccd`` / ``ccd_to_far`` / ``airy_first_zero`` / ``params.scale``).
   Asserting ``summary["scale"] == params.scale`` catches a second copy of
   ``L/(f_c·M_t)`` that happens to agree at the design point but not off it.
3. **One field table** — ``LINK_FIELDS`` (what the expander renders) and
   ``DESIGN_POINT`` (what the conversion defaults to) must stay the same set of
   names, or a renamed field silently falls back to its default while the widget
   shows something else. ``build_params`` raises on unknown names for the same
   reason.
4. **The operator-facing text** — ``summary_rows`` is what is actually read out
   loud next to the camera, so its unit and rounding are pinned too.
"""

from __future__ import annotations

import pytest

from ao_shaping.gui.slm.link_design_assist import (
    DEFAULT_FAR_SIDE,
    DESIGN_POINT,
    FAR_UNITS,
    LINK_FIELDS,
    build_params,
    compute_link_summary,
    compute_reverse,
    far_side_to_m,
    format_ccd_length,
    format_far_length,
    summary_rows,
)
from ao_shaping.utils.wavefront.link_design import far_to_ccd

#: Readout window used for the design-point summary (the GS recipe's ``cam_size``).
DESIGN_WINDOW_PX = 250


def _summary(**overrides: float) -> dict[str, float]:
    """Design-point summary, optionally with one link field overridden."""
    return compute_link_summary(
        0.10, window_side_px=DESIGN_WINDOW_PX, params=build_params(**overrides)
    )


def test_design_point_matches_the_locked_issue_204_values() -> None:
    """10 cm @ 1 km → 200 µm = 91 px, Airy 28.85 µm = 13 px, scale 500."""
    summary = _summary()

    assert summary["ccd_side_um"] == pytest.approx(200.0, abs=0.5)
    assert round(summary["ccd_side_px"]) == 91
    assert summary["coherent_cells"] == pytest.approx(8.46, abs=0.01)
    assert summary["airy_first_zero_um"] == pytest.approx(28.85, abs=0.05)
    assert round(summary["airy_first_zero_px"]) == 13
    assert summary["scale"] == pytest.approx(500.0, rel=1e-12)
    assert summary["window_far_m"] == pytest.approx(0.275, rel=1e-6)


def test_a_100_mm_exit_pupil_gives_9_40_coherent_cells() -> None:
    """The notebook's second golden point (100 mm objective)."""
    summary = _summary(exit_pupil_diameter_mm=100.0)

    assert summary["coherent_cells"] == pytest.approx(9.40, abs=0.01)


def test_reverse_91_px_reads_back_as_10_cm() -> None:
    """The popover's reverse section: 91 px of the diagnostic camera = 10 cm."""
    reverse = compute_reverse(91, params=build_params())

    assert reverse["far_side_m"] == pytest.approx(0.1001, abs=1e-3)


def test_the_reverse_round_trip_returns_the_far_side() -> None:
    """Forward then reverse, rounded to whole pixels, must land on 10 cm."""
    summary = _summary()
    reverse = compute_reverse(round(summary["ccd_side_px"]), params=build_params())

    assert reverse["far_side_m"] == pytest.approx(0.10, abs=1e-3)


@pytest.mark.parametrize(("value", "unit", "metres"), [(10.0, "cm", 0.1), (10.0, "mm", 0.01), (0.1, "m", 0.1)])
def test_far_side_to_m_converts_every_offered_unit(
    value: float, unit: str, metres: float
) -> None:
    """Each unit the selectbox offers maps to metres; the default is one of them."""
    assert far_side_to_m(value, unit) == pytest.approx(metres, rel=1e-12)


def test_the_default_far_side_unit_is_an_offered_unit() -> None:
    """``DEFAULT_FAR_SIDE["unit"]`` indexes ``FAR_UNITS`` when the selectbox opens."""
    assert str(DEFAULT_FAR_SIDE["unit"]) in FAR_UNITS


def test_an_unknown_unit_is_rejected() -> None:
    """A typo must raise, not silently convert as if it were metres."""
    with pytest.raises(ValueError):
        far_side_to_m(10.0, "km")


def test_build_params_rejects_an_unknown_field() -> None:
    """A renamed field must fail loudly instead of falling back to its default."""
    with pytest.raises(TypeError):
        build_params(unknown_field=1.0)


def test_build_params_defaults_are_the_design_point() -> None:
    """No overrides = the locked point (compared through derived values, not identity)."""
    params = build_params()

    assert params.target_distance == pytest.approx(1.0e3)
    assert params.effective_focal_length == pytest.approx(2.0)
    assert params.scale == pytest.approx(500.0)
    assert params.ccd_pixel == pytest.approx(2.2e-6)


def test_every_link_field_is_a_design_point_key() -> None:
    """The expander's field table and the conversion defaults share one name set."""
    assert {field.name for field in LINK_FIELDS} == set(DESIGN_POINT)


def test_every_link_field_default_is_inside_its_range() -> None:
    """A default outside ``[lo, hi]`` makes the number_input raise on render."""
    for field in LINK_FIELDS:
        assert field.lo <= field.default <= field.hi, field.name


def test_summary_rows_carry_the_operator_facing_numbers() -> None:
    """The rendered text is the operator's only read-out: pin unit and rounding."""
    text = " ".join(value for _label, value in summary_rows(_summary()))

    assert "200.0 µm = 91 px" in text
    assert "8.46" in text


def test_the_scale_is_not_re_derived_by_the_helper() -> None:
    """``scale`` and the CCD length must be the ones ``link_design`` reports.

    A second copy of ``L/(f_c·M_t)`` would agree at the design point, so this also
    checks a non-design point (M_t = 5 → scale 1000) against ``far_to_ccd``.
    """
    params = build_params(telescope_magnification=5.0)
    summary = compute_link_summary(0.10, window_side_px=DESIGN_WINDOW_PX, params=params)
    w_ccd_m, w_ccd_px = far_to_ccd(0.10, params)

    assert summary["scale"] == params.scale
    assert summary["scale"] == pytest.approx(1000.0, rel=1e-12)
    assert summary["ccd_side_m"] == pytest.approx(w_ccd_m, rel=1e-12)
    assert summary["ccd_side_px"] == pytest.approx(w_ccd_px, rel=1e-12)


def test_format_far_length_switches_unit_at_1_m_and_1_cm() -> None:
    """Far-field lengths read in m above 1 m, cm above 1 cm, mm below."""
    assert format_far_length(0.275) == "27.50 cm"
    assert format_far_length(1.5) == "1.500 m"
    assert format_far_length(0.00275) == "2.75 mm"


def test_format_ccd_length_is_always_um() -> None:
    """The camera plane keeps its own scale: always µm, one decimal."""
    assert format_ccd_length(200e-6) == "200.0 µm"
