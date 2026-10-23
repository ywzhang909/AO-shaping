"""Tests for session-level bench configuration validation (``session_profile``).

All offline: the probe callback is a fake device producing synthetic frames
(Gaussian peak at a chosen location + background), no driver or SDK is touched.

Incident background: the 2f bench's default sampling window was built on the
fitting-day zero-order position; a later session measured the zero order
~72 px away, and carrying the old window forward degraded the results
silently (forward refit loss 0.928 -> 0.356 after recentering, model CV
1.497 -> 1.238).  What is pinned here is the **contract**:

1. everything within tolerance  ->  ``reuse`` per item, no new profile;
2. zero order off by ~72 px     ->  ``spot_centroid`` / ``window_centre`` are
   ``rebuild``, and the new profile's window is centred on the measured spot
   with margin >= ``window_margin_px``;
3. off by 3 px                   ->  ``reuse`` (within tolerance);
4. exposure off by 2x            ->  ``exposure`` is ``rebuild``;
5. missing / unmeasurable        ->  ``reject`` with an explicit reason (never
   a silent ``reuse``);
6. ``window_margin_px`` tripped  ->  ``rebuild`` with a larger window;
7. the report survives a JSON round trip, every reason non-empty.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from ao_shaping.tools.slm.panel import session_profile as sp

#: Fitting-day zero order and window (the incident's stored configuration).
STORED_SPOT = (1359.8, 1008.5)
#: Measured zero order in the later session (~72 px away).
MEASURED_SPOT = (1418.0, 966.0)


def _synthetic_frame(
    centre_xy: tuple[float, float], *, shape: tuple[int, int] | None = None
) -> np.ndarray:
    """A synthetic CCD frame: Gaussian peak at ``centre_xy`` on zero background.

    ``centre_xy`` is ``(x=column, y=row)`` in camera pixels.  The frame is
    sized to contain the peak (40 px of background margin) unless ``shape``
    is given explicitly.
    """
    cx, cy = centre_xy
    if shape is None:
        height = max(int(math.ceil(cy)) + 40, 64)
        width = max(int(math.ceil(cx)) + 40, 64)
    else:
        height, width = shape
    yy, xx = np.mgrid[:height, :width]
    return 1000.0 * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * 9.0**2)))


def _stored_profile(**overrides) -> sp.SessionProfile:
    base = dict(
        spot_centroid_xy=STORED_SPOT,
        window_centre_xy=STORED_SPOT,
        window_half_span_px=120.0,
        sample_step_px=2.0,
        exposure_ms=0.4,
        focus_diopter=0.5607,
        best_focal_mm=1783.5,
        panel_shape=(1200, 1920),
        slm_id=1,
        cam_id=0,
        recorded_utc="2026-10-17T12:00:00+00:00",
        source="scripts/slm/hw_forward_fit.py (fitting-day session)",
    )
    base.update(overrides)
    return sp.SessionProfile(**base)


def _make_probe(
    centre_xy: tuple[float, float],
    *,
    frame: np.ndarray | None = None,
    exposure_ms: float | None = 0.4,
    focus_diopter: float | None = 0.5607,
    best_focal_mm: float | None = 1783.5,
    calls: list | None = None,
):
    def probe() -> sp.ProbeResult:
        if calls is not None:
            calls.append(1)
        return sp.ProbeResult(
            frame=frame if frame is not None else _synthetic_frame(centre_xy),
            exposure_ms=exposure_ms,
            focus_diopter=focus_diopter,
            best_focal_mm=best_focal_mm,
        )

    return probe


def _verdicts(report: sp.ValidationReport) -> dict[str, str]:
    return {item.item: item.verdict for item in report.items}


# ---------------------------------------------------------------------------
# 1. everything matches
# ---------------------------------------------------------------------------
def test_full_match_all_reuse():
    stored = _stored_profile()
    probe = _make_probe((1361.0, 1006.5))  # offset ~3.2 px, within 5 px
    report = sp.validate_or_rebuild(stored, probe)

    assert report.status == sp.VERDICT_REUSE
    assert report.new_profile is None
    verdicts = _verdicts(report)
    assert set(verdicts) == set(sp._ITEM_NAMES)
    assert all(v == sp.VERDICT_REUSE for v in verdicts.values())
    # every reason must be non-empty
    assert all(item.reason.strip() for item in report.items)


def test_validate_calls_probe_exactly_once():
    calls: list[int] = []
    stored = _stored_profile()
    sp.validate_or_rebuild(stored, _make_probe(MEASURED_SPOT, calls=calls))
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# 2. zero order off by ~72 px (the incident)
# ---------------------------------------------------------------------------
def test_72px_shift_rebuilds_window_around_measured_spot():
    stored = _stored_profile()
    report = sp.validate_or_rebuild(stored, _make_probe(MEASURED_SPOT))

    assert report.status == sp.VERDICT_REBUILD
    assert report.new_profile is not None
    verdicts = _verdicts(report)
    assert verdicts["spot_centroid"] == sp.VERDICT_REBUILD
    assert verdicts["window_centre"] == sp.VERDICT_REBUILD
    # the other items still match
    assert verdicts["exposure"] == sp.VERDICT_REUSE
    assert verdicts["focus"] == sp.VERDICT_REUSE

    new = report.new_profile
    # new window centre == measured zero order
    assert new.window_centre_xy == pytest.approx(MEASURED_SPOT)
    assert new.spot_centroid_xy == pytest.approx(MEASURED_SPOT)
    # margin of the new window (centred on the spot) is >= window_margin_px
    assert new.window_half_span_px >= sp.Tolerance().window_margin_px
    # unchanged fields carry over
    assert new.exposure_ms == 0.4
    assert new.sample_step_px == 2.0
    assert new.panel_shape == (1200, 1920)
    # the rebuilt window is exactly the recentre_window output around the spot
    window = new.camera_xy()
    assert window.shape == (121, 121, 2)
    centre_pixel = window[60, 60]
    assert centre_pixel[0] == pytest.approx(MEASURED_SPOT[0])
    assert centre_pixel[1] == pytest.approx(MEASURED_SPOT[1])


def test_72px_shift_off_axis_margin_still_ok():
    """With the stored window centred on the stored spot, the ~72 px shift
    (dx=58.2, dy=42.5) still leaves >= 40 px on both axes, so the margin item
    is ``reuse`` -- it is the centre items that fire."""
    stored = _stored_profile()
    report = sp.validate_or_rebuild(stored, _make_probe(MEASURED_SPOT))
    assert report.verdict_of("window_margin").verdict == sp.VERDICT_REUSE


# ---------------------------------------------------------------------------
# 3. zero order off by 3 px -> reuse
# ---------------------------------------------------------------------------
def test_3px_shift_is_reuse():
    stored = _stored_profile()
    report = sp.validate_or_rebuild(stored, _make_probe((1362.3, 1010.0)))
    assert report.status == sp.VERDICT_REUSE
    assert report.new_profile is None
    assert _verdicts(report)["spot_centroid"] == sp.VERDICT_REUSE


# ---------------------------------------------------------------------------
# 4. exposure off by 2x -> rebuild
# ---------------------------------------------------------------------------
def test_exposure_2x_rebuild():
    stored = _stored_profile()
    report = sp.validate_or_rebuild(
        stored, _make_probe(STORED_SPOT, exposure_ms=0.8)
    )
    assert report.status == sp.VERDICT_REBUILD
    assert _verdicts(report)["exposure"] == sp.VERDICT_REBUILD
    assert report.new_profile is not None
    # the rebuilt profile adopts the probe's current exposure
    assert report.new_profile.exposure_ms == 0.8
    # spot-related items stay reuse
    assert _verdicts(report)["spot_centroid"] == sp.VERDICT_REUSE


def test_exposure_within_relative_tolerance_is_reuse():
    stored = _stored_profile()
    # 0.4 -> 0.48 is 20% < 25%
    report = sp.validate_or_rebuild(stored, _make_probe(STORED_SPOT, exposure_ms=0.48))
    assert _verdicts(report)["exposure"] == sp.VERDICT_REUSE


# ---------------------------------------------------------------------------
# 5. missing / unmeasurable -> reject, never silent reuse
# ---------------------------------------------------------------------------
def test_missing_stored_spot_centroid_rejects():
    stored = _stored_profile(spot_centroid_xy=None)
    report = sp.validate_or_rebuild(stored, _make_probe(STORED_SPOT))
    item = report.verdict_of("spot_centroid")
    assert item.verdict == sp.VERDICT_REJECT
    assert item.reason.strip()
    assert report.status == sp.VERDICT_REJECT
    assert report.new_profile is None


def test_unmeasurable_centroid_rejects():
    stored = _stored_profile()
    dark = np.zeros((400, 500))  # no signal -> argmax is meaningless
    report = sp.validate_or_rebuild(stored, _make_probe(STORED_SPOT, frame=dark))
    item = report.verdict_of("spot_centroid")
    assert item.verdict == sp.VERDICT_REJECT
    assert "无法测量" in item.reason or "unmeasurable" in item.reason
    assert report.status == sp.VERDICT_REJECT


def test_missing_exposure_rejects():
    stored = _stored_profile(exposure_ms=None)
    report = sp.validate_or_rebuild(stored, _make_probe(STORED_SPOT))
    assert _verdicts(report)["exposure"] == sp.VERDICT_REJECT


def test_probe_without_exposure_report_rejects():
    stored = _stored_profile()
    report = sp.validate_or_rebuild(stored, _make_probe(STORED_SPOT, exposure_ms=None))
    assert _verdicts(report)["exposure"] == sp.VERDICT_REJECT


def test_missing_focus_rejects():
    stored = _stored_profile(focus_diopter=None, best_focal_mm=None)
    report = sp.validate_or_rebuild(stored, _make_probe(STORED_SPOT))
    assert _verdicts(report)["focus"] == sp.VERDICT_REJECT


def test_no_stored_profile_at_all_rejects_everything():
    report = sp.validate_or_rebuild(None, _make_probe(STORED_SPOT))
    assert report.status == sp.VERDICT_REJECT
    assert report.new_profile is None
    assert len(report.items) == len(sp._ITEM_NAMES)
    for item in report.items:
        assert item.verdict == sp.VERDICT_REJECT
        assert item.reason.strip()


def test_session_profile_validation_errors():
    with pytest.raises(sp.SessionProfileError):
        sp.SessionProfile(exposure_ms=-1.0)
    with pytest.raises(sp.SessionProfileError):
        sp.SessionProfile(spot_centroid_xy=(1.0, float("nan")))
    with pytest.raises(sp.SessionProfileError):
        sp.Tolerance(centroid_px=-1.0)


# ---------------------------------------------------------------------------
# 6. window_margin_px trips -> rebuild with a larger window
# ---------------------------------------------------------------------------
def test_window_too_small_rebuilds_with_larger_span():
    # the probe spot is integral so the discrete argmax recovers it exactly
    probe_spot = (1360.0, 1008.0)  # 0.58 px from the stored spot -> reuse
    stored = _stored_profile(window_half_span_px=30.0)  # 30 < 40 px required
    report = sp.validate_or_rebuild(stored, _make_probe(probe_spot))
    item = report.verdict_of("window_margin")
    assert item.verdict == sp.VERDICT_REBUILD
    assert "余量" in item.reason
    assert report.new_profile is not None
    # enlarged so the margin (== half span, window centred on the spot) meets
    # the tolerance, and the window is centred on the measured zero order
    assert report.new_profile.window_half_span_px >= sp.Tolerance().window_margin_px
    assert report.new_profile.window_centre_xy == pytest.approx(probe_spot)


def test_window_outside_rebuilds():
    # stored window far away from the measured spot -> spot outside the window
    stored = _stored_profile(window_centre_xy=(200.0, 200.0))
    report = sp.validate_or_rebuild(stored, _make_probe(MEASURED_SPOT))
    item = report.verdict_of("window_margin")
    assert item.verdict == sp.VERDICT_REBUILD
    assert "窗外" in item.reason


# ---------------------------------------------------------------------------
# 7. report JSON round trip
# ---------------------------------------------------------------------------
def test_report_json_roundtrip():
    stored = _stored_profile()
    report = sp.validate_or_rebuild(stored, _make_probe(MEASURED_SPOT))
    payload = json.dumps(report.to_dict(), ensure_ascii=False)
    restored = sp.ValidationReport.from_dict(json.loads(payload))

    # structural equality after the round trip
    assert restored.to_dict() == report.to_dict()
    assert restored.status == report.status
    assert restored.new_profile is not None
    assert restored.new_profile.window_centre_xy == pytest.approx(MEASURED_SPOT)
    # every reason non-empty, on both sides
    assert all(item.reason.strip() for item in restored.items)
    # item names preserved
    assert {i.item for i in restored.items} == set(sp._ITEM_NAMES)


def test_profile_json_roundtrip():
    stored = _stored_profile()
    restored = sp.SessionProfile.from_dict(json.loads(json.dumps(stored.to_dict())))
    assert restored.to_dict() == stored.to_dict()
    assert restored.focus_diopter_effective() == pytest.approx(0.5607)


def test_profile_save_load_roundtrip(tmp_path):
    stored = _stored_profile()
    path = sp.save_profile(stored, tmp_path / sp.PROFILE_FILENAME)
    assert path.is_file()
    loaded = sp.load_profile(path)
    assert loaded.to_dict() == stored.to_dict()
    with pytest.raises(sp.SessionProfileError):
        sp.load_profile(tmp_path / "does_not_exist.json")


# ---------------------------------------------------------------------------
# probe helpers
# ---------------------------------------------------------------------------
def test_spot_centroid_of_frame_uses_argmax_not_geometry():
    frame = _synthetic_frame((123.0, 45.0))
    assert sp.spot_centroid_of_frame(frame) == pytest.approx((123.0, 45.0))
    # off-centre peak must NOT come back as the geometric centre
    assert sp.spot_centroid_of_frame(frame) != pytest.approx((250.0, 200.0))


def test_spot_centroid_unmeasurable_frames():
    assert sp.spot_centroid_of_frame(np.zeros((10, 10))) is None
    assert sp.spot_centroid_of_frame(np.full((10, 10), -3.0)) is None
    assert sp.spot_centroid_of_frame(np.full((10, 10), float("nan"))) is None
    assert sp.spot_centroid_of_frame(np.zeros((4,))) is None


def test_probe_helpers_report_unmeasurable_as_none():
    dark = np.zeros((10, 10))
    assert sp.probe_spot_centroid(lambda: sp.ProbeResult(frame=dark)) is None
    assert sp.probe_exposure_ms(lambda: sp.ProbeResult(frame=dark, exposure_ms=None)) is None
    assert sp.probe_focus_diopter(
        lambda: sp.ProbeResult(frame=dark, focus_diopter=None, best_focal_mm=None)
    ) is None


def test_probe_focus_from_focal_length():
    value = sp.probe_focus_diopter(
        lambda: sp.ProbeResult(frame=_synthetic_frame((10.0, 10.0)), best_focal_mm=1783.5)
    )
    assert value == pytest.approx(1000.0 / 1783.5, rel=1e-12)


def test_rebuild_for_spot_convenience():
    profile = sp.SessionProfile.rebuild_for_spot(
        MEASURED_SPOT,
        half_span_px=120.0,
        exposure_ms=0.4,
        focus_diopter=0.5607,
        source="manual test",
    )
    assert profile.window_centre_xy == pytest.approx(MEASURED_SPOT)
    assert profile.spot_centroid_xy == pytest.approx(MEASURED_SPOT)
    window = profile.camera_xy()
    assert window.shape == (121, 121, 2)
    # window spans exactly +- half_span around the spot
    xs = window[..., 0]
    ys = window[..., 1]
    assert xs.min() == pytest.approx(MEASURED_SPOT[0] - 120.0)
    assert xs.max() == pytest.approx(MEASURED_SPOT[0] + 120.0)
    assert ys.min() == pytest.approx(MEASURED_SPOT[1] - 120.0)
    assert ys.max() == pytest.approx(MEASURED_SPOT[1] + 120.0)


def test_recenter_for_spot_keeps_other_fields():
    stored = _stored_profile()
    new = stored.recenter_for_spot(MEASURED_SPOT)
    assert new.window_centre_xy == pytest.approx(MEASURED_SPOT)
    assert new.exposure_ms == 0.4
    assert new.sample_step_px == 2.0
    assert new.panel_shape == (1200, 1920)
    # the old profile is untouched (frozen)
    assert stored.window_centre_xy == STORED_SPOT
    assert new.source.startswith("SessionProfile.recenter_for_spot")


def test_custom_tolerances():
    stored = _stored_profile()
    # a 72 px shift is fine under a huge tolerance
    loose = sp.Tolerance(centroid_px=100.0)
    report = sp.validate_or_rebuild(stored, _make_probe(MEASURED_SPOT), loose)
    assert report.status == sp.VERDICT_REUSE
    # ... and an 8 px shift is not
    tight = sp.Tolerance(centroid_px=5.0)
    report = sp.validate_or_rebuild(stored, _make_probe((1367.8, 1008.5)), tight)
    assert report.status == sp.VERDICT_REBUILD
    assert math.hypot(1367.8 - STORED_SPOT[0], 1008.5 - STORED_SPOT[1]) > 5.0
