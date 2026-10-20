"""Unit contract for the shared SPGD-PIB scaffold (R-69 P2 / issue #91).

``optimizer/wfless/pib.py`` and ``optimizer/combined_optimizer.py`` are a
same-family fork: both open a camera and a DM, take a flat-field reading, and
then run a paired-measurement SPGD loop. The mechanics were duplicated and had
drifted, so a fix applied to one path missed the other.

``optimizer/spid_pib_loop.py`` single-homes those mechanics. These tests pin
each piece's contract in isolation with fake devices (no SLM, no camera, no
simulation bench); the end-to-end behaviour is pinned by
``test_spid_pib_path_characterization.py``, which freezes the trajectory each
path produces.

The divergences the scaffold deliberately keeps at the call site are pinned
here too, so that "unify the mechanics" cannot quietly become "unify the
policy": the two paths use different RNG streams, different brightness
statistics and different centre conventions.
"""

from __future__ import annotations

import numpy as np
import pytest

from ao_shaping.optimizer import spid_pib_loop as scaffold


class _FakeCam:
    """Camera stub: hands out canned frames, records the window it was given."""

    def __init__(self, frame: np.ndarray) -> None:
        self.frame = frame
        self.window_center: tuple[int, int] | None = None
        self.window_size: tuple[int, int] | None = None
        self.grabs = 0

    def __enter__(self) -> _FakeCam:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        return None

    def get_numpy_image(self, n_frames: int) -> np.ndarray:
        self.grabs += 1
        return self.frame

    def reset_window(
        self, center: tuple[int, int], size: tuple[int, int]
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        self.window_center = center
        self.window_size = size
        return size, center


class _FakeDM:
    """DM stub: records every voltage write, reports a fixed neighbour check."""

    def __init__(self, n: int = 4, safe: bool = True) -> None:
        self.DM_Num = n
        self.V_Min = -12.0
        self.V_Max = 12.0
        self.default_dm_unit_mask = np.ones(n, dtype=bool)
        self.safe = safe
        self.written: list[np.ndarray] = []
        self.opened = 0
        self.closed = 0

    def open(self) -> None:
        self.opened += 1

    def close(self) -> None:
        self.closed += 1

    def is_connected(self) -> bool:
        return self.opened > 0

    def send_voltages(self, vs: np.ndarray, wait: float = 0.0) -> np.ndarray:
        arr = np.asarray(vs, dtype=np.float64)
        self.written.append(arr)
        return arr

    def check_dm_unit_grad_safe(self, vs: np.ndarray) -> bool:
        return self.safe


class _FakeOptimizer:
    """Optimizer stub: returns a fixed descent step."""

    def __init__(self, step: float = 0.5) -> None:
        self.step = step
        self.gradients: list[np.ndarray] = []

    def update(self, gradient: np.ndarray) -> np.ndarray:
        self.gradients.append(gradient)
        return np.full_like(gradient, self.step)


def _patch_bench(monkeypatch, cam: _FakeCam, detected=(11, 22)):
    """Point the scaffold at fakes and neutral hardware helpers."""
    monkeypatch.setattr(scaffold, "_camera_cls", lambda cam_type: lambda **kw: cam)
    monkeypatch.setattr(
        scaffold, "capture_with_exposure", lambda cam, *a, **kw: cam.frame
    )
    monkeypatch.setattr(
        scaffold, "resolve_spot_center", lambda cam, img, center, **kw: detected
    )
    monkeypatch.setattr(scaffold, "log_center_brightness", lambda *a, **kw: None)
    monkeypatch.setattr(scaffold, "get_camera_exposure_ms", lambda cam: 20.0)


def _bench_kwargs(**overrides):
    kwargs: dict = dict(
        cam_type="",
        cam_id=0,
        exposure_time_ms=20.0,
        cam_size=64,
        target_max_brightness=0.0,
        center=None,
        dm=None,
        create_dm=lambda: _FakeDM(),
        init_v=None,
        dm_unit_mask=None,
        init_fallback_target_max=220,
    )
    kwargs.update(overrides)
    return kwargs


# ------------------------------------------------------------------ bench ----
def test_bench_reports_the_centre_it_detected_and_the_window_centre(monkeypatch):
    """The two centres are different things and the caller picks one.

    The SPGD path measures against the centre ``reset_window`` returns, the
    combined path against the centre detected on the probe. Collapsing them
    would move one path's metrics, so the scaffold exposes both.
    """
    cam = _FakeCam(np.zeros((8, 8), dtype=np.uint8))
    _patch_bench(monkeypatch, cam, detected=(11, 22))

    with scaffold.open_pib_bench(**_bench_kwargs()) as bench:
        assert bench.center == (11, 22)
        assert bench.window_center == (11, 22)
        assert cam.window_center == (11, 22)
        assert cam.window_size == (64, 64)


def test_bench_normalises_the_unit_mask_to_float(monkeypatch):
    """A bool mask must become 0.0/1.0 before it scales a perturbation."""
    cam = _FakeCam(np.zeros((8, 8), dtype=np.uint8))
    _patch_bench(monkeypatch, cam)
    dm = _FakeDM()
    mask = np.array([True, False, True, False])

    with scaffold.open_pib_bench(**_bench_kwargs(dm=dm, dm_unit_mask=mask)) as bench:
        assert bench.unit_mask.dtype == np.float64
        np.testing.assert_array_equal(bench.unit_mask, [1.0, 0.0, 1.0, 0.0])


def test_bench_sends_the_initial_voltages_once(monkeypatch):
    """The flat-field voltage write happens before anything is measured."""
    cam = _FakeCam(np.zeros((8, 8), dtype=np.uint8))
    _patch_bench(monkeypatch, cam)
    dm = _FakeDM()

    with scaffold.open_pib_bench(**_bench_kwargs(dm=dm, init_v=[1.0, 2.0, 3.0, 4.0])):
        assert len(dm.written) == 1
        np.testing.assert_array_equal(dm.written[0], [1.0, 2.0, 3.0, 4.0])


def test_bench_does_not_close_a_dm_it_did_not_create(monkeypatch):
    """An injected DM belongs to the caller; closing it here would break it."""
    cam = _FakeCam(np.zeros((8, 8), dtype=np.uint8))
    _patch_bench(monkeypatch, cam)
    dm = _FakeDM()

    with scaffold.open_pib_bench(**_bench_kwargs(dm=dm)):
        pass

    assert dm.closed == 0


def test_bench_closes_a_dm_it_created_when_the_setup_fails(monkeypatch):
    """A failed probe must not leave the DM open on the hardware."""
    cam = _FakeCam(np.zeros((8, 8), dtype=np.uint8))
    _patch_bench(monkeypatch, cam)
    dm = _FakeDM()

    def boom(*args, **kwargs):
        raise RuntimeError("probe failed")

    monkeypatch.setattr(scaffold, "capture_with_exposure", boom)

    with pytest.raises(RuntimeError, match="probe failed"):
        with scaffold.open_pib_bench(**_bench_kwargs(create_dm=lambda: dm)):
            pass

    assert dm.closed == 1


def test_bench_does_not_close_the_dm_when_the_caller_body_fails(monkeypatch):
    """The caller's ``finally`` owns the close; the scaffold must not double-close."""
    cam = _FakeCam(np.zeros((8, 8), dtype=np.uint8))
    _patch_bench(monkeypatch, cam)
    dm = _FakeDM()

    with pytest.raises(ValueError, match="caller failed"):
        with scaffold.open_pib_bench(**_bench_kwargs(create_dm=lambda: dm)):
            raise ValueError("caller failed")

    assert dm.closed == 0


# ------------------------------------------------------------------ epoch ----
def _epoch_kwargs(**overrides):
    kwargs: dict = dict(
        dm=_FakeDM(),
        cam=_FakeCam(np.zeros((8, 8), dtype=np.uint8)),
        init_v=np.zeros(4, dtype=np.float64),
        delta=1.0,
        unit_mask=np.ones(4, dtype=np.float64),
        rng=np.random.default_rng(0),
        optimizer=_FakeOptimizer(),
        evaluate=lambda img: (1.0, 0.5),
        maximize=True,
        diff_sign=1.0,
    )
    kwargs.update(overrides)
    return kwargs


def test_epoch_measures_both_polarities_of_the_perturbation():
    """The pair is what makes the gradient: +perturb then -perturb."""
    dm = _FakeDM()
    kwargs = _epoch_kwargs(dm=dm, init_v=np.array([10.0, 10.0, 10.0, 10.0]))

    result = scaffold.spid_pib_epoch(**kwargs)

    assert len(dm.written) == 2
    plus, minus = dm.written
    np.testing.assert_array_equal(plus, 10.0 + result.disturb_v)
    np.testing.assert_array_equal(minus, 10.0 - result.disturb_v)
    assert np.all(np.abs(result.disturb_v) == 1.0)


def test_epoch_uses_the_shared_sign_helper():
    """The gradient must come from ``spgd_gradient``, not from a local sign."""
    optimizer = _FakeOptimizer()
    kwargs = _epoch_kwargs(optimizer=optimizer)

    result = scaffold.spid_pib_epoch(**kwargs)

    expected = scaffold.spgd_gradient(
        1.0, 1.0, result.disturb_v, maximize=True
    )
    np.testing.assert_allclose(optimizer.gradients[0], expected)


def test_epoch_clips_the_update_to_the_dm_voltage_range():
    """A step beyond the range is clipped, not applied verbatim."""
    dm = _FakeDM()
    dm.V_Min, dm.V_Max = -1.0, 1.0
    kwargs = _epoch_kwargs(dm=dm, optimizer=_FakeOptimizer(step=5.0))

    result = scaffold.spid_pib_epoch(**kwargs)

    assert np.all(result.init_v <= 1.0)
    assert np.all(result.init_v >= -1.0)


def test_epoch_keeps_the_old_voltages_when_the_neighbour_check_fails():
    """A rejected update must not move the mirror."""
    dm = _FakeDM(safe=False)
    kwargs = _epoch_kwargs(dm=dm, optimizer=_FakeOptimizer(step=0.5))

    result = scaffold.spid_pib_epoch(**kwargs)

    assert result.applied is False
    np.testing.assert_array_equal(result.init_v, np.zeros(4, dtype=np.float64))


def test_epoch_aborts_before_applying_anything_when_the_window_closes():
    """Closing the live view stops the loop without a half-applied step."""
    dm = _FakeDM()
    kwargs = _epoch_kwargs(dm=dm, on_render=lambda img, volts, epoch: False)

    result = scaffold.spid_pib_epoch(**kwargs)

    assert result.aborted is True
    assert result.applied is False
    np.testing.assert_array_equal(result.init_v, np.zeros(4, dtype=np.float64))
    assert len(dm.written) == 2


def test_epoch_reports_the_brightness_the_saturation_hook_returns():
    """The brightness statistic is the caller's policy, not the scaffold's."""
    kwargs = _epoch_kwargs(on_saturation=lambda pos, neg: 42.0)

    result = scaffold.spid_pib_epoch(**kwargs)

    assert result.brightness == 42.0


def test_epoch_scales_the_logged_difference_by_the_callers_sign():
    """``diff`` is a log column; its sign is the caller's convention."""
    kwargs = _epoch_kwargs(evaluate=lambda img: (3.0, 0.5), diff_sign=-1.0)

    result = scaffold.spid_pib_epoch(**kwargs)

    assert result.diff == pytest.approx((3.0 - 3.0) * -1.0)
    kwargs = _epoch_kwargs(evaluate=lambda img: (3.0, 0.5), diff_sign=1.0)
    result = scaffold.spid_pib_epoch(**kwargs)
    assert result.diff == pytest.approx(0.0)


# ------------------------------------------------------- single-home guard --
def test_the_two_paths_have_no_copy_of_the_epoch_mechanics():
    """The scaffold is the only home of the paired-measurement mechanics."""
    import ao_shaping.optimizer.combined_optimizer as combined
    import ao_shaping.optimizer.wfless.pib as pib

    for module in (pib, combined):
        src = module.__file__
        text = open(src, encoding="utf-8").read()
        assert "spid_pib_epoch" in text, f"{src} does not use the shared scaffold"
        assert "spgd_gradient" not in text, (
            f"{src} computes the SPGD sign itself; the scaffold owns it"
        )
        assert "send_voltages(init_v + disturb_v)" not in text, (
            f"{src} re-implements the paired measurement; the scaffold owns it"
        )
