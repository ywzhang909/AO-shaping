"""End-to-end epoch-loop tests for :func:`optimize_slm_zernike_pib`.

Why this file exists
--------------------
``_captures`` entries are 4-tuples ``(sign, img, coeffs, phase)``. The shipped
line ``_pos_c, pos_img, pos_phase = _pos_captures[0][2:]`` sliced a **2**-tuple
and unpacked it into **3** names, so ``ValueError: not enough values to unpack
(expected 3, got 2)`` was raised on the **first epoch of every run**, in both
default and ABBA mode. The whole suite stayed green because nothing executed the
epoch loop -- 412 unit tests covered the helpers, never the driver.

That makes this module the regression gate: here the *real* optimizer runs its
*real* loop against hermetic fakes, so a one-line source regression cannot ship
again. Note the fix also feeds ``best_c = _pos_c.copy()`` and
``_apply_best_on_exit`` -- the restore path asserted by
:func:`test_loop_uses_first_positive_frame_for_best_restore`.

No hardware is involved. There is **no** ``sim`` camera in the registry (only
``daheng`` / ``miicam`` / ``ffmpeg`` / ``image_folder``), so ``create_camera``
and ``Santec.from_params`` are monkeypatched on the ``slm_zernike_pib`` module,
exactly where the optimizer looks them up.

Measured ``get_numpy_image`` call counts with this harness (``cam_size=64``,
``center="max"``, fixed exposure, 2 epochs, fake SLM)::

    mode      epochs  captures   phases   slots
    default       1          8        4       6
    default       2         10        6       8
    default       3         12        8      10
    abba          1         10        6       8
    abba          2         14       10      12
    abba          3         18       14      16

    per-epoch captures: default (2-1)=2 and (3-2)=2; abba (2-1)=4 and (3-2)=4
    fixed cost        : total(3) - 3 * per_epoch = 6 in BOTH modes

The constant 6 splits into **3 one-time setup acquisitions** (initial frame ->
``resolve_spot_center`` re-capture for ``center="max"`` -> re-windowed
``init_img``) and **3 post-loop acquisitions** in ``_apply_best_on_exit`` (the
full-frame probe plus the raw before/after pair), which only runs because the
fake run actually improves. Because that tail is mode-independent, the
difference ``epochs=2 minus epochs=1`` cleanly isolates the per-epoch cost --
that is what the two counting tests assert.
"""

from __future__ import annotations

import time as _time
from typing import Any

import numpy as np
import pytest

from ao_shaping.optimizer.wfless import slm_zernike_pib as pib_mod
from ao_shaping.optimizer.wfless.slm_zernike_pib import (
    SlmZernikePibConfig,
    optimize_slm_zernike_pib,
)
from ao_shaping.runners.runner_common import (
    CameraParamsPib,
    ObjectiveTarget,
    SlmParamsPib,
)

# --- bench geometry -----------------------------------------------------------
FULL_SIDE = 512  # full sensor is FULL_SIDE x FULL_SIDE
WINDOW = 64  # camera_config.cam_size
TARGET_SIZE = 8.0  # objective target extent, camera px
EXPOSURE_MS = 1.0  # > 0 -> fixed exposure, exactly one capture per acquisition
N_MAX = 2  # 6 Zernike modes -> small, fast polynomials
SPOT_SIGMA = 2.6  # flat-phase spot sigma, px
PEAK_FLAT = 180.0  # flat-phase spot peak; kept < 255 so nothing ever saturates
R_BUCKET = 2  # small bucket on purpose: the flat spot LEAKS energy, so a tighter
# spot raises the ``pib`` ratio decisively instead of saturating at 1.0

# Phase -> response coupling. ``phase_spread`` (std of the Zernike phase image)
# is EVEN in the coefficient vector and ``phase_tilt`` (its mean) is ODD, so the
# pair gives both a strictly improving positive frame (tight, brighter spot) and
# a non-zero +/- asymmetry for the SPGD difference. The spread is normalised
# through ``s / (s + 1)`` so the coupling is scale-free: it works whatever
# amplitude ``_zernike_to_phase`` happens to produce, and stays monotone.
SPREAD_SHRINK = 0.60  # sigma *= (1 - SPREAD_SHRINK * spread_norm)
SPREAD_GROW = 0.80  # peak *= (1 + SPREAD_GROW * spread_norm)
TILT_GAIN = 2.0  # tanh() pre-gain applied to the odd term
TILT_GROW = 0.15  # peak *= (1 + TILT_GROW * tanh(tilt))


class _Bench:
    """Shared optical state: what the fake SLM displays is what the camera sees."""

    def __init__(self) -> None:
        self.phase_spread = 0.0
        self.phase_tilt = 0.0
        self.memory_slots: list[int] = []
        self.grayscale_writes: list[int] = []


class _FakeCamera:
    """Only the driver surface the optimizer touches:

    ``get_numpy_image``, ``reset_window`` and a writable ``exposure_time_ms``.
    """

    def __init__(self, bench: _Bench) -> None:
        self.bench = bench
        self.exposure_time_ms = EXPOSURE_MS
        self.captures = 0
        self.frames: list[np.ndarray] = []
        self.windows: list[tuple[tuple[int, int], tuple[int, int]]] = []
        self._shape = (FULL_SIDE, FULL_SIDE)
        self._spot = (FULL_SIDE // 2, FULL_SIDE // 2)

    # -- driver surface --------------------------------------------------
    def get_numpy_image(self, n_sample: int = 1, *args: Any, **kwargs: Any) -> np.ndarray:
        self.captures += 1
        frame = self._render()
        self.frames.append(frame)
        return frame

    def reset_window(
        self, center: tuple[int, int], size: tuple[int, int]
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """Mimic ``clamp_center_to_frame`` + ``reset_window``: return the window
        size and the **full-frame** centre (the optimizer re-locates the 0-order
        on the re-windowed image)."""
        cx, cy = int(center[0]), int(center[1])
        w, h = int(size[0]), int(size[1])
        self._shape = (h, w)
        self.windows.append(((w, h), (cx, cy)))
        return (w, h), (cx, cy)

    # -- context manager --------------------------------------------------
    def __enter__(self) -> _FakeCamera:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    # -- rendering --------------------------------------------------------
    def _render(self) -> np.ndarray:
        h, w = self._shape
        raw_spread = self.bench.phase_spread
        spread = raw_spread / (raw_spread + 1.0)  # scale-free, always in [0, 1)
        tilt = float(np.tanh(self.bench.phase_tilt * TILT_GAIN))
        sigma = max(SPOT_SIGMA * (1.0 - SPREAD_SHRINK * spread), 0.35)
        peak = min(PEAK_FLAT * (1.0 + SPREAD_GROW * spread + TILT_GROW * tilt), 250.0)

        yy = np.arange(h, dtype=np.float64)[:, None]
        xx = np.arange(w, dtype=np.float64)[None, :]
        spot_y, spot_x = self._spot
        if self.windows:
            # the 0-order sits at the full-frame centre, so its window-local
            # position follows whichever window was requested last
            full_x, full_y = self.windows[-1][1]
            spot_y = spot_y - (full_y - h // 2)
            spot_x = spot_x - (full_x - w // 2)

        frame = peak * np.exp(
            -(((yy - spot_y) ** 2 + (xx - spot_x) ** 2) / (2.0 * sigma**2))
        )
        return frame.astype(np.uint8)


class _FakeSlm:
    """Only the surface the optimizer touches: ``create_phase_from_array``,
    ``display_data`` (rotating slot) and ``set_grayscale`` (flat fallback)."""

    MAX_GRAY = 1023

    def __init__(self, bench: _Bench) -> None:
        self.bench = bench
        self.phases: list[np.ndarray] = []

    def create_phase_from_array(self, phase: Any, *args: Any, **kwargs: Any) -> np.ndarray:
        arr = np.asarray(phase, dtype=np.float64)
        finite = np.isfinite(arr)
        if finite.any():
            self.bench.phase_spread = float(np.std(arr[finite]))
            self.bench.phase_tilt = float(np.mean(arr[finite]))
        else:
            self.bench.phase_spread = 0.0
            self.bench.phase_tilt = 0.0
        self.phases.append(arr)
        gray = np.mod(arr, 2.0 * np.pi) / (2.0 * np.pi) * self.MAX_GRAY
        return np.clip(gray, 0.0, self.MAX_GRAY).astype(np.uint16)

    def display_data(
        self,
        gray: np.ndarray,
        memory_number: int | None = None,
        memory_mode: int | None = None,
    ) -> None:
        self.bench.memory_slots.append(int(memory_number))

    def set_grayscale(self, gray: int) -> None:
        self.bench.grayscale_writes.append(int(gray))

    # -- context manager --------------------------------------------------
    def __enter__(self) -> _FakeSlm:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


class _FakeTime:
    """``time`` stand-in: records ``sleep`` calls, forwards everything else."""

    def __init__(self) -> None:
        self.slept: list[float] = []

    def sleep(self, seconds: float) -> None:
        self.slept.append(float(seconds))

    def __getattr__(self, name: str) -> Any:
        return getattr(_time, name)


class _Harness:
    def __init__(
        self,
        bench: _Bench,
        camera: _FakeCamera,
        slm: _FakeSlm,
        fake_time: _FakeTime,
    ) -> None:
        self.bench = bench
        self.camera = camera
        self.slm = slm
        self.fake_time = fake_time


def _install_fakes(monkeypatch: pytest.MonkeyPatch) -> _Harness:
    bench = _Bench()
    camera = _FakeCamera(bench)
    slm = _FakeSlm(bench)
    fake_time = _FakeTime()

    class _StubSantec:
        @staticmethod
        def from_params(params: SlmParamsPib) -> _FakeSlm:
            return slm

    monkeypatch.setattr(pib_mod, "create_camera", lambda params: camera)
    monkeypatch.setattr(pib_mod, "Santec", _StubSantec)
    monkeypatch.setattr(pib_mod, "time", fake_time)
    monkeypatch.setattr(pib_mod, "SLM_RESPONSE_TIME_S", 0.0)
    # module-level slot rotation is shared state -> reset for isolation
    monkeypatch.setitem(pib_mod._SLOT_STATE, "slot", pib_mod._SLOT_MIN - 1)
    return _Harness(bench, camera, slm, fake_time)


def _config(epochs: int, *, abba: bool) -> SlmZernikePibConfig:
    return SlmZernikePibConfig(
        center="max",
        epochs=epochs,
        algorithm="spgd",
        optimizer_type="adamod",
        delta=0.2,
        lr=0.5,
        random_seed=7,
        abba_sampling=abba,
        camera=CameraParamsPib(
            cam_size=WINDOW,
            exposure_time_ms=EXPOSURE_MS,
            target_max_brightness=40,
            r_bucket=R_BUCKET,
            target_size=TARGET_SIZE,
            # plain ``pib`` with NO subordinate target_shape: passing one would
            # upgrade the objective to ``shape`` (a minimised radius) via
            # ``ObjectiveSpec.resolve``.
            target=ObjectiveTarget(name="pib"),
            max_roi_energy_loss=1.0,  # 1.0 disables the ROI energy-loss guard
            auto_exposure=False,
        ),
        slm=SlmParamsPib(n_max=N_MAX, zernike_radius=300.0),
        record_phase=False,
    )


def _run(monkeypatch: pytest.MonkeyPatch, epochs: int, *, abba: bool = False) -> _Harness:
    harness = _install_fakes(monkeypatch)
    harness.recorder = optimize_slm_zernike_pib(_config(epochs, abba=abba))
    return harness


def _epoch_phases(slm: _FakeSlm, epoch: int, n_signs: int) -> list[np.ndarray]:
    """Phase images displayed by epoch ``epoch`` (1-based).

    Slot 0 of the display log is the initial (flat) phase, then each epoch
    contributes exactly ``n_signs`` phase writes.
    """
    start = 1 + (epoch - 1) * n_signs
    return slm.phases[start : start + n_signs]


def _decode_signs(epoch_phases: list[np.ndarray]) -> list[int]:
    """Recover the SPGD sign actually displayed for each frame of one epoch.

    Within an epoch the perturbation direction is constant, so each frame is
    ``base +/- delta * d``. De-meaning the stack makes the per-frame residual
    proportional to the signed direction: the first frame is the ``+`` one, and
    the sign of its inner product with the others recovers ``+`` / ``-``.
    """
    stack = np.stack([p.ravel() for p in epoch_phases])
    centred = stack - stack.mean(axis=0)
    reference = centred[0]
    return [1 if float(np.dot(row, reference)) >= 0.0 else -1 for row in centred]


def test_default_path_runs_and_captures_twice_per_epoch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default sampling: two captures per epoch, both epochs recorded."""
    one = _run(monkeypatch, 1)
    two = _run(monkeypatch, 2)

    # setup acquisitions are a fixed constant, so the DIFFERENCE isolates the
    # per-epoch cost: default -> 2 captures
    per_epoch = two.camera.captures - one.camera.captures
    assert per_epoch == 2, (
        f"expected 2 captures/epoch (default), got {per_epoch} "
        f"(epochs=1 -> {one.camera.captures}, epochs=2 -> {two.camera.captures})"
    )
    # and there IS a non-trivial setup to isolate from
    assert one.camera.captures >= 3

    # one phase write per capture, on top of the initial (flat) phase
    assert len(two.slm.phases) - len(one.slm.phases) == 2

    assert [row["_epoch"] for row in two.recorder.history] == [0, 1, 2]
    assert all(np.isfinite(frame).all() for frame in two.camera.frames)


def test_abba_path_runs_and_captures_four_times_per_epoch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ABBA sampling: four captures per epoch, both epochs recorded."""
    one = _run(monkeypatch, 1, abba=True)
    two = _run(monkeypatch, 2, abba=True)

    per_epoch = two.camera.captures - one.camera.captures
    assert per_epoch == 4, (
        f"expected 4 captures/epoch (abba), got {per_epoch} "
        f"(epochs=1 -> {one.camera.captures}, epochs=2 -> {two.camera.captures})"
    )
    assert one.camera.captures >= 3

    assert len(two.slm.phases) - len(one.slm.phases) == 4

    assert [row["_epoch"] for row in two.recorder.history] == [0, 1, 2]
    assert two.camera.captures == one.camera.captures + 4


def test_abba_produces_two_positive_and_two_negative_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ABBA really alternates: the signs written to the SLM are ``+ - - +``."""
    assert pib_mod._spgd_capture_signs(False) == (1, -1)
    assert pib_mod._spgd_capture_signs(True) == (1, -1, -1, 1)

    harness = _run(monkeypatch, 2, abba=True)
    for epoch in (1, 2):
        signs = _decode_signs(_epoch_phases(harness.slm, epoch, 4))
        assert signs == [1, -1, -1, 1], f"epoch {epoch} signs were {signs}"
        assert signs.count(1) == 2
        assert signs.count(-1) == 2


def test_loop_uses_first_positive_frame_for_best_restore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression guard for the ``_pos_c`` binding.

    ``best_c = _pos_c.copy()`` only runs when an epoch *improves*, and the
    restore then writes the best phase back. If the fake run never improved,
    ``set_grayscale(0)`` (the flat fallback) would show up instead.
    """
    harness = _run(monkeypatch, 2, abba=True)

    # improved -> the real best-phase restore, never the flat fallback
    # (``_apply_best_on_exit`` falls back to ``set_grayscale(0)`` otherwise).
    assert harness.bench.grayscale_writes == []
    # the restore wrote phase(s) beyond initial + the 4x2 in-loop captures
    assert len(harness.bench.memory_slots) > 1 + 4 * 2
    # slots rotate and never repeat consecutively (firmware no-op otherwise)
    assert len(set(harness.bench.memory_slots)) == len(harness.bench.memory_slots)
    assert harness.bench.memory_slots == sorted(harness.bench.memory_slots)

    assert len(harness.camera.frames) == harness.camera.captures
    assert [row["_epoch"] for row in harness.recorder.history] == [0, 1, 2]
    assert all("_c" in row for row in harness.recorder.history)
    # the tracked best is monotonically non-decreasing across recorded epochs
    best = [row["best_pib"] for row in harness.recorder.history]
    assert best == sorted(best), f"best_pib was not monotone: {best}"
    assert best[-1] > best[0], f"the run never improved over flat: {best}"