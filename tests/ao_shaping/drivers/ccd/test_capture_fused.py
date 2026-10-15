"""Offline anchors for :meth:`BaseCamera.capture_fused`.

The bracketed multi-exposure capture lives on the **base** class, so every
backend inherits it with no per-driver code; these tests therefore exercise it
through a small purpose-built fake rather than through Daheng/MiiCam, whose
capture paths need real hardware (and, for MiiCam, an SDK).

Three things are pinned here, and each one is a trap the implementation had to
route around rather than a property it inherited:

1. **Every capture passes ``skip_first=True``.** The first frame after an
   exposure change can be stale -- ``reset_exposure_time``'s own docstring
   records that MiiCam needs a full Stop -> set -> Start cycle -- so the method
   owns that decision and does not expose it.
2. **The caller's exposure is restored in a ``finally``.** A bracket that dies
   half-way (one exposure read raises) must still leave the camera where the
   caller found it, or the next unrelated measurement silently inherits a
   bracket exposure.
3. **The returned ``AveragedFrame`` needs an *integer* source dtype.** Its
   ``__new__`` does ``float(np.iinfo(source_dtype).max)``, and
   ``get_numpy_image(n_sample > 1)`` hands back a float32 ``AveragedFrame``, so
   the naive "pass the frame's own dtype" raises ``ValueError``.

The synthetic scene's core carries a high-frequency texture that a long exposure
clips to a flat plateau: that is what makes "the fusion recovered detail the long
frame lost" an assertion rather than a vibe.

Everything is pure NumPy, hardware-free, and runs in well under a second.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from loguru import logger

from ao_shaping.drivers.ccd.base import AveragedFrame, BaseCamera
from ao_shaping.utils.image import exposure_fusion as _fusion_module

#: Exposure ladder used by most tests. The ratio is 10x, so the long frame's
#: core is clipped hard while the short frame still resolves it.
LADDER = [0.12, 1.2]

#: Counts the scene's brightest pixel produces at ``LADDER[0]``. Chosen so the
#: short frame sits at ~60/255 (dark but well above the noise floor) while
#: ``LADDER[1]`` would reach 600 and clip.
_SHORT_PEAK_COUNTS = 60.0


def _scene(shape: tuple[int, int] = (48, 48)) -> np.ndarray:
    """A blob whose core carries a resolvable high-frequency texture.

    Normalised to a peak of 1.0 so the fake camera's counts read directly as
    ``scene * counts_per_unit_ms * exposure_ms``.
    """
    rows, cols = np.mgrid[0 : shape[0], 0 : shape[1]]
    radius2 = (rows - shape[0] / 2.0) ** 2 + (cols - shape[1] / 2.0) ** 2
    core = np.exp(-radius2 / (2.0 * (shape[0] / 6.0) ** 2))
    scene = core * (1.0 + 0.35 * np.cos(rows * np.pi / 2.0))
    return scene / float(scene.max())


def _core_mask(shape: tuple[int, int] = (48, 48), radius: float = 4.0) -> np.ndarray:
    """Boolean mask of the region that saturates at the long ladder exposure.

    4 px is measured, not guessed: the scene's texture modulates the blob by
    +-35%, so the saturated plateau stops around r ~ 4.5 and a wider mask would
    reach pixels that are merely bright rather than clipped.
    """
    rows, cols = np.mgrid[0 : shape[0], 0 : shape[1]]
    radius2 = (rows - shape[0] / 2.0) ** 2 + (cols - shape[1] / 2.0) ** 2
    return radius2 <= radius * radius


def _relative_error(out: np.ndarray, reference: np.ndarray) -> float:
    """``mean |out - ref|`` normalised by ``mean |ref|``."""
    scale = float(np.mean(np.abs(reference)))
    if scale <= 0.0:
        return float("inf")
    return float(np.mean(np.abs(out - reference))) / scale


class _FakeCamera(BaseCamera):
    """An offline ``BaseCamera`` -- no SDK, no device, no files.

    Counts follow ``clip(scene * counts_per_unit_ms * exposure_ms, 0, full_scale)``,
    i.e. brightness is **linear in exposure** exactly as a real detector's is, and
    the longer exposure clips the textured core. Every exposure write and every
    capture is recorded so a test can assert the exact call sequence.

    ``self.cam`` is deliberately never assigned, mirroring ``FFmpegCamera`` and
    ``SimPibCCD``: if ``capture_fused`` had grown a Daheng-style ``assert self.cam``
    guard, constructing this fake would fail and two real backends would break.
    """

    def __init__(
        self,
        *,
        scene: np.ndarray | None = None,
        dtype: Any = np.uint8,
        counts_per_unit_ms: float = _SHORT_PEAK_COUNTS / LADDER[0],
        initial_exposure_ms: float = 7.0,
        expose_exposure: bool = True,
        fail_on_shot: int | None = None,
        frame_full_scale: float | None = None,
    ):
        super().__init__(cam_id=0, exposure_time_ms=initial_exposure_ms)
        self.scene = (
            _scene() if scene is None else np.asarray(scene, dtype=np.float64)
        )
        self.dtype = np.dtype(dtype)
        self.counts_per_unit_ms = float(counts_per_unit_ms)
        self.fail_on_shot = fail_on_shot
        self.frame_full_scale = frame_full_scale
        self.initial_exposure_ms = float(initial_exposure_ms)
        #: exposure writes, in order (each ladder step plus the restore).
        self.exposure_writes: list[float] = []
        #: ``(exposure_ms, n_sample, skip_first)`` per capture, in order.
        self.shots: list[tuple[float, int, bool]] = []
        self.captured: list[np.ndarray] = []
        if not expose_exposure:
            # Neither ``exposure_time_ms`` nor ``exposure_time`` reads back as a
            # number -- exactly what ``common.get_camera_exposure_ms`` reports by
            # raising AttributeError.
            del self.exposure_time_ms
            self.exposure_time = None

    # -- the capture path under test ---------------------------------------
    def reset_exposure_time(self, time_ms: float) -> float:
        self.exposure_writes.append(float(time_ms))
        self.exposure_time_ms = float(time_ms)
        return float(time_ms)

    def get_numpy_image(
        self,
        n_sample: int = 1,
        skip_first: bool = True,
    ) -> np.ndarray:
        assert n_sample > 0
        self.shots.append(
            (float(self.exposure_time_ms), int(n_sample), bool(skip_first))
        )
        if self.fail_on_shot is not None and len(self.shots) - 1 == self.fail_on_shot:
            raise RuntimeError("simulated capture failure")
        counts = self._counts()
        if self.frame_full_scale is not None:
            frame = AveragedFrame(counts, self.dtype)
            frame.full_scale = float(self.frame_full_scale)
        elif n_sample > 1:
            frame = AveragedFrame(counts, self.dtype)
        else:
            frame = counts
        self.captured.append(np.asarray(frame))
        return frame

    def _counts(self) -> np.ndarray:
        ceiling = float(np.iinfo(self.dtype).max)
        raw = self.scene * self.counts_per_unit_ms * float(self.exposure_time_ms)
        return np.clip(raw, 0.0, ceiling).astype(self.dtype)

    # -- the rest of the BaseCamera surface, inert --------------------------
    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        pass

    def initialize(self) -> None:
        pass

    def open(self) -> None:
        pass

    def close(self) -> None:
        pass

    def reset_window(self, center: Any, size: Any) -> Any:
        return (tuple(size), tuple(center))

    def enable_auto_exposure(self, enable: bool = True, mode: int = 1) -> bool:
        return True

    def get_auto_exposure_state(self) -> dict:
        return {"enabled": False, "mode": 0, "target": 0}

    def set_auto_exposure_range(
        self,
        max_time_ms: int = 350,
        min_time_ms: int = 0,
        max_gain: int = 300,
        min_gain: int = 100,
    ) -> bool:
        return True

    @staticmethod
    def get_cam_list() -> list:
        return []


def _capture_warnings() -> tuple[int, list[str]]:
    """Register a loguru sink; returns ``(sink_id, messages)``.

    ``caplog`` only sees the stdlib ``logging`` module and loguru does not
    propagate there by default, so a loguru warning is invisible to it.
    """
    messages: list[str] = []
    return logger.add(lambda m: messages.append(str(m)), level="WARNING"), messages


class TestCaptureSequence:
    def test_one_frame_per_exposure_in_order_and_skip_first_always(self) -> None:
        """Pin trap A: the stale-first-frame rule is the method's own, not the
        caller's."""
        cam = _FakeCamera()
        cam.capture_fused(LADDER)

        assert cam.shots == [(0.12, 1, True), (1.2, 1, True)]
        # exposure writes: one per ladder step, then the restore.
        assert cam.exposure_writes == [0.12, 1.2, cam.initial_exposure_ms]
        assert len(cam.captured) == 2

    def test_n_sample_is_forwarded_unchanged(self) -> None:
        cam = _FakeCamera()
        cam.capture_fused(LADDER, n_sample=4)
        assert [shot[1] for shot in cam.shots] == [4, 4]


class TestFusion:
    def test_fused_recovers_detail_the_long_exposure_clipped(self) -> None:
        """Pin the reason the ladder exists at all.

        At 1.2 ms the textured core clips to a flat 255 plateau; at 0.12 ms it is
        still ~60 counts with its structure intact. The fusion must come back
        below saturation *and* non-flat inside that core.
        """
        cam = _FakeCamera()
        fused = cam.capture_fused(LADDER)

        short_frame, long_frame = (np.asarray(f) for f in cam.captured)
        core = _core_mask()

        # The control: the long frame really did destroy the structure.
        assert float(long_frame[core].min()) == 255.0
        assert float(long_frame[core].max()) == 255.0
        # ...while the short frame kept it.
        assert float(short_frame[core].max() - short_frame[core].min()) > 5.0

        assert fused.dtype == np.float32
        assert float(fused.max()) < 0.9 * 255.0
        assert float(fused[core].max() - fused[core].min()) > 1.0
        assert float(fused[core].mean()) < 0.5 * float(long_frame[core].mean())

    def test_the_real_exposure_list_reaches_the_fusion(self) -> None:
        """Pin the Stage-0 contract at the driver boundary.

        Detector counts scale linearly with exposure, so the long frame of a
        bracket is systematically brighter; without ``exposures_ms`` the
        well-exposedness term hands it the whole bracket and the "fusion" *is*
        that frame. Here the long frame peaks at 240/255 (kept under saturation
        so the comparison is about weighting, not about clipping), and the mean
        weight ratio lands at ~62x -- the regime the fusion module pins
        separately. ``capture_fused`` must therefore be visibly on the other
        side of it.
        """
        from ao_shaping.utils.image.exposure_fusion import (
            compute_fusion_weights,
            exposure_fusion,
        )

        cam = _FakeCamera(counts_per_unit_ms=200.0)
        fused = cam.capture_fused(LADDER)
        short_frame, long_frame = (np.asarray(frame) for frame in cam.captured)
        assert long_frame.max() == 240  # unclipped, so this is purely a weighting question

        weights = np.stack([compute_fusion_weights(f) for f in (short_frame, long_frame)])
        assert float(weights[1].mean() / weights[0].mean()) > 10.0

        # Dropping the bracket metadata degenerates to the long frame...
        without_exposures = exposure_fusion([short_frame, long_frame])[
            "laplacian_pyramid"
        ]
        assert _relative_error(without_exposures, long_frame) < 0.2
        # ...while the real ladder does not.
        assert _relative_error(fused, long_frame) > 0.5


class TestExposureRestore:
    def test_restores_the_callers_exposure_on_the_normal_path(self) -> None:
        cam = _FakeCamera(initial_exposure_ms=3.5)
        cam.capture_fused(LADDER)
        assert cam.exposure_time_ms == 3.5
        assert cam.exposure_writes[-1] == 3.5

    def test_restores_even_when_capture_raises_mid_bracket(self) -> None:
        """Pin trap B: a plain ``if`` after the loop would skip the restore."""
        cam = _FakeCamera(initial_exposure_ms=3.5, fail_on_shot=1)
        with pytest.raises(RuntimeError, match="simulated capture failure"):
            cam.capture_fused(LADDER)

        assert cam.exposure_time_ms == 3.5
        assert cam.exposure_writes == [0.12, 1.2, 3.5]

    def test_restore_false_leaves_the_camera_on_the_last_ladder_exposure(self) -> None:
        cam = _FakeCamera(initial_exposure_ms=3.5)
        cam.capture_fused(LADDER, restore_exposure=False)
        assert cam.exposure_time_ms == LADDER[-1]
        assert cam.exposure_writes == LADDER

    def test_restore_false_is_honoured_on_the_error_path_too(self) -> None:
        cam = _FakeCamera(initial_exposure_ms=3.5, fail_on_shot=1)
        with pytest.raises(RuntimeError):
            cam.capture_fused(LADDER, restore_exposure=False)
        assert cam.exposure_time_ms == LADDER[-1]
        assert cam.exposure_writes == LADDER

    def test_backend_without_a_readable_exposure_still_fuses(self) -> None:
        """Pin trap G: ``AttributeError`` from the read is a warning, not a
        failure -- the bracket runs and the camera simply stays where it ended."""
        sink_id, messages = _capture_warnings()
        try:
            cam = _FakeCamera(expose_exposure=False)
            fused = cam.capture_fused(LADDER)
        finally:
            logger.remove(sink_id)

        assert isinstance(fused, AveragedFrame)
        assert cam.exposure_writes == LADDER
        assert cam.exposure_time_ms == LADDER[-1]
        assert any("no readable exposure time" in message for message in messages)


class TestValidation:
    def test_a_single_exposure_is_rejected(self) -> None:
        cam = _FakeCamera()
        with pytest.raises(ValueError, match=r"at least two exposures, got \[0.12\]"):
            cam.capture_fused([0.12])
        assert cam.exposure_writes == []

    def test_an_empty_bracket_is_rejected(self) -> None:
        cam = _FakeCamera()
        with pytest.raises(ValueError, match=r"got \[\]"):
            cam.capture_fused([])

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_exposure_is_rejected(self, bad: float) -> None:
        cam = _FakeCamera()
        with pytest.raises(ValueError, match=r"exposures_ms\[1\].*must be a finite"):
            cam.capture_fused([0.12, bad])
        assert cam.exposure_writes == []

    @pytest.mark.parametrize("bad", [0.0, -0.5])
    def test_non_positive_exposure_is_rejected(self, bad: float) -> None:
        cam = _FakeCamera()
        with pytest.raises(ValueError, match=r"exposures_ms\[1\].*must be a finite"):
            cam.capture_fused([0.12, bad])
        assert cam.exposure_writes == []

    @pytest.mark.parametrize("bad", [0, -1])
    def test_non_positive_sample_count_is_rejected(self, bad: int) -> None:
        """``assert`` is stripped under ``python -O``, so this must raise."""
        cam = _FakeCamera()
        with pytest.raises(ValueError, match=r"n_sample must be >= 1, got 0|-1"):
            cam.capture_fused(LADDER, n_sample=bad)
        assert cam.exposure_writes == []


class TestResultContract:
    @pytest.mark.parametrize(
        "dtype,expected", [(np.uint8, 255.0), (np.uint16, 65535.0)]
    )
    def test_full_scale_follows_the_detector_bit_depth(
        self, dtype: Any, expected: float
    ) -> None:
        """Pin trap E from the outside: ``AveragedFrame.full_scale`` must report
        the real saturation level, not a float dtype's nonexistent one."""
        result = _FakeCamera(dtype=dtype).capture_fused(LADDER)
        assert isinstance(result, AveragedFrame)
        assert result.full_scale == expected

    def test_a_non_standard_saturation_level_survives(self) -> None:
        """A 14-bit detector (16383) has no matching std dtype, so the value has
        to be carried explicitly rather than inherited from ``np.iinfo``."""
        cam = _FakeCamera(frame_full_scale=16383.0)
        result = cam.capture_fused(LADDER)
        assert result.full_scale == 16383.0
        assert result.full_scale != float(np.iinfo(np.uint16).max)

    def test_averaged_frame_cannot_be_built_from_a_float_dtype(self) -> None:
        """The trap itself, so the reason for the dtype juggling stays visible:
        ``np.iinfo(np.float32)`` raises, which is what ``capture_fused`` would
        hit if it forwarded ``get_numpy_image(n_sample > 1)``'s dtype."""
        with pytest.raises(ValueError):
            AveragedFrame(np.zeros((4, 4)), np.dtype(np.float32))

    def test_works_on_a_backend_that_returns_averaged_frames(self) -> None:
        """``n_sample > 1`` makes ``get_numpy_image`` hand back a float32
        ``AveragedFrame`` -- the path that trips the dtype trap."""
        cam = _FakeCamera()
        result = cam.capture_fused(LADDER, n_sample=3)

        assert isinstance(result, AveragedFrame)
        assert result.dtype == np.float32
        assert result.full_scale == 255.0
        assert all(isinstance(np.asarray(f), np.ndarray) for f in cam.captured)

    def test_result_is_finite_two_dimensional_and_in_range(self) -> None:
        result = _FakeCamera().capture_fused(LADDER)

        assert result.ndim == 2
        assert result.shape == (48, 48)
        assert result.dtype == np.float32
        assert bool(np.isfinite(result).all())
        assert float(result.min()) >= 0.0
        assert float(result.max()) <= result.full_scale


class _ClampingCamera(_FakeCamera):
    """A backend that clamps, the way ``DahengCamera`` does.

    ``DahengCamera.reset_exposure_time`` clips the request into the device's
    live ``ExposureTime`` range, warns, writes the clipped value and returns
    it. ``common.set_camera_exposure_ms`` then **throws that return value away**
    and reports the request instead, so the clamped value is only observable
    by reading the camera back.
    """

    def __init__(self, *, min_ms: float, max_ms: float = 10_000.0, **kwargs: Any):
        super().__init__(**kwargs)
        self.min_ms = float(min_ms)
        self.max_ms = float(max_ms)

    def reset_exposure_time(self, time_ms: float) -> float:
        effective = float(min(max(float(time_ms), self.min_ms), self.max_ms))
        self.exposure_writes.append(float(time_ms))
        self.exposure_time_ms = effective
        return effective


class TestExposureClamping:
    """Stage 0 must normalise by the exposure the hardware *actually* used.

    ``set_camera_exposure_ms`` returns the request even when the backend
    clamped it, so passing the caller's ladder straight through to
    ``exposure_fusion`` silently mis-weights every clamped frame by the clamp
    ratio. With the device floor at 0.011 ms, asking for 0.005 ms yields a
    2.2x error in that frame's contribution and no warning at all.
    """

    @staticmethod
    def _spy(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
        """Capture what ``exposure_fusion`` was actually handed.

        ``capture_fused`` imports the symbol inside the method body, so
        patching the module attribute is picked up at call time.
        """
        seen: dict[str, Any] = {}
        real = _fusion_module.exposure_fusion

        def spy(sequence, **kwargs):
            seen["exposures_ms"] = list(kwargs["exposures_ms"])
            return real(sequence, **kwargs)

        monkeypatch.setattr(_fusion_module, "exposure_fusion", spy)
        return seen

    def test_fusion_normalises_by_the_clamped_exposure_not_the_request(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen = self._spy(monkeypatch)
        cam = _ClampingCamera(min_ms=0.011)

        cam.capture_fused([0.005, 0.5])

        assert cam.exposure_writes == [0.005, 0.5, cam.initial_exposure_ms]
        assert seen["exposures_ms"] == [0.011, 0.5]

    def test_unchanged_exposures_pass_through_untouched(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen = self._spy(monkeypatch)

        _ClampingCamera(min_ms=0.001).capture_fused(LADDER)

        assert seen["exposures_ms"] == LADDER

    def test_an_upper_clamp_is_read_back_too(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen = self._spy(monkeypatch)

        _ClampingCamera(min_ms=0.001, max_ms=50.0).capture_fused([2.0, 500.0])

        assert seen["exposures_ms"] == [2.0, 50.0]

    def test_a_backend_without_a_readable_exposure_falls_back_to_the_request(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No readback is possible, so the request is the best available answer
        -- and it must not raise."""
        seen = self._spy(monkeypatch)
        cam = _FakeCamera(expose_exposure=False)

        result = cam.capture_fused(LADDER)

        assert seen["exposures_ms"] == LADDER
        assert bool(np.isfinite(result).all())

    def test_clamping_emits_a_warning(self) -> None:
        sink, messages = _capture_warnings()
        try:
            _ClampingCamera(min_ms=0.011).capture_fused([0.005, 0.5])
        finally:
            logger.remove(sink)

        assert any("clamped to" in message for message in messages)