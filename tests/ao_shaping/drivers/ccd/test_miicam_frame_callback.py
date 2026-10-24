"""MIICAM SDK frame callback: a lost frame must be counted and logged, not swallowed.

Before this, both handlers in ``MIICamera._frame_callback`` were
``except Exception: pass``. Raising there is not an option (it would unwind through
the vendor SDK's own thread), but silence meant a dropped frame surfaced only as a
timeout in ``get_callback_frame`` with nothing to attribute it to -- the same class of
failure as the metrics-reading-zero incidents recorded in TODO.md 8.9.

These tests run without the MIICAM SDK: the module binds ``miicam = None`` when the
import fails, and ``_frame_callback`` reads ``miicam.MIICAM_EVENT_IMAGE`` before its
try block, so the stub below supplies the constants the callback needs.
"""

from __future__ import annotations

import pytest
from loguru import logger

from ao_shaping.drivers.ccd.miicam import driver as drv


class _EventConstants:
    """Stand-in for the ``miicam`` module surface used by ``_frame_callback``."""

    MIICAM_EVENT_IMAGE = 1
    MIICAM_EVENT_STILLIMAGE = 2

    class MiicamFrameInfoV3:
        def __init__(self) -> None:
            self.width = 2
            self.height = 2


@pytest.fixture()
def stub_sdk(monkeypatch: pytest.MonkeyPatch) -> type[_EventConstants]:
    monkeypatch.setattr(drv, "miicam", _EventConstants)
    return _EventConstants


@pytest.fixture()
def cam(stub_sdk) -> drv.MIICamera:
    camera = drv.MIICamera(cam_id=0, exposure_time_ms=1.0, capture_mode="callback")
    # _get_buffer_params/_decode_image work off the frame geometry, normally set by
    # open(); give them a real 2x2 frame so the callback reaches the pull step.
    camera.cam_width = 2
    camera.cam_height = 2
    return camera


class _FailingPuller:
    def __init__(self) -> None:
        self.calls = 0

    def pull_image_v4(self, buffer, bits, frame_info) -> None:
        self.calls += 1
        raise RuntimeError("boom")

    def pull_still_image(self, buffer, bits, frame_info) -> None:
        self.calls += 1
        raise RuntimeError("boom")


class _WorkingPuller:
    def pull_image_v4(self, buffer, bits, frame_info) -> None:
        return None

    def pull_still_image(self, buffer, bits, frame_info) -> None:
        return None


def _warning_sink() -> tuple[int, list[str]]:
    """loguru does not propagate to stdlib logging, so caplog cannot see it."""
    messages: list[str] = []
    return logger.add(lambda m: messages.append(str(m)), level="WARNING"), messages


class TestFrameLossIsObservable:
    def test_failing_pull_does_not_raise_out_of_the_callback(self, cam) -> None:
        """The SDK thread must not unwind -- that is why the handler exists at all."""
        cam._frame_puller = _FailingPuller()
        cam._frame_callback(_EventConstants.MIICAM_EVENT_IMAGE, None)

    def test_failure_is_counted_and_logged_with_the_cause(self, cam) -> None:
        cam._frame_puller = _FailingPuller()
        sink_id, messages = _warning_sink()
        try:
            cam._frame_callback(_EventConstants.MIICAM_EVENT_IMAGE, None)
        finally:
            logger.remove(sink_id)

        assert cam._callback_error_count == 1
        assert len(messages) == 1, "the first loss must be visible at WARNING"
        assert "boom" in messages[0]
        assert "nEvent=1" in messages[0]
        assert "累计第 1 次" in messages[0]

    def test_stillimage_branch_is_instrumented_too(self, cam) -> None:
        """Both handlers were silent; a fix to only one would leave half the hole."""
        cam._frame_puller = _FailingPuller()
        cam._frame_callback(_EventConstants.MIICAM_EVENT_STILLIMAGE, None)
        assert cam._callback_error_count == 1

    def test_repeated_loss_is_counted_but_rate_limited(self, cam) -> None:
        """Per-frame WARNING at camera rate would flood the log (~1800/min at 30 fps)."""
        cam._frame_puller = _FailingPuller()
        sink_id, messages = _warning_sink()
        try:
            for _ in range(100):
                cam._frame_callback(_EventConstants.MIICAM_EVENT_IMAGE, None)
        finally:
            logger.remove(sink_id)

        assert cam._callback_error_count == 100, "every loss must still be counted"
        assert len(messages) == 2, "WARNING on the first and every 100th only"

    def test_a_good_frame_is_not_counted_as_a_loss(self, cam) -> None:
        """Without this, the counter could drift upward on success and lie."""
        cam._frame_puller = _WorkingPuller()
        received: list[object] = []
        cam._frame_callback(
            _EventConstants.MIICAM_EVENT_IMAGE,
            None,
            user_callback=lambda img, info: received.append(img),
        )
        assert len(received) == 1
        assert cam._callback_error_count == 0


class TestCounterIsPerSession:
    def test_starting_a_new_callback_session_resets_the_count(self, cam, monkeypatch) -> None:
        """A lifetime counter would mis-target the rate limit on the next run."""
        cam._frame_puller = _FailingPuller()
        cam._frame_callback(_EventConstants.MIICAM_EVENT_IMAGE, None)
        assert cam._callback_error_count == 1

        class _FakeCam:
            def StartPullModeWithCallback(self, cb, owner) -> None:
                return None

            def Stop(self) -> None:
                return None

        monkeypatch.setattr(cam, "cam", _FakeCam())
        cam._callback_mode_active = False
        session = drv._CallbackSession(cam)
        session.start()
        try:
            assert cam._callback_error_count == 0
        finally:
            session.stop()
