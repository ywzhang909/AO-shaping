"""opened_device: hardware-open retry, no-op on config errors, always closes.

The device genuine failures (``CameraError`` / ``SantecError``) are plain
``Exception`` subclasses, so the retry layer must treat them as retryable.
config/缺 SDK errors must never be retried.
"""
from __future__ import annotations

import pytest

from ao_shaping.tools.slm.transfer.hardware import DeviceOpenError, opened_device


class _FakeDevice:
    """Records open/close/sleep cadence against a scripted open outcome."""

    def __init__(self, failures: int = 0, exc: type[Exception] = RuntimeError):
        self.failures = failures
        self.open_exc = exc
        self.open_calls = 0
        self.close_calls = 0
        self.slept = []
        self.closed = 0

    def open(self):
        self.open_calls += 1
        if self.open_calls <= self.failures:
            raise self.open_exc("open failed")

    def close(self):
        self.close_calls += 1


def test_succeeds_when_third_open_succeeds(monkeypatch):
    sleeps = []
    monkeypatch.setattr("ao_shaping.tools.slm.transfer.hardware.time.sleep", sleeps.append)
    device = _FakeDevice(failures=2)
    with opened_device(device, attempts=3, delay_seconds=0.5) as handle:
        assert handle is device
    assert device.open_calls == 3
    # 两次失败的 open 各 best-effort close, exit 时再 close 一次。
    assert device.close_calls == 3
    # 失败的 open 之间各 sleep 一次。
    assert sleeps == [0.5, 0.5]


def test_close_after_each_failure_and_no_double_close_on_success(monkeypatch):
    sleeps = []
    monkeypatch.setattr("ao_shaping.tools.slm.transfer.hardware.time.sleep", sleeps.append)
    device = _FakeDevice(failures=2)
    with opened_device(device, attempts=3, delay_seconds=0.5):
        pass
    # 每次失败的 open 后有 best-effort close; 成功后 exit 再 close 一次。
    assert device.close_calls == 3


def test_final_failure_raises_device_open_error(monkeypatch):
    sleeps = []
    monkeypatch.setattr("ao_shaping.tools.slm.transfer.hardware.time.sleep", sleeps.append)
    device = _FakeDevice(failures=5)
    with pytest.raises(DeviceOpenError) as info:
        with opened_device(device, attempts=3, delay_seconds=0.5):
            pytest.fail("body must not run when open never succeeds")
    err = info.value
    assert err.attempts == 3
    assert err.device is device
    assert err.exit_code == 4
    assert "_FakeDevice" in str(err)
    assert "3" in str(err)
    # cause 保留原异常。
    assert isinstance(err.__cause__, RuntimeError)
    assert device.open_calls == 3
    assert device.close_calls == 3
    assert sleeps == [0.5, 0.5]


@pytest.mark.parametrize("bad", [ImportError, ValueError, TypeError])
def test_config_or_sdk_missing_is_never_retried(monkeypatch, bad):
    sleeps = []
    monkeypatch.setattr("ao_shaping.tools.slm.transfer.hardware.time.sleep", sleeps.append)
    device = _FakeDevice(failures=99, exc=bad)
    with pytest.raises(bad):
        with opened_device(device, attempts=3, delay_seconds=0.5):
            pytest.fail("body must not run")
    assert device.open_calls == 1
    assert sleeps == []


def test_body_exception_still_closes(monkeypatch):
    sleeps = []
    monkeypatch.setattr("ao_shaping.tools.slm.transfer.hardware.time.sleep", sleeps.append)
    device = _FakeDevice(failures=0)
    with pytest.raises(RuntimeError, match="boom"):
        with opened_device(device, attempts=3, delay_seconds=0.5):
            raise RuntimeError("boom")
    assert device.close_calls == 1
    assert sleeps == []


def test_invalid_attempts_is_value_error():
    device = _FakeDevice()
    with pytest.raises(ValueError):
        with opened_device(device, attempts=0):
            pytest.fail("body must not run")