"""Serialization wrapper for ``validate_transfer_shapes`` -- mocks only, no hardware.

The wrapper must (a) forward every validator argument unchanged, (b) never wait,
lock or import device modules without ``--execute``, (c) hold the shared bench
lock across the whole validator call and record evidence in ``finally``, and
(d) fail closed when the wait/acquire fails. Nothing here opens a device.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from ao_shaping.tools.kernels import device_mutex as dm

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts/slm/validate_transfer_shapes_hardware.py"


def _load():
    spec = importlib.util.spec_from_file_location(
        "validate_transfer_shapes_hardware", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


wrapper = _load()


class _FakeLock:
    """In-memory stand-in for ``DeviceLock``; records ordering in ``events``."""

    def __init__(
        self,
        path: Path,
        events: list[str] | None = None,
        *,
        acquire_ok: bool = True,
        held: bool = False,
        release_error: Exception | None = None,
    ) -> None:
        self.path = path
        self.events = events if events is not None else []
        self.acquire_ok = acquire_ok
        self._held = held
        self.release_error = release_error
        self.acquire_calls = 0
        self.release_calls = 0

    def acquire(self, timeout: float = 0.0) -> bool:
        self.acquire_calls += 1
        self.events.append("acquire")
        self._held = self.acquire_ok
        return self._held

    def release(self) -> None:
        self.release_calls += 1
        self.events.append("release")
        self._held = False
        if self.release_error is not None:
            raise self.release_error

    def is_locked(self) -> bool:
        return self._held

    def holder(self):  # only used in the failure message
        return None


def _verdict(
    *,
    ok: bool = True,
    status: str = "ok",
    kind: str | None = None,
    lock: object | None = None,
    holder: object | None = None,
    waited_s: float = 0.5,
    detail: str = "",
    remedy: str = "",
    headline: str = "idle",
):
    return SimpleNamespace(
        ok=ok,
        status=status,
        kind=kind,
        lock=lock,
        holder=holder,
        holder_is_me=False,
        waited_s=waited_s,
        detail=detail,
        remedy=remedy,
        headline=lambda: headline,
    )


def _execute_argv(out: Path, *extra: str) -> list[str]:
    return [
        "--checkpoint",
        "frozen.json",
        "--out",
        str(out),
        "--zero-xy",
        "1220",
        "962",
        "--execute",
        *extra,
    ]


def test_help_lists_wrapper_and_validator_options(monkeypatch, capsys):
    monkeypatch.setattr(
        wrapper,
        "_validator_help_text",
        lambda: (
            "usage: validate_transfer_shapes.py ...\n"
            "  --zero-xy ZERO_X ZERO_Y\n"
            "  --execute"
        ),
    )

    assert wrapper.main(["--help"]) == 0

    out = capsys.readouterr().out
    assert "--wait-until-idle" in out
    assert "--zero-xy" in out
    assert "--execute" in out


def test_plan_only_forwards_verbatim_without_wait_or_lock(monkeypatch, tmp_path):
    monkeypatch.setattr(
        dm, "wait_until_idle", lambda *a, **k: pytest.fail("wait must not run")
    )
    monkeypatch.setattr(
        dm,
        "DeviceLock",
        SimpleNamespace(default=lambda: pytest.fail("lock must not be taken")),
    )
    calls: list[list[str]] = []
    monkeypatch.setattr(
        wrapper, "_invoke_validator", lambda argv: calls.append(list(argv)) or 0
    )

    argv = [
        "--checkpoint",
        "frozen.json",
        "--out",
        str(tmp_path / "run"),
        "--zero-xy",
        "1220",
        "962",
        "--plan-only",
    ]
    assert wrapper.main(argv) == 0

    assert calls == [argv]
    assert not (tmp_path / "run").exists()
    assert not (tmp_path / "run.device_lock.json").exists()


def test_execute_waits_passes_ids_and_holds_lock_through_validator(
    monkeypatch, tmp_path
):
    events: list[str] = []
    lock = _FakeLock(tmp_path / "bench.lock", events)
    seen: dict[str, object] = {}

    def fake_wait(seconds, **kwargs):
        events.append("wait")
        seen["seconds"] = seconds
        seen["kwargs"] = kwargs
        return _verdict()

    monkeypatch.setattr(dm, "wait_until_idle", fake_wait)
    monkeypatch.setattr(dm, "DeviceLock", SimpleNamespace(default=lambda: lock))

    out = tmp_path / "run"

    def fake_validator(argv):
        events.append("validator")
        assert lock.is_locked(), "lock must be held across the validator call"
        out.mkdir()  # what the real validator does when it starts measuring
        return 0

    monkeypatch.setattr(wrapper, "_invoke_validator", fake_validator)

    rc = wrapper.main(
        _execute_argv(
            out,
            "--camera",
            "daheng",
            "--cam-id",
            "3",
            "--slm-number",
            "2",
            "--wait-until-idle",
            "30",
        )
    )

    assert rc == 0
    assert seen["seconds"] == 30.0
    kwargs = seen["kwargs"]
    assert kwargs["slm_number"] == 2
    assert kwargs["cam_id"] == 3
    assert kwargs["cam_type"] == "daheng"
    assert [Path(p).name for p in kwargs["own_paths"]] == [
        SCRIPT.name,
        "validate_transfer_shapes.py",
    ]
    assert events == ["wait", "acquire", "validator", "release"]

    doc = json.loads((out / "device_lock.json").read_text(encoding="utf-8"))
    assert doc["schema"] == "validate_transfer_shapes_hardware/device_lock/1"
    assert doc["lock_mode"] == "wait+acquire"
    assert doc["device_wait"]["status"] == "ok"
    assert doc["device_wait"]["budget_s"] == 30.0
    assert doc["lock"]["path"] == str(lock.path)
    assert doc["lock"]["held"] is False
    assert doc["lock"]["released"] is True
    assert doc["lock"]["acquired_utc"]
    assert doc["lock"]["released_utc"]
    assert doc["lock"]["held_s"] is not None
    assert doc["validator_invoked"] is True
    assert doc["validator_exit_code"] == 0


def test_lock_returned_by_wait_is_reused_not_reacquired(monkeypatch, tmp_path):
    events: list[str] = []
    lock = _FakeLock(tmp_path / "bench.lock", events, held=True)
    monkeypatch.setattr(
        dm, "wait_until_idle", lambda *a, **k: _verdict(lock=lock)
    )
    monkeypatch.setattr(
        dm,
        "DeviceLock",
        SimpleNamespace(default=lambda: pytest.fail("must reuse the wait's lock")),
    )
    out = tmp_path / "run"

    def fake_validator(argv):
        assert lock.is_locked()
        out.mkdir()
        return 0

    monkeypatch.setattr(wrapper, "_invoke_validator", fake_validator)

    assert wrapper.main(_execute_argv(out)) == 0
    assert lock.acquire_calls == 0
    assert lock.release_calls == 1
    assert events == ["release"]


def test_wait_timeout_fails_closed_and_writes_sibling_evidence(
    monkeypatch, tmp_path
):
    holder = SimpleNamespace(
        pid=4242, cmdline="python scripts/slm/other_hw.py --execute"
    )
    monkeypatch.setattr(
        dm,
        "wait_until_idle",
        lambda *a, **k: _verdict(
            ok=False,
            status="timeout",
            kind="slm_busy",
            holder=holder,
            detail="lock held",
            remedy="wait or power-cycle the SLM",
            headline="SLM busy",
        ),
    )
    calls: list[list[str]] = []
    monkeypatch.setattr(
        wrapper, "_invoke_validator", lambda argv: calls.append(list(argv)) or 0
    )
    out = tmp_path / "run"

    assert wrapper.main(_execute_argv(out)) == 1

    assert calls == []
    assert not out.exists()
    evidence = tmp_path / "run.device_lock.json"
    assert evidence.is_file()
    doc = json.loads(evidence.read_text(encoding="utf-8"))
    assert doc["device_wait"]["status"] == "timeout"
    assert doc["device_wait"]["kind"] == "slm_busy"
    assert doc["device_wait"]["holder_pid"] == 4242
    assert doc["device_wait"]["holder_cmdline"].endswith("--execute")
    assert doc["device_wait"]["remedy"] == "wait or power-cycle the SLM"
    assert doc["validator_invoked"] is False
    assert doc["lock"]["acquired_utc"] is None


def test_lock_acquisition_failure_never_calls_validator(monkeypatch, tmp_path):
    lock = _FakeLock(tmp_path / "bench.lock", acquire_ok=False)
    monkeypatch.setattr(dm, "wait_until_idle", lambda *a, **k: _verdict())
    monkeypatch.setattr(dm, "DeviceLock", SimpleNamespace(default=lambda: lock))
    calls: list[list[str]] = []
    monkeypatch.setattr(
        wrapper, "_invoke_validator", lambda argv: calls.append(list(argv)) or 0
    )
    out = tmp_path / "run"

    assert wrapper.main(_execute_argv(out)) == 1

    assert calls == []
    assert not out.exists()
    doc = json.loads((tmp_path / "run.device_lock.json").read_text(encoding="utf-8"))
    assert doc["lock"]["path"] == str(lock.path)
    assert doc["lock"]["held"] is False
    assert doc["validator_invoked"] is False


def test_zero_wait_bypasses_wait_and_lock(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        dm, "wait_until_idle", lambda *a, **k: pytest.fail("wait disabled")
    )
    monkeypatch.setattr(
        dm,
        "DeviceLock",
        SimpleNamespace(default=lambda: pytest.fail("lock disabled")),
    )
    out = tmp_path / "run"

    def fake_validator(argv):
        out.mkdir()
        return 0

    monkeypatch.setattr(wrapper, "_invoke_validator", fake_validator)

    assert wrapper.main(_execute_argv(out, "--wait-until-idle", "0")) == 0

    doc = json.loads((out / "device_lock.json").read_text(encoding="utf-8"))
    assert doc["lock_mode"] == "disabled"
    assert doc["wait_until_idle_s"] == 0.0
    assert doc["device_wait"]["status"] == "skipped"
    assert "disabled" in doc["device_wait"]["detail"]
    assert doc["lock"]["path"] is None
    assert doc["lock"]["held"] is False
    assert doc["validator_exit_code"] == 0


def test_release_error_is_recorded_without_masking_validator_exit(
    monkeypatch, tmp_path
):
    events: list[str] = []
    lock = _FakeLock(
        tmp_path / "bench.lock", events, release_error=RuntimeError("release boom")
    )
    monkeypatch.setattr(dm, "wait_until_idle", lambda *a, **k: _verdict())
    monkeypatch.setattr(dm, "DeviceLock", SimpleNamespace(default=lambda: lock))
    out = tmp_path / "run"

    def fake_validator(argv):
        out.mkdir()
        return 7

    monkeypatch.setattr(wrapper, "_invoke_validator", fake_validator)

    assert wrapper.main(_execute_argv(out)) == 7

    doc = json.loads((out / "device_lock.json").read_text(encoding="utf-8"))
    assert doc["validator_exit_code"] == 7
    assert doc["lock"]["released"] is False
    assert "RuntimeError: release boom" in doc["lock"]["release_error"]
    assert doc["lock"]["released_utc"]


def test_validator_failure_before_output_dir_writes_sibling_not_directory(
    monkeypatch, tmp_path
):
    lock = _FakeLock(tmp_path / "bench.lock")
    monkeypatch.setattr(dm, "wait_until_idle", lambda *a, **k: _verdict())
    monkeypatch.setattr(dm, "DeviceLock", SimpleNamespace(default=lambda: lock))
    monkeypatch.setattr(wrapper, "_invoke_validator", lambda argv: 3)
    out = tmp_path / "run"

    assert wrapper.main(_execute_argv(out)) == 3

    assert not out.exists(), "a sibling evidence file must not create the run dir"
    evidence = tmp_path / "run.device_lock.json"
    assert evidence.is_file()
    doc = json.loads(evidence.read_text(encoding="utf-8"))
    assert doc["validator_exit_code"] == 3
    assert doc["lock"]["released"] is True


def test_negative_wait_is_rejected_before_any_device_work(monkeypatch):
    monkeypatch.setattr(
        dm, "wait_until_idle", lambda *a, **k: pytest.fail("must not run")
    )

    with pytest.raises(SystemExit) as excinfo:
        wrapper.main(["--checkpoint", "c.json", "--out", "run", "--zero-xy", "0", "0",
                      "--wait-until-idle", "-1", "--execute"])

    assert excinfo.value.code == 2


def test_help_and_plan_only_never_import_device_modules(tmp_path):
    """Prove import freedom in a fresh interpreter (module cache is per-process)."""
    code = r'''
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("wrapper_import_guard", {script!r})
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod._validator_help_text = lambda: "usage: validate_transfer_shapes.py\n  --zero-xy X Y"
mod._invoke_validator = lambda argv: 0
help_rc = mod.main(["--help"])
plan_rc = mod.main(["--checkpoint", "c.json", "--out", "run", "--zero-xy", "1220", "962"])
forbidden = sorted(
    name for name in sys.modules
    if name == "ao_shaping.tools.kernels.device_mutex"
    or name.startswith("ao_shaping.drivers")
    or name.split(".")[0] in ("gxipy", "MvImport")
)
print(json.dumps({{"help": help_rc, "plan": plan_rc, "forbidden": forbidden}}))
'''.format(script=str(SCRIPT))

    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(tmp_path),
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert payload == {"help": 0, "plan": 0, "forbidden": []}
