"""Lock-safe replay wrapper -- mocks only, no hardware.

The wrapper must (a) refuse without ``--execute`` without waiting, locking or
importing device modules, (b) consume ``--execute`` instead of forwarding it
(``replay_phases`` has no such flag), (c) hold the shared bench lock across the
whole replay process and release it in ``finally``, (d) fail closed when the
wait or the acquire fails, and (e) refuse a run directory or an exposure list it
cannot evidence. Nothing here opens a device.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from ao_shaping.tools.kernels import device_mutex as dm

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts/slm/replay_transfer_shapes_hardware.py"


def _load():
    spec = importlib.util.spec_from_file_location(
        "replay_transfer_shapes_hardware", SCRIPT
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

    def acquire(self, timeout: float = 0.0, *, poll_s: float = 0.5) -> bool:
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

    def holder(self):
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


def _argv(out: Path, *extra: str, execute: bool = True) -> list[str]:
    argv = [
        "--initial-run",
        "archived/run_a",
        "--runs",
        "archived/run_a",
        "--out",
        str(out),
        "--exposures",
        "30",
        "45",
        *extra,
    ]
    return [*argv, "--execute"] if execute else argv


def _forbid_devices(monkeypatch) -> None:
    def _boom(*_a, **_k):
        raise AssertionError("no device module may be touched on this path")

    monkeypatch.setattr(dm, "wait_until_idle", _boom)
    monkeypatch.setattr(dm, "DeviceLock", SimpleNamespace(default=_boom))


def test_help_lists_wrapper_and_replay_options(monkeypatch, capsys):
    monkeypatch.setattr(
        wrapper,
        "_replay_help_text",
        lambda: (
            "usage: replay_phases [-h] --initial-run INITIAL_RUN\n"
            "  --exposures EXPOSURES [EXPOSURES ...]\n"
            "  --frames FRAMES"
        ),
    )

    assert wrapper.main(["--help"]) == 0

    out = capsys.readouterr().out
    assert "--wait-until-idle" in out
    assert "--execute" in out
    assert "--exposures" in out
    assert "--frames" in out


def test_missing_execute_refuses_without_touching_devices(monkeypatch, tmp_path, capsys):
    _forbid_devices(monkeypatch)
    out = tmp_path / "run"

    assert wrapper.main(_argv(out, execute=False)) == 2

    assert not out.exists()
    assert not (tmp_path / "run.device_lock.json").exists()
    assert "--execute is required" in capsys.readouterr().err


def test_execute_gate_is_consumed_not_forwarded(monkeypatch, tmp_path):
    # replay_phases has no --execute: forwarding it would make argparse fail in
    # the child after the lock was taken.
    lock = _FakeLock(tmp_path / "bench.lock", held=True)
    monkeypatch.setattr(dm, "wait_until_idle", lambda *a, **k: _verdict(lock=lock))
    calls: list[list[str]] = []
    monkeypatch.setattr(
        wrapper, "_invoke_replay", lambda argv: calls.append(list(argv)) or 0
    )

    assert wrapper.main(_argv(tmp_path / "run", "--frames", "5")) == 0

    assert len(calls) == 1
    forwarded = calls[0]
    assert "--execute" not in forwarded
    assert forwarded[forwarded.index("--exposures") + 1:][:2] == ["30", "45"]
    assert forwarded[forwarded.index("--frames") + 1] == "5"
    # ... and the gate itself is what let the replay run.
    assert lock.events == ["release"]


def test_wait_timeout_never_replays_and_writes_sibling_evidence(
    monkeypatch, tmp_path, capsys
):
    holder = SimpleNamespace(pid=4242, cmdline="python scripts/slm/other_hw.py --execute")
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
        wrapper, "_invoke_replay", lambda argv: calls.append(list(argv)) or 0
    )
    out = tmp_path / "run"

    assert wrapper.main(_argv(out)) == 1

    assert calls == []
    assert not out.exists()
    evidence = tmp_path / "run.device_lock.json"
    assert evidence.is_file()
    doc = json.loads(evidence.read_text(encoding="utf-8"))
    assert doc["device_wait"]["status"] == "timeout"
    assert doc["device_wait"]["kind"] == "slm_busy"
    assert doc["device_wait"]["holder_pid"] == 4242
    assert doc["device_wait"]["holder_cmdline"].endswith("--execute")
    assert doc["replay_invoked"] is False
    assert doc["lock"]["acquired_utc"] is None
    assert "not started" in capsys.readouterr().err


def test_lock_acquisition_failure_never_replays(monkeypatch, tmp_path):
    lock = _FakeLock(tmp_path / "bench.lock", acquire_ok=False)
    monkeypatch.setattr(dm, "wait_until_idle", lambda *a, **k: _verdict())
    monkeypatch.setattr(dm, "DeviceLock", SimpleNamespace(default=lambda: lock))
    calls: list[list[str]] = []
    monkeypatch.setattr(
        wrapper, "_invoke_replay", lambda argv: calls.append(list(argv)) or 0
    )
    out = tmp_path / "run"

    assert wrapper.main(_argv(out)) == 1

    assert calls == []
    assert lock.acquire_calls == 1
    assert not out.exists()
    doc = json.loads((tmp_path / "run.device_lock.json").read_text(encoding="utf-8"))
    assert doc["lock"]["held"] is False
    assert doc["replay_invoked"] is False


def test_lock_is_held_across_the_replay_and_evidence_lands_in_run_dir(
    monkeypatch, tmp_path
):
    events: list[str] = []
    lock = _FakeLock(tmp_path / "bench.lock", events, held=True)
    monkeypatch.setattr(dm, "wait_until_idle", lambda *a, **k: _verdict(lock=lock))

    def _fake_replay(argv: list[str]) -> int:
        events.append("replay")
        Path(argv[argv.index("--out") + 1]).mkdir(parents=True)
        assert lock.is_locked(), "the bench lock must be held during the replay"
        return 0

    monkeypatch.setattr(wrapper, "_invoke_replay", _fake_replay)
    out = tmp_path / "run"

    assert wrapper.main(_argv(out)) == 0

    assert events == ["replay", "release"]
    doc = json.loads((out / "device_lock.json").read_text(encoding="utf-8"))
    assert doc["replay_invoked"] is True
    assert doc["replay_exit_code"] == 0
    assert doc["lock"]["held"] is False
    assert doc["lock"]["released"] is True
    assert doc["lock"]["held_s"] is not None
    assert doc["exposures_explicit"] is True


def test_replay_exit_code_propagates_and_the_lock_is_still_released(
    monkeypatch, tmp_path
):
    lock = _FakeLock(tmp_path / "bench.lock", held=True)
    monkeypatch.setattr(dm, "wait_until_idle", lambda *a, **k: _verdict(lock=lock))
    monkeypatch.setattr(wrapper, "_invoke_replay", lambda argv: 3)

    assert wrapper.main(_argv(tmp_path / "run")) == 3

    assert lock.release_calls == 1
    assert lock.is_locked() is False


def test_release_fault_does_not_mask_the_replay_exit_code(
    monkeypatch, tmp_path, capsys
):
    lock = _FakeLock(
        tmp_path / "bench.lock", held=True, release_error=RuntimeError("boom")
    )
    monkeypatch.setattr(dm, "wait_until_idle", lambda *a, **k: _verdict(lock=lock))
    monkeypatch.setattr(wrapper, "_invoke_replay", lambda argv: 3)

    assert wrapper.main(_argv(tmp_path / "run")) == 3

    doc = json.loads((tmp_path / "run.device_lock.json").read_text(encoding="utf-8"))
    assert doc["lock"]["released"] is False
    assert "boom" in doc["lock"]["release_error"]
    assert "release failed" in capsys.readouterr().err


def test_wait_disabled_neither_waits_nor_locks(monkeypatch, tmp_path):
    def _boom(*_a, **_k):
        raise AssertionError("--wait-until-idle 0 must not wait or lock")

    monkeypatch.setattr(dm, "wait_until_idle", _boom)
    monkeypatch.setattr(dm, "DeviceLock", SimpleNamespace(default=_boom))
    calls: list[list[str]] = []
    monkeypatch.setattr(
        wrapper, "_invoke_replay", lambda argv: calls.append(list(argv)) or 0
    )
    out = tmp_path / "run"

    assert wrapper.main([*_argv(out), "--wait-until-idle", "0"]) == 0

    assert len(calls) == 1
    doc = json.loads((tmp_path / "run.device_lock.json").read_text(encoding="utf-8"))
    assert doc["lock_mode"] == "disabled"
    assert doc["lock"]["released"] is None
    doc_wait = doc["device_wait"]
    assert doc_wait["budget_s"] == 0.0
    assert doc_wait["status"] == "skipped"


def test_wait_exception_never_replays(monkeypatch, tmp_path):
    def _boom(*_a, **_k):
        raise OSError("lock file unavailable")

    monkeypatch.setattr(dm, "wait_until_idle", _boom)
    calls: list[list[str]] = []
    monkeypatch.setattr(
        wrapper, "_invoke_replay", lambda argv: calls.append(list(argv)) or 0
    )

    assert wrapper.main(_argv(tmp_path / "run")) == 1

    assert calls == []
    doc = json.loads((tmp_path / "run.device_lock.json").read_text(encoding="utf-8"))
    assert doc["device_wait"]["status"] == "error"
    assert "lock file unavailable" in doc["device_wait"]["detail"]


def test_out_is_required_before_waiting(monkeypatch, tmp_path, capsys):
    _forbid_devices(monkeypatch)

    assert wrapper.main(["--initial-run", "a", "--runs", "a", "--execute"]) == 2

    assert "--out is required" in capsys.readouterr().err


def test_exposures_must_be_explicit_before_waiting(monkeypatch, tmp_path, capsys):
    # The module default 50/100 ms is a historical sweep; the acceptance plan
    # allows only exposures taken from the current camera range on site.
    _forbid_devices(monkeypatch)
    out = tmp_path / "run"

    argv = ["--initial-run", "a", "--runs", "a", "--out", str(out), "--execute"]
    assert wrapper.main(argv) == 2

    assert "--exposures must be given explicitly" in capsys.readouterr().err
    assert not out.exists()


@pytest.mark.parametrize("value", ["-1", "nan", "inf"])
def test_wait_budget_must_be_finite_and_non_negative(monkeypatch, tmp_path, value):
    _forbid_devices(monkeypatch)

    assert wrapper.main([*_argv(tmp_path / "run"), "--wait-until-idle", value]) == 2


def test_device_modules_stay_unimported_without_execute(tmp_path):
    # A copy of the module in a fresh interpreter: importing the wrapper must not
    # drag device code in, so a refusal stays a refusal.
    child = (
        "import importlib.util, json, sys\n"
        f"spec = importlib.util.spec_from_file_location('w', {str(SCRIPT)!r})\n"
        "m = importlib.util.module_from_spec(spec)\n"
        "sys.modules['w'] = m\n"
        "spec.loader.exec_module(m)\n"
        "assert m.main(['--initial-run', 'a', '--runs', 'a', '--out', 'x']) == 2\n"
        "bad = sorted(n for n in sys.modules if n == 'ao_shaping.tools.kernels.device_mutex'"
        " or n.startswith('ao_shaping.drivers'))\n"
        "print(json.dumps(bad))\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", child],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(tmp_path),
        env=env,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout.strip().splitlines()[-1]) == []
