"""#56 slice (4) guard: the module-size rule must be *callable and wired*, not remembered.

``docs/dev/coding_conventions.md`` §9 says a module over 800 lines should be split.
That rule had no teeth at all: no workflow, no test, no exit code — the count of
oversized modules just grew. These tests pin the three things that make it a gate:
an exit code CI can act on, a detector that provably fires, and a workflow that
actually calls it.

The ratchet contract is deliberate and is pinned here too: the baseline may only
ever be tightened. A future "rebase the baseline to today's sizes to get green"
edit would silently re-allow every regression the gate exists to catch, so the
tests assert shrinking passes and growth fails rather than merely asserting the
sizes match.

The dry-run contract matches the terminology gate (always exit 0, print the
count); ``--check`` is the only mode allowed to fail.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
_GATE = _REPO / "scripts" / "check_module_size.py"
_WORKFLOW = _REPO / ".github" / "workflows" / "ci.yml"
_BASELINE = _REPO / "scripts" / "_common" / "module_size_baseline.json"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_GATE), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def _seed_module(root: Path, rel: str, lines: int) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('"""probe"""\n' + "x = 1\n" * (lines - 1), encoding="utf-8")
    return path


def _seed_baseline(root: Path, mapping: dict[str, int]) -> Path:
    import json

    path = root / "baseline.json"
    path.write_text(json.dumps(mapping), encoding="utf-8")
    return path


def test_check_mode_is_green_on_the_real_repository() -> None:
    """The CI assertion itself: ``--check`` on the real repo must be green."""
    proc = _run("--check")
    assert proc.returncode == 0, (
        f"--check exited {proc.returncode}:\n{proc.stdout}\n{proc.stderr}"
    )
    assert "模块规模门禁未通过" not in proc.stdout, proc.stdout


def test_check_mode_fires_on_a_newly_oversized_module(tmp_path: Path) -> None:
    """Non-vacuity: a gate that cannot fail is not a gate.

    The whole point of the ratchet is that a *new* 800+ line module is refused
    even though 130 of them already exist.
    """
    root = tmp_path / "repo"
    _seed_module(root, "src/probe_module.py", 850)
    baseline = _seed_baseline(tmp_path, {})

    proc = _run("--check", "--root", str(root), "--baseline", str(baseline))
    assert proc.returncode == 1, (
        f"--check accepted a new 850-line module (rc={proc.returncode}):\n{proc.stdout}"
    )
    assert "新越线" in proc.stdout, proc.stdout
    assert "src/probe_module.py" in proc.stdout, proc.stdout


def test_check_mode_fires_when_a_baselined_module_grows(tmp_path: Path) -> None:
    """The ratchet direction that matters: existing debt must not get bigger."""
    root = tmp_path / "repo"
    _seed_module(root, "src/probe_module.py", 900)
    baseline = _seed_baseline(tmp_path, {"src/probe_module.py": 880})

    proc = _run("--check", "--root", str(root), "--baseline", str(baseline))
    assert proc.returncode == 1, (
        f"--check accepted growth 880 -> 900 (rc={proc.returncode}):\n{proc.stdout}"
    )
    assert "增长" in proc.stdout, proc.stdout


def test_shrinking_is_always_allowed(tmp_path: Path) -> None:
    """Splitting a module pays off immediately; only growth is refused.

    Without this, someone "fixing" the gate by making it compare exact sizes
    would break every in-progress split.
    """
    root = tmp_path / "repo"
    _seed_module(root, "src/probe_module.py", 700)
    baseline = _seed_baseline(tmp_path, {"src/probe_module.py": 4000})

    proc = _run("--check", "--root", str(root), "--baseline", str(baseline))
    assert proc.returncode == 0, (
        f"--check rejected a 4000 -> 700 split (rc={proc.returncode}):\n{proc.stdout}"
    )


def test_modules_under_the_threshold_are_ignored(tmp_path: Path) -> None:
    """Scope: the gate manages oversized modules, it does not police all sizes."""
    root = tmp_path / "repo"
    _seed_module(root, "src/small.py", 120)
    baseline = _seed_baseline(tmp_path, {})

    proc = _run("--check", "--root", str(root), "--baseline", str(baseline))
    assert proc.returncode == 0, proc.stdout
    assert "合计: 0 个模块" in proc.stdout, proc.stdout


def test_dry_run_keeps_its_zero_exit_contract(tmp_path: Path) -> None:
    """Reading the report must never fail, even when it finds offenders."""
    root = tmp_path / "repo"
    _seed_module(root, "src/probe_module.py", 850)

    proc = _run("--root", str(root))
    assert proc.returncode == 0, (
        f"dry-run must exit 0 even with findings:\n{proc.stdout}\n{proc.stderr}"
    )
    assert "合计: 1 个模块" in proc.stdout, proc.stdout


def test_update_baseline_and_check_are_mutually_exclusive() -> None:
    """One mode writes the ratchet, the other is a verdict; together ambiguous."""
    proc = _run("--check", "--update-baseline")
    assert proc.returncode == 2, f"expected refusal, got rc={proc.returncode}"
    assert "互斥" in (proc.stdout + proc.stderr), proc.stdout + proc.stderr


def test_root_isolation_holds(tmp_path: Path) -> None:
    """``--root`` must own the scan, or a test run reports the real repo's files."""
    root = tmp_path / "repo"
    _seed_module(root, "src/probe_module.py", 850)
    proc = _run("--check", "--root", str(root), "--baseline", str(_BASELINE))
    # 真实仓库的 130 个受管模块不应出现在 tmp-root 的报告里
    assert "合计: 1 个模块" in proc.stdout, proc.stdout


def test_ci_workflow_actually_calls_the_gate() -> None:
    """The whole point of #56 slice (4): the rule existed, nothing ran it."""
    text = _WORKFLOW.read_text(encoding="utf-8")
    assert "check_module_size.py --check" in text, (
        "ci.yml no longer invokes the module-size gate -- the rule is back to "
        "depending on someone remembering to run it by hand"
    )
    job = text.split("module-size:", 1)[1]
    assert "continue-on-error" not in job, (
        "the module-size job was downgraded to advisory: findings would report "
        "but never block, which is the failure mode this job was added to fix"
    )


def test_the_gate_is_stdlib_only() -> None:
    """``ci.yml`` runs this script with bare ``python3`` and installs nothing.

    Only module-level imports are checked: a deferred import inside a function
    is not loaded at start-up, so it cannot crash the invocation CI makes.
    """
    tree = ast.parse(_GATE.read_text(encoding="utf-8"))
    stdlib = set(sys.stdlib_module_names)
    offenders: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            roots = [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            roots = [] if node.level else [(node.module or "").split(".")[0]]
        else:
            continue
        offenders += [r for r in roots if r and r not in stdlib]
    assert not offenders, (
        f"the module-size gate now imports {sorted(set(offenders))} at module "
        "level; the CI job installs nothing, so the gate crashes there"
    )


def test_the_gate_runs_without_site_packages() -> None:
    """The same contract enforced by execution: ``-S`` hides site-packages."""
    proc = subprocess.run(
        [sys.executable, "-S", str(_GATE), "--check"],
        capture_output=True,
        text=True,
        check=False,
        cwd=_REPO,
    )
    assert "ModuleNotFoundError" not in proc.stderr, (
        f"the gate needs a package CI never installs:\n{proc.stderr}"
    )
    assert proc.returncode == 0, (
        f"stdlib-only --check exited {proc.returncode}:\n{proc.stdout}\n{proc.stderr}"
    )


def test_importing_the_gate_scans_and_writes_nothing(tmp_path: Path) -> None:
    """Import must be inert: the scan belongs to ``main()``, not module scope."""
    import contextlib
    import importlib.util
    import io

    saved_argv = list(sys.argv)
    sys.argv = ["pytest", "--update-baseline", "--root", str(tmp_path)]
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            spec = importlib.util.spec_from_file_location("_module_size_probe", _GATE)
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)
    finally:
        sys.argv = saved_argv

    assert buf.getvalue() == "", f"importing the gate printed: {buf.getvalue()!r}"
    assert not list(tmp_path.rglob("*.json")), "importing the gate wrote a baseline"
    assert callable(getattr(module, "main", None)), "main() is gone"


def test_the_baseline_file_is_committed() -> None:
    """The ratchet is only portable if its baseline travels with the repo."""
    assert _BASELINE.is_file(), (
        "scripts/_common/module_size_baseline.json is missing: on a clean CI "
        "checkout every oversized module would look brand new and the gate "
        "would fail on the first commit"
    )


def test_oversized_scripts_are_reported_but_not_gated(tmp_path: Path) -> None:
    """``scripts/`` is AI-facing one-off code: visible in the report, not a CI red.

    The 2026-10-29 decision (AGENTS.md「scripts/ 目录：AI 一次性脚本」) is that
    splitting a report script purely for line count buys nothing — the script has
    one consumer and won't be extended. What *does* matter there is moving
    reusable helpers into ``src/`` and deleting finished probes. So the gate
    keeps showing them (drift stays visible) but only ``src/`` can fail the run.
    """
    root = tmp_path / "repo"
    _seed_module(root, "scripts/probe_report.py", 900)
    baseline = _seed_baseline(tmp_path, {})

    proc = _run("--check", "--root", str(root), "--baseline", str(baseline))
    assert proc.returncode == 0, (
        "an oversized script failed the gate; scripts/ is report-only:\n"
        f"{proc.stdout}"
    )
    assert "scripts/probe_report.py" in proc.stdout, "the script vanished from the report"
    assert "仅报告" in proc.stdout, proc.stdout


def test_baseline_only_covers_the_gated_root() -> None:
    """A baseline entry the gate never reads is debt nobody is tracking.

    If ``scripts/`` entries crept back into the baseline, ``--update-baseline``
    would be recording ratchets for modules that can never fail — and the file
    would drift from "what the gate enforces" to "every big file we ever had".
    """
    import json

    data = json.loads(_BASELINE.read_text(encoding="utf-8"))
    stray = sorted(k for k in data if not k.startswith("src/"))
    assert not stray, (
        f"{len(stray)} non-src entries in the ratchet baseline (e.g. {stray[:3]}); "
        "scripts/ is report-only, so it must not appear here"
    )
