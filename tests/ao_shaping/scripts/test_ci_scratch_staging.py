"""Why the CI harness must copy "contents into" the scratch data dir.

``scripts/build/run_pytest_ci.sh`` seeds ``$SCRATCH/data`` (bind-mounted at
``/workspace/data``) from ``$HOME/ao-testdata``. The old line was::

    rm -rf "$SCRATCH/data" 2>/dev/null || true
    cp -r "$SEED_DATA" "$SCRATCH/data"

Both halves are quietly wrong. The container runs as root, so a previous run leaves
root-owned directories that a normal-user ``rm -rf`` cannot remove; ``|| true`` hides
that failure. ``cp -r src dst`` then **nests** into the surviving destination, so the
fixtures land at ``data/ao-testdata/...`` instead of ``data/...``. Nothing reports
this: tests simply lose their fixtures, and 8 gui/r50 tests fail with
"CSV not found: /workspace/data/1300-5-enriched.csv" -- a red that has nothing to do
with the code under test. Measured on 2026-10-24, where it cost a full debugging pass
before the mount was inspected.

The safe form is ``cp -a "$SEED_DATA/." "$SCRATCH/data/"``: it copies the *contents*
into the directory whether or not the directory exists.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

HARNESS = Path(__file__).resolve().parents[3] / "scripts" / "build" / "run_pytest_ci.sh"


def _bash(script: str, cwd: Path) -> None:
    proc = subprocess.run(["bash", "-e", "-c", script], cwd=cwd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


@pytest.fixture()
def seed(tmp_path: Path) -> Path:
    root = tmp_path / "seed"
    (root / "configs").mkdir(parents=True)
    (root / "fixture.csv").write_text("a,b\n")
    return root


def test_recursive_copy_into_existing_dir_nests(seed: Path, tmp_path: Path) -> None:
    """Reproduces the hazard: a second ``cp -r src dst`` creates dst/src."""
    dst = tmp_path / "dst"
    dst.mkdir()
    _bash(f'cp -r "{seed}" "{dst}"', cwd=tmp_path)
    assert (dst / "seed" / "fixture.csv").exists(), "expected the nesting behaviour this bug relies on"
    assert not (dst / "fixture.csv").exists(), "fixtures are invisible at the top level -- the silent failure"


def test_content_copy_form_is_idempotent(seed: Path, tmp_path: Path) -> None:
    """The form the harness now uses: contents land at the top level, run after run."""
    dst = tmp_path / "dst"
    for _ in range(2):
        dst.mkdir(parents=True, exist_ok=True)
        _bash(f'cp -a "{seed}/." "{dst}/"', cwd=tmp_path)
    assert (dst / "fixture.csv").exists()
    assert (dst / "configs").is_dir()
    assert not (dst / "seed").exists(), "the safe form must never nest"


def test_harness_uses_the_safe_form() -> None:
    script = HARNESS.read_text(encoding="utf-8")
    assert 'cp -a "$SEED_DATA/." "$SCRATCH/data/"' in script, (
        "the harness must copy seed contents into $SCRATCH/data; `cp -r src dst` nests "
        "when dst survives cleanup and silently hides every fixture"
    )
    assert 'cp -r "$SEED_DATA" "$SCRATCH/data"' not in script, "the nesting form is back"
    assert "FATAL: staged" in script, "staging must fail loudly when the entry count disagrees"


def test_cleanup_failure_is_not_swallowed() -> None:
    """A bare ``rm -rf … || true`` followed by an unconditional cp is the old bug."""
    script = HARNESS.read_text(encoding="utf-8")
    assert 'if [ -e "$SCRATCH/data" ]' in script, (
        "a normal-user rm -rf cannot remove root-owned leftovers from a container run; "
        "the harness must detect the surviving directory and clean it as root"
    )
    assert 'rm -rf "$SCRATCH/data" 2>/dev/null || true\n    cp' not in script


def test_harness_still_executes_when_bash_missing() -> None:
    assert shutil.which("bash"), "these tests assume bash is the harness interpreter"
