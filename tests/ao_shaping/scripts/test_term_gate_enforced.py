"""#194 guard: the glossary gate must be *callable and wired*, not remembered.

AGENTS.md makes ``term_unify_20261016.py`` a required pre-commit check, yet the
rule had no teeth: the unifier exited 0 whatever it found, scanned only
markdown, and neither CI nor any test invoked it. These tests pin the three
things that make it a gate -- an exit code CI can act on, a detector that
provably fires, and a workflow that actually calls it.

The dry-run contract stays exactly as AGENTS.md documents it (always exit 0,
print the count); ``--check`` is the only mode allowed to fail.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
_UNIFIER = _REPO / "scripts" / "_common" / "term_unify_20261016.py"
_WORKFLOW = _REPO / ".github" / "workflows" / "ci.yml"

#: ``.gitignore``-free prose that the unifier's own table rewrites (T-11: 台架 -> 实验系统).
_VIOLATION = "# 探针\n\n本次在台架上完成验证。\n"
_CANONICAL = "# 探针\n\n本次在实验系统上完成验证。\n"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_UNIFIER), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def _seed(root: Path, text: str) -> Path:
    (root / "probe.md").write_text(text, encoding="utf-8")
    return root


def test_the_repository_is_term_clean_under_check_mode() -> None:
    """The CI assertion itself: ``--check`` on the real repo must be green."""
    proc = _run("--check")
    assert proc.returncode == 0, (
        f"--check exited {proc.returncode}:\n{proc.stdout}\n{proc.stderr}"
    )
    assert "合计: 0 个文件, 0 行" in proc.stdout, proc.stdout


def test_check_mode_fires_on_a_planted_violation(tmp_path: Path) -> None:
    """Non-vacuity: a gate that cannot fail is not a gate."""
    proc = _run("--check", "--root", str(_seed(tmp_path, _VIOLATION)))
    assert proc.returncode == 1, (
        f"--check accepted a forbidden term (rc={proc.returncode}):\n{proc.stdout}"
    )
    assert "术语门禁未通过" in proc.stdout, proc.stdout


def test_check_mode_accepts_the_canonical_spelling(tmp_path: Path) -> None:
    """The detector must not fail on the glossary's own right column."""
    proc = _run("--check", "--root", str(_seed(tmp_path, _CANONICAL)))
    assert proc.returncode == 0, (
        f"--check rejected canonical prose (rc={proc.returncode}):\n{proc.stdout}"
    )


def test_dry_run_keeps_its_zero_exit_contract(tmp_path: Path) -> None:
    """AGENTS.md tells people to read the dry-run count; it must never fail them.

    Without this, a future "make dry-run fail too" edit breaks every documented
    invocation in the same commit that adds the gate.
    """
    proc = _run("--root", str(_seed(tmp_path, _VIOLATION)))
    assert proc.returncode == 0, (
        f"dry-run must exit 0 even when it finds drift:\n{proc.stdout}\n{proc.stderr}"
    )
    assert "1 个文件" in proc.stdout, proc.stdout


def test_apply_and_check_cannot_be_combined(tmp_path: Path) -> None:
    """One mode writes files, the other is a verdict; together they are ambiguous."""
    proc = _run("--apply", "--check", "--root", str(_seed(tmp_path, _VIOLATION)))
    assert proc.returncode != 0, "--apply --check must be refused"
    assert "互斥" in (proc.stdout + proc.stderr), proc.stdout + proc.stderr


def test_ci_workflow_actually_calls_the_gate() -> None:
    """The whole point of #194: the script existed, nothing ran it."""
    text = _WORKFLOW.read_text(encoding="utf-8")
    assert "term_unify_20261016.py --check" in text, (
        "ci.yml no longer invokes the terminology gate -- the rule is back to "
        "depending on someone remembering to run it by hand"
    )
    job = text.split("terminology:", 1)[1].split("pytest:", 1)[0]
    assert "continue-on-error" not in job, (
        "the terminology job was downgraded to advisory: findings would report "
        "but never block, which is the failure mode this job was added to fix"
    )

#: English prose whose last word is ``pupil``, followed by a Chinese paragraph. The
#: context window used to span the newline, see a CJK neighbour, and gloss the English
#: word -- it rewrote exactly this shape in ``scripts/README.md``.
_ENGLISH_PUPIL_WRAP = (
    "fits the model offline (focal scale, pupil\n与世界坐标系的对齐另行说明。\n"
)
#: Chinese narrative: the one place the glossary *does* want the gloss.
_CHINESE_PUPIL = "调整 pupil 的共轭位置后再采集。\n"


def test_english_pupil_at_a_line_end_stays_bare(tmp_path: Path) -> None:
    """Scope: the gloss is for 中文叙述, not for an English word that wraps.

    Regression: the 40-char window crossed the newline, so the following Chinese
    paragraph made ``pupil`` look like Chinese narrative and it was rewritten to
    ``pupil（瞳面）`` mid-sentence in ``scripts/README.md``. That false positive is
    now a hard CI failure, so it would block every merge that touches the line.
    """
    proc = _run("--check", "--root", str(_seed(tmp_path, _ENGLISH_PUPIL_WRAP)))
    assert proc.returncode == 0, (
        f"English prose was treated as Chinese narrative:\n{proc.stdout}"
    )
    assert "0 个文件" in proc.stdout, proc.stdout


def test_chinese_narrative_gets_the_gloss_exactly_once(tmp_path: Path) -> None:
    """The rule must still fire where it is meaningful -- and stay idempotent."""
    root = _seed(tmp_path, _CHINESE_PUPIL)
    _run("--apply", "--root", str(root))
    text = (root / "probe.md").read_text(encoding="utf-8")
    assert "pupil（瞳面）" in text, text

    again = _run("--check", "--root", str(root))
    assert again.returncode == 0, (
        f"a second pass wants to annotate again (double gloss):\n{again.stdout}"
    )

#: A ``.py`` whose only prose is Chinese. The ``--root`` seam was first honoured for
#: ``*.md`` only, so a temp-root run still scanned -- and rewrote -- the real repo.
_PY_VIOLATION = 'def probe() -> None:\n    """本次在台架上完成验证。"""\n'


def test_root_isolation_also_holds_for_python_prose(tmp_path: Path) -> None:
    """``--root`` must own *every* scan, or a test run mutates the checkout.

    Regression: the ``*.py`` pipeline resolved ``REPO`` instead of ``ROOT``, so
    ``--apply --root <tmp>`` rewrote files in the real worktree while a temp-root
    ``--check`` reported unrelated repo paths (the totals were the giveaway).
    """
    root = tmp_path / "py"
    root.mkdir()
    (root / "probe.py").write_text(_PY_VIOLATION, encoding="utf-8")

    proc = _run("--check", "--root", str(root))
    assert proc.returncode == 1, proc.stdout
    assert "probe.py" in proc.stdout, proc.stdout
    assert "合计: 1 个文件, 1 行" in proc.stdout, proc.stdout
