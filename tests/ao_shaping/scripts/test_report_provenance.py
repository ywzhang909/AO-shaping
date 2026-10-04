"""The ``report/`` tree must stay linked to the scripts that produce it.

**What this pins.** On 2026-10-05 the experiment reports moved out of ``docs/``
into a repo-root ``report/`` so they would not be mixed with device
documentation. That move touched three things at once, and each can rot silently:

1. the **provenance header** inside every report (rendered from
   ``scripts/_common/provenance.py``) -- a generator rewrite or a hand edit drops
   it, and then a reader cannot tell what produced the numbers;
2. the **registry** -- a report added without a ``REPORTS`` entry is invisible to
   the index, and a renamed script leaves a dangling path in every report;
3. the **links** -- the header's script link is relative, so a report nested at a
   different depth silently points at the wrong place.

A grep cannot catch any of these: the header is machine-generated, the registry
is data, and a wrong-depth link still *looks* like a link. So the checks are
structural, the same reason ``test_conventions.py`` walks the AST instead of
matching lines.

The sync script is invoked as a subprocess rather than imported, because
``--check`` is the contract being tested -- importing it would let a refactor of
its internals silently redefine what "clean" means.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
REPORT_DIR = REPO / "report"
SYNC = REPO / "scripts" / "sync_report_provenance.py"

sys.path.insert(0, str(REPO))
from scripts._common.provenance import REPORTS, script_link  # noqa: E402

_MARKDOWN_LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")

#: A link target only counts as a file reference if it ends in a known extension.
#: Without this, prose notation is misread as a link: ``report/slm/report2.md``
#: writes Zernike modes as ``[5](2,0)defocus``, and the tuple ``2,0`` was being
#: resolved as a path. Matched with :func:`re.search`, not :meth:`re.match` --
#: an anchored pattern would reject every multi-segment path and make the whole
#: check pass vacuously.
_LINKABLE = re.compile(
    r"\.(?:md|markdown|png|jpe?g|gif|csv|tsv|json|npy|npz|bmp|tiff?|txt|pdf|"
    r"xlsx?|docx?)$",
    re.IGNORECASE,
)

#: Links that were already broken before the 2026-10-05 move: ``report -> (href,
#: reason)``. Each needs a justification, because an unexplained baseline is just
#: a suppressed failure.
_PREEXISTING_BROKEN: dict[str, dict[str, str]] = {
    "report/slm/slm_rms_spgd/0516.md": {
        "image/0516/1778923338908.png": (
            "screenshot never committed -- searched the whole repo by name, it "
            "exists nowhere. The embed has always rendered broken."
        ),
        "image/0516/1778923774713.png": (
            "screenshot never committed -- searched the whole repo by name, it "
            "exists nowhere. The embed has always rendered broken."
        ),
    },
}


def _link_targets(text: str) -> list[str]:
    """File-ish link targets in ``text``, with anchors/line refs stripped."""
    out = []
    for href in _MARKDOWN_LINK.findall(text):
        href = href.strip()
        if href.startswith(("http://", "https://", "#", "mailto:")):
            continue
        href = href.split("#", 1)[0].split(":", 1)[0]  # drop anchor / :47 line ref
        if href and _LINKABLE.search(href):
            out.append(href)
    return out


def _report_markdown() -> list[Path]:
    """Every markdown file under ``report/`` except the generated index."""
    return [
        p
        for p in sorted(REPORT_DIR.rglob("*.md"))
        if p.relative_to(REPO).as_posix() != "report/README.md"
    ]


def test_the_report_tree_exists_and_is_not_empty() -> None:
    """A vacuous pass is worse than a failure: assert there is something to guard."""
    assert REPORT_DIR.is_dir(), "report/ is missing -- the reports were never moved"
    assert len(_report_markdown()) >= 40, (
        f"only {len(_report_markdown())} reports found; the move should have brought "
        "the whole tree, so a near-empty report/ means the sweep went wrong"
    )


def test_sync_check_reports_no_drift() -> None:
    """``--check`` must exit 0: every header present and byte-identical."""
    proc = subprocess.run(
        [sys.executable, str(SYNC), "--check"],
        capture_output=True,
        text=True,
        cwd=str(REPO),
        check=False,
    )
    assert proc.returncode == 0, (
        "report provenance is stale -- run "
        "`python scripts/sync_report_provenance.py`:\n"
        f"{proc.stdout}\n{proc.stderr}"
    )


@pytest.mark.parametrize("path", _report_markdown(), ids=lambda p: p.as_posix()[-60:])
def test_every_report_carries_a_provenance_block(path: Path) -> None:
    """Each report states its script (or that it is hand-written) and its env."""
    key = path.relative_to(REPO).as_posix()
    assert key in REPORTS, (
        f"{key} has no entry in scripts/_common/provenance.py::REPORTS, so it is "
        "missing from report/README.md and gets no provenance header"
    )
    text = path.read_text(encoding="utf-8-sig")
    assert "<!-- provenance:start -->" in text, f"{key}: provenance header absent"
    assert "<!-- provenance:end -->" in text, f"{key}: provenance header unterminated"
    assert text.count("<!-- provenance:start -->") == 1, (
        f"{key}: stacked provenance headers -- the sync script is not idempotent"
    )
    assert "**生成脚本**" in text and "**运行环境**" in text, (
        f"{key}: header is missing the script or the environment field"
    )


@pytest.mark.parametrize("key", sorted(REPORTS), ids=lambda k: k[-55:])
def test_registry_script_paths_exist(key: str) -> None:
    """A registry entry naming a script that does not exist is worse than none."""
    prov = REPORTS[key]
    for script in (prov.script, *prov.extra_scripts):
        if not script:
            continue
        assert (REPO / script).is_file(), (
            f"{key} names {script}, which does not exist. Either the generator was "
            "renamed or the registry is stale."
        )


def test_every_generating_script_lives_under_scripts_or_tests() -> None:
    """The *generating* entry point must be a script or a test, never ``src/``.

    The report rule (AGENTS.md) puts markdown *generation* in ``scripts/``, so a
    ``src/`` primary script means the registry points at the code that computes
    the numbers rather than the code that writes the report. ``extra_scripts``
    are exempt: a legitimate one is a ``src/`` data producer that the renderer
    merely reads from.
    """
    for key, prov in REPORTS.items():
        if prov.script:
            assert prov.script.startswith(("scripts/", "tests/")), (
                f"{key}: generating script {prov.script} is not under scripts/ or "
                "tests/ -- report generation belongs in scripts/ (AGENTS.md)"
            )
        for extra in prov.extra_scripts:
            assert extra.startswith(("scripts/", "tests/", "src/")), (
                f"{key}: associated script {extra} is not a repo module path"
            )


@pytest.mark.parametrize("key", sorted(REPORTS), ids=lambda k: k[-55:])
def test_header_script_links_resolve(key: str) -> None:
    """The relative link in a header must point at a file that exists.

    Depth is supplied by the caller, so this also pins that a report nested more
    deeply than the registry author assumed gets a link matching its own depth.
    """
    path = REPO / key
    depth = len(path.relative_to(REPO).parts) - 1
    prov = REPORTS[key]
    for script in (prov.script, *prov.extra_scripts):
        if not script:
            continue
        href = script_link(script, depth).split("](")[1].rstrip(")")
        assert (path.parent / href).resolve().is_file(), (
            f"{key}: header link {href!r} does not resolve "
            f"(depth={depth}, script={script})"
        )


def test_commands_name_the_same_script_as_the_registry() -> None:
    """The repro command must invoke the declared script.

    A copy-paste slip here is invisible to a reader who trusts the command, and
    it is the field people actually paste.
    """
    for key, prov in REPORTS.items():
        if not prov.command:
            assert not prov.script, (
                f"{key}: declares script {prov.script} but no repro command"
            )
            continue
        assert prov.script in prov.command, (
            f"{key}: command {prov.command!r} does not invoke the declared script "
            f"{prov.script!r}"
        )


def test_index_links_resolve() -> None:
    """Every link in the generated ``report/README.md`` must resolve on disk."""
    index = REPORT_DIR / "README.md"
    assert index.is_file(), "report/README.md is missing -- run the sync script"
    broken = [h for h in _link_targets(index.read_text(encoding="utf-8-sig"))
              if not (index.parent / h).exists()]
    assert not broken, f"report/README.md links to missing files: {broken}"


@pytest.mark.parametrize(
    "path", _report_markdown(), ids=lambda p: p.as_posix()[-60:]
)
def test_report_relative_links_resolve(path: Path) -> None:
    """Relative links inside a report must still resolve after the move.

    The move preserved directory depth for almost every report, which is why the
    figure links kept working -- but a few changed depth
    (``micro_deformable_mirror/freq_test.md`` lost a level,
    ``centroid_test_visualization`` lost one), and those are exactly the ones a
    blanket rename silently breaks.
    """
    key = path.relative_to(REPO).as_posix()
    broken = [
        h for h in _link_targets(path.read_text(encoding="utf-8-sig"))
        if not (path.parent / h).exists()
    ]
    allowed = _PREEXISTING_BROKEN.get(key)
    if allowed is None:
        assert not broken, f"{key}: broken relative links {broken}"
        return
    unexpected = [h for h in broken if h not in allowed]
    assert not unexpected, (
        f"{key}: new broken links {unexpected} — only the baselined ones in "
        "_PREEXISTING_BROKEN are tolerated"
    )


def test_the_preexisting_baseline_is_still_justified() -> None:
    """Each baselined link needs a real reason, and must still be broken.

    The second half matters as much as the first: if someone re-attaches the
    missing screenshots, the entry must be removed rather than left guarding
    nothing, which is how a baseline rots into a no-op.
    """
    for key, links in _PREEXISTING_BROKEN.items():
        path = REPO / key
        assert path.is_file(), f"{key} is baselined but no longer exists"
        broken = {
            h for h in _link_targets(path.read_text(encoding="utf-8-sig"))
            if not (path.parent / h).exists()
        }
        for href, reason in links.items():
            assert reason and not reason.startswith("TBD"), f"{key} {href}: no reason"
            assert href in broken, (
                f"{key}: {href} is baselined as broken but now resolves — remove the "
                "entry so new breakage in this report is caught again"
            )
