"""The Zernike amp report's glossary appendix must be emitted, not hand-pasted.

``scripts/ml/generate_zernike_amp_report.py`` writes ``report.md`` wholesale, so any section it
does not emit disappears the next time the report is regenerated. The glossary appendix existed
only inside the committed artifact (TODO 8.9 item 27 caught one row of it, the ``piston`` row).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
REPORT = REPO / "report/zernike_phase2amp/zernike_amp_report/report.md"
GENERATOR = REPO / "scripts/ml/generate_zernike_amp_report.py"

HEADING = "## 附录 A: 名词解释 (Glossary)"


@pytest.fixture(scope="module")
def appendix() -> str:
    from scripts.ml import generate_zernike_amp_report as gen

    return gen.GLOSSARY_APPENDIX


def test_the_committed_report_ends_with_exactly_the_emitted_appendix(appendix: str) -> None:
    """Regenerating the report reproduces the committed file's tail byte for byte."""
    text = REPORT.read_text(encoding="utf-8")
    assert text.endswith(appendix), (
        "the committed report's tail is not the appendix the generator emits, so regenerating "
        f"it would rewrite {len(appendix)} bytes of glossary. Generator text and artifact have "
        "drifted -- move the change into GLOSSARY_APPENDIX, never into the .md."
    )


def test_the_appendix_is_a_real_section_not_an_empty_stub(appendix: str) -> None:
    """Anti-vacuity: the pinned tail must be the glossary, not a truncation of it."""
    assert len(appendix) > 1500, f"appendix is only {len(appendix)} bytes -- too short to be the glossary"
    assert HEADING in appendix, "the pinned tail does not contain the glossary heading"
    assert appendix.startswith("\n---\n\n" + HEADING), (
        "the appendix must carry its own separator: the generated body ends at a code fence, "
        "so the rule before the heading is part of the emitted text"
    )


def test_every_glossary_row_survives_into_the_generator(appendix: str) -> None:
    """Each row of the artifact's glossary table exists in the emitted text, pipes included."""
    artifact = REPORT.read_text(encoding="utf-8")
    table = artifact[artifact.index(HEADING) :]
    rows = [line for line in table.splitlines() if line.startswith("| **")]
    assert len(rows) >= 10, f"only {len(rows)} glossary rows found -- the artifact changed shape"
    missing = [r.split("|")[1].strip() for r in rows if r not in appendix]
    assert not missing, (
        f"{len(missing)} glossary row(s) are in the artifact but not in what the generator "
        f"emits: {missing}. The generator, not the .md, is the source of this file."
    )


#: A section that exists only in the committed artifact, i.e. regeneration deletes it. Its
#: splice point is mid-document, and verifying a mid-document splice needs the generator to
#: actually run -- which it cannot, because its inputs are gitignored (``logs*``, TODO 8.9
#: item 33). So it is recorded instead of guessed at. A ratchet: it may shrink, never grow.
KNOWN_UNEMITTED_HEADINGS = {"### U-Net 对比模型"}


def test_no_generated_section_lives_only_in_the_artifact() -> None:
    """Every section heading in the artifact is either emitted by the generator or baselined.

    Heading presence is only a proxy -- data-composed headings like ``### 6.1 `none__turb-none```
    never appear as literals -- which is why this is pinned per report pair rather than run
    repo-wide, where it reported 346 bogus orphans across 85 reports.
    """
    artifact = REPORT.read_text(encoding="utf-8")
    heads = [ln.strip() for ln in artifact.splitlines() if re.match(r"^#{2,4} ", ln)]
    assert len(heads) >= 20, f"only {len(heads)} heading(s) scanned -- the artifact changed shape"
    unemitted = {h for h in heads if h not in GENERATOR.read_text(encoding="utf-8")}
    assert unemitted == KNOWN_UNEMITTED_HEADINGS, (
        f"sections nobody emits -- new: {sorted(unemitted - KNOWN_UNEMITTED_HEADINGS)}, "
        f"resolved: {sorted(KNOWN_UNEMITTED_HEADINGS - unemitted)}. Pasting a section into "
        "report.md loses it on the next regeneration; move it into the generator and drop it "
        "from the baseline."
    )


def test_the_generator_appends_the_constant_at_the_write_site() -> None:
    """The constant must reach the file; an unused constant reproduces the original defect."""
    src = GENERATOR.read_text(encoding="utf-8")
    site = re.search(r'write_text\(\s*(text \+ GLOSSARY_APPENDIX|GLOSSARY_APPENDIX)', src)
    assert site, (
        "the write site does not append GLOSSARY_APPENDIX, so a regenerated report still "
        "deletes the glossary"
    )
