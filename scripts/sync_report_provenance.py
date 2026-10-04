"""Insert / refresh the provenance header in every report under ``report/``.

Every report answers "which script produced this, and what do I need to run it?"
through a header rendered from :mod:`scripts._common.provenance`. That header is
**generated**, not hand-written, for the same reason the numbers in a report are:
hand-maintained provenance drifts the moment a generator is renamed, which is
exactly the failure this migration had to clean up.

So this script owns those headers, plus ``report/README.md`` (the index). It is
idempotent -- running it twice changes nothing -- and it is the *only* thing that
should edit either.

    python scripts/sync_report_provenance.py            # headers + index
    python scripts/sync_report_provenance.py --check    # verify only, exit 1 on drift
    python scripts/sync_report_provenance.py --verbose  # list every file touched

Why not have each generator emit its own header instead? Because that couples the
header's wording to ~25 heterogeneous markdown emitters, and a generator that
drifts silently loses its provenance. Here there is one emitter, a registry, and a
check mode; ``tests/ao_shaping/scripts/test_report_provenance.py`` runs the check
in CI-shaped form.

Reports absent from ``REPORTS`` are reported as unregistered and left untouched --
that is a registry omission to fix, not a file to mangle.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._common.provenance import (  # noqa: E402
    END_MARKER,
    REPORTS,
    START_MARKER,
    _strip_header,
    insert_header,
    render_index,
)

START = START_MARKER
END = END_MARKER
REPORT_DIR = ROOT / "report"
INDEX = REPORT_DIR / "README.md"


def _read(path: Path) -> tuple[str, bool]:
    """Read ``path`` returning ``(text, had_bom)``, newlines normalised to LF.

    Two normalisations are load-bearing, and both were found by a bug:

    * the **BOM**. A UTF-8-BOM file's first line is ``\\ufeff# Title``, so a naive
      ``startswith("# ")`` misses the H1 and the header lands nowhere.
    * the **newlines**. ``Path.write_text`` translates ``\\n`` to CRLF on Windows,
      while a raw byte decode does not. Comparing the two would report drift on a
      file that is byte-for-byte correct after line-ending translation -- and
      ``str.splitlines()`` hides it, so a diff of the two texts looks empty.
      Normalising here makes the check independent of how the writer opened the
      file.
    """
    raw = path.read_bytes()
    had_bom = raw.startswith(b"\xef\xbb\xbf")
    text = raw.decode("utf-8-sig")
    return text.replace("\r\n", "\n"), had_bom


def _write(path: Path, text: str, had_bom: bool) -> None:
    data = text.encode("utf-8")
    path.write_bytes(b"\xef\xbb\xbf" + data if had_bom else data)


def _strip_existing(text: str) -> str:
    """Kept for callers that only need the block removed; placement lives in
    :func:`scripts._common.provenance.insert_header`."""
    return _strip_header(text)


def _depth(path: Path) -> int:
    """Directory depth of ``path`` below the repo root (report/x/y.md -> 2)."""
    return len(path.relative_to(ROOT).parts) - 1


def process(check: bool, verbose: bool) -> int:
    drift: list[str] = []
    unregistered: list[str] = []
    touched = 0

    for md in sorted(REPORT_DIR.rglob("*.md")):
        key = md.relative_to(ROOT).as_posix()
        if key == INDEX.relative_to(ROOT).as_posix():
            continue
        if key not in REPORTS:
            unregistered.append(key)
            continue
        original, had_bom = _read(md)
        updated = insert_header(original, key)
        if updated is None:
            drift.append(f"{key}: no heading to anchor the header")
            continue
        if updated != original:
            touched += 1
            if verbose:
                print(f"  header: {key}")
            if check:
                drift.append(f"{key}: provenance header is missing or stale")
            else:
                _write(md, updated, had_bom)

    index_body = render_index()
    if INDEX.exists() and INDEX.read_text(encoding="utf-8") == index_body:
        pass
    else:
        if verbose:
            print("  index:  report/README.md")
        if not check:
            INDEX.write_text(index_body, encoding="utf-8")
        if check:
            drift.append("report/README.md: index is stale")

    print(f"registered reports: {len(REPORTS)}")
    print(f"headers updated:    {touched}")
    if unregistered:
        print(f"UNREGISTERED ({len(unregistered)}) -- add them to scripts/_common/provenance.py:")
        for key in unregistered:
            print(f"  - {key}")
    if drift:
        print("DRIFT:")
        for line in drift:
            print(f"  ! {line}")
        return 1
    if check:
        print("check: clean")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="verify without writing; exit 1 on drift")
    ap.add_argument("--verbose", action="store_true", help="print every file touched")
    args = ap.parse_args()
    return process(check=args.check, verbose=args.verbose)


if __name__ == "__main__":
    raise SystemExit(main())
