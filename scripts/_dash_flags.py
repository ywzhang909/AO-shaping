#!/usr/bin/env python3
"""Rewrite underscore CLI flags to dash form across the repository.

Flags are matched as whole tokens so a multi-segment name like
``--exposure-time_ms`` is rewritten correctly -- a naive ``s/--x_y/--x-y/``
pattern silently misses any flag whose underscore is not in the first segment.

Two things are deliberately NOT touched:

* **Attribute/dict access.** ``config.cam_type`` and ``args.exposure_time_ms``
  stay snake_case; argparse derives the dest from the long flag, so the
  attribute is unaffected by the flag rename.
* **Module docstrings of the guard test itself**, which must keep quoting the
  old spelling to document what it rejects.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]

#: Matches a whole flag token and captures each underscore-bearing tail.
FLAG = re.compile(r"--([a-z0-9]+((?:-[a-z0-9]+)*_[a-z0-9_-]+))")

#: Test modules whose *docstrings* quote the old spelling on purpose.
PROSE_KEEP = {
    "tests/ao_shaping/runners/test_cli_flag_naming.py",
}

SUFFIXES = {".py", ".md", ".json", ".sh", ".yaml", ".yml", ".toml", ".txt"}


def rewrite(text: str) -> tuple[str, int]:
    count = 0

    def sub(match: re.Match[str]) -> str:
        nonlocal count
        token = match.group(0)
        fixed = "--" + match.group(1).replace("_", "-")
        if fixed != token:
            count += 1
        return fixed

    return FLAG.sub(sub, text), count


def main(argv: list[str]) -> int:
    targets = argv[1:] or ["src", "tests", "scripts", "docs", "report", "README.md", "AGENTS.md", "TODO.md"]
    changed: list[tuple[str, int]] = []
    for target in targets:
        path = REPO / target
        files = [path] if path.is_file() else sorted(path.rglob("*"))
        for file_path in files:
            if file_path.suffix not in SUFFIXES or not file_path.is_file():
                continue
            rel = file_path.relative_to(REPO).as_posix()
            if rel in PROSE_KEEP:
                continue
            original = file_path.read_text(encoding="utf-8-sig")
            updated, n = rewrite(original)
            if n:
                file_path.write_text(updated, encoding="utf-8")
                changed.append((rel, n))

    for rel, n in sorted(changed):
        print(f"{n:4d}  {rel}")
    print(f"\n{sum(n for _, n in changed)} replacements in {len(changed)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))