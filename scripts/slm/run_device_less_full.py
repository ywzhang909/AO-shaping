"""Device-less full-grid benchmark driver (thin wrapper).

The report writer now lives in
``scripts/slm/generate_beam_shaping_benchmark_report.py`` (F-14: report generation
must live in ``scripts/``, not in ``algorithm/``). This script is kept because
scripts/README.md documents it, and it only forwards to that writer.

Produces under report/benchmarks/device_less_full/:
  - beam_shaping_benchmark_metrics.csv / .md  (9-cell grid)
  - gif/ 6 evolution GIFs (gs / spgd-sim x square/circle/gaussian)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# `scripts.*` is imported as a package path, so the repo ROOT itself must be
# importable -- under a file-form launch (`python scripts/x/y.py`) the cwd is
# not on sys.path and only the script's own directory is.
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.slm.generate_beam_shaping_benchmark_report import main as generate

OUT = Path(__file__).resolve().parents[2] / "report" / "benchmarks" / "device_less_full"
GIF = OUT / "gif"


def main() -> int:
    """Forward to the canonical writer."""
    return generate()


if __name__ == "__main__":
    raise SystemExit(main())
