"""Golden harness for generate_hwdataset_corpus_report.build_report.

Runs collect() over the local corpus (sub-second) and prints
build_report(facts, None) -- the degraded arm (no dist_stats figures).
Diff before/after the section split.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "scripts" / "ml" / "generate_hwdataset_corpus_report.py"
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    spec = importlib.util.spec_from_file_location("_hwrep", GEN)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_hwrep"] = module
    spec.loader.exec_module(module)
    facts = module.collect()
    print(module.build_report(facts, None))


if __name__ == "__main__":
    main()
