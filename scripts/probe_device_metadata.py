"""Does the corpus carry ANY device metadata worth conditioning on?

The request is to fuse camera exposure and other device parameters into the forward model.
Before building that arm it is worth checking whether the parameter actually varies: a
constant input cannot inform a prediction, and adding it would only add parameters and a
false sense of progress.

Checked per family, because a value that varies across the corpus can still be constant
*within* the family being modelled -- which is the only place it would be used.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(r"D:\Projects\TIFO\AO-shaping")
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from ml.hwdataset import (  # noqa: E402
    HwPhaseImageDataset,
    MaterialiserConfig,
    build_hw_index,
)
from ml.zernike import inverse_design as inv  # noqa: E402

PROBE = 40


def main() -> None:
    index = build_hw_index(index_cache="data/hw_index_cache.json", progress_every=0)
    dataset = HwPhaseImageDataset(
        index, config=MaterialiserConfig(grid=inv.GRID), use_cache=True
    )

    print("=" * 104)
    print("DEVICE METADATA: does anything actually vary where it would be used?")
    print("=" * 104)
    print(f"{'family':<30}{'n':>7}{'exposure values':>34}{'varies?':>9}")

    rows = []
    for family in sorted({r.family for r in index.records}):
        try:
            sub = index.filter(families=[family])
            ds = HwPhaseImageDataset(sub, config=MaterialiserConfig(grid=inv.GRID), use_cache=True)
            n = len(ds)
            if n == 0:
                continue
            exposures = sorted({round(float(ds[i]["exposure_ms"]), 4) for i in range(min(n, PROBE))})
            shown = exposures if len(exposures) <= 4 else [exposures[0], "...", exposures[-1]]
            varies = len(exposures) > 1
            rows.append((family, n, exposures, varies))
            print(f"{family:<30}{n:>7}{str(shown):>34}{'YES' if varies else 'no':>9}")
        except Exception as exc:  # noqa: BLE001 - a family may have no usable payload
            print(f"{family:<30}{'--':>7}{type(exc).__name__:>34}{'?':>9}")

    print("\nreading:")
    constant = [r for r in rows if not r[3]]
    print(f"  {len(constant)}/{len(rows)} families have a CONSTANT exposure over the probed "
          f"records.")
    print("  A constant input carries zero information: it cannot inform a prediction, so")
    print("  conditioning on it would add parameters and a false sense of progress.")
    if all(not r[3] for r in rows):
        print("\n  Exposure fusion is therefore NOT built as an arm -- there is nothing to fuse.")
        print("  Recorded here so the negative is on file rather than silently skipped.")


if __name__ == "__main__":
    main()
