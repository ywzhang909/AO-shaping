"""Serialise the beam-shaping benchmark to CSV / markdown / GIF.

This is the report writer for
:mod:`ao_shaping.algorithm.signal_processing.beam_shaping_benchmark`, and it
lives here rather than in ``algorithm/`` because AGENTS.md is explicit:

    "report generation MUST live in `scripts/`, not in the algorithm layer"

The split follows the repo's own convention for the ``algorithm`` package: the
algorithm layer computes, the caller renders. So
``beam_shaping_benchmark`` keeps ``run_benchmark`` / ``run_benchmark_suite``
pure -- they no longer take an ``output_dir`` at all -- while the two pure
helpers they hand off to live with them:

* ``build_gif_frames(target, simulated)`` builds PIL frames in memory. Frame
  construction is computation, not I/O, so it belongs with the producer.
* ``to_dataframe(rows)`` is the ``DataFrame`` projection that
  ``run_benchmark_suite`` *returns*; the returned type is part of the API.
* ``HPRINT_KEYS`` is the column contract, so it also stays with the producer.

Everything that touches the filesystem is here.

Note the naming trap: ``slm_shaping_bench.strehl()`` is a **mean-removed cosine
similarity**, not a physical Strehl ratio (see ``report/zotero_objectives``). The
report does not report a Strehl column, which is why that conflict cannot leak
into this table.

**Usage**::

    python scripts/generate_beam_shaping_benchmark_report.py --out-dir report/benchmarks/device_less_full

**Fully offline** -- pure numpy/PIL, no hardware, no device.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ao_shaping.algorithm.signal_processing.beam_shaping_benchmark import (  # noqa: E402
    DEFAULT_MAX_FRAMES,
    HPRINT_KEYS,
    build_gif_frames,
    run_benchmark,
    run_benchmark_suite,
)

DEFAULT_OUT = ROOT / "report" / "benchmarks" / "device_less_full"


def write_table(df: pd.DataFrame, output_dir: str | Path) -> Path:
    """Write the combined metrics CSV + markdown table. Returns the directory."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "beam_shaping_benchmark_metrics.csv", index=False)

    met = df.groupby("algorithm")["area_met"].sum().to_dict() if not df.empty else {}
    total = df.groupby("algorithm").size().to_dict() if not df.empty else {}
    tally = ", ".join(f"{a} {int(met.get(a, 0))}/{int(total.get(a, 0))}" for a in sorted(total))
    lines = [
        "# Beam-Shaping Benchmark (simulation)",
        "",
        "> **Read `area_met` before comparing algorithms.** It is the only column "
        "that says whether the algorithm produced the requested target *at all*; "
        "`uniformity_cv` and `fill_ratio` on a row that failed it describe noise.",
        f"> Area check passed, per algorithm: {tally}.",
        "> `spgd-sim` currently fails every shape (measured area collapses to ~1 px, "
        "CV 16-34), so its uniformity numbers are meaningless device-less -- the "
        "documented failure, not a regression. `backprop` is hardware-class: it "
        "passes square and fails the other two.",
        ">",
        "> ⚠️ `slm_shaping_bench.strehl()` is a mean-removed cosine similarity, not a "
        "physical Strehl ratio. No Strehl column appears here for that reason.",
        ">",
        f"- Grid: {df['shape'].count() if not df.empty else 0} cells "
        "(3 algorithms x 3 shapes)",
        f"- Regenerate: `python scripts/generate_beam_shaping_benchmark_report.py` "
        "(fully offline, ~100 s)",
        "> The sibling `.csv` is written too but is **not committed** -- the repo has a "
        "global `*.csv` ignore rule and no CSV under `docs/` is tracked. Re-run to "
        "get it; this markdown is the tracked artefact.",
        "> The sibling `.csv` is written too but is **not committed** -- the repo has a "
        "global `*.csv` ignore rule and no CSV under `docs/` is tracked. Re-run to "
        "get it; this markdown is the tracked artefact.",
        "> The sibling `.csv` is written too but is **not committed** -- the repo has a "
        "global `*.csv` ignore rule and no CSV under `docs/` is tracked. Re-run to "
        "get it; this markdown is the tracked artefact.",
        "",
        "| algorithm | shape | requested | measured | area_met | fill_ratio | "
        "uniformity_cv | encircled_energy | elapsed_s |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"| {row['algorithm']} | {row['shape']} | {row['requested_area']} "
            f"| {row['measured_area']} | {row['area_met']} "
            f"| {row['fill_ratio']:.3f} | {row['uniformity_cv']:.3f} "
            f"| {row['encircled_energy']:.3f} | {row['elapsed_s']:.3f} |"
        )
    lines.append("")
    (out / "beam_shaping_benchmark_metrics.md").write_text("\n".join(lines), encoding="utf-8")
    return out


def write_artifacts(
    result: dict, output_dir: str | Path, *, max_frames: int = DEFAULT_MAX_FRAMES
) -> Path:
    """Write one run's metrics CSV/MD and evolution GIF. Returns the directory."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    header = list(HPRINT_KEYS)
    with (out / f"{result['algorithm']}_{result['shape']}_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fh:
        import csv

        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerow([result.get(k, "") for k in header])

    lines = [
        f"# Benchmark: {result['algorithm']} × {result['shape']}",
        "",
        f"- requested_area: {result['requested_area']}",
        f"- measured_area: {result['measured_area']}  (met: {result['area_met']}, "
        f"fill_ratio: {result['fill_ratio']:.3f})",
        f"- uniformity_cv: {result['uniformity_cv']:.4f}",
        f"- encircled_energy: {result['encircled_energy']:.4f}",
        f"- elapsed_s: {result['elapsed_s']:.3f}",
        "",
    ]
    (out / f"{result['algorithm']}_{result['shape']}_metrics.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    # GIF: target -> simulated, over a fixed frame budget.
    target = np.asarray(result["target"])
    simulated = np.asarray(result["simulated"])
    if target.ndim == 2 and simulated.ndim == 2 and target.shape == simulated.shape:
        frames = build_gif_frames(target, simulated, max_frames=max_frames)
        gif_path = out / f"{result['algorithm']}_{result['shape']}_evolution.gif"
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=1000 / 15,
            loop=0,
            optimize=True,
        )
    return out


def main(out_dir: Path | None = None, *, iterations: int = 300, gifs: bool = True) -> int:
    """Regenerate the full device-less benchmark. No hardware involved."""
    out = Path(out_dir) if out_dir else DEFAULT_OUT
    out.mkdir(parents=True, exist_ok=True)

    rows, df = run_benchmark_suite(iterations=iterations)
    write_table(df, out)
    (out / "suite_stdout.txt").write_text(df.to_string(), encoding="utf-8")
    print(f"suite rows: {len(rows)}")

    if gifs:
        gif_dir = out / "gif"
        for algo in ("gs", "spgd-sim"):
            for shape in ("square", "circle", "gaussian"):
                sub = gif_dir / f"{algo}_{shape}"
                result = run_benchmark(
                    algorithm=algo, shape=shape,
                    iterations=iterations, seed=42, max_frames=30, device=None,
                )
                write_artifacts(result, sub, max_frames=30)
                n = len(list(sub.glob("*.gif")))
                print(
                    f"  gif {algo:9s} {shape:8s} fill={result['fill_ratio']:.3f} "
                    f"area_met={result['area_met']} gifs={n}"
                )
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
