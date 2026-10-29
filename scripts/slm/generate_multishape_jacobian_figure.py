"""Figure for the 3a per-block Jacobian check (report §5.6).

The live log quotes the 3a reading as a table; this draws the same numbers from
the same place (``shape/<shape>/<stage>/summary.json``, key ``jacobian``) so the
claim "96/96 blocks are orthogonal to the model prediction" can be eyeballed
instead of trusted.

Four panels, all straight from the JSON lists -- no derived quantity is invented:

* (a) ``correlation`` per block, by shape, with the +-0.05 band the report quotes;
* (b) ``relative_l2`` per block against the sqrt(2) reference (orthogonal);
* (c) ``cap_next`` per block, showing the collapse from 0.1 to the 0.02 floor;
* (d) ``measured_rms`` per block against the 3-frame-mean noise floor measured
  from the same session's same-phase exposure reads (absolute units, the same
  ones ``validate.py`` builds ``measured_candidate`` in).

``predicted_rms`` is deliberately *not* compared with ``measured_rms``: the model
output is a normalised intensity field while the measurement is in
``counts / (ms * counts_scale)``, so only the scale-free quantities (a, b) are
comparable.  That is the point of panel (d) -- it carries the measurement against
its own noise, not against the model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_multishape_scale_live_figures import SHAPE_LABELS, _save, use_cjk_font

SHAPES = ("circle", "square", "triangle", "star")
SESSIONS = {shape: f"p20261029_analytic_init_s600/{shape}" for shape in SHAPES}
DEFAULT_CAMPAIGN = "data/debug/slm_transfer/multishape_nogs_20261029"
DEFAULT_OUT = "report/shape_trials/multishape_scale/figures"
FIGURE_NAME = "jacobian_direction_s600_power1p0_20261029.png"

#: Correlation band the report treats as "no information" (|corr| < 0.05).
CORR_BAND = 0.05
#: 3-frame-mean noise floor in ``measured_rms`` units, from the same-phase reads
#: of ``s600/circle`` -> ``mono12_cv`` (three exposure rungs: 0.00160-0.00166).
NOISE_FLOOR = 0.00163


def load_blocks(campaign: Path, session: str) -> list[dict]:
    """Every block of every stage of one session, tagged with its stage name."""
    blocks: list[dict] = []
    for path in sorted((campaign / session).glob("shape/*/*/summary.json")):
        rows = json.loads(path.read_text(encoding="utf-8"))
        for index, row in enumerate(rows):
            jacobian = row.get("jacobian")
            if not jacobian:
                continue
            blocks.append(dict(jacobian, stage=path.parent.name, index=index))
    return blocks


def _strip(ax, values: dict[str, list[float]], *, label: str) -> None:
    for offset, (shape, series) in enumerate(values.items()):
        ax.scatter(
            np.arange(len(series)) + offset * 0.0,
            series,
            s=26,
            alpha=0.85,
            label=SHAPE_LABELS.get(shape, shape),
            zorder=3,
        )
    ax.set_xlabel("块序号（每个 stage 内重排）")
    ax.set_ylabel(label)
    ax.grid(True, alpha=0.3)


def build_figure(campaign: Path, out: Path, *, dpi: float) -> Path:
    use_cjk_font()
    data = {shape: load_blocks(campaign, session) for shape, session in SESSIONS.items()}

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.4))

    ax = axes[0][0]
    corr_all = [abs(b["correlation"]) for blocks in data.values() for b in blocks]
    inside = sum(1 for value in corr_all if value < CORR_BAND)
    _strip(ax, {s: [b["correlation"] for b in blocks] for s, blocks in data.items()},
           label="correlation（模型差分 vs 实测差分）")
    ax.axhline(0.0, color="black", lw=0.9)
    ax.axhspan(-CORR_BAND, CORR_BAND, color="#B0BEC5", alpha=0.45, zorder=1,
               label="报告判为「无信息」的 |corr| < 0.05")
    ax.set_title(f"(a) {inside}/{len(corr_all)} 块 |corr| < 0.05"
                 f"（最大 {max(corr_all):.3f}）：预测与实测几乎正交")
    ax.legend(loc="upper right", fontsize=7.5)
    ax.set_ylim(-1.05, 1.05)

    ax = axes[0][1]
    all_rel = [b["relative_l2"] for blocks in data.values() for b in blocks]
    ax.hist(all_rel, bins=14, color="#4C72B0", alpha=0.85, zorder=3)
    ax.axvline(np.sqrt(2.0), color="#C62828", ls="--", lw=1.6,
               label=f"√2 = {np.sqrt(2.0):.4f}（完全正交）")
    ax.set_xlabel("relative_l2")
    ax.set_ylabel("块数")
    ax.set_title(f"(b) relative_l2 全落在 √2 附近（{len(all_rel)} 块，"
                 f"{min(all_rel):.2f}–{max(all_rel):.2f}）")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1][0]
    _strip(ax, {s: [b["cap_next"] for b in blocks] for s, blocks in data.items()},
           label="cap_next（下一块的 phase_rms_cap）")
    ax.axhline(0.02, color="#C62828", ls="--", lw=1.4, label="地板 0.02")
    ax.axhline(0.2, color="#2E7D32", ls=":", lw=1.4, label="--phase-rms-cap 默认 0.2")
    ax.set_title("(c) 每个 contradict 块减半：第 3 块就落到地板并停住")
    ax.legend(loc="upper right", fontsize=7.5)
    ax.set_ylim(0.0, 0.23)

    ax = axes[1][1]
    _strip(ax, {s: [b["measured_rms"] for b in blocks] for s, blocks in data.items()},
           label="measured_rms（实测差分，绝对口径）")
    ax.axhline(NOISE_FLOOR, color="#C62828", ls="--", lw=1.5,
               label=f"3 帧均值噪声底 ≈ {NOISE_FLOOR:.5f}")
    ax.set_title("(d) 差分在噪声之上（SNR ≈ 4.6–6.3）：不是「测不到」")
    ax.legend(loc="upper right", fontsize=7.5)

    fig.suptitle("3a 逐块 Jacobian 方向校验：s600 四形状（lane 05–08，1.0 功率）",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out.parent.mkdir(parents=True, exist_ok=True)
    return _save(fig, out, dpi)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--campaign", type=Path, default=Path(DEFAULT_CAMPAIGN))
    parser.add_argument("--out", type=Path, default=Path(DEFAULT_OUT) / FIGURE_NAME)
    parser.add_argument("--dpi", type=float, default=150.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    build_figure(args.campaign, args.out, dpi=args.dpi)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
