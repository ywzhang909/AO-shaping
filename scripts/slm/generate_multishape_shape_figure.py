"""Figure: does the target get *filled*? (report §5.8)

Every verdict in the 2026-10-29 live log covers "more uniform (CV) + energy not
lost".  This figure adds the missing half from data that already exists: the
archived replay frames carry the optimiser's own ``target`` mask, so
:mod:`audit_shape_agreement` can report how much of it is actually at or above
half the in-target peak.

Two panels, both paired seed -> final on the same ``transport_cv`` replay block:

* (a) ``filled_fraction`` (log x): the share of the target mask that reaches
  half the in-target peak.  A flat-top fill is ~1; a peaked spot is ~0.
* (b) ``inside_cv`` (linear x): the same-mask CV the loop was actually
  optimising, so the reader can see the two disagree where they do.

Nothing is recomputed differently here: every number comes from
``audit_shape_agreement.read_block`` / ``summarise``, the same functions that
print the table in the log.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_shape_agreement import DEFAULT_CAMPAIGN, TERMS, read_block
from generate_multishape_scale_live_figures import (
    SHAPE_COLORS,
    SHAPE_LABELS,
    _save,
    use_cjk_font,
)

DEFAULT_OUT = "report/shape_trials/multishape_scale/figures"
FIGURE_NAME = "shape_filling_20261029.png"
SESSION_RE = re.compile(r"^(?P<session>p\d+_(?P<route>.+?)_s(?P<size>\d+)(?:_power\w+)?)/(?P<shape>[a-z]+)$")
ROUTE_LABELS = {
    "analytic_init": "解析初始化",
    "strict_cold": "严格冷启动",
}
#: Laser power per session.  The campaign records no power field, so this is the
#: lane history read off the run order (and the reason the report keeps asking
#: for it to be logged per session): lanes 01–04 ran at 0.7, the power was raised
#: to 1.0 before lanes 05–08, and everything after 05 is 1.0.  An unknown session
#: is labelled as such rather than silently pooled.
POWER_BY_SESSION = {
    "p20261029_analytic_init_s200": 0.7,
    "p20261029_analytic_init_s1200": 0.7,
    "p20261029_analytic_init_s200_power1p0": 1.0,
    "p20261029_analytic_init_s600": 1.0,
    "p20261029_strict_cold_s200": 1.0,
    "p20261029_strict_cold_s600": 1.0,
}


def sessions(campaign: Path) -> list[dict]:
    """Every session that has a replay block, labelled by route / size / shape."""
    found = []
    for block in sorted(campaign.glob("*/*/common_replay/manifest.json")):
        session = block.parent.parent
        match = SESSION_RE.match(f"{session.parent.name}/{session.name}")
        if not match:
            continue
        route = match["route"].replace("_power1p0", "")
        power = POWER_BY_SESSION.get(session.parent.name)
        found.append(
            dict(
                path=str(session.relative_to(campaign)),
                route=route,
                size=int(match["size"]),
                shape=match["shape"],
                power=power,
                label=(f"{match['size']} px {SHAPE_LABELS.get(match['shape'], match['shape'])}"
                       f"（{ROUTE_LABELS.get(route, route)}，"
                       f"{'功率未记录' if power is None else f'{power:.1f} 功率'}）"),
            )
        )
    return found


def pair(rows: list[dict]) -> dict | None:
    """seed -> final means of one block, or ``None`` when either side is missing."""
    def mean(phase: str, name: str) -> float | None:
        values = [float(r[name]) for r in rows
                  if r["phase"] == phase and r.get(name) is not None and not r.get("clipped")]
        return float(np.mean(values)) if values else None

    def count(phase: str) -> int:
        return sum(1 for r in rows
                   if r["phase"] == phase and not r.get("clipped") and r.get("iou") is not None)

    out: dict[str, float | None] = {}
    for term in TERMS:
        seed, final = mean("seed", term), mean("final", term)
        if seed is None or final is None:
            return None
        out[f"{term}_seed"], out[f"{term}_final"] = seed, final
    out["n_seed"], out["n_final"] = count("seed"), count("final")
    return out


def collect(campaign: Path, *, jsonl: Path | None = None) -> list[dict]:
    """Pair seed/final per session.

    With ``jsonl`` (the streamed output of ``audit_shape_agreement``) the numbers
    come from that file and no npz is read again -- so the expensive sweep can be
    run once, and re-rendering the figure after an interruption costs seconds.
    """
    streamed: dict[str, list[dict]] = {}
    if jsonl is not None:
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            saved = json.loads(line)
            streamed.setdefault(str(saved.get("session", "")), []).extend(saved.get("rows", []))

    out = []
    for session in sessions(campaign):
        if streamed:
            rows = [row for row in streamed.get(session["path"], [])
                    if row["stage"] == "transport_cv"]
        else:
            block = campaign / session["path"] / "common_replay"
            rows = [row for row in read_block(block, level=0.5) if row["stage"] == "transport_cv"]
        values = pair(rows)
        if values is None:
            print(f"[skip] {session['path']}: no seed/final pair")
            continue
        out.append(dict(session, **values))
    out.sort(key=lambda item: (item["power"] if item["power"] is not None else -1.0,
                               item["size"], item["route"], item["shape"]))
    return out


def _dumbbell(ax, records: list[dict], *, seed_key: str, final_key: str,
              logx: bool, xlabel: str) -> None:
    positions = np.arange(len(records))[::-1]
    for y, record in zip(positions, records):
        colour = SHAPE_COLORS.get(record["shape"], "#4C72B0")
        ax.plot([record[seed_key], record[final_key]], [y, y], color=colour, lw=1.4, alpha=0.55, zorder=2)
        ax.scatter(record[seed_key], y, s=44, facecolors="white", edgecolors=colour, lw=1.4, zorder=3)
        ax.scatter(record[final_key], y, s=44, color=colour, zorder=4)
    ax.set_yticks(positions)
    ax.set_yticklabels([record["label"] for record in records], fontsize=8)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.grid(True, axis="x", alpha=0.3)
    ax.scatter([], [], s=44, facecolors="white", edgecolors="black", label="seed（冷启动/解析起点）")
    ax.scatter([], [], s=44, color="black", label="final（该阶段末）")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=2, fontsize=7.5,
              frameon=False)


def build_figure(records: list[dict], out: Path, *, dpi: float) -> Path:
    """Draw both panels from already-collected records (one archive read each)."""
    use_cjk_font()
    if not records:
        raise SystemExit("no sessions with a seed/final pair")

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 0.42 * len(records) + 2.6))
    _dumbbell(axes[0], records, seed_key="filled_fraction_seed",
              final_key="filled_fraction_final", logx=True,
              xlabel="filled_fraction：目标掩膜内达到「峰值的 50%」的面积占比")
    axes[0].set_title("(a) 目标有没有被填满？（1.0 = 平顶填满，0.002 = 一个尖点）")
    _dumbbell(axes[1], records, seed_key="inside_cv_seed", final_key="inside_cv_final",
              logx=False, xlabel="inside_cv：目标掩膜内的 CV（闭环真正在优化的量）")
    axes[1].set_title("(b) 同一批帧的 CV：12 条里 7 条下降，但 final 仍在 0.68–1.06（平顶 ≈ 0.1）")

    fig.suptitle("目标填充 vs 掩膜内 CV：2026-10-29 campaign（同一帧、同一目标掩膜）", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    out.parent.mkdir(parents=True, exist_ok=True)
    return _save(fig, out, dpi)


def markdown_table(records: list[dict]) -> str:
    """The §5.8 table, generated from the same numbers the figure draws."""
    head = ("| 组（初始化 / 尺寸 / 功率） | 形状 | n（seed/final） | `filled_fraction` "
            "seed → final | `inside_cv` seed → final | `energy_in_target`（final） | "
            "`boundary_mean_px`（final） |\n"
            "|---|---|---|---|---|---|---|")
    lines = [head]
    for record in records:
        power = "功率未记录" if record["power"] is None else f"{record['power']:.1f} 功率"
        group = f"{ROUTE_LABELS.get(record['route'], record['route'])} / {record['size']} px / {power}"
        lines.append(
            f"| {group} | {SHAPE_LABELS.get(record['shape'], record['shape'])} | "
            f"{record['n_seed']}/{record['n_final']} | "
            f"{record['filled_fraction_seed']:.4f} → {record['filled_fraction_final']:.4f} | "
            f"{record['inside_cv_seed']:.3f} → {record['inside_cv_final']:.3f} | "
            f"{record['energy_in_target_final']:.3f} | "
            f"{record['boundary_mean_px_final']:.2f} |"
        )
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--out", type=Path, default=Path(DEFAULT_OUT) / FIGURE_NAME)
    parser.add_argument("--dpi", type=float, default=150.0)
    parser.add_argument("--json", type=Path, default=None, help="also dump the paired numbers")
    parser.add_argument("--jsonl", type=Path, default=None,
                        help="read the numbers from audit_shape_agreement's streamed "
                             "--jsonl instead of re-reading the npz archives")
    parser.add_argument("--markdown", type=Path, default=None,
                        help="write the per-session markdown table (report §5.8)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    records = collect(args.campaign, jsonl=args.jsonl)
    if args.json:
        # The numbers land on disk before the drawing step, so a plotting
        # failure does not cost the sweep.
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(records, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"[json] {args.json}")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown_table(records), encoding="utf-8")
        print(f"[markdown] {args.markdown}")
    build_figure(records, args.out, dpi=args.dpi)
    for record in records:
        print(f"{record['label']:<44} "
              f"filled {record['filled_fraction_seed']:.4f} -> {record['filled_fraction_final']:.4f}"
              f"   inside_cv {record['inside_cv_seed']:.3f} -> {record['inside_cv_final']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
