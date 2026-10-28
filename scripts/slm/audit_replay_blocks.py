#!/usr/bin/env python
"""离线重放审计：把已归档的 CCD 重放块做成可判读的三类结果 (免设备)。

为什么要有这个脚本
------------------

2026-10-28 多形状 × 三尺度复盘的四条结论都要靠**重放块**支撑, 而当时没有任何
一处代码把重放块按统一口径读出来: `common_replay` 的 48 行里既有 20/50/90 ms
三个曝光、又有四个形状两个相位来源, 报告里的"末块 CV"是从 `manifest.json` 的
`metrics.cv` 直接抄的 —— 那个 CV 用的是**当次 stage 的** `objective.core`,
dense 段每一步都把 core 腐蚀一次、采样步长减半, 于是同一帧在两个段里会报出不同
的"均匀度"。同一份重放里还有被 250/255 判过的 8 bit 锚点帧, 以及一次 SLM 掉线
后作废的 4 个块。

本脚本不重算物理、不碰设备, 只做三件事:

1. **冻结孔径**: 用*全画幅传感器像素*定义一次目标框 / 核 / 零级排除盘
   (`protocol.FixedAperture`), 对每个已归档重放帧算同一口径的 CV、相对能量、
   零级排除比例、饱和像素比例;
2. **按重放条件分组**: 同一 `(曝光, 位深, 形状, 相位来源)` 才可比 —— 曝光之间
   的 CV 差 (实测 20 → 90 ms 会把 CV 从 0.419 推到 0.759) 远大于待测效应;
3. **给每组配对判定**: 用组内重复的展开估计噪声, 再用
   `protocol.verdict_with_uncertainty` 判定"更平"是否真的越过噪声且没丢能量。

用法
----

    python scripts/slm/audit_replay_blocks.py                        # 默认审计 s1200
    python scripts/slm/audit_replay_blocks.py --session s600_fixed
    python scripts/slm/audit_replay_blocks.py --json out.json --no-table

输出是**给人读的表 + 机器可读的 JSON**; 不写任何结论性文字, 因为"这次整形成功
了吗"是报告作者的事, 这里只保证口径统一、饱和可见、缺测可见。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:  # 允许直接 python scripts/... 运行
    sys.path.insert(0, str(ROOT / "src"))

from ao_shaping.tools.slm.transfer.protocol import (
    REPLAY_PEAK_GUARD_FRACTION,
    FixedAperture,
    aperture_metrics,
    judge_pair,
    repeat_statistics,
)

CAMPAIGN = ROOT / "data" / "debug" / "slm_transfer" / "multishape_scale_20261028"

#: Replay-block directories, in the order they were acquired.
REPLAY_DIRS = ("common_replay", "common_replay_r2", "common_replay_ladder_trial",
               "common_replay_dense")

SHAPES = ("square", "triangle", "star", "circle")


def _aperture_from_config(directory: Path, config: dict, *,
                          full_scale: float | None) -> FixedAperture | None:
    """Build the frozen aperture from the run that produced the replayed phases.

    A replay block's own ``config.json`` lists the stage runs it replays but not
    their geometry, so the aperture comes from the **first** source run's
    ``config.json`` (``initial_run``) -- the stage that produced the seed phase.
    ``target_xy`` is the requested target centre and ``side`` its edge length, so
    the aperture is the target square in sensor pixels, plus that run's core
    erosion and zero-order disk. Returns ``None`` when no source config can be
    found: a guessed aperture would make two blocks look comparable when they are
    not.
    """
    initial = config.get("initial_run")
    if not initial:
        return None
    source = Path(initial)
    candidates = [source / "config.json", directory.parent / "shape" / "square" / "transport_cv" / "config.json"]
    for candidate in candidates:
        if candidate.is_file():
            stage_config = json.loads(candidate.read_text(encoding="utf-8"))
            break
    else:
        return None
    if "side" not in stage_config or "target_xy" not in stage_config:
        return None
    zero_xy = stage_config.get("zero_xy")
    core_margin = stage_config.get("core_margin")
    zero_radius = stage_config.get("zero_radius")
    return FixedAperture(
        centre_xy=(float(stage_config["target_xy"][0]), float(stage_config["target_xy"][1])),
        half_side=float(stage_config["side"]) / 2.0,
        # A transport_cv stage runs with core_margin/zero_radius unset (None):
        # that stage has no core erosion, so 0.0 is the same geometry, not a fallback guess.
        core_margin_px=0.0 if core_margin is None else float(core_margin),
        zero_xy=None if zero_xy is None else (float(zero_xy[0]), float(zero_xy[1])),
        zero_radius=0.0 if zero_radius is None else float(zero_radius),
        full_scale=full_scale,
    )



def _label_parts(label: str) -> tuple[str, str, str]:
    """Split ``<stage>_<shape>_<phase>`` into its three parts.

    Stage names contain underscores (``mono12_continue``), so the split is done
    from the right, against the known shape vocabulary. A label that does not
    match is returned as ``("<unparsed>", label, "")`` rather than raising: an
    unparsed label is still worth counting and showing.
    """
    for shape in SHAPES:
        marker = f"_{shape}_"
        if marker in label:
            stage, phase = label.split(marker, 1)
            return stage, shape, phase
    return "<unparsed>", label, ""


def audit_block(directory: Path, *, guard: float = REPLAY_PEAK_GUARD_FRACTION) -> dict:
    """Recompute the frozen-aperture read-out for one replay block directory.

    Reads only ``config.json``, ``manifest.json`` and the archived ``*.npz``
    frames; touches no hardware and imports no torch (via ``protocol``).
    """
    config_path = directory / "config.json"
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        return dict(directory=str(directory), present=False, rows=[])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    full_scale = float(manifest[0]["full_scale"]) if manifest else 4095.0
    aperture = (_aperture_from_config(directory, config, full_scale=full_scale)
                if config else None)
    rows = []
    for row in manifest:
        stage, shape, phase = _label_parts(str(row["label"]))
        entry = dict(label=row["label"], stage=stage, shape=shape, phase=phase,
                     exposure_ms=row.get("exposure_ms"), bit_depth=row.get("bit_depth"),
                     full_scale=full_scale, recorded_cv=(row.get("metrics") or {}).get("cv"),
                     recorded_ee=(row.get("metrics") or {}).get("ee_roi"),
                     file=row.get("file"))
        if aperture is None:
            entry["error"] = "no source config for a frozen aperture"
        else:
            path = directory / str(row["file"])
            if not path.is_file():
                entry["error"] = "frame file missing"
            else:
                with np.load(path) as payload:
                    frame = payload["raw_frames"].mean(0)
                # Guard on the *frame* peak, not on the manifest's core_peak:
                # a clipped pixel outside the core still means the exposure is
                # not the one the comparison assumes.
                entry["peak"] = float(frame.max())
                entry["clipped"] = bool(entry["peak"] >= guard * full_scale)
                metrics = aperture_metrics(frame, aperture)
                entry.update(cv=metrics["cv"], ee_rel=metrics["ee_rel"],
                             zero_excluded_fraction=metrics["zero_excluded_fraction"],
                             saturated_fraction=metrics["saturated_fraction"])
        rows.append(entry)
    return dict(directory=str(directory), present=True, rows=rows,
                aperture=None if aperture is None else aperture.describe(),
                released=(directory / "released.json").is_file())


def group_rows(rows: list[dict]) -> dict:
    """Group by ``(exposure_ms, bit_depth, shape)``: the only comparable sets.

    Two rows under one key differ only by *which phase* was replayed, which is
    exactly the variable the comparison is about. Grouping across exposures is
    refused because a re-metered exposure moves CV more than the effect.
    """
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        if row.get("error") or row.get("cv") is None:
            continue
        groups.setdefault((row["exposure_ms"], row["bit_depth"], row["shape"]), []).append(row)
    return groups


def summarise_block(block: dict) -> dict:
    """Per-condition spread plus phase-to-phase comparisons for one replay block.

    Two spreads are reported, and they answer different questions:

    * ``replicates`` -- when a block replays the **same** phase more than once
      (a reference block does), its spread *is* the measurement noise, and that is
      what a verdict is tested against;
    * ``phase_spread`` -- across all phases in the condition, which describes how
      far the stages moved but says nothing about noise.

    With one measurement per phase there is no replicate spread, so every
    comparison reports ``insufficient_repeats`` instead of borrowing the
    phase-to-phase spread as if it were noise. That borrowing is exactly what made
    the earlier "2 pp" band look like a measurement when it was not.
    """
    rows = block["rows"]
    usable = [row for row in rows if not row.get("error")]
    clipped = [row["label"] for row in usable if row.get("clipped")]
    groups = {}
    for key, members in group_rows(usable).items():
        exposure, depth, shape = key
        phase_spread = repeat_statistics([row["cv"] for row in members])
        by_phase: dict[str, list[dict]] = {}
        for row in members:
            by_phase.setdefault(row["phase"], []).append(row)
        replicates = {phase: repeat_statistics([row["cv"] for row in phase_rows])
                      for phase, phase_rows in by_phase.items() if len(phase_rows) > 1}
        # The noise used by verdicts: replicate spread where it exists, otherwise
        # nothing (the verdict then reports "insufficient_repeats").
        noise = next(iter(replicates.values()), None)
        reference_phase = "seed" if "seed" in by_phase else None
        pairs = []
        if reference_phase is not None:
            reference = by_phase[reference_phase]
            for phase, phase_rows in by_phase.items():
                if phase == reference_phase:
                    continue
                # Compare the first frame of each phase: one measurement each, so
                # the difference is a stage difference, not an average. A tie is
                # reported as a tie (`judge_pair`), not as a "wrong direction".
                pairs.append(dict(
                    phase=phase,
                    verdict=judge_pair(
                        dict(cv=reference[0]["cv"], ee_rel=reference[0]["ee_rel"]),
                        dict(cv=phase_rows[0]["cv"], ee_rel=phase_rows[0]["ee_rel"],
                             saturated_fraction=phase_rows[0].get("saturated_fraction")),
                        noise=noise)))
        groups[f"{exposure:g}ms/{depth}bit/{shape}"] = dict(
            n=phase_spread["n"],
            cv_mean=phase_spread["mean"], cv_std=phase_spread["std"],
            cv_range=phase_spread["range"],
            ee_rel_mean=float(np.mean([r["ee_rel"] for r in members])),
            zero_excluded_fraction=float(np.mean([r["zero_excluded_fraction"] for r in members])),
            phases=sorted(by_phase),
            phase_counts={phase: len(phase_rows) for phase, phase_rows in by_phase.items()},
            replicates={phase: dict(n=stats["n"], half_width=stats["half_width"])
                        for phase, stats in replicates.items()},
            pairs=pairs)
    return dict(directory=block["directory"], present=block["present"],
                released=block.get("released", False), n_rows=len(rows),
                n_usable=len(usable), clipped=clipped,
                errors=[dict(label=row.get("label"), error=row["error"])
                        for row in rows if row.get("error")],
                groups=groups)



def _format_table(summary: dict) -> str:
    """Render one block's summary as fixed-width text for a terminal."""
    lines = [f"## {Path(summary['directory']).name}  "
             f"({summary['n_usable']}/{summary['n_rows']} rows usable"
             f"{', released' if summary['released'] else ', NOT released'})"]
    if not summary["present"]:
        lines.append("    (directory absent)")
        return "\n".join(lines)
    for error in summary["errors"]:
        lines.append(f"    ! {error['label']}: {error['error']}")
    if summary["clipped"]:
        lines.append(f"    ! clipped (>= {REPLAY_PEAK_GUARD_FRACTION:.0%} full scale), "
                     f"excluded from any comparison: {', '.join(summary['clipped'])}")
    for key, group in summary["groups"].items():
        std = "n/a" if group["cv_std"] is None else f"{group['cv_std']:.4f}"
        counts = group["phase_counts"]
        phases = "  phases=" + ",".join(f"{name}x{counts[name]}" for name in group["phases"])
        lines.append(f"  {key:28s} n={group['n']}  cv={group['cv_mean']:.4f} (sd {std})  "
                     f"ee_rel={group['ee_rel_mean']:.4f}  zero_excl={group['zero_excluded_fraction']:.3f}"
                     f"{phases}")
        for phase, stats in group.get("replicates", {}).items():
            # The only row in the table that is a measurement of noise: the same
            # phase captured more than once under one condition.
            lines.append(f"      replicate spread {phase:6s} n={stats['n']}  "
                         f"+/-{stats['half_width']:.4f} (95% t)")
        for pair in group["pairs"]:
            verdict = pair["verdict"]
            supported = {True: "supported", False: "no", None: "insufficient"}[verdict["supported"]]
            delta = verdict.get("delta_cv")
            delta_text = "n/a" if delta is None else f"{delta:+.4f}"
            lines.append(f"      seed -> {pair['phase']:8s} dCV={delta_text}  "
                         f"{supported:12s} ({verdict['reason']})")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--campaign", type=Path, default=CAMPAIGN,
                        help="campaign root (default: the 2026-10-28 multishape campaign)")
    parser.add_argument("--session", default="s1200", help="session directory under the campaign")
    parser.add_argument("--replay-dir", action="append", default=None,
                        help="replay block directory name (repeatable; default: all known ones)")
    parser.add_argument("--json", type=Path, default=None, help="write the machine-readable summary here")
    parser.add_argument("--no-table", action="store_true", help="suppress the text table")
    args = parser.parse_args(argv)

    session = args.campaign / args.session
    if not session.is_dir():
        parser.error(f"session directory not found: {session}")
    names = args.replay_dir or [name for name in REPLAY_DIRS if (session / name).is_dir()]
    if not names:
        parser.error(f"no replay directories under {session}")

    summaries = [summarise_block(audit_block(session / name)) for name in names]
    if not args.no_table:
        for summary in summaries:
            print(_format_table(summary))
            print()
    payload = dict(session=str(session), blocks=summaries,
                   peak_guard_fraction=REPLAY_PEAK_GUARD_FRACTION)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
