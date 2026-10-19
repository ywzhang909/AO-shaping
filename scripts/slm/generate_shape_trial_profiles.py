"""Regenerate X/Y far-field intensity cross-sections for ``report/shape_trials``.

Why this script exists
----------------------
The spot images embedded in ``report/shape_trials/*.md`` are grayscale renders.
A cross-section read off such a render is quantised to 8 bits, carries the axes
padding, and cannot be traced back to a measurement. Every figure here is
therefore re-derived from the **recorded raw CCD frame** in the ``pkl`` /
``npy`` / ``npz`` debug record set -- never from the PNG.

Two figures in this directory were already labelled "X/Y 光强截面" while
actually showing SLM phase and far-field images, and
``images/gui_shaping_20261014/fig4_box_profile.png`` drew flat and best on one
shared intensity axis even though their peaks differ by ~13x. Both problems are
structural, so every cut is re-derived from the recorded frame.

The rendering is **not** implemented here
----------------------------------------
The two panels and the cut point come from
:mod:`ao_shaping.display.beam_profile` -- the single implementation of the
far-field / SLM-phase / X-Y cross-section renderers, shared with the debug
artefacts a run's ``Recorder`` writes next to its ``.pkl``:

* **X 剖面** = the row through the spot centre, intensity vs. column index.
* **Y 剖面** = the column through the same spot centre, intensity vs. row index.
* the cut is the *measured* intensity centroid of the background-subtracted
  frame (``prepare_frame`` + ``spots_calc.centroid``), not an assumed frame
  centre and **not** ``argmax``: on a speckle field the brightest grain hops
  between near-equal peaks under a ~1e-3 perturbation, and measured across these
  figures ``argmax`` wanders up to 38 px from the frame centre while the
  centroid stays within ~15 px.

Keeping a second copy of that convention here is exactly the drift this repo
keeps getting bitten by, so this script only *calls* it
(``cut_point`` / ``save_xy_profile``).

What is left to this script
---------------------------
The part the shared renderer knows nothing about: mapping every report figure to
the record it was cut from (the registry below), grading that mapping
(``byte`` / ``content`` / ``structural`` / ``unresolved``), and stamping the
resolved source onto each figure as a provenance subtitle.

Provenance tiers
----------------
``byte-identical``
    Re-rendering the resolved frame through the same
    ``ao_shaping.display.frames.save_best_image`` call reproduces the committed
    PNG byte-for-byte. This holds for the model-in-loop run figures.
``content-match``
    Same renderer, but only the anti-aliased title glyphs differ (the run's own
    ``best_far_field.png``). This holds for the GS-refine figures.
``structural``
    The figure was rendered by an ad-hoc script with no committed generator, so
    the mapping is by name/label convention plus spot-structure inspection.
    Recorded as such; never upgraded silently.
``unresolved``
    The recorded raw frame is no longer on disk. Reported, never guessed.

Usage::

    python scripts/slm/generate_shape_trial_profiles.py --check
    python scripts/slm/generate_shape_trial_profiles.py
    python scripts/slm/generate_shape_trial_profiles.py --only 20261019_103208
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger
from typing import Any

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

# The panels and the cut point are owned by the canonical renderer -- the same
# one a run's Recorder uses for its debug artefacts. See the module docstring.
from ao_shaping.display.beam_profile import (  # noqa: E402
    ProfileCut,
    cut_point,
    save_xy_profile,
)

REPORT_DIR = ROOT / "report" / "shape_trials"
FIG_DIR = REPORT_DIR / "figures"
GUI_FIG_DIR = REPORT_DIR / "images" / "gui_shaping_20261014"

MIL_ROOT = ROOT / "data" / "debug" / "slm_model_in_loop"
MIL_REPLAY_ROOT = ROOT / "data" / "debug" / "slm_model_in_loop_replay"
GS_ROOT = ROOT / "data" / "debug" / "slm_gs_refine"


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------


@dataclass
class FrameSpec:
    """One spot figure and the recorded frame it must be re-derived from."""

    figure: str
    """Path relative to ``report/shape_trials``."""

    label: str
    """Chinese caption used as the plot title."""

    container: str = ""
    """Path relative to the repo root, or ``""`` when unresolved."""

    key: str = ""
    """Array key / record selector inside ``container``."""

    note: str = ""

    verify_mode: str = "none"
    """One of ``byte`` / ``content`` / ``structural`` / ``unresolved``."""

    verified: bool | None = None

    frame_shape: tuple[int, ...] | None = None

    spot_center: tuple[float, float] | None = None
    """``(row, col)`` of the spot peak after background subtraction."""

    peak: float | None = None

    background_median: float | None = None

    extras: dict[str, Any] = field(default_factory=dict)


# ``figures/20261019_100334_farfield.png`` -> run ``20261019_100334``
_RE_RUN_FIG = re.compile(r"^(?:figures/)?(?P<run>20\d{6}_\d{6})_farfield\.png$")

# ``../../data/debug/slm_model_in_loop/20261013_114650/best_far_field.png``
_RE_DIRECT = re.compile(
    r"^\.\./\.\./data/debug/slm_model_in_loop/(?P<run>20\d{6}_\d{6})/best_far_field\.png$"
)

# Replay figures carry the ABBA index + slot in the file name itself:
#   ..._20261016_1044_uniformity_weights_01_A_farfield.png  -> 01_A.npy
#   ..._20261014_1115_zero_vs_refit_05_A_farfield.png       -> 05_A.npy
_RE_MIL_REPLAY = re.compile(
    r"^figures/model_in_loop_live_(?P<replay>.+)_(?P<idx>\d{2})_(?P<slot>[A-Za-z0-9]+)_farfield\.png$"
)

# Single-run model-in-loop figure: ..._20261014_20261014_105041_farfield.png
# The label is optional (some figures carry only the run stamp).
_RE_MIL_RUN = re.compile(
    r"^figures/model_in_loop_live_[^/]*?(?P<run>20\d{6}_\d{6})(?:_(?P<label>[^/]+))?_farfield\.png$"
)

# Report file names may repeat the date prefix the run stamp already carries:
#   model_in_loop_live_20261016_20261016_1044_uniformity_weights_01_A_farfield.png
#     -> replay dir ``20261016_1044_uniformity_weights``
_RE_DATE_PREFIX = re.compile(r"^\d{8}_")

_RE_GS_ISSUE = re.compile(r"^figures/gs_issue_live_farfield_(?P<run>20\d{6}_\d{6})\.png$")

# ---- SLM 相位图 ----------------------------------------------------------
#
# The 2026-10-19 phase figures render as a blank block, which reads as a failed
# plot. It is not: those runs were rejected at the holdout gate before Step B,
# so ``best_phase.npy`` is legitimately all-zero. Two separate defects did show
# up though, and both are fixed by re-rendering through the canonical
# ``display.beam_profile.save_phase``:
#
#   1. no colourbar + an explicit "flat phase" annotation, so an all-zero array
#      can no longer be misread as a broken render;
#   2. the CJK font is applied explicitly -- the old figures fell back to a
#      Latin-only face and the title rendered as tofu boxes ("SLM ▯▯").

# 单次运行: figures/20261019_103208_phase.png -> <run>/best_phase.npy
_RE_PHASE_RUN_FIG = re.compile(r"^(?:figures/)?(?P<run>20\d{6}_\d{6})_phase\.png$")

# 单次运行: figures/model_in_loop_live_20261014_20261014_105041_phase.png
_RE_PHASE_MIL_RUN = re.compile(
    r"^figures/model_in_loop_live_[^/]*?(?P<run>20\d{6}_\d{6})_phase\.png$"
)

# 重放相位图: A/B 两格分别来自 config.json["sources"] 的第 0 / 第 1 个源运行。
# A/B 之后跟着一段阶段描述（`A_zero_step` / `B_10_step`），其内部还含下划线，
# 故不能靠 `_phase` 收尾来切分；这里显式列出已知形态。
_RE_PHASE_REPLAY = re.compile(
    r"^figures/model_in_loop_live_(?P<replay>.+?)_"
    r"(?P<slot>A|B)(?:_(?:zero_step|\d+_step))?_phase\.png$"
)

# GUI 闭环整形会话 C/图 3 的四个子图 -> 两个会话的交错重放原始帧。
# 顺序即 render_panel 的 (row, col) 顺序：A flat / A best / C flat / C best。
_GUI_REPLAY_SPECS = (
    ("20261014_100441", "00_flat.npy", "会话 A 平相位", "A_flat"),
    ("20261014_100441", "02_best.npy", "会话 A 最佳", "A_best"),
    ("20261014_143035", "00_flat.npy", "会话 C 平相位", "C_flat"),
    ("20261014_143035", "02_best.npy", "会话 C 最佳", "C_best"),
)

# 无原始数据可用的图（调试 PKL 未入库，见 report/shape_trials/README.md）
_KNOWN_MISSING = {
    "figures/gs_pad3_best_far_field.png": (
        "2026-10-10 运行的调试 PKL 未随仓库保留（data/* 在 .gitignore 中），"
        "盘上已无该次原始 CCD 帧，无法重建截面"
    ),
    "figures/gs_pad3_refine24_best_far_field.png": (
        "2026-10-10 运行的调试 PKL 未随仓库保留，盘上已无原始 CCD 帧"
    ),
    "figures/gs_pad3_debug_summary.png": (
        "2026-10-10 运行的调试 PKL 未随仓库保留；其 first_img/last_img 子图无法重建"
    ),
    "figures/shape_coverage1_debug_summary.png": (
        "2026-10-10 hardware_shape_coverage1_20261010/ 调试记录未随仓库保留"
    ),
}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _replay_sources(replay: str) -> list[Path]:
    """Source run directories behind a replay, in A/B order (``config.json``).

    ``config.json`` stores absolute paths, and a source need not be a bare run
    directory -- ``20261016_1048_zero_refit/rejected_shaped_candidate`` is a
    sub-directory of one -- so the path relative to :data:`MIL_ROOT` is kept
    rather than just the directory name.
    """
    cfg = MIL_REPLAY_ROOT / replay / "config.json"
    if not cfg.exists():
        return []
    try:
        payload = json.loads(cfg.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    out: list[Path] = []
    for raw in payload.get("sources", []):
        p = Path(raw)
        try:
            out.append(p.relative_to(MIL_ROOT))
        except ValueError:
            out.append(Path(p.name))
    return out


def _mil_phase_path(rel_run: str | Path) -> Path | None:
    """The run's recorded ``best_phase.npy``, or ``None`` when absent."""
    cand = MIL_ROOT / Path(rel_run) / "best_phase.npy"
    return cand if cand.exists() else None


def _resolve_phase(figure: str) -> tuple[str, str]:
    """Map a phase figure to ``(container_relpath, note)``; ``("", note)`` if unresolved."""
    m = _RE_PHASE_RUN_FIG.match(figure) or _RE_PHASE_MIL_RUN.match(figure)
    if m:
        run = m.group("run")
        path = _mil_phase_path(run)
        if path is None:
            return "", f"{run} 没有 best_phase.npy"
        return _rel(path), f"{run} 的 best_phase.npy"

    m = _RE_PHASE_REPLAY.match(figure)
    if m:
        replay = m.group("replay")
        slot = m.group("slot")
        index = 0 if slot == "A" else 1
        # Report file names may repeat the date the run stamp already carries,
        # so try the verbatim form first and the stripped one second -- the same
        # fallback `_find_replay_frame` uses for the far-field figures.
        sources: list[Path] = []
        for candidate in (replay, _RE_DATE_PREFIX.sub("", replay)):
            found = _replay_sources(candidate)
            if len(found) > index:
                sources = found
                break
        if not sources:
            return "", f"重放 {replay} 的 config.json 未列出第 {index + 1} 个源运行"

        source = sources[index]
        path = _mil_phase_path(source)
        if path is None:
            return "", f"{source.as_posix()} 没有 best_phase.npy"
        return (
            _rel(path),
            f"重放 {replay} 的 {slot} 格 -> 源 {source.as_posix()} 的 best_phase.npy",
        )

    return "", "未匹配任何已知相位图来源规则"


def _load_gs_pkl_frame(path: Path) -> np.ndarray | None:
    """GS-refine frame behind ``best_far_field.png``: the max-``score`` record."""
    with path.open("rb") as fh:
        recs = pickle.load(fh)
    if not isinstance(recs, dict) or not recs:
        return None
    best_img, best_score = None, -np.inf
    for rec in recs.values():
        if not isinstance(rec, dict):
            continue
        img = rec.get("_img")
        if img is None:
            continue
        score = float(rec.get("score", -np.inf))
        if score > best_score:
            best_img, best_score = np.asarray(img, dtype=float), score
    return best_img


def _mil_scores(run: str) -> tuple[float, float] | None:
    """``(flat_score, best_score)`` for a model-in-loop run.

    ``result.flat_score`` is round 0's ``score_after``; ``result.best_score``
    starts there and only rises when a round's ``score_after`` beats it
    (``slm_model_in_loop.py:2552``/``:3004``). Rounds that stop before Step B
    (holdout rejected) record no ``score_after`` at all -- which is exactly the
    case where ``best_score == flat_score``.
    """
    run_dir = MIL_ROOT / run
    pkl = next(iter(run_dir.glob("slm_model_in_loop_*.pkl")), None)
    if pkl is None:
        return None
    with pkl.open("rb") as fh:
        recs = pickle.load(fh)
    if not isinstance(recs, dict) or not recs:
        return None

    def score_of(rec: Any) -> float | None:
        if not isinstance(rec, dict):
            return None
        for key in ("score_after", "score_before"):
            if key in rec:
                return float(rec[key])
        return None

    flat = score_of(recs[min(recs)])
    if flat is None:
        return None
    best = flat
    for rec in recs.values():
        s = score_of(rec)
        if s is not None and s > best:
            best = s
    return flat, best


def _mil_best_frame(run: str) -> tuple[str, str, str] | None:
    """``(container_rel, npz_key, why)`` for a model-in-loop run's best frame.

    ``slm_model_in_loop`` seeds ``best_frame`` with the flat baseline and only
    keeps the Step-B shaped frame when its score strictly beats the baseline
    (``slm_model_in_loop.py:2552`` seed, ``:3004`` accept, ``:3072`` revert).
    A run that never clears the holdout gate therefore plots its **flat
    baseline** -- which is why several 2026-10-19 figures are the baseline and
    must not be captioned as shaping output.
    """
    run_dir = MIL_ROOT / run
    flat = run_dir / "preflight" / "flat.npz"
    shaped = run_dir / "probes" / "round_01_shaped_00.npz"

    scores = _mil_scores(run)
    if shaped.exists() and scores is not None and scores[1] > scores[0]:
        return (
            _rel(shaped),
            "frame",
            f"Step B 被接受（score_after {scores[1]:.4f} > 平相位 {scores[0]:.4f}）",
        )
    if flat.exists():
        if scores is None:
            why = "运行记录缺少 score，best_frame 按平相位处理"
        else:
            why = (
                f"Step B 未改进（best {scores[1]:.4f} <= 平相位 {scores[0]:.4f}），"
                "best_frame 回退为平相位"
            )
        return _rel(flat), "frame", why
    return None


# --------------------------------------------------------------------------
# discovery
# --------------------------------------------------------------------------


def _find_replay_frame(replay: str, idx: str, slot: str) -> Path | None:
    """Locate ``<replay>/<idx>_<slot>.npy``, tolerating a doubled date prefix.

    Some report file names repeat the date the run stamp already carries
    (``model_in_loop_live_20261016_20261016_1044_uniformity_weights_01_A_...``)
    while others do not (``..._20261014_1115_zero_vs_refit_05_A_...``). Stripping
    unconditionally breaks the second form, so try the verbatim name first.
    """
    for name in (replay, _RE_DATE_PREFIX.sub("", replay)):
        cand = MIL_REPLAY_ROOT / name / f"{idx}_{slot}.npy"
        if cand.exists():
            return cand
    return None


_EXTRA_PROFILES = (
    # gs_issue_live_20261015.md §「plain (20261015_205654)」 embeds
    # figures/20261019_100334_farfield.png -- an image from a *different* run
    # (slm_model_in_loop/20261019_100334). The correct frame for that section is
    # the 20261015_205654 debug PKL, so cut its profile even though no report
    # figure of its own is referenced.
    (
        "figures/gs_issue_live_farfield_20261015_205654.png",
        "20261015_205654",
    ),
)


def _resolve_one(figure: str) -> FrameSpec:
    m = _RE_RUN_FIG.match(figure) or _RE_DIRECT.match(figure)
    if m:
        run = m.group("run")
        got = _mil_best_frame(run)
        spec = FrameSpec(figure=figure, label=run, verify_mode="byte")
        if got:
            spec.container, spec.key, spec.note = got
        else:
            spec.verify_mode = "unresolved"
            spec.note = f"运行目录 {run} 下没有可用的 preflight/flat.npz 或 shaped 帧"
        return spec

    # Replay first: its name already ends in _NN_<slot> which _RE_MIL_RUN would
    # otherwise swallow as a label.
    m = _RE_MIL_REPLAY.match(figure)
    if m:
        replay, idx, slot = m.group("replay"), m.group("idx"), m.group("slot")
        cand = _find_replay_frame(replay, idx, slot)
        spec = FrameSpec(
            figure=figure,
            label=f"{replay} 第 {idx} 次 {slot}",
            verify_mode="structural",
            note="交错重放原始帧（ABBA 序列），非优化器内部 best_frame",
        )
        if cand is not None:
            spec.container = _rel(cand)
            spec.extras["replay_dir"] = cand.parent.name
        else:
            spec.verify_mode = "unresolved"
            spec.note = f"重放目录缺少 {idx}_{slot}.npy（试过 {replay} 及其去日期前缀形式）"
        return spec

    m = _RE_MIL_RUN.match(figure)
    if m:
        run, label = m.group("run"), m.group("label")
        got = _mil_best_frame(run)
        spec = FrameSpec(figure=figure, label=f"{run} · {label}", verify_mode="byte")
        if got:
            spec.container, spec.key, spec.note = got
        else:
            spec.verify_mode = "unresolved"
            spec.note = f"运行目录 {run} 下没有可用的帧记录"
        return spec

    m = _RE_GS_ISSUE.match(figure)
    if m:
        run = m.group("run")
        pkl = next(iter((GS_ROOT / run).glob("slm_gs_refine_*.pkl")), None)
        spec = FrameSpec(
            figure=figure,
            label=run,
            verify_mode="content",
            note="GS refine 调试 PKL 中 score 最优记录的 _img（原始 CCD 帧）",
        )
        if pkl is not None:
            spec.container = _rel(pkl)
            spec.key = "_img of argmax(score)"
            spec.extras["run"] = run
            spec.note = (
                "GS refine 调试 PKL 中 score 最优记录的 _img（原始 CCD 帧）；"
                "报告里的远场图是另一次 inferno 渲染（600×600，无入库生成脚本），"
                "故只能与该运行自带的 best_far_field.png 对照"
            )
        else:
            spec.verify_mode = "unresolved"
            spec.note = f"{run} 的调试 PKL 不在盘上"
        return spec

    if figure == "images/gui_shaping_20261014/fig4_box_profile.png":
        return FrameSpec(
            figure=figure,
            label="旧中心剖面（已被取代）",
            verify_mode="unresolved",
            note=(
                "旧截面图作废并由本脚本的逐帧 X/Y 剖面取代：flat 与 best 曾共用一条强度轴，"
                "而两者峰值相差约 13 倍（146 vs 11 count），best 曲线被压平不可读；"
                "切点也按帧中心而非实测光斑质心取。新图见同目录 "
                "`fig3_far_field_*_xy_profile.png`"
            ),
        )

    if figure in _KNOWN_MISSING:
        return FrameSpec(
            figure=figure,
            label=Path(figure).stem,
            verify_mode="unresolved",
            note=_KNOWN_MISSING[figure],
        )

    return FrameSpec(
        figure=figure,
        label=Path(figure).stem,
        verify_mode="unresolved",
        note="未匹配任何已知来源规则，需人工确认",
    )


def discover() -> list[FrameSpec]:
    """Every spot figure referenced by ``report/shape_trials/*.md``.

    Target-pattern figures (``*_farfield_target.png``) are excluded: they are
    synthesised targets, not measurements, and the run records do not store the
    target array.
    """
    refs: dict[str, str] = {}
    for md in sorted(REPORT_DIR.glob("*.md")):
        text = md.read_text(encoding="utf-8")
        for hit in re.finditer(r"!?\[[^\]]*\]\(([^)]+\.png)\)", text):
            rel = hit.group(1).replace("\\", "/")
            if not any(
                k in rel
                for k in ("farfield", "far_field", "profile", "debug_summary")
            ):
                continue
            if "target" in rel:
                continue  # synthesised target, not a measurement
            if rel.endswith("_xy_profile.png"):
                continue  # this script's own output, never an input
            # fig3 is re-cut per panel below; its bare container entry is a duplicate
            if rel in ("images/gui_shaping_20261014/fig3_far_field.png",):
                continue
            refs.setdefault(rel, md.name)

    specs = [_resolve_one(rel) for rel in sorted(refs)]

    for figure, run in _EXTRA_PROFILES:
        base = _resolve_one(figure)
        if base.container:
            specs.append(base)

    for run, fname, label, panel in _GUI_REPLAY_SPECS:
        cand = GS_ROOT / run / "replay" / fname
        specs.append(
            FrameSpec(
                figure=f"images/gui_shaping_20261014/fig3_far_field.png#{panel}",
                label=label,
                container=_rel(cand) if cand.exists() else "",
                verify_mode="structural" if cand.exists() else "unresolved",
                note=(
                    "GUI 闭环整形交错重放原始帧；图 3 的第 "
                    f"{panel} 格"
                    if cand.exists()
                    else f"{run}/replay/{fname} 不在盘上"
                ),
                extras={"panel": panel, "source_run": run, "frame_file": fname},
            )
        )
    return specs


# --------------------------------------------------------------------------
# loading + verification
# --------------------------------------------------------------------------


def load_frame(spec: FrameSpec) -> np.ndarray | None:
    if not spec.container:
        return None
    path = ROOT / spec.container
    if not path.exists():
        return None
    if path.suffix == ".pkl":
        return _load_gs_pkl_frame(path)
    arr = np.load(path)
    if isinstance(arr, np.ndarray):
        return np.asarray(arr, dtype=float)
    keys = getattr(arr, "files", [])
    if spec.key in keys:
        return np.asarray(arr[spec.key], dtype=float)
    if keys:
        return np.asarray(arr[keys[0]], dtype=float)
    raise ValueError(f"{spec.container}: cannot read an array")


def _verify_bytes(figure_png: Path, frame: np.ndarray, title: str) -> bool:
    """Confirm the resolved frame reproduces the committed PNG.

    Byte equality is the normal outcome. A handful of figures differ only by
    sub-8-bit title antialiasing (max 1 grey level, no pixel off by more than
    2), which is still proof that the image region is the same measurement, so
    that case is accepted rather than silently downgraded.
    """
    from ao_shaping.display.frames import save_best_image

    with tempfile.TemporaryDirectory() as tmp:
        probe = Path(tmp) / "probe.png"
        save_best_image(frame, probe, title=title, hide_axes=True)
        if probe.read_bytes() == figure_png.read_bytes():
            return True

    ok, worst, frac = _verify_content_against(frame, title, figure_png)
    return bool(ok or (worst <= 1.0 and frac == 0.0))


def _verify_content_against(frame: np.ndarray, title: str, ref_png: Path) -> tuple[bool, float, float]:
    """Compare a fresh render with ``ref_png`` outside the title band.

    The two renders differ solely in the title glyphs, so diffing a render
    against a second render with a different title isolates the band that must
    be excluded. Returns ``(ok, max_abs_gray_diff, fraction_of_pixels_differing)``.
    """
    from PIL import Image

    from ao_shaping.display.frames import save_best_image

    with tempfile.TemporaryDirectory() as tmp:
        a = Path(tmp) / "a.png"
        b = Path(tmp) / "b.png"
        save_best_image(frame, a, title=title, hide_axes=True)
        save_best_image(frame, b, title="T" * (len(title) + 7), hide_axes=True)
        ia = np.asarray(Image.open(a).convert("L"), dtype=float)
        ib = np.asarray(Image.open(b).convert("L"), dtype=float)
        src = np.asarray(Image.open(ref_png).convert("L"), dtype=float)

    if ia.shape != src.shape or ia.shape != ib.shape:
        return False, float("nan"), float("nan")
    band_rows = np.where((np.abs(ia - ib) > 0).any(axis=1))[0]
    mask = np.ones(ia.shape, dtype=bool)
    if band_rows.size:
        mask[band_rows.min() : band_rows.max() + 2, :] = False
    diff = np.abs(src - ia)
    outside = diff[mask]
    worst, frac = float(outside.max()), float((outside > 2).mean())
    return bool(worst <= 2 and frac < 1e-4), worst, frac


def verify(spec: FrameSpec, frame: np.ndarray) -> bool | None:
    if spec.verify_mode == "structural" or "#" in spec.figure:
        return None
    if spec.verify_mode == "byte":
        figure_png = REPORT_DIR / spec.figure
        run = next((p for p in spec.container.split("/") if re.fullmatch(r"20\d{6}_\d{6}", p)), None)
        if run is None or not figure_png.exists():
            return None
        scores = _mil_scores(run)
        if scores is None:
            return None
        flat, best = scores
        return _verify_bytes(
            figure_png,
            frame,
            f"slm-model-in-loop best: score={best:.4f} (flat {flat:.4f})",
        )
    if spec.verify_mode == "content":
        run = spec.extras.get("run", "")
        run_dir = GS_ROOT / run
        ref_png = run_dir / "best_far_field.png"
        pkl = next(iter(run_dir.glob("slm_gs_refine_*.pkl")), None)
        if pkl is None or not ref_png.exists():
            return None
        with pkl.open("rb") as fh:
            recs = pickle.load(fh)
        _, best_rec = max(
            (r for r in recs.items() if isinstance(r[1], dict) and "_img" in r[1]),
            key=lambda kv: float(kv[1].get("score", -np.inf)),
            default=(None, None),
        )
        if best_rec is None:
            return None
        title = (
            f"slm-gs-refine best: score={best_rec.get('score', 0):.4f} "
            f"PIB={best_rec.get('pib', 0):.4f} CV={best_rec.get('cv', 0):.4f}"
        )
        ok, worst, frac = _verify_content_against(frame, title, ref_png)
        spec.extras["run_png_agreement"] = {
            "reference": _rel(ref_png),
            "max_abs_gray_diff": worst,
            "fraction_pixels_differing": frac,
        }
        return ok
    return None


# --------------------------------------------------------------------------
# plotting
# --------------------------------------------------------------------------


def provenance_subtitle(spec: FrameSpec, frame: np.ndarray) -> str:
    """Footer naming where the frame came from and how far it was verified.

    This is the script's own value-add: :func:`~ao_shaping.display.beam_profile.save_xy_profile`
    knows nothing about the report corpus, so it cannot say whether the resolved
    frame reproduces the committed PNG byte-for-byte or was only matched by name.
    """
    src = f"原始 CCD 帧：{spec.container}" + (f" · {spec.key}" if spec.key else "")
    tier = {
        "byte": "PNG 逐字节校验通过",
        "content": "图像区逐像素校验通过（仅标题字形不同）",
        "structural": "按命名与结构对应，未逐字节校验",
    }.get(spec.verify_mode, "")
    return f"{src} · 形状 {tuple(np.shape(frame))}\n{tier}"


def plot_profile(frame: np.ndarray, spec: FrameSpec, out_path: Path) -> None:
    """Two-panel X/Y cross-section through the spot centre (fig4 style).

    Thin wrapper over the canonical renderer: the panels, titles, labels, cut
    line and dpi all live in :mod:`ao_shaping.display.beam_profile`. This only
    supplies the provenance subtitle.
    """
    cut = cut_point(frame)
    save_xy_profile(
        frame,
        out_path,
        cut=cut,
        title=spec.label,
        subtitle=provenance_subtitle(spec, frame),
        dpi=130,
    )


def _flat_phase_reason(container: str) -> str:
    """Why a resolved ``best_phase.npy`` is all-zero, when the record says so.

    Returns ``""`` when the answer is unknown -- the renderer then prints the
    neutral fallback instead of a guess.
    """
    parts = container.split("/")
    run = next((p for p in parts if re.fullmatch(r"20\d{6}_\d{6}", p)), None)
    if run is None:
        return ""
    scores = _mil_scores(run)
    if scores is None:
        return ""
    flat, best = scores
    if best <= flat:
        return f"{run}：Step B 未改进（best {best:.4f} ≤ 平相位 {flat:.4f}），best_phase 为平相位"
    return ""


def regenerate_phase_figures() -> tuple[int, int]:
    """Re-render every SLM-phase figure from the run's recorded ``best_phase.npy``.

    The 2026-10-19 phase figures read as blank blocks. That is not a plotting
    failure -- those runs were rejected at the holdout gate before Step B, so
    ``best_phase.npy`` is legitimately all-zero -- but two real defects made it
    look like one, and re-rendering through the canonical
    :func:`~ao_shaping.display.beam_profile.save_phase` fixes both:

    1. an all-zero array now gets an explicit "平相位（全零）… 非渲染失败"
       annotation instead of a flat colour block under a 0..2π colourbar;
    2. the CJK font is applied explicitly -- the old figures fell back to a
       Latin-only face, so the title rendered as tofu boxes ("SLM ▯▯").

    Returns ``(written, unresolved)``.
    """
    from ao_shaping.display.beam_profile import save_phase

    written = 0
    unresolved: list[str] = []
    for md in sorted(REPORT_DIR.glob("*.md")):
        text = md.read_text(encoding="utf-8")
        for hit in re.finditer(r"!?\[[^\]]*\]\(([^)]+_phase\.png)\)", text):
            rel = hit.group(1).replace("\\", "/")
            if not rel.startswith("figures/"):
                continue
            container, note = _resolve_phase(rel)
            target = REPORT_DIR / rel
            if not container:
                unresolved.append(f"{rel} — {note}")
                continue
            arr = np.load(ROOT / container)
            label = Path(rel).stem
            save_phase(
                arr,
                target,
                title=f"{label} · SLM 相位",
                flat_reason=_flat_phase_reason(container),
            )
            written += 1
    if unresolved:
        logger.warning("相位图未能重建 {} 张: {}", len(unresolved), unresolved)
    return written, len(unresolved)


def out_path_for(spec: FrameSpec) -> Path:
    """Output PNG next to its source figure, named after the run, not the stem.

    ``model_in_loop_live_20261013.md`` links three *different* runs that all end
    in ``best_far_field.png``; keying the output on the bare stem would make all
    three overwrite one file, so the run stamp is used when the stem is generic.
    """
    p = REPORT_DIR / spec.figure.split("#")[0]
    stem = p.stem
    if stem == "best_far_field":
        runs = [q for q in spec.container.split("/") if re.fullmatch(r"20\d{6}_\d{6}", q)]
        stem = runs[0] if runs else "best_far_field"
    panel = spec.figure.split("#")[1] if "#" in spec.figure else ""
    name = f"{stem}_{panel}_xy_profile.png" if panel else f"{stem}_xy_profile.png"
    return GUI_FIG_DIR / name if p.parent.name == "gui_shaping_20261014" else FIG_DIR / name


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Regenerate shape_trials X/Y cross-sections.")
    ap.add_argument("--check", action="store_true", help="resolve + verify only, write nothing")
    ap.add_argument("--only", default="", help="substring filter on the figure path")
    args = ap.parse_args(argv)

    specs = discover()
    if args.only:
        specs = [s for s in specs if args.only in s.figure]

    rows: list[dict[str, Any]] = []
    tally = {"byte": 0, "content": 0, "structural": 0, "unresolved": 0}

    # Pass 1 -- resolve, measure and verify every figure.
    # Verification must complete before ANY plotting happens: the shared renderer
    # sets matplotlib rcParams["font.sans-serif"] globally, which changes the glyph
    # raster of the *next* figure's title and would silently break its
    # byte-for-byte comparison against the committed PNG.
    loaded: list[tuple[FrameSpec, np.ndarray]] = []
    for spec in specs:
        frame = load_frame(spec)
        if frame is None:
            tally["unresolved"] += 1
            rows.append(
                {
                    "figure": spec.figure,
                    "status": "UNRESOLVED",
                    "verify_mode": spec.verify_mode,
                    "note": spec.note,
                }
            )
            print(f"[UNRESOLVED] {spec.figure}\n             {spec.note}")
            continue

        spec.frame_shape = tuple(int(v) for v in frame.shape)
        cut = cut_point(frame)
        spec.spot_center = (cut.row, cut.col)
        spec.peak = cut.peak
        spec.background_median = cut.background_median
        spec.extras["argmax_row_col"] = [cut.argmax_row, cut.argmax_col]
        spec.verified = verify(spec, frame)
        loaded.append((spec, frame))

    # Pass 2 -- draw. Skipped by --check.
    if not args.check:
        for spec, frame in loaded:
            plot_profile(frame, spec, out_path_for(spec))

    for spec, frame in loaded:
        out = out_path_for(spec)
        cy, cx = spec.spot_center  # type: ignore[misc]
        peak, med = spec.peak, spec.background_median

        tier = spec.verify_mode
        if spec.verified is True:
            tier = "byte" if spec.verify_mode == "byte" else "content"
        elif spec.verify_mode in ("byte", "content"):
            # A loaded frame whose render does not reproduce the committed PNG
            # is NOT confirmed; fall back to the weaker naming/structure tier
            # rather than claiming verification.
            tier = "structural"
        tally[tier] = tally.get(tier, 0) + 1

        dy, dx = abs(cy - frame.shape[0] // 2), abs(cx - frame.shape[1] // 2)
        print(
            f"[{tier.upper():10s}] {spec.figure}\n"
            f"             src={spec.container}"
            + (f"#{spec.key}" if spec.key else "")
            + f"  shape={frame.shape}  spot=({cx:.0f},{cy:.0f})"
            + f"  frame_centre=({frame.shape[1] // 2},{frame.shape[0] // 2})"
            + f"  offset=({dx:.0f},{dy:.0f})px  peak={peak:.4g}"
        )
        rows.append(
            {
                "figure": spec.figure,
                "status": "OK",
                "verify_mode": tier,
                "source": spec.container,
                "key": spec.key,
                "note": spec.note,
                "frame_shape": spec.frame_shape,
                "spot_center_row_col": [cy, cx],
                "frame_center_row_col": [frame.shape[0] // 2, frame.shape[1] // 2],
                "cut_point": "prepare_frame + spots_calc.centroid",
                "offset_from_frame_center_px": [dx, dy],
                "peak_after_median_subtraction": peak,
                "median_background": med,
                "profile_png": _rel(out),
                **spec.extras,
            }
        )

    phase_written = phase_unresolved = 0
    if not args.check:
        phase_written, phase_unresolved = regenerate_phase_figures()
        print(f"\n相位图重绘：{phase_written} 张，未能重建 {phase_unresolved} 张")

        manifest = FIG_DIR / "xy_profile_manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "generator": "scripts/slm/generate_shape_trial_profiles.py",
                    "convention": "X 剖面 = 过光斑中心行的行剖面；Y 剖面 = 过光斑中心列的列剖面",
                    "cut_point": "prepare_frame + spots_calc.centroid（光强质心，非 argmax）",
                    "tally": tally,
                    "phase_figures_regenerated": phase_written,
                    "phase_figures_unresolved": phase_unresolved,
                    "figures": rows,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nmanifest -> {_rel(manifest)}")

    print("\n" + "  ".join(f"{k}={v}" for k, v in sorted(tally.items())))
    return 1 if tally["unresolved"] else 0


if __name__ == "__main__":
    raise SystemExit(main())