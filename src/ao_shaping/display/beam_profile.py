"""Canonical far-field / SLM-phase / X-Y cross-section renderers.

This is the **single** implementation of the three companion images that every
recorded optimisation frame should ship with:

``far-field``
    the camera frame itself, via :func:`~ao_shaping.display.frames.save_best_image`.
``phase``
    the SLM pattern, either raw radians or the uint16 grayscale actually sent.
``X/Y cross-section``
    the row / column intensity cuts through the spot.

Why the cut point is a centroid, not ``argmax``
----------------------------------------------
``ao_shaping.utils.image.beam_metrics.zero_order_center`` picks a zero-order by
``argmax`` because it must choose one order out of several that are far apart.
Cutting a *single speckle field* is a different job: the brightest grain hops
between near-equal peaks under a ~1e-3 perturbation. Measured across the
``report/shape_trials`` corpus, ``argmax`` wandered up to 38 px from the frame
centre while the background-subtracted intensity centroid stayed within ~15 px.
So the cut follows :func:`ao_shaping.utils.image.beam_metrics.prepare_frame` +
:func:`ao_shaping.utils.image.spots_calc.centroid` -- the same canonical pair
the beam metrics use -- and :attr:`ProfileCut.argmax_row_col` is carried purely
as a reported cross-check.

Layering
--------
``display/`` sits above ``utils/`` and may import it; the reverse is forbidden,
so ``utils/io/file.py`` imports *this* module lazily from inside the function
that plots (see ``tests/ao_shaping/utils/test_utils_leaf_no_rendering_import.py``).
Nothing here is imported at package import time.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from ao_shaping.utils.image.beam_metrics import prepare_frame
from ao_shaping.utils.image.spots_calc import centroid

__all__ = (
    "ProfileCut",
    "cut_point",
    "plot_xy_profile",
    "save_xy_profile",
    "save_phase",
    "save_far_field",
    "save_frame_artifacts",
)

#: Prefixes tried, in order, when looking for a CJK-capable font.
CJK_FONTS = ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans")

#: Record keys that hold an SLM pattern rather than a camera frame. Used by the
#: auto-detection in :func:`_partition_keys` so a uint16 panel is never mistaken
#: for a far-field frame.
PHASE_LIKE_DTYPES = (np.uint16,)


@dataclass(frozen=True)
class ProfileCut:
    """Where the X/Y cross-sections cut, and how that was decided."""

    row: float
    """Cut row, i.e. the row the X (horizontal) profile is taken from."""

    col: float
    """Cut column, i.e. the column the Y (vertical) profile is taken from."""

    peak: float
    """Maximum after background removal."""

    background_median: float
    """Median of the raw frame -- the background estimate ``prepare_frame`` used."""

    argmax_row: float
    argmax_col: float
    """Brightest pixel, reported only so a reader can see the disagreement."""

    frame_shape: tuple[int, int]

    @property
    def frame_center(self) -> tuple[float, float]:
        return (self.frame_shape[1] / 2.0, self.frame_shape[0] / 2.0)

    @property
    def offset_from_frame_center(self) -> tuple[float, float]:
        """``(dx, dy)`` of the cut from the frame centre, in pixels."""
        cx, cy = self.frame_center
        return (self.col - cx, self.row - cy)

    def summary(self) -> str:
        dx, dy = self.offset_from_frame_center
        return (
            f"切点（光强质心）({self.col:.0f}, {self.row:.0f})\n"
            f"图幅中心 ({self.frame_center[0]:.0f}, {self.frame_center[1]:.0f})"
            f"  偏移 ({dx:+.0f}, {dy:+.0f}) px\n"
            f"最亮像素 ({self.argmax_col:.0f}, {self.argmax_row:.0f})\n"
            f"背景中位数 {self.background_median:.4g}  峰值 {self.peak:.4g}"
        )


def cut_point(frame: np.ndarray) -> ProfileCut:
    """Locate the cross-section cut for one camera frame.

    Args:
        frame: 2-D intensity frame, any numeric dtype.

    Returns:
        The cut row/column plus the evidence behind it.
    """
    f = np.asarray(frame, dtype=float)
    if f.ndim != 2:
        raise ValueError(f"expected a 2-D frame, got shape {f.shape}")
    if not np.any(np.isfinite(f)):
        raise ValueError("frame has no finite samples")
    median = float(np.median(f))
    prepared = np.asarray(prepare_frame(f), dtype=float)
    col, row = centroid(prepared, return_float=True)
    peak_row, peak_col = np.unravel_index(int(np.argmax(prepared)), prepared.shape)
    return ProfileCut(
        row=float(row),
        col=float(col),
        peak=float(prepared.max()),
        background_median=median,
        argmax_row=float(peak_row),
        argmax_col=float(peak_col),
        frame_shape=(int(f.shape[0]), int(f.shape[1])),
    )


def _apply_cjk_font() -> None:
    from matplotlib import font_manager

    import matplotlib.pyplot as plt

    available = {f.name for f in font_manager.fontManager.ttflist}
    for family in CJK_FONTS:
        if family in available:
            plt.rcParams["font.sans-serif"] = [family]
            break
    plt.rcParams["axes.unicode_minus"] = False


def plot_xy_profile(
    frame: np.ndarray,
    *,
    cut: ProfileCut | None = None,
    title: str = "",
    subtitle: str = "",
    figsize: tuple[float, float] = (11.0, 4.3),
    dpi: int = 130,
):
    """Build the two-panel X/Y cross-section figure for one frame.

    Left panel: intensity vs. column index, cut along ``cut.row``.
    Right panel: intensity vs. row index, cut along ``cut.col``.

    Args:
        frame:    2-D camera frame.
        cut:      Pre-computed cut; recomputed from *frame* when omitted.
        title:    Suptitle.
        subtitle: Footer naming the provenance of the frame.
        figsize:  Figure size in inches.
        dpi:      Resolution used only by the saving wrapper.

    Returns:
        The :class:`matplotlib.figure.Figure`; the caller closes it.
    """
    import matplotlib.pyplot as plt

    _apply_cjk_font()
    f = np.asarray(frame, dtype=float)
    if cut is None:
        cut = cut_point(f)
    cut = _realign(cut, f.shape)

    row_i, col_i = int(round(cut.row)), int(round(cut.col))
    row_i = min(max(row_i, 0), f.shape[0] - 1)
    col_i = min(max(col_i, 0), f.shape[1] - 1)

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    panels = (
        (axes[0], f[row_i, :], col_i, f"过光斑中心行的行剖面（y = {row_i}）", "像素 x"),
        (axes[1], f[:, col_i], row_i, f"过光斑中心列的列剖面（x = {col_i}）", "像素 y"),
    )
    for ax, profile, centre, panel_title, axis_label in panels:
        ax.plot(np.arange(profile.shape[0]), profile, lw=1.1, color="#1f77b4")
        ax.axvline(centre, color="#d62728", ls="--", lw=1.0, alpha=0.85)
        ax.set_title(panel_title, fontsize=11)
        ax.set_xlabel(axis_label)
        ax.set_ylabel("强度（count）")
        ax.grid(alpha=0.3)

    axes[0].annotate(
        cut.summary(),
        xy=(0.02, 0.97),
        xycoords="axes fraction",
        va="top",
        ha="left",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#999", alpha=0.88),
    )
    if title:
        fig.suptitle(title, fontsize=12)
    if subtitle:
        fig.text(0.5, 0.012, subtitle, ha="center", fontsize=8, color="#444")
    fig.tight_layout(rect=(0, 0.075 if subtitle else 0.0, 1, 0.94 if title else 1.0))
    return fig


def _realign(cut: ProfileCut, shape: tuple[int, ...]) -> ProfileCut:
    """Clamp a cut computed for one frame so it is valid for *shape*."""
    if cut.frame_shape == (shape[0], shape[1]):
        return cut
    return ProfileCut(
        row=cut.row,
        col=cut.col,
        peak=cut.peak,
        background_median=cut.background_median,
        argmax_row=cut.argmax_row,
        argmax_col=cut.argmax_col,
        frame_shape=(int(shape[0]), int(shape[1])),
    )


def save_xy_profile(
    frame: np.ndarray,
    out_path: str | Path,
    *,
    cut: ProfileCut | None = None,
    title: str = "",
    subtitle: str = "",
    dpi: int = 130,
) -> Path:
    """Render and write the X/Y cross-section figure. Returns the written path."""
    import matplotlib.pyplot as plt

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plot_xy_profile(frame, cut=cut, title=title, subtitle=subtitle, dpi=dpi)
    try:
        fig.savefig(path, dpi=dpi)
    finally:
        plt.close(fig)
    return path


def save_far_field(
    frame: np.ndarray,
    out_path: str | Path,
    *,
    title: str = "",
    marker_side: int | None = None,
    hide_axes: bool = True,
) -> Path:
    """Write the camera frame, delegating to the canonical best-image renderer."""
    from ao_shaping.display.frames import save_best_image

    return Path(
        save_best_image(
            np.asarray(frame),
            Path(out_path),
            title=title,
            marker_side=marker_side,
            hide_axes=hide_axes,
        )
    )


def save_phase(
    phase: np.ndarray,
    out_path: str | Path,
    *,
    title: str = "",
    dpi: int = 150,
) -> Path:
    """Write the SLM pattern.

    A ``uint16`` array is the grayscale actually sent to the panel (2π ==
    device ``_max_gray``) and is rendered on a gray ramp so the wrap period is
    legible. Anything else is treated as raw unwrapped radians and wrapped to
    [0, 2π) for display only -- the stored array is never modified.
    """
    import matplotlib.pyplot as plt

    arr = np.asarray(phase)
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if arr.dtype in PHASE_LIKE_DTYPES:
        shown, cmap, vmin, vmax = arr, "gray", 0.0, float(arr.max())
    else:
        wrapped = np.mod(np.asarray(arr, dtype=float), 2 * np.pi)
        shown, cmap, vmin, vmax = wrapped, "gray", 0.0, float(2 * np.pi)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(shown, cmap=cmap, vmin=vmin, vmax=vmax)
    if title:
        ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    try:
        fig.savefig(path, dpi=dpi)
    finally:
        plt.close(fig)
    return path


def _name(key: str) -> str:
    """Record key -> file-name fragment (``_img`` -> ``img``, no double dash)."""
    return key.lstrip("_") or "panel"


def _artifact_path(stem: str, epoch: int, key: str, kind: str) -> str:
    """``<stem>_e0007_<key>_<kind>``, without the kind suffix when redundant."""
    frag = _name(key)
    suffix = "" if frag == kind else f"_{kind}"
    return f"{stem}_e{epoch:04d}_{frag}{suffix}.png"


def _partition_keys(
    data: dict[int, dict],
    frame_keys: Sequence[str] | None,
    phase_keys: Sequence[str] | None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Decide which record keys are far-field frames and which are phases.

    Explicit arguments select *candidates*; they do not override dtype. A
    ``uint16`` array is the SLM grayscale actually sent to the panel, and
    rendering one as a far-field image -- let alone running
    ``prepare_frame`` + ``centroid`` over it to draw an "X/Y 光强截面" of a
    *phase* -- produces a confident, meaningless artefact. ``slm_gs_refine``
    really does list ``_phase`` / ``_phase_pos`` / ``_phase_neg`` inside
    ``img_keys``, so this reclassification is load-bearing, not defensive.

    Without explicit keys, a 2-D array that keeps one shape across every epoch
    is taken as a camera frame -- the same structural test the existing debug
    summary panel relies on.
    """
    explicit_frames = tuple(frame_keys or ())
    explicit_phases = tuple(phase_keys or ())

    candidates = explicit_frames or {
        k for rec in data.values() for k in rec
    }

    frames: list[str] = []
    phases: list[str] = list(explicit_phases)
    for key in sorted(candidates):
        arrays = [np.asarray(rec[key]) for rec in data.values() if rec.get(key) is not None]
        arrays = [a for a in arrays if a.ndim == 2]
        if not arrays:
            if key in explicit_phases:
                phases.append(key)
            continue
        if any(a.dtype in PHASE_LIKE_DTYPES for a in arrays):
            if key not in phases:
                phases.append(key)
            continue
        if explicit_frames or all(a.shape == arrays[0].shape for a in arrays):
            frames.append(key)
    return tuple(frames), tuple(phases)


def select_epochs(
    epochs: Sequence[int],
    *,
    obj_key: str | None = None,
    data: dict[int, dict] | None = None,
    mode: str = "summary",
) -> list[int]:
    """Pick which epochs get companion images.

    Args:
        epochs:  All epochs, any order; duplicates are impossible (dict keys).
        obj_key: Objective column used to find the best epoch, when known.
        data:    The record dict, required when *obj_key* is given.
        mode:    ``"none"`` -> empty, ``"summary"`` -> first/best/last (bounded,
                 and the default because incremental runners re-save the whole
                 history on every flush, which would otherwise make image
                 writing quadratic), ``"all"`` -> every epoch.

    Returns:
        Epoch ids in ascending order.
    """
    ordered = sorted(epochs)
    if mode == "none" or not ordered:
        return []
    if mode == "all":
        return ordered
    if mode != "summary":
        raise ValueError(f"unknown frame-artifact mode {mode!r}")

    picks = {ordered[0], ordered[-1]}
    if obj_key and data:
        scores = [(data[e][obj_key], e) for e in ordered if e in data and obj_key in data[e]]
        if scores:
            picks.add(max(scores)[1])
    return sorted(picks)


def save_frame_artifacts(
    data: dict[int, dict],
    out_dir: str | Path,
    *,
    frame_keys: Sequence[str] | None = None,
    phase_keys: Sequence[str] | None = None,
    mode: str = "summary",
    obj_key: str | None = None,
    stem: str = "frame",
    dpi: int = 130,
    title_prefix: str = "",
) -> dict[str, str]:
    """Write far-field / X-Y profile / phase images for the selected epochs.

    This is the function :func:`ao_shaping.utils.io.file.save_optimization_debug_artifacts`
    calls right after it writes the pickle, so a run's images and its ``.pkl``
    are always produced together.

    Naming is deterministic and sorts correctly::

        <stem>_e{epoch:04d}_{frame_key}_farfield.png
        <stem>_e{epoch:04d}_{frame_key}_xy_profile.png
        <stem>_e{epoch:04d}_{phase_key}_phase.png

    Args:
        data:       ``{epoch: record}`` as written to the pickle.
        out_dir:    Directory for the images.
        frame_keys: Record keys holding camera frames; auto-detected if omitted.
        phase_keys: Record keys holding SLM patterns; auto-detected if omitted.
        mode:       See :func:`select_epochs`.
        obj_key:    Objective column for picking the best epoch.
        stem:       File-name prefix.
        dpi:        Resolution for the cross-section figures.
        title_prefix: Prepended to each figure title.

    Returns:
        Mapping of written path -> artefact kind. Empty when nothing was written.
    """
    if mode == "none":
        return {}
    frames, phases = _partition_keys(data, frame_keys, phase_keys)
    if not frames and not phases:
        return {}

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    for epoch in select_epochs(data.keys(), obj_key=obj_key, data=data, mode=mode):
        rec = data.get(epoch) or {}
        label = f"epoch {epoch}" + (f" · {title_prefix}" if title_prefix else "")
        for key in frames:
            arr = rec.get(key)
            if arr is None or np.asarray(arr).ndim != 2:
                continue
            arr = np.asarray(arr)
            written[
                str(
                    save_far_field(
                        arr,
                        out / _artifact_path(stem, epoch, key, "farfield"),
                        title=f"{label} · {key}",
                    )
                )
            ] = "farfield"
            written[
                str(
                    save_xy_profile(
                        arr,
                        out / _artifact_path(stem, epoch, key, "xy_profile"),
                        title=label,
                        subtitle=f"记录字段 {key} · 切点为扣背景后的光强质心",
                        dpi=dpi,
                    )
                )
            ] = "xy_profile"
        for key in phases:
            arr = rec.get(key)
            if arr is None or np.asarray(arr).ndim != 2:
                continue
            written[
                str(
                    save_phase(
                        np.asarray(arr),
                        out / _artifact_path(stem, epoch, key, "phase"),
                        title=f"{label} · {key}",
                    )
                )
            ] = "phase"
    return written


def frame_artifact_paths(
    out_dir: str | Path, stem: str, epochs: Iterable[int]
) -> list[Path]:
    """The paths :func:`save_frame_artifacts` would write, without writing them.

    Lets a caller assert the contract (and lets tests check naming) cheaply.
    """
    base = Path(out_dir)
    return [base / f"{stem}_e{int(e):04d}" for e in sorted(epochs)]


def describe_artifact(frame: np.ndarray) -> dict[str, Any]:
    """Small JSON-able description of a frame, for debug sidecars."""
    cut = cut_point(frame)
    dx, dy = cut.offset_from_frame_center
    return {
        "shape": list(cut.frame_shape),
        "peak": cut.peak,
        "background_median": cut.background_median,
        "cut_row_col": [cut.row, cut.col],
        "argmax_row_col": [cut.argmax_row, cut.argmax_col],
        "offset_from_frame_center_px": [dx, dy],
    }