"""Shared helpers for AO-Shaping runner scripts.

A single place for the patterns that appear verbatim in 3+ runner files,
keeping each runner focused on what makes it unique while pulling
boilerplate out of every file.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.axes as mpl_axes
import matplotlib.pyplot as plt
import numpy as np

from loguru import logger

from ao_shaping.drivers.dm import create_dm, list_reachable_dm_types
from ao_shaping.drivers.dm.base import DM
from ao_shaping.utils.image.display import plot_funcs  # noqa: E402  (after matplotlib)


# ---------------------------------------------------------------------------
# Save-path construction
# ---------------------------------------------------------------------------


def build_debug_save_paths(
    root_dir,
    context_subdir: Path | str,
) -> tuple[Path, Path]:
    """Build the date-stamped save directory + file-stem prefix used by all runners.

    Consistently mirrors the pattern::

        save_dir = gen_date_dir(Path(root_dir) / context_subdir)
        saved_file_name = save_dir / f"{context_subdir}_{save_dir.name}"

    Args:
        root_dir:     Top-level run-output root (e.g. ``"data"``).
        context_subdir:
            The runner-specific sub-path *relative to root_dir*.
            May be a :class:`~pathlib.Path` or a plain string, e.g.::

                ``"flatten_zernike"``
                ``"flatten_voltages"``
                ``Path("pipeline")``

    Returns:
        A 2-tuple ``(save_dir, saved_file_name)`` where ``save_dir`` is the
        full :class:`~pathlib.Path` to the date-stamped save directory and
        ``saved_file_name`` is a stem :class:`~pathlib.Path` (no extension)
        inside it, used to derive ``.png`` / ``.pkl`` / ``.json`` / ``.zip``
        siblings.
    """
    from ao_shaping.utils import gen_date_dir

    save_dir = gen_date_dir(Path(root_dir) / context_subdir)
    saved_file_name = save_dir / f"{context_subdir}_{save_dir.name}"
    return save_dir, saved_file_name


# ---------------------------------------------------------------------------
# Optimisation-debug visualisation block
# ---------------------------------------------------------------------------


def make_debug_wavefront_ax_plots(
    ax: object,
    init_wavefront,
    opt_wavefront,
    init_title: str = "init wavefront",
    opt_title: str = "opt wavefront",
    orientation: str = "horizontal",
) -> None:
    """Place init / optimised wavefront images on a pair of Axes.

    Consumed with the 2×2 debug grid layout::

        fig, ax = plt.subplots(2, 2, figsize=(12, 9))
        ...  # rms_history  → ax[0,0],   voltages  → ax[0,1]
        make_debug_wavefront_ax_plots(ax[1], init_wf, min_wf)   # init→col0, opt→col1

    Args:
        ax:             A 1-D Axes slice such as ``ax[1]`` (shaped n_cols,).
        init_wavefront: Wavefront array for the initial state.
        opt_wavefront:  Wavefront array for the optimised state.
        init_title:     Title string for the *init* panel.
        opt_title:      Title string for the *opt* panel.
        orientation:    Colorbar orientation (``"horizontal"`` or
                        ``"vertical"``).
    """
    im0 = plot_funcs["wavefront"](init_wavefront, ax[0], init_title)
    fig = ax[0].get_figure()
    plt.colorbar(im0, ax=ax[0], orientation=orientation)
    im1 = plot_funcs["wavefront"](opt_wavefront, ax[1], opt_title)
    plt.colorbar(im1, ax=ax[1], orientation=orientation)


# Objective column names recognised in data-mode records (slm-pib).
_DATA_MODE_OBJECTIVE_KEYS = (
    "pib",
    "radiu",
    "avg_radiu",
    "rmse",
    "shape",
    "roi_pib",
    "rms_pib",
)

# Objectives that are MINIMISED (used to pick the best-epoch frame/coeffs).
_DATA_MODE_MINIMIZED_KEYS = ("radiu", "rmse")

# Guard-penalised rows use J < -100 to signal abandonment (energy guard).
_DATA_MODE_PENALTY_THRESHOLD = -100.0


def _infer_objective_key(data: dict[int, dict]) -> str | None:
    """Return the objective column name present in the first data record."""
    first = next(iter(data.values()), {})
    for key in _DATA_MODE_OBJECTIVE_KEYS:
        if key in first:
            return key
    return None


def _best_epoch(data: dict[int, dict], obj_key: str | None) -> int | None:
    """Return the epoch whose recorded ``best_<obj>`` is optimal.

    ``best_<obj>`` is the running-best value recorded on each row, so the row
    where it reaches its optimum is the epoch that produced the best search
    state (its ``_c`` / ``_img`` show the optimal coefficients / spot).
    """
    if obj_key is None:
        return None
    best_key = f"best_{obj_key}"
    candidates = [e for e in data if best_key in data[e] and "_img" in data[e]]
    if not candidates:
        return None
    if obj_key in _DATA_MODE_MINIMIZED_KEYS:
        return min(candidates, key=lambda e: float(data[e][best_key]))
    return max(candidates, key=lambda e: float(data[e][best_key]))


def _overlay_target_roi(
    ax: "mpl_axes.Axes", img, target_box: dict | None, label: str
) -> None:
    """imshow ``img`` on ``ax`` and overlay the target-shaped ROI border.

    The ROI mask is generated by ``target_shape_roi`` (deferred import to keep
    this shared runner module free of an optimizer dependency at import time)
    and drawn as a contour so any target shape renders as a closed outline.

    Args:
        ax:        Target matplotlib Axes.
        img:       2-D image array (window-local frame, same frame as the
                   ROI centre recorded in ``target_box``).
        target_box:
            Optional dict with ``center`` (window-local x,y), ``shape``,
            ``size``, ``aspect_ratio`` keys, as recorded by the optimizer.
        label:     Panel title.
    """
    ax.set_title(label)
    if img is None:
        ax.text(0.5, 0.5, "no _img", ha="center", va="center")
        return
    ax.imshow(np.asarray(img))
    if not target_box:
        return
    try:
        from ao_shaping.optimizer.wfless.slm_zernike_pib import target_shape_roi

        img_h = int(np.asarray(img).shape[0])
        img_w = int(np.asarray(img).shape[1])
        cx, cy = (float(v) for v in target_box["center"])
        roi = target_shape_roi(
            (img_h, img_w),
            center=(cx, cy),
            shape=str(target_box["shape"]),
            size=float(target_box["size"]),
            aspect_ratio=float(target_box.get("aspect_ratio", 4.0 / 3.0)),
        )
        ax.contour(roi.astype(np.int8), levels=[0.5], colors="red", linewidths=1.5)
    except Exception:
        logger.warning("target ROI overlay failed for {}", label)


def _save_data_mode_debug_artifacts(
    data: dict[int, dict],
    png_path: Path,
    pkl_path: Path,
    json_path: Path,
    title: str,
    json_payload: dict | None,
    target_box: dict | None = None,
) -> Path:
    """Write PNG / pkl / json debug artifacts for a ``{epoch: record}`` dict.

    Figure layout (2×2): objective history line plot (top-left), best
    coefficient bar chart (top-right), first ``_img`` (bottom-left), best
    ``_img`` (bottom-right). When a ``target_box`` dict (window-local
    ``center`` / ``shape`` / ``size`` / ``aspect_ratio``) is supplied, the
    target-shaped ROI border is overlaid on both spot panels, matching the
    axis_beam_runner convention of showing the shaping target on the frames.
    """
    epochs = sorted(data.keys())
    obj_key = _infer_objective_key(data)
    best_ep = _best_epoch(data, obj_key)

    fig, ax = plt.subplots(2, 2, figsize=(12, 9))

    if obj_key is not None:
        xs = [e for e in epochs if obj_key in data[e]]

        def _clean(vals: list[float]) -> list[float]:
            # Drop guard-penalised evaluations (energy guard records J-1e3, so
            # values below the penalty threshold are abandonment rows, not real
            # objective measurements.
            return [v for v in vals if v > _DATA_MODE_PENALTY_THRESHOLD]

        ys = _clean([data[e][obj_key] for e in xs])
        xc = [
            e
            for e, v in zip(xs, [data[e][obj_key] for e in xs])
            if v > _DATA_MODE_PENALTY_THRESHOLD
        ]
        ax[0, 0].plot(xc, ys, label=obj_key)
        # Overlay the rms_pib component terms + J whenever present: the raw
        # objective alone hides whether a change came from pib/rms/energy.
        term_keys = [
            k
            for k in ("J", "pib_term", "rms_term", "ee_term")
            if k in data.get(xs[0] if xs else epochs[0], {}) and k != obj_key
        ]
        for k in term_keys:
            tk_xs_all = [e for e in xs if k in data[e]]
            tk_vals = [data[e][k] for e in tk_xs_all]
            tk_pairs = [
                (e, v)
                for e, v in zip(tk_xs_all, tk_vals)
                if v > _DATA_MODE_PENALTY_THRESHOLD
            ]
            if tk_pairs:
                fx, fy = zip(*tk_pairs)
                ax[0, 0].plot(fx, list(fy), "--", alpha=0.7, label=k)
        ax[0, 0].set_xlabel("epoch")
        ax[0, 0].set_ylabel(obj_key)
        ax[0, 0].legend(fontsize=7)
    ax[0, 0].set_title(title)

    c_arr = None
    for e in reversed(epochs) if best_ep is None else [best_ep]:
        if "_c" in data[e]:
            c_arr = np.asarray(data[e]["_c"])
            break
    ax[0, 1].set_title(
        f"best coefficients{' @ epoch ' + str(best_ep) if best_ep is not None else ''}"
    )
    if c_arr is not None:
        ax[0, 1].bar(range(len(c_arr)), c_arr)
    else:
        ax[0, 1].text(0.5, 0.5, "no _c", ha="center", va="center")

    imgs = [data[e]["_img"] for e in epochs if "_img" in data[e]]
    best_img = None
    if best_ep is not None and "_img" in data[best_ep]:
        best_img = data[best_ep]["_img"]
    elif imgs:
        best_img = imgs[-1]
    _overlay_target_roi(
        ax[1, 0],
        imgs[0] if imgs else None,
        target_box,
        f"first _img{' (init spot)' if target_box else ''}",
    )
    _overlay_target_roi(
        ax[1, 1],
        best_img,
        target_box,
        f"best _img @ epoch {best_ep}" if best_ep is not None else "best _img",
    )

    plt.tight_layout()
    plt.savefig(png_path)
    plt.close()

    with open(pkl_path, "wb") as f:
        pickle.dump(data, f)
    with open(json_path, "w", encoding="utf8") as f:
        json.dump(json_payload, f, ensure_ascii=False, indent=4)

    return png_path


def save_optimization_debug_artifacts(
    records=None,
    save_dir: Path | None = None,
    saved_file_name: Path | None = None,
    min_epoch: int | None = None,
    min_metric: float | None = None,
    best_coeff_key: str | None = None,
    init_wavefront=None,
    opt_wavefront=None,
    init_title: str = "init wavefront",
    opt_title: str = "opt wavefront",
    plot_params_note: str | None = None,
    *,
    data: dict[int, dict] | None = None,
    png_path: Path | None = None,
    pkl_path: Path | None = None,
    json_path: Path | None = None,
    title: str = "",
    json_payload: dict | None = None,
    target_box: dict | None = None,
) -> Path | None:
    """Emit the standard debug artifacts for a completed optimisation run.

    Two mutually exclusive modes:

    * **Wavefront mode** (default): ``records`` is a Recorder / OptHistory
      object; writes the 2×2 debug PNG + compressed dataframe. This is the
      historical behaviour used by ``ga_zernike_runner``,
      ``greedy_zernike_runner`` and ``rms_zernike_runner``.
    * **Data mode**: ``data`` is a ``{epoch: record}`` dict; writes a 2×2
      PNG (objective history / best coefficients / first & last image), a
      pickled copy of ``data`` and a JSON sidecar of ``json_payload``.
      Used by ``slm_pib_runner``.

    Args:
        records:           Recorder / OptHistory object with ``get_sublist()``,
                           ``get_best_iter()``, ``first``, and
                           ``save_dataframe``. (wavefront mode)
        save_dir:          Directory in which to write output files.
        saved_file_name:   UUID filename prefix (no extension).
        min_epoch:         Epoch index of the best result.
        min_metric:        Primary metric value at *min_epoch*
                           (RMS, PIB, …).
        best_coeff_key:    Dictionary key for the coefficient vector
                           in the best-epoch record (e.g. ``"_c"`` or ``"_v"``).
        init_wavefront:    Wavefront array for the initial state.
        opt_wavefront:     Wavefront array for the optimised state.
        init_title:        Colorbar / panel title for the initial WF.
        opt_title:         Colorbar / panel title for the optimised WF.
        plot_params_note:  Optional suffix appended to the *voltages* panel
                           title (e.g. ``"epoch=200"``).
        data:              ``{epoch: record}`` dict of scalar/array fields.
                           (data mode; mutually exclusive with ``records``)
        png_path:          Output path for the figure. (data mode)
        pkl_path:          Output path for the pickled ``data``. (data mode)
        json_path:         Output path for the JSON ``json_payload``. (data mode)
        title:             Figure title. (data mode)
        json_payload:      Dict serialised to ``json_path``. (data mode)
        target_box:        Optional dict with window-local ``center`` (x,y),
                           ``shape``, ``size``, ``aspect_ratio`` keys (as
                           recorded by the optimizer) — when given, the
                           target-shaped ROI border is overlaid on the
                           init/best spot panels of the data-mode PNG.
                           (data mode)

    Returns:
        ``png_path`` in data mode, ``None`` in wavefront mode.
    """
    if data is not None:
        if records is not None:
            raise ValueError(
                "pass either records (wavefront mode) or data (data mode), not both"
            )
        if png_path is None or pkl_path is None or json_path is None:
            raise ValueError("data mode requires png_path, pkl_path and json_path")
        return _save_data_mode_debug_artifacts(
            data=data,
            png_path=png_path,
            pkl_path=pkl_path,
            json_path=json_path,
            title=title,
            json_payload=json_payload,
            target_box=target_box,
        )

    # --- wavefront mode (historical behaviour) ---
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))

    rms_values = records.get_sublist()
    plot_funcs["rms_history"](rms_values, ax[0, 0], min_epoch, min_metric)

    best_coeffs = records.get_best_iter()[0][best_coeff_key]
    title_suffix = (
        f"{plot_params_note}"
        if plot_params_note
        else f"{min_metric:.3f} @ epoch {min_epoch}"
    )
    plot_funcs["voltages"](best_coeffs, ax[0, 1], title_suffix)

    make_debug_wavefront_ax_plots(
        ax[1], init_wavefront, opt_wavefront, init_title=init_title, opt_title=opt_title
    )

    plt.tight_layout()
    plt.savefig(saved_file_name.with_suffix(".png"))
    plt.close()

    records.save_dataframe(saved_file_name.with_suffix(".zip"), compression="zip")
    return None


# ---------------------------------------------------------------------------
# save_capture_and_dataframe convenience
# ---------------------------------------------------------------------------


def save_recorder_artifacts(
    records,
    save_dir: Path,
    saved_file_name: Path,
) -> None:
    """Save ``.zip`` dataframe + ``.png`` figure for a recorder, exactly as
    ``ga_zernike`` / ``greedy_zernike`` / ``rms_zernike`` do to wrap up
    their ``if debug`` block before returning.

    The figure / data-array content is up to the caller (they should call
    :func:`save_optimization_debug_artifacts` or build the axes directly);
    this helper handles only the final two ``save`` / ``close`` calls.

    Args:
        records:         Recorder / OptHistory object.
        save_dir:        Directory in which to write output files.
        saved_file_name: UUID filename prefix (no extension).
                            (Obtained from
                            :func:`build_debug_save_paths`.)
    """
    records.save_dataframe(saved_file_name.with_suffix(".zip"), compression="zip")


# ---------------------------------------------------------------------------
# DM resolution
# ---------------------------------------------------------------------------


def resolve_dm(dm_type: str | None, **kwargs) -> DM:
    """Resolve the DM type and create a DM instance.

    Mirrors the DM-selection block shared by the ``wf`` / ``pipeline`` /
    ``pib`` / ``combined`` / ``dm-matrix`` runners: an explicit
    ``--dm_type`` is lowercased and used directly; otherwise the reachable
    DM types are probed and the single reachable one is chosen, with errors
    for zero / multiple candidates.

    Args:
        dm_type: Explicit DM type name, or ``None`` for auto-detection.
        **kwargs: Extra constructor kwargs forwarded to ``create_dm``
            (e.g. ``keep_when_exit``, ``max_neibor_diff``,
            ``dm_neibor_diff``).

    Returns:
        A created DM instance.

    Raises:
        RuntimeError: If no DM is reachable, or multiple DMs are reachable
            while ``dm_type`` is ``None``.
    """
    if dm_type is not None:
        dm_type = dm_type.lower()
        logger.info("Using specified DM type: {}", dm_type)
    else:
        reachable = list_reachable_dm_types()
        if len(reachable) == 1:
            dm_type = reachable[0]
            logger.info("Auto-detected reachable DM: {}", dm_type)
        elif len(reachable) == 0:
            raise RuntimeError(
                "No DM reachable. Specify --dm_type explicitly or connect a DM."
            )
        else:
            raise RuntimeError(
                f"Multiple DMs reachable ({', '.join(reachable)}). "
                f"Specify --dm_type explicitly to choose one."
            )

    return create_dm(dm_type, **kwargs)
