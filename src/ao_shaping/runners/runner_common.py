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
_DATA_MODE_OBJECTIVE_KEYS = ("pib", "radiu", "avg_radiu")


def _infer_objective_key(data: dict[int, dict]) -> str | None:
    """Return the objective column name present in the first data record."""
    first = next(iter(data.values()), {})
    for key in _DATA_MODE_OBJECTIVE_KEYS:
        if key in first:
            return key
    return None


def _save_data_mode_debug_artifacts(
    data: dict[int, dict],
    png_path: Path,
    pkl_path: Path,
    json_path: Path,
    title: str,
    json_payload: dict | None,
) -> Path:
    """Write PNG / pkl / json debug artifacts for a ``{epoch: record}`` dict.

    Figure layout (2×2): objective history line plot (top-left), best
    coefficient bar chart (top-right), first ``_img`` (bottom-left), last
    ``_img`` (bottom-right).
    """
    epochs = sorted(data.keys())

    fig, ax = plt.subplots(2, 2, figsize=(12, 9))

    obj_key = _infer_objective_key(data)
    if obj_key is not None:
        xs = [e for e in epochs if obj_key in data[e]]
        ys = [data[e][obj_key] for e in xs]
        ax[0, 0].plot(xs, ys)
        ax[0, 0].set_xlabel("epoch")
        ax[0, 0].set_ylabel(obj_key)
    ax[0, 0].set_title(title)

    c_arr = None
    for e in reversed(epochs):
        if "_c" in data[e]:
            c_arr = np.asarray(data[e]["_c"])
            break
    ax[0, 1].set_title("best coefficients")
    if c_arr is not None:
        ax[0, 1].bar(range(len(c_arr)), c_arr)
    else:
        ax[0, 1].text(0.5, 0.5, "no _c", ha="center", va="center")

    imgs = [data[e]["_img"] for e in epochs if "_img" in data[e]]
    for ax_i, label, img in (
        (ax[1, 0], "first _img", imgs[0] if imgs else None),
        (ax[1, 1], "last _img", imgs[-1] if imgs else None),
    ):
        ax_i.set_title(label)
        if img is not None:
            ax_i.imshow(np.asarray(img))
        else:
            ax_i.text(0.5, 0.5, "no _img", ha="center", va="center")

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

    Returns:
        ``png_path`` in data mode, ``None`` in wavefront mode.
    """
    if data is not None:
        if records is not None:
            raise ValueError(
                "pass either records (wavefront mode) or data (data mode), not both"
            )
        return _save_data_mode_debug_artifacts(
            data=data,
            png_path=png_path,
            pkl_path=pkl_path,
            json_path=json_path,
            title=title,
            json_payload=json_payload,
        )

    # --- wavefront mode (historical behaviour) ---
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))

    rms_values = records.get_sublist()
    plot_funcs["rms_history"](rms_values, ax[0, 0], min_epoch, min_metric)

    best_coeffs = records.get_best_iter()[0][best_coeff_key]
    title_suffix = (
        f"{plot_params_note}" if plot_params_note
        else f"{min_metric:.3f} @ epoch {min_epoch}"
    )
    plot_funcs["voltages"](best_coeffs, ax[0, 1], title_suffix)

    make_debug_wavefront_ax_plots(ax[1], init_wavefront, opt_wavefront,
                                   init_title=init_title, opt_title=opt_title)

    plt.tight_layout()
    plt.savefig(saved_file_name.with_suffix(".png"))
    plt.close()

    records.save_dataframe(saved_file_name.with_suffix(".zip"),
                          compression="zip")
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
    records.save_dataframe(
        saved_file_name.with_suffix(".zip"), compression="zip"
    )


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
