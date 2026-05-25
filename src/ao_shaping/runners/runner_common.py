"""Shared helpers for AO-Shaping runner scripts.

A single place for the patterns that appear verbatim in 3+ runner files,
keeping each runner focused on what makes it unique while pulling
boilerplate out of every file.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ao_shaping.utils.display import plot_funcs  # noqa: E402  (after matplotlib)


# ---------------------------------------------------------------------------
# Save-path construction
# ---------------------------------------------------------------------------

def build_debug_save_paths(
    root_dir,
    context_subdir: Path | str,
) -> Path:
    """Build the date-stamped save directory used by all runners.

    Consistently mirrors the pattern::

        save_dir = gen_date_dir(Path(root_dir) / context_subdir)

    Args:
        root_dir:     Top-level run-output root (e.g. ``"data"``).
        context_subdir:
            The runner-specific sub-path *relative to root_dir*.
            May be a :class:`~pathlib.Path` or a plain string, e.g.::

                ``"flatten_zernike"``
                ``"flatten_voltages"``
                ``Path("pipeline")``

    Returns:
        Full :class:`~pathlib.Path` to the date-stamped save directory.
    """
    from ao_shaping.utils import gen_date_dir
    return gen_date_dir(Path(root_dir) / context_subdir)


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


def save_optimization_debug_artifacts(
    records,
    save_dir: Path,
    saved_file_name: Path,
    min_epoch: int,
    min_metric: float,
    best_coeff_key: str,
    init_wavefront,
    opt_wavefront,
    init_title: str,
    opt_title: str,
    plot_params_note: str | None = None,
) -> None:
    """Emit the standard debug PNG + compressed dataframe for a completed
    optimisation run.

    This is the single shared sequence that used to appear in
    ``ga_zernike_runner``, ``greedy_zernike_runner``, and
    ``rms_zernike_runner``:

    .. code-block:: python

        fig, ax = plt.subplots(2, 2, figsize=(12, 9))
        plot_funcs["rms_history"](rms_values, ax[0, 0], min_epoch, min_metric)
        plot_funcs["voltages"](..., ax[0, 1], ...)
        # wavefront panels via make_debug_wavefront_ax_plots / manual draw
        plt.savefig(saved_file_name.with_suffix(".png"))
        plt.close()
        records.save_dataframe(saved_file_name.with_suffix(".zip"),
                               compression="zip")

    Callers who need a non-standard figure layout should build the figure
    themselves and only use :func:`build_debug_save_paths` + the slice of
    this routine they want.

    Args:
        records:           Recorder / OptHistory object with ``get_sublist()``,
                           ``get_best_iter()``, ``first``, and
                           ``save_dataframe``.
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
    """
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
