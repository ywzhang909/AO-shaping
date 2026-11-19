"""DM voltage rounding / persistence / PIB debug-plot helpers.

Extracted verbatim from :mod:`ao_shaping.runners.runner_common` (#205 slicing)
so that module stops owning the ``nlight_dm`` runners' shared IO. The public
names are re-exported from ``runner_common`` for backward compatibility; the
single implementation lives here.

Semantics are load-bearing and must not be "simplified":

* ``round_to_int`` (half-to-even) and ``truncate_to_int`` are intentionally
  different -- see https://github.com/ywzhang909/AO-shaping/issues/142.
* ``load_init_v`` warns and cold-starts on a missing file (#91 P1).
* ``save_best_voltages`` computes the dated directory once and reuses it for
  every recorder, so ``pipeline_runner``'s WF + CCD recorders land side by side.
* ``plot_pib_debug_grid`` imports matplotlib / ``plot_funcs`` lazily so this
  module carries no hard matplotlib dependency at import time.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from loguru import logger

from ao_shaping.utils.io.cli_helpers import get_date_dir_name

if TYPE_CHECKING:
    from ao_shaping.utils.io.file import Recorder

__all__ = [
    "load_init_v",
    "plot_pib_debug_grid",
    "round_to_int",
    "save_best_voltages",
    "truncate_to_int",
]


def round_to_int(arr: Any) -> np.ndarray:
    """Round half-to-even, then cast to int (the voltage-sink convention).

    One spelling for the ``process_fn`` shared by ``wf`` / ``pib`` / ``combined``:
    ``np.round`` and ``np.around(x).astype(int)`` are bit-identical once written
    through ``np.savetxt(..., fmt="%d")``.

    NOT interchangeable with :func:`truncate_to_int`; see
    https://github.com/ywzhang909/AO-shaping/issues/142
    """
    return np.around(np.asarray(arr)).astype(int)


def truncate_to_int(arr: Any) -> np.ndarray:
    """Truncate toward zero -- rounding would change the saved voltages.

    ``pipeline_runner`` has always sinked voltages this way. On a representative
    sample it disagrees with :func:`round_to_int` on 7 of 13 values (2.7 -> 2 not
    3, 99.5 -> 99 not 100, -2.7 -> -2 not -3), so the two are kept apart on
    purpose rather than collapsed into one helper.
    """
    return np.asarray(arr).astype(int)


def load_init_v(load_file: str) -> list:
    """Load initial DM voltages from a text file, or ``[]`` when absent.

    Shared by ``axis_beam_runner`` / ``combined_runner`` / ``pipeline_runner``
    (``wf_runner`` has no ``load_file`` at all). Semantics: falsy token → ``[]``;
    existing file → ``np.loadtxt(...).tolist()``; missing path → warning + ``[]``.

    Behavior alignment (#91 P1): ``pipeline_runner`` previously crashed with
    ``FileNotFoundError`` on a missing path (no existence check); the unified
    behaviour is warning + cold start. ``combined_runner`` was silently
    returning ``[]`` on a missing path and now also logs the warning.
    ``axis_beam_runner``'s ``"rms"`` sentinel is judged at the call site before
    delegating here.
    """
    if not load_file:
        return []
    if Path(load_file).exists():
        return np.loadtxt(load_file).tolist()
    logger.warning(f"load_file {load_file} not exists")
    return []


def save_best_voltages(
    records: Recorder | list[Recorder] | tuple[Recorder, ...],
    root_dir: str | Path,
    *,
    process_fn=round_to_int,
) -> Path:
    """Sink best voltages of one or more recorders into ``flatten_voltages/<date>/``.

    Shared scaffolding of the four ``nlight_dm`` runners (#91 P1): the directory
    is computed ONCE and reused for every recorder, so ``pipeline_runner``'s two
    recorders (WF + CCD) land side by side in the same directory exactly as
    before. ``process_fn`` defaults to :func:`round_to_int`;
    ``pipeline_runner`` passes :func:`truncate_to_int` (the two must NOT be
    collapsed, see #142).
    """
    save_dir = Path(root_dir) / "flatten_voltages" / get_date_dir_name()
    if not isinstance(records, (list, tuple)):
        records = [records]
    for rec in records:
        rec.save_best(saved_dir=save_dir, target="_v", process_fn=process_fn, fmt="%d")
    return save_dir


def plot_pib_debug_grid(
    *,
    init_img: np.ndarray,
    best_img: np.ndarray,
    history_values,
    voltages: np.ndarray,
    init_title: str,
    best_title: str,
    history_title: str,
    save_path: str | Path,
) -> None:
    """Draw the classic 2x2 PIB debug grid (init/best image, history, voltages).

    Shared by ``combined_runner`` and ``axis_beam_runner`` (#91 P1);
    ``axis_beam_runner`` parameterises every label by its objective column,
    ``combined_runner`` uses the ``pib`` spelling — both preserved verbatim via
    the title arguments. ``pipeline_runner``'s 2x4 WF+CCD grid and
    ``wf_runner``'s wavefront grid are different layouts and stay inline.
    matplotlib / plot_funcs are imported lazily so this module does not gain a
    hard matplotlib dependency at import time (mirrors ``Recorder.save_best``).
    """
    import matplotlib.pyplot as plt

    from ao_shaping.display.image_display import plot_funcs

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    plot_funcs["img"](init_img, ax[0, 0], init_title)
    axim = plot_funcs["img"](best_img, ax[0, 1], best_title)
    fig.colorbar(axim, ax=[ax[0, 0], ax[0, 1]], orientation="horizontal")
    plot_funcs["pib_history"](history_values, ax[1, 0], title=history_title)
    plot_funcs["voltages"](voltages, ax[1, 1], "Best Voltages")
    plt.savefig(save_path)
    plt.close()
