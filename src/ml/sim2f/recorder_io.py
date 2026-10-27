"""Record a 2f-bench session with the repository's ``Recorder`` on-disk contract.

What this module is
===================
A thin, **hardware-free** adapter that lets one session on the 2f experiment
system (SLM + CCD) be written down with the *same* column vocabulary as every
other optimisation run in this repository:

* the column names come from :mod:`ao_shaping.utils.io.recorder_keys` -- no
  new key names, no parallel spellings;
* the bytes are written by :func:`ao_shaping.utils.io.file.save_history`
  (CSV + one ``.npy`` sidecar per array column per row), i.e. the very same
  path the other runners take.

It exists because the corpus reader (:mod:`ml.hwdataset.index`) and the report
generators already understand ``_id`` / ``_epoch`` / ``_img`` / ``_c`` /
``_phase``, while a bench session would otherwise invent a private vocabulary
-- exactly the "two implementations drift" failure this repository keeps
hitting.  So the rule here is narrow and absolute: **no new key names**.

What lands on disk
==================
``<out_dir>/bench_session.csv``
    One row per measurement.  The scalar columns hold plain values; the array
    columns hold numpy's *edge-item summary* of the array (numpy's own
    ``str()`` of an array above 1000 elements is a 6-line digest, not the
    payload).  The authoritative array bytes are in the sidecars.
``<out_dir>/bench_session_sidecars/``
    ``bench_session_<column>_<row:04d>.npy`` -- one file per array column per
    row, written by the canonical sidecar mechanism.
``<out_dir>/bench_session_meta.json``
    Row count, column list, per-column dtype/shape summary, and the **byte
    volume of every array column**.  The sidecars are large -- a
    ``1200x1920`` uint16 panel is 4.4 MiB per row -- so the volume has to be
    visible up front instead of discovered after the disk fills up.

Row layout
==========
======================  =========================================================
Column                  Content
======================  =========================================================
``_id``                 Row index, assigned by :meth:`Recorder.append`.
``_epoch``              The sample ``index`` the caller passed in.
``_img``                The **measured** CCD frame, 2-D float32.
``_c``                  The **command** phase grid, 2-D float32.
``_phase``              The uint16 grayscale **actually sent to the panel**.
``kind``                Free-form sample label (plain name, not a key).
``condition``           One of :data:`CONDITIONS` (the A1/B1/B2/A2 quartet).
``index``               The sample index (plain name, not a key).
``<metric>``            Every scalar of the ``metrics`` mapping, verbatim.
======================  =========================================================

``_c`` and the "length carries the meaning" convention
------------------------------------------------------
:mod:`ao_shaping.utils.io.recorder_keys` documents ``_c`` as a *command
vector* whose meaning is carried by its length (a triangular Zernike count vs
a ``grid*grid`` freeform grid).  This module stores the **2-D** command grid,
so ``len(_c)`` is the grid *side*, not ``grid*grid``.  That agrees with the
"length carries the meaning" convention whenever the side is itself a perfect
square (``64 = 8**2``, ``256 = 16**2``), which is why the grid size is worth
picking deliberately; ``128`` is *not* a perfect square.  Two mitigating facts
keep this safe for the corpus reader: :func:`ml.hwdataset.index` classifies a
record by ``_phase`` first (a uint16 ``_phase`` is a panel-gray record no
matter what ``_c`` holds), and this module always writes ``_phase``.

Measured 2026-10-27: pickling the inner :class:`Recorder` and running
:func:`ml.hwdataset.build_hw_index` over it keeps all four rows and classifies
every one of them as ``PhaseSource.PANEL_GRAY``, with ``n_terms == 128``
(``len(_c)``).  So the recorded session is directly consumable by the corpus
reader -- which is the whole point of reusing the vocabulary.

No hardware
===========
This module imports no driver and opens no device: it only moves arrays and
numbers.  A session driver stays in charge of the acquisition and hands the
measurements over.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from loguru import logger

from ao_shaping.utils.io.file import Recorder
from ao_shaping.utils.io.recorder_keys import (
    RECORD_COEFFS,
    RECORD_EPOCH,
    RECORD_ID,
    RECORD_IMAGE,
    RECORD_KEYS,
    RECORD_PHASE,
)

__all__ = [
    "CSV_NAME",
    "CSV_STEM",
    "CONDITIONS",
    "COLUMN_CONDITION",
    "COLUMN_INDEX",
    "COLUMN_KIND",
    "META_NAME",
    "SIDECAR_DIR_NAME",
    "BenchRecorder",
    "MissingMarkError",
]

#: Name of the CSV the session is written to.
CSV_NAME = "bench_session.csv"
#: Stem of :data:`CSV_NAME`; also the prefix of every sidecar file name.
CSV_STEM = "bench_session"
#: Name of the sidecar directory (one ``.npy`` per array column per row).
SIDECAR_DIR_NAME = "bench_session_sidecars"
#: Name of the metadata sidecar.
META_NAME = "bench_session_meta.json"

#: Plain (non-canonical) column names.  They carry no leading underscore on
#: purpose: a leading underscore means "canonical recorder key", and a column
#: that is *not* a canonical key must not pretend to be one.
COLUMN_KIND = "kind"
COLUMN_CONDITION = "condition"
COLUMN_INDEX = "index"

#: The conditions of the session's A1/B1/B2/A2 quartet measurement.
CONDITIONS: tuple[str, ...] = ("a1", "b1", "b2", "a2")

#: Column names this module owns besides the canonical keys.
_PLAIN_COLUMNS: tuple[str, ...] = (COLUMN_KIND, COLUMN_CONDITION, COLUMN_INDEX)


class MissingMarkError(KeyError):
    """The ``mark`` metric is missing from a sample.

    ``Recorder.append`` only *asserts* that the mark is present, and an
    ``assert`` disappears under ``python -O``; this module therefore checks
    the mark itself and raises a named exception instead.
    """


class BenchRecorder:
    """Record one bench session's per-frame measurements, ``Recorder`` style.

    The class owns a :class:`ao_shaping.utils.io.file.Recorder` and adds the
    bench-specific part: the fixed set of columns (measured frame, command
    phase, panel grayscale) and the plain-name bookkeeping columns.

    Example:
        >>> rec = BenchRecorder(mark="ncc_net")
        >>> rec.append_sample(
        ...     index=0,
        ...     kind="flat",
        ...     condition="a1",
        ...     frame=frame,
        ...     command_phase=phase,
        ...     panel_grayscale=gray,
        ...     metrics={"ncc_net": 0.87},
        ... )  # doctest: +SKIP
        >>> paths = rec.save("data/debug/bench_session")  # doctest: +SKIP
    """

    def __init__(self, *, mark: str = "ncc_net", mode: str = "max") -> None:
        """Bind the recorder to one objective column.

        Args:
            mark: Name of the metric column the session is judged by.  It must
                be present in every sample's ``metrics`` mapping.
            mode: ``"max"`` or ``"min"`` -- the direction
                :meth:`Recorder.get_best_iter` searches in.

        Raises:
            ValueError: If ``mark`` is empty.
        """
        if not isinstance(mark, str) or not mark:
            raise ValueError(f"mark must be a non-empty string, got {mark!r}")
        self.mark = mark
        self.recorder = Recorder(mark=mark, mode=mode)
        self.mode = self.recorder.mode

    # ------------------------------------------------------------------
    # inspection
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of samples appended so far."""
        return len(self.recorder.history)

    @property
    def history(self) -> list[dict[str, Any]]:
        """Return the recorded rows (the live list, not a copy)."""
        return self.recorder.history

    @property
    def dataframe(self):
        """Return the rows as a :class:`pandas.DataFrame`."""
        return self.recorder.dataframe

    # ------------------------------------------------------------------
    # recording
    # ------------------------------------------------------------------

    def append_sample(
        self,
        *,
        index: int,
        kind: str,
        condition: str,
        frame: np.ndarray,
        command_phase: np.ndarray,
        panel_grayscale: np.ndarray,
        metrics: Mapping[str, float],
    ) -> None:
        """Append one measurement.

        Args:
            index: Sample index; it becomes the row's ``_epoch``.
            kind: Free-form sample label (e.g. ``"flat"``, ``"shaped"``).
            condition: One of :data:`CONDITIONS` -- the A1/B1/B2/A2 quartet
                of the session.
            frame: The **measured** CCD frame; 2-D, stored as float32.
            command_phase: The **command** phase grid in radians; 2-D,
                stored as float32.  A freeform grid of ``grid*grid`` cells.
            panel_grayscale: The uint16 grayscale **actually sent to the
                panel**; 2-D, stored as uint16.
            metrics: The scalar metrics of this row (e.g. ``ncc_flat``,
                ``ncc_net``, ``objective_l1``, ``ee_in_target``,
                ``zero_order_offset_px``).  It must contain the ``mark``.

        Raises:
            MissingMarkError: If ``metrics`` does not contain the ``mark``.
            ValueError: If a condition is unknown, an array is not 2-D, or a
                metric name is not allowed.
            TypeError: If ``metrics`` is not a mapping.
        """
        if not isinstance(metrics, Mapping):
            raise TypeError(f"metrics must be a mapping, got {type(metrics).__name__}")
        if self.mark not in metrics:
            raise MissingMarkError(
                f"mark {self.mark!r} is missing from the metrics of sample "
                f"{index!r}; available metrics: {sorted(metrics)}"
            )
        if condition not in CONDITIONS:
            raise ValueError(
                f"unknown condition {condition!r}; expected one of {list(CONDITIONS)}"
            )

        row: dict[str, Any] = {
            RECORD_EPOCH: int(index),
            RECORD_IMAGE: _as_2d(frame, np.float32, "frame"),
            RECORD_COEFFS: _as_2d(command_phase, np.float32, "command_phase"),
            RECORD_PHASE: _as_2d(panel_grayscale, np.uint16, "panel_grayscale"),
            COLUMN_KIND: str(kind),
            COLUMN_CONDITION: str(condition),
            COLUMN_INDEX: int(index),
        }
        row.update(self._metric_columns(metrics, index))
        # ``Recorder.append`` assigns the row id and asserts the mark is
        # present; the mark was checked above, so the assert cannot fire.
        self.recorder.append(row)

    def _metric_columns(
        self, metrics: Mapping[str, float], index: int
    ) -> dict[str, float]:
        """Turn the metrics mapping into row columns.

        Args:
            metrics: The scalar metrics of one sample.
            index: The sample index, used in the error messages.

        Returns:
            The columns to merge into the row.

        Raises:
            ValueError: If a metric name would shadow a canonical key or one
                of the plain bookkeeping columns, or if a value is not a
                scalar.
        """
        columns: dict[str, float] = {}
        for key, value in metrics.items():
            name = str(key)
            if name.startswith("_"):
                raise ValueError(
                    f"metric {name!r} of sample {index!r} starts with an "
                    "underscore, which is reserved for the canonical recorder "
                    f"keys {list(RECORD_KEYS)}; use a plain name"
                )
            if name in RECORD_KEYS:
                raise ValueError(
                    f"metric {name!r} of sample {index!r} shadows the canonical "
                    "recorder key of the same name"
                )
            if name in _PLAIN_COLUMNS:
                raise ValueError(
                    f"metric {name!r} of sample {index!r} shadows the "
                    f"bookkeeping column of the same name {list(_PLAIN_COLUMNS)}"
                )
            try:
                columns[name] = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"metric {name!r} of sample {index!r} is not a scalar: "
                    f"{value!r}"
                ) from exc
        return columns

    # ------------------------------------------------------------------
    # saving
    # ------------------------------------------------------------------

    def save(self, out_dir: str | Path) -> dict[str, str]:
        """Write the session to disk.

        The CSV and the sidecars are written by the canonical
        :func:`ao_shaping.utils.io.file.save_history` path; the metadata
        sidecar is written next to them so the array volume is visible
        without opening the sidecars.

        Calling this twice on the same object is safe: the files are
        overwritten, and the sidecar file names are a function of the row
        count, which only ever grows.

        Args:
            out_dir: Directory to write into; created if missing.

        Returns:
            ``{"csv": ..., "sidecars": ..., "json": ...}`` -- the three
            written paths.
        """
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        csv_path = out / CSV_NAME
        sidecar_dir = out / SIDECAR_DIR_NAME
        meta_path = out / META_NAME

        frame = self.recorder.save_dataframe(csv_path, sidecar_dir=sidecar_dir)
        meta = self._build_meta(frame, csv_path=csv_path, sidecar_dir=sidecar_dir)
        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        logger.info(
            "bench session written: {} rows, {} array columns, {} sidecar bytes",
            meta["row_count"],
            len(meta["array_bytes"]),
            meta["sidecar_bytes_on_disk"],
        )
        return {
            "csv": str(csv_path),
            "sidecars": str(sidecar_dir),
            "json": str(meta_path),
        }

    def _build_meta(
        self, frame, *, csv_path: Path, sidecar_dir: Path
    ) -> dict[str, Any]:
        """Summarise the written session.

        Args:
            frame: The :class:`pandas.DataFrame` that was written.
            csv_path: The CSV path (for its size on disk).
            sidecar_dir: The sidecar directory (for its size on disk).

        Returns:
            The metadata mapping, JSON-serialisable.
        """
        array_columns = _array_columns(frame)
        summary = _column_summary(frame)
        array_bytes = {
            col: int(summary[col]["bytes"]) for col in array_columns
        }
        sidecar_files = sorted(sidecar_dir.glob("*.npy")) if sidecar_dir.is_dir() else []
        expected_sidecars = int(len(frame)) * len(array_columns)
        if sidecar_files and len(sidecar_files) != expected_sidecars:
            # The array-column rule is restated in ``_array_columns``; if the
            # canonical writer ever changes it, this is where it shows up.
            logger.warning(
                "sidecar count {} does not match rows x array columns "
                "({} x {} = {}); the array-column rule may have drifted",
                len(sidecar_files),
                len(frame),
                len(array_columns),
                expected_sidecars,
            )
        rows = []
        for position, row in enumerate(self.recorder.history):
            rows.append(
                {
                    "row": position,
                    "id": int(row.get(RECORD_ID, position)),
                    "epoch": int(row.get(RECORD_EPOCH, position)),
                    COLUMN_INDEX: int(row.get(COLUMN_INDEX, position)),
                    COLUMN_KIND: str(row.get(COLUMN_KIND, "")),
                    COLUMN_CONDITION: str(row.get(COLUMN_CONDITION, "")),
                }
            )
        return {
            "schema": "ml.sim2f.recorder_io/1",
            "mark": self.mark,
            "mode": self.recorder.mode.value,
            "row_count": int(len(frame)),
            "columns": [str(col) for col in frame.columns],
            "column_summary": summary,
            "array_columns": array_columns,
            "array_bytes": array_bytes,
            "array_bytes_total": int(sum(array_bytes.values())),
            "csv": csv_path.name,
            "csv_bytes": int(csv_path.stat().st_size) if csv_path.is_file() else 0,
            "sidecar_dir": sidecar_dir.name,
            "sidecar_file_count": len(sidecar_files),
            "sidecar_bytes_on_disk": int(
                sum(path.stat().st_size for path in sidecar_files)
            ),
            "rows": rows,
        }


def _as_2d(value: Any, dtype: Any, name: str) -> np.ndarray:
    """Return ``value`` as a 2-D array of ``dtype``, copied.

    The copy is not paranoia: a camera driver may hand out a buffer it
    reuses for the next frame, so a recorded row must own its bytes.

    Args:
        value: The array-like to convert.
        dtype: Target numpy dtype.
        name: Field name, used in the error message.

    Returns:
        A 2-D array of ``dtype``.

    Raises:
        ValueError: If the result is not 2-D.
    """
    array = np.array(value, dtype=dtype)
    if array.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got shape {array.shape}")
    return array


def _array_columns(frame) -> list[str]:
    """Return the columns the sidecar writer will treat as arrays.

    The rule is the canonical one of
    :func:`ao_shaping.utils.io.file._save_array_sidecars`: a column is an
    array column when its first non-null entry is an ``np.ndarray``.  It is
    restated here (rather than imported) because that helper is private, and
    the restatement is checked against the files actually on disk by
    :meth:`BenchRecorder._build_meta` through ``sidecar_file_count``.

    Args:
        frame: The written :class:`pandas.DataFrame`.

    Returns:
        The array column names, in column order.
    """
    columns: list[str] = []
    for col in frame.columns:
        values = frame[col].dropna()
        if len(values) > 0 and isinstance(values.iloc[0], np.ndarray):
            columns.append(str(col))
    return columns


def _column_summary(frame) -> dict[str, dict[str, Any]]:
    """Summarise every column of the written frame.

    Args:
        frame: The written :class:`pandas.DataFrame`.

    Returns:
        One entry per column: its kind (``"array"`` / ``"scalar"``), its
        dtype, its shape (arrays only) and its byte volume.
    """
    summary: dict[str, dict[str, Any]] = {}
    for col in frame.columns:
        name = str(col)
        arrays = [
            value
            for value in frame[col].to_numpy()
            if isinstance(value, np.ndarray)
        ]
        if arrays:
            shapes = {tuple(int(dim) for dim in array.shape) for array in arrays}
            summary[name] = {
                "kind": "array",
                "dtype": str(arrays[0].dtype),
                "shape": [int(dim) for dim in arrays[0].shape],
                "shape_is_uniform": len(shapes) == 1,
                "count": len(arrays),
                "bytes": int(sum(int(array.nbytes) for array in arrays)),
            }
        else:
            summary[name] = {
                "kind": "scalar",
                "dtype": str(frame[col].dtype),
                "bytes": int(frame[col].memory_usage(deep=False)),
            }
    return summary
