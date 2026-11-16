"""Per-iteration debug recorder for the differentiable Step B phase solve.

Why this module exists
----------------------
Step B runs hundreds of Adam steps on a full-pixel SLM phase. When a round is
slow or stalls, the loss curve in ``step_b_trace`` cannot answer the two questions
that matter:

* **Is the forward model the bottleneck?** The wall clock splits into the
  ``forward+backward`` (the physical propagation, the CUDA kernels, the convolutions)
  and the ``optimizer step`` (usually elementwise and cheap). Recording the split
  per iteration shows which one to attack before touching the algorithm.
* **What is the gradient doing?** A loss that flattens while ``|grad|`` stays large
  means the step size is wrong; a ``|grad|`` collapse means a stationary point; a
  ``|grad|`` that oscillates means the learning rate is above the stability limit.
  None of that is visible from the loss alone.

Each iteration therefore records the forward-model loss, the gradient norm / max /
RMS, the parameter update norm (``||phase_after - phase_before||`` -- the literal
"phase gradient change" of the solve) and the two timings. With
``save_gradients=True`` the gradient and phase snapshots are also persisted, so a
stalled run can be replayed offline without re-opening the devices.

Design constraints
------------------
* **Off by default.** ``None`` instead of a recorder keeps the solve bit-for-bit:
  the recorder never touches the optimiser state, the RNG or the phase tensor it is
  handed (it copies), so enabling it can change the wall clock but not the result.
* **Crash-safe.** A line is appended to ``<tag>_debug.jsonl`` every iteration, so a
  run that dies mid-solve still yields the iterations it reached. ``close()`` writes
  the aggregated ``<tag>_debug.json``.
* **Leaf module.** numpy + stdlib only: no torch, no ``ao_shaping`` import, no device
  access. The caller passes plain floats and (optionally) numpy arrays.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

#: Per-iteration line file (append-only, crash-safe).
JSONL_NAME = "{tag}_debug.jsonl"
#: Aggregated summary written by :meth:`StepBDebugRecorder.close`.
SUMMARY_NAME = "{tag}_debug.json"
#: Optional gradient / phase snapshot archive.
ARRAYS_NAME = "{tag}_gradients.npz"

#: Default stride for array snapshots: one frame in ten keeps a 512x512 region to
#: roughly 30 MB per 600 iterations at float16.
DEFAULT_SAVE_EVERY = 10


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


@dataclass(frozen=True)
class StepBStepRecord:
    """Scalars for one differentiable-optimisation iteration.

    Attributes:
        iteration: Zero-based iteration index.
        loss: The forward-model loss evaluated at ``phase_before`` (the same value
            the optimiser is about to descend).
        grad_norm: L2 norm of the phase gradient.
        grad_max: Largest absolute gradient component.
        grad_rms: Root-mean-square gradient component.
        phase_delta_norm: L2 norm of the parameter update applied this iteration
            (``||phase_after - phase_before||``); zero when the caller cannot
            supply the post-step phase.
        forward_s: Wall time of ``forward + backward`` (seconds).
        step_s: Wall time of the optimiser step (seconds). Near zero for L-BFGS,
            whose line search calls the forward pass internally, so its cost lands
            in ``forward_s`` instead.
        timestamp_utc: ISO-8601 UTC timestamp of the record.
    """

    iteration: int
    loss: float
    grad_norm: float
    grad_max: float
    grad_rms: float
    phase_delta_norm: float
    forward_s: float
    step_s: float
    timestamp_utc: str = ""


def gradient_stats(grad: Any) -> tuple[float, float, float]:
    """Return ``(norm, max_abs, rms)`` for a gradient array or torch tensor.

    Accepts anything convertible to a numpy array; a tensor is copied to the host
    exactly once per iteration, which is a debug-only cost.
    """
    array = np.asarray(grad, dtype=np.float64)
    if array.size == 0:
        return 0.0, 0.0, 0.0
    return (
        float(np.linalg.norm(array)),
        float(np.max(np.abs(array))),
        float(np.sqrt(np.mean(array**2))),
    )


def _reduce(values: list[float]) -> dict[str, float]:
    """Mean / median / min / max of a non-empty list; zeros for an empty one."""
    if not values:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


class StepBDebugRecorder:
    """Collect and persist per-iteration Step B diagnostics.

    Args:
        directory: Output directory (created when missing).
        tag: File-name prefix, so several rounds in one run can be told apart.
        save_gradients: Also store gradient (and phase) snapshots in an ``.npz``.
        save_every: Stride for array snapshots; ``1`` stores every iteration.
        array_dtype: On-disk dtype for snapshots (``float16`` halves the size).
        context: Free-form metadata merged into the summary (round index, config).

    Example:
        >>> rec = StepBDebugRecorder("data/debug/run/step_b", tag="round1")
        >>> rec.record(StepBStepRecord(0, 1.0, 0.5, 0.1, 0.01, 0.0, 0.02, 0.001))
        >>> rec.close()  # doctest: +SKIP
    """

    def __init__(
        self,
        directory: str | Path,
        *,
        tag: str = "step_b",
        save_gradients: bool = False,
        save_every: int = DEFAULT_SAVE_EVERY,
        array_dtype: str = "float16",
        context: dict[str, Any] | None = None,
    ) -> None:
        if save_every < 1:
            raise ValueError(f"save_every must be >= 1, got {save_every}")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.tag = tag
        self.save_gradients = bool(save_gradients)
        self.save_every = int(save_every)
        self.array_dtype = array_dtype
        self.context = dict(context or {})

        self._jsonl_path = self.directory / JSONL_NAME.format(tag=tag)
        self._summary_path = self.directory / SUMMARY_NAME.format(tag=tag)
        self._arrays_path = self.directory / ARRAYS_NAME.format(tag=tag)
        self._records: list[StepBStepRecord] = []
        self._grads: list[np.ndarray] = []
        self._phases: list[np.ndarray] = []
        self._array_iterations: list[int] = []
        self._closed = False
        # Truncate any stale file from a previous run with the same tag.
        self._jsonl_path.write_text("", encoding="utf-8")

    # -- recording ---------------------------------------------------------
    def record(
        self,
        record: StepBStepRecord | None = None,
        *,
        iteration: int | None = None,
        loss: float | None = None,
        grad: Any = None,
        phase_before: Any = None,
        phase_after: Any = None,
        forward_s: float = 0.0,
        step_s: float = 0.0,
    ) -> StepBStepRecord:
        """Append one iteration.

        Either pass a prebuilt :class:`StepBStepRecord`, or the individual fields
        (with ``grad`` as an array/tensor whose statistics are computed here).
        When ``grad`` is given but a prebuilt record is not, the gradient stats are
        derived from it; when both are given the record wins and ``grad`` is used
        only for the snapshot.
        """
        if record is None:
            if iteration is None or loss is None:
                raise ValueError("pass a StepBStepRecord or both iteration and loss")
            norm = maxabs = rms = 0.0
            if grad is not None:
                norm, maxabs, rms = gradient_stats(grad)
            delta = 0.0
            if phase_before is not None and phase_after is not None:
                delta = float(
                    np.linalg.norm(
                        np.asarray(phase_after, dtype=np.float64)
                        - np.asarray(phase_before, dtype=np.float64)
                    )
                )
            record = StepBStepRecord(
                iteration=int(iteration),
                loss=float(loss),
                grad_norm=norm,
                grad_max=maxabs,
                grad_rms=rms,
                phase_delta_norm=delta,
                forward_s=float(forward_s),
                step_s=float(step_s),
                timestamp_utc=_utcnow(),
            )
        elif not record.timestamp_utc:
            record = StepBStepRecord(**{**asdict(record), "timestamp_utc": _utcnow()})

        self._records.append(record)
        with self._jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(record)) + "\n")

        if self.save_gradients and record.iteration % self.save_every == 0:
            if grad is not None:
                self._grads.append(
                    np.asarray(grad, dtype=self.array_dtype).copy()
                )
            if phase_after is not None:
                self._phases.append(
                    np.asarray(phase_after, dtype=self.array_dtype).copy()
                )
            self._array_iterations.append(record.iteration)
        return record

    # -- reporting ---------------------------------------------------------
    def speed_summary(self) -> dict[str, Any]:
        """Split the measured wall clock into forward vs optimiser step.

        ``forward_share`` is the fraction of the solve spent in the physical
        forward/backward pass; when it dominates, speed work belongs in the
        propagation, not the optimiser.
        """
        forward = [r.forward_s for r in self._records]
        step = [r.step_s for r in self._records]
        total_forward = float(sum(forward))
        total_step = float(sum(step))
        total = total_forward + total_step
        losses = [r.loss for r in self._records]
        grads = [r.grad_norm for r in self._records]
        return {
            "iterations": len(self._records),
            "total_s": total,
            "forward_total_s": total_forward,
            "step_total_s": total_step,
            "forward_share": (total_forward / total) if total > 0 else 0.0,
            "forward_s": _reduce(forward),
            "step_s": _reduce(step),
            "loss_first": losses[0] if losses else None,
            "loss_last": losses[-1] if losses else None,
            "loss_best": min(losses) if losses else None,
            "loss_best_iteration": (
                int(np.argmin(losses)) if losses else None
            ),
            "grad_norm_first": grads[0] if grads else None,
            "grad_norm_last": grads[-1] if grads else None,
        }

    def close(self) -> Path | None:
        """Write the summary (and array archive) and return the summary path.

        Idempotent: a second call returns the same path without rewriting.
        """
        if self._closed:
            return self._summary_path
        self._closed = True
        payload: dict[str, Any] = {
            "tag": self.tag,
            "directory": str(self.directory),
            "save_every": self.save_every,
            "save_gradients": self.save_gradients,
            "array_count": len(self._array_iterations),
            "speed": self.speed_summary(),
            "context": self.context,
            "records": [asdict(r) for r in self._records],
        }
        self._summary_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if self.save_gradients and self._array_iterations:
            arrays: dict[str, np.ndarray] = {
                "iterations": np.asarray(self._array_iterations, dtype=np.int64),
                "loss": np.asarray([r.loss for r in self._records], dtype=np.float64),
                "grad_norm": np.asarray(
                    [r.grad_norm for r in self._records], dtype=np.float64
                ),
                "phase_delta_norm": np.asarray(
                    [r.phase_delta_norm for r in self._records], dtype=np.float64
                ),
            }
            if self._grads:
                arrays["grad"] = np.stack(self._grads)
            if self._phases:
                arrays["phase"] = np.stack(self._phases)
            np.savez_compressed(self._arrays_path, **arrays)
        return self._summary_path

    def __enter__(self) -> StepBDebugRecorder:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
