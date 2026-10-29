"""Headless parallel model-fitting monitor for SLM transfer capture.

This module runs as a **separate process** from capture: it never opens the SLM
or any other device. Capture owns the SLM USB exclusively and writes
``sample_*.npz`` + ``manifest.json`` to a directory; this worker streams a
``loss_trace.jsonl`` and ``parallel_fit_provenance.json`` from that same
directory, and — once the loss is good enough or has plateaued — drops a
``stop_acquisition.request`` marker into the capture directory so capture can
stop probing early. With ``--finalize`` it delegates the canonical fit to
:func:`ao_shaping.tools.slm.transfer.fit.fit` once capture is complete.

The benefit of running in parallel is *earlier loss visibility*: after only a
few probes the worker can already fit a short model and report that the loss is
already low, so acquisition can stop before the full probe schedule finishes.

No device imports at module scope: this file is importable without torch being
installed. Torch is imported lazily, via the headless transfer fitting helpers
(:func:`build_fit_model`, :func:`fit_loss`), which never touch a device.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from ao_shaping.tools.slm.transfer.capture import clock


# --------------------------------------------------------------------------- #
# Early stopping
# --------------------------------------------------------------------------- #
@dataclass
class EarlyStopper:
    """Track training loss and decide when to stop acquisition.

    ``update(loss)`` is called once per poll with the poll's best loss. The
    stopper reports ``good_enough`` as soon as a loss at or below
    ``good_enough_loss`` is seen, and ``stop_fit`` once ``patience`` polls have
    passed without an improvement larger than ``min_delta``. Both flags are read
    by the worker to decide whether to stop acquisition early.
    """
    patience: int = 8
    min_delta: float = 1e-4
    good_enough_loss: float = 0.0  # <= 0 disables the good-enough rule
    best: float = float("inf")
    wait: int = 0
    good_enough: bool = False
    stop_fit: bool = False

    def update(self, loss: float) -> bool:
        if self.good_enough_loss > 0 and loss <= self.good_enough_loss:
            self.good_enough = True
        if loss < self.best - self.min_delta:
            self.best = loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stop_fit = True
        return self.good_enough or self.stop_fit

    def state(self) -> dict:
        return dict(best=float(self.best), wait=self.wait, patience=self.patience,
                    min_delta=self.min_delta, good_enough_loss=self.good_enough_loss,
                    good_enough=self.good_enough, stop_fit=self.stop_fit)


# --------------------------------------------------------------------------- #
# Adaptive learning rate
# --------------------------------------------------------------------------- #
@dataclass
class LRScheduler:
    """Warmup + plateau-reduce schedule applied across polls.

    The worker rebuilds its model from the warm checkpoint every poll, so the
    optimizer's base learning rates are re-created each time. This scheduler
    stores the per-group base LR (keyed by the ``name`` field
    :func:`build_fit_model` adds to every param group) and multiplies it by
    ``scale``. ``scale`` ramps linearly during ``warmup_polls``, then halves
    (down to ``min_scale``) whenever ``reduce_patience`` polls pass without a
    ``min_delta`` improvement.
    """
    warmup_polls: int = 3
    warmup_start: float = 0.1
    reduce_patience: int = 3
    reduce_factor: float = 0.5
    min_scale: float = 0.05
    min_delta: float = 1e-4
    scale: float = 1.0
    _warmup_done: int = 0
    _reduce_wait: int = 0
    _prev_best: float = float("inf")
    _base_lr: dict = field(default_factory=dict)

    def step(self, poll_best_loss: float) -> float:
        """Advance the schedule using this poll's best loss; return new ``scale``."""
        if self._warmup_done < self.warmup_polls:
            frac = (self._warmup_done + 1) / self.warmup_polls
            self.scale = self.warmup_start + (1.0 - self.warmup_start) * frac
            self._warmup_done += 1
            self._reduce_wait = 0
            self._prev_best = min(self._prev_best, poll_best_loss)
            return self.scale
        if poll_best_loss < self._prev_best - self.min_delta:
            self._prev_best = poll_best_loss
            self._reduce_wait = 0
        else:
            self._reduce_wait += 1
            if self._reduce_wait >= self.reduce_patience:
                self.scale = max(self.min_scale, self.scale * self.reduce_factor)
                self._reduce_wait = 0
        return self.scale

    def apply(self, optimizer) -> None:
        """Multiply each param group's LR by ``scale`` (capturing base LRs once)."""
        for group in optimizer.param_groups:
            name = group.get("name", "unknown")
            if name not in self._base_lr:
                self._base_lr[name] = group["lr"]
            group["lr"] = self._base_lr[name] * self.scale

    def state(self) -> dict:
        return dict(scale=float(self.scale), warmup_polls=self.warmup_polls,
                    warmup_start=self.warmup_start, warmup_done=self._warmup_done,
                    reduce_patience=self.reduce_patience, reduce_factor=self.reduce_factor,
                    min_scale=self.min_scale, reduce_wait=self._reduce_wait,
                    prev_best=float(self._prev_best),
                    base_lr={k: float(v) for k, v in self._base_lr.items()})


# --------------------------------------------------------------------------- #
# Worker
# --------------------------------------------------------------------------- #
class ParallelFitWorker:
    """Headless monitor that fits alongside capture and signals early stop.

    The worker is deliberately a *monitor*, not the source of the canonical
    model. Each poll it rebuilds the model fresh from the warm ``checkpoint``
    (avoiding persistent-optimizer subtleties across polls), runs a short batch
    of ``poll_iterations`` steps, records the best loss seen, and lets
    :class:`EarlyStopper` and :class:`LRScheduler` persist their state across
    polls. When the stopper fires, the worker writes ``stop_acquisition.request``
    into the capture directory so capture can stop probing; if ``finalize`` is
    set it additionally runs the full canonical fit via ``fit.fit``.
    """

    def __init__(self, capture_dir, checkpoint, *, span, out, device=None,
                 adaptation="full", radiometry="exposure", amplitude_center_lr=None,
                 poll_iterations=50, early_stopper=None, lr_scheduler=None,
                 signal_stop=True):
        self.capture_dir = Path(capture_dir)
        self.checkpoint = Path(checkpoint)
        self.span = span
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.adaptation = adaptation
        self.radiometry = radiometry
        self.amplitude_center_lr = amplitude_center_lr
        self.poll_iterations = poll_iterations
        self.early_stopper = early_stopper or EarlyStopper()
        self.lr_scheduler = lr_scheduler or LRScheduler()
        self.signal_stop = signal_stop
        self.poll_index = 0
        self.history = []
        self.global_best_loss = float("inf")
        self.provenance = self._init_provenance()

    def _init_provenance(self) -> dict:
        return dict(module="parallel_fit",
                    capture_dir=str(self.capture_dir.resolve()),
                    checkpoint=str(self.checkpoint.resolve()),
                    span=self.span, device=self.device, adaptation=self.adaptation,
                    radiometry=self.radiometry,
                    amplitude_center_lr=self.amplitude_center_lr,
                    poll_iterations=self.poll_iterations,
                    early_stop=self.early_stopper.state(),
                    lr_schedule=self.lr_scheduler.state(),
                    started_utc=clock()["timestamp_utc"],
                    start_monotonic_ns=clock()["monotonic_ns"])

    # --- capture-directory signals ---------------------------------------- #
    def _signal_stop(self) -> None:
        if not self.signal_stop:
            return
        (self.capture_dir / "stop_acquisition.request").write_text(
            json.dumps(dict(reason="parallel_fit_early_stop", poll_index=self.poll_index,
                            **clock()), indent=2), encoding="utf-8")

    def _user_requested_stop(self) -> bool:
        return (self.capture_dir / "stop_acquisition.request").is_file()

    def _capture_done(self) -> bool:
        return (self.capture_dir / ".capture_done").is_file()

    def _row_count(self) -> int:
        return len(list(self.capture_dir.glob("sample_*.npz")))

    # --- one monitoring poll ---------------------------------------------- #
    def poll_once(self) -> dict:
        """Fit one short batch from the current capture; return a loss-trace row.

        Returns a dict with a ``status`` of ``"waiting"`` (dataset not ready),
        ``"failed"`` (nonfinite loss), or ``"poll"`` / ``"early_stop"``.
        """
        import numpy as np
        import torch
        from ao_shaping.tools.slm.transfer.fit import build_fit_model, fit_loss

        started = clock()
        try:
            comps = build_fit_model(self.capture_dir, self.checkpoint, span=self.span,
                                    device=self.device, adaptation=self.adaptation,
                                    amplitude_center_lr=self.amplitude_center_lr,
                                    radiometry=self.radiometry)
        except (ValueError, FileNotFoundError, KeyError, StopIteration) as exc:
            return dict(poll=self.poll_index, status="waiting", error=str(exc), **started)

        self.lr_scheduler.apply(comps.model.optimizer)
        scale = self.lr_scheduler.scale
        best_loss, best_entry, best_pred = float("inf"), None, None
        for it in range(self.poll_iterations):
            comps.model.optimizer.zero_grad()
            try:
                loss, entry, predicted = fit_loss(comps.model, comps, self.radiometry, it)
            except RuntimeError as exc:
                return dict(poll=self.poll_index, status="failed", error=str(exc), **started)
            if entry["loss"] < best_loss:
                best_loss, best_entry, best_pred = entry["loss"], entry, predicted
            if it < self.poll_iterations - 1:
                loss.backward()
                comps.model.optimizer.step()
                with torch.no_grad():
                    comps.model.d.clamp_(-2, 2)
                    comps.model.center.clamp_(0, 1920)
                    comps.model.log_width.clamp_(float(np.log(80)), float(np.log(800)))

        if best_loss < self.global_best_loss:
            self.global_best_loss = best_loss
            np.savez_compressed(self.out / "best_prediction.npz", prediction=best_pred)

        self.lr_scheduler.step(best_loss)
        triggered = self.early_stopper.update(best_loss)
        row = dict(poll=self.poll_index, status="early_stop" if triggered else "poll",
                   samples=self._row_count(), best_loss=float(best_loss),
                   best_iteration=best_entry["iteration"], lr_scale=float(scale),
                   model_parameters=best_entry["parameters"],
                   metrics=best_entry["metrics"],
                   early_stop=self.early_stopper.state(),
                   lr_schedule=self.lr_scheduler.state(), **started,
                   end_monotonic_ns=time.monotonic_ns())
        self.history.append(row)
        with (self.out / "loss_trace.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        if triggered:
            self._signal_stop()
        return row

    # --- finalize (canonical fit via fit.fit) ----------------------------- #
    def _finalize(self, out, iterations) -> None:
        from types import SimpleNamespace

        from ao_shaping.tools.slm.transfer.fit import fit

        out = Path(out)
        fit_args = SimpleNamespace(
            dataset=self.capture_dir, checkpoint=self.checkpoint, out=out, span=self.span,
            iterations=iterations, adaptation=self.adaptation, device=self.device,
            amplitude_center_lr=self.amplitude_center_lr, radiometry=self.radiometry,
            samples=None, record_responses=False)
        fit(fit_args)
        self.provenance["finalize"] = dict(out=str(out.resolve()), iterations=iterations,
                                           **clock())

    # --- run loop --------------------------------------------------------- #
    def run(self, *, poll_seconds=5.0, max_polls=200, finalize=False,
            finalize_out=None, finalize_iterations=2000, once=False):
        """Drive polls until early stop, capture done, user stop, or ``max_polls``."""
        exit_reason = None
        while True:
            if self._user_requested_stop():
                exit_reason = "user_requested_stop"
                break
            row = self.poll_once()
            status = row.get("status")
            if status == "waiting":
                if once or self._capture_done():
                    exit_reason = "capture_done_insufficient_data" if self._capture_done() else "once_waiting"
                    break
                time.sleep(poll_seconds)
                continue
            if status == "failed":
                if once:
                    exit_reason = "once_failed"
                    break
                time.sleep(poll_seconds)
                continue
            self.poll_index += 1
            if status == "early_stop":
                exit_reason = "early_stop"
                if finalize:
                    self._finalize(finalize_out or (self.out / "canonical"), finalize_iterations)
                break
            if once:
                exit_reason = "once"
                break
            if self._capture_done():
                exit_reason = "capture_done"
                if finalize:
                    self._finalize(finalize_out or (self.out / "canonical"), finalize_iterations)
                break
            time.sleep(poll_seconds)
            if self.poll_index >= max_polls:
                exit_reason = "max_polls"
                if finalize:
                    self._finalize(finalize_out or (self.out / "canonical"), finalize_iterations)
                break

        self.provenance["exit"] = dict(reason=exit_reason, poll_index=self.poll_index, **clock())
        self._write_provenance()
        return self.provenance["exit"]

    def _write_provenance(self) -> None:
        self.provenance["early_stop"] = self.early_stopper.state()
        self.provenance["lr_schedule"] = self.lr_scheduler.state()
        self.provenance["polls"] = self.poll_index
        self.provenance["loss_trace_rows"] = len(self.history)
        self.provenance["global_best_loss"] = float(self.global_best_loss)
        (self.out / "parallel_fit_provenance.json").write_text(
            json.dumps(self.provenance, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Headless parallel fit worker for SLM transfer capture.")
    parser.add_argument("--capture-dir", required=True, type=Path,
                        help="directory capture is writing sample_*.npz + manifest.json into")
    parser.add_argument("--checkpoint", required=True, type=Path,
                        help="warm model to start from (frozen_parameters.json or nominal.json)")
    parser.add_argument("--span", type=int, default=128,
                        help="window span in pixels (matches capture's window_span_px)")
    parser.add_argument("--out", type=Path, default=None,
                        help="worker output dir (default: <capture-dir>/parallel_fit)")
    parser.add_argument("--device", default=None, help="torch device (auto/cpu/cuda)")
    parser.add_argument("--adaptation", default="full", choices=["full", "alignment"])
    parser.add_argument("--radiometry", default="exposure", choices=["exposure", "bias", "absolute"])
    parser.add_argument("--amplitude-center-lr", type=float, default=None)
    parser.add_argument("--poll-iterations", type=int, default=50)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--max-polls", type=int, default=200)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--good-enough-loss", type=float, default=0.0)
    parser.add_argument("--warmup-polls", type=int, default=3)
    parser.add_argument("--warmup-start", type=float, default=0.1)
    parser.add_argument("--lr-reduce-patience", type=int, default=3)
    parser.add_argument("--lr-reduce-factor", type=float, default=0.5)
    parser.add_argument("--lr-min-scale", type=float, default=0.05)
    parser.add_argument("--finalize", action="store_true",
                        help="run the canonical fit.fit once capture is complete / stopped")
    parser.add_argument("--finalize-iterations", type=int, default=2000)
    parser.add_argument("--finalize-out", type=Path, default=None)
    parser.add_argument("--no-signal-stop", action="store_true",
                        help="do not write stop_acquisition.request into the capture dir")
    parser.add_argument("--once", action="store_true", help="run a single poll and exit (for tests)")
    args = parser.parse_args(argv)

    out = args.out or (args.capture_dir / "parallel_fit")
    worker = ParallelFitWorker(
        args.capture_dir, args.checkpoint, span=args.span, out=out, device=args.device,
        adaptation=args.adaptation, radiometry=args.radiometry,
        amplitude_center_lr=args.amplitude_center_lr, poll_iterations=args.poll_iterations,
        signal_stop=not args.no_signal_stop,
        early_stopper=EarlyStopper(patience=args.patience, min_delta=args.min_delta,
                                  good_enough_loss=args.good_enough_loss),
        lr_scheduler=LRScheduler(warmup_polls=args.warmup_polls, warmup_start=args.warmup_start,
                                reduce_patience=args.lr_reduce_patience,
                                reduce_factor=args.lr_reduce_factor,
                                min_scale=args.lr_min_scale))
    exit_info = worker.run(poll_seconds=args.poll_seconds, max_polls=args.max_polls,
                           finalize=args.finalize, finalize_out=args.finalize_out,
                           finalize_iterations=args.finalize_iterations, once=args.once)
    print(json.dumps(exit_info))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
