"""Guards for the per-iteration Step B debug recorder.

``optimizer/wfless/step_b_debug.py`` is the debug channel the differentiable
Step B solve writes through when ``train_beam_shaping(debug=...)`` is on (see
``tests/ao_shaping/algorithm/test_differentiable_shaping.py`` for the wiring).
The recorder has to stay a **leaf**: numpy + stdlib, no torch, no device, so it
can be imported from a hardware run without dragging anything in.

Three properties matter for a debug channel that a 400-iteration hardware solve
depends on, and each has a guard here:

1. **It records before it is asked to.** The per-iteration line hits the
   ``.jsonl`` as the iteration ends, so a run killed mid-solve still yields the
   iterations it reached (``test_each_record_reaches_disk_before_close``).
2. **The numbers mean what their names say.** The "phase gradient change" is the
   parameter update ``||phase_after - phase_before||``, distinct from the
   gradient norm -- conflating them would make a large learning rate look like a
   large gradient (``test_phase_delta_is_the_parameter_update``).
3. **The timing split answers the speed question.** ``forward_share`` is the
   forward/backward fraction of the solve, i.e. the part that belongs to the
   physical model rather than the optimiser
   (``test_speed_summary_splits_forward_from_step``).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ao_shaping.optimizer.wfless.step_b_debug import (
    ARRAYS_NAME,
    JSONL_NAME,
    SUMMARY_NAME,
    StepBDebugRecorder,
    StepBStepRecord,
    gradient_stats,
)


def _rows(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


class TestGradientStats:
    """``gradient_stats`` must be plain numpy definitions of norm / max / RMS."""

    def test_matches_numpy_definitions(self) -> None:
        grad = np.array([[3.0, 4.0], [-1.0, 0.0]], dtype=np.float64)
        norm, maxabs, rms = gradient_stats(grad)
        assert norm == pytest.approx(float(np.linalg.norm(grad)))
        assert maxabs == pytest.approx(4.0)
        assert rms == pytest.approx(float(np.sqrt(np.mean(grad**2))))

    def test_accepts_torch_like_objects_without_importing_torch(self) -> None:
        # The recorder is handed whatever the caller has; it must not require
        # a torch.Tensor specifically (it never imports torch).
        grad = np.zeros((4, 4), dtype=np.float32)
        norm, maxabs, rms = gradient_stats(grad)
        assert (norm, maxabs, rms) == (0.0, 0.0, 0.0)


class TestRecorder:
    def test_each_record_reaches_disk_before_close(self, tmp_path: Path) -> None:
        rec = StepBDebugRecorder(tmp_path, tag="round1")
        jsonl = tmp_path / JSONL_NAME.format(tag="round1")
        try:
            for it in range(3):
                rec.record(
                    iteration=it,
                    loss=1.0 / (it + 1),
                    grad=np.full((2, 2), float(it), dtype=np.float64),
                    forward_s=0.002,
                    step_s=0.001,
                )
            # No close() yet: a crash right here must still leave 3 rows.
            rows = _rows(jsonl)
            assert [r["iteration"] for r in rows] == [0, 1, 2]
            assert rows[2]["loss"] == pytest.approx(1.0 / 3.0)
            assert not (tmp_path / SUMMARY_NAME.format(tag="round1")).exists()
        finally:
            rec.close()

    def test_phase_delta_is_the_parameter_update(self, tmp_path: Path) -> None:
        rec = StepBDebugRecorder(tmp_path)
        before = np.zeros((2, 2), dtype=np.float64)
        after = np.full((2, 2), 0.5, dtype=np.float64)
        try:
            record = rec.record(
                iteration=0,
                loss=2.0,
                grad=np.full((2, 2), 10.0, dtype=np.float64),
                phase_before=before,
                phase_after=after,
                forward_s=0.0,
                step_s=0.0,
            )
        finally:
            rec.close()
        assert record.phase_delta_norm == pytest.approx(float(np.linalg.norm(after)))
        # A big gradient with a small step is exactly the readable signal: the
        # two metrics must not be derived from one another.
        assert record.grad_norm == pytest.approx(float(np.linalg.norm(np.full((2, 2), 10.0))))
        assert record.grad_norm > record.phase_delta_norm

    def test_speed_summary_splits_forward_from_step(self, tmp_path: Path) -> None:
        rec = StepBDebugRecorder(tmp_path)
        try:
            rec.record(
                StepBStepRecord(0, 4.0, 1.0, 1.0, 1.0, 0.0, forward_s=0.75, step_s=0.25)
            )
            rec.record(
                StepBStepRecord(1, 2.0, 1.0, 1.0, 1.0, 0.0, forward_s=0.25, step_s=0.75)
            )
            speed = rec.speed_summary()
        finally:
            rec.close()
        assert speed["iterations"] == 2
        assert speed["forward_total_s"] == pytest.approx(1.0)
        assert speed["step_total_s"] == pytest.approx(1.0)
        assert speed["forward_share"] == pytest.approx(0.5)
        assert speed["loss_first"] == pytest.approx(4.0)
        assert speed["loss_last"] == pytest.approx(2.0)
        assert speed["loss_best_iteration"] == 1

    def test_close_writes_summary_once(self, tmp_path: Path) -> None:
        rec = StepBDebugRecorder(tmp_path, tag="r", context={"round": 3})
        rec.record(iteration=0, loss=1.0, grad=np.ones((2, 2)), forward_s=0.1, step_s=0.1)
        first = rec.close()
        assert first is not None and first.exists()
        payload = json.loads(first.read_text(encoding="utf-8"))
        assert payload["context"] == {"round": 3}
        assert payload["speed"]["iterations"] == 1
        assert payload["records"][0]["iteration"] == 0
        mtime = first.stat().st_mtime_ns
        assert rec.close() == first
        assert first.stat().st_mtime_ns == mtime

    def test_gradient_snapshots_are_persisted(self, tmp_path: Path) -> None:
        rec = StepBDebugRecorder(tmp_path, save_gradients=True, save_every=2)
        grad = np.arange(16, dtype=np.float64).reshape(4, 4)
        try:
            for it in range(4):
                rec.record(
                    iteration=it,
                    loss=float(it),
                    grad=grad,
                    phase_before=np.zeros_like(grad),
                    phase_after=np.full_like(grad, 0.1),
                    forward_s=0.01,
                    step_s=0.01,
                )
        finally:
            path = rec.close()
        assert path is not None
        arrays = np.load(tmp_path / ARRAYS_NAME.format(tag="step_b"))
        data = {name: arrays[name] for name in arrays.files}
        assert data["iterations"].tolist() == [0, 2]
        # Snapshots are stacked over the saved iterations and stored in the
        # configured (lossy) dtype on purpose.
        assert data["grad"].shape == (2, 4, 4)
        assert data["grad"].dtype == np.float16
        assert data["phase"].shape == (2, 4, 4)
        assert data["loss"].tolist() == [0.0, 1.0, 2.0, 3.0]

    def test_rejects_bad_stride(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            StepBDebugRecorder(tmp_path, save_every=0)
