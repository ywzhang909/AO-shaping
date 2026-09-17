from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

import ao_shaping.tools.slm.slm_phase_response as phase_response
from ao_shaping.tools.slm.slm_phase_response import (
    PhaseCase,
    _delta,
    _metrics,
    _resolve_payload,
    build_sequence,
    run_phase_probe,
)


class FakeCamera:
    def __init__(self, frames: list[np.ndarray]):
        self.frames = frames
        self.index = 0

    def get_numpy_image(self, **kwargs) -> np.ndarray:
        frame = self.frames[self.index]
        self.index += 1
        return frame


class FakeSLM:
    def __init__(self):
        self.Panel_Res = (4, 4)
        self.displayed_slot = 2
        self.writes: list[tuple[np.ndarray, int]] = []

    def get_displayed_memory_number(self) -> int:
        return self.displayed_slot

    def display_data(self, pattern: np.ndarray, memory_number: int) -> None:
        self.displayed_slot = memory_number
        self.writes.append((pattern, memory_number))


def test_metrics_reports_energy_centroid_and_global_maximum() -> None:
    image = np.zeros((3, 3), dtype=np.float64)
    image[1, 1] = 4.0

    metrics = _metrics(image)

    assert metrics["total"] == 4.0
    assert metrics["peak"] == 4.0
    assert metrics["centroid"] == (1.0, 1.0)
    assert metrics["argmax_rc"] == (1, 1)
    assert metrics["ee90_r"] == 0.0


def test_delta_is_root_mean_square_difference() -> None:
    a = np.array([[0.0, 1.0], [2.0, 3.0]])
    b = np.array([[1.0, 1.0], [0.0, 3.0]])

    assert _delta(a, b) == pytest.approx(np.sqrt(5.0 / 4.0))


def test_build_sequence_wraps_cases_with_flat_controls() -> None:
    cases = [
        PhaseCase("first", "first pattern", lambda slm: np.zeros((2, 2))),
        PhaseCase("second", "second pattern", lambda slm: np.ones((2, 2))),
    ]

    sequence = build_sequence(cases)

    assert [item[0] for item in sequence] == [
        "A_flat",
        "B_first",
        "C_second",
        "D_flat_again",
    ]
    assert sequence[0][2] is None
    assert sequence[-1][2] is None


def test_resolve_payload_uses_raw_flat_gray_and_callable() -> None:
    slm = cast(Any, object())
    flat = _resolve_payload(None, slm, 3, 2)
    pattern = _resolve_payload(lambda _: np.full((2, 3), 7, dtype=np.uint16), slm, 3, 2)

    assert flat.dtype == np.uint16
    assert flat.shape == (2, 3)
    assert np.all(flat == 512)
    assert np.all(pattern == 7)


def test_run_phase_probe_rotates_slots_and_writes_artifacts(
    tmp_path,
    monkeypatch,
) -> None:
    frames = [np.zeros((4, 4), dtype=np.float64) for _ in range(14)]
    camera = FakeCamera(frames)
    slm = FakeSLM()
    case = PhaseCase(
        "pattern", "test pattern", lambda _: np.ones((4, 4), dtype=np.uint16)
    )
    monkeypatch.setattr(
        phase_response.random, "choice", lambda candidates: candidates[0]
    )
    monkeypatch.setattr(phase_response.time, "sleep", lambda _: None)
    monkeypatch.setattr(phase_response, "_render", lambda frames, metrics, out: None)

    result = run_phase_probe(
        camera=camera,
        slm=cast(Any, slm),
        cases=[case],
        out=tmp_path,
        n_sample=1,
        settle_s=0.0,
        slot_min=2,
        slot_max=4,
        exposure_ms=0.1,
    )

    assert [slot for _, slot in slm.writes] == [3, 2, 3]
    assert list(result["metrics"]) == ["A_flat", "B_pattern", "D_flat_again"]
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "A_flat_pattern.npy").exists()
    assert (tmp_path / "B_pattern_pattern.npy").exists()
