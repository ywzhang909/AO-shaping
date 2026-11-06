"""Bit-for-bit characterization of the square SPGD epoch loop (#205 batch 28).

Batches 25-28 keep moving blocks out of ``_run_square_search_stage``. A
*verbatim* block move is provable by AST, but the batch that collapsed the two
``+/- disturb_c`` measurements into one shared record is not -- so its
equivalence is pinned numerically here, the way batch 1 pinned the sweep split
with ``REF_ALL_NORMAL``.

The camera is deliberately **phase dependent**: the blob's offset and
anisotropy follow the last phase written to the panel. The fixed blob in
``test_slm_square_objective_search_sim.py`` yields ``pos["q"] == neg["q"]`` in
every epoch, so the SPGD gradient is identically zero there and a pos/neg
mix-up would be invisible. With a phase-driven frame the gradient is non-zero
and every downstream consumer -- best-tracking, the in-ROI energy guard, the
``lr == 0`` schedule, the recorder rows -- actually reads the pair.

Golden tuple is ``(n_rows, rows_hash, writes_hash, n_slm_writes)``: the row
hash covers every column of every epoch, the write hash covers the arrays
pushed to the SLM.

Both hashes are **quantized, not bit-exact, on purpose**.  A bit-exact golden is
a flake generator: the SPGD chain is float32/float64 arithmetic whose reduction
order depends on how BLAS/FFT split work across the cores a runner happens to
have, so the last bits legitimately differ between machines, and a golden that
hashes them turns that into a red shard that disappears on re-run -- the exact
"intermittent, name lost" class this repository already has one open case of
(TODO.md 8.9, tools shard 1 failed / 1107 passed).  Quantizing to 4 decimals /
5 significant digits sits ~4 orders of magnitude above that drift and ~2 below
the smallest behavior change worth catching: mutating ``-disturb_c`` to
``+disturb_c`` (a real pos/neg swap) still turns all six cases red, which is the
property this net is for.

Re-baseline after an *intentional* behavior change (repo's opt-in convention):

    AO_SQUARE_CHAR_UPDATE=/tmp/square_golden.json \
        ./scripts/build/run_pytest_ci.sh <this file> -q

then paste the JSON into ``EXPECTED``.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ao_shaping.optimizer.wfless import slm_square_shaping as sq

#: Quantization applied before hashing.  See the module docstring: this is a
#: platform-drift margin, not a tolerance for behavior.
_QUANT_DECIMALS = 4
_FLOAT_SIG_DIGITS = ".5g"

_H, _W = 64, 64

CASES: dict[str, dict[str, Any]] = {
    "adaptive": dict(objective="quality", epochs=6, adaptive=True, stagnate_win=2,
                     stagnate_boost=1.5, lr=0.0),
    "pearson_lr0": dict(objective="pearson", epochs=3, lr=0.0, delta=0.1),
    "quality_default": dict(objective="quality", epochs=3, delta=0.1),
    "tight_guard": dict(objective="quality", epochs=3, max_roi_energy_loss=0.001),
    "w_pbr_aspect": dict(objective="quality", epochs=3, w_pbr=0.2, w_aspect=0.3),
    "zernike_basis": dict(objective="quality", epochs=2, basis="zernike", n_max=4, phase_grid=8),
}

EXPECTED: dict[str, tuple[int, str, str, int]] = {
    "adaptive": (
        7,
        "295b27254f0a576b5453bbb60c5a64e3ca269cbb",
        "250dab79dcf19359693381f681f1b1669eed74f6",
        38,
    ),
    "pearson_lr0": (
        4,
        "73e3b07574416a296e3aa62511fa3df552e55a01",
        "7363dbbb25e0b3d336928134e626c3e23ab623dd",
        20,
    ),
    "quality_default": (
        4,
        "e82677ca57eb81a7d9e1a366e7d2e442db61f5a6",
        "7363dbbb25e0b3d336928134e626c3e23ab623dd",
        20,
    ),
    "tight_guard": (
        4,
        "e82677ca57eb81a7d9e1a366e7d2e442db61f5a6",
        "7363dbbb25e0b3d336928134e626c3e23ab623dd",
        20,
    ),
    "w_pbr_aspect": (
        4,
        "c6161cd6667d7337809f5886c758879ed91d229e",
        "7363dbbb25e0b3d336928134e626c3e23ab623dd",
        20,
    ),
    "zernike_basis": (
        3,
        "cbcd94790dfc545f01caa39a4fb0b1a35c6d9ddb",
        "2c0cf84446c95777490b1ef80d476e7d98678829",
        14,
    ),
}


def _quant(v: Any) -> np.ndarray:
    """Round to the golden's quantization, in a fixed dtype and byte order.

    ``<f8`` is not decoration: hashing native-endian bytes would make the golden
    machine-class dependent for no benefit.
    """
    return np.round(np.asarray(v, dtype=np.float64), _QUANT_DECIMALS).astype("<f8")


def _cell(v: Any) -> Any:
    """Canonical, order-stable form of one recorded value."""
    if isinstance(v, np.ndarray):
        return "arr:" + hashlib.sha1(_quant(v).tobytes()).hexdigest()[:12]
    if isinstance(v, (float, np.floating)):
        return None if not np.isfinite(float(v)) else format(float(v), _FLOAT_SIG_DIGITS)
    if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
        return int(v)
    return str(v)


class _Bench:
    """SLM + camera pair whose frame is a deterministic function of the panel."""

    def __init__(self) -> None:
        self.last: np.ndarray | None = None
        self.writes: list[np.ndarray] = []

    def camera(self) -> "_BenchCamera":
        return _BenchCamera(self)

    def slm(self) -> "_BenchSLM":
        return _BenchSLM(self)


class _BenchCamera:
    def __init__(self, bench: _Bench) -> None:
        self.bench = bench
        self.shape = (_H, _W)
        self.exposure_time_ms = 80.0

    def __enter__(self) -> "_BenchCamera":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def open(self) -> None:
        return None

    def close(self) -> None:
        return None

    def is_connected(self) -> bool:
        return True

    def get_numpy_image(self, n_sample: int = 1, skip_first: bool = False) -> np.ndarray:
        flat = self.bench.last
        m = float(np.mean(flat)) if flat is not None and flat.size else 0.0
        s = float(np.std(flat)) if flat is not None and flat.size else 0.0
        yy, xx = np.mgrid[0:_H, 0:_W]
        cy, cx = (_H - 1) / 2.0, (_W - 1) / 2.0
        dy, dx = 4.0 * np.sin(3.0 * m), 4.0 * np.cos(2.0 * m)
        k = np.tanh(30.0 * s)
        sy, sx = 5.0 * (1.0 + 0.45 * k), 5.0 * (1.0 - 0.45 * k)
        blob = np.exp(-(((xx - cx - dx) ** 2) / (2 * sx**2) + ((yy - cy - dy) ** 2) / (2 * sy**2)))
        return (blob * 200.0).astype(np.float64)

    def reset_window(self, center: Any, size: Any) -> Any:
        return self.shape, center

    def set_exposure_time(self, exposure_time_ms: float) -> None:
        return None

    def get_exposure_time(self) -> float:
        return self.exposure_time_ms

    def enable_auto_exposure(self, *a: object, **k: object) -> None:
        return None

    def get_auto_exposure_state(self) -> Any:
        return None

    def set_auto_exposure_range(self, *a: object, **k: object) -> None:
        return None

    @staticmethod
    def get_cam_list() -> list:
        return ["sim"]


class _BenchSLM:
    Gray_Scale_bits = 10

    def __init__(self, bench: _Bench) -> None:
        self.bench = bench

    def __enter__(self) -> "_BenchSLM":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def open(self) -> None:
        return None

    def close(self) -> None:
        return None

    def is_connected(self) -> bool:
        return True

    def display_data(self, data: Any, *a: object, **k: object) -> None:
        arr = np.asarray(data).astype(np.float64)
        self.bench.last = arr.ravel().copy()
        self.bench.writes.append(arr.copy())

    def display_phase(self, phase: Any, *a: object, **k: object) -> None:
        self.display_data(phase)

    def create_phase_from_array(self, phase: Any, *a: object, **k: object) -> np.ndarray:
        arr = np.asarray(phase).astype(np.float64)
        self.bench.writes.append(arr)
        return np.zeros(np.asarray(phase).shape, dtype=np.uint16)

    def display_memory(self, *a: object, **k: object) -> None:
        return None

    def set_grayscale(self, gray: object, *a: object, **k: object) -> None:
        return None

    def get_displayed_memory_number(self) -> int:
        return 0

    @property
    def _max_gray(self) -> int:
        return 1023


def _digest(case: str, monkeypatch: pytest.MonkeyPatch) -> tuple[tuple[int, str, str, int], list]:
    """Golden tuple for one case, plus the canonical rows behind it.

    The rows come back alongside the hash so a failure can print actual numbers:
    a golden that only reports "hash mismatch" is how an intermittent red ends up
    with no named cause (TODO.md 8.9, the unresolved tools-shard flake).
    """
    bench = _Bench()
    monkeypatch.setattr(sq, "create_camera", lambda *a, **k: bench.camera())
    monkeypatch.setattr(sq, "Santec", lambda *a, **k: bench.slm())
    over = dict(CASES[case])
    epochs = over.pop("epochs")
    base = dict(cam_type="sim", target_side=20, cam_size=64, phase_grid=6, random_seed=0)
    base.update(over)
    rec = sq.optimize_slm_square(center=(32, 32), epochs=epochs,
                                 config=sq.SlmSquareConfig(**base))
    rows = [{str(k): _cell(v) for k, v in row.items()} for row in rec.history]
    writes = hashlib.sha1(
        b"".join(_quant(w).tobytes() for w in bench.writes)
    ).hexdigest()
    return (len(rows), hashlib.sha1(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
            writes, len(bench.writes)), rows


def _preview(rows: list) -> str:
    """The first two canonical rows, shortened enough to live in an assert message."""
    text = json.dumps(rows[:2], sort_keys=True)
    return text if len(text) <= 700 else text[:700] + "...(truncated)"


class TestSquareEpochCharacterization:
    """The epoch loop must reproduce its recorded history exactly."""

    @pytest.mark.parametrize("case", sorted(CASES))
    def test_epoch_loop_matches_its_golden_digest(self, case: str,
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
        got, rows = _digest(case, monkeypatch)
        want = EXPECTED[case]
        update = os.environ.get("AO_SQUARE_CHAR_UPDATE")
        if update:
            path = Path(update)
            captured = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            captured[case] = list(got)
            path.write_text(json.dumps(captured, indent=1, sort_keys=True), encoding="utf-8")
            pytest.skip(f"{case}: golden captured into {update}")
        assert got[0] == want[0], f"{case}: {got[0]} rows, golden {want[0]}"
        assert got[3] == want[3], f"{case}: {got[3]} SLM writes, golden {want[3]}"
        assert got[1] == want[1], (
            f"{case}: recorded metrics drifted from the golden rows; "
            f"first rows now = {_preview(rows)}"
        )
        assert got[2] == want[2], (
            f"{case}: phases sent to the SLM drifted; "
            f"first rows now = {_preview(rows)}"
        )

    def test_the_camera_actually_depends_on_the_panel(self) -> None:
        """Guard the guard: a phase-independent blob makes the gradient zero.

        Without this the whole class above would pass for a loop that never
        moves the coefficients -- exactly the blind spot the fixed-blob harness
        has for a pos/neg swap.
        """
        bench = _Bench()
        cam = bench.camera()
        base = cam.get_numpy_image()
        bench.last = np.linspace(0.0, 1.0, 36)
        assert not np.array_equal(base, cam.get_numpy_image())
