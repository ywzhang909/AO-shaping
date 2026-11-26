"""``_SimBench`` 湍流通道 (#253 M5b) 的三条契约。

- 不传 disturbance (默认): 前向求值的场**逐字节**等于 phase+aberration,
  屏路径一步都不走 —— 接入前后孪生输出必须逐位一致 (既有孪生报告的可复现性)。
- 传静态屏: 场 = phase+aberration+光瞳裁剪重采样后的屏, 且每次 measure 取屏
  (计数) —— dynamic/static 的语义归 SimDisturbance, bench 只负责"每次都问"。
- ``last_frame``: measure 后必为返回帧本身 (孪生 recorder _img 的唯一来源)。
"""
from __future__ import annotations

import numpy as np

from ao_shaping.drivers.sim.slm_shaping_bench import ShapingBenchConfig
from ao_shaping.optimizer.wfless.slm_model_in_loop import SlmModelInLoopConfig, _SimBench


class _StubDisturbance:
    def __init__(self, screen: np.ndarray) -> None:
        self._screen = np.asarray(screen, dtype=np.float64)
        self.calls = 0

    def phase(self) -> np.ndarray:
        self.calls += 1
        return self._screen


def _bench(config: SlmModelInLoopConfig, *, disturbance=None):
    bench = _SimBench(config, disturbance=disturbance)
    seen: list[np.ndarray] = []

    def spy(phase: np.ndarray, cfg: ShapingBenchConfig) -> np.ndarray:
        seen.append(np.array(phase, copy=True))
        return np.full((64, 64), 3.0)

    bench._forward = spy
    return bench, seen


def _config() -> SlmModelInLoopConfig:
    return SlmModelInLoopConfig(cam_type="sim", region=64, n_orders=4,
                                cam_size=32, seed=5)


def test_no_disturbance_field_is_byte_identical_to_phase_plus_aberration() -> None:
    bench, seen = _bench(_config())
    phase = np.linspace(-1.0, 1.0, 64 * 64).reshape(64, 64)
    bench.display(phase)
    frame = bench.measure()
    expected = phase + bench._aberration
    assert np.array_equal(seen[0], expected)
    assert np.array_equal(frame, bench.last_frame)


def test_static_screen_is_cropped_to_pupil_and_added_on_every_measure() -> None:
    from scipy.ndimage import zoom

    config = _config()
    screen = np.arange(1200 * 1920, dtype=np.float64).reshape(1200, 1920) * 1e-6
    disturbance = _StubDisturbance(screen)
    bench, seen = _bench(config, disturbance=disturbance)
    phase = np.zeros((64, 64))
    bench.display(phase)
    bench.measure()
    bench.measure()
    assert disturbance.calls == 2
    cx, cy = int(config.pupil_center[0]), int(config.pupil_center[1])
    half = int(round(config.panel_span_px / 2.0))
    crop = screen[cy - half:cy + half, cx - half:cx + half]
    expected_screen = zoom(crop, (64 / crop.shape[0], 64 / crop.shape[1]), order=1)
    residual = seen[1] - bench._aberration
    assert np.allclose(residual, expected_screen, rtol=1e-9, atol=1e-12)


def test_dynamic_semantics_belong_to_the_disturbance_object() -> None:
    class _Advancing(_StubDisturbance):
        def phase(self) -> np.ndarray:
            out = super().phase()
            self._screen = self._screen + 0.01
            return out

    config = _config()
    bench, seen = _bench(config, disturbance=_Advancing(np.zeros((1200, 1920))))
    bench.display(np.zeros((64, 64)))
    bench.measure()
    bench.measure()
    assert not np.array_equal(seen[0], seen[1]), "bench 每次求值都必须重新取屏"
