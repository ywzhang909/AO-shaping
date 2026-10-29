"""光路无光的早判: 每一条断言都对应 2026-10-28/29 那次失败的一个具体环节。

背景: 光路失效时 ``adapt_exposure`` 连跑 48 s 才抛一句
``RuntimeError("exposure did not converge")``, 而那句话**不区分**「光路没光」与
「曝光参数越界」; 两天后一条被挡掉一部分的光路 (光通量 22%) 又照样通过了只看
峰值的检查。所以这里要锁住四件事:

1. 暗阶梯在第 2 档就停 (不是 12 档);
2. 抛的是带实测数字的专用异常, 退出码 3, 且原因分成「遮挡/未出光」与
   「曝光上限不足」两类 (处置方向不同);
3. 有光的阶梯一个字都不改 (同曝光序列、同 status), 弱光但会涨的情况不许误判;
4. 「有光但进不了接受带」仍是原来的不收敛错误 —— 别把两类错并成一类。
"""
from __future__ import annotations

import numpy as np
import pytest

from ao_shaping.tools.slm.panel.focus_probe import ExposureStep, adapt_exposure
from ao_shaping.tools.slm.transfer.light_guard import (
    CAUSE_BLOCKED,
    CAUSE_EXPOSURE_CEILING,
    EXIT_OPTICAL_PATH_INVALID,
    LightGuard,
    LadderReading,
    OpticalPathInvalid,
    verify_light,
)
from ao_shaping.utils.image.exposure_control import FrameNoiseFloor


def _frame(peak: float, *, size: int = 64, spot: int = 7) -> np.ndarray:
    """A frame whose despiked peak is ``peak`` (an N x N spot, zero background).

    The spot is more than one pixel wide on purpose: ``robust_peak`` runs a 3x3
    median, which deletes an isolated hot pixel but leaves a real spot intact.
    """
    frame = np.zeros((size, size), dtype=np.uint8)
    start = (size - spot) // 2
    frame[start:start + spot, start:start + spot] = int(round(min(peak, 255.0)))
    return frame


def _step(index: int, exposure: float, peak: float, *, is_last: bool = False,
          threshold: float = 4.0, sigma: float = 1.0) -> ExposureStep:
    return ExposureStep(index=index, exposure_ms=exposure, peak_counts=peak,
                        noise=FrameNoiseFloor(median_counts=0.0, sigma_counts=sigma,
                                              threshold_counts=threshold),
                        is_last=is_last)


class _Bench:
    """A camera whose peak is ``counts_per_ms * exposure`` (linear, as measured)."""

    def __init__(self, counts_per_ms: float) -> None:
        self.counts_per_ms = float(counts_per_ms)
        self.exposure_ms = 0.0
        self.reads = 0
        self.exposures: list[float] = []

    def set_exposure(self, ms: float) -> float:
        self.exposure_ms = float(ms)
        self.exposures.append(float(ms))
        return float(ms)

    def read(self) -> np.ndarray:
        self.reads += 1
        return _frame(self.counts_per_ms * self.exposure_ms)


# -- 1. 暗阶梯早停 ---------------------------------------------------------

def test_a_dark_ladder_stops_after_two_rungs_not_the_whole_budget():
    """光路无光时, 12 档预算只该花掉 2 档 (2026-10-28 是 48 s 走完才报错)。"""
    bench = _Bench(counts_per_ms=0.0)
    guard = LightGuard(label="probe test", max_steps=12)
    with pytest.raises(OpticalPathInvalid) as caught:
        adapt_exposure(bench.set_exposure, bench.read, start_ms=0.02, floor_ms=0.02,
                       ceiling_ms=6.0, full_scale=255, max_steps=12,
                       on_step=guard.observe)
    exc = caught.value
    assert exc.exit_code == EXIT_OPTICAL_PATH_INVALID == 3
    assert len(guard.readings) == 2
    # 每档 = discard 2 帧 + 1 帧读数; 旧的 12 档要 36 帧。
    assert bench.reads == 6
    assert exc.reason == "flat"
    assert exc.exposures_ms == (0.02, 0.08)
    assert not guard.seen_light


def test_the_exception_quotes_the_measured_numbers_and_one_likely_cause():
    bench = _Bench(counts_per_ms=0.0)
    guard = LightGuard(label="probe 007", max_steps=12)
    with pytest.raises(OpticalPathInvalid) as caught:
        adapt_exposure(bench.set_exposure, bench.read, start_ms=0.02, floor_ms=0.02,
                       ceiling_ms=6.0, full_scale=255, max_steps=12,
                       on_step=guard.observe)
    text = caught.value.report()
    assert "光路无光" in text and "probe 007" in text
    assert "0.02→0.08 ms" in text          # 曝光档
    assert "噪声底" in text                 # 判据的参照
    assert "0.000 counts/ms" in text        # 峰值率 (与探针同一口径)
    assert "遮挡" in text or "激光" in text  # 单一最可能原因
    assert "probe_optical_path_validity.py" in text  # 下一步该跑什么
    assert caught.value.as_dict()["cause"] == CAUSE_BLOCKED


def test_a_peak_that_still_rises_points_at_the_exposure_ceiling_not_the_path():
    """峰值随曝光上升却始终没到噪声底 ⇒ 先怀疑会话曝光上限, 而不是去拆光路。"""
    guard = LightGuard(label="probe test", max_steps=3)
    guard.observe(_step(1, 0.02, 0.0))
    guard.observe(_step(2, 0.08, 1.0))
    with pytest.raises(OpticalPathInvalid) as caught:
        guard.observe(_step(3, 0.32, 3.0, is_last=True))
    assert caught.value.reason == "exhausted"
    assert caught.value.as_dict()["cause"] == CAUSE_EXPOSURE_CEILING
    assert "AO_DAHENG_MAX_EXPOSURE_MS" in caught.value.report()


# -- 2. 有光时一个字节都不改 -----------------------------------------------

def test_a_lit_ladder_converges_exactly_as_it_did_without_the_guard():
    guarded, plain = _Bench(counts_per_ms=2000.0), _Bench(counts_per_ms=2000.0)
    guard = LightGuard(label="probe test", max_steps=6)
    with_guard = adapt_exposure(guarded.set_exposure, guarded.read, start_ms=0.02,
                                floor_ms=0.02, ceiling_ms=6.0, full_scale=255,
                                max_steps=6, on_step=guard.observe)
    without = adapt_exposure(plain.set_exposure, plain.read, start_ms=0.02,
                             floor_ms=0.02, ceiling_ms=6.0, full_scale=255, max_steps=6)
    assert with_guard == without
    assert with_guard.status == "ok"
    assert guarded.exposures == plain.exposures
    assert guard.seen_light


def test_a_weak_but_rising_peak_is_not_declared_dark():
    """2 counts 的真信号在 4 倍曝光后会到 8 counts —— 不许把它当成噪声底卡死。"""
    bench = _Bench(counts_per_ms=50.0)   # 0.02 ms -> 1, 0.08 -> 4, 0.32 -> 16 counts
    guard = LightGuard(label="probe test", max_steps=6)
    setting = adapt_exposure(bench.set_exposure, bench.read, start_ms=0.02,
                             floor_ms=0.02, ceiling_ms=6.0, full_scale=255,
                             max_steps=6, on_step=guard.observe)
    assert guard.seen_light
    assert setting.status == "ok"


def test_light_once_seen_is_never_revoked():
    guard = LightGuard(label="probe test", max_steps=3)
    guard.observe(_step(1, 0.02, 40.0))
    guard.observe(_step(2, 0.08, 0.0))
    guard.observe(_step(3, 0.32, 0.0, is_last=True))   # 不再抛「无光」
    assert guard.seen_light
    assert guard.peak_at_light_counts == 40.0


# -- 3. 「有光但进不了接受带」仍走原来的不收敛错误 -------------------------

def test_light_that_never_reaches_the_band_keeps_the_old_non_convergence_error():
    """峰值高于噪声底、却始终进不了 (0.25, 0.80): 这是参数问题, 归旧的 RuntimeError。"""
    bench = _Bench(counts_per_ms=100.0)
    guard = LightGuard(label="probe test", max_steps=4)
    setting = adapt_exposure(bench.set_exposure, bench.read, start_ms=0.02,
                             floor_ms=0.02, ceiling_ms=0.2, full_scale=255,
                             max_steps=4, on_step=guard.observe)
    assert setting.status == "dim"      # 调用方仍抛 "exposure did not converge"
    assert guard.seen_light


# -- 4. 启动前的便宜检查 ---------------------------------------------------

def test_the_precheck_returns_at_the_first_rung_that_sees_light():
    bench = _Bench(counts_per_ms=400.0)     # 0.05 ms -> 20 counts > 噪声底
    verdict = verify_light(bench.read, bench.set_exposure, floor_ms=0.02,
                           ceiling_ms=6.0, frames=2, label="startup")
    assert verdict["ok"] is True
    assert bench.exposures == [0.05]
    assert bench.reads == 2
    assert verdict["peak_counts"] > verdict["noise_floor_counts"]


def test_the_precheck_raises_when_every_rung_is_noise():
    bench = _Bench(counts_per_ms=0.0)
    with pytest.raises(OpticalPathInvalid) as caught:
        verify_light(bench.read, bench.set_exposure, floor_ms=0.02, ceiling_ms=6.0,
                     label="startup")
    exc = caught.value
    assert exc.reason == "precheck"
    assert exc.exposures_ms == (0.05, 0.4, 3.2)
    assert exc.budget == 3
    assert bench.reads == 6             # 3 档 x 2 帧, 不是整段阶梯
    assert "startup" in exc.report()


def test_the_precheck_rungs_are_clamped_to_the_session_range():
    bench = _Bench(counts_per_ms=0.0)
    with pytest.raises(OpticalPathInvalid) as caught:
        verify_light(bench.read, bench.set_exposure, floor_ms=1.0, ceiling_ms=2.0)
    assert caught.value.exposures_ms == (1.0, 2.0)


def test_the_verdict_dict_is_machine_readable():
    reading = LadderReading(index=1, exposure_ms=0.4, peak_counts=3.0,
                            noise=FrameNoiseFloor(0.0, 1.0, 4.0))
    exc = OpticalPathInvalid(reason="flat", cause=CAUSE_BLOCKED, readings=(reading,),
                             budget=12, what="probe 001")
    payload = exc.as_dict()
    assert payload["exit_code"] == 3
    assert payload["cause"] == CAUSE_BLOCKED
    assert payload["readings"][0]["noise_floor_counts"] == 4.0
    assert payload["counts_per_ms"] == pytest.approx(7.5)


# -- 5. 退出码是契约: CLI 顶层把异常映射成 3, 不是 1 ----------------------

def _dark_path() -> OpticalPathInvalid:
    reading = LadderReading(index=2, exposure_ms=0.4, peak_counts=3.0,
                            noise=FrameNoiseFloor(0.0, 1.0, 4.0))
    return OpticalPathInvalid(reason="flat", cause=CAUSE_BLOCKED, readings=(reading,),
                              budget=12, what="lane 02")


def test_the_capture_cli_maps_a_dark_path_to_exit_code_three(monkeypatch, tmp_path):
    """``python -m ...capture`` 的退出码是 lane 脚本分流的依据, 必须钉住。

    click 之外的 argparse 入口只把 ``SystemExit`` 交给 shell; 若这里退回普通
    异常, 进程退出码是 1, 与「代码缺陷」混在一起, lane 也就没法 ``exit 3`` 分流。
    """
    import sys

    from ao_shaping.tools.slm.transfer import capture as capture_module

    monkeypatch.setattr(capture_module, "probes", lambda *a, **k: [])
    monkeypatch.setattr(capture_module, "capture", lambda args: (_ for _ in ()).throw(_dark_path()))
    monkeypatch.setattr(sys, "argv", ["capture", "--out", str(tmp_path / "c"),
                                      "--execute", "--count", "6"])
    with pytest.raises(SystemExit) as caught:
        capture_module.main()
    assert caught.value.code == EXIT_OPTICAL_PATH_INVALID


def test_the_validate_cli_maps_a_dark_path_to_exit_code_three(monkeypatch, tmp_path):
    import sys

    from ao_shaping.tools.slm.transfer import validate as validate_module

    monkeypatch.setattr(validate_module, "run", lambda args: (_ for _ in ()).throw(_dark_path()))
    monkeypatch.setattr(sys, "argv", ["validate", "--checkpoint", str(tmp_path / "frozen.json"),
                                      "--out", str(tmp_path / "out"),
                                      "--zero-xy", "0", "0"])
    with pytest.raises(SystemExit) as caught:
        validate_module.main()
    assert caught.value.code == EXIT_OPTICAL_PATH_INVALID


def test_the_workflow_cli_maps_a_dark_path_to_exit_code_three(monkeypatch, tmp_path):
    import sys

    from ao_shaping.tools.slm.transfer import workflow as workflow_module

    monkeypatch.setattr(workflow_module, "TransferSession", lambda **kwargs: object())
    monkeypatch.setattr(workflow_module, "run_session",
                        lambda config, offline=False: (_ for _ in ()).throw(_dark_path()))
    monkeypatch.setattr(sys, "argv", ["workflow", "run", "--out", str(tmp_path / "w"),
                                      "--execute"])
    with pytest.raises(SystemExit) as caught:
        workflow_module.main()
    assert caught.value.code == EXIT_OPTICAL_PATH_INVALID
