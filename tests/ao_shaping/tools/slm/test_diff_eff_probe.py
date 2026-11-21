"""轻量衍射效率探针 (``probe`` 子命令 + ``probe.py``) 的离线测试。

不碰硬件: SLM / 相机都用测试替身 (FakeSLM / FakeCamera) 经 ``slm_factory`` /
``cam_factory`` 注入。覆盖:

1. 内存槽轮换 (相邻两次显示不重复) + ``SlotRotator.display_data`` 已含等待,
   调用方**不重复 sleep**;
2. 平场灰度走 raw uint16 路径 (``np.full(..., dtype=np.uint16)``), 不经
   ``create_phase_from_array``;
3. 零级以帧内 ``argmax`` 定位 (画面故意把亮点放在**非中心**);
4. 经 ``BaseCamera`` 注册表创建相机 (默认工厂就是 ``create_camera``);
5. 满量程检查经 ``full_scale(img, camera=cam)``, 过曝触发 warning;
6. CLI: ``probe`` 已注册, 参数/help 正常。
"""

from __future__ import annotations

import importlib
import math
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ao_shaping.tools.slm.calibration.diffraction_efficiency.probe import (
    ProbeConfig,
    integrate_lobes,
    locate_zero_order,
    probe_diffraction_efficiency,
)


# ---------------------------------------------------------------------------
# 测试替身
# ---------------------------------------------------------------------------


class FakeSLM:
    """鸭子类型 SLM: 记录显示过的槽位与灰度图, 供轮换/raw 路径断言。"""

    MAX_GRAYSCALE_VALUE = 1023

    def __init__(self, shape: tuple[int, int] = (8, 8)) -> None:
        self.Panel_Res = (shape[1], shape[0])
        self.displayed_slot = 3  # 初始槽, SlotRotator 必须避开
        self.writes: list[tuple[np.ndarray, int]] = []
        self.sleeps: list[float] = []
        self._max_gray = 993

    def get_displayed_memory_number(self) -> int:
        return self.displayed_slot

    def display_data(
        self, pattern: np.ndarray, memory_number: int, wait_time_s: float | None = None
    ) -> int:
        # SlotRotator.display_data 传的是 (slm, gray, wait_time_s), 由它自己 sleep;
        # 这里只记录, 不 sleep。若调用方重复 sleep, 会体现在 monkeypatch 的计数里。
        self.writes.append((np.array(pattern), memory_number))
        self.displayed_slot = memory_number
        return memory_number


class _FakeDarkFrame:
    """暗帧替身, 与 ``DriverDarkFrame`` 鸭子兼容。"""

    def __init__(self, mean: np.ndarray, exposure_ms: float, frames: int = 8) -> None:
        self.mean = np.asarray(mean, dtype=np.float64)
        self.exposure_ms = float(exposure_ms)
        self.frames = frames
        self.std = None

    def subtract(self, frame: np.ndarray, clip: bool = False) -> np.ndarray:
        out = np.asarray(frame, dtype=np.float64) - self.mean
        return np.clip(out, 0.0, None) if clip else out


class FullScaleCamera:
    """最小相机替身: 生效满量程 255 (模拟 8-bit 后端)。"""

    def __init__(
        self,
        frames: list[np.ndarray],
        *,
        dark_mean: np.ndarray | None = None,
        dark_error: Exception | None = None,
    ) -> None:
        self.frames = frames
        self.index = 0
        self.calls: list[dict[str, Any]] = []
        self._dark_mean = dark_mean
        self._dark_error = dark_error
        self.dark_calls: list[float] = []

    def get_dark_frame(self, exposure_ms: float) -> "_FakeDarkFrame | None":
        self.dark_calls.append(float(exposure_ms))
        if self._dark_error is not None:
            raise self._dark_error
        if self._dark_mean is None:
            return None
        return _FakeDarkFrame(self._dark_mean, exposure_ms)

    def get_numpy_image(self, n_sample: int = 1, skip_first: bool = True) -> np.ndarray:
        self.calls.append({"n_sample": n_sample, "skip_first": skip_first})
        frame = self.frames[self.index % len(self.frames)]
        self.index += 1
        return frame

    def effective_full_scale(self, img: np.ndarray) -> float:
        return 255.0

    def __enter__(self) -> "FullScaleCamera":
        return self

    def __exit__(self, *args: Any) -> None:
        return None


def _two_spot_frame(shape: tuple[int, int], peak_value: float = 100.0) -> np.ndarray:
    """构造一个亮点在**非中心** (cy=2, cx=shape[1]-2) 的帧。"""
    frame = np.zeros(shape, dtype=np.float64)
    frame[2, shape[1] - 2] = peak_value
    return frame


@contextmanager
def _loguru_records():
    from loguru import logger

    seen: list[str] = []
    handler_id = logger.add(lambda m: seen.append(str(m)), level="WARNING")
    try:
        yield seen
    finally:
        logger.remove(handler_id)


@contextmanager
def _fake_devices(
    frames: list[np.ndarray],
    shape: tuple[int, int] = (8, 8),
    *,
    dark_mean: np.ndarray | None = None,
    dark_error: Exception | None = None,
):
    """给 probe 注入假 SLM / 假相机, 返回 (slm, camera) 及工厂记录。"""
    slm = FakeSLM(shape)
    camera = FullScaleCamera(frames, dark_mean=dark_mean, dark_error=dark_error)
    cam_calls: list[dict[str, Any]] = []

    def slm_factory():
        @contextmanager
        def ctx():
            yield slm

        return ctx()

    def cam_factory(cam_type, cam_id, exposure_time_ms=20.0, **kwargs):
        cam_calls.append(
            {
                "cam_type": cam_type,
                "cam_id": cam_id,
                "exposure_time_ms": exposure_time_ms,
            }
        )
        return camera

    yield slm, camera, cam_calls, slm_factory, cam_factory


# ---------------------------------------------------------------------------
# argmax 定位 + 瓣归属
# ---------------------------------------------------------------------------


def test_locate_zero_order_uses_argmax_on_offcenter_frame() -> None:
    frame = _two_spot_frame((9, 11), peak_value=123.0)
    cy, cx, peak = locate_zero_order(frame)
    assert (cy, cx) == (2.0, 9.0)
    assert peak == pytest.approx(123.0)


def test_integrate_lobes_returns_three_ratios_and_coarse_label() -> None:
    frame = _two_spot_frame((9, 11), peak_value=100.0)
    measurement = integrate_lobes(frame, cy=2.0, cx=9.0, period_px=4, axis="x")
    assert set(measurement["ratios"]) == {"zero", "plus_one", "minus_one"}
    assert measurement["calibrated"] is False
    assert "粗略/未定标" in measurement["note"]
    # 三占比之和为 1 (有信号时)
    assert sum(measurement["ratios"][k] for k in ("zero", "plus_one", "minus_one")) == (
        pytest.approx(1.0)
    )


# ---------------------------------------------------------------------------
# probe_diffraction_efficiency: 设备纪律
# ---------------------------------------------------------------------------


def test_probe_rotates_slots_and_never_repeats(tmp_path: Path) -> None:
    frame = _two_spot_frame((8, 8), peak_value=100.0)
    with _fake_devices([frame]) as (slm, _cam, _calls, slm_factory, cam_factory):
        probe_diffraction_efficiency(
            ProbeConfig(out_dir=tmp_path, frames=1),
            slm_factory=slm_factory,
            cam_factory=cam_factory,
        )
    assert len(slm.writes) == 2  # 平场 + 闪耀
    slots = [slot for _gray, slot in slm.writes]
    assert slots[0] != slots[1]
    assert slots[0] != 3 and slots[1] != 3  # 初始槽被避开


def test_probe_sends_flat_as_raw_uint16(tmp_path: Path) -> None:
    frame = _two_spot_frame((8, 8), peak_value=100.0)
    with _fake_devices([frame]) as (slm, _cam, _calls, slm_factory, cam_factory):
        probe_diffraction_efficiency(
            ProbeConfig(out_dir=tmp_path, frames=1, period_px=4, gray_depth=993),
            slm_factory=slm_factory,
            cam_factory=cam_factory,
        )
    flat, slot = slm.writes[0]
    assert flat.dtype == np.uint16
    assert flat.shape == (8, 8)
    assert int(flat.min()) == int(flat.max()) == 0
    blaze, _ = slm.writes[1]
    assert blaze.dtype == np.uint16
    # 周期 4 整除面板宽度 8 ⇒ 斜坡最后一列达到满灰度 993。
    assert int(blaze.max()) == 993


def test_probe_does_not_double_sleep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``SlotRotator.display_data`` 已 sleep, 调用方不得再 ``time.sleep``。"""
    import ao_shaping.drivers.slm.santec.driver as santec_driver

    sleeps: list[float] = []
    monkeypatch.setattr(santec_driver.time, "sleep", lambda s: sleeps.append(s))

    frame = _two_spot_frame((8, 8), peak_value=100.0)
    with _fake_devices([frame]) as (slm, _cam, _calls, slm_factory, cam_factory):
        probe_diffraction_efficiency(
            ProbeConfig(out_dir=tmp_path, frames=1, settle_s=0.25),
            slm_factory=slm_factory,
            cam_factory=cam_factory,
        )
    # 两次显示 = 两次 sleep, 每次 0.25s (来自 SlotRotator, 没有额外 sleep)。
    assert sleeps == [0.25, 0.25]


def test_probe_uses_camera_registry_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """无注入时默认工厂就是 ``create_camera`` (BaseCamera 注册表)。"""
    import ao_shaping.drivers.ccd.common as ccd_common

    frame = _two_spot_frame((8, 8), peak_value=100.0)
    seen: dict[str, Any] = {}

    def fake_create_camera(cam_type, cam_id, exposure_time_ms=20.0, **kwargs):
        seen.update(cam_type=cam_type, cam_id=cam_id, exposure_time_ms=exposure_time_ms)
        return FullScaleCamera([frame])

    monkeypatch.setattr(ccd_common, "create_camera", fake_create_camera)
    # probe.py 里是 ``from ...common import create_camera`` (函数内导入), 所以
    # monkeypatch 模块属性即可让默认路径拿到替身。
    with _fake_devices([frame]) as (_slm, _cam, _calls, slm_factory, _cam_factory):
        probe_diffraction_efficiency(
            ProbeConfig(out_dir=tmp_path, frames=1, cam_type="sim", cam_id=2),
            slm_factory=slm_factory,
            # 不注入 cam_factory → 走 create_camera 注册表
            cam_factory=None,
        )
    assert seen == {"cam_type": "sim", "cam_id": 2, "exposure_time_ms": 5.0}


def test_probe_cam_factory_records_registery_arguments(tmp_path: Path) -> None:
    frame = _two_spot_frame((8, 8), peak_value=100.0)
    with _fake_devices([frame]) as (_slm, _cam, calls, slm_factory, cam_factory):
        probe_diffraction_efficiency(
            ProbeConfig(
                out_dir=tmp_path, frames=1, cam_type="miicam", cam_id=7, exposure_ms=2.5
            ),
            slm_factory=slm_factory,
            cam_factory=cam_factory,
        )
    assert calls == [{"cam_type": "miicam", "cam_id": 7, "exposure_time_ms": 2.5}]


def test_probe_uses_camera_get_numpy_image_with_skip_first(tmp_path: Path) -> None:
    frame = _two_spot_frame((8, 8), peak_value=100.0)
    with _fake_devices([frame]) as (_slm, camera, _calls, slm_factory, cam_factory):
        probe_diffraction_efficiency(
            ProbeConfig(out_dir=tmp_path, frames=3),
            slm_factory=slm_factory,
            cam_factory=cam_factory,
        )
    assert camera.calls == [
        {"n_sample": 3, "skip_first": True},
        {"n_sample": 3, "skip_first": True},
    ]


def test_probe_saturation_guard_warns(tmp_path: Path) -> None:
    frame = _two_spot_frame((8, 8), peak_value=250.0)  # > 0.8 × 255
    with _fake_devices([frame]) as (slm, _cam, _calls, slm_factory, cam_factory):
        with _loguru_records() as records:
            probe_diffraction_efficiency(
                ProbeConfig(out_dir=tmp_path, frames=1),
                slm_factory=slm_factory,
                cam_factory=cam_factory,
            )
    assert any("过曝" in message for message in records)
    assert slm.writes  # 采集确实发生


def test_probe_locates_zero_order_from_baseline_not_blazed(tmp_path: Path) -> None:
    """灰度深度接近 2π 时零级几乎熄灭, 不能从闪耀帧 argmax 定位零级。

    让平场帧的亮点在 (2, 6), 闪耀帧的亮点 (一级) 在 (2, 2): 若误用闪耀帧
    argmax, 零级会被错定到 (2, 2)。
    """
    baseline = np.zeros((8, 8), dtype=np.float64)
    baseline[2, 6] = 100.0
    blazed = np.zeros((8, 8), dtype=np.float64)
    blazed[2, 2] = 150.0
    with _fake_devices([baseline, blazed]) as (
        slm,
        _cam,
        _calls,
        slm_factory,
        cam_factory,
    ):
        result = probe_diffraction_efficiency(
            ProbeConfig(out_dir=tmp_path, frames=1),
            slm_factory=slm_factory,
            cam_factory=cam_factory,
        )
    assert (result["zero_order"]["cy"], result["zero_order"]["cx"]) == (2.0, 6.0)
    assert result["zero_order"]["from"] == "baseline argmax"
    assert result["zero_order"]["blazed_frame_peak"] == pytest.approx(150.0)


def test_probe_writes_summary_json_and_csv(tmp_path: Path) -> None:
    frame = _two_spot_frame((8, 8), peak_value=100.0)
    with _fake_devices([frame]) as (_slm, _cam, _calls, slm_factory, cam_factory):
        result = probe_diffraction_efficiency(
            ProbeConfig(out_dir=tmp_path, frames=1, save_frames=True),
            slm_factory=slm_factory,
            cam_factory=cam_factory,
        )
    assert result["artifacts"]["summary_json"]
    assert result["artifacts"]["summary_csv"]
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "summary.csv").exists()
    assert (tmp_path / "baseline.npy").exists()
    assert (tmp_path / "blazed.npy").exists()
    assert result["lobe"]["note"]


# ---------------------------------------------------------------------------
# CLI 表面
# ---------------------------------------------------------------------------


def test_probe_subcommand_registered_and_flags_exposed() -> None:
    from click.testing import CliRunner

    module = importlib.import_module(
        "ao_shaping.tools.slm.calibration.diffraction_efficiency.__main__"
    )
    runner = CliRunner()
    top = runner.invoke(module.main, ["--help"])
    assert top.exit_code == 0, top.output
    assert "probe" in top.output

    help_result = runner.invoke(module.main, ["probe", "--help"])
    assert help_result.exit_code == 0, help_result.output
    for flag in (
        "--out-dir",
        "--slm-number",
        "--slm-wavelength",
        "--cam-type",
        "--cam-id",
        "--exposure-ms",
        "--gray-depth",
        "--period-px",
        "--axis",
        "--frames",
        "--settle-s",
        "--save-frames",
        "--subtract-dark",
    ):
        assert flag in help_result.output


def test_probe_params_defaults_match_design() -> None:
    module = importlib.import_module(
        "ao_shaping.tools.slm.calibration.diffraction_efficiency.__main__"
    )
    params = module.ProbeParams()
    assert params.out_dir == Path("data/diffraction_efficiency/probe")
    assert params.slm_number == 1
    assert params.slm_wavelength == 1064
    assert params.cam_type == "daheng"
    assert params.cam_id == 0
    assert params.exposure_ms == 5.0
    assert params.gray_depth == 0
    assert params.period_px == 32
    assert params.axis == "x"
    assert params.frames == 3
    assert params.settle_s == 0.25
    assert params.save_frames is False
    assert params.subtract_dark is True


def test_config_validates_inputs() -> None:
    with pytest.raises(ValueError):
        ProbeConfig(period_px=1)
    with pytest.raises(ValueError):
        ProbeConfig(axis="z")
    with pytest.raises(ValueError):
        ProbeConfig(frames=0)


def test_probe_reads_two_pi_grayscale_from_device_when_unspecified(
    tmp_path: Path,
) -> None:
    frame = _two_spot_frame((8, 8), peak_value=100.0)
    with _fake_devices([frame]) as (slm, _cam, _calls, slm_factory, cam_factory):
        result = probe_diffraction_efficiency(
            ProbeConfig(out_dir=tmp_path, frames=1, period_px=4),
            slm_factory=slm_factory,
            cam_factory=cam_factory,
        )
    # gray_depth=0 -> 设备读 _max_gray (993), 不是硬件满量程 1023。
    assert result["config"]["gray_depth"] == 993
    assert int(slm.writes[-1][0].max()) == 993


# ---------------------------------------------------------------------------
# 暗帧 / 背景
# ---------------------------------------------------------------------------


def test_probe_subtracts_available_dark_frame(tmp_path: Path) -> None:
    dark = np.full((8, 8), 20.0)
    baseline_raw = _two_spot_frame((8, 8), peak_value=100.0) + dark
    blazed_raw = _two_spot_frame((8, 8), peak_value=40.0) + dark
    with _fake_devices([baseline_raw, blazed_raw], dark_mean=dark) as (
        _slm,
        cam,
        _calls,
        slm_factory,
        cam_factory,
    ):
        result = probe_diffraction_efficiency(
            ProbeConfig(out_dir=tmp_path, frames=1, save_frames=True),
            slm_factory=slm_factory,
            cam_factory=cam_factory,
        )
    assert cam.dark_calls == [5.0]
    assert result["dark"]["available"] is True
    assert result["dark"]["requested"] is True
    assert result["dark"]["mean_counts"] == pytest.approx(20.0)
    assert result["dark"]["reason"] is None
    # 扣暗后的峰值 = 100; 未扣则是 120。
    assert result["peaks"]["baseline_counts"] == pytest.approx(100.0)
    assert result["zero_order"]["baseline_peak"] == pytest.approx(100.0)
    saved = np.load(tmp_path / "baseline.npy")
    assert saved[2, 6] == pytest.approx(100.0)
    raw = np.load(tmp_path / "baseline_raw.npy")
    assert raw[2, 6] == pytest.approx(120.0)


def test_probe_records_missing_dark_without_failing(tmp_path: Path) -> None:
    frame = _two_spot_frame((8, 8), peak_value=100.0)
    with _loguru_records() as records:
        with _fake_devices([frame]) as (_slm, _cam, _calls, slm_factory, cam_factory):
            result = probe_diffraction_efficiency(
                ProbeConfig(out_dir=tmp_path, frames=1),
                slm_factory=slm_factory,
                cam_factory=cam_factory,
            )
    assert result["dark"]["available"] is False
    assert "无同曝光暗帧" in result["dark"]["reason"]
    assert result["peaks"]["baseline_counts"] == pytest.approx(100.0)
    assert any("暗帧不可用" in message for message in records)


def test_probe_survives_dark_lookup_error_and_no_subtract_flag(tmp_path: Path) -> None:
    frame = _two_spot_frame((8, 8), peak_value=100.0)
    with _fake_devices([frame], dark_error=RuntimeError("no serial")) as (
        _slm,
        _cam,
        _calls,
        slm_factory,
        cam_factory,
    ):
        result = probe_diffraction_efficiency(
            ProbeConfig(out_dir=tmp_path, frames=1),
            slm_factory=slm_factory,
            cam_factory=cam_factory,
        )
    assert result["dark"]["available"] is False
    assert "no serial" in result["dark"]["reason"]

    with _fake_devices([frame]) as (_slm, cam, _calls, slm_factory, cam_factory):
        disabled = probe_diffraction_efficiency(
            ProbeConfig(out_dir=tmp_path / "off", frames=1, subtract_dark=False),
            slm_factory=slm_factory,
            cam_factory=cam_factory,
        )
    assert cam.dark_calls == []
    assert disabled["dark"]["requested"] is False
    assert disabled["dark"]["available"] is False
    assert "disabled" in disabled["dark"]["reason"]


# ---------------------------------------------------------------------------
# 相对效率指标
# ---------------------------------------------------------------------------


def test_relative_efficiency_normalizes_against_baseline_zero_window(
    tmp_path: Path,
) -> None:
    baseline_raw = np.zeros((64, 64), dtype=np.float64)
    baseline_raw[32, 32] = 100.0
    blazed_raw = np.zeros((64, 64), dtype=np.float64)
    blazed_raw[32, 40] = 50.0  # +1 级 (x 轴, separation = 64/(2*8) = 4 -> max 8)
    with _fake_devices([baseline_raw, blazed_raw]) as (
        _slm,
        _cam,
        _calls,
        slm_factory,
        cam_factory,
    ):
        result = probe_diffraction_efficiency(
            ProbeConfig(out_dir=tmp_path, frames=1, period_px=8, axis="x"),
            slm_factory=slm_factory,
            cam_factory=cam_factory,
        )
    relative = result["relative_efficiency"]
    assert relative["baseline_zero_window_counts"] == pytest.approx(100.0)
    assert relative["plus_one"] == pytest.approx(0.5)
    assert relative["minus_one"] == pytest.approx(0.0)
    assert relative["first_orders_sum"] == pytest.approx(0.5)
    assert "非绝对衍射效率" in relative["note"]


def test_summary_json_schema_covers_dark_and_metrics(tmp_path: Path) -> None:
    import json

    frame = _two_spot_frame((8, 8), peak_value=100.0)
    with _fake_devices([frame]) as (_slm, _cam, _calls, slm_factory, cam_factory):
        result = probe_diffraction_efficiency(
            ProbeConfig(out_dir=tmp_path, frames=1),
            slm_factory=slm_factory,
            cam_factory=cam_factory,
        )
    payload = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert payload["probe"] == "diffraction_efficiency"
    assert {
        "config",
        "zero_order",
        "lobe",
        "relative_efficiency",
        "peaks",
        "dark",
    } <= set(payload)
    assert {"available", "requested", "exposure_ms", "reason"} <= set(payload["dark"])
    assert {"plus_one", "minus_one", "first_orders_sum"} <= set(
        payload["relative_efficiency"]
    )
    assert payload["config"]["subtract_dark"] == (result["config"]["subtract_dark"])
