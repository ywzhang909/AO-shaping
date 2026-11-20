"""实测闭环 Bench 的设备事务测试（不打开真实 SLM / CCD / 激光）。

``HardwareBench`` 是唯一碰真实驱动的一层，所以这里用注入的假 SLM / 假相机把
「时序契约」钉死：

* ``open()`` 先开 SLM、过驱动板准入、再开相机，并清空覆盖相位、重设曝光；
* ``apply_phase()`` 走 ``create_phase_from_array`` → 轮换内存槽 → 下发 → 稳定等待，
  等待时间不短于标称稳定时间；
* ``capture()`` 先丢过渡帧再取新帧，帧号单调、曝光/满量程回读随帧带出；
* 驱动板不应答时 ``open()`` 必须失败并关掉已开句柄（整段测量无效，不能带着假面板跑）。
"""

from __future__ import annotations

import numpy as np
import pytest

from ao_shaping.gui.slm.closed_loop_bench import HardwareBench
from ao_shaping.utils.slm.transition_timeline import pixel_flip_wait_s

_SENSOR = 4


class _FakeSLM:
    """记录每次调用与参数的假 SLM。"""

    def __init__(self) -> None:
        self.init_kwargs: dict = {}
        self.opened = False
        self.closed = False
        self.overlays: list = []
        self.displayed: list[tuple[np.ndarray, float | None, int | None]] = []
        self.board_calls = 0
        self.board_error: Exception | None = None

    def open(self) -> None:
        self.opened = True

    def close(self) -> None:
        self.closed = True

    def require_drive_board(self) -> str:
        self.board_calls += 1
        if self.board_error is not None:
            raise self.board_error
        return "BOARD-SN"

    def set_phase_overlay(self, phase) -> None:
        self.overlays.append(phase)

    def create_phase_from_array(self, phase_rad, max_grayscale=None) -> np.ndarray:
        # 返回与输入同形的 0..1023 斜坡，ptp=1023，稳定等待按最大跳变估。
        phase = np.asarray(phase_rad)
        ramp = np.linspace(0, 1023, phase.shape[1], dtype=np.uint16)
        return np.tile(ramp, (phase.shape[0], 1))

    def display_data(
        self, phase_gray, wait_time_s=None, memory_number=None, memory_mode=0
    ) -> int:
        self.displayed.append(
            (np.array(phase_gray, copy=True), wait_time_s, memory_number)
        )
        return int(memory_number or 0)

    def get_wavelength_info(self) -> tuple[int, int]:
        return (1064, 993)


class _FakeCamera:
    """记录取帧次数与曝光设置的假相机。"""

    def __init__(self) -> None:
        self.init_args: tuple = ()
        self.init_kwargs: dict = {}
        self.opened = False
        self.closed = False
        self.frames_taken = 0
        self.resets: list[float] = []
        self.exposure_ms = 20.0

    def open(self) -> None:
        self.opened = True

    def close(self) -> None:
        self.closed = True

    def reset_exposure_time(self, time_ms: float) -> float:
        self.resets.append(float(time_ms))
        self.exposure_ms = float(time_ms)
        return self.exposure_ms

    def get_exposure_time_ms(self) -> float:
        return self.exposure_ms

    def get_numpy_image(self, n_sample: int = 1, skip_first: bool = True) -> np.ndarray:
        self.frames_taken += 1
        return np.full((_SENSOR, _SENSOR), self.frames_taken, dtype=np.uint8)

    def effective_full_scale(self, img: np.ndarray) -> float:
        return 255.0


def _bench(
    *,
    slm: _FakeSLM | None = None,
    cam: _FakeCamera | None = None,
    sleeper: list[float] | None = None,
    **kwargs,
) -> tuple[HardwareBench, _FakeSLM, _FakeCamera, list[float]]:
    slm = slm if slm is not None else _FakeSLM()
    cam = cam if cam is not None else _FakeCamera()
    slept = sleeper if sleeper is not None else []

    def slm_factory(**kw):
        slm.init_kwargs = dict(kw)
        return slm

    def camera_factory(camera_type, cam_id=0, exposure_time_ms=20.0):
        cam.init_args = (camera_type,)
        cam.init_kwargs = {"cam_id": cam_id, "exposure_time_ms": exposure_time_ms}
        cam.exposure_ms = float(exposure_time_ms)
        return cam

    bench = HardwareBench(
        slm_factory=slm_factory,
        camera_factory=camera_factory,
        sleep=slept.append,
        **kwargs,
    )
    return bench, slm, cam, slept


def test_open_configures_devices_and_discards_transition_frames() -> None:
    """open() 按契约配置两端，并先丢掉过渡帧再认为相机就绪。"""
    bench, slm, cam, _ = _bench(discard_frames=3)
    bench.open()

    # SLM 用面板/波长/内存模式构造；相机用类型 + id + 曝光构造。
    assert slm.init_kwargs == {"slm_number": 1, "wavelength": 1064, "video_mode": 0}
    assert cam.init_args == ("daheng",)
    assert cam.init_kwargs == {"cam_id": 0, "exposure_time_ms": 20.0}

    assert slm.opened and cam.opened
    assert slm.board_calls == 1  # 驱动板准入在取第一帧前跑过
    assert slm.overlays == [None]  # 清空覆盖相位，避免叠加上次的像差
    assert cam.resets == [20.0]
    assert cam.frames_taken == 3  # open() 丢掉 discard_frames 帧
    assert bench.exposure_ms == 20.0
    assert bench._rotator is not None

    bench.close()


def test_apply_phase_writes_rotated_slot_and_waits_the_flip_time() -> None:
    """下发走 create_phase_from_array → 轮换槽 → display_data，再等稳定。"""
    bench, slm, _cam, slept = _bench(settle_s=0.0)
    bench.open()
    slm.displayed.clear()
    slept.clear()

    ctrl = np.zeros((8, 8), dtype=np.float64)
    phase_id = bench.apply_phase(ctrl)

    assert len(slm.displayed) == 1
    gray, wait_time_s, slot = slm.displayed[0]
    assert gray.dtype == np.uint16
    assert gray.shape == ctrl.shape
    assert wait_time_s == 0  # 等待由 Bench 自己 sleep，显式可控
    assert 2 <= slot <= 125  # SlotRotator 的可用槽范围
    assert phase_id == f"slot-{slot}"

    # 等待 = max(灰度翻转时间(1023 跳变), settle_s)；settle_s=0 时即为翻转时间。
    expected = max(pixel_flip_wait_s(1023), 0.0)
    assert slept == [expected]
    assert expected > 0.0

    bench.close()


def test_capture_discards_then_returns_auditable_frame() -> None:
    """capture() 先丢过渡帧再取新帧；帧号单调、曝光/满量程随帧带出。"""
    bench, _slm, cam, _ = _bench(discard_frames=2)
    bench.open()
    before = cam.frames_taken

    frame = bench.capture()

    assert cam.frames_taken == before + 2 + 1  # 丢弃 2 帧 + 取 1 帧
    assert frame.frame.ndim == 2
    assert frame.exposure_ms == 20.0
    assert frame.full_scale == 255.0
    assert frame.frame_id == 1
    assert frame.monotonic_ns > 0

    second = bench.capture()
    assert second.frame_id == 2  # 单调递增，可供审计先后

    bench.close()


def test_describe_reports_config_and_wavelength_info() -> None:
    bench, _slm, _cam, _ = _bench()
    bench.open()
    info = bench.describe()
    assert info["slm_number"] == 1
    assert info["wavelength"] == 1064
    assert info["camera"] == "daheng:0"
    assert info["slm_wavelength_info"] == (1064, 993)
    bench.close()


def test_close_closes_camera_then_slm_and_is_idempotent() -> None:
    bench, slm, cam, _ = _bench()
    bench.open()
    bench.close()
    assert cam.closed is True
    assert slm.closed is True
    # 再关一次不应抛异常。
    bench.close()
    assert bench._slm is None
    assert bench._camera is None


def test_open_fails_closed_when_drive_board_is_silent() -> None:
    """驱动板不应答：open() 抛错并关掉已开句柄，绝不带着假面板取帧。"""
    slm = _FakeSLM()
    slm.board_error = RuntimeError("drive board silent")
    bench, _slm, cam, _ = _bench(slm=slm)

    with pytest.raises(RuntimeError, match="drive board silent"):
        bench.open()

    assert slm.opened is True  # SLM 开过
    assert slm.closed is True  # 失败后已关闭
    assert cam.closed is True
    assert cam.opened is False  # 准入在开相机之前，相机从未打开
    assert bench._slm is None


def test_apply_phase_and_capture_require_open() -> None:
    bench, _slm, _cam, _ = _bench()
    with pytest.raises(RuntimeError):
        bench.apply_phase(np.zeros((8, 8)))
    with pytest.raises(RuntimeError):
        bench.capture()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"exposure_ms": 0.0},
        {"exposure_ms": -1.0},
        {"settle_s": -0.1},
        {"discard_frames": -1},
    ],
)
def test_invalid_configuration_is_rejected(kwargs) -> None:
    with pytest.raises(ValueError):
        HardwareBench(**kwargs)
