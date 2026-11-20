"""实测闭环的设备事务实现：SLM 下发 → 稳定 → CCD 取新帧。

与 :mod:`closed_loop_shaping` 的纯逻辑分开，原因是两件事的测试代价完全不同：
前者是纯 torch/numpy，能在一秒内跑几十轮；后者一构造就碰驱动。拆开之后闭环
状态机可以用假 Bench 全量离线验证，这里只留一层薄薄的、可注入工厂的设备适配。

线程约定
--------
**所有公开方法都必须在同一个 worker 线程里调用**（``apply_phase`` / ``capture`` /
``close``）。SLM 句柄与相机句柄不可跨线程、不可重入；页面层负责把整场闭环放进
一个专用 QThread，不在这里偷偷建线程。

"这帧晚于这次写入"的保证
------------------------
1. ``apply_phase`` 写完后按灰度步长估算翻转时间并额外 sleep 稳定时间；
2. ``capture`` 先丢掉 ``discard_frames`` 帧（SDK 缓冲 + 写入过渡帧），再取一帧；
3. 每帧带本地单调 ``frame_id`` 与 ``monotonic_ns``，闭环记录里可审计先后顺序。

现有代码里 ``differentiable_beam`` 也用同样的"先丢几帧"手法，这里沿用而不发明
第二套同步协议。
"""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np

from ao_shaping.gui.slm.closed_loop_shaping import CapturedFrame

__all__ = ["HardwareBench"]


class HardwareBench:
    """真实 SLM + CCD 的 :class:`~closed_loop_shaping.Bench` 实现。

    ``slm_factory`` / ``camera_factory`` 默认在第一次使用前惰性导入真实驱动；
    测试可以传入返回假设备的可调用对象，从而完全离线验证下发/丢弃帧/回读时序。
    """

    def __init__(
        self,
        *,
        slm_number: int = 1,
        wavelength: int | None = 1064,
        camera_type: str = "daheng",
        cam_id: int | str = 0,
        exposure_ms: float = 20.0,
        settle_s: float = 0.15,
        discard_frames: int = 3,
        slm_factory: Callable[..., Any] | None = None,
        camera_factory: Callable[..., Any] | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        if exposure_ms <= 0 or not np.isfinite(exposure_ms):
            raise ValueError("exposure_ms must be finite and positive")
        if settle_s < 0 or not np.isfinite(settle_s):
            raise ValueError("settle_s must be nonnegative and finite")
        if discard_frames < 0:
            raise ValueError("discard_frames must be >= 0")
        self.slm_number = int(slm_number)
        self.wavelength = None if wavelength is None else int(wavelength)
        self.camera_type = str(camera_type)
        self.cam_id = cam_id
        self.exposure_ms = float(exposure_ms)
        self.settle_s = float(settle_s)
        self.discard_frames = int(discard_frames)

        self._slm_factory = slm_factory
        self._camera_factory = camera_factory
        self._sleep = sleep if sleep is not None else time.sleep
        self._slm: Any | None = None
        self._camera: Any | None = None
        self._rotator: Any | None = None
        self._frame_id = 0
        self._last_step_gray = 0.0

    # -- Bench 接口 -------------------------------------------------------- #

    def describe(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "slm_number": self.slm_number,
            "wavelength": self.wavelength,
            "camera": f"{self.camera_type}:{self.cam_id}",
            "exposure_ms": self.exposure_ms,
            "settle_s": self.settle_s,
            "discard_frames": self.discard_frames,
        }
        if self._slm is not None:
            try:
                info["slm_wavelength_info"] = self._slm.get_wavelength_info()
            except Exception:  # 诊断项，拿不到不影响闭环
                pass
        return info

    def open(self) -> None:
        """打开并配置设备；应在 worker 线程里、任何其他调用之前执行一次。"""
        if self._slm is not None or self._camera is not None:
            raise RuntimeError("HardwareBench already opened")
        slm_factory = self._slm_factory
        if slm_factory is None:
            from ao_shaping.drivers.slm.santec import Santec

            slm_factory = Santec
        camera_factory = self._camera_factory
        if camera_factory is None:
            from ao_shaping.drivers.ccd.common import create_camera

            camera_factory = create_camera

        slm = slm_factory(
            slm_number=self.slm_number,
            wavelength=self.wavelength,
            video_mode=0,
        )
        camera = camera_factory(
            self.camera_type, cam_id=self.cam_id, exposure_time_ms=self.exposure_ms
        )
        self._slm = slm
        self._camera = camera
        try:
            slm.open()
            # 驱动板不应答时整段测量全无效 (#186)：必须在开相机、取第一帧之前拦下。
            # 仿真面板 / 测试替身没有 require_drive_board，duck-typed 视为 no-op。
            from ao_shaping.drivers.slm.factory import require_drive_board_entry

            require_drive_board_entry(slm)
            camera.open()
            # 覆盖相位绝不能把上一次的叠加相位带进闭环。
            slm.set_phase_overlay(None)
            effective = camera.reset_exposure_time(self.exposure_ms)
            self.exposure_ms = float(effective)
            from ao_shaping.drivers.slm.santec.driver import SlotRotator

            self._rotator = SlotRotator(slm)
            # 丢掉改曝光/开相机后的过渡帧，第一张有效帧才用于起点测量。
            self._discard()
        except Exception:
            self.close()
            raise

    def apply_phase(self, phase: np.ndarray) -> str:
        """把控制网格相位（rad）写进 SLM，并在返回前完成稳定等待。"""
        if self._slm is None:
            raise RuntimeError("HardwareBench.open() must be called first")
        from ao_shaping.utils.slm.transition_timeline import pixel_flip_wait_s

        phase = np.asarray(phase, dtype=np.float64)
        if phase.ndim != 2:
            raise ValueError(f"phase must be a 2-D control grid, got {phase.shape}")
        gray = self._slm.create_phase_from_array(phase)
        # 写入量按 uint16 灰度码估算：即使相邻两轮相位差很小也不省等待，
        # 因为真值要看上一轮实际下发的码（含 LUT/矫正），这里用当前帧极值近似。
        self._last_step_gray = float(np.ptp(gray)) if gray.size else 0.0
        settle = max(pixel_flip_wait_s(self._last_step_gray), self.settle_s)
        slot = self._rotator.next_slot()
        self._slm.display_data(gray, wait_time_s=0, memory_number=slot)
        self._sleep(settle)
        return f"slot-{slot}"

    def capture(self) -> CapturedFrame:
        """丢掉过渡帧后取一张新帧，附带可审计的帧号与曝光回读。"""
        if self._camera is None:
            raise RuntimeError("HardwareBench.open() must be called first")
        self._discard()
        image = np.asarray(self._camera.get_numpy_image(n_sample=1, skip_first=True))
        if image.ndim != 2:
            raise ValueError(f"expected a 2-D camera frame, got {image.shape}")
        exposure = float(self._camera.get_exposure_time_ms())
        full_scale = float(self._camera.effective_full_scale(image))
        self._frame_id += 1
        return CapturedFrame(
            frame=image,
            frame_id=self._frame_id,
            exposure_ms=exposure,
            full_scale=full_scale,
            monotonic_ns=time.monotonic_ns(),
        )

    def close(self) -> None:
        """尽力关闭两个句柄；可重复调用，不抛出。"""
        camera, self._camera = self._camera, None
        slm, self._slm = self._slm, None
        self._rotator = None
        for handle in (camera, slm):  # 先关相机，再关 SLM
            if handle is None:
                continue
            try:
                handle.close()
            except Exception:  # 关闭失败不该掩盖主结论
                pass

    # -- 内部 -------------------------------------------------------------- #

    def _discard(self) -> None:
        for _ in range(self.discard_frames):
            self._camera.get_numpy_image(n_sample=1, skip_first=True)
