"""冻结 ROI: 续跑块必须把 ROI 窗口摆在**同一个**位置。

2026-10-21 实测的分块续跑失败有两层, 两层都要钉住:

1. **框中心用错了坐标系** —— 目标框中心用的是**全幅**传感器坐标, 而评分用的帧是
   相机裁出来的 ROI 窗口 (256x256)。全幅坐标 (例如 (1434, 973)) 落在窗口之外,
   框整个跑到画面外, 报错只有一句 "target square ... is empty"。
2. **窗口会平移** —— 续跑块若**重新定位**零级, 落点可能是一个近等的散斑颗粒,
   ROI 窗口随之平移一两像素; 窗口坐标系整个挪动, 上一块冻结的框就指到了别处,
   块边界上的指标不可比。

这里用一套假的相机/SLM 把第 2 条钉住: 全幅画面上故意放一颗**更亮**的散斑颗粒 ——
重新定位就会落到它身上; 续跑状态里带了**全幅**零级位置时, 窗口必须照它摆。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from ao_shaping.optimizer.wfless.slm_gs_refine import (
    SlmGsRefineConfig,
    optimize_slm_gs_refine,
    resume_state_from_recorder,
)

#: 假传感器的画面尺寸 ``(h, w)``。
SENSOR_SHAPE = (480, 640)

#: 真正的零级 (全幅坐标 ``(x, y)``) —— ROI 窗口应当以它为中心。
SPOT_XY = (300, 200)

#: 一颗**更亮**的散斑颗粒: 重新定位零级就会落到它身上 (窗口因此挪走)。
DECOY_XY = (450, 300)

#: 数字孪生缩到的面板尺寸 (h, w)。
SMALL_PANEL = (120, 192)

#: ROI 窗口边长 (像素)。
CAM_SIZE = 256


class _FakeCamera:
    """一台会记录窗口摆放位置的假相机。

    全幅画面上有两颗光斑: 真正的零级 (:data:`SPOT_XY`) 与一颗更亮的散斑
    (:data:`DECOY_XY`)。开窗之后画面就是那个窗口的裁剪。
    """

    def __init__(self) -> None:
        self.window_calls: list[tuple[tuple[int, int], tuple[int, int]]] = []
        self._window: tuple[int, int, int, int] | None = None

    # --- 设备生命周期 -----------------------------------------------------
    def __enter__(self) -> _FakeCamera:
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False

    def open(self) -> None:
        return None

    def close(self) -> None:
        return None

    # --- 采集 --------------------------------------------------------------
    def reset_exposure_time(self, time_ms: float) -> float:
        return float(time_ms)

    def get_exposure_time_ms(self) -> float:
        return 1.0

    def reset_window(
        self,
        center: tuple[int, int],
        size: tuple[int, int],
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        cx, cy = int(center[0]), int(center[1])
        width, height = int(size[0]), int(size[1])
        self.window_calls.append(((cx, cy), (width, height)))
        self._window = (cx - width // 2, cy - height // 2, width, height)
        return (width, height), (cx, cy)

    def get_numpy_image(self, n_sample: int = 1, skip_first: bool = True) -> np.ndarray:
        return self._render()

    def _render(self) -> np.ndarray:
        if self._window is None:
            x0, y0 = 0, 0
            height, width = SENSOR_SHAPE
        else:
            x0, y0, width, height = self._window
        yy, xx = np.mgrid[0:height, 0:width]
        frame = np.zeros((height, width), dtype=np.float64)
        for (px, py), peak in ((SPOT_XY, 60.0), (DECOY_XY, 200.0)):
            frame += peak * np.exp(
                -((xx - (px - x0)) ** 2 + (yy - (py - y0)) ** 2) / (2 * 2.0**2)
            )
        return frame

    def windowed_frame(self, center: tuple[int, int]) -> np.ndarray:
        """按 ``center`` 开窗并返回那一帧 (窗口坐标系)。"""
        self.reset_window(center=center, size=(CAM_SIZE, CAM_SIZE))
        return self._render()


class _FakeSLM:
    """一块什么都不做的假面板。"""

    Panel_Res = (SMALL_PANEL[1], SMALL_PANEL[0])
    wavelength = 1064

    def __enter__(self) -> _FakeSLM:
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False

    def open(self) -> None:
        return None

    def close(self) -> None:
        return None

    def create_phase_from_array(self, phase_rad: np.ndarray) -> np.ndarray:
        return np.zeros(SMALL_PANEL, dtype=np.uint16)

    def display_data(self, gray: np.ndarray, memory_number: int | None = None) -> None:
        return None

    def get_displayed_memory_number(self) -> int:
        return 5

    def _display_memory(self, slot: int) -> None:
        return None


@pytest.fixture
def fake_hardware(monkeypatch):
    """把优化器接到假相机 / 假面板上 (不碰任何硬件)。"""
    from ao_shaping.optimizer.wfless import slm_gs_refine as gsr

    camera = _FakeCamera()
    slm = _FakeSLM()
    monkeypatch.setattr(gsr, "open_camera", lambda config: camera)
    monkeypatch.setattr(gsr, "open_slm", lambda config: slm)
    monkeypatch.setattr(gsr, "panel_shape_from_slm", lambda handle: SMALL_PANEL)
    return camera, slm


def _config(**overrides: Any) -> SlmGsRefineConfig:
    base = dict(
        # 非 sim ⇒ 走硬件路径 (零级定位 + ROI 窗口摆放那段)。
        cam_type="miicam",
        epochs=1,
        chunk_epochs=1,
        phase_grid=4,
        delta=0.2,
        lr=0.05,
        seed=3,
        beam_radius_px=24.0,
        cam_size=CAM_SIZE,
        target_side=12,
        gs_iters=4,
        n_eval_frames=1,
        settle_wait_s=0.001,
        settle_tol=0.02,
        settle_max_wait_s=0.001,
        settle_max_discard=1,
        progress_every=1,
        record_debug=True,
    )
    base.update(overrides)
    return SlmGsRefineConfig(**base)


def _resume_state(camera: _FakeCamera, *, full_center: tuple[int, int] | None):
    """按 ``full_center`` 摆好窗口, 造一份「上一块末态」的续跑状态。"""
    center = full_center if full_center is not None else SPOT_XY
    flat = camera.windowed_frame(center)
    return {
        "base_panel": np.zeros(SMALL_PANEL, dtype=np.float64),
        "dof": np.zeros(16, dtype=np.float64),
        "flat_frame": flat,
        "center_y": CAM_SIZE // 2,
        "center_x": CAM_SIZE // 2,
        "full_frame_center": full_center,
        "best_score": 0.5,
        "best_phase": np.zeros(SMALL_PANEL, dtype=np.float64),
        "best_frame": flat,
        "best_stage": "flat",
        "epoch_offset": 2,
    }


class TestFrozenRoi:
    def test_resume_places_the_roi_at_the_recorded_full_frame_centre(
        self, fake_hardware
    ):
        """续跑状态带了全幅零级位置 ⇒ 窗口必须照它摆 (不重新定位)。"""
        camera, _ = fake_hardware
        state = _resume_state(camera, full_center=SPOT_XY)

        rec = optimize_slm_gs_refine(_config(resume_state=state))

        assert camera.window_calls, "续跑块根本没有摆 ROI 窗口"
        placed, size = camera.window_calls[0]
        assert placed == SPOT_XY, (
            f"ROI 窗口摆在 {placed}, 不是记录的全幅零级 {SPOT_XY} —— "
            f"重新定位零级会让窗口坐标系平移"
        )
        assert size == (CAM_SIZE, CAM_SIZE)
        # 记录器也要把全幅中心带出去, 下一块才能继续沿用。
        assert rec.full_frame_center == SPOT_XY

    def test_resume_does_not_follow_the_brighter_speckle(self, fake_hardware):
        """全幅画面上更亮的散斑不该把窗口带走 —— 重新定位零级正是这个风险。"""
        camera, _ = fake_hardware
        state = _resume_state(camera, full_center=SPOT_XY)

        optimize_slm_gs_refine(_config(resume_state=state))

        placed, _ = camera.window_calls[0]
        assert placed != DECOY_XY

    def test_fresh_run_still_locates_the_zero_order_on_the_full_frame(
        self, fake_hardware
    ):
        """没有记录值时 (首块 / 仿真路径) 仍然重新定位零级 —— 行为不变。"""
        camera, _ = fake_hardware

        rec = optimize_slm_gs_refine(_config())

        placed, _ = camera.window_calls[0]
        # 全幅画面上最亮的是散斑颗粒 ⇒ 重新定位会落在它身上 (旧行为)。
        assert placed == DECOY_XY
        assert rec.full_frame_center == DECOY_XY

    def test_resume_state_from_the_recorder_carries_the_full_frame_centre(
        self, fake_hardware
    ):
        """记录器 → 续跑状态 这一步也要带上全幅零级位置。"""
        camera, _ = fake_hardware
        state = _resume_state(camera, full_center=SPOT_XY)
        rec = optimize_slm_gs_refine(_config(resume_state=state))

        flat = np.asarray(
            next(r for r in rec.history if r["stage"] == "refine")["_img"],
            dtype=np.float64,
        )
        packed = resume_state_from_recorder(
            rec, flat_frame=flat, center_y=rec.center_y, center_x=rec.center_x
        )
        assert packed["full_frame_center"] == SPOT_XY
        # 框中心仍然是**窗口内**坐标。
        assert packed["center_x"] == CAM_SIZE // 2
        assert packed["center_y"] == CAM_SIZE // 2
