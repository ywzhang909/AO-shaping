"""Backend tests for the defocus probe: one CLI, hardware **or** simulation.

What is pinned here
-------------------

The probe (``scripts/slm/hw_gs_defocus_probe.py``) is a *hardware* instrument
that must also run on the digital twin. The three things that can silently
break when a second backend is added are:

1. **the device contract** -- the probe drives the panel through
   ``set_phase_overlay`` (radians, added *before* radian→gray) and
   ``display_data``; a backend that does not implement them fails late, in the
   middle of a sweep;
2. **the recorded provenance** -- ``run_config.json`` / the plan name the
   driver that produced the numbers. The hardware strings are pinned by the
   artefacts already in the repository, so they must stay **verbatim**;
3. **the hardware call sequence** -- the probe's order of operations (flat
   frame → window → frozen zero-order → baseline → sweep) is what makes the
   numbers comparable with the earlier rounds. It is pinned call by call.

The tests use **fakes**, never a device: a recording panel and a camera that
renders a deterministic Gaussian spot. The simulation backend is exercised with
a fake that implements the twin's contract *exactly* (see
:class:`FakeSimSLM`), so the contract is checked here rather than assumed.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scripts.slm.hw_gs_defocus_probe import (
    HARDWARE_CCD_DRIVER,
    HARDWARE_SLM_DRIVER,
    SIM_CCD_DRIVER,
    SIM_SLM_DRIVER,
    build_optics,
    build_parser,
    build_plan,
    compose_overlay,
    device_driver_strings,
    overlay_api_text,
    panel_pupil_chirp,
    run_execute,
)

#: 面板几何 (与 ``bench_constants`` 一致: 1920x1200)。
PANEL_SHAPE = (1200, 1920)

#: 相机窗口 (与 ``--roi-px`` 的默认值一致)。
ROI_PX = 250

#: 假相机的整幅画面: 零级落在正中, 窗口是它周围的一块。
FRAME_SHAPE = (512, 512)
SPOT_XY = (256, 256)
SPOT_FWHM_PX = 12.0
SPOT_AMPLITUDE = 180.0
SPOT_BACKGROUND = 3.0


# ---------------------------------------------------------------------------
# 假面板 / 假相机
# ---------------------------------------------------------------------------
class RecordingSLM:
    """记录调用序列的假面板。

    灰度换算与真驱动一致 (弧度 → 2π=993 灰度), 所以 ``display_data`` 收到的是
    一张**真的** uint16 图, 而不是随手编的数 —— 这样「叠加层加在弧度→灰度之前」
    这条约定才有意义。
    """

    def __init__(self, panel_shape: tuple[int, int] = PANEL_SHAPE) -> None:
        self.panel_shape = tuple(panel_shape)
        self.calls: list[tuple[Any, ...]] = []
        #: 叠加层的**内容** (按调用顺序); ``calls`` 里只留形状, 好比较。
        self.overlays: list[np.ndarray | None] = []
        self.overlay: np.ndarray | None = None
        self._open = False

    # -- 生命周期 ---------------------------------------------------------
    def open(self) -> None:
        self.calls.append(("open",))
        self._open = True

    def close(self) -> None:
        self.calls.append(("close",))
        self._open = False

    def __enter__(self) -> "RecordingSLM":
        self.open()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- 相位 / 灰度 ------------------------------------------------------
    def create_phase_from_array(self, phase_rad: np.ndarray) -> np.ndarray:
        rad = np.asarray(phase_rad, dtype=np.float64)
        self.calls.append(("create_phase_from_array", tuple(rad.shape)))
        return (np.mod(rad, 2.0 * np.pi) / (2.0 * np.pi) * 993.0).astype(np.uint16)

    def display_data(self, gray: np.ndarray, *args: Any, **kwargs: Any) -> int:
        self.calls.append(
            (
                "display_data",
                tuple(args),
                tuple(sorted((str(k), v) for k, v in kwargs.items())),
            )
        )
        return int(kwargs.get("memory_number") or 0)

    def set_phase_overlay(self, phase_rad: np.ndarray | None) -> None:
        self.calls.append(
            ("set_phase_overlay", None if phase_rad is None else tuple(np.shape(phase_rad)))
        )
        self.overlays.append(
            None if phase_rad is None else np.asarray(phase_rad, dtype=np.float64)
        )
        self.overlay = (
            None if phase_rad is None else np.asarray(phase_rad, dtype=np.float64)
        )


class FakeSantecSLM(RecordingSLM):
    """硬件面板的替身: 多出内存槽那一套接口 (``preserve_slm_display`` 要用)。"""

    def get_displayed_memory_number(self) -> int:
        self.calls.append(("get_displayed_memory_number",))
        return 7

    def get_current_grayscale(self) -> int:
        self.calls.append(("get_current_grayscale",))
        return 511

    def _display_memory(self, slot: int) -> None:
        self.calls.append(("_display_memory", int(slot)))


class FakeSimSLM(RecordingSLM):
    """仿真孪生的替身 —— 签名**逐字**照孪生契约。

    契约 (``ao_shaping.drivers.sim.slm_pib_sim.SimSLMPib``)::

        display_data(gray, memory_number=None, memory_mode=None, wait_time_s=None)

    孪生**没有**内存槽接口 (``get_displayed_memory_number`` /
    ``get_current_grayscale`` / ``_display_memory``): 仿真面板不轮换槽位, 所以
    探针里那套「排除运行前正在显示的槽」的保护对它没有意义, 也不该被调用。
    """

    def display_data(  # 签名照孪生契约
        self,
        gray: np.ndarray,
        memory_number: int | None = None,
        memory_mode: int | None = None,
        wait_time_s: float | None = None,
    ) -> int:
        self.calls.append(
            ("display_data", memory_number, memory_mode, wait_time_s)
        )
        return int(memory_number or 0)


class FakeCamera:
    """假 CCD: 一张确定性的「高斯光斑 + 本底」画面。

    ``reset_window`` 真的裁画面 (硬件上 CCD 开窗), 所以 ROI 口径可以被断言。
    """

    def __init__(
        self,
        shape: tuple[int, int] = FRAME_SHAPE,
        spot_xy: tuple[int, int] = SPOT_XY,
    ) -> None:
        self.shape = tuple(shape)
        self.spot_xy = tuple(spot_xy)
        self.calls: list[tuple[Any, ...]] = []
        self.window: tuple[int, int, int] | None = None
        self.exposure_ms = 0.12
        self._frame = self._render()

    def _render(self) -> np.ndarray:
        h, w = self.shape
        yy, xx = np.mgrid[:h, :w]
        cy, cx = self.spot_xy[1], self.spot_xy[0]
        sigma = SPOT_FWHM_PX / (2.0 * math.sqrt(2.0 * math.log(2.0)))
        blob = SPOT_AMPLITUDE * np.exp(
            -(((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma**2))
        )
        return np.clip(blob + SPOT_BACKGROUND, 0.0, 255.0).astype(np.uint8)

    # -- 生命周期 ---------------------------------------------------------
    def open(self) -> None:
        self.calls.append(("open",))

    def close(self) -> None:
        self.calls.append(("close",))

    def __enter__(self) -> "FakeCamera":
        self.open()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- 采集 -------------------------------------------------------------
    def get_numpy_image(self, n_sample: int = 1, skip_first: bool = False) -> np.ndarray:
        self.calls.append(("get_numpy_image", int(n_sample)))
        frame = self._frame
        if self.window is not None:
            cx, cy, size = self.window
            x0 = max(int(cx) - size // 2, 0)
            y0 = max(int(cy) - size // 2, 0)
            frame = frame[y0 : y0 + size, x0 : x0 + size]
        return frame

    def reset_window(
        self, center: tuple[int, int], size: tuple[int, int]
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        self.calls.append(
            ("reset_window", (int(center[0]), int(center[1])), (int(size[0]), int(size[1])))
        )
        self.window = (int(center[0]), int(center[1]), int(size[0]))
        return ((int(size[0]), int(size[1])), (int(center[0]), int(center[1])))

    def reset_exposure_time(self, time_ms: float) -> float:
        self.calls.append(("reset_exposure_time", float(time_ms)))
        self.exposure_ms = float(time_ms)
        return float(time_ms)

    def get_exposure_time_ms(self) -> float:
        return float(self.exposure_ms)


# ---------------------------------------------------------------------------
# 跑一次探针 (假设备)
# ---------------------------------------------------------------------------
def _run_probe(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    cam_type: str = "daheng",
    slm: Any | None = None,
    dz_mm: tuple[str, ...] = ("0", "5"),
    extra: tuple[str, ...] = (),
) -> dict[str, Any]:
    """用假面板 + 假相机跑一次 ``run_execute``, 返回现场。"""
    from ao_shaping.drivers.ccd import common as ccd_common
    from ao_shaping.drivers.slm import factory as slm_factory

    panel = slm if slm is not None else FakeSantecSLM()
    camera = FakeCamera()
    created: dict[str, Any] = {}

    def _create_camera(camera_type, cam_id=0, exposure_time_ms=20.0, **kwargs):
        created["type"] = str(camera_type)
        created["kwargs"] = dict(kwargs)
        return camera

    monkeypatch.setattr(slm_factory, "open_slm", lambda params: panel)
    monkeypatch.setattr(ccd_common, "create_camera", _create_camera)
    # 曝光上限: 用环境变量给一个确定值, 于是探针不必跑那条「倍增阶梯找饱和」
    # (它要整帧**整数**图, 假相机满足不了, 而且它跟本文件要验的东西无关)。
    monkeypatch.setenv("AO_DAHENG_MAX_EXPOSURE_MS", "1.0")

    gs = tmp_path / "gs_phase.npy"
    np.save(gs, np.zeros((64, 64), dtype=np.float64))
    args = build_parser().parse_args(
        [
            "--execute",
            "--dz-mm",
            *dz_mm,
            "--frames",
            "1",
            "--settle-wait-s",
            "0",
            "--settle-max-wait-s",
            "0.2",
            "--gs-phase",
            str(gs),
            "--out-root",
            str(tmp_path / "root"),
            "--exposure-ms",
            "0.2",
            "--roi-px",
            str(ROI_PX),
            # 像元不再有默认值; 假相机没有型号可查, 所以显式给一个 (2.2 µm =
            # Daheng MER2-507-23GM, 与本文件假设备模拟的那台一致)。
            "--camera-pixel-um",
            "2.2",
            "--cam-type",
            cam_type,
            *extra,
        ]
    )
    optics = build_optics(args)
    plan = build_plan(args, optics)
    out_dir = run_execute(args, optics, plan)
    return {
        "slm": panel,
        "cam": camera,
        "args": args,
        "optics": optics,
        "plan": plan,
        "out_dir": out_dir,
        "camera_created": created,
    }


def _names(calls: list[tuple[Any, ...]]) -> list[str]:
    return [str(call[0]) for call in calls]


def _overlays(slm: Any) -> list[np.ndarray | None]:
    return list(slm.overlays)


# ---------------------------------------------------------------------------
# 驱动字符串: 硬件逐字不变, 仿真换一套措辞
# ---------------------------------------------------------------------------
class TestDriverStrings:
    def test_the_hardware_strings_are_verbatim(self):
        """第三轮已入库的产物里就是这两条 —— 逐字, 一个字符都不许动。"""
        assert HARDWARE_SLM_DRIVER == (
            "ao_shaping.drivers.slm.santec.Santec (memory 模式, video_mode=0)"
        )
        assert HARDWARE_CCD_DRIVER == (
            "ao_shaping.drivers.ccd.common.create_camera('daheng')"
        )
        assert device_driver_strings("daheng") == (
            HARDWARE_SLM_DRIVER,
            HARDWARE_CCD_DRIVER,
        )

    def test_the_simulation_strings_name_the_twin(self):
        assert device_driver_strings("sim") == (SIM_SLM_DRIVER, SIM_CCD_DRIVER)
        assert "SimSLMPib" in SIM_SLM_DRIVER
        assert "create_camera('sim')" in SIM_CCD_DRIVER

    @pytest.mark.parametrize(
        "cam_type, slm_driver, ccd_driver",
        [
            ("daheng", HARDWARE_SLM_DRIVER, HARDWARE_CCD_DRIVER),
            ("sim", SIM_SLM_DRIVER, SIM_CCD_DRIVER),
        ],
    )
    def test_the_plan_names_the_driver_of_the_backend(
        self, cam_type: str, slm_driver: str, ccd_driver: str
    ):
        args = build_parser().parse_args(
            # 像元不再有默认值: 读型号配置要 import 驱动包, 而本文件是离线测试。
            ["--cam-type", cam_type, "--camera-pixel-um", "2.2"]
        )
        plan = build_plan(args, build_optics(args))
        assert plan["devices"][0]["driver"] == slm_driver
        assert plan["devices"][1]["driver"] == ccd_driver

    def test_the_overlay_api_text_follows_the_backend(self):
        """叠加层的措辞必须说清「加在弧度→灰度之前」—— 离焦能不能出现全靠它。"""
        hardware = overlay_api_text("daheng")
        assert hardware.startswith("Santec.set_phase_overlay(phase_rad)")
        assert "弧度→灰度之前相加" in hardware
        simulation = overlay_api_text("sim")
        assert simulation.startswith("SimSLMPib.set_phase_overlay(phase_rad)")
        assert "弧度→灰度**之前**相加" in simulation
        assert "相加后" in simulation

    def test_the_cli_takes_the_backend_and_the_simulation_knobs(self):
        args = build_parser().parse_args(
            ["--cam-type", "sim", "--sim-seed", "7", "--sim-far-field-padding", "4"]
        )
        assert args.cam_type == "sim"
        assert args.sim_seed == 7
        assert args.sim_far_field_padding == 4

    def test_the_default_padding_matches_the_reference_sweep(self):
        """与 ``run_gs_refine_zernike_sweep.py`` 的同名选项同名同义。"""
        from scripts.slm.run_gs_refine_zernike_sweep import (
            DEFAULT_SIM_FAR_FIELD_PADDING,
        )

        args = build_parser().parse_args([])
        assert args.sim_far_field_padding == DEFAULT_SIM_FAR_FIELD_PADDING


# ---------------------------------------------------------------------------
# 硬件路径: 调用序列与产物
# ---------------------------------------------------------------------------
class TestHardwarePath:
    def test_the_panel_is_driven_through_the_hardware_contract(
        self, monkeypatch, tmp_path
    ):
        site = _run_probe(monkeypatch, tmp_path)
        slm = site["slm"]
        names = _names(slm.calls)
        assert names[0] == "open"
        assert "create_phase_from_array" in names
        assert "display_data" in names
        assert "set_phase_overlay" in names

    def test_the_overlay_has_the_panel_shape_and_is_cleared_at_the_end(
        self, monkeypatch, tmp_path
    ):
        site = _run_probe(monkeypatch, tmp_path)
        overlays = _overlays(site["slm"])
        assert overlays, "探针必须给面板叠加瞳面相位"
        # 本轮没给焦面矫正 (--defocus-focal-mm), 所以前面几次是「清空」;
        # 扫描点上的叠加层是面板尺寸的瞳面相位, 收尾必须再清一次。
        for overlay in overlays:
            if overlay is None:
                continue
            assert overlay.shape == PANEL_SHAPE
        assert overlays[-1] is None, "结束必须清掉叠加层 (它是驱动的持久状态)"

    def test_the_sweep_overlay_is_the_focus_panel_plus_the_chirp(
        self, monkeypatch, tmp_path
    ):
        """叠加层 = 焦面矫正 + 二次相位 (+ 去倾斜斜坡), 相加进**同一张**图。"""
        site = _run_probe(monkeypatch, tmp_path)
        args = site["args"]
        optics = site["optics"]
        overlays = [o for o in _overlays(site["slm"]) if o is not None]
        chirp = panel_pupil_chirp(
            5e-3, optics, sign=args.chirp_sign, alpha=args.chirp_alpha
        )
        expected = compose_overlay(None, chirp)
        assert expected is not None
        assert any(
            overlay.shape == expected.shape
            and np.allclose(overlay, expected, atol=1e-9)
            for overlay in overlays
        ), "Δz=5 mm 的叠加层必须正好是那一条二次相位"

    def test_the_hardware_call_sequence_is_pinned(
        self, monkeypatch, tmp_path
    ):
        """硬件路径的**调用顺序**逐字钉住 —— 顺序变了, 数字就不可比了。

        两次 ``open`` 是脚本本身的历史写法 (``with ... as slm`` 会开一次, 里面
        又显式 ``slm.open()`` 一次, 见脚本里那段注释), 不是笔误。
        """
        site = _run_probe(monkeypatch, tmp_path)
        assert _names(site["slm"].calls) == [
            # 打开面板 (脚本的历史写法: ``with ... as slm`` 开一次, 里面再显式
            # ``slm.open()`` 一次 —— 见脚本里那段注释)。
            "open",
            "open",
            # 「排除运行前正在显示的槽」: preserve_slm_display 一次 + 探针自己一次。
            "get_displayed_memory_number",
            "get_displayed_memory_number",
            # 定位零级: 平相位 → 抓全幅 → (相机开窗) → 抓 ROI。
            "set_phase_overlay",
            "create_phase_from_array",
            "display_data",
            "create_phase_from_array",
            "display_data",
            # 基线帧 (GS 相位已施加)。
            "set_phase_overlay",
            "create_phase_from_array",
            "display_data",
            # 扫描: 每个 Δz 一次「叠加层 + 显示」。
            "set_phase_overlay",
            "create_phase_from_array",
            "display_data",
            "set_phase_overlay",
            "create_phase_from_array",
            "display_data",
            # 收尾: 清叠加层 → 恢复原显示 → 关面板。
            "set_phase_overlay",
            "_display_memory",
            "close",
        ]

    def test_the_run_config_records_the_hardware_driver_verbatim(
        self, monkeypatch, tmp_path
    ):
        site = _run_probe(monkeypatch, tmp_path)
        config = json.loads(
            (site["out_dir"] / "run_config.json").read_text(encoding="utf-8")
        )
        assert config["plan"]["devices"][0]["driver"] == HARDWARE_SLM_DRIVER
        assert config["plan"]["devices"][1]["driver"] == HARDWARE_CCD_DRIVER
        assert config["cam_type"] == "daheng"
        # 硬件那轮的产物字段集不许因为「加了仿真分支」而变。
        assert "backend" not in config
        assert "sim_seed" not in config

    def test_the_window_and_the_frozen_zero_order_are_recorded(
        self, monkeypatch, tmp_path
    ):
        site = _run_probe(monkeypatch, tmp_path)
        config = json.loads(
            (site["out_dir"] / "run_config.json").read_text(encoding="utf-8")
        )
        windows = [c for c in site["cam"].calls if c[0] == "reset_window"]
        assert len(windows) == 1, "窗口只开一次, 之后全程冻结"
        assert windows[0][2] == (ROI_PX, ROI_PX)
        # 窗口开在**全幅**零级上; ROI 里的零级是开窗之后重新定位并冻结的。
        full_x, full_y = config["zero_order_full_frame_xy"]
        assert windows[0][1] == (int(full_x), int(full_y))
        cx, cy = config["zero_order_roi_xy_frozen"]
        assert (int(cx), int(cy)) == (ROI_PX // 2, ROI_PX // 2)
        assert config["roi_px"] == ROI_PX

    def test_the_measurements_are_written_and_finite(
        self, monkeypatch, tmp_path
    ):
        site = _run_probe(monkeypatch, tmp_path)
        assert (site["out_dir"] / "measurements.csv").is_file()
        assert (site["out_dir"] / "measurements.json").is_file()
        rows = json.loads(
            (site["out_dir"] / "measurements.json").read_text(encoding="utf-8")
        )
        # 基线行 + 两个 Δz 点。
        assert [row["dz_mm"] for row in rows] == [0.0, 0.0, 5.0]
        for row in rows:
            for key in ("cv", "pib", "score", "frame_peak", "spot_fwhm_px"):
                assert math.isfinite(float(row[key])), (key, row)


# ---------------------------------------------------------------------------
# 仿真路径: 同一套 CLI, 同一结构的产物
# ---------------------------------------------------------------------------
class TestSimulationPath:
    def test_the_same_cli_runs_on_the_twin_contract(
        self, monkeypatch, tmp_path
    ):
        """假面板按孪生契约实现 —— 探针在仿真模式下必须能跑完。"""
        site = _run_probe(
            monkeypatch, tmp_path, cam_type="sim", slm=FakeSimSLM()
        )
        rows = json.loads(
            (site["out_dir"] / "measurements.json").read_text(encoding="utf-8")
        )
        assert [row["dz_mm"] for row in rows] == [0.0, 0.0, 5.0]
        for row in rows:
            assert math.isfinite(float(row["score"]))

    def test_the_twin_never_sees_a_memory_slot_call(
        self, monkeypatch, tmp_path
    ):
        """孪生没有内存槽接口 —— 探针不许去调它 (否则就是 AttributeError)。"""
        site = _run_probe(
            monkeypatch, tmp_path, cam_type="sim", slm=FakeSimSLM()
        )
        names = _names(site["slm"].calls)
        assert "get_displayed_memory_number" not in names
        assert "_display_memory" not in names
        assert "get_current_grayscale" not in names

    def test_the_camera_window_matches_the_roi_in_simulation(
        self, monkeypatch, tmp_path
    ):
        """仿真相机按 ``cam_size`` 裁远场 ⇒ 它必须拿到与 ROI 同样的边长。"""
        site = _run_probe(
            monkeypatch, tmp_path, cam_type="sim", slm=FakeSimSLM()
        )
        assert site["camera_created"]["type"] == "sim"
        assert site["camera_created"]["kwargs"]["cam_size"] == ROI_PX

    def test_the_hardware_camera_gets_no_extra_kwargs(
        self, monkeypatch, tmp_path
    ):
        """硬件相机的构造调用**逐字不变** (大恒不接受 ``cam_size``)。"""
        site = _run_probe(monkeypatch, tmp_path)
        assert site["camera_created"]["type"] == "daheng"
        assert site["camera_created"]["kwargs"] == {}

    def test_the_run_config_marks_the_simulation(
        self, monkeypatch, tmp_path
    ):
        site = _run_probe(
            monkeypatch, tmp_path, cam_type="sim", slm=FakeSimSLM()
        )
        config = json.loads(
            (site["out_dir"] / "run_config.json").read_text(encoding="utf-8")
        )
        assert config["backend"] == "sim"
        assert config["cam_type"] == "sim"
        assert config["plan"]["devices"][0]["driver"] == SIM_SLM_DRIVER
        assert config["plan"]["devices"][1]["driver"] == SIM_CCD_DRIVER
        # 复现所需的仿真参数必须落盘。
        assert config["sim_far_field_padding"] == 16
        assert config["sim_seed"] == 0

    def test_the_simulation_exposure_is_not_probed_for_saturation(
        self, monkeypatch, tmp_path
    ):
        """仿真没有饱和概念 ⇒ 不跑那条「倍增阶梯」(它会白烧远场 FFT)。"""
        site = _run_probe(
            monkeypatch, tmp_path, cam_type="sim", slm=FakeSimSLM()
        )
        resets = [c for c in site["cam"].calls if c[0] == "reset_exposure_time"]
        assert len(resets) == 1, "仿真只写一次曝光, 不做阶梯探测"
        assert resets[0][1] == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# 端到端: 真孪生 (契约落地后自动生效)
# ---------------------------------------------------------------------------
def _twin_overlay_ready() -> bool:
    """仿真孪生的 ``set_phase_overlay`` 契约落地了没有。"""
    try:
        from ao_shaping.drivers.sim import slm_pib_sim
    except ImportError:  # pragma: no cover - 孪生包不可导入
        return False
    return hasattr(slm_pib_sim.SimSLMPib, "set_phase_overlay")


@pytest.mark.skipif(
    not _twin_overlay_ready(),
    reason=(
        "仿真孪生的 set_phase_overlay / display_data(wait_time_s) 契约尚未落地 "
        "(并行 lane); 落地后本测试自动生效"
    ),
)
class TestRealTwinEndToEnd:
    """真孪生上跑一遍: 产物齐、数值有限、离焦响应真的出现。

    代价: 一次远场 FFT ≈ 1.2 s (补零 2), 整轮 ≈ 6 s。
    """

    def test_the_probe_produces_a_full_artefact_set(self, tmp_path):
        gs = tmp_path / "gs_phase.npy"
        np.save(gs, np.zeros((64, 64), dtype=np.float64))
        args = build_parser().parse_args(
            [
                "--execute",
                "--cam-type",
                "sim",
                "--dz-mm",
                "0",
                "5",
                "--frames",
                "1",
                "--settle-wait-s",
                "0",
                "--settle-max-wait-s",
                "0.2",
                "--gs-phase",
                str(gs),
                "--out-root",
                str(tmp_path / "root"),
                "--sim-far-field-padding",
                "2",
            ]
        )
        optics = build_optics(args)
        plan = build_plan(args, optics)
        out_dir = run_execute(args, optics, plan)

        assert (out_dir / "run_config.json").is_file()
        assert (out_dir / "measurements.csv").is_file()
        rows = json.loads((out_dir / "measurements.json").read_text(encoding="utf-8"))
        assert len(rows) >= 2
        for row in rows:
            for key in ("cv", "pib", "score", "frame_peak", "spot_fwhm_px"):
                assert math.isfinite(float(row[key])), (key, row)

        # 离焦响应必须真的出现: Δz 越大光斑越糊 (框是冻结的, 所以功率会掉)。
        baseline = rows[0]
        deflected = max(rows[1:], key=lambda row: float(row["dz_mm"]))
        assert float(deflected["spot_fwhm_px"]) > float(baseline["spot_fwhm_px"])
