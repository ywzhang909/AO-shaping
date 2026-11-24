"""单会话复用 (2026-11-24): runner 侧 ``--reuse-device-session`` 的离线锁定。

背景: ab_session_20261124 耗时分析定位到逐 (shape, stage) 段各自
open/probe/close camera+SLM, 8 段固定开销 666 s (段耗时 41%)。方案 A
(用户选定) 让 runner 以 ``shared_device_session`` 打开一次设备、经
``validate.run(args, session=...)`` 跨段复用; 默认路径 (不传 session)
行为与历史逐字节一致。本文件锁三件事:

1. 默认关闭: CLI flag / ``execute_campaign`` / ``run_stages`` 的缺省
   都不开共享会话, ``validate.run`` 的 ``session`` 缺省 None;
2. 复用路径的设备生命周期: 一次 open, 段间不 close, 位深经驱动契约
   换档, 曝光/探针结果跨段携带;
3. 全 resume 时不白开设备 (与 run_stages 的复用谓词一致)。
"""
from __future__ import annotations

import contextlib
import inspect
from pathlib import Path

from click.testing import CliRunner

import ao_shaping.drivers.ccd.common as ccd_common
import ao_shaping.drivers.slm.santec as santec_mod
import ao_shaping.runners.slm.transfer_shapes_runner as runner_mod
import ao_shaping.tools.slm.transfer.hardware as hardware_mod
from ao_shaping.tools.slm.transfer import validate

REPO = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# 1. 默认关闭 —— 不传 session 的老调用形态逐字节不变
# ---------------------------------------------------------------------------

def test_run_session_defaults_to_none():
    """``validate.run`` 的 ``session`` 缺省必须是 None (老调用行为不变)。"""
    signature = inspect.signature(validate.run)
    assert signature.parameters["session"].default is None


def test_runner_flag_defaults_off():
    """CLI ``--reuse-device-session`` 缺省 False: 默认路径不开共享会话。"""
    param = next(p for p in runner_mod.run.params if p.name == "reuse_device_session")
    assert param.default is False


def test_run_stages_keeps_the_single_arg_call_form_without_session():
    """session=None 时 ``run_stages`` 仍以单参数调 ``validate_stage.run``。

    GUI/测试对 ``validate.run`` 的 monkeypatch 多为单参数 lambda;
    只有共享会话才追加关键字, 保证这些调用点不被迫跟随改动。
    """
    source = inspect.getsource(runner_mod.run_stages)
    assert "if session is None:" in source
    assert "validate_stage.run(args, session=session)" in source


# ---------------------------------------------------------------------------
# 2. 共享会话的设备生命周期 (假设备, 不碰真驱动)
# ---------------------------------------------------------------------------

class _FakeCam:
    def __init__(self) -> None:
        self.bit_depth: int | None = None
        self.closed = False

    def set_bit_depth(self, depth: int) -> None:
        self.bit_depth = depth

    def close(self) -> None:
        self.closed = True


class _FakeSlm:
    def __init__(self) -> None:
        self.overlay = "unset"
        self.closed = False

    def set_phase_overlay(self, value) -> None:
        self.overlay = value

    def close(self) -> None:
        self.closed = True


def _patch_devices(monkeypatch, opens: list, closes: list):
    cams: list[_FakeCam] = []
    slms: list[_FakeSlm] = []

    def fake_create_camera(camera, cam_id, exposure_time_ms=None, **options):
        opens.append(("camera", camera, cam_id, exposure_time_ms, options))
        cam = _FakeCam()
        cam.bit_depth = options.get("bit_depth")
        cams.append(cam)
        return cam

    def fake_santec(**kwargs):
        opens.append(("slm", kwargs))
        slm = _FakeSlm()
        slms.append(slm)
        return slm

    @contextlib.contextmanager
    def fake_opened(device):
        try:
            yield device
        finally:
            closes.append(type(device).__name__)
            device.close()

    monkeypatch.setattr(ccd_common, "create_camera", fake_create_camera)
    monkeypatch.setattr(santec_mod, "Santec", fake_santec)
    monkeypatch.setattr(hardware_mod, "opened_device", fake_opened)
    monkeypatch.setattr(hardware_mod, "prepare_camera_sdk", lambda camera: None)
    return cams, slms


def test_shared_device_session_opens_each_device_once(monkeypatch):
    """一次 ``shared_device_session`` = 恰好一次 camera open + 一次 SLM open。

    摄像机按首段位深构造 (daheng 才带 bit_depth 选项); 退出时关闭一次。
    """
    opens: list = []
    closes: list = []
    cams, slms = _patch_devices(monkeypatch, opens, closes)

    with validate.shared_device_session(
            camera="daheng", cam_id=3, slm_number=1, bit_depth=12) as session:
        assert len(opens) == 2
        assert opens[0] == ("camera", "daheng", 3, .02, {"bit_depth": 12})
        assert session.cam is cams[0]
        assert session.slm is slms[0]
        assert cams[0].bit_depth == 12
        assert slms[0].overlay is None  # 构造期清 overlay
    assert closes == ["_FakeSlm", "_FakeCam"]  # 内层 SLM 先退出
    assert cams[0].closed and slms[0].closed


def test_shared_device_session_skips_bit_depth_for_non_daheng(monkeypatch):
    """非 daheng 相机不带位深档: 构造期不下发 bit_depth 选项。"""
    opens: list = []
    closes: list = []
    _patch_devices(monkeypatch, opens, closes)

    with validate.shared_device_session(
            camera="miicam", cam_id=0, slm_number=1, bit_depth=12):
        assert opens[0] == ("camera", "miicam", 0, .02, {})


def test_switch_shared_bit_depth_only_touches_daheng_when_depth_changes():
    """段间位深换档: 同位深不动, 跨位深走 set_bit_depth, 非 daheng 原样通过。"""
    cam = _FakeCam()
    cam.bit_depth = 12
    session = validate.SharedDeviceSession(cam=cam, slm=None)

    validate._switch_shared_bit_depth(session, camera="daheng", bit_depth=12)
    assert cam.bit_depth == 12  # 同位深: 未触发 set_bit_depth
    calls = []
    cam.set_bit_depth = lambda depth: calls.append(depth)
    validate._switch_shared_bit_depth(session, camera="daheng", bit_depth=8)
    assert calls == [8]

    cam2 = _FakeCam()
    session2 = validate.SharedDeviceSession(cam=cam2, slm=None)
    validate._switch_shared_bit_depth(session2, camera="miicam", bit_depth=8)
    assert cam2.bit_depth is None  # 非 daheng: 不改


def test_run_validate_session_borrows_and_writes_back():
    """复用段的源级契约: 借用不重开 (nullcontext), 段尾回写曝光/位深。"""
    source = inspect.getsource(validate._run_validate_session)
    assert "contextlib.nullcontext(session.cam)" in source
    assert "contextlib.nullcontext(session.slm)" in source
    assert "_switch_shared_bit_depth(session" in source
    # 同位深才携带上一段曝光作阶梯起点
    assert "session.bit_depth == bit_depth" in source
    # 段尾写回: 供同位深的下一段作起点
    assert "session.exposure_ms = float(get_camera_exposure_ms(cam))" in source
    # 探针缓存: 会话首段之后不再重跑 probe_environment
    assert "session.probe_results" in source


# ---------------------------------------------------------------------------
# 3. execute_campaign: 共享会话只在有待跑段时开
# ---------------------------------------------------------------------------

def test_execute_campaign_pins_the_reuse_gate():
    """源级契约: flag 关 = nullcontext(None); 开 = 先查待跑段再开。"""
    source = inspect.getsource(runner_mod.execute_campaign)
    assert 'values.get("reuse_device_session", False)' in source
    assert "contextlib.nullcontext(None)" in source
    assert "validate_stage.shared_device_session(" in source
    # 待跑段判定必须复用 run_stages 的同一条 resume 谓词, 两处不许漂移。
    assert "stage_complete(out / \"shape\" / shape / STAGE_RECIPES[idx].name" in source


def test_plan_only_still_opens_no_device(tmp_path):
    """计划模式 (无 --execute) 不得打开设备, 即便带 --reuse-device-session。"""
    result = CliRunner().invoke(
        runner_mod.run,
        ["--out", str(tmp_path / "session"), "--shapes", "square",
         "--reuse-device-session"],
        catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert "未指定 --execute" in result.output
    assert not (tmp_path / "session").exists()
