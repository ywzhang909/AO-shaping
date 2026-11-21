"""``diagnostics_tab`` (设备诊断尾 tab) 的离线单测。

全部离线 —— 永不打开 SLM/CCD/激光, 也永不启动 streamlit 实验进程: ``st`` 换成最小
记录替身, ``subprocess.run`` / ``with_slm_ownership`` 只在需要时用记录替身替换。

覆盖:
* 两个命令 builder 的参数顺序与 flag (``separation_px=0`` 不传、phase ``--pattern
  blaze``、``--out-dir`` / ``--tag`` 唯一);
* 互斥门闩 + ack gate 的纯逻辑 (:func:`run_blocked`);
* 渲染时的按钮 disabled 组合, 以及空状态下不碰硬件;
* ``multi_slm_controller.main()`` 的 tab 标签顺序 (mock ``st.tabs`` 记录);
* 两条渲染纪律 (无 ``st.divider()``、无 ``type="primary"``)。
"""

from __future__ import annotations

import ast
from contextlib import nullcontext
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from ao_shaping.gui.slm import diagnostics_tab as diag
from ao_shaping.gui.slm import multi_slm_controller as mc

SOURCE = Path(diag.__file__)


# ---------------------------------------------------------------------------
# 记录式 streamlit 替身
# ---------------------------------------------------------------------------


class _Ctx:
    """可 ``with`` 的占位 (column / expander / 空的 tab)。"""

    def __init__(self) -> None:
        self.log: list[str] | None = None

    def __enter__(self) -> _Ctx:
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False


class _St:
    """只记录调用与控件值的 streamlit 替身 (够驱动诊断页全部渲染分支)。"""

    def __init__(
        self,
        session_state: dict[str, Any] | None = None,
        *,
        buttons: set[str] | None = None,
        checkboxes: dict[str, bool] | None = None,
    ) -> None:
        self.session_state = session_state if session_state is not None else {}
        self.subheaders: list[str] = []
        self.captions: list[str] = []
        self.warnings: list[str] = []
        self.errors: list[str] = []
        self.successes: list[str] = []
        self.buttons: list[dict[str, Any]] = []
        self.checkboxes_seen: list[dict[str, Any]] = []
        self.code_blocks: list[str] = []
        self.images: list[str] = []
        self.jsons: list[Any] = []
        self._button_hits = buttons or set()
        self._checkbox_values = checkboxes or {}

    # -- 上下文 ---------------------------------------------------------
    def spinner(self, *args: Any, **kwargs: Any) -> _Ctx:
        return _Ctx()

    def columns(self, count: int) -> list[_Ctx]:
        return [_Ctx() for _ in range(int(count))]

    def expander(self, *args: Any, **kwargs: Any) -> _Ctx:
        return _Ctx()

    # -- 文本 -----------------------------------------------------------
    def subheader(self, body: str, **kwargs: Any) -> None:
        self.subheaders.append(str(body))

    def caption(self, body: str, **kwargs: Any) -> None:
        self.captions.append(str(body))

    def warning(self, body: str, **kwargs: Any) -> None:
        self.warnings.append(str(body))

    def error(self, body: str, **kwargs: Any) -> None:
        self.errors.append(str(body))

    def success(self, body: str, **kwargs: Any) -> None:
        self.successes.append(str(body))

    def code(self, body: str, **kwargs: Any) -> None:
        self.code_blocks.append(str(body))

    def image(self, body: Any, **kwargs: Any) -> None:
        self.images.append(str(body))

    def json(self, body: Any, **kwargs: Any) -> None:
        self.jsons.append(body)

    # -- 控件 -----------------------------------------------------------
    def number_input(self, label: str, **kwargs: Any) -> Any:
        key = kwargs.get("key")
        if key and key in self.session_state:
            return self.session_state[key]
        return kwargs.get("value", 0)

    def selectbox(self, label: str, options: Any, **kwargs: Any) -> Any:
        key = kwargs.get("key")
        if key and key in self.session_state:
            return self.session_state[key]
        index = int(kwargs.get("index", 0) or 0)
        return options[index]

    def text_input(self, label: str, **kwargs: Any) -> str:
        key = kwargs.get("key")
        if key and key in self.session_state:
            return str(self.session_state[key])
        return str(kwargs.get("value", ""))

    def checkbox(self, label: str, **kwargs: Any) -> bool:
        key = kwargs.get("key")
        self.checkboxes_seen.append({"label": str(label), "key": key})
        if key and key in self._checkbox_values:
            return bool(self._checkbox_values[key])
        return bool(kwargs.get("value", False))

    def button(self, label: str, **kwargs: Any) -> bool:
        key = kwargs.get("key")
        self.buttons.append(
            {"label": str(label), "key": key, "disabled": kwargs.get("disabled")}
        )
        return bool(key in self._button_hits)


def _install(monkeypatch: pytest.MonkeyPatch, stub: _St) -> _St:
    for name in (
        "spinner",
        "columns",
        "expander",
        "subheader",
        "caption",
        "warning",
        "error",
        "success",
        "code",
        "image",
        "json",
        "number_input",
        "selectbox",
        "text_input",
        "checkbox",
        "button",
    ):
        monkeypatch.setattr(diag.st, name, getattr(stub, name))
    monkeypatch.setattr(diag.st, "session_state", stub.session_state)
    return stub


# ---------------------------------------------------------------------------
# 命令拼装
# ---------------------------------------------------------------------------


def test_diffraction_command_flag_order_and_values() -> None:
    command = diag.build_diffraction_command(
        out_dir="data/diagnostics/diffraction/stamp",
        slm_number=2,
        slm_wavelength=1064,
        cam_type="daheng",
        cam_id=3,
        exposure_ms=5.0,
        gray_depth=0,
        period_px=16,
        axis="x",
        separation_px=471.0,
        lobe_axis="y",
        frames=3,
        settle_s=0.25,
        save_frames=False,
        subtract_dark=True,
    )
    assert command[1:4] == ["-m", diag.DIFFRACTION_MODULE, "probe"]
    assert command[4:] == [
        "--out-dir",
        "data/diagnostics/diffraction/stamp",
        "--slm-number",
        "2",
        "--slm-wavelength",
        "1064",
        "--cam-type",
        "daheng",
        "--cam-id",
        "3",
        "--exposure-ms",
        "5.0",
        "--gray-depth",
        "0",
        "--period-px",
        "16",
        "--axis",
        "x",
        "--separation-px",
        "471.0",
        "--lobe-axis",
        "y",
        "--frames",
        "3",
        "--settle-s",
        "0.25",
        "--no-save-frames",
        "--subtract-dark",
    ]


def test_diffraction_command_omits_separation_when_zero() -> None:
    """``separation_px=0`` 是「未定标」, 必须不传 ``--separation-px``。"""
    command = diag.build_diffraction_command(out_dir="out", separation_px=0.0)
    assert "--separation-px" not in command
    assert "0.0" not in command or command[command.index("--exposure-ms") + 1] == "5.0"


def test_diffraction_command_explicit_positive_flags() -> None:
    command = diag.build_diffraction_command(
        out_dir="out", save_frames=True, subtract_dark=False
    )
    assert "--save-frames" in command
    assert "--no-save-frames" not in command
    assert "--no-subtract-dark" in command
    assert "--subtract-dark" not in command


def test_phase_command_pins_blaze_and_unique_tag() -> None:
    command = diag.build_phase_speed_command(
        out_dir="data/diagnostics/phase_speed/stamp",
        tag="phase_speed_stamp",
        slm_number=1,
        wavelength_nm=1064,
        cam_type="daheng",
        cam_id=0,
        exposure_ms=5.0,
        gray_a=0,
        gray_b=993,
        n_frames=60,
        settle_s=0.0,
        period_px=16.0,
        save_frames=False,
    )
    assert command[1:3] == ["-m", diag.PHASE_MODULE]
    assert "--pattern" in command
    assert command[command.index("--pattern") + 1] == "blaze"
    assert "flat" not in command
    assert command[command.index("--tag") + 1] == "phase_speed_stamp"
    assert (
        command[command.index("--out-dir") + 1] == "data/diagnostics/phase_speed/stamp"
    )
    assert "--save-frames" not in command
    assert command[-2:] == ["--tag", "phase_speed_stamp"]


def test_phase_command_save_frames_flag() -> None:
    command = diag.build_phase_speed_command(out_dir="out", tag="t", save_frames=True)
    assert "--save-frames" in command


# ---------------------------------------------------------------------------
# 产物目录 / 标签唯一性
# ---------------------------------------------------------------------------


def test_resolve_out_dir_is_per_run_and_per_tool() -> None:
    start = datetime(2026, 11, 21, 10, 0, 0, 0)
    first = diag.resolve_out_dir(diag.TOOL_DIFFRACTION, moment=start)
    second = diag.resolve_out_dir(
        diag.TOOL_DIFFRACTION, moment=start + timedelta(microseconds=1)
    )
    phase = diag.resolve_out_dir(diag.TOOL_PHASE_SPEED, moment=start)

    assert first != second
    assert first.parent.name == diag.TOOL_DIFFRACTION
    assert phase.parent.name == diag.TOOL_PHASE_SPEED
    assert first.parents[1].name == "diagnostics"
    # 不落在 CLI 的默认目录里 (那会被下一次运行盖掉)。
    assert "diffraction_efficiency" not in first.parts
    assert "phase_speed" in phase.parts[-2]


def test_default_phase_tag_is_time_unique() -> None:
    start = datetime(2026, 11, 21, 10, 0, 0, 0)
    first = diag.default_phase_tag(start)
    second = diag.default_phase_tag(start + timedelta(microseconds=1))
    assert first != second
    assert first.startswith(diag.TOOL_PHASE_SPEED)


# ---------------------------------------------------------------------------
# 门闩 / ack gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("acknowledged", "running", "expected"),
    [
        (False, False, True),
        (True, False, False),
        (False, True, True),
        (True, True, True),
    ],
)
def test_run_blocked_truth_table(
    acknowledged: bool, running: bool, expected: bool
) -> None:
    assert diag.run_blocked(acknowledged, running) is expected


def test_shared_defaults_uses_bench_camera(monkeypatch: pytest.MonkeyPatch) -> None:
    from ao_shaping.gui.slm import bench_settings

    monkeypatch.setattr(bench_settings, "bench_camera", lambda: ("miicam", 4))
    defaults = diag.shared_defaults()
    assert defaults["cam_type"] == "miicam"
    assert defaults["cam_id"] == 4
    assert "实验级" in defaults["camera_source"]


def test_shared_defaults_falls_back_when_bench_camera_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ao_shaping.gui.slm import bench_settings

    def _boom() -> tuple[str, int]:
        raise RuntimeError("没有 session_state")

    monkeypatch.setattr(bench_settings, "bench_camera", _boom)
    defaults = diag.shared_defaults()
    assert defaults["cam_type"] == "daheng"
    assert defaults["cam_id"] == 0
    assert defaults["slm_number"] == 1
    assert defaults["wavelength_nm"] == 1064
    assert "CLI 默认" in defaults["camera_source"]


def test_shared_defaults_rejects_unknown_camera_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ao_shaping.gui.slm import bench_settings

    monkeypatch.setattr(bench_settings, "bench_camera", lambda: ("webcam", 0))
    defaults = diag.shared_defaults()
    assert defaults["cam_type"] == "daheng"
    assert "回退" in defaults["camera_source"]


# ---------------------------------------------------------------------------
# 渲染: 空状态不碰硬件, 按钮按门闩 disabled
# ---------------------------------------------------------------------------


def test_render_offline_shows_empty_hint_and_disabled_buttons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = _install(monkeypatch, _St())

    def _no_run(**kwargs: Any) -> None:
        raise AssertionError("没有点击按钮, 不应进入子进程运行")

    monkeypatch.setattr(diag, "launch_run", _no_run)
    diag.render_diagnostics_tab()

    assert stub.subheaders[0] == "实验诊断"
    assert "共享设备设置" in stub.subheaders
    assert "衍射效率" in stub.subheaders
    assert "相位响应" in stub.subheaders
    # 独占确认的 checkbox 必须逐工具出现且文案固定。
    ack_labels = [item["label"] for item in stub.checkboxes_seen]
    assert ack_labels.count(diag.ACK_LABEL) == 2
    # 未勾选 ⇒ 两个运行按钮都 disabled。
    assert [b["label"] for b in stub.buttons] == [
        diag.DIFF_RUN_LABEL,
        diag.PHASE_RUN_LABEL,
    ]
    assert all(b["disabled"] is True for b in stub.buttons)
    # 相位区那条「不是帧率提升」的文案必须在页面出现。
    assert any("不是 SLM 像素上升时间" in item for item in stub.warnings)
    # 空状态点名下一步。
    assert any("再点「运行衍射效率测量」" in item for item in stub.captions)
    # 没有结果时不臆造 JSON / 图片。
    assert stub.jsons == []
    assert stub.images == []


def test_render_disables_both_buttons_while_another_tool_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = _install(
        monkeypatch,
        _St(
            session_state={diag.DIAG_RUNNING_KEY: True},
            checkboxes={"diag_diff_ack": True, "diag_phase_ack": True},
        ),
    )
    diag.render_diagnostics_tab()
    assert all(b["disabled"] is True for b in stub.buttons)


def test_render_warns_when_separation_is_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _install(
        monkeypatch,
        _St(session_state={"diag_diff_separation_px": 0.0}),
    )
    diag.render_diagnostics_tab()
    assert any("separation_px=0" in item for item in stub.warnings)


def test_render_result_card_reports_failure_not_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = _install(
        monkeypatch,
        _St(
            session_state={
                diag.DIFF_RESULT_KEY: {
                    "tool": diag.TOOL_DIFFRACTION,
                    "returncode": 3,
                    "status": "failed",
                    "completed_at": "2026-11-21 10:00:00",
                    "out_dir": "data/diagnostics/diffraction/x",
                    "artifacts": [],
                    "tail": "Traceback:\n设备未就绪",
                }
            }
        ),
    )
    diag.render_result_card(diag.TOOL_DIFFRACTION, empty_hint="空")
    assert stub.successes == []
    assert stub.errors and "3" in stub.errors[0]
    assert stub.code_blocks and "设备未就绪" in stub.code_blocks[0]


def test_launch_run_resets_latch_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """设备交接抛错时, 门闩仍要复位, 并把失败写进记录。"""
    stub = _install(monkeypatch, _St())

    def _boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("SLM 被占用")

    monkeypatch.setattr(diag, "with_slm_ownership", _boom)
    record = diag.launch_run(
        tool=diag.TOOL_PHASE_SPEED,
        command=["python", "-m", diag.PHASE_MODULE],
        slm_number=1,
        out_dir=Path("data/diagnostics/phase_speed/x"),
        result_key=diag.PHASE_RESULT_KEY,
        tag="t",
    )
    assert record["status"] == "failed"
    assert "SLM 被占用" in record["tail"]
    assert stub.session_state[diag.DIAG_RUNNING_KEY] is False
    assert stub.session_state[diag.PHASE_RESULT_KEY] is record


def test_launch_run_uses_ownership_and_passes_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = _install(monkeypatch, _St())
    seen: dict[str, Any] = {}

    class _Ownership:
        def __init__(self, slm_num: int) -> None:
            seen["slm_num"] = slm_num

        def __enter__(self) -> _Ownership:
            seen["latch_inside"] = stub.session_state[diag.DIAG_RUNNING_KEY]
            return self

        def __exit__(self, *exc: Any) -> bool:
            return False

    def _fake_run(command: list[str], **kwargs: Any) -> Any:
        seen["command"] = list(command)
        seen["kwargs"] = kwargs
        return _Completed(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(diag, "with_slm_ownership", _Ownership)
    monkeypatch.setattr(diag.subprocess, "run", _fake_run)

    record = diag.launch_run(
        tool=diag.TOOL_DIFFRACTION,
        command=["python", "-m", diag.DIFFRACTION_MODULE, "probe"],
        slm_number=2,
        out_dir=Path("data/diagnostics/diffraction/x"),
        result_key=diag.DIFF_RESULT_KEY,
    )
    assert record["status"] == "ok"
    assert record["returncode"] == 0
    assert seen["slm_num"] == 2
    assert seen["latch_inside"] is True
    assert seen["command"][:4] == ["python", "-m", diag.DIFFRACTION_MODULE, "probe"]
    assert seen["kwargs"]["encoding"] == "utf-8"
    assert stub.session_state[diag.DIAG_RUNNING_KEY] is False


class _Completed:
    """``subprocess.CompletedProcess`` 的替身。"""

    def __init__(self, *, returncode: int, stdout: str, stderr: str) -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


# ---------------------------------------------------------------------------
# multi_slm_controller: 固定尾 tab 的顺序
# ---------------------------------------------------------------------------


class _TabCtx:
    def __init__(self, label: str, calls: list[str]) -> None:
        self.label = label
        self.calls = calls

    def __enter__(self) -> _TabCtx:
        self.calls.append(self.label)
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False


class _MainSt:
    """``main()`` 的最小替身: 记录 tab 标签, 其余全部 no-op。"""

    def __init__(self) -> None:
        self.session_state: dict[str, Any] = {"available_slms": [1]}
        self.tab_labels: list[str] = []
        self.titles: list[str] = []
        self.sidebar = nullcontext()

    def set_page_config(self, **kwargs: Any) -> None:
        return None

    def title(self, body: str, **kwargs: Any) -> None:
        self.titles.append(str(body))

    def header(self, body: str, **kwargs: Any) -> None:
        self.titles.append(str(body))

    def expander(self, *args: Any, **kwargs: Any) -> _Ctx:
        return _Ctx()

    def info(self, body: str, **kwargs: Any) -> None:
        return None

    def tabs(self, labels: list[str]) -> list[_TabCtx]:
        self.tab_labels = list(labels)
        return [_TabCtx(label, self.titles) for label in labels]


def test_main_appends_diagnostics_tab_last(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _MainSt()
    rendered: list[str] = []

    monkeypatch.setattr(mc, "st", stub)
    monkeypatch.setattr(mc, "connected_slm_numbers", lambda: [1])
    monkeypatch.setattr(mc, "render_slm_sidebar", lambda: None)
    monkeypatch.setattr(mc, "display_slm_status", lambda n: None)
    monkeypatch.setattr(mc, "render_phase_control", lambda n: None)
    monkeypatch.setattr(mc, "render_phase_preview", lambda n: None)
    monkeypatch.setattr(mc, "render_closed_loop_shaping", lambda n: None)
    monkeypatch.setattr(mc, "render_pattern_helper_tab", lambda n: None)
    monkeypatch.setattr(mc, "render_ccd_live", lambda: rendered.append("ccd"))
    monkeypatch.setattr(
        mc, "render_diagnostics_tab", lambda: rendered.append("diagnostics")
    )

    mc.main()

    assert stub.tab_labels == [
        "SLM 1 相位控制",
        "SLM 1 PatternHelper",
        "实时 CCD 画面",
        "设备诊断",
    ]
    # 每个 tab 恰好被消费一次, 且诊断页落在最后。
    assert rendered == ["ccd", "diagnostics"]


# ---------------------------------------------------------------------------
# 渲染纪律 (scope: 本模块)
# ---------------------------------------------------------------------------


def test_source_has_no_divider_or_primary_button() -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "divider":
                offenders.append(f"L{node.lineno} st.divider()")
            for keyword in node.keywords:
                if (
                    keyword.arg == "type"
                    and isinstance(keyword.value, ast.Constant)
                    and keyword.value.value == "primary"
                ):
                    offenders.append(f"L{node.lineno} type=primary")
    assert offenders == []
