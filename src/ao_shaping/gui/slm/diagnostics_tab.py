"""设备诊断页 (固定尾 tab): 衍射效率探针 + 相位响应.

两个实验测量工具共用一份页面, 但**采集与分析仍留在工具目录里**:

* ``python -m ao_shaping.tools.slm.calibration.diffraction_efficiency probe``
* ``python -m ao_shaping.tools.slm.phase_speed`` (图案固定 ``blaze``)

本模块只做三件事: 收集参数、拼装命令、跑子进程并把产物如实展示。由此定下三条
纪律:

* **不修改两个工具目录**: 命令走 ``python -m ...``, 页面里不写第二份采集/分析
  口径; 工具目录的未提交改动属于别的车道。
* **不宣称实验帧率提升**: 相位响应量的是 **SLM+CCD 系统响应**, 既不是 SLM 像素
  上升时间, 也不表示 SLM 实验帧率能提高 —— 界面文案按此固定。
* **不并行**: 两个工具共享 ``st.session_state["diag_running"]`` 门闩, 任一在跑时
  两个运行按钮都 ``disabled``; 都套 :func:`with_slm_ownership`, 因为 Santec 的
  USB 句柄不可重入, 工具进程要自己开 SLM/相机。

纯函数 (:func:`build_diffraction_command` / :func:`build_phase_speed_command` /
:func:`resolve_out_dir` / :func:`run_blocked` 等) 与 ``st.*`` 隔离, 全部可离线
单测; 测试里不得 monkeypatch 真硬件。
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st

from ao_shaping.drivers.ccd.common import CAMERA_TYPE_CHOICES
from ao_shaping.gui.slm.slm_ownership import with_slm_ownership
from ao_shaping.gui.slm.slm_registry import MAX_SLMS
from ao_shaping.utils.io.paths import data_subdir

__all__ = [
    "ACK_LABEL",
    "DIFF_RUN_LABEL",
    "PHASE_RUN_LABEL",
    "build_diffraction_command",
    "build_phase_speed_command",
    "camera_type_index",
    "command_tail",
    "default_phase_tag",
    "resolve_out_dir",
    "render_diagnostics_tab",
    "run_blocked",
    "shared_defaults",
]

#: 探针子命令所在的模块 (CLI 结构: ``... diffraction_efficiency probe``)。
DIFFRACTION_MODULE = "ao_shaping.tools.slm.calibration.diffraction_efficiency"
#: 相位响应工具模块。图案在 GUI 里固定为 blaze —— flat 在 0 级几乎不变, 看不到阶跃。
PHASE_MODULE = "ao_shaping.tools.slm.phase_speed"
PHASE_PATTERN = "blaze"

TOOL_DIFFRACTION = "diffraction"
TOOL_PHASE_SPEED = "phase_speed"

#: 两个工具共享的互斥门闩: 任一在跑时为 True, 两个按钮都 disabled。
DIAG_RUNNING_KEY = "diag_running"
DIFF_PARAMS_KEY = "diag_diffraction_params"
DIFF_RESULT_KEY = "diag_diffraction_result"
PHASE_PARAMS_KEY = "diag_phase_params"
PHASE_RESULT_KEY = "diag_phase_result"

ACK_LABEL = "我已确认本次允许进行硬件诊断，设备空闲且可由此任务独占"
DIFF_RUN_LABEL = "运行衍射效率测量"
PHASE_RUN_LABEL = "运行相位响应测量"

#: 失败时只贴末尾这么多行 —— 完整日志在产物目录里, 页面不铺满屏幕。
TAIL_LINES = 8


# ---------------------------------------------------------------------------
# 纯函数: 命令拼装与产物路径
# ---------------------------------------------------------------------------


def _stamp(moment: datetime | None = None) -> str:
    """``YYYYmmdd_HHMMSS_ffffff`` —— 微秒级, 同一秒内连点也不会覆盖旧结果。"""
    return (moment or datetime.now()).strftime("%Y%m%d_%H%M%S_%f")


def resolve_out_dir(tool: str, *, moment: datetime | None = None) -> Path:
    """单次运行的独立产物目录 ``data/diagnostics/<tool>/<时间戳>``。

    不沿用 CLI 的默认目录 (``data/diffraction_efficiency/probe`` 等): 那是「上一次
    的结果」, 页面跑第二次就会把它盖掉。
    """
    return data_subdir("diagnostics", str(tool), _stamp(moment))


def default_phase_tag(moment: datetime | None = None) -> str:
    """相位响应的默认产物标签 (``{tag}_result.json`` / ``{tag}_trace.png``)。"""
    return f"{TOOL_PHASE_SPEED}_{_stamp(moment)}"


def build_diffraction_command(
    *,
    out_dir: str | Path,
    slm_number: int = 1,
    slm_wavelength: int = 1064,
    cam_type: str = "daheng",
    cam_id: int = 0,
    exposure_ms: float = 5.0,
    gray_depth: int = 0,
    period_px: int = 16,
    axis: str = "x",
    separation_px: float = 471.0,
    lobe_axis: str = "y",
    frames: int = 3,
    settle_s: float = 0.25,
    save_frames: bool = False,
    subtract_dark: bool = True,
) -> list[str]:
    """拼 ``diffraction_efficiency probe`` 命令行。

    ``separation_px <= 0`` 表示「没定标」: **不传** ``--separation-px``, 让工具按
    周期估算并把结果标成粗略/未定标 —— 传 0 反而像「已标定且间距为 0」。
    """
    command = [
        sys.executable,
        "-m",
        DIFFRACTION_MODULE,
        "probe",
        "--out-dir",
        str(out_dir),
        "--slm-number",
        str(int(slm_number)),
        "--slm-wavelength",
        str(int(slm_wavelength)),
        "--cam-type",
        str(cam_type),
        "--cam-id",
        str(int(cam_id)),
        "--exposure-ms",
        str(float(exposure_ms)),
        "--gray-depth",
        str(int(gray_depth)),
        "--period-px",
        str(int(period_px)),
        "--axis",
        str(axis),
    ]
    if float(separation_px) > 0:
        command += ["--separation-px", str(float(separation_px))]
    command += [
        "--lobe-axis",
        str(lobe_axis),
        "--frames",
        str(int(frames)),
        "--settle-s",
        str(float(settle_s)),
        "--save-frames" if save_frames else "--no-save-frames",
        "--subtract-dark" if subtract_dark else "--no-subtract-dark",
    ]
    return command


def build_phase_speed_command(
    *,
    out_dir: str | Path,
    tag: str,
    slm_number: int = 1,
    wavelength_nm: int = 1064,
    cam_type: str = "daheng",
    cam_id: int = 0,
    exposure_ms: float = 5.0,
    gray_a: int = 0,
    gray_b: int = 993,
    n_frames: int = 60,
    settle_s: float = 0.0,
    period_px: float = 16.0,
    save_frames: bool = False,
) -> list[str]:
    """拼 ``phase_speed`` 命令行; 图案固定 ``blaze`` (不提供 flat 选项)。"""
    command = [
        sys.executable,
        "-m",
        PHASE_MODULE,
        "--out-dir",
        str(out_dir),
        "--slm-number",
        str(int(slm_number)),
        "--wavelength-nm",
        str(int(wavelength_nm)),
        "--cam-type",
        str(cam_type),
        "--cam-id",
        str(int(cam_id)),
        "--exposure-ms",
        str(float(exposure_ms)),
        "--gray-a",
        str(int(gray_a)),
        "--gray-b",
        str(int(gray_b)),
        "--n-frames",
        str(int(n_frames)),
        "--settle-s",
        str(float(settle_s)),
        "--pattern",
        PHASE_PATTERN,
        "--period-px",
        str(float(period_px)),
    ]
    if save_frames:
        command.append("--save-frames")
    command += ["--tag", str(tag)]
    return command


def run_blocked(acknowledged: bool, running: bool) -> bool:
    """运行按钮是否 ``disabled``: 未勾选独占确认, 或另一工具正在跑。

    抽成纯函数是为了让「门闩 + ack gate」两条规则能被离线单测钉住, 而不必驱动
    Streamlit 按钮。
    """
    return (not bool(acknowledged)) or bool(running)


def camera_type_index(cam_type: str) -> int:
    """``selectbox`` 的默认索引; 未知类型退回第一项 (daheng) 而不是抛错。"""
    text = str(cam_type)
    return CAMERA_TYPE_CHOICES.index(text) if text in CAMERA_TYPE_CHOICES else 0


def shared_defaults() -> dict[str, Any]:
    """共享设备设置的默认值: 相机取实验级设置, 其余取工具 CLI 默认。

    取不到 (无 ``session_state`` / 后台线程) 时**不猜**: 回落到 CLI 默认值, 并把
    来源写进界面 caption。
    """
    cam_type, cam_id = "daheng", 0
    source = "工具 CLI 默认值"
    try:
        from ao_shaping.gui.slm import bench_settings

        cam_type, cam_id = bench_settings.bench_camera()
        source = "实验级「相机」设置"
    except (
        Exception
    ):  # policy: 诊断页不因缺 session_state 而崩；取不到就用 CLI 默认值并在界面标明
        pass
    text = str(cam_type)
    if text not in CAMERA_TYPE_CHOICES:
        source = f"{source}（{text} 不在本工具支持列表, 已回退 daheng）"
        text = "daheng"
    return {
        "slm_number": 1,
        "wavelength_nm": 1064,
        "cam_type": text,
        "cam_id": int(cam_id),
        "camera_source": source,
    }


def command_tail(completed: Any, limit: int = TAIL_LINES) -> str:
    """失败输出末尾 ``limit`` 行 (stderr 优先, 空则 stdout)。"""
    stderr = (getattr(completed, "stderr", None) or "").strip()
    stdout = (getattr(completed, "stdout", None) or "").strip()
    text = stderr or stdout
    return "\n".join(text.splitlines()[-int(limit) :])


def artifact_names(tool: str, *, tag: str = "") -> tuple[str, ...]:
    """该工具在 ``--out-dir`` 下应写出的文件名 (用来展示真实产物)。"""
    if tool == TOOL_DIFFRACTION:
        return ("summary.json", "summary.csv")
    return (f"{tag}_result.json", f"{tag}_trace.png")


def collect_artifacts(tool: str, out_dir: Path, *, tag: str = "") -> list[Path]:
    """``out_dir`` 里**实际存在**的预期产物 (不臆造缺失文件)。"""
    return [
        out_dir / name
        for name in artifact_names(tool, tag=tag)
        if (out_dir / name).is_file()
    ]


def build_run_record(
    tool: str,
    command: list[str],
    completed: Any,
    *,
    out_dir: Path,
    tag: str = "",
) -> dict[str, Any]:
    """把一次子进程结果整理成 session_state 里的记录 (真实退出码 / 产物)。"""
    returncode = int(getattr(completed, "returncode", -1))
    record: dict[str, Any] = {
        "tool": str(tool),
        "command": list(command),
        "returncode": returncode,
        "status": "ok" if returncode == 0 else "failed",
        "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "out_dir": str(out_dir),
        "artifacts": [str(path) for path in collect_artifacts(tool, out_dir, tag=tag)],
    }
    if record["status"] != "ok":
        record["tail"] = command_tail(completed)
    return record


def failure_record(
    tool: str,
    command: list[str],
    error: BaseException,
    *,
    out_dir: Path,
    tag: str = "",
) -> dict[str, Any]:
    """子进程根本没起来 (设备交接失败等) 时的记录 —— 同样标 ``failed``。"""
    return {
        "tool": str(tool),
        "command": list(command),
        "returncode": None,
        "status": "failed",
        "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "out_dir": str(out_dir),
        "artifacts": [],
        "tail": f"{type(error).__name__}: {error}",
    }


# ---------------------------------------------------------------------------
# 子进程运行 (设备交接 + 互斥门闩)
# ---------------------------------------------------------------------------


def _spinner_text(tool: str) -> str:
    if tool == TOOL_DIFFRACTION:
        return "正在测衍射效率：SLM 独占中，请勿触碰设备…"
    return "正在测相位响应：SLM 独占中，请勿触碰设备…"


def launch_run(
    *,
    tool: str,
    command: list[str],
    slm_number: int,
    out_dir: Path,
    result_key: str,
    tag: str = "",
) -> dict[str, Any]:
    """跑一次工具进程; ``try/finally`` 保证门闩复位。

    为什么先 :func:`with_slm_ownership`: 工具进程要自己开 SLM/相机, Santec 的 USB
    句柄不可重入 —— 必须先把 GUI 这一份交还, 跑完 ``finally`` 再取回。
    """
    st.session_state[DIAG_RUNNING_KEY] = True
    try:
        with st.spinner(_spinner_text(tool)):
            with with_slm_ownership(int(slm_number)):
                completed = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    check=False,
                )
        record = build_run_record(
            tool, command, completed, out_dir=Path(out_dir), tag=tag
        )
    except (
        Exception
    ) as exc:  # policy: 页面把设备交接/进程启动失败显示成失败, 绝不当成功
        record = failure_record(tool, command, exc, out_dir=Path(out_dir), tag=tag)
    finally:
        st.session_state[DIAG_RUNNING_KEY] = False
    st.session_state[result_key] = record
    return record


# ---------------------------------------------------------------------------
# 产物展示
# ---------------------------------------------------------------------------


def _first_json(artifacts: list[Path]) -> Any | None:
    for path in artifacts:
        if path.suffix.lower() != ".json":
            continue
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # policy: 产物读不动只影响展示, 不能把一次成功的运行报成失败
            return None
    return None


def _first_image(artifacts: list[Path]) -> Path | None:
    for path in artifacts:
        if path.suffix.lower() in (".png", ".jpg", ".jpeg"):
            return path
    return None


def render_result_card(tool: str, *, empty_hint: str) -> None:
    """如实展示上一次运行: 状态 / 退出码 / 完成时间 / 产物 / JSON / 图片。"""
    key = DIFF_RESULT_KEY if tool == TOOL_DIFFRACTION else PHASE_RESULT_KEY
    record = st.session_state.get(key)
    if not isinstance(record, Mapping) or not record:
        st.caption(empty_hint)
        return

    ok = record.get("status") == "ok"
    summary = (
        f"状态 {record.get('status')} · 退出码 {record.get('returncode')} · "
        f"完成于 {record.get('completed_at')}"
    )
    if ok:
        st.success(summary)
    else:
        st.error(summary)
        tail = str(record.get("tail") or "").strip()
        if tail:
            st.code(tail, language=None)

    out_dir = record.get("out_dir")
    if out_dir:
        st.caption(f"产物目录: {out_dir}")

    artifacts = [Path(str(item)) for item in record.get("artifacts") or []]
    if not artifacts:
        st.warning(
            "产物目录里没有预期文件 —— 按上面的末尾输出判断是设备问题还是参数问题。"
        )
    for path in artifacts:
        st.caption(f"产物: {path}")

    payload = _first_json(artifacts)
    if payload is not None:
        st.json(payload)
    image = _first_image(artifacts)
    if image is not None:
        st.image(str(image))


# ---------------------------------------------------------------------------
# 页面
# ---------------------------------------------------------------------------


def _render_shared_settings() -> dict[str, Any]:
    """共享设备设置区。相机默认取实验级设置, 取不到就用 CLI 默认并标明。"""
    st.subheader("共享设备设置")
    defaults = shared_defaults()
    st.caption(
        f"相机类型/编号来自{defaults['camera_source']}；SLM 编号与波长取工具 CLI "
        "默认值 (1 / 1064 nm)，可在下面覆盖。"
    )
    cols = st.columns(4)
    with cols[0]:
        slm_number = st.number_input(
            "SLM 编号",
            min_value=1,
            max_value=MAX_SLMS,
            value=int(defaults["slm_number"]),
            step=1,
            key="diag_slm_number",
        )
    with cols[1]:
        wavelength_nm = st.number_input(
            "波长 nm",
            min_value=400,
            max_value=2000,
            value=int(defaults["wavelength_nm"]),
            step=1,
            key="diag_wavelength_nm",
        )
    with cols[2]:
        cam_type = st.selectbox(
            "相机类型",
            CAMERA_TYPE_CHOICES,
            index=camera_type_index(defaults["cam_type"]),
            key="diag_cam_type",
        )
    with cols[3]:
        cam_id = st.number_input(
            "相机编号",
            min_value=0,
            value=int(defaults["cam_id"]),
            step=1,
            key="diag_cam_id",
        )
    return {
        "slm_number": int(slm_number),
        "wavelength_nm": int(wavelength_nm),
        "cam_type": str(cam_type),
        "cam_id": int(cam_id),
    }


def _render_diffraction_section(settings: Mapping[str, Any]) -> None:
    st.subheader("衍射效率")
    st.caption(
        "轻量探针: 单个灰度深度上测 0 级与 ±1 级的能量占比，不需要光路标定 .npz；"
        "未给 ±1 级间距时结果只作相对指标（粗略/未定标）。"
    )
    left, right = st.columns(2)
    with left:
        exposure_ms = st.number_input(
            "曝光时间 ms",
            min_value=0.05,
            max_value=1000.0,
            value=5.0,
            step=1.0,
            key="diag_diff_exposure_ms",
        )
        period_px = st.number_input(
            "光栅周期 px",
            min_value=2,
            max_value=512,
            value=16,
            step=1,
            key="diag_diff_period_px",
        )
        axis = st.selectbox("光栅轴向", ("x", "y"), index=0, key="diag_diff_axis")
    with right:
        separation_px = st.number_input(
            "±1 级间距 px（0=不定标）",
            min_value=0.0,
            max_value=2000.0,
            value=471.0,
            step=1.0,
            key="diag_diff_separation_px",
        )
        st.caption(
            "设为 0 表示不提供 --separation-px，结果标为「粗略/未定标」；"
            "本台推荐 471，其他光路须重新标定。"
        )
        lobe_axis = st.selectbox(
            "±1 级位移方向", ("auto", "x", "y"), index=2, key="diag_diff_lobe_axis"
        )
        frames = st.number_input(
            "每点帧数",
            min_value=1,
            max_value=50,
            value=3,
            step=1,
            key="diag_diff_frames",
        )
    with st.expander("高级参数", expanded=False):
        gray_depth = st.number_input(
            "灰度深度",
            min_value=0,
            max_value=1023,
            value=0,
            step=1,
            key="diag_diff_gray_depth",
        )
        settle_s = st.number_input(
            "稳定等待 s",
            min_value=0.0,
            max_value=10.0,
            value=0.25,
            step=0.05,
            key="diag_diff_settle_s",
        )
        save_frames = st.checkbox(
            "保存原始帧", value=False, key="diag_diff_save_frames"
        )
        subtract_dark = st.checkbox(
            "扣除暗场", value=True, key="diag_diff_subtract_dark"
        )

    if float(separation_px) <= 0:
        st.warning(
            "separation_px=0：本次不传 --separation-px，±1 级位置按光栅周期估算"
            "（本台误差可达 6 倍），结果只能标为「粗略/未定标」。要定量请先标定并"
            "填入实测间距。"
        )

    running = bool(st.session_state.get(DIAG_RUNNING_KEY, False))
    acknowledged = st.checkbox(ACK_LABEL, value=False, key="diag_diff_ack")
    if st.button(
        DIFF_RUN_LABEL,
        key="diag_diff_run",
        disabled=run_blocked(acknowledged, running),
    ):
        out_dir = resolve_out_dir(TOOL_DIFFRACTION)
        params: dict[str, Any] = {
            "exposure_ms": float(exposure_ms),
            "gray_depth": int(gray_depth),
            "period_px": int(period_px),
            "axis": str(axis),
            "separation_px": float(separation_px),
            "lobe_axis": str(lobe_axis),
            "frames": int(frames),
            "settle_s": float(settle_s),
            "save_frames": bool(save_frames),
            "subtract_dark": bool(subtract_dark),
        }
        command = build_diffraction_command(
            out_dir=out_dir,
            slm_number=int(settings["slm_number"]),
            slm_wavelength=int(settings["wavelength_nm"]),
            cam_type=str(settings["cam_type"]),
            cam_id=int(settings["cam_id"]),
            **params,
        )
        st.session_state[DIFF_PARAMS_KEY] = {
            "settings": dict(settings),
            "params": dict(params),
            "out_dir": str(out_dir),
            "command": list(command),
        }
        launch_run(
            tool=TOOL_DIFFRACTION,
            command=command,
            slm_number=int(settings["slm_number"]),
            out_dir=out_dir,
            result_key=DIFF_RESULT_KEY,
        )

    render_result_card(
        TOOL_DIFFRACTION,
        empty_hint=(
            "还没有本次会话的结果 —— 在「高级参数」里确认灰度深度/稳定等待，"
            "勾选上面的独占确认，再点「运行衍射效率测量」。"
        ),
    )


def _render_phase_section(settings: Mapping[str, Any]) -> None:
    st.subheader("相位响应")
    st.warning(
        "测量范围：SLM+CCD 系统响应；不是 SLM 像素上升时间，也不表示 SLM 实验帧率"
        "提升；使用 blaze 图案。"
    )
    st.caption(
        "图案固定为 blaze（只读）：flat 平场在 0 级几乎不变，看不到灰度阶跃的响应。"
    )
    left, right = st.columns(2)
    with left:
        exposure_ms = st.number_input(
            "曝光时间 ms",
            min_value=0.05,
            max_value=1000.0,
            value=5.0,
            step=1.0,
            key="diag_phase_exposure_ms",
        )
        gray_a = st.number_input(
            "阶跃前灰度 gray_a",
            min_value=0,
            max_value=1023,
            value=0,
            step=1,
            key="diag_phase_gray_a",
        )
        gray_b = st.number_input(
            "阶跃后灰度 gray_b",
            min_value=0,
            max_value=1023,
            value=993,
            step=1,
            key="diag_phase_gray_b",
        )
    with right:
        n_frames = st.number_input(
            "阶跃后帧数",
            min_value=5,
            max_value=2000,
            value=60,
            step=1,
            key="diag_phase_n_frames",
        )
        period_px = st.number_input(
            "blaze 周期 px",
            min_value=2.0,
            max_value=512.0,
            value=16.0,
            step=1.0,
            key="diag_phase_period_px",
        )
    with st.expander("高级参数", expanded=False):
        settle_s = st.number_input(
            "写入后额外稳定等待 s",
            min_value=0.0,
            max_value=10.0,
            value=0.0,
            step=0.05,
            key="diag_phase_settle_s",
        )
        save_frames = st.checkbox(
            "保存原始帧", value=False, key="diag_phase_save_frames"
        )
        if "diag_phase_tag" not in st.session_state:
            st.session_state["diag_phase_tag"] = default_phase_tag()
        tag = st.text_input("产物标签", key="diag_phase_tag")

    running = bool(st.session_state.get(DIAG_RUNNING_KEY, False))
    acknowledged = st.checkbox(ACK_LABEL, value=False, key="diag_phase_ack")
    if st.button(
        PHASE_RUN_LABEL,
        key="diag_phase_run",
        disabled=run_blocked(acknowledged, running),
    ):
        out_dir = resolve_out_dir(TOOL_PHASE_SPEED)
        params = {
            "exposure_ms": float(exposure_ms),
            "gray_a": int(gray_a),
            "gray_b": int(gray_b),
            "n_frames": int(n_frames),
            "settle_s": float(settle_s),
            "period_px": float(period_px),
            "save_frames": bool(save_frames),
            "tag": str(tag),
        }
        command = build_phase_speed_command(
            out_dir=out_dir,
            slm_number=int(settings["slm_number"]),
            wavelength_nm=int(settings["wavelength_nm"]),
            cam_type=str(settings["cam_type"]),
            cam_id=int(settings["cam_id"]),
            **params,
        )
        st.session_state[PHASE_PARAMS_KEY] = {
            "settings": dict(settings),
            "params": dict(params),
            "out_dir": str(out_dir),
            "command": list(command),
        }
        launch_run(
            tool=TOOL_PHASE_SPEED,
            command=command,
            slm_number=int(settings["slm_number"]),
            out_dir=out_dir,
            result_key=PHASE_RESULT_KEY,
            tag=str(tag),
        )

    render_result_card(
        TOOL_PHASE_SPEED,
        empty_hint=(
            "还没有本次会话的结果 —— 先确认曝光不超量程（本台 0 级过强时轻微过曝"
            "才看得见图案），勾选上面的独占确认，再点「运行相位响应测量」。"
        ),
    )


def render_diagnostics_tab() -> None:
    """固定尾 tab「设备诊断」的入口。"""
    st.subheader("实验诊断")
    st.caption(
        "这两个工具会**独占** SLM / CCD / 激光：运行时 GUI 会先交还 SLM 句柄，"
        "操作期间请勿触碰设备，也不要并行跑整形或标定。未获准占用设备时只做离线"
        "集成与验证，不要点击运行按钮。"
    )
    settings = _render_shared_settings()
    _render_diffraction_section(settings)
    _render_phase_section(settings)
