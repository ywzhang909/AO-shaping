"""面向可微优化模型的专用 Streamlit 控制界面。

定位
----
本页是**迁移学习闭环整形**（``ao_shaping.tools.slm.transfer``，少参数物理预测
模型 + 当前光路重测几何 + 检查点迁移 + CCD 锚定 A–B–B–A 验收）的专用操作台：
单 SLM + 单相机，页面内容只有这条管线需要的控件。

与 ``multi_slm_controller.py`` 的关系
------------------------------------
**不重复实现**任何通用逻辑，全部 import 复用：

- 设备发现 / 连接 / 相机枚举 / bench 设置：``multi_slm_controller`` 的
  ``_initialize_slm_state`` / ``_refresh_device_list`` / ``render_slm_sidebar``。
- 参数渲染与运行编排：``shaping_controls.TransferMethod``。参数控件与
  ``render_closed_loop_shaping`` 里的**同名同 key**，运行走同一条
  ``ShapingMethod.run``（``shaping_session`` 装饰器 → 设备交还 → 下发接受相位），
  不在本页写第二套 orchestrator 或第二条 stage 序列。

合并掉的重复功能
----------------
- 侧栏「CCD参数设置」不再第二次调用 ``render_bench_settings()`` ——
  ``render_slm_sidebar()`` 内部已调用一次，重复调用会让 ``bench_cam_type``
  等 widget key 撞车直接崩页。
- 去掉多 SLM 页签、pattern 生成（Zernike / 湍流屏）、GS-refine 与 model-in-loop
  适配器选择、相位预览等与本管线无关的控件。
- 光路参数（相机 / 曝光 / 读出窗口 / 入射中心 / 几何）**只在侧栏**出现一次，
  主区通过 ``bench_settings.merge_bench_values`` 取回，不再铺第二份控件。

页面顺序（与 ``render_closed_loop_shaping`` 同一契约）
-----------------------------------------------------
    光路状态条 → 目标与预算 → 高级参数 → 校验区 → 运行 → 进度/轨迹

校验必须早于按钮，按钮才能 ``disabled``；运行中不重复排进度明细（
``TransferMethod`` 自己有一份实时面板）。轨迹图（CCD 迭代过程 / 模型预测 /
指标折线）只在运行结束后铺一次，数据来自 :func:`collect_traces`。
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import streamlit as st

from ao_shaping.gui.slm import bench_settings
from ao_shaping.gui.slm.multi_slm_controller import render_slm_sidebar
from ao_shaping.gui.slm.shaping_controls import (
    TransferMethod,
    render_bench_status,
    snapshot_summary_line,
    with_slm_ownership,
)
from ao_shaping.tools.slm.panel.probe_policy import (
    STATE_LOW_SNR,
    STATE_NO_RESPONSE,
    STATE_OK,
    STATE_WEAK_PATH,
    VERDICT_STATES,
)
from ao_shaping.gui.slm.transfer_progress import (
    collect_progress,
    collect_traces,
    missing_trace_labels,
    render_progress,
    render_traces,
)
from ao_shaping.gui.slmshape.connect import (
    connected_slm_numbers as _connected_slm_numbers,
    ensure_state as _ensure_state,
    render_connection_hint,
)
from ao_shaping.gui.slmshape.layout import (
    PLAIN_LAYOUT as _PLAIN_LAYOUT,
    Layout as _Layout,
    layout as _layout_factory,
    render_layout_selector as _render_layout_selector,
)

#: 页面标题 / 侧栏品牌名。
PAGE_TITLE = "可微优化模型控制器"

#: 结果区在 ``st.session_state`` 里的键：最近一次运行写下的会话目录。
_SESSION_DIR_KEY = "diff_opt_session_dir"

#: 侧栏「页面布局」的折叠面板句柄名（本页自己的 key，不串到别的页面）。
_SUBHEADER_BOX_KEY = "diff_opt_subheader_box"

#: 侧栏「页面布局」选择器自己的 key。
_SUBHEADER_OPTION_KEY = "diff_opt_layout_choice"

#: 「默认主区域」选项的文案。
_PLAIN_LAYOUT = "（无，默认主区域）"


# --------------------------------------------------------------------------- #
# 渲染目标 —— 委派给 slmshape.layout（_Layout / _layout 已在上方导入/定义）
# --------------------------------------------------------------------------- #


def _layout() -> _Layout:
    return _layout_factory(_SUBHEADER_BOX_KEY)



# --------------------------------------------------------------------------- #
# 主体各段
# --------------------------------------------------------------------------- #
def render_connection_hint() -> None:
    """没有任何已连接 SLM 时的页面主体。"""
    st.title(PAGE_TITLE)
    st.info("请在左侧连接至少一个 SLM")
    st.caption(
        "本页只需一台 SLM 与一台相机：光路参数在左侧「CCD参数设置」，"
        "标定按钮同在那里。"
    )


def render_status_header(slm_num: int, prefix: str) -> None:
    """标题 + 一句话说明 + 光路状态条（这条光路现在能不能信）。"""
    st.title(PAGE_TITLE)
    st.caption(
        f"SLM {slm_num} · 单相机 · 迁移学习闭环整形（物理预测模型）。与 CLI "
        "`python -m ao_shaping.tools.slm.transfer` 同一入口：先名义初始化 / "
        "迁移历史检查点，再用 CCD 锚定短块与 A–B–B–A 配对验收生成目标相位。"
    )
    status = render_bench_status(prefix, slm_num)
    if status:
        st.caption(status)
    manual = bench_settings.manual_fields(prefix)
    if manual:
        st.warning(
            "光路参数有手动覆盖，标定不会改动它们："
            + "；".join(bench_settings.field_hint(prefix, name) for name in manual)
        )


# --------------------------------------------------------------------------- #
# 光路快速探测 —— 一次给出三份实测结果
#
# 分工是硬的: **采集与判定都不在本页**。
#   * 采集（曝光阶梯 + 离焦 sweep + 落盘）只有一份: ``scripts/slm/probe_optical_path_validity.py``;
#   * 四态判定只有一份: ``tools/slm/panel/probe_policy.decide``（脚本也 import 它）。
# 本页只做三件事: 开跑（交还设备后起子进程）、读回它写的 ``optical_path_probe.json``、
# 把三个结论渲染出来 + 把建议曝光写回探测值表。任何「读读数 → 判四态」的逻辑都
# 不得在这里重写 —— 两处判定迟早会对同一份读数给出不同结论。
# --------------------------------------------------------------------------- #

#: CLI 探测脚本（采集 + JSON/PNG 落盘）。
_PROBE_SCRIPT = (
    Path(__file__).resolve().parents[4] / "scripts" / "slm" / "probe_optical_path_validity.py"
)

#: 证据根目录。与脚本 ``--out-root`` 默认值一致，这里**显式**传下去，
#: 免得两边的默认值各走各的。
_PROBE_OUT_ROOT = Path(__file__).resolve().parents[4] / "data" / "debug" / "optical_path_probe"

#: 证据 JSON 的 schema（脚本写、本页读；不认识就不猜字段）。
_PROBE_SCHEMA = "optical_path_probe/1"

#: 最近一次探测结果在 ``st.session_state`` 里的键：写回曝光会触发一次 rerun，
#: 结果必须活过那一刻，否则操作员只看到一闪。
_PROBE_OUTCOME_KEY = "diff_opt_optical_probe"

#: 探测读数注入点在 ``st.session_state`` 里的键。产品路径永远不写它（``None``
#: ⇒ 实跑 CLI）；只有无设备测试在 ``run()`` 之前塞一个假 payload / 路径进来，
#: 复用同一条渲染链而不必真的开设备。
_PROBE_READINGS_KEY = "diff_opt_probe_readings"

#: 四态 → 一行结论。状态名与判定都来自 ``probe_policy``，本页只负责措辞。
_PROBE_HEADLINES: dict[str, str] = {
    STATE_OK: "光路正常：测得出、也测得准，光通量在本系统历史水平内",
    STATE_LOW_SNR: (
        "信噪比不足：**测不准**（不是光路坏）—— 先把峰值抬到 100-200 counts 再复测"
    ),
    STATE_WEAK_PATH: (
        "光路被衰减：光斑可见、离焦也响应，但 counts/ms 只有本系统历史的"
        "一半以下（2026-10-29 实测 6.15 → 1.38 counts/ms，22%）"
    ),
    STATE_NO_RESPONSE: "面板没在调制：光斑清晰可见，但离焦改变不了任何读数",
}

#: 侧栏相机扫描里算「现在能用」的状态（``_scan_camera_status`` 的措辞）。
_PROBE_USABLE_CAM_STATUS = "可用"

#: ``probe_readings`` 可注入的三种形态:
#:   * ``None`` ⇒ 按 CLI 实跑（子进程 + 读回证据）；
#:   * callable ⇒ 返回 payload dict，或返回证据 JSON 的路径；
#:   * ``str`` / ``Path`` ⇒ 直接读那份证据 JSON。
ProbeReadings = Callable[[], Any] | str | Path | None


class ProbeUnavailable(RuntimeError):
    """这次探测没给出可用读数（脚本缺失 / 设备被占 / 证据读不到 / 不是探测证据）。

    单独一个异常类型是为了让页面**只**把这一类失败渲染成 ``st.error`` ——
    真正的编程错误仍然要炸出来，不被一句友好文案吞掉。
    """


def _probe_evidence_files() -> set[Path]:
    """证据目录下已有的全部 ``optical_path_probe.json``。"""
    if not _PROBE_OUT_ROOT.exists():
        return set()
    return set(_PROBE_OUT_ROOT.rglob("optical_path_probe.json"))


def _load_probe_json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise ProbeUnavailable(f"读不到探测证据 {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ProbeUnavailable(f"探测证据不是合法 JSON {path}: {exc}") from exc


def _check_probe_payload(payload: Any, origin: str) -> dict[str, Any]:
    """校验证据形状，返回可安全取值的 dict。

    只校验**本页要读的**那几个字段: 判定本身已经在脚本里做完了，这里再算一遍
    等于第二份判定。
    """
    if not isinstance(payload, Mapping):
        raise ProbeUnavailable(f"{origin} 不是光路探测证据（顶层不是对象）")
    schema = payload.get("schema")
    if schema is not None and str(schema) != _PROBE_SCHEMA:
        raise ProbeUnavailable(
            f"{origin} 的 schema 是 {schema!r}，本页只认 {_PROBE_SCHEMA!r}"
        )
    missing = [
        key
        for key in ("verdict", "ladder_summary")
        if not isinstance(payload.get(key), Mapping)
    ]
    if missing:
        raise ProbeUnavailable(f"{origin} 缺 {' / '.join(missing)}")
    state = payload["verdict"].get("state")
    if state not in VERDICT_STATES:
        raise ProbeUnavailable(
            f"{origin} 的四态取值 {state!r} 不在 {VERDICT_STATES} 里"
        )
    return dict(payload)


def _run_probe_cli(slm_num: int) -> tuple[dict[str, Any], Path]:
    """实跑一次 CLI 探测，返回 ``(payload, 证据 JSON 路径)``。

    为什么是**子进程**而不是 import 进来调: 采集/落盘已经在 ``scripts/`` 里，
    把它 import 进 GUI 等于让产品代码依赖一次性脚本；反过来在页面里写第二份
    采集就是两套口径。

    为什么先 :func:`with_slm_ownership`: Santec 的 USB 句柄不可重入，脚本要开
    自己的 SLM/相机，所以必须先把 GUI 这一份交还（同整形运行交还给 optimizer
    的那一步），跑完 ``finally`` 里再取回来。
    """
    if not _PROBE_SCRIPT.is_file():
        raise ProbeUnavailable(f"找不到光路探测脚本 {_PROBE_SCRIPT}")
    cam_type, cam_id = bench_settings.bench_camera()
    session = bench_settings.bench_session_id(f"slm{slm_num}")
    command = [
        sys.executable,
        str(_PROBE_SCRIPT),
        "--slm-number",
        str(int(slm_num)),
        "--cam-type",
        str(cam_type),
        "--cam-id",
        str(int(cam_id)),
        "--out-root",
        str(_PROBE_OUT_ROOT),
    ]
    if session:
        command += ["--session", session]

    before = _probe_evidence_files()
    with st.spinner("正在实测光路：曝光阶梯 → 离焦响应（约 1-2 分钟，请勿触碰设备）…"):
        with with_slm_ownership(int(slm_num)):
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=str(_PROBE_SCRIPT.parents[2]),
            )

    # 退出码 2 不是失败: 那是「跑完了，但结论不是 ok」—— 三份结果照给。
    if completed.returncode not in (0, 2):
        output = (completed.stderr or "").strip() or (completed.stdout or "").strip()
        tail = "\n".join(output.splitlines()[-8:])
        raise ProbeUnavailable(
            f"探测脚本退出码 {completed.returncode}（设备/光路没准备好）:\n{tail}"
        )

    produced = _probe_evidence_files() - before
    candidates = produced or _probe_evidence_files()
    if not candidates:
        raise ProbeUnavailable(
            f"探测跑完但没找到 optical_path_probe.json（看 {_PROBE_OUT_ROOT}）"
        )
    path = max(candidates, key=lambda item: item.stat().st_mtime)
    return _check_probe_payload(_load_probe_json(path), str(path)), path


def _obtain_probe_readings(
    slm_num: int, probe_readings: ProbeReadings
) -> tuple[dict[str, Any], Path | None]:
    """按注入形态取读数；未注入时实跑 CLI。"""
    if probe_readings is None:
        return _run_probe_cli(slm_num)
    produced = probe_readings() if callable(probe_readings) else probe_readings
    if isinstance(produced, Mapping):
        return _check_probe_payload(produced, "注入的读数"), None
    path = Path(produced)
    return _check_probe_payload(_load_probe_json(path), str(path)), path


def _write_back_exposure(payload: Mapping[str, Any], evidence: str | None) -> float | None:
    """把建议曝光写回**探测值表**，返回写进去的 ms（没得写 ⇒ ``None``）。

    两个调用缺一不可，理由不同:

    * :func:`bench_settings.record_exposure_probe` —— 探测值表 + 取样时刻，
      状态栏据此显示「探测值」与时效（``EXPOSURE_STALE_S``）；这是「实测」的
      唯一留痕点，不经过它写出来的曝光没有取样时刻。
    * :func:`bench_settings.queue_pending` —— 侧栏曝光控件已被实例化，widget
      key 此刻是锁住的，只能排队等下一次 ``flush_pending``。

    **不**直接改 widget 会话状态: 那既绕过了取样时刻，也会和 Streamlit 的
    widget 生命周期打架。
    """
    ladder = payload["ladder_summary"]
    recommended = ladder.get("recommended_exposure")
    if not isinstance(recommended, Mapping) or recommended.get("exposure_ms") is None:
        return None
    exposure_ms = float(recommended["exposure_ms"])
    bench_settings.record_exposure_probe(
        "",
        exposure_ms,
        {
            "session_id": bench_settings.bench_session_id(""),
            "probe_state": payload["verdict"]["state"],
            "probe_evidence": evidence or "",
            "peak_counts": float(recommended.get("peak_counts") or 0.0),
            "in_target_band": bool(recommended.get("in_target_band", False)),
        },
    )
    bench_settings.queue_pending("", {"bench_exposure_ms": exposure_ms})
    return exposure_ms


def _collect_probe_outcome(
    slm_num: int, probe_readings: ProbeReadings
) -> dict[str, Any]:
    """跑一次探测，返回可跨 rerun 存下来的结果（失败也只落在 ``error`` 里）。"""
    try:
        payload, path = _obtain_probe_readings(slm_num, probe_readings)
    except ProbeUnavailable as exc:
        return {"error": str(exc), "payload": None, "evidence": None, "written_ms": None}
    try:
        written = _write_back_exposure(payload, str(path) if path else None)
    except Exception as exc:  # policy: 探测结果本身有效，写回失败只是「没写进去」，必须把两者分开报给操作员，而不是把结果一起丢掉
        return {
            "error": None,
            "payload": payload,
            "evidence": str(path) if path else None,
            "written_ms": None,
            "write_error": str(exc),
        }
    return {
        "error": None,
        "payload": payload,
        "evidence": str(path) if path else None,
        "written_ms": written,
        "write_error": None,
    }


def _probe_blocked_reasons(slm_num: int | None) -> list[str]:
    """现在为什么按不了（每条都是操作员能动手的事，空列表 = 可以按）。"""
    reasons: list[str] = []
    if slm_num is None or slm_num not in _connected_slm_numbers():
        reasons.append(
            "未连接 SLM：探测有一半读数要往面板下平场/离焦相位，没有面板就没有"
            "「响应」这一半"
        )
    if bench_settings.bench_busy():
        reasons.append("设备正被一次整形/标定占用：探测要独占 SLM 与相机")
    scanned = st.session_state.get("camera_status")
    if isinstance(scanned, list) and not any(
        isinstance(entry, Mapping)
        and entry.get("status") == _PROBE_USABLE_CAM_STATUS
        for entry in scanned
    ):
        seen = sorted(
            {
                str(entry.get("status"))
                for entry in scanned
                if isinstance(entry, Mapping) and entry.get("status")
            }
        )
        detail = f"（扫描到 {', '.join(seen)}）" if seen else "（扫描结果为空）"
        reasons.append(
            "没有可用相机" + detail + "：另一半读数来自 CCD；"
            "侧栏「刷新相机状态」确认一次再试"
        )
    return reasons


def _render_probe_facts(payload: Mapping[str, Any]) -> None:
    """① 的数值面: 噪声底、峰值、可用曝光窗口。"""
    ladder = payload["ladder_summary"]
    noise = ladder.get("noise_floor")
    if isinstance(noise, Mapping) and noise:
        st.caption(
            "噪声底 "
            f"{float(noise.get('threshold_counts') or 0.0):.1f} counts"
            f"（中位数 {float(noise.get('median_counts') or 0.0):.1f} + 4σ"
            f" {float(noise.get('sigma_counts') or 0.0):.2f}）@ "
            f"{float(noise.get('exposure_ms') or 0.0):g} ms"
            f" · 该档峰值 {float(noise.get('peak_counts') or 0.0):.1f} counts"
        )
    best = ladder.get("best_usable_peak_counts")
    window = ladder.get("usable_window_ms") or []
    st.caption(
        f"阶梯最佳可用峰值 {float(best or 0.0):.0f} counts · 可用曝光 "
        + ("/".join(f"{float(v):g}" for v in window) + " ms" if window else "（无）")
    )


def _render_probe_verdict(payload: Mapping[str, Any]) -> None:
    """① 光路四态 + 依据 + 建议。"""
    verdict = payload["verdict"]
    state = str(verdict["state"])
    st.markdown(f"**① 光路四态** — `{state}` · {_PROBE_HEADLINES[state]}")
    headline = _PROBE_HEADLINES[state].split("：")[0]
    if state == STATE_OK:
        st.success(headline)
    elif state == STATE_NO_RESPONSE:
        st.error(headline)
    else:
        st.warning(headline)
    _render_probe_facts(payload)
    reasons = [str(item) for item in verdict.get("reasons") or []]
    if reasons:
        st.markdown("依据：\n" + "\n".join(f"- {item}" for item in reasons))
    advice = [str(item) for item in verdict.get("advice") or []]
    if advice:
        st.markdown("建议：\n" + "\n".join(f"- {item}" for item in advice))


def _render_probe_exposure(payload: Mapping[str, Any], outcome: Mapping[str, Any]) -> None:
    """② 建议曝光 + 是否写回。"""
    st.markdown("**② 建议曝光**")
    ladder = payload["ladder_summary"]
    recommended = ladder.get("recommended_exposure")
    if not isinstance(recommended, Mapping) or recommended.get("exposure_ms") is None:
        st.info(
            "不写回：曝光阶梯里没有可用窗口（没有一档把峰值抬到噪声底之上，"
            "或者每一档都饱和）。这张表本身就是结论 —— 见 ① 的依据。"
        )
        return
    band = recommended.get("target_band_counts")
    band_text = (
        f"{float(band[0]):g}-{float(band[1]):g} counts"
        if isinstance(band, (list, tuple)) and len(band) == 2
        else "目标区间"
    )
    measured = (
        f"{float(recommended['exposure_ms']):g} ms"
        f"（该档峰值 {float(recommended.get('peak_counts') or 0.0):.0f} counts，"
        f"{'落在' if recommended.get('in_target_band') else '不在'} {band_text}）"
    )
    written = outcome.get("written_ms")
    if written is None:
        why = outcome.get("write_error") or "没有可写的值"
        st.warning(f"建议 {measured}；**未写回**：{why}")
    else:
        st.success(
            f"建议 {measured}；**已写回**侧栏曝光的探测值（{float(written):g} ms，"
            "带取样时刻 —— 状态栏显示为探测值）"
        )
    if str(payload["verdict"]["state"]) != STATE_OK:
        st.caption(
            "注意：四态结论不是 ok。写回的只是「当前光路下测得准的那一档曝光」，"
            "不代表光路已经修好 —— 别把「加曝光后判 ok」当成修好了。"
        )


def _render_probe_evidence(payload: Mapping[str, Any], outcome: Mapping[str, Any]) -> None:
    """③ 证据落盘目录。"""
    st.markdown("**③ 证据落盘**")
    evidence = outcome.get("evidence")
    if not evidence:
        st.caption("这次读数是注入的（测试/离线），没有落盘证据。")
        return
    path = Path(str(evidence))
    st.code(str(path), language=None)
    details = [f"created_utc {payload.get('created_utc', '（无）')}"]
    siblings = sorted(
        item.name for item in path.parent.glob("optical_path_probe.*") if item != path
    )
    if siblings:
        details.append("同目录: " + ", ".join(siblings))
    st.caption(" · ".join(details))
    png = path.with_name("optical_path_probe.png")
    if png.is_file():
        st.image(str(png), caption="曝光-峰值 / 离焦响应")


def _render_probe_outcome(outcome: Mapping[str, Any]) -> None:
    """把一次探测的三份结果铺出来（失败只铺一条可执行的错误）。"""
    error = outcome.get("error")
    if error:
        st.error(str(error))
        st.caption(
            "设备被占时先确认没有别的会话在跑；光路本身没通时按 CLI 同一条链排查"
            "（`python -m ao_shaping.tools.slm.panel.diagnose`）。"
        )
        return
    payload = outcome.get("payload")
    if not isinstance(payload, Mapping):
        return
    _render_probe_verdict(payload)
    _render_probe_exposure(payload, outcome)
    _render_probe_evidence(payload, outcome)


def render_optical_probe(
    slm_num: int | None,
    prefix: str,
    probe_readings: ProbeReadings = None,
) -> None:
    """「一键光路快速探测」: 四态 + 建议曝光（写回）+ 证据目录。

    Args:
        slm_num: 已连接的 SLM 编号；``None`` 表示本页还没有设备（按钮 disabled）。
        prefix: 按钮 key 的来源（本页是 ``slm1`` 之类）。
        probe_readings: 读数来源，见 :data:`ProbeReadings`。缺省实跑 CLI；
            注入 callable / 路径是为了让这套渲染能在无设备环境里被测试。
    """
    st.markdown("#### 光路快速探测")
    st.caption(
        "一次给出三份实测结果：① 光路四态（含噪声底 / 峰值）② 建议曝光（并写回"
        "侧栏探测值）③ 证据落盘目录。与 CLI "
        "`python scripts/slm/probe_optical_path_validity.py` 同一份采集与同一份判定"
        "（`tools/slm/panel/probe_policy.decide`），本页只渲染、不重算。"
    )
    blocked = _probe_blocked_reasons(slm_num)
    clicked = st.button(
        "一键光路快速探测",
        key=f"{prefix or 'bench'}_optical_probe",
        type="secondary",
        disabled=bool(blocked),
        use_container_width=True,
        help="按当前光路实测：8 档曝光阶梯 + 离焦 ±1 rad 响应，约 1-2 分钟，"
        "期间独占 SLM 与相机。",
    )
    if blocked:
        st.caption("暂不可用：" + "；".join(blocked))

    if clicked and not blocked:
        outcome = _collect_probe_outcome(int(slm_num), probe_readings)
        st.session_state[_PROBE_OUTCOME_KEY] = outcome
        _render_probe_outcome(outcome)
        if outcome.get("written_ms") is not None:
            # 排队值必须在下一次 flush（控件重新实例化）之前落下 —— 一次性
            # rerun，不是定时刷新。
            st.rerun()
        return

    outcome = st.session_state.get(_PROBE_OUTCOME_KEY)
    if not outcome:
        st.caption("尚无探测结果。")
        return
    _render_probe_outcome(outcome)


def render_parameters(method: TransferMethod, layout: _Layout) -> dict[str, Any]:
    """目标与预算 → 高级参数，并并入侧栏光路值（光路控件只在侧栏）。"""
    with layout:
        values = method.render_inputs()
    bench_settings.merge_bench_values(
        values, method.prefix, method.slm_num, method.key
    )
    method.hint_bench_calibration(values)
    return values


def render_validation(
    method: TransferMethod, values: dict[str, Any], layout: _Layout
) -> list[str]:
    """校验区必须在按钮之前，按钮才能 ``disabled``。"""
    errors: list[str] = []
    bench_error = bench_settings.validate_bench_inputs()
    if bench_error:
        errors.append(bench_error)
    method_error = method.validate(values)
    if method_error:
        errors.append(method_error)
    with layout:
        for error in errors:
            st.error(error)
    return errors


def render_run_controls(
    method: TransferMethod, errors: list[str], layout: _Layout
) -> tuple[bool, bool]:
    """调试开关 + 运行按钮。返回 ``(clicked, debug)``。"""
    with layout:
        debug = st.checkbox(
            "保存调试产物 (data/debug/…)",
            value=False,
            key=f"{method.prefix}_shp_debug",
            disabled=bool(errors),
            help="与 CLI --debug 同布局；迁移学习的证据目录不受它影响 —— "
            "该管线从第一阶段就写 journal/audit，没有「不保存」模式",
        )
        clicked = st.button(
            method.action,
            key=f"{method.prefix}_shp_run_{method.key}",
            type="primary",
            disabled=bool(errors) or bench_settings.bench_busy(),
            use_container_width=True,
        )
    return bool(clicked), bool(debug)


def render_session_progress() -> None:
    """最近一次运行的进度明细与三张轨迹图（读会话目录，不重算指标）。

    运行期间进度由 ``TransferMethod`` 自己的进度条 + 明细面板实时刷新；本函数
    只在**运行结束后**把同一条会话目录的最终状态与轨迹再排一遍，所以运行中直接
    跳过，否则同一份明细会在页面上出现两次。
    """
    raw = st.session_state.get(_SESSION_DIR_KEY)
    if not raw or bench_settings.bench_busy():
        return
    out = Path(raw)
    st.subheader("运行进度")
    render_progress(collect_progress(out), running=False)
    traces = collect_traces(out)
    render_traces(traces)
    if not any(traces.get(key) for key in ("ccd", "fit", "prediction", "blocks")):
        # 一张图都排不出来时 ``render_traces`` 早退（没有「轨迹」小标题），
        # 页面仍要说清是这个目录还没写完，而不是整块区域凭空消失。
        st.caption(
            "该会话目录尚无轨迹产物：" + "、".join(missing_trace_labels(traces)) + "。"
        )


# --------------------------------------------------------------------------- #
# 侧栏布局选择
# --------------------------------------------------------------------------- #
def _subheader_options() -> list[str]:
    """「页面布局」的选项：默认主区，或挂到专用子标题折叠面板。"""
    return [_PLAIN_LAYOUT, "专用子标题（折叠面板）"]


def render_subheader_selector() -> None:
    """侧栏「页面布局」：把主体挂进 ``st.status`` 折叠面板，或留在主区。

    实现与 ``multi_slm_controller`` 的专用子标题一致（``st.status`` +
    会话态句柄），但句柄存在**本页自己的 key** 上，不会串到别的页面。
    交互控件在折叠面板内照常可用。
    """
    _render_layout_selector(
        _SUBHEADER_OPTION_KEY, _SUBHEADER_BOX_KEY, PAGE_TITLE
    )


# --------------------------------------------------------------------------- #
# 主体编排
# --------------------------------------------------------------------------- #
def _render_body(layout: _Layout, probe_readings: ProbeReadings = None) -> None:
    """主体内容：状态条 → 光路探测 → 参数 → 校验 → 运行 → 进度。"""
    connected = _connected_slm_numbers()
    if not connected:
        with layout:
            render_connection_hint()
            # 没设备也要把按钮画出来（disabled + 原因）：操作员才能看到「差在哪」
            # 而不是整段消失。
            render_optical_probe(None, "bench", probe_readings)
        return

    slm_num = connected[0]
    prefix = f"slm{slm_num}"

    with layout:
        render_status_header(slm_num, prefix)
        render_optical_probe(slm_num, prefix, probe_readings)

    method = TransferMethod(prefix, slm_num)
    values = render_parameters(method, layout)
    errors = render_validation(method, values, layout)
    clicked, debug = render_run_controls(method, errors, layout)

    if clicked:
        # 计划目录要在 ``run`` 之前拿到：阶段中途失败时 ``run`` 返回 ``None``，
        # 事后无从得知证据写在哪，而失败阶段恰恰最需要看产物。
        planned = method.build_params(values).out
        st.session_state[_SESSION_DIR_KEY] = str(planned)
        method.run(values, debug=debug)
        return

    with layout:
        render_session_progress()
        if not bench_settings.effective_state(prefix):
            st.caption("尚无运行结果。")
        st.caption(snapshot_summary_line(prefix))


def main() -> None:
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon="🧮",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _ensure_state()

    with st.sidebar:
        # 「CCD参数设置」已在 render_slm_sidebar() 内部渲染一次 —— 再调
        # render_bench_settings() 会重复声明 bench_cam_type 等 key 让页面崩掉。
        render_slm_sidebar()
        render_subheader_selector()

    _render_body(_layout(), st.session_state.get(_PROBE_READINGS_KEY))


if __name__ == "__main__":
    main()
