"""campaign 监控：逐块轨迹 / 段间闸 / 成本表 / 统一进度字幕。

这个模块的契约一半是**措辞**（进度只有一种写法、CV 越低越好、停止点在段边界），一半是
**降级**（``cost_table`` 与 ``campaign_control`` 与本模块并行落地，缺了页面上只能多一行
说明，不许抛异常）。两样都不该靠"点开整页 AppTest 再逐条匹配元素"来验证，所以:

- 数据层（:func:`collect_campaign_traces` / :func:`progress_caption`）直接单测，
  合成目录的写法照 ``tests/ao_shaping/tools/slm/transfer/test_campaign_progress.py``
  （那是 campaign 产物结构的唯一 fixture 口径）；
- 渲染层（三个 ``render_*``）用一页极小的 AppTest 脚本单独跑，这样"成本模块没落地"
  与"成本模块已经落地"两个分支都能测 —— 真页面要等到并行那一半合进来才走得到。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from ao_shaping.gui.slm import campaign_monitor as cm
from ao_shaping.tools.slm.transfer.campaign_progress import collect_campaign_progress

# --------------------------------------------------------------------------- #
# 合成产物（结构照 campaign_progress 的 fixture）
# --------------------------------------------------------------------------- #
def _row(label: str, cv: float | None, *, core_peak: float = 200.0,
         exposure_ms: float = 40.0, saturated: int = 0) -> dict:
    """一行 manifest。``cv is None`` 就是"这一块没写出 CV"（旧/半截产物）。"""
    row: dict = {
        "label": label,
        "file": f"{label}.npz",
        "core_peak": core_peak,
        "saturation_pixels": saturated,
        "exposure_ms": exposure_ms,
        "bit_depth": 12,
        "full_scale": 4095.0,
    }
    if cv is not None:
        row["metrics"] = {"cv": cv, "mean": 1.0, "ee_roi": 0.4, "target_peak": 1.0}
    return row


def _write_stage(root: Path, shape: str, stage: str, *, blocks: int, ccd_check: bool,
                 seed_cv: float = 1.0, final_cv: float = 0.5) -> Path:
    """写一个阶段目录（段首行 + 逐块行 + 段末行，行序与真实产物一致）。"""
    stage_dir = root / "shape" / shape / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    rows = [_row(f"{shape}_seed", seed_cv)]
    for index in range(blocks):
        rows.append(_row(f"{shape}_{index}_A1", 0.90 - 0.01 * index))
        if ccd_check:
            rows.append(_row(f"{shape}_{index}_B", 0.85 - 0.01 * index))
    rows.append(_row(f"{shape}_final", final_cv))
    (stage_dir / "manifest.json").write_text(json.dumps(rows), encoding="utf-8")
    return stage_dir


def _write_journal(root: Path, entries: list[tuple[str, str]],
                   stamps: list[str] | None = None,
                   plan: tuple[str, ...] | None = None) -> None:
    """最小 journal：``(stage, status)`` 列表，可选逐条 ``timestamp_utc`` 与 ``plan``。"""
    payload: list[dict] = []
    if plan:
        payload.append({"plan": {"shapes": list(plan)}})
    for index, (stage, status) in enumerate(entries):
        entry: dict = {"stage": stage, "status": status}
        if stamps is not None:
            entry["timestamp_utc"] = stamps[index]
        payload.append(entry)
    (root / "journal.json").write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture()
def campaign(tmp_path: Path) -> Path:
    """一个形状跑到第二段的一半：段首块序必须按**计划**块数累加。"""
    out = tmp_path / "multishape" / "s600"
    out.mkdir(parents=True)
    _write_stage(out, "square", "transport_cv", blocks=4, ccd_check=False)
    _write_stage(out, "square", "mono12_cv", blocks=3, ccd_check=True)
    _write_stage(out, "square", "mono12_continue", blocks=1, ccd_check=True)
    _write_journal(out, [("square/transport_cv", "complete"),
                         ("square/mono12_cv", "running"),
                         ("square/mono12_continue", "running")], plan=("square",))
    return out


# --------------------------------------------------------------------------- #
# 数据层：逐块轨迹
# --------------------------------------------------------------------------- #
def test_traces_use_planned_offsets_so_a_stop_mid_stage_never_shifts_later_blocks(
        campaign: Path) -> None:
    """段首块序 = 前面各段**计划**块数累加（4 / 8 / 12），不是"已读到几块"。

    拿已读块数累加，会把中途停下的段（本 fixture 的 ``mono12_cv`` 只写了 3 块）之后的
    所有点整体左移，于是轨迹图上"第几块开始变好"就成了假的。
    """
    rows = cm.collect_campaign_traces(campaign)
    assert [row["block"] for row in rows] == list(range(4)) + [4, 5, 6] + [12]
    assert {row["shape"] for row in rows} == {"square"}
    assert [row["stage"] for row in rows][:5] == ["transport_cv"] * 4 + ["mono12_cv"]


def test_traces_carry_exposure_and_core_peak_next_to_every_cv(campaign: Path) -> None:
    """CV 必须和曝光、核心峰值一起读：单看 CV 分不清"变均匀"与"变暗"。"""
    first = cm.collect_campaign_traces(campaign)[0]
    assert first["cv"] == pytest.approx(0.90)
    assert first["peak"] == pytest.approx(200.0)
    assert first["exposure_ms"] == pytest.approx(40.0)


def test_a_block_without_cv_is_none_and_kept_off_the_chart(tmp_path: Path) -> None:
    """缺 CV 的块给 ``None``，不补 0、不给 ``"—"``：``"—"`` 在图上没有位置。"""
    out = tmp_path / "partial"
    stage_dir = out / "shape" / "square" / "transport_cv"
    stage_dir.mkdir(parents=True)
    (stage_dir / "manifest.json").write_text(json.dumps([
        _row("square_0_A1", None),
        _row("square_1_A1", 0.42),
    ]), encoding="utf-8")

    rows = cm.collect_campaign_traces(out)
    assert [row["cv"] for row in rows] == [None, pytest.approx(0.42)]
    # 峰值与曝光仍是实测值（它们没缺），所以这一行只在 CV 上"不可画"。
    assert rows[0]["peak"] == pytest.approx(200.0)
    frame = cm._trace_frame(rows)
    assert list(frame.index) == [1]


def test_missing_directory_yields_an_empty_trace_list(tmp_path: Path) -> None:
    """目录还不存在（刚勾完计划、第一次还没点开始）不许抛。"""
    assert cm.collect_campaign_traces(tmp_path / "nope") == []


# --------------------------------------------------------------------------- #
# 数据层：统一进度字幕
# --------------------------------------------------------------------------- #
def test_progress_caption_has_one_format(campaign: Path) -> None:
    """``当前段 i/n · 块 k/m · 已用 hh:mm:ss`` —— 全页只有这一种写法。"""
    progress = collect_campaign_progress(campaign)
    text = cm.progress_caption(progress)
    assert re.fullmatch(r"当前段 \d+/\d+ · 块 \d+/\d+ · 已用 \d\d:\d\d:\d\d", text)


def test_progress_caption_denominator_comes_from_the_plan_not_the_directory(
        campaign: Path) -> None:
    """列出两个形状时，分母是 8 段 / 56 块，而不是目录里读到的 3 段。

    目录只留已经跑过的段，所以拿读取器的 ``*_total`` 当分母，跑到一半会诚实地写
    「3/3」—— 那正是这一行要修的东西（issue #215）。
    """
    progress = collect_campaign_progress(campaign)
    text = cm.progress_caption(progress, planned_shapes=("square", "triangle"))
    assert "当前段 1/8" in text
    assert "块 8/56" in text


def test_progress_caption_falls_back_to_the_journal_plan(campaign: Path) -> None:
    """页面没给计划时用 journal 的 ``plan.shapes``（这份目录实际按什么跑的）。"""
    _write_journal(campaign, [("square/transport_cv", "complete")], plan=("square", "star"))
    text = cm.progress_caption(collect_campaign_progress(campaign))
    assert "当前段 1/8" in text


def test_progress_caption_elapsed_comes_from_the_journal_clock(campaign: Path) -> None:
    """已用时间是 journal 首末墙钟差（跨重启也对），不是页面自己计的时。"""
    _write_journal(campaign, [("square/transport_cv", "complete"),
                              ("square/mono12_cv", "running")],
                   stamps=["2026-10-28T10:00:00.000000+00:00",
                           "2026-10-28T10:01:40.000000+00:00"],
                   plan=("square",))
    text = cm.progress_caption(collect_campaign_progress(campaign))
    assert text.endswith("已用 00:01:40")


def test_progress_caption_prefers_an_explicit_elapsed(campaign: Path) -> None:
    """显式给秒数时不再读 journal（页面用自己的计时器时不必造两份口径）。"""
    text = cm.progress_caption(collect_campaign_progress(campaign), elapsed_s=3725)
    assert text.endswith("已用 01:02:05")


def test_progress_caption_without_any_reading_still_reads_like_a_sentence(
        tmp_path: Path) -> None:
    """空目录也给出同一句话（全 0），而不是空字符串或异常。"""
    text = cm.progress_caption(collect_campaign_progress(tmp_path))
    assert text == "当前段 0/0 · 块 0/0 · 已用 00:00:00"


# --------------------------------------------------------------------------- #
# 渲染层：一页极小的脚本，逐个面板单独跑
# --------------------------------------------------------------------------- #
def _run(source: str) -> AppTest:
    """跑一段临时 Streamlit 脚本（只 import 本模块的那几个渲染函数）。"""
    at = AppTest.from_string(source, default_timeout=120)
    at.run()
    assert not at.exception, [str(exc.value) for exc in at.exception]
    return at


def _captions(at: AppTest) -> str:
    return "\n".join(element.value for element in at.caption)


def test_chart_is_a_single_vega_line_chart_with_a_caption_that_says_which_axis_is_what(
        campaign: Path) -> None:
    """一张折线图 + 一句"横轴是什么、CV 越低越好"的说明（空图会被读成"没有变化"）。"""
    at = _run(f"""
import streamlit as st
from ao_shaping.gui.slm import campaign_monitor
campaign_monitor.render_campaign_traces({str(campaign)!r})
""")
    assert len(at.get("vega_lite_chart")) == 1
    text = _captions(at)
    assert "段序·块序" in text
    assert "越低越好" in text
    # 逐像素/原始帧不许出现在这张图上：只有 CV 一个纵轴。
    assert "px" not in text


def test_chart_says_so_when_there_is_nothing_to_plot(tmp_path: Path) -> None:
    """一个 CV 都没有时印「尚无逐块数据」，不画空坐标轴。"""
    empty = tmp_path / "empty"
    empty.mkdir()
    at = _run(f"""
import streamlit as st
from ao_shaping.gui.slm import campaign_monitor
campaign_monitor.render_campaign_traces({str(empty)!r})
""")
    assert len(at.get("vega_lite_chart")) == 0
    assert "尚无逐块数据" in _captions(at)


def test_stage_gate_is_disabled_and_explains_itself_when_nothing_is_running(
        campaign: Path) -> None:
    """没有 campaign 在跑时按钮禁用：给他一个"点了但什么都不会发生"的按钮是骗人。"""
    at = _run(f"""
import streamlit as st
from ao_shaping.gui.slm import campaign_monitor
campaign_monitor.render_stage_gate({str(campaign)!r}, key_prefix="t", running=False)
""")
    gate = [button for button in at.button if button.key == "t_stage_pause"]
    assert len(gate) == 1
    assert gate[0].label == "在本段结束后暂停"
    assert gate[0].disabled is True
    assert "只在运行中有意义" in _captions(at)


def _make_unimportable(monkeypatch: pytest.MonkeyPatch, module_name: str) -> None:
    """把 ``ao_shaping.tools.slm.transfer.<module_name>`` 显式变成"不在盘上"。

    这两个并行模块（``campaign_control`` / ``cost_table``）随时会真的落盘，而这里要钉的
    是**模块缺失时页面怎么降级**，不是"它还交不上来"。所以不靠盘面状态，显式模拟：

    1. 抹掉包上的同名属性 —— 否则 ``from 包 import 名`` 直接取到属性，根本不走 import；
    2. 在 ``sys.modules`` 里塞 ``None`` —— 这是 CPython 的既有约定
       （``import of … halted; None in sys.modules``），import 必抛 ``ImportError``。

    两处改动都由 monkeypatch 在用例结束时还原，不污染后面的用例。
    """
    import sys

    import ao_shaping.tools.slm.transfer as package

    monkeypatch.delattr(package, module_name, raising=False)
    monkeypatch.setitem(sys.modules, f"{package.__name__}.{module_name}", None)


def test_stage_gate_downgrades_to_one_line_when_the_control_module_is_missing(
        campaign: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``campaign_control`` 缺失时：禁用按钮 + 一行说明，页面不许炸。

    「缺失」是显式模拟的（见 :func:`_make_unimportable`），所以模块真的在盘上时这条
    依然测得到降级分支。这一条同时钉住"不在这里写第二份实现"：模块缺失时本模块只
    降级，不自己读写某个"暂停请求"文件。
    """
    _make_unimportable(monkeypatch, "campaign_control")
    # 助手先走一遍：确认模拟确实让 import 失败（走的是 ``ImportError`` 那条分支）。
    assert cm._campaign_control() is None

    at = _run(f"""
import streamlit as st
from ao_shaping.gui.slm import campaign_monitor
campaign_monitor.render_stage_gate({str(campaign)!r}, key_prefix="t", running=True)
""")
    assert [button.disabled for button in at.button] == [True]
    assert "尚未就绪" in _captions(at)
    # 降级路径不许自己动手写请求文件（那是控制模块的唯一职责）。
    assert not (campaign / "stop_after_stage.request").exists()


def _install_fake_control(monkeypatch: pytest.MonkeyPatch, *, pending: bool) -> list:
    """装一个假的 ``campaign_control``：请求状态可变，调用按序记下。

    真实模块的 ``stage_stop_requested`` 是"读目录里有没有请求文件"，所以这个假实现也
    必须是**有状态**的：第一次渲染时按 ``pending`` 报告，点过「暂停」之后同一次会话里
    再渲染就该看到「取消」。固定返回 ``True`` 的假实现只能测到其中一个状态。
    """
    calls: list[tuple[str, str]] = []
    state = {"pending": pending}

    class _Control:
        @staticmethod
        def stage_stop_requested(out) -> bool:
            calls.append(("requested", str(out)))
            return state["pending"]

        @staticmethod
        def request_stage_stop(out) -> None:
            calls.append(("request", str(out)))
            state["pending"] = True

        @staticmethod
        def clear_stage_stop(out) -> None:
            calls.append(("clear", str(out)))
            state["pending"] = False

    import ao_shaping.tools.slm.transfer as package

    monkeypatch.setattr(package, "campaign_control", _Control, raising=False)
    return calls


@pytest.fixture()
def fake_control(monkeypatch: pytest.MonkeyPatch) -> list:
    """已有一个待处理的暂停请求（页面该显示「取消暂停请求」）。"""
    return _install_fake_control(monkeypatch, pending=True)


@pytest.fixture()
def fake_control_idle(monkeypatch: pytest.MonkeyPatch) -> list:
    """还没有暂停请求（页面该显示「在本段结束后暂停」并可以点）。"""
    return _install_fake_control(monkeypatch, pending=False)


def test_stage_gate_offers_cancel_when_a_request_is_pending(
        campaign: Path, fake_control: list) -> None:
    """已请求暂停时按钮变成「取消暂停请求」，并说清"本段跑完就停"。"""
    at = _run(f"""
import streamlit as st
from ao_shaping.gui.slm import campaign_monitor
campaign_monitor.render_stage_gate({str(campaign)!r}, key_prefix="t", running=True)
""")
    buttons = [button for button in at.button if button.key == "t_stage_cancel"]
    assert len(buttons) == 1
    assert buttons[0].label == "取消暂停请求"
    assert "本段跑完就停" in _captions(at)
    assert ("requested", str(campaign)) in fake_control


def test_stage_gate_request_button_writes_the_request(
        campaign: Path, fake_control_idle: list) -> None:
    """点「在本段结束后暂停」→ 调 ``request_stage_stop``（不碰 stop_event）。"""
    at = _run(f"""
import streamlit as st
from ao_shaping.gui.slm import campaign_monitor
campaign_monitor.render_stage_gate({str(campaign)!r}, key_prefix="t", running=True)
""")
    at.button(key="t_stage_pause").click().run()
    assert ("request", str(campaign)) in fake_control_idle
    # 请求是在本轮**渲染之后**写下的，所以按钮翻面要看下一次重绘：这一步同时验证
    # 写入不会触发 st.rerun（否则页面会在「暂停/取消」之间自激闪烁）。
    at.run()
    assert not at.exception
    assert len([button for button in at.button if button.key == "t_stage_cancel"]) == 1


def test_cost_panel_downgrades_to_one_line_when_the_cost_module_is_missing(
        campaign: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``cost_table`` 缺失时：一行说明，不编一张全 0 的表。

    缺失同样是显式模拟的；「模块在、但没有历史」另有一条测试说「首次运行」——
    两句话不能互相顶替。
    """
    _make_unimportable(monkeypatch, "cost_table")
    assert cm._cost_table() is None

    at = _run(f"""
import streamlit as st
from ao_shaping.gui.slm import campaign_monitor
campaign_monitor.render_cost_panel({str(campaign)!r}, planned_blocks=56, key_prefix="t")
""")
    assert not at.dataframe
    assert "尚未就绪" in _captions(at)


@pytest.fixture()
def fake_empty_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    """成本模块在，但目录里还没有任何历史（第一次跑）。"""

    class _Empty:
        @staticmethod
        def read_campaign_costs(_out) -> dict:
            return {}

        @staticmethod
        def estimate_campaign(_rows) -> dict:
            return {"low": 0.0, "mid": 0.0, "high": 0.0}

    import ao_shaping.tools.slm.transfer as package

    monkeypatch.setattr(package, "cost_table", _Empty, raising=False)


def test_cost_panel_says_first_run_when_there_is_no_history(
        campaign: Path, fake_empty_cost: None) -> None:
    at = _run(f"""
import streamlit as st
from ao_shaping.gui.slm import campaign_monitor
campaign_monitor.render_cost_panel({str(campaign)!r}, planned_blocks=56, key_prefix="t")
""")
    assert not at.dataframe
    assert "首次运行，无历史成本" in _captions(at)


def test_cost_table_columns_and_the_estimate_line(campaign: Path) -> None:
    """四列中文表 + 一行按计划块数折算的估算（低–高，括号里中位）。"""
    table = _FakeCostTable()
    rows = cm._cost_rows(table.read_campaign_costs(campaign))
    assert [list(row) for row in rows] == [list(cm._COST_COLUMNS)]
    assert rows[0]["段"] == "正方形/transport_cv"
    assert rows[0]["实测秒"] == pytest.approx(100.0)
    assert rows[0]["每块秒"] == pytest.approx(25.0)

    caption = cm._estimate_caption(table, table.read_campaign_costs(campaign), 56)
    assert caption == "预计 56 块 ≈ 00:10:00–00:20:00（中位 00:15:00）"
    # 没有计划块数就不估算 ——「预计 0 块」比不写更糟。
    assert cm._estimate_caption(table, table.read_campaign_costs(campaign), None) == ""


@pytest.mark.parametrize(("helper", "module_name"),
                         [("_campaign_control", "campaign_control"),
                          ("_cost_table", "cost_table")])
def test_helpers_refuse_a_module_that_is_present_but_lacks_the_api(
        monkeypatch: pytest.MonkeyPatch, helper: str, module_name: str) -> None:
    """真模块在盘上以后，唯一还可能缺的是**函数**（API 漂移）——那也必须是降级。

    空模块骗过 import，却骗不过 ``hasattr`` 守卫；这里钉住的正是那一步。
    """
    import types

    import ao_shaping.tools.slm.transfer as package

    monkeypatch.setattr(package, module_name, types.ModuleType(module_name), raising=False)
    assert getattr(cm, helper)() is None


def test_stage_gate_wires_the_real_control_module_to_the_directory(campaign: Path) -> None:
    """真模块在场时的闭环：点一次按钮，请求文件真的落在目录里；取消要真的删掉。"""
    from ao_shaping.tools.slm.transfer.campaign_control import STOP_AFTER_STAGE_FILE

    at = _run(f"""
import streamlit as st
from ao_shaping.gui.slm import campaign_monitor
campaign_monitor.render_stage_gate({str(campaign)!r}, key_prefix="t", running=True)
""")
    at.button(key="t_stage_pause").click().run()
    assert (campaign / STOP_AFTER_STAGE_FILE).exists()

    at.run()
    at.button(key="t_stage_cancel").click().run()
    assert not (campaign / STOP_AFTER_STAGE_FILE).exists()


def test_cost_panel_reads_a_real_recorded_stage_cost(campaign: Path) -> None:
    """真成本模块：段目录里的 ``cost.json`` → 一行表 + 一行按计划块数的估算。

    这条同时守住适配层与真返回形状的一致性 —— 估算那行必须收到成本模块**自己的**
    键（``seconds`` / ``blocks``），收到上面那张中文表就会算出全 0。
    """
    from ao_shaping.tools.slm.transfer.cost_table import record_stage_cost

    record_stage_cost(campaign / "shape" / "square" / "transport_cv",
                      stage="transport_cv", shape="square", seconds=100.0, blocks=4)
    at = _run(f"""
import streamlit as st
from ao_shaping.gui.slm import campaign_monitor
campaign_monitor.render_cost_panel({str(campaign)!r}, planned_blocks=56, key_prefix="t")
""")
    assert len(at.dataframe) == 1
    text = _captions(at)
    assert "预计 56 块" in text
    # 25 秒/块 × 56 块 = 23 分 20 秒；全 0 就说明估算喂错了形状。
    assert "00:23:20" in text


def test_cost_rows_accept_both_mapping_and_row_sequence() -> None:
    """成本模块给 ``{段: {...}}`` 或 ``[{...}]`` 都得认，否则页面要按它的返回形状改代码。"""
    from_map = cm._cost_rows({"square/mono12_cv": {"seconds": 60.0, "blocks": 8}})
    from_list = cm._cost_rows([{"stage": "square/mono12_cv", "elapsed_s": 60.0,
                                "n_blocks": 8}])
    assert from_map == from_list
    assert from_list[0]["每块秒"] == pytest.approx(7.5)


def test_cost_rows_tolerate_unknown_keys_and_missing_numbers() -> None:
    """不认识的键照印原名、缺的数字给 ``None``：同一列混排字符串会让整张表消失。"""
    rows = cm._cost_rows({"square/unknown_stage": {"seconds": 12.0}})
    assert rows[0]["段"] == "正方形/unknown_stage"
    assert rows[0]["块数"] is None
    assert rows[0]["每块秒"] is None


class _FakeCostTable:
    """假成本模块：两种调用约定各一个方法，用来验证适配层不依赖具体签名。"""

    @staticmethod
    def read_campaign_costs(_out) -> dict:
        return {"square/transport_cv": {"seconds": 100.0, "blocks": 4}}

    @staticmethod
    def estimate_campaign(_rows, _planned) -> dict:
        return {"low": 600.0, "mid": 900.0, "high": 1200.0}
