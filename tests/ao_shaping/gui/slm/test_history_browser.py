"""历史会话浏览器 / 光路指纹推荐卡的渲染契约（无设备、无真实数据目录）。

这一层只排版，所以测试也只锁排版的三件真事：

1. **空输入不炸、说人话** —— 没有候选时是一句 caption 加 ``None``，不是异常；
2. **推荐与拒绝都要看得见** —— 推荐卡的分数与逐维度说明必须渲染出来；被拒候选
   带着原因出现在选项里，且「应用」按钮必须是禁用的（不能点下去才在后端炸）；
3. **历史列表可点选** —— 选中某一行要把它**自己的**目录返回给调用方，``stale``
   条目只打徽标、不从列表里消失。

依赖层（``checkpoint_registry`` / ``optical_session``）在这几条测试里被替换成确定
的假实现：渲染层不该靠真实打分规则来证明自己排版正确，反过来也不该因为另一条
lane 调整权重而变红。
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest
from streamlit.testing.v1 import AppTest

from ao_shaping.gui.slm import history_browser as hb

# --------------------------------------------------------------------------- #
# 夹具：一个只调用一个 renderer 的最小 harness 页
# --------------------------------------------------------------------------- #
HARNESS = """\
import streamlit as st
from ao_shaping.gui.slm import history_browser as hb

mode = st.session_state["hb_test_mode"]
if mode == "card":
    hb.render_fingerprint_card(st.session_state["hb_test_fp"])
elif mode == "load":
    applied = []
    result = hb.render_load_panel(
        current_fp=st.session_state.get("hb_test_current"),
        candidates=st.session_state.get("hb_test_candidates") or [],
        on_apply=applied.append,
    )
    st.session_state["hb_test_applied"] = [item.get("path") for item in applied]
    st.session_state["hb_test_result"] = None if result is None else result.get("path")
else:
    picked = hb.render_history_browser(root=st.session_state["hb_test_root"])
    st.session_state["hb_test_picked"] = None if picked is None else str(picked)
"""

FP: dict[str, Any] = {
    "created_at": "2026-10-27T14:03:11.123456+00:00",
    "window_span": 420.0,
    "window_step": 40.0,
    "zero_xy": (1442.0, 706.0),
    "exposure_ms": 0.4,
    "camera_format": "Mono12",
    "control_grid": (256, 256),
}


def _page(mode: str, **state: Any) -> AppTest:
    """跑一次 harness：``hb_test_<name>`` 就是 harness 读的 state 键。"""
    page = AppTest.from_string(HARNESS, default_timeout=60)
    page.session_state["hb_test_mode"] = mode
    for name, value in state.items():
        page.session_state[f"hb_test_{name}"] = value
    page.run()
    return page


def _frame_text(page: AppTest) -> str:
    """页面上所有 ``st.dataframe`` 的文本（卡片与推荐表都是 dataframe）。"""
    return " ".join(str(frame.value) for frame in page.dataframe)


def _captions(page: AppTest) -> str:
    return " ".join(element.value for element in page.caption)


def _candidate(path: str, *, created: str, cv: float, quality: float, shape: str = "square") -> dict:
    return {
        "path": path,
        "kind": "checkpoint",
        "shape": shape,
        "created_at": created,
        "cv": cv,
        "fit_quality": quality,
        "sha256": "c" * 64,
        "fingerprint": {},
        "stale": False,
    }


def _verdict(entries: list[dict], *, recommended: int = 1, rejected: int = 0,
             score: float | None = 0.83) -> dict[str, Any]:
    """推荐层的确定假返回：一条被拒（0 号）、一条推荐（``recommended``）。"""
    reasons = ("零级位移 120 px 超过 50 px",)
    ranked = []
    for index, entry in enumerate(entries):
        blocked = index == rejected
        ranked.append(
            {
                "candidate": dict(entry),
                "score": score if index == recommended else 0.21,
                "dimensions": [
                    {"name": "窗口", "weight": 0.4, "value": 1.0, "reason": "窗口跨度与周期一致"},
                    {
                        "name": "零级",
                        "weight": 0.6,
                        "value": 0.72 if not blocked else 0.10,
                        "reason": "零级偏差 3 px" if not blocked else "零级偏差 120 px",
                    },
                ],
                "rejected": blocked,
                "reasons": list(reasons) if blocked else [],
            }
        )
    return {
        "recommended": ranked[recommended],
        "ranked": ranked,
        "threshold": {"min_score": 0.6, "max_origin_shift_px": 50.0},
    }


# --------------------------------------------------------------------------- #
# 1. 指纹卡
# --------------------------------------------------------------------------- #
def test_fingerprint_card_writes_a_dash_for_every_missing_field() -> None:
    """空指纹也要能看：六行齐全、值是「—」，不是异常也不是 0。"""
    page = _page("card", fp={})
    assert not page.exception
    text = _frame_text(page)
    for label in ("指纹时间", "窗口/周期", "零级位置", "曝光", "相机格式", "控制分辨率"):
        assert label in text
    assert text.count(hb.MISSING) == 6
    assert "本指纹用于与历史检查点比对" in _captions(page)


def test_fingerprint_card_renders_the_six_fields() -> None:
    """齐全的指纹按约定格式落位（窗口/周期、零级、曝光、相机格式、控制分辨率）。"""
    page = _page("card", fp=FP)
    assert not page.exception
    text = _frame_text(page)
    assert "2026-10-27 14:03" in text
    assert "420 px / 步长 40 px" in text
    assert "(1442, 706) px" in text
    assert "0.4 ms" in text
    assert "Mono12" in text
    assert "256 × 256" in text
    assert hb.MISSING not in text


# --------------------------------------------------------------------------- #
# 2. 推荐与装载
# --------------------------------------------------------------------------- #
def test_load_panel_without_candidates_says_so_and_returns_none() -> None:
    """没有候选：一句 caption + ``None``，不画推荐卡也不画按钮。"""
    page = _page("load", candidates=[])
    assert not page.exception
    assert "未找到可用的历史检查点" in _captions(page)
    assert page.session_state["hb_test_result"] is None
    assert [element.key for element in page.button] == []


def test_recommended_candidate_renders_score_and_dimension_breakdown(monkeypatch) -> None:
    """推荐卡必须把分数与逐维度 breakdown 都摆出来（否则"为什么推荐它"没法核对）。"""
    entries = [
        _candidate("old/a", created="2026-10-27T12:00:00Z", cv=0.52, quality=0.61),
        _candidate("old/b", created="2026-10-27T14:03:00Z", cv=0.39, quality=0.83),
    ]
    monkeypatch.setattr(hb, "recommend_load", lambda current, candidates: _verdict(entries))
    page = _page("load", candidates=entries, current=FP)
    assert not page.exception
    assert any("0.83" in element.value for element in page.markdown)
    assert "窗口跨度与周期一致" in _frame_text(page)
    assert "零级偏差 3 px" in _frame_text(page)
    # 默认选中推荐的那一条。
    assert page.selectbox(key="hb_load_choice").value == "2026-10-27 14:03 · square · CV 0.39 · 质量 0.83"


@pytest.mark.parametrize(("score", "sentence"), [
    (0.85, "推荐（可直接应用）"),
    (0.70, "谨慎（可用，但请核对指纹差异后再应用）"),
    (0.50, "不推荐（低于可用阈值）"),
])
def test_recommendation_line_carries_the_shared_verdict_sentence(
        monkeypatch, score: float, sentence: str) -> None:
    """推荐行 = 分数 + 三档结论。档名与阈值都来自 ``optical_session``，本页只翻一句话。

    三个分数按编排口径取（0.85 / 0.70 / 0.50）：结论词必须跟着分档走，否则
    「推荐（可直接应用）」会出现在一条其实只是勉强可用的检查点上。
    """
    entries = [
        _candidate("old/a", created="2026-10-27T12:00:00Z", cv=0.52, quality=0.61),
        _candidate("old/b", created="2026-10-27T14:03:00Z", cv=0.39, quality=0.83),
    ]
    monkeypatch.setattr(hb, "recommend_load",
                        lambda current, candidates: _verdict(entries, score=score))
    page = _page("load", candidates=entries, current=FP)
    assert not page.exception
    line = next(element.value for element in page.markdown if "推荐检查点" in element.value)
    assert f"兼容分数 **{score:.2f}** —— {sentence}" in line


def test_recommendation_line_without_a_score_states_no_verdict(monkeypatch) -> None:
    """无分数（没给当前光路指纹）⇒ 只写「兼容分数 —」，三档结论一个词都不出现。

    给未知值硬安一个结论，两种误读都要命：写成「不推荐」会让操作员白核对一遍
    指纹，写成「推荐」则直接跳过核对。
    """
    entries = [
        _candidate("old/a", created="2026-10-27T12:00:00Z", cv=0.52, quality=0.61),
        _candidate("old/b", created="2026-10-27T14:03:00Z", cv=0.39, quality=0.83),
    ]
    monkeypatch.setattr(hb, "recommend_load",
                        lambda current, candidates: _verdict(entries, score=None))
    page = _page("load", candidates=entries, current=FP)
    assert not page.exception
    line = next(element.value for element in page.markdown if "推荐检查点" in element.value)
    assert f"兼容分数 **{hb.MISSING}**" in line
    assert " —— " not in line
    for sentence in hb.VERDICT_SENTENCES.values():
        assert sentence not in line


def test_verdict_sentence_table_covers_every_tier_of_the_shared_function() -> None:
    """三档档名是 ``compatibility_verdict`` 的词：表必须逐档有句子（改档名会红）。

    这是「结论只有一个来源」的那一半守卫 —— 表与函数各写一套档名时，缺键只会
    静默丢掉结论（推荐行退回只有分数），而那正是这条链要避免的失效方式。
    """
    from ao_shaping.tools.slm.transfer.optical_session import compatibility_verdict

    tiers = {compatibility_verdict(1.0), compatibility_verdict(0.7), compatibility_verdict(0.0)}
    assert set(hb.VERDICT_SENTENCES) == tiers


def test_page_asks_the_shared_verdict_and_keeps_no_bare_threshold() -> None:
    """源码级守卫：本页不许出现 ``0.6`` / ``0.8`` 这类裸阈值，结论只许问函数。

    一旦有人在这里写 ``if score > 0.8``，推荐行就跟 ``compatibility_verdict``
    分叉了 —— 而分叉出来的那一档正是「可不经确认直接应用」。用 AST 扫 ``(0, 1)``
    的浮点常量：本页只有分数/权重这个量级，任何裸阈值都落在区间内，文案里的
    ``0.87`` 之类字符串常量不会误报。
    """
    tree = ast.parse(Path(hb.__file__).read_text(encoding="utf-8"))
    stray = [node.value for node in ast.walk(tree)
             if isinstance(node, ast.Constant) and isinstance(node.value, float)
             and 0.0 < node.value < 1.0]
    assert not stray, f"推荐行出现了裸阈值字面量 {stray}：档位应问 compatibility_verdict"
    calls = {node.func.id for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert "compatibility_verdict" in calls, \
        "推荐行的结论必须来自 optical_session.compatibility_verdict（不是本地判断）"


def test_rejected_candidate_carries_its_reason_and_cannot_be_applied(monkeypatch) -> None:
    """被拒候选仍出现在选项里（标原因），但「应用」按钮禁用——不是点下去才失败。"""
    entries = [
        _candidate("old/a", created="2026-10-27T12:00:00Z", cv=0.52, quality=0.61),
        _candidate("old/b", created="2026-10-27T14:03:00Z", cv=0.39, quality=0.83),
    ]
    monkeypatch.setattr(hb, "recommend_load", lambda current, candidates: _verdict(entries))
    page = _page("load", candidates=entries, current=FP)

    blocked = [option for option in page.selectbox(key="hb_load_choice").options if "不兼容" in option]
    assert len(blocked) == 1 and "零级位移 120 px 超过 50 px" in blocked[0]
    assert "零级位移 120 px 超过 50 px" in _frame_text(page)      # 被拒原因表

    page.selectbox(key="hb_load_choice").select(blocked[0]).run()
    assert not page.exception
    assert page.button(key="hb_load_apply").proto.disabled is True
    assert "该检查点不可应用" in _captions(page)
    assert page.session_state["hb_test_result"] is None


def test_apply_returns_the_candidate_and_calls_the_callback(monkeypatch) -> None:
    """点「应用该检查点」：先回调 ``on_apply``，再把同一个候选 dict 返回给页面。"""
    entries = [
        _candidate("old/a", created="2026-10-27T12:00:00Z", cv=0.52, quality=0.61),
        _candidate("old/b", created="2026-10-27T14:03:00Z", cv=0.39, quality=0.83),
    ]
    monkeypatch.setattr(hb, "recommend_load", lambda current, candidates: _verdict(entries))
    page = _page("load", candidates=entries, current=FP)
    page.button(key="hb_load_apply").click().run()
    assert not page.exception
    assert page.session_state["hb_test_applied"] == ["old/b"]
    assert page.session_state["hb_test_result"] == "old/b"


# --------------------------------------------------------------------------- #
# 3. 历史浏览
# --------------------------------------------------------------------------- #
def _history_entry(path: Path, *, created: str, stale: bool = False, shape: str = "square") -> dict:
    return {
        "path": str(path),
        "kind": "checkpoint",
        "shape": shape,
        "created_at": created,
        "cv": 0.39,
        "fit_quality": 0.83,
        "sha256": "d" * 64,
        "fingerprint": dict(FP),
        "stale": stale,
    }


def test_history_browser_returns_the_selected_session_directory(monkeypatch, tmp_path) -> None:
    """「选用该会话」返回的是**那一行自己的**目录，不是列表里的第一个。"""
    root = tmp_path / "slm_transfer"
    first = root / "s1200"
    second = root / "s600"
    entries = [
        _history_entry(first, created="2026-10-27T14:03:00Z"),
        _history_entry(second, created="2026-10-26T09:10:00Z"),
    ]
    monkeypatch.setattr(hb, "scan_checkpoint_index", lambda root, **_: entries)
    page = _page("history", root=root)
    assert not page.exception
    assert page.session_state["hb_test_picked"] is None

    page.button(key="hb_use_1").click().run()
    assert not page.exception
    assert page.session_state["hb_test_picked"] == str(second)


def test_history_browser_keeps_stale_entries_visible_with_a_badge(monkeypatch, tmp_path) -> None:
    """``stale`` 只打徽标：条目仍在列表里（2/2），provenance 里写明它已过期。"""
    root = tmp_path / "slm_transfer"
    entries = [
        _history_entry(root / "fresh", created="2026-10-27T14:03:00Z"),
        _history_entry(root / "old", created="2026-10-20T08:00:00Z", stale=True),
    ]
    monkeypatch.setattr(hb, "scan_checkpoint_index", lambda root, **_: entries)
    page = _page("history", root=root)
    assert not page.exception
    assert "2 / 2 个检查点" in _captions(page)
    marks = " ".join(element.value for element in page.markdown)
    assert "已过期（stale）" in marks
    # ``st.badge`` 是彩色 badge 的 markdown 包装，所以它在 markdown 元素里而不是单独一类。
    assert "已过期" in marks and "badge" in marks
    assert [str(element.label) for element in page.expander if "已过期" in str(element.label)]


def test_history_browser_reports_an_empty_root_instead_of_failing(monkeypatch, tmp_path) -> None:
    """空目录：一句 caption，返回 ``None``，页面不报错。"""
    monkeypatch.setattr(hb, "scan_checkpoint_index", lambda root, **_: [])
    page = _page("history", root=tmp_path / "nothing")
    assert not page.exception
    assert "还没有历史检查点" in _captions(page)
    assert page.session_state["hb_test_picked"] is None


def test_default_root_follows_the_campaign_convention(monkeypatch, tmp_path) -> None:
    """缺省根目录沿用 ``campaign_root()`` 的字面量，旧布局只做回落、不自造第三个。"""
    monkeypatch.chdir(tmp_path)
    assert hb._resolve_root(None) == hb.FALLBACK_ROOT

    (tmp_path / hb.DEFAULT_ROOT).mkdir(parents=True)
    assert hb._resolve_root(None) == hb.DEFAULT_ROOT
    assert hb._resolve_root(tmp_path / "explicit") == tmp_path / "explicit"
    assert hb.DEFAULT_ROOT == Path("data") / "debug" / "slm_transfer"
