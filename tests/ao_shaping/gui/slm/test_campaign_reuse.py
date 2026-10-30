"""⑦ 的模型复用面板（``gui/slm/campaign_reuse.py``）：指纹读哪一份、面板画什么、怎么降级。

本文件只测**这块自己**：``campaign_fingerprint`` 的三条取值路径、``has_fingerprint_readings``
的判据、``render_reuse_panel`` 的卡与候选、``history_browser`` 缺失时的降级，以及
「阈值 / 打分只有 ``optical_session`` 一份」这条源码守卫（目标文件因此是本模块，而不是
``multishape_campaign_ui.py`` —— 搬走之后那条守卫就得跟着搬，否则它守的是一个已经没有
这段代码的文件）。

整页 ⑦ 里确实出现了这块（而不是只有独立渲染体时才有）留在
``test_multishape_campaign_ui.py`` 的集成用例里：那一条测的是「⑦ 会把复盘读数与复用面板
串在一起」，本文件测的是串在一起的那一块本身。
"""

from __future__ import annotations

import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

from ao_shaping.gui.slm import campaign_reuse

#: 只调一个渲染体的最小 harness：指纹卡与候选列表都在 ``render_reuse_panel`` 里面。
REUSE_HARNESS = """\
import streamlit as st
from pathlib import Path
from ao_shaping.gui.slm import campaign_reuse as ui

ui.render_reuse_panel(Path(st.session_state["msc_reuse_dir"]))
"""

RECORDED_FP: dict = {
    "created_at": "2026-10-27T14:03:11.123456+00:00",
    "window_span": 420.0,
    "window_step": 20.0,
    "zero_xy": [1442.0, 706.0],
    "zero_radius": 30.0,
    "exposure_ms": 5.0,
    "camera_format": "Mono12",
    "control_grid": [256, 256],
    "pixel_size_um": 3.45,
}


def _write_recorded_fingerprint(campaign: Path, **overrides) -> None:
    """会话目录里那份已落盘的 ``fingerprint.json``（采集侧写的，⑦ 只读）。"""
    payload = {**RECORDED_FP, **overrides}
    (campaign / "fingerprint.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_checkpoint(campaign: Path, *, created: str = "2026-10-27T14:03:00Z") -> None:
    """一个 ``scan_checkpoint_index`` 扫得到的会话：journal ＋ 冻结参数。"""
    campaign.mkdir(parents=True, exist_ok=True)
    (campaign / "journal.json").write_text(json.dumps([
        {"stage": "plan", "plan": {"shapes": ["square"]}, "timestamp_utc": created}],
    ), encoding="utf-8")
    (campaign / "frozen_parameters.json").write_text(json.dumps({
        "control_grid": [256, 256], "created_at": created}), encoding="utf-8")


def _write_blocks(campaign: Path, shape: str, stage: str, *, blocks: int) -> None:
    """写一个阶段的 ``manifest.json``（段首行 + 逐块行，每块一个 CV）。"""
    stage_dir = campaign / "shape" / shape / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    common = {"core_peak": 200.0, "exposure_ms": 40.0, "saturation_pixels": 0,
              "bit_depth": 12, "full_scale": 4095.0}
    rows = [{"label": f"{shape}_seed", "file": f"{shape}_seed.npz",
             "metrics": {"cv": 1.0, "mean": 1.0, "ee_roi": 0.4, "target_peak": 1.0},
             **common}]
    for index in range(blocks):
        rows.append({"label": f"{shape}_{index}_A1", "file": f"{shape}_{index}.npz",
                     "metrics": {"cv": 0.90 - 0.01 * index, "mean": 1.0,
                                 "ee_roi": 0.4, "target_peak": 1.0},
                     **common})
    (stage_dir / "manifest.json").write_text(json.dumps(rows), encoding="utf-8")


def _reuse_page(directory: Path) -> AppTest:
    page = AppTest.from_string(REUSE_HARNESS, default_timeout=60)
    page.session_state["msc_reuse_dir"] = str(directory)
    page.run()
    return page


def _columns(value) -> list[str]:
    """表格的列名（``st.dataframe`` 在 AppTest 里是 DataFrame；别的元素没有列）。"""
    try:
        return [str(name) for name in value.columns]
    except AttributeError:
        return []


def _fingerprint_tables(page: AppTest) -> list:
    """指纹卡的六行表（列名固定为 ``项目`` / ``值``）。"""
    return [element.value for element in page.dataframe
            if "项目" in _columns(element.value)]


def test_campaign_fingerprint_prefers_the_recorded_fingerprint_file(tmp_path: Path) -> None:
    """有 ``fingerprint.json`` 就直接读它 —— 采集时记下的才是那一次的读数。"""
    campaign = tmp_path / "stamp" / "s1200"
    _write_checkpoint(campaign)
    _write_recorded_fingerprint(campaign, exposure_ms=7.5)
    fingerprint = campaign_reuse.campaign_fingerprint(campaign)
    assert fingerprint is not None
    assert fingerprint["window_span"] == 420.0
    assert fingerprint["exposure_ms"] == 7.5
    assert campaign_reuse.has_fingerprint_readings(fingerprint) is True


def test_campaign_fingerprint_recomputes_from_artifacts_and_never_writes(tmp_path: Path) -> None:
    """没有 ``fingerprint.json`` 就按目录里已有产物现场算，且**只读**（不落盘）。"""
    campaign = tmp_path / "stamp" / "s1200"
    _write_blocks(campaign, "square", "mono12_cv", blocks=2)   # manifest: peak / exposure_ms
    (campaign / "journal.json").write_text(json.dumps([
        {"stage": "plan", "plan": {"shapes": ["square"], "window_span": 620.0,
                                  "window_step": 20.0, "control_grid": [256, 256]},
         "timestamp_utc": "2026-10-28T09:00:00Z"}],
    ), encoding="utf-8")

    fingerprint = campaign_reuse.campaign_fingerprint(campaign)
    assert fingerprint is not None
    assert fingerprint["window_span"] == 620.0           # 来自 journal 的 plan
    assert fingerprint["exposure_ms"] == 40.0            # 来自阶段 manifest
    assert fingerprint["camera_format"] is None          # 缺字段写 None，不猜
    assert not (campaign / "fingerprint.json").exists(), "⑦ 是回看历史，不许写指纹"


def test_campaign_fingerprint_on_an_empty_directory_has_no_readings(tmp_path: Path) -> None:
    """空目录：算得出一份指纹对象，但里面除 ``created_at`` 全空 ⇒ 页面写「未记录」。"""
    empty = tmp_path / "empty"
    empty.mkdir()
    fingerprint = campaign_reuse.campaign_fingerprint(empty)
    assert fingerprint is not None
    assert fingerprint["created_at"]                  # compute 现场补的时间戳
    assert campaign_reuse.has_fingerprint_readings(fingerprint) is False
    assert [key for key, value in fingerprint.items()
            if value is not None] == ["created_at"]


def test_reuse_panel_shows_the_card_and_feeds_the_compat_filter(tmp_path: Path) -> None:
    """有指纹 + 有检查点：卡画出来、指纹投给历史浏览器的兼容过滤、候选照常列出。"""
    campaign = tmp_path / "stamp" / "s1200"
    _write_checkpoint(campaign)
    _write_recorded_fingerprint(campaign)

    page = _reuse_page(campaign)
    assert not page.exception, [str(exc.value)[:200] for exc in page.exception]
    tables = _fingerprint_tables(page)
    assert tables, "指纹卡没有渲染出来"
    assert [str(name) for name in tables[0]["项目"]] == ["指纹时间", "窗口/周期", "零级位置",
                                                        "曝光", "相机格式", "控制分辨率"]
    # 「当前光路指纹」由调用方投递 —— 没投时历史浏览器的兼容过滤是禁用的。
    fed = page.session_state["msc_hist_current_fp"]
    assert fed["window_span"] == 420.0 and fed["zero_xy"] == [1442.0, 706.0]
    captions = " ".join(element.value for element in page.caption)
    assert "个检查点" in captions, captions          # 候选来自历史浏览器内部的那次扫描


def test_reuse_panel_recomputes_the_card_when_no_fingerprint_file_exists(tmp_path: Path) -> None:
    """无指纹但有产物：照样画卡（读数来自现场计算），不逼人先去跑一次采集。"""
    campaign = tmp_path / "stamp" / "s1200"
    _write_blocks(campaign, "square", "mono12_cv", blocks=2)
    (campaign / "journal.json").write_text(json.dumps([
        {"stage": "plan", "plan": {"shapes": ["square"], "window_span": 900.0,
                                  "window_step": 20.0},
         "timestamp_utc": "2026-10-28T09:00:00Z"}],
    ), encoding="utf-8")

    page = _reuse_page(campaign)
    assert not page.exception, [str(exc.value)[:200] for exc in page.exception]
    assert _fingerprint_tables(page), "现场算出的指纹也应该画成卡"
    assert page.session_state["msc_hist_current_fp"]["window_span"] == 900.0


def test_reuse_panel_says_unrecorded_on_an_empty_directory(tmp_path: Path) -> None:
    """空目录：一行「未记录」＋ 历史浏览器的「还没有检查点」，都不许抛。"""
    empty = tmp_path / "empty"
    empty.mkdir()
    page = _reuse_page(empty)
    assert not page.exception, [str(exc.value)[:200] for exc in page.exception]
    assert not _fingerprint_tables(page), "全空的指纹不该排成一张全是「—」的卡"
    captions = " ".join(element.value for element in page.caption)
    assert "未记录" in captions
    assert "还没有历史检查点" in captions


def test_reuse_panel_degrades_to_one_caption_when_the_module_is_missing(
        tmp_path: Path, monkeypatch) -> None:
    """``history_browser`` 缺失 ⇒ 一行说明（与 ``campaign_monitor`` 的降级同一种写法）。"""
    import sys

    campaign = tmp_path / "stamp" / "s1200"
    _write_checkpoint(campaign)
    # 只留「模块没装」这唯一差别：sys.modules 置 None ＋ 抹掉包上的属性，
    # ``from ao_shaping.gui.slm import history_browser`` 于是真的抛 ImportError。
    monkeypatch.setitem(sys.modules, "ao_shaping.gui.slm.history_browser", None)
    monkeypatch.delattr(sys.modules["ao_shaping.gui.slm"], "history_browser", raising=False)
    assert campaign_reuse._history_browser() is None

    page = _reuse_page(campaign)
    assert not page.exception, [str(exc.value)[:200] for exc in page.exception]
    captions = " ".join(element.value for element in page.caption)
    assert "尚未就绪" in captions
    assert not _fingerprint_tables(page), "模块缺失时不该还画出半张卡"


def test_reuse_panel_keeps_the_threshold_and_the_score_in_optical_session_only() -> None:
    """阈值/权重只有 ``optical_session`` 一份：页面里不许有第二个 0.60/0.80，也不许自己算分。"""
    source = Path(campaign_reuse.__file__).read_text(encoding="utf-8")
    for literal in ("0.60", "0.80"):
        assert literal not in source, f"页面里出现了第二个阈值字面量 {literal}"
    for call in ("compatibility_score(", "recommend_load(", "scan_checkpoint_index("):
        assert call not in source, f"页面自己扫了一遍/算了一份：{call}"
