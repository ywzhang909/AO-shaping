"""campaign 操作台：无设备也要把计划、派生表、检查卡、历史全画出来。

本页的卖点是"形状 + 尺寸是唯一必填输入"，所以测试锁三件事：

1. **离线可用** —— 没接 SLM 时不抛异常、不报 error，计划矩阵 / 派生表 / 检查卡 /
   历史会话照常渲染（issue #215 的硬要求：没设备的机器上也能看一眼再决定）；
2. **派生真的随尺寸变** —— 换档位就得换采样窗（1200 → 900.0 px，600 → 620.0 px），
   否则"自动辅助层"只是一张写死的表；
3. **门禁在按钮上** —— 未连接 / 黄项未勾 / 红停项时「开始」必须禁用，而不是点下去
   才在后台线程里炸。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from ao_shaping.gui.slm import multishape_campaign_ui as ui

# AppTest.from_file 相对**调用文件**解析路径，所以这里按本测试位置拼绝对路径
# （仓库根 = parents[4]，与 tests/ao_shaping/gui/slm/test_diff_opt_controller.py 同款）。
APP = str(Path(__file__).resolve().parents[4] / "src" / "ao_shaping" / "gui" / "slm"
          / "multishape_campaign_ui.py")

SHAPE_KEYS = tuple(f"msc_shape_{shape}" for shape in ("square", "triangle", "star", "circle"))


@pytest.fixture()
def page() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    return at


def _settings_markdown(page: AppTest) -> str:
    """② 的派生表（HTML 表）所在的那段 markdown。"""
    for element in page.markdown:
        if "window_span" in (element.value or ""):
            return element.value
    raise AssertionError("派生表没有渲染出来")


def _row(table: str, name: str) -> str:
    """表里 ``name`` 那一行（``side`` 天生就是「手工」，所以只能逐行判来源）。"""
    start = table.index(f"<code>{name}</code>")
    return table[start:table.index("</tr>", start)]


def _columns(value) -> list[str]:
    """表格的列名（``st.dataframe`` 在 AppTest 里是 DataFrame；别的元素没有列）。"""
    try:
        return [str(name) for name in value.columns]
    except AttributeError:
        return []


def _matrix_frame(page: AppTest):
    """③ 的计划矩阵表（有「块数」列那张）。"""
    for element in page.dataframe:
        if "块数" in _columns(element.value):
            return element.value
    raise AssertionError("计划矩阵表没有渲染出来")


# --------------------------------------------------------------------------- #
# 1. 离线渲染
# --------------------------------------------------------------------------- #
def test_page_renders_offline_without_an_error(page: AppTest) -> None:
    assert not page.exception
    # 未连接时给的是 info（可执行提示），不是 error。检查卡的红停项走 st.error，
    # 那是**结论**不是故障，所以这里只挡"页面自己炸了"那种不以检查项开头的 error。
    assert any("未连接 SLM" in (element.value or "") for element in page.info)


def test_page_offers_the_four_campaign_shapes(page: AppTest) -> None:
    labels = {element.key for element in page.checkbox}
    assert set(SHAPE_KEYS) <= labels
    # rectangle 不在这份 16 段协议里：摆出来等于让操作员跑一个没有对照的 lane。
    assert "msc_shape_rectangle" not in labels


def test_page_renders_every_section_title(page: AppTest) -> None:
    titles = " ".join(element.value for element in page.subheader)
    for heading in ("① 目标", "② 自动辅助设置", "③ 计划预览", "④ 模型目录",
                    "⑤ 运行前检查", "⑥ 运行 / 逐块进度", "⑦ 历史会话"):
        assert heading in titles, heading


# --------------------------------------------------------------------------- #
# 2. 派生随尺寸变
# --------------------------------------------------------------------------- #
def test_derived_window_span_follows_the_default_side(page: AppTest) -> None:
    """1200 px 档的采样窗半跨是 900 px，且来源标成「派生」而不是「默认」。"""
    row = _row(_settings_markdown(page), "window_span")
    assert "900.0" in row
    assert "派生" in row


def test_switching_the_side_rederives_the_window_span(page: AppTest) -> None:
    page.session_state["msc_side_preset"] = 600
    page.run()
    assert not page.exception
    table = _settings_markdown(page)
    assert "620.0" in _row(table, "window_span")
    assert "900.0" not in _row(table, "window_span")


def test_derived_table_marks_the_manual_override(page: AppTest) -> None:
    """高级覆盖一开，该行的来源必须变「手工」，且说明里留着被替换掉的派生值。"""
    page.checkbox(key="msc_ov_on_window_span").check()
    page.run()
    assert not page.exception
    row = _row(_settings_markdown(page), "window_span")
    assert "手工" in row
    assert "900.0" in row            # 派生值仍可见（被谁换掉的要看得见）


def test_resetting_the_overrides_restores_the_derived_values(page: AppTest) -> None:
    """「按尺寸重新推导」必须真的回到派生值：开关复位、来源不再是「手工」。

    这条盯的是 widget 键的**时序**：带 key 的控件在本次运行里实例化之后就不能再写/
    删它的 ``st.session_state``，所以复位走「先记标记 → 下轮运行开头清键」两段式。
    """
    page.checkbox(key="msc_ov_on_window_span").check()
    page.run()
    assert "手工" in _row(_settings_markdown(page), "window_span")

    page.button(key="msc_reset_overrides").click()
    page.run()
    assert not page.exception
    row = _row(_settings_markdown(page), "window_span")
    assert "手工" not in row
    assert "派生" in row
    assert page.checkbox(key="msc_ov_on_window_span").value is False


# --------------------------------------------------------------------------- #
# 3. 计划预览与门禁
# --------------------------------------------------------------------------- #
def test_preview_shows_sixteen_segments_and_112_blocks(page: AppTest) -> None:
    frame = _matrix_frame(page)
    assert len(frame) == 16                                  # 4 形状 × 4 阶段
    assert int(frame["块数"].sum()) == 112


def test_start_button_is_disabled_while_disconnected(page: AppTest) -> None:
    button = page.button(key="msc_start")
    assert button.proto.disabled is True
    captions = " ".join(element.value for element in page.caption)
    assert "未连接 SLM" in captions


def test_unchecking_every_shape_keeps_the_history_browsable(page: AppTest) -> None:
    for key in SHAPE_KEYS:
        page.checkbox(key=key).uncheck()
    page.run()
    assert not page.exception
    assert any("至少选一个形状" in (element.value or "") for element in page.warning)


def test_checks_card_is_rendered_offline(page: AppTest) -> None:
    """检查卡六项在无设备时也画得出来（含"未知"档，不许静默当通过）。"""
    captions = " ".join(element.value for element in page.caption)
    assert "零级位置" in captions
    tables = [element.value for element in page.dataframe
              if "检查" in _columns(element.value)]
    assert tables, "检查卡没有渲染成表"
    labels = set(tables[0]["检查"])
    assert {"几何自洽", "磁盘预算", "时间预算", "模型新鲜度", "饱和预估", "曝光收敛史"} == labels
    # 红停项只可能是这六项之一（页面自身不该抛 error）。
    for element in page.error:
        assert any(str(element.value).startswith(f"{label}：") for label in labels), element.value


# --------------------------------------------------------------------------- #
# 4. 离线读盘助手（空目录 / 坏产物都不许抛）
# --------------------------------------------------------------------------- #
def test_campaign_dirs_is_empty_on_a_fresh_clone(tmp_path: Path) -> None:
    assert ui._campaign_dirs(tmp_path / "missing") == []
    assert ui._campaign_dirs(tmp_path) == []


def test_campaign_dirs_sorts_by_journal_freshness(tmp_path: Path) -> None:
    """按 ``journal.json`` 的 mtime 倒序：目录名是 ``s1200``/``s600``，按名字排看不出新旧。"""
    old = tmp_path / "a" / "s1200"
    new = tmp_path / "b" / "s200"
    for directory, stamp in ((old, 1_000_000), (new, 2_000_000)):
        directory.mkdir(parents=True)
        (directory / "journal.json").write_text("[]", encoding="utf-8")
        (directory / "journal.json").touch()
    os.utime(old / "journal.json", (stamp, stamp))
    os.utime(new / "journal.json", (stamp + 1000, stamp + 1000))
    assert [path.name for path in ui._campaign_dirs(tmp_path)] == ["s200", "s1200"]


def test_model_dirs_lists_only_directories_with_a_fit_result(tmp_path: Path) -> None:
    usable = tmp_path / "stamp" / "s1200"
    (usable / "fit").mkdir(parents=True)
    (usable / "fit" / "result.json").write_text("{}", encoding="utf-8")
    (tmp_path / "stamp" / "s600" / "capture").mkdir(parents=True)
    assert ui._model_dirs(tmp_path) == [usable]
    assert ui._model_dirs(tmp_path / "missing") == []


def test_known_zero_xy_prefers_the_geometry_plan(tmp_path: Path) -> None:
    campaign = tmp_path / "stamp" / "s1200"
    campaign.mkdir(parents=True)
    (campaign / "geometry_plan.json").write_text(
        json.dumps({"zero_xy": [1294.8, 982.4]}), encoding="utf-8")
    point, source = ui._known_zero_xy([campaign])
    assert point == (1294.8, 982.4)
    assert source.endswith("geometry_plan.json")


def test_known_zero_xy_falls_back_to_the_journal_then_admits_ignorance(tmp_path: Path) -> None:
    campaign = tmp_path / "stamp" / "s600"
    campaign.mkdir(parents=True)
    (campaign / "journal.json").write_text(
        json.dumps([{"stage": "geometry", "zero_xy": [10.0, 20.0]}]), encoding="utf-8")
    assert ui._known_zero_xy([campaign]) == ((10.0, 20.0), f"{campaign.name}/journal.json")
    # 两个都读不到 → (None, 原因)，几何检查据此报「未知」而不是编一个位置。
    broken = tmp_path / "stamp" / "s200"
    broken.mkdir(parents=True)
    (broken / "journal.json").write_text("{not json", encoding="utf-8")
    point, source = ui._known_zero_xy([broken])
    assert point is None
    assert source


# --------------------------------------------------------------------------- #
# 5. 交给 execute_campaign 的关键字 / 断点续跑
# --------------------------------------------------------------------------- #
def _write_journal(campaign: Path, entries: list[tuple[str, str]],
                   shapes: tuple[str, ...] = ("square",),
                   plan: tuple[str, ...] | None = None) -> None:
    """最小 journal：只写 ``stage`` / ``status``，形状目录一并建出来。

    ``plan`` 写的是 runner 落盘的 ``{"plan": {"shapes": [...]}}`` 条目 —— 那是**这份
    目录实际按什么计划跑的**，进度分母优先取它（换过勾选之后也仍然对）。
    """
    for shape in shapes:
        (campaign / "shape" / shape).mkdir(parents=True, exist_ok=True)
    payload: list[dict] = []
    if plan:
        payload.append({"plan": {"shapes": list(plan)}})
    payload += [{"stage": stage, "status": status} for stage, status in entries]
    (campaign / "journal.json").write_text(json.dumps(payload), encoding="utf-8")


def test_runner_kwargs_are_all_accepted_by_execute_campaign() -> None:
    """页面拼出来的每个键都必须是 ``slm-transfer-shapes`` 的参数名。

    ``execute_campaign`` 自己按 ``run.params`` 校验未知键（拼错一个就在第一次真跑时
    抛 ``TypeError``），所以"可接受的键"= click 参数表（去掉 ``out_dir``/``execute``/
    ``records_only`` 这三个不进 execute 的）。这条测试把那个失败点提到 import 期。

    另一半是回归点：``CampaignSettings.kwargs()`` **没有** ``model_dir`` / ``resume``
    （前者由 ④ 单独选、后者由 ⑥ 的断点续跑开关给），页面漏传就等于悄悄用 click 默认值
    —— 从零重训 + 重跑已完成段，几小时后才发现。
    """
    import inspect

    from ao_shaping.runners.slm.transfer_shapes_runner import execute_campaign, run

    # ``execute_campaign`` 收 ``**kwargs``，单看它的签名检不出拼错的键，所以两边都钉：
    # 签名确实有 VAR_KEYWORD（键名靠 run.params 校验），且参数表里真的有这些名字。
    assert any(param.kind is inspect.Parameter.VAR_KEYWORD
               for param in inspect.signature(execute_campaign).parameters.values())
    accepted = {param.name for param in run.params} - {"out_dir", "execute", "records_only"}

    settings = ui.derive_settings(("square",), 600.0)
    snapshot = ui._runner_kwargs(settings=settings, slm_number=1,
                                 model_dir=Path("data") / "debug" / "model", resume=True)
    assert set(snapshot) <= accepted, sorted(set(snapshot) - accepted)
    assert {"model_dir", "resume"} <= set(snapshot)
    # 派生表本来就不含这两项（也不含 out/execute）：页面的显式补齐不是多余动作。
    assert set(settings.kwargs()).isdisjoint({"model_dir", "resume", "out", "execute"})
    # runner 会对 ``model_dir`` 调 ``.resolve()``，传 str 会 AttributeError。
    assert isinstance(snapshot["model_dir"], Path)
    assert snapshot["resume"] is True
    assert ui._runner_kwargs(settings=settings, slm_number=1, model_dir=None,
                             resume=False)["model_dir"] is None


def test_completed_segments_lists_only_finished_journal_rows(tmp_path: Path) -> None:
    """「将跳过」= journal 的 complete/reused；``stopped`` 还有块没跑，不许算进去。"""
    campaign = tmp_path / "stamp" / "s1200"
    _write_journal(campaign, [("square/transport_cv", "complete"),
                              ("square/mono12_cv", "stopped"),
                              ("square/mono12_continue", "reused")])
    assert ui.completed_segments(campaign) == ["square/transport_cv",
                                               "square/mono12_continue"]
    # 没有 journal = 新目录，没什么可续的。
    assert ui.completed_segments(tmp_path / "missing") == []
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    assert ui.completed_segments(fresh) == []


def test_resume_control_shows_the_segments_that_would_be_skipped(page: AppTest,
                                                                tmp_path: Path) -> None:
    """⑥ 里那个开关：目录已有 journal 才出现，默认勾上，并列出将复用的段。"""
    campaign = tmp_path / "stamp" / "s1200"
    _write_journal(campaign, [("square/transport_cv", "complete"),
                              ("square/mono12_cv", "stopped")])
    page.text_input(key="msc_out_text").input(str(campaign))
    page.run()
    assert not page.exception

    box = page.checkbox(key=ui._RESUME_KEY)
    assert box.value is True
    assert "1 段" in box.label
    reused = " ".join(element.value for element in page.caption)
    assert "square/transport_cv" in reused
    # stopped 的段还有块没跑：列进「将复用」会让操作员以为收工了。
    assert "square/mono12_cv" not in reused.split("将复用：", 1)[1].split("（共", 1)[0]

    # 取消勾选 → 下一次「开始」传 resume=False，并且旁边必须写清它不等于"重跑"。
    page.checkbox(key=ui._RESUME_KEY).uncheck()
    page.run()
    assert page.checkbox(key=ui._RESUME_KEY).value is False
    assert "FileExistsError" in " ".join(element.value for element in page.caption)


def test_resume_control_is_absent_for_a_fresh_out_dir(page: AppTest,
                                                      tmp_path: Path) -> None:
    page.text_input(key="msc_out_text").input(str(tmp_path / "fresh"))
    page.run()
    assert not page.exception
    assert ui._RESUME_KEY not in {element.key for element in page.checkbox}


# --------------------------------------------------------------------------- #
# 6. 分母来自计划 / 形状名 / 光路无光
# --------------------------------------------------------------------------- #
ALL_STAGES = ("transport_cv", "mono12_cv", "mono12_continue", "mono12_dense")


def _stage_table(page: AppTest):
    """⑥ 的逐段表（同时有「形状」与「状态」两列；⑤ 检查卡也有「状态」列）。"""
    for element in page.dataframe:
        columns = set(_columns(element.value))
        if {"形状", "状态"} <= columns:
            return element.value
    raise AssertionError("逐段表没有渲染出来")


def test_progress_denominator_comes_from_the_plan_not_the_directory(page: AppTest,
                                                                    tmp_path: Path) -> None:
    """跑一半的目录只能读到 4 段，进度条必须写 4/16 —— 不是 4/4。

    ``CampaignProgress.totals`` 的 ``*_total`` 只数**本目录里已出现**的段（它的
    docstring 也这么写），所以直接拿它当分母，一份跑到 1/4 的 campaign 会显示
    「4/4 段」—— 2026-10-28 那次最容易被误读的一行。分母取计划矩阵。
    """
    campaign = tmp_path / "stamp" / "s1200"
    _write_journal(campaign, [(f"square/{stage}", "complete") for stage in ALL_STAGES],
                   shapes=("square",))
    page.text_input(key="msc_out_text").input(str(campaign))
    page.run()
    assert not page.exception

    text = page.get("progress")[0].proto.text
    assert "段 4/16" in text, text                     # 16 = 4 形状 × 4 阶段（计划）
    assert "/112" in text, text                        # 112 = 计划块数
    captions = " ".join(element.value for element in page.caption)
    assert "读到 4 段（计划 16 段）" in captions


def test_plan_shapes_prefers_the_journal_plan_entry(tmp_path: Path) -> None:
    """历史目录里只有 square 的产物，但计划写着 4 个形状时，分母仍是 16 段。"""
    from ao_shaping.tools.slm.transfer.campaign_progress import collect_campaign_progress

    campaign = tmp_path / "stamp" / "s1200"
    _write_journal(campaign, [("plan", "complete"),
                              ("square/transport_cv", "complete")], shapes=("square",))
    journal = json.loads((campaign / "journal.json").read_text(encoding="utf-8"))
    journal[0]["plan"] = {"shapes": ["square", "triangle", "star", "circle"]}
    (campaign / "journal.json").write_text(json.dumps(journal), encoding="utf-8")

    progress = collect_campaign_progress(campaign)
    assert ui.journal_plan_shapes(progress) == ("square", "triangle", "star", "circle")
    assert ui.plan_shapes(progress) == ("square", "triangle", "star", "circle")
    # 没有 plan 条目（老目录）时：plan 空、plan_shapes 退回目录里出现过的形状，而不是空元组。
    plain = tmp_path / "old" / "s600"
    _write_journal(plain, [("square/transport_cv", "complete")], shapes=("square",))
    plain_progress = collect_campaign_progress(plain)
    assert ui.journal_plan_shapes(plain_progress) == ()
    assert ui.plan_shapes(plain_progress) == ("square",)


def test_every_shape_name_goes_through_shape_labels_with_a_fallback(page: AppTest,
                                                                    tmp_path: Path) -> None:
    """形状名一律 ``SHAPE_LABELS.get(name, name)``：词表与形状表**不保证**两边一致。

    ``transfer_progress.SHAPE_LABELS`` 是从 ``TARGET_SHAPE_LABELS`` 取的一份副本，历史
    目录里还可能出现词表外的名字（``rectangle`` 从未进过 campaign、更老的目录甚至可能有
    别的）。这里不假设两边的键集相等：有标签就印标签，没有就印原名，绝不 KeyError。
    """
    from ao_shaping.gui.slm.transfer_progress import SHAPE_LABELS

    campaign = tmp_path / "stamp" / "s600"
    _write_journal(campaign, [("circle/transport_cv", "complete"),
                              ("rectangle/mono12_cv", "running"),
                              ("hexagon/transport_cv", "complete")],
                   shapes=("circle", "rectangle", "hexagon"))
    page.text_input(key="msc_out_text").input(str(campaign))
    page.run()
    assert not page.exception
    shown = set(_stage_table(page)["形状"])
    expected = {SHAPE_LABELS.get(name, name) for name in ("circle", "rectangle", "hexagon")}
    assert shown == expected
    # 词表外的名字必须原样印出来，而不是被静默丢掉或显示成空。
    assert "hexagon" in shown


def test_optical_path_banner_renders_the_report_and_the_cause_advice(page: AppTest) -> None:
    """光路无光不是「campaign 失败」：印 ``report()`` 原文 ＋ 按 cause 的处置方向。"""
    report = ("光路无光 —— 峰值不随曝光走 (what=shape)\n"
              "  0.02→0.08 ms 峰值 0.0 counts (噪声底 4.0)")
    page.session_state[ui._OPTICAL_KEY] = {"reason": "flat", "cause": "exposure_ceiling",
                                          "report": report}
    page.run()
    assert not page.exception
    banner = " ".join(element.value for element in page.error)
    assert "光路无光（退出码 3）" in banner
    assert "reason=flat" in banner and "cause=exposure_ceiling" in banner
    # report() 逐字节照印（多行、含实测数字），不重排也不摘要。
    assert [element.value for element in page.code] == [report]
    captions = " ".join(element.value for element in page.caption)
    assert "AO_DAHENG_MAX_EXPOSURE_MS" in captions

    # blocked 的方向相反：先看光路，不是调曝光上限。
    page.session_state[ui._OPTICAL_KEY] = {"reason": "flat", "cause": "blocked",
                                          "report": report}
    page.run()
    captions = " ".join(element.value for element in page.caption)
    assert "先去看光路" in captions
    assert "AO_DAHENG_MAX_EXPOSURE_MS" not in captions

def test_progress_tables_survive_missing_values(tmp_path: Path) -> None:
    """缺值必须是 ``None`` 而不是 ``"—"``：混排会让 ``st.dataframe`` 整表消失。

    ``st.dataframe`` 走 arrow 序列化。同一列里既有 ``0.6574`` 又有 ``"—"`` 时会退化成
    object 列，arrow 按 double 转换抛 ``ArrowInvalid``，Streamlit 只把栈打到日志、页面
    少一张表。触发条件是「某段还没有块读数」（刚开始 / 被 ``stopped`` / 读到半截产物），
    也就是操作员最需要看进度的那一刻 —— 所以这里直接对着 arrow 断言，而不是断言界面。
    """
    pa = pytest.importorskip("pyarrow")
    pd = pytest.importorskip("pandas")

    from ao_shaping.tools.slm.transfer.campaign_progress import (
        BlockRow,
        CampaignProgress,
        StageRow,
    )

    block = BlockRow(shape="square", stage="mono12_cv", index=0, cv=ui.MISSING,
                     saturated_px=ui.MISSING, core_peak=ui.MISSING,
                     exposure_ms=116.8, accepted=ui.MISSING, note=ui.MISSING)
    stage = StageRow(shape="square", stage="mono12_cv", status="running",
                     blocks=(block,), seed_cv=ui.MISSING, final_cv=1.2,
                     peak_inside=ui.MISSING, peak_full=None, exposure_ms=116.8)
    progress = CampaignProgress(out=tmp_path, stages=(stage,), journal=())

    stage_rows = ui.stage_table_rows(progress)
    block_rows = ui.block_table_rows(stage)
    assert stage_rows[0]["段首 CV"] is None
    assert stage_rows[0]["段末 CV"] == pytest.approx(1.2)
    assert block_rows[0]["CV"] is None and block_rows[0]["验收"] is None
    for rows in (stage_rows, block_rows):
        pa.Table.from_pandas(pd.DataFrame(rows))   # 原先在这里抛 ArrowInvalid


# --------------------------------------------------------------------------- #
# 7. 监控读数：统一进度 / CV 轨迹 / 段间闸 / 成本
# --------------------------------------------------------------------------- #
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


def test_idle_panel_draws_the_three_monitor_blocks(page: AppTest, tmp_path: Path,
                                                  monkeypatch) -> None:
    """⑥ 的空闲面板：CV 轨迹 + 段间闸（禁用）+ 成本表，三个块都得在。

    空闲视图不是"没东西可看"：断线重开后第一步就是看这份盘上读数，所以轨迹图与成本
    表在这里和运行中一样要画；只有闸必须禁用（没有 campaign 在跑）。
    """
    # 换到空目录：历史区没有可挑的 campaign，这张图只可能来自 ⑥。
    monkeypatch.chdir(tmp_path)
    campaign = tmp_path / "stamp" / "s1200"
    _write_journal(campaign, [("square/transport_cv", "complete")], plan=("square",))
    _write_blocks(campaign, "square", "transport_cv", blocks=4)

    page.text_input(key="msc_out_text").input(str(campaign))
    page.run()
    assert not page.exception

    # 轨迹：一张折线图，说明里写清横轴是什么、CV 越低越好（空图会被读成"没有变化"）。
    assert len(page.get("vega_lite_chart")) == 1
    captions = " ".join(element.value for element in page.caption)
    assert "段序·块序" in captions
    assert "越低越好" in captions

    # 段间闸：空闲时禁用，键带 ⑥ 的前缀（⑦ 同时读同一目录时不能撞键）。
    gate = [button for button in page.button if button.key == "cm_idle_stage_pause"]
    assert len(gate) == 1
    assert gate[0].disabled is True
    assert "只在运行中有意义" in captions

    # 成本表：模块已就绪给表、未就绪给一行说明，两种都算"这一块在"。
    assert "**成本**" in " ".join(element.value for element in page.markdown
                                 if element.value)
    assert ("首次运行" in captions) or ("尚未就绪" in captions)


def test_progress_line_is_the_shared_formatter_sentence(page: AppTest,
                                                        tmp_path: Path) -> None:
    """进度条上那句话必须由 ``campaign_monitor.progress_caption`` 生成。

    页面里任何一处手写的"跑到第几段第几块"都会和它漂移成两种写法，而操作员会拿其中一个
    去对另一个。所以这里不是"格式像"，而是与同一个函数的输出**逐字相等**。
    """
    from ao_shaping.gui.slm import campaign_monitor as cm
    from ao_shaping.tools.slm.transfer.campaign_progress import collect_campaign_progress

    campaign = tmp_path / "stamp" / "s1200"
    _write_journal(campaign, [("square/transport_cv", "complete")], plan=("square",))
    _write_blocks(campaign, "square", "transport_cv", blocks=4)

    page.text_input(key="msc_out_text").input(str(campaign))
    page.run()
    assert not page.exception

    text = page.get("progress")[0].proto.text
    expected = cm.progress_caption(collect_campaign_progress(campaign),
                                   planned_shapes=("square",))
    assert text == expected
    assert "段 1/4" in text and "块 4/28" in text and "已用 " in text


def test_idle_panel_and_history_view_can_read_the_same_directory(tmp_path: Path,
                                                                monkeypatch) -> None:
    """⑥ 与 ⑦ 同时读同一个目录：两套前缀互不干扰，⑦ 不出暂停闸。

    ``render_progress_panel`` 现在被三处视图共用，同一次脚本运行里出现两份是常态
    （默认输出目录正好也是历史里那一条）。不带前缀的同名控件会让整页变成一条
    ``DuplicateElementKey`` 异常 —— 那不是"界面有点小问题"，是页面全白。
    """
    monkeypatch.chdir(tmp_path)          # campaign_root() = <tmp>/data/debug/slm_transfer
    campaign = tmp_path / "data" / "debug" / "slm_transfer" / "stamp" / "s1200"
    _write_journal(campaign, [("square/transport_cv", "complete")])
    _write_blocks(campaign, "square", "transport_cv", blocks=4)

    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    at.text_input(key="msc_out_text").input(str(campaign))
    at.run()
    assert not at.exception, [str(exc.value)[:200] for exc in at.exception]

    keys = {button.key for button in at.button}
    assert "cm_idle_stage_pause" in keys
    assert not any(key and key.startswith("cm_hist_") for key in keys)
    # 两个面板各画一张轨迹图：同一个目录被读了两次，不是只画一次。
    assert len(at.get("vega_lite_chart")) == 2


# --------------------------------------------------------------------------- #
# 8. ⑦ 的模型复用（接已完工的历史浏览器；面板本体见 test_campaign_reuse.py）
# --------------------------------------------------------------------------- #

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


def _fingerprint_tables(page: AppTest) -> list:
    """指纹卡的六行表（列名固定为 ``项目`` / ``值``）。"""
    return [element.value for element in page.dataframe
            if "项目" in _columns(element.value)]


def test_history_section_renders_the_reuse_panel_for_the_picked_directory(
        tmp_path: Path, monkeypatch) -> None:
    """⑦ 整段：挑中的目录既出逐段读数，也出这条光路的模型复用。"""
    monkeypatch.chdir(tmp_path)
    campaign = tmp_path / "data" / "debug" / "slm_transfer" / "stamp" / "s1200"
    _write_journal(campaign, [("square/transport_cv", "complete")])
    _write_blocks(campaign, "square", "transport_cv", blocks=2)
    _write_recorded_fingerprint(campaign)

    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    assert not at.exception, [str(exc.value)[:200] for exc in at.exception]

    markdown = " ".join(element.value for element in at.markdown if element.value)
    assert "模型复用（这条光路）" in markdown
    assert _fingerprint_tables(at), "⑦ 没有画出指纹卡"
    # 逐段/逐块读数视图还在（同一段里，不挪走）。
    assert _stage_table(at) is not None
    assert at.session_state["msc_hist_current_fp"]["window_span"] == 420.0
