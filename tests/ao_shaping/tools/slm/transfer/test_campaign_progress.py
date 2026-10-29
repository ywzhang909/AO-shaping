"""campaign 进度读取器的测试：合成目录覆盖结构/降级，真实产物覆盖逐位数字。

合成目录是必需的，不是锦上添花：真实产物 ``data/debug/slm_transfer/...`` 不入库
（``data/`` 是 machine-local），所以真实 fixture 的用例必须能干净 skip，而
「16 段 / 112 块 / 合计」这些结构性断言必须永远有东西可跑。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ao_shaping.tools.slm.transfer.campaign_progress import (
    COMPLETE_STATUSES,
    MISSING,
    CampaignProgress,
    collect_campaign_progress,
)
from ao_shaping.tools.slm.transfer.stages import STAGE_RECIPES

#: 真实一次 campaign 的产物目录（2026-10-28，s1200 lane）。不入库 ⇒ 用例要能 skip。
REAL_FIXTURE = Path("data/debug/slm_transfer/multishape_scale_20261028/s1200")


def _row(label: str, cv: float, *, core_peak: float = 100.0, peak: float | None = None,
         saturated: int = 0, exposure_ms: float = 135.196, bit_depth: int = 12) -> dict:
    """一行 manifest（字段照 ``validate.run`` 的 ``capture`` 写出的那份）。

    ``peak is None`` 就是**旧 manifest**：``peak`` 字段是后加的，那批产物没有它，
    读取器必须报 ``None`` 而不是 ``0.0``。
    """
    row = {
        "label": label,
        "file": f"{label}.npz",
        "metrics": {"cv": cv, "mean": 1.0, "ee_roi": 0.4, "target_peak": 1.0},
        "core_peak": core_peak,
        "saturation_pixels": saturated,
        "exposure_ms": exposure_ms,
        "bit_depth": bit_depth,
        "full_scale": float(2**bit_depth - 1),
    }
    if peak is not None:
        row["peak"] = peak
    return row


def _write_stage(
    root: Path, shape: str, stage: str, *, blocks: int, ccd_check: bool,
    seed_cv: float, final_cv: float, summary: bool = True, accepted: bool = True,
    peaks: bool = True,
) -> Path:
    """写一个阶段目录（manifest + 可选 summary），行序与真实产物一致。

    ``peaks=False`` 写的是**没有全幅 ``peak`` 字段**的旧 manifest。
    """
    stage_dir = root / "shape" / shape / stage
    stage_dir.mkdir(parents=True, exist_ok=True)

    def full(label_peak: float) -> float | None:
        return label_peak if peaks else None

    rows = [_row(f"{shape}_seed", seed_cv, core_peak=102.0, peak=full(380.0))]
    for index in range(blocks):
        rows.append(_row(f"{shape}_{index}_A1", 0.90 - 0.01 * index, core_peak=200.0,
                         peak=full(900.0), saturated=1))
        if ccd_check:
            rows.append(_row(f"{shape}_{index}_B", 0.85 - 0.01 * index, core_peak=210.0,
                             peak=full(950.0)))
    # 段末是 ``<shape>_final``；``_final_adaptive`` 是另一次采集，不能被当成段末。
    rows.append(_row(f"{shape}_final_adaptive", 0.50, core_peak=150.0, peak=full(1200.0)))
    # 段内峰值让 ``_seed_replay`` 顶到最高：它既不是块行也不是段首/段末，只有"全行"口径才会取到它。
    rows.append(_row(f"{shape}_seed_replay", seed_cv + 0.01, core_peak=120.0, peak=full(4095.0)))
    rows.append(_row(f"{shape}_final", final_cv, core_peak=230.0, peak=full(3600.0)))
    (stage_dir / "manifest.json").write_text(json.dumps(rows), encoding="utf-8")

    if summary:
        entries = [
            {"shape": shape, "block": index, "accepted": accepted and index % 2 == 0,
             "acceptance": "sequential_ccd", "accept_metric": "cv",
             "baseline": 0.90 - 0.01 * index, "candidate": 0.85 - 0.01 * index}
            for index in range(blocks)
        ]
        (stage_dir / "summary.json").write_text(json.dumps(entries), encoding="utf-8")
    return stage_dir


def _write_journal(root: Path, entries: list[dict]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "journal.json").write_text(json.dumps(entries), encoding="utf-8")


def _full_shape(root: Path, shape: str = "square") -> None:
    """一个形状跑完四段（28 块），journal 记 complete。"""
    for index, recipe in enumerate(STAGE_RECIPES):
        _write_stage(root, shape, recipe.name, blocks=recipe.blocks,
                     ccd_check=recipe.ccd_check, seed_cv=0.9 - 0.05 * index,
                     final_cv=0.8 - 0.05 * index)
    _write_journal(root, [{"stage": f"{shape}/{recipe.name}", "status": "complete"}
                          for recipe in STAGE_RECIPES])


def test_structure_and_totals_for_one_shape(tmp_path: Path) -> None:
    """一形状四段 = 4 段 / 28 块，journal 状态 complete。"""
    _full_shape(tmp_path)

    progress = collect_campaign_progress(tmp_path)

    assert isinstance(progress, CampaignProgress)
    assert [stage.stage for stage in progress.stages] == [r.name for r in STAGE_RECIPES]
    assert [len(stage.blocks) for stage in progress.stages] == [r.blocks for r in STAGE_RECIPES]
    assert progress.totals == {
        "segments_total": 4,
        "segments_done": 4,
        "blocks_total": 28,
        "blocks_done": 28,
    }
    assert all(stage.status == "complete" for stage in progress.stages)


def test_block_rows_use_the_candidate_frame_and_summary_verdict(tmp_path: Path) -> None:
    """块 CV 取候选帧（有 B 取 B），accepted 取 summary.json —— manifest 里没有它。"""
    _full_shape(tmp_path)

    stage = collect_campaign_progress(tmp_path).stages[1]  # mono12_cv：ccd_check=True
    first, second = stage.blocks[0], stage.blocks[1]

    assert first.cv == pytest.approx(0.85)
    assert first.accepted is True
    assert second.accepted is False
    assert "0.9000" in first.note and "0.8500" in first.note
    assert "sequential_ccd/cv" in first.note
    # 饱和/峰值取该块**最后一帧**（候选帧 B），不是 A1。
    assert first.saturated_px == 0
    assert first.core_peak == pytest.approx(210.0)
    # 只采 A1 的段（transport_cv）报 A1 的饱和像素
    a1_only = collect_campaign_progress(tmp_path).stages[0].blocks[0]
    assert a1_only.saturated_px == 1


def test_seed_and_final_use_exact_labels_not_first_and_last_rows(tmp_path: Path) -> None:
    """段首/段末必须取 ``<shape>_seed`` / ``<shape>_final``，不能按行序取首末。

    manifest 的末行恰好是 ``_final``，但 ``_final_adaptive``（0.50）在它之前 ——
    真实产物里这两行的先后顺序不可依赖，所以这里让 ``_seed_replay`` 排在 ``_final``
    之前，检查读取器不受行序影响。
    """
    root = tmp_path
    _write_stage(root, "square", "transport_cv", blocks=4, ccd_check=False,
                 seed_cv=1.0408, final_cv=0.7766)

    stage = collect_campaign_progress(root).stages[0]

    assert stage.seed_cv == pytest.approx(1.0408)
    assert stage.final_cv == pytest.approx(0.7766)  # 不是 0.50（final_adaptive）
    assert stage.peak_inside == pytest.approx(230.0)  # 段内 core_peak 最大值
    assert stage.exposure_ms == pytest.approx(135.196)


def test_peak_full_takes_every_row_while_peak_inside_takes_the_core(tmp_path: Path) -> None:
    """``peak_full`` = 全部行的全幅 ``peak`` 最大值（与报告「段内峰值」同口径）。

    合成 manifest 让**段内峰值落在 ``_seed_replay`` 上**（4095，既不是块行也不是
    段首/段末）：只有"全部行"口径才会取到它，按块行或按首末行取都会得到别的数字。
    ``peak_inside`` 仍然只走 ``core_peak``（最大值在 ``_final``，230）⇒ 两个字段
    答的是两个问题，数值必须不同。
    """
    _write_stage(tmp_path, "square", "mono12_cv", blocks=8, ccd_check=True,
                 seed_cv=0.7, final_cv=0.65)

    stage = collect_campaign_progress(tmp_path).stages[0]

    assert stage.peak_full == pytest.approx(4095.0)
    assert stage.peak_inside == pytest.approx(230.0)


def test_peak_full_is_none_for_an_old_manifest_without_the_peak_field(
    tmp_path: Path,
) -> None:
    """旧 manifest 没有 ``peak`` 字段：报 ``None``，绝不当成 ``0.0``。

    判据是"这一行到底带没带该字段"，不是"读数是否为零" —— 0 是一个真实的峰值
    （全黑帧），``None`` 才是"没这个数"。
    """
    _write_stage(tmp_path, "square", "mono12_cv", blocks=8, ccd_check=True,
                 seed_cv=0.7, final_cv=0.65, peaks=False)

    stage = collect_campaign_progress(tmp_path).stages[0]

    assert stage.peak_full is None
    assert stage.peak_full != 0.0


def test_stopped_stage_passes_through_with_only_the_blocks_that_exist(
    tmp_path: Path,
) -> None:
    """journal 写 ``stopped``（协作式段间停止）：状态原样透传，块只报已存在的。

    半途停下的段既不能算完成（``segments_done`` 不计），也不能补造未跑的块
    （``blocks_done`` 只加实际行数，``blocks_total`` 仍是计划量）。
    """
    _write_stage(tmp_path, "square", "mono12_cv", blocks=3, ccd_check=True,
                 seed_cv=0.7, final_cv=0.65)  # 8 块里跑完 3 块就被停
    _write_journal(tmp_path, [{"stage": "square/mono12_cv", "status": "stopped"}])

    progress = collect_campaign_progress(tmp_path)

    stage = progress.stages[0]
    assert stage.status == "stopped"
    assert stage.status not in COMPLETE_STATUSES
    assert [block.index for block in stage.blocks] == [0, 1, 2]
    assert progress.totals == {
        "segments_total": 1,
        "segments_done": 0,
        "blocks_total": 8,
        "blocks_done": 3,
    }


def test_running_stage_counts_as_not_done(tmp_path: Path) -> None:
    """journal 里 running 的段不算完成，块数照实报。"""
    _full_shape(tmp_path)
    _write_journal(tmp_path, [{"stage": "square/transport_cv", "status": "complete"},
                              {"stage": "square/mono12_cv", "status": "running"}])

    progress = collect_campaign_progress(tmp_path)

    statuses = {stage.stage: stage.status for stage in progress.stages}
    assert statuses["mono12_cv"] == "running"
    # transport_cv 有 complete 行；后两段无 journal 行 → 按 manifest 兜底判 complete。
    assert statuses["mono12_dense"] == "complete"
    assert progress.totals["segments_done"] == 3
    assert progress.totals["blocks_done"] == 28


def test_missing_journal_falls_back_to_manifest(tmp_path: Path) -> None:
    """手工拷进来的目录没有 journal 行：按 manifest 块数判 complete / partial。"""
    _write_stage(tmp_path, "square", "mono12_cv", blocks=8, ccd_check=True,
                 seed_cv=0.7, final_cv=0.65)
    _write_stage(tmp_path, "square", "mono12_continue", blocks=3, ccd_check=True,
                 seed_cv=0.7, final_cv=0.65)  # 只跑了 3 块（计划 12）

    progress = collect_campaign_progress(tmp_path)

    statuses = {stage.stage: stage.status for stage in progress.stages}
    assert statuses == {"mono12_cv": "complete", "mono12_continue": "partial"}
    assert progress.totals["blocks_total"] == 8 + 12


@pytest.mark.parametrize("mode", ["empty", "corrupt_journal", "no_manifest", "no_summary",
                                 "unrelated_shape_dir"])
def test_missing_or_partial_products_never_raise(tmp_path: Path, mode: str) -> None:
    """缺文件、半截 JSON、只有 journal：一律降级成 MISSING/空元组，从不抛异常。"""
    if mode == "corrupt_journal":
        (tmp_path / "journal.json").write_text("{not json", encoding="utf-8")
    elif mode == "no_manifest":
        (tmp_path / "shape" / "square" / "mono12_cv").mkdir(parents=True)
    elif mode == "no_summary":
        _write_stage(tmp_path, "square", "mono12_cv", blocks=8, ccd_check=True,
                     seed_cv=0.7, final_cv=0.65, summary=False)
    elif mode == "unrelated_shape_dir":
        (tmp_path / "shape" / "hexagon").mkdir(parents=True)

    progress = collect_campaign_progress(tmp_path)

    assert isinstance(progress, CampaignProgress)
    if mode in ("empty", "corrupt_journal", "unrelated_shape_dir"):
        assert progress.stages == ()
        assert progress.totals["blocks_done"] == 0
    if mode == "no_summary":
        block = progress.stages[0].blocks[0]
        assert block.accepted is MISSING
        assert "无 summary.json" in block.note
    if mode == "no_manifest":
        assert progress.stages[0].blocks == ()
        assert progress.stages[0].seed_cv is MISSING


def test_unknown_stage_directory_is_ignored(tmp_path: Path) -> None:
    """只按 ``STAGE_RECIPES`` 认段：目录里多出来的名字不参与（否则合计会漂）。"""
    _write_stage(tmp_path, "square", "mono12_cv", blocks=8, ccd_check=True,
                 seed_cv=0.7, final_cv=0.65)
    (tmp_path / "shape" / "square" / "experimental_stage").mkdir(parents=True)

    progress = collect_campaign_progress(tmp_path)

    assert [stage.stage for stage in progress.stages] == ["mono12_cv"]


@pytest.mark.skipif(not REAL_FIXTURE.exists(), reason="真实产物在 data/ 下，不入库")
def test_real_campaign_fixture_matches_report_table() -> None:
    """真实产物：16 段 / 112 块，正方形四段的段首→段末 CV 与报告 §3.2 逐位一致。"""
    progress = collect_campaign_progress(REAL_FIXTURE)

    assert len(progress.stages) == 16
    assert sum(len(stage.blocks) for stage in progress.stages) == 112
    assert progress.totals == {
        "segments_total": 16,
        "segments_done": 16,
        "blocks_total": 112,
        "blocks_done": 112,
    }

    square = {stage.stage: stage for stage in progress.stages if stage.shape == "square"}
    expected = {
        "transport_cv": (1.0408, 0.7766),
        "mono12_cv": (0.7038, 0.6574),
        "mono12_continue": (0.6836, 0.6442),
        "mono12_dense": (0.6472, 0.6443),
    }
    for stage, (seed_cv, final_cv) in expected.items():
        row = square[stage]
        assert round(row.seed_cv, 4) == seed_cv, stage
        assert round(row.final_cv, 4) == final_cv, stage
    assert [len(square[stage].blocks) for stage in expected] == [4, 8, 12, 4]

    # 报告 §3.2「段内峰值」列（manifest 全部行的最大**全幅** peak，含 seed/seed_replay/final）。
    # 值取自 multishape_scale_live_20261028.md 该表「正方形」四行，逐位一致。
    report_peak_full = {
        "transport_cv": 255.0,
        "mono12_cv": 3923.0,
        "mono12_continue": 4095.0,
        "mono12_dense": 3465.0,
    }
    for stage, peak in report_peak_full.items():
        assert square[stage].peak_full == peak, stage
    # 核心峰值是另一个量：它不该等于上表（否则说明口径退回全幅了）。
    assert square["transport_cv"].peak_inside != report_peak_full["transport_cv"]
