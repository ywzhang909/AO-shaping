# -*- coding: utf-8 -*-
"""离线门禁: S1f 随机抖动归因判读脚本（预注册 §3.4）。

覆盖:
- 合成帧注入后 ``zero_order_reference`` / ``principal_position`` 的定位正确;
- ``pairwise_max_px`` / ``rms_px`` / ``drift_metrics`` 的数值定义;
- 裁定三分支（ratio<=1.3 -> V1, >=2.0 -> V2, 中间 -> V3）与 ``drift_material``
  的 1.0 px 边界;
- ``tmp_path`` 端到端: 合成采集目录（``frames/*.npy`` + ``capture_journal.jsonl``）
  跑 ``judge``，断言 schema 与字段齐全、JSON 可序列化、``verdict`` 合法;
- 缺帧 / 帧形状不一致必须抛 ``JudgmentError``，不静默。

不打开设备、不取锁、不写仓库内任何文件。
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pytest

SCRIPT = importlib.import_module("scripts.slm.s1f_jitter_attribution_judge_20261127")

FRAME_SHAPE: tuple[int, int] = (200, 200)
#: ``find_spots`` 会跳过 row<100 且 col<100 的角区杂光, 故光斑必须放在该区之外.
ZERO_ROW, ZERO_COL = 120, 110
ROW_TYPE_DISPLACEMENT = 58  # 本系统 H 图样的 row-type 主轴位移量级


# ---------------------------------------------------------------------------
# 合成帧
# ---------------------------------------------------------------------------
def add_spot(
    frame: np.ndarray, row: float, col: float, peak: float, sigma: float = 2.0
) -> None:
    """在 ``frame`` 上叠一个高斯光斑（截断 4 sigma，亚像素中心）。"""
    radius = 8
    r0 = max(0, int(np.floor(row)) - radius)
    r1 = min(frame.shape[0], int(np.floor(row)) + radius + 1)
    c0 = max(0, int(np.floor(col)) - radius)
    c1 = min(frame.shape[1], int(np.floor(col)) + radius + 1)
    yy, xx = np.indices((r1 - r0, c1 - c0), dtype=np.float64)
    blob = peak * np.exp(
        -(((yy + r0) - row) ** 2 + ((xx + c0) - col) ** 2) / (2.0 * sigma**2)
    )
    frame[r0:r1, c0:c1] += blob


def flat_frame(row: float = ZERO_ROW, col: float = ZERO_COL) -> np.ndarray:
    """只有 0 级光斑的 flat 帧。"""
    frame = np.zeros(FRAME_SHAPE, dtype=np.float64)
    add_spot(frame, row, col, peak=4000.0)
    return frame


def h_frame(
    row_offset: float = 0.0, col_offset: float = 0.0, peak: float = 500.0
) -> np.ndarray:
    """0 级 + 一个偏离 0 级的主瓣（变体 A/B 的 H 帧）。"""
    frame = flat_frame()
    add_spot(
        frame,
        ZERO_ROW + ROW_TYPE_DISPLACEMENT + row_offset,
        ZERO_COL + col_offset,
        peak=peak,
    )
    return frame


# ---------------------------------------------------------------------------
# 0 级参考与主瓣
# ---------------------------------------------------------------------------
def test_zero_order_reference_uses_brightest_spot() -> None:
    frame = flat_frame()
    add_spot(frame, 160.0, 60.0, peak=10.0)  # 远处弱杂光, 不应被当成 0 级
    row, col = SCRIPT.zero_order_reference(frame)
    assert row == pytest.approx(ZERO_ROW, abs=1e-6)
    assert col == pytest.approx(ZERO_COL, abs=1e-6)


def test_zero_order_reference_rejects_empty_frame() -> None:
    with pytest.raises(SCRIPT.JudgmentError):
        SCRIPT.zero_order_reference(np.zeros(FRAME_SHAPE, dtype=np.float64))


def test_principal_position_adds_ref_back() -> None:
    ref = (float(ZERO_ROW), float(ZERO_COL))
    spot = SCRIPT.principal_position(h_frame(), ref)
    assert spot is not None
    assert spot["dr"] == pytest.approx(ROW_TYPE_DISPLACEMENT, abs=0.02)
    assert spot["dc"] == pytest.approx(0.0, abs=0.02)
    assert spot["row"] == pytest.approx(ZERO_ROW + ROW_TYPE_DISPLACEMENT, abs=0.02)
    assert spot["col"] == pytest.approx(ZERO_COL, abs=0.02)
    assert spot["dist"] == pytest.approx(
        float(np.hypot(spot["dr"], spot["dc"])), abs=1e-9
    )


def test_principal_position_none_when_only_zero_order() -> None:
    ref = (float(ZERO_ROW), float(ZERO_COL))
    assert SCRIPT.principal_position(flat_frame(), ref) is None


# ---------------------------------------------------------------------------
# 散布 / 漂移数值定义
# ---------------------------------------------------------------------------
def test_pairwise_max_is_max_of_all_pair_distances() -> None:
    points = [(100.0, 200.0), (103.0, 204.0), (100.5, 200.0)]
    assert SCRIPT.pairwise_max_px(points) == pytest.approx(5.0, abs=1e-12)
    assert SCRIPT.pairwise_max_px(points[:1]) == 0.0


def test_rms_px_is_hypot_of_axis_stds() -> None:
    points = [(100.0, 200.0), (103.0, 204.0)]
    rms, std_row, std_col = SCRIPT.rms_px(points)
    assert std_row == pytest.approx(1.5, abs=1e-12)
    assert std_col == pytest.approx(2.0, abs=1e-12)
    assert rms == pytest.approx(2.5, abs=1e-12)
    metrics = SCRIPT.spread_metrics(points)
    assert metrics == {
        "pairwise_max_px": pytest.approx(5.0, abs=1e-12),
        "rms_px": pytest.approx(2.5, abs=1e-12),
        "std_row": pytest.approx(1.5, abs=1e-12),
        "std_col": pytest.approx(2.0, abs=1e-12),
    }


def test_group_reference_is_per_axis_median() -> None:
    frames = [
        flat_frame(),
        flat_frame(),
        flat_frame(row=ZERO_ROW + 9.0, col=ZERO_COL - 9.0),  # 离群帧
    ]
    ref, positions = SCRIPT.group_zero_order_reference(frames, "flat_head")
    assert len(positions) == 3
    assert ref[0] == pytest.approx(ZERO_ROW, abs=1e-6)
    assert ref[1] == pytest.approx(ZERO_COL, abs=1e-6)


def test_drift_metrics_distance_and_material_boundary() -> None:
    exact = SCRIPT.drift_metrics((10.0, 20.0), (11.0, 20.0))
    assert exact["head_to_tail_px"] == pytest.approx(1.0, abs=1e-12)
    assert exact["dr"] == pytest.approx(1.0, abs=1e-12)
    assert exact["dc"] == 0.0
    assert exact["drift_material"] is True  # 边界: >= 1.0 记物质漂移

    below = SCRIPT.drift_metrics((10.0, 20.0), (10.999, 20.0))
    assert below["head_to_tail_px"] < 1.0
    assert below["drift_material"] is False

    diagonal = SCRIPT.drift_metrics((0.0, 0.0), (3.0, 4.0))
    assert diagonal["head_to_tail_px"] == pytest.approx(5.0, abs=1e-12)


# ---------------------------------------------------------------------------
# 裁定分支
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("b_rms", "a_rms", "expected"),
    [
        (1.0, 1.0, SCRIPT.VERDICT_V1),
        (1.3, 1.0, SCRIPT.VERDICT_V1),  # 下边界含
        (1.31, 1.0, SCRIPT.VERDICT_V3),
        (1.99, 1.0, SCRIPT.VERDICT_V3),
        (2.0, 1.0, SCRIPT.VERDICT_V2),  # 上边界含
        (3.0, 1.0, SCRIPT.VERDICT_V2),
    ],
)
def test_classify_verdict_branches(b_rms: float, a_rms: float, expected: str) -> None:
    result = SCRIPT.classify(b_rms, a_rms)
    assert result["verdict"] == expected
    assert result["ratio"] == pytest.approx(b_rms / a_rms)
    assert result["verdict_label"] == SCRIPT.VERDICT_LABELS[expected]


def test_classify_zero_floor_is_undetermined() -> None:
    result = SCRIPT.classify(0.5, 0.0)
    assert result["verdict"] == SCRIPT.VERDICT_UNDETERMINED
    assert result["ratio"] is None
    assert result["ratio_note"]


def test_attribute_components_and_shares() -> None:
    arms = {"A": {"rms_px": 1.0}, "B": {"rms_px": 2.0}}
    drift = {"head_to_tail_px": 1.5}
    attribution = SCRIPT.attribute(arms, drift)
    assert attribution["readout_floor_rms"] == pytest.approx(1.0)
    assert attribution["write_screen_rms"] == pytest.approx(1.0)
    assert attribution["drift_px"] == pytest.approx(1.5)
    assert attribution["ratio"] == pytest.approx(2.0)
    assert attribution["verdict"] == SCRIPT.VERDICT_V2
    assert attribution["readout_share"] == pytest.approx(0.5)
    assert attribution["write_screen_share"] == pytest.approx(0.5)


def test_attribute_write_screen_never_negative() -> None:
    arms = {"A": {"rms_px": 2.0}, "B": {"rms_px": 1.5}}
    attribution = SCRIPT.attribute(arms, {"head_to_tail_px": 0.2})
    assert attribution["write_screen_rms"] == 0.0
    assert attribution["verdict"] == SCRIPT.VERDICT_V1
    assert attribution["write_screen_share"] == pytest.approx(0.0)
    assert attribution["readout_share"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 采集目录合成（端到端）
# ---------------------------------------------------------------------------
FLAT_FRAMES = 16
ARM_FRAMES = 24


def write_capture(
    root: Path,
    *,
    arm_a_offsets: list[float] | None = None,
    arm_b_offsets: list[float] | None = None,
    drift_tail: float = 0.0,
) -> Path:
    """按 §3.2 顺序合成 96 帧 + journal（文件名 ``{idx:04d}_{pattern_id}.npy``）。"""
    frames_dir = root / "frames"
    frames_dir.mkdir(parents=True)
    if arm_a_offsets is None:
        arm_a_offsets = [0.10 * (index % 3) for index in range(ARM_FRAMES)]
    if arm_b_offsets is None:
        arm_b_offsets = [0.10 * (index % 5) for index in range(ARM_FRAMES)]

    segments: list[tuple[str, str, str, int, float]] = [
        ("flat_head", "flat", "flat_drift", FLAT_FRAMES, 0.0),
        ("s1f_a", "A", "same_write", ARM_FRAMES, 0.0),
        ("flat_mid", "flat", "flat_drift", FLAT_FRAMES, 0.0),
        ("s1f_b", "B", "repeat_write", ARM_FRAMES, 0.0),
        ("flat_tail", "flat", "flat_drift", FLAT_FRAMES, drift_tail),
    ]
    journal: list[dict[str, object]] = []
    index = 0
    for stem, arm, journal_arm, count, shift in segments:
        for frame_index in range(count):
            if stem == "s1f_a":
                pattern_id = f"s1f_a{frame_index + 1:02d}"
                frame = h_frame(row_offset=arm_a_offsets[frame_index])
            elif stem == "s1f_b":
                pattern_id = f"s1f_b{frame_index + 1:02d}"
                frame = h_frame(row_offset=arm_b_offsets[frame_index])
            else:
                pattern_id = stem
                frame = flat_frame(row=ZERO_ROW + shift)
            filename = f"{index:04d}_{pattern_id}.npy"
            np.save(frames_dir / filename, frame.astype(np.uint16))
            journal.append(
                {
                    "schema": "issue247-s1f-jitter-attribution-capture/v1",
                    "event": "frame_saved",
                    "index": index,
                    "pattern_id": pattern_id,
                    "arm": journal_arm,
                    "frame_in_arm": frame_index,
                    "filename": filename,
                }
            )
            index += 1
    (root / "capture_journal.jsonl").write_text(
        "\n".join(json.dumps(row) for row in journal) + "\n", encoding="utf-8"
    )
    return root


def test_judge_end_to_end(tmp_path: Path) -> None:
    capture_dir = write_capture(tmp_path / "run_01", drift_tail=1.2)
    payload = SCRIPT.judge(capture_dir)

    assert payload["schema"] == SCRIPT.JUDGMENT_SCHEMA
    assert payload["n_frames"] == 96
    assert payload["arm_counts"] == {
        "A": ARM_FRAMES,
        "B": ARM_FRAMES,
        "flat_head": FLAT_FRAMES,
        "flat_mid": FLAT_FRAMES,
        "flat_tail": FLAT_FRAMES,
    }
    assert payload["frame_shape"] == list(FRAME_SHAPE)
    assert set(payload["zero_order_refs"]) == {"zr_head", "zr_mid", "zr_tail"}
    assert payload["zero_order_refs"]["zr_tail"]["row"] == pytest.approx(
        ZERO_ROW + 1.2, abs=0.05
    )
    assert set(payload["arms"]) == {"A", "B"}
    for arm, ref_key in (("A", "zr_head"), ("B", "zr_mid")):
        record = payload["arms"][arm]
        assert record["ref_key"] == ref_key
        assert record["n_frames"] == ARM_FRAMES
        assert len(record["positions"]) == ARM_FRAMES
        for key in ("pairwise_max_px", "rms_px", "std_row", "std_col"):
            assert isinstance(record[key], float)
        assert record["rms_px"] > 0.0
        assert record["pairwise_max_px"] >= record["rms_px"]

    assert set(payload["drift"]) == {"head_to_tail_px", "dr", "dc", "drift_material"}
    assert payload["drift"]["drift_material"] is True

    attribution = payload["attribution"]
    assert set(attribution) >= {
        "readout_floor_rms",
        "write_screen_rms",
        "drift_px",
        "ratio",
        "verdict",
        "verdict_label",
    }
    assert attribution["verdict"] in SCRIPT.VERDICTS
    assert attribution["readout_floor_rms"] == pytest.approx(
        payload["arms"]["A"]["rms_px"]
    )
    assert attribution["drift_px"] == pytest.approx(
        payload["drift"]["head_to_tail_px"]
    )

    cross = payload["crosscheck"]
    assert cross["run_05_intra_upload_ref_px"] == [1.68, 1.85]
    assert cross["A_rms_px"] == payload["arms"]["A"]["rms_px"]

    # 判读 JSON 必须能按 allow_nan=False 落盘（不得出现 inf/nan）。
    out = tmp_path / "judgment.json"
    SCRIPT.write_json(out, payload)
    reloaded = json.loads(out.read_text(encoding="utf-8"))
    assert reloaded["schema"] == SCRIPT.JUDGMENT_SCHEMA


def test_judge_falls_back_to_filename_glob_without_frame_events(tmp_path: Path) -> None:
    capture_dir = write_capture(tmp_path / "run_02")
    journal_path = capture_dir / "capture_journal.jsonl"
    journal_path.write_text(
        json.dumps({"event": "upload", "pattern_id": "s1f_a01"}) + "\n",
        encoding="utf-8",
    )
    payload = SCRIPT.judge(capture_dir)
    assert payload["arm_counts"]["A"] == ARM_FRAMES


def test_judge_missing_frames_is_an_error(tmp_path: Path) -> None:
    capture_dir = write_capture(tmp_path / "run_03")
    missing = next((capture_dir / "frames").glob("*_s1f_b12.npy"))
    missing.unlink()
    with pytest.raises(SCRIPT.JudgmentError):
        SCRIPT.judge(capture_dir)


def test_judge_missing_journal_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(SCRIPT.JudgmentError):
        SCRIPT.judge(tmp_path / "absent")


def test_judge_shape_mismatch_is_an_error(tmp_path: Path) -> None:
    capture_dir = write_capture(tmp_path / "run_04")
    bad = next((capture_dir / "frames").glob("*_s1f_a07.npy"))
    np.save(bad, np.zeros((FRAME_SHAPE[0], FRAME_SHAPE[1] + 1), dtype=np.uint16))
    with pytest.raises(SCRIPT.JudgmentError, match="shape"):
        SCRIPT.judge(capture_dir)


def test_judge_three_dimensional_frame_is_an_error(tmp_path: Path) -> None:
    capture_dir = write_capture(tmp_path / "run_05")
    bad = next((capture_dir / "frames").glob("*_flat_head.npy"))
    np.save(bad, np.zeros((4, 4, 4), dtype=np.uint16))
    with pytest.raises(SCRIPT.JudgmentError, match="2-D"):
        SCRIPT.judge(capture_dir)


def test_main_returns_nonzero_on_failure(tmp_path: Path) -> None:
    assert SCRIPT.main(["--capture-dir", str(tmp_path / "absent"), "--out", str(tmp_path)]) == 2


def test_main_writes_json_and_tables(tmp_path: Path) -> None:
    capture_dir = write_capture(tmp_path / "run_06")
    out_dir = tmp_path / "out"
    code = SCRIPT.main(
        ["--capture-dir", str(capture_dir), "--out", str(out_dir), "--judgment-name", "probe"]
    )
    assert code == 0
    payload = json.loads((out_dir / "probe.json").read_text(encoding="utf-8"))
    assert payload["schema"] == SCRIPT.JUDGMENT_SCHEMA
    assert "S1f" in (out_dir / "probe_tables.md").read_text(encoding="utf-8")

# ---------------------------------------------------------------------------
# §9 数据质量门禁（非归因判据）
# ---------------------------------------------------------------------------
def test_judge_rejects_no_light_capture(tmp_path: Path) -> None:
    capture_dir = write_capture(tmp_path / "run_no_light")
    for path in (capture_dir / "frames").glob("*.npy"):
        frame = np.full(FRAME_SHAPE, 15, dtype=np.uint16)
        frame[0, 0] = 41  # 无光: 对比度 26 < 50
        np.save(path, frame)
    with pytest.raises(SCRIPT.JudgmentError, match="no usable light"):
        SCRIPT.judge(capture_dir)
