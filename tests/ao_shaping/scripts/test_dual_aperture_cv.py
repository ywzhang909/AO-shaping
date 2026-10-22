"""``scripts/slm/dual_aperture_cv.py`` 的离线测试.

**为什么这些测试存在.** 双口径 CV 的全部价值在于「每个数字都带着它属于哪个
口径」, 而口径一旦搞错（最常见的是把窗口中心与画幅中心搞混）**报告仍然
看起来完全正常** —— 只是数字全错. 所以这里钉的是三件事:

1. **两个口径确实是两个口径**: 内层（口径 A, 小平坦区 rms）比外层（口径 B,
   目标方框 std/mean）更宽容, 纯均匀区域下两者都 ≈ 0.
2. **中心是被真正使用的**: 把中心改成画幅中心, CV 必须显著变化 —— 若脚本
   偷偷用画幅几何中心兜底, 这个断言会当场失败.
3. **口径 B 与运行时口径一致**: 端到端跑一遍假运行目录, 重算值与历史 CSV 里
   记录的 ``uniformity_cv`` 对得上（退出码 0）, 改错之后 ``--strict`` 退出码 1.

**完全离线**: 合成帧 + 临时目录, 不打开任何设备.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "slm" / "dual_aperture_cv.py"

FRAME_SIZE = 128
SPOT_CENTER = (70, 55)
SPOT_SIDE = 40
TARGET_SIDE = 40
PLATEAU_FRACTION = 0.4
NOISE_SEED = 20261022
HISTORY_NAME = "slm_model_in_loop_20261022_120000.csv"


def _load_script() -> Any:
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    spec = importlib.util.spec_from_file_location("_dual_aperture_cv", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["_dual_aperture_cv"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script() -> Any:
    return _load_script()


# ---------------------------------------------------------------------------
# 合成帧
# ---------------------------------------------------------------------------


def _shaped_frame() -> np.ndarray:
    """边长 40、中心 (70, 55) 的均匀方斑 + 平滑边缘跌落 + 固定种子噪声."""
    yy, xx = np.mgrid[0:FRAME_SIZE, 0:FRAME_SIZE]
    half_core = SPOT_SIDE / 2 - 2.0
    distance = (
        np.maximum(np.abs(xx - SPOT_CENTER[0]), np.abs(yy - SPOT_CENTER[1]))
        - half_core
    )
    profile = np.clip(1.0 - distance / 2.0, 0.0, 1.0)
    frame = 100.0 * profile
    rng = np.random.default_rng(NOISE_SEED)
    frame = frame + rng.normal(0.0, 2.0, size=frame.shape)
    return np.clip(frame, 0.0, None)


def _flat_frame() -> np.ndarray:
    """平场帧: 中心 (70, 55) 的高斯焦斑（无噪声 ⇒ argmax 恰落在中心像素）."""
    yy, xx = np.mgrid[0:FRAME_SIZE, 0:FRAME_SIZE]
    r2 = (xx - SPOT_CENTER[0]) ** 2 + (yy - SPOT_CENTER[1]) ** 2
    return 100.0 * np.exp(-r2 / (2.0 * 6.0**2))


def _target_region(frame: np.ndarray) -> np.ndarray:
    """按运行时的窗口公式（``half = target_side // 2``）取目标方框."""
    half = TARGET_SIDE // 2
    cx, cy = SPOT_CENTER
    return frame[cy - half : cy + half, cx - half : cx + half]


def _independent_cv(frame: np.ndarray) -> float:
    """口径 B 的独立实现（std/mean over the target box）."""
    region = _target_region(frame)
    mean = float(np.mean(region))
    return float(np.std(region)) / max(mean, 1e-10)


# ---------------------------------------------------------------------------
# 1. 两个口径确实是两个口径
# ---------------------------------------------------------------------------


def test_plateau_aperture_is_more_forgiving_than_target_aperture(
    script: Any,
) -> None:
    """内层口径（口径 A）必须比外层口径（口径 B）更宽容, 且两者都很小."""
    frame = _shaped_frame()
    metrics = script.aperture_metrics(
        frame, SPOT_CENTER, TARGET_SIDE, PLATEAU_FRACTION
    )
    assert metrics["cv_plateau_inner"] < metrics["cv_target_full"], (
        "内层口径没有比目标方框更宽容 —— 两个口径算成了同一个东西"
    )
    assert metrics["cv_plateau_inner"] < 0.1, metrics["cv_plateau_inner"]
    assert metrics["cv_target_full"] < 0.4, metrics["cv_target_full"]


def test_uniform_region_gives_zero_cv_on_both_apertures(script: Any) -> None:
    """目标方框内完全均匀 ⇒ 两个口径的 CV 都 ≈ 0."""
    frame = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=np.float64)
    half = TARGET_SIDE // 2
    cx, cy = SPOT_CENTER
    frame[cy - half : cy + half, cx - half : cx + half] = 100.0
    metrics = script.aperture_metrics(
        frame, SPOT_CENTER, TARGET_SIDE, PLATEAU_FRACTION
    )
    assert metrics["cv_target_full"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["cv_plateau_inner"] == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# 2. 坐标系守卫: 中心必须被真正使用
# ---------------------------------------------------------------------------


def test_centre_is_used_not_the_canvas_centre(script: Any) -> None:
    """把中心改成画幅中心, CV 必须显著变化."""
    frame = _shaped_frame()
    at_spot = script.aperture_metrics(
        frame, SPOT_CENTER, TARGET_SIDE, PLATEAU_FRACTION
    )
    at_canvas = script.aperture_metrics(
        frame, (FRAME_SIZE // 2, FRAME_SIZE // 2), TARGET_SIDE, PLATEAU_FRACTION
    )
    assert at_spot["cv_target_full"] < at_canvas["cv_target_full"] - 0.1, (
        "用画幅中心算出来的 CV 竟然比用真实光轴中心还小 —— "
        "脚本可能偷偷用了画幅几何中心"
    )
    assert at_spot["cv_target_full"] != pytest.approx(
        at_canvas["cv_target_full"], abs=1e-6
    )


def test_plateau_window_is_centred_and_inside_the_canvas(script: Any) -> None:
    """内层窗口与目标方框同心, 且被裁剪在画幅内."""
    x0, x1, y0, y1 = script.plateau_window(
        SPOT_CENTER, TARGET_SIDE, PLATEAU_FRACTION, (FRAME_SIZE, FRAME_SIZE)
    )
    assert (x1 - x0) == (y1 - y0) == 16
    assert (x0 + x1) // 2 == SPOT_CENTER[0]
    assert (y0 + y1) // 2 == SPOT_CENTER[1]
    assert 0 <= x0 < x1 <= FRAME_SIZE
    assert 0 <= y0 < y1 <= FRAME_SIZE


# ---------------------------------------------------------------------------
# 3. 指标字典的键
# ---------------------------------------------------------------------------


def test_aperture_metrics_exposes_both_apertures(script: Any) -> None:
    frame = _shaped_frame()
    metrics = script.aperture_metrics(
        frame, SPOT_CENTER, TARGET_SIDE, PLATEAU_FRACTION
    )
    for key in (
        "cv_target_full",
        "cv_plateau_inner",
        "ee_target",
        "peak",
        "mean_target",
        "plateau_side_px",
    ):
        assert key in metrics, key
    assert metrics["plateau_side_px"] == 16.0


# ---------------------------------------------------------------------------
# 4. 端到端
# ---------------------------------------------------------------------------


def _make_run_dir(tmp_path: Path, *, uniformity_cv: float) -> Path:
    """搭一个假的 ``slm-model-in-loop --debug`` 运行目录."""
    run_dir = tmp_path / "run"
    probes = run_dir / "probes"
    probes.mkdir(parents=True)
    flat = _flat_frame()
    shaped = _shaped_frame()
    np.savez_compressed(probes / "round_00_flat_00.npz", frame=flat)
    np.savez_compressed(probes / "round_01_shaped_00.npz", frame=shaped)

    manifest = [
        {
            "file": "round_00_flat_00.npz",
            "round": 0,
            "split": "flat",
            "family": None,
            "index": 0,
            "exposure_ms": 0.02,
            "timestamp_utc": "2026-10-22T00:00:00.000000+00:00",
            "correlation": None,
        },
        {
            "file": "round_01_shaped_00.npz",
            "round": 1,
            "split": "shaped",
            "family": None,
            "index": 0,
            "exposure_ms": 0.02,
            "timestamp_utc": "2026-10-22T00:01:00.000000+00:00",
            "correlation": 0.91,
        },
    ]
    (probes / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (run_dir / "run_config.json").write_text(
        json.dumps(
            {
                "target_side": TARGET_SIDE,
                "cam_type": "sim",
                "region": 64,
                "far_field_padding": 8,
                "cam_size": FRAME_SIZE,
                "panel_span_px": 900.0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    with (run_dir / HISTORY_NAME).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["stage", "round", "uniformity_cv", "accepted", "reason"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "stage": "calibration",
                "round": 0,
                "uniformity_cv": f"{_independent_cv(flat):.10g}",
                "accepted": "True",
                "reason": "baseline",
            }
        )
        writer.writerow(
            {
                "stage": "round",
                "round": 1,
                "uniformity_cv": f"{uniformity_cv:.10g}",
                "accepted": "True",
                "reason": "accepted",
            }
        )
    return run_dir


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def test_end_to_end_writes_two_rows_per_frame_and_passes_self_check(
    script: Any, tmp_path: Path
) -> None:
    """端到端: 退出码 0, CSV 里该帧有两行（两个口径）, 自检通过."""
    shaped = _shaped_frame()
    expected_cv = _independent_cv(shaped)
    run_dir = _make_run_dir(tmp_path, uniformity_cv=expected_cv)
    out_csv = tmp_path / "out.csv"

    code = script.main(["--from-dir", str(run_dir), "--csv", str(out_csv)])
    assert code == 0

    rows = _read_csv(out_csv)
    assert len(rows) == 2, rows
    assert {row["aperture"] for row in rows} == {"target_full", "plateau_inner"}
    by_aperture = {row["aperture"]: row for row in rows}

    target_row = by_aperture["target_full"]
    assert float(target_row["cv"]) == pytest.approx(expected_cv, rel=1e-9)
    assert float(target_row["cv_check_rel_delta"]) == pytest.approx(0.0, abs=1e-9)
    assert float(target_row["cv_recorded"]) == pytest.approx(expected_cv, rel=1e-9)
    # 中心是从平场帧重算出来的, 不是画幅中心。
    assert (target_row["center_x"], target_row["center_y"]) == (
        str(SPOT_CENTER[0]),
        str(SPOT_CENTER[1]),
    )
    assert target_row["target_side_px"] == str(TARGET_SIDE)
    assert float(target_row["plateau_side_px"]) == 16.0

    plateau_row = by_aperture["plateau_inner"]
    assert float(plateau_row["cv"]) < float(target_row["cv"])
    # 口径标注必须逐字出现在每一行里。
    for row in rows:
        assert row["aperture_definition"] == (
            script.APERTURE_DEFINITIONS[row["aperture"]]
        )
    assert (
        script.APERTURE_DEFINITIONS["target_full"]
        != script.APERTURE_DEFINITIONS["plateau_inner"]
    )


def test_strict_fails_when_the_recorded_cv_is_wrong(
    script: Any, tmp_path: Path
) -> None:
    """把历史 CSV 里的 ``uniformity_cv`` 改错 ⇒ ``--strict`` 必须退出码 1."""
    shaped = _shaped_frame()
    expected_cv = _independent_cv(shaped)
    run_dir = _make_run_dir(tmp_path, uniformity_cv=expected_cv)
    history = run_dir / HISTORY_NAME
    history.write_text(
        history.read_text(encoding="utf-8").replace(f"{expected_cv:.10g}", "0.9"),
        encoding="utf-8",
    )

    out_csv = tmp_path / "out.csv"
    code = script.main(
        ["--from-dir", str(run_dir), "--csv", str(out_csv), "--strict"]
    )
    assert code == 1, "改错运行时记录的 uniformity_cv 之后, --strict 没有报警"

    # 非 strict 时只报警不失败, 但偏差必须被记下来。
    code = script.main(["--from-dir", str(run_dir), "--csv", str(out_csv)])
    assert code == 0
    rows = _read_csv(out_csv)
    delta = float(rows[0]["cv_check_rel_delta"])
    assert delta > 0.05, delta


def test_load_session_reads_config_frames_and_history(
    script: Any, tmp_path: Path
) -> None:
    """``load_session`` 必须把 config / frames / history 都读出来."""
    shaped = _shaped_frame()
    run_dir = _make_run_dir(tmp_path, uniformity_cv=_independent_cv(shaped))
    session = script.load_session(run_dir)
    assert session.config["target_side"] == TARGET_SIDE
    assert session.history_path is not None
    assert len(session.history) == 2
    splits = sorted(record.split for record in session.frames)
    assert splits == ["flat", "shaped"]
    shaped_records = [
        record for record in session.frames if record.split == "shaped"
    ]
    assert len(shaped_records) == 1
    assert shaped_records[0].round == 1
    assert shaped_records[0].frame.shape == (FRAME_SIZE, FRAME_SIZE)


def test_rounds_filter_does_not_hide_the_flat_frame(script: Any, tmp_path: Path) -> None:
    """``--rounds 1`` 不能把定位中心所需的平场帧一起滤掉."""
    shaped = _shaped_frame()
    run_dir = _make_run_dir(tmp_path, uniformity_cv=_independent_cv(shaped))
    out_csv = tmp_path / "out.csv"
    code = script.main(
        [
            "--from-dir",
            str(run_dir),
            "--rounds",
            "1",
            "--csv",
            str(out_csv),
        ]
    )
    assert code == 0, "只统计第 1 轮时找不到平场帧 —— 中心定位被轮次过滤误伤"
    rows = _read_csv(out_csv)
    assert len(rows) == 2
    assert {row["round"] for row in rows} == {"1"}
    assert (rows[0]["center_x"], rows[0]["center_y"]) == (
        str(SPOT_CENTER[0]),
        str(SPOT_CENTER[1]),
    )


def test_missing_flat_frame_exits_with_code_2(script: Any, tmp_path: Path) -> None:
    """既没有 preflight/flat.npz 也没有 split=flat 的帧 ⇒ 退出码 2, 不兜底."""
    run_dir = tmp_path / "run"
    probes = run_dir / "probes"
    probes.mkdir(parents=True)
    np.savez_compressed(probes / "round_01_shaped_00.npz", frame=_shaped_frame())
    (probes / "manifest.json").write_text(
        json.dumps(
            [
                {
                    "file": "round_01_shaped_00.npz",
                    "round": 1,
                    "split": "shaped",
                    "family": None,
                    "index": 0,
                    "exposure_ms": 0.02,
                    "timestamp_utc": "2026-10-22T00:01:00.000000+00:00",
                    "correlation": 0.91,
                }
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "run_config.json").write_text(
        json.dumps({"target_side": TARGET_SIDE, "cam_type": "sim"}),
        encoding="utf-8",
    )
    out_csv = tmp_path / "out.csv"
    code = script.main(["--from-dir", str(run_dir), "--csv", str(out_csv)])
    assert code == 2, "找不到平场帧时必须明确报错退出（退出码 2）"
    assert not out_csv.exists()

    # 显式给出中心后同一目录可用。
    code = script.main(
        [
            "--from-dir",
            str(run_dir),
            "--center",
            f"{SPOT_CENTER[0]},{SPOT_CENTER[1]}",
            "--csv",
            str(out_csv),
        ]
    )
    assert code == 0
    rows = _read_csv(out_csv)
    assert (rows[0]["center_x"], rows[0]["center_y"]) == (
        str(SPOT_CENTER[0]),
        str(SPOT_CENTER[1]),
    )


def test_markdown_output_labels_every_cv_with_its_aperture(
    script: Any, tmp_path: Path
) -> None:
    """MD 里每个 CV 数字后面必须写明是哪个口径."""
    shaped = _shaped_frame()
    run_dir = _make_run_dir(tmp_path, uniformity_cv=_independent_cv(shaped))
    out_md = tmp_path / "out.md"
    out_json = tmp_path / "out.json"
    code = script.main(
        [
            "--from-dir",
            str(run_dir),
            "--md",
            str(out_md),
            "--json",
            str(out_json),
        ]
    )
    assert code == 0
    text = out_md.read_text(encoding="utf-8")
    assert "口径 B：目标方框 std/mean" in text
    assert "口径 A：小平坦区 rms/mean" in text
    assert script.APERTURE_DEFINITIONS["target_full"] in text
    assert script.APERTURE_DEFINITIONS["plateau_inner"] in text

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["aperture_definitions"] == script.APERTURE_DEFINITIONS
    assert payload["self_check"]["passed"] is True
    assert payload["self_check"]["reason"] == "ok"
    assert len(payload["rows"]) == 2


def test_self_check_is_reported_as_not_performed_without_a_history_csv(
    script: Any, tmp_path: Path
) -> None:
    """没有历史 CSV 时自检必须明确说"未做", 而不是谎称通过."""
    shaped = _shaped_frame()
    run_dir = _make_run_dir(tmp_path, uniformity_cv=_independent_cv(shaped))
    (run_dir / HISTORY_NAME).unlink()
    out_json = tmp_path / "out.json"
    code = script.main(["--from-dir", str(run_dir), "--json", str(out_json)])
    assert code == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["self_check"]["performed"] is False
    assert payload["self_check"]["reason"] == "no_history_csv"
    assert payload["self_check"]["passed"] is False
