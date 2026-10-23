"""离线报告生成器 ``scripts/slm/generate_capability_report.py`` 的行为测试。

用合成 run.json（2 块 + final，帧用小随机数组）验证：

* 报告与两张图生成成功；
* 逐块表 / 末尾对比表的数值与 run.json 一致（含 Δ÷3σ）；
* 缺 final 帧时跳过对比图且正文注明（不伪造数据）；
* ``--force`` 的覆盖语义；
* ``delta_over_3sigma`` 的边界行为。
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "slm" / "generate_capability_report.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("gen_capability_report_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


from scripts.slm.generate_capability_report import measure_final


def _make_run_dir(
    tmp_path: Path,
    name: str = "square_20260101_000000",
    *,
    with_final_section: bool = True,
    with_final_frames: bool = True,
    n_blocks: int = 2,
) -> Path:
    run_dir = tmp_path / name
    run_dir.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(7)

    blocks = []
    for i in range(1, n_blocks + 1):
        baseline = 1.10 - 0.02 * (i - 1)
        candidate = baseline - 0.02
        blocks.append(
            dict(
                block=i,
                baseline=baseline,
                candidate=candidate,
                delta=candidate - baseline,
                model_before=2.0 - 0.1 * (i - 1),
                model_after=2.0 - 0.1 * (i - 1) - 0.05,
                drift=0.001 * (i - 1) if i % 2 else -0.002,
                zero_fraction_baseline=0.10,
                zero_fraction_candidate=0.09,
            )
        )
        for label in ("A1", "B1", "B2", "A2"):
            np.save(run_dir / f"block{i}_{label}_1.npy", rng.random((200, 200)).astype(np.float32))

    records = dict(
        target_side_px=84.85,
        carrier_period_px=32.0,
        carrier_axis="x",
        blocks=blocks,
        three_sigma_resolution=0.01,
        checkpoint="D:\\chk\\refit_20260101",
        started_utc="2026-01-01T00:00:00+00:00",
        spot_xy=[100.0, 100.0],
    )
    if with_final_section:
        records["final"] = dict(
            shift_px=4.0,
            best_block=n_blocks,
            best_cv=blocks[-1]["candidate"],
            measured=dict(
                best=dict(
                    at_zero=dict(cv=1.0, mean_core=100.0, peak=500.0, zero_fraction=0.10),
                    at_plus=dict(cv=0.3, mean_core=1.0, peak=10.0, zero_fraction=0.001),
                    at_minus=dict(cv=0.5, mean_core=0.6, peak=9.0, zero_fraction=0.001),
                ),
                best_plus_carrier=dict(
                    at_zero=dict(cv=2.0, mean_core=5.0, peak=60.0, zero_fraction=0.006),
                    at_plus=dict(cv=0.15, mean_core=2.0, peak=30.0, zero_fraction=0.001),
                    at_minus=dict(cv=0.8, mean_core=0.3, peak=9.0, zero_fraction=0.0005),
                ),
            ),
        )
    if with_final_frames:
        np.save(run_dir / "final_best.npy", rng.random((200, 200)).astype(np.float32))
        np.save(run_dir / "final_best_plus_carrier.npy", rng.random((200, 200)).astype(np.float32))
    (run_dir / "run.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return run_dir


def _run_generator(*args) -> dict:
    module = _load_module()
    return module.main(list(args))


def _parse_table_row(text: str, prefix: str) -> list[str]:
    """取正文中以 ``| <prefix>`` 开头的第一个表格行，拆成单元格。"""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(f"| {prefix}"):
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            return cells
    raise AssertionError(f"未找到以 {prefix!r} 开头的表格行")


def test_full_run_generates_report_and_figures(tmp_path) -> None:
    run_dir = _make_run_dir(tmp_path)
    out_dir = tmp_path / "out"
    summary = _run_generator("--run-dir", str(run_dir), "--out-dir", str(out_dir))

    report = out_dir / "capability_square_20260101_000000.md"
    figs = out_dir / "figures"
    blocks_png = figs / "20260101_capability_square_20260101_000000_blocks.png"
    final_png = figs / "20260101_000000_final_compare.png"
    assert report.is_file()
    assert blocks_png.is_file()
    assert final_png.is_file()
    assert summary["blocks"] == 2 and summary["final_present"] is True
    assert summary["figures"]["blocks"]["ok"] is True
    assert summary["figures"]["final_compare"]["ok"] is True

    text = report.read_text(encoding="utf-8")
    for heading in ("## 摘要", "## 1. 方法与边界", "## 2. 闭环逐块表",
                    "## 3. 末尾对比", "## 4. 观察与限制", "## 5. 复现"):
        assert heading in text
    assert "模型方向尚未验证，本报告只陈述实测" in text
    assert run_dir.name in text
    assert summary["stamp"] == "20260101_000000"


def test_block_table_values_match_run_json(tmp_path) -> None:
    run_dir = _make_run_dir(tmp_path)
    out_dir = tmp_path / "out"
    _run_generator("--run-dir", str(run_dir), "--out-dir", str(out_dir))
    text = (out_dir / "capability_square_20260101_000000.md").read_text(encoding="utf-8")
    records = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    resolution = records["three_sigma_resolution"]

    for block in records["blocks"]:
        cells = _parse_table_row(text, f"{block['block']} ")
        # 列: cells[0]=块, 1=基线, 2=候选, 3=Δ, 4=Δ÷3σ, 5=漂移, 6=零级·基线,
        #     7=零级·候选, 8=模型前→后
        assert float(cells[0]) == block["block"]
        assert abs(float(cells[1]) - block["baseline"]) < 5e-6
        assert abs(float(cells[2]) - block["candidate"]) < 5e-6
        assert abs(float(cells[3]) - block["delta"]) < 5e-6
        expected_ratio = block["delta"] / resolution
        assert abs(float(cells[4]) - expected_ratio) < 5e-3
        assert abs(float(cells[5]) - block["drift"]) < 5e-6
        assert abs(float(cells[6]) - block["zero_fraction_baseline"]) < 5e-5
        assert abs(float(cells[7]) - block["zero_fraction_candidate"]) < 5e-5
        model_before, model_after = cells[8].split("→")
        assert abs(float(model_before.strip()) - block["model_before"]) < 5e-5
        assert abs(float(model_after.strip()) - block["model_after"]) < 5e-5


def test_final_table_comes_from_measured_displacement(tmp_path) -> None:
    """§3 的读数必须来自**实测位移**，不再沿用 run.json 的 at_plus/at_minus（旧口径已作废）。

    2026-10-23：SLM 竖向安装 ⇒ 面板 x 向斜坡在 CCD 上是 y 向位移；按 x 轴摆框量到的是
    背景不对称。这条测试锁住新口径：表格三列（零点 / 载波@零点 / 载波@实测落点），
    且正文**不出现** at_plus/at_minus 这两个已经失效的键名。
    """
    run_dir = _make_run_dir(tmp_path)
    out_dir = tmp_path / "out"
    _run_generator("--run-dir", str(run_dir), "--out-dir", str(out_dir))
    text = (out_dir / "capability_square_20260101_000000.md").read_text(encoding="utf-8")

    assert "实测落点" in text
    # 旧口径的框名只允许出现在「已作废」的更正说明里（§4 明确写了这条更正）
    if "at_plus" in text or "at_minus" in text:
        assert "已作废" in text, "旧口径框名只能随更正说明出现"
        section3 = text.split("## 3.")[1].split("## 4.")[0]
        assert "at_plus" not in section3 and "at_minus" not in section3, "§3 表格不得再用旧框名"

    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    measured = measure_final(run, run_dir)
    assert measured is not None
    stats = measured["stats"]
    assert stats, "末帧存在时必须给出三处读数"

    cv_row = _parse_table_row(text, "CV（目标掩膜）")
    mean_row = _parse_table_row(text, "mean_core（目标掩膜）")
    for col, key in ((1, "best_at_zero"), (2, "carrier_at_zero"), (3, "carrier_at_displaced")):
        assert abs(float(cv_row[col]) - stats[key]["cv"]) < 5e-5, key
        assert abs(float(mean_row[col]) - stats[key]["mean_core"]) < 5e-4, key

    displacement = measured["displacement"]
    if displacement is not None:
        # 位移量必须原样出现在正文（方向与量值都来自实测）
        assert f"{abs(float(displacement['dx'])):.1f}" in text or f"{abs(float(displacement['dy'])):.1f}" in text


def test_missing_final_frames_skip_figure_and_note_in_md(tmp_path) -> None:
    run_dir = _make_run_dir(tmp_path, with_final_frames=False)
    out_dir = tmp_path / "out"
    summary = _run_generator("--run-dir", str(run_dir), "--out-dir", str(out_dir))

    final_png = out_dir / "figures" / "20260101_000000_final_compare.png"
    assert not final_png.exists(), "帧缺失时不得生成对比图（不伪造数据）"
    assert summary["figures"]["final_compare"]["ok"] is False
    assert "缺少最终帧" in summary["figures"]["final_compare"]["reason"]

    text = (out_dir / "capability_square_20260101_000000.md").read_text(encoding="utf-8")
    assert "末尾对比图未生成" in text
    assert "缺少最终帧" in text
    # 逐块图不受影响
    assert (out_dir / "figures" / "20260101_capability_square_20260101_000000_blocks.png").is_file()


def test_missing_final_section_degrades_gracefully(tmp_path) -> None:
    run_dir = _make_run_dir(tmp_path, with_final_section=False, with_final_frames=False)
    out_dir = tmp_path / "out"
    summary = _run_generator("--run-dir", str(run_dir), "--out-dir", str(out_dir))
    assert summary["final_present"] is False
    text = (out_dir / "capability_square_20260101_000000.md").read_text(encoding="utf-8")
    assert "末尾对比未执行" in text
    assert not (out_dir / "figures" / "20260101_000000_final_compare.png").exists()


def test_force_semantics(tmp_path) -> None:
    run_a = _make_run_dir(tmp_path, "square_20260101_000000")
    run_b = _make_run_dir(tmp_path, "square_20260102_000000")
    out_dir = tmp_path / "out"
    report_name = "capability_report.md"
    common = ["--report-name", report_name, "--out-dir", str(out_dir)]

    _run_generator("--run-dir", str(run_a), *common)
    report = out_dir / report_name
    assert report.is_file()

    # 同一次运行重跑（报告已指向 run A）→ 允许
    _run_generator("--run-dir", str(run_a), *common)
    # 指向另一次运行（run B）→ 默认拒绝
    with pytest.raises(SystemExit):
        _run_generator("--run-dir", str(run_b), *common)
    text = report.read_text(encoding="utf-8")
    assert run_b.name not in text
    # --force → 允许覆盖
    _run_generator("--run-dir", str(run_b), *common, "--force")
    text = report.read_text(encoding="utf-8")
    assert run_b.name in text
    assert run_a.name not in text


def test_missing_run_json_exits() -> None:
    module = _load_module()
    with pytest.raises(SystemExit):
        module.main(["--run-dir", str(REPO / "data" / "debug" / "nonexistent_run_dir")])


def test_delta_over_3sigma() -> None:
    module = _load_module()
    assert module.delta_over_3sigma(-0.02, 0.01) == pytest.approx(-2.0)
    assert module.delta_over_3sigma(0.02, 0.01) == pytest.approx(2.0)
    # 分辨率缺失 / 非正 / 非法输入 → NaN
    assert math.isnan(module.delta_over_3sigma(0.02, 0.0))
    assert math.isnan(module.delta_over_3sigma(0.02, None))
    assert math.isnan(module.delta_over_3sigma(float("nan"), 0.01))
    assert math.isnan(module.delta_over_3sigma("bad", 0.01))


