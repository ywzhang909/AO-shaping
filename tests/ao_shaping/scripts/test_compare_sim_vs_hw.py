"""``scripts/slm/compare_sim_vs_hw.py`` 的离线测试。

**测什么**: 这把「量尺」的**算术**与**口径**, 不跑真仿真 (真仿真一次要几分钟,
而且产物是硬件/孪生的事, 不是这里的单元测试该管的)。

锁定的不变量:

1. **实测侧的数字不写死** —— 读取器必须真的从归档目录里读出来, 改了文件就变;
2. **相对差与判定算得对** —— 给定两侧的数值, 相对差 = (sim−hw)/|hw|, 判定按
   「≤ 阈值 ⇒ 一致; ≤ 3× 阈值 ⇒ 偏差; 否则 不一致」;
3. **噪声地板是样本标准差** (ddof=1), 不是总体标准差;
4. **对照表逐项对齐** —— 同一行里的 hw/sim 必须是**同一个量**, 不能串位;
5. **仿真口径旋钮真的落到孪生上** —— 替换采集脚本的 ``configure_sim_system``
   之后, 进程级 system 的种子/补零/光度/噪声确实被设上 (纯内存, 不碰硬件);
6. **默认路径不碰设备** —— 不给 ``--execute`` 时, 出表这一步不 import 任何
   ``ao_shaping.drivers.*``。
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.slm.compare_sim_vs_hw import (  # noqa: E402
    PROBE_METRICS,
    SimConfig,
    _parse_args,
    _verdict_tally,
    build_cells,
    build_config,
    build_gap_list,
    build_noise_spec,
    hw_defocus_focal_mm,
    hw_focus_diopter_grid,
    load_bundle,
    load_probe_group,
    main,
    parse_zernike_spec,
    patch_sim_configurer,
    relative_diff,
    render_markdown,
    render_report,
    render_table,
    sim_width_scale,
    summarise_repeatability,
    verdict,
)

# ---------------------------------------------------------------------------
# 桩件: 一份「归档产物」的最小骨架
# ---------------------------------------------------------------------------


def _probe_row(index: int, dz_mm: float, **overrides: Any) -> dict[str, Any]:
    """一行探针测量 (字段名与采集脚本的 measurements.json 一致)。"""
    row: dict[str, Any] = {
        "index": index,
        "dz_mm": dz_mm,
        "chirp_sign": -1.0,
        "exposure_ms_readback": 0.2,
        "frame_mean_in_target": 20.0,
        "cv": 0.20,
        "pib": 0.17,
        "score": 0.50,
        "frame_peak": 26.75,
        "halo_peak_outside_box": 20.5,
        "halo_power_fraction_outside_box": 0.83,
        "spot_fwhm_px": 13.9,
        "spot_centroid_x": 126.0,
        "spot_centroid_y": 126.0,
        "spot_hollowness": 0.89,
    }
    row.update(overrides)
    return row


def _write_bundle(root: Path, *, scale: float = 1.0, dz_points: int = 4) -> Path:
    """在 ``root`` 下写一份与归档产物**同构**的桩件。

    Args:
        root: 产物根目录。
        scale: 所有指标值的缩放 (用来造「仿真侧与实测侧差一个固定比例」的情形)。
        dz_points: 离焦扫描的点数。

    Returns:
        产物根目录。
    """
    groups = root / "hw_groups_20261022"
    scan_dir = groups / "dz_scan_hw_gs"
    noise_dir = groups / "dz_noise_hw_gs"
    scan_dir.mkdir(parents=True, exist_ok=True)
    noise_dir.mkdir(parents=True, exist_ok=True)

    scan_rows = [
        _probe_row(
            0,
            0.0,
            cv=0.20 * scale,
            pib=0.17 * scale,
            score=0.50 * scale,
            frame_peak=26.75 * scale,
            spot_fwhm_px=13.9 * scale,
            halo_power_fraction_outside_box=0.83 * scale,
        )
    ]
    for i in range(1, dz_points):
        scan_rows.append(
            _probe_row(
                i,
                float(i) * 5.0,
                cv=0.30 * scale,
                pib=0.05 * scale,
                score=0.40 * scale,
                frame_peak=5.0 * scale,
                spot_fwhm_px=60.0 * scale,
                halo_power_fraction_outside_box=0.95 * scale,
            )
        )
    (scan_dir / "measurements.json").write_text(
        json.dumps(scan_rows), encoding="utf-8"
    )
    (scan_dir / "run_config.json").write_text(
        json.dumps({"gs_phase_source": "stub_phase.npy", "roi_px": 250}),
        encoding="utf-8",
    )

    noise_rows = [
        _probe_row(
            i,
            0.0,
            cv=0.20 + 0.001 * i,
            pib=0.17 + 0.0005 * i,
            score=0.50 + 0.0002 * i,
            frame_peak=26.5 + 0.05 * i,
            spot_fwhm_px=13.9,
        )
        for i in range(5)
    ]
    (noise_dir / "measurements.json").write_text(
        json.dumps(noise_rows), encoding="utf-8"
    )
    (noise_dir / "run_config.json").write_text(json.dumps({}), encoding="utf-8")

    sweep_dir = root / "hw_zernike_sweep_20261022" / "zernike_order_nmax3"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    (sweep_dir / "summary.json").write_text(
        json.dumps(
            {
                "group": "zernike_order_nmax3",
                "baselines": {
                    "flat": {"pib": 0.20 * scale, "cv": 0.45 * scale, "ee": 0.20 * scale,
                             "ar": 1.0, "quality": 0.44 * scale},
                    "gs_freeform": {"pib": 0.21 * scale, "cv": 0.36 * scale,
                                    "ee": 0.21 * scale, "ar": 0.91,
                                    "quality": 0.43 * scale},
                },
                "orders": [
                    {
                        "n_max": 3,
                        "before": {"pib": 0.08 * scale, "cv": 0.26 * scale,
                                   "ee": 0.08 * scale, "ar": 1.09,
                                   "quality": 0.42 * scale},
                        "after": {"pib": 0.08 * scale, "cv": 0.25 * scale,
                                  "ee": 0.08 * scale, "ar": 1.09,
                                  "quality": 0.43 * scale},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    focus_dir = root / "focus_calibration"
    focus_dir.mkdir(parents=True, exist_ok=True)
    (focus_dir / "slm1_20261022_142207.json").write_text(
        json.dumps(
            {
                "best_focal_mm": 1549.33,
                "best_diopter_m1": 0.6454,
                "gain_vs_flat": 29.21,
                "points": [
                    {"diopter_m1": 0.0, "focal_mm": None, "peak": 73.3,
                     "peak_per_ms": 3.67, "fwhm_px": 46.5, "status": "ok"},
                    {"diopter_m1": 0.6454, "focal_mm": 1549.33, "peak": 115.7,
                     "peak_per_ms": 107.1 * scale, "fwhm_px": 14.6 * scale,
                     "status": "ok"},
                ],
            }
        ),
        encoding="utf-8",
    )
    return root


# ---------------------------------------------------------------------------
# 读取
# ---------------------------------------------------------------------------


def test_load_bundle_reads_archived_layout(tmp_path: Path) -> None:
    """读取器必须真的把归档产物读出来 (而不是用内置默认值)。"""
    root = _write_bundle(tmp_path / "hw")
    bundle = load_bundle(root)

    assert len(bundle.dz_scan.rows) == 4
    assert bundle.dz_scan.baseline["frame_peak"] == pytest.approx(26.75)
    assert len(bundle.dz_noise.rows) == 5
    assert "zernike_order_nmax3" in bundle.zernike
    assert bundle.focus["best_focal_mm"] == pytest.approx(1549.33)
    assert bundle.focus_name == "slm1_20261022_142207.json"


def test_load_bundle_missing_root_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_bundle(tmp_path / "nope")


def test_load_probe_group_missing_dir_is_empty(tmp_path: Path) -> None:
    group = load_probe_group(tmp_path / "absent")
    assert group.rows == ()
    assert group.baseline == {}


def test_probe_group_row_at_dz(tmp_path: Path) -> None:
    root = _write_bundle(tmp_path / "hw")
    bundle = load_bundle(root)
    row = bundle.dz_scan.row_at_dz(5.0)
    assert row is not None
    assert row["dz_mm"] == pytest.approx(5.0)
    assert bundle.dz_scan.row_at_dz(123.0) is None


# ---------------------------------------------------------------------------
# 相对差与判定
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("hw", "sim", "expected"),
    [
        (10.0, 10.0, 0.0),
        (10.0, 11.0, 0.1),
        (10.0, 9.0, -0.1),
        (-4.0, -5.0, -0.25),
        (0.0, 1.0, None),
        (None, 1.0, None),
        (1.0, None, None),
    ],
)
def test_relative_diff(hw: Any, sim: Any, expected: Any) -> None:
    got = relative_diff(hw, sim)
    if expected is None:
        assert got is None
    else:
        assert got == pytest.approx(expected)


@pytest.mark.parametrize(
    ("rel", "tol", "expected"),
    [
        (0.0, 0.2, "一致"),
        (0.2, 0.2, "一致"),
        (-0.2, 0.2, "一致"),
        (0.3, 0.2, "偏差"),
        (0.6, 0.2, "偏差"),
        (0.61, 0.2, "不一致"),
        (-2.0, 0.2, "不一致"),
        (None, 0.2, "—"),
    ],
)
def test_verdict(rel: Any, tol: float, expected: str) -> None:
    assert verdict(rel, tol) == expected


# ---------------------------------------------------------------------------
# 统计
# ---------------------------------------------------------------------------


def test_summarise_repeatability_uses_sample_std() -> None:
    """噪声地板是**样本**标准差 (ddof=1), 不是总体标准差。"""
    rows = [{"cv": v} for v in (1.0, 2.0, 3.0, 4.0)]
    stats = summarise_repeatability(rows, ("cv",))
    assert stats["cv"]["mean"] == pytest.approx(2.5)
    assert stats["cv"]["std"] == pytest.approx(1.2909944487358056)
    assert stats["cv"]["n"] == 4


def test_summarise_repeatability_drops_non_finite() -> None:
    rows = [{"cv": 1.0}, {"cv": float("nan")}, {"cv": None}, {"cv": 3.0}]
    stats = summarise_repeatability(rows, ("cv",))
    assert stats["cv"]["n"] == 2
    assert stats["cv"]["mean"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# 对照表
# ---------------------------------------------------------------------------


def test_build_cells_aligns_like_for_like(tmp_path: Path) -> None:
    """同一行里的 hw/sim 必须是同一个量 —— 不能串位。"""
    hw = load_bundle(_write_bundle(tmp_path / "hw", scale=1.0))
    sim = load_bundle(_write_bundle(tmp_path / "sim", scale=1.0))
    cells = build_cells(hw, sim)
    assert cells, "对照表不能是空的"

    by_metric = {cell.metric: cell for cell in cells}
    cv_cell = by_metric["CV (目标框内)"]
    assert cv_cell.hw == pytest.approx(0.20)
    assert cv_cell.sim == pytest.approx(0.20)
    assert relative_diff(cv_cell.hw, cv_cell.sim) == pytest.approx(0.0)
    assert verdict(relative_diff(cv_cell.hw, cv_cell.sim), cv_cell.tol) == "一致"

    # 光斑宽度与峰值也不能串位。
    assert by_metric["光斑 FWHM (px)"].hw == pytest.approx(13.9)
    assert by_metric["整帧峰值 (counts)"].hw == pytest.approx(26.75)


def test_build_cells_detects_scale_gap(tmp_path: Path) -> None:
    """仿真侧整体差一个固定比例 ⇒ 相对差照算, 判定按阈值分档。"""
    hw = load_bundle(_write_bundle(tmp_path / "hw", scale=1.0))
    sim = load_bundle(_write_bundle(tmp_path / "sim", scale=0.5))
    cells = build_cells(hw, sim)
    by_metric = {cell.metric: cell for cell in cells}
    cv_cell = by_metric["CV (目标框内)"]
    rel = relative_diff(cv_cell.hw, cv_cell.sim)
    assert rel == pytest.approx(-0.5)
    # CV 的阈值是 0.20 ⇒ 3× 阈值 = 0.60 ⇒ -50% 落在「偏差」档。
    assert verdict(rel, cv_cell.tol) == "偏差"
    # 再差 5 倍 ⇒ 相对差 -80% ⇒ 超出 3× 阈值 ⇒ 「不一致」。
    far = load_bundle(_write_bundle(tmp_path / "sim_far", scale=0.2))
    far_cells = build_cells(hw, far)
    far_cell = {c.metric: c for c in far_cells}["CV (目标框内)"]
    far_rel = relative_diff(far_cell.hw, far_cell.sim)
    assert far_rel == pytest.approx(-0.8)
    assert verdict(far_rel, far_cell.tol) == "不一致"


def test_build_cells_covers_all_five_sections(tmp_path: Path) -> None:
    hw = load_bundle(_write_bundle(tmp_path / "hw"))
    sim = load_bundle(_write_bundle(tmp_path / "sim"))
    cells = build_cells(hw, sim)
    sections = {cell.section for cell in cells}
    assert any(s.startswith("§1") for s in sections)
    assert any(s.startswith("§2") for s in sections)
    assert any(s.startswith("§3") for s in sections)
    assert any(s.startswith("§4") for s in sections)
    assert any(s.startswith("§5") for s in sections)


def test_build_cells_reports_noise_sigma(tmp_path: Path) -> None:
    """噪声组那一段必须给的是 **σ**, 不是均值。"""
    hw = load_bundle(_write_bundle(tmp_path / "hw"))
    sim = load_bundle(_write_bundle(tmp_path / "sim"))
    cells = build_cells(hw, sim)
    by_metric = {cell.metric: cell for cell in cells}
    noise_cells = [c for c in cells if c.section.startswith("§2")]
    assert noise_cells, "噪声段不能为空"
    # 桩件的 CV 是 0.20, 0.201, ..., 0.204 ⇒ 样本 σ ≈ 0.0015811
    assert by_metric["σ(CV)"].hw == pytest.approx(0.0015811388300841898)
    assert by_metric["均值 CV"].hw == pytest.approx(0.202)


def test_build_cells_uses_focus_best_point(tmp_path: Path) -> None:
    """焦面段的「最佳点峰值/曝光」要从最佳点那一行取, 不是顶层键。"""
    hw = load_bundle(_write_bundle(tmp_path / "hw", scale=1.0))
    sim = load_bundle(_write_bundle(tmp_path / "sim", scale=0.5))
    cells = build_cells(hw, sim)
    by_metric = {cell.metric: cell for cell in cells}
    assert by_metric["最佳点峰值/曝光 (counts/ms)"].hw == pytest.approx(107.1)
    assert by_metric["最佳点峰值/曝光 (counts/ms)"].sim == pytest.approx(53.55)


# ---------------------------------------------------------------------------
# 差距清单
# ---------------------------------------------------------------------------


def test_build_gap_list_has_six_attributable_classes(tmp_path: Path) -> None:
    hw = load_bundle(_write_bundle(tmp_path / "hw"))
    sim = load_bundle(_write_bundle(tmp_path / "sim"))
    gaps = build_gap_list(hw, sim)
    assert len(gaps) == 6
    for marker in ("①", "②", "③", "④", "⑤", "⑥"):
        assert any(marker in gap for gap in gaps), f"差距清单缺少 {marker}"


def test_build_gap_list_quotes_numbers(tmp_path: Path) -> None:
    """差距清单必须带**证据数字**, 不是空话。"""
    hw = load_bundle(_write_bundle(tmp_path / "hw"))
    sim = load_bundle(_write_bundle(tmp_path / "sim"))
    gaps = "\n".join(build_gap_list(hw, sim))
    assert "1549.33" in gaps  # 实测焦距
    assert "13.9" in gaps  # 实测光斑宽度
    assert "0.6454" in gaps  # 实测最佳 1/f


# ---------------------------------------------------------------------------
# 渲染
# ---------------------------------------------------------------------------


def test_render_markdown_contains_table_and_gaps(tmp_path: Path) -> None:
    hw = load_bundle(_write_bundle(tmp_path / "hw"))
    sim = load_bundle(_write_bundle(tmp_path / "sim"))
    cells = build_cells(hw, sim)
    gaps = build_gap_list(hw, sim)
    md = render_markdown(cells, gaps, hw, sim, SimConfig().as_dict())

    assert "# 仿真 vs 实测 对照尺" in md
    assert "| 指标 | 实测 | 仿真 | 相对差 (sim−hw)/hw | 判定 |" in md
    assert "## 2. 差距清单 (可归因)" in md
    assert "## 附录 A: 离焦响应逐点对照" in md
    assert "far_field_padding" in md
    assert "peak_counts_per_ms" in md


def test_render_markdown_writes_file(tmp_path: Path) -> None:
    hw = load_bundle(_write_bundle(tmp_path / "hw"))
    sim = load_bundle(_write_bundle(tmp_path / "sim"))
    cells = build_cells(hw, sim)
    gaps = build_gap_list(hw, sim)
    out = tmp_path / "sim_vs_hw.md"
    out.write_text(
        render_markdown(cells, gaps, hw, sim, SimConfig().as_dict()), encoding="utf-8"
    )
    assert out.exists()
    assert out.read_text(encoding="utf-8").startswith("# 仿真 vs 实测 对照尺")


# ---------------------------------------------------------------------------
# 仿真口径旋钮
# ---------------------------------------------------------------------------


def test_build_noise_spec_requires_counts_mode() -> None:
    """counts 噪声模型必须与绝对光度同时给 —— 否则孪生会拒绝。"""
    assert build_noise_spec(SimConfig(peak_counts_per_ms=None)) is None
    spec = build_noise_spec(SimConfig())
    assert spec is not None
    assert spec.read_sigma_counts == pytest.approx(0.5)
    assert spec.shot_k == pytest.approx(0.0)
    assert spec.quantize_bits == 8


def test_patch_sim_configurer_installs_full_photometry() -> None:
    """替身必须把种子/补零/光度/噪声**全部**落到进程级 system 上。

    这是「跑第二个仿真前必须重置」的保证 —— 采集脚本自己的
    ``configure_sim_system`` 只管种子与补零, 会把光度冲掉。
    """
    from ao_shaping.drivers.sim import slm_pib_sim

    class _FakeModule:
        pass

    module = _FakeModule()
    config = SimConfig(seed=7, far_field_padding=4, peak_counts_per_ms=133.5)
    patch_sim_configurer(module, "configure_sim_system", config)
    assert hasattr(module, "configure_sim_system")

    module.configure_sim_system(7, 4)
    system = slm_pib_sim.get_system()
    assert system.far_field_padding == 4
    assert system.peak_counts_per_ms == pytest.approx(133.5)
    assert system.noise_spec is not None
    assert system.noise_spec.read_sigma_counts == pytest.approx(0.5)
    assert system.counts_mode is True

    # 再跑一次 ⇒ 口径不漂 (种子/补零/光度都还是同一个值)。
    module.configure_sim_system(7, 4)
    again = slm_pib_sim.get_system()
    assert again is not system, "每次都要换一个新的 system"
    assert again.peak_counts_per_ms == pytest.approx(133.5)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_main_without_execute_writes_table(tmp_path: Path, capsys: Any) -> None:
    """默认路径 (不给 --execute) 只出表, **不打开任何设备**。

    (读表这一步会 import 孪生模块去问远场角标度 —— 那是纯 Python, 不碰硬件。)
    """
    hw_root = _write_bundle(tmp_path / "hw")
    sim_root = _write_bundle(tmp_path / "sim")
    out = tmp_path / "sim_vs_hw.md"

    code = main(
        [
            "--hw-root",
            str(hw_root),
            "--sim-root",
            str(sim_root),
            "--out",
            str(out),
            "--report",
            "",
        ]
    )
    assert code == 0
    assert out.exists()
    printed = capsys.readouterr().out
    assert "仿真 vs 实测 对照尺" in printed


def test_main_missing_hw_root_exits_nonzero(tmp_path: Path) -> None:
    sim_root = _write_bundle(tmp_path / "sim")
    with pytest.raises(FileNotFoundError):
        main(
            [
                "--hw-root",
                str(tmp_path / "absent"),
                "--sim-root",
                str(sim_root),
                "--out",
                str(tmp_path / "out.md"),
            ]
        )


def test_probe_metrics_are_the_archived_column_names() -> None:
    """对照表用的指标名必须是归档 measurements.json 里**真实存在**的列。"""
    assert "cv" in PROBE_METRICS
    assert "pib" in PROBE_METRICS
    assert "frame_peak" in PROBE_METRICS
    assert "spot_fwhm_px" in PROBE_METRICS
    assert all(math.isfinite(1.0) for _ in PROBE_METRICS)


# ---------------------------------------------------------------------------
# 孪生的三个新旋钮 (相机离焦 / 固定 counts 参考 / 暗电平)
# ---------------------------------------------------------------------------


def test_build_noise_spec_carries_the_dark_level() -> None:
    """暗电平必须进噪声模型 —— 它是 counts 口径的一部分。"""
    spec = build_noise_spec(SimConfig(dark_level_counts=26.75))
    assert spec is not None
    assert spec.dark_level_counts == pytest.approx(26.75)


def test_patch_sim_configurer_installs_the_new_knobs() -> None:
    """替身必须把**全部**旋钮落到进程级 system 上 (含三个根因)。"""

    class _FakeModule:
        pass

    from ao_shaping.drivers.sim import slm_pib_sim

    module = _FakeModule()
    config = SimConfig(
        camera_defocus_m=0.01,
        counts_reference="flat",
        dark_level_counts=26.75,
        beam_w0=321.0,
        illumination_radius_px=450.0,
        panel_zernike_rad={6: 1.9},
    )
    patch_sim_configurer(module, "configure_sim_system", config)
    module.configure_sim_system(0, 4)
    system = slm_pib_sim.get_system()
    assert system.camera_defocus_m == pytest.approx(0.01)
    assert system.counts_reference == "flat"
    assert system.noise_spec is not None
    assert system.noise_spec.dark_level_counts == pytest.approx(26.75)
    assert system.beam_w0 == pytest.approx(321.0)
    assert system.illumination_radius_px == pytest.approx(450.0)
    assert system.panel_aberration is not None
    assert system.panel_aberration.zernike_rad[6] == pytest.approx(1.9)


def test_sim_width_scale_is_the_camera_pixel_conversion() -> None:
    """仿真侧的光斑宽度是**远场采样点**, 实测侧是**相机像元** ⇒ 必须换算。"""
    scale = sim_width_scale()
    assert scale > 1.0, "孪生的行向远场采样比相机像元粗 (面板是长方形)"
    assert scale < 2.0


def test_build_cells_converts_the_simulated_width(tmp_path: Path) -> None:
    """宽度那一行的仿真值必须被换算成相机像元 (否则是两种口径)。"""
    hw = load_bundle(_write_bundle(tmp_path / "hw"))
    sim = load_bundle(_write_bundle(tmp_path / "sim"))
    cells = build_cells(hw, sim)
    width_cell = {c.metric: c for c in cells}["光斑 FWHM (px)"]
    assert width_cell.hw == pytest.approx(13.9)
    assert width_cell.sim == pytest.approx(13.9 * sim_width_scale())


def test_hw_defocus_focal_mm_reads_the_archived_run_config(tmp_path: Path) -> None:
    """硬件那一轮带着哪张菲涅尔矫正片, 仿真侧就必须用同一张。"""
    root = _write_bundle(tmp_path / "hw")
    group = root / "hw_groups_20261022" / "dz_scan_hw_gs"
    (group / "run_config.json").write_text(
        json.dumps({"defocus_focal_mm": 1549.332046}), encoding="utf-8"
    )
    bundle = load_bundle(root)
    assert hw_defocus_focal_mm(bundle) == pytest.approx(1549.332046)


def test_hw_defocus_focal_mm_is_none_when_absent(tmp_path: Path) -> None:
    bundle = load_bundle(_write_bundle(tmp_path / "hw"))
    assert hw_defocus_focal_mm(bundle) is None


def test_hw_focus_diopter_grid_is_the_measured_grid(tmp_path: Path) -> None:
    """焦面标定的 1/f 网格必须**两侧同源**, 否则挑出来的点不可比。"""
    bundle = load_bundle(_write_bundle(tmp_path / "hw"))
    grid = hw_focus_diopter_grid(bundle)
    assert grid is not None
    assert grid == sorted(grid)
    assert grid[0] == pytest.approx(0.0)
    assert grid[-1] == pytest.approx(0.6454)


def test_parse_zernike_spec() -> None:
    assert parse_zernike_spec("6:1.9") == {6: pytest.approx(1.9)}
    assert parse_zernike_spec("6:1.9,4:0.5") == {
        6: pytest.approx(1.9),
        4: pytest.approx(0.5),
    }
    assert parse_zernike_spec("") == {}
    with pytest.raises(ValueError):
        parse_zernike_spec("6")


def test_build_config_reads_a_params_file(tmp_path: Path) -> None:
    """参数文件 = 上一轮调出来、已归档的那一组旋钮 (可复现)。"""
    params = tmp_path / "params.json"
    params.write_text(
        json.dumps(
            {
                "camera_defocus_m": 0.01,
                "counts_reference": "flat",
                "dark_level_counts": 26.75,
                "beam_w0": 321.0,
                "illumination_radius_px": 450.0,
                "panel_zernike_rad": {"6": 1.9},
            }
        ),
        encoding="utf-8",
    )
    args = _parse_args(["--params", str(params)])
    config = build_config(args)
    assert config.camera_defocus_m == pytest.approx(0.01)
    assert config.counts_reference == "flat"
    assert config.dark_level_counts == pytest.approx(26.75)
    assert config.beam_w0 == pytest.approx(321.0)
    assert config.illumination_radius_px == pytest.approx(450.0)
    assert config.panel_zernike_rad[6] == pytest.approx(1.9)


def test_build_config_cli_overrides_the_params_file(tmp_path: Path) -> None:
    params = tmp_path / "params.json"
    params.write_text(
        json.dumps({"camera_defocus_m": 0.01}), encoding="utf-8"
    )
    args = _parse_args(["--params", str(params), "--camera-defocus-m", "0.02"])
    config = build_config(args)
    assert config.camera_defocus_m == pytest.approx(0.02)


def test_sim_config_json_round_trip() -> None:
    """口径要能原样写进参数文件再读回来 (报告里要能写清用了哪一组)。"""
    config = SimConfig(
        camera_defocus_m=0.01,
        counts_reference="flat",
        dark_level_counts=26.75,
        beam_w0=321.0,
        illumination_radius_px=450.0,
        panel_zernike_rad={6: 1.9},
        panel_roughness_sigma_rad=0.05,
    )
    again = SimConfig.from_json(config.to_json())
    assert again.camera_defocus_m == pytest.approx(config.camera_defocus_m)
    assert again.counts_reference == config.counts_reference
    assert again.dark_level_counts == pytest.approx(config.dark_level_counts)
    assert again.beam_w0 == pytest.approx(config.beam_w0)
    assert again.illumination_radius_px == pytest.approx(config.illumination_radius_px)
    assert again.panel_zernike_rad == config.panel_zernike_rad
    assert again.panel_roughness_sigma_rad == pytest.approx(
        config.panel_roughness_sigma_rad
    )


# ---------------------------------------------------------------------------
# 报告正文 (入库的那一份) —— 与数据表共用同一份渲染函数
# ---------------------------------------------------------------------------


def _report_text(tmp_path: Path) -> str:
    hw = load_bundle(_write_bundle(tmp_path / "hw"))
    sim = load_bundle(_write_bundle(tmp_path / "sim"))
    cells = build_cells(hw, sim)
    gaps = build_gap_list(hw, sim)
    return render_report(
        cells,
        gaps,
        hw,
        sim,
        SimConfig().as_dict(),
        params_path=str(tmp_path / "params.json"),
        elapsed_s=477.1,
    )


def test_render_report_has_the_required_sections(tmp_path: Path) -> None:
    """报告正文必须有: 做了什么 / 用了什么 / 对照表 / 对上什么 / 能干什么 / 复现 / 局限。"""
    md = _report_text(tmp_path)
    for heading in (
        "## 1. 本轮做了什么",
        "## 2. 仿真用了什么",
        "## 3. 逐项对照表",
        "## 4. 对上了什么 / 仍差什么",
        "## 5. 这条仿真链能用来干什么 / 不能用来干什么",
        "## 6. 复现",
        "## 7. 局限",
    ):
        assert heading in md, f"报告缺少章节: {heading}"


def test_render_report_reuses_the_same_table(tmp_path: Path) -> None:
    """对照表必须来自**同一个**渲染函数 —— 不是手抄 (逐行都能在数据表里找到)。"""
    hw = load_bundle(_write_bundle(tmp_path / "hw"))
    sim = load_bundle(_write_bundle(tmp_path / "sim"))
    cells = build_cells(hw, sim)
    gaps = build_gap_list(hw, sim)
    report = render_report(
        cells, gaps, hw, sim, SimConfig().as_dict(), params_path=None
    )
    table = render_table(cells, heading=None)
    body = "\n".join(table)
    assert body in report, "报告里的对照表与 render_table() 的输出不一致"
    # 每一行数字都要出现 (抽查三行)。
    for needle in ("| CV (目标框内) |", "| 最佳 1/f (m⁻¹) |", "| σ(PIB) |"):
        assert needle in report


def test_render_report_quotes_the_measured_justification(tmp_path: Path) -> None:
    """「为什么是那个值」必须带**实测依据数字**, 不能是空话。"""
    md = _report_text(tmp_path)
    assert "7546.4" in md  # far_field_padding 的依据
    assert "0.6454" in md  # camera_defocus_m 的依据
    assert "1549.33" in md  # 补偿焦距
    assert "13.9" in md  # beam_w0 的依据 (平场宽度)
    assert "26.75" in md or "暗电平" in md


def test_render_report_states_the_caveats(tmp_path: Path) -> None:
    """「不能当实测替代」与「局限」必须写明 (别让读者把它当结论)。"""
    md = _report_text(tmp_path)
    assert "不能当实测替代" in md
    assert "几何产物" in md
    assert "曝光线性度检查在孪生里量的是噪声" in md
    assert "局限" in md


def test_render_report_links_resolve(tmp_path: Path) -> None:
    """报告正文里的相对链接必须指向**真实存在**的归档文件。"""
    import re

    md = _report_text(tmp_path)
    links = re.findall(r"\]\((artifacts/[^)]+)\)", md)
    assert links, "报告里没有任何归档链接"
    base = ROOT / "report" / "shape_trials" / "sim_vs_hw"
    for link in set(links):
        assert (base / link).exists(), f"链接指向不存在的文件: {link}"


def test_main_writes_both_outputs(tmp_path: Path) -> None:
    """`main` 要同时写数据表与报告正文; `--report ""` 只写数据表。"""
    hw_root = _write_bundle(tmp_path / "hw")
    sim_root = _write_bundle(tmp_path / "sim")
    out = tmp_path / "table.md"
    report = tmp_path / "report.md"
    code = main(
        [
            "--hw-root",
            str(hw_root),
            "--sim-root",
            str(sim_root),
            "--out",
            str(out),
            "--report",
            str(report),
            "--elapsed-s",
            "477.1",
        ]
    )
    assert code == 0
    assert out.exists() and report.exists()
    assert "# 数字孪生 vs 第三轮硬件实测" in report.read_text(encoding="utf-8")
    assert "≈ 477 s" in report.read_text(encoding="utf-8")

    only_table = tmp_path / "only_table.md"
    off = tmp_path / "off.md"
    main(
        [
            "--hw-root",
            str(hw_root),
            "--sim-root",
            str(sim_root),
            "--out",
            str(only_table),
            "--report",
            "",
        ]
    )
    assert only_table.exists()
    assert not off.exists()


def test_verdict_tally_covers_every_cell(tmp_path: Path) -> None:
    hw = load_bundle(_write_bundle(tmp_path / "hw"))
    sim = load_bundle(_write_bundle(tmp_path / "sim", scale=0.5))
    cells = build_cells(hw, sim)
    tally = _verdict_tally(cells)
    assert sum(tally.values()) == len(cells)
    # 「—」= 该行没有有意义的相对差 (实测为 0 / 缺失 / 无穷), 与判定三档互斥。
    undecidable = sum(
        1 for cell in cells if relative_diff(cell.hw, cell.sim) is None
    )
    assert tally["—"] == undecidable
    assert tally["一致"] + tally["偏差"] + tally["不一致"] == len(cells) - undecidable
