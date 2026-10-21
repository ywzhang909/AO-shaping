"""Tests for the **cross-group merge audit** of the hardware defocus/Zernike report.

The Zernike sweep was split into several short runs (Stage A + one group per
order), so the report merges **many** ``summary.json`` files into one set of
numbers.  The merge used to be silent: a scalar field that differed between
two groups (``w0_px`` / ``target_side`` / ``center_xy`` / ``mask_center_xy``)
was simply taken from the first group, and the mismatch was recorded in
``merged["merge_warnings"]`` — which **nothing ever read**.  A folder left over
from an earlier (failed) attempt therefore silently changed the numbers of a
later round, and no reader could tell.

These tests pin the fix:

* the report body names the group directories that took part in the merge;
* a cross-group mismatch is **visible in the report body** (field name, both
  values, and the group each value came from);
* a consistent set of groups produces **no** warning block (no empty heading,
  no "none" placeholder);
* the audit block never leaks an absolute path into the report.

Everything runs on **synthetic** data in ``tmp_path`` — the real corpus under
``data/debug/`` is never touched, and no device is opened.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.slm import generate_hw_defocus_zernike_report as gen

ROOT = Path(__file__).resolve().parents[3]

#: 两组之间的 ``w0_px`` 差值 (照抄第二轮实测到的那个差: 5.742 vs 5.807)。
W0_A = 5.742078245970301
W0_B = 5.807177426723872

#: 报告正文里那段警告的标记 —— 「无警告时正文里没有警告段」就查它。
WARNING_MARK = "跨组字段不一致"


# ---------------------------------------------------------------------------
# 假数据 (synthetic fixtures)
# ---------------------------------------------------------------------------
def _blob(size: int = 256, sigma: float = 6.0, peak: float = 1000.0) -> np.ndarray:
    """一帧合成的高斯光斑 (探针离线重算指标要的就是这个)。"""
    yy, xx = np.indices((size, size))
    r2 = (xx - size / 2.0) ** 2 + (yy - size / 2.0) ** 2
    return (peak * np.exp(-r2 / (2.0 * sigma**2))).astype(np.float64)


def _write_probe_group(root: Path, name: str = "dz_scan_hw_gs") -> Path:
    """造一个**离焦扫描组** (含原始帧), 让生成器有一条 dz 扫描可写。

    帧是必需的: 没有帧时 ``load_probe_run`` 把所有行标成 ``frame_missing``,
    正文里就没有可用行, 生成器会直接报错。
    """
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    config = {
        "zero_order_roi_xy_frozen": [128, 128],
        "zero_order_full_frame_xy": [1433, 971],
        "target_side_px": 44,
        "roi_px": 256,
        "exposure_ms_working": 0.4,
        "frames_per_measurement": 3,
        "defocus_focal_mm": 1523.8,
        "gs_phase_source": "",
        "group_meta": {"role": "dz_scan", "phase": "hardware_gs"},
    }
    (d / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    rows = []
    # 真实的探针布局: 基线行 (叠加层为空) + 一行 Δz=0 的**全零**二次相位 (扫描行)
    # + 正负半轴各一点。Δz=0 必须出现**两次**: 少了扫描那一行, 正文的
    # 「离焦响应」表就没有 Δz=0 这个参照点。
    for idx, (dz, sign) in enumerate(
        [(0.0, 0.0), (0.0, 1.0), (1.5007, 1.0), (-1.5007, -1.0)]
    ):
        rows.append(
            {
                "index": idx,
                "dz_mm": dz,
                "chirp_sign": sign,
                "chirp_edge_waves": 3.8,
                "cv": 0.5,
                "pib": 0.2,
                "score": 0.35,
                "frame_peak": 100.0,
            }
        )
        np.savez(d / f"frame_{idx:03d}.npz", frame=_blob())
    (d / "measurements.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return d


def _order(n_max: int) -> dict:
    """一份 ``orders`` 记录 (生成器正文要读的字段都在这里)。"""
    return {
        "n_max": n_max,
        "n_modes": 10,
        "fit_rms_rad": 0.24,
        "max_abs_coefficient": 0.04,
        "clip_warning": False,
        "before": {"cv": 0.19, "pib": 0.107},
        "after": {"cv": 0.188, "pib": 0.106, "ee": 0.106, "quality": 0.517},
        "spgd_improved": True,
        "spgd_J_max": 0.02,
        "spgd_J_min": 0.01,
    }


def _write_summary(
    root: Path,
    group: str,
    *,
    w0: float = W0_A,
    n_max: int | None = None,
) -> Path:
    """造一份 ``summary.json`` (泽尼克扫描一组的产物)。"""
    d = root / group
    d.mkdir(parents=True, exist_ok=True)
    data = {
        "group": group,
        "w0_px": w0,
        "target_side": 2.0 * w0,
        "center_xy": [1433, 971],
        "mask_center_xy": [126, 128],
        "frame_shape": [256, 256],
        "config": {
            "exposure_time_ms": 0.2,
            "defocus_focal_mm": 1523.8,
            "group": group,
        },
        "baselines": {
            "flat": {"cv": 0.49, "pib": 0.19, "ee": 0.15, "ar": 0.89, "quality": 0.35},
            "gs_freeform": {
                "cv": 0.44,
                "pib": 0.27,
                "ee": 0.21,
                "ar": 1.0,
                "quality": 0.45,
            },
        },
        "orders": [_order(n_max)] if n_max is not None else [],
    }
    (d / "summary.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    return d


def _fake_root(tmp_path: Path, *, inconsistent: bool) -> Path:
    """一份假的数据根: 一条离焦扫描组 + 三份泽尼克 ``summary.json``。

    ``inconsistent=True`` 时, ``zernike_stage_a`` 那份的 ``w0_px`` 与别的组
    **故意不同** (照抄第二轮实测到的 5.742 vs 5.807)。
    """
    root = tmp_path / "fake_root"
    root.mkdir(parents=True, exist_ok=True)
    _write_probe_group(root)
    _write_summary(root, "zernike_order_nmax3", n_max=3)
    _write_summary(root, "zernike_order_nmax4", n_max=4)
    _write_summary(
        root,
        "zernike_stage_a",
        w0=W0_B if inconsistent else W0_A,
    )
    return root


def _fake_sim_g1() -> list[dict]:
    """假的仿真 G1 曲线 (正文的对拍节按 ±1.5007 mm 取值, 所以那两点必须在)。"""
    rows = []
    for dz in (-2.5, -1.5007, 0.0, 1.5007, 2.5):
        rows.append(
            {
                "dz_mm": dz,
                "dz_units": dz,
                "side_px": 44,
                "cv": 0.42,
                "pib": 0.2,
                "score": 0.5,
                "cv_fixed": 0.42,
                "pib_fixed": 0.2,
                "score_fixed": 0.5 - 0.01 * abs(dz),
                "sigma_px": 1.0,
            }
        )
    return rows


def _run_generator(
    tmp_path: Path,
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> str:
    """在临时目录里跑一次生成器, 返回**报告正文**。"""
    report = tmp_path / "hw_defocus_zernike_20261021_r9.md"
    figs = tmp_path / "figs"
    monkeypatch.setattr(gen, "load_sim_g1", _fake_sim_g1)
    monkeypatch.setattr(gen, "FIG_DIR", figs)
    monkeypatch.setattr(gen, "FIG_TAG", "")
    monkeypatch.setattr(gen, "REPORT_DATE", gen.DEFAULT_REPORT_DATE)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_hw_defocus_zernike_report.py",
            "--groups-root",
            str(root),
            "--zernike-dir",
            str(root),
            # 不指向任何真实证据文件 ⇒ §1.3 走「本轮没有附证据」那条路
            "--focus-evidence",
            str(tmp_path / "no_such_focus_evidence.json"),
            "--report-key",
            str(report),
            "--figures-dir",
            str(figs),
        ],
    )
    assert gen.main() == 0
    assert report.is_file()
    return report.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# load_zernike: 合并本身
# ---------------------------------------------------------------------------
class TestLoadZernikeMergeAudit:
    """合并时必须把「跨组不一致」记成**结构化**的记录 (而不是一条没人看的字符串)。"""

    def test_mismatch_is_recorded_with_both_values_and_groups(self, tmp_path):
        root = _fake_root(tmp_path, inconsistent=True)
        merged = gen.load_zernike(root)
        fields = {w["field"]: w for w in merged["merge_warnings"]}
        assert "w0_px" in fields
        warning = fields["w0_px"]
        assert {warning["kept"], warning["other"]} == {W0_A, W0_B}
        assert {warning["kept_group"], warning["other_group"]} == {
            "zernike_order_nmax3",
            "zernike_stage_a",
        }

    def test_consistent_groups_produce_no_warning(self, tmp_path):
        root = _fake_root(tmp_path, inconsistent=False)
        merged = gen.load_zernike(root)
        assert merged["merge_warnings"] == []

    def test_summary_groups_are_relative_directory_names(self, tmp_path):
        root = _fake_root(tmp_path, inconsistent=False)
        merged = gen.load_zernike(root)
        assert sorted(merged["summary_groups"]) == [
            "zernike_order_nmax3",
            "zernike_order_nmax4",
            "zernike_stage_a",
        ]
        # 组目录名而不是全路径: 全路径带机器目录名, 换台机器就对不上
        for name in merged["summary_groups"]:
            assert not Path(name).is_absolute()
            assert str(root) not in name

    def test_orders_are_stitched_per_n_max(self, tmp_path):
        root = _fake_root(tmp_path, inconsistent=False)
        merged = gen.load_zernike(root)
        assert [o["n_max"] for o in merged["orders"]] == [3, 4]


# ---------------------------------------------------------------------------
# 报告正文: 警告必须真的出现在正文里
# ---------------------------------------------------------------------------
class TestReportBodyShowsTheMismatch:
    """报告正文里能看到这条不一致 (含字段名与两个值)。"""

    def test_body_names_the_field_and_both_values(self, tmp_path, monkeypatch):
        root = _fake_root(tmp_path, inconsistent=True)
        body = _run_generator(tmp_path, root, monkeypatch)
        assert WARNING_MARK in body
        assert "`w0_px`" in body
        # 两个值都在 (显示格式: 6 位小数)
        assert "5.742078" in body
        assert "5.807177" in body

    def test_body_names_the_groups_the_values_came_from(self, tmp_path, monkeypatch):
        root = _fake_root(tmp_path, inconsistent=True)
        body = _run_generator(tmp_path, root, monkeypatch)
        assert "zernike_stage_a" in body
        assert "zernike_order_nmax3" in body

    def test_body_lists_the_merged_group_directories(self, tmp_path, monkeypatch):
        """参与合并的组目录名要列出来 —— 读者一眼能看出这批数据来自几组。"""
        root = _fake_root(tmp_path, inconsistent=False)
        body = _run_generator(tmp_path, root, monkeypatch)
        assert "zernike_order_nmax3" in body
        assert "zernike_order_nmax4" in body
        assert "zernike_stage_a" in body
        assert "3** 份 `summary.json`" in body

    def test_consistent_groups_leave_no_warning_block(self, tmp_path, monkeypatch):
        root = _fake_root(tmp_path, inconsistent=False)
        body = _run_generator(tmp_path, root, monkeypatch)
        assert WARNING_MARK not in body
        # 也不能留下空标题或「无」之类的占位
        assert "⚠️ **跨组字段不一致" not in body
        # 但组目录清单还在 (那是溯源, 不是警告)
        assert "数据来源与同源性核查" in body

    def test_audit_block_never_leaks_an_absolute_path(self, tmp_path):
        """审计段里只出现**组目录名**, 不出现绝对路径。"""
        root = _fake_root(tmp_path, inconsistent=True)
        merged = gen.load_zernike(root)
        block = gen.merge_audit_block(merged)
        assert str(root) not in block
        assert str(tmp_path) not in block
        assert "zernike_stage_a" in block
