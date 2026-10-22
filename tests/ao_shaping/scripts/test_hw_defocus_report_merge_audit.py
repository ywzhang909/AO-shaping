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

On top of that, the audit must **discriminate** two very different kinds of
difference (measured on the second hardware round):

* **(A) benign — same-round localisation jitter.**  Each per-order group
  re-locates the zero order **on its own frame**, so the centres scatter by
  **±1 px** while ``w0_px`` / ``target_side`` / ``center_xy`` stay identical.
  That must be listed, but **not** under the ``⚠️`` heading.
* **(B) malignant — data from another round mixed in.**  The optical state
  really changed: ``w0_px`` differs by **1.13%**.  That must stay under the
  ``⚠️`` heading.

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
    n_maxes: list[int] | None = None,
    center_xy: list[int] | None = None,
    mask_center_xy: list[int] | None = None,
    frame_shape: list[int] | None = None,
) -> Path:
    """造一份 ``summary.json`` (泽尼克扫描一组的产物)。

    默认值照抄第二轮实测到的那一组: ``w0_px`` 5.742078、``target_side`` 2·w0、
    ``center_xy`` [1433,971] (全幅传感器坐标)、``mask_center_xy`` [126,128]
    (256x256 窗口内坐标)、``frame_shape`` [256,256]。
    """
    d = root / group
    d.mkdir(parents=True, exist_ok=True)
    if n_maxes is not None:
        orders = [_order(n) for n in n_maxes]
    else:
        orders = [_order(n_max)] if n_max is not None else []
    data = {
        "group": group,
        "w0_px": w0,
        "target_side": 2.0 * w0,
        "center_xy": list(center_xy) if center_xy is not None else [1433, 971],
        "mask_center_xy": (
            list(mask_center_xy) if mask_center_xy is not None else [126, 128]
        ),
        "frame_shape": list(frame_shape) if frame_shape is not None else [256, 256],
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
        "orders": orders,
    }
    (d / "summary.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    return d


def _fake_root(
    tmp_path: Path,
    *,
    inconsistent: bool,
    mask_jitter: bool = False,
    center_shift: int = 0,
    dup_n_max: bool = False,
) -> Path:
    """一份假的数据根: 一条离焦扫描组 + 三份泽尼克 ``summary.json``。

    - ``inconsistent=True``: ``zernike_stage_a`` 那份的 ``w0_px`` (以及按 2·w0 算的
      ``target_side``) 与别的组**故意不同** —— 照抄第二轮实测到的 5.742 vs 5.807,
      相对差 **1.13%** ⇒ 光学状态真的变了 (恶性)。
    - ``mask_jitter=True``: ``zernike_order_nmax4`` 的 ``mask_center_xy`` 差 **1 px**
      —— 照抄第二轮逐阶组各自重新定位零级的实测散布 (良性)。
    - ``center_shift``: ``zernike_stage_a`` 的 ``center_xy`` 整体平移多少 px。
    - ``dup_n_max=True``: ``zernike_stage_a`` 里也放一份 n_max=3 ⇒ 同一个 n_max 出现两次。
    """
    root = tmp_path / "fake_root"
    root.mkdir(parents=True, exist_ok=True)
    _write_probe_group(root)
    _write_summary(root, "zernike_order_nmax3", n_max=3)
    _write_summary(
        root,
        "zernike_order_nmax4",
        n_max=4,
        mask_center_xy=[127, 128] if mask_jitter else None,
    )
    _write_summary(
        root,
        "zernike_stage_a",
        w0=W0_B if inconsistent else W0_A,
        center_xy=[1433 + center_shift, 971 + center_shift] if center_shift else None,
        n_maxes=[3] if dup_n_max else None,
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


# ---------------------------------------------------------------------------
# 第三轮 (2026-10-21 续跑坐标修复之后) 的实测形状
# ---------------------------------------------------------------------------
#: 第三轮的 7 个块 (Stage A 续跑) —— 生产者记的 ``frame_shape`` 是**全幅**
#: ``probe_shape``, 但 ``mask_center_xy`` 已经是**窗口内**坐标 (优化器冻结的 ROI 中心)。
ROUND3_BLOCK_GROUPS = (
    "zernike_stage_a",
    "zernike_stage_a_1",
    "zernike_stage_a_2",
    "zernike_stage_a_3",
    "zernike_stage_a_4",
    "zernike_stage_a_5",
    "zernike_stage_a_6",
)
ROUND3_BLOCK_FRAME_SHAPE = [1944, 2592]
ROUND3_BLOCK_MASK_CENTER = [127, 128]

#: 第三轮的 8 个逐阶组 —— ``frame_shape`` 是 ROI 窗口, ``mask_center_xy`` 落在
#: 实测到的四种取值上 (与第二轮同款散布)。
ROUND3_ORDER_FRAME_SHAPE = [256, 256]
ROUND3_ORDER_MASK_CENTERS = {
    "zernike_order_nmax3": [127, 127],
    "zernike_order_nmax4": [127, 128],
    "zernike_order_nmax6": [127, 127],
    "zernike_order_nmax8": [126, 127],
    "zernike_order_nmax11": [126, 128],
    "zernike_order_nmax15": [126, 128],
    "zernike_order_nmax20": [126, 128],
    "zernike_order_nmax25": [126, 128],
}

#: 第二轮实测到的形状: 块的 ``mask_center_xy`` 是**全幅**坐标 (超出 256 窗口)。
ROUND2_BLOCK_MASK_CENTER = [1433, 971]


def _round3_root(tmp_path: Path) -> Path:
    """第三轮形状的假数据根: 7 个块 + 8 个逐阶组 (共 15 份 summary)。

    两组的 ``frame_shape`` **不同** (块 = 全幅, 逐阶组 = ROI 窗口), 但两组的
    ``mask_center_xy`` **都落在共同窗口 (256x256) 内** ⇒ 它们**同空间** ⇒ 可比。
    """
    root = tmp_path / "round3_root"
    root.mkdir(parents=True, exist_ok=True)
    _write_probe_group(root)
    for group in ROUND3_BLOCK_GROUPS:
        _write_summary(
            root,
            group,
            mask_center_xy=list(ROUND3_BLOCK_MASK_CENTER),
            frame_shape=list(ROUND3_BLOCK_FRAME_SHAPE),
        )
    for group, mask in ROUND3_ORDER_MASK_CENTERS.items():
        _write_summary(
            root,
            group,
            n_max=int(group.rsplit("nmax", 1)[1]),
            mask_center_xy=list(mask),
            frame_shape=list(ROUND3_ORDER_FRAME_SHAPE),
        )
    return root


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


def _audit_section(body: str) -> str:
    """从报告正文里抠出「数据来源与同源性核查」那一段 (§3 开头)。

    报告正文里**别处**也有「⚠️」(与合并无关的提示), 所以「这一段有没有 ⚠️」只能
    在这一段里判。
    """
    start = body.index("**数据来源与同源性核查**")
    end = body.index("### 3.1", start)
    return body[start:end]


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


# ---------------------------------------------------------------------------
# 判别: 定位抖动 (良性) vs 真的混进别轮数据 (恶性)
# ---------------------------------------------------------------------------
class TestMergeAuditDiscriminatesJitterFromContamination:
    """跨组差异必须**分两类**报, 不能一律塞进「⚠️ 跨组字段不一致」。

    实测有两类完全不同的差异:

    - **(A) 同轮内的定位抖动 (良性)** —— 逐阶组各自在**自己那一帧**上重新定位零级,
      散布 **±1 px**; ``w0_px`` / ``target_side`` / ``center_xy`` 完全一致。
    - **(B) 真的混进别轮数据 (恶性)** —— 光学状态真的变了 (焦面/曝光/口径不同),
      实测 ``w0_px`` 相对差 **1.13%**。
    """

    def test_one_pixel_mask_center_jitter_is_benign(self, tmp_path):
        """只有 ±1 px 的 ``mask_center_xy`` 差异 ⇒ 良性, **不出现**「⚠️」标题。"""
        root = _fake_root(tmp_path, inconsistent=False, mask_jitter=True)
        merged = gen.load_zernike(root)
        block = gen.merge_audit_block(merged)
        assert "同轮内的定位抖动" in block
        assert "⚠️" not in block
        # 良性差异也要**列出来** (读者能核对): 两个取值 + 各自来源组
        assert "`[126, 128]`（`zernike_order_nmax3`）" in block
        assert "`[127, 128]`（`zernike_order_nmax4`）" in block

    def test_benign_jitter_says_it_does_not_change_the_calibre(self, tmp_path):
        """良性差异的措辞要说清「各组在自己那一帧上重新定位零级」+「不影响口径」。"""
        root = _fake_root(tmp_path, inconsistent=False, mask_jitter=True)
        block = gen.merge_audit_block(gen.load_zernike(root))
        assert "自己那一帧" in block
        assert "重新定位零级" in block
        assert "±1 px" in block
        assert "不影响口径" in block

    def test_w0_relative_diff_is_suspect(self, tmp_path):
        """``w0_px`` 差 1.13% ⇒ 恶性: 出现「⚠️」标题, 两个值与各自来源组都列出。"""
        root = _fake_root(tmp_path, inconsistent=True)
        block = gen.merge_audit_block(gen.load_zernike(root))
        assert "⚠️" in block
        assert "疑似跨轮次数据混入" in block
        assert "`w0_px`" in block
        assert "`5.742078`（`zernike_order_nmax3`）" in block
        assert "`5.807177`（`zernike_stage_a`）" in block
        # 相对差也要写出来 (读者能核对判定依据)
        assert "1.13%" in block

    def test_center_xy_shift_of_three_px_is_suspect(self, tmp_path):
        """像素坐标类字段逐轴偏差 > 2 px ⇒ 恶性。"""
        root = _fake_root(tmp_path, inconsistent=False, center_shift=3)
        block = gen.merge_audit_block(gen.load_zernike(root))
        assert "⚠️" in block
        assert "疑似跨轮次数据混入" in block
        assert "`center_xy`" in block
        assert "`[1433, 971]`（`zernike_order_nmax3`）" in block
        assert "`[1436, 974]`（`zernike_stage_a`）" in block

    def test_suspect_row_states_which_value_the_report_uses(self, tmp_path):
        """恶性一类里要能看出「合并采用了哪个值、来自哪个组」。"""
        root = _fake_root(tmp_path, inconsistent=True)
        block = gen.merge_audit_block(gen.load_zernike(root))
        assert "合并采用" in block
        assert "`5.742078`（`zernike_order_nmax3`）" in block
        # 采用的是**第一份含该字段的** summary 的值 ⇒ 来自 zernike_order_nmax3
        assert "zernike_stage_a" in block

    def test_duplicate_n_max_is_recorded_and_flagged(self, tmp_path):
        """同一个 n_max 出现在两个组 ⇒ 记下来并在正文里提示。"""
        root = _fake_root(tmp_path, inconsistent=False, dup_n_max=True)
        merged = gen.load_zernike(root)
        dups = [w for w in merged["merge_warnings"] if w.get("kind") == "orders"]
        assert len(dups) == 1
        assert dups[0]["n_max"] == 3
        assert dups[0]["kept_group"] == "zernike_stage_a"
        assert dups[0]["other_group"] == "zernike_order_nmax3"
        block = gen.merge_audit_block(merged)
        assert "同一个 n_max 出现了多次" in block
        assert "⚠️" in block

    def test_thresholds_and_their_rationale_are_written_out(self, tmp_path):
        """判定依据 (2 px / 0.5% 是什么、为什么) 必须写在正文里。"""
        root = _fake_root(tmp_path, inconsistent=True)
        block = gen.merge_audit_block(gen.load_zernike(root))
        assert "2 px" in block
        assert "0.5%" in block
        assert "±1 px" in block
        assert "1.13%" in block

    def test_benign_section_says_the_calibre_is_identical_when_it_is(self, tmp_path):
        """没有恶性差异时, 良性段可以说「口径完全一致」。"""
        root = _fake_root(tmp_path, inconsistent=False, mask_jitter=True)
        block = gen.merge_audit_block(gen.load_zernike(root))
        assert "完全一致" in block
        assert "不影响口径" in block

    def test_benign_section_never_claims_an_identical_calibre_when_it_is_not(
        self, tmp_path
    ):
        """口径字段有超阈值差异时, 良性段**不能**说「口径完全一致」。"""
        root = _fake_root(tmp_path, inconsistent=True, mask_jitter=True)
        block = gen.merge_audit_block(gen.load_zernike(root))
        benign = block[
            block.index("**同轮内的定位抖动（良性差异）**") : block.index(
                "⚠️ **跨组字段不一致"
            )
        ]
        assert "完全一致" not in benign
        assert "不涉及" in benign
        # 良性段里**不许**出现「⚠️」标记 (否则读者会把良性差异当成报警)
        assert "⚠️" not in benign

    def test_different_coordinate_frames_are_neither_jitter_nor_contamination(
        self, tmp_path
    ):
        """窗口不同 ⇒ ``mask_center_xy`` 不可直接比较: 既不算抖动, 也不算混入。

        (Stage A 在全幅 1944x2592 上跑, 逐阶组在 256x256 窗口上跑 —— 两者的
        ``mask_center_xy`` 不是同一个坐标系。)
        """
        root = tmp_path / "fake_root"
        root.mkdir(parents=True, exist_ok=True)
        _write_probe_group(root)
        _write_summary(root, "zernike_order_nmax3", n_max=3)
        _write_summary(
            root,
            "zernike_stage_a",
            mask_center_xy=[1433, 971],
            frame_shape=[1944, 2592],
        )
        block = gen.merge_audit_block(gen.load_zernike(root))
        assert "坐标参考系不同" in block
        assert "不参与判定" in block
        # 不可比 ⇒ 不能报成「疑似混入」
        assert "疑似跨轮次数据混入" not in block
        assert "⚠️" not in block

    def test_report_body_separates_the_two_kinds(self, tmp_path, monkeypatch):
        """报告正文里两类差异的**措辞不同**: 良性不带「⚠️」, 恶性带。"""
        root = _fake_root(tmp_path, inconsistent=False, mask_jitter=True)
        body = _run_generator(tmp_path, root, monkeypatch)
        audit = _audit_section(body)
        assert "同轮内的定位抖动" in audit
        assert "⚠️" not in audit
        # 良性差异照样列出来 (读者能核对)
        assert "`[126, 128]`" in audit
        assert "`[127, 128]`" in audit

    def test_report_body_flags_contamination(self, tmp_path, monkeypatch):
        root = _fake_root(tmp_path, inconsistent=True)
        body = _run_generator(tmp_path, root, monkeypatch)
        audit = _audit_section(body)
        assert "疑似跨轮次数据混入" in audit
        assert "5.807177" in audit
        assert "⚠️" in audit


# ---------------------------------------------------------------------------
# 可比性判据: 看**取值**, 不看「frame_shape 是否相等」
# ---------------------------------------------------------------------------
class TestComparabilityUsesValuesNotFrameShapeEquality:
    """两个生产者写的 ``frame_shape`` **不是一个意思** ⇒ 不能拿它判可比性。

    实测 (第三轮真实数据):

    - 块 (Stage A 续跑) 记 ``frame_shape = [1944, 2592]`` (全幅 ``probe_shape``),
      但它的 ``mask_center_xy`` 已经是**窗口内**的 ``[127, 128]`` (2026-10-21 的
      续跑坐标修复, 提交 07139db6, 之后);
    - 逐阶组记 ``frame_shape = [256, 256]`` (ROI 窗口), ``mask_center_xy`` 落在
      ``[126,128]`` / ``[127,127]`` / ``[127,128]`` / ``[126,127]`` 四种上。

    ⇒ 两者**其实同空间** (都是窗口内坐标), 必须判**可比** ⇒ 交叉核对恢复生效。
    而第二轮的块 (``mask_center_xy = [1433, 971]``, 全幅) **确实**超出 256 窗口
    ⇒ 仍判**不可比** (已入库的第二轮报告结论不变)。
    """

    def test_round3_shape_is_comparable_despite_different_frame_shapes(self, tmp_path):
        """第三轮形状: 帧形状不同, 但两个掩膜中心**都落在共同窗口内** ⇒ 可比。"""
        root = _round3_root(tmp_path)
        block = gen.merge_audit_block(gen.load_zernike(root))
        # 「坐标参考系不同」这一段**不该出现**
        assert "坐标参考系不同" not in block
        assert "坐标系不同" not in block
        # 反而进「良性抖动」段
        assert "**同轮内的定位抖动（良性差异）**" in block
        assert "⚠️" not in block
        # 块与逐阶组的 1 px 差**列出来** (交叉核对恢复生效)
        assert "`[126, 128]`" in block
        assert "`[127, 128]`" in block
        assert "`[127, 127]`" in block
        assert "`[126, 127]`" in block
        # 块的那一组也在表里
        assert "zernike_stage_a" in block

    def test_round3_shape_report_body_has_no_incomparable_section(
        self, tmp_path, monkeypatch
    ):
        """第三轮形状跑完整份报告: 正文里**不出现**「坐标参考系不同」。"""
        root = _round3_root(tmp_path)
        body = _run_generator(tmp_path, root, monkeypatch)
        audit = _audit_section(body)
        assert "坐标参考系不同" not in audit
        assert "坐标系不同" not in audit
        assert "**同轮内的定位抖动（良性差异）**" in audit
        assert "⚠️" not in audit

    def test_round2_shape_is_still_incomparable(self, tmp_path):
        """第二轮形状 (块的掩膜中心是全幅坐标) ⇒ 仍判**不可比**。"""
        root = tmp_path / "round2_root"
        root.mkdir(parents=True, exist_ok=True)
        _write_probe_group(root)
        _write_summary(
            root,
            "zernike_order_nmax3",
            n_max=3,
            mask_center_xy=[126, 128],
            frame_shape=[256, 256],
        )
        _write_summary(
            root,
            "zernike_stage_a",
            mask_center_xy=list(ROUND2_BLOCK_MASK_CENTER),
            frame_shape=[1944, 2592],
        )
        block = gen.merge_audit_block(gen.load_zernike(root))
        assert "**坐标参考系不同（不参与判定）**" in block
        assert "zernike_stage_a" in block
        # 不可比 ⇒ 不算抖动, 也不算混入
        assert "**同轮内的定位抖动（良性差异）**" not in block
        assert "疑似跨轮次数据混入" not in block
        assert "⚠️" not in block

    def test_out_of_window_value_is_reported_as_full_frame(self, tmp_path):
        """不可比时要说清是「落在窗口外 ⇒ 全幅坐标」, 而不是「窗口不同」。"""
        root = tmp_path / "round2_root"
        root.mkdir(parents=True, exist_ok=True)
        _write_probe_group(root)
        _write_summary(root, "zernike_order_nmax3", n_max=3, frame_shape=[256, 256])
        _write_summary(
            root,
            "zernike_stage_a",
            mask_center_xy=list(ROUND2_BLOCK_MASK_CENTER),
            frame_shape=[1944, 2592],
        )
        block = gen.merge_audit_block(gen.load_zernike(root))
        assert "全幅传感器" in block
        assert "共同窗口" in block
        # **不能**用「窗口不同所以不可比」这种在第三轮为假的理由
        assert "窗口不同所以" not in block
        assert "窗口（`frame_shape`）不同的两组之间" not in block

    def test_same_frame_shape_jitter_stays_benign(self, tmp_path):
        """回归: 逐阶组之间 (``frame_shape`` 相同, 差 1 px) ⇒ 良性抖动。"""
        root = tmp_path / "same_shape_root"
        root.mkdir(parents=True, exist_ok=True)
        _write_probe_group(root)
        _write_summary(
            root,
            "zernike_order_nmax3",
            n_max=3,
            mask_center_xy=[126, 128],
            frame_shape=[256, 256],
        )
        _write_summary(
            root,
            "zernike_order_nmax4",
            n_max=4,
            mask_center_xy=[127, 128],
            frame_shape=[256, 256],
        )
        block = gen.merge_audit_block(gen.load_zernike(root))
        assert "坐标参考系不同" not in block
        assert "**同轮内的定位抖动（良性差异）**" in block
        assert "±1 px" in block
        assert "⚠️" not in block

    def test_center_xy_is_always_comparable_across_frame_shapes(self, tmp_path):
        """``center_xy`` 是全幅传感器坐标 ⇒ **不套**「小窗口」规则, 永远可比。

        两组的 ``frame_shape`` 不同, ``center_xy`` 差 1 px ⇒ 走「良性抖动」,
        **不能**被判成「坐标参考系不同」。
        """
        root = tmp_path / "center_root"
        root.mkdir(parents=True, exist_ok=True)
        _write_probe_group(root)
        _write_summary(
            root,
            "zernike_order_nmax3",
            n_max=3,
            mask_center_xy=[127, 128],
            frame_shape=[256, 256],
        )
        _write_summary(
            root,
            "zernike_stage_a",
            mask_center_xy=[127, 128],
            center_xy=[1434, 972],
            frame_shape=[1944, 2592],
        )
        block = gen.merge_audit_block(gen.load_zernike(root))
        assert "坐标参考系不同" not in block
        assert "**同轮内的定位抖动（良性差异）**" in block
        assert "`center_xy`" in block
        assert "`[1433, 971]`" in block
        assert "`[1434, 972]`" in block

    def test_window_edge_is_inclusive(self, tmp_path):
        """边界自洽: 上界**含端点** (坐标 ≤ 窗口边长 算「落在窗口内」)。"""
        shapes = {"a": [256, 256], "b": [1944, 2592]}
        # 恰好落在上界上 ⇒ 窗口内 ⇒ 可比
        comparable, _note = gen._window_local_comparability(
            "a", "b", shapes, [256, 256], [256, 256]
        )
        assert comparable is True
        # 超出上界 ⇒ 全幅 ⇒ 不可比
        comparable, note = gen._window_local_comparability(
            "a", "b", shapes, [126, 128], [257, 128]
        )
        assert comparable is False
        assert "全幅传感器" in note

    def test_missing_frame_shape_is_treated_as_comparable(self, tmp_path):
        """缺 ``frame_shape`` ⇒ 按可比处理 (缺元数据不该把差异藏起来)。"""
        comparable, _note = gen._window_local_comparability(
            "a", "b", {"a": None}, [126, 128], [1433, 971]
        )
        assert comparable is True

    def test_non_coordinate_value_is_treated_as_comparable(self, tmp_path):
        """取值不是坐标对 ⇒ 无法证明它是全幅坐标 ⇒ 按可比处理。"""
        comparable, _note = gen._window_local_comparability(
            "a", "b", {"a": [256, 256], "b": [1944, 2592]}, None, [1433, 971]
        )
        assert comparable is True


class TestAuditBlockIsRoundAgnostic:
    """§3 的同源性核查段必须**只讲本轮的数据**。

    早先的版本把「第三轮的块是 ``[127, 128]``」**写死**在正文里, 而第三轮真跑出来
    的是 ``[128, 129]`` ⇒ 正文会印一句与**同一节的数据表**对不上的话 (读者一眼能看出来)。

    ⇒ 判据两条:
      1. 段里「本轮」的取值必须等于**本轮数据**的取值;
      2. 段里不得出现**某一轮专属**的说法 (轮次名) —— 段是轮次无关的, 轮次由文件/日期决定。
    """

    @staticmethod
    def _block(tmp_path: Path, centers: dict[str, list[int]], shapes: dict[str, list[int]] | None = None) -> str:
        """造一份假数据根, 返回它的 §3 同源性核查段。"""
        root = tmp_path / "fake_root"
        root.mkdir(parents=True, exist_ok=True)
        _write_probe_group(root)
        for i, (group, center) in enumerate(centers.items()):
            _write_summary(
                root,
                group,
                n_max=3 + i,
                mask_center_xy=center,
                frame_shape=(shapes or {}).get(group),
            )
        return gen.merge_audit_block(gen.load_zernike(root))

    def test_this_rounds_values_are_printed(self, tmp_path):
        """「本轮各组的取值」那一句必须列的是**本轮数据**的取值。"""
        block = self._block(
            tmp_path,
            {
                "zernike_order_nmax3": [128, 129],
                "zernike_order_nmax4": [126, 127],
            },
        )
        # 取值按数值排序后列出 (与正文里的排序一致)
        assert "取值为 `[126, 127]` / `[128, 129]`" in block

    def test_no_round_specific_wording(self, tmp_path):
        """段里不得出现轮次名 (「第三轮」) —— 写死轮次名就会在某轮变成一句不成立的话。"""
        block = self._block(tmp_path, {"zernike_order_nmax3": [128, 129]})
        assert "第三轮" not in block
        assert "本轮" in block

    def test_large_deviation_is_reported_as_suspect(self, tmp_path):
        """大偏差 (第三轮实测有一组差 123 px) 必须被**报出来**, 不能被当成抖动吞掉。"""
        block = self._block(
            tmp_path,
            {
                "zernike_order_nmax3": [128, 129],
                "zernike_order_nmax25": [6, 6],
            },
        )
        assert "[6, 6]" in block
        assert "疑似跨轮次数据混入" in block
        assert "逐轴偏差 123 px > 2 px" in block
        assert "本轮实测散布 **±123 px**" in block

    def test_value_outside_common_window_is_named(self, tmp_path):
        """落在共同窗口之外的取值要**点名说是谁** (可能是别的组, 也可能是合并采用值)。"""
        block = self._block(
            tmp_path,
            {
                "zernike_order_nmax3": [128, 129],
                "zernike_stage_a": [1433, 971],
            },
            shapes={"zernike_order_nmax3": [256, 256], "zernike_stage_a": [1944, 2592]},
        )
        assert "不在同一空间" in block
        assert "落在共同窗口 256x256 之外" in block
        assert "全幅传感器" in block

    def test_spread_is_computed_from_this_rounds_data(self, tmp_path):
        """散布必须由本轮数据算出来 (两组的逐轴偏差 2 px), 不是某个写死的数。"""
        block = self._block(
            tmp_path,
            {
                "zernike_order_nmax3": [128, 129],
                "zernike_order_nmax4": [126, 127],
            },
        )
        assert "本轮实测散布 **±2 px**" in block


