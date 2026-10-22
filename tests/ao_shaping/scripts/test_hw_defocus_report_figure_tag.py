"""Tests for the figure-stamp derivation of the hardware defocus report.

One series (``report/shape_trials/hw_defocus_zernike/``) can hold **several**
reports — the same day may add another round (``..._20261022_r2``).  Each
report owns its figures, so the stamp that is appended to every figure file
name must be **unique per report inside the series**.

The old rule ("stamp only when the file name *ends* with 8 digits") gave the
same-day round an **empty** stamp, so it silently overwrote the default
report's figures.  The stamp is now derived from the **whole** suffix after
the series prefix.

The **per-group** figure families (far-field grid, cross sections, per-order
Zernike shots, defocus response) had no stamp at all, so two rounds of the
series wrote the *same* file names and the first round's report ended up
showing the second round's pictures.  Every figure name must carry the stamp.

These tests call the functions directly; the generator itself is importable
without any device (it only ever reads artefacts).
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.slm import generate_hw_defocus_zernike_report as gen
from scripts.slm.generate_hw_defocus_zernike_report import (
    DEFAULT_REPORT_DATE,
    DEFAULT_REPORT_STEM,
    REPORT_KEY,
    figure_tag,
    report_date,
)

ROOT = Path(__file__).resolve().parents[3]
GENERATOR = ROOT / "scripts" / "slm" / "generate_hw_defocus_zernike_report.py"

#: 第一轮的采集目录 (根 ``.gitignore`` 忽略 ``data/*``, 所以这批原始数据**不入库**)。
#: 端到端的 ``--figures-only`` 校验依赖它 —— 缺了就跳过, 不凭记忆重建数字。
HW_GROUPS_ROOT = ROOT / "data" / "debug" / "hw_groups_20261020"
HW_ZERNIKE_ROOT = ROOT / "data" / "debug" / "hw_zernike_sweep_20261020b"
FOCUS_EVIDENCE = (
    ROOT / "data" / "debug" / "bench_settings" / "focus_probe" / "slm1_20261020_093932.json"
)

#: 报告里所有图族的前缀。**每一个**都必须带戳 —— 漏掉一个, 同系列的两轮就会互相覆盖。
FIGURE_NAME_PREFIXES = (
    "hw_defocus_score_vs_dz",
    "hw_defocus_steering_vs_dz",
    "hw_defocus_fwhm_vs_dz",
    "hw_defocus_response_",
    "hw_farfield_",
    "hw_xsec_",
    "hw_zernike_",
)

#: 同系列里现存的几份报告 (文件名 stem)。
STEMS = (
    "hw_defocus_zernike_20261020",  # 默认那份 (图名已入库, 不带戳)
    "hw_defocus_zernike_20261021",  # 第二轮
    "hw_defocus_zernike_20261022_r2",  # 同日追加的第三轮
)


def _key(stem: str) -> str:
    return f"report/shape_trials/hw_defocus_zernike/{stem}.md"


class TestFigureTag:
    """图名戳: 由系列前缀之后的**整段后缀**决定。"""

    def test_default_report_has_empty_stamp(self):
        # 默认那份的图已入库 (`figures/hw_defocus_score_vs_dz.png`), 改名会断链。
        assert figure_tag(_key(DEFAULT_REPORT_STEM)) == ""

    def test_second_round_keeps_its_date_stamp(self):
        assert figure_tag(_key("hw_defocus_zernike_20261021")) == "_20261021"

    def test_same_day_round_gets_its_own_unique_stamp(self):
        # ⚠️ 旧实现只在「以 8 位数字结尾」时给戳 -> 这一份拿到空戳 -> 覆盖默认那份的图。
        assert figure_tag(_key("hw_defocus_zernike_20261022_r2")) == "_20261022_r2"

    def test_stamps_are_pairwise_distinct(self):
        stamps = [figure_tag(_key(stem)) for stem in STEMS]
        assert len(set(stamps)) == len(stamps)

    def test_suffix_is_sanitised_to_alnum_underscore(self):
        assert figure_tag(_key("hw_defocus_zernike_20261022-r2")) == "_20261022_r2"

    def test_foreign_stem_raises(self):
        # 不属于本系列的名字必须**报错**, 而不是悄悄给一个空戳去覆盖别人的图。
        with pytest.raises(ValueError, match="hw_defocus_zernike_"):
            figure_tag(_key("other_series_20261021"))

    def test_missing_suffix_raises(self):
        with pytest.raises(ValueError, match="后缀"):
            figure_tag(_key("hw_defocus_zernike_"))


class TestReportDate:
    """报告日期: 取文件名里第一段 8 位数字, 取不到就回落默认那份的。"""

    def test_default_report_date(self):
        assert report_date(_key(DEFAULT_REPORT_STEM)) == "2026-10-20"

    def test_second_round_date(self):
        assert report_date(_key("hw_defocus_zernike_20261021")) == "2026-10-21"

    def test_same_day_round_keeps_that_day(self):
        # 轮次后缀不是日期 —— 同日追加的一轮仍是那一天 (2026-10-22)。
        assert report_date(_key("hw_defocus_zernike_20261022_r2")) == "2026-10-22"

    def test_no_date_in_stem_falls_back_to_default(self):
        assert report_date(_key("hw_defocus_zernike_round_x")) == DEFAULT_REPORT_DATE


def test_default_report_key_is_the_untagged_one():
    """默认的 REPORT_KEY 指向不带戳的那份 —— 改了它默认那份的图就得改名。"""
    assert Path(REPORT_KEY).stem == DEFAULT_REPORT_STEM


def _figure_name_fstrings() -> list[ast.JoinedStr]:
    """生成器源码里所有拼图名的 f-string。"""
    tree = ast.parse(GENERATOR.read_text(encoding="utf-8"))
    return [n for n in ast.walk(tree) if isinstance(n, ast.JoinedStr)]


def _literal_prefix(node: ast.JoinedStr) -> str:
    """f-string 的**字面**开头 (形如 ``hw_farfield_``) —— 用来认出它属于哪个图族。"""
    if node.values and isinstance(node.values[0], ast.Constant):
        value = node.values[0].value
        if isinstance(value, str):
            return value
    return ""


def _uses_fig_tag(node: ast.JoinedStr) -> bool:
    return any(
        isinstance(part, ast.FormattedValue)
        and isinstance(part.value, ast.Name)
        and part.value.id == "FIG_TAG"
        for part in node.values
    )


class TestEveryFigureNameCarriesTheStamp:
    """**每一个**图名都要带戳 —— 逐组图族也算。"""

    def test_no_untagged_figure_name_survives(self):
        offenders = [
            _literal_prefix(node)
            for node in _figure_name_fstrings()
            if _literal_prefix(node).startswith(FIGURE_NAME_PREFIXES)
            and not _uses_fig_tag(node)
        ]
        assert not offenders, (
            f"这些图名不带 FIG_TAG, 同系列的两轮会写进同一个文件: {offenders}"
        )

    def test_the_four_per_group_families_are_covered(self):
        """四个逐组图族都在检查范围内 (漏掉一个就等于漏掉一次覆盖)。"""
        prefixes = {_literal_prefix(node) for node in _figure_name_fstrings()}
        assert "hw_farfield_" in prefixes
        assert "hw_xsec_" in prefixes
        assert "hw_zernike_" in prefixes
        assert "hw_defocus_response_" in prefixes

    def test_copy_zernike_farfields_writes_tagged_names(self, tmp_path, monkeypatch):
        """泽尼克逐阶图 (16 张) 是从扫描目录**复制**过来的 —— 复制时也要带戳。"""
        sweep = tmp_path / "sweep"
        sweep.mkdir()
        for tag in ("before", "after"):
            (sweep / f"{tag}_nmax3.png").write_bytes(b"png")
        monkeypatch.setattr(gen, "FIG_DIR", tmp_path / "figs")
        monkeypatch.setattr(gen, "FIG_TAG", "_20261021")
        made = gen.copy_zernike_farfields(sweep, [{"n_max": 3}])
        assert made == [
            "figures/hw_zernike_before_nmax3_20261021.png",
            "figures/hw_zernike_after_nmax3_20261021.png",
        ]
        assert (tmp_path / "figs" / "hw_zernike_before_nmax3_20261021.png").is_file()
        assert not (tmp_path / "figs" / "hw_zernike_before_nmax3.png").exists()


class TestFiguresOnlyMode:
    """``--figures-only``: 只重画图, 报告正文一个字都不写。"""

    def test_flag_defaults_to_off(self):
        assert gen.build_parser().parse_args([]).figures_only is False

    def test_flag_is_accepted(self):
        assert gen.build_parser().parse_args(["--figures-only"]).figures_only is True

    def test_help_says_why_it_exists(self):
        help_text = gen.build_parser().format_help()
        assert "--figures-only" in help_text
        assert "不写报告正文" in help_text

    @pytest.mark.skipif(
        not (HW_GROUPS_ROOT / "dz_scan_hw_gs" / "measurements.json").is_file(),
        reason="第一轮的采集数据不在库里 (data/* 不入库) —— 端到端校验跳过",
    )
    def test_end_to_end_leaves_the_report_body_untouched(self, tmp_path):
        """跑一遍真数据: 图画出来了, 报告 .md **没有**被写。"""
        # 报告名仍要带系列前缀 —— 图名戳由它推导, 不属于本系列的名字会被拒绝。
        report_out = tmp_path / "hw_defocus_zernike_20261020.md"
        figs = tmp_path / "figs"
        proc = subprocess.run(
            [
                sys.executable,
                str(GENERATOR),
                "--groups-root",
                str(HW_GROUPS_ROOT),
                "--groups-root",
                str(HW_ZERNIKE_ROOT),
                "--focus-evidence",
                str(FOCUS_EVIDENCE),
                "--report-key",
                str(report_out),
                "--figures-dir",
                str(figs),
                "--figures-only",
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
            timeout=600,
        )
        assert proc.returncode == 0, proc.stderr
        assert not report_out.exists(), "--figures-only 不该写报告正文"
        assert (figs / "hw_defocus_score_vs_dz.png").is_file()
        assert (figs / "hw_farfield_dz_scan_hw_gs.png").is_file()
