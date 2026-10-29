"""GS 方扫报告的**输入契约归档**。

报告不只写结论，还要把"这次是从哪些文件算出来的"复制进 ``artifacts/``。
本模块负责这三件事：列出要复制的源、复制、以及 ``--check`` 时逐字节核对
副本没被改过。

⚠️ 这个目录名是**跟着参数变**的（换 ``--retest-root`` 就换目录），所以这里
读的是 :data:`_hw_gs_square_scans_model.PATHS` 里的字段，不是模块级常量。
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence


# 与 facade 同在 scripts/slm/ 下，parents[2] 算出的仓库根一致。
# 用**逐个 insert** 而不是 for 循环：scripts/ 那层的守卫
# (tests/ao_shaping/scripts/test_script_file_form_imports.py) 靠解析
# `sys.path.insert` 的字面参数判断「这个脚本能不能以文件方式启动」，
# for 循环里的循环变量它解析不了（会整文件跳过，等于没检查）。
# src 在 scripts 之前：scripts/ml 会遮蔽 src/ml。
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from scripts.slm._hw_gs_square_scans_figbase import (
    _sha256,
    write_frames_selection,
)
from scripts.slm._hw_gs_square_scans_fmt import (
    _rel,
)
from scripts.slm._hw_gs_square_scans_model import (
    PATHS,
    REPORT_KEY,
    RETEST_STAGES,
    RETEST_STAGE_CN,
    RetestRun,
    Run,
    STAGES,
    STAGE_CN,
)


def atomic_artifact_sources(atomic: RetestRun, target: Path) -> list[tuple[Path, Path]]:
    """§13 归档的 (源, 目标) 清单。"""
    pairs: list[tuple[Path, Path]] = [(atomic.root / "manifest.json", target / "manifest.json")]
    for name in STAGES:
        stage_dir = atomic.root / name
        for fname in ("measurements.csv", "measurements.json", "run_config.json"):
            src = stage_dir / fname
            if src.exists():
                pairs.append((src, target / name / fname))
        for cfg in atomic.stages[name].configs:
            pairs.append((stage_dir / cfg.config / "metrics.json",
                          target / name / cfg.config / "metrics.json"))
    selected = atomic.root / "sides" / "selected.json"
    if selected.exists():
        pairs.append((selected, target / "sides" / "selected.json"))
    return pairs


def _copy_farfield_to_artifacts(payload: dict[str, Any], figures: Sequence[Path],
                                target: Path | None = None) -> int:
    """把拼图复制进 ``artifacts/<batch>/`` 并写 ``frames_selection.json``。"""
    target = PATHS.artifact_dir if target is None else target
    target.mkdir(parents=True, exist_ok=True)
    copied = 0
    for path in figures:
        if not path.exists():
            continue
        shutil.copy2(path, target / path.name)
        copied += 1
    write_frames_selection(payload, target)
    return copied


def _check_farfield(run: Run, payload: dict[str, Any], figures: Sequence[Path],
                    artifact_dir: Path | None = None) -> list[str]:
    """``--check``: 拼图存在、与 artifacts 副本逐字节一致, 且 frames_selection.json 未过期。"""
    problems: list[str] = []
    if not payload.get("mosaics"):
        return problems
    artifact_dir = PATHS.artifact_dir if artifact_dir is None else artifact_dir
    drawn = {path.name for path in figures}
    for mosaic in payload["mosaics"]:
        name = mosaic["figure"]
        if name not in drawn:
            problems.append(f"拼图缺失: figures/{name}")
            continue
        source = PATHS.fig_dir / name
        copy = artifact_dir / name
        if not copy.exists():
            problems.append(f"拼图副本缺失: {_rel(copy)}")
        elif _sha256(source) != _sha256(copy):
            problems.append(f"拼图与 artifacts 副本不一致: {name}")
    expected = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path = artifact_dir / "frames_selection.json"
    if not path.exists():
        problems.append(f"缺失: {_rel(path)}")
    elif path.read_text(encoding="utf-8") != expected:
        problems.append("frames_selection.json 与现场规划不一致 (帧/裁剪/峰值已变)")
    return problems


def artifact_sources(run: Run) -> list[tuple[Path, Path]]:
    """(源文件, 目标文件) 清单 —— 复制与校验共用这一份, 不各写一遍。"""
    pairs: list[tuple[Path, Path]] = [
        (run.root / "manifest.json", PATHS.artifact_dir / "manifest.json")
    ]
    for name in STAGES:
        for fname in ("measurements.csv", "measurements.json", "run_config.json"):
            pairs.append((run.root / name / fname, PATHS.artifact_dir / name / fname))
    if run.log_path is not None:
        pairs.append((run.log_path, PATHS.artifact_dir / "run.log"))
    return pairs


def retest_artifact_sources(retest: RetestRun, target: Path) -> list[tuple[Path, Path]]:
    """复测批次的 (源, 目标) 清单。

    复测是原子化落盘, 所以每配置还多一份 ``metrics.json`` 与一个日志;
    段级 ``run_config.json`` **有就复制、没有就跳过** (v1 那批没有)。
    """
    pairs: list[tuple[Path, Path]] = [(retest.root / "manifest.json", target / "manifest.json")]
    for name in RETEST_STAGES:
        stage_dir = retest.root / name
        for fname in ("measurements.csv", "measurements.json", "run_config.json"):
            src = stage_dir / fname
            if src.exists():
                pairs.append((src, target / name / fname))
        for cfg in retest.stages[name].configs:
            pairs.append((stage_dir / cfg.config / "metrics.json",
                          target / name / cfg.config / "metrics.json"))
            if cfg.log_path is not None:
                pairs.append((cfg.log_path, target / "logs" / cfg.log_path.name))
    return pairs


def _copy_artifact_set(pairs: Sequence[tuple[Path, Path]], target: Path,
                       label: str, run_root: Path) -> list[Path]:
    """把一组 (源, 目标) 逐字节复制, 并写该目录的 ``PROVENANCE.json``。"""
    written: list[Path] = []
    for src, dst in pairs:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        written.append(dst)
    provenance = {
        "report": REPORT_KEY,
        "generator": "scripts/slm/generate_hw_gs_square_scans_report.py",
        "batch": label,
        "run_root": _rel(run_root),
        "files": {
            dst.relative_to(target).as_posix(): {
                "source": _rel(src),
                "sha256": _sha256(src),
                "bytes": src.stat().st_size,
            }
            for src, dst in pairs
        },
    }
    path = target / "PROVENANCE.json"
    path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    written.append(path)
    return written


def _check_artifact_set(pairs: Sequence[tuple[Path, Path]], target: Path) -> list[str]:
    """一组副本必须存在、与源逐字节一致, 且 ``PROVENANCE.json`` 未过期。"""
    problems: list[str] = []
    for src, dst in pairs:
        if not dst.exists():
            problems.append(f"artifacts 缺失: {_rel(dst)}")
            continue
        if dst.stat().st_size != src.stat().st_size or _sha256(dst) != _sha256(src):
            problems.append(f"artifacts 副本与源不一致: {_rel(dst)}")
    prov_path = target / "PROVENANCE.json"
    if not prov_path.exists():
        problems.append(f"artifacts 缺失: {_rel(prov_path)}")
        return problems
    recorded = json.loads(prov_path.read_text(encoding="utf-8")).get("files", {})
    expected = {
        dst.relative_to(target).as_posix(): _sha256(src) for src, dst in pairs
    }
    if set(recorded) != set(expected):
        problems.append(
            f"{_rel(prov_path)}: 记录的文件集合与本次输入不符 "
            f"(记录 {sorted(recorded)}, 期望 {sorted(expected)})"
        )
    for name, digest in expected.items():
        if name in recorded and recorded[name].get("sha256") != digest:
            problems.append(f"{_rel(prov_path)} 的 sha256 过期: {name}")
    return problems


def copy_artifacts(run: Run) -> list[Path]:
    """把基线的输入产物**逐字节**复制进 ``artifacts/``。"""
    return _copy_artifact_set(artifact_sources(run), PATHS.artifact_dir, "baseline", run.root)


def check_artifacts(run: Run) -> list[str]:
    """``--check``: 基线副本必须存在、与源逐字节一致, 且溯源 JSON 未过期。"""
    return _check_artifact_set(artifact_sources(run), PATHS.artifact_dir)


def _atomic_artifact_manifest(atomic: RetestRun) -> list[tuple[str, str]]:
    """§13 归档目录的文件说明 (与实际复制的文件一一对应)。"""
    rows: list[tuple[str, str]] = [("manifest.json", "三段汇总 (行数 / 自变量 / 模式 / 配置名)")]
    for name in STAGES:
        stage_dir = atomic.root / name
        rows.append((f"{name}/measurements.csv",
                     f"{STAGE_CN[name]} 全部指标 — §13 的**唯一来源**"))
        rows.append((f"{name}/measurements.json", "同上的 JSON 版 (用于交叉校验)"))
        if (stage_dir / "run_config.json").exists():
            rows.append((f"{name}/run_config.json",
                         "段级参数 (无 `plan`; 含 mode / preflight / side_max / 冻结 0 级)"))
    if (atomic.root / "sides" / "selected.json").exists():
        rows.append(("sides/selected.json", "第 1 段选定边长与它的分数"))
    for name in STAGES:
        for cfg in atomic.stages[name].configs:
            rows.append((f"{name}/{cfg.config}/metrics.json",
                         f"{cfg.config} 单配置指标 + `preflight`(含本进程冻结的 0 级)"))
    rows.append(("frames_selection.json", "§13 四张拼图的逐格来源 + sha256 + 裁剪/拉伸参数"))
    rows.append(("PROVENANCE.json", "源路径 + sha256 + 生成脚本"))
    return rows


def _artifact_manifest() -> list[tuple[str, str]]:
    """``artifacts/`` 里该有的文件与说明 (顺序即渲染顺序)。"""
    rows: list[tuple[str, str]] = [("manifest.json", "三段汇总 (行数 / 自变量 / 创建时间)")]
    for name in STAGES:
        rows.append((f"{name}/measurements.csv",
                     f"{STAGE_CN[name]} 全部指标 — 报告里测量值的**唯一来源**"))
        rows.append((f"{name}/measurements.json", "同上的 JSON 版 (含 NaN 语义, 用于交叉校验)"))
        rows.append((f"{name}/run_config.json", "本次运行参数 + 冻结的 0 级 + 曝光上下限"))
    rows.append(("run.log", "采集日志原文 — settle 判据超时与关键量的出处"))
    rows.append(("PROVENANCE.json", "源路径 + sha256 + 生成脚本"))
    return rows


def _retest_artifact_manifest(retest: RetestRun) -> list[tuple[str, str]]:
    """复测归档目录的文件说明 (顺序即渲染顺序, 与实际复制的文件一一对应)。"""
    rows: list[tuple[str, str]] = [
        ("manifest.json", "复测两段汇总 (行数 / 自变量 / 模式 / 配置名)")
    ]
    for name in RETEST_STAGES:
        stage_dir = retest.root / name
        rows.append((f"{name}/measurements.csv",
                     f"{RETEST_STAGE_CN[name]} 复测指标 — §11 复测值/重复性对照的来源"))
        rows.append((f"{name}/measurements.json", "同上的 JSON 版 (用于交叉校验)"))
        if (stage_dir / "run_config.json").exists():
            rows.append((f"{name}/run_config.json",
                         "段级运行参数 (无 `plan`; 含 mode / preflight / 冻结 0 级)"))
        for cfg in retest.stages[name].configs:
            rows.append((f"{name}/{cfg.config}/metrics.json",
                         f"{cfg.config} 单配置指标 + `preflight`(含本进程冻结的 0 级)"))
        for cfg in retest.stages[name].configs:
            if cfg.log_path is not None:
                rows.append((f"logs/{cfg.log_path.name}", f"{cfg.config} 采集日志原文"))
    rows.append(("PROVENANCE.json", "源路径 + sha256 + 生成脚本"))
    return rows
