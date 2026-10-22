#!/usr/bin/env python
"""把某一轮硬件采集的原始产物按轮归档到报告目录 (输入契约)。

为什么需要这个脚本
------------------

``report/shape_trials/hw_defocus_zernike/artifacts/`` 是这一系列的**输入契约**:
报告正文里的每个数字都要能在归档里找到对应的原始记录。前两轮 (``round1_20261020`` /
``round2_20261021``) 的归档是**手工**做的 —— 一次性、不可复现, 而且第二轮那次正好赶上
「第三轮首次尝试把第二轮 11 个组的原始记录覆盖了」, 于是归档下来的是**现存的那一份**,
而不是第二轮原始数据 (这件事已写进 ``artifacts/README.md``)。

所以从这一轮起归档改由脚本完成: 动作可复现、可审计, 并且会把
「数据根里缺哪些组」「哪些大文件没归档」明确写出来 —— 缺组意味着那一轮有组没跑成,
静默跳过等于把这个事实藏起来。

只归档原始记录, 不做任何数值处理
--------------------------------

按文件类型**白名单**拷贝 (逐字节 ``copy2``, 保留时间戳):

* 离焦 / 噪声 / 远场组: ``measurements.json`` / ``measurements.csv`` / ``run_config.json``
* 泽尼克组: ``summary.json`` / ``stage_a.json`` / ``spgd_nmax*.csv`` /
  ``before_nmax*.png`` / ``after_nmax*.png``
* 焦面标定证据: ``slm*.json``
* 分组根的索引: ``manifest.json`` / ``groups_manifest.json``
* 采集计划 (可选, ``--no-plan`` 可关): ``plan.json``

**不归档**体积大的东西: 逐帧 ``frame_*.npz``、相位 ``*.npy``、``stage_a.npz``
(每个 17–56 MB)。它们仍留在 ``data/debug/``, 本脚本只在输出里说明这一点。

不覆盖已有归档
--------------

归档是**快照**: 被悄悄改写就失去意义了。所以目标轮次目录已存在且非空时**报错退出**,
只有显式 ``--force`` 才覆盖 —— 且覆盖是「删掉旧快照再重建」, 这样快照里不会残留
上一版的文件。

默认不写盘
----------

默认只**演练**: 打印将要归档 / 跳过的内容, 不碰文件系统。真的写盘要显式加 ``--write``。
于是「对着真实报告目录跑一次看看会做什么」是安全的动作。

用法
----

::

    # 演练 (默认): 打印将归档什么、跳过什么、缺哪些组
    python scripts/slm/archive_hw_defocus_round.py

    # 真的写入 (落到 report/.../artifacts/<轮次名>/)
    python scripts/slm/archive_hw_defocus_round.py --write

    # 换一轮 / 换数据根
    python scripts/slm/archive_hw_defocus_round.py --write \\
        --round round4_20261022 \\
        --groups-root data/debug/hw_groups_20261022 \\
        --sweep-root data/debug/hw_zernike_sweep_20261022

    # 归档到别处 (自检 / 传阅)
    python scripts/slm/archive_hw_defocus_round.py --write --out C:/temp/round3
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

#: 仓库根 (``scripts/slm/<file>.py`` -> 上溯两级)。**从 ``__file__`` 推导**,
#: 绝不能写死某台机器的绝对路径 (换机器就静默写到别处去了)。
ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# 默认值 (对准第三轮, 但逻辑里不写死「第三轮」—— 换一轮只要改参数)
# ---------------------------------------------------------------------------
DEFAULT_ROUND = "round3_20261022"
DEFAULT_GROUPS_ROOT = "data/debug/hw_groups_20261022"
DEFAULT_SWEEP_ROOT = "data/debug/hw_zernike_sweep_20261022"
DEFAULT_OUT = "report/shape_trials/hw_defocus_zernike/artifacts"

#: 焦面标定证据目录的**默认位置**: 它是分组根的一个子目录 (由焦面标定组的
#: ``--debug-dir`` 写死), 所以跟着分组根走。
FOCUS_DIR_NAME = "focus_calibration"

#: 离焦 / 噪声 / 远场组的三件套。
PROBE_FILES: tuple[str, ...] = ("measurements.json", "measurements.csv", "run_config.json")

#: 泽尼克组的原始记录。``stage_a.json`` 记录 Stage A 那一块是否跑完 ——
#: 前两轮的归档里也有它, 所以这一轮继续带上。
ZERNIKE_FILES: tuple[str, ...] = ("summary.json", "stage_a.json")

#: 泽尼克组里带阶数的文件 (按模式匹配, 因为文件名里带 ``nmax<N>``)。
ZERNIKE_GLOBS: tuple[str, ...] = ("spgd_nmax*.csv", "before_nmax*.png", "after_nmax*.png")

#: 分组根的**索引**文件: 它们记录这一轮跑了哪些组、各组状态如何。
#: 前两轮的归档里都有, 缺了就没法从归档本身判断「这轮本来该有哪些组」。
#: ``groups_manifest*.json`` 带上后缀变体 —— 失败轮的清单副本也是那一轮的证据。
GROUP_INDEX_FILES: tuple[str, ...] = ("manifest.json", "groups_manifest.json")
GROUP_INDEX_GLOBS: tuple[str, ...] = ("groups_manifest*.json",)

#: 焦面标定证据的文件名模式 (焦面标定探针写 ``slm<编号>_<时间戳>.json``)。
FOCUS_GLOBS: tuple[str, ...] = ("slm*.json",)

#: 体积大、不入库的东西 —— 只在输出里说明它们仍留在 ``data/debug/``。
BIG_FILE_GLOBS: tuple[str, ...] = ("frame_*.npz", "*.npy", "stage_a.npz")

#: 「这个目录里算有记录」的判据 (用于判定一组到底跑没跑成)。
GROUP_RECORD_MARKERS: tuple[str, ...] = (
    "measurements.json",
    "measurements.csv",
    "run_config.json",
    "summary.json",
    "stage_a.json",
)


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def repo_rel(path: Path) -> str:
    """把路径写成**仓库相对**、**正斜杠**的形式。

    为什么: 输出里的路径要能直接贴进文档 / 提交信息 / 脚本参数, 机器相关的
    绝对路径既不可读也不可移植。仓库外的路径 (自检用的临时目录) 退化成绝对路径,
    但仍然是正斜杠。
    """
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _resolve(path_text: str) -> Path:
    """命令行给的路径: 相对路径按**仓库根**解析 (脚本可以从任何 cwd 运行)。"""
    path = Path(path_text)
    if path.is_absolute():
        return path
    return ROOT / path


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _ext_of(path: Path) -> str:
    """扩展名 (小写, 无扩展名时给一个占位)。"""
    suffix = path.suffix.lower()
    return suffix if suffix else "(无扩展名)"


def _is_big(path: Path) -> bool:
    """这个文件是不是「体积大、不入库」的那一类。"""
    return any(path.match(pattern) for pattern in BIG_FILE_GLOBS)


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------
@dataclass
class GroupRecord:
    """数据根里的一个组目录, 以及它该归档什么。"""

    name: str
    source: Path
    keep: list[Path] = field(default_factory=list)
    """要归档的文件 (白名单命中)。"""
    skip: list[Path] = field(default_factory=list)
    """同一目录里**不**归档的文件 (逐帧 / 相位 / 基线截图 …)。"""


@dataclass
class ArchiveEntry:
    """一条待拷贝记录。"""

    src: Path
    dest: Path
    category: str
    """归档类别 (分组记录 / 泽尼克记录 / 焦面标定 / 分组索引 / 采集计划)。"""


@dataclass
class ArchivePlan:
    """一次归档的完整计划 (先算清楚, 再落盘)。"""

    round_name: str
    round_dir: Path
    entries: list[ArchiveEntry] = field(default_factory=list)
    skipped: list[tuple[Path, str]] = field(default_factory=list)
    """(被跳过的文件, 原因)。"""
    groups: list[tuple[str, str, int]] = field(default_factory=list)
    """(来源标签, 组名, 归档文件数) —— 归档了哪些组, 一目了然。"""
    missing_index_files: list[str] = field(default_factory=list)
    """数据根里**没有**的索引文件 (名字列出来, 不当成「跳过的文件」计数)。"""
    missing_groups: list[str] = field(default_factory=list)
    """计划里有、数据根里没有的组。"""
    empty_groups: list[str] = field(default_factory=list)
    """目录在、但一条记录都没有的组 (多半是崩在中途)。"""
    unplanned_groups: list[str] = field(default_factory=list)
    """数据根里有、计划里没有的组。"""
    focus_dir: Path | None = None
    plan_source: Path | None = None
    plan_archived: bool = False


# ---------------------------------------------------------------------------
# 采集计划
# ---------------------------------------------------------------------------
def load_plan(path: Path) -> list[dict[str, Any]]:
    """读采集计划, 返回它的组列表。

    计划是**脚本生成**的 (``scripts/slm/build_hw_sweep_plan.py``), 形如
    ``{"root": ..., "groups": [{"group": ..., "args": [...]}, ...]}``。
    """
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} 的顶层必须是 JSON 对象")
    groups = data.get("groups")
    if not isinstance(groups, list):
        raise ValueError(f"{path} 缺 'groups' 列表")
    return [g for g in groups if isinstance(g, dict)]


def expected_group_dirs(
    plan_groups: list[dict[str, Any]],
    groups_root: Path,
    sweep_root: Path,
) -> dict[str, list[Path]]:
    """计划里的每个组**应该**在哪个目录 (可能不止一个候选)。

    判定顺序与 ``scripts/slm/run_hw_sweep_groups.py::resolve_base_dir`` 一致:
    显式给了 ``--out-root`` / ``--out`` / ``--debug-dir`` 就用它, 否则在两个数据根里
    按组名找。**不猜**: 找不到就落在分组根, 并且会被记成缺组。
    """
    out: dict[str, list[Path]] = {}
    for raw in plan_groups:
        name = str(raw.get("group", "")).strip()
        if not name:
            continue
        args = [str(a) for a in (raw.get("args") or [])]
        candidates: list[Path] = []
        for flag in ("--out-root", "--out"):
            if flag in args:
                idx = args.index(flag)
                if idx + 1 < len(args):
                    # 组目录建在 ``<base>/<组名>/`` 下 (与驱动脚本一致)。
                    candidates.append(_resolve(args[idx + 1]) / name)
                    break
        if not candidates and "--debug-dir" in args:
            idx = args.index("--debug-dir")
            if idx + 1 < len(args):
                base = _resolve(args[idx + 1])
                # 焦面标定探针把证据**平铺**写在 debug-dir 里 (不加组名一层),
                # 但两种布局都认, 所以两个候选都留着。
                candidates.extend([base, base / name])
        if not candidates:
            candidates.extend([groups_root / name, sweep_root / name])
        out[name] = candidates
    return out


def _has_records(directory: Path) -> bool:
    """这个目录里有没有「算一条记录」的文件。"""
    if not directory.is_dir():
        return False
    for marker in GROUP_RECORD_MARKERS:
        if (directory / marker).is_file():
            return True
    for pattern in ZERNIKE_GLOBS:
        if next(directory.glob(pattern), None) is not None:
            return True
    for pattern in FOCUS_GLOBS:
        if next(directory.glob(pattern), None) is not None:
            return True
    return False


def _is_empty_dir(directory: Path) -> bool:
    """目录存在但里面一条记录都没有 (组跑崩在中途的痕迹)。"""
    if not directory.is_dir():
        return False
    return not _has_records(directory)


# ---------------------------------------------------------------------------
# 收集
# ---------------------------------------------------------------------------
def _pick_group_files(group_dir: Path) -> tuple[list[Path], list[Path]]:
    """组目录里哪些文件归档、哪些不归档 (白名单 vs 其余)。"""
    names = {p.name for p in group_dir.iterdir() if p.is_file()}
    keep: list[Path] = []
    # 判组靠**目录内容**而不是名字 (与报告生成器 ``discover_groups`` 同一判据):
    # 含 ``measurements.json`` 的是离焦探针组, 含 ``summary.json`` 的是泽尼克组。
    if "measurements.json" in names:
        keep.extend(group_dir / name for name in PROBE_FILES if name in names)
    if "summary.json" in names:
        keep.extend(group_dir / name for name in ZERNIKE_FILES if name in names)
        for pattern in ZERNIKE_GLOBS:
            keep.extend(sorted(p for p in group_dir.glob(pattern) if p.is_file()))
    keep_set = {p.resolve() for p in keep}
    skip = [
        p for p in sorted(group_dir.iterdir())
        if p.is_file() and p.resolve() not in keep_set
    ]
    return sorted(keep, key=lambda p: p.name), skip


def collect_groups(root: Path) -> list[GroupRecord]:
    """收一个数据根里的所有组。"""
    records: list[GroupRecord] = []
    if not root.is_dir():
        return records
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        keep, skip = _pick_group_files(child)
        if not keep:
            continue
        records.append(GroupRecord(name=child.name, source=child, keep=keep, skip=skip))
    return records


def collect_focus_files(focus_dir: Path) -> tuple[list[Path], list[Path]]:
    """焦面标定目录里的证据 JSON。"""
    if not focus_dir.is_dir():
        return [], []
    keep: list[Path] = []
    for pattern in FOCUS_GLOBS:
        keep.extend(sorted(p for p in focus_dir.glob(pattern) if p.is_file()))
    keep_set = {p.resolve() for p in keep}
    skip = [
        p for p in sorted(focus_dir.iterdir())
        if p.is_file() and p.resolve() not in keep_set
    ]
    return sorted(keep, key=lambda p: p.name), skip


def collect_sweep_root_files(sweep_root: Path) -> tuple[list[Path], list[Path]]:
    """扫描根**平铺**在根下的泽尼克记录 (第一轮的老布局)。

    第一轮的扫描根是平铺的 (``summary.json`` 与 png/csv 直接躺在根下), 第二轮起
    改成「一组一个子目录」。两种都认, 否则第一轮那样的根会被归档成空目录。
    """
    if not sweep_root.is_dir():
        return [], []
    names = {p.name for p in sweep_root.iterdir() if p.is_file()}
    keep: list[Path] = []
    for name in ZERNIKE_FILES:
        if name in names:
            keep.append(sweep_root / name)
    for pattern in ZERNIKE_GLOBS:
        keep.extend(sorted(p for p in sweep_root.glob(pattern) if p.is_file()))
    keep_set = {p.resolve() for p in keep}
    skip = [
        p for p in sorted(sweep_root.iterdir())
        if p.is_file() and p.resolve() not in keep_set
    ]
    return sorted(keep, key=lambda p: p.name), skip


# ---------------------------------------------------------------------------
# 计划一次归档
# ---------------------------------------------------------------------------
def build_plan(
    *,
    round_name: str,
    groups_root: Path,
    sweep_root: Path,
    focus_dir: Path | None,
    out_root: Path,
    plan_path: Path | None,
    archive_plan: bool,
) -> ArchivePlan:
    """算出这一次归档**要做什么** (不碰文件系统)。

    先算清楚再落盘: 归档是快照, 中途失败留半个目录比什么都不做更糟。
    """
    plan = ArchivePlan(
        round_name=round_name,
        round_dir=out_root / round_name,
        focus_dir=focus_dir,
    )

    # 归档里的子目录名 = 来源根自己的目录名 —— 归档与数据根一一对应,
    # 读的人不需要猜某个组属于哪一轮。
    groups_sub = groups_root.name or "groups"
    sweep_sub = sweep_root.name or "sweep"

    # --- 分组根的组 + 索引 ------------------------------------------------
    for record in collect_groups(groups_root):
        for src in record.keep:
            plan.entries.append(
                ArchiveEntry(
                    src=src,
                    dest=plan.round_dir / groups_sub / record.name / src.name,
                    category="分组记录",
                )
            )
        plan.groups.append(("分组", record.name, len(record.keep)))
        for src in record.skip:
            plan.skipped.append((src, "组目录内未列入白名单的文件"))

    plan_resolved = plan_path.resolve() if plan_path is not None else None
    index_names: set[str] = set(GROUP_INDEX_FILES)
    for pattern in GROUP_INDEX_GLOBS:
        index_names.update(p.name for p in groups_root.glob(pattern) if p.is_file())
    for name in sorted(index_names):
        src = groups_root / name
        if src.is_file():
            plan.entries.append(
                ArchiveEntry(
                    src=src,
                    dest=plan.round_dir / groups_sub / name,
                    category="分组索引",
                )
            )
        else:
            # 文件**不存在**不是「跳过」: 跳过是说「有这个文件但没归档」。
            # 两者混在一起会让跳过计数失真。
            plan.missing_index_files.append(name)

    # 分组根**根层**的其他文件 (例如失败轮的 manifest 副本) 一律不归档, 但必须
    # 点名 —— 静默不抄等于读者以为归档里就是全部。
    for src in sorted(groups_root.glob("*")):
        if not src.is_file():
            continue
        if src.name in index_names:
            continue
        if plan_resolved is not None and src.resolve() == plan_resolved:
            continue
        plan.skipped.append((src, "分组根内未列入白名单的文件"))

    # --- 泽尼克扫描根的组 + 平铺记录 -------------------------------------
    for record in collect_groups(sweep_root):
        for src in record.keep:
            plan.entries.append(
                ArchiveEntry(
                    src=src,
                    dest=plan.round_dir / sweep_sub / record.name / src.name,
                    category="泽尼克记录",
                )
            )
        plan.groups.append(("泽尼克", record.name, len(record.keep)))
        for src in record.skip:
            plan.skipped.append((src, "组目录内未列入白名单的文件"))

    sweep_keep, sweep_skip = collect_sweep_root_files(sweep_root)
    for src in sweep_keep:
        plan.entries.append(
            ArchiveEntry(
                src=src,
                dest=plan.round_dir / sweep_sub / src.name,
                category="泽尼克记录",
            )
        )
    for src in sweep_skip:
        plan.skipped.append((src, "扫描根内未列入白名单的文件"))

    # --- 焦面标定证据 -----------------------------------------------------
    if focus_dir is not None:
        keep, skip = collect_focus_files(focus_dir)
        for src in keep:
            plan.entries.append(
                ArchiveEntry(
                    src=src,
                    dest=plan.round_dir / FOCUS_DIR_NAME / src.name,
                    category="焦面标定",
                )
            )
        # 焦面标定也算这一轮的一「组」: 它在计划里是一个组, 只是产物是平铺在
        # 目录里的 JSON 而不是「一个组一个子目录」。只跑了焦面标定的那一轮
        # 同样是有内容的轮次, 不该被空轮守卫拦下。
        if keep:
            plan.groups.append(("焦面标定", FOCUS_DIR_NAME, len(keep)))
        for src in skip:
            plan.skipped.append((src, "焦面标定目录内未列入白名单的文件"))

    # --- 采集计划 ---------------------------------------------------------
    if plan_path is not None and plan_path.is_file():
        plan.plan_source = plan_path
        if archive_plan:
            # 计划是**这一轮的输入契约**: 它列出了本该有哪些组, 归档里没有它,
            # 读者就无法判断「缺组」是计划如此还是采集丢的。放在分组根那个
            # 子目录里 —— 计划本来就在分组根里 (前两轮的归档也是这个位置)。
            try:
                plan_path.resolve().relative_to(groups_root.resolve())
                dest = plan.round_dir / groups_sub / plan_path.name
            except ValueError:
                dest = plan.round_dir / plan_path.name
            plan.entries.append(
                ArchiveEntry(src=plan_path, dest=dest, category="采集计划")
            )
            plan.plan_archived = True
    elif plan_path is not None:
        plan.skipped.append((plan_path, "指定的采集计划文件不存在"))

    # --- 缺组 / 空组 / 计划外的组 ----------------------------------------
    if plan_path is not None and plan_path.is_file():
        expected = expected_group_dirs(
            load_plan(plan_path), groups_root=groups_root, sweep_root=sweep_root
        )
        found = {record.name for record in collect_groups(groups_root)}
        found |= {record.name for record in collect_groups(sweep_root)}
        for name, candidates in expected.items():
            if name in found:
                continue
            if any(_has_records(candidate) for candidate in candidates):
                continue
            if any(candidate.is_dir() for candidate in candidates):
                plan.empty_groups.append(name)
            else:
                plan.missing_groups.append(name)
        plan.missing_groups.sort()
        plan.empty_groups.sort()
        plan.unplanned_groups = sorted(name for name in found if name not in expected)

    return plan


# ---------------------------------------------------------------------------
# 执行
# ---------------------------------------------------------------------------
def _guard_round_name(round_name: str) -> None:
    """轮次名必须是**单层**目录名。

    为什么要这么严: ``--force`` 会删掉目标目录再重建, 如果轮次名里带
    ``..`` 或路径分隔符, 一次手滑就能删掉仓库里的别的东西。
    """
    if not round_name or round_name in (".", ".."):
        raise ValueError(f"轮次名不合法: {round_name!r}")
    if "/" in round_name or "\\" in round_name:
        raise ValueError(f"轮次名只能是单层目录名 (不含路径分隔符): {round_name!r}")


def execute(plan: ArchivePlan, *, force: bool) -> int:
    """把计划落盘, 返回拷贝的文件数。"""
    _guard_round_name(plan.round_name)
    round_dir = plan.round_dir

    if round_dir.exists():
        if not force:
            raise FileExistsError(
                f"归档目录已存在且非空: {repo_rel(round_dir)} —— 归档是快照, "
                "被悄悄改写就失去意义了。确认要覆盖再加 --force。"
            )
        # 覆盖 = 删掉旧快照再重建: 否则上一版多出来的文件会留在归档里,
        # 读的人分不清哪一份是这一轮的。
        shutil.rmtree(round_dir)

    copied = 0
    for entry in plan.entries:
        entry.dest.parent.mkdir(parents=True, exist_ok=True)
        # copy2 保留时间戳: 归档是那一轮采集的**快照**, 时间戳本身就是证据。
        shutil.copy2(entry.src, entry.dest)
        copied += 1
    return copied


def write_manifest(
    plan: ArchivePlan,
    *,
    groups_root: Path,
    sweep_root: Path,
    out_root: Path,
    copied: int,
) -> Path:
    """写归档自带的清单 (``archive_manifest.json``)。

    为什么要有: 归档目录里只有一堆 JSON/CSV/PNG, 读的人看不出「这轮本该有哪些组、
    哪些没跑成、哪些大文件没进来」。清单把这三件事写下来, 归档因此**自解释**。
    """
    payload = {
        "round": plan.round_name,
        "archived_utc": _utc_now_iso(),
        "generator": repo_rel(Path(__file__)),
        "sources": {
            "groups_root": repo_rel(groups_root),
            "sweep_root": repo_rel(sweep_root),
            "focus_dir": repo_rel(plan.focus_dir) if plan.focus_dir else None,
            "plan": repo_rel(plan.plan_source) if plan.plan_source else None,
        },
        "destination": repo_rel(out_root / plan.round_name),
        "archived_files": [
            {
                "path": repo_rel(entry.dest),
                "source": repo_rel(entry.src),
                "category": entry.category,
                "bytes": entry.src.stat().st_size,
                # 源文件的**修改时间**: 归档是那一轮的快照, 而「这批文件到底是
                # 哪一次采集写的」正是第二轮踩过的坑 (第三轮首次尝试把第二轮
                # 11 个组的记录覆盖了)。带着时间戳, 读者至少能看出异常。
                "source_mtime": datetime.fromtimestamp(
                    entry.src.stat().st_mtime, tz=timezone.utc
                ).isoformat(timespec="seconds"),
            }
            for entry in plan.entries
        ],
        "skipped_files": [
            {"path": repo_rel(path), "reason": reason} for path, reason in plan.skipped
        ],
        "missing_groups": plan.missing_groups,
        "empty_groups": plan.empty_groups,
        "unplanned_groups": plan.unplanned_groups,
        "missing_index_files": plan.missing_index_files,
        "plan_archived": plan.plan_archived,
        "archived_file_count": copied,
    }
    path = plan.round_dir / "archive_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------
def _count_by_ext(paths: list[Path]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in paths:
        key = _ext_of(path)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _fmt_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "无"
    return " / ".join(f"{ext} {count}" for ext, count in counts.items())


def report(plan: ArchivePlan, *, groups_root: Path, sweep_root: Path, write: bool) -> None:
    """打印这一次归档的**完整交代** (演练与实写同一套输出)。"""
    mode = "实写" if write else "演练 (未写盘)"
    print(f"[归档] 轮次 {plan.round_name} — {mode}")
    print(f"  分组根: {repo_rel(groups_root)}")
    print(f"  扫描根: {repo_rel(sweep_root)}")
    if plan.focus_dir is not None:
        print(f"  焦面标定: {repo_rel(plan.focus_dir)}")
    print(f"  目标目录: {repo_rel(plan.round_dir)}")

    archived = [entry.src for entry in plan.entries]
    print(
        f"  归档 {len(archived)} 个文件 ({_fmt_counts(_count_by_ext(archived))}) "
        f"→ {repo_rel(plan.round_dir)}"
    )
    for label, group, n_files in plan.groups:
        print(f"    [{label}] {group}: {n_files} 个文件")
    skipped = [path for path, _ in plan.skipped]
    print(f"  跳过 {len(skipped)} 个文件 ({_fmt_counts(_count_by_ext(skipped))})")
    for path, reason in plan.skipped:
        if reason.endswith("未列入白名单的文件"):
            continue
        print(f"    - {repo_rel(path)}: {reason}")

    # 大文件单独点名: 它们留在 data/debug/, 报告里的图无法只靠归档重建 ——
    # 读者必须知道这一点 (与 artifacts/README.md 的「不在库里的东西」一节呼应)。
    big = [path for path in skipped if _is_big(path)]
    if big:
        total_mb = sum(path.stat().st_size for path in big) / (1024 * 1024)
        print(
            f"  其中逐帧 frame_*.npz / 相位 *.npy / stage_a.npz 共 {len(big)} 个 "
            f"({total_mb:.1f} MB) 未入库, 仍留在 {repo_rel(groups_root.parent)} 下"
        )

    if plan.missing_groups:
        print(f"  ⚠ 缺组 (计划里有、数据根里没有): {', '.join(plan.missing_groups)}")
    else:
        print("  缺组: 无")
    if plan.empty_groups:
        print(f"  ⚠ 空组 (目录在、但没有记录): {', '.join(plan.empty_groups)}")
    if plan.unplanned_groups:
        print(f"  ⚠ 计划外的组 (数据根里有、计划里没有): {', '.join(plan.unplanned_groups)}")
    if plan.missing_index_files:
        print(
            f"  (数据根里没有这些索引文件: {', '.join(plan.missing_index_files)})"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--round", default=DEFAULT_ROUND, help="轮次名 (归档目录名)")
    parser.add_argument("--groups-root", default=DEFAULT_GROUPS_ROOT,
                        help="分组根 (离焦 / 噪声 / 远场组)")
    parser.add_argument("--sweep-root", default=DEFAULT_SWEEP_ROOT,
                        help="泽尼克扫描根 (Stage A + 逐阶组)")
    parser.add_argument("--focus-dir", default=None,
                        help="焦面标定证据目录 (默认 <groups-root>/focus_calibration)")
    parser.add_argument("--plan", default=None,
                        help="采集计划 JSON (默认自动找 <groups-root>/plan.json)")
    parser.add_argument("--no-plan", action="store_true",
                        help="不把采集计划一起归档")
    parser.add_argument("--out", default=DEFAULT_OUT,
                        help="归档根目录 (轮次目录建在它下面)")
    parser.add_argument("--write", action="store_true",
                        help="真的写盘 (默认只演练, 不碰文件系统)")
    parser.add_argument("--force", action="store_true",
                        help="目标轮次目录已存在且非空时覆盖 (先删旧快照再重建)")
    return parser


def main(argv: list[str] | None = None) -> int:
    """跑一次归档。

    Returns:
        进程退出码: 0 = 成功 (或只是演练); 2 = 用法 / 归档条件不满足。
    """
    args = _build_parser().parse_args(argv)

    groups_root = _resolve(args.groups_root)
    sweep_root = _resolve(args.sweep_root)
    out_root = _resolve(args.out)

    if args.focus_dir:
        focus_dir: Path | None = _resolve(args.focus_dir)
    else:
        focus_dir = groups_root / FOCUS_DIR_NAME

    if args.plan:
        plan_path: Path | None = _resolve(args.plan)
    else:
        candidate = groups_root / "plan.json"
        plan_path = candidate if candidate.is_file() else None

    if not groups_root.is_dir():
        print(
            f"分组根不存在: {repo_rel(groups_root)} —— 这一轮还没采集? "
            "(采集产物落在 data/debug/<根名>/ 下)",
            file=sys.stderr,
        )
        return 2
    if not sweep_root.is_dir():
        print(
            f"扫描根不存在: {repo_rel(sweep_root)} —— 这一轮还没采集? "
            "(采集产物落在 data/debug/<根名>/ 下)",
            file=sys.stderr,
        )
        return 2

    try:
        plan = build_plan(
            round_name=args.round,
            groups_root=groups_root,
            sweep_root=sweep_root,
            focus_dir=focus_dir,
            out_root=out_root,
            plan_path=plan_path,
            archive_plan=not args.no_plan,
        )
    except (OSError, ValueError) as exc:
        print(f"归档计划生成失败: {exc}", file=sys.stderr)
        return 2

    if plan_path is None:
        print(
            "  ⚠ 没有找到采集计划, 无法判定「缺组」 (用 --plan 指定计划文件)",
            file=sys.stderr,
        )

    # **空轮不许归档**: 这一轮一个组都没产出时, 归档出来的只是一个空壳快照 ——
    # 它看起来像「这轮跑过了, 但什么也没留下」, 而真相是「这轮根本没跑」。
    # 两者的区别对读报告的人很重要, 所以宁可报错也不落盘。
    # ⚠ 判据是**组记录数**, 不是文件数: 计划 JSON 与索引文件不算组记录 ——
    # 拿它们凑数会让「一个组都没跑」这一轮看起来像归档成功了。
    if not plan.groups:
        print(
            "数据根里没有任何组记录 —— 这一轮还没采集, 不归档 "
            "(否则会归档出一个空壳快照, 读的人分不清「没跑」和「跑了但没留下」)",
            file=sys.stderr,
        )
        return 2

    report(plan, groups_root=groups_root, sweep_root=sweep_root, write=args.write)

    if not args.write:
        print("  (演练结束: 加 --write 才落盘)")
        return 0

    try:
        copied = execute(plan, force=args.force)
    except (OSError, ValueError) as exc:
        print(f"归档失败: {exc}", file=sys.stderr)
        return 2

    manifest = write_manifest(
        plan,
        groups_root=groups_root,
        sweep_root=sweep_root,
        out_root=out_root,
        copied=copied,
    )
    print(f"  归档清单: {repo_rel(manifest)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
