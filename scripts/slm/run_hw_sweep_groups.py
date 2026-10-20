#!/usr/bin/env python
"""一次只跑一组变量的硬件探测: 按计划 (plan) 分组驱动。

为什么
----

两个硬件探针脚本各扫一组变量:

* ``scripts/slm/hw_gs_defocus_probe.py`` —— 自变量 **Δz** (离焦间距, mm);
* ``scripts/slm/run_gs_refine_zernike_sweep.py`` —— 自变量 **n_max** (泽尼克阶数)。

把它们放进**同一次运行**有两个坏处: 一组失败整轮作废; 产物说不清是哪一组贡献的。
所以每次调用只探一组 (``--group`` / ``--group-meta``), 而**编排**交给本脚本:
它读一份 JSON 计划, 把每一组当作**独立子进程**按顺序跑完。

🔴 **为什么是独立子进程**: 一组卡死 (Santec SDK 的原生等待不受 Python 侧超时保护)
或崩溃时, 不能把后面几组一起带走; 而且每一组都要有自己的设备生命周期 (open →
测量 → close), 同一进程里连着跑两组会共用残留的驱动状态。

计划文件 (JSON)::

    {
      "root": "data/debug/hw_groups_20261020",
      "groups": [
        {"group": "dz_noise", "variable": "dz_mm", "meta": {"exposure_ms": "0.2"},
         "script": "scripts/slm/hw_gs_defocus_probe.py",
         "args": ["--execute", "--defocus-focal-mm", "1597.4",
                  "--exposure-ms", "0.2", "--dz-mm", "0"]},
        {"group": "dz_scan", "variable": "dz_mm", "script": "scripts/slm/hw_gs_defocus_probe.py",
         "args": ["--execute", "--dz-mm", "0", "5", "10"]}
      ]
    }

* ``root`` —— 这一轮的根目录; 每一组的产物目录是 ``<root>/<组名>/``, 驱动自己的
  索引写在 ``<root>/groups_manifest.json``。
* ``script`` —— 要跑的命令。**以 ``.py`` 结尾时用当前解释器 (``sys.executable``)
  启动**, 计划里写的是仓库相对路径, 用哪个解释器该由跑计划的环境决定; 其余情况
  (``script`` 本身就是可执行文件) 原样使用。
* ``args`` —— 追加给脚本的参数。

用法::

    # 只看计划, 什么都不执行
    python scripts/slm/run_hw_sweep_groups.py --plan plan.json --dry-run

    # 真跑
    python scripts/slm/run_hw_sweep_groups.py --plan plan.json --timeout-s 3600

    # 把已经跑完的目录登记进来 (不重跑)
    python scripts/slm/run_hw_sweep_groups.py --plan plan.json \\
        --adopt-existing dz_scan report/shape_trials/hw_gs_defocus_probe_20261020_015426/dz_scan
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts._common.group_manifest import (  # noqa: E402
    group_output_dir,
    load_manifest,
    utc_now_iso,
)

#: 驱动自己的索引文件名 (与两个探针脚本的 ``manifest.json`` 分开, 免得互相覆盖)。
GROUPS_MANIFEST_NAME = "groups_manifest.json"

#: 一组「已经跑完」的判据: 目录里至少有下面之一 (两个探针各自的落盘标志)。
ARTEFACT_MARKERS = ("measurements.json", "summary.json")

#: 组命令的输出留多少尾巴进索引 (前部被截掉 —— 崩在最后一行才是有用的信息)。
OUTPUT_TAIL_CHARS = 4000


# ---------------------------------------------------------------------------
# 计划
# ---------------------------------------------------------------------------
@dataclass
class GroupSpec:
    """计划里的一组 (要跑什么, 以及它是哪一组)。"""

    group: str
    variable: str = ""
    meta: dict[str, str] = field(default_factory=dict)
    script: str = ""
    args: list[str] = field(default_factory=list)
    adopted_dir: str | None = None
    """非空 = 这一组已经跑完, 直接登记这个目录, 不重跑。"""


def load_plan(path: Path) -> dict[str, Any]:
    """读计划文件。

    Args:
        path: 计划 JSON 的路径。

    Returns:
        计划字典 (含 ``root`` 与 ``groups``)。

    Raises:
        ValueError: 文件不是合法 JSON, 或缺 ``groups`` 列表。
        OSError: 文件读不出来。
    """
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} 的顶层必须是 JSON 对象")
    if not isinstance(data.get("groups"), list):
        raise ValueError(f"{path} 缺 'groups' 列表")
    return data


def build_argv(script: str, args: list[str]) -> list[str]:
    """组命令 = ``[脚本] + 参数``。

    ``script`` 以 ``.py`` 结尾时前面加当前解释器 (``sys.executable``): 计划里写的
    是仓库相对路径, 用哪个解释器必须由**跑计划的那个环境**决定, 而不是由写计划
    的人决定 (换机器 / 换 venv 时计划不用改)。

    Args:
        script: 命令本体。
        args: 追加的参数。

    Returns:
        可直接交给 :func:`subprocess.run` 的参数列表。
    """
    tail = [str(a) for a in args]
    if str(script).lower().endswith(".py"):
        return [sys.executable, str(script), *tail]
    return [str(script), *tail]


def build_env() -> dict[str, str]:
    """子进程的环境变量 = 当前环境 + 仓库的四个搜索路径。

    探针脚本要 import ``ao_shaping`` / ``scripts.*``, 而计划里写的是仓库相对
    路径, 所以 PYTHONPATH 必须带上仓库根与 ``src``。**只在缺失时追加**, 不覆盖
    调用者已有的设置。
    """
    env = os.environ.copy()
    wanted = [
        str(ROOT),
        str(ROOT / "src"),
        str(ROOT / "libs"),
        str(ROOT / "libs" / "OOPAO"),
    ]
    existing = env.get("PYTHONPATH", "")
    parts = [p for p in wanted if p not in existing.split(os.pathsep)]
    if existing:
        parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def resolve_base_dir(spec: GroupSpec, root: Path) -> Path:
    """这一组的**根目录** (组目录建在它下面)。

    计划里显式给了 ``--out-root`` (离焦探针) 或 ``--out`` (泽尼克扫描) 就用它,
    否则用驱动自己的根目录 —— 那时组目录就是 ``<root>/<组名>/``。
    """
    for flag in ("--out-root", "--out"):
        if flag in spec.args:
            idx = spec.args.index(flag)
            if idx + 1 < len(spec.args):
                return Path(str(spec.args[idx + 1]))
    return Path(root)


def resolve_out_dir(spec: GroupSpec, root: Path) -> Path:
    """这一组的产物目录 = ``<根目录>/<组名>/``。"""
    return group_output_dir(resolve_base_dir(spec, root), spec.group)


# ---------------------------------------------------------------------------
# 登记 (adopt)
# ---------------------------------------------------------------------------
def read_group_metadata(directory: Path) -> tuple[str, dict[str, str]]:
    """从已完成的组目录里读回 ``(variable, meta)``。

    两个探针都把自己的组元数据写进组目录 (``run_config.json`` / ``summary.json``),
    所以登记一个计划外的组时, 它的自变量与固定参数不必在计划里重抄一遍。

    Args:
        directory: 组目录。

    Returns:
        ``(variable, meta)``; 读不到时是 ``("", {})``。
    """
    for name in ("run_config.json", "summary.json"):
        path = directory / name
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        raw_meta = data.get("group_meta")
        meta = (
            {str(k): str(v) for k, v in raw_meta.items()}
            if isinstance(raw_meta, dict)
            else {}
        )
        return str(data.get("variable", "") or ""), meta
    return "", {}


def check_adopt_dir(directory: Path) -> str | None:
    """检查一个待登记的目录: 存在, 且有落盘产物。

    Args:
        directory: 待登记目录。

    Returns:
        ``None`` 表示可以登记; 否则是给操作者看的中文原因。
    """
    if not directory.exists():
        return f"目录不存在: {directory}"
    if not directory.is_dir():
        return f"不是目录: {directory}"
    if not any((directory / marker).is_file() for marker in ARTEFACT_MARKERS):
        return (
            f"目录里没有落盘产物 (需要 {' 或 '.join(ARTEFACT_MARKERS)} 之一): {directory}"
        )
    return None


# ---------------------------------------------------------------------------
# 执行
# ---------------------------------------------------------------------------
def _tail(value: Any, limit: int = OUTPUT_TAIL_CHARS) -> str:
    """把子进程输出截成尾巴 (bytes / str / None 都能吃)。"""
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return str(value)[-limit:]


def _manifest_n_rows(base: Path, group: str) -> int | None:
    """从组自己的清单里读这一组的行数 (读不到就是 ``None``)。"""
    for entry in load_manifest(base):
        if str(entry.get("group", "")) == group:
            try:
                return int(entry.get("n_rows", 0))
            except (TypeError, ValueError):
                return None
    return None


def _driver_modules() -> frozenset[str]:
    """当前已加载的 ``ao_shaping.drivers.*`` 模块名。"""
    return frozenset(
        m for m in sys.modules if m.startswith("ao_shaping.drivers")
    )


def assert_no_devices(before: frozenset[str]) -> None:
    """dry-run 的硬保证: 计划解析过程中**没有新**导入 ``ao_shaping.drivers``。

    比的是「开始之前」与「现在」的差集, 而不是「整个进程里一个驱动模块都没有」:
    后者会把**同进程里别的模块**引进来的驱动包算到 dry-run 头上 —— 例如测试
    进程里先 import 了 ``scripts.slm.run_gs_refine_zernike_sweep`` (它要
    ``ao_shaping.drivers.slm.factory``), 那是误判。判据是 dry-run 自己**没有
    牵进**驱动包。
    """
    leaked = sorted(m for m in _driver_modules() - before)
    if leaked:
        raise SystemExit(
            "dry-run 违约: 计划解析过程中导入了驱动包:\n  " + "\n  ".join(leaked)
        )


def run_group(
    spec: GroupSpec,
    root: Path,
    timeout_s: float | None,
) -> dict[str, Any]:
    """把一组当作独立子进程跑完, 返回它的索引条目。

    **失败不中断整轮**: 退出码非零记 ``failed``, 超时记 ``timeout``, 两种情况
    都继续跑下一组 —— 一组坏掉不该让后面几组陪葬。

    Args:
        spec: 这一组的计划。
        root: 驱动自己的根目录 (组目录的缺省位置)。
        timeout_s: 单组超时 (s); ``None`` = 不限。

    Returns:
        索引条目 (字段见模块文档)。
    """
    argv = build_argv(spec.script, spec.args)
    base = resolve_base_dir(spec, root)
    out_dir = group_output_dir(base, spec.group)
    started = time.time()
    record: dict[str, Any] = {
        "group": spec.group,
        "variable": spec.variable,
        "meta": dict(spec.meta),
        "script": spec.script,
        "argv": list(argv),
        "start_utc": utc_now_iso(),
        "end_utc": None,
        "duration_s": 0.0,
        "exit_code": None,
        "status": "failed",
        "out_dir": str(out_dir),
        "out_dir_exists": out_dir.is_dir(),
        "n_rows": None,
    }
    print(f"[组] {spec.group} (自变量 {spec.variable or '?'}) -> {' '.join(argv)}",
          flush=True)
    try:
        completed = subprocess.run(  # noqa: S603 - 命令来自计划文件, 本来就要执行
            argv,
            cwd=str(ROOT),
            env=build_env(),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        record["exit_code"] = int(completed.returncode)
        record["status"] = "ok" if completed.returncode == 0 else "failed"
        record["stdout_tail"] = _tail(completed.stdout)
        record["stderr_tail"] = _tail(completed.stderr)
    except subprocess.TimeoutExpired as exc:
        record["status"] = "timeout"
        record["exit_code"] = None
        record["stdout_tail"] = _tail(exc.stdout)
        record["stderr_tail"] = _tail(exc.stderr)
    except OSError as exc:
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
    record["end_utc"] = utc_now_iso()
    record["duration_s"] = round(max(time.time() - started, 0.0), 6)
    record["out_dir_exists"] = out_dir.is_dir()
    rows = _manifest_n_rows(base, spec.group)
    if rows is not None:
        record["n_rows"] = rows
    print(f"[组] {spec.group}: {record['status']} "
          f"({record['duration_s']:.1f}s, exit={record['exit_code']})", flush=True)
    return record


def adopt_group(spec: GroupSpec, directory: Path) -> dict[str, Any]:
    """把一个已完成的测量目录登记为某一组 (不重跑)。

    Args:
        spec: 这一组的计划 (组名 / 自变量 / 固定参数; 计划里没写的从目录里读)。
        directory: 已完成的组目录。

    Returns:
        索引条目, ``status`` 为 ``"adopted"``。
    """
    variable, meta = read_group_metadata(directory)
    started = time.time()
    record = {
        "group": spec.group,
        "variable": spec.variable or variable,
        "meta": dict(spec.meta) if spec.meta else meta,
        "script": spec.script,
        "argv": [],
        "start_utc": utc_now_iso(),
        "end_utc": utc_now_iso(),
        "duration_s": round(max(time.time() - started, 0.0), 6),
        "exit_code": None,
        "status": "adopted",
        "out_dir": str(directory),
        "out_dir_exists": True,
        "n_rows": _manifest_n_rows(directory.parent, spec.group),
    }
    print(f"[组] {spec.group}: adopted (已有目录 {directory})", flush=True)
    return record


def write_groups_manifest(
    root: Path,
    records: list[dict[str, Any]],
    *,
    plan_path: str,
    created_utc: str,
) -> Path:
    """写 (或覆盖) ``<root>/groups_manifest.json``。

    **每组跑完就写一次**: 整轮被强杀时, 已经跑完的那几组仍然在索引里。

    Args:
        root: 根目录。
        records: 已完成的组条目。
        plan_path: 计划文件路径 (记进索引, 便于复现)。
        created_utc: 本轮索引的创建时刻 (第一次写时定下, 之后不变)。

    Returns:
        索引文件路径。
    """
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "root": str(root),
        "plan": str(plan_path),
        "created_utc": created_utc,
        "updated_utc": utc_now_iso(),
        "groups": records,
        "summary": [
            f"{r['group']}: {r['status']} ({float(r['duration_s']):.1f}s)"
            for r in records
        ],
    }
    path = root / GROUPS_MANIFEST_NAME
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--plan", required=True, help="计划 JSON 的路径")
    parser.add_argument("--root", default=None,
                        help="覆盖计划里的 root (这一轮的根目录)")
    parser.add_argument("--only", action="append", default=[], metavar="GROUP",
                        help="只跑这一组 (可重复); 计划里没有这一组就报错")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印计划 (组 / 自变量 / 命令), 什么都不执行")
    parser.add_argument("--print-plan", action="store_true",
                        help="把解析后的计划按 JSON 打到 stdout 然后退出")
    parser.add_argument("--adopt-existing", action="append", nargs=2,
                        default=[], metavar=("GROUP", "DIR"),
                        help="把一个已完成的测量目录登记为这一组 (不重跑); 可重复")
    parser.add_argument("--timeout-s", type=float, default=None,
                        help="单组超时 (s); 不给 = 不限")
    return parser


def resolve_specs(
    plan: dict[str, Any],
    adopt_map: dict[str, Path],
    only: list[str],
) -> list[GroupSpec]:
    """把计划 + 登记请求合成一张「要处理的组」的表。

    Args:
        plan: 计划字典。
        adopt_map: ``{组名: 已完成目录}``。
        only: 只处理这些组 (空 = 全部)。

    Returns:
        组列表 (计划顺序; 计划外的登记组追加在末尾)。

    Raises:
        ValueError: 某一组没有组名, 或 ``--only`` 指名的组不在计划里。
    """
    specs: list[GroupSpec] = []
    for raw in plan.get("groups", []):
        if not isinstance(raw, dict):
            raise ValueError("计划里的每一组都必须是 JSON 对象")
        group = str(raw.get("group", "")).strip()
        if not group:
            raise ValueError("计划里的每一组都必须有非空的 'group'")
        raw_meta = raw.get("meta") or {}
        if not isinstance(raw_meta, dict):
            raise ValueError(f"组 {group} 的 'meta' 必须是对象")
        specs.append(
            GroupSpec(
                group=group,
                variable=str(raw.get("variable", "") or ""),
                meta={str(k): str(v) for k, v in raw_meta.items()},
                script=str(raw.get("script", "") or ""),
                args=[str(a) for a in (raw.get("args") or [])],
            )
        )
    known = {s.group for s in specs}
    for group, directory in adopt_map.items():
        if group not in known:
            specs.append(GroupSpec(group=group, adopted_dir=str(directory)))
    for spec in specs:
        if spec.group in adopt_map:
            spec.adopted_dir = str(adopt_map[spec.group])
    if only:
        wanted = set(only)
        missing = sorted(wanted - {s.group for s in specs})
        if missing:
            raise ValueError(f"计划里没有这些组: {missing}")
        specs = [s for s in specs if s.group in wanted]
    return specs


def main(argv: list[str] | None = None) -> int:
    """按计划逐组跑 (或只打印计划)。

    Returns:
        进程退出码: 0 = 全部组成功 (或只是打印计划); 1 = 有组失败/超时;
        2 = 用法或计划有问题。
    """
    args = _build_parser().parse_args(argv)
    # dry-run 的「没有牵进驱动包」检查: 比的是**开始之前**与**现在**的差集。
    drivers_before = _driver_modules()
    try:
        plan = load_plan(Path(args.plan))
    except (OSError, ValueError) as exc:
        print(f"计划读取失败: {exc}", file=sys.stderr)
        return 2

    root_text = str(args.root) if args.root else str(plan.get("root", "") or "")
    if not root_text:
        print("计划里没有 root, 请用 --root 指定根目录", file=sys.stderr)
        return 2
    root = Path(root_text)

    adopt_map: dict[str, Path] = {}
    for group, directory in args.adopt_existing:
        path = Path(str(directory))
        problem = check_adopt_dir(path)
        if problem is not None:
            print(f"登记失败 [{group}]: {problem}", file=sys.stderr)
            return 2
        adopt_map[str(group)] = path

    try:
        specs = resolve_specs(plan, adopt_map, list(args.only))
    except ValueError as exc:
        print(f"计划有问题: {exc}", file=sys.stderr)
        return 2

    if args.print_plan:
        print(
            json.dumps(
                {
                    "root": str(root),
                    "plan": str(args.plan),
                    "timeout_s": args.timeout_s,
                    "groups": [
                        {
                            "group": s.group,
                            "variable": s.variable,
                            "meta": s.meta,
                            "script": s.script,
                            "argv": build_argv(s.script, s.args),
                            "adopted_dir": s.adopted_dir,
                        }
                        for s in specs
                    ],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        assert_no_devices(drivers_before)
        return 0

    if args.dry_run:
        print(f"根目录: {root}")
        for spec in specs:
            argv = build_argv(spec.script, spec.args)
            print(f"  组 {spec.group} (自变量 {spec.variable or '?'})")
            print(f"    命令: {' '.join(argv)}")
            if spec.adopted_dir:
                print(f"    登记已有目录: {spec.adopted_dir}")
        print("dry-run: 没有执行任何命令, 也没有打开任何设备")
        assert_no_devices(drivers_before)
        return 0

    created_utc = utc_now_iso()
    records: list[dict[str, Any]] = []
    for spec in specs:
        if spec.adopted_dir:
            record = adopt_group(spec, Path(spec.adopted_dir))
        else:
            record = run_group(spec, root, args.timeout_s)
        records.append(record)
        # 每组跑完立刻写 —— 整轮被强杀时, 已完成的组仍然在索引里。
        write_groups_manifest(
            root, records, plan_path=str(args.plan), created_utc=created_utc
        )

    manifest_path = write_groups_manifest(
        root, records, plan_path=str(args.plan), created_utc=created_utc
    )
    failed = [r for r in records if r["status"] in ("failed", "timeout")]
    print(f"\n索引: {manifest_path}")
    for line in (
        f"{r['group']}: {r['status']} ({float(r['duration_s']):.1f}s)"
        for r in records
    ):
        print(f"  {line}")
    if failed:
        print(f"有 {len(failed)} 组失败/超时 (其余组已照常完成并记录)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
