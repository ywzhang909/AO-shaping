"""模块规模门禁（ratchet）。

目的
----
``docs/dev/coding_conventions.md`` §9 写着"超过 800 行应考虑拆分"，但这条一直是
**靠人记着**——仓库里 >800 行的文件从 155 个涨到 137+ 个（且多为新建的报告脚本），
期间没有任何东西拦过。本脚本把这条规则变成 CI 能执行的退出码。

为什么是 ratchet 而不是绝对阈值
-------------------------------
落地当天就有 137 个文件 >800 行、14 个 >3000 行。若直接按绝对值判红，门禁第一天
就是红的，于是它会被 `continue-on-error` 降级成"报告但不拦"，退化成和没有一样。
所以：

- **存量**：登记进 baseline，允许存在（留给切片(4) 逐个拆）；
- **增量**：新文件越过 800 行 → 判红；存量文件**行数增长** → 判红；
- 允许变小，绝不允许变大。拆分后从 baseline 里移除即可永久收紧。

为什么门禁只管 `src/`
---------------------
`scripts/` 下是**给 AI 用的脚本**——某次实验/某份报告/某次排查是怎么算出来的。
使用者是发起那次任务的 AI 或人，不是终端用户，所以"为行数好看而拆文件"在那里
没有收益（拆完只是把一次性的代码挪了个地方，还多一层 import 路径要维护）。
2026-10-29 起：`scripts/` 仍**计入报告**（一眼能看到哪个脚本在膨胀），但**不参与
退出码**。相应地把复用价值的东西迁进 `src/`、把过期的删掉，才是那里的正确动作。
判据与细节见 `AGENTS.md`「scripts/ 目录：AI 一次性脚本」。

用法
----
    python3 scripts/check_module_size.py                 # 报告（永远 exit 0）
    python3 scripts/check_module_size.py --check         # 门禁（有违规 exit 1）
    python3 scripts/check_module_size.py --update-baseline   # 重写并打印棘轮方向

``--update-baseline`` 是本门禁**唯一的复位按钮**：它既能收紧也能放宽。收紧是
拆分的结算动作，放宽则是把门禁对某个文件关掉。这两件事在 diff 里长得一样（都是
baseline 里一个数字变了），所以写入时**必须报方向**——见 ``ratchet_changes``。

``--root`` 只作用于扫描范围（测试隔离用）；baseline 始终读仓库里那一份。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: 约定 §9 的"应考虑拆分"线。越过即进入 baseline 受管。
WARN = 800
#: 必拆线：Tier A。仅报告，不额外判红（它们已在 baseline 里，增长会被抓住）。
HARD = 3000

_SCAN_ROOTS = ("src", "scripts")

#: 只有主项目 `src/` 参与门禁。`scripts/` 是 AI 一次性脚本（见模块 docstring 与
#: `AGENTS.md`「scripts/ 目录：AI 一次性脚本」），计入报告但不判红。
_ENFORCED_PREFIXES = ("src/",)

#: 路径片段命中即跳过。vendored SDK 与 tests 不计：前者在 CI 的干净 checkout 里
#: 根本不存在（untracked），后者是"测试文件长"而非"模块长"。
_EXCLUDED_PARTS = {"__pycache__", ".venv", "tests", "_miicam_sdk", "libs", ".git"}

DEFAULT_BASELINE = REPO / "scripts" / "_common" / "module_size_baseline.json"


def _is_excluded(path: Path) -> bool:
    return any(part in _EXCLUDED_PARTS for part in path.parts)


def _count_lines(path: Path) -> int:
    try:
        return len(path.read_text(encoding="utf-8", errors="replace").splitlines())
    except OSError:
        return 0


def scan(root: Path) -> dict[str, int]:
    """返回 {相对 root 的 posix 路径: 行数}，只收录越过 WARN 的模块。"""
    found: dict[str, int] = {}
    for base in _SCAN_ROOTS:
        target = root / base
        if not target.is_dir():
            continue
        for path in sorted(target.rglob("*.py")):
            if _is_excluded(path.relative_to(root)):
                continue
            lines = _count_lines(path)
            if lines > WARN:
                found[path.relative_to(root).as_posix()] = lines
    return dict(sorted(found.items(), key=lambda kv: (-kv[1], kv[0])))


def load_baseline(path: Path) -> dict[str, int]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): int(v) for k, v in data.items() if isinstance(v, int)}


def _enforced(rel: str) -> bool:
    """该模块是否受门禁约束（`scripts/` 只报告）。"""
    return rel.startswith(_ENFORCED_PREFIXES)


def violations(
    current: dict[str, int], baseline: dict[str, int]
) -> list[tuple[str, int, int | None]]:
    """(路径, 当前行数, baseline 行数|None)。None 表示新越线。

    只返回受门禁约束的模块：`scripts/` 的膨胀在报告里可见，但不拦 CI。
    """
    found: list[tuple[str, int, int | None]] = []
    for rel, lines in current.items():
        if not _enforced(rel):
            continue
        prev = baseline.get(rel)
        if prev is None:
            found.append((rel, lines, None))
        elif lines > prev:
            found.append((rel, lines, prev))
    return found


def ratchet_changes(
    previous: dict[str, int], current: dict[str, int]
) -> tuple[
    list[tuple[str, int, int]], list[tuple[str, int, int]], list[str], list[str]
]:
    """把一次 baseline 重写拨动的方向分成四类。

    ``--update-baseline`` 是本门禁唯一的复位按钮，而它在 diff 里看不出方向：
    「拆完把 1097 收紧成 989」与「特性改完把 870 抬高成 947」都是 JSON 里一个
    数字变了。实测有两次特性提交是这样把门禁关掉的（`54782eec` focus_probe
    +42、`15ea0a22` daheng/miicam/slm_pib_sim +77/+4/+4），两次都不是恶意，只是
    写入动作没有报方向。所以这里把方向算出来给调用方打印。

    返回 ``(raised, tightened, added, removed)``：
    ``raised`` / ``tightened`` 是 ``(路径, 旧值, 新值)``，``added`` / ``removed``
    是路径名。``added`` 指本次新进入受管集合的文件，``removed`` 指本次跌出集合
    （或已删除）的文件——后者同样是收紧，只是表现成条目消失。
    """
    raised: list[tuple[str, int, int]] = []
    tightened: list[tuple[str, int, int]] = []
    for rel, lines in current.items():
        if rel not in previous:
            continue
        if lines > previous[rel]:
            raised.append((rel, previous[rel], lines))
        elif lines < previous[rel]:
            tightened.append((rel, previous[rel], lines))
    added = sorted(rel for rel in current if rel not in previous)
    removed = sorted(rel for rel in previous if rel not in current)
    return (
        sorted(raised, key=lambda item: item[0]),
        sorted(tightened, key=lambda item: item[0]),
        added,
        removed,
    )


def _print_ratchet_direction(
    previous: dict[str, int], current: dict[str, int]
) -> None:
    """按 ``ratchet_changes`` 的结果打印方向；抬高的一律逐条列出。"""
    raised, tightened, added, removed = ratchet_changes(previous, current)
    print(
        f"棘轮方向: 收紧 {len(tightened)} / 抬高 {len(raised)} /"
        f" 新增 {len(added)} / 移除 {len(removed)}"
    )
    for rel, old, new in tightened:
        print(f"  收紧: {rel} {old} -> {new} 行 (-{old - new})")
    for rel, old, new in raised:
        print(f"  抬高: {rel} {old} -> {new} 行 (+{new - old})")
    for rel in added:
        print(f"  新增: {rel} ({current[rel]} 行) —— 新越线文件应拆分而非登记")
    for rel in removed:
        print(f"  移除: {rel} (不再 > {WARN} 行, 或已删除)")
    if raised:
        print(
            "  注意: 本次重写**放宽**了门禁: 若这些增长不是经审查的功能性改动"
            "（拆分应只产生『收紧』），请在提交信息里写明理由, 否则请先拆分。"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="模块规模门禁：> %(WARN)s 行的模块受管，只允许变小。" % {"WARN": WARN}
    )
    parser.add_argument(
        "--check", action="store_true", help="把报告变成退出码（有违规 exit 1）"
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="用当前行数重写 baseline（拆分收紧后用它落闸）",
    )
    parser.add_argument("--root", default=str(REPO), help="扫描根（测试隔离用）")
    parser.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    args = parser.parse_args(argv)

    if args.check and args.update_baseline:
        print("--check 与 --update-baseline 互斥", file=sys.stderr)
        return 2

    root = Path(args.root)
    baseline_path = Path(args.baseline)
    current = scan(root)
    gated = {rel: n for rel, n in current.items() if _enforced(rel)}

    if args.update_baseline:
        previous = load_baseline(baseline_path)
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(
            json.dumps(gated, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            f"baseline 已更新: {len(gated)} 个受管模块 -> {baseline_path}"
            f"（scripts/ 另有 {len(current) - len(gated)} 个 > {WARN} 行，仅报告）"
        )
        _print_ratchet_direction(previous, gated)
        return 0

    baseline = load_baseline(baseline_path)
    bad = violations(current, baseline)
    tier_a = [(r, n) for r, n in gated.items() if n > HARD]
    ungated = [(r, n) for r, n in current.items() if not _enforced(r)]

    for rel, lines in current.items():
        if not _enforced(rel):
            print(f"  {lines:>6} 行  {rel}  (仅报告, 不参与门禁)")
            continue
        prev = baseline.get(rel)
        if prev is None:
            print(f"  {lines:>6} 行  {rel}")
        elif lines > prev:
            print(f"  {lines:>6} 行  {rel}  (+{lines - prev}, baseline {prev})")
        else:
            print(f"  {lines:>6} 行  {rel}  (<= baseline {prev})")
    print(
        f"合计: {len(current)} 个模块 > {WARN} 行"
        f" (受管 src/ {len(gated)} 个, 其中 {len(tier_a)} 个 > {HARD} 行必拆;"
        f" scripts/ {len(ungated)} 个仅报告)"
    )

    if not args.check:
        return 0

    if bad:
        print("模块规模门禁未通过:")
        for rel, lines, prev in bad:
            if prev is None:
                print(f"  新越线: {rel} ({lines} 行 > {WARN})")
            else:
                print(f"  增长: {rel} {prev} -> {lines} 行 (+{lines - prev})")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
