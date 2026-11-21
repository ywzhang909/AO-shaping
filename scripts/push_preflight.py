"""#251 P2-A —— push 前 60 秒自检 (缺陷 4 的通道封堵)。

ci169 的教训: `parents[N]` 深度错、机器绝对路径、本地 `_fmt` 重定义、dry-run
读 ``data/`` —— 这些守卫全部存在且按设计工作, 但只在 push 之后的完整分片里
才跑到, 每个错误都要付一整段 CI 红窗口。本脚本把「新脚本类守卫 + 三道全仓
门禁」提前到本地 push 前:

    python scripts/push_preflight.py

跑: ruff → 术语 → 模块体积棘轮 → 四个静态守卫文件 (repo-root bootstrap /
R-25 helper 复发 / dry-run 自包含 / golden 容差)。全绿退出 0; 任一失败列出
名字退出非 0。完整回归仍归 CI; 本脚本只拦「60 秒内可知」的四类。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CHECKS: list[tuple[str, list[str]]] = [
    ("ruff", [sys.executable, "-m", "ruff", "check", "."]),
    ("terminology", [sys.executable, "scripts/_common/term_unify_20261016.py", "--check"]),
    ("module-size", [sys.executable, "scripts/check_module_size.py", "--check"]),
    (
        "script-guards",
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "-o",
            "addopts=",
            "tests/ao_shaping/scripts/test_script_repo_root_bootstrap.py",
            "tests/ao_shaping/scripts/test_common_helpers_not_reintroduced.py",
            "tests/ao_shaping/scripts/test_dry_run_self_contained.py",
            "tests/ao_shaping/test_golden_tolerance_policy.py",
            "tests/ao_shaping/test_call_graph_span_drift.py",
        ],
    ),
]


def main() -> int:
    failed: list[str] = []
    for name, cmd in CHECKS:
        print(f"== preflight: {name}", flush=True)
        result = subprocess.run(cmd, cwd=ROOT, check=False)
        if result.returncode != 0:
            failed.append(name)
    if failed:
        print(f"preflight FAILED: {failed} —— 修复后重跑, 不要 push", file=sys.stderr)
        return 1
    print("preflight OK —— 四类已知通道全绿 (完整回归仍由 CI 裁决)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
