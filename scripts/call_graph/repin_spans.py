"""#249 A —— call_graph 双 JSON 的函数 span 一次性重钉 (字节格式守恒)。

`docs/dev/call_graph_wiki/raw/assets/call_graph.json` 与 `call_graph_final.json`
的节点 `lineno`/`end` 会随每次行号平移静默漂移 (#249 实测 12→40→47 三连涨,
无 CI 守卫)。本脚本按 (file, qualname 末段) 对拍磁盘 AST, 把 JSON 钉回实测
span; 已搬移的函数按 `MOVED` 表回指新文件/新模块。

格式契约 (实测逐字节成立, 由 `test_call_graph_span_drift.py` 把守):
文件内容 == `json.dumps(data, ensure_ascii=False, indent=1) + "\\n"`,
且两份 JSON 逐字节一致 —— 因此内存改字段后整体重 dump 即可, 禁止
`indent=2` / `sorted_keys` 等会重排 9600+ 行的写法。

用法: `python scripts/call_graph/repin_spans.py [--check]`
--check 只报告不写盘; 存在无法解析的节点 (函数消失且不在 MOVED 表) 时退出非 0。
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

P = Path(__file__).resolve().parent          # scripts/call_graph/
REPO = P.parents[1]                            # repo root
ASSETS = REPO / "docs" / "dev" / "call_graph_wiki" / "raw" / "assets"
FILES = [ASSETS / "call_graph.json", ASSETS / "call_graph_final.json"]

#: 一次性历史迁移 (第 173 批, #205 hoist 后的回指): (旧 file, 旧 qualname) ->
#: (新 file, 新 module, 新 qualname)。键命中后按新位置对拍 span。
MOVED: dict[tuple[str, str], tuple[str, str, str]] = {
    ("src/ao_shaping/runners/runner_common.py", name): (
        "src/ao_shaping/runners/voltage_io.py",
        "ao_shaping.runners.voltage_io",
        name,
    )
    for name in ("save_best_voltages", "load_init_v", "plot_pib_debug_grid")
}
MOVED.update(
    {
        # 第 173 批曾把这三处指向 model_in_loop_io.py; 上游 e7d9a305 (MIL 外围
        # 移除) 删掉该模块并把函数重新内联回 runner_common —— 回指旧源。
        ("src/ao_shaping/runners/model_in_loop_io.py", name): (
            "src/ao_shaping/runners/runner_common.py",
            "ao_shaping.runners.runner_common",
            name,
        )
        for name in (
            "save_model_in_loop_results",
            "save_model_in_loop_config",
            "_model_in_loop_json_payload",
        )
    }
)
MOVED[("src/ao_shaping/runners/slm/transfer_shapes_runner.py", "_measured_peak_rates")] = (
    "src/ao_shaping/tools/slm/transfer/session_record.py",
    "ao_shaping.tools.slm.transfer.session_record",
    "measured_peak_rates",
)

_span_cache: dict[str, dict[str, list[tuple[int, int | None]]]] = {}


def disk_spans(rel: str) -> dict[str, list[tuple[int, int | None]]]:
    if rel not in _span_cache:
        out: dict[str, list[tuple[int, int | None]]] = {}
        path = REPO / rel
        if path.is_file():
            tree = ast.parse(path.read_text(encoding="utf-8-sig", errors="replace"))
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                    out.setdefault(node.name, []).append((node.lineno, node.end_lineno))
        _span_cache[rel] = out
    return _span_cache[rel]


def repin(data: dict, *, write: bool) -> tuple[int, int, int, list[str]]:
    span_fixed = moved = ok = 0
    unresolved: list[str] = []
    for key, node in data["nodes"].items():
        if not isinstance(node, dict) or node.get("lineno") is None:
            continue
        if (node["file"], node["qualname"]) in MOVED:
            node["file"], node["module"], node["qualname"] = MOVED[(node["file"], node["qualname"])]
            moved += 1
        rel, fn = node["file"], node["qualname"].split(".")[-1]
        actual = disk_spans(rel).get(fn)
        if not actual:
            unresolved.append(f"{key}: {rel}::{fn} 不在磁盘且不在 MOVED 表")
            continue
        if len(actual) == 1:
            ln, en = actual[0]
        else:
            hit = [a for a in actual if a[0] == node["lineno"]]
            ln, en = hit[0] if hit else actual[0]
        if (node["lineno"], node["end"]) != (ln, en):
            node["lineno"], node["end"] = ln, en
            span_fixed += 1
        else:
            ok += 1
    if write:
        out = json.dumps(data, ensure_ascii=False, indent=1) + "\n"
        for f in FILES:
            f.write_text(out, encoding="utf-8")
    return span_fixed, moved, ok, unresolved


def main() -> int:
    check = "--check" in sys.argv
    data = json.loads(FILES[0].read_text(encoding="utf-8"))
    span_fixed, moved, ok, unresolved = repin(data, write=not check)
    verb = "stale" if check else "repinned"
    print(f"{verb}: spans={span_fixed} moved={moved} already-ok={ok} unresolved={len(unresolved)}")
    for line in unresolved:
        print("UNRESOLVED", line)
    if not check and FILES[0].read_bytes() != FILES[1].read_bytes():
        print("FATAL: 两份 JSON 写盘后不一致", file=sys.stderr)
        return 1
    return 1 if (unresolved or (check and span_fixed)) else 0


if __name__ == "__main__":
    raise SystemExit(main())
