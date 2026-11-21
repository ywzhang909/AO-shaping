"""#249 B —— call_graph 双 JSON 的函数 span 必须与磁盘 AST 一致。

`call_graph.json` / `call_graph_final.json` 的节点 `lineno`/`end` 曾无守卫地
静默漂移 (12→40→47 三连涨, #249 实测)。第 173 批 A 方案一次性清零存量
(重钉 49 span + 8 个搬移节点回指新文件), 本守卫封住再累积的通道:
任何 commit 改了函数行号而未同步重钉 JSON → 红; 该文件在 TODO 台账留痕
(census 同 commit 约定的黄灯态) → 黄。通道语义复用 `census_repin.yellow_or_red`
(#251 P0-A), 不再发明第二套。
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

from tests.ao_shaping.census_repin import yellow_or_red

_ROOT = Path(__file__).resolve().parents[2]
_ASSETS = _ROOT / "docs" / "dev" / "call_graph_wiki" / "raw" / "assets"
_MAIN = _ASSETS / "call_graph.json"
_FINAL = _ASSETS / "call_graph_final.json"

_span_cache: dict[str, dict[str, list[tuple[int, int | None]]]] = {}


def _disk_spans(rel: str) -> dict[str, list[tuple[int, int | None]]]:
    if rel not in _span_cache:
        out: dict[str, list[tuple[int, int | None]]] = {}
        path = _ROOT / rel
        if path.is_file():
            tree = ast.parse(path.read_text(encoding="utf-8-sig", errors="replace"))
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                    out.setdefault(node.name, []).append((node.lineno, node.end_lineno))
        _span_cache[rel] = out
    return _span_cache[rel]


def _drift(data: dict) -> tuple[list[str], list[str]]:
    """返回 (漂移文件列表, 明细行)。按 (file, qualname 末段) 对拍磁盘 AST。"""
    drift_files: list[str] = []
    detail: list[str] = []
    for key, node in data["nodes"].items():
        if not isinstance(node, dict) or node.get("lineno") is None:
            continue
        rel, fn = node["file"], node["qualname"].split(".")[-1]
        actual = _disk_spans(rel).get(fn)
        if actual is None:
            drift_files.append(rel)
            detail.append(f"{key}: {rel}::{fn} 不在磁盘 (函数被搬移/删除, JSON 未同步)")
            continue
        if (node["lineno"], node["end"]) not in actual:
            drift_files.append(rel)
            detail.append(f"{key}: JSON ({node['lineno']},{node['end']}) vs 磁盘 {actual}")
    return drift_files, detail


def test_the_two_graph_jsons_stay_byte_identical() -> None:
    assert _MAIN.read_bytes() == _FINAL.read_bytes(), (
        "#249: call_graph.json 与 call_graph_final.json 必须逐字节一致 "
        "(生成链 finalize_graph.py 就是写两份); 分叉 = 重钉只改了一边"
    )


def test_function_spans_match_the_disk_ast() -> None:
    data = json.loads(_MAIN.read_text(encoding="utf-8"))
    drift_files, detail = _drift(data)
    if drift_files:
        yellow_or_red(
            sorted(set(drift_files)),
            "call_graph JSON span 漂移 (#249): " + "; ".join(detail[:12]),
        )


def test_the_guard_itself_catches_a_shifted_span() -> None:
    """负样本: span 差一行、函数不存在, 都必须进漂移清单。"""
    real_key = next(
        k
        for k, v in json.loads(_MAIN.read_text(encoding="utf-8"))["nodes"].items()
        if isinstance(v, dict) and v.get("lineno") is not None
    )
    base = json.loads(_MAIN.read_text(encoding="utf-8"))["nodes"][real_key]
    shifted = dict(base, lineno=base["lineno"] + 1)
    gone = dict(base, qualname="no_such_function_xyz_249")
    nodes = {real_key: shifted, "gone": gone}
    drift_files, detail = _drift({"nodes": nodes})
    assert len(detail) == 2 and base["file"] in drift_files
