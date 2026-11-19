"""Generate the AO-shaping main.py call-graph LLM-wiki (dev handbook).

Every page is built from call_graph_final.json (real AST data) + the function's
actual docstring. No invented content.
"""
from __future__ import annotations

import ast
from loguru import logger
import json
import re
from collections import defaultdict
from pathlib import Path

import sys
P = Path(__file__).parent
src = (P / "build_call_graph.py").read_text().split('if __name__ == "__main__":')[0]
ns = {"__file__": str(P / "build_call_graph.py")}
exec(compile(src, "bcg", "exec"), ns)

REPO = P.parents[1]
WIKI = REPO / "docs" / "dev" / "call_graph_wiki"
BASE = WIKI
CG = WIKI / "raw" / "assets"
data = json.loads((CG / "call_graph_final.json").read_text())
nodes, edges, cmd = data["nodes"], data["edges"], data["cmd_entries"]
mods = ns["load_modules"]()

# ---- similar-function relations (build_similarity.py output, optional) ----
SIM_PATH = CG / "similar_functions.json"
SIM = None
if SIM_PATH.exists():
    SIM = json.loads(SIM_PATH.read_text())

def sim_partners(qual):
    """All linked similar partners of qual: [(qual, score, reason), ...] sorted."""
    if not SIM:
        return []
    out = []
    for p in SIM["pairs"]:
        if p["a"] == qual:
            out.append((p["b"], p["score"], p["reason"]))
        elif p["b"] == qual:
            out.append((p["a"], p["score"], p["reason"]))
    return sorted(out, key=lambda x: -x[1])

def sim_block(qual):
    """Markdown section '## 相似函数（自动）' for one qual; '' if none."""
    parts = sim_partners(qual)
    if not parts:
        return ""
    body = "\n## 相似函数（自动）\n\n"
    body += "与本页函数结构/职责相近的函数（相似度得分，来自 `build_similarity.py`；" \
            "含义与阈值见 [[relations/similar-functions|相似函数总览]]）：\n\n"
    for q, score, reason in parts:
        body += f"- {score:.3f} — {wiki_link(q)}（{reason}）\n"
    return body + "\n"

DATE = "2026-10-13"

# ---------- page registry ----------
# slug -> (qualname, title, tags, kind)
PAGES: dict[str, tuple] = {}

def reg(slug, qual, title, tags, kind):
    PAGES[slug] = (qual, title, tags, kind)

reg("entry/main-py", "main.main", "main.py（仓库根入口）", ["entry", "cli"], "entry")
reg("entry/cli", "ao_shaping.main.cli", "cli（Click 根命令组）", ["entry", "cli"], "entry")

CMD_ORDER = [
    ("wf", "波前 RMS 优化（DM 电压 + WFS）"),
    ("pib", "轴向光束 PIB 优化（DM 电压 + CCD）"),
    ("pipeline", "串行流水线：先 WF 后 PIB"),
    ("combined", "AdaMOD 综合 PIB 优化（自适应桶收缩）"),
    ("dm-matrix", "DM 响应矩阵标定（push-pull）"),
    ("hadamard-matrix", "Hadamard 响应矩阵标定（并行扰动）"),
    ("alt-voltage", "R50 交替电压下发 + ADC 同步采集"),
    ("full-voltage", "AsyncMicroDM 全量交替电压下发"),
    ("zernike-matrix", "SLM Zernike 响应矩阵标定 + 离线矫正自检"),
    ("closed-loop", "基于响应矩阵的闭环波前优化"),
    ("rms-zernike", "SLM Zernike RMS 优化（SPGD/启发式）"),
    ("ga-zernike", "遗传算法 Zernike 波前优化"),
    ("greedy-zernike", "贪婪局部搜索 Zernike 波前优化"),
    ("slm-pib", "SLM Zernike 整形（spgd/heuristic 子命令）"),
    ("slm-gsnet", "SLM freeform 方形整形（spgd/heuristic/train）"),
    ("slm-gs-refine", "GS 预整形 + 全频随机相位 SPGD 细化"),
    ("slm-model-in-loop", "物理预测模型闭环方形整形（反复迭代）"),
    ("slm-transfer-shapes", "多形状物理预测模型闭环整形（从零训练 → 逐形状整形 → 共同重放 → 记录）"),
    ("slm-lut", "SLM 灰度-相位 LUT 标定（衍射效率反演）"),
    ("slm-diagnose", "SLM 面板硬件自检（逐级定位）"),
]
# explicit command -> runners-package symbol (two names don't derive from the
# CLI name: closed-loop -> zernike_closed_loop_run; the two tools commands live in
# _TOOL_ENTRIES instead, since they never reach the asset's cmd_entries.
CMD_SYMS = {
    "wf": "wf_run",
    "pib": "pib_run",
    "pipeline": "pipeline_run",
    "combined": "combined_run",
    "dm-matrix": "dm_matrix_run",
    "hadamard-matrix": "hadamard_matrix_run",
    "alt-voltage": "alt_voltage_run",
    "full-voltage": "full_voltage_run",
    "zernike-matrix": "zernike_matrix_run",
    "closed-loop": "zernike_closed_loop_run",
    "rms-zernike": "rms_zernike_run",
    "ga-zernike": "ga_zernike_run",
    "greedy-zernike": "greedy_zernike_run",
    "slm-pib": "slm_pib_run",
    "slm-gsnet": "slm_gsnet_run",
    "slm-gs-refine": "slm_gs_refine_run",
    "slm-model-in-loop": "slm_model_in_loop_run",
    "slm-transfer-shapes": "slm_transfer_shapes_run",
    "slm-lut": "slm-lut",
    "slm-diagnose": "slm-diagnose",
}
#: ``slm-lut`` and ``slm-diagnose`` are tools commands, so they never appear in the
#: asset's ``cmd_entries``; they are the only tolerated misses, named explicitly.
_TOOL_ENTRIES = {
    "slm-lut": "ao_shaping.tools.slm.calibration.lut_runner.run",
    "slm-diagnose": "ao_shaping.tools.slm.panel.diagnose.main",
}


def cmd_qualifier(name):
    """Resolve a CLI name to its entry function, or refuse to invent one.

    Two of the three copies of this lookup used to end in a bare ``else`` that handed
    back diagnose's entry point for *any* unresolved name, so a command that was no
    longer registered still got published -- page, table row and index entry -- with a
    fabricated mapping, in a wiki whose SCHEMA calls this table the authority on the
    CLI surface.
    """
    qual = cmd.get(CMD_SYMS[name]) or _TOOL_ENTRIES.get(name)
    if qual is None:
        raise LookupError(
            f"{name}: no entry function for {CMD_SYMS[name]!r} in cmd_entries -- if the "
            "command is gone, drop it from CMD_ORDER/CMD_SYMS; if it is a tools command, "
            "add it to _TOOL_ENTRIES"
        )
    return qual


def registered_cli_commands():
    """The names ``main.py`` registers, read from its AST.

    The registry above is hand-maintained and had already kept a retired command while
    stating the count as a literal, so generation now stops if the two disagree.
    """
    tree = ast.parse((REPO / "src" / "ao_shaping" / "main.py").read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_command"
        ):
            for kw in node.keywords:
                if kw.arg == "name" and isinstance(kw.value, ast.Constant):
                    names.add(kw.value.value)
    return names


_drift = sorted({name for name, _ in CMD_ORDER} ^ registered_cli_commands())
if _drift:
    raise SystemExit(
        f"wiki CLI registry disagrees with main.py on {len(_drift)} name(s): {_drift} "
        "-- regenerate the asset or fix CMD_ORDER before publishing a wrong table"
    )


for name, desc in CMD_ORDER:
    reg(f"commands/{name}", cmd_qualifier(name), f"{name} — {desc}", ["command"], "command")

reg("functions/optimizer-rms-dm", "ao_shaping.optimizer.wf.rms.optimizer_rms_dm",
    "optimizer_rms_dm（DM 电压 SPGD 波前 RMS 优化）", ["optimizer", "wavefront", "spgd"], "function")
reg("functions/optimize-pib-wfless", "ao_shaping.optimizer.wfless.pib.optimize_pib",
    "optimize_pib（DM 电压无波前传感 PIB 优化）", ["optimizer", "pib", "wfless"], "function")
reg("functions/optimize-pib-combined", "ao_shaping.optimizer.combined_optimizer.optimize_pib",
    "optimize_pib（AdaMOD 综合 PIB + 自适应桶收缩）", ["optimizer", "pib", "adapmod"], "function")
reg("functions/optimizer-ga", "ao_shaping.optimizer.wf.ga_zernike.optimizer_ga",
    "optimizer_ga（遗传算法 Zernike 优化）", ["optimizer", "zernike", "ga"], "function")
reg("functions/optimizer-greedy", "ao_shaping.optimizer.wf.greedy_zernike.optimizer_greedy",
    "optimizer_greedy（贪婪局部搜索 Zernike 优化）", ["optimizer", "zernike", "greedy"], "function")
reg("functions/optimizer-rms-slm", "ao_shaping.optimizer.wf.rms_by_zernike.optimizer_rms_slm",
    "optimizer_rms_slm（SLM Zernike RMS 优化，AdaMOD/AdamW）", ["optimizer", "zernike", "slm"], "function")
reg("functions/optimize-slm-square", "ao_shaping.optimizer.wfless.slm_square_shaping.optimize_slm_square",
    "optimize_slm_square（SLM 方形远场 SPGD 优化）", ["optimizer", "shaping", "square", "spgd"], "function")
reg("functions/calibrate-dm-matrix", "ao_shaping.optimizer.wf.dm_response_matrix.calibrate_dm_response_matrix",
    "calibrate_dm_response_matrix（DM 响应矩阵标定内核）", ["calibration", "dm", "matrix"], "function")
reg("functions/save-dm-matrix", "ao_shaping.optimizer.wf.dm_response_matrix.save_dm_response_matrix",
    "save_dm_response_matrix（DM 矩阵落盘）", ["calibration", "io"], "function")
reg("functions/save-zernike-matrix", "ao_shaping.optimizer.wf.zernike_response_matrix.save_zernike_response_matrix",
    "save_zernike_response_matrix（Zernike 矩阵落盘）", ["calibration", "io"], "function")
reg("functions/spgd-gradient", "ao_shaping.optimizer.spgd.spgd_gradient",
    "spgd_gradient（SPGD 梯度估计，float64 防溢出）", ["algorithm", "spgd", "gradient"], "function")
reg("functions/run-heuristic-search", "ao_shaping.algorithm.heuristic.search.run_heuristic_search",
    "run_heuristic_search（启发式搜索统一驱动）", ["algorithm", "heuristic", "ga", "pso"], "function")
reg("functions/resolve-dm", "ao_shaping.drivers.dm._registry.resolve_dm",
    "resolve_dm（DM 类型解析与实例创建）", ["driver", "dm", "registry"], "function")
reg("functions/create-camera", "ao_shaping.drivers.ccd.common.create_camera",
    "create_camera（CCD 相机实例工厂）", ["driver", "ccd", "factory"], "function")
reg("functions/resolve-wfs", "ao_shaping.drivers.wfs._registry.resolve_wfs",
    "resolve_wfs（WFS 类型解析与实例创建）", ["driver", "wfs", "registry"], "function")
reg("functions/with-params", "ao_shaping.utils.cli.params.with_params",
    "with_params（dataclass → Click options 装饰器）", ["cli", "utils", "params"], "function")
reg("functions/recorder", "ao_shaping.utils.io.file.Recorder",
    "Recorder（优化过程记录器）", ["io", "recorder", "utils"], "class")
reg("functions/debug-artifacts", "ao_shaping.utils.io.file.save_optimization_debug_artifacts",
    "save_optimization_debug_artifacts / save_recorder_debug_artifacts（调试产物落盘）", ["io", "debug"], "function")
reg("functions/path-dirs", "ao_shaping.utils.io.file.gen_date_dir",
    "gen_date_dir / make_date_dir / 时间戳工具（目录与命名）", ["io", "utils", "time"], "function")
reg("functions/debug-mode", "ao_shaping.utils.io.cli_helpers.get_debug_mode",
    "get_debug_mode / resolve_debug（DEBUG 判定）", ["cli", "debug", "utils"], "function")
reg("functions/zernike-calc", "ao_shaping.utils.wavefront.zernike_calc.zernike_modes",
    "zernike_modes / calc_n_zernike_terms（Zernike 模式与阶数）", ["zernike", "wavefront", "utils"], "function")
reg("functions/spots", "ao_shaping.utils.image.spots_calc.centroid",
    "centroid / radius / pib_ratio_mask（光斑量测）", ["image", "spots", "utils"], "function")
reg("functions/matrix-utils", "ao_shaping.utils.wavefront.matrix_utils.compute_pinv",
    "compute_pinv / compute_lstsq / safe_pinv / effective_cond（矩阵伪逆与条件数）", ["matrix", "utils", "calibration"], "function")
reg("functions/slm-lut", "ao_shaping.utils.slm.slm_lut.build_inverse_lut",
    "build_inverse_lut / invert_depth_scan / depth_pattern（SLM LUT 数学）", ["slm", "lut", "utils"], "function")
reg("functions/runner-common", "ao_shaping.runners.runner_common.parse_center",
    "runner_common（命令共享工具：center 解析 / 配置构建 / 结果保存）", ["runner", "utils"], "module")

# ---------- helpers ----------
qual2slug = {q: s for s, (q, _, _, _) in PAGES.items()}
# alias: additional real qualnames that should link to an existing page
ALIAS = {
    "ao_shaping.utils.io.file.save_recorder_debug_artifacts": "functions/debug-artifacts",
    "ao_shaping.utils.io.file.save_optimization_debug_artifacts": "functions/debug-artifacts",
    "ao_shaping.utils.io.file.gen_file_path_uuid": "functions/path-dirs",
    "ao_shaping.utils.io.file.get_init_V_by_rms": "functions/path-dirs",
    "ao_shaping.utils.io.file.build_debug_save_paths": "functions/debug-artifacts",
    "ao_shaping.utils.io.file.get_debug_mode": "functions/debug-mode",
    "ao_shaping.utils.io.file.resolve_debug": "functions/debug-mode",
    "ao_shaping.utils.io.timestamp.make_date_dir": "functions/path-dirs",
    "ao_shaping.utils.io.timestamp.format_ts": "functions/path-dirs",
    "ao_shaping.utils.io.cli_helpers.get_date_dir_name": "functions/path-dirs",
    "ao_shaping.utils.io.cli_helpers.get_timestamp_str": "functions/path-dirs",
    "ao_shaping.utils.wavefront.matrix_utils.calc_n_zernike_terms": "functions/matrix-utils",
    "ao_shaping.utils.wavefront.zernike_calc.calc_n_zernike_terms": "functions/zernike-calc",
    "ao_shaping.utils.image.spots_calc.radius": "functions/spots",
    "ao_shaping.utils.image.spots_calc.power_in_bucket_mask": "functions/spots",
    "ao_shaping.utils.image.spots_calc.pib_ratio_mask": "functions/spots",
    "ao_shaping.utils.slm.slm_lut.depth_pattern": "functions/slm-lut",
    "ao_shaping.utils.slm.slm_lut.offset_pattern": "functions/slm-lut",
    "ao_shaping.utils.slm.slm_lut.stack_halves": "functions/slm-lut",
    "ao_shaping.utils.slm.slm_lut.invert_depth_scan": "functions/slm-lut",
    "ao_shaping.utils.slm.slm_lut.invert_offset_scan": "functions/slm-lut",
    "ao_shaping.utils.slm.slm_lut.save_lut": "functions/slm-lut",
    "ao_shaping.runners.runner_common.patch_sim_square_shaping": "functions/runner-common",
    "ao_shaping.runners.runner_common.parse_point": "functions/runner-common",
    "ao_shaping.runners.runner_common.build_gs_refine_config": "functions/runner-common",
    "ao_shaping.runners.runner_common.build_model_in_loop_config": "functions/runner-common",
    "ao_shaping.runners.runner_common.save_gs_refine_results": "functions/runner-common",
    "ao_shaping.runners.runner_common.save_model_in_loop_results": "functions/runner-common",
}
qual2slug.update(ALIAS)

def slug_of(qual):
    if qual in qual2slug:
        return qual2slug[qual]
    return None

def wiki_link(qual, label=None):
    s = slug_of(qual)
    if s:
        lab = label or qual.split(".")[-1]
        return f"[[{s}|`{lab}`]]"
    return f"`{qual}`"

def hw_class(qual):
    if "drivers.dm" in qual:
        return "DM 变形镜"
    if "drivers.slm" in qual:
        return "SLM 空间光调制器"
    if "drivers.ccd" in qual:
        return "CCD 相机"
    if "drivers.wfs" in qual:
        return "WFS 波前传感器"
    if "drivers.tm" in qual:
        return "时序控制"
    if "drivers.sim" in qual:
        return "仿真数字孪生"
    return None

# reachability per entry (functions only, depth-limited)
# Remap unresolved "ao_shaping.utils.X" re-exports to their real module location
_REMAP = {}
def _build_remap():
    resolved = [q for q, i in nodes.items() if i]
    for q in list(nodes):
        i = nodes[q]
        if i is None and q.startswith("ao_shaping.utils."):
            name = q.rsplit(".", 1)[-1]
            cands = [r for r in resolved if r.rsplit(".", 1)[-1] == name
                     and r.startswith("ao_shaping.utils.") and r != q]
            if len(set(cands)) == 1:
                _REMAP[q] = cands[0]
_build_remap()

def rmap(qual):
    return _REMAP.get(qual, qual)

def reachable(entry, max_depth=6):
    seen, frontier, depth = {entry}, [entry], 0
    hw, alg, io = set(), set(), set()
    while frontier and depth < max_depth:
        depth += 1
        nxt = []
        for u in frontier:
            for e in edges:
                if e["from"] != u:
                    continue
                v = rmap(e["to"])
                if v in seen or v not in nodes or nodes[v] is None:
                    continue
                seen.add(v)
                m = nodes[v]["module"]
                h = hw_class(v)
                if h:
                    hw.add(h)
                if "optimizer." in m or "algorithm." in m:
                    alg.add(v)
                if re.search(r"save|debug_artifacts|gen_date_dir|make_date_dir", v):
                    io.add(v)
                nxt.append(v)
        frontier = nxt
    return seen, hw, alg, io

def callees(qual, level1_only=True):
    out = defaultdict(int)
    for e in edges:
        if e["from"] == qual and e["to"] in nodes:
            out[rmap(e["to"])] += e["calls"]
    return sorted(out.items(), key=lambda kv: -kv[1])

def callers(qual):
    out = defaultdict(int)
    for e in edges:
        if e["to"] == qual and e["from"] in nodes:
            out[e["from"]] += e["calls"]
    return sorted(out.items(), key=lambda kv: -kv[1])

def fmt_args(info):
    a = list(info["args"])
    if len(a) > 12:
        s = ", ".join(a[:12]) + f", … (+{len(a)-12} 个)"
    else:
        s = ", ".join(a)
    return s or "—"

def fmt_doc(doc, maxlen=4000):
    doc = doc.strip()
    if len(doc) > maxlen:
        doc = doc[:maxlen] + "\n\n…（docstring 截断）"
    return doc

def signature_block(qual):
    info = nodes.get(qual)
    if not info:
        return None, None
    return f"{qual.split('.')[-1]}({fmt_args(info)})", info

# ---------- page builders ----------
def fm(title, tags, kind, source, extra=""):
    t = ",".join(tags)
    return (
        f"---\n"
        f"title: {title}\n"
        f"created: {DATE}\n"
        f"updated: {DATE}\n"
        f"type: {kind}\n"
        f"tags: [{t}]\n"
        f"source: {source}\n"
        f"{extra}"
        f"---\n\n"
    )

def entry_main_py():
    body = fm("main.py（仓库根入口）", ["entry", "cli"], "entry", "main.py")
    body += "## 职责\n\n仓库根入口文件，7 行。唯一工作：\n\n```python\nfrom ao_shaping.main import cli\n\nif __name__ == \"__main__\":\n    cli()\n```\n\n"
    body += "把 `python main.py …` 的全部调用转交给 [[entry/cli|`ao_shaping.main.cli`]]。" \
            "真正的命令注册、全局选项、调试模式切换都在 [[entry/cli]] 内完成。\n\n"
    body += "## 调用链\n\n```\npython main.py <command> [opts]\n  └─ cli()                     ao_shaping/main.py:82\n" \
            "     └─ click 分发 20 个 add_command 注册的 runner（见 [[entry/cli]]）\n```\n\n"
    body += "## 关联\n\n- [[entry/cli]]\n- 全部命令页见 [[index|目录]] 的「CLI 命令」一节\n"
    return "entry/main-py.md", body

def entry_cli():
    body = fm("cli（Click 根命令组）", ["entry", "cli"], "entry", "src/ao_shaping/main.py:82")
    body += "## 职责\n\nClick 根命令组 `@click.group()`，负责三件事：\n\n"
    body += "(1) 解析全局选项 `--dir`（数据保存根目录，默认 `data`）与 `--debug`（调试模式旗标）；" \
            "debug 判定为 `get_debug_mode() or bool(debug)`，即 `--debug` 旗标、环境变量 `DEBUG=1/true` 任一开启即生效" \
            "（详见 [[functions/debug-mode]]），开启后 loguru 切到 DEBUG 级日志。\n\n"
    body += "(2) 通过 `ctx.ensure_object(dict)` 把 `dir` 与 `debug` 存入 `ctx.obj`，供子命令经 `@click.pass_context` 读取。\n\n"
    body += (
        f"(3) 注册 {len(CMD_ORDER)} 个子命令（{len(CMD_ORDER) - len(_TOOL_ENTRIES)} 个 runner "
        f"符号 + {len(_TOOL_ENTRIES)} 个 tools 命令）。命令注册采用 **懒加载**："
        "`ao_shaping.runners.__init__` 用模块级 `__getattr__` + `_LAZY_RUNNERS` 映射，"
        "只有命令真正被调用时才 import 对应 runner 模块"
        "（避免 `python -m` 时的 sys.modules 警告，见 R-27）。\n\n"
    )
    body += "## 签名\n\n```python\ndef cli(ctx: click.Context, dir: str, debug: bool | None)\n```\n\n"
    body += f"## 注册的 {len(CMD_ORDER)} 个命令\n\n"
    body += "| 命令 | runner 入口函数 | 页面 |\n|---|---|---|\n"
    for name, desc in CMD_ORDER:
        qual = cmd_qualifier(name)
        body += f"| `{name}` | `{qual.split('.',1)[1]}` | [[commands/{name}]] |\n"
    body += "\n## 调用关系\n\n- 被 `main.py` 的 `__main__` 块调用\n- 调用 [[functions/debug-mode|`get_debug_mode`]]\n- 分发到各 [[commands/wf|命令入口]]（下表）\n"
    return "entry/cli.md", body

def cmd_page(name, desc):
    qual = cmd_qualifier(name)
    info = nodes.get(qual)
    if not info:
        return f"commands/{name}.md", f"## {name}\n\n（未在调用图中解析到入口函数 {qual}）\n"
    fileloc = f"{info['file']}:{info['lineno']}"
    tags = ["command"]
    # reachability for hardware/algorithm
    seen, hw, alg, io = reachable(qual)
    body = fm(f"{name}（{desc}）", tags, "command", fileloc)
    body += f"## 一句话\n\n{desc}\n\n"
    body += "## 入口函数\n\n"
    sig, _ = signature_block(qual)
    body += f"`{qual.split('.',1)[1]}`（`{fileloc}`）\n\n"
    body += f"```python\n{sig}\n```\n\n"
    body += "## docstring（原文节选）\n\n```\n" + fmt_doc(info["doc"]) + "\n```\n\n"
    # hardware / algorithm / outputs
    hwset = sorted(hw)
    algset = sorted(alg)
    body += "## 调用链分析（自动）\n\n"
    body += f"**硬件**：{ '、'.join(hwset) if hwset else '（无直接硬件，纯计算/仿真）' }\n\n"
    if algset:
        body += "**调用的优化器/算法**：\n\n"
        for a in algset:
            body += f"- {wiki_link(a)}\n"
        body += "\n"
    if io:
        body += f"**结果/调试落盘**：{ '、'.join(f'`{x.split(chr(46))[-1]}`' for x in sorted(io)) }\n\n"
    body += "## 直接调用（1 级，按调用次数）\n\n"
    direct = callees(qual)
    if direct:
        for t, n in direct:
            node = nodes.get(t)
            tag = "" if node else " *(未解析/外部)*"
            body += f"- {wiki_link(t)}{tag}\n"
    else:
        body += "- （无）\n"
    body += "\n## 谁调用它\n\n- [[entry/cli|`cli`]]（`cli.add_command` 注册为 `" + name + "`）\n"
    body += sim_block(qual)
    # cross links
    body += "\n## 相关页面\n\n- [[entry/cli]]\n"
    for a in algset[:4]:
        s = slug_of(a)
        if s:
            body += f"- {wiki_link(a)}\n"
    return f"commands/{name}.md", body

def definition_source(mod_name: str, sym_name: str) -> str:
    """Locate a symbol's definition line by AST, for the pages written by hand.

    ``func_page_recorder`` pinned ``utils/io/file.py:426`` as a literal in its front-matter,
    and the class it names has since moved to 469 -- 43 lines of drift that no check noticed,
    because the citation was prose rather than data. The asset cannot answer either: its
    ``Recorder`` node is unresolved (``None``), which is exactly why that page is hand-written.
    """
    rel = f"src/{mod_name.replace('.', '/')}.py"
    tree = ast.parse((REPO / rel).read_text(encoding="utf-8-sig"), filename=rel)
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.name == sym_name
        ):
            return f"{rel}:{node.lineno}"
    raise LookupError(f"{mod_name}.{sym_name}: no definition found in {rel}")


def func_page_recorder():
    body = (
        "---\n"
        "title: Recorder（优化过程记录器）\n"
        f"created: {DATE}\n"
        f"updated: {DATE}\n"
        "type: class\n"
        "tags: [io, recorder, utils]\n"
        f"source: {definition_source("ao_shaping.utils.io.file", "Recorder")}\n"
        "---\n\n"
        "# Recorder（优化过程记录器）\n\n"
        "## 职责\n\n"
        "逐行记录优化过程的运行记录器：每个优化器在每次迭代（或每个采样）后 `append`\n"
        "一行 dict（含 `mark` 标记列），最终汇总为 DataFrame，`get_best_*` 系列按指定列取最优行。"
        "各命令页\"结果/调试落盘\"段里的 `save_recorder_debug_artifacts` 就是消费 `Recorder`\n"
        "产物的落盘函数（见 [[functions/debug-artifacts]]）。\n\n"
        f"## 结构（`{definition_source('ao_shaping.utils.io.file', 'Recorder')}`）\n\n"
        "```python\nclass Recorder:\n"
        "    def __init__(self, mark: str = \"J\", mode: Literal[\"max\", \"min\"] = \"max\")\n"
        "    def append(self, record: dict)                    # 追加一行（自动补 _id）\n"
        "    def postprocess_feature(self, feature_name, func, column=\"\")  # 注册派生特征列\n"
        "    def _ensure_postprocess_applied(self, df) -> DataFrame\n```\n\n"
        "- `mark`：记录行的标记列名（断言 `mark in record`），用于区分不同优化器/阶段的行。\n"
        "- `mode`：`\"max\"`/`\"min\"`，决定 `get_best_*` 取最大还是最小。\n"
        "- `history`：`list[dict]`，每行一个 `_id` 自增。\n"
        "- `postprocess_feature`：注册 `func(row_dict) -> value`，落盘/导出时自动补算派生列，\n"
        "  失败时 `logger.warning` 并置 `None`（单列失败不中断整表）。\n\n"
        "## 谁使用它\n\n"
        "- 各优化器（[[functions/optimizer-rms-dm]]、[[functions/optimize-pib-wfless]]、\n"
        "  [[functions/optimize-slm-square]] 等）在迭代循环内 `rec.append({...})`。\n"
        "- [[functions/debug-artifacts|`save_recorder_debug_artifacts`]] 在 `--debug` 时把\n"
        "  `Recorder.history` 导出为 pkl/csv。\n\n"
        "## 相关页面\n\n"
        "- [[functions/debug-artifacts]]\n- [[functions/path-dirs]]\n- [[entry/cli]]\n"
    )
    return "functions/recorder.md", body

def func_page(slug):
    if slug == "functions/recorder":
        return func_page_recorder()
    qual, title, tags, kind = PAGES[slug]
    info = nodes.get(qual)
    if not info:
        # class or non-resolved
        return f"{slug.split('/')[0]}/{slug.split('/')[1]}.md", f"## {title}\n\n（占位）\n"
    fileloc = f"{info['file']}:{info['lineno']}"
    body = fm(title, tags, "function" if kind == "function" else "class", fileloc)
    body += "## 签名\n\n"
    sig, _ = signature_block(qual)
    ret = f" → `{info['returns']}`" if info["returns"] else ""
    body += f"```python\n{sig}{ret}\n```\n\n"
    if info["decorators"]:
        body += "## 装饰器\n\n```python\n" + "\n".join(info["decorators"]) + "\n```\n\n"
    body += "## docstring（原文）\n\n```\n" + fmt_doc(info["doc"], 6000) + "\n```\n\n"
    body += "## 直接调用（出边）\n\n"
    direct = callees(qual)
    if direct:
        for t, n in direct[:20]:
            tag = "" if nodes.get(t) else " *(未解析/外部)*"
            body += f"- {wiki_link(t)}{tag}\n"
    else:
        body += "- （叶子函数，无仓库内调用）\n"
    body += "\n## 谁调用它（入边）\n\n"
    inc = callers(qual)
    if inc:
        for t, n in inc[:20]:
            body += f"- {wiki_link(t)}\n"
    else:
        body += "- （无仓库内调用方；可能经反射/CLI/测试调用）\n"
    body += sim_block(qual)
    body += "\n## 相关页面\n\n- [[entry/cli]]\n"
    return f"{slug.split('/')[0]}/{slug.split('/')[1]}.md", body

def similar_functions_page():
    if not SIM:
        return None
    m = SIM["method"]
    body = fm(
        "相似函数（自动）— 函数间相似关联总览",
        ["similar", "relations"],
        "diagram",
        "scripts/call_graph/build_similarity.py",
    )
    body += "## 这是什么\n\n"
    body += '调用图回答「谁调用谁」，本页回答**「哪些函数是彼此的同族/同构实现」**：' \
            "不共享调用边、但结构或职责高度相近的函数对（near-duplicate / sibling " \
            "implementations）。这是独立于调用图的**第二条关系层**，用于：\n\n"
    body += "- 改一个函数时，提醒「还有几个同族函数需要同步改」（如两处 `optimize_pib`）；\n"
    body += "- 重构时识别可合并/可抽象的重复实现；\n"
    body += "- 新增功能时参考「同类功能在仓库里已有的几种做法」。\n\n"
    body += "## 判定方法（自动，可重跑）\n\n"
    body += "由 `scripts/call_graph/build_similarity.py` 对 `call_graph_final.json` 的全部已解析" \
            "函数两两打分（确定性权重，无随机、无模型）：\n\n"
    body += "| 信号 | 权重 | 含义 |\n|---|---|---|\n"
    body += (f"| 被调集合 Jaccard（乘 `min(1, shared/2)` 阻尼） | {m['weights']['jaccard_callees']} | "
             "两个函数调用结构越像，越可能是同族实现 |\n")
    body += (f"| 名称相似度 | {m['weights']['name_similarity']} | "
             "同名（如两处 `optimize_pib`）计 1；通用 CLI 入口名 `run` 计 0（避免假相似） |\n")
    body += (f"| 同文件 / 同模块 | {m['weights']['same_file_or_module']} | "
             "同文件 1.0、同模块 0.5 |\n")
    body += (f"| 调用方集合 Jaccard | {m['weights']['jaccard_callers']} | "
             "服务对象越像，越可能是同族 |\n")
    body += (f"| 共享被调 ≥5 奖励 | {m['weights']['many_shared_callees_bonus']} | "
             "调用结构高度重叠的额外加分 |\n\n")
    body += (f"得分 ≥ **{m['link_threshold']}** 记为**相似对**（写入各函数页「相似函数（自动）」段）；"
             f"≥ **{m['cluster_threshold']}** 的相似对合并成**相似簇**（下表）。"
             "数据文件：`raw/assets/similar_functions.json`（每对含逐维得分与理由）。"
             "重跑 `python3 scripts/call_graph/build_similarity.py` 即可刷新。\n\n")
    body += "## 相似簇（≥ 簇阈值，自动）\n\n"
    if SIM["clusters"]:
        body += "| 簇 | 成员 |\n|---|---|\n"
        for c in SIM["clusters"]:
            body += "| " + " — ".join(wiki_link(x) for x in c) + " |\n"
    else:
        body += "（本次运行未发现达到簇阈值的相似簇。）\n"
    body += "\n## 全部相似对（≥ 链接阈值，按得分降序）\n\n"
    body += "| 得分 | 函数 A | 函数 B | 判定理由 |\n|---|---|---|---|\n"
    for p in SIM["pairs"]:
        body += (f"| {p['score']:.3f} | {wiki_link(p['a'])} | {wiki_link(p['b'])} | "
                 f"{p['reason']}（同调 {p['jaccard_callees']:.2f} / 共享 {p['shared_callees']} 个被调） |\n")
    body += "\n## 使用建议\n\n"
    body += ("1. **改一个函数前**先看本页与它的「相似函数（自动）」段——同族函数通常需要同步修改，"
             "否则会出现行为漂移（仓库里 `optimize_pib` 双实现是典型例子）。\n")
    body += ("2. 得分 0.30–0.45 的多数是「同一思路在不同设备域/不同算法上的平行实现」"
             "（DM 族 vs SLM 族），**不是**复制粘贴——合并前需确认硬件语义是否可抽象。\n")
    body += ("3. 新代码改动后重跑 `build_call_graph.py` + `build_similarity.py` + `gen_wiki_pages.py`，"
             "本页与全部函数页自动刷新。\n")
    return "relations/similar-functions.md", body

def build():
    for d in ("entry", "commands", "functions", "relations", "assets"):
        (BASE / d).mkdir(parents=True, exist_ok=True)
    (BASE / "raw").mkdir(exist_ok=True)
    (BASE / "raw/assets").mkdir(exist_ok=True)

    # copy raw artifacts
    (BASE / "raw/assets/call_graph.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=1))

    written = []
    # entry pages
    f, b = entry_main_py()
    (BASE / f).write_text(b)
    written.append(f)
    f, b = entry_cli()
    (BASE / f).write_text(b)
    written.append(f)
    # command pages
    for name, desc in CMD_ORDER:
        f, b = cmd_page(name, desc)
        (BASE / f).write_text(b)
        written.append(f)
    # function pages
    for slug in PAGES:
        if slug.startswith(("entry/", "commands/")):
            continue
        f, b = func_page(slug)
        (BASE / f).write_text(b)
        written.append(f)
    # relations overview page (similar functions)
    page = similar_functions_page()
    if page:
        f, b = page
        (BASE / f).write_text(b)
        written.append(f)
    return written

if __name__ == "__main__":
    w = build()
    logger.info(f"generated {len(w)} pages")
    for x in sorted(w):
        logger.info(f" {x}")
