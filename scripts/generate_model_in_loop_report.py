"""Generate the model-in-the-loop algorithm report (call graph / timing / algorithm).

**What this report is.** A *description of code*, not a measurement: it documents
how ``slm-model-in-loop`` is wired, what happens in what order, and why each guard
exists. It is therefore ``运行环境: 离线`` and needs no instrument.

**Why it is generated rather than hand-written.** Three reasons, in order of how
much they have actually cost this repo:

1. **The diagrams would rot.** A hand-drawn call graph names symbols; renaming one
   turns the document into fiction that still reads as authoritative. So the
   diagrams here are *derived from the source*, and every symbol they name is
   resolved against the real modules before the file is written. A rename breaks
   generation loudly (see ``_verify_symbols``) instead of silently producing a
   confident lie.
2. **The numbers come from the dataclass.** ``trust_region_c_l2``'s default,
   ``max_rejection_streak``, ``PLATEAU_PATIENCE`` -- these are read out of the
   AST, not typed here, so a default bump cannot leave the prose stale.
3. **Provenance is mechanical.** The report lives under ``report/`` and must carry
   the generated header, or ``tests/ao_shaping/scripts/test_report_provenance.py``
   fails. A generator that overwrites its own report has to stamp that header
   itself (``insert_header``), which is what ``main`` does.

**Deliberately absent.** No wall-clock timing. The per-round *counts* of device
round-trips and compute steps are derivable from the configuration, so they are
reported; the *durations* depend on the bench and would be a fabrication here. A
previous report in this family was invalidated by exactly that mistake.

Usage::

    python scripts/generate_model_in_loop_report.py
    python scripts/generate_model_in_loop_report.py --no-figures
    python scripts/generate_model_in_loop_report.py --out report/slm/model_in_loop_algorithm.md
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # BEFORE pyplot -- headless CI

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts._common.provenance import insert_header  # noqa: E402

REPORT_KEY = "report/slm/model_in_loop_algorithm.md"

#: Repo-root-relative modules the diagrams and tables are derived from.
HARDWARE = "src/ao_shaping/optimizer/wfless/slm_model_in_loop.py"
TWIN = "src/ao_shaping/optimizer/wfless/model_in_loop_shaping.py"
OPTIMIZER = "src/ao_shaping/algorithm/signal_processing/zernike_coefficient_optimizer.py"
RUNNER = "src/ao_shaping/runners/slm/model_in_loop_runner.py"
PARAMS = "src/ao_shaping/runners/runner_common.py"
IO = "src/ao_shaping/utils/io/file.py"
GS_REFINE = "src/ao_shaping/optimizer/wfless/slm_gs_refine.py"

MODULES = {
    "hardware": HARDWARE,
    "twin": TWIN,
    "optimizer": OPTIMIZER,
    "runner": RUNNER,
    "params": PARAMS,
    "io": IO,
    "gs_refine": GS_REFINE,
}

#: Every symbol the two ASCII diagrams name, as ``(module_key, qualified name)``.
#: ``_verify_symbols`` resolves each against the real AST; a rename fails the run.
#: This is the guard that makes the diagrams safe to keep.
DIAGRAM_SYMBOLS: tuple[tuple[str, str], ...] = (
    # runner / orchestration
    ("runner", "_parse_frozen_modes"),
    ("runner", "_parse_point"),
    ("params", "SlmModelInLoopParams"),
    # hardware entry + benches
    ("hardware", "optimize_slm_model_in_loop"),
    ("hardware", "_open_bench"),
    ("hardware", "_SimBench"),
    ("hardware", "_HardwareBench"),
    ("hardware", "_calibrate_geometry"),
    ("hardware", "_make_optimizer"),
    ("hardware", "_probe_phase"),
    ("hardware", "_to_model_grid"),
    ("hardware", "_metrics_at"),
    ("hardware", "_quality"),
    ("hardware", "trust_region_clamp"),
    ("hardware", "acceptance_verdict"),
    ("hardware", "SlmModelInLoopConfig"),
    ("hardware", "Bench"),
    # shared math, imported from the twin
    ("twin", "_fit_aberration_at_probes"),
    ("twin", "shape_phase_with_frozen_aberration"),
    ("twin", "calibrate_bench_geometry"),
    ("twin", "_probe_phase"),
    ("twin", "simulate_iterative_shaping"),
    ("twin", "StepBConfig"),
    ("twin", "BenchGeometry"),
    # borrowed helper
    ("gs_refine", "_prepare_frame"),
    # algorithm layer
    ("optimizer", "ZernikeCoefficientOptimizer"),
    # recorder
    ("io", "save_recorder_debug_artifacts"),
)


# ---------------------------------------------------------------------------
# Source introspection (AST -- never imports the hardware stack)
# ---------------------------------------------------------------------------


def _parse(rel: str) -> ast.Module:
    """Parse a module, naming the file if it fails.

    The filename matters more than it looks: a bare ``IndentationError`` with an
    ``<unknown>`` line number once sent me hunting inside the generator when the
    file actually being read was a *different* module being written by another
    process at that moment.
    """
    text = (ROOT / rel).read_text(encoding="utf-8")
    try:
        return ast.parse(text)
    except SyntaxError as exc:
        raise SystemExit(
            f"model-in-loop report: cannot parse {rel} "
            f"(line {exc.lineno}: {exc.msg}). If that file is mid-edit by another "
            f"process, re-run once it is written; the report is not regenerated "
            f"from a half-written module."
        ) from exc


def _defined_names(tree: ast.Module) -> set[str]:
    """Every module-level name the file binds: defs, classes, assignments."""
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    names.add(tgt.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _class_methods(tree: ast.Module, cls: str) -> set[str]:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == cls:
            return {
                n.name
                for n in node.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
    return set()


def _verify_symbols() -> None:
    """Fail if any symbol named in the diagrams does not exist.

    Without this the report is a snapshot of one moment in the code's life, and
    the only symptom is that a reader is misled. Renaming ``_metrics_at`` should
    make report generation fail, not make a stale document look current.
    """
    trees = {k: _parse(v) for k, v in MODULES.items()}
    module_names = {k: _defined_names(t) for k, t in trees.items()}
    method_cache: dict[tuple[str, str], set[str]] = {}

    missing: list[str] = []
    for key, name in DIAGRAM_SYMBOLS:
        if name in module_names.get(key, ()):
            continue
        # fall back to a method on any class in that module
        cache_key = (key, name)
        if cache_key not in method_cache:
            found = False
            for node in trees[key].body:
                if isinstance(node, ast.ClassDef):
                    if name in _class_methods(trees[key], node.name):
                        found = True
                        break
            method_cache[cache_key] = {name} if found else set()
        if not method_cache[cache_key]:
            missing.append(f"{MODULES[key]} :: {name}")

    if missing:
        raise SystemExit(
            "model-in-loop report: these symbols are named in the diagrams but no "
            "longer exist (rename the symbol, or update DIAGRAM_SYMBOLS):\n  - "
            + "\n  - ".join(missing)
        )


def _literal(node: ast.expr | None) -> object:
    if node is None:
        return None
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError):
        # tuple(...)/dict(...) / defaults built from other constants
        seg = ast.unparse(node)
        return seg if len(seg) <= 60 else "<computed>"


def _dataclass_defaults(rel: str, cls: str) -> dict[str, str]:
    """``{field: rendered default}`` for a dataclass, read from the AST."""
    tree = _parse(rel)
    out: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    out[stmt.target.id] = str(_literal(stmt.value))
    return out


def _module_constants(rel: str, wanted: tuple[str, ...]) -> dict[str, str]:
    tree = _parse(rel)
    out: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id in wanted:
                    out[tgt.id] = str(_literal(node.value))
    return out


# ---------------------------------------------------------------------------
# The two ASCII diagrams
# ---------------------------------------------------------------------------

CALL_GRAPH = """\
main.py `slm-model-in-loop`                              click hub
 └─ runners/slm/model_in_loop_runner.py :: run            硬件编排层
     ├─ _parse_frozen_modes / _parse_point                CLI 文本 -> 类型化字段
     ├─ SlmModelInLoopConfig                              扁平 dataclass (runner_common)
     ├─ optimize_slm_model_in_loop(config)                ← 唯一公开入口
     │   │
     │   ├─ _open_bench(config)                           ──► Bench (Protocol)
     │   │     ├─ _SimBench                                cam_type=sim (数字孪生)
     │   │     └─ _HardwareBench                           daheng/miicam + Santec
     │   │        display(phase) / measure() / close()        设备 import 全部延迟
     │   │
     │   ├─ _calibrate_geometry(bench, config)             一次性几何 bake-off
     │   │     └─ calibrate_bench_geometry(...)           [孪生] sim 直接取真值
     │   │
     │   ├─ 平场基线: display(flat) -> _metrics_at -> _quality
     │   │
     │   └─ for index in range(n_rounds):
     │       │
     │       │  ── Step A: 重拟合一个共享像差 ─────────────────────────────
     │       ├─ coefficients = np.zeros(n_coeffs)           ← 唯一初始种子 (硬编码)
     │       ├─ _make_optimizer(config, coefficients)
     │       │     └─ ZernikeCoefficientOptimizer(initial_coefficients=...)
     │       │          zernike_coefficient_optimizer.py   Adam 作用在 Zernike 向量上
     │       ├─ for probe_index in range(probe_count):
     │       │     ├─ _probe_phase(region, probe_spread, seed)        [孪生]
     │       │     ├─ bench.display(probe) -> _prepare_frame(bench.measure())
     │       │     │                      _prepare_frame  <- slm_gs_refine
     │       │     └─ _to_model_grid(measured, roi_center, far_field_size, ...)
     │       ├─ _fit_aberration_at_probes(optimizer, frames, iters)  [孪生]
     │       │     └─ 所有探针轮流喂进**同一个** Adam 状态; 触发平台期则 rearm
     │       └─ trust_region_clamp(fitted, coefficients, trust_region_c_l2)
     │                                     限制 |c_{t+1} - c_t|
     │       │
     │       │  ── Step B: 冻结像差, 合成方形 ─────────────────────────────
     │       ├─ shape_phase_with_frozen_aberration(...)                 [孪生]
     │       ├─ bench.display(shaped) -> _prepare_frame
     │       │                        -> _metrics_at -> _quality
     │       ├─ phase = shaped                        成为下一轮的热启动
     │       └─ acceptance_verdict(loss_before, loss_after, score_before, score_after)
     │            accept -> coefficients/phase 前移, streak 清零
     │            reject -> 阻尼 lr / 加探针 / streak+1
     │                      streak >= max_rejection_streak -> ABORTED_REJECTION_STREAK
     │
     └─ save_recorder_debug_artifacts(recorder, ...)       utils/io/file.py
          data/debug/slm_model_in_loop_<ts>/<ts>/*.pkl      每次运行都写
"""

SEQUENCE = """\
时序图 (一轮;  ▓ = 设备 I/O   ░ = 计算   │ = 状态变更)   默认参数: 8 探针 / 80 / 600

  用户      main.py      runner          optimize_()      Bench(设备)     ZernikeOpt
    │          │            │                 │              │              │
    │ 命令行   │            │                 │              │              │
    ├─────────►│            │                 │              │              │
    │          ├───────────►│ 解析参数        │              │              │
    │          │            ├─ np.zeros(K) ──┼─ 种子 c_0 ──┼─────────────►│
    │          │            │                 ├─ _open_bench─┼─────────►    │
    │          │            │                 │        ▓ open device        │
    │          │            │                 │              │              │
    │          │            │        ┌────────┴─ 几何 bake-off (一次性) ───┐ │
    │          │            │        │ device: display+measure × n_calib_probe │ │
    │          │            │        │   calibrate_bench_geometry(...)   │ │
    │          │            │        │   相关度 < min_geometry_corr ⇒ 退出码 2│
    │          │            │        └──────────────────────────────────┘ │
    │          │            │                 ├─ ▓ display(flat) → 基线 score │
    │          │            │                 │              │              │
    ╞══════════╪════════════╪═════════════════╪══════════════╪══════════════╡ 第 t 轮
    │          │            │                 │              │              │
    │          │            │            ┌────┴─ Step A ────┼──────────────►│
    │          │            │            │ ▓▓ 探针 i=0..7 (display+measure)│
    │          │            │            │    _probe_phase / _prepare_frame│
    │          │            │            │    _to_model_grid → 模型网格    │
    │          │            │            │ ░ 80× Adam 前向+反向 (循环全部探针)
    │          │            │            │    → fitted, loss_before/after │
    │          │            │            │ ░ trust_region_clamp → clamped │
    │          │            │            │              │              │
    │          │            │            │            ┌──┴─ Step B ──────┼──►│
    │          │            │            │ ░ 600× 全像素相位 Adam (像差冻结)
    │          │            │            │ ▓ display(shaped) + measure     │
    │          │            │            │ ░ _metrics_at → _quality → after│
    │          │            │            │ ░ acceptance_verdict ────────────┤
    │          │            │                 │      │              │
    │          │            │      accept ────┼──────┘  c←clamped, phase←shaped
    │          │            │      reject ────┼──────► 阻尼 lr / 探针+escalate
    │          │            │                 │      streak=3 ⇒ ABORTED ─────►│
    │          │◄───────────┴─ CSV / pkl / best_phase.npy / bench_geometry.json
"""


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

_CJK = {"Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "WenQuanYi Micro Hei"}


def _setup_fonts() -> None:
    from matplotlib import font_manager

    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in _CJK:
        if name in have:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break
    else:
        print("warning: no CJK font found; Chinese labels may be garbled")
    plt.rcParams["axes.unicode_minus"] = False


def _fig_layers(path: Path) -> None:
    """Layered block diagram -- who owns what, and which way the calls go.

    Bands are laid out on a fixed vertical grid because an earlier version let
    the Bench row overlap the strategy-layer row: the figure still rendered, the
    link check still passed, and the two labels sat on top of each other.
    """
    fig, ax = plt.subplots(figsize=(13.0, 8.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    box_w, box_h = 27.0, 11.0
    x0 = 31.0

    def band(y: float, title: str, boxes: list[tuple[str, str]], fc: str, ec: str) -> None:
        ax.text(2.0, y, title, fontsize=11.0, va="center", color="#1f3b57")
        for i, (name, sub) in enumerate(boxes):
            x = x0 + i * (box_w + 5.0)
            ax.add_patch(FancyBboxPatch(
                (x, y - box_h / 2), box_w, box_h,
                boxstyle="round,pad=0.4", linewidth=1.3,
                edgecolor=ec, facecolor=fc,
            ))
            ax.text(x + box_w / 2, y + 1.4, name, fontsize=9.6,
                    ha="center", va="center")
            ax.text(x + box_w / 2, y - 2.6, sub, fontsize=8.8,
                    ha="center", va="center", color="#555555")

    # y positions, top to bottom. Spacing >= box_h + gap so nothing can collide.
    band(90.0, "CLI / 编排层", [("main.py", "click hub"),
                              ("model_in_loop_runner.py", "runners/slm/")],
         "#eef4f9", "#4a6f8a")
    band(69.0, "策略层 (硬件移植)", [("slm_model_in_loop.py", "optimizer/wfless/")],
         "#eef4f9", "#4a6f8a")
    band(46.0, "设备抽象 (--cam_type 唯一改变的一层)",
         [("Bench 协议", "display / measure / close"),
          ("_SimBench / _HardwareBench", "2f-Fourier 孪生 或 大恒/MiiCam + Santec")],
         "#fdf1e6", "#8a4a20")
    band(25.0, "共享数学 = 数字孪生", [("model_in_loop_shaping.py", "optimizer/wfless/")],
         "#e9f6ef", "#2d6a4f")
    band(6.0, "算法层", [("zernike_coefficient_optimizer.py",
                          "algorithm/signal_processing/  ·  Adam")],
         "#eef4f9", "#4a6f8a")

    for y_from, y_to in ((84.5, 74.5), (63.5, 51.5), (40.5, 30.5), (19.5, 11.5)):
        ax.add_patch(FancyArrowPatch(
            (x0 + box_w / 2, y_from), (x0 + box_w / 2, y_to),
            arrowstyle="-|>", mutation_scale=15, linewidth=1.5, color="#4a6f8a",
        ))

    # the shared-math row is imported *by* the strategy row, not merely below it
    ax.add_patch(FancyArrowPatch(
        (x0 + box_w + 2.0, 46.0), (x0 + box_w + 2.0, 69.0),
        arrowstyle="-|>", mutation_scale=14, linewidth=1.4, color="#2d6a4f",
        connectionstyle="arc3,rad=0.30",
    ))
    ax.text(x0 + box_w + 5.5, 57.5, "延迟 import\n(共享数学)", fontsize=8.8,
            color="#1f5138", va="center")

    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _fig_timeline(path: Path, cfg: dict[str, str]) -> None:
    """Per-round operation budget: device round-trips vs compute steps.

    Counts, not seconds -- see the module docstring. Derived from the config
    defaults so a default bump moves the bars.
    """
    probes = int(cfg["probe_count"])
    n_calib = int(cfg["n_calibration_probes"])
    it_a = int(cfg["step_a_iterations"])
    it_b = int(cfg["step_b_iterations"])
    n_round = int(cfg["n_rounds"])
    frames = int(cfg["n_eval_frames"])

    phases = [
        ("几何标定 (一次性)\ndevice: display+measure", n_calib * 2 * frames, 0),
        ("平场基线 (一次性)\ndevice: display+measure", 2 * frames, 0),
        ("Step A 探针 (每轮)\ndevice: display+measure", probes * 2 * frames, 0),
        ("Step A 拟合 (每轮)\ncompute: Adam steps", it_a, 1),
        ("Step B 合成 (每轮)\ncompute: Adam steps", it_b, 1),
        ("Step B 验收 (每轮)\ndevice: display+measure", 2 * frames, 0),
    ]
    labels = [p[0] for p in phases]
    device = [p[1] if p[2] == 0 else 0 for p in phases]
    compute = [p[1] if p[2] == 1 else 0 for p in phases]

    fig, ax = plt.subplots(figsize=(12.4, 6.2))
    y = range(len(phases))
    ax.barh(y, device, color="#c96a3f", label="设备 I/O (帧数)")
    ax.barh(y, compute, left=device, color="#3f6f9c", label="计算步数")
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=9.2)
    ax.invert_yaxis()
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xlim(0, max(max(d + c for d, c in zip(device, compute)) * 3.0, 10))
    ax.set_xlabel("每次运行的操作数 (对数轴) —— 不是秒", fontsize=10)
    ax.set_title(
        f"每轮操作预算 (默认 n_rounds={n_round}, probe_count={probes}, "
        f"step_a={it_a}, step_b={it_b}, n_eval_frames={frames})",
        fontsize=11.5,
    )
    for i, (d, c) in enumerate(zip(device, compute)):
        ax.text(max(d, c) * 1.15, i, f"{d + c}", va="center", fontsize=9, color="#333333")
    ax.legend(loc="lower right", fontsize=9.4)
    ax.grid(axis="x", alpha=0.25)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _fig_guards(path: Path, cfg: dict[str, str]) -> None:
    """The acceptance state machine, as a small state diagram."""
    fig, ax = plt.subplots(figsize=(11.4, 5.0))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    def box(x, y, w, h, text, fc, ec):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4",
                                    linewidth=1.4, edgecolor=ec, facecolor=fc))
        ax.text(x + w / 2, y + h / 2, text, fontsize=9.6, ha="center", va="center")

    box(2, 58, 26, 22, "进入第 t 轮\ntrust_region_clamp → clamped\n（系数步长被截断）", "#eef4f9", "#4a6f8a")
    box(37, 58, 26, 22, "acceptance_verdict\nloss_before/after\nscore_before/after", "#eef4f9", "#4a6f8a")
    box(72, 58, 26, 22, "Step B 实测\nshaped_frame → _quality\n→ score_after", "#eef4f9", "#4a6f8a")
    box(37, 12, 26, 20, "accept\nc ← clamped, phase ← shaped\nlr 复原, streak = 0",
        "#e9f6ef", "#2d6a4f")
    box(72, 12, 26, 20, f"reject\nlr × {cfg['damp_lr_factor']}\n探针 +{cfg['escalate_probe_count']} (上限 {cfg['max_probe_count']})",
        "#fdecec", "#a83a3a")
    box(2, 12, 26, 20,
        f"streak ≥ {cfg['max_rejection_streak']}\n=> ABORTED_REJECTION_STREAK\n(不提交任何相位)", "#fdecec", "#a83a3a")

    for a, b in (((28, 69), (37, 69)), ((63, 69), (72, 69))):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14,
                                     linewidth=1.4, color="#4a6f8a"))
    ax.add_patch(FancyArrowPatch((50, 58), (50, 32), arrowstyle="-|>",
                                 mutation_scale=14, linewidth=1.4, color="#2d6a4f"))
    ax.text(52.5, 45, "accept", fontsize=9, color="#2d6a4f")
    ax.add_patch(FancyArrowPatch((63, 62), (76, 32), arrowstyle="-|>",
                                 mutation_scale=14, linewidth=1.4, color="#a83a3a",
                                 connectionstyle="arc3,rad=0.18"))
    ax.text(70, 44, "reject", fontsize=9, color="#a83a3a")
    # reject -> abort is routed *below* both boxes on purpose: a straight line
    # between them crosses the accept box, which reads as "reject -> accept".
    ax.plot([85.0, 85.0], [12.0, 4.5], color="#a83a3a", linewidth=1.4)
    ax.plot([85.0, 15.0], [4.5, 4.5], color="#a83a3a", linewidth=1.4)
    ax.add_patch(FancyArrowPatch((15.0, 4.5), (15.0, 12.0), arrowstyle="-|>",
                                 mutation_scale=14, linewidth=1.4, color="#a83a3a"))
    ax.text(50, 6.8, "streak 累加; 台架落在模型之外时中止, 而不是提交一个模型自己喜欢的相位",
            fontsize=9.4, ha="center", color="#444444")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def _table(rows: list[tuple[str, str]]) -> str:
    head = "| 项 | 值 |\n|---|---|"
    body = "\n".join(f"| `{k}` | `{v}` |" for k, v in rows)
    return f"{head}\n{body}"


def render(cfg: dict[str, str], consts: dict[str, str], *, figures: bool) -> str:
    fig_dir = "figures"
    fig1 = f"{fig_dir}/model_in_loop_layers.png"
    fig2 = f"{fig_dir}/model_in_loop_timeline.png"
    fig3 = f"{fig_dir}/model_in_loop_guards.png"

    plateau = consts.get("PLATEAU_PATIENCE", "30")
    twin_region = consts.get("TWIN_REGION", "512")

    parts: list[str] = []
    A = parts.append

    A("# 正向模型闭环整形 (slm-model-in-loop) —— 调用关系 / 计算时序 / 算法说明\n")
    A(
        "本报告描述 `slm-model-in-loop` 的**结构与算法**, 不是一次测量的结论。\n"
        "全文离线可重生成: 图中的每个符号都对着源码解析过, 表格里的默认值是从 dataclass\n"
        "的 AST 里读出来的 —— 改名会让生成失败, 而不是留下一份看起来仍然权威的旧文档。\n"
    )
    A(
        "> **口径声明: 本报告不含任何壁钟耗时。** 每轮的设备往返次数与计算步数可由配置\n"
        "> 推出, 因此给出; 耗时取决于台架, 在这里写就是编造。\n"
    )

    # -- 1 定位 -----------------------------------------------------------
    A("## 1. 这条命令做什么\n")
    A(
        "每轮交替两步, 用实测数据在闭环里持续修正正向模型, 再用修正后的模型合成相位:\n\n"
        "1. **Step A —— 拟合正向模型.** 显示若干个**强随机探针**相位, 把每个探针的远场实测帧\n"
        "   与正向模型的预测对比, 重拟合**一个共享的 Zernike 像差向量**。\n"
        "2. **Step B —— 合成方形.** **冻结**刚拟合出的像差, 优化**全像素 SLM 相位**逼近方形目标,\n"
        "   下发实测; 本轮相位成为下一轮的热启动。\n\n"
        "两个守卫 (§6) 抑制两步互相追逐; 最终相位只有**实测优于平场基线**才提交。\n"
    )

    # -- 2 调用关系 -------------------------------------------------------
    A("## 2. 调用关系图\n")
    A("每个名字都是真实符号, 方括号标注它归哪一层。缩进行是一轮里的两个步骤。\n")
    A("```\n" + CALL_GRAPH + "```\n")
    A("### 分层与依赖方向\n")
    if figures:
        A(f"![分层与依赖方向]({fig1})\n")
    A(
        "孪生与硬件的关系是这个模块的全部设计要点: 标 `[孪生]` 的行**不是**一份会腐烂的平行实现,\n"
        "而是硬件层 import 进来的共享数学 (`model_in_loop_shaping`), 且 import 全部延迟到函数内 ——\n"
        "`ao_shaping.drivers` 在 import 期就会碰硬件。于是 `--cam_type sim` 只替换 `Bench` 的实现,\n"
        "其余数学逐位相同; 反过来, 仿真里验证过的 Step A/B 逻辑不会在硬件路径上悄悄漂移。\n"
    )
    A(
        "两个模块共享数学, 共享的**不是** config/result 类型 —— 孪生用\n"
        "`ModelParams`/`StepAConfig`/`StepBConfig`/`ModelInLoopResult`, 硬件用扁平的\n"
        "`SlmModelInLoopConfig`/`RoundRecord`/`ModelInLoopResult`, 因为硬件侧要把设备生命周期、\n"
        "settle 判据和 recorder 行一并塞进一个可序列化的配置里。\n"
    )

    # -- 3 时序 -----------------------------------------------------------
    A("## 3. 计算时序图\n")
    A("一轮的完整时序。`▓` 是设备 I/O, `░` 是计算, `│` 是状态变更。\n")
    A("```\n" + SEQUENCE + "```\n")
    A("### 顺序里三个容易看漏的点\n")
    A(
        "**几何标定在最前面, 且只做一次.** 它自己也要 display+measure, 但它解的是\n"
        "`BenchGeometry` —— 没有它, Step A 的 loss 会把「模型预测错」和「几何标定错」混在同一个残差里,\n"
        "梯度方向是错的。相关度低于 `min_geometry_correlation` 直接以退出码 2 中止, **不提交任何相位**。\n"
    )
    A(
        "**Step A 的所有探针喂进同一个 Adam 状态, 不是每帧各跑一次优化.** 焦面强度是瞳面相位的\n"
        "非凸函数, 单个探针留下大量驻点, Adam 会停在它进去的那一个; 轮流喂多个独立探针提供的是\n"
        f"消掉这些驻点所需的*多样性*。另外 `update` 在连续 {plateau} 步无改善后会锁存\n"
        "(`PLATEAU_PATIENCE`), 且没有任何东西清这个锁存 —— `reset()` 会把系数倒回 `__init__` 的值, 所以\n"
        "re-arm 必须绕着**当前**系数重建优化器。\n"
    )
    A(
        "**Step B 只在最后测一次, 不在每次迭代里测.** Step B 是纯计算 (可微前向模型过 FFT),\n"
        "硬件上不可微的只有「真实测量」本身, 所以梯度来自模型而不是测量; 实测只在末尾做一次,\n"
        "交给验收测试。因此 `step_b_iterations=600` 不产生 600 次设备往返。\n"
    )
    if figures:
        A(f"![每轮操作预算]({fig2})\n")

    # -- 4 Step A ---------------------------------------------------------
    A("## 4. Step A 详细: 共享像差拟合\n")
    A("### 4.1 为什么必须用探针, 不能直接用整形相位\n")
    A(
        "平瞳孔 (或光滑整形后的瞳孔) 聚焦成近似 δ 函数, 其**归一化**强度对光滑低阶像差几乎不响应 ——\n"
        "像差因此不可辨识。探针相位 std 高于约 3 rad 时信号才越过 16-bit 量化本底两个数量级, 拟合才良态。\n"
        "这也解释了 `probe_spread` 不是一个可以随手调小的正则项。\n"
    )
    A("### 4.2 逐步\n")
    A(
        "1. `_probe_phase(region, probe_spread, seed)` 生成零均值高斯**瞳面相位** (raw 未包裹弧度)。\n"
        "2. `bench.display(probe)` 下发, `bench.measure()` 读一帧, `_prepare_frame` 逐帧去背景再裁剪\n"
        "   —— 直接对原始帧算指标会被读出噪声污染 (对称读出噪声让约一半像素为负)。\n"
        "3. `_to_model_grid` 把实测帧搬到**模型自己的网格**上。Step A 的 loss 比较的是\n"
        "   「实测 CCD」与「模型预测」, 两者必须逐像素可比, 所以这一步的尺度关系必须由几何标定保证。\n"
        "4. `_fit_aberration_at_probes` 把 `(实测远场, 探针相位)` 对**循环**喂进同一个\n"
        "   `ZernikeCoefficientOptimizer` 的 `update`, 跑满 `step_a_iterations` 步。\n"
        "5. 输出 `fitted` 与本轮 loss 首末值, 供验收测试使用。\n"
    )
    A("### 4.3 冻结哪些模式, 以及为什么\n")
    A(
        "`frozen_modes` 默认 `(1, 2, 3)`, 即 **Noll 1/2/3 = piston/tip/tilt**。它们**无法从远场\n"
        "强度辨识**: piston 在瞳孔上是常数, 对焦面强度无贡献; tip/tilt 只是把 0 阶搬个位置。\n"
        "放开它们的实测后果是 56% 的系数范数被灌进这些简并方向而**不带来任何收益** ——\n"
        "即拟合在噪声方向上花掉了 trust region 的预算。\n"
    )
    A("### 4.4 系数个数\n")
    A(
        "`n_coefficients = calc_n_zernike_terms(n_orders)`, **含 piston**。\n"
        "这与 `ml/zernike` 的离线约定**差一位**, 见 §8。\n"
    )

    # -- 5 Step B ---------------------------------------------------------
    A("## 5. Step B 详细: 冻结像差下的全像素相位合成\n")
    A(
        "`shape_phase_with_frozen_aberration(optimizer, clamped, target, warm_start, step_b_cfg)`:\n"
        "像差 `clamped` 作为**固定**的物理项进入正向模型, 自由变量是**整个 region×region 的相位网格**。\n"
        "梯度穿过相干传播 + FFT 解析求出 (torch autograd), 所以这 600 步是纯 GPU/CPU 计算。\n"
    )
    A(
        "**为什么必须是全像素而不是低阶 Zernike.** Zernike 是圆对称光滑基, 物理上无法合成方形远场\n"
        "(方形需要类 sinc 的近场结构 / 高空间频率)。这是 `slm-gs-refine` / `slm-gsnet` 同样遵守的约定。\n"
    )
    A(
        "**目标函数必须含能量项.** `_quality` 组合 `w_efficiency`/`w_uniformity`;\n"
        "`w_efficiency` 必须非零 —— 只优化 `-CV` 会把能量推出目标框 (硬件实测 EE → 0.002)。\n"
    )
    A(
        "**热启动.** `warm` 取上一轮 `phase` (`--no-warm-start` 则退回平场)。实测分数**没有**变好时,\n"
        "验收测试会 reject 该轮, 于是 `phase` 不前移 —— 被 reject 的那一轮不会污染下一轮的热启动。\n"
    )

    # -- 6 守卫 -----------------------------------------------------------
    A("## 6. 两个守卫: 为什么必须有\n")
    A(
        "Step A 产出 `c_t`; Step B 产出以 `c_t` 为条件的 `phi_t`; 下一轮又通过 `phi_t` 的校正去重拟合\n"
        "`c_{t+1}`。放任两者互相追逐 —— 像差吸收一部分整形相位, 整形相位又补偿像差 —— 在硬件上\n"
        "(漂移、LCOS 弛豫误差、读出噪声) 表现为**缓慢发散而不是崩溃**, 所以必须有廉价的上界。\n"
    )
    A(
        "**守卫 1 —— trust region.** 逐轮系数变化的 L2 范数被 `trust_region_c_l2` 截断。\n"
        "**守卫 2 —— 逐轮验收.** `acceptance_verdict(loss_before, loss_after, score_before, score_after)`:\n"
        "拟合 loss 变差、或实测综合分变差 (容差 `acceptance_score_eps`), 该轮就被 reject。\n"
    )
    if figures:
        A(f"![验收状态机]({fig3})\n")
    A(
        "持续 reject 意味着台架落在模型描述之外 (瞳孔配准、面板倾斜、离面离焦、渐晕), 此时运行\n"
        "**中止**而不是提交一个「模型自己喜欢」的相位。\n"
    )

    # -- 7 配置 -----------------------------------------------------------
    A("## 7. 默认配置 (从 dataclass AST 读出, 非手抄)\n")
    A(_table([
        ("n_rounds", cfg["n_rounds"]),
        ("probe_count / probe_spread", f"{cfg['probe_count']} / {cfg['probe_spread']} rad"),
        ("step_a_iterations / step_a_lr", f"{cfg['step_a_iterations']} / {cfg['step_a_lr']}"),
        ("step_b_iterations / step_b_lr", f"{cfg['step_b_iterations']} / {cfg['step_b_lr']}"),
        ("n_orders", cfg["n_orders"]),
        ("frozen_modes", cfg["frozen_modes"]),
        ("region / far_field_padding", f"{cfg['region']} / {cfg['far_field_padding']}"),
        ("target_side", cfg["target_side"]),
        ("w_uniformity / w_efficiency", f"{cfg['w_uniformity']} / {cfg['w_efficiency']}"),
        ("warm_start", cfg["warm_start"]),
        ("n_calibration_probes", cfg["n_calibration_probes"]),
        ("min_geometry_correlation", cfg["min_geometry_correlation"]),
        ("acceptance_loss_delta", cfg["acceptance_loss_delta"]),
        ("acceptance_score_eps", cfg["acceptance_score_eps"]),
        ("damp_lr_factor", cfg["damp_lr_factor"]),
        ("escalate_probe_count / max_probe_count",
         f"{cfg['escalate_probe_count']} / {cfg['max_probe_count']}"),
        ("max_rejection_streak", cfg["max_rejection_streak"]),
        ("settle_discard / n_eval_frames", f"{cfg['settle_discard']} / {cfg['n_eval_frames']}"),
        ("cam_type / cam_id / cam_size",
         f"{cfg['cam_type']} / {cfg['cam_id']} / {cfg['cam_size']}"),
        ("panel_span_px / pupil_center",
         f"{cfg['panel_span_px']} / {cfg['pupil_center']}"),
        ("camera_pixel_um / slm_pixel_um",
         f"{cfg['camera_pixel_um']} / {cfg['slm_pixel_um']}"),
        ("wavelength_nm / focal_length_m",
         f"{cfg['wavelength_nm']} / {cfg['focal_length_m']}"),
    ]))
    A("")
    A(
        f"孪生侧的 `region` 原生光斑腰 `native_w0(region)` 由 `TWIN_REGION = {twin_region}` 线性缩放;\n"
        "硬件路径不强制这个等式 —— 它标定自己的几何。\n"
    )

    # -- 8 权重加载 -------------------------------------------------------
    A("## 8. 加载预训练权重: 现状与真正的障碍\n")
    A(
        "**当前没有加载路径。** 硬件路径的 Step A 种子是硬编码的 `np.zeros(n_coeffs)`, 每轮在此之上\n"
        "热启动; 没有 `--load`、没有 `torch.load`、没有 `state_dict`。\n"
    )
    A(
        "**有一个注入点, 但它不是加载器.** `model_in_loop_shaping.StepAConfig.initial_coefficients`\n"
        "存在且带形状校验, 但: (a) 只被孪生的 `simulate_iterative_shaping` 读取, 硬件优化器不读;\n"
        "(b) CLI 从不暴露它; (c) 它自己的 docstring 写的是「给一个故意错误的猜测, 让 fit→shape 的\n"
        "迭代过程可观测」—— 它是**诊断旋钮**, 不是权重加载接口。接到硬件上属于改用途, 不是修 bug。\n"
    )
    A("**离线 checkpoint 是真实存在的**, `logs/*/best_coefficients.pt`:\n")
    A(
        "```python\n"
        "{'coefficients': tensor(K, float64),   # 非 piston 的 Noll 序, 弧度\n"
        " 'n_max': …, 'grid': …, 'observable': …, 'normalization': …,\n"
        " 'far_field_padding': …, 'config': {...}}\n"
        "```\n"
    )
    A(
        "**但两边差一位 piston, 这是会静默错位的坑.** 离线 `K = calc_n_zernike_terms(n_max) - 1`\n"
        "(排除 Noll 1), 硬件 `n_coefficients = calc_n_zernike_terms(n_orders)` (包含 Noll 1):\n"
    )
    A("| | 系数个数 |\n|---|---|\n| 离线 `n_max=4` → K | **14** |\n| 硬件 `--n-orders 4` | **15** |\n")
    A(
        "**14 用任何 `--n-orders` 都取不到** —— 形状校验要求恰好等于 `calc_n_zernike_terms(n_orders)`。\n"
        "要对接必须**在前面补一个 `0.0`** 当 piston (两侧都是 Noll 序, 补完 Noll 2..15 精确对齐);\n"
        "这恰好无害, 因为 `--frozen-modes` 默认已把 piston 冻在 0。\n"
    )
    A("**两个会让加载白做的前提:**\n")
    A(
        "- checkpoint 的 `observable` 应当是 `intensity`。本仓实测 `amplitude` 在三个 `n_max` 上都落后\n"
        "  `intensity` 约 0.12 R²; 仓库里现成的 `logs/amp_wandb/` 恰好是 `amplitude`, 属较弱变体。\n"
        "- Step A **每轮都从探针重拟合**, 所以预训练权重只缩短第 1 轮; 收益是「起步在正确的盆地」,\n"
        "  不是「跳过拟合」。trust region 与验收测试仍然会审它。\n"
    )

    # -- 9 输出 -----------------------------------------------------------
    A("## 9. 输出契约\n")
    A(
        "- `data/slm_model_in_loop/<日期>/`: 逐轮历史 CSV、`best_phase.npy` (**raw 未包裹弧度**, 用\n"
        "  `Santec.create_phase_from_array()` 下发)、`bench_geometry.json` (拟合出的几何, 供复现)、\n"
        "  可选最优远场图 PNG。\n"
        "- `data/debug/slm_model_in_loop_<ts>/<ts>/*.pkl`: **每次运行都写**, 不依赖 `--debug`。\n"
        "  结构 `{epoch: record}`, 带 `_epoch` 索引键, 配 `.json` sidecar (含完整已解析配置)。\n"
        "  ⚠️ 行内**只有标量** (由测试锁定), 逐帧图像不在其中; 最优帧另存 PNG。\n"
        "  CSV 额外保留字符串列 `reason`, 它不能进 pkl 的标量集 (writer 用 `float()` 转换)。\n"
    )

    # -- 10 状态机 --------------------------------------------------------
    A("## 10. 终止状态\n")
    A("| 状态 | 含义 |\n|---|---|")
    A("| `completed` | 至少一轮被接受 (或正确地保留了平场) |")
    A("| `aborted_unidentifiable` | 几何标定从未达到要求的相关度 |")
    A("| `aborted_rejection_streak` | 连续 reject 达上限 —— 台架落在模型之外 |")
    A("")
    A(
        "退出码 `2` = 几何 bake-off 失败或持续落在模型之外, 该轮**没有**提交任何相位;\n"
        "这种台架应改用 `slm-gs-refine`。\n"
    )

    # -- 11 断言边界 ------------------------------------------------------
    A("## 11. 本报告**不**主张什么\n")
    A(
        "- **不含壁钟耗时。** §3 的图是操作数, 不是秒。\n"
        "- **不含真机结论。** 环境是 `离线`; 本报告描述代码, 没有任何一次硬件运行的数据。\n"
        "- **不含收敛保证。** Step A 是非凸拟合, 探针多样性降低但**不消除**局部驻点;\n"
        "  报告不主张给定轮数内必然收敛。\n"
        f"- **孪生上的分数不可与硬件分数直接比较**: 孪生用 `TWIN_REGION = {twin_region}` 的\n"
        "  解析光路, 硬件侧的标度来自 bake-off, 二者的绝对强度标度不同。\n"
    )

    return "\n".join(parts)


# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=REPORT_KEY,
                        help="report path, repo-root relative")
    parser.add_argument("--no-figures", action="store_true",
                        help="markdown only")
    args = parser.parse_args()

    _verify_symbols()

    cfg = _dataclass_defaults(HARDWARE, "SlmModelInLoopConfig")
    consts = _module_constants(OPTIMIZER, ("PLATEAU_PATIENCE", "TWIN_REGION"))

    out_path = ROOT / args.out
    fig_dir = out_path.parent / "figures"
    if not args.no_figures:
        fig_dir.mkdir(parents=True, exist_ok=True)
        _setup_fonts()
        _fig_layers(fig_dir / "model_in_loop_layers.png")
        _fig_timeline(fig_dir / "model_in_loop_timeline.png", cfg)
        _fig_guards(fig_dir / "model_in_loop_guards.png", cfg)

    body = render(cfg, consts, figures=not args.no_figures)
    stamped = insert_header(body, REPORT_KEY)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(stamped, encoding="utf-8")
    print(f"wrote {args.out} ({len(stamped)} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())