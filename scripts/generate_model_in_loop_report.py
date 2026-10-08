"""Generate the model-in-the-loop usage write-up (call graph / timing / algorithm).

**What this document is.** A *functional description* -- how the command is wired,
what runs in what order, why each guard exists, and how the forward-prediction
model is put together. It is **not** the conclusion of a measurement, so it lives
in ``docs/`` and needs no instrument.

**Why the diagrams are generated rather than hand-written.** Three reasons, in
order of how much they have actually cost this repo:

1. **The diagrams would rot.** A hand-drawn call graph names symbols; renaming one
   turns the document into fiction that still reads as authoritative. So the
   diagrams are *derived from the source*: every symbol they name is resolved
   against the real modules before the file is written, plus a check in the other
   direction -- that every declared symbol is actually mentioned in the output. A
   rename breaks generation loudly instead of silently producing a confident lie.
2. **The numbers come from the code.** The config table's defaults are read out of
   the dataclass AST; the learned model's parameter counts are measured by
   importing it and calling ``count_parameters``. Neither is typed in by hand, and
   when torch is missing the document degrades to *no number* rather than an
   unverified one.
3. **Traceability is mechanical.** The document writes its own header, so a reader
   can always find the script that produced it.

**Why ``docs/`` and not ``report/``.** AGENTS.md splits the two: ``report/<topic>/``
holds the conclusions of measurements, ``docs/`` holds device/usage documentation.
The *hardware measurement* of this same runner is
``report/slm/model_in_loop_bench_calibration.md``; the two are deliberately kept
apart. Consequently this file is **not** in ``scripts/_common/provenance.py::REPORTS``
-- that registry renders ``report/README.md`` and assumes every key is under
``report/``, so a ``docs/`` key there would compute the wrong relative path.

**Deliberately absent.** No wall-clock timing. The per-round *counts* of device
round-trips and compute steps are derivable from the configuration, so they are
reported; the *durations* depend on the bench and would be a fabrication here. A
previous report in this family was invalidated by exactly that mistake.

**Diagrams are mermaid**, because this document is read on GitHub where mermaid
renders. ASCII art needed hand-aligned boxes to stay legible and still could not
show control flow or state.

Usage::

    python scripts/generate_model_in_loop_report.py
    python scripts/generate_model_in_loop_report.py --no-figures
    python scripts/generate_model_in_loop_report.py --out docs/slm/model_in_loop_algorithm.md
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

from scripts._common.provenance import script_link  # noqa: E402

#: Where the generated write-up lands. ``docs/``, not ``report/``: this is a usage /
#: architecture description, and AGENTS.md reserves ``report/<topic>/`` for the
#: conclusions of measurements. Not registered in ``provenance.REPORTS`` because that
#: registry renders ``report/README.md`` and assumes every key lives under ``report/``.
DOC_KEY = "docs/slm/model_in_loop_algorithm.md"

#: Appended after each embedded diagram so the next paragraph is separated by a
#: blank line -- markdown needs one, and `"\\n".join` alone would not provide it.
NEWLINE = "\n"

#: Repo-root-relative modules the diagrams and tables are derived from.
HARDWARE = "src/ao_shaping/optimizer/wfless/slm_model_in_loop.py"
TWIN = "src/ao_shaping/optimizer/wfless/model_in_loop_shaping.py"
OPTIMIZER = "src/ao_shaping/algorithm/signal_processing/zernike_coefficient_optimizer.py"
RUNNER = "src/ao_shaping/runners/slm/model_in_loop_runner.py"
PARAMS = "src/ao_shaping/runners/runner_common.py"
IO = "src/ao_shaping/utils/io/file.py"
GS_REFINE = "src/ao_shaping/optimizer/wfless/slm_gs_refine.py"
LEARNED = "src/ml/zernike/forward_model.py"
PHYSICS = "src/ml/zernike/models.py"
AMP_CKPT = "src/ml/zernike/amp_checkpoint.py"
FRAUNHOFER = "src/ao_shaping/utils/wavefront/fraunhofer.py"

MODULES = {
    "hardware": HARDWARE,
    "twin": TWIN,
    "optimizer": OPTIMIZER,
    "runner": RUNNER,
    "params": PARAMS,
    "io": IO,
    "gs_refine": GS_REFINE,
    "learned": LEARNED,
    "physics": PHYSICS,
    "amp_ckpt": AMP_CKPT,
    "fraunhofer": FRAUNHOFER,
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
    # the optional offline-checkpoint seed (Wave 3) and its read seam (Wave 2)
    ("hardware", "_seed_coefficients_from_checkpoint"),
    ("amp_ckpt", "load_trained_forward_model"),
    ("amp_ckpt", "check_geometry"),
    ("amp_ckpt", "UNVERIFIED_GEOMETRY_FIELDS"),
    ("amp_ckpt", "TrainedForwardModel"),
    # the one canonical Fraunhofer propagator (Wave 1)
    ("fraunhofer", "focal_field"),
    ("fraunhofer", "focal_intensity"),
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
    # the learned forward model (the section on its structure)
    ("learned", "ZernikeCoeffConfig"),
    ("learned", "ZernikeCoeffConvNet"),
    ("learned", "ZernikeCoeffMLP"),
    ("learned", "_ConvBlock"),
    ("learned", "_make_norm"),
    ("learned", "build_forward_model"),
    ("learned", "count_parameters"),
    ("learned", "peak_normalize"),
    ("learned", "DEFAULT_N_COEFFS"),
    # the physics forward model, for contrast
    ("physics", "ZernikeAmpModel"),
    # the analytic model Step A actually differentiates through
    ("optimizer", "forward_intensity"),
    ("optimizer", "_far_field_intensity"),
    ("optimizer", "_intensity_loss"),
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
# The mermaid diagrams
#
# mermaid rather than ASCII art: this document is read on GitHub, where mermaid
# renders. The ASCII version needed hand-aligned boxes to stay legible and still
# could not show control flow or state.
# ---------------------------------------------------------------------------

CALL_GRAPH = """\
```mermaid
graph TD
    CLI["main.py · slm-model-in-loop"]

    subgraph L1["① 硬件编排层 · runners/slm/model_in_loop_runner.py"]
        direction TB
        RUN["run(ctx, params: SlmModelInLoopParams)"]
        PARSE["_parse_frozen_modes / _parse_point"]
        CFG["SlmModelInLoopConfig<br/>(runner_common.py 扁平 dataclass)"]
        RUN --> PARSE --> CFG
    end

    subgraph L2["② 策略层 (硬件移植) · optimizer/wfless/slm_model_in_loop.py"]
        direction TB
        OPT["optimize_slm_model_in_loop(config)<br/>← 唯一公开入口"]
        SEED["_seed_coefficients_from_checkpoint(config)<br/>开设备之前 · 仅当 --forward-checkpoint"]
        OPEN["_open_bench(config)"]
        GEOM["_calibrate_geometry(bench, config)<br/>一次性几何 bake-off"]
        MKOPT["_make_optimizer(config, coefficients)"]
        CLAMP["trust_region_clamp(...)"]
        QUAL["_metrics_at / _quality"]
        TGRID["_to_model_grid(...)"]
        ACC["acceptance_verdict(...)"]
        VERDICT{{"ModelInLoopStatus"}}
    end

    subgraph LCK["②' 离线权重读缝 · ml/zernike/amp_checkpoint.py"]
        direction TB
        LOAD["load_trained_forward_model(path)"]
        CHK["check_geometry(...)<br/>不逐项相符即报错"]
    end

    subgraph LFR["②'' 唯一的 Fraunhofer 传播 · utils/wavefront/fraunhofer.py"]
        direction TB
        FR["focal_field / focal_intensity<br/>中心补零 + fftshift(fft2(ifftshift))"]
    end

    subgraph LB["③ 设备抽象 · 唯一随 --cam_type 改变的一层"]
        direction TB
        BENCH["Bench (Protocol)<br/>display / measure / close"]
        SIM["_SimBench<br/>2f-Fourier 数字孪生"]
        HW["_HardwareBench<br/>daheng/miicam + Santec"]
        BENCH -.实现.- SIM
        BENCH -.实现.- HW
    end

    subgraph L3["④ 共享数学 = 数字孪生 · optimizer/wfless/model_in_loop_shaping.py"]
        direction TB
        GEOMSOLVE["calibrate_bench_geometry(...)"]
        PROBE["_probe_phase(...)"]
        STEPA["_fit_aberration_at_probes(...)"]
        STEPB["shape_phase_with_frozen_aberration(...)"]
        TWINRUN["simulate_iterative_shaping(config)<br/>--cam_type sim 走的就是这条"]
    end

    subgraph L4["⑤ 算法层 · algorithm/signal_processing"]
        direction TB
        ZCO["ZernikeCoefficientOptimizer<br/>Adam 作用在 Zernike 系数向量上"]
    end

    subgraph LX["借用的帧预处理 · slm_gs_refine.py"]
        direction TB
        PREP["_prepare_frame"]
    end

    subgraph LO["⑥ 记录 · utils/io/file.py"]
        direction TB
        SAVE["save_recorder_debug_artifacts(...)<br/>每次运行都写 pkl + json sidecar"]
    end

    CLI --> RUN
    CFG --> OPT
    OPT -->|"种子: 默认 np.zeros(n_coeffs)"| SEED
    SEED -.延迟 import.- LOAD
    LOAD --> CHK
    CHK -->|"不逐项相符 → 开设备前就报错"| SEED
    OPT --> OPEN
    OPEN --> BENCH
    OPT --> GEOM
    GEOM -.延迟 import.- GEOMSOLVE
    OPT -->|"平场基线"| QUAL
    QUAL -->|"基准 score"| OPT

    OPT -->|"每轮: coefficients (零种子 或 checkpoint 种子)"| MKOPT
    MKOPT --> ZCO
    ZCO -.解析远场.- FR
    OPT -->|"每轮 × probe_count"| PROBE
    PROBE --> BENCH
    BENCH -->|"measure()"| PREP
    PREP --> TGRID
    TGRID --> STEPA
    STEPA --> ZCO
    ZCO --> CLAMP
    CLAMP --> STEPB
    STEPB -->|"display(shaped) + measure()"| BENCH
    BENCH --> QUAL
    QUAL --> ACC
    ACC -->|"accept"| OPT
    ACC -->|"reject"| OPT
    ACC -->|"streak ≥ max_rejection_streak"| VERDICT
    OPT --> SAVE

    SIM -.孪生不另写一份, 只是换掉 Bench.-> TWINRUN
    TWINRUN -.共享同一批数学.- GEOMSOLVE
    TWINRUN -.-> STEPA
    TWINRUN -.-> STEPB
```"""

SEQUENCE = """\
```mermaid
sequenceDiagram
    autonumber
    actor U as 用户
    participant M as main.py
    participant R as runner
    participant O as optimize_slm_model_in_loop
    participant B as Bench (设备)
    participant Z as ZernikeCoefficientOptimizer

    U->>M: slm-model-in-loop <命令行>
    M->>R: run(params)
    R->>O: optimize_slm_model_in_loop(config)

    opt --forward-checkpoint 已给
        O->>O: _seed_coefficients_from_checkpoint(config) 〔开设备之前〕
        Note over O: 不逐项相符 → 此刻就报错, 不烧台架时间
    end

    rect rgb(245, 240, 235)
        Note over O,B: 一次性 —— 几何标定
        O->>Z: coefficients = np.zeros(n_coeffs) 〔默认零种子〕
        O->>B: _open_bench(config)
        B-->>O: 设备已连接 (import 全部延迟)
        loop n_calibration_probes 次
            O->>B: display(probe) + measure()
            B-->>O: 实测远场帧
        end
        O->>O: calibrate_bench_geometry(...)
        alt 相关度 < min_geometry_correlation
            O-->>U: 退出码 2 —— 不提交任何相位
        end
    end

    rect rgb(240, 245, 240)
        O->>B: display(flat)
        B-->>O: _prepare_frame(measure())
        O->>O: _metrics_at → _quality → score_before
    end

    rect rgb(238, 243, 249)
        Note over O,Z: 第 t 轮 —— Step A: 重拟合一个共享像差
        loop probe_count 次
            O->>O: _probe_phase(region, probe_spread, seed)
            O->>B: display(probe)
            B-->>O: measure() → _prepare_frame
            O->>O: _to_model_grid(...) 搬到模型网格
        end
        O->>Z: step_a_iterations 次前向+反向<br/>(轮流喂全部探针, 同一 Adam 状态)
        Z-->>O: fitted + loss_before/after
        O->>O: trust_region_clamp → clamped
    end

    rect rgb(249, 240, 245)
        Note over O,B: Step B: 像差冻结, 合成方形
        O->>Z: step_b_iterations 次全像素相位 Adam
        Z-->>O: shaped
        O->>B: display(shaped)
        B-->>O: _prepare_frame → _metrics_at → _quality
    end

    O->>O: acceptance_verdict(loss, score)
    alt accept
        O->>O: c ← clamped, phase ← shaped, streak = 0
    else reject
        O->>O: lr 阻尼, 探针 +escalate, streak += 1
    end
    O-->>R: ModelInLoopResult
    R-->>U: CSV / pkl / best_phase.npy / bench_geometry.json
```"""

GUARD_STATE = """\
```mermaid
stateDiagram-v2
    [*] --> 进入第t轮: trust_region_clamp → clamped
    进入第t轮 --> 验收: acceptance_verdict<br/>loss 与 score 前后对比
    验收 --> accept: 未变差
    验收 --> reject: loss 变差 或<br/>score 变差超 acceptance_score_eps
    accept --> 下一轮: c ← clamped<br/>phase ← shaped<br/>lr 复原, streak = 0
    reject --> 下一轮: lr × damp_lr_factor<br/>探针 +escalate_probe_count<br/>(上限 max_probe_count)
    reject --> aborted_rejection_streak: streak ≥ max_rejection_streak
    aborted_rejection_streak --> [*]: 不提交任何相位
    下一轮 --> 进入第t轮
```"""

LEARNED_PIPELINE = """\
```mermaid
flowchart LR
    subgraph PRE["前处理 · Dataset 拥有输入契约"]
        direction TB
        C1["实测 Zernike 系数<br/>Noll 序 · raw 弧度"]
        C2["零填充到 136 维<br/>(实测长度 15 / 36 / 78)"]
        C3["模型内部不做输入归一化"]
        C1 --> C2 --> C3
    end

    subgraph DEEP["深度模块 · ZernikeCoeffConvNet"]
        direction TB
        B1["coeff_proj<br/>Linear 136 → 256·4·4 = 4096<br/>再 view 成 (B,256,4,4)<br/>★ 空间结构在这里被创造"]
        D1["stage1 Upsample×2 → _ConvBlock<br/>256 → 256"]
        D2["stage2 Upsample×2 → _ConvBlock<br/>256 → 128"]
        D3["stage3 Upsample×2 → _ConvBlock<br/>128 → 64"]
        D4["stage4 Upsample×2 → _ConvBlock<br/>64 → 32"]
        B1 --> D1 --> D2 --> D3 --> D4
    end

    subgraph BLOCK["每个 _ConvBlock"]
        direction LR
        Q1["Conv2d 3×3"] --> Q2["Norm<br/>(GroupNorm 默认)"] --> Q3["GELU"] --> Q4["Conv2d 3×3"] --> Q5["Norm"]
    end

    subgraph POST["后处理 · 输出契约"]
        direction TB
        P1["head = Conv2d 32→1, k=1<br/>raw / 无界 · 不加激活"]
        P2["peak_normalize<br/>逐样本 amax, 除数下限 1e-12"]
        P3["(B,1,64,64) 每样本最大值恰为 1.0"]
        P1 --> P2 --> P3
    end

    PRE -->|"(B,136) 弧度"| DEEP
    DEEP -->|"(B,32,64,64)"| POST
```"""


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



# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def _table(rows: list[tuple[str, str]]) -> str:
    head = "| 项 | 值 |\n|---|---|"
    body = "\n".join(f"| `{k}` | `{v}` |" for k, v in rows)
    return f"{head}\n{body}"


def _learned_model_facts() -> dict[str, int]:
    """Measured facts about ``ml.zernike.forward_model``, or ``{}`` if unavailable.

    Imported rather than recomputed: hand-rolling ``Linear``/``Conv2d``/``GroupNorm``
    parameter arithmetic -- or re-deriving ``calc_n_zernike_terms`` -- inside this
    generator would each be a second copy of something the code already owns, i.e.
    exactly the duplicate-implementation drift this repo keeps paying for. Returns
    ``{}`` when torch is unavailable so the document degrades to *no number* rather
    than an unverified one.
    """
    try:
        from ml.zernike.forward_model import (  # noqa: PLC0415
            DEFAULT_N_COEFFS,
            ZernikeCoeffConfig,
            ZernikeCoeffConvNet,
            ZernikeCoeffMLP,
            count_parameters,
        )
    except Exception as exc:  # torch missing / ml dependency group not installed
        print(f"note: cannot measure the learned model's facts "
              f"({type(exc).__name__}: {exc}); that section omits those numbers")
        return {}
    config = ZernikeCoeffConfig()
    return {
        "n_coeffs": int(DEFAULT_N_COEFFS),
        "conv": count_parameters(ZernikeCoeffConvNet(config)),
        "mlp": count_parameters(ZernikeCoeffMLP(config)),
    }


def _section_forward_model(cfg: dict[str, str]) -> list[str]:
    """The forward-prediction model, in three stages.

    Which model this is matters, because the repo has three and they are easy to
    confuse: the one Step A differentiates through is *not* the learned one.
    """
    facts = _learned_model_facts()
    width = facts.get("n_coeffs", 136)
    out: list[str] = []
    A = out.append

    A("## 8. 正向预测模型的结构 (前处理 / 深度模块 / 后处理)\n")
    A(
        "本仓有**三个**都叫「正向模型」的东西, 用途完全不同。先分清, 再讲结构 —— 混淆这三者\n"
        "本项目已经付出过代价: 离线 checkpoint 的 piston 约定与闭环约定差一位 (§9)。\n"
    )
    A("| 模型 | 模块 | 形态 | 谁在用 |\n|---|---|---|---|")
    A("| **解析 FFT** (闭环内联) | `ZernikeCoefficientOptimizer._far_field_intensity` | 无参数: `exp(i·patch)` → 中心补零 → `fftshift(fft2(ifftshift))` → `\\|F\\|²` | **`slm-model-in-loop` 的 Step A 真正在优化的那个**; Step B 的梯度也穿过它 |")
    A("| **物理 + 可学习系数** | `ZernikeAmpModel` | 整个数据集共享**一个**系数向量 (`nn.Parameter`, 零初始化) | 离线标定与逆向设计; `K = calc_n_zernike_terms(n_max) − 1` (不含 piston) |")
    A("| **学习式前向网络** | `ZernikeCoeffConvNet` / `ZernikeCoeffMLP` | 系数投影 + 卷积解码器 | 离线训练与评测; 本节余下部分讲它 |\n")
    A(
        "> 也就是说: **闭环跑的是解析 FFT 那一个**, 不是下面的神经网络。讲后者是因为 §9 的\n"
        "> checkpoint 来自它, 而它与闭环约定的差异正是那条坑。\n"
    )
    A(
        "解析那一条的完整链条很短, 也正因如此才值得写下来: "
        "`patch = (phase_slm + aberration) × aperture_t` → `field = amplitude × exp(i·patch)`\n"
        "→ 中心补零到 `far_field_size` → `fftshift(fft2(ifftshift(field), norm=\"ortho\"))` →\n"
        "强度取 `re² + im²`。预测入口是 `forward_intensity`, 损失是 `_intensity_loss` —— 它把强度\n"
        "按 `intensity / (max + PEAK_EPS)` 归一化后取 MSE(`PEAK_EPS = 1e-8`)。\n"
        "**这里没有可学参数**: Step A 拟合的是那一个共享 Zernike 像差向量, 不是网络权重。\n"
    )

    A("### 8.1 前处理: 输入侧刻意不归一化\n")
    A(
        f"输入是 `(B, {width})` 的 **Noll 序、raw 弧度**系数向量"
        f"(`DEFAULT_N_COEFFS = calc_n_zernike_terms(15)`)。实测语料里的向量长度不一\n"
        f"(15 / 36 / 78), 统一**零填充**到 {width} 维 —— 于是有 {width - 78}–{width - 15} 个输入维"
        "**结构上恒为零**。\n"
    )
    A(
        "**`forward` 内部不做任何输入归一化**, 这是有意的: 输入契约归 Dataset 所有, 模型不该\n"
        "偷偷再缩一次。代价是语料系数本身很小 —— 实测 `max|c| = 0.0617 rad`、`rms = 0.0142` ——\n"
        "所以桥接层用 `nn.init.xavier_uniform_` 而不是零/极小初始化, 让 `Linear` 去学这些小尺度。\n"
    )

    A("### 8.2 深度模块: 系数投影 + 卷积解码器\n")
    A(LEARNED_PIPELINE + NEWLINE)
    A(
        "**为什么是「系数投影 + 卷积解码」而不是 flatten 到 `grid*grid` 的 MLP.** 这个映射是确定性的、\n"
        "光滑的, 而且它的输出有**真实的二维局部性**: 类 Airy 光斑的位置跟着 tip/tilt 走, 径向结构跟着\n"
        "高阶模式走。flatten 到向量的 MLP 把这个几何丢掉, 得靠单个权重矩阵重新学出平移等变性。\n"
        "于是 `coeff_proj` 先把一个裸向量**创造**成常数特征图, 之后交给平移等变的卷积解码。\n"
    )
    A(
        "**可达性是一个被强制的不变量**: `grid == bottleneck · 2^len(features)`。默认\n"
        "`grid=64`、`features=(256,128,64,32)` → `len=4` 个 2 倍上采样级, 于是 `bottleneck` 被**推导**为\n"
        "`64 // 16 = 4`, 不是挑出来的。`grid` 是这个等式里动不了的一边(它是数据集物化的网格, 也是\n"
        "`(B,1,grid,grid)` 的输出契约), 所以让路的一定是 `bottleneck`。`__init__` 里不满足就报错, 并\n"
        "直接告诉你该拧哪个旋钮。\n"
    )
    A(
        "**每个 `_ConvBlock` 是 `Conv3×3 → Norm → GELU → Conv3×3 → Norm`.** 三点讲究:\n"
        "- **先上采样再细化**: 最近邻上采样不引入新的可平均的值, 所以随后的 3×3 在更细的尺度上\n"
        "  看到的是**真实邻域**, 而不是插值出来的平台。\n"
        "- **默认 GroupNorm 而不是 BatchNorm**: 目标是逐帧 peak 归一化的, 逐样本尺度已经被归一化掉了;\n"
        "  而 `BatchNorm` 还带每步更新的 running 统计量, 在 `eval()` 里被消费, 这会让训练时的行为\n"
        "  **耦合**到它见过的 batch 顺序与历史。在几百条 64×64 记录上, 这种耦合是纯方差。\n"
        "  `GroupNorm` 逐样本按通道组归一化, train/eval 行为一致, 跨 epoch 不带状态。\n"
        "  `batch` / `none` 仍可选, 是为了让这个选择**可测**而不是只能被断言。\n"
        "- `_make_norm` 在通道宽度不被 `norm_groups` 整除时按 `gcd(channels, norm_groups)` **降组**\n"
        "  并记一条 debug, 而不是静默取整 —— 静默取整会把「你要求的组数根本不可用」藏起来。\n"
    )

    A("### 8.3 后处理: 输出契约是 raw + peak 归一化\n")
    A(
        "`head` 是 `Conv2d(32, 1, kernel_size=1)`, **raw 且无界, 刻意不加激活**。这一点是硬要求:\n"
        "`ml.phase.unet.UNetGenerator` 的图像头字面就是 `self.sigmoid(self.final_conv(x))`, 出不来\n"
        "`(0,1)` 之外的值, 会把动态范围**压平** —— 越过轨道的值全被压成同一个数, 模型就再也分不出\n"
        "「很亮的核」和「亮的核」。`test_forward_model.py::TestAntiSigmoidHead` 是让未来有人\n"
        "「顺手复用 U-Net」而大声失败的回归守卫。\n"
    )
    A(
        "逐帧尺度由调用方施加: `peak_normalize(x) = x / clamp(amax(x, dim=(-2,-1)), min=1e-12)`。\n"
        "它的三条性质都是契约的一部分: **负值不裁掉**; **除数不跨 batch 共享**, 只逐样本;\n"
        "下界 `1e-12` 与 `ml.zernike.models._EPS` 一致, 免得两个归一化器漂移。\n"
    )
    A(
        "**由此得到一个必须记住的推论**: `denormalized()` 返回的张量 `argmax` **按构造恰为 1.0**。\n"
        "所以这个模型的输出**回答不了任何绝对亮度问题**(到了传感器多少光、激光漂没漂), 只回答\n"
        "相对结构。\n"
    )

    A("### 8.4 基线臂与参数量\n")
    A(
        "`build_forward_model(config)` 按 `config.architecture` (`\"conv\"` / `\"mlp\"`) 实例化其中一条臂。\n"
        "`ZernikeCoeffMLP` 是**刻意的对照组**: 输出契约与 conv 臂完全一致 (raw、无界、`(B,1,g,g)`),\n"
        "但把 `grid*grid` 全部从一个 `Linear` 里吐出来 —— 丢掉的正是 2-D 局部性, 而这正是该对照要\n"
        "检验的假设。\n"
    )
    if facts:
        A(
            f"默认 `ZernikeCoeffConfig()` 下用 `count_parameters` **实测**: "
            f"**conv {facts['conv']:,}**, **mlp {facts['mlp']:,}** —— MLP 参数量更大而输出更差,\n"
            "这正是「结构对, 而不是容量大」的论据。\n"
        )
    else:
        A("_参数量需要 torch 才能测; 本次运行环境缺 torch, 故不填数字 —— 不用手算的数代替实测。_\n")

    A("### 8.5 评测口径\n")
    A(
        "**只看 R² / correlation / 光斑域指标, 永远不要用 MSE/PSNR/SSIM 给这个模型选超参。**\n"
        "目标是逐帧 peak 归一化的, 预测与目标的 `max` 都被钉在 1.0, 于是那三个指标主要在度量\n"
        "**归一化本身**而不是拟合质量。实测反例: 某个 sibling 任务上「总能量」归一化给出\n"
        "PSNR 72 dB、SSIM 0.9996, 而 R² 反而**更差**。\n"
    )
    return out


def _verify_mentioned(body: str) -> None:
    """Fail if a declared diagram symbol never appears in the document.

    The source-side check (:func:`_verify_symbols`) catches renames; this one
    catches the opposite drift -- a symbol left in ``DIAGRAM_SYMBOLS`` after its
    diagram was rewritten, which would leave the guard claiming to protect
    something the document no longer mentions.
    """
    absent = [name for _, name in DIAGRAM_SYMBOLS if name not in body]
    if absent:
        raise SystemExit(
            "model-in-loop write-up: DIAGRAM_SYMBOLS lists symbols the document "
            "never mentions (diagram rewritten, list not updated):\n  - "
            + "\n  - ".join(absent)
        )


def render(cfg: dict[str, str], consts: dict[str, str], *, figures: bool) -> str:
    fig_dir = "figures"
    fig2 = f"{fig_dir}/model_in_loop_timeline.png"

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
    A(
        "每个节点都是真实符号, ①…⑥ 标出它归哪一层。虚线 = 延迟 import "
        "(硬件栈在 import 期就碰硬件, 所以这些 import 全在函数内部)。\n"
    )
    A(CALL_GRAPH + NEWLINE)
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
    A(
        "一轮的完整时序。`rect` 的底色区分阶段, 与 §3 的操作预算图配色一致: "
        "米色 = 一次性几何标定, 浅绿 = 平场基线, 蓝 = Step A, 粉 = Step B。\n"
    )
    A(SEQUENCE + NEWLINE)
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
        "这与 `ml/zernike` 的离线约定**差一位**, 见 §9。\n"
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
    A(GUARD_STATE + NEWLINE)
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

    # -- 8 正向模型结构 ---------------------------------------------------
    parts.extend(_section_forward_model(cfg))

    # -- 9 权重加载 -------------------------------------------------------
    A("## 9. 加载预训练权重: 现状与真正的障碍\n")
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
    A("## 10. 输出契约\n")
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
    A("## 11. 终止状态\n")
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
    A("## 12. 本报告**不**主张什么\n")
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


def _doc_header(key: str) -> str:
    """Provenance header for a ``docs/`` write-up.

    Uses the canonical :func:`script_link` so the relative path is computed from
    the file's own depth rather than hardcoded, but deliberately does **not** use
    ``provenance.insert_header``: that is reserved for registry-managed reports,
    and the marker comments it emits belong to the sync pass to rewrite.
    """
    depth = len(Path(key).parts) - 1
    script = "scripts/generate_model_in_loop_report.py"
    return "\n".join((
        f"> **生成脚本**: {script_link(script, depth)}",
        "> **复现命令**: `python scripts/generate_model_in_loop_report.py`",
        "> **运行环境**: 离线 (纯源码静态分析 + 可选 torch 实测; 不开设备)",
        "> **说明**: 本文档是**使用/结构说明**, 不是测量结论 —— 硬件实测另见",
        "> [`report/slm/model_in_loop_bench_calibration.md`]"
        "(../../report/slm/model_in_loop_bench_calibration.md)。",
    ))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=DOC_KEY,
                        help="output path, repo-root relative")
    parser.add_argument("--no-figures", action="store_true",
                        help="skip the matplotlib figure (mermaid always renders)")
    args = parser.parse_args()

    _verify_symbols()

    cfg = _dataclass_defaults(HARDWARE, "SlmModelInLoopConfig")
    consts = _module_constants(OPTIMIZER, ("PLATEAU_PATIENCE", "TWIN_REGION"))

    out_path = ROOT / args.out
    fig_dir = out_path.parent / "figures"
    if not args.no_figures:
        fig_dir.mkdir(parents=True, exist_ok=True)
        _setup_fonts()
        _fig_timeline(fig_dir / "model_in_loop_timeline.png", cfg)

    body = render(cfg, consts, figures=not args.no_figures)
    _verify_mentioned(body)
    header = _doc_header(args.out)
    lines = body.splitlines()
    at = next((i for i, ln in enumerate(lines) if ln.startswith("# ")), 0)
    rest = lines[at + 1:]
    while rest and not rest[0].strip():
        rest.pop(0)
    stamped = "\n".join(lines[: at + 1] + ["", header, ""] + rest).rstrip("\n") + "\n"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(stamped, encoding="utf-8")
    print(f"wrote {args.out} ({len(stamped)} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())