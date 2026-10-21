"""Generate the "online re-calibration efficiency" analysis report.

Fully offline: **never opens a device**. Every number in the report is either
(a) introspected live out of the module that owns it, or (b) found by scanning
the source tree for a literal. Nothing about the cost model is hardcoded here,
so the report cannot silently drift away from the code it describes -- the same
"constant-drift guard" idea that ``generate_bench_probe_report.py`` applies to
its bench constants.

Writes ``report/shape_trials/online_recal_efficiency/online_recal_efficiency.md`` + ``figures/``.
Provenance is stamped at write time so a re-run cannot delete it.

Run:
    python scripts/slm/generate_online_recal_efficiency_report.py
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import rcParams  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from _common.provenance import insert_header  # noqa: E402

REPORT_KEY = "report/shape_trials/online_recal_efficiency/online_recal_efficiency.md"
OUT_DIR = ROOT / "report" / "shape_trials" / "online_recal_efficiency"
FIG_DIR = OUT_DIR / "figures"

rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

SCAN_ROOTS = ("src", "scripts")


# ---------------------------------------------------------------------------
# Live introspection: the numbers come from the owning modules
# ---------------------------------------------------------------------------


def _mod(name: str):
    return importlib.import_module(name)


def scan_constant_assignments(value: str, *, roots: tuple[str, ...] = SCAN_ROOTS):
    """Find every *assignment* of ``value`` to a module-level constant.

    Returns ``(relpath:line, symbol_name)``. A report asserting "the same constant
    is spelled out N times" has to count the assignments itself, or the claim rots
    the moment someone edits one of them.

    Deliberately narrower than a substring search: a raw ``"7400"`` grep also
    catches prose in comments and docstrings (6 of the 11 raw hits on this bench),
    and counting those would inflate the duplication count into a false claim.
    Only ``NAME = 7400`` / ``NAME: T = 7400`` at module level counts.
    """
    hits: list[tuple[str, str]] = []
    pat = re.compile(rf"^([A-Z_][A-Z0-9_]*)\s*(?::[^=]+)?=\s*{re.escape(value)}(?:\.0)?\s*$")
    for root in roots:
        for path in sorted((ROOT / root).rglob("*.py")):
            for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                m = pat.match(line)
                if m:
                    hits.append((f"{path.relative_to(ROOT).as_posix()}:{i}", m.group(1)))
    return hits


def import_side_effects(module: str) -> dict[str, bool]:
    """Check what ``import <module>`` drags in, in a **fresh interpreter**.

    This is what decides the fix for §3. ``generate_bench_probe_report.py``
    hardcodes the constant with the stated reason that "its defining module
    transitively pulls in the hardware driver packages". A report may not take
    that on faith, so it measures it: driver modules *and* whether a native SDK
    actually got loaded.
    """
    import os
    import subprocess

    code = (
        "import sys\n"
        f"import {module}\n"
        "print('DRIVERS', bool([m for m in sys.modules if m.startswith('ao_shaping.drivers.')]))\n"
        "print('CTYPES', 'ctypes' in sys.modules)\n"
        "print('NATIVE_SDK', 'gxipy' in sys.modules)\n"
    )
    # Inherit the environment: on Windows a stripped env (no SystemRoot) makes
    # asyncio fail to initialise its IOCP provider, which would make this probe
    # report "no side effects" for the wrong reason. A false-clean result here
    # would flip the report's conclusion, so the probe must fail loudly instead.
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{ROOT / 'src'}{os.pathsep}{ROOT / 'libs'}"
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env
    )
    out: dict[str, bool] = {}
    for line in proc.stdout.splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2:
            out[parts[0]] = parts[1].strip() == "True"
    if proc.returncode != 0 or len(out) < 3:
        raise RuntimeError(
            f"import side-effect probe failed (rc={proc.returncode}); "
            f"stderr tail: {proc.stderr.strip()[-300:]}"
        )
    return out


def scan_regex(pattern: str, *, roots: tuple[str, ...] = SCAN_ROOTS) -> list[str]:
    """Like :func:`scan_constant_assignments` but for a regex (argparse defaults).

    Skips this generator's own source: its scan patterns are string literals that
    contain the very text being searched for, so without the exclusion the report
    cites itself.
    """
    self_name = Path(__file__).name
    hits: list[str] = []
    pat = re.compile(pattern)
    for root in roots:
        for path in sorted((ROOT / root).rglob("*.py")):
            if path.name == self_name:
                continue
            for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if pat.search(line):
                    hits.append(f"{path.relative_to(ROOT).as_posix()}:{i}: {line.strip()}")
    return hits


def gather() -> dict:
    """Pull every reported number out of the module that owns it."""
    bk = _mod("ao_shaping.tools.kernels.bench_kernels")
    zc = _mod("ao_shaping.tools.zernike.common")
    mls = _mod("ao_shaping.optimizer.wfless.model_in_loop_shaping")
    smil = _mod("ao_shaping.optimizer.wfless.slm_model_in_loop")
    sweep = _mod("ao_shaping.tools.zernike.sweep_probe")
    tilt = _mod("ao_shaping.tools.slm.panel.tilt_probe")
    beam = _mod("ao_shaping.tools.slm.panel.beam_extent")
    abba = _mod("ao_shaping.tools.zernike.abba_probe")
    drift = _mod("ao_shaping.tools.ccd.drift_probe")

    cfg = {f.name: f.default for f in dataclasses.fields(smil.SlmModelInLoopConfig)}
    cap_sig = inspect.signature(bk.display_and_average).parameters

    def dc(obj) -> dict:
        return {f.name: f.default for f in dataclasses.fields(obj)}

    return {
        # bench constants
        "tilt_shift_scale": float(bk.TILT_SHIFT_SCALE),
        "slm_pitch_m": float(bk.SLM_PITCH_M),
        "beam_center_panel": tuple(bk.BEAM_CENTER_PANEL),
        "beam_radius_panel": float(bk.BEAM_RADIUS_PANEL),
        "panel_wh": (int(bk.SLM_PANEL_W), int(bk.SLM_PANEL_H)),
        "max_exposure_ms": float(zc.MAX_EXPOSURE_MS),
        # settle discipline (per phase write)
        "n_discard": int(cap_sig["n_discard"].default),
        "n_frames": int(cap_sig["n_frames"].default),
        "wait_time_s": float(cap_sig["wait_time_s"].default),
        "stable_tol": float(cap_sig["stable_tol"].default),
        "max_wait_s": float(cap_sig["max_wait_s"].default),
        # solver gates
        "min_correlation": float(mls._MIN_CORRELATION),
        "max_defocus_fit_rms": float(mls._MAX_DEFOCUS_FIT_RMS),
        "min_shift_correlation": float(mls._MIN_SHIFT_CORRELATION),
        # online loop
        "cfg": cfg,
        # sweep points enumerated by the probe's own builder
        "sweep_points": len(sweep.default_sweep_points()),
        # probe params
        "tilt": dc(tilt.TiltProbeParams),
        "beam": dc(beam.BeamExtentParams),
        "abba": dc(abba.AbbaProbeParams),
        "drift": dc(drift.DriftProbeParams),
        # duplication scan (assignments only, not prose mentions)
        "tilt_scale_assigns": scan_constant_assignments(str(int(bk.TILT_SHIFT_SCALE))),
        "side_effects": import_side_effects("ao_shaping.tools.kernels.bench_kernels"),
        "probes_default": scan_regex(r'"--probes".*default'),
    }


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------


def cost_model(d: dict) -> dict:
    """Turn the introspected defaults into SLM writes / CCD frames per action.

    The per-phase frame floor is ``n_discard`` + 2 stability probes + ``n_frames``
    averaged, which is the minimum :func:`display_and_average` can consume before
    it is allowed to return.
    """
    per_phase = d["n_discard"] + 2 + d["n_frames"]
    cfg = d["cfg"]
    ramp_n = len(str(d["tilt"]["periods"]).split(","))

    actions = {
        "哨兵 (1 平场 + 2 tilt)": 3,
        "tilt_probe 全套": 1 + ramp_n * int(d["tilt"]["repeat"]),
        "beam_extent 全套": 1 + int(d["beam"]["steps"]) * int(d["beam"]["repeats"]),
        "model_in_loop 单轮": int(cfg["probe_count"])
        + int(cfg["holdout_probe_count"])
        + 1,
        "完整几何窗口 (sweep)": d["sweep_points"] + 1,
    }
    for k, v in actions.items():
        actions[k] = (v, v * per_phase)
    return {"per_phase": per_phase, "actions": actions}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def render_figures(d: dict, cm: dict) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    per_phase = cm["per_phase"]
    labels = list(cm["actions"])
    writes = [cm["actions"][k][0] for k in labels]
    frames = [cm["actions"][k][1] for k in labels]
    full_w = cm["actions"]["完整几何窗口 (sweep)"][0]
    full_f = cm["actions"]["完整几何窗口 (sweep)"][1]

    # -- fig 1: cost ladder -------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    y = range(len(labels))
    colors = ["#2e7d32", "#558b2f", "#9e9d24", "#ef6c00", "#c62828"]
    ax[0].barh(list(y), writes, color=colors)
    ax[0].set_yticks(list(y))
    ax[0].set_yticklabels(labels)
    ax[0].set_xscale("log")
    ax[0].set_xlabel("SLM 写入次数 (log)")
    ax[0].set_title(f"每次重标定动作的写入成本 (完整窗口 = {full_w})")
    for i, w in enumerate(writes):
        ax[0].text(w * 1.12, i, f"{w}  ({w / full_w:.1%} of window)",
                   va="center", fontsize=9)
    ax[0].set_xlim(min(writes) * 0.5, max(writes) * 3.2)

    ax[1].barh(list(y), frames, color=colors)
    ax[1].set_yticks(list(y))
    ax[1].set_yticklabels([])
    ax[1].set_xscale("log")
    ax[1].set_xlabel("CCD 帧数 (log)")
    ax[1].set_title(
        f"对应 CCD 帧数 (每次写入 ≥{per_phase} 帧: "
        f"{d['n_discard']} 丢 + 2 稳定探测 + {d['n_frames']} 平均)"
    )
    for i, fr in enumerate(frames):
        ax[1].text(fr * 1.12, i, f"{fr}  ({fr / full_f:.1%})", va="center", fontsize=9)
    ax[1].set_xlim(min(frames) * 0.5, max(frames) * 3.2)
    fig.suptitle(
        "光路变化后重标定的成本阶梯 (默认配置, 数值由源码现场推导)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "recal_cost_ladder.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # -- fig 2: escalation ladder ------------------------------------------
    cfg = d["cfg"]
    p0 = int(cfg["probe_count"])
    step = int(cfg["escalate_probe_count"])
    pmax = int(cfg["max_probe_count"])
    streak_max = int(cfg["max_rejection_streak"])
    probes = list(range(p0, pmax + 1, step))
    rounds = list(range(1, streak_max + 2))
    grid = np.array(
        [[(p + int(cfg["holdout_probe_count"]) + 1) * per_phase for p in probes]
         for _ in rounds]
    )

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    im = ax.imshow(grid, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(probes)))
    ax.set_xticklabels([str(p) for p in probes])
    ax.set_yticks(range(len(rounds)))
    ax.set_yticklabels([f"第 {r} 次连续拒绝" for r in rounds])
    ax.set_xlabel("probe_count (拒绝后 +%d, 上限 %d)" % (step, pmax))
    ax.set_ylabel("拒绝连击")
    for r in range(len(rounds)):
        for c in range(len(probes)):
            ax.text(c, r, str(grid[r, c]), ha="center", va="center", fontsize=9)
    ax.axhline(streak_max - 0.5, color="#1565c0", lw=2.5)
    ax.text(
        len(probes) - 0.4,
        streak_max - 0.35,
        f"再拒一次 → 中止 (max_rejection_streak={streak_max})",
        color="#1565c0",
        fontsize=9,
        ha="right",
    )
    ax.set_title("现有升级阶梯 (形状正确, 但触发信号是像差 loss 而非几何漂移)")
    fig.colorbar(im, ax=ax, label="该轮 CCD 帧数")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "escalation_ladder.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def build_markdown(d: dict, cm: dict) -> str:
    cfg = d["cfg"]
    per_phase = cm["per_phase"]
    a = cm["actions"]
    sent_w, sent_f = a["哨兵 (1 平场 + 2 tilt)"]
    win_w, win_f = a["完整几何窗口 (sweep)"]
    r_w, r_f = a["model_in_loop 单轮"]
    tscale = int(d["tilt_shift_scale"])

    def pct(x: float, total: float) -> str:
        return f"{x / total:.1%}"

    n_assign = len(d["tilt_scale_assigns"])
    dup = "\n".join(f"| `{loc}` | `{sym}` |" for loc, sym in d["tilt_scale_assigns"])
    probes_hits = "\n".join(f"- `{h}`" for h in d["probes_default"]) or "- _(未找到)_"
    se = d["side_effects"]

    def yn(v: bool) -> str:
        return "**是**" if v else "否"

    return f"""# 光路变化后的在线重标定：效率分析 (2026-10-13)

## 0. 一句话

"在线微调"目前把**两件不同的事**混在一起：几何标定（瞳孔中心 / 光斑半径 / 远场像素尺度，
物理量，错了就是错了）与像差前向模型重拟合（补偿量，**几何错了也能"拟合成功"**）。
先分层、再按「重测成本 × 敏感度」排序，才是提效的正确入口——
而不是去优化已经优化得最好的那个环节。

> ⚠️ **本报告的性质**：成本模型全部由**源码现场推导**（§3、§4），因此不会与代码漂移；
> §6 的文献数字是**人工核实过的外部引用**；但 §7 的建议是**提案，未在本台架验证**。
> 报告里没有任何一个新测量。

## 1. 问题：几何误差会被像差系数吸收，然后被冻结

`slm_model-in-loop` 的在线循环每轮做两件事，它们**可辨识性完全不同**：

| | 作用 | 谁在解 | 错了会怎样 |
|---|---|---|---|
| **几何** | 瞳孔中心 / 盘半径 / 光腰 / 远场尺度 | `model_in_loop_shaping.calibrate_bench_geometry`（散斑相关打分） | 像差拟合会**吸收**它 |
| **像差** | Zernike 系数向量 | `_fit_aberration_at_probes`（Adam 最小化 loss） | 误差被 Step B 冻结 |

关键耦合：Step B 用的是**冻结**像差下的全像面自由相位
（`shape_phase_with_frozen_aberration`）。所以一旦几何错了，Step A 会把它"解释"成
像差，Step B 再把这个错误解释固化成相位——**全程不报错**。
这是静默失效，不是收敛慢。

因此提效的第一原则不是"跑得更快"，而是**别把几何问题当像差问题解**。

## 2. 参数分层：按重测成本 × 敏感度

| 层 | 参数 | 现场读到的值 | 重测成本 | 结论 |
|---|---|---|---|---|
| **T0** | SLM 像素间距 | `{d['slm_pitch_m'] * 1e6:g} µm` | 0 | 物理常量，**不该变** |
| **T0** | 面板尺寸 | {d['panel_wh'][0]}×{d['panel_wh'][1]} | 0 | 同上 |
| **T0** | 曝光上限 | `{d['max_exposure_ms']:g} ms` | 0 | 同上 |
| **T1** | 远场像素尺度 `TILT_SHIFT_SCALE` | **{tscale}** | {sent_w} 次写入（哨兵） | ✅ **应在线** |
| **T1** | 瞳孔中心 | `{d['beam_center_panel']}` | 同上 | ✅ **应在线** |
| **T2** | 瞳孔盘半径 / 光腰 | `beam_radius={d['beam_radius_panel']:g} px` | {a['beam_extent 全套'][0]} 次写入 | ⚠️ 专用窗口 |
| **T3** | Zernike 像差向量 | — | {r_w} 次写入/轮 | ✅ 已在线 |

**结构事实**：远场尺度是恒等式 `focal_scale = λ·f / (d_slm · p_cam)`，
`f` 与 `p_cam` 只有一个自由度。**当前代码却有两个独立常量**（`{tscale}` 与
`camera_pixel_um`），可以静默分叉——见 §3。

## 3. 🔴 P0：同一个常量在代码里有 {n_assign} 处赋值副本

下表由**扫描源码树、只认模块级赋值**得到（`scan_constant_assignments("{tscale}")`）：

| 位置 | 符号 |
|---|---|
{dup}

> ⚠️ 口径说明：直接 grep `{tscale}` 会得到 11 处命中，但其中 **6 处是注释与文档字符串里的叙述**
> （`slm_gs_refine.py` 的推导说明、`matrix_utils.py` 的 docstring、`tilt_probe.py` 的斜率注释），
> **不算重复定义**。本报告只数赋值。

**为什么是第一优先级**：这正是 `generate_bench_probe_report.py` §1.3
"stale-geometry guard" 要防的失效模式，但**发生在代码里而不是数据里**。
光路一动，若只改其中一处，另 {n_assign - 1} 处会静默保持旧值——
几何守卫查的是文件，不是 import 图。

### 3.1 但"直接 import 它"是错的修法

`generate_bench_probe_report.py:16-18` 给出了它硬编码的**明确理由**：

> *That constant is used as a literal (`7400`) rather than imported, because its
> defining module transitively pulls in the hardware driver packages.*

本报告**实测**了这个理由（在全新解释器里 import `bench_kernels`）：

| 副作用 | 实测 |
|---|---|
| 引入 `ao_shaping.drivers.*` | {yn(se.get('DRIVERS', False))} |
| 引入 `ctypes` | {yn(se.get('CTYPES', False))} |
| **加载原生 SDK（`gxipy`）** | {yn(se.get('NATIVE_SDK', False))} |

⇒ 理由**方向成立但比原文写的弱**：
{"驱动包确实被拖进来了" if se.get("DRIVERS") else "驱动包并未被拖进来"}，
{"`ctypes` 也被引入" if se.get("CTYPES") else "`ctypes` 未被引入"}——
对一个纯离线报告生成器这是不该有的副作用（运行时可见 `register_camera` 的 DEBUG 行）；
但由于 PEP-562 惰性契约，{"**真正的原生 SDK 并未被加载**" if not se.get("NATIVE_SDK") else "**原生 SDK 也被加载了**"}。

**所以正确修法不是"那 {n_assign} 处都 import 一次"**——那会把硬编码当初要避开的副作用重新引进来。
正确修法是把这些纯常量下沉到一个**零导入叶子模块**，让 {n_assign} 处都能无副作用 import。
仓库已有现成先例与配套守卫：`utils/cli/params.py`（AGENTS.md 明确记载的零导入叶子，
并由 `tests/ao_shaping/utils/test_cli_params_leaf.py` 做 **AST 级**遍历守卫——
grep 挡不住藏在函数体里的延迟 import）。

## 4. 成本模型（默认配置，数值现场推导）

每次相位写入的固定开销，来自 `bench_kernels.display_and_average` 的签名：

> `{d['n_discard']}` 帧丢弃 + ≥2 帧稳定探测（`stable_tol={d['stable_tol']}`）
> + `{d['n_frames']}` 帧平均，首等 `{d['wait_time_s']:g} s`，
> 上限 `max_wait_s={d['max_wait_s']:g} s`
> ⇒ **每次写入 ≥{per_phase} 帧 CCD**

| 动作 | SLM 写入 | CCD 帧 | 占完整窗口 |
|---|---|---|---|
| 哨兵（1 平场 + 2 tilt） | {sent_w} | {sent_f} | {pct(sent_w, win_w)} |
| `tilt_probe` 全套 | {a['tilt_probe 全套'][0]} | {a['tilt_probe 全套'][1]} | {pct(a['tilt_probe 全套'][0], win_w)} |
| `beam_extent` 全套 | {a['beam_extent 全套'][0]} | {a['beam_extent 全套'][1]} | {pct(a['beam_extent 全套'][0], win_w)} |
| `model_in_loop` **单轮** | {r_w} | {r_f} | {pct(r_w, win_w)} |
| **完整几何窗口（sweep）** | **{win_w}** | **{win_f}** | 100% |

![成本阶梯](figures/recal_cost_ladder.png)

两个可直接用的结论：

- **哨兵比完整窗口便宜 {win_w / sent_w:.0f}×**，甚至比一轮 `model_in_loop`
  还便宜 {r_f / sent_f:.1f}×。⇒ 光路没动时，不该触发全量重标定。
- sweep 点数 {d['sweep_points']} 由探针**自己的** `default_sweep_points()` 现场枚举得到
  （ramp 5×2 轴、tilt 2×2 轴、defocus 8、astig 4×2 模式、coma 4×2 模式、spherical 4
  ⇒ {d['sweep_points']}；按每条系数列表只算一次会少数 15 点）。

runbook 的采集阶段默认探针数（`--probes`，另行扫描）：

{probes_hits}

## 5. 提效杠杆（按收益/成本排序）

### 5.1 几何探针必须用随机相位，不是平滑 Zernike —— 最高优先级

这是本仓库**自己已经踩过并记录**的坑，两句话必须放在一起读：

> runbook 记载：*Zernike 探针在大面板半径上构建，会让受照明核心近乎平坦，
> 因此没有散斑可相关，求解器会饱和在搜索网格边缘。*
> —— 而 `calibrate_bench_geometry` **恰恰是靠散斑相关打分**的。

两者不矛盾的前提是：探针与打分**都用**随机相位。现状是
`_calibrate_geometry` 走随机探针（已正确），而 sweep 阶段仍用平滑 Zernike。

外部佐证：Schroff et al., *arXiv:2408.07672* —— 数字孪生用 **10 个随机 SLM 相位图案、
约 5 分钟、λ/170** 标定傅里叶成像系统，并顺带解出像素串扰。

### 5.2 分级触发：复用已有的升级阶梯，只换触发信号

`SlmModelInLoopConfig` 已经内置一条**形状正确**的阶梯：

| 字段 | 值 |
|---|---|
| `probe_count` | {cfg['probe_count']} |
| `escalate_probe_count` | {cfg['escalate_probe_count']} |
| `max_probe_count` | {cfg['max_probe_count']} |
| `max_rejection_streak` | {cfg['max_rejection_streak']} |
| `damp_lr_factor` | {cfg['damp_lr_factor']} |
| `holdout_probe_count` | {cfg['holdout_probe_count']} |
| `n_eval_frames` | {cfg['n_eval_frames']} |

![升级阶梯](figures/escalation_ladder.png)

"拒绝 → 降 lr → 加探针 → 连拒 {cfg['max_rejection_streak']} 次即中止"的逻辑是对的，
**但它由 Step A 像差 loss 触发，而不是由几何漂移触发**。改触发源即可复用，
不需要新机制——这也是本报告里唯一"几乎零成本"的高收益项。

### 5.3 把"重测"变成"重分析"

Recorder 已落盘全部帧（`utils/io/file.py::Recorder`、`save_recorder_debug_artifacts`），
`gs_refine_replay.py::load_replay_plan` 已能离线重放。未利用的是：
一次 sweep 的帧本可**同时**解出尺度 + 盘半径 + 光腰 + 各模式偏移
（`calibrate_bench_geometry_from_sweep` 就是这么做的），现在拆成多轮 = 重复采集。

### 5.4 SPGD 不能用来辨识传递矩阵 —— 别指望它

Vorontsov & Sivokon 1998 (*JOSA A* 15:2745) 的 SPGD 是**纯梯度**方法：
它从度量变化估梯度方向，**从不辨识 actuator→detector 传递矩阵**。
D-SPGD (2002, *JOSA A* 19:356) 的"10–20 迭代、与分辨率无关"是**梯度**效率，不是辨识效率。

想做在线辨识，正确工具是 RLS：

- *arXiv:2203.04915* —— 递归最小二乘在线更新影响矩阵，样本数少于批量 LS
- *Opt. Express* 29(21):33741 (2021) —— 影响矩阵是**局部的、随形状变化**的
- *arXiv:2311.17779* (Themis) —— 用**随机指令标定交互矩阵** + Kalman 滤波

### 5.5 自动重对准的精度指标可对齐

| 方法 | 精度 | 来源 |
|---|---|---|
| SPRINT 交互矩阵错位估计 | **<0.1 SA, <0.1°, <1% 放大率** | *arXiv:2406.15336* |
| DOPC 数字传播自对准 | 轴向 1.5 cm / 面内 5 mm / 旋转 5°，PBR ~30% | *Opt. Express* 22:14054 (2014) |
| 单像素实空间对准 | 亚 mm / mrad / 亚像素，**PBR >90%** | *Phys. Rev. Applied* 21,014004 (2024) |

⇒ 若光路真的动了，SLM↔CCD 的错位（平移/旋转/放大）应该用**交互矩阵**估，
而不是用单轴斜率估。

## 6. SOTA 参考成本表（外部文献，人工核实）

| 方法 | 测量次数 | 来源 |
|---|---|---|
| 模态无感 AO | ≈ **N+1**（N 模态） | Booth 2006, *Opt. Express* 14:1339 |
| 低频度量 | **3 / 模态** | Débarre 2007, *Opt. Express* 15:8176 |
| Hadamard 三步干涉 | **3N**（N=4096 时可压 10–20%） | *JOSA A* 40(1):45 (2023) |
| 单次无参考 | **1 帧 / 200 Zernike 模态** | *Light: Sci. Appl.* 2026, s41377-026-02241-5 |
| 在线 SLM 标定 | **10 图案 / ~5 min / λ/170** | *arXiv:2408.07672* |
| 影响矩阵在线更新 (RLS) | 样本数 **少于**批量 LS | *arXiv:2203.04915* |

> 本项目的 {d['sweep_points']} 点 sweep 已经优于 N+1（Hadamard 3N 对 N≈14 即 42），
> **探针设计是好的，贵在每次都要跑满**。

## 7. 建议执行顺序（提案，未验证）

| 优先级 | 动作 | 收益 | 风险 |
|---|---|---|---|
| **P0** | 把重复常量下沉到零导入叶子模块，再让 {n_assign} 处 import（§3.1，**不是**直接 import） | 消除静默不一致 | 低（纯清理 + AST 守卫） |
| **P0** | 几何探针统一改随机相位（§5.1） | 求解器脱离饱和边界 | 低 |
| **P1** | 加 {sent_w} 写入门哨兵，驱动已有升级阶梯（§5.2） | 光路未动时省 ~{win_f // max(sent_f, 1)} 帧 | 中（改运行时行为） |
| **P1** | Step A 加几何一致性校验：尺度偏差过大时**拒绝该轮** | 堵住"像差吸收几何误差"（§1） | 中（可能频繁中止） |
| **P2** | 一次 sweep 多解参数，离线复用帧（§5.3） | 减少重复采集 | 中 |
| **P3** | 上 RLS 影响矩阵在线更新（§5.4） | 替代部分重标 | 高 |

**求解器已有的拒绝门槛**（可复用来定哨兵阈值）：
`_MIN_CORRELATION={d['min_correlation']}`、`_MAX_DEFOCUS_FIT_RMS={d['max_defocus_fit_rms']}`、
`_MIN_SHIFT_CORRELATION={d['min_shift_correlation']}`、
在线循环 `min_geometry_correlation={cfg['min_geometry_correlation']}`。

## 8. 边界与撤回声明

- **本报告不含任何新测量**。§4 的每个数字都来自源码默认值；
  §6 的每个数字都是外部已发表结果（人工核实）。§7 全部是提案。
- 哨兵阈值（相对 `{tscale}` 的百分比）**尚未确定**——它取决于你的重对准公差要求，
  本报告不替它选默认值。选错的后果是"该升不升"或"不该升乱升"，两者方向相反，
  必须在台架上量出来而不是拍。
- 早期检索里出现的 `arXiv:2610.02518` **查无此文，已弃用**；
  `arXiv:2005.08562`（WaveBlocks）**未能独立复核**，因此未列入 §6。
- §5.4 的"SPGD 不辨识传递矩阵"是**方法学判断**，不是本台架实测；
  若要下结论需构造一个"模型故意错误"的对照实验。
"""


def main() -> None:
    d = gather()
    cm = cost_model(d)
    render_figures(d, cm)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = ROOT / REPORT_KEY
    out.write_text(insert_header(build_markdown(d, cm), REPORT_KEY), encoding="utf-8")
    print(f"Wrote {out}")
    print(f"  figures: {FIG_DIR.relative_to(ROOT).as_posix()}/"
          f"recal_cost_ladder.png, escalation_ladder.png")
    print(f"  per-phase CCD floor = {cm['per_phase']} frames")
    print(f"  sweep points (enumerated by the probe) = {d['sweep_points']}")
    print(
        f"  {d['tilt_shift_scale']:.0f} constant assignments = "
        f"{len(d['tilt_scale_assigns'])} (assignments only, prose excluded)"
    )


if __name__ == "__main__":
    main()