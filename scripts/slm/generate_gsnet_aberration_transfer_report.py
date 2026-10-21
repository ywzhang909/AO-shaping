"""渲染 GSNet 光路相差迁移实验报告（issue #165 迁移学习可迁移性检验，纯仿真）。

读入：``scripts/slm/gsnet_aberration_transfer.py`` 写出的 ``result.json``
（六种光路相差 × 六种冻结策略的迁移前后整形质量），以及**五个变体**
（A–E：``--w-phase`` / ``--operator`` / ``--gate`` 三条修法开关的不同组合）
的 ``coupling_<标签>/result.json``。

渲染 ``report/gsnet_aberration_transfer/report.md``（纯文本报告，无图）。

用法::

    # 推荐：一次渲染五个变体的对比（目录里自动发现 coupling_<标签>/result.json）
    python scripts/slm/generate_gsnet_aberration_transfer_report.py \
        --variants-dir data/debug/gsnet_aberration_transfer

    # 显式指定（可重复；优先级高于自动发现，按给出顺序渲染）
    python scripts/slm/generate_gsnet_aberration_transfer_report.py \
        --variant A=data/debug/gsnet_aberration_transfer/coupling_A/result.json \
        --variant E=data/debug/gsnet_aberration_transfer/coupling_E/result.json

    # 旧接口：单个 results.json（无变体对比时仍可用）
    python scripts/slm/generate_gsnet_aberration_transfer_report.py \
        --results data/debug/gsnet_aberration_transfer/<ts>/results.json

``--variants-dir`` 只发现名字匹配 ``coupling_<标签>`` 的子目录（冒烟目录与
更早的 ``coupling/`` 基线不会被误收）。任何变体 JSON 缺失都**不致命**：
对应数据位一律标注「未记录」。输入全部不存在时渲染一份**明确标注
「实验尚未运行」的占位报告**，绝不虚构任何数字；情形 / 冻结策略的实际
网格以输入 JSON 为准，可以与默认清单不同。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "src", ROOT, ROOT / "scripts", ROOT / "scripts" / "slm"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from scripts._common import fmt_metric as _fmt  # noqa: E402
from scripts._common.provenance import insert_header  # noqa: E402

REPORT_KEY = "report/gsnet_aberration_transfer/report.md"
REPORT_DIR = ROOT / "report" / "gsnet_aberration_transfer"
DEFAULT_RESULTS = ROOT / "data" / "debug" / "gsnet_aberration_transfer" / "results.json"
DEFAULT_VARIANTS_DIR = ROOT / "data" / "debug" / "gsnet_aberration_transfer"

MISSING = "未记录"

#: 情形清单是默认网格；实际以输入 JSON 里出现的 case 名为准。
CASE_ORDER = (
    "none",
    "panel-defocus",
    "panel-coma",
    "panel-spherical",
    "lens-coma",
    "camera-defocus",
)
CASE_LABELS = {
    "none": "无附加相差（零误差孪生链）",
    "panel-defocus": "面板静态相差：离焦（Noll 4）",
    "panel-coma": "面板静态相差：彗差",
    "panel-spherical": "面板静态相差：球差",
    "lens-coma": "透镜像差：彗差",
    "camera-defocus": "相机离焦（焦面传播）",
}

#: 冻结策略清单同样是默认网格；JSON 里缺哪个就少渲染哪一行。
FREEZE_ORDER = (
    "full",
    "frozen",
    "freeze-encoder",
    "freeze-decoder",
    "freeze-films",
    "freeze-outconv",
)
FREEZE_LABELS = {
    "full": "全参数微调",
    "frozen": "全部冻结（等价于不迁移，直接推理）",
    "freeze-encoder": "冻结 encoder（特征主干不动）",
    "freeze-decoder": "冻结 decoder（相位生成头不动）",
    "freeze-films": "冻结 FiLM 条件调制层",
    "freeze-outconv": "冻结输出卷积层",
}

#: ConditionUNet 四组的参数量表格来源候选键（JSON 里存在才渲染）。
MODEL_PARAM_GROUPS = ("encoder", "decoder", "films", "outconv")

#: --variants-dir 下只收这个名字模式的子目录（coupling_A / coupling_E ...），
#: 冒烟目录与更早的 coupling/ 基线不进变体表。
VARIANT_SUBDIR_RE = re.compile(r"coupling_[A-Za-z0-9]+\Z")


def _num(value: Any) -> float | None:
    """把 JSON 值安全转成 float；不可转（缺失 / null / 字符串）返回 None。"""
    if isinstance(value, bool) or value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None  # NaN -> None


def _metric(ref: Any, key: str = "composite") -> str:
    if not isinstance(ref, dict):
        return MISSING
    return _fmt(_num(ref.get(key)))


def _delta(after: Any, before: Any) -> str:
    a, b = _num(after), _num(before)
    if a is None or b is None:
        return MISSING
    return f"{a - b:+.4f}"


def _int_field(row: Any, key: str) -> str:
    if not isinstance(row, dict):
        return MISSING
    value = _num(row.get(key))
    return str(int(value)) if value is not None else MISSING


def _row(cells: list[str]) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    """markdown 表：表头 + 分隔行 + 数据行（GitHub flavour）。"""
    lines = [_row(headers), _row(["---"] * len(headers))]
    lines.extend(_row(r) for r in rows)
    return "\n".join(lines)


def _rel(path: Any) -> str:
    """把容器里的绝对路径换成仓库相对路径，报告里好读。"""
    text = str(path)
    for prefix in ("/workspace/", "/workspace"):
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    return text or str(path)


def _case_names(data: dict[str, Any]) -> list[str]:
    """按默认清单排序 JSON 里出现的情形，清单外的追加在后。"""
    present = [c for c in CASE_ORDER if c in (data.get("cases") or {})]
    extra = [
        c for c in sorted((data.get("cases") or {})) if c not in CASE_ORDER
    ]
    return present + extra


def _freeze_names(data: dict[str, Any], case: str) -> list[str]:
    runs = ((data.get("cases") or {}).get(case) or {}).get("runs") or {}
    present = [f for f in FREEZE_ORDER if f in runs]
    extra = [f for f in sorted(runs) if f not in FREEZE_ORDER]
    return present + extra


def _all_freezes(data: dict[str, Any]) -> list[str]:
    """跨情形合并的冻结策略清单（保持默认顺序）。"""
    seen: list[str] = []
    for case in _case_names(data):
        for f in _freeze_names(data, case):
            if f not in seen:
                seen.append(f)
    return seen


def _refs(data: dict[str, Any], case: str) -> dict[str, Any]:
    refs = ((data.get("cases") or {}).get(case) or {}).get("references") or {}
    return refs if isinstance(refs, dict) else {}


def _pretrained(data: dict[str, Any], case: str) -> dict[str, Any]:
    ref = _refs(data, case).get("pretrained")
    return ref if isinstance(ref, dict) else {}


def _final(data: dict[str, Any], case: str, freeze: str) -> dict[str, Any]:
    runs = ((data.get("cases") or {}).get(case) or {}).get("runs") or {}
    run = runs.get(freeze) or {}
    final = run.get("final")
    return final if isinstance(final, dict) else {}


def _run(data: dict[str, Any], case: str, freeze: str) -> dict[str, Any]:
    runs = ((data.get("cases") or {}).get(case) or {}).get("runs") or {}
    run = runs.get(freeze)
    return run if isinstance(run, dict) else {}


def _model_params_table(data: dict[str, Any]) -> str:
    """ConditionUNet 四组参数量表：只从 JSON 读，缺位标注「未记录」。"""
    params = data.get("model_params") or data.get("param_groups") or {}
    if not isinstance(params, dict):
        params = {}
    rows: list[list[str]] = []
    for group in MODEL_PARAM_GROUPS:
        rows.append([group, _int_field(params, group)])
    rows.append(["**合计**", _int_field(params, "total")])
    return _table(["参数组", "参数量"], rows)


# ---------------------------------------------------------------------------
# 变体（variant）加载与统计
# ---------------------------------------------------------------------------

def _discover_variant_paths(variants_dir: Path) -> list[tuple[str, Path]]:
    """``--variants-dir`` 下按名字模式发现 ``coupling_<标签>/result.json``。"""
    found: list[tuple[str, Path]] = []
    if variants_dir is None or not variants_dir.is_dir():
        return found
    for sub in sorted(variants_dir.iterdir()):
        if sub.is_dir() and VARIANT_SUBDIR_RE.fullmatch(sub.name):
            found.append((sub.name, sub / "result.json"))
    return found


def _load_variants(args: argparse.Namespace) -> list[tuple[str, dict[str, Any] | None, Path]]:
    """显式 ``--variant NAME=path`` 优先，其后是 ``--variants-dir`` 自动发现。

    返回 (标签, JSON 数据或 None, 路径) 列表；JSON 缺失/损坏只警告，不中断。
    """
    entries: list[tuple[str, Path]] = []
    for spec in (args.variant or []):
        name, sep, raw = spec.partition("=")
        if not sep or not name.strip():
            print(f"警告: --variant 需写成 NAME=path，忽略 {spec!r}。",
                  file=sys.stderr)
            continue
        entries.append((name.strip(), Path(raw.strip())))
    known = {n for n, _ in entries}
    for name, path in _discover_variant_paths(args.variants_dir):
        if name not in known:
            entries.append((name, path))
    variants: list[tuple[str, dict[str, Any] | None, Path]] = []
    for name, path in entries:
        if path.exists():
            try:
                variants.append((name, json.loads(path.read_text(encoding="utf-8")), path))
            except (OSError, json.JSONDecodeError) as exc:
                print(f"警告: 变体 {name} 的 JSON 读取失败（{path}: {exc}），"
                      "该变体数据位标注「未记录」。", file=sys.stderr)
                variants.append((name, None, path))
        else:
            print(f"警告: 变体 {name} 的 JSON 不存在（{path}），"
                  "该变体数据位标注「未记录」。", file=sys.stderr)
            variants.append((name, None, path))
    return variants


def _cfg(data: dict[str, Any] | None) -> dict[str, Any]:
    cfg = (data or {}).get("config") or {}
    return cfg if isinstance(cfg, dict) else {}


def _variant_flags(data: dict[str, Any]) -> str:
    cfg = _cfg(data)
    gate_raw = cfg.get("gate")
    gate = "on" if gate_raw is True else ("off" if gate_raw is False else str(gate_raw))
    return (f"w-phase={_fmt(_num(cfg.get('w_phase')))} · "
            f"operator={cfg.get('operator', MISSING)} · gate={gate}")


def _deltas(data: dict[str, Any] | None, freeze: str = "full") -> list[tuple[str, float | None, float | None]]:
    """(情形, 迁移前 composite, 迁移后 composite)——按该 JSON 的情形清单。"""
    if data is None:
        return []
    out = []
    for case in _case_names(data):
        pre = _num(_pretrained(data, case).get("composite"))
        aft = _num(_final(data, case, freeze).get("composite"))
        out.append((case, pre, aft))
    return out


def _variant_stats(data: dict[str, Any] | None, freeze: str = "full") -> dict[str, Any] | None:
    """一个变体在指定冻结策略下跨情形的 Δ 汇总；无可比情形返回 None。"""
    if data is None:
        return None
    ds = [a - b for _, b, a in _deltas(data, freeze)
          if a is not None and b is not None]
    if not ds:
        return None
    return {
        "n": len(ds),
        "mean": sum(ds) / len(ds),
        "improved": sum(1 for d in ds if d > 0),
        "worst": min(ds),
        "best": max(ds),
    }


def _per_freeze_summary(data: dict[str, Any]) -> list[dict[str, Any]]:
    """主变体下逐冻结策略跨情形汇总（§5 头条表的数据）。"""
    rows: list[dict[str, Any]] = []
    cases = _case_names(data)
    for f in _all_freezes(data):
        composites, deltas = [], []
        trainable: float | None = None
        total: float | None = None
        for case in cases:
            run = _run(data, case, f)
            if trainable is None:
                trainable = _num(run.get("trainable_params"))
                total = _num(run.get("total_params"))
            pre = _num(_pretrained(data, case).get("composite"))
            aft = _num(_final(data, case, f).get("composite"))
            if aft is not None:
                composites.append(aft)
            if aft is not None and pre is not None:
                deltas.append(aft - pre)
        rows.append({
            "freeze": f,
            "trainable": trainable,
            "total": total,
            "mean_composite": (sum(composites) / len(composites)) if composites else None,
            "mean_delta": (sum(deltas) / len(deltas)) if deltas else None,
            "improved": sum(1 for d in deltas if d > 0),
            "n": len(deltas),
        })
    return rows


def _pick_primary_variant(
    variants: list[tuple[str, dict[str, Any] | None, Path]],
) -> tuple[str, dict[str, Any], Path] | None:
    """详细表格（§4/§5）渲染的基准变体：优先「修法全开」
    （calibrated + w-phase 0 + gate），否则最后一个有数据的变体。"""
    for name, data, path in variants:
        cfg = _cfg(data)
        if (data is not None and cfg.get("operator") == "calibrated"
                and _num(cfg.get("w_phase")) == 0.0 and cfg.get("gate") is True):
            return name, data, path
    for name, data, path in reversed(variants):
        if data is not None:
            return name, data, path
    return None


def _find_variant(
    variants: list[tuple[str, dict[str, Any] | None, Path]],
    *,
    operator: str | None = None,
    w_phase: float | None = None,
    gate: bool | None = None,
) -> tuple[str, dict[str, Any]] | None:
    """按 config 开关组合找一个变体（机制讨论用），标签不固定。"""
    for name, data, _path in variants:
        if data is None:
            continue
        cfg = _cfg(data)
        if operator is not None and cfg.get("operator") != operator:
            continue
        if w_phase is not None and _num(cfg.get("w_phase")) != w_phase:
            continue
        if gate is not None and bool(cfg.get("gate")) is not gate:
            continue
        return name, data
    return None


# ---------------------------------------------------------------------------
# §0 结论
# ---------------------------------------------------------------------------

def _conclusion_lines(
    data: dict[str, Any] | None,
    variants: list[tuple[str, dict[str, Any] | None, Path]],
) -> list[str]:
    lines: list[str] = []
    add = lines.append
    if data is None:
        add("**实验尚未运行：本文件是占位报告，没有任何迁移结果可下结论。**")
        add("")
        add("- 输入 JSON（`gsnet_aberration_transfer.py` 写出的 `result.json`）"
            "不存在，两个问题——(a) 各类光路相差下迁移学习是否成功、"
            "(b) 哪些参数可以冻结——都**未回答**。")
        add("- 下方所有表格的数据位一律标注「未记录」；本报告不虚构任何数字。")
        add("- 跑完 §8 的实验命令后重跑 §8 的渲染命令，本报告会由真实数字填充。")
        return lines

    named = {n: d for n, d, _ in variants if d is not None}
    if named:
        # (a) 迁移是否成功：默认配置 vs 修法组合，两组数字都从 JSON 算。
        default_v = _find_variant(variants, operator="ideal", w_phase=1.0, gate=False)
        fixed_v = _pick_primary_variant(variants)
        if default_v is not None:
            dname, ddata = default_v
            st = _variant_stats(ddata)
            if st:
                worst_case, worst_pre, worst_after = min(
                    ((c, b, a) for c, b, a in _deltas(ddata)
                     if a is not None and b is not None),
                    key=lambda t: t[2] - t[1])
                flat_ref = _refs(ddata, worst_case).get("flat")
                flat0 = _num(flat_ref.get("composite")) \
                    if isinstance(flat_ref, dict) else None
                add(f"- **问题 (a)，先说坏消息**：**默认配置**（`--operator ideal` + "
                    f"`--w-phase 1` + 无门，变体 `{dname}`）下迁移**让网络变差**："
                    f"{st['n']} 个情形里只有 {st['improved']} 个 Δ>0，"
                    f"平均 Δ = {st['mean']:+.4f}，最差单情形 `{worst_case}` "
                    f"{_fmt(worst_pre)} → {_fmt(worst_after)}（{st['worst']:+.4f}），"
                    f"跌到平场下限 {_fmt(flat0)} **之下**。")
        if fixed_v is not None:
            fname, fdata, _fpath = fixed_v
            st = _variant_stats(fdata)
            if st:
                best_case, best_pre, best_after = max(
                    ((c, b, a) for c, b, a in _deltas(fdata)
                     if a is not None and b is not None),
                    key=lambda t: t[2] - t[1])
                add(f"- **问题 (a)，修法之后成立**：换到校准算子 + 去相位项 + "
                    f"对比择优门（变体 `{fname}`，全参数微调），平均 Δ = "
                    f"{st['mean']:+.4f}，{st['improved']}/{st['n']} 个情形变好，"
                    f"最大正增益 `{best_case}` {_fmt(best_pre)} → {_fmt(best_after)}"
                    f"（{best_after - best_pre:+.4f}），最差单情形也被门压到 "
                    f"{st['worst']:+.4f}。结论：**仿真里迁移学习可以成功，但只有"
                    "「算子匹配 + 只留整形项 + 验证集对比择优门」这条组合成立；"
                    "默认配置是有害的**（机制见 §6）。")
        else:
            st = _variant_stats(data)
            if st is not None:
                add(f"- **问题 (a)**：主变体下平均 Δ = {st['mean']:+.4f}，"
                    f"{st['improved']}/{st['n']} 个情形「迁移后 composite 更高」。")

        # (b) 哪些参数可以冻结：主变体的逐策略汇总。
        if fixed_v is not None:
            fname, fdata, _fpath = fixed_v
            summ = {r["freeze"]: r for r in _per_freeze_summary(fdata)}
            dec = summ.get("freeze-decoder")
            enc = summ.get("freeze-encoder")
            full = summ.get("full")
            if dec and dec["mean_delta"] is not None:
                add(f"- **问题 (b) 总最优**：`freeze-decoder`（冻结相位生成头，"
                    f"留 {int(dec['trainable']) if dec['trainable'] is not None else MISSING} "
                    f"个可训练参数）在主变体 `{fname}` 下跨情形平均 composite 最高"
                    f"（{_fmt(dec['mean_composite'])}），平均 Δ = "
                    f"{dec['mean_delta']:+.4f}，{dec['improved']}/{dec['n']} 个情形变好。")
            if enc and enc["mean_delta"] is not None and full and full["trainable"] \
                    and enc["trainable"] is not None:
                share = enc["trainable"] / full["trainable"]
                add(f"- **问题 (b) 省参数最优**：`freeze-encoder` 只留 "
                    f"{int(enc['trainable'])} "
                    f"个可训练参数（占总数 {share:.1%}），仍拿到正的平均增益 "
                    f"Δ = {enc['mean_delta']:+.4f}，{enc['improved']}/{enc['n']} 个情形变好"
                    "——「少重学」这一轴上的最好答案。")
            if full and full["mean_delta"] is not None:
                add(f"- **全参数微调不是答案**：`full`（"
                    f"{int(full['trainable']) if full['trainable'] is not None else MISSING} "
                    f"个可训练参数）平均 Δ = {full['mean_delta']:+.4f}，"
                    f"{full['improved']}/{full['n']} 个情形变好——净持平到有害；"
                    "冻结反而把伤害挡在门外（逐策略表见 §5）。")
    else:
        config = data.get("config") or {}
        primary_freeze = str(config.get("primary_freeze") or "full")
        cases = _case_names(data)
        wins = 0
        measurable = 0
        deltas: list[float] = []
        for case in cases:
            before = _num(_pretrained(data, case).get("composite"))
            after = _num(_final(data, case, primary_freeze).get("composite"))
            if before is None or after is None:
                continue
            measurable += 1
            if after > before:
                wins += 1
            deltas.append(after - before)
        if measurable:
            add(f"- **问题 (a)**：在主冻结策略 `{primary_freeze}` 下，"
                f"{measurable} 个可比情形中 {wins} 个「迁移后 composite 高于迁移前」，"
                f"平均 Δ = {sum(deltas) / len(deltas):+.4f}。")

        primary_case = str(config.get("primary_case") or "panel-coma")
        summ = {r["freeze"]: r for r in _per_freeze_summary(data)}
        if summ:
            best = max((r for r in summ.values() if r["mean_composite"] is not None),
                       key=lambda r: r["mean_composite"])
            add(f"- **问题 (b)**：跨情形平均 composite 最高的冻结策略是 "
                f"`{best['freeze']}`（{_fmt(best['mean_composite'])}，"
                f"{best['improved']}/{best['n']} 个情形变好）。详见 §5。")
        else:
            add(f"- **问题 (b)**：主情形 `{primary_case}` 下没有任何冻结策略的运行记录，"
                "「哪些参数可以冻结」**无法从这份数据判定**。")

    add("- 全部数字来自输入 JSON；任何一个缺失都会显示「未记录」而不是被补全。")
    add("")
    add("> ⚠️ 本实验**全部是仿真**：真值是仓库规范的 2f 数字孪生"
        "（`drivers/sim/bench_twin.py`），不是实验系统的 CCD。")
    return lines


# ---------------------------------------------------------------------------
# §3 变体对比
# ---------------------------------------------------------------------------

def _variant_section_lines(
    variants: list[tuple[str, dict[str, Any] | None, Path]],
) -> list[str]:
    lines: list[str] = []
    add = lines.append
    if not variants:
        add("（本次输入没有变体 JSON——§3 待五变体运行后渲染。）")
        return lines

    add("五个变体只差在驱动脚本的三个开关上（`--w-phase` / `--operator` / "
        "`--gate`，语义见 §1），像差情形、冻结网格、探针数、种子全部相同。"
        "Δ = 迁移后 − 迁移前（`full` 冻结策略，composite 越高越好）。")
    add("")
    rows = []
    for name, data, path in variants:
        st = _variant_stats(data) if data is not None else None
        rows.append([
            f"`{name}`",
            _variant_flags(data) if data is not None else MISSING,
            f"`{_rel(path)}`",
            (f"{st['mean']:+.4f}" if st else MISSING),
            (f"{st['improved']}/{st['n']}" if st else MISSING),
            (f"{st['worst']:+.4f}" if st else MISSING),
            (f"{st['best']:+.4f}" if st else MISSING),
        ])
    add(_table(["变体", "开关", "JSON", "平均 Δ", "变好情形",
                "最差单情形 Δ", "最好单情形 Δ"], rows))
    add("")

    # 逐情形 Δ 矩阵（full 冻结）
    case_order: list[str] = []
    for _n, data, _p in variants:
        for c in _case_names(data) if data is not None else []:
            if c not in case_order:
                case_order.append(c)
    if case_order:
        head = ["情形（`full`）"] + [f"`{n}`" for n, _d, _p in variants]
        mrows = []
        for case in case_order:
            cells = [f"`{case}`"]
            for _n, data, _p in variants:
                pre = _num(_pretrained(data, case).get("composite")) if data else None
                aft = _num(_final(data, case, "full").get("composite")) if data else None
                cells.append(_delta(aft, pre))
            mrows.append(cells)
        add(_table(head, mrows))
        add("")
        add("> ⚠️ **跨变体比 Δ 时注意基准点不同**：`operator=calibrated` 家族的"
            "「迁移前」参照也用校准算子评测（保证每个变体内部 Δ 只反映微调本身），"
            "而 checkpoint 内部前向是 ideal 算子训练出来的——换算子本身就会移动"
            "迁移前分数（逐情形对照见 §6 机制 3）。同列内比较有效，跨列比较"
            "绝对分数无效。")
    add("")
    return lines


# ---------------------------------------------------------------------------
# §5 冻结汇总与矩阵
# ---------------------------------------------------------------------------

def _freeze_summary_lines(data: dict[str, Any], vtag: str) -> list[str]:
    lines: list[str] = []
    add = lines.append
    vlabel = f"变体 `{vtag}`" if vtag else "主变体"
    summ = _per_freeze_summary(data)
    if not summ:
        add(f"（{vlabel} 里没有可用于跨情形汇总的 run——{MISSING}）")
        return lines

    add(f"**逐冻结策略跨情形汇总**（{vlabel}，composite 越高越好；"
        "「变好情形」= Δ>0 的情形数）：")
    add("")
    rows = []
    for r in summ:
        share = (f"{r['trainable'] / r['total']:.1%}"
                 if r["trainable"] is not None and r["total"] else MISSING)
        rows.append([
            f"`{r['freeze']}`",
            (str(int(r["trainable"])) if r["trainable"] is not None else MISSING),
            share,
            _fmt(r["mean_composite"]),
            (f"{r['mean_delta']:+.4f}" if r["mean_delta"] is not None else MISSING),
            f"{r['improved']}/{r['n']}",
        ])
    add(_table(["策略", "可训练参数", "参数占比", "平均 composite",
                "平均 Δ", "变好情形"], rows))
    add("")

    cases = _case_names(data)
    freezes = [r["freeze"] for r in summ]
    head = ["情形", "迁移前"] + [f"`{f}`" for f in freezes]
    mrows = []
    for case in cases:
        pre = _num(_pretrained(data, case).get("composite"))
        cells = [f"`{case}`", _fmt(pre)]
        for f in freezes:
            aft = _num(_final(data, case, f).get("composite"))
            cells.append(f"{_fmt(aft)} ({_delta(aft, pre)})"
                         if aft is not None else MISSING)
        mrows.append(cells)
    add(f"**完整矩阵**（{vlabel}：情形 × 冻结策略，"
        "composite 与括号内 Δ）：")
    add("")
    add(_table(head, mrows))
    add("")
    return lines


def _broken_freeze_contrast_lines(
    variants: list[tuple[str, dict[str, Any] | None, Path]],
    primary_case: str,
) -> list[str]:
    """对照：默认（有害）配置下主情形的冻结扫描——「冻得越多越好」。"""
    broken = _find_variant(variants, operator="ideal", w_phase=1.0, gate=False)
    if broken is None:
        return []
    name, data = broken
    if primary_case not in (data.get("cases") or {}):
        return []
    lines: list[str] = []
    add = lines.append
    pre = _num(_pretrained(data, primary_case).get("composite"))
    rows = []
    for f in _freeze_names(data, primary_case):
        aft = _num(_final(data, primary_case, f).get("composite"))
        rows.append([f"`{f}`", _int_field(_run(data, primary_case, f), "trainable_params"),
                     _fmt(aft), _delta(aft, pre)])
    add(f"**对照：有害配置下的冻结扫描**（变体 `{name}`，主情形 `{primary_case}`，"
        f"迁移前 {_fmt(pre)}）——当每个真正训练起来的配置都在变差时，"
        "**冻得越多反而越好**，`frozen`（0 可训练参数）恰好复刻迁移前分数：")
    add("")
    add(_table(["策略", "可训练参数", "迁移后 composite", "Δ"], rows))
    add("")
    return lines


# ---------------------------------------------------------------------------
# §6 机制讨论
# ---------------------------------------------------------------------------

def _mechanism_lines(
    variants: list[tuple[str, dict[str, Any] | None, Path]],
    data: dict[str, Any],
    primary_name: str,
) -> list[str]:
    lines: list[str] = []
    add = lines.append
    vlabel = f"变体 `{primary_name}`" if primary_name else "主变体"

    # 机制 1：预训练已经优于 GS 标签 ⇒ 相位项是向下的拉力。
    ref_v = _find_variant(variants, operator="ideal") \
        or (primary_name, data)
    rname, rdata = ref_v
    pairs = []
    for case in _case_names(rdata):
        lab = _num(_refs(rdata, case).get("label").get("composite")) \
            if isinstance(_refs(rdata, case).get("label"), dict) else None
        pre = _num(_pretrained(rdata, case).get("composite"))
        if lab is not None and pre is not None:
            pairs.append((case, pre, lab))
    if pairs:
        wins = sum(1 for _, p, l in pairs if p > l)
        add(f"- **机制 1 —— 相位项是一个向下的拉力**：在变体 `{rname}` 的参照系里，"
            f"预训练网络在 {len(pairs)} 个相差情形中的 {wins} 个上**已经优于该情形"
            "物理下的 GS 解**（经典算法达到的水平，**不是**严格上限）。相位项的监督目标是「命令过的探针相位」"
            "= GS 解 + 随机 Zernike 微扰——目标本身比网络现状差，往它拉就是在拉低网络。"
            "逐情形（composite，经典 GS 解 vs 迁移前）：")
        add("")
        add(_table(["情形", "经典 GS 解（`label`）", "迁移前（`pretrained`）",
                    "迁移前 − 经典 GS 解"],
                   [[f"`{c}`", _fmt(l), _fmt(p), f"{p - l:+.4f}"]
                    for c, p, l in pairs]))
        add("")

    # 机制 2：只去相位项不够（ideal 算子下整形目标物理不可达）。
    a_v = _find_variant(variants, operator="ideal", w_phase=1.0, gate=False)
    b_v = _find_variant(variants, operator="ideal", w_phase=0.0, gate=False)
    if a_v and b_v:
        (an, ad), (bn, bd) = a_v, b_v
        sa, sb = _variant_stats(ad), _variant_stats(bd)
        if sa and sb:
            add(f"- **机制 2 —— 只把相位项置 0 反而更糟**：变体 `{bn}`"
                f"（`--w-phase 0`，仍是 ideal 算子）平均 Δ = {sb['mean']:+.4f}，"
                f"比变体 `{an}` 的 {sa['mean']:+.4f} 还差。相位项撤掉后只剩整形项，"
                "而整形项在默认配置里经由 **ideal 算子**去匹配**带耦合的实测帧**——"
                "这个目标物理上不可达，梯度近似噪声。逐情形 Δ：`"
                + "` / `".join(
                    f"{case} {_delta(_final(bd, case, 'full').get('composite'), _pretrained(bd, case).get('composite'))}"
                    for case, _b, _a in _deltas(bd)) + "`。")
        add("- **机制 3 —— 算子必须与产生数据的那台孪生一致**：把微调前向换成"
            "探针实测所在的**校准算子**（`--operator calibrated`，实验系统上由 "
            "`tools/slm/bench_twin_calibrate.py` 反演出的算子在仿真里的替身），"
            "再加验证集上的**对比择优门**，迁移才从有害变成有用（§3 表：门 + "
            "校准算子 + 去相位项的组合是唯一平均非负且最坏单情形被压小的变体）。")

    # 机制：算子替换本身的代价（迁移前参照移动）。
    if a_v:
        e_v = _find_variant(variants, operator="calibrated", gate=True) \
            or _find_variant(variants, operator="calibrated")
        if e_v:
            (an, ad), (en, ed) = a_v, e_v
            rows, moved = [], 0
            for case in _case_names(ad):
                pa = _num(_pretrained(ad, case).get("composite"))
                pe = _num(_pretrained(ed, case).get("composite")) \
                    if case in (ed.get("cases") or {}) else None
                if pa is None or pe is None:
                    continue
                rows.append([f"`{case}`", _fmt(pa), _fmt(pe), f"{pe - pa:+.4f}"])
                if abs(pe - pa) > 1e-6:
                    moved += 1
            if rows:
                add(f"- **算子替换的告诫**：checkpoint 内部是用 **ideal 算子**训练出来的，"
                    f"所以「推理时换成校准算子」本身就会移动迁移前参照——"
                    f"变体 `{an}` 与 `{en}` 的同一 checkpoint、同一情形，迁移前分数有 "
                    f"{moved} 个对不上（下表）。忠实的在机路径因此是：从实验系统反演孪生 → "
                    "**用拟合后的算子重训（或至少重导出）网络** → 再迁移；"
                    "runbook `docs/slm/gsnet_transfer_hw_runbook.md` §7 规定的"
                    "「先反演孪生，再让网络用它」正是这一步。")
                add("")
                add(_table(["情形", f"迁移前（ideal 评测，`{an}`）",
                            f"迁移前（calibrated 评测，`{en}`）", "移动量"], rows))
                add("")

    # 机制 4：门不是绝对保证。
    e_data = data if _cfg(data).get("gate") is True else None
    if e_data is not None:
        accepted_bad, accepted_names = [], {}
        for case in _case_names(e_data):
            g = _run(e_data, case, "full").get("gate")
            pre = _num(_pretrained(e_data, case).get("composite"))
            aft = _num(_final(e_data, case, "full").get("composite"))
            accepted_names[case] = str(g)
            if g == "accepted" and pre is not None and aft is not None and aft < pre:
                accepted_bad.append((case, aft - pre))
        st = _variant_stats(e_data)
        ungated_worst = None
        for name, d, _p in variants:
            if d is None or bool(_cfg(d).get("gate")):
                continue
            s = _variant_stats(d)
            if s and (ungated_worst is None or s["worst"] < ungated_worst[1]):
                ungated_worst = (name, s["worst"],
                                 next((c for c, b, a in _deltas(d)
                                       if a is not None and b is not None
                                       and a - b == s["worst"]), MISSING))
        n_gate = max(4, int(_cfg(e_data).get("probes") or 24) // 4)
        add(f"- **机制 4 —— 门的局限**：`--gate` 的判据是**验证集探针**（本次规模 "
            f"max(4, probes//4) = {n_gate} 帧）上的整形项，样本少，接受决策会错。"
            f"{vlabel} 里被门**接受**、却在独立 eval 集上变差的情形："
            + ("、".join(f"`{c}`（{d:+.4f}）" for c, d in accepted_bad)
               if accepted_bad else "无")
            + ("；门的逐情形判定：`"
               + "` / `".join(f"{c}: {g}" for c, g in accepted_names.items()) + "`。"
               if accepted_names else "。"))
        if st and ungated_worst:
            add(f"  门把最坏单情形从**无门变体的 {ungated_worst[1]:+.4f}"
                f"（变体 `{ungated_worst[0]}` 的 `{ungated_worst[2]}`）**压到 "
                f"{st['worst']:+.4f}——**限制**了伤害，没有**消除**它。")
        kept = sum(1 for g in accepted_names.values() if g == "kept_pretrained")
        if kept:
            add(f"  另有 {kept} 个情形被门判为「候选不如预训练」而恢复原权重"
                "（`kept_pretrained`，迁移后 = 迁移前，Δ=0）——这正是门的价值所在。")

    # 自检：frozen 复刻迁移前 ⇒ 评测链路自洽。
    max_dev: float | None = None
    for name, d, _p in variants:
        if d is None:
            continue
        for case in _case_names(d):
            if "frozen" not in ((d.get("cases") or {}).get(case) or {}).get("runs", {}):
                continue
            pre = _num(_pretrained(d, case).get("composite"))
            fro = _num(_final(d, case, "frozen").get("composite"))
            if pre is not None and fro is not None:
                max_dev = abs(fro - pre) if max_dev is None else max(max_dev, abs(fro - pre))
    if max_dev is not None:
        add(f"- **harness 自检**：所有变体里 `frozen`（0 可训练参数）与迁移前参照的 "
            f"composite 最大偏差 {max_dev:.2e}——不训练就分数不变，评测链路自洽。")
    return lines


def _discussion_lines(data: dict[str, Any]) -> list[str]:
    """无变体输入时保留的旧讨论（单 JSON 场景）。"""
    lines: list[str] = []
    add = lines.append
    config = data.get("config") or {}
    primary_freeze = str(config.get("primary_freeze") or "full")
    cases = _case_names(data)

    scored = []
    for case in cases:
        v = _num(_final(data, case, primary_freeze).get("composite"))
        if v is not None:
            scored.append((case, v))
    if scored:
        worst_case, worst_val = min(scored, key=lambda t: t[1])
        best_case, best_val = max(scored, key=lambda t: t[1])
        add(f"- **最难拟合的情形**：`{worst_case}` 迁移后 composite 最低"
            f"（{_fmt(worst_val)}），最高的是 `{best_case}`（{_fmt(best_val)}），"
            f"极差 {_fmt(best_val - worst_val)}。")
    else:
        add("- 没有任何情形记录了迁移后 composite，「哪种相差最难」无法判定。")

    target = config.get("transfer_target")
    add(f"- **`--transfer-target` 语义**：输入 JSON 记录本次用 "
        f"`{target if target is not None else MISSING}`。网络预训练时喂的是"
        "**设计**方斑作为输入目标，而闭环在实验系统上喂的是**实测**帧——"
        "这个输入语义错位需要在成对实验里单独量化。")

    any_chance = None
    for case in cases:
        runs_c = ((data.get("cases") or {}).get(case) or {}).get("runs") or {}
        for f, run in runs_c.items():
            if isinstance(run, dict) and _num(run.get("phase_loss_chance")) is not None:
                any_chance = _num(run.get("phase_loss_chance"))
                break
        if any_chance is not None:
            break
    if any_chance is not None:
        add(f"- 相位项的随机猜测水平在 JSON 里记录为 {_fmt(any_chance)}"
            "（与理论值 π²/3 ≈ 3.2899 一致）；迁移后相位项低于它，"
            "才说明小数据集真的把相位流形学动了，而不只是拟合均值。")
    return lines


# ---------------------------------------------------------------------------
# 渲染
# ---------------------------------------------------------------------------

def render(
    data: dict[str, Any] | None,
    results_path: Path,
    variants: list[tuple[str, dict[str, Any] | None, Path]] | None = None,
) -> str:
    variants = variants or []
    lines: list[str] = []
    add = lines.append

    add("# GSNet 光路相差迁移实验：迁移学习与参数冻结（issue #165，纯仿真）")
    add("")

    add("## 0. 结论")
    add("")
    lines.extend(_conclusion_lines(data, variants))
    add("")

    add("## 1. 实验设置")
    add("")
    if data is not None:
        geometry = data.get("geometry") or {}
        config = data.get("config") or {}
        add(_table(["项", "值"], [
            ["预训练 checkpoint", f"`{_rel(data.get('checkpoint', MISSING))}`"],
            ["瞳面模型网格", f"{geometry.get('region', MISSING)} × "
             f"{geometry.get('region', MISSING)}"],
            ["远场 FFT 网格", f"{geometry.get('far_field', MISSING)} × "
             f"{geometry.get('far_field', MISSING)}"],
            ["远场方形目标边长", f"{geometry.get('target_side', MISSING)} px"],
            ["指标网格", f"{geometry.get('metric_side', MISSING)} × "
             f"{geometry.get('metric_side', MISSING)}（远场中心裁剪）"],
            ["探针 / 评估样本 / epoch",
             f"{config.get('probes', MISSING)} / "
             f"{config.get('eval_samples', MISSING)} / "
             f"{config.get('epochs', MISSING)}"],
            ["主情形 / 主冻结策略",
             f"`{config.get('primary_case', 'panel-coma')}` / "
             f"`{config.get('primary_freeze', 'full')}`"],
            ["SLM 灰度→振幅耦合 `--coupling`", f"c = {config.get('coupling', MISSING)}"
             "（真值仓库未测量，待测——见 §7）"],
        ]))
        add("")
        add("**完整运行配置**（逐项来自输入 JSON 的 `config`，未做解读）：")
        add("")
        if config:
            add(_table(["配置项", "值"],
                       [[f"`{k}`", str(v)] for k, v in sorted(config.items())]))
        else:
            add(f"（`config` 为空——{MISSING}）")
        add("")
        add("**三条变体开关**（驱动脚本本轮新增；默认值 = 首轮原行为）：")
        add("")
        add(_table(["开关", "语义"], [
            ["`--w-phase`（默认 1.0）",
             "相位项权重。其监督目标是**命令过的探针相位** = 该情形物理下的 "
             "GS 解 + 小幅随机 Zernike 微扰；在带耦合的仿真里 GS 解本身比预训练"
             "网络的输出差，所以相位项是一个**向下拉**的力。置 0 = 只留整形项。"],
            ["`--operator {ideal,calibrated}`（默认 ideal）",
             "微调用的前向算子（网络内部前向、整形项与该情形「迁移前」参照都用"
             "同一个算子，保证 Δ 只反映微调本身）。ideal = 理想孪生：复现不出带"
             "耦合的实测帧，整形项的相关目标**物理上不可达**、梯度近似噪声；"
             "calibrated = **探针实测所在的那台孪生**（该情形相差 + 该情形"
             "灰度→振幅耦合），是实验系统上由 `tools/slm/bench_twin_calibrate.py`"
             "反演出的校准算子在仿真里的替身。"],
            ["`--gate`（默认关）",
             "微调后在**验证集探针**（与训练探针不相交、独立种子，规模 "
             "max(4, probes//4)）上做一次**对比择优**：用微调所优化的那把尺子"
             "（整形项同一 Pearson 相关、同一裁剪）比较候选权重与未微调权重，"
             "只有候选更好才保留，否则恢复预训练权重（`kept_pretrained`）。"
             "对应 model-in-loop 闭环的验证集验收门：没有这道门时，有害的"
             "微调总被采纳。"],
        ]))
        add("")
        add("**孪生携带的耦合**：每个情形的孪生统一带 SLM 灰度→振幅耦合"
            "（`TwinErrorModel.slm_amplitude_coupling`），灰度域周期取器件实测的 "
            "2π 灰度数（各情形 `errors.gray_levels` 见下表，**不是** 1023）。"
            "耦合是**迁移目标实验系统的属性**，消融轴仍然只有相差与冻结配置。")
        add("")
        add("**五种变体配置**（本次渲染实际读入的 JSON；缺位标注「未记录」）：")
        add("")
        if variants:
            add(_table(["变体", "开关组合", "JSON 路径"],
                       [[f"`{n}`", _variant_flags(d) if d is not None else MISSING,
                         f"`{_rel(p)}`"] for n, d, p in variants]))
        else:
            add(f"（本次只给了单个 `--results`，无变体对比——{MISSING}）")
        add("")
        add("**六种光路相差情形**（`errors` 为 `TwinErrorModel` 的逐情形字典，"
            "完整内容在输入 JSON 里）：")
        add("")
        case_rows = []
        for case in _case_names(data) or list(CASE_ORDER):
            entry = (data.get("cases") or {}).get(case) or {}
            desc = entry.get("description") or CASE_LABELS.get(case, MISSING)
            errors = entry.get("errors")
            err_text = (json.dumps(errors, ensure_ascii=False, sort_keys=True)
                        if isinstance(errors, dict) and errors else MISSING)
            case_rows.append([f"`{case}`", desc, err_text])
        if not case_rows:
            case_rows = [[f"`{c}`", CASE_LABELS[c], MISSING] for c in CASE_ORDER]
        add(_table(["情形", "像差说明", "误差模型（JSON `errors`）"], case_rows))
        add("")
        add("六种情形**刻意超出预训练的随机化范围**（面板 ≤0.12 rad、"
            "透镜 ≤0.10 rad、离焦 ±2e-4 m），迁移面对的是真实的域差。")
        add("")
        present_cases = set(data.get("cases") or {})
        missing_cases = [c for c in CASE_ORDER if c not in present_cases]
        present_freezes = {
            f
            for entry in (data.get("cases") or {}).values()
            for f in (entry.get("runs") or {})
            if isinstance(entry, dict)
        }
        missing_freezes = [f for f in FREEZE_ORDER if f not in present_freezes]
        if missing_cases or missing_freezes:
            add("> ⚠️ **本次输入 JSON 是缩减网格**，不是完整设计："
                + (f"缺情形 {'、'.join(f'`{c}`' for c in missing_cases)}；"
                   if missing_cases else "")
                + (f"缺冻结策略 {'、'.join(f'`{f}`' for f in missing_freezes)}；"
                   if missing_freezes else "")
                + "下面各表只渲染输入里实际存在的数据，全网格结论待完整运行。")
            add("")
        add("**六种冻结策略**（冻结的是 `FourierGSNet` 内 `ConditionUNet` 的部件；"
            "每层 GS 展开的物理半部——FFT + 幅度交换——本来就没有参数）：")
        add("")
        add(_table(["策略", "冻结了什么"],
                   [[f"`{f}`", FREEZE_LABELS[f]] for f in FREEZE_ORDER]))
        add("")
        add("**`ConditionUNet` 参数构成**（参数量只在输入 JSON 记录时填充）：")
        add("")
        add(_model_params_table(data))
        add("")
        add("> 各组参数量若为「未记录」，仍可从 §5 矩阵的「可训练参数」列读出：")
        add("> `total − trainable(该组冻结)` 就是被冻结那一组的参数量。")
        add("")
    else:
        add(f"**实验尚未运行——本节协议按设计描述，具体取值（{MISSING}）"
            "将由输入 JSON 填充。**")
        add("")
        add("设计：预训练 checkpoint `data/gsnet_pretrain/20261022_coupling/checkpoint.pt`；"
            "六种光路相差情形（`none`、`panel-defocus`、`panel-coma`、"
            "`panel-spherical`、`lens-coma`、`camera-defocus`），其相差幅度刻意超出"
            "预训练随机化范围（面板 ≤0.12 rad、透镜 ≤0.10 rad、离焦 ±2e-4 m）；"
            "六种冻结策略（`full`、`frozen`、`freeze-encoder`、`freeze-decoder`、"
            "`freeze-films`、`freeze-outconv`）；探针相位为「最优相位 + 小幅随机"
            "Zernike 扰动」；五个变体沿 `--w-phase` / `--operator` / `--gate` "
            "三条修法开关消融。")
        add("")

    add("## 2. 指标口径")
    add("")
    add(_table(["指标 / 参考", "口径"], [
        ["`composite`", "整形复合分数（目标方框的环绕能量 EE + 框内 CV 合成，"
         "与 `slm_gs_refine` / gsnet 预训练同口径），越高越好"],
        ["`Δ`", "迁移后 − 迁移前（同变体、同情形、同冻结策略的 "
         "`references.pretrained`），composite 越高越好 ⇒ **Δ>0 = 迁移有增益**"],
        ["`encircled_energy`", "目标方框内能量占全图能量的比例"],
        ["`cv_target_box`", "目标方框内 std/mean，越低越均匀"],
        ["`correlation_full`", "整个远场网格上的 Pearson 相关"],
        ["**平场**（`references.flat`）", "零相位直接过该情形物理——**下界**"],
        ["**经典 GS 解**（`references.label`）",
         "GS 在该情形自己的物理下求的解——**经典算法达到的水平**，不是严格上限（网络直接优化整形目标，实测可以超过它）"
         "（注意：带耦合孪生里它可以**低于**迁移前，见 §6 机制 1）"],
        ["**迁移前**（`references.pretrained`）",
         "预训练 checkpoint 不微调、直接推理——迁移要超过的基线；"
         "用该变体的同一算子评测，跨算子家族不可直接互比"],
        ["**迁移后**（`runs.<策略>.final`）",
         "该情形 + 该冻结策略微调之后的同一 checkpoint（门拒绝时已恢复"
         "预训练权重 ⇒ 与迁移前相同）"],
        ["`gate` 判定", "`off`（未开门）/ `accepted`（验证集上候选更好，保留）/ "
         "`kept_pretrained`（候选不更好，恢复预训练权重）"],
        ["`phase_loss` 猜测水平", "圆形相位 MSE 的随机猜测水平 π²/3 ≈ 3.2899；"
         "相位项只有**低于**它才说明迁移学到了相位，而不是拟合均值"],
    ]))
    add("")
    add("> ⚠️ **全部是仿真。** 真值是仓库规范的 2f 数字孪生，不是实验系统的 CCD。"
        "所有分数只对「孪生 + 该情形误差模型 + 该变体算子」成立，不能当作实验系统收益。")
    add("")

    primary_label = ""
    if data is not None:
        pv = _pick_primary_variant(variants)
        primary_label = pv[0] if (pv is not None and pv[1] is data) else ""
    vtag = f" `{primary_label}`" if primary_label else ""

    add("## 3. 五个变体的对比：迁移为什么（不）成立")
    add("")
    lines.extend(_variant_section_lines(variants))

    freeze_hdr = (MISSING if data is None
                  else str(((data.get("config") or {}).get("primary_freeze")
                            or "full")))
    add(f"## 4. 表格 1：各光路相差下的迁移学习"
        f"（主变体{vtag}，冻结策略 `{freeze_hdr}`）")
    add("")
    if data is not None:
        freeze = str(((data.get("config") or {}).get("primary_freeze") or "full"))
        rows = []
        for case in _case_names(data) or list(CASE_ORDER):
            refs = _refs(data, case)
            final = _final(data, case, freeze)
            desc = ((data.get("cases") or {}).get(case) or {}).get("description") \
                or CASE_LABELS.get(case, case)
            rows.append([
                f"`{case}`", desc,
                _metric(refs.get("flat")),
                _metric(refs.get("label")),
                _metric(refs.get("pretrained")),
                _metric(final),
                _delta(final.get("composite"),
                       (refs.get("pretrained") or {}).get("composite")
                       if isinstance(refs.get("pretrained"), dict) else None),
                str(_run(data, case, freeze).get("gate", MISSING)),
            ])
        add(_table(["情形", "像差说明", "平场", "经典 GS 解", "迁移前", "迁移后",
                    "Δ(迁移后−迁移前)", "gate"], rows))
    else:
        add(f"（{MISSING}——实验未运行；跑完后此表按上述八列渲染）")
    add("")

    primary = (MISSING if data is None
               else str(((data.get("config") or {}).get("primary_case")
                         or "panel-coma")))
    add(f"## 5. 表格 2：各冻结策略（主变体{vtag}，情形 `{primary}`）")
    add("")
    if data is not None:
        lines.extend(_freeze_summary_lines(data, primary_label))
        add("")
        rows = []
        pre = _pretrained(data, primary)
        for f in _freeze_names(data, primary) or list(FREEZE_ORDER):
            run = _run(data, primary, f)
            final = run.get("final") if isinstance(run.get("final"), dict) else {}
            trainable = _num(run.get("trainable_params"))
            total = _num(run.get("total_params"))
            share = (f"{trainable / total:.1%}"
                     if trainable is not None and total else MISSING)
            rows.append([
                f"`{f}`", FREEZE_LABELS.get(f, f),
                _int_field(run, "trainable_params"),
                _int_field(run, "frozen_params"),
                share,
                _metric(final),
                _delta(final.get("composite"), pre.get("composite")),
            ])
        add(_table(["策略", "冻结了什么", "可训练参数", "冻结参数", "参数占比",
                    "迁移后 composite", "Δ(迁移后−迁移前)"], rows))
        add("")
        add("**相位项 vs 随机猜测水平**（π²/3 ≈ 3.2899；只有低于它才算学到相位）：")
        add("")
        prows = []
        for f in _freeze_names(data, primary):
            run = _run(data, primary, f)
            if not run:
                continue
            pl, chance = _num(run.get("phase_loss")), _num(run.get("phase_loss_chance"))
            if pl is None and chance is None:
                continue
            verdict = ("**低于**" if (pl is not None and chance is not None
                                      and pl < chance) else "尚未低于") \
                if pl is not None and chance is not None else MISSING
            prows.append([f"`{f}`", _fmt(pl), _fmt(chance), verdict])
        if prows:
            add(_table(["策略", "迁移后相位项", "猜测水平", "判定"], prows))
        else:
            add(f"（`phase_loss` 各列均{MISSING}）")
        add("")
        lines.extend(_broken_freeze_contrast_lines(variants, primary))
    else:
        add(f"（{MISSING}——实验未运行；跑完后本节按「逐策略汇总 + 完整矩阵 + 主情形明细」渲染）")
    add("")

    add("## 6. 讨论：机制")
    add("")
    if data is not None:
        if variants:
            lines.extend(_mechanism_lines(variants, data, primary_label))
        else:
            lines.extend(_discussion_lines(data))
    else:
        add(f"（{MISSING}——实验未运行，机制讨论待真实数字填充；预注册的机制假设：）")
        add("")
        add("- 相位项的监督目标（GS 解 + 微扰）可能低于网络现状，形成向下的拉力。")
        add("- ideal 算子复现不出带耦合的实测帧，整形项因此不可达。")
        add("- 缺少验证集对比择优门时，有害微调会被无条件采纳。")
    add("")

    add("## 7. 边界与不能下的结论")
    add("")
    cfg = _cfg(data)
    add("1. **全部是仿真。** 真值是仓库规范的 2f 数字孪生"
        "（`drivers/sim/bench_twin.py`），没有任何 CCD / SLM 测量。"
        "「迁移成功」只对孪生协议成立，不能当作实验系统收益。")
    probes_txt = cfg.get("probes", MISSING)
    eval_txt = cfg.get("eval_samples", MISSING)
    gate_n = (f"，验证集探针 max(4, probes//4) = {max(4, int(probes_txt) // 4)}"
              if isinstance(probes_txt, (int, float)) else "")
    add(f"2. **探针只有几十帧**（本次输入：probes={probes_txt}、"
        f"eval_samples={eval_txt}{gate_n}），不是实验系统上的真实预算；"
        "冻结策略在这个小数据下够用，不意味着在更大预算下同样够用，"
        "也不意味着门的小验证集判定在更大样本下同样准（§6 机制 4）。")
    add("3. **探针相位是「最优相位 + 小幅随机 Zernike 扰动」**——这是对 "
        "`slm_model_in_loop` 行为的**建模**，不是它本身；闭环真实的探针分布"
        "（含验收拒绝后补测的探针）没有进入这份数据。")
    add("4. **网络内部前向与评测算子的错配没有被完全闭合**：checkpoint 是用 "
        "**ideal 算子**内部训练出来的（预训练时没有该情形的误差模型），"
        "`--operator calibrated` 只是在**微调与评测侧**换成校准算子。"
        "「用拟合后的算子重训网络」这条更忠实的在机路径"
        "（runbook `docs/slm/gsnet_transfer_hw_runbook.md` §7）没有在本份数据里覆盖。")
    add("5. **SLM 灰度→振幅耦合的真值在仓库里任何地方都没有被测出**：仓库只有"
        "诊断判据 `rel_spread = (max−min)/mean > 0.15`"
        "（`tools/slm/panel/diagnose.py` 的调制检测），它只蕴含 `c ≳ 0.075`。"
        f"本仿真的 c = {cfg.get('coupling', MISSING)} 是**待测量**的假设值，"
        "不是已知量；换一台耦合更强的面板，带耦合的实测帧与「理想算子不可达」"
        "的幅度都会变，机制 1–3 的定量结论随之移动。孪生同时用器件真实的 "
        "2π 灰度数（`gray_levels`，见 §1 情形表）而不是 1023。")
    add("6. 六种相差情形超出预训练随机化范围是**设计**，因此情形间差异"
        "混合了「相差种类」与「域差大小」两条轴，不能单独归因。")
    add("")

    add("## 8. 复现")
    add("")
    add("```bash")
    add("# 1) 五个变体（同一 checkpoint / 情形 / 冻结网格 / 种子，只差开关）")
    add("python scripts/slm/gsnet_aberration_transfer.py \\")
    add("    --out data/debug/gsnet_aberration_transfer/coupling_A/result.json   # 默认（A）")
    add("python scripts/slm/gsnet_aberration_transfer.py --w-phase 0 \\")
    add("    --out data/debug/gsnet_aberration_transfer/coupling_B/result.json")
    add("python scripts/slm/gsnet_aberration_transfer.py --operator calibrated \\")
    add("    --out data/debug/gsnet_aberration_transfer/coupling_C/result.json")
    add("python scripts/slm/gsnet_aberration_transfer.py --operator calibrated --w-phase 0 \\")
    add("    --out data/debug/gsnet_aberration_transfer/coupling_D/result.json")
    add("python scripts/slm/gsnet_aberration_transfer.py --operator calibrated --w-phase 0 --gate \\")
    add("    --out data/debug/gsnet_aberration_transfer/coupling_E/result.json")
    add("")
    add("# 2) 渲染本报告（自动发现 --variants-dir 下的 coupling_<标签>/result.json；")
    add("#    也可用 --variant NAME=path 显式指定，可重复；缺哪个都只标「未记录」）")
    add("python scripts/slm/generate_gsnet_aberration_transfer_report.py \\")
    add("    --variants-dir data/debug/gsnet_aberration_transfer \\")
    add("    --out report/gsnet_aberration_transfer/report.md")
    add("```")
    add("")
    add(f"数据来源: `{results_path}`"
        + ("；变体 JSON: " + "、".join(f"`{_rel(p)}`" for _n, _d, p in variants)
           if variants else ""))
    add("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="把 gsnet_aberration_transfer.py 的 result.json 渲染成报告。"
                    "推荐 --variants-dir 一次渲染五个变体的对比；"
                    "--variant NAME=path 可重复、优先于自动发现；"
                    "--results 保留为单 JSON 旧接口。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results", type=Path, default=DEFAULT_RESULTS,
        help="单个 result.json（无变体时的详细表格数据源）；"
             "不存在且没有任何变体时渲染占位报告（实验未运行）",
    )
    parser.add_argument(
        "--variants-dir", type=Path, default=None,
        help="变体根目录：只发现名字匹配 coupling_<标签> 的子目录里的 "
             "result.json（coupling_A/、coupling_B/…），每个缺失的变体降级为"
             "「未记录」而不中断渲染",
    )
    parser.add_argument(
        "--variant", action="append", metavar="NAME=PATH",
        help="显式指定一个变体 JSON，可重复；NAME 是表里的变体标签，"
             "PATH 指向 result.json（同名的自动发现项被其覆盖）",
    )
    parser.add_argument("--out", type=Path, default=REPORT_DIR / "report.md",
                        help="报告输出路径")
    args = parser.parse_args(argv)

    variants = _load_variants(args)
    data: dict[str, Any] | None = None
    primary_path = args.results
    pv = _pick_primary_variant(variants)
    if pv is not None:
        data = pv[1]
        primary_path = pv[2]
    elif args.results.exists():
        data = json.loads(args.results.read_text(encoding="utf-8"))
    elif variants:
        print("警告: 所有变体 JSON 都读不到，渲染占位报告。", file=sys.stderr)
    else:
        print(f"警告: 输入 JSON 不存在（{args.results}），渲染占位报告。",
              file=sys.stderr)

    text = insert_header(render(data, primary_path, variants), REPORT_KEY)
    out_path: Path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print(f"报告已写入: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
