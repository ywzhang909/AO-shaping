"""光路变化后的自动校准自检：哪些存量标定还能用、哪些必须重测/重拟合。

2026-10-23 的场景：把 CCD 移到焦平面之后，此前所有**与离焦强相关**的标定都失效：

* 冻结模型的 ``defocus_diopter``（拟合时约 −0.645 m⁻¹ ⇒ 等效焦距 ~1.55 m）描述的是
  **旧光路**的离焦；CCD 到焦面后它应当≈0，否则模型的远场形状会系统性偏；
* ``focus.json``（``best_focal_mm`` ~1783 mm、增益 31×）是**旧离焦状态**下的最佳补偿；
* 模型采样窗（``camera_xy``）的中心建在旧零点上（实测已偏 72 px）；
* 目标尺寸标定（``signature`` + 有效窗口）是在旧光路下扫出来的。

2026-11-07 更新：原先写死的 2026-10-22/23 路径已被后续布局取代（模型 checkpoint 由
``result.json`` 改为 ``frozen_parameters.json``，目录也搬到了 ``data/debug/slm_transfer/``），
照旧常量跑会直接崩。现在四条路径都是 CLI 可覆盖，默认指向**当日仍在用**的产物：

* ``--checkpoint`` 会话 ``20261107_135252`` 的末轮拟合 ``fit.round2``（与已部署的
  ``data/slm_models/<SLM 序列号>/latest/frozen_parameters.json`` 逐字节相同）；
* ``--focus`` 当日焦点扫描 ``20261107_focus_scan/focus.json``；
* ``--calibration`` 目标尺寸标定 ``data/calibration/target_size_calibration.json``；
* ``--run`` 会话目录 ``20261107_135252``（提供实测帧以定零点）。

要审旧数据就把对应参数指过去，例如旧 checkpoint 目录
``data/debug/physical_forward_defocus/refit_after_focus_20261023_145446``（``result.json`` 布局）。

判定按「产物记录时刻 vs 本次会话起点」给出：会话之后记录的 ⇒ ``reuse``，之前记录的 ⇒
``stale``（需重测/重拟合）；模型离焦项另按物理判据 ``|defocus_diopter| > 0.05 m⁻¹``
判 ``stale``（CCD 在焦面时它应当≈0）。产物缺失不抛异常，记 ``missing`` 行并继续。

本脚本只读存量产物、离线给出逐项判定与处置建议；能离线重建的（采样窗）直接给出新值，
需要设备的（焦点、模型重拟合、尺寸重扫）明确标 ``needs-device``。

删除条件 (AGENTS.md 「scripts/ 目录」规则 3): 本脚本**不是**一次性探针 —— 每次
光路变化后都要重跑 (2026-10-23 CCD 移到焦平面后跑过一轮, 光路此后仍在变动:
2026-10-27 无 GS 计划、2026-10-28 多形状多尺度 campaign)。2026-10-23 那轮的
判定结论未被单独引用为 report/ 条目, 规则 3① 不满足。当存量标定的审计被自动
门禁接管 (如 ``src/ao_shaping/tools/slm/transfer/validate.py`` 的验收逻辑或
provenance 门禁覆盖同类判定)、光路稳定到不再需要逐项自检时, 可删。
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]

# 默认审计对象（2026-11-07 当前光路下的存量产物）。
CHECKPOINT = ROOT / "data/debug/slm_transfer/20261107_135252/fit.round2"
FOCUS = ROOT / "data/debug/slm_transfer/20261107_focus_scan/focus.json"
CALIBRATION = ROOT / "data/calibration/target_size_calibration.json"
RUN = ROOT / "data/debug/slm_transfer/20261107_135252"

#: 离焦项的物理判据：CCD 已在焦面时，模型的离焦项应当≈0（m⁻¹）。
DEFOCUS_STALE_ABS = 0.05

#: 采样窗中心与实测零点的容差（px）：超过即算偏了。
WINDOW_CENTRE_TOLERANCE_PX = 5.0


def _row(item: str, stored: str, expected: str, verdict: str, action: str) -> dict:
    return dict(item=item, stored=stored, expected_after_change=expected, verdict=verdict, action=action)


def _resolve_checkpoint(path: Path) -> tuple[Path | None, str]:
    """定位模型 checkpoint 文件，返回 ``(文件, 布局名)``。

    新布局：目录下有 ``frozen_parameters.json``（``defocus_diopter`` 在顶层）；
    旧布局：目录下只有 ``result.json``（``defocus_diopter`` 在 ``parameters`` 里）。
    直接给文件路径时按该文件处理。
    """
    if path.is_file():
        return path, path.name
    if path.is_dir():
        for name in ("frozen_parameters.json", "result.json"):
            candidate = path / name
            if candidate.is_file():
                return candidate, name
    return None, ""


def _checkpoint_diopter(checkpoint_file: Path) -> float | None:
    """读 ``defocus_diopter``；两种布局都支持，字段缺失返回 ``None``。"""
    payload = json.loads(checkpoint_file.read_text(encoding="utf-8"))
    if "defocus_diopter" in payload:
        return float(payload["defocus_diopter"])
    parameters = payload.get("parameters", {})
    if "defocus_diopter" in parameters:
        return float(parameters["defocus_diopter"])
    return None


def _recorded_epoch(payload: dict, *keys: str) -> float | None:
    """从产物里取「记录时刻」为 epoch 秒；无时间戳字段返回 ``None``。"""
    for key in keys:
        raw = payload.get(key)
        if isinstance(raw, str) and raw:
            try:
                parsed = datetime.fromisoformat(raw)
            except ValueError:
                continue
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.timestamp()
    return None


def _reference_epoch(run_dir: Path, checkpoint_file: Path | None) -> float | None:
    """本次光路的参考时刻：会话 ``session.json`` 的 ``timestamp_utc``。

    退化顺序：会话 ``manifest_checksums.json`` 的 ``generated_utc``（旧布局的 run 目录）→
    checkpoint 落盘时刻 → ``None``（此时一律判 ``unknown``）。
    """
    session = run_dir / "session.json"
    if session.is_file():
        epoch = _recorded_epoch(json.loads(session.read_text(encoding="utf-8")), "timestamp_utc")
        if epoch is not None:
            return epoch
    manifest = run_dir / "manifest_checksums.json"
    if manifest.is_file():
        epoch = _recorded_epoch(json.loads(manifest.read_text(encoding="utf-8")), "generated_utc")
        if epoch is not None:
            return epoch
    if checkpoint_file is not None and checkpoint_file.is_file():
        return checkpoint_file.stat().st_mtime
    return None


def _date_verdict(recorded_epoch: float | None, reference_epoch: float | None) -> str:
    """按记录时刻判 ``reuse`` / ``stale``；时刻缺失判 ``unknown``。"""
    if recorded_epoch is None or reference_epoch is None:
        return "unknown"
    return "reuse" if recorded_epoch >= reference_epoch else "stale"


def _load_measured_zero(run_dir: Path) -> tuple[tuple[float, float] | None, str]:
    """取实测零点 (col, row)，返回 ``(零点, 来源说明)``。

    优先用会话自带的 ``optical_probe.json``（专测的平场零级，比成形光斑的
    峰值位置更接近真实零点）；没有就退到实测帧的峰值位置，兼容两种布局：
    旧的 ``stage_e/rep0/p0_A1_f*.npy``（多帧 npy）与新的
    ``anchored*/capture_*.npz``（``raw_frames`` 为 (帧, 行, 列)）。
    """
    probe = run_dir / "optical_probe.json"
    if probe.is_file():
        payload = json.loads(probe.read_text(encoding="utf-8"))
        zero = payload.get("current", {}).get("zero_xy")
        if zero and len(zero) == 2:
            return (float(zero[0]), float(zero[1])), "optical_probe.json 的 current.zero_xy"

    legacy = sorted((run_dir / "stage_e" / "rep0").glob("p0_A1_f*.npy"))
    if legacy:
        stack = [np.load(p).astype(np.float64) for p in legacy]
        frame = np.mean(stack, axis=0)
        row, col = np.unravel_index(int(np.argmax(frame)), frame.shape)
        return (float(col), float(row)), f"{legacy[0].name}（{len(legacy)} 帧平均）的峰值"
    modern = sorted(run_dir.glob("anchored*/capture_*.npz"))
    if modern:
        with np.load(modern[0]) as payload:
            raw = payload["raw_frames"].astype(np.float64)
        frame = raw.mean(axis=0)
        row, col = np.unravel_index(int(np.argmax(frame)), frame.shape)
        return (float(col), float(row)), f"{modern[0].name}（{raw.shape[0]} 帧平均）的峰值"
    return None, ""


def _window_centre(inputs_npz: Path) -> tuple[float, float] | None:
    """模型采样窗 ``camera_xy`` 的中心 (col, row)；文件缺失返回 ``None``。"""
    if not inputs_npz.is_file():
        return None
    with np.load(inputs_npz) as payload:
        xy = payload["camera_xy"]
    return (float(xy[..., 0].min() + xy[..., 0].max()) / 2.0,
            float(xy[..., 1].min() + xy[..., 1].max()) / 2.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT,
                        help="模型 checkpoint 目录（frozen_parameters.json 新布局 / result.json 旧布局）")
    parser.add_argument("--focus", type=Path, default=FOCUS, help="焦点扫描产物 focus.json")
    parser.add_argument("--calibration", type=Path, default=CALIBRATION, help="目标尺寸标定 json")
    parser.add_argument("--run", type=Path, default=RUN, help="会话目录（提供实测帧以定零点）")
    parser.add_argument("--check", action="store_true", help="存在需要重做的项时返回非零退出码")
    parser.add_argument("--json", type=Path, default=None, help="把报告写到该路径")
    args = parser.parse_args()

    rows: list[dict] = []
    checkpoint_file, checkpoint_layout = _resolve_checkpoint(args.checkpoint)
    reference_epoch = _reference_epoch(args.run, checkpoint_file)
    logger.info("审计对象：")
    logger.info(f"  checkpoint  = {args.checkpoint} → {checkpoint_file or '缺失'}")
    logger.info(f"  focus       = {args.focus} → {'在' if args.focus.is_file() else '缺失'}")
    logger.info(f"  calibration = {args.calibration} → {'在' if args.calibration.is_file() else '缺失'}")
    logger.info(f"  run         = {args.run}")

    # ---- 1. 模型的离焦项 ------------------------------------------------------
    diopter: float | None = None
    if checkpoint_file is None:
        logger.warning(f"checkpoint 缺失（未找到 frozen_parameters.json / result.json）：{args.checkpoint}")
        rows.append(_row("模型离焦项 defocus_diopter", f"缺失（{args.checkpoint}）",
                         "≈ 0（CCD 已在焦面）", "missing",
                         "needs-device：重跑拟合并给出 checkpoint"))
    else:
        diopter = _checkpoint_diopter(checkpoint_file)
        if diopter is None:
            logger.warning(f"checkpoint 缺 defocus_diopter 字段：{checkpoint_file}")
            rows.append(_row("模型离焦项 defocus_diopter",
                             f"缺 defocus_diopter 字段（{checkpoint_file.name}）",
                             "≈ 0（CCD 已在焦面）", "missing",
                             "needs-device：重跑拟合并给出 checkpoint"))
        else:
            implied_focal_mm = 1000.0 / abs(diopter) if diopter else float("inf")
            stale = abs(diopter) > DEFOCUS_STALE_ABS
            rows.append(_row(
                "模型离焦项 defocus_diopter",
                f"{diopter:+.4f} m⁻¹（等效焦距 {implied_focal_mm:.0f} mm，{checkpoint_layout}）",
                "≈ 0（CCD 已在焦面）",
                "stale" if stale else "reuse",
                "needs-device：在焦面上重采探针并重拟合（先做 B± 方向检验再用于优化）"
                if stale else "可复用：已在当前光路上重拟合（离焦项 ≈ 0）"))

    # ---- 2. 焦面补偿标定 ------------------------------------------------------
    focus_payload: dict | None = None
    if args.focus.is_file():
        focus_payload = json.loads(args.focus.read_text(encoding="utf-8"))
    else:
        logger.warning(f"焦点扫描产物缺失：{args.focus}")
    if focus_payload is not None:
        focus = focus_payload.get("focus", focus_payload)
        best = float(focus.get("best_focal_mm", float("nan")))
        gain = float(focus.get("gain_vs_flat", float("nan")))
        verdict = _date_verdict(_recorded_epoch(focus_payload, "recorded_utc"), reference_epoch)
        rows.append(_row(
            "焦面补偿 focus.json", f"best_focal_mm={best:.1f}，增益 {gain:.1f}×",
            "补偿量应显著变小（原为旧离焦的补偿）",
            verdict,
            "needs-device：重跑焦点扫描（probe_session_focus / focus_preflight）"
            if verdict != "reuse" else "可复用：当前光路上的焦点扫描"))
        implied = 1000.0 / abs(best) if best and np.isfinite(best) else float("nan")
        if diopter is not None and implied and np.isfinite(implied):
            rows.append(_row(
                "两者一致性（模型 |d| vs 1/f）", f"|d|/（1/f）= {abs(diopter) / implied:.2f}",
                "光路变化后两者应同时趋近 0", "info",
                "若变化后两者仍相差 >30% ⇒ 说明其一未重测"))
        else:
            rows.append(_row("两者一致性（模型 |d| vs 1/f）", "不可比",
                             "光路变化后两者应同时趋近 0", "unknown",
                             "补齐模型 checkpoint 与焦点扫描后再比对"))
    else:
        rows.append(_row("焦面补偿 focus.json", f"缺失（{args.focus}）", "—", "missing",
                         "needs-device：重新标定"))

    # ---- 3. 模型采样窗 --------------------------------------------------------
    inputs_npz = (checkpoint_file.parent / "inputs.npz") if checkpoint_file is not None else None
    window_centre = _window_centre(inputs_npz) if inputs_npz is not None else None
    spot, measured_note = _load_measured_zero(args.run)
    if window_centre is None:
        rows.append(_row("模型采样窗中心", f"无 camera_xy（{args.checkpoint}）", "—", "missing",
                         "needs-device：给出带 inputs.npz 的 checkpoint"))
    elif spot is None:
        rows.append(_row("模型采样窗中心", "无实测帧可对照", "—", "unknown", "needs-device：抓一帧定零点"))
    else:
        offset = float(np.hypot(spot[0] - window_centre[0], spot[1] - window_centre[1]))
        rows.append(_row(
            "模型采样窗中心 vs 实测零点", f"窗 ({window_centre[0]:.0f}, {window_centre[1]:.0f}) / "
            f"零点 ({spot[0]:.0f}, {spot[1]:.0f})，偏 {offset:.0f} px（实测零点取自 {measured_note}）",
            "窗心 = 本次实测零点",
            "stale" if offset > WINDOW_CENTRE_TOLERANCE_PX else "reuse",
            "可离线重建：recentre_window(spot, half_span=..., step=2) 后作为 camera_xy"))

    # ---- 4. 目标尺寸标定 ------------------------------------------------------
    if args.calibration.is_file():
        calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
        verdict = _date_verdict(_recorded_epoch(calibration, "recorded_utc"), reference_epoch)
        rows.append(_row(
            "目标尺寸标定", f"signature={calibration.get('signature', '?')[:16]}，"
            f"best={calibration.get('best_size', float('nan')):.1f} px",
            "签名不变（λ/f/N/p 未变）但**有效窗口**依赖旧离焦",
            verdict,
            "needs-device：重拟合后重扫 40–240 px（有效性门 + HDR 度量）"
            if verdict != "reuse" else "可复用：当前光路上的尺寸扫描"))
    else:
        logger.warning(f"目标尺寸标定缺失：{args.calibration}")
        rows.append(_row("目标尺寸标定", f"缺失（{args.calibration}）", "—", "missing",
                         "needs-device：先跑扫描"))

    # ---- 汇总 ----------------------------------------------------------------
    stale = [r for r in rows if r["verdict"] in {"stale", "missing"}]
    logger.info("光路变化后的自动校准自检")
    logger.info("=" * 100)
    width = max(len(r["item"]) for r in rows)
    for item in rows:
        logger.info(f"  {item['item']:<{width}} | {item['verdict']:<7} | 存量: {item['stored']}")
        logger.info(f"  {'':<{width}} | {'':<7} | 期望: {item['expected_after_change']}")
        logger.info(f"  {'':<{width}} | {'':<7} | 处置: {item['action']}")
    logger.info("-" * 100)
    logger.info(f"  需要重做: {len(stale)} / {len(rows)} 项；其中可离线重建的只有采样窗，"
          f"其余需设备（焦点扫描、模型重拟合、尺寸重扫）。")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(dict(run=str(args.run), items=rows, stale=len(stale),
                                             paths=dict(checkpoint=str(args.checkpoint),
                                                        focus=str(args.focus),
                                                        calibration=str(args.calibration),
                                                        run=str(args.run),
                                                        checkpoint_file=str(checkpoint_file or ""),
                                                        checkpoint_layout=checkpoint_layout)),
                                        indent=2, ensure_ascii=False), encoding="utf-8")
    if args.check and stale:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
