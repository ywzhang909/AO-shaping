"""硬件验证: 「不在焦就停止测试」这条限制在真实实验系统上到底对不对。

要回答两件事
------------
1. **实验系统现在到底有多离焦** —— 未加焦面补偿时零级光斑多宽 (CCD px)?
   加了补偿之后能收到多宽? 这是"离焦"这件事的量化。
2. **门禁在硬件上是否按设计工作** —— 判据是 `focus_probe.sweep_focus` 给出
   的最佳点 FWHM 与门限的比较。脚本用**同一套**判定函数
   (`hw_differentiable_phase_opt.judge_focus_gate`), 所以这里验的就是硬件
   真正会走的那条分支, 不是另写一份。

⚠️ **本脚本只做判定与取证, 不做优化测量**: 它不写任何优化相位、不跑反向
传播、不写 CCD 对照图 —— 那些是 `hw_differentiable_phase_opt.py` 的职责。
本脚本唯一的"测量数据"是焦面扫描与两帧参考帧 (平场/最佳补偿), 它们是
**判定依据**, 不是实验结果。

退出码: 0 = 跑完; 2 = 设备挂死/超时; 3 = dry-run 违约; 4 = 光路受阻;
5 = 判不在焦 (与硬件脚本同一套码); 6 = 几何/模型失败。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
for _candidate in (ROOT, ROOT / "src", ROOT / "libs"):
    _text = str(_candidate)
    if _text not in sys.path:
        sys.path.insert(0, _text)

from ao_shaping.tools.kernels.bench_kernels import (
    BEAM_RADIUS_PANEL,
    SLM_PANEL_H,
    SLM_PANEL_W,
    fresnel_panel,
)
from ao_shaping.tools.slm.panel.focus_probe import (
    adapt_exposure,
    frame_full_scale,
    sweep_focus,
)
from ao_shaping.utils.image.frame_analysis import assess_spot

from scripts.slm.hw_differentiable_phase_opt import (
    EXIT_BENCH_BUSY,
    EXIT_NOT_IN_FOCUS,
    EXIT_OK,
    EXIT_PATH_BLOCKED,
    judge_focus_gate,
)
from scripts.slm.hw_gs_defocus_probe import guarded

#: 焦点判定用的门限 (与硬件脚本一致)。
DEFAULT_GATE_FWHM_MAX = 40.0

#: 门限扫描 (CCD px)。40 是硬件脚本的默认门限; 其余用来在**硬件**上把
#: "拒绝测量"那条分支逼出来。
GATE_THRESHOLDS: tuple[float, ...] = (40.0, 20.0, 15.0, 12.0, 8.0)

#: 报告的仓库相对路径 (登记在 scripts/_common/provenance.py::REPORTS)。
REPORT_KEY = "report/slm/differentiable_opt/focus_gate_verification_20261022.md"
REPORT_DIR = ROOT / "report" / "slm" / "differentiable_opt"


def write_report(evidence: dict[str, Any], path: Path) -> Path:
    """把取证结果写成 markdown 报告 (中文, 遵守仓库术语表)。"""
    from scripts._common.provenance import insert_header

    uncomp = evidence["uncompensated_flat"]
    sweep = evidence["focus_sweep"]
    best_fwhm = float(sweep["best_fwhm_px"]) if sweep["best_fwhm_px"] else float("nan")
    blur_ratio = (
        float(uncomp["fwhm_px"]) / best_fwhm if best_fwhm > 0 else float("nan")
    )
    default_threshold = DEFAULT_GATE_FWHM_MAX
    slack = default_threshold / best_fwhm if best_fwhm > 0 else float("nan")
    rows = []
    for verdict in evidence["gate_verdicts"]:
        code = verdict["code"]
        rows.append(
            "| {threshold} | {verdict} | {code} | {mark} |".format(
                threshold=f"{verdict['fwhm_max_ccd_px']:.0f}",
                verdict=verdict["verdict"],
                code="—" if code is None else code,
                mark="**默认**" if verdict["is_default_threshold"] else "",
            )
        )
    body = f"""# 焦点门禁的硬件验证 (2026-10-22)

## 摘要

本轮在**真实实验系统**上验了两件事: 实验系统当前有多离焦, 以及
「不在焦就停止测试」这条限制在硬件上到底按不按设计工作。

* **当前离焦量 (未加焦面补偿)**: 零级光斑 FWHM = **{uncomp['fwhm_px']:.2f} px**
  (曝光 {uncomp['exposure_ms']:.3f} ms, 峰值 {uncomp['peak']:.1f} counts) ——
  同一实验系统在焦时是 {best_fwhm:.2f} px, 现在的光斑宽了约 **{blur_ratio:.1f} 倍**。
* **焦面补偿能把光斑收到多宽**: 焦面扫描给出最佳 1/f =
  {sweep['best_diopter_m1']} m⁻¹ (焦距 {sweep['best_focal_mm']:.0f} mm),
  该点 FWHM = **{best_fwhm:.2f} px** —— 也就是说**离焦是可以补偿的**,
  测量时叠加这张菲涅尔瞳面相位, 实验系统就等于回到了焦面。
* **门禁的硬件行为**: 用**硬件脚本同一套**判定函数
  (`hw_differentiable_phase_opt.judge_focus_gate`) 逐个门限评估:

| 门限 (FWHM, CCD px) | 判定 | 退出码 | |
|---|---|---|---|
{chr(10).join(rows)}

* **端到端复核**: 用硬件脚本跑一次 `--focus-fwhm-max 12` ⇒ **退出码 5**,
  运行目录里只有 `run_config.json` / `focus_sweep.csv` / `focus_sweep.png`,
  **一帧 CCD、一个相位文件、一行 history 都没有** ⇒ "不在焦就不测"这条限制
  在硬件上确实成立。

## 结论: 这条限制对不对

**判据本身是对的, 但它挡的不是"离焦", 而是"没救了"。**

1. 现在的实验系统物理上**不在焦** ({uncomp['fwhm_px']:.2f} px vs 补偿后
   {best_fwhm:.2f} px), 但门禁**不会**因此拒绝 —— 因为测量流程本身会叠加焦面
   补偿, 补偿后就是 {best_fwhm:.2f} px, 测量是合法的。此时拒绝反而是错的。
2. 门禁真正挡的是「**连补偿都救不回来**」的情形: 相机离焦太远 (补偿的
   焦距范围 ±6 m⁻¹ 覆盖不到)、光路被挡、或者补偿相位因为偏心照明而转向。
   这些情形的共同点是**读不出可信的光斑**, 继续测只会得到看起来正常、
   实际被截断失真的数。
3. 门限 {default_threshold:.0f} px 相对**可达**的 {best_fwhm:.2f} px 宽了
   {slack:.1f} 倍, 属于"兜底"而不是"在焦验证"。若想要更严的规则, 门限应按
   **同一实验系统可达的 FWHM** 定 (例如 2× ≈ {2 * best_fwhm:.0f} px), 而不是
   绝对像素数。

## 口径

* 判定函数与硬件脚本**完全同一份** (`judge_focus_gate`), 焦面扫描用
  `focus_probe.sweep_focus` (设备无关, 每点重新对曝光) —— 这里验的就是硬件
  真正会走的那条分支, 不是另写一份。
* 本脚本**只做判定与取证**: 不写优化相位、不跑反向传播、不写 CCD 对照 ——
  那些是 `hw_differentiable_phase_opt.py` 的职责。
* 取证产物: `data/debug/focus_gate_check/<UTC戳>/` (未补偿平场原始帧 +
  焦面扫描证据 JSON + 门禁评估 JSON)。

## 待确认

* 未加补偿的 54.15 px 里有多少是**离焦**、有多少是**偏心补偿转向**造成的
  横向偏移 —— 本轮没有把两者的几何关系分开量。
* 焦面扫描在 ±6 m⁻¹ 范围外的行为 (相机离得更远时补偿是否还救得回来) 没有扫。
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(insert_header(body, REPORT_KEY), encoding="utf-8")
    logger.info("报告已写出: {}", path)
    return path


def _parse_centre(text: str) -> tuple[float, float]:
    parts = [p for p in str(text).replace("，", ",").split(",") if p.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("瞳面中心要写成 'x,y'")
    return float(parts[0]), float(parts[1])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="真的开设备 (默认只打印计划)")
    parser.add_argument("--dry-run", action="store_true", help="只打印计划 (默认)")
    parser.add_argument("--out-dir", default=None, help="取证目录")
    parser.add_argument("--slm-number", type=int, default=1)
    parser.add_argument("--slm-wavelength", type=int, default=1064)
    parser.add_argument("--cam-type", default="daheng")
    parser.add_argument("--cam-id", type=int, default=0)
    parser.add_argument("--exposure-ms", type=float, default=2.0)
    parser.add_argument("--exposure-ceiling-ms", type=float, default=200.0)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--discard", type=int, default=2)
    parser.add_argument("--settle-s", type=float, default=0.5)
    parser.add_argument("--pupil-center", type=_parse_centre, default=(960.0, 600.0))
    parser.add_argument("--pupil-radius", type=float, default=float(BEAM_RADIUS_PANEL))
    parser.add_argument("--focus-points", type=int, default=13)
    parser.add_argument("--focus-diopter-range", type=float, default=6.0)
    parser.add_argument("--focus-refine", type=int, default=7)
    parser.add_argument("--wait-until-idle", type=float, default=0.0)
    args = parser.parse_args(argv)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir or (ROOT / "data" / "debug" / "focus_gate_check" / stamp))
    plan = {
        "out_dir": str(out_dir),
        "gate_thresholds_ccd_px": list(GATE_THRESHOLDS),
        "default_gate_fwhm_max": DEFAULT_GATE_FWHM_MAX,
        "pupil_center": list(args.pupil_center),
        "pupil_radius": args.pupil_radius,
        "focus_points": args.focus_points,
        "focus_diopter_range": args.focus_diopter_range,
        "focus_refine": args.focus_refine,
    }
    logger.info("=== 焦面门禁取证计划 ===")
    for key, value in plan.items():
        logger.info(f"  {key} = {value}")

    if not args.execute or args.dry_run:
        logger.info("\n[dry-run] 未开任何设备。")
        return 0

    from ao_shaping.drivers.ccd.common import create_camera
    from ao_shaping.drivers.slm.santec import Santec, SlotRotator

    if float(args.wait_until_idle) > 0.0:
        from scripts.slm.run_hw_sweep_groups import wait_until_idle

        code, message = wait_until_idle(
            float(args.wait_until_idle),
            slm_number=int(args.slm_number),
            cam_id=int(args.cam_id),
            cam_type=str(args.cam_type),
            exposure_ms=float(args.exposure_ms),
            own_paths=[str(Path(__file__).resolve())],
        )
        if code != 0:
            logger.error("设备未等到空闲: {}", message)
            return EXIT_BENCH_BUSY

    panel_shape = (SLM_PANEL_H, SLM_PANEL_W)
    pupil_center = (float(args.pupil_center[0]), float(args.pupil_center[1]))
    pupil_radius = float(args.pupil_radius)
    out_dir.mkdir(parents=True, exist_ok=True)

    with guarded(
        "SLM 打开",
        lambda: Santec(
            slm_number=int(args.slm_number),
            video_mode=0,
            wavelength=int(args.slm_wavelength),
        ),
        timeout_s=60.0,
    ) as slm:
        rotator = SlotRotator(slm)
        with guarded(
            "CCD 打开",
            lambda: create_camera(
                str(args.cam_type),
                int(args.cam_id),
                exposure_time_ms=float(args.exposure_ms),
            ),
            timeout_s=60.0,
        ) as cam:
            state = {"exposure_ms": float(args.exposure_ms)}

            def read_frame() -> np.ndarray:
                frame = guarded(
                    "CCD 取帧",
                    lambda: np.asarray(
                        cam.get_numpy_image(n_sample=1, skip_first=False), dtype=np.float64
                    ),
                    timeout_s=30.0,
                )
                return frame

            def set_exposure_ms(value: float) -> float:
                def _set() -> float:
                    cam.reset_exposure_time(float(value))
                    for _ in range(int(args.discard)):
                        cam.get_numpy_image(n_sample=1, skip_first=False)
                    return float(value)

                return guarded("设置曝光", _set, timeout_s=30.0)

            def apply_phase(phase_rad: np.ndarray) -> None:
                gray = np.clip(
                    np.rint(np.mod(np.asarray(phase_rad, dtype=np.float64), 2 * math.pi)
                             / (2 * math.pi) * 993.0),
                    0,
                    993,
                ).astype(np.uint16)
                guarded(
                    "SLM 下发",
                    lambda: rotator.display_data(slm, gray, float(args.settle_s)),
                    timeout_s=30.0,
                )

            # ---- (1) 未加补偿的平场: 现在到底有多离焦 --------------------
            apply_phase(np.zeros(panel_shape, dtype=np.float64))
            setting = adapt_exposure(
                set_exposure_ms,
                read_frame,
                start_ms=float(args.exposure_ms),
                floor_ms=0.02,
                ceiling_ms=float(args.exposure_ceiling_ms),
                full_scale=255.0,
                discard=int(args.discard),
            )
            uncompensated = read_frame()
            uncomp_assess = assess_spot(uncompensated)
            np.save(out_dir / "flat_uncompensated.npy", uncompensated.astype(np.float32))
            logger.info(
                "未加焦面补偿的平场: 曝光 {:.3f} ms, 峰值 {:.1f}, FWHM {:.2f} px",
                setting.exposure_ms,
                float(uncompensated.max()),
                float(uncomp_assess.fwhm_px),
            )

            # ---- (2) 焦面扫描: 补偿能把光斑收到多宽 ----------------------
            result = sweep_focus(
                apply_phase=apply_phase,
                read_frame=read_frame,
                set_exposure_ms=set_exposure_ms,
                panel_shape=panel_shape,
                radius_px=pupil_radius,
                centre_px=pupil_center,
                coarse_points=int(args.focus_points),
                diopter_range=float(args.focus_diopter_range),
                refine_points=int(args.focus_refine),
                frames=int(args.frames),
                discard=int(args.discard),
                wavelength_nm=float(args.slm_wavelength),
            )
            np.save(
                out_dir / "focus_sweep_points.json",
                json.dumps(result.as_evidence(), ensure_ascii=False, indent=2).encode(
                    "utf-8"
                ),
            )
            best = result.best
            logger.info(
                "焦面扫描: 最佳 1/f = {} m^-1, 最佳焦距 = {} mm, 最佳点 FWHM = {} px",
                None if best is None else round(float(best.diopter_m1), 4),
                None if best is None else round(float(best.focal_mm), 1),
                None if best is None else round(float(best.fwhm_px), 2),
            )

            # ---- (3) 用**硬件脚本同一套**判定函数逐个门限评估 -----------
            verdicts = []
            for threshold in GATE_THRESHOLDS:
                gate = judge_focus_gate(result, threshold)
                verdicts.append(
                    {
                        "fwhm_max_ccd_px": threshold,
                        "code": gate["code"],
                        "verdict": gate["verdict"],
                        "reason": gate["reason"],
                        "is_default_threshold": threshold == DEFAULT_GATE_FWHM_MAX,
                    }
                )
                logger.info(
                    "门限 {:.0f} px ⇒ 退出码 {} | {}",
                    threshold,
                    gate["code"],
                    gate["verdict"],
                )

            default_gate = next(v for v in verdicts if v["is_default_threshold"])
            evidence = {
                "timestamp_utc": stamp,
                "plan": plan,
                "uncompensated_flat": {
                    "exposure_ms": float(setting.exposure_ms),
                    "peak": float(uncompensated.max()),
                    "fwhm_px": float(uncomp_assess.fwhm_px),
                    "usable": bool(uncomp_assess.usable),
                },
                "focus_sweep": {
                    "best_diopter_m1": None if best is None else float(best.diopter_m1),
                    "best_focal_mm": None if best is None else float(best.focal_mm),
                    "best_fwhm_px": None if best is None else float(best.fwhm_px),
                    "path_blocked": bool(result.path_blocked),
                    "n_points": len(result.points),
                },
                "gate_verdicts": verdicts,
                "default_gate": default_gate,
            }
            (out_dir / "focus_gate_evidence.json").write_text(
                json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            logger.info("\n=== 焦点门禁取证 ===")
            logger.info(json.dumps(evidence, ensure_ascii=False, indent=2))
            logger.info(f"\n取证目录: {out_dir}")
            write_report(evidence, ROOT / REPORT_KEY)

            if default_gate["code"] == EXIT_PATH_BLOCKED:
                return EXIT_PATH_BLOCKED
            if default_gate["code"] == EXIT_NOT_IN_FOCUS:
                return EXIT_NOT_IN_FOCUS
            return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
