"""GS 方扫报告的**收尾自检**。

报告渲染完还要自证两件事才算能入库：

* :func:`check_residuals` —— §13.8 说"相位残差是这么重算出来的"，那就要拿
  存盘相位对一遍，对不上说明落盘路径被人改过；
* :func:`check_invariants` —— score 分解、框面积恒等式、CSV↔JSON 一致、
  日志↔run_config 一致，四条恒等式全过才算"表里的数字没被改写过"。

和 :mod:`_hw_gs_square_scans_analysis` 的分界：**那一层算结论，这一层验算**。
验算逻辑单独放，是因为它们读的是"已算完的结论"，放在一起会让分析函数看起来
像是自己验自己。
"""

from __future__ import annotations

import sys
from pathlib import Path


# 与 facade 同在 scripts/slm/ 下，parents[2] 算出的仓库根一致。
# 用**逐个 insert** 而不是 for 循环：scripts/ 那层的守卫
# (tests/ao_shaping/scripts/test_script_file_form_imports.py) 靠解析
# `sys.path.insert` 的字面参数判断「这个脚本能不能以文件方式启动」，
# for 循环里的循环变量它解析不了（会整文件跳过，等于没检查）。
# src 在 scripts 之前：scripts/ml 会遮蔽 src/ml。
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from scripts.slm._hw_gs_square_scans_io import (
    load_json,
)
from scripts.slm._hw_gs_square_scans_model import (
    AREA_IDENTITY_ATOL,
    EXPOSURE_MATCH_ATOL,
    Findings,
    RESIDUAL_RECON_ATOL,
    RESIDUAL_RMS_RTOL,
    ResidualSet,
    RetestRun,
    Run,
    SCORE_IDENTITY_ATOL,
)

def check_residuals(residuals: ResidualSet | None) -> list[str]:
    """§13.8 的自检: 重算残差 RMS 与记录值一致、重算相位与存盘相位逐点一致。

    **两种模式 (生成 / ``--check``) 都跑** —— 它只读产物, 与画不画图无关。
    返回问题清单 (空 = 通过); 每条都同时打印重算值与记录值。
    """
    problems: list[str] = []
    if residuals is None:
        return problems
    for panel in residuals.panels:
        rel = abs(panel.rms_recomputed - panel.rms_recorded) / max(abs(panel.rms_recorded), 1e-12)
        if rel >= RESIDUAL_RMS_RTOL:
            problems.append(
                f"n_max={panel.n_max}: 重算残差 RMS {panel.rms_recomputed:.9f} rad != "
                f"metrics.json 记录 {panel.rms_recorded:.9f} rad (相对误差 {rel:.2%}, "
                f"容差 {RESIDUAL_RMS_RTOL:.0%})"
            )
        if panel.recon_delta >= RESIDUAL_RECON_ATOL:
            problems.append(
                f"n_max={panel.n_max}: 重算相位与 {panel.config}/pupil_phase.npy 最大偏差 "
                f"{panel.recon_delta:.3e} (重算 vs 存盘; 容差 {RESIDUAL_RECON_ATOL:.0e})"
            )
    return problems

def check_invariants(
    run: Run, f: Findings, retest: RetestRun | None = None
) -> list[str]:
    """与「渲染」无关的结构性检查 —— 数据自身必须自洽。"""
    problems: list[str] = []

    if retest is not None:
        for name, stage in retest.stages.items():
            for i, row in enumerate(stage.rows):
                pib = float(row["pib"])
                cv = float(row["cv"])
                expect = 0.5 * pib + 0.5 / (1.0 + cv)
                if abs(expect - float(row["score"])) > SCORE_IDENTITY_ATOL:
                    problems.append(
                        f"复测 {name} 第 {i} 行: score {row['score']} != "
                        f"0.5·PIB + 0.5/(1+CV) = {expect}"
                    )
                side = float(row["box_side_px"])
                if abs(side * side - float(row["box_area_px"])
                       - float(row["box_clipped_px"])) > AREA_IDENTITY_ATOL:
                    problems.append(f"复测 {name} 第 {i} 行: 框面积恒等式不成立")
                if abs(float(row["exposure_ms_readback"]) - retest.exposure_ms) > EXPOSURE_MATCH_ATOL:
                    problems.append(
                        f"复测 {name} 第 {i} 行: 读回曝光 {row['exposure_ms_readback']} "
                        f"!= 生效 {retest.exposure_ms}"
                    )
            # 段级 run_config (有就核对; v1 那批没有 —— 缺文件与缺 `plan` 都必须容忍)
            cfg_path = retest.root / name / "run_config.json"
            if cfg_path.exists():
                cfg = load_json(cfg_path)
                if abs(float(cfg.get("exposure_ms_working", float("nan")))
                       - retest.exposure_ms) > EXPOSURE_MATCH_ATOL:
                    problems.append(f"复测 {name}/run_config.json 的生效曝光与 preflight 不一致")

    for name, stage in run.stages.items():
        for i, row in enumerate(stage.rows):
            pib = float(row["pib"])
            cv = float(row["cv"])
            score = float(row["score"])
            expect = 0.5 * pib + 0.5 / (1.0 + cv)
            if abs(expect - score) > SCORE_IDENTITY_ATOL:
                problems.append(
                    f"{name} 第 {i} 行: score {score} != 0.5·PIB + 0.5/(1+CV) = {expect}"
                )
            side = float(row["box_side_px"])
            clipped = float(row["box_clipped_px"])
            area = float(row["box_area_px"])
            if abs(side * side - area - clipped) > AREA_IDENTITY_ATOL:
                problems.append(
                    f"{name} 第 {i} 行: 框面积恒等式不成立 "
                    f"({side}² − {area} != {clipped})"
                )
            if abs(float(row["exposure_ms_readback"]) - run.exposure_ms) > EXPOSURE_MATCH_ATOL:
                problems.append(
                    f"{name} 第 {i} 行: 读回曝光 {row['exposure_ms_readback']} "
                    f"!= 生效 {run.exposure_ms}"
                )

    # 单调性结论必须与正文一致
    if f.pib_violations:
        problems.append(f"PIB 并非严格单调 (反向步 {f.pib_violations}) —— 正文需改写")
    peak = max(f.score)
    if sum(1 for v in f.score if v == peak) > 1:
        problems.append("score 的最大值不唯一 (有并列) —— 选点/叙述都不可判")

    # 日志 ↔ run_config
    for name, logged, actual, ok in run.log_crosscheck:
        if not ok:
            problems.append(f"日志与 run_config 不一致: {name} 日志={logged} 配置={actual}")
    return problems
