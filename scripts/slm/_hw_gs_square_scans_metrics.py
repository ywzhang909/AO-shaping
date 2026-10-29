"""GS 方扫报告的**判据与统计量**。

这一层是纯函数：给出一列数，算 Spearman / Pearson / 单调性 / 离群倍数 /
对数指数，以及"这行要不要打 `†` `‡` `※` 标记"的判据。

刻意不依赖 IO 与出图层：判据本身是**口径**，口径要能被单独读和单独测，
不能和"从哪个文件读来的""画在哪张图上"缠在一起。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Sequence
import numpy as np


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

from scripts.slm._hw_gs_square_scans_model import (
    DARK_FRAME_FRACTION,
    FIELD_BASELINE_FLAT,
    FIELD_RETEST_FLAT,
    OUTLIER_CV_RATIO,
    RetestStage,
    Run,
    SATURATION_COUNTS,
    Stage,
)

# re-export: analysis/figures/sections/report 四个 facade 从这里取 pearson/spearman;
# 实现本体在 ml.zernike.eval_stats (统计唯一来源)，勿删本 import。
from ml.zernike.eval_stats import pearson, spearman

def score_terms(pib: float, cv: float) -> tuple[float, float]:
    """``score = 0.5·PIB + 0.5·1/(1+CV)`` 的两项 (与采集脚本同一口径)。"""
    return 0.5 * pib, 0.5 / (1.0 + cv)

def monotonic_report(values: Sequence[float], *, increasing: bool = True) -> list[int]:
    """返回**破坏**单调性的下标 (i 表示 values[i] 相对 values[i-1] 反向)。"""
    out: list[int] = []
    for i in range(1, len(values)):
        if increasing and values[i] <= values[i - 1]:
            out.append(i)
        if not increasing and values[i] >= values[i - 1]:
            out.append(i)
    return out

def flag_rows(run: Run, stage: Stage) -> list[str]:
    """逐行的旗标: ``†`` settle 超时, ``‡`` 帧电平异常, ``※`` CV 离群, ``§`` 框出画。

    ``※`` 刻意用**邻行**做参照而不是只用段中位数: 边长段的 CV 本身随边长单调爬升
    (ρ ≈ 0.996), 若只比中位数就会把趋势的尾巴 (1700/1800 px) 当成离群点。判据是
    「比中位数高 >1.5× **且** 比邻行中位数高 >1.5× **且** 邻行本身不离中位数太远」,
    于是只有真正从平稳背景里跳出来的点会被标 —— 本轮的 ``n_max = 18``、``period = 40``。
    """
    med_peak = stage.median("frame_peak")
    med_cv = stage.median("cv")
    cvs = stage.col("cv")
    out: list[str] = []
    for i, row in enumerate(stage.rows):
        marks: list[str] = []
        if run.settle_for(stage.name, i) is not None:
            marks.append("†")
        peak = float(row["frame_peak"])
        if math.isfinite(med_peak) and med_peak > 0 and peak < DARK_FRAME_FRACTION * med_peak:
            marks.append("‡")
        cv = float(row["cv"])
        neighbours = [cvs[j] for j in (i - 1, i + 1) if 0 <= j < len(cvs)]
        if (
            neighbours
            and math.isfinite(med_cv) and med_cv > 0
            and cv > OUTLIER_CV_RATIO * med_cv
            and cv > OUTLIER_CV_RATIO * float(np.median(neighbours))
            and float(np.median(neighbours)) < 1.2 * med_cv
        ):
            marks.append("※")
        if float(row["box_clipped_px"]) > 0:
            marks.append("§")
        out.append("".join(marks))
    return out

def saturation_flag(row: dict[str, str]) -> str:
    peak = float(row["frame_peak"])
    if peak >= SATURATION_COUNTS:
        return "饱和"
    if peak >= 0.95 * SATURATION_COUNTS:
        return "接近饱和"
    return "未饱和"

def retest_power_ratio() -> tuple[float, float, float]:
    """由**现场记录**算功率比: ``(基线 counts/ms, 复测 counts/ms, 比值)``。"""
    base = FIELD_BASELINE_FLAT.counts_per_ms()
    now = FIELD_RETEST_FLAT.counts_per_ms()
    return base, now, base / now

def _neighbour_deviation(cvs: Sequence[float], index: int) -> float:
    """第 ``index`` 行相对左右邻居中位数的相对偏差 (无邻居返回 nan)。"""
    neighbours = [cvs[j] for j in (index - 1, index + 1) if 0 <= j < len(cvs)]
    if not neighbours:
        return float("nan")
    ref = float(np.median(neighbours))
    return cvs[index] / ref - 1.0 if ref else float("nan")

def _loglog_exponent(xs: Sequence[float], ys: Sequence[float]) -> float:
    """两点 log-log 斜率 (``y ∝ x^-α`` 里的 α); 用两端点, 不拟合。"""
    if len(xs) < 2 or min(xs) <= 0 or min(ys) <= 0:
        return float("nan")
    return -math.log(ys[-1] / ys[0]) / math.log(xs[-1] / xs[0])

def column_contract(
    baseline: Stage, retest: RetestStage, needed: Sequence[str]
) -> list[tuple[str, str, str]]:
    """逐列说明「这一列两批都有吗」—— 复测的列集合与基线**不同**, 不能假设相同。"""
    rows: list[tuple[str, str, str]] = []
    for key in needed:
        rows.append((
            f"`{key}`",
            "有" if key in baseline.headers else "**缺**",
            "有" if key in retest.headers else "**缺**",
        ))
    return rows
