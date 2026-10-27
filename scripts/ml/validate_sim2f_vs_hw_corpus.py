"""Measure how similar the ``ml.sim2f`` twin is to real 2f-bench pickles.

What this answers
=================
``runs/sim2f_bench/README.md`` line 138 makes "geometry identifiable" gate G0-4
a hard prerequisite of the whole benchmark lane::

    G0-4 几何可辨识 | `ncc(实测 net 帧, 孪生预测 net 帧) ≥ 0.5` | 映射/ROI/标度错, 先修几何再测

That sentence is a *claim about the twin*, and until now it was only ever
checked against the 24-sample session inside ``runs/sim2f_bench/session/``.
This script checks it against the much larger, real, already-collected corpus in
``data/debug/`` -- one record at a time, phase in, measured net frame vs twin
predicted net frame out.

It also answers a question the gate cannot: **is the twin's output at the right
scale at all?** The measured 64x64 window and the twin's 256x256 camera frame are
not the same sampling, so a fixed geometric mapping has to be chosen; the
mapping's scale is swept (see below) and the result is reported as a curve, not
as a single number.

What is being compared, and what is deliberately not
====================================================
Only **scale-free** metrics are allowed, because the twin is known to be wrong in
absolute photometry (``runs/sim2f_bench/README.md`` line 140 records PIB/EE at
-79 % and FWHM at +26 %):

* ``ncc`` -- zero-mean (Pearson) correlation of the two 64x64 frames;
* normalized L1 -- ``sum|a/sum(a) - b/sum(b)|``, i.e. twice the total variation of
  two unit-sum distributions, in ``[0, 2]``. This is the same shape as the
  ``L_shape`` metric ``Sim2f``'s teacher optimises, so a "similar" verdict here
  means "close on the objective the twin is used with".

Absolute Strehl / PIB / EE / FWHM / peak / total energy are **not** computed as
verdict evidence. The one width-based diagnostic is a *ratio* of two widths
measured the same way on the same phase, which is scale-free by construction; it
is labelled a diagnostic and is not part of the verdict.

No second implementation of anything
====================================
* the corpus is indexed by ``ml.hwdataset.build_hw_index``;
* the phase is reduced by ``ml.hwdataset.records.Materialiser`` (same coherent
  block mean, same panel ROI, same Zernike engine as training);
* the measured frame is cropped by ``ml.hwdataset.transforms.farfield_frame_to_grid``;
* the twin is ``ml.sim2f.optics.Sim2f`` with ``Sim2fConfig.calibrated()``.

The only new code here is the geometry mapping between the twin's camera frame
and the measured 64x64 window, and the two metrics.

The geometry mapping, pre-registered
====================================
``k`` = twin CCD pixels per measured pixel. Measured index ``i`` (0..63) samples
the twin at ``128 + (i - 31.5) * k`` on both axes, bilinearly. ``128`` is the
twin's optical axis (verified ``argmax`` at ``(128, 128)`` for a flat phase);
``31.5`` is where ``farfield_frame_to_grid`` puts the measured 0-order
(``top = centre - grid // 2`` puts it at index 32, i.e. 31.5 in pixel-centre
coordinates). Larger ``k`` **shrinks** the twin pattern inside the 64-px window.

``k`` is expected to be **1.0**: the corpus sidecar declares the same 2.2 um
camera pitch as ``Sim2fConfig.ccd_pixel_um``. So the sweep is a check of the
twin's focal-plane scale, not a free knob to be fitted.

Read-only, no hardware
======================
``data/debug/`` is opened read-only (the index unpickles, nothing is written),
and the twin runs on CPU with the detector disabled -- no ``--backend hw`` path
exists in this script.

Outputs
=======
``report/sim2f_bench/twin_vs_hw_corpus/20261027/report.md`` plus figures and JSON
artefacts, all written by :func:`main`. Every number in the markdown comes from
this run; the pre-registration block is a module constant so it cannot be edited
after seeing the numbers.

Usage
=====
    python scripts/ml/validate_sim2f_vs_hw_corpus.py
    python scripts/ml/validate_sim2f_vs_hw_corpus.py --limit-files 4   # smoke
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[2]
# Order matters: `src` lands in front of the repo root so `ml` resolves to
# `src/ml`. `scripts/` is never put on the path at all -- it contains a
# `scripts/ml/` folder that would shadow `src/ml` (documented trap, see
# `tests/ao_shaping/scripts/test_script_file_form_imports.py`).
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from ml.hwdataset import build_hw_index
from ml.hwdataset.records import Materialiser, MaterialiserConfig
from ml.hwdataset.transforms import farfield_frame_to_grid
from ml.sim2f.config import Sim2fConfig
from ml.sim2f.optics import Sim2f
from scripts._common import fmt_general, markdown_table, savefig

# ---------------------------------------------------------------------------
# Where things live
# ---------------------------------------------------------------------------

#: Registry key of the report this script writes (``scripts/_common/provenance.py``).
REPORT_KEY = "report/sim2f_bench/twin_vs_hw_corpus/20261027/report.md"
OUT_DIR = ROOT / "report" / "sim2f_bench" / "twin_vs_hw_corpus" / "20261027"
FIG_DIR = OUT_DIR / "figures"
ART_DIR = OUT_DIR / "artifacts"

#: The corpus, and the date prefix inside it. The dumps are named
#: ``20261025_173026`` -- **no dashes**; a ``2026-10-25*`` glob matches nothing.
CORPUS_ROOT = ROOT / "data" / "debug" / "model_in_loop_hw_collect"
CORPUS_PATTERN = "20261025*"

# ---------------------------------------------------------------------------
# Geometry and thresholds (pre-registered -- see PREREGISTRATION)
# ---------------------------------------------------------------------------

#: Side of the measured window: what ``farfield_frame_to_grid`` returns here.
GRID_MEASURED = 64
#: Side of the phase grid fed to the twin: ``Sim2fConfig.grid``, and 450 px of
#: panel radius x 2 = 900 px span, the twin's own ``region_span_px``.
GRID_PHASE = 128
#: Panel ROI half-width. 450 px is the illumination radius the corpus sidecar
#: declares (``zernike_radius=200`` is the Zernike aperture inside it) and what
#: ``ml.sim2f.bench.PUPIL_RADIUS_PX`` uses, so phase and twin describe one pupil.
PANEL_RADIUS_PX = 450.0
#: The twin's optical axis, in twin camera pixels.
TWIN_AXIS_PX = 128.0
#: Where ``farfield_frame_to_grid`` puts the measured 0-order, in pixel-centre
#: coordinates (``top = centre - grid // 2`` -> index 32 -> 31.5).
MEASURED_ANCHOR_PX = (GRID_MEASURED - 1) / 2.0

#: The pre-registered 1-D sweep. Widened from the first plan ``[0.5, 2.0]`` after
#: a 40-record geometry probe was still rising at 2.0; the upper bound is 4.0
#: because ``64 * 4 = 256`` twin px is the whole twin camera frame.
K_GRID: tuple[float, ...] = (
    0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0,
)
#: The scale the two pixel pitches imply. G0-4 has no scaling freedom, so its
#: answer is evaluated here, not at the swept optimum.
K_APRIORI = 1.0

#: G0-4's own number, reused verbatim instead of inventing a new bar.
NCC_BAR = 0.5
#: ``sum|a'-b'| <= 0.5`` on unit-sum frames, i.e. total variation <= 0.25.
L1_BAR = 0.5
#: Fewer records than this is "insufficient" no matter what the numbers say.
MIN_RECORDS = 10
#: How much lower than the optimum both neighbours must be for the optimum to
#: count as interior (absolute ncc).
INTERIOR_MARGIN = 0.01
#: Integer-shift search radius, measured pixels.
SHIFT_RADIUS_PX = 3

#: Printed before any statistic exists, embedded verbatim in the report, and
#: dumped into the JSON. Do not move it below the measurements.
PREREGISTRATION = """\
预登记判据 (写于任何统计量被计算之前; 之后一个字都不许改, 更不许放宽)

== 语料与口径 ==
1. 语料: `data/debug/model_in_loop_hw_collect/20261025*` (54 个 pkl, 312 条记录)。选它的理由:
   它是这批实验产物里唯一同时含 `_phase` (1200x1920, 弧度) 与 `_img` (整幅 CCD) 且体量可承受的族,
   采集日 (2026-10-25) 晚于孪生标定日 (2026-10-23) ⇒ 时间上是验证集样本。只读, 不写不删不改名。
2. 相位: `ml.hwdataset.Materialiser` 把 `_phase` 相干块平均到 128x128 (瞳面中心 (960,600),
   半径 450 px), 得到 (cos, sin)。与孪生 128 格 / 900 px 同口径。
3. 实测帧: 同一个 materialiser 取同一 pickle 的 LRU 里的 `_img`, 再走
   `farfield_frame_to_grid(..., 64, "raw")` —— 以实测帧 argmax (3x3 去尖峰) 为锚的 64x64 窗口,
   不缩放、不归一化。
4. 孪生: `Sim2f(Sim2fConfig.calibrated())`, device=cpu, `forward(..., detector=False)` ⇒ 256x256 强度帧。
   相位以 (cos, sin) 相量喂入 (孪生自己归一到单位幅度), 避免在多帧相干平均过的相位上再做 arctan2 往返。
   不做任何逐样本的缩放 / 归一化 / 峰值对齐。
5. 度量只有两种, 都与绝对量无关: ncc = 零均值皮尔逊相关; 归一化 L1 = Σ|a/Σa − b/Σb| ∈ [0,2]
   (= 2 x 全变差)。禁止用绝对 Strehl / PIB / EE / FWHM / 峰值 / 总能量作判据。
6. 几何映射: k = 孪生 CCD 像素 / 实测像素。实测第 i 个索引 (i=0..63) 在孪生上取
   `128 + (i − 31.5)·k` (两轴同式), 双线性插值, 越界截到边缘像素。k 越大 => 孪生图案被缩小。
   `128` 是孪生光学轴 (平场 argmax 实测恰在此), `31.5` 是 `farfield_frame_to_grid` 放置 0 级的位置。
7. 扫描网格 (预登记, 12 点): k ∈ {0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0}。
   上界取 4.0 因为 64x4 = 256 = 孪生相机整幅。先验预期 k = 1.0 —— 侧车声明的相机像元 2.2 µm 与
   `Sim2fConfig.ccd_pixel_um` 相同, 故该扫描是对**标度**的检验, 不是自由旋钮。
8. 验证集协议: 最优点 k* 只在偶数序号记录上选 (中位 ncc 最大; 并列时取离 k=1.0 最近者);
   判定指标只在奇数序号记录上报告。

== 判定 ==
9. 几何可辨识 G: n >= 10, 且 k* 不在网格端点, 且 k* 两侧相邻点都比 k* 低 >= 0.01, 且
   全语料 k* 与奇数验证集 k* 相差 <= 一个网格步长。任一不满足 => G 未辨识, 并指名是哪一条。
10. 度量判据 M(k): 在该 k 上 中位 ncc >= 0.5 且 中位归一化 L1 <= 0.5 (⇔ 全变差 <= 0.25)。
    ncc 的 0.5 直接沿用已经预注册的 G0-4 门限 (`runs/sim2f_bench/README.md:138`), 不新造阈值。
11. 三值结论:
    similar      iff M(k*) 成立 且 G 可辨识;
    not_similar  iff M(k*) 不成立 且 (G 可辨识 或 扫描的**每一个** k 上 M 都不成立
                 —— 此时几何无法救回结论);
    insufficient 其余 (即 M 的成立与否取决于 k*, 而 k* 又不可辨识; 或 n < 10)。
12. 任何一条阈值都不因实测结果而调整。未达到就如实记为 not_similar, 并给诊断, 不找理由。

== 附属诊断 (只解释, 不参与判定) ==
13. 扫描整条 ncc 曲线与整条 L1 曲线 (两个半集各自报告)。
14. 分组中位: mode_id (0 无像差 / 2 离焦) x coefficient (0/1), 以及 probe (0/1/2)。
15. k* 处整数平移搜索 |Δ| <= 3 实测像素: 若最优平移非零, 说明 0 级锚点在成型帧上不等于真 0 级。
16. 标定版 (Sim2fConfig.calibrated()) 与裸版 (Sim2fConfig()) 孪生在同一 k* 处对照。
17. 光斑宽度比 (实测 / 孪生, **同一相位**): 只作宽度量级提示, 绝不以孪生平场作参照。
"""

#: Appended **after** the first full run, and printed next to the block above so a
#: reader sees exactly what changed. The block above is never edited.
AMENDMENT = """\
预登记缺陷与加固 (第一次全量运行之后追加; 上面的第 1-17 条一字未改)

缺陷 —— 第一次全量运行才暴露, 属**语料构成**, 与任何度量结果无关:
  a. 312 条记录只含 18 个不同的 (相位, 帧) 对: 54 个 dump 反复存同一批
     6 个指令条件 x 3 个测量轮次, 同一个对被存了 6-36 次 (实测分布见报告 §2.1)。
     按记录算 n 会把这个样本平均放大 17 倍, 中位数于是被复制次数而不是物理条件加权。
  b. 第 8 条的「偶数/奇数序号」在这批 dump 的文件布局下**与族完全混杂**: 每个 dump
     前 3 条是 (mode_id=0, coefficient=0), 后 3 条是 (mode_id=2, coefficient=1),
     而每份 6 条是偶数 ⇒ 全部无像差记录必落偶数位, 全部离焦记录必落奇数位。
     首轮运行实测证实: 偶数 156 条全为 mode_id=0, 奇数 156 条全为 mode_id=2。
     ⇒ 那不是验证集, 而是把两个不同族当成两半来比。

加固 (只收紧, 不放宽任何阈值):
  1. 统计单位改为**去重后的 (相位, 帧) 对**: 用相位格 (cos, sin) + 实测帧的内容哈希
     判定同一性 —— 不用度量值, 否则是循环论证。第 9/10/11 条的 n 一律指去重后的个数。
  2. 第 8 条的验证集协议**替换**为「逐轮一致性」: 同一指令条件下 3 个轮次各自给出整条
     ncc 曲线与最优点, 要求三轮的 k* 与全体 k* 相差都不超过一个网格步长。这比按序号
     对半分更对题 —— 它检验的正是「换一次测量, 最优点还成立吗」。
  3. 阈值 (ncc >= 0.5, L1 <= 0.5, 内部性 0.01, n >= 10) 一个都没动。
  4. 判定规则 11 的三值映射不变。
"""


# ---------------------------------------------------------------------------
# Metrics (scale-free only)
# ---------------------------------------------------------------------------

def zero_mean_ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation of two frames, flattened.

    Args:
        a: First frame, any shape.
        b: Second frame, same size.

    Returns:
        Correlation in ``[-1, 1]``, or NaN when either side is constant (a
        constant frame has no zeroth moment to divide by, and reporting 0.0
        there would look like a measured decorrelation).
    """
    left = np.asarray(a, dtype=np.float64).ravel()
    right = np.asarray(b, dtype=np.float64).ravel()
    left = left - left.mean()
    right = right - right.mean()
    denom = math.sqrt(float(left @ left) * float(right @ right))
    if not np.isfinite(denom) or denom <= 0.0:
        return float("nan")
    return float((left @ right) / denom)


def normalised_l1(a: np.ndarray, b: np.ndarray) -> float:
    """``sum|a/sum(a) - b/sum(b)|`` -- twice the total variation, in ``[0, 2]``.

    Unit-sum normalisation is what makes this comparable to the twin's own
    ``L_shape`` objective and immune to the known absolute-photometry error.

    Args:
        a: First frame, any shape.
        b: Second frame, same size.

    Returns:
        The normalised L1 distance, or NaN when either frame sums to <= 0.
    """
    left = np.asarray(a, dtype=np.float64)
    right = np.asarray(b, dtype=np.float64)
    sum_left = float(left.sum())
    sum_right = float(right.sum())
    if not (np.isfinite(sum_left) and np.isfinite(sum_right)):
        return float("nan")
    if sum_left <= 0.0 or sum_right <= 0.0:
        return float("nan")
    return float(np.abs(left / sum_left - right / sum_right).sum())


def sample_twin(
    frame: np.ndarray,
    k: float,
    *,
    center_y: float = TWIN_AXIS_PX,
    center_x: float = TWIN_AXIS_PX,
    grid: int = GRID_MEASURED,
    anchor: float = MEASURED_ANCHOR_PX,
) -> np.ndarray:
    """Resample the twin camera frame onto the measured window's pixel grid.

    Measured index ``i`` samples the twin at ``center + (i - anchor) * k`` on each
    axis, bilinearly. ``k > 1`` therefore covers more twin pixels per measured
    pixel, i.e. **shrinks** the twin pattern.

    Args:
        frame: Twin camera frame, 2-D.
        k: Twin CCD pixels per measured pixel (the swept quantity).
        center_y: Twin row the measured anchor maps to.
        center_x: Twin column the measured anchor maps to.
        grid: Output side.
        anchor: Measured index that sits on ``center`` (31.5: pixel centres).

    Returns:
        ``(grid, grid) float64``, no rescaling of intensity.
    """
    values = np.asarray(frame, dtype=np.float64)
    height, width = values.shape
    rows = center_y + (np.arange(grid) - anchor) * float(k)
    cols = center_x + (np.arange(grid) - anchor) * float(k)
    rows = np.clip(rows, 0.0, height - 1.0)
    cols = np.clip(cols, 0.0, width - 1.0)
    row0 = np.clip(np.floor(rows).astype(int), 0, height - 2)
    col0 = np.clip(np.floor(cols).astype(int), 0, width - 2)
    weight_row = np.clip(rows - row0, 0.0, 1.0)
    weight_col = np.clip(cols - col0, 0.0, 1.0)

    f00 = values[np.ix_(row0, col0)]
    f01 = values[np.ix_(row0, col0 + 1)]
    f10 = values[np.ix_(row0 + 1, col0)]
    f11 = values[np.ix_(row0 + 1, col0 + 1)]
    top = f00 * (1.0 - weight_col) + f01 * weight_col
    bottom = f10 * (1.0 - weight_col) + f11 * weight_col
    return top * (1.0 - weight_row)[:, None] + bottom * weight_row[:, None]


def fwhm_px(frame: np.ndarray) -> float:
    """Row/column half-maximum width through the brightest pixel, geometric mean.

    A *width*, used only inside a measured-vs-twin ratio on the same phase (see
    the module docstring on why no absolute width is a verdict input).

    Args:
        frame: 2-D frame.

    Returns:
        ``sqrt(width_row * width_col)`` in pixels, or NaN when the frame has no
        positive signal above its median.
    """
    values = np.asarray(frame, dtype=np.float64)
    values = np.clip(values - float(np.median(values)), 0.0, None)
    if values.size == 0 or not np.isfinite(values).all():
        return float("nan")
    peak = float(values.max())
    if peak <= 0.0:
        return float("nan")
    row, col = np.unravel_index(int(np.argmax(values)), values.shape)
    half = 0.5 * peak
    across = np.where(values[row, :] >= half)[0]
    down = np.where(values[:, col] >= half)[0]
    if across.size == 0 or down.size == 0:
        return float("nan")
    width_row = float(across[-1] - across[0] + 1)
    width_col = float(down[-1] - down[0] + 1)
    return math.sqrt(width_row * width_col)


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def describe(values: Iterable[float]) -> dict[str, float | int]:
    """Count, median, quartiles, IQR, min and max of the finite values.

    Args:
        values: Raw per-record values; non-finite entries are dropped and the
            survivor count is reported, so a NaN-heavy metric is visible instead
            of being silently averaged away.

    Returns:
        ``{"n", "median", "q25", "q75", "iqr", "min", "max"}``; NaN statistics
        when nothing finite survives.
    """
    array = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if array.size == 0:
        return {
            "n": 0,
            "median": float("nan"),
            "q25": float("nan"),
            "q75": float("nan"),
            "iqr": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    q25, median, q75 = np.percentile(array, [25.0, 50.0, 75.0])
    return {
        "n": int(array.size),
        "median": float(median),
        "q25": float(q25),
        "q75": float(q75),
        "iqr": float(q75 - q25),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _pick_k(medians: Sequence[float]) -> int:
    """Index of the sweep maximum; ties resolved toward the a-priori scale.

    Args:
        medians: Median ncc per point of :data:`K_GRID`.

    Returns:
        Index into :data:`K_GRID`. The a-priori tie-break matters because a
        plateau must not be resolved by float noise in favour of a scale nothing
        predicts.
    """
    best = float("-inf")
    chosen = 0
    for index, value in enumerate(medians):
        if not np.isfinite(value):
            continue
        if value > best:
            best, chosen = value, index
            continue
        if value == best and abs(K_GRID[index] - K_APRIORI) < abs(
            K_GRID[chosen] - K_APRIORI
        ):
            chosen = index
    return chosen


def interior_check(medians: Sequence[float]) -> dict[str, Any]:
    """Is the sweep maximum an interior point whose neighbours are clearly lower?

    Args:
        medians: Median ncc per point of :data:`K_GRID`.

    Returns:
        ``{"index", "k", "value", "interior", "reason", "left", "right"}``; the
        two neighbour values are reported even when the test fails, because
        "flat" and "at the edge" are different diagnoses.
    """
    index = _pick_k(medians)
    value = float(medians[index])
    last = len(K_GRID) - 1
    if index == 0 or index == last:
        return {
            "index": index,
            "k": K_GRID[index],
            "value": value,
            "interior": False,
            "reason": "端点",
            "left": None,
            "right": None,
        }
    left = float(medians[index - 1])
    right = float(medians[index + 1])
    drops = (value - left, value - right)
    interior = all(drop >= INTERIOR_MARGIN for drop in drops)
    reason = (
        "内部且两侧都低"
        if interior
        else f"两侧未都低 {INTERIOR_MARGIN:g} (左 {value - left:+.4f}, 右 {value - right:+.4f})"
    )
    return {
        "index": index,
        "k": K_GRID[index],
        "value": value,
        "interior": interior,
        "reason": reason,
        "left": left,
        "right": right,
    }


# ---------------------------------------------------------------------------
# Corpus and measurement
# ---------------------------------------------------------------------------

def _relative(path: Path) -> str:
    """Repo-relative POSIX path, for a report that travels."""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:  # pragma: no cover - a corpus outside the repo
        return str(path)


def _meta(payload: Any, name: str) -> int | float | None:
    """One scalar out of a payload record, or ``None`` when absent or not numeric.

    Args:
        payload: The unpickled record mapping.
        name: Field to read (``mode_id``, ``coefficient``, ``probe``).

    Returns:
        The value as a plain ``int``/``float``, or ``None``. Bools are rejected
        on purpose: ``True`` is a valid ``int`` in Python and would silently
        become ``mode_id = 1``, a family that does not exist.
    """
    if not hasattr(payload, "get"):
        return None
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        return None
    return int(value) if isinstance(value, (int, np.integer)) else float(value)


def measure_corpus(limit_files: int | None, progress_every: int = 25) -> dict[str, Any]:
    """Index the corpus, measure every record, then collapse exact duplicates.

    Two passes over the same in-memory arrays, so the pickles are read once:

    1. per record -- phase, measured frame, twin frame, and the full ncc/L1 sweep
       across :data:`K_GRID`;
    2. per **distinct** record, once ``k*`` is chosen -- the sampled twin at
       ``k*``, the integer-shift diagnostic, the width ratio and the
       uncalibrated twin.

    Duplicates are found by a content hash of the phase grid **and** the measured
    frame, never by metric equality: the corpus re-stores the same measurements in
    many dumps (measured: 312 records -> 18 distinct pairs), and de-duplicating on
    the metric would be circular.

    Each distinct pair also carries a ``round`` label -- its ordinal within its
    ``(mode_id, coefficient, probe)`` group -- because that is this corpus's
    natural replication axis: three measurement rounds of six commanded
    conditions. The per-round curves are the consistency check that replaces the
    degenerate index-parity split (see :data:`AMENDMENT`).

    Args:
        limit_files: Cap on pickles, applied after sorting (deterministic smoke
            runs). ``None`` means the whole corpus.
        progress_every: Print a progress line every N records; ``<= 0`` silences.

    Returns:
        ``{"index", "sidecar", "per_record", "distinct", "duplication",
        "selection", "k_star", "distinct_medians", "round_medians",
        "round_picks", "timings", "config"}``.
    """
    started = time.perf_counter()
    index = build_hw_index(
        [str(CORPUS_ROOT / CORPUS_PATTERN)],
        limit_files=limit_files,
        progress_every=0,
    )
    indexed = time.perf_counter()

    materialiser = Materialiser(
        MaterialiserConfig(grid=GRID_PHASE, panel_radius=PANEL_RADIUS_PX),
        use_cache=False,
    )
    bench = Sim2f(Sim2fConfig.calibrated(), device="cpu")
    bare_bench = Sim2f(Sim2fConfig(), device="cpu")
    built = time.perf_counter()

    per_record: list[dict[str, Any]] = []
    for position, ref in enumerate(index.records):
        cos_grid, sin_grid = materialiser.phase_grid(ref)
        payload = materialiser.store.get(ref)
        measured = farfield_frame_to_grid(payload["_img"], GRID_MEASURED, mode="raw")
        phasor = torch.as_tensor(
            np.stack([cos_grid, sin_grid])[None], dtype=torch.float32
        )
        with torch.no_grad():
            twin = bench.forward(phasor, detector=False)[0, 0].to(torch.float32).numpy()

        ncc: dict[str, float] = {}
        l1: dict[str, float] = {}
        for k in K_GRID:
            sampled = sample_twin(twin, k)
            ncc[f"{k:g}"] = zero_mean_ncc(measured, sampled)
            l1[f"{k:g}"] = normalised_l1(measured, sampled)

        digest = _digest(cos_grid, sin_grid, measured)
        per_record.append(
            {
                "index": position,
                "path": _relative(ref.path),
                "position_in_payload": int(ref.position),
                "phase_source": ref.source.value,
                "exposure_ms": ref.exposure_ms,
                "mode_id": _meta(payload, "mode_id"),
                "coefficient": _meta(payload, "coefficient"),
                "probe": _meta(payload, "probe"),
                "digest": digest,
                "ncc": ncc,
                "l1": l1,
                "measured": measured,
                "twin": twin,
                "phasor": phasor.numpy().astype(np.float32),
            }
        )
        if progress_every > 0 and (position + 1) % progress_every == 0:
            print(f"  sweep: {position + 1}/{len(index.records)}", flush=True)

    swept = time.perf_counter()

    by_digest: dict[str, list[int]] = {}
    for record in per_record:
        by_digest.setdefault(record["digest"], []).append(record["index"])
    distinct = [per_record[members[0]] for members in by_digest.values()]
    for record in distinct:
        record["replicas"] = len(by_digest[record["digest"]])

    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for record in distinct:
        groups.setdefault(
            (record["mode_id"], record["coefficient"], record["probe"]), []
        ).append(record)
    for members in groups.values():
        for ordinal, record in enumerate(
            sorted(members, key=lambda item: item["digest"])
        ):
            record["round"] = ordinal

    distinct_medians = [
        describe(r["ncc"][f"{k:g}"] for r in distinct)["median"] for k in K_GRID
    ]
    selection = interior_check(distinct_medians)
    k_star = float(selection["k"])

    rounds = sorted({r["round"] for r in distinct})
    round_medians = {
        label: [
            describe(
                r["ncc"][f"{k:g}"] for r in distinct if r["round"] == label
            )["median"]
            for k in K_GRID
        ]
        for label in rounds
    }
    round_picks = {
        label: _pick_k(values) for label, values in round_medians.items()
    }

    for record in distinct:
        twin = record["twin"]
        sampled = sample_twin(twin, k_star)
        record["ncc_at_k"] = zero_mean_ncc(record["measured"], sampled)
        record["l1_at_k"] = normalised_l1(record["measured"], sampled)
        record["ncc_at_apriori"] = zero_mean_ncc(
            record["measured"], sample_twin(twin, K_APRIORI)
        )

        best = (-2.0, 0, 0)
        for dy in range(-SHIFT_RADIUS_PX, SHIFT_RADIUS_PX + 1):
            for dx in range(-SHIFT_RADIUS_PX, SHIFT_RADIUS_PX + 1):
                shifted = sample_twin(
                    twin,
                    k_star,
                    center_y=TWIN_AXIS_PX + dy * k_star,
                    center_x=TWIN_AXIS_PX + dx * k_star,
                )
                value = zero_mean_ncc(record["measured"], shifted)
                if np.isfinite(value) and value > best[0]:
                    best = (value, dy, dx)
        record["ncc_shifted"] = best[0]
        record["shift_y"] = int(best[1])
        record["shift_x"] = int(best[2])

        with torch.no_grad():
            twin_bare = (
                bare_bench.forward(
                    torch.as_tensor(record["phasor"], dtype=torch.float32),
                    detector=False,
                )[0, 0]
                .to(torch.float32)
                .numpy()
            )
        bare_sampled = sample_twin(twin_bare, k_star)
        record["ncc_bare"] = zero_mean_ncc(record["measured"], bare_sampled)
        record["l1_bare"] = normalised_l1(record["measured"], bare_sampled)

        record["fwhm_measured"] = fwhm_px(record["measured"])
        record["fwhm_twin"] = fwhm_px(sampled)
        twin_width = record["fwhm_twin"]
        record["fwhm_ratio"] = (
            record["fwhm_measured"] / twin_width
            if np.isfinite(twin_width) and twin_width > 0.0
            else float("nan")
        )
        record["twin_at_k"] = sampled
        del record["twin"]
        del record["phasor"]

    kept = {id(record) for record in distinct}
    for record in per_record:
        # The duplicates never enter a statistic; dropping their arrays keeps the
        # payload from being held twice while the figures are drawn. The distinct
        # representatives keep `measured` (the example figure needs it).
        if id(record) in kept:
            continue
        record.pop("twin", None)
        record.pop("phasor", None)
        record.pop("measured", None)

    measured_all = time.perf_counter()
    sidecar = dict(index.records[0].sidecar) if index.records else {}
    replica_histogram: dict[str, int] = {}
    for count in by_digest.values():
        key = str(len(count))
        replica_histogram[key] = replica_histogram.get(key, 0) + 1
    return {
        "index": index,
        "sidecar": {str(k): v for k, v in sidecar.items()},
        "per_record": per_record,
        "distinct": distinct,
        "duplication": {
            "records": len(per_record),
            "distinct": len(distinct),
            "replica_histogram": replica_histogram,
            "replication_factor": (
                len(per_record) / len(distinct) if distinct else float("nan")
            ),
            "combos": len(groups),
            "rounds": len(rounds),
        },
        "selection": selection,
        "k_star": k_star,
        "distinct_medians": distinct_medians,
        "round_medians": round_medians,
        "round_picks": round_picks,
        "timings": {
            "index_s": indexed - started,
            "build_s": built - indexed,
            "sweep_s": swept - built,
            "diagnostics_s": measured_all - swept,
            "total_s": measured_all - started,
        },
        "config": {
            "corpus_root": _relative(CORPUS_ROOT),
            "corpus_pattern": CORPUS_PATTERN,
            "files": index.files_scanned,
            "records": len(per_record),
            "excluded": {k.value: int(v) for k, v in index.excluded.items()},
            "exposure_ms_values": sorted(
                {float(r["exposure_ms"]) for r in per_record if r["exposure_ms"] is not None}
            ),
            "payload_files": len({r["path"] for r in per_record}),
            "grid_phase": GRID_PHASE,
            "grid_measured": GRID_MEASURED,
            "panel_radius_px": PANEL_RADIUS_PX,
            "k_grid": list(K_GRID),
            "k_apriori": K_APRIORI,
            "ncc_bar": NCC_BAR,
            "l1_bar": L1_BAR,
            "min_records": MIN_RECORDS,
            "interior_margin": INTERIOR_MARGIN,
            "shift_radius_px": SHIFT_RADIUS_PX,
            "twin": {
                "config": "Sim2fConfig.calibrated()",
                "defocus_m": float(bench.cfg.errors.defocus_m),
                "zero_order_eta": float(bench.cfg.errors.zero_order_eta),
                "lut_gamma": float(bench.cfg.errors.lut_gamma),
                "ccd_px": int(bench.cfg.ccd_px),
                "ccd_pixel_um": float(bench.cfg.ccd_pixel_um),
                "fft_n": int(bench.fft_n),
                "sampling_ratio": float(bench.cfg.sampling_ratio),
                "beam_w0_model_px": float(bench.cfg.beam_w0_model_px),
                "pupil_radius_px": float(bench.cfg.pupil_radius_px),
            },
        },
    }


def _digest(
    cos_grid: np.ndarray, sin_grid: np.ndarray, measured: np.ndarray
) -> str:
    """Content hash of one (phase, measured frame) pair.

    Both halves matter: the same commanded phase can be paired with a different
    frame (a re-measurement), and the same frame with a different phase. Hashing
    the phase *grid* rather than the raw panel keeps the hash independent of panel
    resolution, which is not part of the comparison.

    Args:
        cos_grid: ``(grid, grid)`` phase cosine.
        sin_grid: ``(grid, grid)`` phase sine.
        measured: The 64x64 measured window, at full float32 precision.

    Returns:
        A 16-hex-character digest.
    """
    hasher = hashlib.blake2b(digest_size=8)
    hasher.update(np.ascontiguousarray(cos_grid, dtype=np.float32).tobytes())
    hasher.update(np.ascontiguousarray(sin_grid, dtype=np.float32).tobytes())
    hasher.update(np.ascontiguousarray(measured, dtype=np.float32).tobytes())
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _family_key(record: dict[str, Any]) -> str:
    """Human label for the ``mode_id x coefficient`` family of one record."""
    mode = record.get("mode_id")
    coefficient = record.get("coefficient")
    mode_label = {0: "无像差", 2: "离焦 1.0 rad"}.get(mode, f"mode_id={mode}")
    return f"{mode_label} / coefficient={fmt_general(coefficient, ':g')}"


def score(measurement: dict[str, Any]) -> dict[str, Any]:
    """Apply the pre-registered verdict rules to the de-duplicated measurement.

    The unit of every statistic is the **distinct** (phase, frame) pair. Records
    are reported alongside, never instead: this corpus re-stores each pair ~17
    times, so a record-weighted median would be a weighting artefact of the dump
    layout rather than a property of the bench.

    Args:
        measurement: Output of :func:`measure_corpus`.

    Returns:
        ``{"sample", "sweep", "geometry", "verdict", ...}``.
    """
    distinct = measurement["distinct"]
    records = measurement["per_record"]
    k_star = measurement["k_star"]
    key_star = f"{k_star:g}"

    def panel(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
        """Median/IQR summary of one subset at ``k*`` plus its family breakdown."""
        families: dict[str, list[dict[str, Any]]] = {}
        probes: dict[float | None, list[dict[str, Any]]] = {}
        for record in rows:
            families.setdefault(_family_key(record), []).append(record)
            probes.setdefault(record.get("probe"), []).append(record)
        return {
            "n": len(rows),
            "ncc": describe(r["ncc_at_k"] for r in rows),
            "l1": describe(r["l1_at_k"] for r in rows),
            "ncc_apriori": describe(r["ncc_at_apriori"] for r in rows),
            "ncc_shifted": describe(r["ncc_shifted"] for r in rows),
            "ncc_bare": describe(r["ncc_bare"] for r in rows),
            "l1_bare": describe(r["l1_bare"] for r in rows),
            "fwhm_ratio": describe(r["fwhm_ratio"] for r in rows),
            "families": {
                label: {
                    "n": len(members),
                    "ncc": describe(m["ncc_at_k"] for m in members),
                    "l1": describe(m["l1_at_k"] for m in members),
                }
                for label, members in sorted(families.items())
            },
            "probes": {
                fmt_general(probe, ":g"): {
                    "n": len(members),
                    "ncc": describe(m["ncc_at_k"] for m in members),
                    "l1": describe(m["l1_at_k"] for m in members),
                }
                for probe, members in sorted(
                    probes.items(), key=lambda item: (item[0] is None, item[0])
                )
            },
        }

    sweep: list[dict[str, Any]] = []
    for k in K_GRID:
        key = f"{k:g}"
        entry = {
            "k": k,
            "ncc": describe(r["ncc"][key] for r in distinct),
            "l1": describe(r["l1"][key] for r in distinct),
            "records_ncc": describe(r["ncc"][key] for r in records),
        }
        entry["meets"] = bool(
            np.isfinite(entry["ncc"]["median"])
            and np.isfinite(entry["l1"]["median"])
            and entry["ncc"]["median"] >= NCC_BAR
            and entry["l1"]["median"] <= L1_BAR
        )
        sweep.append(entry)

    selection = measurement["selection"]
    k_index = K_GRID.index(measurement["k_star"]) if measurement["k_star"] in K_GRID else -1
    round_agreement = {
        f"第{label + 1}轮": {
            "k": K_GRID[pick],
            "index": int(pick),
            "gap_steps": int(abs(pick - k_index)),
            "agrees": bool(abs(pick - k_index) <= 1),
            "medians": measurement["round_medians"][label],
        }
        for label, pick in measurement["round_picks"].items()
    }
    geometry = {
        "n_distinct": len(distinct),
        "enough_records": len(distinct) >= MIN_RECORDS,
        "interior": selection,
        "rounds": round_agreement,
        "rounds_agree": bool(
            all(entry["agrees"] for entry in round_agreement.values())
        ),
    }
    geometry["identified"] = bool(
        geometry["enough_records"]
        and selection["interior"]
        and geometry["rounds_agree"]
    )

    sample = panel(distinct)
    meets_at_kstar = bool(
        sample["ncc"]["median"] >= NCC_BAR and sample["l1"]["median"] <= L1_BAR
    )
    any_point_meets = any(entry["meets"] for entry in sweep)

    round_detail = [
        f"{label}: k = {entry['k']:g}, 与全体 k* 相差 {entry['gap_steps']} 步"
        for label, entry in round_agreement.items()
        if not entry["agrees"]
    ]
    if not geometry["enough_records"]:
        verdict = "insufficient"
        reason = (
            f"去重后只有 {geometry['n_distinct']} 个不同的 (相位, 帧) 对 "
            f"(< {MIN_RECORDS}), 不足以支撑任何结论"
        )
    elif meets_at_kstar and geometry["identified"]:
        verdict = "similar"
        reason = "在 k* 上同时满足中位 ncc 与中位归一化 L1 的门限, 且几何可辨识"
    elif (not meets_at_kstar) and (geometry["identified"] or not any_point_meets):
        verdict = "not_similar"
        reason = (
            "k* 上中位 ncc 或中位归一化 L1 未达门限; "
            + (
                "几何可辨识"
                if geometry["identified"]
                else "且扫描的每一个 k 都未达门限, 故几何不确定性无法救回结论"
            )
        )
    else:
        verdict = "insufficient"
        reason = (
            f"度量结论随 k 变化 (扫描里既有达标的 k 也有不达标的 k), 而几何不可辨识: "
            f"{selection['reason']}"
            if not meets_at_kstar
            else f"几何不可辨识, 结论取决于 k*: {selection['reason']}"
        )

    l1_min = min(
        (entry["l1"]["median"] for entry in sweep if np.isfinite(entry["l1"]["median"])),
        default=float("nan"),
    )
    ncc_max = max(
        (entry["ncc"]["median"] for entry in sweep if np.isfinite(entry["ncc"]["median"])),
        default=float("nan"),
    )
    plateau = [
        entry["k"]
        for entry in sweep
        if np.isfinite(entry["ncc"]["median"])
        and np.isfinite(ncc_max)
        and entry["ncc"]["median"] >= ncc_max - INTERIOR_MARGIN
    ]
    boundary = (
        f"扫描 {len(K_GRID)} 个 k 上归一化 L1 的最小值 = {fmt_general(l1_min, '.4f')} "
        f"(门限 {L1_BAR:g}), ncc 的最大值 = {fmt_general(ncc_max, '+.4f')} "
        f"(门限 {NCC_BAR:g})"
        + (
            "; L1 在整个 k 范围内都够不到, 因此 similar 不可达"
            if np.isfinite(l1_min) and l1_min > L1_BAR
            else ""
        )
    )

    combinations: list[dict[str, Any]] = []
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for record in distinct:
        grouped.setdefault(
            (record.get("mode_id"), record.get("coefficient"), record.get("probe")),
            [],
        ).append(record)
    for key, members in sorted(
        grouped.items(), key=lambda item: (item[0][0] or 0, item[0][2] or 0)
    ):
        combinations.append(
            {
                "label": f"{_family_key(members[0])} / probe={fmt_general(key[2], ':g')}",
                "n_rounds": len(members),
                "ncc": describe(m["ncc_at_k"] for m in members),
                "l1": describe(m["l1_at_k"] for m in members),
            }
        )

    return {
        "k_star": k_star,
        "sample": sample,
        "sweep": sweep,
        "geometry": geometry,
        "meets_at_kstar": meets_at_kstar,
        "any_point_meets": bool(any_point_meets),
        "verdict": verdict,
        "reason": reason,
        "boundary": boundary,
        "l1_min_over_sweep": l1_min,
        "ncc_max_over_sweep": ncc_max,
        "plateau_k": {"k_min": min(plateau) if plateau else float("nan"),
                      "k_max": max(plateau) if plateau else float("nan"),
                      "points": len(plateau)},
        "bare_gap": {
            "ncc": float(sample["ncc_bare"]["median"] - sample["ncc"]["median"]),
            "l1": float(sample["l1_bare"]["median"] - sample["l1"]["median"]),
        },
        "shift_gain": float(
            sample["ncc_shifted"]["median"] - sample["ncc"]["median"]
        ),
        "rounds_failing": round_detail,
        "per_round": {
            f"第{label + 1}轮": {
                "k_pick": K_GRID[pick],
                "ncc": describe(
                    r["ncc_at_k"] for r in distinct if r["round"] == label
                ),
                "l1": describe(r["l1_at_k"] for r in distinct if r["round"] == label),
            }
            for label, pick in measurement["round_picks"].items()
        },
        "per_combination": combinations,
        "ncc_at_apriori": {
            "median": describe(r["ncc_at_apriori"] for r in distinct)["median"],
            "min": describe(r["ncc_at_apriori"] for r in distinct)["min"],
            "n": len(distinct),
        },
        "records_at_kstar": {
            "n": len(records),
            "ncc": describe(r["ncc"][key_star] for r in records),
            "l1": describe(r["l1"][key_star] for r in records),
        },
    }


def g04_answer(result: dict[str, Any]) -> dict[str, Any]:
    """Answer G0-4 in both of its defensible readings.

    G0-4 has no scaling freedom, so it is evaluated at :data:`K_APRIORI`, where
    the two declared 2.2 um pixel pitches put the mapping. The session that wrote
    the gate reported ``min`` over its 24 samples, so ``min`` is the primary
    reading and the median is reported next to it.

    Args:
        result: Output of :func:`score`.

    Returns:
        ``{"k", "min", "median", "n", "passes_min", "passes_median"}``.
    """
    at_apriori = result["ncc_at_apriori"]
    return {
        "k": K_APRIORI,
        "min": at_apriori["min"],
        "median": at_apriori["median"],
        "n": at_apriori["n"],
        "passes_min": bool(np.isfinite(at_apriori["min"]) and at_apriori["min"] >= NCC_BAR),
        "passes_median": bool(
            np.isfinite(at_apriori["median"]) and at_apriori["median"] >= NCC_BAR
        ),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _configure_fonts() -> None:
    """CJK-capable fonts, with a fallback that still renders Latin label text."""
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def render_figures(
    measurement: dict[str, Any], result: dict[str, Any], out_dir: Path
) -> list[str]:
    """Draw the five report figures.

    Args:
        measurement: Output of :func:`measure_corpus`.
        result: Output of :func:`score`.
        out_dir: Figure directory.

    Returns:
        File names (relative to ``out_dir``) of the figures written, in report
        order.
    """
    _configure_fonts()
    out_dir.mkdir(parents=True, exist_ok=True)
    k_star = result["k_star"]
    distinct = measurement["distinct"]
    written: list[Path] = []

    # -- 1. ncc vs k: distinct median + IQR, the record-weighted line, and the
    #       three measurement rounds as thin curves ---------------------------
    figure, axes = plt.subplots(figsize=(8.4, 4.8))
    medians = [entry["ncc"]["median"] for entry in result["sweep"]]
    lows = [entry["ncc"]["q25"] for entry in result["sweep"]]
    highs = [entry["ncc"]["q75"] for entry in result["sweep"]]
    axes.plot(K_GRID, medians, marker="o", ms=5, color="tab:blue", label="去重后中位")
    axes.fill_between(K_GRID, lows, highs, color="tab:blue", alpha=0.18, linewidth=0)
    axes.plot(
        K_GRID,
        [entry["records_ncc"]["median"] for entry in result["sweep"]],
        ls="--",
        color="tab:gray",
        lw=1,
        label="按记录加权中位 (含 17x 复制)",
    )
    for label, entry in measurement["round_medians"].items():
        n_pairs = sum(1 for record in distinct if record["round"] == label)
        axes.plot(
            K_GRID,
            entry,
            ls=":",
            lw=1.1,
            alpha=0.85,
            label=f"第{label + 1}轮 ({n_pairs} 个对)",
        )
    axes.axhline(NCC_BAR, color="tab:red", ls="--", lw=1, label=f"G0-4 门限 {NCC_BAR:g}")
    axes.axvline(k_star, color="tab:green", ls=":", lw=1.2, label=f"k* = {k_star:g}")
    axes.axvline(K_APRIORI, color="black", ls="-", lw=0.8, alpha=0.5, label="先验 k = 1")
    axes.set_xlabel("k = 孪生 CCD 像素 / 实测像素")
    axes.set_ylabel("ncc (零均值)")
    axes.set_title("孪生 vs 实测: ncc 随几何标度 k 的整条曲线")
    axes.grid(alpha=0.25)
    axes.legend(fontsize=7.5, ncol=2)
    path = out_dir / "ncc_vs_k.png"
    savefig(figure, path)
    written.append(path)

    # -- 2. normalised L1 vs k ----------------------------------------------
    figure, axes = plt.subplots(figsize=(8.2, 4.4))
    axes.plot(
        K_GRID,
        [entry["l1"]["median"] for entry in result["sweep"]],
        marker="o",
        ms=5,
        color="tab:purple",
        label="去重后中位",
    )
    axes.fill_between(
        K_GRID,
        [entry["l1"]["q25"] for entry in result["sweep"]],
        [entry["l1"]["q75"] for entry in result["sweep"]],
        color="tab:purple",
        alpha=0.18,
        linewidth=0,
    )
    axes.axhline(L1_BAR, color="tab:red", ls="--", lw=1, label=f"归一化 L1 门限 {L1_BAR:g}")
    axes.axvline(k_star, color="tab:green", ls=":", lw=1, label=f"k* = {k_star:g}")
    axes.set_xlabel("k = 孪生 CCD 像素 / 实测像素")
    axes.set_ylabel("归一化 L1 (= 2 x 全变差)")
    axes.set_title("孪生 vs 实测: 归一化 L1 随 k 的整条曲线")
    axes.grid(alpha=0.25)
    axes.legend(fontsize=8)
    path = out_dir / "l1_vs_k.png"
    savefig(figure, path)
    written.append(path)

    # -- 3. per-family ncc at k* -------------------------------------------
    families = result["sample"]["families"]
    labels = list(families)
    data = [
        [
            record["ncc_at_k"]
            for record in distinct
            if _family_key(record) == label and np.isfinite(record["ncc_at_k"])
        ]
        for label in labels
    ]
    figure, axes = plt.subplots(figsize=(8.2, 4.4))
    axes.boxplot(data, tick_labels=labels, widths=0.45, showfliers=True)
    axes.axhline(NCC_BAR, color="tab:red", ls="--", lw=1, label=f"G0-4 门限 {NCC_BAR:g}")
    for position, values in enumerate(data, start=1):
        jitter = (np.arange(len(values)) % 5 - 2) * 0.02
        axes.plot(position + jitter, values, "o", ms=4, alpha=0.7, color="tab:blue")
    axes.set_ylabel("ncc @ k* (零均值)")
    axes.set_title(f"逐族 ncc (去重后, n={len(distinct)}, k* = {k_star:g})")
    axes.grid(alpha=0.25, axis="y")
    axes.legend(fontsize=8)
    plt.setp(axes.get_xticklabels(), rotation=10, fontsize=8)
    path = out_dir / "per_family_ncc.png"
    savefig(figure, path)
    written.append(path)

    # -- 4. examples --------------------------------------------------------
    picks: list[int] = []
    if distinct:
        order = sorted(
            range(len(distinct)),
            key=lambda i: distinct[i]["ncc_at_k"]
            if np.isfinite(distinct[i]["ncc_at_k"])
            else -2.0,
        )
        picks = [order[0], order[len(order) // 2], order[-1]]
        for probe in (0, 1, 2):
            candidate = next(
                (
                    i
                    for i in range(len(distinct))
                    if distinct[i].get("probe") == probe and i not in picks
                ),
                None,
            )
            if candidate is not None:
                picks.append(candidate)
    figure, axes = plt.subplots(
        len(picks) or 1, 3, figsize=(10.5, 2.7 * max(len(picks), 1))
    )
    axes = np.atleast_2d(axes)
    for row, index in enumerate(picks):
        record = distinct[index]
        measured = record["measured"]
        twin = record["twin_at_k"]
        axes[row, 0].imshow(measured, cmap="inferno")
        axes[row, 0].set_title(
            f"实测 #{index} ({_family_key(record)}, probe={fmt_general(record.get('probe'), ':g')})",
            fontsize=8,
        )
        axes[row, 1].imshow(twin, cmap="inferno")
        axes[row, 1].set_title(
            f"孪生 @ k*={k_star:g}  ncc={record['ncc_at_k']:+.3f}", fontsize=8
        )
        axes[row, 2].imshow(
            measured / max(float(measured.sum()), 1e-12)
            - twin / max(float(twin.sum()), 1e-12),
            cmap="coolwarm",
        )
        axes[row, 2].set_title("差值 (各自归一化)", fontsize=8)
    for axis in axes.ravel():
        axis.set_xticks([])
        axis.set_yticks([])
    figure.tight_layout()
    path = out_dir / "examples.png"
    savefig(figure, path)
    written.append(path)

    # -- 5. shift diagnostic ------------------------------------------------
    shifts = np.array(
        [
            (r["shift_y"], r["shift_x"])
            for r in distinct
            if np.isfinite(r["ncc_shifted"])
        ],
        dtype=int,
    )
    figure, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    if shifts.size:
        grid = np.zeros((2 * SHIFT_RADIUS_PX + 1, 2 * SHIFT_RADIUS_PX + 1))
        for dy, dx in shifts:
            grid[dy + SHIFT_RADIUS_PX, dx + SHIFT_RADIUS_PX] += 1
        image = axes[0].imshow(grid, cmap="viridis", origin="lower")
        axes[0].set_xticks(
            range(2 * SHIFT_RADIUS_PX + 1),
            [str(v) for v in range(-SHIFT_RADIUS_PX, SHIFT_RADIUS_PX + 1)],
        )
        axes[0].set_yticks(
            range(2 * SHIFT_RADIUS_PX + 1),
            [str(v) for v in range(-SHIFT_RADIUS_PX, SHIFT_RADIUS_PX + 1)],
        )
        axes[0].set_xlabel("Δx (实测像素)")
        axes[0].set_ylabel("Δy (实测像素)")
        axes[0].set_title("逐样本最优整数平移", fontsize=9)
        figure.colorbar(image, ax=axes[0], shrink=0.8)
        delta = np.array(
            [
                r["ncc_shifted"] - r["ncc_at_k"]
                for r in distinct
                if np.isfinite(r["ncc_shifted"])
            ]
        )
        axes[1].hist(delta, bins=12, color="tab:orange")
        axes[1].set_xlabel("ncc(平移后) − ncc(k*)")
        axes[1].set_title("平移带来的 ncc 增量", fontsize=9)
    else:
        axes[0].text(0.5, 0.5, "无可用数据", ha="center", va="center")
        axes[1].axis("off")
    figure.tight_layout()
    path = out_dir / "shift_diagnostic.png"
    savefig(figure, path)
    written.append(path)

    return [p.relative_to(out_dir).as_posix() for p in written]


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

_G = partial(fmt_general, missing_nonfinite="n/a")
_G3 = partial(fmt_general, spec="+.3f", missing_nonfinite="n/a")


def _image_line(known: dict[str, str], name: str, alt: str) -> str:
    """Markdown image for figure ``name``, or an explicit "not generated" line.

    A missing figure must not become a broken link -- ``test_report_provenance``
    resolves every relative link in the report -- and it must not silently vanish
    either, because ``--skip-figures`` has to be visible in the text.

    Args:
        known: Map of figure stem to relative path, from :func:`render_report`.
        name: Figure stem to embed.
        alt: Alt text, in the user's language.

    Returns:
        One markdown line.
    """
    if name in known:
        return f"![{alt}](figures/{known[name]})"
    return f"（图 `{name}` 未生成: 本次运行使用了 `--skip-figures`）"


def _metric_rows(sweep: Sequence[dict[str, Any]]) -> list[list[str]]:
    """Sweep-table rows: one per ``k``, de-duplicated statistics plus the record-weighted median."""
    rows = []
    for entry in sweep:
        rows.append(
            [
                f"{entry['k']:g}",
                str(entry["ncc"]["n"]),
                _G3(entry["ncc"]["median"]),
                _G3(entry["ncc"]["q25"]) + " .. " + _G3(entry["ncc"]["q75"]),
                _G3(entry["records_ncc"]["median"]),
                _G(entry["l1"]["median"], ".4f"),
                _G(entry["l1"]["q25"], ".4f") + " .. " + _G(entry["l1"]["q75"], ".4f"),
                "是" if entry["meets"] else "否",
            ]
        )
    return rows


def _family_rows(panel: dict[str, Any]) -> list[list[str]]:
    """Per-family rows: n, median ncc with IQR, median L1 with IQR."""
    rows = []
    for label, data in panel["families"].items():
        rows.append(
            [
                label,
                str(data["n"]),
                _G3(data["ncc"]["median"]),
                _G3(data["ncc"]["q25"]) + " .. " + _G3(data["ncc"]["q75"]),
                _G(data["l1"]["median"], ".4f"),
                _G(data["l1"]["q25"], ".4f") + " .. " + _G(data["l1"]["q75"], ".4f"),
            ]
        )
    return rows


def _probe_rows(panel: dict[str, Any]) -> list[list[str]]:
    """Per-``probe`` rows, the same columns as the family table."""
    rows = []
    for probe, data in panel["probes"].items():
        rows.append(
            [
                probe,
                str(data["n"]),
                _G3(data["ncc"]["median"]),
                _G3(data["ncc"]["q25"]) + " .. " + _G3(data["ncc"]["q75"]),
                _G(data["l1"]["median"], ".4f"),
                _G(data["l1"]["q25"], ".4f") + " .. " + _G(data["l1"]["q75"], ".4f"),
            ]
        )
    return rows


def render_report(
    measurement: dict[str, Any],
    result: dict[str, Any],
    gate: dict[str, Any],
    figures: Sequence[str],
) -> str:
    """Render the markdown report.

    Args:
        measurement: Output of :func:`measure_corpus`.
        result: Output of :func:`score`.
        gate: Output of :func:`g04_answer`.
        figures: Figure file names relative to the report directory.

    Returns:
        The report text, without the provenance header (``sync_report_provenance``
        owns that block).
    """
    config = measurement["config"]
    sample = result["sample"]
    duplication = measurement["duplication"]
    geometry = result["geometry"]
    twin = config["twin"]
    fig = {Path(name).stem: name for name in figures}

    verdict_label = {
        "similar": "**相似** (`similar`)",
        "not_similar": "**不相似** (`not_similar`)",
        "insufficient": "**功效不足** (`insufficient`)",
    }[result["verdict"]]

    l1_curve = [
        (entry["k"], entry["l1"]["median"])
        for entry in result["sweep"]
        if np.isfinite(entry["l1"]["median"])
    ]
    l1_best = (
        {
            "k": min(l1_curve, key=lambda item: item[1])[0],
            "value": min(item[1] for item in l1_curve),
        }
        if l1_curve
        else {"k": float("nan"), "value": float("nan")}
    )
    replicas = duplication["replica_histogram"]
    replica_text = ", ".join(
        f"{count} 次 x {pairs} 个对"
        for count, pairs in sorted(replicas.items(), key=lambda item: -int(item[0]))
    )

    lines = [
        "# 孪生 `ml.sim2f` 与 2f 实验语料的一致性 (纯离线, 只读 `data/debug/`)",
        "",
        f"- 判定: {verdict_label}",
        f"- 依据: {result['reason']}",
        f"- 边界: {result['boundary']}",
        f"- 几何最优点: `k* = {result['k_star']:g}` "
        f"(孪生 CCD 像素 / 实测像素; 先验预期 `k = {K_APRIORI:g}`), "
        f"内部性: {'是' if geometry['interior']['interior'] else '**否**'}"
        f" ({geometry['interior']['reason']})",
        f"- G0-4 (`ncc >= {NCC_BAR:g}`) 在 `k = {K_APRIORI:g}` 上: "
        f"min = {_G3(gate['min'])} → {'通过' if gate['passes_min'] else '**未通过**'}; "
        f"median = {_G3(gate['median'])} → {'通过' if gate['passes_median'] else '**未通过**'} "
        f"(n={gate['n']} 个不同的对)",
        f"- 语料: `{config['corpus_root']}/{config['corpus_pattern']}` — "
        f"{duplication['records']} 条记录 / {config['payload_files']} 个 pkl, "
        f"其中只有 **{duplication['distinct']} 个不同的 (相位, 帧) 对** "
        f"(复制 {duplication['replication_factor']:.1f} 倍)",
        "- 生成脚本: `scripts/ml/validate_sim2f_vs_hw_corpus.py`",
        "",
        _image_line(fig, "ncc_vs_k", "ncc 随 k 的曲线"),
        "",
        "## 1. 预登记判据",
        "",
        "### 1.1 原文 (写于任何统计量被计算之前)",
        "",
        "```",
        PREREGISTRATION.rstrip(),
        "```",
        "",
        "### 1.2 预登记缺陷与加固 (第一次全量运行之后追加; 1.1 一字未改)",
        "",
        "```",
        AMENDMENT.rstrip(),
        "```",
        "",
        "## 2. 语料与口径",
        "",
        "为什么要这一族, 以及任务书里点名的几族为什么用不了:",
        "",
        markdown_table(
            ["族", "结论", "理由"],
            [
                [
                    "`model_in_loop_hw_collect/20261025*` (本报告采用)",
                    "可用",
                    f"{config['files']} 个 pkl / {duplication['records']} 条记录, 每条同时含 "
                    "`_phase` (1200x1920 弧度) 与 `_img` (整幅 CCD); "
                    "采集日 2026-10-25 晚于孪生标定日 2026-10-23 ⇒ 时间上验证集; "
                    "曝光取值单一, 无需跨曝光归一",
                ],
                [
                    "`sim_vs_hw_20261023_v2/`, `sim_vs_hw_20261023/`, "
                    "`sim_vs_hw_20261022/`, `hw_ab_new_20261023/`, "
                    "`hw_ab_old_20261023/`, `hw_groups_20261022/`",
                    "**用不了**",
                    "这些族里**没有 `.pkl`**: 帧存为 `.npz` + `measurements.json`, "
                    "`ml.hwdataset` 的索引只能从 pickle 里定位相位与帧, 索引结果为 0 条",
                ],
                [
                    "`slm_transfer/` (1.8 GB), `model_in_loop_hw_sweep/` (62.9 GB)",
                    "弃用",
                    "前者以逐次 `npz` 记录的迁移实验为主, 与「一条相位对应一帧」的口径不符; "
                    "后者体量 62.9 GB, 单次全量读取代价远超本问题所需",
                ],
                [
                    "`slm_gs_refine/` (862 MB)",
                    "弃用",
                    "采集于当前光路改动之前, 与 `Sim2fConfig.calibrated()` 的标定不同源",
                ],
            ],
        ),
        "",
        "逐条口径 (全部复用既有实现, 本脚本只新增几何映射与两个度量):",
        "",
        markdown_table(
            ["环节", "实现", "关键参数"],
            [
                [
                    "语料索引",
                    "`ml.hwdataset.build_hw_index`",
                    f"{config['files']} 个 pkl, 排除 {sum(config['excluded'].values())} 条",
                ],
                [
                    "相位 → 128x128 相量",
                    "`ml.hwdataset.records.Materialiser.phase_grid` "
                    "(相干块平均, 非算术平均)",
                    f"grid={config['grid_phase']}, 瞳面半径 {config['panel_radius_px']:g} px, "
                    "瞳面中心 (960, 600) px",
                ],
                [
                    "实测帧 → 64x64",
                    "`ml.hwdataset.transforms.farfield_frame_to_grid(..., "
                    f"{config['grid_measured']}, \"raw\")`",
                    "以整幅帧 argmax (3x3 去尖峰) 为锚, 不缩放不裁剪归一",
                ],
                [
                    "孪生",
                    "`ml.sim2f.optics.Sim2f(Sim2fConfig.calibrated())`, CPU",
                    f"产 {twin['ccd_px']}x{twin['ccd_px']} 强度帧; fft_n={twin['fft_n']}; "
                    f"相机离焦 {1e3 * twin['defocus_m']:.4f} mm; "
                    f"0 级占比 {twin['zero_order_eta']:g}",
                ],
                [
                    "几何映射",
                    "本脚本 `sample_twin`: 实测索引 i 取孪生 "
                    f"`{TWIN_AXIS_PX:g} + (i − {MEASURED_ANCHOR_PX:g})·k`, 双线性",
                    f"k ∈ {{{', '.join(f'{k:g}' for k in K_GRID)}}}",
                ],
                [
                    "去重",
                    "本脚本 `_digest`: 相位格 (cos, sin) + 实测帧的内容哈希",
                    f"{duplication['records']} 条 → {duplication['distinct']} 个不同的对; "
                    f"{replica_text}",
                ],
                [
                    "度量",
                    "零均值 ncc; 归一化 L1 = Σ|a/Σa − b/Σb|",
                    f"门限 ncc >= {NCC_BAR:g}, L1 <= {L1_BAR:g}",
                ],
            ],
        ),
        "",
        f"侧车 (与孪生标定相关的字段): 曝光 `{_sidecar(measurement, 'exposure_ms')}` ms, "
        f"相机窗口 `region={_sidecar(measurement, 'region')}`, "
        f"`far_field_size={_sidecar(measurement, 'far_field_size')}`, "
        f"`defocus_rad={_sidecar(measurement, 'defocus_rad')}`, "
        f"`zernike_radius={_sidecar(measurement, 'zernike_radius')}`, "
        f"`pupil_center_panel={_sidecar(measurement, 'pupil_center_panel')}`, "
        f"`camera_zero_order={_sidecar(measurement, 'camera_zero_order')}`。",
        "",
        "### 2.1 语料的真实规模 (结论前必读)",
        "",
        f"{duplication['records']} 条记录按内容哈希只剩 "
        f"**{duplication['distinct']} 个不同的 (相位, 帧) 对**: "
        f"{duplication['combos']} 个指令条件 (`mode_id x coefficient x probe`) x "
        f"{duplication['rounds']} 个测量轮次。同一个对被反复存进不同 dump, "
        f"复制倍数的分布是 {replica_text}。",
        "",
        "因此:",
        "",
        "1. 本报告所有中位数 / IQR 都按**去重后的对**统计 (n = "
        f"{duplication['distinct']}), 不按 {duplication['records']} 条记录 —— "
        "否则复制次数会替物理条件加权。",
        f"2. 按记录加权的同一读数也在 §3 表里给出一列作参照: "
        f"k* 处中位 ncc {_G3(result['records_at_kstar']['ncc']['median'])} "
        f"(去重后 {_G3(sample['ncc']['median'])}), "
        f"中位 L1 {_G(result['records_at_kstar']['l1']['median'], '.4f')} "
        f"(去重后 {_G(sample['l1']['median'], '.4f')})。",
        f"3. 有效样本是 {duplication['distinct']} 个**设计条件** "
        f"({duplication['combos']} 个指令条件 x {duplication['rounds']} 轮), "
        f"不是 {duplication['distinct']} 个从总体上独立抽到的样本; "
        "§5.3 的逐条件表把 3 轮之间的散布摊开来看。",
        "",
        "### 2.2 与任务书的三处不符 (如实记录)",
        "",
        markdown_table(
            ["任务书写的", "仓库实际的", "处理"],
            [
                [
                    "`ao_shaping.runners.gsnet_offline` 的辅助函数",
                    "该模块不存在; 真实路径是 `ml.gsnet_debug.offline`",
                    "本脚本没有直接调用它 —— 相位/帧的还原已由 `ml.hwdataset` "
                    "内部复用同一批函数, 再调一次等于第二份实现",
                ],
                [
                    "任务书点名的 6 个 `sim_vs_hw*` / `hw_ab*` / `hw_groups*` 族",
                    "均无 `.pkl`, 索引为 0 条",
                    "改用 `model_in_loop_hw_collect/20261025*`, 理由见上表",
                ],
                [
                    "扫描网格原计划 `k ∈ [0.5, 2.0]`",
                    "40 条记录的口径/几何预跑显示 2.0 处曲线仍在上升",
                    "**在看主结果之前**把网格扩到 0.25..4.0 (12 点), 上界由 "
                    "`64x4 = 256` = 孪生相机整幅决定; 扩充动机与时刻一并披露在此",
                ],
            ],
        ),
        "",
        "## 3. 几何扫描: 整条曲线, 最优点, 是否真的内部",
        "",
        f"统计单位是去重后的 {duplication['distinct']} 个对; 「按记录」一列只作参照。"
        f"`k*` 取自去重后的中位 ncc 曲线, 三轮各自的最优点另见 §5.2。",
        "",
        markdown_table(
            ["k", "n", "中位 ncc", "ncc IQR", "中位 ncc (按记录)", "中位 L1", "L1 IQR", "M 成立?"],
            _metric_rows(result["sweep"]),
        ),
        "",
        f"- 最优点 `k* = {geometry['interior']['k']:g}`, 中位 ncc "
        f"{_G3(geometry['interior']['value'])}; 两侧相邻: "
        f"{_G3(geometry['interior']['left'])} / {_G3(geometry['interior']['right'])} "
        f"⇒ {geometry['interior']['reason']}",
        f"- 内部性判据要求两侧都低 >= {INTERIOR_MARGIN:g}: "
        f"{'满足' if geometry['interior']['interior'] else '**不满足**'}; "
        f"落在最高值 {INTERIOR_MARGIN:g} 以内的网格点是 "
        f"`k ∈ [{result['plateau_k']['k_min']:g}, "
        f"{result['plateau_k']['k_max']:g}]` "
        f"({result['plateau_k']['points']} 个), 这就是峰的分辨率 —— 比这更细的 `k` 差异"
        "在这批数据上不可分辨",
        "- 逐轮一致性 (替代验证集协议): "
        + "; ".join(
            f"第{label[-2]}轮 k = {entry['k']:g}"
            for label, entry in geometry["rounds"].items()
        )
        + f" ⇒ {'三轮一致' if geometry['rounds_agree'] else '**三轮不一致**'}",
        f"- 几何可辨识: {'是' if geometry['identified'] else '**否**'}"
        + (
            ""
            if geometry["identified"]
            else f" (原因: {geometry['interior']['reason']}; "
            f"n={geometry['n_distinct']} "
            f"{'>=' if geometry['enough_records'] else '<'} {MIN_RECORDS}; "
            f"三轮{'一致' if geometry['rounds_agree'] else '不一致'})"
        ),
        "",
        "读法: 曲线峰值平台在 "
        f"`k ∈ [{result['plateau_k']['k_min']:g}, {result['plateau_k']['k_max']:g}]`"
        f" (argmax `k = {geometry['interior']['k']:g}`), 而不是两个像元节距所指的 "
        f"`k = {K_APRIORI:g}`。"
        "`k` 越大 = 孪生图案被缩得越小, 所以峰值大于 1 意味着**孪生自身的聚焦光斑比实测更宽**: "
        "要把它压到实测的尺度上, 必须缩。这是「标度不对」, 不是「映射选错」。"
        f"另需注意两条曲线不同步: 中位 L1 的最低点在 `k = {l1_best['k']:g}` "
        f"({_G(l1_best['value'], '.4f')}), 而 ncc 的最高点在 `k = {result['k_star']:g}` —— "
        "ncc 由峰的形状决定, L1 由总质量落在哪里决定, 两者在这里指向不同的标度, "
        "所以「最优点」本身不是一个可以单独引用的标定值。",
        "",
        _image_line(fig, "l1_vs_k", "归一化 L1 随 k"),
        "",
        "## 4. G0-4 回答",
        "",
        "G0-4 原文 (`runs/sim2f_bench/README.md:138`): "
        "`ncc(实测 net 帧, 孪生预测 net 帧) >= 0.5`, 不过的含义是"
        "\"映射/ROI/标度错, 先修几何再测\"。",
        "",
        "这道门**没有缩放自由度**, 所以只能在与像素节距相符的 `k = "
        f"{K_APRIORI:g}` 上回答; 扫描最优点 `k* = {result['k_star']:g}` 的读数作为第二行给出, "
        "用来看清「差多少是几何, 差多少是保真度」。",
        "",
        markdown_table(
            ["读数", "k", "n", "值", f"门限 {NCC_BAR:g}", "结论"],
            [
                [
                    "min ncc",
                    f"{K_APRIORI:g}",
                    str(gate["n"]),
                    _G3(gate["min"]),
                    f">= {NCC_BAR:g}",
                    "**通过**" if gate["passes_min"] else "**未通过**",
                ],
                [
                    "中位 ncc",
                    f"{K_APRIORI:g}",
                    str(gate["n"]),
                    _G3(gate["median"]),
                    f">= {NCC_BAR:g}",
                    "通过" if gate["passes_median"] else "**未通过**",
                ],
                [
                    "中位 ncc",
                    f"{result['k_star']:g} (k*)",
                    str(sample["n"]),
                    _G3(sample["ncc"]["median"]),
                    f">= {NCC_BAR:g}",
                    "通过" if sample["ncc"]["median"] >= NCC_BAR else "**未通过**",
                ],
            ],
        ),
        "",
        f"**回答**: 以实际会话使用的 `min` 口径, G0-4 在本语料上 "
        f"{'通过' if gate['passes_min'] else '**未通过**'} "
        f"(min ncc {_G3(gate['min'])}, 门限 {NCC_BAR:g}, 差 {_G3(gate['min'] - NCC_BAR)}"
        ")。门限按预登记不动。注意本语料只有 "
        f"{gate['n']} 个不同的对, 所以这条 min 的含义是「{gate['n']} 个设计条件下最差的那一个」, "
        "不是「一个总体的最差分位」。"
        + (
            "中位数通过而最小值不通过 ⇒ 失败是局部的: 多数条件已能被孪生对上形状, 少数条件对不上。"
            if gate["passes_median"] and not gate["passes_min"]
            else ""
        ),
        "",
        "## 5. 逐族指标",
        "",
        f"去重后 {sample['n']} 个对 (每族 9 个, 每个 probe 6 个)。逐条值在 "
        "`artifacts/per_record.json` 里, 这里的 IQR 是 `q25 .. q75`。",
        "",
        "### 5.1 `mode_id x coefficient`",
        "",
        markdown_table(
            ["族", "n", "中位 ncc", "ncc IQR", "中位归一化 L1", "L1 IQR"],
            _family_rows(sample),
        ),
        "",
        "`mode_id` 与 `coefficient` 在这批 dump 里一一对应 (0↔0, 2↔1), "
        "所以这张表实际只有两族, 不是四族。",
        "",
        "### 5.2 `probe` 与逐轮一致性",
        "",
        markdown_table(
            ["probe", "n", "中位 ncc", "ncc IQR", "中位归一化 L1", "L1 IQR"],
            _probe_rows(sample),
        ),
        "",
        markdown_table(
            ["测量轮次", "该轮最优点 k", "中位 ncc @ k*", "中位 L1 @ k*", "与全体 k* 相差"],
            [
                [
                    label,
                    f"{entry['k_pick']:g}",
                    _G3(entry["ncc"]["median"]),
                    _G(entry["l1"]["median"], ".4f"),
                    f"{geometry['rounds'][label]['gap_steps']} 步",
                ]
                for label, entry in result["per_round"].items()
            ],
        ),
        "",
        "### 5.3 逐条件: 三轮之间的散布",
        "",
        "同一个指令条件的三轮实测并不是同一个数 —— 这张表是理解「单一中位数能代表什么」的边界。",
        "",
        markdown_table(
            ["条件", "轮数", "ncc 最小", "ncc 中位", "ncc 最大", "L1 最小", "L1 中位", "L1 最大"],
            [
                [
                    item["label"],
                    str(item["n_rounds"]),
                    _G3(item["ncc"]["min"]),
                    _G3(item["ncc"]["median"]),
                    _G3(item["ncc"]["max"]),
                    _G(item["l1"]["min"], ".4f"),
                    _G(item["l1"]["median"], ".4f"),
                    _G(item["l1"]["max"], ".4f"),
                ]
                for item in result["per_combination"]
            ],
        ),
        "",
        _image_line(fig, "per_family_ncc", "逐族 ncc"),
        "",
        "## 6. 附属诊断 (只解释, 不参与判定)",
        "",
        markdown_table(
            ["诊断", "读数", "说明"],
            [
                [
                    "k* 处整条分布",
                    f"中位 ncc {_G3(sample['ncc']['median'])}, "
                    f"IQR {_G3(sample['ncc']['q25'])} .. {_G3(sample['ncc']['q75'])}, "
                    f"范围 {_G3(sample['ncc']['min'])} .. {_G3(sample['ncc']['max'])}; "
                    f"中位 L1 {_G(sample['l1']['median'], '.4f')}",
                    f"ncc 从 {_G3(sample['ncc']['min'])} 跨到 "
                    f"{_G3(sample['ncc']['max'])}, 说明「一个数」掩盖了条件间与轮次间的差异",
                ],
                [
                    "整数平移搜索 (|Δ| <= "
                    f"{config['shift_radius_px']} px)",
                    f"中位增益 {_G3(result['shift_gain'])}, "
                    f"平移后中位 ncc {_G3(sample['ncc_shifted']['median'])}",
                    "实测窗口以**成型帧的 argmax** 为锚, 而那未必是真 0 级; "
                    f"增益 {_G3(result['shift_gain'])} 就是锚点偏移的量级 "
                    "(它只作用于 ncc, L1 未随之重算, 故不改变判定)",
                ],
                [
                    "标定版 vs 裸版孪生 (同一 k*)",
                    f"标定版中位 ncc {_G3(sample['ncc']['median'])}, "
                    f"裸版 {_G3(sample['ncc_bare']['median'])}; "
                    f"裸版中位 L1 {_G(sample['l1_bare']['median'], '.4f')}",
                    "裸版去掉相机离焦与 0 级串扰。"
                    f"差值 (裸 − 标定) = ncc {_G3(result['bare_gap']['ncc'])}, "
                    f"L1 {_G(sample['l1_bare']['median'] - sample['l1']['median'], '+.4f')}; "
                    "这个差值同时依赖 `k` 与测量轮次, 而 k* 本身不可辨识 (§3), 所以这一行**不能**读成"
                    "「标定项好或坏」的结论",
                ],
                [
                    "光斑宽度比 (实测 / 孪生, 同一相位)",
                    f"中位 {_G(sample['fwhm_ratio']['median'], '.3f')}, "
                    f"IQR {_G(sample['fwhm_ratio']['q25'], '.3f')} .. "
                    f"{_G(sample['fwhm_ratio']['q75'], '.3f')}",
                    "比值 < 1 = 孪生光斑更宽, 与 k* > 1 的方向一致; 只作量级提示",
                ],
                [
                    "孪生模型的标称束半径",
                    f"`beam_w0_model_px = {twin['beam_w0_model_px']:g}` 面板像素",
                    "这是**瞳面**照明的高斯半径, 不是远场光斑宽度; "
                    "远场宽度由离焦与相位决定, 不能拿它与实测散斑宽直接比",
                ],
            ],
        ),
        "",
        "「标定版 vs 裸版」这一行要按上表读: 标定项 (相机离焦 "
        f"{1e3 * twin['defocus_m']:.4f} mm + 0 级占比 {twin['zero_order_eta']:g}) "
        f"在本 k* 上让 ncc 变化 {_G3(result['bare_gap']['ncc'])}。这个差值同时依赖 `k` "
        "与测量轮次, 而 `k*` 在这批数据上不可辨识 (§3), 所以它**不能**读成"
        "「标定项好或坏」的结论。",
        "",
        _image_line(fig, "examples", "示例"),
        "",
        "从示例图第 3 列可以直接读出为什么「ncc 过、L1 不过」: 差值集中在中心约 10 px 的"
        "小范围里, 而孪生图案是一圈圈**平滑**的衍射环。也就是说 ncc 主要由「能量集中在中心」"
        "这条包络贡献, 而实测画面里那层散斑结构孪生根本没有; 归一化 L1 对质量分布的形状更敏感, "
        "因此它给出一个 1.2 以上的距离。这两个数不矛盾, 它们量的是不同的东西。",
        "",
        _image_line(fig, "shift_diagnostic", "平移诊断"),
        "",
        "## 7. 局限 (读结论前必须知道)",
        "",
        f"1. **有效样本只有 {duplication['distinct']} 个设计条件。** "
        f"{duplication['records']} 条记录是同一个 6 条件 x 3 轮的小实验被平均存了 "
        f"{duplication['replication_factor']:.0f} 遍; 任何「总体」措辞都不成立, "
        f"本报告的每一句都只覆盖这 {duplication['distinct']} 个条件。",
        "2. **90° 轴交换无法在这里检出。** 侧车明确写着面板与相机坐标轴交换了 90°, "
        "而本语料的像差族以无像差 / 纯离焦为主 (近似圆对称), 旋转 90° 后的图案与原来几乎一样。"
        "要检出它必须做二面体变换 (转置 / 翻转) 对照, 本报告没有做。",
        "3. **实测窗口的锚点不是真 0 级。** `farfield_frame_to_grid` 用整幅帧的 argmax 定窗, "
        "而成型帧的全局最大是成型后的亮点; 偏移量由 §6 的整数平移搜索给出。",
        "4. **`k*` 是折衷, 不是标定值。** 孪生的光斑宽度与实测的散斑宽度不一致时, 任何单一 `k` "
        "都不可能同时对上中心和边缘; 曲线是宽的, 把 `k*` 读成「正确的几何标度」是过度解读。",
        "5. **只有一种度量族。** ncc 看形状相关, 归一化 L1 看分布距离; 二者都不含绝对量。"
        "本报告因此**不能**回答「孪生的绝对光度对不对」 —— 那需要 PIB / EE 这类量, "
        "而它们已被 `runs/sim2f_bench/README.md` 记录为已知偏差 −79 % / +26 %。",
        "",
        "## 8. 复现",
        "",
        "```bash",
        "python scripts/ml/validate_sim2f_vs_hw_corpus.py",
        "```",
        "",
        f"全量耗时 {measurement['timings']['total_s']:.1f} s "
        f"(索引 {measurement['timings']['index_s']:.1f} s, "
        f"建台 {measurement['timings']['build_s']:.1f} s, "
        f"扫描 {measurement['timings']['sweep_s']:.1f} s, "
        f"诊断 {measurement['timings']['diagnostics_s']:.1f} s)。"
        "脚本不写 `data/debug/`, 不碰任何设备, 孪生在 CPU 上跑。",
        "",
        "产物:",
        "",
        markdown_table(
            ["文件", "内容"],
            [
                ["`artifacts/metrics.json`", "判定、几何、逐族、逐轮、扫描曲线、预登记原文"],
                ["`artifacts/per_record.json`", "逐记录 ncc / L1 / 平移 / 宽度比 / 去重摘要"],
                *[[f"`{name}`", "见正文对应小节"] for name in figures],
            ],
        ),
        "",
    ]
    return "\n".join(lines)


def _sidecar(measurement: dict[str, Any], key: str) -> str:
    """One sidecar value formatted for prose, or ``未记录``."""
    value = measurement["sidecar"].get(key)
    if value is None:
        return "未记录"
    if isinstance(value, (list, tuple)):
        return "(" + ", ".join(str(v) for v in value) + ")"
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def _clean(value: Any) -> Any:
    """Make a value strict-JSON safe (no NaN/Infinity, no numpy scalars)."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def write_artefacts(
    measurement: dict[str, Any],
    result: dict[str, Any],
    gate: dict[str, Any],
    out_dir: Path,
) -> list[Path]:
    """Write ``metrics.json`` and ``per_record.json``.

    JSON, not CSV: ``.gitignore`` excludes ``*.csv`` repo-wide, so a CSV artefact
    linked from the report would be a broken link on a clean clone.

    Args:
        measurement: Output of :func:`measure_corpus`.
        result: Output of :func:`score`.
        gate: Output of :func:`g04_answer`.
        out_dir: Artefact directory.

    Returns:
        The files written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.json"
    records_path = out_dir / "per_record.json"

    metrics = {
        "generated_by": "scripts/ml/validate_sim2f_vs_hw_corpus.py",
        "report_key": REPORT_KEY,
        "pre_registration": PREREGISTRATION.splitlines(),
        "amendment": AMENDMENT.splitlines(),
        "config": measurement["config"],
        "timings_s": measurement["timings"],
        "duplication": measurement["duplication"],
        "k_star": result["k_star"],
        "sweep": result["sweep"],
        "geometry": result["geometry"],
        "verdict": {
            "value": result["verdict"],
            "reason": result["reason"],
            "boundary": result["boundary"],
            "meets_at_kstar": result["meets_at_kstar"],
            "any_point_meets": result["any_point_meets"],
            "l1_min_over_sweep": result["l1_min_over_sweep"],
        },
        "g04": gate,
        "sample": result["sample"],
        "records_at_kstar": result["records_at_kstar"],
        "per_round": result["per_round"],
        "per_combination": result["per_combination"],
    }
    metrics_path.write_text(
        json.dumps(_clean(metrics), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    per_record = [
        {
            key: value
            for key, value in record.items()
            if key not in ("measured", "twin_at_k")
        }
        for record in measurement["per_record"]
    ]
    records_path.write_text(
        json.dumps(_clean(per_record), indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return [metrics_path, records_path]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    """Command-line parser."""
    parser = argparse.ArgumentParser(
        description="Validate the ml.sim2f twin against real 2f-bench pickles."
    )
    parser.add_argument(
        "--limit-files",
        type=int,
        default=None,
        help="Cap the number of pickles (sorted), for a smoke run; default: all",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        help=f"Output directory (default: {OUT_DIR.relative_to(ROOT).as_posix()})",
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Measure and write JSON + markdown only (figures are slow on some hosts)",
    )
    return parser


def _print_summary(
    result: dict[str, Any], gate: dict[str, Any], measurement: dict[str, Any]
) -> None:
    """Print the compact terminal summary."""
    sample = result["sample"]
    duplication = measurement["duplication"]
    print()
    print(
        f"records: {duplication['records']} -> distinct pairs: "
        f"{duplication['distinct']} (x{duplication['replication_factor']:.1f})"
    )
    print(
        f"k* = {result['k_star']:g}  "
        f"(先验 {K_APRIORI:g}); 内部性: {result['geometry']['interior']['interior']}; "
        f"三轮一致: {result['geometry']['rounds_agree']}; "
        f"几何可辨识: {result['geometry']['identified']}"
    )
    print(
        f"@ k*: ncc median {sample['ncc']['median']:+.4f} "
        f"[{sample['ncc']['q25']:+.4f}, {sample['ncc']['q75']:+.4f}], "
        f"L1 median {sample['l1']['median']:.4f}"
    )
    print(
        f"G0-4 @ k=1: min {gate['min']:+.4f} "
        f"({'pass' if gate['passes_min'] else 'FAIL'}); "
        f"median {gate['median']:+.4f} "
        f"({'pass' if gate['passes_median'] else 'FAIL'})"
    )
    print(f"verdict: {result['verdict']} — {result['reason']}")
    print(f"boundary: {result['boundary']}")


def main(argv: Sequence[str] | None = None) -> int:
    """Measure, score, and write the report.

    Args:
        argv: Arguments after the script name; ``None`` means ``sys.argv[1:]``.

    Returns:
        Process exit code: 0 when the report was written, 2 for an unusable run
        (no records). A ``not_similar`` verdict is **not** an error.
    """
    args = _build_parser().parse_args(argv)
    out_dir: Path = args.out_dir
    print(PREREGISTRATION)

    started = time.perf_counter()
    measurement = measure_corpus(args.limit_files)
    if not measurement["per_record"]:
        print(
            f"error: no records under {measurement['config']['corpus_root']}/"
            f"{CORPUS_PATTERN}",
            file=sys.stderr,
        )
        return 2

    result = score(measurement)
    gate = g04_answer(result)
    figures: list[str] = []
    if not args.skip_figures:
        figures = render_figures(measurement, result, out_dir / "figures")

    report = render_report(measurement, result, gate, figures)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    written = write_artefacts(measurement, result, gate, out_dir / "artifacts")

    _print_summary(result, gate, measurement)
    print()
    print(f"wrote {out_dir / 'report.md'}")
    for path in written:
        print(f"wrote {path}")
    for name in figures:
        print(f"wrote {out_dir / 'figures' / name}")
    print(f"wall clock {time.perf_counter() - started:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
