"""诊断 2f 实验系统数字孪生 (``ml.sim2f``) 的远场尺度 / 离焦标定是否已失效。

纯离线, 只读 ``data/debug/`` 与已入库源码; 不碰硬件, 不改任何已入库文件。

两条独立假说
============

* **H1 (远场尺度)**: ``region_span_px`` (128 格 ↔ 面板 px 的映射) 是否已失效?
  在 ``[600, 1200]`` (步长 25) 上扫描, 判据见 :data:`PREREGISTRATION`。
* **H2 (离焦 / 零级)**: ``camera_defocus = 10.0847 mm`` / ``zero_order_eta = 0.1``
  是否已失效? 走既有的 ``bench_twin_calibrate.fit_twin`` 反演 (Stage 1 + Stage 2)。

口径 (全部尺度无关)
==================

* 语料: ``data/debug/model_in_loop_hw_collect/20261025*`` (54 个 pkl / 312 条记录),
  按内容哈希去重后 **18 组不同的 (相位, 帧)**。
* 相位: ``ml.hwdataset.records.Materialiser`` 的相干块平均 (128x128, 瞳面半径 450 px)。
* 实测帧: ``farfield_frame_to_grid(..., 64, "raw")``, 以实测帧 argmax 为锚。
* 孪生: ``ml.sim2f.optics.Sim2f`` (CPU), 相位以 (cos, sin) 相量喂入。
* 几何映射: 复用 ``validate_sim2f_vs_hw_corpus.py`` 的 ``sample_twin``
  (孪生 CCD 像元 / 实测像元 = k), a-priori 取 k = 1。
* 度量只有两种: 零均值 ncc 与归一化 L1 (均尺度无关)。宽度比只作诊断。
  **禁止** 用绝对 Strehl / PIB / EE / FWHM / 峰值 / 总能量作判据。

耦合关系 (扫描前查清, 结论见报告 §2.1)
======================================

``Sim2fConfig`` 里 ``region_span_px`` 只通过**导出量**起作用, 没有任何独立旋钮跟着动:

===========================  ==========================  =====================
导出量                      依赖                        物理含义
===========================  ==========================  =====================
``pupil_pitch_m``            ∝ region_span_px            瞳面采样节距
``far_field_pitch_m``        ∝ 1 / region_span_px        焦面模型网格节距
``sampling_ratio``           ∝ 1 / region_span_px        相机像元 / 模型采样点
``beam_w0_model_px``         ∝ 1 / region_span_px        入射高斯腰 (模型采样点)
``aperture_m``               ∝ region_span_px            等效孔径 (诊断量)
``psf_scale_m``              ∝ 1 / region_span_px        衍射极限标度 (诊断量)
===========================  ==========================  =====================

关键事实: **入射束腰的物理尺寸 = ``beam_w0_px`` × 面板像元, 与 region_span_px 无关**
(``beam_w0_model_px × pupil_pitch_m`` 相消), 因此 0 级光斑的**物理**宽度对
region_span_px 不变; 变的是**相位图样**在焦面的物理尺度 (∝ 1/region_span_px)。
这是 H1 扫描能测到、而不能测到的东西, 报告 §2.2 明确区分。

术语
====

实验系统 / 实验 / 瞳面 / 物理预测模型 / 参考（真值）/ 变体。
"""

from __future__ import annotations

import argparse
from loguru import logger
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
# 顺序要紧: ROOT/src 必须在最前 (``ml`` 解析到 src/ml), ROOT 在其后 (``scripts`` 可解析)。
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
from ml.sim2f.config import OpticsErrors, Sim2fConfig
from ml.sim2f.optics import Sim2f
from scripts.ml.validate_sim2f_vs_hw_corpus import (
    CORPUS_PATTERN,
    CORPUS_ROOT,
    GRID_MEASURED,
    GRID_PHASE,
    K_APRIORI,
    MEASURED_ANCHOR_PX,
    PANEL_RADIUS_PX,
    _digest,
    describe,
    fwhm_px,
    normalised_l1,
    sample_twin,
    zero_mean_ncc,
)

# ---------------------------------------------------------------------------
# 预登记判据 —— 写于任何统计量被计算之前; 之后一个字都不许改
# ---------------------------------------------------------------------------

#: H1 扫描网格 (``region_span_px``, 面板 px)。
REGION_SPAN_GRID: tuple[int, ...] = tuple(range(600, 1201, 25))

#: H1 (a) 宽度比合格带 (实测 / 孪生, 同一相位, k = 1)。
H1_WIDTH_BAND: tuple[float, float] = (0.90, 1.10)

#: H1 (b) 语料中位 ncc 门限 (k = 1, a-priori, 不做缩放补偿)。
H1_NCC_BAR: float = 0.75

#: H1 (c) 内部性: 左右邻居的中位 ncc 各低至少这么多。
H1_INTERIOR_MARGIN: float = 0.01

#: H2 (a) 反演出的 ``defocus_m`` 与历史值 10.0847 mm 的相对差门限。
H2_DEFOUS_REL_DELTA: float = 0.20

#: H2 (b) 反演后语料中位 ncc 的提升门限 (相对基线配置)。
H2_NCC_GAIN: float = 0.05

#: H2 (c) 归一化 L1 门限。
H2_L1_BAR: float = 0.50

#: H2 反演的历史基准值 (m), 用于相对差。
HISTORICAL_DEFOUS_M: float = 0.010084735

#: H2 反演的边界覆盖 (m)。模块默认的 ±0.5 mm **不包含** 历史值 10.08 mm, 且
#: ``_clip_layout`` 会把初值裁进边界 —— 必须显式放宽, 否则初值当场被毁掉。
H2_DEFOUS_BOUNDS: tuple[float, float] = (-0.02, 0.02)

#: H2 反演用的面板 Zernike 模式 (Noll 4 = 面板离焦, 与相机离焦近似简并)。
H2_PANEL_ZERNIIKE_MODES: tuple[int, ...] = (4,)

#: H2 反演用的 numpy 孪生裁剪窗口 (模型采样点)。32 格 × 4.618 µm = 147.8 µm,
#: 覆盖实测 64 px × 2.2 µm = 140.8 µm 的窗口 (略宽, 边缘按复制外推)。
H2_CROP_PX: int = 32

PREREGISTRATION = """\
预登记判据 (写于任何统计量被计算之前; 之后一个字都不许改, 更不许放宽)

== 语料与口径 ==
1. 语料: `data/debug/model_in_loop_hw_collect/20261025*` (54 个 pkl, 312 条记录), 按
   (相位格, 实测帧) 的内容哈希去重后 18 组不同的对。只读, 不写不删不改名。
2. 相位: `ml.hwdataset.records.Materialiser` 相干块平均到 128x128 (瞳面半径 450 px);
   孪生以 (cos, sin) 相量喂入 (孪生自己归一), 避免在多帧相干平均过的相位上再做 arctan2 往返。
3. 实测帧: `farfield_frame_to_grid(..., 64, "raw")`, 以实测帧 argmax (3x3 去尖峰) 为锚。
4. 孪生: `Sim2f` (CPU), `forward(..., detector=False)`; 256x256 强度帧。
5. 度量: 零均值 ncc; 归一化 L1 = Σ|a/Σa − b/Σb| ∈ [0,2]。宽度比 (实测/孪生, 同一相位,
   k = 1) 只作诊断。禁止用绝对 Strehl / PIB / EE / FWHM / 峰值 / 总能量作判据。
6. 几何映射: 复用 `validate_sim2f_vs_hw_corpus.py` 的 `sample_twin` (孪生 CCD 像元 / 实测像元
   = k)。a-priori 取 k = 1 (两侧像元节距相同); 判据里不做任何缩放补偿。

== H1 (远场尺度: region_span_px) ==
7. 扫描网格 (预登记): region_span_px ∈ {600, 625, ..., 1200} (25 点)。只动这一个旋钮;
   其余 (grid, far_field_padding, ccd_px, ccd_pixel_um, beam_w0_px, errors) 全部保持
   `Sim2fConfig.calibrated()` 的值 —— 耦合量 (pupil_pitch_m / far_field_pitch_m /
   sampling_ratio / beam_w0_model_px / psf_scale_m) 是**导出量**, 随 region_span_px 自动变。
8. H1 接受: 存在 region_span_px 使得
   (a) 同相位光斑宽度比 (实测 / 孪生, k = 1) 的中位数 ∈ [0.90, 1.10], 且
   (b) 该配置下语料中位 ncc (k = 1, a-priori) >= 0.75, 且
   (c) 该点在扫描网格上是**内点** (左右邻居的中位 ncc 各低 >= 0.01)。
   -> 不满足 (c) 一律记「尺度未被识别」, 不许取边界值当答案。
9. H1 三值: 接受 (8 全中) / 否决 (无任何点满足 (a) 与 (b)) / 未被识别 (有点满足 (a)(b),
   但该点不是内点, 或整条曲线没有可辨认的内点极大)。
10. 曲线必报: 宽度比 vs 网格、中位 ncc vs 网格、最小 ncc vs 网格; 标出内点 / 平台。
11. 宽度比与 ncc 同在 k = 1 上取 (不做缩放补偿): 若允许为每个 region_span_px 单独挑一个 k,
    那就把「尺度对不对」这个待检量又变成了自由旋钮。上一条 lane 落盘的 0.544 是 k = 2 上的
    口径, 本文另列一列 k = 2 的诊断值以便横向比读, 但**不进判据**。

== H2 (离焦 / 零级: camera_defocus, zero_order_eta) ==
12. 反演入口: 既有的 `bench_twin_calibrate.fit_twin` (不另写一份拟合逻辑)。Stage 1 与
    Stage 2 都要跑; Stage 2 由模块自身**同时放开 Stage 1 的粗参数** (面板 Noll 4 与相机离焦
    在远场近似简并, 只精调空间参数无法打破)。seed 固定, 两次运行逐位一致。
13. Stage 1 的 defocus_m 边界必须显式放宽到覆盖历史值 10.0847 mm —— 模块默认边界 ±0.5 mm
    不含它, 且模块会把初值裁进边界。
14. H2 接受: 反演出的 defocus_m 与 10.0847 mm 相差 **> 20%**, 且反演后语料中位 ncc
    (k = 1) 提升 **>= 0.05**, 且归一化 L1 **<= 0.5**。
15. 基线配置: 若 H1 接受则取 H1 接受配置; 否则取 region_span_px = 900 (历史标定值)。
    两假说都接受时只报「H1+H2 联合」这一种配置, 不许再做第三轮微调 (那是 winner's curse)。
16. Stage 2 的空间参数若不可辨识 (与相机离焦简并) 就明确说不可辨识, 不许硬报数。
17. 零级占比 zero_order_eta: numpy 孪生**没有**这个旋钮, `fit_twin` 反演不到它;
    只能作诊断扫描, 不进 H2 判据。

== 复现与诚实 ==
18. 任何阈值都不因实测结果而调整; 未达到就如实记为否决, 并给诊断, 不找理由。
19. 全部原始数字落 JSON; 两次运行逐位一致 (固定 seed / 固定语料顺序)。
"""


# ---------------------------------------------------------------------------
# 输出目录
# ---------------------------------------------------------------------------

OUT_DIR = ROOT / "report" / "sim2f_bench" / "optical_scale_20261027"
FIG_DIR = OUT_DIR / "figures"
ART_DIR = OUT_DIR / "artifacts"

#: 溯源登记: 本次**不**改 `scripts/_common/provenance.py` (另一条 lane 在动那个文件)。
PROVENANCE_PENDING = (
    "**待登记**: 本报告的生成脚本是 `scripts/ml/diagnose_sim2f_optical_scale.py`, "
    "但本次**不**改 `scripts/_common/provenance.py` (另一条 lane 正在改那个文件), "
    "因此 `scripts/sync_report_provenance.py` 尚不能为本文注入溯源块; 登记完成后重跑即可。"
)


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------

_CJK_FONTS = ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans")


def configure_fonts() -> None:
    """配置中文字体 (matplotlib 必须先 `use("Agg")` 再设字体)。"""
    from matplotlib import font_manager

    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in _CJK_FONTS:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, *plt.rcParams["font.sans-serif"]]
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["savefig.dpi"] = 150


def fmt(value: Any, spec: str = ".4g") -> str:
    """数值格式化 (NaN / inf 显示为 n/a)。"""
    if value is None:
        return "n/a"
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return str(int(value))
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "n/a"
    return format(number, spec)


def jsonable(value: Any) -> Any:
    """把 numpy 标量 / 数组转成 JSON 能吃的类型 (非有限浮点写 null)。"""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


# ---------------------------------------------------------------------------
# 语料
# ---------------------------------------------------------------------------


def load_pairs(
    limit_files: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """读语料并按内容哈希去重, 返回 18 组不同的 (相位相量, 实测帧) 与语料元数据。

    判同口径与 ``validate_sim2f_vs_hw_corpus.py`` 完全一致 (同一个 ``_digest``),
    所以两组「18 对」是同一组。

    Args:
        limit_files: 只读前 N 个 pkl (冒烟测试用); ``None`` = 全量。

    Returns:
        ``(pairs, meta)``; ``pairs`` 每项含 ``phasor`` (1,2,128,128) float32、
        ``measured`` (64,64) float64、元数据与 ``digest``。
    """
    index = build_hw_index(
        [str(CORPUS_ROOT / CORPUS_PATTERN)], limit_files=limit_files, progress_every=0
    )
    materialiser = Materialiser(
        MaterialiserConfig(grid=GRID_PHASE, panel_radius=PANEL_RADIUS_PX), use_cache=False
    )
    replicas: dict[str, int] = {}
    pairs: list[dict[str, Any]] = []
    exposures: set[float] = set()
    for ref in index.records:
        cos_grid, sin_grid = materialiser.phase_grid(ref)
        payload = materialiser.store.get(ref)
        measured = farfield_frame_to_grid(payload["_img"], GRID_MEASURED, mode="raw")
        digest = _digest(cos_grid, sin_grid, measured)
        replicas[digest] = replicas.get(digest, 0) + 1
        exposure = getattr(ref, "exposure_ms", None)
        if exposure is not None:
            exposures.add(float(exposure))
        if any(pair["digest"] == digest for pair in pairs):
            continue
        pairs.append(
            {
                "phasor": torch.as_tensor(
                    np.stack([cos_grid, sin_grid])[None], dtype=torch.float32
                ),
                "measured": np.asarray(measured, dtype=np.float64),
                "mode_id": payload.get("mode_id"),
                "coefficient": payload.get("coefficient"),
                "probe": payload.get("probe"),
                "digest": digest,
            }
        )
    pairs.sort(key=lambda item: item["digest"])
    for pair in pairs:
        pair["replicas"] = replicas[pair["digest"]]
    meta = {
        "corpus_root": "data/debug/model_in_loop_hw_collect",
        "corpus_pattern": CORPUS_PATTERN,
        "records": len(index.records),
        "files": index.files_scanned,
        "replica_histogram": {
            str(count): sum(1 for v in replicas.values() if v == count)
            for count in sorted(set(replicas.values()))
        },
        "exposure_ms_values": sorted(exposures),
        "digests": [pair["digest"] for pair in pairs],
    }
    return pairs, meta


def cross_check_against_prior_lane(digests: Sequence[str]) -> dict[str, Any]:
    """与上一条 lane 落盘的 ``per_record.json`` 对账, 证明是同一组 18 对。

    Args:
        digests: 本次去重得到的摘要列表。

    Returns:
        对账结果 (对方存在 / 摘要集合是否一致)。
    """
    path = (
        ROOT
        / "report"
        / "sim2f_bench"
        / "twin_vs_hw_corpus"
        / "20261027"
        / "artifacts"
        / "per_record.json"
    )
    if not path.exists():
        return {"available": False, "reason": f"找不到 {path.name}"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"available": False, "reason": "对方 per_record.json 读不出来"}
    prior = {str(item.get("digest")) for item in payload if isinstance(item, dict)}
    prior.discard("None")
    return {
        "available": True,
        "prior_distinct": len(prior),
        "digests_match": prior == set(digests),
        "only_here": sorted(set(digests) - prior),
        "only_prior": sorted(prior - set(digests)),
    }


# ---------------------------------------------------------------------------
# H1: region_span_px 扫描
# ---------------------------------------------------------------------------


def scan_region_span(
    pairs: Sequence[dict[str, Any]], spans: Sequence[int]
) -> list[dict[str, Any]]:
    """在每个 region_span_px 上算中位 ncc / 最小 ncc / 宽度比 / 中位 L1 (k = 1)。

    除判据用的 k = 1 之外, 另记一列 k = 2 的诊断值 (与上一条 lane 落盘的
    0.544 宽度比同口径, 便于横向比读)。

    Args:
        pairs: :func:`load_pairs` 的输出。
        spans: 要扫描的 region_span_px 列表。

    Returns:
        每个跨度一行的 dict (含导出量与逐对数值)。
    """
    rows: list[dict[str, Any]] = []
    measured_widths = [fwhm_px(pair["measured"]) for pair in pairs]
    for span in spans:
        cfg = replace(Sim2fConfig.calibrated(), region_span_px=float(span))
        bench = Sim2f(cfg, device="cpu")
        nccs: list[float] = []
        l1s: list[float] = []
        ratios: list[float] = []
        nccs_k2: list[float] = []
        ratios_k2: list[float] = []
        twin_widths: list[float] = []
        for pair in pairs:
            with torch.no_grad():
                twin = bench.forward(pair["phasor"], detector=False)[0, 0].numpy()
            sampled = sample_twin(twin, K_APRIORI)
            sampled_k2 = sample_twin(twin, 2.0)
            nccs.append(zero_mean_ncc(pair["measured"], sampled))
            l1s.append(normalised_l1(pair["measured"], sampled))
            twin_width = fwhm_px(sampled)
            twin_widths.append(twin_width)
            ratios.append(
                fwhm_px(pair["measured"]) / twin_width
                if np.isfinite(twin_width) and twin_width > 0.0
                else float("nan")
            )
            nccs_k2.append(zero_mean_ncc(pair["measured"], sampled_k2))
            twin_width_k2 = fwhm_px(sampled_k2)
            ratios_k2.append(
                fwhm_px(pair["measured"]) / twin_width_k2
                if np.isfinite(twin_width_k2) and twin_width_k2 > 0.0
                else float("nan")
            )
        rows.append(
            {
                "region_span_px": int(span),
                "derived": {
                    "pupil_pitch_m": cfg.pupil_pitch_m,
                    "far_field_pitch_m": cfg.far_field_pitch_m,
                    "sampling_ratio": cfg.sampling_ratio,
                    "beam_w0_model_px": cfg.beam_w0_model_px,
                    "aperture_m": cfg.aperture_m,
                    "psf_scale_m": cfg.psf_scale_m,
                    "fft_length": cfg.fft_length,
                },
                "ncc": describe(nccs),
                "l1": describe(l1s),
                "width_ratio": describe(ratios),
                "ncc_k2": describe(nccs_k2),
                "width_ratio_k2": describe(ratios_k2),
                "twin_width_px": describe(twin_widths),
                "measured_width_px": describe(measured_widths),
                "per_pair": {
                    "ncc": nccs,
                    "l1": l1s,
                    "width_ratio": ratios,
                    "ncc_k2": nccs_k2,
                    "width_ratio_k2": ratios_k2,
                },
            }
        )
    return rows


def judge_h1(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """按预登记判据给 H1 三值结论。

    Args:
        rows: :func:`scan_region_span` 的输出。

    Returns:
        ``{"verdict", "reason", ...}``。
    """
    medians = [row["ncc"]["median"] for row in rows]
    best_index = int(np.nanargmax(medians))
    candidates: list[int] = []
    for index, row in enumerate(rows):
        ratio_ok = H1_WIDTH_BAND[0] <= row["width_ratio"]["median"] <= H1_WIDTH_BAND[1]
        ncc_ok = row["ncc"]["median"] >= H1_NCC_BAR
        if ratio_ok and ncc_ok:
            candidates.append(index)
    interior: list[int] = []
    for index in candidates:
        if index in (0, len(rows) - 1):
            continue
        left = medians[index - 1]
        right = medians[index + 1]
        if (
            np.isfinite(left)
            and np.isfinite(right)
            and left <= medians[index] - H1_INTERIOR_MARGIN
            and right <= medians[index] - H1_INTERIOR_MARGIN
        ):
            interior.append(index)
    if interior:
        pick = interior[0]
        verdict = "接受"
        reason = (
            f"region_span_px = {rows[pick]['region_span_px']} 同时满足 "
            f"(a) 宽度比中位数 {fmt(rows[pick]['width_ratio']['median'], '.4f')} ∈ "
            f"[{H1_WIDTH_BAND[0]}, {H1_WIDTH_BAND[1]}], (b) 中位 ncc "
            f"{fmt(medians[pick], '+.4f')} >= {H1_NCC_BAR}, (c) 内点"
        )
    elif candidates:
        verdict = "未被识别"
        reason = (
            f"有 {len(candidates)} 个点满足 (a)(b), 但都不是内点 (左右邻居的中位 ncc 各低 "
            f"< {H1_INTERIOR_MARGIN}), 故尺度未被识别; 不许取边界值当答案"
        )
    else:
        verdict = "否决"
        reason = (
            f"{len(rows)} 个 region_span_px 上没有一点同时满足 (a) 宽度比 ∈ "
            f"[{H1_WIDTH_BAND[0]}, {H1_WIDTH_BAND[1]}] 与 (b) 中位 ncc >= {H1_NCC_BAR}"
        )
    return {
        "verdict": verdict,
        "reason": reason,
        "candidate_indices": candidates,
        "interior_indices": interior,
        "best_index": best_index,
        "best_region_span_px": rows[best_index]["region_span_px"],
        "best_median_ncc": medians[best_index],
        "interior_margin": H1_INTERIOR_MARGIN,
        "width_band": list(H1_WIDTH_BAND),
        "ncc_bar": H1_NCC_BAR,
    }


# ---------------------------------------------------------------------------
# H2: 用既有的 bench_twin_calibrate.fit_twin 反演
# ---------------------------------------------------------------------------


def _numpy_twin_config(region_span_px: float, crop_px: int, errors: Any) -> Any:
    """构造与 torch 孪生**同名映射**的 numpy ``BenchTwinConfig`` (parity 口径)。

    字段逐个照抄 ``Sim2fConfig.calibrated()``: region / region_span_px /
    far_field_padding / beam_w0_px / 波长 / 焦距 / 像元节距, 与
    ``tests/ao_shaping/ml/sim2f/test_twin_parity.py`` 的 ``_twin_config`` 同口径。

    Args:
        region_span_px: 瞳面网格覆盖的面板边长 (px)。
        crop_px: 焦面裁剪窗口 (模型采样点)。
        errors: numpy ``TwinErrorModel``。

    Returns:
        ``BenchTwinConfig``。
    """
    from ao_shaping.drivers.sim.bench_twin import BenchTwinConfig

    sim = Sim2fConfig.calibrated()
    return BenchTwinConfig(
        wavelength_m=sim.wavelength_m,
        slm_pixel_um=sim.panel_pixel_um,
        ccd_pixel_um=sim.ccd_pixel_um,
        focal_length_m=sim.focal_length_m,
        region=int(sim.grid),
        region_span_px=float(region_span_px),
        far_field_padding=int(sim.far_field_padding),
        beam_w0_px=float(sim.beam_w0_px),
        center_crop=True,
        crop_px=int(crop_px),
        peak_adu=100.0,
        noise_adu=0.0,
        seed=0,
        errors=errors,
    )


def _measured_onto_crop(
    measured: np.ndarray, crop_px: int, p_fft: float, j_axis: int
) -> np.ndarray:
    """把实测 64x64 帧重采样到 numpy 孪生的裁剪网格 (相机像元 -> 模型采样点)。

    映射链 (与 ``sample_twin`` 同源, 逐级可核):

    * 实测像元 i ↔ 孪生相机像元 ``128 + (i - 31.5)`` (a-priori k = 1);
    * 孪生相机像元 k ↔ 模型格 ``256 + (k - 128) / ratio``;
    * numpy 孪生裁剪格 j ↔ 模型格 ``start + j``, 光学轴在格 ``j_axis``。

    合并后: 裁剪格 j ↔ 实测像元 ``31.5 + (j - j_axis) * ratio``,
    其中 ``ratio = p_fft / ccd_pitch`` (一个模型格摊到多少个相机像元)。越界按边缘复制。

    Args:
        measured: 实测 64x64 帧。
        crop_px: 裁剪窗口边长 (模型采样点)。
        p_fft: 焦面模型网格节距 (m)。
        j_axis: 裁剪窗口内光学轴所在的格点下标。

    Returns:
        ``(crop_px, crop_px)`` float64 重采样帧。
    """
    ccd_pitch = float(Sim2fConfig.calibrated().ccd_pixel_um) * 1e-6
    ratio = float(p_fft) / ccd_pitch
    grid = np.arange(crop_px, dtype=np.float64)
    coords = MEASURED_ANCHOR_PX + (grid - j_axis) * ratio
    xs = np.clip(coords, 0.0, measured.shape[1] - 1.0)
    ys = np.clip(coords, 0.0, measured.shape[0] - 1.0)
    x0 = np.floor(xs).astype(int)
    y0 = np.floor(ys).astype(int)
    x1 = np.minimum(x0 + 1, measured.shape[1] - 1)
    y1 = np.minimum(y0 + 1, measured.shape[0] - 1)
    fx = (xs - x0)[None, :] * np.ones((crop_px, 1))
    fy = (ys - y0)[:, None] * np.ones((1, crop_px))
    return (
        measured[np.ix_(y0, x0)] * (1 - fx) * (1 - fy)
        + measured[np.ix_(y0, x1)] * fx * (1 - fy)
        + measured[np.ix_(y1, x0)] * (1 - fx) * fy
        + measured[np.ix_(y1, x1)] * fx * fy
    )


def invert_defocus(
    pairs: Sequence[dict[str, Any]], region_span_px: float
) -> dict[str, Any]:
    """用 ``bench_twin_calibrate.fit_twin`` 反演相机离焦 (Stage 1 + Stage 2)。

    Args:
        pairs: :func:`load_pairs` 的输出。
        region_span_px: 反演用的瞳面网格覆盖面板边长 (px)。

    Returns:
        含 Stage 1 / Stage 2 的初值、收敛值、评估次数、耗时与可辨识性所需数据。
    """
    from ao_shaping.drivers.sim.bench_twin import BenchTwin, TwinErrorModel
    from ao_shaping.tools.slm.bench_twin_calibrate import TwinSample, fit_twin

    sim_errors = Sim2fConfig.calibrated().errors
    errors = TwinErrorModel(
        lut_gamma=sim_errors.lut_gamma,
        lut_phase_max_rad=sim_errors.lut_phase_max_rad,
        gray_levels=sim_errors.gray_levels,
        defocus_m=sim_errors.defocus_m,
        slm_zernike={mode: 0.0 for mode in H2_PANEL_ZERNIIKE_MODES},
        slm_zernike_n_max=4,
    )
    config = _numpy_twin_config(region_span_px, H2_CROP_PX, errors)
    twin = BenchTwin(config)
    m = twin.far_field_size
    start = (m - H2_CROP_PX) // 2
    j_axis = m // 2 - start

    samples: list[TwinSample] = []
    for index, pair in enumerate(pairs):
        phase_rad = np.arctan2(
            pair["phasor"][0, 1].numpy().astype(np.float64),
            pair["phasor"][0, 0].numpy().astype(np.float64),
        )
        measured = _measured_onto_crop(pair["measured"], H2_CROP_PX, twin.p_fft, j_axis)
        samples.append(TwinSample(phase_rad, measured, f"pair{index:02d}"))

    bounds = {
        "coarse": {"defocus_m": H2_DEFOUS_BOUNDS},
        "fine": {"slm_zernike": (-1.0, 1.0), "tilt": (-0.05, 0.05)},
    }

    started = time.perf_counter()
    result = fit_twin(
        samples,
        config,
        bounds=bounds,
        stage1=True,
        stage2=True,
        maxiter=12,
        popsize=8,
        maxiter_fine=60,
        seed=0,
        verbose=True,
    )
    elapsed = time.perf_counter() - started

    # Stage 1 单独的解 (用于暴露 Stage 2 的简并): 同一入口, 只跑 Stage 1。
    started_s1 = time.perf_counter()
    stage1_only = fit_twin(
        samples,
        config,
        bounds=bounds,
        stage1=True,
        stage2=False,
        maxiter=12,
        popsize=8,
        seed=0,
        verbose=False,
    )
    elapsed_s1 = time.perf_counter() - started_s1

    final = result.config.errors
    only = stage1_only.config.errors
    return {
        "region_span_px": float(region_span_px),
        "crop_px": H2_CROP_PX,
        "p_fft_m": float(twin.p_fft),
        "far_field_size": int(m),
        "j_axis": int(j_axis),
        "n_samples": len(samples),
        "bounds": {"defocus_m": list(H2_DEFOUS_BOUNDS)},
        "stage1_only": {
            "defocus_m": float(only.defocus_m),
            "lut_gamma": float(only.lut_gamma),
            "lut_phase_max_rad": float(only.lut_phase_max_rad),
            "loss": float(stage1_only.loss),
            "stages": {k: float(v) for k, v in stage1_only.stages.items()},
            "seconds": float(elapsed_s1),
            "n_evaluations": int(stage1_only.n_evaluations),
        },
        "stage1_plus_stage2": {
            "defocus_m": float(final.defocus_m),
            "lut_gamma": float(final.lut_gamma),
            "lut_phase_max_rad": float(final.lut_phase_max_rad),
            "slm_zernike": {
                int(k): float(v) for k, v in sorted(final.slm_zernike.items())
            },
            "tilt_x_rad": float(final.tilt_x_rad),
            "tilt_y_rad": float(final.tilt_y_rad),
            "loss": float(result.loss),
            "stages": {k: float(v) for k, v in result.stages.items()},
            "seconds": float(elapsed),
            "n_evaluations": int(result.n_evaluations),
        },
        "historical_defocus_m": HISTORICAL_DEFOUS_M,
        "relative_change": abs(float(final.defocus_m) - HISTORICAL_DEFOUS_M)
        / HISTORICAL_DEFOUS_M,
    }


def corpus_metrics(
    pairs: Sequence[dict[str, Any]], region_span_px: float, errors: OpticsErrors
) -> dict[str, Any]:
    """在给定 region_span_px 与误差模型下算语料中位 ncc / 最小 ncc / 中位 L1 (k = 1)。

    Args:
        pairs: :func:`load_pairs` 的输出。
        region_span_px: 瞳面网格覆盖面板边长 (px)。
        errors: 光学误差模型。

    Returns:
        ``{"ncc": describe(...), "l1": describe(...), "width_ratio": describe(...)}``。
    """
    cfg = replace(
        Sim2fConfig.calibrated(), region_span_px=float(region_span_px), errors=errors
    )
    bench = Sim2f(cfg, device="cpu")
    nccs: list[float] = []
    l1s: list[float] = []
    ratios: list[float] = []
    for pair in pairs:
        with torch.no_grad():
            twin = bench.forward(pair["phasor"], detector=False)[0, 0].numpy()
        sampled = sample_twin(twin, K_APRIORI)
        nccs.append(zero_mean_ncc(pair["measured"], sampled))
        l1s.append(normalised_l1(pair["measured"], sampled))
        twin_width = fwhm_px(sampled)
        ratios.append(
            fwhm_px(pair["measured"]) / twin_width
            if np.isfinite(twin_width) and twin_width > 0.0
            else float("nan")
        )
    return {
        "ncc": describe(nccs),
        "l1": describe(l1s),
        "width_ratio": describe(ratios),
    }


def scan_zero_order_eta(
    pairs: Sequence[dict[str, Any]], region_span_px: float
) -> list[dict[str, Any]]:
    """诊断扫描 zero_order_eta (numpy 孪生没有这个旋钮, 反演不到它)。

    Args:
        pairs: :func:`load_pairs` 的输出。
        region_span_px: 瞳面网格覆盖面板边长 (px)。

    Returns:
        每个 eta 一行的中位 ncc / 最小 ncc / 中位 L1。
    """
    rows: list[dict[str, Any]] = []
    for eta in (0.0, 0.05, 0.1, 0.15, 0.2, 0.3):
        errors = replace(Sim2fConfig.calibrated().errors, zero_order_eta=float(eta))
        metrics = corpus_metrics(pairs, region_span_px, errors)
        rows.append(
            {
                "zero_order_eta": float(eta),
                "ncc_median": metrics["ncc"]["median"],
                "ncc_min": metrics["ncc"]["min"],
                "l1_median": metrics["l1"]["median"],
            }
        )
    return rows


# ---------------------------------------------------------------------------
# 图
# ---------------------------------------------------------------------------


def render_figures(
    h1_rows: Sequence[dict[str, Any]],
    h1: dict[str, Any],
    eta_rows: Sequence[dict[str, Any]],
) -> list[str]:
    """出图: H1 三条曲线 + 零级占比诊断。

    Args:
        h1_rows: :func:`scan_region_span` 的输出。
        h1: :func:`judge_h1` 的输出。
        eta_rows: :func:`scan_zero_order_eta` 的输出。

    Returns:
        写出的图文件名列表。
    """
    from scripts._common import savefig

    names: list[str] = []
    spans = [row["region_span_px"] for row in h1_rows]
    med = [row["ncc"]["median"] for row in h1_rows]
    mins = [row["ncc"]["min"] for row in h1_rows]
    ratios = [row["width_ratio"]["median"] for row in h1_rows]

    figure, axes = plt.subplots(2, 1, figsize=(8.2, 7.0), sharex=True)
    axes[0].plot(spans, med, "o-", color="tab:blue", label="中位 ncc (k=1)")
    axes[0].plot(spans, mins, "s--", color="tab:orange", alpha=0.85, label="最小 ncc (k=1)")
    axes[0].axhline(H1_NCC_BAR, color="tab:red", linestyle=":", label=f"门限 {H1_NCC_BAR:g}")
    best = h1["best_index"]
    axes[0].plot(
        [spans[best]],
        [med[best]],
        "o",
        ms=12,
        mfc="none",
        mec="tab:purple",
        label="曲线极大值",
    )
    for index in h1["interior_indices"]:
        axes[0].plot(
            [spans[index]],
            [med[index]],
            "s",
            ms=12,
            mfc="none",
            mec="tab:green",
            label="内点" if index == h1["interior_indices"][0] else None,
        )
    axes[0].set_ylabel("ncc")
    axes[0].set_title("H1: region_span_px 扫描 (a-priori k=1, 不做缩放补偿)")
    axes[0].legend(loc="lower right", fontsize=8)
    axes[0].grid(alpha=0.25)

    axes[1].plot(spans, ratios, "o-", color="tab:purple")
    axes[1].axhspan(
        H1_WIDTH_BAND[0],
        H1_WIDTH_BAND[1],
        color="tab:green",
        alpha=0.15,
        label=f"合格带 [{H1_WIDTH_BAND[0]}, {H1_WIDTH_BAND[1]}]",
    )
    axes[1].set_xlabel("region_span_px (面板 px)")
    axes[1].set_ylabel("光斑宽度比 (实测 / 孪生)")
    axes[1].legend(loc="lower right", fontsize=8)
    axes[1].grid(alpha=0.25)
    savefig(figure, FIG_DIR / "h1_region_span_scan.png")
    names.append("h1_region_span_scan.png")

    figure, axis = plt.subplots(figsize=(6.4, 4.0))
    axis.plot(
        [row["zero_order_eta"] for row in eta_rows],
        [row["ncc_median"] for row in eta_rows],
        "o-",
        label="中位 ncc (k=1)",
    )
    axis.plot(
        [row["zero_order_eta"] for row in eta_rows],
        [row["ncc_min"] for row in eta_rows],
        "s--",
        label="最小 ncc (k=1)",
    )
    axis.axvline(0.1, color="tab:red", linestyle=":", label="历史值 0.1")
    axis.set_xlabel("zero_order_eta")
    axis.set_ylabel("ncc")
    axis.set_title("诊断: 零级占比扫描 (不进 H2 判据)")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    savefig(figure, FIG_DIR / "h2_zero_order_eta_scan.png")
    names.append("h2_zero_order_eta_scan.png")
    return names


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------


def render_report(payload: dict[str, Any], figures: Sequence[str]) -> str:
    """渲染报告正文。

    Args:
        payload: 全部结果。
        figures: 图文件名列表。

    Returns:
        Markdown 文本。
    """
    from scripts._common import markdown_table

    h1 = payload["h1"]
    h2 = payload["h2"]
    corpus = payload["corpus"]
    lines: list[str] = []
    add = lines.append

    add("# 2f 实验系统数字孪生的远场尺度 / 离焦标定是否已失效 (纯离线诊断)")
    add("")
    add(f"- H1 ({h1['verdict']}): {h1['reason']}")
    add(f"- H2 ({h2['verdict']}): {h2['reason']}")
    add(f"- 一句话: {payload['one_liner']}")
    add(f"- 溯源: {PROVENANCE_PENDING}")
    add("")
    for name in figures:
        add(f"![{name}](figures/{name})")
        add("")

    add("## 0. 预注册判据 (写于任何统计量被计算之前)")
    add("")
    add("```")
    add(PREREGISTRATION)
    add("```")
    add("")
    add(
        "判据原文来自任务书, 在扫描与反演**之前**写进 "
        "`scripts/ml/diagnose_sim2f_optical_scale.py` 的模块常量 "
        "(`PREREGISTRATION` / `H1_*` / `H2_*`), 之后一字未改。"
    )
    add("")

    add("## 1. 语料与口径")
    add("")
    add(
        f"语料 `{corpus['corpus_root']}/{corpus['corpus_pattern']}`: "
        f"{corpus['records']} 条记录 / {corpus['files']} 个 pkl, 内容哈希去重后 "
        f"**{corpus['distinct']} 组**不同的 (相位, 帧); 相位与实测帧的还原与 "
        "`validate_sim2f_vs_hw_corpus.py` 完全同源 (同一个 `_digest`), 所以与上一条 lane "
        "是同一组 18 对 (全量运行时; 见下表的摘要对账)。"
    )
    add("")
    add(
        "度量只有两种尺度无关的量: 零均值 ncc 与归一化 L1 = Σ|a/Σa − b/Σb|; "
        "宽度比 (实测 / 孪生, 同一相位) 只作诊断。绝对 Strehl / PIB / EE / FWHM / 峰值 / "
        "总能量一律不作判据。"
    )
    add("")
    add("### 1.1 语料元数据")
    add("")
    add(
        markdown_table(
            ["项", "值"],
            [
                ["记录数 / pkl 数", f"{corpus['records']} / {corpus['files']}"],
                ["去重后不同的对", str(corpus["distinct"])],
                [
                    "复制倍数分布",
                    ", ".join(
                        f"{count}x×{n}" for count, n in corpus["replica_histogram"].items()
                    ),
                ],
                ["曝光 (ms)", ", ".join(fmt(v, "g") for v in corpus["exposure_ms_values"])],
                ["孪生配置", "Sim2fConfig.calibrated()"],
                [
                    "几何映射",
                    "sample_twin (孪生 CCD 像元 / 实测像元 = k), a-priori k = 1",
                ],
                ["与上一条 lane 对账", corpus["cross_check"]["summary"]],
            ],
        )
    )
    add("")

    add("## 2. H1: region_span_px 扫描")
    add("")
    add("### 2.1 耦合关系 (扫描前查清)")
    add("")
    add(
        "`Sim2fConfig` 里 `region_span_px` **没有**任何独立旋钮跟着动; 它只通过导出量起作用。"
        "下表由脚本在扫描点上实测导出, 不是抄来的:"
    )
    add("")
    rows = []
    for row in h1["rows"][:: max(1, len(h1["rows"]) // 6)]:
        derived = row["derived"]
        rows.append(
            [
                str(row["region_span_px"]),
                fmt(derived["pupil_pitch_m"] * 1e6, ".3f"),
                fmt(derived["far_field_pitch_m"] * 1e6, ".3f"),
                fmt(derived["sampling_ratio"], ".4f"),
                fmt(derived["beam_w0_model_px"], ".3f"),
                fmt(derived["psf_scale_m"] * 1e6, ".3f"),
            ]
        )
    add(
        markdown_table(
            [
                "region_span_px",
                "pupil_pitch_m (µm)",
                "far_field_pitch_m (µm)",
                "sampling_ratio",
                "beam_w0_model_px",
                "psf_scale_m (µm)",
            ],
            rows,
        )
    )
    add("")
    add(
        "**关键事实**: 入射束腰的物理尺寸 = `beam_w0_px` × 面板像元, 与 `region_span_px` "
        "**无关** (`beam_w0_model_px × pupil_pitch_m` 相消)。因此 0 级光斑的**物理**宽度对 "
        "`region_span_px` 不变; 变的是**相位图样**在焦面的物理尺度 (∝ 1/region_span_px)。"
        "这条结构性事实决定了 H1 的 (a) 判据 (宽度比) 在原理上几乎不随这条旋钮动 —— "
        "§2.2 的曲线正是它的直接体现。"
    )
    add("")
    add("### 2.2 扫描曲线")
    add("")
    add(
        "**扫描前的几何探针 (披露)**: 在冻结上面的判据之前, 先跑过一次 5 点 "
        "(600/750/900/1050/1200) 的探针, 目的是查清耦合关系 (哪几个量真的随 "
        "region_span_px 动) 与确认 k=1 的映射自洽。探针**没有**产生新的判据, 也**没有**"
        "改动任何阈值; 它只是让本次扫描的网格与报告里 §2.1 的耦合表有据可依。"
    )
    add("")
    rows = []
    for row in h1["rows"]:
        rows.append(
            [
                str(row["region_span_px"]),
                fmt(row["ncc"]["median"], "+.4f"),
                fmt(row["ncc"]["min"], "+.4f"),
                fmt(row["ncc"]["q25"], "+.4f")
                + " .. "
                + fmt(row["ncc"]["q75"], "+.4f"),
                fmt(row["width_ratio"]["median"], ".4f"),
                fmt(row["l1"]["median"], ".4f"),
                fmt(row["ncc_k2"]["median"], "+.4f"),
                fmt(row["width_ratio_k2"]["median"], ".4f"),
            ]
        )
    add(
        markdown_table(
            [
                "region_span_px",
                "中位 ncc (k=1)",
                "最小 ncc (k=1)",
                "ncc IQR (k=1)",
                "宽度比中位 (k=1)",
                "中位 L1 (k=1)",
                "中位 ncc (k=2, 诊断)",
                "宽度比中位 (k=2, 诊断)",
            ],
            rows,
        )
    )
    add("")
    add(
        f"曲线极大值在 `region_span_px = {h1['best_region_span_px']}` "
        f"(中位 ncc {fmt(h1['best_median_ncc'], '+.4f')}); 内部性门限: "
        f"左右邻居各低 ≥ {h1['interior_margin']}。"
    )
    add("")
    add(
        "最后两列是 **诊断列**: 上一条 lane 落盘的宽度比 0.544 就是在 k = 2 上算的, "
        "这里保留同一口径以便横向比读; **判据只用 k = 1** 的两列 (判据原文: a-priori k = 1, "
        "不做任何缩放补偿) —— 若允许为每个 region_span_px 单独挑一个 k, "
        "「尺度对不对」这个待检量就又变成了自由旋钮。"
    )
    add("")
    add(f"**H1 结论: {h1['verdict']}** —— {h1['reason']}。")
    add("")

    add("## 3. H2: camera_defocus / zero_order_eta 反演")
    add("")
    add(
        "反演走既有入口 `ao_shaping.tools.slm.bench_twin_calibrate.fit_twin` "
        "(numpy 孪生, 与 torch 孪生同名映射同几何: region / region_span_px / "
        "far_field_padding / beam_w0_px 全部取自 `Sim2fConfig.calibrated()`), "
        "Stage 1 与 Stage 2 都跑; Stage 2 由模块自身**同时放开 Stage 1 的粗参数** "
        "(面板 Noll 4 与相机离焦在远场近似简并)。"
    )
    add("")
    add("### 3.1 Stage 1 / Stage 2 状态")
    add("")
    if not h2.get("fit"):
        add("(本次运行跳过了 H2 反演; 见 `artifacts/h2_fit.json` 的 `verdict` 字段。)")
        add("")
    else:
        add(
            markdown_table(
                ["项", "值"],
                [
                    ["初值 defocus_m (mm)", fmt(h2["initial_defocus_m"] * 1e3, ".4f")],
                    [
                        "Stage 1 defocus_m (mm)",
                        fmt(h2["fit"]["stage1_only"]["defocus_m"] * 1e3, ".4f"),
                    ],
                    [
                        "Stage 1 损失 / 评估次数 / 耗时 (s)",
                        f"{fmt(h2['fit']['stage1_only']['loss'], '.4f')} / "
                        f"{h2['fit']['stage1_only']['n_evaluations']} / "
                        f"{fmt(h2['fit']['stage1_only']['seconds'], '.1f')}",
                    ],
                    [
                        "Stage 1+2 defocus_m (mm)",
                        fmt(h2["fit"]["stage1_plus_stage2"]["defocus_m"] * 1e3, ".4f"),
                    ],
                    [
                        "Stage 1+2 面板 Noll 4 (rad)",
                        ", ".join(
                            fmt(v, ".4f")
                            for v in h2["fit"]["stage1_plus_stage2"]["slm_zernike"].values()
                        )
                        or "n/a",
                    ],
                    [
                        "Stage 1+2 tilt x / y (rad)",
                        f"{fmt(h2['fit']['stage1_plus_stage2']['tilt_x_rad'], '.4g')} / "
                        f"{fmt(h2['fit']['stage1_plus_stage2']['tilt_y_rad'], '.4g')}",
                    ],
                    [
                        "Stage 1+2 损失 / 评估次数 / 耗时 (s)",
                        f"{fmt(h2['fit']['stage1_plus_stage2']['loss'], '.4f')} / "
                        f"{h2['fit']['stage1_plus_stage2']['n_evaluations']} / "
                        f"{fmt(h2['fit']['stage1_plus_stage2']['seconds'], '.1f')}",
                    ],
                    [
                        "与历史值 10.0847 mm 的相对差",
                        fmt(h2["relative_change"] * 100, ".1f") + " %",
                    ],
                ],
            )
        )
        add("")
        add(f"**Stage 2 空间参数可辨识性**: {h2['identifiability']}")
        add("")
    add("### 3.2 零级占比 (诊断, 不进判据)")
    add("")
    add(
        "numpy 孪生**没有** `zero_order_eta` 这个旋钮, `fit_twin` 反演不到它 "
        "(预登记第 17 条), 因此只能作诊断扫描:"
    )
    add("")
    add(
        markdown_table(
            ["zero_order_eta", "中位 ncc", "最小 ncc", "中位 L1"],
            [
                [
                    fmt(row["zero_order_eta"], ".2f"),
                    fmt(row["ncc_median"], "+.4f"),
                    fmt(row["ncc_min"], "+.4f"),
                    fmt(row["l1_median"], ".4f"),
                ]
                for row in payload["eta_rows"]
            ],
        )
    )
    add("")

    add("## 4. 三值结论")
    add("")
    add(
        markdown_table(
            ["假说", "结论", "依据"],
            [
                ["H1 远场尺度 (region_span_px)", h1["verdict"], h1["reason"]],
                ["H2 离焦 / 零级", h2["verdict"], h2["reason"]],
            ],
        )
    )
    add("")
    add(f"**一句话**: {payload['one_liner']}")
    add("")

    add("## 5. 对「标定已失效」的推论与代价")
    add("")
    for line in payload["implications"]:
        add(line)
        add("")

    add("## 6. 复现")
    add("")
    add("```bash")
    add("python scripts/ml/diagnose_sim2f_optical_scale.py")
    add("```")
    add("")
    add(
        f"全量耗时 {fmt(payload['timings']['total_s'], '.1f')} s "
        f"(语料 {fmt(payload['timings']['corpus_s'], '.1f')} s, "
        f"H1 扫描 {fmt(payload['timings']['h1_s'], '.1f')} s, "
        f"H2 反演 {fmt(payload['timings']['h2_s'], '.1f')} s)。"
        "脚本不写 `data/debug/`, 不碰任何设备, 孪生在 CPU 上跑; "
        "seed 固定, 两次运行逐位一致。"
    )
    add("")
    add("产物:")
    add("")
    add("| 文件 | 内容 |")
    add("|---|---|")
    add("| `artifacts/metrics.json` | 全部数字 (H1 逐点 + 逐对、H2 各级、结论) |")
    add("| `artifacts/h1_scan.json` | H1 扫描网格与逐对数值 |")
    add("| `artifacts/h2_fit.json` | H2 反演的 Stage 1 / Stage 2 明细 |")
    for name in figures:
        add(f"| `figures/{name}` | 曲线图 |")
    add("")

    add("## 7. 坑")
    add("")
    for line in payload["pitfalls"]:
        add(line)
        add("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# 结论文字
# ---------------------------------------------------------------------------


def _implications(
    h1: dict[str, Any], h2: dict[str, Any], payload: dict[str, Any]
) -> list[str]:
    """写 §5 的推论与代价 (按 H1 / H2 的结论分支)。"""
    lines: list[str] = []
    h1_ok = h1["verdict"] == "接受"
    h2_ok = h2["verdict"] == "接受"
    if h1_ok:
        lines.append(
            f"**H1 接受 ⇒ 「128 格 ↔ 面板 px」的映射确实变了。** 被接受的 "
            f"`region_span_px = {h1['accepted_region_span_px']}` 与历史值 900 差 "
            f"{h1['accepted_delta_px']:+d} px。推论与代价如下。"
        )
        lines.append(
            "1. **已训练网络**: 语料 (标签/监督) 是按 `region_span_px = 900` 生成的。改了映射, "
            "网络学到的「相位 → 远场」映射就整体错位, 必须**重训** (或用新映射重生成语料后重训)。"
            "已入库的 checkpoint (`runs/sim2f_fair/train/model_best.pt`) 不能直接用。"
        )
        lines.append(
            "2. **面板嵌入**: `compose_pupil_dof(radius_px=450)` 把 128 格铺到 900 px 见方; "
            "映射变了以后这个半径要跟着改成 "
            f"`radius_px = {h1['accepted_region_span_px'] // 2}` (取整), 否则下发的相位"
            "落在错误的物理尺度上, 远场是错的而不是平移的。"
        )
    else:
        lines.append(
            f"**H1 未接受 ({h1['verdict']}) ⇒ 没有证据说「128 格 ↔ 面板 px」的映射变了。** "
            "因此**不需要**因为这条假说去重训网络或改面板嵌入半径。"
        )
        lines.append(
            "1. **已训练网络**: 语料仍按 `region_span_px = 900` 生成。若 H2 也被接受, "
            "那么网络学到的仍然是**旧离焦**下的远场, 换到新离焦后会有系统偏差 —— "
            "但这是离焦的问题, 不是映射的问题, 处置见下。"
        )
        lines.append(
            "2. **面板嵌入**: `compose_pupil_dof(radius_px=450)` 保持不变 "
            "(它与 `region_span_px = 900` 是配套的)。"
        )
    if h2_ok:
        lines.append(
            "3. **H2 接受 ⇒ 相机离焦确实失效。** 反演值 "
            f"{fmt(h2['fit']['stage1_plus_stage2']['defocus_m'] * 1e3, '.4f')} mm 与历史的 "
            "10.0847 mm 相差 "
            f"{fmt(h2['relative_change'] * 100, '.1f')} %。这意味着孪生里的焦面散焦量错了, "
            "凡是**依赖焦面绝对形态**的东西都要重做: (i) 孪生生成标签/监督的语料要按新离焦重生成, "
            "**已训练网络需要重训**; (ii) 面板嵌入半径**不动** (它与映射/离焦无关); "
            "(iii) 任何用孪生远场做验收门限 (G0-4 之类) 的基线数字都要重测。"
        )
    else:
        lines.append(
            "3. **H2 未接受 ⇒ 没有证据说相机离焦 / 零级占比失效。** 因此也不需要因为离焦去重训网络。"
            f"真正的差距 (中位 ncc {fmt(payload['baseline_ncc'], '+.4f')} vs 目标 "
            f"{fmt(H1_NCC_BAR, '+.4f')}) 仍未被归因, 本报告不猜它来自哪个量。"
        )
    lines.append(
        "4. **无论哪条假说的结论如何, 本报告都不改任何已入库文件** "
        "(`src/**`、`runs/sim2f_bench/README.md`、上一条 lane 的脚本与报告都只读)。"
        "真要落地, 应当是另一条 lane 显式做「改标定量 → 重生成语料 → 重训 → 重测门禁」的完整闭环, "
        "而不是在这里顺手改一个常数。"
    )
    return lines


def _pitfalls(payload: dict[str, Any]) -> list[str]:
    """写 §7 的坑。"""
    return [
        "1. **语料只有 18 组不同的 (相位, 帧)**, 而 312 条记录是同一批条件被复制 ~17 遍。"
        "所有中位数按去重后的 18 组统计; 有效样本是 6 个指令条件 × 3 轮, 不是 18 个独立抽样。",
        "2. **本脚本 import 的是同一条 lane 尚未入库的脚本** "
        "(`scripts/ml/validate_sim2f_vs_hw_corpus.py`, 提供 `_digest` / `sample_twin` / "
        "`zero_mean_ncc` / `normalised_l1` / `fwhm_px` / `describe`)。那一侧一旦改名或改语义, "
        "本脚本的数会跟着变 —— 这是**共享实现**的代价, 不是本脚本自己抄了一份。",
        "3. **H2 的反演跑在 numpy 孪生上, 语料指标跑在 torch 孪生上。** 两者靠仓库既有的 "
        "parity 测试 (`tests/ao_shaping/ml/sim2f/test_twin_parity.py`, 本次实测 7/7 通过) "
        "对齐 (同名映射同几何), 但 parity 只保证到 2e-3 / 5e-3 的相对 L2, 不是逐位相同。",
        "4. **numpy 孪生的裁剪网格比 CCD 像元粗**: 反演时实测 64×64 帧被重采样到 32×32 的"
        "模型格 (一个模型格 = 2.099 个相机像元), 即实测的细散斑被降采样掉了。这是孪生本身的"
        "带宽限制 (焦面模型网格节距 4.618 µm > CCD 像元 2.2 µm), 不是本次实现引入的失真; "
        "torch 孪生走的是同一条路 (`resample` 把模型强度积分到相机像元上)。",
        "5. **面板 Zernike 的归一化口径两个孪生不同**: numpy 用 `region / 2` (半宽), torch 用 "
        "`pupil_radius_px = 48`。所以 Stage 2 反演出的面板 Noll 4 系数**不能**直接搬进 torch 孪生; "
        "本报告只把相机离焦搬过去 (它是米制标量, 跨孪生同义)。",
        "6. **numpy 孪生没有零级占比旋钮**: `TwinErrorModel` 里没有 `zero_order_eta` 对应项 "
        "(仓库 parity 测试明确写了「zero_order_eta has no twin counterpart」), 所以 "
        "`fit_twin` 反演不到它; H2 的零级部分只能作诊断扫描, 不进判据。",
        "7. **实测窗口的锚点不是真 0 级**: `farfield_frame_to_grid` 用整幅帧的 argmax 定窗, "
        "而成型帧的全局最大是成型后的亮点。上一条 lane 的整数平移搜索给出中位增益 +0.063, "
        "说明锚点有约 1 px 的偏移。",
        "8. **90° 轴交换在语料里查不出来**: 侧车写明面板与相机坐标轴交换 90°, 而这批语料的"
        "像差族以无像差 / 纯离焦为主 (近似圆对称), 转置后几乎一样。",
        "9. **`region_span_px` 改不动 0 级光斑宽度** (§2.1): 入射束腰的物理尺寸与它无关, "
        "所以 H1 的宽度比判据 (a) 在原理上就极难满足 —— 这是本次扫描暴露的**结构性**事实, "
        "不是扫描网格不够细。要改 0 级宽度, 得动 `beam_w0_px` (入射束腰), 而那是另一个旋钮、"
        "不在本次预登记范围内, 本报告不去试它。",
        "10. **绝对量仍不可用**: 本报告只用 ncc / 归一化 L1 / 宽度比; 绝对 Strehl / PIB / EE / "
        "FWHM / 峰值 / 总能量一律没算, 也没进任何判据。",
        "11. **本报告不注册溯源**: `scripts/_common/provenance.py` 由另一条 lane 改动中, 本次不碰; "
        "`scripts/sync_report_provenance.py --check` 会因本文未登记而报缺失, 登记后重跑即消。",
    ]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """脚本入口。

    Args:
        argv: 命令行参数。

    Returns:
        进程退出码。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit-files",
        type=int,
        default=None,
        help="只读前 N 个 pkl (冒烟测试用); 默认全量 54 个",
    )
    parser.add_argument("--skip-h2", action="store_true", help="跳过 H2 反演 (冒烟测试用)")
    args = parser.parse_args(argv)

    configure_fonts()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ART_DIR.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()

    t0 = time.perf_counter()
    pairs, corpus_meta = load_pairs(args.limit_files)
    corpus_seconds = time.perf_counter() - t0
    logger.info(f"[corpus] {len(pairs)} distinct pairs in {corpus_seconds:.1f}s")

    cross = cross_check_against_prior_lane(corpus_meta["digests"])
    corpus_meta["distinct"] = len(pairs)
    corpus_meta["cross_check"] = cross
    corpus_meta["cross_check"]["summary"] = (
        "对方 per_record.json 不存在, 无法对账"
        if not cross.get("available")
        else (
            f"与上一条 lane 的 {cross['prior_distinct']} 组摘要"
            f"{'一致' if cross['digests_match'] else '**不一致**'}"
        )
    )

    t0 = time.perf_counter()
    h1_rows = scan_region_span(pairs, REGION_SPAN_GRID)
    h1_seconds = time.perf_counter() - t0
    logger.info(f"[H1] scanned {len(h1_rows)} spans in {h1_seconds:.1f}s")
    h1 = judge_h1(h1_rows)
    logger.info(f"[H1] verdict = {h1['verdict']}: {h1['reason']}")

    # H2 基线配置: H1 接受则用 H1 接受配置, 否则用历史标定值 900 (预登记第 15 条)。
    if h1["verdict"] == "接受":
        baseline_span = float(h1["rows"][h1["interior_indices"][0]]["region_span_px"])
        h1["accepted_region_span_px"] = int(baseline_span)
        h1["accepted_delta_px"] = int(baseline_span) - 900
    else:
        baseline_span = 900.0
        h1["accepted_region_span_px"] = None
        h1["accepted_delta_px"] = None

    baseline_metrics = corpus_metrics(pairs, baseline_span, Sim2fConfig.calibrated().errors)
    logger.info(
        f"[H2] baseline region_span_px={baseline_span:.0f} "
        f"median ncc={baseline_metrics['ncc']['median']:+.4f} "
        f"median L1={baseline_metrics['l1']['median']:.4f}",

    )

    eta_rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    if args.skip_h2:
        h2_seconds = 0.0
        h2 = {
            "verdict": "跳过",
            "reason": "冒烟测试: --skip-h2",
            "fit": {},
            "identifiability": "n/a",
            "relative_change": float("nan"),
            "initial_defocus_m": HISTORICAL_DEFOUS_M,
        }
    else:
        logger.info("[H2] fit_twin stage1+stage2 ...")
        fit = invert_defocus(pairs, baseline_span)
        h2_seconds = time.perf_counter() - t0
        logger.info(f"[H2] fit_twin done in {h2_seconds:.1f}s")
        inverted_errors = replace(
            Sim2fConfig.calibrated().errors,
            defocus_m=float(fit["stage1_plus_stage2"]["defocus_m"]),
        )
        inverted_metrics = corpus_metrics(pairs, baseline_span, inverted_errors)
        eta_rows = scan_zero_order_eta(pairs, baseline_span)

        panel_defocus = abs(float(fit["stage1_plus_stage2"]["slm_zernike"].get(4, 0.0)))
        camera_defocus = abs(float(fit["stage1_plus_stage2"]["defocus_m"]))
        stage1_defocus = abs(float(fit["stage1_only"]["defocus_m"]))
        if panel_defocus < 1e-3:
            identifiability = (
                "面板 Noll 4 停在 0 (没有吸收任何离焦), 所以相机离焦与面板离焦在这一组数据上"
                "没有互相抢; 相机离焦由 Stage 1+2 单独决定"
            )
        else:
            identifiability = (
                f"面板 Noll 4 拿到 {fmt(panel_defocus, '.3g')} rad, 相机离焦 "
                f"{fmt(camera_defocus, '.3g')} m (Stage 1 单独给的是 "
                f"{fmt(stage1_defocus, '.3g')} m) —— 面板 Noll 4 与相机离焦在远场近似简并, "
                "**份额不可辨识**, 报出的 defocus_m 只能当作这一组简并方向上的一个代表值"
            )

        defocus_ok = fit["relative_change"] > H2_DEFOUS_REL_DELTA
        ncc_gain = float(
            inverted_metrics["ncc"]["median"] - baseline_metrics["ncc"]["median"]
        )
        l1_ok = inverted_metrics["l1"]["median"] <= H2_L1_BAR
        if defocus_ok and ncc_gain >= H2_NCC_GAIN and l1_ok:
            verdict = "接受"
            reason = (
                f"反演 defocus_m = "
                f"{fmt(fit['stage1_plus_stage2']['defocus_m'] * 1e3, '.4f')} mm, 与 10.0847 mm "
                f"差 {fmt(fit['relative_change'] * 100, '.1f')} % (> 20 %); 中位 ncc "
                f"{fmt(baseline_metrics['ncc']['median'], '+.4f')} -> "
                f"{fmt(inverted_metrics['ncc']['median'], '+.4f')} (提升 {fmt(ncc_gain, '+.4f')}, "
                f">= 0.05); 中位 L1 {fmt(inverted_metrics['l1']['median'], '.4f')} (<= 0.5)"
            )
        else:
            verdict = "否决"
            failed = []
            if not defocus_ok:
                failed.append(
                    f"defocus_m 相对差 {fmt(fit['relative_change'] * 100, '.1f')} % 未超 20 %"
                )
            if ncc_gain < H2_NCC_GAIN:
                failed.append(f"中位 ncc 提升 {fmt(ncc_gain, '+.4f')} 未达 0.05")
            if not l1_ok:
                failed.append(
                    f"中位 L1 {fmt(inverted_metrics['l1']['median'], '.4f')} 未达 0.5"
                )
            reason = "; ".join(failed)
        h2 = {
            "verdict": verdict,
            "reason": reason,
            "fit": fit,
            "identifiability": identifiability,
            "baseline_region_span_px": float(baseline_span),
            "baseline_metrics": baseline_metrics,
            "inverted_metrics": inverted_metrics,
            "ncc_gain": ncc_gain,
            "defocus_ok": bool(defocus_ok),
            "ncc_gain_ok": bool(ncc_gain >= H2_NCC_GAIN),
            "l1_ok": bool(l1_ok),
            "initial_defocus_m": HISTORICAL_DEFOUS_M,
            "relative_change": float(fit["relative_change"]),
        }

    figures = render_figures(h1_rows, h1, eta_rows)

    total_seconds = time.perf_counter() - started
    accepted = [
        name for name, block in (("H1", h1), ("H2", h2)) if block["verdict"] == "接受"
    ]
    if "H1" in accepted and "H2" in accepted:
        one_liner = (
            "**标定已失效**: 远场尺度映射与相机离焦**都**要改, 网络必须重训、面板嵌入半径必须改。"
        )
    elif accepted:
        one_liner = (
            f"**标定已失效**: {accepted[0]} 成立 ⇒ "
            + (
                "网络必须重训, 面板嵌入半径要改。"
                if accepted[0] == "H1"
                else "网络需要按新离焦重训, 面板嵌入半径不动。"
            )
        )
    else:
        one_liner = (
            "**两条假说都没被判为接受**: 现有证据不足以说「标定已失效」; "
            "中位 ncc 仍停在 "
            f"{fmt(baseline_metrics['ncc']['median'], '+.4f')} (门限 {fmt(H1_NCC_BAR, '+.4f')}), "
            "归一化 L1 仍停在 "
            f"{fmt(baseline_metrics['l1']['median'], '.4f')} (门限 {fmt(H2_L1_BAR, '.2f')})。"
            "**不据此改任何已入库标定量。**"
        )

    payload: dict[str, Any] = {
        "generated_by": "scripts/ml/diagnose_sim2f_optical_scale.py",
        "preregistration": PREREGISTRATION,
        "corpus": corpus_meta,
        "h1": {"verdict": h1["verdict"], "reason": h1["reason"], "rows": h1_rows, **h1},
        "h2": h2,
        "eta_rows": eta_rows,
        "baseline_ncc": baseline_metrics["ncc"]["median"],
        "one_liner": one_liner,
        "timings": {
            "corpus_s": corpus_seconds,
            "h1_s": h1_seconds,
            "h2_s": h2_seconds,
            "total_s": total_seconds,
        },
    }
    payload["implications"] = _implications(h1, h2, payload)
    payload["pitfalls"] = _pitfalls(payload)

    (ART_DIR / "metrics.json").write_text(
        json.dumps(jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (ART_DIR / "h1_scan.json").write_text(
        json.dumps(jsonable(h1_rows), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (ART_DIR / "h2_fit.json").write_text(
        json.dumps(jsonable(h2), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT_DIR / "report.md").write_text(render_report(payload, figures), encoding="utf-8")
    logger.info(f"wrote {OUT_DIR / 'report.md'}")
    logger.info(f"wall clock {total_seconds:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
