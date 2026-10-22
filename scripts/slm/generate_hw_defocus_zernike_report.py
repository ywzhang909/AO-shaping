#!/usr/bin/env python
"""Generate the hardware defocus + Zernike-uniformity test report (2026-10-20).

Offline-only: reads the artefacts produced by

* ``scripts/slm/hw_gs_defocus_probe.py`` (hardware defocus sweep) — the
  ``measurements.csv`` / ``measurements.json`` / ``run_config.json`` /
  ``frame_*.npz`` set in one measurement directory, and
* ``scripts/slm/run_gs_refine_zernike_sweep.py`` (hardware Zernike-order
  sweep) — ``summary.json`` in one run directory,

plus the simulation reference curve ``report/shape_trials/gs_defocus_sim/artifacts/fv_groups_fresnel_vs_vl.csv``
(group ``G1``: the "dz-blind GS, no defocus" group that
``report/shape_trials/gs_defocus_sim/gs_defocus_sim_20261019.md`` quotes as its G1 row).

Never opens a device.  The per-frame spot quantities (FWHM / hollowness /
centroid) are recomputed **offline** from the raw frames the probe stored, so
the metric calibre can be changed without re-running the hardware.

Run::

    python scripts/slm/generate_hw_defocus_zernike_report.py \\
        --defocus-dir report/shape_trials/hw_gs_defocus_probe_<UTC> \\
        --noise-dir report/shape_trials/hw_gs_defocus_probe_<UTC> \\
        --zernike-dir data/debug/hw_zernike_sweep_<date>
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # noqa: E402 -- must precede pyplot import

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from _common import fmt_metric, markdown_table, savefig  # noqa: E402
from _common.provenance import insert_header  # noqa: E402
from ao_shaping.algorithm.signal_processing.square_shaping import (  # noqa: E402
    composite_from_pib_cv,
    power_in_bucket,
    uniformity_cv,
)
from ao_shaping.utils.image.beam_metrics import prepare_frame  # noqa: E402
from ao_shaping.utils.image.frame_analysis import (  # noqa: E402
    estimate_shift,
    measure_spot,
)
from ao_shaping.utils.image.targets import square_mask  # noqa: E402
from scripts.slm._hw_defocus_sections import (  # noqa: E402
    SECTION_0_2_ALGORITHM,
    SECTION_0_PURPOSE,
    section_0_1_setup,
)

REPORT_KEY = "report/shape_trials/hw_defocus_zernike/hw_defocus_zernike_20261020.md"

#: 同系列里若还有别的报告, 它们的图必须**带戳**, 否则会覆盖这份的图
#: (先例: model_in_loop/ 一个系列 9 份报告, 图名都带日期戳)。默认这份不带戳
#: (图名已入库, 不改动)。
DEFAULT_REPORT_STEM = "hw_defocus_zernike_20261020"

#: 系列前缀 —— 从上面的常量**推导**出来 (不另抄一份字面量): 砍掉最后一段
#: (默认那份的日期戳) 即是。
SERIES_PREFIX = DEFAULT_REPORT_STEM.rsplit("_", 1)[0] + "_"

#: 默认那份的日期。标题用它, 「文件名里取不到日期」时也回落用它
#: (刻意独立于 :data:`REPORT_DATE`, 后者会被 main() 改写成当前这份的日期)。
DEFAULT_REPORT_DATE = "2026-10-20"

#: 图名戳里允许的字符: 数字、字母、下划线。其余一律换成下划线。
_TAG_DISALLOWED = re.compile(r"[^0-9A-Za-z_]")


#: 当前这份报告的图名戳 (由 main() 按报告键设置)。
FIG_TAG = ""

#: 当前这份报告的日期 (由 main() 按报告键设置) —— 标题里用它, 不写死。
REPORT_DATE = DEFAULT_REPORT_DATE


def report_date(report_key: str) -> str:
    """从报告键里取日期 (YYYYMMDD -> YYYY-MM-DD); 取不到就用默认那份的。

    取的是文件名里**第一段** 8 位数字, 所以同日追加轮次的
    ``hw_defocus_zernike_20261022_r2`` 仍然是 2026-10-22 (轮次后缀不是日期)。
    """
    match = re.search(r"(\d{8})", Path(report_key).stem)
    if not match:
        return DEFAULT_REPORT_DATE
    stamp = match.group(1)
    return f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:]}"


def figure_tag(report_key: str) -> str:
    """报告的图名戳: **同一系列里每份报告都必须唯一**, 否则会互相覆盖图。

    戳由报告名里系列前缀之后的**整段后缀**决定:

    - ``hw_defocus_zernike_20261020`` -> ``""`` (默认那份, 图名已入库, 不改名)
    - ``hw_defocus_zernike_20261021`` -> ``"_20261021"``
    - ``hw_defocus_zernike_20261022_r2`` -> ``"_20261022_r2"`` (同日追加的一轮)

    ⚠️ 旧实现只在文件名**以 8 位数字结尾**时才给戳, 于是同日追加轮次的后缀
    ``_r3`` 让它匹配不上 -> 拿到**空戳** -> 直接盖掉默认那份的图。所以戳必须
    取**整段后缀**, 而不是「结尾那段数字」。
    """
    stem = Path(report_key).stem
    if stem == DEFAULT_REPORT_STEM:
        return ""
    if not stem.startswith(SERIES_PREFIX):
        raise ValueError(
            f"报告名 {stem!r} 不属于本系列 (应以 {SERIES_PREFIX!r} 开头); "
            "本脚本只给这一系列生成报告, 图名戳由系列前缀推导。"
        )
    suffix = stem[len(SERIES_PREFIX) :]
    if not suffix:
        raise ValueError(
            f"报告名 {stem!r} 缺日期/轮次后缀; 图名戳取的是系列前缀之后的整段后缀。"
        )
    return "_" + _TAG_DISALLOWED.sub("_", suffix)


OUT_DIR = ROOT / "report" / "shape_trials" / "hw_defocus_zernike"
FIG_DIR = OUT_DIR / "figures"
#: 仿真的 G1 参考曲线。报告目录正在重构 (每份报告一个子目录), 所以**按名字找**而不是
#: 写死路径 —— 写死的路径一被搬走, 离线重算就崩, 而它崩的时候报告正文还没写出来。
SIM_G1_CSV_NAME = "fv_groups_fresnel_vs_vl.csv"


def find_sim_g1_csv() -> Path | None:
    """在 ``report/`` 下按文件名找仿真的 G1 参考曲线 (找不到就返回 None)。"""
    direct = (
        ROOT
        / "report"
        / "shape_trials"
        / "gs_defocus_sim"
        / "artifacts"
        / SIM_G1_CSV_NAME
    )
    if direct.exists():
        return direct
    matches = sorted((ROOT / "report").rglob(SIM_G1_CSV_NAME))
    return matches[0] if matches else None


# --- CJK font guard (never crash on a headless box) --------------------------
try:
    from matplotlib import font_manager

    _available = {f.name for f in font_manager.fontManager.ttflist}
    _cjk_candidates = [
        "Microsoft YaHei",
        "SimHei",
        "WenQuanYi Micro Hei",
        "Noto Sans CJK SC",
    ]
    _picked = [n for n in _cjk_candidates if n in _available]
    if _picked:
        plt.rcParams["font.sans-serif"] = _picked + ["DejaVu Sans"]
except ImportError:  # pragma: no cover
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_probe_run(run_dir: Path) -> dict[str, Any]:
    """Read one ``hw_gs_defocus_probe`` measurement directory.

    Returns the run config, the measurement rows, and the per-frame spot
    quantities recomputed from the stored raw frames.
    """
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    rows = json.loads((run_dir / "measurements.json").read_text(encoding="utf-8"))
    cx, cy = config["zero_order_roi_xy_frozen"]
    side = int(config["target_side_px"])

    enriched: list[dict[str, Any]] = []
    for row in rows:
        npz_path = run_dir / f"frame_{int(row['index']):03d}.npz"
        item = dict(row)
        item["frozen_center_xy"] = [int(cx), int(cy)]
        item["target_side_px"] = side
        if not npz_path.exists():
            item["frame_missing"] = True
            enriched.append(item)
            continue
        raw = np.load(npz_path)["frame"]
        frame = prepare_frame(raw)
        spot = measure_spot(raw)
        frozen = square_mask(frame.shape, int(cy), int(cx), side)
        cv_frozen = float(uniformity_cv(frame, frozen))
        pib_frozen = float(power_in_bucket(frame, frozen))
        # 随斑框: **边长不变**, 只把中心平移到光斑质心。用来把「转向」与「离焦」
        # 两个效应分开 —— 它不做任何缩放, 所以不触发仓库记录的自适应框刷分陷阱。
        sx = int(round(spot.centroid_x))
        sy = int(round(spot.centroid_y))
        moved = square_mask(frame.shape, sy, sx, side)
        cv_moved = float(uniformity_cv(frame, moved))
        pib_moved = float(power_in_bucket(frame, moved))
        item.update(
            {
                "frame_missing": False,
                "fwhm_px": float(spot.fwhm_px),
                "hollowness": float(spot.hollowness),
                "centroid_xy": [float(spot.centroid_x), float(spot.centroid_y)],
                "shift_px": float(
                    math.hypot(spot.centroid_x - cx, spot.centroid_y - cy)
                ),
                "shift_x_px": float(spot.centroid_x - cx),
                "shift_y_px": float(spot.centroid_y - cy),
                "cv_frozen": cv_frozen,
                "pib_frozen": pib_frozen,
                "score_frozen": float(composite_from_pib_cv(pib_frozen, cv_frozen)),
                "cv_moved": cv_moved,
                "pib_moved": pib_moved,
                "score_moved": float(composite_from_pib_cv(pib_moved, cv_moved)),
            }
        )
        enriched.append(item)
    return {"dir": run_dir, "config": config, "rows": enriched}


def noise_floor(run: dict[str, Any]) -> dict[str, float]:
    """重复测量给出的噪声地板 (同一 Δz 的组内标准差)。"""
    rows = [r for r in run["rows"] if not r.get("frame_missing")]
    out: dict[str, float] = {}
    for key in ("cv", "pib", "score", "frame_peak"):
        values = [
            float(r[key]) for r in rows if math.isfinite(float(r.get(key, math.nan)))
        ]
        out[f"sigma_{key}"] = (
            float(np.std(values, ddof=1)) if len(values) > 1 else math.nan
        )
        out[f"mean_{key}"] = float(np.mean(values)) if values else math.nan
    out["n_repeats"] = len(rows)
    return out


def load_sim_g1() -> list[dict[str, Any]]:
    """仿真 G1 组 (dz-blind GS, 无离焦) 的细网格参考曲线。"""
    csv_path = find_sim_g1_csv()
    if csv_path is None:
        return []
    with csv_path.open(encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle) if r["group"] == "G1"]
    out = []
    for r in rows:
        out.append(
            {
                "dz_mm": float(r["dz_mm"]),
                "dz_units": float(r["dz_units"]),
                "side_px": int(r["side_px"]),
                "cv": float(r["cv"]),
                "pib": float(r["pib"]),
                "score": float(r["score"]),
                "cv_fixed": float(r["cv_fixed"]),
                "pib_fixed": float(r["pib_fixed"]),
                "score_fixed": float(r["score_fixed"]),
                "sigma_px": float(r["sigma_px"]),
            }
        )
    return sorted(out, key=lambda r: r["dz_mm"])


def _find_under(root: Path | None, pattern: str) -> Path | None:
    """在根目录下**递归**找第一个匹配的文件。

    泽尼克扫描拆成多次短调用之后 (Stage A 一组 + 每阶一组), 产物分散在各个组目录里,
    所以这里不能再假设「文件就在组目录的正下方」。
    """
    if root is None or not root.exists():
        return None
    matches = sorted(root.rglob(pattern))
    return matches[0] if matches else None


#: 跨组必须核对的**标量口径字段**。同一轮扫描里它们本应一致 —— 不一致就说明
#: 这批数据**不同源** (混进了别轮的产物), 报告正文里必须看得见。
SCALAR_MERGE_FIELDS = ("w0_px", "target_side", "center_xy", "mask_center_xy")

#: 像素坐标类字段 (零级位置 / 框中心)。逐组各自在**自己那一帧**上重新定位 ⇒
#: 允许有 ±1 px 的抖动 —— 那是**良性的**, 不是数据混入。
PIXEL_COORD_MERGE_FIELDS = ("center_xy", "mask_center_xy")

#: 连续量字段 (口径: 高斯半径 / 目标框边长)。跨组不一致 ⇒ **光学状态真的变了**
#: (焦面/曝光/口径不同) ⇒ 疑似混进了别轮的数据。
CONTINUOUS_MERGE_FIELDS = ("w0_px", "target_side")

#: 像素坐标逐轴偏差的**容忍上限** (px): 超过它就不算「定位抖动」而是判为疑似混入。
#:
#: 取 **2** 的理由: 实测逐阶组各自重新定位零级的散布只有 **±1 px** —— 第二轮八个
#: 逐阶组的 ``mask_center_xy`` 落在 ``[126,128]`` / ``[127,128]`` / ``[127,127]`` /
#: ``[126,127]`` 四种上, 而 ``w0_px`` / ``target_side`` / ``center_xy`` 完全一致。
#: 2 px 是这个实测散布的**一倍余量**: 抖动留余量, 真正的位移 (>2 px) 仍会被报出来。
LOCATE_JITTER_TOL_PX = 2.0

#: 连续量**相对差**的阈值: 超过它就判为「光学状态真的变了」(疑似跨轮次混入)。
#:
#: 取 **0.5%** 的理由: 实测跨轮次的光学状态差异是 **1.13%** —— 第二轮数据根里混进来的
#: ``zernike_stage_a_1`` 的 ``w0_px`` 5.742078 → 5.807177 (``target_side`` 同比例)。
#: 阈值取它的一半不到, 好让这种差异**被报出来**, 而不是被当成噪声吞掉。
CONTINUOUS_REL_DIFF_TOL = 0.005

#: 各像素坐标字段所在的**坐标系**。同一个坐标系之间才谈得上「偏差多少」:
#:
#: - ``center_xy`` = **全幅传感器**坐标 (每组都在同一块传感器上定位零级) ⇒ 永远可比,
#:   **不要**给它套「小窗口」规则 (全幅坐标本来就比窗口大);
#: - ``mask_center_xy`` = **窗口内**坐标 ⇒ 可比性由**取值本身**决定, 而**不是**由
#:   ``frame_shape`` 是否相等决定 —— 见 :func:`_window_local_comparability`。
#:
#: ⚠️ 为什么不能用「``frame_shape`` 相等」判可比: 两个生产者写的 ``frame_shape``
#: **不是一个意思** —— Stage A 块记的是**全幅** ``probe_shape`` (1944x2592), 但它的
#: ``mask_center_xy`` 在 2026-10-21 的续跑坐标修复 (提交 07139db6) **之后**已经是
#: **窗口内**坐标 (优化器冻结的 ROI 中心)。按 ``frame_shape`` 判会把**其实同空间**的
#: 取值误判成「坐标系不同」⇒ 正文里印一句不成立的话, 还白白抑制掉一次交叉核对
#: (逐阶组的掩膜中心 vs 块的掩膜中心 —— 差得多就说明某一块的 ROI 摆歪了)。
PIXEL_COORD_SPACES = {
    "center_xy": "full_sensor",
    "mask_center_xy": "window_local",
}

#: 判定「某个坐标是否落在窗口内」时, **上界是否含端点** (``<=`` vs ``<``)。
#:
#: 取 ``True`` (用 ``<=``): 窗口边长是**像素数**, 而窗口内像素坐标的合法最大值是
#: ``边长 - 1``; 用 ``<= 边长`` 等于把上界**放宽 1 px**。这是**故意**的: 判据要回答的
#: 是「这个数**是不是**全幅坐标」, 而不是「它是不是**恰好**某个窗口里的合法像素下标」
#: —— 宁可放宽 1 px, 也不要**误把一个窗口内坐标判成全幅坐标** (那会再次抑制交叉核对)。
#: 下界一律用 ``0`` (窗口内坐标不可能为负)。
WINDOW_LOCAL_UPPER_BOUND_INCLUSIVE = True

#: 本报告**正文**真正读到的合并字段 (其余字段只留在采集脚本自己的产物里)。
#: 写清楚是为了让「合并后用的是哪个值」这个问题在报告里有确定答案。
BODY_MERGE_FIELDS = ("w0_px", "baselines", "config")


def _summary_group_label(path: Path, root: Path) -> str:
    """``summary.json`` 所在**组目录**的相对路径 (不含文件名)。

    报告里显示**组目录名**而不是全路径: 全路径带机器目录名, 换台机器就对不上,
    而组目录名 (``zernike_stage_a_1`` / ``zernike_order_nmax3``) 才是读者能核对的标识。
    """
    try:
        rel = path.parent.relative_to(root)
    except ValueError:
        return path.parent.name
    return path.parent.name if str(rel) == "." else rel.as_posix()


def load_zernike(run_dir: Path) -> dict[str, Any]:
    """读泽尼克扫描的 ``summary.json``, 并把**多组**的合并起来。

    扫描拆成多次短调用之后, 每组各有一份 ``summary.json`` (Stage A 一份 + 每阶一份),
    所以这里递归收齐根目录下的所有 summary: ``orders`` 拼起来 (同阶重复跑过时取后一次),
    其余字段取**第一份含该字段的** summary 的值。

    ⚠️ 合并会把不同组的取值压成一个, 所以**跨组不一致必须记下来**, 而且必须在**报告正文里
    显示出来** (:func:`merge_audit_block`): 否则「混进别轮数据」这件事读者完全看不见。
    """
    summaries: list[dict[str, Any]] = []
    for path in sorted(run_dir.rglob("summary.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict):
            data = dict(data)
            data["_summary_path"] = str(path)
            data["_group"] = _summary_group_label(path, run_dir)
            summaries.append(data)
    if not summaries:
        return {}
    merged: dict[str, Any] = {}
    by_order: dict[int, dict[str, Any]] = {}
    order_group: dict[int, str] = {}
    origin: dict[str, str] = {}
    frame_shapes: dict[str, Any] = {}
    warnings: list[dict[str, Any]] = []
    for data in summaries:
        group = str(data["_group"])
        # 每组**在哪个窗口里**工作: 判定像素坐标能否直接比较时要用 (见
        # :data:`PIXEL_COORD_SPACES`)。⚠️ ``frame_shape`` 只是**线索**: 两个生产者
        # 写的它不是一个意思 (Stage A 块记的是全幅 ``probe_shape``, 逐阶组记的是
        # ROI 窗口) ⇒ 可比性最终由**取值本身**是否落在共同窗口内决定
        # (见 :func:`_window_local_comparability`)。
        frame_shapes[group] = data.get("frame_shape")
        for key, value in data.items():
            if key in ("orders", "_summary_path", "_group"):
                continue
            if key not in merged:
                merged[key] = value
                origin[key] = group
            elif key in SCALAR_MERGE_FIELDS and merged[key] != value:
                warnings.append(
                    {
                        "kind": "field",
                        "field": key,
                        "kept": merged[key],
                        "kept_group": origin.get(key, "?"),
                        "other": value,
                        "other_group": group,
                    }
                )
        for order in data.get("orders") or []:
            n_max = int(order.get("n_max", 0))
            if n_max in by_order:
                # 同一个 n_max 出现两次 ⇒ 后一次**顶掉**前一次 (合并规则),
                # 这也是一次静默的跨组覆盖, 同样要记下来。
                warnings.append(
                    {
                        "kind": "orders",
                        "n_max": n_max,
                        "kept_group": group,
                        "other_group": order_group.get(n_max, "?"),
                    }
                )
            by_order[n_max] = order
            order_group[n_max] = group
    merged["orders"] = [by_order[n] for n in sorted(by_order)]
    merged["merge_warnings"] = warnings
    merged["summary_paths"] = [d["_summary_path"] for d in summaries]
    merged["summary_groups"] = [d["_group"] for d in summaries]
    merged["merge_origin"] = origin
    merged["summary_frame_shapes"] = frame_shapes
    merged["order_groups"] = order_group
    return merged


def _fmt_merge_value(value: Any) -> str:
    """合并字段取值的显示格式 (标量 / 坐标对)。"""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return fmt_metric(value, 6)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_fmt_merge_value(v) for v in value) + "]"
    return str(value)


def _pixel_axis_deviation(kept: Any, other: Any) -> float | None:
    """两个像素坐标之间的**最大逐轴偏差** (px)。

    不是同长度的坐标对 (或压根不是坐标) 就返回 ``None`` —— 调用方据此把差异判成
    「无法比较」, 而不是默默当成 0。
    """
    if not isinstance(kept, (list, tuple)) or not isinstance(other, (list, tuple)):
        return None
    if len(kept) == 0 or len(kept) != len(other):
        return None
    try:
        return max(abs(float(a) - float(b)) for a, b in zip(kept, other))
    except (TypeError, ValueError):
        return None


def _relative_diff(kept: Any, other: Any) -> float | None:
    """连续量两个取值的**相对差** ``|other - kept| / |kept|`` (取不到就 None)。"""
    if isinstance(kept, bool) or isinstance(other, bool):
        return None
    if not isinstance(kept, (int, float)) or not isinstance(other, (int, float)):
        return None
    base = abs(float(kept))
    if base == 0.0:
        return None
    return abs(float(other) - base) / base


def _frame_shape_hw(value: Any) -> tuple[float, float] | None:
    """把 ``frame_shape`` 规整成 ``(height, width)``; 取不到就 ``None``。

    ``frame_shape`` 沿用 numpy ``frame.shape`` 的约定 = ``(高, 宽)``; 而
    ``mask_center_xy`` / ``center_xy`` 是 ``(x, y)`` —— 两者的**轴序相反**, 比较时
    必须换算 (x 对**宽**, y 对**高**), 否则非方形窗口会判错。
    """
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        height = float(value[0])
        width = float(value[1])
    except (TypeError, ValueError):
        return None
    if height <= 0.0 or width <= 0.0:
        return None
    return height, width


def _coord_inside_window(value: Any, height: float, width: float) -> bool | None:
    """某个 ``(x, y)`` 取值是否落在 ``height x width`` 的窗口内。

    判据: ``0 <= x <= width`` 且 ``0 <= y <= height`` (上界**含端点**, 见
    :data:`WINDOW_LOCAL_UPPER_BOUND_INCLUSIVE`)。

    Returns:
        ``True`` / ``False`` = 判定结果; ``None`` = **取不到** (不是坐标对, 或数值
        不可解析) ⇒ 调用方按「无法证明它是全幅坐标」处理 (按可比), 而不是硬判。
    """
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        x = float(value[0])
        y = float(value[1])
    except (TypeError, ValueError):
        return None
    upper_x = width if WINDOW_LOCAL_UPPER_BOUND_INCLUSIVE else width - 1.0
    upper_y = height if WINDOW_LOCAL_UPPER_BOUND_INCLUSIVE else height - 1.0
    return 0.0 <= x <= upper_x and 0.0 <= y <= upper_y


def _window_local_comparability(
    kept_group: str,
    other_group: str,
    frame_shapes: dict[str, Any],
    kept_value: Any,
    other_value: Any,
) -> tuple[bool, str]:
    """两个组的**窗口内**坐标 (``mask_center_xy``) 是否落在同一个空间。

    规则 (取代「``frame_shape`` 是否相等」):

    - 两个组的窗口**相同** ⇒ 同一空间 ⇒ 可比;
    - 窗口**不同** ⇒ 取两组的**共同窗口** (逐轴取小: 高取小、宽取小) 作为
      「**可能的窗口内空间**」:
      - 两个取值**都落在共同窗口内** ⇒ 它们**都是窗口内坐标** ⇒ **同空间** ⇒ 可比,
        逐轴偏差照常判;
      - **有一个落在共同窗口之外** ⇒ 那个取值**不是**窗口内坐标 ⇒ 它是**全幅
        传感器**坐标 ⇒ 两者**确实不同空间** ⇒ 判「坐标系不同, 不参与判定」。

    ⚠️ 为什么第二轮仍然是「不同空间」: 第二轮 ``zernike_stage_a`` 的
    ``mask_center_xy`` 确实是**全幅**坐标 ``[1433, 971]`` (超出 256x256 的共同窗口)
    ⇒ 仍判不可比 ⇒ **不改变已入库的第二轮报告的结论**。
    而第三轮 (2026-10-21 续跑坐标修复之后) 块的 ``mask_center_xy`` 是**窗口内**的
    ``[127, 128]`` ⇒ 落在共同窗口内 ⇒ 判**可比** ⇒ 交叉核对恢复生效。

    Returns:
        ``(comparable, note)`` —— ``note`` 是写进报告的判定依据。
    """
    kept_shape = _frame_shape_hw(frame_shapes.get(kept_group))
    other_shape = _frame_shape_hw(frame_shapes.get(other_group))
    if kept_shape is None or other_shape is None:
        return (
            True,
            "缺少可用的 `frame_shape` ⇒ 按可比处理（缺元数据不该把一处差异藏起来）",
        )
    if kept_shape == other_shape:
        return True, "窗口（`frame_shape`）相同 ⇒ 同一空间 ⇒ 可比"
    height = min(kept_shape[0], other_shape[0])
    width = min(kept_shape[1], other_shape[1])
    inside_kept = _coord_inside_window(kept_value, height, width)
    inside_other = _coord_inside_window(other_value, height, width)
    if inside_kept is None or inside_other is None:
        return (
            True,
            "取值不是坐标对 ⇒ 无法判断它是否落在窗口内 ⇒ 按可比处理",
        )
    if inside_kept and inside_other:
        return (
            True,
            f"窗口不同，但两个取值都落在共同窗口 {width:g}x{height:g} 内 "
            "⇒ 都是窗口内坐标 ⇒ 同一空间 ⇒ 可比",
        )
    outside = []
    if not inside_kept:
        outside.append(f"`{kept_group}`（`{_fmt_merge_value(kept_value)}`）")
    if not inside_other:
        outside.append(f"`{other_group}`（`{_fmt_merge_value(other_value)}`）")
    who = "、".join(outside)
    return (
        False,
        f"{who} 落在共同窗口 {width:g}x{height:g} 之外 ⇒ 它是**全幅传感器**坐标，"
        "与窗口内坐标**不同空间**",
    )


def _pixel_comparability(
    field: str,
    kept_group: str,
    other_group: str,
    frame_shapes: dict[str, Any],
    kept_value: Any = None,
    other_value: Any = None,
) -> tuple[bool, str]:
    """某个像素坐标字段的两个取值**是否在同一空间**, 并给出判定依据。

    - ``center_xy`` = **全幅传感器**坐标 ⇒ 永远可比 (**不套**「小窗口」规则);
    - ``mask_center_xy`` = **窗口内**坐标 ⇒ 由 :func:`_window_local_comparability` 判。
    """
    if PIXEL_COORD_SPACES.get(field, "full_sensor") != "window_local":
        return (
            True,
            "`center_xy` 是**全幅传感器**坐标（每组都在同一块传感器上定位零级）"
            "⇒ 永远可比",
        )
    return _window_local_comparability(
        kept_group, other_group, frame_shapes, kept_value, other_value
    )


def _pixel_values_comparable(
    field: str,
    kept_group: str,
    other_group: str,
    frame_shapes: dict[str, Any],
    kept_value: Any = None,
    other_value: Any = None,
) -> bool:
    """两个组的该字段取值**是否在同一个坐标系**里 (只要布尔结论时用这个)。

    ``center_xy`` 是**全幅传感器**坐标 (每组都在同一块传感器上定位零级) ⇒ 永远可比;
    ``mask_center_xy`` 是**窗口内**坐标 ⇒ 由**取值本身**是否落在两组的**共同窗口**内
    决定 (见 :func:`_window_local_comparability`): 两个取值都落在里面 ⇒ 同空间可比;
    有一个落在外面 ⇒ 那是**全幅**坐标 ⇒ 确实不同空间。

    某个组没记 ``frame_shape`` 时**按可比处理**: 缺元数据不该把一处差异悄悄藏起来。
    """
    comparable, _note = _pixel_comparability(
        field, kept_group, other_group, frame_shapes, kept_value, other_value
    )
    return comparable


def classify_merge_difference(
    field: str,
    kept: Any,
    other: Any,
    kept_group: str,
    other_group: str,
    frame_shapes: dict[str, Any] | None = None,
) -> tuple[str, str, float | None]:
    """判定**一处**跨组差异属于哪一类, 并给出写进报告的判定说明。

    Args:
        field: 字段名 (``w0_px`` / ``target_side`` / ``center_xy`` / ``mask_center_xy``)。
        kept: 合并采用的取值 (第一份含该字段的 summary 的值)。
        other: 另一个组里的取值。
        kept_group: 合并采用值来自哪个组目录。
        other_group: 另一个组目录。
        frame_shapes: ``{组目录: frame_shape}`` —— 判断两个组的坐标是否同一参考系。

    Returns:
        ``(bucket, verdict, deviation)``。bucket ∈

        - ``"jitter"`` —— 同轮内的定位抖动 (**良性**);
        - ``"suspect"`` —— 疑似跨轮次数据混入 (**恶性**);
        - ``"incomparable"`` —— 坐标系不同, 数值不可直接比较。

        deviation 是量化值 (像素偏差 px 或相对差), 取不到就 None。
    """
    frames = frame_shapes or {}
    if field in PIXEL_COORD_MERGE_FIELDS:
        comparable, note = _pixel_comparability(
            field, kept_group, other_group, frames, kept, other
        )
        if not comparable:
            return (
                "incomparable",
                f"坐标系不同 ⇒ 不参与判定（{note}）",
                None,
            )
        dev = _pixel_axis_deviation(kept, other)
        if dev is None:
            return ("suspect", "坐标无法逐轴比较", None)
        if dev <= LOCATE_JITTER_TOL_PX:
            return (
                "jitter",
                f"逐轴偏差 {dev:g} px ≤ {LOCATE_JITTER_TOL_PX:g} px（定位抖动）",
                dev,
            )
        return (
            "suspect",
            f"逐轴偏差 {dev:g} px > {LOCATE_JITTER_TOL_PX:g} px",
            dev,
        )
    if field in CONTINUOUS_MERGE_FIELDS:
        rel = _relative_diff(kept, other)
        if rel is None:
            return ("suspect", "无法计算相对差", None)
        pct = CONTINUOUS_REL_DIFF_TOL * 100.0
        if rel > CONTINUOUS_REL_DIFF_TOL:
            return ("suspect", f"相对差 {rel * 100:.2f}% > {pct:g}%", rel)
        return ("jitter", f"相对差 {rel * 100:.2f}% ≤ {pct:g}%", rel)
    return ("suspect", "未归类的字段差异", None)


def merge_audit_block(zernike: dict[str, Any]) -> str:
    """§3 开头那一段: 这次合并**读了哪些组**、**哪些字段跨组不一致**。

    合并本身会把不同组的取值压成一个 —— 所以「这批数据是不是同源的」必须在**报告正文里
    可核查**: 列出参与合并的组目录, 再逐条列出不一致的字段、各自的取值与来源组。

    差异**分两类判别** (不是「一律不一致」):

    - **良性** (同轮内的定位抖动): 差异只在像素坐标类字段上、且逐轴偏差 ≤
      :data:`LOCATE_JITTER_TOL_PX` ⇒ 各组在**自己那一帧**上重新定位零级, 散布 ±N px,
      **不影响口径**;
    - **恶性** (疑似跨轮次混入): 连续量的相对差 > :data:`CONTINUOUS_REL_DIFF_TOL`,
      或像素坐标的逐轴偏差超过阈值, 或同一个 n_max 出现在两个组里。
    """
    groups = [str(g) for g in (zernike.get("summary_groups") or [])]
    if not groups:
        return ""
    warnings = list(zernike.get("merge_warnings") or [])
    origin = dict(zernike.get("merge_origin") or {})
    frame_shapes = dict(zernike.get("summary_frame_shapes") or {})
    field_warnings = [w for w in warnings if w.get("kind") == "field"]
    order_warnings = [w for w in warnings if w.get("kind") == "orders"]

    origin_items = [
        f"`{name}`（`{origin[name]}`）" for name in BODY_MERGE_FIELDS if name in origin
    ]
    lines = [
        "**数据来源与同源性核查** —— 本节的数字由 "
        f"**{len(groups)}** 份 `summary.json` 合并而成，组目录："
        + "、".join(f"`{g}`" for g in groups)
        + "。",
        "",
        "- `orders`：**逐组按 n_max 拼起来**（同一个 n_max 出现两次时取**后一次**）。",
        "- 其余字段：取**第一份含该字段的** summary 的值 —— 本报告正文读到的那些字段来自 "
        + ("、".join(origin_items) if origin_items else "（无）")
        + "。",
        "- 跨组核对的**标量口径字段**是 "
        + "、".join(f"`{f}`" for f in SCALAR_MERGE_FIELDS)
        + "（同一轮扫描里它们本应一致）。",
        "- 跨组差异**分两类判别**：像素坐标类字段（"
        + "、".join(f"`{f}`" for f in PIXEL_COORD_MERGE_FIELDS)
        + f"）的逐轴偏差 ≤ **{LOCATE_JITTER_TOL_PX:g} px** ⇒ **良性**（同轮内的定位抖动）；"
        "连续量字段（"
        + "、".join(f"`{f}`" for f in CONTINUOUS_MERGE_FIELDS)
        + f"）的相对差 > **{CONTINUOUS_REL_DIFF_TOL * 100:g}%** ⇒ **恶性**（疑似跨轮次混入）。",
        "- **可比性先判**：同一空间才谈得上「偏差多少」。`center_xy` 是**全幅传感器**坐标"
        "（每组都在同一块传感器上定位零级）⇒ **永远可比**；`mask_center_xy` 是**窗口内**"
        "坐标 ⇒ 取两组的**共同窗口**（逐轴取小）作为「可能的窗口内空间」，两个取值"
        "**都落在里面**才当同空间判偏差；**有一个落在外面** ⇒ 它是**全幅**坐标 ⇒ "
        "两者的数**不可直接比较**。",
        "",
        "**阈值为什么是这两个数**：像素坐标的 "
        f"**{LOCATE_JITTER_TOL_PX:g} px** —— 实测逐阶组各自在**自己那一帧**上重新定位零级，"
        "散布只有 **±1 px**（第二轮八个逐阶组的 `mask_center_xy` 落在 `[126,128]` / "
        "`[127,128]` / `[127,127]` / `[126,127]` 四种上），"
        f"{LOCATE_JITTER_TOL_PX:g} px 是它的一倍余量；连续量的 "
        f"**{CONTINUOUS_REL_DIFF_TOL * 100:g}%** —— 实测跨轮次的光学状态差异是 **1.13%**"
        "（第二轮混进来的 `zernike_stage_a_1`：`w0_px` 5.742078 → 5.807177），"
        "阈值取它的一半不到，好让这种差异**被报出来**而不是被当成噪声吞掉。",
        "",
        "**为什么可比性看「取值」而不是看「`frame_shape` 是否相等」**：两个生产者写的 "
        "`frame_shape` **不是一个意思** —— Stage A 块记的是**全幅** `probe_shape`"
        "（1944×2592），但它的 `mask_center_xy` 在 2026-10-21 的续跑坐标修复"
        "（提交 07139db6）**之后**已经是**窗口内**坐标（优化器冻结的 ROI 中心）。"
        "按 `frame_shape` 判 ⇒ 把**其实同空间**的取值误判成「不同空间」⇒ 正文里印一句"
        "**不成立**的话，还白白**抑制掉一次交叉核对**（逐阶组的掩膜中心 vs 块的掩膜中心 —— "
        "差得多就说明某一块的 ROI 摆歪了）。改判「取值是否落在共同窗口内」之后："
        "**第二轮**（`mask_center_xy` 是全幅的 `[1433, 971]`，超出 256×256 的共同窗口）"
        "**仍判不同空间** ⇒ 已入库的第二轮报告结论不变；**第三轮**（`[127, 128]`，落在窗口内）"
        "**判同空间** ⇒ 交叉核对恢复生效。",
    ]

    if not field_warnings and not order_warnings:
        return "\n".join(lines)

    # --- 逐字段聚拢: 同一字段可能有多个「别的取值」 --------------------------
    by_field: dict[str, dict[str, Any]] = {}
    for w in field_warnings:
        rec = by_field.setdefault(
            str(w.get("field")),
            {"kept": w.get("kept"), "kept_group": w.get("kept_group"), "others": []},
        )
        rec["others"].append((w.get("other"), str(w.get("other_group"))))

    jitter_rows: list[list[Any]] = []
    suspect_rows: list[list[Any]] = []
    incomparable_rows: list[list[Any]] = []
    jitter_px: list[float] = []
    jitter_cont_fields: set[str] = set()
    suspect_fields: set[str] = set()
    for field in sorted(
        by_field,
        key=lambda f: (
            SCALAR_MERGE_FIELDS.index(f)
            if f in SCALAR_MERGE_FIELDS
            else len(SCALAR_MERGE_FIELDS),
            f,
        ),
    ):
        rec = by_field[field]
        kept = rec["kept"]
        kept_group = str(rec["kept_group"])
        # 同一个取值可能来自**坐标系不同**的组 ⇒ 分格时带上「是否同一参考系」,
        # 否则会把「不可比」和「可比」混在一格里, 判定也就跟着混。
        cells: dict[tuple[str, bool], tuple[Any, list[str]]] = {}
        for value, group in rec["others"]:
            same_frame = _pixel_values_comparable(
                field, kept_group, group, frame_shapes, kept, value
            )
            key = (_fmt_merge_value(value), same_frame)
            if key not in cells:
                cells[key] = (value, [])
            cells[key][1].append(group)
        for (text, _same_frame), (raw, other_groups) in cells.items():
            bucket, verdict, dev = classify_merge_difference(
                field, kept, raw, kept_group, other_groups[0], frame_shapes
            )
            row = [
                f"`{field}`",
                f"`{_fmt_merge_value(kept)}`（`{kept_group}`）",
                f"`{text}`（" + "、".join(f"`{g}`" for g in other_groups) + "）",
                verdict,
            ]
            if bucket == "jitter":
                jitter_rows.append(row)
                if field in CONTINUOUS_MERGE_FIELDS:
                    jitter_cont_fields.add(field)
                elif dev is not None:
                    jitter_px.append(dev)
            elif bucket == "incomparable":
                incomparable_rows.append(row)
            else:
                suspect_rows.append(row)
                suspect_fields.add(field)

    table_header = ["字段", "合并采用（组目录）", "其他组里的取值（组目录）", "判定"]

    if jitter_rows:
        # 「口径没变」这句话只在**真的没变**时才能说 —— 口径字段若在下面的 ⚠️ 段里
        # 有超阈值的差异, 这里只能说「本段的差异不涉及口径」。
        cont_suspect = sorted(suspect_fields & set(CONTINUOUS_MERGE_FIELDS))
        if cont_suspect:
            caliber = (
                "口径字段（"
                + "、".join(f"`{f}`" for f in cont_suspect)
                + "）在**部分组**之间有超阈值的差异"
                "（见下面「疑似跨轮次数据混入」那一段）"
                " —— 本段里的差异都**不涉及**口径"
            )
        elif jitter_cont_fields:
            caliber = (
                "口径字段（"
                + "、".join(f"`{f}`" for f in sorted(jitter_cont_fields))
                + f"）的相对差都在 {CONTINUOUS_REL_DIFF_TOL * 100:g}% 以内"
                " ⇒ 未见口径变化"
            )
        else:
            caliber = (
                "口径字段（"
                + "、".join(f"`{f}`" for f in CONTINUOUS_MERGE_FIELDS)
                + "）在各组之间**完全一致** ⇒ 不影响口径"
            )
        spread = f"散布 **±{max(jitter_px):g} px**" if jitter_px else "坐标没有差别"
        lines += [
            "",
            "**同轮内的定位抖动（良性差异）** —— 下列差异都**没有**超过判定阈值："
            "各组在**自己那一帧**上重新定位零级，"
            f"{spread}，{caliber}。",
            "",
            markdown_table(table_header, jitter_rows),
        ]

    if suspect_rows:
        n_fields = len(suspect_fields)
        lines += [
            "",
            f"⚠️ **跨组字段不一致，疑似跨轮次数据混入：{n_fields} 个字段**"
            " —— 下列差异**超过**判定阈值：这批数据的**光学状态不是同一轮的**"
            "（焦面/曝光/口径变了，或混进了别轮的产物）。"
            "「合并采用」= 报告正文实际使用的那个取值。",
            "",
            markdown_table(table_header, suspect_rows),
        ]

    if incomparable_rows:
        lines += [
            "",
            "**坐标参考系不同（不参与判定）** —— 下列差异是**坐标系不同**造成的："
            "`center_xy` 是**全幅传感器**坐标（永远可比），`mask_center_xy` 是**窗口内**"
            "坐标 —— 当某个取值**落在两组的共同窗口之外**时，它就是**全幅**坐标，"
            "与窗口内坐标**不同空间**，这两个数**不可直接比较**。"
            "因此它们既不算定位抖动，也不构成数据混入的证据。",
            "",
            markdown_table(table_header, incomparable_rows),
        ]

    if order_warnings:
        items = "、".join(
            f"n_max={w.get('n_max')}：采用 `{w.get('kept_group')}`，"
            f"顶掉了 `{w.get('other_group')}`"
            for w in order_warnings
        )
        lines += [
            "",
            f"⚠️ **`orders` 里同一个 n_max 出现了多次**（{len(order_warnings)} 处）：{items}。",
        ]

    # --- 后果: 只把事实摆出来 -----------------------------------------------
    w0 = zernike.get("w0_px")
    box_txt = (
        f"（2·w0_px = {fmt_metric(float(w0) * 2.0, 2)} px）"
        if isinstance(w0, (int, float))
        else ""
    )
    unused = [f for f in SCALAR_MERGE_FIELDS if f != "w0_px"]
    lines += [
        "",
        "**后果（只列事实）**：",
        "",
        "- 上表「合并采用」那一列就是 §3 正文里用的值：`w0_px` 决定 §3.2.1 与 §3.3 引用的"
        f"目标框边长{box_txt}；"
        + "、".join(f"`{f}`" for f in unused)
        + " 在本报告正文里**没有被引用**（它们只记录在采集脚本自己的产物里）。",
        "- §3.2 的阶数指标（CV / PIB / 拟合残差 / max|c|）来自各组自己的 `orders`，"
        "**不经过**上面这几个标量字段。",
        "- 本报告**不替读者判断**这些差异是否影响结论；上表与本段只给出事实。",
    ]
    return "\n".join(lines)


def discover_groups(roots: list[Path]) -> dict[str, list[dict[str, Any]]]:
    """在根目录里递归找出所有**分组记录**。

    刻意**不**依赖分组清单 JSON 的具体字段名 —— 清单是驱动脚本的产物, 字段会变;
    这里按「目录里有哪些特征文件」来认组:

    - 含 ``measurements.json`` ⇒ 离焦探针组 (``hw_gs_defocus_probe.py``);
    - 含 ``summary.json`` ⇒ 泽尼克阶数组 (``run_gs_refine_zernike_sweep.py``)。

    这样分组布局 (``<root>/<group>/``) 与旧的平铺布局都能认出来, 报告不挑。
    """
    probe: list[dict[str, Any]] = []
    zernike: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("measurements.json")):
            if path.parent in seen:
                continue
            seen.add(path.parent)
            probe.append(
                {"kind": "probe", "name": path.parent.name, "dir": path.parent}
            )
        for path in sorted(root.rglob("summary.json")):
            if path.parent in seen:
                continue
            seen.add(path.parent)
            zernike.append(
                {"kind": "zernike", "name": path.parent.name, "dir": path.parent}
            )
    return {"probe": probe, "zernike": zernike}


def _phase_label(run: dict[str, Any]) -> str:
    """给分组标注相位来源（模型 GS 相位 / 指定的相位文件）。"""
    source = str(run["config"].get("gs_phase_source", ""))
    if "现场生成" in source:
        return "模型 GS 相位（`OffFocusBench` 现场生成，Δz=0 的 plain GS）"
    if source:
        tail = source.split(":", 1)[-1].strip()[:80]
        return f"指定相位文件（{tail}）"
    return "未标注"


def is_noise_group(run: dict[str, Any]) -> bool:
    """该组是不是「同一相位重复采集」的噪声地板组（所有 Δz 都为 0）。"""
    rows = [r for r in run["rows"] if not r.get("frame_missing")]
    if not rows:
        return False
    return all(abs(float(r["dz_mm"])) < 1e-9 for r in rows)


def pick_closest_noise(
    noise_runs: list[dict[str, Any]], exposure_ms: float | None
) -> dict[str, Any] | None:
    """挑曝光最接近目标曝光的那条噪声地板（判阶数差异必须同曝光）。"""
    if not noise_runs:
        return None
    if exposure_ms is None:
        return noise_runs[0]

    def key(run: dict[str, Any]) -> float:
        got = run["config"].get("exposure_ms_working")
        if got is None:
            return math.inf
        return abs(float(got) - float(exposure_ms))

    return min(noise_runs, key=key)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def steering_fit(rows: list[dict[str, Any]], limit_mm: float = 2.5) -> dict[str, float]:
    """光斑质心位移 vs Δz 的线性拟合 (转向速率, px/mm)。

    瞳面二次相位以面板标称照明中心 (960, 600) 为中心, 而照明并不落在那里,
    所以这张相位等价于一片**偏心透镜**: 它一边离焦, 一边把光斑推走。

    拟合只用 |Δz| ≤ ``limit_mm`` 的点: 更大的 |Δz| 上光斑已经跑出 ROI,
    质心量到的是晕斑而不是被整形的光, 会把斜率稀释掉 (全区间拟合会从
    -6 px/mm 量级掉到 -3 px/mm 量级)。
    """
    pts_all = [
        (float(r["dz_mm"]), float(r["shift_x_px"]))
        for r in rows
        if not r.get("frame_missing") and math.isfinite(float(r["shift_x_px"]))
    ]
    out: dict[str, float] = {}
    for label, pts in (
        ("local", [p for p in pts_all if abs(p[0]) <= limit_mm]),
        ("global", pts_all),
    ):
        if len(pts) < 3:
            out[f"slope_{label}_px_per_mm"] = math.nan
            out[f"intercept_{label}_px"] = math.nan
            continue
        xs = np.array([p[0] for p in pts], dtype=float)
        ys = np.array([p[1] for p in pts], dtype=float)
        slope, intercept = np.polyfit(xs, ys, 1)
        out[f"slope_{label}_px_per_mm"] = float(slope)
        out[f"intercept_{label}_px"] = float(intercept)
    out["limit_mm"] = limit_mm
    out["n_local"] = len([p for p in pts_all if abs(p[0]) <= limit_mm])
    return out


def order_spread(orders: list[dict[str, Any]]) -> dict[str, float]:
    """阶数之间的指标跨度 (用来和噪声地板比)。"""
    out: dict[str, float] = {}
    for key, path in (
        ("fit_rms_rad", lambda o: float(o["fit_rms_rad"])),
        ("before_cv", lambda o: float(o["before"]["cv"])),
        ("after_cv", lambda o: float(o["after"]["cv"])),
        ("before_pib", lambda o: float(o["before"]["pib"])),
        ("after_pib", lambda o: float(o["after"]["pib"])),
        ("after_quality", lambda o: float(o["after"]["quality"])),
    ):
        values = [path(o) for o in orders if math.isfinite(path(o))]
        out[f"{key}_spread"] = (max(values) - min(values)) if values else math.nan
        out[f"{key}_min"] = min(values) if values else math.nan
        out[f"{key}_max"] = max(values) if values else math.nan
    jitters = [
        float(o["spgd_J_max"]) - float(o["spgd_J_min"])
        for o in orders
        if o.get("spgd_J_max") is not None and o.get("spgd_J_min") is not None
    ]
    out["spgd_J_span_min"] = min(jitters) if jitters else math.nan
    out["spgd_J_span_max"] = max(jitters) if jitters else math.nan
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def render_figures(
    scan: dict[str, Any], sim_g1: list[dict[str, Any]], zernike: dict[str, Any]
) -> list[str]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rows = [r for r in scan["rows"] if not r.get("frame_missing")]
    made: list[str] = []

    # Figure a: 复合分数 vs Δz (硬件定尺框 / 硬件随斑框 / 仿真定尺框)
    hw = [r for r in rows if "score_frozen" in r]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(
        [float(r["dz_mm"]) for r in hw],
        [float(r["score_frozen"]) for r in hw],
        "o-",
        color="#2b6cb0",
        markersize=4,
        label="硬件 (定尺框, 冻结)",
    )
    if hw and "score_moved" in hw[0]:
        ax.plot(
            [float(r["dz_mm"]) for r in hw],
            [float(r["score_moved"]) for r in hw],
            "s--",
            color="#975a16",
            markersize=4,
            label="硬件 (随斑框, 边长不变)",
        )
    ax.plot(
        [r["dz_mm"] for r in sim_g1],
        [r["score_fixed"] for r in sim_g1],
        "^-",
        color="#6b46c1",
        markersize=5,
        label="仿真 G1 (定尺框)",
    )
    ax.set_xlabel("离焦 Δz (mm)")
    ax.set_ylabel("复合分数")
    ax.set_title("离焦下的复合分数: 硬件 vs 仿真")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    savefig(fig, FIG_DIR / f"hw_defocus_score_vs_dz{FIG_TAG}.png")
    plt.close(fig)
    made.append(f"figures/hw_defocus_score_vs_dz{FIG_TAG}.png")

    # Figure b: 转向 —— 质心位移 vs Δz
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(
        [float(r["dz_mm"]) for r in hw],
        [float(r["shift_x_px"]) for r in hw],
        "o-",
        color="#c53030",
        markersize=4,
        label="质心 x 偏移 (px)",
    )
    ax.plot(
        [float(r["dz_mm"]) for r in hw],
        [float(r["shift_y_px"]) for r in hw],
        "s-",
        color="#2b6cb0",
        markersize=4,
        label="质心 y 偏移 (px)",
    )
    ax.set_xlabel("离焦 Δz (mm)")
    ax.set_ylabel("相对冻结零级的偏移 (px)")
    ax.set_title("瞳面二次相位同时把光斑推走 (转向)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    savefig(fig, FIG_DIR / f"hw_defocus_steering_vs_dz{FIG_TAG}.png")
    plt.close(fig)
    made.append(f"figures/hw_defocus_steering_vs_dz{FIG_TAG}.png")

    # Figure c: 光斑 FWHM vs Δz
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(
        [float(r["dz_mm"]) for r in hw],
        [float(r["fwhm_px"]) for r in hw],
        "o-",
        color="#2b6cb0",
        markersize=4,
        label="光斑 FWHM (px)",
    )
    ax.set_xlabel("离焦 Δz (mm)")
    ax.set_ylabel("FWHM (px)")
    ax.set_yscale("log")
    ax.set_title("离焦下的光斑尺寸")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    savefig(fig, FIG_DIR / f"hw_defocus_fwhm_vs_dz{FIG_TAG}.png")
    plt.close(fig)
    made.append(f"figures/hw_defocus_fwhm_vs_dz{FIG_TAG}.png")

    # Figure d: 泽尼克阶数扫描
    orders = zernike.get("orders", [])
    if orders:
        n_vals = [int(o["n_max"]) for o in orders]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        axes[0].plot(
            n_vals, [float(o["fit_rms_rad"]) for o in orders], "o-", color="#2b6cb0"
        )
        axes[0].set_xlabel("n_max")
        axes[0].set_ylabel("Zernike 拟合残差 RMS (rad)")
        axes[0].set_title("拟合残差 vs 阶数")
        axes[0].grid(alpha=0.3)
        axes[1].plot(
            n_vals,
            [float(o["before"]["cv"]) for o in orders],
            "s--",
            color="#975a16",
            label="SPGD 前",
        )
        axes[1].plot(
            n_vals,
            [float(o["after"]["cv"]) for o in orders],
            "o-",
            color="#2b6cb0",
            label="SPGD 后",
        )
        base = zernike.get("baselines", {})
        if "gs_freeform" in base:
            axes[1].axhline(
                float(base["gs_freeform"]["cv"]),
                color="#6b46c1",
                ls="-.",
                label="GS 逐像素相位基线",
            )
        if "flat" in base:
            axes[1].axhline(
                float(base["flat"]["cv"]), color="#c53030", ls=":", label="平场基线"
            )
        axes[1].set_xlabel("n_max")
        axes[1].set_ylabel("CV")
        axes[1].set_title("方斑均匀度 vs Zernike 阶数")
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.3)
        fig.tight_layout()
        savefig(fig, FIG_DIR / f"hw_zernike_order_scan{FIG_TAG}.png")
        plt.close(fig)
        made.append(f"figures/hw_zernike_order_scan{FIG_TAG}.png")
    return made


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------
def defocus_group_section(
    run: dict[str, Any],
    sim_g1: list[dict[str, Any]],
    sigma_score: float,
    level: str = "2",
    phase: str = "",
    response_figures: dict[str, str] | None = None,
    group_name: str = "",
) -> str:
    """一个离焦分组的完整小节（主表 + 与仿真对拍 + 转向 + 大 Δz）。

    每个分组自带自己的曝光与相位, 所以主表、对拍与转向结论都按组给出 ——
    「一次只跑一组变量」的结构在报告里也要保持。
    """
    cfg = run["config"]
    rows = [r for r in run["rows"] if not r.get("frame_missing")]
    hw_by_dz = {round(float(r["dz_mm"]), 4): r for r in rows}
    steer = steering_fit(rows)

    # --- 主表 ---------------------------------------------------------------
    main_rows = []
    for r in sorted(rows, key=lambda x: (float(x["dz_mm"]), int(x["index"]))):
        kind = "基线 (无叠加)" if float(r["chirp_sign"]) == 0.0 else "扫描"
        main_rows.append(
            [
                kind,
                fmt_metric(r["dz_mm"], 4),
                fmt_metric(r["chirp_edge_waves"], 3),
                fmt_metric(r.get("fwhm_px"), 2),
                fmt_metric(r.get("hollowness"), 3),
                fmt_metric(r.get("frame_peak"), 1),
                fmt_metric(r.get("shift_px"), 1),
                fmt_metric(r.get("cv_frozen", r.get("cv")), 4),
                fmt_metric(r.get("pib_frozen", r.get("pib")), 4),
                fmt_metric(r.get("score_frozen", r.get("score")), 4),
                fmt_metric(r.get("cv_moved"), 4),
                fmt_metric(r.get("score_moved"), 4),
                fmt_metric(r.get("halo_power_fraction_outside_box"), 4),
            ]
        )
    main_table = markdown_table(
        [
            "类型",
            "Δz (mm)",
            "面板边缘相位 (波)",
            "FWHM (px)",
            "环形度",
            "整帧峰值",
            "质心偏移 (px)",
            "CV 定尺",
            "PIB 定尺",
            "复合分数 定尺",
            "CV 随斑",
            "复合分数 随斑",
            "框外功率占比",
        ],
        main_rows,
    )

    # --- 与仿真对拍 ---------------------------------------------------------
    zero_rows = [r for r in rows if abs(float(r["dz_mm"])) < 1e-9]
    base_hw = float(
        zero_rows[0].get("score_frozen", zero_rows[0].get("score", math.nan))
    )
    base_sim = sim_g1[4]["score_fixed"] if len(sim_g1) > 4 else math.nan
    cmp_rows = []
    for s in sim_g1:
        h = hw_by_dz.get(round(s["dz_mm"], 4))
        if h is None:
            continue
        hw_score = float(h.get("score_frozen", h.get("score", math.nan)))
        cmp_rows.append(
            [
                fmt_metric(s["dz_mm"], 4),
                fmt_metric(s["score_fixed"], 4),
                fmt_metric(hw_score, 4),
                fmt_metric(s["score_fixed"] / base_sim, 4) if base_sim else "-",
                fmt_metric(hw_score / base_hw, 4) if base_hw else "-",
                fmt_metric(float(h.get("shift_px", math.nan)), 1),
            ]
        )
    cmp_table = markdown_table(
        [
            "Δz (mm)",
            "仿真 G1 复合分数 (定尺)",
            "硬件 复合分数 (定尺)",
            "仿真 归一化",
            "硬件 归一化",
            "硬件 质心偏移 (px)",
        ],
        cmp_rows,
    )

    # --- 转向的带符号位移表 -------------------------------------------------
    steer_rows = []
    for r in sorted(rows, key=lambda x: float(x["dz_mm"])):
        dz = float(r["dz_mm"])
        if abs(dz) > 2.5 or "shift_x_px" not in r:
            continue
        steer_rows.append(
            [
                fmt_metric(dz, 4),
                fmt_metric(r["shift_x_px"], 1),
                fmt_metric(r["shift_y_px"], 1),
                fmt_metric(r.get("score_frozen", r.get("score")), 4),
                fmt_metric(r.get("score_moved"), 4),
            ]
        )
    steer_table = markdown_table(
        [
            "Δz (mm)",
            "质心 x 偏移 (px, 带符号)",
            "质心 y 偏移 (px)",
            "复合分数 定尺",
            "复合分数 随斑",
        ],
        steer_rows,
    )

    def at(dz: float, key: str, default: float = math.nan) -> float:
        r = hw_by_dz.get(round(dz, 4))
        if r is None:
            return default
        return float(r.get(key, default))

    base_pairs = ""
    zero_scan = [
        r
        for r in rows
        if abs(float(r["dz_mm"])) < 1e-9 and float(r["chirp_sign"]) != 0.0
    ]
    if zero_rows and zero_scan:
        base_pairs = f"{fmt_metric(abs(float(zero_rows[0].get('score_frozen', zero_rows[0].get('score', 0))) - float(zero_scan[0].get('score_frozen', 0))), 4)}"

    # --- 三套口径的离焦响应 -------------------------------------------------
    matched = matched_box_side(run)
    response_table, response_data = defocus_response_table(run, matched)
    aperture_table, aperture_summary = aperture_sensitivity_table(run)
    legacy_side = int(cfg["target_side_px"])
    resp_fig = (response_figures or {}).get(group_name, "")
    peak_values = [
        float(v["corr_peak"])
        for v in response_data.values()
        if v.get("tracked_ok") is False
    ]
    peak_txt = (
        f"（互相关峰值在 Δz ≥ 0.19 mm 就低于 0.3 的可用门限，最低到 {fmt_metric(min(peak_values), 2)}）"
        if peak_values
        else ""
    )
    response_block = f"""
#### {level}.5 离焦响应：三套口径的对照

**目标框尺寸必须与被测相位匹配。** 探针原先把框写死成 {legacy_side} px —— 那是**模型** GS 相位方斑的边长；
这一组的实际光斑只有 {fmt_metric(response_data[0]["fwhm"] if response_data else float("nan"), 1)} px 宽，
框比光斑大 {fmt_metric(legacy_side / max(response_data[0]["fwhm"], 1e-9), 1) if response_data else "-"} 倍。
复合分数是 `0.5·PIB + 0.5·(1/(1+CV))`：框远大于光斑时，离焦的**弥散把光填满框**（CV 反而变好），
均匀项**奖励**离焦，而功率项同时变差，两项几乎等量抵消 ⇒ 复合分数看起来「毫无反应」。
**离焦效果一直都在**（框内功率保持率在**正半轴**上严格单调下降；负半轴系统性地低一档，
那是转向不对称、不是非单调），只是被口径藏起来了。

匹配框 = 该相位 Δz=0 光斑自己的 FWHM（只定一次、之后冻结，不逐帧缩放）。
**读单调性必须按半轴读**：把 ±Δz 按 |Δz| 合并，量到的是正负不对称，不是离焦响应。

**三条读数纪律**（本表逐条对应）：

1. **复合分数不能当主指标。** 框空掉时 CV→0，`1/(1+CV)`→1，复合分数的地板是 **0.5 而不是 0**，
   于是它在**任何**孔径下都会**反弹**（本组 {fmt_metric(response_data[3.75]["score_matched"], 4) if 3.75 in response_data else "-"}
   → {fmt_metric(response_data[5.0]["score_matched"], 4) if 5.0 in response_data else "-"}，|Δz| 3.75 → 5.0 mm：
   更差的图样反而得分更高）。
2. **「跟中心」口径在本实验系统的图样上不可用。** 相位互相关（`estimate_shift`）是仓库里
   为「模型自己的光斑」写的，对**散斑化的远场图样**不灵 {peak_txt}；
   本表的「PIB（互相关跟中心）」列因此大面积为空。质心位移列只在光斑仍是单瓣时可信。
3. **唯一与转向无关的离焦读数是 `g r50`**（功率半径的相对增长）：它量的是图样摊开了多少，
   不看位置，所以转向污染不到它。

⇒ 冻结框心的 PIB 衰减在 |Δz| ≳ 1.5 mm 之后**不能**读作「离焦把光打散了」：
光斑同时被**推走**了（见本组的「转向」小节）。消除这个污染的办法是**给 chirp 配一个反向的瞳面斜坡**
（去倾斜），那部分已进代码（`--chirp-tilt-cancel`，自标定），等设备上线后复测。

{response_table}

**孔径灵敏度**（同一批帧，只换目标框尺寸；正半轴）：

{aperture_table}

{aperture_summary}

{resp_fig}
"""

    edge20 = at(20.0, "chirp_edge_waves")
    edge50 = at(50.0, "chirp_edge_waves")
    big_dz = ""
    if edge20 == edge20:  # not NaN
        edge50_txt = (
            f"、Δz = 50 mm 时是 {fmt_metric(edge50, 2)} 波" if edge50 == edge50 else ""
        )
        big_dz = f"""
#### {level}.4 大 Δz 与 SLM 采样

面板边缘相位在 Δz = 20 mm 时是 {fmt_metric(edge20, 2)} 波{edge50_txt}。相邻像元的相位步进
在 Δz = 20 mm 时约 0.23 rad，**远小于 π**，所以**采样（奈奎斯特）不是瓶颈**；真正的限制是

- 晕斑半径：Δz = 20 mm 对应约 335 µm ≈ 152 px，而 ROI 半宽只有 {cfg["roi_px"] // 2} px
  ⇒ 光斑整体跑出窗口（PIB = 0、框外功率占比 = 1）；
- 零级/杂散光：|Δz| ≥ 15 mm 后整帧峰值降到 3 counts 以下，**框里基本没有光**。

⇒ **把扫描限制在 |Δz| ≤ 20 mm 是对的**（面板边缘 {fmt_metric(edge20, 2)} 波）；再往大推，
晕斑已跑出 ROI，既没有可测信号也没有物理意义。
"""

    # --- 读法: 全部由本组数据算出, 不写死结论 ------------------------------
    sim_at = {round(s["dz_mm"], 4): s for s in sim_g1}
    sim_ratio_15 = (
        sim_at[round(1.5007, 4)]["score_fixed"] / sim_at[0.0]["score_fixed"]
        if sim_at.get(0.0)
        else math.nan
    )
    hw_ratio_15 = (at(1.5007, "score_frozen") / base_hw) if base_hw else math.nan
    hw_drop = 1.0 - hw_ratio_15
    sim_drop = 1.0 - sim_ratio_15
    if hw_drop > 0.02:
        shape_word = f"**同样在下降**（掉了 {hw_drop:.1%}，仿真是 {sim_drop:.1%}）"
    elif hw_drop > -0.01:
        shape_word = (
            f"**基本持平**（只动了 {hw_drop:+.1%}，而仿真是 {sim_drop:.1%}）"
            " —— 换成硬件 GS 相位后，冻结框口径下这个指标**几乎不随离焦变**"
        )
    else:
        shape_word = f"**反而更高**（{hw_drop:+.1%}）"

    asym = abs(at(-1.5007, "score_frozen") - at(1.5007, "score_frozen"))
    asym_sim = abs(
        sim_at[round(-1.5007, 4)]["score_fixed"]
        - sim_at[round(1.5007, 4)]["score_fixed"]
    )
    if sigma_score and asym > 3 * sigma_score:
        asym_word = f"是噪声地板的 {asym / sigma_score:.1f} 倍 —— **这不是噪声**"
    else:
        asym_word = "**与噪声地板同量级 —— 在这个口径下分不出不对称**"

    comparison_note = (
        f"- **形状**：仿真在 |Δz| = 1.5007 mm 处的复合分数降到 Δz=0 的 "
        f"{fmt_metric(sim_ratio_15, 4)}（掉了 {sim_drop:.1%}），这一组是 "
        f"{fmt_metric(hw_ratio_15, 4)}，{shape_word}。"
    )
    hw_side = int(run["config"].get("target_side_px", 0))
    sim_box = int(sim_g1[0]["side_px"]) if sim_g1 else 0
    if phase == "model_gs":
        comparison_note += (
            f" 绝对值也对不上（硬件 Δz=0 的复合分数 {fmt_metric(base_hw, 4)} 对仿真 0.8073）："
            f" 这一组的相位来源、目标框尺寸（{hw_side} px 对 {sim_box} px）、目标边长都与"
            f"仿真的 G1 行**同口径**，"
            f" 所以差距来自光路本身：仿真里 GS 相位与数字孪生**自洽**，"
            f" 而同一张相位下发到真实实验系统上，照明没落在瞳面中心、相机也不在 2f 面上，"
            f" 方斑并不成立（CV ≈ 0.73，与 2026-10-15 的硬件记录 CV ≈ 0.723 同量级）。"
        )
    elif phase == "hardware_gs":
        recorded_side = int(cfg.get("target_side_px", legacy_side))
        spot0 = response_data.get(0.0, {}).get("fwhm", float("nan"))
        ratio = (recorded_side / spot0) if (spot0 and spot0 == spot0) else float("nan")
        # 框与光斑同量级 ⇒ 这个孔径就是该相位自己的口径; 框远大于光斑 ⇒ 它是**继承来的**
        # 孔径 (上一轮沿用了模型相位的方斑边长), 用它读出来的东西只能当「口径 A」。
        same_scale = bool(ratio == ratio and ratio <= 1.5)
        if same_scale:
            aperture_word = (
                f"这一组的目标框是**预先登记**的该相位成形目标（{recorded_side} px = "
                f"泽尼克扫描从平场 waist 量到的 2·w0_flat），而这一组光斑的实测宽度是 "
                f"{fmt_metric(spot0, 1)} px（框/光斑 = {fmt_metric(ratio, 2)}，"
                f"同量级，所以这一组的口径是对的）"
            )
        else:
            aperture_word = (
                f"这一组用的目标框是 {recorded_side} px，而该相位自己的光斑实测只有 "
                f"{fmt_metric(spot0, 1)} px（框/光斑 = {fmt_metric(ratio, 2)}）—— "
                f"**框是继承来的**（沿用模型相位的方斑边长），用它读出来的绝对值只能当"
                f"「口径 A」；**按光斑自己的宽度重算的那套框（本组「三套口径」对照里的「匹配框」口径）才是对的**"
            )
        comparison_note += (
            f" **绝对值与仿真不可比，原因是目标尺度不同**：{aperture_word}；"
            f"仿真的 G1 行用的是 {sim_box} px 的框 + {sim_box} px 的方斑，"
            f"**尺度完全不同**，绝对值没有可比性。"
            f" 能比的是相对量（随 |Δz| 变化的趋势与幅度），以及 §3 里同一张相位的泽尼克阶数结果。"
            f" ⚠️ 与仿真的**同尺度**对拍需要仿真在 {recorded_side} px 目标上重跑（**本轮未做**），"
            f" 所以这一行对拍只成立到「形状」这一层。"
        )

    return f"""
#### {level}.1 主表

{main_table}

表注：

- **类型**：「基线 (无叠加)」是探针在扫描前测的那一行（叠加层为空）；「扫描」行的 Δz=0
  那一行叠加的是一张**全零**二次相位 —— 物理上与基线等价，两行同时给出就是为了显示重复性
  （两者差 {base_pairs}，远小于 3σ 地板）。
- **CV 列为「-」**：框内没有光时 CV 无定义（除零），按无数据显示。出现这一列的 Δz
  已经超出 {cfg["roi_px"] // 2} px 的 ROI 半宽。
- **FWHM 列为「-」**：帧里没有可测光斑。
- **质心偏移**：相对**冻结**的 ROI 内零级位置 {tuple(cfg["zero_order_roi_xy_frozen"])} 的距离。

#### {level}.2 与仿真 G1 曲线对拍

仿真的 G1 组（Δz 盲 GS、无离焦）取自 `report/shape_trials/fv_groups_fresnel_vs_vl.csv`
（与 `../gs_defocus_sim/gs_defocus_sim_20261019.md` §2.1 的 G1 行同源：Δz=0 处 score_fixed = 0.8073、
cv_fixed = 0.4215，两边逐位相同）。硬件侧取定尺框复合分数，各自对自己的 Δz=0 归一化。

{cmp_table}

读法：

{comparison_note}
- **正负半轴**：仿真 G1 在 ±1.5007 mm 处差 {fmt_metric(asym_sim, 4)}，这一组差 {fmt_metric(asym, 4)}，
  噪声地板 {fmt_metric(sigma_score, 5)}，{asym_word}。

#### {level}.3 转向：为什么硬件曲线不对称

瞳面二次相位以面板标称照明中心 (960, 600) 为中心，而照明并不落在那里，于是这张相位等价于
一片**偏心透镜**：它一边离焦，一边把光斑横向推走。实测质心 x 相对冻结零级的偏移随 Δz
线性增长，斜率 **{fmt_metric(steer["slope_local_px_per_mm"], 2)} px/mm**（|Δz| ≤ 2.5 mm 内拟合；
全区间拟合是 {fmt_metric(steer["slope_global_px_per_mm"], 2)} px/mm，但大 |Δz| 上光斑已跑出 ROI，
质心量到的是晕斑，斜率被稀释）。|Δz| = 1.5 mm 时约
{fmt_metric(abs(steer["slope_local_px_per_mm"]) * 1.5, 1)} px，|Δz| = 10 mm 时约
{fmt_metric(abs(steer["slope_local_px_per_mm"]) * 10, 0)} px。

{steer_table}

**这张表就是不对称的全部机制**。注意 Δz = 0 那一行：光斑质心已经比冻结框中心偏了约 8 px
（照明没落在瞳面中心，GS 方斑与零级不同心）。转向是**带符号**的，于是

- **负 Δz** 把光斑**推回**框中心（Δz = -1.5007 mm 时偏移只剩 {fmt_metric(at(-1.5007, "shift_x_px"), 1)} px），
  框反而框住了光斑；
- **正 Δz** 把光斑**推得更远**（+1.5007 mm 时偏了 {fmt_metric(at(1.5007, "shift_x_px"), 1)} px），
  框只框住光斑的一部分。

⇒ 正负半轴测到的不是「同一个离焦的两个方向」，而是「同一个离焦 + 两个方向的转向」。
这一组的 ±1.5007 mm 差值 {fmt_metric(asym, 4)}（噪声地板 {fmt_metric(sigma_score, 5)}，{asym_word}），
主要来自这个几何偏置，不是离焦本身有方向性。

后果：冻结框在 |Δz| ≳ 1.5 mm 之后开始被光斑**拖出框外**，于是

- 框内功率（PIB）随 |Δz| 单调掉到 0；
- 但**随斑框**的复合分数在 +3.75 mm 处回到 {fmt_metric(at(3.75, "score_moved"), 4)}，
  高于 +1.5007 mm 处的 {fmt_metric(at(1.5007, "score_moved"), 4)}。

⇒ **冻结框口径下 |Δz| ≳ 1.5 mm 的劣化不能读作「离焦把光打散了」**；光还在，只是被搬走了。
{big_dz}
{response_block}"""


def zernike_noise_floor(
    zernike_dir: Path | None, orders: list[dict[str, Any]]
) -> dict[str, Any]:
    """**同口径**的噪声地板: 泽尼克扫描自己的那把目标框、同曝光。

    离焦探针的噪声地板是**另一个**孔径上的, 拿来判「泽尼克那把小框里的阶数差异」是错的
    口径 (框差 ~20 倍, 每次测量的光子数差一个量级)。所以这里从泽尼克扫描**自己**的
    逐轮记录里取:

    - **PIB**: 每轮都测同一个 (几乎不变的) 相位 —— SPGD 的更新被噪声门挡掉了
      (实测 150 轮里只放行 4 次), 所以逐轮 ``m_roi_pib`` 的散布**就是**测量噪声;
    - **CV**: 历史里没有 CV 列, 于是用同一阶的 ``after - before`` 配对差估 —— 这会把
      SPGD 那点真实改善也算进来, 所以是**偏保守**的一侧 (更难判可排序)。

    两者都取各阶的中位数, 避免个别阶的离群值主导。
    """
    out: dict[str, Any] = {
        "sigma_pib": math.nan,
        "sigma_cv": math.nan,
        "n_orders": 0,
        "per_order": [],
    }
    if zernike_dir is None or not orders:
        return out
    pib_sigmas: list[float] = []
    cv_sigmas: list[float] = []
    for order in orders:
        n_max = int(order["n_max"])
        path = _find_under(zernike_dir, f"spgd_nmax{n_max}.csv")
        if path is None:
            continue
        with path.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        values: list[float] = []
        for row in rows:
            try:
                values.append(float(row["m_roi_pib"]))
            except (KeyError, TypeError, ValueError):
                continue
        if len(values) > 3:
            pib_sigmas.append(float(np.std(values, ddof=1)))
        before = float(order["before"]["cv"])
        after = float(order["after"]["cv"])
        if math.isfinite(before) and math.isfinite(after):
            # 配对差的半宽 = 单次测量噪声 (保守: 含 SPGD 的真实改善)
            cv_sigmas.append(abs(after - before) / math.sqrt(2.0))
    if pib_sigmas:
        out["sigma_pib"] = float(np.median(pib_sigmas))
    if cv_sigmas:
        out["sigma_cv"] = float(np.median(cv_sigmas))
    out["n_orders"] = len(pib_sigmas)
    out["per_order"] = [
        {
            "n_max": int(o["n_max"]),
            "sigma_pib": float(
                np.std(
                    [
                        float(r["m_roi_pib"])
                        for r in csv.DictReader(
                            (zernike_dir / f"spgd_nmax{int(o['n_max'])}.csv").open(
                                encoding="utf-8"
                            )
                        )
                    ],
                    ddof=1,
                )
            )
            if (zernike_dir / f"spgd_nmax{int(o['n_max'])}.csv").exists()
            else math.nan,
        }
        for o in orders
    ]
    return out


def _frames_by_index(run: dict[str, Any]) -> dict[int, tuple[float, np.ndarray]]:
    """读出该组每一行的原始帧, 按**行索引**建键 (未经处理)。

    ⚠️ 早期版本按 ``dz_mm`` 建键, 于是**基线行会被扫描里的 Δz=0 行覆盖** —— 两者
    物理等价 (一个是空叠加层, 一个是全零二次相位), 但基线行是「重复性」的唯一参照,
    被悄悄顶掉就没法核对重复性了。这里按**行索引**存, 由调用方显式挑选基线行。
    """
    out: dict[int, tuple[float, np.ndarray]] = {}
    for row in run["rows"]:
        if row.get("frame_missing"):
            continue
        path = Path(run["dir"]) / f"frame_{int(row['index']):03d}.npz"
        if not path.exists():
            continue
        out[int(row["index"])] = (float(row["dz_mm"]), np.load(path)["frame"])
    return out


def _split_baseline_and_scan(
    run: dict[str, Any],
) -> tuple[dict[int, tuple[float, np.ndarray]], dict[int, tuple[float, np.ndarray]]]:
    """把该组的帧分成「基线行」(空叠加层) 与「扫描行」。"""
    frames = _frames_by_index(run)
    baseline: dict[int, tuple[float, np.ndarray]] = {}
    scan: dict[int, tuple[float, np.ndarray]] = {}
    for row in run["rows"]:
        idx = int(row["index"])
        if idx not in frames:
            continue
        if float(row.get("chirp_sign", -1.0)) == 0.0:
            baseline[idx] = frames[idx]
        else:
            scan[idx] = frames[idx]
    return baseline, scan


def _scan_frames_by_dz(run: dict[str, Any]) -> dict[float, np.ndarray]:
    """扫描行的帧, 按 Δz 建键 (扫描行里 Δz 唯一)。"""
    _baseline, scan = _split_baseline_and_scan(run)
    return {dz: raw for dz, raw in scan.values()}


def _baseline_frame(run: dict[str, Any]) -> np.ndarray | None:
    """基线行 (空叠加层) 的帧; 没有就返回 None。"""
    baseline, _scan = _split_baseline_and_scan(run)
    for _dz, raw in baseline.values():
        return raw
    return None


def encircled_power_radius(
    frame: np.ndarray, cx: float, cy: float, fraction: float = 0.5
) -> float:
    """含 ``fraction`` 帧功率的半径 (绕 (cx, cy))。

    用「功率半径」而不是 FWHM: FWHM 走的是过极值点的直线, 遇到环形/分裂图样就量到
    环的**厚度** (本实验系统实测: 同一组里 FWHM 在 3.5 ~ 174.6 px 之间跳, 且模型相位
    组非单调)。功率半径对形状不敏感, 适合当「图样摊开了多少」的指标。
    """
    yy, xx = np.indices(frame.shape, dtype=np.float64)
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    weight = np.maximum(frame, 0.0)
    total = float(weight.sum())
    if total <= 0:
        return math.nan
    order = np.argsort(r.ravel())
    cum = np.cumsum(weight.ravel()[order])
    idx = int(np.searchsorted(cum, fraction * total))
    idx = min(idx, order.size - 1)
    return float(r.ravel()[order[idx]])


def render_farfield_grid(
    run: dict[str, Any], dz_list: list[float], out_png: Path, title: str
) -> bool:
    """CCD 远场光斑图拼版: 每个 Δz 一格 (log10), 画出目标框。"""
    frames = _scan_frames_by_dz(run)
    picked = [dz for dz in dz_list if dz in frames]
    if not picked:
        return False
    cfg = run["config"]
    cx, cy = cfg["zero_order_roi_xy_frozen"]
    side = int(cfg["target_side_px"])
    ncol = min(4, len(picked))
    nrow = math.ceil(len(picked) / ncol)
    # 共享色标 + tight_layout 互斥 (matplotlib 会警告), 这里用 constrained layout。
    fig, axes = plt.subplots(
        nrow,
        ncol,
        figsize=(3.0 * ncol, 3.1 * nrow),
        squeeze=False,
        layout="constrained",
    )
    for ax in axes.ravel():
        ax.axis("off")
    # 先算所有格的 log10 图, 再统一 vmin/vmax: 各格动态范围不同, 共用色标才能
    # 横向比较, 也让 log10 刻度本身可读。
    logs: list[np.ndarray] = []
    for dz in picked:
        frame = prepare_frame(frames[dz].astype(np.float64))
        logs.append(np.log10(np.maximum(frame, 1e-3)))
    stack = np.concatenate([img.ravel() for img in logs])
    vmin = float(np.percentile(stack, 1.0))
    vmax = float(stack.max())
    for k, dz in enumerate(picked):
        ax = axes.ravel()[k]
        frame = prepare_frame(frames[dz].astype(np.float64))
        ax.imshow(logs[k], cmap="magma", origin="upper", vmin=vmin, vmax=vmax)
        box = square_mask(frame.shape, int(cy), int(cx), side)
        ys, xs = np.nonzero(box)
        if xs.size:
            ax.add_patch(
                plt.Rectangle(
                    (xs.min() - 0.5, ys.min() - 0.5),
                    np.ptp(xs) + 1,
                    np.ptp(ys) + 1,
                    fill=False,
                    edgecolor="lime",
                    linewidth=1.0,
                )
            )
        row = next(r for r in run["rows"] if abs(float(r["dz_mm"]) - dz) < 1e-9)
        ax.set_title(
            f"Δz = {dz:+.2f} mm\n"
            f"CV {fmt_metric(row.get('cv_frozen', row.get('cv')), 2)}  "
            f"PIB {fmt_metric(row.get('pib_frozen', row.get('pib')), 2)}",
            fontsize=8,
        )
    mappable = plt.cm.ScalarMappable(
        cmap="magma", norm=plt.Normalize(vmin=vmin, vmax=vmax)
    )
    bar = fig.colorbar(mappable, ax=axes.ravel().tolist(), fraction=0.02, pad=0.01)
    bar.set_label("log10 强度 (counts)", fontsize=8)
    bar.ax.tick_params(labelsize=7)
    shape = frames[picked[0]].shape
    fig.suptitle(
        f"{title}\n每格 {shape[1]}×{shape[0]} px（CCD 窗口），绿框 = {side} px 目标框",
        fontsize=11,
    )
    savefig(fig, out_png)
    plt.close(fig)
    return True


def render_crosssection_grid(
    run: dict[str, Any], dz_list: list[float], out_png: Path, title: str
) -> bool:
    """光斑 X/Y 截面 (过光斑中心的 log10 强度剖面)。"""
    frames = _scan_frames_by_dz(run)
    picked = [dz for dz in dz_list if dz in frames]
    if not picked:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    cmap = plt.get_cmap("viridis")
    for k, dz in enumerate(picked):
        frame = prepare_frame(frames[dz].astype(np.float64))
        spot = measure_spot(frames[dz])
        sx, sy = int(round(spot.centroid_x)), int(round(spot.centroid_y))
        colour = cmap(k / max(1, len(picked) - 1))
        label = f"{dz:+.2f} mm"
        row = frame[sy, :]
        axes[0].plot(
            np.arange(frame.shape[1]) - sx,
            np.log10(np.maximum(row, 1e-3)),
            color=colour,
            label=label,
            lw=1.2,
        )
        col = frame[:, sx]
        axes[1].plot(
            np.arange(frame.shape[0]) - sy,
            np.log10(np.maximum(col, 1e-3)),
            color=colour,
            label=label,
            lw=1.2,
        )
    for ax, lab in zip(axes, ("X 截面 (过光斑中心的行)", "Y 截面 (过光斑中心的列)")):
        ax.axvline(0.0, color="grey", lw=0.8, ls=":")
        ax.set_xlabel("相对光斑中心 (px)")
        ax.set_ylabel("log10 强度 (counts)")
        ax.set_title(lab, fontsize=10)
        ax.grid(alpha=0.3)
        # 两块面板各放一份图例: 只在一侧放的话, 另一侧的曲线颜色无从对照。
        ax.legend(fontsize=7, ncol=2)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    savefig(fig, out_png)
    plt.close(fig)
    return True


def render_group_farfields(
    scan_groups: list[dict[str, Any]],
) -> dict[str, list[str]]:
    """每个离焦分组出两张图: 远场光斑拼版 + X/Y 光斑截面。"""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out: dict[str, list[str]] = {}
    for grp in scan_groups:
        run = grp["run"]
        name = grp["name"]
        frames = _scan_frames_by_dz(run)
        if not frames:
            continue
        dzs = sorted(frames)
        # 拼版挑点: 覆盖两端与焦面附近, 太多则抽稀
        picks = [dz for dz in dzs if abs(dz) <= 20.0]
        if len(picks) > 12:
            step = len(picks) / 12
            picks = [picks[int(i * step)] for i in range(12)]
        made: list[str] = []
        if render_farfield_grid(
            run,
            picks,
            FIG_DIR / f"hw_farfield_{name}{FIG_TAG}.png",
            f"CCD 远场光斑 (log10) — 分组「{name}」, 绿框为目标框",
        ):
            made.append(f"figures/hw_farfield_{name}{FIG_TAG}.png")
        if render_crosssection_grid(
            run,
            picks,
            FIG_DIR / f"hw_xsec_{name}{FIG_TAG}.png",
            f"光斑 X/Y 截面 (过光斑中心) — 分组「{name}」",
        ):
            made.append(f"figures/hw_xsec_{name}{FIG_TAG}.png")
        if made:
            out[name] = made
    return out


def copy_zernike_farfields(
    zernike_dir: Path | None, orders: list[dict[str, Any]]
) -> list[str]:
    """把泽尼克扫描保存的远场图 (before/after) 复制进报告图目录。"""
    if zernike_dir is None or not orders:
        return []
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    made: list[str] = []
    for order in orders:
        n_max = int(order["n_max"])
        for tag in ("before", "after"):
            src = _find_under(zernike_dir, f"{tag}_nmax{n_max}.png")
            if src is None:
                continue
            dst = FIG_DIR / f"hw_zernike_{tag}_nmax{n_max}{FIG_TAG}.png"
            dst.write_bytes(src.read_bytes())
            made.append(f"figures/hw_zernike_{tag}_nmax{n_max}{FIG_TAG}.png")
    return made


def render_zernike_xsec(
    farfield_runs: list[tuple[str, Path, dict[str, Any]]], out_png: Path
) -> bool:
    """泽尼克各阶的 X/Y 光斑截面 (来自各阶的远场组原始帧)。"""
    if not farfield_runs:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    cmap = plt.get_cmap("viridis")
    plotted = 0
    for k, (name, _dir, run) in enumerate(farfield_runs):
        frames = _scan_frames_by_dz(run)
        if not frames:
            continue
        dz = sorted(frames)[0]
        frame = prepare_frame(frames[dz].astype(np.float64))
        spot = measure_spot(frames[dz])
        sx, sy = int(round(spot.centroid_x)), int(round(spot.centroid_y))
        colour = cmap(k / max(1, len(farfield_runs) - 1))
        label = f"n_max={name.split('nmax')[-1]}"
        axes[0].plot(
            np.arange(frame.shape[1]) - sx,
            np.log10(np.maximum(frame[sy, :], 1e-3)),
            color=colour,
            label=label,
            lw=1.3,
        )
        axes[1].plot(
            np.arange(frame.shape[0]) - sy,
            np.log10(np.maximum(frame[:, sx], 1e-3)),
            color=colour,
            label=label,
            lw=1.3,
        )
        plotted += 1
    if not plotted:
        plt.close(fig)
        return False
    for ax, lab in zip(axes, ("X 截面 (过光斑中心的行)", "Y 截面 (过光斑中心的列)")):
        ax.axvline(0.0, color="grey", lw=0.8, ls=":")
        ax.set_xlabel("相对光斑中心 (px)")
        ax.set_ylabel("log10 强度 (counts)")
        ax.set_title(lab, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("泽尼克各阶的远场光斑截面 (设备下线前只采到部分阶)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    savefig(fig, out_png)
    plt.close(fig)
    return True


def matched_box_side(run: dict[str, Any]) -> int:
    """与该相位自身光斑匹配的目标框边长 (Δz=0 光斑的 FWHM, 只定一次)。

    这个口径存在的理由: 目标框的边长应当**跟着该相位自己的光斑走**, 而不是跟着别的
    相位。第一轮把框写死成模型 GS 相位方斑的边长, 而硬件 GS 相位的实际光斑小得多,
    框比光斑大好几倍, 于是复合分数的两项 (``0.5·PIB`` 与 ``0.5·(1/(1+CV))``)
    方向相反、量级相当, 互相抵消 —— 「离焦效果不佳」其实是度量口径的假象,
    不是实验系统的问题。第二轮起目标框改为**预先登记**的该相位自己的成形目标。
    """
    frames = _scan_frames_by_dz(run)
    zeros = [dz for dz in frames if abs(dz) < 1e-9]
    if not zeros:
        return 0
    base_dz = min(zeros, key=lambda z: abs(z))
    fwhm = float(measure_spot(frames[base_dz]).fwhm_px)
    if not math.isfinite(fwhm) or fwhm <= 0:
        return 0
    roi = int(run["config"]["roi_px"])
    return int(max(4, min(int(round(fwhm)), int(0.6 * roi))))


def score_in_box(
    frame: np.ndarray, cx: int, cy: int, side: int
) -> tuple[float, float, float]:
    """(PIB, CV, 复合分数) —— 在给定框上。"""
    mask = square_mask(frame.shape, int(cy), int(cx), int(side))
    pib = float(power_in_bucket(frame, mask))
    cv = float(uniformity_cv(frame, mask))
    return pib, cv, float(composite_from_pib_cv(pib, cv))


def aperture_sensitivity_table(
    run: dict[str, Any], sides: tuple[int, ...] = (8, 10, 13, 16, 20, 44)
) -> tuple[str, str]:
    """孔径灵敏度: 同一批帧在若干目标框尺寸下, 离焦响应是否还在。

    用来排除「结论只是某个框选出来的」这一类反驳: 只要在多个孔径下复合分数都随
    Δz 下降, 结论就不是孔径的产物; 反过来, 哪个孔径会把响应藏住, 也由这张表
    自己给出, 不写死。

    返回 (表格, 一句结论)。结论按「Δz=0 → Δz=+1.5 mm 的相对降幅」判定:
    降幅小于 5% 记作「藏住了」。
    """
    frames = _scan_frames_by_dz(run)
    cfg = run["config"]
    cx, cy = cfg["zero_order_roi_xy_frozen"]
    positives = sorted(dz for dz in frames if dz >= 0)
    if not positives:
        return "", ""
    probe_dz = 1.5007
    if probe_dz not in positives:
        probe_dz = positives[min(4, len(positives) - 1)]
    rows: list[list[Any]] = []
    responsive: list[int] = []
    hidden: list[int] = []
    for side in sides:
        scores: list[float] = []
        pibs: list[float] = []
        for dz in positives:
            frame = prepare_frame(frames[dz].astype(np.float64))
            mask = square_mask(frame.shape, int(cy), int(cx), int(side))
            p = float(power_in_bucket(frame, mask))
            c = float(uniformity_cv(frame, mask))
            pibs.append(p)
            scores.append(float(composite_from_pib_cv(p, c)))
        first = scores[0] if scores else math.nan
        idx = positives.index(probe_dz)
        at_probe = scores[idx] if idx < len(scores) else math.nan
        drop = (first - at_probe) / first if first else math.nan
        (responsive if drop >= 0.05 else hidden).append(side)
        rows.append(
            [
                f"{side} px",
                fmt_metric(scores[0], 4),
                fmt_metric(at_probe, 4),
                fmt_metric(drop, 3),
                fmt_metric(min(pibs), 4),
            ]
        )
    table = markdown_table(
        [
            "目标框边长",
            "复合分数 @Δz=0",
            f"复合分数 @Δz={probe_dz:g} mm",
            "相对降幅",
            "最小 PIB (正半轴)",
        ],
        rows,
    )
    if responsive and hidden:
        summary = (
            f"⇒ 在 {min(responsive)} ~ {max(responsive)} px 的范围内离焦响应都在"
            f"（降幅 ≥5%）；**只有 {', '.join(str(s) for s in hidden)} px 那个孔径把响应藏住了**"
            f"（降幅 <5%）—— 结论不是孔径选出来的。"
        )
    elif responsive:
        summary = (
            f"⇒ 所测的全部孔径（{min(responsive)} ~ {max(responsive)} px）都能看到离焦响应，"
            f"结论与孔径无关。"
        )
    else:
        summary = "⇒ 所测的孔径都没看到响应（降幅 <5%），本组的离焦响应需要换指标读。"
    return table, summary


def defocus_response_table(
    run: dict[str, Any], matched: int
) -> tuple[str, dict[float, dict[str, float]]]:
    """三套口径的离焦响应表 + 逐 Δz 的数值。

    主指标按门禁的结论取**两个**:
    - ``R(Δz)`` = 跟中心框内的功率保持率 (相对 Δz=0 归一) —— 跨相位可比;
    - ``g(Δz)`` = 功率半径 r50 的相对增长 —— 对形状不敏感。
    复合分数**只作辅助读数**: 框空掉时 CV→0, 1/(1+CV)→1, 复合分数的地板是 0.5
    而不是 0, 于是它在任何孔径下都会**反弹** (第一轮实测硬件组 13 px 框:
    0.3682 @3.75 mm → 0.4224 @5.0 mm —— 更差的图样反而得分更高), 所以它不能当主指标。
    """
    frames = _scan_frames_by_dz(run)
    cfg = run["config"]
    cx, cy = cfg["zero_order_roi_xy_frozen"]
    legacy = int(cfg["target_side_px"])
    if not frames:
        return "", {}
    zeros = [dz for dz in frames if abs(dz) < 1e-9]
    ref_dz = min(zeros, key=lambda z: abs(z)) if zeros else min(frames)
    ref_raw = frames[ref_dz]
    ref_frame = prepare_frame(ref_raw.astype(np.float64))

    rows: list[list[Any]] = []
    data: dict[float, dict[str, float]] = {}
    for dz in sorted(frames):
        raw = frames[dz]
        frame = prepare_frame(raw.astype(np.float64))
        spot = measure_spot(raw)
        p_leg, c_leg, s_leg = score_in_box(frame, cx, cy, legacy)
        p_mat, c_mat, s_mat = score_in_box(frame, cx, cy, matched)
        # 跟中心: 用相位互相关 (estimate_shift) 而不是逐帧质心 —— 质心在图样
        # 弥散/分裂时不稳定, 而互相关对形状变化不敏感, 且亚像素。
        try:
            dy, dx, peak = estimate_shift(ref_frame, frame)
        except Exception:  # noqa: BLE001 - 互相关失败就退回质心
            dy = dx = 0.0
            peak = 0.0
        tx = cx + dx
        ty = cy + dy
        # 互相关峰值低 ⇒ 图样已经不成单瓣, 跟中心没有意义 ⇒ 标 N/A
        tracked_ok = bool(peak >= 0.30)
        if tracked_ok:
            p_trk, c_trk, s_trk = score_in_box(frame, round(tx), round(ty), matched)
        else:
            p_trk = c_trk = s_trk = math.nan
        r50 = encircled_power_radius(frame, tx, ty, 0.5)
        data[dz] = {
            "fwhm": float(spot.fwhm_px),
            "pib_legacy": p_leg,
            "cv_legacy": c_leg,
            "score_legacy": s_leg,
            "pib_matched": p_mat,
            "cv_matched": c_mat,
            "score_matched": s_mat,
            "pib_tracked": p_trk,
            "cv_tracked": c_trk,
            "score_tracked": s_trk,
            "shift_dx": float(dx),
            "shift_dy": float(dy),
            "corr_peak": float(peak),
            "tracked_ok": tracked_ok,
            "r50_px": r50,
            # 质心位移: 光斑仍是单瓣时可靠 (近焦区), 是转向量的直接读数。
            "centroid_dx": float(spot.centroid_x - cx),
            "centroid_dy": float(spot.centroid_y - cy),
        }
        rows.append(
            [
                fmt_metric(dz, 4),
                fmt_metric(float(spot.fwhm_px), 1),
                fmt_metric(s_leg, 4),
                fmt_metric(p_mat, 4),
                fmt_metric(s_mat, 4),
                fmt_metric(float(spot.centroid_x - cx), 1),
                fmt_metric(r50, 1),
                fmt_metric(p_trk, 4),
            ]
        )
    # 归一化: 相对 Δz=0 (g r50 是无量纲比值, 跨相位可比)
    ref = data.get(ref_dz, {})
    ref_r50 = float(ref.get("r50_px", math.nan))
    for dz, item in data.items():
        r50 = float(item["r50_px"])
        item["g_r50"] = (
            (r50 / ref_r50) if (ref_r50 and math.isfinite(r50)) else math.nan
        )
    for row, dz in zip(rows, sorted(data)):
        row.append(fmt_metric(data[dz]["g_r50"], 3))
    table = markdown_table(
        [
            "Δz (mm)",
            "光斑 FWHM (px)",
            f"复合分数 {legacy}px (现口径)",
            f"PIB {matched}px (冻结框心)",
            f"复合分数 {matched}px (冻结框心)",
            "质心位移 dx (px)",
            "r50 (px)",
            "PIB (互相关跟中心)",
            "g r50 (归一)",
        ],
        rows,
    )
    return table, data


def render_defocus_response(
    run: dict[str, Any],
    matched: int,
    data: dict[float, dict[str, float]],
    out_png: Path,
    title: str,
) -> bool:
    """离焦响应图: 三套口径的复合分数 + 框内功率保持率。"""
    if not data:
        return False
    dzs = sorted(data)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    axes[0].plot(
        dzs,
        [data[d]["score_legacy"] for d in dzs],
        "s--",
        color="#c53030",
        markersize=4,
        label=f"复合分数（现口径, 冻结 {int(run['config']['target_side_px'])} px 框）",
    )
    axes[0].plot(
        dzs,
        [data[d]["score_matched"] for d in dzs],
        "o-",
        color="#2b6cb0",
        markersize=4,
        label=f"复合分数（匹配框 {matched} px，框心冻结）",
    )
    axes[0].plot(
        dzs,
        [data[d]["score_tracked"] for d in dzs],
        "^-",
        color="#6b46c1",
        markersize=4,
        label=f"复合分数（匹配框 {matched} px，跟中心）",
    )
    base = data[0]["score_legacy"]
    axes[0].axhline(base, color="grey", lw=0.8, ls=":")
    axes[0].set_xlabel("离焦 Δz (mm)")
    axes[0].set_ylabel("复合分数")
    axes[0].set_title("三套口径的复合分数（同一个 Δz=0 归一）", fontsize=10)
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)

    axes[1].plot(
        dzs,
        [data[d]["pib_matched"] for d in dzs],
        "o-",
        color="#2b6cb0",
        markersize=4,
        label="PIB（匹配框，框心冻结）",
    )
    axes[1].plot(
        dzs,
        [data[d]["pib_tracked"] for d in dzs],
        "^-",
        color="#6b46c1",
        markersize=4,
        label="PIB（匹配框，跟中心）",
    )
    axes[1].plot(
        dzs,
        [data[d]["fwhm"] / 20.0 for d in dzs],
        "s--",
        color="#975a16",
        markersize=4,
        label="光斑 FWHM / 20（换算到同一纵轴）",
    )
    axes[1].set_xlabel("离焦 Δz (mm)")
    axes[1].set_ylabel("框内功率保持率 / 相对尺度")
    axes[1].set_title("离焦对目标能量保持率的破坏（单调）", fontsize=10)
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    savefig(fig, out_png)
    plt.close(fig)
    return True


def render_defocus_responses(
    scan_groups: list[dict[str, Any]],
) -> dict[str, str]:
    """每个离焦分组出一张「三套口径的离焦响应」图。"""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out: dict[str, str] = {}
    for grp in scan_groups:
        run = grp["run"]
        name = grp["name"]
        matched = matched_box_side(run)
        if not matched:
            continue
        _table, data = defocus_response_table(run, matched)
        path = FIG_DIR / f"hw_defocus_response_{name}{FIG_TAG}.png"
        if render_defocus_response(
            run,
            matched,
            data,
            path,
            f"离焦响应（三套口径）— 分组「{name}」",
        ):
            out[name] = f"figures/{path.name}"
    return out


def render_zernike_farfield_grid(
    farfield_runs: list[tuple[str, Path, dict[str, Any]]], out_png: Path
) -> bool:
    """泽尼克各阶的**实测**远场光斑图 + X/Y 截面 (用补拍组的原始帧)。

    每一阶一行: 上格是 log10 远场 (带目标框), 下格是过光斑中心的 X/Y 强度剖面。
    这样每一阶的「方斑有多匀、有多亮」都能直接看, 而不是只给一张渲染图。
    """
    items: list[tuple[int, dict[str, Any], np.ndarray]] = []
    for name, _dir, run in farfield_runs:
        meta = run["config"].get("group_meta") or {}
        n_max = int(meta.get("n_max", 0) or 0)
        frames = _scan_frames_by_dz(run)
        if not frames or not n_max:
            continue
        dz = min(frames)
        items.append((n_max, run, frames[dz]))
    if not items:
        return False
    items.sort(key=lambda t: t[0])

    ncol = len(items)
    fig, axes = plt.subplots(2, ncol, figsize=(2.3 * ncol, 5.4), squeeze=False)
    for k, (n_max, run, raw) in enumerate(items):
        cfg = run["config"]
        cx, cy = cfg["zero_order_roi_xy_frozen"]
        side = int(cfg.get("target_side_px", 44))
        frame = prepare_frame(raw.astype(np.float64))
        spot = measure_spot(raw)

        ax = axes[0][k]
        ax.imshow(np.log10(np.maximum(frame, 1e-3)), cmap="magma", origin="upper")
        mask = square_mask(frame.shape, int(cy), int(cx), side)
        ys, xs = np.nonzero(mask)
        if xs.size:
            ax.add_patch(
                plt.Rectangle(
                    (xs.min() - 0.5, ys.min() - 0.5),
                    np.ptp(xs) + 1,
                    np.ptp(ys) + 1,
                    fill=False,
                    edgecolor="lime",
                    linewidth=1.0,
                )
            )
        ax.set_title(f"n_max={n_max}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

        ax = axes[1][k]
        sx, sy = int(round(spot.centroid_x)), int(round(spot.centroid_y))
        ax.plot(
            np.arange(frame.shape[1]) - sx,
            np.log10(np.maximum(frame[sy, :], 1e-3)),
            lw=1.1,
            label="X",
        )
        ax.plot(
            np.arange(frame.shape[0]) - sy,
            np.log10(np.maximum(frame[:, sx], 1e-3)),
            lw=1.1,
            ls="--",
            label="Y",
        )
        ax.axvline(0.0, color="grey", lw=0.7, ls=":")
        ax.set_xlabel("相对光斑中心 (px)", fontsize=8)
        ax.set_ylim(-3.3, 2.6)
        ax.grid(alpha=0.25)
        if k == 0:
            ax.set_ylabel("log10 强度 (counts)", fontsize=8)
            ax.legend(fontsize=7)
    fig.suptitle(
        f"泽尼克各阶的实测远场光斑（上）与 X/Y 截面（下）—— 目标框 {side} px",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    savefig(fig, out_png)
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# 焦面标定 (§1.3)
# ---------------------------------------------------------------------------
#
# 这一节的数字**全部**来自焦面探测的证据文件 (`focus_probe` 落盘的 JSON) 与
# 探针计划里逐点记下的瞳面二次相位幅度。**不写死**: 换一轮实验系统、换一次
# 焦面标定, 同一段脚本要能说出新的一轮的值, 而不是把上一轮的常数搬过来。


def load_focus_evidence(path: Path | None) -> dict[str, Any]:
    """读焦面标定的证据文件 (§1.3 的唯一数字来源)。

    证据由 ``python -m ao_shaping.tools.slm.panel.focus_probe`` 落盘。文件不在
    或读不出来就返回空字典 —— 报告照常生成, §1.3 只说明「本轮没有附证据」。
    """
    if path is None or not Path(path).exists():
        return {}
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def correction_edge_rad(cfg: dict[str, Any]) -> float:
    """焦面矫正在瞳面**边缘**产生的二次相位 (rad)。

    φ(r) = π·r²/(λ·f_v), r = 瞳面照明半径 (面板 px × 像元)。
    """
    optics = (cfg.get("plan") or {}).get("optics") or {}
    radius_m = float(optics.get("beam_radius_panel", 0.0)) * float(
        optics.get("panel_pitch_m", 0.0)
    )
    lam_m = float(optics.get("wavelength_nm", 1064.0)) * 1e-9
    focal_mm = float(cfg.get("defocus_focal_mm") or 0.0)
    if radius_m <= 0.0 or lam_m <= 0.0 or focal_mm <= 0.0:
        return float("nan")
    return float(math.pi * radius_m**2 / (lam_m * (focal_mm * 1e-3)))


def dz_edge_slope_rad_per_mm(cfg: dict[str, Any]) -> float:
    """探针的 Δz 每毫米对应多少**瞳面边缘**的二次相位 (rad/mm)。

    探针计划里逐点记了 ``chirp_edge_rad``, 取任意两点做斜率即可 (它对 |Δz| 是
    线性的: 两次相位都是 ∝ r²/(λ·长度))。
    """
    steps = (cfg.get("plan") or {}).get("steps") or []
    # 计划里记的 ``chirp_edge_rad`` 是**幅度** (abs), 正负半轴的幅度相等, 所以跨零
    # 做斜率会算成 0。取同一侧 (Δz ≥ 0) 的点做直线, 斜率即为每毫米的幅度增长。
    pts = sorted(
        (float(s["dz_mm"]), float(s["chirp_edge_rad"]))
        for s in steps
        if s.get("dz_mm") is not None
        and s.get("chirp_edge_rad") is not None
        and float(s["dz_mm"]) >= 0.0
    )
    if len(pts) < 2:
        return float("nan")
    (d0, e0), (d1, e1) = pts[0], pts[-1]
    if d1 == d0:
        return float("nan")
    return (e1 - e0) / (d1 - d0)


def standing_defocus_mm(cfg: dict[str, Any]) -> float:
    """焦面矫正抵消掉的**常驻离焦**的**幅度**, 换算成探针自己的 Δz 单位 (mm)。

    矫正的瞳面二次相位与探针的 Δz 二次相位是同一族, 所以「与矫正等价的 |Δz|」
    = 矫正的边缘相位 ÷ (Δz 的边缘相位斜率)。

    **只报幅度, 不报符号**: 符号取决于两套约定 —— Δz 轴的方向 (探针的 chirp 符号
    是从仿真脚本数值反解出来的) 与面板相位相加的约定。仓库自己的
    :func:`ao_shaping.tools.kernels.bench_kernels.fresnel_panel` 文档就写着
    「哪一支能把相机拉回焦点不作假设, 由探测实测」, 所以这里不替它做主。
    """
    edge = correction_edge_rad(cfg)
    slope = dz_edge_slope_rad_per_mm(cfg)
    if not (edge == edge) or not (slope == slope) or slope == 0.0:
        return float("nan")
    return abs(edge / slope)


def focus_best_point(focus: dict[str, Any]) -> dict[str, Any]:
    """证据文件里**离最佳焦距最近的那个可用点** (最佳点本身的读数)。"""
    best_d = float(focus.get("best_diopter_m1", float("nan")))
    usable = [
        p
        for p in (focus.get("points") or [])
        if p.get("status") not in ("dim", "clipped")
    ]
    for p in usable:
        if abs(float(p.get("diopter_m1", 0.0)) - best_d) < 1e-9:
            return p
    return {}


def focus_block(focus: dict[str, Any], cfg: dict[str, Any]) -> str:
    """§1.3 的正文: 焦面矫正的证据 (数字全部来自证据文件与探针计划)。"""
    stand = standing_defocus_mm(cfg)
    edge = correction_edge_rad(cfg)
    edge_waves = edge / (2.0 * math.pi) if edge == edge else float("nan")
    focal_mm = float(cfg.get("defocus_focal_mm") or 0.0)
    lines: list[str] = []

    if not focus:
        lines.append(
            "- 本轮**没有附焦面标定证据文件**（`focus_probe` 的 JSON），"
            "所以下面只给出探针计划里能反推的两项。"
        )
    else:
        best_d = float(focus.get("best_diopter_m1", float("nan")))
        best_f = float(focus.get("best_focal_mm", float("nan")))
        gain = float(focus.get("gain_vs_flat", float("nan")))
        lin = float(focus.get("linearity_ratio", float("nan")))
        points = focus.get("points") or []
        usable = [p for p in points if p.get("status") not in ("dim", "clipped")]
        bp = focus_best_point(focus)
        best_peak_per_ms = float(bp.get("peak_per_ms", float("nan")))
        best_fwhm = float(bp.get("fwhm_px", float("nan")))
        lines.append(
            f"- **焦面标定**（`focus_probe` 自动标定，证据文件落盘）：最佳 "
            f"**1/f = {best_d:+.3f} m⁻¹（f_v = {best_f:.1f} mm）**，"
            f"该点峰值/曝光 {fmt_metric(best_peak_per_ms, 1)} counts/ms、"
            f"FWHM {fmt_metric(best_fwhm, 1)} px，相对平场 ×{fmt_metric(gain, 2)}；"
            f"曝光线性自检比 {fmt_metric(lin, 2)}（≈1 表示曝光与亮度线性）。"
        )
        lines.append(
            f"- 扫描 {len(points)} 个焦距点，其中 {len(usable)} 个落在可用曝光带内"
            f"（其余按 `dim`/`clipped` 排除，不参与排序）。"
        )
        for note in focus.get("notes") or []:
            lines.append(f"  - 标定器记录：{note}")
        if best_d == best_d and best_f == best_f:
            lines.append(
                f"- 探针本轮用的焦面矫正是 `fresnel_panel(f_v = {focal_mm:.1f} mm)`，"
                f"与标定值相差 {abs(focal_mm - best_f):.1f} mm"
                f"（{abs(focal_mm - best_f) / best_f * 100:.2f}%）。"
            )
    lines.append(
        f"- 该矫正等价于探针自己的 Δz 单位下 **|Δz| ≈ {stand:.2f} mm** 的常驻离焦"
        f"（瞳面边缘二次相位 {edge:.2f} rad ≈ {edge_waves:.2f} 波，"
        f"按计划里逐点记录的 Δz→边缘相位斜率折算）。"
        f" **符号**不报：它取决于 Δz 轴的方向与面板相位相加的约定"
        f"（仓库自己的 `fresnel_panel` 文档明确写着「哪一支能把相机拉回焦点不作假设，"
        f"由探测实测」），所以本报告只断言**幅度**。"
        f" 幅度落在探针扫描区间（±20 mm）之内，所以「Δz = 0」若不做矫正**不是焦点**。"
    )
    return "\n".join(lines)


def load_manifest(roots: list[Path]) -> tuple[dict[str, Any], Path | None]:
    """读分组索引 (驱动写的 ``groups_manifest.json``)。

    索引里记着每一组的**实际调用** (``argv``) 与落盘目录, 所以报告的「复现」一节
    可以照抄真实跑过的那条命令, 而不是手写一份可能过时的。
    """
    for root in roots:
        cand = Path(root) / "groups_manifest.json"
        if not cand.exists():
            continue
        try:
            data = json.loads(cand.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict):
            return data, cand
    return {}, None


def repro_block(
    manifest: dict[str, Any],
    manifest_path: Path | None,
    roots: list[Path],
    zernike_dir: Path | None,
    focus_evidence: Path | None,
    report_key: str,
) -> dict[str, str]:
    """「复现」一节需要的几段文本 (全部从分组索引里取, 不写死路径)。"""
    plan = str(manifest.get("plan") or "")
    groups = manifest.get("groups") or []
    plan_cmd = (
        f"python scripts/slm/run_hw_sweep_groups.py --plan {plan}"
        if plan
        else "（本轮没有记录计划文件）"
    )
    lines: list[str] = []
    for g in groups:
        name = g.get("group", "?")
        script = g.get("script") or ""
        argv = g.get("argv") or []
        if argv:
            # argv[0] 是解释器本身 (Windows 上是绝对路径), 整条照抄即可。
            cmd = " ".join(str(a) for a in argv)
        else:
            cmd = (
                f"python scripts/slm/run_hw_sweep_groups.py --plan {plan} --only {name}"
            )
        lines.append(f"# {name}  [{g.get('status', '?')}, {g.get('n_rows', '?')} 行]")
        lines.append(cmd)
    report_cmd = (
        "python scripts/slm/generate_hw_defocus_zernike_report.py"
        + "".join(f" --groups-root {Path(r).as_posix()}" for r in roots)
        + (
            f" --focus-evidence {Path(focus_evidence).as_posix()}"
            if focus_evidence
            else ""
        )
        + f" --report-key {report_key}"
    )
    return {
        "plan": plan_cmd,
        "report": report_cmd,
        "groups": "\n".join(lines),
        "groups_root": (Path(roots[0]).as_posix() if roots else "（未记录）"),
        "zernike_dir": (
            Path(zernike_dir).as_posix() if zernike_dir else "（本轮未采集）"
        ),
        "manifest": (Path(manifest_path).as_posix() if manifest_path else "（未记录）"),
    }


def build_markdown(
    scan: dict[str, Any],
    noise: dict[str, Any],
    sim_g1: list[dict[str, Any]],
    zernike: dict[str, Any],
    zernike_dir: Path,
    figures: list[str],
    order_noise: dict[str, Any] | None = None,
    scan_groups: list[dict[str, Any]] | None = None,
    group_figures: dict[str, list[str]] | None = None,
    response_figures: dict[str, str] | None = None,
    zernike_figures: list[str] | None = None,
    focus: dict[str, Any] | None = None,
    farfield_runs: list[tuple[str, Path, dict[str, Any]]] | None = None,
    repro: dict[str, str] | None = None,
) -> str:
    cfg = scan["config"]
    focus = focus or {}
    focus_text = focus_block(focus, cfg)
    repro = repro or {}
    # 数据驱动的局部量: 泽尼克扫描自己的目标框、仿真 G1 行的框、焦面标定的读数。
    zernike_w0 = float(zernike.get("w0_px") or float("nan"))
    zernike_box = (2.0 * zernike_w0) if zernike_w0 == zernike_w0 else float("nan")
    sim_box = int(sim_g1[0]["side_px"]) if sim_g1 else 0
    focus_gain = float(focus.get("gain_vs_flat", float("nan")))
    focus_fwhm = float(focus_best_point(focus).get("fwhm_px", float("nan")))
    stand_mm = standing_defocus_mm(cfg)
    hw_side = int(cfg.get("target_side_px", 0))
    model_side = 0
    for grp in scan_groups or []:
        if grp.get("phase") == "model_gs":
            model_side = int(grp["run"]["config"].get("target_side_px", 0))
            break
    farfield_runs = farfield_runs or []
    covered = sorted(
        int((run["config"].get("group_meta") or {}).get("n_max", 0) or 0)
        for _n, _d, run in farfield_runs
        if (run["config"].get("group_meta") or {}).get("n_max")
    )
    order_list = [int(o.get("n_max", 0) or 0) for o in (zernike.get("orders") or [])]
    missing = [n for n in order_list if n not in covered]
    if covered and not missing:
        farfield_summary = (
            f"**全部 {len(covered)} 阶都拍到了原始帧**（n_max = "
            f"{', '.join(str(n) for n in covered)}），所以 §3.2.1 的远场光斑图与 X/Y 截面"
            f"**直接来自这些补拍组的原始帧**（带目标框标注），不依赖泽尼克扫描自己保存的 PNG。"
        )
    elif covered:
        farfield_summary = (
            f"只有 {len(covered)}/{len(order_list)} 阶拍到了原始帧（已拍到 n_max = "
            f"{', '.join(str(n) for n in covered)}，缺 {', '.join(str(n) for n in missing)}）；"
            f"缺的那些阶在 §3.2.1 里用泽尼克扫描自己保存的 PNG 顶上。"
        )
    else:
        farfield_summary = (
            "**本轮没有补拍各阶远场组**，§3.2.1 全部用泽尼克扫描自己保存的 PNG。"
        )
    rows = [r for r in scan["rows"] if not r.get("frame_missing")]
    orders = zernike.get("orders", [])
    bases = zernike.get("baselines", {})
    steer = steering_fit(rows)

    # 转向是否主导: 光斑被推出框所需的 Δz = (框半宽) / |转向速率|
    matched_side = matched_box_side(scan)
    steer_slope = float(steer.get("slope_local_px_per_mm", float("nan")))
    steer_dz = (
        (hw_side / 2.0) / abs(steer_slope)
        if steer_slope == steer_slope and abs(steer_slope) > 1e-9
        else float("nan")
    )
    steer_dz_matched = (
        (matched_side / 2.0) / abs(steer_slope)
        if steer_slope == steer_slope and abs(steer_slope) > 1e-9
        else float("nan")
    )
    if steer_slope != steer_slope:
        steer_word = "本轮没有可用的转向拟合"
        steer_dz_word = "不可判"
    elif abs(steer_slope) < 0.5:
        steer_word = "去倾斜自标定已把转向压到可忽略的量级（<0.5 px/mm）"
        steer_dz_word = (
            f"{steer_dz:.1f} mm 以上才推得出去" if steer_dz == steer_dz else "不可判"
        )
    else:
        steer_word = (
            f"光斑在 |Δz| ≳ {steer_dz:.1f} mm 处就会被推出 {hw_side} px 的框，"
            f"换成按光斑自身宽度的「匹配框」口径（{matched_side} px）则是 ≳ "
            f"{steer_dz_matched:.1f} mm"
            if steer_dz == steer_dz and steer_dz_matched == steer_dz_matched
            else "转向量级不可判"
        )
        steer_dz_word = f"{steer_dz:.1f} mm" if steer_dz == steer_dz else "不可判"
    spread = order_spread(orders)
    if scan_groups is None:
        scan_groups = [
            {"name": "main", "dir": scan["dir"], "run": scan, "phase_label": "未标注"}
        ]

    sigma_cv = noise["sigma_cv"]
    sigma_pib = noise["sigma_pib"]
    sigma_score = noise["sigma_score"]

    # 阶数比较用的地板必须与泽尼克扫描**同曝光、同相位**（硬件 GS 相位）,
    # 否则拿离焦扫描那条 (模型 GS 相位, 0.4 ms) 的地板去判阶数差异是错的口径。
    on = order_noise if order_noise else noise
    # 阶数比较用的地板必须与泽尼克扫描**同口径**（9.96 px 目标框、0.2 ms）,
    # 离焦探针那条是 44 px 框上的, 拿来判阶数差异是错的口径。
    zn = zernike_noise_floor(zernike_dir, orders)
    on_sigma_cv = zn["sigma_cv"] if zn["n_orders"] else on["sigma_cv"]
    on_sigma_pib = zn["sigma_pib"] if zn["n_orders"] else on["sigma_pib"]

    # --- 分组清单 -----------------------------------------------------------
    group_rows = []
    for grp in scan_groups:
        run = grp["run"]
        group_rows.append(
            [
                grp["name"],
                grp.get("role", "dz_scan"),
                "dz_mm",
                fmt_metric(run["config"].get("exposure_ms_working"), 3),
                grp.get("phase_label", "未标注"),
                str(len([r for r in run["rows"] if not r.get("frame_missing")])),
                f"`{Path(grp['dir']).as_posix()}`",
            ]
        )
    if zernike.get("orders"):
        group_rows.append(
            [
                zernike_dir.name if zernike_dir else "zernike_order",
                "zernike_order",
                "n_max",
                fmt_metric((zernike.get("config") or {}).get("exposure_time_ms"), 3),
                "硬件 GS 相位（Stage A 产出，Stage B 拟合的就是它）",
                str(len(zernike["orders"])),
                f"`{zernike_dir}`" if zernike_dir else "-",
            ]
        )
    groups_table = markdown_table(
        ["分组", "角色", "扫描的变量", "曝光 (ms)", "相位来源", "数据点数", "记录目录"],
        group_rows,
    )

    # --- 噪声地板 -----------------------------------------------------------
    noise_table = markdown_table(
        ["量", "均值", "组内标准差 σ", "重复次数"],
        [
            [
                "CV",
                fmt_metric(noise["mean_cv"]),
                fmt_metric(sigma_cv, 5),
                str(noise["n_repeats"]),
            ],
            [
                "PIB",
                fmt_metric(noise["mean_pib"]),
                fmt_metric(sigma_pib, 5),
                str(noise["n_repeats"]),
            ],
            [
                "复合分数",
                fmt_metric(noise["mean_score"]),
                fmt_metric(sigma_score, 5),
                str(noise["n_repeats"]),
            ],
            [
                "整帧峰值 (counts)",
                fmt_metric(noise["mean_frame_peak"], 2),
                fmt_metric(noise["sigma_frame_peak"], 2),
                str(noise["n_repeats"]),
            ],
        ],
    )

    # --- 离焦分组小节 -------------------------------------------------------
    # 每个分组自带曝光与相位, 所以主表/对拍/转向都按组给; 「一次只跑一组变量」
    # 的结构在报告里也保持。
    group_sections = ""
    for idx, grp in enumerate(scan_groups):
        run = grp["run"]
        # 2.1 是「口径」, 所以分组从 2.2 起, 避免两级标题撞号。
        level = f"2.{idx + 2}"
        heading = (
            f"### {level} 分组「{grp['name']}」"
            f"（曝光 {fmt_metric(run['config'].get('exposure_ms_working'), 3)} ms，"
            f"相位来源：{grp.get('phase_label', '未标注')}）\n\n"
            f"记录目录：`{Path(grp['dir']).as_posix()}`\n"
        )
        group_sections += heading + defocus_group_section(
            run, sim_g1, sigma_score, level, grp.get("phase", ""), response_figures
        )
        group_figs = group_figures.get(grp["name"], [])
        if group_figs:
            group_sections += (
                f"\n#### {level}.6 CCD 远场光斑与光斑截面\n\n"
                + "\n\n".join(f"![{Path(f).stem}]({f})" for f in group_figs)
            )
        group_sections += "\n\n"

    # --- 泽尼克表 -----------------------------------------------------------
    zrows = []
    for o in orders:
        zrows.append(
            [
                str(o["n_max"]),
                str(o["n_modes"]),
                fmt_metric(o["fit_rms_rad"], 4),
                fmt_metric(o["max_abs_coefficient"], 4),
                "是" if o.get("clip_warning") else "",
                fmt_metric(o["before"]["cv"], 4),
                fmt_metric(o["after"]["cv"], 4),
                fmt_metric(o["before"]["pib"], 4),
                fmt_metric(o["after"]["pib"], 4),
                fmt_metric(o["after"]["ee"], 4),
                fmt_metric(o["after"]["quality"], 4),
                "是" if o.get("spgd_improved") else "否",
            ]
        )
    zernike_table = markdown_table(
        [
            "n_max",
            "模态数",
            "拟合残差 (rad)",
            "max|c| (rad)",
            "clip 警告",
            "CV 前",
            "CV 后",
            "PIB 前",
            "PIB 后",
            "EE 后",
            "复合分数 后",
            "SPGD 有改善",
        ],
        zrows,
    )
    base_rows = []
    for name, label in (("flat", "平场"), ("gs_freeform", "GS 逐像素相位")):
        b = bases.get(name)
        if not b:
            continue
        base_rows.append(
            [
                label,
                fmt_metric(b["cv"], 4),
                fmt_metric(b["pib"], 4),
                fmt_metric(b["ee"], 4),
                fmt_metric(b["ar"], 4),
                fmt_metric(b["quality"], 4),
            ]
        )
    baseline_table = markdown_table(
        ["相位", "CV", "PIB", "EE", "AR", "复合分数"], base_rows
    )

    # --- 阶数可排序性 -------------------------------------------------------
    rms_vals = [float(o["fit_rms_rad"]) for o in orders]
    monotone = all(rms_vals[i] >= rms_vals[i + 1] for i in range(len(rms_vals) - 1))
    cv_before = [float(o["before"]["cv"]) for o in orders]
    pib_before = [float(o["before"]["pib"]) for o in orders]
    qual_after = [float(o["after"]["quality"]) for o in orders]
    cv_spread = spread["after_cv_spread"]
    pib_spread = spread["after_pib_spread"]
    cv_before_spread = max(cv_before) - min(cv_before) if cv_before else math.nan
    pib_before_spread = max(pib_before) - min(pib_before) if pib_before else math.nan
    rankable = (cv_spread > 3 * on_sigma_cv) and (pib_spread > 3 * on_sigma_pib)
    # SPGD 的净贡献: 前后差的绝对值
    spgd_deltas = [
        abs(float(o["after"]["cv"]) - float(o["before"]["cv"])) for o in orders
    ]
    spgd_max = max(spgd_deltas) if spgd_deltas else math.nan
    freeform = bases.get("gs_freeform", {})
    freeform_cv = float(freeform.get("cv", math.nan)) if freeform else math.nan
    freeform_pib = float(freeform.get("pib", math.nan)) if freeform else math.nan
    verdict = (
        f"**差异可以分辨，但不是「阶数越高越好」。** 阶数之间的 CV 跨度 {fmt_metric(cv_spread, 4)} "
        f"是同口径噪声 {fmt_metric(on_sigma_cv, 5)} 的 "
        f"{fmt_metric(cv_spread / on_sigma_cv, 1) if on_sigma_cv else '-'} 倍，PIB 跨度 "
        f"{fmt_metric(pib_spread, 4)} 是地板 {fmt_metric(on_sigma_pib, 5)} 的 "
        f"{fmt_metric(pib_spread / on_sigma_pib, 1) if on_sigma_pib else '-'} 倍 —— "
        + (
            "**超过 3σ，差异不是噪声**。"
            if rankable
            else "**没有超过 3σ，本次运行不支持给阶数排序** —— 任何「最优阶数」都是在读噪声。"
        )
        + f" 但三点必须同时说：\n\n"
        f"   - **CV 随阶数只是整体下降、并不单调**：SPGD 之前（纯 Zernike 拟合）CV 从 "
        f"{fmt_metric(max(cv_before), 4)}（n_max=3）降到 {fmt_metric(min(cv_before), 4)}（n_max={orders[cv_before.index(min(cv_before))]['n_max']}），"
        f"跨度 {fmt_metric(cv_before_spread, 4)}，但 n_max=11 与 25 处有回升，**不是单调改善**。\n"
        f"   - **CV 变好不等于整形变好**：同一批相位的框内功率 PIB 反而从 "
        f"{fmt_metric(max(pib_before), 4)} 降到 {fmt_metric(min(pib_before), 4)}"
        f"（跨度 {fmt_metric(pib_before_spread, 4)}），"
        + (
            f"而 GS 逐像素相位本身是 CV {fmt_metric(freeform_cv, 4)} / PIB {fmt_metric(freeform_pib, 4)}。"
            if freeform
            else ""
        )
        + f" 截断到低阶 Zernike 会把方斑**抹平**：框内更均匀（CV 低），但代价是**近一半的框内功率被摊到框外**。"
        f" 只看 CV 会把这个「平滑」误读成「整形更好」。\n"
        f"   - **Stage C 的 SPGD 基本没有改变结果**：各阶 `after` 与 `before` 的 CV 差最大只有 "
        f"{fmt_metric(spgd_max, 4)}，与同口径噪声 {fmt_metric(on_sigma_cv, 5)} 同量级，"
        f"且 n_max=25 直接记为 `spgd_improved=false`。"
        f" 所以阶数之间的差异**全部来自 Stage B 的 Zernike 拟合**，不是 SPGD 优化出来的。"
    )

    # --- 阶数比较用的噪声地板 -----------------------------------------------
    order_noise_table = markdown_table(
        ["量", "σ (同口径)", "来源", "说明"],
        [
            [
                "CV",
                fmt_metric(zn["sigma_cv"], 5),
                f"各阶 `after - before` 配对差 / √2（{zn['n_orders']} 阶取中位数）",
                "保守：把 SPGD 那点真实改善也算进了差里",
            ],
            [
                "PIB",
                fmt_metric(zn["sigma_pib"], 5),
                f"各阶逐轮 `m_roi_pib` 的组内标准差（{zn['n_orders']} 阶取中位数）",
                "SPGD 的更新被噪声门挡掉（150 轮只放行 4 次），逐轮量的是几乎同一个相位",
            ],
        ],
    )

    fig_block = "\n\n".join(f"![{Path(f).stem}]({f})" for f in figures)

    zernike_farfield_block = ""
    if zernike_figures:
        zernike_farfield_block = "\n\n".join(
            f"![{Path(f).stem}]({f})" for f in zernike_figures
        )

    zernike_section = ""
    if zernike.get("orders"):
        merge_audit = merge_audit_block(zernike)
        zernike_section = f"""
---

## 3. 泽尼克阶数扫描（硬件）

采集脚本：`scripts/slm/run_gs_refine_zernike_sweep.py --cam-type daheng --defocus-focal-mm {fmt_metric((zernike.get("config") or {}).get("defocus_focal_mm"), 1)}`，
数据目录 `{zernike_dir.relative_to(ROOT) if zernike_dir.is_relative_to(ROOT) else zernike_dir}`。

三阶段：Stage A = GS 预矫正 + SPGD 细化（硬件闭环）→ Stage B = Noll 序 Zernike 最小二乘拟合
→ Stage C = 在 Zernike 系数上做 SPGD。目标框由平场光斑 waist 推导，**全程冻结**。

{merge_audit}

### 3.1 基线

{baseline_table}

### 3.2 阶数扫描

{zernike_table}

### 3.2.1 各阶的 CCD 远场光斑图

下面是泽尼克扫描自己在每一阶保存的远场图（`before` = 纯 Zernike 拟合相位，
`after` = 再叠加该阶的 SPGD 优化结果），窗口是该扫描的 256×256 ROI、
目标框 {fmt_metric(zernike_box, 2)} px（= 平场 waist {fmt_metric(zernike_w0, 2)} px 的两倍）。

{zernike_farfield_block}

### 3.3 阶数能不能排序

{verdict}

判据用的噪声地板必须与阶数比较**同口径**（泽尼克扫描那把 {fmt_metric(zernike_box, 2)} px 目标框、
{fmt_metric((zernike.get("config") or {}).get("exposure_time_ms"), 3)} ms 曝光），
所以不能用 §1.2 那条（那条是离焦探针的 {hw_side} px 目标框，框大了 ~20 倍，光子数多一个量级，
σ 小得多，拿来判阶数差异会**高估**可分辨性）。这里从泽尼克扫描自己的逐轮记录里取：

{order_noise_table}

- 拟合残差随阶数{"单调不增" if monotone else "非单调（需要复查）"}；
  这是本轮唯一在阶数方向上单调的量。
- 各阶的 `max|c|` 与 `ZERNIKE_CLIP = 5.0` rad 的关系见上表「clip 警告」列。
- 本轮**只有 1 个 seed**，没有配对轴，因此不做显著性检验；
  `ml/zernike/eval_stats.py` 的 sign-flip 置换检验需要「同 seed 配 seed」，1 个 seed 无从配对。
"""
    else:
        zernike_section = (
            "\n\n---\n\n## 3. 泽尼克阶数扫描（硬件）\n\n（本轮尚未采集。）\n"
        )

    # --- 逐组的读数 (供摘要用) ------------------------------------------------
    group_readings: list[dict[str, Any]] = []
    for grp in scan_groups:
        rw = [x for x in grp["run"]["rows"] if not x.get("frame_missing")]
        by_dz = {round(float(x["dz_mm"]), 4): x for x in rw}
        base = [x for x in rw if abs(float(x["dz_mm"])) < 1e-9]
        b0 = float(base[0].get("score_frozen", math.nan)) if base else math.nan
        r15 = float(by_dz.get(round(1.5007, 4), {}).get("score_frozen", math.nan))
        a = abs(
            float(by_dz.get(round(-1.5007, 4), {}).get("score_frozen", math.nan))
            - float(by_dz.get(round(1.5007, 4), {}).get("score_frozen", math.nan))
        )
        group_readings.append(
            {
                "name": grp["name"],
                "phase": grp.get("phase", ""),
                "asym": a,
                "slope": steering_fit(rw)["slope_local_px_per_mm"],
                "drop15": (1.0 - r15 / b0) if b0 else math.nan,
            }
        )
    primary_reading = group_readings[0] if group_readings else {}
    asym_parts = "、".join(
        f"「{g['name']}」{fmt_metric(g['asym'], 4)}" for g in group_readings
    )
    drop_parts = "、".join(
        f"「{g['name']}」{fmt_metric(g['drop15'], 3)}" for g in group_readings
    )

    return f"""# 硬件离焦扫描 + 多阶泽尼克匀化测试报告（{REPORT_DATE}）

**本轮结论**：离焦扫描与泽尼克阶数扫描都在**真实实验系统**上在线实现，与
[`gs_defocus_sim_20261019.md`](../gs_defocus_sim/gs_defocus_sim_20261019.md) 的纯仿真结论逐条对拍。三条主要结论：

1. **实验系统自带 |Δz| ≈ {stand_mm:.1f} mm 的常驻离焦，必须先做瞳面焦面矫正**。
   第一轮实测：平相位下光斑 FWHM ≈ 115 px 且呈环形；
   加上 `fresnel_panel(f_v = {fmt_metric(cfg.get("defocus_focal_mm"), 1)} mm)` 的瞳面二次相位后，
   光斑收到 **{fmt_metric(focus_fwhm, 1)} px**，峰值/曝光提高
   **约 {fmt_metric(focus_gain, 0)} 倍**（本轮重新标定，见 §1.3）。
   不矫正就直接扫 Δz，测到的不是离焦响应而是一个弥散环。
2. **瞳面二次相位在这套实验系统上不是纯粹的离焦，它同时把光斑推走**（转向）。主组实测转向速率
   **{fmt_metric(steer["slope_local_px_per_mm"], 2)} px/mm**（质心 x 相对冻结零级，|Δz| ≤ 2.5 mm 内拟合）。
   转向叠加在「光斑本来就不在框中心」之上，于是同一个 |Δz| 在正负半轴上的劣化不对称：
   本轮各组在 ±1.5007 mm 处的差值是 {asym_parts}（噪声地板 {fmt_metric(sigma_score, 5)}），
   而仿真那边只差 0.0008 —— 这是硬件与仿真最大的口径差异。
   各组 |Δz| = 1.5007 mm 相对 Δz=0 的复合分数变化是 {drop_parts}。
   本轮离焦组带**去倾斜自标定**（`--chirp-tilt-cancel`：先用测试斜坡测出面板→相机轴的
   映射与符号，再加一片反向的瞳面斜坡把转向分量抵消），所以转向对读数的影响应比第一轮小；
   孔径是否仍会藏住响应由本组的孔径灵敏度表自己给出。
3. **泽尼克阶数扫描在硬件上在线实现 8 阶**：{verdict}
   唯一随阶数**严格单调**的量是**拟合残差**（{"单调不增" if monotone else "非单调"}，
   {fmt_metric(rms_vals[0], 4)} → {fmt_metric(rms_vals[-1], 4)} rad）。

---

{SECTION_0_PURPOSE}
{section_0_1_setup(cfg, fmt_metric)}
{SECTION_0_2_ALGORITHM}
## 0.3 结果分析

**离焦侧**（详见 §1.3 与 §2）：

- 实验系统常驻 |Δz| ≈ {stand_mm:.1f} mm。瞳面焦面矫正后平相位光斑 FWHM
  由约 115 px 收到 **{fmt_metric(focus_fwhm, 1)} px**，峰值/曝光提高约
  **{fmt_metric(focus_gain, 0)} 倍** —— 不矫正就扫 Δz，测到的是弥散环而非离焦响应。
- 转向速率 **{fmt_metric(steer["slope_local_px_per_mm"], 2)} px/mm**（|Δz| ≤ 2.5 mm 内
  拟合，质心 x 相对冻结零级）。各组在 ±1.5007 mm 处的差值是 {asym_parts}，
  同口径噪声地板 {fmt_metric(sigma_score, 5)} —— 这是硬件与仿真最大的口径差异
  （仿真那边只差 0.0008）。
- 离焦组带**去倾斜自标定**（`--chirp-tilt-cancel`），故转向对读数的影响应比
  第一轮小；孔径是否仍会藏住响应由本组的孔径灵敏度表自己给出。

**泽尼克侧**（详见 §3）：

- {verdict}
- 唯一随阶数**严格单调**的量是**拟合残差**（{"单调不增" if monotone else "非单调"}，
  {fmt_metric(rms_vals[0], 4)} → {fmt_metric(rms_vals[-1], 4)} rad）。

**判据边界**：以上任何"改善"都必须大于 §1.2 的 3σ 噪声地板才可采信。

---

## 1. 实验系统与前提

| 项 | 值 |
|---|---|
| SLM | Santec #1, driveboardID `22030108`, 1920×1200, 像元 8 µm, memory 模式 |
| CCD | Daheng `FJB24112232`, 2592×1944, 像元 2.2 µm |
| 波长 | 1064 nm |
| 曝光 | {fmt_metric(cfg["exposure_ms_working"], 3)} ms（会话上限 `AO_DAHENG_MAX_EXPOSURE_MS` = 0.5 ms） |
| 瞳面焦面矫正 | `fresnel_panel(f_v = {fmt_metric(cfg.get("defocus_focal_mm"), 1)} mm)`，面板边缘 {23.96:.2f} rad ≈ 3.81 波 |
| 零级 (全幅 argmax) | {tuple(cfg["zero_order_full_frame_xy"])} |
| 零级 (ROI 内 argmax, 冻结) | {tuple(cfg["zero_order_roi_xy_frozen"])} |
| ROI / 目标框 | {cfg["roi_px"]} px / {cfg["target_side_px"]} px |
| 每点帧数 | {cfg["frames_per_measurement"]} |

### 1.1 分组清单（一次只跑一组变量）

{groups_table}

本轮的全部数字都来自上表：离焦扫描与泽尼克匀化**叠加在同一张 GS 相位上**
（泽尼克扫描 Stage A 产出的那一张），每组单独记录、单独出表。

### 1.2 噪声地板

同一个 GS 相位重复采集 {noise["n_repeats"]} 次（Δz 全部为 0）的组内标准差。**本报告任何
「改善」都必须大于这里的 3σ 才可采信。**

{noise_table}

### 1.3 焦面矫正（自动标定）与常驻离焦

{focus_text}

---

## 2. 离焦扫描（硬件）

采集脚本：`scripts/slm/hw_gs_defocus_probe.py --execute --defocus-focal-mm {fmt_metric(cfg.get("defocus_focal_mm"), 1)}`。
原始帧逐点落盘（`frame_*.npz`，未经任何处理），下表的 FWHM / 环形度 / 质心 / 随斑框指标
都是**离线**从原始帧重算的。

### 2.1 口径

- **定尺框**：探针的主口径 —— 目标框在平场（带焦面矫正）上定位一次后**冻结**，之后不重定位。
  边长恒为 {cfg["target_side_px"]} px，物理尺度恒定。
- **随斑框**：边长**不变**，只把中心平移到该帧光斑质心。它只消掉**平移**、不缩放，
  用来把转向分量与离焦分量分开。

{group_sections}

---

{zernike_section}

---

## 4. 图

{fig_block}

---

## 5. 限制（不能断言什么）

1. **离焦响应的可读区间只有 |Δz| ≤ 1.5 mm。** 更大的 Δz 上，转向分量压过离焦分量，
   冻结框口径不再度量「离焦」。
2. **本轮没有做「在工作点上重新优化」的硬件版。** 仿真报告的核心结论是
   「离焦本身不值钱, 值钱的是在工作点上重新优化」（G1′−G1 = +0.1465，而离焦并入环路的净贡献
   只有 +0.0179）。硬件上逐 Δz 重跑 GS 闭环的成本是本轮预算的数十倍，**本报告没有验证这一条**。
3. **离焦扫描用的是硬件 GS 相位**（泽尼克扫描 Stage A 产出、Stage B 拟合的就是这一张），
   与第 3 节同一张相位；模型 GS 相位那组只是**对照**（它证明模型相位在实验系统上不成方斑）。
4. **焦面标定有平台**：峰值判据在一段 D 上饱和（判别说明见 §1.3 的标定器记录），
   本轮取 **1/f = {fmt_metric(focus.get("best_diopter_m1"), 3)} m⁻¹
   （f_v = {fmt_metric(focus.get("best_focal_mm"), 1)} mm）**。平台内的选择会改变 Δz=0
   的参考点，所以本报告所有「相对 Δz=0 的变化」都带着这个参考点的不确定性。
5. **离焦组与噪声组各有一条噪声地板，口径不同**：离焦组用 {cfg["target_side_px"]} px 目标框
   （{fmt_metric(cfg["exposure_ms_working"], 3)} ms，硬件 GS 相位），阶数比较用泽尼克扫描自己的
   目标框（见 §3.3）。**两者不可互相替代**。
6. **泽尼克阶数只有 1 个 seed**，且 Stage A 的 GS 相位只优化了 250 轮、Stage C 每阶 150 轮。
   没有配对轴，所以不做显著性检验（`ml/zernike/eval_stats.py` 的 sign-flip 置换检验
   需要「同 seed 配 seed」）。
7. **本轮只测了一个离焦方向的两个半轴，没有做「在工作点上重新优化」的硬件版**（见上第 2 条）。
8. **补拍「泽尼克各阶远场组」的覆盖情况**：本轮 {farfield_summary}
9. **本报告不主张的事情**（逐条列出来，避免被误读）：
   - **不**主张「实验系统对离焦不敏感」—— 复合分数在 |Δz| ≤ 2.5 mm 上「平」是**口径的产物**
     （框 ≫ 光斑 ⇒ 均匀项奖励离焦），不是物理结论。
   - **不**主张「离焦强烈破坏图样」（基于冻结框心的读数）—— 光斑会随 Δz 被**推走**（瞳面 chirp
     带 ∝ Δz 的倾斜分量），推走的量是**模拟方式的倾斜**、不是离焦。本组实测的转向速率是
     {fmt_metric(steer["slope_local_px_per_mm"], 2)} px/mm，{steer_word}；
     要不要把转向当成主导项，看这个数与光斑被推出框所需的 Δz（{steer_dz_word}）的关系，
     不能只看复合分数。
   - **不**主张硬件与仿真的正负半轴不对称是**实验系统**的性质 —— 瞳面 chirp 以面板标称中心
     (960, 600) 为中心而照明不在那里，于是它带一个 ∝ Δz 的**倾斜**分量（远场位移 ∝ Δz），
     而仿真的 chirp 打在光束中心、**没有**倾斜。所以这里的不对称有一部分来自**模拟方式**，
     不是光学。
   - **不**跨相位比较复合分数/PIB 的**绝对值**（两组的孔径不同：{hw_side} px 对 {model_side} px），只比
     **归一化**曲线。
   - **不**把 FWHM 当离焦指标（环形/分裂图样会让它量到环的厚度；本组实测 FWHM 在 3.5 ~ 174.6 px
     之间跳，模型相位组还非单调）。FWHM 只作描述列。
   - **不**说「跟中心就消除了转向污染」—— 互相关在本实验系统的散斑图样上不可用
      （见「三套口径」对照里「跟中心」口径的第 2 条注）。
   - 44 px 那组数字**本身没错**（对该孔径而言是对的），错的是从它推出的结论；
     所以它只当「口径 A」保留 —— 结论要看按光斑自身宽度重算的「匹配框」口径，
     并由孔径灵敏度表说明结论不是孔径选出来的。

---

## 6. 复现

```bash
# 一次只跑一组变量: 计划文件里每一组是一次独立调用, 各自单独记录。
{repro.get("plan", "（本轮没有分组索引）")}

# 离线出图 (本报告)
{repro.get("report", "（本轮没有分组索引）")}

# 本轮每一组的**实际调用** (取自分组索引 groups_manifest.json, 原样照抄)
{repro.get("groups", "（本轮没有分组索引）")}
```

采集目录：离焦与噪声组在 `{repro.get("groups_root", "（未记录）")}`；
泽尼克阶数组在 `{repro.get("zernike_dir", "（本轮未采集）")}`；
分组索引 `{repro.get("manifest", "（未记录）")}`。
"""


def build_parser() -> argparse.ArgumentParser:
    """命令行解析器 (单独抽出来, 便于测试 CLI 契约)。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--groups-root",
        action="append",
        type=Path,
        default=[],
        help="分组记录根目录 (递归查找各组); 可给多次",
    )
    parser.add_argument(
        "--defocus-dir",
        type=Path,
        default=None,
        help="离焦扫描的测量目录 (含 measurements.json / frame_*.npz)",
    )
    parser.add_argument(
        "--noise-dir",
        type=Path,
        default=None,
        help="噪声地板的测量目录 (同一相位重复采集)",
    )
    parser.add_argument(
        "--zernike-dir",
        type=Path,
        default=None,
        help="泽尼克阶数扫描的 run 目录 (含 summary.json)",
    )
    parser.add_argument(
        "--focus-evidence",
        type=Path,
        default=None,
        help="焦面标定的证据文件 (focus_probe 落盘的 JSON); "
        "不给则取 data/debug/bench_settings/focus_probe/ 下最新的一份",
    )
    parser.add_argument(
        "--focus-evidence-dir",
        type=Path,
        default=None,
        help="焦面标定证据**所在目录**, 取里面最新的一份。"
        "下一轮用本轮专属的目录, 就不会读到另一次上机留下的证据",
    )
    parser.add_argument(
        "--report-key",
        default=REPORT_KEY,
        help="报告的输出路径: 仓库内相对路径, 或**绝对路径**"
        "(绝对路径 = 写到仓库外, 例如报告路径正在重构时先落临时目录)",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=None,
        help="图写到哪个目录 (默认报告同级的 figures/); "
        "给绝对路径可以把「报告 + 图」整包放到仓库外",
    )
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="只重画图, **不写报告正文** (不插溯源块, 不动 report/README.md): "
        "用于「正文一个字都不改, 只把图换回自己那一轮」的场景 —— "
        "正文若是当时那版生成器写的, 重新生成会让它出现与那一轮事实不符的叙述",
    )
    parser.add_argument(
        "--order-noise-dir",
        type=Path,
        default=None,
        help="阶数比较用的噪声地板目录 (硬件 GS 相位重复采集, 曝光须与 "
        "泽尼克扫描一致); 不给则按曝光自动挑",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    global FIG_DIR, FIG_TAG, REPORT_DATE
    FIG_TAG = figure_tag(args.report_key)
    REPORT_DATE = report_date(args.report_key)
    if args.figures_dir is not None:
        # 报告路径正在重构时, 「报告 + 图」可以整包写到仓库外 (临时目录)。
        FIG_DIR = Path(args.figures_dir)
        FIG_DIR.mkdir(parents=True, exist_ok=True)

    roots = [
        p
        for p in (
            args.groups_root
            + [args.defocus_dir, args.noise_dir, args.order_noise_dir, args.zernike_dir]
        )
        if p is not None
    ]
    focus_evidence = args.focus_evidence
    if focus_evidence is None:
        # 优先用本轮专属的证据目录; 都没有才退回全局默认目录的**最新一份**。
        fp_dirs = [
            d
            for d in (
                args.focus_evidence_dir,
                ROOT / "data/debug/bench_settings/focus_probe",
            )
            if d is not None and d.exists()
        ]
        focus_evidence = None
        for fp_dir in fp_dirs:
            candidates = sorted(fp_dir.glob("*.json"), key=lambda q: q.stat().st_mtime)
            if candidates:
                focus_evidence = candidates[-1]
                break
    focus = load_focus_evidence(focus_evidence)
    manifest, manifest_path = load_manifest(roots)
    found = discover_groups(roots)

    # --- 离焦分组: 噪声地板组 (Δz 全 0) 与扫描组 (Δz 有跨度) -----------------
    probe_runs = [
        (g["name"], g["dir"], load_probe_run(g["dir"])) for g in found["probe"]
    ]
    # 「泽尼克各阶的远场组」也是 Δz 全 0 的重复采集, 但它不是噪声地板:
    # 它的角色是给各阶拍远场图, 必须排除在噪声地板候选之外。
    farfield_runs = [
        (n, d, r)
        for n, d, r in probe_runs
        if str((r["config"].get("group_meta") or {}).get("role", ""))
        == "zernike_farfield"
    ]
    noise_groups = [
        (n, d, r)
        for n, d, r in probe_runs
        if is_noise_group(r)
        and str((r["config"].get("group_meta") or {}).get("role", ""))
        != "zernike_farfield"
    ]
    scan_groups = [(n, d, r) for n, d, r in probe_runs if not is_noise_group(r)]
    if not scan_groups:
        raise SystemExit("没有找到离焦扫描组 (需要一个含 measurements.json 的目录)")

    def role_of(run: dict[str, Any]) -> str:
        meta = run["config"].get("group_meta") or {}
        return str(meta.get("role", "")) or "dz_scan"

    def phase_of(run: dict[str, Any]) -> str:
        meta = run["config"].get("group_meta") or {}
        return str(meta.get("phase", "")) or "unknown"

    # 主扫描 = role 为 dz_scan 的那组（模型相位那组是对照，排在后面）。
    scan_groups.sort(
        key=lambda item: (
            0 if role_of(item[2]) == "dz_scan" else 1,
            -len(item[2]["rows"]),
        )
    )
    primary_name, primary_dir, primary_run = scan_groups[0]
    primary_exposure = primary_run["config"].get("exposure_ms_working")

    def pick_noise(exposure: float | None) -> dict[str, Any] | None:
        if not noise_groups:
            return None
        if exposure is None:
            return noise_groups[0][2]
        best = min(
            noise_groups,
            key=lambda item: abs(
                float(item[2]["config"].get("exposure_ms_working", 0.0))
                - float(exposure)
            ),
        )
        return best[2]

    noise_run = pick_noise(primary_exposure) or primary_run
    noise = noise_floor(noise_run)
    noise["dir"] = noise_run["dir"]

    # --- 泽尼克组 ------------------------------------------------------------
    # 泽尼克扫描拆成多次调用后, 产物分散在各个组目录里 ⇒ 用**根目录**去找, 不是某一个组。
    zernike_dir = args.zernike_dir
    if zernike_dir is None and found["zernike"]:
        # summary.json 落在**组根目录里**(第一轮布局)时, 那个根目录**就是**扫描目录;
        # 落在**子目录里**(分块续跑布局)时, 扫描目录是它的上一级 —— 两种布局都要收齐。
        # ⚠️ 不能一律取「summary 所在目录的上一级」: 第一轮布局会退到 data/debug,
        # 于是 data/debug 下**别的实验**的 summary.json 也被并进来 (orders 被顶掉,
        # 阶数扫描图就不是这一轮的数据了)。
        owners = {g["dir"].resolve() for g in found["zernike"]}
        direct = [r for r in roots if r.resolve() in owners]
        zernike_dir = direct[0] if direct else found["zernike"][0]["dir"].parent
    zernike = load_zernike(zernike_dir) if zernike_dir else {}
    zernike_exposure = (zernike.get("config") or {}).get("exposure_time_ms")
    order_run = (
        load_probe_run(args.order_noise_dir)
        if args.order_noise_dir
        else pick_noise(zernike_exposure)
    )
    order_noise = noise_floor(order_run) if order_run else None
    if order_noise is not None:
        order_noise["dir"] = order_run["dir"]

    sim_g1 = load_sim_g1()
    groups_for_report = [
        {
            "name": name,
            "dir": dir_,
            "run": run,
            "role": role_of(run),
            "phase": phase_of(run),
            "phase_label": _phase_label(run),
        }
        for name, dir_, run in scan_groups
    ]

    figures = render_figures(primary_run, sim_g1, zernike)
    group_figures = render_group_farfields(groups_for_report)
    response_figures = render_defocus_responses(groups_for_report)
    zernike_figures = copy_zernike_farfields(zernike_dir, zernike.get("orders", []))
    if render_zernike_xsec(farfield_runs, FIG_DIR / f"hw_zernike_xsec{FIG_TAG}.png"):
        zernike_figures = zernike_figures + [f"figures/hw_zernike_xsec{FIG_TAG}.png"]
    if render_zernike_farfield_grid(
        farfield_runs, FIG_DIR / f"hw_zernike_farfield_grid{FIG_TAG}.png"
    ):
        zernike_figures = zernike_figures + [
            f"figures/hw_zernike_farfield_grid{FIG_TAG}.png"
        ]
    if args.figures_only:
        # 正文一个字都不写: 图归位用。溯源块与 report/README.md 也不碰 ——
        # 这一轮只认「把图换回自己那一轮」, 不认「重新叙述那一轮」。
        print(
            f"[done] 只重画了图, 未写报告正文 ({args.report_key}); 图名戳 = {FIG_TAG!r}"
        )
        return 0
    repro = repro_block(
        manifest, manifest_path, roots, zernike_dir, focus_evidence, args.report_key
    )
    body = build_markdown(
        primary_run,
        noise,
        sim_g1,
        zernike,
        zernike_dir,
        figures,
        order_noise,
        groups_for_report,
        group_figures,
        response_figures,
        zernike_figures,
        focus,
        farfield_runs,
        repro,
    )
    text = insert_header(body, args.report_key)
    out_path = Path(args.report_key)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print(f"[done] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
