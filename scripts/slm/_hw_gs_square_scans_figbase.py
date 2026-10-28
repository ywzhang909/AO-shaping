"""GS 方扫报告的**取帧与拼图规划**。

出图之前的准备工作：在产物目录里定位到每一格的帧、读帧均值、按边长算裁剪
框、把若干格排成拼图版式。

与 :mod:`_hw_gs_square_scans_figures` 的分界是：**这里不碰 matplotlib**。
"挑哪些帧、怎么排"是数据问题，"怎么画"是绘图问题，分开之后换绘图后端
不需要动取帧逻辑。
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence
import numpy as np

import matplotlib

matplotlib.use("Agg")  # 必须在 import matplotlib.pyplot 之前

import matplotlib.pyplot as plt

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

from _common import savefig


from scripts.slm._hw_gs_square_scans_fmt import (
    _rel,
)
from scripts.slm._hw_gs_square_scans_model import (
    CROP_FACTOR,
    CellPick,
    FrameRef,
    MOSAICS,
    MosaicSpec,
    PARAM_NPZ_CSV_ATOL,
    PATHS,
    PEAK_MATCH_ATOL,
    RetestRun,
    Run,
    SATURATION_COUNTS,
    SIDE_PX_NPZ_CSV_ATOL,
    Stage,
    VARIABLE,
)

def setup_matplotlib() -> None:
    """全局绘图约定: CJK 字体 + 负号不显示成方块。"""
    from matplotlib import font_manager

    installed = {f.name for f in font_manager.fontManager.ttflist}
    preferred = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
    family = [name for name in preferred if name in installed] or ["DejaVu Sans"]
    matplotlib.rcParams["font.sans-serif"] = family
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["axes.unicode_minus"] = False

def locate_frames(stage_dir: Path, row: dict[str, str], headers: Sequence[str]) -> FrameRef:
    """按**实际布局**定位该行的帧。

    两种布局都要支持, 判据是 CSV 里有没有 ``config`` 列 (原子布局每配置一个子目录):

    * 原子布局: ``<stage>/<config>/frame_*.npz``
    * 旧整段布局: ``<stage>/frame_<index:03d>.npz``

    找不到就返回 ``layout="missing"`` —— 调用方负责"跳过该格 + 记缺帧", 不静默少画。
    """
    if "config" in headers and row.get("config"):
        config_dir = stage_dir / str(row["config"])
        if config_dir.is_dir():
            paths = sorted(config_dir.glob("frame_*.npz"))
            if paths:
                return FrameRef(paths=tuple(paths), layout="atomic")
        return FrameRef(paths=(), layout="missing")
    if "index" in headers and row.get("index"):
        path = stage_dir / f"frame_{int(float(row['index'])):03d}.npz"
        if path.exists():
            return FrameRef(paths=(path,), layout="legacy")
        return FrameRef(paths=(), layout="missing")
    return FrameRef(paths=(), layout="missing")

def load_frame_mean(paths: Sequence[Path]) -> np.ndarray:
    """把一格的若干 npz **逐个**读入并求均值 (一次只在内存里留一个整帧)。"""
    total: np.ndarray | None = None
    for path in paths:
        with np.load(path) as payload:
            if "frame" not in payload:
                raise KeyError(f"{path} 里没有 'frame' 数组 (实际键: {list(payload.keys())})")
            frame = np.asarray(payload["frame"], dtype=np.float32)
        if total is None:
            total = np.zeros_like(frame, dtype=np.float32)
        elif frame.shape != total.shape:
            raise ValueError(f"{path} 的帧形状 {frame.shape} 与首帧 {total.shape} 不一致")
        total += frame
    if total is None:
        raise ValueError("没有可读的帧")
    total /= float(len(paths))
    return total

def frame_params(path: Path) -> dict[str, Any]:
    """读 npz 里那个 0 维 ``params`` (JSON 字符串) —— 用于帧 ↔ CSV 交叉校验。"""
    with np.load(path) as payload:
        if "params" not in payload:
            return {}
        raw = payload["params"]
    text = str(raw.item()) if getattr(raw, "shape", ()) == () else str(raw)
    try:
        return dict(json.loads(text))
    except (TypeError, ValueError):
        return {}

def _check_frame_params(stage: str, cfg: CellPick, ref: FrameRef) -> None:
    """npz 自带的 ``params`` 必须与该行 CSV 一致 —— 否则拼图会张冠李戴。"""
    if not ref.paths:
        return
    params = frame_params(ref.paths[0])
    if not params:
        return
    checks = (("cv", "cv"), ("pib", "pib"), ("score", "score"),
              ("frame_peak", "frame_peak"))
    for key, column in checks:
        if key not in params:
            continue
        if abs(float(params[key]) - float(cfg.row[column])) > PARAM_NPZ_CSV_ATOL:
            raise ValueError(
                f"{stage}/{cfg.slug}: npz params 的 {key}={params[key]} 与 CSV 的 "
                f"{column}={cfg.row[column]} 不一致 —— 帧与 CSV 行对不上"
            )
    if "side_px" in params and stage == "sides":
        if abs(float(params["side_px"]) - float(cfg.row["side_px"])) > SIDE_PX_NPZ_CSV_ATOL:
            raise ValueError(
                f"{stage}/{cfg.slug}: npz params 的 side_px={params['side_px']} 与 CSV 不一致"
            )

def batch_tag(run: Run) -> str:
    """图名里的批次后缀 = run-root 目录名 (换 run-root 不会互相覆盖)。"""
    return run.root.name

def _cell_title(pick: CellPick, stage: Stage) -> str:
    """两行标题: 第一行是配置, 第二行是指标 (单行会压到相邻格)。"""
    row = pick.row
    peak = float(row["frame_peak"])
    sat = "  饱和" if peak >= SATURATION_COUNTS else ""
    if pick.stage == "sides":
        head = f"side={float(row['side_px']):.0f} px"
        if pick.role == "GS 参照":
            head = "GS 参照（未截断）  " + head
        body = (f"CV={float(row['cv']):.3f}  PIB={float(row['pib']):.3f}  "
                f"score={float(row['score']):.3f}  peak={peak:.0f} counts")
    elif pick.stage == "zernike":
        head = f"n_max={float(row['n_max']):.0f}"
        body = (f"CV={float(row['cv']):.3f}  PIB={float(row['pib']):.3f}  "
                f"score={float(row['score']):.3f}  peak={peak:.0f} counts")
    else:
        head = f"P={float(row['blaze_period_px']):.0f} px"
        body = (f"预测位移={float(row['predicted_shift_px']):.1f} px  "
                f"实测 y={float(row['measured_shift_y_px']):.1f} px  "
                f"CV={float(row['cv']):.3f}  PIB={float(row['pib']):.3f}")
    return f"{head}{sat}\n{body}"

def _crop_box(row: dict[str, str], shape: tuple[int, int], factor: float
              ) -> tuple[int, int, int, int, bool]:
    """以 ``box_center`` 为中心、边长 ``box_side × factor`` 的裁剪框 (已夹到画幅内)。"""
    height, width = shape
    side = float(row["box_side_px"]) * factor
    cx, cy = float(row["box_center_x"]), float(row["box_center_y"])
    half = side / 2.0
    x0 = int(round(cx - half))
    x1 = int(round(cx + half))
    y0 = int(round(cy - half))
    y1 = int(round(cy + half))
    clipped = x0 < 0 or y0 < 0 or x1 > width or y1 > height
    x0, x1 = max(0, x0), min(width, x1)
    y0, y1 = max(0, y0), min(height, y1)
    return x0, x1, y0, y1, clipped

def _annotate_cell(ax, pick: CellPick, *, crop: tuple[int, int, int, int] | None,
                   zero_order: tuple[float, float] | None = None,
                   frame_centre: tuple[float, float] | None = None) -> None:
    """叠加: 目标框 (虚线) / 预测框心 (十字) / 实测质心 (圆点) / 出画提示。"""
    row = pick.row
    side = float(row["box_side_px"])
    cx, cy = float(row["box_center_x"]), float(row["box_center_y"])
    if crop is None:
        x_off = y_off = 0.0
    else:
        x_off, y_off = float(crop[0]), float(crop[2])
    left, top = cx - side / 2 - x_off, cy - side / 2 - y_off
    ax.add_patch(plt.Rectangle((left, top), side, side, fill=False, ec="#00e5ff",
                               ls="--", lw=1.1, label="目标框"))
    ax.plot([cx - x_off], [cy - y_off], marker="+", color="#ffd400", ms=13, mew=1.6,
            ls="none", label="预测框心")
    ax.plot([float(row["box_centroid_x"]) - x_off], [float(row["box_centroid_y"]) - y_off],
            marker="o", mfc="none", mec="#ff3b30", ms=9, mew=1.6, ls="none",
            label="实测质心")
    if zero_order is not None:
        ax.plot([zero_order[0]], [zero_order[1]], marker="*", color="#ffffff", ms=15,
                ls="none", label="0 级 (argmax)")
    if frame_centre is not None:
        ax.plot([frame_centre[0]], [frame_centre[1]], marker="x", color="#7fff00",
                ms=9, mew=1.4, ls="none", label="画面中心")
    clipped = float(row.get("box_clipped_px", 0.0) or 0.0)
    if clipped > 0:
        ax.text(0.02, 0.02, f"框出画 {clipped:.0f} px", transform=ax.transAxes,
                fontsize=7.5, color="#ff3b30", va="bottom", ha="left",
                bbox=dict(fc="black", ec="none", alpha=0.45, pad=1.5))

def plan_mosaic(run: Run, spec: MosaicSpec, picks: Sequence[CellPick]) -> dict[str, Any]:
    """**不读像素**地规划一张拼图: 每格的来源、帧数、裁剪框、拉伸参数。

    峰值取自 CSV 的 ``frame_peak`` (绘制时再与帧的实际最大值逐格比对), 所以
    ``--check`` 可以只靠这一步就判断 ``frames_selection.json`` 是否过期, 不必重画。
    """
    stage_dirs = {name: run.root / name for name in run.stages}
    records: list[dict[str, Any]] = []
    for pick in picks:
        stage = run.stages[pick.stage]
        ref = locate_frames(stage_dirs[pick.stage], pick.row, stage.headers)
        planned = int(stage.run_config.get("frames_per_measurement", 0) or 0)
        shape = tuple(int(v) for v in stage.run_config.get("frame_shape_hw", ()))
        peak = float(pick.row["frame_peak"])
        record: dict[str, Any] = {
            "mosaic": spec.key,
            "stage": pick.stage,
            "config": pick.slug,
            "value": {VARIABLE[pick.stage]: pick.value},
            "role": pick.role,
            "note": pick.note,
            "layout": ref.layout,
            "sources": [{"path": _rel(path), "sha256": _sha256(path)} for path in ref.paths],
            "frames_present": len(ref.paths),
            "frames_planned": planned,
            "missing": not ref.paths,
            "peak": peak,
            "saturated": peak >= SATURATION_COUNTS,
            "cv": float(pick.row["cv"]),
            "pib": float(pick.row["pib"]),
            "score": float(pick.row["score"]),
            "box_clipped_px": float(pick.row.get("box_clipped_px", 0.0) or 0.0),
            "frame_shape_hw": list(shape) if shape else None,
            "norm": {"vmin": 1.0, "vmax": max(peak, 1.0 + 1e-6)},
            "crop": None,
        }
        if spec.crop and shape:
            x0, x1, y0, y1, clipped = _crop_box(pick.row, shape, CROP_FACTOR)
            record["crop"] = {"x0": x0, "x1": x1, "y0": y0, "y1": y1,
                              "factor": CROP_FACTOR, "clipped_by_frame": clipped}
        if ref.paths:
            # 读一遍像素: 峰值必须等于 CSV 的 frame_peak (帧 ↔ CSV 的硬校验), 顺便
            # 记下**方框内**的中位/99 分位电平 —— 大边长格在对数显示下几乎全黑,
            # 有没有这两个数决定了读者能不能分清"数据太暗"和"画错了"。
            image = load_frame_mean(ref.paths)
            peak_px = float(image.max())
            if abs(peak_px - peak) > PEAK_MATCH_ATOL:
                raise ValueError(
                    f"{pick.stage}/{pick.slug}: 帧最大值 {peak_px:.6g} 与 CSV frame_peak "
                    f"{peak:.6g} 不一致 —— 帧与 CSV 行对不上"
                )
            height, width = image.shape
            side = float(pick.row["box_side_px"])
            cx, cy = float(pick.row["box_center_x"]), float(pick.row["box_center_y"])
            by0 = max(0, int(round(cy - side / 2)))
            by1 = min(height, int(round(cy + side / 2)))
            bx0 = max(0, int(round(cx - side / 2)))
            bx1 = min(width, int(round(cx + side / 2)))
            box = image[by0:by1, bx0:bx1]
            record["box_median"] = float(np.median(box))
            record["box_p99"] = float(np.percentile(box, 99))
            record["box_pixels"] = int(box.size)
            record["peak"] = peak_px
            del image, box
        else:
            record["box_median"] = None
            record["box_p99"] = None
            record["box_pixels"] = 0
        records.append(record)
    return {"mosaic": spec.key, "title": spec.title,
            "figure": f"{spec.stem}_{batch_tag(run)}.png", "cells": records}

def draw_mosaic(run: Run, spec: MosaicSpec, picks: Sequence[CellPick],
                plan: dict[str, Any], out_path: Path) -> bool:
    """按 ``plan`` 画图: 一格一格读帧、求均值、裁剪、随即释放。"""
    from matplotlib.colors import LogNorm

    setup_matplotlib()
    stage_dirs = {name: run.root / name for name in run.stages}
    cells = [(pick, record) for pick, record in zip(picks, plan["cells"])
             if not record["missing"]]
    if not cells:
        return False

    nrows = math.ceil(len(cells) / spec.ncols)
    fig, axes = plt.subplots(nrows, spec.ncols,
                             figsize=(spec.cell_inch * spec.ncols,
                                      spec.cell_inch * nrows + 0.5),
                             squeeze=False)
    flat = axes.ravel()
    for ax in flat[len(cells):]:
        ax.axis("off")
    for ax, (pick, record) in zip(flat, cells):
        ref = locate_frames(stage_dirs[pick.stage], pick.row, run.stages[pick.stage].headers)
        image = load_frame_mean(ref.paths)
        if abs(float(image.max()) - float(record["peak"])) > PEAK_MATCH_ATOL:
            raise ValueError(
                f"{pick.stage}/{pick.slug}: 帧最大值 {float(image.max()):.6g} 与 CSV "
                f"frame_peak {record['peak']:.6g} 不一致 —— 帧与 CSV 行对不上"
            )
        crop = record["crop"]
        if crop is not None:
            if image.shape != tuple(record["frame_shape_hw"]):
                raise ValueError(
                    f"{pick.stage}/{pick.slug}: 帧形状 {image.shape} 与 run_config 的 "
                    f"{record['frame_shape_hw']} 不一致")
            shown = image[crop["y0"]:crop["y1"], crop["x0"]:crop["x1"]]
        else:
            shown = image
        ax.imshow(shown, cmap="inferno",
                  norm=LogNorm(vmin=record["norm"]["vmin"], vmax=record["norm"]["vmax"]),
                  interpolation="nearest")
        zero_order = None
        frame_centre = None
        if crop is None:
            peak_index = np.unravel_index(int(np.argmax(shown)), shown.shape)
            zero_order = (float(peak_index[1]), float(peak_index[0]))
            height, width = shown.shape
            frame_centre = ((width - 1) / 2.0, (height - 1) / 2.0)
        _annotate_cell(ax, pick, crop=None if crop is None else
                       (crop["x0"], crop["x1"], crop["y0"], crop["y1"]),
                       zero_order=zero_order, frame_centre=frame_centre)
        title = _cell_title(pick, run.stages[pick.stage])
        if crop is None and zero_order is not None:
            frozen = run.zero_order_xy
            title += (f"\n0 级(argmax)=({zero_order[0]:.0f}, {zero_order[1]:.0f})  "
                      f"冻结 0 级=({frozen[0]:.0f}, {frozen[1]:.0f})  "
                      f"画面中心=({frame_centre[0]:.0f}, {frame_centre[1]:.0f})")
        ax.set_title(title, fontsize=7.5, linespacing=1.25)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.98, 0.02, f"×{record['frames_present']} 帧均值", transform=ax.transAxes,
                fontsize=7.5, color="white", ha="right", va="bottom",
                bbox=dict(fc="black", ec="none", alpha=0.45, pad=1.5))
        del image, shown
    fig.suptitle(spec.title, fontsize=11)
    fig.tight_layout(h_pad=2.2, w_pad=0.8)
    savefig(fig, out_path)
    return True

def plan_farfield(run: Run, selection: dict[str, list[CellPick]]) -> dict[str, Any]:
    """4 张拼图的规划 + ``frames_selection.json`` 载荷 (不读像素)。"""
    payload: dict[str, Any] = {
        "batch": batch_tag(run),
        "run_root": _rel(run.root),
        "display": {
            "crop_factor": CROP_FACTOR,
            "norm": "LogNorm(vmin=1, vmax=该格峰值)",
            "dpi": 150,
            "interpolation": "nearest",
            "cmap": "inferno",
        },
        "mosaics": [],
    }
    for spec in MOSAICS:
        picks = selection.get(spec.key) or []
        if not picks:
            continue
        payload["mosaics"].append(plan_mosaic(run, spec, picks))
    return payload

def render_farfield(run: Run, selection: dict[str, list[CellPick]],
                    payload: dict[str, Any]) -> list[Path]:
    """按已规划好的 ``payload`` 画 4 张拼图。"""
    PATHS.fig_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for spec in MOSAICS:
        picks = selection.get(spec.key) or []
        plan = next((m for m in payload["mosaics"] if m["mosaic"] == spec.key), None)
        if not picks or plan is None:
            continue
        out_path = PATHS.fig_dir / plan["figure"]
        if draw_mosaic(run, spec, picks, plan, out_path):
            paths.append(out_path)
    return paths

def write_frames_selection(payload: dict[str, Any], target: Path) -> Path:
    """``frames_selection.json`` (排序键、无时间戳 ⇒ 逐字节可复现)。"""
    target.parent.mkdir(parents=True, exist_ok=True)
    path = target / "frames_selection.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    return path

def residual_figure_name(atomic: RetestRun) -> str:
    """fig12 的文件名 —— 正文、`--check` 与 artifacts 三处共用同一份拼装。"""
    return f"fig12_zernike_phase_residual_{batch_tag(atomic)}.png"

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
