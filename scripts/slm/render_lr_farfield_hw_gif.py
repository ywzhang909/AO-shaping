"""把 ``hw_lr_farfield_sweep_20261127`` 的实测采集归档渲染成 GIF / 总览 / 指标表。

输入布局（由采集脚本写出，本脚本只读）::

    <run>/run_config.json          # params / lrs / iters / capture_every
    <run>/journal.jsonl            # 每帧一行: lr, iter, exposure_ms, peak, saturated,
                                   #            counts, centroid, fwhm, loss, flat
    <run>/flat/iter_XXXX.npz       # 平场参考帧（本脚本只做存在性说明，不做暗减）
    <run>/lr_0p02/iter_0000.npz    # 键: frame / phase_model / exposure_ms / lr / iter / loss
    <run>/lr_0p05/...              # 目录名 = lr_<数值，小数点换 p>

输出::

    <out>.gif                  # 每个 GIF 帧 = 一个迭代步；帧内按 lr 分行同时显示（4 行）
    <out>_contact_sheet.png    # 行 = lr，列 = 抽样迭代的静态总览
    <out>_metrics.csv          # journal.jsonl 汇总（peak / 饱和 / counts / fwhm / loss）

缺帧（某 lr 某迭代的 npz 不存在）画灰底 ``missing``，不静默跳过。

显示口径
--------
* ``--window`` 以该 lr **首帧**质心为中心裁剪，之后所有迭代固定同一窗口，避免光斑
  抖动导致的窗口漂移；窗口越界时整体平移进画面，帧比窗口小时补零。
* ``--log``（默认开）显示 ``log10(1+I)``；``--normalize global``（默认）让整组共用
  一个灰度上限（可跨 lr 比亮度），``frame`` 为逐帧峰值归一。
* 本脚本 **不做暗减**，显示的是归档里的原始 CCD 帧；页脚明确标注这一点。

纯离线：只读 ``.npz`` / ``.json`` / ``.jsonl``，不 import 任何设备驱动、不打开硬件。

用法::

    .venv\\Scripts\\python.exe scripts/slm/render_lr_farfield_hw_gif.py \\
        --run-dir data/debug/slm_transfer/hw_lr_farfield_20261127/run_xxxx \\
        --out report/shape_trials/hw_lr_farfield/run_xxxx.gif

    .venv\\Scripts\\python.exe scripts/slm/render_lr_farfield_hw_gif.py --self-test
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from PIL import Image, ImageDraw, ImageFont

#: 标题/页脚的中文字体（Pillow 直接吃 ttc 路径）；缺失时回落到 ASCII 字体并切英文文案。
CJK_FONT = Path(r"C:\Windows\Fonts\msyh.ttc")
ASCII_FONT_CANDIDATES = (
    Path(r"C:\Windows\Fonts\consola.ttf"),
    Path(r"C:\Windows\Fonts\arial.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
)

#: GIF 每帧的调色板颜色数（多行网格以灰度为主，64 色足够且体积可控）。
GIF_COLORS = 64
#: 全局归一化的灰度上限分位（避开极少数热像素）。
GLOBAL_VMAX_QUANTILE = 0.995

BG = (24, 26, 30)
FG = (232, 232, 232)
MUTED = (150, 155, 163)
BORDER = (102, 108, 118)
MISSING_FILL = (58, 62, 70)


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JournalRecord:
    """``journal.jsonl`` 的一行（字段缺失时为 ``None``）。"""

    lr: float
    iter: int
    exposure_ms: float | None
    peak: float | None
    saturated: bool
    counts: float | None
    fwhm: float | None
    loss: float | None
    flat: bool


@dataclass(frozen=True)
class CellData:
    """一个 ``(lr, iter)`` 的实测远场帧。"""

    lr: float
    iter: int
    frame: np.ndarray
    exposure_ms: float | None
    loss: float | None
    source: Path


@dataclass(frozen=True)
class RunData:
    """一次采集归档的只读视图。"""

    run_dir: Path
    config: dict[str, Any]
    lrs: tuple[float, ...]
    iters: tuple[int, ...]
    cells: dict[tuple[str, int], CellData]
    records: dict[tuple[str, int], JournalRecord]
    warnings: tuple[str, ...]

    @property
    def name(self) -> str:
        return self.run_dir.name


@dataclass(frozen=True)
class RenderOptions:
    """渲染口径。"""

    window: int = 256
    log_scale: bool = True
    normalize: str = "global"
    duration_ms: int = 380
    every: int = 1
    cell_px: int = 180
    sheet_cell_px: int = 150

    def __post_init__(self) -> None:
        if self.window <= 0:
            raise ValueError(f"--window 必须为正: {self.window}")
        if self.every < 1:
            raise ValueError(f"--every 必须 >= 1: {self.every}")
        if self.normalize not in {"frame", "global"}:
            raise ValueError(f"--normalize 只能是 frame/global: {self.normalize}")
        if self.duration_ms <= 0:
            raise ValueError(f"--duration-ms 必须为正: {self.duration_ms}")


@dataclass(frozen=True)
class RenderResult:
    """一次渲染的产物与统计。"""

    gif_path: Path
    contact_sheet_path: Path | None
    metrics_csv_path: Path | None
    gif_frames: int
    missing_cells: int
    lrs: tuple[float, ...]
    iters: tuple[int, ...]


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------


def _lr_key(lr: float) -> str:
    """浮点 lr 的稳定字典键（避免 0.02 / 0.020000001 分裂）。"""
    return f"{lr:g}"


def _parse_lr_dirname(name: str) -> float | None:
    """``lr_0p02`` -> ``0.02``；不符合命名返回 ``None``。"""
    if not name.startswith("lr_"):
        return None
    body = name[3:]
    try:
        return float(body.replace("p", "."))
    except ValueError:
        return None


def _sort_key(pair: tuple[str, int]) -> tuple[float, int]:
    lr_key, it = pair
    return (float(lr_key), it)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("读取 {} 失败: {}", path, exc)
        return None
    return data if isinstance(data, dict) else None


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def _frame_from_npz(path: Path) -> tuple[np.ndarray, float | None, float | None]:
    """读一帧 npz；只读 ``frame`` / ``exposure_ms`` / ``loss``，不碰设备。"""
    with np.load(path, allow_pickle=False) as data:
        frame = np.asarray(data["frame"])
        if frame.ndim == 3:
            frame = frame[0] if frame.shape[0] == 1 else frame[..., 0]
        exposure = _as_float(data["exposure_ms"]) if "exposure_ms" in data else None
        loss = _as_float(data["loss"]) if "loss" in data else None
    if frame.ndim != 2:
        raise ValueError(f"frame 维度应为 2，实际 {frame.shape}: {path}")
    return frame.astype(np.float64, copy=False), exposure, loss


def _parse_iters(raw: Any) -> list[int]:
    """``run_config.json`` 的 ``iters`` 可为次数（int）或显式列表。"""
    if raw is None:
        return []
    if isinstance(raw, bool):
        return []
    if isinstance(raw, (int, float)):
        return list(range(max(0, int(raw))))
    if isinstance(raw, list):
        out: list[int] = []
        for item in raw:
            value = _as_float(item)
            if value is not None:
                out.append(int(value))
        return sorted(set(out))
    return []


def _load_journal(path: Path) -> tuple[dict[tuple[str, int], JournalRecord], list[str]]:
    """逐行读 jsonl；坏行只警告，不中断。"""
    records: dict[tuple[str, int], JournalRecord] = {}
    warnings: list[str] = []
    if not path.is_file():
        return records, [f"缺少 journal.jsonl: {path}"]
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                warnings.append(f"journal 第 {lineno} 行不是 JSON: {exc}")
                continue
            if not isinstance(row, dict):
                warnings.append(f"journal 第 {lineno} 行不是对象")
                continue
            lr = _as_float(row.get("lr"))
            it = _as_float(row.get("iter"))
            if lr is None or it is None:
                warnings.append(f"journal 第 {lineno} 行缺 lr/iter")
                continue
            rec = JournalRecord(
                lr=lr,
                iter=int(it),
                exposure_ms=_as_float(row.get("exposure_ms")),
                peak=_as_float(row.get("peak")),
                saturated=_as_bool(row.get("saturated")),
                counts=_as_float(row.get("counts")),
                fwhm=_as_float(row.get("fwhm")),
                loss=_as_float(row.get("loss")),
                flat=_as_bool(row.get("flat")),
            )
            records[(_lr_key(lr), rec.iter)] = rec
    return records, warnings


def load_run(run_dir: Path) -> RunData:
    """读取一次采集归档（``run_config`` + 各 lr 目录 + journal）。"""
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run 目录不存在: {run_dir}")
    warnings: list[str] = []
    config = _load_json(run_dir / "run_config.json")
    if config is None:
        config = {}
        warnings.append("缺少或无法读取 run_config.json，lr/迭代从目录与 npz 推断")

    cells: dict[tuple[str, int], CellData] = {}
    for lr_dir in sorted(run_dir.iterdir()):
        if not lr_dir.is_dir():
            continue
        lr = _parse_lr_dirname(lr_dir.name)
        if lr is None:
            continue
        for npz_path in sorted(lr_dir.glob("iter_*.npz")):
            try:
                it = int(npz_path.stem.split("_", 1)[1])
            except (IndexError, ValueError):
                warnings.append(f"无法从文件名解析迭代号: {npz_path}")
                continue
            try:
                frame, exposure, loss = _frame_from_npz(npz_path)
            except (OSError, ValueError, KeyError) as exc:
                warnings.append(f"读取帧失败 {npz_path}: {exc}")
                continue
            cells[(_lr_key(lr), it)] = CellData(
                lr=lr,
                iter=it,
                frame=frame,
                exposure_ms=exposure,
                loss=loss,
                source=npz_path,
            )

    config_lrs = [_as_float(v) for v in config.get("lrs", [])] if isinstance(config.get("lrs"), list) else []
    lrs = sorted({value for value in config_lrs if value is not None} | {cell.lr for cell in cells.values()})
    if not lrs:
        raise RuntimeError(f"{run_dir} 下没有 lr_*/iter_*.npz，也没有配置 lrs")

    iters = set(_parse_iters(config.get("iters")))
    iters |= {it for _lr, it in cells}
    if not iters and config.get("iters") is None:
        warnings.append("run_config 与目录都没有迭代号")
    ordered_iters = sorted(iters)

    records, journal_warnings = _load_journal(run_dir / "journal.jsonl")
    warnings.extend(journal_warnings)
    return RunData(
        run_dir=run_dir,
        config=config,
        lrs=tuple(lrs),
        iters=tuple(ordered_iters),
        cells=cells,
        records=records,
        warnings=tuple(warnings),
    )


# ---------------------------------------------------------------------------
# 图像处理：质心、窗口、灰度映射
# ---------------------------------------------------------------------------


def _spot_center(frame: np.ndarray) -> tuple[float, float]:
    """阈值以上强度加权质心；信噪比不足时退回最亮点。返回 ``(x, y)`` = (列, 行)。"""
    arr = np.asarray(frame, dtype=np.float64)
    h, w = arr.shape
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    noise = 1.4826 * mad
    threshold = median + max(6.0 * noise, 1e-9)
    weights = np.where(arr > threshold, arr - median, 0.0)
    total = float(weights.sum())
    if total <= 0.0 or not math.isfinite(total):
        peak = np.unravel_index(int(np.argmax(arr)), arr.shape)
        return float(peak[1]), float(peak[0])
    xs = np.arange(w, dtype=np.float64)
    ys = np.arange(h, dtype=np.float64)
    cx = float((weights.sum(axis=0) * xs).sum() / total)
    cy = float((weights.sum(axis=1) * ys).sum() / total)
    return cx, cy


def _crop_axis(length: int, center: float, size: int) -> tuple[int, int, int, int]:
    """返回 ``(src_start, src_stop, dst_start, dst_stop)``；大帧平移进画面，小帧居中补零。"""
    if length >= size:
        start = int(round(center - size / 2.0))
        start = min(max(start, 0), length - size)
        return start, start + size, 0, size
    pad_before = (size - length) // 2
    return 0, length, pad_before, size - length - pad_before


def _crop_window(frame: np.ndarray, center: tuple[float, float], size: int) -> np.ndarray:
    """按固定尺寸裁剪/补零，保证每个 lr 的窗口跨迭代恒定。"""
    h, w = frame.shape
    x0, x1, dx0, dx1 = _crop_axis(w, center[0], size)
    y0, y1, dy0, dy1 = _crop_axis(h, center[1], size)
    out = np.zeros((size, size), dtype=np.float64)
    out[dy0:dy1, dx0:dx1] = frame[y0:y1, x0:x1]
    return out


def _prepare_display(frame: np.ndarray, center: tuple[float, float], options: RenderOptions) -> np.ndarray:
    """裁剪 + 可选 log10(1+I)；返回用于跨帧统一归一化的浮点图。"""
    cropped = _crop_window(frame, center, options.window)
    if options.log_scale:
        return np.log10(1.0 + np.maximum(cropped, 0.0))
    return np.maximum(cropped, 0.0)


def _pooled_vmax(arrays: list[np.ndarray]) -> float:
    if not arrays:
        return 1.0
    pooled = np.concatenate([np.asarray(arr).ravel()[::7] for arr in arrays])
    if pooled.size == 0:
        return 1.0
    value = float(np.quantile(pooled, GLOBAL_VMAX_QUANTILE))
    return value if value > 0.0 else 1.0


def _to_uint8(values: np.ndarray, vmax: float) -> np.ndarray:
    if vmax <= 0.0 or not math.isfinite(vmax):
        return np.zeros(values.shape, dtype=np.uint8)
    scaled = np.clip(values / vmax, 0.0, 1.0) * 255.0
    return scaled.astype(np.uint8)


# ---------------------------------------------------------------------------
# 字体与文字
# ---------------------------------------------------------------------------


def _pick_font(candidates: tuple[Path, ...]) -> Path | None:
    for path in candidates:
        if path.is_file():
            return path
    return None


class _Fonts:
    """中文字体优先；找不到 CJK 字体时退回 ASCII 文案，避免出豆腐块。"""

    def __init__(self) -> None:
        cjk_path = _pick_font((CJK_FONT,))
        self.cjk = cjk_path is not None
        path = cjk_path or _pick_font(ASCII_FONT_CANDIDATES)
        if path is None:
            raise RuntimeError(f"找不到可用字体: 既没有 {CJK_FONT} 也没有 {ASCII_FONT_CANDIDATES}")
        self.path = path
        self.title = ImageFont.truetype(str(path), 22)
        self.label = ImageFont.truetype(str(path), 17)
        self.note = ImageFont.truetype(str(path), 13)
        self.cell = ImageFont.truetype(str(path), 12)

    def t(self, cjk: str, ascii_text: str) -> str:
        return cjk if self.cjk else ascii_text


def _wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    """按像素宽度逐字符折行（中文无空格，按字符切即可）。"""
    if not text:
        return [""]
    lines: list[str] = []
    current = ""
    for char in text:
        candidate = current + char
        if current and font.getlength(candidate) > max_width:
            lines.append(current)
            current = char
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def _fmt_optional(value: float | None, fmt: str = "{:.3f}") -> str:
    if value is None or not math.isfinite(value):
        return "-"
    return fmt.format(value)


# ---------------------------------------------------------------------------
# 网格合成
# ---------------------------------------------------------------------------


def _draw_cell(
    canvas: Image.Image,
    box: tuple[int, int, int, int],
    values: np.ndarray | None,
    vmax: float,
    fonts: _Fonts,
    caption_left: str,
    caption_right: str,
) -> None:
    """画一个 cell：可用帧按灰度贴图，缺帧画灰底 ``missing``。"""
    x0, y0, x1, y1 = box
    width, height = x1 - x0, y1 - y0
    draw = ImageDraw.Draw(canvas)
    if values is None:
        draw.rectangle(box, fill=MISSING_FILL)
        draw.text(
            ((x0 + x1) / 2, (y0 + y1) / 2 - 6),
            fonts.t("missing", "missing"),
            font=fonts.label,
            fill=(210, 212, 216),
            anchor="mm",
        )
    else:
        image = Image.fromarray(_to_uint8(values, vmax))
        image = image.resize((width, height), Image.Resampling.BILINEAR)
        canvas.paste(image.convert("RGB"), (x0, y0))
    caption_h = min(18, max(12, height // 8))
    draw.rectangle((x0, y1 - caption_h, x1, y1), fill=(0, 0, 0))
    if values is not None:
        baseline = y1 - caption_h // 2
        draw.text((x0 + 4, baseline), caption_left, font=fonts.cell, fill=FG, anchor="lm")
        draw.text((x1 - 4, baseline), caption_right, font=fonts.cell, fill=FG, anchor="rm")
    draw.rectangle(box, outline=BORDER, width=1)


def _compose_grid(
    run: RunData,
    rows: tuple[float, ...],
    cols: tuple[int, ...],
    prepared: dict[tuple[str, int], np.ndarray],
    global_vmax: float,
    options: RenderOptions,
    fonts: _Fonts,
    *,
    cell_px: int,
    title: str,
    subtitle: str,
    footer: str,
    column_headers: bool,
) -> tuple[Image.Image, int]:
    """合成一张多行网格图：行 = lr，列 = 迭代；返回 ``(图, 缺帧数)``。"""
    margin = 16
    gutter = 92
    gap = 10
    header_h = 84 if subtitle else 60
    if column_headers:
        header_h += 22

    grid_w = gutter + len(cols) * cell_px + (len(cols) - 1) * gap
    width = max(560, margin * 2 + grid_w)
    x_start = (width - grid_w) // 2

    footer_lines = _wrap_text(footer, fonts.note, width - 2 * margin)
    footer_h = len(footer_lines) * 18 + margin

    row_count = len(rows)
    height = header_h + row_count * cell_px + (row_count - 1) * gap + footer_h
    canvas = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(canvas)

    draw.text((x_start, 12), title, font=fonts.title, fill=FG, anchor="la")
    if subtitle:
        draw.text((x_start, 46), subtitle, font=fonts.note, fill=MUTED, anchor="la")
    if not column_headers and cols:
        draw.text((width - margin, 16), f"iter {cols[0]:04d}", font=fonts.label, fill=FG, anchor="ra")

    if column_headers:
        for col_index, it in enumerate(cols):
            cell_x0 = x_start + gutter + col_index * (cell_px + gap)
            draw.text(
                (cell_x0 + cell_px / 2, header_h - 8),
                f"iter {it:04d}",
                font=fonts.cell,
                fill=MUTED,
                anchor="ms",
            )

    missing = 0
    for row_index, lr in enumerate(rows):
        y0 = header_h + row_index * (cell_px + gap)
        label = f"lr={lr:g}"
        draw.text((x_start - 12, y0 + cell_px / 2), label, font=fonts.label, fill=FG, anchor="rm")
        for col_index, it in enumerate(cols):
            key = (_lr_key(lr), it)
            values = prepared.get(key)
            record = run.records.get(key)
            cell_data = run.cells.get(key)
            peak = record.peak if record is not None and record.peak is not None else (
                float(values.max()) if values is not None else None
            )
            loss = record.loss if record is not None and record.loss is not None else (
                cell_data.loss if cell_data is not None else None
            )
            vmax = global_vmax if options.normalize == "global" else (
                float(values.max()) if values is not None else 1.0
            )
            if values is None:
                missing += 1
            box = (
                x_start + gutter + col_index * (cell_px + gap),
                y0,
                x_start + gutter + col_index * (cell_px + gap) + cell_px,
                y0 + cell_px,
            )
            _draw_cell(
                canvas,
                box,
                values,
                vmax if vmax > 0 else 1.0,
                fonts,
                fonts.t(f"峰值 {_fmt_optional(peak, '{:.0f}')}", f"peak {_fmt_optional(peak, '{:.0f}')}"),
                fonts.t(f"loss {_fmt_optional(loss)}", f"loss {_fmt_optional(loss)}"),
            )

    footer_y = header_h + row_count * cell_px + (row_count - 1) * gap + 12
    for line_index, line in enumerate(footer_lines):
        draw.text((margin, footer_y + line_index * 18), line, font=fonts.note, fill=MUTED, anchor="la")
    return canvas, missing


# ---------------------------------------------------------------------------
# 渲染主流程
# ---------------------------------------------------------------------------


def _exposure_desc(run: RunData) -> str:
    values = {cell.exposure_ms for cell in run.cells.values() if cell.exposure_ms is not None}
    if not values:
        return "曝光未知"
    low, high = min(values), max(values)
    if math.isclose(low, high):
        return f"曝光 {low:g} ms"
    return f"曝光 {low:g}-{high:g} ms"


def _normalize_desc(options: RenderOptions) -> str:
    if options.normalize == "global":
        return "整组共用灰度上限(p99.5)"
    return "逐帧峰值归一"


def _log_desc(options: RenderOptions) -> str:
    return "log10(1+I)" if options.log_scale else "线性灰度"


def _lr_centers(run: RunData, options: RenderOptions) -> dict[str, tuple[float, float]]:
    """每个 lr 用最早一帧确定质心，之后固定。"""
    centers: dict[str, tuple[float, float]] = {}
    for lr in run.lrs:
        key = _lr_key(lr)
        available = sorted(it for (lr_key, it) in run.cells if lr_key == key)
        if available:
            centers[key] = _spot_center(run.cells[(key, available[0])].frame)
        else:
            cell = next((c for c in run.cells.values() if _lr_key(c.lr) == key), None)
            height, width = cell.frame.shape[:2] if cell is not None else (options.window, options.window)
            centers[key] = (width / 2.0, height / 2.0)
    return centers


def _sample_iters(iters: tuple[int, ...], every: int) -> tuple[int, ...]:
    return tuple(iters[::every])


def _write_metrics_csv(path: Path, run: RunData) -> int:
    """journal + 帧可用性汇总；返回数据行数。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted(set(run.cells) | set(run.records), key=_sort_key)
    fieldnames = [
        "lr",
        "iter",
        "exposure_ms",
        "peak",
        "saturated",
        "counts",
        "fwhm",
        "loss",
        "flat",
        "frame_present",
        "frame_file",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for lr_key, it in keys:
            record = run.records.get((lr_key, it))
            cell = run.cells.get((lr_key, it))
            lr = record.lr if record is not None else (cell.lr if cell is not None else float(lr_key))
            writer.writerow(
                {
                    "lr": f"{lr:g}",
                    "iter": it,
                    "exposure_ms": _fmt_optional(record.exposure_ms if record else (cell.exposure_ms if cell else None), "{:.6g}"),
                    "peak": _fmt_optional(record.peak if record else None, "{:.6g}"),
                    "saturated": "" if record is None else int(record.saturated),
                    "counts": _fmt_optional(record.counts if record else None, "{:.6g}"),
                    "fwhm": _fmt_optional(record.fwhm if record else None, "{:.6g}"),
                    "loss": _fmt_optional(record.loss if record else (cell.loss if cell else None), "{:.6g}"),
                    "flat": "" if record is None else int(record.flat),
                    "frame_present": int(cell is not None),
                    "frame_file": "" if cell is None else cell.source.relative_to(run.run_dir).as_posix(),
                }
            )
    return len(keys)


def render_run(
    run_dir: Path,
    out: Path,
    options: RenderOptions,
    *,
    contact_sheet: Path | None = None,
    metrics_csv: Path | None = None,
) -> RenderResult:
    """读归档并写出 GIF / 总览 PNG / 指标 CSV。"""
    run = load_run(run_dir)
    for warning in run.warnings:
        logger.warning(warning)
    if not run.iters:
        raise RuntimeError(f"{run_dir} 没有可渲染的迭代")

    fonts = _Fonts()
    sampled = _sample_iters(run.iters, options.every)
    centers = _lr_centers(run, options)

    prepared: dict[tuple[str, int], np.ndarray] = {}
    for lr in run.lrs:
        key = _lr_key(lr)
        for it in sampled:
            cell = run.cells.get((key, it))
            if cell is not None:
                prepared[(key, it)] = _prepare_display(cell.frame, centers[key], options)
    global_vmax = _pooled_vmax(list(prepared.values()))

    subtitle = fonts.t(
        f"行=lr，一帧=一个迭代步；窗口 {options.window}px 以各 lr 首帧质心固定",
        f"rows=lr, one frame per iter; fixed {options.window}px window at first-frame centroid",
    )
    footer = fonts.t(
        f"{run.name} | {_exposure_desc(run)} | {_log_desc(options)} | {_normalize_desc(options)}"
        " | 脚本未做暗减，直接显示归档原始 CCD 帧",
        f"{run.name} | {_exposure_desc(run)} | {_log_desc(options)} | {_normalize_desc(options)}"
        " | no dark subtraction, raw archived CCD frames",
    )
    title = fonts.t("实测远场 · 不同 lr 随迭代变化", "Measured far field: lr sweep over iterations")

    total_slots = len(run.lrs) * len(sampled)
    missing_total = sum(
        1 for lr in run.lrs for it in sampled if (_lr_key(lr), it) not in prepared
    )
    footer = footer + fonts.t(
        f" | 缺帧 {missing_total}/{total_slots}（灰底 missing）" if missing_total else " | 无缺帧",
        f" | missing {missing_total}/{total_slots} (grey cells)" if missing_total else " | no missing cells",
    )

    gif_frames: list[Image.Image] = []
    for it in sampled:
        frame_img, _ = _compose_grid(
            run,
            run.lrs,
            (it,),
            prepared,
            global_vmax,
            options,
            fonts,
            cell_px=options.cell_px,
            title=title,
            subtitle=subtitle,
            footer=footer,
            column_headers=False,
        )
        gif_frames.append(frame_img)

    out.parent.mkdir(parents=True, exist_ok=True)
    paletted = [
        frame.quantize(colors=GIF_COLORS, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
        for frame in gif_frames
    ]
    paletted[0].save(
        out,
        save_all=True,
        append_images=paletted[1:],
        duration=options.duration_ms,
        loop=0,
        optimize=False,
        disposal=2,
    )
    with Image.open(out) as check:
        written_frames = int(check.n_frames)

    sheet_path: Path | None = None
    if contact_sheet is not None:
        sheet_footer = footer + fonts.t(
            f" | 静态总览列 = --every={options.every} 抽样后的全部迭代",
            f" | contact sheet: all iterations sampled by --every={options.every}",
        )
        sheet, _ = _compose_grid(
            run,
            run.lrs,
            sampled,
            prepared,
            global_vmax,
            options,
            fonts,
            cell_px=options.sheet_cell_px,
            title=title,
            subtitle=subtitle,
            footer=sheet_footer,
            column_headers=True,
        )
        contact_sheet.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(contact_sheet)
        sheet_path = contact_sheet

    csv_path: Path | None = None
    if metrics_csv is not None:
        _write_metrics_csv(metrics_csv, run)
        csv_path = metrics_csv

    logger.info(
        "完成: {} 帧 GIF={} · lr={} · iter={}–{} · 缺帧 {}/{}",
        written_frames,
        out,
        [f"{lr:g}" for lr in run.lrs],
        sampled[0] if sampled else "-",
        sampled[-1] if sampled else "-",
        missing_total,
        total_slots,
    )
    return RenderResult(
        gif_path=out,
        contact_sheet_path=sheet_path,
        metrics_csv_path=csv_path,
        gif_frames=written_frames,
        missing_cells=missing_total,
        lrs=run.lrs,
        iters=sampled,
    )


# ---------------------------------------------------------------------------
# 自检：合成假 run，跑完整渲染流程
# ---------------------------------------------------------------------------


def _synthetic_frame(rng: np.random.Generator, size: int, sigma: float, amplitude: float) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    center = size / 2.0
    blob = amplitude * np.exp(-((xx - center) ** 2 + (yy - center) ** 2) / (2.0 * sigma**2))
    noise = rng.normal(0.0, 3.0, size=(size, size))
    return np.clip(blob + noise, 0.0, None).astype(np.uint16)


def _write_synthetic_run(root: Path, lrs: list[float], iters: int, size: int) -> Path:
    rng = np.random.default_rng(20261127)
    run_dir = root / "run_self_test"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "flat").mkdir(exist_ok=True)
    (run_dir / "run_config.json").write_text(
        json.dumps(
            {
                "params": {"exposure_ms": 0.4, "self_test": True},
                "lrs": lrs,
                "iters": iters,
                "capture_every": 1,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    flat = _synthetic_frame(rng, size, 9.0, 40.0)
    np.savez(run_dir / "flat" / "iter_0000.npz", frame=flat, exposure_ms=0.4)
    journal_lines: list[str] = []
    for lr in lrs:
        dir_name = "lr_" + f"{lr:g}".replace(".", "p")
        lr_dir = run_dir / dir_name
        lr_dir.mkdir(exist_ok=True)
        for it in range(iters):
            sigma = max(2.5, 7.0 - 0.7 * it) + (0.4 if lr > 0.03 else 0.0)
            frame = _synthetic_frame(rng, size, sigma, 900.0)
            loss = 0.4 * math.exp(-0.35 * it) + 0.01
            np.savez(
                lr_dir / f"iter_{it:04d}.npz",
                frame=frame,
                phase_model=np.zeros((size, size), dtype=np.float32),
                exposure_ms=0.4,
                lr=lr,
                iter=it,
                loss=loss,
            )
            journal_lines.append(
                json.dumps(
                    {
                        "lr": lr,
                        "iter": it,
                        "exposure_ms": 0.4,
                        "peak": float(frame.max()),
                        "saturated": False,
                        "counts": float(frame.sum()),
                        "centroid": [size / 2.0, size / 2.0],
                        "fwhm": float(sigma * 2.355),
                        "loss": loss,
                        "flat": False,
                    }
                )
            )
    (run_dir / "journal.jsonl").write_text("\n".join(journal_lines) + "\n", encoding="utf-8")
    return run_dir


def self_test() -> int:
    """合成 2 lr × 5 迭代的假 run，跑完整流程并断言产物。"""
    lrs = [0.02, 0.05]
    iters = 5
    size = 64
    with tempfile.TemporaryDirectory(prefix="lr_farfield_selftest_") as tmp:
        root = Path(tmp)
        run_dir = _write_synthetic_run(root, lrs, iters, size)
        out_dir = root / "out"
        out = out_dir / "selftest.gif"
        sheet = out_dir / "selftest_contact_sheet.png"
        csv_path = out_dir / "selftest_metrics.csv"
        options = RenderOptions(
            window=48,
            log_scale=True,
            normalize="global",
            duration_ms=80,
            every=1,
            cell_px=96,
            sheet_cell_px=90,
        )
        result = render_run(run_dir, out, options, contact_sheet=sheet, metrics_csv=csv_path)

        expected_frames = iters
        assert out.is_file(), f"GIF 未生成: {out}"
        assert sheet.is_file(), f"总览 PNG 未生成: {sheet}"
        assert csv_path.is_file(), f"指标 CSV 未生成: {csv_path}"
        assert result.gif_frames == expected_frames, f"GIF 帧数 {result.gif_frames} != {expected_frames}"
        assert result.missing_cells == 0, f"自检数据不应有缺帧: {result.missing_cells}"
        with Image.open(out) as gif:
            assert int(gif.n_frames) == expected_frames, f"读回帧数 {gif.n_frames} != {expected_frames}"
            gif_size = gif.size
        with Image.open(sheet) as png:
            sheet_size = png.size
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            csv_rows = sum(1 for _ in csv.reader(handle)) - 1
        assert csv_rows == len(lrs) * iters, f"CSV 行数 {csv_rows} != {len(lrs) * iters}"

        logger.info("self-test GIF: {} ({}x{}, {} 帧, {:.1f} KiB)", out, gif_size[0], gif_size[1], result.gif_frames, out.stat().st_size / 1024)
        logger.info("self-test PNG: {} ({}x{}, {:.1f} KiB)", sheet, sheet_size[0], sheet_size[1], sheet.stat().st_size / 1024)
        logger.info("self-test CSV: {} ({} 行数据)", csv_path, csv_rows)
    logger.info("self-test 临时目录已清理")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="把 hw_lr_farfield_sweep_20261127 的实测归档渲染成多行网格 GIF（纯离线）。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir", type=Path, help="采集归档目录（含 run_config.json / journal.jsonl / lr_*）")
    parser.add_argument("--out", type=Path, help="输出 GIF 路径")
    parser.add_argument("--every", type=int, default=1, help="迭代抽样步长")
    parser.add_argument("--window", type=int, default=256, help="以质心为中心的显示窗口像素")
    parser.add_argument(
        "--log",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="用 log10(1+I) 显示（--no-log 为线性）",
    )
    parser.add_argument("--normalize", choices=("frame", "global"), default="global", help="灰度归一化口径")
    parser.add_argument("--duration-ms", type=int, default=380, help="GIF 每帧时长 (ms)")
    parser.add_argument("--cell-px", type=int, default=180, help="GIF 单元格像素")
    parser.add_argument("--sheet-cell-px", type=int, default=150, help="静态总览单元格像素")
    parser.add_argument("--contact-sheet", type=Path, default=None, help="静态总览 PNG（默认由 --out 派生）")
    parser.add_argument("--metrics-csv", type=Path, default=None, help="指标 CSV（默认由 --out 派生）")
    parser.add_argument("--no-contact-sheet", action="store_true", help="不输出静态总览")
    parser.add_argument("--no-metrics-csv", action="store_true", help="不输出指标 CSV")
    parser.add_argument("--self-test", action="store_true", help="合成假 run 跑完整流程（不读真实数据、不碰设备）")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()
    if args.run_dir is None or args.out is None:
        parser.error("真实渲染需要 --run-dir 与 --out（或使用 --self-test）")

    options = RenderOptions(
        window=args.window,
        log_scale=bool(args.log),
        normalize=args.normalize,
        duration_ms=args.duration_ms,
        every=args.every,
        cell_px=args.cell_px,
        sheet_cell_px=args.sheet_cell_px,
    )
    out: Path = args.out
    contact_sheet = None if args.no_contact_sheet else (args.contact_sheet or out.with_name(f"{out.stem}_contact_sheet.png"))
    metrics_csv = None if args.no_metrics_csv else (args.metrics_csv or out.with_name(f"{out.stem}_metrics.csv"))

    try:
        render_run(args.run_dir, out, options, contact_sheet=contact_sheet, metrics_csv=metrics_csv)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        logger.error("渲染失败: {}", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
