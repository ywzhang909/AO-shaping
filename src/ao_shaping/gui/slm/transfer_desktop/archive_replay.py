"""归档回放弹窗 (离线): 递归扫描 debug 根下所有历史 ``*.pkl`` 并按需回放。

为什么单独一个模块
------------------
迁移学习闭环页跑完之后, 能拿来复盘的是 ``data/debug/slm_transfer`` 下**每一次会话**
落盘的证据。历史里 ``*.pkl`` 只有两种文件名, 却有三种内容口味:

* ``capture_records.pkl`` —— 一块一条记录, 记录里 ``file`` 指向同目录的
  ``capture_<序号>.npz``。npz 里是 ``raw_frames`` (多帧原始 CCD)、``panel_phase`` /
  ``panel_gray`` (送到 SLM 的相位)、``camera_xy`` 与 ``core`` 采样网格。回放它 =
  逐条用 :func:`load_capture` 读兄弟 npz, 得到页面里同样的三块视图。
* ``iterations.pkl`` (相位口味, ``anchored/`` / ``model/`` / ``response/``) ——
  一条一次迭代, 带 ``phase`` (弧度 ndarray)、``terms`` (loss / ee_roi / ...), **无 CCD 帧**。
* ``iterations.pkl`` (拟合口味, ``fit/``) —— 一条一次迭代, 带 ``loss`` / ``parameters``,
  **既无相位也无 CCD 帧**。

本模块把三者都列进左侧归档列表 (整个 debug 根递归扫, 按修改时间从新到旧), 点谁才
``pickle.load`` 谁 (98 MB 的相位文件要几秒, 扫描阶段绝不打开 pkl)。三种口味都**能回放**:
有 CCD 就出三视图, 只有相位就出相位预览 + CCD 占位, 只有拟合指标就出行文提示 —— 一个
损坏文件也只是在窗口里显示一行红字, 弹窗不抛异常、不关窗。

为什么不用 ``pattern_controls._phase_to_preview``
------------------------------------------------
那个函数本身是纯 numpy + matplotlib, 但它的模块顶部 ``import`` 了
:class:`~ao_shaping.drivers.slm.santec.Santec` —— 为了一个色图把 SLM 驱动拉进
无设备的回放进程不合适。所以这里给出同口径的 :func:`phase_to_preview`
(同样的 ``0..max_gray`` 归一化、同样的 ``twilight`` / 灰度两种色图), 不在本模块
import 任何 ``drivers.ccd`` / ``drivers.slm`` 符号; 测试会把这一点钉住。

与页面的关系
------------
本模块只提供 :func:`open_archive_replay`, 由页面在自己的按钮里调用, 默认根目录为
:data:`DEBUG_ROOT`。不反向 import 页面 (页面会惰性 import 硬件自检与零级读取)。
全程只读, 不打开相机 / SLM。
"""

from __future__ import annotations

import pickle
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger
from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtGui import QImage, QPixmap, QResizeEvent
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

__all__ = [
    "DEBUG_ROOT",
    "ArchiveEntry",
    "ArchivePayload",
    "ArchiveRecord",
    "ArchiveReplayDialog",
    "CaptureData",
    "discover_archives",
    "discover_captures",
    "load_archive",
    "load_capture",
    "open_archive_replay",
    "phase_to_preview",
]

#: 归档回放扫描根目录 (与 ``history_browser.DEFAULT_ROOT`` / ``params.plan_session_dir``
#: 同一约定: 相对当前工作目录)。所有历史会话都在这个根下面, 递归找 ``*.pkl``。
DEBUG_ROOT = Path("data") / "debug" / "slm_transfer"

#: 归档 capture 的文件名通配符 (只认 ``anchored/`` 下这一种; ``discover_captures`` 用)。
CAPTURE_GLOB = "capture_*.npz"

#: 所有历史归档的通配符: 只要满足 ``*.pkl`` 都能回放。
ARCHIVE_GLOB = "*.pkl"

#: SLM 面板的灰度满量程 (10 bit)。与 ``Santec.MAX_GRAYSCALE_VALUE`` 同值; 这里写字面
#: 量, 免得为了一个常数 import 驱动。
PREVIEW_MAX_GRAY = 1023

#: 相位预览的两种色图, 文案与 ``pattern_controls.PREVIEW_COLORMAP_LABELS`` 一致。
PREVIEW_COLORMAPS: tuple[str, ...] = ("相位色图 (twilight)", "灰度")

#: 读取/解码失败时兜住的异常: 坏 npz 走 BadZipFile / KeyError, 坏 pkl 走
#: UnpicklingError / EOFError / AttributeError / ModuleNotFoundError / MemoryError 等。
#: 兜住它们, 单个损坏文件只上屏一行提示, 弹窗绝不崩。
_LOAD_ERRORS: tuple[type[BaseException], ...] = (
    OSError,
    KeyError,
    ValueError,
    EOFError,
    zipfile.BadZipFile,
    pickle.UnpicklingError,
    AttributeError,
    ModuleNotFoundError,
    ImportError,
    MemoryError,
)

#: 鼠标拖动滑块的防抖间隔 (ms): 大帧分析是重活, 别每个 tick 都跑一遍。
_SLIDER_DEBOUNCE_MS = 140

#: 左侧归档列表面板宽度 (px)。
_ARCHIVE_LIST_WIDTH = 320

#: 三种内容口味对应的可读提示 (无 CCD / 无相位时的占位文案)。
_NOTICE_ITER_PHASE = "该归档只有迭代相位, 无 CCD 帧"
_NOTICE_ITER_FIT = "该归档只有拟合指标 (无相位 / 无 CCD 帧)"
_NOTICE_UNKNOWN = "该归档没有可回放的相位或 CCD 帧"

#: 各种口味在 meta 行里优先展示的指标键 (按此顺序, 缺就跳过)。
_DISPLAY_KEYS: dict[str, tuple[str, ...]] = {
    "capture_records": ("cv", "ee_roi", "mean", "target_pearson", "core_peak", "peak"),
    "iterations_phase": (
        "shape",
        "block",
        "iteration",
        "loss",
        "ee_roi",
        "l_cv",
        "shape_loss",
        "phase_rms",
    ),
    "iterations_fit": (
        "iteration",
        "loss",
        "defocus_diopter",
        "gain",
        "sigma_ccd_px",
        "background",
        "bias_counts",
    ),
    "unknown": (),
}


# --------------------------------------------------------------------- 纯逻辑
@dataclass(frozen=True)
class CaptureData:
    """一块 capture 的展示数据 (只含上屏需要的字段)。"""

    index: int
    path: Path
    #: CCD 画面 (ROI 裁剪后, 已做分位拉伸的 uint8)。
    display_frame: np.ndarray
    #: 同一裁剪区的原始灰度 (analyzer 用它算剖面, 数值保持原样)。
    raw_frame: np.ndarray
    #: SLM 相位预览 (float32 RGB, 值域 0..1)。
    phase_rgb: np.ndarray
    #: 裁剪区 ``(y0, y1, x0, x1)``, 在原始 CCD 像素坐标下。
    roi: tuple[int, int, int, int]
    #: 这块 capture 里平均了几帧 ``raw_frames``。
    frame_count: int
    raw_peak: float
    raw_mean: float


@dataclass(frozen=True)
class ArchiveEntry:
    """扫描到的一个 ``*.pkl`` 归档 (纯文件系统信息, 不含任何解码内容)。"""

    path: Path
    #: 相对扫描根的路径 (posix 斜杠), 例如 ``physical_20261102c/anchored/capture_records.pkl``。
    relative: str
    #: 由文件名 stem 派生: ``capture_records`` / ``iterations`` (其余 stem 归 ``other``)。
    kind: str
    size_bytes: int
    mtime: float


@dataclass(frozen=True)
class ArchiveRecord:
    """归一化后的一条可回放记录 (不管原 pkl 是哪种口味, 都收敛到这 5 个字段)。"""

    index: int
    label: str
    #: CCD 来源 (兄弟 npz 路径) 或 ``None``。路径存在但文件缺失时保留, 由界面标记不可用。
    ccd_path: Path | None
    #: 迭代相位 ndarray 或 ``None``。
    phase: np.ndarray | None
    #: 该记录要展示的指标字典 (capture metrics / iteration terms / fit parameters)。
    metrics: dict


@dataclass
class ArchivePayload:
    """一次 :func:`load_archive` 的结果: 记录列表 + 可用视图 (由 flavor/notice 表达)。"""

    entry: ArchiveEntry
    records: list[ArchiveRecord]
    #: ``capture_records`` / ``iterations_phase`` / ``iterations_fit`` / ``unknown``。
    flavor: str
    #: 无 CCD / 无相位时显示在画面与提示行的占位文案 (全视图口味为 ``""``)。
    notice: str = ""


def discover_captures(session_dir: str | Path) -> list[Path]:
    """列出会话目录下 ``anchored/capture_*.npz``, 按序号升序。

    ``session_dir`` 也可以直接是 ``anchored/`` 目录; 目录不存在时返回空表,
    由界面显示"没有可回放的 capture", 不抛异常。
    """
    root = Path(session_dir)
    anchored = root / "anchored"
    base = anchored if anchored.is_dir() else root
    if not base.is_dir():
        return []
    return sorted(path for path in base.glob(CAPTURE_GLOB) if path.is_file())


def discover_archives(root: str | Path) -> list[ArchiveEntry]:
    """递归列出 ``root`` 下所有 ``*.pkl``, 按修改时间**从新到旧** (同 mtime 按相对路径升序)。

    只做文件系统枚举 (``rglob`` + ``stat``), **绝不打开任何 pkl** —— 根目录里有几百个
    pkl / 几十 GB, 单个相位 pkl 可达 98 MB, 扫描阶段解码会卡死界面。根目录不存在 /
    不是目录时返回 ``[]`` (界面显示提示, 不抛异常)。
    """
    base = Path(root)
    if not base.is_dir():
        return []
    entries: list[ArchiveEntry] = []
    for path in base.rglob(ARCHIVE_GLOB):
        if not path.is_file():
            continue
        try:
            st = path.stat()
            relative = path.relative_to(base).as_posix()
        except (OSError, ValueError):
            relative = path.name
            try:
                st = path.stat()
            except OSError:
                continue
        kind = "capture_records" if path.stem == "capture_records" else "iterations"
        if path.stem not in ("capture_records", "iterations"):
            kind = "other"
        entries.append(
            ArchiveEntry(
                path=path,
                relative=relative,
                kind=kind,
                size_bytes=int(st.st_size),
                mtime=float(st.st_mtime),
            )
        )
    entries.sort(key=lambda e: (-e.mtime, e.relative))
    return entries


def capture_number(path: Path) -> int:
    """从 ``capture_0012.npz`` 取出 12; 名字不合规时退回 0。"""
    stem = Path(path).stem
    _, _, suffix = stem.partition("_")
    try:
        return int(suffix)
    except ValueError:
        return 0


def _beam_roi(
    shape: tuple[int, int],
    camera_xy: np.ndarray | None,
    core: np.ndarray | None,
) -> tuple[int, int, int, int]:
    """由 ``camera_xy`` + ``core`` 得出光斑所在的裁剪框 ``(y0, y1, x0, x1)``。

    整幅 1944x2592 里光斑只占中间一小块, 不裁剪的话画面和剖面都被背景主导。
    ``core`` 是第一帧的目标核心区; 没有可用网格时退回以画面中心为基准的方框。
    """
    height, width = int(shape[0]), int(shape[1])
    xy = None
    if camera_xy is not None:
        candidate = np.asarray(camera_xy)
        if candidate.ndim == 3 and candidate.shape[-1] == 2:
            xy = candidate
    if xy is not None:
        mask = np.ones(xy.shape[:2], dtype=bool)
        if core is not None:
            candidate_core = np.asarray(core)
            if candidate_core.shape == mask.shape:
                mask = candidate_core.astype(bool)
        if mask.any():
            xs = xy[..., 0][mask].astype(float)
            ys = xy[..., 1][mask].astype(float)
            x0, x1 = float(xs.min()), float(xs.max())
            y0, y1 = float(ys.min()), float(ys.max())
            margin_x = max(8.0, 0.08 * (x1 - x0))
            margin_y = max(8.0, 0.08 * (y1 - y0))
            left = max(0, int(np.floor(x0 - margin_x)))
            right = min(width, int(np.ceil(x1 + margin_x)) + 1)
            top = max(0, int(np.floor(y0 - margin_y)))
            bottom = min(height, int(np.ceil(y1 + margin_y)) + 1)
            if right > left and bottom > top:
                return top, bottom, left, right

    half = max(32, int(0.15 * min(height, width)))
    cx, cy = width // 2, height // 2
    left = max(0, cx - half)
    right = min(width, cx + half)
    top = max(0, cy - half)
    bottom = min(height, cy + half)
    return top, bottom, left, right


def _display_uint8(frame: np.ndarray) -> np.ndarray:
    """按 1% / 99.9% 分位做线性拉伸, 让 16 bit 画面在 8 bit 显示上看得出结构。"""
    arr = np.asarray(frame, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((0, 0), dtype=np.uint8)
    low = float(np.percentile(arr, 1.0))
    high = float(np.percentile(arr, 99.9))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(np.nanmin(arr))
        high = float(np.nanmax(arr))
    if high <= low:
        high = low + 1.0
    scaled = np.clip((arr - low) / (high - low), 0.0, 1.0)
    return (scaled * 255.0).astype(np.uint8)


def phase_to_preview(
    phase_gray: np.ndarray,
    max_gray: int | None = None,
    colormap: str | None = None,
) -> np.ndarray:
    """原始 SLM 灰度 → RGB 预览图 (float32, 值域 0..1)。

    与 ``pattern_controls._phase_to_preview`` 同口径: 先除以面板满量程并钳到
    ``0..1``, 再走 ``twilight`` (循环色图, 0 与 2π 同色) 或灰度; 不 import 驱动。
    """
    if max_gray is None:
        max_gray = PREVIEW_MAX_GRAY
    max_gray = int(max(1, int(max_gray)))

    normalized = np.clip(
        np.asarray(phase_gray, dtype=np.float32) / float(max_gray),
        0.0,
        1.0,
    )
    if colormap == "灰度":
        gray = np.minimum(1.0, normalized * (1023 / max_gray))
        return np.stack([gray, gray, gray], axis=-1)

    import matplotlib

    cmap = matplotlib.colormaps["twilight"]
    rgba = cmap(normalized)
    return np.asarray(rgba[:, :, :3], dtype=np.float32)


def _phase_gray_from(data) -> np.ndarray | None:
    """优先用归档里的 ``panel_gray``; 缺了就由 ``panel_phase`` (弧度) 现算。"""
    if "panel_gray" in data.files:
        return np.asarray(data["panel_gray"])
    if "panel_phase" in data.files:
        phase = np.asarray(data["panel_phase"], dtype=np.float32)
        wrapped = np.mod(np.nan_to_num(phase), 2.0 * np.pi)
        return wrapped / (2.0 * np.pi) * float(PREVIEW_MAX_GRAY)
    return None


def _phase_array_to_gray(phase: np.ndarray) -> np.ndarray:
    """迭代 pkl 的 ``phase`` 是未包裹弧度 (可为负 / 可 >2π): 取模后映射到灰度。"""
    arr = np.nan_to_num(np.asarray(phase, dtype=np.float32))
    wrapped = np.mod(arr, 2.0 * np.pi)
    return (wrapped / (2.0 * np.pi) * float(PREVIEW_MAX_GRAY)).astype(np.float32)


def load_capture(path: str | Path, index: int | None = None) -> CaptureData:
    """读一个 ``capture_*.npz`` 并返回上屏数据 (只读这一个文件)。

    失败时把 ``OSError`` / ``KeyError`` / ``ValueError`` / ``zipfile.BadZipFile``
    原样抛给调用方 —— 弹窗负责把它变成可见提示, 见 ``_LOAD_ERRORS``。
    """
    capture_path = Path(path)
    if index is None:
        index = capture_number(capture_path)
    with np.load(capture_path, allow_pickle=False) as data:
        frames = np.asarray(data["raw_frames"])
        camera_xy = np.asarray(data["camera_xy"]) if "camera_xy" in data.files else None
        core = np.asarray(data["core"]) if "core" in data.files else None
        gray = _phase_gray_from(data)
        if gray is None:
            raise KeyError("panel_gray/panel_phase")
        phase_rgb = phase_to_preview(gray)

    if frames.ndim == 2:
        frames = frames[None]
    if frames.size == 0 or frames.ndim != 3:
        raise ValueError(f"raw_frames 形状不可用: {frames.shape}")

    frame_count = int(frames.shape[0])
    raw = frames.mean(axis=0, dtype=np.float32)
    del frames

    roi = _beam_roi(raw.shape, camera_xy, core)
    top, bottom, left, right = roi
    cropped = np.asarray(raw[top:bottom, left:right])
    return CaptureData(
        index=int(index),
        path=capture_path,
        display_frame=_display_uint8(cropped),
        raw_frame=cropped,
        phase_rgb=phase_rgb,
        roi=roi,
        frame_count=frame_count,
        raw_peak=float(cropped.max()) if cropped.size else 0.0,
        raw_mean=float(cropped.mean()) if cropped.size else 0.0,
    )


def _rgb_uint8(rgb: np.ndarray) -> np.ndarray:
    """float32 0..1 RGB → 连续 uint8 RGB (给 ``QImage`` 直接引用)。"""
    arr = np.asarray(rgb, dtype=np.float32)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    return np.ascontiguousarray((np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8))


def _to_pixmap(rgb: np.ndarray) -> QPixmap:
    """RGB 数组 → ``QPixmap`` (``copy`` 后不再依赖 numpy 缓冲区寿命)。"""
    arr = _rgb_uint8(rgb)
    if arr.size == 0:
        return QPixmap()
    height, width = int(arr.shape[0]), int(arr.shape[1])
    image = QImage(
        arr.data,
        width,
        height,
        int(arr.strides[0]),
        QImage.Format.Format_RGB888,
    )
    return QPixmap.fromImage(image.copy())


# --------------------------------------------------------------- pkl 归一化
def _as_records(raw: object) -> list[dict]:
    """把一个 pkl 的反序列化结果收敛成 ``list[dict]`` (非 dict 元素占位成 ``{}``)。"""
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        return []
    return [rec if isinstance(rec, dict) else {} for rec in raw]


def _payload_capture_records(
    entry: ArchiveEntry, records: list[dict]
) -> ArchivePayload:
    out: list[ArchiveRecord] = []
    for i, rec in enumerate(records):
        name = rec.get("file")
        ccd = entry.path.parent / str(name) if name else None
        metrics: dict = {}
        raw_metrics = rec.get("metrics")
        if isinstance(raw_metrics, dict):
            metrics.update(raw_metrics)
        for key in ("peak", "core_peak", "saturation_pixels", "exposure_ms"):
            if key in rec:
                metrics.setdefault(key, rec[key])
        out.append(
            ArchiveRecord(
                index=i,
                label=str(rec.get("label") or f"block {i}"),
                ccd_path=ccd,
                phase=None,
                metrics=metrics,
            )
        )
    return ArchivePayload(entry, out, "capture_records", "")


def _payload_iterations_phase(
    entry: ArchiveEntry, records: list[dict]
) -> ArchivePayload:
    out: list[ArchiveRecord] = []
    for i, rec in enumerate(records):
        phase = rec.get("phase")
        arr = np.asarray(phase) if phase is not None else None
        metrics: dict = {}
        terms = rec.get("terms")
        if isinstance(terms, dict):
            metrics.update(terms)
        for key in ("shape", "block", "iteration"):
            if key in rec:
                metrics[key] = rec[key]
        label = (
            f"{rec.get('shape', '?')} · block {rec.get('block', '?')} · "
            f"iter {rec.get('iteration', i)}"
        )
        out.append(ArchiveRecord(i, label, None, arr, metrics))
    return ArchivePayload(entry, out, "iterations_phase", _NOTICE_ITER_PHASE)


def _payload_iterations_fit(entry: ArchiveEntry, records: list[dict]) -> ArchivePayload:
    out: list[ArchiveRecord] = []
    for i, rec in enumerate(records):
        metrics: dict = {}
        parameters = rec.get("parameters")
        if isinstance(parameters, dict):
            metrics.update(parameters)
        for key in ("iteration", "loss"):
            if key in rec:
                metrics[key] = rec[key]
        out.append(
            ArchiveRecord(i, f"fit iter {rec.get('iteration', i)}", None, None, metrics)
        )
    return ArchivePayload(entry, out, "iterations_fit", _NOTICE_ITER_FIT)


def _payload_unknown(entry: ArchiveEntry, records: list[dict]) -> ArchivePayload:
    out = [ArchiveRecord(i, f"记录 {i}", None, None, {}) for i in range(len(records))]
    return ArchivePayload(entry, out, "unknown", _NOTICE_UNKNOWN)


def load_archive(entry: ArchiveEntry) -> ArchivePayload:
    """``pickle.load`` 一个 pkl (一次), 把记录归一化成 :class:`ArchivePayload`。

    按内容而不是仅靠文件名判断口味 (文件名 stem 只是提示): 有 ``file`` → capture_records;
    有 ``phase`` → 迭代相位; 有 ``parameters`` / ``loss`` → 拟合指标; 其余 → 通用降级。
    解码失败抛 ``_LOAD_ERRORS`` 里的异常, 由弹窗转成可见提示。
    """
    with entry.path.open("rb") as handle:
        raw = pickle.load(handle)
    records = _as_records(raw)
    sample = records[0] if records else {}
    if entry.kind == "capture_records" or "file" in sample:
        return _payload_capture_records(entry, records)
    if "phase" in sample:
        return _payload_iterations_phase(entry, records)
    if "parameters" in sample or "loss" in sample:
        return _payload_iterations_fit(entry, records)
    return _payload_unknown(entry, records)


# ---------------------------------------------------------------- 小工具
def _human_size(num_bytes: int) -> str:
    if num_bytes >= 1 << 20:
        return f"{num_bytes / (1 << 20):.1f} MB"
    if num_bytes >= 1 << 10:
        return f"{num_bytes / (1 << 10):.1f} KB"
    return f"{num_bytes} B"


def _short(value: object) -> str:
    """把一个指标值压成一行 (浮点 4 位有效数字, 数组只列前 4 个)。"""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4g}"
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return str(int(value))
    arr = np.asarray(value)
    if arr.ndim > 0:
        flat = arr.ravel()
        head = ", ".join(_short(item) for item in flat[:4])
        return f"[{head}{'…' if flat.size > 4 else ''}]"
    return str(value)


def _format_metrics(metrics: dict, keys: tuple[str, ...]) -> str:
    parts = [
        f"{key}={_short(metrics[key])}"
        for key in keys
        if key in metrics and metrics[key] is not None
    ]
    return " · ".join(parts)


# ------------------------------------------------------------------- 弹窗
class ArchiveReplayDialog(QDialog):
    """离线回放弹窗: 左侧归档列表 + CCD 画面 + X/Y 剖面 + SLM 相位预览, 只读。

    非模态: 操作员可以一边开着回放一边看页面。列表由 :func:`discover_archives`
    递归扫整个 debug 根得到; 点某个 pkl 才 ``pickle.load`` (惰性), 每次切记录只读
    **该记录**的兄弟 npz (读完即弃, 不缓存整批帧)。三种口味都降级可看, 坏文件不崩。
    """

    def __init__(
        self,
        root: str | Path,
        parent: QWidget | None = None,
        preselect: str | Path | None = None,
    ) -> None:
        super().__init__(parent)
        self._root = Path(root)
        self._archives = discover_archives(self._root)
        self._visible: list[ArchiveEntry] = []
        self._record_counts: dict[Path, int] = {}
        self._payload: ArchivePayload | None = None
        self._index = 0
        self._sample: CaptureData | None = None
        self._phase_full: QPixmap | None = None
        self._active_row = -1

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(_SLIDER_DEBOUNCE_MS)
        self._timer.timeout.connect(self._reload_current)

        self.setModal(False)
        self.setWindowTitle("归档回放 · 离线")
        self.setMinimumSize(QSize(1100, 760))
        self.resize(QSize(1360, 860))
        self._build()
        self._populate_archives(preselect)

    # ------------------------------------------------------------ 构建
    def _build(self) -> None:
        # 惰性 import: 这两个部件所在的包会拉起 Qt 与 beam_quality 的壳, 且
        # 本模块的 import 契约是"不碰硬件驱动"。
        from ao_shaping.gui.ccd.beam_quality.profile_widget import ProfileWidget
        from ao_shaping.gui.ccd.beam_quality.video_widget import VideoWidget

        root = QVBoxLayout(self)

        self._header = QLabel(self)
        self._header.setWordWrap(True)
        self._header.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        root.addWidget(self._header)

        controls = QHBoxLayout()
        self._prev = QPushButton("上一块", self)
        self._next = QPushButton("下一块", self)
        self._slider = QSlider(Qt.Orientation.Horizontal, self)
        self._slider.setMinimum(0)
        self._slider.setSingleStep(1)
        self._slider.setPageStep(1)
        self._position = QLabel("—", self)
        self._prev.clicked.connect(self._on_prev)
        self._next.clicked.connect(self._on_next)
        self._slider.valueChanged.connect(self._on_slider)
        controls.addWidget(self._prev)
        controls.addWidget(self._slider, 1)
        controls.addWidget(self._next)
        controls.addWidget(self._position)
        root.addLayout(controls)

        outer = QSplitter(Qt.Orientation.Horizontal, self)

        # 左: 归档列表面板 (过滤框 + 列表)。
        archive_panel = QWidget(outer)
        archive_layout = QVBoxLayout(archive_panel)
        archive_layout.setContentsMargins(0, 0, 0, 0)
        self._filter = QLineEdit(archive_panel)
        self._filter.setPlaceholderText("过滤归档 (路径 / 口味)…")
        self._filter.setClearButtonEnabled(True)
        self._filter.textChanged.connect(self._on_filter)
        self._list = QListWidget(archive_panel)
        self._list.setMinimumWidth(_ARCHIVE_LIST_WIDTH)
        self._list.setStyleSheet("background-color: #12121a; color: #d8d8e0;")
        self._list.currentRowChanged.connect(self._on_archive_activated)
        archive_layout.addWidget(self._filter)
        archive_layout.addWidget(self._list, 1)

        # 右: 原有的 CCD/剖面 ↔ 相位  splitter。
        body = QSplitter(Qt.Orientation.Horizontal, outer)

        left = QWidget(body)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self._frame_caption = QLabel("CCD 画面 (raw_frames 多帧平均)", left)
        self._video = VideoWidget(left)
        self._profile_caption = QLabel("X / Y 剖面 (高斯拟合)", left)
        self._profiles = ProfileWidget(left)
        left_layout.addWidget(self._frame_caption)
        left_layout.addWidget(self._video, 3)
        left_layout.addWidget(self._profile_caption)
        left_layout.addWidget(self._profiles, 2)

        right = QWidget(body)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self._phase_caption = QLabel("SLM 相位预览 (panel_phase · twilight)", right)
        self._phase = QLabel("无相位预览", right)
        self._phase.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._phase.setMinimumSize(QSize(420, 300))
        self._phase.setStyleSheet("background-color: #12121a; color: #8a8a96;")
        right_layout.addWidget(self._phase_caption)
        right_layout.addWidget(self._phase, 1)

        body.addWidget(left)
        body.addWidget(right)
        body.setStretchFactor(0, 3)
        body.setStretchFactor(1, 2)
        body.setSizes([820, 480])

        outer.addWidget(archive_panel)
        outer.addWidget(body)
        outer.setStretchFactor(0, 0)
        outer.setStretchFactor(1, 1)
        outer.setSizes([_ARCHIVE_LIST_WIDTH, 1040])
        root.addWidget(outer, 1)

        self._meta = QLabel("", self)
        self._meta.setWordWrap(True)
        self._message = QLabel("", self)
        self._message.setWordWrap(True)
        self._message.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        root.addWidget(self._meta)
        root.addWidget(self._message)

    # ------------------------------------------------------------ 列表
    def _item_text(self, entry: ArchiveEntry) -> str:
        count = self._record_counts.get(entry.path)
        suffix = f" · {count} 条记录" if count is not None else ""
        return f"{entry.relative}\n[{entry.kind}] · {_human_size(entry.size_bytes)}{suffix}"

    def _update_item_text(self, entry: ArchiveEntry, count: int) -> None:
        self._record_counts[entry.path] = count
        for row, visible in enumerate(self._visible):
            if visible.path == entry.path:
                item = self._list.item(row)
                if item is not None:
                    item.setText(self._item_text(entry))
                break

    def _rebuild_list(self, text: str) -> None:
        needle = (text or "").strip().lower()
        self._visible = [
            entry
            for entry in self._archives
            if needle in entry.relative.lower() or needle in entry.kind.lower()
        ]
        self._list.blockSignals(True)
        self._list.clear()
        for entry in self._visible:
            self._list.addItem(self._item_text(entry))
        self._list.blockSignals(False)

    def _preselect_row(self, preselect: str | Path | None) -> int:
        if preselect is None:
            return 0
        target = Path(preselect).resolve()
        best = -1
        for row, entry in enumerate(self._visible):
            if entry.path.resolve().is_relative_to(target):
                if entry.kind == "capture_records":
                    return row
                if best < 0:
                    best = row
        return best if best >= 0 else 0

    def _populate_archives(self, preselect: str | Path | None) -> None:
        self._header.setText(
            f"归档根目录: {self._root}\n"
            f"共 {len(self._archives)} 个 pkl 归档 (递归扫描, 按修改时间从新到旧) "
            "· 只读, 不打开相机 / SLM"
        )
        self._rebuild_list("")
        if not self._visible:
            self._show_no_archives("")
            return
        row = self._preselect_row(preselect)
        self._list.blockSignals(True)
        self._list.setCurrentRow(row)
        self._list.blockSignals(False)
        item = self._list.item(row)
        if item is not None:
            self._list.scrollToItem(item)
        self.select_archive(row)

    def apply_filter(self, text: str) -> None:
        """按 ``text`` (大小写不敏感, 匹配相对路径或口味) 过滤左侧列表。"""
        self._filter.blockSignals(True)
        self._filter.setText(text)
        self._filter.blockSignals(False)
        self._on_filter(text)

    def _on_filter(self, text: str) -> None:
        active = self._payload.entry if self._payload is not None else None
        self._rebuild_list(text)
        if not self._visible:
            self._show_no_archives(text)
            return
        row = 0
        if active is not None:
            for index, entry in enumerate(self._visible):
                if entry.path == active.path:
                    row = index
                    break
        self._list.blockSignals(True)
        self._list.setCurrentRow(row)
        self._list.blockSignals(False)
        if self._payload is None or self._payload.entry.path != self._visible[row].path:
            self.select_archive(row)

    def _on_archive_activated(self, row: int) -> None:
        """用户点选: 先把"加载中…"画出来再同步解码 (大 pkl 要几秒)。"""
        if not (0 <= row < len(self._visible)):
            return
        self._set_loading(self._visible[row])
        app = QApplication.instance()
        if app is not None:
            app.processEvents()
        self.select_archive(row)

    def select_archive(self, row: int) -> None:
        """同步加载第 ``row`` 个可见归档并渲染 (测试 / 预选 / 截图直接调, 不依赖事件循环)。"""
        if not (0 <= row < len(self._visible)):
            return
        entry = self._visible[row]
        self._active_row = row
        self._set_loading(entry)
        self._load_and_render(entry)

    # ------------------------------------------------------------ 数据
    def _load_and_render(self, entry: ArchiveEntry) -> None:
        try:
            payload = load_archive(entry)
        except _LOAD_ERRORS as exc:
            self._show_error(entry.path, exc)
            self._payload = None
            self._set_slider_range(0)
            return
        self._payload = payload
        self._update_item_text(entry, len(payload.records))
        self._set_slider_range(len(payload.records))
        if payload.records:
            self.set_index(0)
        else:
            self._show_empty_archive(payload)

    def _set_loading(self, entry: ArchiveEntry) -> None:
        self._message.setTextFormat(Qt.TextFormat.PlainText)
        self._message.setText(f"加载中… {entry.relative}")
        self._phase_caption.setText("SLM 相位预览")
        self._phase.setText("加载中…")
        self._phase_full = None
        self._set_video_placeholder("加载中…")
        self._profiles.clear()
        self._meta.setText("")

    def _show_no_archives(self, text: str) -> None:
        self._payload = None
        self._sample = None
        self._phase_full = None
        self._set_slider_range(0)
        self._set_video_placeholder("没有可回放的归档")
        self._profiles.clear()
        self._phase.setText("无相位预览")
        self._meta.setText("")
        self._message.setTextFormat(Qt.TextFormat.PlainText)
        if text:
            self._message.setText(f"没有匹配的归档 (过滤: {text})。")
        else:
            self._message.setText("该根目录下没有可回放的 pkl 归档 (递归扫描 *.pkl)。")

    def _show_empty_archive(self, payload: ArchivePayload) -> None:
        self._sample = None
        self._phase_full = None
        self._set_video_placeholder("该归档没有记录")
        self._profiles.clear()
        self._phase.setText("无相位 / 无 CCD 帧")
        self._meta.setText("")
        self._message.setTextFormat(Qt.TextFormat.PlainText)
        self._message.setText("该 pkl 里没有可回放的记录 (空列表)。")

    def _set_video_placeholder(self, text: str) -> None:
        self._video.clear()
        label = getattr(self._video, "_label", None)
        if isinstance(label, QLabel):
            label.setText(text)

    # ------------------------------------------------------- 公开只读接口
    def archives(self) -> list[ArchiveEntry]:
        """扫描到的全部归档 (从新到旧)。"""
        return list(self._archives)

    def archive_count(self) -> int:
        """列表里归档总数 (未过滤前扫描到的 pkl 数)。"""
        return len(self._archives)

    @property
    def current_archive(self) -> ArchiveEntry | None:
        return self._payload.entry if self._payload is not None else None

    @property
    def current_payload(self) -> ArchivePayload | None:
        return self._payload

    @property
    def current_sample(self) -> CaptureData | None:
        """当前成功上屏的那一块 (无 CCD / 读取失败时为 ``None``)。"""
        return self._sample

    def capture_count(self) -> int:
        """当前归档里可回放的记录数 (0 表示没有归档 / 没有记录)。"""
        return len(self._payload.records) if self._payload is not None else 0

    def current_index(self) -> int:
        """当前选中的记录下标 (从 0 开始)。"""
        return self._index

    def message_text(self) -> str:
        """底部提示行原文 (测试与截图脚本都读它)。"""
        return self._message.text()

    # ------------------------------------------------------------ 切记录
    def _record_count(self) -> int:
        return len(self._payload.records) if self._payload is not None else 0

    def _position_text(self, index: int) -> str:
        total = self._record_count()
        label = (
            self._payload.records[index].label
            if self._payload is not None
            else f"块 {index}"
        )
        return f"{label} · {index + 1} / {total}"

    def _set_slider_range(self, count: int) -> None:
        if count <= 0:
            self._slider.setRange(0, 0)
            self._slider.setEnabled(False)
            self._prev.setEnabled(False)
            self._next.setEnabled(False)
            self._position.setText("0 / 0")
            return
        self._slider.setEnabled(True)
        self._prev.setEnabled(True)
        self._next.setEnabled(True)
        self._slider.setRange(0, count - 1)

    def set_index(self, index: int) -> None:
        """切到第 ``index`` 条记录并立即读盘 (越界自动钳位)。"""
        total = self._record_count()
        if not total:
            return
        index = max(0, min(total - 1, int(index)))
        self._index = index
        self._timer.stop()
        self._slider.blockSignals(True)
        self._slider.setValue(index)
        self._slider.blockSignals(False)
        self._position.setText(self._position_text(index))
        self._reload_current()

    def _reload_current(self) -> None:
        payload = self._payload
        if payload is None or not payload.records:
            return
        index = max(0, min(self._index, len(payload.records) - 1))
        record = payload.records[index]
        self._position.setText(self._position_text(index))

        if record.ccd_path is not None and record.ccd_path.is_file():
            try:
                sample = load_capture(record.ccd_path, index=record.index)
            except _LOAD_ERRORS as exc:
                self._render_no_ccd(payload, record, f"兄弟 npz 读取失败: {exc}")
                return
            self._sample = sample
            self._render_capture(sample, record)
            return
        self._sample = None
        missing = (
            f"归档中的 CCD 帧缺失: {record.ccd_path.name}"
            if record.ccd_path is not None
            else None
        )
        self._render_no_ccd(payload, record, missing)

    def _show_error(self, path: Path, exc: BaseException) -> None:
        """坏 pkl / npz 只上屏一行红字, 弹窗继续可用 (不抛、不关窗)。"""
        logger.warning("归档读取失败 {}: {}", path, exc)
        self._sample = None
        self._phase_full = None
        self._set_video_placeholder("读取失败")
        self._profiles.clear()
        self._phase.setText("相位不可用")
        self._meta.setText("")
        self._message.setText(
            f'<span style="color:#c0392b">读取失败: {path.name} — {exc}</span>'
        )
        self._message.setTextFormat(Qt.TextFormat.RichText)

    def _render_capture(self, sample: CaptureData, record: ArchiveRecord) -> None:
        self._frame_caption.setText("CCD 画面 (raw_frames 多帧平均)")
        self._profile_caption.setText("X / Y 剖面 (高斯拟合)")
        self._phase_caption.setText("SLM 相位预览 (panel_phase · twilight)")
        metrics = self._analyze(sample.raw_frame)
        self._video.set_frame(sample.display_frame, metrics, fps=0.0)
        self._profiles.update_metrics(metrics)

        self._phase_full = _to_pixmap(sample.phase_rgb)
        if self._phase_full.isNull():
            self._phase.clear()
            self._phase.setText("相位不可用")
        else:
            self._apply_phase_pixmap()

        top, bottom, left, right = sample.roi
        self._meta.setText(
            "显示: raw_frames "
            f"{sample.frame_count} 帧平均 · 裁剪 ROI y{top}..{bottom} / "
            f"x{left}..{right} · 峰值 {sample.raw_peak:.0f} · "
            f"均值 {sample.raw_mean:.1f} (亮暗按 1%–99.9% 分位拉伸上屏)"
        )
        if metrics is not None:
            self._meta.setText(
                self._meta.text() + f" · 质心 ({metrics.cx:.1f}, {metrics.cy:.1f}) px"
            )
        extra = _format_metrics(record.metrics, _DISPLAY_KEYS["capture_records"])
        if extra:
            self._meta.setText(self._meta.text() + f" · {record.label} · {extra}")
        self._message.setTextFormat(Qt.TextFormat.PlainText)
        self._message.setText("")

    def _render_no_ccd(
        self, payload: ArchivePayload, record: ArchiveRecord, reason: str | None
    ) -> None:
        """无 CCD 帧: 相位预览 (若有) + CCD 占位 + 指标行文, 绝不抛异常。"""
        notice = reason or payload.notice or _NOTICE_UNKNOWN
        self._frame_caption.setText("CCD 画面 (不可用)")
        self._profile_caption.setText("X / Y 剖面 (不可用)")
        self._video.clear()
        self._profiles.clear()
        self._set_video_placeholder(notice)

        phase = record.phase
        if phase is not None and np.asarray(phase).ndim == 2:
            self._phase_full = _to_pixmap(phase_to_preview(_phase_array_to_gray(phase)))
            if not self._phase_full.isNull():
                self._phase_caption.setText("SLM 相位预览 (迭代 phase · twilight)")
                self._apply_phase_pixmap()
            else:
                self._phase.setText("相位不可用")
        else:
            self._phase_full = None
            self._phase.setText(notice)
            self._phase_caption.setText("SLM 相位预览 (不可用)")

        keys = _DISPLAY_KEYS.get(payload.flavor, ())
        line = record.label
        extra = _format_metrics(record.metrics, keys)
        if extra:
            line += f" · {extra}"
        self._meta.setText(line)
        self._message.setTextFormat(Qt.TextFormat.PlainText)
        self._message.setText(notice)

    @staticmethod
    def _analyze(frame: np.ndarray):
        """剖面/质心分析 (大画面先 2x2 降采样, 坐标再还原)。"""
        from ao_shaping.gui.ccd.beam_quality.analyzer import (
            analyze_frame,
            bin_2x2,
        )

        if frame.size == 0:
            return None
        if min(frame.shape) >= 400:
            return analyze_frame(bin_2x2(frame)).rescaled(2.0)
        return analyze_frame(frame)

    def _apply_phase_pixmap(self) -> None:
        if self._phase_full is None or self._phase_full.isNull():
            return
        target = self._phase.size()
        if target.width() < 2 or target.height() < 2:
            target = QSize(480, 360)
        self._phase.setPixmap(
            self._phase_full.scaled(
                target,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    # ------------------------------------------------------------ 交互
    def _on_slider(self, value: int) -> None:
        if not self._record_count():
            return
        self._index = max(0, min(self._record_count() - 1, int(value)))
        self._position.setText(self._position_text(self._index))
        self._timer.start()

    def _on_prev(self) -> None:
        self.set_index(self._index - 1)

    def _on_next(self) -> None:
        self.set_index(self._index + 1)

    def resizeEvent(self, event: QResizeEvent) -> None:  # Qt 覆写点
        super().resizeEvent(event)
        self._apply_phase_pixmap()


def open_archive_replay(
    parent: QWidget | None,
    root: str | Path | None = None,
    preselect: str | Path | None = None,
) -> ArchiveReplayDialog:
    """打开 (或复用) 归档回放弹窗, 返回它。

    ``root`` 默认 :data:`DEBUG_ROOT` (整个 debug 根, 递归扫所有历史 pkl)。
    ``preselect`` 传某个会话目录时, 会优先选中该会话下的 ``capture_records.pkl``。
    弹窗挂在 ``parent`` 上并记在 ``parent._archive_replay_dialog``, 页面不必自己
    持有引用; 再次调用会先关掉旧窗, 避免叠出多个。
    """
    base = Path(root) if root is not None else DEBUG_ROOT
    existing = getattr(parent, "_archive_replay_dialog", None)
    if isinstance(existing, ArchiveReplayDialog):
        existing.close()
        existing.deleteLater()
    dialog = ArchiveReplayDialog(base, parent, preselect=preselect)
    if parent is not None:
        setattr(parent, "_archive_replay_dialog", dialog)
    dialog.show()
    dialog.raise_()
    return dialog
