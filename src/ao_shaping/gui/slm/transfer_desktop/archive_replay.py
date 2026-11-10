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
  一条一次迭代, 带 ``phase`` (弧度 ndarray)、``terms`` (loss / ee_roi / ...), 本身
  **不内嵌 CCD 帧**, 但当它与同目录的 ``capture_records.pkl`` + 兄弟 ``capture_*.npz``
  一起落盘时, 回放按 ``(shape, block)`` join 到块首的实测 CCD 画面 (见
  :func:`_sibling_ccd_by_block`)。
* ``iterations.pkl`` (拟合口味, ``fit/``) —— 一条一次迭代, 带 ``loss`` / ``parameters``,
  **既无相位也无 CCD 帧**。

本模块把三者都列进左侧归档表格 (整个 debug 根递归扫, 按修改时间从新到旧), 点谁才
``pickle.load`` 谁 (98 MB 的相位文件要几秒, 扫描阶段绝不打开 pkl)。三种口味都**能回放**:
有 CCD 就出三视图, 只有相位就出相位预览 + CCD 占位, 只有拟合指标就出行文提示 —— 一个
损坏文件也只是在窗口里显示一行红字, 弹窗不抛异常、不关窗。

列表之外还有两件事
------------------
* **归档**: 每行一个「归档」按钮, 确认后把这个 ``*.pkl`` **移动**到
  ``<根>/dump/`` 下 (保留原有相对路径 —— 几百个归档只有两个文件名, 平铺会互相覆盖;
  兄弟 ``capture_*.npz`` 留在原处)。回收目录不参与扫描, 归档完该行即消失。
* **自动播放**: 一个 fps 输入框 + 播放/停止开关, 用独立定时器按 ``1000/fps`` 毫秒
  一拍循环推进记录 (走到最后一条回到第 0 条), 每拍直接渲染, 不走滑块防抖。

为什么不用 ``pattern_controls._phase_to_preview``
------------------------------------------------
那个函数本身是纯 numpy + matplotlib, 但它的模块顶部 ``import`` 了
:class:`~ao_shaping.drivers.slm.santec.Santec` —— 为了一个色图把 SLM 驱动拉进
无设备的回放进程不合适。所以这里给出同阈值的 :func:`phase_to_preview`
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
import re
import shutil
import zipfile
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path, PurePosixPath

import numpy as np
from loguru import logger
from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtGui import QCloseEvent, QImage, QPixmap, QResizeEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSlider,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ao_shaping.utils.io import paths

__all__ = [
    "ARCHIVE_COLUMNS",
    "DEBUG_ROOT",
    "DUMP_DIRNAME",
    "FPS_DEFAULT",
    "FPS_MAX",
    "FPS_MIN",
    "ArchiveEntry",
    "ArchivePayload",
    "ArchiveRecord",
    "ArchiveReplayDialog",
    "CaptureData",
    "archive_to_dump",
    "debug_root",
    "discover_archives",
    "discover_captures",
    "dump_root_for",
    "format_mtime",
    "fps_to_interval_ms",
    "is_own_session_archive",
    "load_archive",
    "load_capture",
    "open_archive_replay",
    "phase_to_preview",
]

#: 归档回放扫描根目录 (与 ``history_browser.DEFAULT_ROOT`` / ``params.plan_session_dir``
#: 同一约定: 源码检出下相对当前工作目录)。所有历史会话都在这个根下面, 递归找 ``*.pkl``。
#: 这是字面量约定; 实际取值走 :func:`debug_root`（打包成 exe 后落到 exe 旁的 ``data/``）。
DEBUG_ROOT = Path("data") / "debug" / "slm_transfer"
#: 未被打补丁时 :data:`DEBUG_ROOT` 的原始值（身份哨兵）。
_DEFAULT_DEBUG_ROOT = DEBUG_ROOT


def debug_root() -> Path:
    """归档回放扫描根目录（**调用时**求值）。

    顺序: 被覆盖的模块常量（测试 / 调用方 monkeypatch）→ 数据目录解析器。源码检出下
    两者逐位相同; 冻结成 exe 后真正的约定是 ``<exe 目录>/data/debug/slm_transfer``。
    """
    if DEBUG_ROOT != _DEFAULT_DEBUG_ROOT:
        return DEBUG_ROOT
    return paths.data_subdir("debug", "slm_transfer")

#: 归档 capture 的文件名通配符 (只认 ``anchored/`` 下这一种; ``discover_captures`` 用)。
CAPTURE_GLOB = "capture_*.npz"

#: 所有历史归档的通配符: 只要满足 ``*.pkl`` 都能回放。
ARCHIVE_GLOB = "*.pkl"

#: 归档回收目录名: 行内「归档」按钮把 pkl **移动**到 ``<根>/dump`` 下, 且保留原有
#: 相对路径 —— 几百个归档只有两个文件名 (``capture_records.pkl`` / ``iterations.pkl``),
#: 平铺进一个目录会互相覆盖。
DUMP_DIRNAME = "dump"

#: 自动播放的 fps 区间与默认值 (一拍 = :func:`fps_to_interval_ms` 毫秒)。
FPS_MIN = 0.5
FPS_MAX = 30.0
FPS_DEFAULT = 2.0

#: 时间列的展示格式。
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"

#: 归档列表的四列表头。
ARCHIVE_COLUMNS: tuple[str, ...] = ("名称", "时间", "大小", "操作")
_COL_NAME, _COL_TIME, _COL_SIZE, _COL_ACTION = range(len(ARCHIVE_COLUMNS))

#: 自动播放按钮的两个状态文案。
_PLAY_LABEL = "▶ 自动播放"
_STOP_LABEL = "■ 停止"

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
_ARCHIVE_LIST_WIDTH = 460

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
    #: 原始 SLM 相位灰度 (uint16, 值域 0..1023, 取自 ``panel_gray``)。
    phase_gray: np.ndarray
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


def is_in_dump(relative: str) -> bool:
    """相对路径是否落在回收目录 ``dump/`` 下 (已归档的 pkl 不再出现在列表里)。"""
    head = relative.replace("\\", "/").split("/", 1)[0]
    return head == DUMP_DIRNAME


#: 本程序 (迁移学习桌面页) 会话目录名的模式: ``YYYYMMDD_HHMMSS`` 或同秒冲突时
#: 的 ``YYYYMMDD_HHMMSS-N`` —— 与 :func:`ao_shaping.gui.slm.transfer_desktop.
#: params.plan_session_dir` 的产出一致。CLI / campaign 等其他来源用的是
#: ``physical_20261102c`` / ``multishape_nogs_20261104`` 这类带名字前缀的目录,
#: 不匹配这个模式。「只看本程序保存的归档」开关按它区分。
_OWN_SESSION_RE = re.compile(r"^\d{8}_\d{6}(-\d+)?$")


def is_own_session_archive(relative: str) -> bool:
    """归档是否落在本程序 (``plan_session_dir`` 时间戳目录) 保存的会话里。

    只看相对路径**第一段** (会话目录名), 不打开任何文件 —— 与扫描阶段「绝不
    解码」的约定一致。dump/ 下的路径不会出现在列表里, 这里不特殊处理。
    """
    head = relative.replace("\\", "/").split("/", 1)[0]
    return _OWN_SESSION_RE.match(head) is not None


def discover_archives(root: str | Path) -> list[ArchiveEntry]:
    """递归列出 ``root`` 下所有 ``*.pkl``, 按修改时间**从新到旧** (同 mtime 按相对路径升序)。

    只做文件系统枚举 (``rglob`` + ``stat``), **绝不打开任何 pkl** —— 根目录里有几百个
    pkl / 几十 GB, 单个相位 pkl 可达 98 MB, 扫描阶段解码会卡死界面。根目录不存在 /
    不是目录时返回 ``[]`` (界面显示提示, 不抛异常)。

    回收目录 ``<根>/dump`` 下的 pkl 会被跳过: 它们是行内「归档」按钮搬过去的,
    不该再出现在待归档列表里。
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
        if is_in_dump(relative):
            continue
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

    与 ``pattern_controls._phase_to_preview`` 同阈值: 先除以面板满量程并钳到
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
        phase_gray=np.asarray(gray),
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


def _sibling_ccd_by_block(entry: ArchiveEntry) -> dict[tuple[str, int], Path]:
    """把同目录 ``capture_records.pkl`` 的 CCD 帧按 ``(shape, block)`` 索引成表。

    相位回放 (``iterations.pkl``) 本身不带 CCD 帧 —— 但 2026-11-04 起一套迁移会话
    往往在**同一个目录**里把 ``iterations.pkl`` 与 ``capture_records.pkl`` + 兄弟
    ``capture_*.npz`` 一起落盘: 前者是每次相位更新, 后者是每块实测的 CCD 画面。
    缺 CCD 的相位口味本来只能显示「只迭代相位」占位, 明明照片就在隔壁躺着。

    这里按 ``({shape}_{block}_A1)`` 锚记录找回块首的 CCD npz (join 键与
    :func:`recording.block_records` 一致, 全库 932 块实测 100% 命中)。任何读取失败都
    退回空表 —— 回放不能因为缺一张图就抛异常, 由原来的占位提示接管。
    """
    sibling = entry.path.with_name("capture_records.pkl")
    if not sibling.is_file():
        return {}
    try:
        with sibling.open("rb") as stream:
            raw = pickle.load(stream)
    except _LOAD_ERRORS:
        return {}
    out: dict[tuple[str, int], Path] = {}
    recs = _as_records(raw)
    files = {r.get("label"): r.get("file") for r in recs}
    for label, name in files.items():
        if not isinstance(label, str) or not name:
            continue
        shape, _, rest = label.partition("_")
        blk_part, sep, _suffix = rest.partition("_")
        if sep != "_" or not blk_part.isdigit():
            continue
        blk = int(blk_part)
        out.setdefault((shape, blk), entry.path.parent / str(name))
    return out


def _payload_iterations_phase(
    entry: ArchiveEntry, records: list[dict]
) -> ArchivePayload:
    ccd_by_block = _sibling_ccd_by_block(entry)
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
        ccd = None
        shape = rec.get("shape")
        block = rec.get("block")
        if isinstance(shape, str) and isinstance(block, int):
            ccd = ccd_by_block.get((shape, block))
        out.append(ArchiveRecord(i, label, ccd, arr, metrics))
    if ccd_by_block:
        return ArchivePayload(entry, out, "iterations_phase", "")
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


def format_mtime(mtime: float) -> str:
    """时间列文案: ``2026-11-02 13:04:05`` (越界 / 坏时间戳给 ``—``)。"""
    try:
        return datetime.fromtimestamp(float(mtime)).strftime(TIME_FORMAT)
    except (OSError, OverflowError, ValueError):
        return "—"


def fps_to_interval_ms(fps: float) -> int:
    """自动播放一拍的毫秒数 (``fps`` 越大间隔越短, 最小 1 ms 保证还能画)。"""
    return max(1, int(round(1000.0 / max(0.01, float(fps)))))


def dump_root_for(root: str | Path) -> Path:
    """``root`` 下的归档回收目录 (``<root>/dump``)。"""
    return Path(root) / DUMP_DIRNAME


def archive_root_of(entry: ArchiveEntry) -> Path:
    """由归档条目反推它当初的扫描根 (``path`` 去掉 ``relative`` 那几层)。"""
    parts = [
        part
        for part in entry.relative.replace("\\", "/").split("/")
        if part and part != "."
    ]
    if not parts or ".." in parts:
        return Path(entry.path).parent
    return Path(entry.path).parents[len(parts) - 1]


def archive_to_dump(entry: ArchiveEntry, dump_root: str | Path | None = None) -> Path:
    """把一个归档 **移动** 到回收目录并返回目标路径 (保留原有相对路径)。

    只搬这一个 ``*.pkl`` —— 兄弟 ``capture_*.npz`` 留在原处: 一行代表的就是这一个
    归档, 而 discovery 只认 pkl, 搬走它该行就消失了 (想复原把文件搬回去即可)。

    * 目标已存在 → :class:`FileExistsError` (**绝不覆盖**);
    * 相对路径不可用 → :class:`ValueError`;
    * ``shutil.move`` 失败 → 原样抛 :class:`OSError` / :class:`shutil.Error`,
      由界面接住上屏一行提示, 列表保留该行。

    纯文件系统操作, 不解码 pkl, 可单独测。
    """
    dest_root = (
        Path(dump_root)
        if dump_root is not None
        else dump_root_for(archive_root_of(entry))
    )
    relative = PurePosixPath(entry.relative.replace("\\", "/"))
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"归档相对路径不可用: {entry.relative!r}")
    dest = dest_root / relative
    if dest.exists():
        raise FileExistsError(f"回收目录里已经有同名归档: {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.move(str(entry.path), str(dest))
    except (OSError, shutil.Error) as exc:  # pragma: no cover - 依赖文件系统
        logger.warning(f"归档移动失败: {entry.path} -> {dest}: {exc}")
        raise
    logger.info(f"已归档 (移动): {entry.relative} -> {dest}")
    return dest


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
# ---------------------------------------------------------------- 导出 (纯逻辑, 不碰硬件)
_EXPORT_MAX_GRAY = 1023


def phase_to_gray(phase):
    arr = np.asarray(phase, dtype=np.float64)
    wrapped = np.mod(np.nan_to_num(arr), 2.0 * np.pi)
    return np.rint(wrapped / (2.0 * np.pi) * float(_EXPORT_MAX_GRAY)).astype(np.uint16)


def gray_to_phase(gray):
    arr = np.asarray(gray, dtype=np.float64)
    return arr / float(_EXPORT_MAX_GRAY) * 2.0 * np.pi


def _rad_to_gray(phase):
    return phase_to_gray(phase)


def _fmt_phase_value(value, radians=False):
    if radians:
        return repr(float(value))
    return str(int(round(float(value))))


def write_phase_csv(dest, data, radians=False):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(data)
    if arr.ndim != 2:
        raise ValueError("phase array must be 2D")
    height, width = int(arr.shape[0]), int(arr.shape[1])
    lines = []
    lines.append("Y/X," + ",".join(str(j) for j in range(width)))
    for i in range(height):
        vals = [_fmt_phase_value(v, radians=radians) for v in arr[i].ravel()]
        lines.append(str(i) + "," + ",".join(vals))
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dest


def export_phase_csv(dest, data, radians=False):
    return write_phase_csv(dest, data, radians=radians)


def export_phase_slm_csv(dest, data):
    arr = np.asarray(data)
    if np.issubdtype(arr.dtype, np.integer):
        gray = arr
    else:
        gray = phase_to_gray(arr)
    rad = gray_to_phase(gray)
    return write_phase_csv(dest, rad, radians=True)


def export_phase_radians_csv(dest, data):
    return write_phase_csv(dest, np.asarray(data, dtype=np.float64), radians=True)


def _arr_to_uint8(frame):
    arr = np.asarray(frame)
    if np.issubdtype(arr.dtype, np.floating):
        if arr.size == 0:
            return np.zeros(arr.shape, dtype=np.uint8)
        lo, hi = np.percentile(arr, (1.0, 99.9))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
            if hi <= lo:
                hi = lo + 1.0
        out = np.clip((arr - lo) / (hi - lo) * 255.0, 0.0, 255.0)
        return out.astype(np.uint8)
    if arr.dtype == np.uint8:
        return arr
    if np.issubdtype(arr.dtype, np.integer):
        if arr.size and int(arr.max()) > 255:
            return np.clip(arr.astype(np.float64) / 65535.0 * 255.0, 0, 255).astype(np.uint8)
        return arr.astype(np.uint8)
    return arr.astype(np.uint8)


def write_ccd_image(dest, frame, fmt=None):
    from PIL import Image
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    arr = _arr_to_uint8(frame)
    if arr.ndim == 2:
        img = Image.fromarray(arr, mode="L")
    else:
        img = Image.fromarray(arr)
    if fmt is None:
        fmt = dest.suffix.lstrip(".").lower() or "png"
    fmt = fmt.lower()
    if fmt in ("tif", "tiff"):
        img.save(dest, format="TIFF")
    elif fmt == "png":
        img.save(dest, format="PNG")
    else:
        img.save(dest)
    return dest


def export_ccd_png(dest, frame):
    return write_ccd_image(dest, frame, fmt="png")


def export_ccd_tiff(dest, frame):
    return write_ccd_image(dest, frame, fmt="tiff")


def export_ccd_image(dest, frame, fmt="png"):
    return write_ccd_image(dest, frame, fmt=fmt)
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
        #: 表格行 → 归档 (过滤只是隐藏行, 所以行序恒等于 ``_archives`` 的顺序)。
        self._rows: list[ArchiveEntry] = list(self._archives)
        self._record_counts: dict[Path, int] = {}
        self._payload: ArchivePayload | None = None
        self._index = 0
        self._sample: CaptureData | None = None
        self._sample_mtime: float | None = None
        self._phase_full: QPixmap | None = None
        self._active_row = -1

        # 滑块防抖: 拖动时不读盘, 停手 140 ms 才加载。
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(_SLIDER_DEBOUNCE_MS)
        self._timer.timeout.connect(self._reload_current)

        # 自动播放: 单独的定时器, 每拍直接渲染 (不走防抖, 否则高 fps 会被饿死)。
        self._play_timer = QTimer(self)
        self._play_timer.setInterval(fps_to_interval_ms(FPS_DEFAULT))
        self._play_timer.timeout.connect(self._on_play_tick)

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

        # 自动播放: fps + 播放/停止 (独立定时器, 见 ``_play_timer``)。
        controls.addSpacing(12)
        controls.addWidget(QLabel("fps", self))
        self._fps = QDoubleSpinBox(self)
        self._fps.setRange(FPS_MIN, FPS_MAX)
        self._fps.setDecimals(1)
        self._fps.setSingleStep(0.5)
        self._fps.setValue(FPS_DEFAULT)
        self._fps.setFixedWidth(88)
        self._fps.setToolTip(
            f"自动播放速度 ({FPS_MIN}–{FPS_MAX} fps), 播放中改动下一拍生效"
        )
        self._fps.valueChanged.connect(self._on_fps_changed)
        self._play = QPushButton(_PLAY_LABEL, self)
        self._play.setCheckable(True)
        self._play.setToolTip("按当前 fps 循环播放本归档的所有记录")
        self._play.toggled.connect(self._on_play_toggled)
        controls.addWidget(self._fps)
        controls.addWidget(self._play)
        root.addLayout(controls)

        # 导出按钮行 (选中帧的相位 / CCD 图)
        export_row = QHBoxLayout()
        self._btn_phase_slm = QPushButton("导出相位 (SLM灰度)", self)
        self._btn_phase_rad = QPushButton("导出相位 (弧度CSV)", self)
        self._btn_ccd_png = QPushButton("导出CCD (PNG)", self)
        self._btn_ccd_tiff = QPushButton("导出CCD (TIFF)", self)
        for b in (self._btn_phase_slm, self._btn_phase_rad, self._btn_ccd_png, self._btn_ccd_tiff):
            export_row.addWidget(b)
        self._btn_phase_slm.clicked.connect(self._on_export_phase_slm)
        self._btn_phase_rad.clicked.connect(self._on_export_phase_radians)
        self._btn_ccd_png.clicked.connect(self._on_export_ccd_png)
        self._btn_ccd_tiff.clicked.connect(self._on_export_ccd_tiff)
        root.addLayout(export_row)

        outer = QSplitter(Qt.Orientation.Horizontal, self)

        # 左: 归档列表面板 (过滤框 + 列表)。
        archive_panel = QWidget(outer)
        archive_layout = QVBoxLayout(archive_panel)
        archive_layout.setContentsMargins(0, 0, 0, 0)
        self._filter = QLineEdit(archive_panel)
        self._filter.setPlaceholderText("过滤归档 (路径 / 口味)…")
        self._filter.setClearButtonEnabled(True)
        self._filter.textChanged.connect(self._on_filter)
        self._table = QTableWidget(0, len(ARCHIVE_COLUMNS), archive_panel)
        self._table.setHorizontalHeaderLabels(list(ARCHIVE_COLUMNS))
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        # 排序关掉: 行序就是 discover_archives 的"从新到旧", 行号 ↔ 归档一一对应。
        self._table.setSortingEnabled(False)
        self._table.setAlternatingRowColors(True)
        self._table.setWordWrap(False)
        self._table.setMinimumWidth(_ARCHIVE_LIST_WIDTH)
        self._table.setStyleSheet("background-color: #12121a; color: #d8d8e0;")
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(_COL_NAME, QHeaderView.ResizeMode.Stretch)
        for column in (_COL_TIME, _COL_SIZE, _COL_ACTION):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        self._table.currentCellChanged.connect(self._on_current_cell_changed)
        archive_layout.addWidget(self._filter)
        # 「只看本程序保存的归档」: 只保留迁移学习桌面页自己跑出来的会话
        # (``plan_session_dir`` 的时间戳目录名), 把 CLI / campaign 的历史归档藏掉。
        self._own_only = QCheckBox("只看本程序保存的归档", archive_panel)
        self._own_only.setToolTip(
            "开 = 只列迁移学习桌面页保存的会话 (时间戳目录);\n"
            "关 = 递归列出整个 debug 根下所有历史归档"
        )
        self._own_only.toggled.connect(self._on_filter)
        archive_layout.addWidget(self._own_only)
        archive_layout.addWidget(self._table, 1)

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
        outer.setSizes([_ARCHIVE_LIST_WIDTH, 900])
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
    def _row_of(self, entry: ArchiveEntry) -> int:
        """表格里 ``entry`` 所在的行号 (已归档 / 被过滤掉时为 ``-1``)。"""
        for row, row_entry in enumerate(self._rows):
            if row_entry.path == entry.path:
                return row
        return -1

    def _set_row(self, row: int, entry: ArchiveEntry) -> None:
        """填一行四列: 名称 / 时间 / 大小 / 操作 (归档按钮)。"""
        flags = Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled
        name = QTableWidgetItem(entry.relative)
        # 395 个归档只有两个文件名, 所以名称列必须是完整相对路径; 绝对路径进 tooltip。
        name.setToolTip(f"{entry.path}\n[{entry.kind}]")
        name.setData(Qt.ItemDataRole.UserRole, float(entry.mtime))
        name.setFlags(flags)
        self._table.setItem(row, _COL_NAME, name)

        when = QTableWidgetItem(format_mtime(entry.mtime))
        when.setData(Qt.ItemDataRole.UserRole, float(entry.mtime))
        when.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        when.setFlags(flags)
        self._table.setItem(row, _COL_TIME, when)

        size = QTableWidgetItem(_human_size(entry.size_bytes))
        size.setData(Qt.ItemDataRole.UserRole, int(entry.size_bytes))
        size.setTextAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        size.setFlags(flags)
        self._table.setItem(row, _COL_SIZE, size)

        button = QPushButton("归档", self._table)
        button.setFixedWidth(64)
        button.setToolTip(
            f"把 {entry.relative} 移动到 {DUMP_DIRNAME}/ 下 (保留相对路径, 只搬这个 pkl)"
        )
        # partial 绑 entry 而不是行号: 行会因为过滤/归档而移动, entry 不会。
        button.clicked.connect(partial(self._on_dump_clicked, entry))
        self._table.setCellWidget(row, _COL_ACTION, button)

    def _rebuild_table(self) -> None:
        """按 ``_archives`` (从新到旧) 重建整张表, 一行一个归档。"""
        self._rows = list(self._archives)
        self._table.blockSignals(True)
        self._table.setRowCount(len(self._rows))
        for row, entry in enumerate(self._rows):
            self._set_row(row, entry)
        self._table.blockSignals(False)

    def _update_item_text(self, entry: ArchiveEntry, count: int) -> None:
        """加载完才知道记录数: 写进名称列的 tooltip (列宽留给路径)。"""
        self._record_counts[entry.path] = count
        row = self._row_of(entry)
        if row < 0:
            return
        item = self._table.item(row, _COL_NAME)
        if item is not None:
            item.setToolTip(f"{entry.path}\n[{entry.kind}] · {count} 条记录")

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
        self._update_header()
        self._rebuild_table()
        self._visible = list(self._rows)
        if not self._visible:
            self._show_no_archives("")
            return
        row = self._preselect_row(preselect)
        self._table.blockSignals(True)
        self._table.setCurrentCell(row, _COL_NAME)
        self._table.blockSignals(False)
        item = self._table.item(row, _COL_NAME)
        if item is not None:
            self._table.scrollToItem(item)
        self.select_archive(row)

    def _update_header(self) -> None:
        own_only = self._own_only.isChecked() if hasattr(self, "_own_only") else False
        scope = (
            f"本程序会话 {len(self._visible)} 个"
            if own_only
            else f"共 {len(self._archives)} 个"
        )
        self._header.setText(
            f"归档根目录: {self._root}\n"
            f"{scope} pkl 归档 (递归扫描, 按修改时间从新到旧) "
            f"· 归档按钮把 pkl 移到 {DUMP_DIRNAME}/ 下 · 只读, 不打开相机 / SLM"
        )

    def apply_filter(self, text: str) -> None:
        """按 ``text`` (大小写不敏感, 匹配相对路径或口味) 过滤左侧列表。"""
        self._filter.blockSignals(True)
        self._filter.setText(text)
        self._filter.blockSignals(False)
        self._on_filter(text)

    def _on_filter(self, *args: object) -> None:
        # textChanged 带文本参数、toggled 带布尔参数 —— 统一忽略, 现读控件。
        needle = (self._filter.text() or "").strip().lower()
        own_only = self._own_only.isChecked()
        self._table.blockSignals(True)
        for row, entry in enumerate(self._rows):
            hidden = bool(needle) and not (
                needle in entry.relative.lower() or needle in entry.kind.lower()
            )
            if not hidden and own_only and not is_own_session_archive(entry.relative):
                hidden = True
            self._table.setRowHidden(row, hidden)
        self._table.blockSignals(False)
        self._visible = [
            entry
            for row, entry in enumerate(self._rows)
            if not self._table.isRowHidden(row)
        ]
        if not self._visible:
            self._show_no_archives(self._filter.text())
            return
        self._update_header()
        row = 0
        active = self._payload.entry if self._payload is not None else None
        if active is not None:
            for index, entry in enumerate(self._visible):
                if entry.path == active.path:
                    row = index
                    break
        self._table.blockSignals(True)
        self._table.setCurrentCell(row, _COL_NAME)
        self._table.blockSignals(False)
        if self._payload is None or self._payload.entry.path != self._visible[row].path:
            self.select_archive(row)

    def _on_current_cell_changed(
        self, row: int, _column: int, _prev_row: int, _prev_column: int
    ) -> None:
        self._on_archive_activated(row)

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
        # 换归档必须停自动播放: 记录条数与下标都换了。
        self.stop_playback()
        self._active_row = row
        self._set_loading(entry)
        self._load_and_render(entry)

    # ------------------------------------------------------------ 归档 (移到 dump)
    def _confirm_dump(self, entry: ArchiveEntry) -> bool:
        """归档确认框 (默认「否」): 单列一个函数, 测试可以整只换掉它。"""
        dest = dump_root_for(self._root) / PurePosixPath(
            entry.relative.replace("\\", "/")
        )
        answer = QMessageBox.question(
            self,
            "归档确认",
            "把这个归档「移动」(不是复制、不是删除) 到回收目录?\n\n"
            f"源文件: {entry.relative}\n"
            f"目标: {dest}\n\n"
            f"只移动这一个 pkl [{entry.kind} · {_human_size(entry.size_bytes)}], "
            "同目录的兄弟 npz 留在原处; 想复原把它搬回原路径即可。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

    def _on_dump_clicked(self, entry: ArchiveEntry) -> None:
        """行内「归档」按钮: 确认 → :func:`archive_to_dump` → 从列表摘掉这一行。"""
        row = self._row_of(entry)
        if row < 0:
            return
        if not self._confirm_dump(entry):
            self._message.setText(f"已取消归档: {entry.relative}")
            return
        try:
            dest = archive_to_dump(entry, dump_root_for(self._root))
        except FileExistsError as exc:
            QMessageBox.warning(self, "归档未执行", str(exc))
            self._message.setText(f"归档未执行 (目标已存在): {exc}")
            return
        except (OSError, shutil.Error, ValueError) as exc:
            logger.warning(f"归档失败: {entry.path}: {exc}")
            QMessageBox.warning(self, "归档失败", f"{entry.relative} 移动失败: {exc}")
            self._message.setText(f"归档失败, 该归档仍在列表里: {exc}")
            return
        self._drop_archived_row(entry, row, dest)

    def _drop_archived_row(self, entry: ArchiveEntry, row: int, dest: Path) -> None:
        was_current = (
            self._payload is not None and self._payload.entry.path == entry.path
        )
        self._archives = [item for item in self._archives if item.path != entry.path]
        self._rows = [item for item in self._rows if item.path != entry.path]
        self._visible = [item for item in self._visible if item.path != entry.path]
        # 摘行时屏蔽信号: 否则 currentCellChanged 会把下一行当成"用户点选"重新加载。
        self._table.blockSignals(True)
        self._table.removeRow(row)
        self._table.blockSignals(False)
        self._update_header()
        if not was_current:
            if self._payload is not None:
                self._active_row = self._row_of(self._payload.entry)
            self._message.setText(f"已归档: {entry.relative} → {dest}")
            return
        self.stop_playback()
        self._payload = None
        self._sample = None
        self._phase_full = None
        self._active_row = -1
        self._set_slider_range(0)
        self._set_video_placeholder("当前归档已移入回收目录")
        self._profiles.clear()
        self._phase.setText("无相位预览")
        self._meta.setText("")
        self._message.setText(f"已归档并移出列表: {entry.relative} → {dest}")

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
            self._play.setEnabled(False)
            self._position.setText("0 / 0")
            return
        self._slider.setEnabled(True)
        self._prev.setEnabled(True)
        self._next.setEnabled(True)
        self._play.setEnabled(True)
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
            # 块内多条迭代共用同一兄弟 npz: 路径与 mtime 没变就直接复用已解码的
            # sample (只重画逐拍相位), 免得每拍重读几十 MB 把播放拖成幻灯片。
            try:
                mtime = record.ccd_path.stat().st_mtime
            except OSError:
                mtime = None
            if (
                self._sample is not None
                and self._sample.path == record.ccd_path
                and mtime is not None
                and self._sample_mtime == mtime
            ):
                self._render_capture(self._sample, record)
                return
            try:
                sample = load_capture(record.ccd_path, index=record.index)
            except _LOAD_ERRORS as exc:
                self._render_no_ccd(payload, record, f"兄弟 npz 读取失败: {exc}")
                self._sample = None
                self._sample_mtime = None
                return
            self._sample = sample
            self._sample_mtime = mtime
            self._render_capture(sample, record)
            return
        self._sample = None
        self._sample_mtime = None
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
        metrics = self._analyze(sample.raw_frame)
        self._video.set_frame(sample.display_frame, metrics, fps=0.0)
        self._profiles.update_metrics(metrics)

        # 相位预览优先画**逐条迭代相位** (与无 CCD 分支同口径): 兄弟 npz 里的
        # panel_phase 是块级采集那一刻的相位, 一个块 ~10 条迭代共用同一份 —— 画
        # 它的话块内播放时相位和 CCD 一起冻住, 看着就像「没法动态播放」。迭代
        # 相位每条都在变, 画面逐拍推进; CCD 仍按块 (实测每块只有一张, 不编造)。
        phase = record.phase
        if phase is not None and np.asarray(phase).ndim == 2:
            self._phase_caption.setText("SLM 相位预览 (迭代 phase · twilight)")
            self._phase_full = _to_pixmap(
                phase_to_preview(_phase_array_to_gray(phase))
            )
        else:
            self._phase_caption.setText("SLM 相位预览 (panel_phase · twilight)")
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
        # 手动拖动 = 接管播放。
        self.stop_playback()
        self._index = max(0, min(self._record_count() - 1, int(value)))
        self._position.setText(self._position_text(self._index))
        self._timer.start()

    def _on_prev(self) -> None:
        self.stop_playback()
        self.set_index(self._index - 1)

    def _on_next(self) -> None:
        self.stop_playback()
        self.set_index(self._index + 1)

    # ------------------------------------------------------------ 自动播放
    def is_playing(self) -> bool:
        """自动播放是否正在跑。"""
        return self._play_timer.isActive()

    def fps(self) -> float:
        """当前 fps 设定。"""
        return float(self._fps.value())

    def set_fps(self, fps: float) -> None:
        """设定 fps (钳到 :data:`FPS_MIN`–:data:`FPS_MAX`; 播放中下一拍生效)。"""
        self._fps.setValue(max(FPS_MIN, min(FPS_MAX, float(fps))))

    def start_playback(self) -> None:
        """按当前 fps 起播 (没有记录时不起播, 按钮弹回)。"""
        if not self._record_count():
            self.stop_playback()
            self._message.setText("当前归档没有可播放的记录")
            return
        self._play_timer.setInterval(fps_to_interval_ms(self._fps.value()))
        self._play_timer.start()
        self._refresh_play_button()

    def stop_playback(self) -> None:
        """停播并把按钮弹回「▶ 自动播放」(换归档 / 手动操作 / 关窗都走这里)。"""
        if self._play_timer.isActive():
            self._play_timer.stop()
        if self._play.isChecked():
            self._play.blockSignals(True)
            self._play.setChecked(False)
            self._play.blockSignals(False)
        self._refresh_play_button()

    def _refresh_play_button(self) -> None:
        self._play.setText(_STOP_LABEL if self.is_playing() else _PLAY_LABEL)

    def _on_play_toggled(self, checked: bool) -> None:
        if checked:
            self.start_playback()
        else:
            self.stop_playback()

    def _on_fps_changed(self, value: float) -> None:
        self._play_timer.setInterval(fps_to_interval_ms(value))

    def _on_play_tick(self) -> None:
        total = self._record_count()
        if not total:
            self.stop_playback()
            return
        # fps 改了就按新值走下一拍。
        self._play_timer.setInterval(fps_to_interval_ms(self._fps.value()))
        # 直接渲染: 走滑块防抖的话 fps 高一点就永远等不到那一拍。
        self.set_index((self._index + 1) % total)

    # ------------------------------------------------------------ 导出
    def _current_phase_gray(self) -> np.ndarray | None:
        """当前帧的原始 SLM 相位灰度 (uint16 0..1023); 无当前帧返回 ``None``。"""
        sample = self._sample
        if sample is None or sample.phase_gray.size == 0:
            return None
        return sample.phase_gray

    def _current_ccd(self) -> np.ndarray | None:
        """当前帧的原始 CCD 灰度 (ROI 裁剪后); 无当前帧返回 ``None``。"""
        sample = self._sample
        if sample is None or sample.raw_frame.size == 0:
            return None
        return sample.raw_frame

    def _pick_save_path(self, title: str, default_name: str, flt: str) -> Path | None:
        path, _ = QFileDialog.getSaveFileName(self, title, default_name, flt)
        if not path:
            return None
        return Path(path)

    def _on_export_phase_slm(self) -> None:
        gray = self._current_phase_gray()
        if gray is None:
            self._message.setText("没有已加载的相位帧, 无法导出 (SLM灰度)")
            return
        dest = self._pick_save_path(
            "导出相位 · SLM灰度阈值", "phase_slm.csv", "CSV 文件 (*.csv);;所有文件 (*)"
        )
        if dest is None:
            return
        export_phase_slm_csv(dest, gray)
        self._message.setText(f"已导出相位 (SLM灰度阈值): {Path(dest).name}")

    def _on_export_phase_radians(self) -> None:
        gray = self._current_phase_gray()
        if gray is None:
            self._message.setText("没有已加载的相位帧, 无法导出 (弧度CSV)")
            return
        dest = self._pick_save_path(
            "导出相位 · 弧度制", "phase_radians.csv", "CSV 文件 (*.csv);;所有文件 (*)"
        )
        if dest is None:
            return
        export_phase_radians_csv(dest, gray_to_phase(gray))
        self._message.setText(f"已导出相位 (弧度制): {Path(dest).name}")

    def _on_export_ccd_png(self) -> None:
        frame = self._current_ccd()
        if frame is None:
            self._message.setText("没有已加载的 CCD 帧, 无法导出 (PNG)")
            return
        dest = self._pick_save_path(
            "导出 CCD 帧 · PNG", "ccd.png", "PNG 图片 (*.png);;所有文件 (*)"
        )
        if dest is None:
            return
        export_ccd_png(dest, frame)
        self._message.setText(f"已导出 CCD (PNG): {Path(dest).name}")

    def _on_export_ccd_tiff(self) -> None:
        frame = self._current_ccd()
        if frame is None:
            self._message.setText("没有已加载的 CCD 帧, 无法导出 (TIFF)")
            return
        dest = self._pick_save_path(
            "导出 CCD 帧 · TIFF", "ccd.tiff", "TIFF 图片 (*.tif *.tiff);;所有文件 (*)"
        )
        if dest is None:
            return
        export_ccd_tiff(dest, frame)
        self._message.setText(f"已导出 CCD (TIFF): {Path(dest).name}")

    def closeEvent(self, event: QCloseEvent) -> None:  # Qt 覆写点
        self.stop_playback()
        self._timer.stop()
        super().closeEvent(event)

    def done(self, result: int) -> None:  # Qt 覆写点
        self.stop_playback()
        self._timer.stop()
        super().done(result)

    def resizeEvent(self, event: QResizeEvent) -> None:  # Qt 覆写点
        super().resizeEvent(event)
        self._apply_phase_pixmap()


def open_archive_replay(
    parent: QWidget | None,
    root: str | Path | None = None,
    preselect: str | Path | None = None,
) -> ArchiveReplayDialog:
    """打开 (或复用) 归档回放弹窗, 返回它。

    ``root`` 默认 :func:`debug_root` (整个 debug 根, 递归扫所有历史 pkl)。
    ``preselect`` 传某个会话目录时, 会优先选中该会话下的 ``capture_records.pkl``。
    弹窗挂在 ``parent`` 上并记在 ``parent._archive_replay_dialog``, 页面不必自己
    持有引用; 再次调用会先关掉旧窗, 避免叠出多个。
    """
    base = Path(root) if root is not None else debug_root()
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
