"""按相机序列号留存的暗帧库 (CCD 配置数据)。

为什么放进配置数据而不是 run 产物
------------------------------------
暗帧是**相机状态, 不是实验状态**: 它只取决于传感器、曝光与位深, 在相机温度
没有明显变化前跨 run 有效。放进某次 run 的 ``--out`` 目录, 每个脚本都得重采
同一张暗帧; 放进 ``data/ccd_configs`` 并按序列号分目录, 换一台相机不会串台,
同一台相机则可以被 :class:`~ao_shaping.drivers.ccd.base.BaseCamera` 直接匹配加载。

目录布局
--------
``<CCD_CONFIG_DIR>/ccd/<serial>/darks/``::

    manifest.json                 # 索引: schema / 相机元数据 / 各曝光条目
    dark-exp-<exposure>ms.npz     # ``mean`` (必有) 与 ``std`` (可选), float32

``CCD_CONFIG_DIR`` 经 :func:`ao_shaping.utils.io.paths.device_config_dir` 解析
(环境变量 → 配置文件 → ``data/ccd_configs``), 与 ``CCD_CONFIG`` 及
:func:`ao_shaping.drivers.ccd.common.select_exposure_from_flats` 的平场帧同源。
阵列压缩存储: 暗帧几乎恒定, 全幅 20 MB 的 float32 均值压缩后只需一小部分。

契约
----
* **manifest 是唯一索引**, 阵列文件绝不被 glob 出来 —— 手放一个 npz 不会被误认。
* 条目里的 ``exposure_ms`` 是**设备回读值**, 不是请求值 (issue #32: 请求值不是
  传感器真正跑的曝光); ``requested_exposure_ms`` 另存请求值供诊断。
* 加载不臆造数据: manifest 缺失 ⇒ 空集; schema 不符 ⇒ 报错; 阵列缺失 ⇒ 在
  真正取用该曝光时报错, 不静默跳过。
* 匹配必须显式: :meth:`DarkFrameSet.match` 按 ``rel_tol`` 判同曝光,
  :meth:`DarkFrameSet.nearest` 是调用方主动选择的退路, 默认不启用。
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

#: manifest schema 版本。字段语义变化时**必须**递增, 加载端据此拒绝旧格式,
#: 而不是按缺省值猜。
SCHEMA = "ao_shaping.ccd.dark_frames/v1"

#: 每个序列号目录下的暗帧子目录名。
DARK_DIR_NAME = "darks"
MANIFEST_NAME = "manifest.json"

_ARRAY_KEY = "mean"
_STD_KEY = "std"

#: 序列号里可能带路径分隔符或冒号 (Windows 驱动器/网卡派生 id), 目录名必须净化。
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ccd_config_dir() -> Path:
    """CCD 配置目录 (``data/ccd_configs/ccd``), 与 ``CCD_CONFIG`` 同一解析器。

    函数体内 import ``paths`` 是刻意的: ``ao_shaping.drivers.ccd.base`` 会
    import 本模块, 而 ``paths`` 会拖进 ``utils.io`` —— 放在模块作用域会让
    ``import ao_shaping.drivers.ccd.base`` 凭空读配置/连带 utils.io 的导入链。
    """
    from ao_shaping.utils.io import paths

    return paths.device_config_dir("ccd") / "ccd"


def serial_dir_name(serial: str) -> str:
    """序列号 → 合法的目录名 (幂等, 保证同一序列号总落在同一目录)。"""
    cleaned = _UNSAFE.sub("_", str(serial)).strip("._-")
    return cleaned or "unknown"


def dark_frames_dir(serial: str, *, root: str | Path | None = None) -> Path:
    """暗帧目录 ``<root>/<serial>/darks``; ``root`` 缺省为 :func:`ccd_config_dir`。"""
    base = Path(root) if root is not None else ccd_config_dir()
    return base / serial_dir_name(serial) / DARK_DIR_NAME


def _exposure_filename(exposure_ms: float) -> str:
    return f"dark-exp-{exposure_ms:g}ms.npz"


@dataclass(frozen=True)
class DarkFrameEntry:
    """manifest 里的一条暗帧索引 (不含阵列本体)。"""

    exposure_ms: float
    file: str
    frames: int
    shape: tuple[int, int]
    mean: float | None = None
    std: float | None = None
    peak: float | None = None
    minimum: float | None = None
    bit_depth: int | None = None
    full_scale: float | None = None
    captured_at: str | None = None
    requested_exposure_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "exposure_ms": float(self.exposure_ms),
            "requested_exposure_ms": (
                None
                if self.requested_exposure_ms is None
                else float(self.requested_exposure_ms)
            ),
            "file": self.file,
            "frames": int(self.frames),
            "shape": [int(self.shape[0]), int(self.shape[1])],
            "mean": None if self.mean is None else float(self.mean),
            "std": None if self.std is None else float(self.std),
            "peak": None if self.peak is None else float(self.peak),
            "min": None if self.minimum is None else float(self.minimum),
            "bit_depth": self.bit_depth,
            "full_scale": (
                None if self.full_scale is None else float(self.full_scale)
            ),
            "captured_at": self.captured_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DarkFrameEntry:
        exposure = float(data["exposure_ms"])
        if not np.isfinite(exposure) or exposure <= 0:
            raise ValueError(f"暗帧条目 exposure_ms 非法: {data.get('exposure_ms')!r}")
        file = str(data["file"])
        if not file or Path(file).name != file:
            raise ValueError(f"暗帧条目 file 必须是目录内文件名: {file!r}")
        shape_raw = data.get("shape")
        if not isinstance(shape_raw, (list, tuple)) or len(shape_raw) != 2:
            raise ValueError(f"暗帧条目 shape 必须是 [rows, cols]: {shape_raw!r}")
        shape = (int(shape_raw[0]), int(shape_raw[1]))
        if shape[0] <= 0 or shape[1] <= 0:
            raise ValueError(f"暗帧条目 shape 非法: {shape_raw!r}")
        frames = int(data.get("frames", 0))
        if frames <= 0:
            raise ValueError(f"暗帧条目 frames 必须 > 0: {data.get('frames')!r}")

        def _opt_float(key: str) -> float | None:
            value = data.get(key)
            return None if value is None else float(value)

        bit_depth = data.get("bit_depth")
        return cls(
            exposure_ms=exposure,
            file=file,
            frames=frames,
            shape=shape,
            mean=_opt_float("mean"),
            std=_opt_float("std"),
            peak=_opt_float("peak"),
            minimum=_opt_float("min"),
            bit_depth=None if bit_depth is None else int(bit_depth),
            full_scale=_opt_float("full_scale"),
            captured_at=(
                None if data.get("captured_at") is None else str(data["captured_at"])
            ),
            requested_exposure_ms=_opt_float("requested_exposure_ms"),
        )


@dataclass(frozen=True)
class DarkFrame:
    """一条被加载出来的暗帧 (阵列已读入内存)。"""

    exposure_ms: float
    mean: np.ndarray
    std: np.ndarray | None
    frames: int
    full_scale: float | None
    entry: DarkFrameEntry

    def subtract(
        self,
        frame: np.ndarray,
        *,
        clip: bool = False,
        dtype: np.dtype | type = np.float32,
    ) -> np.ndarray:
        """``frame - mean`` (float), 可选把负值钳到 0。

        ``clip`` 默认关闭: 暗帧本身不含光, 相减后的负值 (=读出噪声) 是真实
        信息; 只有下游要求非负 (如对数/比值) 时才该钳。
        """
        arr = np.asarray(frame)
        if arr.shape != self.mean.shape:
            raise ValueError(
                f"暗帧形状 {self.mean.shape} 与画面 {arr.shape} 不一致; "
                "换过 ROI 的暗帧不能跨 ROI 复用"
            )
        out = arr.astype(dtype, copy=False) - self.mean.astype(dtype, copy=False)
        return np.clip(out, 0.0, None) if clip else out


@dataclass(frozen=True)
class DarkFrameSet:
    """一台相机在某个配置目录下的全部暗帧索引。"""

    serial: str
    directory: Path
    camera: Mapping[str, Any]
    entries: tuple[DarkFrameEntry, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def exposures_ms(self) -> tuple[float, ...]:
        return tuple(entry.exposure_ms for entry in self.entries)

    def entry_for(
        self, exposure_ms: float, *, rel_tol: float = 0.02
    ) -> DarkFrameEntry | None:
        """找与 ``exposure_ms`` 相对误差在 ``rel_tol`` 内的条目 (不读阵列)。"""
        target = float(exposure_ms)
        if not np.isfinite(target) or target <= 0:
            raise ValueError(f"exposure_ms 必须是正有限值: {exposure_ms!r}")
        if rel_tol < 0:
            raise ValueError(f"rel_tol 不能为负: {rel_tol!r}")
        best: DarkFrameEntry | None = None
        best_delta = np.inf
        for entry in self.entries:
            delta = abs(entry.exposure_ms - target)
            if delta <= rel_tol * abs(target) and delta < best_delta:
                best, best_delta = entry, delta
        return best

    def match(self, exposure_ms: float, *, rel_tol: float = 0.02) -> DarkFrame | None:
        """取同曝光的暗帧; 没有落在 ``rel_tol`` 内的条目时返回 ``None``。"""
        entry = self.entry_for(exposure_ms, rel_tol=rel_tol)
        return None if entry is None else self._load(entry)

    def nearest(self, exposure_ms: float) -> DarkFrame | None:
        """取曝光最接近的一条 (调用方显式选择时才用, 不是隐式回退)。"""
        if not self.entries:
            return None
        target = float(exposure_ms)
        if not np.isfinite(target) or target <= 0:
            raise ValueError(f"exposure_ms 必须是正有限值: {exposure_ms!r}")
        entry = min(
            self.entries,
            key=lambda item: (abs(item.exposure_ms - target), item.exposure_ms),
        )
        return self._load(entry)

    def _load(self, entry: DarkFrameEntry) -> DarkFrame:
        path = self.directory / entry.file
        if not path.is_file():
            raise FileNotFoundError(
                f"暗帧索引指向的阵列不存在: {path} (manifest 与文件不同步)"
            )
        with np.load(path) as data:
            if _ARRAY_KEY not in data:
                raise ValueError(f"暗帧文件缺少 '{_ARRAY_KEY}' 阵列: {path}")
            mean = np.asarray(data[_ARRAY_KEY], dtype=np.float32)
            std = (
                np.asarray(data[_STD_KEY], dtype=np.float32)
                if _STD_KEY in data
                else None
            )
        if mean.shape != entry.shape:
            raise ValueError(
                f"暗帧阵列形状 {mean.shape} 与 manifest 条目 {entry.shape} 不一致: {path}"
            )
        return DarkFrame(
            exposure_ms=entry.exposure_ms,
            mean=mean,
            std=std,
            frames=entry.frames,
            full_scale=entry.full_scale,
            entry=entry,
        )


def load_dark_frames(
    serial: str, *, root: str | Path | None = None
) -> DarkFrameSet:
    """读某序列号的暗帧索引; 目录/清单不存在时返回**空集**而不是报错。

    空集是合法状态 (这台相机还没采过暗帧), 但格式错误必须报错: 静默把损坏的
    manifest 当空集, 会让后续曝光校准在没有任何暗帧的情况下继续跑。
    """
    directory = dark_frames_dir(serial, root=root)
    manifest = directory / MANIFEST_NAME
    if not manifest.is_file():
        return DarkFrameSet(serial=str(serial), directory=directory, camera={})
    data = json.loads(manifest.read_text(encoding="utf-8"))
    schema = data.get("schema")
    if schema != SCHEMA:
        raise ValueError(
            f"暗帧 manifest schema 不符: {manifest} 里是 {schema!r}, 期望 {SCHEMA!r}"
        )
    entries_raw = data.get("entries", [])
    if not isinstance(entries_raw, list):
        raise ValueError(f"暗帧 manifest entries 必须是列表: {manifest}")
    entries = tuple(DarkFrameEntry.from_dict(item) for item in entries_raw)
    return DarkFrameSet(
        serial=str(data.get("serial") or serial),
        directory=directory,
        camera=dict(data.get("camera") or {}),
        entries=entries,
    )


def load_dark_frame(
    serial: str,
    exposure_ms: float,
    *,
    root: str | Path | None = None,
    rel_tol: float = 0.02,
    allow_nearest: bool = False,
) -> DarkFrame | None:
    """按序列号 + 曝光取暗帧。

    ``allow_nearest=False`` (默认) 只接受 ``rel_tol`` 内的同曝光条目; 曝光不匹配
    时返回 ``None``, 调用方必须自己决定要不要退到
    :meth:`DarkFrameSet.nearest` —— 跨曝光套用暗帧会引入与曝光成正比的偏置。
    """
    frames = load_dark_frames(serial, root=root)
    found = frames.match(exposure_ms, rel_tol=rel_tol)
    if found is None and allow_nearest:
        found = frames.nearest(exposure_ms)
    return found


def _same_exposure(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))


def save_dark_capture(
    serial: str,
    *,
    exposure_ms: float,
    mean: np.ndarray,
    frames: int,
    std: np.ndarray | None = None,
    requested_exposure_ms: float | None = None,
    bit_depth: int | None = None,
    full_scale: float | None = None,
    camera: Mapping[str, Any] | None = None,
    root: str | Path | None = None,
    captured_at: str | None = None,
) -> DarkFrameEntry:
    """写入一条暗帧并更新 manifest (同曝光的旧条目被替换)。

    写盘是原子的: 阵列先写 ``*.tmp`` 再 :func:`os.replace`, manifest 同理。
    采集中断时最多留下一个 ``.tmp``, 不会出现「manifest 已指向半截阵列」的状态。
    """
    mean_arr = np.asarray(mean, dtype=np.float32)
    if mean_arr.ndim != 2 or mean_arr.size == 0:
        raise ValueError(f"暗帧 mean 必须是二维非空阵列, 实际 {mean_arr.shape}")
    if not np.all(np.isfinite(mean_arr)):
        raise ValueError("暗帧 mean 含非有限值 (NaN/Inf), 拒绝写入配置数据")
    if frames <= 0:
        raise ValueError(f"frames 必须 > 0: {frames!r}")
    exposure = float(exposure_ms)
    if not np.isfinite(exposure) or exposure <= 0:
        raise ValueError(f"exposure_ms 必须是正有限值: {exposure_ms!r}")
    std_arr: np.ndarray | None = None
    if std is not None:
        std_arr = np.asarray(std, dtype=np.float32)
        if std_arr.shape != mean_arr.shape:
            raise ValueError(
                f"暗帧 std 形状 {std_arr.shape} 与 mean {mean_arr.shape} 不一致"
            )

    directory = dark_frames_dir(serial, root=root)
    directory.mkdir(parents=True, exist_ok=True)
    file_name = _exposure_filename(exposure)
    final_path = directory / file_name
    tmp_path = final_path.with_name(final_path.name + ".tmp")
    payload: dict[str, np.ndarray] = {_ARRAY_KEY: mean_arr}
    if std_arr is not None:
        payload[_STD_KEY] = std_arr
    with tmp_path.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    os.replace(tmp_path, final_path)

    entry = DarkFrameEntry(
        exposure_ms=exposure,
        file=file_name,
        frames=int(frames),
        shape=(int(mean_arr.shape[0]), int(mean_arr.shape[1])),
        mean=float(np.mean(mean_arr, dtype=np.float64)),
        std=None if std_arr is None else float(np.mean(std_arr, dtype=np.float64)),
        peak=float(np.max(mean_arr)),
        minimum=float(np.min(mean_arr)),
        bit_depth=bit_depth,
        full_scale=None if full_scale is None else float(full_scale),
        captured_at=captured_at or _utc_now(),
        requested_exposure_ms=(
            None if requested_exposure_ms is None else float(requested_exposure_ms)
        ),
    )

    manifest = directory / MANIFEST_NAME
    existing: dict[str, Any] = {}
    if manifest.is_file():
        raw = json.loads(manifest.read_text(encoding="utf-8"))
        if raw.get("schema") != SCHEMA:
            raise ValueError(
                f"现有暗帧 manifest schema 不符 ({raw.get('schema')!r} != {SCHEMA!r}): "
                f"{manifest}; 拒绝在未知格式上追加"
            )
        existing = raw
    old_entries = [
        DarkFrameEntry.from_dict(item)
        for item in existing.get("entries", [])
    ]
    kept = [
        old
        for old in old_entries
        if not _same_exposure(old.exposure_ms, entry.exposure_ms)
    ]
    for old in old_entries:
        if _same_exposure(old.exposure_ms, entry.exposure_ms) and old.file != entry.file:
            stale = directory / old.file
            if stale.is_file():
                stale.unlink()
    kept.append(entry)
    kept.sort(key=lambda item: item.exposure_ms)

    camera_meta = dict(existing.get("camera") or {})
    if camera:
        camera_meta.update({k: v for k, v in camera.items() if v is not None})
    camera_meta["serial"] = str(serial)
    manifest_payload = {
        "schema": SCHEMA,
        "serial": str(serial),
        "updated_at": _utc_now(),
        "camera": camera_meta,
        "entries": [item.to_dict() for item in kept],
    }
    tmp_manifest = manifest.with_name(manifest.name + ".tmp")
    tmp_manifest.write_text(
        json.dumps(manifest_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp_manifest, manifest)
    return entry


__all__ = [
    "SCHEMA",
    "DARK_DIR_NAME",
    "DarkFrame",
    "DarkFrameEntry",
    "DarkFrameSet",
    "ccd_config_dir",
    "dark_frames_dir",
    "load_dark_frame",
    "load_dark_frames",
    "save_dark_capture",
    "serial_dir_name",
]
