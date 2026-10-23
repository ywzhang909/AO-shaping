"""会话级实验系统配置的校验与按需重建.

为什么有它 (2026-10-23 实测事故)
--------------------------------

2f 实验系统的「默认配置」—— 模型前向的相机采样窗 ``camera_xy``
(121×121 @2 px, x 1241–1481 / y 902–1142) —— 是按**拟合数据集采集当天**的
U 支零点 (1359.8, 1008.5) 建的。换会话后, 同一约定下实测零点到了
(1418, 966): 右移 57 px、上移 43 px。沿用旧窗**不报错, 只会让结果悄悄
变差**: 零点落在窗宽 88%、窗高 34% 处, 一侧背景余量只剩 34 px (另一侧
146 px), 背景与能量效率 (EE) 分母不再对称, 模型面板看起来也偏 —— 但
评分区 (60 px 掩框) 仍然完整, 所以「验证通过」的迹象不会暴露它。按实测
零点重建窗后, 正向重拟合 loss 0.928 → 0.356 (−62%), 模型 CV
1.497 → 1.238。

原则
----

> **每次光路的默认配置都不可信。** 凡是要靠迭代/采集得到的配置项, 先做
> **快速检验**: 在容差内就复用, 超差或缺失就重建, 并把「复用 / 重建 /
> 拒绝」的判定**逐项**记录下来。

模块边界
--------

* **注入式探针**: 调用方传入「显示一个相位并取一帧」的回调 (返回
  :class:`ProbeResult`)。本模块只做「argmax 定位零点 → 比对 → 判定」,
  模块级**不** import 任何驱动/SDK/torch。零点定位用 ``argmax`` —— 0 级
  是画面的**全局最大值**, 不是几何中心 (光轴不一定穿过画面中心)。
* **不复制窗口构造**: 重建直接调用
  :func:`ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity.recentre_window`
  (仓库里重建 ``camera_xy`` 采样窗的唯一实现)。
* **关联但不触碰**: 设备互斥 (:mod:`ao_shaping.tools.kernels.device_mutex`)
  与焦面前置阶段 (:mod:`ao_shaping.tools.slm.panel.focus_preflight`) 保持
  原样; 本模块不开设备, 也不读它们的证据文件。

协议脚本在哪一步调用它 (一句话)
------------------------------

在**开设备后** (设备互斥等待/加锁成功、面板与相机都打开)、**焦面前置
固定后**、**正式采集前**: 显示平场或一个已知相位, 抓一帧喂给
:func:`validate_or_rebuild` 逐项判定 —— 出现 ``reject`` 就停下修复
(缺失/不可测, 绝不静默沿用), 出现 ``rebuild`` 就用返回的
``new_profile`` 作为本会话配置 (窗心按实测零点重建)。

术语
----

中文叙述遵守 ``docs/terminology_glossary.md``: 光学口径一律写「瞳面」,
实验设备一律写「实验系统」, A/B 对照语境一律写「变体」。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
    recentre_window,
)

__all__ = [
    "VERDICT_REUSE",
    "VERDICT_REBUILD",
    "VERDICT_REJECT",
    "PROFILE_FILENAME",
    "ProbeResult",
    "ProbeFn",
    "SessionProfile",
    "SessionProfileError",
    "Tolerance",
    "ItemVerdict",
    "ValidationReport",
    "spot_centroid_of_frame",
    "probe_spot_centroid",
    "probe_exposure_ms",
    "probe_focus_diopter",
    "validate_or_rebuild",
    "save_profile",
    "load_profile",
]

#: 逐项判定: 容差内, 沿用。
VERDICT_REUSE = "reuse"
#: 逐项判定: 超差, 以本次实测值为准重建该配置项。
VERDICT_REBUILD = "rebuild"
#: 逐项判定: 缺失或无法测量 —— **不得**静默当 reuse, 流程应停下。
VERDICT_REJECT = "reject"

#: 会话配置落盘证据的默认文件名 (与 ``focus_preflight.json`` 同级存放)。
PROFILE_FILENAME = "session_profile.json"

#: 逐检验的配置项 (报告里的固定顺序)。
_ITEM_NAMES: tuple[str, ...] = (
    "spot_centroid",
    "window_centre",
    "window_margin",
    "sample_step",
    "exposure",
    "focus",
    "panel_shape",
    "slm_id",
    "cam_id",
)


class SessionProfileError(RuntimeError):
    """会话配置无效: 字段缺失 / 不是数值 / 非正, 或证据文件不可读。

    本模块全程**显式失败**: 缺失的字段判 ``reject``, 坏文件抛异常,
    绝不静默回落成默认值 —— 默认配置不可信, 正是本模块要防的。
    """


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _xy(value: Any, label: str) -> tuple[float, float] | None:
    """归一成二元浮点坐标 (x, y); 缺失为 ``None``, 其余报错。"""
    if value is None:
        return None
    try:
        x, y = float(value[0]), float(value[1])
    except (TypeError, ValueError) as exc:
        raise SessionProfileError(f"{label} 不是二元数值: {value!r}") from exc
    if not (math.isfinite(x) and math.isfinite(y)):
        raise SessionProfileError(f"{label} 含非有限值: {value!r}")
    return (x, y)


def _positive(value: Any, label: str) -> float | None:
    """归一成正的有限浮点; 缺失为 ``None``, 其余报错。"""
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise SessionProfileError(f"{label} 不是数值: {value!r}") from exc
    if not math.isfinite(out) or out <= 0.0:
        raise SessionProfileError(f"{label} 必须是正的有限值: {value!r}")
    return out


def _finite_opt(value: Any, label: str) -> float | None:
    """归一成有限浮点 (可负, 可零); 缺失为 ``None``, 其余报错。"""
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise SessionProfileError(f"{label} 不是数值: {value!r}") from exc
    if not math.isfinite(out):
        raise SessionProfileError(f"{label} 不是有限值: {value!r}")
    return out


def _int_opt(value: Any, label: str) -> int | None:
    if value is None:
        return None
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise SessionProfileError(f"{label} 不是整数: {value!r}") from exc
    return out


def _panel_shape_opt(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    try:
        h, w = int(value[0]), int(value[1])
    except (TypeError, ValueError) as exc:
        raise SessionProfileError(f"panel_shape 不是二元整数: {value!r}") from exc
    if h <= 0 or w <= 0:
        raise SessionProfileError(f"panel_shape 必须为正: {value!r}")
    return (h, w)


def _xy_json(value: tuple[float, float] | None) -> list[float] | None:
    """坐标 → JSON 友好的二元列表 (``None`` 原样)。"""
    return None if value is None else [float(value[0]), float(value[1])]


def _jsonable(value: Any) -> Any:
    """递归转成 JSON 可序列化 (元组→列表, ndarray→列表)。"""
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _restore_pair(value: Any) -> Any:
    """JSON 往返后把二元列表还原成坐标元组; 其余原样。"""
    if isinstance(value, list) and len(value) == 2:
        try:
            return (float(value[0]), float(value[1]))
        except (TypeError, ValueError):
            return value
    return value


# ---------------------------------------------------------------------------
# 会话配置
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SessionProfile:
    """2f 实验系统的会话级配置 (所有字段可 JSON 序列化)。

    每个字段都可为 ``None`` —— 「缺失」是合法状态, 但校验时必须判
    ``reject``, 不得静默当 reuse。

    Attributes:
        spot_centroid_xy: 实测零点 (零级衍射光斑), 相机像素 ``(x, y)``。
        window_centre_xy: 模型前向相机采样窗的窗心, 相机像素。
        window_half_span_px: 窗的单边跨度 (相机 px)。
        sample_step_px: 窗的采样步长 (相机 px), 与拟合口径一致。
        exposure_ms: 本会话曝光 (ms)。
        focus_diopter: 焦面补偿的屈光度 (m⁻¹); 只有 ``best_focal_mm`` 时可为 ``None``。
        best_focal_mm: 焦面补偿的焦距 (mm); 只有 ``focus_diopter`` 时可为 ``None``。
        panel_shape: 面板形状 ``(height, width)``。
        slm_id: 面板编号。
        cam_id: 相机编号。
        recorded_utc: 记录时刻 (UTC, ISO 8601)。
        source: 哪个脚本 / 哪次运行写的 (溯源)。
    """

    spot_centroid_xy: tuple[float, float] | None = None
    window_centre_xy: tuple[float, float] | None = None
    window_half_span_px: float | None = None
    sample_step_px: float | None = None
    exposure_ms: float | None = None
    focus_diopter: float | None = None
    best_focal_mm: float | None = None
    panel_shape: tuple[int, int] | None = None
    slm_id: int | None = None
    cam_id: int | None = None
    recorded_utc: str = ""
    source: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "spot_centroid_xy", _xy(self.spot_centroid_xy, "spot_centroid_xy"))
        object.__setattr__(self, "window_centre_xy", _xy(self.window_centre_xy, "window_centre_xy"))
        object.__setattr__(
            self, "window_half_span_px", _positive(self.window_half_span_px, "window_half_span_px")
        )
        object.__setattr__(self, "sample_step_px", _positive(self.sample_step_px, "sample_step_px"))
        object.__setattr__(self, "exposure_ms", _positive(self.exposure_ms, "exposure_ms"))
        object.__setattr__(self, "focus_diopter", _finite_opt(self.focus_diopter, "focus_diopter"))
        object.__setattr__(self, "best_focal_mm", _positive(self.best_focal_mm, "best_focal_mm"))
        object.__setattr__(self, "panel_shape", _panel_shape_opt(self.panel_shape))
        object.__setattr__(self, "slm_id", _int_opt(self.slm_id, "slm_id"))
        object.__setattr__(self, "cam_id", _int_opt(self.cam_id, "cam_id"))

    # -- 序列化 -----------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """JSON 可序列化的配置字典。"""
        return {
            "spot_centroid_xy": _xy_json(self.spot_centroid_xy),
            "window_centre_xy": _xy_json(self.window_centre_xy),
            "window_half_span_px": self.window_half_span_px,
            "sample_step_px": self.sample_step_px,
            "exposure_ms": self.exposure_ms,
            "focus_diopter": self.focus_diopter,
            "best_focal_mm": self.best_focal_mm,
            "panel_shape": None if self.panel_shape is None else list(self.panel_shape),
            "slm_id": self.slm_id,
            "cam_id": self.cam_id,
            "recorded_utc": str(self.recorded_utc),
            "source": str(self.source),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SessionProfile:
        """从 JSON 读出的字典构造配置 (缺字段按 ``None`` 处理)。"""

        def pair(key: str) -> tuple[float, float] | None:
            value = data.get(key)
            if value is None:
                return None
            return (float(value[0]), float(value[1]))

        panel = data.get("panel_shape")
        return cls(
            spot_centroid_xy=pair("spot_centroid_xy"),
            window_centre_xy=pair("window_centre_xy"),
            window_half_span_px=data.get("window_half_span_px"),
            sample_step_px=data.get("sample_step_px"),
            exposure_ms=data.get("exposure_ms"),
            focus_diopter=data.get("focus_diopter"),
            best_focal_mm=data.get("best_focal_mm"),
            panel_shape=None if panel is None else (int(panel[0]), int(panel[1])),
            slm_id=data.get("slm_id"),
            cam_id=data.get("cam_id"),
            recorded_utc=str(data.get("recorded_utc") or ""),
            source=str(data.get("source") or ""),
        )

    # -- 便捷构造与派生 ----------------------------------------------------
    @classmethod
    def rebuild_for_spot(
        cls,
        spot_xy: Any,
        *,
        half_span_px: float | None = 120.0,
        step_px: float | None = 2.0,
        exposure_ms: float | None = None,
        focus_diopter: float | None = None,
        best_focal_mm: float | None = None,
        panel_shape: tuple[int, int] | None = (1200, 1920),
        slm_id: int | None = None,
        cam_id: int | None = None,
        source: str = "SessionProfile.rebuild_for_spot",
    ) -> SessionProfile:
        """以**实测零点**为中心构造一份配置 (窗心 = 零点, 半边跨度可调)。

        没有任何保存配置可用 (全部 ``reject``) 时的显式重建入口。

        Args:
            spot_xy: 实测零点 ``(x, y)`` (相机 px)。
            half_span_px: 窗的单边跨度 (相机 px)。
            step_px: 采样步长 (相机 px)。
            exposure_ms: 曝光 (ms)。
            focus_diopter: 屈光度 (m⁻¹)。
            best_focal_mm: 焦距 (mm)。
            panel_shape: 面板形状 ``(height, width)``。
            slm_id: 面板编号。
            cam_id: 相机编号。
            source: 溯源 (哪个脚本 / 哪次运行写的)。

        Returns:
            新建的会话配置 (``window_centre_xy`` 与零点重合)。
        """
        spot = _xy(spot_xy, "spot_xy")
        if spot is None:
            raise SessionProfileError("rebuild_for_spot 需要实测零点 (spot_xy 不能为 None)")
        return cls(
            spot_centroid_xy=spot,
            window_centre_xy=spot,
            window_half_span_px=_positive(half_span_px, "half_span_px"),
            sample_step_px=_positive(step_px, "step_px"),
            exposure_ms=_positive(exposure_ms, "exposure_ms"),
            focus_diopter=_finite_opt(focus_diopter, "focus_diopter"),
            best_focal_mm=_positive(best_focal_mm, "best_focal_mm"),
            panel_shape=_panel_shape_opt(panel_shape),
            slm_id=_int_opt(slm_id, "slm_id"),
            cam_id=_int_opt(cam_id, "cam_id"),
            recorded_utc=_now_utc(),
            source=source,
        )

    def recenter_for_spot(
        self,
        spot_xy: Any,
        *,
        half_span_px: float | None = None,
        source: str = "",
    ) -> SessionProfile:
        """沿用本配置的其余字段, 只把窗重建到**实测零点**上 (窗心 = 零点)。

        Args:
            spot_xy: 本次实测零点 ``(x, y)`` (相机 px)。
            half_span_px: 新的窗单边跨度; ``None`` = 沿用本配置。
            source: 新配置的溯源; 缺省自动带上旧配置的来源。

        Returns:
            重建后的新配置 (本对象不变 —— frozen)。
        """
        spot = _xy(spot_xy, "spot_xy")
        if spot is None:
            raise SessionProfileError("recenter_for_spot 需要实测零点 (spot_xy 不能为 None)")
        return SessionProfile(
            spot_centroid_xy=spot,
            window_centre_xy=spot,
            window_half_span_px=half_span_px if half_span_px is not None else self.window_half_span_px,
            sample_step_px=self.sample_step_px,
            exposure_ms=self.exposure_ms,
            focus_diopter=self.focus_diopter,
            best_focal_mm=self.best_focal_mm,
            panel_shape=self.panel_shape,
            slm_id=self.slm_id,
            cam_id=self.cam_id,
            recorded_utc=_now_utc(),
            source=source or f"SessionProfile.recenter_for_spot <- {self.source}",
        )

    def focus_diopter_effective(self) -> float | None:
        """本配置对应的屈光度 (m⁻¹): 优先 ``focus_diopter``, 只有焦距时按 1000/f 换算。"""
        if self.focus_diopter is not None:
            return self.focus_diopter
        if self.best_focal_mm is not None and self.best_focal_mm > 0.0:
            return 1000.0 / self.best_focal_mm
        return None

    def camera_xy(self) -> np.ndarray:
        """本配置对应的采样窗坐标 (H, W, 2), 直接调用唯一的窗重建实现。

        Returns:
            ``recentre_window`` 的输出: ``[..., 0]`` 是 x、``[..., 1]`` 是 y,
            窗心在 ``window_centre_xy``。

        Raises:
            SessionProfileError: 窗心 / 半边跨度 / 步长任一缺失。
        """
        if self.window_centre_xy is None:
            raise SessionProfileError("window_centre_xy 缺失, 无法构造 camera_xy")
        if self.window_half_span_px is None:
            raise SessionProfileError("window_half_span_px 缺失, 无法构造 camera_xy")
        if self.sample_step_px is None:
            raise SessionProfileError("sample_step_px 缺失, 无法构造 camera_xy")
        return recentre_window(
            self.window_centre_xy,
            half_span_px=self.window_half_span_px,
            step_px=self.sample_step_px,
        )


# ---------------------------------------------------------------------------
# 探针 (注入式)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ProbeResult:
    """一次探针的返回: 一张相机帧 + 探针读到的会话状态。

    Attributes:
        frame: 2D 相机帧 (任意数值 dtype, 可含负的读出噪声)。
        exposure_ms: 本帧曝光 (ms); ``None`` = 未读到。
        focus_diopter: 当前焦面补偿的屈光度 (m⁻¹); ``None`` = 未读到。
        best_focal_mm: 当前焦面补偿的焦距 (mm); ``None`` = 未读到。
    """

    frame: np.ndarray
    exposure_ms: float | None = None
    focus_diopter: float | None = None
    best_focal_mm: float | None = None


#: 注入式探针: 「显示一个相位并取一帧」的回调 (本模块不 import 任何驱动)。
ProbeFn = Callable[[], ProbeResult]


def spot_centroid_of_frame(frame: np.ndarray) -> tuple[float, float] | None:
    """用 ``argmax`` 在帧上定位零级 (0 级衍射光斑)。

    2f 实验系统的 0 级是帧的**全局最大**点, 绝不是几何中心 (光轴不一定
    穿过帧中心, 实测零级与帧中心可差数十 px)。下列情形判**无法测量**,
    返回 ``None`` (而不是给一个看起来像坐标的假值): 帧不是 2D / 为空 /
    没有任何有限值 / 峰值非正 (无信号)。

    Args:
        frame: 2D 相机帧。

    Returns:
        零点 ``(x=列, y=行)`` (相机 px), 或 ``None``。
    """
    arr = np.asarray(frame, dtype=np.float64)
    if arr.ndim != 2 or arr.size == 0:
        return None
    if not np.isfinite(arr).any():
        return None
    idx = np.unravel_index(int(np.nanargmax(arr)), arr.shape)
    peak = float(arr[idx])
    if not math.isfinite(peak) or peak <= 0.0:
        return None
    return (float(idx[1]), float(idx[0]))


def probe_spot_centroid(probe: ProbeFn) -> tuple[float, float] | None:
    """跑一次探针 (显示一个相位并取一帧), 返回零点坐标; ``None`` = 无法测量。"""
    return spot_centroid_of_frame(probe().frame)


def probe_exposure_ms(probe: ProbeFn) -> float | None:
    """跑一次探针, 返回本帧曝光 (ms); 未读到 / 非法时 ``None``。"""
    value = probe().exposure_ms
    try:
        out = float(value)  # type: ignore[assignment]
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) and out > 0.0 else None


def probe_focus_diopter(probe: ProbeFn) -> float | None:
    """跑一次探针, 返回当前焦面补偿的屈光度 (m⁻¹); 未读到 / 非法时 ``None``。

    优先 ``focus_diopter``; 只有 ``best_focal_mm`` 时按 ``1000/f`` 换算。
    """
    result = probe()
    if result.focus_diopter is not None:
        try:
            out = float(result.focus_diopter)
        except (TypeError, ValueError):
            return None
        if math.isfinite(out):
            return out
    if result.best_focal_mm is not None:
        try:
            focal = float(result.best_focal_mm)
        except (TypeError, ValueError):
            return None
        if math.isfinite(focal) and focal > 0.0:
            return 1000.0 / focal
    return None


# ---------------------------------------------------------------------------
# 容差
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Tolerance:
    """逐配置项的容差与判据。

    Attributes:
        centroid_px: 零点与存储零点的最大允许偏移 (相机 px, 欧氏距离)。
        exposure_relative: 曝光的最大允许**相对**偏移 (|Δ| / 存储值)。
        focus_diopter_abs: 屈光度的最大允许**绝对**偏移 (m⁻¹)。
        window_margin_px: 零点到窗边的最小允许余量 (相机 px), 不得小于它。
    """

    centroid_px: float = 5.0
    exposure_relative: float = 0.25
    focus_diopter_abs: float = 0.05
    window_margin_px: float = 40.0

    def __post_init__(self) -> None:
        for name in ("centroid_px", "exposure_relative", "focus_diopter_abs", "window_margin_px"):
            value = getattr(self, name)
            try:
                out = float(value)  # type: ignore[assignment]
            except (TypeError, ValueError) as exc:
                raise SessionProfileError(f"Tolerance.{name} 不是数值: {value!r}") from exc
            if not math.isfinite(out) or out < 0.0:
                raise SessionProfileError(f"Tolerance.{name} 必须是非负有限值: {value!r}")


# ---------------------------------------------------------------------------
# 逐项判定
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ItemVerdict:
    """单个配置项的判定 (必须带非空理由)。

    Attributes:
        item: 配置项名 (如 ``spot_centroid`` / ``exposure``)。
        verdict: ``reuse`` / ``rebuild`` / ``reject``。
        stored: 存储值 (JSON 可序列化, 或 ``None``)。
        measured: 本次实测值 (JSON 可序列化, 或 ``None``)。
        reason: 判定的理由 (人话, 非空)。
    """

    item: str
    verdict: str
    stored: Any
    measured: Any
    reason: str

    def __post_init__(self) -> None:
        if self.verdict not in (VERDICT_REUSE, VERDICT_REBUILD, VERDICT_REJECT):
            raise SessionProfileError(f"未知判定 {self.verdict!r} (item={self.item!r})")
        if not str(self.reason).strip():
            raise SessionProfileError(f"item {self.item!r} 的理由为空: 每个判定都必须说明自己")

    def to_dict(self) -> dict[str, Any]:
        return {
            "item": self.item,
            "verdict": self.verdict,
            "stored": _jsonable(self.stored),
            "measured": _jsonable(self.measured),
            "reason": str(self.reason),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ItemVerdict:
        return cls(
            item=str(data.get("item") or ""),
            verdict=str(data.get("verdict") or ""),
            stored=_restore_pair(data.get("stored")),
            measured=_restore_pair(data.get("measured")),
            reason=str(data.get("reason") or ""),
        )


@dataclass(frozen=True)
class ValidationReport:
    """一次会话配置校验的逐项判定。

    Attributes:
        items: 每个配置项的判定 (全部带理由)。
        checked_utc: 校验时刻 (UTC, ISO 8601)。
        new_profile: 需要重建时的新配置 (窗心以实测零点重建); 全部 reuse
            或存在 reject 时为 ``None``。
    """

    items: tuple[ItemVerdict, ...]
    checked_utc: str
    new_profile: SessionProfile | None = None

    @property
    def rejected(self) -> bool:
        """任一配置项为 ``reject`` (缺失 / 无法测量)。"""
        return any(v.verdict == VERDICT_REJECT for v in self.items)

    @property
    def any_rebuild(self) -> bool:
        """任一配置项为 ``rebuild`` (超差)。"""
        return any(v.verdict == VERDICT_REBUILD for v in self.items)

    @property
    def status(self) -> str:
        """总体状态: ``reject`` (有缺失/无法测量, 必须停下) / ``rebuild``
        (有超差, 用 ``new_profile``) / ``reuse`` (全部在容差内)。"""
        if self.rejected:
            return VERDICT_REJECT
        if self.any_rebuild:
            return VERDICT_REBUILD
        return VERDICT_REUSE

    def verdict_of(self, item: str) -> ItemVerdict:
        """取某个配置项的判定; 报告里没有它时报错 (不静默)。"""
        for verdict in self.items:
            if verdict.item == item:
                return verdict
        raise SessionProfileError(f"报告里没有 {item!r} 的判定")

    def to_dict(self) -> dict[str, Any]:
        """JSON 可序列化的报告字典 (可整体 ``json.dumps``)。"""
        return {
            "status": self.status,
            "checked_utc": self.checked_utc,
            "items": [verdict.to_dict() for verdict in self.items],
            "new_profile": None if self.new_profile is None else self.new_profile.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ValidationReport:
        """从 ``to_dict`` 的 JSON 还原报告 (真往返)。"""
        items = tuple(ItemVerdict.from_dict(entry) for entry in data.get("items", []))
        profile = data.get("new_profile")
        return cls(
            items=items,
            checked_utc=str(data.get("checked_utc") or ""),
            new_profile=None if profile is None else SessionProfile.from_dict(profile),
        )


# ---------------------------------------------------------------------------
# 逐项比较
# ---------------------------------------------------------------------------
def _reject(item: str, stored: Any, measured: Any, reason: str) -> ItemVerdict:
    return ItemVerdict(item=item, verdict=VERDICT_REJECT, stored=stored, measured=measured, reason=reason)


def _check_spot_centroid(
    stored: SessionProfile, measured: tuple[float, float] | None, tol: Tolerance
) -> ItemVerdict:
    stored_xy = stored.spot_centroid_xy
    if stored_xy is None:
        return _reject(
            "spot_centroid",
            None,
            _xy_json(measured),
            "存储配置缺 spot_centroid_xy (实测零点): 无法比较, 不得静默当 reuse",
        )
    if measured is None:
        return _reject(
            "spot_centroid",
            _xy_json(stored_xy),
            None,
            "探针帧无法定位零点 (无法测量): 帧非法或无正峰值, 不得静默当 reuse",
        )
    offset = math.hypot(measured[0] - stored_xy[0], measured[1] - stored_xy[1])
    if offset <= tol.centroid_px:
        return ItemVerdict(
            "spot_centroid",
            VERDICT_REUSE,
            _xy_json(stored_xy),
            _xy_json(measured),
            f"零点偏移 {offset:.2f} px 在容差内 ({tol.centroid_px:g} px), 存储零点仍可用",
        )
    return ItemVerdict(
        "spot_centroid",
        VERDICT_REBUILD,
        _xy_json(stored_xy),
        _xy_json(measured),
        f"零点偏移 {offset:.2f} px 超差 ({tol.centroid_px:g} px): 存储零点来自拟合当天会话, "
        f"必须以本次实测零点重建",
    )


def _check_window_centre(
    stored: SessionProfile, measured: tuple[float, float] | None, tol: Tolerance
) -> ItemVerdict:
    stored_xy = stored.window_centre_xy
    if stored_xy is None:
        return _reject(
            "window_centre",
            None,
            _xy_json(measured),
            "存储配置缺窗心 window_centre_xy: 无法比较, 不得静默当 reuse",
        )
    if measured is None:
        return _reject(
            "window_centre",
            _xy_json(stored_xy),
            None,
            "零点无法测量, 窗心无从核对, 不得静默当 reuse",
        )
    offset = math.hypot(measured[0] - stored_xy[0], measured[1] - stored_xy[1])
    if offset <= tol.centroid_px:
        return ItemVerdict(
            "window_centre",
            VERDICT_REUSE,
            _xy_json(stored_xy),
            _xy_json(measured),
            f"窗心与实测零点偏移 {offset:.2f} px 在容差内 ({tol.centroid_px:g} px), 窗仍可用",
        )
    return ItemVerdict(
        "window_centre",
        VERDICT_REBUILD,
        _xy_json(stored_xy),
        _xy_json(measured),
        f"窗心与实测零点偏移 {offset:.2f} px 超差 ({tol.centroid_px:g} px): 窗必须以实测零点重建",
    )


def _check_window_margin(
    stored: SessionProfile, measured: tuple[float, float] | None, tol: Tolerance
) -> ItemVerdict:
    centre = stored.window_centre_xy
    half = stored.window_half_span_px
    if centre is None or half is None:
        return _reject(
            "window_margin",
            _xy_json(centre),
            None,
            "存储配置缺窗心或窗半边跨度 (window_half_span_px): 余量无法计算, 不得静默当 reuse",
        )
    if measured is None:
        return _reject(
            "window_margin",
            None,
            None,
            "零点无法测量, 余量无从计算, 不得静默当 reuse",
        )
    margin_x = half - abs(measured[0] - centre[0])
    margin_y = half - abs(measured[1] - centre[1])
    margin = min(margin_x, margin_y)
    if margin >= tol.window_margin_px:
        return ItemVerdict(
            "window_margin",
            VERDICT_REUSE,
            {"centre": list(centre), "half_span_px": half},
            _xy_json(measured),
            f"零点到窗边最小余量 {margin:.1f} px (x {margin_x:.1f} / y {margin_y:.1f}), "
            f"不低于要求的 {tol.window_margin_px:g} px",
        )
    if margin < 0.0:
        side = "x" if margin_x < margin_y else "y"
        return ItemVerdict(
            "window_margin",
            VERDICT_REBUILD,
            {"centre": list(centre), "half_span_px": half},
            _xy_json(measured),
            f"零点已落在窗外 ({side} 侧余量 {margin:.1f} px < 0): 必须以实测零点重建窗",
        )
    return ItemVerdict(
        "window_margin",
        VERDICT_REBUILD,
        {"centre": list(centre), "half_span_px": half},
        _xy_json(measured),
        f"零点到窗边最小余量 {margin:.1f} px 低于要求的 {tol.window_margin_px:g} px: "
        f"窗需重建并放大",
    )


def _check_sample_step(stored: SessionProfile) -> ItemVerdict:
    if stored.sample_step_px is None:
        return _reject(
            "sample_step",
            None,
            None,
            "存储配置缺采样步长 sample_step_px: 无法确认与拟合口径一致, 不得静默当 reuse",
        )
    return ItemVerdict(
        "sample_step",
        VERDICT_REUSE,
        stored.sample_step_px,
        None,
        f"采样步长 {stored.sample_step_px:g} px 是固定拟合口径 (探针无法测量), 在场检查通过",
    )


def _check_exposure(
    stored: SessionProfile, measured: float | None, tol: Tolerance
) -> ItemVerdict:
    stored_ms = stored.exposure_ms
    if stored_ms is None:
        return _reject(
            "exposure",
            None,
            measured,
            "存储配置缺曝光 exposure_ms: 无法比较, 不得静默当 reuse",
        )
    if measured is None:
        return _reject(
            "exposure",
            stored_ms,
            None,
            "探针未报告曝光 (无法测量): 不得静默当 reuse",
        )
    relative = abs(measured - stored_ms) / abs(stored_ms)
    if relative <= tol.exposure_relative:
        return ItemVerdict(
            "exposure",
            VERDICT_REUSE,
            stored_ms,
            measured,
            f"曝光 {measured:g} ms vs 存储 {stored_ms:g} ms, 相对偏移 {relative:.1%} "
            f"在容差内 ({tol.exposure_relative:.0%})",
        )
    return ItemVerdict(
        "exposure",
        VERDICT_REBUILD,
        stored_ms,
        measured,
        f"曝光 {measured:g} ms vs 存储 {stored_ms:g} ms, 相对偏移 {relative:.1%} 超差 "
        f"({tol.exposure_relative:.0%}): 采用探针当前曝光",
    )


def _check_focus(
    stored: SessionProfile, measured: float | None, tol: Tolerance
) -> ItemVerdict:
    stored_diopter = stored.focus_diopter_effective()
    if stored_diopter is None:
        return _reject(
            "focus",
            None,
            measured,
            "存储配置缺焦面 (focus_diopter / best_focal_mm 都没有): 无法比较, 不得静默当 reuse",
        )
    if measured is None:
        return _reject(
            "focus",
            stored_diopter,
            None,
            "探针未报告焦面 (无法测量): 焦面补偿可能已变, 不得静默当 reuse",
        )
    diff = abs(measured - stored_diopter)
    if diff <= tol.focus_diopter_abs:
        return ItemVerdict(
            "focus",
            VERDICT_REUSE,
            stored_diopter,
            measured,
            f"光焦度 {measured:.4f} m⁻¹ vs 存储 {stored_diopter:.4f} m⁻¹, 偏移 {diff:.4f} m⁻¹ "
            f"在容差内 ({tol.focus_diopter_abs:g} m⁻¹)",
        )
    return ItemVerdict(
        "focus",
        VERDICT_REBUILD,
        stored_diopter,
        measured,
        f"光焦度 {measured:.4f} m⁻¹ vs 存储 {stored_diopter:.4f} m⁻¹, 偏移 {diff:.4f} m⁻¹ 超差: "
        f"采用当前焦面补偿",
    )


def _check_panel_shape(stored: SessionProfile) -> ItemVerdict:
    if stored.panel_shape is None:
        return _reject("panel_shape", None, None, "存储配置缺 panel_shape: 不得静默当 reuse")
    return ItemVerdict(
        "panel_shape",
        VERDICT_REUSE,
        list(stored.panel_shape),
        None,
        f"面板形状 {stored.panel_shape[0]}×{stored.panel_shape[1]} 由探针开面板确认, 在场检查通过",
    )


def _check_identity(value: int | None, name: str, device: str) -> ItemVerdict:
    if value is None:
        return _reject(name, None, None, f"存储配置缺 {name}: 不得静默当 reuse")
    return ItemVerdict(
        name,
        VERDICT_REUSE,
        value,
        None,
        f"{name}={value} 由探针开{device}确认, 在场检查通过",
    )


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def _rebuild_profile(
    stored: SessionProfile,
    items: tuple[ItemVerdict, ...],
    measured_centroid: tuple[float, float] | None,
    measured_exposure: float | None,
    measured_focus: float | None,
    tol: Tolerance,
    rebuild_source: str,
    checked_utc: str,
) -> SessionProfile | None:
    """组装重建后的新配置; 没有 ``rebuild`` 项 (或零点无法测量) 时为 ``None``。"""
    if not any(verdict.verdict == VERDICT_REBUILD for verdict in items):
        return None
    if measured_centroid is None:
        return None
    by_name = {verdict.item: verdict for verdict in items}
    stored_half = stored.window_half_span_px if stored.window_half_span_px is not None else tol.window_margin_px
    new_half = max(stored_half, tol.window_margin_px)
    new_exposure = (
        measured_exposure
        if by_name["exposure"].verdict == VERDICT_REBUILD and measured_exposure is not None
        else stored.exposure_ms
    )
    stored_diopter = stored.focus_diopter_effective()
    new_diopter = (
        measured_focus
        if by_name["focus"].verdict == VERDICT_REBUILD and measured_focus is not None
        else stored_diopter
    )
    new_focal = (
        1000.0 / new_diopter
        if new_diopter is not None and new_diopter > 0.0
        else stored.best_focal_mm
    )
    return SessionProfile(
        spot_centroid_xy=measured_centroid,
        window_centre_xy=measured_centroid,
        window_half_span_px=new_half,
        sample_step_px=stored.sample_step_px,
        exposure_ms=new_exposure,
        focus_diopter=new_diopter,
        best_focal_mm=new_focal,
        panel_shape=stored.panel_shape,
        slm_id=stored.slm_id,
        cam_id=stored.cam_id,
        recorded_utc=checked_utc,
        source=rebuild_source,
    )


def validate_or_rebuild(
    stored: SessionProfile | None,
    probe: ProbeFn,
    tolerances: Tolerance | None = None,
    *,
    rebuild_source: str = "session_profile.validate_or_rebuild",
) -> ValidationReport:
    """把存储的会话配置逐项校验, 超差时以本次实测值重建。

    原则: **每次光路的默认配置都不可信** —— 先做快速检验, 容差内复用,
    超差或缺失就重建 / 拒绝, 并把「复用 / 重建 / 拒绝」的判定**逐项**记录
    (每项都带非空理由, 可 JSON 落盘)。

    Args:
        stored: 保存的会话配置。``None`` = 没有保存的配置, 全部配置项直接
            ``reject`` (不静默降级)。
        probe: 「显示一个相位并取一帧」的回调 (注入式; 本模块不 import 任何
            驱动)。返回 :class:`ProbeResult`。
        tolerances: 逐项容差; 缺省用 :class:`Tolerance` 的默认值。
        rebuild_source: 重建出的新配置的 ``source`` (哪个脚本 / 哪次运行写的)。

    Returns:
        逐项判定报告。总体 ``status``:

        * ``reuse`` — 全部在容差内, ``new_profile`` 为 ``None``;
        * ``rebuild`` — 有超差项, ``new_profile`` 是新配置 (窗心以实测零点
          重建, 曝光 / 焦面采用当前实测值);
        * ``reject`` — 有缺失 / 无法测量的项, **流程应停下**, 不得静默继续。
    """
    tol = tolerances if tolerances is not None else Tolerance()
    checked = _now_utc()

    if stored is None:
        items = tuple(
            _reject(
                name,
                None,
                None,
                f"没有保存的会话配置 (stored is None): {name} 缺失, 无法校验",
            )
            for name in _ITEM_NAMES
        )
        return ValidationReport(items=items, checked_utc=checked, new_profile=None)

    result = probe()
    measured_centroid = spot_centroid_of_frame(result.frame)
    measured_exposure = probe_exposure_ms(lambda: result)
    measured_focus = probe_focus_diopter(lambda: result)

    items: list[ItemVerdict] = [
        _check_spot_centroid(stored, measured_centroid, tol),
        _check_window_centre(stored, measured_centroid, tol),
        _check_window_margin(stored, measured_centroid, tol),
        _check_sample_step(stored),
        _check_exposure(stored, measured_exposure, tol),
        _check_focus(stored, measured_focus, tol),
        _check_panel_shape(stored),
        _check_identity(stored.slm_id, "slm_id", "面板"),
        _check_identity(stored.cam_id, "cam_id", "相机"),
    ]
    new_profile = _rebuild_profile(
        stored,
        tuple(items),
        measured_centroid,
        measured_exposure,
        measured_focus,
        tol,
        rebuild_source,
        checked,
    )
    return ValidationReport(items=tuple(items), checked_utc=checked, new_profile=new_profile)


# ---------------------------------------------------------------------------
# 落盘
# ---------------------------------------------------------------------------
def save_profile(profile: SessionProfile, path: str | Path) -> Path:
    """把会话配置落盘 (JSON, 带溯源)。

    Args:
        profile: 要保存的会话配置。
        path: 目标文件路径 (缺省目录会自动创建)。

    Returns:
        写出的文件路径。
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(profile.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return target


def load_profile(path: str | Path) -> SessionProfile:
    """从落盘文件读会话配置; 文件缺失 / 损坏时**显式报错** (无静默默认值)。

    Args:
        path: ``session_profile.json`` (或任何 :func:`save_profile` 写出的文件)。

    Returns:
        读出的会话配置 (文件里没写 ``source`` 时用文件路径回填)。

    Raises:
        SessionProfileError: 文件不存在 / 不可读 / 顶层不是对象。
    """
    target = Path(path)
    if not target.is_file():
        raise SessionProfileError(f"会话配置不存在: {target}")
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SessionProfileError(f"会话配置不可读: {target} ({exc})") from exc
    if not isinstance(payload, dict):
        raise SessionProfileError(f"会话配置格式不对: {target} 的顶层不是对象")
    profile = SessionProfile.from_dict(payload)
    if not profile.source:
        profile = replace(profile, source=str(target))
    return profile
