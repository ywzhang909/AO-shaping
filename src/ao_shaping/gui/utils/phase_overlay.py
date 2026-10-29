"""发送叠加相位（controller 层代理）：离焦 + 底相位的 controller 层承载方案。

本模块是**只依赖驱动只读接口**的 controller 层叠加方案，全部代码在
``gui/utils/``（按用户 2026-10-19 约束「修改尽量只动 gui/utils」）。它把叠加
**固定在 controller 层**：只要 session 句柄是 :class:`OverlaidSantec`，叠加就
始终生效，与优化器自开/换句柄无关。

接线现状（已接入 controller，见本 commit）
----------------------------------
上游 acb97b1f 把离焦/底相位做成**驱动层双槽**（``Santec._phase_overlay`` 弧度制 /
``Santec._base_phase`` 灰度制，在 ``Santec.create_phase_from_array`` 内部统一施加），
由 ``multi_slm_controller._apply_slm_overlays`` 推送。controller 现已把该路径切到
本模块的代理：

* ``connect_slm`` / ``shaping_controls._reacquire_slm_handle`` 在存 session 句柄前
  用 ``OverlaidSantec(slm)`` 包装真实 Santec（连接 + 闭环整形后重取两条路径一致）；
* ``_apply_slm_overlays`` 成为**单一分发点**：先算出 :class:`PhaseOverlay`（离焦受
  开关门控、底相位受其复选框门控），再经 :func:`apply_overlays_to_handle` 按句柄
  类型写入真源（代理句柄 → 写代理层并清零其驱动槽；裸驱动句柄 → 写驱动槽），
  维护「一个句柄只叠加一次」的 invariant，代理 / 裸驱动两条路径互不双重叠加。

驱动层双槽（acb97b1f）的字段与测试**原样保留**（``set_phase_overlay`` /
``set_base_phase`` / ``_phase_overlay`` / ``_base_phase`` 及 30 个驱动槽测试不动）；
代理方案作为叠加层承载路径运行于其上，两者按句柄类型二选一、不叠加。

防双重叠加（invariant）
-----------------------
对同一个句柄，叠加层只能来自**一个**真源：

* 句柄是 :class:`OverlaidSantec` → 叠加来自**代理层**；:func:`apply_overlays_to_handle`
  写入代理层时会把其**驱动槽清零**，故经代理的发送只叠加一次。
* 句柄是裸 :class:`Santec`（或带 ``set_phase_overlay``/``set_base_phase`` 的替身）
  → 叠加来自**驱动槽**（上游既有设计）。

与驱动槽的逐位等价
------------------
``OverlaidSantec.create_phase_from_array`` 的叠加数学与驱动槽逐位一致（离线单测
``tests/ao_shaping/gui/utils/test_phase_overlay_proxy.py`` 已验证 max diff 0）：

* 离焦（弧度）：在 ``_resize_to_panel`` 之后、弧度→灰度转换**之前**加到输入相位
  上（代理先做幂等的 ``_resize_to_panel``，同尺寸时 no-op；``nan_to_num`` 语义
  与驱动一致）；
* 底相位（灰度）：在驱动 矫正/LUT/平移/mod **之后**以灰度叠加（驱动槽版在
  mod **之前**叠加；对已 mod 灰度再 ``mod`` 恒等，逐位一致）。

透明性
------
除 ``create_phase_from_array`` / ``display_phase`` / 叠加管理方法外，一切
属性/方法/dunder 透传到真实驱动。``set_phase_overlay`` / ``get_phase_overlay`` /
``set_base_phase`` / ``get_base_phase`` 在代理上实现为**读写代理层**（兼容驱动
签名），使旧调用点在代理句柄上继续工作，而不再落到裸驱动槽里（那会双重叠加）。
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class PhaseOverlay:
    """发送叠加相位的合成器（controller 层，无硬件 I/O）。

    持有两层叠加，均为**面板坐标**固定尺寸（与 SLM 面板分辨率一致），与每次
    发送的输入图案尺寸解耦：

    * ``phase_rad``：弧度制相位叠加（离焦），在 弧度→灰度 转换**之前**加到
      输入相位上（resize 之后）。
    * ``base_gray``：灰度制底相位，在矫正/LUT 之后、``mod`` 之前加到灰度上。

    两层相互独立，任一层可为 ``None``（不叠加）。:meth:`compose` 是纯函数：
    每次现算，绝不累积；尺寸不匹配时告警并跳过该层（不卡核心发送路径），
    与旧驱动槽行为一致。
    """

    __slots__ = ("phase_rad", "base_gray")

    def __init__(
        self,
        phase_rad: NDArray[np.floating] | None = None,
        base_gray: NDArray[np.uint16] | None = None,
    ) -> None:
        self.phase_rad = (
            np.array(phase_rad, dtype=np.float64, copy=True)
            if phase_rad is not None
            else None
        )
        self.base_gray = (
            np.array(base_gray, dtype=np.uint16, copy=True)
            if base_gray is not None
            else None
        )

    @property
    def has_phase(self) -> bool:
        return self.phase_rad is not None

    @property
    def has_base(self) -> bool:
        return self.base_gray is not None

    @property
    def empty(self) -> bool:
        return not self.has_phase and not self.has_base

    def compose(
        self,
        panel_phase: NDArray[np.floating],
        max_grayscale: int,
    ) -> NDArray[np.uint16]:
        """把两层叠加套在「弧度→灰度」裸转换上的**参考实现**（纯函数）。

        参数
        ----
        panel_phase:
            **已 resize 到面板尺寸**的弧度制输入相位（弧度域，未转换）。
        max_grayscale:
            2π 弧度对应的灰度值（``Santec._max_gray``，默认 1023）。

        注意：本方法只做「离焦(弧度)→弧度→灰度→底相位(灰度)→mod」的**裸叠加**，
        不含 矫正CSV / LUT / 平移——那些是驱动的职责。实际发送路径用
        :meth:`OverlaidSantec.create_phase_from_array`（把本类的叠加逻辑插在
        驱动转换前后）；本方法供测试与离线验证。
        """
        out = panel_phase
        if self.phase_rad is not None:
            if self.phase_rad.shape == out.shape:
                out = out + self.phase_rad
            else:
                logger.warning(
                    "离焦叠加尺寸 %s 与面板 %s 不一致，已跳过（请检查离焦设置）",
                    self.phase_rad.shape,
                    out.shape,
                )
        gray = (out / (2.0 * np.pi)) * max_grayscale
        if self.base_gray is not None:
            if self.base_gray.shape == gray.shape:
                gray = gray + self.base_gray
            else:
                logger.warning(
                    "底相位尺寸 %s 与面板 %s 不一致，已跳过（请检查底相位设置）",
                    self.base_gray.shape,
                    gray.shape,
                )
        return np.mod(gray, max_grayscale).astype(np.uint16)


class OverlaidSantec:
    """透明代理：把 :class:`PhaseOverlay` 套在 ``Santec`` 发送路径外。

    控制器在连接/整形后重取时，用本类包装真实 :class:`Santec` 并把它存进
    ``st.session_state``。所有经代理 ``create_phase_from_array`` /
    ``display_phase`` 的发送路径都会经过本代理，叠加因此**全路径一致**；
    其余属性/方法/dunder 全部透传到真实驱动，代理对现有代码完全透明。

    叠加层管理
    ----------
    ``self._overlay``（一个 :class:`PhaseOverlay`）是代理上叠加层的唯一真源。
    控制器改叠加时调 :meth:`set_overlay`（或 :func:`apply_overlays_to_handle`）
    替换该引用（幂等、无硬件 I/O），代理下一次发送即用新层。
    :meth:`set_phase_overlay` / :meth:`set_base_phase` 是兼容驱动签名的别名，
    同样读写代理层。
    """

    __slots__ = ("_slm", "_overlay")

    def __init__(self, slm: Any, overlay: PhaseOverlay | None = None) -> None:
        object.__setattr__(self, "_slm", slm)
        object.__setattr__(
            self, "_overlay", overlay if overlay is not None else PhaseOverlay()
        )

    # ------------------------------------------------------------------
    # 叠加层管理（controller 调用 / 旧接口兼容）
    # ------------------------------------------------------------------
    def set_overlay(self, overlay: PhaseOverlay | None) -> None:
        """替换叠加层。幂等、无硬件 I/O。传 ``None`` 表示清空（等价全关）。"""
        object.__setattr__(
            self, "_overlay", overlay if overlay is not None else PhaseOverlay()
        )

    @property
    def overlay(self) -> PhaseOverlay:
        return self._overlay

    @property
    def raw(self) -> Any:
        """返回被包装的真实 Santec（供需要裸驱动的路径，如设备状态检查、
        或 ``_sync_slm_session_state`` 读 ``_serial_number``）。"""
        return self._slm

    # 旧接口兼容：读写**代理层**（不是驱动槽），保证经代理的发送只叠加一次。
    def set_phase_overlay(self, phase_rad: NDArray[np.floating] | None) -> None:
        """兼容驱动 ``set_phase_overlay`` 签名：写入**代理层**的离焦（弧度）。

        ``None`` 清空离焦层。注意与裸驱动不同——本方法**不**写驱动槽。
        """
        base = self._overlay.base_gray
        object.__setattr__(self, "_overlay", PhaseOverlay(phase_rad, base))

    def get_phase_overlay(self) -> NDArray[np.floating] | None:
        """返回代理层离焦（弧度）的副本，无叠加时 None（兼容驱动签名）。"""
        ov = self._overlay
        if ov.phase_rad is None:
            return None
        return ov.phase_rad.copy()

    def set_base_phase(self, phase_gray: NDArray[np.uint16] | None) -> None:
        """兼容驱动 ``set_base_phase`` 签名：写入**代理层**的底相位（灰度）。

        ``None`` 清空底相位层。注意与裸驱动不同——本方法**不**写驱动槽。
        """
        phase = self._overlay.phase_rad
        object.__setattr__(self, "_overlay", PhaseOverlay(phase, phase_gray))

    def get_base_phase(self) -> NDArray[np.uint16] | None:
        """返回代理层底相位（灰度）的副本，无叠加时 None（兼容驱动签名）。"""
        ov = self._overlay
        if ov.base_gray is None:
            return None
        return ov.base_gray.copy()

    # ------------------------------------------------------------------
    # 拦截：create_phase_from_array / display_phase（叠加注入点）
    # ------------------------------------------------------------------
    def create_phase_from_array(
        self, phase_rad: Any, max_grayscale: int | None = None
    ) -> NDArray[np.uint16]:
        """离焦(弧度)→驱动纯转换→底相位(灰度)。与旧驱动槽逐位等价。

        离焦层是面板坐标固定尺寸，故先把输入 ``_resize_to_panel``（同尺寸时
        幂等 no-op），在 弧度→灰度 之前加离焦，再做与驱动一致的
        ``nan_to_num(nan=0)``（保证 NaN 语义与驱动槽版一致），然后把数组交给
        驱动纯转换（驱动内二次 resize 对面板尺寸数组是 no-op）；底相位层也是
        面板坐标，在驱动 矫正/LUT/平移/mod 之后以灰度叠加（对已 mod 灰度再
        mod 恒等，与驱动槽在 mod 前叠加逐位一致）。驱动槽在代理路径上保持空，
        故驱动纯转换不含叠加。
        """
        slm = self._slm
        if max_grayscale is None:
            max_grayscale = int(getattr(slm, "_max_gray", 1023))
        # 离焦（弧度）：相位空间，先于 弧度→灰度 转换。输入先对齐到面板尺寸。
        panel_phase = slm._resize_to_panel(phase_rad)
        if self._overlay.has_phase:
            if self._overlay.phase_rad.shape == panel_phase.shape:
                panel_phase = panel_phase + self._overlay.phase_rad
            else:
                logger.warning(
                    "离焦叠加尺寸 %s 与面板 %s 不一致，已跳过（请检查离焦设置）",
                    self._overlay.phase_rad.shape,
                    panel_phase.shape,
                )
        # NaN 语义与驱动一致：叠加后、转换前 nan_to_num（驱动内再执行一次是
        # no-op，逐位等价）。
        np.nan_to_num(panel_phase, copy=False, nan=0)
        # 驱动纯转换：弧度→灰度 + 矫正CSV + LUT + 平移 + mod（驱动槽在代理
        # 路径上保持空，无叠加）。
        gray = slm.create_phase_from_array(panel_phase, max_grayscale=max_grayscale)
        # 底相位（灰度）：灰度空间，矫正/LUT/平移/mod 之后以灰度叠加（再 mod
        # 恒等）。
        if self._overlay.has_base:
            if self._overlay.base_gray.shape == gray.shape:
                gray = np.mod(
                    gray.astype(np.float64) + self._overlay.base_gray,
                    max_grayscale,
                ).astype(np.uint16)
            else:
                logger.warning(
                    "底相位尺寸 %s 与面板 %s 不一致，已跳过（请检查底相位设置）",
                    self._overlay.base_gray.shape,
                    gray.shape,
                )
        return gray

    def display_phase(
        self,
        phase_rad: Any,
        wait_time_s: float | None = None,
        memory_number: int | None = None,
        memory_mode: int = 0,
    ) -> int:
        """拦截 ``display_phase``：先经**代理** ``create_phase_from_array``（含
        叠加）转灰度，再透传给真实驱动的 ``display_data`` 上屏。

        若走 ``__getattr__`` 透传到裸驱动，驱动内部 ``self.create_phase_from_array``
        是裸转换、不含叠加——故此处必须显式拦截，与旧驱动槽在
        ``display_phase`` 路径上的叠加行为保持一致。
        """
        phase_gray = self.create_phase_from_array(phase_rad)
        return self._slm.display_data(
            phase_gray,
            wait_time_s=wait_time_s,
            memory_number=memory_number,
            memory_mode=memory_mode,
        )

    # ------------------------------------------------------------------
    # 透明透传：其余一切属性/方法/dunder 转发到真实驱动
    # ------------------------------------------------------------------
    def __getattr__(self, name: str) -> Any:
        # 仅在实例/类上都找不到时才走这里（__slots__ 的 _slm/_overlay 在实例
        # 字典，__init__ 里 object.__setattr__ 已设好，不会落到这里）。
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        try:
            return getattr(self._slm, name)
        except AttributeError as exc:
            raise AttributeError(f"OverlaidSantec has no attribute {name!r}") from exc

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ("_slm", "_overlay"):
            object.__setattr__(self, name, value)
        else:
            setattr(self._slm, name, value)

    def __repr__(self) -> str:
        slm_repr = repr(self._slm)
        return (
            f"OverlaidSantec({slm_repr}, "
            f"overlay=phase:{self._overlay.has_phase}/base:{self._overlay.has_base})"
        )

    # ------------------------------------------------------------------
    # 上下文管理器透传
    # ------------------------------------------------------------------
    # ``__getattr__`` 对所有 dunder 都 raise AttributeError（避免拦截 dunder
    # 查找），所以 ``with proxy`` 默认拿不到 ``__enter__``/``__exit__``。优化器
    # 与 GUI 都以 ``with slm:`` 形式使用句柄（``with slm, cam:`` / ``with
    # cam_ctx as cam, slm_ctx as slm:``），代理必须把生命周期委托给被包装的
    # 真实驱动（``Santec.__enter__`` → ``open()``、``__exit__`` → ``close()``）。
    def __enter__(self) -> Any:
        return self._slm.__enter__()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Any:
        return self._slm.__exit__(exc_type, exc_val, exc_tb)

    # Streamlit/pickle 边界：序列化时只序列化代理自身，不带走整个驱动。
    def __getstate__(self) -> dict[str, Any]:
        return {"_slm": self._slm, "_overlay": self._overlay}

    def __setstate__(self, state: dict[str, Any]) -> None:
        object.__setattr__(self, "_slm", state["_slm"])
        object.__setattr__(self, "_overlay", state["_overlay"])

    def __copy__(self) -> "OverlaidSantec":
        # 浅拷贝：共享 _slm 与 _overlay 引用（与代理语义一致）。直接返回新实例，
        # 不能再 _copy.copy(新实例) —— 那会递归触发本方法。
        return OverlaidSantec(self._slm, self._overlay)

    def __deepcopy__(self, memo: dict) -> "OverlaidSantec":
        import copy as _copy

        return OverlaidSantec(
            _copy.deepcopy(self._slm, memo),
            _copy.deepcopy(self._overlay, memo),
        )


def apply_overlays_to_handle(
    handle: Any, overlay: PhaseOverlay | None
) -> None:
    """把 ``overlay`` 幂等地应用到 ``handle`` 上（代理 or 裸驱动），无硬件 I/O。

    这是叠加层到句柄的**唯一分发点**，按句柄类型决定写入哪个真源，维护
    「一个句柄只叠加一次」的 invariant：

    * ``handle`` 是 :class:`OverlaidSantec` → 写入**代理层**，并把其**驱动槽清零**
      （``raw.set_phase_overlay(None)`` / ``raw.set_base_phase(None)``），使经
      代理的发送只叠加代理层、不双重叠加。
    * ``handle`` 是裸驱动（``Santec`` 或带 ``set_phase_overlay`` /
      ``set_base_phase`` 的替身）→ 写入**驱动槽**（上游 acb97b1f 既有设计）。
    * ``handle`` 为 ``None`` → 直接返回。

    写入失败（如设备未就绪）仅记录 warning，不抛异常（叠加层失败不卡连接/
    整形主流程）。
    """
    if handle is None:
        return
    phase = None if overlay is None else overlay.phase_rad
    base = None if overlay is None else overlay.base_gray
    try:
        if isinstance(handle, OverlaidSantec):
            handle.set_overlay(PhaseOverlay(phase, base))
            # 清空驱动槽，维持「经代理的发送只叠加代理层」的 invariant。
            raw = handle.raw
            if hasattr(raw, "set_phase_overlay"):
                raw.set_phase_overlay(None)
            if hasattr(raw, "set_base_phase"):
                raw.set_base_phase(None)
        else:
            if hasattr(handle, "set_phase_overlay"):
                handle.set_phase_overlay(phase)
            if hasattr(handle, "set_base_phase"):
                handle.set_base_phase(base)
    except Exception as exc:  # policy: 叠加层失败不卡连接/整形 - 叠加是显示语义，句柄本身仍可用
        logger.warning("叠加层应用到句柄失败: %s", exc)
