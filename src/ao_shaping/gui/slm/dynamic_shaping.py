"""SLM 动态整形的纯时间轴引擎与逐帧下发线程。

为什么是「灰度域插值」而不是「相位域插值」
--------------------------------------------
SLM 面板真正接收的是 **最终灰度码**（弧度→灰度 + 矫正 + LUT + 平移都已在
``create_phase_from_array`` 里做完）。两条相位各自再走一遍该管线后再按帧混合，
会有一个隐藏的二义性：相位插值的中间帧本身不是任何一端生成器的输出，却要被
再施加一次矫正 —— 端点正确、中间未必。

本模块改为：**起点与终点都先走一遍与静态下发完全相同的管线拿到最终灰度**，
然后在灰度码上做逐像素、走最短环绕路径的线性混合::

    Δ = ((target - start + 512) mod 1024) - 512
    frame(t) = (start + round(Δ · ease(t))) mod 1024

这样：

* 端点**逐码相等**（t=0 是起点、t=1 是目标），不会出现「过渡结束但差一点」；
* 中途帧不再经过任何编码/矫正，写进面板的就是这一帧算出来的码（``display_data``
  的 raw 路径），规避了「缓存的显示相位被二次矫正」的已知坑；
* 每帧的最大码变化可精确预估 —— 这正是 LCOS 翻转等待与「过渡速率是否被面板
  拉慢」的判据（:func:`estimate_transition`）。

最短环绕路径（而不是直接 ``target - start``）保证每个像素的码变化不超过
512 码（半周期），既减少相位跳变在中途造成的闪烁，也让同样的过渡时长里
单帧步进更小。

时间轴
------
帧号由**墙钟时间**决定（``t = 已用时间 / 时长``），不是「第 n 帧 / 总帧数」。
硬件慢时直接跳过中间帧，过渡仍在目标时长附近结束；时间轴暂停时把暂停时长
从墙钟里扣掉，继续后从同一进度接续，而不是重播。

线程与执行器句柄
----------------
:class:`DynamicShapingWorker` 是独立 ``QThread``（不是线程池任务）：一次过渡
持续数秒到数分钟，且只有「打开 → 连续下发 → 关闭」一条独占设备的路径。
暂停/继续/停止用 ``threading.Event`` 与后台线程通信，控件只在主线程碰。

整段过渡**只 open/close 一次**：``with Santec(...)`` 持有单一设备句柄，循环里
只用同一句柄写中间帧，结束 / 停止 / 异常都由 ``with`` 退出时关闭。绝不逐帧
open/close —— 项目实测背对背开关 SLM 会触发驱动板假死，UI 的暂停/继续/停止
因此都映射到这一个长生命周期执行器，而不是反复点「下发」。

每帧写**不同的内存槽**（:class:`~ao_shaping.drivers.slm.santec.SlotRotator`）：
固件把「重写正在显示的同一槽位」当作 no-op，同槽反复写会让 LCOS 静默不刷新，
中间帧根本不上屏。

每帧写完后停留 ``max(用户设定, 该帧最大码变化估出的翻转时间)``：像素没翻稳就
写下一帧，会被下一帧直接覆盖。停留由执行器统一控制（调用 ``display_data`` 时
``wait_time_s=0`` 关掉驱动内部的自动等待），所以 UI 的「每步停留」只是下限，
永远不能让停留短于物理翻转时间。

远场不是相位的线性函数
----------------------
灰度码逐帧逼近目标只保证**相位**在分中间步变化；远场强度是相位的傅里叶变换，
同样的码步长在远场上可能表现为亮度重分配而不是形状均匀变形。界面文案因此
只说「相位逐步演化 / 分多个中间步逼近目标」，不承诺光斑线性变形。开环插值
本身是近似；若后续接入闭环/分阶段优化，可把每帧的序列来源替换掉而不动 UI。
序列回放本身有先例：``scripts/slm/replay_saved_phases_20261029.py`` 按 label
把一串相位逐步显示；本页只是把「帧序列来源」换成由端点插值生成。
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from PySide6.QtCore import QThread, Signal

from ao_shaping.utils.io.file import ROOT_DIR

__all__ = [
    "EASING_LABELS",
    "MAX_DWELL_MS",
    "MAX_GRAY_CODE",
    "TransitionEstimate",
    "TransitionPlan",
    "DynamicShapingWorker",
    "blend_gray",
    "ease",
    "estimate_transition",
    "far_field_images",
    "frame_thumbnail",
    "gray_to_phase",
    "pixel_flip_wait_s",
    "wrapped_gray_delta",
]

# 纯时间轴数学（常量 / 缓动 / TransitionPlan / 灰度混合）已迁到 Qt 无关的
# ``utils/slm/transition_timeline.py``，供 headless 采集脚本共用同一份实现。
# 这里 re-export，公共接口与 ``__all__`` 保持不变。
from ao_shaping.utils.slm.transition_timeline import (
    EASING_LABELS,
    MAX_DWELL_MS,
    MAX_GRAY_CODE,
    TransitionEstimate,
    TransitionPlan,
    blend_gray,
    ease,
    estimate_transition,
    gray_to_phase,
    pixel_flip_wait_s,
    wrapped_gray_delta,
)
from ao_shaping.utils.slm.transition_timeline import (
    as_panel_array as _as_panel_array,
)


def frame_thumbnail(gray: np.ndarray, max_side: int = 420) -> np.ndarray:
    """当前帧的 8-bit 缩略图（uint8, 0…255），供界面实时显示。

    用步进抽样而不是插值：帧率高时每帧都要发一次缩略图，重采样会成为主线程
    的负担；缩略图只证明「画面在连续变化」，不承担测量职责。
    """
    arr = _as_panel_array(gray, "帧灰度")
    step = max(1, int(np.ceil(max(arr.shape) / float(max_side))))
    small = arr[::step, ::step]
    return np.clip(small.astype(np.uint16) >> 2, 0, 255).astype(np.uint8)


#: 远场示意的调色板（黑 → 深紫 → 品红 → 橙 → 米白）。与灰度相比，色阶能让
#: 「能量往哪边挪」在缩略图尺寸下也看得见。
_COLOR_STOPS = np.array([0.0, 0.35, 0.62, 0.84, 1.0], dtype=np.float64)
_COLOR_RGB = np.array(
    [
        [8, 9, 14],
        [52, 20, 78],
        [152, 36, 94],
        [238, 124, 58],
        [255, 247, 218],
    ],
    dtype=np.float64,
)


def _apply_colormap(norm: np.ndarray) -> np.ndarray:
    """0…1 强度 → RGB uint8（分段线性，与界面其余部分同一套暖色）。"""
    shaped = norm ** 0.9
    out = np.empty(shaped.shape + (3,), dtype=np.uint8)
    for channel in range(3):
        out[..., channel] = np.interp(
            shaped, _COLOR_STOPS, _COLOR_RGB[:, channel]
        ).astype(np.uint8)
    return out


def far_field_images(
    grays: list[np.ndarray],
    max_side: int = 240,
    sigma: float = 0.16,
) -> list[np.ndarray]:
    """把若干帧灰度算成同一归一化的远场强度 RGB 图（示意用）。

    模型是「等幅高斯照明 + 2f 傅里叶变换」：SLM 上一幅相位屏调制复振幅，
    透镜后焦面即其傅里叶谱。实际光束包络、像差与相机响应都不在此模型里，
    因此界面必须标注「示意」，不能当成实测预测。归一化在所有帧之间共用，
    否则每帧各自拉满会让「变化」看起来比实际更剧烈。
    """
    if not grays:
        return []
    fields: list[np.ndarray] = []
    for gray in grays:
        arr = _as_panel_array(gray, "帧灰度")
        step = max(1, int(np.ceil(max(arr.shape) / float(max_side))))
        small = arr[::step, ::step]
        phase = gray_to_phase(small)
        h, w = phase.shape
        yy = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
        xx = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
        envelope = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma)).astype(
            np.float32
        )
        fields.append(envelope * np.exp(1j * phase))
    intensities = [
        np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(field))) ** 2)
        for field in fields
    ]
    vmax = max(float(np.percentile(inten, 99.7)) for inten in intensities)
    vmax = max(vmax, 1e-9)
    images: list[np.ndarray] = []
    for inten in intensities:
        norm = np.clip(inten / vmax, 0.0, 1.0).astype(np.float32)
        images.append(_apply_colormap(norm))
    return images


def _sha256_codes(arr: np.ndarray) -> str:
    import hashlib

    return hashlib.sha256(np.ascontiguousarray(arr, dtype=np.uint16).tobytes()).hexdigest()


def write_transition_record(payload: dict[str, Any], path: Path | None = None) -> Path:
    """把过渡计划/结果写成 JSON（``data/debug/dynamic_shaping/``）。

    开跑前先落盘计划，进程中途被杀也留下一份可审计的意图；结束时重写同一
    文件补上结果。写盘失败只记日志、不阻断下发（与静态页的元数据策略一致）。
    """
    if path is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = ROOT_DIR / "data" / "debug" / "dynamic_shaping"
        path = out_dir / f"{stamp}_dynamic_transition.json"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
    except Exception as exc:  # 磁盘/权限/编码失败不得反过来阻断已获准的下发
        logger.warning("动态整形元数据落盘失败（不阻断下发）: {}", exc)
    return path


class DynamicShapingWorker(QThread):
    """把一幅起点灰度平滑过渡到目标相位的后台线程。

    只在用户点「开始过渡」并确认后构造。设备在本线程内打开并在结束时关闭；
    面板在关闭后仍保持最后一帧（LCOS 是保持型显示），所以「停止」不会自动回零。

    Signals:
        opened: 设备已打开、帧序列已就绪（主线程可把状态切成「过渡中」）。
        progress: ``(已发帧数 1 基, 总帧数, 有效已用秒)``。
        preview: 当前帧缩略图（``np.ndarray[uint8]``）。
        paused: 暂停状态变化。
        finished_ok: ``(已发帧数, 总耗时)``，最后一帧就是目标相位。
        stopped: ``(已发帧数, 总耗时)``，面板停在过渡中间。
        failed: ``(失败前已发帧数, 可读原因)``，设备异常/中断。
    """

    opened = Signal()
    progress = Signal(int, int, float)
    preview = Signal(object)
    paused = Signal(bool)
    finished_ok = Signal(int, float)
    stopped = Signal(int, float)
    failed = Signal(int, str)

    def __init__(
        self,
        *,
        slm_number: int,
        wavelength: int,
        target_phase: np.ndarray,
        plan: TransitionPlan,
        start_mode: str,
        start_phase: np.ndarray | None = None,
        start_gray: np.ndarray | None = None,
        pattern_type: str = "",
        pattern_label: str = "",
        params: dict[str, Any] | None = None,
        start_params: dict[str, Any] | None = None,
        optical_path: dict[str, Any] | None = None,
        parent: Any = None,
    ) -> None:
        super().__init__(parent)
        self._slm_number = int(slm_number)
        self._wavelength = int(wavelength)
        self._target_phase = np.asarray(target_phase)
        self._plan = plan
        if start_mode not in ("flat", "phase", "gray"):
            raise ValueError(f"未知起点类型: {start_mode!r}")
        self._start_mode = start_mode
        self._start_phase = None if start_phase is None else np.asarray(start_phase)
        self._start_gray = None if start_gray is None else np.asarray(start_gray)
        self._pattern_type = pattern_type
        self._pattern_label = pattern_label
        self._params = dict(params or {})
        self._start_params = dict(start_params or {})
        self._optical_path = dict(optical_path or {})

        #: set = 允许推进；clear = 暂停（线程在帧边界停住，不写面板）。
        self._run_event = threading.Event()
        self._run_event.set()
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------ 控制

    def request_pause(self) -> None:
        """请求暂停。线程在下一帧边界停住；当前帧保持显示。"""
        self._run_event.clear()

    def request_resume(self) -> None:
        self._run_event.set()

    def request_stop(self) -> None:
        """请求停止。正在写的那一帧会写完，之后关闭设备并发出 ``stopped``。"""
        self._stop_event.set()
        # 解除暂停，否则暂停中的线程收不到停止信号。
        self._run_event.set()

    @property
    def plan(self) -> TransitionPlan:
        return self._plan

    # ------------------------------------------------------------------ 执行

    def _resolve_start_gray(self, slm: Any, panel: tuple[int, int]) -> np.ndarray:
        """把界面选择的起点解析成最终灰度码。"""
        if self._start_mode == "flat":
            # 扁平相位严禁走 create_phase_from_array()（该函数对输入 mod 2π →
            # 灰度，且会叠加矫正/LUT/底相位，把"平场"变成非均匀码；见
            # drivers/slm/AGENTS.md「SLM flat-phase 严禁走 create_phase_from_array()」）。
            # 平场起点一律 raw 直发纯零码，与驱动 flat 的权威处置一致。
            gray = np.full(panel, 0, dtype=np.uint16)
        elif self._start_mode == "phase":
            if self._start_phase is None:
                raise ValueError("起点参数缺失：未生成起始相位")
            if self._start_phase.shape != panel:
                raise ValueError(
                    f"起始相位尺寸 {self._start_phase.shape} 与面板 {panel} 不一致"
                )
            gray = np.asarray(
                slm.create_phase_from_array(self._start_phase), dtype=np.uint16
            )
        else:
            if self._start_gray is None:
                raise ValueError("起点参数缺失：本页还没有可续接的上一帧")
            gray = np.asarray(self._start_gray, dtype=np.uint16)
            if gray.shape != panel:
                raise ValueError(f"上一帧尺寸 {gray.shape} 与面板 {panel} 不一致")
        if gray.shape != panel:
            raise RuntimeError(
                f"起点灰度尺寸 {gray.shape} 与面板 {panel} 不一致（驱动未按面板回填）"
            )
        return gray.copy()

    def _hold(self, seconds: float) -> float:
        """保持当前帧 ``seconds`` 秒，期间可被停止/暂停打断。

        返回未睡完的时间：>0 表示被打断（暂停时由主循环在恢复后续睡，停止时
        主循环直接退出）。分成 20 ms 小片，避免停止要等到一步停留结束才响应。
        """
        deadline = time.monotonic() + max(0.0, float(seconds))
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return 0.0
            if self._stop_event.is_set() or not self._run_event.is_set():
                return remaining
            time.sleep(min(0.02, remaining))

    def run(self) -> None:
        """线程入口：打开设备一次 → 逐帧写不同槽 → 关闭。异常一律转成 ``failed``。"""
        from ao_shaping.drivers.slm.santec import Santec, SlotRotator

        sent = 0
        elapsed = 0.0
        metadata_path: Path | None = None
        record: dict[str, Any] = {
            "slm_number": self._slm_number,
            "wavelength_nm": self._wavelength,
            "pattern_type": self._pattern_type,
            "pattern_label": self._pattern_label,
            "pattern_params": self._params,
            "start_mode": self._start_mode,
            "start_params": self._start_params,
            "plan": asdict(self._plan),
            "optical_path": self._optical_path,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "outcome": "started",
        }
        try:
            with Santec(
                slm_number=self._slm_number,
                wavelength=self._wavelength,
                video_mode=0,
            ) as slm:
                # 单句柄贯穿整段过渡：循环里只写帧，不 open/close。退出 with 时关闭。
                slm.set_phase_overlay(None)
                rotator = SlotRotator(slm)
                panel = (int(slm.Panel_Res[1]), int(slm.Panel_Res[0]))
                start_gray = self._resolve_start_gray(slm, panel)
                target_gray = np.asarray(
                    slm.create_phase_from_array(self._target_phase), dtype=np.uint16
                )
                if target_gray.shape != panel:
                    raise RuntimeError(
                        f"目标灰度尺寸 {target_gray.shape} 与面板 {panel} 不一致"
                    )
                estimate = estimate_transition(self._plan, start_gray, target_gray)
                record.update(
                    {
                        "start_gray_sha256": _sha256_codes(start_gray),
                        "target_gray_sha256": _sha256_codes(target_gray),
                        "estimate": asdict(estimate),
                        "executor": "single-handle + SlotRotator",
                    }
                )
                metadata_path = write_transition_record(record)

                self.opened.emit()
                n = self._plan.frame_count
                interval = self._plan.frame_interval_s
                user_dwell = self._plan.dwell_s
                preview_stride = max(1, int(round(self._plan.fps / 10.0)))
                t0 = time.monotonic()
                paused_at: float | None = None
                last_index = -1
                pending_hold = 0.0
                prev_frame: np.ndarray | None = None

                while sent < n:
                    if self._stop_event.is_set():
                        break

                    # 暂停：在帧边界停住，保持当前帧显示；继续后把暂停时长从
                    # 墙钟里扣掉，进度从原处接续。
                    if not self._run_event.is_set():
                        if paused_at is None:
                            paused_at = time.monotonic()
                            self.paused.emit(True)
                        self._run_event.wait(0.05)
                        continue
                    if paused_at is not None:
                        t0 += time.monotonic() - paused_at
                        paused_at = None
                        self.paused.emit(False)

                    # 暂停打断了「每步停留」：继续时先把没停够的时间补回来，
                    # 否则暂停会让某些中间帧没翻稳就被下一帧覆盖。
                    if pending_hold > 0.0:
                        pending_hold = self._hold(pending_hold)
                        if pending_hold > 0.0:
                            continue

                    elapsed = time.monotonic() - t0
                    index = min(
                        n - 1,
                        int(max(0.0, elapsed) / self._plan.duration_s * (n - 1)),
                    )
                    if index <= last_index:
                        pending_hold = self._hold(min(0.01, interval / 4.0))
                        continue

                    frame = blend_gray(
                        start_gray,
                        target_gray,
                        ease(self._plan.progress_of_frame(index), self._plan.easing),
                    )
                    # 每步停留 = max(用户设定, 该帧最大码变化估出的翻转时间)；
                    # 首帧没有上一帧基准，保守按满幅翻转处理。
                    if prev_frame is None:
                        step_gray = MAX_GRAY_CODE
                    else:
                        step_gray = int(
                            np.abs(
                                frame.astype(np.int32) - prev_frame.astype(np.int32)
                            ).max()
                        )
                    dwell = max(pixel_flip_wait_s(step_gray), user_dwell)
                    # 逐帧轮换内存槽：同槽重写在固件里是 no-op，LCOS 不会刷新。
                    slot = rotator.next_slot()
                    slm.display_data(frame, wait_time_s=0, memory_number=slot)
                    prev_frame = frame
                    last_index = index
                    sent = index + 1
                    self.progress.emit(sent, n, elapsed)
                    if sent == 1 or index % preview_stride == 0 or index == n - 1:
                        self.preview.emit(frame_thumbnail(frame))

                    # 配速：至少停留 dwell，再按名义间隔补到下一帧的时刻。
                    # 落后就立即发下一帧（时间轴优先，跳帧不拖慢）。
                    next_due = t0 + (index + 1) * interval
                    pending_hold = self._hold(
                        max(dwell, next_due - time.monotonic())
                    )

                elapsed = time.monotonic() - t0
                record.update(
                    {
                        "outcome": "stopped" if self._stop_event.is_set() else "completed",
                        "frames_sent": sent,
                        "frames_planned": n,
                        "elapsed_s": round(elapsed, 3),
                        "finished_at": datetime.now().isoformat(timespec="seconds"),
                    }
                )
                write_transition_record(record, metadata_path)
                if self._stop_event.is_set():
                    self.stopped.emit(sent, elapsed)
                else:
                    self.finished_ok.emit(sent, elapsed)
        except Exception as exc:  # 设备边界：任何异常都必须变成界面结论
            logger.exception(
                "动态整形中断 (SLM {}, 第 {} 帧): {}", self._slm_number, sent + 1, exc
            )
            record.update(
                {
                    "outcome": "failed",
                    "frames_sent": sent,
                    "elapsed_s": round(elapsed, 3),
                    "error": str(exc),
                    "finished_at": datetime.now().isoformat(timespec="seconds"),
                }
            )
            write_transition_record(record, metadata_path)
            self.failed.emit(
                sent,
                f"{exc} —— 面板保留第 {sent} 帧；请检查 SLM {self._slm_number} 连接后重试。",
            )
