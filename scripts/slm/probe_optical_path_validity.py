"""光路有效性探针: 曝光阶梯 + 离焦响应, 判定光路能不能继续做几何标定。

为什么需要它
------------
2026-10-22 的 MIL 硬件 campaign 两次死在几何标定, 报的是
``no swept mode produced a usable width response; the spot width does not
respond to any of them on this bench``
(``optimizer/wfless/model_in_loop_shaping.py::calibrate_bench_geometry_from_sweep``)。
那条报错只说「宽度不响应」, 完全没提**信噪比不足**这一层: 当时的曝光窗口是
1-10 ms (峰值 24-213 counts), 而几何 sweep 靠「光斑宽度随扫描模式变化」来拟合
焦面标度 —— 那个宽度响应整个淹在噪声里。

所以几何标定之前应当先有一个**前置判定**: 光路到底通不通、够不够亮、SLM 到底
有没有在调制。这个探针一次跑完给出四态判定:

* ``ok`` —— 光斑明显高于噪声底, **且**离焦带来了显著响应 (宽度或峰值的变化超过
  噪声水平) ⇒ 光路有效, 可以继续标定。
* ``low_snr`` —— 光斑勉强高于噪声底, 或离焦响应淹没在噪声里 ⇒ 光路未必坏, 是
  **曝光/信噪比不足**; 给出建议曝光 (峰值落在 ~100-200 counts 的那一档)。
* ``weak_path`` —— 光斑可见、离焦也响应, 但**每毫秒峰值只有本系统历史 ``ok``
  探测中位数的 X%** ⇒ 光路被部分遮挡 / 衰减了。这一态是 2026-10-29 实测补上的:
  当天把光路挡掉一部分后, 探针仍判 ``ok`` (峰值 68 counts 在噪声底 4 counts 之上,
  离焦响应也显著), 而光通量已从 6.15 掉到 1.38 counts/ms。**曝光能补回峰值,
  补不回被挡掉的能量** —— 若靠加大曝光把它抬进 100-200 counts 再判 ok, 整段标定
  就建在衰减后的光束上。
* ``no_response`` —— 光斑清晰可见但**离焦毫无响应** ⇒ 光路或 SLM 可疑 (面板没调制 /
  光束不在模型以为的位置 / 离焦没真正加上去); 给出排查方向
  (``tools/slm/slm_diagnose.py`` 的三步链: 冻结 / 调制 / 线性)。

**判定必须给出实测数字** (噪声底、可用窗口、离焦前后的峰值与宽度、判定阈值),
不许只给结论。

证据落盘: ``data/debug/optical_path_probe/<UTC戳>/`` —— 每档曝光与每种离焦状态的
npz (原始帧)、``optical_path_probe.json`` (全部读数 + 判定 + 阈值 + 建议)、
``optical_path_probe.png`` (曝光-峰值曲线 + 离焦响应曲线)。

退出码: ``0`` = 判定为 ``ok``; ``2`` = 判定为 ``low_snr`` / ``weak_path`` /
``no_response`` (可以当上层门禁用)。

用法::

    python scripts/slm/probe_optical_path_validity.py --wait-until-idle 600
    python scripts/slm/probe_optical_path_validity.py --session s200_circle --wait-until-idle 600

``--session`` 只影响证据落盘位置 (``<out-root>/<session>/<UTC戳>/``): 生成器的每条
lane 都带它, 于是「这次探测门禁的是哪条 lane」在目录名上就能对出来。
"""

from __future__ import annotations

import argparse
from loguru import logger
import importlib.util
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _candidate in (ROOT, ROOT / "src", ROOT / "libs", ROOT / "libs" / "OOPAO"):
    _text = str(_candidate)
    if _text not in sys.path:
        sys.path.insert(0, _text)

from ao_shaping.drivers.ccd.common import (
    create_camera,
    get_camera_exposure_ms,
    get_camera_exposure_range,
    set_camera_exposure_ms,
)
from ao_shaping.tools.kernels import device_mutex

# Fresnel D 段的缺省光瞳几何 = fresnel_panel 自己的缺省 (面板坐标 px)。
from ao_shaping.tools.kernels.bench_kernels import (
    BEAM_CENTER_PANEL as FOCUS_BEAM_CENTER_PANEL,
    BEAM_RADIUS_PANEL as FOCUS_BEAM_RADIUS_PANEL,
)
from ao_shaping.tools.kernels.device_mutex import DeviceLock, WAITABLE_KINDS
from ao_shaping.utils.image.beam_metrics import (
    beam_radius_second_moment,
    prepare_frame,
)
from ao_shaping.utils.image.frame_analysis import (
    DEFAULT_MIN_PEAK,
    measure_spot,
    spot_fwhm,
)
from ao_shaping.utils.io.optical_path_record import build_optical_path_record
from ao_shaping.utils.slm.bench_constants import (
    BEAM_CENTER_PANEL,
    BEAM_RADIUS_PANEL,
    SLM_PANEL_H,
    SLM_PANEL_W,
)
from ao_shaping.utils.slm.phase_display import (
    display_phase as display_phase_on_slm,
)
from ao_shaping.tools.slm.panel.display_state import slm_self_check
from ao_shaping.tools.slm.panel.focus_probe import (
    EXPOSURE_MODE_ADAPT as FOCUS_EXPOSURE_MODE_ADAPT,
    EXPOSURE_MODE_FIXED as FOCUS_EXPOSURE_MODE_FIXED,
    FIXED_REPEAT_MAX as FOCUS_REPEAT_MAX,
    FIXED_REPEAT_MIN as FOCUS_REPEAT_MIN,
    FLOOR_MS as FOCUS_FLOOR_MS,
    sweep_focus,
)
from ao_shaping.utils.wavefront.zernike_utils import (
    generate_zernike_phase,
)

#: Fresnel D 标定段的缺省值 (设备无关实现在 focus_probe.sweep_focus)。
#: 这里的 coarse/range 比 focus_probe 自己的缺省更窄: S0 会话里 D 只是一个附带
#: 标定项, 不能让它把整次设备会话拖成一次完整的焦面探测。
FOCUS_COARSE_POINTS = 13
FOCUS_DIOPTER_RANGE = 1.5
FOCUS_REFINE_POINTS = 7
FOCUS_FRAMES = 3
#: 曝光线性度必须落在 [0.8, 1.25] 才相信 peak/ms 排序 (focus_probe 同一判据)。
FOCUS_LINEARITY_MIN = 0.8
FOCUS_LINEARITY_MAX = 1.25
#: 预注册 R1 §10.4 D1: 最优点落在边界时**只**允许自动外扩一次。
FOCUS_MAX_EXTENDS = 1

#: 采集驱动 (``scripts/slm/run_hw_sweep_groups.py``) —— 设备互斥的探针与分类
#: 都在那里, 这里按路径加载复用, 不另写一份。
SWEEP_SCRIPT = ROOT / "scripts" / "slm" / "run_hw_sweep_groups.py"

#: 待落盘的标准光路记录 (SKILL §4): ``run_probe`` 只组装输入, ``main`` 在**放锁
#: 之后**才补齐 ``occupancy.released_at`` 并写盘 —— 记录写完之后才释放设备,
#: ``released_at`` 就只能永远是 null。
_PENDING_RECORD: dict[str, Any] | None = None

# ---------------------------------------------------------------------------
# 四态判定与它的阈值: 只有一份实现 (``tools/slm/panel/probe_policy``)。
# GUI 的「光路快速探测」按钮消费同一个 ``decide`` —— 这里 import 回来, 不写第二份。
# ---------------------------------------------------------------------------
from ao_shaping.tools.slm.panel.probe_policy import (
    PEAK_RELATIVE_FLOOR,
    PEAK_RESOLUTION_MAX,
    RESPONSE_SIGMAS,
    STATE_LOW_SNR,
    STATE_NO_RESPONSE,
    STATE_OK,
    STATE_WEAK_PATH,
    WIDTH_RELATIVE_FLOOR,
    WIDTH_RESOLUTION_MAX,
    decide,
)

# ---------------------------------------------------------------------------
# 采集侧阈值 (全部写死在这里, 报告会原样抄进 JSON, 便于事后核对)。
# 判定侧阈值 (显著性倍数 / 分辨率上限 / 相对变化下限) 与 ``decide`` 一起在
# ``probe_policy`` 里, 见上面的 import。
# ---------------------------------------------------------------------------
#: 噪声底的显著性倍数: 峰值必须超过 ``中位数 + k * sigma`` 才算「有光」。
#: 与 ``utils/image/exposure_control.py::NOISE_FLOOR_SIGMAS`` 同值 (4.0)。
NOISE_FLOOR_SIGMAS = 4.0
#: 帧的 MAD 恰好为 0 时用的噪声尺度 (1 count = 数字化的最小可表达差异)。
NOISE_FLOOR_FALLBACK_SIGMA = 1.0
#: 峰值达到满量程的这个比例就算饱和 (该档不再可用)。
SATURATION_FRACTION = 0.95
#: 宽度的 sigma 下限 (px) —— 亚像素级的散布按 0 计会让判据过于苛刻。
FWHM_SIGMA_FLOOR_PX = 0.3
#: 峰值的 sigma 下限 (counts)。
PEAK_SIGMA_FLOOR_COUNTS = 1.0
#: 建议曝光的目标峰值区间 —— **0-255 灰度口径**, 不是原始 counts。
#:
#: 它是在 8-bit 时代标定的 (那时 counts 就是 0-255 灰度)。12/16-bit 相机上
#: ``peak_raw`` 是 0-4095 / 0-65535, 直接拿 100-200 去比只剩满量程的 2.4%-4.9%,
#: 会挑到远低于目标的暗档 (S0 的 Mono12 重跑就会踩到)。``analyse_ladder`` 按
#: ``full_scale / 255`` 折算成 raw counts 再比较, 于是「满量程的 39%-78%」在
#: 所有位深上一致。
RECOMMENDED_PEAK_RANGE = (100.0, 200.0)
#: Noll 4 = 离焦 (仓库约定)。
DEFOCUS_NOLL_INDEX = 4
#: S0 前置条件要求的相机位深 (bit)。8 bit 下 1 count ≈ 7% 满量程, 均匀度/线性度
#: 结论会被量化噪声主导 (见 docs/dev/hw_plan_consolidated_20261118.md S0);
#: ``--bit-depth`` 的默认值就是它, 位深是满量程的唯一来源 (ccd/AGENTS.md 满量程)。
S0_BIT_DEPTH = 12

#: 判定「这一帧就是平场参考」的离焦容差: 名义 Δz = 0 的那些帧, 其 ``defocus_rad``
#: 与 0 只差在浮点误差量级内, 用严格 ``== 0`` 会把它们错判成非平场。只适用于离焦轴 ——
#: 第 255 行判断的是**振幅**是否为 0, 那是另一个量, 故意不共用这个容差。
FLAT_DEFOCUS_EPS_RAD = 1e-12
#: 建议曝光的落点 (**0-255 灰度**口径) —— 区间的中点; ``analyse_ladder`` 按
#: ``full_scale / 255`` 折算成 raw counts 后再挑最近的一档。
RECOMMENDED_PEAK_TARGET = 0.5 * (
    RECOMMENDED_PEAK_RANGE[0] + RECOMMENDED_PEAK_RANGE[1]
)


#: 光通量相对历史 ``ok`` 中位数的下限。低于它判 ``weak_path``。
#:
#: 口径来自 2026-10-29 的同一台实验系统: 19 次 ``ok`` 探测的 counts/ms 中位数 6.15
#: (相互离散在 5.6-8.0, 另有一次 20.6 的强耦合), 而部分挡光后掉到 1.38 —— 即 22%。
#: 取 0.5 既抓得住这种量级的衰减 (还留了 2 倍余量给光源老化/重新标定), 也不会因为
#: 一次 ±20% 的日常漂移就报警。
DEFAULT_MIN_THROUGHPUT_RATIO: float = 0.5

#: 参考值至少要由这么多次历史 ``ok`` 探测给出; 样本更少时中位数本身就在漂, 不下判定。
MIN_REFERENCE_SAMPLES: int = 3

#: 落盘式重定基的文件名 (放在 ``--out-root`` 下)。历史中位数在两种情况下没法用:
#: 光功率被改过 (整批数的量级全变), 或按序列号匹配后只剩几条过期的旧样本。
#: 2026-10-29 实测: 当前 SLM 序列号 23020027、当天上午的记录是 22030108, 于是
#: 参考落到 3.45 counts/ms, 而当场实测 28.22 被判成「739%」仍然 ``ok`` —— 门禁被
#: 稀释到只剩「完全无光」还挡得住。``--rebaseline`` 是操作者的一次显式声明:
#: 「以这一刻的光通量为新的正常值」, 下一次探测起生效。
REFERENCE_BASELINE_FILE: str = "reference_baseline.json"

#: 历史参考的中位数若是这么久的旧记录, 提示它可能在光功率/器件变动后失效。
REFERENCE_STALE_DAYS: float = 2.0


@dataclass
class FrameStats:
    """一帧的读数。

    **原始帧与处理后帧分开记**: ``prepare_frame`` 做的是「减中值 + clip 到 0」,
    它把探测器本底拿掉, 于是 ``peak_prepared`` 与 ``peak_raw`` 差一个本底。
    噪声底、饱和判据用原始帧 (绝对 counts), 形状类指标用处理后帧。
    """

    peak_raw: float
    peak_prepared: float
    mean_raw: float
    median_raw: float
    p99_raw: float
    std_raw: float
    fwhm_px: float
    fwhm_raw_px: float
    second_moment_radius_px: float
    centroid_x: float
    centroid_y: float
    hollowness: float

    def as_dict(self) -> dict[str, float]:
        """转成可直接 JSON 化的字典。"""
        return {
            "peak_raw": self.peak_raw,
            "peak_prepared": self.peak_prepared,
            "mean_raw": self.mean_raw,
            "median_raw": self.median_raw,
            "p99_raw": self.p99_raw,
            "std_raw": self.std_raw,
            "fwhm_px": self.fwhm_px,
            "fwhm_raw_px": self.fwhm_raw_px,
            "second_moment_radius_px": self.second_moment_radius_px,
            "centroid_x": self.centroid_x,
            "centroid_y": self.centroid_y,
            "hollowness": self.hollowness,
        }


def frame_stats(frame: np.ndarray) -> FrameStats:
    """量一帧: 峰值/均值/p99/质心/FWHM/二阶矩半径。

    宽度用 :func:`spot_fwhm` (与几何 sweep 同一个口径), 但喂给它的是
    :func:`prepare_frame` 处理过的帧 —— 原始帧带着探测器本底, 半高位置会被本底
    推偏, 宽度读数会偏大。二阶矩半径用
    :func:`beam_radius_second_moment` (ISO-11146), 同样在处理后帧上算。
    """
    raw = np.asarray(frame, dtype=np.float64)
    prepared = prepare_frame(raw)
    spot = measure_spot(prepared)
    return FrameStats(
        peak_raw=float(raw.max()),
        peak_prepared=float(prepared.max()),
        mean_raw=float(raw.mean()),
        median_raw=float(np.median(raw)),
        p99_raw=float(np.percentile(raw, 99)),
        std_raw=float(raw.std()),
        fwhm_px=float(spot_fwhm(prepared)),
        fwhm_raw_px=float(spot_fwhm(raw)),
        second_moment_radius_px=float(beam_radius_second_moment(prepared)),
        centroid_x=float(spot.centroid_x),
        centroid_y=float(spot.centroid_y),
        hollowness=float(spot.hollowness),
    )


def robust_noise_floor(frame: np.ndarray) -> dict[str, float]:
    """从一帧估计噪声底 (中位数 / sigma / 阈值 / 峰值)。

    sigma 用稳健尺度 ``1.4826 * MAD``: 光斑只占画幅的一小块, 不会把 MAD 抬起来。
    阈值是 ``中位数 + NOISE_FLOOR_SIGMAS * sigma`` —— 峰值超过它才算「有光」。
    """
    data = np.asarray(frame, dtype=np.float64)
    median = float(np.median(data))
    mad = float(np.median(np.abs(data - median)))
    sigma = 1.4826 * mad
    if sigma <= 0.0:
        sigma = NOISE_FLOOR_FALLBACK_SIGMA
    return {
        "median_counts": median,
        "sigma_counts": sigma,
        "threshold_counts": median + NOISE_FLOOR_SIGMAS * sigma,
        "peak_counts": float(data.max()),
    }


def build_defocus_phase(
    amplitude_rad: float,
    radius_px: int,
    pupil_center: tuple[int, int] = BEAM_CENTER_PANEL,
    panel_shape: tuple[int, int] = (SLM_PANEL_H, SLM_PANEL_W),
) -> np.ndarray:
    """离焦相位 (raw 未包裹弧度), 走 canonical 的 Zernike 入口。

    Noll 4 = 离焦 (仓库约定)。幅度是 **Noll 单位 RMS** 归一化下的系数, 所以
    幅度 1.0 = 瞳面内 RMS 相位 1 rad (瞳面边缘约 ±1.73 rad)。

    离焦盘**以瞳面中心为准** (``pupil_center``, 面板坐标), 而不是面板几何中心:
    离焦盘一旦偏心, 它带上一级倾斜与像散, 就不再是纯离焦了。孔径外的相位按 0
    处理 (平场), 与 ``tools/kernels/sweep_analysis.py`` 的同一口径。

    ``generate_zernike_phase`` 在孔径外返回 NaN, 这里填 0。
    """
    height, width = int(panel_shape[0]), int(panel_shape[1])
    if abs(float(amplitude_rad)) < 1e-12:
        return np.zeros((height, width), dtype=np.float64)
    radius = int(max(int(radius_px), 1))
    cx = min(max(int(pupil_center[0]), 0), width - 1)
    cy = min(max(int(pupil_center[1]), 0), height - 1)
    if (cx, cy) == (width // 2, height // 2):
        # 瞳面中心就是面板中心 ⇒ 直接在整个面板上生成 (canonical 调用, 与其它
        # 脚本逐位一致)。
        raw = generate_zernike_phase(
            {DEFOCUS_NOLL_INDEX: float(amplitude_rad)},
            resolution=(width, height),
            n_max=DEFOCUS_NOLL_INDEX,
            radius=radius,
        )
        return np.nan_to_num(
            np.asarray(raw, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0
        )
    # 瞳面中心偏离面板中心 ⇒ 在以瞳面为准的方块上生成再贴回面板 (与
    # ``tools/kernels/bench_kernels.py::zernike_panel`` 同一几何)。
    size = 2 * radius + 1
    patch = generate_zernike_phase(
        {DEFOCUS_NOLL_INDEX: float(amplitude_rad)},
        resolution=(size, size),
        n_max=DEFOCUS_NOLL_INDEX,
        radius=radius,
    )
    patch = np.nan_to_num(np.asarray(patch, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    out = np.zeros((height, width), dtype=np.float64)
    x0, y0 = cx - radius, cy - radius
    x1, y1 = cx + radius + 1, cy + radius + 1
    dx0, dy0 = max(0, -x0), max(0, -y0)
    x0c, y0c = max(x0, 0), max(y0, 0)
    x1c, y1c = min(x1, width), min(y1, height)
    if x1c > x0c and y1c > y0c:
        out[y0c:y1c, x0c:x1c] = patch[dy0 : dy0 + (y1c - y0c), dx0 : dx0 + (x1c - x0c)]
    return out


def capture_frames(cam: Any, n_frames: int) -> list[np.ndarray]:
    """连抓 ``n_frames`` 帧原始画面。"""
    frames: list[np.ndarray] = []
    for _ in range(max(1, int(n_frames))):
        frames.append(np.asarray(cam.get_numpy_image(n_sample=1)))
    return frames


def _save_npz(
    path: Path,
    frames: list[np.ndarray],
    meta: dict[str, Any],
) -> None:
    """把原始帧与元信息落成一个 npz。"""
    np.savez_compressed(path, frames=np.stack(frames), **{k: np.asarray(v) for k, v in meta.items()})


def _frame_full_scale(cam: Any, frame: np.ndarray) -> float:
    """一帧的生效满量程 —— **绑定相机生效位深, 不绑帧 dtype**。

    大恒 MONO12 的帧 dtype 是 ``uint16``, 按 ``np.iinfo(dtype).max`` 会得到 65535
    而真实满量程是 4095 (issue #246): 饱和帧被误判成「过暗」, 后续的饱和/线性度
    判据全错量纲。相机契约 (``effective_full_scale`` → ``_full_scale``) 是唯一口径;
    dtype 只在没有相机信息时兜底 (ffmpeg / image_folder 这类回放后端)。
    """
    effective = getattr(cam, "effective_full_scale", None)
    if callable(effective):
        try:
            value = float(effective(frame))
        except Exception:  # 兜底路径不能因为一个后端实现缺陷而中断采集
            value = 0.0
        if value > 0.0:
            return value
    fallback = getattr(cam, "_full_scale", None)
    if callable(fallback):
        try:
            value = float(fallback(np.asarray(frame).dtype))
        except Exception:  # 同上: 兜底失败才落到 dtype 推导
            value = 0.0
        if value > 0.0:
            return value
    dtype = np.asarray(frame).dtype
    return (
        float(np.iinfo(dtype).max) if np.issubdtype(dtype, np.integer) else 255.0
    )


def measure_exposure_ladder(
    cam: Any,
    exposures_ms: list[float],
    n_frames: int,
    out_dir: Path,
) -> list[dict[str, Any]]:
    """曝光阶梯: 每档抓 ``n_frames`` 帧, 记峰值/均值/p99/质心/FWHM。

    阶梯的每一档都对应一个 npz (``exposure_<ms>ms.npz``), 存的是**原始帧**。
    """
    readings: list[dict[str, Any]] = []
    for requested in exposures_ms:
        set_camera_exposure_ms(cam, float(requested))
        effective = float(get_camera_exposure_ms(cam))
        frames = capture_frames(cam, n_frames)
        stats = [frame_stats(frame) for frame in frames]
        peaks = [s.peak_raw for s in stats]
        noise = robust_noise_floor(frames[0])
        full_scale = _frame_full_scale(cam, frames[0])
        record: dict[str, Any] = {
            "requested_ms": float(requested),
            "effective_ms": effective,
            "n_frames": len(frames),
            "full_scale": full_scale,
            "peak_raw": float(max(peaks)),
            "peak_raw_per_frame": [float(p) for p in peaks],
            "peak_prepared": float(max(s.peak_prepared for s in stats)),
            "mean_raw": float(np.mean([s.mean_raw for s in stats])),
            "median_raw": float(np.mean([s.median_raw for s in stats])),
            "p99_raw": float(np.max([s.p99_raw for s in stats])),
            "fwhm_px": float(np.median([s.fwhm_px for s in stats])),
            "fwhm_per_frame": [float(s.fwhm_px) for s in stats],
            "second_moment_radius_px": float(
                np.median([s.second_moment_radius_px for s in stats])
            ),
            "centroid_x": float(np.median([s.centroid_x for s in stats])),
            "centroid_y": float(np.median([s.centroid_y for s in stats])),
            "hollowness": float(np.median([s.hollowness for s in stats])),
            "noise_median_counts": noise["median_counts"],
            "noise_sigma_counts": noise["sigma_counts"],
            "noise_threshold_counts": noise["threshold_counts"],
            "npz": f"exposure_{float(requested):g}ms.npz",
        }
        _save_npz(
            out_dir / record["npz"],
            frames,
            {
                "requested_ms": float(requested),
                "effective_ms": effective,
                "peak_raw": record["peak_raw"],
            },
        )
        readings.append(record)
    return readings


def measure_defocus_sweep(
    cam: Any,
    slm: Any,
    amplitudes_rad: list[float],
    exposure_ms: float,
    n_frames: int,
    radius_px: int,
    pupil_center: tuple[int, int],
    settle_s: float,
    out_dir: Path,
) -> list[dict[str, Any]]:
    """离焦扫描: 平场与若干离焦幅度之间切换, 每个状态取同样曝光下的帧。

    末尾再做一次**平场复测**, 用来量整轮的漂移 (面板重显示 + 曝光不变), 也就是
    「同一个状态能重复测到多少」的真实验证上限。
    """
    set_camera_exposure_ms(cam, float(exposure_ms))
    effective = float(get_camera_exposure_ms(cam))
    states: list[dict[str, Any]] = []

    def _capture_state(amplitude: float, tag: str) -> dict[str, Any]:
        phase = build_defocus_phase(amplitude, radius_px, pupil_center)
        display_phase_on_slm(slm, phase, settle_s)
        frames = capture_frames(cam, n_frames)
        stats = [frame_stats(frame) for frame in frames]
        record: dict[str, Any] = {
            "tag": tag,
            "defocus_rad": float(amplitude),
            "exposure_ms": float(exposure_ms),
            "effective_ms": effective,
            "n_frames": len(frames),
            "peak_raw": float(np.mean([s.peak_raw for s in stats])),
            "peak_raw_max": float(max(s.peak_raw for s in stats)),
            "peak_raw_per_frame": [float(s.peak_raw) for s in stats],
            "peak_prepared": float(max(s.peak_prepared for s in stats)),
            "fwhm_px": float(np.median([s.fwhm_px for s in stats])),
            "fwhm_per_frame": [float(s.fwhm_px) for s in stats],
            "second_moment_radius_px": float(
                np.median([s.second_moment_radius_px for s in stats])
            ),
            "centroid_x": float(np.median([s.centroid_x for s in stats])),
            "centroid_y": float(np.median([s.centroid_y for s in stats])),
            "hollowness": float(np.median([s.hollowness for s in stats])),
            "npz": f"defocus_{tag}.npz",
        }
        _save_npz(
            out_dir / record["npz"],
            frames,
            {
                "defocus_rad": float(amplitude),
                "exposure_ms": float(exposure_ms),
                "peak_raw": record["peak_raw"],
            },
        )
        return record

    for amplitude in amplitudes_rad:
        tag = f"{float(amplitude):+g}rad".replace("+", "p").replace("-", "m")
        states.append(_capture_state(float(amplitude), tag))
    states.append(_capture_state(0.0, "flat_repeat"))
    return states


def ladder_flux(ladder: list[dict[str, Any]]) -> dict[str, Any]:
    """峰值 counts/ms —— 只用**未饱和**的可用档拟合, 以及参与拟合的档位 (ms)。

    为什么单独成函数: 它有两个消费者。① 本次判定 (:func:`analyse_ladder`);
    ② :func:`bench_reference` 读**旧**记录 —— 2026-10-29 之前落盘的 json 没有
    ``ladder_summary.counts_per_ms``, 只能从它自己的 ``exposure_ladder`` 按同一
    口径重算, 否则历史参考恒为空、这条门禁永远不生效 (首次上线实测就踩到了)。

    为什么用峰值/毫秒而不是全帧总和: 遮挡把光**散掉**, 不是均匀减光 —— 2026-10-29
    实测挡掉一部分后峰值掉到 22% (6.15 → 1.38 counts/ms), 而全帧总和只掉约 30%。
    饱和档的峰值被满量程削平, 进拟合会把斜率压得比真实值低, 让正常光路看起来像
    衰减, 所以一律排除。
    """
    points: list[dict[str, Any]] = []
    for record in ladder:
        usable = record.get("usable")
        if usable is None:  # 旧记录只保证有 carries_signal / saturated
            usable = bool(record.get("carries_signal")) and not bool(
                record.get("saturated")
            )
        if usable:
            points.append(record)
    points.sort(key=lambda r: float(r["effective_ms"]))
    flux = float("nan")
    if len(points) >= 2:
        xs = np.array([float(r["effective_ms"]) for r in points])
        ys = np.array([float(r["peak_raw"]) for r in points])
        design = np.stack([np.ones(xs.size), xs], axis=1)
        flux = float(np.linalg.lstsq(design, ys, rcond=None)[0][1])
    elif points:
        flux = float(points[-1]["peak_raw"]) / float(points[-1]["effective_ms"])
    return {
        "counts_per_ms": flux,
        "window_ms": [float(r["effective_ms"]) for r in points],
    }


def analyse_ladder(
    ladder: list[dict[str, Any]],
    full_scale: float,
) -> dict[str, Any]:
    """从曝光阶梯里取出噪声底、可用窗口与建议曝光。

    * **噪声底**: 最短曝光档那一帧的 ``中位数 + 4 * sigma`` 与它的峰值。
    * **可用窗口**: 峰值高于噪声底、且没饱和的档位。
    * **建议曝光**: 可用窗口里峰值落在 ~100-200 **灰度** 的那一档 (目标带按
      ``full_scale / 255`` 折算成 raw counts, 所以 8/12/16 bit 上含义一致);
      没有就取可用窗口里最亮且不饱和的那一档。
    """
    shortest = min(ladder, key=lambda r: r["effective_ms"])
    scales = [float(r["full_scale"]) for r in ladder if "full_scale" in r]
    if scales:
        full_scale = max(scales)
    noise_floor = {
        "exposure_ms": float(shortest["effective_ms"]),
        "median_counts": float(shortest["noise_median_counts"]),
        "sigma_counts": float(shortest["noise_sigma_counts"]),
        "threshold_counts": float(shortest["noise_threshold_counts"]),
        "peak_counts": float(shortest["peak_raw"]),
    }
    saturation_counts = SATURATION_FRACTION * full_scale
    # 目标带以 0-255 灰度标定 (8-bit 时代 counts == 灰度), 这里折算到本次实测
    # 满量程的 raw counts: Mono12 上 100-200 灰度 = 1606-3212 counts。不折算就会
    # 把「满量程的 2.4%-4.9%」当成目标, 挑出比预期暗 16 倍的那一档。
    scale_to_counts = float(full_scale) / 255.0
    band_lo = float(RECOMMENDED_PEAK_RANGE[0]) * scale_to_counts
    band_hi = float(RECOMMENDED_PEAK_RANGE[1]) * scale_to_counts
    band_target = float(RECOMMENDED_PEAK_TARGET) * scale_to_counts
    for record in ladder:
        record["carries_signal"] = bool(record["peak_raw"] > noise_floor["threshold_counts"])
        record["saturated"] = bool(record["peak_raw"] >= saturation_counts)
        record["usable"] = bool(record["carries_signal"] and not record["saturated"])

    usable = [r for r in ladder if r["usable"]]
    recommended: dict[str, Any] | None = None
    if usable:
        in_band = [r for r in usable if band_lo <= r["peak_raw"] <= band_hi]
        pool = in_band or usable
        pick = min(
            pool,
            key=lambda r: abs(float(r["peak_raw"]) - band_target),
        )
        recommended = {
            "exposure_ms": float(pick["effective_ms"]),
            # peak_counts 保持**原始 counts** (图上的纵轴就是 raw axis);
            # peak_gray 是同一读数折回 0-255 灰度, 便于跨位深对照历史记录。
            "peak_counts": float(pick["peak_raw"]),
            "peak_gray": float(pick["peak_raw"]) * 255.0 / float(full_scale),
            "in_target_band": bool(in_band),
            "target_band_counts": [band_lo, band_hi],
            "target_band_gray": [
                float(RECOMMENDED_PEAK_RANGE[0]),
                float(RECOMMENDED_PEAK_RANGE[1]),
            ],
        }

    # 阶梯的单调性: 改曝光后第一帧就是新曝光是大恒驱动层的不变量, 阶梯不单调
    # 说明有别的因素 (错档 / 增益漂移 / 曝光没真正生效)。
    peaks = [float(r["peak_raw"]) for r in sorted(ladder, key=lambda r: r["effective_ms"])]
    monotonic = all(b >= a for a, b in zip(peaks, peaks[1:], strict=False))
    linearity_r2 = float("nan")
    if len(peaks) >= 3:
        xs = np.array([r["effective_ms"] for r in sorted(ladder, key=lambda r: r["effective_ms"])])
        ys = np.array(peaks)
        mask = ys > noise_floor["threshold_counts"]
        if int(mask.sum()) >= 3:
            design = np.stack([np.ones(int(mask.sum())), xs[mask]], axis=1)
            sol, *_ = np.linalg.lstsq(design, ys[mask], rcond=None)
            resid = design @ sol - ys[mask]
            ss_tot = float(np.sum((ys[mask] - ys[mask].mean()) ** 2))
            ss_res = float(np.sum(resid**2))
            linearity_r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    # 光通量 (counts/ms): 口径 (只拟合未饱和的可用档) 见 :func:`ladder_flux`。
    flux = ladder_flux(ladder)

    return {
        "noise_floor": noise_floor,
        "saturation_counts": float(saturation_counts),
        # 目标带的两个视角: raw counts (随满量程缩放) 与 0-255 灰度 (常量)。
        "target_band_counts": [band_lo, band_hi],
        "target_band_gray": [
            float(RECOMMENDED_PEAK_RANGE[0]),
            float(RECOMMENDED_PEAK_RANGE[1]),
        ],
        "usable_window_ms": [float(r["effective_ms"]) for r in usable],
        "max_ladder_peak_counts": float(max((r["peak_raw"] for r in ladder), default=0.0)),
        "best_usable_peak_counts": float(max((r["peak_raw"] for r in usable), default=0.0)),
        "recommended_exposure": recommended,
        "ladder_monotonic": bool(monotonic),
        "ladder_linearity_r2": linearity_r2,
        "counts_per_ms": flux["counts_per_ms"],
        "counts_per_ms_window_ms": flux["window_ms"],
    }


def _state_summary(state: dict[str, Any]) -> dict[str, Any]:
    """一个离焦状态的读数汇总 (逐帧样本的均值/散布)。"""
    peaks = [float(v) for v in state.get("peak_raw_per_frame", [])] or [
        float(state["peak_raw"])
    ]
    widths = [float(v) for v in state.get("fwhm_per_frame", [])] or [
        float(state["fwhm_px"])
    ]
    return {
        "defocus_rad": float(state["defocus_rad"]),
        "n": len(peaks),
        "peak_mean": float(np.mean(peaks)),
        "peak_std": float(np.std(peaks, ddof=1)) if len(peaks) > 1 else 0.0,
        "fwhm_mean": float(np.mean(widths)),
        "fwhm_std": float(np.std(widths, ddof=1)) if len(widths) > 1 else 0.0,
    }


def analyse_defocus_response(
    states: list[dict[str, Any]],
    noise_threshold_counts: float,
) -> dict[str, Any]:
    """从离焦扫描里量「离焦有没有真的起作用」。

    两条判据, 互相独立:

    1. **统计判据** —— 某个幅度上的读数与平场的差, 超过**合并标准误**的
       ``RESPONSE_SIGMAS`` 倍 (``SEM = sigma * sqrt(1/N_flat + 1/N_state)``,
       比较的是均值, 所以要用标准误而不是单帧标准差)。**每个非零幅度都要看**
       —— 响应可以是不对称的 (实验系统自身的离焦偏置会让 +A 与 −A 的效果差很多),
       只看其中一个会漏掉真正的响应。
    2. **物理下限判据** —— 相对变化达到 :data:`WIDTH_RELATIVE_FLOOR` /
       :data:`PEAK_RELATIVE_FLOOR`。离焦 ±0.5 rad 在物理上必然让光斑宽度变化
       ≥20%、峰值变化 ≥25%; 达不到这个量级就是「没真正加上去」。这条**只在测得
       准的时候**才允许: 噪声也能造出很大的相对变化。

    同时算出**分辨率**: ``3 * sigma / 基线值`` —— 「能分辨多小的相对变化」。
    分辨率比物理下限还差 ⇒ 即使面板真的在调制我们也看不出来 ⇒ 这是
    **信噪比不足** (``low_snr``), 不是「面板没调制」 (``no_response``)。

    噪声尺度取自**平场**的那几组: 组内逐帧散布与**组间差** (平场复测) 取大者。
    组间差必须算进来 —— 它量的是「换一次显示再测一遍」能差多少, 而几何 sweep
    恰恰是在换显示状态下比较的, 只用组内散布会低估噪声。
    """
    summaries = [_state_summary(s) for s in states]
    flat = [s for s in summaries if abs(s["defocus_rad"]) < FLAT_DEFOCUS_EPS_RAD]
    non_flat = [s for s in summaries if abs(s["defocus_rad"]) >= FLAT_DEFOCUS_EPS_RAD]
    if not flat:
        flat = [summaries[0]]

    def _sigma(groups: list[dict[str, Any]], mean_key: str, std_key: str, floor: float) -> float:
        within = [
            float(g[std_key]) for g in groups if int(g["n"]) > 1 and g[std_key] > 0.0
        ]
        pooled = float(np.sqrt(np.mean(np.square(within)))) if within else 0.0
        between = 0.0
        if len(groups) >= 2:
            means = [float(g[mean_key]) for g in groups]
            between = (max(means) - min(means)) / math.sqrt(2.0)
        return max(pooled, between, floor)

    sigma_peak = _sigma(flat, "peak_mean", "peak_std", PEAK_SIGMA_FLOOR_COUNTS)
    sigma_fwhm = _sigma(flat, "fwhm_mean", "fwhm_std", FWHM_SIGMA_FLOOR_PX)

    base_peak = float(np.mean([g["peak_mean"] for g in flat]))
    base_fwhm = float(np.mean([g["fwhm_mean"] for g in flat]))
    n_flat = max(sum(int(g["n"]) for g in flat), 1)

    peak_signal = 0.0
    width_signal = 0.0
    peak_signal_at = 0.0
    width_signal_at = 0.0
    peak_relative = 0.0
    width_relative = 0.0
    peak_stat = False
    width_stat = False
    per_amp: list[dict[str, Any]] = []
    for state, summary in zip(states, summaries, strict=True):
        peak = float(summary["peak_mean"])
        fwhm = float(summary["fwhm_mean"])
        n_state = max(int(summary["n"]), 1)
        # 该状态自己的噪声 (暗状态的光子噪声更大), 与平场的噪声取大者。
        sigma_peak_state = max(sigma_peak, float(summary["peak_std"]))
        sigma_fwhm_state = max(sigma_fwhm, float(summary["fwhm_std"]))
        sem_peak = sigma_peak_state * math.sqrt(1.0 / n_flat + 1.0 / n_state)
        sem_fwhm = sigma_fwhm_state * math.sqrt(1.0 / n_flat + 1.0 / n_state)
        d_peak = peak - base_peak
        d_fwhm = fwhm - base_fwhm
        is_flat = abs(summary["defocus_rad"]) < FLAT_DEFOCUS_EPS_RAD
        if not is_flat:
            if abs(d_peak) >= peak_signal:
                peak_signal = abs(d_peak)
                peak_signal_at = summary["defocus_rad"]
            if abs(d_fwhm) >= width_signal:
                width_signal = abs(d_fwhm)
                width_signal_at = summary["defocus_rad"]
            if base_peak > 0:
                peak_relative = max(peak_relative, abs(d_peak) / base_peak)
            if base_fwhm > 0:
                width_relative = max(width_relative, abs(d_fwhm) / base_fwhm)
            peak_stat = peak_stat or abs(d_peak) >= RESPONSE_SIGMAS * sem_peak
            width_stat = width_stat or abs(d_fwhm) >= RESPONSE_SIGMAS * sem_fwhm
        per_amp.append(
            {
                "defocus_rad": summary["defocus_rad"],
                "peak_raw": peak,
                "peak_std": summary["peak_std"],
                "fwhm_px": fwhm,
                "fwhm_std": summary["fwhm_std"],
                "second_moment_radius_px": float(state["second_moment_radius_px"]),
                "centroid_x": float(state["centroid_x"]),
                "centroid_y": float(state["centroid_y"]),
                "hollowness": float(state["hollowness"]),
                "delta_peak_counts": d_peak,
                "delta_fwhm_px": d_fwhm,
                "sem_peak_counts": float(sem_peak),
                "sem_fwhm_px": float(sem_fwhm),
                "carries_signal": bool(peak > noise_threshold_counts),
                "visible": bool(peak >= DEFAULT_MIN_PEAK),
                "peak_statistically_significant": (
                    None if is_flat else bool(abs(d_peak) >= RESPONSE_SIGMAS * sem_peak)
                ),
                "fwhm_statistically_significant": (
                    None if is_flat else bool(abs(d_fwhm) >= RESPONSE_SIGMAS * sem_fwhm)
                ),
            }
        )

    width_resolution = (
        float(RESPONSE_SIGMAS * sigma_fwhm / base_fwhm) if base_fwhm > 0 else float("inf")
    )
    peak_resolution = (
        float(RESPONSE_SIGMAS * sigma_peak / base_peak) if base_peak > 0 else float("inf")
    )
    precision_ok = bool(
        width_resolution <= WIDTH_RESOLUTION_MAX and peak_resolution <= PEAK_RESOLUTION_MAX
    )
    # 相对变化这条兜底判据**只在测得准的时候**才允许: 噪声也能造出很大的相对
    # 变化 (暗帧里宽度读数乱跳), 不加这道门会把噪声当成响应。
    responds_relative = bool(
        precision_ok
        and (width_relative >= WIDTH_RELATIVE_FLOOR or peak_relative >= PEAK_RELATIVE_FLOOR)
    )
    return {
        "baseline_peak_counts": base_peak,
        "baseline_fwhm_px": base_fwhm,
        "flat_group_peak_means": [float(g["peak_mean"]) for g in flat],
        "flat_group_fwhm_means": [float(g["fwhm_mean"]) for g in flat],
        "sigma_peak_counts": sigma_peak,
        "sigma_fwhm_px": sigma_fwhm,
        "peak_snr": float(base_peak / sigma_peak) if sigma_peak > 0 else float("inf"),
        "width_snr": float(base_fwhm / sigma_fwhm) if sigma_fwhm > 0 else float("inf"),
        "peak_signal_counts": float(peak_signal),
        "width_signal_px": float(width_signal),
        "peak_signal_at_rad": float(peak_signal_at),
        "width_signal_at_rad": float(width_signal_at),
        "peak_relative_change": float(peak_relative),
        "width_relative_change": float(width_relative),
        "peak_resolution": peak_resolution,
        "width_resolution": width_resolution,
        "precision_ok": precision_ok,
        "peak_statistically_significant": peak_stat,
        "width_statistically_significant": width_stat,
        "peak_significant": bool(peak_stat or (precision_ok and peak_relative >= PEAK_RELATIVE_FLOOR)),
        "width_significant": bool(width_stat or (precision_ok and width_relative >= WIDTH_RELATIVE_FLOOR)),
        "responds": bool(peak_stat or width_stat or responds_relative),
        "per_amp": per_amp,
    }


def _age_days(stamp: Any) -> float | None:
    """把记录里的 ISO 时间戳换成「多少天前」; 解析不了就返回 ``None``。"""
    if not isinstance(stamp, str) or not stamp:
        return None
    try:
        then = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError:
        return None
    if then.tzinfo is None:
        then = then.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - then).total_seconds() / 86400.0


def _load_reference_baseline(
    path: Path,
    *,
    device: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """读落盘的重定基基线; 文件缺失/数值不可用/设备序列号不符时返回 ``None``。

    序列号不符就不沿用: 换了 SLM 或相机之后「那一刻的正常值」不再代表现在。
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    value = payload.get("counts_per_ms")
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) <= 0:
        return None
    record_device = payload.get("device") or {}
    for key in ("slm_serial", "cam_serial"):
        wanted = (device or {}).get(key)
        if wanted and record_device.get(key) and record_device.get(key) != wanted:
            return None
    return {
        "source": "rebaseline",
        "counts_per_ms": float(value),
        "n": 1,
        "min_counts_per_ms": float(value),
        "max_counts_per_ms": float(value),
        "full_scale": payload.get("full_scale"),
        "runs": [str(path)],
        "recorded_utc": payload.get("recorded_utc"),
        "age_days": _age_days(payload.get("recorded_utc")),
        "skipped": {},
    }


def _save_reference_baseline(
    path: Path,
    *,
    counts_per_ms: float,
    device: dict[str, Any] | None,
    full_scale: float | None,
    session: str = "",
    note: str = "",
) -> dict[str, Any]:
    """把「这一刻的光通量」写成新的参考基线 (下次探测起生效)。"""
    payload = {
        "counts_per_ms": float(counts_per_ms),
        "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "device": dict(device or {}),
        "full_scale": full_scale,
        "session": session,
        "note": note,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def bench_reference(
    out_root: Path,
    *,
    device: dict[str, Any] | None = None,
    full_scale: float | None = None,
) -> dict[str, Any] | None:
    """本实验系统历史 ``ok`` 探测的光通量中位数 (counts/ms)。

    为什么口径只取 ``ok``: 一次已经判 ``low_snr`` 的运行不是「正常工作时的样子」,
    拿它进中位数会把标准越拉越低, 于是同一条坏光路连着跑几次就把自己判成正常了。

    取值的优先级: 落盘的重定基基线 (``reference_baseline.json``, 见
    :data:`REFERENCE_BASELINE_FILE`) → 历史 ``ok`` 记录的中位数。

    设备匹配: ``slm_serial`` / ``cam_serial`` 读得到时按序列号匹配 (换 SLM 或换相机
    后旧参考不再适用)。位深只在调用方给出 ``full_scale`` 时强制一致 —— 探测在
    采集之前就要拿参考, 那时位深还没测出来, 所以默认不按位深过滤 (这条以前写在
    docstring 里但代码没做, 2026-10-29 更正)。

    样本少于 :data:`MIN_REFERENCE_SAMPLES` 时返回 ``None`` —— 中位数在 1-2 个样本上
    没有意义, 宁可不下判定。返回的 ``skipped`` 记账说明其余记录为什么没进样本池,
    ``age_days`` 给出样本池里最新一条有多旧。
    """
    baseline = _load_reference_baseline(out_root / REFERENCE_BASELINE_FILE, device=device)
    if baseline is not None:
        return baseline

    samples: list[float] = []
    scales: list[float] = []
    used: list[str] = []
    skipped = {
        "unreadable": 0,
        "not_ok": 0,
        "serial_mismatch": 0,
        "scale_mismatch": 0,
        "no_flux": 0,
    }
    newest_utc: str | None = None
    for path in sorted(out_root.rglob("optical_path_probe.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):  # 读到半截文件不应让门禁自己崩掉
            skipped["unreadable"] += 1
            continue
        verdict = record.get("verdict") or {}
        if verdict.get("state") != STATE_OK:
            skipped["not_ok"] += 1
            continue
        ladder_summary = record.get("ladder_summary") or {}
        value = ladder_summary.get("counts_per_ms")
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            # 2026-10-29 之前落盘的记录没有这一项: 从它自己的阶梯按同一口径重算。
            value = ladder_flux(record.get("exposure_ladder") or [])["counts_per_ms"]
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            skipped["no_flux"] += 1
            continue
        ladder = record.get("exposure_ladder") or []
        scale = float(ladder[0].get("full_scale") or 0.0) if ladder else 0.0
        if full_scale and scale and abs(scale - float(full_scale)) > 1e-9:
            skipped["scale_mismatch"] += 1
            continue
        record_device = record.get("device") or {}
        for key in ("slm_serial", "cam_serial"):
            wanted = (device or {}).get(key)
            if wanted and record_device.get(key) and record_device.get(key) != wanted:
                skipped["serial_mismatch"] += 1
                break
        else:
            samples.append(float(value))
            scales.append(scale)
            used.append(str(path.parent))
            stamp = record.get("created_utc")
            if isinstance(stamp, str) and (newest_utc is None or stamp > newest_utc):
                newest_utc = stamp
    if len(samples) < MIN_REFERENCE_SAMPLES:
        return None
    return {
        "source": "history",
        "counts_per_ms": float(np.median(samples)),
        "n": len(samples),
        "min_counts_per_ms": float(min(samples)),
        "max_counts_per_ms": float(max(samples)),
        "full_scale": max(scales) if scales else None,
        "runs": used[-5:],
        "recorded_utc": newest_utc,
        "age_days": _age_days(newest_utc),
        "skipped": skipped,
    }


def plot_results(
    ladder: list[dict[str, Any]],
    ladder_summary: dict[str, Any],
    response: dict[str, Any],
    out_path: Path,
) -> None:
    """曝光-峰值曲线 + 离焦响应曲线 (峰值/宽度 vs 离焦幅度)。"""
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    ax = axes[0]
    xs = [float(r["effective_ms"]) for r in ladder]
    ys = [float(r["peak_raw"]) for r in ladder]
    ax.plot(xs, ys, "o-", color="#c2410c", label="峰值 (原始帧)")
    threshold = float(ladder_summary["noise_floor"]["threshold_counts"])
    ax.axhline(threshold, ls="--", color="#64748b", label=f"噪声底 {threshold:.1f} counts")
    # 饱和线随生效满量程走 (Mono12 上 0.95*4095 ≈ 3891 counts, 不是 8-bit 的 242)。
    saturation_counts = float(ladder_summary["saturation_counts"])
    ax.axhline(
        saturation_counts,
        ls="--",
        color="#dc2626",
        label=f"饱和线 {saturation_counts:.0f} counts",
    )
    usable = [
        (float(r["effective_ms"]), float(r["peak_raw"]))
        for r in ladder
        if r.get("usable")
    ]
    if usable:
        ax.fill_between(
            [x for x, _ in usable],
            0,
            [y for _, y in usable],
            color="#16a34a",
            alpha=0.12,
            label="可用窗口",
        )
    recommended = ladder_summary.get("recommended_exposure")
    if recommended is not None:
        ax.plot(
            [recommended["exposure_ms"]],
            [recommended["peak_counts"]],
            marker="*",
            ms=16,
            color="#16a34a",
            label=f"建议曝光 {recommended['exposure_ms']:g} ms",
        )
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xlabel("曝光 (ms)")
    ax.set_ylabel("峰值 (counts)")
    ax.set_title("曝光阶梯: 峰值 vs 曝光")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    per_amp = sorted(
        (p for p in response["per_amp"] if p["defocus_rad"] != 0.0),
        key=lambda p: float(p["defocus_rad"]),
    )
    amps = [float(p["defocus_rad"]) for p in per_amp]
    widths = [float(p["fwhm_px"]) for p in per_amp]
    peaks = [float(p["peak_raw"]) for p in per_amp]
    ax.plot(amps, widths, "s-", color="#1d4ed8", label="光斑 FWHM (px)")
    ax.axhline(
        float(response["baseline_fwhm_px"]),
        ls="--",
        color="#64748b",
        label=f"平场 FWHM {response['baseline_fwhm_px']:.1f} px",
    )
    ax.set_xlabel("离焦幅度 (rad, Noll 4)")
    ax.set_ylabel("FWHM (px)", color="#1d4ed8")
    ax.tick_params(axis="y", labelcolor="#1d4ed8")
    ax.set_title("离焦响应: 宽度/峰值 vs 离焦幅度")

    twin = ax.twinx()
    twin.plot(amps, peaks, "o-", color="#b45309", label="峰值 (counts)")
    twin.set_ylabel("峰值 (counts)", color="#b45309")
    twin.tick_params(axis="y", labelcolor="#b45309")

    lines = ax.get_lines() + twin.get_lines()
    labels = [line.get_label() for line in lines]
    ax.legend(lines, labels, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def load_sweep_module() -> Any:
    """按路径加载采集驱动 (``scripts/slm/run_hw_sweep_groups.py``)。

    ⚠️ **设备互斥已经不再经过它** —— 判别 / 探针 / 等待都直接用权威模块
    :mod:`ao_shaping.tools.kernels.device_mutex` (同一份实现, 不再按路径加载一份
    可能已经过期的副本)。这个函数留着只是因为别处的调试代码会按名字找它。

    **必须先注册进 ``sys.modules`` 再执行**: ``dataclasses`` 通过
    ``sys.modules[cls.__module__].__dict__`` 解析 ``KW_ONLY`` /
    ``_HAS_DEFAULT_FACTORY``, 按路径加载而不注册的话, 里面每个 dataclass 都会炸
    ``'NoneType' object has no attribute '__dict__'``。
    """
    spec = importlib.util.spec_from_file_location("run_hw_sweep_groups", SWEEP_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load the device probe from {SWEEP_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def acquire_devices(args: argparse.Namespace) -> DeviceLock:
    """开跑前拿到设备, 并把**设备锁拿在手里**。

    为什么必须持锁到采集结束 (不是等到它空闲就放掉): SLM 的 USB 句柄是独占的,
    谁先打开谁拥有; SLM 与大恒相机的**互斥锁**也一样 —— 只等到「空闲」就放锁,
    另一个会话会在我们采集的中途拿走句柄 (表现为 -10003, 或更糟的 -10032
    「USB其他错误」出现在面板扫描中途)。

    三种情况在此分开 (由权威模块判定, 不再由本脚本猜):

    * **别人持有** ⇒ 按 ``--wait-until-idle`` 的预算等, 并报出持有者的 pid 与命令行;
    * **自己这一棵进程树持有** ⇒ 不是冲突, 直接放行 (旧判据在这里把「自己启动的
      子进程」当成「别的会话」⇒ 永远抢不到设备);
    * **锁空 + SDK 报 -10003** ⇒ **SDK 陈旧状态**: 排队等不来, 提示给 SLM 控制器
      物理断电重置。

    Args:
        args: 命令行参数。

    Returns:
        已经拿在手里的设备锁 (调用方在 ``finally`` 里 ``release``)。

    Raises:
        SystemExit: 设备等不来 (没插上 / 光路不通 / 陈旧句柄 / 探针自己没装好)。
    """
    budget = float(args.wait_until_idle)
    verdict = device_mutex.wait_until_idle(
        budget,
        slm_number=int(args.slm_number),
        cam_id=int(args.cam_id),
        cam_type=str(args.cam_type),
        exposure_ms=float(args.exposures[0]),
        own_paths=[Path(__file__).resolve()],
    )
    if verdict.ok:
        lock = verdict.lock if verdict.lock is not None else DeviceLock.default()
        if not lock.is_locked():
            lock.acquire(timeout=0.0)
        return lock
    if verdict.status == "stale":
        raise SystemExit(
            f"设备不可用 ({verdict.kind}): {verdict.headline()} —— {verdict.remedy}"
        )
    if verdict.status == "broken":
        raise SystemExit(f"设备探针没有给出可用结论: {verdict.detail}")
    if verdict.status == "timeout":
        raise SystemExit(
            f"设备被占: {verdict.reason} (加/放大 --wait-until-idle 可以等; "
            f"当前预算 {budget:.0f}s)"
        )
    raise SystemExit(
        f"设备不可用 ({verdict.kind or 'unknown'}): {verdict.headline()} —— "
        f"{verdict.remedy or verdict.detail}"
        + ("" if verdict.kind in WAITABLE_KINDS else " (加 --wait-until-idle 也等不来)")
    )


@dataclass
class ProbeConfig:
    """一次探测的配置 (落进 JSON, 便于复现)。"""

    exposures_ms: list[float] = field(default_factory=list)
    defocus_amps_rad: list[float] = field(default_factory=list)
    frames_per_exposure: int = 2
    frames_per_defocus: int = 5
    defocus_radius_px: int = BEAM_RADIUS_PANEL
    pupil_center: tuple[int, int] = BEAM_CENTER_PANEL
    defocus_exposure_ms: float | None = None
    settle_s: float = 0.5
    slm_number: int = 1
    slm_wavelength_nm: int = 1064
    cam_type: str = "daheng"
    cam_id: int = 0
    #: 请求的相机位深 (bit); 开相机后按设备回读校验, 不符即报错。
    bit_depth: int = S0_BIT_DEPTH
    wait_until_idle_s: float = 0.0
    #: Fresnel D 标定段 (默认关闭: 老会话行为逐位不变)。
    focus_calibration: bool = False
    focus_coarse_points: int = FOCUS_COARSE_POINTS
    focus_diopter_range: float = FOCUS_DIOPTER_RANGE
    focus_refine_points: int = FOCUS_REFINE_POINTS
    focus_frames: int = FOCUS_FRAMES
    focus_floor_ms: float | None = None
    focus_ceiling_ms: float | None = None
    #: 曝光策略: ``adapt`` = 逐点自适应 (老行为); ``fixed`` = 整段扫描单曝光,
    #: 排序与守门见 ``focus_probe.sweep_focus``。
    focus_exposure_mode: str = FOCUS_EXPOSURE_MODE_ADAPT
    #: 单曝光模式下预注册的唯一曝光 (ms); 覆盖 S0 推荐曝光, 越界仍按
    #: ``focus_floor_ms`` / ``focus_ceiling_ms`` 夹取。``None`` = 用推荐曝光起步。
    focus_fixed_exposure_ms: float | None = None
    #: sweep_focus 的 centre_px = Fresnel 透镜的 SLM 面板光瞳中心 (panel px);
    #: 缺省 = 本探针自己的 pupil_center, 与前面段的离焦相位同一口径。
    focus_centre_px: tuple[float, float] | None = None
    focus_radius_px: float | None = None


def _parse_float_list(text: str, label: str) -> list[float]:
    values: list[float] = []
    for chunk in str(text).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            values.append(float(chunk))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"{label} 里 {chunk!r} 不是数字") from exc
    if not values:
        raise argparse.ArgumentTypeError(f"{label} 不能是空列表")
    return values


def _parse_float_pair(text: str, label: str) -> tuple[float, float]:
    """把 ``"x,y"`` 解析成浮点坐标对 (零级中心允许亚像素)。"""
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"{label} 必须是 'x,y', 收到 {text!r}")
    try:
        return float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{label} 里 {text!r} 不是数字对") from exc


def _parse_center(text: str) -> tuple[int, int]:
    """把 ``"x,y"`` 解析成面板坐标。"""
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"--pupil-center 必须是 'x,y', 收到 {text!r}")
    try:
        return (int(float(parts[0])), int(float(parts[1])))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"--pupil-center 不是两个数字: {text!r}") from exc


def build_parser() -> argparse.ArgumentParser:
    """命令行参数。"""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--exposures",
        default="0.05,0.2,1,2,5,10,20,50",
        help="曝光阶梯 (ms, 逗号分隔)",
    )
    parser.add_argument(
        "--defocus-amps",
        default="0,0.5,1.0,-1.0",
        help="离焦幅度 (rad, Noll 4, 逗号分隔; 0 = 平场)",
    )
    parser.add_argument("--slm-number", type=int, default=1, help="SLM 编号")
    parser.add_argument("--slm-wavelength", type=int, default=1064, help="SLM 工作波长 (nm)")
    parser.add_argument("--cam-type", default="daheng", help="相机类型")
    parser.add_argument("--cam-id", type=int, default=0, help="相机编号")
    parser.add_argument(
        "--bit-depth",
        type=int,
        default=S0_BIT_DEPTH,
        choices=(8, 12, 16),
        help="相机输出位深 (bit); S0 前置条件是 Mono12 —— 8 bit 下 1 count ≈ 7%% "
             "满量程, 均匀度/线性度结论会被量化噪声主导。只对登记了 "
             "supported_bit_depths 的后端 (daheng/miicam) 生效: 开相机后按设备"
             "回读校验, 不符即报错 (其它后端没有位深可切, 按帧 dtype 兜底满量程)",
    )
    parser.add_argument(
        "--focus-calibration",
        action="store_true",
        help="跑 Fresnel 离焦补偿 D 标定 (默认关); 结果写进 S0 记录的 "
        "fresnel_compensation_m_inv 与 focus_calibration.json",
    )
    parser.add_argument(
        "--focus-coarse-points",
        type=int,
        default=FOCUS_COARSE_POINTS,
        help="Fresnel D 标定的粗扫点数 (默认 %(default)s)",
    )
    parser.add_argument(
        "--focus-diopter-range",
        type=float,
        default=FOCUS_DIOPTER_RANGE,
        help="Fresnel D 标定的屈光度扫描半径 (m⁻¹, 扫 ±该值; 默认 %(default)s)",
    )
    parser.add_argument(
        "--focus-refine-points",
        type=int,
        default=FOCUS_REFINE_POINTS,
        help="Fresnel D 标定极值点附近的细扫点数 (默认 %(default)s)",
    )
    parser.add_argument(
        "--focus-frames",
        type=int,
        default=FOCUS_FRAMES,
        help="Fresnel D 标定每个扫描点平均的帧数 (默认 %(default)s)",
    )
    parser.add_argument(
        "--focus-exposure-mode",
        choices=("adapt", "fixed"),
        default="adapt",
        help="Fresnel D 标定的曝光策略: adapt = 逐点自适应曝光 (老行为); "
        "fixed = 整段扫描用同一个曝光 (峰值含加性本底, 跨曝光 peak/ms 归一化"
        "不可用, 单曝光下改判同曝光复测)",
    )
    parser.add_argument(
        "--focus-fixed-exposure-ms",
        type=float,
        default=None,
        help="单曝光模式 (--focus-exposure-mode fixed) 的唯一曝光 (ms); "
        "覆盖 S0 推荐曝光, 越界按 --focus-floor-ms/--focus-ceiling-ms 夹取",
    )
    parser.add_argument(
        "--focus-floor-ms",
        type=float,
        default=None,
        help="Fresnel D 标定的曝光下限 (ms); 缺省 = focus_probe.FLOOR_MS",
    )
    parser.add_argument(
        "--focus-ceiling-ms",
        type=float,
        default=None,
        help="Fresnel D 标定的曝光上限 (ms); 缺省 = --exposures 的最大值",
    )
    parser.add_argument(
        "--focus-centre-px",
        type=_parse_focus_centre,
        default=None,
        help="Fresnel 透镜在 SLM 面板上的光瞳中心 'x,y' (panel px); "
        "缺省 = 本次 --pupil-center, 再缺省 = BEAM_CENTER_PANEL",
    )
    parser.add_argument(
        "--focus-radius-px",
        type=float,
        default=None,
        help=(
            "Fresnel 透镜的面板照明半径 (px); 缺省 = BEAM_RADIUS_PANEL "
            f"({FOCUS_BEAM_RADIUS_PANEL:g})"
        ),
    )
    parser.add_argument(
        "--wait-until-idle",
        type=float,
        default=0.0,
        help="设备被占时的等待预算 (s); 0 = 不等, 立刻报错",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ROOT / "data" / "debug" / "optical_path_probe",
        help="证据输出根目录",
    )
    parser.add_argument(
        "--session",
        default="",
        help="会话名: 证据落到 <out-root>/<session>/<UTC戳>/, 便于把一次探测对到它门禁的那个 lane; "
             "缺省 = 直接落 <out-root>/<UTC戳>/",
    )
    parser.add_argument(
        "--min-throughput-ratio",
        type=float,
        default=DEFAULT_MIN_THROUGHPUT_RATIO,
        help="光通量下限: 峰值 counts/ms 低于参考值的这个比例即判 weak_path "
             f"(默认 {DEFAULT_MIN_THROUGHPUT_RATIO:g}; 设 0 = 关掉比较)",
    )
    parser.add_argument(
        "--reference-counts-per-ms",
        type=float,
        default=None,
        help="显式给出光通量参考 (counts/ms); 缺省 = 取 --out-root 下历史 ok 探测的中位数",
    )
    parser.add_argument(
        "--no-reference",
        action="store_true",
        help="不做光通量比较 (回到 ok / low_snr / no_response 三态)",
    )
    parser.add_argument(
        "--rebaseline",
        action="store_true",
        help="把本次实测的光通量写成新的参考基线 (<out-root>/reference_baseline.json), "
             "下次探测起生效; 本次自身不判 weak_path (它就是标准本身)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=2,
        help="曝光阶梯每档抓几帧",
    )
    parser.add_argument(
        "--defocus-frames",
        type=int,
        default=5,
        help="离焦扫描每个状态抓几帧 (平场 + 平场复测 ⇒ 噪声尺度按组估计)",
    )
    parser.add_argument(
        "--defocus-radius",
        type=int,
        default=BEAM_RADIUS_PANEL,
        help="离焦 Zernike 的孔径半径 (面板 px)",
    )
    parser.add_argument(
        "--pupil-center",
        default=f"{BEAM_CENTER_PANEL[0]},{BEAM_CENTER_PANEL[1]}",
        help="瞳面中心 (面板 px, x,y) —— 离焦盘以它为准, 偏心会带上倾斜与像散",
    )
    parser.add_argument(
        "--defocus-exposure-ms",
        type=float,
        default=None,
        help="离焦扫描用的曝光 (ms); 缺省 = 自动取建议曝光",
    )
    parser.add_argument(
        "--settle-s",
        type=float,
        default=0.5,
        help="每次改面板后的稳定等待 (s)",
    )
    parser.add_argument("--no-plot", action="store_true", help="不画图")
    return parser


def main(argv: list[str] | None = None) -> int:
    """跑一次探测, 返回退出码 (0 = ok, 2 = low_snr / no_response)。"""
    args = build_parser().parse_args(argv)
    global _PENDING_RECORD
    _PENDING_RECORD = None
    # 🔴 设备锁**整段探测期间**都握在手里 (不是等到空闲就放掉): 只放到「空闲」
    # 就撒手, 另一个会话会在采集的中途拿走句柄。
    lock = acquire_devices(args)
    try:
        return run_probe(args)
    finally:
        lock.release()
        # 放锁之后才写记录: occupancy.released_at 才是真的释放时刻。
        _write_pending_record()


def _utc_iso() -> str:
    """当前 UTC 时刻 (秒级 ISO 8601)。"""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _git_short_sha() -> str | None:
    """当前检出的短 sha; 不是 git 仓库 / 没有 git 时返回 ``None``。"""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    sha = proc.stdout.strip()
    return sha or None


def _rel_to_root(path: Path) -> str:
    """仓库内的路径写成相对形式 (记录要能跨机器读懂); 仓库外原样返回。"""
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _collect_record_devices(
    slm: Any,
    camera: Any,
    config: ProbeConfig,
    *,
    full_scale: float | None,
    exposure_used_ms: float,
    device_info: dict[str, Any],
    reasons: dict[str, str],
) -> dict[str, Any]:
    """标准记录 (SKILL §4) 的 ``devices`` 段; 拿不到的字段在 ``reasons`` 里说明原因。"""
    try:
        info: dict[str, Any] = dict(slm.get_device_info() or {})
    except Exception:  # 设备信息拿不到不该让记录写不出来
        info = {}

    drive_board_id = info.get("driveboard_id")
    if not drive_board_id:
        reasons["devices.slm.drive_board_id"] = (
            "驱动板 ID 读不到 (驱动板未应答时身份回退到选板 ID, 见 Santec.get_device_info)"
        )
    option_board_id = info.get("optionboard_id")
    if option_board_id is None and info.get("serial_source") == "optionboard_fallback":
        # 身份回退时 identity_id 就是选板 ID (静默回落的唯一线索)。
        option_board_id = info.get("identity_id")

    # ReadSU 状态码: 驱动只把「驱动板应答」暴露成 ok 布尔, 没有原始返回码,
    # 所以这里记 0 = 应答、null = 不可读。
    try:
        status = slm.drive_board_status()
    except Exception:  # 只读健康检查, 拿不到就不记状态码
        status = None
    readsu_status_at_open = 0 if status is not None and bool(status.ok) else None
    if readsu_status_at_open is None:
        reasons["devices.slm.readsu_status_at_open"] = (
            "驱动板 ReadSU 没有可用状态码 (drive_board_status 不可读或驱动板未应答)"
        )

    # 物理 2π 灰度 = Santec 相位表的设备读数 (诊断/上报值, 不参与 rad→gray 编码)。
    # 这里**绝不**回落到配置的编码标度 (缺省 1023) 冒充设备值: 读不到就写 null
    # 并在 reasons 里说明, 由 build_optical_path_record 登记到 missing。
    two_pi_grayscale: int | None = None
    try:
        _wavelength_readback, two_pi_readback = slm.get_wavelength_info()
        candidate = int(two_pi_readback)
        if candidate > 0:
            two_pi_grayscale = candidate
        else:
            reasons["devices.slm.two_pi_grayscale"] = (
                f"SLM 设备回报的 2π 灰度无效 ({candidate}) —— 相位表可能未编程"
            )
    except Exception as exc:  # 只读字段, 失败不该阻断记录落盘
        reasons["devices.slm.two_pi_grayscale"] = (
            f"SLM 设备 2π 灰度读不到 ({type(exc).__name__}: {exc})"
        )

    pixel_size_um = getattr(camera, "pixel_size_um", None)
    if not isinstance(pixel_size_um, (int, float)) or float(pixel_size_um) <= 0:
        pixel_size_um = None
        reasons["devices.camera.pixel_size_um"] = "该相机后端未登记像元尺寸"
    reasons["devices.camera.averaging_frames"] = (
        "探针不做帧平均: 每档取 N 帧峰值最大值 (N = --frames / --defocus-frames)"
    )
    reasons["devices.camera.discard_frames_after_exposure_change"] = (
        "探针改曝光后不丢帧 (曝光阶梯的第一帧即新曝光)"
    )

    return {
        "slm": {
            "model": "Santec SLM-200",
            "drive_board_id": drive_board_id,
            "option_board_id": option_board_id,
            "slm_number": int(config.slm_number),
            "wavelength_nm": int(config.slm_wavelength_nm),
            "two_pi_grayscale": two_pi_grayscale,
            "panel_res": [int(SLM_PANEL_W), int(SLM_PANEL_H)],
            "readsu_status_at_open": readsu_status_at_open,
            "open_mode": "memory",
            "power_cycled_this_session": False,
        },
        "camera": {
            "model": getattr(camera, "model_name", None) or str(config.cam_type),
            "serial": device_info.get("cam_serial"),
            "bit_depth": getattr(camera, "bit_depth", None),
            "full_scale": full_scale,
            "exposure_range_ms": device_info.get("cam_exposure_range_ms"),
            "exposure_used_ms": float(exposure_used_ms),
            "gain": getattr(camera, "gain", None),
            "averaging_frames": None,
            "discard_frames_after_exposure_change": None,
            "pixel_size_um": pixel_size_um,
        },
        "wfs": "not used",
        "dm": "not used",
    }


# ---------------------------------------------------------------------------
# Fresnel D 标定段 (--focus-calibration, 默认关闭)
# ---------------------------------------------------------------------------
# 目的: 让 S0 记录里的 ``fresnel_compensation_m_inv`` 变成**本窗口实测值**, 而不
# 是一直硬编码 ``None``。设备无关的扫描与判定口径全在
# :func:`ao_shaping.tools.slm.panel.focus_probe.sweep_focus`, 这里只做
# 「设备 ↔ 参数」的接线、中心/半径的显式给定与证据落盘。

#: 证据里写清「这份 D 是怎么来的」, 便于后人独立复算, 不把结论当事实引用。
_FOCAL_CALIBRATION_METHOD = (
    "focus_probe.sweep_focus: 每个屈光度点重设曝光后按 peak/ms (counts/ms) 排序, "
    "极值点做抛物线细化并复测; D 只在有可用点、曝光线性度落在 0.8–1.25、"
    "且光路未受阻时才写入 S0 记录, 否则 fresnel_compensation_m_inv 保持 null"
)


def _parse_focus_centre(text: str) -> tuple[float, float]:
    """``--focus-centre-px``: SLM **面板**光瞳中心 (panel px, 浮点)。"""
    return _parse_float_pair(text, "--focus-centre-px")


def _resolve_focus_centre(
    config: ProbeConfig,
) -> tuple[tuple[float, float], str]:
    """给出 sweep_focus 的照明光瞳中心 (**SLM 面板坐标**, 不是相机帧坐标) 与来源。

    ``centre_px`` 是 :func:`fresnel_panel` 的 ``pupil_center``: 它决定 Fresnel 相位
    加在面板的哪个圆上。相机帧只通过 ``read_frame`` 提供峰值读数, 与它的坐标空间
    无关 —— 所以这里**绝不**用相机帧里测到的零级质心/argmax 兜底 (量纲错: 那是
    2f Fourier 后焦面上的空间频率坐标)。

    缺省 = 本次 S0 探针自己的光瞳中心 (config ``pupil_center``, CLI 默认
    (960, 600)), 与同一次探测里离焦相位的口径一致; 它也拿不到时才回落到
    ``BEAM_CENTER_PANEL``。
    """
    if config.focus_centre_px is not None:
        return (
            float(config.focus_centre_px[0]),
            float(config.focus_centre_px[1]),
        ), "cli"
    pupil_center = config.pupil_center
    if pupil_center is not None:
        return (float(pupil_center[0]), float(pupil_center[1])), "config_pupil_center"
    return (
        float(FOCUS_BEAM_CENTER_PANEL[0]),
        float(FOCUS_BEAM_CENTER_PANEL[1]),
    ), "default_beam_center"


def _run_slm_self_check_for_focus(
    config: ProbeConfig, slm: Any
) -> dict[str, Any] | str | None:
    """``--focus-calibration`` 的相位自检前置 (只读回, 不写相位)。

    返回 ``dict`` = 自检回读记录, ``str`` = 失败原因 (标定段因此中止), ``None`` =
    未开标定段 (老行为逐位不变, 连调用都不发生)。

    失败**不抛**: 只中止这一段标定, 前面的曝光阶梯 / 离焦响应读数仍然有效 ——
    S0 的判定与退出码不因为 D 没测出来而作废。
    """
    if not config.focus_calibration:
        return None
    try:
        return dict(slm_self_check(slm))
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        logger.error(
            f"SLM 相位自检失败, Fresnel D 标定段中止 (其余探测读数仍有效): {reason}"
        )
        return reason


def _resolve_camera_full_scale(
    camera: Any, read_frame: Any
) -> tuple[float | None, str]:
    """相机的生效满量程 —— **按设备位深/相机契约取, 不按帧 dtype 猜**。

    大恒 MONO12 的帧 dtype 是 ``uint16``, 按 dtype 会得到 65535 而真实满量程是
    4095 (issue #246): 每个点都按自己的曝光归一, 满量程错了 ⇒ peak/ms 排序全错。
    """
    reported = getattr(camera, "bit_depth", None)
    if reported is not None:
        return float((1 << int(reported)) - 1), "camera_bit_depth"
    resolver = getattr(camera, "effective_full_scale", None)
    if callable(resolver):
        try:
            value = float(resolver(np.asarray(read_frame())))
        except Exception:  # 兜底路径不能因为后端实现缺陷而中止标定
            value = 0.0
        if value > 0.0:
            return value, "camera_effective_full_scale"
    return None, "unavailable"


def _evaluate_focus_result(result: Any) -> tuple[str, str | None]:
    """把 :class:`FocusProbeResult` 折成 ``(status, 拒绝原因)``。

    status = ``ok`` 时 ``reject_reason`` 为 ``None``; 否则 ``ok`` 之外的每个值都要
    带一句人能读懂的失败原因, 它会被写进 S0 记录的 ``reasons``。

    强制判据 (与 ``docs/dev/hw_window_20261123_attribution_prereg.md`` §10.4 对应):

    - **D1 内点**: 最优点必须落在扫描窗口内部, 压端点 ⇒ ``out_of_range``;
    - **D2 单峰**: 最优点两侧各至少 2 个可用采样点的 ``peak_per_ms`` 严格单调
      下降 (越往外越低), 否则 ``ambiguous``;
    - **D3 (按模式)**: ``adapt`` 走跨曝光线性度; ``fixed`` 模式 (单曝光扫描) 把
      它换成 **同曝光复测** —— ``peak_per_ms`` 被加性本底 ``peak(ms)=k*ms+B``
      破坏, 跨曝光比值不再能校验, 只剩「同一点、同曝光重测两次峰值一致」这一条
      守门。

    D2 两种模式都用 ``peak_per_ms`` 而不是 raw ``peak``: 单曝光扫描只是把每个
    peak 除以**同一个**固定曝光, 只差一个常数因子, 单调性完全不变, 所以一条
    代码路径同时服务 ``adapt`` 与 ``fixed``。D4 (饱和) / D5 (可复算) 在采集与
    落盘侧判定, 不在本函数里。
    """
    if bool(getattr(result, "path_blocked", False)):
        reason = str(getattr(result, "blockage_reason", "") or "画面里没有局域光斑")
        return "path_blocked", f"光路受阻: {reason}"
    best = getattr(result, "best", None)
    best_diopter = getattr(result, "best_diopter_m1", None)
    usable = [
        p for p in getattr(result, "points", ()) if getattr(p, "status", "") == "ok"
    ]
    if best is None or best_diopter is None or not usable:
        notes = "; ".join(getattr(result, "notes", ())) or "扫描范围内没有可用点"
        return "no_usable_point", f"没有可用点: {notes}"
    # D1: 只允许一次外扩; 外扩之后最优点仍压在 (实测) 扫描边界上, 说明真正的 D
    # 在窗口之外 —— 这个值不能当标定结果写进 S0 记录。
    scan_range = getattr(result, "range_m1", None)
    if scan_range is not None:
        lower, upper = float(scan_range[0]), float(scan_range[1])
        value = float(best_diopter)
        if math.isclose(value, lower, rel_tol=1e-6, abs_tol=1e-9) or math.isclose(
            value, upper, rel_tol=1e-6, abs_tol=1e-9
        ):
            return (
                "out_of_range",
                f"最优点 {value:+.3f} m⁻¹ 仍落在扫描边界 "
                f"[{lower:+.3f}, {upper:+.3f}] (最多外扩 {FOCUS_MAX_EXTENDS} 次), "
                "真实 D 可能在窗口外",
            )
    # D2: 最优点必须是单峰 —— 两侧各至少 2 个可用采样点的 peak_per_ms 严格
    # 单调下降 (越往外越低)。真实 R4 (20261124_102701Z) 的左侧第二点反而抬高,
    # 峰值有两个候选 ⇒ 那个 -0.5833 不许当 D 写进记录。
    ordered = sorted(usable, key=lambda p: float(p.diopter_m1))
    winner = next(
        (
            p
            for p in ordered
            if math.isclose(
                float(p.diopter_m1), float(best_diopter), rel_tol=1e-9, abs_tol=1e-9
            )
        ),
        None,
    )
    if winner is None:
        winner = max(ordered, key=lambda p: float(p.peak_per_ms))
    index = ordered.index(winner)
    diopters = [float(p.diopter_m1) for p in ordered]
    peaks = [float(p.peak_per_ms) for p in ordered]
    if index < 2 or index > len(ordered) - 3:
        return (
            "ambiguous",
            f"最优点 {diopters[index]:+.3f} m⁻¹ 内点两侧采样点不足 2 个 "
            f"(左 {index} 点, 右 {len(ordered) - index - 1} 点), 峰形不唯一",
        )
    left_ok = peaks[index - 2] < peaks[index - 1] < peaks[index]
    right_ok = peaks[index] > peaks[index + 1] > peaks[index + 2]
    if not (left_ok and right_ok):
        broken: list[str] = []
        if not left_ok:
            broken.append(
                f"左: {diopters[index - 2]:+.3f}→{peaks[index - 2]:.0f}, "
                f"{diopters[index - 1]:+.3f}→{peaks[index - 1]:.0f}, "
                f"最优→{peaks[index]:.0f}"
            )
        if not right_ok:
            broken.append(
                f"右: 最优→{peaks[index]:.0f}, "
                f"{diopters[index + 1]:+.3f}→{peaks[index + 1]:.0f}, "
                f"{diopters[index + 2]:+.3f}→{peaks[index + 2]:.0f}"
            )
        return (
            "ambiguous",
            f"最优点 {diopters[index]:+.3f} m⁻¹ 两侧 peak/ms 未各单调下降 2 点 "
            f"({'; '.join(broken)}), 峰形不唯一",
        )
    if getattr(result, "exposure_mode", FOCUS_EXPOSURE_MODE_ADAPT) == (
        FOCUS_EXPOSURE_MODE_FIXED
    ):
        # 单曝光模式: peak_per_ms 被加性本底弄坏了 (peak(ms)=k*ms+B, 跨曝光
        # 比值天然偏 1), 所以 D3 从「跨曝光线性度」换成「同曝光复测」。
        repeat = getattr(result, "repeat_ratio", None)
        if repeat is None:
            return (
                "repeatability_unavailable",
                "单曝光模式缺少同曝光复测点, 峰值排序不可信",
            )
        if not (FOCUS_REPEAT_MIN <= float(repeat) <= FOCUS_REPEAT_MAX):
            return (
                "repeatability_failed",
                f"同曝光复测比 {float(repeat):.3f} 偏离 1 ({FOCUS_REPEAT_MIN:g}-"
                f"{FOCUS_REPEAT_MAX:g}, 单曝光扫描无跨曝光归一化, 只能靠重复性"
                "守门), 峰值排序不可信",
            )
        return "ok", None
    ratio = getattr(result, "linearity_ratio", None)
    if ratio is None:
        return (
            "linearity_unavailable",
            "FocusProbeResult 未提供曝光线性度, peak/ms 排序不可信",
        )
    if not (FOCUS_LINEARITY_MIN <= float(ratio) <= FOCUS_LINEARITY_MAX):
        return (
            "linearity_failed",
            f"曝光线性度 {float(ratio):.2f} 偏离 1 ({FOCUS_LINEARITY_MIN:g}–"
            f"{FOCUS_LINEARITY_MAX:g}), peak/ms 排序不可信",
        )
    return "ok", None


def _write_focus_calibration_json(out_dir: Path, evidence: dict[str, Any]) -> Path:
    """证据文件落盘 (UTF-8 无 BOM), 与 S0 记录里的 ``fresnel_calibration`` 同源。"""
    path = out_dir / "focus_calibration.json"
    path.write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return path


def _focus_evidence(
    *,
    status: str,
    reject_reason: str | None,
    config: ProbeConfig,
    centre_px: tuple[float, float],
    centre_source: str,
    radius_px: float,
    floor_ms: float,
    ceiling_ms: float,
    start_exposure_ms: float,
    full_scale: float | None,
    full_scale_source: str,
    pixel_um: float,
    self_check: dict[str, Any] | str | None,
    slm_serial: str | None,
    cam_serial: str | None,
    measured: dict[str, Any] | None,
    run_id: str,
) -> dict[str, Any]:
    evidence: dict[str, Any] = {
        "method": _FOCAL_CALIBRATION_METHOD,
        "run_id": run_id,
        "timestamp_utc": _utc_iso(),
        "status": status,
        "accepted": status == "ok",
        "reject_reason": reject_reason,
        "centre_px": [float(centre_px[0]), float(centre_px[1])],
        "centre_source": centre_source,
        "radius_px": float(radius_px),
        "full_scale": full_scale,
        "full_scale_source": full_scale_source,
        "pixel_um": float(pixel_um),
        "slm_self_check": self_check,
        "slm_serial": slm_serial,
        "cam_serial": cam_serial,
        "requested": {
            "coarse_points": int(config.focus_coarse_points),
            "diopter_range": float(config.focus_diopter_range),
            "refine_points": int(config.focus_refine_points),
            "frames": int(config.focus_frames),
            "max_extends": FOCUS_MAX_EXTENDS,
            "floor_ms": float(floor_ms),
            "ceiling_ms": float(ceiling_ms),
            "exposure_mode": str(
                getattr(config, "focus_exposure_mode", FOCUS_EXPOSURE_MODE_ADAPT)
            ),
            "fixed_exposure_ms_requested": getattr(
                config, "focus_fixed_exposure_ms", None
            ),
            "start_exposure_ms": float(start_exposure_ms),
            "wavelength_nm": int(config.slm_wavelength_nm),
        },
    }
    if measured is not None:
        # best_diopter_m1 / points / linearity_ratio / notes 等原始读数。
        evidence.update(measured)
        best_diopter = measured.get("best_diopter_m1")
        best_point = next(
            (
                point
                for point in measured.get("points", [])
                if point.get("diopter_m1") == best_diopter
            ),
            None,
        )
        evidence["peak_per_ms"] = (
            float(best_point["peak_per_ms"]) if best_point is not None else None
        )
        evidence["exposure_ms"] = (
            float(best_point["exposure_ms"]) if best_point is not None else None
        )
    else:
        evidence["peak_per_ms"] = None
        evidence["exposure_ms"] = None
    return evidence


def _run_focus_calibration(
    *,
    slm: Any,
    camera: Any,
    config: ProbeConfig,
    centre_px: tuple[float, float],
    centre_source: str,
    radius_px: float,
    floor_ms: float,
    ceiling_ms: float,
    start_exposure_ms: float,
    self_check: dict[str, Any] | str | None,
    out_dir: Path,
    run_id: str,
    slm_serial: str | None = None,
    cam_serial: str | None = None,
) -> dict[str, Any]:
    """跑一次 Fresnel D 扫描, 返回 (并落盘) 证据对象。

    Args:
        slm / camera: 已经打开 (握在设备锁里) 的设备。
        config: 本次探测的配置, 标定段的参数全从它来。
        centre_px / centre_source: :func:`_resolve_focus_centre` 的产物。
        self_check: :func:`_run_slm_self_check_for_focus` 的产物; 是字符串即失败。
    """
    # 单曝光模式: 预注册的曝光优先于 S0 阶梯的推荐值 —— 整段扫描的唯一曝光
    # 必须在运行命令里就写死 (R3 的教训: 混两种曝光会毁掉排序)。越界仍由
    # sweep_focus 夹到 [floor_ms, ceiling_ms]。
    focus_mode = str(
        getattr(config, "focus_exposure_mode", FOCUS_EXPOSURE_MODE_ADAPT)
    )
    override_ms = getattr(config, "focus_fixed_exposure_ms", None)
    if focus_mode == FOCUS_EXPOSURE_MODE_FIXED:
        if override_ms is not None:
            logger.info(
                f"单曝光模式: 使用预注册曝光 {float(override_ms):g} ms "
                f"(覆盖 S0 推荐 {float(start_exposure_ms):g} ms, 越界按 "
                f"{float(floor_ms):g}-{float(ceiling_ms):g} ms 夹取)"
            )
            start_exposure_ms = float(override_ms)
    elif override_ms is not None:
        logger.info(
            f"--focus-fixed-exposure-ms={float(override_ms):g} 只在 "
            "--focus-exposure-mode fixed 下生效, 当前 adapt 模式已忽略"
        )

    logger.info(
        f"Fresnel 标定: mode={focus_mode} "
        f"coarse_points={int(config.focus_coarse_points)} "
        f"diopter_range={float(config.focus_diopter_range)} "
        f"refine={int(config.focus_refine_points)} frames={int(config.focus_frames)} "
        f"centre=({centre_px[0]:g}, {centre_px[1]:g}) [{centre_source}], "
        f"radius={float(radius_px):g} px, 曝光窗口 "
        f"{float(floor_ms):g}-{float(ceiling_ms):g} ms, 起曝 "
        f"{float(start_exposure_ms):g} ms"
    )

    evidence: dict[str, Any]
    if isinstance(self_check, str):
        evidence = _focus_evidence(
            status="slm_self_check_failed",
            reject_reason=(f"SLM 相位自检失败：{self_check}，需人工检查面板状态后重排"),
            config=config,
            centre_px=centre_px,
            centre_source=centre_source,
            radius_px=radius_px,
            floor_ms=floor_ms,
            ceiling_ms=ceiling_ms,
            start_exposure_ms=start_exposure_ms,
            full_scale=None,
            full_scale_source="skipped",
            pixel_um=0.0,
            self_check=self_check,
            slm_serial=slm_serial,
            cam_serial=cam_serial,
            measured=None,
            run_id=run_id,
        )
        _write_focus_calibration_json(out_dir, evidence)
        logger.error(
            f"Fresnel 标定中止 (SLM 相位自检失败, 不写 D): {evidence['reject_reason']}"
        )
        return evidence

    pixel_um = float(getattr(camera, "pixel_size_um", 0.0) or 0.0)
    if pixel_um <= 0.0:
        evidence = _focus_evidence(
            status="unavailable",
            reject_reason=f"相机未登记像元尺寸 (pixel_size_um={pixel_um:g}), 无法生成 Fresnel 相位",
            config=config,
            centre_px=centre_px,
            centre_source=centre_source,
            radius_px=radius_px,
            floor_ms=floor_ms,
            ceiling_ms=ceiling_ms,
            start_exposure_ms=start_exposure_ms,
            full_scale=None,
            full_scale_source="skipped",
            pixel_um=pixel_um,
            self_check=self_check,
            slm_serial=slm_serial,
            cam_serial=cam_serial,
            measured=None,
            run_id=run_id,
        )
        _write_focus_calibration_json(out_dir, evidence)
        logger.error(
            f"Fresnel 标定中止 (相机的像元尺寸不可用): {evidence['reject_reason']}"
        )
        return evidence

    def read_frame() -> np.ndarray:
        return np.asarray(camera.get_numpy_image(n_sample=1))

    full_scale, full_scale_source = _resolve_camera_full_scale(camera, read_frame)

    def _set_exposure_ms(ms: float) -> float:
        """改曝光并回读生效值 (驱动会夹到设备范围, 请求值不是生效值)。"""
        set_camera_exposure_ms(camera, float(ms))
        return float(get_camera_exposure_ms(camera))

    status = "ok"
    reject_reason: str | None = None
    measured: dict[str, Any] | None = None
    try:
        result = sweep_focus(
            apply_phase=lambda phase: display_phase_on_slm(slm, phase, config.settle_s),
            read_frame=read_frame,
            set_exposure_ms=_set_exposure_ms,
            centre_px=centre_px,
            radius_px=radius_px,
            coarse_points=int(config.focus_coarse_points),
            diopter_range=float(config.focus_diopter_range),
            refine_points=int(config.focus_refine_points),
            start_exposure_ms=float(start_exposure_ms),
            floor_ms=float(floor_ms),
            ceiling_ms=float(ceiling_ms),
            frames=int(config.focus_frames),
            max_extends=FOCUS_MAX_EXTENDS,
            full_scale=full_scale,
            wavelength_nm=float(config.slm_wavelength_nm),
            pixel_um=pixel_um,
            exposure_mode=focus_mode,
        )
        measured = dict(result.as_evidence())
        status, reject_reason = _evaluate_focus_result(result)
    except Exception as exc:  # 设备抖动不该让整次探测死掉, 但不许把 D 写出来
        status = "error"
        reject_reason = f"Fresnel 标定执行失败 ({type(exc).__name__}: {exc})"

    evidence = _focus_evidence(
        status=status,
        reject_reason=reject_reason,
        config=config,
        centre_px=centre_px,
        centre_source=centre_source,
        radius_px=radius_px,
        floor_ms=floor_ms,
        ceiling_ms=ceiling_ms,
        start_exposure_ms=start_exposure_ms,
        full_scale=full_scale,
        full_scale_source=full_scale_source,
        pixel_um=pixel_um,
        self_check=self_check,
        slm_serial=slm_serial,
        cam_serial=cam_serial,
        measured=measured,
        run_id=run_id,
    )
    path = _write_focus_calibration_json(out_dir, evidence)

    best = evidence.get("best_diopter_m1")
    if best is not None:
        logger.info(
            f"Fresnel 标定: best={float(best):+.3f} m⁻¹, "
            f"peak/ms={float(evidence['peak_per_ms']):.2f}, "
            f"exposure={float(evidence['exposure_ms']):g} ms, "
            f"status={status}" + (f" ({reject_reason})" if reject_reason else "")
        )
    else:
        logger.info(f"Fresnel 标定: 无可用结论 (status={status}): {reject_reason}")
    logger.info(f"Fresnel 标定证据: {path}")
    return evidence


def _maybe_run_focus_calibration(
    *,
    slm: Any,
    camera: Any,
    config: ProbeConfig,
    start_exposure_ms: float,
    out_dir: Path,
    run_id: str,
    self_check: dict[str, Any] | str | None = None,
    slm_serial: str | None = None,
    cam_serial: str | None = None,
) -> dict[str, Any] | None:
    """标定段总开关: 关闭时直接返回 ``None`` (与没有这段的行为逐位一致)。"""
    if not config.focus_calibration:
        return None
    centre_px, centre_source = _resolve_focus_centre(config)
    radius_px = (
        float(config.focus_radius_px)
        if config.focus_radius_px is not None
        else float(FOCUS_BEAM_RADIUS_PANEL)
    )
    floor_ms = (
        float(config.focus_floor_ms)
        if config.focus_floor_ms is not None
        else float(FOCUS_FLOOR_MS)
    )
    ceiling_ms = (
        float(config.focus_ceiling_ms)
        if config.focus_ceiling_ms is not None
        else (float(max(config.exposures_ms)) if config.exposures_ms else 50.0)
    )
    return _run_focus_calibration(
        slm=slm,
        camera=camera,
        config=config,
        centre_px=centre_px,
        centre_source=centre_source,
        radius_px=radius_px,
        floor_ms=floor_ms,
        ceiling_ms=ceiling_ms,
        start_exposure_ms=start_exposure_ms,
        self_check=self_check,
        out_dir=out_dir,
        run_id=run_id,
        slm_serial=slm_serial,
        cam_serial=cam_serial,
    )


def _record_inputs(
    *,
    run_id: str,
    config_id: str,
    probe_report: Path,
    config: ProbeConfig,
    slm: Any,
    camera: Any,
    device_info: dict[str, Any],
    exposure_used_ms: float,
    full_scale: float | None,
    ladder_summary: dict[str, Any],
    response: dict[str, Any],
    verdict: dict[str, Any],
    acquired_at: str,
    focus_calibration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """把已经采到的读数整理成 ``build_optical_path_record`` 的输入。

    只做搬运, 不碰设备: 设备相关的读取全在 :func:`_collect_record_devices` 里,
    到了这一步已经全是普通数字/字符串。

    ``focus_calibration`` 是 :func:`_run_focus_calibration` 的证据对象。给了它就
    用**实测的** D 填 ``fresnel_compensation_m_inv`` 并把证据挂进
    ``optical_path.fresnel_calibration``; 没给 (默认关) 时逐位保持老行为:
    D = ``None`` + 那句「探针不做 Fresnel 补偿标定」的 reason。
    """
    reasons: dict[str, str] = {}
    devices = _collect_record_devices(
        slm,
        camera,
        config,
        full_scale=full_scale,
        exposure_used_ms=exposure_used_ms,
        device_info=device_info,
        reasons=reasons,
    )

    geometry = ROOT / "data" / "slm_geometry_calib.npz"
    geometry_mtime: str | None = None
    if geometry.exists():
        geometry_mtime = datetime.fromtimestamp(
            geometry.stat().st_mtime, tz=timezone.utc
        ).isoformat(timespec="seconds")
    reasons["optical_path.aperture"] = "探针不读光阑状态 (aperture 需现场填写)"
    reasons["optical_path.fresnel_compensation_m_inv"] = (
        "探针不做 Fresnel 补偿标定 (compensation 由焦面标定给出)"
    )
    reasons["exposure_check.settle_timeout"] = (
        "探针用固定 settle 等待, 不检测 settle 超时"
    )

    saturation_counts = float(ladder_summary["saturation_counts"])
    baseline_peak = float(response["baseline_peak_counts"])

    # Fresnel D: 只在标定段真跑过、且通过了它的自检 (曝光线性度 / 有可用点 /
    # 光路没被挡) 时才写实测值。可疑值绝不进记录 —— 保持 None 并在 reason 里写
    # 明实测失败的原因, 这样 D 缺失永远是「没测出来」而不是「测了不放心」。
    fresnel_compensation: float | None = None
    if focus_calibration is not None:
        status = str(focus_calibration.get("status") or "unknown")
        reject_reason = str(focus_calibration.get("reject_reason") or "无可用结论")
        best = focus_calibration.get("best_diopter_m1")
        if bool(focus_calibration.get("accepted")) and best is not None:
            fresnel_compensation = float(best)
            reasons.pop("optical_path.fresnel_compensation_m_inv", None)
        elif status == "slm_self_check_failed":
            # 自检失败的 reject_reason 已带「SLM 相位自检失败：…」前缀, 这里再补
            # status 与「D 保持 null」, 免得记录里停在那句「探针不做标定」的旧话上。
            reasons["optical_path.fresnel_compensation_m_inv"] = (
                f"{reject_reason} (status={status}; D 保持 null, 未写入记录)"
            )
        else:
            reasons["optical_path.fresnel_compensation_m_inv"] = (
                f"Fresnel 标定实测失败 (status={status}), D 保持 null 未写入记录: "
                f"{reject_reason}"
            )

    optical_path: dict[str, Any] = {
        "config_id": config_id,
        "aperture": None,
        "fresnel_compensation_m_inv": fresnel_compensation,
        "geometry_calib_file": (_rel_to_root(geometry) if geometry.exists() else None),
        "geometry_calib_mtime": geometry_mtime,
        "probe_status": verdict["state"],
        "probe_report": _rel_to_root(probe_report),
    }
    if focus_calibration is not None:
        # 与 focus_calibration.json 同源: 同一个 dict, 不改一份留另一份。
        optical_path["fresnel_calibration"] = dict(focus_calibration)

    return {
        "run_id": run_id,
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "operator_note": "",
        "devices": devices,
        "optical_path": optical_path,
        "exposure_check": {
            "peak_counts_at_used_exposure": baseline_peak,
            "saturation_threshold_counts": saturation_counts,
            "saturated": bool(baseline_peak >= saturation_counts),
            "settle_timeout": None,
        },
        # released_at / closed_normally 由 main() 在放锁之后填 (见 _write_pending_record)。
        "occupancy": {
            "acquired_at": acquired_at,
            "released_at": None,
            "closed_normally": None,
        },
        "git_commit": _git_short_sha(),
        "probe_status": verdict["state"],
        "reasons": reasons,
    }


def _write_pending_record() -> Path | None:
    """落盘 ``optical_path_record.json`` (放锁之后, 这样 ``released_at`` 才是真时刻)。

    写失败只告警, **不改退出码**: 记录是旁证, 探针的判定不能因为它丢了。
    """
    global _PENDING_RECORD
    pending, _PENDING_RECORD = _PENDING_RECORD, None
    if pending is None:
        return None
    inputs = dict(pending["inputs"])
    occupancy = dict(inputs.get("occupancy") or {})
    occupancy["released_at"] = _utc_iso()
    occupancy["closed_normally"] = True
    inputs["occupancy"] = occupancy
    try:
        record = build_optical_path_record(**inputs)
        # build_optical_path_record intentionally keeps optical_path to its fixed
        # schema fields; the Fresnel evidence is this probe's additive extension.
        # Attach it after the canonical builder so its full independently
        # recomputable payload survives in optical_path_record.json.
        focus_evidence = (inputs.get("optical_path") or {}).get("fresnel_calibration")
        if focus_evidence is not None:
            record.setdefault("optical_path", {})["fresnel_calibration"] = (
                focus_evidence
            )
        path = Path(pending["out_dir"]) / "optical_path_record.json"
        path.write_text(
            json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except Exception as exc:  # 记录写盘失败不得影响判定/退出码
        logger.warning(f"标准光路记录写盘失败 (不影响判定): {exc}")
        return None
    logger.info(f"标准记录: {path}")
    return path


def run_probe(args: argparse.Namespace) -> int:
    """探测主体 (调用方已拿到设备锁)。"""
    exposures = _parse_float_list(args.exposures, "--exposures")
    amplitudes = _parse_float_list(args.defocus_amps, "--defocus-amps")
    pupil_center = _parse_center(args.pupil_center)
    for exposure in exposures:
        if exposure <= 0:
            raise SystemExit(f"曝光必须为正: {exposure}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    session = str(getattr(args, "session", "") or "").strip()
    out_dir = Path(args.out_root) / session / stamp if session else Path(args.out_root) / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    # 设备锁在进入 run_probe 之前(acquire_devices → run_probe)刚拿到, 这里就是
    # 标准记录里 occupancy.acquired_at 的时刻。
    acquired_at = _utc_iso()
    run_tag = session or "probe"
    run_id = f"{stamp[0:8]}_{stamp[9:15]}_{run_tag}"

    config = ProbeConfig(
        exposures_ms=[float(v) for v in exposures],
        defocus_amps_rad=[float(v) for v in amplitudes],
        frames_per_exposure=int(args.frames),
        frames_per_defocus=int(args.defocus_frames),
        defocus_radius_px=int(args.defocus_radius),
        pupil_center=pupil_center,
        defocus_exposure_ms=(
            float(args.defocus_exposure_ms)
            if args.defocus_exposure_ms is not None
            else None
        ),
        settle_s=float(args.settle_s),
        slm_number=int(args.slm_number),
        slm_wavelength_nm=int(args.slm_wavelength),
        cam_type=str(args.cam_type),
        cam_id=int(args.cam_id),
        bit_depth=int(getattr(args, "bit_depth", S0_BIT_DEPTH)),
        wait_until_idle_s=float(args.wait_until_idle),
        focus_calibration=bool(getattr(args, "focus_calibration", False)),
        focus_coarse_points=int(
            getattr(args, "focus_coarse_points", FOCUS_COARSE_POINTS)
        ),
        focus_diopter_range=float(
            getattr(args, "focus_diopter_range", FOCUS_DIOPTER_RANGE)
        ),
        focus_refine_points=int(
            getattr(args, "focus_refine_points", FOCUS_REFINE_POINTS)
        ),
        focus_frames=int(getattr(args, "focus_frames", FOCUS_FRAMES)),
        focus_floor_ms=(
            float(args.focus_floor_ms) if args.focus_floor_ms is not None else None
        ),
        focus_ceiling_ms=(
            float(args.focus_ceiling_ms) if args.focus_ceiling_ms is not None else None
        ),
        focus_exposure_mode=str(
            getattr(args, "focus_exposure_mode", FOCUS_EXPOSURE_MODE_ADAPT)
        ),
        focus_fixed_exposure_ms=(
            float(args.focus_fixed_exposure_ms)
            if getattr(args, "focus_fixed_exposure_ms", None) is not None
            else None
        ),
        focus_centre_px=args.focus_centre_px,
        focus_radius_px=(
            float(args.focus_radius_px) if args.focus_radius_px is not None else None
        ),
    )
    # 标准记录 (SKILL §4) 的 config_id: 有 lane 名就用 lane 名, 否则用相机+波长
    # 拼一个能对出"这是哪条光路"的标识。
    config_id = session or (
        f"optical_path_probe_{config.cam_type}_{config.slm_wavelength_nm}nm"
    )

    from ao_shaping.drivers.slm.santec import Santec

    slm = Santec(
        slm_number=config.slm_number,
        wavelength=config.slm_wavelength_nm,
        video_mode=0,  # 恒用内存模式: DVI 模式的 open() 会挂死
    )
    camera = create_camera(
        config.cam_type,
        cam_id=config.cam_id,
        exposure_time_ms=float(exposures[0]),
        bit_depth=int(config.bit_depth),
    )

    device_info: dict[str, Any] = {}
    with slm:
        # SLM 相位自检前置: 只在 --focus-calibration 时做, 且在任何相位下发之前。
        # 失败只中止新加的 D 标定段 (见 _run_focus_calibration), S0 的曝光阶梯与
        # 离焦响应照跑, 判定/退出码不受影响。
        focus_self_check = _run_slm_self_check_for_focus(config, slm)
        with camera:
            # 位深必须按设备回读校验: 请求值不是生效值 (gxipy 越界/拉流中写入会
            # 静默保留旧 PixelFormat)。带着错的位深继续, 满量程与全部饱和判据
            # 就都是错量纲 —— 这是 S0 记录落成 8-bit / full_scale 255 的根因。
            _verify_camera_bit_depth(camera, int(config.bit_depth))
            device_info = _collect_device_info(slm, camera)
            logger.info(
                f"设备就绪: SLM #{config.slm_number} "
                f"(序列号 {device_info.get('slm_serial')}), 相机 {config.cam_type} "
                f"#{config.cam_id} (序列号 {device_info.get('cam_serial')})",

            )
            lo, hi = get_camera_exposure_range(camera)
            logger.info(f"相机曝光范围: {lo:g}-{hi:g} ms")

            # 光通量参考: 在采集之前就取 (某次参考缺失时不必先花几分钟采完才发现),
            # 设备已开 ⇒ 序列号可用于匹配历史记录。
            reference: dict[str, Any] | None = None
            baseline_path = Path(args.out_root) / REFERENCE_BASELINE_FILE
            rebaseline = bool(getattr(args, "rebaseline", False))
            if rebaseline:
                logger.info(
                    f"光通量参考: 本次重定基 (--rebaseline) ⇒ 本次不判 weak_path, "
                    f"实测值写入 {baseline_path}"
                )
            elif not bool(getattr(args, "no_reference", False)):
                explicit = getattr(args, "reference_counts_per_ms", None)
                if explicit is not None:
                    value = float(explicit)
                    reference = {
                        "source": "explicit",
                        "counts_per_ms": value,
                        "n": 1,
                        "min_counts_per_ms": value,
                        "max_counts_per_ms": value,
                        "full_scale": None,
                        "runs": [],
                    }
                else:
                    reference = bench_reference(Path(args.out_root), device=device_info)
                if reference is None:
                    logger.info(
                        "光通量参考: 历史 ok 探测不足 "
                        f"{MIN_REFERENCE_SAMPLES} 次, 本次不判 weak_path "
                        "(可用 --reference-counts-per-ms 显式给出, 或用 --rebaseline 重定基)"
                    )
                else:
                    logger.info(
                        f"光通量参考: {reference['counts_per_ms']:.2f} counts/ms "
                        f"({reference['source']}, n={reference['n']}, "
                        f"范围 {reference['min_counts_per_ms']:.2f}-"
                        f"{reference['max_counts_per_ms']:.2f}); "
                        f"下限 {float(args.min_throughput_ratio):.0%} ⇒ "
                        f"{float(args.min_throughput_ratio) * reference['counts_per_ms']:.2f} counts/ms"
                    )
                    skipped = reference.get("skipped") or {}
                    if any(skipped.values()):
                        logger.info(
                            "  参考样本池的排除记账: "
                            + ", ".join(f"{k}={v}" for k, v in skipped.items() if v)
                        )
                    age = reference.get("age_days")
                    if (
                        reference.get("source") == "history"
                        and isinstance(age, (int, float))
                        and age > REFERENCE_STALE_DAYS
                    ):
                        logger.warning(
                            f"  参考中位数来自 {age:.1f} 天前的记录 ⇒ 光功率/器件变动后它不再适用, "
                            "用 --rebaseline 重定基"
                        )

            # -- 阶段一: 曝光阶梯 (面板保持平场) --------------------------
            flat_phase = build_defocus_phase(
                0.0, config.defocus_radius_px, config.pupil_center
            )
            display_phase_on_slm(slm, flat_phase, config.settle_s)
            logger.info("\n[阶段一] 曝光阶梯 (平场):")
            ladder = measure_exposure_ladder(
                camera, exposures, config.frames_per_exposure, out_dir
            )
            for record in ladder:
                logger.info(
                    f"  {record['effective_ms']:>8.3f} ms  峰值={record['peak_raw']:>7.1f} "
                    f"(处理后 {record['peak_prepared']:>7.1f})  "
                    f"均值={record['mean_raw']:>7.3f}  p99={record['p99_raw']:>7.1f}  "
                    f"FWHM={record['fwhm_px']:>6.2f} px  "
                    f"质心=({record['centroid_x']:.0f}, {record['centroid_y']:.0f})",

                )

            # 满量程来自阶梯里各档的实测生效值 (相机位深), 不是 dtype 也不是 255 字面量。
            measured_full_scale = max(
                (float(row.get("full_scale") or 0.0) for row in ladder), default=0.0
            )
            ladder_summary = analyse_ladder(ladder, measured_full_scale or 255.0)
            noise = ladder_summary["noise_floor"]
            logger.info(
                f"  噪声底: 中位数 {noise['median_counts']:.1f} + "
                f"{NOISE_FLOOR_SIGMAS:g}σ ({noise['sigma_counts']:.2f}) = "
                f"{noise['threshold_counts']:.1f} counts @ {noise['exposure_ms']:g} ms",

            )
            logger.info(
                "  可用窗口: "
                + (
                    ", ".join(f"{v:g}" for v in ladder_summary["usable_window_ms"])
                    + " ms"
                    if ladder_summary["usable_window_ms"]
                    else "(空)"
                ),

            )
            recommended = ladder_summary.get("recommended_exposure")
            if recommended is not None:
                logger.info(
                    f"  建议曝光: {recommended['exposure_ms']:g} ms "
                    f"(峰值 {recommended['peak_counts']:.0f} counts)",

                )

            # -- 阶段二: 离焦响应 ---------------------------------------
            if config.defocus_exposure_ms is not None:
                defocus_exposure = float(config.defocus_exposure_ms)
            elif recommended is not None:
                defocus_exposure = float(recommended["exposure_ms"])
            else:
                usable = [r for r in ladder if r.get("usable")]
                pool = usable or ladder
                defocus_exposure = float(max(pool, key=lambda r: r["peak_raw"])["effective_ms"])
            logger.info(
                f"\n[阶段二] 离焦响应 (曝光 {defocus_exposure:g} ms, "
                f"幅度 {config.defocus_amps_rad}):",

            )
            states = measure_defocus_sweep(
                camera,
                slm,
                config.defocus_amps_rad,
                defocus_exposure,
                config.frames_per_defocus,
                config.defocus_radius_px,
                config.pupil_center,
                config.settle_s,
                out_dir,
            )
            for state in states:
                logger.info(
                    f"  离焦 {state['defocus_rad']:>+6.2f} rad  "
                    f"峰值={state['peak_raw']:>7.1f}  "
                    f"FWHM={state['fwhm_px']:>6.2f} px  "
                    f"二阶矩半径={state['second_moment_radius_px']:>6.2f} px  "
                    f"质心=({state['centroid_x']:.0f}, {state['centroid_y']:.0f})  "
                    f"[{state['tag']}]",

                )

            response = analyse_defocus_response(
                states, float(noise["threshold_counts"])
            )
            logger.info(
                f"  平场: 峰值 {response['baseline_peak_counts']:.1f} counts, "
                f"FWHM {response['baseline_fwhm_px']:.2f} px; "
                f"逐帧散布 σ(峰值)={response['sigma_peak_counts']:.2f} counts, "
                f"σ(FWHM)={response['sigma_fwhm_px']:.2f} px",

            )
            logger.info(
                f"  离焦带来的变化 (各幅度中最大): 峰值 "
                f"{response['peak_signal_counts']:.1f} counts "
                f"({response['peak_relative_change']:.1%}, 离焦 "
                f"{response['peak_signal_at_rad']:+g} rad), "
                f"宽度 {response['width_signal_px']:.2f} px "
                f"({response['width_relative_change']:.1%}, 离焦 "
                f"{response['width_signal_at_rad']:+g} rad); "
                f"判据 = {RESPONSE_SIGMAS:g}×合并标准误 或 相对下限 "
                f"({WIDTH_RELATIVE_FLOOR:.0%} / {PEAK_RELATIVE_FLOOR:.0%}) ⇒ "
                f"宽度显著={response['width_significant']}, "
                f"峰值显著={response['peak_significant']}",

            )
            logger.info(
                f"  分辨率: 宽度 {response['width_resolution']:.1%} "
                f"(上限 {WIDTH_RESOLUTION_MAX:.0%}), 峰值 "
                f"{response['peak_resolution']:.1%} "
                f"(上限 {PEAK_RESOLUTION_MAX:.0%})",

            )

            throughput: dict[str, Any] | None = None
            flux = float(ladder_summary.get("counts_per_ms") or float("nan"))
            if reference is not None and math.isfinite(flux) and reference["counts_per_ms"] > 0:
                reference_flux = float(reference["counts_per_ms"])
                throughput = {
                    "counts_per_ms": flux,
                    "reference_counts_per_ms": reference_flux,
                    "reference_source": reference["source"],
                    "reference_n": int(reference["n"]),
                    "reference_runs": list(reference.get("runs") or []),
                    "ratio": flux / reference_flux,
                    "min_ratio": float(args.min_throughput_ratio),
                    "fit_window_ms": ladder_summary.get("counts_per_ms_window_ms"),
                }
                logger.info(
                    f"  光通量: {flux:.2f} counts/ms = 参考的 "
                    f"{throughput['ratio']:.0%} "
                    f"(下限 {throughput['min_ratio']:.0%})"
                )

            verdict = decide(ladder_summary, response, throughput=throughput)

            if rebaseline and math.isfinite(flux):
                scales = [
                    float(row.get("full_scale") or 0.0)
                    for row in (ladder or [])
                    if float(row.get("full_scale") or 0.0) > 0
                ]
                saved = _save_reference_baseline(
                    baseline_path,
                    counts_per_ms=flux,
                    device=device_info,
                    full_scale=max(scales) if scales else None,
                    session=str(getattr(args, "session", "") or ""),
                    note="operator declared this flux as the new normal",
                )
                logger.info(
                    f"  已重定基: 参考 = {saved['counts_per_ms']:.2f} counts/ms "
                    f"({saved['recorded_utc']}); 下一次探测起生效"
                )

            # -- 阶段三: Fresnel D 标定 (默认关闭) -------------------------
            # 必须在放锁之前: 前面两段的曝光/位深纪律已经走完, 这段的曝光还要按
            # 每点重新设, 全程握着同一份 SLM / 相机句柄。
            focus_evidence = _maybe_run_focus_calibration(
                slm=slm,
                camera=camera,
                config=config,
                start_exposure_ms=float(defocus_exposure),
                out_dir=out_dir,
                run_id=run_id,
                self_check=focus_self_check,
                slm_serial=device_info.get("slm_serial"),
                cam_serial=device_info.get("cam_serial"),
            )

            # 标准记录 (SKILL §4, ``ao-optical-path-record/v1``): 在这里只组装输入 ——
            # 设备还开着 (设备标识要现场读), 而写盘要等 main() 放锁之后才能补上
            # occupancy.released_at, 所以先挂到模块级的 _PENDING_RECORD 上。
            global _PENDING_RECORD
            _PENDING_RECORD = {
                "out_dir": out_dir,
                "inputs": _record_inputs(
                    run_id=run_id,
                    config_id=config_id,
                    probe_report=out_dir / "optical_path_probe.json",
                    config=config,
                    slm=slm,
                    camera=camera,
                    device_info=device_info,
                    exposure_used_ms=float(defocus_exposure),
                    full_scale=(
                        max(float(row.get("full_scale") or 0.0) for row in ladder)
                        or None
                    ),
                    ladder_summary=ladder_summary,
                    response=response,
                    verdict=verdict,
                    acquired_at=acquired_at,
                    focus_calibration=focus_evidence,
                ),
            }

    thresholds = {
        "noise_floor_sigmas": NOISE_FLOOR_SIGMAS,
        "saturation_fraction": SATURATION_FRACTION,
        "response_sigmas": RESPONSE_SIGMAS,
        "fwhm_sigma_floor_px": FWHM_SIGMA_FLOOR_PX,
        "peak_sigma_floor_counts": PEAK_SIGMA_FLOOR_COUNTS,
        "min_visible_peak_counts": DEFAULT_MIN_PEAK,
        # 目标带按**本次实测满量程**折算 (Mono12: 1606-3212 counts), 与记录里
        # 其它 raw-count 数字同一量纲; 0-255 灰度口径见 target_band_gray。
        "recommended_peak_range_counts": list(ladder_summary["target_band_counts"]),
        "recommended_peak_range_gray": list(ladder_summary["target_band_gray"]),
        "width_relative_floor": WIDTH_RELATIVE_FLOOR,
        "peak_relative_floor": PEAK_RELATIVE_FLOOR,
        "width_resolution_max": WIDTH_RESOLUTION_MAX,
        "peak_resolution_max": PEAK_RESOLUTION_MAX,
        "defocus_noll_index": DEFOCUS_NOLL_INDEX,
    }
    payload = {
        "schema": "optical_path_probe/1",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": [sys.executable, *sys.argv],
        "config": {
            "exposures_ms": config.exposures_ms,
            "defocus_amps_rad": config.defocus_amps_rad,
            "frames_per_exposure": config.frames_per_exposure,
            "frames_per_defocus": config.frames_per_defocus,
            "defocus_radius_px": config.defocus_radius_px,
            "pupil_center_px": list(config.pupil_center),
            "defocus_exposure_ms": defocus_exposure,
            "settle_s": config.settle_s,
            "slm_number": config.slm_number,
            "slm_wavelength_nm": config.slm_wavelength_nm,
            "cam_type": config.cam_type,
            "cam_id": config.cam_id,
            "bit_depth": config.bit_depth,
            "wait_until_idle_s": config.wait_until_idle_s,
        },
        "device": device_info,
        "exposure_ladder": ladder,
        "ladder_summary": ladder_summary,
        "defocus_sweep": states,
        "defocus_response": response,
        "throughput": throughput,
        "thresholds": thresholds,
        "verdict": verdict,
    }
    json_path = out_dir / "optical_path_probe.json"
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if not args.no_plot:
        plot_results(
            ladder,
            ladder_summary,
            response,
            out_dir / "optical_path_probe.png",
        )

    logger.info(f"\n判定: {verdict['state']}")
    for reason in verdict["reasons"]:
        logger.info(f"  - {reason}")
    for item in verdict["advice"]:
        logger.info(f"  * {item}")
    logger.info(f"证据: {out_dir}")
    return int(verdict["exit_code"])


def _verify_camera_bit_depth(camera: Any, requested: int) -> int | None:
    """核对相机**设备回读**的位深; 不符即报错, 绝不带着错的满量程继续。

    位深决定满量程 (8/12/16 bit → 255/4095/65535), 全部饱和/线性度/均匀度判据都
    按满量程折算 (ccd/AGENTS.md 满量程小节)。8 bit 下 1 count ≈ 7% 满量程, S0 的
    均匀度结论会直接失效 —— 所以这里宁可停, 也不静默继续。

    没有位深真值的后端 (ffmpeg / image_folder / 仿真: ``bit_depth`` 恒 ``None``)
    不参与校验: 它们的满量程按帧 dtype 兜底, 不存在「位深错了」这回事。

    返回:
        设备回读的位深; 该后端不报告位深时返回 ``None``。

    抛出:
        RuntimeError: 回读值 ≠ 请求值 (调用方拿到的位深不可信)。
    """
    reported = getattr(camera, "bit_depth", None)
    if reported is None:
        logger.warning(
            f"相机 {type(camera).__name__} 不报告位深 (bit_depth=None): "
            f"跳过 --bit-depth={int(requested)} 的校验, 满量程按帧 dtype 兜底"
        )
        return None
    actual = int(reported)
    if actual != int(requested):
        supported = getattr(camera, "supported_bit_depths", None)
        raise RuntimeError(
            f"相机位深与请求不符: 请求 {int(requested)} bit, 设备回读 {actual} bit。"
            f"满量程会按回读位深 ({(1 << actual) - 1}) 折算, 与本次探测假设的 "
            f"{(1 << int(requested)) - 1} 不是同一量纲; 继续下去饱和/均匀度判据全错。"
            f"该后端位深档位: supported_bit_depths={supported} "
            "(拉流中替换 PixelFormat 不生效, 位深必须在开流前写好)。"
        )
    full_scale = (1 << actual) - 1
    logger.info(f"相机位深: {actual} bit (设备回读 = 请求值, 满量程 {full_scale})")
    return actual


def _collect_device_info(slm: Any, camera: Any) -> dict[str, Any]:
    """收集设备标识 (序列号等), 失败不致命。"""
    info: dict[str, Any] = {}
    try:
        info["slm_serial"] = slm.serial_number
    except Exception:  # 设备信息拿不到不影响判定
        info["slm_serial"] = None
    try:
        info["slm_video_mode"] = int(getattr(slm, "video_mode", 0))
    except Exception:  # 保留宽泛捕获: 本处契约是任何失败都不应向上抛出, 具体类型不可枚举
        info["slm_video_mode"] = None
    try:
        info["slm_wavelength_nm"] = getattr(slm, "wavelength", None)
    except Exception:  # 保留宽泛捕获: 本处契约是任何失败都不应向上抛出, 具体类型不可枚举
        info["slm_wavelength_nm"] = None
    for attr in ("serial_number", "serial", "sn"):
        value = getattr(camera, attr, None)
        if isinstance(value, str) and value:
            info["cam_serial"] = value
            break
    else:
        info["cam_serial"] = None
    try:
        hardware = camera.get_hardware_info()
        if isinstance(hardware, dict):
            info["cam_hardware"] = {
                str(k): str(v) for k, v in hardware.items() if isinstance(v, (str, int, float))
            }
    except Exception:  # 大恒驱动没有实现 get_hardware_info
        pass
    try:
        lo, hi = get_camera_exposure_range(camera)
        info["cam_exposure_range_ms"] = [float(lo), float(hi)]
    except Exception:  # 保留宽泛捕获: 内部调用链 get_camera_exposure_range 的失败类型不可枚举
        pass
    # 已按设备回读校验过的位深 (见 _verify_camera_bit_depth); 无位深真值的后端记 None。
    # 它是满量程的唯一来源, 标准记录里必须能对出「这一份数据是几 bit 采的」。
    try:
        bit_depth = getattr(camera, "bit_depth", None)
        info["cam_bit_depth"] = int(bit_depth) if bit_depth is not None else None
    except (TypeError, ValueError):  # 后端报了个非整数的位深 ⇒ 按「无真值」记
        info["cam_bit_depth"] = None
    return info


if __name__ == "__main__":
    raise SystemExit(main())
