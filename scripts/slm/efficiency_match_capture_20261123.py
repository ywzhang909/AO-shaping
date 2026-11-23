# -*- coding: utf-8 -*-
"""效率匹配性实验的采集端: 冻结 16 帧矩阵 (2026-11-23)。

为什么要有这个脚本
------------------
``docs/dev/efficiency_match_experiment_plan_20261121.md`` 预注册了「当前光路下
远场整形效率 η_s 与同光路 blaze 衍射效率 η_d 是否匹配」的比较实验, 但它的
采集矩阵只在计划文档里冻结过, 没有可执行的落盘实现。本脚本就是那个采集端:
按 §4 的 8 行顺序走两遍 (共 16 帧), 同一曝光、同一设备窗口、同一 SLM LUT,
把 16 张原始帧、相位/灰度、设备身份与逐帧 journal 全部落盘, 供离线分析脚本
复算 η_s / η_d 与预注册判据。

采集矩阵 (冻结; 见 ``MATRIX_ROWS`` / ``MATRIX_INTERPRETATION``)
---------------------------------------------------------------
第一遍: flat_pre → blaze_grating → shaping_target → dark → flat_mid →
blaze_grating → shaping_target → flat_post; 第二遍同样 8 行, id 加 ``_r2``
后缀。合计 flat 6 / blaze 4 / shaping 4 / dark 2。

相位构造
--------
* ``flat``: 全零相位 → 灰度用 ``np.full(shape, 0, uint16)`` raw 路径直发,
  **绝不**走 ``create_phase_from_array`` (它按 mod 2π 折算会破坏 uint16 灰度)。
* ``blaze_grating``: 单一连续 blaze ramp (2π/period, 默认 period=128 面板像元,
  轴默认 panel_x), 复用 1064 nm 面板几何 —— **不是**二值 0/π 光栅
  (计划 §2.3: 二值光栅多级能量分散, 不作 η_d 的一阶口径)。
* ``shaping_target``: 从 ``--shaping-phase`` 读入的面板分辨率弧度相位,
  shape 必须等于实测 ``Panel_Res``, 否则硬失败并报告两个 shape。
* ``dark``: 见下方暗帧纪律。

暗帧纪律 (诚实优先; 无软件快门, 脚本不能熄灭激光)
--------------------------------------------------
1. 相机暗帧库: 同序列号 + 同曝光 (``cam.get_dark_frame(exposure_ms)``,
   严格匹配, ``allow_nearest=False``) → ``dark_source="camera_library"``,
   暗帧均值存 ``raw/dark_library_<exp>ms.npy``;
2. ``--dark-frames-dir``: 用户已采暗帧 (.npy/.npz, 文件名带 ms 数, 同曝光) →
   ``dark_source="provided"``, 拷贝存 ``raw/dark_provided_<exp>ms.npy``;
3. 都没有 → ``dark_source="missing"``: 其余帧照常采集,
   ``summary.status="ok_dark_missing"``, journal 逐帧记录。
   **绝不伪造暗帧**; 分析端按「不可判定」处理。

设备纪律
--------
* 默认 **plan-only**: 不 import 任何设备驱动、不建目录; ``--dry-run-plan``
  打印完整计划 (含 16 帧 id 顺序) 后干净退出。
* ``--execute`` 才碰硬件: 先 ``DeviceLock.default()`` 等到独占, 再建**全新**
  ``--out`` 目录 (已存在 → 硬失败, 不覆盖历史数据)。
* SLM 只用内存模式 (``video_mode=0``); 连续显示一律经 ``SlotRotator`` 换槽
  (它内部已 sleep, 调用方**不得**再补 sleep)。
* 相位→灰度统一经 :func:`ao_shaping.utils.slm.phase_display.phase_to_slm_grayscale`
  (在线传 ``slm=`` 走驱动管线; 离线 ``slm=None`` 走同一标度解析的纯数学路径)。
* 段间冷却: 矩阵每 4 行一个段边界 (dark 后 / flat_post 后), sleep
  ``--cooldown-s`` (默认 20 s, <20 直接拒绝), 实际等待秒数记入 journal。
* 释放顺序 (finally): ``slm.set_grayscale(0)`` → ``cam.close()`` →
  ``slm.close()`` → ``lock.release()``; ``released.json`` 与 ``release.json``
  (计划 §5 命名) 双写, 内容一致。
* 饱和: 0.9×满量程记 saturated 像素数, **记录但不中止** (本实验允许定位帧
  轻微过曝), journal 标记该帧不得作能量积分基准。

冻结参数 (2026-11-23 用户授权窗口)
----------------------------------
整形相位输入 (哈希对 tobytes 校验)、曝光 0.05 ms、ROI 几何 (``ROI_FROZEN``)
全部冻结写入 ``config.json`` 的 ``roi_frozen`` / ``geometry_source`` 块;
``--self-test`` 断言它们被完整落盘。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from ao_shaping.utils.slm.phase_display import phase_to_slm_grayscale

# ---------------------------------------------------------------------------
# 冻结常量 (预注册; 改动 = 换实验)
# ---------------------------------------------------------------------------
SCRIPT_TAG = "efficiency_match_20261123"
PLAN_SCHEMA = "efficiency-match-capture/v1"
SUMMARY_SCHEMA = "efficiency-match-capture-summary/v1"
JOURNAL_SCHEMA = "efficiency-match-capture-journal/v1"
CONFIG_SCHEMA = "efficiency-match-capture-config/v1"
RECORD_SCHEMA = "ao-optical-path-record/v1"
RELEASE_SCHEMA = "efficiency-match-capture-release/v1"

#: 计划 §4 的 8 行表 (id 基名, 相位类别), 顺序冻结; 走两遍 = 16 帧。
MATRIX_ROWS: tuple[tuple[str, str], ...] = (
    ("flat_pre", "flat"),
    ("blaze_grating", "blaze"),
    ("shaping_target", "shaping"),
    ("dark", "dark"),
    ("flat_mid", "flat"),
    ("blaze_grating", "blaze"),
    ("shaping_target", "shaping"),
    ("flat_post", "flat"),
)

#: 16 帧不是 "每相位 4 帧": §4 原文表循环两轮 (x2 traversal)。
MATRIX_INTERPRETATION = (
    "§4 table原文 ×2 traversal → 16 frames; differs from '4 frames per phase': "
    "flat 6 / blaze 4 / shaping 4 / dark 2"
)

#: SLM / CCD 常量 (issue247 同源; Panel_Res / 波长 / 2π 灰度运行时读回)。
DEFAULT_PANEL_SHAPE = (1200, 1920)  # (rows, cols); execute 时被 Panel_Res 覆盖
NATIVE_CCD_SHAPE = (1944, 2592)  # full native Daheng frame, never windowed
G_REF = 1023  # rad->gray 编码标度 (非物理 2π 灰度, 运行时另读)
TWO_PI = 2.0 * np.pi

#: blaze ramp 参数 (计划 §2.3: 单一连续 ramp, 2π/period, 默认 128 px, panel_x)。
DEFAULT_BLAZE_PERIOD_PX = 128.0
DEFAULT_BLAZE_AXIS = 1  # 1 = panel_x (列方向), 0 = panel_y (行方向)

#: 曝光 / settle / 冷却 / 饱和 (冻结窗口: 2026-11-23 六步探测 0.05 ms 未饱和,
#: 平场峰值 2941/4095, 噪声底 20.9 counts)。
DEFAULT_EXPOSURE_MS = 0.05
DEFAULT_SETTLE_S = 0.8
DEFAULT_COOLDOWN_S = 20.0
MIN_COOLDOWN_S = 20.0
DEFAULT_SATURATION_FRACTION = 0.9
BIT_DEPTH_REQUIRED = 12
FULL_SCALE_REQUIRED = 4095
LOCK_POLL_S = 1.0
DEFAULT_WAIT_IDLE_S = 1800.0

#: 冻结的整形相位输入 (用户 2026-11-23 授权; 哈希对数组 tobytes 验证)。
SHAPING_PHASE_PATH = Path(
    "data/debug/efficiency_match/shaping_target_panel_20261123.npy"
)
SHAPING_PHASE_SHA256 = (
    "f749c3940de902b02d4fa8af6ba708f9834d98741b32fcfed66241a80248116f"
)
SHAPING_PHASE_SOURCE_RUN = (
    "data/debug/differentiable_opt/20261117_022658/phase_final.npy "
    "经 hw_differentiable_phase_opt.phase_model_to_panel 最近邻块映射到面板; "
    "该次运行 focus_compensation_focal_mm = inf (无菲涅尔补偿, 面板相位即纯相位)"
)

#: 冻结 ROI / 几何 (分析端只读这里, 不得自行改)。几何来自 2026-11-17 diffopt
#: 运行实测 (k_mean=7563.905…, rotation 0.1675°, mirror=false, 0 级中心
#: (1104.84, 1224.42), fwhm0=16)。side_px=61.2: 目标边长 60 CCD px ÷ 远场采样
#: 间距 1.9698 px/样本 → 31 样本 → 61.1 px, 取 61.2; separation_seed_px=59.1
#: = 7563.905/128 (周期 128 面板像元的 +1 级预测位置)。
ROI_FROZEN: dict[str, Any] = {
    "r0_px": 48.0,
    "side_px": 61.2,
    "separation_seed_px": 59.1,
    "separation_search_px": 25.0,
    "r1_px": 60.0,
    "zero_order_exclusion_px": 48.0,
    "saturation_fraction": 0.9,
    "blaze_period_panel_px": 128.0,
    "blaze_axis": "panel_x",
    "geometry": {
        "kx": 7551.141568012036,
        "ky": 7576.669164768378,
        "k_mean": 7563.905366390207,
        "rotation_deg": 0.16752399978553,
        "mirror": False,
        "center_cycx": [1104.8427044240864, 1224.4160271278206],
        "fwhm0": 16.0,
        "measured_utc": "20261117_022701",
        "source": "data/debug/differentiable_opt/20261117_022658/run_config.json",
    },
}

#: roi_frozen 顶层键与 geometry 子键 (self-test / pytest 断言完整落盘用)。
ROI_KEYS = (
    "r0_px",
    "side_px",
    "separation_seed_px",
    "separation_search_px",
    "r1_px",
    "zero_order_exclusion_px",
    "saturation_fraction",
    "blaze_period_panel_px",
    "blaze_axis",
    "geometry",
)
GEOMETRY_KEYS = (
    "kx",
    "ky",
    "k_mean",
    "rotation_deg",
    "mirror",
    "center_cycx",
    "fwhm0",
    "measured_utc",
    "source",
)

#: 文件名里的曝光毫秒数 (dark-exp-0.05ms.npz / exp_5ms.npy 等都认)。
_MS_NAME_RE = re.compile(r"(\d+(?:\.\d+)?)\s*ms", re.IGNORECASE)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def digest(array: np.ndarray) -> str:
    """数组内容 (tobytes, C 连续) 的 SHA256 —— 相位/灰度/帧哈希统一口径。"""
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# 相位构造 (纯函数, 无设备)
# ---------------------------------------------------------------------------
def flat_phase(shape: tuple[int, int]) -> np.ndarray:
    """全零弧度相位 (raw 未包裹 —— 唯一 wrap 点在 rad→gray)。"""
    return np.zeros((int(shape[0]), int(shape[1])), dtype=np.float64)


def blaze_phase(
    shape: tuple[int, int],
    *,
    period_px: float = DEFAULT_BLAZE_PERIOD_PX,
    axis: int = DEFAULT_BLAZE_AXIS,
) -> np.ndarray:
    """单一连续 blaze ramp: 每 ``period_px`` 面板像元一个完整 2π。

    沿 ``axis`` (1 = panel_x 列方向, 0 = panel_y 行方向) 线性增长, 另一轴
    恒为 0 —— 连续 ramp 把能量集中到 +1 级 (计划 §2.3), 不是二值 0/π 光栅。
    相位 raw 未包裹 (整数倍 2π 的折叠由 rad→gray 的 mod 完成)。
    """
    if axis not in (0, 1):
        raise ValueError("axis must be 0 or 1")
    if period_px <= 0:
        raise ValueError("period_px must be > 0")
    rows, cols = int(shape[0]), int(shape[1])
    n = cols if axis == 1 else rows
    ramp = TWO_PI * np.arange(n, dtype=np.float64) / float(period_px)
    phase = np.zeros((rows, cols), dtype=np.float64)
    if axis == 1:
        phase[:, :] = ramp[np.newaxis, :]
    else:
        phase[:, :] = ramp[:, np.newaxis]
    return phase


def validate_shaping_phase(
    arr: np.ndarray, panel_shape: tuple[int, int], source: str
) -> np.ndarray:
    """整形相位的纯校验: shape 必须等于面板 Panel_Res, 浮点弧度。

    返回 float32 视图; 违规抛 ``ValueError``, 信息里同时给出文件 shape 与
    面板 shape (两个都报, 便于排障)。
    """
    if arr.ndim != 2 or tuple(arr.shape) != tuple(panel_shape):
        raise ValueError(
            f"整形相位 shape {tuple(arr.shape)} != 面板 Panel_Res "
            f"{tuple(panel_shape)} (source={source})"
        )
    if not np.issubdtype(arr.dtype, np.floating):
        raise ValueError(
            f"整形相位必须是浮点弧度数组, 实际 dtype={arr.dtype} (source={source})"
        )
    return arr.astype(np.float32, copy=False)


def load_shaping_phase(path: Path, panel_shape: tuple[int, int]) -> np.ndarray:
    """读整形目标相位 (.npy, 面板分辨率弧度); 文件缺失/shape 不符即硬失败。"""
    if not path.is_file():
        raise FileNotFoundError(
            f"整形相位文件不存在: {path} (用 --shaping-phase 指定面板分辨率 .npy)"
        )
    arr = np.asarray(np.load(path))
    return validate_shaping_phase(arr, panel_shape, str(path))


def phase_gray_offline(phase: np.ndarray, max_gray: int = G_REF) -> np.ndarray:
    """离线 (plan/self-test) 弧度→灰度: 与在线同一标度解析入口。

    在线执行用 ``phase_to_slm_grayscale(phase, slm=slm)`` 走驱动管线
    (波前矫正 + LUT + 平移); 这里 ``slm=None`` 走纯数学 fallback ——
    与驱动路径共用同一个标度解析函数 (见 phase_display 契约)。
    """
    return phase_to_slm_grayscale(
        np.asarray(phase, dtype=np.float64), max_grayscale=int(max_gray)
    )


# ---------------------------------------------------------------------------
# 冻结矩阵 (纯函数)
# ---------------------------------------------------------------------------
def matrix_purpose(base_id: str) -> str:
    """§4 表「目的」列的机器可读文本。"""
    return {
        "flat_pre": "0 级基准与漂移前参考",
        "blaze_grating": "+1 级衍射效率",
        "shaping_target": "当前光路远场整形效率",
        "dark": "同曝光暗帧",
        "flat_mid": "中途功率/光路漂移检查",
        "flat_post": "末端 0 级基准",
    }[base_id]


def build_matrix(rounds: int = 2) -> list[dict[str, Any]]:
    """§4 的 8 行 × ``rounds`` 轮 → 逐帧条目 (id / phase_id / 轮 / 行)。

    第二轮起 id 加 ``_r2`` 后缀; 返回顺序即采集顺序, 冻结不可重排。
    """
    frames: list[dict[str, Any]] = []
    for round_no in range(1, int(rounds) + 1):
        suffix = "" if round_no == 1 else f"_r{round_no}"
        for row_no, (base_id, phase_id) in enumerate(MATRIX_ROWS, start=1):
            frames.append(
                dict(
                    id=f"{base_id}{suffix}",
                    phase_id=phase_id,
                    round=round_no,
                    row=row_no,
                    purpose=matrix_purpose(base_id),
                )
            )
    return frames


def build_config(
    args: argparse.Namespace,
    *,
    panel_shape: tuple[int, int],
    shaping_phase_path: Path,
    shaping_phase_sha: str,
    shaping_phase_shape: tuple[int, int] | None,
    device: dict[str, Any] | None,
    dark_source: str,
    out_dir: Path,
) -> dict[str, Any]:
    """config.json 的内容 (设备字段运行时读回, plan-only 时为 null/空)。"""
    return {
        "schema": CONFIG_SCHEMA,
        "script": SCRIPT_TAG,
        "matrix_interpretation": MATRIX_INTERPRETATION,
        "plan": {
            "schema": PLAN_SCHEMA,
            "matrix": [f["id"] for f in build_matrix()],
            "matrix_order": [list(r) for r in MATRIX_ROWS],
            "n_frames": len(build_matrix()),
        },
        "exposure": {
            "requested_ms": float(args.exposure_ms),
            "readback_ms": None,
            "all_frames_same_exposure": True,
        },
        "settle_s": float(args.settle_s),
        "cooldown_s": float(args.cooldown_s),
        "saturation_fraction": float(args.saturation_fraction),
        "frames_per_phase": dict(flat=6, blaze=4, shaping=4, dark=2),
        "blaze": {
            "period_px": float(args.blaze_period_px),
            "axis": int(args.blaze_axis),
            "axis_name": "panel_x" if int(args.blaze_axis) == 1 else "panel_y",
            "kind": "continuous_ramp_not_binary",
        },
        "shaping_phase": {
            "path": str(shaping_phase_path),
            "sha256": shaping_phase_sha,
            "panel_shape": (
                list(shaping_phase_shape) if shaping_phase_shape is not None else None
            ),
            "source_run": SHAPING_PHASE_SOURCE_RUN,
            "compensation_focal_mm": None,
            "compensation_note": "inf / 无菲涅尔补偿 (面板相位即纯相位)",
        },
        "panel_shape": list(panel_shape),
        "device": device or {},
        "dark_source": dark_source,
        "out_dir": str(out_dir),
        "roi_frozen": dict(ROI_FROZEN),
        "geometry_source": dict(ROI_FROZEN["geometry"]),
        "timestamp_utc": utc_now(),
    }


def build_plan(args: argparse.Namespace, shaping_phase_path: Path) -> dict[str, Any]:
    """plan.json 骨架 (execute 前写盘; 设备/曝光回读在 config.json 补齐)。"""
    shaping_phase_sha = ""
    shaping_phase_shape: tuple[int, int] | None = None
    if shaping_phase_path.is_file():
        arr = np.load(shaping_phase_path)
        shaping_phase_sha = digest(arr)
        shaping_phase_shape = tuple(int(v) for v in arr.shape)
    cfg = build_config(
        args,
        panel_shape=DEFAULT_PANEL_SHAPE,
        shaping_phase_path=shaping_phase_path,
        shaping_phase_sha=shaping_phase_sha,
        shaping_phase_shape=shaping_phase_shape,
        device=None,
        dark_source="pending",
        out_dir=Path(str(args.out or "")),
    )
    return {
        "schema": PLAN_SCHEMA,
        "matrix_interpretation": MATRIX_INTERPRETATION,
        "matrix": [
            dict(f, phase_hash=None, exposure_ms=None) for f in build_matrix()
        ],
        "config": cfg,
    }


def final_status(dark_source: str) -> str:
    """结束状态: 暗帧缺失是显式降级, 不是失败也不是成功。"""
    return "ok_dark_missing" if dark_source == "missing" else "ok"


def build_summary(
    *,
    frame_records: list[dict[str, Any]],
    saturation_count: int,
    peak_counts: dict[str, float],
    dark_source: str,
    elapsed_s: float,
) -> dict[str, Any]:
    """summary.json: 状态 / 帧数 / 饱和计数 / 峰值 / 耗时 / 暗帧来源。"""
    return {
        "schema": SUMMARY_SCHEMA,
        "status": final_status(dark_source),
        "n_frames": len(frame_records),
        "saturation_count": int(saturation_count),
        "peak_counts": dict(peak_counts),
        "elapsed_s": round(float(elapsed_s), 3),
        "dark_source": dark_source,
        "matrix_interpretation": MATRIX_INTERPRETATION,
        "timestamp_utc": utc_now(),
    }


# ---------------------------------------------------------------------------
# 暗帧纪律 (诚实优先; 库 → 用户提供 → missing, 绝不伪造)
# ---------------------------------------------------------------------------
def exposure_from_name(stem: str) -> float | None:
    """从文件名提取曝光毫秒数; 提不出就返回 None (不猜)。"""
    m = _MS_NAME_RE.search(stem)
    if m:
        return float(m.group(1))
    for token in re.split(r"[-_ .]+", stem):
        try:
            return float(token)
        except ValueError:
            continue
    return None


def load_dark_array(path: Path) -> np.ndarray:
    """读一张暗帧 (.npy 裸数组 / .npz 含 ``mean``); 非二维即报错。"""
    if path.suffix.lower() == ".npy":
        arr = np.asarray(np.load(path))
    else:
        data = np.load(path)
        key = "mean" if "mean" in data.files else data.files[0]
        arr = np.asarray(data[key])
    if arr.ndim != 2:
        raise ValueError(f"暗帧必须是二维帧数组: {path} -> {arr.shape}")
    return arr.astype(np.float32, copy=False)


def resolve_dark_from_dir(
    dark_dir: Path, exposure_ms: float, *, rel_tol: float = 0.02
) -> dict[str, Any] | None:
    """用户提供的暗帧目录: 文件名 ms 数与读回曝光匹配 (tol 内) 才用。

    无同曝光条目时返回 ``None`` —— 不隐式跨曝光套用 (跨曝光暗帧带正比偏置)。
    """
    if not dark_dir.is_dir():
        return None
    target = float(exposure_ms)
    tol = abs(target) * float(rel_tol)
    for path in sorted(dark_dir.iterdir()):
        if path.suffix.lower() not in (".npy", ".npz") or not path.is_file():
            continue
        value = exposure_from_name(path.stem)
        if value is None:
            logger.debug("暗帧文件名里提不出曝光数, 跳过: {}", path.name)
            continue
        if abs(value - target) <= tol:
            return dict(
                source="provided",
                path=str(path),
                exposure_ms=value,
                mean=load_dark_array(path),
            )
    return None


# ---------------------------------------------------------------------------
# 释放证据 (released.json + release.json 双写, 内容一致)
# ---------------------------------------------------------------------------
def write_release_files(out: Path, release: dict[str, Any]) -> list[Path]:
    """写 ``released.json`` (issue247 命名) 与 ``release.json`` (计划 §5 命名)。

    计划 §5 要 ``release.json``, 历史脚本用 ``released.json`` —— 两个都写,
    内容一致, 避免与任一命名约定不符。
    """
    release = dict(release)
    release["schema"] = RELEASE_SCHEMA
    paths = [out / "released.json", out / "release.json"]
    for path in paths:
        write_json(path, release)
    return paths


# ---------------------------------------------------------------------------
# 离线 self-test (无设备、无目录创建、不 import 驱动)
# ---------------------------------------------------------------------------
def run_self_test() -> int:
    """断言冻结矩阵 / 相位 / 哈希 / 暗帧语义 / roi_frozen 落盘。"""
    ns = argparse.Namespace(
        exposure_ms=DEFAULT_EXPOSURE_MS,
        settle_s=DEFAULT_SETTLE_S,
        cooldown_s=DEFAULT_COOLDOWN_S,
        saturation_fraction=DEFAULT_SATURATION_FRACTION,
        blaze_period_px=DEFAULT_BLAZE_PERIOD_PX,
        blaze_axis=DEFAULT_BLAZE_AXIS,
        out=None,
    )

    # 1) 16 帧矩阵: 顺序、id、轮次、每相位计数
    frames = build_matrix()
    ids = [f["id"] for f in frames]
    expected = [
        "flat_pre",
        "blaze_grating",
        "shaping_target",
        "dark",
        "flat_mid",
        "blaze_grating",
        "shaping_target",
        "flat_post",
        "flat_pre_r2",
        "blaze_grating_r2",
        "shaping_target_r2",
        "dark_r2",
        "flat_mid_r2",
        "blaze_grating_r2",
        "shaping_target_r2",
        "flat_post_r2",
    ]
    assert ids == expected, ids
    counts = dict(flat=0, blaze=0, shaping=0, dark=0)
    for f in frames:
        counts[f["phase_id"]] += 1
    assert counts == dict(flat=6, blaze=4, shaping=4, dark=2), counts
    assert [f["round"] for f in frames[:8]] == [1] * 8
    assert [f["round"] for f in frames[8:]] == [2] * 8
    assert all(1 <= f["row"] <= 8 for f in frames), frames

    # 2) blaze 相位: 只沿声明轴变化, 连续 ramp 2π/period, 同参数哈希稳定
    phase = blaze_phase(DEFAULT_PANEL_SHAPE, period_px=128.0, axis=1)
    assert phase.shape == DEFAULT_PANEL_SHAPE
    assert np.allclose(
        phase[0, 128:] - phase[0, :-128], TWO_PI, rtol=0.0, atol=1e-9
    ), "周期 128 不闭合 (raw 连续 ramp 每周期增 2π)"
    assert abs(float(phase[0, 128]) - TWO_PI) < 1e-9, "单周期增量不是 2π"
    assert float(np.ptp(phase[:, 0])) == 0.0, "panel_x ramp 不应沿行变化"
    phase_y = blaze_phase(DEFAULT_PANEL_SHAPE, period_px=128.0, axis=0)
    assert float(np.ptp(phase_y[0, :])) == 0.0, "panel_y ramp 不应沿列变化"
    assert digest(phase) == digest(
        blaze_phase(DEFAULT_PANEL_SHAPE, period_px=128.0, axis=1)
    )
    assert digest(phase) != digest(
        blaze_phase(DEFAULT_PANEL_SHAPE, period_px=192.0, axis=1)
    )
    gray = phase_gray_offline(phase)
    assert gray.dtype == np.uint16 and int(gray.max()) <= G_REF
    assert np.array_equal(gray[3, :], gray[700, :]), "灰度行间应一致"

    # 3) 整形相位 shape/dtype 校验失败路径 (纯函数, 不落盘)
    try:
        load_shaping_phase(Path("efficiency_selftest_missing.npy"), DEFAULT_PANEL_SHAPE)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("缺失整形相位文件必须抛 FileNotFoundError")
    try:
        validate_shaping_phase(
            np.zeros((600, 800), dtype=np.float32),
            DEFAULT_PANEL_SHAPE,
            "selftest-bad-shape",
        )
    except ValueError as exc:
        assert "(600, 800)" in str(exc) and "(1200, 1920)" in str(exc), str(exc)
    else:
        raise AssertionError("错误 shape 必须抛 ValueError 并同时报告两个 shape")
    try:
        validate_shaping_phase(
            np.zeros(DEFAULT_PANEL_SHAPE, dtype=np.uint8),
            DEFAULT_PANEL_SHAPE,
            "selftest-bad-dtype",
        )
    except ValueError as exc:
        assert "dtype" in str(exc), str(exc)
    else:
        raise AssertionError("非浮点 dtype 必须抛 ValueError")
    ok = validate_shaping_phase(
        np.zeros(DEFAULT_PANEL_SHAPE, dtype=np.float64), DEFAULT_PANEL_SHAPE, "ok"
    )
    assert ok.dtype == np.float32

    # 4) 哈希稳定: 同内容同哈希; 内容或 shape 变化 → 不同哈希
    same = np.zeros((4, 4), dtype=np.float32)
    assert digest(same) == digest(np.zeros((4, 4), dtype=np.float32))
    assert digest(same) != digest(np.ones((4, 4), dtype=np.float32))
    assert digest(same) != digest(np.zeros((4, 5), dtype=np.float32))
    # 冻结输入文件在场时, 哈希必须与预注册值一致 (本地 data/ 不入库则跳过)。
    if SHAPING_PHASE_PATH.is_file():
        assert digest(np.load(SHAPING_PHASE_PATH)) == SHAPING_PHASE_SHA256

    # 5) dark-missing 语义: 状态显式降级, 帧数不受影响
    assert final_status("missing") == "ok_dark_missing"
    assert final_status("camera_library") == "ok"
    fake_records = [dict(id=f["id"], phase_id=f["phase_id"]) for f in frames]
    summary = build_summary(
        frame_records=fake_records,
        saturation_count=0,
        peak_counts={},
        dark_source="missing",
        elapsed_s=1.0,
    )
    assert summary["status"] == "ok_dark_missing"
    assert summary["n_frames"] == 16
    assert summary["dark_source"] == "missing"
    assert summary["matrix_interpretation"] == MATRIX_INTERPRETATION

    # 6) roi_frozen / geometry_source 完整落盘 (config.json 的冻结块)
    cfg = build_config(
        ns,
        panel_shape=DEFAULT_PANEL_SHAPE,
        shaping_phase_path=SHAPING_PHASE_PATH,
        shaping_phase_sha=SHAPING_PHASE_SHA256,
        shaping_phase_shape=DEFAULT_PANEL_SHAPE,
        device=None,
        dark_source="pending",
        out_dir=Path("plan_only"),
    )
    roi = cfg["roi_frozen"]
    for key in ROI_KEYS:
        assert key in roi, key
    for key in GEOMETRY_KEYS:
        assert key in roi["geometry"], key
    assert roi["r0_px"] == 48.0
    assert roi["side_px"] == 61.2
    assert roi["separation_seed_px"] == 59.1
    assert roi["separation_search_px"] == 25.0
    assert roi["r1_px"] == 60.0
    assert roi["zero_order_exclusion_px"] == 48.0
    assert roi["saturation_fraction"] == 0.9
    assert roi["blaze_period_panel_px"] == 128.0
    assert roi["blaze_axis"] == "panel_x"
    geo = roi["geometry"]
    assert geo["kx"] == 7551.141568012036
    assert geo["ky"] == 7576.669164768378
    assert geo["k_mean"] == 7563.905366390207
    assert geo["rotation_deg"] == 0.16752399978553
    assert geo["mirror"] is False
    assert geo["center_cycx"] == [1104.8427044240864, 1224.4160271278206]
    assert geo["fwhm0"] == 16.0
    assert geo["measured_utc"] == "20261117_022701"
    assert geo["source"] == "data/debug/differentiable_opt/20261117_022658/run_config.json"
    geo_src = cfg["geometry_source"]
    for key in GEOMETRY_KEYS:
        assert key in geo_src, key
    assert geo_src["k_mean"] == geo["k_mean"]
    assert geo_src["source"] == geo["source"]
    assert cfg["shaping_phase"]["sha256"] == SHAPING_PHASE_SHA256
    assert cfg["shaping_phase"]["compensation_focal_mm"] is None
    assert "无菲涅尔补偿" in cfg["shaping_phase"]["compensation_note"]

    # 7) plan: 16 帧 id 顺序 + matrix_interpretation 显式存在
    plan = build_plan(ns, SHAPING_PHASE_PATH)
    assert [f["id"] for f in plan["matrix"]] == expected
    assert plan["matrix_interpretation"] == MATRIX_INTERPRETATION
    assert "×2 traversal" in plan["matrix_interpretation"]
    if SHAPING_PHASE_PATH.is_file():
        assert plan["config"]["shaping_phase"]["sha256"] == SHAPING_PHASE_SHA256

    logger.info("efficiency match capture self-test: all invariants passed")
    return 0


# ---------------------------------------------------------------------------
# 硬件 helpers (import 只出现在 execute 路径内)
# ---------------------------------------------------------------------------
def acquire_device_lock(wait_s: float) -> Any | None:
    """Acquire ``DeviceLock.default()``, polling for up to ``wait_s`` seconds."""
    from ao_shaping.tools.kernels.device_mutex import DeviceLock

    lock = DeviceLock.default()
    deadline = time.monotonic() + max(0.0, float(wait_s))
    reported = False
    while True:
        if lock.acquire(timeout=0.0):
            return lock
        if not reported:
            holder = lock.holder()
            logger.warning(
                "bench device lock occupied by {}; waiting up to {:.1f}s",
                holder if holder else "unknown",
                wait_s,
            )
            reported = True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        time.sleep(min(LOCK_POLL_S, remaining))


def read_camera_pixel_size(cam: Any) -> float | None:
    """像元尺寸只信驱动回读, 绝不硬编码。"""
    value = getattr(cam, "pixel_size_um", None)
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def camera_model_name(cam: Any) -> str | None:
    """相机型号 (驱动回读; 拿不到返回 None, 不猜)。"""
    value = getattr(cam, "model_name", None)
    if callable(value):
        try:
            value = value()
        except Exception as exc:
            logger.debug("读取相机型号失败: {}", exc)
            return None
    return value if isinstance(value, str) and value else None


def frame_stats(
    frame: np.ndarray,
    full_scale: float,
    saturation_fraction: float = DEFAULT_SATURATION_FRACTION,
) -> dict[str, Any]:
    """饱和统计: 计数记录, 不中止 (本实验允许定位帧轻微过曝)。"""
    threshold = float(saturation_fraction) * float(full_scale)
    return {
        "peak": float(np.max(frame)),
        "saturated_pixels": int(np.count_nonzero(frame >= threshold)),
        "saturation_threshold": float(threshold),
    }


# ---------------------------------------------------------------------------
# Hardware execution
# ---------------------------------------------------------------------------
def execute(args: argparse.Namespace, plan: dict[str, Any]) -> int:
    out: Path = args.out
    lock = acquire_device_lock(args.wait_until_idle)
    if lock is None:
        logger.error("could not acquire bench device lock within {}s", args.wait_until_idle)
        return 2

    status = "not-started"
    error: str | None = None
    exit_code = 1
    created_out = False
    slm: Any = None
    cam: Any = None
    frame_records: list[dict[str, Any]] = []
    dark_source = "missing"
    acquired_at = utc_now()
    t_start = time.monotonic()

    release: dict[str, Any] = {
        "slm_flat_gray0": None,
        "slm_closed": False,
        "camera_closed": False,
        "lock_released": False,
        "closed_normally": False,
        "acquired_at_utc": acquired_at,
        "error": None,
    }

    try:
        out.mkdir(parents=True, exist_ok=False)
        created_out = True
        (out / "raw").mkdir(exist_ok=True)
        (out / "phases").mkdir(exist_ok=True)
        write_json(out / "plan.json", plan)
        journal_path = out / "journal.jsonl"

        # 驱动 import 只出现在这里: 锁已持有且 --out 是全新目录。
        from ao_shaping.drivers.ccd.common import (
            create_camera,
            get_camera_exposure_ms,
            set_camera_exposure_ms,
        )
        from ao_shaping.drivers.slm.factory import require_drive_board_entry
        from ao_shaping.drivers.slm.santec import Santec, SlotRotator
        from ao_shaping.tools.slm.transfer.hardware import prepare_camera_sdk
        from ao_shaping.utils.io.optical_path_record import build_optical_path_record

        prepare_camera_sdk("daheng")
        slm = Santec(slm_number=args.slm_number, video_mode=0)
        slm.open()
        require_drive_board_entry(slm)
        panel_shape = (int(slm.Panel_Res[1]), int(slm.Panel_Res[0]))
        wavelength_nm, two_pi_gray = slm.get_wavelength_info()
        slm.set_phase_overlay(None)
        slm_serial = getattr(slm, "serial_number", None)

        cam = create_camera(
            "daheng",
            args.cam_id,
            exposure_time_ms=float(args.exposure_ms),
            bit_depth=BIT_DEPTH_REQUIRED,
        )
        cam.open()
        effective_bits = getattr(cam, "bit_depth", None)
        if effective_bits != BIT_DEPTH_REQUIRED:
            raise RuntimeError(
                f"Mono12 bit-depth readback is {effective_bits!r}, expected "
                f"{BIT_DEPTH_REQUIRED}"
            )
        requested = float(np.clip(float(args.exposure_ms), 0.001, 1000.0))
        set_camera_exposure_ms(cam, requested)
        exposure_ms = float(get_camera_exposure_ms(cam))
        cam_serial = getattr(cam, "serial_number", None) or getattr(cam, "sn", None)
        pixel_um = read_camera_pixel_size(cam)
        cam_model = camera_model_name(cam)

        # ---- 相位与灰度 (flat 走 np.full raw 路径; 其余走驱动管线) ----
        flat = flat_phase(panel_shape)
        blaze = blaze_phase(
            panel_shape, period_px=float(args.blaze_period_px), axis=int(args.blaze_axis)
        )
        shaping_path = Path(args.shaping_phase)
        shaping = load_shaping_phase(shaping_path, panel_shape)
        shaping_sha = digest(shaping)

        flat_gray = np.full(panel_shape, 0, dtype=np.uint16)
        blaze_gray = phase_to_slm_grayscale(blaze, slm=slm)
        shaping_gray = phase_to_slm_grayscale(shaping, slm=slm)

        np.save(out / "phases" / "flat_phase.npy", flat)
        np.save(out / "phases" / "flat_gray.npy", flat_gray)
        np.save(out / "phases" / "blaze_phase.npy", blaze)
        np.save(out / "phases" / "blaze_gray.npy", blaze_gray)
        np.save(out / "phases" / "shaping_phase.npy", shaping)
        np.save(out / "phases" / "shaping_gray.npy", shaping_gray)
        write_json(
            out / "phases" / "hashes.json",
            {
                "flat_phase": digest(flat),
                "flat_gray": digest(flat_gray),
                "blaze_phase": digest(blaze),
                "blaze_gray": digest(blaze_gray),
                "shaping_phase": shaping_sha,
                "shaping_gray": digest(shaping_gray),
            },
        )

        # ---- 暗帧解析 (库 → 用户提供 → missing; 形状不符也算 missing) ----
        dark_frame: np.ndarray | None = None
        dark_note: str | None = None
        dark_exposure_ms: float | None = None
        try:
            dark_lib = cam.get_dark_frame(
                float(exposure_ms), rel_tol=0.02, allow_nearest=False
            )
        except Exception as exc:
            dark_lib = None
            dark_note = f"相机暗帧库读取失败: {type(exc).__name__}: {exc}"
            logger.warning(dark_note)
        if dark_lib is not None:
            mean = np.asarray(dark_lib.mean, dtype=np.float32)
            if tuple(mean.shape) != NATIVE_CCD_SHAPE:
                dark_note = (
                    f"暗帧库均值 shape {tuple(mean.shape)} != 相机全幅 "
                    f"{NATIVE_CCD_SHAPE}, 不能作本次暗帧"
                )
                logger.warning(dark_note)
            else:
                dark_frame = mean
                dark_exposure_ms = float(dark_lib.exposure_ms)
                dark_source = "camera_library"
                np.save(out / "raw" / f"dark_library_{exposure_ms:g}ms.npy", mean)
        if dark_frame is None and args.dark_frames_dir is not None:
            provided = resolve_dark_from_dir(Path(args.dark_frames_dir), exposure_ms)
            if provided is None:
                dark_note = (
                    f"--dark-frames-dir 内没有与读回曝光 {exposure_ms} ms 匹配的暗帧"
                )
            elif tuple(np.shape(provided["mean"])) != NATIVE_CCD_SHAPE:
                dark_note = (
                    f"提供暗帧 shape {tuple(np.shape(provided['mean']))} != 相机全幅 "
                    f"{NATIVE_CCD_SHAPE}"
                )
                logger.warning(dark_note)
            else:
                dark_frame = np.asarray(provided["mean"], dtype=np.float32)
                dark_exposure_ms = float(provided["exposure_ms"])
                dark_source = "provided"
                np.save(out / "raw" / f"dark_provided_{exposure_ms:g}ms.npy", dark_frame)
        if dark_source == "missing":
            reason = dark_note or "相机暗帧库与 --dark-frames-dir 都没有同曝光暗帧"
            logger.warning(
                "暗帧缺失 (dark_source=missing): {}; 其余帧照常采集, "
                "分析端按「不可判定」处理, 绝不伪造暗帧",
                reason,
            )

        device = {
            "slm": {
                "model": "Santec SLM-200",
                "slm_number": int(args.slm_number),
                "serial": slm_serial,
                "wavelength_nm": int(wavelength_nm),
                "two_pi_grayscale": two_pi_gray,
                "panel_res": list(panel_shape),
                "video_mode": 0,
                "encoding_max_gray": G_REF,
            },
            "camera": {
                "model": cam_model or "Daheng MER2-507-23GM",
                "serial": cam_serial,
                "bit_depth": int(effective_bits),
                "full_scale": int(FULL_SCALE_REQUIRED),
                "pixel_size_um": pixel_um,
                "native_shape": list(NATIVE_CCD_SHAPE),
            },
        }
        config = build_config(
            args,
            panel_shape=panel_shape,
            shaping_phase_path=shaping_path,
            shaping_phase_sha=shaping_sha,
            shaping_phase_shape=tuple(shaping.shape),
            device=device,
            dark_source=dark_source,
            out_dir=out,
        )
        config["exposure"]["readback_ms"] = exposure_ms
        write_json(out / "config.json", config)

        # ---- 采集矩阵 (SlotRotator 换槽并自带等待; 调用方不再 sleep) ----
        rotator = SlotRotator(slm)
        frame_index = 0
        saturation_count = 0
        peak_counts: dict[str, float] = {}
        cooldown_s = float(args.cooldown_s)
        gray_arrays = dict(flat=flat_gray, blaze=blaze_gray, shaping=shaping_gray)
        phase_hashes = {
            "flat": digest(flat),
            "blaze": digest(blaze),
            "shaping": shaping_sha,
        }

        def grab() -> np.ndarray:
            frame = np.asarray(cam.get_numpy_image(1, True))
            if frame.dtype != np.uint16 or frame.shape != NATIVE_CCD_SHAPE:
                raise RuntimeError(
                    f"expected uint16 frame {NATIVE_CCD_SHAPE}, got "
                    f"{frame.dtype} {frame.shape}"
                )
            return frame.copy()

        def light_row(frame_id: str, phase_id: str) -> dict[str, Any]:
            nonlocal saturation_count
            gray = gray_arrays[phase_id]
            write_return = rotator.display_data(slm, gray, float(args.settle_s))
            frame = grab()
            full_scale = float(cam.effective_full_scale(frame))
            if full_scale != float(FULL_SCALE_REQUIRED):
                raise RuntimeError(
                    f"Mono12 full scale readback {full_scale}, expected "
                    f"{FULL_SCALE_REQUIRED}"
                )
            stats = frame_stats(frame, full_scale, args.saturation_fraction)
            np.save(out / "raw" / f"{frame_index:04d}_{frame_id}.npy", frame)
            saturated = int(stats["saturated_pixels"]) > 0
            if saturated:
                saturation_count += 1
                logger.warning(
                    "帧 {} 有 {} 个像素 >= {:.0f} (饱和门限); 记录并继续采集, "
                    "该帧不得作能量积分基准",
                    frame_id,
                    stats["saturated_pixels"],
                    stats["saturation_threshold"],
                )
            peak_counts[frame_id] = stats["peak"]
            return dict(
                schema=JOURNAL_SCHEMA,
                id=frame_id,
                phase_id=phase_id,
                frame_index=frame_index,
                timestamp_utc=utc_now(),
                exposure_ms=exposure_ms,
                slm_serial=slm_serial,
                slm_write_slot=getattr(rotator, "last_slot", None),
                slm_write_return=write_return,
                phase_hash=phase_hashes[phase_id],
                gray_range=[int(gray.min()), int(gray.max())],
                saturated_pixels=int(stats["saturated_pixels"]),
                usable_for_energy=not saturated,
                dark_source=dark_source,
                laser_power_w=None,
                laser_power_note="激光功率计无读数通道; 漂移由 flat 帧监测",
            )

        def dark_row(frame_id: str) -> dict[str, Any]:
            dark_rel = None
            if dark_frame is not None:
                if dark_source == "camera_library":
                    dark_rel = f"raw/dark_library_{exposure_ms:g}ms.npy"
                else:
                    dark_rel = f"raw/dark_provided_{exposure_ms:g}ms.npy"
            return dict(
                schema=JOURNAL_SCHEMA,
                id=frame_id,
                phase_id="dark",
                frame_index=frame_index,
                timestamp_utc=utc_now(),
                exposure_ms=exposure_ms,
                slm_serial=slm_serial,
                slm_write_slot=None,
                slm_write_return=None,
                phase_hash=None,
                gray_range=None,
                saturated_pixels=None,
                usable_for_energy=False,
                dark_source=dark_source,
                dark_file=dark_rel,
                dark_exposure_ms=dark_exposure_ms,
                dark_note=(
                    None
                    if dark_frame is not None
                    else (dark_note or "无同曝光暗帧; 本帧位无数据, 不伪造")
                ),
                laser_power_w=None,
                laser_power_note="激光功率计无读数通道; 漂移由 flat 帧监测",
            )

        # 预热: flat 显示 + 丢 3 帧, 不入 journal (SlotRotator 已等待)。
        rotator.display_data(slm, flat_gray, float(args.settle_s))
        for _ in range(3):
            grab()

        matrix = build_matrix()
        for item in matrix:
            if item["phase_id"] == "dark":
                row = dark_row(item["id"])
            else:
                row = light_row(item["id"], item["phase_id"])
            append_jsonl(journal_path, row)
            frame_records.append(row)
            frame_index += 1

            # 段间冷却: 每 4 行一个段边界 (dark 后 / flat_post 后);
            # 最后一帧之后不再冷却。实际等待秒数记入 journal。
            if item["row"] % 4 == 0 and item is not matrix[-1]:
                t0 = time.monotonic()
                time.sleep(cooldown_s)
                append_jsonl(
                    journal_path,
                    dict(
                        event="cooldown",
                        id=f"cooldown_after_{item['id']}",
                        requested_s=cooldown_s,
                        actual_s=round(time.monotonic() - t0, 3),
                        timestamp_utc=utc_now(),
                    ),
                )

        write_json(
            out / "summary.json",
            build_summary(
                frame_records=frame_records,
                saturation_count=saturation_count,
                peak_counts=peak_counts,
                dark_source=dark_source,
                elapsed_s=time.monotonic() - t_start,
            ),
        )

        # ---- optical_path_record (issue247 同款描述性结构) ----
        reasons: dict[str, str] = {}
        if pixel_um is None:
            reasons["devices.camera.pixel_size_um"] = (
                "camera driver cam.pixel_size_um unavailable; not hardcoded"
            )
        record = build_optical_path_record(
            run_id=f"{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_efficiency_match",
            date=f"{datetime.now(timezone.utc):%Y-%m-%d}",
            operator_note=(
                "efficiency match capture: 16-frame frozen matrix; descriptive "
                "raw capture only, analysis is a separate offline step"
            ),
            devices={
                "slm": {
                    "model": "Santec SLM-200",
                    "slm_number": int(args.slm_number),
                    "wavelength_nm": int(wavelength_nm),
                    "two_pi_grayscale": two_pi_gray,
                    "panel_res": list(panel_shape),
                    "open_mode": "memory",
                },
                "camera": {
                    "model": cam_model or "Daheng MER2-507-23GM",
                    "serial": cam_serial,
                    "bit_depth": int(effective_bits),
                    "full_scale": int(FULL_SCALE_REQUIRED),
                    "exposure_used_ms": exposure_ms,
                    "averaging_frames": 1,
                    "discard_frames_after_exposure_change": 3,
                    "pixel_size_um": pixel_um,
                },
            },
            optical_path={
                "probe_status": (
                    "efficiency-match-capture (descriptive; offline analysis pending)"
                ),
            },
            exposure_check={
                "peak_counts_at_used_exposure": (
                    peak_counts.get("flat_pre") if peak_counts else None
                ),
                "saturation_threshold_counts": float(
                    DEFAULT_SATURATION_FRACTION * FULL_SCALE_REQUIRED
                ),
                "saturated": saturation_count > 0,
                "settle_timeout": False,
            },
            occupancy={
                "acquired_at": acquired_at,
                "closed_normally": True,
            },
            reasons=reasons,
        )
        write_json(out / "optical_path_record.json", record)

        status = final_status(dark_source)
        exit_code = 0
    except FileExistsError:
        status = "out_exists"
        error = f"--out already exists: {out}"
        logger.error(error)
        exit_code = 2
    except Exception as exc:
        status = "error"
        error = f"{type(exc).__name__}: {exc}"
        logger.error(error)
        exit_code = 1
    finally:
        # 每条路径都释放, 顺序冻结: set_grayscale(0) → cam.close →
        # slm.close → lock.release。
        if slm is not None:
            try:
                slm.set_grayscale(0)
                release["slm_flat_gray0"] = "ok"
            except Exception as exc:
                release["slm_flat_gray0"] = f"failed: {exc}"
        if cam is not None:
            try:
                cam.close()
                release["camera_closed"] = True
            except Exception as exc:
                release["camera_closed"] = False
                logger.error("camera close failed: {}", exc)
        if slm is not None:
            try:
                slm.close()
                release["slm_closed"] = True
            except Exception as exc:
                release["slm_closed"] = False
                logger.error("slm close failed: {}", exc)
        if lock is not None:
            try:
                lock.release()
                release["lock_released"] = True
            except Exception as exc:
                release["lock_released"] = False
                logger.error("device lock release failed: {}", exc)
        release["error"] = error
        release["closed_normally"] = (
            release["slm_closed"] is True
            and release["camera_closed"] is True
            and release["lock_released"] is True
            and error is None
        )
        release["released_at_utc"] = utc_now()
        if created_out:
            # released.json (issue247 命名) + release.json (计划 §5 命名) 双写。
            # out 不是本运行新建的 (out_exists) 就绝不写, 避免污染别人的目录。
            write_release_files(out, release)
        else:
            logger.warning(
                "--out 未由本运行创建, 释放证据只在日志里: {}", out
            )

    logger.info("status={} exit_code={}", status, exit_code)
    return exit_code


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Efficiency-match capture (plan-only by default): the frozen 16-frame "
            "matrix from docs/dev/efficiency_match_experiment_plan_20261121.md §4 "
            "(8 rows x 2 rounds, interleaved flat/blaze/shaping/dark), one "
            "exposure, honest dark-frame discipline. Raw capture only; no "
            "analysis, no threshold decisions."
        )
    )
    parser.add_argument("--out", type=Path, default=None, help="fresh output directory")
    parser.add_argument(
        "--execute", action="store_true", help="open hardware and acquire (only way to touch devices)"
    )
    parser.add_argument(
        "--self-test", action="store_true", help="assert offline invariants; no devices, no --out"
    )
    parser.add_argument(
        "--dry-run-plan", action="store_true", help="print the frozen plan incl. 16 frame ids; no devices"
    )
    parser.add_argument(
        "--shaping-phase",
        type=Path,
        default=SHAPING_PHASE_PATH,
        help="shaping target phase .npy (panel-resolution float radians)",
    )
    parser.add_argument(
        "--dark-frames-dir",
        type=Path,
        default=None,
        help="user-captured dark frames directory (matched by same exposure)",
    )
    parser.add_argument("--exposure-ms", type=float, default=DEFAULT_EXPOSURE_MS)
    parser.add_argument("--settle-s", type=float, default=DEFAULT_SETTLE_S)
    parser.add_argument(
        "--cooldown-s",
        type=float,
        default=DEFAULT_COOLDOWN_S,
        help="inter-segment cooldown seconds (>= 20 s per plan; actual wait is journaled)",
    )
    parser.add_argument(
        "--saturation-fraction",
        type=float,
        default=DEFAULT_SATURATION_FRACTION,
        help="record saturated-pixel count at this fraction of full scale; do not abort",
    )
    parser.add_argument(
        "--blaze-period-px",
        type=float,
        default=DEFAULT_BLAZE_PERIOD_PX,
        help="blaze period in panel pixels (2*pi per period; default 128)",
    )
    parser.add_argument(
        "--blaze-axis",
        type=int,
        choices=(0, 1),
        default=DEFAULT_BLAZE_AXIS,
        help="blaze ramp axis: 1 = panel_x (default), 0 = panel_y",
    )
    parser.add_argument(
        "--wait-until-idle", type=float, default=DEFAULT_WAIT_IDLE_S,
        help="seconds to wait for the bench device lock",
    )
    parser.add_argument("--slm-number", type=int, default=1)
    parser.add_argument("--cam-id", type=int, default=0)
    args = parser.parse_args(argv)

    if args.self_test:
        return run_self_test()

    if args.exposure_ms <= 0 or args.settle_s < 0 or args.cooldown_s < 0:
        parser.error("--exposure-ms>0, --settle-s>=0 and --cooldown-s>=0 required")
    if args.blaze_period_px <= 0:
        parser.error("--blaze-period-px>0 required")
    if args.cooldown_s < MIN_COOLDOWN_S:
        parser.error(
            f"--cooldown-s >= {MIN_COOLDOWN_S:g} required (plan: inter-segment cooldown)"
        )
    if not 0 < args.saturation_fraction <= 1:
        parser.error("0 < --saturation-fraction <= 1 required")

    plan = build_plan(args, Path(args.shaping_phase))

    if args.dry_run_plan:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return 0

    if args.out is None:
        parser.error("--out is required unless --self-test or --dry-run-plan is given")
    if not args.execute:
        # plan-only: 不建目录、不 import 驱动, 只打印计划摘要。
        print(
            json.dumps(
                {
                    "schema": PLAN_SCHEMA,
                    "mode": "plan-only",
                    "matrix": [f["id"] for f in plan["matrix"]],
                    "n_frames": len(plan["matrix"]),
                    "matrix_interpretation": MATRIX_INTERPRETATION,
                    "exposure_ms": float(args.exposure_ms),
                    "shaping_phase": plan["config"]["shaping_phase"],
                    "roi_frozen": plan["config"]["roi_frozen"],
                    "note": "use --execute to open hardware (needs explicit approval)",
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0
    return execute(args, plan)


if __name__ == "__main__":
    sys.exit(main())
