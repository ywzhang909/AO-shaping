"""动态整形·速度解耦采集：SLM 下发节拍与 CCD 测量节拍彼此独立（2026-11-20）。

为什么要有这个脚本
------------------
GUI 的 :class:`~ao_shaping.gui.slm.dynamic_shaping.DynamicShapingWorker` 已经把
「灰度域最短环绕插值 + 墙钟时间轴」做对了，但它继承 ``QThread``，模块级 import
PySide6，无法在 headless 采集环境里复用。本脚本把**同一套纯函数**
（:mod:`ao_shaping.utils.slm.transition_timeline`）搬到命令行采集流程：

* **端点各走一次完整管线**：起点与终点分别经 ``create_phase_from_array``（弧度→
  灰度 + 波前矫正 + LUT + 平移）拿到最终灰度码，各一次；扁平起点按驱动纪律直接
  构造灰度，绝不进 ``create_phase_from_array``。
* **中间帧只做灰度混合**：每帧 = ``blend_gray(start, target, t)``，绝不逐帧再调
  ``create_phase_from_array``（那会把已矫正的码二次矫正）。
* **速度解耦**：SLM 帧派发在主线程按墙钟时间轴推进；CCD 抓取在独立线程按自己的
  节拍（``--measure at-end|every|periodic``）运行。派发线程从不等待测量线程，
  测量线程也不持有派发所需的任何锁 —— 两路节奏不会互相串行化。

设备纪律
--------
* 默认 **plan-only**：不 import 任何设备驱动、不创建目录，只打印完整 argv 与计划
  摘要后干净退出。只有显式 ``--execute`` 才打开 SLM/CCD。
* ``--execute`` 时单进程打开 SLM + CCD **各一次**，``try/finally`` 释放；每次
  display / grab / open / close 都记入 ``<out>/device_log.json``。
* ``--wait-until-idle``（默认 120 s；0 = 同时关闭等待与加锁）在开设备前走
  :mod:`ao_shaping.tools.kernels.device_mutex`，并在整个采集期间持有设备锁。
  与 ``shape_transition_timing_20261106.py`` / ``direction_holdout_capture_20261119.py``
  同一口径。

计划摘要里的「每帧最大码变化」
------------------------------
精确值只有在拿到端点灰度数组后才能算。因此：

* 传了 ``--start-gray`` / ``--target-gray``（或相位 .npy，plan-only 用无设备的
  编码标度回退）→ 用 :func:`estimate_transition` 给出**逐码精确**估计；
* 什么都没传 → 用半周期 512 码的**全量程最坏上界**，并在摘要里显式标注
  ``basis=full-scale-bound``，绝不伪装成精确值。

用法::

    # 离线计划（不碰硬件、不建目录）
    python scripts/slm/dynamic_shaping_capture_20261120.py --out /tmp/dyn --duration-s 3

    # 硬件采集（本轮不执行，仅示例）
    python scripts/slm/dynamic_shaping_capture_20261120.py --execute --out ... \\
        --start-phase start.npy --target-phase target.npy --measure every --measure-every 5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shlex
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ao_shaping.utils.slm.transition_timeline import (
    EASING_LABELS,
    MAX_GRAY_CODE,
    TransitionEstimate,
    TransitionPlan,
    blend_gray,
    estimate_transition,
    pixel_flip_wait_s,
    wrapped_gray_delta,
)

MEASURE_STRATEGIES = ("at-end", "every", "periodic")
PLAN_SCHEMA = "speed_decouple_plan/1"
LABEL_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
DEFAULT_PANEL = (1920, 1200)
#: 半周期：10-bit 面板的最短环绕码差绝对值上限（wrapped_gray_delta 的范围 -512…511）。
MAX_WRAPPED_DELTA = (MAX_GRAY_CODE + 1) // 2  # 1024 周期的一半 = 512


# --------------------------------------------------------------------- 基础工具


def utc_now() -> str:
    """微秒精度 UTC 时间戳。"""
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def clock() -> dict[str, Any]:
    return {"timestamp_utc": utc_now(), "monotonic_ns": time.monotonic_ns()}


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"JSON 不支持的类型: {type(value)!r}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_input(raw: str | Path, name: str) -> Path:
    """路径解析规则：先按 CWD，再按仓库根 ROOT；失败即抛错（plan-only 阶段）。"""
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        paths = [candidate, ROOT / candidate]
    else:
        paths = [candidate]
    for p in paths:
        if p.is_file():
            return p.resolve()
    raise SystemExit(f"{name} 文件不存在: {raw!s}（已尝试 CWD 与仓库根 {ROOT}）")


def load_npy(path: Path | None, name: str) -> np.ndarray | None:
    """读取离线 .npy 端点；``None`` 表示没提供。不做任何目录创建。"""
    if path is None:
        return None
    arr = np.load(path, allow_pickle=False)
    if arr.ndim != 2:
        raise ValueError(f"{name} 必须是二维 .npy，收到 shape={arr.shape}")
    return arr


class RecordSink:
    """线程安全的 JSONL 记录器；``path=None`` 时只留内存（测试用）。"""

    def __init__(self, path: Path | None = None) -> None:
        self.records: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._fh = None
        if path is not None:
            self._fh = path.open("w", encoding="utf-8")

    def write(self, record: dict[str, Any]) -> None:
        with self._lock:
            self.records.append(record)
            if self._fh is not None:
                self._fh.write(json.dumps(record, ensure_ascii=False, default=_json_default))
                self._fh.write("\n")
                self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


class DeviceLog:
    """设备动作的 JSONL 风格日志（落盘为 ``device_log.json`` 的数组）。"""

    def __init__(self, path: Path | None = None) -> None:
        self.entries: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self.path = path

    def note(self, action: str, **fields: Any) -> None:
        entry: dict[str, Any] = {"action": action, **clock()}
        entry.update(fields)
        with self._lock:
            self.entries.append(entry)

    def flush(self) -> None:
        if self.path is None:
            return
        self.path.write_text(
            json.dumps(self.entries, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )


# --------------------------------------------------------------------- 参数 / 计划


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, required=True, help="输出目录（plan-only 不创建）")
    p.add_argument("--plan", type=Path, default=None,
                   help="多端点批次计划 JSON（speed_decouple_plan/1）；给了就忽略单对端点参数")
    p.add_argument("--raw-frames", choices=("none", "stack"), default="none",
                   help="是否保存逐帧原始 CCD 帧（默认 none，保持原行为）")
    p.add_argument("--inter-transition-settle-s", type=float, default=0.5,
                   help="相邻 transition 之间的 SLM 稳定等待（秒）")
    p.add_argument("--on-error", choices=("abort", "continue"), default="abort",
                   help="某 transition 失败后中止或继续")
    p.add_argument("--duration-s", type=float, default=3.0, help="过渡时长（秒）")
    p.add_argument("--fps", type=int, default=10, help="目标帧率（帧/秒）")
    p.add_argument("--easing", choices=sorted(EASING_LABELS), default="smooth")
    p.add_argument("--dwell-ms", type=float, default=0.0,
                   help="每步停留下限（毫秒）；0 = 只按像素翻转时间自动")
    p.add_argument("--start-gray", type=Path, default=None, help="起点最终灰度 .npy (uint16)")
    p.add_argument("--target-gray", type=Path, default=None, help="终点最终灰度 .npy (uint16)")
    p.add_argument("--start-phase", type=Path, default=None, help="起点弧度相位 .npy")
    p.add_argument("--target-phase", type=Path, default=None, help="终点弧度相位 .npy")
    p.add_argument("--start-flat", type=int, default=None,
                   help="扁平起点灰度码（不进 create_phase_from_array）")
    p.add_argument("--panel-w", type=int, default=DEFAULT_PANEL[0])
    p.add_argument("--panel-h", type=int, default=DEFAULT_PANEL[1])

    p.add_argument("--measure", choices=MEASURE_STRATEGIES, default="at-end",
                   help="CCD 测量节拍：末尾一次 / 每 N 帧 / 固定周期")
    p.add_argument("--measure-every", type=int, default=5, help="--measure every 的帧间隔")
    p.add_argument("--measure-interval-s", type=float, default=0.25,
                   help="--measure periodic 的周期（秒）")
    p.add_argument("--core-half", type=int, default=32, help="峰.core 半宽（像素）")
    p.add_argument("--target-mask", type=Path, default=None,
                   help="目标区域 bool .npy，用于 target_uniformity")
    p.add_argument("--uniformity-min", type=float, default=0.9,
                   help="target_uniformity 接受阈值")

    p.add_argument("--slm-number", type=int, default=1)
    p.add_argument("--wavelength", type=int, default=1064)
    p.add_argument("--camera", default="daheng")
    p.add_argument("--cam-id", type=int, default=0)
    p.add_argument("--exposure-ms", type=float, default=None)
    p.add_argument("--bit-depth", type=int, default=12)
    p.add_argument("--wait-until-idle", type=float, default=120.0, metavar="SECONDS",
                   help="开设备前最多等待秒数；0 = 关闭等待与加锁")
    p.add_argument("--execute", action="store_true", help="真正打开硬件（默认只打印计划）")
    args = p.parse_args(argv)

    if not math.isfinite(args.wait_until_idle) or args.wait_until_idle < 0:
        p.error("--wait-until-idle 必须是非负有限数")
    if args.measure_every < 1:
        p.error("--measure-every 必须 >= 1")
    if args.measure_interval_s <= 0:
        p.error("--measure-interval-s 必须 > 0")
    if args.core_half < 1:
        p.error("--core-half 必须 >= 1")
    if not math.isfinite(args.inter_transition_settle_s) or args.inter_transition_settle_s < 0:
        p.error("--inter-transition-settle-s 必须是非负有限数")
    if args.plan is not None:
        # 给了 --plan 就忽略单对端点参数，不做互斥检查。
        return args
    starts = [bool(args.start_gray), bool(args.start_phase), args.start_flat is not None]
    if sum(starts) > 1:
        p.error("--start-gray / --start-phase / --start-flat 三选一")
    if bool(args.target_gray) and bool(args.target_phase):
        p.error("--target-gray / --target-phase 二选一")
    return args


def build_plan(args: argparse.Namespace) -> TransitionPlan:
    """参数 → 共享时间轴对象（Qt 无关，校验集中在 TransitionPlan）。"""
    return TransitionPlan(
        duration_s=float(args.duration_s),
        fps=int(args.fps),
        easing=str(args.easing),
        dwell_ms=float(args.dwell_ms),
    )


def _offline_phase_to_gray(phase: np.ndarray, panel: tuple[int, int]) -> np.ndarray:
    """无设备时相位→灰度的编码标度回退，仅用于 plan-only 精确摘要。

    与 :func:`ao_shaping.utils.slm.phase_display.phase_to_slm_grayscale` 的
    ``slm=None`` 分支同一口径（mod 2π × 1023，无矫正/LUT/平移）。``--execute``
    一律走设备 ``create_phase_from_array``，绝不用本函数生成下发帧。
    """
    from ao_shaping.utils.slm.phase_display import phase_to_slm_grayscale

    gray = np.asarray(
        phase_to_slm_grayscale(
            np.asarray(phase), max_grayscale=MAX_GRAY_CODE, slm=None
        ),
        dtype=np.uint16,
    )
    if gray.shape != panel:
        raise ValueError(f"相位灰度尺寸 {gray.shape} 与面板 {panel} 不一致")
    return gray


def offline_gray(
    args: argparse.Namespace, which: str, panel: tuple[int, int]
) -> np.ndarray | None:
    """plan-only 下解析端点灰度；没有提供端点时返回 ``None``（摘要改用上界）。"""
    if which == "start":
        if args.start_gray is not None:
            return np.asarray(load_npy(args.start_gray, "start-gray"), dtype=np.uint16)
        if args.start_phase is not None:
            return _offline_phase_to_gray(load_npy(args.start_phase, "start-phase"), panel)
        if args.start_flat is not None:
            return np.full(panel, int(args.start_flat), dtype=np.uint16)
        return None
    if args.target_gray is not None:
        return np.asarray(load_npy(args.target_gray, "target-gray"), dtype=np.uint16)
    if args.target_phase is not None:
        return _offline_phase_to_gray(load_npy(args.target_phase, "target-phase"), panel)
    return None


def summarize(
    args: argparse.Namespace,
    plan: TransitionPlan,
    start_gray: np.ndarray | None,
    target_gray: np.ndarray | None,
) -> dict[str, Any]:
    """计划摘要：帧数 / 时长 / 每帧最大码变化 / 翻转等待 / 预计总时长。"""
    summary: dict[str, Any] = {
        "duration_s": plan.duration_s,
        "fps": plan.fps,
        "easing": plan.easing,
        "dwell_ms": plan.dwell_ms,
        "frames": plan.frame_count,
        "frame_interval_s": plan.frame_interval_s,
        "measure": {
            "strategy": args.measure,
            "every_n_frames": args.measure_every,
            "interval_s": args.measure_interval_s,
        },
        "out": str(args.out.resolve()),
        "execute": bool(args.execute),
    }
    if start_gray is not None and target_gray is not None:
        est: TransitionEstimate = estimate_transition(plan, start_gray, target_gray)
        summary.update(
            max_code_change_per_frame=est.max_step_gray,
            pixel_flip_wait_s=est.flip_wait_s,
            expected_total_duration_s=est.expected_duration_s,
            limited_by_panel=est.limited_by_panel,
            basis="exact: endpoint gray arrays available",
        )
    else:
        # 没有任何端点数组：用半周期 512 码算出**上界**，绝不冒充精确值。
        step = MAX_WRAPPED_DELTA / max(1, plan.frame_count - 1)
        flip = pixel_flip_wait_s(step)
        dwell = max(flip, plan.dwell_s)
        first_wait = pixel_flip_wait_s(float(MAX_GRAY_CODE))
        summary.update(
            max_code_change_per_frame=step,
            pixel_flip_wait_s=flip,
            expected_total_duration_s=(plan.frame_count - 1)
            * max(plan.frame_interval_s, dwell) + first_wait,
            limited_by_panel=dwell > plan.frame_interval_s * 1.05,
            basis="full-scale-bound: no endpoint arrays supplied (max |wrapped delta|=512)",
        )
    return summary


# --------------------------------------------------------------------- 多端点批次计划


def load_plan(path: Path) -> dict[str, Any]:
    """解析 / 校验 ``--plan`` JSON，解析端点路径并附 SHA256。

    纯离线：只读 JSON 与端点 .npy 文件，不 import 驱动、不创建目录。
    任何校验错误一律抛 ``SystemExit``（plan-only 阶段即报错退出）。
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"--plan 读取失败: {path} -- {exc}") from exc
    if not isinstance(raw, dict):
        raise SystemExit("--plan 顶层必须是 JSON 对象")
    if raw.get("schema") != PLAN_SCHEMA:
        raise SystemExit(f"--plan schema 必须是 {PLAN_SCHEMA!r}，收到 {raw.get('schema')!r}")

    def fail(msg: str) -> None:
        raise SystemExit(f"--plan 非法: {msg}")

    panel = raw.get("panel")
    if (
        not isinstance(panel, (list, tuple))
        or len(panel) != 2
        or any(not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in panel)
    ):
        fail(f"panel 必须是 [W, H] 正整数，收到 {panel!r}")
    panel_t = (int(panel[0]), int(panel[1]))

    duration = raw.get("duration_s")
    fps = raw.get("fps")
    if not isinstance(duration, (int, float)) or isinstance(duration, bool) or not math.isfinite(duration) or duration <= 0:
        fail(f"duration_s 必须是正有限数，收到 {duration!r}")
    if not isinstance(fps, int) or isinstance(fps, bool) or fps <= 0:
        fail(f"fps 必须是正整数，收到 {fps!r}")
    easing = raw.get("easing", "smooth")
    if easing not in EASING_LABELS:
        fail(f"easing 未知: {easing!r}")
    measure = raw.get("measure", {"strategy": "every", "every_n_frames": 1})
    if not isinstance(measure, dict) or measure.get("strategy") not in MEASURE_STRATEGIES:
        fail(f"measure.strategy 必须是 {MEASURE_STRATEGIES}，收到 {measure!r}")
    every_n = measure.get("every_n_frames", 1)
    if not isinstance(every_n, int) or isinstance(every_n, bool) or every_n < 1:
        fail(f"measure.every_n_frames 必须 >= 1，收到 {every_n!r}")
    bit_depth = raw.get("bit_depth", 12)
    if not isinstance(bit_depth, int) or isinstance(bit_depth, bool) or bit_depth < 1:
        fail(f"bit_depth 必须是正整数，收到 {bit_depth!r}")
    exposure_ms = raw.get("exposure_ms")
    if exposure_ms is not None and (
        not isinstance(exposure_ms, (int, float))
        or isinstance(exposure_ms, bool)
        or not math.isfinite(exposure_ms)
        or exposure_ms <= 0
    ):
        fail(f"exposure_ms 必须是正有限数，收到 {exposure_ms!r}")

    transitions = raw.get("transitions")
    if not isinstance(transitions, list) or not transitions:
        fail("transitions 必须是非空数组")
    labels: set[str] = set()
    resolved: list[dict[str, Any]] = []
    for idx, tr in enumerate(transitions):
        if not isinstance(tr, dict):
            fail(f"transitions[{idx}] 必须是对象")
        label = tr.get("label")
        if not isinstance(label, str) or not LABEL_RE.fullmatch(label):
            fail(f"transitions[{idx}].label 只能含 [A-Za-z0-9_.-]，收到 {label!r}")
        if label in labels:
            fail(f"重复 label: {label!r}")
        labels.add(label)
        group = tr.get("group")
        if not isinstance(group, dict):
            fail(f"{label}.group 必须是对象")
        endpoint_out: dict[str, dict[str, Any]] = {}
        for endpoint_name in ("start", "target"):
            endpoint = tr.get(endpoint_name)
            if not isinstance(endpoint, dict):
                fail(f"{label}.{endpoint_name} 必须是对象")
            kind = endpoint.get("kind")
            allowed = ("flat", "phase", "gray") if endpoint_name == "start" else ("phase", "gray")
            if kind not in allowed:
                fail(f"{label}.{endpoint_name}.kind 必须是 {allowed}，收到 {kind!r}")
            if kind == "flat":
                gray = endpoint.get("gray")
                if not isinstance(gray, int) or isinstance(gray, bool) or not (0 <= gray <= MAX_GRAY_CODE):
                    fail(f"{label}.start.gray 须在 0..{MAX_GRAY_CODE}，收到 {gray!r}")
                endpoint = dict(endpoint)
            else:
                p = endpoint.get("path")
                if not isinstance(p, str) or not p:
                    fail(f"{label}.{endpoint_name}.path 必须是非空字符串")
                endpoint = dict(endpoint)
                endpoint["path"] = str(_resolve_input(p, f"{label}.{endpoint_name}.path"))
                endpoint["sha256"] = _sha256_file(Path(endpoint["path"]))
                arr = np.load(endpoint["path"], allow_pickle=False)
                if arr.ndim != 2 or tuple(arr.shape) != tuple(reversed(panel_t)):
                    fail(
                        f"{label}.{endpoint_name} 数组 shape={arr.shape} "
                        f"与 panel [W,H]={panel_t} 不一致"
                    )
                if kind == "gray" and arr.size and int(np.asarray(arr).max()) > MAX_GRAY_CODE:
                    # 只有灰码顶点需要校验上界；相位弧度不在此限。
                    fail(
                        f"{label}.{endpoint_name} 灰度越界 "
                        f"(max={int(np.asarray(arr).max())} > {MAX_GRAY_CODE})"
                    )
            endpoint_out[endpoint_name] = endpoint
        mask_path = tr.get("target_mask")
        resolved_mask = None
        if mask_path is not None:
            if not isinstance(mask_path, str) or not mask_path:
                fail(f"{label}.target_mask 必须是非空字符串或 null")
            p = _resolve_input(mask_path, f"{label}.target_mask")
            resolved_mask = {
                "path": str(p),
                "sha256": _sha256_file(p),
            }
            mask = np.load(p, allow_pickle=False)
            if tuple(mask.shape) != tuple(reversed(panel_t)):
                fail(f"{label}.target_mask shape={mask.shape} 与 panel {panel_t} 不一致")
        resolved.append({
            "label": label,
            "group": dict(group),
            "start": endpoint_out["start"],
            "target": endpoint_out["target"],
            "target_mask": resolved_mask,
        })

    plan = TransitionPlan(
        duration_s=float(duration), fps=int(fps), easing=str(easing), dwell_ms=0.0
    )
    return {
        "schema": PLAN_SCHEMA,
        "source": str(path.resolve()),
        "panel": [int(panel[0]), int(panel[1])],
        "duration_s": float(duration),
        "fps": int(fps),
        "easing": str(easing),
        "measure": {"strategy": str(measure["strategy"]), "every_n_frames": int(every_n)},
        "exposure_ms": exposure_ms,
        "bit_depth": int(bit_depth),
        "frame_count": plan.frame_count,
        "frame_interval_s": plan.frame_interval_s,
        "transitions": resolved,
    }


def plan_argv(args: argparse.Namespace, plan: dict[str, Any]) -> argparse.Namespace:
    """把批次计划映射成单对参数视图，供 ``build_plan`` / 摘要逻辑复用。"""
    view = argparse.Namespace()
    for key in (
        "duration_s", "fps", "easing", "dwell_ms", "measure", "measure_every",
        "measure_interval_s", "target_mask", "core_half", "uniformity_min",
        "wait_until_idle", "slm_number", "wavelength", "camera", "cam_id",
        "exposure_ms", "bit_depth", "execute", "out", "raw_frames",
        "inter_transition_settle_s", "on_error",
    ):
        setattr(view, key, getattr(args, key, None))
    view.duration_s = plan["duration_s"]
    view.fps = plan["fps"]
    view.easing = plan["easing"]
    view.measure = plan["measure"]["strategy"]
    view.measure_every = plan["measure"]["every_n_frames"]
    view.exposure_ms = plan["exposure_ms"]
    view.bit_depth = plan["bit_depth"]
    view.target_mask = None
    view.start_gray = view.start_phase = view.start_flat = None
    view.target_gray = view.target_phase = None
    view.panel_w, view.panel_h = plan["panel"]
    return view


def print_plan(argv: Sequence[str], args: argparse.Namespace, summary: dict[str, Any]) -> None:
    """完整 argv + 计划摘要，纯打印，不落盘不建目录。"""
    print("argv: " + shlex.join([Path(sys.argv[0]).name, *argv]))
    print(f"PLAN ONLY (execute={summary['execute']}): no driver import, no directory created")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))
    print("Re-run with --execute to open SLM/CCD and acquire.")


# --------------------------------------------------------------------- 帧派发


@dataclass
class SharedState:
    """派发线程 → 测量线程的单向状态；只有 int/float/bool 的原子赋值。"""

    index: int = -1
    t: float = 0.0
    max_change: float = 0.0
    flip_wait: float = 0.0
    done: bool = False


def dispatch_frames(
    plan: TransitionPlan,
    start_gray: np.ndarray,
    target_gray: np.ndarray,
    *,
    display: Callable[[int, np.ndarray], dict[str, Any]],
    state: SharedState,
    sink: RecordSink,
    sleeper: Callable[[float], None] = time.sleep,
    clock_fn: Callable[[], float] = time.monotonic,
) -> list[dict[str, Any]]:
    """按墙钟时间轴下发灰度帧；只调 ``display(index, gray)``，不碰相位管线。

    ``t`` 由墙钟决定：硬件慢时跳进度而不是拉长过渡；首帧 t=0（起点逐码精确），
    末帧 t=1（终点逐码精确）。每帧停留 ``max(dwell_ms, 翻转等待)``。
    """
    if start_gray.shape != target_gray.shape:
        raise ValueError(f"起止灰度尺寸不一致: {start_gray.shape} vs {target_gray.shape}")
    t0 = clock_fn()
    prev = np.asarray(start_gray, dtype=np.uint16)
    frames: list[dict[str, Any]] = []
    index = 0
    while True:
        now = clock_fn()
        t = min(1.0, max(0.0, (now - t0) / plan.duration_s))
        gray = blend_gray(start_gray, target_gray, t)
        max_change = float(np.abs(wrapped_gray_delta(prev, gray)).max()) if gray.size else 0.0
        flip = pixel_flip_wait_s(max_change)
        dwell = max(flip, plan.dwell_s)
        extra = display(index, gray)
        rec: dict[str, Any] = {
            "event": "frame",
            **clock(),
            "frame_index": index,
            "progress_t": t,
            "max_code_change": max_change,
            "pixel_flip_wait": flip,
            "dwell_s": dwell,
        }
        rec.update(extra or {})
        sink.write(rec)
        frames.append(rec)
        state.index = index
        state.t = t
        state.max_change = max_change
        state.flip_wait = flip
        prev = gray
        sleeper(dwell)
        index += 1
        if t >= 1.0:
            break
        deadline = t0 + index * plan.frame_interval_s
        now2 = clock_fn()
        if now2 < deadline:
            sleeper(deadline - now2)
    state.done = True
    return frames


# --------------------------------------------------------------------- CCD 采样


def compute_frame_stats(
    frame: np.ndarray,
    core_half: int,
    target_mask: np.ndarray | None = None,
    uniformity_min: float = 0.9,
) -> dict[str, Any]:
    """峰值 / 峰位（0 级 argmax）/ 核心 CV / 目标区均匀度。"""
    img = np.asarray(frame, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError(f"CCD 帧必须是二维，收到 shape={img.shape}")
    peak = float(img.max()) if img.size else 0.0
    y, x = (int(v) for v in np.unravel_index(int(np.argmax(img)), img.shape)) if img.size else (0, 0)
    y0, y1 = max(0, y - core_half), min(img.shape[0], y + core_half + 1)
    x0, x1 = max(0, x - core_half), min(img.shape[1], x + core_half + 1)
    core = img[y0:y1, x0:x1]
    mean = float(core.mean()) if core.size else 0.0
    cv = float(np.sqrt((((core / mean) - 1.0) ** 2).mean())) if mean > 0 else float("nan")

    uniformity: float | None = None
    accepted: bool | None = None
    if target_mask is not None:
        mask = np.asarray(target_mask, dtype=bool)
        if mask.shape != img.shape:
            raise ValueError(f"target_mask 尺寸 {mask.shape} 与帧 {img.shape} 不一致")
        vals = img[mask]
        m2 = float(vals.mean()) if vals.size else 0.0
        cv2 = float(np.sqrt((((vals / m2) - 1.0) ** 2).mean())) if m2 > 0 else float("nan")
        uniformity = max(0.0, 1.0 - cv2) if math.isfinite(cv2) else None
        accepted = None if uniformity is None else bool(uniformity >= uniformity_min)
    return {
        "peak": peak,
        "center_xy": [x, y],
        "core_cv": cv if math.isfinite(cv) else None,
        "target_uniformity": uniformity,
        "target_accepted": accepted,
    }


class CCDSampler(threading.Thread):
    """独立测量线程：按自己的节拍抓帧，从不阻塞 SLM 派发。

    ``state`` 只有派发线程写、本线程读；本线程唯一写入的是 ``sink``（内部有锁）
    与自己的局部计数。``stop`` 只用于异常退出，正常退出由 ``state.done`` 触发。
    """

    def __init__(
        self,
        cam: Any,
        state: SharedState,
        *,
        strategy: str,
        sink: RecordSink,
        device_log: DeviceLog,
        every_n: int = 5,
        interval_s: float = 0.25,
        core_half: int = 32,
        target_mask: np.ndarray | None = None,
        uniformity_min: float = 0.9,
        stop: threading.Event | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        raw_stack: "RawFrameStack | None" = None,
    ) -> None:
        super().__init__(name="ccd-sampler", daemon=True)
        if strategy not in MEASURE_STRATEGIES:
            raise ValueError(f"未知测量策略: {strategy!r}")
        self.cam = cam
        self.state = state
        self.strategy = strategy
        self.sink = sink
        self.device_log = device_log
        self.every_n = max(1, int(every_n))
        self.interval_s = float(interval_s)
        self.core_half = int(core_half)
        self.target_mask = target_mask
        self.uniformity_min = float(uniformity_min)
        self.stop = stop if stop is not None else threading.Event()
        self._sleeper = sleeper
        self.raw_stack = raw_stack

    def _grab(self) -> None:
        frame_raw = self.cam.get_numpy_image(n_sample=1)
        frame = np.asarray(frame_raw, dtype=np.float64)
        if frame.ndim == 3:
            frame = frame.mean(0)
        stats = compute_frame_stats(
            frame, self.core_half, self.target_mask, self.uniformity_min
        )
        self.device_log.note("grab", **stats)
        record = {
            "event": "measure",
            **clock(),
            "observed_frame_index": self.state.index,
            "observed_progress_t": self.state.t,
            **stats,
        }
        self.sink.write(record)
        if self.raw_stack is not None:
            self.raw_stack.on_frame(frame, stats, record)

    def run(self) -> None:  # pragma: no cover - 线程体由 decoupled 测试覆盖
        last = -1
        while not self.stop.is_set():
            if self.strategy == "at-end":
                if not self.state.done:
                    self._sleeper(0.01)
                    continue
                self._grab()
                return
            if self.strategy == "every":
                if self.state.index < 0:
                    self._sleeper(0.005)
                    continue
                if self.state.index - last < self.every_n and not self.state.done:
                    self._sleeper(0.005)
                    continue
                if self.state.index == last and self.state.done:
                    return
                last = self.state.index
                self._grab()
                if self.state.done:
                    return
                continue
            # periodic：按固定周期抓，完全不看派发进度。
            self._sleeper(self.interval_s)
            if self.state.index < 0 and not self.state.done:
                continue
            self._grab()
            if self.state.done:
                return


# --------------------------------------------------------------------- 原始帧堆叠


class RawFrameStack:
    """逐帧 uint16 原始帧 memmap 写入器：首帧惰性预分配、顺序写。

    帧尺寸取 **CCD 原生全幅**（冻结计划禁止物理开窗），不能按 SLM 面板尺寸预分配：
    面板 (H_slm, W_slm) 与相机 (H_ccd, W_ccd) 量级完全不同（如 1944x2592 vs
    1200x1920），按面板预分配会让每帧都判 shape 不一致。相机帧尺寸在第一次抓帧
    前不可靠，故 ``self.shape is None`` 时在 :meth:`on_frame` 里用首个真实帧定型。
    """

    def __init__(
        self,
        path: Path,
        capacity: int,
        shape: tuple[int, int] | None = None,
        *,
        cam: Any,
        bit_depth: int,
        exposure_ms: float | None,
        requested_measure_every: int,
    ) -> None:
        self.path = path
        self.capacity = int(capacity)
        # shape 仅作可选旁证；None 表示由首个相机帧决定（推荐）。绝不用面板尺寸兜底。
        self.shape: tuple[int, int] | None = (
            None if shape is None else (int(shape[0]), int(shape[1]))
        )
        self.cam = cam
        self.bit_depth = int(bit_depth)
        self.exposure_ms = exposure_ms
        self.requested_measure_every = int(requested_measure_every)
        self.effective_full_scale: float | None = None
        self.full_scale_source: str = "unresolved (no frames captured)"
        self.records: list[dict[str, Any]] = []
        self._t0_ns: int | None = None
        self._mmap: Any = None

    def _open_memmap(self) -> None:
        assert self.shape is not None
        self._mmap = np.lib.format.open_memmap(
            self.path, mode="w+", dtype=np.uint16,
            shape=(self.capacity, *self.shape),
        )

    def on_frame(self, raw: np.ndarray, stats: dict[str, Any], context: dict[str, Any]) -> None:
        """CCDSampler 每抓一帧调用一次：写图像 + 追加元数据记录。"""
        i = len(self.records)
        if i >= self.capacity:
            return  # 容量上限：超出的帧不写，避免越界覆盖
        frame = np.asarray(raw)
        if frame.ndim == 3:
            frame = frame.mean(0)
        if self.shape is None:
            # 首帧定型：按 CCD 原生全幅预分配，之后逐帧与原幅比对。
            self.shape = (int(frame.shape[0]), int(frame.shape[1]))
            self._open_memmap()
        if tuple(frame.shape) != self.shape:
            raise ValueError(f"原始帧 shape={frame.shape} 与预分配 {self.shape} 不一致")
        if self.effective_full_scale is None:
            self.effective_full_scale, self.full_scale_source = _frame_full_scale(
                self.cam, frame
            )
        self._mmap[i] = np.asarray(frame, dtype=np.uint16)
        if self._t0_ns is None:
            self._t0_ns = int(context.get("monotonic_ns", time.monotonic_ns()))
        t_rel_s = (int(context.get("monotonic_ns", self._t0_ns)) - self._t0_ns) * 1e-9
        peak = float(stats.get("peak", 0.0))
        saturated = int(np.count_nonzero(np.asarray(frame) >= self.effective_full_scale))
        self.records.append({
            "i": i,
            "slm_frame_index": int(context.get("observed_frame_index", -1)),
            "t_rel_s": round(t_rel_s, 9),
            "monotonic_ns": int(context.get("monotonic_ns", time.monotonic_ns())),
            "peak": peak,
            "saturated_pixels": saturated,
            "uniformity": stats.get("target_uniformity"),
        })

    def finalize(self, *, slm_frames_dispatched: int, elapsed_s: float) -> dict[str, Any]:
        if self._mmap is not None:
            self._mmap.flush()
        n = len(self.records)
        meta: dict[str, Any] = {
            "n_measured": n,
            "capacity": self.capacity,
            # 零帧时未知原生幅面：None，绝不回填面板尺寸。
            "shape": None if self.shape is None else list(self.shape),
            "dtype": "uint16",
            "bit_depth": self.bit_depth,
            "exposure_ms": self.exposure_ms,
            "raw_dtype_max": int(np.iinfo(np.uint16).max),
            "effective_full_scale": (
                self.effective_full_scale
                if self.effective_full_scale is not None
                else float(np.iinfo(np.uint16).max)
            ),
            "full_scale_source": self.full_scale_source,
            "achieved_measure_fps": (n / elapsed_s) if elapsed_s > 0 else 0.0,
            "requested_measure_every": self.requested_measure_every,
            "slm_frames_dispatched": int(slm_frames_dispatched),
            "measurements_per_slm_frame": (
                n / slm_frames_dispatched if slm_frames_dispatched > 0 else 0.0
            ),
            "records": self.records,
        }
        (self.path.parent / "frames_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )
        return meta


def _frame_full_scale(cam: Any, frame: np.ndarray) -> tuple[float, str]:
    """issue #246：帧 dtype 的 iinfo 上限 ≠ 生效位深满量程，一律问相机。"""
    resolver = getattr(cam, "effective_full_scale", None)
    if callable(resolver):
        try:
            value = float(resolver(frame))
            if math.isfinite(value) and value > 0:
                return value, "camera.effective_full_scale"
        except Exception:  # pragma: no cover - 相机实现差异兜底
            pass
    return float(np.iinfo(np.asarray(frame).dtype).max), "dtype.iinfo fallback"


def _plan_endpoint_gray(
    slm: Any,
    endpoint: dict[str, Any],
    which: str,
    panel: tuple[int, int],
    device_log: DeviceLog,
) -> np.ndarray:
    """批次计划的端点灰度：flat 直构；phase/gray 各走一次 create_phase_from_array。"""
    kind = endpoint["kind"]
    if kind == "flat":
        # 扁平起点绝不进 create_phase_from_array。
        return np.full(panel, int(endpoint["gray"]), dtype=np.uint16)
    if kind == "gray":
        gray = np.asarray(load_npy(Path(endpoint["path"]), f"{which}-gray"), dtype=np.uint16)
    else:
        phase = np.asarray(load_npy(Path(endpoint["path"]), f"{which}-phase"))
        gray = np.asarray(slm.create_phase_from_array(phase), dtype=np.uint16)
        device_log.note("create_phase_from_array", endpoint=which, shape=list(phase.shape))
    if gray.shape != panel:
        raise ValueError(f"{which} 灰度尺寸 {gray.shape} 与面板 {panel} 不一致")
    if gray.size and int(gray.max()) > MAX_GRAY_CODE:
        raise ValueError(f"{which} 灰度越界 (max={int(gray.max())} > {MAX_GRAY_CODE})")
    return gray


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _batch_timing(plan: dict[str, Any]) -> float:
    step = MAX_WRAPPED_DELTA / max(1, plan["frame_count"] - 1)
    flip = pixel_flip_wait_s(step)
    dwell = max(flip, 0.0)
    return (plan["frame_count"] - 1) * max(plan["frame_interval_s"], dwell) + pixel_flip_wait_s(
        float(MAX_GRAY_CODE)
    )


def _print_batch_plan(argv: Sequence[str], plan: dict[str, Any], out: Path) -> None:
    """plan-only 批次摘要：每条 transition 一行 + 总数与预计时长。"""
    print("argv: " + shlex.join([Path(sys.argv[0]).name, *argv]))
    print("PLAN ONLY (execute=False): no driver import, no directory created")
    print(f"schema={plan['schema']} panel={plan['panel']} duration_s={plan['duration_s']} "
          f"fps={plan['fps']} easing={plan['easing']}")
    total_s = 0.0
    max_change = MAX_WRAPPED_DELTA / max(1, plan["frame_count"] - 1)
    per_s = _batch_timing(plan)
    for tr in plan["transitions"]:
        total_s += per_s
        print(
            f"  {tr['label']}: start={tr['start']['kind']} target={tr['target']['kind']} "
            f"frames={plan['frame_count']} max_delta_per_frame={max_change:.3f} "
            f"(basis=full-scale-bound)"
        )
    print(f"transitions={len(plan['transitions'])} total_frames="
          f"{plan['frame_count'] * len(plan['transitions'])} "
          f"expected_total_duration_s={total_s:.3f}")
    print("Re-run with --execute to open SLM/CCD and acquire.")


# --------------------------------------------------------------------- execute


def endpoint_gray(
    slm: Any,
    args: argparse.Namespace,
    which: str,
    panel: tuple[int, int],
    device_log: DeviceLog,
) -> np.ndarray:
    """端点灰度管线：**每个端点至多一次** create_phase_from_array。"""
    if which == "start":
        if args.start_gray is not None:
            gray = np.asarray(load_npy(args.start_gray, "start-gray"), dtype=np.uint16)
        elif args.start_phase is not None:
            phase = np.asarray(load_npy(args.start_phase, "start-phase"))
            gray = np.asarray(slm.create_phase_from_array(phase), dtype=np.uint16)
            device_log.note("create_phase_from_array", endpoint="start", shape=list(phase.shape))
        elif args.start_flat is not None:
            gray = np.full(panel, int(args.start_flat), dtype=np.uint16)
        else:
            gray = np.full(panel, 0, dtype=np.uint16)  # GUI 默认起点：flat 0
            device_log.note("default_start", gray_code=0)
    else:
        if args.target_gray is not None:
            gray = np.asarray(load_npy(args.target_gray, "target-gray"), dtype=np.uint16)
        elif args.target_phase is not None:
            phase = np.asarray(load_npy(args.target_phase, "target-phase"))
            gray = np.asarray(slm.create_phase_from_array(phase), dtype=np.uint16)
            device_log.note("create_phase_from_array", endpoint="target", shape=list(phase.shape))
        else:
            raise SystemExit("--execute 必须提供 --target-gray 或 --target-phase")
    if gray.shape != panel:
        raise ValueError(f"{which} 灰度尺寸 {gray.shape} 与面板 {panel} 不一致")
    if gray.size and int(gray.max()) > MAX_GRAY_CODE:
        raise ValueError(f"{which} 灰度越界 (max={int(gray.max())} > {MAX_GRAY_CODE})")
    return gray


def _wait_and_lock(args: argparse.Namespace) -> tuple[Any, dict[str, Any], Callable[[], None]]:
    """``--wait-until-idle`` → 持锁；返回 (lock, 证据, release 回调)。"""
    if args.wait_until_idle <= 0:
        evidence = {"status": "skipped", "budget_s": 0.0,
                    "detail": "waiting disabled (--wait-until-idle 0)"}
        return None, evidence, lambda: None
    from ao_shaping.tools.kernels import device_mutex
    from ao_shaping.tools.kernels.device_mutex import DeviceLock

    verdict = device_mutex.wait_until_idle(
        args.wait_until_idle,
        slm_number=args.slm_number,
        cam_id=args.cam_id,
        cam_type=args.camera,
        own_paths=[Path(__file__).resolve()],
    )
    evidence: dict[str, Any] = {
        "status": verdict.status,
        "budget_s": args.wait_until_idle,
        "waited_s": round(verdict.waited_s, 3),
        "holder_pid": verdict.holder.pid if verdict.holder is not None else None,
        "detail": verdict.detail,
        "remedy": verdict.remedy,
    }
    if not verdict.ok:
        raise SystemExit(
            f"device unavailable ({verdict.kind or verdict.status}): "
            f"{verdict.reason or verdict.detail or verdict.status} -- {verdict.remedy}"
        )
    lock = verdict.lock
    if lock is None:
        lock = DeviceLock.default()
        if not lock.acquire(timeout=0.0):
            raise SystemExit("device lock just taken by another session; rerun")
    evidence["lock_path"] = str(getattr(lock, "path", ""))
    return lock, evidence, lock.release


def execute(args: argparse.Namespace, summary: dict[str, Any]) -> int:
    """唯一的硬件路径：开一次、跑完、finally 释放；返回进程退出码。"""
    from ao_shaping.drivers.ccd.common import create_camera, get_camera_exposure_ms
    from ao_shaping.drivers.slm.santec import Santec, SlotRotator
    from loguru import logger

    out: Path = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)  # 只有 --execute 才建目录
    batch: dict[str, Any] | None = load_plan(args.plan) if args.plan is not None else None
    single_plan = build_plan(args)
    if batch is not None:
        _write_json(out / "run_plan.json", {
            **batch,
            "resolved_at": utc_now(),
            "cli": {
                "raw_frames": args.raw_frames,
                "inter_transition_settle_s": args.inter_transition_settle_s,
                "on_error": args.on_error,
                "wait_until_idle": args.wait_until_idle,
            },
        })
    sink = RecordSink(out / "frames.jsonl")
    device_log = DeviceLog(out / "device_log.json")
    lock, lock_evidence, release_lock = _wait_and_lock(args)
    slm: Any = None
    cam: Any = None
    sampler: CCDSampler | None = None
    stop = threading.Event()
    state = SharedState()
    run_items: list[dict[str, Any]] = []
    if batch is None:
        run_items.append({
            "label": "default",
            "start": None,  # 单端口仍走 endpoint_gray(args)
            "target": None,
            "target_mask_path": args.target_mask,
            "plan": single_plan,
        })
    else:
        for tr in batch["transitions"]:
            run_items.append({
                "label": tr["label"],
                "start": tr["start"],
                "target": tr["target"],
                "target_mask_path": tr["target_mask"],
                "plan": TransitionPlan(
                    duration_s=batch["duration_s"],
                    fps=batch["fps"],
                    easing=batch["easing"],
                    dwell_ms=0.0,
                ),
            })
    try:
        device_log.note("open", device=f"camera:{args.camera}", cam_id=args.cam_id)
        cam = create_camera(
            args.camera, args.cam_id,
            exposure_time_ms=(batch["exposure_ms"] if batch else args.exposure_ms) or 0.5,
            bit_depth=int(batch["bit_depth"] if batch else args.bit_depth),
        )
        cam.open()
        exposure = float(get_camera_exposure_ms(cam))
        device_log.note("open", device=f"slm#{args.slm_number}", mode="memory (video_mode=0)")
        slm = Santec(slm_number=args.slm_number, wavelength=args.wavelength, video_mode=0)
        slm.open()
        panel = (int(slm.Panel_Res[1]), int(slm.Panel_Res[0]))
        if batch is not None and (int(batch["panel"][1]), int(batch["panel"][0])) != panel:
            raise ValueError(f"--plan panel={batch['panel']} 与设备面板 {panel} 不一致")
        rotator = SlotRotator(slm)

        def display(index: int, gray: np.ndarray) -> dict[str, Any]:
            slot = rotator.display_data(slm, gray, wait_time_s=0.0)
            device_log.note("display", frame_index=index, slot=slot)
            return {"slot": slot}

        for order, item in enumerate(run_items):
            label = item["label"]
            tr_plan: TransitionPlan = item["plan"]
            device_log.note("transition_begin", label=label, order=order)
            if sampler is not None and sampler.is_alive():
                stop.set()
                sampler.join(timeout=2.0)
                sampler = None
            if order > 0 and args.inter_transition_settle_s > 0:
                device_log.note(
                    "inter_transition_settle", label=label,
                    seconds=args.inter_transition_settle_s,
                )
                time.sleep(args.inter_transition_settle_s)
            try:
                mask_spec = item["target_mask_path"]
                target_mask = (
                    np.asarray(load_npy(Path(mask_spec["path"]), "target-mask"), dtype=bool)
                    if isinstance(mask_spec, dict)
                    else (
                        np.asarray(load_npy(mask_spec, "target-mask"), dtype=bool)
                        if mask_spec is not None else None
                    )
                )
                if batch is None:
                    start_gray = endpoint_gray(slm, args, "start", panel, device_log)
                    target_gray = endpoint_gray(slm, args, "target", panel, device_log)
                else:
                    start_gray = _plan_endpoint_gray(
                        slm, item["start"], "start", panel, device_log
                    )
                    target_gray = _plan_endpoint_gray(
                        slm, item["target"], "target", panel, device_log
                    )
                est = estimate_transition(tr_plan, start_gray, target_gray)
                max_measurements = math.ceil(tr_plan.duration_s * tr_plan.fps) + 2
                tr_dir = out / label if batch is not None else out
                if batch is not None:
                    tr_dir.mkdir(parents=True, exist_ok=True)
                    _write_json(tr_dir / "transition.json", {
                        "label": label,
                        "group": next(
                            t["group"] for t in batch["transitions"] if t["label"] == label
                        ),
                        "start": item["start"],
                        "target": item["target"],
                        "target_mask": mask_spec,
                        "frame_count": tr_plan.frame_count,
                        "duration_s": tr_plan.duration_s,
                        "fps": tr_plan.fps,
                        "easing": tr_plan.easing,
                        "measure": batch["measure"],
                        "exposure_ms": batch["exposure_ms"],
                        "bit_depth": batch["bit_depth"],
                        "max_measurements": max_measurements,
                    })
                stack: RawFrameStack | None = None
                if args.raw_frames == "stack":
                    # 原始帧保持 CCD 原生全幅（冻结计划禁止物理开窗）；
                    # 传入 None 表示幅面由首个真实相机帧决定，不用 SLM 面板尺寸。
                    stack = RawFrameStack(
                        tr_dir / "frames.npy", max_measurements, None,
                        cam=cam,
                        bit_depth=int(batch["bit_depth"] if batch else args.bit_depth),
                        exposure_ms=(batch["exposure_ms"] if batch else args.exposure_ms),
                        requested_measure_every=(
                            batch["measure"]["every_n_frames"] if batch else args.measure_every
                        ),
                    )
                # 每个 transition 重建采样线程与 stop/state：绝不跨 transition 复用旧状态。
                stop = threading.Event()
                state = SharedState()
                sampler = CCDSampler(
                    cam, state,
                    strategy=(batch["measure"]["strategy"] if batch else args.measure),
                    sink=sink,
                    device_log=device_log,
                    every_n=(
                        batch["measure"]["every_n_frames"] if batch else args.measure_every
                    ),
                    interval_s=args.measure_interval_s,
                    core_half=args.core_half,
                    target_mask=target_mask,
                    uniformity_min=args.uniformity_min,
                    stop=stop,
                    raw_stack=stack,
                )
                t_start = time.monotonic()
                sampler.start()
                frames = dispatch_frames(
                    tr_plan, start_gray, target_gray,
                    display=display, state=state, sink=sink,
                )
                dispatch_s = time.monotonic() - t_start
                sampler.join(timeout=5.0)
                if sampler.is_alive():
                    stop.set()
                    sampler.join(timeout=2.0)
                    logger.warning("CCD 采样线程未在 5 s 内结束，已发停止信号")
                measured = sum(
                    1 for r in sink.records if r.get("event") == "measure"
                )
                if batch is None:
                    measured = sum(1 for r in sink.records if r.get("event") == "measure")
                    summary.update(
                        executed=True,
                        exposure_ms=exposure,
                        panel=list(panel),
                        frames_dispatched=len(frames),
                        frames_measured=measured,
                        dispatch_wall_s=dispatch_s,
                        sampler_alive=sampler.is_alive(),
                        device_lock=lock_evidence,
                        max_code_change_per_frame=est.max_step_gray,
                        expected_total_duration_s=est.expected_duration_s,
                    )
                else:
                    meta: dict[str, Any] = {}
                    if stack is not None:
                        meta = stack.finalize(
                            slm_frames_dispatched=len(frames), elapsed_s=dispatch_s
                        )
                        measured = meta["n_measured"]
                    run_items[order].update(
                        status="ok",
                        frames_dispatched=len(frames),
                        frames_measured=measured,
                        achieved_measure_fps=meta.get("achieved_measure_fps"),
                        dispatch_wall_s=round(dispatch_s, 6),
                        max_code_change_per_frame=est.max_step_gray,
                    )
                device_log.note("transition_end", label=label, status="ok")
            except Exception as exc:
                device_log.note("transition_end", label=label, status="failed", error=str(exc))
                if batch is None:
                    raise
                run_items[order].update(
                    status="failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
                if args.on_error == "abort":
                    break
        if batch is not None:
            _write_json(out / "run_summary.json", [
                {
                    "label": r["label"],
                    "frames": r["plan"].frame_count,
                    "frames_dispatched": r.get("frames_dispatched", 0),
                    "frames_measured": r.get("frames_measured", 0),
                    "achieved_measure_fps": r.get("achieved_measure_fps"),
                    "elapsed_s": r.get("dispatch_wall_s", 0.0),
                    "status": r.get("status", "skipped"),
                    "error": r.get("error"),
                }
                for r in run_items
            ])
            summary.update(
                executed=True,
                batch=True,
                exposure_ms=exposure,
                panel=list(panel),
                device_lock=lock_evidence,
                transitions=len(run_items),
            )
        return 0 if all(r.get("status", "skipped") != "failed" for r in run_items) else 1
    finally:
        stop.set()
        if sampler is not None and sampler.is_alive():
            sampler.join(timeout=2.0)
        if slm is not None:
            try:
                slm.close()
                device_log.note("close", device=f"slm#{args.slm_number}")
            except Exception as exc:  # pragma: no cover - 保护主异常
                device_log.note("close_error", device="slm", error=str(exc))
        if cam is not None:
            try:
                cam.close()
                device_log.note("close", device=f"camera:{args.camera}")
            except Exception as exc:  # pragma: no cover - 保护主异常
                device_log.note("close_error", device="camera", error=str(exc))
        release_lock()
        device_log.flush()
        sink.close()
        _write_json(out / "summary.json", summary)


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    if args.plan is not None:
        batch = load_plan(args.plan)
        if not args.execute:
            _print_batch_plan(argv, batch, args.out)
            return 0
        return execute(args, {
            "execute": True,
            "schema": batch["schema"],
            "plan_source": batch["source"],
        })
    plan = build_plan(args)
    panel = (int(args.panel_w), int(args.panel_h))
    if not args.execute:
        start_gray = offline_gray(args, "start", panel)
        target_gray = offline_gray(args, "target", panel)
        print_plan(argv, args, summarize(args, plan, start_gray, target_gray))
        return 0
    return execute(args, summarize(args, plan, None, None))


if __name__ == "__main__":
    raise SystemExit(main())
