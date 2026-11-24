"""闭环 refit A/B–ABBA 主实验的相位回放外壳 (带共享设备锁)。

.. warning::
   本脚本是**忠实重建**: 原文件被并发会话删除且经核查从 git 不可恢复,
   现按文档化 CLI 契约重建 (``docs/dev/hw_capture_script_inventory_20261124.md``
   第 56 行: ``--phase-a`` / ``--phase-b`` / ``--mask-npz`` / ``--out`` /
   ``--execute`` / ``--bit-depth`` / ``--wait-until-idle``), **实验前需重新验证**。

用途
----
闭环 refit 的 A/B–ABBA 主实验
(``report/shape_trials/physical_refine/closed_loop_refit_abba_plan_20261124.md``):
在同一设备会话内按 **A → B → B → A** 顺序, 依次在 SLM 上显示
``--phase-a`` / ``--phase-b`` 相位并用 CCD 各采集一帧, 以交错序列抵消漂移。
A 臂 = 默认配置相位, B 臂 = refit 配置相位。本脚本只做回放与记录,
不做配对判定 (判定由离线分析按冻结口径完成)。

安全语义 (与 ``validate_transfer_shapes_hardware.py`` 同一条外壳约定)
---------------------------------------------------------------------
1. 没有 ``--execute`` 时**完全不碰设备**: 不等待、不加锁、不导入任何设备模块,
   只打印拒绝原因并列出所需参数;
2. ``--execute`` 时才懒导入 ``ao_shaping.tools.kernels.device_mutex`` 与设备驱动;
3. ``--wait-until-idle`` 秒数内等不到设备空闲就**不启动**; 锁一直握到回放结束,
   在 ``finally`` 里释放 (释放失败只记录, 不掩盖回放退出码);
   ``--wait-until-idle 0`` 是仓库约定: 既不等待也不加锁, 证据里记为 disabled;
4. ``--out`` 必填且目录**不可复用** (``exist_ok=False``); 设备锁证据
   ``device_lock.json`` 写入 ``--out``; 若失败发生在建目录之前, 改写同级
   ``<out>.device_lock.json``, 绝不留下空运行目录;
5. ``--bit-depth`` 缺省为 ``None`` = 驱动默认位深, 满量程按生效位深解析
   (不绑帧 dtype)。

证据
----
``--out`` 下: ``captures/<index>_<label>.npz`` 四帧 + ``summary.json``
(schema ``replay_closed_loop_abba/summary/1``): A-B-B-A 顺序、每级相位文件
sha256、每级质量指标 (mean/std/CV/peak/饱和标志/掩膜内能量占比)、
``released.json`` (设备关闭证明) 与 ``device_lock.json``。
不新增任何契约之外的指标或门禁。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

#: 锁包络 (等待记录 / 释放 / 证据写入) 复用验证器外壳的同一份实现。
import validate_transfer_shapes_hardware as _envelope

CAMERA_TYPE = "daheng"
CAM_ID = 0
SLM_NUMBER = 1
SUMMARY_NAME = "summary.json"
SUMMARY_SCHEMA = "replay_closed_loop_abba/summary/1"
SEQUENCE = ("A", "B", "B", "A")
DEFAULT_WAIT_S = _envelope.DEFAULT_WAIT_S


def _utc_now() -> str:
    """与其它硬件脚本一致的 UTC 时间戳格式。"""
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def build_parser() -> argparse.ArgumentParser:
    """ABBA 回放 CLI 契约 (重建口径, 见模块 docstring)。"""
    parser = argparse.ArgumentParser(
        description="闭环 refit A/B–ABBA 相位回放 (A→B→B→A), --execute 才开设备",
        allow_abbrev=False,
    )
    parser.add_argument("--phase-a", type=Path, required=True,
                        help="A 臂相位文件 (npz 或 npy, 弧度制)")
    parser.add_argument("--phase-b", type=Path, required=True,
                        help="B 臂相位文件 (npz 或 npy, 弧度制)")
    parser.add_argument("--mask-npz", type=Path, default=None,
                        help="可选掩膜 npz (core/ROI); 提供时统计掩膜内指标")
    parser.add_argument("--out", type=Path, required=True,
                        help="证据输出目录, 必须不存在 (不可复用)")
    parser.add_argument("--execute", action="store_true",
                        help="设备门: 只有给出它才会等待/加锁并打开 SLM+CCD")
    parser.add_argument("--bit-depth", type=int, default=None,
                        help="相机位深; 缺省 None = 驱动默认")
    parser.add_argument("--wait-until-idle", type=float, default=DEFAULT_WAIT_S,
                        help="在飞保护等待秒数 (0 = 既不等待也不加锁)")
    return parser


def _sha256_file(path: Path) -> str:
    """相位文件字节的 sha256 (证据口径)。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_phase(path: Path) -> np.ndarray:
    """加载 npz/npy 相位; npz 优先取 ``phase`` 键, 否则取第一个数组。"""
    if not path.is_file():
        raise FileNotFoundError(f"相位文件不存在: {path}")
    data = np.load(path)
    if isinstance(data, np.lib.npyio.NpzFile):
        with data:
            key = "phase" if "phase" in data.files else data.files[0]
            arr = np.asarray(data[key], dtype=np.float64)
    else:
        arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"相位必须是 2D 数组, {path} 得到 shape={arr.shape}")
    return arr


def _load_mask(path: Path) -> np.ndarray:
    """加载掩膜 npz; 优先 ``mask`` 键, 否则第一个数组, 转 bool。"""
    with np.load(path) as data:
        key = "mask" if "mask" in data.files else data.files[0]
        mask = np.asarray(data[key])
    return mask.astype(bool)


def _frame_metrics(
    frame: np.ndarray, full_scale: float, mask: np.ndarray | None
) -> dict[str, Any]:
    """每级质量指标 (最小集合): mean/std/CV/peak/饱和标志/掩膜能量占比。"""
    values = frame.astype(np.float64)
    region = values[mask] if mask is not None else values.ravel()
    mean = float(region.mean())
    std = float(region.std())
    peak = float(values.max())
    metrics: dict[str, Any] = {
        "mean": mean,
        "std": std,
        "cv": std / mean if mean > 0 else float("nan"),
        "peak": peak,
        "saturated": bool(peak >= full_scale),
    }
    if mask is not None:
        total = float(values.sum())
        metrics["mask_energy_fraction"] = (
            float(values[mask].sum()) / total if total > 0 else float("nan")
        )
    return metrics


def _resolve_full_scale(cam: Any, frame: np.ndarray, bit_depth: int | None) -> float:
    """满量程: 生效位深优先, 其次 CLI 位深, 兜底 255 (不绑帧 dtype)。"""
    effective = getattr(cam, "effective_full_scale", None)
    if callable(effective):
        try:
            return float(effective(frame))
        except Exception:
            pass
    if bit_depth is not None:
        return float((1 << int(bit_depth)) - 1)
    return 255.0


def _run_abba(args: argparse.Namespace) -> None:
    """设备体: 打开一次 SLM+CCD, 按 A→B→B→A 显示相位并逐帧采集。"""
    # 懒导入: 只有 --execute 且锁已拿到之后才触达设备模块。
    from ao_shaping.drivers.ccd.common import create_camera
    from ao_shaping.drivers.slm.santec import Santec

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=False)
    captures_dir = out_dir / "captures"
    captures_dir.mkdir()

    phases = {"A": _load_phase(args.phase_a), "B": _load_phase(args.phase_b)}
    phase_sha = {"A": _sha256_file(args.phase_a), "B": _sha256_file(args.phase_b)}
    phase_src = {"A": str(args.phase_a), "B": str(args.phase_b)}
    mask = _load_mask(args.mask_npz) if args.mask_npz is not None else None

    cam_kwargs: dict[str, Any] = {}
    if args.bit_depth is not None:
        cam_kwargs["bit_depth"] = int(args.bit_depth)
    cam = create_camera(CAMERA_TYPE, cam_id=CAM_ID, **cam_kwargs)
    slm = Santec(slm_number=SLM_NUMBER, video_mode=0)
    cam_open = False
    slm_open = False
    stages: list[dict[str, Any]] = []
    try:
        cam.open()
        cam_open = True
        slm.open()
        slm_open = True
        require = getattr(slm, "require_drive_board", None)
        if callable(require):
            require()
        for index, label in enumerate(SEQUENCE):
            slm.display_phase(phases[label])
            frame = np.asarray(cam.get_numpy_image(1))
            if mask is not None and mask.shape != frame.shape:
                raise ValueError(
                    f"掩膜尺寸 {mask.shape} 与帧 {frame.shape} 不一致"
                )
            frame_name = f"{index}_{label}.npz"
            np.savez_compressed(captures_dir / frame_name, frame=frame)
            full_scale = _resolve_full_scale(cam, frame, args.bit_depth)
            stages.append({
                "index": index,
                "label": label,
                "phase_source": phase_src[label],
                "phase_sha256": phase_sha[label],
                "frame_file": f"captures/{frame_name}",
                "captured_at": _utc_now(),
                "metrics": _frame_metrics(frame, full_scale, mask),
            })
            logger.info("ABBA 第 {} 级 ({}) 采集完成: peak={}",
                        index, label, stages[-1]["metrics"]["peak"])
    finally:
        released = {"camera_closed": False, "slm_closed": False}
        if cam_open:
            try:
                cam.close()
                released["camera_closed"] = True
            except Exception as exc:
                logger.warning("相机关闭失败: {}", exc)
        if slm_open:
            try:
                slm.close()
                released["slm_closed"] = True
            except Exception as exc:
                logger.warning("SLM 关闭失败: {}", exc)
        released["released_at"] = _utc_now()
        try:
            (out_dir / "released.json").write_text(
                json.dumps(released, indent=2), encoding="utf-8"
            )
        except Exception as exc:
            logger.warning("released.json 写入失败: {}", exc)

    summary = {
        "schema": SUMMARY_SCHEMA,
        "created_at": _utc_now(),
        "sequence": list(SEQUENCE),
        "bit_depth": args.bit_depth,
        "wait_until_idle_s": args.wait_until_idle,
        "mask_npz": str(args.mask_npz) if args.mask_npz is not None else None,
        "stages": stages,
        "rebuilt_notice": (
            "按文档化 CLI 契约重建 (hw_capture_script_inventory_20261124), "
            "上机前需重新验证"
        ),
    }
    (out_dir / SUMMARY_NAME).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _run_execute(args: argparse.Namespace) -> int:
    """等待、持锁跑 ABBA 回放并记录证据 (与验证器外壳同一协议)。"""
    # 本脚本唯一的设备侧模块入口: 无 --execute 的路径永远到不了这里。
    from ao_shaping.tools.kernels import device_mutex

    doc: dict[str, Any] = {
        "schema": _envelope.SCHEMA,
        "script": Path(__file__).name,
        "phase_a": str(args.phase_a),
        "phase_b": str(args.phase_b),
        "mask_npz": str(args.mask_npz) if args.mask_npz is not None else None,
        "bit_depth": args.bit_depth,
        "wait_until_idle_s": float(args.wait_until_idle),
        "lock_mode": "wait+acquire" if args.wait_until_idle > 0 else "disabled",
        "device_wait": None,
        "lock": {
            "path": None,
            "held": False,
            "acquired_utc": None,
            "released_utc": None,
            "held_s": None,
            "released": None,
            "release_error": None,
        },
        "replay_invoked": False,
        "replay_error": None,
        "written_utc": None,
    }
    lock = None
    exit_code = 0
    try:
        if args.wait_until_idle > 0:
            try:
                verdict = device_mutex.wait_until_idle(
                    args.wait_until_idle,
                    slm_number=SLM_NUMBER,
                    cam_id=CAM_ID,
                    cam_type=CAMERA_TYPE,
                    own_paths=[Path(__file__).resolve()],
                )
            except Exception as exc:
                doc["device_wait"] = {
                    "budget_s": float(args.wait_until_idle),
                    "status": "error",
                    "detail": f"{type(exc).__name__}: {exc}",
                }
                print(f"ERROR: 设备空闲等待失败, 回放未启动: "
                      f"{type(exc).__name__}: {exc}", file=sys.stderr)
                return 1
            doc["device_wait"] = _envelope._wait_record(
                verdict, args.wait_until_idle
            )
            if not bool(getattr(verdict, "ok", False)):
                print(
                    f"ERROR: 设备不可用, 回放未启动: "
                    f"{_envelope._verdict_headline(verdict)}",
                    file=sys.stderr,
                )
                return 1
            if getattr(verdict, "lock", None) is not None:
                lock = verdict.lock
            else:
                try:
                    lock = device_mutex.DeviceLock.default()
                    acquired = lock.acquire(timeout=0.0)
                except Exception as exc:
                    doc["lock"]["path"] = (
                        str(lock.path) if lock is not None else None
                    )
                    print(f"ERROR: 设备锁获取失败, 回放未启动: "
                          f"{type(exc).__name__}: {exc}", file=sys.stderr)
                    lock = None
                    return 1
                if not acquired:
                    doc["lock"]["path"] = str(lock.path)
                    print("ERROR: 等待后设备锁仍被占用, 回放未启动",
                          file=sys.stderr)
                    lock = None
                    return 1
            doc["lock"]["path"] = str(lock.path)
            doc["lock"]["held"] = True
            doc["lock"]["acquired_utc"] = _utc_now()
        else:
            doc["device_wait"] = _envelope._disabled_wait_record()

        doc["replay_invoked"] = True
        try:
            _run_abba(args)
        except Exception as exc:
            doc["replay_error"] = f"{type(exc).__name__}: {exc}"
            print(f"ERROR: ABBA 回放失败: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
            exit_code = 1
        return exit_code
    finally:
        _envelope._release_lock(lock, doc["lock"])
        _envelope._write_evidence(args.out, doc)


def main(argv: list[str] | None = None) -> int:
    """入口: 无 --execute 拒绝并列出所需参数; 有则走共享锁包络。"""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not math.isfinite(args.wait_until_idle) or args.wait_until_idle < 0:
        parser.error("--wait-until-idle 必须是有限的非负数")

    if not args.execute:
        print("拒绝: 未给 --execute, 不打开任何设备。")
        print("真实回放需要: --phase-a/--phase-b/--out [--mask-npz] "
              "[--bit-depth] [--wait-until-idle] --execute")
        parser.print_help()
        return 2

    if args.out.exists():
        print(f"拒绝: --out 目录已存在, 不允许复用: {args.out}")
        return 2

    return _run_execute(args)


if __name__ == "__main__":
    raise SystemExit(main())
