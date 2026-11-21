"""P3 上游资格方向门 holdout 采集：±Z5..Z11 @ 0.05 rad，两次 pass。

用途（2026-11-19）：方向门（`direction_holdout_gate_20261119.py`）需要**未参与
拟合**的单模相位对来判"模型预测方向 vs 实测方向"的 Pearson。本脚本一次采集
一个 pass，共 14 个 holdout 探针（7 个模式 × ±0.05 rad），split 一律 ``test``，
另加标定段（``reference_start`` + 带符号 ``ramp_x/ramp_y``）与 ``reference_end``。

实现沿用 ``scripts/slm/zernike_probe_hardware.py`` 的做法：不重写硬件流程，而是
patch ``capture.probes`` 交出一份自定义探针序列，其余（曝光阶梯、清单、安全
关断）全部交给生产 ``transfer/capture.py``。因此数据集布局与
``transfer/fit.load_dataset`` 完全一致：

- 标定行：``reference_start`` + ``ramp_*``（来自 ``capture.probes``）+ ``reference_end``
- holdout 行：``holdout_{i:03d}_Z{mode}_{sign}{amp:.2f}``，split=test

默认只打印计划（plan-only），不接触任何硬件；``--execute`` 才会打开 SLM/CCD。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from ao_shaping.tools.slm.transfer import capture
from ao_shaping.utils.wavefront.zernike_utils import generate_zernike_phase

LOCK_EVIDENCE_NAME = "device_lock.json"

# 冻结的采集口径（可被 CLI 覆盖，但这些默认值就是计划冻结的取值）。
DEFAULT_MODES: tuple[int, ...] = (5, 6, 7, 8, 9, 10, 11)
DEFAULT_AMP_RAD = 0.05


def _utc_now() -> str:
    """返回微秒精度的 UTC 时间戳字符串。"""
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def write_lock_evidence(
    out: Path,
    wait_until_idle_s: float,
    device_wait: dict[str, object],
    lock_record: dict[str, object],
) -> Path | None:
    """把设备锁的等待/获取/释放证据写到 ``config.json`` 旁边。

    ``capture.capture`` 独占 ``config.json``（``exist_ok=False``），所以证据单独
    成文件，绝不改动 capture 的配置。尽力而为：目录不可写只告警，不抛出，以免
    掩盖采集本身的异常。
    """
    doc = {
        "schema": "direction_holdout_capture_20261119/device_lock/1",
        "wait_until_idle_s": wait_until_idle_s,
        "device_wait": device_wait,
        "lock": lock_record,
        "written_utc": _utc_now(),
    }
    path = out / LOCK_EVIDENCE_NAME
    try:
        path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    except OSError as exc:
        print(
            f"WARNING: could not write lock evidence {path}: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return None
    return path


def build_plan(
    modes: tuple[int, ...] | list[int] = DEFAULT_MODES,
    amp: float = DEFAULT_AMP_RAD,
) -> list[tuple[str, str, int, float]]:
    """构造 holdout 探针计划，返回 (label, split, mode, amp) 四元组列表。

    顺序固定为：对每个模式先 +amp 再 -amp（与 ``direction_holdout_gate`` 的
    ``_Amp`` 配对正则兼容，标签以 ``_Z{mode}_{sign}{amp:.2f}`` 结尾）。
    """
    rows: list[tuple[str, str, int, float]] = []
    index = 0
    for mode in modes:
        for signed_amp in (amp, -amp):
            rows.append(
                (
                    f"holdout_{index:03d}_Z{int(mode)}_{signed_amp:+.2f}",
                    "test",
                    int(mode),
                    float(signed_amp),
                )
            )
            index += 1
    return rows


def make_probes(plan: list[tuple[str, str, int, float]], seed: int):
    """返回替换 ``capture.probes`` 的探针生成器工厂。

    标定段与 ``zernike_probe_hardware.make_probes`` 逐字一致：取
    ``capture.probes(size, 0, seed, periods)[:-1]``（即 ``reference_start`` 加
    带符号 ramp），末尾再补一条零相位 ``reference_end``。
    """
    original = capture.probes

    def probes(size: int, count: int, seed_arg: int, periods=(64, 128)):
        # 标定段：reference_start + 带符号 ramp（丢掉原序列末尾的 reference_end）。
        yield from list(original(size, 0, seed_arg, periods))[:-1]
        for label, split, mode, amp in plan:
            phase = generate_zernike_phase({int(mode): float(amp)}, (size, size))
            phase = np.nan_to_num(np.asarray(phase, dtype=np.float32), nan=0.0)
            yield label, split, phase
        yield "reference_end", "calibration", np.zeros((size, size), np.float32)

    return probes


def build_plan_doc(args: argparse.Namespace, plan: list[tuple[str, str, int, float]]) -> dict:
    """组装写入 ``probe_plan.json`` 的计划文档。"""
    return {
        "schema": "direction_holdout_capture_20261119/1",
        "grid": args.grid,
        "modes": [int(m) for m in args.modes],
        "amp_rad": float(args.amp),
        "frames": args.frames,
        "seed": args.seed,
        "n_holdout": len(plan),
        "plan": [
            {"label": label, "split": split, "mode": int(mode), "amp_rad": float(amp)}
            for label, split, mode, amp in plan
        ],
    }


def main() -> None:
    """CLI 入口：默认 plan-only，``--execute`` 才打开硬件。"""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--grid", type=int, default=128)
    parser.add_argument("--modes", type=int, nargs="+", default=list(DEFAULT_MODES))
    parser.add_argument("--amp", type=float, default=DEFAULT_AMP_RAD)
    parser.add_argument("--frames", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1104229)
    parser.add_argument(
        "--exposure-max-steps",
        type=int,
        default=2,
        help="曝光阶梯的最大步数（传给 capture）。",
    )
    parser.add_argument("--camera", default="daheng")
    parser.add_argument("--cam-id", type=int, default=0)
    parser.add_argument("--slm-number", type=int, default=1)
    parser.add_argument(
        "--wait-until-idle",
        type=float,
        default=120.0,
        metavar="SECONDS",
        help=(
            "打开硬件前（仅 --execute）最多等待 SECONDS 秒让设备锁/其它硬件进程释放，"
            "随后在整个采集期间持有 device_mutex 锁（默认 120）。"
            "传 0 同时关闭等待与加锁（仓库约定：0 = 关）。"
        ),
    )
    parser.add_argument(
        "--exposure-ms",
        type=float,
        default=None,
        metavar="MS",
        help="交给 capture 的起始曝光（ms）；省略则用相机记住的有效曝光。",
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    if not math.isfinite(args.wait_until_idle) or args.wait_until_idle < 0:
        parser.error("--wait-until-idle 必须是非负有限数")
    if args.amp <= 0:
        parser.error("--amp 必须为正数（符号由计划自动展开为 ±）")

    plan = build_plan(args.modes, args.amp)
    plan_doc = build_plan_doc(args, plan)

    if not args.execute:
        for label, split, mode, amp in plan:
            print(f"{split:12s} Z{mode:<3d} {amp:+.2f} {label}")
        print(
            f"\nPLAN ONLY: {len(plan)} holdout probes + 标定段"
            "（reference_start, ramps, reference_end）。"
        )
        print(f"Resolved --out: {args.out}")
        print("Re-run with --execute to open hardware and acquire.")
        print(
            f"Device lock: --wait-until-idle {args.wait_until_idle:g}s "
            f"(default 120s; 0 disables the wait and the lock)."
        )
        print(
            "NOTE: --execute requires a fresh --out directory (capture uses exist_ok=False)."
        )
        return

    # capture() 自己创建 args.out（exist_ok=False），此处不得预先建目录。
    namespace = SimpleNamespace(
        out=args.out,
        execute=True,
        camera=args.camera,
        cam_id=args.cam_id,
        slm_number=args.slm_number,
        grid=args.grid,
        ramp_periods=(64, 128),
        count=0,
        frames=args.frames,
        seed=args.seed,
        exposure_max_steps=args.exposure_max_steps,
        exposure_ms=args.exposure_ms,
        save_dataset=True,
        device=None,
        checkpoint=None,
    )

    # ---- 延迟导入设备驱动（这是唯一导入驱动的处） --------------------------- #
    from ao_shaping.tools.kernels import device_mutex
    from ao_shaping.tools.kernels.device_mutex import DeviceLock

    # ---- 设备锁（整个采集期间持有） ---------------------------------------- #
    lock = None
    device_wait: dict[str, object]
    if args.wait_until_idle > 0:
        wait_kwargs: dict[str, object] = {
            "slm_number": args.slm_number,
            "cam_id": args.cam_id,
            "cam_type": args.camera,
            "own_paths": [Path(__file__).resolve()],
        }
        # 指定了起始曝光时用它做探测；否则让 wait_until_idle 用自己的默认值。
        if args.exposure_ms is not None:
            wait_kwargs["exposure_ms"] = float(args.exposure_ms)
        verdict = device_mutex.wait_until_idle(args.wait_until_idle, **wait_kwargs)
        device_wait = {
            "budget_s": args.wait_until_idle,
            "status": verdict.status,
            "waited_s": round(verdict.waited_s, 3),
            "own_process_tree": verdict.holder_is_me,
            "holder_pid": verdict.holder.pid if verdict.holder is not None else None,
            "holder_cmdline": (
                verdict.holder.cmdline if verdict.holder is not None else None
            ),
            "detail": verdict.detail,
            "remedy": verdict.remedy,
        }
        if not verdict.ok:
            headline = verdict.reason or verdict.detail or verdict.status
            raise SystemExit(
                f"device unavailable ({verdict.kind or verdict.status}): "
                f"{headline} -- {verdict.remedy}"
            )
        # wait_until_idle 在把我们清场后会把已拿到的锁交回来；否则自己取共享锁。
        if verdict.lock is not None:
            lock = verdict.lock
        else:
            lock = DeviceLock.default()
            if not lock.acquire(timeout=0.0):
                raise SystemExit(
                    "device lock could not be acquired right after the idle wait; "
                    "another session took it — rerun (or wait for it to finish)"
                )
    else:
        device_wait = {
            "budget_s": 0.0,
            "status": "skipped",
            "waited_s": 0.0,
            "own_process_tree": False,
            "holder_pid": None,
            "holder_cmdline": None,
            "detail": "waiting disabled (--wait-until-idle 0)",
            "remedy": "",
        }

    lock_record: dict[str, object] = {
        "path": str(lock.path) if lock is not None else None,
        "held": lock.is_locked() if lock is not None else False,
        "acquired_utc": _utc_now() if lock is not None else None,
        "held_s": None,
        "released": None,
        "released_utc": None,
        "release_error": None,
    }

    def release_lock() -> None:
        """释放设备锁，并把持有/释放结果写进证据记录。"""
        if lock_record["acquired_utc"] is not None:
            lock_record["held_s"] = round(
                (
                    datetime.now(timezone.utc)
                    - datetime.fromisoformat(str(lock_record["acquired_utc"]))
                ).total_seconds(),
                3,
            )
        if lock is None:
            lock_record["released_utc"] = _utc_now()
            return
        try:
            lock.release()
            lock_record["released"] = True
        except Exception as exc:  # pragma: no cover - 面向操作者的保护
            lock_record["release_error"] = f"{type(exc).__name__}: {exc}"
            print(
                f"WARNING: device lock release failed: {lock_record['release_error']}",
                file=sys.stderr,
            )
        finally:
            lock_record["released_utc"] = _utc_now()

    try:
        with patch.object(capture, "probes", make_probes(plan, args.seed)):
            capture.capture(namespace)
    finally:
        release_lock()
        write_lock_evidence(args.out, args.wait_until_idle, device_wait, lock_record)
    (args.out / "probe_plan.json").write_text(
        json.dumps(plan_doc, indent=2), encoding="utf-8"
    )
    print(f"Acquisition complete -> {args.out}")


if __name__ == "__main__":
    main()
