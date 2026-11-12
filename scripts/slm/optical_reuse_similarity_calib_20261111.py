"""Live calibration: is the flat-phase spot-shape similarity a real reuse gate?

Question
--------
光路复用原本只判「零点漂移 <= 5 px」。实测同一条光路的重复取帧零点能差 16 px
(2026-11-05), 于是 5 px 闸门把同一条光路判成「换了」; 而反过来换了光路时零点
差异可能很小。本脚本要在**当前台面**上量出新的形状判据
(:func:`ao_shaping.tools.slm.transfer.optical_probe.compare_flat_frames`) 的两个
分布, 并检查阈值 :data:`DEFAULT_MIN_FLAT_SIMILARITY` 是否落在中间:

* **正样本** —— 本次会话内重复取帧 (同一光路, 含真实抖动与零点漂移);
* **负样本** —— 与磁盘上 2026-11-05 / 2026-11-02 的平场帧比, 那些是**另一条**
  光路 (零点在 (1377, 637) / 不同 ROI)。

判据全部走生产代码: ``compare_flat_frames`` / ``assess_reuse`` / ``save_flat_frame``
—— 脚本不另写一份相似度实现 (scripts 禁止第二份实现)。

Devices
-------
只开 **一次** SLM #1 + 相机 #0, 全程一个进程 (背对背开关会让驱动板假死, 只能
手动重启)。开场做相位自检 (``current_grayscale == 0``); 平场相位 (全零) 下发
后抓帧; 退出即关闭, 并在报告里写明占用与释放。

Outputs
-------
``data/debug/slm_transfer/<YYYYMMDD_HHMMSS>_optical_reuse_calib/``:

* ``raw_frames.npz``: 本次抓的全部单帧 + 参照帧 (3 帧均值);
* ``reference_flat.npz``: 生产 :func:`save_flat_frame` 写下的参照帧 (验证落盘路径);
* ``calib.json``: 每帧的峰值/背景/亚像素零点 + 全部相似度 + 生产判决。

Delete when
-----------
阈值结论已写进 ``optical_probe.DEFAULT_MIN_FLAT_SIMILARITY`` 的注释与 report,
且后续会话不再需要重测这个分布。
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from loguru import logger

REPO = Path(__file__).resolve().parents[2]

#: 负样本: 磁盘上与**当前**光路不同的平场帧 (``(run, 文件, 备注)``)。
STORED_NEGATIVES: list[tuple[str, str, str]] = [
    (
        "20261105_164025",
        "response/flat_probe.npz",
        "2026-11-05 采集段, 零点 (1377, 637) —— 与当前光路不同",
    ),
    (
        "20261105_171947",
        "response/flat_probe.npz",
        "2026-11-05 另一次采集, 零点同上",
    ),
    (
        "20261105_203528",
        "response/flat_probe.npz",
        "2026-11-05 第三次采集, 零点同上",
    ),
    (
        "physical_20261102c",
        "response/flat_probe.npz",
        "2026-11-02 不同相机 ROI (1944x2592), 生产会按尺寸直接拒",
    ),
]

#: 判定一场平场帧「有信号」的最低对比度 (峰/背景), 与采集段 ``assess_spot`` 同量级。
MIN_CONTRAST = 8.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行 (默认离线: 只打印计划, 不开设备)。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="真的打开 SLM #1 + 相机 #0 (不加只打印计划)",
    )
    parser.add_argument("--camera", default="daheng", help="相机后端名")
    parser.add_argument("--cam-id", type=int, default=0, help="相机 id")
    parser.add_argument("--slm-number", type=int, default=1, help="SLM 编号")
    parser.add_argument(
        "--exposure-ms",
        type=float,
        default=0.02,
        help="固定曝光 (ms), 与 measure_probe 的探针帧一致",
    )
    parser.add_argument("--frames", type=int, default=6, help="单帧抓几张")
    parser.add_argument("--ref-frames", type=int, default=3, help="参照帧的均值帧数")
    parser.add_argument(
        "--settle-s", type=float, default=0.5, help="下发平场相位后等多久再抓"
    )
    parser.add_argument("--discard", type=int, default=2, help="先丢掉几张旧帧")
    parser.add_argument(
        "--wait-until-idle",
        type=float,
        default=0.0,
        help="开设备前等空闲的预算 (s); 0 = 不探不动",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="输出目录; 缺省 data/debug/slm_transfer/<ts>_optical_reuse_calib",
    )
    return parser.parse_args(argv)


def default_out_dir() -> Path:
    """本次运行的输出目录 (时间戳命名, 与 slm_transfer 其它 run 同构)。"""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO / "data" / "debug" / "slm_transfer" / f"{stamp}_optical_reuse_calib"


def report_occupancy(phase: str, devices: str) -> None:
    """按 ``docs/dev/hardware_discipline.md`` 报告设备占用 (开始/结束各一次)。"""
    logger.info(f"[设备占用·{phase}] {devices}")


def precheck_bench(budget_s: float) -> dict:
    """只读检查台面: 谁在占用, 需要的话等它空闲。

    不打开任何设备, 只读进程表 / 锁文件。

    Args:
        budget_s: 等空闲的预算 (s); ``0`` 只报告不等。

    Returns:
        可写进 ``calib.json`` 的记录。
    """
    from ao_shaping.tools.kernels.device_mutex import (
        find_device_using_processes,
        format_occupier_lines,
    )

    record: dict = {"budget_s": float(budget_s), "occupiers": [], "wait": None}
    hits = find_device_using_processes(own_paths=[__file__])
    record["occupiers"] = [{"pid": proc.pid, "cmdline": proc.cmdline} for proc in hits]
    if hits:
        logger.warning("台面上有进程可能占着设备:")
        for line in format_occupier_lines(hits):
            logger.warning(line)
    else:
        logger.info("台面上没有别的设备进程 (只读检查)")
    if budget_s > 0:
        from ao_shaping.tools.kernels.device_mutex import wait_until_idle

        verdict = wait_until_idle(budget_s, own_paths=[__file__])
        record["wait"] = {
            "ok": bool(verdict.ok),
            "status": verdict.status,
            "waited_s": float(verdict.waited_s),
            "reason": verdict.reason,
        }
        logger.info(
            f"等空闲: status={verdict.status} waited={verdict.waited_s:.1f}s "
            f"{verdict.reason}"
        )
    return record


def slm_self_check(slm: object) -> dict:
    """开场相位自检: ``get_current_grayscale()`` 必须是 0 (平场/关)。

    按本仓约定, ``get_displayed_memory_number()`` 抛 ``SantecError(code=1)`` 在
    grayscale 模式下属正常 (没有 memory slot); ``get_displayed_phase()`` 只是驱动
    侧写缓存, 不是设备回读。
    """
    from ao_shaping.drivers.slm.santec import SantecError

    record: dict = {}
    get_gray = getattr(slm, "get_current_grayscale", None)
    if get_gray is None:
        record["grayscale"] = None
    else:
        record["grayscale"] = float(get_gray())
        if record["grayscale"] != 0.0:
            raise RuntimeError(
                f"SLM 打开后灰度不是 0 (读回 {record['grayscale']}) —— 先别下发相位"
            )
    slot = None
    get_slot = getattr(slm, "get_displayed_memory_number", None)
    if get_slot is not None:
        try:
            slot = get_slot()
        except SantecError as exc:
            if exc.code != 1:
                raise
            slot = None
    record["memory_slot"] = slot
    get_phase = getattr(slm, "get_displayed_phase", None)
    if get_phase is not None:
        try:
            _, source = get_phase()
            record["phase_source"] = str(source)
        except Exception as exc:  # policy: 缓存读失败不影响自检结论
            logger.debug(f"get_displayed_phase 读不到 (缓存): {exc!r}")
    logger.info(f"SLM 相位自检: {record}")
    return record


def grab(cam: object, count: int) -> np.ndarray:
    """连抓 ``count`` 张**单帧** (不做平均), 返回 ``(count, H, W)``。

    单帧是形状对比里最严格的一侧: 平均会抹掉噪声, 而真实探针帧
    (``measure_probe``) 就是单帧。
    """
    frames = [cam.get_numpy_image(n_sample=1, skip_first=False) for _ in range(count)]
    return np.stack([np.asarray(frame) for frame in frames])


def frame_stats(frame: np.ndarray) -> dict:
    """一帧的峰值 / 背景 / 对比度 / 亚像素零点。"""
    from ao_shaping.tools.slm.transfer.optical_probe import (
        DEFAULT_CENTRE_HALF_WINDOW,
        _flat_centre,
        _prepare_flat,
    )

    data = np.asarray(frame, dtype=np.float64)
    prepared = _prepare_flat(data)
    centre = _flat_centre(prepared, DEFAULT_CENTRE_HALF_WINDOW)
    background = float(np.median(data))
    peak = float(data.max())
    return {
        "peak": peak,
        "background": background,
        "contrast": peak / max(background, 1e-9),
        "saturated_px": int((data >= data.max()).sum()),
        "zero_xy": [float(centre[0]), float(centre[1])],
    }


def compare_pair(current: np.ndarray, reference: np.ndarray) -> dict:
    """两帧的形状比对 (生产实现), 尺寸不同时先各自取对心窗再比。

    ``compare_flat_frames`` 要求两帧尺寸一致 (同一相机同一 ROI), 这是生产语义:
    换 ROI 就不该复用。跨 ROI 的老帧只作趋势参考, 这里先按各自亚像素零点取
    129x129 窗, 再交给同一个函数。

    另报**真实全帧**零点: 尺寸不同时 ``compare_flat_frames`` 是在各自对心窗上算的,
    其 ``drift_px`` 因此≈0 (窗口已各自对心), 真实的全帧零点位移被丢掉了; 这里额外
    用未裁切的全帧重算亚像素零点, 给出 ``current_xy`` / ``reference_xy`` 与二者
    欧氏距离 ``abs_drift_px``, 供跨 ROI 分支读。

    Args:
        current: 一帧平场。
        reference: 另一帧平场。

    Returns:
        ``{"similarity", "drift_px", "window_px", "same_shape", "current_xy",
        "reference_xy", "abs_drift_px"}``; 比对失败时 ``{"error": ...}``。
    """
    from ao_shaping.tools.slm.transfer.optical_probe import (
        DEFAULT_CENTRE_HALF_WINDOW,
        DEFAULT_FLAT_HALF_WINDOW,
        _flat_centre,
        _flat_window,
        _prepare_flat,
        compare_flat_frames,
    )

    a = np.asarray(current, dtype=np.float64)
    b = np.asarray(reference, dtype=np.float64)
    same_shape = a.shape == b.shape
    if not same_shape:
        half = DEFAULT_FLAT_HALF_WINDOW
        pa = _prepare_flat(a)
        pb = _prepare_flat(b)
        a = _flat_window(pa, _flat_centre(pa, DEFAULT_CENTRE_HALF_WINDOW), half)
        b = _flat_window(pb, _flat_centre(pb, DEFAULT_CENTRE_HALF_WINDOW), half)
    try:
        result = compare_flat_frames(a, b)
        # 真实**全帧**亚像素零点 (未裁切): 跨 ROI 时 ``compare_flat_frames`` 是在各自
        # 对心窗上算的, 其 ``drift_px`` 因此≈0, 丢掉了被窗口吃掉的全帧零点位移; 这里
        # 用未裁切的全帧重算, 供跨 ROI 分支读真实位移。
        centre_current = _flat_centre(
            _prepare_flat(current), DEFAULT_CENTRE_HALF_WINDOW
        )
        centre_reference = _flat_centre(
            _prepare_flat(reference), DEFAULT_CENTRE_HALF_WINDOW
        )
    except ValueError as exc:
        return {"error": str(exc), "same_shape": same_shape}
    return {
        "similarity": float(result.similarity),
        "drift_px": [float(result.drift_px[0]), float(result.drift_px[1])],
        "window_px": int(result.window_px),
        "same_shape": bool(same_shape),
        "current_xy": [float(centre_current[0]), float(centre_current[1])],
        "reference_xy": [float(centre_reference[0]), float(centre_reference[1])],
        "abs_drift_px": float(
            np.hypot(
                centre_current[0] - centre_reference[0],
                centre_current[1] - centre_reference[1],
            )
        ),
    }


def load_stored_negative(run: str, relative: str) -> np.ndarray | None:
    """读一张磁盘上的负样本平场帧 (取对比度最高的那一张)。"""
    path = REPO / "data" / "debug" / "slm_transfer" / run / relative
    if not path.is_file():
        logger.warning(f"负样本缺失: {path}")
        return None
    try:
        with np.load(path, allow_pickle=True) as payload:
            raw = np.asarray(payload["raw_frames"], dtype=np.float64)
    except (OSError, KeyError, ValueError) as exc:
        logger.warning(f"负样本读不出 ({path}): {exc!r}")
        return None
    stats = [frame_stats(frame) for frame in raw]
    best = int(np.argmax([item["contrast"] for item in stats]))
    logger.info(f"负样本 {run}: 取第 {best} 张 (对比度 {stats[best]['contrast']:.1f})")
    return raw[best]


def summarize(values: list[float]) -> dict:
    """一组相似度 → min/median/max (空列表返回 None 值)。"""
    if not values:
        return {"n": 0, "min": None, "median": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "min": float(arr.min()),
        "median": float(np.median(arr)),
        "max": float(arr.max()),
    }


def run_device(args: argparse.Namespace, out: Path) -> dict:
    """打开设备一次, 抓帧, 比对, 关设备; 返回可序列化的结果。"""
    from ao_shaping.drivers.ccd.common import create_camera
    from ao_shaping.drivers.slm.santec import Santec
    from ao_shaping.tools.slm.transfer.hardware import (
        opened_device,
        prepare_camera_sdk,
    )
    from ao_shaping.tools.slm.transfer.optical_probe import (
        DEFAULT_MIN_FLAT_SIMILARITY,
        assess_reuse,
        OpticalProbe,
        save_flat_frame,
    )

    prepare_camera_sdk(args.camera)
    cam = create_camera(args.camera, args.cam_id, exposure_time_ms=args.exposure_ms)
    result: dict = {"exposure_ms": float(args.exposure_ms), "steps": []}

    report_occupancy(
        "开始", f"SLM #{args.slm_number} + 相机 {args.camera}#{args.cam_id} (激光常开)"
    )
    closed_normally = False
    try:
        with opened_device(cam) as cam_handle:
            with opened_device(
                Santec(slm_number=args.slm_number, wavelength=1064, video_mode=0)
            ) as slm:
                result["slm_check"] = slm_self_check(slm)
                serial = getattr(slm, "serial_number", None)
                result["slm_serial"] = None if serial is None else str(serial)
                # 平场相位: 与 measure_probe / 采集段一致的全零面板
                slm.set_phase_overlay(None)
                slm.display_data(slm.create_phase_from_array(np.zeros((1200, 1920))))
                time.sleep(float(args.settle_s))
                result["steps"].append(f"下发平场相位, 等 {args.settle_s}s")

                if args.discard > 0:
                    grab(cam_handle, args.discard)
                    result["steps"].append(f"丢弃 {args.discard} 张旧帧")

                singles = grab(cam_handle, int(args.frames))
                reference_block = grab(cam_handle, int(args.ref_frames))
                reference = reference_block.mean(axis=0, dtype=np.float64)
                result["steps"].append(
                    f"抓 {args.frames} 张单帧 + {args.ref_frames} 张参照块"
                )

                # 平场复位 (与 measure_probe / 采集段 finally 一致)
                slm.display_data(slm.create_phase_from_array(np.zeros((1200, 1920))))
        closed_normally = True
    finally:
        report_occupancy(
            "结束",
            f"SLM #{args.slm_number} + 相机 {args.camera}#{args.cam_id} 已释放"
            f" (正常关闭={closed_normally})",
        )

    result["closed_normally"] = closed_normally
    np.savez_compressed(out / "raw_frames.npz", singles=singles, reference=reference)
    flat_name = save_flat_frame(
        out,
        reference,
        spot_xy=tuple(frame_stats(reference)["zero_xy"]),
        exposure_ms=float(args.exposure_ms),
        slm_serial=result.get("slm_serial"),
    )
    result["saved_flat_frame"] = flat_name

    # ---- 正样本: 本次会话内 ----
    stats = [frame_stats(frame) for frame in singles]
    for index, item in enumerate(stats):
        item["index"] = index
    result["frames"] = stats
    logger.info("本次抓帧:")
    for item in stats:
        logger.info(
            f"  f{item['index']:02d} peak={item['peak']:.0f} "
            f"bg={item['background']:.1f} contrast={item['contrast']:.1f} "
            f"zero=({item['zero_xy'][0]:.2f}, {item['zero_xy'][1]:.2f})"
        )

    positives: list[dict] = []
    for i in range(len(singles)):
        for j in range(i + 1, len(singles)):
            if min(stats[i]["contrast"], stats[j]["contrast"]) < MIN_CONTRAST:
                continue
            pair = compare_pair(singles[i], singles[j])
            pair.update({"kind": "single-single", "i": i, "j": j})
            positives.append(pair)
    for i, frame in enumerate(singles):
        if stats[i]["contrast"] < MIN_CONTRAST:
            continue
        pair = compare_pair(frame, reference)
        pair.update({"kind": "single-reference", "i": i, "j": None})
        positives.append(pair)

    # ---- 负样本: 磁盘上的另一条光路 ----
    negatives: list[dict] = []
    for run, _relative, note in STORED_NEGATIVES:
        stored = _stored_cache.get(run)
        if stored is None:
            logger.warning(f"负样本没预读到, 跳过: {run}")
            continue
        pair = compare_pair(reference, stored)
        pair.update({"kind": "stored", "run": run, "note": note})
        negatives.append(pair)
        if "similarity" in pair:
            extra = (
                f" abs_drift={pair['abs_drift_px']:.1f}px"
                if not pair["same_shape"]
                else ""
            )
            logger.info(
                f"  负样本 {run}: sim={pair['similarity']:.3f} "
                f"drift=({pair['drift_px'][0]:+.1f}, {pair['drift_px'][1]:+.1f}) "
                f"same_shape={pair['same_shape']}{extra}"
            )

    # ---- 生产判决: 同样的帧喂给 assess_reuse ----
    def decide(frame_a: np.ndarray, frame_b: np.ndarray) -> dict:
        zero_a = tuple(frame_stats(frame_a)["zero_xy"])
        zero_b = tuple(frame_stats(frame_b)["zero_xy"])
        probe_a = OpticalProbe(zero_a, (0.02, 1000.0), "mono8", 8, "x")
        probe_b = OpticalProbe(zero_b, (0.02, 1000.0), "mono8", 8, "x")
        verdict = assess_reuse(
            probe_a, probe_b, current_frame=frame_a, reference_frame=frame_b
        )
        return {"same": bool(verdict.same), "criteria": verdict.criteria}

    result["positives"] = positives
    result["negatives"] = negatives
    result["production_decision_on_positives"] = [
        {**item, "decision": decide(singles[item["i"]], reference)}
        for item in positives
        if item["kind"] == "single-reference"
    ]
    result["production_decision_on_negatives"] = [
        {"run": item["run"], "decision": decide(reference, _stored_cache[item["run"]])}
        for item in negatives
        if item["run"] in _stored_cache
    ]

    pos = summarize([item["similarity"] for item in positives if "similarity" in item])
    neg = summarize([item["similarity"] for item in negatives if "similarity" in item])
    same_shape_neg = summarize(
        [
            item["similarity"]
            for item in negatives
            if item.get("similarity") is not None and item.get("same_shape")
        ]
    )
    result["summary"] = {
        "threshold": float(DEFAULT_MIN_FLAT_SIMILARITY),
        "positives": pos,
        "negatives": neg,
        "negatives_same_shape": same_shape_neg,
        "separation_ok": bool(
            pos["min"] is not None
            and neg["max"] is not None
            and pos["min"] > neg["max"]
        ),
        "threshold_inside_gap": bool(
            pos["min"] is not None
            and neg["max"] is not None
            and neg["max"] < DEFAULT_MIN_FLAT_SIMILARITY <= pos["min"]
        ),
    }
    return result


#: 负样本帧缓存 (生产判决要重算一次, 免得再读盘)。
_stored_cache: dict[str, np.ndarray] = {}


def load_negative_cache() -> None:
    """把负样本帧预读进模块级缓存。"""
    for run, relative, _note in STORED_NEGATIVES:
        stored = load_stored_negative(run, relative)
        if stored is not None:
            _stored_cache[run] = stored


def print_plan(args: argparse.Namespace, out: Path) -> None:
    """离线模式: 说明将要做什么, 不开设备。"""
    logger.info("离线模式 (未加 --execute): 不会打开任何设备。")
    logger.info(
        f"计划: 开 SLM #{args.slm_number} + 相机 {args.camera}#{args.cam_id} 一次, "
        f"平场相位, 抓 {args.frames} 张单帧 + {args.ref_frames} 张参照块 "
        f"(@{args.exposure_ms} ms), 再与 {len(STORED_NEGATIVES)} 组磁盘负样本比对。"
    )
    logger.info(f"输出目录: {out}")
    existing = [
        path
        for path in (REPO / "data" / "debug" / "slm_transfer").glob(
            "2026*/response/flat_probe.npz"
        )
    ]
    logger.info(f"磁盘上可用作负样本的平场帧: {len(existing)} 个")


def main(argv: list[str] | None = None) -> int:
    """入口: 离线打印计划, ``--execute`` 才动设备。"""
    args = parse_args(argv)
    out = Path(args.out) if args.out else default_out_dir()
    out.mkdir(parents=True, exist_ok=True)

    record: dict = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "argv": [str(item) for item in (argv if argv is not None else [])],
        "precheck": precheck_bench(float(args.wait_until_idle)),
    }
    if not args.execute:
        print_plan(args, out)
        (out / "calib.json").write_text(
            json.dumps({**record, "executed": False}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return 0

    load_negative_cache()
    result = run_device(args, out)
    payload = {**record, **result, "executed": True}
    (out / "calib.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    summary = result["summary"]
    threshold = summary["threshold"]
    logger.info("=== 形状相似度分布 ===")
    logger.info(
        f"正样本 (同光路): n={summary['positives']['n']} "
        f"min={summary['positives']['min']} median={summary['positives']['median']}"
    )
    logger.info(
        f"负样本 (换光路): n={summary['negatives']['n']} "
        f"max={summary['negatives']['max']}"
    )
    logger.info(
        f"阈值 {threshold:.2f} 是否落在两簇之间: {summary['threshold_inside_gap']}"
    )
    logger.info(f"输出: {out / 'calib.json'}")
    return 0 if summary["threshold_inside_gap"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
