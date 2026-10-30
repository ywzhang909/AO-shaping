"""Replay every archived initial/final phase of the 2026-10-29 black-box sessions.

为什么需要这个脚本 (2026-10-29 21:0x)
------------------------------------
两次会话 (``blackbox_path_20261029_194837`` 平相位种 / ``..._gs_20261029_201406``
GS 种) 各跑出三个形状的相位, 但两次的曝光读回相差 36 倍 (0.119 ms vs 4.32 ms),
于是**指标不可比**: 原始 EE / CV 的差距主要由加性背景 (杂散光) 与曝光决定,
而不是两套相位本身。要判定"相位质量是否真的不同", 唯一干净的做法是把两次
会话**存档的同一批相位**在同一曝光下交错重放。

现有入口做不到这件事: ``transfer/replay_phases.py`` 选择器只认 campaign 产物
里的 ``<shape>_seed`` / ``<shape>_final`` 标签, 无法同时取两次会话的种子与终态,
也不支持按组各选一个曝光; ``transfer/capture_response.py`` 只做曝光响应采集。
所以本脚本只做一件事: 从已存 npz 里取出 ``model_phase``, 原路
(``to_panel`` → ``create_phase_from_array`` → ``display_data``) 重放 —— 与
``transfer/validate.py`` 的下发路径逐字节相同, 相位自带载波, **不再叠加任何
载波或静态矫正**。

曝光怎么定
----------
不抄历史值。先用 0.02 ms 对整组相位各读一帧 (安全), 按"传感器对曝光线性"
外推到目标峰值, 再整组验证一遍; 若仍有像素越过上限就按上限/实测峰一次性收缩。
得到两个**各自无饱和**的公共曝光:

* ``common`` —— 覆盖全部相位 (含峰值最高的一族), 用于严格同曝光的跨会话比较;
* ``high``   —— 只覆盖 GS 会话那 6 个扩散型相位, 用于高信噪比复看;
  common 与 high 作用在**同一批相位**上, 因此两者之差就是曝光/加性背景的标度。

设备契约: 只在 ``--execute`` 下开设备; memory 模式 (``video_mode=0``); 先拿
``DeviceLock`` 跨进程锁; 每次下发换内存槽 (``display_data`` 自带轮转); 退出时
关闭设备并释放锁, 释放证据写进 ``release.json``。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]

#: 存档会话: 键是短名, 只用于标签与目录名。
RUNS: dict[str, str] = {
    "r1": "data/shape_trials/blackbox_path_20261029_194837",
    "r2": "data/shape_trials/blackbox_path_gs_20261029_201406",
}
SHAPES = ("square", "triangle", "rectangle")
GRID = 128
FULL_SCALE = 4095.0  # 12-bit 原生位深, 与两次会话的 config.json 一致


def clock() -> dict:
    from datetime import datetime, timezone

    return dict(timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="microseconds"),
                monotonic_ns=time.monotonic_ns())


def sha(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


@dataclass(frozen=True)
class Phase:
    """一条待重放的相位: 标签 + 原生 128x128 控制网格 (含载波) + 出处。"""

    label: str
    group: str          # "flat" / "common" / "high"
    phase: np.ndarray
    source: str
    run: str = ""
    shape: str = ""
    kind: str = ""      # "flat" / "seed" / "final"


def _row(directory: Path, stage: str, label: str) -> tuple[np.ndarray, str]:
    """按 manifest 的 label 取回该次采集的 ``model_phase`` (逐字节原文)。"""
    rows = json.loads((directory / stage / "manifest.json").read_text(encoding="utf-8"))
    row = next(r for r in rows if r["label"] == label)
    with np.load(directory / stage / row["file"], allow_pickle=True) as z:
        phase = np.array(z["model_phase"], dtype=float)
    return phase, str((directory / stage / row["file"]).relative_to(ROOT))


def load_phases() -> tuple[list[Phase], dict[str, dict[str, np.ndarray]]]:
    """取出 1 条平相位参考 + 每形状每会话的种子/终态, 并核对种子出处映射。

    种子不是会话自己的 manifest 行: 它是同一会话 ``model/<shape>_final`` 的那一
    帧 (config.json 的 ``seed_provenance`` 指向 model 目录)。这里**实测相等性**
    而不是相信标签 —— 不相等就直接报错, 不做静默替换。
    """
    grids: dict[str, dict[str, np.ndarray]] = {}
    phases: list[Phase] = [Phase("flat", "flat", np.zeros((GRID, GRID)),
                                 "model grid zeros (panel = flat, no carrier)",
                                 kind="flat")]
    rows = json.loads((ROOT / RUNS["r1"] / "anchored" / "manifest.json").read_text(encoding="utf-8"))
    for shape in SHAPES:
        row = next(r for r in rows if r["label"] == f"{shape}_final")
        with np.load(ROOT / RUNS["r1"] / "anchored" / row["file"], allow_pickle=True) as z:
            grids[shape] = dict(camera_xy=np.array(z["camera_xy"], dtype=float),
                                target=np.array(z["target"], dtype=np.float32),
                                core=np.array(z["core"], dtype=bool))
    for run, directory in RUNS.items():
        root = ROOT / directory
        for shape in SHAPES:
            anchored_seed, seed_file = _row(root, "anchored", f"{shape}_seed")
            model_final, _ = _row(root, "model", f"{shape}_final")
            if not np.array_equal(anchored_seed, model_final):
                raise RuntimeError(f"{run}/{shape}: anchored 种子与 model 终态不是同一相位")
            final, final_file = _row(root, "anchored", f"{shape}_final")
            phases.append(Phase(f"{shape}_{run}_seed", "common", anchored_seed, seed_file,
                                run=run, shape=shape, kind="seed"))
            phases.append(Phase(f"{shape}_{run}_final", "common", final, final_file,
                                run=run, shape=shape, kind="final"))
    return phases, grids


def group_labels(phases: list[Phase]) -> dict[str, list[str]]:
    common = [p.label for p in phases if p.group in {"flat", "common"}]
    high = [p.label for p in phases if p.group == "common" and p.run == "r2"]
    return dict(common=common, high=high)


class Objective:
    """只承载被测指标需要的三块掩膜 (与 ShapeObjective 的推导一致)。"""

    def __init__(self, target: np.ndarray, core: np.ndarray):
        self.target = np.asarray(target, dtype=np.float32)
        self.box = self.target > .5
        self.core = np.asarray(core, dtype=bool)


def sample(frame: np.ndarray, xy: np.ndarray) -> np.ndarray:
    return frame[xy[..., 1].astype(int), xy[..., 0].astype(int)]


def measure(frame: np.ndarray, grids: dict[str, dict[str, np.ndarray]]) -> dict[str, dict]:
    """每个形状掩膜都量一遍: 同一条相位对三个目标都有读数, 便于交叉看。"""
    from ao_shaping.algorithm.signal_processing.multishape_objective import image_metrics

    out: dict[str, dict] = {}
    for shape, grid in grids.items():
        metrics = image_metrics(sample(frame, grid["camera_xy"]), Objective(grid["target"], grid["core"]))
        out[shape] = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                      for k, v in metrics.items()}
    return out


def self_check(args) -> int:
    """免设备验收: 用存档帧重算指标, 与存档 manifest 的同一行对照。"""
    phases, grids = load_phases()
    print(f"phases: {len(phases)}")
    for p in phases:
        print(f"  {p.label:<24} {p.phase.shape} min {p.phase.min():+.2f} max {p.phase.max():+.2f} "
              f"sha {sha(p.phase)[:12]}  <- {p.source}")
    worst = 0.0
    # 只能与**自己那张网格**对照: 存档行的指标是用该会话自己的 camera_xy / 掩膜
    # 算出来的。r1 行与我的规范网格同源, 因此必须逐位一致; r2 行网格差 ~0.2 px,
    # 只作参考读数, 不参与判定。
    for run, directory in RUNS.items():
        root = ROOT / directory
        rows = json.loads((root / "anchored" / "manifest.json").read_text(encoding="utf-8"))
        with np.load(root / "anchored" / rows[0]["file"], allow_pickle=True) as z:
            canon = np.array(z["camera_xy"], dtype=float)
        own = {}
        for shape in SHAPES:
            row = next(r for r in rows if r["label"] == f"{shape}_final")
            with np.load(root / "anchored" / row["file"], allow_pickle=True) as z:
                own[shape] = dict(camera_xy=np.array(z["camera_xy"], dtype=float),
                                  target=np.array(z["target"], dtype=np.float32),
                                  core=np.array(z["core"], dtype=bool))
        for label in ("square_seed", "square_final", "triangle_final", "rectangle_final"):
            row = next(r for r in rows if r["label"] == label)
            with np.load(root / "anchored" / row["file"], allow_pickle=True) as z:
                raw = np.asarray(z["raw_frames"], dtype=np.uint16)
            got = measure(raw.mean(0), own)[label.split("_")[0]]
            for key in ("cv", "ee_roi", "mean", "target_peak"):
                want = float(row["metrics"][key])
                rel = abs(got[key] - want) / max(abs(want), 1e-12)
                if run == "r1":
                    worst = max(worst, rel)
                if rel > 1e-6:
                    print(f"  {run}/{label} {key}: {got[key]!r} vs 存档 {want!r} (rel {rel:.1e})")
            if int(raw.max()) != int(row["peak"]):
                print(f"  {run}/{label} peak: {raw.max()} vs 存档 {row['peak']}")
        # 规范网格 (r1) 与本次会话网格的差异, 决定跨会话比较是否同掩膜。
        base = json.loads((ROOT / RUNS["r1"] / "anchored" / "manifest.json").read_text(encoding="utf-8"))
        with np.load(ROOT / RUNS["r1"] / "anchored" / base[0]["file"], allow_pickle=True) as z:
            canon = np.array(z["camera_xy"], dtype=float)
        print(f"  {run}: 网格相对 r1 最大位移 {float(np.abs(own['square']['camera_xy'] - canon).max()):.4f} px")
    print(f"recompute worst relative deviation (r1, 必须 0): {worst:.3e}")
    return 0 if worst <= 1e-9 else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=None,
                    help="输出目录 (默认 <tmp>/ao_shaping 之外的仓库内 data/shape_trials/blackbox_replay_<时间戳>)")
    ap.add_argument("--execute", action="store_true", help="真正访问设备; 缺省只打印计划")
    ap.add_argument("--self-check", action="store_true", help="免设备验收指标通路后退出")
    ap.add_argument("--frames", type=int, default=3, help="每个标签每轮的帧数")
    ap.add_argument("--passes", type=int, default=2, help="交错轮数 (第 2 轮反序)")
    ap.add_argument("--settle-s", type=float, default=.6)
    ap.add_argument("--target-fraction", type=float, default=.80, help="选曝光时的目标峰值占比")
    ap.add_argument("--ceiling-fraction", type=float, default=.90, help="允许的实测峰值上限占比")
    ap.add_argument("--lock-timeout-s", type=float, default=900.0)
    ap.add_argument("--cam-id", type=int, default=0)
    ap.add_argument("--slm-number", type=int, default=1)
    args = ap.parse_args(argv)

    if args.self_check:
        return self_check(args)

    phases, grids = load_phases()
    labels = group_labels(phases)
    by_label = {p.label: p for p in phases}
    print(f"common 组 {len(labels['common'])} 条: {labels['common']}")
    print(f"high   组 {len(labels['high'])} 条: {labels['high']}")
    if not args.execute:
        print("计划模式: 加 --execute 才会访问设备")
        return 0

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = args.out or (ROOT / "data" / "shape_trials" / f"blackbox_replay_{stamp}")
    if out.exists():
        raise SystemExit(f"输出目录已存在, 拒绝覆盖: {out}")
    out.mkdir(parents=True)
    (out / "raw").mkdir()

    from ao_shaping.drivers.ccd.common import (create_camera, get_camera_exposure_ms,
                                               get_camera_exposure_range)
    from ao_shaping.drivers.slm.santec import Santec
    from ao_shaping.tools.kernels.device_mutex import DeviceLock
    from ao_shaping.tools.slm.transfer.capture import to_panel
    from ao_shaping.tools.slm.transfer.hardware import prepare_camera_sdk

    lock = DeviceLock.default()
    if not lock.acquire(timeout=args.lock_timeout_s):
        raise SystemExit(f"设备被占用: {lock.holder()}")
    logger.info("已取得设备互斥锁 {}", lock.holder())

    ceiling_counts = args.ceiling_fraction * FULL_SCALE
    manifest: list[dict] = []
    ladder: list[dict] = []
    exposures: dict[str, float] = {}
    frames_written = 0
    handles: dict = {}

    def dump(name: str, payload) -> None:
        text = json.dumps(payload, ensure_ascii=False, indent=2, default=float)
        (out / name).write_text(text, encoding="utf-8")

    try:
        prepare_camera_sdk("daheng")
        with create_camera("daheng", args.cam_id, exposure_time_ms=.02, bit_depth=12) as cam:
            handles["cam"] = cam
            with Santec(slm_number=args.slm_number, wavelength=1064, video_mode=0) as slm:
                handles["slm"] = slm
                slm.set_phase_overlay(None)
                floor_ms, ceiling_ms = get_camera_exposure_range(cam)
                logger.info("曝光范围 {} .. {} ms", floor_ms, ceiling_ms)
                shape_info = None

                def read_frame() -> np.ndarray:
                    return np.asarray(cam.get_numpy_image(n_sample=1, skip_first=False))

                def set_exposure(ms: float) -> float:
                    cam.reset_exposure_time(float(ms))
                    return float(get_camera_exposure_ms(cam))

                def show(phase: np.ndarray):
                    gray = slm.create_phase_from_array(to_panel(phase))
                    slm.display_data(gray)
                    return gray

                def probe(label: str, ms: float) -> dict:
                    """改曝光 -> 落相位 -> 等稳定 -> 读一帧, 只用于定曝光。"""
                    nonlocal shape_info
                    phase = by_label[label]
                    readback = set_exposure(ms)
                    show(phase.phase)
                    time.sleep(args.settle_s)
                    stamp = clock()
                    frame = read_frame()
                    if shape_info is None:
                        shape_info = dict(shape=list(frame.shape), dtype=str(frame.dtype),
                                          observed_max=int(frame.max()))
                    row = dict(stamp=stamp, exposure_ms_requested=float(ms), exposure_ms_read=readback,
                               rawmax=int(frame.max()), peak=float(np.percentile(frame, 99.99)),
                               median=float(np.median(frame)), label=label)
                    ladder.append(row)
                    return row

                # 1) 低曝光摸底 (0.02 ms 对整组都安全): 实测峰值 -> 线性外推。
                probe_ms = max(floor_ms, min(.02, ceiling_ms))
                base = max(probe(lbl, probe_ms)["rawmax"] for lbl in labels["common"])
                per_ms = max(base, 1.0) / probe_ms
                want = args.target_fraction * FULL_SCALE
                exposures["common"] = float(np.clip(want / per_ms, floor_ms, ceiling_ms))
                logger.info("common 曝光初值 {:.4f} ms (0.02 ms 下最大峰 {})",
                            exposures["common"], base)

                # 2) 整组验证 + 必要时一次性收缩/放大 (传感器对曝光线性)。
                for attempt in range(3):
                    ms = exposures["common"]
                    peaks = {lbl: probe(lbl, ms)["rawmax"] for lbl in labels["common"]}
                    hot = max(peaks.values())
                    logger.info("common 验证 #{}: {:.4f} ms 最大峰 {} (上限 {:.0f})",
                                attempt + 1, ms, hot, ceiling_counts)
                    if hot > ceiling_counts:
                        exposures["common"] = float(np.clip(ms * ceiling_counts / hot, floor_ms, ceiling_ms))
                        continue
                    if hot < .5 * want and ms < ceiling_ms:
                        exposures["common"] = float(np.clip(ms * want / max(hot, 1.0), floor_ms, ceiling_ms))
                        continue
                    break

                # 3) high 组: 同一批 GS 会话相位在同一曝光下的实测峰值 -> 外推。
                ms_common = exposures["common"]
                hot_common = max(probe(lbl, ms_common)["rawmax"] for lbl in labels["high"])
                per_ms_high = max(hot_common, 1.0) / ms_common
                exposures["high"] = float(np.clip(want / per_ms_high, floor_ms, ceiling_ms))
                for attempt in range(3):
                    ms = exposures["high"]
                    peaks = {lbl: probe(lbl, ms)["rawmax"] for lbl in labels["high"]}
                    hot = max(peaks.values())
                    logger.info("high 验证 #{}: {:.4f} ms 最大峰 {} (上限 {:.0f})",
                                attempt + 1, ms, hot, ceiling_counts)
                    if hot > ceiling_counts:
                        exposures["high"] = float(np.clip(ms * ceiling_counts / hot, floor_ms, ceiling_ms))
                        continue
                    if hot < .5 * want and ms < ceiling_ms:
                        exposures["high"] = float(np.clip(ms * want / max(hot, 1.0), floor_ms, ceiling_ms))
                        continue
                    break
                for group, ms in exposures.items():
                    peaks = [probe(lbl, ms)["rawmax"] for lbl in labels[group]]
                    if max(peaks) > ceiling_counts:
                        dump("ladder.json", ladder)
                        raise RuntimeError(f"{group}: final exposure verification exceeds ceiling")
                dump("ladder.json", ladder)
                dump("device_state.json", dict(
                    camera_connected=cam.is_connected(), slm_open=bool(slm.is_open),
                    bit_depth=12, video_mode=0, camera_format=shape_info,
                ))
                dump("config.json", dict(
                    script=str(Path(__file__).resolve().relative_to(ROOT)),
                    created_utc=clock()["timestamp_utc"],
                    out=str(out.relative_to(ROOT)),
                    inputs={k: str(ROOT / v) for k, v in RUNS.items()},
                    phases=[dict(label=p.label, group=p.group, run=p.run, shape=p.shape,
                                 kind=p.kind, source=p.source, sha256=sha(p.phase)) for p in phases],
                    exposures_ms=exposures, frames=args.frames, passes=args.passes,
                    settle_s=args.settle_s, target_fraction=args.target_fraction,
                    ceiling_fraction=args.ceiling_fraction, full_scale=FULL_SCALE,
                    camera=dict(type="daheng", id=args.cam_id,
                                exposure_range_ms=[floor_ms, ceiling_ms], observed=shape_info),
                    slm=dict(number=args.slm_number, wavelength_nm=1064, video_mode=0),
                    note="相位取自存档 model_phase, 原路 to_panel -> create_phase_from_array, 不叠加载波",
                ))

                def capture(label: str, group: str, exposure_ms: float) -> dict:
                    nonlocal frames_written
                    phase = by_label[label]
                    set_exposure(exposure_ms)
                    start = time.perf_counter()
                    gray = show(phase.phase)
                    written = time.perf_counter()
                    time.sleep(args.settle_s)
                    raws, stamps = [], []
                    for _ in range(args.frames):
                        stamp = clock()
                        frame = read_frame()
                        raws.append(np.asarray(frame, dtype=np.uint16).copy())
                        stamps.append(dict(**stamp, end_monotonic_ns=time.monotonic_ns(),
                                           exposure_ms=float(get_camera_exposure_ms(cam))))
                    raw = np.stack(raws)
                    read_done = time.perf_counter()
                    name = f"{group}_{label}_{frames_written:04d}.npy"
                    np.save(out / "raw" / name, raw)
                    frames_written += 1
                    image = raw.mean(0)
                    row = dict(label=label, group=group, run=phase.run, shape=phase.shape,
                               kind=phase.kind, source=phase.source,
                               phase_sha256=sha(phase.phase),
                               gray_sha256=sha(gray), gray_min=int(gray.min()), gray_max=int(gray.max()),
                               exposure_ms=exposure_ms,
                               exposure_ms_read=[s["exposure_ms"] for s in stamps],
                               file=f"raw/{name}", frames=int(raw.shape[0]),
                               rawmax=int(raw.max()), peak=float(np.percentile(image, 99.99)),
                               median=float(np.median(image)),
                               saturation_pixels=int((raw >= np.floor(FULL_SCALE * .98)).sum()),
                               metrics=measure(image, grids), stamps=stamps,
                               timing=dict(write_s=written - start, settle_s=args.settle_s,
                                           read_s=read_done - written,
                                           total_s=time.perf_counter() - start))
                    manifest.append(row)
                    dump("manifest.json", manifest)
                    logger.info("{} {} 曝光 {:.4f} ms 峰值 {} 饱和像素 {}",
                                group, label, exposure_ms, row["rawmax"], row["saturation_pixels"])
                    return row

                dump("exposures.json", exposures)
                for group in ("common", "high"):
                    order = labels[group]
                    for p in range(args.passes):
                        seq = order if p % 2 == 0 else list(reversed(order))
                        for label in seq:
                            capture(label, group, exposures[group])

                logger.info("采集完成: {} 次采集 / {} 帧", len(manifest), frames_written)
                handles.update(cam=cam, slm=slm,
                               cam_connected_before_close=cam.is_connected(),
                               slm_open_before_close=bool(slm.is_open))
    finally:
        cam = handles.get("cam")
        slm = handles.get("slm")

        def describe(holder):
            """把 LockHolder 落成可 JSON 化的字段 (它本身不是标量)。"""
            if holder is None:
                return None
            return dict(pid=int(holder.pid), cmdline=str(holder.cmdline),
                        acquired_at=str(holder.acquired_at), unknown=bool(holder.unknown))

        before = describe(lock.holder())
        released = False
        try:
            lock.release()
            released = not lock.is_locked()
        except Exception as exc:
            logger.exception("释放设备互斥锁失败: {}", exc)
        logger.info("设备锁释放完成={} (释放前持有者 {})", released, before)
        dump("release.json", dict(
            lock_released=released,
            lock_holder_before_release=before,
            lock_holder_after_release=describe(lock.holder()),
            lock_is_locked_after_release=lock.is_locked(),
            lock_path=str(lock.path),
            camera_connected_before_close=handles.get("cam_connected_before_close"),
            camera_connected_after_close=cam.is_connected() if cam is not None else None,
            slm_open_before_close=handles.get("slm_open_before_close"),
            slm_open_after_close=bool(slm.is_open) if slm is not None else None,
            released_utc=clock()["timestamp_utc"],
        ))


if __name__ == "__main__":
    raise SystemExit(main())
