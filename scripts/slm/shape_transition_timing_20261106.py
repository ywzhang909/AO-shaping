"""可微整形算法在形状之间切换的速度计时（square→circle→triangle→square）。

本脚本回答一个问题：**当相位不是预生成的**，可微整形算法从"零相位"出发、并在每个
后续形状用"上一形状的已接收相位"作为热启动时，切换到目标形状要多快？判据是实测
CCD 核心区变异系数 ``CV ≤ --cv-threshold``（用户明确定义：**实测核心 CV**）。

与 ``run_multishape_capability.py`` 一致的约定（含踩坑教训）：
* **同帧定心**：先显示种子、抓一帧，用这一帧的 argmax 同时建模型侧掩膜与实测侧掩膜；
* **闭环不加倾斜**：优化器在零级位置做目标形状，载波（倾斜）只作为可选的最终展示叠加
  （本计时脚本内部不加载波，保持纯形状切换速度）；
* **判据用实测掩膜**：接收阈值只认 CCD 实测核心 CV，不认模型预测。
* **接受即携带**：一旦某形状达标，用它的已接收相位作下一形状 ``seed_phase``（热启动）；
  **第一个形状固定从零相位网格开始**。

设备纪律（AGENTS.md）：
* 没 ``--execute`` 时不 import 驱动、不打开任何设备；仍解析参数、写 ``plan.json``，
  然后打印 ``skip: requires --execute`` 干净退出。
* 有 ``--execute`` 时单进程打开 Santec #N（内存模式 video_mode=0）与大恒 CCD 一次，
  finally 释放；每次 ``display_data`` 与 ``grab`` 写进 ``device_log.json``。

用法：
    python scripts/slm/shape_transition_timing_20261106.py --out /path/to/out   # 离线：只写 plan.json
    python scripts/slm/shape_transition_timing_20261106.py --execute --out ...  # 硬件计时
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from loguru import logger

from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
    DifferentiableUniformityOptimizer,
    FrozenDefocusForward,
    load_frozen_parameters,
    recentre_window,
)
from ao_shaping.algorithm.signal_processing.multishape_objective import (
    ShapeObjective,
    target_at,
)
from ao_shaping.utils.image.target.patterns import create_target_shape

#: 与 square_grow_diff_sweep_20261106.py 一致的控制网格 / FFT / 采样步长。
CONTROL_GRID = 640
FFT_SIZE = 2048
WINDOW_STEP_PX = 2.0
#: 实测全帧核心掩膜在软边上内缩的像素数（与 capability 的 inner_shrink 同类，避免软边）。
MEASURED_CORE_ERODE_PX = 2

#: 经标定的 Daheng model640 光路检查点（控制网格 640、FFT 2048）。
DEFAULT_CHECKPOINT = (
    ROOT
    / "data/debug/slm_transfer/multishape_nogs_20261104/p20261104/model640/frozen_parameters.json"
)

#: 全帧相机的传感器几何（离线下唯一来源；--execute 时被驱动回读覆盖）。
CAMERA_SENSOR_PX = {
    "daheng": (2592, 1944),
}


def clock() -> dict:
    return dict(
        timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="microseconds"),
        monotonic_ns=time.monotonic_ns(),
    )


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_shapes(raw: str) -> list[str]:
    return [s for s in [p.strip() for p in raw.split(",")] if s]


def build_plan(args: argparse.Namespace) -> dict:
    """设备/模型构建之前的运行决定（与 square_grow 的 build_plan 同构）。"""
    sensor = (args.camera_width, args.camera_height) if (args.camera_width and args.camera_height) \
        else CAMERA_SENSOR_PX[args.camera]
    return dict(
        checkpoint=str(args.checkpoint),
        checkpoint_sha256=sha256_file(args.checkpoint),
        out=str(args.out),
        shapes=split_shapes(args.shapes),
        side=float(args.side),
        cv_threshold=float(args.cv_threshold),
        max_iterations=int(args.max_iterations),
        max_blocks=int(args.max_blocks),
        check_every=int(args.check_every),
        updates_per_block=int(args.updates_per_block),
        lr=float(args.lr),
        phase_rms_cap=float(args.phase_rms_cap),
        carrier_period_px=float(args.carrier_period_px),
        carrier_axis=str(args.carrier_axis),
        window_half_span_px=float(args.window_half_span_px),
        warm_start=bool(args.warm_start),
        frames=int(args.frames),
        exposure_ms=float(args.exposure_ms),
        settle_s=float(args.settle_s),
        control_grid=CONTROL_GRID,
        fft_size=FFT_SIZE,
        window_step_px=WINDOW_STEP_PX,
        noise_floor=str(args.noise_floor) if args.noise_floor else None,
        camera=str(args.camera),
        cam_id=int(args.cam_id),
        slm_number=int(args.slm_number),
        camera_sensor_px=list(sensor),
        metrics=["per-shape: iterations-to-acceptance, wall-clock seconds",
                 "accepted/measured core CV on CCD"],
        script=__file__,
        timestamp=datetime.now(timezone.utc).isoformat(timespec="microseconds"),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shapes", default="square,circle,triangle,square",
                        help="循环顺序，逗号分隔")
    parser.add_argument("--side", type=float, default=400.0,
                        help="目标边/直径（相机像元），所有形状同尺寸保证比较公平")
    parser.add_argument("--window-half-span-px", type=float, default=260.0)
    parser.add_argument("--cv-threshold", type=float, default=0.10,
                        help="实测 CCD 核心 CV 达到此值即接收")
    parser.add_argument("--max-iterations", type=int, default=300)
    parser.add_argument("--max-blocks", type=int, default=200)
    parser.add_argument("--check-every", type=int, default=10,
                        help="每几次 optimizer.update 检查一次 CCD")
    parser.add_argument("--updates-per-block", type=int, default=10,
                        help="两次候选相位提取之间的 update 次数（须等于 --check-every）")
    parser.add_argument("--stall-window-blocks", type=int, default=5,
                        help="连续无进展的块数上限；超过即判定停滞，提前结束本形状进入下一形状")
    parser.add_argument("--stall-min-improve", type=float, default=0.005,
                        help="停滞窗口内实测 CV 相对窗口起点的改善绝对值须达到此值，否则视为无进展")
    parser.add_argument("--log-every-blocks", type=int, default=1,
                        help="逐块进度 logger 的节流步长（每 N 块打一次，0 关闭逐块日志）")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--phase-rms-cap", type=float, default=0.06)
    parser.add_argument("--carrier-period-px", type=float, default=32.0)
    parser.add_argument("--carrier-axis", choices=("x", "y"), default="x")
    parser.add_argument("--frames", type=int, default=5, help="每次抓取平均的帧数")
    parser.add_argument("--exposure-ms", type=float, default=0.5)
    parser.add_argument("--settle-s", type=float, default=0.4)
    parser.add_argument("--warm-start", action="store_true", default=True,
                        help="后续形状从上一形状已接收相位热启动（默认开）；首形状恒从零")
    parser.add_argument("--execute", action="store_true", help="打开硬件执行")
    parser.add_argument("--wait-until-idle", type=float, default=0.0,
                        help="打开设备前轮询等待其它 python 进程退出的秒数")
    parser.add_argument("--slm-number", type=int, default=1)
    parser.add_argument("--cam-id", type=int, default=0)
    parser.add_argument("--noise-floor", type=Path, default=None,
                        help="可选 three_sigma_resolution 地板 JSON（本脚本仅记录，不参与判据）")
    parser.add_argument("--camera", choices=("daheng",), default="daheng")
    parser.add_argument("--camera-width", type=int, default=None)
    parser.add_argument("--camera-height", type=int, default=None)
    return parser.parse_args(argv)


def wait_for_idle(seconds: float) -> None:
    """轮询：没有别的进程持有本光路设备才继续（逻辑同 capability 脚本行 159-170）。"""
    if seconds <= 0:
        return
    import subprocess

    marker_keys = ("hw_", "defocus", "differentiable", "diffraction_efficiency",
                   "capture_", "multishape", "shape_transition_timing")
    deadline = time.time() + float(seconds)
    while time.time() < deadline:
        listing = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "Select-Object -ExpandProperty CommandLine"],
            capture_output=True, text=True, timeout=30,
        ).stdout
        busy = [
            line for line in listing.splitlines()
            if any(key in line for key in marker_keys)
            and "shape_transition_timing" not in line
        ]
        if not busy:
            return
        logger.info("[wait] 仍有进程占用设备，等待…")
        time.sleep(15)
    logger.warning(f"[wait] {seconds}s 后仍有进程占用设备，继续（可能冲突）")


def open_devices(args: argparse.Namespace, centre_ij, device_log: list[dict]):
    """打开 Santec + 大恒 CCD 一次；返回 (cam, slm, exposure_ms)。"""
    from ao_shaping.drivers.ccd.common import (
        create_camera,
        get_camera_exposure_ms,
        get_camera_exposure_range,
    )
    from ao_shaping.drivers.slm.santec import Santec

    def note(action: str, **fields) -> None:
        entry = dict(action=action, **fields, **clock())
        device_log.append(entry)
        logger.info(f"[device] {action} {fields}")

    note("open", device="camera:daheng", cam_id=args.cam_id)
    cam = create_camera("daheng", args.cam_id, exposure_time_ms=0.05)
    cam.open()
    try:
        floor, ceiling = get_camera_exposure_range(cam)
        exposure = float(args.exposure_ms) if args.exposure_ms else max(floor, 0.5)
        exposure = float(np.clip(exposure, floor, ceiling))
        cam.reset_exposure_time(exposure)
        exposure = float(get_camera_exposure_ms(cam))
        note("open", device=f"slm#{args.slm_number}", mode="memory (video_mode=0)")
        slm = Santec(slm_number=args.slm_number, wavelength=1064, video_mode=0)
        slm.open()
        slm.set_phase_overlay(None)
        note("ready", exposure_ms=exposure,
             sensor=(int(getattr(cam, "cam_width", 0) or 0),
                     int(getattr(cam, "cam_height", 0) or 0)),
             centre=list(centre_ij) if centre_ij is not None else None)
        return cam, slm, exposure
    except Exception:
        cam.close()
        raise


def close_devices(slm, cam, device_log: list[dict], *, slm_number: int | None = None,
                  camera: str | None = None) -> None:
    if slm is not None:
        try:
            from ao_shaping.utils.slm.phase_display import phase_to_slm_grayscale

            panel = np.zeros((1200, 1920), np.float32)
            slm.display_data(phase_to_slm_grayscale(panel, slm=slm))
        except Exception as exc:  # 清理失败不掩盖主流程结果，但要留痕
            device_log.append(dict(action="clear_failed", error=repr(exc), **clock()))
        try:
            slm.close()
        finally:
            device_log.append(dict(action="release", device=f"slm#{slm_number}", **clock()))
    if cam is not None:
        try:
            cam.close()
        finally:
            device_log.append(dict(action="release", device=f"camera:{camera}", **clock()))
    logger.info(f"[device] release SLM #{slm_number} and camera {camera}")


def grab_average(cam, frames: int, *, height: int | None = None) -> np.ndarray:
    """抓取并平均 ``frames`` 帧；返回 float64 全帧图像。"""
    collected = []
    for _ in range(max(int(frames), 1)):
        raw = np.asarray(cam.get_numpy_image(n_sample=1), dtype=float)
        if raw.ndim == 1:
            raw = raw.reshape(height or 1944, 2592)
        collected.append(raw.mean(0) if raw.ndim == 3 else raw)
    return np.mean(np.stack(collected), axis=0)


def display_phase(slm, phase_grid: np.ndarray, *, device_log: list[dict] | None = None) -> None:
    """把控制网格相位经 to_panel 展开为面板灰度并显示。"""
    from ao_shaping.tools.slm.transfer.capture import to_panel
    from ao_shaping.utils.slm.phase_display import phase_to_slm_grayscale

    if device_log is not None:
        device_log.append(dict(action="display", **clock()))
    slm.display_data(phase_to_slm_grayscale(to_panel(np.asarray(phase_grid, np.float32)), slm=slm))


def monitor_cv(frame: np.ndarray, core: np.ndarray) -> float:
    """核心区变异系数（float；核心空或均值<=0 时返回 NaN）。"""
    values = np.asarray(frame, dtype=np.float64)[core]
    mean = float(values.mean()) if values.size else float("nan")
    if not np.isfinite(mean) or mean <= 0:
        return float("nan")
    return float(np.sqrt((((values / mean) - 1) ** 2).mean()))


def model_cv(forward, grid: np.ndarray, core: np.ndarray) -> float:
    """模型侧核心 CV（只读校验/进度，不参与接收判据）。"""
    import torch

    with torch.no_grad():
        image = forward.predict(
            torch.tensor(grid[None], dtype=torch.float32, device=forward._device),
            exposure_ms=1.0,
        )[0]
        image = image.detach().cpu().numpy().astype(np.float64)
    return monitor_cv(np.maximum(image, 0.0), np.asarray(core, dtype=bool))


def sample_ccd_to_window(frame: np.ndarray, camera_xy: np.ndarray) -> np.ndarray:
    """把全帧 CCD 帧采样到冻结前向的相机窗网格（scipy map_coordinates）。"""
    from scipy.ndimage import map_coordinates

    coords = np.stack(
        [camera_xy[..., 1] - 0.5, camera_xy[..., 0] - 0.5], axis=0
    ).reshape(2, -1)
    return map_coordinates(
        np.asarray(frame, dtype=np.float64), coords, order=1, mode="constant"
    ).reshape(camera_xy.shape[:2])


def full_frame_raster(shape: str, frame_shape: tuple[int, int], centre_ij,
                      side: float) -> tuple[np.ndarray, np.ndarray]:
    """在全帧相机网格上光栅化目标形状，返回 (box, core) 掩膜（实测掩膜用）。

    create_target_shape 支持 circle/square/triangle（光栅路径），故 circle 也能在此使用。
    centre_ij = (row, col) 为 CCD 里的零点。
    """
    from scipy.ndimage import binary_erosion, map_coordinates

    height, width = int(frame_shape[0]), int(frame_shape[1])
    row0, col0 = float(centre_ij[0]), float(centre_ij[1])
    size = max(512, 2 * int(np.ceil((side + 8) / 2)))
    raster = create_target_shape(
        shape, (size, size), side=side, radius_ratio=side / size,
        center=(size / 2, size / 2),
    )
    yy, xx = np.mgrid[:height, :width]
    sampled = map_coordinates(
        raster,
        [yy - row0 + size / 2, xx - col0 + size / 2],
        order=1, mode="constant",
    )
    box = sampled > 0.5
    core = binary_erosion(box, iterations=int(MEASURED_CORE_ERODE_PX))
    if not core.any():
        core = box  # 极小目标时退化为框内全部，避免空掩膜
    return box, core


def run_shape(
    *,
    order_index: int,
    shape: str,
    args: argparse.Namespace,
    forward: FrozenDefocusForward,
    camera_xy: np.ndarray,
    spot_xy: tuple[float, float],
    slm,
    cam,
    exposure_ms: float,
    device_log: list[dict],
    warm_seed: np.ndarray | None,
    out: Path,
    base_frame: np.ndarray,
) -> tuple[dict, str, np.ndarray]:
    """优化单个形状到接收；返回 (row, seed_source, carried_grid)。

    ``spot_xy`` 是 CCD 全帧上的 ``(col, row)`` 像素坐标（同帧定心的 argmax）。
    模型侧 ``target_at`` 与实测侧 ``full_frame_raster`` 都在 **CCD 物理像素坐标**
    空间采样，故都以此为中心 —— 与采样窗网格下标（``forward.sample_grid_index``）
    不可互用。
    """
    grid = int(forward.params.grid_size)
    if warm_seed is not None and warm_seed.shape != (grid, grid):
        raise ValueError(
            f"warm-start seed shape {warm_seed.shape} != model grid {(grid, grid)}"
        )
    if warm_seed is None:
        seed_grid = np.zeros((grid, grid), dtype=np.float32)
        seed_source = "zero"
    else:
        seed_grid = np.array(warm_seed, dtype=np.float32)
        seed_source = "warm_start_prev"

    # 模型侧目标/掩膜：本形状的采样窗光栅目标 -> ShapeObjective（mode="shape"，circle 可用）
    # target_at 采样在 CCD 物理像素坐标（xy 为 camera_xy CCD 像素网格），故 centre 必须
    # 用 CCD 像素坐标 spot_xy=(col,row)，不能用采样窗网格下标 centre_ij（坐标空间不同）。
    target_model = target_at(camera_xy, shape, spot_xy, float(args.side)).astype(np.float32)
    objective = ShapeObjective(target_model)
    # 实测侧全帧掩膜：以目标形状光栅化在全帧网格上（circle/square/triangle 都支持），
    # 使接收判据针对"该形状"的实测核心。中心必须用 CCD 全帧像素坐标 spot_xy
    # （(col,row)），而非采样窗网格下标 centre_ij —— 两者坐标空间不同。
    _, measured_core = full_frame_raster(
        shape, base_frame.shape, (spot_xy[1], spot_xy[0]), float(args.side)
    )
    optimizer = DifferentiableUniformityOptimizer(
        forward, objective, seed_grid,
        lr=float(args.lr), step_rms_cap=float(args.phase_rms_cap), device="cpu",
    )

    shape_start = time.monotonic()
    start_iso = datetime.now(timezone.utc).isoformat(timespec="microseconds")
    best_grid: np.ndarray = np.array(seed_grid, dtype=np.float32)
    best_cv: float = float("inf")
    accepted_bool = False
    accepted_cv: float | None = None
    iterations_used = 0
    iter_rows: list[dict] = []
    iterations_out = out / "iterations"
    iterations_out.mkdir(parents=True, exist_ok=True)
    # 候选帧落盘目录必须先建：块循环内 np.save 直接写 phase/（此前只在函数末 mkdir）。
    phase_out = out / "phase"
    phase_out.mkdir(parents=True, exist_ok=True)

    max_blocks = int(args.max_blocks)
    check_every = int(args.check_every)
    if int(args.updates_per_block) != check_every:
        logger.warning(f"--updates-per-block {args.updates_per_block} != --check-every "
                       f"{check_every}; 用 --check-every")
    updates_per_block = check_every

    def flush_iter_table() -> None:
        (iterations_out / f"{order_index}_{shape}.jsonl").write_text(
            "\n".join(json.dumps(row) for row in iter_rows) + "\n", encoding="utf-8"
        )

    total_updates = 0
    # ---- 无进展停滞检测状态（模型无法进一步优化时提前止损，不死循环）----
    # 判据：以窗口起点的 best_cv 为锚，连续 stall_window 块内实测 CV 未相对锚点
    # 改善 ≥ stall_min_improve，即判定停滞、结束本形状进入下一形状。
    stall_window = max(1, int(args.stall_window_blocks))
    stall_min_improve = float(args.stall_min_improve)
    log_every = int(args.log_every_blocks)
    anchor_cv = float("inf")   # 当前停滞窗口起点的 best_cv
    window_blocks = 0          # 窗口内自锚点起未达改善阈值的连续块数
    stop_reason = "max_blocks"
    for _block in range(1, max_blocks + 1):
        if total_updates >= int(args.max_iterations):
            stop_reason = "max_iterations"
            break
        block_start = time.monotonic()
        # ---- block of updates，keep lowest-loss 候选相位（经 update 中间态）----
        lowest_loss = float("inf")
        best_delta: np.ndarray | None = None
        for _ in range(updates_per_block):
            optimizer.update()
            total_updates += 1
            loss = optimizer.last_loss
            if loss is not None and loss < lowest_loss:
                lowest_loss = float(loss)
                best_delta = np.array(optimizer.delta_phase, dtype=np.float32)
        if best_delta is None:
            best_delta = np.array(optimizer.delta_phase, dtype=np.float32)
        candidate_grid = (best_delta + seed_grid).astype(np.float32)

        # ---- 实测评判（仅 --execute；离线下不触硬件）----
        measured_cv: float | None = None
        measured_cv_window: float | None = None
        model_cv_val: float | None = None
        if args.execute:
            display_phase(slm, candidate_grid, device_log=device_log)
            time.sleep(float(args.settle_s))
            device_log.append(dict(action="capture", exposure_ms=exposure_ms, **clock()))
            frame = grab_average(cam, int(args.frames), height=base_frame.shape[0])
            # 主判据：全帧实测掩膜核心 CV（与 capability 的 box_f/core_f 两掩膜模式一致）。
            measured_cv = monitor_cv(frame, measured_core)
            # 补充诊断：把实测帧采样到模型窗，用 objective.core 计算窗内 CV
            #（对应 square_grow 的 sample_ccd_to_window + 模型掩膜路径）。
            windowed = sample_ccd_to_window(frame, camera_xy)
            measured_cv_window = monitor_cv(windowed, objective.core)
            model_cv_val = model_cv(forward, candidate_grid, objective.core)
            np.save(phase_out / f"measured_{order_index}_{shape}_block{_block}.npy",
                    frame.astype(np.float32))

        iter_rows.append(dict(
            shape=shape, order_index=order_index, iter=total_updates,
            seconds_from_shape_start=round(time.monotonic() - shape_start, 4),
            model_cv=(None if model_cv_val is None else round(model_cv_val, 6)),
            measured_cv=(None if measured_cv is None else round(measured_cv, 6)),
            measured_cv_window=(None if measured_cv_window is None
                                else round(measured_cv_window, 6)),
            accepted_bool=False,
        ))
        flush_iter_table()

        if measured_cv is not None and np.isfinite(measured_cv) and measured_cv < best_cv:
            best_grid = np.array(candidate_grid, dtype=np.float32)
            best_cv = float(measured_cv)

        # ---- 接收判定：实测核心 CV <= 阈值（用户明确选择）----
        if parsed := (measured_cv is not None and np.isfinite(measured_cv)
                      and measured_cv <= float(args.cv_threshold)):
            accepted_bool = True
            accepted_cv = float(measured_cv)
            iterations_used = total_updates
            best_grid = np.array(candidate_grid, dtype=np.float32)
            iter_rows[-1]["accepted_bool"] = True
            flush_iter_table()
            stop_reason = "accepted"
            break

        # ---- 逐块进度日志（可实时看到优化是否在推进）----
        block_ms = (time.monotonic() - block_start) * 1000.0
        if log_every > 0 and (_block % log_every == 0):
            logger.info(
                f"[shape {order_index}/{shape}] block {_block}/{max_blocks} · "
                f"step {total_updates}/{int(args.max_iterations)} · "
                f"loss {lowest_loss if lowest_loss != float('inf') else float('nan'):.4f} · "
                f"real_cv {measured_cv if measured_cv is not None else float('nan'):.4f} · "
                f"model_cv {model_cv_val if model_cv_val is not None else float('nan'):.4f} · "
                f"best_cv {best_cv if best_cv != float('inf') else float('nan'):.4f} · "
                f"block {block_ms:.0f} ms"
            )

        # ---- 无进展停滞检测：窗口内 best_cv 未改善即提前止损，进入下一形状 ---- 
        if best_cv != float("inf"):
            if best_cv < anchor_cv - stall_min_improve:
                # 窗口内已实现实质改善：刷新锚点并清空停滞计数
                anchor_cv = best_cv
                window_blocks = 0
            else:
                window_blocks += 1
                if window_blocks >= stall_window:
                    stop_reason = f"stall_{stall_window}_blocks_no_{stall_min_improve}_improve"
                    logger.warning(
                        f"[shape {order_index}/{shape}] STALL: best_cv {best_cv:.4f} "
                        f"未在 {stall_window} 块内改善 ≥{stall_min_improve}，提前结束本形状进入下一形状"
                    )
                    break

    elapsed = time.monotonic() - shape_start
    if not accepted_bool:
        iterations_used = total_updates

    row = dict(
        shape=shape, order_index=order_index, start_time_iso=start_iso,
        iterations=iterations_used,
        seconds=round(elapsed, 4),
        accepted_cv=(None if accepted_cv is None else round(accepted_cv, 6)),
        min_cv=(None if best_cv == float("inf") else round(best_cv, 6)),
        accepted_bool=bool(accepted_bool),
        seed_source=seed_source,
        stop_reason=stop_reason,
    )
    (out / "phase").mkdir(parents=True, exist_ok=True)
    tag = "accepted" if accepted_bool else "best"
    np.save(out / "phase" / f"stage{order_index}_{shape}_{tag}_grid.npy", best_grid)
    np.save(out / "phase" / f"stage{order_index}_{shape}_seed.npy", seed_grid)

    # ---- 终盘：逐步梯度变化 + 优化指标全量落盘（评估模型可表征性）----
    # optimizer.loss_history 每步含 gradient_norm / increment_rms / phase_rms /
    # raw_to_final_step_cosine / candidate_model_loss_change 等（update() 内记录）。
    history_out = out / "history"
    history_out.mkdir(parents=True, exist_ok=True)
    steps = optimizer.loss_history
    with (history_out / f"{order_index}_{shape}.jsonl").open(
        "w", encoding="utf-8"
    ) as fh:
        for step, rec in enumerate(steps, start=1):
            rec = dict(rec)
            rec = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                   for k, v in rec.items()}
            fh.write(json.dumps({"step": step, **rec}) + "\n")

    return row, seed_source, best_grid


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # ---- 离线可写 plan.json：即便没有 --execute 也先落盘（可验证 CLI+import 管线）----
    plan = build_plan(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")

    if not args.execute:
        print("skip: requires --execute")
        logger.info(f"plan written to {out / 'plan.json'}; hardware skipped (no --execute)")
        return 0

    wait_for_idle(float(args.wait_until_idle))

    params = load_frozen_parameters(args.checkpoint)
    grid = int(params.grid_size)
    device_log: list[dict] = []
    timing_rows: list[dict] = []
    slm = cam = None
    exit_code = 1
    try:
        cam, slm, exposure_ms = open_devices(args, centre_ij=None, device_log=device_log)
        # 驱动回读的真实传感器几何取代离线表
        sensor_w = int(getattr(cam, "cam_width", 0) or 0)
        sensor_h = int(getattr(cam, "cam_height", 0) or 0)
        if sensor_w and sensor_h:
            plan["camera_sensor_live"] = [sensor_w, sensor_h]
            args.camera_width, args.camera_height = sensor_w, sensor_h

        # 同帧定心：显示零相位 → 抓帧 → 用该帧 argmax 建两侧掩膜
        from ao_shaping.tools.slm.transfer.capture import to_panel
        from ao_shaping.utils.slm.phase_display import phase_to_slm_grayscale

        seed_zero = np.zeros((grid, grid), dtype=np.float32)
        device_log.append(dict(action="display_zero_seed", **clock()))
        slm.display_data(phase_to_slm_grayscale(to_panel(seed_zero), slm=slm))
        time.sleep(float(args.settle_s))
        device_log.append(dict(action="capture_base", exposure_ms=exposure_ms, **clock()))
        base = grab_average(cam, int(args.frames), height=args.camera_height or 1944)
        np.save(out / "base.npy", base.astype(np.float32))
        spot_row, spot_col = (int(v) for v in np.unravel_index(int(np.argmax(base)), base.shape))
        logger.info(f"[spot] 同帧零点 ({spot_col}, {spot_row}) seed peak {base.max():.0f}")
        plan["spot_xy"] = [float(spot_col), float(spot_row)]
        (out / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")

        camera_xy = recentre_window((float(spot_col), float(spot_row)),
                                    half_span_px=float(args.window_half_span_px),
                                    step_px=WINDOW_STEP_PX)
        forward = FrozenDefocusForward(params, camera_xy, device="cpu")
        forward.exposure_ms = float(exposure_ms)
        centre = forward.sample_grid_index(float(spot_col), float(spot_row))
        step_ccd = forward.sample_step_ccd()
        plan["centre_model"] = [float(centre[0]), float(centre[1])]
        plan["step_ccd"] = list(step_ccd)

        # 模型侧掩膜与实测侧掩膜都在 run_shape 内按各形状重建（模型 target_at +
        # ShapeObjective；实测 full_frame_raster）。这里仅放初始 plan 的占位。
        plan["mask_measured"] = None

        warm_seed: np.ndarray | None = None
        for i, shape in enumerate(split_shapes(args.shapes)):
            seed_for_this = None if (i == 0 or not args.warm_start) else warm_seed
            logger.info(f"[shape {i}] {shape} seed={'zero' if seed_for_this is None else 'warm_start_prev'}")
            row, _src, carried = run_shape(
                order_index=i, shape=shape, args=args, forward=forward,
                camera_xy=camera_xy,
                spot_xy=(float(spot_col), float(spot_row)),
                slm=slm, cam=cam, exposure_ms=exposure_ms, device_log=device_log,
                warm_seed=seed_for_this, out=out, base_frame=base,
            )
            timing_rows.append(row)
            (out / "timing.jsonl").write_text(
                "\n".join(json.dumps(r) for r in timing_rows) + "\n", encoding="utf-8"
            )
            logger.info(f"[shape {i}] {shape}: iters {row['iterations']}, "
                        f"{row['seconds']}s, accepted={row['accepted_bool']} "
                        f"cv={row['accepted_cv'] or row['min_cv']}")
            warm_seed = carried  # 无论是否接收都携带（已接收相位或最优）作为下一形状种子

        plan["complete"] = True
        (out / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
        exit_code = 0
    finally:
        if device_log:
            # 在关闭后补写关闭日志（close_devices 会继续 append）
            close_devices(slm, cam, device_log, slm_number=args.slm_number, camera=args.camera)
            (out / "device_log.json").write_text(
                json.dumps(device_log, indent=2), encoding="utf-8"
            )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())