"""Capture current-path phase diversity without reusing historical geometry.

Raw frames, displayed grayscale, raw phase and per-frame clocks are retained.
Defaults to a device-free plan; hardware is opened only with --execute.

The measurements are handed to the fit stage **in memory** by default: one
``sample_%04d.npz`` per probe carries ``raw_frames`` (frames x 1944 x 2592 uint8),
a float64 ``frame`` (40 MB), ``panel_phase`` (9 MB) and the exposure-probe stack,
i.e. ~60-80 MB and most of a probe's wall clock. ``fit.load_dataset`` only reads
``model_phase`` and ``raw_frames``, so the default path keeps exactly those two
arrays per probe in RAM and writes none of the npz files. The on-disk dataset
stays available as an opt-in (``--save-dataset`` / ``TransferSession.save_dataset``)
because the offline ``--dataset`` path and the analysis scripts under
``scripts/slm/`` read it; when enabled it writes the same keys and the same
content as before, so a recorded dataset stays byte-compatible.

Exposure is armed with **one** multiplicative correction plus one verification
read (the sensor is linear in exposure, so the correction is exact); the bound is
widened for the cold start, where the first probe reads a few percent of full
scale and needs a factor of ~9. When that single step lands outside the accepted
band anyway — a probe that concentrates enough light to overshoot into
saturation — the ladder is resumed from the exposure it computed but never
measured, with a wider step budget (:data:`EXPOSURE_RETRY_STEPS`), instead of
failing the whole lane: it already knows which way to go, it only ran out of rungs.
"""

from __future__ import annotations
from loguru import logger

import argparse
import hashlib
import json
import pickle
import time
from collections.abc import Callable
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from .light_guard import LightGuard, OpticalPathInvalid, verify_light
from .hardware import DeviceOpenError
from .exposure_policy import (
    AUTO_TOKEN,
    argparse_exposure_type,
    policy_source,
    record_effective_exposure,
    resolve_seed_exposure,
)
from .live_frame import _resolve_observer, frame_payload, notify_frame

#: Exposure search for one probe: one exact correction + one verification read.
#: The old loop allowed 12 multiplicative steps under a (0.15, 6.0) clamp; a cold
#: start at the camera floor (peak ~5% of full scale) needs a factor of ~9, which
#: that clamp cannot do in one step. Both values are recorded in ``config.json``.
EXPOSURE_MAX_STEPS = 2
EXPOSURE_RATIO_BOUNDS = (0.1, 50.0)

#: locate() 屏蔽零级邻域的磁盘半径 (px)。2026-11-09 光路零级未压制 (E0/E1~1.2),
#: argmax 锁零级 -> ramp_matrix 全废; 60 px 覆盖零级 FWHM (~80 px) 的大部又不吃掉
#: period>=48 的一级 (~113 px 外)。
ZERO_ORDER_EXCLUDE_RADIUS = 60.0

#: 一次走不到接受带就接着走: ``EXPOSURE_RETRIES`` 次, 每次给 ``EXPOSURE_RETRY_STEPS``
#: 档预算。2 档的阶梯只够「一次乘法修正 + 一次复核」, 那一步一旦跨过头就**没有档位
#: 往下走**了 —— 2026-11-03 17:40 的会话正是这样死在 probe_012_zernike 上: 它从上一个
#: 探针停住的 1.428 ms 起阶梯, 峰值 255 (饱和) → 修正到 0.642 ms 仍是 255 → 两档用完,
#: 阶梯已经把下一档算成 0.289 ms 却没档位去测它, 于是第 13 条探针 abort 掉整个会话,
#: 前面 12 条好数据一起作废。重试就从那个**算好但没测过**的档位接着走, 乘法阶梯自己
#: 会收敛。
EXPOSURE_RETRIES = 2
EXPOSURE_RETRY_STEPS = 6


def clock() -> dict:
    return dict(
        timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="microseconds"),
        monotonic_ns=time.monotonic_ns(),
    )


def exposure_setter(cam) -> Callable[[float], float]:
    """Wrap a camera into the ``set_exposure_ms`` callback the light guard wants.

    ``reset_exposure_time`` returns the effective exposure on the hardware driver,
    but the leaf backends (sim / image_folder) are allowed to return ``None``, and
    the value that matters for the ``exposure_ceiling`` diagnosis is the one the
    **sensor** reports, not the one we asked for — ``get_camera_exposure_ms`` is
    already the single readback entry point (issue #32, ``回读恒为 3.0``). So set,
    then read back, instead of trusting a return value that half the backends
    omit.
    """
    from ao_shaping.drivers.ccd.common import get_camera_exposure_ms

    def set_exposure(ms: float) -> float:
        cam.reset_exposure_time(float(ms))
        return get_camera_exposure_ms(cam)

    return set_exposure


def to_panel(grid: np.ndarray) -> np.ndarray:
    """Expand the exact model controls, without averaging wrapped phases."""
    size = grid.shape[0]
    if grid.shape != (size, size) or 1920 % size or 1200 % (1920 // size):
        raise ValueError("grid must exactly tile the physical panel")
    pitch = 1920 // size
    return np.repeat(np.repeat(grid, pitch, axis=0), pitch, axis=1)[:1200]


def locate(
    frame: np.ndarray,
    *,
    exclude_xy: tuple[float, float] | None = None,
    exclude_radius: float = ZERO_ORDER_EXCLUDE_RADIUS,
) -> tuple[float, float]:
    """Locate the dominant component using a smoothed peak and local moments.

    ``exclude_xy`` masks a disk around that point (the session zero order) so
    the argmax can fall through to the first diffraction order. Without it a
    beam whose zero order is not suppressed locks every ramp spot onto the
    zero order and ``fit.geometry()`` extracts a degenerate ramp matrix.
    """
    smooth = gaussian_filter(np.asarray(frame, dtype=float), 3)
    if exclude_xy is not None:
        yy, xx = np.mgrid[: smooth.shape[0], : smooth.shape[1]]
        keep = (xx - exclude_xy[0]) ** 2 + (yy - exclude_xy[1]) ** 2 > exclude_radius**2
        masked = np.where(keep, smooth, -np.inf)
        row, col = np.unravel_index(masked.argmax(), masked.shape)
    else:
        row, col = np.unravel_index(smooth.argmax(), smooth.shape)
    y0, y1 = max(0, row - 30), min(frame.shape[0], row + 31)
    x0, x1 = max(0, col - 30), min(frame.shape[1], col + 31)
    weights = np.maximum(frame[y0:y1, x0:x1] - np.median(frame), 0)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    return float((xx * weights).sum() / weights.sum()), float(
        (yy * weights).sum() / weights.sum()
    )


class _LivePrediction:
    """采集段实时预测：有 checkpoint 用历史参数，没有则用理想物理参数。

    两条路径都只初始化一次，并等本次 ``reference_start`` 的实测零点出现后
    重建 origin/相机窗；否则旧会话的零点会平移当前预测。理想模型还必须等
    本次 signed-ramp 几何到齐，不能借用历史几何，也不必先写 bootstrap 文件。
    """

    def __init__(
        self, checkpoint: Path | None, *, capture_grid: int, device: str | None
    ) -> None:
        self.device = device
        self.capture_grid = capture_grid
        self.params = None
        if checkpoint is not None:
            from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
                load_frozen_parameters,
            )

            self.params = load_frozen_parameters(checkpoint)
        self.model_grid = (
            int(self.params.grid_size) if self.params is not None else capture_grid
        )
        self.forward = None
        self.origin: tuple[float, float] | None = None
        self.camera_xy: np.ndarray | None = None
        # 历史网格不同才准备转换缓存。坐标与零面板只分配一次；逐帧成本是
        # 一次 to_panel + map_coordinates + forward，不重复读 JSON 或重建模型。
        self._panel: np.ndarray | None = None
        self._sample_coordinates: tuple[np.ndarray, np.ndarray] | None = None
        if capture_grid != self.model_grid:
            gy, gx = np.mgrid[: self.model_grid, : self.model_grid] * (
                1920 / self.model_grid
            )
            self._sample_coordinates = (gy, gx)
            self._panel = np.zeros((1920, 1920), dtype=np.float32)

    def attach_reference(self, reference_row: dict) -> None:
        """保存本次平场实测零点；历史模型可随即建 forward，理想模型继续等几何。"""
        if self.forward is not None or self.camera_xy is not None:
            return
        from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
            recentre_window,
        )
        from ao_shaping.tools.slm.transfer.fit import spot

        measured = spot(reference_row)
        self.origin = (float(measured[0]), float(measured[1]))
        self.camera_xy = recentre_window(self.origin, half_span_px=120.0)
        if self.params is not None:
            self._build_forward(self.params)

    def _build_forward(self, params) -> None:
        """以当前会话零点覆盖参数中的 origin，再创建唯一的 frozen forward。"""
        if self.origin is None or self.camera_xy is None:
            raise RuntimeError("reference_start is required before live prediction")
        from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
            FrozenDefocusForward,
        )

        self.params = replace(params, origin_xy=self.origin)
        self.forward = FrozenDefocusForward(
            self.params, self.camera_xy, device=self.device
        )

    def attach_geometry(self, rows: list[dict]) -> None:
        """signed-ramp 到齐后直接构造零校正理想模型，不等待离线 bootstrap。"""
        if self.params is not None:
            return
        from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
            FrozenDefocusParameters,
        )
        from ao_shaping.tools.slm.transfer.fit import geometry, valid

        matrix, _ = geometry([row for row in rows if valid(row)])
        ramp_matrix = tuple(tuple(float(value) for value in row) for row in matrix)
        self._build_forward(
            FrozenDefocusParameters(
                defocus_diopter=0.0,
                origin_xy=self.origin,
                amplitude_center_xy=(960.0, 600.0),
                amplitude_width_xy=(400.0, 300.0),
                residual_noll_radians={j: 0.0 for j in range(5, 12)},
                gain=1.0,
                background=0.0,
                sigma_ccd_px=0.0,
                ramp_matrix=ramp_matrix,
                fft_size=max(1024, self.capture_grid),
                pixel_samples=3,
                grid_size=self.capture_grid,
            )
        )

    def _model_phase(self, phase_grid: np.ndarray) -> np.ndarray:
        """按 fit.load_dataset 的面板阈值，把当前采集控制网格送到模型网格。"""
        phase = np.asarray(phase_grid, dtype=np.float32)
        if self._panel is None:
            return phase
        from scipy.ndimage import map_coordinates

        # 与 fit.load_dataset 的非控制网格分支一致：1920² 画布的上方 1200
        # 行是实际面板，其余保持零；不能直接把控制数组 zoom 到另一尺寸。
        self._panel[:1200] = to_panel(phase)
        return map_coordinates(self._panel, self._sample_coordinates, order=1)

    def predict(self, phase_grid: np.ndarray | None) -> np.ndarray | None:
        if self.forward is None or phase_grid is None:
            return None
        model_phase = self._model_phase(phase_grid)
        return self.forward.predict(model_phase)[0].detach().cpu().numpy()


def probes(size: int, count: int, seed: int, periods=(64, 128)):
    from ao_shaping.tools.slm.transfer.probes import probe_grid

    zero = np.zeros((size, size), np.float32)
    yield "reference_start", "calibration", zero
    yy, xx = np.mgrid[:size, :size] * (1920 / size)
    for axis, coord in (("x", xx), ("y", yy)):
        for period in periods:
            for sign in (-1, 1):
                yield (
                    f"ramp_{axis}_{period}_{sign}",
                    "calibration",
                    sign * 2 * np.pi * coord / period,
                )
    for index in range(count):
        family, grid = probe_grid(index, size, seed)
        # All families occur in both disjoint splits.
        yield (
            f"probe_{index:03d}_{family}",
            "validation" if index % 6 >= 3 else "train",
            grid,
        )
    yield "reference_end", "calibration", zero


def _resolve_capture_exposure(args) -> tuple[int, str, float | None, dict]:
    """开场曝光三态种子解析: 返回 (步数上限, token 来源, 种子毫秒, 相机 kwargs)。"""
    exposure_steps = int(getattr(args, "exposure_max_steps", EXPOSURE_MAX_STEPS))
    if exposure_steps < 2:
        raise ValueError("exposure_max_steps must be >= 2")
    # 开场曝光不再写死 0.02: 三态种子策略见 exposure_policy (auto / default / 数字)。
    # None = 「别碰相机」, 于是 create_camera 连 exposure_time_ms 关键字都不收到。
    exposure_source = policy_source(getattr(args, "exposure_ms", None))
    seed_exposure = resolve_seed_exposure(
        getattr(args, "exposure_ms", None),
        camera=args.camera,
        cam_id=args.cam_id,
    )
    camera_kwargs: dict = (
        {} if seed_exposure is None else {"exposure_time_ms": seed_exposure}
    )
    return exposure_steps, exposure_source, seed_exposure, camera_kwargs


def _cap_sample_row(attempt, clocks, exposure, exposure_clocks, flat_frame_name, label, live_prediction, path, raw, setting, split, x, y):
# Manifest row build plus live-prediction annotation lifted verbatim
    row = dict(
        file=path,
        label=label,
        split=split,
        exposure_ms=exposure,
        frames=clocks,
        exposure_probe_clocks=exposure_clocks,
        exposure_setting=asdict(setting),
        peak=float(raw.max()),
        saturation_pixels=int((raw >= 255).sum()),
        exposure_attempts=attempt,
        spot_xy=[x, y],
        **clock(),
    )
    if label == "reference_start" and live_prediction[0] is not None:
        try:
            # 第一张平场读完后才有一次会话的零点；下一张曝光/
            # 探针帧开始即可预测，参考帧自身不伪造 predicted。
            live_prediction[0].attach_reference(row)
        except Exception as exc:  # policy: 物理预测模型构建失败 → 禁用 live 预测并继续采集
            logger.warning(
                f"Live capture prediction disabled while building the forward model: {exc!r}"
            )
            live_prediction[0] = None
    return row




def _cap_spot_xy(args, exposure, flat_frame_name, frame, label, session_zero, slm_serial):
# Probe spot location plus flat-frame sidecar lifted verbatim
    if label == "reference_start":
        x, y = locate(frame)
        session_zero = (x, y)
        # 平场参照帧 (平面相位下的 0 级斑) 落盘: 下次光路复用拿它
        # 做形状判比。少一帧形状证据不该让一次几十分钟的采集失败
        # (复用自动退回零点判比), 所以这里兜住能真实抛出的类型:
        # 惰性导入的 ImportError, 以及帧/曝光量纲不合的 TypeError
        # / ValueError。磁盘类错误由 save_flat_frame 自己吞掉并
        # 返回 None —— 不用宽 except, 见 test_conventions 的棘轮。
        try:
            from ao_shaping.tools.slm.transfer.optical_probe import (
                save_flat_frame,
            )

            flat_frame_name = save_flat_frame(
                args.out,
                frame,
                spot_xy=(x, y),
                exposure_ms=exposure,
                slm_serial=slm_serial[0],
            )
        except (ImportError, OSError, TypeError, ValueError) as exc:
            logger.warning(f"平场参照帧保存失败: {exc!r}")
    elif label.startswith("ramp_") and session_zero is not None:
        # 零级未被压制时 argmax 会锁零级 -> 屏蔽零级邻域后找一级。
        x, y = locate(frame, exclude_xy=session_zero)
    else:
        x, y = locate(frame)
    return flat_frame_name, session_zero, x, y




def _cap_record_dataset(args, exposure_frames, frame, gray, grid, panel, path, raw, samples, save_dataset):
# Save-dataset branch lifted verbatim
    if save_dataset:
        np.savez_compressed(
            args.out / path,
            raw_frames=raw,
            frame=frame,
            model_phase=grid.astype(np.float32),
            panel_phase=panel.astype(np.float32),
            panel_gray=gray,
            exposure_probe_frames=np.stack(exposure_frames),
        )
    samples[path] = dict(
        model_phase=grid.astype(np.float32), raw_frames=raw
    )
    return 




def _cap_reuse_guard(label, live_prediction, manifest):
# Checkpoint reuse guard branch lifted verbatim
    if (
        live_prediction[0] is not None
        and manifest
        and manifest[-1]["label"].startswith("ramp_")
        and not label.startswith("ramp_")
    ):
        try:
            # 最后一个 ramp 行已写入 manifest；在首个非 ramp 帧
            # 读图前建好理想模型，让该帧就能携带 predicted。
            live_prediction[0].attach_geometry(manifest)
        except Exception as exc:  # policy: 理想模型构建失败 → 禁用 live 预测并继续采集
            logger.warning(
                f"Live capture prediction disabled while building the ideal model: {exc!r}"
            )
            live_prediction[0] = None
    return 




def _cap_verify_light(args, cam, ceiling, floor, read):
# Optical-path light verification plus sidecar lifted verbatim
    light = verify_light(
        read,
        exposure_setter(cam),
        floor_ms=floor,
        ceiling_ms=ceiling,
        label=f"capture {args.out.name}",
    )
    (args.out / "light_check.json").write_text(
        json.dumps({**light, **clock()}, default=str, indent=2),
        encoding="utf-8",
    )
    return 




def _cap_live_reader(cam, live_label, live_phase, live_prediction, live_seen, observer):
# Live frame reader closure factory lifted verbatim
    def read():
        raw = np.asarray(cam.get_numpy_image(n_sample=1, skip_first=False))
        if raw.ndim != 2:
            raise ValueError(
                f"Expected a monochrome 2D camera frame, got {raw.shape}"
            )
        live_seen[0] += 1
        predicted = None
        if live_prediction[0] is not None:
            try:
                predicted = live_prediction[0].predict(live_phase[0])
            except Exception as exc:
                # 第一次失败即停用，避免每帧重复报错；原始测量不受影响。
                logger.warning(
                    f"Live capture prediction disabled after a frame failed: {exc!r}"
                )
                live_prediction[0] = None
        notify_frame(
            observer,
            frame_payload(
                frame=raw,
                label=live_label[0],
                phase="capture",
                phase_grid=live_phase[0],
                predicted=predicted,
                frame_index=live_seen[0],
            ),
        )
        return raw
    return read




def _cap_load_checkpoint(args, checkpoint_value, live_prediction, observer):
# Observer checkpoint loading lifted verbatim
    if observer is not None:
        checkpoint = Path(checkpoint_value) if checkpoint_value else None
        try:
            if checkpoint is None or checkpoint.exists():
                # 无 checkpoint 也创建预测器：平场定位后、本次 signed-ramp
                # 几何到齐时直接构造零校正理想模型，而非等离线 bootstrap。
                live_prediction[0] = _LivePrediction(
                    checkpoint,
                    capture_grid=args.grid,
                    device=getattr(args, "device", None),
                )
            else:
                logger.warning(
                    f"Live capture prediction checkpoint does not exist: {checkpoint}"
                )
        except Exception as exc:
            # 实时预测只是辅助画面；模型不可用时采集仍须继续。
            logger.warning(
                f"Live capture prediction disabled during initialization: {exc!r}"
            )
    return 



def capture(args, *, on_frame=None):
    """Measure the probe plan; return ``{manifest file name: fit payload}``.

    ``on_frame(payload)`` is an optional live observer. Every CCD read in this
    stage -- the light check, each exposure-ladder rung, and every probe frame --
    delivers one frame through it (see :func:`.live_frame.frame_payload`). It is
    best-effort: a raising observer is logged, never propagated.

    The returned mapping is what ``fit.load_dataset(samples=...)`` consumes, so
    the fit stage never needs the npz files. Payload keys are exactly the ones
    the fit reads (``model_phase`` on the control grid, ``raw_frames`` averaged
    by the fit); ``panel_phase`` / ``panel_gray`` / ``frame`` /
    ``exposure_probe_frames`` are written only on the ``save_dataset`` path.

    Memory: ``raw_frames`` is kept as the uint8 stack (~15 MB for 3 frames of
    1944x2592), i.e. ~0.5 GB for the 34-entry default plan. The fit only ever
    averages it, so a mean-frame payload would cut that to ~40 MB — deliberately
    not done, because that would change what the fit consumes.
    """
    from .hardware import opened_device, prepare_camera_sdk

    prepare_camera_sdk(args.camera)
    from ao_shaping.drivers.ccd.common import (
        create_camera,
        get_camera_exposure_ms,
        get_camera_exposure_range,
    )
    from ao_shaping.drivers.slm.santec import Santec
    from ao_shaping.tools.slm.panel.focus_probe import adapt_exposure
    from ao_shaping.utils.image.exposure_control import configured_daheng_exposure_limit

    save_dataset = bool(getattr(args, "save_dataset", False))
    observer = _resolve_observer(args, on_frame)
    exposure_steps, exposure_source, seed_exposure, camera_kwargs = (
        _resolve_capture_exposure(args)
    )
    args.out.mkdir(parents=True, exist_ok=False)
    config = {
        **vars(args),
        **clock(),
        "save_dataset": save_dataset,
        "exposure_adaptation": dict(
            max_steps=exposure_steps, ratio_bounds=list(EXPOSURE_RATIO_BOUNDS)
        ),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (args.out / "config.json").write_text(
        json.dumps(config, default=str, indent=2), encoding="utf-8"
    )
    manifest, samples = [], {}
    early_stopped = False
    slm_serial: list[str | None] = [None]
    flat_frame_name: str | None = None
    with opened_device(
        create_camera(args.camera, args.cam_id, **camera_kwargs)
    ) as cam:
        with opened_device(
            Santec(slm_number=args.slm_number, wavelength=1064, video_mode=0)
        ) as slm:
            # 读取这台 SLM 的设备标识 (open() 时已读到); 供 workflow 在 fit/
            # 重训后按序列号保存最新模型。读不到时保持 None —— 只是不留序列号,
            # 不报错 (离线/仿真面板没有该属性或读不到)。
            try:
                slm_serial[0] = getattr(slm, "serial_number", None)
            except Exception as exc:  # policy: 序列号读不到仅跳过按序列号保存（离线/仿真面板无此属性）
                logger.warning(f"无法读取 SLM 序列号, 跳过最新模型按序列号保存: {exc!r}")
            slm.set_phase_overlay(None)
            try:
                floor, ceiling = get_camera_exposure_range(cam)
                limit = (
                    configured_daheng_exposure_limit() if args.camera == "daheng" else 0
                )
                if limit > 0:
                    ceiling = min(ceiling, limit)
                # 阶梯起点: 有种子就 max(floor, seed) (与改动前的 max(floor, 0.02) 同式),
                # 没有种子 (``default``) 就从相机自己回读的生效曝光起步。
                exposure = (
                    max(floor, seed_exposure)
                    if seed_exposure is not None
                    else float(get_camera_exposure_ms(cam))
                )

                # 实时画面: read() 是这一段的唯一取图漏斗 —— 光路检查、曝光阶梯
                # (probe_read) 与探针取帧都从它取图, 所以一处挂钩覆盖三处。
                # 标签随探针更新; 帧序号在整段里单调, 便于对照 manifest。
                live_label = [f"光路检查 · {args.out.name}"]
                live_seen = [0]
                # SLM 当前被驱动到的**控制网格**相位 (与 validate.py 的约定一致:
                # ``phase_grid`` 是控制网格, 不是面板位图)。``read()`` 是光路检查 /
                # 曝光阶梯 / 探针取帧共用的单一读取漏斗, 而这三个来源的相位各不相同,
                # 所以用一个可变单元格跟踪「此刻 SLM 上是什么相位」, 在每次真的写进
                # SLM 之前落值。没有有效相位 (尚无探针、也没有可供显示的网格) 就记
                # None —— 绝不伪造一个相位; 但它不是坏状态, 只是还没轮到驱动 SLM。
                live_phase = [None]
                live_prediction = [None]
                checkpoint_value = getattr(args, "checkpoint", None)
                _cap_load_checkpoint(args, checkpoint_value, live_prediction, observer)

                read = _cap_live_reader(cam, live_label, live_phase, live_prediction, live_seen, observer)

                # 便宜的光路检查, 在探针族之前: 2026-10-28 光路失效那次, 单条探针
                # 先烧掉整个阶梯 (每档 0.6 s settle + 落盘) 才报「曝光不收敛」,
                # 一条 lane 几十秒换来一句无法诊断的话。这里先显示平场 (第一号
                # 探针 reference_start 本来就是它, 所以不多一次显示语义), 再用
                # 0.05/0.4/3.2 ms 三档少数几帧回答「有没有光」; 没有光就直接抛
                # OpticalPathInvalid (exit code 3), 不进探针族。
                slm.display_data(
                    slm.create_phase_from_array(
                        to_panel(np.zeros((args.grid, args.grid), np.float32))
                    )
                )
                live_phase[0] = np.zeros((args.grid, args.grid), np.float32)
                time.sleep(0.6)
                _cap_verify_light(args, cam, ceiling, floor, read)

                # reference_start 是本会话零级；之后 ramp 帧用它屏蔽零级邻域再定位。
                session_zero: tuple[float, float] | None = None
                for label, split, grid in probes(
                    args.grid,
                    args.count,
                    args.seed,
                    getattr(args, "ramp_periods", (64, 128)),
                ):
                    live_label[0] = f"探针 {label}"
                    # The parallel-fit worker (or an operator) may ask capture to
                    # stop early; honour it before taking the next probe.
                    if (args.out / "stop_acquisition.request").is_file():
                        early_stopped = True
                        break
                    _cap_reuse_guard(label, live_prediction, manifest)
                    panel = to_panel(grid)
                    gray = slm.create_phase_from_array(panel)
                    slm.display_data(gray)
                    live_phase[0] = grid
                    time.sleep(0.6)
                    exposure_frames, exposure_clocks = [], []

                    def probe_read(_clocks=exposure_clocks, _frames=exposure_frames):
                        _clocks.append(
                            {
                                **clock(),
                                "exposure_ms": float(get_camera_exposure_ms(cam)),
                            }
                        )
                        frame = read()
                        if save_dataset:
                            _frames.append(frame.copy())
                        return frame

                    budgets = (exposure_steps,) + (
                        EXPOSURE_RETRY_STEPS,
                    ) * EXPOSURE_RETRIES
                    for attempt, steps in enumerate(budgets, start=1):
                        setting = adapt_exposure(
                            cam.reset_exposure_time,
                            probe_read,
                            start_ms=exposure,
                            floor_ms=floor,
                            ceiling_ms=ceiling,
                            full_scale=255,
                            max_steps=steps,
                            ratio_bounds=EXPOSURE_RATIO_BOUNDS,
                            on_step=LightGuard(
                                label=f"probe {label}", max_steps=steps
                            ).observe,
                        )
                        exposure = float(get_camera_exposure_ms(cam))
                        frames, clocks = [], []
                        for _ in range(args.frames):
                            stamp = clock()
                            frames.append(read().copy())
                            clocks.append(
                                {
                                    **stamp,
                                    "end_monotonic_ns": time.monotonic_ns(),
                                    "exposure_ms": float(get_camera_exposure_ms(cam)),
                                }
                            )
                        raw = np.stack(frames)
                        frame = raw.mean(0)
                        if setting.status == "ok" and raw.max() < 250:
                            break
                        # 阶梯停在了接受带外 (clipped = 峰值达到量程饱和, dim = 到上限仍太暗)。
                        # 从阶梯**算好、但还没测过**的那一档接着走, 而不是把整条 lane
                        # 判死: 撞过头只是这一步跨度太大, 换一档就能回来。
                        if attempt < len(budgets):
                            logger.warning(
                                f"{label}: exposure ladder ended {setting.status} "
                                f"(measured {exposure:.5f} ms -> peak={setting.peak:.1f} "
                                f"fraction={setting.fraction:.3f}, attempt={attempt}); "
                                f"resuming at {float(setting.exposure_ms):.5f} ms with "
                                f"{budgets[attempt]} steps"
                            )
                            exposure = float(setting.exposure_ms)
                    path = f"sample_{len(manifest):04d}.npz"
                    _cap_record_dataset(args, exposure_frames, frame, gray, grid, panel, path, raw, samples, save_dataset)
                    flat_frame_name, session_zero, x, y = _cap_spot_xy(args, exposure, flat_frame_name, frame, label, session_zero, slm_serial)
                    row = _cap_sample_row(attempt, clocks, exposure, exposure_clocks, flat_frame_name, label, live_prediction, path, raw, setting, split, x, y)
                    manifest.append(row)
                    (args.out / "manifest.json").write_text(
                        json.dumps(manifest, indent=2), encoding="utf-8"
                    )
                    with (args.out / "capture_records.pkl").open("wb") as stream:
                        pickle.dump(manifest, stream, protocol=5)
                    logger.info(
                        f"{label}: exposure={exposure:.5f} peak={raw.max():.1f} spot=({x:.1f},{y:.1f}) {setting.status}"
                    )
                    if setting.status != "ok" or raw.max() >= 250:
                        raise RuntimeError(
                            f"Unusable exposure for {label} after {attempt} ladder "
                            f"attempt(s): exposure={exposure:.5f} ms peak={raw.max():.1f} "
                            f"status={setting.status}; data retained, releasing devices"
                        )
            finally:
                slm.display_data(slm.create_phase_from_array(np.zeros((1200, 1920))))
                time.sleep(0.3)
        # 光路复用探针的**参照条件**: 本次采集用的相机曝光区间与取图格式。
        # 与实测零点 (拟合后写进 frozen_parameters.json 的 origin) 一起构成
        # 「这条光路的条件」, 供下次会话的光路复用判比用。
        from ao_shaping.tools.slm.transfer.optical_probe import (
            read_bit_depth as _camera_bit_depth,
            read_camera_format as _camera_format,
        )

        probe_record: dict = dict(
            exposure_range_ms=[float(floor), float(ceiling)],
            camera_format=_camera_format(cam),
            bit_depth=_camera_bit_depth(cam),
            slm_serial=slm_serial[0],
        )
        if flat_frame_name is not None:
            # 平场帧真的落盘了才登记文件名 (形状判比按这个键找帧)。
            probe_record["flat_frame"] = flat_frame_name
        (args.out / "_optical_probe.json").write_text(
            json.dumps(probe_record, indent=2),
            encoding="utf-8",
        )
        # 曝光阶梯落定 (相机还开着): 回读**生效**曝光并记住它 —— 这正是下一次同一条
        # 光路 `auto` 种子的来源。只在采集真的走完时记录: 阶梯没收住就没有可用曝光,
        # 记下来只会污染种子。记录本身best-effort, 绝不打断采集。
        effective_exposure = record_effective_exposure(
            cam,
            args.cam_id,
            exposure,
            source=exposure_source,
            backend=args.camera,
        )
        # 键名不能叫 `exposure_policy`: 那是 per_block / per_shape 的**配对曝光**策略
        # 字符串 (artifacts.py 与 validate.py 都按字符串读它)。这里记的是开场种子。
        config["exposure_seed_policy"] = {
            "source": exposure_source,
            "requested": getattr(args, "exposure_ms", None),
            "seed_ms": seed_exposure,
            "effective_ms": effective_exposure,
            "floor_ms": float(floor),
        }
        (args.out / "config.json").write_text(
            json.dumps(config, default=str, indent=2), encoding="utf-8"
        )
    (args.out / ".capture_done").write_text(
        json.dumps(
            dict(
                early_stopped=early_stopped,
                probes=len(manifest),
                slm_serial=slm_serial[0],
                **clock(),
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    return samples


def precheck_light(
    *,
    camera: str,
    cam_id: int,
    exposure_ms: str | float | None = None,
    label: str = "启动前光路检查",
    on_frame=None,
) -> dict:
    """开一次相机, 读少数几帧, 回答「这一路现在有没有光」。

    只给**跳过采集**的启动路径用 (``--model-dir`` 复用一份已拟合的模型)。正常
    路径里 :func:`capture` 已经在它自己打开的相机上做了同一件事 —— 同一实现
    :func:`ao_shaping.tools.slm.transfer.light_guard.verify_light`, 所以这里不再
    重复判据, 也不该被正常路径调用 (那会白开一次设备)。

    为什么跳过采集的路径需要它: 那份模型是在**原来那束光**上拟合的, 光路失效后
    继续用, 第一个变暗的读数要等到整形段第一块才出现, 而模型加载、目标与种子
    准备都已经跑完。这一步把坏光路挡在这些分钟级工作之前。

    面板上显示的是上一次会话留下的相位 —— 不影响「有没有光」的判定 (只影响光斑
    形状), 所以这里不为此再打开 SLM。

    Args:
        camera: ``daheng`` / ``sim`` 等相机类型。
        cam_id: 相机下标。
        exposure_ms: 开场曝光策略原值 (``auto`` / ``default`` / 正数毫秒, None=auto),
            与采集段同一个口径 —— 光检用的曝光档位不该和正式采集不一样。
        label: 写进异常的对象名。

    Returns:
        :func:`verify_light` 的判定字典 (已带实测数字)。

    Raises:
        OpticalPathInvalid: 每一档都只有噪声底。
    """
    from .hardware import opened_device, prepare_camera_sdk

    prepare_camera_sdk(camera)
    from ao_shaping.drivers.ccd.common import create_camera, get_camera_exposure_range
    from ao_shaping.utils.image.exposure_control import configured_daheng_exposure_limit

    seed_exposure = resolve_seed_exposure(exposure_ms, camera=camera, cam_id=cam_id)
    with opened_device(
        create_camera(
            camera,
            cam_id,
            **({} if seed_exposure is None else {"exposure_time_ms": seed_exposure}),
        )
    ) as cam:
        floor, ceiling = get_camera_exposure_range(cam)
        limit = configured_daheng_exposure_limit() if camera == "daheng" else 0
        if limit > 0:
            ceiling = min(ceiling, limit)

        def read():
            raw = np.asarray(cam.get_numpy_image(n_sample=1, skip_first=False))
            notify_frame(
                on_frame, frame_payload(frame=raw, label=label, phase="capture")
            )
            return raw

        return verify_light(
            read, exposure_setter(cam), floor_ms=floor, ceiling_ms=ceiling, label=label
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--camera", default="daheng")
    parser.add_argument("--cam-id", type=int, default=0)
    parser.add_argument("--slm-number", type=int, default=1)
    parser.add_argument("--grid", type=int, default=128)
    parser.add_argument("--ramp-periods", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--seed", type=int, default=102319)
    parser.add_argument("--exposure-max-steps", type=int, default=EXPOSURE_MAX_STEPS)
    parser.add_argument(
        "--exposure-ms",
        type=argparse_exposure_type("--exposure-ms"),
        default=AUTO_TOKEN,
        help="Seed exposure for the capture ladder (ms), not a fixed value: "
        "'auto' (default) reuses the last effective exposure recorded for this "
        "camera backend/id, falling back to 0.02; 'default' leaves the camera's "
        "own configured exposure untouched (the ladder starts from the read-back "
        "value); a positive number seeds the ladder at that many milliseconds.",
    )
    parser.add_argument(
        "--save-dataset",
        dest="save_dataset",
        action="store_true",
        default=True,
        help="Write one sample_*.npz per probe (the standalone recorder's "
        "default). Off keeps the measurements in memory only.",
    )
    parser.add_argument(
        "--no-save-dataset",
        dest="save_dataset",
        action="store_false",
        help="Skip the per-probe npz files; the fit stage consumes the "
        "in-memory payload instead",
    )
    args = parser.parse_args()
    if args.frames < 2 or args.count < 6:
        parser.error("frames >= 2 and count >= 6 required")
    plan = [
        (label, split, list(grid.shape))
        for label, split, grid in probes(args.grid, args.count, args.seed)
    ]
    if args.execute:
        # 光路无光是实验系统上的预期状态, 不是缺陷: 只报实测数字, 退出码 3, 不打
        # traceback (与探针的 2、普通异常的 1 分开)。
        try:
            capture(args)
        except OpticalPathInvalid as exc:
            logger.error(exc.report())
            raise SystemExit(exc.exit_code) from None
        except DeviceOpenError as exc:
            logger.error(str(exc))
            raise SystemExit(exc.exit_code) from None
    else:
        logger.info(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
