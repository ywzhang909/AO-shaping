"""One session directory for capture, transfer, shaping and independent audit."""

from __future__ import annotations
from loguru import logger

import argparse
import json
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ao_shaping.utils.image.target.metrics import POLYGON_SHAPE_CHOICES
from ao_shaping.utils.slm.bench_constants import TILT_SHIFT_SCALE

from .capture import clock
from .device import DEVICE_CHOICES, resolve_device
from .light_guard import OpticalPathInvalid
from .preflight import check_geometry, check_parameters


@dataclass(frozen=True)
class TransferSession:
    out: Path
    dataset: Path | None = None
    checkpoint: Path | None = None
    shapes: tuple[str, ...] = ("square", "rectangle", "triangle")
    count: int = 24
    iterations: int = 200
    patience: int = 8
    min_delta: float = 1e-4
    blocks: int = 4
    updates: int = 10
    frames: int = 3
    side: float = 80.0
    period: float = 64.0
    seed: int = 102319
    camera: str = "daheng"
    cam_id: int = 0
    slm_number: int = 1
    calibration: bool | None = None
    adaptation: str = "auto"
    accept_metric: str = "loss"
    objective: str = "shape"
    seed_run: Path | None = None
    seed_policy: str = "strict"
    #: Optional A–B–B–A paired CCD acceptance (default off). Off: the
    #: candidate is accepted on the optimizer's own loss and no CCD verdict is
    #: measured -- the summary then says ``acceptance="model_only"``.
    abba: bool = False
    #: Retrain loop driven by a CCD-vs-model acceptance threshold (default off).
    #: Off keeps the historical single-round behaviour exactly. On, ``run_session``
    #: fits, folds validation data in on retrains, and re-runs the CCD validation
    #: stages until the bound below is met, the user hits Stop, or the safety cap
    #: :attr:`acceptance_max_rounds` is reached.
    acceptance_enabled: bool = False
    #: Which acceptance score to gate the loop on. ``"quality"`` combines
    #: correlation and relative_l2 via :func:`validation_quality`.
    acceptance_metric: str = "quality"
    #: Algorithm-suggested lower bound on the mean validation correlation
    #: (also :func:`needs_full_adaptation`'s escalation edge). Must be in [0, 1].
    acceptance_lower: float = 0.85
    #: Algorithm-suggested upper bound on the mean validation relative_l2
    #: (also :func:`needs_full_adaptation`'s escalation edge). Must be > 0.
    acceptance_upper: float = 0.25
    #: Safety cap on retrain rounds; the loop stops gracefully when reached.
    acceptance_max_rounds: int = 8
    #: Lightweight optical-path reuse (default off). Off keeps the historical
    #: capture+fit+predict sequence byte-for-byte. On, ``run_session`` first runs
    #: a cheap :class:`~ao_shaping.tools.slm.transfer.optical_probe.OpticalProbe`
    #: comparing the current path (zero spot, exposure range, camera format) with
    #: the saved latest model's conditions; if unchanged it skips capture AND fit
    #: and jumps straight to the CCD-anchored blocks using the reused checkpoint.
    #: When reuse takes the fast path the acceptance loop is skipped entirely
    #: (the reused result is recorded as accepted).
    reuse_latest: bool = False
    #: Write one ``sample_*.npz`` per probe. Off (default) keeps the measurements
    #: in memory and hands them straight to the fit; on, the capture writes the
    #: same keys/content as before, for the offline ``--dataset`` path and the
    #: ``scripts/slm/`` analyses.
    save_dataset: bool = False
    #: Compute device for the torch stages (fit, synthesis, CCD blocks):
    #: ``"auto"`` follows the machine (CUDA when available, else CPU), ``"cpu"``
    #: reproduces the historical numbers, ``"cuda"`` fails loudly if unavailable.
    #: The *resolved* value is what ``session.json`` and the stage configs record.
    device: str = "auto"
    initial_phase: str = "gs"
    exposure_max_steps: int = 2
    phase_lr: float = 0.01
    phase_rms_cap: float = 0.2
    adaptive_step: bool = False
    adaptive_exposure: bool = False
    #: CCD sampling window for the anchored blocks: half-span and step in
    #: native CCD pixels. The 220/2 default is a small-target setting; a
    #: 1200-px target needs a half-span of at least ~640 or its box leaves the
    #: window and the run fails the boundary check.
    window_span: float = 220.0
    window_step: float = 2.0
    #: Radius (native CCD px) excluded from the core metric around the zero
    #: order, and the extra erosion applied to the core beyond the window step.
    zero_radius: float = 0.0
    core_margin: float | None = None
    #: Control grid of the probe capture; must tile the panel exactly.
    capture_grid: int = 128
    #: Opt-in (default off, behaviour unchanged): before a capture, raise
    #: ``capture_grid`` to the smallest valid grid whose FFT reach covers the
    #: requested target (:func:`decide_capture_grid`). Never lowers a grid.
    auto_capture_grid: bool = False
    #: ADC bit depth of the CCD blocks (8 or 12).
    bit_depth: int = 8
    #: Control resolution used for the shaping stages. ``0`` keeps the fit's own
    #: resolution; a finer grid is a *sampling* export of the same optical
    #: parameters (geometric_seed.export_frozen_parameters), which is what the
    #: 1200-px runs use so the phase is not limited by the probe grid.
    control_grid: int = 0
    fft_size: int = 2048
    #: Learning rate of the illumination-position parameter (panel px/step).
    #: ``None`` keeps the fit default; a disturbed bench whose beam moved on the
    #: panel needs more than the default to reach the measured position.
    amplitude_center_lr: float | None = None

    @property
    def calibration_enabled(self):
        return (
            self.calibration
            if self.calibration is not None
            else {"square", "triangle", "rectangle"}.issubset(self.shapes)
        )

    def validate(self):
        if str(self.device).strip().lower() not in DEVICE_CHOICES:
            raise ValueError(
                f"unknown device {self.device!r}; expected one of {list(DEVICE_CHOICES)}"
            )
        if self.exposure_max_steps < 2 or self.phase_lr <= 0 or self.phase_rms_cap <= 0:
            raise ValueError("invalid exposure search or phase step budget")
        if self.initial_phase not in {"gs", "plane", "random"}:
            raise ValueError("unknown initial phase")
        if self.seed_run and self.initial_phase != "gs":
            raise ValueError("cold initial phase cannot use a historical seed run")
        if self.objective not in {"shape", "cv", "formation"}:
            raise ValueError("unknown objective")
        if self.accept_metric not in {"loss", "cv"}:
            raise ValueError("unknown acceptance metric")
        if self.acceptance_metric not in {"quality"}:
            raise ValueError("unknown acceptance metric")
        if self.acceptance_max_rounds < 1:
            raise ValueError("acceptance_max_rounds >= 1 required")
        if not 0.0 <= self.acceptance_lower <= 1.0:
            raise ValueError("acceptance_lower must be in [0, 1]")
        if self.acceptance_upper <= 0.0:
            raise ValueError("acceptance_upper must be > 0")
        if self.seed_policy not in {"strict", "transfer"}:
            raise ValueError("unknown seed policy")
        if self.seed_run:
            for name in ("manifest.json", "config.json"):
                if not (self.seed_run / name).is_file():
                    raise FileNotFoundError(self.seed_run / name)
        if self.adaptation not in {"auto", "alignment", "full"}:
            raise ValueError("unknown adaptation mode")
        if min(self.count, self.frames) < 3 or self.count < 6 or self.iterations < 0:
            raise ValueError("count >= 6, frames >= 3 and iterations >= 0 required")
        if min(self.blocks, self.updates) < 1 or self.side <= 0 or self.period < 4:
            raise ValueError("invalid target or optimization parameters")
        if self.window_span <= 0 or self.window_step <= 0 or self.zero_radius < 0:
            raise ValueError("invalid camera window or zero-order exclusion")
        if self.bit_depth not in (8, 12):
            raise ValueError("bit depth must be 8 or 12")
        if (
            self.capture_grid < 16
            or 1920 % self.capture_grid
            or 1200 % (1920 // self.capture_grid)
        ):
            raise ValueError("capture grid must exactly tile the 1920x1200 panel")
        if self.control_grid and (
            1920 % self.control_grid
            or 1200 % (1920 // self.control_grid)
            or self.fft_size < self.control_grid
        ):
            raise ValueError(
                "control grid must tile the panel exactly and FFT must cover it"
            )
        if (
            not self.shapes
            or len(set(self.shapes)) != len(self.shapes)
            or not set(self.shapes) <= set(POLYGON_SHAPE_CHOICES)
        ):
            raise ValueError("unique supported shapes required")
        # Parameter-level pre-flight. A carrier period the calibration never
        # measured (or a target that cannot fit the sampling window at all) used
        # to be discovered minutes later, in the first CCD block, as "empty
        # target core" (2026-10-26: period 4 px -> 1865 px displacement).
        problems = check_parameters(
            side=float(self.side),
            period=float(self.period),
            count=int(self.count),
            iterations=int(self.iterations),
            blocks=int(self.blocks),
            updates=int(self.updates),
            frames=int(self.frames),
            shapes=tuple(self.shapes),
            half_span_px=float(self.window_span),
        )
        errors = [problem for problem in problems if problem.severity == "error"]
        if errors:
            first = errors[0]
            raise ValueError(f"{first.message} → {first.hint}")
        if self.dataset and not (self.dataset / "manifest.json").is_file():
            raise FileNotFoundError(self.dataset / "manifest.json")
        if self.checkpoint and not self.checkpoint.exists():
            raise FileNotFoundError(self.checkpoint)
        if self.calibration_enabled and not {
            "square",
            "triangle",
            "rectangle",
        }.issubset(self.shapes):
            raise ValueError(
                "calibration requires square/triangle/rectangle; use --skip-calibration for other selections"
            )

    def plan(self, offline=False):
        self.validate()
        note = "Device stages require --execute; nominal cold start is not a calibrated model"
        if not self.abba:
            note += (
                "; A–B–B–A acceptance skipped: candidates are accepted on the model loss "
                'only and no CCD verdict is measured (summary acceptance="model_only")'
            )
        return dict(
            settings=asdict(self),
            opens_devices=not offline,
            resolved=dict(
                adaptation=self.adaptation if self.checkpoint else "full",
                calibration=self.calibration_enabled,
                acceptance=(
                    "paired_abba"
                    if self.abba
                    else "looped"
                    if self.acceptance_enabled
                    else "model_only"
                ),
                acceptance_metric=self.acceptance_metric,
                acceptance_lower=self.acceptance_lower,
                acceptance_upper=self.acceptance_upper,
                acceptance_max_rounds=self.acceptance_max_rounds,
                save_dataset=self.save_dataset,
                # The request; ``run_session`` replaces it with the
                # resolved device before writing session.json.
                device=self.device,
            ),
            stages=([] if not self.reuse_latest else ["optical_probe"])
            + (["reuse_capture"] if self.dataset else ["capture"])
            + ([] if self.checkpoint else ["nominal_initialization"])
            + ["fit"]
            + (["control_resolution_export"] if self.control_grid else [])
            + (
                [
                    (
                        "transferred_phase_differentiable_synthesis"
                        if self.seed_run
                        else "GS_and_differentiable_synthesis"
                        if self.initial_phase == "gs"
                        else "cold_differentiable_synthesis"
                    )
                ]
                if offline
                else (
                    []
                    if (self.seed_run or self.initial_phase != "gs")
                    and not self.calibration_enabled
                    else ["CCD_model_blocks"]
                )
                + (
                    [
                        "exposure_response_capture",
                        "exposure_response_analysis",
                        "blur_calibration",
                    ]
                    if self.calibration_enabled
                    else []
                )
                + ["CCD_anchored_blocks", "raw_data_audit", "summary"]
                + (["direction_check"] if self.abba else [])
            ),
            note=note,
        )


@dataclass(frozen=True)
class CaptureGridDecision:
    """采集网格自动升档的结论（纯数据，写进 ``session.json`` 供回看）。"""

    #: 会话原本的网格 (``TransferSession.capture_grid``)。
    requested: int
    #: 决定采用的网格（``raised`` 为 False 时等于 ``requested``）。
    chosen: int
    #: 目标够得着的最小合法网格；``None`` = 任何合法网格都够不着。
    needed: int | None
    #: 是否真的升了档。
    raised: bool
    #: 目标包围盒边长（px）与位移（px），仅用于文案。
    extent_px: float
    offset_px: float
    #: 采用网格的可达半径（px）。
    reach_px: float


def decide_capture_grid(config, *, slope_px: float = TILT_SHIFT_SCALE) -> CaptureGridDecision:
    """按目标自动定采集网格（纯函数，不开设备、不抛）。

    关闭开关、已有数据集、或当前网格已够得着时，``chosen`` 等于 ``requested``
    （绝不降档）。升档时取 :func:`required_capture_grid` 的结果。位移优先用载波周期
    外推（``carrier_offset_px``），没有倾斜相位时目标留在零级，位移为 0。
    """
    from ao_shaping.algorithm.signal_processing.multishape_objective import (
        _fft_reach_px,
        carrier_offset_px,
        fft_max_side_px,
        required_capture_grid,
        target_extent_px,
    )

    requested = int(config.capture_grid)
    shapes = tuple(getattr(config, "shapes", ()) or ("square",))
    aspect = float(getattr(config, "aspect_ratio", 2.0))
    tilt = bool(getattr(config, "tilt", False))
    period = float(getattr(config, "period", 0.0) or 0.0)
    offset = carrier_offset_px(period, slope_px) if tilt else 0.0
    extent = target_extent_px(float(config.side), shapes=shapes, aspect_ratio=aspect)
    needed = required_capture_grid(
        float(config.side), slope_px, shapes=shapes, aspect_ratio=aspect,
        offset_px=offset,
    )
    enabled = bool(getattr(config, "auto_capture_grid", False))
    fits_now = extent <= fft_max_side_px(requested, slope_px, offset_px=offset) + 1e-9
    if not enabled or fits_now or needed is None or needed <= requested:
        chosen = requested
    else:
        chosen = needed
    return CaptureGridDecision(
        requested=requested, chosen=chosen, needed=needed,
        raised=chosen != requested, extent_px=extent, offset_px=offset,
        reach_px=_fft_reach_px(chosen, slope_px),
    )


def _notify(on_stage, name, status):
    """Best-effort observer call; the journal remains the source of truth."""
    if on_stage is None:
        return
    try:
        on_stage(name, status)
    except Exception as exc:  # a UI hook must never abort a measured session
        from loguru import logger

        logger.warning("transfer stage hook failed for {} ({}): {}", name, status, exc)


def _fit_quality(session_dir, fit_dir=None):
    """Mean validation correlation/relative_l2 from a finished fit directory.

    Reads the fit's ``result.json`` (per-row ``metrics``) against the manifest
    split in ``provenance.json`` (``rows``), restricted to non-``train`` rows,
    matching :func:`validation_quality`. ``fit_dir`` overrides the default
    ``<session_dir>/fit`` (the retrain loop names later rounds
    ``fit.roundN``). Returns ``None`` when any artifact is missing/incomplete --
    never raises.
    """
    fit_dir = Path(fit_dir) if fit_dir is not None else Path(session_dir) / "fit"
    result_path = fit_dir / "result.json"
    provenance_path = fit_dir / "provenance.json"
    try:
        if not result_path.is_file() or not provenance_path.is_file():
            return None
        result = json.loads(result_path.read_text(encoding="utf-8"))
        rows = json.loads(provenance_path.read_text(encoding="utf-8"))["rows"]
        metrics_list = result.get("metrics") or []
        if len(metrics_list) != len(rows):
            return None
        selected = [
            m for row, m in zip(rows, metrics_list) if row.get("split") != "train"
        ]
        if not selected:
            return None
        correlation = float(np.mean([m["correlation"] for m in selected]))
        relative_l2 = float(np.mean([m["relative_l2"] for m in selected]))
        if not np.isfinite([correlation, relative_l2]).all():
            return None
        return dict(correlation=correlation, relative_l2=relative_l2)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def acceptance_verdict(session_dir):
    """Derive the acceptance verdict for a finished session from files alone.

    Pure reader (GUI-facing): never touches devices, only reads ``session.json``
    (threshold config) and either the per-round ``rounds.json`` record written by
    :func:`run_session` or, failing that, the round-0 fit artifacts. Returns a
    dict with keys ``{metric, lower, upper, measured_correlation,
    measured_relative_l2, accepted, rounds}``. Bulletproof: any missing artifact
    yields a safe ``accepted=False`` verdict rather than raising.
    """
    session_dir = Path(session_dir)
    lower = 0.85
    upper = 0.25
    metric = "quality"
    try:
        session = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
        settings = session.get("settings", {})
        lower = float(settings.get("acceptance_lower", lower))
        upper = float(settings.get("acceptance_upper", upper))
        metric = str(settings.get("acceptance_metric", metric))
    except Exception:  # policy: 会话配置读取失败时沿用上方默认值，不中断验收判定
        logger.debug("读取会话验收配置失败，沿用默认值: {}", session_dir)
    rounds = []
    rounds_path = session_dir / "rounds.json"
    if rounds_path.is_file():
        try:
            loaded = json.loads(rounds_path.read_text(encoding="utf-8"))
            rounds = [dict(record) for record in loaded if isinstance(record, dict)]
        except Exception:  # policy: 轮次记录损坏时回退为空列表，后续按无历史轮次处理
            rounds = []
    measured_correlation = None
    measured_relative_l2 = None
    if rounds:
        last = dict(rounds[-1])
        measured_correlation = last.get("measured_correlation")
        measured_relative_l2 = last.get("measured_relative_l2")
    else:
        quality = _fit_quality(session_dir)
        if quality is not None:
            measured_correlation = quality["correlation"]
            measured_relative_l2 = quality["relative_l2"]
            rounds = [dict(round=0, **quality, accepted=None)]
    if measured_correlation is None or measured_relative_l2 is None:
        accepted = False
    else:
        accepted = (
            measured_correlation >= lower and measured_relative_l2 <= upper
        )
    return dict(
        metric=metric,
        lower=lower,
        upper=upper,
        measured_correlation=measured_correlation,
        measured_relative_l2=measured_relative_l2,
        accepted=accepted,
        rounds=len(rounds),
    )


def _write_reference_probe(fit_out: Path, dataset: Path, checkpoint: Path) -> None:
    """把本次会话的光路条件写成参照侧车, 随模型存进 ``latest``。

    零点来自本次拟合的 ``frozen_parameters.json`` (``origin``), 曝光区间与取图
    格式来自采集段写下的 ``_optical_probe.json``。缺哪一项就少写哪一项 ——
    下次复用判比时缺项视作「不可比」, 于是走正常全流程。

    Args:
        fit_out: 拟合输出目录 (侧车写在这里, ``save_latest_model`` 会一并复制)。
        dataset: 采集目录 (读采集写下的侧车)。
        checkpoint: 冻结参数文件 (读零点)。
    """
    from ao_shaping.tools.slm.transfer.optical_probe import FLAT_FROZEN_SIDECAR

    record: dict = {}
    try:
        frozen = json.loads(Path(checkpoint).read_text(encoding="utf-8"))
        origin = frozen.get("origin")
        if isinstance(origin, (list, tuple)) and len(origin) == 2:
            record["origin"] = [float(origin[0]), float(origin[1])]
    except (OSError, ValueError, TypeError):
        pass
    sidecar = Path(dataset) / FLAT_FROZEN_SIDECAR
    if sidecar.is_file():
        try:
            captured = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            captured = {}
        for key in ("exposure_range_ms", "camera_format", "bit_depth", "slm_serial"):
            if captured.get(key) is not None:
                record[key] = captured[key]
    try:
        (Path(fit_out) / FLAT_FROZEN_SIDECAR).write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )
    except OSError as exc:  # 写不了侧车只是少一条判比依据, 不阻断会话
        logger.warning("光路复用参照侧车写不下: {!r}", exc)


@dataclass(frozen=True)
class _ReusePlan:
    """光路复用快路径的结果: 复用哪一份 latest 模型, 以及为什么。"""

    #: 复用的冻结参数文件 (``latest/frozen_parameters.json``)。
    checkpoint: Path
    #: 复用的零点 (相机像元), 取自 latest 模型的 ``origin``。
    zero_xy: list[float]
    #: 本次实测探针。
    current: object
    #: latest 里记录的参照探针。
    reference: object
    #: 人类可读的判决文案。
    verdict: str


def _write_optical_probe_skip(
    out: Path,
    reason: str,
    *,
    current=None,
    serial=None,
    latest=None,
) -> None:
    """光路复用**被跳过**时写一份可读说明 ``<out>/optical_probe.json``。

    只写文件、不打开任何设备。复用是可选的加速, 写这份说明失败绝不能影响
    会话 → 全包 ``try/except``, 静默降级为一条警告日志 (绝不抛)。产物带
    ``skipped: True`` + ``reason`` (人类可读), 供「运行处」显示跳过原因;
    有可用的探针/序列号/latest 时一并记下, 便于人工排查。
    """
    try:
        payload = {"skipped": True, "reason": reason}
        if serial is not None:
            payload["serial"] = str(serial)
        if latest is not None:
            payload["latest"] = str(latest)
        if current is not None:
            payload["current"] = current.to_dict()
        out.mkdir(parents=True, exist_ok=True)
        (out / "optical_probe.json").write_text(
            json.dumps(payload, ensure_ascii=False, default=str, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:  # 记录写失败不影响会话
        logger.warning("光路复用: 写跳过说明失败: {!r}", exc)


def _attempt_optical_reuse(config: TransferSession, out: Path, *, device=None, on_stage=None):
    """试一次光路复用: 开设备 → 读序列号 → 找 latest → 量探针 → 比对。

    任何一步失败 (设备打不开、没有 latest、探针读不出、光路已变) 都返回
    ``None`` —— 调用方据此走正常全流程。**不抛异常**: 复用是可选加速, 不是
    会话成败的判据。

    Args:
        config: 本次会话配置。
        out: 会话输出目录 (写探针记录)。
        device: 已解析的 torch 设备 (本函数不用于计算, 只透传记录)。
        on_stage: 阶段观察者 (被通知 ``optical_probe`` 的 running/complete/failed)。

    Returns:
        :class:`_ReusePlan`, 或 None (不能复用)。
    """
    from . import optical_probe
    from .latest_model import models_root

    del device  # 探针不需要 torch 计算
    _notify(on_stage, "optical_probe", "running")
    try:
        # 探针自己开相机 + SLM (一次), 并在打开后读序列号 (与 capture.py 同口径)。
        current = optical_probe.measure_probe(
            camera=config.camera,
            cam_id=config.cam_id,
            slm_number=config.slm_number,
        )
    except Exception as exc:  # 探针失败不是会话失败
        logger.warning("光路复用: 探针失败, 走正常全流程: {!r}", exc)
        _write_optical_probe_skip(
            out,
            f"{optical_probe.REUSE_SKIP_PROBE_FAILED}："
            f"{type(exc).__name__}：{str(exc)[:120]}",
        )
        _notify(on_stage, "optical_probe", "failed")
        return None
    serial = current.slm_serial
    if not serial:
        logger.info("光路复用: 读不到 SLM 序列号, 走正常全流程")
        _write_optical_probe_skip(
            out, optical_probe.REUSE_SKIP_NO_SERIAL, current=current
        )
        _notify(on_stage, "optical_probe", "complete")
        return None
    latest = models_root() / str(serial) / "latest"
    if not (latest / "frozen_parameters.json").is_file():
        logger.info("光路复用: {} 没有 latest 模型, 走正常全流程", latest)
        _write_optical_probe_skip(
            out,
            optical_probe.REUSE_SKIP_NO_LATEST,
            current=current,
            serial=serial,
        )
        _notify(on_stage, "optical_probe", "complete")
        return None
    reference = optical_probe.probe_from_latest(latest)
    same = optical_probe.same_path(current, reference)
    verdict = optical_probe.format_probe_verdict(
        same, current=current, reference=reference
    )
    logger.info(verdict)
    (out / "optical_probe.json").write_text(
        json.dumps(
            dict(
                serial=str(serial),
                latest=str(latest),
                same_path=bool(same),
                current=current.to_dict(),
                reference=None if reference is None else reference.to_dict(),
                verdict=verdict,
            ),
            default=str,
            indent=2,
        ),
        encoding="utf-8",
    )
    _notify(on_stage, "optical_probe", "complete")
    if not same:
        return None
    # 复用: 零点取自 latest 模型自身的 ``origin`` (拟合时写入的实测零点)。
    try:
        frozen = json.loads(
            (latest / "frozen_parameters.json").read_text(encoding="utf-8")
        )
        origin = frozen["origin"]
        zero_xy = [float(origin[0]), float(origin[1])]
    except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
        logger.warning("光路复用: latest 模型零点读不出, 走正常全流程: {!r}", exc)
        return None
    # 网格充分性: 光路相同但 latest 网格更窄时, 复用会把够不着的目标原样交回。
    # 需要的网格取自本次会话 (同一份 decide_capture_grid 规则); 任何异常都只跳过复用。
    try:
        needed = decide_capture_grid(config).needed
        have = int(frozen.get("grid_size", 0))
        if needed is not None and have > 0 and have < needed:
            logger.info("光路复用: latest 网格 {} < 本次目标需要的 {}, 走正常全流程", have, needed)
            _write_optical_probe_skip(
                out, optical_probe.REUSE_SKIP_GRID, current=current, serial=serial, latest=latest
            )
            return None
    except Exception as exc:  # 复用是可选加速, 判定失败不影响会话
        logger.warning("光路复用: 网格充分性判定失败, 走正常全流程: {!r}", exc)
        return None
    return _ReusePlan(
        checkpoint=latest / "frozen_parameters.json",
        zero_xy=zero_xy,
        current=current,
        reference=reference,
        verdict=verdict,
    )


def _run_reused_session(
    config: TransferSession,
    out: Path,
    reuse: "_ReusePlan",
    *,
    device,
    on_stage=None,
    on_frame=None,
    stage,
    validate,
    summary,
    direction,
):
    """光路复用命中后的会话: 跳过采集与拟合, 直接跑 CCD 锚定块。

    只跑 CCD-anchored 块 (不跑 CCD-model 块, 不跑曝光响应标定) —— 复用的
    模型已经是「这条光路上的标定模型」, 缺的是这次会话的**实测**判定。

    复用路径下**跳过验收回路**: 没有新的拟合, 就没有可比的验证集指标, 复用
    结果直接记为 accepted (见 ``rounds.json``)。

    Returns:
        会话输出目录。
    """
    from .artifacts import audit_run

    anchored_out = out / "anchored"
    checkpoint = reuse.checkpoint
    zero = list(reuse.zero_xy)
    options = dict(
        checkpoint=checkpoint,
        zero_xy=zero,
        period=config.period,
        side=config.side,
        window_span=config.window_span,
        window_step=config.window_step,
        zero_radius=config.zero_radius,
        core_margin=config.core_margin,
        bit_depth=config.bit_depth,
        initial_phase=config.initial_phase,
        random_seed=config.seed,
        phase_lr=config.phase_lr,
        phase_rms_cap=config.phase_rms_cap,
        adaptive_step=config.adaptive_step,
        adaptive_exposure=config.adaptive_exposure,
        accept_metric=config.accept_metric,
        objective=config.objective,
        shapes=list(config.shapes),
        updates=config.updates,
        frames=config.frames,
        abba=config.abba,
        acceptance_enabled=bool(config.acceptance_enabled),
        device=device,
        camera=config.camera,
        cam_id=config.cam_id,
        slm_number=config.slm_number,
        tilt=getattr(config, "tilt", False),
        aspect_ratio=getattr(config, "aspect_ratio", 2.0),
    )
    # 复用路径没有 ``model`` 目录可播种, 因此 ``seed_run=None``: 锚定块走
    # GS 种子 (validate.run 的 gs_seed 分支), 用复用的冻结模型做前向。
    stage(
        "CCD_anchored_blocks",
        lambda: validate.run(
            SimpleNamespace(
                **options,
                out=anchored_out,
                seed_run=None,
                seed_policy="strict",
                mode="anchored",
                blocks=config.blocks,
                execute=True,
                on_frame=on_frame,
            )
        ),
    )

    def audit():
        result = audit_run(anchored_out)
        (out / "audit.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )

    stage("raw_data_audit", audit)
    # 复用路径没有新的拟合 → 没有可比的验证集指标 → 复用结果直接记为 accepted。
    rounds = [dict(round=0, train_all=False, reused=True, accepted=True)]
    (out / "rounds.json").write_text(json.dumps(rounds, indent=2), encoding="utf-8")
    stage("summary", lambda: summary.summarize(anchored_out, out / "summary"))
    if config.abba:
        stage("direction_check", lambda: direction.analyze(anchored_out, out / "direction"))
    logger.info("光路复用: 跳过采集+拟合, 直接复用 latest 模型跑 CCD 锚定块")
    print(f"Session complete: {out}", flush=True)
    return out


def _acquire_session_dataset(config, *, out, device, stage, on_frame, offline):
    """Capture a fresh dataset, or pre-check the path of a recorded one.

    Verbatim from the prologue of ``run_session``: with no recorded dataset the
    capture stage runs and its manifest feeds the early geometry gate; with a
    recorded dataset the capture is skipped entirely, so the cheap light check is
    the only thing that can notice a broken optical path before the fit.

    Returns ``(dataset, samples)``; ``samples`` is the in-memory capture payload
    (``None`` when a recorded dataset was reused, which is what tells the fit to
    read from disk).
    """
    from . import capture

    dataset = config.dataset.resolve() if config.dataset else out / "capture"
    samples = None
    if config.dataset is None:
        samples = stage(
            "capture",
            lambda: capture.capture(
                SimpleNamespace(
                    out=dataset,
                    camera=config.camera,
                    cam_id=config.cam_id,
                    slm_number=config.slm_number,
                    grid=config.capture_grid,
                    count=config.count,
                    frames=config.frames,
                    seed=config.seed,
                    execute=True,
                    exposure_max_steps=config.exposure_max_steps,
                    save_dataset=config.save_dataset,
                    checkpoint=config.checkpoint,
                    device=device,
                ),
                on_frame=on_frame,
            ),
        )
        # 采集刚结束 (不是复用已录数据集): 用这一批 manifest 行提前判「倾斜 + 目标
        # 边长」是否越界, 不等整轮拟合跑完再暴露 (2026-10-26 "empty target core"
        # 的教训)。与拟合后判同一把尺子 (同 batch 行现拟合倾斜相位矩阵), 是提前量而
        # 非另一套近似; 条件不足 (无有效倾斜相位/零点) 时不阻断。
        if config.dataset is None:
            from .early_geometry import check_early_geometry, format_early_geometry

            rows = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
            early = check_early_geometry(
                rows,
                side=config.side,
                period=config.period,
                shapes=config.shapes,
                half_span_px=config.window_span,
                step_px=config.window_step,
                tilt=getattr(config, "tilt", False),
            )
            if early.ok:
                text = format_early_geometry(early)
                if text:
                    logger.info("early geometry check passed:\n{}", text)
            else:
                raise RuntimeError(
                    "采集阶段几何预检拦截越界: \n"
                    + format_early_geometry(early)
                )
    elif not offline:
        # 复用已录数据集时采集段整个被跳过, 于是**没有任何人**看一眼光路就让拟合
        # 与整形跑起来。这里补一次便宜检查 (同一实现 verify_light): 2026-10-28
        # 的教训是光路失效要几十秒到几分钟才暴露, 而这一路本来可以几秒就停。
        stage(
            "light_check",
            lambda: capture.precheck_light(
                camera=config.camera,
                cam_id=config.cam_id,
                label=f"session {config.out.name}",
                on_frame=on_frame,
            ),
        )
    return dataset, samples


def _round_geometry_preflight(config, *, dataset, checkpoint):
    """Locate the measured zero order and fail early on an out-of-window target.

    Verbatim from the round loop of ``run_session``.  The parameter check upstream
    cannot see this combination: it needs the *measured* ramp gain, and the
    alternative was discovering the problem in the first CCD block as an "empty
    target core" (2026-10-26).

    Returns ``(zero, reference)`` -- the rounded zero-order position and the
    ``reference_start`` manifest row the synthesis needs for its exposure.
    """
    from . import fit

    rows = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    reference = next(
        r for r in rows if r["label"] == "reference_start" and fit.valid(r)
    )
    zero = np.round(fit.spot(reference)).tolist()
    # Geometry pre-flight: with the *measured* zero and the *fitted* ramp gain,
    # recompute where the target lands in this session's sampling window. The
    # parameter check above cannot see that combination (it needs the measured
    # gain), and the alternative was discovering it in the first CCD block as
    # "empty target core" (2026-10-26). Fail here, with the numbers that say
    # which knob to turn.
    frozen = json.loads(checkpoint.read_text(encoding="utf-8"))
    geometry_problems = check_geometry(
        zero_xy=zero,
        ramp_matrix=np.asarray(frozen["ramp_matrix"], dtype=float),
        side=float(config.side),
        period=float(config.period),
        shapes=tuple(config.shapes),
        half_span_px=float(config.window_span),
        step_px=float(config.window_step),
        tilt=getattr(config, "tilt", False),
    )
    geometry_errors = [
        problem for problem in geometry_problems if problem.severity == "error"
    ]
    if geometry_errors:
        first = geometry_errors[0]
        raise ValueError(f"{first.message} → {first.hint}")
    return zero, reference


def _offline_synthesize(config, *, checkpoint, zero, reference, out, device):
    """Synthesize shaping phases from the frozen model, without any device.

    Verbatim from the nested ``synthesize`` closure of ``run_session``'s offline
    branch.  The seed comes from ``--seed-run``, a cold start, or GS depending on
    ``config.initial_phase``; every shape writes its own npz plus the loss trace,
    and the run is stamped with the device that actually produced it.
    """
    from . import validate

    from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
        FrozenDefocusForward,
        load_frozen_parameters,
        recentre_window,
    )

    params = load_frozen_parameters(checkpoint)
    xy = recentre_window(
        tuple(zero),
        half_span_px=float(config.window_span),
        step_px=float(config.window_step),
    )
    forward = FrozenDefocusForward(params, xy, device=device)
    forward.exposure_ms = float(reference["exposure_ms"])
    tilt = bool(getattr(config, "tilt", False))
    shift = (
        np.asarray(params.ramp_matrix)[:, 0] / config.period
        if tilt
        else np.zeros(2, dtype=float)
    )
    center = np.asarray(zero) + shift
    aspect = float(getattr(config, "aspect_ratio", 2.0))
    for shape in config.shapes:
        target = validate.target_at(
            xy, shape, center, config.side, aspect_ratio=aspect
        )
        if config.seed_run:
            from .seeding import accepted_seed

            phase, _ = accepted_seed(
                config.seed_run,
                shape,
                forward,
                xy,
                target,
                policy=config.seed_policy,
                centre=center,
            )
        elif config.initial_phase != "gs":
            from .cold_start import cold_phase

            phase, _ = cold_phase(
                forward, center, kind=config.initial_phase, seed=config.seed
            )
        else:
            phase = validate.gs_seed(forward, center, config.side, shape)
        objective = validate.ShapeObjective(
            target,
            cv_only=config.objective == "cv",
            formation=config.objective == "formation",
        )
        from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
            DifferentiableUniformityOptimizer,
        )

        optimizer = DifferentiableUniformityOptimizer(
            forward,
            objective,
            phase,
            lr=config.phase_lr,
            step_rms_cap=config.phase_rms_cap,
            device=device,
        )
        for _ in range(config.updates):
            optimizer.update()
        if not np.isfinite(optimizer.current_phase).all():
            raise RuntimeError("nonfinite offline optimized phase")
        np.savez_compressed(
            out / f"{shape}_offline_seed.npz",
            model_phase=phase,
            optimized_phase=optimizer.current_phase,
            prediction=forward.predict(phase).detach().cpu().numpy(),
            optimized_prediction=forward.predict(optimizer.current_phase)
            .detach()
            .cpu()
            .numpy(),
            target=target,
            camera_xy=xy,
            exposure_ms=forward.exposure_ms,
        )
        (out / f"{shape}_offline_iterations.json").write_text(
            json.dumps(optimizer.loss_history, indent=2), encoding="utf-8"
        )
    # Self-describing: which device actually produced these phases.
    (out / "synthesis.json").write_text(
        json.dumps(
            dict(
                device=device,
                device_requested=config.device,
                shapes=list(config.shapes),
                initial_phase=config.initial_phase,
                updates=config.updates,
                **clock(),
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

def _round_calibration_stages(config, *, out, suffix, prefix, model_out, checkpoint,
                              options, stage, on_frame):
    """Exposure-response capture/analysis and the blur calibration for one round.

    Verbatim from the hardware branch of ``run_session``'s round loop, including
    its guard: a single-shape run leaves ``calibration_enabled`` false and none of
    these stages run.

    Mutates ``options["checkpoint"]`` to the blur-calibrated parameters, exactly
    as the inline block did -- the anchored blocks downstream read the calibrated
    model through ``options``, not through the local ``checkpoint``.
    """

    if config.calibration_enabled:
        from . import blur, capture_response, response

        response_out = out / f"response{suffix}"
        response_analysis_out = out / f"response_analysis{suffix}"
        blur_out = out / f"blur{suffix}"
        stage(
            f"{prefix}exposure_response_capture",
            lambda model_out=model_out, response_out=response_out: capture_response.capture(
                SimpleNamespace(
                    source=model_out,
                    out=response_out,
                    shape="square",
                    frames=config.frames,
                    camera=config.camera,
                    cam_id=config.cam_id,
                    slm_number=config.slm_number,
                    on_frame=on_frame,
                )
            ),
        )
        stage(
            f"{prefix}exposure_response_analysis",
            lambda response_out=response_out, response_analysis_out=response_analysis_out: response.analyze(
                response_out, response_analysis_out
            ),
        )
        stage(
            f"{prefix}blur_calibration",
            lambda model_out=model_out, checkpoint=checkpoint, blur_out=blur_out: blur.calibrate(
                SimpleNamespace(
                    source=model_out,
                    checkpoint=checkpoint,
                    out=blur_out,
                    sigmas=[0.0, 0.5, 1.0, 2.0, 3.515916109085083],
                )
            ),
        )
        options["checkpoint"] = blur_out / "frozen_parameters.json"

def run_session(
    config: TransferSession, *, offline=False, on_stage=None, on_frame=None,
    should_stop=None,
):
    """No subprocess chains; each completed/failed stage is journaled immediately.

    ``on_stage(name, status)`` is an optional observer called whenever a stage
    starts / completes / fails -- the same moment the journal line is written.
    It exists so the GUI can show progress through *this* entry point instead of
    re-implementing the stage sequence (a second orchestrator would drift).
    A raising observer must not wreck the session, so its errors are swallowed
    with a warning; the journal and the run itself stay authoritative.

    ``on_frame(payload)`` is the live CCD observer, forwarded unchanged to every
    stage that opens the camera: the capture stage, the standalone light check,
    the model and anchored blocks, and the exposure-response capture. Each of
    those fires it **on every CCD read** (see
    :func:`~ao_shaping.tools.slm.transfer.live_frame.frame_payload`), and every
    call site wraps it in the same best-effort guard. ``None`` (the CLI /
    offline default) leaves the measured path byte-for-byte unchanged.

    ``should_stop`` is an optional no-arg callback consulted between acceptance
    retrain rounds. When it returns True (the GUI's Stop button), the loop halts
    gracefully, keeps the best result so far, and does not error. Ignored unless
    ``config.acceptance_enabled``.
    """
    config.validate()
    if offline and config.dataset is None:
        raise ValueError("--offline requires a recorded --dataset")
    # 自动升档只对会采集的会话生效：录制数据集的网格由 manifest 决定，改了就与拟合对不上。
    grid_decision = None
    if config.dataset is None and getattr(config, "auto_capture_grid", False):
        grid_decision = decide_capture_grid(config)
        if grid_decision.raised:
            # dataclasses.replace 只搬字段：桌面用 object.__setattr__ 挂的 tilt /
            # aspect_ratio 运行时属性会丢，必须原样带回（下游用 getattr 读它们）。
            field_names = {f.name for f in fields(config)}
            extras = {k: v for k, v in vars(config).items() if k not in field_names}
            config = replace(config, capture_grid=grid_decision.chosen)
            for key, value in extras.items():
                object.__setattr__(config, key, value)
            logger.info(
                "采集网格自动升档: {} → {} (目标 {:g} px, 可达 ±{:.0f} px)",
                grid_decision.requested, grid_decision.chosen,
                grid_decision.extent_px, grid_decision.reach_px,
            )
    out = config.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    # Resolve the compute device once, here, so ``session.json`` records the one
    # the stages will actually use (``config.device`` is the *request*: "auto").
    device = resolve_device(config.device)
    # 在升档之后才生成 plan, 使 session.json 的 settings 记录的是实际采用的网格。
    plan = config.plan(offline)
    plan["resolved"]["device"] = device
    if grid_decision is not None:
        plan["resolved"]["capture_grid_decision"] = asdict(grid_decision)
    (out / "session.json").write_text(
        json.dumps(dict(**plan, **clock()), default=str, indent=2), encoding="utf-8"
    )
    journal = []

    def stage(name, operation):
        entry = dict(stage=name, status="running", **clock())
        journal.append(entry)
        write = lambda: (out / "journal.json").write_text(
            json.dumps(journal, default=str, indent=2), encoding="utf-8"
        )
        write()
        print(f"[{name}] started", flush=True)
        _notify(on_stage, name, "running")
        try:
            result = operation()
        except BaseException as exc:
            entry.update(
                status="failed", error=f"{type(exc).__name__}: {exc}", end=clock()
            )
            write()
            _notify(on_stage, name, "failed")
            raise
        entry.update(status="complete", end=clock())
        write()
        _notify(on_stage, name, "complete")
        return result

    from . import bootstrap, direction, fit, optical_probe, summary, validate

    # --- Optical-path reuse fast path (opt-in via ``reuse_latest``) ----------
    # Off (the default) the code below is byte-for-byte the historical one.
    # On, and only on, we first spend one cheap probe (camera + SLM open, one
    # frame) to compare the *current* optical path against the saved latest
    # model. Unchanged -> the capture and the fit are skipped and the session
    # jumps straight to the CCD-anchored blocks with the reused checkpoint.
    if config.reuse_latest and config.dataset is None and not offline:
        reuse = _attempt_optical_reuse(config, out, device=device, on_stage=on_stage)
        if reuse is not None:
            return _run_reused_session(
                config,
                out,
                reuse,
                device=device,
                on_stage=on_stage,
                on_frame=on_frame,
                stage=stage,
                validate=validate,
                summary=summary,
                direction=direction,
            )
    else:
        # 光路复用没走 (开关未开 / 数据集会话 / 离线): 写一行跳过说明,
        # 供「运行处」说明「为什么没走复用」。写记录失败不影响会话。
        if not config.reuse_latest:
            _write_optical_probe_skip(out, optical_probe.REUSE_SKIP_DISABLED)
        elif config.dataset is not None:
            _write_optical_probe_skip(out, optical_probe.REUSE_SKIP_DATASET)
        else:
            _write_optical_probe_skip(out, optical_probe.REUSE_SKIP_OFFLINE)

    dataset, samples = _acquire_session_dataset(
        config, out=out, device=device, stage=stage, on_frame=on_frame, offline=offline
    )
    checkpoint = (
        config.checkpoint.resolve()
        if config.checkpoint
        else out / "initial/frozen_parameters.json"
    )
    if config.checkpoint is None:

        def initialize():
            params, provenance = bootstrap.bootstrap(dataset)
            checkpoint.parent.mkdir()
            checkpoint.write_text(json.dumps(params, indent=2), encoding="utf-8")
            (checkpoint.parent / "provenance.json").write_text(
                json.dumps(provenance, indent=2), encoding="utf-8"
            )

        stage("nominal_initialization", initialize)
    # The capture stage's in-memory payloads go straight into the fit; only a
    # reused (recorded) dataset is read back from disk.
    # The fit and the CCD validation stages form one *round*: the acceptance
    # loop below repeats them. Round 0 keeps the historical names and paths
    # byte-for-byte; re-rounds get their own directories (``fit.round1`` ...)
    # because every one of these stages refuses to write into an existing
    # directory.
    max_rounds = config.acceptance_max_rounds if config.acceptance_enabled else 1
    rounds: list[dict] = []
    accepted = not config.acceptance_enabled
    stopped_by_user = False
    last_round = 0
    # The warm start stays the *initial* checkpoint for every round: each round
    # refits from the same nominal model with a different training fold, so the
    # rounds stay comparable (a warm start from the previous round would
    # confound "more data" with "better initial guess").
    warm_start = checkpoint
    for round_index in range(max_rounds):
        if round_index > 0:
            # The user may have pressed Stop while the previous round ran.
            if should_stop is not None:
                try:
                    stopped_by_user = bool(should_stop())
                except Exception as exc:  # a raising observer must not abort
                    logger.warning("acceptance stop callback failed ({}): {}", type(exc).__name__, exc)
                    stopped_by_user = False
                if stopped_by_user:
                    logger.warning("acceptance loop stopped by user before round {}", round_index)
                    break
            logger.warning(
                "acceptance threshold not met after round {}; retraining with validation data folded in",
                round_index - 1,
            )
        last_round = round_index
        suffix = "" if round_index == 0 else f".round{round_index}"
        fit_out = out / f"fit{suffix}"
        model_out = out / f"model{suffix}"
        anchored_out = out / f"anchored{suffix}"
        prefix = "" if round_index == 0 else f"acceptance_round_{round_index}_"
        # Round >= 1 folds the validation phases into training: the acceptance
        # bound was not met with them held out, so the extra data has to earn
        # its keep. The first round keeps the split so the bound is measured
        # on genuinely disjoint phases.
        train_all = round_index > 0
        stage(
            f"{prefix}fit",
            lambda fit_out=fit_out, train_all=train_all: fit.fit(
                SimpleNamespace(
                    dataset=dataset,
                    checkpoint=warm_start,
                    out=fit_out,
                    iterations=config.iterations,
                    patience=config.patience,
                    min_delta=config.min_delta,
                    span=120,
                    samples=samples,
                    device=device,
                    amplitude_center_lr=config.amplitude_center_lr,
                    adaptation=config.adaptation if config.checkpoint else "full",
                    train_all=train_all,
                )
            ),
        )
        checkpoint = fit_out / "frozen_parameters.json"
        if config.control_grid:
            from .geometric_seed import export_frozen_parameters

            control_out = out / f"model{config.control_grid}{suffix}"
            stage(
                f"{prefix}control_resolution_export",
                lambda checkpoint=checkpoint, control_out=control_out: export_frozen_parameters(
                    checkpoint, control_out, grid=config.control_grid, fft=config.fft_size
                ),
            )
            checkpoint = control_out / "frozen_parameters.json"
        zero, reference = _round_geometry_preflight(
            config, dataset=dataset, checkpoint=checkpoint
        )
        # Keep the latest fitted model per SLM serial, both on the first fit and
        # after every threshold retrain. Offline sessions have no serial and skip.
        from .latest_model import save_latest_model, serial_from_capture

        # 光路复用探针的**参照条件**随模型一起存进 latest: 曝光区间与取图格式
        # 来自本次采集写下的侧车, 零点来自本次拟合的 frozen_parameters.json。
        _write_reference_probe(fit_out, dataset, checkpoint)
        save_latest_model(checkpoint.parent, serial_from_capture(out))
        options = dict(
            checkpoint=checkpoint,
            zero_xy=zero,
            period=config.period,
            side=config.side,
            window_span=config.window_span,
            window_step=config.window_step,
            zero_radius=config.zero_radius,
            core_margin=config.core_margin,
            bit_depth=config.bit_depth,
            initial_phase=config.initial_phase,
            random_seed=config.seed,
            phase_lr=config.phase_lr,
            phase_rms_cap=config.phase_rms_cap,
            adaptive_step=config.adaptive_step,
            adaptive_exposure=config.adaptive_exposure,
            accept_metric=config.accept_metric,
            objective=config.objective,
            shapes=list(config.shapes),
            updates=config.updates,
            frames=config.frames,
            abba=config.abba,
            acceptance_enabled=bool(config.acceptance_enabled),
            device=device,
            camera=config.camera,
            cam_id=config.cam_id,
            slm_number=config.slm_number,
            tilt=getattr(config, "tilt", False),
            aspect_ratio=getattr(config, "aspect_ratio", 2.0),
        )
        if offline:

            stage(
                "transferred_phase_differentiable_synthesis"
                if config.seed_run
                else "GS_and_differentiable_synthesis"
                if config.initial_phase == "gs"
                else "cold_differentiable_synthesis",
                lambda checkpoint=checkpoint, zero=zero, reference=reference:
                _offline_synthesize(
                    config,
                    checkpoint=checkpoint,
                    zero=zero,
                    reference=reference,
                    out=out,
                    device=device,
                ),
            )
        else:
            from .artifacts import audit_run

            if (
                not config.seed_run and config.initial_phase == "gs"
            ) or config.calibration_enabled:
                stage(
                    f"{prefix}CCD_model_blocks",
                    lambda options=options, model_out=model_out: validate.run(
                        SimpleNamespace(
                            **options,
                            out=model_out,
                            seed_run=config.seed_run,
                            seed_policy=config.seed_policy,
                            mode="model",
                            blocks=2,
                            execute=True,
                            on_frame=on_frame,
                        )
                    ),
                )
                _round_calibration_stages(
                    config,
                    out=out,
                    suffix=suffix,
                    prefix=prefix,
                    model_out=model_out,
                    checkpoint=checkpoint,
                    options=options,
                    stage=stage,
                    on_frame=on_frame,
                )
            stage(
                f"{prefix}CCD_anchored_blocks",
                lambda options=options, anchored_out=anchored_out, model_out=model_out: validate.run(
                    SimpleNamespace(
                        **options,
                        out=anchored_out,
                        seed_run=config.seed_run
                        if not config.calibration_enabled
                        and (config.seed_run or config.initial_phase != "gs")
                        else model_out,
                        seed_policy=config.seed_policy
                        if config.seed_run and not config.calibration_enabled
                        else "strict",
                        mode="anchored",
                        blocks=config.blocks,
                        execute=True,
                        on_frame=on_frame,
                    )
                ),
            )

            def audit(anchored_out=anchored_out, suffix=suffix):
                result = audit_run(anchored_out)
                (out / f"audit{suffix}.json").write_text(
                    json.dumps(result, indent=2), encoding="utf-8"
                )

            stage(f"{prefix}raw_data_audit", audit)
            # The acceptance verdict for this round: the mean correlation and
            # relative_l2 over the fit's *validation* phases. A missing or
            # unreadable fit artifact yields ``None`` -> the round counts as
            # not accepted and the loop tries again (bounded by the cap).
            quality = _fit_quality(out, fit_out)
            measured_correlation = None if quality is None else quality["correlation"]
            measured_relative_l2 = None if quality is None else quality["relative_l2"]
            if quality is None:
                accepted = False
            else:
                accepted = (
                    measured_correlation >= config.acceptance_lower
                    and measured_relative_l2 <= config.acceptance_upper
                )
            rounds.append(
                dict(
                    round=round_index,
                    train_all=train_all,
                    measured_correlation=measured_correlation,
                    measured_relative_l2=measured_relative_l2,
                    accepted=accepted,
                    fit_directory=str(fit_out.relative_to(out)),
                    anchored_directory=str(anchored_out.relative_to(out)),
                )
            )
            if config.acceptance_enabled:
                # Only the looped path writes the round record: with the gate
                # off the session directory keeps exactly the artifacts it had
                # before.
                (out / "rounds.json").write_text(
                    json.dumps(rounds, indent=2), encoding="utf-8"
                )
            if not config.acceptance_enabled:
                break
            if accepted:
                logger.info(
                    "acceptance threshold met at round {}: correlation={:.4f} relative_l2={:.4f}",
                    round_index,
                    measured_correlation,
                    measured_relative_l2,
                )
                break
            logger.warning(
                "acceptance threshold not met at round {}: correlation={} relative_l2={} "
                "(need correlation >= {} and relative_l2 <= {})",
                round_index,
                measured_correlation,
                measured_relative_l2,
                config.acceptance_lower,
                config.acceptance_upper,
            )
        if offline:
            # The offline path has no CCD round to loop over; it stays a
            # single pass.
            break

    # The summary and the direction check read the *last* round's anchored
    # blocks, so they run once, after the loop converges or stops. The offline
    # path never produces anchored blocks, so it has no summary (unchanged).
    if not offline:
        summary_run = out / (f"anchored.round{last_round}" if last_round else "anchored")
        stage("summary", lambda: summary.summarize(summary_run, out / "summary"))
        if config.abba:
            # The direction check reads B1/B2 and the minus probe; with the paired
            # blocks switched off those captures do not exist.
            stage(
                "direction_check",
                lambda: direction.analyze(summary_run, out / "direction"),
            )
    if config.acceptance_enabled:
        logger.info(
            "acceptance loop finished: rounds={} accepted={} stopped_by_user={}",
            len(rounds),
            accepted,
            stopped_by_user,
        )
    print(f"Session complete: {out}", flush=True)
    return out


def _log_to_stdout():
    """Keep the CLI's own INFO on stdout.

    PowerShell turns loguru's default stderr lines into NativeCommandError
    records, and the plan/execute output *is* the command's result (the
    ``--help``-frozen CLI contract asserts it on stdout). ``logger`` stays the
    module-level one: importing it inside a function makes it a local and
    breaks every other call in ``main``.
    """
    import sys

    logger.remove()
    logger.add(sys.stdout, level="INFO", colorize=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "run"):
        p = commands.add_parser(name)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--dataset", type=Path)
        p.add_argument("--checkpoint", type=Path)
        p.add_argument(
            "--seed-run", type=Path, help="Start from accepted phases; never invoke GS"
        )
        p.add_argument(
            "--seed-policy", choices=("strict", "transfer"), default="strict"
        )
        p.add_argument(
            "--initial-phase", choices=("gs", "plane", "random"), default="gs"
        )
        p.add_argument("--exposure-max-steps", type=int, default=2)
        p.add_argument("--phase-lr", type=float, default=0.01)
        p.add_argument("--phase-rms-cap", type=float, default=0.2)
        p.add_argument("--adaptive-step", action="store_true")
        p.add_argument("--adaptive-exposure", action="store_true")
        p.add_argument(
            "--shapes", nargs="+", default=["square", "rectangle", "triangle"]
        )
        for key, default in (
            ("count", 24),
            ("iterations", 60),
            ("blocks", 4),
            ("updates", 10),
            ("frames", 3),
            ("seed", 102319),
            ("cam-id", 0),
            ("slm-number", 1),
        ):
            p.add_argument("--" + key, type=int, default=default)
        p.add_argument("--camera", default="daheng")
        p.add_argument(
            "--device",
            choices=DEVICE_CHOICES,
            default="auto",
            help="Compute device for fit/synthesis/CCD blocks: auto = CUDA when "
            "available else CPU; cpu reproduces the historical numbers",
        )
        p.add_argument("--side", type=float, default=80.0)
        p.add_argument("--period", type=float, default=64.0)
        p.add_argument(
            "--window-span",
            type=float,
            default=220.0,
            help="CCD window half-span (native px); a 1200-px target needs >= ~640",
        )
        p.add_argument(
            "--window-step",
            type=float,
            default=2.0,
            help="CCD sampling step (native px) used for the mask and metrics",
        )
        p.add_argument(
            "--zero-radius",
            type=float,
            default=0.0,
            help="Radius (native px) excluded from the core around the zero order",
        )
        p.add_argument(
            "--core-margin",
            type=float,
            default=None,
            help="Extra erosion (native px) of the core beyond the window step",
        )
        p.add_argument(
            "--capture-grid",
            type=int,
            default=128,
            help="Control grid of the probe capture (must tile the panel)",
        )
        p.add_argument(
            "--bit-depth",
            type=int,
            choices=(8, 12),
            default=8,
            help="ADC bit depth of the CCD blocks",
        )
        p.add_argument(
            "--control-grid",
            type=int,
            default=0,
            help="Export the fitted optics at this control grid before shaping "
            "(0 = keep the fit resolution; a 1200-px target uses 640)",
        )
        p.add_argument(
            "--fft-size",
            type=int,
            default=2048,
            help="FFT size of the exported control-resolution checkpoint",
        )
        p.add_argument(
            "--amplitude-center-lr",
            type=float,
            default=None,
            help="Learning rate of the illumination position (panel px/step); "
            "a disturbed bench needs more than the default to reach it",
        )
        p.add_argument(
            "--adaptation", choices=("auto", "alignment", "full"), default="auto"
        )
        p.add_argument("--accept-metric", choices=("loss", "cv"), default="loss")
        p.add_argument(
            "--objective", choices=("shape", "cv", "formation"), default="shape"
        )
        p.add_argument(
            "--no-abba",
            dest="abba",
            action="store_false",
            help="Skip the A–B–B–A acceptance; accept on the model loss only "
            "(summary acceptance=model_only, no CCD verdict)",
        )
        p.add_argument(
            "--abba",
            dest="abba",
            action="store_true",
            help="Enable optional A–B–B–A CCD acceptance",
        )
        p.set_defaults(abba=False)
        p.add_argument(
            "--save-dataset",
            action="store_true",
            help="Write one sample_*.npz per probe (needed by the offline --dataset "
            "path and scripts/slm analyses); off keeps them in memory",
        )
        p.add_argument("--skip-calibration", dest="calibration", action="store_false")
        p.add_argument("--calibration", dest="calibration", action="store_true")
        p.set_defaults(calibration=None)
        mode = p.add_mutually_exclusive_group()
        mode.add_argument("--offline", action="store_true")
        mode.add_argument("--execute", action="store_true")
    replay = commands.add_parser(
        "replay", help="Audit and recompute saved measurements, without devices"
    )
    replay.add_argument("--run", type=Path, required=True)
    replay.add_argument("--out", type=Path, required=True)
    audit = commands.add_parser(
        "audit", help="Check raw frames, exposures, phases, masks and A/B decisions"
    )
    audit.add_argument("--run", type=Path, required=True)
    audit.add_argument("--out", type=Path)
    record = commands.add_parser(
        "record", help="Rebuild a finished run's Recorder table and GIF; no devices"
    )
    record.add_argument(
        "--run",
        type=Path,
        required=True,
        help="the anchored/ or model/ directory of a finished run",
    )
    record.add_argument(
        "--out", type=Path, help="output directory (default <run>/recording)"
    )
    record.add_argument("--fps", type=float, default=2.0)
    doctor = commands.add_parser(
        "doctor", help="Enumerate Daheng without opening a stream or accessing SLM"
    )
    doctor.add_argument("--out", type=Path)
    simulate = commands.add_parser(
        "simulate", help="Create known-truth linear-camera transfer probes; no devices"
    )
    simulate.add_argument("--out", type=Path, required=True)
    simulate.add_argument("--power", type=float, default=1.2)
    simulate.add_argument("--shift", type=float, nargs=2, default=(12.0, -8.0))
    simulate.add_argument("--count", type=int, default=12)
    simulate.add_argument("--seed", type=int, default=102419)
    pretrain = commands.add_parser(
        "pretrain",
        help="Fit the low-dimensional estimator on a "
        "matched-model synthetic library; no devices",
    )
    pretrain.add_argument("--out", type=Path, required=True)
    pretrain.add_argument("--combos", type=int, default=60)
    pretrain.add_argument("--probes", type=int, default=8)
    pretrain.add_argument("--fft", type=int, default=128)
    pretrain.add_argument("--grid", type=int, default=32)
    pretrain.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="ridge penalty; default is pretrain.DEFAULT_ALPHA",
    )
    pretrain.add_argument("--seed", type=int, default=102419)
    args = parser.parse_args()

    # The CLI owns its logging: the plan and every stage summary are program output that
    # callers parse, so they belong on stdout, not on loguru's default stderr sink.  This
    # also keeps routine INFO records from becoming PowerShell NativeCommandError entries.
    import sys

    logger.remove()
    logger.add(sys.stdout, level="INFO", colorize=False)

    if args.command == "pretrain":
        from .pretrain import DEFAULT_ALPHA, pretrain as run_pretrain

        if args.probes < 2:
            parser.error("probes >= 2 required")
        result = run_pretrain(
            args.out,
            combos=args.combos,
            probes_per_combo=args.probes,
            fft_size=args.fft,
            grid_size=args.grid,
            alpha=DEFAULT_ALPHA if args.alpha is None else args.alpha,
            seed=args.seed,
        )
        logger.info(
            json.dumps(
                dict(
                    out=str(result["out"]),
                    combinations=result["combinations"],
                    timings=result["timings"],
                    metrics=result["metrics"],
                ),
                indent=2,
            )
        )
        return

    if args.command == "simulate":
        from .simulation import generate_dataset

        if args.count < 6:
            parser.error("count >= 6 required")
        checkpoint = generate_dataset(
            args.out,
            power=args.power,
            shift=args.shift,
            count=args.count,
            seed=args.seed,
        )
        logger.info(f"Synthetic probes complete; nominal checkpoint: {checkpoint}")
        return
    if args.command == "doctor":
        from .hardware import enumerate_camera

        result = enumerate_camera()
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with args.out.open("x", encoding="utf-8") as stream:
                json.dump(result, stream, default=str, indent=2)
        logger.info(json.dumps(result, default=str, indent=2))
        return
    if args.command == "record":
        from .recording import rebuild_run

        _log_to_stdout()
        logger.info(json.dumps(rebuild_run(args.run, args.out, fps=args.fps), indent=2))
        return
    if args.command in {"audit", "replay"}:
        from .artifacts import audit_run

        result = audit_run(args.run)
        if args.command == "replay":
            from . import direction, summary

            args.out.mkdir(parents=True, exist_ok=False)
            (args.out / "audit.json").write_text(
                json.dumps(result, indent=2), encoding="utf-8"
            )
            summary.summarize(args.run, args.out / "summary")
            direction.analyze(args.run, args.out / "direction")
        elif args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with args.out.open("x", encoding="utf-8") as stream:
                json.dump(result, stream, indent=2)
        logger.info(
            json.dumps(
                {k: v for k, v in result.items() if k != "file_sha256"}, indent=2
            )
        )
        return
    config = TransferSession(
        **{
            k: v
            for k, v in vars(args).items()
            if k not in {"command", "offline", "execute"}
        }
    )
    if args.command == "plan" or not (args.execute or args.offline):
        _log_to_stdout()
        logger.info(json.dumps(config.plan(args.offline), default=str, indent=2))
    else:
        _log_to_stdout()
        # 光路无光 (OpticalPathInvalid) 是实验系统上的预期状态: 只报实测数字并
        # 用专用退出码 3 (探针的 non-ok 是 2, 普通异常是 1), 不打 traceback。
        try:
            run_session(config, offline=args.offline)
        except OpticalPathInvalid as exc:
            logger.error(exc.report())
            raise SystemExit(exc.exit_code) from None
