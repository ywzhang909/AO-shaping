"""Replay a saved phase with circular smoothing and a zero-order-free CV.

The measured phase comes either from the latest accepted *transfer* shaping
present (``--phase-source transfer``, the default; see
:mod:`scripts.slm._transfer_phase`) or from a saved GS ``best_phase.npy``
(``--phase-source gs``, the original contract). Each candidate is bracketed by
the original phase at a fixed exposure. All frames and UTC timestamps are saved
under data/debug for paired analysis.

Transfer artifacts carry no SLM-panel pupil geometry (their ``config.json``
knows the camera-side zero order only), so the circular-smoothing pupil mask
falls back to the bench constants and ``config.json`` records that provenance
as ``geometry_source``. The camera-side target side, which transfer has no
equivalent of, is measured from the flat reference frame unless
``--target-side-px`` is given.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from ao_shaping.algorithm.signal_processing.square_shaping import uniformity_cv
from ao_shaping.optimizer.wfless._measure import display_and_average
from ao_shaping.tools.slm.panel.display_state import preserve_slm_display
from ao_shaping.utils.image.beam_metrics import (
    clamp_side,
    measure_bright_span,
    prepare_frame,
    zero_order_center,
    zero_order_exclusion_mask,
)
from ao_shaping.utils.slm.bench_constants import (
    BEAM_CENTER_PANEL,
    BEAM_RADIUS_PANEL,
    SLM_PANEL_H,
    SLM_PANEL_W,
)
from ao_shaping.utils.image.exposure_control import (
    configured_daheng_exposure_limit,
    probe_exposure_ceiling_ms,
)
from ao_shaping.utils.image.targets import square_mask
from ao_shaping.utils.slm.phase_embed import panel_shape_from_slm
from ao_shaping.utils.slm.phase_uniformity import (
    circular_smooth_phase,
    wrapped_gradient_rms,
)
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: 被测相位来源: 最新 transfer 整形终稿 / 已保存的 GS 精修相位。
PHASE_SOURCE_TRANSFER = "transfer"
PHASE_SOURCE_GS = "gs"

#: ``geometry_source`` 取值。transfer 产物没有面板几何，只能用实验系统常量；
#: GS 的几何来自它自己的 run config。
GEOMETRY_SOURCE_BENCH = "bench_constants:BEAM_CENTER_PANEL,BEAM_RADIUS_PANEL"
GEOMETRY_SOURCE_GS_CONFIG = "gs_refine_config"

#: ``target_side_source`` 取值：GS config / CLI / 平场参考帧实测。
TARGET_SIDE_SOURCE_GS_CONFIG = "gs_refine_config"
TARGET_SIDE_SOURCE_CLI = "--target-side-px"
TARGET_SIDE_SOURCE_MEASURED = "measured_flat_half_max_span"

#: ``summary.json`` 顶部来源块从 run config 里取的键（与 GS 来源可区分）。
SUMMARY_PROVENANCE_KEYS = (
    "phase_source",
    "source_phase",
    "source_phase_sha256",
    "geometry_source",
    "target_side_px",
    "target_side_source",
)


@dataclass(frozen=True, eq=False)
class SourcePhase:
    """被测相位及其几何来源（transfer 或旧 GS），不持有任何设备句柄。

    ``panel_phase`` 已经是 ``(1200, 1920)`` 面板弧度，交给
    :func:`~ao_shaping.optimizer.wfless._measure.display_and_average` 的
    ``slm.create_phase_from_array`` 完成 rad→gray，和 GS 相位走同一条链。
    """

    kind: str
    path: Path
    panel_phase: np.ndarray
    pupil_center_xy: tuple[int, int]
    pupil_radius_px: float
    geometry_source: str
    panel_sha256: str
    model_phase: np.ndarray | None = None
    config_path: Path | None = None
    configured_target_side_px: int | None = None

    def metadata(self) -> dict[str, object]:
        """写进 ``config.json`` 的来源块，标明真实路径与几何出处。"""
        return {
            "phase_source": self.kind,
            "source_phase": str(self.path),
            "source_phase_sha256": self.panel_sha256,
            "source_phase_shape": list(self.panel_phase.shape),
            "source_model_shape": (
                list(self.model_phase.shape) if self.model_phase is not None else None
            ),
            "source_config": str(self.config_path) if self.config_path is not None else None,
            "source_pupil_center_xy": list(self.pupil_center_xy),
            "source_pupil_radius_px": self.pupil_radius_px,
            "geometry_source": self.geometry_source,
        }


def phase_sha256(phase: np.ndarray) -> str:
    """面板相位的内容摘要，用于确认重放的确实是哪个数组。"""
    return hashlib.sha256(
        np.ascontiguousarray(phase, dtype=np.float64).tobytes()
    ).hexdigest()


def _require_panel_shape(panel: np.ndarray, path: Path) -> np.ndarray:
    """相位必须正好是物理面板尺寸，否则下发前就失败（不等设备报错）。"""
    panel = np.asarray(panel, dtype=np.float64)
    if panel.shape != (SLM_PANEL_H, SLM_PANEL_W):
        raise ValueError(
            f"{path} 相位形状 {panel.shape} != 面板 ({SLM_PANEL_H}, {SLM_PANEL_W})"
        )
    return panel


def _resolve_transfer_source(transfer_phase: Path | None) -> SourcePhase:
    """transfer 终稿 → 面板弧度；几何取实验系统常量（产物里没有面板几何）。"""
    # 懒加载：scripts/slm 既可以当包导入，也可以直接执行脚本。
    try:
        from scripts.slm._transfer_phase import load_transfer_phase
    except ImportError:  # 直接 `python scripts/slm/gs_uniformity_probe.py`
        from _transfer_phase import load_transfer_phase  # type: ignore[no-redef]
    path, model, panel = load_transfer_phase(transfer_phase)
    panel = _require_panel_shape(panel, path)
    return SourcePhase(
        kind=PHASE_SOURCE_TRANSFER,
        path=Path(path),
        panel_phase=panel,
        pupil_center_xy=(int(BEAM_CENTER_PANEL[0]), int(BEAM_CENTER_PANEL[1])),
        pupil_radius_px=float(BEAM_RADIUS_PANEL),
        geometry_source=GEOMETRY_SOURCE_BENCH,
        panel_sha256=phase_sha256(panel),
        model_phase=np.asarray(model, dtype=np.float64),
    )


def _resolve_gs_source(phase_file: Path | None) -> SourcePhase:
    """旧 GS 契约不变：``best_phase.npy`` + 同目录 ``slm_gs_refine_*.json``。"""
    if phase_file is None:
        raise ValueError("--phase-source gs 需要 GS 相位文件（best_phase.npy）")
    path = Path(phase_file)
    if not path.is_file() or path.name != "best_phase.npy":
        raise ValueError("GS 相位文件必须是已保存的 best_phase.npy")
    config_path = next(iter(sorted(path.parent.glob("slm_gs_refine_*.json"))), None)
    if config_path is None:
        raise FileNotFoundError(f"缺少 GS 配置 {path.parent}/slm_gs_refine_*.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    panel = _require_panel_shape(np.load(path, allow_pickle=False), path)
    center_xy = tuple(int(v) for v in config["pupil_center"])
    saved_side = int(config.get("target_side") or 0)
    return SourcePhase(
        kind=PHASE_SOURCE_GS,
        path=path,
        panel_phase=panel,
        pupil_center_xy=(center_xy[0], center_xy[1]),
        pupil_radius_px=float(config["beam_radius_px"]),
        geometry_source=f"{GEOMETRY_SOURCE_GS_CONFIG}:{config_path.name}",
        panel_sha256=phase_sha256(panel),
        config_path=config_path,
        configured_target_side_px=saved_side if saved_side > 0 else None,
    )


def resolve_phase_source(
    phase_file: Path | None = None,
    *,
    phase_source: str = PHASE_SOURCE_TRANSFER,
    transfer_phase: Path | None = None,
) -> SourcePhase:
    """解析被测相位来源，不打开任何设备。

    Args:
        phase_file: 旧 GS 路径的 ``best_phase.npy``；transfer 来源下忽略。
        phase_source: ``transfer``（默认，最新终稿）或 ``gs``。
        transfer_phase: transfer 相位 ``.npy``/``.npz``；``None`` 用最新终稿。

    Returns:
        面板弧度与几何来源。

    Raises:
        ValueError: 未知来源、GS 契约不满足，或相位形状不是物理面板。
        FileNotFoundError: 相位文件或 GS 配置缺失。
    """
    if phase_source == PHASE_SOURCE_TRANSFER:
        return _resolve_transfer_source(transfer_phase)
    if phase_source == PHASE_SOURCE_GS:
        return _resolve_gs_source(phase_file)
    raise ValueError(
        f"未知 phase_source: {phase_source!r}（可选 {PHASE_SOURCE_TRANSFER} / {PHASE_SOURCE_GS}）"
    )


def resolve_target_side(
    flat: np.ndarray,
    *,
    requested: int | None,
    requested_source: str = TARGET_SIDE_SOURCE_CLI,
) -> tuple[int, str]:
    """定相机侧目标框边长，返回 ``(边长, 来源)``。

    显式边长（GS config 或 CLI）原样使用；否则用平场参考帧亮区包围盒的
    长边实测——transfer 产物没有可用的目标边长，实测比编造常量诚实。
    """
    if requested is not None:
        side = int(requested)
        if side <= 0:
            raise ValueError("target_side_px 必须为正")
        return clamp_side(side, flat.shape[0], flat.shape[1]), requested_source
    width, height = measure_bright_span(flat, peak_frac=0.5)
    if width <= 0 or height <= 0:
        raise RuntimeError(
            "平场参考帧没有亮区，无法实测目标框边长；请显式给 --target-side-px"
        )
    return (
        clamp_side(max(width, height), flat.shape[0], flat.shape[1]),
        TARGET_SIDE_SOURCE_MEASURED,
    )


def resolve_cli_paths(
    paths: list[Path], *, phase_source: str
) -> tuple[Path | None, Path]:
    """位置参数 → ``(GS 相位, 输出目录)``。

    GS 保留旧用法 ``best_phase.npy out_dir``；transfer 只给输出目录。
    """
    if phase_source == PHASE_SOURCE_GS:
        if len(paths) != 2:
            raise ValueError(
                "--phase-source gs 需要 <best_phase.npy> <output_dir> 两个位置参数"
            )
        return paths[0], paths[1]
    if len(paths) != 1:
        raise ValueError("--phase-source transfer 只接受 <output_dir> 一个位置参数")
    return None, paths[0]


def summary_provenance(config: dict | None) -> dict[str, object]:
    """``summary.json`` 的来源块；没有 run config 时明确标 ``unknown``。"""
    if not config:
        return {"phase_source": "unknown", "source_phase": None}
    return {key: config.get(key) for key in SUMMARY_PROVENANCE_KEYS}


def build_summary(rows: list[dict], provenance: dict[str, object]) -> dict:
    """配对统计 + 来源块，一起写进 ``summary.json``。"""
    return {**provenance, **summarize_pairs(rows)}


def pupil_mask(shape: tuple[int, int], center_xy: tuple[int, int], radius: float) -> np.ndarray:
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return (xx - center_xy[0]) ** 2 + (yy - center_xy[1]) ** 2 <= radius**2


def frame_metrics(
    frame: np.ndarray, target: np.ndarray, zero_core: np.ndarray
) -> dict[str, float]:
    valid = target & ~zero_core
    clean = prepare_frame(frame)
    return {
        "cv_raw": uniformity_cv(frame, target),
        "cv_no_zero": uniformity_cv(frame, valid),
        "cv_no_zero_background_subtracted": uniformity_cv(clean, valid),
        "target_mean": float(np.mean(frame[target])),
        "target_mean_no_zero": float(np.mean(frame[valid])),
        "frame_peak": float(np.max(frame)),
        "frame_median": float(np.median(frame)),
        "zero_core_pixels": int(np.count_nonzero(zero_core)),
        "target_pixels": int(np.count_nonzero(target)),
    }


def summarize_pairs(rows: list[dict]) -> dict:
    """Compare each smoothed frame against its bracketing raw-phase frames."""
    if len(rows) % 3:
        raise ValueError("expected baseline/candidate/baseline triples")
    pairs = []
    for before, candidate, after in zip(rows[::3], rows[1::3], rows[2::3]):
        if (before["label"], candidate["label"], after["label"]) != (
            "baseline_before", "smoothed", "baseline_after"
        ):
            raise ValueError("invalid paired measurement order")
        reference = 0.5 * (before["cv_no_zero"] + after["cv_no_zero"])
        pairs.append({
            "sigma_px": candidate["sigma_px"],
            "cv_no_zero_candidate": candidate["cv_no_zero"],
            "cv_no_zero_reference": reference,
            "delta_cv_no_zero": candidate["cv_no_zero"] - reference,
            "phase_gradient_rms_rad_per_pixel": candidate["phase_gradient_rms_rad_per_pixel"],
        })
    by_sigma = {}
    for sigma in sorted({pair["sigma_px"] for pair in pairs}):
        values = [pair["delta_cv_no_zero"] for pair in pairs if pair["sigma_px"] == sigma]
        by_sigma[str(sigma)] = {
            "n_pairs": len(values),
            "mean_delta_cv_no_zero": float(np.mean(values)),
            "pair_deltas": values,
        }
    return {"pairs": pairs, "by_sigma": by_sigma}


def collect(
    phase_file: Path | None = None,
    output_dir: Path | None = None,
    *,
    phase_source: str = PHASE_SOURCE_TRANSFER,
    transfer_phase: Path | None = None,
    target_side_px: int | None = None,
    sigmas: tuple[float, ...] = (2.0, 5.0, 10.0, 20.0),
    repeats: int = 2,
    exposure_ms: float = 0.12,
    frames: int = 4,
    zero_radius_px: float | None = None,
) -> list[dict]:
    """重放 ``phase_source`` 相位并采集配对帧。

    Args:
        phase_file: 旧 GS 路径的 ``best_phase.npy``（仅 ``phase_source="gs"``）。
        output_dir: 本次采集目录（必须不存在）。
        phase_source: ``transfer``（默认，最新终稿）或 ``gs``。
        transfer_phase: transfer 相位 ``.npy``/``.npz``；``None`` 用最新终稿。
        target_side_px: 相机侧目标框边长；GS 下由 config 接管，transfer 下
            ``None`` 表示按平场参考帧实测。
    """
    from ao_shaping.drivers.ccd.common import create_camera
    from ao_shaping.drivers.slm.santec import Santec, SantecError

    if output_dir is None:
        raise ValueError("output_dir 必填")
    source = resolve_phase_source(
        phase_file, phase_source=phase_source, transfer_phase=transfer_phase
    )
    if not sigmas or any(not np.isfinite(s) or s <= 0 for s in sigmas):
        raise ValueError("sigmas must be finite and positive")
    if repeats < 1 or frames < 1 or exposure_ms < 0.02:
        raise ValueError("repeats and frames must be positive; exposure >= 0.02 ms")
    raw = source.panel_phase
    pupil = pupil_mask(
        raw.shape, source.pupil_center_xy, source.pupil_radius_px
    )
    candidates = {sigma: circular_smooth_phase(raw, sigma, pupil) for sigma in sigmas}
    gradients = {0.0: wrapped_gradient_rms(raw, pupil)}
    gradients.update({sigma: wrapped_gradient_rms(phase, pupil) for sigma, phase in candidates.items()})
    output_dir.mkdir(parents=True, exist_ok=False)
    for sigma, phase in candidates.items():
        np.save(output_dir / f"phase_sigma_{sigma:g}.npy", phase)
    rows: list[dict] = []
    run_config: dict | None = None
    with Santec(slm_number=1, wavelength=1064, video_mode=0) as slm:
        with create_camera("daheng", 0, exposure_time_ms=0.02) as cam:
            with preserve_slm_display(slm):
                if tuple(panel_shape_from_slm(slm)) != raw.shape:
                    raise ValueError("saved phase does not match the SLM panel")
                try:
                    protected_slot = slm.get_displayed_memory_number()
                except SantecError as exc:
                    if exc.code != 1:
                        raise
                    protected_slot = None
                slots = [s for s in (126, 127, 124) if s != protected_slot][:2]
                slot_index = 0

                def capture(phase: np.ndarray) -> np.ndarray:
                    nonlocal slot_index
                    image = display_and_average(
                        cam, slm, phase, n_frames=frames, settle_wait_s=0.5,
                        settle_tol=0.02, settle_max_wait_s=6.0,
                        settle_max_discard=40,
                        memory_number=slots[slot_index % len(slots)],
                    )
                    slot_index += 1
                    if np.max(image) >= 250:
                        raise RuntimeError("CCD frame near saturation; stop and lower exposure")
                    return image

                flat_phase = np.zeros(raw.shape, dtype=np.float64)
                cam.reset_exposure_time(0.02)
                capture(flat_phase)
                ceiling = configured_daheng_exposure_limit()
                if not ceiling:
                    ceiling = probe_exposure_ceiling_ms(
                        cam.reset_exposure_time,
                        lambda: np.asarray(cam.get_numpy_image(n_sample=1)),
                    ).ceiling_ms
                working_ms = min(exposure_ms, ceiling)
                if working_ms < 0.02:
                    raise RuntimeError("resolved exposure ceiling is below camera minimum")
                actual = float(cam.reset_exposure_time(working_ms))
                if not 0 < actual <= ceiling + 1e-9:
                    raise RuntimeError(f"unsafe exposure readback: {actual} ms")
                full_flat = capture(flat_phase)
                cx_full, cy_full = zero_order_center(full_flat, refine=False, despike_k=3)
                half = 125
                if (cx_full - half < 0 or cy_full - half < 0
                        or cx_full + half >= full_flat.shape[1]
                        or cy_full + half >= full_flat.shape[0]):
                    raise ValueError("zero order too close to CCD edge for 250 px ROI")
                cam.reset_window(center=(cx_full, cy_full), size=(250, 250))
                flat = capture(flat_phase)
                cx, cy = zero_order_center(flat, refine=False, despike_k=3)
                requested_side = (
                    source.configured_target_side_px
                    if source.configured_target_side_px is not None
                    else target_side_px
                )
                target_side, target_side_source = resolve_target_side(
                    flat,
                    requested=requested_side,
                    requested_source=(
                        TARGET_SIDE_SOURCE_GS_CONFIG
                        if source.kind == PHASE_SOURCE_GS
                        else TARGET_SIDE_SOURCE_CLI
                    ),
                )
                target = square_mask(flat.shape, cy, cx, target_side)
                zero_core = zero_order_exclusion_mask(
                    flat, target, radius_px=zero_radius_px
                )
                np.savez_compressed(output_dir / "flat_reference.npz",
                                    frame=flat, target=target, zero_core=zero_core,
                                    phase_source=source.kind,
                                    source_phase=str(source.path),
                                    source_phase_sha256=source.panel_sha256,
                                    geometry_source=source.geometry_source,
                                    target_side_px=target_side)
                run_config = {
                    **source.metadata(),
                    "target_side_px": target_side,
                    "target_side_source": target_side_source,
                    "ccd_full_zero_center_xy": [cx_full, cy_full],
                    "ccd_roi_zero_center_xy": [cx, cy],
                    "exposure_ms_requested": exposure_ms,
                    "exposure_ms_working": actual,
                    "exposure_ceiling_ms": ceiling,
                    "frames_per_measurement": frames,
                    "repeats": repeats,
                    "sigmas_px": sigmas,
                    "zero_core_pixels": int(zero_core.sum()),
                    "phase_gradient_rms_rad_per_pixel": gradients,
                }
                (output_dir / "config.json").write_text(
                    json.dumps(run_config, indent=2), encoding="utf-8"
                )

                def record(label: str, sigma: float, pair: int) -> None:
                    phase = raw if sigma == 0 else candidates[sigma]
                    frame = capture(phase)
                    readback = float(cam.get_exposure_time_ms())
                    if abs(readback - actual) > 1e-9 or readback > ceiling + 1e-9:
                        raise RuntimeError(f"CCD exposure changed during comparison: {readback} ms")
                    index = len(rows)
                    row = {
                        "index": index,
                        "pair": pair,
                        "label": label,
                        "sigma_px": sigma,
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
                        "exposure_ms_readback": readback,
                        "phase_gradient_rms_rad_per_pixel": gradients[sigma],
                        **frame_metrics(frame, target, zero_core),
                    }
                    np.savez_compressed(output_dir / f"frame_{index:03d}.npz", frame=frame)
                    rows.append(row)
                    (output_dir / "measurements.json").write_text(
                        json.dumps(rows, indent=2), encoding="utf-8"
                    )
                    logger.info(json.dumps(row))

                for repeat in range(repeats):
                    order = sigmas if repeat % 2 == 0 else tuple(reversed(sigmas))
                    for sigma in order:
                        pair = len(rows) // 3
                        record("baseline_before", 0.0, pair)
                        record("smoothed", sigma, pair)
                        record("baseline_after", 0.0, pair)
    (output_dir / "summary.json").write_text(
        json.dumps(build_summary(rows, summary_provenance(run_config)), indent=2),
        encoding="utf-8",
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        type=Path,
        nargs="*",
        metavar="PATH",
        help=(
            "--phase-source gs: <best_phase.npy> <output_dir>；"
            "--phase-source transfer: <output_dir>"
        ),
    )
    parser.add_argument(
        "--phase-source",
        choices=(PHASE_SOURCE_TRANSFER, PHASE_SOURCE_GS),
        default=PHASE_SOURCE_TRANSFER,
        help="被测相位来源（默认最新 transfer 整形终稿）",
    )
    parser.add_argument(
        "--transfer-phase",
        type=Path,
        default=None,
        help="transfer 相位 .npy/.npz；默认用 _transfer_phase 的最新终稿",
    )
    parser.add_argument(
        "--target-side-px",
        type=int,
        default=None,
        help="相机侧目标框边长；transfer 来源下缺省按平场亮区实测",
    )
    parser.add_argument("--sigma", nargs="+", type=float, default=[2.0, 5.0, 10.0, 20.0])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--exposure-ms", type=float, default=0.12)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--zero-radius-px", type=float)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    phase_file, output_dir = resolve_cli_paths(
        list(args.paths), phase_source=args.phase_source
    )
    if args.summarize_only:
        rows = json.loads((output_dir / "measurements.json").read_text(encoding="utf-8"))
        config_path = output_dir / "config.json"
        config = (
            json.loads(config_path.read_text(encoding="utf-8"))
            if config_path.is_file()
            else None
        )
        (output_dir / "summary.json").write_text(
            json.dumps(build_summary(rows, summary_provenance(config)), indent=2),
            encoding="utf-8",
        )
        return
    collect(
        phase_file,
        output_dir,
        phase_source=args.phase_source,
        transfer_phase=args.transfer_phase,
        target_side_px=args.target_side_px,
        sigmas=tuple(args.sigma),
        repeats=args.repeats,
        exposure_ms=args.exposure_ms,
        frames=args.frames,
        zero_radius_px=args.zero_radius_px,
    )


if __name__ == "__main__":
    main()
