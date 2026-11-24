"""采集纯 GS 1000 px 对照实验的 C 变体 (先矫正光路像差, 再发 GS)。

本脚本只做 ``report/slm/gs_control_1000px/report.md`` §8.2 第 2 项定义的待办测量:
**不重算 GS、不换优化器**。``B_panel`` 直接取已存的 1000 px GS 面板相位, 唯一新增
的量是共用光路像差 (由随机探针帧经仓库既有的
:func:`~ao_shaping.optimizer.wfless.model_in_loop_shaping.calibrate_shared_aberration`
拟合), 然后::

    C_panel = B_panel - pupil_phase_to_panel(像差瞳面相位)

像差与 C 都在 B 的坐标系里: 面板 1920x1200, 光束中心 (960, 600), 半径 450,
相机像元 2.2 µm, 标度 ``TILT_SHIFT_SCALE = 7400`` (一个 GS 网格像素 = 8.2222
相机像素), 面板轴 0 -> 相机 x 且无符号翻转。0 级位置 (默认 1257, 915) 只用于探针帧
的裁片窗口 (``far_field_size = 512``), 不参与像差拟合的几何反演; ``--execute`` 时
优先用 ``open_bench`` 预检实测到的 0 级, 实测不可用时才退回 CLI 值。

验收口径 (报告 §8.2):

* 8 张强随机探针帧 (``probe_spread = 4.0``) 进拟合, 2 张验证集帧**只做检验,
  不进拟合**;
* 拟合冻结 Noll 1/2/3, ``region=64`` / ``far_field_size=512`` / ``n_orders=10`` /
  ``epochs=120`` / ``lr=0.05`` / ``pupil_oversampling=14``;
* 验证集预测相关低于门槛时不采 C, 只写降级 meta;
* 探针曝光由 :func:`~ao_shaping.tools.slm.panel.exposure_adapt.adapt_exposure`
  **单独**确定 (B 的 200 ms 会让探针帧饱和), 逐帧曝光与读回写进 NPZ 的 ``params``;
* 重放 B / C / 平相位 / 平相位+矫正四帧同曝光同稳定判据 (4 帧平均, 丢 3 帧,
  等 0.5 s, 容差 0.02, 最长 6 s), 全部走 ``BenchSession.capture``;
* 产物 ``c.npz`` / ``replay_b.npz`` / ``flat.npz`` / ``cflat.npz`` 用 ``frame`` 键
  (float32; 生成器也会回退 ``frames[0]``), ``correction_meta.json`` 给出
  ``probe_count`` / ``heldout_corr`` / ``rms_before`` / ``rms_after`` 与曝光溯源。

``rms_before`` / ``rms_after`` 是**像差真值与拟合残差的 RMS (rad)**, 只有探针带真值
(仿真/离线注入) 时才有定义; 硬件探针没有真值, 两个字段写 ``null``, 另给
``fitted_aberration_rms_rad`` (拟合出的像差本身的 RMS), 不拿它冒充足值残差。

设备纪律: 默认 ``--dry-run``, 不 import / 不打开任何驱动包。只有 ``--execute`` 才在
拿到 ``DeviceLock`` (超时 1800 s) 之后访问设备, 并在 ``finally`` 里释放锁。缺少任何
输入或口径不一致时**退出 0**, 写 ``psf_peak_comparable=false`` /
``geometry_ok=null``, 不伪造任何帧或指标。

离线重建示例 (没有设备, 只有已存 B 相位与上次拟合写下的系数)::

    python scripts/slm/gs_control_corrected_arm.py --dry-run \
        --aberration-coeffs-from <out>/correction_meta.json \
        --out-dir <out>/offline_rebuild

输入缺失时上面的命令同样退出 0, 只在 meta 里报告缺哪个文件。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ao_shaping.optimizer.wfless.model_in_loop_shaping import FitAberrationResult

ROOT = Path(__file__).resolve().parents[2]
# `scripts.*` 以包路径被导入, 所以仓库 ROOT 本身必须在 sys.path 上; 直接执行
# `python scripts/slm/x.py` 时 cwd 不在 sys.path, 只有脚本自己的目录在。
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from loguru import logger

from ao_shaping.algorithm.signal_processing.zernike_coefficient_optimizer import (
    ZernikeCoefficientOptimizer,
)
from ao_shaping.tools.slm.panel.exposure_adapt import adapt_exposure
from ao_shaping.utils.image.beam_metrics import prepare_frame
from ao_shaping.utils.image.frame_analysis import crop_roi
from ao_shaping.utils.slm.bench_constants import TILT_SHIFT_SCALE
from scripts.slm.hw_gs_defocus_probe import assert_no_drivers
from scripts.slm.hw_gs_square_scans import (
    Bench,
    build_bench,
    open_bench,
    parse_xy,
    pupil_phase_to_panel,
)

#: B 的已存面板相位; 本脚本绝不重新跑 GS 去生成它。
DEFAULT_B_PANEL = Path(
    "data/debug/hw_gs_square_scans/20261030_gs1000/sides/side_1000/phase_000.npy"
)
#: 报告 §8.2 冻结的 0 级位置 (相机坐标 x,y), 只在预检没给出实测值时使用。
DEFAULT_ZERO_ORDER = "1257,915"
DEFAULT_OUT_PARENT = Path("data/debug/gs_control_corrected_arm")
#: 8 bit 相机的满量程; 探针/平台帧都用同一个读数口径。
DEFAULT_FULL_SCALE = 255.0
DEVICE_LOCK_TIMEOUT_S = 1800.0


@dataclass(frozen=True)
class ProbeRecord:
    """一张探针帧与产生它的瞳面网格相位 (各自在模型/相机的原生坐标里)。"""

    label: str
    phase: np.ndarray
    image: np.ndarray
    exposure_ms: float
    readback_ms: float
    coefficients: np.ndarray | None = None


def build_parser() -> argparse.ArgumentParser:
    """与 :mod:`hw_gs_square_scans` 同一套 CLI 约定 (argparse, 默认 dry-run)。"""
    p = argparse.ArgumentParser(
        description="C 变体: B 面板相位减去拟合光路像差; 默认 dry-run 不碰设备"
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true",
                      help="只做离线检查 / 从已有系数重建 C (默认); 不 import 驱动包")
    mode.add_argument("--execute", action="store_true",
                      help="真正拿设备锁并采集; 这是唯一会 import 驱动包的分支")

    p.add_argument("--b-panel", type=Path, default=DEFAULT_B_PANEL,
                   help=f"B 已存面板相位 (默认 {DEFAULT_B_PANEL})")
    p.add_argument("--out-dir", type=Path, default=None,
                   help=f"产物目录 (默认 {DEFAULT_OUT_PARENT}/<UTC 时间戳>)")
    p.add_argument("--aberration-coeffs-from", type=Path, default=None,
                   help="已有 correction_meta.json (含 coefficients 键); "
                        "只允许 --dry-run 离线重建 C, --execute 必须现场从探针拟合")

    p.add_argument("--zero-order", default=DEFAULT_ZERO_ORDER,
                   help=f"0 级位置 X,Y (默认 {DEFAULT_ZERO_ORDER}; 预检实测值优先)")
    p.add_argument("--side-px", type=float, default=1000.0,
                   help="方框边长 (相机 px), 只记录进 meta")
    p.add_argument("--beam-radius-px", type=float, default=450.0,
                   help="照明半径 (面板 px)")
    p.add_argument("--wavelength-nm", type=float, default=0.0,
                   help="波长 (nm); 0 = 静态读取驱动常量, 不 import 驱动包")
    p.add_argument("--camera-pixel-um", type=float, default=2.2,
                   help="CCD 像元 (µm); dry-run 必须显式给, 不读设备")

    p.add_argument("--cam-type", default="daheng", choices=("daheng", "miicam", "sim"),
                   help="相机后端 (sim = 数字孪生)")
    p.add_argument("--cam-id", type=int, default=0, help="相机编号")
    p.add_argument("--cam-size", type=int, default=2500, help="仿真窗口边长")
    p.add_argument("--slm-number", type=int, default=1, help="SLM 编号")
    p.add_argument("--sim-seed", type=int, default=0, help="仿真噪声种子")
    p.add_argument("--sim-far-field-padding", type=int, default=4,
                   help="仿真远场补零因子")

    # 与 B 相同的取帧口径 (BenchSession.capture 读的就是这几项)。
    p.add_argument("--exposure-ms", type=float, default=200.0,
                   help="重放 B / C / 平相位 / 平相位+矫正的曝光 (ms)")
    p.add_argument("--frames", type=int, default=4,
                   help="每测量平均的帧数 (默认 4)")
    p.add_argument("--settle-n-discard", type=int, default=3,
                   help="稳定判据前的丢弃帧数 (默认 3)")
    p.add_argument("--settle-wait-s", type=float, default=0.5,
                   help="稳定判据等待 (s, 默认 0.5)")
    p.add_argument("--settle-tol", type=float, default=0.02,
                   help="稳定容差 (默认 0.02)")
    p.add_argument("--settle-max-wait-s", type=float, default=6.0,
                   help="稳定最长等待 (s, 默认 6)")

    # 探针曝光单独定; 探针帧不能沿用 200 ms 的平台曝光。
    p.add_argument("--probe-exposure-start-ms", type=float, default=0.2,
                   help="探针曝光阶梯起点 (ms)")
    p.add_argument("--probe-exposure-floor-ms", type=float, default=0.02,
                   help="探针曝光下限 (ms)")
    p.add_argument("--full-scale", type=float, default=DEFAULT_FULL_SCALE,
                   help="帧满量程计数 (默认 255, 8 bit 相机)")

    p.add_argument("--probe-count", type=int, default=8, help="进拟合的强随机探针帧数")
    p.add_argument("--validation-count", type=int, default=2, help="验证集探针帧数")
    p.add_argument("--probe-spread", type=float, default=4.0,
                   help="探针瞳面高斯相位标准差 (rad)")
    p.add_argument("--probe-seed", type=int, default=0, help="探针 RNG 种子")
    p.add_argument("--heldout-corr-min", type=float, default=0.99,
                   help="验证集预测相关门槛; 低于它不采 C")

    p.add_argument("--region", type=int, default=64, help="像差拟合瞳面网格边长")
    p.add_argument("--far-field-size", type=int, default=512,
                   help="像差拟合远场网格边长 / 探针裁片边长")
    p.add_argument("--n-orders", type=int, default=10, help="Zernike 最高径向阶")
    p.add_argument("--epochs", type=int, default=120, help="每条记录的 Adam 步数")
    p.add_argument("--lr", type=float, default=0.05, help="Adam 学习率")
    p.add_argument("--pupil-oversampling", type=int, default=14,
                   help="拟合模型的瞳面双线性放大因子 (物理链口径, 不是调参)")
    return p


def _num_or_none(value: Any) -> float | None:
    """非有限数写成 ``None`` 而不是让 NaN 进 JSON。"""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _json_params(payload: dict[str, Any]) -> str:
    """NPZ 的 ``params`` 是 JSON 字符串 (生成器的 ``_decode_params`` 口径)。"""
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def save_frame(path: Path, frame: np.ndarray, params: dict[str, Any]) -> None:
    """落一帧 CCD 图像: ``frame`` float32 + ``params`` JSON 字符串。"""
    np.savez_compressed(
        path,
        frame=np.asarray(frame, dtype=np.float32),
        params=_json_params(params),
    )


def write_meta(path: Path, meta: dict[str, Any]) -> None:
    """落 correction_meta.json; 顶层的非有限浮点一律转成 null。"""
    cleaned = {
        key: _num_or_none(value) if isinstance(value, (float, np.floating)) else value
        for key, value in meta.items()
    }
    path.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2), encoding="utf-8")


def model_psf_fwhm_px(bench: Bench) -> float:
    """理想圆孔衍射的 FWHM (相机 px), 与报告生成器同一公式。"""
    return 0.843 * 1.22 / (2.0 * float(bench.beam_radius_px)) * float(TILT_SHIFT_SCALE)


def degraded_meta(
    reason: str,
    args: argparse.Namespace,
    bench: Bench,
    *,
    probe_count: int = 0,
    validation_count: int = 0,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """缺输入 / 口径不一致时的统一降级 meta: 不伪造帧, 不伪造可比性。"""
    meta: dict[str, Any] = {
        "psf_peak_comparable": False,
        "geometry_ok": None,
        "probe_count": int(probe_count),
        "validation_probe_count": int(validation_count),
        "heldout_corr": None,
        "rms_before": None,
        "rms_after": None,
        "coefficients": [],
        "model_psf_fwhm_px": model_psf_fwhm_px(bench),
        "reason": reason,
        "geometry": dict(
            zero_order_xy=list(args.zero_order_xy),
            box_side_px=float(args.side_px),
            camera_px_per_gs_px=bench.camera_px_per_gs_px,
            tilt_shift_scale=float(TILT_SHIFT_SCALE),
            panel_shape=list(bench.panel_shape),
            beam_center_panel=list(bench.beam_center_panel),
            beam_radius_px=bench.beam_radius_px,
        ),
        "capture": dict(
            frames=int(args.frames),
            settle_n_discard=int(args.settle_n_discard),
            settle_wait_s=float(args.settle_wait_s),
            settle_tol=float(args.settle_tol),
            settle_max_wait_s=float(args.settle_max_wait_s),
        ),
        "exposure_provenance": exposure_provenance(args, None, [], None),
    }
    if extra:
        meta.update(extra)
    return meta


def exposure_provenance(
    args: argparse.Namespace,
    probe_exposure_ms: float | None,
    probe_readbacks: list[float],
    plateau_readbacks: dict[str, float] | None,
) -> dict[str, Any]:
    """曝光来源与逐帧读回, 让生成器能判断 C / 重放 B 是否同曝光。"""
    return dict(
        full_scale_counts=float(args.full_scale),
        probe_exposure_ms=_num_or_none(probe_exposure_ms),
        probe_exposure_start_ms=float(args.probe_exposure_start_ms),
        probe_exposure_floor_ms=float(args.probe_exposure_floor_ms),
        probe_exposure_readback_ms=[_num_or_none(v) for v in probe_readbacks],
        plateau_exposure_ms=float(args.exposure_ms),
        plateau_exposure_readback_ms=plateau_readbacks,
    )


def make_optimizer(args: argparse.Namespace) -> ZernikeCoefficientOptimizer:
    """构造与共用拟合**同一个**物理预测模型 (基/尺度/oversampling 都不许漂移)。

    只用于验证集预测与系数 -> 瞳面相位的基展开, 不在上面跑任何优化; 固定 CPU
    让离线重建与硬件运行给出逐位相同的相位。
    """
    return ZernikeCoefficientOptimizer(
        n_orders=int(args.n_orders),
        region=int(args.region),
        lr=float(args.lr),
        max_iterations=max(1, int(args.epochs)),
        dtype="float64",
        device="cpu",
        seed=None,
        frozen_modes=(1, 2, 3),
        far_field_size=int(args.far_field_size),
        pupil_oversampling=int(args.pupil_oversampling),
    )


def aberration_phase_from_coefficients(
    coefficients: np.ndarray, optimizer: ZernikeCoefficientOptimizer
) -> np.ndarray:
    """Noll 系数 -> 拟合网格上的像差瞳面相位 (未包裹弧度)。

    直接用拟合器自己的 Zernike 基, 不另写一套多项式; 否则系数与相位会差一个
    归一化或 Noll 排序, C 就不再是 B 的共轭修正。
    """
    values = np.asarray(coefficients, dtype=np.float64).ravel()
    if values.size != optimizer.n_coefficients:
        raise ValueError(
            f"coefficients 长度 {values.size} != 期望 {optimizer.n_coefficients}"
        )
    basis = optimizer.generate_basis()
    return np.tensordot(values, basis, axes=(0, 0))


def build_c_panel(b_panel: np.ndarray, phase: np.ndarray, bench: Bench) -> np.ndarray:
    """C = B 面板相位 - 像差瞳面相位映射到面板 (B 的坐标系)。"""
    aberr_panel = pupil_phase_to_panel(phase, bench)
    if aberr_panel.shape != b_panel.shape:
        raise ValueError(f"像差面板 {aberr_panel.shape} 与 B 面板 {b_panel.shape} 不一致")
    return np.asarray(b_panel, dtype=np.float64) - aberr_panel


def load_b_panel(path: Path, bench: Bench) -> np.ndarray | None:
    """读已存 B 面板相位; 缺失或形状不对都返回 None (不伪造)。"""
    if not path.is_file():
        logger.warning("缺少 B 面板相位: {}", path)
        return None
    try:
        panel = np.asarray(np.load(path), dtype=np.float64)
    except (OSError, ValueError) as exc:
        logger.warning("B 面板相位读取失败 {}: {}", path, exc)
        return None
    if panel.shape != bench.panel_shape:
        logger.warning("B 面板形状 {} 与面板几何 {} 不一致", panel.shape, bench.panel_shape)
        return None
    return panel


def load_coefficients(path: Path, expected: int) -> np.ndarray | None:
    """从已有 correction_meta.json 读系数, 供离线重建 C 使用。"""
    if not path.is_file():
        logger.warning("缺少系数来源: {}", path)
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("系数来源读取失败 {}: {}", path, exc)
        return None
    raw = payload.get("coefficients") if isinstance(payload, dict) else payload
    if raw is None:
        logger.warning("系数来源 {} 没有 coefficients 键", path)
        return None
    values = np.asarray(raw, dtype=np.float64).ravel()
    if values.size != expected or not np.all(np.isfinite(values)):
        logger.warning("系数长度/取值不合法: {} (期望 {})", values.size, expected)
        return None
    return values


def fit_shared_aberration(
    records: list[ProbeRecord], args: argparse.Namespace
) -> FitAberrationResult:
    """唯一的像差拟合入口: 直接调仓库 canonical 实现, 不另写优化循环。

    只收**训练**探针; 验证集帧由 :func:`heldout_correlation` 单独评估。
    """
    from ao_shaping.optimizer.wfless.model_in_loop_shaping import (
        CalibrationRecord,
        calibrate_shared_aberration,
    )

    calibration = [
        CalibrationRecord(
            image=record.image,
            phase=record.phase,
            label=record.label,
            coefficients=record.coefficients,
        )
        for record in records
    ]
    return calibrate_shared_aberration(
        calibration,
        region=int(args.region),
        far_field_size=int(args.far_field_size),
        n_orders=int(args.n_orders),
        epochs=int(args.epochs),
        lr=float(args.lr),
        frozen_modes=(1, 2, 3),
        pupil_oversampling=int(args.pupil_oversampling),
    )


def heldout_correlation(
    coefficients: np.ndarray,
    validation: list[ProbeRecord],
    optimizer: ZernikeCoefficientOptimizer,
) -> float | None:
    """验证集帧与拟合模型的 Pearson 相关 (只评估, 不更新任何系数)。

    模型瞳面轴 0 是面板的行方向, 而面板轴 0 映射到相机 x, 所以预测远场在和
    相机裁片比较前转置一次; 两个轴都不翻转 (报告 §8.2 的冻结口径)。Pearson
    相关对整体增益/偏置不变, 因此不需要再做峰值归一化。
    """
    values: list[float] = []
    for record in validation:
        prediction = np.asarray(
            optimizer.forward_intensity(coefficients, record.phase), dtype=np.float64
        ).T
        measured = np.asarray(record.image, dtype=np.float64)
        if prediction.shape != measured.shape:
            logger.warning("验证集形状不一致: 模型 {} vs 实测 {}",
                           prediction.shape, measured.shape)
            return None
        pred = prediction.ravel()
        meas = measured.ravel()
        if pred.size < 2 or not np.any(pred) or not np.any(meas):
            return None
        corr = float(np.corrcoef(pred, meas)[0, 1])
        if np.isfinite(corr):
            values.append(corr)
    return float(np.mean(values)) if values else None


def rms_from_ground_truth(
    records: list[ProbeRecord], coefficients: np.ndarray
) -> tuple[float | None, float | None]:
    """已知注入系数时给出拟合前后 RMS (rad); 硬件探针没有真值就返回 null。"""
    truths = [r.coefficients for r in records if r.coefficients is not None]
    if not truths:
        return None, None
    known = np.asarray(truths, dtype=np.float64)
    if known.ndim != 2 or known.shape[1] != coefficients.size:
        return None, None
    before = float(np.sqrt(np.mean(known**2)))
    after = float(np.sqrt(np.mean((known - coefficients) ** 2)))
    return before, after


def flat_fwhm(frame: np.ndarray, half: int = 60) -> tuple[float | None, float | None]:
    """平相位帧的 x/y FWHM, 与生成器一样走 ``prepare_frame`` + 峰值剖面。"""
    prepared = prepare_frame(np.asarray(frame, dtype=np.float64))
    if prepared.size == 0 or not np.any(prepared):
        return None, None
    iy, ix = np.unravel_index(int(np.argmax(prepared)), prepared.shape)
    h = int(min(half, ix, prepared.shape[1] - ix - 1))
    v = int(min(half, iy, prepared.shape[0] - iy - 1))

    def _width(profile: np.ndarray) -> float | None:
        peak = float(np.max(profile))
        if peak <= 0.0:
            return None
        above = np.where(profile >= peak / 2.0)[0]
        return float(above[-1] - above[0] + 1) if above.size else None

    return (
        _width(prepared[iy, max(0, ix - h):ix + h + 1]),
        _width(prepared[max(0, iy - v):iy + v + 1, ix]),
    )


def _probe_panels(
    args: argparse.Namespace, bench: Bench
) -> list[tuple[np.ndarray, np.ndarray]]:
    """(瞳面网格相位, 面板相位) 对; 用 canonical 的高斯探针生成器。"""
    from ao_shaping.optimizer.wfless.model_in_loop_shaping import _probe_phase

    panels: list[tuple[np.ndarray, np.ndarray]] = []
    total = int(args.probe_count) + int(args.validation_count)
    for index in range(total):
        phase = _probe_phase(
            int(args.region), float(args.probe_spread), int(args.probe_seed) + index
        )
        panels.append((phase, pupil_phase_to_panel(phase, bench)))
    return panels


def run_offline(args: argparse.Namespace, bench: Bench, out_dir: Path) -> dict[str, Any]:
    """dry-run: 只校验几何/输入; 有 B 相位+已有系数时离线重建 C (不开设备)。"""
    assert_no_drivers()
    logger.info("离线几何: 面板 {}, 0 级 {}, 相机 px/GS px = {:.4f}",
                bench.panel_shape, args.zero_order_xy, bench.camera_px_per_gs_px)

    b_panel = load_b_panel(args.b_panel, bench)
    if b_panel is None:
        return degraded_meta(f"missing_or_inconsistent_b_panel:{args.b_panel}", args, bench)
    if args.aberration_coeffs_from is None:
        return degraded_meta(
            "no_coefficients: dry-run 需要 --aberration-coeffs-from 才能重建 C; "
            "没有系数时只校验 B 相位与几何",
            args, bench,
            extra={"b_panel": str(args.b_panel), "b_panel_ok": True},
        )

    optimizer = make_optimizer(args)
    coefficients = load_coefficients(args.aberration_coeffs_from, optimizer.n_coefficients)
    if coefficients is None:
        return degraded_meta(
            f"missing_or_inconsistent_coefficients:{args.aberration_coeffs_from}",
            args, bench,
            extra={"b_panel": str(args.b_panel), "b_panel_ok": True},
        )

    phase = aberration_phase_from_coefficients(coefficients, optimizer)
    c_panel = build_c_panel(b_panel, phase, bench)
    np.save(out_dir / "c_panel.npy", c_panel.astype(np.float32))
    meta = degraded_meta(
        "offline_rebuild_no_replay: 由已存 B + 系数重建 C; 本次没有重放/验证帧",
        args, bench,
        extra={
            "b_panel": str(args.b_panel),
            "b_panel_ok": True,
            "coefficients": coefficients.tolist(),
            "aberration_coeffs_from": str(args.aberration_coeffs_from),
            "fitted_aberration_rms_rad": _num_or_none(float(np.sqrt(np.mean(phase**2)))),
        },
    )
    logger.info("离线重建完成: {} (像差瞳面 RMS {:.3f} rad)",
                out_dir / "c_panel.npy", float(np.sqrt(np.mean(phase**2))))
    return meta


def run_device(args: argparse.Namespace, bench: Bench, out_dir: Path) -> dict[str, Any]:
    """拿设备锁 -> 探针拟合 -> C/对照采集 -> 写 meta; 锁在 finally 释放。"""
    from ao_shaping.drivers.ccd.common import get_camera_exposure_ms, set_camera_exposure_ms
    from ao_shaping.tools.kernels.device_mutex import DeviceLock

    lock = DeviceLock.default()
    if not lock.acquire(timeout=DEVICE_LOCK_TIMEOUT_S):
        logger.warning("设备锁被 {} 占用, 本次不采集", lock.holder() or "未知进程")
        return degraded_meta("device_lock_unavailable", args, bench)
    logger.info("已拿设备锁 (超时 {} s); 采集期间独占 SLM #{} / {}#{}",
                DEVICE_LOCK_TIMEOUT_S, int(args.slm_number), args.cam_type, int(args.cam_id))
    try:
        return _acquire_under_lock(
            args, bench, out_dir, get_camera_exposure_ms, set_camera_exposure_ms
        )
    finally:
        lock.release()
        logger.info("设备锁已释放 (SLM #{} / {}#{})",
                    int(args.slm_number), args.cam_type, int(args.cam_id))


def _acquire_under_lock(
    args: argparse.Namespace,
    bench: Bench,
    out_dir: Path,
    get_exposure: Any,
    set_exposure: Any,
) -> dict[str, Any]:
    """已持锁的采集主体; 单独一层函数让锁的生命周期一眼可见。

    探针与四张对照帧共用**一次** ``open_bench``: 预检只做一遍, 0 级只冻结一次,
    探针与重放之间的唯一差别就是曝光设定 (逐帧写进 ``params``)。
    """
    b_panel = load_b_panel(args.b_panel, bench)
    if b_panel is None:
        return degraded_meta(f"missing_or_inconsistent_b_panel:{args.b_panel}", args, bench)
    optimizer = make_optimizer(args)

    probe_records: list[ProbeRecord] = []
    probe_exposure_ms: float | None = None
    probe_readbacks: list[float] = []
    plateau_readbacks: dict[str, float] = {}
    fit_info: dict[str, Any] = {}
    heldout_corr: float | None = None
    rms_before: float | None = None
    rms_after: float | None = None

    with open_bench(args, bench, focus_panel=None, abort_dir=out_dir,
                    abort_context={"variant": "C", "b_panel": str(args.b_panel)}) as session:
        measured_zero = getattr(session, "zero_order_xy", None)
        if (measured_zero is not None and len(measured_zero) == 2
                and all(np.isfinite(v) for v in measured_zero)):
            zero_order = (float(measured_zero[0]), float(measured_zero[1]))
        else:
            zero_order = tuple(args.zero_order_xy)
        logger.info("探针裁片 0 级 = {} (预检实测 {} / CLI {})",
                    zero_order, measured_zero, args.zero_order_xy)

        def current_exposure() -> float:
            return float(get_exposure(session.cam))

        def apply_exposure(value: float) -> float:
            readback = set_exposure(session.cam, float(value))
            return _num_or_none(readback) or current_exposure()

        if args.aberration_coeffs_from is not None:
            return degraded_meta(
                "aberration_coeffs_from_disallowed_in_execute", args, bench)
        coefficients: np.ndarray | None = None
        probe_panels = _probe_panels(args, bench)
        first_panel = probe_panels[0][1]

        def read_probe_frame() -> np.ndarray:
            return np.asarray(session.capture(first_panel), dtype=np.float64)

        setting = adapt_exposure(
            apply_exposure,
            read_probe_frame,
            start_ms=float(args.probe_exposure_start_ms),
            floor_ms=float(args.probe_exposure_floor_ms),
            ceiling_ms=float(session.exposure_ceiling_ms),
            full_scale=float(args.full_scale),
        )
        probe_exposure_ms = float(setting.exposure_ms)
        logger.info("探针曝光 = {:.4f} ms (峰值 {:.1f}, 状态 {})",
                    probe_exposure_ms, float(setting.peak), setting.status)
        half = int(args.far_field_size) // 2
        for index, (phase, panel) in enumerate(probe_panels):
            before = apply_exposure(probe_exposure_ms)
            frame = np.asarray(session.capture(panel), dtype=np.float64)
            after = current_exposure()
            label = ("probe_train_" if index < int(args.probe_count)
                     else "probe_validation_") + f"{index:02d}"
            save_frame(out_dir / f"{label}.npz", frame, dict(
                label=label,
                variant="aberration_probe",
                exposure_ms=probe_exposure_ms,
                exposure_ms_readback=after,
                exposure_ms_readback_before=before,
                probe_spread=float(args.probe_spread),
                probe_seed=int(args.probe_seed) + index,
                crop_center_xy=list(zero_order),
                crop_half=half,
            ))
            probe_readbacks.append(after)
            probe_records.append(ProbeRecord(
                label=label,
                phase=phase,
                image=crop_roi(frame, zero_order, half),
                exposure_ms=probe_exposure_ms,
                readback_ms=after,
            ))

        train = probe_records[:int(args.probe_count)]
        validation = probe_records[int(args.probe_count):]
        if len(train) != int(args.probe_count) or len(validation) != int(args.validation_count):
            return degraded_meta("insufficient_probe_frames", args, bench,
                                 probe_count=len(train), validation_count=len(validation))
        result = fit_shared_aberration(train, args)
        coefficients = np.asarray(result.coefficients, dtype=np.float64).copy()
        heldout_corr = heldout_correlation(coefficients, validation, optimizer)
        rms_before, rms_after = rms_from_ground_truth(train, coefficients)
        fit_info = dict(
            source="calibrate_shared_aberration",
            n_records=int(result.n_records),
            iterations=int(result.iterations),
            labels=list(result.labels),
            heldout_labels=[r.label for r in validation],
        )
        if heldout_corr is None or float(heldout_corr) < float(args.heldout_corr_min):
            logger.warning("验证集相关 {} < 门槛 {}, 不采 C",
                           heldout_corr, args.heldout_corr_min)
            return degraded_meta(
                "heldout_corr_below_threshold", args, bench,
                probe_count=len(train), validation_count=len(validation),
                extra={
                    "heldout_corr": _num_or_none(heldout_corr),
                    "rms_before": _num_or_none(rms_before),
                    "rms_after": _num_or_none(rms_after),
                    "coefficients": coefficients.tolist(),
                    "fit": fit_info,
                    "exposure_provenance": exposure_provenance(
                        args, probe_exposure_ms, probe_readbacks, None),
                },
            )

        aberr_phase = aberration_phase_from_coefficients(coefficients, optimizer)
        c_panel = build_c_panel(b_panel, aberr_phase, bench)
        np.save(out_dir / "c_panel.npy", c_panel.astype(np.float32))
        flat_panel = np.zeros(bench.panel_shape, dtype=np.float64)
        cflat_panel = -pupil_phase_to_panel(aberr_phase, bench)

        requested = float(args.exposure_ms)
        captured: dict[str, np.ndarray] = {}
        for name, panel in (("replay_b", b_panel), ("c", c_panel),
                            ("flat", flat_panel), ("cflat", cflat_panel)):
            before = apply_exposure(requested)
            frame = np.asarray(session.capture(panel), dtype=np.float64)
            after = current_exposure()
            plateau_readbacks[name] = after
            save_frame(out_dir / f"{name}.npz", frame, dict(
                variant=name,
                exposure_ms=requested,
                exposure_ms_readback=after,
                exposure_ms_readback_before=before,
                frames=int(args.frames),
                settle_n_discard=int(args.settle_n_discard),
                settle_wait_s=float(args.settle_wait_s),
                settle_tol=float(args.settle_tol),
                settle_max_wait_s=float(args.settle_max_wait_s),
                zero_order_xy=list(zero_order),
                box_side_px=float(args.side_px),
                phase_convention="raw_rad_unwrapped",
            ))
            captured[name] = frame
            logger.info("已采 {} (曝光读回 {:.4f} ms)", name, after)

    flat_fw = flat_fwhm(captured["flat"])
    model_fw = model_psf_fwhm_px(bench)
    ratios = [model_fw / value for value in flat_fw
              if value is not None and value > 0.0 and np.isfinite(value)]
    geometry_ok = all(0.5 <= ratio <= 2.0 for ratio in ratios) if ratios else None
    exposures_match = all(
        abs(plateau_readbacks[name] - plateau_readbacks["flat"]) <= 1e-9
        for name in ("replay_b", "c", "cflat")
    )
    meta = degraded_meta(
        "ok" if geometry_ok is not False and exposures_match else "exposure_or_geometry_mismatch",
        args, bench,
        probe_count=int(args.probe_count) if probe_records else 0,
        validation_count=int(args.validation_count) if probe_records else 0,
    )
    meta.update({
        "coefficients": coefficients.tolist(),
        "heldout_corr": _num_or_none(heldout_corr),
        "rms_before": _num_or_none(rms_before),
        "rms_after": _num_or_none(rms_after),
        "fit": fit_info,
        "fitted_aberration_rms_rad": _num_or_none(float(np.sqrt(np.mean(aberr_phase**2)))),
        "flat_fwhm_px": [_num_or_none(v) for v in flat_fw],
        "model_psf_ratio": [_num_or_none(model_fw / v) if v else None for v in flat_fw],
        "psf_peak_comparable": bool(exposures_match),
        "geometry_ok": geometry_ok,
        "exposure_provenance": exposure_provenance(
            args, probe_exposure_ms, probe_readbacks, plateau_readbacks),
    })
    return meta


def main() -> None:
    """入口; 缺输入 / 口径不一致都走降级 meta, 进程退出码保持 0。"""
    args = build_parser().parse_args()
    args.zero_order_xy = parse_xy(args.zero_order)
    if args.out_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        args.out_dir = DEFAULT_OUT_PARENT / stamp
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bench = build_bench(args)  # --wavelength-nm 0 在这里静态解析成驱动常量
    if args.execute and args.aberration_coeffs_from is not None:
        meta = degraded_meta("aberration_coeffs_from_disallowed_in_execute", args, bench)
    else:
        meta = run_device(args, bench, out_dir) if args.execute else run_offline(
            args, bench, out_dir)
    write_meta(out_dir / "correction_meta.json", meta)
    logger.info("correction_meta: {}", out_dir / "correction_meta.json")


if __name__ == "__main__":
    main()
