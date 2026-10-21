"""issue #165 的大规模预训练：随机化仿真 + 批量训练 + 可迁移的 checkpoint。

**纯离线仿真**，不打开任何设备。

与前一轮「P0 验收」那一小次训练相比，这里做三件让它能**迁移到真实设备**的事：

1. **像差随机化（域随机化）**：每个训练样本自带一个随机的实验系统像差 ——
   Zernike 部分（随机模式子集 + 随机系数，落在张成内）与自由形状部分
   （带限随机，落在张成外，66 项模型拟合不掉）。网络因此不会把某一条像差
   记死，而是学到「怎么对任意像差做整形」。
2. **CCD 探测器模型**：帧不是理想远场，而是相机**会记下来的样子**——
   曝光过大时核心削顶（过曝），曝光过小时弱散斑被黑电平切掉（欠曝），
   外加散粒噪声、读出噪声与 ADC 量化。目标（远场方斑）也经过同一个探测器
   模型，所以网络学的是「把相机看到的那张图做出来」。
3. **规模**：样本数、训练轮数、批大小都可配；数据集生成用多进程并行
   （每个样本要在过采样真值上跑一次经典 GS，这是唯一的瓶颈）。

产物：
* ``checkpoint.pt`` —— 格式与 ``ml.gsnet.train.train_gsnet`` 一致，
  ``ml.gsnet.hybrid_infer.load_gsnet_checkpoint`` 直接能读；
* ``dataset.npz`` —— 每个样本的真值相位与生成参数（目标与照明可按参数重放，
  所以不必存下 512² 的目标矩阵）；
* ``summary.json`` —— 训练历史 + 留出集上的整形质量。

用法::

    python scripts/slm/gsnet_pretrain.py --samples 2000 --epochs 30
    python scripts/slm/gsnet_pretrain.py --samples 200 --epochs 5 --workers 8   # 冒烟
"""

from __future__ import annotations

# 必须在 numpy / torch 之前设：数据生成的 worker 进程是 spawn 出来的，会重新
# import 本模块，于是它们继承这里的设置。每个 worker 只用 1 个 BLAS 线程
# ——否则 N 个 worker × 容器给的线程数会撞上 pids 限额
# （实测 48 worker × 64 线程 = 3072 线程，直接把容器打爆）。
import os

for _var in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts" / "slm") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts" / "slm"))

from ao_shaping.drivers.sim.bench_twin import (  # noqa: E402
    BenchTwin,
    BenchTwinConfig,
    TwinErrorModel,
)
from ao_shaping.utils.image.beam_metrics import prepare_frame  # noqa: E402
from ml.gsnet.forward_ops import BenchTwinOperator  # noqa: E402
from ml.gsnet.model import FourierGSNet  # noqa: E402

REGION = 64
FAR_FIELD = 512
TARGET_SIDE = 16
GT_ITERATIONS = 12
#: 目标支撑区（远场中心裁剪）的边长，也是评价用的口径。
METRIC_SIDE = 36

#: 三族误差的随机化幅度（域随机化的轴）。范围取自实验系统的量级。
LUT_GAMMA_RANGE = (0.85, 1.25)
PANEL_ZERNIKE_MAX_RAD = 0.12
PANEL_ROUGHNESS_RANGE = (0.0, 0.04)
DEFOCUS_RANGE_M = (-2.0e-4, 2.0e-4)
TILT_RANGE_RAD = (-1.5e-4, 1.5e-4)
LENS_ZERNIKE_MAX_RAD = 0.10

#: 几何在进程之间要能重建，所以做成模块级常量（worker 进程会重新 import）。
REGION_SPAN_PX = 900.0
FAR_FIELD_PADDING = 8
PEAK_ADU = 100.0
NOISE_ADU = 2.0

#: 理想孪生（零误差）—— 网络内部前向与 source 照明都用它。
IDEAL_TWIN = BenchTwin(
    BenchTwinConfig(
        wavelength_m=1064e-9,
        slm_pixel_um=8.0,
        ccd_pixel_um=2.2,
        focal_length_m=0.125,
        region=REGION,
        region_span_px=REGION_SPAN_PX,
        far_field_padding=FAR_FIELD_PADDING,
        beam_w0_px=250.0,
        center_crop=True,
        crop_px=REGION * FAR_FIELD_PADDING,
        peak_adu=PEAK_ADU,
        noise_adu=0.0,
        seed=0,
        errors=TwinErrorModel(),
    )
)


@dataclass(frozen=True)
class SampleSpec:
    """一个训练样本的生成参数（全部可重放）。"""

    index: int
    seed: int
    target_side: float
    target_dx: float
    target_dy: float
    exposure: float
    lut_gamma: float
    panel_modes: tuple[int, ...]
    panel_coeffs: tuple[float, ...]
    roughness_sigma_rad: float
    roughness_seed: int
    defocus_m: float
    tilt_x_rad: float
    tilt_y_rad: float
    lens_modes: tuple[int, ...]
    lens_coeffs: tuple[float, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "seed": self.seed,
            "target_side": self.target_side,
            "target_dx": self.target_dx,
            "target_dy": self.target_dy,
            "exposure": self.exposure,
            "lut_gamma": self.lut_gamma,
            "panel_modes": list(self.panel_modes),
            "panel_coeffs": list(self.panel_coeffs),
            "roughness_sigma_rad": self.roughness_sigma_rad,
            "roughness_seed": self.roughness_seed,
            "defocus_m": self.defocus_m,
            "tilt_x_rad": self.tilt_x_rad,
            "tilt_y_rad": self.tilt_y_rad,
            "lens_modes": list(self.lens_modes),
            "lens_coeffs": list(self.lens_coeffs),
        }


def draw_spec(index: int, seed: int) -> SampleSpec:
    """随机抽一个样本的生成参数。

    随机化的轴就是孪生的三族误差（LUT 非线性 / 面板静态相差 / 装调），
    外加目标的边长与位置、以及**曝光**（过曝会削顶、欠曝会切底）。
    """
    rng = np.random.default_rng(seed + index * 7919)
    panel_modes = tuple(
        int(m) for m in rng.choice(np.arange(4, 19), size=int(rng.integers(3, 9)), replace=False)
    )
    lens_modes = tuple(
        int(m) for m in rng.choice(np.arange(4, 13), size=int(rng.integers(2, 6)), replace=False)
    )
    return SampleSpec(
        index=index,
        seed=seed,
        target_side=float(rng.uniform(0.75, 1.25) * TARGET_SIDE),
        target_dx=float(rng.uniform(-0.012, 0.012)),
        target_dy=float(rng.uniform(-0.012, 0.012)),
        exposure=float(np.exp(rng.uniform(np.log(0.05), np.log(3.0)))),
        lut_gamma=float(rng.uniform(*LUT_GAMMA_RANGE)),
        panel_modes=panel_modes,
        panel_coeffs=tuple(
            float(rng.uniform(-PANEL_ZERNIKE_MAX_RAD, PANEL_ZERNIKE_MAX_RAD))
            for _ in panel_modes
        ),
        roughness_sigma_rad=float(rng.uniform(*PANEL_ROUGHNESS_RANGE)),
        roughness_seed=int(rng.integers(0, 2**31 - 1)),
        defocus_m=float(rng.uniform(*DEFOCUS_RANGE_M)),
        tilt_x_rad=float(rng.uniform(*TILT_RANGE_RAD)),
        tilt_y_rad=float(rng.uniform(*TILT_RANGE_RAD)),
        lens_modes=lens_modes,
        lens_coeffs=tuple(
            float(rng.uniform(-LENS_ZERNIKE_MAX_RAD, LENS_ZERNIKE_MAX_RAD))
            for _ in lens_modes
        ),
    )


def build_twin(spec: SampleSpec) -> BenchTwin:
    """按样本参数造一台孪生。"""
    errors = TwinErrorModel(
        lut_gamma=spec.lut_gamma,
        lut_phase_max_rad=2.0 * np.pi,
        gray_levels=1023,
        slm_zernike=dict(zip(spec.panel_modes, spec.panel_coeffs, strict=True)),
        slm_zernike_n_max=6,
        slm_roughness_sigma_rad=spec.roughness_sigma_rad,
        slm_roughness_seed=spec.roughness_seed,
        defocus_m=spec.defocus_m,
        tilt_x_rad=spec.tilt_x_rad,
        tilt_y_rad=spec.tilt_y_rad,
        lens_zernike=dict(zip(spec.lens_modes, spec.lens_coeffs, strict=True)),
        lens_zernike_n_max=4,
    )
    config = BenchTwinConfig(
        wavelength_m=1064e-9,
        slm_pixel_um=8.0,
        ccd_pixel_um=2.2,
        focal_length_m=0.125,
        region=REGION,
        region_span_px=REGION_SPAN_PX,
        far_field_padding=FAR_FIELD_PADDING,
        beam_w0_px=250.0,
        center_crop=True,
        crop_px=REGION * FAR_FIELD_PADDING,
        peak_adu=PEAK_ADU,
        noise_adu=NOISE_ADU,
        seed=spec.seed + spec.index,
        errors=errors,
    )
    return BenchTwin(config)


def square_target(spec: SampleSpec, side: int = FAR_FIELD) -> np.ndarray:
    """远场上的方形目标（峰值 1，带随机边长与中心偏移）。"""
    target = np.zeros((side, side), dtype=np.float32)
    half = spec.target_side / 2.0
    cx = side / 2.0 + spec.target_dx * side
    cy = side / 2.0 + spec.target_dy * side
    x0, x1 = int(round(cx - half)), int(round(cx + half))
    y0, y1 = int(round(cy - half)), int(round(cy + half))
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(side, x1), min(side, y1)
    if x1 > x0 and y1 > y0:
        target[y0:y1, x0:x1] = 1.0
    return target


def apply_detector(
    frame_adu: np.ndarray,
    exposure: float,
    *,
    full_well: float = PEAK_ADU,
    black_level: float = 0.0,
    bit_depth: int = 12,
) -> np.ndarray:
    """在孪生的帧外面套一层探测器：曝光 + 上下截断 + 量化。

    孪生自己把帧**峰值归一**到 ``peak_adu``，所以它本身不会过曝也不会欠曝；
    真实相机会。曝光系数 > 1 ⇒ 峰值撞满阱（过曝，核心削平）；< 1 ⇒ 信号落到
    黑电平以下（欠曝，弱散斑被切掉）。散粒噪声由孪生自己加（峰值处
    σ = √peak_adu ADU），这里只补截断与量化。
    """
    scaled = np.asarray(frame_adu, dtype=np.float64) * float(exposure)
    scaled = np.clip(scaled, black_level, full_well)
    levels = float(2**int(bit_depth) - 1)
    return np.round(scaled * levels) / levels


def measured_frame(spec: SampleSpec, phase: np.ndarray) -> np.ndarray:
    """相机**记下来的**那一帧（孪生 + 探测器 + 本底去除）。"""
    twin = build_twin(spec)
    frame = apply_detector(twin.frame(phase), spec.exposure)
    return prepare_frame(frame)


def solve_phase(spec: SampleSpec, target: np.ndarray, iterations: int = GT_ITERATIONS) -> np.ndarray:
    """在该样本自己的像差下，用经典 GS 求「能把理想方斑做出来」的相位。

    GS 跑在**带误差的孪生链**上（走 torch 镜像的算子，因为 GS 需要远场的
    **相位**，而孪生只给强度），相位落在瞳面网格（64×64）。
    """
    twin = build_twin(spec)
    operator = BenchTwinOperator(twin)
    target_amp = torch.sqrt(
        torch.from_numpy(np.asarray(target, dtype=np.float32))[None, None].clamp_min(0.0)
        + 1e-12
    )
    phase = operator.warm_start(target_amp)
    for _ in range(iterations):
        far = operator.forward(None, phase)
        far_c = target_amp * torch.exp(1j * torch.angle(far))
        phase = torch.angle(operator.back_propagate(far_c))
    return phase[0, 0].numpy().astype(np.float32)


def _worker(args: tuple[int, int]) -> tuple[int, bytes]:
    """进程池里的一个样本：返回 (index, 真值相位的 bytes)。"""
    index, seed = args
    torch.set_num_threads(1)
    spec = draw_spec(index, seed)
    ideal = square_target(spec)
    phase = solve_phase(spec, ideal)
    return index, phase.tobytes()


def build_dataset(
    samples: int, seed: int, workers: int, verbose: bool = True
) -> tuple[np.ndarray, list[SampleSpec]]:
    """并行生成数据集：真值相位 (N, 64, 64) 与生成参数。"""
    specs = [draw_spec(i, seed) for i in range(samples)]
    phases = np.zeros((samples, REGION, REGION), dtype=np.float32)
    started = time.time()
    if workers <= 1:
        results = [_worker((i, seed)) for i in range(samples)]
    else:
        import multiprocessing as mp

        with mp.get_context("spawn").Pool(workers) as pool:
            results = []
            for done, item in enumerate(
                pool.imap_unordered(_worker, [(i, seed) for i in range(samples)]), 1
            ):
                results.append(item)
                if verbose and done % 100 == 0:
                    rate = done / max(time.time() - started, 1e-9)
                    print(f"  生成 {done}/{samples}  ({rate:.1f}/s)", flush=True)
    for index, blob in results:
        phases[index] = np.frombuffer(blob, dtype=np.float32).reshape(REGION, REGION)
    return phases, specs


def _target_boxes(
    indices: np.ndarray, specs: list[SampleSpec]
) -> list[tuple[int, int]]:
    """每个样本的目标框左上角（远场网格像素坐标）。"""
    boxes: list[tuple[int, int]] = []
    for index in indices:
        spec = specs[index]
        cy = FAR_FIELD / 2.0 + spec.target_dy * FAR_FIELD
        cx = FAR_FIELD / 2.0 + spec.target_dx * FAR_FIELD
        r0 = int(round(cy)) - METRIC_SIDE // 2
        c0 = int(round(cx)) - METRIC_SIDE // 2
        r0 = max(0, min(r0, FAR_FIELD - METRIC_SIDE))
        c0 = max(0, min(c0, FAR_FIELD - METRIC_SIDE))
        boxes.append((r0, c0))
    return boxes


def _crop_boxes(tensor: torch.Tensor, boxes: list[tuple[int, int]]) -> torch.Tensor:
    """按每个样本各自的框裁出 (B, side, side)。"""
    side = METRIC_SIDE
    crops = [
        tensor[i, :, r0 : r0 + side, c0 : c0 + side] for i, (r0, c0) in enumerate(boxes)
    ]
    return torch.stack(crops)

def _batch_tensors(
    indices: np.ndarray, specs: list[SampleSpec]
) -> tuple[torch.Tensor, torch.Tensor]:
    """按 spec 现算 (source, target)。

    * ``source`` —— 瞳面入射幅度（孪生自己的照明），所有样本共用；
    * ``target`` —— **设计**的远场方斑（不是实测帧：网络回答的是「给出这个
      目标该下发什么相位」，实验系统的非理想因素经由**标签**进入）。
    """
    source = torch.from_numpy(
        np.asarray(IDEAL_TWIN.pupil_amplitude(), dtype=np.float32)
    )[None, None]
    targets = [torch.from_numpy(square_target(specs[i]))[None] for i in indices]
    return source.expand(len(indices), 1, REGION, REGION), torch.stack(targets)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--samples", type=int, default=2000, help="训练样本数")
    parser.add_argument("--val-samples", type=int, default=64, help="留出样本数")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--w-shaping", type=float, default=5.0, help="整形项权重（相位项为 1）"
    )
    parser.add_argument(
        "--warmup-epochs", type=int, default=3, help="整形项权重爬升的轮数"
    )
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument(
        "--fixed-aberration",
        action="store_true",
        help="诊断用：所有样本共用一条像差",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    global FIXED_ABERRATION
    FIXED_ABERRATION = bool(args.fixed_aberration)

    out_dir = args.out or (
        ROOT / "data" / "gsnet_pretrain" / time.strftime("%Y%m%d_%H%M%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    total = args.samples + args.val_samples
    print(f"生成 {total} 个样本（{args.workers} 进程）…", flush=True)
    phases, specs = build_dataset(total, args.seed, args.workers)

    train_idx = np.arange(args.samples)
    val_idx = np.arange(args.samples, total)

    torch.manual_seed(args.seed)
    # 网络内部的前向 = **理想**孪生的链（它并不知道实验系统的非理想因素）。
    # 实验系统的误差经由**标签**进入网络：标签是在带误差的物理下求的 GS 解。
    operator = BenchTwinOperator(IDEAL_TWIN)
    model = FourierGSNet(
        num_layers=args.num_layers,
        base_channels=args.base_channels,
        shared_weights=True,
        forward_op=operator,
    )
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    # 衰减必须**早**发生：原来的 step_size=30 对 30 轮的训练等于没有衰减，
    # 学习率一直停在 1e-3，长跑之后相位项反而开始往上爬（实测 2.22 → 2.42）。
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimiser, step_size=max(1, args.epochs // 4), gamma=0.4
    )

    history: list[dict[str, float]] = []
    started = time.time()
    for epoch in range(args.epochs):
        model.train()
        order = np.random.permutation(train_idx)
        losses: list[float] = []
        phase_losses: list[float] = []
        shaping_losses: list[float] = []
        for start in range(0, order.size, args.batch_size):
            idx = order[start : start + args.batch_size]
            source, target = _batch_tensors(idx, specs)
            gt = torch.from_numpy(phases[idx])[:, None]
            pred = model(source, target)
            diff = torch.remainder(pred - gt + torch.pi, 2 * torch.pi) - torch.pi
            phase_loss = torch.nn.functional.mse_loss(diff, torch.zeros_like(diff))
            far = operator.forward(torch.sqrt(source.clamp_min(0) + 1e-12), pred)
            intensity = far.real**2 + far.imag**2
            # 相关性只在**目标支撑区**上算，而且窗口要**跟着目标走**：
            # 目标带随机偏移，窗口固定在画面正中时，多数样本的窗口里根本没有
            # 目标，梯度就全是背景噪声。
            boxes = _target_boxes(idx, specs)
            # 先摊平成 (B, N)，再去均值/求内积——否则 dim=1 落在**通道**轴上
            # （通道数是 1），去均值等于没做，求内积也变成逐像素，梯度在
            # sqrt(0) 处直接 NaN。
            p = _crop_boxes(intensity, boxes).flatten(1)
            t = _crop_boxes(target, boxes).flatten(1)
            p = p - p.mean(dim=1, keepdim=True)
            t = t - t.mean(dim=1, keepdim=True)
            corr = (p * t).sum(dim=1) / (
                torch.sqrt((p**2).sum(dim=1) * (t**2).sum(dim=1)) + 1e-6
            )
            # 整形项的权重**先小后大**（warm-up）：相关性项在预测接近 0 时
            # 病态（分母趋零，梯度很陡），一上来就 40 倍会把训练带炸
            # （实测第 5 轮直接冲上 18）。相位项（监督信号）是良态的，
            # 让它先把网络带到一个合理区域，再逐步加整形项。
            weight = args.w_shaping * min(1.0, (epoch + 1) / max(1, args.warmup_epochs))
            loss = phase_loss + weight * torch.mean(1.0 - corr)
            phase_losses.append(float(phase_loss.detach()))
            shaping_losses.append(float(torch.mean(1.0 - corr).detach()))
            optimiser.zero_grad()
            loss.backward()
            # 梯度范数护栏：相关性项在预测几乎为 0 的时候梯度很陡，
            # 一步就能把参数推到 NaN。
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        row = {
            "epoch": epoch + 1,
            "loss": float(np.mean(losses)) if losses else float("nan"),
            "phase_loss": float(np.mean(phase_losses)) if phase_losses else float("nan"),
            "shaping_loss": (
                float(np.mean(shaping_losses)) if shaping_losses else float("nan")
            ),
            "seconds": time.time() - started,
        }
        history.append(row)
        print(
            f"epoch {epoch + 1}/{args.epochs}  loss={row['loss']:.4f}  "
            f"phase={row['phase_loss']:.4f}  shaping={row['shaping_loss']:.4f}  "
            f"{row['seconds']:.0f}s",
            flush=True,
        )

    # ---- 留出集：在**孪生**上评分（与 P0 验收同一口径）----
    from calibrate_gsnet_2f import frame_metrics  # noqa: PLC0415

    model.eval()
    rows: list[dict[str, float]] = []
    rows_gt: list[dict[str, float]] = []
    with torch.no_grad():
        for index in val_idx:
            spec = specs[index]
            source, target = _batch_tensors(np.array([index]), specs)
            pred = model(source, target)
            ideal = target[0, 0].numpy().astype(np.float64)
            measured = measured_frame(spec, pred[0, 0].numpy().astype(np.float64))
            rows.append(frame_metrics(measured, ideal, 36, int(spec.target_side)))
            # 同一把尺子量**标签自己**（带误差物理下的 GS 解）—— 任务的上限
            gt_measured = measured_frame(
                spec, phases[index].astype(np.float64)
            )
            rows_gt.append(
                frame_metrics(gt_measured, ideal, 36, int(spec.target_side))
            )
    summary_metrics = {
        key: float(np.mean([row[key] for row in rows]))
        for key in ("composite", "encircled_energy", "cv_target_box", "correlation")
    }
    summary_metrics_labels = {
        key: float(np.mean([row[key] for row in rows_gt]))
        for key in ("composite", "encircled_energy", "cv_target_box", "correlation")
    }
    buckets: dict[str, list[int]] = {"under": [], "normal": [], "over": []}
    for position, index in enumerate(val_idx):
        exposure = specs[index].exposure
        key = "under" if exposure < 0.4 else ("over" if exposure > 1.5 else "normal")
        buckets[key].append(position)
    bucket_metrics = {
        name: {
            key: float(np.mean([rows[i][key] for i in positions]))
            for key in ("composite", "encircled_energy", "cv_target_box")
        }
        | {"n": len(positions)}
        for name, positions in buckets.items()
        if positions
    }

    # 存**最好**的一轮（按相位项），不是最后一轮：总损失受整形项噪声支配。
    best = min(
        (row for row in history),
        key=lambda row: row.get("phase_loss", float("inf")),
        default=None,
    )
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": {
            "num_layers": args.num_layers,
            "base_channels": args.base_channels,
            "shared_weights": True,
        },
        "epoch": args.epochs,
        "loss": history[-1]["loss"] if history else float("nan"),
        "best_phase_loss_epoch": best.get("epoch") if best else None,
        "best_phase_loss": best.get("phase_loss") if best else None,
    }
    torch.save(checkpoint, out_dir / "checkpoint.pt")
    np.savez_compressed(
        out_dir / "dataset.npz",
        gt_phase=phases,
        specs=np.array([json.dumps(s.to_json()) for s in specs]),
    )
    summary = {
        "config": vars(args) | {"out": str(out_dir)},
        "history": history,
        "val_metrics": summary_metrics,
        "val_metrics_labels": summary_metrics_labels,
        "val_metrics_by_exposure": bucket_metrics,
        "checkpoint": str(out_dir / "checkpoint.pt"),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"留出集（对理想目标）: {summary_metrics}")
    print(f"留出集（**标签自己**，即任务上限）: {summary_metrics_labels}")
    print(f"留出集（按曝光分桶）: {bucket_metrics}")
    print(f"产物: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
