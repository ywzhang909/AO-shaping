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

from ao_shaping.utils.wavefront.bench2f import (  # noqa: E402
    Bench2FGeometry,
    Bench2FGroundTruth,
    Bench2FPropagator,
    CCDSensor,
    ZernikeAberrationField,
)
from ml.gsnet.forward_ops import Bench2FOperator, IdealFFTOperator  # noqa: E402
from ml.gsnet.model import FourierGSNet  # noqa: E402

REGION = 64
FAR_FIELD = 512
TARGET_SIDE = 16
GT_ITERATIONS = 12
#: 目标支撑区（远场中心裁剪）的边长，也是评价用的口径。
METRIC_SIDE = 36

#: 几何在进程之间要能重建，所以做成模块级常量（worker 进程会重新 import）。
GEOMETRY = Bench2FGeometry(
    region=REGION,
    far_field_size=FAR_FIELD,
    panel_disc_radius_px=450.0,
    beam_waist_panel_px=450.0,
    camera_px_per_model_px=8.0,
)


@dataclass(frozen=True)
class SampleSpec:
    """一个训练样本的生成参数（全部可重放）。"""

    index: int
    seed: int
    source_waist: float
    source_dx: float
    source_dy: float
    target_side: float
    target_dx: float
    target_dy: float
    exposure: float
    sensor_seed: int
    aberration_seed: int

    def to_json(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "seed": self.seed,
            "source_waist": self.source_waist,
            "source_dx": self.source_dx,
            "source_dy": self.source_dy,
            "target_side": self.target_side,
            "target_dx": self.target_dx,
            "target_dy": self.target_dy,
            "exposure": self.exposure,
            "sensor_seed": self.sensor_seed,
            "aberration_seed": self.aberration_seed,
        }


def draw_spec(index: int, seed: int) -> SampleSpec:
    """随机抽一个样本的生成参数。

    曝光在 log 均匀的 [0.05, 3] 上抽：<1 是欠曝（弱散斑被黑电平切掉），
    >1 是过曝（核心削顶）。
    """
    rng = np.random.default_rng(seed + index * 7919)
    return SampleSpec(
        index=index,
        seed=seed,
        source_waist=float(rng.uniform(0.7, 1.3)),
        source_dx=float(rng.uniform(-0.08, 0.08)),
        source_dy=float(rng.uniform(-0.08, 0.08)),
        target_side=float(rng.uniform(0.6, 1.4) * TARGET_SIDE),
        # 目标的**位置**偏移只取很小一段：网络这一版学的是「形状」，
        # 指向由闭环自己的 ROI（argmax 锚定）负责。偏移一大，网络预测的
        # 远场（起点在画面中心）与目标就对不上，损失窗口里没有信号，
        # 相关性的梯度会炸。
        target_dx=float(rng.uniform(-0.012, 0.012)),
        target_dy=float(rng.uniform(-0.012, 0.012)),
        exposure=float(np.exp(rng.uniform(np.log(0.05), np.log(3.0)))),
        sensor_seed=int(rng.integers(0, 2**31 - 1)),
        aberration_seed=int(rng.integers(0, 2**31 - 1)),
    )


def source_intensity(spec: SampleSpec) -> np.ndarray:
    """瞳面照明（高斯，带随机腰宽与中心偏移），峰值归一。"""
    axis = np.arange(REGION) - (REGION - 1) / 2.0
    waist = (REGION / 2.0) * spec.source_waist
    xx, yy = np.meshgrid(
        axis - spec.source_dx * REGION, axis - spec.source_dy * REGION, indexing="ij"
    )
    src = np.exp(-2.0 * (xx**2 + yy**2) / (waist**2))
    return (src / src.max()).astype(np.float32)


def square_target(spec: SampleSpec, side: int = FAR_FIELD) -> np.ndarray:
    """远场方斑（带随机边长与中心偏移）。"""
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


#: 诊断开关：置 True 时所有样本共用同一条像差（用于把「学不到」拆成
#: 「规模不够」还是「任务本身不可能」）。
FIXED_ABERRATION = False


def random_aberration(spec: SampleSpec) -> torch.Tensor:
    """随机的实验系统像差：张成内的 Zernike + 张成外的自由形状。

    Zernike 部分用**随机模式子集**（不是固定的前若干项），这样网络不会
    只见过某几阶；自由形状部分带限随机，66 项模型拟合不掉——它代表真实
    设备上「物理模型解释不了」的那部分。
    """
    rng = np.random.default_rng(
        0 if FIXED_ABERRATION else spec.aberration_seed
    )
    field = ZernikeAberrationField(region=REGION, n_max=10)
    basis = field.basis().numpy()
    n_terms = basis.shape[0]
    chosen = rng.choice(np.arange(3, n_terms), size=int(rng.integers(4, 18)), replace=False)
    coeffs = rng.normal(0.0, 1.0, size=chosen.size)
    zernike = np.tensordot(coeffs, basis[chosen], axes=(0, 0))
    zernike_rms = float(rng.uniform(0.15, 0.9))
    scale = zernike_rms / max(float(np.sqrt((zernike**2).mean())), 1e-9)
    zernike = zernike * scale

    raw = rng.normal(0.0, 1.0, size=(REGION, REGION))
    k = np.fft.fftfreq(REGION)[:, None]
    l = np.fft.fftfreq(REGION)[None, :]
    kernel = np.exp(-2.0 * (np.pi**2) * (2.0**2) * (k**2 + l**2))
    freeform = np.real(np.fft.ifft2(np.fft.fft2(raw) * kernel))
    freeform_rms = float(rng.uniform(0.02, 0.15))
    fscale = freeform_rms / max(float(np.sqrt((freeform**2).mean())), 1e-9)
    total = zernike + freeform * fscale
    return torch.from_numpy(total.astype(np.float32))[None, None]


def build_sensor(spec: SampleSpec) -> CCDSensor:
    """该样本的 CCD 探测器（噪声/量化位深固定，曝光按样本随机）。"""
    return CCDSensor(
        exposure=spec.exposure,
        full_well=1.0,
        black_level=0.0,
        bit_depth=12,
        shot_noise=0.02,
        read_noise=0.004,
        generator=torch.Generator().manual_seed(spec.sensor_seed),
    )


def measured_target(spec: SampleSpec) -> np.ndarray:
    """相机**记下来的**目标帧（方斑过探测器模型）。"""
    ideal = torch.from_numpy(square_target(spec))[None, None]
    return build_sensor(spec).apply(ideal)[0, 0].numpy().astype(np.float32)


def _pupil_back(far_field: torch.Tensor, pupil_side: int) -> torch.Tensor:
    back = torch.fft.ifft2(torch.fft.ifftshift(far_field, dim=(-2, -1)), dim=(-2, -1))
    back = torch.fft.fftshift(back, dim=(-2, -1))
    side = back.shape[-1]
    half = (side - pupil_side) // 2
    return back[..., half : half + pupil_side, half : half + pupil_side]


def solve_phase(
    spec: SampleSpec, target: np.ndarray, iterations: int = GT_ITERATIONS
) -> np.ndarray:
    """在该样本自己的像差下，用经典 GS 求「能把**理想**方斑做出来」的相位。

    GS 跑在**过采样真值**上（512²），相位落在瞳面网格（64²）。

    目标是**理想**方斑而不是相机记下来的那一帧：相位决定远场的形状，
    曝光只决定整帧的亮度。标签必须是「好相位」，否则网络学的是怎么把
    一张欠曝/过曝的帧复制一遍——那在闭环里没有用（闭环自己会用实测帧
    继续优化，网络只负责给一个**起点**）。
    """
    twin = Bench2FGroundTruth(
        GEOMETRY, freeform=None, illumination_exponent=2.0, shot_noise=0.0, read_noise=0.0
    )
    aberration = random_aberration(spec)
    # GS 的幅度约束用**归一化**后的**理想**目标：相位决定远场的形状，
    # 曝光只决定整帧的亮度。
    ideal = torch.from_numpy(target).clamp_min(0.0)[None, None]
    ideal = ideal / ideal.max().clamp_min(1e-12)
    target_amp = torch.sqrt(ideal + 1e-12)
    phase = torch.angle(_pupil_back(target_amp.to(torch.complex64), REGION))
    for _ in range(iterations):
        far = twin.complex_far_field(phase, aberration=aberration)
        far_c = target_amp * torch.exp(1j * torch.angle(far))
        phase = torch.angle(_pupil_back(far_c, REGION))
    return phase[0, 0].numpy().astype(np.float32)


def _worker(args: tuple[int, int]) -> tuple[int, bytes, list[float]]:
    """进程池里的一个样本：返回 (index, gt_phase 的 bytes, 目标参数)。"""
    index, seed = args
    torch.set_num_threads(1)
    spec = draw_spec(index, seed)
    ideal = square_target(spec)
    phase = solve_phase(spec, ideal)
    return index, phase.tobytes(), [
        spec.target_side,
        spec.target_dx,
        spec.target_dy,
        spec.exposure,
        spec.sensor_seed,
    ]


def build_dataset(
    samples: int, seed: int, workers: int, verbose: bool = True
) -> tuple[np.ndarray, list[SampleSpec], np.ndarray, np.ndarray]:
    """并行生成数据集：真值相位 (N, 64, 64) 与目标参数。"""
    specs = [draw_spec(i, seed) for i in range(samples)]
    phases = np.zeros((samples, REGION, REGION), dtype=np.float32)
    # 网络的**输入**目标是相机记下来的那一帧（欠曝/过曝都被削顶或淹没）；
    # 评价另用理想帧。
    targets = np.zeros((samples, FAR_FIELD, FAR_FIELD), dtype=np.float16)
    ideals = np.zeros((samples, FAR_FIELD, FAR_FIELD), dtype=np.float16)
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
                if verbose and done % 50 == 0:
                    rate = done / max(time.time() - started, 1e-9)
                    print(f"  生成 {done}/{samples}  ({rate:.1f}/s)", flush=True)
    for index, blob, _params in results:
        phases[index] = np.frombuffer(blob, dtype=np.float32).reshape(REGION, REGION)
    for spec in specs:
        targets[spec.index] = measured_target(spec).astype(np.float16)
        ideals[spec.index] = square_target(spec).astype(np.float16)
    return phases, specs, targets, ideals



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
    indices: np.ndarray, specs: list[SampleSpec], targets: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sources = np.stack([source_intensity(specs[i]) for i in indices])
    return (
        torch.from_numpy(sources)[:, None],
        torch.from_numpy(targets[indices].astype(np.float32))[:, None],
        torch.from_numpy(targets[indices].astype(np.float32))[:, None],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--samples", type=int, default=2000, help="训练样本数")
    parser.add_argument("--val-samples", type=int, default=64, help="留出样本数")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
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
    phases, specs, targets, ideals = build_dataset(total, args.seed, args.workers)

    train_idx = np.arange(args.samples)
    val_idx = np.arange(args.samples, total)

    torch.manual_seed(args.seed)
    propagator = Bench2FPropagator(GEOMETRY, n_max=10)
    operator = Bench2FOperator(propagator)
    model = FourierGSNet(
        num_layers=args.num_layers,
        base_channels=args.base_channels,
        shared_weights=True,
        forward_op=operator,
    )
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=30, gamma=0.5)

    history: list[dict[str, float]] = []
    started = time.time()
    for epoch in range(args.epochs):
        model.train()
        order = np.random.permutation(train_idx)
        losses: list[float] = []
        for start in range(0, order.size, args.batch_size):
            idx = order[start : start + args.batch_size]
            source, target, _ = _batch_tensors(idx, specs, targets)
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
            loss = phase_loss + 40.0 * torch.mean(1.0 - corr)
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
            "seconds": time.time() - started,
        }
        history.append(row)
        print(
            f"epoch {epoch + 1}/{args.epochs}  loss={row['loss']:.4f}  "
            f"{row['seconds']:.0f}s",
            flush=True,
        )

    # ---- 留出集：在**过采样真值**上评分（与 P0 验收同一口径） ----
    from calibrate_gsnet_2f import frame_metrics  # noqa: PLC0415

    model.eval()
    rows: list[dict[str, float]] = []
    rows_measured: list[dict[str, float]] = []
    rows_gt: list[dict[str, float]] = []
    with torch.no_grad():
        for index in val_idx:
            spec = specs[index]
            source, target, _ = _batch_tensors(np.array([index]), specs, targets)
            pred = model(source, target)
            twin = Bench2FGroundTruth(
                GEOMETRY,
                freeform=None,
                illumination_exponent=2.0,
                shot_noise=0.0,
                read_noise=0.0,
            )
            ideal = ideals[index].astype(np.float64)
            # 度量窗口跟着目标走（和仓库的 ROI 指标一致）
            cy = FAR_FIELD / 2.0 + spec.target_dy * FAR_FIELD
            cx = FAR_FIELD / 2.0 + spec.target_dx * FAR_FIELD
            box = (cy, cx)
            far = twin.forward(
                pred, aberration=random_aberration(spec)
            )[0, 0].numpy().astype(np.float64)
            # 同一把尺子量**标签自己**（GS 解）：它给出这个任务的**上限**。
            # 网络追不上时，先看标签本身能得多少分——否则分不清是网络不行
            # 还是标签本身就不成形。
            gt_far = twin.forward(
                torch.from_numpy(phases[index])[None, None],
                aberration=random_aberration(spec),
            )[0, 0].numpy().astype(np.float64)
            rows_gt.append(
                frame_metrics(gt_far, ideal, 36, int(spec.target_side), box)
            )
            rows.append(
                frame_metrics(far, ideal, 36, int(spec.target_side), box)
            )
            measured = targets[index].astype(np.float64)
            rows_measured.append(
                frame_metrics(far, measured, 36, int(spec.target_side), box)
            )
    summary_metrics = {
        key: float(np.mean([row[key] for row in rows]))
        for key in ("composite", "encircled_energy", "cv_target_box", "correlation")
    }
    summary_metrics_measured = {
        key: float(np.mean([row[key] for row in rows_measured]))
        for key in ("composite", "encircled_energy", "cv_target_box", "correlation")
    }
    summary_metrics_labels = {
        key: float(np.mean([row[key] for row in rows_gt]))
        for key in ("composite", "encircled_energy", "cv_target_box", "correlation")
    }
    # 按曝光分桶：欠曝/正常/过曝。**不能把三桶平均成一个数**——
    # 欠曝的目标几乎全是噪声、过曝的目标被削顶，两者的上限本来就低。
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

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": {
            "num_layers": args.num_layers,
            "base_channels": args.base_channels,
            "shared_weights": True,
        },
        "epoch": args.epochs,
        "loss": history[-1]["loss"] if history else float("nan"),
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
        "val_metrics_vs_measured_target": summary_metrics_measured,
        "val_metrics_labels": summary_metrics_labels,
        "val_metrics_by_exposure": bucket_metrics,
        "checkpoint": str(out_dir / "checkpoint.pt"),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"留出集（对理想目标）: {summary_metrics}")
    print(f"留出集（对相机记下的目标）: {summary_metrics_measured}")
    print(f"留出集（**标签自己**，即任务上限）: {summary_metrics_labels}")
    print(f"留出集（按曝光分桶）: {bucket_metrics}")
    print(f"产物: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
