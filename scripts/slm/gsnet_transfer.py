"""issue #165 P1 的**迁移学习**：把大规模仿真预训练的网络，用真实实验系统的
(相位 → 帧) 数据微调，然后导出可交给 ``--init-from-gsnet`` 的相位。

**离线路径**（``--from-dir``）与**在机路径**（``--hardware``）共用同一套微调逻辑。

数据从哪来
----------
model-in-loop 的每一次探针测量都会落盘（``<debug 目录>/probes/*.npz``）：

* ``phase``      —— 下发给 SLM 的**瞳面网格相位**（弧度）＝**标签**；
* ``model_grid`` —— 相机帧映射到**远场模型网格**后的结果＝**网络输入的目标**；
* ``raw_frame`` / ``frame`` —— 相机原始帧 / 去底噪后的帧。

外加 ``<debug 目录>/source_amplitude.npy`` —— 标定出来的瞳面照明振幅＝
**网络的 source 输入**。

所以真实数据上的监督信号是**现成的**：命令了哪个相位，就测到了哪张帧。
微调就是让网络学会「给出这张帧的那个相位」，也就是真实台架上的逆问题。

损失 = 相位项（对命令相位）+ 整形项（**用标定过的前向算子**把网络预测的相位
推到相机看到的那张帧上）。第二项是关键：它把真实台架的孔径、照明、像差与
探测器行为都算进来，而预训练时这些只是**随机化过的先验**。

用法
----
离线（拿一段已有的 model-in-loop 运行）::

    python scripts/slm/gsnet_transfer.py \
        --checkpoint data/gsnet_pretrain/<ts>/checkpoint.pt \
        --from-dir data/debug/slm_model_in_loop/<ts> \
        --epochs 20 --lr 1e-4

在机（现场采一批探针再微调）::

    python scripts/slm/gsnet_transfer.py \
        --checkpoint data/gsnet_pretrain/<ts>/checkpoint.pt \
        --hardware --capture 32 \
        --cam-type daheng --cam-id 0 --slm-number 1 --slm-wavelength 1064 \
        --region 128 --far-field-padding 8 --exposure-time-ms auto
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ao_shaping.utils.wavefront.bench2f import (  # noqa: E402
    Bench2FGeometry,
    Bench2FPropagator,
)
from ml.gsnet.forward_ops import Bench2FOperator  # noqa: E402
from ml.gsnet.hybrid_infer import load_gsnet_checkpoint  # noqa: E402
from ml.gsnet.model import FourierGSNet  # noqa: E402


def _load_probe_dataset(
    run_dir: Path, limit: int | None = None
) -> dict[str, np.ndarray]:
    """从一段 model-in-loop 运行里读出迁移学习要用的数据。"""
    probes_dir = run_dir / "probes"
    if not probes_dir.is_dir():
        raise FileNotFoundError(f"no probes directory under {run_dir}")
    amplitude_path = run_dir / "source_amplitude.npy"
    if not amplitude_path.exists():
        raise FileNotFoundError(
            f"{amplitude_path} missing: run the loop with --debug so the calibrated "
            "pupil amplitude is saved"
        )
    amplitude = np.load(amplitude_path).astype(np.float32)
    if amplitude.ndim == 1:
        # 一维剖面（中心对称）也能用：沿两个方向铺成方阵即可。
        amplitude = np.tile(amplitude.reshape(-1, 1), (1, amplitude.size))

    sources: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for path in sorted(probes_dir.glob("*.npz")):
        with np.load(path) as data:
            if "phase" not in data or "model_grid" not in data:
                continue
            sources.append(amplitude)
            targets.append(np.asarray(data["model_grid"], dtype=np.float32))
            labels.append(np.asarray(data["phase"], dtype=np.float32))
        if limit is not None and len(labels) >= limit:
            break
    if not labels:
        raise ValueError(f"no usable probe records under {probes_dir}")
    return {
        "source": np.stack(sources),
        "target": np.stack(targets),
        "label": np.stack(labels),
    }


def _capture_live(
    args: argparse.Namespace, out_dir: Path
) -> dict[str, np.ndarray]:
    """在实验系统上现场采一批 (相位, 帧) 对。

    复用 model-in-loop 的硬件台架（``_HardwareBench``）与它的帧→远场模型网格
    映射（``_to_model_grid``）：那两条链路就是闭环真正用的，采到的数据与闭环
    看到的完全一致。
    """
    from ao_shaping.optimizer.wfless.slm_model_in_loop import (
        SlmModelInLoopConfig,
        _HardwareBench,
        _to_model_grid,
    )

    config = SlmModelInLoopConfig(
        cam_type=args.cam_type,
        cam_id=args.cam_id,
        slm_number=args.slm_number,
        slm_wavelength=args.slm_wavelength,
        region=args.region,
        far_field_padding=args.far_field_padding,
        cam_size=args.cam_size,
        target_side=args.target_side,
        exposure_time_ms=0.0 if args.exposure_time_ms == "auto" else args.exposure_time_ms,
        max_exposure_ms=args.max_exposure_ms,
        debug_dir=str(out_dir),
    )
    bench = _HardwareBench(config)
    bench.open()
    try:
        source = np.asarray(bench.source_amplitude, dtype=np.float32)
        sources: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        rng = np.random.default_rng(args.seed)
        for index in range(args.capture):
            phase = rng.uniform(-np.pi, np.pi, size=(args.region, args.region))
            bench.display(phase)
            frame = bench.measure()
            grid = _to_model_grid(
                frame,
                bench.signal_center_local,
                int(config.region) * int(config.far_field_padding),
                1.0,
                0,
            )
            sources.append(source)
            targets.append(np.asarray(grid, dtype=np.float32))
            labels.append(phase.astype(np.float32))
        return {
            "source": np.stack(sources),
            "target": np.stack(targets),
            "label": np.stack(labels),
        }
    finally:
        bench.close()


def _geometry_from_arrays(dataset: dict[str, np.ndarray]) -> Bench2FGeometry:
    """按数据实际的网格建几何（瞳面 = 标签的边长，远场 = 目标的边长）。"""
    region = int(dataset["label"].shape[-1])
    far_field = int(dataset["target"].shape[-1])
    return Bench2FGeometry(
        region=region,
        far_field_size=far_field,
        panel_disc_radius_px=450.0,
        beam_waist_panel_px=450.0,
        camera_px_per_model_px=8.0,
    )


def finetune(
    model: FourierGSNet,
    operator: Bench2FOperator,
    dataset: dict[str, np.ndarray],
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
    verbose: bool = True,
) -> list[dict[str, float]]:
    """在真实数据上微调。

    相位项把网络拉向**命令过的相位**；整形项把网络预测的相位经**标定前向**
    推到**相机看到的那张帧**上。
    """
    torch.manual_seed(seed)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    source = torch.from_numpy(dataset["source"])[:, None]
    target = torch.from_numpy(dataset["target"])[:, None]
    label = torch.from_numpy(dataset["label"])[:, None]
    source_amp = torch.sqrt(source.clamp_min(0.0) + 1e-12)
    n = source.shape[0]
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(n)
        phase_losses: list[float] = []
        shaping_losses: list[float] = []
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            pred = model(source[idx], target[idx])
            diff = torch.remainder(pred - label[idx] + torch.pi, 2 * torch.pi) - torch.pi
            phase_loss = torch.nn.functional.mse_loss(diff, torch.zeros_like(diff))
            far = operator.forward(source_amp[idx], pred)
            intensity = far.real**2 + far.imag**2
            half = (intensity.shape[-1] - min(36, intensity.shape[-1])) // 2
            sl = slice(half, half + min(36, intensity.shape[-1]))
            p = intensity[:, :, sl, sl].flatten(1)
            t = target[idx][:, :, sl, sl].flatten(1)
            p = p - p.mean(dim=1, keepdim=True)
            t = t - t.mean(dim=1, keepdim=True)
            corr = (p * t).sum(dim=1) / (
                torch.sqrt((p**2).sum(dim=1) * (t**2).sum(dim=1)) + 1e-12
            )
            shaping_loss = torch.mean(1.0 - corr)
            loss = phase_loss + 40.0 * shaping_loss
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            phase_losses.append(float(phase_loss.detach()))
            shaping_losses.append(float(shaping_loss.detach()))
        row = {
            "epoch": epoch + 1,
            "phase_loss": float(np.mean(phase_losses)) if phase_losses else float("nan"),
            "shaping_loss": (
                float(np.mean(shaping_losses)) if shaping_losses else float("nan")
            ),
        }
        history.append(row)
        if verbose:
            print(
                f"epoch {epoch + 1}/{epochs}  phase={row['phase_loss']:.4f}  "
                f"shaping={row['shaping_loss']:.4f}",
                flush=True,
            )
    return history


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True, help="预训练 checkpoint")
    parser.add_argument(
        "--from-dir",
        type=Path,
        default=None,
        help="一段已有的 model-in-loop 运行目录（离线迁移）",
    )
    parser.add_argument(
        "--hardware", action="store_true", help="在实验系统上现场采一批数据再微调"
    )
    parser.add_argument("--capture", type=int, default=32, help="现场采集的探针数")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--cam-type", choices=["daheng", "miicam"], default="daheng")
    parser.add_argument("--cam-id", type=int, default=0)
    parser.add_argument("--slm-number", type=int, default=1)
    parser.add_argument("--slm-wavelength", type=int, default=1064)
    parser.add_argument("--region", type=int, default=128)
    parser.add_argument("--far-field-padding", type=int, default=8)
    parser.add_argument("--cam-size", type=int, default=300)
    parser.add_argument("--target-side", type=int, default=100)
    parser.add_argument("--exposure-time-ms", default="auto")
    parser.add_argument("--max-exposure-ms", type=float, default=0.4)
    args = parser.parse_args(argv)

    if (args.from_dir is None) == (not args.hardware):
        parser.error("exactly one of --from-dir / --hardware is required")

    out_dir = args.out or (
        ROOT / "data" / "debug" / "gsnet_transfer" / time.strftime("%Y%m%d_%H%M%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.hardware:
        dataset = _capture_live(args, out_dir)
    else:
        dataset = _load_probe_dataset(args.from_dir)

    geometry = _geometry_from_arrays(dataset)
    propagator = Bench2FPropagator(geometry, n_max=10)
    operator = Bench2FOperator(propagator)

    model = load_gsnet_checkpoint(args.checkpoint, forward_op=operator)
    history = finetune(
        model,
        operator,
        dataset,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    # 导出：网络在一张「目标帧」上的预测相位，可直接喂给 --init-from-gsnet
    model.eval()
    with torch.no_grad():
        source = torch.from_numpy(dataset["source"][:1])[:, None]
        target = torch.from_numpy(dataset["target"][:1])[:, None]
        phase = model(source, target)[0, 0].numpy().astype(np.float64)
    np.save(out_dir / "phase.npy", phase)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "num_layers": model.num_layers,
                "base_channels": model.base_channels,
                "shared_weights": model.shared_weights,
            },
        },
        out_dir / "checkpoint.pt",
    )
    final_phase_loss = history[-1]["phase_loss"] if history else float("nan")
    chance = float(np.pi**2 / 3.0)
    summary = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        }
        | {"out": str(out_dir)},
        "n_samples": int(dataset["source"].shape[0]),
        "region": geometry.region,
        "far_field_size": geometry.far_field_size,
        "history": history,
        "phase": str(out_dir / "phase.npy"),
        # 相位项的**随机猜测水平**：相位差在 (-π, π] 上均匀分布时，
        # 圆周 MSE 的期望是 π²/3。没有这个参照，读者分不清「学到了」和
        # 「还在瞎猜」。
        "phase_loss_chance_level": chance,
        "phase_loss_below_chance": bool(
            (history[-1]["phase_loss"] if history else chance) < chance
        ),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"迁移学习完成：{out_dir}  "
        f"(相位项 {final_phase_loss:.4f}，随机猜测水平 {chance:.4f}，"
        f"{'低于' if summary['phase_loss_below_chance'] else '尚未低于'})"
    )
    print(f"  相位：{out_dir / 'phase.npy'}（{geometry.region}×{geometry.region}，弧度）")
    print(
        "  交给闭环：python src/ao_shaping/main.py --debug slm-model-in-loop "
        f"--init-from-gsnet {out_dir / 'phase.npy'} ..."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
