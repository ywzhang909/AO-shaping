"""issue #165 的可学习性对照：同一套训练代码，换训练目标的来源。

回答一个具体问题：**FourierGSNet 学不到「过采样真值上的 GS 解」，是训练代码
的问题，还是网络架构的问题？**

协议：同一份数据、同一 seed、同一预算、同一个网络，只换**训练目标**：

* ``same`` —— 目标是在**瞳面网格**（region×region）上跑同尺寸经典 GS 得到的解，
  即旧架构（内部远场是混叠的同尺寸 FFT）能表达的那一类解；
* ``bench`` —— 目标是在**过采样真值**（far_field_size×far_field_size）上跑经典
  GS 得到的解。物理上它才是「好相位」，但它需要网络内部有一个比瞳面网格
  更细的远场才能表达。

两个目标的评价都在**过采样真值**上做，所以差别不是「目标好不好」，
而是「网络能不能学到它」。

**Fully offline** —— 数字孪生，无设备。

用法::

    python scripts/slm/gsnet_learnability_control.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts" / "slm") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts" / "slm"))

import calibrate_gsnet_2f as C  # noqa: E402
from ml.gsnet.forward_ops import BenchTwinOperator  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--grid", type=int, default=64)
    parser.add_argument("--far-field-padding", type=int, default=8)
    parser.add_argument("--region-span-px", type=float, default=900.0)
    parser.add_argument("--noise-adu", type=float, default=2.0)
    parser.add_argument("--target-side", type=int, default=24)
    parser.add_argument("--n-samples", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--eval-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument(
        "--w-shaping",
        type=float,
        default=40.0,
        help="整形项权重；0 = 只用相位回归项（纯监督）",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="输出 JSON（默认 data/debug/gsnet_learnability/<ts>/result.json）",
    )
    args = parser.parse_args(argv)

    region = args.grid
    far_field = region * args.far_field_padding
    twin = C.build_twin(
        args.seed,
        region=region,
        region_span_px=args.region_span_px,
        far_field_padding=args.far_field_padding,
        noise_adu=args.noise_adu,
        randomize=True,
    )
    ideal_twin = C.build_twin(
        args.seed,
        region=region,
        region_span_px=args.region_span_px,
        far_field_padding=args.far_field_padding,
        noise_adu=0.0,
        randomize=False,
    )
    op_bench = BenchTwinOperator(twin)
    op_ideal = BenchTwinOperator(ideal_twin)

    source = torch.from_numpy(
        np.asarray(twin.pupil_amplitude(), dtype=np.float32)
    )[None, None]

    # 两个训练目标：
    # * bench —— 在**带误差的物理**下求的 GS 解（实验系统的非理想因素经由
    #   标签进入网络）；
    # * ideal —— 在**理想物理**下求的 GS 解（网络被要求去追一个忽略实验系统
    #   误差的相位）。
    rng = np.random.default_rng(args.seed)
    targets = []
    gt_bench = []
    gt_ideal = []
    for _ in range(args.n_samples):
        side = float(args.target_side) * float(rng.uniform(0.75, 1.25))
        target = torch.from_numpy(
            C._square_target(side, far_field)
        ).to(torch.float32)[None, None]
        target_amp = torch.sqrt(target.clamp_min(0.0) + 1e-12)
        targets.append(target[0])
        gt_bench.append(
            C._classical_gs(op_bench, target_amp, iterations=20)[0, 0].clone()
        )
        gt_ideal.append(
            C._classical_gs(op_ideal, target_amp, iterations=20)[0, 0].clone()
        )
    target_all = torch.stack(targets)
    gt_bench = torch.stack(gt_bench)[:, None]
    gt_ideal = torch.stack(gt_ideal)[:, None]

    cfg = C.ArmBConfig(
        region=region,
        region_span_px=args.region_span_px,
        far_field_padding=args.far_field_padding,
        n_train=args.n_samples,
        n_eval=args.eval_samples,
        epochs=args.epochs,
        metric_side=36,
        target_side=args.target_side,
        seed=args.seed,
        noise_adu=args.noise_adu,
    )
    operator = op_ideal

    # 整形项权重可调：0 = 纯相位回归（网络只被要求复现目标相位），
    # 40 = 与 ml.gsnet.train 的默认一致。
    original_shaping_loss = C._shaping_loss
    C._shaping_loss = (
        lambda op, sa, ph, tg: original_shaping_loss(op, sa, ph, tg) * 0.0
    ) if args.w_shaping == 0.0 else original_shaping_loss

    rows: dict[str, dict[str, float]] = {}
    for label, gt in (("ideal", gt_ideal), ("bench", gt_bench)):
        model = C._train_arm(
            operator,
            source.expand(gt.shape[0], 1, region, region),
            target_all,
            gt,
            cfg,
        )
        metrics = C._evaluate_network(
            model,
            twin,
            source.expand(args.eval_samples, 1, region, region),
            target_all[: args.eval_samples],
            36,
            args.target_side,
        )
        rows[label] = {
            "composite": metrics["composite"]["mean"],
            "correlation": metrics["correlation"]["mean"],
            "correlation_full": metrics["correlation_full"]["mean"],
            "encircled_energy": metrics["encircled_energy"]["mean"],
            "cv_target_box": metrics["cv_target_box"]["mean"],
        }

    C._shaping_loss = original_shaping_loss
    out = {
        "config": vars(args) | {"out": str(args.out or "")},
        "gt_rms_rad": {
            "bench": float(torch.sqrt((gt_bench**2).mean())),
            "ideal": float(torch.sqrt((gt_ideal**2).mean())),
        },
        "arms": rows,
    }
    out_path = args.out or (
        ROOT / "data" / "debug" / "gsnet_learnability" / time.strftime("%Y%m%d_%H%M%S")
        / "result.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    for label in ("ideal", "bench"):
        row = rows[label]
        print(
            f"gt={label:6} composite={row['composite']:.4f} "
            f"corr={row['correlation']:+.4f} corr_full={row['correlation_full']:+.4f} "
            f"EE={row['encircled_energy']:.4f} cv={row['cv_target_box']:.4f}"
        )
    print(f"产物: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
