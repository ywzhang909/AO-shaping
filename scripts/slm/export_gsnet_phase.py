"""把 FourierGSNet 的预测相位导出成 ``--init-from-gsnet`` 能吃的 ``.npy``。

issue #165 的 P1 实验系统路径需要一个文件：**瞳面模型网格**上的相位
（``(region, region)``，原始弧度，未包裹），交给
``slm-model-in-loop --init-from-gsnet`` 当 Step B 热启动。

本脚本走离线链路（``ml.gsnet.hybrid_infer.load_gsnet_checkpoint`` + 一个
可插拔前向算子），不打开任何设备。

用法::

    python scripts/slm/export_gsnet_phase.py \
        --checkpoint data/gsnet_train/run-*/fourier_gsnet_best.pt \
        --source data/debug/gsnet_init/20261021/source.npy \
        --target data/debug/gsnet_init/20261021/target.npy \
        --out data/debug/gsnet_init/20261021/phase.npy

``--forward`` 选择网络内部用的前向算子：

* ``ideal``（默认）—— 规范居中 FFT，与仓库现有 FourierGSNet 一致；
* ``calibrated`` —— 本台架 2f 物理前向（仓库规范的数字孪生 ``drivers/sim/bench_twin.py`` 的
  torch 镜像，可用 ``--twin-config`` 换上在真实台架上反演出来的参数），
  可用 ``--aberration`` 带上拟合出来的实验系统像差。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ml.gsnet.forward_ops import BenchTwinOperator, IdealFFTOperator  # noqa: E402
from ml.gsnet.hybrid_infer import load_gsnet_checkpoint  # noqa: E402


def _load_grid(path: Path) -> np.ndarray:
    array = np.load(path)
    if array.ndim != 2:
        raise ValueError(f"{path} must be a 2-D grid, got shape {array.shape}")
    return np.asarray(array, dtype=np.float64)


def build_operator(args: argparse.Namespace):
    """按 ``--forward`` 构造网络内部用的前向算子。

    ``calibrated`` 用**仓库规范的 2f 数字孪生**（``drivers/sim/bench_twin.py``）
    的 torch 镜像；``--twin-config`` 可以指向一个反演出来的孪生参数 JSON
    （``tools/slm/bench_twin_calibrate.save_fit`` 的产物），于是网络内部的前向
    用的是**在真实台架上拟合过**的那台实验系统。
    """
    if args.forward == "ideal":
        return IdealFFTOperator(args.grid, convention="canonical")
    from ao_shaping.drivers.sim.bench_twin import (
        BenchTwin,
        BenchTwinConfig,
        TwinErrorModel,
    )

    errors = TwinErrorModel()
    if args.twin_config is not None:
        import json

        payload = json.loads(args.twin_config.read_text(encoding="utf-8"))
        fitted = payload.get("config", payload)
        errors = TwinErrorModel(
            **{
                key: value
                for key, value in fitted.get("errors", {}).items()
                if key in TwinErrorModel.__dataclass_fields__
            }
        )
    config = BenchTwinConfig(
        region=args.grid,
        region_span_px=args.region_span_px,
        far_field_padding=args.far_field_padding,
        beam_w0_px=250.0,
        peak_adu=100.0,
        noise_adu=0.0,
        seed=0,
        errors=errors,
    )
    return BenchTwinOperator(BenchTwin(config))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True, help="源面强度 .npy")
    parser.add_argument("--target", type=Path, required=True, help="目标远场强度 .npy")
    parser.add_argument("--out", type=Path, required=True, help="输出相位 .npy")
    parser.add_argument(
        "--twin-config",
        type=Path,
        default=None,
        help="反演出来的孪生参数 JSON（bench_twin_calibrate.save_fit 的产物）",
    )
    parser.add_argument("--region-span-px", type=float, default=900.0)
    parser.add_argument("--far-field-padding", type=int, default=8)
    parser.add_argument(
        "--forward",
        choices=["ideal", "calibrated"],
        default="ideal",
        help="网络内部用的前向算子",
    )
    parser.add_argument(
        "--aberration",
        type=Path,
        default=None,
        help="拟合出来的实验系统像差 .npy（仅 --forward calibrated 用）",
    )
    parser.add_argument("--grid", type=int, default=64, help="瞳面模型网格边长")
    args = parser.parse_args(argv)

    source = _load_grid(args.source)
    target = _load_grid(args.target)
    if source.shape != target.shape:
        raise ValueError(
            f"source {source.shape} and target {target.shape} must have the same shape"
        )
    if source.shape != (args.grid, args.grid):
        raise ValueError(
            f"source/target must be ({args.grid}, {args.grid}) on this grid, "
            f"got {source.shape}"
        )

    operator = build_operator(args)
    model = load_gsnet_checkpoint(args.checkpoint, forward_op=operator)

    source_t = torch.from_numpy(source).to(torch.float32)[None, None]
    target_t = torch.from_numpy(target).to(torch.float32)[None, None]
    with torch.no_grad():
        phase = model(source_t, target_t)

    out = phase[0, 0].numpy().astype(np.float64)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, out)

    summary: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "source": str(args.source),
        "target": str(args.target),
        "forward": args.forward,
        "grid": args.grid,
        "phase_rms_rad": float(np.sqrt((out**2).mean())),
        "phase_max_abs_rad": float(np.abs(out).max()),
        "out": str(args.out),
    }
    args.out.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(
        "\n下一步（实验系统）:\n"
        f"  python src/ao_shaping/main.py --debug slm-model-in-loop "
        f"--init-from-gsnet {args.out} ...\n"
        "⚠️ 仿真已实测：信任域门（L2 ≤ 0.5 rad）会挡掉 rms 1.7 rad 量级的"
        "真实整形相位，见 report/gsnet_2f_calibration/report.md §5.2 —— "
        "上机前先决定门 1 怎么改。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
