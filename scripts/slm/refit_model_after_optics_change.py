"""光路变化后的模型重拟合：用新采集的探针数据重拟冻结前向，并换用重建的采样窗。

关键点（都是 2026-10-23 实测换来的教训）：

* **采样窗必须按本次实测零点重建**（旧窗心偏了 80 px），否则拟合在错的区域上做；
* **重拟合的 fit loss 不能当"模型变好"的判据** —— 同一天 4/4 轮重拟合把 loss 从
  0.93 降到 0.36，却没带来任何实测改善。本脚本因此只负责产出新检查点，是否可用
  必须由随后的 **B± 方向检验** 决定；
* 侧向几何（``ramp_matrix``）不随 CCD 沿轴移动而变，所以沿用旧上游链；
  只有离焦项 ``defocus_diopter`` 应显著变小（光路到焦面后 ≈0）。
"""

from __future__ import annotations

import argparse
from loguru import logger
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import torch

torch.set_num_threads(4)

from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
    load_frozen_parameters,
)


def load_pairs(dataset: Path, run_dir: Path | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """读 (相位 121x121?, 实测帧)。数据集用 ``sample_*.npz`` 的 ``panel_phase`` + ``frame``。"""
    panels, frames = [], []
    for path in sorted(dataset.glob("sample_*.npz")):
        with np.load(path) as payload:
            panels.append(payload["panel_phase"].astype(np.float64))
            frames.append(payload["frame"].astype(np.float64))
    if not panels:
        raise SystemExit(f"{dataset} 里没有 sample_*.npz")
    return np.stack(panels), np.stack(frames), np.array(frames[0].shape)


def restore_frozen(model, snapshots: dict) -> None:
    """把冻结参数的数值还原到快照（每步优化后调用）。

    为什么用"还原"而不是 `requires_grad=False`：`PhysicalForwardFit` 的 `d` /
    `center` / `log_width` / `residual` 是在 `__init__` 里建的可训练张量，没有暴露
    逐参数开关；在优化循环里每步把数值拷回快照，等价于参数级冻结，且不必改动
    `fit_defocus_forward_dataset` 的公共语义。
    """
    import torch

    for name, value in snapshots.items():
        target = getattr(model, name)
        with torch.no_grad():
            target.data.copy_(value)


def snapshot_parameters(model) -> dict:
    """快照所有可冻结参数（小步重拟合用来证明"真的没动"）。"""
    return {
        name: getattr(model, name).detach().clone()
        for name in ("d", "origin", "center", "log_width", "residual")
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="新采集目录（含 sample_*.npz）")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="初始检查点；**必须显式给**（历史配置不可信，不设默认值）")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=150)
    parser.add_argument("--window-half-span-px", type=float, default=200.0)
    parser.add_argument("--step-px", type=float, default=2.0)
    parser.add_argument("--spot-xy", type=float, nargs=2, default=None)
    parser.add_argument("--freeze-defocus", action="store_true",
                        help="冻结离焦 d（全量重拟合会把梯度弄失效，见模块 docstring）")
    parser.add_argument("--freeze-amplitude", action="store_true",
                        help="冻结照明幅度中心/宽度")
    parser.add_argument("--freeze-residual", action="store_true",
                        help="冻结面板残差 Noll 系数")
    parser.add_argument("--small-step", action="store_true",
                        help="便捷开关 = 同时打开上面三个 ⇒ 只有 origin 能动")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.out.mkdir(parents=True, exist_ok=False)

    from scripts.slm.diagnose_defocus_sampling import fixed_panel_basis
    from scripts.slm.fit_defocus_forward_dataset import PhysicalForwardFit

    from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import recentre_window

    panels, frames, frame_shape = load_pairs(args.dataset, None)
    old = load_frozen_parameters(args.checkpoint)
    size = old.grid_size
    pitch = 1920 // size
    rows = panels[0].shape[0] // pitch

    # 128 网格相位（块均值，与模型 pitch 一致）
    phases = np.zeros((len(panels), size, size))
    for index, panel in enumerate(panels):
        block = panel[: rows * pitch].reshape(rows, pitch, size, pitch).mean(axis=(1, 3))
        phases[index, :block.shape[0]] = block

    spot = (float(args.spot_xy[0]), float(args.spot_xy[1])) if args.spot_xy else None
    if spot is None:
        peak = frames[0]
        row, col = np.unravel_index(int(np.argmax(peak)), peak.shape)
        spot = (float(col), float(row))
    xy = recentre_window(spot, half_span_px=args.window_half_span_px, step_px=args.step_px)
    ys = np.clip(xy[..., 1], 0, frame_shape[0] - 1).astype(int)
    xs = np.clip(xy[..., 0], 0, frame_shape[1] - 1).astype(int)
    images = np.stack([frame[ys, xs] for frame in frames])

    freeze_defocus = bool(args.freeze_defocus or args.small_step)
    freeze_residual = bool(args.freeze_residual or args.small_step)
    learn_amplitude = not bool(args.freeze_amplitude or args.small_step)

    model = PhysicalForwardFit(phases, images, xy, np.asarray(old.ramp_matrix), 0.0,
                               np.ones(len(phases), bool), old.amplitude_center_xy,
                               learn_amplitude=learn_amplitude, initial_defocus=old.defocus_diopter,
                               learn_residual=not freeze_residual, fft_size=old.fft_size,
                               pixel_samples=old.pixel_samples,
                               basis_arrays=fixed_panel_basis(size), batch_size=len(phases))
    before = snapshot_parameters(model)
    frozen_names = {
        "d": freeze_defocus,
        "center": not learn_amplitude,
        "log_width": not learn_amplitude,
        "residual": freeze_residual,
    }
    frozen = {name: before[name] for name, is_frozen in frozen_names.items() if is_frozen}

    losses = []
    for _ in range(args.iterations):
        losses.append(float(model.update()))
        if frozen:
            restore_frozen(model, frozen)
    residual = {int(k): float(v) for k, v in zip(model.residual_modes, model.residual.detach().tolist())}

    after = snapshot_parameters(model)
    deltas = {
        "defocus_diopter": float(after["d"] - before["d"]),
        "origin": [float(v) for v in (after["origin"] - before["origin"])],
        "amplitude_center": [float(v) for v in (after["center"] - before["center"])],
        "amplitude_width": [float(v) for v in (after["log_width"].exp() - before["log_width"].exp())],
        "residual_max_abs": float((after["residual"] - before["residual"]).abs().max()),
    }

    parameters = dict(defocus_diopter=float(model.d.detach()),
                      origin=[float(v) for v in model.origin.detach()],
                      amplitude_center=[float(v) for v in model.center.detach()],
                      amplitude_width=[float(v) for v in model.log_width.exp().detach()],
                      residual_noll_radians=residual,
                      gain=old.gain, background=old.background, sigma_ccd_px=old.sigma_ccd_px)
    (args.out / "result.json").write_text(json.dumps(dict(parameters=parameters, training_score=losses[-1],
                                                          refit=dict(iterations=args.iterations,
                                                                     first_loss=losses[0],
                                                                     last_loss=losses[-1])),
                                                     indent=2, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(args.out / "inputs.npz", phases=phases, measured=images, camera_xy=xy)
    source = json.loads((args.checkpoint / "config.json").read_text(encoding="utf-8"))["source"]
    (args.out / "config.json").write_text(json.dumps(dict(source=source), indent=2), encoding="utf-8")
    record = dict(
        recorded_utc=datetime.now(timezone.utc).isoformat(timespec="microseconds"),
        dataset=str(args.dataset), spot_xy=list(spot), window_half_span_px=args.window_half_span_px,
        window_step_px=args.step_px, samples=int(len(phases)), iterations=args.iterations,
        first_loss=losses[0], last_loss=losses[-1], parameters=parameters,
        old_diopter=old.defocus_diopter, new_diopter=parameters["defocus_diopter"],
        implied_focal_mm=1000.0 / abs(parameters["defocus_diopter"]) if parameters["defocus_diopter"] else None,
        dataset_sha256=hashlib.sha256((args.dataset / "manifest.json").read_bytes()).hexdigest()
        if (args.dataset / "manifest.json").exists() else None,
    )
    (args.out / "refit.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(json.dumps({k: v for k, v in record.items() if k != "parameters"}, indent=2, ensure_ascii=False))
    logger.info("\n⚠ 重拟合的 loss 下降**不是**可用性证据；下一步必须做 B± 方向检验再用于优化。")


if __name__ == "__main__":
    main()





