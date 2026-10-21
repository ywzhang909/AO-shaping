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

离线 + 只解冻部分参数组（``--freeze``，见下方「冻结策略」）::

    python scripts/slm/gsnet_transfer.py \
        --checkpoint data/gsnet_pretrain/<ts>/checkpoint.pt \
        --from-dir data/debug/slm_model_in_loop/<ts> \
        --epochs 20 --lr 1e-4 --freeze freeze-encoder

离线 + 标定过的前向算子（``--twin-config``，见下方「标定算子与验证门」）::

    python scripts/slm/gsnet_transfer.py \
        --checkpoint data/gsnet_pretrain/<ts>/checkpoint.pt \
        --from-dir data/debug/slm_model_in_loop/<ts> \
        --epochs 20 --lr 1e-4 \
        --twin-config data/debug/bench_twin_fit/<ts>/fit.json

离线 + 验证门（``--gate``，见下方「标定算子与验证门」）::

    python scripts/slm/gsnet_transfer.py \
        --checkpoint data/gsnet_pretrain/<ts>/checkpoint.pt \
        --from-dir data/debug/slm_model_in_loop/<ts> \
        --epochs 20 --lr 1e-4 --gate

在机（现场采一批探针再微调）::

    python scripts/slm/gsnet_transfer.py \
        --checkpoint data/gsnet_pretrain/<ts>/checkpoint.pt \
        --hardware --capture 32 \
        --cam-type daheng --cam-id 0 --slm-number 1 --slm-wavelength 1064 \
        --region 128 --far-field-padding 8 --exposure-time-ms auto

冻结策略（``--freeze``）
------------------------
迁移数据只有实验系统上采的少量探针帧（每帧都要一次曝光），而预训练来自
大规模仿真。因此**真正要从这点数据里重新学的参数量，就是能否学会的约束
本身**：``--freeze`` 选择冻结哪个参数组（encoder / decoder / films /
outconv 的预设组合），只让必要的部分动。默认 ``full`` ＝不冻结、全部训练，
与旧行为逐位一致；``frozen`` ＝全部冻结（不微调，只走前向导出）。

⚠️ 被冻结组的 BatchNorm running statistics 必须**保持**冻结：裸
``model.train()`` 会把它们放回训练模式，冻结名存实亡。所以训练循环每个
epoch 调用 ``ml.gsnet.freezing.prepare_for_training``（先 ``train()`` 再
重新施加冻结），而不是 ``model.train()``。哪种冻结策略真正有效由专门的
对比实验测量（``scripts/slm/gsnet_aberration_transfer.py``），本脚本只
提供开关，不预设结论。

标定算子与验证门（``--twin-config`` / ``--gate``）
--------------------------------------------------
整形项是**经由本脚本构造的前向算子**评估的，而实测帧来自实验系统。默认
（不给 ``--twin-config``）算子是**理想孪生**（无相差、无灰度→振幅耦合），
它复现不出带相差实验系统测到的帧 —— 整形项的相关目标于是**物理上不可达**，
梯度近似噪声。仿真实测（驱动 ``scripts/slm/gsnet_aberration_transfer.py``，
结果 ``report/gsnet_aberration_transfer/report.md``）：6 个像差情形里默认
配置让 5 个**变差**（最差 composite 0.3401 → 0.2738，低于平场下限 0.2911）。
``--twin-config`` 指向 ``tools/slm/bench_twin_calibrate.py`` 的 ``save_fit``
产物（在实验系统上反演出的孪生参数 JSON），整形项与网络内部前向改走这台
**标定过的**算子，整形项才变成可实现的目标（``export_gsnet_phase.py`` 的
``--twin-config`` 是同一约定）。孪生的几何必须与数据一致（``region`` ＝
探针相位边长、算子输出边长 ＝ 目标帧边长），不符时脚本报明确错误，而不是
静默产出尺寸错误的远场。

``--gate`` 是另一半：**没有这道门，一次有害的微调总会被采纳。** 开启后微调
结束时做一次对比择优 —— 在**验证集**探针（探针记录的**最后** 25%、至少 2 帧，
与训练探针不相交、划分确定性）上，用微调所优化的**同一把尺子**（整形项的
同一 Pearson 相关、同一中心裁剪，越高越好）比较微调后与未微调的权重，**只有
严格更好才保留**微调权重；否则恢复未微调权重。对应 model-in-loop 闭环的门 3
（验证集 Pearson 验收）。仿真实测：门把最坏情形的损害从 −0.0837 压到
−0.0325 —— 验证集规模小，门**限制损害但不消除损害**。判定记入 summary.json
的 ``gate``（``"off"`` / ``"accepted"`` / ``"kept_pretrained"``）；判给预训练
时，导出的 ``phase.npy`` 与 ``checkpoint.pt`` 都是**未微调**的那套权重。
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

from ao_shaping.drivers.sim.bench_twin import (  # noqa: E402
    BenchTwin,
    BenchTwinConfig,
    TwinErrorModel,
)
from ml.gsnet.forward_ops import BenchTwinOperator  # noqa: E402
from ml.gsnet.freezing import (  # noqa: E402
    FREEZE_SPEC_NAMES,
    FreezeSpec,
    group_parameter_counts,
    prepare_for_training,
)
from ml.gsnet.hybrid_infer import load_gsnet_checkpoint  # noqa: E402
from ml.gsnet.model import FourierGSNet  # noqa: E402

#: 整形项（及 ``--gate`` 同一把尺子）在远场上的中心裁剪边长。
_METRIC_SIDE = 36


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
    """在实验系统上采一批 (相位, 帧) 对。

    **不自己重写采集**：直接跑一次仓库自己的 model-in-loop（``--debug``），
    它会写出 ``probes/*.npz``（命令过的 ``phase`` ＝标签、映射到远场模型网格的
    ``model_grid`` ＝目标）与 ``source_amplitude.npy``（网络的 source）。
    复用同一条链路的好处是：采到的数据与闭环真正看到的一致，设备生命周期、
    曝光标定、几何求解全走仓库已经验证过的代码。
    """
    from ao_shaping.optimizer.wfless.slm_model_in_loop import (
        SlmModelInLoopConfig,
        optimize_slm_model_in_loop,
    )

    run_dir = out_dir / "mil_capture"
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
        n_rounds=args.capture_rounds,
        probe_count=args.capture_probes,
        holdout_probe_count=0,
        step_a_iterations=0,
        step_b_iterations=0,
        debug_dir=str(run_dir),
    )
    result = optimize_slm_model_in_loop(config)
    if not result.probe_records:
        raise RuntimeError(
            "the model-in-loop capture produced no probe records — nothing to "
            "transfer on"
        )
    return _load_probe_dataset(run_dir, limit=None)


def _geometry_from_arrays(dataset: dict[str, np.ndarray]) -> BenchTwin:
    """按数据实际的网格建孪生（瞳面 = 标签的边长，远场 = 目标的边长）。"""
    region = int(dataset["label"].shape[-1])
    far_field = int(dataset["target"].shape[-1])
    return BenchTwin(
        BenchTwinConfig(
            region=region,
            region_span_px=900.0,
            far_field_padding=max(1, far_field // max(1, region)),
            beam_w0_px=250.0,
            peak_adu=100.0,
            noise_adu=0.0,
            seed=0,
            errors=TwinErrorModel(),
        )
    )


def _twin_from_fit(
    path: Path, dataset: dict[str, np.ndarray]
) -> tuple[BenchTwin, Any]:
    """从实验系统上反演出的孪生 JSON（``bench_twin_calibrate.save_fit`` 的产物）
    重建前向算子用的孪生，并核验几何与数据一致。

    ``load_fit`` 返回 ``TwinFitResult``，其 ``.config`` 是完整的
    ``BenchTwinConfig``（几何 + 标定误差），所以误差与几何都取自**那台拟合
    出来的孪生**，不再回落理想模型。几何仍须对上数据：孪生的 ``region`` ＝
    探针相位（标签）边长，算子实际输出的远场边长（孪生的中心裁剪
    ``crop_side``）＝ 目标帧边长；不符时抛明确错误（多半是孪生在别的
    ``region`` / ``far_field_padding`` 下拟合的），而不是静默产出尺寸错误的
    远场。

    Returns:
        ``(twin, fit)``：给 ``BenchTwinOperator`` 的孪生，与原始
        ``TwinFitResult``（summary.json 要记录它的标定参数）。
    """
    from ao_shaping.tools.slm.bench_twin_calibrate import load_fit  # 延迟导入：默认路径不需要 scipy

    fit = load_fit(path)
    twin = BenchTwin(fit.config)
    region_data = int(dataset["label"].shape[-1])
    far_data = int(dataset["target"].shape[-1])
    if twin.region != region_data:
        raise ValueError(
            f"--twin-config twin region={twin.region} != probe phase side "
            f"{region_data}: the fitted twin was calibrated on a different pupil "
            f"grid than this data; refit it on the current bench with "
            f"region={region_data}"
        )
    far_twin = int(twin.crop_side)
    if far_twin != far_data:
        raise ValueError(
            f"--twin-config twin far-field side={far_twin} (region={twin.region}, "
            f"far_field_padding={twin.config.far_field_padding}, "
            f"crop_px={twin.config.crop_px}; crop_px=None means the crop defaults "
            f"to region) != target frame side {far_data}: the fitted twin was "
            f"calibrated at a different far-field geometry than this data; refit "
            f"with region={region_data}, far_field_padding="
            f"{max(1, far_data // max(1, region_data))} and crop_px={far_data}"
        )
    return twin, fit


def _shaping_corr(intensity: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """中心裁剪上的 Pearson 相关 — 微调整形项与 ``--gate`` **共用这一把尺子**。

    裁剪位置按 ``intensity``（算子输出）的边长决定并同时套用到 ``target`` 上
    —— 这是 ``finetune`` 原有行为的逐字保留（两者边长可以不同，裁剪以算子
    输出为准），门因此裁决的正是被优化的那件事。
    """
    side = min(_METRIC_SIDE, intensity.shape[-1])
    half = (intensity.shape[-1] - side) // 2
    sl = slice(half, half + side)
    p = intensity[:, :, sl, sl].flatten(1)
    t = target[:, :, sl, sl].flatten(1)
    p = p - p.mean(dim=1, keepdim=True)
    t = t - t.mean(dim=1, keepdim=True)
    return (p * t).sum(dim=1) / (
        torch.sqrt((p**2).sum(dim=1) * (t**2).sum(dim=1)) + 1e-12
    )


def _gate_shaping_score(
    model: FourierGSNet,
    operator: BenchTwinOperator,
    dataset: dict[str, np.ndarray],
) -> float:
    """``--gate`` 的判据：验证集探针上整形项 Pearson 相关的均值（越高越好）。

    与微调整形项是**同一实现**（``_shaping_corr``）、同一算子；模型先
    ``eval()`` —— 门裁决的是会被部署的推理形态（BatchNorm 用 running
    statistics）。
    """
    model.eval()
    source = torch.from_numpy(dataset["source"])[:, None]
    target = torch.from_numpy(dataset["target"])[:, None]
    source_amp = torch.sqrt(source.clamp_min(0.0) + 1e-12)
    with torch.no_grad():
        pred = model(source, target)
        far = operator.forward(source_amp, pred)
        corr = _shaping_corr(far.real**2 + far.imag**2, target)
    return float(corr.mean())


def _split_for_gate(
    dataset: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """确定性的验证划分：留出探针记录的**最后** 25%（至少 2 帧），其余训练。

    记录顺序就是 ``_load_probe_dataset`` 的 ``sorted(glob)`` 顺序，所以划分
    只依赖数据本身。训练与验证不相交。

    Raises:
        ValueError: 探针太少，留出 ≥2 帧后训练集就空了（至少需要 3 条记录）。
    """
    n = int(dataset["source"].shape[0])
    n_val = max(2, int(np.ceil(0.25 * n)))
    if n_val >= n:
        raise ValueError(
            f"--gate needs at least 3 probe records to hold out >= 2 validation "
            f"frames while keeping a non-empty training split (got {n})"
        )
    train_dataset = {key: value[: n - n_val] for key, value in dataset.items()}
    val_dataset = {key: value[n - n_val:] for key, value in dataset.items()}
    return train_dataset, val_dataset


def finetune(
    model: FourierGSNet,
    operator: BenchTwinOperator,
    dataset: dict[str, np.ndarray],
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
    verbose: bool = True,
    *,
    freeze_spec: FreezeSpec = FreezeSpec.from_name("full"),
) -> list[dict[str, float]]:
    """在真实数据上微调。

    相位项把网络拉向**命令过的相位**；整形项把网络预测的相位经**标定前向**
    推到**相机看到的那张帧**上。

    Args:
        freeze_spec: 哪些参数组保持冻结（默认 ``full``＝全部训练，与旧行为
            一致）。每个 epoch 都经 ``prepare_for_training`` 重新施加冻结，
            保证被冻结组的 BatchNorm 统计不被 ``train()`` 放回训练模式。

    Returns:
        每 epoch 一行的损失历史；若全部参数被冻结（无任何可训练参数）则跳过
        优化并返回**空历史**。
    """
    torch.manual_seed(seed)
    prepare_for_training(model, freeze_spec)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        if verbose:
            print(
                "没有任何可训练参数（--freeze frozen：四个参数组全部冻结），"
                "跳过微调优化循环，直接按预训练权重导出。",
                flush=True,
            )
        return []
    optimiser = torch.optim.Adam(trainable_params, lr=lr)
    source = torch.from_numpy(dataset["source"])[:, None]
    target = torch.from_numpy(dataset["target"])[:, None]
    label = torch.from_numpy(dataset["label"])[:, None]
    source_amp = torch.sqrt(source.clamp_min(0.0) + 1e-12)
    n = source.shape[0]
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        prepare_for_training(model, freeze_spec)
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
            corr = _shaping_corr(intensity, target[idx])
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
    parser.add_argument(
        "--capture-rounds", type=int, default=2, help="现场采集用的闭环轮数"
    )
    parser.add_argument(
        "--capture-probes", type=int, default=8, help="每轮的探针数"
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument(
        "--freeze",
        choices=list(FREEZE_SPEC_NAMES),
        default="full",
        help=(
            "微调时保持冻结的参数组预设（encoder / decoder / films / outconv "
            "的组合）；默认 full ＝不冻结、全部训练，与旧行为一致"
        ),
    )
    parser.add_argument(
        "--twin-config",
        type=Path,
        default=None,
        help=(
            "在实验系统上反演出来的孪生参数 JSON"
            "（tools/slm/bench_twin_calibrate.py 的 save_fit 产物）；给了就"
            "用这台标定孪生（误差＋几何）构造整形项与网络内部前向的算子，"
            "替代默认的理想孪生；几何与数据不符时报错"
        ),
    )
    parser.add_argument(
        "--gate",
        action="store_true",
        help=(
            "微调后做对比择优（门 3）：在验证集探针（探针记录的最后 25%%、"
            "至少 2 帧，与训练探针不相交）上用整形项的同一把尺子比较微调后"
            "与未微调的权重，只有严格更好才保留微调权重，否则恢复未微调权重"
            "（导出的 phase/checkpoint 也随之是未微调的）"
        ),
    )
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

    try:
        freeze_spec = FreezeSpec.from_name(args.freeze)
    except ValueError as exc:  # argparse choices 已拦截，此处兜底保证不出 traceback
        parser.error(str(exc))

    out_dir = args.out or (
        ROOT / "data" / "debug" / "gsnet_transfer" / time.strftime("%Y%m%d_%H%M%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.hardware:
        dataset = _capture_live(args, out_dir)
    else:
        dataset = _load_probe_dataset(args.from_dir)

    fit = None
    if args.twin_config is not None:
        twin, fit = _twin_from_fit(args.twin_config, dataset)
    else:
        twin = _geometry_from_arrays(dataset)
    operator = BenchTwinOperator(twin)

    model = load_gsnet_checkpoint(args.checkpoint, forward_op=operator)
    # 先施加冻结并计数：summary 要报告这次运行到底要重新学多少参数。
    # finetune 内部每个 epoch 会经 prepare_for_training 重新施加同一冻结。
    freeze_counts = prepare_for_training(model, freeze_spec)
    group_counts = group_parameter_counts(model)

    train_dataset = dataset
    val_dataset: dict[str, np.ndarray] | None = None
    if args.gate:
        train_dataset, val_dataset = _split_for_gate(dataset)

    gate = "off"
    gate_pre: float | None = None
    gate_post: float | None = None
    pretrained_state: dict[str, torch.Tensor] | None = None
    if args.gate:
        # 未微调权重先留一份副本：它是门的参照，也是 kept_pretrained 时
        # 恢复的对象（load_state_dict 连同 BatchNorm running statistics 一起
        # 还原，部署形态逐位回到 checkpoint）。
        pretrained_state = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
        }
        gate_pre = _gate_shaping_score(model, operator, val_dataset)

    history = finetune(
        model,
        operator,
        train_dataset,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
        freeze_spec=freeze_spec,
    )

    if args.gate:
        gate_post = _gate_shaping_score(model, operator, val_dataset)
        # 门 3（对比择优）：候选只有在验证集上把同一把尺子做得**严格更好**才
        # 保留；否则恢复未微调权重 —— 没有这道门时，有害的微调总会被采纳。
        gate = "accepted" if gate_post > gate_pre else "kept_pretrained"
        if gate == "kept_pretrained":
            model.load_state_dict(pretrained_state)

    # 导出：网络在一张「目标帧」上的预测相位，可直接喂给 --init-from-gsnet。
    # 走到这里 model 里已经是**会被部署**的那套权重（gate 判给预训练时已恢复）。
    model.eval()
    with torch.no_grad():
        source = torch.from_numpy(train_dataset["source"][:1])[:, None]
        target = torch.from_numpy(train_dataset["target"][:1])[:, None]
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
        "region": int(dataset["label"].shape[-1]),
        "far_field_size": int(dataset["target"].shape[-1]),
        "gate": gate,
        "gate_split": (
            {
                "train": int(train_dataset["source"].shape[0]),
                "val": int(dataset["source"].shape[0])
                - int(train_dataset["source"].shape[0]),
            }
            if args.gate
            else None
        ),
        "gate_shaping_pearson": (
            {"pretrained": gate_pre, "candidate": gate_post} if args.gate else None
        ),
        # 最终会被部署的是哪套权重（gate 判给预训练时导出物也是未微调的）。
        "deployed_weights": "pretrained" if gate == "kept_pretrained" else "finetuned",
        "trainable_params": freeze_counts["trainable"],
        "total_params": freeze_counts["total"],
        "frozen_params": freeze_counts["frozen"],
        "group_parameter_counts": group_counts,
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
    if fit is not None:
        # 用了哪台标定孪生、它的几何与关键误差参数（取自 TwinFitResult.config，
        # 不重读 JSON）：一次运行要能自描述。
        errors = fit.config.errors
        summary["twin"] = {
            "path": str(args.twin_config),
            "fit_loss": float(fit.loss),
            "region": int(fit.config.region),
            "far_field_padding": int(fit.config.far_field_padding),
            "crop_px": fit.config.crop_px,
            "gray_levels": int(errors.gray_levels),
            "lut_gamma": float(errors.lut_gamma),
            "lut_phase_max_rad": float(errors.lut_phase_max_rad),
            "slm_amplitude_coupling": float(errors.slm_amplitude_coupling),
            "slm_amplitude_coupling_phase_rad": float(
                errors.slm_amplitude_coupling_phase_rad
            ),
            "slm_zernike_modes": len(errors.slm_zernike),
            "lens_zernike_modes": len(errors.lens_zernike),
            "defocus_m": float(errors.defocus_m),
            "tilt_x_rad": float(errors.tilt_x_rad),
            "tilt_y_rad": float(errors.tilt_y_rad),
            "slm_roughness_sigma_rad": float(errors.slm_roughness_sigma_rad),
        }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"迁移学习完成：{out_dir}  "
        f"(相位项 {final_phase_loss:.4f}，随机猜测水平 {chance:.4f}，"
        f"{'低于' if summary['phase_loss_below_chance'] else '尚未低于'})"
    )
    if args.gate:
        print(
            f"  门（对比择优，验证集 {summary['gate_split']['val']} 帧）："
            f"整形 Pearson 未微调 {gate_pre:.4f} → 微调后 {gate_post:.4f} ⇒ {gate}"
            + (
                "（已恢复未微调权重：导出的 phase/checkpoint 都是未微调的）"
                if gate == "kept_pretrained"
                else ""
            )
        )
    if fit is not None:
        print(f"  前向算子：标定孪生（{args.twin_config}）")
    region_side = int(dataset["label"].shape[-1])
    print(f"  相位：{out_dir / 'phase.npy'}（{region_side}×{region_side}，弧度）")
    print(
        "  交给闭环：python src/ao_shaping/main.py --debug slm-model-in-loop "
        f"--init-from-gsnet {out_dir / 'phase.npy'} ..."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
