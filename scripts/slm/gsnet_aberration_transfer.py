"""issue #165 后续：在仿真里给实验系统**加各种光路相差**，测迁移学习能不能成功，
以及 `FourierGSNet` 的哪些参数组可以**冻结**以减少需要重新学习的参数量。

**纯离线仿真**，不打开任何设备。回答两个问题：

1. **像差扫描**：给 2f 数字孪生注入一条具体的光路相差（面板离焦 / 彗差 /
   球差、透镜像差、相机离焦）后，「预训练 checkpoint → 少量探针微调」这条
   迁移路径在仿真里还能不能成功（迁移后的整形质量是否接近该相差下的
   GS 上限）？
2. **冻结扫描**：`ml.gsnet.freezing` 的四个参数组（encoder / decoder / films /
   outconv）中冻结哪一组，能让微调只动最少的参数而不丢掉迁移？

像差幅度是**故意**超出预训练域随机化范围的（``gsnet_pretrain.py``：面板
Zernike ±0.12 rad、透镜 ±0.10 rad、离焦 ±2e-4 m），本脚本用 0.30 rad / 3e-4 m，
让迁移有一个**真实的域差距**要闭合 —— 若幅度落在预训练分布内，「迁移成功」
不说明任何事。

每个情形的孪生都统一带上 SLM 灰度→振幅耦合（``--coupling``，默认 c=0.15，
灰度域周期取设备的 2π 灰度数 993）：耦合是**迁移目标实验系统的属性**，
消融轴仍然只有像差与冻结配置。

探针的构造方式（对应在机路径）
------------------------------
每个探针相位 = 该相差下的 **GS 标签相位** + 一个**零均值随机 Zernike 微扰**
（Noll 4..12 随机子模式，rms ≈ ``--perturb-rms`` rad）。这正是
``slm_model_in_loop`` 在实验系统上做的事：探针是当前最优相位附近的小幅随机
偏移，所以监督信号是「一个好相位」，不是噪声。

``--transfer-target`` 的语义（必须区分）
----------------------------------------
网络微调时看到的**目标输入**有两种：

* ``measured``（默认，忠实于在机路径 ``gsnet_transfer.py``）：探针经该相差
  孪生**实测**的帧（峰值归一后喂入）；
* ``designed``：**设计**的方斑 —— 与预训练时相同的输入语义。

预训练时网络的目标输入是设计方斑，而在机路径给的是实测帧；这个开关用来
量化「输入语义不一致」在多大程度上导致迁移失败。

评分口径
--------
全部用 ``calibrate_gsnet_2f.frame_metrics``（不重写）。每个留出样本：网络预测
相位 → 该相差孪生实测（探测器 + ``prepare_frame``）→ 与**设计方斑**比。
参照系三条：平场（零相位下限）、该相差下的 GS 解（上限）、微调前的
checkpoint（对每个冻结配置相同，每情形只算一次）。

三个变体开关（默认值 = 上述原行为，数字可复现）
----------------------------------------------
首轮全网格实测：凡是真正训练起来的冻结配置都让网络**变差**（``none`` 情形
跌到 0.2738，低于平场下限 0.2911；``frozen``（0 可训练参数）在 6 个情形里
4 个最好）。据此加入三条候选修法，逐条隔离验证：

* ``--w-phase``：相位项权重（默认 1.0）。监督目标是**命令过的探针相位**
  （= 该情形物理下的 GS 解 + 微扰），而在带耦合的仿真里 GS 解本身比预训练
  网络的输出更差（例：``panel-coma`` GS 上限 0.3113 < 迁移前 0.3216），于是
  相位项是一个**向下拉**的力。置 0 = 只留整形项。
* ``--operator {ideal,calibrated}``（默认 ``ideal``）：微调用的前向算子。
  ``ideal`` = 理想孪生（网络不知道相差，原行为）；``calibrated`` = **探针实测
  所在的那台孪生**（带该情形相差与该情形灰度→振幅耦合），它是实验系统上由
  ``tools/slm/bench_twin_calibrate.py`` 从系统自身测量反演出的校准算子在仿真里
  的替身。理想算子复现不出带耦合的实测帧，于是整形项的相关目标**物理上不可达**、
  梯度近似噪声；``calibrated`` 让整形项变成可实现的目标。网络内部前向与整形项
  **用同一个算子**；每情形的「迁移前」参照也用同一个算子，保证 Δ 只反映微调本身。
* ``--gate``（默认关）：微调后做一次**对比择优**——在**验证集探针**（与训练
  探针不相交、单独种子生成，规模 ``max(4, probes // 4)``）上，用微调所优化的
  那把尺子（整形项的同一 Pearson 相关、同一裁剪）比较候选权重与未微调权重，
  **只有候选更好才保留**；否则恢复预训练权重并记 ``gate: "kept_pretrained"``
  （保留则记 ``accepted``，关闭时记 ``off``）。这对应 model-in-loop 闭环的
  门 3（验证集 Pearson 验收）：没有这道门时，有害的微调总被采纳。

用法::

    python scripts/slm/gsnet_aberration_transfer.py                 # 全网格
    python scripts/slm/gsnet_aberration_transfer.py \
        --probes 6 --eval-samples 4 --epochs 2 --cases none,panel-coma \
        --freezes full,frozen --out /tmp/smoke/result.json          # 冒烟
    python scripts/slm/gsnet_aberration_transfer.py \
        --operator calibrated --w-phase 0 --gate                    # 变体
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from dataclasses import dataclass, replace
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
from ao_shaping.utils.wavefront.zernike_utils import generate_zernike_phase  # noqa: E402
from ml.gsnet.forward_ops import BenchTwinOperator  # noqa: E402
from ml.gsnet.freezing import (  # noqa: E402
    FREEZE_SPEC_NAMES,
    FreezeSpec,
    apply_freeze,
    prepare_for_training,
)
from ml.gsnet.hybrid_infer import load_gsnet_checkpoint  # noqa: E402

# ---- 几何：与 gsnet_pretrain.py 逐字段一致（checkpoint 就是它产出的）----
REGION = 64
FAR_FIELD = 512
TARGET_SIDE = 16
METRIC_SIDE = 36
GT_ITERATIONS = 12
REGION_SPAN_PX = 900.0
FAR_FIELD_PADDING = 8
PEAK_ADU = 100.0
NOISE_ADU = 2.0

#: 设备在 1064 nm 的 2π 灰度是 993（Santec 驱动从 EEPROM 动态查询，532 nm 为
#: 998）；1023 只是 10 位满量程，不是相位级数。灰度→振幅耦合的灰度域周期
#: 必须等于 2π 灰度数，所以每个情形的孪生都用 993（与 gsnet_pretrain 一致）。
SLM_TWO_PI_GRAY_LEVELS = 993

#: 相位项的随机猜测水平：相位差在 (-π, π] 上均匀时圆周 MSE 的期望（π²/3）。
PHASE_LOSS_CHANCE = float(np.pi**2 / 3.0)

#: 网络内部前向 = **理想**孪生（它并不知道相差）；相差只经由标签与实测帧进入。
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

#: 像差情形表：名称 → (中文说明, 在理想误差模型上的增量)。
#: 幅度 0.30 rad / 3e-4 m 超出预训练随机化范围（见模块 docstring）。
CASE_TABLE: dict[str, tuple[str, TwinErrorModel]] = {
    "none": ("无像差（对照）", TwinErrorModel()),
    "panel-defocus": (
        "面板离焦 (Noll 4) 0.30 rad",
        TwinErrorModel(slm_zernike={4: 0.30}, slm_zernike_n_max=6),
    ),
    "panel-coma": (
        "面板彗差 (Noll 7) 0.30 rad",
        TwinErrorModel(slm_zernike={7: 0.30}, slm_zernike_n_max=6),
    ),
    "panel-spherical": (
        "面板球差 (Noll 11) 0.30 rad",
        TwinErrorModel(slm_zernike={11: 0.30}, slm_zernike_n_max=6),
    ),
    "lens-coma": (
        "透镜彗差 (Noll 7) 0.30 rad",
        TwinErrorModel(lens_zernike={7: 0.30}, lens_zernike_n_max=4),
    ),
    "camera-defocus": ("相机离焦 3e-4 m", TwinErrorModel(defocus_m=3.0e-4)),
}


def build_aberrated_twin(errors: TwinErrorModel, seed: int) -> BenchTwin:
    """按与预训练完全相同的几何造一台带该情形误差的孪生（帧带噪声）。

    Args:
        errors: 该情形的三族误差参数。
        seed: 噪声 RNG 种子。

    Returns:
        ``BenchTwin`` 实例（``noise_adu = NOISE_ADU``）。
    """
    return BenchTwin(
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
            noise_adu=NOISE_ADU,
            seed=seed,
            errors=errors,
        )
    )


def square_target_fixed() -> np.ndarray:
    """远场 (512, 512) 上居中的方斑目标（边长 TARGET_SIDE，峰值 1）。

    构造与 ``gsnet_pretrain.square_target`` 相同（``target_side=TARGET_SIDE``、
    ``dx=dy=0`` 的特例）。
    """
    target = np.zeros((FAR_FIELD, FAR_FIELD), dtype=np.float32)
    half = TARGET_SIDE / 2.0
    cx = cy = FAR_FIELD / 2.0
    x0, x1 = int(round(cx - half)), int(round(cx + half))
    y0, y1 = int(round(cy - half)), int(round(cy + half))
    if x1 > x0 and y1 > y0:
        target[y0:y1, x0:x1] = 1.0
    return target


def peak_normalise(frame: np.ndarray) -> np.ndarray:
    """把非负帧峰值归一到 1（网络输入的标度，与设计方斑同量级）。"""
    arr = np.clip(np.asarray(frame, dtype=np.float32), 0.0, None)
    peak = float(arr.max())
    if peak > 0.0:
        arr = arr / peak
    return arr


def random_zernike_perturbation(rng: np.random.Generator) -> np.ndarray:
    """瞳面网格上的零均值随机 Zernike 微扰（Noll 4..12 随机子模式）。

    径向阶覆盖 Noll 4..12（n ≤ 4），系数均匀随机；孔径外置 0；再减均值、
    按整幅 rms 缩到 ``args.perturb_rms``。Zernike 数学只走 canonical 入口
    ``utils/wavefront/zernike_utils.generate_zernike_phase``。

    Args:
        rng: 该情形专属的随机数发生器。

    Returns:
        ``(REGION, REGION)`` float64 相位（raw 弧度）。
    """
    n_modes = int(rng.integers(3, 7))
    chosen = rng.choice(np.arange(4, 13), size=n_modes, replace=False)
    coeffs = {int(m): float(rng.uniform(-1.0, 1.0)) for m in chosen}
    raw = generate_zernike_phase(
        coeffs, resolution=(REGION, REGION), n_max=4, radius=REGION / 2.0
    )
    field = np.nan_to_num(np.asarray(raw, dtype=np.float64), nan=0.0)
    field -= field.mean()
    rms = float(np.sqrt((field**2).mean()))
    if rms > 0.0:
        field *= _CELL_ARGS["perturb_rms"] / rms
    return field


@dataclass
class CaseData:
    """一个像差情形的全部数据（可由 (情形名, 种子) 完全重放）。"""

    name: str
    description: str
    errors: TwinErrorModel
    target: np.ndarray                #: 设计方斑 (FAR_FIELD, FAR_FIELD)
    label_phase: np.ndarray           #: 该相差物理下的 GS 解（上限）(REGION, REGION)
    probe_phases: np.ndarray          #: (N, REGION, REGION) 命令过的探针相位
    eval_phases: np.ndarray           #: (M, REGION, REGION) 留出探针相位
    probe_inputs: np.ndarray          #: (N, FAR_FIELD, FAR_FIELD) 微调目标输入（按 transfer-target）
    eval_inputs: np.ndarray           #: (M, FAR_FIELD, FAR_FIELD) 留出样本的目标输入
    #: --gate 开启时才有：验证集探针（与训练探针不相交、独立种子）与其目标输入。
    gate_phases: np.ndarray | None = None
    gate_inputs: np.ndarray | None = None


#: 工作进程里要用的标量参数（spawn 重新 import 本模块，函数级读取）。
_CELL_ARGS: dict[str, Any] = {}


def _probe_twin_seed(case_index: int) -> int:
    """探针实测所在那台孪生的噪声种子。

    ``build_case_data`` 与 ``build_operator(calibrated)`` 共用这一条式子，
    保证「校准算子」与产生探针帧的那台孪生是同一台（含同一相差与同一耦合）。
    """
    return int(_CELL_ARGS["seed"]) + case_index


def build_operator(
    mode: str,
    errors: TwinErrorModel,
    case_index: int,
) -> BenchTwinOperator:
    """微调用的前向算子（网络内部前向与整形项**必须共用同一个**）。

    Args:
        mode: ``ideal`` = 理想孪生（网络不知道相差，原行为）；``calibrated`` =
            探针实测所在的那台孪生（带该情形相差与灰度→振幅耦合），即实验系统上
            由 ``tools/slm/bench_twin_calibrate.py`` 反演的校准算子在仿真里的替身。
        errors: 该情形的误差模型（含统一叠加的耦合）。
        case_index: 情形在网格里的序号（决定孪生噪声种子）。

    Returns:
        ``BenchTwinOperator`` 实例。
    """
    if mode == "calibrated":
        return BenchTwinOperator(build_aberrated_twin(errors, _probe_twin_seed(case_index)))
    return BenchTwinOperator(IDEAL_TWIN)


def _network_input(twin: BenchTwin, phases: np.ndarray, target: np.ndarray) -> np.ndarray:
    """按 ``--transfer-target`` 得到网络微调/验证时看到的**目标输入**。

    ``measured``：探针在**给定这台孪生**上实测的帧（峰值归一）；``designed``：
    设计方斑（与预训练相同的输入语义）。
    """
    from gsnet_pretrain import apply_detector  # noqa: PLC0415  (复用预训练的探测器模型)

    if _CELL_ARGS["transfer_target"] == "designed":
        return np.repeat(target[None], len(phases), axis=0)
    frames = [apply_detector(twin.frame(p), 1.0) for p in phases]
    return np.stack([peak_normalise(prepare_frame(f)) for f in frames])


def build_case_data(name: str, case_index: int) -> CaseData:
    """按情形名与索引重建全部数据（主进程与 worker 共用 → 逐位一致）。

    ``CASE_TABLE`` 只定义像差增量（实验的自变量轴）；2π 灰度数与灰度→振幅
    耦合是**目标实验系统本身的属性**，统一叠加到每个情形的孪生上，
    不改变情形之间的可比性。
    """
    description, case_errors = CASE_TABLE[name]
    errors = replace(
        case_errors,
        gray_levels=SLM_TWO_PI_GRAY_LEVELS,
        slm_amplitude_coupling=_CELL_ARGS["coupling"],
    )
    probe_twin = build_aberrated_twin(errors, _probe_twin_seed(case_index))
    eval_twin = build_aberrated_twin(errors, _CELL_ARGS["seed"] + case_index + 555)
    target = square_target_fixed()

    # 上限：在**该情形自己的物理**下解 GS（走 torch 镜像算子，因为 GS 要远场相位）。
    operator_ab = BenchTwinOperator(probe_twin)
    target_amp = torch.from_numpy(np.asarray(target, dtype=np.float32))[None, None].clamp_min(0.0)
    target_amp = torch.sqrt(target_amp + 1e-12)
    phase = operator_ab.warm_start(target_amp)
    for _ in range(GT_ITERATIONS):
        far = operator_ab.forward(None, phase)
        far_c = target_amp * torch.exp(1j * torch.angle(far))
        phase = torch.angle(operator_ab.back_propagate(far_c))
    label = phase[0, 0].numpy().astype(np.float64)

    # 探针 = 标签 + 零均值随机 Zernike 微扰（模仿 model-in-loop 的小幅 excursion）。
    probe_rng = np.random.default_rng(_CELL_ARGS["seed"] + 6151 * case_index)
    eval_rng = np.random.default_rng(_CELL_ARGS["seed"] + 6151 * case_index + 104729)
    probe_phases = np.stack(
        [label + random_zernike_perturbation(probe_rng) for _ in range(_CELL_ARGS["probes"])]
    )
    eval_phases = np.stack(
        [label + random_zernike_perturbation(eval_rng) for _ in range(_CELL_ARGS["eval_samples"])]
    )

    gate_phases: np.ndarray | None = None
    gate_inputs: np.ndarray | None = None
    if _CELL_ARGS.get("gate"):
        # 门用的验证集探针：独立种子（与训练/评分两套 rng 都不相交），规模
        # max(4, probes // 4)。帧取在**产生训练探针的那台孪生**上，所以门度量的
        # 正是微调所优化的那件事（同一算子、同一实测语义）。
        gate_rng = np.random.default_rng(
            _CELL_ARGS["seed"] + 6151 * case_index + 987_654_321
        )
        n_gate = max(4, int(_CELL_ARGS["probes"]) // 4)
        gate_phases = np.stack(
            [label + random_zernike_perturbation(gate_rng) for _ in range(n_gate)]
        )
        gate_inputs = _network_input(probe_twin, gate_phases, target)

    return CaseData(
        name=name,
        description=description,
        errors=errors,
        target=target,
        label_phase=label,
        probe_phases=probe_phases,
        eval_phases=eval_phases,
        probe_inputs=_network_input(probe_twin, probe_phases, target),
        eval_inputs=_network_input(eval_twin, eval_phases, target),
        gate_phases=gate_phases,
        gate_inputs=gate_inputs,
    )


def _batch_tensors(
    phases: np.ndarray, inputs: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(source, target, label) 张量对，source 与预训练一样是理想孪生的瞳面照明。"""
    source = torch.from_numpy(
        np.asarray(IDEAL_TWIN.pupil_amplitude(), dtype=np.float32)
    )[None, None]
    n_samples = phases.shape[0]
    source_t = source.expand(n_samples, 1, REGION, REGION).contiguous()
    target_t = torch.from_numpy(inputs.astype(np.float32))[:, None]
    label_t = torch.from_numpy(phases.astype(np.float32))[:, None]
    return source_t, target_t, label_t


def _batch_losses(
    model: torch.nn.Module,
    operator: BenchTwinOperator,
    source: torch.Tensor,
    source_amp: torch.Tensor,
    target: torch.Tensor,
    label: torch.Tensor,
    idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """一个 batch 的 (相位项, 整形项)，公式与 ``gsnet_transfer.finetune`` 逐字一致。

    相位项是对**命令过的探针相位**的圆周 MSE；整形项是网络内部前向（``operator``
    —— ``ideal`` 为理想孪生，``calibrated`` 为探针实测所在的那台孪生）的远场强度
    与**目标输入**（实测帧或设计方斑）在远场中心 ``METRIC_SIDE`` 裁剪上的 1 − Pearson。
    """
    pred = model(source[idx], target[idx])
    diff = torch.remainder(pred - label[idx] + torch.pi, 2 * torch.pi) - torch.pi
    phase_loss = torch.nn.functional.mse_loss(diff, torch.zeros_like(diff))
    far = operator.forward(source_amp[idx], pred)
    intensity = far.real**2 + far.imag**2
    side = min(METRIC_SIDE, intensity.shape[-1])
    half = (intensity.shape[-1] - side) // 2
    sl = slice(half, half + side)
    p = intensity[:, :, sl, sl].flatten(1)
    t = target[idx][:, :, sl, sl].flatten(1)
    p = p - p.mean(dim=1, keepdim=True)
    t = t - t.mean(dim=1, keepdim=True)
    corr = (p * t).sum(dim=1) / (
        torch.sqrt((p**2).sum(dim=1) * (t**2).sum(dim=1)) + 1e-12
    )
    return phase_loss, torch.mean(1.0 - corr)


def finetune_case(
    model: torch.nn.Module,
    operator: BenchTwinOperator,
    data: CaseData,
    spec: FreezeSpec,
    cell_seed: int,
) -> tuple[dict[str, int], list[dict[str, float]]]:
    """按冻结配置微调一个情形的探针数据。

    ``prepare_for_training``（``train()`` 后再施加冻结）**每个 epoch 调一次**：
    只调 ``model.train()`` 会把冻结组的 BatchNorm 统计放回训练模式 —— 那正是
    ``ml.gsnet/freezing.py`` 存在的理由。可训练参数为 0（``frozen``）时完全跳过
    优化，只记录一次不更新的损失。总损失 ``w_phase·相位项 + w_shaping·整形项``，
    ``w_phase`` 默认 1.0（与原行为逐字一致），置 0 即只用整形项。

    Returns:
        ``(apply_freeze 的参数量计数, 每 epoch 的 history 行)``。
    """
    args = _CELL_ARGS
    torch.manual_seed(cell_seed)
    counts = apply_freeze(model, spec)
    trainable = [p for p in model.parameters() if p.requires_grad]
    source, target, label = _batch_tensors(data.probe_phases, data.probe_inputs)
    source_amp = torch.sqrt(source.clamp_min(0.0) + 1e-12)
    history: list[dict[str, float]] = []
    if not trainable:
        model.eval()
        with torch.no_grad():
            idx = torch.arange(source.shape[0])
            phase_loss, shaping_loss = _batch_losses(
                model, operator, source, source_amp, target, label, idx
            )
        history.append(
            {
                "epoch": 0,
                "phase_loss": float(phase_loss),
                "shaping_loss": float(shaping_loss),
            }
        )
        return counts, history

    optimiser = torch.optim.Adam(trainable, lr=args["lr"])
    n = source.shape[0]
    for epoch in range(args["epochs"]):
        prepare_for_training(model, spec)
        order = torch.randperm(n)
        phase_losses: list[float] = []
        shaping_losses: list[float] = []
        for start in range(0, n, args["batch_size"]):
            idx = order[start : start + args["batch_size"]]
            phase_loss, shaping_loss = _batch_losses(
                model, operator, source, source_amp, target, label, idx
            )
            loss = args["w_phase"] * phase_loss + args["w_shaping"] * shaping_loss
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            phase_losses.append(float(phase_loss.detach()))
            shaping_losses.append(float(shaping_loss.detach()))
        history.append(
            {
                "epoch": epoch + 1,
                "phase_loss": float(np.mean(phase_losses)) if phase_losses else float("nan"),
                "shaping_loss": (
                    float(np.mean(shaping_losses)) if shaping_losses else float("nan")
                ),
            }
        )
    return counts, history


def gate_shaping_loss(
    model: torch.nn.Module,
    operator: BenchTwinOperator,
    data: CaseData,
) -> float:
    """``--gate`` 的判据：验证集探针上的整形项均值（越小越好）。

    直接调 ``_batch_losses`` 取整形项 —— 与微调优化的**完全同一把尺子**（同一
    Pearson 相关、同一 ``METRIC_SIDE`` 裁剪、同一算子），所以门裁决的就是被优化
    的那件事。相位项在此不参与判定。
    """
    if data.gate_phases is None or data.gate_inputs is None:
        raise ValueError("gate 已开启但验证集探针缺失（build_case_data 未生成）")
    source, target, label = _batch_tensors(data.gate_phases, data.gate_inputs)
    source_amp = torch.sqrt(source.clamp_min(0.0) + 1e-12)
    model.eval()
    with torch.no_grad():
        _, shaping_loss = _batch_losses(
            model,
            operator,
            source,
            source_amp,
            target,
            label,
            torch.arange(source.shape[0]),
        )
    return float(shaping_loss)


def score_holdout(
    model: torch.nn.Module,
    data: CaseData,
    eval_twin_seed: int,
) -> dict[str, float]:
    """在留出集上评分：预测相位 → 该相差孪生实测 → 与设计方斑比。

    评分复用 ``calibrate_gsnet_2f.frame_metrics``（口径与预训练验收一致）。
    ``flat`` / ``label`` 参照也走同一条实测链（phase 直接来自孪生下发）。
    """
    from calibrate_gsnet_2f import frame_metrics  # noqa: PLC0415

    twin = build_aberrated_twin(data.errors, eval_twin_seed)

    def measure(phase: np.ndarray) -> np.ndarray:
        from gsnet_pretrain import apply_detector  # noqa: PLC0415

        return prepare_frame(apply_detector(twin.frame(phase), 1.0))

    rows: list[dict[str, float]] = []
    source = torch.from_numpy(
        np.asarray(IDEAL_TWIN.pupil_amplitude(), dtype=np.float32)
    )[None, None]
    model.eval()
    with torch.no_grad():
        for i in range(data.eval_phases.shape[0]):
            pred = model(
                source.clone(),
                torch.from_numpy(data.eval_inputs[i : i + 1])[None],
            )
            phase = pred[0, 0].numpy().astype(np.float64)
            rows.append(
                frame_metrics(
                    measure(phase),
                    data.target.astype(np.float64),
                    METRIC_SIDE,
                    TARGET_SIDE,
                )
            )
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in (
            "correlation",
            "nrmse",
            "correlation_full",
            "nrmse_full",
            "encircled_energy",
            "cv_target_box",
            "cv_flat_rms",
            "composite",
        )
    }


def reference_metrics(data: CaseData, eval_twin_seed: int) -> dict[str, dict[str, float]]:
    """平场下限与该相差下的 GS 上限（同一把尺子：frame_metrics 对设计方斑）。"""
    from calibrate_gsnet_2f import frame_metrics  # noqa: PLC0415
    from gsnet_pretrain import apply_detector  # noqa: PLC0415

    twin = build_aberrated_twin(data.errors, eval_twin_seed)
    keys = (
        "correlation",
        "nrmse",
        "correlation_full",
        "nrmse_full",
        "encircled_energy",
        "cv_target_box",
        "cv_flat_rms",
        "composite",
    )
    out: dict[str, dict[str, float]] = {}
    for label_name, phase in (
        ("flat", np.zeros((REGION, REGION), dtype=np.float64)),
        ("label", data.label_phase.astype(np.float64)),
    ):
        rows = [
            frame_metrics(
                prepare_frame(apply_detector(twin.frame(phase), 1.0)),
                data.target.astype(np.float64),
                METRIC_SIDE,
                TARGET_SIDE,
            )
            for _ in range(_CELL_ARGS["eval_samples"])
        ]
        out[label_name] = {key: float(np.mean([row[key] for row in rows])) for key in keys}
    return out


def _cell_worker(payload: tuple[dict[str, Any], str, int, str]) -> tuple[str, str, dict[str, Any]]:
    """进程池里的一个网格单元：(case, freeze) → 微调 + 留出评分。

    每单元自己重建 case 数据（确定性种子，与主进程逐位一致），加载**未微调**的
    checkpoint，微调后评分。全局 ``--seed`` 按 (情形, 冻结) 偏移出单元种子，
    所以结果与 worker 数、完成顺序无关。
    """
    cell_args, case_name, case_index, freeze_name = payload
    _CELL_ARGS.clear()
    _CELL_ARGS.update(cell_args)
    torch.set_num_threads(2)
    spec = FreezeSpec.from_name(freeze_name)
    data = build_case_data(case_name, case_index)
    operator = build_operator(cell_args["operator"], data.errors, case_index)
    model = load_gsnet_checkpoint(Path(cell_args["checkpoint"]), forward_op=operator)
    cell_seed = (
        cell_args["seed"]
        + 7919 * case_index
        + 3571 * FREEZE_SPEC_NAMES.index(freeze_name)
    )
    gated = bool(cell_args.get("gate"))
    pretrained_state = (
        {k: v.detach().clone() for k, v in model.state_dict().items()} if gated else None
    )
    gate_pre = gate_shaping_loss(model, operator, data) if gated else None
    counts, history = finetune_case(model, operator, data, spec, cell_seed)
    gate_post = gate_shaping_loss(model, operator, data) if gated else None
    gate = "off"
    if gated:
        # 门 3（对比择优）：候选只有在验证集探针上把整形项做得**严格更好**时才保留，
        # 否则恢复预训练权重 —— 让 final 度量反映真正会被部署的那套权重。
        gate = "accepted" if gate_post < gate_pre else "kept_pretrained"
        if gate == "kept_pretrained":
            model.load_state_dict(pretrained_state)
    metrics = score_holdout(
        model, data, cell_args["seed"] + case_index + 555
    )
    final_phase_loss = history[-1]["phase_loss"] if history else float("nan")
    run = {
        "trainable_params": counts["trainable"],
        "total_params": counts["total"],
        "frozen_params": counts["frozen"],
        "w_phase": float(cell_args["w_phase"]),
        "operator": str(cell_args["operator"]),
        "gate": gate,
        "gate_shaping_pretrained": (float(gate_pre) if gated else None),
        "gate_shaping_candidate": (float(gate_post) if gated else None),
        "final": {
            key: metrics[key]
            for key in ("composite", "encircled_energy", "cv_target_box", "correlation_full")
        },
        "phase_loss": final_phase_loss,
        "phase_loss_chance": PHASE_LOSS_CHANCE,
        "history": history,
    }
    return case_name, freeze_name, run


def errors_to_dict(errors: TwinErrorModel) -> dict[str, Any]:
    """把 ``TwinErrorModel`` 导出为 JSON 可序列化的普通字典。"""
    return {
        "lut_gamma": float(errors.lut_gamma),
        "lut_phase_max_rad": float(errors.lut_phase_max_rad),
        "gray_levels": int(errors.gray_levels),
        "lut_table": None if errors.lut_table is None else "omitted",
        "slm_zernike": {str(int(k)): float(v) for k, v in errors.slm_zernike.items()},
        "slm_zernike_n_max": int(errors.slm_zernike_n_max),
        "slm_roughness_sigma_rad": float(errors.slm_roughness_sigma_rad),
        "slm_roughness_seed": int(errors.slm_roughness_seed),
        "defocus_m": float(errors.defocus_m),
        "tilt_x_rad": float(errors.tilt_x_rad),
        "tilt_y_rad": float(errors.tilt_y_rad),
        "lens_zernike": {str(int(k)): float(v) for k, v in errors.lens_zernike.items()},
        "lens_zernike_n_max": int(errors.lens_zernike_n_max),
        "slm_amplitude_coupling": float(errors.slm_amplitude_coupling),
        "slm_amplitude_coupling_phase_rad": float(errors.slm_amplitude_coupling_phase_rad),
    }


def print_tables(
    cases: dict[str, dict[str, Any]],
    case_order: list[str],
    primary_case: str,
    primary_freeze: str,
) -> None:
    """打印两张中文 markdown 表（像差扫描 / 冻结扫描）。"""

    def delta(case: dict[str, Any], freeze: str) -> float:
        return case["runs"][freeze]["final"]["composite"] - case["references"]["pretrained"]["composite"]

    print(f"\n## 表格 1：像差扫描（冻结配置 = `{primary_freeze}`，指标 = composite）\n")
    print("| case | 像差说明 | 平场 | 标签上限 | 迁移前 | 迁移后 | 迁移后−迁移前 |")
    print("|---|---|---|---|---|---|---|")
    for name in case_order:
        case = cases[name]
        after = case["runs"][primary_freeze]["final"]["composite"]
        pre = case["references"]["pretrained"]["composite"]
        print(
            f"| {name} | {case['description']} "
            f"| {case['references']['flat']['composite']:.4f} "
            f"| {case['references']['label']['composite']:.4f} "
            f"| {pre:.4f} | {after:.4f} | {after - pre:+.4f} |"
        )

    primary = cases[primary_case]
    print(f"\n## 表格 2：冻结策略扫描（情形 = `{primary_case}`，指标 = composite）\n")
    print(
        "| freeze | 可训练参数 | 冻结参数 | 平场 | 标签上限 | 迁移前 | 迁移后 | 迁移后−迁移前 |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for freeze_name, run in primary["runs"].items():
        after = run["final"]["composite"]
        pre = primary["references"]["pretrained"]["composite"]
        print(
            f"| {freeze_name} | {run['trainable_params']} | {run['frozen_params']} "
            f"| {primary['references']['flat']['composite']:.4f} "
            f"| {primary['references']['label']['composite']:.4f} "
            f"| {pre:.4f} | {after:.4f} | {after - pre:+.4f} |"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "data" / "gsnet_pretrain" / "20261021_twin" / "checkpoint.pt",
        help="预训练 checkpoint（gsnet_pretrain.py 产出）",
    )
    parser.add_argument("--cases", type=str, default=",".join(CASE_TABLE), help="逗号分隔的像差情形")
    parser.add_argument(
        "--freezes", type=str, default=",".join(FREEZE_SPEC_NAMES), help="逗号分隔的冻结配置"
    )
    parser.add_argument("--primary-case", type=str, default="panel-coma", help="冻结扫描的基准情形")
    parser.add_argument("--primary-freeze", type=str, default="full", help="像差扫描的基准冻结配置")
    parser.add_argument("--probes", type=int, default=24, help="每情形的微调探针数")
    parser.add_argument("--eval-samples", type=int, default=16, help="每情形的留出评估样本数")
    parser.add_argument("--epochs", type=int, default=10, help="每网格单元的微调轮数")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--w-shaping", type=float, default=40.0, help="整形项权重（与 gsnet_transfer 一致）")
    parser.add_argument(
        "--w-phase",
        type=float,
        default=1.0,
        help="相位项权重（默认 1.0 = 原行为；0.0 = 只用整形项。GS 标签本身可能低于"
             "预训练水平，此时相位项是向下的拉力）",
    )
    parser.add_argument(
        "--operator",
        choices=("ideal", "calibrated"),
        default="ideal",
        help="微调用的前向算子：ideal = 理想孪生（原行为）；calibrated = 探针实测所在的"
             "那台孪生（带该情形相差与灰度→振幅耦合），即实验系统上由 "
             "tools/slm/bench_twin_calibrate.py 反演的校准算子在仿真里的替身",
    )
    parser.add_argument(
        "--gate",
        action="store_true",
        help="微调后做对比择优：在验证集探针上比较候选与未微调权重的整形项，"
             "候选不更好就恢复预训练权重（记 gate=kept_pretrained）",
    )
    parser.add_argument("--perturb-rms", type=float, default=0.08, help="探针 Zernike 微扰的 rms（rad）")
    parser.add_argument(
        "--coupling",
        type=float,
        default=0.15,
        help="SLM 灰度→振幅耦合强度 c，叠加到每个情形的孪生上（实验系统的属性，不是消融轴）",
    )
    parser.add_argument(
        "--transfer-target",
        choices=("measured", "designed"),
        default="measured",
        help="微调目标输入：实测帧（在机语义）或设计方斑（预训练语义）",
    )
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(4, max(1, (mp.cpu_count() or 4) // 2)),
        help="微调网格单元的并行进程数（1 = 串行）",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="JSON 输出路径（默认 data/debug/gsnet_aberration_transfer/<时间戳>/result.json）",
    )
    args = parser.parse_args(argv)

    case_order = [name.strip() for name in args.cases.split(",") if name.strip()]
    freeze_order = [name.strip() for name in args.freezes.split(",") if name.strip()]
    for name in case_order:
        if name not in CASE_TABLE:
            parser.error(f"unknown case {name!r}; expected one of {sorted(CASE_TABLE)}")
    for name in freeze_order:
        try:
            FreezeSpec.from_name(name)
        except ValueError as exc:
            parser.error(str(exc))
    if args.primary_case not in case_order:
        parser.error(f"--primary-case {args.primary_case!r} must be in --cases")
    if args.primary_freeze not in freeze_order:
        parser.error(f"--primary-freeze {args.primary_freeze!r} must be in --freezes")
    if not args.checkpoint.exists():
        print(f"找不到 checkpoint：{args.checkpoint}", flush=True)
        return 1

    out_path = args.out or (
        ROOT / "data" / "debug" / "gsnet_aberration_transfer" / time.strftime("%Y%m%d_%H%M%S") / "result.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    _CELL_ARGS.clear()
    _CELL_ARGS.update(
        {
            "seed": int(args.seed),
            "probes": int(args.probes),
            "eval_samples": int(args.eval_samples),
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "batch_size": int(args.batch_size),
            "w_shaping": float(args.w_shaping),
            "w_phase": float(args.w_phase),
            "operator": str(args.operator),
            "gate": bool(args.gate),
            "perturb_rms": float(args.perturb_rms),
            "coupling": float(args.coupling),
            "transfer_target": str(args.transfer_target),
            "checkpoint": str(args.checkpoint),
        }
    )

    started = time.time()
    cases_out: dict[str, dict[str, Any]] = {}
    print(
        f"情形 {case_order} × 冻结 {freeze_order}（transfer-target={args.transfer_target}，"
        f"operator={args.operator}，w_phase={args.w_phase}，gate={args.gate}）…",
        flush=True,
    )

    # 每情形一次：case 数据 + 上下限参照 + 微调前的验证集评分。
    case_index_of = {name: i for i, name in enumerate(case_order)}
    for name in case_order:
        index = case_index_of[name]
        data = build_case_data(name, index)
        references = reference_metrics(data, args.seed + index + 555)
        # 「迁移前」参照用**本次选定的同一个算子**：否则 Δ(迁移后−迁移前) 会把
        # 换算子本身混进迁移的效应里。
        operator = build_operator(args.operator, data.errors, index)
        pretrained_model = load_gsnet_checkpoint(args.checkpoint, forward_op=operator)
        references["pretrained"] = score_holdout(pretrained_model, data, args.seed + index + 555)
        cases_out[name] = {
            "description": data.description,
            "errors": errors_to_dict(data.errors),
            "references": references,
            "runs": {},
        }
        print(
            f"  [{name}] 参照：平场 {references['flat']['composite']:.4f} / "
            f"上限 {references['label']['composite']:.4f} / "
            f"迁移前 {references['pretrained']['composite']:.4f}"
            f"  ({time.time() - started:.0f}s)",
            flush=True,
        )

    # 全网格微调：每单元独立、种子固定 → 与并行度无关。
    tasks = [
        (dict(_CELL_ARGS), name, case_index_of[name], freeze_name)
        for name in case_order
        for freeze_name in freeze_order
    ]
    done = 0
    if args.workers <= 1:
        results = [_cell_worker(task) for task in tasks]
    else:
        with mp.get_context("spawn").Pool(args.workers) as pool:
            results = list(pool.imap_unordered(_cell_worker, tasks))
    for name, freeze_name, run in results:
        cases_out[name]["runs"][freeze_name] = run
        done += 1
        print(
            f"  [{done}/{len(tasks)}] {name} × {freeze_name}："
            f"可训练 {run['trainable_params']}/{run['total_params']}，"
            f"composite {run['final']['composite']:.4f}，"
            f"phase_loss {run['phase_loss']:.4f}（随机水平 {PHASE_LOSS_CHANCE:.4f}）"
            + (
                f"，gate {run['gate']}"
                f"（整形 {run['gate_shaping_pretrained']:.4f} → "
                f"{run['gate_shaping_candidate']:.4f}）"
                if run["gate"] != "off"
                else ""
            )
            + f"  ({time.time() - started:.0f}s)",
            flush=True,
        )
    # runs 按 --freezes 的顺序排列（进程完成顺序无关）。
    for name in case_order:
        cases_out[name]["runs"] = {
            freeze_name: cases_out[name]["runs"][freeze_name] for freeze_name in freeze_order
        }

    sweeps = {
        "aberration": [
            {
                "case": name,
                "freeze": args.primary_freeze,
                "composite": cases_out[name]["runs"][args.primary_freeze]["final"]["composite"],
                "trainable_params": cases_out[name]["runs"][args.primary_freeze]["trainable_params"],
            }
            for name in case_order
        ],
        "freeze": [
            {
                "freeze": freeze_name,
                "trainable_params": cases_out[args.primary_case]["runs"][freeze_name]["trainable_params"],
                "composite": cases_out[args.primary_case]["runs"][freeze_name]["final"]["composite"],
            }
            for freeze_name in freeze_order
        ],
    }
    summary = {
        "config": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        }
        | {"out": str(out_path)},
        "checkpoint": str(args.checkpoint),
        "geometry": {
            "region": REGION,
            "far_field": FAR_FIELD,
            "target_side": TARGET_SIDE,
            "metric_side": METRIC_SIDE,
        },
        "cases": cases_out,
        "sweeps": sweeps,
    }
    out_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print_tables(cases_out, case_order, args.primary_case, args.primary_freeze)
    print(f"\n结果：{out_path}  （{time.time() - started:.0f}s）", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
