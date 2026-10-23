"""目标尺寸（方形目标边长）扫描: 模型侧 dry-run + 设备侧 --execute 验证.

问题
----

2f 实验系统下，方形目标边长存在**最优区间**：太小 ⇒ 零级泄漏 / 盒装不满；
太大 ⇒ 衍射与高频丢失，能量摊不满大盒。本脚本把
:mod:`ao_shaping.tools.slm.panel.target_size` 的解析界限 + 峰值窗口 + 单峰拟合
落到**冻结物理前向**（``data/debug/physical_forward_defocus/uniformity_refit_20261022_r1``
检查点，调用 :mod:`ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity`
的 ``load_frozen_parameters`` / ``FrozenDefocusForward`` / ``build_masks`` /
``DifferentiableUniformityOptimizer``，只调用不修改）上：

1. 对每个候选边长（``--sizes`` 显式 px 列表优先；否则
   ``OpticalLimits.candidate_sizes_px`` 默认 0.05–0.80 × 实用上限）用**同一份**
   零相位初值 + 迭代数 + 步长上限跑可微均匀度优化；
2. 在模型图上算**防退化指标**（2026-10-23 修订，分子/分母口径见下方
   「指标口径」节，并逐字写进 ``sweep.json`` 的 ``metric_definitions``）；
3. ``select_best_window``（先按 valid 过滤、峰值 + 1% 平台、退化指标显式
   拒斥）与 ``fit_optimum``（log 尺寸上二次单峰）各给一个判决；
4. 落 ``logs/target_size_sweep/<UTC 时间戳>/``：``sweep.csv`` / ``sweep.json`` /
   ``curve.png``，并把查找表 ``target_size_calibration.json`` 写到
   ``data/calibration/``（换 (λ, f, N, p, 相机) 后 ``TargetSizeCalibration.load``
   的签名校验会**显式拒绝**旧表）。

指标口径（防退化修订 2026-10-23）
----------------------------------

* ``efficiency`` = E(目标盒, 优化图) / E(参考窗)。**参考窗在所有尺寸上是同一
  固定区域**：模型侧 = 整个 121×121 采样网格（2 px/采样，240 px 跨度），实测侧
  = 实测零点中心 241×241 相机像元窗。旧口径「盒/同尺寸窗」分母跟着盒子变大，
  是恒等式不是效率，已弃用。
* ``efficiency_vs_baseline`` = efficiency − efficiency_baseline（未优化零相位
  图），是优化器相对「几何恒等式」的净增益，**参与打分**。
* ``uniformity`` = CV(core)，只在 ``valid=True``（mean_core ≥ k×噪声地板，
  默认 k=3）时计入；空盒 CV 被剔除，不再污染窗口选择。
* ``fill`` = 目标盒内被点亮像素占比（强度 ≥ k×噪声地板，与有效性同一阈值）。
* ``zero_order_ratio`` = E(固定零级盘, 半径 15 px) / E(参考窗)；``zero_abs`` 是
  盘内能量**绝对值**，避免比值随目标尺寸变化被稀释。
* ``fidelity_proxy``（dry-run）= 同一相位在冻结模型下的预测图与**已采实测
  A1 帧**（``--ref-frame-dir``，按零点同窗对齐）的 Pearson 相关 —— 不碰设备。

协议脚本在哪一步用它（一句话）
------------------------------

在**开设备后、正式采集前**，用 ``TargetSizeCalibration.load(path)`` 读
``target_size_calibration.json``（签名不匹配会被显式拒绝并要求重标定），
``cal.recommend("square")``（默认 ``constraint="largest_valid"`` = 最大有效
尺寸）取出推荐边长（相机像元），直接作为 ``build_masks(shape, centre_ij,
box_px, inner_shrink_px, step_ccd_px)`` 的 ``box_px`` 冻结方框/核心掩膜。

🔴 **默认 dry-run：只跑模型侧扫描，绝不 import 或 open 任何设备。**
``--execute`` 分支才 import 驱动（Santec SLM + Daheng 相机），流程：

1. 可选 ``--wait-until-idle`` 走 ``run_hw_sweep_groups.wait_until_idle`` 公平仲裁
   （不抢设备）；
2. 上**平场**（raw 灰度直发，不走 ``create_phase_from_array``）→ 抓帧 →
   ``prepare_frame`` 去读出噪声后 **argmax 定位零级**（不假设它在帧中心）；
3. 用 ``recentre_window`` 以实测零级为中心重建模型采样窗（换会话后零点会漂，
   沿用检查点旧窗会让零级贴边），重建 ``FrozenDefocusForward`` 后重跑模型扫描；
4. 逐尺寸两态采集（平场 = A 态、优化相位 = B 态，各 ``--frames`` 帧、先丢
   ``--discard`` 帧）；相位经 ``phase_to_slm_grayscale(phase, slm=slm)`` 转灰度，
   ``SlotRotator`` 轮换内存槽（同槽连写是固件 no-op）；
5. 实测指标用同一套防退化口径（固定 241×241 参考窗、固定 15 px 零级盘、
   噪声地板 = 平场背景标准差），模型↔实测强度相关（fidelity）合并进同一份
   CSV/JSON，标定表落盘（``method="model+hardware sweep"``）。

运行
----

.. code-block:: bash

    # 无硬件机器上也能跑: 只跑模型侧 (全粗扫 16 档)
    python scripts/slm/sweep_target_size.py --dry-run

    # 快速冒烟 (3 档 × 4 步)
    python scripts/slm/sweep_target_size.py --dry-run --limit-sizes 3 --steps 4

    # 细扫: 显式 px 列表 (40–240, 步长 20)
    python scripts/slm/sweep_target_size.py --dry-run --sizes 40,60,80,100,120,140,160,180,200,220,240 --steps 8

    # 装好设备后: 模型侧 + 设备侧验证
    python scripts/slm/sweep_target_size.py --execute --wait-until-idle 600

退出码: 0 = 正常; 2 = 设备挂死/超时（:func:`guarded` 硬退出）;
6 = 模型侧扫描失败（检查点缺失 / 参数缺失 / 候选尺寸不足）;
7 = 设备被别的会话占用, 等不到空闲。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

ROOT = Path(__file__).resolve().parents[2]
# `scripts.*` 以包路径被 import, 所以仓库根本身必须可 import —— 以文件形式启动
# (`python scripts/x/y.py`) 时 cwd 不在 sys.path 上, 只有脚本自己的目录在。
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib

matplotlib.use("Agg")  # 必须在 pyplot 之前

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from ao_shaping.tools.slm.panel.target_size import (
    DEFAULT_SYSTEM,
    OpticalLimits,
    SizeMetrics,
    TargetSizeCalibration,
    composite_scores,
    default_fractions,
    fit_optimum,
    is_valid,
    select_best_window,
)
from scripts.slm.hw_gs_defocus_probe import guarded

# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #

DEFAULT_CHECKPOINT = (
    ROOT / "data" / "debug" / "physical_forward_defocus" / "uniformity_refit_20261022_r1"
)
DEFAULT_OUT_ROOT = ROOT / "logs" / "target_size_sweep"
DEFAULT_CALIBRATION_PATH = ROOT / "data" / "calibration" / "target_size_calibration.json"

#: 模型侧扫描
DEFAULT_STEPS = 20
DEFAULT_LR = 0.01
DEFAULT_CAP = 0.1
#: 拟合会话的零级 (相机像元, x/y)；缺省时在模型图上 argmax 自定位。
DEFAULT_ZERO_ORDER_XY = (1432.0, 968.0)
#: 固定零级盘半径 (相机像元)。2026-10-23 修订: 固定 15 px, 不随目标尺寸变化。
DEFAULT_ZERO_ORDER_RADIUS_PX = 15.0
DEFAULT_INNER_SHRINK_PX = 3
#: 有效性/点亮阈值系数 k: valid = mean_core ≥ k×噪声地板 (2026-10-23 实测 k=3)
DEFAULT_VALID_K = 3.0

#: 设备侧采样窗 (与拟合口径一致: 121 点 × 2 px 步长); 实测参考窗 = (2·half+1)²
#: = 241×241 相机像元, 所有尺寸共用 (模型侧参考窗 = 整个采样网格, 同几何口径)。
DEFAULT_HALF_SPAN_PX = 120.0
DEFAULT_WINDOW_STEP_PX = 2.0

#: fidelity_proxy 参考帧 (已采实测 A1 帧目录; dry-run 不碰设备, 用存档帧)
DEFAULT_REF_FRAME_DIR = (
    ROOT / "data" / "debug" / "defocus_differentiable" / "20261023_101822" / "stage_e" / "rep0"
)
DEFAULT_REF_LABEL = "A1"

#: 设备侧采集
DEFAULT_FLAT_GRAY = 512
DEFAULT_FRAMES = 5
DEFAULT_DISCARD = 2
DEFAULT_SETTLE_S = 0.5
DEFAULT_EXPOSURE_MS = 2.0
OPEN_TIMEOUT_S = 180.0
CLOSE_TIMEOUT_S = 30.0
ACQUIRE_TIMEOUT_S = 60.0

PANEL_SHAPE = (1200, 1920)
GRID_PITCH_PX = 15  # 1920 / 128: 控制网格一格 = 15×15 面板像元

EXIT_OK = 0
EXIT_DEVICE_HANG = 2
EXIT_MODEL_FAILED = 6
EXIT_BENCH_BUSY = 7

CJK_FONTS = ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans")

#: 每个量的分子/分母定义 (逐字写进 sweep.json 的 ``metric_definitions`` 块)
METRIC_DEFINITIONS: dict[str, str] = {
    "efficiency": (
        "分子 = E(目标盒, 优化后图像); 分母 = E(参考窗)。参考窗在所有尺寸上是同一"
        "固定区域: 模型侧 = 整个 121×121 采样网格 (2 px/采样, 240 px 跨度), 实测侧 = "
        "实测零点中心 241×241 相机像元窗; 盒比参考窗大时可 > 1"
    ),
    "efficiency_baseline": (
        "同 efficiency 定义, 但用**未优化基线图** (零相位, 同盒同参考窗)"
    ),
    "efficiency_vs_baseline": (
        "efficiency − efficiency_baseline = 优化器相对几何恒等式的净增益; "
        "参与合成分数的 efficiency 项 (composite_scores / select_best_window)"
    ),
    "uniformity": (
        "CV(core) = sqrt(mean((v/μ−1)²)), 优化后图像; 只在 valid=True 时计入, "
        "invalid 空盒点从窗口选择与单峰拟合中整点剔除"
    ),
    "mean_core": "目标区 (core 掩膜) 内平均强度 (图像单位); 有效性判据的量",
    "fill": (
        "目标盒内被点亮像素占比 (像素强度 ≥ k×噪声地板), k 与有效性同一阈值; "
        "直接回答「这个尺寸的盒填上了多少」"
    ),
    "valid": (
        "mean_core ≥ k×噪声地板 (is_valid, 默认 k=3)。模型侧地板 = 冻结前向的 "
        "background 参数; 实测侧地板 = 平场帧去本底后的背景标准差"
    ),
    "zero_order_ratio": (
        "分子 = E(固定零级盘, 半径 15 px, 以零级为中心); 分母 = E(参考窗); "
        "零级盘固定, 不随目标尺寸变化"
    ),
    "zero_abs": "固定零级盘内能量绝对值 (图像单位), 避免比值随目标尺寸变化被稀释",
    "fidelity_proxy": (
        "同一相位在冻结模型下的预测图与已采实测帧 (--ref-frame-dir / --ref-label, "
        "缺省 A1) 的 Pearson 相关, 同窗对齐: 模型采样点按其相机坐标相对模型零级的"
        "偏移 + 实测零级定位到实测像素 (最近邻), 两图共享「零级为中心」约定"
    ),
}


def setup_cjk_font() -> None:
    """给 matplotlib 配中文字体 (找不到就退回默认字体, 不报错)。"""
    matplotlib.rcParams["font.sans-serif"] = list(CJK_FONTS)
    matplotlib.rcParams["axes.unicode_minus"] = False


# --------------------------------------------------------------------------- #
# 模型侧扫描
# --------------------------------------------------------------------------- #
@dataclass
class SweepResult:
    """一次模型侧扫描的完整结果 (供 dry-run 落盘与 --execute 复用)。"""

    limits: OpticalLimits
    checkpoint: Path
    parameters: dict[str, Any]
    camera_xy: np.ndarray
    centre_ij: tuple[int, int]
    zero_order_xy: tuple[float, float]
    sample_step_ccd: tuple[float, float]
    fractions: tuple[float, ...]
    sizes: tuple[float, ...]
    points: list[SizeMetrics]
    phases: dict[float, np.ndarray]
    model_images: dict[float, np.ndarray]
    final_losses: dict[float, float]
    steps: int
    lr: float
    cap: float
    zero_order_radius_px: float
    inner_shrink_px: int
    noise_floor: float | None
    valid_k: float
    ref_frame_dir: Path | None
    ref_zero_rc: tuple[int, int] | None
    ref_aligned: np.ndarray | None


def parse_float_list(spec: str) -> tuple[float, ...]:
    """解析逗号分隔的数字列表 (分号也容忍)。"""
    values = [float(v) for v in spec.replace(";", ",").split(",") if v.strip()]
    if not values:
        raise ValueError(f"数字列表为空: {spec!r}")
    return tuple(values)


def parse_fractions(spec: str | None) -> tuple[float, ...] | None:
    """解析逗号分隔的分数列表; ``None`` 用默认 0.05–0.80。"""
    if spec is None:
        return None
    return parse_float_list(spec)


def load_checkpoint(checkpoint: Path) -> tuple[Any, np.ndarray]:
    """读检查点: 冻结参数 + 相机采样网格 (121×121×2, 2 px 步长)。"""
    from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
        load_frozen_parameters,
    )

    with np.load(checkpoint / "inputs.npz") as payload:
        camera_xy = np.array(payload["camera_xy"], dtype=np.float64)
    parameters = load_frozen_parameters(checkpoint)
    return parameters, camera_xy


def _flat_model_image(forward: Any) -> np.ndarray:
    """零相位下的模型远场 (相机采样网格)。"""
    grid_size = int(forward.params.grid_size)
    image = forward.predict(np.zeros((1, grid_size, grid_size), dtype=np.float32))
    if image.ndim == 3:
        image = image[0]
    return image.detach().cpu().numpy()


def _model_centre(
    forward: Any,
    camera_xy: np.ndarray,
    override_xy: tuple[float, float] | None,
    flat_image: np.ndarray | None = None,
) -> tuple[tuple[int, int], tuple[float, float]]:
    """模型采样网格上的零级位置。

    缺省: 在**零相位模型图**上 argmax（零级 = 全局最大, 不用几何中心猜）；
    给定: 用 ``sample_grid_index`` 把相机坐标映回网格下标。
    """
    if override_xy is not None:
        col, row = forward.sample_grid_index(float(override_xy[0]), float(override_xy[1]))
        return (int(col), int(row)), (float(override_xy[0]), float(override_xy[1]))
    if flat_image is None:
        flat_image = _flat_model_image(forward)
    row, col = np.unravel_index(int(np.argmax(flat_image)), flat_image.shape)
    return (int(col), int(row)), (float(camera_xy[row, col, 0]), float(camera_xy[row, col, 1]))


def load_reference_frame(ref_dir: Path, label: str) -> tuple[np.ndarray, tuple[int, int]]:
    """读取 ``ref_dir`` 下 ``p*_<label>_f*.npy`` 全部帧取平均, 去本底后 argmax 定位零级。

    返回:
        (去本底后的平均帧, 零级 (row, col))
    """
    from ao_shaping.utils.image.beam_metrics import prepare_frame

    files = sorted(ref_dir.glob(f"p*_{label}_f*.npy"))
    if not files:
        raise FileNotFoundError(f"{ref_dir} 下找不到实测参考帧 (p*_{label}_f*.npy)")
    frames = [np.asarray(np.load(f), dtype=np.float64) for f in files]
    frame = np.mean(np.stack(frames, axis=0), axis=0)
    frame = prepare_frame(frame)
    row, col = np.unravel_index(int(np.argmax(frame)), frame.shape)
    return frame, (int(row), int(col))


def align_reference_to_model(
    ref_frame: np.ndarray,
    ref_zero_rc: tuple[int, int],
    camera_xy: np.ndarray,
    centre_ij: tuple[int, int],
) -> np.ndarray:
    """把实测参考帧对齐到模型采样网格 (同窗)。

    每个模型采样 (r, c) 对应的实测像素 = 实测零级 + (该采样的相机坐标 − 模型零级的
    相机坐标)，取最近邻。两图由此共享「零级为中心」的约定 —— 模型网格与实测帧的
    零级位置 (会话漂移) 与采样奇偶 (2 px vs 1 px) 都不同, 只有按零点为原点对齐,
    相关才是同一窗口的比较。
    """
    centre_col, centre_row = centre_ij
    zero_x = float(camera_xy[centre_row, centre_col, 0])
    zero_y = float(camera_xy[centre_row, centre_col, 1])
    ref_row, ref_col = ref_zero_rc
    height, width = camera_xy.shape[:2]
    fh, fw = ref_frame.shape
    out = np.zeros((height, width), dtype=np.float64)
    dx = np.round(camera_xy[:, :, 0] - zero_x).astype(np.int64)
    dy = np.round(camera_xy[:, :, 1] - zero_y).astype(np.int64)
    cols = ref_col + dx
    rows = ref_row + dy
    ok = (rows >= 0) & (rows < fh) & (cols >= 0) & (cols < fw)
    out[ok] = ref_frame[rows[ok], cols[ok]]
    return out


def compute_metrics(
    arr: np.ndarray,
    base_arr: np.ndarray,
    box: np.ndarray,
    core: np.ndarray,
    zero_disk: np.ndarray,
    ref_mask: np.ndarray,
    noise_floor: float,
    k: float,
) -> dict[str, float | bool]:
    """防退化指标 (模型侧 / 实测侧同一口径; 分子/分母见 ``METRIC_DEFINITIONS``)。

    核心区内没有光时 (``mean ≤ 1e-12``) CV 记 1e3 —— 显式的最差值, 不是静默 0
    (0 会让一个空盒拿到最高均匀度分)。
    """
    ref_e = float(arr[ref_mask].sum())
    ref_e = ref_e if ref_e > 0.0 else 1e-12
    ref_b = float(base_arr[ref_mask].sum())
    ref_b = ref_b if ref_b > 0.0 else 1e-12
    efficiency = float(arr[box].sum()) / ref_e
    efficiency_baseline = float(base_arr[box].sum()) / ref_b
    values = arr[core]
    mean_core = float(values.mean())
    if mean_core > 1e-12:
        uniformity = float(np.sqrt(((values / mean_core) - 1.0) ** 2).mean())
    else:
        uniformity = 1e3
    threshold = k * noise_floor
    fill = float((arr[box] >= threshold).mean())
    zero_abs = float(arr[zero_disk].sum())
    zero_ratio = zero_abs / ref_e
    return {
        "efficiency": efficiency,
        "efficiency_baseline": efficiency_baseline,
        "efficiency_vs_baseline": efficiency - efficiency_baseline,
        "uniformity": uniformity,
        "mean_core": mean_core,
        "fill": fill,
        "valid": is_valid(mean_core, noise_floor, k=k),
        "zero_order_ratio": min(max(zero_ratio, 0.0), 1.0),
        "zero_abs": zero_abs,
    }


def run_model_sweep(
    args: argparse.Namespace,
    *,
    camera_xy: np.ndarray | None = None,
    zero_xy: tuple[float, float] | None = None,
) -> SweepResult:
    """模型侧扫描: 每个候选尺寸跑一遍可微均匀度优化, 记防退化指标。

    ``camera_xy`` / ``zero_xy`` 供 ``--execute`` 分支用**实测零级重建的窗口**
    覆盖检查点里的旧窗口; 候选尺寸: ``--sizes`` (显式 px 列表) 优先, 否则
    ``--fractions`` × 实用上限 (可按 ``--limit-sizes`` 截断)。
    """
    import torch

    from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
        DifferentiableUniformityOptimizer,
        FrozenDefocusForward,
        UniformityObjective,
        build_masks,
    )

    parameters, ckpt_camera_xy = load_checkpoint(args.checkpoint)
    cam_xy = camera_xy if camera_xy is not None else ckpt_camera_xy
    forward = FrozenDefocusForward(parameters, cam_xy, device="cpu")

    limits = DEFAULT_SYSTEM
    if getattr(args, "sizes", None):
        fractions = default_fractions()  # 仅作记录; 实际尺寸由 --sizes 给定
        candidates = tuple(sorted(float(v) for v in parse_float_list(args.sizes)))
    else:
        fractions = parse_fractions(getattr(args, "fractions", None)) or default_fractions()
        candidates = limits.candidate_sizes_px(fractions)
        if args.limit_sizes > 0:
            candidates = candidates[: args.limit_sizes]

    # 零相位基线图 (一次计算, 所有尺寸共用) 与零级位置
    flat_image = _flat_model_image(forward)
    centre, anchor_xy = _model_centre(forward, cam_xy, zero_xy, flat_image)

    # 模型侧噪声地板: 显式值 > 冻结前向的 background 参数 > 平场图 1% 分位
    noise_floor = args.noise_floor
    if noise_floor is None:
        bg = getattr(parameters, "background", None)
        if bg is None:
            to_dict = getattr(parameters, "to_dict", None)
            if callable(to_dict):
                bg = to_dict().get("background")
        noise_floor = float(bg) if bg else 0.0
    if noise_floor <= 0.0:
        noise_floor = float(np.percentile(flat_image, 1))

    # fidelity_proxy 参考帧 (已采实测 A1 帧, 同窗对齐; --no-ref-frame 时跳过)
    ref_frame_dir: Path | None = None
    ref_zero_rc: tuple[int, int] | None = None
    ref_aligned: np.ndarray | None = None
    if not args.no_ref_frame:
        ref_frame_dir = Path(args.ref_frame_dir)
        ref_frame, ref_zero_rc = load_reference_frame(ref_frame_dir, args.ref_label)
        ref_aligned = align_reference_to_model(ref_frame, ref_zero_rc, cam_xy, centre)

    step = forward.sample_step_ccd()
    half_step = 0.5 * (step[0] + step[1])
    radius_samples = args.zero_order_radius_px / half_step
    height, width = cam_xy.shape[:2]
    yy, xx = np.mgrid[:height, :width]
    zero_disk = (xx - centre[0]) ** 2 + (yy - centre[1]) ** 2 <= radius_samples**2
    # 模型侧参考窗 = 整个采样网格 (与实测侧 241×241 窗同几何口径)
    ref_mask = np.ones((height, width), dtype=bool)

    grid_size = int(forward.params.grid_size)
    seed = np.zeros((grid_size, grid_size), dtype=np.float32)

    points: list[SizeMetrics] = []
    phases: dict[float, np.ndarray] = {}
    images: dict[float, np.ndarray] = {}
    losses: dict[float, float] = {}
    for size in candidates:
        box, core = build_masks(
            cam_xy.shape[:2], centre, int(round(size)), args.inner_shrink_px, step
        )
        objective = UniformityObjective(box, core, w_phase=0.01)
        optimizer = DifferentiableUniformityOptimizer(
            forward,
            objective,
            seed.copy(),
            lr=args.lr,
            step_rms_cap=args.cap,
            device="cpu",
        )
        for _ in range(args.steps):
            optimizer.update()
        phase = optimizer.current_phase
        image = forward.predict(
            torch.as_tensor(phase[None], dtype=torch.float32, device=forward._device)
        )
        if image.ndim == 3:
            image = image[0]
        arr = image.detach().cpu().numpy()
        m = compute_metrics(arr, flat_image, box, core, zero_disk, ref_mask, noise_floor, args.valid_k)
        fidelity = None
        if ref_aligned is not None:
            fidelity = pearson_correlation(arr, ref_aligned)
        point = SizeMetrics(
            size_px=float(size),
            efficiency=m["efficiency"],
            uniformity=m["uniformity"],
            zero_order_ratio=m["zero_order_ratio"],
            fidelity=fidelity,
            target_type="square",
            efficiency_baseline=m["efficiency_baseline"],
            efficiency_vs_baseline=m["efficiency_vs_baseline"],
            zero_abs=max(0.0, m["zero_abs"]),
            mean_core=max(0.0, m["mean_core"]),
            fill=m["fill"],
            valid=bool(m["valid"]),
        )
        points.append(point)
        key = float(size)
        phases[key] = phase
        images[key] = arr
        losses[key] = float(optimizer.last_loss) if optimizer.last_loss is not None else float("nan")
        fid_str = "NA" if fidelity is None else f"{fidelity:.3f}"
        logger.info(
            "size={:8.1f}px  eff={:.4f} (vs_base={:+.4f})  CV={:.4f}  mean_core={:.4f}  "
            "fill={:.3f}  valid={}  zero={:.4f} (abs={:.3f})  fid_proxy={}  loss={:.5f}",
            size,
            m["efficiency"],
            m["efficiency_vs_baseline"],
            m["uniformity"],
            m["mean_core"],
            m["fill"],
            m["valid"],
            m["zero_order_ratio"],
            m["zero_abs"],
            fid_str,
            losses[key],
        )
    return SweepResult(
        limits=limits,
        checkpoint=Path(args.checkpoint),
        parameters=parameters.to_dict(),
        camera_xy=cam_xy,
        centre_ij=centre,
        zero_order_xy=anchor_xy,
        sample_step_ccd=step,
        fractions=fractions,
        sizes=candidates,
        points=points,
        phases=phases,
        model_images=images,
        final_losses=losses,
        steps=int(args.steps),
        lr=float(args.lr),
        cap=float(args.cap),
        zero_order_radius_px=float(args.zero_order_radius_px),
        inner_shrink_px=int(args.inner_shrink_px),
        noise_floor=float(noise_floor),
        valid_k=float(args.valid_k),
        ref_frame_dir=ref_frame_dir,
        ref_zero_rc=ref_zero_rc,
        ref_aligned=ref_aligned,
    )


def analyse(
    points: list[SizeMetrics], weight_uniformity: float, weight_efficiency: float
) -> tuple[
    tuple[float | None, float | None, float | None, list[tuple[str, str]], str],
    tuple[float | None, float | None, float],
]:
    """峰值窗口 + 单峰拟合 (两个判决都保留, 不互相替代)。

    窗口可能为 ``(None, None, None, ...)`` + 显式说明 (无有效点 / 指标全部退化);
    单峰拟合在有效点 <3 时不再抛错, 返回显式 ``None`` 顶点。
    """
    window = select_best_window(
        points, weight_uniformity=weight_uniformity, weight_efficiency=weight_efficiency
    )
    try:
        fit = fit_optimum(points)
    except ValueError:
        fit = (None, None, 0.0)  # 有效点 <3: 显式「无法拟合」, 不是假顶点
    return window, fit


def _minmax_local(x: np.ndarray) -> np.ndarray:
    """min-max 归一 (常数列 → 全 0); 仅本脚本画图的辅助, 指标口径在 target_size 模块。"""
    x = np.asarray(x, dtype=np.float64)
    span = float(x.max() - x.min())
    if span <= 0.0:
        return np.zeros_like(x)
    return (x - x.min()) / span


def best_balance_size(valid_points: list[SizeMetrics]) -> float:
    """``constraint="best_balance"`` 的取值: (1 − minmax(CV)) × minmax(efficiency) 的 argmax。"""
    cv = np.array([p.uniformity for p in valid_points], dtype=np.float64)
    eff = np.array([p.efficiency for p in valid_points], dtype=np.float64)
    scores = (1.0 - _minmax_local(cv)) * _minmax_local(eff)
    return float(valid_points[int(np.argmax(scores))].size_px)


def tradeoff_notes(points: list[SizeMetrics]) -> str:
    """有效/无效尺寸的取舍说明 (写进标定表 notes)。"""
    valid_pts = [p for p in points if p.valid]
    if not valid_pts:
        return (
            "取舍: 该区间内没有有效点 (所有尺寸 mean_core < k×噪声地板, 没有可填满的"
            "目标) —— 需增大照明/曝光或改小目标"
        )
    best_cv = min(valid_pts, key=lambda p: p.uniformity)
    largest = max(valid_pts, key=lambda p: p.size_px)
    return (
        f"取舍: {best_cv.size_px:.0f} px 均匀性最好 (CV={best_cv.uniformity:.3f}), "
        f"{largest.size_px:.0f} px 填得最大 (fill={largest.fill:.3f}, 最大有效尺寸); "
        f"recommend() 默认 constraint='largest_valid' → {largest.size_px:.0f} px"
    )


def build_calibration(
    result: SweepResult,
    window: tuple[float | None, float | None, float | None, list[tuple[str, str]], str],
    fit: tuple[float | None, float | None, float],
    *,
    method: str,
    notes: str,
) -> TargetSizeCalibration:
    """把扫描结果装成查找表。

    ``best_size`` 优先级: 单峰顶点 → 窗口峰值 → 最大有效尺寸 (窗口被拒斥时的
    显式回退)。窗口为 ``None`` 时不写 ``windows`` (推荐走 ``recommend()`` 的
    有效点约束路径)。
    """
    lo, hi, best = window[0], window[1], window[2]
    best_size: float | None = None
    if fit[0] is not None:
        best_size = float(fit[0])
    elif best is not None:
        best_size = float(best)
    else:
        valid_pts = [p for p in result.points if p.valid]
        if valid_pts:
            best_size = max(float(p.size_px) for p in valid_pts)
    win = None if lo is None else (float(lo), float(hi))
    return TargetSizeCalibration(
        signature=result.limits.signature(),
        limits=result.limits,
        points=list(result.points),
        window=win,
        windows={} if lo is None else {"square": (float(lo), float(hi))},
        best_size=best_size,
        method=method,
        notes=notes,
    )


# --------------------------------------------------------------------------- #
# 落盘
# --------------------------------------------------------------------------- #
CSV_FIELDS_MODEL = [
    "size_px",
    "valid",
    "mean_core",
    "fill",
    "efficiency",
    "efficiency_baseline",
    "efficiency_vs_baseline",
    "uniformity",
    "zero_order_ratio",
    "zero_abs",
    "fidelity_proxy",
    "model_final_loss",
]
CSV_FIELDS_HW = [
    "hw_efficiency",
    "hw_efficiency_baseline",
    "hw_efficiency_vs_baseline",
    "hw_uniformity",
    "hw_mean_core",
    "hw_fill",
    "hw_valid",
    "hw_zero_order_ratio",
    "hw_zero_abs",
    "hw_fidelity",
    "hw_slot",
]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def sweep_csv_rows(result: SweepResult) -> list[dict[str, Any]]:
    rows = []
    for point in result.points:
        key = float(point.size_px)
        rows.append(
            {
                "size_px": point.size_px,
                "valid": point.valid,
                "mean_core": point.mean_core,
                "fill": point.fill,
                "efficiency": point.efficiency,
                "efficiency_baseline": point.efficiency_baseline,
                "efficiency_vs_baseline": point.efficiency_vs_baseline,
                "uniformity": point.uniformity,
                "zero_order_ratio": point.zero_order_ratio,
                "zero_abs": point.zero_abs,
                "fidelity_proxy": point.fidelity,
                "model_final_loss": result.final_losses.get(key),
            }
        )
    return rows


def sweep_json_payload(
    result: SweepResult,
    window: tuple[float | None, float | None, float | None, list[tuple[str, str]], str],
    fit: tuple[float | None, float | None, float],
    *,
    mode: str,
    calibration_path: Path | None,
    hw_points: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    lo, hi, best, degenerate, message = window
    valid_pts = [p for p in result.points if p.valid]
    recommend: dict[str, float] = {}
    if valid_pts:
        recommend = {
            "largest_valid": max(float(p.size_px) for p in valid_pts),
            "best_balance": best_balance_size(valid_pts),
        }
    payload: dict[str, Any] = {
        "mode": mode,
        "checkpoint": str(result.checkpoint),
        "limits": {
            "wavelength_m": result.limits.wavelength_m,
            "focal_m": result.limits.focal_m,
            "slm_pixels": list(result.limits.slm_pixels),
            "pixel_pitch_m": result.limits.pixel_pitch_m,
            "camera_pixel_m": result.limits.camera_pixel_m,
            "camera_shape": list(result.limits.camera_shape),
        },
        "signature": result.limits.signature(),
        "parameters": result.parameters,
        "fractions": list(result.fractions),
        "candidate_sizes_px": [float(v) for v in result.sizes],
        "centre_ij": list(result.centre_ij),
        "zero_order_xy": list(result.zero_order_xy),
        "sample_step_ccd": list(result.sample_step_ccd),
        "zero_order_radius_px": result.zero_order_radius_px,
        "inner_shrink_px": result.inner_shrink_px,
        "noise_floor_model": result.noise_floor,
        "valid_k": result.valid_k,
        "ref_frame_dir": None if result.ref_frame_dir is None else str(result.ref_frame_dir),
        "ref_zero_rc": None if result.ref_zero_rc is None else list(result.ref_zero_rc),
        "metric_definitions": METRIC_DEFINITIONS,
        "points": [p.to_dict() for p in result.points],
        "final_losses": {str(k): v for k, v in result.final_losses.items()},
        "window": None if lo is None else [float(lo), float(hi), float(best)],
        "window_message": message,
        "degenerate": [[name, reason] for name, reason in degenerate],
        "fit_optimum": None if fit is None else list(fit),
        "recommend": recommend,
        "calibration_path": None if calibration_path is None else str(calibration_path),
    }
    if hw_points is not None:
        payload["hw_points"] = hw_points
    return payload


def plot_curves(
    result: SweepResult,
    window: tuple[float | None, float | None, float | None, list[tuple[str, str]], str],
    fit: tuple[float | None, float | None, float],
    path: Path,
    *,
    weight_uniformity: float = 0.5,
    weight_efficiency: float = 0.5,
) -> None:
    """尺寸 → 分数 / 指标 曲线 (峰值平台 + 单峰顶点; 无效点用 × 标出)。"""
    setup_cjk_font()
    lo, hi, best, degenerate, message = window
    valid_pts = [p for p in result.points if p.valid]
    sizes_all = [p.size_px for p in result.points]
    sizes = [p.size_px for p in valid_pts]
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.2), sharex=True)
    ax = axes[0]
    scores = None
    if valid_pts:
        deg_names = {name for name, _ in degenerate}
        w_u = 0.0 if "uniformity" in deg_names else weight_uniformity
        w_e = 0.0 if "efficiency" in deg_names else weight_efficiency
        if w_u + w_e > 0.0:
            scores = composite_scores(valid_pts, w_u, w_e)
            ax.plot(sizes, scores, "o-", color="#1f77b4", label="合成分数 (均匀度 + 效率增量)")
    if lo is not None:
        ax.axvspan(lo, hi, color="#27ae60", alpha=0.18, label=f"峰值平台 {lo:.0f}–{hi:.0f} px")
    if fit is not None and fit[0] is not None and scores is not None:
        nearest = int(np.argmin([abs(s - fit[0]) for s in sizes]))
        ax.plot(
            [fit[0]],
            [float(scores[nearest])],
            "*",
            color="#d62728",
            markersize=16,
            label=f"单峰顶点 {fit[0]:.0f} px (r²={fit[2]:.2f})",
        )
    for p in result.points:
        if not p.valid:
            ax.plot([p.size_px], [0.0], "x", color="#95a5a6", markersize=9, label=None)
    if any(not p.valid for p in result.points):
        ax.plot([], [], "x", color="#95a5a6", markersize=9, label="无效点 (mean_core < k×地板)")
    ax.set_ylabel("合成分数")
    ax.set_title("目标尺寸扫描: 分数与平台 (模型侧)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.plot(
        sizes_all,
        [p.efficiency_vs_baseline for p in result.points],
        "s-",
        color="#2f6fb0",
        label="效率增量 (vs 未优化基线)",
    )
    ax.plot(
        sizes_all,
        [p.zero_order_ratio for p in result.points],
        "^--",
        color="#7f8c8d",
        label="零级占比 (固定盘)",
    )
    ax2 = ax.twinx()
    ax2.plot(
        sizes_all,
        [min(p.uniformity, 12.0) for p in result.points],
        "o-",
        color="#c0392b",
        label="CV (均匀度, 越小越好)",
    )
    ax2.plot(sizes_all, [p.fill for p in result.points], "d:", color="#16a085", label="fill (点亮占比)")
    ax2.set_ylabel("CV / fill")
    ax.set_xlabel("目标边长 / 相机像元")
    ax.set_ylabel("效率增量 / 零级占比")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="center left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def print_summary(
    result: SweepResult,
    window: tuple[float | None, float | None, float | None, list[tuple[str, str]], str],
    fit: tuple[float | None, float | None, float],
    calibration_path: Path | None,
) -> None:
    lo, hi, best, degenerate, message = window
    valid_pts = [p for p in result.points if p.valid]
    print()
    print("=" * 72)
    print("光学界限 (解析):")
    print(f"  焦面单像素 Δx = λf/(N_x·p) = {result.limits.far_field_sample_m * 1e6:.4f} µm")
    print(f"  最大无混叠视场 FOV = λf/p   = {result.limits.slm_fov_m * 1e3:.4f} mm")
    print(f"  实用上限 = min(FOV, 传感器短边) = {result.limits.practical_max_px:.1f} px (相机像元)")
    print(f"  系统签名 = {result.limits.signature()}")
    print(
        f"零级锚点: xy={tuple(round(v, 2) for v in result.zero_order_xy)} "
        f"网格下标 {result.centre_ij}, 采样步长 {tuple(round(v, 3) for v in result.sample_step_ccd)} px"
    )
    print(f"模型噪声地板 (valid = mean_core ≥ k={result.valid_k:g}×地板): {result.noise_floor}")
    if result.ref_frame_dir is not None:
        print(f"fidelity_proxy 参考帧: {result.ref_frame_dir} (零级 {result.ref_zero_rc})")
    else:
        print("fidelity_proxy 参考帧: 未加载 (--no-ref-frame)")
    print("候选尺寸 (相机像元): " + ", ".join(f"{v:.1f}" for v in result.sizes))
    print(f"有效点: {len(valid_pts)}/{len(result.points)} 个 (空盒点已标记无效, 不参与窗口选择)")
    if lo is not None:
        print(f"峰值窗口 select_best_window: [{lo:.1f}, {hi:.1f}] px, 峰值 {best:.1f} px")
    else:
        print(f"峰值窗口: 已拒斥 —— {message}")
    if degenerate:
        for name, reason in degenerate:
            print(f"  [退化指标] {name}: {reason}")
    else:
        print("  [退化指标] 无 —— 全部指标有区分度")
    if fit[0] is None:
        print(f"单峰拟合 fit_optimum: 不呈单峰 (r²={fit[2]:.3f}) —— 顶点显式 None")
    else:
        print(
            f"单峰拟合 fit_optimum: 顶点 {fit[0]:.1f} px, 半宽 {fit[1]:.1f} px, r²={fit[2]:.3f}"
        )
    if valid_pts:
        print(
            f"推荐 recommend(): largest_valid={max(p.size_px for p in valid_pts):.1f} px, "
            f"best_balance={best_balance_size(valid_pts):.1f} px"
        )
    if calibration_path is not None:
        print(f"标定表已写入: {calibration_path}")
    print("=" * 72)


# --------------------------------------------------------------------------- #
# dry-run (默认): 模型侧扫描, 不 import 任何驱动
# --------------------------------------------------------------------------- #
def run_dry(args: argparse.Namespace) -> int:
    setup_cjk_font()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_root) / stamp
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    print(f"运行目录: {out_dir}")
    print(f"检查点: {args.checkpoint}")

    zero_xy = (
        (float(args.zero_order_x), float(args.zero_order_y))
        if args.zero_order_x is not None and args.zero_order_y is not None
        else DEFAULT_ZERO_ORDER_XY
    )
    try:
        result = run_model_sweep(args, zero_xy=zero_xy)
    except (FileNotFoundError, KeyError, ValueError, OSError) as exc:
        logger.error("模型侧扫描失败: {}", exc)
        return EXIT_MODEL_FAILED

    try:
        window, fit = analyse(result.points, args.w_uniformity, args.w_efficiency)
    except ValueError as exc:
        logger.error("曲线分析失败 (样本不足?): {}", exc)
        write_csv(out_dir / "sweep.csv", sweep_csv_rows(result), CSV_FIELDS_MODEL)
        return EXIT_MODEL_FAILED

    notes = (
        f"checkpoint={result.checkpoint}; steps={result.steps}; lr={result.lr}; "
        f"cap={result.cap}; zero_order={result.zero_order_xy}; "
        f"noise_floor={result.noise_floor:.6g}; k={result.valid_k:g}; "
        + tradeoff_notes(result.points)
    )
    calibration = build_calibration(result, window, fit, method="model-only sweep", notes=notes)
    calibration_path: Path | None = None
    if not args.no_calibration:
        calibration_path = calibration.save(args.calibration_path)

    write_csv(out_dir / "sweep.csv", sweep_csv_rows(result), CSV_FIELDS_MODEL)
    payload = sweep_json_payload(
        result, window, fit, mode="model-only dry-run", calibration_path=calibration_path
    )
    (out_dir / "sweep.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    plot_curves(
        result,
        window,
        fit,
        fig_dir / f"{stamp}_curve.png",
        weight_uniformity=args.w_uniformity,
        weight_efficiency=args.w_efficiency,
    )
    print_summary(result, window, fit, calibration_path)
    return EXIT_OK


# --------------------------------------------------------------------------- #
# --execute: 模型侧 + 设备侧两态采集
# --------------------------------------------------------------------------- #
def grid_phase_to_panel_phase(
    grid_phase: np.ndarray, panel_shape: tuple[int, int] = PANEL_SHAPE
) -> np.ndarray:
    """128×128 控制网格相位 → 1200×1920 面板相位 (最近邻上采样, 与
    ``generate_defocus_differentiable_report.panel_to_grid`` 互逆: 网格第 i 格
    覆盖面板 [15i, 15i+15) 像素块, 取前 1200 行 = 照明区 y<1200)。"""
    phase = np.asarray(grid_phase, dtype=np.float32)
    grid_side = int(phase.shape[0])
    if phase.shape != (grid_side, grid_side):
        raise ValueError(f"控制网格必须为方阵, 得到 {phase.shape}")
    pitch = panel_shape[1] // grid_side
    if pitch * grid_side != panel_shape[1]:
        raise ValueError(f"网格边长 {grid_side} 不能整除面板宽 {panel_shape[1]}")
    big = np.kron(phase, np.ones((pitch, pitch), dtype=np.float32))
    return big[: panel_shape[0], : panel_shape[1]]


@contextmanager
def guarded_device(
    label: str,
    factory: Callable[[], Any],
    *,
    open_timeout_s: float = OPEN_TIMEOUT_S,
    close_timeout_s: float = CLOSE_TIMEOUT_S,
) -> Iterator[Any]:
    """打开一个设备并在退出时关闭它, **打开与关闭都套超时保护** (模式同
    ``hw_differentiable_phase_opt.guarded_device``): Santec 的 SDK 等待在 C 层,
    Python 侧超时必须硬退出, 否则陪着一个假死进程。"""
    device = factory()
    guarded(f"{label} open()", device.open, timeout_s=open_timeout_s)
    try:
        yield device
    finally:
        guarded(f"{label} close()", device.close, timeout_s=close_timeout_s)


def acquire_mean_frame(cam: Any, *, frames: int, discard: int) -> np.ndarray:
    """丢 ``discard`` 帧后平均 ``frames`` 帧 (整段套超时保护)。"""

    def _run() -> np.ndarray:
        for _ in range(int(discard)):
            cam.get_numpy_image(n_sample=1)
        raw = [np.asarray(cam.get_numpy_image(n_sample=1)) for _ in range(int(frames))]
        return np.mean(np.stack(raw, axis=0), axis=0)

    return guarded("CCD 取帧", _run, timeout_s=ACQUIRE_TIMEOUT_S)


def crop_window(frame: np.ndarray, centre_rc: tuple[int, int], half_px: int) -> np.ndarray:
    """以 (row, col) 为中心裁 (2·half+1)² 的窗; 越界部分补 0 (不改变能量口径)。"""
    height, width = frame.shape
    row, col = centre_rc
    crop = np.zeros((2 * half_px + 1, 2 * half_px + 1), dtype=np.float64)
    r0 = row - half_px
    c0 = col - half_px
    src_r0 = max(r0, 0)
    src_c0 = max(c0, 0)
    dst_r0 = max(-r0, 0)
    dst_c0 = max(-c0, 0)
    h = min(height - src_r0, 2 * half_px + 1 - dst_r0)
    w = min(width - src_c0, 2 * half_px + 1 - dst_c0)
    if h > 0 and w > 0:
        crop[dst_r0 : dst_r0 + h, dst_c0 : dst_c0 + w] = frame[src_r0 : src_r0 + h, src_c0 : src_c0 + w]
    return crop


def pearson_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """两个等长展平向量的 Pearson 相关 (常数向量给 0, 不是 NaN)。"""
    va = np.asarray(a, dtype=np.float64).ravel()
    vb = np.asarray(b, dtype=np.float64).ravel()
    if va.shape != vb.shape or va.size < 3:
        raise ValueError("fidelity 对比需要等长、≥3 个样本的向量")
    sa = va - va.mean()
    sb = vb - vb.mean()
    denom = float(np.sqrt((sa**2).sum()) * np.sqrt((sb**2).sum()))
    if denom <= 1e-18:
        return 0.0
    return float((sa * sb).sum() / denom)


def run_execute(args: argparse.Namespace) -> int:
    """真正开设备: 实测零级 → 重建窗口 → 模型扫描 → 逐尺寸两态采集 → 落盘。"""
    # ---- 设备 import 全部锁在 --execute 分支内 -----------------------------
    from ao_shaping.drivers.ccd.common import create_camera
    from ao_shaping.drivers.slm.santec import Santec, SlotRotator
    from ao_shaping.utils.image.beam_metrics import prepare_frame
    from ao_shaping.utils.slm.phase_display import phase_to_slm_grayscale

    from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
        build_masks,
        recentre_window,
    )

    setup_cjk_font()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_root) / stamp
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    print(f"运行目录: {out_dir}")

    # ---- P0.5: 等设备空出来 (公平仲裁, 不抢) --------------------------------
    wait_s = float(getattr(args, "wait_until_idle", 0.0) or 0.0)
    if wait_s > 0.0:
        from scripts.slm.run_hw_sweep_groups import wait_until_idle

        logger.info("等待设备空闲 (预算 {:.0f}s) ...", wait_s)
        code, message = wait_until_idle(
            wait_s,
            slm_number=int(args.slm_number),
            cam_id=int(args.cam_id),
            cam_type=str(args.cam_type),
            exposure_ms=float(args.exposure_ms),
            # ⚠️ 只按 pid 排除自己, 不传 own_paths (会把同一脚本的另一个会话误判成自己)
            own_pids=(os.getpid(),),
        )
        if code != 0:
            logger.error("设备未等到空闲: {}", message)
            return EXIT_BENCH_BUSY
        logger.info("设备空闲, 开始采集")

    # ---- P1: 打开设备 ------------------------------------------------------
    slm_factory = lambda: Santec(  # 工厂闭包, 延迟求值
        slm_number=int(args.slm_number), video_mode=0, wavelength=int(args.slm_wavelength)
    )
    cam_factory = lambda: create_camera(          args.cam_type, int(args.cam_id), exposure_time_ms=float(args.exposure_ms)
    )

    with guarded_device("SLM", slm_factory) as slm:
        slm_serial = slm.get_serial_number()
        logger.info("SLM #{} 序列号: {}", args.slm_number, slm_serial)
        with guarded_device("CCD", cam_factory) as cam:
            logger.info("CCD 序列号: {}", getattr(cam, "sn", None))
            rotator = SlotRotator(slm)

            def display_panel_phase(phase_rad: np.ndarray) -> int:
                """优化相位 → 面板相位 → 灰度 → 换槽上屏。"""
                panel = grid_phase_to_panel_phase(phase_rad)
                gray = phase_to_slm_grayscale(panel, slm=slm)
                return rotator.display_data(slm, gray, float(args.settle_s))

            def display_flat() -> int:
                """平场: raw 灰度直发 (严禁走 create_phase_from_array)。"""
                gray = np.full(PANEL_SHAPE, DEFAULT_FLAT_GRAY, dtype=np.uint16)
                return rotator.display_data(slm, gray, float(args.settle_s))

            # ---- P2: 平场 → argmax 定位零级 (不用几何中心猜) ----------------
            display_flat()
            flat_frame = acquire_mean_frame(cam, frames=args.frames, discard=args.discard)
            prepared = prepare_frame(flat_frame)
            zero_row, zero_col = (int(v) for v in np.unravel_index(int(np.argmax(prepared)), prepared.shape))
            zero_xy = (float(zero_col), float(zero_row))
            logger.info("实测零级: (x, y) = ({:.1f}, {:.1f}) 相机像元", *zero_xy)
            np.savez_compressed(
                out_dir / "flat_frame.npz", frame=np.asarray(flat_frame, dtype=np.float32)
            )

            # ---- P3: 以实测零级重建模型采样窗, 重跑模型扫描 ------------------
            window_xy = recentre_window(
                zero_xy, half_span_px=DEFAULT_HALF_SPAN_PX, step_px=DEFAULT_WINDOW_STEP_PX
            )
            result = run_model_sweep(args, camera_xy=window_xy, zero_xy=zero_xy)
            window, fit = analyse(result.points, args.w_uniformity, args.w_efficiency)
            logger.info(
                "模型侧判决: 窗口 {} 单峰顶点 {}",
                (
                    f"[{window[0]:.1f}–{window[1]:.1f}] px"
                    if window[0] is not None
                    else f"拒斥 ({window[4]})"
                ),
                f"{fit[0]:.1f} px" if fit[0] is not None else "None (不呈单峰)",
            )

            # ---- P4: 逐尺寸两态采集 (平场 = A 态 / 优化相位 = B 态) ----------
            # 实测窗 = 零级中心 (2·half+1)² 像元, 即**固定参考窗** (241×241);
            # 与模型窗 (121 采样 × 2 px) 同几何: 模型第 k 个采样 ↔ 像素偏移 2k,
            # 所以同 box_px 在两套网格上物理尺寸一致。
            half_px = int(DEFAULT_HALF_SPAN_PX)  # 120 px 半宽 → 241×241 实测参考窗
            centre_rc = (zero_row, zero_col)
            step_hw = (1.0, 1.0)
            side = 2 * half_px + 1
            yy, xx = np.mgrid[:side, :side]
            centre_i = half_px
            zero_disk_hw = (xx - centre_i) ** 2 + (yy - centre_i) ** 2 <= args.zero_order_radius_px**2
            ref_mask_hw = np.ones((side, side), dtype=bool)  # 实测参考窗 = 整个 241×241 窗

            display_flat()
            frame_a = acquire_mean_frame(cam, frames=args.frames, discard=args.discard)
            crop_a = prepare_frame(crop_window(frame_a, centre_rc, half_px))
            # 实测噪声地板: 去本底后背景 ≈ N(0, σ), 取背景像素 (低于 25% 分位) 的标准差
            q25 = float(np.percentile(crop_a, 25))
            bg_mask = crop_a < q25
            noise_floor_hw = float(np.std(crop_a[bg_mask])) if bg_mask.any() else 0.0
            if noise_floor_hw <= 0.0:
                noise_floor_hw = float(np.std(crop_a))
            logger.info("实测噪声地板 (平场背景 σ): {:.6g}", noise_floor_hw)

            hw_points: list[dict[str, Any]] = []
            for point in result.points:
                size = float(point.size_px)
                box, core = build_masks(
                    (side, side),
                    (centre_i, centre_i),
                    int(round(size)),
                    args.inner_shrink_px,
                    step_hw,
                )
                slot = display_panel_phase(result.phases[size])
                frame_b = acquire_mean_frame(cam, frames=args.frames, discard=args.discard)
                np.savez_compressed(
                    out_dir / f"frame_b_{size:.0f}px.npz",
                    frame=np.asarray(frame_b, dtype=np.float32),
                )
                crop = prepare_frame(crop_window(frame_b, centre_rc, half_px))
                m = compute_metrics(crop, crop_a, box, core, zero_disk_hw, ref_mask_hw, noise_floor_hw, args.valid_k)
                m_a = compute_metrics(crop_a, crop_a, box, core, zero_disk_hw, ref_mask_hw, noise_floor_hw, args.valid_k)
                model_image = result.model_images[size]
                # 模型窗 (121 采样, 2 px 步长) ↔ 实测窗 (241 像元, 1 px):
                # 采样 i ↔ 像素 2i, 偶数步长下采样后逐点对齐 (121×121 = 121×121)
                fidelity = pearson_correlation(crop[::2, ::2], model_image)
                hw_points.append(
                    {
                        "size_px": size,
                        "hw_efficiency": m["efficiency"],
                        "hw_efficiency_baseline": m_a["efficiency"],
                        "hw_efficiency_vs_baseline": m["efficiency"] - m_a["efficiency"],
                        "hw_uniformity": m["uniformity"],
                        "hw_mean_core": m["mean_core"],
                        "hw_fill": m["fill"],
                        "hw_valid": m["valid"],
                        "hw_zero_order_ratio": m["zero_order_ratio"],
                        "hw_zero_abs": m["zero_abs"],
                        "hw_fidelity": fidelity,
                        "slot": int(slot),
                    }
                )
                logger.info(
                    "hw size={:8.1f}px  eff={:.4f} (vs_flat={:+.4f})  CV={:.4f}  fill={:.3f}  "
                    "valid={}  zero={:.4f} (abs={:.3f})  fidelity={:.3f}  槽={}",
                    size,
                    m["efficiency"],
                    m["efficiency"] - m_a["efficiency"],
                    m["uniformity"],
                    m["fill"],
                    m["valid"],
                    m["zero_order_ratio"],
                    m["zero_abs"],
                    fidelity,
                    slot,
                )

            # ---- P5: 落盘 (同一份 CSV 加 hw_ 列, 标定表加实测点) ------------
            model_rows = sweep_csv_rows(result)
            for row, hw in zip(model_rows, hw_points):
                row.update({k: v for k, v in hw.items() if k != "size_px"})
            write_csv(out_dir / "sweep.csv", model_rows, CSV_FIELDS_MODEL + CSV_FIELDS_HW)
            payload = sweep_json_payload(
                result,
                window,
                fit,
                mode="model+hardware",
                calibration_path=args.calibration_path,
                hw_points=hw_points,
            )
            (out_dir / "sweep.json").write_text(
                json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            plot_curves(
                result,
                window,
                fit,
                fig_dir / f"{stamp}_curve.png",
                weight_uniformity=args.w_uniformity,
                weight_efficiency=args.w_efficiency,
            )

            hw_points_ts = [
                SizeMetrics(
                    size_px=float(hw["size_px"]),
                    efficiency=float(hw["hw_efficiency"]),
                    uniformity=float(hw["hw_uniformity"]),
                    zero_order_ratio=float(hw["hw_zero_order_ratio"]),
                    fidelity=float(hw["hw_fidelity"]),
                    target_type="square",
                    efficiency_baseline=float(hw["hw_efficiency_baseline"]),
                    efficiency_vs_baseline=float(hw["hw_efficiency_vs_baseline"]),
                    zero_abs=max(0.0, float(hw["hw_zero_abs"])),
                    mean_core=max(0.0, float(hw["hw_mean_core"])),
                    fill=float(hw["hw_fill"]),
                    valid=bool(hw["hw_valid"]),
                )
                for hw in hw_points
            ]
            calibration = build_calibration(
                result,
                window,
                fit,
                method="model+hardware sweep",
                notes=(
                    f"checkpoint={result.checkpoint}; zero_order={result.zero_order_xy}; "
                    f"slm={slm_serial}; frames={args.frames}; "
                    f"hw_noise_floor={noise_floor_hw:.6g}; "
                    + tradeoff_notes(result.points)
                ),
            )
            calibration.points = list(result.points) + hw_points_ts
            calibration_path = calibration.save(args.calibration_path)
            print_summary(result, window, fit, calibration_path)
            return EXIT_OK


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="目标尺寸 (方形目标边长) 自动标定: 模型侧扫描 (默认) + --execute 设备侧验证",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="只跑模型侧扫描, 不 import 任何驱动 (缺省行为)")
    parser.add_argument("--execute", action="store_true",
                        help="模型侧扫描后再开 SLM + 相机做两态采集 (需要设备)")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
                        help="冻结前向检查点目录 (缺省 uniformity_refit_20261022_r1)")
    parser.add_argument("--fractions", type=str, default=None,
                        help="候选分数 (逗号分隔), 缺省 0.05–0.80 步长 0.05")
    parser.add_argument("--sizes", type=str, default=None,
                        help="候选边长显式列表 (相机像元, 逗号分隔, 如 40,60,...,240); "
                             "给出时优先于 --fractions (细扫用)")
    parser.add_argument("--limit-sizes", type=int, default=0,
                        help="只扫前 N 个候选 (0 = 全部; 仅对 --fractions 模式有效)")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                        help="每个尺寸的优化步数")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Adam 学习率")
    parser.add_argument("--cap", type=float, default=DEFAULT_CAP,
                        help="每步相位增量照明加权 RMS 上限 (rad)")
    parser.add_argument("--zero-order-x", type=float, default=None,
                        help="零级 x (相机像元); 缺省用拟合会话零点, 未给时用模型图 argmax 自定位")
    parser.add_argument("--zero-order-y", type=float, default=None,
                        help="零级 y (相机像元)")
    parser.add_argument("--zero-order-radius-px", type=float, default=DEFAULT_ZERO_ORDER_RADIUS_PX,
                        help="固定零级盘半径 (相机像元, 缺省 15; 固定, 不随目标尺寸变化)")
    parser.add_argument("--inner-shrink-px", type=int, default=DEFAULT_INNER_SHRINK_PX,
                        help="核心掩膜内缩量 (相机像元)")
    parser.add_argument("--w-uniformity", type=float, default=0.5,
                        help="合成分数中均匀度项权重")
    parser.add_argument("--w-efficiency", type=float, default=0.5,
                        help="合成分数中效率增量项权重")
    parser.add_argument("--ref-frame-dir", type=Path, default=DEFAULT_REF_FRAME_DIR,
                        help="已采实测参考帧目录 (fidelity_proxy: 模型预测 vs 实测帧相关)")
    parser.add_argument("--ref-label", type=str, default=DEFAULT_REF_LABEL,
                        help="参考帧标签 (p*_<label>_f*.npy, 缺省 A1 = 未优化基线态)")
    parser.add_argument("--no-ref-frame", action="store_true",
                        help="不加载参考帧 (fidelity_proxy 为 None)")
    parser.add_argument("--noise-floor", type=float, default=None,
                        help="噪声地板 (图像单位); 缺省自动: 模型侧 = 冻结前向 background, "
                             "实测侧 = 平场背景标准差")
    parser.add_argument("--valid-k", type=float, default=DEFAULT_VALID_K,
                        help="有效性/点亮阈值系数 k (valid = mean_core ≥ k×噪声地板, 缺省 3)")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT,
                        help="扫描产物根目录 (每次运行建 <UTC 时间戳> 子目录)")
    parser.add_argument("--calibration-path", type=Path, default=DEFAULT_CALIBRATION_PATH,
                        help="标定表 (查找表) 落盘路径")
    parser.add_argument("--no-calibration", action="store_true",
                        help="只写扫描产物, 不写 target_size_calibration.json")
    # --execute 专属
    parser.add_argument("--slm-number", type=int, default=1)
    parser.add_argument("--slm-wavelength", type=int, default=1064)
    parser.add_argument("--cam-type", type=str, default="daheng")
    parser.add_argument("--cam-id", type=int, default=0)
    parser.add_argument("--exposure-ms", type=float, default=DEFAULT_EXPOSURE_MS)
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES,
                        help="每个状态平均的帧数 (两态各一次)")
    parser.add_argument("--discard", type=int, default=DEFAULT_DISCARD,
                        help="每批先丢掉的帧数 (LCOS 翻转稳定)")
    parser.add_argument("--settle-s", type=float, default=DEFAULT_SETTLE_S,
                        help="上屏后等待 (s)")
    parser.add_argument("--wait-until-idle", type=float, default=0.0,
                        help="先等设备空闲的预算 (s, 0 = 不等)")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.execute:
        return run_execute(args)
    return run_dry(args)


if __name__ == "__main__":
    sys.exit(main())


