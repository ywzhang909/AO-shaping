# -*- coding: utf-8 -*-
"""
SLM + CCD 训练数据采集器（面向 ml/hwdataset 语料库）

本模块是 SLM 工作台的训练数据采集器，对标现有 DM+WFS 的
`src/ao_shaping/tools/train_data_collect.py`（后者收集 `_cam_axis`/`_cam_focal`/`_wavefront`）。
本采集器改为产出 `ml/hwdataset` 能直接索引的记录字段：`_img`、`_phase`/`_c`、`exp_t`。

设计目标（针对 report/hwdataset_corpus/report.md 中发现的分布缺口）：
- 填补目标覆盖率（objective coverage）不足（32%）的问题：通过 `build_plan` 在
  `--exposure-ms` × `--cam-size` 的交叉组合中循环采样，刻意覆盖曝光时长与视场尺寸（FOV）两类缺口。
- 提升曝光（exposure）分布稀疏性：显式列出曝光列表并在计划中均匀穿插。
- 丰富自由相位（freeform）网格多样性：支持 `--phase-grid` 控制自由相位面板尺寸（24² 对应 576 长度系数）。
- 增强视场（FOV）多样性：支持 `--cam-size`/`--region` 参数组合，记录 `cam_size`、`region`、`far_field_size` 等元数据。

关键约束（zernike_radius）：
`ml/hwdataset/records.py` 在重建 `PhaseSource.ZERNIKE` 相位时使用硬编码常量
`_ZERNIKE_APERTURE_RADIUS = 300.0`，且**不会**读取 sidecar 中的 `zernike_radius`。
因此：
- 始终将 `zernike_radius` 写入 sidecar（自描述元数据）；
- 在 `--help` 中显式提示该约束；
- 当 `--store-panel` 开启时，额外持久化 uint16 全面板 `_phase`，避免依赖重建以获得忠实回放。

硬件纪律（严格遵守）：
- 相位写入一律使用 `bench_kernels.display_and_average`：它已处理槽位轮换（调用 `display_data()` 且**不传** `memory_number`）
  以及“丢弃前若干帧→稳定性判定→平均”的沉降流程。**绝不**重新实现沉降、`time.sleep` 或槽位选择。
- **绝不**将 uint16 灰度全面板直接传入 `Santec.create_phase_from_array()`（后者将输入视作弧度，会破坏 uint16 值）。
- SLM 仅使用内存模式（`video_mode=0`）；**严禁** DVI 模式（`video_mode=1` 曾导致 120–300 s 挂起）。
- 零阶光斑定位：使用 `ao_shaping.utils.image.beam_metrics.zero_order_center(frame)`，其返回值基于整帧 `argmax`，**不是**帧中心。
  在 dedenoised 的平场帧上**仅定位一次**，随后在全程**冻结** ROI。
- 仅在 `main()` 内导入驱动（`Santec`、`open_camera`）和 `ml.hwdataset`（后者会拉起 torch）。
- 使用 `loguru.logger`，**不要**使用 `print()`。
- 模块首行 `from __future__ import annotations`；仅使用绝对导入。
- 中文注释/文档字符串，与 `tools/slm` 邻近模块风格保持一致。
"""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from loguru import logger

# Canonical imports（模块作用域，均不触发驱动/torch）
from ao_shaping.tools.slm.bench_kernels import (
    display_and_average,
    measure_spot,
    zernike_panel,
)
from ao_shaping.utils.image.beam_metrics import zero_order_center
from ao_shaping.utils.io.file import Recorder, save_recorder_debug_artifacts
from ao_shaping.utils.slm.phase_display import phase_to_slm_grayscale
from ao_shaping.utils.wavefront import zernike_calc, zernike_utils

# 语料家族前缀（最长前缀匹配，顺序极为关键：slm_pib_online 必须排在 slm_pib 之前）
# 参见：src/ml/hwdataset/index.py:200-210
_RECOGNISED_FAMILIES: tuple[str, ...] = (
    "model_in_loop_hw_collect",
    "model_in_loop_hw_sweep",
    "bench_stability",
    "sim_calib_abba",
    "slm_gsnet_square",
    "slm_zernike_shaping",
    "slm_pib_online",
    "slm_pib",
    "recorder_",
)

# 36 长系数在以下家族中视作 FREEFORM（24² 网格），其余一律视作 ZERNIKE（n_max=7）
_FREEFORM_36_FAMILIES: frozenset[str] = frozenset(
    {"slm_gsnet_square", "sim_calib_abba"}
)


@dataclass(frozen=True)
class CollectParams:
    """
    采集参数集合（全部采集开关与硬件元数据）

    字段说明：
    - root_dir：调试产物根目录（最终传入 save_recorder_debug_artifacts）
    - subdir_prefix：文件名前缀，必须是 ml/hwdataset 识别的家族前缀之一
    - samples：计划采样总数（build_plan 根据交叉组合循环生成）
    - mode：相位生成模式：{"zernike", "freeform", "turbulence", "flat"}
    - n_max：Zernike 最大阶数（mode="zernike" 时生效）
    - phase_grid：自由相位网格边长（mode="freeform" 时生效，对应 24²→576）
    - turbulence_count：湍流模式下叠加的正弦分量个数
    - turbulence_kmax：湍流空间频率上限（像素单位的归一化概念）
    - zernike_radius：Zernike 光瞳半径（像素）。**注意**：records.py 硬编码 300.0 用于 ZERNIKE 重建
    - beam_radius：光斑半径估计值（元数据）
    - pupil_center_panel：(cx, cy) 面板坐标（元数据）
    - exposure_ms：曝光时长列表（ms），交叉组合主轴之一
    - cam_size：远场相机视场尺寸列表（像素，如 250,320,512），交叉组合主轴之一
    - region：相机感兴趣区域（元数据）
    - far_field_size：远场图像尺寸（元数据）
    - camera_type：相机类型标识（传递给 open_camera）
    - cam_id：远场相机 ID
    - pupil_cam_id：瞳孔相机 ID（None 表示不采集）
    - slm_number：SLM 编号（元数据）
    - slm_wavelength：波长（nm，元数据）
    - target_shape/target_size：目标光斑形状/尺寸（元数据）
    - algorithm/epochs：优化算法与轮数（元数据）
    - objective：本批次目标标识（元数据）
    - collector：采集器名称（固定为本模块名）
    - store_panel：是否额外保存 uint16 全面板 `_phase`（规避 300.0 半径重建偏差）
    - flush_every：每采集多少条记录就调用一次 save_recorder_debug_artifacts
    - settle_n_frames/settle_n_discard/settle_wait_time_s/settle_stable_tol/settle_max_wait_s：沉降参数
    - seed：随机种子（None 表示不固定）
    - no_hw/dry_run：仅打印计划而不触发硬件
    """

    root_dir: Path = field(default_factory=lambda: Path("data"))
    subdir_prefix: str = "slm_zernike_shaping"
    samples: int = 12
    mode: str = "zernike"
    n_max: int = 7
    phase_grid: int = 24
    turbulence_count: int = 6
    turbulence_kmax: float = 2.0
    zernike_radius: float = 300.0
    beam_radius: float = 150.0
    pupil_center_panel: tuple[float, float] = (600.0, 960.0)
    exposure_ms: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
    cam_size: tuple[int, ...] = (256, 320, 512)
    region: tuple[int, int, int, int] | None = None
    far_field_size: tuple[int, int] | None = None
    camera_type: str = "sim"
    cam_id: int = 0
    pupil_cam_id: int | None = None
    slm_number: int = 1
    slm_wavelength: float = 1064.0
    target_shape: str = "gaussian"
    target_size: float = 50.0
    algorithm: str = "spgd"
    epochs: int = 0
    objective: str = "shape_uniformity"
    collector: str = "slm_train_data_collect"
    store_panel: bool = False
    flush_every: int = 6
    settle_n_frames: int = 4
    settle_n_discard: int = 3
    settle_wait_time_s: float | None = 0.5
    settle_stable_tol: float = 0.02
    settle_max_wait_s: float = 6.0
    seed: int | None = 1234
    no_hw: bool = False
    dry_run: bool = False


@dataclass(frozen=True)
class PlanPoint:
    """
    单个计划采样点

    属性：
    - index：采样序号（0-based）
    - mode：相位模式
    - exposure_ms：本点曝光时长（ms）
    - cam_size：本点视场尺寸（像素）
    - coeffs：Zernike 系数映射（mode="zernike" 时）
    - phase_grid：自由相位网格边长（mode="freeform" 时）
    - turb_count/turb_kmax：湍流参数（mode="turbulence" 时）
    """

    index: int
    mode: str
    exposure_ms: float
    cam_size: int
    coeffs: Mapping[Any, float] | None = None
    phase_grid: int | None = None
    turb_count: int | None = None
    turb_kmax: float | None = None


def _get_recognised_families() -> tuple[str, ...]:
    """获取识别家族列表（函数内导入 ml.hwdataset 时回退用）"""
    try:
        from ml.hwdataset import index as hwi  # type: ignore

        fams = getattr(hwi, "_RECOGNISED_FAMILIES", None)
        if isinstance(fams, tuple) and fams:
            return fams
    except Exception:
        pass
    return _RECOGNISED_FAMILIES


def validate_family_prefix(prefix: str) -> str:
    """
    校验子目录前缀是否为 ml/hwdataset 可识别的家族前缀

    Args:
        prefix: 待校验前缀

    Returns:
        规范化后的前缀

    Raises:
        ValueError: 前缀不在识别列表中
    """
    fams = _get_recognised_families()
    if prefix in fams:
        return prefix
    # 尝试最长前缀匹配（与 index.py 思路一致）
    best = None
    best_len = -1
    for f in fams:
        if prefix.startswith(f) and len(f) > best_len:
            best = f
            best_len = len(f)
    if best is not None:
        logger.warning("前缀 '%s' 未精确匹配，已按最长前缀匹配为 '%s'", prefix, best)
        return best
    raise ValueError(
        f"subdir_prefix '{prefix}' 不在 ml/hwdataset 识别列表中。\n"
        f"可用前缀（按顺序）：{list(fams)}\n"
        "提示：slm_pib_online 必须排在 slm_pib 之前。"
    )


def build_plan(params: CollectParams) -> list[PlanPoint]:
    """
    构建采样计划（纯函数，无硬件）

    以 `exposure_ms` × `cam_size` 交叉组合为骨架循环生成 `samples` 个点，
    从而刻意覆盖曝光时长与视场尺寸两类分布缺口。

    Args:
        params: 采集参数

    Returns:
        PlanPoint 列表
    """
    if params.seed is not None:
        random.seed(params.seed)
        np.random.seed(params.seed)

    exp_list = list(params.exposure_ms)
    cam_list = list(params.cam_size)
    if not exp_list:
        exp_list = [1.0]
    if not cam_list:
        cam_list = [256]

    plan: list[PlanPoint] = []
    total = params.samples
    for i in range(total):
        exp_ms = exp_list[i % len(exp_list)]
        csize = cam_list[i % len(cam_list)]
        mode = params.mode
        if mode == "zernike":
            # 随机 Noll 系数（排除 piston，Noll 1）
            n_terms = zernike_calc.calc_n_zernike_terms(params.n_max)
            coeffs: dict[int, float] = {}
            for j in range(2, n_terms + 1):  # 从 Noll 2 开始
                coeffs[j] = random.uniform(-0.5, 0.5)
            plan.append(
                PlanPoint(
                    index=i,
                    mode=mode,
                    exposure_ms=float(exp_ms),
                    cam_size=int(csize),
                    coeffs=coeffs,
                )
            )
        elif mode == "freeform":
            plan.append(
                PlanPoint(
                    index=i,
                    mode=mode,
                    exposure_ms=float(exp_ms),
                    cam_size=int(csize),
                    phase_grid=int(params.phase_grid),
                )
            )
        elif mode == "turbulence":
            plan.append(
                PlanPoint(
                    index=i,
                    mode=mode,
                    exposure_ms=float(exp_ms),
                    cam_size=int(csize),
                    turb_count=int(params.turbulence_count),
                    turb_kmax=float(params.turbulence_kmax),
                )
            )
        elif mode == "flat":
            plan.append(
                PlanPoint(
                    index=i,
                    mode=mode,
                    exposure_ms=float(exp_ms),
                    cam_size=int(csize),
                )
            )
        else:
            # 回退到 zernike
            n_terms = zernike_calc.calc_n_zernike_terms(params.n_max)
            coeffs = {j: random.uniform(-0.3, 0.3) for j in range(2, n_terms + 1)}
            plan.append(
                PlanPoint(
                    index=i,
                    mode="zernike",
                    exposure_ms=float(exp_ms),
                    cam_size=int(csize),
                    coeffs=coeffs,
                )
            )
    return plan


def describe_plan(plan: list[PlanPoint], params: CollectParams) -> str:
    """
    生成计划表格字符串（用于 --no-hw 打印）

    Args:
        plan: 采样计划
        params: 采集参数

    Returns:
        人类可读表格
    """
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("SLM 训练数据采集计划（--no-hw/--dry-run）")
    lines.append("=" * 80)
    lines.append(f"subdir_prefix: {params.subdir_prefix}")
    lines.append(f"samples: {len(plan)}")
    lines.append(f"mode: {params.mode}")
    lines.append(f"n_max: {params.n_max} | phase_grid: {params.phase_grid}")
    lines.append(f"exposure_ms: {params.exposure_ms}")
    lines.append(f"cam_size: {params.cam_size}")
    lines.append(f"zernike_radius: {params.zernike_radius}  （注意：records.py 硬编码 300.0 用于 ZERNIKE 重建）")
    lines.append(f"store_panel: {params.store_panel}")
    lines.append(f"flush_every: {params.flush_every}")
    lines.append("-" * 80)
    lines.append(f"{'idx':>4} {'mode':>10} {'exp_ms':>10} {'cam_size':>10} {'detail':<40}")
    lines.append("-" * 80)
    for p in plan:
        if p.mode == "zernike" and p.coeffs:
            nz = len(p.coeffs)
            # 同时给出"随机化的非 piston 项数"与"实际写入 _c 的向量长度"——
            # 两者差 1 (piston 槽), 而必须是后者等于三角数, 否则整条记录会被
            # ml.hwdataset 判为 ODD_COEFFICIENT_LENGTH 静默丢弃。
            n_terms = zernike_calc.calc_n_zernike_terms(int(params.n_max))
            detail = f"_c len={n_terms} (noll 2..{n_terms} 随机, 含 piston 槽)"
        elif p.mode == "freeform" and p.phase_grid:
            detail = f"grid={p.phase_grid}x{p.phase_grid}"
        elif p.mode == "turbulence":
            detail = f"count={p.turb_count} kmax={p.turb_kmax}"
        else:
            detail = "flat"
        lines.append(
            f"{p.index:>4} {p.mode:>10} {p.exposure_ms:>10.3f} {p.cam_size:>10} {detail:<40}"
        )
    lines.append("=" * 80)
    return "\n".join(lines)


def make_phase(point: PlanPoint, params: CollectParams) -> np.ndarray:
    """
    根据计划点生成 raw 弧度相位（未包裹）

    Returns:
        2D ndarray（弧度），形状由面板决定（此处返回计算用相位网格，实际下发前可按需处理）
    """
    panel_h = 1200
    panel_w = 1920
    if point.mode == "zernike":
        # `PlanPoint.coeffs` 用的是 **Noll 序号** (1-based int) 键, 因为 `_c`
        # 必须按 Noll 序存; 而 `zernike_panel` 要的是 **{(n, m): amp}**
        # (径向阶 + 方位) 键。两者由 canonical 的 `noll_to_nm` 连接 ——
        # 不要手写查表 (Noll 1976 约定, noll 4=(2,0) / 5=(2,-2) / 11=(4,0))。
        nm_coeffs = {
            zernike_calc.noll_to_nm(int(noll)): float(amp)
            for noll, amp in (point.coeffs or {}).items()
        }
        # zernike_panel 的形参是 `coefficients` (bench_kernels:373), radius 声明为 int。
        # 面板相位仅用于下发显示; 记录里存的是上面那个系数向量。
        phase = zernike_panel(
            coefficients=nm_coeffs,
            radius=int(params.zernike_radius),
            pupil_center=(int(params.pupil_center_panel[0]), int(params.pupil_center_panel[1])),
            panel_shape=(panel_h, panel_w),
        )
        return phase.astype(np.float32)
    elif point.mode == "freeform":
        g = int(point.phase_grid or params.phase_grid)
        # 随机光滑场（低频）
        rng = np.random.default_rng(params.seed + point.index if params.seed is not None else None)
        base = rng.normal(0.0, 0.3, size=(g, g)).astype(np.float32)
        # 简单双线性上采样到面板（粗略光滑）
        from scipy.ndimage import zoom  # 延迟导入（可选依赖？仓库常用；若缺失回退）

        try:
            up = np.asarray(zoom(base, (panel_h / g, panel_w / g), order=1))
        except Exception:
            # 回退：最近邻
            up = np.repeat(np.repeat(base, panel_h // g, axis=0), panel_w // g, axis=1)
            if up.shape != (panel_h, panel_w):
                up = np.pad(up, ((0, panel_h - up.shape[0]), (0, panel_w - up.shape[1])), mode="edge")
        return np.asarray(up, dtype=np.float32)
    elif point.mode == "turbulence":
        rng = np.random.default_rng(params.seed + point.index if params.seed is not None else None)
        phase = np.zeros((panel_h, panel_w), dtype=np.float32)
        count = int(point.turb_count or params.turbulence_count)
        kmax = float(point.turb_kmax or params.turbulence_kmax)
        cx, cy = params.pupil_center_panel
        Y, X = np.mgrid[0:panel_h, 0:panel_w]
        for _ in range(count):
            kx = rng.uniform(-kmax, kmax)
            ky = rng.uniform(-kmax, kmax)
            amp = rng.uniform(0.1, 0.4)
            phase += amp * np.sin(2.0 * math.pi * (kx * (X - cx) / 1000.0 + ky * (Y - cy) / 1000.0))
        return phase.astype(np.float32)
    else:  # flat
        return np.zeros((panel_h, panel_w), dtype=np.float32)


def _effective_store_panel(params: CollectParams) -> bool:
    """是否写入 uint16 全面板 `_phase`。

    ``turbulence`` 模式的相位表示**只有**面板 (它没有系数向量), 所以即使没有
    ``--store-panel`` 也必须存, 否则记录既无 ``_c`` 也无 ``_phase``, 会被
    ``ml.hwdataset`` 判为 ``MISSING_PHASE`` 静默丢弃。其余模式严格尊重用户开关。
    """
    return bool(params.store_panel) or params.mode == "turbulence"


def runner_sidecar_block(params: CollectParams, point: PlanPoint) -> dict[str, Any]:
    """把 runner 层参数原样序列化进 sidecar，保证与各 runner 的口径一致。

    为什么用 :mod:`ao_shaping.runners.runner_common` 而不是本地再抄一份字段名:
    本仓库已经因为"重复实现漂移"被咬过多次（见 ``tools/slm/params.py`` 关于
    ``_fmt`` 漂移的记录）。``CameraParams`` / ``SlmParams`` / ``RunParams`` 是
    **CLI 选项名与默认值的单一事实源**；``config_payload`` 又负责丢掉未设置的
    (None / 等于 dataclass 默认值) 字段, 因此这里产出的块与 runner 自己写的
    sidecar **同名字段**、``ml/hwdataset`` 与各 report 的读者不会看到第二种口径。

    为什么是**函数内延迟导入**: ``tools/slm/params.py`` 明确规定
    "MUST NOT import ``ao_shaping.runners``, directly or indirectly"
    (因为 ``runners/slm/zernike_matrix_runner.py`` 反向 import ``tools.slm.*``,
    模块作用域互相 import 会成环)。本模块是叶子 (没有任何 runner import 它),
    所以在函数体内导入既拿到了单一事实源, 又不会形成导入环。

    Returns:
        可直接并入 sidecar 的 dict; 序列化链路自身失败时返回 ``{}``
        (采集本身不应因元数据序列化而失败, 但会在日志里留下告警)。
    """
    try:
        from ao_shaping.runners.runner_common import (
            CameraParams,
            RunParams,
            SlmParams,
            config_payload,
        )
    except ImportError as exc:  # 极端环境下 runner 层不可导入, 不阻断采集
        logger.warning(
            "无法导入 runner_common ({}), sidecar 将缺少 runner 参数块; "
            "口径可能与 runner 自身不一致。",
            exc,
        )
        return {}

    camera = CameraParams(
        cam_id=int(params.cam_id),
        cam_type=str(params.camera_type),
        exposure_time_ms=float(point.exposure_ms),
        cam_size=int(point.cam_size),
    )
    slm = SlmParams(
        slm_number=int(params.slm_number),
        slm_wavelength=int(params.slm_wavelength),
        n_max=int(params.n_max),
        zernike_radius=float(params.zernike_radius),
    )
    run = RunParams(dir=str(params.root_dir), seed=params.seed)
    return {
        "runner_params": {
            "camera": config_payload(camera),
            "slm": config_payload(slm),
            "run": config_payload(run),
        }
    }


def sidecar_payload(params: CollectParams, point: PlanPoint) -> dict[str, Any]:
    """
    构造写入 sidecar 的元数据负载

    必须包含：objective、zernike_radius、n_max、phase_grid、mode、cam_size、region、
    far_field_size、exposure_ms、exposure_time_ms、cam_type、cam_id、pupil_cam_id、
    slm_number、slm_wavelength、pupil_center_panel、beam_radius、target_shape、target_size、
    algorithm、epochs、samples、collector、settle 字典。

    过滤 None 值。
    """
    settle_dict = {
        "n_frames": int(params.settle_n_frames),
        "n_discard": int(params.settle_n_discard),
        "wait_time_s": params.settle_wait_time_s,
        "stable_tol": float(params.settle_stable_tol),
        "max_wait_s": float(params.settle_max_wait_s),
    }
    payload: dict[str, Any] = {
        "objective": params.objective,
        "zernike_radius": float(params.zernike_radius),
        "n_max": int(params.n_max),
        "phase_grid": int(point.phase_grid or params.phase_grid),
        "mode": point.mode,
        "cam_size": int(point.cam_size),
        "region": params.region,
        "far_field_size": params.far_field_size,
        "exposure_ms": float(point.exposure_ms),
        "exposure_time_ms": float(point.exposure_ms),
        "cam_type": params.camera_type,
        "cam_id": int(params.cam_id),
        "pupil_cam_id": params.pupil_cam_id,
        "slm_number": int(params.slm_number),
        "slm_wavelength": float(params.slm_wavelength),
        "pupil_center_panel": list(params.pupil_center_panel),
        "beam_radius": float(params.beam_radius),
        "target_shape": params.target_shape,
        "target_size": float(params.target_size),
        "algorithm": params.algorithm,
        "epochs": int(params.epochs),
        "samples": int(params.samples),
        "collector": params.collector,
        "settle": settle_dict,
    }
    # 过滤 None
    payload = {k: v for k, v in payload.items() if v is not None}
    # runner 层参数块 (来自 runner_common, 与各 runner 同口径)
    payload.update(runner_sidecar_block(params, point))
    return payload


def acquire(
    cam,
    slm,
    pupil_cam,
    plan: list[PlanPoint],
    params: CollectParams,
    on_record: Callable[[dict[str, Any]], None] | None = None,
) -> Iterable[dict[str, Any]]:
    """
    逐点采集（生成器）。设备由调用方注入。

    每个点：
    - 生成相位（raw 弧度）并下发：通过 `display_and_average` 获取远场平均图像
    - 若 pupil_cam 非 None，额外采集瞳孔图像（键 `pupil`）
    - 构造 Recorder 行：包含 `mark="J"`、`exp_t`、`_epoch`、`_img`，可选 `_c` 或 `_phase`

    Args:
        cam: 远场相机实例
        slm: Santec SLM 实例（memory mode）
        pupil_cam: 瞳孔相机实例或 None
        plan: 采样计划
        params: 采集参数
        on_record: 每条记录回调（可选）

    Yields:
        记录字典（传递给 save_recorder_debug_artifacts 的 history 项）
    """
    # 定位零阶（仅一次，基于 dedenoised 平场帧）
    zero_center: tuple[float, float] | None = None
    try:
        flat_phase = np.zeros((1200, 1920), dtype=np.float32)
        flat_img = display_and_average(
            cam=cam,
            slm=slm,
            phase_rad=flat_phase,
            n_frames=params.settle_n_frames,
            n_discard=params.settle_n_discard,
            wait_time_s=params.settle_wait_time_s,
            stable_tol=params.settle_stable_tol,
            max_wait_s=params.settle_max_wait_s,
        )
        zc = zero_order_center(np.asarray(flat_img))
        zero_center = (float(zc[0]), float(zc[1]))
        logger.info("零阶定位（冻结）：(%.1f, %.1f)", zero_center[0], zero_center[1])
    except Exception as e:
        logger.warning("零阶定位失败：%s（将不冻结 ROI）", e)
        zero_center = None

    rec = Recorder(mark="J", mode="max")
    for p in plan:
        phase_rad = make_phase(p, params)
        # 下发并平均
        img = display_and_average(
            cam=cam,
            slm=slm,
            phase_rad=phase_rad,
            n_frames=params.settle_n_frames,
            n_discard=params.settle_n_discard,
            wait_time_s=params.settle_wait_time_s,
            stable_tol=params.settle_stable_tol,
            max_wait_s=params.settle_max_wait_s,
        )
        img_arr = np.asarray(img)
        # 采集瞳孔（可选）
        pupil_arr = None
        if pupil_cam is not None:
            try:
                pupil_arr = np.asarray(pupil_cam.get_numpy_image())
            except Exception as e:
                logger.debug("瞳孔相机采集失败：%s", e)
                pupil_arr = None
        # 构造记录
        #
        # `Recorder(mark="J")` 要求记录里真的有 "J" 键。本采集器**不做优化**,
        # 所以 J 不是任何 shaping 目标, 而是"帧峰值亮度"这个可复算的**代理量**
        # (与记录里的 _img 一一对应, 任何人都能从 pkl 重算校验)。
        # 真正的优化目标只出现在 sidecar 的 `objective` 字段里。
        spot = measure_spot(np.asarray(img, dtype=np.float64))
        record: dict[str, Any] = {
            "J": float(spot.peak),
            "peak": float(spot.peak),
            "spot_cx": float(spot.centroid_x),
            "spot_cy": float(spot.centroid_y),
            "exp_t": float(p.exposure_ms),
            "_epoch": int(p.index),
            "_img": img_arr,
        }
        if pupil_arr is not None:
            record["pupil"] = pupil_arr
        # 附加相位信息：优先 _c（Zernike/freeform），仅在 store_panel 时附加 _phase
        if p.mode == "zernike" and p.coeffs is not None:
            # `_c` 必须是**完整**的 Noll 向量 (含 piston 槽), 长度恰为
            # calc_n_zernike_terms(n_max) —— 15 (n_max=4) / 36 (n_max=7) / ...
            #
            # 为什么不能只存 noll>=2: 少一个就是 n_terms-1 (n_max=4 时 14),
            # 而 14 不是三角数, `ml.hwdataset.index._classify` 会判为
            # ODD_COEFFICIENT_LENGTH 并把整条记录**静默丢弃** (实测确认)。
            # detector 看不见 piston, 故槽 0 恒为 0.0, 只为把长度对齐。
            n_terms = zernike_calc.calc_n_zernike_terms(int(params.n_max))
            c_vec = np.zeros(n_terms, dtype=np.float32)
            for noll, val in p.coeffs.items():
                j = int(noll) - 1  # Noll 1-based -> 0-based
                if 0 <= j < n_terms:
                    c_vec[j] = float(val)
            record["_c"] = c_vec
        elif p.mode == "freeform" and p.phase_grid is not None:
            # freeform: 长度 = grid**2 (完全平方) -> 判为 FREEFORM。
            # 24 -> 576。注意 grid=6 时 36 同时是三角数, 只有落在
            # _FREEFORM_FAMILIES (slm_gsnet_square / sim_calib_abba) 的
            # 文件名下才判为 freeform —— 见模块 docstring 的命名约束。
            g = int(p.phase_grid)
            rng = np.random.default_rng(
                params.seed + p.index if params.seed is not None else None
            )
            base = rng.normal(0.0, 0.2, size=(g, g)).astype(np.float32)
            record["_c"] = base.reshape(-1).astype(np.float32)
        if _effective_store_panel(params):
            # 保存 uint16 全面板 `_phase`。
            #
            # 对 turbulence 模式这是**唯一**的相位表示 (它没有系数向量), 所以
            # 即使没开 --store-panel 也必须存, 否则该条记录既无 `_c` 也无 `_phase`,
            # 会被判 MISSING_PHASE 而丢弃。
            #
            # 对 zernike/freeform 则要**权衡**: `_phase` 一旦存在, `_classify`
            # 规定它就是下发面板 (index.py:683, 无视同时存在的 `_c`), source 会
            # 从 ZERNIKE/FREEFORM 变成 PANEL_GRAY 并丢掉 n_max —— main() 会就
            # 此告警。忠实的 Zernike 重放请保持 --zernike-radius 与
            # records.py 的 _ZERNIKE_APERTURE_RADIUS 一致 (默认 300)。
            try:
                gray = phase_to_slm_grayscale(phase_rad, slm=slm)
                record["_phase"] = np.asarray(gray, dtype=np.uint16)
            except Exception as e:
                logger.warning(
                    "未能保存 _phase 面板 (第 %d 条): %s —— 该条记录将只有 _c, "
                    "ZERNIKE 来源会被 ml/hwdataset 按 300.0 px 重建, 与实际下发的 "
                    "%s px 不符。",
                    p.index,
                    e,
                    params.zernike_radius,
                )
        rec.append(record)
        if on_record is not None:
            try:
                on_record(dict(record))
            except Exception:
                pass
        yield record


def main(argv: Sequence[str] | None = None) -> int:
    """
    CLI 入口

    Args:
        argv: 命令行参数

    Returns:
        退出码
    """
    parser = argparse.ArgumentParser(
        description="SLM + CCD 训练数据采集器（面向 ml/hwdataset）",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "注意事项：\n"
            "- zernike_radius 写入 sidecar，但 ml/hwdataset/records.py 硬编码 _ZERNIKE_APERTURE_RADIUS=300.0 用于 ZERNIKE 重建。\n"
            "  建议配合 --store-panel 保存 uint16 _phase 以获得忠实回放。\n"
            "- 零阶定位基于整帧 argmax（zero_order_center），定位一次后全程冻结。\n"
            "- display_and_average 已处理槽位轮换与沉降；不要手动 time.sleep。\n"
            "- SLM 仅用 memory mode（video_mode=0），严禁 DVI mode。\n"
            "- subdir_prefix 必须是 ml/hwdataset 识别的家族前缀（slm_pib_online 须排在 slm_pib 前）。"
        ),
    )
    parser.add_argument("--root-dir", type=Path, default=Path("data"))
    parser.add_argument("--subdir-prefix", type=str, default="slm_zernike_shaping")
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--mode", choices=["zernike", "freeform", "turbulence", "flat"], default="zernike")
    parser.add_argument("--n-max", type=int, default=7)
    parser.add_argument("--phase-grid", type=int, default=24)
    parser.add_argument("--turbulence-count", type=int, default=6)
    parser.add_argument("--turbulence-kmax", type=float, default=2.0)
    parser.add_argument("--zernike-radius", type=float, default=300.0)
    parser.add_argument("--beam-radius", type=float, default=150.0)
    parser.add_argument("--pupil-center-panel", type=str, default="600,960")
    parser.add_argument("--exposure-ms", type=str, default="0.5,1.0,2.0,4.0")
    parser.add_argument("--cam-size", type=str, default="256,320,512")
    parser.add_argument("--region", type=str, default=None)
    parser.add_argument("--far-field-size", type=str, default=None)
    parser.add_argument("--cam-type", type=str, default="sim")
    parser.add_argument("--cam-id", type=int, default=0)
    parser.add_argument("--pupil-cam-id", type=int, default=None)
    parser.add_argument("--slm-number", type=int, default=1)
    parser.add_argument("--slm-wavelength", type=float, default=1064.0)
    parser.add_argument("--target-shape", type=str, default="gaussian")
    parser.add_argument("--target-size", type=float, default=50.0)
    parser.add_argument("--algorithm", type=str, default="spgd")
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--objective", type=str, default="shape_uniformity")
    parser.add_argument("--collector", type=str, default="slm_train_data_collect")
    parser.add_argument("--store-panel", action="store_true", default=False)
    parser.add_argument("--no-store-panel", dest="store_panel", action="store_false")
    parser.add_argument("--flush-every", type=int, default=6)
    parser.add_argument("--settle-n-frames", type=int, default=4)
    parser.add_argument("--settle-n-discard", type=int, default=3)
    parser.add_argument("--settle-wait-time-s", type=float, default=0.5)
    parser.add_argument("--settle-stable-tol", type=float, default=0.02)
    parser.add_argument("--settle-max-wait-s", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--no-hw", action="store_true", help="仅打印计划，不触发硬件")
    parser.add_argument("--dry-run", action="store_true", help="同 --no-hw")

    args = parser.parse_args(argv)

    # 解析元组参数
    def parse_floats(s: str | None) -> tuple[float, ...]:
        if not s:
            return ()
        return tuple(float(x.strip()) for x in s.split(","))

    def parse_ints(s: str | None) -> tuple[int, ...]:
        if not s:
            return ()
        return tuple(int(x.strip()) for x in s.split(","))

    def parse_intpair(s: str | None) -> tuple[int, int] | None:
        if not s:
            return None
        parts = [int(x.strip()) for x in s.split(",")]
        return (parts[0], parts[1]) if len(parts) >= 2 else None

    def parse_intquad(s: str | None) -> tuple[int, int, int, int] | None:
        if not s:
            return None
        parts = [int(x.strip()) for x in s.split(",")]
        return (parts[0], parts[1], parts[2], parts[3]) if len(parts) >= 4 else None

    pupil_c = parse_floats(args.pupil_center_panel)
    pupil_center = (pupil_c[0], pupil_c[1]) if len(pupil_c) >= 2 else (600.0, 960.0)

    params = CollectParams(
        root_dir=args.root_dir,
        subdir_prefix=validate_family_prefix(args.subdir_prefix),
        samples=args.samples,
        mode=args.mode,
        n_max=args.n_max,
        phase_grid=args.phase_grid,
        turbulence_count=args.turbulence_count,
        turbulence_kmax=args.turbulence_kmax,
        zernike_radius=args.zernike_radius,
        beam_radius=args.beam_radius,
        pupil_center_panel=pupil_center,
        exposure_ms=parse_floats(args.exposure_ms),
        cam_size=parse_ints(args.cam_size),
        region=parse_intquad(args.region),
        far_field_size=parse_intpair(args.far_field_size),
        camera_type=args.cam_type,
        cam_id=args.cam_id,
        pupil_cam_id=args.pupil_cam_id,
        slm_number=args.slm_number,
        slm_wavelength=args.slm_wavelength,
        target_shape=args.target_shape,
        target_size=args.target_size,
        algorithm=args.algorithm,
        epochs=args.epochs,
        objective=args.objective,
        collector=args.collector,
        store_panel=args.store_panel,
        flush_every=args.flush_every,
        settle_n_frames=args.settle_n_frames,
        settle_n_discard=args.settle_n_discard,
        settle_wait_time_s=args.settle_wait_time_s,
        settle_stable_tol=args.settle_stable_tol,
        settle_max_wait_s=args.settle_max_wait_s,
        seed=args.seed,
        no_hw=args.no_hw or args.dry_run,
        dry_run=args.dry_run,
    )

    if params.store_panel and params.mode in ("zernike", "freeform"):
        logger.warning(
            "--store-panel 与 mode=%s 同时开启: `ml/hwdataset.index._classify` "
            "规定 `_phase` 存在时它**就是**下发面板 (index.py:683, 无视同时存在的 "
            "`_c`), 因此这些记录会被判为 PANEL_GRAY 而不是 %s, `n_max`/`n_terms` "
            "信息将丢失。要保留 %s 语义就别加 --store-panel。",
            params.mode,
            "ZERNIKE" if params.mode == "zernike" else "FREEFORM",
            params.mode,
        )
    if params.store_panel and params.mode == "zernike":
        logger.warning(
            "忠实的 Zernike 重放有两条路: (1) 保持 --zernike-radius %s "
            "(= ml/hwdataset/records.py 的 _ZERNIKE_APERTURE_RADIUS, 该常量**不读** "
            "sidecar 的 zernike_radius), 只存 `_c` 即可精确重建; 或 (2) 换非 300 "
            "半径并加 --store-panel 存面板, 但代价是 source 变成 PANEL_GRAY。",
            params.zernike_radius,
        )

    plan = build_plan(params)
    print(describe_plan(plan, params))

    if params.no_hw:
        logger.info("已打印计划（--no-hw/--dry-run），未触发任何硬件。")
        return 0

    # 仅在硬件路径导入驱动
    try:
        from ao_shaping.drivers.slm import Santec  # type: ignore
        from ao_shaping.utils.image.hardware_utils import open_camera  # type: ignore
    except Exception as e:
        logger.error("导入硬件驱动失败：%s", e)
        return 1

    cam = None
    slm = None
    pupil_cam = None
    try:
        # 打开 SLM（memory mode，video_mode=0）
        slm = Santec(slm_number=params.slm_number, video_mode=0)
        slm.open()
        logger.info("SLM #%d 已打开（memory mode）", params.slm_number)

        # 打开远场相机
        cam = open_camera(
            camera_type=params.camera_type,
            cam_id=params.cam_id,
            exposure_ms=float(plan[0].exposure_ms if plan else 1.0),
            bit_depth=8,
        )
        cam.open()
        logger.info("远场相机已打开（type=%s, id=%d）", params.camera_type, params.cam_id)

        # 打开瞳孔相机（可选）
        if params.pupil_cam_id is not None:
            try:
                pupil_cam = open_camera(
                    camera_type=params.camera_type,
                    cam_id=params.pupil_cam_id,
                    exposure_ms=float(plan[0].exposure_ms if plan else 1.0),
                    bit_depth=8,
                )
                pupil_cam.open()
                logger.info("瞳孔相机已打开（id=%d）", params.pupil_cam_id)
            except Exception as e:
                logger.warning("瞳孔相机打开失败：%s", e)
                pupil_cam = None

        # 采集
        history: list[dict[str, Any]] = []
        flushed = 0

        # ``save_recorder_debug_artifacts`` 读的是 ``res.history`` ——
        # 传普通 dict 会 AttributeError 并被下面的 except 吞掉, 结果是
        # "脚本跑完了但一个文件都没落"。所以必须真的构造 Recorder。
        recorder = Recorder(mark="J", mode="max")
        # writer 只保留白名单里的键: 少写一个字段, 下游就看不到它。
        scalar_keys = ("exp_t", "peak", "spot_cx", "spot_cy", "J")
        img_keys = ("_img", "pupil")
        d1_keys = ("_c",)
        d2_keys = ("_phase",)

        def _flush(rows: list[dict[str, Any]], *, final: bool) -> bool:
            """把一批记录写成 debug 产物; 成功返回 True。"""
            if not rows:
                return True
            batch = Recorder(mark="J", mode="max")
            batch.history = list(rows)
            point = plan[rows[-1]["_epoch"]] if 0 <= rows[-1]["_epoch"] < len(plan) else plan[0]
            save_recorder_debug_artifacts(
                res=batch,
                root_dir=str(params.root_dir),
                subdir_prefix=params.subdir_prefix,
                scalar_keys=scalar_keys,
                objective_keys=(),
                img_keys=img_keys,
                d1_keys=d1_keys,
                d2_keys=d2_keys,
                json_payload=sidecar_payload(params, point),
                title="slm_train_data_collect",
            )
            logger.info(
                "已刷写%s产物（记录 %d 条）", "最终" if final else "中间", len(rows)
            )
            return True

        for rec in acquire(
            cam, slm, pupil_cam, plan, params, on_record=lambda r: history.append(r)
        ):
            recorder.append(rec)
            if params.flush_every > 0 and len(history) >= params.flush_every:
                try:
                    _flush(history, final=False)
                    flushed += 1
                    history.clear()
                except Exception as e:
                    logger.error(
                        "中间刷写失败（第 %d 批）：%s —— 继续采集, 结束时仍会尝试落盘",
                        flushed,
                        e,
                    )
        # 最终刷写
        if history:
            try:
                _flush(history, final=True)
            except Exception as e:
                logger.error("最终刷写失败：%s", e)
                return 1
        return 0
    except Exception as e:
        logger.exception("采集过程中发生异常：%s", e)
        return 1
    finally:
        # 关闭设备
        try:
            if pupil_cam is not None:
                pupil_cam.close()
        except Exception:
            pass
        try:
            if cam is not None:
                cam.close()
        except Exception:
            pass
        try:
            if slm is not None:
                slm.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
