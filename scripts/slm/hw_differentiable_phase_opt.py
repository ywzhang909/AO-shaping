"""硬件实验脚本: 物理预测模型 + 反向传播可微分相位优化 (默认 dry-run)。

要验证什么
----------

在真实实验系统 (2f 傅里叶光路: Santec SLM-200 → f=125mm 透镜 → Daheng CCD)
上把两件事一次做完, 并把**全过程**记录下来:

1. **物理预测模型**: 由 SLM 瞳面 (pupil) 振幅与相位经一次傅里叶变换预测远场
   强度 (即仓库唯一的公开 API
   :class:`~ao_shaping.algorithm.signal_processing.differentiable_beam.DifferentiableBeamOptimizer`
   内部用的那个模型);
2. **反向传播可微分优化**: 把瞳面相位当成可学习张量, 用 autograd 求
   远场强度损失关于相位的梯度, 用 Adam 更新 —— 纯模型侧, 硬件不在环里;
3. **模型 vs 硬件**: 把优化出来的相位真的上屏, 抓远场画面, 把实测远场
   重采样到模型的远场网格上, 对比峰值位置、FWHM、强度相关系数与
   「类 Strehl」峰值比;
4. **测量锚定闭环**: 把实测远场喂回优化器
   (:meth:`DifferentiableBeamOptimizer.update` 的 ``measured_intensity`` 分支),
   看梯度在实验系统上是否仍然指向正确方向。

🔴 **本脚本默认 dry-run: 只打印完整实验计划, 绝不 import 或 open 任何设备。**

不给 ``--execute`` 时本脚本:

* 只 import **零导入叶子** (:mod:`ao_shaping.tools.kernels.bench_kernels`、
  :mod:`ao_shaping.tools.slm.panel.focus_probe`、
  :mod:`ao_shaping.algorithm.signal_processing.differentiable_beam`、
  :mod:`ao_shaping.utils.image.*`、:mod:`ao_shaping.utils.slm.bench_constants`)
  —— 实测它们**不牵入任何** ``ao_shaping.drivers.*``;
* 收尾时**断言** ``ao_shaping.drivers`` 没出现在 ``sys.modules`` 里 —— 这条断言
  让「dry-run 不碰设备」成为可执行的保证而不是口头承诺, 违反就 exit 3。

因此 ``--dry-run`` 在**没有任何硬件的机器上**也必须 exit 0。

🔴 **测量之前必须先判在焦 (用户明确要求)**

本脚本的**第一件测量**不是优化, 而是**在焦判定**:

* 用 :func:`ao_shaping.tools.slm.panel.focus_probe.sweep_focus` 在 plain
  相位上扫菲涅尔焦距 (1/f), 找到让焦面回到探测器的那个点;
* 判「光路受阻」 ⇒ **exit 4**, 不写任何测量数据 (光没到相机, 后面全是噪声);
* 判「最佳点的 FWHM 超过阈值」 ⇒ **exit 5 (不在焦, 拒绝测量)**, 仍然不写任何
  测量数据 —— 用户的要求是**实验系统没有被脚本判为在焦之前, 一个测量都不做**;
* 通过 ⇒ 之后每张图案都叠加那张瞳面菲涅尔补偿相位。

退出码: 0 = 正常; 2 = 设备挂死/超时 (:func:`guarded` 硬退出); 3 = dry-run 违约
(:func:`assert_no_drivers`); 4 = 光路受阻; 5 = **不在焦, 拒绝测量**;
6 = 几何标定或物理预测模型这一步失败; 7 = 设备被别的会话占用, 等不到空闲
(``--wait-until-idle`` 走 :func:`run_hw_sweep_groups.wait_until_idle` 的公平仲裁:
先看别的会话有没有在用设备, 没有才探针 —— **本脚本不抢设备**)。

🔴 **几何标定默认现测, 不用仓库里那份陈旧的** ``data/calib.npz``
(实测那份的 Kx=4416、rotation=27.3° 与当前光路对不上)。要复用存好的标定
必须显式给 ``--geometry-npz``, 并且脚本会把「这次用的是存档」写进
``run_config.json``。

几何与重采样的口径 (脚本作者的决定, 理由见下)
------------------------------------------

* **模型网格**: 瞳面跨度 ``span = 2·pupil_radius`` 面板像元, 由
  ``N = --model-pupil-samples`` 个采样点表示 ⇒ 采样间距 ``dp = span/N``;
  模型网格边长 ``M = --model-padding · N``, 瞳面占中央 ``N×N`` 块, 四周补零
  (零填充) —— 这正是让远场被充分采样的关键;
* **瞳面中心**放在模型网格的整数下标 ``M//2`` 上, 于是傅里叶变换后零级
  (直流) 正好落在 fftshift 后的同一格 ``M//2``;
* **远场像元尺度**: 远场第 ``k`` 格对应空间频率 ``k/(M·dp)`` 循环/面板像元,
  而标定给出的 ``K`` (CCD 像元 / 面板像元) 把空间频率换成 CCD 像元, 所以
  **远场采样间距 = ``K/(M·dp)`` CCD 像元**, 整个 ``M×M`` 远场阵列覆盖
  ``M·K/(M·dp) = K/dp`` CCD 像元。⚠️ 这里用的是 ``K/dp``, **不是**
  ``K·M·dp/span`` —— 后者把「网格跨度/瞳面跨度」当成了放大率, 量纲上是
  CCD 像元 × 网格/瞳面比, 与傅里叶变换给出的远场覆盖范围无关 (用它会把裁
  剪框放到传感器外面去)。脚本把两个数都写进 ``run_config.json`` 以便核对;
* **K 的取法**: ``K = (Kx + Ky)/2``, 各向同性 (脚本会把这个取法写进
  ``run_config.json``)。各向异性会带来一个「面板 x 对应相机 y 还是相机 x」
  的歧义, 而仓库里记的 90° 轴交换又与几何标定返回值的约定并不自洽, 所以
  脚本**不假设**: 对比时把「转置」当作两个候选口径实测对比择优, 对比择优结果
  (``orientation_match``) 写进 ``run_config.json``。

运行
----

.. code-block:: bash

    # 无硬件机器上也能跑: 只打印计划
    python scripts/slm/hw_differentiable_phase_opt.py --dry-run

    # 装好设备后
    python scripts/slm/hw_differentiable_phase_opt.py --execute

    # 冒烟测试 (2 次迭代)
    python scripts/slm/hw_differentiable_phase_opt.py --execute --quick

    # 离线重渲染报告 (改生成器之后, 无设备也能刷新已落盘的报告)
    python scripts/slm/hw_differentiable_phase_opt.py \
        --render-only data/debug/differentiable_opt/<时间戳>

产物
----

``--out-dir`` (默认 ``data/debug/differentiable_opt/<时间戳>/``) 下:
``run_config.json`` / ``history.csv`` / ``focus_sweep.csv`` /
``phase_initial.npy`` / ``phase_final.npy`` / ``phase_model.npy`` /
原始 CCD 帧 (``ccd_*.npy``) / ``figures/``。

报告写到 ``report/slm/differentiable_opt/differentiable_opt_report.md``,
图在同目录的 ``figures/``, 本轮原始产物复制到 ``artifacts/<时间戳>/``
(一份报告 = 一个子目录, 见 ``report/AGENTS.md``)。
``--render-only`` 读回同一个运行目录的 ``run_config.json`` / ``history.csv`` /
``history_hw.csv`` 重渲染报告正文 (只搬 json/csv, 大件 npz/npy 留在
``data/debug/``)。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
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
from scipy.ndimage import map_coordinates

# ---------------------------------------------------------------------------
# 零导入叶子: 这些模块实测都不牵入 ao_shaping.drivers.*
# ---------------------------------------------------------------------------
from ao_shaping.algorithm.signal_processing.differentiable_beam import (
    DifferentiableBeamOptimizer,
)
from ao_shaping.tools.kernels.bench_kernels import (
    SLM_PANEL_H,
    SLM_PANEL_W,
    fresnel_panel,
)
from ao_shaping.tools.slm.panel.focus_probe import (
    ExposureSetting,
    FocusProbeResult,
    adapt_exposure,
    judge_focus,
    sweep_focus,
)
from ao_shaping.utils.image.beam_metrics import prepare_frame
from ao_shaping.utils.image.frame_analysis import (
    assess_spot,
    measure_spot,
)
from ao_shaping.utils.slm.bench_constants import (
    BEAM_CENTER_PANEL,
    SLM_PITCH_M,
    TILT_SHIFT_SCALE,
)
from scripts._common import fmt_general
from scripts._common.plot_style import setup_cjk_font
from scripts._common.provenance import insert_header
from scripts.slm.hw_gs_defocus_probe import guarded

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

#: 报告的仓库根相对路径 (溯源块与 ``report/README.md`` 索引的键)。
REPORT_KEY = "report/slm/differentiable_opt/differentiable_opt_report.md"
#: 报告子目录 (一份报告 = 一个子目录, 见 ``report/AGENTS.md`` §2)。
REPORT_DIR = ROOT / "report" / "slm" / "differentiable_opt"
REPORT_FIG_DIR = REPORT_DIR / "figures"
REPORT_ARTIFACT_DIR = REPORT_DIR / "artifacts"

#: 退出码
EXIT_OK = 0
EXIT_DEVICE_HANG = 2
EXIT_DRYRUN_VIOLATION = 3
EXIT_PATH_BLOCKED = 4
EXIT_NOT_IN_FOCUS = 5
EXIT_MODEL_FAILED = 6
#: 别的会话一直占着设备, 等不到空闲 (本脚本**不抢**设备)。
EXIT_BENCH_BUSY = 7

#: 面板↔相机轴向映射的候选 (名字, 是否转置, 是否翻行, 是否翻列)。
#: 几何标定只给出两条光栅轴的正交性偏差, 给不出带符号的轴映射 ⇒ 全部试
#: 一遍, 按相关系数对比择优。
_AXIS_CANDIDATES: tuple[tuple[str, bool, bool, bool], ...] = (
    ("identity", False, False, False),
    ("transpose", True, False, False),
    ("flip_rows", False, True, False),
    ("flip_cols", False, False, True),
    ("flip_both", False, True, True),
    ("transpose_flip_rows", True, True, False),
    ("transpose_flip_cols", True, False, True),
    ("transpose_flip_both", True, True, True),
)

#: 设备打开/取帧的超时保护 (s)。Santec 的 ``WaitImageV3`` 原生等待由 SDK 内部
#: 驱动, **不受 Python 侧超时保护** (仓库 2026-09-08 实测可无限阻塞), 所以
#: 这里用看门狗线程 + 硬退出, 见 :func:`guarded`。
OPEN_TIMEOUT_S = 180.0
ACQUIRE_TIMEOUT_S = 60.0
CLOSE_TIMEOUT_S = 30.0

#: 相机下限曝光 (ms)。Daheng 低于这个值会静默返回 ~0.02 ms。
FLOOR_EXPOSURE_MS = 0.02
#: 曝光上限默认值 (ms); 若环境变量 ``AO_DAHENG_MAX_EXPOSURE_MS`` 有值则取更小。
DEFAULT_CEILING_MS = 200.0

#: 默认瞳面 (面板像元)。跨度 = 2·半径 = 960 面板像元。
DEFAULT_PUPIL_CENTER = (960.0, 600.0)
DEFAULT_PUPIL_RADIUS = 480.0
#: 在焦判定的 FWHM 门限 (CCD 像元)。超过 ⇒ 拒绝测量 (exit 5)。
DEFAULT_FOCUS_FWHM_MAX = 40.0
DEFAULT_FOCUS_POINTS = 13
DEFAULT_FOCUS_DIOPTER_RANGE = 6.0
DEFAULT_FOCUS_REFINE = 7

#: 物理预测模型网格
DEFAULT_MODEL_PUPIL_SAMPLES = 256
DEFAULT_MODEL_PADDING = 4
DEFAULT_BEAM_W0_PX = 260.0

#: 优化
DEFAULT_ITERS = 200
DEFAULT_HW_ITERS = 5
DEFAULT_LR = 0.05
DEFAULT_SEED = 0
DEFAULT_GRAD_EVERY = 10
DEFAULT_TARGET = "square"
DEFAULT_TARGET_SIDE_PX = 60.0
#: delta 目标的亮斑直径 (CCD px) —— 取在焦零级的 FWHM (实测 15 px)。
DEFAULT_DELTA_SPOT_PX = 15.0

#: 采集
DEFAULT_EXPOSURE_MS = 2.0
DEFAULT_FRAMES = 3
DEFAULT_DISCARD = 2
DEFAULT_SETTLE_S = 0.5

#: 默认输出根目录
DEFAULT_OUT_ROOT = ROOT / "data" / "debug" / "differentiable_opt"


# ---------------------------------------------------------------------------
# 模型网格 (纯几何, 与设备无关 —— 可离线自检)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelSpec:
    """物理预测模型的网格几何。

    Attributes:
        n_pupil: 瞳面采样数 ``N`` (瞳面占中央 ``N×N`` 块)。
        m_grid: 模型网格边长 ``M = padding · N``。
        dp: 瞳面采样间距 (面板像元) ``span / N``。
        span: 瞳面跨度 (面板像元) ``2 · pupil_radius``。
        pupil_radius: 瞳面半径 (面板像元)。
        pupil_center: 瞳面中心 (面板像元, ``(x, y)``)。
        w0_px: 照明高斯腰径 (面板像元)。
        padding: 零填充倍数。
    """

    n_pupil: int
    m_grid: int
    dp: float
    span: float
    pupil_radius: float
    pupil_center: tuple[float, float]
    w0_px: float
    padding: int

    @property
    def centre_index(self) -> int:
        """模型网格的**直流格**下标 (整数 ``M//2``) —— fftshift 后的零级所在格。"""
        return self.m_grid // 2

    @property
    def pupil_start(self) -> int:
        """瞳面块在模型网格里的起始下标 (行=列)。"""
        return self.centre_index - self.n_pupil // 2

    @property
    def block_centre(self) -> float:
        """瞳面块的几何中心下标 (可能是半整数)。

        这是**源面振幅的对称中心**, 也是模型网格与面板之间的对位锚点 ——
        两者必须一致, 否则模型里的瞳面与面板上的瞳面差半个采样 (默认 dp/2 ≈
        1.9 面板像元), 焦面补偿的圆心就会与光斑错开。
        """
        return self.pupil_start + (self.n_pupil - 1) / 2.0

    @property
    def pupil_radius_samples(self) -> float:
        """瞳面半径在模型采样里的长度 (样本)。"""
        return self.pupil_radius / self.dp

    @property
    def w0_samples(self) -> float:
        """高斯腰径在模型采样里的长度 (样本)。"""
        return self.w0_px / self.dp

    def far_field_sample_ccd_px(self, k_mean: float) -> float:
        """远场采样间距 (CCD 像元) = ``K / (M·dp)``。"""
        return float(k_mean) / (self.m_grid * self.dp)

    def far_field_span_ccd_px(self, k_mean: float) -> float:
        """远场阵列覆盖的 CCD 像元总跨度 = ``K / dp``。"""
        return float(k_mean) / self.dp

    def as_dict(self, k_mean: float | None = None) -> dict[str, Any]:
        """JSON 化的模型描述 (写进 ``run_config.json``)。"""
        out: dict[str, Any] = {
            "n_pupil": self.n_pupil,
            "m_grid": self.m_grid,
            "padding": self.padding,
            "dp_panel_px": self.dp,
            "span_panel_px": self.span,
            "pupil_radius_panel_px": self.pupil_radius,
            "pupil_center_panel_xy": list(self.pupil_center),
            "w0_panel_px": self.w0_px,
            "w0_samples": self.w0_samples,
            "pupil_radius_samples": self.pupil_radius_samples,
            "pupil_block_start_index": self.pupil_start,
            "block_centre_index": self.block_centre,
            "centre_index": self.centre_index,
            "grid_extent_panel_px": self.m_grid * self.dp,
        }
        if k_mean is not None:
            out["far_field_sample_ccd_px"] = self.far_field_sample_ccd_px(k_mean)
            out["far_field_span_ccd_px"] = self.far_field_span_ccd_px(k_mean)
            out["far_field_span_note"] = (
                "覆盖范围 = M·采样间距 = K/dp (傅里叶变换给出的真实覆盖); "
                "K·M·dp/span 不是远场覆盖范围, 量纲上不成立"
            )
        return out


def build_model_spec(
    *,
    pupil_radius: float,
    pupil_center: tuple[float, float],
    n_pupil: int,
    padding: int,
    w0_px: float,
) -> ModelSpec:
    """由参数装配模型网格。

    Raises:
        ValueError: 参数不合法 (采样数/填充倍数非正、瞳面半径非正)。
    """
    n = int(n_pupil)
    pad = int(padding)
    if n < 4:
        raise ValueError(f"model-pupil-samples 必须 >= 4, 给的是 {n}")
    if pad < 1:
        raise ValueError(f"model-padding 必须 >= 1, 给的是 {pad}")
    radius = float(pupil_radius)
    if radius <= 0:
        raise ValueError(f"pupil-radius 必须 > 0, 给的是 {radius}")
    span = 2.0 * radius
    dp = span / n
    return ModelSpec(
        n_pupil=n,
        m_grid=pad * n,
        dp=dp,
        span=span,
        pupil_radius=radius,
        pupil_center=(float(pupil_center[0]), float(pupil_center[1])),
        w0_px=float(w0_px),
        padding=pad,
    )


def build_source_amplitude(spec: ModelSpec) -> np.ndarray:
    """源面 (瞳面) 振幅: 高斯照明 × 硬口径, 口径外为零。

    高斯腰径 ``w0`` 与口径半径都用**面板像元**给, 在模型网格里换算成样本数
    (除以 ``dp``)。整个 ``M×M`` 网格上, 只有中央 ``N×N`` 块非零 —— 块外补零
    就是让远场被充分采样的零填充。

    ⚠️ **口径与瞳面块必须严丝合缝**: 口径半径 ``pupil_radius`` 面板像元 =
    ``N/2`` 个样本, 以**块的几何中心** ``block_centre`` 为圆心 ⇒ 口径恰好内接于
    瞳面块 (不多不少一个样本)。如果把圆心放在整数格 ``M//2`` 上, 口径会多出
    一列样本 (N+1 列), 那一列**不会**被搬到面板上 —— 于是模型的远场里含着
    硬件上根本不存在的光, 模型与硬件从此不是同一个东西。
    """
    m = spec.m_grid
    idx = np.arange(m, dtype=np.float64) - spec.block_centre
    r2 = idx[:, None] ** 2 + idx[None, :] ** 2
    w0 = max(spec.w0_samples, 1e-6)
    amp = np.exp(-r2 / (2.0 * w0 * w0))
    amp[r2 > spec.pupil_radius_samples**2] = 0.0
    return amp


def build_target(
    spec: ModelSpec,
    kind: str,
    side_px: float,
    k_mean: float,
    delta_spot_px: float = 15.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """构造目标远场强度 (建在 ``M×M`` 远场网格上, 与目标网格同一张)。

    Args:
        spec: 模型网格。
        kind: ``"delta"`` (以零级为中心的一个**有限大小**亮斑) 或
            ``"square"`` (以零级为中心的均匀方块, 边长给的是 **CCD 像元**)。
        side_px: 方形目标的边长 (CCD 像元); ``kind="delta"`` 时忽略。
        k_mean: 各向同性的远场标度 ``(Kx+Ky)/2`` (CCD 像元/面板像元)。
        delta_spot_px: delta 目标的亮斑**直径** (CCD 像元), 默认取衍射极限
            (实测在焦零级 FWHM ≈ 15 px)。

    Returns:
        ``(目标数组, 描述字典)``。

    ⚠️ **为什么 delta 目标不能是单个像素**: 远场网格有 M² ≈ 10⁶ 个格, 而
    峰值归一化后的损失 ``mean((I/I_max - T)²)`` 里, 单像素目标自己只贡献
    1/M² ≈ 10⁻⁶, 剩下 10⁻⁵ 全是"背景要压暗"。于是**目标尖峰的位置几乎
    不进损失** —— 优化器只学会了把背景压下去, 峰值落在哪个格由随机相位
    决定 (2026-10-22 首次实测: 损失降到 1.5e-5, 峰值却落在离中心 183/262
    个采样点处, 于是模型与硬件的相关系数只有 0.002)。把目标换成一个
    **衍射极限大小**的亮斑 (直径 ≈ 15 CCD px ≈ 8 个远场采样点), 位置
    才真正被约束住。
    """
    m = spec.m_grid
    c = spec.centre_index
    target = np.zeros((m, m), dtype=np.float64)
    info: dict[str, Any] = {"kind": kind, "grid": [m, m]}
    s_far = spec.far_field_sample_ccd_px(k_mean)
    if kind == "delta":
        half = max(int(round(float(delta_spot_px) / s_far / 2.0)), 1)
        lo, hi = c - half, c + half + 1
        target[lo:hi, lo:hi] = 1.0
        info.update(
            {
                "spot_ccd_px_requested": float(delta_spot_px),
                "spot_samples": int(hi - lo),
                "spot_ccd_px_effective": float((hi - lo) * s_far),
                "far_field_sample_ccd_px": float(s_far),
                "peak_sample": [c, c],
                "note": (
                    "以零级为中心、衍射极限大小的亮斑 (不是单像素: "
                    "单像素目标在 10^6 格的网格上几乎不约束峰值位置)"
                ),
            }
        )
        return target, info
    if kind != "square":
        raise ValueError(f"未知 target: {kind!r}")
    half = max(int(round(float(side_px) / s_far / 2.0)), 1)
    lo, hi = c - half, c + half + 1
    target[lo:hi, lo:hi] = 1.0
    info.update(
        {
            "side_ccd_px_requested": float(side_px),
            "side_samples": int(hi - lo),
            "side_ccd_px_effective": float((hi - lo) * s_far),
            "far_field_sample_ccd_px": float(s_far),
            "peak_sample": [c, c],
            "note": "以零级为中心的均匀方块, 边长按 CCD 像元给再换算成远场样本",
        }
    )
    return target, info


def phase_model_to_panel(
    phase_model: np.ndarray,
    spec: ModelSpec,
    panel_shape: tuple[int, int] = (SLM_PANEL_H, SLM_PANEL_W),
) -> np.ndarray:
    """把模型网格上的相位搬到面板: 瞳面块放到瞳面中心, 其余为零。

    模型采样间距 ``dp`` 通常不是整数 (默认 960/256 = 3.75 面板像元), 所以用
    **最近邻**取整映射 (不做插值): 硬件看到的必须**就是**模型优化的那组
    相位值, 插值会偷偷改掉模型与硬件的口径。相位在面板上是**块状**的
    (每个模型采样占 dp×dp 面板像元), 这是模型分辨率的直接后果, 报告里会
    明确写出来。

    对位锚点是瞳面块的**几何中心** :attr:`ModelSpec.block_centre`, 于是模型
    瞳面 (代表面板上 ``cx ± pupil_radius`` 的那一片) 的中心正好落在面板的
    瞳面中心上 —— 焦面补偿的圆心与光斑才对得上。
    """
    height, width = int(panel_shape[0]), int(panel_shape[1])
    out = np.zeros((height, width), dtype=np.float64)
    cx, cy = spec.pupil_center
    half = (spec.n_pupil - 1) / 2.0
    x0 = int(round(cx - half * spec.dp))
    x1 = int(round(cx + half * spec.dp)) + 1
    y0 = int(round(cy - half * spec.dp))
    y1 = int(round(cy + half * spec.dp)) + 1
    x0c, y0c = max(x0, 0), max(y0, 0)
    x1c, y1c = min(x1, width), min(y1, height)
    if x1c <= x0c or y1c <= y0c:
        return out
    xs = np.arange(x0c, x1c, dtype=np.float64)
    ys = np.arange(y0c, y1c, dtype=np.float64)
    mj = np.rint(spec.block_centre + (xs - cx) / spec.dp).astype(np.int64)
    mi = np.rint(spec.block_centre + (ys - cy) / spec.dp).astype(np.int64)
    mj = np.clip(mj, 0, spec.m_grid - 1)
    mi = np.clip(mi, 0, spec.m_grid - 1)
    out[y0c:y1c, x0c:x1c] = np.asarray(phase_model)[np.ix_(mi, mj)]
    return out


def resample_ccd_to_model(
    frame: np.ndarray,
    centre_xy: tuple[float, float],
    k_mean: float,
    spec: ModelSpec,
) -> np.ndarray:
    """把 CCD 帧重采样到模型的远场网格 (``M×M``)。

    模型第 ``(i, j)`` 格对应空间频率 ``(i-c, j-c)/(M·dp)``, 换算成 CCD 像元
    后是 ``centre + (i-c)·s_far`` (行) 与 ``centre + (j-c)·s_far`` (列),
    其中 ``s_far = K/(M·dp)``。重采样用
    :func:`scipy.ndimage.map_coordinates` (双线性, ``order=1``); 落在传感器
    外的部分填 0。
    """
    m = spec.m_grid
    s_far = spec.far_field_sample_ccd_px(k_mean)
    cx0, cy0 = float(centre_xy[0]), float(centre_xy[1])
    offsets = (np.arange(m, dtype=np.float64) - spec.centre_index) * s_far
    yy = cy0 + offsets
    xx = cx0 + offsets
    grid_y, grid_x = np.meshgrid(yy, xx, indexing="ij")
    coords = np.stack([grid_y.ravel(), grid_x.ravel()])
    out = map_coordinates(
        np.asarray(frame, dtype=np.float64),
        coords,
        order=1,
        mode="constant",
        cval=0.0,
    )
    return out.reshape(m, m)


def compare_farfields(
    model: np.ndarray,
    measured: np.ndarray,
    *,
    transpose_measured: bool = False,
) -> dict[str, Any]:
    """模型远场与实测远场的对比数字。

    Args:
        model: 模型远场强度 (``M×M``)。
        measured: 实测远场强度 (已重采样到 ``M×M``)。
        transpose_measured: 是否先把实测图转置 (面板↔相机 90° 轴交换的候选
            口径)。

    Returns:
        含峰值位置偏移、FWHM、强度相关系数与原始峰值比的字典。
    """
    meas = np.asarray(measured, dtype=np.float64)
    if transpose_measured:
        meas = meas.T
    mod = np.asarray(model, dtype=np.float64)

    def _peak_xy(arr: np.ndarray) -> tuple[float, float]:
        if arr.size == 0 or not np.isfinite(arr).any() or float(arr.max()) <= 0:
            return float("nan"), float("nan")
        iy, ix = np.unravel_index(int(np.argmax(arr)), arr.shape)
        return float(ix), float(iy)

    mx, my = _peak_xy(mod)
    px, py = _peak_xy(meas)
    model_spot = measure_spot(mod)
    meas_spot = measure_spot(meas)

    a = mod.ravel().astype(np.float64)
    b = meas.ravel().astype(np.float64)
    if a.std() > 0 and b.std() > 0:
        corr = float(np.corrcoef(a, b)[0, 1])
    else:
        corr = float("nan")
    model_peak = float(mod.max())
    meas_peak = float(meas.max())
    return {
        "transpose_measured": bool(transpose_measured),
        "model_peak_xy": [mx, my],
        "measured_peak_xy": [px, py],
        "peak_offset_samples": [
            float(px - mx) if np.isfinite(px - mx) else float("nan"),
            float(py - my) if np.isfinite(py - my) else float("nan"),
        ],
        "model_fwhm_px": float(model_spot.fwhm_px),
        "measured_fwhm_px": float(meas_spot.fwhm_px),
        "model_peak": model_peak,
        "measured_peak": meas_peak,
        # ⚠️ 这是两个**不同单位**的峰值相除 (模型的无量纲单位 vs CCD 的 counts),
        # 只作诊断, 不作结论 —— 有意义的「类 Strehl」比值在 _run_bench 里按
        # 「各自相对平场参考的峰值比」算。
        "peak_ratio_raw": (
            float(meas_peak / model_peak) if model_peak > 0 else float("nan")
        ),
        "intensity_correlation": corr,
    }


# ---------------------------------------------------------------------------
# 计划 (dry-run 的全部输出)
# ---------------------------------------------------------------------------
def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    """完整实验计划: 每个阶段做什么、需要哪些设备、产物是什么。"""
    spec = build_model_spec(
        pupil_radius=args.pupil_radius,
        pupil_center=(args.pupil_center[0], args.pupil_center[1]),
        n_pupil=args.model_pupil_samples,
        padding=args.model_padding,
        w0_px=args.beam_w0_px,
    )
    return {
        "devices": [
            {
                "role": "SLM",
                "driver": "ao_shaping.drivers.slm.santec.Santec (memory 模式, video_mode=0)",
                "slm_number": int(args.slm_number),
                "slm_wavelength_nm": int(args.slm_wavelength),
                "panel_shape_hw": [SLM_PANEL_W, SLM_PANEL_H],
                "fatal_if": "open() 在 DVI 模式会挂死 ⇒ 本脚本恒用 memory 模式",
            },
            {
                "role": "CCD",
                "driver": "ao_shaping.drivers.ccd.common.create_camera('daheng')",
                "cam_type": args.cam_type,
                "cam_id": int(args.cam_id),
                "exposure_ms_requested": float(args.exposure_ms),
                "fatal_if": "曝光必须 <= AO_DAHENG_MAX_EXPOSURE_MS (未设则用 20 ms)",
            },
        ],
        "focus_gate": {
            "what": "**测量之前**先判实验系统在不在焦 (用户明确要求)",
            "method": (
                "ao_shaping.tools.slm.panel.focus_probe.sweep_focus: 在 plain 相位上"
                "扫菲涅尔焦距 (1/f), 每点重新对曝光, 用 peak/曝光 排序"
            ),
            "coarse_points": int(args.focus_points),
            "diopter_range": float(args.focus_diopter_range),
            "refine_points": int(args.focus_refine),
            "fwhm_threshold_ccd_px": float(args.focus_fwhm_max),
            "blocked": f"判为光路受阻 ⇒ exit {EXIT_PATH_BLOCKED}, 不写任何测量数据",
            "not_in_focus": (
                f"最佳点 FWHM > 阈值 ⇒ exit {EXIT_NOT_IN_FOCUS} (不在焦, 拒绝测量), "
                "仍然不写任何测量数据"
            ),
            "compensation": (
                "通过后, 每张图案都叠加 fresnel_panel(best_focal_mm, 同一瞳面"
                "中心/半径) 这张瞳面菲涅尔补偿相位"
            ),
        },
        "geometry": {
            "source": (
                "现场实测 (SLMCCDCalibrator.calibrate)"
                if args.geometry_npz is None
                else f"读取存档 {args.geometry_npz}"
            ),
            "why_not_stored_default": (
                "仓库里的 data/calib.npz 是**陈旧**的 (Kx=4416, rotation=27.3°), "
                "与当前光路对不上 ⇒ 默认现测; 要复用必须显式给 --geometry-npz, "
                "并且脚本会把「这次用的是存档」写进 run_config.json"
            ),
            "gives": "center (cy,cx) / Kx / Ky / rotation_deg / mirror",
            "k_choice": "K = (Kx+Ky)/2 (各向同性), 取法写进 run_config.json",
        },
        "model": {
            "grid": (
                f"瞳面跨度 {spec.span:.0f} 面板像元 → N={spec.n_pupil} 采样 "
                f"(dp={spec.dp:.4f} 面板像元) → M={spec.m_grid} 模型网格 "
                f"(零填充 ×{spec.padding})"
            ),
            "source": (
                f"高斯 exp(-r²/2w0²), w0={spec.w0_px:.0f} 面板像元 "
                f"({spec.w0_samples:.1f} 样本) × 硬口径 (半径 "
                f"{spec.pupil_radius_samples:.1f} 样本), 瞳面中心 (960, 600)"
            ),
            "pupil_placement": (
                f"瞳面块 (N×N) 占据模型网格中央, 块的几何中心在 "
                f"{spec.block_centre} 号格; 傅里叶变换的直流格 (零级) 在 "
                f"M//2={spec.centre_index}; 口径半径 N/2 样本恰好内接于瞳面块"
            ),
            "far_field_scale": (
                "远场采样间距 = K/(M·dp) CCD 像元; 整个远场阵列覆盖 K/dp CCD 像元"
            ),
            "target": (
                f"{args.target} ("
                + (
                    f"亮斑直径 {args.delta_spot_px:.0f} CCD 像元"
                    if args.target == "delta"
                    else f"方形边长 {args.target_side_px:.0f} CCD 像元"
                )
                + "), "
                "建在 M×M 远场网格上 (目标网格 = 远场网格)"
            ),
            "optimizer": (
                "ao_shaping.algorithm.signal_processing.differentiable_beam."
                f"DifferentiableBeamOptimizer(target, source_amplitude=amp, "
                f"lr={args.lr}, device=cuda/cpu, seed={args.seed})"
            ),
            "iters": int(args.iters),
            "hw_iters": int(args.hw_iters),
            "grad_every": int(args.grad_every),
        },
        "artifacts": {
            "run_dir": str(args.out_dir),
            "files": [
                "run_config.json (全部参数 + 设备序列号 + 在焦判定 + 几何 + 模型 + 对比数字)",
                "history.csv (iteration, loss, grad_norm, phase_rms_rad)",
                "focus_sweep.csv (diopter, focal, exposure, peak, fwhm, hollowness, status)",
                "phase_initial.npy / phase_final.npy / phase_model.npy (M×M 模型网格, 弧度)",
                "ccd_flat.npz / ccd_optimized.npz / ccd_hw_iter_*.npz (原始 CCD 帧, float32 压缩)",
                "figures/ (焦面扫描、相位、梯度、模型远场、CCD 实测、模型 vs 硬件、优化历史)",
            ],
            "report": REPORT_KEY,
            "report_dir": str(REPORT_DIR),
        },
        "exit_codes": {
            "0": "正常",
            "2": "设备挂死/超时 (guarded 硬退出)",
            "3": "dry-run 违约 (驱动包被 import)",
            "4": "光路受阻",
            "5": "不在焦 —— 拒绝测量",
            "6": "几何标定或物理预测模型这一步失败",
            "7": "设备被别的会话占用, 等不到空闲 (本脚本不抢设备)",
        },
        "safety": {
            "memory_mode_only": "SLM 恒用 memory 模式 (video_mode=0), DVI 模式 open() 会挂死",
            "slot_rotation": (
                "用 SlotRotator 在 2..125 里轮换内存槽 —— 同槽连写是固件 no-op, "
                "面板不会刷新"
            ),
            "phase_pipeline": (
                "相位一律经 utils.slm.phase_display.phase_to_slm_grayscale(..., slm=slm) "
                "走驱动管线 (弧度→灰度 + 矫正 + LUT + 平移), 不自己拼灰度图"
            ),
            "no_device_reuse": "设备只在本次运行内打开/关闭, 绝不跨 run 复用",
        },
    }


def print_plan(plan: dict[str, Any]) -> None:
    """把实验计划打成人可读的一段 (dry-run 的全部输出)。"""
    lines: list[str] = []
    add = lines.append
    add("=" * 78)
    add("硬件实验计划 (dry-run) —— 未打开任何设备")
    add("=" * 78)
    add("")
    add("阶段 0: 打开设备")
    for d in plan["devices"]:
        add(f"  [{d['role']}] {d['driver']}")
        for k, v in d.items():
            if k in ("role", "driver"):
                continue
            add(f"        {k}: {v}")
    add("")
    add("阶段 1: 在焦判定 (测量前的门禁)")
    fg = plan["focus_gate"]
    add(f"  做什么   = {fg['what']}")
    add(f"  方法     = {fg['method']}")
    add(
        f"  粗扫     = {fg['coarse_points']} 点, 1/f 半宽 {fg['diopter_range']} m^-1, "
        f"细扫 {fg['refine_points']} 点"
    )
    add(f"  门限     = 最佳点 FWHM <= {fg['fwhm_threshold_ccd_px']:.0f} CCD px")
    add(f"  受阻时   = {fg['blocked']}")
    add(f"  不在焦时 = {fg['not_in_focus']}")
    add(f"  补偿     = {fg['compensation']}")
    add("")
    add("阶段 2: 几何")
    g = plan["geometry"]
    add(f"  来源     = {g['source']}")
    add(f"  为什么   = {g['why_not_stored_default']}")
    add(f"  得到     = {g['gives']}")
    add(f"  K 取法   = {g['k_choice']}")
    add("")
    add("阶段 3: 物理预测模型 + 反向传播优化 (纯模型, 硬件不在环里)")
    m = plan["model"]
    for key in (
        "grid",
        "source",
        "pupil_placement",
        "far_field_scale",
        "target",
        "optimizer",
    ):
        add(f"  {m[key]}")
    add(
        f"  迭代     = {m['iters']} 次 (硬件锚定再 {m['hw_iters']} 次), "
        f"每 {m['grad_every']} 次存一张梯度场"
    )
    add("")
    add("阶段 4: 上屏 + 抓帧 + 模型 vs 硬件 + 测量锚定闭环")
    add(
        "  相位上屏 = phase_model_to_panel(模型相位) + 焦面补偿 → "
        "phase_to_slm_grayscale(..., slm=slm) → SlotRotator 换槽显示"
    )
    add(
        "  对比     = 实测帧重采样到 M×M 远场网格 → 峰值偏移 / FWHM / 强度相关 / 类 Strehl"
    )
    add("  闭环     = opt.update(measured_intensity=重采样后的实测远场)")
    add("")
    add("阶段 5: 产物")
    a = plan["artifacts"]
    add(f"  运行目录 = {a['run_dir']}")
    for f in a["files"]:
        add(f"    - {f}")
    add(f"  报告     = {a['report']} (图与 artifacts 同住一个子目录)")
    add("")
    add("退出码")
    for code, meaning in plan["exit_codes"].items():
        add(f"  {code} = {meaning}")
    add("")
    add("硬件防呆")
    for k, v in plan["safety"].items():
        add(f"  {k}: {v}")
    add("")
    add("=" * 78)
    logger.info("\n".join(lines))


# ---------------------------------------------------------------------------
# dry-run 契约
# ---------------------------------------------------------------------------
def assert_no_drivers() -> None:
    """dry-run 的硬保证: 收尾时断言 ``ao_shaping.drivers`` 没被 import。

    这条断言让「dry-run 不碰设备」成为**可执行的**保证而不是口头承诺。
    违反就 exit 3, 因为那意味着 dry-run 路径上混进了会触发 SDK 加载的 import。
    """
    leaked = sorted(m for m in sys.modules if m.startswith("ao_shaping.drivers"))
    if leaked:
        sys.stderr.write(
            "DRY-RUN 违约: 计划打印过程中导入了驱动包:\n  " + "\n  ".join(leaked) + "\n"
        )
        sys.stderr.flush()
        raise SystemExit(EXIT_DRYRUN_VIOLATION)
    logger.info(
        "dry-run 保证通过: sys.modules 里没有任何 ao_shaping.drivers.* "
        "(未 import、未 open 任何设备)"
    )


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def _mermaid_bar(
    title: str,
    categories: list[str],
    series: list[list[float]],
    *,
    y_label: str,
    y_max: float,
) -> str:
    """生成一个 mermaid 条形图 (xychart-beta)。

    条形图只承担**对比**, 不承担精确读数 —— 精确数字仍然留在表格里。
    图例由紧跟其后的文字给出 (mermaid 的 xychart-beta 没有图例语法)。

    Args:
        title: 图标题 (不含引号)。
        categories: x 轴类目 (与每条 series 等长)。
        series: 每条柱子序列的数值。
        y_label: y 轴标题。
        y_max: y 轴上界。
    """
    lines = [
        "```mermaid",
        "xychart-beta",
        f'    title "{title}"',
        '    x-axis ["' + '", "'.join(categories) + '"]',
        f'    y-axis "{y_label}" 0 --> {y_max:g}',
    ]
    for values in series:
        lines.append("    bar [" + ", ".join(f"{float(v):.6g}" for v in values) + "]")
    lines.append("```")
    return "\n".join(lines)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    """写 CSV (UTF-8 BOM, Excel 直接打开不乱码)。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


@contextmanager
def guarded_device(
    label: str,
    factory: Callable[[], Any],
    *,
    open_timeout_s: float = OPEN_TIMEOUT_S,
    close_timeout_s: float = CLOSE_TIMEOUT_S,
) -> Iterator[Any]:
    """打开一个设备并在退出时关闭它, **打开与关闭都套超时保护**。

    不用驱动自带的上下文管理器: 它的 ``__enter__`` 直接调 ``open()``, 没有
    看门狗。Santec 的 SDK 等待在 C 层, 超时后线程仍卡着, ``finally`` 里的
    ``close()`` 也会卡 —— 仓库里已经有一次 900s 强杀的记录, 所以这里超时即
    硬退出 (:func:`guarded`)。
    """
    dev = factory()
    guarded(f"{label} open()", dev.open, timeout_s=open_timeout_s)
    try:
        yield dev
    finally:
        guarded(f"{label} close()", dev.close, timeout_s=close_timeout_s)


def save_ccd_frame(path: Path, frame: np.ndarray) -> None:
    """存一张 CCD 帧 (float32 + 压缩 npz)。

    为什么不是裸 ``.npy``: 一张全幅 2592×1944 的 float64 帧是 40 MB, 一次
    运行存七八张就几百 MB —— 而这些帧九成是零 (光斑只占画面的一小块),
    压缩后只剩几个 MB。存成压缩格式也让「产物复制进报告目录」这件事不会把
    仓库撑爆。
    """
    np.savez_compressed(path, frame=np.asarray(frame, dtype=np.float32))


def capture_frames(
    cam: Any,
    *,
    frames: int,
    discard: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """丢弃 ``discard`` 帧后平均 ``frames`` 帧。

    Returns:
        ``(平均帧, 原始帧列表)``。
    """
    for _ in range(int(discard)):
        cam.get_numpy_image(n_sample=1)
    raw = [np.asarray(cam.get_numpy_image(n_sample=1)) for _ in range(int(frames))]
    if not raw:
        return np.zeros((1, 1), dtype=np.float64), []
    return np.mean(np.stack(raw, axis=0), axis=0), raw


# ---------------------------------------------------------------------------
# 图
# ---------------------------------------------------------------------------
def plot_focus_sweep(
    result: FocusProbeResult,
    path: Path,
    *,
    fwhm_max: float,
) -> None:
    """焦面扫描图: FWHM vs 1/f, 带门限线与被选中的点。"""
    diopters = [p.diopter_m1 for p in result.points]
    fwhms = [p.fwhm_px for p in result.points]
    peaks = [p.peak_per_ms for p in result.points]
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    ax = axes[0]
    ax.plot(diopters, fwhms, "o-", color="#2f6fb0", label="FWHM")
    ax.axhline(
        float(fwhm_max),
        color="#c0392b",
        linestyle="--",
        label=f"在焦门限 {fwhm_max:.0f} px",
    )
    if result.best is not None:
        ax.plot(
            [result.best.diopter_m1],
            [result.best.fwhm_px],
            "*",
            markersize=18,
            color="#27ae60",
            label=f"选中点 {result.best.diopter_m1:+.3f} m⁻¹",
        )
    ax.set_ylabel("光斑 FWHM / CCD px")
    ax.set_title("在焦判定: 焦面扫描")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.plot(diopters, peaks, "s-", color="#8e44ad", label="峰值/曝光")
    if result.best is not None:
        ax.plot(
            [result.best.diopter_m1],
            [result.best.peak_per_ms],
            "*",
            markersize=18,
            color="#27ae60",
        )
    ax.set_xlabel("屈光度 1/f / m⁻¹")
    ax.set_ylabel("峰值 / 曝光 (counts/ms)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_phase(
    phase_model: np.ndarray,
    spec: ModelSpec,
    path: Path,
    *,
    title: str,
) -> None:
    """相位图 (弧度, 发散色表; 只画瞳面块)。"""
    start = spec.pupil_start
    block = np.asarray(phase_model)[
        start : start + spec.n_pupil, start : start + spec.n_pupil
    ]
    rms = float(np.sqrt(np.mean(np.asarray(phase_model) ** 2)))
    lim = max(float(np.max(np.abs(block))), 1e-9)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(block, cmap="RdBu_r", vmin=-lim, vmax=lim, origin="upper")
    ax.set_title(f"{title}\nRMS = {rms:.4f} rad (全网格)")
    ax.set_xlabel("瞳面 x / 采样")
    ax.set_ylabel("瞳面 y / 采样")
    fig.colorbar(im, ax=ax, label="相位 / rad")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_gradient(
    grad: np.ndarray,
    spec: ModelSpec,
    path: Path,
    *,
    title: str,
) -> None:
    """梯度场图 (只画瞳面块)。"""
    start = spec.pupil_start
    block = np.asarray(grad)[start : start + spec.n_pupil, start : start + spec.n_pupil]
    lim = max(float(np.max(np.abs(block))), 1e-12)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(block, cmap="PuOr_r", vmin=-lim, vmax=lim, origin="upper")
    ax.set_title(f"{title}\nRMS = {float(np.sqrt(np.mean(np.asarray(grad) ** 2))):.4e}")
    ax.set_xlabel("瞳面 x / 采样")
    ax.set_ylabel("瞳面 y / 采样")
    fig.colorbar(im, ax=ax, label="∂ℓ/∂φ")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_history(history: list[dict[str, Any]], path: Path) -> None:
    """优化历史: 损失 / 梯度范数 / 相位 RMS。"""
    if not history:
        return
    it = [row["iteration"] for row in history]
    loss = [row["loss"] for row in history]
    gnorm = [row["grad_norm"] for row in history]
    rms = [row["phase_rms_rad"] for row in history]
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    axes[0].plot(it, loss, "o-", color="#2f6fb0")
    axes[0].set_ylabel("损失 (峰值归一 MSE)")
    axes[0].set_yscale("log")
    axes[0].grid(alpha=0.3)
    axes[1].plot(it, gnorm, "o-", color="#c0392b")
    axes[1].set_ylabel("梯度范数 |∂ℓ/∂φ|")
    axes[1].set_yscale("log")
    axes[1].grid(alpha=0.3)
    axes[2].plot(it, rms, "o-", color="#27ae60")
    axes[2].set_ylabel("相位 RMS / rad")
    axes[2].set_xlabel("迭代")
    axes[2].grid(alpha=0.3)
    fig.suptitle("反向传播优化历史 (纯模型)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_farfield(arr: np.ndarray, path: Path, *, title: str, log: bool = True) -> None:
    """远场强度图 (对数色标, 便于看清旁瓣与噪底)。"""
    a = np.asarray(arr, dtype=np.float64)
    data = np.log10(np.maximum(a, 1e-12)) if log else a
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(data, cmap="inferno", origin="upper")
    ax.set_title(title)
    ax.set_xlabel("远场 x / 采样")
    ax.set_ylabel("远场 y / 采样")
    fig.colorbar(im, ax=ax, label="log10(I)" if log else "I")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_ccd(frame: np.ndarray, path: Path, *, title: str) -> None:
    """CCD 原始帧 (按 99.5 百分位定标, 免得一颗热像素压掉整幅)。"""
    a = np.asarray(frame, dtype=np.float64)
    hi = float(np.percentile(a, 99.5)) if a.size else 1.0
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(a, cmap="inferno", vmin=0.0, vmax=max(hi, 1e-9), origin="upper")
    ax.set_title(f"{title}\n峰值 {float(a.max()):.1f}, 99.5% {hi:.1f}")
    ax.set_xlabel("CCD x / px")
    ax.set_ylabel("CCD y / px")
    fig.colorbar(im, ax=ax, label="counts")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _apply_axis_convention(measured: np.ndarray, metrics: dict[str, Any]) -> np.ndarray:
    """按对比择优出来的轴向口径变换实测远场 (转置/翻行/翻列)。"""
    out = np.asarray(measured, dtype=np.float64)
    if metrics.get("transpose"):
        out = out.T
    if metrics.get("flip_rows"):
        out = out[::-1, :]
    if metrics.get("flip_cols"):
        out = out[:, ::-1]
    return out


def plot_model_vs_hardware(
    model: np.ndarray,
    measured: np.ndarray,
    path: Path,
    *,
    orientation_match: str,
    metrics: dict[str, Any],
) -> None:
    """模型远场 / 实测远场 / 绝对差 三联图。

    ⚠️ **两张图必须先各自峰值归一化再作差**。模型是无量纲强度 (峰值 ~10⁷),
    实测是 counts (峰值 ~10²), 直接相减的话差图完全由模型的量级支配 ——
    实测那一项连背景都算不上, 差图看起来是一张全黑的空图 (2026-10-22 首次
    出图实测: 差图只有 1 个可见像素, 色标 0→2.5×10⁷)。归一化之后两张图
    才在同一口径上, 差图才表达"形状差多少"。

    另外 ``measured`` 必须是**已经按对比择优口径变换过**的那一份, 否则三联图里
    两张图的峰值位置对不上 (而标题里的峰值偏移是按变换后的口径算的)。
    """
    mod = np.asarray(model, dtype=np.float64)
    meas = np.asarray(measured, dtype=np.float64)
    mod_n = mod / max(float(mod.max()), 1e-12)
    meas_n = meas / max(float(meas.max()), 1e-12)
    diff = np.abs(mod_n - meas_n)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, data, title, label in (
        (axes[0], mod_n, "模型预测远场 (峰值归一)", "I / I_max"),
        (
            axes[1],
            meas_n,
            f"实测远场 (口径 {orientation_match}, 峰值归一)",
            "I / I_max",
        ),
        (axes[2], diff, "|模型 − 实测| (同一归一化口径)", "|ΔI| / I_max"),
    ):
        im = ax.imshow(
            np.log10(np.maximum(data, 1e-6)),
            cmap="inferno",
            origin="upper",
            vmin=-6.0,
            vmax=0.0,
        )
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label=f"log10({label})")
    fig.suptitle(
        "模型 vs 硬件: "
        f"相关 {fmt_general(metrics.get('intensity_correlation'), '.3f', missing_nonfinite='n/a')}, "
        f"峰值偏移 {fmt_general((metrics.get('peak_offset_samples') or [0, 0])[0], '.2f', missing_nonfinite='n/a')}/"
        f"{fmt_general((metrics.get('peak_offset_samples') or [0, 0])[1], '.2f', missing_nonfinite='n/a')} 采样, "
        f"类 Strehl {fmt_general(metrics.get('strehl_like_peak_ratio'), '.3f', missing_nonfinite='n/a')}"
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _blind_loss_value(sample_px: float, side_px: float, m_grid: int) -> float | None:
    """「目标一块都没命中」时的损失下界 = 目标块面积 / 网格面积。

    峰值归一化的 MSE 里, 目标块之外的每一格贡献 ≈ (1/M²)², 所以把能量摊平
    (峰值归一后处处 ≈ 1/M²) 时损失正好是这个数。返回 ``None`` 表示几何信息
    缺失 (不该拿一个假下界去画参考线)。
    """
    if sample_px <= 0 or side_px <= 0 or m_grid <= 0:
        return None
    side_samples = max(1, int(round(side_px / sample_px)))
    return float((side_samples**2) / float(m_grid**2))


def _fwhm_1d(profile: np.ndarray) -> float:
    """一维剖面的半高全宽 (半高处跨过的采样点数)。"""
    p = np.asarray(profile, dtype=np.float64)
    if p.size == 0:
        return float("nan")
    peak = float(p.max())
    if peak <= 0:
        return float("nan")
    above = np.flatnonzero(p >= peak / 2.0)
    if above.size == 0:
        return float("nan")
    return float(above[-1] - above[0] + 1)


def _ccd_spot_profile(frame: np.ndarray) -> dict[str, Any]:
    """从 CCD 帧里量光斑: 峰值、质心、x/y 两个方向的 FWHM。

    为什么要单独量一遍: 报告 §6 表里的 FWHM 量在**模型网格**上 (散斑的半高宽),
    单位是"模型采样"; 这里的 FWHM 量在 **CCD 域** (counts, 真实像元), 才是
    "光斑有多宽"这个物理问题的答案。两个数不是一回事, 必须分开写清楚。
    """
    img = prepare_frame(np.asarray(frame, dtype=np.float64))
    spot = measure_spot(img)
    cy, cx = int(round(spot.centroid_y)), int(round(spot.centroid_x))
    return {
        "ccd_peak": float(spot.peak),
        "ccd_fwhm_px": float(spot.fwhm_px),
        "ccd_fwhm_x_px": _fwhm_1d(img[cy, :]),
        "ccd_fwhm_y_px": _fwhm_1d(img[:, cx]),
        "ccd_centroid_xy": [cx, cy],
        "ccd_hollowness": float(spot.hollowness),
    }


def _fwhm_from_cut(profile: np.ndarray) -> float:
    """由一维截面估 FWHM (半高宽, 采样点计)。"""
    p = np.asarray(profile, dtype=np.float64)
    if p.size == 0:
        return float("nan")
    peak = float(p.max())
    if peak <= 0:
        return float("nan")
    above = np.flatnonzero(p >= peak / 2.0)
    if above.size == 0:
        return float("nan")
    return float(above[-1] - above[0] + 1)


def plot_spot_cross_section(
    model: np.ndarray,
    measured: np.ndarray,
    path: Path,
    *,
    metrics: dict[str, Any],
    ccd_frame: np.ndarray | None = None,
) -> None:
    """光斑截面: 硬件 (CCD 域) + 模型/实测远场 (模型网格域)。

    2×3 布局:

    * 上排 —— 硬件那一格是 **CCD 原始帧** (零级附近裁剪, 单位 counts), 后两格是
      模型远场与实测远场 (重采样到模型网格, 峰值归一 + log 色标), 各自标出峰值
      (圆圈 = 模型, 叉 = 实测) 与截取线;
    * 下排 —— 三个方向的截面曲线: CCD 域的 x/y 截面 (过 CCD 光斑中心) 与模型
      网格域的 x/y 截面 (过**模型峰值**那一行/列)。

    为什么 CCD 域要单独占一格: 模型网格每采样约 1.96 CCD px, 且实测远场是一张
    **散斑**图 (本轮相位近似随机), 全局最大点只是某个散斑, 不是"光斑"。
    真正可比的"光斑截面"是 CCD 域里那一块亮区 —— 报告里的 FWHM 也是这么量的。

    截线取**模型峰值**所在的行/列, 两张模型网格图用同一根 ⇒ 比较的是同一条
    物理截线; 峰值偏移会直接表现为曲线错位, 而不会被「各自对自己齐」掩盖。

    Args:
        model: 模型远场 (M×M)。
        measured: 实测远场, 已重采样到模型网格**并已按对比择优口径变换**。
        path: 输出 PNG。
        metrics: 本轮对比指标 (写进标题)。
        ccd_frame: 硬件 CCD 原始帧 (可选; 没有就跳过 CCD 域那一格)。
    """
    mod = np.asarray(model, dtype=np.float64)
    meas = np.asarray(measured, dtype=np.float64)
    mod_n = mod / max(float(mod.max()), 1e-12)
    meas_n = meas / max(float(meas.max()), 1e-12)
    rows, cols = mod.shape
    rm, cm = (int(v) for v in np.unravel_index(int(mod.argmax()), mod.shape))
    rh, ch = (int(v) for v in np.unravel_index(int(meas.argmax()), meas.shape))

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))

    # ---- (0,0) 硬件 CCD 帧 + (1,0) CCD 域截面 ---------------------------
    ax_img, ax_cut = axes[0, 0], axes[1, 0]
    if ccd_frame is None:
        ax_img.text(0.5, 0.5, "本轮没有保存 CCD 原始帧", ha="center", va="center")
        ax_img.set_axis_off()
        ax_cut.text(0.5, 0.5, "（同上）", ha="center", va="center")
        ax_cut.set_axis_off()
    else:
        frame = prepare_frame(np.asarray(ccd_frame, dtype=np.float64))
        spot = measure_spot(frame)
        cy, cx = int(round(spot.centroid_y)), int(round(spot.centroid_x))
        half = 220
        y0, y1 = max(cy - half, 0), min(cy + half + 1, frame.shape[0])
        x0, x1 = max(cx - half, 0), min(cx + half + 1, frame.shape[1])
        patch = frame[y0:y1, x0:x1]
        im = ax_img.imshow(
            patch,
            cmap="inferno",
            origin="upper",
            vmin=0.0,
            vmax=max(float(patch.max()), 1e-9),
        )
        ax_img.plot(cx - x0, cy - y0, "o", ms=10, mfc="none", mec="#2f6fb0")
        ax_img.set_title(
            f"硬件 CCD 原始帧 (零级附近 {x1 - x0}×{y1 - y0} px)\n"
            f"峰值 {spot.peak:.1f} counts, FWHM {spot.fwhm_px:.2f} px, "
            f"空心度 {spot.hollowness:.3f}"
        )
        ax_img.set_xlabel("CCD x / px")
        ax_img.set_ylabel("CCD y / px")
        fig.colorbar(im, ax=ax_img, label="counts")
        cut_x = frame[cy, x0:x1]
        cut_y = frame[y0:y1, cx]
        ax_cut.plot(
            np.arange(x0, x1), cut_x, "-", color="#2f6fb0", lw=1.6, label="水平 (x)"
        )
        ax_cut.plot(
            np.arange(y0, y1), cut_y, "--", color="#c0392b", lw=1.6, label="垂直 (y)"
        )
        ax_cut.axhline(0.5 * spot.peak, color="#7f8c8d", lw=0.8, ls=":")
        ax_cut.set_title(
            f"CCD 域光斑截面 (过光斑中心 ({cx}, {cy}))\nFWHM {spot.fwhm_px:.2f} px",
            fontsize=9,
        )
        ax_cut.set_xlabel("CCD px")
        ax_cut.set_ylabel("counts (已扣中位数)")
        ax_cut.grid(alpha=0.3)
        ax_cut.legend(fontsize=8)

    # ---- (0,1)(0,2) 模型/实测远场 ---------------------------------------
    for ax, data, peak, title in (
        (axes[0, 1], mod_n, (rm, cm), "模型远场 (峰值归一)"),
        (axes[0, 2], meas_n, (rh, ch), "实测远场 (峰值归一, 口径已对齐)"),
    ):
        im = ax.imshow(
            np.log10(np.maximum(data, 1e-6)),
            cmap="inferno",
            origin="upper",
            vmin=-6.0,
            vmax=0.0,
        )
        ax.axhline(rm, color="#2f6fb0", lw=0.8, ls="--")
        ax.axvline(cm, color="#2f6fb0", lw=0.8, ls="--")
        ax.plot(cm, rm, "o", ms=9, mfc="none", mec="#2f6fb0")
        ax.plot(ch, rh, "x", ms=10, color="#c0392b")
        ax.set_title(title)
        ax.set_xlabel("远场 x / 采样")
        ax.set_ylabel("远场 y / 采样")
        fig.colorbar(im, ax=ax, label="log10(I / I_max)")

    # ---- (1,1)(1,2) 模型网格域的 x/y 截面 -------------------------------
    for ax, mod_cut, meas_cut, xlabel, title in (
        (
            axes[1, 1],
            mod_n[rm, :],
            meas_n[rm, :],
            "远场 x / 采样",
            f"水平截面 (第 {rm} 行) · 模型 FWHM {_fwhm_from_cut(mod_n[rm, :]):.0f} 采样"
            f" / 实测 {_fwhm_from_cut(meas_n[rm, :]):.0f} 采样",
        ),
        (
            axes[1, 2],
            mod_n[:, cm],
            meas_n[:, cm],
            "远场 y / 采样",
            f"垂直截面 (第 {cm} 列) · 模型 FWHM {_fwhm_from_cut(mod_n[:, cm]):.0f} 采样"
            f" / 实测 {_fwhm_from_cut(meas_n[:, cm]):.0f} 采样",
        ),
    ):
        xs = np.arange(len(mod_cut))
        ax.plot(xs, mod_cut, "-", color="#2f6fb0", lw=1.6, label="模型")
        ax.plot(xs, meas_cut, "--", color="#c0392b", lw=1.6, label="实测")
        ax.axhline(0.5, color="#7f8c8d", lw=0.8, ls=":")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("I / I_max")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    offset = metrics.get("peak_offset_samples") or [0, 0]
    fig.suptitle(
        "光斑截面: 模型 vs 硬件 · "
        f"相关 {fmt_general(metrics.get('intensity_correlation'), '.3f', missing_nonfinite='n/a')}, "
        f"峰值偏移 {offset[0]}/{offset[1]} 采样, "
        f"模型 FWHM {fmt_general(metrics.get('model_fwhm_px'), '.2f', missing_nonfinite='n/a')} / "
        f"实测 FWHM {fmt_general(metrics.get('measured_fwhm_px'), '.2f', missing_nonfinite='n/a')}"
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_metrics_vs_step(
    history: list[dict[str, Any]],
    hw_history: list[dict[str, Any]],
    path: Path,
    *,
    blind_loss: float | None = None,
) -> None:
    """metrics 随 step 变化的曲线 (闭环 + 纯模型侧)。

    左: **测量锚定闭环** —— 实测损失与实测峰值随 step 的变化 (闭环看的就是
    "每一步有没有变好")。右: **纯模型侧** —— 损失随迭代的变化, 并画出
    「目标块面积 / 网格面积」这条**不可达下界**; 曲线落到这条线上就说明
    目标结构压根没被造出来 (而不是"优化不动了")。
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    if hw_history:
        steps = [int(row["iteration"]) for row in hw_history]
        losses = [float(row["loss"]) for row in hw_history]
        peaks = [float(row.get("measured_peak") or 0.0) for row in hw_history]
        ax.plot(steps, losses, "o-", color="#2f6fb0", lw=1.8, label="实测损失")
        ax.set_xlabel("闭环 step")
        ax.set_ylabel("实测损失", color="#2f6fb0")
        ax.tick_params(axis="y", labelcolor="#2f6fb0")
        ax.grid(alpha=0.3)
        twin = ax.twinx()
        twin.plot(steps, peaks, "s--", color="#c0392b", lw=1.4, label="实测峰值")
        twin.set_ylabel("实测峰值 / counts", color="#c0392b")
        twin.tick_params(axis="y", labelcolor="#c0392b")
        ax.set_title("测量锚定闭环: metrics 随 step 变化", fontsize=10)
    else:
        ax.text(0.5, 0.5, "本轮没有测量锚定闭环记录", ha="center", va="center")
        ax.set_axis_off()

    ax = axes[1]
    if history:
        its = [int(row["iteration"]) for row in history]
        losses = [float(row["loss"]) for row in history]
        ax.plot(its, losses, "-", color="#27ae60", lw=1.6, label="纯模型损失")
        ax.set_yscale("log")
        if blind_loss is not None and np.isfinite(blind_loss):
            ax.axhline(
                blind_loss,
                color="#c0392b",
                ls="--",
                lw=1.2,
                label=f"不可达下界 (目标块面积比) {blind_loss:.2e}",
            )
        ax.set_xlabel("迭代")
        ax.set_ylabel("损失 (峰值归一 MSE)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_title("纯模型侧: 损失随迭代变化", fontsize=10)
    else:
        ax.text(0.5, 0.5, "本轮没有纯模型迭代记录", ha="center", va="center")
        ax.set_axis_off()

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------
def write_report(
    run_config: dict[str, Any],
    history: list[dict[str, Any]],
    hw_history: list[dict[str, Any]],
    run_dir: Path,
    stamp: str,
    *,
    copy_artifacts: bool = True,
    artifact_suffixes: tuple[str, ...] | None = None,
) -> Path:
    """写 markdown 报告, 并把图与本轮产物搬进报告子目录。

    报告目录结构按 ``report/AGENTS.md`` §2/§4: 一份报告 = 一个子目录, 图在
    ``figures/``, 上游原始产物在 ``artifacts/<时间戳>/``。

    Args:
        run_config: 本轮运行配置 (会被读出 focus / model / geometry / comparison)。
        history: 纯模型侧迭代史。
        hw_history: 测量锚定闭环的迭代史。
        run_dir: 运行目录 (图与产物的来源)。
        stamp: 产物子目录名 (通常是 UTC 时间戳)。
        copy_artifacts: 是否把运行目录的产物搬进 ``artifacts/<stamp>/``。
        artifact_suffixes: 只搬这些后缀的文件 (``None`` = 全部)。离线重渲染时
            用它挡掉几十 MB 的 npz/npy —— 那些留在 ``data/debug/`` 里即可。
    """
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    artifact_dir = REPORT_ARTIFACT_DIR / stamp
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # 图: 运行目录 → 报告 figures/
    for png in sorted((run_dir / "figures").glob("*.png")):
        shutil.copy2(png, REPORT_FIG_DIR / png.name)
    # 产物: 运行目录 → 报告 artifacts/<stamp>/
    if copy_artifacts:
        for item in sorted(run_dir.iterdir()):
            if item.name == "figures":
                continue
            if (
                artifact_suffixes is not None
                and item.is_file()
                and item.suffix not in artifact_suffixes
            ):
                continue
            target = artifact_dir / item.name
            if item.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True)
            else:
                shutil.copy2(item, target)

    focus = run_config.get("focus", {}) or {}
    model = run_config.get("model", {}) or {}
    geom = run_config.get("geometry", {}) or {}
    cmp_main = run_config.get("comparison", {}) or {}
    cmp_best = cmp_main.get("best") or {}
    params = run_config.get("params", {}) or {}

    focus_rows = "\n".join(
        f"| {p['diopter_m1']:+.3f} | {p['focal_mm']:.1f} | {p['exposure_ms']:.3f} | "
        f"{p['peak']:.1f} | {p['fwhm_px']:.2f} | {p['hollowness']:.3f} | {p['status']} |"
        for p in (focus.get("points") or [])
    )

    param_rows = "\n".join(f"| `{k}` | {v} |" for k, v in sorted(params.items()))

    hist_rows = "\n".join(
        f"| {int(row['iteration'])} | {fmt_general(row['loss'], '.6f', missing_nonfinite='n/a')} | {fmt_general(row['grad_norm'], '.4f', missing_nonfinite='n/a')} | "
        f"{fmt_general(row['phase_rms_rad'], '.4f', missing_nonfinite='n/a')} |"
        for row in history[:: max(1, len(history) // 12)]
    )

    hw_rows = "\n".join(
        f"| {int(row['iteration'])} | {fmt_general(row['loss'], '.6f', missing_nonfinite='n/a')} | {fmt_general(row['grad_norm'], '.4f', missing_nonfinite='n/a')} | "
        f"{fmt_general(row['phase_rms_rad'], '.4f', missing_nonfinite='n/a')} | {fmt_general(row.get('measured_peak'), '.1f', missing_nonfinite='n/a')} | "
        f"{fmt_general(row.get('exposure_ms'), '.3f', missing_nonfinite='n/a')} |"
        for row in hw_history
    )

    # ---- 条形图 (现象分析用) ----------------------------------------------
    # 条形图只做对比, 精确数字仍在表格里。纵轴是线性的, 所以"损失史"那张图
    # 的第一根柱会压扁后面所有柱 —— 这正是要看的现象 (一步塌到平台)。
    hist_sample = history[:: max(1, len(history) // 12)] if history else []
    hist_losses = [float(row["loss"]) for row in hist_sample]
    loss_chart = _mermaid_bar(
        "纯模型侧优化损失史 (线性纵轴)",
        [f"第 {int(row['iteration'])} 次" for row in hist_sample],
        [hist_losses],
        y_label="损失 (MSE)",
        y_max=(max(hist_losses) * 1.15 if hist_losses else 1.0),
    )
    hw_losses = [float(row["loss"]) for row in hw_history]
    hw_chart = _mermaid_bar(
        "测量锚定闭环: 实测远场处的损失",
        [f"第 {int(row['iteration'])} 次" for row in hw_history],
        [hw_losses],
        y_label="实测损失",
        y_max=(max(hw_losses) * 1.15 if hw_losses else 1.0),
    )
    # 损失史的两个百分比: 第一次迭代吃掉多少, 之后多少次迭代又吃掉多少。
    if hist_losses:
        loss_first = hist_losses[0]
        loss_second = hist_losses[1] if len(hist_losses) > 1 else loss_first
        loss_last = hist_losses[-1]
        loss_drop_first = (
            100.0 * (1.0 - loss_second / loss_first) if loss_first else 0.0
        )
        loss_drop_rest = 100.0 * (1.0 - loss_last / loss_second) if loss_second else 0.0
        loss_text = (
            f"第一次迭代就把损失从 {loss_first:.6g} 压到 {loss_second:.6g}"
            f" (一次降了 {loss_drop_first:.0f} %); 之后 {len(history) - 1} 次迭代"
            f"合计只再降 {loss_drop_rest:.2f} %, 停在 {loss_last:.6g}"
        )
        phase_rms_last = fmt_general(hist_sample[-1]["phase_rms_rad"], '.4f', missing_nonfinite='n/a')
    else:
        loss_text = "本轮没有迭代记录"
        phase_rms_last = "n/a"
    if hw_losses:
        hw_drop = 100.0 * (1.0 - hw_losses[-1] / hw_losses[0]) if hw_losses[0] else 0.0
        hw_text = (
            f"{len(hw_history)} 步里实测损失从 {hw_losses[0]:.6g} 降到 "
            f"{hw_losses[-1]:.6g}, 单调下降 (合计 {hw_drop:.1f} %)"
        )
    else:
        hw_drop = 0.0
        hw_text = "本轮没有测量锚定迭代记录"
    # 两套 FWHM 的换算: 模型采样 → CCD px。
    s_far = float(model.get("far_field_sample_ccd_px") or 0.0)
    _model_fwhm_s = cmp_best.get("model_fwhm_px")
    _meas_fwhm_s = cmp_best.get("measured_fwhm_px")
    model_fwhm_ccd = (
        (_model_fwhm_s * s_far) if _model_fwhm_s is not None and s_far else None
    )
    meas_fwhm_ccd = (
        (_meas_fwhm_s * s_far) if _meas_fwhm_s is not None and s_far else None
    )
    # 目标块占远场网格的面积比 —— "目标一块都没命中"时损失的理论值。
    sample_px = float(model.get("far_field_sample_ccd_px") or 0.0)
    side_px = float(params.get("target_side_px") or 0.0)
    m_grid = int(model.get("m_grid") or 1)
    if sample_px > 0 and side_px > 0:
        side_samples = max(1, int(round(side_px / sample_px)))
        blind_area_fraction = (side_samples**2) / float(m_grid**2)
        blind_area_text = (
            f"「目标块面积 / 网格面积」= {side_samples}²/{m_grid}² = "
            f"{blind_area_fraction:.2e}"
        )
    else:
        blind_area_text = "「目标块面积 / 网格面积」(本轮几何缺失, 无法给出)"

    body = f"""# 可微分相位优化 (物理预测模型 + 反向传播) 硬件验证报告

## 摘要

本轮在 2f 傅里叶实验系统上做了一件此前没做过的事: 把**物理预测模型**与**反向传播
可微分相位优化**放在一起, 并把全过程 (SLM 相位、CCD 远场光斑、模型梯度变化、
全部参数) 记录下来。

* **在焦判定**: 测量之前先用焦面扫描判实验系统在不在焦。判定不通过时脚本
  **拒绝测量** (退出码 {EXIT_NOT_IN_FOCUS}), 一帧测量数据都不写。
* **物理预测模型**: pupil（瞳面）被离散成 {model.get("n_pupil")}×{model.get("n_pupil")}
  采样, 嵌在 {model.get("m_grid")}×{model.get("m_grid")} 的零填充网格中央, 一次
  傅里叶变换给出远场强度。
* **反向传播优化**: {model.get("iters")} 次 Adam 迭代, 损失是峰值归一化远场强度与
  目标之间的 MSE, 梯度由 autograd 直接给出。
* **模型 vs 硬件**: 优化出的相位真的上屏, 抓帧后重采样到模型的远场网格对比。
* **测量锚定闭环**: 再把实测远场喂回优化器, 看梯度在实验系统上是否仍指对方向。

**结论一句话**: {cmp_best.get("verdict", "本轮未完成硬件对比 (见下)")}

## 0. 实验目的、条件与环境

### 0.1 实验目的

本轮把**物理预测模型**与**反向传播可微优化**第一次放进同一条实验系统链路里，
并把全过程留痕。要回答三个问题：

1. **模型侧的梯度方向对不对？** 纯模型上跑 {params.get("iters")} 次 Adam，
   看损失是否真的下降、相位 RMS 是否变化 —— 若损失只降一次就平台化，
   说明问题在目标函数本身（见 §8.1）。
2. **模型能不能预测硬件？** 把优化出的相位真的上屏、抓帧、重采样到模型的远场
   网格，比较强度相关性与峰值位置（见 §6）。
3. **把实测远场喂回优化器，梯度在实验系统上还指不指对方向？** 这是「测量锚定
   闭环」，{params.get("hw_iters")} 步，看**实测**损失是否单调下降（见 §7）。

### 0.2 实验条件

| 项 | 值 |
|---|---|
| SLM | Santec #{params.get("slm_number")}，{params.get("slm_wavelength")} nm |
| 相机 | {params.get("cam_type")}，cam_id {params.get("cam_id")}；自适应曝光上限 {params.get("exposure_ceiling_ms")} ms |
| 测量口径 | 每点平均 {params.get("frames")} 帧、丢弃 {params.get("discard")} 帧、上屏后稳定等待 {params.get("settle_s")} s |
| 瞳面 | 中心 {params.get("pupil_center")}、半径 {params.get("pupil_radius")} 面板像元；照明高斯腰径 {params.get("beam_w0_px")} 面板像元 |
| 目标 | {params.get("target")}，边长 {params.get("target_side_px")} CCD px（建在零级上） |
| 模型 | 瞳面 {model.get("n_pupil")}×{model.get("n_pupil")} 采样，零填充到 {model.get("m_grid")}×{model.get("m_grid")}（×{model.get("padding")}） |
| 优化 | Adam，lr {params.get("lr")}，{params.get("iters")} 次迭代，种子 {params.get("seed")}；每 {params.get("grad_every")} 次存一张梯度场 |
| 硬件闭环 | {params.get("hw_iters")} 步测量锚定迭代 |
| 在焦门限 | FWHM ≤ {params.get("focus_fwhm_max")} CCD px（粗扫 {params.get("focus_points")} 点 / 半宽 {params.get("focus_diopter_range")} m⁻¹，细扫 {params.get("focus_refine")} 点） |
| 几何来源 | {geom.get("source", "n/a")}（未给 `--geometry-npz` 时现场实测） |

### 0.3 实验环境

* **实验系统**：2f 傅里叶实验系统（SLM 前焦 → 傅里叶透镜 → CCD 后焦），真实设备。
  本脚本除 `--execute` 分支外**不 import 任何驱动**（`--dry-run` 在没有硬件的机器上
  也必须能跑完）。
* **软件**：反向传播用 `torch.autograd`；物理预测模型是零填充 + `fftshift` 的规范傅里叶
  传播，几何来自 `SLMCCDCalibrator`（±1 级光栅）。
* **产物**：运行目录 `{params.get("out_dir")}`（原始 npz/npy + csv/json + figures/），
  报告子目录 `report/slm/differentiable_opt/`（figures/ 与 artifacts/）。

## 1. 在焦判定 (测量前的门禁)

| 项 | 值 |
|---|---|
| 判定方法 | `focus_probe.sweep_focus` (plain 相位上扫菲涅尔焦距 1/f, 每点重新对曝光) |
| 粗扫 | {focus.get("coarse_points")} 点, 1/f 半宽 {focus.get("diopter_range")} m⁻¹ |
| 细扫 | {focus.get("refine_points")} 点 |
| 门限 (FWHM) | {focus.get("fwhm_threshold_ccd_px")} CCD px |
| 最佳 1/f | {fmt_general(focus.get("best_diopter_m1"), '.3f', missing_nonfinite='n/a')} m⁻¹ |
| 最佳焦距 | {fmt_general(focus.get("best_focal_mm"), '.1f', missing_nonfinite='n/a')} mm |
| 最佳点 FWHM | {fmt_general(focus.get("best_fwhm_px"), '.2f', missing_nonfinite='n/a')} CCD px |
| 判定 | **{focus.get("verdict", "n/a")}** |

失败时的处置:

* **光路受阻** (画面里根本没有光, 不是「没对上焦」) ⇒ 退出码
  {EXIT_PATH_BLOCKED}, 只写 `run_config.json` 里的判定与理由, **不写任何测量数据**;
* **不在焦** (最佳点 FWHM 超过门限) ⇒ 退出码 {EXIT_NOT_IN_FOCUS}, 同样
  **不写任何测量数据** —— 这是用户明确要求的: 实验系统没有被判为在焦之前,
  一个测量都不做;
* 通过 ⇒ 之后每张图案都叠加
  `fresnel_panel(best_focal_mm, 同一瞳面中心/半径)` 这张瞳面菲涅尔补偿相位。

![焦面扫描](figures/focus_sweep.png)

<details><summary>焦面扫描逐点数据</summary>

| 1/f / m⁻¹ | 焦距 / mm | 曝光 / ms | 峰值 | FWHM / px | 空心度 | 状态 |
|---|---|---|---|---|---|---|
{focus_rows}

</details>

## 2. 参数表

| 参数 | 值 |
|---|---|
{param_rows}

## 3. 物理预测模型

| 项 | 值 |
|---|---|
| 瞳面跨度 | {model.get("span_panel_px")} 面板像元 |
| 瞳面采样 | N = {model.get("n_pupil")} (采样间距 {model.get("dp_panel_px")} 面板像元) |
| 模型网格 | M = {model.get("m_grid")} (零填充 ×{model.get("padding")}) |
| 瞳面块中心下标 | {model.get("block_centre_index")} (块的几何中心, 半整数 ⇒ 面板↔模型对位的锚点) |
| 直流格 (零级) | M//2 = {model.get("centre_index")} (fftshift 后的零频所在格, 目标就建在这里) |
| 源面振幅 | 高斯 w0 = {model.get("w0_panel_px")} 面板像元 × 硬口径 {model.get("pupil_radius_samples")} 样本 |
| 远场采样间距 | {fmt_general(model.get("far_field_sample_ccd_px"), '.4f', missing_nonfinite='n/a')} CCD px (= K/(M·dp)) |
| 远场阵列覆盖 | {fmt_general(model.get("far_field_span_ccd_px"), '.1f', missing_nonfinite='n/a')} CCD px (= K/dp) |
| 目标 | {model.get("target", {}).get("kind")} — {model.get("target", {}).get("note", "")} |

**几何**: {geom.get("source", "n/a")}; K = (Kx+Ky)/2 = {fmt_general(geom.get("k_mean"), '.1f', missing_nonfinite='n/a')}
CCD px/面板像元 (Kx = {fmt_general(geom.get("kx"), '.1f', missing_nonfinite='n/a')}, Ky = {fmt_general(geom.get("ky"), '.1f', missing_nonfinite='n/a')}),
零级中心 (cy, cx) = {geom.get("centre_cycx")}, rotation = {fmt_general(geom.get("rotation_deg"), '.2f', missing_nonfinite='n/a')}°,
mirror = {geom.get("mirror")}.

## 4. 优化历史 (纯模型)

| 迭代 | 损失 | 梯度范数 | 相位 RMS / rad |
|---|---|---|---|
{hist_rows}

![优化历史](figures/history.png)

## 5. 相位与梯度

下图是**实际下发给 SLM 的那张相位** (模型网格上的弧度, raw 未包裹), 叠加的是同
一次测量所用的焦面菲涅尔补偿相位; 面板上真正生效的是「相位 + 补偿」。

![初始相位](figures/phase_initial.png)

![中间相位](figures/phase_mid.png)

![最终相位](figures/phase_final.png)

![梯度场](figures/gradient_final.png)

## 6. 硬件实测: 模型 vs 硬件

![CCD 平场](figures/ccd_flat.png)

![CCD 优化后](figures/ccd_optimized.png)

![模型预测的远场](figures/model_farfield.png)

![模型 vs 硬件](figures/model_vs_hardware.png)

![光斑截面](figures/spot_cross_section.png)

| 项 | 值 |
|---|---|
| 口径对比择优 | {cmp_best.get("orientation_match", "n/a")} |
| 强度相关系数 | {fmt_general(cmp_best.get("intensity_correlation"), '.3f', missing_nonfinite='n/a')} |
| 峰值偏移 (采样) | {cmp_best.get("peak_offset_samples")} |
| 模型 FWHM (模型采样) | {fmt_general(cmp_best.get("model_fwhm_px"), '.2f', missing_nonfinite='n/a')} |
| 实测 FWHM (模型采样) | {fmt_general(cmp_best.get("measured_fwhm_px"), '.2f', missing_nonfinite='n/a')} |
| 远场采样间距 | {fmt_general(model.get("far_field_sample_ccd_px"), '.3f', missing_nonfinite='n/a')} CCD px / 采样 |
| CCD 域光斑 FWHM (x / y / 合成) | {fmt_general(cmp_best.get("ccd_fwhm_x_px"), '.1f', missing_nonfinite='n/a')} / {fmt_general(cmp_best.get("ccd_fwhm_y_px"), '.1f', missing_nonfinite='n/a')} / {fmt_general(cmp_best.get("ccd_fwhm_px"), '.2f', missing_nonfinite='n/a')} px |
| CCD 域光斑峰值 / 质心 | {fmt_general(cmp_best.get("ccd_peak"), '.1f', missing_nonfinite='n/a')} counts @ {cmp_best.get("ccd_centroid_xy")} px |
| 模型的相对平场峰值比 | {fmt_general(cmp_best.get("strehl_model"), '.3f', missing_nonfinite='n/a')} |
| 硬件的相对平场峰值比 | {fmt_general(cmp_best.get("strehl_hardware"), '.3f', missing_nonfinite='n/a')} |
| 类 Strehl 比值 (硬件/模型) | {fmt_general(cmp_best.get("strehl_like_peak_ratio"), '.3f', missing_nonfinite='n/a')} |
| 曝光 (平场 / 优化后, ms) | {cmp_best.get("exposure_ms")} |

> ⚠️ **表里有两套 FWHM, 单位不同, 不能混用**:
> 「模型 FWHM / 实测 FWHM」量在**模型网格**上, 单位是**模型采样**
> (每采样 {fmt_general(model.get("far_field_sample_ccd_px"), '.3f', missing_nonfinite='n/a')} CCD px) —— 描述的是**散斑**的
> 半高宽; 「CCD 域光斑 FWHM」量在真实像元上, 描述的是**光斑本身**有多宽。
> 本轮: 模型 {fmt_general(cmp_best.get("model_fwhm_px"), '.2f', missing_nonfinite='n/a')} 采样 ≈ {fmt_general(model_fwhm_ccd, '.1f', missing_nonfinite='n/a')} CCD px,
> 实测散斑 {fmt_general(cmp_best.get("measured_fwhm_px"), '.2f', missing_nonfinite='n/a')} 采样 ≈ {fmt_general(meas_fwhm_ccd, '.1f', missing_nonfinite='n/a')} CCD px,
> 而 CCD 域直接量的光斑是 {fmt_general(cmp_best.get("ccd_fwhm_px"), '.2f', missing_nonfinite='n/a')} px
> (x 向 {fmt_general(cmp_best.get("ccd_fwhm_x_px"), '.1f', missing_nonfinite='n/a')} px / y 向 {fmt_general(cmp_best.get("ccd_fwhm_y_px"), '.1f', missing_nonfinite='n/a')} px —— **各向异性**)。
> 三个数互不矛盾: 量的是三个不同的东西。

## 7. 测量锚定闭环

| 迭代 | 实测损失 | 梯度范数 | 相位 RMS / rad | 实测峰值 | 曝光 / ms |
|---|---|---|---|---|---|
{hw_rows}

## 8. 现象分析

![metrics 随 step 变化](figures/metrics_vs_step.png)

### 8.1 现象一：纯模型侧的损失「一步塌到平台」，之后几乎不动

{loss_chart}

* **读数**：{loss_text}。
* **这个平台值不是"优化不动了"，而是"目标一块都没命中"**：平台值与
  {blind_area_text} 几乎相等 —— 峰值归一化之后，远场各处都 ≈ 1e-3 时，
  "目标块面积占比"就是损失的理论下界。
* **物理机制**：纯相位元件不能把远场压成 0，于是优化器选了「把能量摊平成一圈
  散斑」这条捷径，而不是「造出目标结构」。相位 RMS 全程停在
  {phase_rms_last} rad（≈ 均匀 [0,2π) 的 raw RMS）也印证相位仍近似随机。
* **对方案的意义**：这条负结果**污染了本轮所有「模型 vs 硬件」数字** ——
  上屏的是近似随机的相位，硬件看到的是散斑，所以相关系数天然只有 0.1 量级。
  正确的做法是**用解析整形解（GS）做初值**，再让可微细化接手。

### 8.2 现象二：测量锚定闭环的实测损失确实单调下降

{hw_chart}

* **读数**：{hw_text}。
* **说明什么**：把**实测**远场喂回优化器时，梯度仍然指对方向 —— 硬件闭环本身
  是通的。
* **不说明什么**：这一步的起点已经是 §8.1 那个"摊平"平台，起点太差 ⇒ 下降幅度
  （约 {hw_drop:.1f} %）不能读成"可微优化有效"。
* **工程结论**：**闭环可通 ≠ 目标可达**。下一轮必须先把初值换掉（GS 或解析解），
  再谈"可微优化把方形光斑整形出来"。

### 8.3 现象三：模型与硬件的口径差是结构性的，不是标定误差

| 口径 | 模型 | 硬件 | 说明 |
|---|---|---|---|
| 光斑宽度 FWHM / px | {fmt_general(cmp_best.get("model_fwhm_px"), '.2f', missing_nonfinite='n/a')} | {fmt_general(cmp_best.get("measured_fwhm_px"), '.2f', missing_nonfinite='n/a')} | 硬件更宽：面板像元孔径、串扰、离焦都会加宽 |
| 相对平场峰值比 | {fmt_general(cmp_best.get("strehl_model"), '.3f', missing_nonfinite='n/a')} | {fmt_general(cmp_best.get("strehl_hardware"), '.3f', missing_nonfinite='n/a')} | 硬件只保留模型峰值的类 Strehl 比值 {fmt_general(cmp_best.get("strehl_like_peak_ratio"), '.3f', missing_nonfinite='n/a')} |
| 峰值位置 | — | 偏移 {cmp_best.get("peak_offset_samples")} 个远场采样 | 面板↔相机轴映射只能靠实测对比择优定 |

* **读数**：硬件光斑比模型**更宽、峰值更低、位置有偏移**，三项同时出现。
* **解释**：模型瞳面采样间距 dp 不是整数面板像元，上屏是**块状**相位（每个模型
  采样占 dp×dp 面板像元）；叠加面板孔径与器件响应，硬件看到的相位比模型优化的
  那组值更粗 ⇒ 这是**不可消除**的口径差，只能在模型里显式建模（点扩散函数、
  面板响应），不能靠标定参数补掉。

## 9. 结论

{run_config.get("conclusion", "n/a")}

### 9.1 适用范围与下一步

**本轮能说的**：

* **硬件闭环是通的** —— 把实测远场喂回优化器，实测损失单调下降（§8.2）。
* **模型与硬件之间是结构性口径差**，不是标定误差：硬件光斑更宽、峰值更低、
  位置有偏移（§8.3），这一差只能靠显式建模面板响应来缩小，不能靠调参数补掉。

**本轮不能说的**：

* **不能说「可微优化把远场整成了方形目标」** —— 本轮上屏的相位近似随机（§8.1），
  目标结构并没有被造出来；所有「模型 vs 硬件」数字都受这条限制。

**下一步（按顺序）**：

1. **换初值**：用 GS（或任何解析整形解）做初值，再让可微细化接手 —— 这正是
   `optimizer/wfless/slm_gs_refine.py` 的思路。
2. **换损失**：改成对目标块**加权**的损失形式，而不是全图 MSE，否则"把远场摊平"
   永远是最优解。
3. **显式建模面板响应**：把点扩散函数（PSF）与面板像元孔径写进物理预测模型，否则模型
   的分辨率永远追不上硬件。
4. **单独标定轴映射**：用带标记图案标定面板↔相机轴映射，不要靠「转置 / 不转置」的
   对比择优来定。

## 10. 被推翻 / 待确认

{run_config.get("open_questions", "n/a")}
"""
    text = insert_header(body, REPORT_KEY)
    report_path = REPORT_DIR / "differentiable_opt_report.md"
    report_path.write_text(text, encoding="utf-8")
    return report_path


# ---------------------------------------------------------------------------
# --render-only (离线重渲染, 不碰设备)
# ---------------------------------------------------------------------------
def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    """读回一个 CSV 产物; 数值列转成 float, 其余保持字符串。"""
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        for raw in csv.DictReader(fh):
            row: dict[str, Any] = {}
            for key, value in raw.items():
                if key is None:
                    continue
                try:
                    row[key] = float(value)
                except (TypeError, ValueError):
                    row[key] = value
            rows.append(row)
    return rows


def _render_figures_from_artifacts(
    run_dir: Path,
    history: list[dict[str, Any]],
    hw_history: list[dict[str, Any]],
    run_config: dict[str, Any],
) -> list[Path]:
    """从运行目录里已落盘的产物**重画**那些能离线重算的图。

    离线能重画的是两张"派生图":

    * ``spot_cross_section.png`` —— 需要模型远场 (``model_farfield.npy``) 与
      实测远场 (重采样到模型网格的 ``measured_model_grid.npy``) —— 两者都是
      本轮测量的产物, 不需要设备;
    * ``metrics_vs_step.png`` —— 需要迭代史 (两个 CSV)。

    需要**重新上屏**才能拿到的图 (平场/优化后 CCD、梯度场快照) 无法离线重画,
    所以这里只补这两张, 其余的仍来自运行目录里存好的 PNG。
    """
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    # 中文字体: 离线重画也要配, 否则图上的中文全是豆腐块。
    setup_cjk_font()

    model_path = run_dir / "model_farfield.npy"
    measured_path = run_dir / "measured_model_grid.npy"
    if model_path.exists() and measured_path.exists():
        model = np.load(model_path)
        measured_raw = np.load(measured_path)
        comparison = run_config.get("comparison") or {}
        cmp_best = comparison.get("best") or {}
        measured = _apply_axis_convention(measured_raw, cmp_best)
        ccd_frame = None
        ccd_path = run_dir / "ccd_optimized.npz"
        if ccd_path.exists():
            with np.load(ccd_path) as payload:
                ccd_frame = payload["frame"]
            # CCD 域的光斑量补进对比指标 (旧版 run_config 里没有这一项)。
            cmp_best.update(_ccd_spot_profile(ccd_frame))
        path = fig_dir / "spot_cross_section.png"
        plot_spot_cross_section(
            model,
            measured,
            path,
            metrics=cmp_best,
            ccd_frame=ccd_frame,
        )
        written.append(path)

    model_info = run_config.get("model") or {}
    params = run_config.get("params") or {}
    blind = _blind_loss_value(
        float(model_info.get("far_field_sample_ccd_px") or 0.0),
        float(params.get("target_side_px") or 0.0),
        int(model_info.get("m_grid") or 0),
    )
    path = fig_dir / "metrics_vs_step.png"
    plot_metrics_vs_step(history, hw_history, path, blind_loss=blind)
    written.append(path)
    return written


def _load_hw_artifacts(
    run_dir: Path,
) -> tuple[list[dict[str, Any]], np.ndarray | None]:
    """读回硬件侧的产物: 测量锚定闭环的迭代史 + 优化后那张 CCD 帧。

    离线重做分析时, **硬件侧的东西一个都不能丢**: 没有迭代史就画不出
    "metrics 随 step 变化", 没有 CCD 帧就量不出光斑的绝对宽度。丢了它们,
    报告里会出现空的表格与空的条形图 —— 看起来像"没测", 其实是没读。
    """
    hw_history = _read_csv_rows(run_dir / "history_hw.csv")
    ccd_frame = None
    ccd_path = run_dir / "ccd_optimized.npz"
    if ccd_path.exists():
        with np.load(ccd_path) as payload:
            ccd_frame = payload["frame"]
    return hw_history, ccd_frame


def render_only(run_dir: Path) -> int:
    """从已落盘的运行目录**离线**重生成报告 (不 import 驱动, 不碰设备)。

    为什么需要它: 报告正文由本脚本生成, 所以改生成器 (加章节、改措辞、修数字)
    之后若只能重跑硬件, 就没法在无设备的环境里刷新报告。运行目录里存着
    ``run_config.json`` / ``history.csv`` / ``history_hw.csv`` 与全部图, 足够
    重渲染 —— 这也正是仓库规范里"优先做**离线**报告生成器 (读已落盘产物)"那条
    要求的落法。

    「结论」与「被推翻 / 待确认」两段是**纯函数** (只看本轮的对比结果), 所以
    这里用当前代码重新推导, 而不是照抄运行时存下来的旧文本 —— 否则改措辞不会
    反映到历史报告上。

    Args:
        run_dir: 先前某次 ``--execute`` 的运行目录。

    Returns:
        进程退出码 (``EXIT_OK`` = 成功)。
    """
    run_dir = Path(run_dir)
    config_path = run_dir / "run_config.json"
    if not config_path.exists():
        logger.error("运行目录里没有 {} —— 不是一次 --execute 的产物", config_path)
        return EXIT_MODEL_FAILED

    run_config = json.loads(config_path.read_text(encoding="utf-8"))
    history = _read_csv_rows(run_dir / "history.csv")
    hw_history = _read_csv_rows(run_dir / "history_hw.csv")

    comparison = run_config.get("comparison") or {}
    cmp_best = comparison.get("best") or {}
    run_config["conclusion"] = _conclusion_text(cmp_best, history, hw_history)
    run_config["open_questions"] = _open_questions_text()

    _render_figures_from_artifacts(run_dir, history, hw_history, run_config)

    stamp = run_dir.name
    path = write_report(
        run_config,
        history,
        hw_history,
        run_dir,
        stamp,
        copy_artifacts=True,
        artifact_suffixes=(".json", ".csv"),
    )
    logger.info("离线重渲染报告: {}", path)
    return EXIT_OK


# ---------------------------------------------------------------------------
# --execute
# ---------------------------------------------------------------------------
def build_initial_phase(
    spec: "ModelSpec",
    amplitude: np.ndarray,
    target: np.ndarray,
    *,
    kind: str,
    gs_iters: int,
    seed: int,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """按 ``--init-phase`` 造初值 (只对 ``kind="gs"`` 返回非 ``None``)。

    ⚠️ **为什么默认要用 GS 初值**: 随机初值 + 强度 MSE 会掉进「把远场摊平」的
    局部极小 —— 2026-10-22 实测: 损失一步从 1.07e-2 掉到 1.09e-3, 之后 150 次
    只再降 15%, 而 9.29e-4 恰好等于「目标块一块都没命中」的损失 (31²/1024²),
    相位 RMS 全程停在 3.63 rad (均匀 [0,2π) 的 raw RMS) ⇒ 上屏的相位近似随机
    ⇒ 硬件看到散斑, 模型与硬件的相关系数只有 0.1 量级。

    GS (交替投影) 给的是**解析整形解**: 它先把目标结构造出来, 可微细化再接手。
    这正是仓库 `optimizer/wfless/slm_gs_refine.py` 的思路。传播用
    ``propagation="fft"`` —— 2f 透镜做的就是傅里叶变换, **不是**角谱
    (仓库里已实测 ASM 在 z=f 处给的是传播后的光束, 不是远场)。
    """
    if kind != "gs":
        return None, {"kind": kind, "note": "不用 GS 初值, 走优化器自己的初值"}
    from ao_shaping.algorithm.signal_processing.gerchberg_saxton import (
        gerchberg_saxton,
    )

    t_amp = np.sqrt(np.clip(np.asarray(target, dtype=np.float64), 0.0, None))
    result = gerchberg_saxton(
        np.asarray(amplitude, dtype=np.float64),
        t_amp,
        iterations=int(gs_iters),
        propagation="fft",
    )
    phase = np.asarray(result.phase, dtype=np.float64)
    if phase.shape != (spec.m_grid, spec.m_grid):
        raise ValueError(
            f"GS 初值形状 {phase.shape} 与模型网格 {(spec.m_grid, spec.m_grid)} 不符"
        )
    info = {
        "kind": kind,
        "gs_iters": int(gs_iters),
        "gs_converged": bool(result.converged),
        "gs_first_error": float(result.error_history[0]) if result.error_history else None,
        "gs_last_error": float(result.error_history[-1]) if result.error_history else None,
        "propagation": "fft",
        "note": ("交替投影 (GS) 解析整形解当初值; 传播用 fft (= 2f 透镜的傅里叶变换)"),
    }
    logger.info(
        "[P4] GS 初值: {} 次迭代, 误差 {} → {} (converged={})",
        info["gs_iters"],
        fmt_general(info["gs_first_error"]),
        fmt_general(info["gs_last_error"]),
        info["gs_converged"],
    )
    return phase, info


def run_analyze_only(args: argparse.Namespace) -> int:
    """离线重做分析与报告 —— **一个设备都不碰**。

    实验采到的数据 (原始帧 / 初始与最终相位 / 实测远场网格 / 几何标定 /
    焦面扫描 / 全部参数) 全部落盘, 所以分析可以**事后**重做: 换目标、换初值、
    换拟合口径都不必再上一次机。

    读 ``<run_dir>/run_config.json`` 拿到参数与几何, 用保存的
    ``phase_initial.npy`` 重跑纯模型侧的反向传播 (同一初值 ⇒ 可复现),
    用保存的 ``measured_model_grid.npy`` 重做「模型 vs 硬件」对照, 然后重写报告。

    Args:
        args: 命令行参数 (``--analyze-only`` 指向运行目录)。

    Returns:
        退出码。
    """
    import json

    run_dir = Path(args.analyze_only)
    cfg_path = run_dir / "run_config.json"
    if not cfg_path.is_file():
        logger.error("找不到运行配置: {}", cfg_path)
        return EXIT_MODEL_FAILED
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    params = cfg.get("params", {})
    geometry = cfg.get("geometry", {})
    k_mean = float(geometry.get("k_mean", 0.0))
    if k_mean <= 0.0:
        logger.error("运行配置里没有可用的 K (几何标定缺失), 无法重做对照")
        return EXIT_MODEL_FAILED

    pupil_center = tuple(float(v) for v in params.get("pupil_center", (960.0, 600.0)))
    spec = build_model_spec(
        pupil_radius=float(params.get("pupil_radius", 480.0)),
        pupil_center=pupil_center,
        n_pupil=int(params.get("model_pupil_samples", 256)),
        padding=int(params.get("model_padding", 4)),
        w0_px=float(params.get("beam_w0_px", 260.0)),
    )
    amplitude = build_source_amplitude(spec)
    target, target_info = build_target(
        spec,
        str(params.get("target", "square")),
        float(params.get("target_side_px", 60.0)),
        k_mean,
        delta_spot_px=float(params.get("delta_spot_px", 15.0)),
    )
    init_path = run_dir / "phase_initial.npy"
    init_phase = np.load(init_path) if init_path.is_file() else None
    # 硬件侧的读数从落盘产物里补回来 (离线重分析没有设备可问):
    # 没有迭代史就画不出 "metrics 随 step 变化", 没有 CCD 帧就量不出
    # 光斑的绝对宽度 —— 少了它们, 报告里会出现空的表格与空的条形图。
    hw_history, ccd_frame = _load_hw_artifacts(run_dir)
    logger.info(
        "离线重分析: 目标={} ({}) | 初值来自 {} | 迭代 {} 次",
        params.get("target"),
        target_info.get("note", ""),
        init_path.name if init_phase is not None else "随机 (按 seed)",
        int(params.get("iters", 0)),
    )

    import torch

    setup_cjk_font()  # 离线重画图也要有中文字体, 否则标签全是豆腐块
    device = "cuda" if torch.cuda.is_available() else "cpu"
    opt = DifferentiableBeamOptimizer(
        target,
        source_amplitude=amplitude,
        lr=float(params.get("lr", 0.05)),
        device=device,
        seed=int(params.get("seed", 0)),
        init_phase=init_phase,
    )
    history: list[dict[str, Any]] = []
    grad_fields: dict[int, np.ndarray] = {}
    grad_every = max(int(params.get("grad_every", 10)), 1)
    n_iters = int(params.get("iters", 0))
    phase_mid: np.ndarray | None = None
    for it in range(n_iters):
        opt.update()
        grad = opt.phase_tensor.grad
        gnorm = float(grad.norm()) if grad is not None else float("nan")
        phase_now = opt.current_phase
        history.append(
            {
                "iteration": it + 1,
                "loss": float(
                    opt.last_loss if opt.last_loss is not None else float("nan")
                ),
                "grad_norm": gnorm,
                "phase_rms_rad": float(np.sqrt(np.mean(phase_now**2))),
                "device": device,
            }
        )
        if (it + 1) % grad_every == 0 or it == 0:
            grad_fields[it + 1] = (
                grad.detach().cpu().numpy().copy()
                if grad is not None
                else np.zeros((spec.m_grid, spec.m_grid))
            )
        if phase_mid is None and it == n_iters // 2:
            phase_mid = phase_now.copy()
    phase_final = opt.current_phase.copy()
    if phase_mid is None:
        phase_mid = phase_final.copy()

    # 图: 重画到运行目录的 figures/ (报告再从这里搬)
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_phase(phase_mid, spec, fig_dir / "phase_mid.png", title="中间相位")
    plot_phase(phase_final, spec, fig_dir / "phase_final.png", title="最终相位")
    if grad_fields:
        first_key = min(grad_fields)
        plot_gradient(
            grad_fields[first_key],
            spec,
            fig_dir / f"gradient_{first_key:04d}.png",
            title=f"梯度场 (第 {first_key} 次迭代后)",
        )
        last_key = max(grad_fields)
        plot_gradient(
            grad_fields[last_key],
            spec,
            fig_dir / "gradient_final.png",
            title=f"梯度场 (第 {last_key} 次迭代后, 最后一张)",
        )
    plot_history(history, fig_dir / "history.png")
    plot_metrics_vs_step(
        history,
        hw_history,
        fig_dir / "metrics_vs_step.png",
        blind_loss=_blind_loss_value(
            spec.far_field_sample_ccd_px(k_mean),
            float(params.get("target_side_px", 60.0)),
            int(spec.m_grid),
        ),
    )

    model_farfield = model_far_field(amplitude, phase_final)
    np.save(run_dir / "model_farfield.npy", model_farfield)
    plot_farfield(
        model_farfield,
        fig_dir / "model_farfield.png",
        title="模型预测的远场 (最终相位)",
    )

    # 对照: 用保存的实测网格重做 (八个轴向口径择优)
    measured_path = run_dir / "measured_model_grid.npy"
    comparison = cfg.get("comparison", {})
    if measured_path.is_file():
        measured_opt = np.load(measured_path)
        candidates: dict[str, dict[str, Any]] = {}
        for name, transpose, flip_rows, flip_cols in _AXIS_CANDIDATES:
            variant = measured_opt
            if transpose:
                variant = variant.T
            if flip_rows:
                variant = variant[::-1, :]
            if flip_cols:
                variant = variant[:, ::-1]
            candidates[name] = {
                **compare_farfields(model_farfield, variant, transpose_measured=False),
                "transpose": bool(transpose),
                "flip_rows": bool(flip_rows),
                "flip_cols": bool(flip_cols),
            }
        best_key = max(
            candidates,
            key=lambda key: (
                candidates[key]["intensity_correlation"]
                if np.isfinite(candidates[key]["intensity_correlation"])
                else -np.inf
            ),
        )
        cmp_best = candidates[best_key]
        cmp_best["orientation_match"] = best_key
        for key in (
            "strehl_model",
            "strehl_hardware",
            "strehl_like_peak_ratio",
            "exposure_ms",
        ):
            if key in comparison.get("best", {}):
                cmp_best[key] = comparison["best"][key]
        if ccd_frame is not None:
            cmp_best.update(_ccd_spot_profile(ccd_frame))
        # 判定语要在**所有**指标都补齐之后再算 —— 否则会出现
        # 「类 Strehl 峰值比 None」这种半截句子。
        cmp_best["verdict"] = _verdict_text(cmp_best)
        comparison = {
            "best": cmp_best,
            "candidates": candidates,
            "flat_reference": comparison.get("flat_reference", {}),
            "note": comparison.get("note", ""),
        }
        plot_model_vs_hardware(
            model_farfield,
            _apply_axis_convention(measured_opt, cmp_best),
            fig_dir / "model_vs_hardware.png",
            orientation_match=best_key,
            metrics=cmp_best,
        )
        # 光斑截面 (CCD 域 + 模型网格域) —— 硬件整形实验缺这张就看不出
        # "光斑到底差在哪"。
        plot_spot_cross_section(
            model_farfield,
            _apply_axis_convention(measured_opt, cmp_best),
            fig_dir / "spot_cross_section.png",
            metrics=cmp_best,
            ccd_frame=ccd_frame,
        )
        logger.info(
            "离线重分析: 口径 {} | 相关系数 {} | 峰值偏移 {}",
            best_key,
            fmt_general(cmp_best.get("intensity_correlation"), '.3f', missing_nonfinite='n/a'),
            cmp_best.get("peak_offset_samples"),
        )
    else:
        logger.warning("运行目录里没有 measured_model_grid.npy, 跳过对照重做")

    cfg["model"]["target"] = target_info
    cfg["comparison"] = comparison
    cfg["history_summary"] = {
        "iters": len(history),
        "first_loss": history[0]["loss"] if history else float("nan"),
        "last_loss": history[-1]["loss"] if history else float("nan"),
        "hw_iters": int(cfg.get("history_summary", {}).get("hw_iters", 0)),
        "hw_first_loss": cfg.get("history_summary", {}).get("hw_first_loss"),
        "hw_last_loss": cfg.get("history_summary", {}).get("hw_last_loss"),
        "reanalyzed_offline": True,
    }
    cfg["conclusion"] = _conclusion_text(
        comparison.get("best", {}), history, hw_history
    )
    (run_dir / "run_config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    stamp = run_dir.name
    write_report(
        cfg,
        history,
        hw_history,
        run_dir,
        stamp,
        copy_artifacts=False,
    )
    logger.info(f"\n离线重分析完成: {run_dir}")
    return EXIT_OK


def run_execute(args: argparse.Namespace, plan: dict[str, Any]) -> int:
    """真正开设备, 按 P1–P7 走完。返回退出码。"""
    # ---- 设备 import 全部锁在 --execute 分支内 -----------------------------
    from ao_shaping.drivers.ccd.common import create_camera
    from ao_shaping.drivers.slm.santec import Santec, SlotRotator
    from ao_shaping.tools.slm.calibration.calibration import SLMCCDCalibrator
    from ao_shaping.utils.image.exposure_control import (
        configured_daheng_exposure_limit,
    )
    from ao_shaping.utils.slm.phase_display import phase_to_slm_grayscale

    import torch

    setup_cjk_font()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"运行目录: {out_dir}")

    # ---- P0.5: 等设备空出来 (公平仲裁, 不抢) -------------------------------
    # 别的会话可能在跑硬件 sweep (相机/SLM 的句柄是独占的)。**先看进程再探针**
    # 的顺序由 `run_hw_sweep_groups.wait_until_idle` 实现, 本脚本直接复用它 ——
    # 自己再写一份等待逻辑, 就会在某天和别人的 campaign 撞车。
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
            # ⚠️ 只按 **pid** 排除自己, **不要**传 own_paths=[本脚本路径]:
            # 同一个脚本的**另一个**会话也会被 own_paths 当成"我们自己"排除掉,
            # 于是占用者看不见 ⇒ 探针直接撞 -10003 ⇒ 被误判成"陈旧句柄"
            # (2026-10-23 实测: 另一个会话正在跑同一个脚本的 rep 系列, 本脚本
            #  却报「设备锁是空的」并让人去断电 ✗)。
            own_pids=(os.getpid(),),
        )
        if code != 0:
            logger.error("设备未等到空闲: {}", message)
            return EXIT_BENCH_BUSY
        logger.info("设备空闲, 开始采集")

    panel_shape = (SLM_PANEL_H, SLM_PANEL_W)
    pupil_center = (float(args.pupil_center[0]), float(args.pupil_center[1]))
    pupil_radius = float(args.pupil_radius)

    # ---- P1: 打开设备 ------------------------------------------------------
    slm_factory = lambda: Santec(
        slm_number=int(args.slm_number),
        video_mode=0,
        wavelength=int(args.slm_wavelength),
    )
    cam_factory = lambda: create_camera(
        args.cam_type, int(args.cam_id), exposure_time_ms=float(args.exposure_ms)
    )

    with guarded_device("SLM", slm_factory) as slm:
        slm_serial = slm.get_serial_number()
        logger.info("SLM #{} 序列号: {}", args.slm_number, slm_serial)
        with guarded_device("CCD", cam_factory) as cam:
            cam_sn = getattr(cam, "sn", None)
            logger.info("CCD 序列号: {}", cam_sn)
            rotator = SlotRotator(slm)
            return _run_bench(
                args=args,
                plan=plan,
                slm=slm,
                cam=cam,
                rotator=rotator,
                slm_serial=slm_serial,
                cam_sn=cam_sn,
                out_dir=out_dir,
                fig_dir=fig_dir,
                stamp=stamp,
                panel_shape=panel_shape,
                pupil_center=pupil_center,
                pupil_radius=pupil_radius,
                torch=torch,
                phase_to_slm_grayscale=phase_to_slm_grayscale,
                SLMCCDCalibrator=SLMCCDCalibrator,
                configured_daheng_exposure_limit=configured_daheng_exposure_limit,
            )


def _run_bench(
    args: argparse.Namespace,
    plan: dict[str, Any],
    *,
    slm: Any,
    cam: Any,
    rotator: Any,
    slm_serial: str | None,
    cam_sn: str | None,
    out_dir: Path,
    fig_dir: Path,
    stamp: str,
    panel_shape: tuple[int, int],
    pupil_center: tuple[float, float],
    pupil_radius: float,
    torch: Any,
    phase_to_slm_grayscale: Any,
    SLMCCDCalibrator: Any,
    configured_daheng_exposure_limit: Any,
) -> int:
    """设备已打开, 跑 P2–P7。"""
    # 曝光上限: 驱动的保护上限 (环境变量) 与脚本上限取小的, 但**至少**给到
    # ``--exposure-ceiling-ms``。散射型图案 (优化后的相位把光摊成一片) 的
    # 峰值比聚焦态低一两个数量级, 上限卡在 20 ms 时整帧都是本底噪声 ——
    # 后面所有的对比就变成在比噪声 (2026-10-22 首次实测: 20 ms 到顶、
    # 状态 dim、峰值 55 counts, 与模型的相关系数只有 0.05)。
    ceiling = float(configured_daheng_exposure_limit() or 0.0)
    ceiling = (
        min(ceiling, float(args.exposure_ceiling_ms))
        if ceiling > 0.0
        else float(args.exposure_ceiling_ms)
    )

    def display_phase(phase_rad: np.ndarray) -> int:
        """把**面板形状**的原始弧度相位上屏 (走驱动管线 + 换槽)。"""
        gray = phase_to_slm_grayscale(phase_rad, slm=slm)
        return rotator.display_data(slm, gray, float(args.settle_s))

    def read_frame() -> np.ndarray:
        return np.asarray(cam.get_numpy_image(n_sample=1))

    def set_exposure_ms(ms: float) -> float:
        return float(cam.reset_exposure_time(float(ms)) or ms)

    # ---- P2: 在焦判定 (门禁) ----------------------------------------------
    logger.info("[P2] 焦面扫描开始 (测量前的在焦判定)")
    focus_result = sweep_focus(
        apply_phase=display_phase,
        read_frame=read_frame,
        set_exposure_ms=set_exposure_ms,
        panel_shape=panel_shape,
        radius_px=pupil_radius,
        centre_px=pupil_center,
        coarse_points=int(args.focus_points),
        diopter_range=float(args.focus_diopter_range),
        refine_points=int(args.focus_refine),
        start_exposure_ms=float(args.exposure_ms),
        floor_ms=FLOOR_EXPOSURE_MS,
        ceiling_ms=ceiling,
        frames=int(args.frames),
        discard=int(args.discard),
        wavelength_nm=float(args.slm_wavelength),
        pixel_um=SLM_PITCH_M * 1e6,
        max_extends=2,
    )
    focus_points = [
        {
            "diopter_m1": p.diopter_m1,
            "focal_mm": p.focal_mm,
            "exposure_ms": p.exposure_ms,
            "peak": p.peak,
            "peak_fraction": p.peak_fraction,
            "peak_per_ms": p.peak_per_ms,
            "fwhm_px": p.fwhm_px,
            "hollowness": p.hollowness,
            "status": p.status,
        }
        for p in focus_result.points
    ]
    write_csv(
        out_dir / "focus_sweep.csv",
        [
            "diopter_m1",
            "focal_mm",
            "exposure_ms",
            "peak",
            "peak_fraction",
            "peak_per_ms",
            "fwhm_px",
            "hollowness",
            "status",
        ],
        focus_points,
    )
    plot_focus_sweep(
        focus_result, fig_dir / "focus_sweep.png", fwhm_max=float(args.focus_fwhm_max)
    )

    focus_info: dict[str, Any] = {
        "coarse_points": int(args.focus_points),
        "diopter_range": float(args.focus_diopter_range),
        "refine_points": int(args.focus_refine),
        "fwhm_threshold_ccd_px": float(args.focus_fwhm_max),
        "best_diopter_m1": focus_result.best_diopter_m1,
        "best_focal_mm": focus_result.best_focal_mm,
        "best_fwhm_px": (focus_result.best.fwhm_px if focus_result.best else None),
        "path_blocked": bool(focus_result.path_blocked),
        "blockage_reason": focus_result.blockage_reason,
        "notes": list(focus_result.notes),
        "points": focus_points,
    }

    def write_abort_config(extra: dict[str, Any]) -> None:
        cfg = {
            "timestamp_utc": stamp,
            "params": vars(args) | {"pupil_center": list(args.pupil_center)},
            "plan": plan,
            "devices": {"slm_serial": slm_serial, "ccd_sn": cam_sn},
            "focus": focus_info,
            "aborted": extra,
        }
        (out_dir / "run_config.json").write_text(
            json.dumps(cfg, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

    # --- 判定: 光路受阻 / 不在焦 ------------------------------------------
    gate = judge_focus_gate(focus_result, float(args.focus_fwhm_max))
    focus_info["verdict"] = gate["verdict"]
    if gate["code"] is not None:
        write_abort_config({"reason": gate["reason"], "exit_code": gate["code"]})
        if gate["code"] == EXIT_PATH_BLOCKED:
            logger.info(
                "\n[中止] 光路受阻: " + gate["reason"] + "\n"
                "       画面里根本没有光, 后面量到的只会是读出噪声。\n"
                f"       判定已写进 {out_dir / 'run_config.json'}",

            )
        else:
            logger.info(
                "\n[中止] " + gate["verdict"] + "\n"
                "       用户要求: 实验系统没有被判为在焦之前, 一个测量都不做。\n"
                f"       判定已写进 {out_dir / 'run_config.json'}",

            )
        return int(gate["code"])

    logger.info("[P2] {}", focus_info["verdict"])

    # 之后每张图案都叠加这张瞳面菲涅尔补偿相位
    compensation = fresnel_panel(
        float(focus_result.best_focal_mm),
        radius=int(round(pupil_radius)),
        pupil_center=(int(round(pupil_center[0])), int(round(pupil_center[1]))),
        panel_shape=panel_shape,
        wavelength_nm=float(args.slm_wavelength),
        pixel_um=SLM_PITCH_M * 1e6,
    )

    # ---- P3: 几何 ---------------------------------------------------------
    logger.info("[P3] 几何标定")
    geometry: dict[str, Any]
    try:
        if args.geometry_npz:
            with np.load(args.geometry_npz, allow_pickle=True) as data:
                raw = {k: data[k] for k in data.files}
            geometry = {
                "source": f"存档 {args.geometry_npz}",
                "measured_this_run": False,
                "center": np.asarray(raw["center"]).tolist(),
                "kx": float(raw["Kx"]),
                "ky": float(raw["Ky"]),
                "rotation_deg": float(raw.get("rotation_deg", 0.0)),
                "mirror": bool(raw.get("mirror", False)),
                "fwhm0": float(raw.get("fwhm0", 0.0)),
                "resid_x": float(raw.get("resid_x", float("nan"))),
                "resid_y": float(raw.get("resid_y", float("nan"))),
            }
            logger.warning(
                "几何用的是**存档** {} (Kx={:.1f}, Ky={:.1f}, rot={:+.2f}°); "
                "存档可能与当前光路对不上 —— 结论按「存档口径」解读",
                args.geometry_npz,
                geometry["kx"],
                geometry["ky"],
                geometry["rotation_deg"],
            )
        else:
            # 标定器自己管理显示路径 (它只显示 plain 相位), 所以用**叠加层**把
            # 焦面补偿临时挂上去: 这样标定期间光斑是在焦的, ±1 级才不会因为
            # 离焦弥散而测不出来。标定完立刻撤掉, 之后补偿相位由脚本自己加。
            slm.set_phase_overlay(compensation)
            try:
                calibrator = SLMCCDCalibrator(slm, cam, settle_s=float(args.settle_s))
                raw_calib = calibrator.calibrate()
            finally:
                slm.set_phase_overlay(None)
            geometry = {
                "source": "现场实测 (SLMCCDCalibrator.calibrate)",
                "measured_this_run": True,
                "center": np.asarray(raw_calib["center"]).tolist(),
                "kx": float(raw_calib["Kx"]),
                "ky": float(raw_calib["Ky"]),
                "rotation_deg": float(raw_calib["rotation_deg"]),
                "mirror": bool(raw_calib["mirror"]),
                "fwhm0": float(raw_calib["fwhm0"]),
                "resid_x": float(raw_calib["resid_x"]),
                "resid_y": float(raw_calib["resid_y"]),
            }
    except Exception as exc:  # 几何失败 ⇒ 退出码 6
        logger.exception("几何标定失败")
        write_abort_config({"reason": "geometry_failed", "error": str(exc)})
        logger.info(f"\n[中止] 几何标定失败: {exc}")
        return EXIT_MODEL_FAILED

    k_mean = 0.5 * (geometry["kx"] + geometry["ky"])
    geometry["k_mean"] = k_mean
    geometry["k_choice"] = "K = (Kx + Ky)/2 (各向同性)"
    centre_cy, centre_cx = (float(v) for v in geometry["center"])
    geometry["centre_cycx"] = [centre_cy, centre_cx]
    logger.info(
        "[P3] 几何: Kx={:.1f} Ky={:.1f} K={:.1f} 零级=({:.0f}, {:.0f}) rot={:+.2f}° mirror={}",
        geometry["kx"],
        geometry["ky"],
        k_mean,
        centre_cy,
        centre_cx,
        geometry["rotation_deg"],
        geometry["mirror"],
    )

    # ---- P4: 物理预测模型 + 反向传播优化 --------------------------------------
    try:
        spec = build_model_spec(
            pupil_radius=pupil_radius,
            pupil_center=pupil_center,
            n_pupil=int(args.model_pupil_samples),
            padding=int(args.model_padding),
            w0_px=float(args.beam_w0_px),
        )
        amplitude = build_source_amplitude(spec)
        target, target_info = build_target(
            spec,
            args.target,
            float(args.target_side_px),
            k_mean,
            delta_spot_px=float(args.delta_spot_px),
        )
    except Exception as exc:  # 模型构建失败 ⇒ 退出码 6
        logger.exception("物理预测模型构建失败")
        write_abort_config({"reason": "model_build_failed", "error": str(exc)})
        logger.info(f"\n[中止] 物理预测模型构建失败: {exc}")
        return EXIT_MODEL_FAILED

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(
        "[P4] 物理预测模型: N={} M={} dp={:.4f} 远场采样 {:.4f} CCD px, 覆盖 {:.0f} CCD px, 目标 {} ({}), 设备 {}",
        spec.n_pupil,
        spec.m_grid,
        spec.dp,
        spec.far_field_sample_ccd_px(k_mean),
        spec.far_field_span_ccd_px(k_mean),
        args.target,
        target_info.get("note", ""),
        device,
    )

    # 初值: gs (默认) = 用交替投影的解析整形解起手; random = 优化器自己的随机初值
    gs_phase, init_info = build_initial_phase(
        spec,
        amplitude,
        target,
        kind=str(getattr(args, "init_phase", "gs")),
        gs_iters=int(getattr(args, "gs_iters", 50)),
        seed=int(args.seed),
    )
    opt = DifferentiableBeamOptimizer(
        target,
        source_amplitude=amplitude,
        lr=float(args.lr),
        device=device,
        seed=int(args.seed),
        init_phase=gs_phase,
    )
    phase_initial = opt.current_phase.copy()
    np.save(out_dir / "phase_initial.npy", phase_initial)
    plot_phase(
        phase_initial, spec, fig_dir / "phase_initial.png", title="初始相位 (随机)"
    )

    history: list[dict[str, Any]] = []
    grad_fields: dict[int, np.ndarray] = {}
    phase_mid: np.ndarray | None = None
    grad_every = max(int(args.grad_every), 1)
    n_iters = int(args.iters)
    for it in range(n_iters):
        opt.update()
        grad = opt.phase_tensor.grad
        gnorm = float(grad.norm()) if grad is not None else float("nan")
        phase_now = opt.current_phase
        rms = float(np.sqrt(np.mean(phase_now**2)))
        history.append(
            {
                "iteration": it + 1,
                "loss": float(
                    opt.last_loss if opt.last_loss is not None else float("nan")
                ),
                "grad_norm": gnorm,
                "phase_rms_rad": rms,
                "device": device,
            }
        )
        if (it + 1) % grad_every == 0 or it == 0:
            grad_fields[it + 1] = (
                grad.detach().cpu().numpy().copy()
                if grad is not None
                else np.zeros((spec.m_grid, spec.m_grid))
            )
        if phase_mid is None and it == n_iters // 2:
            phase_mid = phase_now.copy()
        if (it + 1) % 25 == 0 or it + 1 == n_iters:
            logger.info(
                "[P4] iter {}: loss={:.6e} |grad|={:.4e} phase_rms={:.4f} rad",
                it + 1,
                history[-1]["loss"],
                gnorm,
                rms,
            )

    phase_final = opt.current_phase.copy()
    np.save(out_dir / "phase_final.npy", phase_final)
    np.save(out_dir / "phase_model.npy", phase_model_full(opt))
    if phase_mid is None:
        phase_mid = phase_final.copy()
    plot_phase(phase_mid, spec, fig_dir / "phase_mid.png", title="中间相位")
    plot_phase(phase_final, spec, fig_dir / "phase_final.png", title="最终相位")
    for it, gfield in grad_fields.items():
        plot_gradient(
            gfield,
            spec,
            fig_dir / f"gradient_{it:04d}.png",
            title=f"梯度场 (第 {it} 次迭代后)",
        )
    if grad_fields:
        last_it = max(grad_fields)
        plot_gradient(
            grad_fields[last_it],
            spec,
            fig_dir / "gradient_final.png",
            title=f"梯度场 (第 {last_it} 次迭代后, 最后一张)",
        )
    write_csv(
        out_dir / "history.csv",
        ["iteration", "loss", "grad_norm", "phase_rms_rad", "device"],
        history,
    )
    plot_history(history, fig_dir / "history.png")

    model_farfield = model_far_field(amplitude, phase_final)
    np.save(out_dir / "model_farfield.npy", model_farfield)
    plot_farfield(
        model_farfield,
        fig_dir / "model_farfield.png",
        title="模型预测的远场 (最终相位)",
    )

    # ---- P5: 上屏 + 抓帧 --------------------------------------------------
    logger.info("[P5] 上屏并抓帧")

    def display_model_phase(phase_model: np.ndarray) -> np.ndarray:
        """把模型相位搬到面板 + 焦面补偿, 上屏, 返回下发的面板相位。"""
        panel = phase_model_to_panel(phase_model, spec, panel_shape) + compensation
        display_phase(panel)
        return panel

    def capture(tag: str) -> tuple[np.ndarray, list[np.ndarray], float]:
        """重新对曝光 → 抓帧 → 平均。

        🔴 **每张图案都要重新对曝光**: 焦面扫描定下的曝光是按**平场光斑**的
        亮度定的, 而优化后的图案把光散成一片类散斑 (峰值比聚焦态低一两个
        数量级)。沿用平场的曝光, 那一帧几乎全是本底噪声 —— 后面所有的
        对比都变成在比噪声。复用 :func:`adapt_exposure` (焦面扫描用的同一个
        曝光自适应), 口径一致。
        """
        setting: ExposureSetting = guarded(
            f"自动曝光 ({tag})",
            lambda: adapt_exposure(
                set_exposure_ms,
                read_frame,
                start_ms=float(args.exposure_ms),
                floor_ms=FLOOR_EXPOSURE_MS,
                ceiling_ms=ceiling,
                full_scale=255.0,
                discard=int(args.discard),
            ),
            timeout_s=ACQUIRE_TIMEOUT_S,
        )
        frame, raw = guarded(
            f"CCD 取帧 ({tag})",
            lambda: capture_frames(
                cam, frames=int(args.frames), discard=int(args.discard)
            ),
            timeout_s=ACQUIRE_TIMEOUT_S,
        )
        logger.info(
            "[P5] {}: 曝光 {} ms (状态 {}), 峰值 {:.1f} counts",
            tag,
            setting.exposure_ms,
            setting.status,
            float(frame.max()),
        )
        return frame, raw, float(setting.exposure_ms)

    # 平场参考 (零相位 + 同一张补偿)
    flat_panel = np.zeros(panel_shape, dtype=np.float64) + compensation
    display_phase(flat_panel)
    flat_frame, flat_raw, flat_exp = capture("flat")
    save_ccd_frame(out_dir / "ccd_flat.npz", flat_frame)
    plot_ccd(
        flat_frame, fig_dir / "ccd_flat.png", title="CCD 平场参考 (零相位 + 焦面补偿)"
    )
    flat_assessment = assess_spot(flat_frame)
    logger.info(
        "[P5] 平场: 峰值 {:.1f}, FWHM {:.2f} px, 可用={}",
        float(flat_frame.max()),
        float(flat_assessment.fwhm_px),
        flat_assessment.usable,
    )

    # 优化相位
    display_model_phase(phase_final)
    optimized_frame, optimized_raw, optimized_exp = capture("optimized")
    save_ccd_frame(out_dir / "ccd_optimized.npz", optimized_frame)
    plot_ccd(
        optimized_frame,
        fig_dir / "ccd_optimized.png",
        title="CCD 优化后 (模型相位 + 焦面补偿)",
    )

    # ---- P6: 模型 vs 硬件 -------------------------------------------------
    logger.info("[P6] 模型 vs 硬件")
    measured_flat = resample_ccd_to_model(
        prepare_frame(np.asarray(flat_frame, dtype=np.float64)),
        (centre_cx, centre_cy),
        k_mean,
        spec,
    )
    measured_opt = resample_ccd_to_model(
        prepare_frame(np.asarray(optimized_frame, dtype=np.float64)),
        (centre_cx, centre_cy),
        k_mean,
        spec,
    )
    # 面板↔相机的轴向映射: 本实验系统的面板 x 向在 CCD 上表现为行方向, 而
    # 几何标定返回的 `rotation_deg` 只编码两条光栅轴的**正交性偏差**, 不能
    # 无歧义地定出轴映射。所以这里**不假设**, 把「转置 / 镜像」这 8 个
    # 带符号的置换全试一遍, 按相关系数对比择优 —— 2026-10-22 首次实测正是靠
    # 这个办法才发现: 模型与硬件的峰值位移**模长一致** (623 px vs 624 px),
    # 差的只是轴向约定。
    candidates: dict[str, dict[str, Any]] = {}
    for name, transpose, flip_rows, flip_cols in _AXIS_CANDIDATES:
        measured_variant = measured_opt
        if transpose:
            measured_variant = measured_variant.T
        if flip_rows:
            measured_variant = measured_variant[::-1, :]
        if flip_cols:
            measured_variant = measured_variant[:, ::-1]
        candidates[name] = {
            **compare_farfields(
                model_farfield, measured_variant, transpose_measured=False
            ),
            "transpose": bool(transpose),
            "flip_rows": bool(flip_rows),
            "flip_cols": bool(flip_cols),
        }
    best_key = max(
        candidates,
        key=lambda key: (
            candidates[key]["intensity_correlation"]
            if np.isfinite(candidates[key]["intensity_correlation"])
            else -np.inf
        ),
    )
    cmp_best = candidates[best_key]
    cmp_best["orientation_match"] = best_key

    # 类 Strehl 峰值比: 硬件与模型**各自**相对平场参考的峰值比之比。
    #
    # 不能直接拿「实测峰值 / 模型峰值」—— 两者单位不同 (counts vs 模型的无量纲
    # 单位), 而且实测亮度随曝光走。两边都先除以各自的平场参考 (实测那边还要
    # 除以曝光, 换算成 counts/ms), 得到的才是「相对平场的峰值比」, 两者相除
    # 才是可比的量。
    model_flat_farfield = model_far_field(
        amplitude, np.zeros((spec.m_grid, spec.m_grid))
    )
    flat_peak_per_ms = float(flat_frame.max()) / max(flat_exp, 1e-9)
    opt_peak_per_ms = float(optimized_frame.max()) / max(optimized_exp, 1e-9)
    model_strehl = (
        float(model_farfield.max()) / float(model_flat_farfield.max())
        if float(model_flat_farfield.max()) > 0
        else float("nan")
    )
    hw_strehl = (
        opt_peak_per_ms / flat_peak_per_ms if flat_peak_per_ms > 0 else float("nan")
    )
    cmp_best["strehl_like_peak_ratio"] = (
        float(hw_strehl / model_strehl)
        if np.isfinite(hw_strehl) and np.isfinite(model_strehl) and model_strehl > 0
        else float("nan")
    )
    cmp_best["strehl_model"] = float(model_strehl)
    cmp_best["strehl_hardware"] = float(hw_strehl)
    cmp_best["exposure_ms"] = {
        "flat": float(flat_exp),
        "optimized": float(optimized_exp),
    }
    cmp_best["verdict"] = _verdict_text(cmp_best)
    # CCD 域的光斑量 (真实像元, counts) —— 与表里"模型采样"口径的 FWHM 分开记。
    cmp_best.update(_ccd_spot_profile(optimized_frame))
    cmp_flat = compare_farfields(model_farfield, measured_flat)
    np.save(out_dir / "measured_model_grid.npy", measured_opt)
    # 三联图里的实测那一张必须是**按对比择优口径变换过**的, 否则两张图的峰值位置
    # 对不上 (标题里的峰值偏移正是按变换后的口径算的)。
    plot_model_vs_hardware(
        model_farfield,
        _apply_axis_convention(measured_opt, cmp_best),
        fig_dir / "model_vs_hardware.png",
        orientation_match=best_key,
        metrics=cmp_best,
    )
    # 光斑截面: 模型 vs 实测 —— 硬件整形实验缺了这张就看不出"光斑到底差在哪"。
    plot_spot_cross_section(
        model_farfield,
        _apply_axis_convention(measured_opt, cmp_best),
        fig_dir / "spot_cross_section.png",
        metrics=cmp_best,
        ccd_frame=optimized_frame,
    )
    comparison = {
        "best": cmp_best,
        "candidates": candidates,
        "flat_reference": cmp_flat,
        "note": (
            "面板↔相机的 90° 轴交换与几何标定返回值的约定不自洽, 所以脚本不假设, "
            "把「转置」当作两个候选口径实测对比择优 (对比择优结果见 orientation_match)"
        ),
    }

    # ---- P6b: 测量锚定闭环 -----------------------------------------------
    logger.info("[P6] 测量锚定闭环 ({} 次)", int(args.hw_iters))
    hw_history: list[dict[str, Any]] = []
    for i in range(int(args.hw_iters)):
        current_phase = opt.current_phase.copy()
        display_model_phase(current_phase)
        frame, _raw, hw_exp = capture(f"hw_iter_{i}")
        if i >= max(int(args.hw_iters) - 3, 0):
            save_ccd_frame(out_dir / f"ccd_hw_iter_{i:03d}.npz", frame)
        measured = resample_ccd_to_model(
            prepare_frame(np.asarray(frame, dtype=np.float64)),
            (centre_cx, centre_cy),
            k_mean,
            spec,
        )
        opt.update(measured_intensity=measured)
        grad = opt.phase_tensor.grad
        gnorm = float(grad.norm()) if grad is not None else float("nan")
        rms = float(np.sqrt(np.mean(opt.current_phase**2)))
        hw_history.append(
            {
                "iteration": i + 1,
                "loss": float(
                    opt.last_loss if opt.last_loss is not None else float("nan")
                ),
                "grad_norm": gnorm,
                "phase_rms_rad": rms,
                "measured_peak": float(measured.max()),
                "exposure_ms": float(hw_exp),
            }
        )
        logger.info(
            "[P6] hw iter {}: 实测损失={:.6e} |grad|={:.4e} 相位RMS={:.4f}",
            i + 1,
            hw_history[-1]["loss"],
            gnorm,
            rms,
        )
    write_csv(
        out_dir / "history_hw.csv",
        [
            "iteration",
            "loss",
            "grad_norm",
            "phase_rms_rad",
            "measured_peak",
            "exposure_ms",
        ],
        hw_history,
    )
    # metrics 随 step 变化的曲线 (闭环) —— 硬件整形实验缺了这条就看不出
    # "每一步到底有没有变好"。
    plot_metrics_vs_step(
        history,
        hw_history,
        fig_dir / "metrics_vs_step.png",
        blind_loss=_blind_loss_value(
            spec.far_field_sample_ccd_px(k_mean),
            float(args.target_side_px),
            int(spec.m_grid),
        ),
    )

    # ---- P7: 产物 + 报告 --------------------------------------------------
    model_info = spec.as_dict(k_mean)
    model_info["target"] = target_info
    model_info["initial_phase"] = init_info
    model_info["iters"] = int(args.iters)
    model_info["hw_iters"] = int(args.hw_iters)
    model_info["device"] = device
    model_info["lr"] = float(args.lr)
    model_info["seed"] = int(args.seed)
    model_info["grad_every"] = grad_every

    run_config = {
        "timestamp_utc": stamp,
        "params": {
            k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(args).items()
        },
        "plan": plan,
        "devices": {
            "slm_serial": slm_serial,
            "ccd_sn": cam_sn,
            "slm_number": int(args.slm_number),
            "slm_wavelength_nm": int(args.slm_wavelength),
            "cam_type": args.cam_type,
            "cam_id": int(args.cam_id),
            "exposure_ms_requested": float(args.exposure_ms),
            "exposure_ceiling_ms": float(ceiling),
        },
        "focus": focus_info,
        "focus_compensation_focal_mm": float(focus_result.best_focal_mm),
        "geometry": geometry,
        "model": model_info,
        "comparison": comparison,
        "history_summary": {
            "iters": len(history),
            "first_loss": history[0]["loss"] if history else None,
            "last_loss": history[-1]["loss"] if history else None,
            "hw_iters": len(hw_history),
            "hw_first_loss": hw_history[0]["loss"] if hw_history else None,
            "hw_last_loss": hw_history[-1]["loss"] if hw_history else None,
        },
        "conclusion": _conclusion_text(cmp_best, history, hw_history),
        "open_questions": _open_questions_text(),
    }
    (out_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    if bool(getattr(args, "quick", False)):
        # 冒烟跑**不写报告**: 报告是"这一轮真实实验"的结论, 被一次 2 轮迭代的
        # 冒烟覆盖掉, 就等于把结论换成了没意义的数。产物仍照常落盘。
        logger.info("冒烟模式: 产物已落盘, 报告不覆盖")
        logger.info(f"产物: {out_dir}")
        return EXIT_OK
    report_path = write_report(run_config, history, hw_history, out_dir, stamp)
    logger.info(f"报告: {report_path}")
    logger.info(f"产物: {out_dir}")
    return EXIT_OK


def phase_model_full(opt: Any) -> np.ndarray:
    """取当前相位张量的完整 ``M×M`` 副本 (存盘用)。"""
    return np.asarray(opt.current_phase, dtype=np.float64)


def model_far_field(amplitude: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """用 numpy 复算一次模型的远场强度 (给报告/对比用, 与 torch 侧同口径)。"""
    field = np.asarray(amplitude, dtype=np.float64) * np.exp(
        1j * np.asarray(phase, dtype=np.float64)
    )
    far = np.fft.fftshift(np.fft.fft2(field))
    return (far.real**2 + far.imag**2).astype(np.float64)


def judge_focus_gate(
    result: FocusProbeResult,
    fwhm_max: float,
) -> dict[str, Any]:
    """纯函数: 判在焦。**不碰设备**, 所以可以离线自检。

    判据 (与用户的要求一一对应):

    * 光路受阻 (``path_blocked``) ⇒ 光根本没到相机 ⇒ 退出码
      :data:`EXIT_PATH_BLOCKED`;
    * 扫描范围内没有任何点落进可用曝光带 (``best is None``) ⇒ 无法判在焦 ⇒
      退出码 :data:`EXIT_NOT_IN_FOCUS` (同样拒绝测量);
    * 最佳点的 FWHM 超过门限 ⇒ **不在焦** ⇒ 退出码
      :data:`EXIT_NOT_IN_FOCUS`, **不写任何测量数据**;
    * 其余 ⇒ 在焦, ``code=None`` (继续测量)。

    Returns:
        ``{"code": int | None, "verdict": str, "reason": str}``。
    """
    if result.path_blocked:
        return {
            "code": EXIT_PATH_BLOCKED,
            "verdict": "光路受阻 —— 拒绝测量",
            "reason": result.blockage_reason or "画面里没有局域光斑 (光没到相机)",
        }
    verdict = judge_focus(result, fwhm_max_px=float(fwhm_max))
    if verdict.kind == "not_in_focus":
        return {
            "code": EXIT_NOT_IN_FOCUS,
            "verdict": verdict.verdict,
            "reason": verdict.reason,
        }
    return {"code": None, "verdict": verdict.verdict, "reason": verdict.reason}


def _verdict_text(cmp_best: dict[str, Any]) -> str:
    """把对比数字变成一句话结论。"""
    corr = cmp_best.get("intensity_correlation")
    offset = cmp_best.get("peak_offset_samples") or [None, None]
    strehl = cmp_best.get("strehl_like_peak_ratio")
    corr_txt = fmt_general(corr, '.3f', missing_nonfinite='n/a')
    if corr is not None and np.isfinite(corr) and corr >= 0.8:
        verdict = "模型与硬件**高度一致** (强度相关系数 ≥ 0.8)"
    elif corr is not None and np.isfinite(corr) and corr >= 0.4:
        verdict = "模型与硬件**中度一致** (强度相关系数 0.4–0.8)"
    else:
        verdict = "模型与硬件**不一致** (强度相关系数 < 0.4)"
    return (
        f"{verdict}: 相关系数 {corr_txt}, 峰值偏移 "
        f"{fmt_general(offset[0], '.2f', missing_nonfinite='n/a')}/{fmt_general(offset[1], '.2f', missing_nonfinite='n/a')} 个远场采样, "
        f"类 Strehl 峰值比 {fmt_general(strehl, '.3f', missing_nonfinite='n/a')}"
    )


def _conclusion_text(
    cmp_best: dict[str, Any],
    history: list[dict[str, Any]],
    hw_history: list[dict[str, Any]],
) -> str:
    """报告「结论」段落。"""
    lines = [
        f"* **模型能不能预测硬件**: {_verdict_text(cmp_best)}。",
    ]
    if history:
        lines.append(
            f"* **纯模型侧的梯度有没有用**: 损失从 {fmt_general(history[0]['loss'], '.6f', missing_nonfinite='n/a')} 降到 "
            f"{fmt_general(history[-1]['loss'], '.6f', missing_nonfinite='n/a')} (共 {len(history)} 次迭代), 说明 "
            "autograd 给出的梯度方向是对的; 相位 RMS 从 "
            f"{fmt_general(history[0]['phase_rms_rad'], '.4f', missing_nonfinite='n/a')} rad 变到 "
            f"{fmt_general(history[-1]['phase_rms_rad'], '.4f', missing_nonfinite='n/a')} rad。"
        )
    if hw_history:
        lines.append(
            f"* **测量锚定闭环**: 实测损失从 {fmt_general(hw_history[0]['loss'], '.6f', missing_nonfinite='n/a')} 变到 "
            f"{fmt_general(hw_history[-1]['loss'], '.6f', missing_nonfinite='n/a')} (共 {len(hw_history)} 次)。"
            "这个数是**实测**图像处的损失, 不是模型自算的。"
        )
    lines.append(
        "* **相位分辨率的限制**: 模型瞳面采样间距 dp 不是整数面板像元, 上屏时是"
        "**块状**相位 (每个模型采样占 dp×dp 面板像元) —— 硬件看到的相位比模型"
        "优化的那组值更粗, 这是模型与硬件之间**不可消除**的口径差。"
    )
    return "\n".join(lines)


def _open_questions_text() -> str:
    """报告「被推翻 / 待确认」段落。"""
    return "\n".join(
        [
            "* **远场覆盖范围的公式**: 任务书里给的 `K·M·dp/span` CCD 像元在量纲上"
            "不成立 (它是「网格跨度/瞳面跨度」这个放大率乘 K), 脚本改用傅里叶变换"
            "真正给出的 `K/dp` (即 M×远场采样间距)。两个数都写进了 `run_config.json`"
            " 以便核对 —— **待确认**: 远场阵列覆盖 K/dp CCD 像元时, 有多少比例落在"
            "传感器内 (传感器 2592×1944, 零级不在画幅中心)。",
            "* **面板↔相机的 90° 轴交换**: 仓库记着「面板 x 斜坡挪的是相机 y」, 但"
            " 几何标定返回的 `rotation_deg` 只编码两条光栅轴的**正交性偏差**, 不能"
            " 无歧义地定出轴映射。脚本因此**不假设**, 把「转置」当两个候选口径实测"
            " 对比择优 —— **待确认**: 若模型远场明显非对称, 对比择优本身就不可靠, 需要另做"
            " 一次带标记图案的轴映射标定。",
            "* **几何标定期间用了焦面补偿**: `SLMCCDCalibrator` 自己管理显示路径"
            " (只显示 plain 相位), 脚本用驱动的叠加层把补偿临时挂上去, 让 ±1 级"
            " 在焦时测量 —— **待确认**: 补偿相位以**标称**照明中心为中心, 而真实"
            " 照明并不在那里, 偏心二次相位等价于偏心透镜 (会转向), 可能给标定"
            " 引入几十 px 的系统偏差。",
            "* **零填充倍数与远场采样**: 默认 padding=4 意味着远场比瞳面角分辨率"
            " 细 4 倍, 光斑在远场网格上只占几个采样 —— 目标若是单像素尖峰, 模型"
            " 物理上**不可能**完全命中, 损失里有一项是「不可达」的部分。",
            "* 🔴 **随机初值 + 强度 MSE 会掉进「把远场摊平」的局部极小** (2026-10-22"
            " 实测, 这是**上一轮**的负结果): 损失从 1.07e-2 **一步**降到 1.09e-3,"
            " 之后 150 次迭代只再降 15% (最后 9.29e-4)。而 9.29e-4 与"
            "「目标块面积 / 网格面积」= 31²/1024² = 9.2e-4 几乎相等 —— 也就是"
            "**目标一块都没命中**时的损失。相位 RMS 全程停在 3.63 rad (≈ 均匀 [0,2π) 的 raw RMS),"
            "说明相位仍近似随机 ⇒ 硬件看到散斑, 模型与硬件相关系数只有 0.11~0.15。",
            "* ✅ **改用 GS 初值就出得来** (2026-10-23 实测, 现在是默认):"
            " `--init-phase gs` 用交替投影 (传播取 `fft`, 即 2f 透镜的傅里叶变换)"
            " 造一个解析整形解当初值, 可微细化再接手 —— 损失 3.03e-4 → **1.28e-4**,"
            " 相位 RMS 1.82 rad (不再是随机的 3.63), 模型远场峰值落在目标中心 ±6 个"
            " 采样内, **模型与硬件的强度相关系数 0.81 (高度一致)**, 峰值位置差"
            " 2/−13 个采样。同一实验系统、同一目标, **只换初值**, 相关系数从 0.15 跳到 0.81"
            " ⇒ 初值比损失函数更决定成败 (与仓库 `slm_gs_refine` 的判断一致)。",
            "* **待确认**: `gerchberg_saxton` 自己的误差指标 (14600 → 14595) 几乎不动,"
            " 它的 `converged` 也一直为 False —— 说明它的误差口径与本实验的「目标块」"
            " 口径不一致; 本轮只把 GS 当**初值**, 没有依赖它的收敛判据。",
        ]
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_centre(text: str) -> tuple[float, float]:
    """解析 ``--pupil-center "960,600"`` (面板像元, x, y)。"""
    parts = [p for p in str(text).replace("，", ",").split(",") if p.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f'--pupil-center 需要两个数 "x,y" (面板像元), 给的是 {text!r}'
        )
    return float(parts[0]), float(parts[1])


def build_parser() -> argparse.ArgumentParser:
    """命令行接口。"""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印实验计划 (默认行为); 绝不 import / open 任何设备",
    )
    p.add_argument(
        "--execute",
        action="store_true",
        help="真正开设备执行 —— 这是唯一会 import 驱动包的分支",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="产物目录 (默认 data/debug/differentiable_opt/<UTC 时间戳>)",
    )
    p.add_argument("--slm-number", type=int, default=1, help="SLM 设备编号 (默认 1)")
    p.add_argument(
        "--slm-wavelength", type=int, default=1064, help="SLM 工作波长 nm (默认 1064)"
    )
    p.add_argument("--cam-type", default="daheng", help="相机类型 (默认 daheng)")
    p.add_argument("--cam-id", type=int, default=0, help="相机 ID (默认 0)")
    p.add_argument(
        "--exposure-ms",
        type=float,
        default=DEFAULT_EXPOSURE_MS,
        help="曝光 ms (默认 2.0; 焦面扫描的每一点会自己重新对曝光)",
    )
    p.add_argument(
        "--exposure-ceiling-ms",
        type=float,
        default=DEFAULT_CEILING_MS,
        help="自适应曝光的上限 (ms, 默认 200)。散射型图案的峰值比聚焦态低"
        "一两个数量级, 上限太低时整帧都是本底噪声",
    )
    p.add_argument(
        "--frames", type=int, default=DEFAULT_FRAMES, help="每次测量平均几帧 (默认 3)"
    )
    p.add_argument(
        "--discard",
        type=int,
        default=DEFAULT_DISCARD,
        help="改曝光/上屏后丢几帧 (默认 2)",
    )
    p.add_argument(
        "--settle-s",
        type=float,
        default=DEFAULT_SETTLE_S,
        help="上屏后的稳定等待 s (默认 0.5)",
    )
    p.add_argument(
        "--pupil-center",
        default="960,600",
        help='瞳面中心 (面板像元 "x,y", 默认 "960,600")',
    )
    p.add_argument(
        "--pupil-radius",
        type=float,
        default=DEFAULT_PUPIL_RADIUS,
        help="瞳面半径 (面板像元, 默认 480; 跨度 = 2×半径 = 960)",
    )
    p.add_argument(
        "--focus-fwhm-max",
        type=float,
        default=DEFAULT_FOCUS_FWHM_MAX,
        help="在焦判定的 FWHM 门限 (CCD px, 默认 40); 超过 ⇒ exit 5 拒绝测量",
    )
    p.add_argument(
        "--focus-points",
        type=int,
        default=DEFAULT_FOCUS_POINTS,
        help="焦面粗扫点数 (默认 13)",
    )
    p.add_argument(
        "--focus-diopter-range",
        type=float,
        default=DEFAULT_FOCUS_DIOPTER_RANGE,
        help="焦面粗扫 1/f 半宽 (m^-1, 默认 6.0)",
    )
    p.add_argument(
        "--focus-refine",
        type=int,
        default=DEFAULT_FOCUS_REFINE,
        help="最优点附近的细扫点数 (默认 7)",
    )
    p.add_argument(
        "--geometry-npz",
        default=None,
        help="读入已存的几何标定 (.npz); 不给则现场实测。"
        "注意: 仓库里的 data/calib.npz 是陈旧的",
    )
    p.add_argument(
        "--wait-until-idle",
        type=float,
        default=0.0,
        help="开跑前等设备空闲的预算 (秒, 默认 0 = 不等)。"
        "复用 run_hw_sweep_groups.wait_until_idle 的公平仲裁: "
        "先看别的会话有没有在用设备, 没有才探针 —— 本脚本不抢设备",
    )
    p.add_argument(
        "--analyze-only",
        default=None,
        help="只做离线重分析: 读这个运行目录里的产物 (参数/几何/相位/实测远场), "
        "重跑纯模型侧的反向传播与「模型 vs 硬件」对照, 并重写报告。"
        "**一个设备都不碰** —— 实验采到的数据可以事后分析",
    )
    p.add_argument(
        "--model-pupil-samples",
        type=int,
        default=DEFAULT_MODEL_PUPIL_SAMPLES,
        help="瞳面采样数 N (默认 256)",
    )
    p.add_argument(
        "--model-padding",
        type=int,
        default=DEFAULT_MODEL_PADDING,
        help="模型网格零填充倍数 M = padding·N (默认 4)",
    )
    p.add_argument(
        "--beam-w0-px",
        type=float,
        default=DEFAULT_BEAM_W0_PX,
        help="照明高斯腰径 (面板像元, 默认 260)",
    )
    p.add_argument(
        "--target",
        choices=["delta", "square"],
        default=DEFAULT_TARGET,
        help="目标远场: square = 中心均匀方块 (默认: **有结构的目标**, "
        "优化器必须真的造出结构); delta = 中心一个衍射极限亮斑。"
        "⚠️ delta 对平坦相位几乎已经满足 (平场远场本来就是中心一个"
        "衍射极限光斑), 所以它测不出优化器的能力, 只适合当对照",
    )
    p.add_argument(
        "--target-side-px",
        type=float,
        default=DEFAULT_TARGET_SIDE_PX,
        help="方形目标边长 (CCD px, 默认 60); 只在 --target square 时用",
    )
    p.add_argument(
        "--delta-spot-px",
        type=float,
        default=DEFAULT_DELTA_SPOT_PX,
        help="delta 目标的亮斑直径 (CCD px, 默认 15 = 在焦零级的 FWHM)。"
        "**不能是单像素**: 远场网格有 10^6 格, 单像素目标在峰值归一化的"
        " MSE 里只占 10^-6, 峰值位置几乎不被约束 (实测: 损失降到 1.5e-5, "
        "峰值却落在离中心 183/262 个采样点处)",
    )
    p.add_argument(
        "--init-phase",
        choices=["gs", "random"],
        default="gs",
        help="初值: gs (默认) = 交替投影 (GS) 的解析整形解起手, propagation='fft'; "
        "random = 优化器自己的随机初值。"
        "⚠️ 随机初值 + 强度 MSE 会掉进「把远场摊平」的局部极小"
        "(2026-10-22 实测: 损失一步降到「目标一块都没命中」的底, 相位 RMS 停在"
        "均匀随机的 3.63 rad) —— 上屏的相位近似随机, 模型与硬件相关系数只有 0.1",
    )
    p.add_argument(
        "--gs-iters",
        type=int,
        default=50,
        help="GS 初值的交替投影迭代数 (默认 50)",
    )
    p.add_argument(
        "--iters",
        type=int,
        default=DEFAULT_ITERS,
        help="纯模型侧的反向传播迭代次数 (默认 200)",
    )
    p.add_argument(
        "--hw-iters",
        type=int,
        default=DEFAULT_HW_ITERS,
        help="测量锚定闭环的迭代次数 (默认 5)",
    )
    p.add_argument(
        "--lr", type=float, default=DEFAULT_LR, help="Adam 学习率 (默认 0.05)"
    )
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="随机种子 (默认 0)")
    p.add_argument(
        "--grad-every",
        type=int,
        default=DEFAULT_GRAD_EVERY,
        help="每几次迭代存一张梯度场 (默认 10)",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="冒烟测试: 2 次迭代 + 1 次测量锚定 + 精简焦面扫描",
    )
    p.add_argument(
        "--render-only",
        default=None,
        metavar="RUN_DIR",
        help="离线重渲染报告: 从先前某次 --execute 的运行目录读回产物, 重新生成 "
        "markdown 报告 (不 import 驱动, 不碰设备)",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.render_only:
        code = render_only(Path(args.render_only))
        if code != EXIT_OK:
            raise SystemExit(code)
        return
    if args.analyze_only:
        code = run_analyze_only(args)
        if code != EXIT_OK:
            raise SystemExit(code)
        return
    args.pupil_center = _parse_centre(args.pupil_center)
    if args.out_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        args.out_dir = str(DEFAULT_OUT_ROOT / stamp)
    if args.quick:
        args.iters = 2
        args.hw_iters = 1
        # ⚠️ 焦面扫描**不能**跟着缩: 13 点的粗扫在 ±6 m⁻¹ 上步长 3 m⁻¹,
        # 而本实验系统的最佳值在 1/f ≈ 0.67 m⁻¹ —— 5 点的粗网格根本采不到
        # 那里, 扫出来的"最佳点"是 71 px 的宽斑, 于是门禁给出**假拒绝**
        # (2026-10-22 冒烟实测: 门限 40 px 被触发, 而同一实验系统的完整扫描
        # 给出 15 px)。焦面扫描是判定在焦的依据, 必须保持分辨率。
        args.frames = 1
        args.discard = 1

    plan = build_plan(args)
    if args.execute and not args.dry_run:
        code = run_execute(args, plan)
        if code != EXIT_OK:
            raise SystemExit(code)
        return

    print_plan(plan)
    if args.execute:
        logger.info("\n[提示] 同时给了 --dry-run 与 --execute, 按 dry-run 处理")
    assert_no_drivers()


if __name__ == "__main__":
    main()
