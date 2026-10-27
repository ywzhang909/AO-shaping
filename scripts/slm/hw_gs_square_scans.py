"""GS 方形光斑三段扫描 (边长 / Zernike 阶数 / 闪耀光栅周期), 结果全部落盘。

🔴 **原子化调用是唯一推荐路径** (见「原子化调用」节): 每类探测都能只跑**一组**
参数, 由外部循环逐个进程执行; ``--plan-commands`` 打印那份命令清单。

做什么
======

只用 **Gerchberg-Saxton** 造方形远场光斑, 沿三个自变量各扫一遍, 每一配置一次
测量 (下发相位 → 等稳定 → 取帧 → 算指标 → 落盘):

1. ``sides``: **正方形边长扫描** —— 从 ``--side-min`` 到「由 0 级位置与 CCD 画面
   边缘算出的最大边长」, 步长 ``--side-step``; 每个边长跑一次 GS 造方斑, 测拟合
   效果 (CV / PIB / EE / AR / 复合分数)。
2. ``zernike``: 取第 1 段里**边长接近 1000** 且复合分数最高的那个边长, 把它的 GS
   相位拟合成 ``--zernike-orders`` 范围的 Zernike 系数, 只用**前 N 阶**重构相位,
   逐个测量 (N = 7..30)。
3. ``blaze``: 在选定相位上叠加**不同周期的闪耀光栅** (面板 x 斜坡), 逐个测量,
   并记录 ``predicted_shift_px`` 与**实测质心位移**。

三段之间只通过「选定边长 / 选定相位」单向传递, 不做任何闭环优化 —— 本脚本里
**没有**任何黑盒随机梯度迭代 (GS 是交替投影, 不存在随机数, 见下)。

🔴 **本脚本默认 ``--dry-run``: 只打印完整实验计划, 绝不 import 或 open 任何设备。**

关键口径 (改之前先读)
====================

边长 → GS 网格像素 (本脚本唯一需要推导的换算)
--------------------------------------------

2f 光路上 CCD 看到的是 SLM 瞳面的傅里叶变换, 所以一个 ``N x N`` 网格 (网格
**正好跨过**照明圆盘的包围方框 ``D = 2·R`` 个面板 px) 的远场采样间距是

    Δx_cam = λ·f / (N · d_eff · p_cam),   d_eff = D·d / N
           = λ·f / (D · d · p_cam)  =  TILT_SHIFT_SCALE / D        ← 与 N 无关

其中 ``TILT_SHIFT_SCALE = λ·f/(d·p_cam)`` 是**实测**的 2f 焦面标度
(:mod:`ao_shaping.utils.slm.bench_constants`)。于是:

* **一个 GS 网格像素 = ``TILT_SHIFT_SCALE / pupil_side_px`` 相机像素**
  (本实验系统: 7400/900 = 8.222 camera px)。**加大 N 不会让远场更细**, 只会把
  FFT 窗口变宽 (窗口 = ``N · Δx_cam``) —— 这是傅里叶采样的定标关系, 不是近似。
* 因此目标方斑边长换算只有一条式子::

      side_gs_px = side_cam_px · pupil_side_px / TILT_SHIFT_SCALE

  它只用到 ``f/p_cam`` 这一个组合, 与 ``--wavelength-nm`` 的锚点选择无关
  (:func:`~ao_shaping.utils.wavefront.matrix_utils.focal_length_from_camera_pixel`
  的 docstring 明确写了这一点)。
* 网格边长由「远场窗口必须装得下整段扫描」反解::

      N = next_pow2( ceil( (side_max + 2·max_blaze_shift) / Δx_cam ) )

  取 2 的幂只是为了 FFT 快。窗口装不下时, 目标方斑之外的相位会在窗口边界**卷绕**
  回来, 所以这一条不是保守估计而是必要条件。

最大边长: 用 0 级位置与画面边缘算, 绝不假设 0 级在画面中心
----------------------------------------------------------

平场帧上 ``argmax`` 定位 0 级 (:func:`~ao_shaping.utils.image.beam_metrics.zero_order_center`)
并**冻结**, 然后

    side_max = 2 · ( min(cx, W-1-cx, cy, H-1-cy) - edge_margin )

向下取到 ``--side-step`` 的整数倍。``edge_margin`` 默认
:data:`~ao_shaping.utils.image.frame_analysis.TARGET_EDGE_MARGIN_PX` (10 px):
框压在帧边上会被读出噪声与窗函数污染, 而这一条正是那个常量的用途。
(**本实验系统的 0 级实测偏离画面中心数百 px, 所以这个式子必须现场量。**)

近 1000 的选点 (显式失败, 不静默退化)
-------------------------------------

``--target-side-near 1000 --near-tolerance 200`` ⇒ 在 ``[800, 1200]`` 内取复合
分数最高者。区间内**一个点都没有**时**报错退出** (退出码 5) 并打印实际边长列表
—— 静默退化成「全局最优」会让第 2 段的输入变成另一个边长的相位, 而清单上仍然
写着「接近 1000」。

光栅周期与位移
--------------

``--blaze-periods`` 默认 ``16 24 32 40 48 64 96 128`` (面板 px, ≥2 才在 Nyquist
之上)。位移用 :func:`~ao_shaping.utils.wavefront.zero_order_overlay.predicted_shift_px`
(已含面板↔相机 90° 交换: 面板 x 斜坡 → 相机 y), 幅值 ``TILT_SHIFT_SCALE/P``。

⚠️ **叠加光栅后冻结的方框失效, 而且整帧 ``argmax`` 也不再是方斑**: 真实瞳面是
双分量 ``U = U_inc·[η + (1−η)e^{iφ}]``, 那个 ``η`` 项不经相位调制, 始终停在 0 级
(衍射极限大小, 实测占 2.86% 能量) —— 它每像素的亮度比摊开到上千 px 的方斑高几个
数量级。所以第 3 段**按预测位移放框** (框心 = 冻结 0 级 + ``predicted_shift_px``),
质心在**框内**量 (:func:`~ao_shaping.utils.image.spots_calc.centroid` 于裁剪窗口),
整帧峰值单独记 —— 这就是「随斑框 + 整帧峰值」的口径。

焦面补偿: 默认**不加**
----------------------

CCD 现在就落在 2f 焦面上 ⇒ 旧的瞳面菲涅尔补偿 (``f_v ≈ 1549 mm``) **已过期**,
默认不加。需要时 ``--defocus-focal-mm <mm>`` 显式给 (``inf`` = 显式选择平场)。
补偿相位走驱动的叠加层 (:meth:`set_phase_overlay`), 与模式相位分开 —— 第 3 段的
闪耀光栅则**加进模式相位数组** (两者都能加, 因为叠加层只有一条通道)。

指标口径 (与 ``scripts/slm/compare_sim_vs_hw.py`` 同一批函数)
-------------------------------------------------------------

* ``cv`` / ``pib`` / ``score`` = ``0.5·PIB + 0.5/(1+CV)`` —— 来自
  :mod:`ao_shaping.algorithm.signal_processing.square_shaping`
  (``uniformity_cv`` / ``power_in_bucket`` / ``composite_from_pib_cv``), 支撑域是
  :func:`ao_shaping.utils.image.targets.square_mask`, 帧先过
  :func:`~ao_shaping.utils.image.beam_metrics.prepare_frame` (减中位数 + 裁负),
  否则读噪声会把 PIB 顶到 1 以上。
* ``ee`` / ``ar`` / ``quality`` —— 来自
  :mod:`ao_shaping.optimizer.wfless.slm_square_shaping`
  (``square_encircled_energy`` / ``square_aspect_ratio`` / ``square_quality_score``),
  即方形整形优化器自己的那套。**惰性 import**: 该模块在模块级就 import
  ``ao_shaping.drivers.*`` 与 ``ao_shaping.display`` (后者牵入 pygame), 顶层导入会
  破坏 dry-run 的「零驱动包」保证。
* 光斑几何 (整帧最亮结构的 FWHM / 空心度) 来自
  :func:`~ao_shaping.utils.image.frame_analysis.measure_spot`。2f 光路上整帧最亮的
  结构通常是 0 级或它的晕斑, **不是**被整形的方斑 —— 字段名带 ``global_`` 就是为了
  这一点。

dry-run 的硬保证
----------------

不给 ``--execute`` 时只 import 零导入叶子与 ``scripts`` 内部的纯函数, 收尾断言
``ao_shaping.drivers`` 没进过 ``sys.modules`` (:func:`assert_no_drivers`, 复用
``hw_gs_defocus_probe``)。违反就 exit 3。因此 ``--dry-run`` 在**没有任何硬件**的
机器上必须 exit 0。

dry-run 里的「画面尺寸 / 0 级位置」是**计划假设**, 不是测量: 用
``--dry-run-frame-shape`` / ``--dry-run-zero-order`` 给 (默认取画幅中心), 打印时会
明确标注。实测里这两个量一律现场量。

退出码: 0 = 正常; 2 = 读帧超时 (复用的 :func:`guarded` 硬退出); 3 = dry-run 违约;
4 = 光路受阻; 5 = 近 1000 区间内没有扫描点; 6 = 缺 ``selected.json`` (第 2/3 段
依赖选点却没跑 ``--scan select``); 7 = 该阶段没有任何可聚合的测量行。

原子化调用 (推荐)
=================

整段扫描的问题是它把「设备生命周期」和「参数网格」绑在一起: 19 点边长 + 24 点
阶数 + 8 点周期一次跑完要好几个小时, 中途任何一次坑 (USB 瞬时错误、光路漂移、
操作者离开) 都会把**整轮**作废, 而已经落盘的点也说不清属于哪一组。

所以现在**一个进程只跑一组参数**, 跨进程靠 ``--run-root`` 累积::

    python scripts/slm/hw_gs_square_scans.py --execute --scan sides  --side-px 700 --run-root <RUN>
    python scripts/slm/hw_gs_square_scans.py --execute --scan zernike --n-max 11  --run-root <RUN>
    python scripts/slm/hw_gs_square_scans.py --execute --scan blaze   --blaze-period 32 --run-root <RUN>

外加三个**离线**子命令 (不打开任何设备, 不 import 驱动包)::

    --scan select      读累积的 sides 测量, 在近 1000 窗口内按 --rank-by 选点, 写
                       ``<RUN>/sides/selected.json`` (第 2/3 段的唯一输入)
    --scan summarize   读累积的全部配置产物, 重建每段 measurements.csv + 根
                       manifest.json + 每段 summary.png
    --plan-commands    只打印可复制的命令行 (每行一个进程), 本身不跑设备

累积布局::

    <RUN>/manifest.json                     根清单 (按段就地更新)
    <RUN>/<type>/measurements.csv           每段一张, 按 config-slug 去重
    <RUN>/<type>/measurements.json          同一份数据的完整记录 (类型保留)
    <RUN>/<type>/run_config.json            段级参数 (最近一次会话; 既有报告脚本读它)
    <RUN>/<type>/summary.png                --scan summarize 画的汇总图
    <RUN>/<type>/<config-slug>/phase_000.npy   下发过的面板相位 (raw 弧度)
    <RUN>/<type>/<config-slug>/pupil_phase.npy GS 网格相位 (第 2/3 段复用它, 省一次 GS)
    <RUN>/<type>/<config-slug>/frame_000.npz   原始平均帧
    <RUN>/<type>/<config-slug>/metrics.json    该配置的一整行 (含 preflight)

``<config-slug>`` 形如 ``side_700`` / ``nmax_11`` / ``blaze_32``。**重跑同一配置
就地替换** (CSV 与 manifest 都不产生第二行), 别的配置不动; 所有 JSON/CSV 都先写
临时文件再 ``os.replace``, 所以 Ctrl-C 不会留下半个文件。

依赖解析 (第 2/3 段依赖第 1 段的选点)
------------------------------------

第 2/3 段从 ``<RUN>/sides/selected.json`` 读「选定边长 + 基准相位」, **不自己选点、
也不自己扫 sides**。基准相位**优先复用** ``<RUN>/sides/<slug>/pupil_phase.npy``
(GS 是确定性的, 缓存与重算逐位一致; 复用省掉每次约 7.5 s 的 GS), 缓存缺失或形状
与本次网格不符才重算, 两种情况都在 ``metrics.json`` 的 ``base_phase_source`` 里
如实记成 ``"cache"`` / ``"recomputed"``。

``selected.json`` 缺失时**报错退出** (退出码 6, 消息里点名 ``--scan select``);
静默退化成「全局最优」或默认 1000 会让产物清单与实际相位脱节。

``--side-px`` 仍可显式指定边长 (旧行为, 作为逃生口), 那时
``selected_source = "cli"`` —— 它不是「选点」, 只是一个显式常量。

路径语义
--------

* ``--run-root``: 批次根 (累积目录)。
* ``--out-root``: 只给 ``--out-root`` 时它就是批次根 (**完全向后兼容**旧语义);
  与 ``--run-root`` **同时**给时, 它在原子模式下覆盖该段的目录
  (``<run-root>/<type>`` 的替换), 整段模式仍按旧语义做根目录。

运行
----

.. code-block:: bash

    # 无硬件机器上也能跑: 只打印计划
    python scripts/slm/hw_gs_square_scans.py --dry-run
    python scripts/slm/hw_gs_square_scans.py --dry-run --dry-run-zero-order 930,1050

    # 原子模式 (推荐): 一个进程一组参数; 先打印命令清单
    python scripts/slm/hw_gs_square_scans.py --plan-commands --run-root data/debug/hw_gs_square_scans/r1
    python scripts/slm/hw_gs_square_scans.py --execute --scan sides --side-px 700 --run-root <RUN>
    python scripts/slm/hw_gs_square_scans.py --scan select --run-root <RUN>
    python scripts/slm/hw_gs_square_scans.py --execute --scan zernike --n-max 11 --run-root <RUN>

    # 整段扫描 (仅调试: 一次几小时; 推荐上面的原子模式 + --plan-commands)
    python scripts/slm/hw_gs_square_scans.py --execute --scan all
    python scripts/slm/hw_gs_square_scans.py --execute --scan sides
    python scripts/slm/hw_gs_square_scans.py --execute --scan zernike --side-px 1000
    python scripts/slm/hw_gs_square_scans.py --execute --scan blaze --side-px 1000

    # 数字孪生 (不碰硬件; 同结构产物)
    python scripts/slm/hw_gs_square_scans.py --execute --cam-type sim --scan sides

产物
----

原子模式见上「累积布局」节。整段模式沿用旧布局: ``<root>/<stage>/`` 每配置
``phase_<idx>.npy`` (raw 弧度, 面板分辨率, 不含叠加层) + ``frame_<idx>.npz``
(原始平均帧, float32) + ``metrics_<idx>.json``; 每段 ``measurements.csv`` +
``measurements.json`` + ``run_config.json`` + ``summary.png``。根目录
``manifest.json`` 汇总三段 (复用 :mod:`scripts._common.group_manifest`, 三段的
``variable`` 分别是 ``side_px`` / ``n_max`` / ``blaze_period_px``)。
"""

from __future__ import annotations

import argparse
from loguru import logger
import csv
import json
import math
import os
import sys
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

ROOT = Path(__file__).resolve().parents[2]
# `scripts.*` 以包路径被导入, 所以仓库 ROOT 本身必须在 sys.path 上; 文件形式启动
# (`python scripts/slm/x.py`) 时 cwd 不在 sys.path, 只有脚本自己的目录在。
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

# ---------------------------------------------------------------------------
# 复用: 全部是仓库里的既有实现, 本文件不另写一套数学
# ---------------------------------------------------------------------------
from ao_shaping.algorithm.signal_processing.gerchberg_saxton import (
    GSResult,
    gerchberg_saxton,
)
from ao_shaping.algorithm.signal_processing.square_shaping import (
    composite_from_pib_cv,
    power_in_bucket,
    uniformity_cv,
)
from ao_shaping.tools.kernels.bench_kernels import fresnel_panel
from ao_shaping.utils.image.beam_metrics import prepare_frame, zero_order_center
from ao_shaping.utils.image.frame_analysis import (
    TARGET_EDGE_MARGIN_PX,
    assess_spot,
    crop_roi,
    measure_spot,
)
from ao_shaping.utils.image.resample import upsample_bilinear_to_shape
from ao_shaping.utils.image.spots_calc import centroid
from ao_shaping.utils.image.targets import square_mask
from ao_shaping.utils.slm.bench_constants import (
    BEAM_CENTER_PANEL,
    BEAM_RADIUS_PANEL,
    SLM_PANEL_H,
    SLM_PANEL_W,
    SLM_PITCH_M,
    TILT_SHIFT_SCALE,
)
from ao_shaping.utils.slm.phase_embed import embed_pupil_square, pupil_side
from ao_shaping.utils.wavefront.matrix_utils import focal_length_from_camera_pixel
from ao_shaping.utils.wavefront.zernike_calc import (
    ZernikeGenerator,
    calc_n_zernike_terms,
)
from ao_shaping.utils.wavefront.zero_order_overlay import (
    apply_zero_order_overlay,
    predicted_shift_px,
    OverlayParams,
    ZeroOrderOverlay,
)
from scripts._common.group_manifest import (
    MANIFEST_NAME,
    group_record,
    load_manifest,
)
from scripts.slm.hw_gs_defocus_probe import (
    ACQUIRE_TIMEOUT_S,
    CAMERA_PIXEL_UM,
    EXIT_PATH_BLOCKED,
    OPEN_TIMEOUT_S,
    SCALE_MEASURED_AT_NM,
    USB_RETRY_ATTEMPTS,
    USB_RETRY_BASE_WAIT_S,
    assert_no_drivers,
    configure_sim_system,
    device_driver_strings,
    guarded,
    protected_memory_slot,
    read_default_wavelength_nm,
    read_exposure_ms,
)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
#: 三段的自变量名 —— 写进根 ``manifest.json`` 的 ``variable`` 字段。
STAGE_VARIABLES: dict[str, str] = {
    "sides": "side_px",
    "zernike": "n_max",
    "blaze": "blaze_period_px",
}
STAGES: tuple[str, ...] = ("sides", "zernike", "blaze")

#: 一段的墙钟预算 (s)。超过它就在开跑前**警告并建议拆段** —— SLM 的 USB 句柄是
#: 独占的, 一段跑到一半被迫中断, 后面的段要重新排队。
PLAN_LIMIT_S = 600.0

#: 单次 GS 迭代的耗时模型: ``1.6e-8 · N² · log2 N`` 秒。
#:
#: 实测 (本机, 1920x1200 面板几何, ``propagation="fft"``, 每迭代 2 次 FFT):
#: N=256 → 8.4 ms; N=512 → 35.2 ms; N=900 → 124.7 ms; N=1024 → 281.4 ms;
#: N=1800 → 1051.6 ms。前四档都落在 1.5-1.6e-8 上, Python 侧开销在小 N 上占比更大
#: (所以模型对 N=256 略微低估, 可接受: 它只用于**开跑前的耗时估计**)。
GS_SECONDS_PER_ITER_UNIT = 1.6e-8

#: 每配置的采集耗时估计 (s) —— SLM 写入 + 稳定性判据 + 平均若干帧。
#:
#: 依据: ``bench_kernels.display_and_average`` 的判据要 3 帧丢弃 + 若干探针帧 +
#: ``n_frames`` 帧平均; 本机实测的同类点耗时落在 8.1 s/点 (几何扫描) 到 24.1 s/点
#: (面板扫描, 含逐帧落盘) 之间。这里取 8.0 作**下限量级**, 可用
#: ``--estimate-capture-s`` 覆盖成你自己实测的值。
DEFAULT_ESTIMATE_CAPTURE_S = 8.0

#: 每帧 ``.npz`` 的磁盘占用估计 (MB, 已压缩, float32)。全画幅 1944x2592 的
#: float32 是 20 MB, 压缩后本机实测落在 5-15 MB。
ESTIMATE_NPZ_MB = 10.0
#: A pair specification (``HxW``, ``X,Y``) carries exactly this many components; the
#: parsers reject anything else rather than guessing which half was meant. Not used for
#: array dimensionality, which is a different question.
PAIR_FIELD_COUNT = 2

#: Highest Zernike order this scan supports. The range check and the ``--n-max`` guard
#: both quote it, and the failure message prints the same 1..30 range, so it lives here.
MAX_ZERNIKE_ORDER = 30

#: A square ROI needs at least this many pixels before its CV/PIB/score metrics mean
#: anything; below it the metrics are returned as NaN instead of computed.
MIN_MASK_PIXELS = 4

#: Order list shown inline in the report before it is abbreviated to head and tail.
#: A display choice, deliberately separate from any hardware limit.
MAX_ORDERS_LISTED_INLINE = 6

#: ``--near-tolerance`` 缺省值 (相机 px)
DEFAULT_NEAR_TOLERANCE_PX = 200

#: ``--target-side-near`` 缺省值 (相机 px) —— 第 1 段选点的目标边长。
DEFAULT_NEAR_TARGET_PX = 1000.0

#: 近 1000 选点在区间内找不到任何点时本脚本的退出码。
#:
#: 与 2 (读帧超时) / 3 (dry-run 违约) / 4 (光路受阻) 区分开: 这是**参数与扫描
#: 网格不匹配**, 操作者改 ``--side-min/--side-step`` 或 ``--near-tolerance`` 即可,
#: 不该去查驱动或光路。
EXIT_NO_NEAR_SIDE = 5

#: 第 2/3 段需要选点, 但 ``<run-root>/sides/selected.json`` 不存在 / 不可用。
#:
#: 与 5 分开: 5 是「扫了但窗口里没有点」(改扫描网格), 6 是「压根没选」(先跑
#: ``--scan select``) —— 两种处置完全不同, 混成一个码会让操作者先怀疑参数。
EXIT_SELECTION_MISSING = 6

#: 离线聚合 (select / summarize) 找不到任何可用的测量行。
EXIT_NO_MEASUREMENTS = 7

#: 离线子命令: 不打开任何设备, 也不 import 驱动包。
OFFLINE_SCANS: tuple[str, ...] = ("select", "summarize")

#: 原子模式一个进程跑**一组**参数; 这些字段把三类探测的「一组」定义在一个地方。
STAGE_VALUE_ARG: dict[str, str] = {
    "sides": "side_px",
    "zernike": "n_max",
    "blaze": "blaze_period_px",
}

#: 原子模式下「一组」怎么在命令行上给 (命令清单与提示都从这里取, 免得各写一遍)。
STAGE_VALUE_FLAG: dict[str, str] = {
    "sides": "--side-px",
    "zernike": "--n-max",
    "blaze": "--blaze-period",
}

#: 原子模式的配置目录名前缀 (``side_700`` / ``nmax_11`` / ``blaze_32``)。
STAGE_SLUG_PREFIX: dict[str, str] = {
    "sides": "side",
    "zernike": "nmax",
    "blaze": "blaze",
}

#: 累积产物里每个配置的三个文件名 (``phase_*.npy`` / ``frame_*.npz`` / ``metrics.json``)。
CONFIG_PHASE_NAME = "phase_000.npy"
CONFIG_PUPIL_PHASE_NAME = "pupil_phase.npy"
CONFIG_FRAME_NAME = "frame_000.npz"
CONFIG_METRICS_NAME = "metrics.json"
CONFIG_CSV_NAME = "measurements.csv"
CONFIG_JSON_NAME = "measurements.json"
SELECTED_NAME = "selected.json"

#: 累积产物按「配置」去重时用的列名; 它就是配置目录名。
CONFIG_KEY = "config"

#: 命令清单里「一批」的墙钟建议上限 (s) —— 与 :data:`PLAN_LIMIT_S` 一致 (10 分钟),
#: 但两者含义不同: 后者是「一段跑不完就拆段」的告警线, 前者是「外部循环每批跑几个
#: 进程」的建议步长。
PLAN_BATCH_S = 600.0

#: GS 网格的最小边长 (样本)。再小就连方斑的边都表示不出来。
MIN_GS_GRID = 64

#: 光栅周期的下界 (面板 px): 周期 1 px 不是斜坡而是棋盘格, 光斑会被甩到几千相机
#: px 之外 (与 ``zero_order_overlay`` 的 Nyquist 说明一致)。
MIN_BLAZE_PERIOD_PX = 2

#: 默认的 Zernike 阶数范围 (Noll 1976 的径向阶 n)。7 阶 = 36 个模式 (含 piston),
#: 30 阶 = 496 个模式。
DEFAULT_ZERNIKE_ORDERS = "7..30"

#: 默认的闪耀光栅周期 (面板 px)
DEFAULT_BLAZE_PERIODS: tuple[int, ...] = (16, 24, 32, 40, 48, 64, 96, 128)

#: Zernike 拟合网格的默认边长 (样本)。**这不是随便取的**:
#:
#: :class:`~ao_shaping.utils.wavefront.zernike_calc.ZernikeGenerator` 的
#: ``RZern`` 会为 ``(G, G, radius, n_orders)`` 建一张 ``G²·nk·8`` 字节的表
#: (nk = (n_orders+1)(n_orders+2)/2)。``n_orders=30`` 时 nk=496, 于是
#: G=900 → 3.2 GB, G=450 → 803 MB, G=256 → 260 MB, G=128 → 65 MB。
#: 选 256: 拟合到 30 阶时每个径向周期仍有 ~4 个样本 (256/2/30), 内存却在 300 MB
#: 以内。需要更高保真就 ``--zernike-fit-grid`` 显式加大 (并清楚代价)。
DEFAULT_ZERNIKE_FIT_GRID = 256

#: ``--dry-run`` 里假设的画幅 (Daheng MER2-507-23GM, 2592x1944)
DEFAULT_FRAME_SHAPE = (1944, 2592)

#: ``--out-root`` 的缺省父目录
DEFAULT_OUT_PARENT = "data/debug/hw_gs_square_scans"


# ---------------------------------------------------------------------------
# 实验系统几何
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Bench:
    """本次运行用到的实验系统几何 (m / px)。

    ``focal_m`` **不参与** GS 的任何换算 —— ``propagation="fft"`` 下它只是
    :func:`gerchberg_saxton` 的一个形状参数 (该函数要求 ``distance > 0``)。它由实测
    焦面标度反解出来只是为了**记进产物**并保持单一事实源
    (:func:`focal_length_from_camera_pixel`)。
    """

    wavelength_nm: float
    focal_m: float
    panel_shape: tuple[int, int]
    beam_center_panel: tuple[int, int]
    beam_radius_px: float
    camera_pixel_um: float

    @property
    def pupil_side_px(self) -> int:
        """照明圆盘的包围方框边长 (面板 px) —— 也就是 GS 网格覆盖的物理宽度。"""
        return pupil_side(self.beam_radius_px)

    @property
    def camera_px_per_gs_px(self) -> float:
        """一个 GS 网格像素等于多少相机像素 (与网格边长无关)。

        推导见模块 docstring: ``Δx_cam = λf/(D·d·p_cam) = TILT_SHIFT_SCALE/D``。
        """
        return float(TILT_SHIFT_SCALE) / float(self.pupil_side_px)

    def side_cam_to_gs(self, side_cam_px: float) -> float:
        """相机 px → GS 网格 px (目标方斑边长)。"""
        return float(side_cam_px) / self.camera_px_per_gs_px

    def gs_grid_for(self, span_cam_px: float) -> int:
        """覆盖 ``span_cam_px`` 相机像素所需的 GS 网格边长 (2 的幂)。

        远场窗口 = ``N · Δx_cam``, 所以 ``N`` 必须让整段扫描 (含光栅位移) 落在窗口
        内, 否则窗口边界会把目标区之外的相位**卷绕**回来。
        """
        need = float(span_cam_px) / self.camera_px_per_gs_px
        n = MIN_GS_GRID
        while n < need:
            n *= 2
        return int(n)


def build_bench(args: argparse.Namespace) -> Bench:
    """组装几何常量。``--wavelength-nm 0`` 在这里解析成驱动包常量。

    ``read_default_wavelength_nm`` 用 ``ast`` **静态读取** ``drivers/slm/factory.py``
    里的 ``DEFAULT_WAVELENGTH_NM``, 不 import 驱动包 —— 这样既保住单一事实源, 又
    不破坏「dry-run 不碰设备」的保证。
    """
    wavelength_nm = float(args.wavelength_nm)
    if wavelength_nm <= 0:
        wavelength_nm = read_default_wavelength_nm()
    args.wavelength_nm = wavelength_nm
    focal = focal_length_from_camera_pixel(
        wavelength_nm=wavelength_nm,
        camera_pixel_um=CAMERA_PIXEL_UM,
        slm_pixel_um=SLM_PITCH_M * 1e6,
        focal_scale_px=TILT_SHIFT_SCALE,
        scale_measured_at_nm=SCALE_MEASURED_AT_NM,
    )
    return Bench(
        wavelength_nm=wavelength_nm,
        focal_m=float(focal),
        panel_shape=(SLM_PANEL_H, SLM_PANEL_W),
        beam_center_panel=(int(BEAM_CENTER_PANEL[0]), int(BEAM_CENTER_PANEL[1])),
        beam_radius_px=float(args.beam_radius_px),
        camera_pixel_um=CAMERA_PIXEL_UM,
    )


# ---------------------------------------------------------------------------
# 纯函数: 边长网格 / 最大边长 / 近 1000 选点 / 阶数解析
# ---------------------------------------------------------------------------
def parse_frame_shape(text: str) -> tuple[int, int]:
    """``"1944x2592"`` / ``"1944,2592"`` → ``(height, width)``。"""
    raw = str(text).lower().replace("x", ",").replace("*", ",")
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) != PAIR_FIELD_COUNT:
        raise ValueError(f"--dry-run-frame-shape 需要 HxW (如 1944x2592), 收到 {text!r}")
    height, width = int(parts[0]), int(parts[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"画面尺寸必须为正, 收到 {height}x{width}")
    return height, width


def parse_xy(text: str) -> tuple[float, float]:
    """``"930,1050"`` → ``(x, y)`` (相机坐标, 与 ``zero_order_center`` 同序)。"""
    parts = [p.strip() for p in str(text).replace(";", ",").split(",") if p.strip()]
    if len(parts) != PAIR_FIELD_COUNT:
        raise ValueError(f"坐标需要 X,Y 形式 (如 930,1050), 收到 {text!r}")
    return float(parts[0]), float(parts[1])


def max_side_from_frame(
    shape: tuple[int, int],
    center: tuple[float, float],
    *,
    margin_px: int = TARGET_EDGE_MARGIN_PX,
    step_px: int = 1,
) -> tuple[int, int]:
    """由 0 级位置与画面边缘算最大边长, 并向下取到 ``step_px`` 的整数倍。

    ``side_max = 2 · ( min(cx, W-1-cx, cy, H-1-cy) - margin )``

    0 级**必须**现场 ``argmax`` 量 (本实验系统实测偏离画面中心数百 px), 传进来的
    ``center`` 就是那个测量值 —— 唯一例外是 ``--dry-run``, 那里的坐标是计划假设。

    Args:
        shape: 画面 ``(height, width)``。
        center: 0 级中心 ``(x, y)``。
        margin_px: 边缘留白 (px): 框压在帧边上会被读出噪声与窗函数污染。
        step_px: 边长步长; 返回值是它的整数倍。

    Returns:
        ``(side_max_raw, side_max_floored)``: 未取整与取整后的最大边长。
        可用半径为负 (0 级贴着帧边) 时两者都是 0。
    """
    height, width = int(shape[0]), int(shape[1])
    cx, cy = float(center[0]), float(center[1])
    if not (np.isfinite(cx) and np.isfinite(cy)):
        raise ValueError(f"0 级坐标非有限: {center!r}")
    radius = min(cx, cy, (width - 1) - cx, (height - 1) - cy) - float(margin_px)
    if radius <= 0.0:
        return 0, 0
    raw = int(2 * int(math.floor(radius)))
    step = max(int(step_px), 1)
    return raw, int((raw // step) * step)


def side_scan_list(side_min: int, side_max: int, step: int) -> tuple[int, ...]:
    """边长扫描点: ``side_min`` 起, 步长 ``step``, 不超过 ``side_max``。

    Raises:
        ValueError: 步长非正, 或区间内一个点都没有 (那时应当先调 ``--side-max`` /
            ``--side-min``, 而不是让一次空扫描静默产生 0 行)。
    """
    step = int(step)
    if step <= 0:
        raise ValueError(f"--side-step 必须为正, 收到 {step}")
    side_min, side_max = int(side_min), int(side_max)
    if side_min <= 0:
        raise ValueError(f"--side-min 必须为正, 收到 {side_min}")
    if side_max < side_min:
        raise ValueError(
            f"最大边长 {side_max} px 小于 --side-min {side_min} px: "
            "0 级离画面边缘太近 (或 --edge-margin-px 太大), 请调小 --side-min"
        )
    return tuple(range(side_min, side_max + 1, step))


class NoNearSideError(LookupError):
    """近 1000 的区间里一个扫描点都没有。

    🔴 **必须报错, 不能退化成全局最优**: 退化的结果是一份「清单说接近 1000, 实际
    是别处的边长」的产物, 而第 2 段的 Zernike 拟合与第 3 段的光栅扫描都会继承那个
    错误的相位, 从数据本身再也看不出来。
    """


def select_near_side(
    rows: list[dict[str, Any]],
    *,
    target_px: float,
    tolerance_px: float,
    key: str = "score",
) -> dict[str, Any]:
    """在 ``[target-tol, target+tol]`` 内取 ``key`` 最大的那一行。

    失败行 (``key`` 非有限) 与区间外的行一律跳过。

    Args:
        rows: 第 1 段的测量行 (含 ``side_px`` 与 ``key``)。
        target_px: 目标边长 (相机 px), 默认 1000。
        tolerance_px: 半宽 (相机 px)。
        key: 排序依据 (``score`` = 复合分数, 或 ``quality``)。

    Returns:
        命中的那一行 (原 dict, 不改)。

    Raises:
        NoNearSideError: 区间内没有任何可用行 —— 异常消息里带上实际边长列表。
    """
    lo = float(target_px) - abs(float(tolerance_px))
    hi = float(target_px) + abs(float(tolerance_px))
    sides = [float(r["side_px"]) for r in rows if "side_px" in r]
    usable = [
        r
        for r in rows
        if "side_px" in r and lo <= float(r["side_px"]) <= hi
        and np.isfinite(float(r.get(key, float("nan"))))
    ]
    if not usable:
        raise NoNearSideError(
            f"边长区间 [{lo:.0f}, {hi:.0f}] px 内没有任何可用扫描点 "
            f"(依据 {key}); 实际边长列表 = {sorted(int(s) for s in sides)}。"
            "请调整 --side-min / --side-step / --near-tolerance, "
            "或直接给 --side-px 指定边长"
        )
    return max(usable, key=lambda r: float(r[key]))


def parse_orders_spec(text: str) -> tuple[int, ...]:
    """解析 Zernike 阶数规格: ``"7..30"`` / ``"7-30"`` / ``"7,9,11"`` / 混合。

    区间语法 (``A..B`` 或 ``A-B``, 闭区间, 升序展开) 是本脚本新加的 ——
    ``run_gs_refine_zernike_sweep.parse_orders`` 只认逗号列表, 而这里默认要 24 个
    连续阶, 逐个写出来既难读又容易漏。**阶数不是 Zernike 数学**, 所以这里自己解析
    不构成重复实现。

    Args:
        text: 规格串。

    Returns:
        去重后按**降序**返回的阶数元组 (先测高阶: 若高阶就把整轮预算花光, 至少
        低阶那些更有用的点已经落盘了)。

    Raises:
        ValueError: 串里有非整数, 或结果是空的 / 阶数超过 30。
    """
    orders: list[int] = []
    for chunk in str(text).replace(";", ",").split(","):
        token = chunk.strip()
        if not token:
            continue
        sep = ".." if ".." in token else ("-" if "-" in token.strip("-") else None)
        if sep is None:
            orders.append(int(token))
            continue
        left, _, right = token.partition(sep)
        first, last = int(left.strip()), int(right.strip())
        if last < first:
            first, last = last, first
        orders.extend(range(first, last + 1))
    if not orders:
        raise ValueError("--zernike-orders 不能为空")
    bad = [n for n in orders if n < 1 or n > MAX_ZERNIKE_ORDER]
    if bad:
        raise ValueError(
            f"--zernike-orders 支持 1..30 阶, 越界: {sorted(set(bad))}。"
            "更高阶在 256 的拟合网格上已不可分辨 (每径向周期 <4 个样本)"
        )
    return tuple(sorted(set(orders), reverse=True))


def modes_for_order(n_max: int) -> int:
    """径向阶 ``n_max`` 覆盖的模式数 (= ``(n+1)(n+2)/2``, 含 Noll 1 piston)。

    委托 :func:`~ao_shaping.utils.wavefront.zernike_calc.calc_n_zernike_terms` ——
    那是仓库里「阶数 → 模式数」的唯一实现, 本函数只是给它一个语义化名字并在此
    断言它等于 Noll 序模式列表的长度 (回归保护: 两套索引一旦漂移, 这里先炸)。
    """
    n_terms = int(calc_n_zernike_terms(int(n_max)))
    if n_terms != (int(n_max) + 1) * (int(n_max) + 2) // 2:
        raise AssertionError(f"calc_n_zernike_terms({n_max}) = {n_terms} 不符合 (n+1)(n+2)/2")
    return n_terms


# ---------------------------------------------------------------------------
# GS 方形光斑 → 面板相位
# ---------------------------------------------------------------------------
def disc_source_amplitude(n_grid: int) -> np.ndarray:
    """GS 源面幅度: 覆盖网格中央的圆盘 (半径 ``n_grid/2``, 留半格边)。

    真实的照明是圆盘 (半径 ``--beam-radius-px`` 面板 px), GS 网格正好跨过它的包围
    方框, 所以圆盘在网格上就是内切圆。**不用**高斯: 实验系统的真实照明剖面没测过,
    均匀圆盘是中性选择 (也不引入任何需要标定的参数)。
    """
    n = int(n_grid)
    yy, xx = np.ogrid[:n, :n]
    centre = 0.5 * (n - 1)
    radius = np.hypot(yy - centre, xx - centre)
    return (radius <= 0.5 * n - 1.0).astype(np.float64)


def square_target_amplitude(n_grid: int, side_px: float) -> np.ndarray:
    """GS 目标幅度: 居中的实心方斑 (边长取整, 夹到 ``[2, n_grid]``)。

    🔴 **幅度标度无关**: 交替投影每一步只取 ``angle``, 而 ``angle`` 对正的实因子
    不变 —— 所以把目标幅度整体乘一个常数得到的**相位逐位相同**。因此这里无需像
    :class:`~ao_shaping.algorithm.signal_processing.square_shaping.UniformSquareShaper`
    那样把目标能量重标到瞳面能量: 那个重标只在**幅度**参与判据时才必要。
    """
    n = int(n_grid)
    side = int(min(max(int(round(float(side_px))), 2), n))
    target = np.zeros((n, n), dtype=np.float64)
    start = (n - side) // 2
    target[start : start + side, start : start + side] = 1.0
    return np.sqrt(target)


def run_gs_square(
    side_cam_px: float,
    bench: Bench,
    *,
    n_grid: int,
    iterations: int,
) -> GSResult:
    """为一个边长跑一次 GS, 返回 :class:`GSResult` (``phase`` 是 raw 弧度)。

    ``propagation="fft"``: 2f 实验系统的焦面就是源面的傅里叶变换 (透镜相位 +
    传播到 ``z = f``), 用单次 FFT 的夫琅禾费模型才对得上光路; ``"asm"`` 会把传播
    距离也算进去, 与 2f 几何不符。

    ``cell_spacing`` 取「包围方框的物理宽度 / 网格边长」: 网格跨过
    ``pupil_side_px`` 个面板像素, 所以有效像素间距是 ``pupil_side_px·d/N``。这样
    远场一格恒等于 ``TILT_SHIFT_SCALE/pupil_side_px`` 相机像素 (见模块 docstring)。

    Args:
        side_cam_px: 目标方斑边长 (相机 px)。
        bench: 几何常量。
        n_grid: GS 网格边长 (样本)。
        iterations: GS 迭代次数。

    Returns:
        :class:`GSResult`; ``phase`` 形状 ``(n_grid, n_grid)``, ``amplitude`` 是同一
        网格上的远场幅度 (可用于算出模型侧的 PIB/CV)。
    """
    n = int(n_grid)
    cell_spacing = bench.pupil_side_px * SLM_PITCH_M / float(n)
    return gerchberg_saxton(
        source_amplitude=disc_source_amplitude(n),
        target_amplitude=square_target_amplitude(n, bench.side_cam_to_gs(side_cam_px)),
        iterations=int(iterations),
        cell_spacing=float(cell_spacing),
        # fft 模式下 distance 不参与传播, 但函数要求 > 0; 传真实焦距, 免得读的人
        # 以为它是个可以随便填的占位符。
        distance=float(bench.focal_m),
        wavelength=bench.wavelength_nm * 1e-9,
        propagation="fft",
    )


def pupil_phase_to_panel(pupil_phase: np.ndarray, bench: Bench) -> np.ndarray:
    """模型网格相位 → 面板分辨率相位 (raw 弧度, 未包裹)。

    两步都是仓库既有实现: ``upsample_bilinear_to_shape`` 把网格拉到照明方框的
    边长, ``embed_pupil_square`` 再把方框嵌进面板 (框外恒为 0 —— 那里没有光, 花
    自由度在它上面只会引入逐像素的灰度跳变)。
    """
    box = upsample_bilinear_to_shape(
        np.asarray(pupil_phase, dtype=np.float64),
        (bench.pupil_side_px, bench.pupil_side_px),
    )
    return embed_pupil_square(
        box,
        bench.panel_shape,
        bench.beam_radius_px,
        bench.beam_center_panel,
    )


# ---------------------------------------------------------------------------
# 指标 (全部复用既有实现)
# ---------------------------------------------------------------------------
def box_centroid(
    frame: np.ndarray, center: tuple[float, float], side_px: float
) -> tuple[float, float]:
    """**方框内**的强度质心 (相机坐标)。

    为什么不能在整帧上算: 2f 光路上整帧最亮的是 0 级 (或其晕斑), 它的位置固定,
    与「方斑被光栅挪到哪里」无关。所以质心必须在框内算。
    ``crop_roi`` + ``centroid`` 都是既有实现 (前者把框外的像素整块丢掉, 后者是
    ``scipy.ndimage.center_of_mass`` 的一层包装)。
    """
    half = max(int(round(float(side_px))) // 2, 1)
    patch = crop_roi(frame, (int(round(center[0])), int(round(center[1]))), half)
    if patch.size == 0 or float(patch.max()) <= 0.0:
        return float("nan"), float("nan")
    mx, my = centroid(patch, return_float=True)
    return float(center[0]) - half + float(mx), float(center[1]) - half + float(my)


def measure_box(
    frame: np.ndarray,
    center: tuple[float, float],
    side_px: float,
) -> dict[str, float]:
    """一帧在给定方框上的全套指标 (raw 帧进, 指标出)。

    口径 (:mod:`...square_shaping`, 与 ``compare_sim_vs_hw`` 同一批函数):

    * ``cv`` 框内变异系数 (越低越好);
    * ``pib`` 框内功率占比 (整帧为分母) —— **必须**在
      :func:`~ao_shaping.utils.image.beam_metrics.prepare_frame` 之后算, 否则读出
      噪声会把比值顶到 1 以上;
    * ``score`` = ``0.5·PIB + 0.5/(1+CV)``;
    * ``ee`` / ``ar`` / ``quality`` 走方形整形优化器那一套
      (:mod:`ao_shaping.optimizer.wfless.slm_square_shaping`), **惰性 import**:
      该模块顶层 import 了 ``ao_shaping.drivers`` 与 ``ao_shaping.display``
      (牵入 pygame), 顶层导入会破坏 dry-run 的零驱动包保证。

    ⚠️ ``ee`` 与 ``pib`` 在 ``prepare_frame`` 之后是**同一个量** (都是「框内能量占
    整帧的比例」, 只是两个模块各实现了一次)。保留两个字段是为了各自与自己的消费者
    对得上: ``pib`` 与 ``compare_sim_vs_hw`` 那张对照表同源, ``ee`` 是
    ``square_quality_score`` 的入参。两条路径都在**已减背景**的帧上算 —— 在 raw 帧
    上算这类比值会被整流出来的读出噪声底座主导 (实测 PIB 顶到 1.012)。

    Args:
        frame: 原始帧 (未减背景)。
        center: 方框中心 ``(x, y)``, 相机坐标。
        side_px: 方框边长 (相机 px)。

    Returns:
        指标字典 (全部 float)。框完全落在帧外时全部为 NaN。
    """
    raw = np.asarray(frame, dtype=np.float64)
    side = int(round(float(side_px)))
    prepared = prepare_frame(raw)
    cx, cy = float(center[0]), float(center[1])
    mask = square_mask(raw.shape, int(round(cy)), int(round(cx)), side)
    inside = int(mask.sum())
    if inside < MIN_MASK_PIXELS:
        nan = float("nan")
        return {
            "cv": nan, "pib": nan, "ee": nan, "ar": nan, "score": nan,
            "quality": nan, "frame_peak": float(raw.max()),
            "box_peak": nan, "box_area_px": float(inside),
            "box_clipped_px": float(max(side * side - inside, 0)),
            "box_centroid_x": nan, "box_centroid_y": nan,
            "box_centroid_offset_x": nan, "box_centroid_offset_y": nan,
            "global_spot_fwhm_px": nan, "global_spot_hollowness": nan,
        }

    cv = float(uniformity_cv(prepared, mask))
    pib = float(power_in_bucket(prepared, mask))

    # 惰性 import (见 docstring): 只在真正测量时付这笔开销。
    from ao_shaping.optimizer.wfless.slm_square_shaping import (
        square_aspect_ratio,
        square_encircled_energy,
        square_quality_score,
    )

    ee = float(square_encircled_energy(prepared, (int(round(cx)), int(round(cy))), side))
    ar = float(square_aspect_ratio(prepared, (int(round(cx)), int(round(cy))), side))
    mx, my = box_centroid(prepared, (cx, cy), side)
    spot = measure_spot(prepared)
    return {
        "cv": cv,
        "pib": pib,
        "ee": ee,
        "ar": ar,
        "score": float(composite_from_pib_cv(pib, cv)),
        "quality": float(square_quality_score(cv, ee, ar)),
        "frame_peak": float(raw.max()),
        "box_peak": float(prepared[mask].max()),
        "box_area_px": float(inside),
        "box_clipped_px": float(max(side * side - inside, 0)),
        "box_centroid_x": mx,
        "box_centroid_y": my,
        "box_centroid_offset_x": (mx - cx) if np.isfinite(mx) else float("nan"),
        "box_centroid_offset_y": (my - cy) if np.isfinite(my) else float("nan"),
        "global_spot_fwhm_px": float(spot.fwhm_px),
        "global_spot_hollowness": float(spot.hollowness),
    }


def model_box_metrics(result: GSResult, side_cam_px: float, bench: Bench) -> dict[str, float]:
    """GS 物理预测模型侧的同一组指标 (``model_*``), 供与实测对照。

    ``GSResult.amplitude`` 是 GS 自己在**模型网格**上的远场幅度, 所以这一组数字
    不花额外 FFT, 也不需要重新实现物理预测模型。它与实测的差正是「模型 vs 实验系统」
    的全部内容 (模型不知道 LUT、面板相差、装调)。

    纳入 ``model_*`` 的只有 ``cv/pib/score``: ``ee`` (方形整形优化器那一套) 需要
    一帧相机量纲的图, 这里没有, 硬凑会让两栏不可比。
    """
    amp = np.asarray(result.amplitude, dtype=np.float64)
    n = int(amp.shape[0])
    side = int(min(max(int(round(bench.side_cam_to_gs(side_cam_px))), 2), n))
    mask = square_mask(amp.shape, n // 2, n // 2, side)
    intensity = amp**2
    if int(mask.sum()) < MIN_MASK_PIXELS or float(intensity.sum()) <= 0.0:
        return {"model_cv": float("nan"), "model_pib": float("nan"),
                "model_score": float("nan"), "model_side_gs_px": float(side)}
    cv = float(uniformity_cv(intensity, mask))
    pib = float(power_in_bucket(intensity, mask))
    return {
        "model_cv": cv,
        "model_pib": pib,
        "model_score": float(composite_from_pib_cv(pib, cv)),
        "model_side_gs_px": float(side),
    }


# ---------------------------------------------------------------------------
# Zernike 拟合 / 截断重构
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ZernikeFit:
    """一次 Zernike 拟合的结果 (全部 raw 弧度 / Noll 序)。"""

    coefficients: np.ndarray
    """Noll 序系数, 长度 = ``modes_for_order(max_order)``; 下标 i ↔ Noll j = i+1。"""

    fit_grid: int
    max_order: int
    reference: np.ndarray
    """拟合用的参考相位 (网格 ``fit_grid x fit_grid``, 已下采样)。"""

    mask: np.ndarray
    """单位圆内的拟合掩码 (``ZernikeGenerator.mask``), 统计残差时只用它。"""


def fit_zernike_pupil(
    pupil_phase: np.ndarray,
    *,
    fit_grid: int,
    max_order: int,
    generator: ZernikeGenerator | None = None,
) -> ZernikeFit:
    """把 GS 网格上的相位拟合成 Zernike 系数 (Noll 1976)。

    半径取 ``fit_grid/2``: 单位圆就是照明圆盘, 与 ``ZernikeGenerator`` 的坐标归一
    化约定一致 (``_build_cached_grid`` 按 ``(pixel - centre)/radius`` 归一)。

    ⚠️ **拟合网格必须下采样**: ``n_orders=30`` 的 ``RZern`` 表是
    ``G²·496·8`` 字节, G=900 时是 3.2 GB (见 :data:`DEFAULT_ZERNIKE_FIT_GRID`)。
    下采样用仓库的 :func:`upsample_bilinear_to_shape` (同一函数两个方向都用), 代价
    是丢掉比 ``pupil_side/fit_grid`` 个面板像素更细的相位结构 —— 那些结构在远场上
    落在 30 阶 Zernike 的表示能力之外, 拟合本来也留不住它们。

    Args:
        pupil_phase: GS 网格相位 (raw 弧度, 方形)。
        fit_grid: 拟合网格边长。
        max_order: 最高径向阶 (决定系数个数)。
        generator: 可选的已建生成器 (原子模式下拟合与重构共用一张 RZern 表, 省掉
            一次约 260 MB 的表构建)。

    Returns:
        :class:`ZernikeFit`。
    """
    grid = int(fit_grid)
    if grid < 32:
        raise ValueError(f"--zernike-fit-grid 太小 ({grid}); 至少 32")
    n_orders = int(max_order)
    reference = upsample_bilinear_to_shape(
        np.asarray(pupil_phase, dtype=np.float64), (grid, grid)
    )
    gen = generator or ZernikeGenerator((grid, grid), radius=grid / 2.0, n_orders=n_orders)
    coefficients = np.asarray(gen.fit(reference), dtype=np.float64)
    expected = modes_for_order(n_orders)
    if coefficients.size < expected:
        raise AssertionError(
            f"拟合出 {coefficients.size} 个系数, 但 {n_orders} 阶应有 {expected} 个"
        )
    return ZernikeFit(
        coefficients=coefficients,
        fit_grid=grid,
        max_order=n_orders,
        reference=reference,
        mask=np.asarray(gen.mask, dtype=bool),
    )


def reconstruct_from_fit(
    fit: ZernikeFit,
    n_max: int,
    *,
    generator: ZernikeGenerator | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """只用**前 N 阶** (Noll 1..K) 重构相位, 并给出相位域的拟合残差。

    重构走 :meth:`ZernikeGenerator.generate_noll`, 系数数组截断到
    ``K = modes_for_order(n_max)``、其余补 0 —— 于是「第 N 阶重构」严格是「把高阶
    系数置零」, 不是另一个模型的解。

    Args:
        fit: :func:`fit_zernike_pupil` 的结果。
        n_max: 本次保留的最高阶。
        generator: 可选的已建生成器 (逐个阶数扫描时复用, 省掉重建 RZern 表的开销)。

    Returns:
        ``(重构相位 (fit_grid 网格, 圆外已被置 0), 统计量)``。统计量含
        ``zernike_modes_used`` / ``zernike_rms_residual_rad`` /
        ``zernike_explained_fraction``。
    """
    used = modes_for_order(int(n_max))
    gen = generator or ZernikeGenerator(
        (fit.fit_grid, fit.fit_grid),
        radius=fit.fit_grid / 2.0,
        n_orders=fit.max_order,
    )
    coeffs = np.zeros(gen.n_modes, dtype=np.float64)
    take = min(int(used), int(fit.coefficients.size), int(gen.n_modes))
    coeffs[:take] = fit.coefficients[:take]
    phase = np.asarray(gen.generate_noll(coeffs), dtype=np.float64)
    # generate_noll 在单位圆外返回 NaN (RZern 的 make_cart_grid 把 rho>1 设成 NaN);
    # 那里没有光, 置 0 与 embed_pupil_square 的框外行为一致。
    phase = np.where(np.isfinite(phase), phase, 0.0)

    mask = fit.mask
    residual = (phase - fit.reference)[mask]
    ref = fit.reference[mask]
    rms_residual = float(np.sqrt(np.mean(residual**2))) if residual.size else float("nan")
    total = float(np.sum((ref - ref.mean()) ** 2))
    explained = float(1.0 - np.sum(residual**2) / total) if total > 0 else float("nan")
    return phase, {
        "zernike_modes_used": float(take),
        "zernike_rms_residual_rad": rms_residual,
        "zernike_explained_fraction": explained,
    }


# ---------------------------------------------------------------------------
# Zernike 拟合缓存 (原子模式下每个配置一个进程, 拟合结果要跨进程复用)
# ---------------------------------------------------------------------------
def fit_cache_dir(
    run_root: Path, *, side_px: int, fit_grid: int, max_order: int
) -> Path:
    """一次拟合的缓存目录: ``<run>/zernike/_fit/side<S>_g<G>_o<O>/``。

    键里带上边长与网格/阶数: 换了边长就是另一个基准相位, 换了网格或阶数就是另一组
    系数 —— 任何一项不同都不能复用同一份缓存。
    """
    return (
        Path(run_root)
        / "zernike"
        / "_fit"
        / f"side{int(side_px)}_g{int(fit_grid)}_o{int(max_order)}"
    )


def load_or_compute_fit(
    run_root: Path,
    pupil_phase: np.ndarray,
    *,
    side_px: int,
    fit_grid: int,
    max_order: int,
) -> tuple[ZernikeFit, str, dict[str, Any]]:
    """取该 (边长, 网格, 阶数) 的 Zernike 拟合: 有缓存就用缓存, 否则算一次并落盘。

    拟合是确定性的 (最小二乘 + 固定的 RZern 网格), 所以缓存与重算得到同一组系数 ——
    缓存只省时间, 不改结果。每次原子进程都要一次拟合, 不复用的话 24 个阶数就要拟 24 次。

    Returns:
        ``(拟合, "cache"|"recomputed", 诊断信息)``。
    """
    cache = fit_cache_dir(
        run_root, side_px=side_px, fit_grid=fit_grid, max_order=max_order
    )
    coeff_path = cache / "zernike_coefficients.npy"
    meta_path = cache / "fit_meta.json"
    info: dict[str, Any] = {"fit_cache_dir": str(cache)}
    generator = ZernikeGenerator(
        (int(fit_grid), int(fit_grid)), radius=float(fit_grid) / 2.0, n_orders=int(max_order)
    )
    expected = modes_for_order(int(max_order))
    cached = None
    if coeff_path.is_file() and meta_path.is_file():
        try:
            cached = np.asarray(np.load(coeff_path), dtype=np.float64)
        except (OSError, ValueError) as exc:
            info["cache_reason"] = f"系数读不出来: {exc}"
            cached = None
    if cached is not None:
        if cached.size < expected:
            info["cache_reason"] = f"缓存只有 {cached.size} 个系数, 需要 {expected}"
            cached = None
        elif not np.isfinite(cached).all():
            info["cache_reason"] = "缓存含非有限值"
            cached = None
    if cached is not None:
        reference = upsample_bilinear_to_shape(
            np.asarray(pupil_phase, dtype=np.float64), (int(fit_grid), int(fit_grid))
        )
        info["loaded"] = str(coeff_path)
        return (
            ZernikeFit(
                coefficients=cached,
                fit_grid=int(fit_grid),
                max_order=int(max_order),
                reference=reference,
                mask=np.asarray(generator.mask, dtype=bool),
            ),
            "cache",
            info,
        )

    fit = fit_zernike_pupil(
        pupil_phase,
        fit_grid=int(fit_grid),
        max_order=int(max_order),
        generator=generator,
    )
    coeff_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(coeff_path, fit.coefficients)
    write_json_atomic(
        meta_path,
        {
            "timestamp_utc": _utc_now_iso(),
            "side_px": int(side_px),
            "fit_grid": int(fit_grid),
            "max_order": int(max_order),
            "n_modes": int(fit.coefficients.size),
            "reference_rms_rad": float(
                np.sqrt(np.mean(np.square(fit.reference[fit.mask])))
                if bool(np.any(fit.mask))
                else float("nan")
            ),
        },
    )
    return fit, "recomputed", info


# ---------------------------------------------------------------------------
# 闪耀光栅
# ---------------------------------------------------------------------------
def blaze_overlay(period_px: float, wavelength_nm: float) -> OverlayParams:
    """一条面板 x 斜坡的叠加参数 (``BLAZE``, 方向 ``vertical``)。"""
    return OverlayParams(
        mode=ZeroOrderOverlay.BLAZE,
        blaze_period_px=float(period_px),
        blaze_direction="vertical",
        wavelength_nm=float(wavelength_nm),
        panel_pixel_um=SLM_PITCH_M * 1e6,
    )


def box_fits(center: tuple[float, float], side_px: float, shape: tuple[int, int]) -> bool:
    """以 ``center`` 为心、边长 ``side_px`` 的方框是否**整个**落在画面内。

    落在外面时 PIB 的分母仍取整帧而分子少了一块 —— 那种量出来的不是「方斑质量」,
    所以每一行都记这个布尔量 (``box_clipped``) 而不是悄悄算下去。
    """
    height, width = int(shape[0]), int(shape[1])
    half = float(side_px) / 2.0
    cx, cy = float(center[0]), float(center[1])
    return bool(
        0 <= cx - half and cx + half <= width - 1
        and 0 <= cy - half and cy + half <= height - 1
    )


def blaze_shift_table(
    periods: tuple[int, ...],
    wavelength_nm: float,
    shape: tuple[int, int],
    *,
    side_px: float,
    center: tuple[float, float],
) -> list[dict[str, Any]]:
    """每个周期的预测位移与「位移后的方框还装不装得进画面」。

    ``predicted_shift_px`` 已含面板↔相机 90° 交换 (面板 x 斜坡 → 相机 y)。位移之后
    方框以「``center`` + 预测位移」为心, 所以这个框必须整个落在画面内。

    Args:
        periods: 光栅周期 (面板 px)。
        wavelength_nm: 波长。
        shape: 画面 ``(height, width)``。
        side_px: 方框边长 (相机 px)。
        center: 冻结的 0 级位置 (相机 px)。

    Returns:
        每项 ``{"blaze_period_px", "predicted_shift_x_px", "predicted_shift_y_px",
        "predicted_shift_px", "box_center_predicted_x", "box_center_predicted_y",
        "fits_frame"}``。规划时 ``center`` 是计划假设; 实测时用真实值再判一遍。
    """
    out: list[dict[str, Any]] = []
    for period in periods:
        dx, dy = predicted_shift_px(blaze_overlay(period, wavelength_nm))
        magnitude = float(math.hypot(dx, dy))
        box_center = (float(center[0]) + float(dx), float(center[1]) + float(dy))
        out.append(
            {
                "blaze_period_px": int(period),
                "predicted_shift_x_px": float(dx),
                "predicted_shift_y_px": float(dy),
                "predicted_shift_px": magnitude,
                "box_center_predicted_x": box_center[0],
                "box_center_predicted_y": box_center[1],
                "fits_frame": box_fits(box_center, side_px, shape),
            }
        )
    return out


# ---------------------------------------------------------------------------
# 耗时估计
# ---------------------------------------------------------------------------
def gs_seconds(n_grid: int, iterations: int) -> float:
    """一次 GS 的预计耗时 (s) —— 模型见 :data:`GS_SECONDS_PER_ITER_UNIT`。"""
    per_iter = GS_SECONDS_PER_ITER_UNIT * float(n_grid) ** 2 * math.log2(max(n_grid, 2))
    return float(iterations) * per_iter


def estimate_stage_seconds(
    n_configs: int,
    *,
    gs_seconds_each: float,
    capture_s: float,
) -> float:
    """一段的预计耗时: 每配置 (GS 可选) + 采集。"""
    return float(n_configs) * (float(gs_seconds_each) + float(capture_s))


# ---------------------------------------------------------------------------
# 累积产物: 路径 / 原子写 / 按配置去重 (跨进程)
# ---------------------------------------------------------------------------
#: 一次写入过程留下的临时文件后缀 (含 pid, 所以并发进程不会互相覆盖)。
_TMP_SUFFIX = ".tmp"


def _abs_path(text: str | Path) -> Path:
    """相对路径按**仓库根**解析 (与脚本自身 ``ROOT`` 一致, 不依赖 cwd)。"""
    path = Path(text)
    return path if path.is_absolute() else (ROOT / path)


def _fail(message: str, code: int) -> SystemExit:
    """打印一条失败说明, 返回**带数字退出码**的 ``SystemExit``。

    ``raise SystemExit("消息")`` 的 ``code`` 是字符串, 进程会退成 1 —— 本脚本用不同
    退出码区分处置 (5 = 窗口里没点, 6 = 缺选点, 7 = 没有可用测量), 所以消息必须单独
    打印, 码必须是 int, 否则操作者拿到的永远是 1。
    """
    sys.stderr.write(message.rstrip() + "\n")
    sys.stderr.flush()
    return SystemExit(int(code))


def run_root_from_args(args: argparse.Namespace) -> Path:
    """``--run-root`` / ``--out-root`` → 批次根目录 (绝对路径)。

    * 两者都给: ``--run-root`` 是批次根; ``--out-root`` 只覆盖**该段目录**
      (见 :func:`stage_dir_from_args`)。
    * 只给 ``--out-root``: 它就是批次根 —— 与旧版语义完全一致。
    * 都不给: ``<DEFAULT_OUT_PARENT>/<UTC 时间戳>``。

    原子模式**必须**显式给 ``--run-root`` (或 ``--out-root``): 每个进程各自取一个
    时间戳会让「同一批」散在多个目录里, 累积就无从谈起。这条由
    :func:`require_accumulation_root` 在开设备之前检查。
    """
    text = getattr(args, "run_root", None) or getattr(args, "out_root", None)
    if not text:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        text = f"{DEFAULT_OUT_PARENT}/{stamp}"
    return _abs_path(str(text))


def require_accumulation_root(args: argparse.Namespace) -> Path:
    """原子 / 离线模式的批次根: 必须由操作者显式给出, 不给就报错。

    一个原子进程只跑一组参数, 「这批属于哪一轮」不能靠时间戳猜 —— 猜错的后果是
    第 2 段读不到第 1 段的选点 (或者读到上一轮的)。
    """
    if not (getattr(args, "run_root", None) or getattr(args, "out_root", None)):
        raise _fail(
            "原子模式必须给 --run-root <dir> (批次根), 例如 --run-root "
            f"{DEFAULT_OUT_PARENT}/r1 —— 每个进程各自取时间戳会让同一批散在多个目录里",
            2,
        )
    return run_root_from_args(args)


def stage_dir_from_args(args: argparse.Namespace, run_root: Path, stage: str) -> Path:
    """该段产物目录: ``<run-root>/<stage>``, 或 ``--out-root`` 的显式覆盖。

    覆盖只在 ``--run-root`` **同时**给出时生效 —— 那时 ``--out-root`` 的含义从
    「批次根」变成「这一段放哪」, 否则旧调用 (只给 ``--out-root``) 会静默改语义。
    """
    out_root = getattr(args, "out_root", None)
    if out_root and getattr(args, "run_root", None):
        return _abs_path(str(out_root))
    return Path(run_root) / stage


def config_slug(stage: str, value: int | float) -> str:
    """一组参数的目录名: ``side_700`` / ``nmax_11`` / ``blaze_32``。

    同一个 slug = 同一个配置 —— CSV/manifest 的去重键就是它, 所以重跑同一配置
    只会就地替换自己。
    """
    key = str(stage).strip().lower()
    if key not in STAGE_SLUG_PREFIX:
        raise ValueError(f"未知的段 {stage!r}; 只接受 {tuple(STAGE_SLUG_PREFIX)}")
    return f"{STAGE_SLUG_PREFIX[key]}_{int(round(float(value)))}"


def config_dir(stage_dir: Path, stage: str, value: int | float) -> Path:
    """一组参数的产物目录 ``<stage-dir>/<config-slug>`` (不保证已存在)。"""
    return Path(stage_dir) / config_slug(stage, value)


def write_text_atomic(path: Path, text: str) -> None:
    """先写同目录临时文件再 ``os.replace`` —— Ctrl-C 不会留下半个文件。

    Windows 上 ``os.replace`` 对同目录内的目标文件是原子的 (覆盖已存在文件也行),
    所以读方永远只看到「旧内容」或「新内容」, 不会看到半截 JSON。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}{_TMP_SUFFIX}{os.getpid()}")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def write_json_atomic(path: Path, payload: Any) -> None:
    """JSON 版的 :func:`write_text_atomic`。"""
    write_text_atomic(path, json.dumps(payload, indent=2, ensure_ascii=False))


def read_json_file(path: Path) -> Any:
    """读回 JSON; 缺失或损坏返回 ``None`` (索引不该让聚合跑不下去)。"""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def as_float(value: Any, default: float = float("nan")) -> float:
    """CSV 里的字符串 → float; 空串 / 非数字 → ``default``。

    ``measurements.csv`` 是跨进程的累积件, 读回来**一律是字符串**; 选点与排序都
    要数值, 所以这个转换在读取边界做一次, 不让它散到各处。
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def flatten_row(row: dict[str, Any]) -> dict[str, Any]:
    """只保留标量字段 —— CSV 一行就是一行, 嵌套的 ``preflight`` 留在 metrics.json。

    ``metrics.json`` 是这一组的完整记录 (含 preflight / 嵌套结构), CSV 只是它的
    扁平投影 (供离线出图与人工查看)。
    """
    flat: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, (dict, list, tuple, set)):
            continue
        if isinstance(value, (str, bool, int, float)) or value is None:
            flat[key] = value
        elif hasattr(value, "item"):  # numpy 标量
            flat[key] = value.item()
        else:
            flat[key] = str(value)
    return flat


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """读累积 CSV (每行一个 dict); 文件不存在时返回空列表。"""
    path = Path(path)
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """把行写成 CSV (列名 = 所有行的并集, 按首次出现顺序), 原子替换。

    列名取并集是必需的: 失败行多一个 ``error`` 列, 而 ``DictWriter`` 只认第一行的
    列名, 照旧写法会在写失败行时抛 ``ValueError``。
    """
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}{_TMP_SUFFIX}{os.getpid()}")
    try:
        with tmp.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, restval="")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def _row_sort_key(row: dict[str, Any], key: str) -> tuple[int, float, str]:
    """按自变量排序, 非有限值排最后 (NaN 不能参与比较, 否则顺序随平台漂)。"""
    value = as_float(row.get(key))
    return (0, value, str(row.get(CONFIG_KEY, ""))) if math.isfinite(value) else (
        1,
        0.0,
        str(row.get(CONFIG_KEY, "")),
    )


def upsert_measurements(
    stage_dir: Path,
    row: dict[str, Any],
    *,
    stage: str,
) -> list[dict[str, Any]]:
    """把一行写进该段的 ``measurements.csv`` / ``measurements.json`` (按 slug 去重)。

    **可重入**: 同一个 ``config`` 只在原位替换, 别的配置不动; 排序按自变量数值
    (不是写入顺序), 所以无论进程按什么顺序跑完, 最终 CSV 都是同一个样子。

    ``measurements.json`` 写**完整记录** (从各配置的 ``metrics.json`` 读回, 类型保留),
    CSV 只写扁平投影 —— 两者是同一份数据的两层视图, 不是两份事实。

    Returns:
        合并后的全部行 (已排序; 来自 CSV 的扁平视图)。
    """
    stage_dir = Path(stage_dir)
    order_key = STAGE_VARIABLES[stage]
    csv_path = stage_dir / CONFIG_CSV_NAME
    existing = read_csv_rows(csv_path)
    flat = flatten_row(row)
    slug = str(flat.get(CONFIG_KEY, ""))
    if not slug:
        raise ValueError(f"测量行缺少 {CONFIG_KEY!r} 字段: {sorted(flat)}")
    merged: list[dict[str, Any]] = [
        dict(r) for r in existing if str(r.get(CONFIG_KEY, "")) != slug
    ]
    merged.append(flat)
    merged.sort(key=lambda r: _row_sort_key(r, order_key))
    write_csv_rows(csv_path, merged)
    typed = load_stage_rows(stage_dir, stage)
    write_json_atomic(stage_dir / CONFIG_JSON_NAME, typed if typed else merged)
    return merged


def update_manifest_atomic(root: Path, entry: dict[str, Any]) -> list[dict[str, Any]]:
    """把一条段条目并入根 ``manifest.json``, 原子落盘, 同名段就地替换。

    复用 :mod:`scripts._common.group_manifest` 的条目形状与容错读取
    (:func:`~scripts._common.group_manifest.load_manifest` 读不出来时返回空列表),
    但写入换成「临时文件 + ``os.replace``」—— 那个模块的 ``update_manifest`` 是
    直接 ``write_text``, 被 Ctrl-C 打断会留下半截清单。
    """
    root = Path(root)
    merged: list[dict[str, Any]] = []
    group = str(entry.get("group", ""))
    replaced = False
    for existing in load_manifest(root):
        if str(existing.get("group", "")) == group:
            merged.append(dict(entry))
            replaced = True
        else:
            merged.append(existing)
    if not replaced:
        merged.append(dict(entry))
    write_json_atomic(root / MANIFEST_NAME, merged)
    return merged


# ---------------------------------------------------------------------------
# 选点文件 (第 1 段 → 第 2/3 段 的唯一输入)
# ---------------------------------------------------------------------------
class SelectionMissing(RuntimeError):
    """需要选点却没有 ``selected.json`` (或它内容不可用)。"""


@dataclass(frozen=True)
class Selection:
    """第 2/3 段用到的「选定边长 + 基准相位」出处。"""

    side_px: int
    source: str
    """``"selected.json"`` (跑过 ``--scan select``) 或 ``"cli"`` (``--side-px`` 显式)。"""

    phase_path: Path | None
    """``selected.json`` 指出的基准相位文件 (``pupil_phase.npy``); ``cli`` 时为 ``None``。"""

    gs_grid: int | None
    gs_iters: int | None
    rank_by: str | None
    score: float | None


def selection_path(run_root: Path) -> Path:
    """``<run-root>/sides/selected.json``。"""
    return Path(run_root) / "sides" / SELECTED_NAME


def save_selection(run_root: Path, payload: dict[str, Any]) -> Path:
    """写 ``selected.json`` (原子), 返回路径。"""
    path = selection_path(run_root)
    write_json_atomic(path, payload)
    return path


def load_selection(run_root: Path, *, side_px_override: int | None = None) -> Selection:
    """读 ``selected.json``; 显式 ``--side-px`` 优先 (旧行为的逃生口)。

    Raises:
        SelectionMissing: 没有 ``--side-px`` 且 ``selected.json`` 缺失 / 不可用。
            消息里点名 ``--scan select`` —— 那时的处置是「先选点」, 不是调参数。
    """
    if side_px_override is not None:
        return Selection(
            side_px=int(side_px_override),
            source="cli",
            phase_path=None,
            gs_grid=None,
            gs_iters=None,
            rank_by=None,
            score=None,
        )
    path = selection_path(run_root)
    hint = (
        "请先跑选点 (它读 <run-root>/sides/measurements.csv, 按 --rank-by 在 "
        f"near-{DEFAULT_NEAR_TARGET_PX:.0f} 窗口内取点):\n"
        "  python scripts/slm/hw_gs_square_scans.py --scan select "
        f"--run-root {run_root}"
    )
    if not path.is_file():
        raise SelectionMissing(f"第 2/3 段需要选点, 但 {path} 不存在。{hint}")
    data = read_json_file(path)
    if not isinstance(data, dict):
        raise SelectionMissing(f"{path} 读不出来 (不是 JSON 对象)。{hint}")
    try:
        side_px = int(data["side_px"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SelectionMissing(f"{path} 里没有可用的 side_px 字段: {data!r}。{hint}") from exc
    if side_px <= 0:
        raise SelectionMissing(f"{path} 里的 side_px={side_px} 不是正边长。{hint}")
    raw_phase = data.get("phase_path")
    phase_path = None
    if raw_phase:
        candidate = Path(str(raw_phase))
        if candidate.is_absolute():
            phase_path = candidate
        else:
            # selected.json 的 phase_path 是**相对 run-root** 写的 (可移植); 老文件
            # 若是相对仓库根写的, 那两个位置里总有一个存在。
            from_run_root = Path(run_root) / candidate
            phase_path = from_run_root if from_run_root.exists() else _abs_path(candidate)
    return Selection(
        side_px=side_px,
        source="selected.json",
        phase_path=phase_path,
        gs_grid=(None if data.get("gs_grid") in (None, "") else int(data["gs_grid"])),
        gs_iters=(
            None if data.get("gs_iters") in (None, "") else int(data["gs_iters"])
        ),
        rank_by=(None if data.get("rank_by") in (None, "") else str(data["rank_by"])),
        score=(
            None
            if data.get("score") in (None, "")
            else as_float(data.get("score"), float("nan"))
        ),
    )


def load_cached_pupil_phase(
    run_root: Path,
    side_px: int,
    *,
    gs_grid: int | None = None,
    phase_path: Path | None = None,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """复用该边长已存盘的 **pupil 网格** GS 相位 (省掉每次约 7.5 s 的 GS)。

    只在**形状与本次网格相符**时复用一个像素; 不符就返回 ``None`` 让调用方重算 ——
    拿形状不对的相位去拟合等于换了实验条件却沿用旧数据。

    Returns:
        ``(相位 或 None, 诊断信息)``。
    """
    side_dir = Path(run_root) / "sides" / config_slug("sides", side_px)
    info: dict[str, Any] = {"side_dir": str(side_dir)}
    candidates: list[Path] = []
    if phase_path is not None:
        candidates.append(Path(phase_path))
    candidates.append(side_dir / CONFIG_PUPIL_PHASE_NAME)
    for candidate in candidates:
        info["candidate"] = str(candidate)
        if not candidate.is_file():
            info["reason"] = "文件不存在"
            continue
        try:
            arr = np.asarray(np.load(candidate), dtype=np.float64)
        except (OSError, ValueError) as exc:
            info["reason"] = f"读不出来: {exc}"
            continue
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            info["reason"] = f"不是方阵: {arr.shape}"
            continue
        if gs_grid is not None and arr.shape != (int(gs_grid), int(gs_grid)):
            info["reason"] = (
                f"形状 {arr.shape} 与本次 GS 网格 {(int(gs_grid), int(gs_grid))} 不符"
            )
            continue
        if not np.isfinite(arr).all():
            info["reason"] = "含非有限值"
            continue
        info.pop("reason", None)
        info["loaded"] = str(candidate)
        return arr, info
    info.setdefault("reason", "没有候选文件")
    return None, info


# ---------------------------------------------------------------------------
# 实验计划 (dry-run 的全部输出)
# ---------------------------------------------------------------------------
def _requested_stages(scan: str) -> tuple[str, ...]:
    """``--scan`` → 实际要跑的段 (保持固定顺序)。

    ``select`` / ``summarize`` 是**离线**子命令 (不跑设备、也没有「段网格」), 它们
    在 :func:`main` 里先被截走 —— 走到这里说明调用方漏了那一步, 直接报错。
    """
    key = str(scan).strip().lower()
    if key == "all":
        return STAGES
    if key in OFFLINE_SCANS:
        raise ValueError(
            f"--scan {key} 是离线子命令, 不走实验计划 (它读已累积的产物); "
            "请直接运行, 不要与 --plan-commands 或 build_plan 混用"
        )
    if key not in STAGES:
        raise ValueError(
            f"--scan 只接受 {{sides,zernike,blaze,all}} 或离线子命令 "
            f"{{'select','summarize'}}, 收到 {scan!r}"
        )
    return (key,)


def atomic_selection(args: argparse.Namespace, stages: tuple[str, ...]) -> dict[str, Any] | None:
    """解析「原子模式」——**一个进程只跑一组参数**时返回那一组, 否则 ``None``。

    触发条件: 只跑某一段 **且** 给了该段的值选择器。

    * ``--scan sides  --side-px 700``   → ``{"stage": "sides", "value": 700}``
    * ``--scan zernike --n-max 11``     → ``{"stage": "zernike", "value": 11}``
    * ``--scan blaze  --blaze-period 32`` → ``{"stage": "blaze", "value": 32}``

    ``--scan all`` / 整段 ``--scan sides`` 不给值选择器时返回 ``None`` —— 那是旧版
    「一次跑完整段」的路径 (保留, 但只用于调试: 一次几小时, 中断就整轮作废)。
    """
    if len(stages) != 1:
        return None
    stage = stages[0]
    raw: Any
    if stage == "sides":
        raw = getattr(args, "side_px", None)
    elif stage == "zernike":
        raw = getattr(args, "n_max", None)
    else:
        raw = getattr(args, "blaze_period", None)
    if raw is None:
        return None
    value = int(raw)
    if value <= 0:
        raise _fail(
            f"--{STAGE_VALUE_ARG[stage].replace('_', '-')} 必须为正, 收到 {value}", 2
        )
    if stage == "zernike" and value > MAX_ZERNIKE_ORDER:
        raise _fail(f"--n-max 支持 1..30, 收到 {value}", 2)
    if stage == "blaze" and value < MIN_BLAZE_PERIOD_PX:
        raise _fail(
            f"--blaze-period 至少 {MIN_BLAZE_PERIOD_PX} 面板 px, 收到 {value} "
            "(周期 1 px 是棋盘格不是斜坡)",
            2,
        )
    return {"stage": stage, "value": value, "variable": STAGE_VARIABLES[stage]}


def build_plan(args: argparse.Namespace, bench: Bench) -> dict[str, Any]:
    """完整计划: 三段各扫什么、最大边长怎么来、预计多久、产物在哪。

    ``--dry-run`` 打印的就是这个结构的渲染结果; ``--execute`` 复用同一份计划
    (所以计划里写的和真跑的是同一件事)。
    """
    requested = _requested_stages(args.scan)
    stages = [s for s in STAGES if s in requested]

    # -- 画面与 0 级: dry-run 用假设值 (明确标注), 实测现场量 -----------------
    frame_shape = parse_frame_shape(args.dry_run_frame_shape)
    zero_assumed = args.dry_run_zero_order is None
    if zero_assumed:
        zero_xy = (frame_shape[1] / 2.0, frame_shape[0] / 2.0)
    else:
        zero_xy = parse_xy(args.dry_run_zero_order)

    side_max_raw, side_max = max_side_from_frame(
        frame_shape, zero_xy, margin_px=int(args.edge_margin_px), step_px=int(args.side_step)
    )
    if str(args.side_max).strip().lower() == "auto":
        side_max_used = side_max
        side_max_note = "auto (由 0 级到画面边缘反解, 再向下取到 --side-step 的整数倍)"
    else:
        try:
            requested_max = int(args.side_max)
        except (TypeError, ValueError) as exc:
            raise SystemExit(
                f"--side-max 需要 'auto' 或整数 px, 收到 {args.side_max!r}"
            ) from exc
        side_max_used = min(requested_max, side_max)
        side_max_note = (
            f"显式 {requested_max} px"
            + ("" if requested_max <= side_max else f" —— 被画面边缘限制为 {side_max} px")
        )

    try:
        sides = side_scan_list(int(args.side_min), int(side_max_used), int(args.side_step))
    except ValueError as exc:
        raise SystemExit(f"边长扫描网格不可用: {exc}") from exc

    # -- 近 1000 的候选 (开跑前就能列出来) ------------------------------------
    near_target = float(args.target_side_near)
    near_tol = float(args.near_tolerance)
    lo = near_target - near_tol
    hi = near_target + near_tol
    near_candidates = [s for s in sides if lo <= s <= hi]

    # -- GS 网格: 远场窗口必须装下整段扫描 -----------------------------------
    periods = tuple(int(p) for p in args.blaze_periods)
    bad_periods = [p for p in periods if p < MIN_BLAZE_PERIOD_PX]
    if bad_periods:
        raise SystemExit(
            f"--blaze-periods 里有周期 < {MIN_BLAZE_PERIOD_PX} 面板 px: {bad_periods} "
            "(周期 1 px 是棋盘格不是斜坡)"
        )
    # 第 3 段的方框用「选定边长」——规划时还不知道, 用近 1000 的目标值作假设。
    planning_side = float(min(max(near_target, sides[0]), sides[-1]))
    if periods:
        blaze_table = blaze_shift_table(
            periods, bench.wavelength_nm, frame_shape,
            side_px=planning_side, center=zero_xy,
        )
    else:
        blaze_table = []
    max_shift = max((float(v["predicted_shift_px"]) for v in blaze_table), default=0.0)
    span_cam = float(side_max_used) + 2.0 * max_shift
    gs_grid_auto = bench.gs_grid_for(span_cam)
    if str(args.gs_grid).strip().lower() == "auto":
        gs_grid = gs_grid_auto
        gs_grid_note = (
            f"auto: 需要装下边长 {side_max_used} + 2×光栅位移 {max_shift:.0f} = "
            f"{span_cam:.0f} camera px; 每格 {bench.camera_px_per_gs_px:.3f} camera px "
            f"⇒ N ≥ {span_cam / bench.camera_px_per_gs_px:.0f}, 取 2 的幂 {gs_grid}"
        )
    else:
        gs_grid = int(args.gs_grid)
        gs_grid_note = f"显式 {gs_grid} (auto 会取 {gs_grid_auto})"
        if gs_grid * bench.camera_px_per_gs_px < span_cam:
            raise SystemExit(
                f"--gs-grid {gs_grid} 的远场窗口只有 "
                f"{gs_grid * bench.camera_px_per_gs_px:.0f} camera px, "
                f"装不下 {span_cam:.0f} camera px (会被窗口边界卷绕); 至少 "
                f"{gs_grid_auto}"
            )

    orders = parse_orders_spec(args.zernike_orders)
    single_gs_s = gs_seconds(gs_grid, int(args.gs_iters))
    capture_s = float(args.estimate_capture_s)

    stage_plans: dict[str, dict[str, Any]] = {}
    if "sides" in stages:
        stage_plans["sides"] = {
            "variable": STAGE_VARIABLES["sides"],
            "n_configs": len(sides),
            "configs": list(sides),
            "gs_seconds_each": single_gs_s,
            "seconds": estimate_stage_seconds(
                len(sides), gs_seconds_each=single_gs_s, capture_s=capture_s
            ),
            "detail": "每个边长跑一次 GS (从零造方斑), 然后测量",
        }
    if "zernike" in stages:
        stage_plans["zernike"] = {
            "variable": STAGE_VARIABLES["zernike"],
            "n_configs": len(orders),
            "configs": list(orders),
            "gs_seconds_each": 0.0,
            "seconds": estimate_stage_seconds(
                len(orders), gs_seconds_each=0.0, capture_s=capture_s
            ),
            "detail": (
                "一次 GS (选定边长) + 一次拟合 + 每个阶数一次截断重构与测量; "
                f"拟合网格 {int(args.zernike_fit_grid)}, 最高 "
                f"{max(orders)} 阶 ({modes_for_order(max(orders))} 个模式)"
            ),
        }
    if "blaze" in stages:
        stage_plans["blaze"] = {
            "variable": STAGE_VARIABLES["blaze"],
            "n_configs": len(periods),
            "configs": list(periods),
            "gs_seconds_each": 0.0,
            "seconds": estimate_stage_seconds(
                len(periods), gs_seconds_each=0.0, capture_s=capture_s
            ),
            "detail": "选定相位上逐周期叠加面板 x 斜坡; 框跟着预测位移走",
        }

    total_s = sum(float(v["seconds"]) for v in stage_plans.values())
    over_budget = {k: v for k, v in stage_plans.items() if float(v["seconds"]) > PLAN_LIMIT_S}

    slm_driver, ccd_driver = device_driver_strings(args.cam_type)
    is_sim = str(args.cam_type).strip().lower() == "sim"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    # 批次根: --run-root 优先, 其次 --out-root (旧语义), 最后时间戳。
    run_root_text = (
        getattr(args, "run_root", None)
        or args.out_root
        or f"{DEFAULT_OUT_PARENT}/{stamp}"
    )
    atomic = atomic_selection(args, tuple(stages))

    focus_text = (
        "不加焦面补偿 (默认) —— CCD 现在就在 2f 焦面上, 旧的瞳面菲涅尔补偿已过期"
        if args.defocus_focal_mm is None
        else (
            f"瞳面菲涅尔补偿 f_v = {float(args.defocus_focal_mm):.1f} mm "
            "(pi·r²/(λ·f_v), 盘外为 0), 走 set_phase_overlay"
        )
    )

    return {
        "stages": stages,
        "atomic": atomic,
        "bench": {
            "wavelength_nm": bench.wavelength_nm,
            "focal_length_m": bench.focal_m,
            "panel_shape": list(bench.panel_shape),
            "panel_pitch_um": SLM_PITCH_M * 1e6,
            "beam_center_panel": list(bench.beam_center_panel),
            "beam_radius_px": bench.beam_radius_px,
            "pupil_side_px": bench.pupil_side_px,
            "camera_pixel_um": bench.camera_pixel_um,
        },
        "frame": {
            "shape": list(frame_shape),
            "zero_order_xy": list(zero_xy),
            "zero_order_assumed": bool(zero_assumed),
            "assumed_note": (
                "--dry-run 的计划假设 (取画幅中心); 实测一律在平场帧上 argmax 现场量"
                if zero_assumed
                else "--dry-run-zero-order 显式给定 (仅用于计划)"
            ),
            "edge_margin_px": int(args.edge_margin_px),
            "side_max_raw": int(side_max_raw),
            "side_max": int(side_max_used),
            "side_max_note": side_max_note,
        },
        "scan": {
            "gs_grid": int(gs_grid),
            "gs_grid_note": gs_grid_note,
            "gs_iters": int(args.gs_iters),
            "camera_px_per_gs_px": bench.camera_px_per_gs_px,
            "fft_window_camera_px": float(gs_grid) * bench.camera_px_per_gs_px,
            "sides": list(sides),
            "near_target": near_target,
            "near_tolerance": near_tol,
            "near_window": [lo, hi],
            "near_candidates": near_candidates,
            "orders": list(orders),
            "orders_modes": {int(n): modes_for_order(int(n)) for n in orders},
            "rank_by": str(args.rank_by),
            "side_px_override": (None if args.side_px is None else int(args.side_px)),
            "blaze_periods": list(periods),
            "blaze_table": blaze_table,
            "blaze_planning_side_px": planning_side,
            "max_blaze_shift_px": max_shift,
            "zernike_fit_grid": int(args.zernike_fit_grid),
        },
        "devices": {
            "slm_driver": slm_driver,
            "ccd_driver": ccd_driver,
            "cam_type": args.cam_type,
            "cam_id": int(args.cam_id),
            "slm_number": int(args.slm_number),
            "slm_wavelength_nm": int(bench.wavelength_nm),
            "exposure_ms_requested": float(args.exposure_ms),
            "frames_per_measurement": int(args.frames),
            "sim_seed": int(args.sim_seed),
            "sim_far_field_padding": int(args.sim_far_field_padding),
            "sim_cam_size": int(args.cam_size),
        },
        "optics_notes": {
            "gs": (
                "gerchberg_saxton(propagation='fft'): 2f 焦面 = 源面的傅里叶变换; "
                "cell_spacing = pupil_side·d/N, 所以远场一格恒为 "
                "TILT_SHIFT_SCALE/pupil_side 相机像素"
            ),
            "gs_deterministic": (
                "GS 初值 = 目标的反向传播 ⇒ 没有随机数; --seed 只影响内存槽轮换"
                "与仿真噪声, 不影响 GS 相位"
            ),
            "side_map": (
                "side_gs_px = side_cam_px · pupil_side_px / TILT_SHIFT_SCALE "
                "(与网格边长无关, 也与波长锚点无关)"
            ),
            "focus": focus_text,
            "blaze": (
                "apply_zero_order_overlay(BLAZE, vertical) 加进模式相位; "
                "预测位移 predicted_shift_px 已含面板↔相机 90° 交换"
            ),
            "metrics": (
                "cv/pib/score 来自 square_shaping (与 compare_sim_vs_hw 同一批函数); "
                "ee/ar/quality 来自 slm_square_shaping (惰性 import); "
                "帧先过 prepare_frame"
            ),
            "zero_order": (
                "平场帧 argmax (zero_order_center) 定位并冻结; 绝不假设在画面中心"
            ),
        },
        "stage_plans": stage_plans,
        "estimate": {
            "capture_s": capture_s,
            "gs_seconds_each": single_gs_s,
            "total_s": total_s,
            "limit_s": PLAN_LIMIT_S,
            "over_budget": list(over_budget),
            "npz_mb": ESTIMATE_NPZ_MB * sum(int(v["n_configs"]) for v in stage_plans.values()),
        },
        "out_root": run_root_text,
        "run_root": run_root_text,
        "is_sim": is_sim,
    }


def print_plan(plan: dict[str, Any]) -> None:
    """把计划打成人可读的一段 (dry-run 的全部输出)。"""
    b = plan["bench"]
    f = plan["frame"]
    s = plan["scan"]
    lines: list[str] = []
    add = lines.append
    add("=" * 78)
    add("GS 方形光斑三段扫描 —— 计划 (dry-run, 未打开任何设备)")
    add("=" * 78)
    add("")
    add("段 (按此顺序跑, 单向传递选定边长 / 选定相位)")
    for name in plan["stages"]:
        sp = plan["stage_plans"][name]
        add(f"  - {name:<8s} 自变量 {sp['variable']:<16s} {sp['n_configs']:>3d} 配置  "
            f"{sp['seconds']:>7.1f}s   {sp['detail']}")
    add("")
    add("实验系统几何")
    add(f"  波长 λ             = {b['wavelength_nm']:.1f} nm")
    add(f"  焦距 f             = {b['focal_length_m'] * 1e3:.3f} mm "
        "(由实测焦面标度 + CCD 像元反解; 只记进产物, 不参与 GS 换算)")
    add(f"  面板               = {b['panel_shape'][1]} x {b['panel_shape'][0]} px @ "
        f"{b['panel_pitch_um']:.1f} um")
    add(f"  照明 (面板坐标)     = 中心 {tuple(b['beam_center_panel'])}, 半径 "
        f"{b['beam_radius_px']:.0f} px ⇒ 包围方框 {b['pupil_side_px']} px")
    add(f"  CCD 像元           = {b['camera_pixel_um']} um")
    add(f"  远场标度           = {s['camera_px_per_gs_px']:.4f} camera px / GS 网格像素 "
        f"(= TILT_SHIFT_SCALE / {b['pupil_side_px']})")
    add("")
    add("0 级与最大边长")
    add(f"  画面尺寸           = {f['shape'][1]} x {f['shape'][0]} px"
        "   ⚠️ 计划假设 (--dry-run-frame-shape; 实测用相机实际画面)")
    add(f"  0 级位置           = ({f['zero_order_xy'][0]:.0f}, {f['zero_order_xy'][1]:.0f})"
        f"   {f['assumed_note']}")
    add(f"  最大边长           = {f['side_max']} px   [{f['side_max_note']}]")
    add(f"    原始值           = {f['side_max_raw']} px = 2·(min(cx, W-1-cx, cy, H-1-cy) - "
        f"{f['edge_margin_px']}), 再向下取到步长整数倍")
    add("")
    add("第 1 段: 边长扫描")
    add(f"  实际边长列表       = {s['sides']}")
    add(f"  点数               = {len(s['sides'])}; 步长见 --side-step; "
        f"选点依据 = {s.get('rank_by', 'score')}")
    add(f"  近 {s['near_target']:.0f} 选点窗口   = [{s['near_window'][0]:.0f}, "
        f"{s['near_window'][1]:.0f}] px (半宽 {s['near_tolerance']:.0f})")
    add(f"  窗口内候选         = {s['near_candidates']}"
        + ("" if s["near_candidates"] else "   ⚠️ 一个候选都没有 —— 开跑会以退出码 5 失败"))
    add("")
    add("第 2 段: Zernike 阶数扫描")
    order_list = [int(n) for n in s["orders"]]
    if len(order_list) <= MAX_ORDERS_LISTED_INLINE:
        order_text = "[" + ", ".join(str(n) for n in order_list) + "]"
    else:
        order_text = (
            "[" + ", ".join(str(n) for n in order_list[:3]) + ", …, "
            + ", ".join(str(n) for n in order_list[-3:]) + "]"
        )
    shown_orders = order_list if len(order_list) <= MAX_ORDERS_LISTED_INLINE else (
        order_list[:3] + order_list[-3:]
    )
    add(f"  阶数               = {order_text}  (共 {len(order_list)} 个)")
    add("  阶数 → 模式数       = " + ", ".join(
        f"{n}→{s['orders_modes'][n]}" for n in shown_orders
    ))
    add(f"  拟合网格           = {s['zernike_fit_grid']} (RZern 内存 ≈ "
        f"{s['zernike_fit_grid'] ** 2 * modes_for_order(max(s['orders'])) * 8 / 1e6:.0f} MB)")
    add("  做法               = 把选定边长的 GS 相位拟合成 Zernike (Noll 1976), "
        "只用前 N 阶重构并测量")
    if "sides" not in plan["stages"]:
        if s.get("side_px_override") is None:
            add("  ⚠️ 没跑第 1 段又没给 --side-px: --execute 会直接退出 "
                "(选定相位无从谈起)")
        else:
            add(f"  选定边长           = {s['side_px_override']} px (--side-px, "
                "第 1 段未跑, 所以没有现场选点)")
    add("")
    add("第 3 段: 闪耀光栅周期扫描 (面板 x 斜坡 → 相机 y)")
    add(f"  方框按边长 {s['blaze_planning_side_px']:.0f} px 规划 (选定边长开跑后才定)")
    add("  周期(panel px)   预测位移(camera px)        框心(预测)              装得下")
    for row in s["blaze_table"]:
        add(f"  {row['blaze_period_px']:>8d}        ({row['predicted_shift_x_px']:+.1f}, "
            f"{row['predicted_shift_y_px']:+.1f}) = {row['predicted_shift_px']:>7.1f}"
            f"     ({row['box_center_predicted_x']:.0f}, "
            f"{row['box_center_predicted_y']:.0f})"
            f"{'':<4s} {'是' if row['fits_frame'] else '否'}")
    add("  提示: 位移后冻结方框失效 —— 框心 = 冻结 0 级 + 预测位移, 质心在框内量, "
        "整帧峰值单独记")
    add("         (整帧 argmax 会被静止的 0 级拿走: 它只占 2.86% 能量却是衍射极限大小)")
    add("")
    add("GS 网格")
    add(f"  N                  = {s['gs_grid']}   [{s['gs_grid_note']}]")
    add(f"  迭代               = {s['gs_iters']} 次; 远场窗口 = "
        f"{s['fft_window_camera_px']:.0f} camera px "
        f"(一格 {s['camera_px_per_gs_px']:.3f} camera px, 与 N 无关)")
    add("")
    add("设备")
    d = plan["devices"]
    add(f"  SLM                = {d['slm_driver']}")
    add(f"  CCD                = {d['ccd_driver']}")
    add(f"  cam_type/cam_id    = {d['cam_type']} / {d['cam_id']}; 曝光请求 "
        f"{d['exposure_ms_requested']} ms (上限遵守 AO_DAHENG_MAX_EXPOSURE_MS)")
    add(f"  每配置平均帧数      = {d['frames_per_measurement']} "
        "(bench_kernels.display_and_average: 3 帧丢弃 + 稳定性判据 + 平均)")
    if plan["is_sim"]:
        add(f"  仿真               = seed {d['sim_seed']}, 远场补零 "
            f"{d['sim_far_field_padding']}, 窗口 {d['sim_cam_size']} sim px")
    add("")
    add("口径要点")
    for key, text in plan["optics_notes"].items():
        add(f"  - {text}")
    add("")
    add("预计耗时 (SLM 的 USB 句柄独占 ⇒ 全部串行)")
    e = plan["estimate"]
    for name in plan["stages"]:
        sp = plan["stage_plans"][name]
        flag = "  ⚠️ 超过预算, 建议拆段" if float(sp["seconds"]) > PLAN_LIMIT_S else ""
        add(f"  {name:<8s} {sp['n_configs']:>3d} × (GS {sp['gs_seconds_each']:.1f}s + "
            f"采集 {e['capture_s']:.1f}s) = {sp['seconds']:>7.1f}s{flag}")
    add(f"  合计               = {e['total_s']:.0f}s ({e['total_s'] / 60:.1f} min); "
        f"预算 {e['limit_s']:.0f}s/段")
    add(f"  磁盘               ≈ {e['npz_mb']:.0f} MB (原始帧, 已压缩)")
    if e["over_budget"]:
        add(f"  ⚠️ 超预算的段: {e['over_budget']} —— 开跑前请用 --scan 拆开跑")
    add("")
    add("原子模式 (推荐: 一个进程只跑一组参数, 跨进程按 --run-root 累积)")
    add("  --scan sides  --side-px <px>         # 一个边长 = 一次进程")
    add("  --scan zernike --n-max <N>           # 一个阶数 = 一次进程 (读 sides/selected.json)")
    add("  --scan blaze  --blaze-period <px>    # 一个周期 = 一次进程 (读 sides/selected.json)")
    add("  --scan select                        # 离线: 近目标窗口选点 → sides/selected.json")
    add("  --scan summarize                     # 离线: 重建每段 CSV + 根 manifest + 汇总图")
    add("  --plan-commands                      # 离线: 打印上面那份命令清单 (每行一个进程)")
    add(f"  整段扫描 (--scan all / 不给值选择器) 仍可用, 仅调试用: 一次约 "
        f"{plan['estimate']['total_s'] / 3600:.1f} h, 中断即整轮作废")
    add("")
    add(f"产物根目录: {plan['out_root']}   (原子模式: <根>/<段>/<配置>/ + <根>/<段>/measurements.csv"
        " + <根>/manifest.json)")
    add("=" * 78)
    logger.info("\n".join(lines))


# ---------------------------------------------------------------------------
# --execute
# ---------------------------------------------------------------------------
@dataclass
class _ScanParams:
    """:class:`~ao_shaping.drivers.slm.factory.BenchDeviceParams` 的结构化实现。

    那个 Protocol 只要求这几个字段, 所以在脚本里本地声明一个 carrier 即可 ——
    这是**适配**, 不是重实现: 设备构造仍然走仓库工厂
    (:func:`~ao_shaping.drivers.slm.factory.open_slm` /
    :func:`~ao_shaping.drivers.ccd.common.create_camera`)。
    """

    cam_type: str
    cam_id: int
    exposure_time_ms: float
    cam_size: int
    slm_number: int
    slm_wavelength: int


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_rows(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    """把测量行写成 CSV + JSON (两件都是原子替换)。

    列名取**所有行的并集** (按首次出现顺序): 失败行多一个 ``error`` 列, 而
    ``DictWriter`` 只认第一行的列名, 照旧写法会在写失败行时抛 ``ValueError``。
    """
    if not rows:
        return
    write_csv_rows(Path(out_dir) / "measurements.csv", rows)
    write_json_atomic(Path(out_dir) / "measurements.json", rows)


def _plot_stage(out_dir: Path, rows: list[dict[str, Any]], stage: str) -> None:
    """一段一张汇总图 (Agg, 中文 rcParams —— 与仓库其它报告脚本一致)。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = [
        "Noto Sans CJK SC", "Microsoft YaHei", "SimHei", "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    xkey = STAGE_VARIABLES[stage]
    sel = sorted(
        (r for r in rows if xkey in r and np.isfinite(as_float(r.get("cv")))),
        key=lambda r: as_float(r.get(xkey)),
    )
    if not sel:
        return
    xs = [as_float(r.get(xkey)) for r in sel]
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 4.5))
    ax = axes[0]
    ax.plot(xs, [as_float(r.get("cv")) for r in sel], "o-", color="#FF5722", linewidth=2,
            label="CV (框内)")
    ax.plot(xs, [as_float(r.get("ar")) for r in sel], "s--", color="#795548", linewidth=1.2,
            label="AR (宽/高)")
    ax.set_xlabel(xkey)
    ax.set_ylabel("CV / AR")
    ax.set_title(f"{stage}: 框内均匀度与形状")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax2 = axes[1]
    ax2.plot(xs, [as_float(r.get("pib")) for r in sel], "o-", color="#2196F3", linewidth=2,
             label="PIB (框内功率占比)")
    ax2.plot(xs, [as_float(r.get("quality")) for r in sel], "s--", color="#9E9E9E",
             linewidth=1.2, label="方形整形评分 (exp(-2CV)/EE/AR 加权)")
    ax2.set_xlabel(xkey)
    ax2.set_ylabel("PIB / 评分")
    ax2.set_title(f"{stage}: 环围能量与方形整形评分")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)

    ax3 = axes[2]
    ax3.plot(xs, [as_float(r.get("score")) for r in sel], "o-", color="#4CAF50", linewidth=2,
             label="复合分数 0.5·PIB+0.5/(1+CV)")
    if any(np.isfinite(as_float(r.get("model_score"))) for r in sel):
        ax3.plot(xs, [as_float(r.get("model_score")) for r in sel], "^:",
                 color="#9C27B0", linewidth=1.4, label="模型预测 (GS 自身远场)")
    ax3.set_xlabel(xkey)
    ax3.set_ylabel("复合分数")
    ax3.set_title(f"{stage}: 得分")
    ax3.grid(alpha=0.3)
    ax3.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / "summary.png", dpi=150)
    plt.close(fig)


def _configure_sim_if_needed(args: argparse.Namespace) -> dict[str, Any] | None:
    """仿真后端的进程级就位: 种子 + 远场补零 (复用探针的实现)。

    Returns:
        ``None`` (硬件) 或 ``{"row_scale":..., "col_scale":..., "window_px":...}``。

    仿真远场的角标度由 ``far_field_padding`` 决定, 且面板是 1920x1200 的长方形 ——
    单一补零因子**无法同时对齐两轴**, 所以换算必须用
    :attr:`SimPibSystem.camera_px_per_model_px` 给的两个比例, 而不是一个数。
    """
    if str(args.cam_type).strip().lower() != "sim":
        return None
    configure_sim_system(int(args.sim_seed), int(args.sim_far_field_padding))
    from ao_shaping.drivers.sim.slm_pib_sim import get_system

    system = get_system()
    row_scale, col_scale = (float(v) for v in system.camera_px_per_model_px)
    return {"row_scale": row_scale, "col_scale": col_scale, "window_px": int(args.cam_size)}


def _sim_frame_to_camera_px(
    frame: np.ndarray, sim_map: dict[str, Any] | None
) -> np.ndarray:
    """仿真帧 → 相机像素口径 (硬件原样返回)。

    只在仿真分支做重采样: 仿真的一个远场采样点是 ``(row, col)`` 个相机像元, 而所有
    方框/质心/位移都是**相机 px** —— 不换算的话仿真与硬件两栏不可比, 而
    ``--cam-type sim`` 存在的意义正是「同一套口径下先离线验证一次」。
    """
    if sim_map is None:
        return np.asarray(frame, dtype=np.float64)
    height, width = np.asarray(frame).shape
    target = (
        max(int(round(height * float(sim_map["row_scale"]))), 2),
        max(int(round(width * float(sim_map["col_scale"]))), 2),
    )
    return upsample_bilinear_to_shape(np.asarray(frame, dtype=np.float64), target)


# ---------------------------------------------------------------------------
# 设备会话: open → pre-flight → (每组参数) → close
# ---------------------------------------------------------------------------
@dataclass
class BenchSession:
    """一次进程内的设备会话 (SLM + CCD), 预检结论跟着会话走。

    「pre-flight」= 曝光上限解析 + 平场帧上 ``argmax`` 定位 0 级 (并冻结) + 光路
    受阻判据。整段扫描与原子模式用的是**同一段**预检代码, 所以两者的 ``preflight``
    字段逐字段可比 —— 这也是原子模式敢让「每个进程重新预检」的原因: 重新预检不会
    引入第二种口径。
    """

    args: argparse.Namespace
    bench: Bench
    slm: Any
    cam: Any
    is_sim: bool
    sim_map: dict[str, Any] | None
    exposure_ceiling_ms: float
    exposure_ms_working: float
    protected_slot: int | None
    zero_order_xy: tuple[float, float]
    frame_shape: tuple[int, int]
    spot: dict[str, Any]
    display_and_average: Callable[..., np.ndarray] = field(repr=False)

    # -- 取帧 --------------------------------------------------------------
    def capture(self, phase_panel: np.ndarray) -> np.ndarray:
        """下发一张面板相位并取平均帧 (带超时保护 + 相机口径换算)。

        走 :func:`~ao_shaping.tools.kernels.bench_kernels.display_and_average`: 它内部
        用 ``display_data(gray)`` **不给槽号**, 由驱动自己在 127 个槽之间轮换 —— 那正是
        「同槽连续 display 是 no-op」那条坑的官方规避方式, 所以本脚本不需要自己挑槽。
        """
        raw = guarded(
            "CCD 取帧",
            lambda: self.display_and_average(
                self.cam, self.slm, phase_panel,
                n_frames=int(self.args.frames),
                n_discard=int(self.args.settle_n_discard),
                wait_time_s=float(self.args.settle_wait_s),
                stable_tol=float(self.args.settle_tol),
                max_wait_s=float(self.args.settle_max_wait_s),
            ),
            timeout_s=ACQUIRE_TIMEOUT_S,
        )
        return _sim_frame_to_camera_px(raw, self.sim_map)

    def capture_retry(self, phase_panel: np.ndarray) -> np.ndarray:
        """取帧, 遇 USB 瞬时错误退避重试。

        驱动的 ``_write_phase`` 内部只试 3 次 (间隔 100 ms), 而 FTDI 的
        ``-10032 USB其他错误`` 常连着三次一起失败 (2026-10-20 实测)。这里再包一层
        0.5/2/8 s 退避; 槽位由 ``display_data`` 自己轮换, 所以重试不会撞上同一个槽。
        """
        from ao_shaping.drivers.slm.santec import SantecError

        last: Exception | None = None
        for attempt in range(USB_RETRY_ATTEMPTS):
            try:
                return self.capture(phase_panel)
            except SantecError as exc:
                last = exc
                wait = USB_RETRY_BASE_WAIT_S * (4 ** attempt)
                logger.info(
                    f"[USB 重试 {attempt + 1}/{USB_RETRY_ATTEMPTS}] "
                    f"失败: {exc}; 等待 {wait:.1f}s 后重试",

                )
                time.sleep(wait)
        assert last is not None
        raise last

    # -- 预检记录 ----------------------------------------------------------
    def preflight_dict(self) -> dict[str, Any]:
        """这次会话的预检结论 (每行的 ``metrics.json`` 都带一份)。

        「pre-flight 通过」不等于「可以不记」: 曝光上限、0 级落点、光斑评估都是
        **这次会话**的事实, 换个会话就会变 —— 不落盘就再也说不清那张图是在什么条件下
        拍的。
        """
        payload: dict[str, Any] = {
            "timestamp_utc": _utc_now_iso(),
            "frame_shape_hw": [int(v) for v in self.frame_shape],
            "zero_order_xy_frozen": [float(v) for v in self.zero_order_xy],
            "zero_order_method": (
                "平场帧 zero_order_center(refine=False, despike_k=3) —— 定位一次即冻结"
            ),
            "exposure_ms_requested": float(self.args.exposure_ms),
            "exposure_ceiling_ms": float(self.exposure_ceiling_ms),
            "exposure_ms_working": float(self.exposure_ms_working),
            "frames_per_measurement": int(self.args.frames),
            "settle": {
                "n_discard": int(self.args.settle_n_discard),
                "wait_s": float(self.args.settle_wait_s),
                "tol": float(self.args.settle_tol),
                "max_wait_s": float(self.args.settle_max_wait_s),
            },
            "spot_assessment": dict(self.spot),
            "defocus_focal_mm": self.args.defocus_focal_mm,
            "slm_number": int(self.args.slm_number),
            "cam_type": str(self.args.cam_type),
            "cam_id": int(self.args.cam_id),
            "protected_memory_slot": self.protected_slot,
        }
        if self.is_sim and self.sim_map is not None:
            payload["backend"] = "sim"
            payload["sim_seed"] = int(self.args.sim_seed)
            payload["sim_far_field_padding"] = int(self.args.sim_far_field_padding)
            payload["sim_camera_px_per_model_px"] = [
                float(self.sim_map["row_scale"]),
                float(self.sim_map["col_scale"]),
            ]
        else:
            payload["backend"] = str(self.args.cam_type)
        return payload


@contextmanager
def open_bench(
    args: argparse.Namespace,
    bench: Bench,
    *,
    focus_panel: np.ndarray | None,
    abort_dir: Path,
    abort_context: dict[str, Any] | None = None,
) -> Iterator[BenchSession]:
    """开设备 → 预检 → ``yield`` 会话 → 关设备 (永远清掉持久叠加层)。

    设备 import 全部锁在这里 (只有 ``--execute`` 会走到), 所以 ``--dry-run`` 的
    「零驱动包」保证不受影响。

    Args:
        focus_panel: 焦面补偿相位 (``None`` = 不补偿)。必须在**定位 0 级之前**就位 ——
            补偿以面板标称中心为心, 而照明并不落在那里 ⇒ 等价于偏心透镜。
        abort_dir: 光路受阻时 ``run_config.json`` 落在哪。
        abort_context: 追加进那份中止记录的上下文 (整段模式写计划, 原子模式写配置)。
    """
    from ao_shaping.drivers.ccd.common import create_camera
    from ao_shaping.drivers.slm.factory import open_slm, sync_device_wavelength
    from ao_shaping.tools.kernels.bench_kernels import display_and_average
    from ao_shaping.tools.slm.panel.display_state import preserve_slm_display
    from ao_shaping.utils.image.exposure_control import (
        configured_daheng_exposure_limit,
        probe_exposure_ceiling_ms,
    )

    is_sim = str(args.cam_type).strip().lower() == "sim"
    sim_map = _configure_sim_if_needed(args)
    if is_sim and sim_map is not None:
        logger.info(
            "仿真后端: seed={}, 远场补零={}, 每仿真采样点 = ({:.3f} 行, {:.3f} 列) 相机像元".format(
                int(args.sim_seed), int(args.sim_far_field_padding),
                sim_map["row_scale"], sim_map["col_scale"],
            ),

        )
    params = _ScanParams(
        cam_type=args.cam_type,
        cam_id=int(args.cam_id),
        exposure_time_ms=float(args.exposure_ms),
        cam_size=int(args.cam_size),
        slm_number=int(args.slm_number),
        slm_wavelength=int(bench.wavelength_nm),
    )

    with guarded("SLM 打开", lambda: open_slm(params), timeout_s=OPEN_TIMEOUT_S) as slm:
        # ⚠️ `Santec.open()` 返回 None —— 不能拿它的返回值当上下文管理器。
        with nullcontext(guarded("SLM open()", lambda: slm.open(), timeout_s=OPEN_TIMEOUT_S)):
            sync_device_wavelength(params, slm)
            cam_kwargs: dict[str, Any] = (
                {"cam_size": int(args.cam_size)} if is_sim else {}
            )
            with create_camera(
                params.cam_type, params.cam_id,
                exposure_time_ms=params.exposure_time_ms, **cam_kwargs,
            ) as cam:
                display_guard = nullcontext() if is_sim else preserve_slm_display(slm)
                with display_guard:
                    protected = protected_memory_slot(slm)
                    if protected is not None:
                        logger.info(f"运行前正在显示的内存槽 (将被排除): {protected}")

                    # -- 曝光上限 -----------------------------------------
                    if is_sim:
                        ceiling = float(args.exposure_ms)
                        working_ms = float(args.exposure_ms)
                    else:
                        ceiling = configured_daheng_exposure_limit()
                        if not ceiling:
                            ceiling = probe_exposure_ceiling_ms(
                                cam.reset_exposure_time,
                                lambda: np.asarray(cam.get_numpy_image(n_sample=1)),
                            ).ceiling_ms
                        working_ms = min(float(args.exposure_ms), float(ceiling))
                        if working_ms < 0.02:
                            raise RuntimeError("解出的曝光上限低于相机最小值 0.02 ms")
                    actual = float(cam.reset_exposure_time(working_ms))
                    logger.info(
                        f"曝光: 请求 {args.exposure_ms} ms -> 上限 {ceiling} ms -> "
                        f"生效 {actual} ms",

                    )

                    session = BenchSession(
                        args=args,
                        bench=bench,
                        slm=slm,
                        cam=cam,
                        is_sim=is_sim,
                        sim_map=sim_map,
                        exposure_ceiling_ms=float(ceiling),
                        exposure_ms_working=float(actual),
                        protected_slot=protected,
                        zero_order_xy=(float("nan"), float("nan")),
                        frame_shape=(int(bench.panel_shape[0]), int(bench.panel_shape[1])),
                        spot={},
                        display_and_average=display_and_average,
                    )
                    try:
                        # -- 0 级: 平场帧上 argmax, 定位一次即冻结 -------------
                        slm.set_phase_overlay(focus_panel)
                        flat = session.capture_retry(
                            np.zeros(bench.panel_shape, dtype=np.float64)
                        )
                        cx0, cy0 = zero_order_center(flat, refine=False, despike_k=3)
                        session.zero_order_xy = (float(cx0), float(cy0))
                        session.frame_shape = (int(flat.shape[0]), int(flat.shape[1]))
                        logger.info(f"0 级 (全幅 argmax, 已冻结): ({cx0}, {cy0}); "
                              f"画面 {flat.shape[1]} x {flat.shape[0]}")

                        assessment = assess_spot(flat)
                        session.spot = assessment.as_dict()
                        if assessment.blocked:
                            abort = {
                                "timestamp_utc": _utc_now_iso(),
                                "aborted": "path_blocked",
                                "spot_assessment": assessment.as_dict(),
                                "zero_order_xy": [int(cx0), int(cy0)],
                                "preflight": session.preflight_dict(),
                                **(abort_context or {}),
                            }
                            write_json_atomic(Path(abort_dir) / "run_config.json", abort)
                            logger.info(
                                "\n[中止] 光路受阻: " + assessment.reason + "\n"
                                "       平场帧里没有任何局域光斑, 扫描只会记下噪声。\n"
                                f"       判定与理由已写进 {Path(abort_dir) / 'run_config.json'}",

                            )
                            raise SystemExit(EXIT_PATH_BLOCKED)
                        if not assessment.usable:
                            logger.info(
                                f"[警告] 平场帧有光但没有可评分的局域光斑 "
                                f"({assessment.reason}); 扫描继续, 但指标要按"
                                "「光斑没聚焦」解读",

                            )
                        yield session
                    finally:
                        # 叠加层是驱动的持久状态: 不清掉的话, 之后任何调用方都会在
                        # 毫不知情的情况下带着这张焦面相位图。
                        slm.set_phase_overlay(None)


def _focus_overlay_panel(args: argparse.Namespace, bench: Bench) -> np.ndarray | None:
    """焦面补偿相位 (``None`` = 不补偿, 即默认)。

    CCD 现在就落在 2f 焦面上 ⇒ 旧的瞳面菲涅尔补偿 (``f_v ≈ 1549 mm``) 已过期, 默认
    不加; 需要时 ``--defocus-focal-mm`` 显式给 (``inf`` = 显式平场)。
    """
    if args.defocus_focal_mm is None:
        return None
    return fresnel_panel(
        float(args.defocus_focal_mm),
        radius=int(bench.beam_radius_px),
        pupil_center=bench.beam_center_panel,
        panel_shape=bench.panel_shape,
        wavelength_nm=bench.wavelength_nm,
        pixel_um=SLM_PITCH_M * 1e6,
    )


def run_execute(args: argparse.Namespace, bench: Bench, plan: dict[str, Any]) -> Path:
    """整段模式: 开设备, 把该段的**全部**配置跑一遍。返回批次根目录。

    ⚠️ 仅调试用: 边长段一次就是几十分钟, 三段连跑以小时计, 中途中断即整轮作废, 且
    已落盘的点没有「属于哪一组」的身份。常规路径是原子模式
    (``--scan <段> --<值>``) + ``--plan-commands``。
    """
    from ao_shaping.drivers.slm.santec import SantecError
    from ao_shaping.utils.wavefront.zernike_calc import ZernikeGenerator

    out_root = Path(plan["out_root"])
    if not out_root.is_absolute():
        out_root = ROOT / out_root
    out_root.mkdir(parents=True, exist_ok=True)
    logger.info(f"根目录: {out_root}")

    selected: dict[str, Any] = {
        "side_px": None, "phase_pupil": None, "gs_grid": None, "zernike_fit": None,
    }
    #: ``{边长: GS 瞳面相位}`` —— GS 是确定性的, 同一个边长重算得到逐位相同的相位;
    #: 这里缓存只为省掉重复的 7 s (第 1 段跑完后再算选定边长的那一张)。
    gs_cache: dict[int, np.ndarray] = {}
    stage_dirs: dict[str, Path] = {}
    stage_rows: dict[str, list[dict[str, Any]]] = {}

    focus_panel = _focus_overlay_panel(args, bench)
    with open_bench(
        args, bench,
        focus_panel=focus_panel,
        abort_dir=out_root,
        abort_context={"plan": plan},
    ) as sess:
        cx0, cy0 = sess.zero_order_xy
        flat_frame_shape = sess.frame_shape
        is_sim = sess.is_sim
        ceiling = sess.exposure_ceiling_ms
        actual = sess.exposure_ms_working
        protected = sess.protected_slot
        assessment = sess.spot

        side_max_raw, side_max = max_side_from_frame(
            flat_frame_shape, (cx0, cy0),
            margin_px=int(args.edge_margin_px), step_px=int(args.side_step),
        )
        if str(args.side_max).strip().lower() == "auto":
            side_max_used = side_max
        else:
            requested_max = int(args.side_max)
            side_max_used = min(requested_max, side_max)
            if requested_max > side_max:
                logger.info(f"[警告] --side-max {requested_max} 超出画面可用半径, "
                      f"已限制为 {side_max} px")
        logger.info(f"最大边长: {side_max} px (原始 {side_max_raw} px)")

        sides = side_scan_list(int(args.side_min), side_max_used, int(args.side_step))

        # -- GS 网格: 用**实测**画面重算 (计划里那份是按假设画幅算的) ---------
        periods = tuple(int(p) for p in args.blaze_periods)
        # 第 3 段的方框边长「选定边长」开跑后才定; 这里用近 1000 的目标值作规划假设
        # (夹到实际边长范围内)。
        near_side = float(
            min(max(float(args.target_side_near), float(sides[0])), float(sides[-1]))
        )
        periods_table = (
            blaze_shift_table(
                periods, bench.wavelength_nm, flat_frame_shape,
                side_px=float(near_side),
                center=(float(cx0), float(cy0)),
            )
            if periods
            else []
        )
        max_shift = max(
            (float(v["predicted_shift_px"]) for v in periods_table), default=0.0
        )
        span_cam = float(side_max_used) + 2.0 * max_shift
        grid_auto = bench.gs_grid_for(span_cam)
        if str(args.gs_grid).strip().lower() == "auto":
            gs_grid = int(grid_auto)
        else:
            gs_grid = int(args.gs_grid)
            if gs_grid * bench.camera_px_per_gs_px < span_cam:
                raise SystemExit(
                    f"--gs-grid {gs_grid} 的远场窗口 "
                    f"{gs_grid * bench.camera_px_per_gs_px:.0f} camera px "
                    f"装不下 {span_cam:.0f} camera px (会被窗口边界卷绕); "
                    f"至少 {grid_auto}"
                )
        if gs_grid != int(plan["scan"]["gs_grid"]):
            logger.info(
                f"[提示] 实测画面下 GS 网格从计划的 "
                f"{int(plan['scan']['gs_grid'])} 变为 {gs_grid}",

            )
        logger.info(
            f"边长列表: {list(sides)}; GS 网格 N={gs_grid} "
            f"(远场窗口 {gs_grid * bench.camera_px_per_gs_px:.0f} camera px, "
            f"需 {span_cam:.0f})",

        )

        def stage_dir(name: str) -> Path:
            path = out_root / name
            path.mkdir(parents=True, exist_ok=True)
            return path

        def record(
            out_dir: Path,
            rows: list[dict[str, Any]],
            phase_panel: np.ndarray,
            frame: np.ndarray,
            step: dict[str, Any],
            center: tuple[float, float],
            side_px: float,
        ) -> dict[str, Any]:
            """记一行 + 落盘该配置的三个文件。"""
            idx = len(rows)
            metrics = measure_box(frame, center, side_px)
            row = {
                "index": idx,
                "timestamp_utc": _utc_now_iso(),
                **step,
                "box_center_x": float(center[0]),
                "box_center_y": float(center[1]),
                "box_side_px": float(side_px),
                "exposure_ms_readback": float(read_exposure_ms(sess.cam)),
                **metrics,
            }
            np.save(out_dir / f"phase_{idx:03d}.npy",
                    np.asarray(phase_panel, dtype=np.float64))
            np.savez_compressed(
                out_dir / f"frame_{idx:03d}.npz",
                frame=np.asarray(frame, dtype=np.float32),
                params=json.dumps(row, ensure_ascii=False),
            )
            write_json_atomic(out_dir / f"metrics_{idx:03d}.json", row)
            rows.append(row)
            _write_rows(out_dir, rows)
            logger.info(
                f"[{out_dir.name} #{idx:03d}] " + " ".join(
                    f"{k}={row[k]:.4g}" if isinstance(row.get(k), float) else f"{k}={row[k]}"
                    for k in ("side_px", "n_max", "blaze_period_px", "cv", "pib", "score")
                    if k in row
                ),

            )
            return row

        def record_failure(
            out_dir: Path, rows: list[dict[str, Any]],
            step: dict[str, Any], exc: Exception,
        ) -> dict[str, Any]:
            """记一行**失败** (取帧阶段的 USB 错误), 指标留 NaN。

            单点失败不该让整段作废: 失败行带着 ``error`` 落盘, 离线出图时按 NaN 跳过。
            """
            idx = len(rows)
            row = {
                "index": idx,
                "timestamp_utc": _utc_now_iso(),
                **step,
                "cv": float("nan"), "pib": float("nan"),
                "ee": float("nan"), "ar": float("nan"),
                "score": float("nan"), "quality": float("nan"),
                "error": str(exc),
            }
            rows.append(row)
            _write_rows(out_dir, rows)
            logger.info(f"[{out_dir.name} #{idx:03d}] 失败: {exc}")
            return row

        def write_stage_meta(
            name: str, out_dir: Path, extra: dict[str, Any]
        ) -> None:
            payload = {
                "timestamp_utc": _utc_now_iso(),
                "stage": name,
                "variable": STAGE_VARIABLES[name],
                "plan": plan,
                "preflight": sess.preflight_dict(),
                "zero_order_xy_frozen": [int(cx0), int(cy0)],
                "frame_shape_hw": [int(v) for v in flat_frame_shape],
                "exposure_ms_requested": float(args.exposure_ms),
                "exposure_ceiling_ms": float(ceiling),
                "exposure_ms_working": actual,
                "frames_per_measurement": int(args.frames),
                "settle": {
                    "n_discard": int(args.settle_n_discard),
                    "wait_s": float(args.settle_wait_s),
                    "tol": float(args.settle_tol),
                    "max_wait_s": float(args.settle_max_wait_s),
                },
                "spot_assessment": dict(assessment),
                "defocus_focal_mm": args.defocus_focal_mm,
                "slm_number": int(args.slm_number),
                "cam_type": args.cam_type,
                "cam_id": int(args.cam_id),
                "protected_memory_slot": protected,
                **extra,
            }
            if is_sim:
                payload["backend"] = "sim"
                payload["sim_seed"] = int(args.sim_seed)
                payload["sim_far_field_padding"] = int(args.sim_far_field_padding)
                payload["sim_camera_px_per_model_px"] = list(
                    (sess.sim_map["row_scale"], sess.sim_map["col_scale"])
                )
            write_json_atomic(out_dir / "run_config.json", payload)

        # ============ 第 1 段: 边长扫描 =========================
        if "sides" in plan["stages"]:
            out_dir = stage_dir("sides")
            stage_dirs["sides"] = out_dir
            rows: list[dict[str, Any]] = []
            stage_rows["sides"] = rows
            write_stage_meta("sides", out_dir, {
                "sides": list(sides),
                "side_max_px": int(side_max_used),
                "side_max_raw_px": int(side_max_raw),
                "gs_grid": int(gs_grid),
                "gs_iters": int(args.gs_iters),
                "near_window_px": [
                    float(args.target_side_near) - float(args.near_tolerance),
                    float(args.target_side_near) + float(args.near_tolerance),
                ],
            })
            for side in sides:
                step = {
                    "side_px": int(side),
                    "gs_grid": int(gs_grid),
                    "gs_iters": int(args.gs_iters),
                }
                t0 = time.time()
                result = run_gs_square(
                    float(side), bench,
                    n_grid=gs_grid, iterations=int(args.gs_iters),
                )
                step["gs_seconds"] = float(time.time() - t0)
                gs_cache[int(side)] = np.asarray(result.phase, dtype=np.float64)
                phase_panel = pupil_phase_to_panel(result.phase, bench)
                step.update(model_box_metrics(result, float(side), bench))
                try:
                    frame = sess.capture_retry(phase_panel)
                except SantecError as exc:
                    record_failure(out_dir, rows, step, exc)
                    continue
                record(
                    out_dir, rows, phase_panel, frame, step,
                    (float(cx0), float(cy0)), float(side),
                )
            _write_rows(out_dir, rows)
            _plot_stage(out_dir, [flatten_row(r) for r in rows], "sides")

            # -- 近 1000 选点 (显式失败, 不静默退化) ---------------------
            try:
                best = select_near_side(
                    rows,
                    target_px=float(args.target_side_near),
                    tolerance_px=float(args.near_tolerance),
                    key=str(args.rank_by),
                )
            except NoNearSideError as exc:
                logger.info(f"\n[中止] {exc}")
                raise SystemExit(EXIT_NO_NEAR_SIDE) from exc
            selected["side_px"] = int(best["side_px"])
            logger.info(
                f"[选定] 边长 {selected['side_px']} px "
                f"(近 {args.target_side_near}±{args.near_tolerance}, "
                f"{args.rank_by}={float(best[args.rank_by]):.4f})",

            )

        # 选定边长: 跑过第 1 段就用它的结果, 否则用 --side-px。
        # 只跑第 1 段时不需要「选定相位」, 所以这一整块跳过 (省掉一次 GS)。
        side_star: int | None = None
        needs_selected_phase = any(s in plan["stages"] for s in ("zernike", "blaze"))
        if selected["side_px"] is None and needs_selected_phase:
            if args.side_px is None:
                raise SystemExit(
                    "只跑 zernike / blaze 时需要 --side-px 指定边长 "
                    "(或 --scan all 让第 1 段现场选)"
                )
            selected["side_px"] = int(args.side_px)
        if selected["side_px"] is not None:
            side_star = int(selected["side_px"])
        if side_star is not None and side_star > side_max_used:
            raise SystemExit(
                f"选定边长 {side_star} 超过画面可用最大边长 {side_max_used}"
            )
        if needs_selected_phase:
            # 第 1 段跑过的那张相位直接从缓存取 (逐位相同), 否则现场算一次。
            if side_star in gs_cache:
                selected["phase_pupil"] = gs_cache[side_star]
            else:
                gs_star = run_gs_square(
                    float(side_star), bench,
                    n_grid=gs_grid, iterations=int(args.gs_iters),
                )
                selected["phase_pupil"] = np.asarray(gs_star.phase, dtype=np.float64)
                gs_cache[side_star] = selected["phase_pupil"]
            selected["gs_grid"] = int(gs_grid)

        # ============ 第 2 段: Zernike 阶数 =====================
        if "zernike" in plan["stages"]:
            out_dir = stage_dir("zernike")
            stage_dirs["zernike"] = out_dir
            rows = []
            stage_rows["zernike"] = rows
            orders = tuple(int(n) for n in plan["scan"]["orders"])
            t_fit = time.time()
            fit = fit_zernike_pupil(
                selected["phase_pupil"],
                fit_grid=int(args.zernike_fit_grid),
                max_order=max(orders),
            )
            fit_seconds = float(time.time() - t_fit)
            selected["zernike_fit"] = fit
            generator = ZernikeGenerator(
                (fit.fit_grid, fit.fit_grid),
                radius=fit.fit_grid / 2.0,
                n_orders=fit.max_order,
            )
            write_stage_meta("zernike", out_dir, {
                "selected_side_px": side_star,
                "orders": list(orders),
                "fit_grid": int(fit.fit_grid),
                "fit_max_order": int(fit.max_order),
                "fit_seconds": fit_seconds,
                "fit_coefficients_noll": [float(c) for c in fit.coefficients],
            })
            np.save(out_dir / "zernike_coefficients.npy", fit.coefficients)
            for n_max in orders:
                step = {"n_max": int(n_max), "selected_side_px": side_star}
                t0 = time.time()
                recon, stats = reconstruct_from_fit(
                    fit, int(n_max), generator=generator
                )
                step.update(stats)
                step["reconstruct_seconds"] = float(time.time() - t0)
                phase_panel = pupil_phase_to_panel(recon, bench)
                try:
                    frame = sess.capture_retry(phase_panel)
                except SantecError as exc:
                    record_failure(out_dir, rows, step, exc)
                    continue
                record(
                    out_dir, rows, phase_panel, frame, step,
                    (float(cx0), float(cy0)), float(side_star),
                )
            _write_rows(out_dir, rows)
            _plot_stage(out_dir, [flatten_row(r) for r in rows], "zernike")

        # ============ 第 3 段: 闪耀光栅周期 =====================
        if "blaze" in plan["stages"]:
            out_dir = stage_dir("blaze")
            stage_dirs["blaze"] = out_dir
            rows = []
            stage_rows["blaze"] = rows
            base_kind = str(args.blaze_base)
            if base_kind == "zernike-best":
                pool = [
                    r for r in stage_rows.get("zernike", [])
                    if np.isfinite(float(r.get("score", float("nan"))))
                ]
                if not pool:
                    raise SystemExit(
                        "--blaze-base zernike-best 需要第 2 段有可用行 "
                        "(配 --scan all, 或改用 --blaze-base gs)"
                    )
                picked = max(pool, key=lambda r: float(r["score"]))
                picked_order = int(picked["n_max"])
                # 第 2 段已经拟合过就用那一份 (同一组系数); 只在它没覆盖到这个阶数
                # 时补一次拟合 —— 否则两处各得一套系数, 「同一个相位」就说不清了。
                existing = selected.get("zernike_fit")
                if existing is not None and existing.max_order >= picked_order:
                    used_fit = existing
                else:
                    used_fit = fit_zernike_pupil(
                        selected["phase_pupil"],
                        fit_grid=int(args.zernike_fit_grid),
                        max_order=picked_order,
                    )
                base_pupil, _ = reconstruct_from_fit(used_fit, picked_order)
                base_label = f"zernike@{picked_order}阶"
            else:
                base_pupil = selected["phase_pupil"]
                base_label = "gs"
            base_panel = pupil_phase_to_panel(base_pupil, bench)
            # 用**实测**的 0 级与选定边长重算位移表 (计划里那份按假设画幅算); 每一行
            # 落盘的 predicted / fits_frame 都以它为准。
            blaze_table = blaze_shift_table(
                periods, bench.wavelength_nm, flat_frame_shape,
                side_px=float(side_star), center=(float(cx0), float(cy0)),
            )
            write_stage_meta("blaze", out_dir, {
                "selected_side_px": side_star,
                "blaze_periods": list(periods),
                "blaze_base": base_label,
                "blaze_direction": "vertical (面板 x 斜坡 → 相机 y)",
                "blaze_table": blaze_table,
            })
            for entry in blaze_table:
                period = int(entry["blaze_period_px"])
                params_blaze = blaze_overlay(period, bench.wavelength_nm)
                dx_pred = float(entry["predicted_shift_x_px"])
                dy_pred = float(entry["predicted_shift_y_px"])
                box_center = (
                    float(entry["box_center_predicted_x"]),
                    float(entry["box_center_predicted_y"]),
                )
                step = {
                    "blaze_period_px": period,
                    "selected_side_px": side_star,
                    "predicted_shift_x_px": dx_pred,
                    "predicted_shift_y_px": dy_pred,
                    "predicted_shift_px": float(entry["predicted_shift_px"]),
                    "box_center_predicted_x": box_center[0],
                    "box_center_predicted_y": box_center[1],
                    "box_clipped": not bool(entry["fits_frame"]),
                }
                if not entry["fits_frame"]:
                    logger.info(
                        f"[警告] 周期 {period} px 的预测位移 "
                        f"{entry['predicted_shift_px']:.0f} px 让方框出画面; "
                        "这一行的 PIB/EE 会偏低 (box_clipped=true)",

                    )
                # 叠加层只有一条通道 (焦面补偿在用), 所以光栅加进**模式相位**。
                phase_panel = apply_zero_order_overlay(base_panel, params_blaze)
                try:
                    frame = sess.capture_retry(phase_panel)
                except SantecError as exc:
                    record_failure(out_dir, rows, step, exc)
                    continue
                row = record(
                    out_dir, rows, phase_panel, frame, step,
                    box_center, float(side_star),
                )
                # 实测位移 = 框内质心 - 冻结 0 级 (与 predicted_* 同一基准)。
                mx = float(row.get("box_centroid_x", float("nan")))
                my = float(row.get("box_centroid_y", float("nan")))
                step["measured_shift_x_px"] = (
                    (mx - float(cx0)) if np.isfinite(mx) else float("nan")
                )
                step["measured_shift_y_px"] = (
                    (my - float(cy0)) if np.isfinite(my) else float("nan")
                )
                step["shift_error_px"] = (
                    float(
                        math.hypot(
                            step["measured_shift_x_px"] - dx_pred,
                            step["measured_shift_y_px"] - dy_pred,
                        )
                    )
                    if np.isfinite(step["measured_shift_x_px"])
                    else float("nan")
                )
                row.update(step)
                # 位移字段是量完质心才有的, 所以这一份 metrics_*.json 要补写一次 ——
                # 否则它与 measurements.json 不一致。
                write_json_atomic(
                    out_dir / f"metrics_{row['index']:03d}.json", row
                )
                _write_rows(out_dir, rows)
            _write_rows(out_dir, rows)
            _plot_stage(out_dir, [flatten_row(r) for r in rows], "blaze")

    # ---- 根清单: 每段写完立刻更新 (后一段崩掉时, 前一段已被索引) -----------
    for name in plan["stages"]:
        if name not in stage_rows:
            continue
        rows = stage_rows[name]
        dir_path = stage_dirs[name]
        summary: dict[str, Any] = {"n_rows": len(rows), "mode": "whole-stage"}
        if name == "sides":
            summary["side_max_px"] = int(side_max_used)
            summary["selected_side_px"] = selected["side_px"]
        if name == "zernike":
            summary["selected_side_px"] = side_star
        if name == "blaze":
            summary["selected_side_px"] = side_star
            summary["blaze_base"] = str(args.blaze_base)
        update_manifest_atomic(
            out_root,
            {
                **group_record(
                    group=name,
                    variable=STAGE_VARIABLES[name],
                    meta={"cam_type": str(args.cam_type),
                          "exposure_ms": str(args.exposure_ms),
                          "gs_iters": str(args.gs_iters)},
                    dir_name=name,
                    n_rows=len(rows),
                ),
                "summary": summary,
                "dir_abs": str(dir_path),
            },
        )

    logger.info(f"\n产物根目录: {out_root}")
    logger.info(f"三段清单: {out_root / 'manifest.json'}")
    return out_root


# ---------------------------------------------------------------------------
# 原子模式: 一个进程 = 一组参数
# ---------------------------------------------------------------------------
def atomic_gs_grid(
    args: argparse.Namespace,
    bench: Bench,
    *,
    side_max_px: float,
    max_shift_px: float,
    hint: int | None = None,
) -> int:
    """原子模式的 GS 网格: 显式 > 第 1 段记录的 (``hint``) > 由远场窗口反解。

    为什么 ``hint`` 要优先: 第 2/3 段要**复用**第 1 段存盘的 pupil 相位, 网格不同
    就复用不了 (那等于重算一次 GS, 而且产物里两张相位不可比)。显式 ``--gs-grid``
    仍然最优先 —— 那是操作者明确要求的口径。
    """
    if str(args.gs_grid).strip().lower() != "auto":
        grid = int(args.gs_grid)
        if grid * bench.camera_px_per_gs_px < float(side_max_px):
            raise _fail(
                f"--gs-grid {grid} 的远场窗口 "
                f"{grid * bench.camera_px_per_gs_px:.0f} camera px 装不下边长 "
                f"{float(side_max_px):.0f} camera px (会被窗口边界卷绕)",
                2,
            )
        return grid
    if hint:
        return int(hint)
    return int(
        bench.gs_grid_for(float(side_max_px) + 2.0 * float(max_shift_px))
    )


def _resolve_base_pupil(
    run_root: Path,
    bench: Bench,
    selection: Selection,
    *,
    gs_grid: int,
    gs_iters: int,
) -> tuple[np.ndarray, str, dict[str, Any]]:
    """基准相位: 优先用第 1 段存盘的 pupil 相位, 缺失/形状不符才重算 GS。

    GS 是确定性的 (初值 = 目标的反向传播, 没有随机数), 所以缓存与重算逐位一致 ——
    缓存只省掉每次约 7.5 s 的 GS, 不改任何口径。

    Returns:
        ``(pupil 相位, "cache"|"recomputed", 诊断信息)``。
    """
    cached, info = load_cached_pupil_phase(
        Path(run_root),
        selection.side_px,
        gs_grid=int(gs_grid),
        phase_path=selection.phase_path,
    )
    if cached is not None:
        return cached, "cache", info
    logger.info(
        f"[基准相位] 没有可复用的缓存 ({info.get('reason')}); 重算 GS "
        f"(边长 {selection.side_px} px, N={int(gs_grid)}, {int(gs_iters)} 迭代)",

    )
    result = run_gs_square(
        float(selection.side_px), bench, n_grid=int(gs_grid), iterations=int(gs_iters)
    )
    return np.asarray(result.phase, dtype=np.float64), "recomputed", info


def _best_zernike_row(args: argparse.Namespace, run_root: Path) -> dict[str, Any]:
    """第 2 段里 ``score`` 最高的那一行 (``--blaze-base zernike-best`` 的输入)。"""
    stage_dir = stage_dir_from_args(args, run_root, "zernike")
    rows = [
        r for r in read_csv_rows(stage_dir / CONFIG_CSV_NAME)
        if math.isfinite(as_float(r.get("score"))) and math.isfinite(as_float(r.get("n_max")))
    ]
    if not rows:
        raise _fail(
            f"--blaze-base zernike-best 需要第 2 段有可用行, 但 "
            f"{stage_dir / CONFIG_CSV_NAME} 里没有 (先跑 --scan zernike, 或改用 "
            "--blaze-base gs)",
            EXIT_NO_MEASUREMENTS,
        )
    return max(rows, key=lambda r: as_float(r.get("score")))


def write_config_artifacts(
    cfg_dir: Path,
    *,
    phase_panel: np.ndarray,
    pupil_phase: np.ndarray | None,
    frame: np.ndarray | None,
    row: dict[str, Any],
) -> None:
    """落盘一组参数的全部文件, **metrics.json 最后写**。

    顺序是有意的: ``metrics.json`` 存在 = 这一组跑完了。Ctrl-C 打断时最坏的情况是
    多一个没有 metrics.json 的目录, 聚合会忽略它 —— 而不是一条指标只有一半的行。
    """
    cfg_dir = Path(cfg_dir)
    cfg_dir.mkdir(parents=True, exist_ok=True)
    np.save(cfg_dir / CONFIG_PHASE_NAME, np.asarray(phase_panel, dtype=np.float64))
    if pupil_phase is not None:
        np.save(cfg_dir / CONFIG_PUPIL_PHASE_NAME, np.asarray(pupil_phase, dtype=np.float64))
    if frame is not None:
        np.savez_compressed(
            cfg_dir / CONFIG_FRAME_NAME,
            frame=np.asarray(frame, dtype=np.float32),
            params=json.dumps(flatten_row(row), ensure_ascii=False),
        )
    write_json_atomic(cfg_dir / CONFIG_METRICS_NAME, row)


def write_stage_run_config(
    stage_dir: Path,
    *,
    stage: str,
    preflight: dict[str, Any],
    extra: dict[str, Any],
) -> None:
    """该段的 ``run_config.json`` (原子模式也写一份, 键名与整段模式同义)。

    为什么原子模式还要写这个「段级」文件: 既有报告脚本
    (``scripts/slm/generate_hw_gs_square_scans_report.py``) 从 ``run_config.json`` 读
    曝光/0 级/settle/拟合参数, 而 ``metrics.json`` 是逐配置的。把 preflight 的键**同时**
    摊平到顶层 (与整段模式一致) 再嵌一份完整的 ``preflight``, 两个读法都能用。

    多个原子进程会各自覆盖它 (``os.replace`` 原子), 所以它是「最近一次会话」的段级记录;
    逐配置的真实来源始终是 ``<config>/metrics.json``。
    """
    payload: dict[str, Any] = {
        "timestamp_utc": _utc_now_iso(),
        "stage": stage,
        "variable": STAGE_VARIABLES[stage],
        "mode": "atomic",
        **{k: v for k, v in preflight.items() if k != "timestamp_utc"},
        "preflight": preflight,
        **extra,
    }
    write_json_atomic(Path(stage_dir) / "run_config.json", payload)


def run_atomic(args: argparse.Namespace, bench: Bench, plan: dict[str, Any]) -> Path:
    """原子模式: 一个进程只跑**一组**参数, 产物并入 ``--run-root`` 的累积件。

    生命周期: open → pre-flight → 一组参数 → close。每组参数的 ``metrics.json`` 都带
    一份 ``preflight``, 所以「这张图是在什么条件下拍的」永远跟着数据走。
    """
    atomic = plan["atomic"]
    if not atomic:
        raise SystemExit("内部错误: run_atomic 需要 plan['atomic']")
    stage = str(atomic["stage"])
    value = int(atomic["value"])
    variable = STAGE_VARIABLES[stage]
    slug = config_slug(stage, value)

    run_root = require_accumulation_root(args)
    stage_dir = stage_dir_from_args(args, run_root, stage)
    cfg_dir = config_dir(stage_dir, stage, value)
    logger.info(f"[{stage} {slug}] 批次根 {run_root}")
    logger.info(f"[{stage} {slug}] 本配置目录 {cfg_dir}")

    # -- 依赖解析: 在开设备之前 (缺 selected.json 就不该占 USB 句柄) -----------
    selection: Selection | None = None
    base_pupil: np.ndarray | None = None
    base_source = ""
    base_info: dict[str, Any] = {}
    if stage in ("zernike", "blaze"):
        selection = load_selection(run_root, side_px_override=args.side_px)
        logger.info(
            f"[{stage} {slug}] 选定边长 {selection.side_px} px "
            f"(来源 {selection.source}"
            + (f", {selection.rank_by}={selection.score:.4f}" if selection.score is not None else "")
            + ")",

        )
        # 缓存命中与否是**离线判定**, 先试一次; 需要重算时等实测网格定了再做。
        if selection.gs_grid is not None:
            cached, base_info = load_cached_pupil_phase(
                run_root,
                selection.side_px,
                gs_grid=int(selection.gs_grid),
                phase_path=selection.phase_path,
            )
            if cached is not None:
                base_pupil, base_source = cached, "cache"
                logger.info(
                    f"[{stage} {slug}] 复用缓存基准相位 {base_info.get('loaded')} "
                    f"(形状 {cached.shape})",

                )

    declared_orders = parse_orders_spec(args.zernike_orders)
    declared_periods = tuple(int(p) for p in args.blaze_periods)
    max_shift = 0.0
    for period in declared_periods:
        dx, dy = predicted_shift_px(blaze_overlay(period, bench.wavelength_nm))
        max_shift = max(max_shift, float(math.hypot(dx, dy)))

    focus_panel = _focus_overlay_panel(args, bench)
    from ao_shaping.drivers.slm.santec import SantecError

    with open_bench(
        args, bench,
        focus_panel=focus_panel,
        abort_dir=cfg_dir,
        abort_context={"stage": stage, "config": slug, "run_root": str(run_root)},
    ) as sess:
        cx0, cy0 = sess.zero_order_xy
        side_max_raw, side_max = max_side_from_frame(
            sess.frame_shape, (cx0, cy0),
            margin_px=int(args.edge_margin_px), step_px=int(args.side_step),
        )
        logger.info(f"[{stage} {slug}] 最大边长 {side_max} px (原始 {side_max_raw} px)")

        gs_grid = atomic_gs_grid(
            args, bench,
            side_max_px=float(side_max),
            max_shift_px=max_shift,
            hint=(None if selection is None else selection.gs_grid),
        )
        gs_iters = int(
            args.gs_iters if selection is None or selection.gs_iters is None
            else selection.gs_iters
        )

        # -- 基准相位 (第 2/3 段) -----------------------------------------
        if stage in ("zernike", "blaze") and base_pupil is None:
            assert selection is not None
            base_pupil, base_source, base_info = _resolve_base_pupil(
                run_root, bench, selection, gs_grid=gs_grid, gs_iters=gs_iters
            )
        side_star = None if selection is None else int(selection.side_px)
        if stage == "sides" and int(value) > side_max:
            raise _fail(
                f"边长 {value} px 超过本次实测的画面可用最大边长 {side_max} px —— "
                "方框会压在帧边上, PIB/EE 就不再是方斑质量了。改小 --side-px, 或调 "
                "--edge-margin-px / 曝光",
                2,
            )
        if side_star is not None and side_star > side_max:
            raise _fail(
                f"选定边长 {side_star} 超过本次实测的画面可用最大边长 {side_max} "
                "(0 级位置或曝光与本批次不一致?)",
                2,
            )
        if base_pupil is not None and int(base_pupil.shape[0]) != int(gs_grid):
            # 缓存相位的形状就是它当初的 GS 网格 —— 以它为准, 否则产物里记的网格与实际
            # 相位不符 (那会让「这一行是用哪个网格算的」永远说不清)。
            logger.info(
                f"[{stage} {slug}] 缓存相位的网格是 {int(base_pupil.shape[0])}, "
                f"本次反解是 {int(gs_grid)}; 按缓存为准记录",

            )
            gs_grid = int(base_pupil.shape[0])

        step: dict[str, Any] = {
            variable: int(value), CONFIG_KEY: slug, "stage": stage,
            "gs_grid": int(gs_grid), "gs_iters": int(gs_iters),
        }
        pupil_phase: np.ndarray | None = None
        phase_panel = np.zeros(bench.panel_shape, dtype=np.float64)
        box_side = float(side_star if side_star is not None else value)
        box_center = (float(cx0), float(cy0))
        base_label = "gs"

        try:
            if stage == "sides":
                t0 = time.time()
                result = run_gs_square(
                    float(value), bench, n_grid=gs_grid, iterations=int(args.gs_iters)
                )
                pupil_phase = np.asarray(result.phase, dtype=np.float64)
                step["gs_seconds"] = float(time.time() - t0)
                step.update(model_box_metrics(result, float(value), bench))
                phase_panel = pupil_phase_to_panel(pupil_phase, bench)
            elif stage == "zernike":
                assert base_pupil is not None
                t_fit = time.time()
                fit, fit_source, fit_info = load_or_compute_fit(
                    run_root, base_pupil,
                    side_px=int(side_star),
                    fit_grid=int(args.zernike_fit_grid),
                    max_order=max(declared_orders),
                )
                # 本进程内这次拟合的墙钟 (命中缓存时 ≈ 0); 键名沿用整段模式的
                # ``fit_seconds``, 让既有报告脚本读得到。
                step["fit_seconds"] = float(time.time() - t_fit)
                logger.info(f"[{stage} {slug}] Zernike 拟合 {fit_source} "
                      f"(最高 {fit.max_order} 阶, {fit.coefficients.size} 系数)")
                t0 = time.time()
                recon, stats = reconstruct_from_fit(fit, int(value))
                pupil_phase = np.asarray(recon, dtype=np.float64)
                step.update(stats)
                step["reconstruct_seconds"] = float(time.time() - t0)
                step.update({
                    "selected_side_px": int(side_star),
                    "selected_source": selection.source if selection is not None else "?",
                    "base_phase_source": base_source,
                    "base_phase_info": base_info,
                    "fit_source": fit_source,
                    "fit_info": fit_info,
                    "fit_grid": int(fit.fit_grid),
                    "fit_max_order": int(fit.max_order),
                    "fit_coefficients_noll": [float(c) for c in fit.coefficients],
                })
                phase_panel = pupil_phase_to_panel(pupil_phase, bench)
            else:  # blaze
                assert base_pupil is not None
                base_label = str(args.blaze_base)
                base_used = base_pupil
                if base_label == "zernike-best":
                    picked = _best_zernike_row(args, run_root)
                    picked_order = int(as_float(picked.get("n_max")))
                    fit, fit_source, fit_info = load_or_compute_fit(
                        run_root, base_pupil,
                        side_px=int(side_star),
                        fit_grid=int(args.zernike_fit_grid),
                        max_order=max(int(max(declared_orders)), picked_order),
                    )
                    base_used, _ = reconstruct_from_fit(fit, picked_order)
                    base_label = f"zernike@{picked_order}阶"
                    step.update({
                        "blaze_base_picked_n_max": picked_order,
                        "blaze_base_picked_score": as_float(picked.get("score")),
                        "fit_source": fit_source,
                        "fit_info": fit_info,
                    })
                base_panel = pupil_phase_to_panel(base_used, bench)
                pupil_phase = np.asarray(base_used, dtype=np.float64)

                entry = blaze_shift_table(
                    (int(value),), bench.wavelength_nm, sess.frame_shape,
                    side_px=float(side_star), center=(float(cx0), float(cy0)),
                )[0]
                dx_pred = float(entry["predicted_shift_x_px"])
                dy_pred = float(entry["predicted_shift_y_px"])
                box_center = (
                    float(entry["box_center_predicted_x"]),
                    float(entry["box_center_predicted_y"]),
                )
                step.update({
                    "selected_side_px": int(side_star),
                    "selected_source": selection.source if selection is not None else "?",
                    "base_phase_source": base_source,
                    "base_phase_info": base_info,
                    "blaze_base": base_label,
                    "predicted_shift_x_px": dx_pred,
                    "predicted_shift_y_px": dy_pred,
                    "predicted_shift_px": float(entry["predicted_shift_px"]),
                    "box_center_predicted_x": box_center[0],
                    "box_center_predicted_y": box_center[1],
                    "box_clipped": not bool(entry["fits_frame"]),
                })
                if not entry["fits_frame"]:
                    logger.info(
                        f"[警告] 周期 {value} px 的预测位移 "
                        f"{entry['predicted_shift_px']:.0f} px 让方框出画面; "
                        "这一行的 PIB/EE 会偏低 (box_clipped=true)",

                    )
                # 叠加层只有一条通道 (焦面补偿在用), 所以光栅加进**模式相位**。
                phase_panel = apply_zero_order_overlay(
                    base_panel, blaze_overlay(float(value), bench.wavelength_nm)
                )

            frame = sess.capture_retry(phase_panel)
            metrics = measure_box(frame, box_center, box_side)
            row: dict[str, Any] = {
                "timestamp_utc": _utc_now_iso(),
                **step,
                "box_center_x": float(box_center[0]),
                "box_center_y": float(box_center[1]),
                "box_side_px": float(box_side),
                "exposure_ms_readback": float(read_exposure_ms(sess.cam)),
                "preflight": sess.preflight_dict(),
                **metrics,
            }
            if stage == "blaze":
                mx = float(row.get("box_centroid_x", float("nan")))
                my = float(row.get("box_centroid_y", float("nan")))
                row["measured_shift_x_px"] = (
                    (mx - float(cx0)) if np.isfinite(mx) else float("nan")
                )
                row["measured_shift_y_px"] = (
                    (my - float(cy0)) if np.isfinite(my) else float("nan")
                )
                row["shift_error_px"] = (
                    float(
                        math.hypot(
                            row["measured_shift_x_px"] - float(step["predicted_shift_x_px"]),
                            row["measured_shift_y_px"] - float(step["predicted_shift_y_px"]),
                        )
                    )
                    if np.isfinite(row["measured_shift_x_px"])
                    else float("nan")
                )
        except SantecError as exc:
            # 单组失败不该留下半个配置: 记一行失败 (指标 NaN), 仍然落盘并进索引。
            logger.info(f"[{stage} {slug}] 失败: {exc}")
            frame = None
            row = {
                "timestamp_utc": _utc_now_iso(),
                **step,
                "cv": float("nan"), "pib": float("nan"), "ee": float("nan"),
                "ar": float("nan"), "score": float("nan"), "quality": float("nan"),
                "error": str(exc),
                "preflight": sess.preflight_dict(),
            }

    write_config_artifacts(
        cfg_dir,
        phase_panel=phase_panel,
        pupil_phase=pupil_phase,
        frame=frame,
        row=row,
    )
    stage_extra: dict[str, Any]
    if stage == "sides":
        stage_extra = {
            "near_window_px": [
                float(args.target_side_near) - abs(float(args.near_tolerance)),
                float(args.target_side_near) + abs(float(args.near_tolerance)),
            ],
            "side_max_px": int(side_max),
            "side_max_raw_px": int(side_max_raw),
        }
    elif stage == "zernike":
        stage_extra = {
            "selected_side_px": int(side_star),
            "orders": [int(n) for n in declared_orders],
            "fit_grid": int(row.get("fit_grid", args.zernike_fit_grid)),
            "fit_max_order": int(row.get("fit_max_order", 0) or 0),
            "fit_seconds": float(row.get("fit_seconds", float("nan"))),
        }
    else:
        stage_extra = {
            "selected_side_px": int(side_star),
            "blaze_periods": [int(p) for p in declared_periods],
            "blaze_base": base_label,
        }
    write_stage_run_config(
        stage_dir,
        stage=stage,
        preflight=dict(row.get("preflight") or sess.preflight_dict()),
        extra=stage_extra,
    )
    rows = upsert_measurements(stage_dir, row, stage=stage)

    summary: dict[str, Any] = {
        "n_rows": len(rows),
        "mode": "atomic",
        "configs": [str(r.get(CONFIG_KEY, "")) for r in rows],
    }
    if side_star is not None:
        summary["selected_side_px"] = int(side_star)
        summary["selected_source"] = selection.source if selection is not None else "?"
    if stage == "blaze":
        summary["blaze_base"] = base_label
    try:
        dir_name = str(stage_dir.relative_to(run_root))
    except ValueError:
        dir_name = str(stage_dir)
    update_manifest_atomic(
        run_root,
        {
            **group_record(
                group=stage,
                variable=variable,
                meta={
                    "cam_type": str(args.cam_type),
                    "exposure_ms": str(args.exposure_ms),
                    "gs_iters": str(args.gs_iters),
                    "mode": "atomic",
                },
                dir_name=dir_name,
                n_rows=len(rows),
            ),
            "summary": summary,
            "dir_abs": str(stage_dir),
        },
    )

    logger.info(
        f"[{stage} {slug}] 完成: cv={as_float(row.get('cv')):.4g} "
        f"pib={as_float(row.get('pib')):.4g} score={as_float(row.get('score')):.4g}; "
        f"{stage_dir / CONFIG_CSV_NAME} 现在有 {len(rows)} 行",

    )
    return cfg_dir


# ---------------------------------------------------------------------------
# 离线子命令: select / summarize
# ---------------------------------------------------------------------------
def run_select(args: argparse.Namespace) -> Path:
    """从累积的 sides 测量里选点, 写 ``<run-root>/sides/selected.json``。

    选点规则与整段模式**完全同一套** (:func:`select_near_side`): 在
    ``--target-side-near ± --near-tolerance`` 内取 ``--rank-by`` 最大者; 窗口里一个点
    都没有就报错 (退出码 5), 绝不退化成全局最优。
    """
    run_root = require_accumulation_root(args)
    stage_dir = stage_dir_from_args(args, run_root, "sides")
    csv_path = stage_dir / CONFIG_CSV_NAME
    raw_rows = read_csv_rows(csv_path)
    rows = [
        {
            **r,
            "side_px": as_float(r.get("side_px")),
            str(args.rank_by): as_float(r.get(str(args.rank_by))),
        }
        for r in raw_rows
        if math.isfinite(as_float(r.get("side_px")))
    ]
    if not rows:
        raise _fail(
            f"{csv_path} 里没有任何带 side_px 的测量行 (退出码 {EXIT_NO_MEASUREMENTS}); "
            "先跑 --scan sides --side-px <px>",
            EXIT_NO_MEASUREMENTS,
        )
    try:
        best = select_near_side(
            rows,
            target_px=float(args.target_side_near),
            tolerance_px=float(args.near_tolerance),
            key=str(args.rank_by),
        )
    except NoNearSideError as exc:
        logger.info(f"\n[中止] {exc}")
        raise SystemExit(EXIT_NO_NEAR_SIDE) from exc

    side_px = int(as_float(best.get("side_px")))
    side_dir = config_dir(stage_dir, "sides", side_px)
    phase_file = side_dir / CONFIG_PUPIL_PHASE_NAME
    try:
        phase_path_text = phase_file.relative_to(run_root).as_posix()
    except ValueError:
        phase_path_text = str(phase_file)
    if not phase_file.is_file():
        logger.info(
            f"[警告] {phase_file} 不存在 (该边长可能是失败行); 第 2/3 段会重算 GS 并在 "
            "metrics.json 里记 base_phase_source=\"recomputed\"",

        )

    payload: dict[str, Any] = {
        "timestamp_utc": _utc_now_iso(),
        "stage": "sides",
        "side_px": side_px,
        "rank_by": str(args.rank_by),
        "rank_value": as_float(best.get(str(args.rank_by))),
        "score": as_float(best.get("score")),
        "quality": as_float(best.get("quality")),
        "cv": as_float(best.get("cv")),
        "pib": as_float(best.get("pib")),
        "target_side_near": float(args.target_side_near),
        "near_tolerance": float(args.near_tolerance),
        "near_window_px": [
            float(args.target_side_near) - abs(float(args.near_tolerance)),
            float(args.target_side_near) + abs(float(args.near_tolerance)),
        ],
        "gs_grid": (
            int(as_float(best.get("gs_grid"))) if math.isfinite(as_float(best.get("gs_grid")))
            else None
        ),
        "gs_iters": (
            int(as_float(best.get("gs_iters"))) if math.isfinite(as_float(best.get("gs_iters")))
            else None
        ),
        "n_sides_rows": len(rows),
        "phase_path": phase_path_text,
        "source_csv": (
            csv_path.relative_to(run_root).as_posix()
            if csv_path.is_relative_to(run_root) else str(csv_path)
        ),
        "grain": (
            "选定边长的基准相位 = 该边长 config 目录里的 pupil_phase.npy "
            "(GS 网格相位); 第 2/3 段优先复用它"
        ),
    }
    path = save_selection(run_root, payload)
    logger.info(
        f"[选定] 边长 {side_px} px (近 {args.target_side_near:.0f}±{args.near_tolerance:.0f}, "
        f"{args.rank_by}={payload['rank_value']:.4f}); 候选 {len(rows)} 行 → {path}",

    )
    return path


def load_stage_rows(stage_dir: Path, stage: str) -> list[dict[str, Any]]:
    """读该段所有**已完成**配置的 ``metrics.json`` (按自变量排序)。

    「已完成」= 那个目录里有 ``metrics.json``。:func:`write_config_artifacts` 最后才写
    它, 所以被打断的配置不会以半截数据混进来。
    """
    rows: list[dict[str, Any]] = []
    for path in sorted(Path(stage_dir).glob(f"*/{CONFIG_METRICS_NAME}")):
        data = read_json_file(path)
        if isinstance(data, dict) and STAGE_VARIABLES[stage] in data:
            rows.append(data)
    rows.sort(key=lambda r: _row_sort_key(r, STAGE_VARIABLES[stage]))
    return rows


def run_summarize(args: argparse.Namespace) -> Path:
    """重建每段 ``measurements.csv`` + 根 ``manifest.json`` + 每段 ``summary.png``。

    以**每个配置目录里的 metrics.json** 为唯一事实源 (原子进程逐个落盘的完整记录),
    所以「谁在哪一轮跑过什么」不依赖 CSV 的写入顺序, 也不怕某个进程被打断。
    """
    run_root = require_accumulation_root(args)
    total = 0
    for stage in STAGES:
        stage_dir = stage_dir_from_args(args, run_root, stage)
        if not stage_dir.is_dir():
            continue
        rows = load_stage_rows(stage_dir, stage)
        if not rows:
            continue
        write_csv_rows(stage_dir / CONFIG_CSV_NAME, [flatten_row(r) for r in rows])
        write_json_atomic(stage_dir / CONFIG_JSON_NAME, rows)
        _plot_stage(stage_dir, rows, stage)
        variable = STAGE_VARIABLES[stage]
        summary: dict[str, Any] = {
            "n_rows": len(rows),
            "mode": "atomic",
            "configs": [str(r.get(CONFIG_KEY, "")) for r in rows],
        }
        side_values = {
            int(as_float(r["selected_side_px"]))
            for r in rows if math.isfinite(as_float(r.get("selected_side_px")))
        }
        if side_values:
            summary["selected_side_px"] = sorted(side_values)
        try:
            dir_name = str(stage_dir.relative_to(run_root))
        except ValueError:
            dir_name = str(stage_dir)
        update_manifest_atomic(
            run_root,
            {
                **group_record(
                    group=stage,
                    variable=variable,
                    meta={"cam_type": str(args.cam_type), "mode": "atomic"},
                    dir_name=dir_name,
                    n_rows=len(rows),
                ),
                "summary": summary,
                "dir_abs": str(stage_dir),
            },
        )
        logger.info(
            f"[{stage}] {len(rows)} 个配置 → {stage_dir / CONFIG_CSV_NAME}"
            f" + summary.png",

        )
        total += len(rows)
    if total == 0:
        raise _fail(
            f"{run_root} 下没有任何已完成的配置产物 (退出码 {EXIT_NO_MEASUREMENTS})",
            EXIT_NO_MEASUREMENTS,
        )
    logger.info(f"根清单: {run_root / MANIFEST_NAME}")
    return run_root


# ---------------------------------------------------------------------------
# --plan-commands: 打印可复制的命令行 (每行一个进程)
# ---------------------------------------------------------------------------
#: 命令清单里的脚本入口 (相对仓库根, 逐行可复制)。
SCRIPT_COMMAND = "python scripts/slm/hw_gs_square_scans.py"

#: 一次 Zernike 拟合的量级估计 (s) —— 建 256²×496 的 RZern 表 + 最小二乘。
#:
#: **未在本机标定**, 只用于命令清单的**分批**建议 (差几秒不影响「每批几个进程」);
#: 实测后请把采集部分用 ``--estimate-capture-s`` 校准。
ESTIMATE_ZERNIKE_FIT_S = 30.0


def _fmt_num(value: Any) -> str:
    """``20.0`` → ``"20"``; 字符串原样返回 (命令清单里的参数要能直接粘贴)。"""
    if isinstance(value, str):
        return value
    try:
        return f"{float(value):g}"
    except (TypeError, ValueError):
        return str(value)


def shared_arg_tokens(args: argparse.Namespace, *, run_root_text: str) -> list[str]:
    """命令清单里每一行都要带上的「其它参数」—— 逐行复制即等价。

    只放**会影响测量**的参数: 相机/曝光/稳定判据/GS/网格/选点窗口/波长/SLM 编号/
    仿真后端/焦面补偿。计划假设类 (``--dry-run-*``) 不放 —— 它们只作用于离线计划。
    """
    tokens: list[str] = [
        "--run-root", run_root_text,
        "--cam-type", str(args.cam_type),
        "--cam-id", str(int(args.cam_id)),
        "--exposure-ms", _fmt_num(args.exposure_ms),
        "--frames", str(int(args.frames)),
        "--settle-n-discard", str(int(args.settle_n_discard)),
        "--settle-wait-s", _fmt_num(args.settle_wait_s),
        "--settle-tol", _fmt_num(args.settle_tol),
        "--settle-max-wait-s", _fmt_num(args.settle_max_wait_s),
        "--gs-iters", str(int(args.gs_iters)),
        "--gs-grid", _fmt_num(args.gs_grid),
        "--side-min", str(int(args.side_min)),
        "--side-step", str(int(args.side_step)),
        "--edge-margin-px", str(int(args.edge_margin_px)),
        "--target-side-near", _fmt_num(args.target_side_near),
        "--near-tolerance", _fmt_num(args.near_tolerance),
        "--rank-by", str(args.rank_by),
        "--zernike-orders", str(args.zernike_orders),
        "--zernike-fit-grid", str(int(args.zernike_fit_grid)),
        "--blaze-base", str(args.blaze_base),
        "--blaze-periods", *[str(int(p)) for p in args.blaze_periods],
        "--beam-radius-px", _fmt_num(args.beam_radius_px),
        "--wavelength-nm", _fmt_num(args.wavelength_nm),
        "--slm-number", str(int(args.slm_number)),
        "--seed", str(int(args.seed)),
        "--sim-seed", str(int(args.sim_seed)),
        "--sim-far-field-padding", str(int(args.sim_far_field_padding)),
        "--cam-size", str(int(args.cam_size)),
    ]
    if args.defocus_focal_mm is not None:
        tokens += ["--defocus-focal-mm", _fmt_num(args.defocus_focal_mm)]
    return tokens


def command_plan_lines(
    plan: dict[str, Any], args: argparse.Namespace
) -> list[str]:
    """完整命令清单: sides → select → zernike → blaze → summarize (按依赖顺序)。

    行数与配置数**逐值一致**: 边长 19 行 (100..1900 步长 100)、select 1 行、阶数 24 行
    (30..7)、周期 8 行、summarize 1 行。``--side-step`` 一改, 边长那一段的行数就跟着变。
    """
    shared = " ".join(shared_arg_tokens(args, run_root_text=str(plan["run_root"])))
    sides = [int(s) for s in plan["scan"]["sides"]]
    orders = [int(n) for n in plan["scan"]["orders"]]
    periods = [int(p) for p in plan["scan"]["blaze_periods"]]

    lines: list[str] = [
        f"{SCRIPT_COMMAND} --execute --scan sides --side-px {side} {shared}"
        for side in sides
    ]
    lines.append(f"{SCRIPT_COMMAND} --scan select {shared}")
    lines.extend(
        f"{SCRIPT_COMMAND} --execute --scan zernike --n-max {n_max} {shared}"
        for n_max in orders
    )
    lines.extend(
        f"{SCRIPT_COMMAND} --execute --scan blaze --blaze-period {period} {shared}"
        for period in periods
    )
    lines.append(f"{SCRIPT_COMMAND} --scan summarize {shared}")
    return lines


def print_command_plan(plan: dict[str, Any], args: argparse.Namespace) -> None:
    """打印命令清单 + 每段耗时估计 + 「按 10 分钟一批」的分批建议。

    ⚠️ 这里的耗时是**模型估计** (GS 用 ``GS_SECONDS_PER_ITER_UNIT``, 采集用
    ``--estimate-capture-s``, 拟合用 :data:`ESTIMATE_ZERNIKE_FIT_S`), 不是实测。
    """
    lines = command_plan_lines(plan, args)
    sides_n = len(plan["scan"]["sides"])
    orders_n = len(plan["scan"]["orders"])
    periods_n = len(plan["scan"]["blaze_periods"])
    gs_grid = int(plan["scan"]["gs_grid"])
    capture_s = float(plan["estimate"]["capture_s"])
    gs_each = float(plan["estimate"]["gs_seconds_each"])
    sides_each = gs_each + capture_s
    zernike_each = capture_s + ESTIMATE_ZERNIKE_FIT_S
    blaze_each = capture_s
    sides_total = sides_n * sides_each
    zernike_total = orders_n * zernike_each
    blaze_total = periods_n * blaze_each
    total = sides_total + zernike_total + blaze_total

    out: list[str] = []
    add = out.append
    add("=" * 78)
    add("命令清单 (每行一个进程; 按依赖顺序, 逐行复制即等价)")
    add("=" * 78)
    add("")
    add(f"  批次根         = {plan['run_root']}")
    add("  ⚠️ 边长/0 级是按 dry-run 假设算的 (--dry-run-frame-shape / --dry-run-zero-order);")
    add("     实测的 0 级位置由每个进程自己 argmax 量, 与这份清单无关。")
    add(f"  相机           = {plan['devices']['ccd_driver']} (cam_type={args.cam_type}, "
        f"cam_id={int(args.cam_id)}); 曝光请求 {_fmt_num(args.exposure_ms)} ms "
        "(硬件受 AO_DAHENG_MAX_EXPOSURE_MS 限制)")
    add(f"  GS             = N={gs_grid}, {int(args.gs_iters)} 迭代 "
        f"(约 {gs_each:.1f} s/次); 采集 {capture_s:.1f} s/配置")
    add("")
    add("  段             行数   每进程估计        小计")
    add(f"  sides         {sides_n:>5d}   {sides_each:>7.1f} s     {sides_total:>8.0f} s "
        f"({sides_total / 60:.1f} min)")
    add(f"  select        {1:>5d}   {0.0:>7.1f} s     {0.0:>8.0f} s (离线)")
    add(f"  zernike       {orders_n:>5d}   {zernike_each:>7.1f} s     {zernike_total:>8.0f} s "
        f"({zernike_total / 60:.1f} min; 含每次一次拟合 ≈ {ESTIMATE_ZERNIKE_FIT_S:.0f} s)")
    add(f"  blaze         {periods_n:>5d}   {blaze_each:>7.1f} s     {blaze_total:>8.0f} s "
        f"({blaze_total / 60:.1f} min)")
    add(f"  summarize     {1:>5d}   {0.0:>7.1f} s     {0.0:>8.0f} s (离线)")
    add(f"  合计          {len(lines):>5d}   ——              {total:>8.0f} s "
        f"({total / 3600:.2f} h)")
    add("")
    add("分批建议 (按 10 分钟一批; 一批 = 一个外部循环里的若干个进程, 串行跑):")

    def per_batch(each_s: float) -> int:
        """一批最多几个进程 (每批 ≤ PLAN_BATCH_S)。"""
        return max(1, int(PLAN_BATCH_S // max(float(each_s), 1e-9)))

    for name, each_s, count in (
        ("sides  ", sides_each, sides_n),
        ("zernike", zernike_each, orders_n),
        ("blaze  ", blaze_each, periods_n),
    ):
        batch = per_batch(each_s)
        add(f"  {name}  ≈ {batch:d} 行/批 ⇒ {math.ceil(count / batch):d} 批")
    add("  每批之间可以关掉进程/放掉 USB 句柄; 累积件 (measurements.csv / manifest.json)")
    add("  保证「跑到哪」随时可查, 中断只损失当前那一组。")
    add("")
    add("-" * 78)
    out.extend(lines)
    add("-" * 78)
    add("提示: 先用 --scan sides 那一段跑完, 再 --scan select, 之后才能跑 zernike / blaze")
    add("     (它们从 <run-root>/sides/selected.json 读选定边长与基准相位)。")
    add("=" * 78)
    logger.info("\n".join(out))


def print_atomic_plan(plan: dict[str, Any], args: argparse.Namespace) -> None:
    """原子模式的 dry-run: 只打印**这一个**配置的计划 (不列整段网格)。"""
    atomic = plan["atomic"]
    stage = str(atomic["stage"])
    value = int(atomic["value"])
    b = plan["bench"]
    s = plan["scan"]
    gs_grid = int(s["gs_grid"])
    capture_s = float(plan["estimate"]["capture_s"])
    gs_each = float(plan["estimate"]["gs_seconds_each"])
    run_root = Path(str(plan["run_root"]))
    stage_dir = stage_dir_from_args(args, run_root, stage)
    cfg_dir = config_dir(stage_dir, stage, value)
    lines: list[str] = []
    add = lines.append
    add("=" * 78)
    add(f"GS 方形光斑 —— 原子模式计划 (dry-run; 只跑这一组: {stage} = {value})")
    add("=" * 78)
    add("")
    add(f"  段 / 自变量         = {stage} / {atomic['variable']}")
    add(f"  本组配置            = {value}   ({config_slug(stage, value)})")
    add(f"  产物                = {cfg_dir}")
    add(f"  累积件              = {stage_dir / CONFIG_CSV_NAME} + "
        f"{run_root / MANIFEST_NAME}")
    add("")
    add("实验系统几何")
    add(f"  波长 λ             = {b['wavelength_nm']:.1f} nm; CCD 像元 {b['camera_pixel_um']} um")
    add(f"  照明 (面板坐标)     = 中心 {tuple(b['beam_center_panel'])}, 半径 "
        f"{b['beam_radius_px']:.0f} px ⇒ 包围方框 {b['pupil_side_px']} px")
    add(f"  远场标度           = {s['camera_px_per_gs_px']:.4f} camera px / GS 网格像素")
    add("")

    if stage == "sides":
        side_gs = value / s["camera_px_per_gs_px"]
        add("这一组")
        add(f"  目标边长           = {value} camera px ⇒ side_gs_px = "
            f"{side_gs:.1f} 网格像素")
        add(f"  GS 网格 N          = {gs_grid} (远场窗口 "
            f"{gs_grid * s['camera_px_per_gs_px']:.0f} camera px)")
        add(f"  GS 迭代            = {int(args.gs_iters)} 次 ≈ {gs_each:.1f} s")
    elif stage == "zernike":
        add("这一组")
        add(f"  阶数 n_max         = {value} ⇒ {modes_for_order(value)} 个模式 (含 piston)")
        add(f"  拟合网格           = {int(args.zernike_fit_grid)}; 拟合最高 "
            f"{max(int(n) for n in s['orders'])} 阶 "
            f"({modes_for_order(max(int(n) for n in s['orders']))} 模式)")
        add("  基准相位           = <run-root>/sides/<slug>/pupil_phase.npy "
            "(命中则 base_phase_source=cache, 否则重算 GS)")
        add("  依赖               = <run-root>/sides/selected.json "
            "(缺失时退出码 6, 提示先跑 --scan select)")
    else:
        overlay = blaze_overlay(float(value), b["wavelength_nm"])
        dx, dy = predicted_shift_px(overlay)
        add("这一组")
        add(f"  光栅周期           = {value} 面板 px (方向 vertical: 面板 x 斜坡 → 相机 y)")
        add(f"  预测位移           = ({dx:+.1f}, {dy:+.1f}) camera px "
            f"= {math.hypot(dx, dy):.1f} px")
        add("  框                 = 冻结 0 级 + 预测位移 (方框随斑走)")
        add("  依赖               = <run-root>/sides/selected.json "
            "(缺失时退出码 6, 提示先跑 --scan select)")
    add("")
    add("设备与采集")
    d = plan["devices"]
    add(f"  SLM                = {d['slm_driver']}")
    add(f"  CCD                = {d['ccd_driver']} (cam_id {int(args.cam_id)})")
    add(f"  曝光请求           = {_fmt_num(args.exposure_ms)} ms "
        "(上限遵守 AO_DAHENG_MAX_EXPOSURE_MS)")
    add(f"  每配置平均帧数      = {int(args.frames)} (+{int(args.settle_n_discard)} 丢弃帧)")
    if plan["is_sim"]:
        add(f"  仿真               = seed {int(args.sim_seed)}, 远场补零 "
            f"{int(args.sim_far_field_padding)}, 窗口 {int(args.cam_size)} sim px")
    add("")
    est = capture_s + (gs_each if stage == "sides" else 0.0)
    if stage == "zernike":
        est += ESTIMATE_ZERNIKE_FIT_S
    add("预计耗时 (单进程)")
    add(f"  {est:.1f} s (采集 {capture_s:.1f} s"
        + (f" + GS {gs_each:.1f} s" if stage == "sides" else "")
        + (f" + 拟合 ≈ {ESTIMATE_ZERNIKE_FIT_S:.0f} s" if stage == "zernike" else "")
        + "); pre-flight 另计")
    add("")
    add("下一步")
    add(f"  {SCRIPT_COMMAND} --execute --scan {stage} "
        f"{STAGE_VALUE_FLAG[stage]} {value} --run-root {plan['run_root']} ...")
    add("  (完整参数集见 --plan-commands 打印的每一行)")
    add("=" * 78)
    logger.info("\n".join(lines))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="GS 方形光斑三段扫描 (边长 / Zernike 阶数 / 闪耀光栅周期), 默认 dry-run"
    )
    p.add_argument("--dry-run", action="store_true",
                   help="只打印实验计划 (默认行为); 绝不 import / open 任何设备")
    p.add_argument("--execute", action="store_true",
                   help="真正开设备执行 —— 这是唯一会 import 驱动包的分支")
    p.add_argument("--scan", default="all",
                   choices=("sides", "zernike", "blaze", "all", "select", "summarize"),
                   help="跑哪一段; 原子模式 = 只跑一组 (配 --side-px / --n-max / "
                        "--blaze-period); sides/zernike/blaze 不给值选择器时跑完整段 "
                        "(仅调试用, 推荐原子模式 + --plan-commands); "
                        "select / summarize 是离线子命令 (不打开设备)")
    p.add_argument("--plan-commands", action="store_true",
                   help="只打印命令清单 (每行一个进程, 按依赖顺序), 不跑设备 —— "
                        "交给外部循环逐个执行")
    p.add_argument("--cam-type", default="daheng", choices=("daheng", "miicam", "sim"),
                   help="相机后端 (sim = 数字孪生, 不碰硬件)")
    p.add_argument("--cam-id", type=int, default=0, help="相机编号 (驱动按下标取)")
    p.add_argument("--cam-size", type=int, default=2500,
                   help="仿真窗口边长 (sim px); 硬件忽略 —— 最大边长要由全画幅边缘算")
    p.add_argument("--exposure-ms", type=float, default=0.2,
                   help="请求曝光 (ms); 硬件会被 AO_DAHENG_MAX_EXPOSURE_MS 限制")
    p.add_argument("--frames", type=int, default=4,
                   help="每配置平均的帧数 (bench_kernels.display_and_average)")
    p.add_argument("--settle-n-discard", type=int, default=3,
                   help="稳定性判据之前的丢弃帧数 (bench_kernels 默认 3)")
    p.add_argument("--settle-wait-s", type=float, default=0.5,
                   help="下发后驱动级等待 (s)")
    p.add_argument("--settle-tol", type=float, default=0.02,
                   help="稳定性判据的相对容差 (bench_kernels 默认 0.02)")
    p.add_argument("--settle-max-wait-s", type=float, default=6.0,
                   help="等待稳定的上限 (s)")
    p.add_argument("--gs-iters", type=int, default=200, help="GS 迭代次数")
    p.add_argument("--gs-grid", default="auto",
                   help="GS 网格边长: auto (按远场窗口反解) 或显式整数")
    p.add_argument("--side-min", type=int, default=100, help="边长扫描起点 (相机 px)")
    p.add_argument("--side-step", type=int, default=100, help="边长扫描步长 (相机 px)")
    p.add_argument("--side-max", default="auto",
                   help="边长扫描上限: auto (画面边缘反解) 或显式 px")
    p.add_argument("--side-px", type=int, default=None,
                   help="原子模式: --scan sides 时只跑这一个边长 (相机 px); "
                        "--scan zernike/blaze 时是**显式覆盖**选定边长 (旧行为, 逃生口: "
                        "那时 selected_source=cli, 不再要求 selected.json)")
    p.add_argument("--n-max", type=int, default=None,
                   help="原子模式: --scan zernike 时只跑这一个阶数 (1..30)")
    p.add_argument("--blaze-period", type=int, default=None,
                   help="原子模式: --scan blaze 时只跑这一个光栅周期 (面板 px)")
    p.add_argument("--edge-margin-px", type=int, default=TARGET_EDGE_MARGIN_PX,
                   help=f"最大边长扣掉的边缘留白 (默认 {TARGET_EDGE_MARGIN_PX})")
    p.add_argument("--target-side-near", type=float, default=1000.0,
                   help="第 1 段选点目标边长 (相机 px)")
    p.add_argument("--near-tolerance", type=float, default=DEFAULT_NEAR_TOLERANCE_PX,
                   help="选点窗口半宽 (相机 px)")
    p.add_argument("--rank-by", default="score", choices=("score", "quality"),
                   help="第 1 段选点依据 (score = 复合分数; quality = 方形整形评分)")
    p.add_argument("--zernike-orders", default=DEFAULT_ZERNIKE_ORDERS,
                   help=f"阶数规格: '7..30' / '7,9,11' (默认 {DEFAULT_ZERNIKE_ORDERS})")
    p.add_argument("--zernike-fit-grid", type=int, default=DEFAULT_ZERNIKE_FIT_GRID,
                   help=f"Zernike 拟合网格边长 (默认 {DEFAULT_ZERNIKE_FIT_GRID}; "
                        "RZern 内存 ∝ G²·模式数, 30 阶下 G=900 会是 3.2 GB)")
    p.add_argument("--blaze-periods", nargs="+", type=int,
                   default=list(DEFAULT_BLAZE_PERIODS),
                   help=f"闪耀光栅周期 (面板 px, 默认 {DEFAULT_BLAZE_PERIODS})")
    p.add_argument("--blaze-base", default="gs", choices=("gs", "zernike-best"),
                   help="第 3 段的基底相位: gs (选定边长的 GS 相位, 默认) 或 "
                        "zernike-best (第 2 段得分最高的阶数重构)")
    p.add_argument("--defocus-focal-mm", type=float, default=None,
                   help="瞳面菲涅尔焦距 (mm): 不给 = 不补偿 (CCD 已在焦面); "
                        "inf = 显式平场")
    p.add_argument("--beam-radius-px", type=float, default=float(BEAM_RADIUS_PANEL),
                   help=f"照明半径 (面板 px, 默认 {BEAM_RADIUS_PANEL})")
    p.add_argument("--wavelength-nm", type=float, default=0.0,
                   help="波长 (nm); 0 = 读驱动包里的 DEFAULT_WAVELENGTH_NM (不 import)")
    p.add_argument("--slm-number", type=int, default=1, help="SLM 编号")
    p.add_argument("--seed", type=int, default=0,
                   help="随机种子 (只影响仿真噪声; GS 初值 = 目标反向传播, 无随机数)")
    p.add_argument("--sim-seed", type=int, default=0, help="仿真噪声模型的种子")
    p.add_argument("--sim-far-field-padding", type=int, default=4,
                   help="仿真远场补零因子 (4 = 复现实测角标度的值)")
    p.add_argument("--out-root", default=None,
                   help=f"产物根目录 (默认 {DEFAULT_OUT_PARENT}/<UTC 时间戳>)。"
                        "只给 --out-root 时它就是批次根 (旧语义); 与 --run-root 同给时, "
                        "原子模式下它覆盖该段目录 (<run-root>/<type> 的替换)")
    p.add_argument("--run-root", default=None,
                   help="批次根目录 (跨进程累积: <根>/<段>/<配置>/ + <根>/<段>/"
                        "measurements.csv + <根>/manifest.json)。原子模式与 select/"
                        "summarize 必须给 (不给就报错, 免得每个进程各取一个时间戳)")
    p.add_argument("--dry-run-frame-shape", default="1944x2592",
                   help="dry-run 计划里假设的画幅 (HxW, 默认 Daheng 1944x2592)")
    p.add_argument("--dry-run-zero-order", default=None,
                   help="dry-run 计划里假设的 0 级位置 X,Y (默认取画幅中心; "
                        "实测一律在平场帧上 argmax 现场量)")
    p.add_argument("--estimate-capture-s", type=float, default=DEFAULT_ESTIMATE_CAPTURE_S,
                   help=f"每配置采集耗时估计 (s, 默认 {DEFAULT_ESTIMATE_CAPTURE_S})")
    return p


def main() -> None:
    args = build_parser().parse_args()
    bench = build_bench(args)  # 顺带把 --wavelength-nm 0 解析成设备常量

    # -- 离线子命令: 不打开任何设备, 也不 import 驱动包 ----------------------
    if args.plan_commands:
        # 命令清单要**完整网格**, 所以内部按 --scan all 生成 (与本次 --scan 无关)。
        plan_args = argparse.Namespace(**vars(args))
        plan_args.scan = "all"
        plan_args.side_px = None
        plan_args.n_max = None
        plan_args.blaze_period = None
        try:
            plan = build_plan(plan_args, bench)
        except ValueError as exc:
            raise SystemExit(f"参数错误: {exc}") from exc
        print_command_plan(plan, args)
        assert_no_drivers()
        return

    scan = str(args.scan).strip().lower()
    if scan in OFFLINE_SCANS:
        if args.execute:
            logger.info(
                f"--scan {scan} 是离线子命令, 不开设备; 已忽略 --execute",

            )
        if scan == "select":
            run_select(args)
        else:
            run_summarize(args)
        assert_no_drivers()
        return

    try:
        plan = build_plan(args, bench)
    except ValueError as exc:
        raise SystemExit(f"参数错误: {exc}") from exc

    atomic = plan["atomic"]
    if args.execute and not args.dry_run:
        if atomic:
            # 原子进程本来就只跑一组: 只有该组本身超预算才值得警告。
            per_config = float(plan["estimate"]["gs_seconds_each"]) + float(
                plan["estimate"]["capture_s"]
            )
            if per_config > PLAN_LIMIT_S:
                logger.info(
                    f"[警告] 单组预计 {per_config:.0f}s 已超过 {PLAN_LIMIT_S:.0f}s 预算 "
                    "(GS 网格偏大?), 但原子模式无法再拆",

                )
            try:
                run_atomic(args, bench, plan)
            except SelectionMissing as exc:
                # 缺选点 = 该先跑 --scan select, 不是去查光路/驱动 (退出码 6)。
                logger.info(f"\n[中止] {exc}")
                raise SystemExit(EXIT_SELECTION_MISSING) from exc
            return
        if float(plan["estimate"]["total_s"]) > PLAN_LIMIT_S * len(plan["stages"]):
            logger.info(
                "[警告] 预计总耗时超过预算 —— 整段模式一次跑完, 建议改用原子模式 "
                "(见 --plan-commands)",

            )
        run_execute(args, bench, plan)
        return

    if args.execute and args.dry_run:
        logger.info("给了 --dry-run 和 --execute, 按 dry-run 处理")
    if atomic:
        print_atomic_plan(plan, args)
    else:
        print_plan(plan)
    assert_no_drivers()


if __name__ == "__main__":
    main()
