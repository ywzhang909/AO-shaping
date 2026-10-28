"""硬件实验探针: 瞳面二次相位模拟「CCD 焦后 Δz」在真实实验系统上的验证 (默认 dry-run)。

要验证什么
----------

把 CCD 沿光轴从焦面往后挪 Δz, 物理真值是 ``U_CCD = ASM(FT(U_SLM), Δz)``。硬件上
**CCD 挪不动**, 只能反过来在瞳面乘一个二次相位
``exp(s·i·α·π·Δz·r²/(λf²))`` —— 这在傍轴极限下与真值等价, 误差按 Δz² 增长。
本脚本在真实实验系统上逐点验证这条近似, 也就是:

* Δz = 0 时 CCD 在焦面 (基线, 也用来定位零级与冻结 ROI);
* 逐个 Δz: **读出 GS 相位 → 用 :meth:`Santec.set_phase_overlay` 叠加对应的瞳面二次相位
  → 等稳定 → 抓帧 → 算 CV/PIB/score → 落盘**;
* 仿真侧的预测见 ``scripts/slm/experiment_gs_defocus_sweep_visuals.py`` 的
  ``scheme_gap_dz0_50mm*.csv``, 本脚本的实测就是去对拍它。

🔴 **本脚本默认 ``--dry-run``: 只打印完整实验计划, 绝不 import 或 open 任何设备。**

不给 ``--execute`` 时本脚本:

* 只 import 三个**零导入叶子** (:mod:`ao_shaping.utils.slm.bench_constants`、
  :mod:`ao_shaping.utils.wavefront.matrix_utils` 与
  :mod:`ao_shaping.tools.kernels.bench_kernels`) —— 实测它们**不牵入任何
  ``ao_shaping.drivers.*``**;
* 波长常量 ``DEFAULT_WAVELENGTH_NM`` 用 ``ast`` 从 ``drivers/slm/factory.py``
  **静态读取** (不 import), 保持单一事实源又不触发驱动包的 import 期副作用;
* 收尾时**断言** ``ao_shaping.drivers`` 没出现在 ``sys.modules`` 里 —— 这条断言
  让「dry-run 不碰设备」成为可执行的保证而不是口头承诺, 违反就 exit 3。

因此 ``--dry-run`` 在**没有任何硬件的机器上**也必须 exit 0。

🔴 **扫描前先自检光路 (pre-flight)**: 基准帧 (Δz=0, GS 相位已施加) 一抓到就判
一次「有没有可评分的局域光斑」。判据看**结构**而不是亮度 —— 光路被挡住时画面里
只有读出噪声, 而噪声帧照样能算出 CV/PIB/光斑, 于是本探针会安安静静地记下 30 行
**纯噪声** 当成测量结果 (2026-10-21 实测)。所以:

* 判为**光路受阻** ⇒ **中止整轮** (退出码 4), 不写 ``measurements.csv``, 只写一份
  带判定与理由的 ``run_config.json``, 并打印三段光路的检查清单;
* 只是**没聚焦** (光到了但光斑太宽) ⇒ **警告并继续**, 那一轮仍然有意义。

退出码: 0 = 正常; 2 = 读帧超时 (:func:`guarded` 硬退出); 3 = dry-run 违约
(:func:`assert_no_drivers`); 4 = 光路受阻。

🔴 **不要按 M=(f+d)/f 重采样。** 有外部材料主张 CCD 挪到 f+d 时图样被放大 ``M``、
target 要按 ``M`` 重采样 —— **对 2f 光路是错的**, 错在漏掉了透镜。Collins/ABCD::

    T(f+d) · Lens(f) · T(f) = [[−d/f, f], [−1/f, 0]]      → A = −d/f, **B ≡ f**

核是 ``e^{−ikxξ/B}`` 而 ``B`` 恒等于 ``f`` ⇒ 核尺度是 ``λf`` 而非 ``λ(f+d)``, **没有
放大**。CCD 离焦只让图样**变模糊**: 单个倾斜平面波经透镜聚到 fθ 之后只是发散, 中心
不跑。本脚本因此不缩放 ROI、不按 d 重采样目标框 (目标框边长在扫描开始前定一次就
**冻结**), 瞳面二次相位的分母取 **λf²**。

🟡 **目标框按被探的那张相位定尺 (``--target-side-px 0`` = 默认)**

目标框的边长默认**不是**写死的: 它由**本次相位自己的** Δz=0 远场光斑量出来
(``side = clamp(round(FWHM), 4, floor(0.6·ROI))``), **只量一次**然后**冻结**,
基线行与所有扫描行共用同一个框, 中心一直是冻结的零级。

为什么不能写死: 历史上写死的 44 px 是**模型** GS 相位方形边长换算到硬件 CCD
像元上的值, 而硬件 GS 相位实际打出的光斑只有 **12.9 px** 宽。框比光斑大 3.4 倍时,
离焦弥散会把整个框**填满**, 均匀度项 ``1/(1+CV)`` 反而**上升** (CV 1.733 → 1.157),
恰好抵消功率项 PIB 的下降, 合成分 ``0.5·PIB + 0.5/(1+CV)`` 于是**看起来是平的** ——
离焦响应被两个项的抵消抹掉了。按相位自己的光斑定尺后响应恢复单调。

⚠️ **框只定一次**: 逐帧按光斑重定尺会把离焦弥散本身藏起来 (弥散越大 → 框越大 →
框内越均匀), 那正是仓库记下的自适应框刷分陷阱。

🟡 **瞳面斜坡去倾斜 (opt-in, 默认关闭)**

瞳面二次相位以面板的**标称**照明中心 (960, 600) 为中心, 而真实照明并不在那里 ——
偏心的二次相位等价于一片**偏心透镜**: 它既离焦, 又**转向** (实测 2026-10-20 组
``dz_scan_hw_gs``: 远场质心随 Δz 走 **-5.2 camera px/mm**)。这个转向是**模拟**的
性质 —— 对拍的仿真把二次相位对准**光束**中心, 所以它没有转向分量。于是冻结的目标框
会随 Δz 挪走, 量到的变成转向而不是离焦, 硬件与仿真也不同口径。

``--chirp-tilt-cancel`` 打开**现场自标定**的去倾斜: 先测出「面板斜坡 → 相机位移」
的映射与**符号** (本实验系统面板/相机轴交换 90°, 而这个符号仓库里没记, 所以只能
现场量), 再逐 Δz 叠一条瞳面斜坡把转向补掉, 最后在标定 Δz 上复测残差。补偿斜坡与
二次相位**相加**进同一张叠加图 (叠加层只有一条通道)。任何一步量不到可用光斑, 或实测
标度与 bench 常量的比值越界, 一律**放弃去倾斜**并按原来的纯二次相位继续 —— 标定失败
不许让整轮扫描作废。不给这个开关时, 行为与引入本选项之前**逐位一致**。

🟡 **零级背景 (硬件侧最该量的东西)**

真实 SLM 的入射场到瞳面是**双分量** ``U = U_inc·[η + (1−η)e^{iφ}]``。零级那一项
``η·U_inc`` 只经过物理透镜, 在 f+d 处散成晕斑, 半径约 ``ρ ≈ d·D/(2f)``, 峰值按
``I_halo ∝ f²/d²`` 下降。仿真侧的预测见 ``halo_zero_order_visuals*.csv``。

所以本探针每次测量都额外记三个量: 整帧峰值、框外峰值、框外功率占比。**注意口径**:
Δz 小时零级峰仍在目标框内 (与整形方形重叠), 这时框外峰值量不到零级, 要看整帧峰值;
Δz 大到零级散出框后, 框外峰值才代表污染。框外功率占比还混了边瓣与焦外弥散, 是**上界**。

硬件防呆 (全部来自仓库已记的坑)
--------------------------------

* **绝不用 DVI 模式** (``video_mode=0``, 驱动默认值; ``open()`` 在 DVI 下会挂
  120s/300s, 且挂起后 memory 模式也挂, 只能物理断电重置)。
* **同内存槽连续 display 是 no-op**: 每次写入都从 ``2..125`` 里**随机**挑一个未占用
  的槽, 并排除运行开始前正在显示的那个 (``get_displayed_memory_number()``;
  它在 ``set_grayscale`` 模式下报错误码 1, 那是**正常**的, 不是故障)。
* **零级必须 ``argmax`` 定位**, 绝不假设在画面中心 (2f 实验系统上画面中心 (1344,760)
  vs 零级 (1441,705) 是实测值)。**定位一次就冻结**: 之后不逐帧重定位, 否则类散斑场
  上 ``argmax`` 会在近等晶粒间跳, 让 CV/PIB 不连续。
* **曝光上限**遵守 ``AO_DAHENG_MAX_EXPOSURE_MS``; 未设置时由画面探测 (倍增阶梯到饱和)。
* **必须减背景再算指标**: :func:`~ao_shaping.optimizer.wfless._measure.display_and_average`
  返回的是 ``prepare_frame`` (减中位数 + 裁负) 的结果, 不是原始帧 —— 否则读出噪声会把
  PIB 顶到 1 以上。
* **叠加层走 :meth:`Santec.set_phase_overlay`** (弧度制, 在 ``create_phase_from_array``
  内部、弧度→灰度**之前**相加), 所以矫正/LUT/平移照常作用于合成相位 —— 这是驱动层
  明确支持的安全位置, 不必自己拼灰度图。

运行
----

.. code-block:: bash

    # 无硬件机器上也能跑: 只打印计划
    python scripts/slm/hw_gs_defocus_probe.py --dry-run

    # 装好设备后
    python scripts/slm/hw_gs_defocus_probe.py --execute
    python scripts/slm/hw_gs_defocus_probe.py --replot <测量目录>   # 离线重新出图

    # 打开去倾斜 (把二次相位的转向补掉; 默认关闭)
    python scripts/slm/hw_gs_defocus_probe.py --execute --chirp-tilt-cancel \
        --chirp-tilt-cal-dz-mm 2.0 --chirp-tilt-test-period-px 1000

同一套 CLI 跑仿真
-----------------

``--cam-type sim`` 让本脚本在**数字孪生**上跑出**同结构**的产物
(``run_config.json`` / ``measurements.csv`` / ``measurements.json`` /
``frame_*.npz`` / ``hw_summary.png`` / 根目录 ``manifest.json``), 不碰任何硬件:

.. code-block:: bash

    python scripts/slm/hw_gs_defocus_probe.py --execute --cam-type sim \
        --dz-mm 0 5 --sim-seed 0 --sim-far-field-padding 4

* ``--sim-seed``: 仿真噪声模型的种子 (固定它 ⇒ 逐位可重复);
* ``--sim-far-field-padding``: 仿真远场 FFT 的补零因子, 与
  ``scripts/slm/run_gs_refine_zernike_sweep.py`` 的同名选项**同名同义**
  (默认 16; 代价按补零的平方增长, 冒烟测试用 4 就够)。

仿真里**离焦是真的**: 瞳面二次相位经 ``set_phase_overlay`` 在弧度→灰度**之前**
加进面板相位, 远场看到的是相加后的相位, 所以光斑会随 Δz 变糊、框内功率会掉。
仿真的最佳焦面由**仿真的几何**决定, 与实测值无关 —— 产物里的驱动字符串也会
写成 ``SimSLMPib`` / ``create_camera('sim')``, 不会与硬件产物混淆。

一次只探一组变量
----------------

本探针是「Δz (离焦间距, mm) = 自变量」这一组。一次调用只探**这一组**:

* ``--group NAME`` —— 这一组的身份 (默认 ``main``); 落盘目录是 ``<根目录>/<组名>/``。
* ``--group-meta KEY=VALUE`` —— 这一组的**固定**参数 (可重复), 逐字写进组目录的
  ``run_config.json`` 与根目录的 ``manifest.json``。例如
  ``--group-meta exposure=0.2 --group-meta focus_focal_mm=1597.4``。
* 这一组扫的变量名固定是 ``dz_mm`` (写进清单的 ``variable`` 字段)。

根目录默认是 ``report/shape_trials/hw_gs_defocus_probe_<UTC 时间戳>/``, 可以用
``--out-root`` 指定 —— **同一轮实验的多个组要共用同一个根目录**, 根目录下的
``manifest.json`` 才会把它们都索引起来 (每组写完立刻更新, 后一组崩掉不影响前一组
已被索引)。

产物写到 ``<根目录>/<组名>/``: 每次测量一个 ``.npz`` (原始帧 + 全部参数)、一份
``measurements.csv``、一份 ``run_config.json``、一张 ``hw_summary.png``。根目录另有一份
``manifest.json`` 列出本根目录下记录过的每一组。npz 里存的是**未经任何处理**的原始帧,
所以离线可以换度量口径重新出图, 不必重连硬件。``--replot`` 接受**根目录**或**组目录**:
给根目录时会把下面每一组都重出图。
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import sys
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
# `scripts.*` is imported as a package path, so the repo ROOT itself must be
# importable -- under a file-form launch (`python scripts/x/y.py`) the cwd is
# not on sys.path and only the script's own directory is.
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

# ---------------------------------------------------------------------------
# 光学常量: 全部来自**零导入叶子**, 导入它们不会牵入 ao_shaping.drivers.*
# (实测: sys.modules 里没有任何 ao_shaping.drivers 出现)。单一事实源仍是
# bench_constants / matrix_utils, 本文件不另立一份。
# ---------------------------------------------------------------------------
from ao_shaping.utils.image.frame_analysis import (
    SpotUsability,
    assess_spot,
    measure_spot,
)
from ao_shaping.utils.slm.bench_constants import (
    BEAM_CENTER_PANEL,
    BEAM_RADIUS_PANEL,
    SLM_PANEL_H,
    SLM_PANEL_W,
    SLM_PITCH_M,
    TILT_SHIFT_SCALE,
)
from ao_shaping.utils.wavefront.matrix_utils import (
    focal_length_from_camera_pixel,
)
from ao_shaping.tools.kernels.bench_kernels import (
    fresnel_panel,
    ramp_panel,
)
from scripts._common.group_manifest import (
    group_output_dir,
    group_record,
    load_manifest,
    parse_group_meta,
    update_manifest,
)

#: 这一组探针扫的变量名 —— 写进分组清单的 ``variable`` 字段。
#: 离焦这一组的自变量是 Δz (mm); 泽尼克那一组扫的是 ``n_max``。
GROUP_VARIABLE = "dz_mm"

#: 2f 焦距标度是在哪个波长下测的 (bench_constants 的 TILT_SHIFT_SCALE 是 1064 nm 下测的)
SCALE_MEASURED_AT_NM = 1064.0
#: CCD 像元 (Daheng MER2-507-23GM, datasheet 常数, 见 drivers/AGENTS.md)
CAMERA_PIXEL_UM = 2.2

#: 瞳面二次相位的符号/前因子。**不是照抄公式**: 仿真脚本
#: ``experiment_gs_defocus_sweep_visuals.py`` 用方案A 的 |U_CCD|^2 数值一致性比对反解
#: (完整网格, 探针 Δz=0.1876mm):
#:   sign = -1 → 相对误差 4.46e-6 ; sign = +1 → 1.65e-3 (差 ~370 倍, 符号唯一)
#:   再对 α 做黄金分割 → alpha = 1.00016 (理论 1, 残差降到 2.24e-7)
#: 硬件侧沿用同一组值, 并可用 CLI 覆盖做复核。
CHIRP_SIGN = -1.0
CHIRP_ALPHA = 1.0001626284163687

#: 默认扫描网格 (mm)。**与仿真脚本的 ΔZ_SCAN_MM 逐点一致**, 两个脚本才能对拍。
DZ_SCAN_MM: tuple[float, ...] = tuple(float(v) for v in range(0, 55, 5))

#: ``--target-side-px`` 的默认值 = **自动定尺**。边长由**本次相位自己的** Δz=0
#: 远场光斑量出来 (见 :func:`resolve_probe_target_side`), 量一次就**冻结**。
#:
#: 🔴 **为什么不能写死一个框**: 历史上写死的 44 px 是**模型** GS 相位的方形边长
#: 换算到硬件 CCD 像元上的值 (43 远场 px × 2.2606 µm/px = 97.2 µm, 换到 2.2 µm
#: 像元上是 44.2 px)。但硬件 GS 相位实际打出的光斑只有 **12.9 px** 宽 —— 框比光斑
#: 大 3.4 倍时, 离焦弥散会把整个框**填满**, 均匀度项 ``1/(1+CV)`` 于是**上升**
#: (CV 1.733 → 1.157), 恰好抵消功率项 PIB 的下降, 合成分
#: ``0.5·PIB + 0.5/(1+CV)`` 于是**看起来是平的**。把框定到相位自己的光斑尺度上,
#: 离焦响应才恢复成单调 (见 ``report/shape_trials/gs_defocus_sim_20261019.md`` §7)。
AUTO_TARGET_SIDE_PX = 0

#: 自动定尺测不出光斑时的兜底边长 (px) —— 即历史上的写死值。
FALLBACK_TARGET_SIDE_PX = 44

#: 自动定尺的夹取区间。下界: 边长太小则框内像素太少, CV 完全被噪声支配;
#: 上界: 框不能超过 ROI 的这个比例, 否则框会被 ROI 边界裁掉一半。
AUTO_TARGET_SIDE_MIN_PX = 4
AUTO_TARGET_SIDE_ROI_FRACTION = 0.6

DEFAULT_ROI_PX = 250

# ---------------------------------------------------------------------------
# 后端模式 → 驱动字符串
# ---------------------------------------------------------------------------
#: 硬件后端的 SLM 驱动串 —— **逐字不变**: 第三轮已入库的产物
#: (``report/shape_trials/hw_defocus_zernike/...``) 里就是这个字符串, 复现与
#: 对拍都按它认后端, 所以**不许**改写。
HARDWARE_SLM_DRIVER = "ao_shaping.drivers.slm.santec.Santec (memory 模式, video_mode=0)"

#: 硬件后端的 CCD 驱动串 —— **逐字不变** (同上)。
HARDWARE_CCD_DRIVER = "ao_shaping.drivers.ccd.common.create_camera('daheng')"

#: 仿真后端的驱动串: 仿真面板 + 仿真相机 (无硬件)。
SIM_SLM_DRIVER = "ao_shaping.drivers.sim.slm_pib_sim.SimSLMPib (仿真面板, 无硬件)"
SIM_CCD_DRIVER = "ao_shaping.drivers.ccd.common.create_camera('sim')"

#: 仿真远场 FFT 的补零因子 —— 与
#: ``scripts/slm/run_gs_refine_zernike_sweep.py`` 的 ``--sim-far-field-padding``
#: **同名同义** (同一个数字、同一个含义: 补零越大, 远场像元越细, 代价按平方增长)。
DEFAULT_SIM_FAR_FIELD_PADDING = 16


def device_driver_strings(cam_type: str) -> tuple[str, str]:
    """(SLM 驱动串, CCD 驱动串) —— 按后端模式生成。

    硬件模式的字符串**逐字**等于第三轮已入库产物里的那两条 (见
    :data:`HARDWARE_SLM_DRIVER` / :data:`HARDWARE_CCD_DRIVER`); 仿真模式换成
    仿真面板与仿真相机。措辞要准 —— 写的是「哪个类 / 哪个工厂入口」, 不是泛泛的
    「仿真」。
    """
    kind = str(cam_type).strip().lower()
    if kind == "sim":
        return SIM_SLM_DRIVER, SIM_CCD_DRIVER
    if kind == "daheng":
        return HARDWARE_SLM_DRIVER, HARDWARE_CCD_DRIVER
    return (
        HARDWARE_SLM_DRIVER,
        f"ao_shaping.drivers.ccd.common.create_camera('{cam_type}')",
    )


def overlay_api_text(cam_type: str) -> str:
    """叠加层 API 的描述 —— 按后端模式给不同的措辞。

    硬件模式的文本**逐字**不变 (与第三轮已入库产物一致); 仿真模式的文本说清楚
    「叠加层在弧度→灰度**之前**相加, 远场看到的是相加后的相位」—— 这一条是离焦
    响应在仿真里能不能出现的关键。
    """
    if str(cam_type).strip().lower() == "sim":
        return (
            "SimSLMPib.set_phase_overlay(phase_rad) —— 弧度制叠加层, 在 "
            "create_phase_from_array 内部、弧度→灰度**之前**相加, 所以远场看到的"
            "是**相加后**的相位 (离焦必须真的让光斑变糊); 尺寸不符只 warning 并跳过; "
            "结束后 set_phase_overlay(None)"
        )
    return (
        "Santec.set_phase_overlay(phase_rad) —— 弧度制, 在 create_phase_from_array "
        "内部、弧度→灰度之前相加, 矫正/LUT/平移照常作用; 结束后 set_phase_overlay(None)"
    )


# ---------------------------------------------------------------------------
# 瞳面斜坡去倾斜 (自标定) 的默认参数
# ---------------------------------------------------------------------------
#: 去倾斜的**标定** Δz (mm)。二次相位的转向量按 Δz 线性, 所以在**一个** Δz 上
#: 量一次就够外推; 取 2.0 mm 是因为实测转向的局部拟合 (|Δz| ≤ 2.5 mm) 在那里
#: 还成立, 且位移 (~10 camera px) 远大于质心噪声。
DEFAULT_CHIRP_TILT_CAL_DZ_MM = 2.0

#: 标定用的**测试**斜坡周期 (panel px)。位移 ∝ 1/period, 所以 1000 px 对应
#: TILT_SHIFT_SCALE/1000 ≈ 7.4 camera px —— 足够大到压过质心噪声, 又不至于
#: 把光斑甩出窗口。
DEFAULT_CHIRP_TILT_TEST_PERIOD_PX = 1000

#: 自标定的**可信区间**: 实测斜率与 bench 常量 TILT_SHIFT_SCALE 的比值必须落在
#: 区间内。比值越界说明光路变过 (或标定时光斑根本没定住), 这次标定不可信 ——
#: 宁可不做去倾斜, 也不要用一个错的映射去补转向。
CHIRP_TILT_RATIO_MIN = 0.5
CHIRP_TILT_RATIO_MAX = 2.0

#: 斜坡的最小周期 (panel px)。周期 1 px 不是斜坡而是棋盘格, 光斑会被甩到几千
#: 相机 px 之外, 那种补偿没有意义。
MIN_RAMP_PERIOD_PX = 2

#: 标定时认为「光斑没动」的位移下限 (camera px)。低于它就当作这次斜坡没生效,
#: 对应的面板轴判为不可用。
CHIRP_TILT_MIN_SHIFT_PX = 0.2

#: 逐行落盘的 Δz 相关字段 (基线行与扫描行共用同一组键)。
#: 去倾斜的字段也在里面 —— 每一行都要能看出「这一行叠了多大的补偿」。
STEP_ROW_FIELDS = (
    "dz_mm", "dz_units", "chirp_sign", "chirp_alpha", "chirp_edge_rad",
    "chirp_edge_waves", "chirp_corner_rad", "defocus_focal_mm",
    "chirp_tilt_enabled", "chirp_tilt_comp_shift_x_px",
    "chirp_tilt_comp_shift_y_px", "chirp_tilt_comp_specs",
)

#: 每次读帧的超时保护 (s)。Santec 的 ``WaitImageV3`` 原生等待由 SDK 内部驱动,
#: **不受 Python 侧超时保护** (2026-09-08 实测可无限阻塞), 所以这里用看门狗线程
#: + 硬退出, 见 :func:`guarded`。
ACQUIRE_TIMEOUT_S = 60.0
OPEN_TIMEOUT_S = 180.0

#: **光路受阻**时本脚本的退出码。
#:
#: 与本脚本的另外两个退出码区分开: 2 = 读帧超时 (见 :func:`guarded`),
#: 3 = dry-run 违约 (见 :func:`assert_no_drivers`)。光路受阻是**操作者要动手**
#: 的故障, 给出自己的码, 免得被当成超时或 dry-run 违约去查错方向。
EXIT_PATH_BLOCKED = 4

#: USB 瞬时错误的重试策略。驱动的 ``_write_phase`` 内部只试 3 次 (间隔 100 ms),
#: 而 FTDI 的 ``-10032 USB其他错误`` 常连着三次一起失败 (2026-10-20 实测: 写
#: 内存#6 连续 3 次 -10032, 整轮扫描中断, 且只落盘了 3 行)。这里再包一层更长
#: 的退避 (0.5 / 2 / 8 s), 并每次换一个 memory slot。
USB_RETRY_ATTEMPTS = 3
USB_RETRY_BASE_WAIT_S = 0.5

HANG_RUNBOOK = """\
────────────────────────────────────────────────────────────
超时: {label} 超过 {timeout:.0f}s 未返回。

已知该症状 (仓库实测, 别再重新诊断一遍):
  * 日志止于「成功打开SLM」, 之后第一次 get_numpy_image() 前无限阻塞 ——
    SDK 的 WaitImageV3 原生等待不受 Python 侧超时保护 (2026-09-08)。
  * SLM 在 DVI 模式 (video_mode=1) 的 open() 可挂 120s/300s, 挂起后 memory
    模式也挂, **只能物理断电重置**。本脚本恒用 memory 模式, 不会触发。

处置:
  1. Get-Process python* | Stop-Process -Force   (强杀残留进程)
  2. 重跑前先确认没有别的进程占着 SLM / CCD (USB 与相机独占)
  3. memory 模式 open() 若超过几秒无日志 -> 对 SLM 控制器**物理断电**重置
────────────────────────────────────────────────────────────"""


# ---------------------------------------------------------------------------
# 光学常量 (dry-run 安全: 只用零导入叶子)
# ---------------------------------------------------------------------------
def read_default_wavelength_nm() -> float:
    """用 ``ast`` 静态读取 ``DEFAULT_WAVELENGTH_NM``, **不 import** 驱动包。

    单一事实源是 ``ao_shaping.drivers.slm.factory`` 里的那个常量; 直接 import 会把
    整个驱动包拉进 ``sys.modules``, 破坏「dry-run 不碰设备」的保证。静态读取保住了
    单一事实源, 又不触发 import 期副作用。读不到就返回 ``None`` 由调用方报错 ——
    不猜、不写死。
    """
    src = (
        ROOT / "src" / "ao_shaping" / "drivers" / "slm" / "factory.py"
    )
    tree = ast.parse(src.read_text(encoding="utf-8"), filename=str(src))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "DEFAULT_WAVELENGTH_NM":
                value = ast.literal_eval(node.value)
                return float(value)
    raise LookupError(
        f"在 {src} 里找不到 DEFAULT_WAVELENGTH_NM; 用 --wavelength-nm 显式指定"
    )


@dataclass(frozen=True)
class Optics:
    """本次运行用到的全部光学常量 (m / px)。"""

    wavelength_nm: float
    focal_m: float
    aperture_m: float
    panel_shape: tuple[int, int]
    beam_center_panel: tuple[int, int]
    beam_radius_panel: float
    panel_pitch_m: float
    camera_pixel_um: float

    @property
    def wavelength_m(self) -> float:
        return self.wavelength_nm * 1e-9

    @property
    def dz_blur_m(self) -> float:
        """Δz_blur = 2.44·λ·f²/D² (与仿真脚本同一定义, 用于对照)。"""
        return 2.44 * self.wavelength_m * self.focal_m**2 / self.aperture_m**2


def build_optics(args: argparse.Namespace) -> Optics:
    """组装光学常量。焦距由实测 2f 标度 + CCD 像元锚点导出 (与仿真脚本同源)。

    ``--wavelength-nm 0`` (默认) 在**这里**解析成驱动包里的那个常量, 而不是交给
    ``main`` 先解析 —— 否则调用顺序一变就会把 0 当波长传进
    ``focal_length_from_camera_pixel`` 并抛一个和真因无关的 ValueError。
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
    return Optics(
        wavelength_nm=wavelength_nm,
        focal_m=focal,
        aperture_m=2.0 * BEAM_RADIUS_PANEL * SLM_PITCH_M,
        panel_shape=(SLM_PANEL_H, SLM_PANEL_W),
        beam_center_panel=(int(BEAM_CENTER_PANEL[0]), int(BEAM_CENTER_PANEL[1])),
        beam_radius_panel=float(BEAM_RADIUS_PANEL),
        panel_pitch_m=SLM_PITCH_M,
        camera_pixel_um=CAMERA_PIXEL_UM,
    )


# ---------------------------------------------------------------------------
# 瞳面二次相位 (面板分辨率, raw 弧度)
# ---------------------------------------------------------------------------
def panel_pupil_chirp(
    dz_m: float,
    optics: Optics,
    *,
    sign: float,
    alpha: float,
    shape: tuple[int, int] | None = None,
    center: tuple[int, int] | None = None,
    mask_radius_px: float | None = None,
) -> np.ndarray:
    """面板分辨率的瞳面二次相位 ``sign·α·π·Δz·r²/(λf²)`` (raw 弧度, **不** mod 2π)。

    ``r`` 用**面板像元**乘以 SLM 像元间距得到 (m), 所以这个系数与仿真脚本
    ``PupilChirp.coefficient`` 逐字相同 —— 前提是模型口径等于面板口径, 那是仿真脚本
    的自检 1/3 在把关。

    ⚠️ **默认遮罩到照明圆盘** (``mask_radius_px=BEAM_RADIUS_PANEL``): 盘外没有光,
    相位再大也不产生任何物理效果, 但它会被 ``create_phase_from_array`` 一起换算成
    灰度并参与波前矫正/LUT, 白白引入逐像素的灰度跳变。遮罩把它压成 0。
    """
    h, w = shape or optics.panel_shape
    cx, cy = center or optics.beam_center_panel
    xs = (np.arange(w) - cx) * optics.panel_pitch_m
    ys = (np.arange(h) - cy) * optics.panel_pitch_m
    r2 = ys[:, None] ** 2 + xs[None, :] ** 2
    phase = sign * alpha * np.pi * dz_m * r2 / (optics.wavelength_m * optics.focal_m**2)
    if mask_radius_px:
        yy, xx = np.ogrid[:h, :w]
        inside = (xx - cx) ** 2 + (yy - cy) ** 2 <= float(mask_radius_px) ** 2
        phase = np.where(inside, phase, 0.0)
    return phase


def chirp_edge_rad(dz_m: float, optics: Optics, *, sign: float, alpha: float) -> float:
    """照明圆盘边缘 ``r = D/2`` 处的相位幅度 (rad) —— 「面板边缘相位」。"""
    r = 0.5 * optics.aperture_m
    return abs(sign * alpha * np.pi * dz_m * r**2 / (optics.wavelength_m * optics.focal_m**2))


# ---------------------------------------------------------------------------
# 瞳面菲涅尔焦面矫正 (面板分辨率, raw 弧度)
# ---------------------------------------------------------------------------
def build_focus_panel(focal_mm: float | None, optics: Optics) -> np.ndarray | None:
    """瞳面菲涅尔焦面矫正相位 ``pi·r²/(λ·f_v)`` (raw 弧度, 盘外为 0)。

    2f 实验系统的 CCD 并不在 2f 焦面上 (实测 :mod:`ao_shaping.tools.slm.panel.focus_probe`
    标定出 1/f = +0.541 m⁻¹, 即 f_v ≈ 1848 mm), 真实光路需要这张瞳面二次相位把
    焦面搬回探测器。``None`` = 不矫正, 与本脚本引入该选项之前逐位一致。

    ⚠️ 几何必须与 :func:`panel_pupil_chirp` 对齐: 同一个照明圆盘 (同一中心、同一
    半径), 否则两张相位在盘边界上不连续, 会在盘缘引入一圈伪结构。
    """
    if focal_mm is None:
        return None
    return fresnel_panel(
        float(focal_mm),
        radius=int(BEAM_RADIUS_PANEL),
        pupil_center=(int(BEAM_CENTER_PANEL[0]), int(BEAM_CENTER_PANEL[1])),
        panel_shape=optics.panel_shape,
        wavelength_nm=optics.wavelength_nm,
        pixel_um=optics.panel_pitch_m * 1e6,
    )


# ---------------------------------------------------------------------------
# 瞳面斜坡去倾斜 (自标定)
# ---------------------------------------------------------------------------
# 🟡 **为什么需要去倾斜**
#
# 瞳面二次相位以面板的**标称**照明中心 (960, 600) 为中心, 而真实照明并不落在
# 那里 —— 一片**偏心**的二次相位等价于一片偏心透镜: 它既离焦, 又**转向**。
# 实测 (2026-10-20, 组 dz_scan_hw_gs): 远场质心随 Δz 以 **-5.2 camera px/mm**
# 移动 (|Δz| ≤ 2.5 mm 的局部拟合)。
#
# 这个转向是**模拟**的性质, 不是实验系统的性质: 对拍的仿真把二次相位对准**光束**
# 中心, 所以它没有转向分量。于是 (a) 冻结的目标框会随 Δz 挪走, 量到的变成
# 转向而不是离焦; (b) 硬件与仿真的对比不同口径。
#
# 去倾斜 = 现场**自标定**出一条瞳面斜坡, 把转向补掉:
#
#   1. 现场测出「面板斜坡 → 相机位移」的映射 (哪条面板轴挪哪条相机轴) 与**符号**;
#   2. 与 Δz=0 基线相减, 得二次相位的每 mm 转向量;
#   3. 逐 Δz 反解出补偿斜坡, 与二次相位**相加**进同一张叠加图 (叠加层只有一条
#      通道, 分两次 set_phase_overlay 会互相覆盖);
#   4. 回到标定 Δz 复测, 记残差。
#
# 🔴 **符号一律现场测, 不许假设**: 本实验系统的面板轴与相机轴**交换 90°**
# (面板 x 斜坡挪的是相机 y), 而且这个**符号**仓库里任何地方都没记 —— 假设错了
# 去倾斜就会把转向**加倍**。
class ChirpTiltCalibrationError(RuntimeError):
    """去倾斜自标定失败。

    调用方必须**降级**为「不做去倾斜」继续跑 (记 ``chirp_tilt_enabled=false``),
    绝不让整轮扫描因为标定失败而作废。
    """


@dataclass(frozen=True)
class RampSpec:
    """一条面板斜坡的规格 (加进叠加层的那一条)。

    ``sign`` 是相位梯度的方向: ``+1`` 是 ``ramp_panel`` 本身, ``-1`` 是它的负
    (相位梯度反向 ⇒ 光斑往反方向走)。
    """

    panel_axis: int
    period_px: int
    sign: float = 1.0


@dataclass(frozen=True)
class RampAxisResponse:
    """面板某条轴的斜坡 → 相机位移的**实测**响应。"""

    panel_axis: int
    camera_axis: int
    shift_px: float
    period_px: int
    shift_vector_px: tuple[float, float] = (0.0, 0.0)

    @property
    def slope(self) -> float:
        """相机 px × panel px —— 位移 ∝ 1/period 的斜率 (与 bench 常量同量纲)。"""
        return float(self.shift_px) * float(self.period_px)

    @property
    def expected_ratio(self) -> float:
        """实测斜率与 bench 常量的比值 (只比**幅度**, 符号是现场学的)。"""
        return abs(self.slope) / float(TILT_SHIFT_SCALE)


@dataclass(frozen=True)
class RampAxisMap:
    """面板斜坡 → 相机位移的实测映射 (每条面板轴一条响应)。"""

    responses: tuple[RampAxisResponse, ...] = ()

    def for_panel_axis(self, panel_axis: int) -> RampAxisResponse | None:
        for response in self.responses:
            if response.panel_axis == int(panel_axis):
                return response
        return None

    def for_camera_axis(self, camera_axis: int) -> RampAxisResponse | None:
        for response in self.responses:
            if response.camera_axis == int(camera_axis):
                return response
        return None


@dataclass(frozen=True, eq=False)
class OverlayState:
    """一次采集对应的叠加层状态。

    ``overlay`` 是真正下发给 SLM 的瞳面相位 (焦面矫正 + 二次相位 + 去倾斜斜坡)。
    ``ramps`` / ``chirp_dz_mm`` 只是**描述** —— 硬件路径用不上, 但离线复算与
    单元测试 (假实现) 需要知道这一帧到底叠了什么。

    ``eq=False``: 字段里有 numpy 数组, 默认的逐字段比较会得到「数组的真值不
    明确」这种没意义的报错 —— 状态本来就只当**载荷**用, 不参与比较。
    """

    label: str
    overlay: np.ndarray | None
    ramps: tuple[RampSpec, ...] = ()
    chirp_dz_mm: float | None = None


@dataclass(frozen=True)
class ChirpTiltCalibration:
    """一次去倾斜自标定的结果。"""

    axis_map: RampAxisMap
    raw_shift_px_per_mm: tuple[float, float]
    cal_dz_mm: float
    test_period_px: int
    cal_centroid_xy: tuple[float, float]
    test_shifts_px: tuple[tuple[float, float], ...] = ()


def compose_overlay(*parts: np.ndarray | None) -> np.ndarray | None:
    """把若干张瞳面相位**相加**成一张 (驱动的叠加层只有一个通道)。

    ``None`` 的项直接跳过 —— 焦面矫正与去倾斜斜坡都是**可选**的。
    """
    acc: np.ndarray | None = None
    for part in parts:
        if part is None:
            continue
        acc = np.asarray(part, dtype=np.float64) if acc is None else acc + part
    return acc


def spot_centroid(frame: np.ndarray) -> tuple[float, float] | None:
    """帧里**可用**的光斑质心 (相机 x/y); 没有可用光斑时返回 ``None``。

    「可用」= 峰值非零且质心有限。全黑帧的质心会被 argmax 落在 (0, 0), 那是
    数组下标而不是光斑 —— 不挡掉它, 去倾斜就会拿着一堆假坐标去标定。
    """
    spot = measure_spot(frame)
    if not (np.isfinite(spot.peak) and spot.peak > 0.0):
        return None
    if not (np.isfinite(spot.centroid_x) and np.isfinite(spot.centroid_y)):
        return None
    return float(spot.centroid_x), float(spot.centroid_y)


class PathBlockedError(RuntimeError):
    """光路受阻 —— 画面里根本没有光, 继续扫只是在测噪声。

    2026-10-21 实测: 光路被挡住时, 本探针会安安静静地记下 30 行**纯读出噪声**
    当成测量结果。人肉翻帧才发现光束根本没进 2f 透镜。判据必须看**结构**
    (峰值停在噪声本底 + 帧里没有局域光斑), 判据本身在
    :func:`ao_shaping.utils.image.frame_analysis.assess_spot`。
    """


def preflight_assessment(baseline_frame: np.ndarray) -> SpotUsability:
    """扫描前的光路自检: 基线帧 (Δz=0, 相位已施加) 里有没有可评分的局域光斑。

    Args:
        baseline_frame: 基线帧 (原始帧或已减背景的帧都可以)。

    Returns:
        :class:`~ao_shaping.utils.image.frame_analysis.SpotUsability`。
    """
    return assess_spot(baseline_frame)


def raise_if_path_blocked(assessment: SpotUsability) -> None:
    """光路受阻就抛 :class:`PathBlockedError` —— 调用方据此中止整轮扫描。

    「只是不够聚焦」(光到了但太宽) **不**抛: 那种情况这一轮扫描仍然有意义,
    只该警告。该不该中止只看「有没有光」这一件事。
    """
    if assessment.blocked:
        raise PathBlockedError(assessment.reason)


def shift_per_mm(
    baseline_xy: tuple[float, float],
    shifted_xy: tuple[float, float],
    dz_mm: float,
) -> tuple[float, float]:
    """把「某个 Δz 上的位移」折算成**每 mm** 的位移 (相机 x/y)。

    二次相位的转向量按 Δz 线性 (转向来自二次相位里那个固定的偏心项, 幅度 ∝ Δz),
    所以在**一个** Δz 上量一次就够外推。
    """
    dz = float(dz_mm)
    if dz == 0.0:
        raise ValueError("Δz = 0 无法折算成每 mm 位移")
    return (
        (float(shifted_xy[0]) - float(baseline_xy[0])) / dz,
        (float(shifted_xy[1]) - float(baseline_xy[1])) / dz,
    )


def predict_shift_px(
    per_mm: tuple[float, float], dz_mm: float
) -> tuple[float, float]:
    """按每 mm 位移外推到某个 Δz 上的期望位移 (相机 x/y)。"""
    return (float(per_mm[0]) * float(dz_mm), float(per_mm[1]) * float(dz_mm))


def learn_ramp_axis_map(
    shifts: dict[int, tuple[float, float]],
    period_px: int,
) -> RampAxisMap:
    """由「每条面板轴的斜坡让光斑挪了多少」学出面板→相机的映射。

    每条面板轴的位移里, **较大的那个分量**就是它驱动的相机轴 (本实验系统面板/
    相机轴交换 90°: 面板 x 斜坡挪的是相机 y); 分量的**符号**一并学 —— 仓库里
    没有任何地方记着这个符号, 所以只能现场量。

    Args:
        shifts: ``{面板轴: (相机 x 位移, 相机 y 位移)}``。
        period_px: 测试斜坡的周期 (panel px)。

    Returns:
        学到的映射 (某个面板轴没测出可用位移时, 它就不在结果里)。
    """
    responses: list[RampAxisResponse] = []
    for panel_axis in (0, 1):
        vector = shifts.get(int(panel_axis))
        if vector is None:
            continue
        if abs(vector[0]) >= abs(vector[1]):
            camera_axis, signed = 0, float(vector[0])
        else:
            camera_axis, signed = 1, float(vector[1])
        responses.append(
            RampAxisResponse(
                panel_axis=int(panel_axis),
                camera_axis=int(camera_axis),
                shift_px=float(signed),
                period_px=int(period_px),
                shift_vector_px=(float(vector[0]), float(vector[1])),
            )
        )
    return RampAxisMap(tuple(responses))


def ramp_camera_shift_px(
    axis_map: RampAxisMap, spec: RampSpec
) -> tuple[float, float]:
    """给定一条斜坡的规格, 预测它在**相机**上造成的位移 (相机 x/y)。

    位移 ∝ 1/period, 所以周期翻倍位移减半 —— 这也是 ``TILT_SHIFT_SCALE`` 的定义。
    """
    response = axis_map.for_panel_axis(spec.panel_axis)
    if response is None:
        return (0.0, 0.0)
    value = float(spec.sign) * response.slope / float(spec.period_px)
    if response.camera_axis == 0:
        return (value, 0.0)
    return (0.0, value)


def compensating_ramp(
    desired_shift_px: tuple[float, float],
    axis_map: RampAxisMap,
    panel_shape: tuple[int, int],
    *,
    min_period_px: int = MIN_RAMP_PERIOD_PX,
) -> tuple[np.ndarray, tuple[RampSpec, ...]]:
    """按**期望的相机位移**反解出要叠加的瞳面斜坡 (每条相机轴最多一条)。

    周期由实测斜率反解: 位移 ∝ 1/period ⇒ ``period = |斜率| / |期望位移|``;
    符号取「让实测响应与期望同向」的那一侧 (斜坡取负即相位梯度反向)。周期只能
    取整 (panel px), 所以实际位移与期望之间有一个 1/(2·period) 的相对误差 ——
    记在返回的规格里, 由调用方核对。

    Args:
        desired_shift_px: 期望的**相机**位移 (x, y)。
        axis_map: 自标定学到的面板→相机映射。
        panel_shape: 面板形状 (h, w)。
        min_period_px: 允许的最小周期 (panel px)。

    Returns:
        ``(叠加相位, 斜坡规格列表)``。无可用响应时返回全零相位与空列表。
    """
    phase: np.ndarray | None = None
    specs: list[RampSpec] = []
    for camera_axis in (0, 1):
        desired = float(desired_shift_px[camera_axis])
        if not np.isfinite(desired) or abs(desired) < 1e-9:
            continue
        response = axis_map.for_camera_axis(camera_axis)
        if response is None:
            continue
        slope = response.slope
        if not np.isfinite(slope) or abs(slope) <= 0.0:
            continue
        period = max(int(round(abs(slope) / abs(desired))), int(min_period_px))
        sign = 1.0 if (slope * desired) > 0.0 else -1.0
        spec = RampSpec(response.panel_axis, period, sign)
        specs.append(spec)
        ramp = ramp_panel(period, response.panel_axis, panel_shape) * sign
        phase = ramp if phase is None else phase + ramp
    if phase is None:
        phase = np.zeros(panel_shape, dtype=np.float64)
    return phase, tuple(specs)


def axis_map_to_dict(axis_map: RampAxisMap) -> dict[str, Any]:
    """把自标定结果摊成可 JSON 化的字典 (写进 ``run_config.json``)。"""
    out: dict[str, Any] = {}
    for response in axis_map.responses:
        out[f"panel_axis_{response.panel_axis}"] = {
            "panel_axis": response.panel_axis,
            "camera_axis": response.camera_axis,
            "shift_px": response.shift_px,
            "period_px": response.period_px,
            "slope_px_per_inv_period": response.slope,
            "expected_ratio": response.expected_ratio,
            "shift_vector_px": list(response.shift_vector_px),
        }
    ratios = [r.expected_ratio for r in axis_map.responses]
    out["measured_vs_expected_ratio"] = float(min(ratios)) if ratios else float("nan")
    return out


def calibrate_chirp_tilt(
    capture: Callable[[OverlayState], np.ndarray],
    *,
    baseline_xy: tuple[float, float],
    focus_panel: np.ndarray | None,
    chirp_panel: np.ndarray,
    cal_dz_mm: float,
    test_period_px: int,
    panel_shape: tuple[int, int],
) -> ChirpTiltCalibration:
    """现场标定「面板斜坡 → 相机位移」的映射, 并量出二次相位的每 mm 转向量。

    标定序列 (每一步都是一次抓帧):

    1. Δz = ``cal_dz_mm`` 的二次相位, 记质心;
    2. 在**同一**状态上只多叠一条测试斜坡 (面板 y), 记质心位移;
    3. 面板 x 再来一次;
    4. 由这两次位移学出面板→相机的映射与符号 (不假设);
    5. 与 Δz=0 基线相减, 得每 mm 转向量。

    Raises:
        ChirpTiltCalibrationError: 任何一步量不到可用光斑, 或实测标度与 bench
            常量的比值落在可信区间之外 —— 调用方必须降级为「不做去倾斜」。
    """
    cal_dz = float(cal_dz_mm)
    if cal_dz == 0.0:
        raise ChirpTiltCalibrationError("标定 Δz 不能为 0 (无法折算成每 mm 位移)")
    period = int(test_period_px)
    if period <= 0:
        raise ChirpTiltCalibrationError(f"测试斜坡周期非法: {period}")

    chirp_only = compose_overlay(focus_panel, chirp_panel)
    cal_xy = spot_centroid(capture(OverlayState("chirp", chirp_only, (), cal_dz)))
    if cal_xy is None:
        raise ChirpTiltCalibrationError(
            f"Δz={cal_dz} mm 的二次相位上量不到可用光斑, 去倾斜标定放弃"
        )

    shifts: dict[int, tuple[float, float]] = {}
    for panel_axis in (0, 1):
        spec = RampSpec(panel_axis, period, 1.0)
        ramp = ramp_panel(period, panel_axis, panel_shape)
        state = OverlayState(
            f"chirp+ramp{panel_axis}",
            compose_overlay(focus_panel, chirp_panel, ramp),
            (spec,),
            cal_dz,
        )
        xy = spot_centroid(capture(state))
        if xy is None:
            raise ChirpTiltCalibrationError(
                f"面板轴 {panel_axis} 的测试斜坡上量不到可用光斑, 去倾斜标定放弃"
            )
        shifts[panel_axis] = (xy[0] - cal_xy[0], xy[1] - cal_xy[1])

    axis_map = learn_ramp_axis_map(shifts, period)
    if not axis_map.responses:
        raise ChirpTiltCalibrationError("两条面板轴都没测出可用位移, 去倾斜标定放弃")
    for response in axis_map.responses:
        if abs(response.shift_px) < CHIRP_TILT_MIN_SHIFT_PX:
            raise ChirpTiltCalibrationError(
                f"面板轴 {response.panel_axis} 的测试斜坡只挪了 "
                f"{response.shift_px:.3f} camera px (低于 {CHIRP_TILT_MIN_SHIFT_PX}), "
                "标定不可信, 去倾斜放弃"
            )
        ratio = response.expected_ratio
        if not (CHIRP_TILT_RATIO_MIN <= ratio <= CHIRP_TILT_RATIO_MAX):
            raise ChirpTiltCalibrationError(
                f"面板轴 {response.panel_axis} 的实测标度 / bench 常量 = {ratio:.3f} "
                f"(可信区间 [{CHIRP_TILT_RATIO_MIN}, {CHIRP_TILT_RATIO_MAX}]) —— "
                "光路可能变过, 标定不可信, 去倾斜放弃"
            )

    raw_per_mm = shift_per_mm(baseline_xy, cal_xy, cal_dz)
    return ChirpTiltCalibration(
        axis_map=axis_map,
        raw_shift_px_per_mm=raw_per_mm,
        cal_dz_mm=cal_dz,
        test_period_px=period,
        cal_centroid_xy=cal_xy,
        test_shifts_px=tuple(shifts[axis] for axis in (0, 1) if axis in shifts),
    )


def verify_chirp_tilt(
    capture: Callable[[OverlayState], np.ndarray],
    *,
    baseline_xy: tuple[float, float],
    focus_panel: np.ndarray | None,
    chirp_panel: np.ndarray,
    comp_phase: np.ndarray | None,
    comp_specs: tuple[RampSpec, ...],
    cal_dz_mm: float,
) -> tuple[float, float] | None:
    """在标定 Δz 上复测一次, 返回**残差**的每 mm 位移; 测不到则返回 ``None``。"""
    state = OverlayState(
        "chirp+comp",
        compose_overlay(focus_panel, chirp_panel, comp_phase),
        comp_specs,
        float(cal_dz_mm),
    )
    xy = spot_centroid(capture(state))
    if xy is None:
        return None
    return shift_per_mm(baseline_xy, xy, float(cal_dz_mm))


def compensation_for_dz(
    dz_mm: float,
    calibration: ChirpTiltCalibration,
    panel_shape: tuple[int, int],
) -> tuple[np.ndarray, tuple[RampSpec, ...], tuple[float, float]]:
    """某个 Δz 上的补偿斜坡。

    期望位移 = **负的**二次相位转向量 (把转向补掉), 周期由实测斜率反解。

    Returns:
        ``(相位, 斜坡规格, 实际达成的相机位移)``。实际位移用**取整后的**周期
        算 —— 周期只能取整 (panel px), 所以它与期望之间有一个
        1/(2·period) 的相对误差, 记下来由调用方核对。
    """
    desired = tuple(
        -value
        for value in predict_shift_px(calibration.raw_shift_px_per_mm, float(dz_mm))
    )
    phase, specs = compensating_ramp(desired, calibration.axis_map, panel_shape)
    realised = [0.0, 0.0]
    for spec in specs:
        dx, dy = ramp_camera_shift_px(calibration.axis_map, spec)
        realised[0] += dx
        realised[1] += dy
    return phase, specs, (realised[0], realised[1])


def run_chirp_tilt_calibration(
    capture: Callable[[OverlayState], np.ndarray],
    *,
    baseline_xy: tuple[float, float],
    focus_panel: np.ndarray | None,
    chirp_panel: np.ndarray,
    cal_dz_mm: float,
    test_period_px: int,
    panel_shape: tuple[int, int],
) -> tuple[ChirpTiltCalibration | None, tuple[float, float] | None, str]:
    """跑一次「自标定 + 验证」, 失败时**降级**而不是抛出去。

    降级是硬要求: 标定只是**锦上添花**的转向补偿, 它失败绝不该让整轮扫描作废
    (那会白扔一整轮实验系统的机时)。所以这里把任何异常都吞成
    ``(None, None, 说明)`` —— 调用方看到 ``None`` 就按纯二次相位继续。

    Returns:
        ``(标定结果, 残差, 说明)``。标定结果为 ``None`` ⇒ **不做去倾斜**。
    """
    try:
        calibration = calibrate_chirp_tilt(
            capture,
            baseline_xy=baseline_xy,
            focus_panel=focus_panel,
            chirp_panel=chirp_panel,
            cal_dz_mm=cal_dz_mm,
            test_period_px=test_period_px,
            panel_shape=panel_shape,
        )
    except Exception as exc:  # 标定失败不许整轮作废
        return None, None, f"标定失败: {exc}"

    comp_phase, comp_specs, _ = compensation_for_dz(
        cal_dz_mm, calibration, panel_shape
    )
    residual = verify_chirp_tilt(
        capture,
        baseline_xy=baseline_xy,
        focus_panel=focus_panel,
        chirp_panel=chirp_panel,
        comp_phase=comp_phase,
        comp_specs=comp_specs,
        cal_dz_mm=cal_dz_mm,
    )
    note = "标定与验证均完成" if residual is not None else "验证帧没有可用光斑, 残差未测到"
    return calibration, residual, note


# ---------------------------------------------------------------------------
# 超时保护
# ---------------------------------------------------------------------------
def guarded(label: str, fn: Callable[[], Any], *, timeout_s: float) -> Any:
    """跑 ``fn`` 并在 ``timeout_s`` 后硬退出, 打印可执行的排障提示。

    为什么必须硬退出而不是抛异常: Santec SDK 的等待在 C 层, 超时后线程仍卡着,
    ``finally`` 里的 ``close()`` 也会卡 —— 仓库里已经有一次 900s 强杀的记录。检测到
    超时就打印排障手册并 ``os._exit``, 让操作者去断电重置, 而不是陪着一个假死进程。
    """
    box: dict[str, Any] = {}

    def _run() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:  # 原样转交给主线程
            box["error"] = exc

    th = threading.Thread(target=_run, daemon=True, name=f"hw-{label}")
    th.start()
    th.join(timeout_s)
    if th.is_alive():
        sys.stderr.write(HANG_RUNBOOK.format(label=label, timeout=timeout_s) + "\n")
        sys.stderr.flush()
        os._exit(2)
    if "error" in box:
        raise box["error"]
    return box.get("value")


# ---------------------------------------------------------------------------
# 实验计划 (dry-run 的全部输出)
# ---------------------------------------------------------------------------
def build_chirp_tilt_plan(args: argparse.Namespace) -> dict[str, Any]:
    """去倾斜选项的计划描述 (dry-run 要把**怎么标定**和**怎么失败**讲清楚)。"""
    enabled = bool(getattr(args, "chirp_tilt_cancel", False))
    cal_dz = float(getattr(args, "chirp_tilt_cal_dz_mm", DEFAULT_CHIRP_TILT_CAL_DZ_MM))
    period = int(
        getattr(args, "chirp_tilt_test_period_px", DEFAULT_CHIRP_TILT_TEST_PERIOD_PX)
    )
    dz_list = [float(v) for v in getattr(args, "dz_mm", []) or []]
    in_scan = any(abs(v - cal_dz) < 1e-9 for v in dz_list)
    return {
        "enabled": enabled,
        "cal_dz_mm": cal_dz,
        "test_period_px": period,
        "cal_dz_in_scan": in_scan,
        "why": (
            "瞳面二次相位以面板的**标称**照明中心 (960, 600) 为中心, 而真实照明"
            "并不在那里 —— 偏心的二次相位等价于一片**偏心透镜**: 它既离焦, 又"
            "**转向** (实测 2026-10-20 组 dz_scan_hw_gs: 远场质心随 Δz 走 "
            "-5.2 camera px/mm)。这个转向是**模拟**的性质 (对拍的仿真把二次相位"
            "对准光束中心, 所以没有转向), 于是冻结的目标框会随 Δz 挪走, 量到的"
            "变成转向而不是离焦, 硬件与仿真也不同口径"
        ),
        "policy": (
            "**opt-in, 默认关闭**: 不给 --chirp-tilt-cancel 时行为与引入本选项"
            "之前逐位一致。打开后逐 Δz 在**同一张**叠加图里加一条瞳面斜坡, 把"
            "转向补掉; 目标框 / ROI / 冻结零级一律不动"
        ),
        "calibration_steps": [
            f"1) 在 Δz = {cal_dz:.3f} mm 叠加二次相位, 抓帧记质心 (与 Δz=0 基线相减 "
            "→ 二次相位的转向量)",
            f"2) 在**同一**状态上只多叠一条测试斜坡 (面板 y, 周期 {period} panel px), "
            "抓帧记质心位移",
            "3) 面板 x 再来一次",
            "4) 由这两次位移**现场学**出「面板轴 → 相机轴」的映射与符号 —— "
            "不假设: 本实验系统面板/相机轴交换 90° (面板 x 斜坡挪的是相机 y), "
            "而且这个符号仓库里任何地方都没记",
            "5) 实测标度与 bench 常量 TILT_SHIFT_SCALE 的比值必须落在 "
            f"[{CHIRP_TILT_RATIO_MIN}, {CHIRP_TILT_RATIO_MAX}] 内, 否则这次标定"
            "不可信",
            "6) 逐 Δz 反解补偿斜坡 (周期 = |实测斜率| / |期望位移|, 符号取与"
            "期望同向的一侧), 与二次相位**相加**进同一张叠加图",
            f"7) 回到 Δz = {cal_dz:.3f} mm 复测, 记**残差** (原始转向与残差都写进"
            "run_config.json)",
        ],
        "fail_safe": (
            "任何一步量不到可用光斑, 或标度比值越界, 一律**放弃去倾斜** "
            "(记 chirp_tilt_enabled=false) 并按原来的纯二次相位继续 —— 绝不因为"
            "标定失败让整轮扫描作废"
        ),
        "note": (
            f"标定 Δz = {cal_dz:.3f} mm "
            + ("在扫描列表内" if in_scan else "**不在**扫描列表内 —— 会额外抓一帧")
        ),
    }


def build_plan(args: argparse.Namespace, optics: Optics) -> dict[str, Any]:
    """完整实验计划: 扫哪些 Δz、每次施加什么相位、预计多久、需要哪些设备。"""
    dz_list = [float(v) for v in args.dz_mm]
    focus_mm = args.defocus_focal_mm
    focus_text = (
        f"瞳面菲涅尔焦面矫正: f_v = {focus_mm:.1f} mm (pi·r²/(λ·f_v), 盘外为 0) —— "
        "CCD 不在 2f 焦面时用它把焦面搬回探测器; 与 Δz 二次相位**相加**进同一张"
        "叠加图 (驱动只有一个叠加通道, 第二次 set_phase_overlay 会覆盖第一次)"
        if focus_mm is not None
        else "不矫正 (CCD 若不在 2f 焦面, 实测光斑会散; 用 --defocus-focal-mm 打开矫正)"
    )
    steps: list[dict[str, Any]] = []
    for mm in dz_list:
        edge = chirp_edge_rad(mm * 1e-3, optics, sign=args.chirp_sign, alpha=args.chirp_alpha)
        steps.append(
            {
                "step": len(steps),
                "dz_mm": mm,
                "dz_units": (mm * 1e-3) / optics.dz_blur_m,
                "chirp_sign": args.chirp_sign,
                "chirp_alpha": args.chirp_alpha,
                "chirp_edge_rad": edge,
                "chirp_edge_waves": edge / (2.0 * np.pi),
                "chirp_corner_rad": edge * np.sqrt(2.0),
                "defocus_focal_mm": focus_mm,
                "action": (
                    "set_phase_overlay(瞳面二次相位) + display_and_average(GS 相位)"
                ),
                "frames": int(args.frames),
            }
        )
    # 时间预算: SLM 写入+稳定 ~0.5-1.5 s/次, CCD 取帧 ~0.1-0.3 s/帧。
    per_step = 1.5 + int(args.frames) * 0.3
    group_name = str(getattr(args, "group", None) or "main")
    group_meta = dict(getattr(args, "group_meta", None) or {})
    requested_side = int(args.target_side_px)
    if requested_side > 0:
        target_box = {
            "requested_side_px": requested_side,
            "mode": "explicit",
            "policy": (
                f"显式 {requested_side} px (--target-side-px 给的); 中心冻结在零级, "
                "全程不变"
            ),
        }
    else:
        target_box = {
            "requested_side_px": 0,
            "mode": "auto",
            "policy": (
                "自动定尺: 用**本次相位自己的** Δz=0 远场光斑定边长 —— "
                "side = clamp(round(FWHM), 4, floor(0.6·ROI)); "
                "**只量一次**然后冻结, 之后每一行 (含基线行) 都用同一个框"
            ),
        }
    target_box["fallback_side_px"] = FALLBACK_TARGET_SIDE_PX
    target_box["why"] = (
        "框必须跟着**被探的那张相位**走: 硬件 GS 相位打出的光斑只有 12.9 px 宽, "
        "而历史上写死的 44 px 是**模型** GS 相位方形边长换算到硬件像元上的值 —— "
        "框比光斑大 3.4 倍时, 离焦弥散把整个框填满, 均匀度项 1/(1+CV) 上升 "
        "(CV 1.733 → 1.157) 恰好抵消功率项 PIB 的下降, 合成分 "
        "0.5·PIB + 0.5/(1+CV) 于是**看起来是平的**。按相位自己的光斑定尺后, "
        "离焦响应恢复单调"
    )
    chirp_tilt = build_chirp_tilt_plan(args)
    slm_driver, ccd_driver = device_driver_strings(args.cam_type)
    return {
        "group": {
            "name": group_name,
            "variable": GROUP_VARIABLE,
            "meta": group_meta,
            "out_root": (
                str(args.out_root) if getattr(args, "out_root", None)
                else "report/shape_trials/hw_gs_defocus_probe_<UTC>"
            ),
            "out_dir": f"<根目录>/{group_name}",
        },
        "devices": [
            {
                "role": "SLM",
                "driver": slm_driver,
                "slm_number": int(args.slm_number),
                "slm_wavelength_nm": int(args.wavelength_nm),
                "panel_shape_hw": list(optics.panel_shape),
                "fatal_if": "open() 在 DVI 模式会挂死 → 本脚本恒用 memory 模式",
            },
            {
                "role": "CCD",
                "driver": ccd_driver,
                "cam_id": int(args.cam_id),
                "exposure_ms_requested": float(args.exposure_ms),
                "roi_px": int(args.roi_px),
                "fatal_if": "曝光必须 <= AO_DAHENG_MAX_EXPOSURE_MS (未设则探测)",
            },
        ],
        "optics": {
            "wavelength_nm": optics.wavelength_nm,
            "focal_length_m": optics.focal_m,
            "aperture_m": optics.aperture_m,
            "panel_shape_hw": list(optics.panel_shape),
            "panel_pitch_m": optics.panel_pitch_m,
            "beam_center_panel": list(optics.beam_center_panel),
            "beam_radius_panel": optics.beam_radius_panel,
            "camera_pixel_um": optics.camera_pixel_um,
            "dz_blur_mm": optics.dz_blur_m * 1e3,
        },
        "gs_phase_source": (
            str(args.gs_phase) if args.gs_phase else
            "现场生成: import scripts.slm.experiment_gs_defocus_offocus 的 OffFocusBench "
            "跑 plain GS (Δz=0) —— 必须在 --execute 分支里 import, 因为它会牵入 "
            "ao_shaping.drivers.sim (纯仿真模块, 不碰硬件, 但会破坏 dry-run 的"
            "「零驱动包」断言)"
        ),
        "upsampling": (
            "upsample_bilinear_to_shape(n_grid, (900, 900)) -> embed_pupil_square(..., "
            "panel, 450, (960, 600)) —— 复用 ao_shaping.utils 的既有实现, 不自己写"
        ),
        "zero_order": (
            "display_and_average 后用 zero_order_center(frame, refine=False) = 全局 argmax "
            "定位 (绝不假设在画面中心); 定位一次即冻结 center 与 target 框, 之后不逐帧重定位"
        ),
        "overlay_api": overlay_api_text(args.cam_type),
        "focus_correction": focus_text,
        "zero_order_halo": (
            "每次测量额外记整帧峰值 / 框外峰值 / 框外功率占比。真实瞳面是双分量 "
            "U=U_inc*[eta+(1-eta)e^{iφ}], 零级只过物理透镜, 在 f+d 处散成晕斑 "
            "(ρ≈d*D/(2f), I_halo ∝ f²/d²)。口径提示: Δz 小时零级峰仍在目标框内, "
            "框外峰值量不到它, 要看整帧峰值; Δz 大到散出框后框外峰值才代表污染"
        ),
        "no_magnification": (
            "不按 M=(f+d)/f 重采样: Collins/ABCD T(f+d)·Lens(f)·T(f) 的 B ≡ f ⇒ "
            "核尺度是 λf 而非 λ(f+d), CCD 离焦只使图样变模糊而不缩放。目标框边长"
            "在扫描开始前定一次就**冻结**, ROI 不随 d 改"
        ),
        "target_box": target_box,
        "chirp_tilt": chirp_tilt,
        "preflight": {
            "what": "扫描开始前先对**基线帧** (Δz=0, GS 相位已施加) 做一次光路自检",
            "criteria": (
                "判据看**结构**而不是亮度: 峰值停在读出噪声本底 **且** 帧里没有任何"
                "局域光斑 ⇒ 光路受阻; 光到了但光斑太宽 ⇒ 只是没聚焦 (另一回事)"
            ),
            "blocked": (
                f"判为受阻就**中止整轮** (退出码 {EXIT_PATH_BLOCKED}), "
                "**不写** measurements.csv —— 只写一份带判定与理由的 run_config.json"
            ),
            "not_usable": "只是不够聚焦时**警告并继续**: 那种情况这一轮仍然有意义",
            "checklist": "光束→SLM 面板 → 2f 透镜 → CCD 三段光路",
        },
        "steps": steps,
        "estimate": {
            "per_step_s": per_step,
            "sweep_s": per_step * len(steps),
            "baseline_and_window_s": 12.0,
            "total_s": per_step * len(steps) + 12.0,
            "note": "不含 SLM/CCD open 与 close (各 5-60 s, SDK 而定)",
        },
        "outputs": (
            "report/shape_trials/hw_gs_defocus_probe_<UTC>/<组名>/: 每次测量一个 npz "
            "(原始帧 + 参数) + measurements.csv + run_config.json + hw_summary.png; "
            "根目录另有 manifest.json 按组索引; npz 存的是未处理原始帧, --replot 可离线重新出图"
        ),
        "predicted_by_simulation": (
            "report/shape_trials/scheme_gap_dz0_50mm*.csv —— 对拍用的仿真预测 "
            "(CV/PIB/score 与逐点差异表)"
        ),
    }


def print_plan(plan: dict[str, Any], optics: Optics) -> None:
    """把实验计划打成人可读的一段 (dry-run 的全部输出)。"""
    o = plan["optics"]
    lines: list[str] = []
    add = lines.append
    add("=" * 78)
    add("硬件实验计划 (dry-run) —— 未打开任何设备")
    add("=" * 78)
    add("")
    g = plan.get("group", {}) or {}
    meta = g.get("meta") or {}
    add("变量组 (一次调用只探这一组)")
    add(f"  组名     = {g.get('name', 'main')}")
    add(f"  自变量   = {g.get('variable', GROUP_VARIABLE)}")
    add(f"  固定参数 = {', '.join(f'{k}={v}' for k, v in meta.items()) or '(无)'}")
    add(f"  根目录   = {g.get('out_root', '')}")
    add(f"  组目录   = {g.get('out_dir', '')}")
    add("")
    add("光学常量 (全部来自仓库零导入叶子, 与仿真脚本同源)")
    add(f"  波长 λ            = {o['wavelength_nm']:.1f} nm")
    add(f"  焦距 f            = {o['focal_length_m'] * 1e3:.3f} mm  (由实测 2f 标度 + CCD 像元导出)")
    add(f"  口径 D            = {o['aperture_m'] * 1e3:.3f} mm")
    add(f"  面板              = {o['panel_shape_hw'][1]} x {o['panel_shape_hw'][0]} px @ "
        f"{o['panel_pitch_m'] * 1e6:.1f} um")
    add(f"  光束 (面板坐标)    = 中心 {tuple(o['beam_center_panel'])}, 半径 "
        f"{o['beam_radius_panel']:.0f} px")
    add(f"  CCD 像元          = {o['camera_pixel_um']} um")
    add(f"  Δz_blur           = {o['dz_blur_mm']:.4f} mm   (2.44 λf²/D²)")
    add("")
    add("需要的设备")
    for d in plan["devices"]:
        add(f"  [{d['role']}] {d['driver']}")
        for k, v in d.items():
            if k in ("role", "driver"):
                continue
            add(f"        {k}: {v}")
    add("")
    add("GS 相位来源")
    add(f"  {plan['gs_phase_source']}")
    add(f"  上采样: {plan['upsampling']}")
    add("")
    add("零级与 ROI (关键: 不能假设 0 级在画面中心)")
    add(f"  {plan['zero_order']}")
    add("")
    add("目标框 (按**本次相位**的光斑定尺)")
    tb = plan["target_box"]
    mode_text = (
        f"显式 {tb['requested_side_px']} px (--target-side-px 给的)"
        if tb["mode"] == "explicit"
        else "自动 (--target-side-px 0, **默认**)"
    )
    add(f"  模式     = {mode_text}")
    add(f"  定尺     = {tb['policy']}")
    add(f"  兜底     = {tb['fallback_side_px']} px (基线帧测不出光斑时)")
    add(f"  为什么   = {tb['why']}")
    add("")
    add("瞳面斜坡去倾斜 (自标定, opt-in)")
    ct = plan["chirp_tilt"]
    add(f"  状态     = {'打开 (--chirp-tilt-cancel)' if ct['enabled'] else '关闭 (默认)'}")
    add(f"  标定 Δz  = {ct['cal_dz_mm']:.3f} mm; 测试斜坡周期 = {ct['test_period_px']} panel px")
    add(f"  为什么   = {ct['why']}")
    add(f"  策略     = {ct['policy']}")
    add("  标定步骤:")
    for line in ct["calibration_steps"]:
        add(f"    {line}")
    add(f"  失败保护 = {ct['fail_safe']}")
    add(f"  备注     = {ct['note']}")
    add("")
    add("零级晕斑 (硬件侧最该量的东西)")
    add(f"  {plan['zero_order_halo']}")
    add("")
    add("光路自检 (扫描前的 pre-flight)")
    pf = plan["preflight"]
    add(f"  做什么   = {pf['what']}")
    add(f"  判据     = {pf['criteria']}")
    add(f"  受阻时   = {pf['blocked']}")
    add(f"  没聚焦时 = {pf['not_usable']}")
    add(f"  检查清单 = {pf['checklist']}")
    add("")
    add("离焦施加方式")
    add(f"  {plan['overlay_api']}")
    add(f"  焦面矫正: {plan['focus_correction']}")
    add(f"  符号/前因子: sign={plan['steps'][0]['chirp_sign']:+.0f}, "
        f"alpha={plan['steps'][0]['chirp_alpha']:.5f} "
        f"(由仿真脚本数值反解: sign=+1 的残差比 sign=-1 高 ~370 倍, 非照抄公式)")
    add(f"  不放大: {plan['no_magnification']}")
    add("")
    add(f"扫描步骤 (共 {len(plan['steps'])} 步, 每个 Δz 一次测量)")
    add("  #   Δz/mm   Δz/Δz_blur   面板边缘相位/rad   /波数   施加内容")
    for s in plan["steps"]:
        add(f"  {s['step']:<3d} {s['dz_mm']:>7.1f} {s['dz_units']:>11.2f} "
            f"{s['chirp_edge_rad']:>19.2f} {s['chirp_edge_waves']:>9.2f}   {s['action']}")
    add("")
    add("预计耗时")
    e = plan["estimate"]
    add(f"  扫描 {len(plan['steps'])} 步 x {e['per_step_s']:.1f}s = {e['sweep_s']:.1f}s; "
        f"基线+开窗 {e['baseline_and_window_s']:.0f}s; 合计 ≈ {e['total_s']:.0f}s "
        f"({e['note']})")
    add("")
    add("产物")
    add(f"  {plan['outputs']}")
    add(f"  对拍基准: {plan['predicted_by_simulation']}")
    add("")
    add("=" * 78)
    print("\n".join(lines))


# ---------------------------------------------------------------------------
# --execute
# ---------------------------------------------------------------------------
@dataclass
class _ProbeParams:
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


def configure_sim_system(seed: int, far_field_padding: int) -> None:
    """把进程级的仿真系统重置到指定的种子与远场补零因子。

    与 ``scripts/slm/run_gs_refine_zernike_sweep.py::configure_sim_system`` 同义:
    ``reset_system`` 不接受 padding 参数, 所以 padding 是在**任何一次远场求值
    之前**改到新 system 上的 (那时 ``_far_field`` 与 ``_pad_buffer`` 还是
    ``None``, 改动安全)。已装的干扰项 (disturbance) 会被带过去。

    Args:
        seed: 仿真噪声模型的 RNG 种子。
        far_field_padding: 远场 FFT 的补零因子。
    """
    from ao_shaping.drivers.sim.slm_pib_sim import get_system, reset_system

    existing = get_system()
    system = reset_system(seed=seed, disturbance=existing.disturbance)
    system.far_field_padding = int(far_field_padding)


def read_exposure_ms(cam: Any) -> float:
    """当前**生效**曝光 (ms)。

    硬件相机有设备回读 (``get_exposure_time_ms``), 读的是设备真值; 仿真相机
    没有这个接口, 回落驱动缓存 (``exposure_time_ms``) —— 仿真里曝光本来就不
    改变画面亮度, 它只是个记录值。
    """
    reader = getattr(cam, "get_exposure_time_ms", None)
    if reader is None:
        return float(getattr(cam, "exposure_time_ms", 0.0))
    return float(reader())


def protected_memory_slot(slm: Any) -> int | None:
    """运行前**正在显示**的内存槽; 没有槽位概念的设备返回 ``None``。

    硬件: ``get_displayed_memory_number()`` 报错误码 1 (set_grayscale 模式) 是
    **正常**的, 不是故障 —— 那时根本没有槽位要排除。仿真面板没有这个接口,
    于是也没有槽位要排除。
    """
    from ao_shaping.drivers.slm.santec import SantecError

    getter = getattr(slm, "get_displayed_memory_number", None)
    if getter is None:
        return None
    try:
        return int(getter())
    except SantecError as exc:
        if exc.code != 1:
            raise
        return None  # set_grayscale 模式下没有槽位, 报错码 1 正常


def group_fields(group: str, meta: dict[str, str] | None = None) -> dict[str, Any]:
    """这一组的身份字段 (组名 / 自变量 / 固定参数)。

    ``run_config.json`` 与根目录的 ``manifest.json`` 都用这三个字段, 所以定义只有一处。
    固定参数**逐字**保留 —— 清单是这一组的事实记录, 推断类型会让人以为参数被归一化过。

    Args:
        group: 组名 (``--group``)。
        meta: 这一组的固定参数 (``--group-meta``)。

    Returns:
        ``{"group": ..., "variable": ..., "group_meta": {...}}``。
    """
    return {
        "group": str(group),
        "variable": GROUP_VARIABLE,
        "group_meta": {str(k): str(v) for k, v in (meta or {}).items()},
    }


def halo_metrics(frame: np.ndarray, box: np.ndarray) -> dict[str, float]:
    """零级晕斑的**实测口径**: 目标框外的峰值与功率占比。

    ⚠️ 与仿真侧 :func:`run_halo` 的纯零级正演**不是同一个量**, 两者不能混:
    框外峰值在 Δz=0 时**量不到零级** (零级峰此时在框内, 与整形方形重叠), 只有
    Δz 足够大、零级散开到框外之后才代表污染。所以这里同时记整帧峰值与外框峰值,
    让读的人能自己判断当前 Δz 下哪个量有意义。

    框外功率占比还混了整形方形自己的边瓣与焦外弥散, 所以它是一个**上界**, 不是
    纯零级贡献 —— 纯零级贡献要看 ``frame_peak`` 相对 η=0 基线的变化。
    """
    total = float(frame.sum())
    outside = frame[~box]
    return {
        "frame_peak": float(frame.max()),
        "halo_peak_outside_box": float(outside.max()) if outside.size else 0.0,
        "halo_power_fraction_outside_box": (
            float(outside.sum() / total) if total > 0 else 0.0
        ),
    }


def spot_geometry(frame: np.ndarray) -> dict[str, float]:
    """一帧的光斑几何: FWHM / 质心 / 空心度。

    口径与离线报告生成器 (``scripts/slm/generate_hw_defocus_zernike_report.py``)
    完全一致: 都在**存下来的那一帧**上量。把它写进测量行, ``measurements.csv``
    就是**自足**的 —— 报告不必再逐个打开 ``frame_*.npz`` 重算光斑。

    Args:
        frame: 探针存盘的那一帧 (原始帧, 未再加工)。

    Returns:
        ``{"spot_fwhm_px", "spot_centroid_x", "spot_centroid_y", "spot_hollowness"}``。
    """
    spot = measure_spot(frame)
    return {
        "spot_fwhm_px": float(spot.fwhm_px),
        "spot_centroid_x": float(spot.centroid_x),
        "spot_centroid_y": float(spot.centroid_y),
        "spot_hollowness": float(spot.hollowness),
    }


def resolve_probe_target_side(
    baseline_frame: np.ndarray,
    roi_px: int,
    requested_side_px: int,
) -> tuple[int, str, float]:
    """定目标框边长: 显式值照用, 否则按**本次相位自己的** Δz=0 光斑定尺。

    🔴 **名字必须带 ``probe_``**: 仓库里另有一个 ``resolve_target_side``
    (``scripts/slm/run_gs_refine_zernike_sweep.py``), 它的规则是 ``2·w0_flat``
    (平场腰径推导) —— 与这里的「按光斑 FWHM 定尺」是**两套不同的策略**。
    同名会让人以为它们是一回事, 所以这里显式改名; 但**不要**去统一两条规则 ——
    它们服务于不同的测量, 各有各的道理。

    框的大小必须跟着**被探的那张相位**走, 不能是别的相位的尺寸: 硬件 GS 相位
    与模型 GS 相位打出的光斑宽度差 12.9 px vs 22.6 px, 用同一个框去量两者,
    量到的根本不是同一件事。

    规则 (自动模式)::

        side = clamp(round(fwhm_px), 4, floor(0.6 * roi_px))

    其中 ``fwhm_px`` 是**基线帧** (Δz=0, GS 相位已施加) 的光斑 FWHM。框只量**一次**
    然后**冻结**: 逐帧按光斑重定尺会把离焦弥散本身藏起来 (弥散越大 → 框越大 →
    框内越均匀), 那正是仓库已经记下的自适应框刷分陷阱。

    Args:
        baseline_frame: 基线帧 (Δz=0, GS 相位已施加)。
        roi_px: ROI 边长 (px), 决定自动边长的上界。
        requested_side_px: ``--target-side-px``; 正数 = 显式框, ``0`` = 自动。

    Returns:
        ``(side, source, fwhm_px)``: 边长、来源标签 (``"explicit"`` /
        ``"auto_fwhm"`` / ``"fallback_44"``)、基线光斑的 FWHM (测不出时是 NaN)。
    """
    if int(requested_side_px) > 0:
        return int(requested_side_px), "explicit", float("nan")

    spot = measure_spot(baseline_frame)
    fwhm = float(spot.fwhm_px)
    if not np.isfinite(fwhm) or fwhm <= 0.0:
        # 基线帧没有可用光斑 (全黑 / 全平 / 退化) —— 退回历史写死值, 不让整轮作废。
        return FALLBACK_TARGET_SIDE_PX, "fallback_44", fwhm
    hi = int(np.floor(AUTO_TARGET_SIDE_ROI_FRACTION * int(roi_px)))
    hi = max(hi, AUTO_TARGET_SIDE_MIN_PX)
    side = int(min(max(int(round(fwhm)), AUTO_TARGET_SIDE_MIN_PX), hi))
    return side, "auto_fwhm", fwhm


def load_gs_phase_panel(args: argparse.Namespace, optics: Optics) -> np.ndarray:
    """拿到面板分辨率 (1200x1920) 的 GS 相位 (raw 弧度)。

    两条来源:

    * ``--gs-phase PATH``: 已有的 GS 相位 ``.npy`` / ``.npz``。面板分辨率的直接用;
      模型网格 (方阵, 边长 != 0) 走**仓库既有**的上采样链。
    * 未给路径: 现场用仿真脚本的已验证 :class:`OffFocusBench` 跑 plain GS (Δz=0),
      再走同一条上采样链。这样实测与仿真的相位来自同一份代码, 对拍才有意义。
    """
    from ao_shaping.utils.image.resample import upsample_bilinear_to_shape
    from ao_shaping.utils.slm.phase_embed import embed_pupil_square, pupil_side

    side = pupil_side(optics.beam_radius_panel)

    def to_panel(model: np.ndarray) -> np.ndarray:
        model = np.asarray(model, dtype=np.float64)
        box = upsample_bilinear_to_shape(model, (side, side))
        return embed_pupil_square(box, optics.panel_shape, optics.beam_radius_panel,
                                  optics.beam_center_panel)

    if args.gs_phase is not None:
        path = Path(args.gs_phase)
        raw = np.load(path) if path.suffix == ".npy" else np.load(path)["phase"]
        raw = np.asarray(raw, dtype=np.float64)
        if raw.shape == optics.panel_shape:
            return raw
        if raw.ndim == 2 and raw.shape[0] == raw.shape[1]:
            return to_panel(raw)
        raise ValueError(
            f"{path} 的形状 {raw.shape} 既不是面板 {optics.panel_shape}, 也不是方形模型网格"
        )

    from scripts.slm.experiment_gs_defocus_offocus import (
        FULL as SIM_FULL,
        QUICK as SIM_QUICK,
        OffFocusBench,
    )

    sim_cfg = SIM_QUICK if args.quick else SIM_FULL
    bench = OffFocusBench(sim_cfg.bench(), n_zernike=sim_cfg.n_zernike)
    gs = bench.gs_phase(0.0, n_iters=sim_cfg.gs_iters, seed=sim_cfg.seed)
    return to_panel(gs)


def run_execute(args: argparse.Namespace, optics: Optics, plan: dict[str, Any]) -> Path:
    """真正开设备, 逐 Δz 采集。返回测量目录。"""
    # ---- 设备 import 全部锁在 --execute 分支内 -----------------------------
    from ao_shaping.drivers.ccd.common import create_camera
    from ao_shaping.drivers.slm.factory import open_slm, sync_device_wavelength
    from ao_shaping.drivers.slm.santec import SantecError
    from ao_shaping.optimizer.wfless._measure import display_and_average
    from ao_shaping.tools.slm.panel.display_state import preserve_slm_display
    from ao_shaping.utils.image.beam_metrics import zero_order_center
    from ao_shaping.utils.image.exposure_control import (
        configured_daheng_exposure_limit,
        probe_exposure_ceiling_ms,
    )
    from ao_shaping.utils.image.targets import square_mask
    from ao_shaping.algorithm.signal_processing.square_shaping import (
        composite_from_pib_cv,
        power_in_bucket,
        uniformity_cv,
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    # 后端模式: 仿真与硬件走**同一套** CLI, 只有几处「只有硬件/只有仿真成立」
    # 的分支不同 (下面每处都标了为什么)。
    is_sim = str(args.cam_type).strip().lower() == "sim"
    if is_sim:
        # 仿真系统必须在**任何一次远场求值之前**就位 (种子 + 补零因子), 否则
        # 第一帧就已经用掉了默认系统的状态。
        configure_sim_system(int(args.sim_seed), int(args.sim_far_field_padding))
        print(f"仿真后端: seed={int(args.sim_seed)}, "
              f"远场补零={int(args.sim_far_field_padding)}")
    # 根目录: 默认按 UTC 时间戳新建; --out-root 可以把同一轮实验的多个组钉到
    # 同一个根目录下 (根目录的 manifest.json 才会把它们都索引起来)。
    out_root = (
        Path(args.out_root)
        if getattr(args, "out_root", None)
        else ROOT / "report" / "shape_trials" / f"hw_gs_defocus_probe_{stamp}"
    )
    if getattr(args, "out_root", None):
        # 显式指定的根目录: 允许已存在 (同一轮实验的多个组共用一个根目录)。
        out_root.mkdir(parents=True, exist_ok=True)
    else:
        # 时间戳根目录: 同秒两次启动是异常 (SLM/CCD 独占), 沿用原来的保护。
        out_root.mkdir(parents=True, exist_ok=False)
    group_name = str(getattr(args, "group", None) or "main")
    group_meta = dict(getattr(args, "group_meta", None) or {})
    # 组目录: <根目录>/<组名>/。测量文件全部落在这里, 根目录只留 manifest.json。
    # 同名组重跑会**覆盖**这一组的旧文件 (重跑一组 = 替换它, 清单里也是替换)。
    out_dir = group_output_dir(out_root, group_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"测量目录: {out_dir}")
    print(f"变量组: {group_name} (自变量 {GROUP_VARIABLE}); 根目录 {out_root}")

    gs_panel = load_gs_phase_panel(args, optics)
    # 焦面矫正只算一次: 它与扫描点无关, 且每次重算都是几百 MB 级的无用开销。
    focus_panel = build_focus_panel(args.defocus_focal_mm, optics)
    params = _ProbeParams(
        cam_type=args.cam_type,
        cam_id=int(args.cam_id),
        exposure_time_ms=float(args.exposure_ms),
        cam_size=int(args.roi_px),
        slm_number=int(args.slm_number),
        slm_wavelength=int(args.wavelength_nm),
    )

    rows: list[dict[str, Any]] = []

    def capture(slm: Any, cam: Any, phase: np.ndarray, slot: int) -> np.ndarray:
        return guarded(
            f"CCD 取帧 (slot={slot})",
            lambda: display_and_average(
                cam, slm, phase,
                n_frames=int(args.frames),
                settle_wait_s=float(args.settle_wait_s),
                settle_tol=0.02,
                settle_max_wait_s=float(args.settle_max_wait_s),
                settle_max_discard=int(args.settle_max_discard),
                memory_number=slot,
            ),
            timeout_s=ACQUIRE_TIMEOUT_S,
        )

    def capture_retry(
        slm: Any, cam: Any, phase: np.ndarray, next_slot: Callable[[], int]
    ) -> np.ndarray:
        """取帧, 遇 USB 瞬时错误退避重试; 全部失败则抛出最后一次异常。

        驱动的 ``_write_phase`` 已经在 ``retrying`` 里试 3 次 (间隔 100 ms), 但
        FTDI 的 ``-10032 USB其他错误`` 常常连着三次一起失败 (2026-10-20 实测:
        写内存#6 连续 3 次都报 -10032, 整轮扫描因此中断)。USB 栈的瞬时拥塞
        靠更长的退避就能过去, 所以这里再包一层 0.5/2/8 s 的退避, 并**每次换
        一个新 memory slot** (重试时那次写入很可能没生效, 换槽位避免写进
        当前正在显示的槽 —— 那会让面板不刷新)。
        """
        last: Exception | None = None
        for attempt in range(USB_RETRY_ATTEMPTS):
            slot = next_slot()
            try:
                return capture(slm, cam, phase, slot)
            except SantecError as exc:
                last = exc
                wait = USB_RETRY_BASE_WAIT_S * (4 ** attempt)
                logger.info(f"[USB 重试 {attempt + 1}/{USB_RETRY_ATTEMPTS}] slot={slot} "
                      f"失败: {exc}; 等待 {wait:.1f}s 后换槽重试")
                time.sleep(wait)
        assert last is not None
        raise last

    with guarded("SLM 打开", lambda: open_slm(params), timeout_s=OPEN_TIMEOUT_S) as slm:
        # ⚠️ `Santec.open()` 返回 None, 不能拿它的返回值当上下文管理器 —— 上一版
        # 这里写成 `with guarded(...):` 而 guarded 回传的是函数返回值, 于是实验系统上
        # 必然炸 `TypeError: 'NoneType' object does not support the context manager
        # protocol`。超时保护本身是对的, 只是需要一个不消费返回值的上下文。
        with nullcontext(
            guarded("SLM open()", lambda: slm.open(), timeout_s=OPEN_TIMEOUT_S)
        ):
            sync_device_wavelength(params, slm)
            # 仿真相机按 ``cam_size`` 裁远场, 所以窗口边长必须**显式**传给它 ——
            # 否则它用自己的默认 250, 而 ``--roi-px`` 改了却不起作用。硬件相机的
            # 窗口由下面的 ``reset_window`` 定 (大恒不接受 ``cam_size``), 所以
            # 这一个 kwarg 只在仿真分支给, 硬件调用逐字不变。
            cam_kwargs: dict[str, Any] = (
                {"cam_size": params.cam_size} if is_sim else {}
            )
            with create_camera(params.cam_type, params.cam_id,
                               exposure_time_ms=params.exposure_time_ms,
                               **cam_kwargs) as cam:
                # 仿真面板没有「当前显示的内存槽」这回事 (它不轮换槽位), 所以
                # 「退出时恢复原显示」那一层保护只对硬件有意义。
                display_guard = (
                    nullcontext() if is_sim else preserve_slm_display(slm)
                )
                with display_guard:
                    # -- 槽位轮换: 排除运行前正在显示的那个, 之后每次随机挑 ----------
                    protected = protected_memory_slot(slm)
                    used: set[int] = set()
                    if protected is not None:
                        used.add(int(protected))
                    rng = np.random.default_rng(int(args.seed))

                    def next_slot() -> int:
                        pool = [s for s in range(2, 126) if s not in used]
                        slot = int(rng.choice(pool))
                        used.add(slot)
                        return slot

                    # -- 曝光上限 -------------------------------------------------
                    if is_sim:
                        # 仿真相机没有饱和概念 (远场被归一化到峰值 ~100), 曝光只是
                        # 一个记录值; 硬件那条「倍增阶梯找饱和」的路在仿真里既不
                        # 适用, 也会白烧掉一串远场 FFT。
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
                    print(f"曝光: 请求 {args.exposure_ms} ms -> 上限 {ceiling} ms -> 生效 "
                          f"{actual} ms")

                    # -- 基线: 定位零级 (argmax) 并**冻结** ROI --------------------
                    # ⚠️ 焦面矫正必须在**定位之前**就位: 矫正相位以
                    # BEAM_CENTER_PANEL 为中心, 而照明并不落在那里, 所以它等价于
                    # 一片**偏心透镜**, 会把光斑推走几十 px (2026-10-20 实测全幅
                    # 质心: 未矫正 ~1352 px -> D=+0.541 m^-1 时 ~1420 px, 偏大
                    # 44 px 的目标框)。若拿未矫正的平相位去定位 ROI, 冻结的目标框
                    # 整个落在光斑之外, 之后每次量到的都是空气。
                    flat = np.zeros(optics.panel_shape, dtype=np.float64)
                    slm.set_phase_overlay(focus_panel)
                    full = capture(slm, cam, flat, next_slot())
                    cx_full, cy_full = zero_order_center(full, refine=False, despike_k=3)
                    print(f"零级 (全幅 argmax): ({cx_full}, {cy_full})")
                    cam.reset_window(center=(cx_full, cy_full),
                                     size=(int(args.roi_px), int(args.roi_px)))
                    roi_flat = capture(slm, cam, flat, next_slot())
                    cx, cy = zero_order_center(roi_flat, refine=False, despike_k=3)
                    # ⚠️ 冻结: 之后所有测量复用这一个 center 与框, 不逐帧重定位。
                    print(f"零级 (ROI 内 argmax, 已冻结): ({cx}, {cy})")

                    # -- 基线帧 (Δz=0, **GS 相位已施加**): 目标框按它定尺 ----------
                    # 框的边长必须由**本次相位自己的**光斑量出来。历史上写死的
                    # 44 px 是**模型** GS 相位的方形边长换算到硬件像元上的值, 而
                    # 硬件 GS 相位打出的光斑只有 12.9 px 宽 —— 框比光斑大 3.4 倍
                    # 时, 离焦弥散把框填满, 均匀度项 1/(1+CV) 上升恰好抵消功率项
                    # PIB 的下降, 合成分于是**看起来是平的**。按相位自己的光斑
                    # 定尺后响应恢复单调。
                    slm.set_phase_overlay(focus_panel)
                    baseline_frame = capture_retry(slm, cam, gs_panel, next_slot)

                    # -- 光路自检 (扫描前的 pre-flight) ---------------------------
                    # 判据看**结构**, 不看亮度: 光路被挡住时画面里只有读出噪声,
                    # 而噪声帧照样能算出 CV/PIB/光斑 —— 2026-10-21 实测本探针就
                    # 那样安安静静记了 30 行纯噪声。所以这里在**扫描开始之前**
                    # 先判一次: 受阻就中止 (不写 measurements.csv), 只是没聚焦
                    # 就警告并继续 (那种情况这一轮仍然有意义)。
                    assessment = preflight_assessment(baseline_frame)
                    if assessment.blocked:
                        abort_config = {
                            "timestamp_utc": stamp,
                            **group_fields(group_name, group_meta),
                            "out_root": str(out_root),
                            "plan": plan,
                            "exposure_ms_requested": float(args.exposure_ms),
                            "exposure_ceiling_ms": float(ceiling),
                            "exposure_ms_working": actual,
                            "frames_per_measurement": int(args.frames),
                            "zero_order_full_frame_xy": [int(cx_full), int(cy_full)],
                            "zero_order_roi_xy_frozen": [int(cx), int(cy)],
                            "roi_px": int(args.roi_px),
                            "path_blocked": True,
                            "spot_assessment": assessment.as_dict(),
                            "aborted": "path_blocked",
                        }
                        (out_dir / "run_config.json").write_text(
                            json.dumps(abort_config, indent=2, ensure_ascii=False),
                            encoding="utf-8",
                        )
                        logger.info(
                            "\n[中止] 光路受阻: " + assessment.reason + "\n"
                            "       基线帧里没有任何局域光斑 (峰值 "
                            f"{assessment.excess:.2f} counts, FWHM "
                            f"{assessment.fwhm_px:.0f} px / 画幅 "
                            f"{assessment.frame_px} px), 扫描只会记下噪声。\n"
                            f"       判定与理由已写进 {out_dir / 'run_config.json'}",

                        )
                        raise SystemExit(EXIT_PATH_BLOCKED)
                    if not assessment.usable:
                        logger.info(
                            "[警告] 基线帧有光但没有可评分的局域光斑 "
                            f"({assessment.reason}); 这一轮仍然继续, "
                            "但指标要按「光斑没聚焦」来解读",

                        )

                    target_side, target_side_source, baseline_fwhm = resolve_probe_target_side(
                        baseline_frame, int(args.roi_px), int(args.target_side_px)
                    )
                    target = square_mask(baseline_frame.shape, cy, cx, target_side)
                    print(f"目标框: 边长 {target_side} px (来源 {target_side_source}, "
                          f"基线光斑 FWHM {baseline_fwhm:.2f} px), 中心冻结在 ({cx}, {cy})")
                    # 这一组的**固定**参数: 定下来的框。写进 group_meta ⇒ 根目录的
                    # manifest.json 也带着它 (清单是这一组的事实记录)。
                    group_meta["target_side_px"] = str(int(target_side))
                    group_meta["target_side_source"] = str(target_side_source)

                    # -- 去倾斜自标定 (opt-in) ------------------------------------
                    # 放在 ROI / 冻结零级 / 目标框都定好之后: 标定要用**同一套**
                    # 坐标 (质心是相机坐标, 与目标框中心无关), 但绝不能碰框本身。
                    def capture_state(state: OverlayState) -> np.ndarray:
                        """按状态设置叠加层并抓一帧 (标定与验证共用同一条路径)。"""
                        slm.set_phase_overlay(state.overlay)
                        return capture_retry(slm, cam, gs_panel, next_slot)

                    chirp_tilt_cal: ChirpTiltCalibration | None = None
                    chirp_tilt_enabled = False
                    chirp_tilt_residual: tuple[float, float] | None = None
                    chirp_tilt_note = "未启用 (--chirp-tilt-cancel 未给)"

                    def compensation_for(
                        dz_mm: float,
                    ) -> tuple[np.ndarray | None, tuple[RampSpec, ...], tuple[float, float]]:
                        """某个 Δz 上要叠加的补偿斜坡。

                        未启用去倾斜时是空的 (行为与引入本选项之前逐位一致)。
                        """
                        if not chirp_tilt_enabled or chirp_tilt_cal is None:
                            return None, (), (0.0, 0.0)
                        return compensation_for_dz(
                            float(dz_mm), chirp_tilt_cal, optics.panel_shape
                        )

                    if bool(getattr(args, "chirp_tilt_cancel", False)):
                        cal_dz = float(args.chirp_tilt_cal_dz_mm)
                        test_period = int(args.chirp_tilt_test_period_px)
                        print(f"[去倾斜] 自标定开始: Δz={cal_dz} mm, "
                              f"测试斜坡周期 {test_period} panel px")
                        try:
                            baseline_xy = spot_centroid(baseline_frame)
                            if baseline_xy is None:
                                raise ChirpTiltCalibrationError("基线帧没有可用光斑")
                            cal_chirp = panel_pupil_chirp(
                                cal_dz * 1e-3, optics,
                                sign=args.chirp_sign, alpha=args.chirp_alpha,
                            )
                            chirp_tilt_cal, chirp_tilt_residual, chirp_tilt_note = (
                                run_chirp_tilt_calibration(
                                    capture_state,
                                    baseline_xy=baseline_xy,
                                    focus_panel=focus_panel,
                                    chirp_panel=cal_chirp,
                                    cal_dz_mm=cal_dz,
                                    test_period_px=test_period,
                                    panel_shape=optics.panel_shape,
                                )
                            )
                            if chirp_tilt_cal is None:
                                chirp_tilt_enabled = False
                                print(f"[去倾斜] {chirp_tilt_note} —— "
                                      "本组按**无去倾斜**继续")
                            else:
                                chirp_tilt_enabled = True
                                for response in chirp_tilt_cal.axis_map.responses:
                                    print(
                                        f"[去倾斜] 面板轴 {response.panel_axis} ("
                                        f"{'y/行' if response.panel_axis == 0 else 'x/列'}) "
                                        f"→ 相机轴 {response.camera_axis}: 实测 "
                                        f"{response.shift_px:+.2f} camera px @ "
                                        f"{response.period_px} px, 实测标度/bench 常量 = "
                                        f"{response.expected_ratio:.3f}"
                                    )
                                print("[去倾斜] 二次相位的转向量: "
                                      f"({chirp_tilt_cal.raw_shift_px_per_mm[0]:+.3f}, "
                                      f"{chirp_tilt_cal.raw_shift_px_per_mm[1]:+.3f}) "
                                      "camera px/mm")
                                comp_phase, comp_specs, comp_shift = compensation_for(cal_dz)
                                if chirp_tilt_residual is not None:
                                    print(
                                        f"[去倾斜] 补偿后残差: "
                                        f"({chirp_tilt_residual[0]:+.3f}, "
                                        f"{chirp_tilt_residual[1]:+.3f}) camera px/mm "
                                        f"(补偿斜坡 {comp_specs}, 预期位移 "
                                        f"({comp_shift[0]:+.2f}, {comp_shift[1]:+.2f}) px)"
                                    )
                                else:
                                    print(f"[去倾斜] {chirp_tilt_note}")
                        except Exception as exc:  # 标定失败不许整轮作废
                            chirp_tilt_cal = None
                            chirp_tilt_enabled = False
                            chirp_tilt_residual = None
                            chirp_tilt_note = f"标定失败: {exc}"
                            print(f"[去倾斜] {chirp_tilt_note} —— 本组按**无去倾斜**继续")
                    # 记录: 这一组跑没跑去倾斜, 清单里必须看得见。
                    group_meta["chirp_tilt_cancel"] = (
                        "true" if bool(getattr(args, "chirp_tilt_cancel", False)) else "false"
                    )
                    group_meta["chirp_tilt_enabled"] = (
                        "true" if chirp_tilt_enabled else "false"
                    )

                    run_config = {
                        "timestamp_utc": stamp,
                        **group_fields(group_name, group_meta),
                        "out_root": str(out_root),
                        "plan": plan,
                        "gs_phase_source": plan["gs_phase_source"],
                        "exposure_ms_requested": float(args.exposure_ms),
                        "exposure_ceiling_ms": float(ceiling),
                        "exposure_ms_working": actual,
                        "frames_per_measurement": int(args.frames),
                        "zero_order_full_frame_xy": [int(cx_full), int(cy_full)],
                        "zero_order_roi_xy_frozen": [int(cx), int(cy)],
                        "roi_px": int(args.roi_px),
                        "target_side_px": int(target_side),
                        "target_side_source": target_side_source,
                        "target_side_px_requested": int(args.target_side_px),
                        "baseline_spot_fwhm_px": float(baseline_fwhm),
                        "path_blocked": bool(assessment.blocked),
                        "spot_assessment": assessment.as_dict(),
                        "slm_number": int(args.slm_number),
                        "cam_id": int(args.cam_id),
                        "cam_type": args.cam_type,
                        "protected_memory_slot": protected,
                        "defocus_focal_mm": args.defocus_focal_mm,
                        "chirp_tilt_cancel": bool(getattr(args, "chirp_tilt_cancel", False)),
                        "chirp_tilt_cal_dz_mm": float(args.chirp_tilt_cal_dz_mm),
                        "chirp_tilt_test_period_px": int(args.chirp_tilt_test_period_px),
                        "chirp_tilt_enabled": bool(chirp_tilt_enabled),
                        "chirp_tilt_raw_shift_px_per_mm": (
                            list(chirp_tilt_cal.raw_shift_px_per_mm)
                            if chirp_tilt_cal is not None else None
                        ),
                        "chirp_tilt_residual_shift_px_per_mm": (
                            list(chirp_tilt_residual)
                            if chirp_tilt_residual is not None else None
                        ),
                        "chirp_tilt_axis_map": (
                            axis_map_to_dict(chirp_tilt_cal.axis_map)
                            if chirp_tilt_cal is not None else None
                        ),
                        "chirp_tilt_note": chirp_tilt_note,
                    }
                    if is_sim:
                        # 仿真产物必须自带「这是仿真跑出来的」标记 + 复现所需的
                        # 仿真参数。硬件分支**不加**这些键: 硬件那轮的产物已经
                        # 入库, 字段集必须逐字不变。
                        run_config["backend"] = "sim"
                        run_config["sim_seed"] = int(args.sim_seed)
                        run_config["sim_far_field_padding"] = int(
                            args.sim_far_field_padding
                        )
                    (out_dir / "run_config.json").write_text(
                        json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8"
                    )

                    def record(step: dict[str, Any], frame: np.ndarray) -> dict[str, Any]:
                        cv = float(uniformity_cv(frame, target))
                        pib = float(power_in_bucket(frame, target))
                        row = {
                            "index": len(rows),
                            "timestamp_utc": datetime.now(timezone.utc).isoformat(
                                timespec="seconds"
                            ),
                            **{k: step[k] for k in STEP_ROW_FIELDS},
                            "exposure_ms_readback": float(read_exposure_ms(cam)),
                            "frame_mean_in_target": float(np.mean(frame[target])),
                            "cv": cv,
                            "pib": pib,
                            "score": float(composite_from_pib_cv(pib, cv)),
                            **halo_metrics(frame, target),
                            **spot_geometry(frame),
                        }
                        np.savez_compressed(
                            out_dir / f"frame_{row['index']:03d}.npz",
                            frame=frame,
                            dz_mm=np.float64(step["dz_mm"]),
                            cv=np.float64(cv),
                            pib=np.float64(pib),
                            params=json.dumps(row, ensure_ascii=False),
                        )
                        rows.append(row)
                        write_measurements(out_dir, rows)
                        logger.info(json.dumps(row, ensure_ascii=False))
                        return row

                    def record_failure(step: dict[str, Any], exc: Exception) -> dict[str, Any]:
                        """记一行**失败** (取帧阶段的 USB 错误), 指标留 NaN。

                        单点失败不该让整轮扫描作废: 失败行带着 ``error`` 字段落盘,
                        离线出图时按 NaN 跳过即可, 其余 Δz 点照常有数据。
                        """
                        row = {
                            "index": len(rows),
                            "timestamp_utc": datetime.now(timezone.utc).isoformat(
                                timespec="seconds"
                            ),
                            **{k: step[k] for k in STEP_ROW_FIELDS},
                            "exposure_ms_readback": float(read_exposure_ms(cam)),
                            "frame_mean_in_target": float("nan"),
                            "cv": float("nan"),
                            "pib": float("nan"),
                            "score": float("nan"),
                            "frame_peak": float("nan"),
                            "halo_peak_outside_box": float("nan"),
                            "halo_power_fraction_outside_box": float("nan"),
                            "spot_fwhm_px": float("nan"),
                            "spot_centroid_x": float("nan"),
                            "spot_centroid_y": float("nan"),
                            "spot_hollowness": float("nan"),
                            "error": str(exc),
                        }
                        rows.append(row)
                        write_measurements(out_dir, rows)
                        logger.info(json.dumps(row, ensure_ascii=False))
                        return row

                    # -- 基线行 (Δz = 0, GS 相位, 只带焦面矫正) -------------------
                    # 帧在上面已经取好 (定框要用它), 这里只是用**已经定好的**框把它
                    # 记成第一行 —— 基线行与扫描行用同一个框, 口径一致。
                    # Δz=0 的补偿斜坡是空的 (期望位移 0), 所以基线行与「不做去
                    # 倾斜」时逐位一致。
                    baseline = {"step": -1, "dz_mm": 0.0, "dz_units": 0.0,
                                "chirp_sign": 0.0, "chirp_alpha": 0.0,
                                "chirp_edge_rad": 0.0, "chirp_edge_waves": 0.0,
                                "chirp_corner_rad": 0.0,
                                "defocus_focal_mm": args.defocus_focal_mm,
                                "chirp_tilt_enabled": bool(chirp_tilt_enabled),
                                "chirp_tilt_comp_shift_x_px": 0.0,
                                "chirp_tilt_comp_shift_y_px": 0.0,
                                "chirp_tilt_comp_specs": "[]"}
                    record(baseline, baseline_frame)

                    # -- 扫描 --------------------------------------------------------
                    try:
                        for step in plan["steps"]:
                            chirp = panel_pupil_chirp(
                                step["dz_mm"] * 1e-3, optics,
                                sign=args.chirp_sign, alpha=args.chirp_alpha,
                            )
                            # 去倾斜的补偿斜坡 (未启用时是空的)。
                            comp_phase, comp_specs, comp_shift = compensation_for(
                                step["dz_mm"]
                            )
                            # ⚠️ 驱动的叠加层只有**一个**通道: 焦面矫正、Δz 二次
                            # 相位与去倾斜斜坡必须相加进同一张图, 分多次
                            # set_phase_overlay 会让后者覆盖前者, 焦点矫正就丢了。
                            overlay = compose_overlay(focus_panel, chirp, comp_phase)
                            slm.set_phase_overlay(overlay)
                            step_row = {
                                **step,
                                "chirp_tilt_enabled": bool(chirp_tilt_enabled),
                                "chirp_tilt_comp_shift_x_px": float(comp_shift[0]),
                                "chirp_tilt_comp_shift_y_px": float(comp_shift[1]),
                                "chirp_tilt_comp_specs": json.dumps(
                                    [
                                        {"panel_axis": s.panel_axis,
                                         "period_px": s.period_px,
                                         "sign": s.sign}
                                        for s in comp_specs
                                    ],
                                    ensure_ascii=False,
                                ),
                            }
                            try:
                                frame = capture_retry(slm, cam, gs_panel, next_slot)
                            except SantecError as exc:
                                # 单点失败不能让整轮扫描作废: 记一行失败, 继续下一个 Δz。
                                logger.info(f"[跳过] Δz={step['dz_mm']} mm 取帧失败: {exc}",
)
                                record_failure(step_row, exc)
                                continue
                            record(step_row, frame)
                    finally:
                        # 叠加层是驱动的持久状态, 必须清掉 —— 否则之后的调用方会在
                        # 毫不知情的情况下一直带着这张离焦相位图。
                        slm.set_phase_overlay(None)

    write_measurements(out_dir, rows)
    plot_summary(out_dir, rows)
    # 根目录的分组清单: 每组写完立刻更新 —— 后面那一组崩掉时, 这一组已经被索引。
    update_manifest(
        out_root,
        group_record(
            group=group_name,
            variable=GROUP_VARIABLE,
            meta=group_meta,
            dir_name=group_name,
            n_rows=len(rows),
        ),
    )
    print(f"\n产物目录: {out_dir}")
    print(f"分组清单: {out_root / 'manifest.json'}")
    return out_dir


def write_measurements(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    """把测量行写成 CSV + JSON (离线重新出图的输入)。

    列名取**所有行的并集** (按首次出现顺序): 失败行多一个 ``error`` 列, 而
    DictWriter 只认第一行的列名, 照旧写法会在写失败行时抛 ``ValueError``。
    """
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with (out_dir / "measurements.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "measurements.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def plot_summary(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    """CV / score vs Δz 的实测曲线 (Agg, 中文 rcParams)。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = [
        "Noto Sans CJK SC", "Microsoft YaHei", "SimHei", "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    sel = sorted((r for r in rows if r["index"] >= 0), key=lambda r: r["dz_mm"])
    if not sel:
        return
    mm = [r["dz_mm"] for r in sel]
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 4.6))
    ax = axes[0]
    ax.plot(mm, [r["cv"] for r in sel], "o-", color="#FF5722", linewidth=2)
    ax.set_xlabel("等效 CCD 焦后位移 Δz / mm")
    ax.set_ylabel("目标框内 CV")
    ax.set_title("实测: 瞳面二次相位施加后的 CV")
    ax.grid(alpha=0.3)
    ax.plot(mm, [r["chirp_edge_waves"] for r in sel], "s--", color="#2196F3", linewidth=1.4,
            label="面板边缘相位 / 波数")
    ax.legend(fontsize=8)

    ax2 = axes[1]
    ax2.plot(mm, [r["score"] for r in sel], "o-", color="#4CAF50", linewidth=2)
    ax2.plot(mm, [r["pib"] for r in sel], "s--", color="#9E9E9E", linewidth=1.4, label="PIB")
    ax2.set_xlabel("等效 CCD 焦后位移 Δz / mm")
    ax2.set_ylabel("score / PIB")
    ax2.set_title("实测: score = 0.5·PIB + 0.5/(1+CV)")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)

    # 零级晕斑: 期望 ∝ 1/d²。整帧峰值与框外峰值都画, 因为 Δz 小时零级还在框内。
    ax3 = axes[2]
    ax3.semilogy(mm, [max(r["frame_peak"], 1e-12) for r in sel], "o-",
                 color="#9C27B0", linewidth=2, label="整帧峰值")
    ax3.semilogy(mm, [max(r["halo_peak_outside_box"], 1e-12) for r in sel], "s--",
                 color="#FF9800", linewidth=1.6, label="框外峰值 (Δz 大时才代表零级)")
    good = [r for r in sel if r["dz_mm"] >= 2.0 and r["frame_peak"] > 0]
    if len(good) >= 3:
        import numpy as _np

        xs = _np.log(_np.array([r["dz_mm"] for r in good]))
        ys = _np.log(_np.array([r["frame_peak"] for r in good]))
        expo = float(_np.polyfit(xs, ys, 1)[0])
        ax3.set_title(f"零级晕斑峰值 vs Δz (拟合指数 {expo:+.2f}; 整条曲线应 ∝ 1/d²)")
    else:
        ax3.set_title("零级晕斑峰值 vs Δz (窗口可信点不足 3 个, 不给指数)")
    ax3.set_xlabel("等效 CCD 焦后位移 Δz / mm")
    ax3.set_ylabel("峰值 (归一化)")
    ax3.grid(alpha=0.3, which="both")
    ax3.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / "hw_summary.png", dpi=150)
    plt.close(fig)


def find_measurement_dirs(src: Path) -> list[Path]:
    """找出 ``src`` 下所有含测量结果的目录。

    ``--replot`` 既接受**组目录** (直接含 ``measurements.json``), 也接受**根目录**
    (含若干组子目录)。根目录优先按 ``manifest.json`` 的索引找; 索引没有就退回
    「扫描子目录」—— 跑到一半崩掉时清单可能还没写, 但测量文件已经在盘上。

    Args:
        src: 用户给的目录。

    Returns:
        含 ``measurements.json`` 的目录列表 (可能为空)。
    """
    if (src / "measurements.json").is_file():
        return [src]
    found: list[Path] = []
    for entry in load_manifest(src):
        candidate = src / str(entry.get("dir", ""))
        if (candidate / "measurements.json").is_file():
            found.append(candidate)
    if not found and src.is_dir():
        found = [
            child
            for child in sorted(src.iterdir())
            if child.is_dir() and (child / "measurements.json").is_file()
        ]
    return found


def _replot_one(src: Path) -> None:
    """对**一个**含 ``measurements.json`` 的目录重新出图。"""
    rows = json.loads((src / "measurements.json").read_text(encoding="utf-8"))
    write_measurements(src, rows)
    plot_summary(src, rows)
    for row in rows:
        npz = src / f"frame_{row['index']:03d}.npz"
        if not npz.is_file():
            continue
        with np.load(npz, allow_pickle=False) as data:
            frame = data["frame"]
        print(f"Δz={row['dz_mm']:>5.1f} mm  CV={row['cv']:.4f}  PIB={row['pib']:.4f}  "
              f"frame{frame.shape} peak={float(frame.max()):.1f}")
    print(f"已重出图: {src / 'hw_summary.png'}")


def run_replot(args: argparse.Namespace) -> None:
    """离线重新出图: 只读测量目录里的 npz/CSV, **不碰任何设备**。

    给**组目录**就重出这一组; 给**根目录**就把下面每一组都重出图。
    """
    src = Path(args.replot)
    dirs = find_measurement_dirs(src)
    if not dirs:
        raise SystemExit(
            f"{src} 下找不到测量结果: 需要 measurements.json, 或它下面的组子目录"
        )
    for one in dirs:
        _replot_one(one)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dry-run", action="store_true",
                   help="只打印实验计划 (默认行为); 绝不 import / open 任何设备")
    p.add_argument("--execute", action="store_true",
                   help="真正开设备执行 —— 这是唯一会 import 驱动包的分支")
    p.add_argument("--replot", default=None,
                   help="离线重新出图: 指向一个已完成的测量目录 (不碰设备)")
    p.add_argument("--dz-mm", nargs="+", type=float, default=list(DZ_SCAN_MM),
                   help=f"Δz 扫描点 (mm), 默认 {DZ_SCAN_MM[0]:.0f}..{DZ_SCAN_MM[-1]:.0f} "
                        f"步长 5, 共 {len(DZ_SCAN_MM)} 点 (与仿真脚本一致)")
    p.add_argument("--gs-phase", default=None,
                   help="GS 相位来源 .npy/.npz (面板分辨率或方形模型网格); "
                        "不给则现场用已验证的 OffFocusBench 生成")
    p.add_argument("--quick", action="store_true", help="现场生成 GS 时用小网格")
    p.add_argument("--chirp-sign", type=float, default=CHIRP_SIGN,
                   help=f"瞳面二次相位符号 (默认 {CHIRP_SIGN:+.0f}, 由仿真数值反解)")
    p.add_argument("--chirp-alpha", type=float, default=CHIRP_ALPHA,
                   help=f"瞳面二次相位前因子 (默认 {CHIRP_ALPHA}, 理论值 1)")
    p.add_argument("--defocus-focal-mm", type=float, default=None,
                   help="瞳面菲涅尔焦面矫正焦距 (mm; 不给 = 不矫正, 探针默认 0; "
                        "2f 实验系统 CCD 不在焦面, 用 focus_probe 标定值, 例 1848.4)")
    p.add_argument("--chirp-tilt-cancel", action="store_true",
                   help="打开瞳面斜坡去倾斜: 现场自标定「面板斜坡 → 相机位移」的"
                        "映射与符号, 再用一条瞳面斜坡把二次相位的**转向**补掉 "
                        "(二次相位以面板标称照明中心为中心 ⇒ 等价偏心透镜, 实测"
                        "转向 -5.2 camera px/mm)。**默认关闭** —— 不给这个开关时"
                        "行为与引入本选项之前逐位一致")
    p.add_argument("--chirp-tilt-cal-dz-mm", type=float,
                   default=DEFAULT_CHIRP_TILT_CAL_DZ_MM,
                   help=f"去倾斜的**标定** Δz (mm, 默认 "
                        f"{DEFAULT_CHIRP_TILT_CAL_DZ_MM}); 应落在扫描区间内, "
                        "不在的话会额外抓一帧")
    p.add_argument("--chirp-tilt-test-period-px", type=int,
                   default=DEFAULT_CHIRP_TILT_TEST_PERIOD_PX,
                   help=f"标定用的**测试**斜坡周期 (panel px, 默认 "
                        f"{DEFAULT_CHIRP_TILT_TEST_PERIOD_PX}); 位移 ∝ 1/周期, "
                        "该周期对应约 7.4 camera px 的位移")
    p.add_argument("--slm-number", type=int, default=1)
    p.add_argument("--cam-id", type=int, default=0)
    p.add_argument("--cam-type", default="daheng",
                   choices=["daheng", "miicam", "sim"],
                   help="相机后端 (daheng/miicam/sim)。sim = 无硬件的数字孪生: "
                        "同一条命令换 --cam-type sim 就在仿真里跑出同结构的产物")
    p.add_argument("--sim-seed", type=int, default=0,
                   help="仿真噪声模型的随机种子 (--cam-type sim only)。固定它 ⇒ "
                        "同样的种子给出逐位相同的远场 (可重复)")
    p.add_argument("--sim-far-field-padding", type=int,
                   default=DEFAULT_SIM_FAR_FIELD_PADDING,
                   help="仿真远场 FFT 的补零因子 (--cam-type sim only); 与 "
                        "run_gs_refine_zernike_sweep.py 的 --sim-far-field-padding "
                        "同名同义 (补零越大远场像元越细, 代价按平方增长)")
    p.add_argument("--wavelength-nm", type=float, default=0.0,
                   help="0 = 静态读取 drivers/slm/factory.DEFAULT_WAVELENGTH_NM")
    p.add_argument("--exposure-ms", type=float, default=0.12)
    p.add_argument("--frames", type=int, default=4)
    p.add_argument("--roi-px", type=int, default=DEFAULT_ROI_PX)
    p.add_argument("--target-side-px", type=int, default=AUTO_TARGET_SIDE_PX,
                   help="目标框边长 (CCD px)。**0 = 自动 (默认)**: 用**本次相位自己的** "
                        "Δz=0 远场光斑定尺 (side = clamp(round(FWHM), 4, "
                        "floor(0.6·ROI))), 边长**只量一次**然后**冻结**, "
                        "基线行与所有扫描行共用同一个框、中心一直是冻结的零级; "
                        "基线帧测不出光斑时退回 44 px。给正数 = 显式框 "
                        "(保持原行为: 框固定、中心冻结)")
    p.add_argument("--settle-wait-s", type=float, default=0.5)
    p.add_argument("--settle-max-wait-s", type=float, default=6.0)
    p.add_argument("--settle-max-discard", type=int, default=40)
    p.add_argument("--seed", type=int, default=0, help="槽位随机选择的种子")
    p.add_argument("--group", default="main",
                   help="这一次调用探的是哪一组变量 (默认 main); 落盘目录是 "
                        "<根目录>/<组名>/, 根目录的 manifest.json 按组索引")
    p.add_argument("--group-meta", dest="group_meta_pairs", action="append",
                   default=[], metavar="KEY=VALUE",
                   help="这一组的固定参数 (可重复), 逐字记进组目录的 run_config.json "
                        "与根目录的 manifest.json, 例: --group-meta exposure=0.2")
    p.add_argument("--out-root", default=None,
                   help="根目录 (默认为 report/shape_trials/hw_gs_defocus_probe_<UTC>); "
                        "同一轮实验的多个组要用同一个根目录, 清单才会把它们都索引起来")
    return p


def main() -> None:
    args = build_parser().parse_args()
    try:
        args.group_meta = parse_group_meta(args.group_meta_pairs)
    except ValueError as exc:
        raise SystemExit(f"参数错误: {exc}") from exc

    if args.replot is not None:
        run_replot(args)
        return

    optics = build_optics(args)  # 这里顺带把 --wavelength-nm 0 解析成设备常量
    plan = build_plan(args, optics)

    if args.execute:
        if args.dry_run:
            logger.info("给了 --dry-run 和 --execute, 按 dry-run 处理")
            print_plan(plan, optics)
            assert_no_drivers()
            return
        run_execute(args, optics, plan)
        return

    print_plan(plan, optics)
    assert_no_drivers()


def assert_no_drivers() -> None:
    """dry-run 的硬保证: 收尾时断言 ``ao_shaping.drivers`` 没被 import。

    这条断言让「dry-run 不碰设备」成为**可执行的**保证而不是口头承诺。违反就 exit 3,
    因为那意味着 dry-run 路径上混进了会触发 SDK 加载的 import。
    """
    leaked = sorted(m for m in sys.modules if m.startswith("ao_shaping.drivers"))
    if leaked:
        sys.stderr.write(
            "DRY-RUN 违约: 计划打印过程中导入了驱动包:\n  "
            + "\n  ".join(leaked)
            + "\n"
        )
        sys.stderr.flush()
        raise SystemExit(3)
    print("dry-run 保证通过: sys.modules 里没有任何 ao_shaping.drivers.* "
          "(未 import、未 open 任何设备)")


if __name__ == "__main__":
    main()